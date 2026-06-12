"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

import time

import cereal.messaging as messaging
from cereal import log, custom

from opendbc.car import structs
from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog
from openpilot.sunnypilot import PARAMS_UPDATE_PERIOD
from openpilot.sunnypilot.livedelay.helpers import get_lat_delay
from openpilot.sunnypilot.modeld_v2.modeld_base import ModelStateBase
from openpilot.sunnypilot.selfdrive.controls.lib.blinker_pause_lateral import BlinkerPauseLateral
from openpilot.sunnypilot.selfdrive.controls.lib.latcontrol_torque_v0 import LatControlTorque as LatControlTorqueV0
from openpilot.sunnypilot.selfdrive.controls.lib.latcontrol_torque_v2 import LatControlTorque as LatControlTorqueV2, LatControlTorqueV21
from openpilot.sunnypilot.selfdrive.controls.lib.latcontrol_torque_v3 import LatControlTorqueV3
from openpilot.sunnypilot.selfdrive.controls.lib.latcontrol_torque_v4 import LatControlTorqueV4, LatControlTorqueV41, LatControlTorqueV5
from openpilot.sunnypilot.selfdrive.controls.lib.torque_versions import (
  TorqueControllerDefinition,
  TorqueControllerRegistry,
  normalize_torque_tune_version,
  resolve_torque_tune_version,
)


TORQUE_CONTROLLER_REGISTRY = TorqueControllerRegistry((
  TorqueControllerDefinition(0.0, LatControlTorqueV0),
  TorqueControllerDefinition(2.0, LatControlTorqueV2),
  TorqueControllerDefinition(2.1, LatControlTorqueV21),
  TorqueControllerDefinition(3.0, LatControlTorqueV3),
  TorqueControllerDefinition(4.0, LatControlTorqueV4),
  TorqueControllerDefinition(4.1, LatControlTorqueV41),
  # 5.0 is the first torque version with active profile-aware
  # command shaping (preview lead + turn-exit source-of-truth).
  # No hidden selector exists; the version is selected
  # directly via TorqueControlTune=5.0.
  TorqueControllerDefinition(5.0, LatControlTorqueV5),
))


class ControlsExt(ModelStateBase):
  def __init__(self, CP: structs.CarParams, params: Params):
    ModelStateBase.__init__(self)
    self.CP = CP
    self.params = params
    self._param_update_time: float = 0.0
    self.blinker_pause_lateral = BlinkerPauseLateral()
    self.smoothed_model_path_curvature = params.get_bool("SmoothedModelPathCurvature")

    cloudlog.info("controlsd_ext is waiting for CarParamsSP")
    self.CP_SP = messaging.log_from_bytes(params.get("CarParamsSP", block=True), custom.CarParamsSP)
    cloudlog.info("controlsd_ext got CarParamsSP")

    self.sm_services_ext = ['radarState', 'selfdriveStateSP']
    self.pm_services_ext = ['carControlSP']

  def initialize_lateral_control(self, lac, CI, dt):
    enforce_torque_control = self.params.get_bool("EnforceTorqueControl")
    torque_selection = self.params.get("TorqueControlTune", return_default=True)
    controls_profile_resolution = getattr(self, "controls_profile_resolution", None)
    active_profile_tune = getattr(controls_profile_resolution, "torque_control_tune", None)
    if active_profile_tune is not None:
      torque_selection = getattr(active_profile_tune, "value", active_profile_tune)
    torque_resolution = resolve_torque_tune_version(torque_selection)
    torque_version = torque_resolution.resolved_version
    native_torque = self.CP.lateralTuning.which() == 'torque'
    if torque_resolution.persist_value is not None:
      self.params.put("TorqueControlTune", torque_resolution.persist_value)

    # Selection contract:
    # - EnforceTorqueControl off: native torque uses v0 compatibility shim; non-native keeps stock controller.
    # - EnforceTorqueControl on: native torque uses selected 0.0/2.0/2.1/3.0/4.0/4.1/5.0 Experimental;
    #   non-native keeps stock controller.
    if not enforce_torque_control:
      if native_torque:
        return LatControlTorqueV0(self.CP, self.CP_SP, CI, dt)  # FIXME-SP: revert when upstream fixes tuning issues with v1
      return lac

    if self.CP.steerControlType == structs.CarParams.SteerControlType.angle:
      return lac

    if not native_torque:
      return lac

    controller_factory = TORQUE_CONTROLLER_REGISTRY.factory_for(torque_version)
    if controller_factory is not None:
      return controller_factory(self.CP, self.CP_SP, CI, dt)
    return lac

  @staticmethod
  def normalize_torque_tune_version(value) -> float | None:
    return normalize_torque_tune_version(value)

  def get_params_sp(self, sm: messaging.SubMaster) -> None:
    if time.monotonic() - self._param_update_time > PARAMS_UPDATE_PERIOD:
      self.blinker_pause_lateral.get_params()
      self.smoothed_model_path_curvature = self.params.get_bool("SmoothedModelPathCurvature")

      lac = getattr(self, "LaC", None)
      lat_control_state = getattr(lac, "CONTROL_STATE", self.CP.lateralTuning.which())
      if lat_control_state == 'torque':
        self.lat_delay = get_lat_delay(self.params, sm["liveDelay"].lateralDelay)
        if self.CP.lateralTuning.which() == 'torque':
          speed_aware_params = self.params.get("LiveTorqueSpeedAdaptiveParams")
          update_speed_aware_params = getattr(lac, "update_speed_aware_params", None)
          if update_speed_aware_params is None:
            extension = getattr(lac, "extension", None)
            update_speed_aware_params = getattr(extension, "update_speed_aware_params", None)
          if update_speed_aware_params is not None:
            update_speed_aware_params(speed_aware_params)

      self._param_update_time = time.monotonic()

  def get_lat_active(self, sm: messaging.SubMaster) -> bool:
    if self.blinker_pause_lateral.update(sm['carState']):
      return False

    ss_sp = sm['selfdriveStateSP']
    if ss_sp.mads.available:
      return bool(ss_sp.mads.active)

    # MADS not available, use stock state to engage
    return bool(sm['selfdriveState'].active)

  @staticmethod
  def get_lead_data(ld: log.RadarState.LeadData) -> dict:
    return {
      "dRel": ld.dRel,
      "yRel": ld.yRel,
      "vRel": ld.vRel,
      "aRel": ld.aRel,
      "vLead": ld.vLead,
      "dPath": ld.dPath,
      "vLat": ld.vLat,
      "vLeadK": ld.vLeadK,
      "aLeadK": ld.aLeadK,
      "fcw": ld.fcw,
      "status": ld.status,
      "aLeadTau": ld.aLeadTau,
      "modelProb": ld.modelProb,
      "radar": ld.radar,
      "radarTrackId": ld.radarTrackId,
    }

  def state_control_ext(self, sm: messaging.SubMaster) -> custom.CarControlSP:
    CC_SP = custom.CarControlSP.new_message()

    CC_SP.leadOne = self.get_lead_data(sm['radarState'].leadOne)
    CC_SP.leadTwo = self.get_lead_data(sm['radarState'].leadTwo)

    # MADS state
    CC_SP.mads = sm['selfdriveStateSP'].mads

    CC_SP.intelligentCruiseButtonManagement = sm['selfdriveStateSP'].intelligentCruiseButtonManagement

    return CC_SP

  @staticmethod
  def publish_ext(CC_SP: custom.CarControlSP, sm: messaging.SubMaster, pm: messaging.PubMaster) -> None:
    cc_sp_send = messaging.new_message('carControlSP')
    cc_sp_send.valid = sm['carState'].canValid
    cc_sp_send.carControlSP = CC_SP

    pm.send('carControlSP', cc_sp_send)

  def run_ext(self, sm: messaging.SubMaster, pm: messaging.PubMaster) -> None:
    CC_SP = self.state_control_ext(sm)
    self.publish_ext(CC_SP, sm, pm)
