"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import math
import time
from math import isclose

import openpilot.cereal.messaging as messaging
from openpilot.cereal import log, custom

from opendbc.car import structs
from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog
from openpilot.sunnypilot import PARAMS_UPDATE_PERIOD
from openpilot.sunnypilot.livedelay.helpers import get_lat_delay
from openpilot.sunnypilot.modeld_v2.modeld_base import ModelStateBase
from openpilot.sunnypilot.selfdrive.controls.lib.blinker_pause_lateral import BlinkerPauseLateral
from openpilot.sunnypilot.selfdrive.controls.lib.latcontrol_torque_v0 import LatControlTorque as LatControlTorqueV0
from openpilot.sunnypilot.custom.lateral.torque_v2_1 import LatControlTorqueV21
from openpilot.sunnypilot.custom.lateral.demand.wiring import LateralDemandAdapter
from openpilot.sunnypilot.custom.lateral.demand.telemetry import publish_model_path_state

TORQUE_TUNE_V0 = 0.0
TORQUE_TUNE_V1 = 1.0
TORQUE_TUNE_V21 = 2.1


def read_torque_control_tune(params: Params) -> float:
  tune = params.get("TorqueControlTune")
  if tune is None:
    return TORQUE_TUNE_V0
  try:
    return float(tune)
  except (TypeError, ValueError):
    cloudlog.warning(f"invalid TorqueControlTune={tune!r}; falling back to v0")
    return TORQUE_TUNE_V0


def select_torque_controller(CP: structs.CarParams, CP_SP, CI, dt, lac, tune: float, params: Params):
  if CP.lateralTuning.which() != 'torque':
    return lac

  enforce_torque = params.get_bool("EnforceTorqueControl")
  if not enforce_torque:
    # When EnforceTorqueControl is off, ignore the stored tune and keep the platform default controller.
    return lac

  if isclose(tune, TORQUE_TUNE_V0):
    return LatControlTorqueV0(CP, CP_SP, CI, dt)
  if isclose(tune, TORQUE_TUNE_V1):
    return lac
  if isclose(tune, TORQUE_TUNE_V21):
    return LatControlTorqueV21(CP, CP_SP, CI, dt)

  cloudlog.warning(f"unknown TorqueControlTune={tune!r}; falling back to v0")
  return LatControlTorqueV0(CP, CP_SP, CI, dt)


class ControlsExt(ModelStateBase):
  def __init__(self, CP: structs.CarParams, params: Params):
    ModelStateBase.__init__(self)
    self.CP = CP
    self.params = params
    self._param_update_time: float = 0.0
    self._live_lat_delay_enabled = self.params.get_bool("LagdToggle")
    self.blinker_pause_lateral = BlinkerPauseLateral()
    # Opt-in custom-2.0 lateral demand pipeline (default off -> stock model curvature).
    self.lateral_demand = LateralDemandAdapter(params)
    self._lateral_demand_enabled = self.lateral_demand.enabled

    cloudlog.info("controlsd_ext is waiting for CarParamsSP")
    self.CP_SP = messaging.log_from_bytes(params.get("CarParamsSP", block=True), custom.CarParamsSP)
    cloudlog.info("controlsd_ext got CarParamsSP")

    self.sm_services_ext = ['radarState', 'selfdriveStateSP']
    self.pm_services_ext = ['carControlSP']
    # ponytail: only hyundai lead_data_ext consumes CC_SP leads; skip the 100Hz 30-field capnp copy elsewhere
    self._cc_sp_wants_leads = CP.brand == 'hyundai'

  def initialize_lateral_control(self, lac, CI, dt):
    return select_torque_controller(self.CP, self.CP_SP, CI, dt, lac, read_torque_control_tune(self.params), self.params)

  def get_params_sp(self, sm: messaging.SubMaster) -> None:
    if time.monotonic() - self._param_update_time > PARAMS_UPDATE_PERIOD:
      self.blinker_pause_lateral.get_params()
      self.lateral_demand.refresh_params()

      self._live_lat_delay_enabled = self.params.get_bool("LagdToggle")
      self.lat_delay = get_lat_delay(self.params, sm["liveDelay"].lateralDelay)

      self._param_update_time = time.monotonic()

  def current_lateral_delay(self, sm: messaging.SubMaster) -> float:
    """Consume valid live-delay estimates at control rate while the live mode is enabled."""
    if self._live_lat_delay_enabled and sm.alive['liveDelay'] and sm.valid['liveDelay']:
      try:
        live_delay = float(sm['liveDelay'].lateralDelay)
      except (TypeError, ValueError):
        live_delay = math.nan
      if math.isfinite(live_delay) and live_delay >= 0.0:
        self.lat_delay = live_delay
    return float(self.lat_delay)

  def _update_lateral_demand_lifecycle(self) -> bool:
    """Custom lateral lifecycle: reset the pipeline when the opt-in toggles off. Returns enabled."""
    enabled = bool(self.lateral_demand.enabled)
    if self._lateral_demand_enabled and not enabled:
      self.lateral_demand.reset()
    self._lateral_demand_enabled = enabled
    return enabled

  def conditioned_lateral_demand(self, sm: messaging.SubMaster, lat_active: bool, CS, roll: float,
                                 new_desired_curvature: float, current_curvature: float,
                                 calibrated_pose, steer_limited_by_safety: bool,
                                 curvature_limited: bool, lat_delay: float,
                                 maneuver_active: bool) -> float:
    """Custom lateral pipeline: raw desired curvature in, Conditioned Lateral Demand out.

    Returns the input unchanged when the pipeline is disabled or a lateral maneuver owns the
    curvature; hard curvature/lat-accel caps stay in controlsd (clip_curvature) and turn this
    into the Processed Lateral Demand. The pipeline itself is fail-closed to the raw curvature.
    """
    enabled = self._update_lateral_demand_lifecycle()
    if not enabled:
      return new_desired_curvature
    if maneuver_active:
      self.lateral_demand.reset()
      return new_desired_curvature
    # Model-Path / pose evidence extraction with source health: stale or unhealthy sources
    # degrade to None rather than faulting the pipeline call.
    try:
      model_recv_time = float(sm.recv_time.get('modelV2', 0.0) or 0.0)
    except (TypeError, ValueError):
      model_recv_time = 0.0
    model_age_s = max(0.0, time.monotonic() - model_recv_time) if math.isfinite(model_recv_time) and model_recv_time > 0.0 else float("inf")
    live_pose = sm['livePose']
    live_pose_yaw_valid = bool(
      sm.alive['livePose'] and sm.valid['livePose']
      and getattr(live_pose, 'inputsOK', False)
      and getattr(live_pose, 'sensorsOK', False)
      and getattr(live_pose, 'posenetOK', False)
      and getattr(getattr(live_pose, 'angularVelocityDevice', None), 'valid', False)
    )
    yaw_rate = calibrated_pose.angular_velocity.z if calibrated_pose is not None and live_pose_yaw_valid else None
    return self.lateral_demand.process(
      lat_active, CS.vEgo, roll, new_desired_curvature, current_curvature, sm['modelV2'],
      getattr(CS, 'steeringPressed', None), model_age_s, yaw_rate,
      getattr(CS, 'steeringRateDeg', None), steer_limited_by_safety,
      bool(CS.leftBlinker), bool(CS.rightBlinker), curvature_limited, lat_delay=lat_delay,
    )

  def lateral_control_handoff(self, LaC, enabled: bool, lat_active: bool) -> None:
    """Torque v2.1-specific handoff: path evidence + override-refresh context."""
    if hasattr(LaC, 'set_under_response_path_evidence_from_lateral_demand'):
      LaC.set_under_response_path_evidence_from_lateral_demand(
        getattr(self.lateral_demand, 'last_result', None),
        active=lat_active, evidence_expected=self._lateral_demand_enabled,
      )
    if hasattr(LaC, 'set_torque_override_refresh_allowed'):
      LaC.set_torque_override_refresh_allowed(not (enabled or lat_active))

  def publish_lateral_telemetry(self, model_path_state, sm: messaging.SubMaster, CS,
                                raw_desired_curvature: float, processed_desired_curvature: float,
                                lat_delay: float) -> None:
    publish_model_path_state(model_path_state, sm, self.lateral_demand, CS.vEgo, CS.aEgo,
                             raw_desired_curvature, processed_desired_curvature, lat_delay)

  def get_lat_active(self, sm: messaging.SubMaster) -> bool:
    if self.blinker_pause_lateral.update(sm['carState']):
      return False

    ss_sp = sm['selfdriveStateSP']
    if ss_sp.mads.available:
      return bool(ss_sp.mads.active)

    # MADS not available, use stock state to engage
    return bool(sm['selfdriveState'].active)

  @staticmethod
  def get_lead_data(_lead, src: log.RadarState.LeadData) -> None:
    _lead.dRel = src.dRel
    _lead.yRel = src.yRel
    _lead.vRel = src.vRel
    _lead.aRel = src.deprecated.aRel
    _lead.vLead = src.vLead
    _lead.dPath = src.deprecated.dPath
    _lead.vLat = src.deprecated.vLat
    _lead.vLeadK = src.vLeadK
    _lead.aLeadK = src.aLeadK
    _lead.fcw = src.deprecated.fcw
    _lead.status = src.present
    _lead.aLeadTau = src.aLeadTau
    _lead.modelProb = src.modelProb
    _lead.radar = src.radar
    _lead.radarTrackId = src.radarTrackId

  def state_control_ext(self, sm: messaging.SubMaster) -> custom.CarControlSP:
    CC_SP = custom.CarControlSP.new_message()

    if self._cc_sp_wants_leads:
      self.get_lead_data(CC_SP.leadOne, sm['radarState'].leadOne)
      self.get_lead_data(CC_SP.leadTwo, sm['radarState'].leadTwo)

    # MADS state
    mads_src = sm['selfdriveStateSP'].mads
    CC_SP.mads.state = mads_src.state
    CC_SP.mads.enabled = mads_src.enabled
    CC_SP.mads.active = mads_src.active
    CC_SP.mads.available = mads_src.available

    # ICBM state
    icbm_src = sm['selfdriveStateSP'].intelligentCruiseButtonManagement
    CC_SP.intelligentCruiseButtonManagement.state = icbm_src.state
    CC_SP.intelligentCruiseButtonManagement.sendButton = icbm_src.sendButton
    CC_SP.intelligentCruiseButtonManagement.vTarget = icbm_src.vTarget

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
