"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

from cereal import messaging, custom
from opendbc.car import structs
from openpilot.common.constants import CV
from openpilot.selfdrive.car.cruise import V_CRUISE_MAX
from openpilot.sunnypilot.selfdrive.controls.lib.dec.dec import DynamicExperimentalController
from openpilot.sunnypilot.selfdrive.controls.lib.e2e_alerts_helper import E2EAlertsHelper
from openpilot.sunnypilot.selfdrive.controls.lib.smart_cruise_control.smart_cruise_control import SmartCruiseControl
from openpilot.sunnypilot.selfdrive.controls.lib.speed_limit.speed_limit_assist import SpeedLimitAssist
from openpilot.sunnypilot.selfdrive.controls.lib.speed_limit.speed_limit_resolver import SpeedLimitResolver
from openpilot.sunnypilot.selfdrive.selfdrived.events import EventsSP
from openpilot.common.params import Params
from openpilot.sunnypilot.custom.longitudinal.lead_anticipation import LeadAnticipation
from openpilot.sunnypilot.custom.longitudinal.modes import EvidenceClass, LongitudinalMode, admitted_evidence
from openpilot.sunnypilot.custom.longitudinal.wiring import CustomLongitudinalAdapter

DecState = custom.LongitudinalPlanSP.DynamicExperimentalControl.DynamicExperimentalControlState
LongitudinalPlanSource = custom.LongitudinalPlanSP.LongitudinalPlanSource


class LongitudinalPlannerSP:
  def __init__(self, CP, CP_SP, mpc):
    self.events_sp = EventsSP()
    self.dec = DynamicExperimentalController(CP, mpc)
    self.scc = SmartCruiseControl()
    self.resolver = SpeedLimitResolver()
    self.sla = SpeedLimitAssist(CP, CP_SP)
    self.source = LongitudinalPlanSource.cruise
    self.e2e_alerts_helper = E2EAlertsHelper()

    self.output_v_target = 0.
    self.output_a_target = 0.

    # Custom-2.0 longitudinal policy (default-on in this fork; fail-closed to stock output).
    self.custom_long = CustomLongitudinalAdapter(Params())
    self.custom_long_output = None
    # §3 lead-motion anticipation: mode-gated shadow/apply shaping of lead accel fed to the MPC
    # (off/shadow/apply via LeadAnticipationMode; compatibility LeadAnticipationEnabled only when
    # mode is absent, fail-closed to the raw radarState).
    self.lead_anticipation = LeadAnticipation(Params())

  def is_e2e(self, sm: messaging.SubMaster) -> bool:
    experimental_mode = sm['selfdriveState'].experimentalMode
    if self.custom_long.enabled:
      return self.custom_long.mode is LongitudinalMode.E2E

    # Custom off: preserve legacy ExperimentalMode + DEC behavior unchanged.
    if not self.dec.active():
      return experimental_mode

    return experimental_mode and self.dec.mode() == "blended"

  def update_targets(self, sm: messaging.SubMaster, v_ego: float, a_ego: float, v_cruise: float) -> tuple[float, float]:
    CS = sm['carState']
    v_cruise_cluster_kph = min(CS.vCruiseCluster, V_CRUISE_MAX)
    v_cruise_cluster = v_cruise_cluster_kph * CV.KPH_TO_MS

    long_enabled = sm['carControl'].enabled
    long_override = sm['carControl'].cruiseControl.override
    self.custom_long.maybe_refresh_params()

    # Smart Cruise Control
    self.scc.update(sm, long_enabled, long_override, v_ego, a_ego, v_cruise)

    # Speed Limit Resolver
    self.resolver.update(v_ego, sm)

    # Speed Limit Assist
    has_speed_limit = self.resolver.speed_limit_valid or self.resolver.speed_limit_last_valid
    self.sla.update(long_enabled, long_override, v_ego, a_ego, v_cruise_cluster, self.resolver.speed_limit,
                    self.resolver.speed_limit_final_last, has_speed_limit, self.resolver.distance, self.events_sp)

    targets = {
      LongitudinalPlanSource.cruise: (v_cruise, a_ego),
      LongitudinalPlanSource.sccVision: (self.scc.vision.output_v_target, self.scc.vision.output_a_target),
      LongitudinalPlanSource.sccMap: (self.scc.map.output_v_target, self.scc.map.output_a_target),
      LongitudinalPlanSource.speedLimitAssist: (self.sla.output_v_target, self.sla.output_a_target),
    }

    filtered_targets = self.custom_longitudinal_targets(targets)
    self.source = min(filtered_targets, key=lambda k: filtered_targets[k][0])
    self.output_v_target, self.output_a_target = filtered_targets[self.source]

    # Opt-in: shape the baseline a_target with the custom-2.0 policy (fail-closed; returns the
    # unchanged target when disabled or on any fault, so default behavior is never affected).
    self.custom_long_output = self.custom_long.evaluate(
      sm, v_ego, a_ego, v_cruise, self.output_a_target, self.scc, self.sla,
    )
    self.output_a_target = self.custom_long_output.a_target
    return self.output_v_target, self.output_a_target

  def custom_longitudinal_should_stop(self, mpc_should_stop: bool, raw_model_should_stop: bool) -> bool | None:
    if not self.custom_long.enabled or self.custom_long_output is None:
      return None
    if self.custom_long.mode is LongitudinalMode.ACC:
      return bool(mpc_should_stop)
    if self.custom_long.mode is LongitudinalMode.E2E:
      return bool(mpc_should_stop or raw_model_should_stop)
    return bool(mpc_should_stop or self.custom_long_output.should_stop)

  def custom_longitudinal_targets(self, targets: dict) -> dict:
    if not self.custom_long.enabled:
      return targets
    admitted = admitted_evidence(self.custom_long.mode, self.custom_long.sources)
    source_to_evidence = {
      LongitudinalPlanSource.cruise: EvidenceClass.CRUISE,
      LongitudinalPlanSource.sccVision: EvidenceClass.CURVE_VISION,
      LongitudinalPlanSource.sccMap: EvidenceClass.CURVE_MAP,
      LongitudinalPlanSource.speedLimitAssist: EvidenceClass.SPEED_LIMIT,
    }
    filtered = {src: target for src, target in targets.items() if source_to_evidence[src] in admitted}
    filtered[LongitudinalPlanSource.cruise] = targets[LongitudinalPlanSource.cruise]
    return filtered

  def _standstill_release_clears_mpc_stop(self, sm: messaging.SubMaster, mpc_a_target: float, mpc_should_stop: bool,
                                          raw_model_a_target: float, raw_model_should_stop: bool) -> tuple[bool, float]:
    if not self.custom_long.enabled or self.custom_long_output is None or not bool(getattr(self.custom_long_output, "standstill_release_allowed", False)):
      return False, float(mpc_a_target)
    if str(getattr(self.custom_long_output, "standstill_release_source", "")) not in ("lead_pullaway", "lead_standstill_launch", "no_lead_launch"):
      return False, float(mpc_a_target)
    if bool(getattr(self.custom_long_output, "should_stop", False)):
      return False, float(mpc_a_target)
    if raw_model_should_stop:
      return False, float(mpc_a_target)
    if self.custom_long.mode is LongitudinalMode.E2E and float(raw_model_a_target) < 0.15:
      return False, float(mpc_a_target)
    cs = sm["carState"]
    controls_state = sm["controlsState"]
    if bool(getattr(cs, "brakePressed", False)) or bool(getattr(cs, "gasPressed", False)):
      return False, float(mpc_a_target)
    if bool(getattr(controls_state, "forceDecel", False)):
      return False, float(mpc_a_target)
    if float(mpc_a_target) < -0.03:
      return False, float(mpc_a_target)
    if not mpc_should_stop:
      return False, float(mpc_a_target)
    release_a = max(float(mpc_a_target), 0.15, float(getattr(self.custom_long_output, "standstill_release_a_target", 0.0)))
    return True, release_a

  def final_longitudinal_output(self, sm: messaging.SubMaster, mpc_a_target: float, mpc_should_stop: bool,
                                raw_model_a_target: float, raw_model_should_stop: bool) -> tuple[float, bool, bool]:
    release_mpc_stop, release_a_target = self._standstill_release_clears_mpc_stop(
      sm, mpc_a_target, mpc_should_stop, raw_model_a_target, raw_model_should_stop)
    mpc_stop = bool(mpc_should_stop and not release_mpc_stop)
    custom_should_stop = self.custom_longitudinal_should_stop(mpc_stop, raw_model_should_stop)
    is_e2e = self.is_e2e(sm)
    should_stop = bool(custom_should_stop if custom_should_stop is not None else (mpc_stop or (raw_model_should_stop and is_e2e)))
    if is_e2e:
      a_target = min(raw_model_a_target, release_a_target if release_mpc_stop else mpc_a_target)
      return float(a_target), should_stop, bool(a_target < mpc_a_target)
    return float(release_a_target if release_mpc_stop else mpc_a_target), bool(should_stop), False

  def update(self, sm: messaging.SubMaster) -> None:
    self.events_sp.clear()
    if not self.custom_long.enabled:   # custom SCC mode replaces DEC; keep DEC dormant when custom is on
      self.dec.update(sm)
    self.e2e_alerts_helper.update(sm, self.events_sp)

  def publish_longitudinal_plan_sp(self, sm: messaging.SubMaster, pm: messaging.PubMaster) -> None:
    plan_sp_send = messaging.new_message('longitudinalPlanSP')

    plan_sp_send.valid = sm.all_checks(service_list=['carState', 'controlsState'])

    longitudinalPlanSP = plan_sp_send.longitudinalPlanSP
    longitudinalPlanSP.longitudinalPlanSource = self.source
    longitudinalPlanSP.vTarget = float(self.output_v_target)
    longitudinalPlanSP.aTarget = float(self.output_a_target)
    longitudinalPlanSP.events = self.events_sp.to_msg()

    # Dynamic Experimental Control compatibility alias. Custom longitudinal owns mode selection
    # when enabled, so DEC is reported inactive to avoid implying legacy DEC is driving actuation.
    dec = longitudinalPlanSP.dec
    if self.custom_long.enabled:
      dec.state = DecState.blended if self.custom_long.mode is LongitudinalMode.SCC else DecState.acc
      dec.enabled = False
      dec.active = False
    else:
      dec.state = DecState.blended if self.dec.mode() == 'blended' else DecState.acc
      dec.enabled = self.dec.enabled()
      dec.active = self.dec.active()

    # Smart Cruise Control
    smartCruiseControl = longitudinalPlanSP.smartCruiseControl
    # Vision Control
    sccVision = smartCruiseControl.vision
    sccVision.state = self.scc.vision.state
    sccVision.vTarget = float(self.scc.vision.output_v_target)
    sccVision.aTarget = float(self.scc.vision.output_a_target)
    sccVision.currentLateralAccel = float(self.scc.vision.current_lat_acc)
    sccVision.maxPredictedLateralAccel = float(self.scc.vision.max_pred_lat_acc)
    sccVision.enabled = self.scc.vision.is_enabled
    sccVision.active = self.scc.vision.is_active
    # Map Control
    sccMap = smartCruiseControl.map
    sccMap.state = self.scc.map.state
    sccMap.vTarget = float(self.scc.map.output_v_target)
    sccMap.aTarget = float(self.scc.map.output_a_target)
    sccMap.enabled = self.scc.map.is_enabled
    sccMap.active = self.scc.map.is_active

    # Speed Limit
    speedLimit = longitudinalPlanSP.speedLimit
    resolver = speedLimit.resolver
    resolver.speedLimit = float(self.resolver.speed_limit)
    resolver.speedLimitLast = float(self.resolver.speed_limit_last)
    resolver.speedLimitFinal = float(self.resolver.speed_limit_final)
    resolver.speedLimitFinalLast = float(self.resolver.speed_limit_final_last)
    resolver.speedLimitValid = self.resolver.speed_limit_valid
    resolver.speedLimitLastValid = self.resolver.speed_limit_last_valid
    resolver.speedLimitOffset = float(self.resolver.speed_limit_offset)
    resolver.distToSpeedLimit = float(self.resolver.distance)
    resolver.source = self.resolver.source
    assist = speedLimit.assist
    assist.state = self.sla.state
    assist.enabled = self.sla.is_enabled
    assist.active = self.sla.is_active
    assist.vTarget = float(self.sla.output_v_target)
    assist.aTarget = float(self.sla.output_a_target)

    # E2E Alerts
    e2eAlerts = longitudinalPlanSP.e2eAlerts
    e2eAlerts.greenLightAlert = self.e2e_alerts_helper.green_light_alert
    e2eAlerts.leadDepartAlert = self.e2e_alerts_helper.lead_depart_alert

    custom_long = longitudinalPlanSP.customLongitudinal
    custom_long.enabled = bool(self.custom_long.enabled)
    custom_long.active = bool(self.custom_long_output.enabled) if self.custom_long_output is not None else bool(self.custom_long.enabled)
    custom_long.shouldStop = bool(self.custom_long_output.should_stop) if self.custom_long_output is not None else False
    custom_long.mode = self._custom_longitudinal_mode_to_telemetry()
    custom_long.selectedIntent = str(getattr(self.custom_long_output, "selected_intent", "" ) or "")
    custom_long.reason = str(getattr(self.custom_long_output, "reason", "" ) or "")

    pm.send('longitudinalPlanSP', plan_sp_send)

  def _custom_longitudinal_mode_to_telemetry(self):
    if self.custom_long.mode is LongitudinalMode.ACC:
      return custom.LongitudinalPlanSP.CustomLongitudinal.CustomLongitudinalMode.acc
    if self.custom_long.mode is LongitudinalMode.E2E:
      return custom.LongitudinalPlanSP.CustomLongitudinal.CustomLongitudinalMode.e2e
    return custom.LongitudinalPlanSP.CustomLongitudinal.CustomLongitudinalMode.scc
