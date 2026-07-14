"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

import math
from typing import cast

from cereal import messaging, custom
from openpilot.common.constants import CV
from openpilot.selfdrive.car.cruise import V_CRUISE_MAX
from openpilot.sunnypilot.selfdrive.controls.lib.dec.dec import DynamicExperimentalController
from openpilot.sunnypilot.selfdrive.controls.lib.e2e_alerts_helper import E2EAlertsHelper
from openpilot.sunnypilot.selfdrive.controls.lib.smart_cruise_control.smart_cruise_control import SmartCruiseControl
from openpilot.sunnypilot.selfdrive.controls.lib.speed_limit.speed_limit_assist import SpeedLimitAssist
from openpilot.sunnypilot.selfdrive.controls.lib.speed_limit.speed_limit_resolver import SpeedLimitResolver
from openpilot.sunnypilot.selfdrive.selfdrived.events import EventsSP
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.car.cruise import V_CRUISE_UNSET
from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState
from openpilot.sunnypilot.custom.longitudinal.finalizer import CustomLongitudinalFinalizer
from openpilot.sunnypilot.custom.longitudinal.cut_out_release import CutOutLeadRelease
from openpilot.sunnypilot.custom.longitudinal.follow_gap import FollowGapScheduler
from openpilot.sunnypilot.custom.longitudinal.moving_lead_cruise_cap import MovingLeadCruiseCap
from openpilot.sunnypilot.custom.longitudinal.modes import EvidenceClass, LongitudinalMode, admitted_evidence
from openpilot.sunnypilot.custom.longitudinal.wiring import CustomLongitudinalAdapter, CustomLongitudinalOutput, MODEL_STALE_AGE_S, _message_age_s

DecState = custom.LongitudinalPlanSP.DynamicExperimentalControl.DynamicExperimentalControlState
LongitudinalPlanSource = custom.LongitudinalPlanSP.LongitudinalPlanSource
EventNameSP = custom.OnroadEventSP.EventName


class _ProxyToFinalizer:
  """Descriptor that forwards a private planner attribute to the finalizer.

  Keeps ``LongitudinalPlannerSP`` backwards-compatible for drive_lab/tests/debug trace
  while the finalizer owns the actual state.
  """

  def __init__(self, attr: str, converter=None):
    self.attr = attr
    self.converter = converter

  def __get__(self, obj, objtype=None):
    if obj is None:
      return self
    return getattr(obj.custom_long_finalizer, self.attr)

  def __set__(self, obj, value):
    if self.converter is not None:
      value = self.converter(value)
    setattr(obj.custom_long_finalizer, self.attr, value)


class LongitudinalPlannerSP:
  def __init__(self, CP, CP_SP, mpc):
    self.CP = CP
    self.events_sp = EventsSP()
    self.dec = DynamicExperimentalController(CP, mpc)
    self.scc = SmartCruiseControl()
    self.resolver = SpeedLimitResolver()
    self.sla = SpeedLimitAssist(CP, CP_SP)
    self.source = LongitudinalPlanSource.cruise
    self.e2e_alerts_helper = E2EAlertsHelper()

    self.output_v_target = 0.
    self.output_a_target = 0.

    # Finalizer owns post-MPC stop-hold/release arbitration state.
    self.custom_long_finalizer = CustomLongitudinalFinalizer(CP)

    self._lead_stop_hold_active = False
    self._lead_stop_hold_gap_increasing_s = 0.0
    self._lead_stop_hold_missing_s = 0.0
    self._lead_stop_hold_lead_id = None
    self._lead_stop_hold_gap_prev_d_rel = None
    self._lead_stop_hold_gap_baseline_d_rel = None
    self._lead_stop_hold_arm_d_rel = None
    self._custom_long_output_telemetry = None
    self._last_release_block_reason = ""
    self._stop_hold_release_slew_a_target = None
    self._stop_hold_release_prep_a_target = None
    self._stop_hold_release_prep_raw_prev = None

    # Custom-2.0 longitudinal policy (default-on in this fork; fail-closed to stock output).
    self.custom_long = CustomLongitudinalAdapter(Params())
    self.custom_long_output = None
    # Dynamic follow-gap scheduler: mode-gated (DynamicFollowGapMode) bounded T_FOLLOW
    # compression on approach; apply is research-gated, fail-closed to the personality baseline.
    self.follow_gap = FollowGapScheduler(Params())
    # Moving-lead cruise cap: mode-gated (MovingLeadCruiseCapMode) bounded cruise-obstacle
    # lowering behind a mildly braking lead; apply is research-gated, fail-closed to raw cruise.
    self.moving_lead_cruise_cap = MovingLeadCruiseCap(Params())
    self.cut_out_release = CutOutLeadRelease()

  # Forwarding accessors: finalizer owns the state; planner exposes it for
  # backward-compatible instrumentation (drive_lab, unit tests, debug trace).
  _lead_stop_hold_active = _ProxyToFinalizer("lead_stop_hold_active", bool)
  _lead_stop_hold_gap_increasing_s = _ProxyToFinalizer("lead_stop_hold_gap_increasing_s", float)
  _lead_stop_hold_missing_s = _ProxyToFinalizer("lead_stop_hold_missing_s", float)
  _lead_stop_hold_lead_id = _ProxyToFinalizer("lead_stop_hold_lead_id")
  _lead_stop_hold_gap_prev_d_rel = _ProxyToFinalizer("lead_stop_hold_gap_prev_d_rel")
  _lead_stop_hold_gap_baseline_d_rel = _ProxyToFinalizer("lead_stop_hold_gap_baseline_d_rel")
  _lead_stop_hold_arm_d_rel = _ProxyToFinalizer("lead_stop_hold_arm_d_rel")
  _stop_hold_release_slew_a_target = _ProxyToFinalizer("stop_hold_release_slew_a_target")
  _stop_hold_release_prep_a_target = _ProxyToFinalizer("stop_hold_release_prep_a_target")
  _stop_hold_release_prep_raw_prev = _ProxyToFinalizer("stop_hold_release_prep_raw_prev")
  _last_release_block_reason = _ProxyToFinalizer("last_release_block_reason", str)
  _custom_long_output_telemetry = _ProxyToFinalizer("custom_long_output_telemetry")

  @staticmethod
  def _sm_item(sm, key):
    if hasattr(sm, 'get'):
      return sm.get(key)
    try:
      return sm[key]
    except Exception:
      return None

  def _reset_lead_stop_hold(self) -> None:
    self.custom_long_finalizer.reset_lead_stop_hold()

  def _sync_active_mode(self, sm) -> None:
    """Adopt the Engagement-Cycle Latched Longitudinal Mode published by selfdrived.

    plannerd never rereads the CustomLongitudinalMode Param while engaged; the active
    value comes from selfdriveStateSP. Until the first message arrives (or in harnesses
    without the service) the adapter keeps its boot-time default.
    """
    try:
      recv_frame = getattr(sm, 'recv_frame', None)
      if recv_frame is not None:
        try:
          if not recv_frame['selfdriveStateSP']:
            return
        except (KeyError, TypeError):
          pass  # harness sm without recv tracking: fall through to the message itself
      ss_sp = self._sm_item(sm, 'selfdriveStateSP')
      if ss_sp is None:
        return
      value = getattr(ss_sp, 'activeLongitudinalMode', None)
      if value is not None:
        self.custom_long.set_active_mode(value)
    except Exception:
      pass

  def _apply_stop_hold_release_slew(self, sm: messaging.SubMaster, a_target: float, release_mpc_stop: bool,
                                    mpc_stop: bool, raw_model_should_stop: bool, should_stop: bool) -> float:
    dt = float(getattr(self, 'dt', DT_MDL))
    return self.custom_long_finalizer._apply_stop_hold_release_slew(
      sm, dt, a_target, release_mpc_stop, mpc_stop, raw_model_should_stop, should_stop)

  def is_e2e(self, sm: messaging.SubMaster) -> bool:
    experimental_mode = sm['selfdriveState'].experimentalMode
    if self.custom_long.enabled:
      return self.custom_long.mode is LongitudinalMode.E2E

    # Custom off: preserve legacy ExperimentalMode + DEC behavior unchanged.
    if not self.dec.active():
      return experimental_mode

    return experimental_mode and self.dec.mode() == "blended"

  # Route 261: vision-only intersection slowdowns dragged the car toward MIN_V (20 km/h)
  # at up to -1.0 m/s^2 for turns the map did not corroborate (one the driver took faster,
  # one a model-path artifact after a lead exited). Rate-limit how fast an uncorroborated
  # vision vTarget may undercut ego speed so the approach is a gentle glide; the model's own
  # turn caution (custom stack / DEC) and map-corroborated curves keep full authority.
  _SCC_VISION_UNCORROBORATED_A_MIN = -0.5   # max implied decel without map corroboration
  _SCC_VISION_SOFTEN_TAU_S = 1.0            # lookahead horizon for the implied-decel bound

  def _soften_uncorroborated_vision_slowdown(self, v_ego: float) -> None:
    vision = self.scc.vision
    if not bool(getattr(vision, "is_active", False)) or bool(getattr(self.scc.map, "is_active", False)):
      return
    floor_v = float(v_ego) + self._SCC_VISION_UNCORROBORATED_A_MIN * self._SCC_VISION_SOFTEN_TAU_S
    if float(vision.output_v_target) < floor_v:
      vision.output_v_target = floor_v
    if float(vision.output_a_target) < self._SCC_VISION_UNCORROBORATED_A_MIN:
      vision.output_a_target = self._SCC_VISION_UNCORROBORATED_A_MIN

  def update_targets(self, sm: messaging.SubMaster, v_ego: float, a_ego: float, v_cruise: float,
                     refresh_custom_long: bool = True, t_follow: float = 1.5) -> tuple[float, float]:
    if refresh_custom_long:
      self.custom_long.maybe_refresh_params()
    self._sync_active_mode(sm)

    CS = sm['carState']
    v_cruise_cluster_kph = min(CS.vCruiseCluster, V_CRUISE_MAX)
    v_cruise_cluster = v_cruise_cluster_kph * CV.KPH_TO_MS

    long_enabled = sm['carControl'].enabled
    long_override = sm['carControl'].cruiseControl.override

    # Smart Cruise Control
    self.scc.update(sm, long_enabled, long_override, v_ego, a_ego, v_cruise)
    self._soften_uncorroborated_vision_slowdown(v_ego)

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

    # Debug/trace collection is diagnostics-only: apply-mode Actuation Verdicts are typed
    # and computed inside the stack regardless of this flag.
    collect_custom_long_debug = getattr(self.custom_long, "debug_trace_mode", "off") == "log"

    # Opt-in: shape the baseline a_target with the custom-2.0 policy (fail-closed; returns the
    # unchanged target when disabled or on any fault, so default behavior is never affected).
    measured_v_ego = float(getattr(CS, "vEgo", v_ego))
    measured_a_ego = float(getattr(CS, "aEgo", a_ego))
    self.custom_long_output = self.custom_long.evaluate(
      sm, measured_v_ego, measured_a_ego, v_cruise, self.output_a_target, self.scc, self.sla,
      t_follow=t_follow,
      collect_debug=collect_custom_long_debug,
    )
    self.output_a_target = self.custom_long_output.a_target
    # Fail-closed fault: request the existing immediateDisable path via the SP event stream
    # while the engagement is still active; the latch resets at the next engagement.
    if self.custom_long.fault_class and bool(getattr(sm['carControl'], 'longActive', False)):
      self.events_sp.add(EventNameSP.customLongitudinalFault)
    return self.output_v_target, self.output_a_target

  def custom_longitudinal_should_stop(self, mpc_should_stop: bool, raw_model_should_stop: bool,
                                      model_stale: bool = False) -> bool | None:
    return self.custom_long_finalizer.custom_longitudinal_should_stop(
      self.custom_long, self.custom_long_output, mpc_should_stop, raw_model_should_stop, model_stale)

  def custom_longitudinal_targets(self, targets: dict) -> dict:
    # SCC-Map is evidence-only for now. Keep its telemetry published separately, but never let it
    # participate in actuator target arbitration until a bounded apply tier exists.
    targets = {src: target for src, target in targets.items() if src != LongitudinalPlanSource.sccMap}
    if not self.custom_long.enabled:
      return targets
    admitted = admitted_evidence(self.custom_long.mode, self.custom_long.sources)
    source_to_evidence = {
      LongitudinalPlanSource.cruise: EvidenceClass.CRUISE,
      LongitudinalPlanSource.sccVision: EvidenceClass.CURVE_VISION,
      LongitudinalPlanSource.speedLimitAssist: EvidenceClass.SPEED_LIMIT,
    }
    filtered = {src: target for src, target in targets.items() if source_to_evidence[src] in admitted}
    filtered[LongitudinalPlanSource.cruise] = targets[LongitudinalPlanSource.cruise]
    return filtered

  def final_longitudinal_output(self, sm: messaging.SubMaster, mpc_a_target: float, mpc_should_stop: bool,
                                raw_model_a_target: float, raw_model_should_stop: bool) -> tuple[float, bool, bool]:
    model_stale = _message_age_s(sm, 'modelV2') > MODEL_STALE_AGE_S
    is_e2e = self.is_e2e(sm)
    dt = float(getattr(self, 'dt', DT_MDL))
    self.custom_long_finalizer.CP = self.CP
    result = self.custom_long_finalizer.finalize(
      sm, self.custom_long, self.custom_long_output, is_e2e, model_stale, dt,
      mpc_a_target, mpc_should_stop, raw_model_a_target, raw_model_should_stop,
      apply_stop_hold_release_slew=self._apply_stop_hold_release_slew,
      reset_lead_stop_hold=self._reset_lead_stop_hold,
    )
    self._custom_long_output_telemetry = result.custom_long_output_telemetry
    self._last_release_block_reason = result.last_release_block_reason
    return result.a_target, result.should_stop, result.e2e_source

  def update(self, sm: messaging.SubMaster) -> None:
    self.custom_long.maybe_refresh_params()  # enabled every tick; tuning on slow cadence inside the adapter
    self._sync_active_mode(sm)  # active mode is selfdrived's Engagement-Cycle Latch, not a Param reread
    self.events_sp.clear()
    custom_long_enabled = bool(self.custom_long.enabled)
    if not custom_long_enabled:   # custom SCC mode replaces DEC; keep DEC dormant when custom is on
      self.dec.update(sm)
    self.e2e_alerts_helper.update(sm, self.events_sp)
    controls_state = self._sm_item(sm, 'controlsState')
    car_state = self._sm_item(sm, 'carState')
    selfdrive_state = self._sm_item(sm, 'selfdriveState')
    long_control_off = bool(getattr(controls_state, 'longControlState', None) == LongCtrlState.off)
    v_cruise_initialized = True
    if car_state is not None:
      v_cruise_initialized = float(getattr(car_state, 'vCruise', V_CRUISE_UNSET)) != V_CRUISE_UNSET
    reset_state = long_control_off if getattr(self.CP, 'openpilotLongitudinalControl', False) else not bool(getattr(selfdrive_state, 'enabled', False))
    reset_state = reset_state or not v_cruise_initialized
    if reset_state:
      self._reset_lead_stop_hold()

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
    telemetry_custom_long_output = (
      cast(CustomLongitudinalOutput | None, self._custom_long_output_telemetry)
      if self._custom_long_output_telemetry is not None else self.custom_long_output
    )
    custom_long.active = bool(telemetry_custom_long_output.enabled) if telemetry_custom_long_output is not None else bool(self.custom_long.enabled)
    custom_long.shouldStop = bool(telemetry_custom_long_output.should_stop) if telemetry_custom_long_output is not None else False
    custom_long.mode = self._custom_longitudinal_mode_to_telemetry()
    custom_long.selectedIntent = str(getattr(telemetry_custom_long_output, "selected_intent", "" ) or "")
    custom_long.reason = str(getattr(telemetry_custom_long_output, "reason", "" ) or "")
    custom_long.faultClass = str(getattr(self.custom_long, "fault_class", "") or "")
    if getattr(self.custom_long, "debug_trace_mode", "off") == "log":
      self._populate_longitudinal_debug_trace(longitudinalPlanSP.longitudinalDebug, sm, telemetry_custom_long_output)
    self._custom_long_output_telemetry = None
    self._last_release_block_reason = ""

    pm.send('longitudinalPlanSP', plan_sp_send)

  def _custom_longitudinal_mode_to_telemetry(self):
    if self.custom_long.mode is LongitudinalMode.ACC:
      return custom.LongitudinalPlanSP.CustomLongitudinal.CustomLongitudinalMode.acc
    if self.custom_long.mode is LongitudinalMode.E2E:
      return custom.LongitudinalPlanSP.CustomLongitudinal.CustomLongitudinalMode.e2e
    return custom.LongitudinalPlanSP.CustomLongitudinal.CustomLongitudinalMode.scc

  def _populate_longitudinal_debug_trace(self, msg, sm: messaging.SubMaster, custom_long_output) -> None:
    try:
      trace = dict(getattr(self, '_last_longitudinal_debug', {}) or {})
      debug = dict(getattr(custom_long_output, 'debug', {}) or {})
      # Inject planner-level release block reason into the standstill_release_confidence trace
      release_block_reason = str(getattr(self, '_last_release_block_reason', '') or '')
      if release_block_reason:
        debug['standstill_release_confidence_block_reason'] = release_block_reason
      msg.enabled = True
      msg.traceMode = str(getattr(self.custom_long, "debug_trace_mode", "off"))
      car_state = self._sm_item(sm, 'carState')
      msg.vEgo = self._safe_float(getattr(car_state, 'vEgo', 0.0) if car_state is not None else 0.0)
      msg.vCruise = self._safe_float(trace.get('v_cruise', 0.0))
      msg.customATarget = self._safe_float(getattr(custom_long_output, 'a_target', 0.0))
      msg.customShouldStop = bool(getattr(custom_long_output, 'should_stop', False))
      msg.customIntent = str(getattr(custom_long_output, 'selected_intent', '') or '')
      msg.customReason = str(getattr(custom_long_output, 'reason', '') or '')
      msg.mpcATarget = self._safe_float(trace.get('mpc_a_target', 0.0))
      msg.mpcShouldStop = bool(trace.get('mpc_should_stop', False))
      msg.modelATarget = self._safe_float(trace.get('model_a_target', 0.0))
      msg.modelShouldStop = bool(trace.get('model_should_stop', False))
      msg.finalATargetUnclipped = self._safe_float(trace.get('final_a_target_unclipped', 0.0))
      msg.finalATargetClipped = self._safe_float(trace.get('final_a_target_clipped', 0.0))
      msg.finalShouldStop = bool(trace.get('final_should_stop', False))
      msg.accelClipMin = self._safe_float(trace.get('accel_clip_min', 0.0))
      msg.accelClipMax = self._safe_float(trace.get('accel_clip_max', 0.0))
      msg.e2eSource = bool(trace.get('e2e_source', False))
      self._populate_cut_in_brake_assist_trace(msg.cutInBrakeAssist, debug)
      self._populate_curve_speed_confidence_trace(msg.curveSpeedConfidence, debug)
      self._populate_standstill_release_confidence_trace(msg.standstillReleaseConfidence, debug)
      self._populate_acc_envelope_trace(msg.accEnvelope, debug)
      self._populate_dynamic_safety_floor_trace(msg.dynamicSafetyFloor, debug)
      self._populate_map_coast_trace(msg.mapCoast, debug)
    except Exception:
      msg.enabled = False
      msg.traceMode = 'off'

  def _populate_feature_trace_common(self, msg, debug: dict, prefix: str) -> None:
    msg.mode = str(debug.get(prefix + 'mode', 'off') or 'off')
    msg.effectiveMode = str(debug.get(prefix + 'effective_mode', msg.mode) or msg.mode)
    msg.applySupported = bool(debug.get(prefix + 'apply_supported', False))
    msg.eligible = bool(debug.get(prefix + 'eligible', False))
    msg.blockReason = str(debug.get(prefix + 'block_reason', '') or '')

  def _populate_cut_in_brake_assist_trace(self, msg, debug: dict) -> None:
    prefix = 'cut_in_brake_assist_'
    self._populate_feature_trace_common(msg, debug, prefix)
    msg.leadIdx = int(self._safe_float(debug.get(prefix + 'lead_idx', -1), -1))
    msg.pathYRel = self._safe_float(debug.get(prefix + 'path_y_rel', 0.0))
    msg.lateralVelocity = self._safe_float(debug.get(prefix + 'lateral_velocity', 0.0))
    msg.ttc = self._safe_float(debug.get(prefix + 'ttc', 0.0))
    msg.requiredDecel = self._safe_float(debug.get(prefix + 'required_decel', 0.0))
    msg.proposedCap = self._safe_float(debug.get(prefix + 'proposed_cap', 0.0))
    msg.confidence = self._safe_float(debug.get(prefix + 'confidence', 0.0))

  def _populate_curve_speed_confidence_trace(self, msg, debug: dict) -> None:
    prefix = 'curve_speed_confidence_'
    self._populate_feature_trace_common(msg, debug, prefix)
    msg.confidence = self._safe_float(debug.get(prefix + 'confidence', 0.0))
    msg.proposedCap = self._safe_float(debug.get(prefix + 'proposed_cap', 0.0))
    msg.source = str(debug.get(prefix + 'source', '') or '')
    msg.active = bool(debug.get(prefix + 'active', False))
    msg.currentLatAccel = self._safe_float(debug.get(prefix + 'current_lat_acc', 0.0))
    msg.maxPredLatAccel = self._safe_float(debug.get(prefix + 'max_pred_lat_acc', 0.0))
    msg.preEntryActive = bool(debug.get(prefix + 'pre_entry_active', False))

  def _populate_standstill_release_confidence_trace(self, msg, debug: dict) -> None:
    prefix = 'standstill_release_confidence_'
    self._populate_feature_trace_common(msg, debug, prefix)
    msg.confidence = self._safe_float(debug.get(prefix + 'confidence', 0.0))
    msg.releaseAllowed = bool(debug.get(prefix + 'release_allowed', False))
    msg.releaseSource = str(debug.get(prefix + 'release_source', '') or '')
    msg.releaseReason = str(debug.get(prefix + 'release_reason', '') or '')
    msg.releaseATarget = self._safe_float(debug.get(prefix + 'release_a_target', 0.0))

  def _populate_acc_envelope_trace(self, msg, debug: dict) -> None:
    prefix = 'acc_envelope_'
    msg.active = bool(debug.get(prefix + 'active', False))
    msg.wouldCap = bool(debug.get(prefix + 'would_cap', False))
    msg.capReason = str(debug.get(prefix + 'cap_reason', '') or '')
    msg.allowedATarget = self._safe_float(debug.get(prefix + 'allowed_a_target', 0.0))
    msg.deltaA = self._safe_float(debug.get(prefix + 'delta_a', 0.0))
    msg.desiredGap = self._safe_float(debug.get(prefix + 'desired_gap', 0.0))
    msg.timeGap = self._safe_float(debug.get(prefix + 'time_gap', 0.0))
    msg.ttc = self._safe_float(debug.get(prefix + 'ttc', 0.0))
    msg.usableStoppingGap = self._safe_float(debug.get(prefix + 'usable_stopping_gap', 0.0))
    msg.requiredStoppingDecel = self._safe_float(debug.get(prefix + 'required_stopping_decel', 0.0))
    msg.closingSpeedDecel = self._safe_float(debug.get(prefix + 'closing_speed_decel', 0.0))
    msg.jerkLimitedATarget = self._safe_float(debug.get(prefix + 'jerk_limited_a_target', 0.0))

  def _populate_dynamic_safety_floor_trace(self, msg, debug: dict) -> None:
    prefix = 'dynamic_safety_floor_'
    msg.active = bool(debug.get(prefix + 'active', False))
    msg.blockReason = str(debug.get(prefix + 'block_reason', '') or '')
    msg.currentSafeDistance = self._safe_float(debug.get(prefix + 'current_safe_distance', 0.0))
    msg.proposedSafeDistance = self._safe_float(debug.get(prefix + 'proposed_safe_distance', 0.0))
    msg.deltaSafeDistance = self._safe_float(debug.get(prefix + 'delta_safe_distance', 0.0))
    msg.dynamicFloorValue = self._safe_float(debug.get(prefix + 'dynamic_floor_value', 0.0))
    msg.kinematicFloorViolation = bool(debug.get(prefix + 'kinematic_floor_violation', False))
    msg.comfortBrakeEffective = self._safe_float(debug.get(prefix + 'comfort_brake_effective', 0.0))
    msg.latencyS = self._safe_float(debug.get(prefix + 'latency_s', 0.0))
    msg.latAccel = self._safe_float(debug.get(prefix + 'lat_accel', 0.0))
    msg.pitch = self._safe_float(debug.get(prefix + 'pitch', 0.0))

  def _populate_map_coast_trace(self, msg, debug: dict) -> None:
    prefix = 'map_coast_'
    msg.mode = str(debug.get(prefix + 'mode', 'off') or 'off')
    msg.vTarget = self._safe_float(debug.get(prefix + 'v_target', 0.0))
    msg.distance = self._safe_float(debug.get(prefix + 'distance', 0.0))
    msg.eligible = bool(debug.get(prefix + 'eligible', False))
    msg.cap = self._safe_float(debug.get(prefix + 'cap', 0.0))
    msg.applied = bool(debug.get(prefix + 'applied', False))
    msg.fault = bool(debug.get(prefix + 'fault', False))
    msg.coastDecel = self._safe_float(debug.get(prefix + 'accel_coast', 0.0))

  @staticmethod
  def _finite_float_or_none(value) -> float | None:
    try:
      v = float(value)
    except (TypeError, ValueError):
      return None
    return v if math.isfinite(v) else None

  @staticmethod
  def _safe_float(value, default: float = 0.0) -> float:
    try:
      v = float(value)
    except (TypeError, ValueError):
      return default
    return v if math.isfinite(v) else default
