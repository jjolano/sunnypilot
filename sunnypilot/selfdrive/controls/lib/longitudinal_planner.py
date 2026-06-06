"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

from __future__ import annotations

import numpy as np

from cereal import messaging, custom
from opendbc.car import structs
from openpilot.common.constants import CV
from openpilot.common.params import Params
from openpilot.selfdrive.car.cruise import V_CRUISE_MAX
from openpilot.selfdrive.controls.lib.longitudinal_decision import (
  CandidateRole,
  DecisionSource,
  LongitudinalArbiter,
  LongitudinalCandidate,
  LongitudinalDecisionTelemetry,
  resolve_longitudinal_decision,
)
from openpilot.selfdrive.controls.lib.longitudinal_modes import (
  DecCompatibilityState,
  LongitudinalActuationType,
  LongitudinalMode,
  LongitudinalModeResolution,
  LongitudinalModeResolver,
  ResolvedLongitudinalImplementation,
)
from openpilot.selfdrive.controls.lib.longitudinal_stacks.adapters import apply_stack_output_to_planner, planner_state_to_stack_output
from openpilot.selfdrive.controls.lib.longitudinal_stacks.custom_v2 import (
  CustomV2Scene,
  CustomV2SceneValidationError,
  build_custom_v2_advisory_candidates,
  build_custom_v2_progress_candidates,
  build_force_slow_candidate,
  build_one_pedal_driver_candidate,
)
from openpilot.selfdrive.controls.lib.longitudinal_stacks.interface import LongitudinalStackOutput, validate_stack_output
from openpilot.selfdrive.controls.lib.longitudinal_stacks.planner_seed import PLANNER_SEED_FLOOR, PlannerSeedCandidate, select_planner_seed_candidate
from openpilot.selfdrive.controls.lib.longitudinal_stacks.policy import (
  SignalProviderCandidate,
  build_sp_candidates_from_signal_providers,
  build_sp_longitudinal_candidates,
  ensure_driver_intent,
  fallback_physical_candidates,
  planner_seed_candidates_to_longitudinal_candidates,
  replace_driver_intent,
)
from openpilot.selfdrive.controls.lib.longitudinal_stacks.registry import make_custom_longitudinal_stack
from openpilot.selfdrive.controls.lib.longitudinal_stacks.selector import (
  CUSTOM_V2,
  SUNNYPILOT_CURRENT,
  StackResolution,
  is_custom_stack,
  resolve_longitudinal_stack,
  stack_id_for_name,
)
from openpilot.sunnypilot.selfdrive.controls.lib.e2e_alerts_helper import E2EAlertsHelper
from openpilot.sunnypilot.selfdrive.controls.lib.smart_cruise_control.smart_cruise_control import SmartCruiseControl
from openpilot.sunnypilot.selfdrive.controls.lib.osm_traffic_control_prior import OsmTrafficControlPrior
from openpilot.sunnypilot.selfdrive.controls.lib.speed_limit.speed_limit_assist import (
  SpeedLimitAssist,
  V_CRUISE_UNSET,
)
from openpilot.sunnypilot.selfdrive.controls.lib.speed_limit.speed_limit_resolver import (
  SpeedLimitResolver,
)
from openpilot.sunnypilot.selfdrive.selfdrived.events import EventsSP
from openpilot.sunnypilot.models.helpers import get_active_bundle

DecState = custom.LongitudinalPlanSP.DynamicExperimentalControl.DynamicExperimentalControlState
LongitudinalModeStatus = custom.LongitudinalPlanSP.LongitudinalModeStatus
LongitudinalModeTelemetryMode = LongitudinalModeStatus.Mode
LongitudinalModeTelemetryImplementation = LongitudinalModeStatus.Implementation
LongitudinalModeTelemetryActuationType = LongitudinalModeStatus.ActuationType
LongitudinalModeTelemetryCompatibilityAliasState = LongitudinalModeStatus.CompatibilityAliasState
LongitudinalPlanSource = custom.LongitudinalPlanSP.LongitudinalPlanSource
StackId = custom.LongitudinalPlanSP.Stack.StackId
DEFAULT_LONGITUDINAL_MODE_RESOLUTION = LongitudinalModeResolution(
  requested_mode=LongitudinalMode.ACC,
  resolved_implementation=ResolvedLongitudinalImplementation.HARDWARE_ACC,
  actuation_type=LongitudinalActuationType.DIRECT,
)
LONGITUDINAL_MODE_TELEMETRY_MODES = {
  LongitudinalMode.ACC: LongitudinalModeTelemetryMode.acc,
  LongitudinalMode.E2E: LongitudinalModeTelemetryMode.e2e,
  LongitudinalMode.SCC: LongitudinalModeTelemetryMode.scc,
}
LONGITUDINAL_MODE_TELEMETRY_IMPLEMENTATIONS = {
  ResolvedLongitudinalImplementation.HARDWARE_ACC: LongitudinalModeTelemetryImplementation.hardwareAcc,
  ResolvedLongitudinalImplementation.MODEL_ACC: LongitudinalModeTelemetryImplementation.modelAcc,
  ResolvedLongitudinalImplementation.E2E: LongitudinalModeTelemetryImplementation.e2e,
  ResolvedLongitudinalImplementation.SCC_ACC: LongitudinalModeTelemetryImplementation.sccAcc,
  ResolvedLongitudinalImplementation.SCC_E2E: LongitudinalModeTelemetryImplementation.sccE2e,
  ResolvedLongitudinalImplementation.ICBM_ADVISORY: LongitudinalModeTelemetryImplementation.icbmAdvisory,
}
LONGITUDINAL_MODE_TELEMETRY_ACTUATION_TYPES = {
  LongitudinalActuationType.DIRECT: LongitudinalModeTelemetryActuationType.direct,
  LongitudinalActuationType.SET_SPEED_ADVISORY: LongitudinalModeTelemetryActuationType.setSpeedAdvisory,
}
LONGITUDINAL_MODE_TELEMETRY_COMPATIBILITY_ALIAS_STATES = {
  DecCompatibilityState.ACC: LongitudinalModeTelemetryCompatibilityAliasState.acc,
  DecCompatibilityState.BLENDED: LongitudinalModeTelemetryCompatibilityAliasState.blended,
}
SPEED_LIMIT_HANDOFF_EXIT_MARGIN = 0.25  # m/s, near enough to manual cruise to return to cruise.
SPEED_LIMIT_HANDOFF_A_TARGET_MAX = 0.0  # m/s^2, coast instead of accelerating during handoff.
SPEED_LIMIT_SPEED_UP_ACCEL_CAP = 0.8  # m/s^2, driver-intent candidate speed-up governor.
SPEED_LIMIT_SPEED_UP_LOOKAHEAD = 2.0  # s, short candidate horizon for active speed-limit increases.
LEAD_SPEEDUP_GUARD_TIME_GAP = 2.2  # s, match the observed uncomfortable closing window.
LEAD_SPEEDUP_GUARD_MIN_DISTANCE = 25.0  # m, low-speed floor for close-lead gating.
LEAD_SPEEDUP_GUARD_CLOSING_V_REL = -0.2  # m/s, ignore noise around matched speed.
LEAD_SPEEDUP_GUARD_A_TARGET_MAX = 0.0  # m/s^2, coast instead of accelerating into the lead.
LEAD_SPEEDUP_GUARD_LATERAL_EXIT_Y_REL = 1.6
SOURCE_SELECTION_HYSTERESIS_V = 0.25
def _select_lower_target(selected_source, selected_v_target, selected_a_target, candidate_source, candidate):
  candidate_v_target, candidate_a_target = candidate
  if candidate_v_target < selected_v_target:
    return candidate_source, candidate_v_target, candidate_a_target
  return selected_source, selected_v_target, selected_a_target


def _decision_source_name(source) -> str:
  return source.value if isinstance(source, DecisionSource) else str(source or "")


def publish_decision_layer_telemetry(longitudinalPlanSP, telemetry: LongitudinalDecisionTelemetry | None) -> None:
  decisionLayer = longitudinalPlanSP.decisionLayer
  decisionLayer.enabled = telemetry is not None
  if telemetry is None:
    decisionLayer.rawSource = ""
    decisionLayer.rawReason = ""
    decisionLayer.appliedReason = ""
    decisionLayer.rawATarget = 0.0
    decisionLayer.appliedATarget = 0.0
    decisionLayer.legacyATarget = 0.0
    decisionLayer.rawVTarget = 0.0
    decisionLayer.accelDelta = 0.0
    decisionLayer.rawShouldStop = False
    decisionLayer.appliedShouldStop = False
    decisionLayer.legacyShouldStop = False
    return

  decisionLayer.rawSource = _decision_source_name(telemetry.raw_source)
  decisionLayer.rawReason = str(telemetry.raw_active_reason)
  decisionLayer.appliedReason = str(telemetry.applied_reason)
  decisionLayer.rawATarget = float(telemetry.raw_a_target)
  decisionLayer.appliedATarget = float(telemetry.applied_a_target)
  decisionLayer.legacyATarget = float(telemetry.legacy_a_target)
  decisionLayer.rawVTarget = float(telemetry.raw_v_target)
  decisionLayer.accelDelta = float(telemetry.accel_delta)
  decisionLayer.rawShouldStop = bool(telemetry.raw_should_stop)
  decisionLayer.appliedShouldStop = bool(telemetry.applied_should_stop)
  decisionLayer.legacyShouldStop = bool(telemetry.legacy_should_stop)


def publish_stack_telemetry(longitudinalPlanSP, resolution: StackResolution, actuated_stack: str,
                            actuated_a_target: float, fault_latched: bool = False, fault_reason: str = "",
                            selected_intent: str = "", selected_reason: str = "",
                            rejected: tuple[tuple[str, str], ...] = (), seed_context: str = "",
                            seed_candidate: str = "") -> None:
  stack = longitudinalPlanSP.stack
  stack.requestedStack = stack_id_for_name(resolution.requested_stack)
  stack.resolvedStack = stack_id_for_name(resolution.resolved_stack)
  stack.actuatedStack = stack_id_for_name(actuated_stack)
  stack.customVersion = str(resolution.custom_version)
  stack.faultLatched = bool(fault_latched)
  stack.faultReason = str(fault_reason)
  stack.actuatedATarget = float(actuated_a_target)
  stack.selectedIntent = str(selected_intent)
  stack.selectedReason = str(selected_reason)
  stack.rejectedIntents = [str(intent) for intent, _reason in rejected]
  stack.rejectedReasons = [str(reason) for _intent, reason in rejected]
  stack.seedContext = str(seed_context)
  stack.seedCandidate = str(seed_candidate)


def publish_longitudinal_mode_telemetry(longitudinalPlanSP, resolution: LongitudinalModeResolution | None) -> None:
  resolution = resolution or DEFAULT_LONGITUDINAL_MODE_RESOLUTION
  longitudinal_mode = longitudinalPlanSP.longitudinalMode
  longitudinal_mode.requestedMode = LONGITUDINAL_MODE_TELEMETRY_MODES.get(
    resolution.requested_mode, LongitudinalModeTelemetryMode.acc
  )
  longitudinal_mode.resolvedImplementation = LONGITUDINAL_MODE_TELEMETRY_IMPLEMENTATIONS.get(
    resolution.resolved_implementation, LongitudinalModeTelemetryImplementation.hardwareAcc
  )
  longitudinal_mode.actuationType = LONGITUDINAL_MODE_TELEMETRY_ACTUATION_TYPES.get(
    resolution.actuation_type, LongitudinalModeTelemetryActuationType.direct
  )
  longitudinal_mode.restrictionStatus = [str(status) for status in resolution.restriction_status]
  longitudinal_mode.unsupportedReason = str(resolution.unsupported_reason)
  longitudinal_mode.compatibilityAliasState = LONGITUDINAL_MODE_TELEMETRY_COMPATIBILITY_ALIAS_STATES.get(
    resolution.compatibility_alias_state, LongitudinalModeTelemetryCompatibilityAliasState.acc
  )


def legacy_dec_enabled_for_mode(resolution: LongitudinalModeResolution) -> bool:
  return bool(resolution.compatibility_alias_state == DecCompatibilityState.BLENDED or resolution.e2e_like)


def should_block_lead_speedup(v_ego: float, lead_status: bool, d_rel: float, v_rel: float, y_rel: float,
                              gas_pressed: bool, brake_pressed: bool) -> bool:
  if not lead_status or gas_pressed or brake_pressed:
    return False
  if abs(y_rel) >= LEAD_SPEEDUP_GUARD_LATERAL_EXIT_Y_REL:
    return False
  if v_rel > LEAD_SPEEDUP_GUARD_CLOSING_V_REL:
    return False

  close_distance = max(LEAD_SPEEDUP_GUARD_MIN_DISTANCE, v_ego * LEAD_SPEEDUP_GUARD_TIME_GAP)
  return d_rel < close_distance


def should_block_lead_speedup_from_context(context, v_ego: float, gas_pressed: bool, brake_pressed: bool) -> bool:
  for state in getattr(context, "states", ()):
    if str(getattr(state, "authority", "none")) == "none":
      continue
    if should_block_lead_speedup(
      v_ego,
      True,
      float(getattr(state, "d_rel", 0.0)),
      float(getattr(state, "v_rel", 0.0)),
      float(getattr(state, "path_y_rel", getattr(state, "y_rel", 0.0))),
      gas_pressed,
      brake_pressed,
    ):
      return True
  return False


def apply_lead_speedup_guard(active: bool, v_ego: float, target: tuple[float, float]) -> tuple[float, float]:
  if not active:
    return target

  v_target, a_target = target
  return min(v_target, v_ego), min(a_target, LEAD_SPEEDUP_GUARD_A_TARGET_MAX)


def select_lowest_longitudinal_target(speed_limit_active, cruise, scc_vision, scc_map, speed_limit_assist, osm_traffic_control,
                                      source_prev=None, v_target_prev=None):
  if speed_limit_active:
    selected_source = LongitudinalPlanSource.speedLimitAssist
    selected_v_target = speed_limit_assist[0]
    selected_a_target = cruise[1]
  else:
    selected_source = LongitudinalPlanSource.cruise
    selected_v_target, selected_a_target = cruise

  selected_source, selected_v_target, selected_a_target = _select_lower_target(
    selected_source, selected_v_target, selected_a_target, LongitudinalPlanSource.sccVision, scc_vision
  )

  selected_source, selected_v_target, selected_a_target = _select_lower_target(
    selected_source, selected_v_target, selected_a_target, LongitudinalPlanSource.sccMap, scc_map
  )
  if not speed_limit_active:
    selected_source, selected_v_target, selected_a_target = _select_lower_target(
      selected_source, selected_v_target, selected_a_target, LongitudinalPlanSource.speedLimitAssist, speed_limit_assist
    )
  selected_source, selected_v_target, selected_a_target = _select_lower_target(
    selected_source, selected_v_target, selected_a_target, LongitudinalPlanSource.osmTrafficControl, osm_traffic_control
  )

  if source_prev is not None and v_target_prev is not None:
    # Always allow switching to a more restrictive (slower) target for safety.
    if selected_v_target < v_target_prev:
      return selected_source, selected_v_target, selected_a_target

    # Only switch to a less restrictive (faster) target if it's significantly faster
    # OR if it's the same target but a different source (allows state machine transitions).
    if selected_v_target > v_target_prev + SOURCE_SELECTION_HYSTERESIS_V or \
       (abs(selected_v_target - v_target_prev) < 1e-4 and selected_source != source_prev):
      return selected_source, selected_v_target, selected_a_target
    else:
      return source_prev, v_target_prev, selected_a_target

  return selected_source, selected_v_target, selected_a_target


class LongitudinalPlannerSP:
  def __init__(self, CP: object, CP_SP: object, mpc):
    self.CP = CP
    self.CP_SP = CP_SP
    if not hasattr(self, "params"):
      self.params = Params()
    self.events_sp = EventsSP()
    self.resolver = SpeedLimitResolver()
    self.scc = SmartCruiseControl()
    self.osm_traffic_control_prior = OsmTrafficControlPrior()
    self.resolver = SpeedLimitResolver()
    self.sla = SpeedLimitAssist(CP, CP_SP)
    self.active_scc = self.scc
    self.active_resolver = self.resolver
    self.active_sla = self.sla
    self.generation = int(model_bundle.generation) if (model_bundle := get_active_bundle()) else None
    self.source = LongitudinalPlanSource.cruise
    self.e2e_alerts_helper = E2EAlertsHelper()

    self.output_v_target = 0.
    self.output_a_target = 0.
    self.decision_candidates_sp = []
    self._speed_limit_handoff_active = False
    self._speed_limit_active_prev = False
    self.speed_limit_handoff_debug: dict[str, object] = {}
    self.longitudinal_stack_resolution = resolve_longitudinal_stack(
      self.params.get("LongitudinalStack", return_default=True), self.CP, self.CP_SP
    )
    self.longitudinal_mode_resolution = LongitudinalModeResolver.resolve(self.params, self.CP)
    self.custom_longitudinal_stack = self._make_custom_longitudinal_stack(self.longitudinal_stack_resolution.resolved_stack)
    self.longitudinal_stack_actuated_stack = SUNNYPILOT_CURRENT
    self.longitudinal_stack_fault_latched = False
    self.longitudinal_stack_fault_reason = ""
    self.longitudinal_stack_selected_intent = ""
    self.longitudinal_stack_selected_reason = ""
    self.longitudinal_stack_rejected: tuple[tuple[str, str], ...] = ()
    self.longitudinal_stack_seed_context = ""
    self.longitudinal_stack_seed_candidate = ""
    self.custom_v2_fault_latched = False
    self.custom_v2_fault_reason = ""

  @staticmethod
  def _make_custom_longitudinal_stack(stack_name: str):
    if not is_custom_stack(stack_name):
      return None
    return make_custom_longitudinal_stack(stack_name)

  def is_e2e(self, sm: messaging.SubMaster) -> bool:
    return bool(getattr(self, "longitudinal_mode_resolution", DEFAULT_LONGITUDINAL_MODE_RESOLUTION).e2e_like)

  def _update_speed_limit_handoff(self, long_enabled: bool, long_override: bool, v_ego: float, v_cruise: float) -> bool:
    has_limit_target = (self.resolver.speed_limit_valid or self.resolver.speed_limit_last_valid) and \
                       self.resolver.speed_limit_final_last > 0. and \
                       self.resolver.speed_limit_final_last != V_CRUISE_UNSET
    manual_cruise_below_limit = v_cruise < self.resolver.speed_limit_final_last
    above_manual_cruise = v_ego > v_cruise + SPEED_LIMIT_HANDOFF_EXIT_MARGIN
    reason = "inactive"

    if self.sla.is_active:
      self._speed_limit_handoff_active = False
      reason = "speed_limit_assist_active"
    elif (self._speed_limit_active_prev and long_enabled and not long_override and has_limit_target and
          manual_cruise_below_limit and above_manual_cruise):
      self._speed_limit_handoff_active = True
      reason = "handoff_active"
    elif not long_enabled or long_override or not has_limit_target or not manual_cruise_below_limit or not above_manual_cruise:
      self._speed_limit_handoff_active = False
      if not long_enabled:
        reason = "longitudinal_disabled"
      elif long_override:
        reason = "driver_override"
      elif not has_limit_target:
        reason = "no_limit_target"
      elif not manual_cruise_below_limit:
        reason = "manual_cruise_not_below_limit"
      elif not above_manual_cruise:
        reason = "ego_not_above_manual_cruise"

    self._speed_limit_active_prev = self.sla.is_active
    self.speed_limit_handoff_debug = {
      "speed_limit_handoff_active": bool(self._speed_limit_handoff_active),
      "manual_cruise_below_limit": bool(manual_cruise_below_limit),
      "above_manual_cruise": bool(above_manual_cruise),
      "reason": reason,
    }
    return self._speed_limit_handoff_active

  def _speed_limit_handoff_target(self, v_ego: float, a_ego: float) -> tuple[float, float]:
    return min(v_ego, self.resolver.speed_limit_final_last), min(a_ego, SPEED_LIMIT_HANDOFF_A_TARGET_MAX)

  def update_targets(self, sm: messaging.SubMaster, v_ego: float, a_ego: float, v_cruise: float,
                     coast_accel: float | None = None) -> tuple[float, float]:
    CS = sm['carState']
    v_cruise_cluster_kph = min(CS.vCruiseCluster, V_CRUISE_MAX)
    v_cruise_cluster = v_cruise_cluster_kph * CV.KPH_TO_MS

    long_enabled = sm['carControl'].enabled
    long_override = sm['carControl'].cruiseControl.override

    self.active_scc = self.scc
    self.active_resolver = self.resolver
    self.active_sla = self.sla
    mode_resolution = getattr(self, "longitudinal_mode_resolution", None)
    acc_mode_direct = bool(
      mode_resolution is not None and
      mode_resolution.requested_mode == LongitudinalMode.ACC and
      mode_resolution.actuation_type == LongitudinalActuationType.DIRECT
    )
    acc_mode_set_speed_advisory = bool(
      mode_resolution is not None and
      mode_resolution.actuation_type == LongitudinalActuationType.SET_SPEED_ADVISORY
    )
    if acc_mode_direct:
      self._speed_limit_handoff_active = False
      self._speed_limit_active_prev = False
      self.decision_candidates_sp = build_sp_longitudinal_candidates(
        False,
        (v_cruise, a_ego),
        (v_cruise, a_ego),
        False,
        (v_cruise, a_ego),
        False,
        (v_cruise, a_ego),
        (v_cruise, a_ego),
        False,
      )
      self.source = LongitudinalPlanSource.cruise
      self.output_v_target = v_cruise
      self.output_a_target = a_ego
      return self.output_v_target, self.output_a_target

    # Smart Cruise Control
    if not acc_mode_set_speed_advisory:
      self.scc.update(sm, long_enabled, long_override, v_ego, a_ego, v_cruise)

    # Speed Limit Resolver
    self.resolver.update(v_ego, sm, coast_accel=coast_accel)

    # Speed Limit Assist
    has_speed_limit = self.resolver.speed_limit_valid or self.resolver.speed_limit_last_valid
    self.sla.update(long_enabled, long_override, v_ego, a_ego, v_cruise_cluster, self.resolver.speed_limit,
                    self.resolver.speed_limit_final_last, has_speed_limit, self.resolver.distance, self.events_sp,
                    coast_accel=coast_accel)

    if not acc_mode_set_speed_advisory:
      self.osm_traffic_control_prior.update(sm, long_enabled, long_override, v_ego, a_ego)

    speed_limit_handoff_active = self._update_speed_limit_handoff(long_enabled, long_override, v_ego, v_cruise)
    speed_limit_active = self.sla.is_active or speed_limit_handoff_active
    speed_limit_assist_target = self._speed_limit_handoff_target(v_ego, a_ego) if speed_limit_handoff_active else (
      self.sla.output_v_target,
      a_ego,
    )
    self.speed_limit_handoff_debug.update({
      "handoff_target_v": float(speed_limit_assist_target[0]) if speed_limit_handoff_active else 0.0,
      "handoff_target_a": float(speed_limit_assist_target[1]) if speed_limit_handoff_active else 0.0,
    })
    cruise_target = (v_cruise, min(a_ego, SPEED_LIMIT_HANDOFF_A_TARGET_MAX)) if speed_limit_handoff_active else (v_cruise, a_ego)
    lead_one = sm['radarState'].leadOne
    primary_lead_context = getattr(self, "primary_lead_context", None)
    stack_resolution = getattr(self, "longitudinal_stack_resolution", None)
    if is_custom_stack(getattr(stack_resolution, "resolved_stack", "")) and getattr(primary_lead_context, "states", ()):
      lead_speedup_guard_active = should_block_lead_speedup_from_context(
        primary_lead_context, v_ego, bool(CS.gasPressed), bool(CS.brakePressed)
      )
    else:
      lead_speedup_guard_active = should_block_lead_speedup(
        v_ego,
        bool(lead_one.status),
        float(lead_one.dRel),
        float(lead_one.vRel),
        float(lead_one.yRel),
        bool(CS.gasPressed),
        bool(CS.brakePressed),
      )
    speed_limit_assist_target = apply_lead_speedup_guard(lead_speedup_guard_active, v_ego, speed_limit_assist_target)
    cruise_target = apply_lead_speedup_guard(lead_speedup_guard_active, v_ego, cruise_target)
    decision_cruise_target = speed_limit_assist_target if speed_limit_active else cruise_target
    if speed_limit_active:
      decision_cruise_target = (
        min(decision_cruise_target[0], v_ego + SPEED_LIMIT_SPEED_UP_ACCEL_CAP * SPEED_LIMIT_SPEED_UP_LOOKAHEAD),
        decision_cruise_target[1],
      )
    decision_speed_limit_active = False  # Active SLA is represented as effective driver intent below.

    scc_vision_target = (self.scc.vision.output_v_target, self.scc.vision.output_a_target)
    scc_map_target = (self.scc.map.output_v_target, self.scc.map.output_a_target)
    osm_target = (self.osm_traffic_control_prior.output_v_target, self.osm_traffic_control_prior.output_a_target)
    scc_vision_active = bool(self.scc.vision.is_active) and not acc_mode_set_speed_advisory
    scc_map_active = bool(self.scc.map.is_active) and not acc_mode_set_speed_advisory
    osm_active = bool(self.osm_traffic_control_prior.active) and not acc_mode_set_speed_advisory

    self.decision_candidates_sp = build_sp_longitudinal_candidates(
      decision_speed_limit_active,
      decision_cruise_target,
      scc_vision_target,
      scc_vision_active,
      scc_map_target,
      scc_map_active,
      speed_limit_assist_target,
      osm_target,
      osm_active,
    )

    self.source, self.output_v_target, self.output_a_target = select_lowest_longitudinal_target(
      speed_limit_active,
      cruise_target,
      scc_vision_target if scc_vision_active else (v_cruise, a_ego),
      scc_map_target if scc_map_active else (v_cruise, a_ego),
      speed_limit_assist_target,
      osm_target if osm_active else (v_cruise, a_ego),
      source_prev=getattr(self, 'source', None),
      v_target_prev=getattr(self, 'output_v_target', None),
    )
    return self.output_v_target, self.output_a_target

  def update(self, sm: messaging.SubMaster) -> None:
    self.events_sp.clear()
    self.longitudinal_mode_resolution = LongitudinalModeResolver.resolve(self.params, self.CP)
    self._update_e2e_alerts_for_mode(sm)

  def _update_e2e_alerts_for_mode(self, sm: messaging.SubMaster) -> None:
    if self.longitudinal_mode_resolution.e2e_like:
      self.e2e_alerts_helper.update(sm, self.events_sp)
    else:
      self.e2e_alerts_helper.green_light_alert = False
      self.e2e_alerts_helper.lead_depart_alert = False

  def _custom_v2_stack_output(self, sunnypilot_output: LongitudinalStackOutput,
                              accel_limits: tuple[float | None, float | None]) -> LongitudinalStackOutput:
    if self.custom_longitudinal_stack is None or self.custom_longitudinal_stack.stack_name != CUSTOM_V2:
      self.custom_longitudinal_stack = make_custom_longitudinal_stack(CUSTOM_V2)
    scene = getattr(self, "custom_v2_scene", CustomV2Scene())
    return self.custom_longitudinal_stack.update(
      sunnypilot_output,
      scene=scene,
      accel_limits=accel_limits,
      decision=getattr(self, "custom_v2_policy_decision", None),
      extra_rejected=getattr(self, "custom_v2_policy_extra_rejected", ()),
    )

  def _resolve_custom_v2_policy_decision(self, sunnypilot_output: LongitudinalStackOutput,
                                         accel_limits: tuple[float | None, float | None]):
    scene = getattr(self, "custom_v2_scene", CustomV2Scene())
    driver_candidates = tuple(
      candidate for candidate in getattr(self, "decision_candidates_sp", ())
      if candidate.role == CandidateRole.DRIVER_INTENT
    )
    provider_candidates = ensure_driver_intent(driver_candidates, sunnypilot_output, scene.v_cruise)
    extra_rejected: list[tuple[str, str]] = []

    one_pedal_candidate, one_pedal_rejected = build_one_pedal_driver_candidate(scene, scene.v_cruise, accel_limits)
    extra_rejected.extend(one_pedal_rejected)
    one_pedal_active = one_pedal_candidate is not None
    if one_pedal_candidate is not None:
      provider_candidates = replace_driver_intent(provider_candidates, one_pedal_candidate)

    advisory_candidates, advisory_rejected = build_custom_v2_advisory_candidates(scene)
    if one_pedal_active:
      extra_rejected.extend((
        (str(candidate.debug.get("custom_v2_intent", "driver_cruise")), "one_pedal_active")
        for candidate in advisory_candidates
      ))
      advisory_candidates = ()
    else:
      extra_rejected.extend(advisory_rejected)

    planner_seed_candidates = tuple(getattr(self, "planner_seed_candidates", ()))
    if planner_seed_candidates:
      selected_planner_seed = select_planner_seed_candidate((
        PlannerSeedCandidate("sunnypilot-current", sunnypilot_output), *planner_seed_candidates,
      ))
      selected_planner_seeds = [] if selected_planner_seed.name == "sunnypilot-current" else [selected_planner_seed]
      selected_planner_seeds.extend(
        candidate for candidate in planner_seed_candidates
        if candidate.selection != PLANNER_SEED_FLOOR and candidate.output.a_target < selected_planner_seed.output.a_target and
        not (selected_planner_seed.group and candidate.group == selected_planner_seed.group) and
        not (
          candidate.name == selected_planner_seed.name and candidate.reason == selected_planner_seed.reason and
          candidate.output.a_target == selected_planner_seed.output.a_target
        )
      )
      planner_seed_candidates = tuple(selected_planner_seeds)
    seed_candidates = planner_seed_candidates_to_longitudinal_candidates(planner_seed_candidates, scene.v_cruise)
    raw_physical_candidates = fallback_physical_candidates(
      seed_candidates, tuple(getattr(self, "longitudinal_decision_candidates", ())), sunnypilot_output
    )
    force_slow_candidate = build_force_slow_candidate(sunnypilot_output, scene, accel_limits)
    physical_candidates = (*seed_candidates, *raw_physical_candidates)
    if force_slow_candidate is not None:
      physical_candidates = (*physical_candidates, force_slow_candidate)

    progress_candidates: tuple[LongitudinalCandidate, ...] = ()
    if not one_pedal_active:
      progress_candidates, progress_rejected = build_custom_v2_progress_candidates(sunnypilot_output, scene, accel_limits)
      extra_rejected.extend(progress_rejected)

    candidates = (*provider_candidates, *advisory_candidates, *physical_candidates, *progress_candidates)
    arbiter = getattr(self, "longitudinal_arbiter", None)
    if arbiter is None:
      arbiter = LongitudinalArbiter()
      self.longitudinal_arbiter = arbiter
    source_stability_v_ego = None if (scene.force_slow_decel or scene.brake_pressed or scene.gas_pressed) else scene.v_ego
    decision = resolve_longitudinal_decision(
      enabled=True,
      candidates=candidates,
      fallback_v_target=max(0.0, scene.v_cruise),
      fallback_a_target=sunnypilot_output.a_target,
      fallback_should_stop=sunnypilot_output.should_stop,
      accel_limits=(float(accel_limits[0]), float(accel_limits[1])),
      arbiter=arbiter,
      v_ego=source_stability_v_ego,
    )
    self.longitudinal_decision = decision
    return decision, tuple(dict.fromkeys(extra_rejected))

  def _set_custom_v2_fault(self, reason: str) -> None:
    self._reset_longitudinal_source_stability()
    self.custom_v2_fault_latched = True
    self.custom_v2_fault_reason = reason
    self.longitudinal_stack_fault_latched = True
    self.longitudinal_stack_fault_reason = reason
    self.events_sp.add(custom.OnroadEventSP.EventName.customLongitudinalStackFault)

  def _reset_longitudinal_source_stability(self) -> None:
    arbiter = getattr(self, "longitudinal_arbiter", None)
    if arbiter is not None:
      arbiter.reset_source_stability()

  def _publish_custom_v2_policy_debug(self, output: LongitudinalStackOutput) -> None:
    debug = output.debug
    self.longitudinal_stack_selected_intent = str(debug.get("custom_v2_selected_intent", ""))
    self.longitudinal_stack_selected_reason = str(debug.get("custom_v2_selected_reason", ""))
    rejected_intents = tuple(str(intent) for intent in debug.get("custom_v2_rejected_intents", ()))
    rejected_reasons = tuple(str(reason) for reason in debug.get("custom_v2_rejected_reasons", ()))
    self.longitudinal_stack_rejected = tuple(zip(rejected_intents, rejected_reasons, strict=False))
    self.longitudinal_stack_seed_context = str(debug.get("custom_v2_seed_context", ""))
    self.longitudinal_stack_seed_candidate = str(debug.get("custom_v2_seed_candidate", ""))

  def apply_longitudinal_stack_selection(self, sm: messaging.SubMaster, has_lead: bool,
                                         accel_limits: tuple[float | None, float | None]) -> None:
    sunnypilot_output = planner_state_to_stack_output(self, has_lead, debug={"adapter": SUNNYPILOT_CURRENT})
    self.longitudinal_stack_actuated_stack = SUNNYPILOT_CURRENT
    self.longitudinal_stack_fault_latched = False
    self.longitudinal_stack_fault_reason = ""
    self.longitudinal_stack_selected_intent = ""
    self.longitudinal_stack_selected_reason = ""
    self.longitudinal_stack_rejected = ()
    self.longitudinal_stack_seed_context = ""
    self.longitudinal_stack_seed_candidate = ""
    self.custom_v2_policy_decision = None
    self.custom_v2_policy_extra_rejected = ()
    self.longitudinal_plan_source = sunnypilot_output.source

    resolved_stack = self.longitudinal_stack_resolution.resolved_stack
    if not is_custom_stack(resolved_stack):
      return

    if self.custom_longitudinal_stack is None or self.custom_longitudinal_stack.stack_name != resolved_stack:
      self.custom_longitudinal_stack = make_custom_longitudinal_stack(resolved_stack)
    if resolved_stack == CUSTOM_V2:
      if not bool(sm['selfdriveState'].enabled):
        self._reset_longitudinal_source_stability()
        self.custom_v2_fault_latched = False
        self.custom_v2_fault_reason = ""
        return
      if self.custom_v2_fault_latched:
        self._reset_longitudinal_source_stability()
        self._set_custom_v2_fault(self.custom_v2_fault_reason)
        return
      try:
        self.custom_v2_policy_decision, self.custom_v2_policy_extra_rejected = self._resolve_custom_v2_policy_decision(
          sunnypilot_output, accel_limits
        )
        custom_v2_output = self._custom_v2_stack_output(sunnypilot_output, accel_limits)
      except CustomV2SceneValidationError as error:
        self._set_custom_v2_fault(error.reason)
        return
      except Exception:
        self._set_custom_v2_fault("custom_exception")
        return
      validation = validate_stack_output(custom_v2_output, accel_limits)
      if not validation.valid:
        self._set_custom_v2_fault(validation.reason)
        return
      apply_stack_output_to_planner(self, custom_v2_output)
      self.longitudinal_stack_actuated_stack = resolved_stack
      self._publish_custom_v2_policy_debug(custom_v2_output)
      return

  def publish_longitudinal_plan_sp(self, sm: messaging.SubMaster, pm: messaging.PubMaster) -> None:
    plan_sp_send = messaging.new_message('longitudinalPlanSP')

    plan_sp_send.valid = sm.all_checks(service_list=['carState', 'controlsState'])

    longitudinalPlanSP = plan_sp_send.longitudinalPlanSP
    longitudinalPlanSP.longitudinalPlanSource = self.source
    longitudinalPlanSP.vTarget = float(self.output_v_target)
    longitudinalPlanSP.aTarget = float(self.output_a_target)
    longitudinalPlanSP.events = self.events_sp.to_msg()
    active_scc = getattr(self, "active_scc", self.scc)
    active_resolver = getattr(self, "active_resolver", self.resolver)
    active_sla = getattr(self, "active_sla", self.sla)

    # Dynamic Experimental Control
    mode_resolution = getattr(self, "longitudinal_mode_resolution", DEFAULT_LONGITUDINAL_MODE_RESOLUTION)
    dec = longitudinalPlanSP.dec
    dec.state = DecState.blended if mode_resolution.compatibility_alias_state == DecCompatibilityState.BLENDED else DecState.acc
    legacy_dec_enabled = legacy_dec_enabled_for_mode(mode_resolution)
    dec.enabled = legacy_dec_enabled
    dec.active = legacy_dec_enabled

    # Smart Cruise Control
    smartCruiseControl = longitudinalPlanSP.smartCruiseControl
    # Vision Control
    sccVision = smartCruiseControl.vision
    sccVision.state = active_scc.vision.state
    sccVision.vTarget = float(active_scc.vision.output_v_target)
    sccVision.aTarget = float(active_scc.vision.output_a_target)
    sccVision.currentLateralAccel = float(active_scc.vision.current_lat_acc)
    sccVision.maxPredictedLateralAccel = float(active_scc.vision.max_pred_lat_acc)
    sccVision.enabled = active_scc.vision.is_enabled
    sccVision.active = active_scc.vision.is_active
    # Map Control
    sccMap = smartCruiseControl.map
    sccMap.state = active_scc.map.state
    sccMap.vTarget = float(active_scc.map.output_v_target)
    sccMap.aTarget = float(active_scc.map.output_a_target)
    sccMap.enabled = active_scc.map.is_enabled
    sccMap.active = active_scc.map.is_active

    # Speed Limit
    speedLimit = longitudinalPlanSP.speedLimit
    resolver = speedLimit.resolver
    resolver.speedLimit = float(active_resolver.speed_limit)
    resolver.speedLimitLast = float(active_resolver.speed_limit_last)
    resolver.speedLimitFinal = float(active_resolver.speed_limit_final)
    resolver.speedLimitFinalLast = float(active_resolver.speed_limit_final_last)
    resolver.speedLimitValid = active_resolver.speed_limit_valid
    resolver.speedLimitLastValid = active_resolver.speed_limit_last_valid
    resolver.speedLimitOffset = float(active_resolver.speed_limit_offset)
    resolver.distToSpeedLimit = float(active_resolver.distance)
    resolver.source = active_resolver.source
    assist = speedLimit.assist
    assist.state = active_sla.state
    assist.enabled = active_sla.is_enabled
    assist.active = active_sla.is_active
    assist.autoCruiseEnabled = getattr(active_sla, "auto_enabled", False)
    assist.vTarget = float(active_sla.output_v_target)
    assist.aTarget = float(active_sla.output_a_target)

    # E2E Alerts
    e2eAlerts = longitudinalPlanSP.e2eAlerts
    e2eAlerts.greenLightAlert = self.e2e_alerts_helper.green_light_alert
    e2eAlerts.leadDepartAlert = self.e2e_alerts_helper.lead_depart_alert

    publish_decision_layer_telemetry(longitudinalPlanSP, getattr(self, "longitudinal_decision_telemetry", None))
    publish_stack_telemetry(
      longitudinalPlanSP,
      getattr(self, "longitudinal_stack_resolution"),
      actuated_stack=getattr(self, "longitudinal_stack_actuated_stack", SUNNYPILOT_CURRENT),
      actuated_a_target=float(self.output_a_target),
      fault_latched=bool(getattr(self, "longitudinal_stack_fault_latched", False)),
      fault_reason=getattr(self, "longitudinal_stack_fault_reason", ""),
      selected_intent=getattr(self, "longitudinal_stack_selected_intent", ""),
      selected_reason=getattr(self, "longitudinal_stack_selected_reason", ""),
      rejected=getattr(self, "longitudinal_stack_rejected", ()),
      seed_context=getattr(self, "longitudinal_stack_seed_context", ""),
      seed_candidate=getattr(self, "longitudinal_stack_seed_candidate", ""),
    )
    publish_longitudinal_mode_telemetry(longitudinalPlanSP, mode_resolution)

    pm.send('longitudinalPlanSP', plan_sp_send)
