from __future__ import annotations

from dataclasses import dataclass, replace
import math
from numbers import Real
from typing import Any

from cereal import log
from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N
from openpilot.selfdrive.controls.lib.longitudinal_decision import (
  CandidateRole,
  DecisionSource,
  LongitudinalCandidate,
  LongitudinalDecision,
  SOURCE_STABILITY_HOLD_REASON,
)
from openpilot.selfdrive.controls.lib.longitudinal_stacks.interface import LongitudinalStackOutput
from openpilot.selfdrive.controls.lib.longitudinal_stacks.policy import (
  CUSTOM_V2_DEBUG_DISABLE_JERK_LIMIT,
  CUSTOM_V2_DEBUG_INTENT,
  CUSTOM_V2_DEBUG_REASON,
  CUSTOM_V2_DEBUG_SEED_CANDIDATE,
  CUSTOM_V2_DEBUG_SEED_CONTEXT,
  CUSTOM_V2_DEBUG_STACK_OUTPUT,
  custom_v2_candidate_with_debug,
  custom_v2_intent_for_source,
  custom_v2_rejections_from_decision,
  selected_candidate_for_decision,
)
from openpilot.selfdrive.controls.lib.longitudinal_stacks.planner_seed import (
  PLANNER_SEED_INTENT_DRIVER_CRUISE,
  PLANNER_SEED_INTENT_LAUNCH,
  PLANNER_SEED_INTENT_LEAD_FOLLOW,
  PLANNER_SEED_INTENT_SAFETY_CAP,
  PLANNER_SEED_INTENT_STOP_APPROACH,
  PLANNER_SEED_MPC_REASON,
  planner_seed_intent_for_reason,
)
from openpilot.selfdrive.controls.lib.longitudinal_stacks.selector import CUSTOM_V2
from openpilot.selfdrive.controls.lib.vehicle_math import stopping_decel
from openpilot.selfdrive.modeld.constants import ModelConstants

MPH_TO_MS = 0.44704

CUSTOM_V2_INTENTS = (
  "driver_cruise",
  "lead_follow",
  "stop_approach",
  "launch",
  "one_pedal",
  "speed_policy",
  "curve_policy",
  "map_caution",
  "comfort_relax",
  "safety_cap",
)

ONE_PEDAL_MODE_OFF = 0
ONE_PEDAL_MODE_CREEP = 1
ONE_PEDAL_MODE_FULL_STOP = 2
ONE_PEDAL_MODES = {ONE_PEDAL_MODE_OFF, ONE_PEDAL_MODE_CREEP, ONE_PEDAL_MODE_FULL_STOP}

NO_LEAD_LAUNCH_ACCEL_MAX_BY_PERSONALITY = {
  log.LongitudinalPersonality.relaxed: 1.10,
  log.LongitudinalPersonality.standard: 1.35,
  log.LongitudinalPersonality.aggressive: 1.55,
}
STOP_APPROACH_COMFORT_DECEL_BY_PERSONALITY = {
  log.LongitudinalPersonality.relaxed: -0.30,
  log.LongitudinalPersonality.standard: -0.38,
  log.LongitudinalPersonality.aggressive: -0.45,
}
NO_LEAD_LAUNCH_MAX_V_EGO = 3.0
LEAD_MOTION_MIN_V = 0.15
LEAD_LATERAL_PROGRESS_BLOCK_Y = 0.6
EXCESS_GAP_MIN = 1.0
PROGRESS_CRUISE_SPEED_MARGIN = 0.2
NO_LEAD_STOP_CLEAR_DISTANCE = 20.0
NO_LEAD_STOP_CLEAR_ACCEL_MIN = -0.5
MAP_ONLY_CAUTION_ACCEL_MIN = -0.3
COMFORT_RELAX_ACCEL_MIN = -0.5
CRUISE_LEEWAY_MIN = 5.0 * MPH_TO_MS
CRUISE_LEEWAY_MAX = 10.0 * MPH_TO_MS
CRUISE_LEEWAY_DOWNHILL_ACCEL = 0.25
CRUISE_LEEWAY_RECOVERY = 2.0 * MPH_TO_MS
STOP_APPROACH_DECEL_MIN = -1.5
SAFETY_FORCE_SLOW_DECEL = -0.2
ONE_PEDAL_LIFT_OFF_COAST_ACCEL = 0.0
ONE_PEDAL_CREEP_TARGET_SPEED = 1.2
ONE_PEDAL_CREEP_ACCEL_GAIN = 0.5
ONE_PEDAL_CREEP_ACCEL_MAX = 0.35
ONE_PEDAL_CREEP_ROLLING_MIN_SPEED = 0.05
ONE_PEDAL_FULL_STOP_ARM_SPEED = 2.5
ONE_PEDAL_FULL_STOP_DECEL = -0.3
ONE_PEDAL_FULL_STOP_HOLD_SPEED = 0.3
SYNTH_TRAJECTORY_DT = 0.2
POSITIVE_PROGRESS_JERK = 4.0
NORMAL_NEGATIVE_RETREAT_JERK = -5.0
A_TARGET_EPS = 1e-4
LONGITUDINAL_PLAN_SOURCE = log.LongitudinalPlan.LongitudinalPlanSource
LEAD_MPC_SOURCE_VALUES = {int(LONGITUDINAL_PLAN_SOURCE.lead0), int(LONGITUDINAL_PLAN_SOURCE.lead1)}
E2E_SOURCE_VALUES = {int(LONGITUDINAL_PLAN_SOURCE.e2e)}
LEAD_MPC_SEED_REASON = "lead_mpc_seed"
E2E_SEED_REASON = "e2e_seed"
PROGRESS_PRESERVING_INTENTS = {
  PLANNER_SEED_INTENT_LAUNCH,
  PLANNER_SEED_INTENT_LEAD_FOLLOW,
  PLANNER_SEED_INTENT_SAFETY_CAP,
  PLANNER_SEED_INTENT_STOP_APPROACH,
}

class CustomV2SceneValidationError(ValueError):
  def __init__(self, reason: str):
    self.reason = reason
    super().__init__(reason)


@dataclass(frozen=True)
class CustomV2Scene:
  v_ego: float = 0.0
  v_cruise: float = 0.0
  a_ego: float = 0.0
  accel_coast: float = 0.0
  force_slow_decel: bool = False
  brake_pressed: bool = False
  gas_pressed: bool = False
  has_lead: bool = False
  lead_v: float = 0.0
  lead_v_rel: float = 0.0
  lead_y_rel: float = 0.0
  lead_gap_excess: float = 0.0
  lead_follow_gap_excess: float | None = None
  lead_lateral_progress_blocked: bool = False
  lead_progress_allowed: bool = False
  lead_opening_prediction: bool = False
  lead_confirmed_pullaway: bool = False
  primary_physical_lead_idx: int = -1
  primary_behavior_lead_idx: int = -1
  primary_lead_reason: str = ""
  primary_lead_authority: str = ""
  alternate_lead_threat_active: bool = False
  shadow_lead_active: bool = False
  lead_release_blocked_reason: str = ""
  stop_threat: bool = False
  independent_stop_threat: bool = False
  model_should_stop: bool = False
  model_stop_distance: float | None = None
  model_desired_accel: float = 0.0
  speed_limit_active: bool = False
  speed_limit_v_target: float = 0.0
  speed_limit_a_target: float = 0.0
  curve_active: bool = False
  curve_a_target: float = 0.0
  map_caution_active: bool = False
  map_caution_confirmed: bool = False
  map_caution_a_target: float = 0.0
  one_pedal_mode: int = ONE_PEDAL_MODE_OFF
  one_pedal_cruise_hold: bool = False
  personality: int = log.LongitudinalPersonality.standard
  fast_lead_motion_evidence_enabled: bool = False
  fast_lead_motion_opening: bool = False
  fast_lead_motion_moving: bool = False
  allow_speed_limit_advisory: bool = True
  allow_curve_advisory: bool = True
  allow_map_caution_advisory: bool = True
  allow_no_lead_progress: bool = True
  allow_lead_progress: bool = True
  lead_pullaway_phase: str = "hold"
  lead_pullaway_reason: str = ""
  lead_pullaway_track_id: int = -1
  lead_pullaway_pulse_timer: float = 0.0
  lead_pullaway_cooldown_timer: float = 0.0
  lead_pullaway_gap_excess: float = 0.0
  lead_pullaway_predicted_gap_opening: float = 0.0
  lead_pullaway_a_floor: float = 0.0
  lead_pullaway_rejected_reason: str = ""
  lead_pullaway_predicted_gap: float = 0.0
  lead_pullaway_safe_accel_cap: float = 0.0
  lead_pullaway_lead_accel_trend: float = 0.0
  lead_pullaway_runway_margin: float = 0.0
  lead_pullaway_runway_margin_now: float = 0.0
  lead_pullaway_runway_margin_t: float = 0.0
  lead_pullaway_runway_creation: float = 0.0
  lead_pullaway_lead_created_runway: bool = False
  lead_pullaway_early_authority: bool = False
  lead_pullaway_early_authority_reason: str = ""
  lead_pullaway_pulse_floor: float = 0.0
  lead_pullaway_pulse_cap: float = 0.0
  lead_pullaway_coast_required: bool = False
  lead_pullaway_pulse_capped_by_runway: bool = False
  lead_pullaway_crawl_cap_released_by_runway: bool = False
  lead_pullaway_low_speed_step_cap_suppressed_by_runway: bool = False
  lead_pullaway_runway_trend: str = "stable"
  lead_pullaway_selected_or_rejected_reason: str = ""


@dataclass(frozen=True)
class CustomV2Decision:
  a_target: float
  should_stop: bool
  selected_intent: str
  selected_reason: str
  rejected: tuple[tuple[str, str], ...]
  limit_jerk: bool = True
  trajectory_output: LongitudinalStackOutput | None = None


def dynamic_cruise_overspeed_leeway(accel_coast: float) -> float:
  confidence = _clip((float(accel_coast) / CRUISE_LEEWAY_DOWNHILL_ACCEL), 0.0, 1.0)
  return CRUISE_LEEWAY_MIN + confidence * (CRUISE_LEEWAY_MAX - CRUISE_LEEWAY_MIN)


def no_lead_stop_clear(scene: CustomV2Scene) -> bool:
  stop_distance = scene.model_stop_distance
  return bool(
    not scene.model_should_stop and
    (stop_distance is None or stop_distance > NO_LEAD_STOP_CLEAR_DISTANCE) and
    scene.model_desired_accel >= NO_LEAD_STOP_CLEAR_ACCEL_MIN
  )


def lead_evidence_releases_stop(scene: CustomV2Scene) -> bool:
  lead_opening = scene.fast_lead_motion_opening if scene.fast_lead_motion_evidence_enabled else scene.lead_v_rel > 0.0
  lead_moving = scene.fast_lead_motion_moving if scene.fast_lead_motion_evidence_enabled else (lead_opening or (scene.v_ego < LEAD_MOTION_MIN_V and scene.lead_v >= LEAD_MOTION_MIN_V))
  return bool(
    scene.has_lead and
    scene.lead_progress_allowed and
    (scene.lead_confirmed_pullaway or scene.lead_opening_prediction or lead_moving) and
    not scene.independent_stop_threat
  )


def _no_lead_launch_accel_max(scene: CustomV2Scene) -> float:
  return NO_LEAD_LAUNCH_ACCEL_MAX_BY_PERSONALITY[_validated_personality(scene.personality)]


def _stop_approach_comfort_decel(scene: CustomV2Scene) -> float:
  return STOP_APPROACH_COMFORT_DECEL_BY_PERSONALITY[_validated_personality(scene.personality)]


class CustomLongitudinalStackV2:
  stack_name = CUSTOM_V2

  def update(self, sunnypilot_output: LongitudinalStackOutput, scene: CustomV2Scene | None = None,
             accel_limits: tuple[float | None, float | None] = (None, None),
             decision: LongitudinalDecision | None = None,
             extra_rejected: tuple[tuple[str, str], ...] = ()) -> LongitudinalStackOutput:
    scene = _validated_scene(scene or CustomV2Scene())
    custom_decision = self._decide_from_core(sunnypilot_output, scene, accel_limits, decision, extra_rejected) if decision is not None else \
      self._decide(sunnypilot_output, scene, accel_limits)
    trajectory_output = custom_decision.trajectory_output or sunnypilot_output
    if _preserve_seed_trajectory(trajectory_output, custom_decision):
      speeds, accels, jerks = trajectory_output.speeds, trajectory_output.accels, trajectory_output.jerks
    else:
      speeds, accels, jerks = _synth_trajectory(trajectory_output, scene, custom_decision.a_target, custom_decision.limit_jerk)
    debug: dict[str, Any] = dict(sunnypilot_output.debug)
    if custom_decision.trajectory_output is not None:
      debug.update(custom_decision.trajectory_output.debug)
    debug.update({
      "custom_stack": self.stack_name,
      "custom_v2_selected_intent": custom_decision.selected_intent,
      "custom_v2_selected_reason": custom_decision.selected_reason,
      "custom_v2_rejected_intents": tuple(intent for intent, _reason in custom_decision.rejected),
      "custom_v2_rejected_reasons": tuple(reason for _intent, reason in custom_decision.rejected),
      "custom_v2_intents": CUSTOM_V2_INTENTS,
      "custom_v2_one_pedal_mode": scene.one_pedal_mode,
      "custom_v2_one_pedal_cruise_hold": scene.one_pedal_cruise_hold,
      "primary_physical_lead_idx": int(scene.primary_physical_lead_idx),
      "primary_behavior_lead_idx": int(scene.primary_behavior_lead_idx),
      "primary_lead_reason": str(scene.primary_lead_reason),
      "primary_lead_authority": str(scene.primary_lead_authority),
      "alternate_lead_threat_active": bool(scene.alternate_lead_threat_active),
      "shadow_lead_active": bool(scene.shadow_lead_active),
      "lead_progress_allowed": bool(scene.lead_progress_allowed),
      "lead_release_blocked_reason": str(scene.lead_release_blocked_reason),
      "lead_pullaway_phase": str(scene.lead_pullaway_phase),
      "lead_pullaway_reason": str(scene.lead_pullaway_reason),
      "lead_pullaway_track_id": int(scene.lead_pullaway_track_id),
      "lead_pullaway_pulse_timer": float(scene.lead_pullaway_pulse_timer),
      "lead_pullaway_cooldown_timer": float(scene.lead_pullaway_cooldown_timer),
      "lead_pullaway_gap_excess": float(scene.lead_pullaway_gap_excess),
      "lead_pullaway_predicted_gap_opening": float(scene.lead_pullaway_predicted_gap_opening),
      "lead_pullaway_a_floor": float(scene.lead_pullaway_a_floor),
      "lead_pullaway_rejected_reason": str(scene.lead_pullaway_rejected_reason),
      "lead_pullaway_predicted_gap": float(scene.lead_pullaway_predicted_gap),
      "lead_pullaway_safe_accel_cap": float(scene.lead_pullaway_safe_accel_cap),
      "lead_pullaway_lead_accel_trend": float(scene.lead_pullaway_lead_accel_trend),
      "lead_pullaway_runway_margin": float(scene.lead_pullaway_runway_margin),
      "lead_pullaway_runway_margin_now": float(scene.lead_pullaway_runway_margin_now),
      "lead_pullaway_runway_margin_t": float(scene.lead_pullaway_runway_margin_t),
      "lead_pullaway_runway_creation": float(scene.lead_pullaway_runway_creation),
      "lead_pullaway_lead_created_runway": bool(scene.lead_pullaway_lead_created_runway),
      "lead_pullaway_early_authority": bool(scene.lead_pullaway_early_authority),
      "lead_pullaway_early_authority_reason": str(scene.lead_pullaway_early_authority_reason),
      "lead_pullaway_pulse_floor": float(scene.lead_pullaway_pulse_floor),
      "lead_pullaway_pulse_cap": float(scene.lead_pullaway_pulse_cap),
      "lead_pullaway_coast_required": bool(scene.lead_pullaway_coast_required),
      "lead_pullaway_pulse_capped_by_runway": bool(scene.lead_pullaway_pulse_capped_by_runway),
      "lead_pullaway_crawl_cap_released_by_runway": bool(scene.lead_pullaway_crawl_cap_released_by_runway),
      "lead_pullaway_low_speed_step_cap_suppressed_by_runway": bool(scene.lead_pullaway_low_speed_step_cap_suppressed_by_runway),
      "lead_pullaway_runway_trend": str(scene.lead_pullaway_runway_trend),
      "lead_pullaway_selected_or_rejected_reason": str(scene.lead_pullaway_selected_or_rejected_reason),
    })
    return replace(
      sunnypilot_output,
      a_target=custom_decision.a_target,
      should_stop=custom_decision.should_stop,
      speeds=speeds,
      accels=accels,
      jerks=jerks,
      debug=debug,
    )

  def _decide(self, output: LongitudinalStackOutput, scene: CustomV2Scene,
              accel_limits: tuple[float | None, float | None]) -> CustomV2Decision:
    a_target = _clip_to_limits(output.a_target, accel_limits)
    selected_intent, selected_reason = _classify_seed(output, scene)
    should_stop = bool(output.should_stop)
    rejected: list[tuple[str, str]] = []
    limit_jerk = True

    stop_active = scene.stop_threat and not scene.has_lead
    blocked = scene.force_slow_decel or scene.brake_pressed or scene.gas_pressed
    one_pedal_enabled = scene.one_pedal_mode != ONE_PEDAL_MODE_OFF
    one_pedal_policy_active = one_pedal_enabled and not scene.one_pedal_cruise_hold

    if one_pedal_enabled and scene.one_pedal_cruise_hold:
      rejected.append(("one_pedal", "temporary_cruise_hold"))

    progress_floors_allowed = _progress_floors_allowed(selected_intent)
    if one_pedal_policy_active:
      if not blocked and not stop_active:
        a_target, should_stop, selected_intent, selected_reason, limit_jerk = self._apply_one_pedal_policy(
          output, a_target, should_stop, selected_intent, selected_reason, scene, accel_limits, rejected,
        )
      elif blocked:
        rejected.append(("one_pedal", "driver_or_force_blocked"))
    elif not blocked and not stop_active and progress_floors_allowed:
      a_target, selected_intent, selected_reason = self._apply_progress_floors(
        a_target, selected_intent, selected_reason, scene, accel_limits, rejected,
        allow_no_lead_progress=scene.allow_no_lead_progress,
        allow_lead_progress=scene.allow_lead_progress and not scene.stop_threat,
      )
    elif not blocked and not stop_active:
      rejected.append((selected_intent, "planner_seed_preserved"))
    elif blocked:
      rejected.append(("launch", "driver_or_force_blocked"))

    if not one_pedal_policy_active:
      a_target, selected_intent, selected_reason = self._apply_advisory_caps(
        a_target, selected_intent, selected_reason, scene, rejected
      )

    if not one_pedal_policy_active and selected_intent in {"speed_policy", "curve_policy", "map_caution"} and _comfort_relax_allowed(scene):
      relaxed = max(a_target, COMFORT_RELAX_ACCEL_MIN)
      if relaxed > a_target:
        rejected.append((selected_intent, "comfort_relax_softened_advisory_decel"))
        a_target = relaxed
        selected_intent = "comfort_relax"
        selected_reason = "clear_margin_advisory_softening"

    if stop_active:
      stop_a_target, stop_reason, hard_stop = _stop_approach_accel(scene, a_target, accel_limits)
      if stop_a_target < a_target or scene.model_should_stop:
        a_target = stop_a_target
        should_stop = should_stop or scene.model_should_stop
        selected_intent = "stop_approach"
        selected_reason = stop_reason
        limit_jerk = not hard_stop

    if scene.force_slow_decel:
      a_target = min(a_target, SAFETY_FORCE_SLOW_DECEL)
      should_stop = True
      selected_intent = "safety_cap"
      selected_reason = "force_slow_decel"
      limit_jerk = False

    return CustomV2Decision(
      a_target=_clip_to_limits(a_target, accel_limits),
      should_stop=should_stop,
      selected_intent=selected_intent,
      selected_reason=selected_reason,
      rejected=tuple(rejected),
      limit_jerk=limit_jerk,
    )

  def _decide_from_core(self, output: LongitudinalStackOutput, scene: CustomV2Scene,
                        accel_limits: tuple[float | None, float | None], decision: LongitudinalDecision,
                        extra_rejected: tuple[tuple[str, str], ...]) -> CustomV2Decision:
    selected = selected_candidate_for_decision(decision)
    rejected = [*custom_v2_rejections_from_decision(decision), *extra_rejected]
    if not decision.enabled or selected is None or decision.winner == DecisionSource.LEGACY_FALLBACK:
      selected_intent, selected_reason = _classify_seed(output, scene)
      return CustomV2Decision(
        a_target=_clip_to_limits(output.a_target, accel_limits),
        should_stop=bool(output.should_stop),
        selected_intent=selected_intent,
        selected_reason=decision.fallback_reason or decision.active_reason or selected_reason,
        rejected=tuple(dict.fromkeys(rejected)),
        trajectory_output=output,
      )

    debug = selected.debug
    selected_intent = str(debug.get(CUSTOM_V2_DEBUG_INTENT) or custom_v2_intent_for_source(selected.source))
    selected_reason = str(debug.get(CUSTOM_V2_DEBUG_REASON) or decision.active_reason or selected.active_reason)
    trajectory_output = debug.get(CUSTOM_V2_DEBUG_STACK_OUTPUT)
    if not isinstance(trajectory_output, LongitudinalStackOutput):
      trajectory_output = output
    else:
      trajectory_debug = dict(trajectory_output.debug)
      trajectory_debug[CUSTOM_V2_DEBUG_SEED_CONTEXT] = str(debug.get(CUSTOM_V2_DEBUG_SEED_CONTEXT, ""))
      trajectory_debug[CUSTOM_V2_DEBUG_SEED_CANDIDATE] = str(debug.get(CUSTOM_V2_DEBUG_SEED_CANDIDATE, ""))
      trajectory_output = replace(trajectory_output, debug=trajectory_debug)

    a_target = float(decision.a_target)
    should_stop = bool(decision.should_stop)
    limit_jerk = not bool(debug.get(CUSTOM_V2_DEBUG_DISABLE_JERK_LIMIT, False))

    if _decision_held_by_source_stability(decision):
      return CustomV2Decision(
        a_target=_clip_to_limits(a_target, accel_limits),
        should_stop=should_stop,
        selected_intent=selected_intent,
        selected_reason=selected_reason,
        rejected=tuple(dict.fromkeys(rejected)),
        limit_jerk=limit_jerk,
        trajectory_output=trajectory_output,
      )

    if selected.role == CandidateRole.DRIVER_INTENT and selected_intent == "driver_cruise":
      a_target = float(selected.a_target)
      should_stop = bool(selected.should_stop)
      trajectory_output = output
      selected_reason = "sunnypilot_current_seed"
    elif selected.role == CandidateRole.ADVISORY_CAP:
      a_target, selected_intent, selected_reason = _advisory_accel_for_selected_candidate(
        selected, scene, a_target, selected_intent, selected_reason
      )
    elif selected.role == CandidateRole.PHYSICAL_HAZARD:
      should_stop = bool(should_stop or trajectory_output.should_stop)

    extra = debug.get("custom_v2_extra_rejected", ())
    if isinstance(extra, tuple):
      rejected.extend((str(intent), str(reason)) for intent, reason in extra)

    if selected.role == CandidateRole.ADVISORY_CAP and selected_intent in {"speed_policy", "curve_policy", "map_caution"} and _comfort_relax_allowed(scene):
      relaxed = max(a_target, COMFORT_RELAX_ACCEL_MIN)
      if relaxed > a_target:
        rejected.append((selected_intent, "comfort_relax_softened_advisory_decel"))
        a_target = relaxed
        selected_intent = "comfort_relax"
        selected_reason = "clear_margin_advisory_softening"

    return CustomV2Decision(
      a_target=_clip_to_limits(a_target, accel_limits),
      should_stop=should_stop,
      selected_intent=selected_intent,
      selected_reason=selected_reason,
      rejected=tuple(dict.fromkeys(rejected)),
      limit_jerk=limit_jerk,
      trajectory_output=trajectory_output,
    )

  def _apply_one_pedal_policy(self, output: LongitudinalStackOutput, a_target: float, should_stop: bool,
                              selected_intent: str, selected_reason: str, scene: CustomV2Scene,
                              accel_limits: tuple[float | None, float | None],
                              rejected: list[tuple[str, str]]) -> tuple[float, bool, str, str, bool]:
    if _one_pedal_preserves_physical_braking(output, selected_intent, a_target, should_stop):
      rejected.append(("one_pedal", "physical_hazard_preserved"))
      return a_target, should_stop, selected_intent, selected_reason, True

    if scene.one_pedal_mode == ONE_PEDAL_MODE_FULL_STOP and scene.v_ego <= ONE_PEDAL_FULL_STOP_ARM_SPEED:
      return (
        _clip_to_limits(ONE_PEDAL_FULL_STOP_DECEL, accel_limits),
        should_stop or scene.v_ego <= ONE_PEDAL_FULL_STOP_HOLD_SPEED,
        "one_pedal",
        "low_speed_terminal_stop",
        True,
      )

    if scene.one_pedal_mode == ONE_PEDAL_MODE_CREEP and _one_pedal_creep_allowed(scene):
      return (
        _clip_to_limits(_one_pedal_creep_accel(scene), accel_limits),
        False,
        "one_pedal",
        "terminal_creep",
        True,
      )

    if scene.one_pedal_mode == ONE_PEDAL_MODE_CREEP and scene.v_ego <= ONE_PEDAL_CREEP_TARGET_SPEED:
      rejected.append(("one_pedal", "terminal_creep_not_authorized"))

    return (
      _clip_to_limits(ONE_PEDAL_LIFT_OFF_COAST_ACCEL, accel_limits),
      False,
      "one_pedal",
      "lift_off_coast",
      True,
    )

  def _apply_progress_floors(self, a_target: float, selected_intent: str, selected_reason: str,
                              scene: CustomV2Scene, accel_limits: tuple[float | None, float | None],
                              rejected: list[tuple[str, str]], allow_no_lead_progress: bool = True,
                              allow_lead_progress: bool = True) -> tuple[float, str, str]:
    cruise_a = _dynamic_cruise_coast_accel(scene, a_target)
    if cruise_a > a_target:
      a_target = cruise_a
      selected_intent = "driver_cruise"
      selected_reason = "dynamic_overspeed_coast_leeway"

    wants_progress = scene.v_cruise > scene.v_ego + PROGRESS_CRUISE_SPEED_MARGIN
    if not wants_progress:
      rejected.append(("launch", "cruise_not_above_ego"))
      return a_target, selected_intent, selected_reason

    if not scene.has_lead and not allow_no_lead_progress:
      rejected.append(("launch", "mode_boundary_blocked"))
    elif not scene.has_lead and scene.v_ego < NO_LEAD_LAUNCH_MAX_V_EGO:
      if no_lead_stop_clear(scene):
        a_target, selected_intent, selected_reason = _apply_floor(
          a_target, selected_intent, selected_reason, _no_lead_launch_accel_max(scene),
          "launch", "no_lead_stop_clear", accel_limits,
        )
      else:
        rejected.append(("launch", "model_stop_not_clear"))

    lead_seed_rejected_reason = scene.lead_pullaway_rejected_reason or scene.lead_release_blocked_reason
    if allow_lead_progress and scene.has_lead and lead_seed_rejected_reason:
      rejected.append(("launch", lead_seed_rejected_reason))
    elif allow_lead_progress and scene.has_lead and _lead_follow_gap_excess(scene) > EXCESS_GAP_MIN:
      rejected.append(("lead_follow", "planner_seed_required"))

    return a_target, selected_intent, selected_reason

  def _apply_advisory_caps(self, a_target: float, selected_intent: str, selected_reason: str,
                           scene: CustomV2Scene, rejected: list[tuple[str, str]]) -> tuple[float, str, str]:
    if scene.speed_limit_active and not scene.allow_speed_limit_advisory:
      rejected.append(("speed_policy", "mode_boundary_blocked"))
    elif scene.speed_limit_active and scene.speed_limit_v_target > 0.0 and scene.speed_limit_v_target < scene.v_ego:
      cap = min(0.0, max(scene.speed_limit_a_target, scene.accel_coast))
      a_target, selected_intent, selected_reason = _apply_cap(
        a_target, selected_intent, selected_reason, cap, "speed_policy", "coast_biased_speed_reduction",
      )
    elif scene.speed_limit_active:
      rejected.append(("speed_policy", "no_speed_reduction_needed"))

    if scene.map_caution_active and not scene.allow_map_caution_advisory:
      rejected.append(("map_caution", "mode_boundary_blocked"))
    elif scene.map_caution_active:
      if scene.map_caution_confirmed:
        cap = min(0.0, scene.map_caution_a_target)
        a_target, selected_intent, selected_reason = _apply_cap(
          a_target, selected_intent, selected_reason, cap, "map_caution", "confirmed_map_caution",
        )
      else:
        rejected.append(("map_caution", "map_only_ignored"))

    if scene.curve_active and not scene.allow_curve_advisory:
      rejected.append(("curve_policy", "mode_boundary_blocked"))
    elif scene.curve_active:
      a_target, selected_intent, selected_reason = _apply_cap(
        a_target, selected_intent, selected_reason, scene.curve_a_target,
        "curve_policy", "existing_custom_curve_thresholds",
      )

    return a_target, selected_intent, selected_reason


def build_one_pedal_driver_candidate(scene: CustomV2Scene, v_target: float,
                                      accel_limits: tuple[float | None, float | None]) -> tuple[LongitudinalCandidate | None, tuple[tuple[str, str], ...]]:
  scene = _validated_scene(scene)
  if scene.one_pedal_mode == ONE_PEDAL_MODE_OFF:
    return None, ()
  if scene.one_pedal_cruise_hold:
    return None, (("one_pedal", "temporary_cruise_hold"),)
  if scene.force_slow_decel or scene.brake_pressed or scene.gas_pressed:
    return None, (("one_pedal", "driver_or_force_blocked"),)

  extra_rejected: tuple[tuple[str, str], ...] = ()
  if scene.one_pedal_mode == ONE_PEDAL_MODE_FULL_STOP and scene.v_ego <= ONE_PEDAL_FULL_STOP_ARM_SPEED:
    a_target = _clip_to_limits(ONE_PEDAL_FULL_STOP_DECEL, accel_limits)
    should_stop = scene.v_ego <= ONE_PEDAL_FULL_STOP_HOLD_SPEED
    reason = "low_speed_terminal_stop"
  elif scene.one_pedal_mode == ONE_PEDAL_MODE_CREEP and _one_pedal_creep_allowed(scene):
    a_target = _clip_to_limits(_one_pedal_creep_accel(scene), accel_limits)
    should_stop = False
    reason = "terminal_creep"
  else:
    if scene.one_pedal_mode == ONE_PEDAL_MODE_CREEP and scene.v_ego <= ONE_PEDAL_CREEP_TARGET_SPEED:
      extra_rejected = (("one_pedal", "terminal_creep_not_authorized"),)
    a_target = _clip_to_limits(ONE_PEDAL_LIFT_OFF_COAST_ACCEL, accel_limits)
    should_stop = False
    reason = "lift_off_coast"

  candidate = custom_v2_candidate_with_debug(
    LongitudinalCandidate(
      source=DecisionSource.CRUISE,
      role=CandidateRole.DRIVER_INTENT,
      v_target=max(0.0, float(v_target)),
      a_target=a_target,
      confidence=1.0,
      urgency=0.1,
      active_reason=reason,
      should_stop=should_stop,
    ),
    intent="one_pedal",
    reason=reason,
    extra_rejected=extra_rejected,
  )
  return candidate, extra_rejected


def build_force_slow_candidate(output: LongitudinalStackOutput, scene: CustomV2Scene,
                               accel_limits: tuple[float | None, float | None]) -> LongitudinalCandidate | None:
  scene = _validated_scene(scene)
  if not scene.force_slow_decel:
    return None
  a_target = _clip_to_limits(min(float(output.a_target), SAFETY_FORCE_SLOW_DECEL), accel_limits)
  force_output = replace(output, a_target=a_target, should_stop=True)
  return custom_v2_candidate_with_debug(
    LongitudinalCandidate(
      source=DecisionSource.STOP_LAUNCH,
      role=CandidateRole.PHYSICAL_HAZARD,
      v_target=max(0.0, scene.v_cruise),
      a_target=a_target,
      confidence=1.0,
      urgency=1.0,
      active_reason="force_slow_decel",
      should_stop=True,
    ),
    intent="safety_cap",
    reason="force_slow_decel",
    output=force_output,
    seed_context="scene_physical",
    disable_jerk_limit=True,
  )


def build_custom_v2_advisory_candidates(scene: CustomV2Scene, *, allow_speed_limit: bool = True,
                                        allow_curve: bool = True,
                                        allow_map_caution: bool = True) -> tuple[tuple[LongitudinalCandidate, ...], tuple[tuple[str, str], ...]]:
  scene = _validated_scene(scene)
  candidates: list[LongitudinalCandidate] = []
  rejected: list[tuple[str, str]] = []

  if scene.speed_limit_active and not allow_speed_limit:
    rejected.append(("speed_policy", "mode_boundary_blocked"))
  elif scene.speed_limit_active and scene.speed_limit_v_target > 0.0 and scene.speed_limit_v_target < scene.v_ego:
    cap = min(0.0, max(scene.speed_limit_a_target, scene.accel_coast))
    candidates.append(custom_v2_candidate_with_debug(
      LongitudinalCandidate(
        source=DecisionSource.SPEED_LIMIT,
        role=CandidateRole.ADVISORY_CAP,
        v_target=max(0.0, scene.speed_limit_v_target),
        a_target=cap,
        confidence=0.85,
        urgency=0.35,
        active_reason="coast_biased_speed_reduction",
        required_a_target=cap,
      ),
      intent="speed_policy",
      reason="coast_biased_speed_reduction",
    ))
  elif scene.speed_limit_active:
    rejected.append(("speed_policy", "no_speed_reduction_needed"))

  if scene.map_caution_active and not allow_map_caution:
    rejected.append(("map_caution", "mode_boundary_blocked"))
  elif scene.map_caution_active:
    if scene.map_caution_confirmed:
      candidates.append(custom_v2_candidate_with_debug(
        LongitudinalCandidate(
          source=DecisionSource.OSM_TRAFFIC_CONTROL,
          role=CandidateRole.ADVISORY_CAP,
          v_target=max(0.0, scene.v_ego - 0.1),
          a_target=min(0.0, scene.map_caution_a_target),
          confidence=0.75,
          urgency=0.55,
          active_reason="confirmed_map_caution",
          required_a_target=min(0.0, scene.map_caution_a_target),
        ),
        intent="map_caution",
        reason="confirmed_map_caution",
      ))
    else:
      rejected.append(("map_caution", "map_only_ignored"))

  if scene.curve_active and not allow_curve:
    rejected.append(("curve_policy", "mode_boundary_blocked"))
  elif scene.curve_active:
    candidates.append(custom_v2_candidate_with_debug(
      LongitudinalCandidate(
        source=DecisionSource.SCC_VISION,
        role=CandidateRole.ADVISORY_CAP,
        v_target=max(0.0, scene.v_ego - 0.1),
        a_target=scene.curve_a_target,
        confidence=0.80,
        urgency=0.45,
        active_reason="existing_custom_curve_thresholds",
        required_a_target=scene.curve_a_target,
      ),
      intent="curve_policy",
      reason="existing_custom_curve_thresholds",
    ))

  return tuple(candidates), tuple(rejected)


def build_custom_v2_progress_candidates(output: LongitudinalStackOutput, scene: CustomV2Scene,
                                        accel_limits: tuple[float | None, float | None], *,
                                        allow_no_lead_progress: bool = True,
                                        allow_lead_progress: bool = True) -> tuple[tuple[LongitudinalCandidate, ...], tuple[tuple[str, str], ...]]:
  # These are custom-v2 scene-derived RELAXATION candidates, intentionally allowed
  # in addition to planner seeds. They are subordinate progress floors, not safety
  # authority, so LongitudinalDecisionCore must rank them below physical hazards
  # and advisory caps. Caller-owned one-pedal handling bypasses them; this builder
  # also blocks them for force_slow, driver brake/gas, and active stop threats.
  # lead_progress_allowed is the only field that may authorize lead-derived positive progress;
  # has_lead alone must never authorize lead progress.
  scene = _validated_scene(scene)
  candidates: list[LongitudinalCandidate] = []
  rejected: list[tuple[str, str]] = []
  blocked = scene.force_slow_decel or scene.brake_pressed or scene.gas_pressed
  stop_active = scene.stop_threat and not scene.has_lead
  if blocked:
    rejected.append(("launch", "driver_or_force_blocked"))
    return (), tuple(rejected)
  if stop_active:
    return (), ()

  cruise_a = _dynamic_cruise_coast_accel(scene, float(output.a_target))
  if cruise_a > float(output.a_target):
    candidates.append(_custom_v2_relaxation_candidate(
      DecisionSource.CRUISE_COAST, "driver_cruise", "dynamic_overspeed_coast_leeway",
      scene, cruise_a, bool(output.should_stop), accel_limits,
    ))

  wants_progress = scene.v_cruise > scene.v_ego + PROGRESS_CRUISE_SPEED_MARGIN
  if not wants_progress:
    rejected.append(("launch", "cruise_not_above_ego"))
    return tuple(candidates), tuple(rejected)

  if not scene.has_lead and not allow_no_lead_progress:
    rejected.append(("launch", "mode_boundary_blocked"))
  elif not scene.has_lead and scene.v_ego < NO_LEAD_LAUNCH_MAX_V_EGO:
    if no_lead_stop_clear(scene):
      candidates.append(_custom_v2_relaxation_candidate(
        DecisionSource.STOP_LAUNCH, "launch", "no_lead_stop_clear", scene,
        _clip_to_limits(_no_lead_launch_accel_max(scene), accel_limits), False, accel_limits,
      ))
    else:
      rejected.append(("launch", "model_stop_not_clear"))

  lead_seed_rejected_reason = scene.lead_pullaway_rejected_reason or scene.lead_release_blocked_reason
  if allow_lead_progress and scene.has_lead and lead_seed_rejected_reason:
    rejected.append(("launch", lead_seed_rejected_reason))
  elif scene.has_lead and _lead_follow_gap_excess(scene) > EXCESS_GAP_MIN:
    rejected.append(("lead_follow", "planner_seed_required"))

  return tuple(candidates), tuple(rejected)


def _custom_v2_relaxation_candidate(source: DecisionSource, intent: str, reason: str, scene: CustomV2Scene,
                                    a_target: float, should_stop: bool,
                                    accel_limits: tuple[float | None, float | None]) -> LongitudinalCandidate:
  return custom_v2_candidate_with_debug(
    LongitudinalCandidate(
      source=source,
      role=CandidateRole.RELAXATION,
      v_target=max(0.0, scene.v_cruise),
      a_target=_clip_to_limits(a_target, accel_limits),
      confidence=0.90,
      urgency=0.55 if source == DecisionSource.STOP_LAUNCH else 0.20,
      active_reason=reason,
      should_stop=should_stop,
    ),
    intent=intent,
    reason=reason,
  )


def _advisory_accel_for_selected_candidate(selected: LongitudinalCandidate, scene: CustomV2Scene, a_target: float,
                                           selected_intent: str, selected_reason: str) -> tuple[float, str, str]:
  if selected.source == DecisionSource.SPEED_LIMIT:
    return min(0.0, max(scene.speed_limit_a_target, scene.accel_coast)), "speed_policy", "coast_biased_speed_reduction"
  if selected.source == DecisionSource.OSM_TRAFFIC_CONTROL:
    return min(0.0, scene.map_caution_a_target), "map_caution", "confirmed_map_caution"
  if selected.source in (DecisionSource.SCC_VISION, DecisionSource.SCC_MAP):
    return scene.curve_a_target, "curve_policy", "existing_custom_curve_thresholds"
  return a_target, selected_intent, selected_reason


def _decision_held_by_source_stability(decision: LongitudinalDecision) -> bool:
  return any(reason == SOURCE_STABILITY_HOLD_REASON for _source, reason in decision.suppressed)


def _dynamic_cruise_coast_accel(scene: CustomV2Scene, a_target: float) -> float:
  if scene.has_lead or scene.stop_threat or scene.force_slow_decel or scene.v_cruise <= 0.0:
    return a_target
  overspeed = scene.v_ego - scene.v_cruise
  if overspeed <= 0.0:
    return a_target

  leeway = dynamic_cruise_overspeed_leeway(scene.accel_coast)
  if overspeed <= leeway:
    return min(0.0, max(a_target, scene.accel_coast))

  recovery = _clip((overspeed - leeway) / CRUISE_LEEWAY_RECOVERY, 0.0, 1.0)
  coast_target = (1.0 - recovery) * min(0.0, scene.accel_coast) + recovery * a_target
  return min(0.0, max(a_target, coast_target))


def _lead_follow_gap_excess(scene: CustomV2Scene) -> float:
  return scene.lead_gap_excess if scene.lead_follow_gap_excess is None else scene.lead_follow_gap_excess


def _stop_approach_accel(scene: CustomV2Scene, current_a_target: float,
                         accel_limits: tuple[float | None, float | None]) -> tuple[float, str, bool]:
  comfort_decel = _stop_approach_comfort_decel(scene)
  stop_a_target = min(current_a_target, comfort_decel)
  selected_reason = "comfort_early_stop_threat"
  hard_stop = False
  if scene.model_stop_distance is not None and scene.model_stop_distance > 0.0:
    required = stopping_decel(scene.v_ego, scene.model_stop_distance, min_distance=1.0)
    if required < comfort_decel:
      stop_a_target = min(stop_a_target, required)
    if scene.model_should_stop:
      stop_a_target = min(stop_a_target, scene.model_desired_accel, required)
    if scene.model_should_stop and required < STOP_APPROACH_DECEL_MIN:
      selected_reason = "hard_model_stop_threat"
      hard_stop = True
  elif scene.model_should_stop:
    stop_a_target = min(stop_a_target, scene.model_desired_accel)
  if hard_stop:
    return _clip_to_limits(stop_a_target, accel_limits), selected_reason, True
  return max(STOP_APPROACH_DECEL_MIN, stop_a_target), selected_reason, False


def _comfort_relax_allowed(scene: CustomV2Scene) -> bool:
  return bool(
    not scene.has_lead and
    not scene.stop_threat and
    not scene.force_slow_decel and
    not scene.brake_pressed and
    not scene.gas_pressed and
    not (scene.map_caution_active and scene.map_caution_confirmed)
  )


def _one_pedal_preserves_physical_braking(output: LongitudinalStackOutput, selected_intent: str,
                                          a_target: float, should_stop: bool) -> bool:
  if selected_intent == PLANNER_SEED_INTENT_SAFETY_CAP:
    return a_target <= 0.0 or should_stop or output.should_stop
  return bool(
    selected_intent in {PLANNER_SEED_INTENT_LEAD_FOLLOW, PLANNER_SEED_INTENT_STOP_APPROACH, PLANNER_SEED_INTENT_SAFETY_CAP} and
    (a_target < 0.0 or should_stop or output.should_stop)
  )


def _one_pedal_creep_allowed(scene: CustomV2Scene) -> bool:
  if scene.v_ego > ONE_PEDAL_CREEP_TARGET_SPEED:
    return False
  if scene.has_lead:
    return lead_evidence_releases_stop(scene)
  return scene.v_ego > ONE_PEDAL_CREEP_ROLLING_MIN_SPEED or no_lead_stop_clear(scene)


def _one_pedal_creep_accel(scene: CustomV2Scene) -> float:
  return _clip(
    (ONE_PEDAL_CREEP_TARGET_SPEED - scene.v_ego) * ONE_PEDAL_CREEP_ACCEL_GAIN,
    0.0,
    ONE_PEDAL_CREEP_ACCEL_MAX,
  )


def _apply_floor(a_target: float, selected_intent: str, selected_reason: str, floor: float, intent: str, reason: str,
                 accel_limits: tuple[float | None, float | None]) -> tuple[float, str, str]:
  floor = _clip_to_limits(floor, accel_limits)
  if floor > a_target:
    return floor, intent, reason
  return a_target, selected_intent, selected_reason


def _apply_cap(a_target: float, selected_intent: str, selected_reason: str, cap: float, intent: str, reason: str) -> tuple[float, str, str]:
  if cap < a_target:
    return cap, intent, reason
  return a_target, selected_intent, selected_reason


def _clip_to_limits(value: float, accel_limits: tuple[float | None, float | None]) -> float:
  lower, upper = accel_limits
  if lower is not None:
    value = max(float(lower), value)
  if upper is not None:
    value = min(float(upper), value)
  return float(value)


def _classify_seed(output: LongitudinalStackOutput, scene: CustomV2Scene) -> tuple[str, str]:
  seed_intent = str(output.seed_intent or "")
  seed_reason = str(output.seed_reason or "")
  if seed_intent:
    if seed_reason == PLANNER_SEED_MPC_REASON and scene.force_slow_decel:
      return PLANNER_SEED_INTENT_SAFETY_CAP, seed_reason
    return seed_intent, seed_reason or seed_intent

  reason = str(output.debug.get("planner_seed_candidate_reason", ""))
  if reason:
    if reason == PLANNER_SEED_MPC_REASON and scene.force_slow_decel:
      return PLANNER_SEED_INTENT_SAFETY_CAP, reason
    return planner_seed_intent_for_reason(
      reason,
      output.has_lead or scene.has_lead,
      output.should_stop,
      output.source,
    ), reason
  if _source_matches(output.source, E2E_SOURCE_VALUES, {"e2e"}):
    return PLANNER_SEED_INTENT_STOP_APPROACH, E2E_SEED_REASON
  if _source_matches(output.source, LEAD_MPC_SOURCE_VALUES, {"lead0", "lead1"}):
    return PLANNER_SEED_INTENT_LEAD_FOLLOW, LEAD_MPC_SEED_REASON
  return PLANNER_SEED_INTENT_DRIVER_CRUISE, "sunnypilot_current_seed"


def _progress_floors_allowed(selected_intent: str) -> bool:
  return selected_intent not in PROGRESS_PRESERVING_INTENTS


def _source_matches(source: object, values: set[int], names: set[str]) -> bool:
  if isinstance(source, Real):
    return int(source) in values
  source_name = str(source or "")
  if source_name in names:
    return True
  try:
    return int(source_name) in values
  except ValueError:
    return False


def _preserve_seed_trajectory(output: LongitudinalStackOutput, decision: CustomV2Decision) -> bool:
  if bool(output.debug.get("planner_seed_scalar", False)):
    return False
  return math.isclose(float(output.a_target), float(decision.a_target), abs_tol=A_TARGET_EPS)


def _validated_scene(scene: CustomV2Scene) -> CustomV2Scene:
  core_fields = (
    "v_ego", "v_cruise", "a_ego", "accel_coast", "lead_v", "lead_v_rel", "lead_y_rel", "lead_gap_excess", "model_desired_accel",
    "lead_pullaway_pulse_timer", "lead_pullaway_cooldown_timer", "lead_pullaway_gap_excess",
    "lead_pullaway_predicted_gap_opening", "lead_pullaway_a_floor", "lead_pullaway_predicted_gap",
    "lead_pullaway_safe_accel_cap", "lead_pullaway_lead_accel_trend", "lead_pullaway_runway_margin",
    "lead_pullaway_runway_margin_now", "lead_pullaway_runway_margin_t", "lead_pullaway_runway_creation",
    "lead_pullaway_pulse_floor", "lead_pullaway_pulse_cap",
  )
  for field_name in core_fields:
    if not _finite(getattr(scene, field_name)):
      raise CustomV2SceneValidationError(f"invalid_scene_{field_name}")

  if scene.model_stop_distance is not None and not _finite(scene.model_stop_distance):
    raise CustomV2SceneValidationError("invalid_scene_model_stop_distance")
  if scene.lead_follow_gap_excess is not None and not _finite(scene.lead_follow_gap_excess):
    raise CustomV2SceneValidationError("invalid_scene_lead_follow_gap_excess")

  speed_limit_active = bool(scene.speed_limit_active)
  if speed_limit_active and not (_finite(scene.speed_limit_v_target) and _finite(scene.speed_limit_a_target)):
    speed_limit_active = False

  curve_active = bool(scene.curve_active)
  if curve_active and not _finite(scene.curve_a_target):
    curve_active = False

  map_caution_active = bool(scene.map_caution_active)
  map_caution_confirmed = bool(scene.map_caution_confirmed)
  if map_caution_active and not _finite(scene.map_caution_a_target):
    map_caution_active = False
    map_caution_confirmed = False

  return replace(
    scene,
    personality=_validated_personality(scene.personality),
    speed_limit_active=speed_limit_active,
    curve_active=curve_active,
    map_caution_active=map_caution_active,
    map_caution_confirmed=map_caution_confirmed,
    lead_progress_allowed=bool(scene.lead_progress_allowed),
    primary_physical_lead_idx=int(scene.primary_physical_lead_idx),
    primary_behavior_lead_idx=int(scene.primary_behavior_lead_idx),
    primary_lead_reason=str(scene.primary_lead_reason),
    primary_lead_authority=str(scene.primary_lead_authority),
    alternate_lead_threat_active=bool(scene.alternate_lead_threat_active),
    shadow_lead_active=bool(scene.shadow_lead_active),
    lead_release_blocked_reason=str(scene.lead_release_blocked_reason),
    lead_pullaway_phase=str(scene.lead_pullaway_phase),
    lead_pullaway_reason=str(scene.lead_pullaway_reason),
    lead_pullaway_track_id=int(scene.lead_pullaway_track_id),
    lead_pullaway_rejected_reason=str(scene.lead_pullaway_rejected_reason),
    lead_pullaway_early_authority_reason=str(scene.lead_pullaway_early_authority_reason),
    lead_pullaway_coast_required=bool(scene.lead_pullaway_coast_required),
    lead_pullaway_pulse_capped_by_runway=bool(scene.lead_pullaway_pulse_capped_by_runway),
    lead_pullaway_lead_created_runway=bool(scene.lead_pullaway_lead_created_runway),
    lead_pullaway_early_authority=bool(scene.lead_pullaway_early_authority),
    lead_pullaway_crawl_cap_released_by_runway=bool(scene.lead_pullaway_crawl_cap_released_by_runway),
    lead_pullaway_low_speed_step_cap_suppressed_by_runway=bool(scene.lead_pullaway_low_speed_step_cap_suppressed_by_runway),
    lead_pullaway_runway_trend=str(scene.lead_pullaway_runway_trend),
    lead_pullaway_selected_or_rejected_reason=str(scene.lead_pullaway_selected_or_rejected_reason),
    one_pedal_mode=_validated_one_pedal_mode(scene.one_pedal_mode),
    one_pedal_cruise_hold=bool(scene.one_pedal_cruise_hold),
  )


def _validated_one_pedal_mode(value: object) -> int:
  try:
    mode = int(value)
  except (TypeError, ValueError):
    return ONE_PEDAL_MODE_OFF
  return mode if mode in ONE_PEDAL_MODES else ONE_PEDAL_MODE_OFF


def _validated_personality(value: object) -> int:
  try:
    personality = int(value)
  except (TypeError, ValueError):
    return log.LongitudinalPersonality.standard
  return personality if personality in NO_LEAD_LAUNCH_ACCEL_MAX_BY_PERSONALITY else log.LongitudinalPersonality.standard


def _finite(value: object) -> bool:
  return isinstance(value, Real) and math.isfinite(float(value))


def _clip(value: float, lower: float, upper: float) -> float:
  return max(lower, min(upper, value))


def _interp(value: float, x0: float, x1: float, y0: float, y1: float) -> float:
  if x1 == x0:
    return y1
  ratio = _clip((value - x0) / (x1 - x0), 0.0, 1.0)
  return y0 + ratio * (y1 - y0)


def _synth_trajectory(output: LongitudinalStackOutput, scene: CustomV2Scene,
                      a_target: float, limit_jerk: bool) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
  speeds_in = tuple(output.speeds)
  accels_in = tuple(output.accels)
  v0 = scene.v_ego if math.isfinite(scene.v_ego) and scene.v_ego >= 0.0 else (float(speeds_in[0]) if speeds_in else 0.0)
  prev_accel = float(accels_in[0]) if accels_in else float(a_target)
  dts = _synth_trajectory_dts()
  accels: list[float] = []
  jerks: list[float] = []
  current_accel = prev_accel
  for dt in dts:
    if limit_jerk:
      delta = _clip(
        float(a_target) - current_accel,
        NORMAL_NEGATIVE_RETREAT_JERK * dt,
        POSITIVE_PROGRESS_JERK * dt,
      )
      next_accel = current_accel + delta
    else:
      next_accel = float(a_target)
    jerks.append((next_accel - current_accel) / dt)
    accels.append(next_accel)
    current_accel = next_accel

  speeds: list[float] = []
  current_speed = max(0.0, v0)
  for accel, dt in zip(accels, dts, strict=True):
    speeds.append(current_speed)
    current_speed = max(0.0, current_speed + accel * dt)
  return tuple(speeds), tuple(accels), tuple(jerks)


def _synth_trajectory_dts(t_idxs: object = None) -> tuple[float, ...]:
  if t_idxs is None:
    t_idxs = ModelConstants.T_IDXS
  try:
    times = tuple(float(t) for t in t_idxs[:CONTROL_N])
  except (TypeError, ValueError):
    return (SYNTH_TRAJECTORY_DT,) * CONTROL_N
  if len(times) < CONTROL_N or not all(math.isfinite(t) for t in times):
    return (SYNTH_TRAJECTORY_DT,) * CONTROL_N

  intervals = [times[idx + 1] - times[idx] for idx in range(CONTROL_N - 1)]
  dts = [*intervals, intervals[-1]]
  if not all(math.isfinite(dt) and dt > 0.0 for dt in dts):
    return (SYNTH_TRAJECTORY_DT,) * CONTROL_N
  return tuple(dts)
