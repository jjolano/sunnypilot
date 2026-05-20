from __future__ import annotations

from dataclasses import dataclass, replace
import math
from numbers import Real
from typing import Any

from cereal import log
from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N
from openpilot.selfdrive.controls.lib.longitudinal_stacks.interface import LongitudinalStackOutput
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

NO_LEAD_LAUNCH_ACCEL_MAX = 0.95
LEAD_PULLAWAY_ACCEL_MAX = 1.20
NO_LEAD_LAUNCH_MAX_V_EGO = 3.0
LEAD_PULLAWAY_MAX_V_EGO = 5.0
LEAD_MOTION_MIN_V = 0.15
EXCESS_GAP_MIN = 1.0
EXCESS_GAP_MAX = 8.0
EXCESS_GAP_ACCEL_MIN = 0.4
EXCESS_GAP_ACCEL_MAX = 1.0
EXCESS_GAP_CLOSING_TAPER = 0.3
EXCESS_GAP_CLOSING_BLOCK = 0.7
EXCESS_GAP_FAST_CLOSING_MID = 3.0
EXCESS_GAP_FAST_CLOSING_ZERO = 6.0
EXCESS_GAP_FAST_CLOSING_MID_CAP = 0.15
NO_LEAD_STOP_CLEAR_DISTANCE = 20.0
NO_LEAD_STOP_CLEAR_ACCEL_MIN = -0.5
MAP_ONLY_CAUTION_ACCEL_MIN = -0.3
COMFORT_RELAX_ACCEL_MIN = -0.5
CRUISE_LEEWAY_MIN = 5.0 * MPH_TO_MS
CRUISE_LEEWAY_MAX = 10.0 * MPH_TO_MS
CRUISE_LEEWAY_DOWNHILL_ACCEL = 0.25
CRUISE_LEEWAY_RECOVERY = 2.0 * MPH_TO_MS
STOP_APPROACH_COMFORT_DECEL = -0.2
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
  lead_gap_excess: float = 0.0
  lead_opening_prediction: bool = False
  lead_confirmed_pullaway: bool = False
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


@dataclass(frozen=True)
class CustomV2Decision:
  a_target: float
  should_stop: bool
  selected_intent: str
  selected_reason: str
  rejected: tuple[tuple[str, str], ...]
  limit_jerk: bool = True


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
  lead_moving = scene.lead_v >= LEAD_MOTION_MIN_V or scene.lead_v_rel > 0.0
  return bool(
    scene.has_lead and
    (scene.lead_confirmed_pullaway or scene.lead_opening_prediction or lead_moving) and
    not scene.independent_stop_threat
  )


def excess_gap_accel_cap(scene: CustomV2Scene) -> float | None:
  if not scene.has_lead or scene.lead_gap_excess <= EXCESS_GAP_MIN:
    return None
  if scene.lead_v < LEAD_MOTION_MIN_V:
    return None

  closing_speed = max(0.0, -scene.lead_v_rel)
  if closing_speed >= EXCESS_GAP_CLOSING_BLOCK and scene.lead_gap_excess < EXCESS_GAP_MAX:
    return None

  cap = _interp(scene.lead_gap_excess, EXCESS_GAP_MIN, EXCESS_GAP_MAX, EXCESS_GAP_ACCEL_MIN, EXCESS_GAP_ACCEL_MAX)
  if closing_speed > EXCESS_GAP_CLOSING_TAPER:
    taper = 1.0 - _clip(
      (closing_speed - EXCESS_GAP_CLOSING_TAPER) / (EXCESS_GAP_CLOSING_BLOCK - EXCESS_GAP_CLOSING_TAPER),
      0.0,
      1.0,
    )
    cap = EXCESS_GAP_ACCEL_MIN + taper * (cap - EXCESS_GAP_ACCEL_MIN)
  if closing_speed > EXCESS_GAP_CLOSING_BLOCK:
    if closing_speed <= EXCESS_GAP_FAST_CLOSING_MID:
      closing_cap = _interp(
        closing_speed, EXCESS_GAP_CLOSING_BLOCK, EXCESS_GAP_FAST_CLOSING_MID,
        EXCESS_GAP_ACCEL_MIN, EXCESS_GAP_FAST_CLOSING_MID_CAP,
      )
    else:
      closing_cap = _interp(
        closing_speed, EXCESS_GAP_FAST_CLOSING_MID, EXCESS_GAP_FAST_CLOSING_ZERO,
        EXCESS_GAP_FAST_CLOSING_MID_CAP, 0.0,
      )
    cap = min(cap, closing_cap)
  return cap


class CustomLongitudinalStackV2:
  stack_name = CUSTOM_V2

  def update(self, sunnypilot_output: LongitudinalStackOutput, scene: CustomV2Scene | None = None,
             accel_limits: tuple[float | None, float | None] = (None, None)) -> LongitudinalStackOutput:
    scene = _validated_scene(scene or CustomV2Scene())
    decision = self._decide(sunnypilot_output, scene, accel_limits)
    if _preserve_seed_trajectory(sunnypilot_output, decision):
      speeds, accels, jerks = sunnypilot_output.speeds, sunnypilot_output.accels, sunnypilot_output.jerks
    else:
      speeds, accels, jerks = _synth_trajectory(sunnypilot_output, scene, decision.a_target, decision.limit_jerk)
    debug: dict[str, Any] = dict(sunnypilot_output.debug)
    debug.update({
      "custom_stack": self.stack_name,
      "custom_v2_selected_intent": decision.selected_intent,
      "custom_v2_selected_reason": decision.selected_reason,
      "custom_v2_rejected_intents": tuple(intent for intent, _reason in decision.rejected[:3]),
      "custom_v2_rejected_reasons": tuple(reason for _intent, reason in decision.rejected[:3]),
      "custom_v2_intents": CUSTOM_V2_INTENTS,
      "custom_v2_one_pedal_mode": scene.one_pedal_mode,
      "custom_v2_one_pedal_cruise_hold": scene.one_pedal_cruise_hold,
    })
    return replace(
      sunnypilot_output,
      a_target=decision.a_target,
      should_stop=decision.should_stop,
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
        allow_lead_progress=not scene.stop_threat,
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
                              rejected: list[tuple[str, str]], allow_lead_progress: bool = True) -> tuple[float, str, str]:
    cruise_a = _dynamic_cruise_coast_accel(scene, a_target)
    if cruise_a > a_target:
      a_target = cruise_a
      selected_intent = "driver_cruise"
      selected_reason = "dynamic_overspeed_coast_leeway"

    wants_progress = scene.v_cruise > scene.v_ego
    if not wants_progress:
      rejected.append(("launch", "cruise_not_above_ego"))
      return a_target, selected_intent, selected_reason

    if not scene.has_lead and scene.v_ego < NO_LEAD_LAUNCH_MAX_V_EGO:
      if no_lead_stop_clear(scene):
        a_target, selected_intent, selected_reason = _apply_floor(
          a_target, selected_intent, selected_reason, NO_LEAD_LAUNCH_ACCEL_MAX,
          "launch", "no_lead_stop_clear", accel_limits,
        )
      else:
        rejected.append(("launch", "model_stop_not_clear"))

    lead_progress_allowed = allow_lead_progress and not (
      scene.has_lead and a_target < 0.0 and scene.lead_v_rel >= 0.0
    )
    if lead_progress_allowed and scene.has_lead and scene.v_ego < LEAD_PULLAWAY_MAX_V_EGO and lead_evidence_releases_stop(scene):
      a_target, selected_intent, selected_reason = _apply_floor(
        a_target, selected_intent, selected_reason, LEAD_PULLAWAY_ACCEL_MAX,
        "launch", "confirmed_lead_pullaway", accel_limits,
      )

    gap_cap = excess_gap_accel_cap(scene) if lead_progress_allowed else None
    if gap_cap is not None:
      a_target, selected_intent, selected_reason = _apply_floor(
        a_target, selected_intent, selected_reason, gap_cap,
        "lead_follow", "excess_gap_progress", accel_limits,
      )
    elif allow_lead_progress and scene.has_lead and scene.lead_gap_excess > EXCESS_GAP_MIN:
      rejected.append(("lead_follow", "closing_speed_guard"))

    return a_target, selected_intent, selected_reason

  def _apply_advisory_caps(self, a_target: float, selected_intent: str, selected_reason: str,
                           scene: CustomV2Scene, rejected: list[tuple[str, str]]) -> tuple[float, str, str]:
    if scene.speed_limit_active and scene.speed_limit_v_target > 0.0 and scene.speed_limit_v_target < scene.v_ego:
      cap = min(0.0, max(scene.speed_limit_a_target, scene.accel_coast))
      a_target, selected_intent, selected_reason = _apply_cap(
        a_target, selected_intent, selected_reason, cap, "speed_policy", "coast_biased_speed_reduction",
      )
    elif scene.speed_limit_active:
      rejected.append(("speed_policy", "no_speed_reduction_needed"))

    if scene.map_caution_active:
      if scene.map_caution_confirmed:
        cap = min(0.0, scene.map_caution_a_target)
        a_target, selected_intent, selected_reason = _apply_cap(
          a_target, selected_intent, selected_reason, cap, "map_caution", "confirmed_map_caution",
        )
      else:
        rejected.append(("map_caution", "map_only_ignored"))

    if scene.curve_active:
      a_target, selected_intent, selected_reason = _apply_cap(
        a_target, selected_intent, selected_reason, scene.curve_a_target,
        "curve_policy", "existing_custom_curve_thresholds",
      )

    return a_target, selected_intent, selected_reason


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


def _stop_approach_accel(scene: CustomV2Scene, current_a_target: float,
                         accel_limits: tuple[float | None, float | None]) -> tuple[float, str, bool]:
  stop_a_target = min(current_a_target, scene.model_desired_accel, STOP_APPROACH_COMFORT_DECEL)
  selected_reason = "comfort_early_stop_threat"
  hard_stop = False
  if scene.model_stop_distance is not None and scene.model_stop_distance > 0.0:
    required = -(scene.v_ego ** 2) / (2.0 * max(scene.model_stop_distance, 1.0))
    stop_a_target = min(stop_a_target, required)
    if scene.model_should_stop and required < STOP_APPROACH_DECEL_MIN:
      selected_reason = "hard_model_stop_threat"
      hard_stop = True
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
  return math.isclose(float(output.a_target), float(decision.a_target), abs_tol=A_TARGET_EPS)


def _validated_scene(scene: CustomV2Scene) -> CustomV2Scene:
  core_fields = (
    "v_ego", "v_cruise", "a_ego", "accel_coast", "lead_v", "lead_v_rel", "lead_gap_excess", "model_desired_accel",
  )
  for field_name in core_fields:
    if not _finite(getattr(scene, field_name)):
      raise CustomV2SceneValidationError(f"invalid_scene_{field_name}")

  if scene.model_stop_distance is not None and not _finite(scene.model_stop_distance):
    raise CustomV2SceneValidationError("invalid_scene_model_stop_distance")

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
    speed_limit_active=speed_limit_active,
    curve_active=curve_active,
    map_caution_active=map_caution_active,
    map_caution_confirmed=map_caution_confirmed,
    one_pedal_mode=_validated_one_pedal_mode(scene.one_pedal_mode),
    one_pedal_cruise_hold=bool(scene.one_pedal_cruise_hold),
  )


def _validated_one_pedal_mode(value: object) -> int:
  try:
    mode = int(value)
  except (TypeError, ValueError):
    return ONE_PEDAL_MODE_OFF
  return mode if mode in ONE_PEDAL_MODES else ONE_PEDAL_MODE_OFF


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
  accels: list[float] = []
  jerks: list[float] = []
  current_accel = prev_accel
  for _idx in range(CONTROL_N):
    if limit_jerk:
      delta = _clip(
        float(a_target) - current_accel,
        NORMAL_NEGATIVE_RETREAT_JERK * SYNTH_TRAJECTORY_DT,
        POSITIVE_PROGRESS_JERK * SYNTH_TRAJECTORY_DT,
      )
      next_accel = current_accel + delta
    else:
      next_accel = float(a_target)
    jerks.append((next_accel - current_accel) / SYNTH_TRAJECTORY_DT)
    accels.append(next_accel)
    current_accel = next_accel

  speeds: list[float] = []
  current_speed = max(0.0, v0)
  for accel in accels:
    speeds.append(current_speed)
    current_speed = max(0.0, current_speed + accel * SYNTH_TRAJECTORY_DT)
  return tuple(speeds), tuple(accels), tuple(jerks)
