from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any

from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N
from openpilot.selfdrive.controls.lib.longitudinal_stacks.interface import LongitudinalStackOutput
from openpilot.selfdrive.controls.lib.longitudinal_stacks.selector import CUSTOM_V2

MPH_TO_MS = 0.44704

CUSTOM_V2_INTENTS = (
  "driver_cruise",
  "lead_follow",
  "stop_approach",
  "launch",
  "speed_policy",
  "curve_policy",
  "map_caution",
  "comfort_relax",
  "safety_cap",
)

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
NO_LEAD_STOP_CLEAR_DISTANCE = 20.0
NO_LEAD_STOP_CLEAR_ACCEL_MIN = -0.5
MAP_ONLY_CAUTION_ACCEL_MIN = -0.3
COMFORT_RELAX_ACCEL_MIN = -0.5
CRUISE_LEEWAY_MIN = 3.0 * MPH_TO_MS
CRUISE_LEEWAY_MAX = 7.0 * MPH_TO_MS
CRUISE_LEEWAY_DOWNHILL_ACCEL = 0.25
CRUISE_LEEWAY_RECOVERY = 2.0 * MPH_TO_MS
STOP_APPROACH_COMFORT_DECEL = -0.2
STOP_APPROACH_DECEL_MIN = -1.5
SAFETY_FORCE_SLOW_DECEL = -0.2
SYNTH_TRAJECTORY_DT = 0.2


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


@dataclass(frozen=True)
class CustomV2Decision:
  a_target: float
  should_stop: bool
  selected_intent: str
  selected_reason: str
  rejected: tuple[tuple[str, str], ...]


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
  return cap


class CustomLongitudinalStackV2:
  stack_name = CUSTOM_V2

  def update(self, sunnypilot_output: LongitudinalStackOutput, scene: CustomV2Scene | None = None,
             accel_limits: tuple[float | None, float | None] = (None, None)) -> LongitudinalStackOutput:
    scene = scene or CustomV2Scene()
    decision = self._decide(sunnypilot_output, scene, accel_limits)
    speeds, accels, jerks = _synth_trajectory(sunnypilot_output, scene, decision.a_target)
    debug: dict[str, Any] = dict(sunnypilot_output.debug)
    debug.update({
      "custom_stack": self.stack_name,
      "custom_v2_selected_intent": decision.selected_intent,
      "custom_v2_selected_reason": decision.selected_reason,
      "custom_v2_rejected_intents": tuple(intent for intent, _reason in decision.rejected[:3]),
      "custom_v2_rejected_reasons": tuple(reason for _intent, reason in decision.rejected[:3]),
      "custom_v2_intents": CUSTOM_V2_INTENTS,
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
    selected_intent = "driver_cruise"
    selected_reason = "sunnypilot_current_seed"
    should_stop = bool(output.should_stop)
    rejected: list[tuple[str, str]] = []
    planner_seed = output.debug.get("custom_v2_seed_context") == "planner"
    allow_v2_lead_progress = not (planner_seed and scene.has_lead)

    stop_released_by_lead = allow_v2_lead_progress and lead_evidence_releases_stop(scene)
    stop_active = scene.stop_threat and not stop_released_by_lead and allow_v2_lead_progress
    blocked = scene.force_slow_decel or scene.brake_pressed or scene.gas_pressed

    if scene.stop_threat and stop_released_by_lead:
      should_stop = False
      rejected.append(("stop_approach", "lead_pullaway_release"))

    if not blocked and not stop_active:
      a_target, selected_intent, selected_reason = self._apply_progress_floors(
        a_target, selected_intent, selected_reason, scene, accel_limits, rejected,
        allow_lead_progress=allow_v2_lead_progress,
      )
    elif blocked:
      rejected.append(("launch", "driver_or_force_blocked"))

    a_target, selected_intent, selected_reason = self._apply_advisory_caps(
      a_target, selected_intent, selected_reason, scene, rejected
    )

    if selected_intent in {"speed_policy", "curve_policy", "map_caution"} and _comfort_relax_allowed(scene):
      relaxed = max(a_target, COMFORT_RELAX_ACCEL_MIN)
      if relaxed > a_target:
        rejected.append((selected_intent, "comfort_relax_softened_advisory_decel"))
        a_target = relaxed
        selected_intent = "comfort_relax"
        selected_reason = "clear_margin_advisory_softening"

    if stop_active:
      stop_a_target = _stop_approach_accel(scene, a_target)
      if stop_a_target < a_target or scene.model_should_stop:
        a_target = stop_a_target
        should_stop = should_stop or scene.model_should_stop
        selected_intent = "stop_approach"
        selected_reason = "comfort_early_stop_threat"

    if scene.force_slow_decel:
      a_target = min(a_target, SAFETY_FORCE_SLOW_DECEL)
      should_stop = True
      selected_intent = "safety_cap"
      selected_reason = "force_slow_decel"

    return CustomV2Decision(
      a_target=_clip_to_limits(a_target, accel_limits),
      should_stop=should_stop,
      selected_intent=selected_intent,
      selected_reason=selected_reason,
      rejected=tuple(rejected),
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

    if allow_lead_progress and scene.has_lead and scene.v_ego < LEAD_PULLAWAY_MAX_V_EGO and lead_evidence_releases_stop(scene):
      a_target, selected_intent, selected_reason = _apply_floor(
        a_target, selected_intent, selected_reason, LEAD_PULLAWAY_ACCEL_MAX,
        "launch", "confirmed_lead_pullaway", accel_limits,
      )

    gap_cap = excess_gap_accel_cap(scene) if allow_lead_progress else None
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
      cap = scene.map_caution_a_target if scene.map_caution_confirmed else max(scene.map_caution_a_target, MAP_ONLY_CAUTION_ACCEL_MIN)
      cap = min(0.0, cap)
      a_target, selected_intent, selected_reason = _apply_cap(
        a_target, selected_intent, selected_reason, cap, "map_caution",
        "confirmed_map_caution" if scene.map_caution_confirmed else "map_only_preparation",
      )

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


def _stop_approach_accel(scene: CustomV2Scene, current_a_target: float) -> float:
  stop_a_target = min(current_a_target, scene.model_desired_accel, STOP_APPROACH_COMFORT_DECEL)
  if scene.model_stop_distance is not None and scene.model_stop_distance > 0.0:
    required = -(scene.v_ego ** 2) / (2.0 * max(scene.model_stop_distance, 1.0))
    stop_a_target = min(stop_a_target, required)
  return max(STOP_APPROACH_DECEL_MIN, stop_a_target)


def _comfort_relax_allowed(scene: CustomV2Scene) -> bool:
  return bool(
    not scene.has_lead and
    not scene.stop_threat and
    not scene.force_slow_decel and
    not scene.brake_pressed and
    not scene.gas_pressed
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


def _clip(value: float, lower: float, upper: float) -> float:
  return max(lower, min(upper, value))


def _interp(value: float, x0: float, x1: float, y0: float, y1: float) -> float:
  if x1 == x0:
    return y1
  ratio = _clip((value - x0) / (x1 - x0), 0.0, 1.0)
  return y0 + ratio * (y1 - y0)


def _synth_trajectory(output: LongitudinalStackOutput, scene: CustomV2Scene,
                      a_target: float) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
  speeds_in = tuple(output.speeds)
  accels_in = tuple(output.accels)
  v0 = scene.v_ego if math.isfinite(scene.v_ego) and scene.v_ego >= 0.0 else (float(speeds_in[0]) if speeds_in else 0.0)
  speeds = tuple(max(0.0, v0 + a_target * SYNTH_TRAJECTORY_DT * idx) for idx in range(CONTROL_N))
  accels = tuple(float(a_target) for _ in range(CONTROL_N))
  prev_accel = float(accels_in[0]) if accels_in else float(a_target)
  initial_jerk = (float(a_target) - prev_accel) / SYNTH_TRAJECTORY_DT
  jerks = (initial_jerk, *tuple(0.0 for _ in range(CONTROL_N - 1)))
  return speeds, accels, jerks
