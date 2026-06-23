import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from openpilot.sunnypilot.custom.lateral.demand.types import DEMAND_SOURCE_MODEL_PATH


LANE_CENTERING_ASSIST_MIN_SPEED = 5.0
LANE_CENTERING_ASSIST_MIN_PATH_QUALITY = 0.85
LANE_CENTERING_ASSIST_OK_REASON = "ok"
LANE_CENTERING_ASSIST_PATH_REASON_COOLDOWN_FRAMES = 50  # 0.5 s at 100 Hz
LANE_CENTERING_ASSIST_PATH_REASON_COOLDOWN_REASON = "path_reason_cooldown"
LANE_CENTERING_ASSIST_NEAR_LOOKAHEAD_T = 0.35
LANE_CENTERING_ASSIST_PREVIEW_T = 1.20
LANE_CENTERING_ASSIST_NEAR_LOOKAHEAD_MIN_M = 3.0
LANE_CENTERING_ASSIST_PREVIEW_MIN_M = 8.0
LANE_CENTERING_ASSIST_LATERAL_DEADBAND = 0.03
LANE_CENTERING_ASSIST_GROWTH_DEADBAND = 0.015
LANE_CENTERING_ASSIST_SIGN_HYSTERESIS_NUDGE = 1e-6
LANE_CENTERING_ASSIST_MAX_LAT_ACCEL = 0.08
LANE_CENTERING_ASSIST_MAX_LAT_ACCEL_BP = [10.0, 20.0, 30.0]
LANE_CENTERING_ASSIST_MAX_LAT_ACCEL_V = [0.12, 0.06, 0.025]
LANE_CENTERING_ASSIST_SPEED_FLOOR = 5.0
LANE_CENTERING_ASSIST_LATERAL_GAIN = 0.00045
LANE_CENTERING_ASSIST_HEADING_GAIN = 0.0020
LANE_CENTERING_ASSIST_GROWTH_GAIN = 0.00070
LANE_CENTERING_ASSIST_BUILD_RATE = 0.00030
LANE_CENTERING_ASSIST_RELEASE_RATE = 0.00060
LANE_CENTERING_ASSIST_STRAIGHT_MIN_SPEED = 24.0
LANE_CENTERING_ASSIST_STRAIGHT_CURVATURE_MAX = 2.5e-4
LANE_CENTERING_ASSIST_STRAIGHT_LATERAL_DEADBAND = 0.06
LANE_CENTERING_ASSIST_STRAIGHT_GROWTH_DEADBAND = 0.035
LANE_CENTERING_ASSIST_STRAIGHT_BUILD_RATE = 0.00008


@dataclass(frozen=True)
class LaneCenteringAssistInputs:
  lat_active: bool
  v_ego: float
  measured_curvature: float
  model_curvature: float
  previous_processed_curvature: float
  path_quality: float
  path_reason: str
  lane_change_shaping_active: bool
  lane_change_blend: float
  curvature_limited: bool
  steering_pressed: bool
  left_blinker: bool
  right_blinker: bool
  position_x: Sequence[float]
  position_y: Sequence[float]
  orientation_z: Sequence[float]
  lane_line_probs: Sequence[float]
  demand_source: str = DEMAND_SOURCE_MODEL_PATH


@dataclass(frozen=True)
class LaneCenteringAssistResult:
  active: bool
  curvature_nudge: float
  lateral_error: float
  heading_error: float
  predicted_lateral_error: float
  confidence: float
  reason: str
  debug: dict[str, float | str | bool] = field(default_factory=dict)


def inactive_lane_centering_assist_result(reason: str = "inactive") -> LaneCenteringAssistResult:
  return LaneCenteringAssistResult(False, 0.0, 0.0, 0.0, 0.0, 0.0, reason, _debug(reason=reason))


class LaneCenteringAssistTracker:
  def __init__(self) -> None:
    self.reset()

  def reset(self) -> None:
    self._filtered_nudge = 0.0
    self._active_sign = 0
    self._reason_cooldown_ticks = 0

  def update(self, inputs: LaneCenteringAssistInputs, dt: float) -> LaneCenteringAssistResult:
    dt = max(_finite_float(dt), 0.0)
    metrics = _lane_centering_metrics(inputs)
    if metrics is None:
      return self._hard_block("invalid_path")

    lateral_error, heading_error, predicted_lateral_error = metrics
    if inputs.path_reason != LANE_CENTERING_ASSIST_OK_REASON:
      self._reason_cooldown_ticks = LANE_CENTERING_ASSIST_PATH_REASON_COOLDOWN_FRAMES

    gate_reason = _gate_reason(inputs)
    if gate_reason is not None:
      return self._hard_block(gate_reason, lateral_error, heading_error, predicted_lateral_error)

    if self._reason_cooldown_ticks > 0:
      self._reason_cooldown_ticks -= 1
      return self._release(LANE_CENTERING_ASSIST_PATH_REASON_COOLDOWN_REASON, dt, lateral_error, heading_error, predicted_lateral_error)

    straight_cruise = _straight_cruise(inputs)
    lateral_deadband = _lateral_deadband(straight_cruise)
    growth_deadband = _growth_deadband(straight_cruise)
    error_sign = _sign(predicted_lateral_error, lateral_deadband)
    now_sign = _sign(lateral_error, lateral_deadband)
    if error_sign == 0:
      error_sign = now_sign
    same_direction = error_sign != 0 and (now_sign == 0 or now_sign == error_sign)
    error_growth = abs(predicted_lateral_error) - abs(lateral_error)
    growing = same_direction and error_growth > growth_deadband
    if not growing:
      return self._release("error_not_growing", dt, lateral_error, heading_error, predicted_lateral_error)

    confidence = _confidence(inputs)
    raw_nudge = confidence * (
      LANE_CENTERING_ASSIST_LATERAL_GAIN * lateral_error +
      LANE_CENTERING_ASSIST_HEADING_GAIN * heading_error +
      LANE_CENTERING_ASSIST_GROWTH_GAIN * (predicted_lateral_error - lateral_error)
    )
    max_nudge = _max_nudge_curvature(inputs.v_ego, straight_cruise)
    target_nudge = float(np.clip(raw_nudge, -max_nudge, max_nudge))
    target_sign = _sign(target_nudge, LANE_CENTERING_ASSIST_SIGN_HYSTERESIS_NUDGE)
    current_sign = _sign(self._filtered_nudge, LANE_CENTERING_ASSIST_SIGN_HYSTERESIS_NUDGE)
    if current_sign != 0 and target_sign != 0 and current_sign != target_sign:
      nudge = _approach(self._filtered_nudge, 0.0, LANE_CENTERING_ASSIST_RELEASE_RATE * dt)
      self._filtered_nudge = nudge
      if abs(nudge) <= LANE_CENTERING_ASSIST_SIGN_HYSTERESIS_NUDGE:
        self._filtered_nudge = 0.0
        self._active_sign = 0
      return LaneCenteringAssistResult(
        abs(self._filtered_nudge) > 0.0, self._filtered_nudge, lateral_error, heading_error, predicted_lateral_error,
        confidence, "sign_hysteresis", _debug(inputs, lateral_error, heading_error, predicted_lateral_error,
                                               confidence, raw_nudge, target_nudge, self._filtered_nudge,
                                               "sign_hysteresis", max_nudge, straight_cruise),
      )

    build_rate = _build_rate(straight_cruise)
    nudge = _approach(self._filtered_nudge, target_nudge, build_rate * dt)
    self._filtered_nudge = nudge
    self._active_sign = _sign(nudge, LANE_CENTERING_ASSIST_SIGN_HYSTERESIS_NUDGE)
    active = abs(nudge) > 0.0
    reason = "growing_lateral_error" if active else "below_deadband"
    return LaneCenteringAssistResult(
      active, nudge, lateral_error, heading_error, predicted_lateral_error, confidence, reason,
      _debug(inputs, lateral_error, heading_error, predicted_lateral_error, confidence, raw_nudge, target_nudge, nudge, reason,
             max_nudge, straight_cruise),
    )

  def _release(self, reason: str, dt: float, lateral_error: float = 0.0, heading_error: float = 0.0,
               predicted_lateral_error: float = 0.0) -> LaneCenteringAssistResult:
    self._filtered_nudge = _approach(self._filtered_nudge, 0.0, LANE_CENTERING_ASSIST_RELEASE_RATE * dt)
    if abs(self._filtered_nudge) <= LANE_CENTERING_ASSIST_SIGN_HYSTERESIS_NUDGE:
      self._filtered_nudge = 0.0
      self._active_sign = 0
    return LaneCenteringAssistResult(
      abs(self._filtered_nudge) > 0.0, self._filtered_nudge, lateral_error, heading_error, predicted_lateral_error, 0.0, reason,
      _debug(reason=reason, lateral_error=lateral_error, heading_error=heading_error,
             predicted_lateral_error=predicted_lateral_error, filtered_nudge=self._filtered_nudge),
    )

  def _hard_block(self, reason: str, lateral_error: float = 0.0, heading_error: float = 0.0,
                  predicted_lateral_error: float = 0.0) -> LaneCenteringAssistResult:
    self._filtered_nudge = 0.0
    self._active_sign = 0
    return LaneCenteringAssistResult(
      False, 0.0, lateral_error, heading_error, predicted_lateral_error, 0.0, reason,
      _debug(reason=reason, lateral_error=lateral_error, heading_error=heading_error,
             predicted_lateral_error=predicted_lateral_error, filtered_nudge=0.0),
    )


def _gate_reason(inputs: LaneCenteringAssistInputs) -> str | None:
  if not inputs.lat_active:
    return "inactive"
  if not _finite(inputs.v_ego, inputs.model_curvature, inputs.measured_curvature, inputs.previous_processed_curvature):
    return "nonfinite"
  if inputs.v_ego < LANE_CENTERING_ASSIST_MIN_SPEED:
    return "low_speed"
  if inputs.demand_source != DEMAND_SOURCE_MODEL_PATH:
    return "non_model_demand"
  if inputs.steering_pressed:
    return "driver_steering"
  lane_change_blend = _finite_optional_float(inputs.lane_change_blend)
  if inputs.left_blinker or inputs.right_blinker or inputs.lane_change_shaping_active or lane_change_blend is None or abs(lane_change_blend) > 1e-3:
    return "lane_change"
  if inputs.curvature_limited:
    return "curvature_limited"
  if inputs.path_reason != LANE_CENTERING_ASSIST_OK_REASON:
    return "path_reason"
  if not _finite(inputs.path_quality) or float(inputs.path_quality) < LANE_CENTERING_ASSIST_MIN_PATH_QUALITY:
    return "low_path_quality"
  return None


def _lane_centering_metrics(inputs: LaneCenteringAssistInputs) -> tuple[float, float, float] | None:
  xs = _finite_array(inputs.position_x)
  ys = _finite_array(inputs.position_y)
  headings = _finite_array(inputs.orientation_z)
  if len(xs) < 2 or len(ys) != len(xs) or len(headings) != len(xs):
    return None
  if xs[-1] <= xs[0]:
    return None
  near_x = min(max(inputs.v_ego * LANE_CENTERING_ASSIST_NEAR_LOOKAHEAD_T, LANE_CENTERING_ASSIST_NEAR_LOOKAHEAD_MIN_M), xs[-1])
  preview_x = min(max(inputs.v_ego * LANE_CENTERING_ASSIST_PREVIEW_T, LANE_CENTERING_ASSIST_PREVIEW_MIN_M), xs[-1])
  lateral_error = float(np.interp(near_x, xs, ys))
  predicted_lateral_error = float(np.interp(preview_x, xs, ys))
  heading_error = float(np.interp(near_x, xs, headings))
  return lateral_error, heading_error, predicted_lateral_error


def _confidence(inputs: LaneCenteringAssistInputs) -> float:
  lane_probs = [_finite_float(prob) for prob in inputs.lane_line_probs]
  lane_probs = [prob for prob in lane_probs if math.isfinite(prob)]
  lane_confidence = min(lane_probs[1], lane_probs[2]) if len(lane_probs) >= 3 else 0.0
  path_confidence = float(np.clip((float(inputs.path_quality) - LANE_CENTERING_ASSIST_MIN_PATH_QUALITY) / 0.15, 0.0, 1.0))
  lane_confidence = float(np.clip((lane_confidence - 0.5) / 0.4, 0.0, 1.0))
  return min(path_confidence, lane_confidence)


def _max_nudge_curvature(v_ego: float, straight_cruise: bool = False) -> float:
  speed = max(abs(_finite_float(v_ego)), LANE_CENTERING_ASSIST_SPEED_FLOOR)
  max_lat_accel = LANE_CENTERING_ASSIST_MAX_LAT_ACCEL
  if straight_cruise:
    max_lat_accel = min(
      max_lat_accel,
      float(np.interp(speed, LANE_CENTERING_ASSIST_MAX_LAT_ACCEL_BP, LANE_CENTERING_ASSIST_MAX_LAT_ACCEL_V)),
    )
  return max_lat_accel / speed**2


def _straight_cruise(inputs: LaneCenteringAssistInputs) -> bool:
  return bool(
    inputs.v_ego >= LANE_CENTERING_ASSIST_STRAIGHT_MIN_SPEED and
    max(abs(inputs.model_curvature), abs(inputs.measured_curvature), abs(inputs.previous_processed_curvature)) <=
    LANE_CENTERING_ASSIST_STRAIGHT_CURVATURE_MAX
  )


def _lateral_deadband(straight_cruise: bool) -> float:
  return LANE_CENTERING_ASSIST_STRAIGHT_LATERAL_DEADBAND if straight_cruise else LANE_CENTERING_ASSIST_LATERAL_DEADBAND


def _growth_deadband(straight_cruise: bool) -> float:
  return LANE_CENTERING_ASSIST_STRAIGHT_GROWTH_DEADBAND if straight_cruise else LANE_CENTERING_ASSIST_GROWTH_DEADBAND


def _build_rate(straight_cruise: bool) -> float:
  return LANE_CENTERING_ASSIST_STRAIGHT_BUILD_RATE if straight_cruise else LANE_CENTERING_ASSIST_BUILD_RATE


def _finite_array(values: Sequence[float]) -> list[float]:
  result = []
  for value in values:
    try:
      finite = float(value)
    except (TypeError, ValueError):
      return []
    if not math.isfinite(finite):
      return []
    result.append(finite)
  return result


def _finite_float(value) -> float:
  try:
    result = float(value)
  except (TypeError, ValueError):
    return 0.0
  return result if math.isfinite(result) else 0.0


def _finite_optional_float(value) -> float | None:
  try:
    result = float(value)
  except (TypeError, ValueError):
    return None
  return result if math.isfinite(result) else None


def _finite(*values: float) -> bool:
  try:
    return all(math.isfinite(float(value)) for value in values)
  except (TypeError, ValueError):
    return False


def _sign(value: float, threshold: float = 0.0) -> int:
  return 1 if value > threshold else (-1 if value < -threshold else 0)


def _approach(value: float, target: float, step: float) -> float:
  step = max(step, 0.0)
  if target > value:
    return min(target, value + step)
  return max(target, value - step)


def _debug(inputs: LaneCenteringAssistInputs | None = None, lateral_error: float = 0.0, heading_error: float = 0.0,
           predicted_lateral_error: float = 0.0, confidence: float = 0.0, raw_nudge: float = 0.0,
           target_nudge: float = 0.0, filtered_nudge: float = 0.0, reason: str = "inactive",
           max_nudge: float = 0.0, straight_cruise: bool = False) -> dict[str, float | str | bool]:
  return {
    "lane_centering_assist_active": abs(filtered_nudge) > 0.0,
    "lane_centering_reason": reason,
    "lane_centering_lateral_error": lateral_error,
    "lane_centering_heading_error": heading_error,
    "lane_centering_predicted_error": predicted_lateral_error,
    "lane_centering_confidence": confidence,
    "lane_centering_raw_nudge": raw_nudge,
    "lane_centering_target_nudge": target_nudge,
    "lane_centering_curvature_nudge": filtered_nudge,
    "lane_centering_max_nudge": max_nudge,
    "lane_centering_straight_cruise": straight_cruise,
    "lane_centering_v_ego": float(inputs.v_ego) if inputs is not None else 0.0,
  }
