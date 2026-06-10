from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from openpilot.selfdrive.controls.lib.lateral_demand import (
  DEMAND_SOURCE_MODEL_PATH,
  ProcessedLateralDemand,
)


LOW_QUALITY_PATH_THRESHOLD = 0.5
STEADY_CURVE_MIN_LAT_ACCEL = 0.10
TURN_IN_MIN_LAT_ACCEL = 0.10
TURN_IN_MIN_ABS_TARGET_RATE = 0.15
TURN_EXIT_MAX_ABS_TARGET = 0.85
TURN_EXIT_COLLAPSE_PER_FRAME = 0.015
STRAIGHT_STABLE_MAX_LAT_ACCEL = 0.05
LANE_CHANGE_BLEND_EPSILON = 1e-3


class LateralMode(str, Enum):
  STRAIGHT_STABLE = "straight_stable"
  STRAIGHT_BIAS_CORRECTION = "straight_bias_correction"
  TURN_IN = "turn_in"
  STEADY_CURVE = "steady_curve"
  TURN_EXIT_RECENTER = "turn_exit_recenter"
  LANE_CHANGE = "lane_change"
  LOW_QUALITY_PATH = "low_quality_path"
  SAFETY_LIMITED = "safety_limited"
  DRIVER_OVERRIDE = "driver_override"


LATERAL_MODE_TO_UINT8 = {
  LateralMode.STRAIGHT_STABLE.value: 1,
  LateralMode.STRAIGHT_BIAS_CORRECTION.value: 2,
  LateralMode.TURN_IN.value: 3,
  LateralMode.STEADY_CURVE.value: 4,
  LateralMode.TURN_EXIT_RECENTER.value: 5,
  LateralMode.LANE_CHANGE.value: 6,
  LateralMode.LOW_QUALITY_PATH.value: 7,
  LateralMode.SAFETY_LIMITED.value: 8,
  LateralMode.DRIVER_OVERRIDE.value: 9,
}

LATERAL_UINT8_TO_MODE = {value: key for key, value in LATERAL_MODE_TO_UINT8.items()}


def lateral_mode_to_uint8(mode: str) -> int:
  return LATERAL_MODE_TO_UINT8.get(mode, 0)


def uint8_to_lateral_mode(value: int) -> str:
  return LATERAL_UINT8_TO_MODE.get(int(value), LateralMode.STRAIGHT_STABLE.value)


@dataclass(frozen=True)
class LateralDemandProfile:
  raw_curvature: float
  processed_curvature: float
  curvature_limited: bool
  path_quality: float
  path_reason: str
  lane_change_shaping_active: bool
  lane_change_blend: float
  demand_source: str
  straight_road_damping_active: bool = False
  desired_lateral_accel: float = 0.0
  desired_lateral_jerk: float = 0.0
  preview_lateral_accel_0_2s: float = 0.0
  preview_lateral_accel_0_5s: float = 0.0
  preview_lateral_accel_1_0s: float = 0.0
  mode: str = LateralMode.STRAIGHT_STABLE.value
  mode_confidence: float = 0.0


@dataclass
class LateralDemandProfileBuilder:
  dt: float = 0.05

  def __post_init__(self) -> None:
    self._previous_target_lateral_accel: float = 0.0
    self._initialised: bool = False

  def reset(self) -> None:
    self._previous_target_lateral_accel = 0.0
    self._initialised = False

  def update(
    self,
    demand: ProcessedLateralDemand,
    v_ego: float,
    *,
    curvature_limited: bool = False,
    saturated: bool = False,
    steer_limited_by_safety: bool = False,
    steering_pressed: bool = False,
  ) -> LateralDemandProfile:
    v_ego = _safe_float(v_ego, 0.0)

    target = _safe_float(demand.processed_curvature, 0.0) * v_ego * v_ego
    if not self._initialised:
      self._previous_target_lateral_accel = target
      self._initialised = True
    previous_target = self._previous_target_lateral_accel
    target_rate = (target - previous_target) / max(float(self.dt), 1e-3)
    self._previous_target_lateral_accel = target

    mode, confidence = _classify_lateral_mode(
      target=target,
      previous_target=previous_target,
      target_rate=target_rate,
      path_quality=_safe_float(demand.path_quality, 1.0),
      path_reason=demand.path_reason or "",
      lane_change_shaping_active=bool(demand.lane_change_shaping_active),
      lane_change_blend=_safe_float(demand.lane_change_blend, 0.0),
      curvature_limited=bool(curvature_limited),
      saturated=bool(saturated),
      steer_limited_by_safety=bool(steer_limited_by_safety),
      steering_pressed=bool(steering_pressed),
      v_ego=v_ego,
    )

    return LateralDemandProfile(
      raw_curvature=_safe_float(demand.raw_curvature, 0.0),
      processed_curvature=_safe_float(demand.processed_curvature, 0.0),
      curvature_limited=bool(demand.curvature_limited),
      path_quality=_safe_float(demand.path_quality, 1.0),
      path_reason=demand.path_reason or "",
      lane_change_shaping_active=bool(demand.lane_change_shaping_active),
      lane_change_blend=_safe_float(demand.lane_change_blend, 0.0),
      demand_source=demand.demand_source or DEMAND_SOURCE_MODEL_PATH,
      straight_road_damping_active=False,
      desired_lateral_accel=target,
      desired_lateral_jerk=target_rate,
      preview_lateral_accel_0_2s=target + target_rate * 0.2,
      preview_lateral_accel_0_5s=target + target_rate * 0.5,
      preview_lateral_accel_1_0s=target + target_rate * 1.0,
      mode=mode,
      mode_confidence=confidence,
    )


def _classify_lateral_mode(
  *,
  target: float,
  previous_target: float,
  target_rate: float,
  path_quality: float,
  path_reason: str,
  lane_change_shaping_active: bool,
  lane_change_blend: float,
  curvature_limited: bool,
  saturated: bool,
  steer_limited_by_safety: bool,
  steering_pressed: bool,
  v_ego: float,
) -> tuple[str, float]:
  if steering_pressed:
    return LateralMode.DRIVER_OVERRIDE.value, 1.0
  if curvature_limited or saturated or steer_limited_by_safety:
    return LateralMode.SAFETY_LIMITED.value, 1.0
  if lane_change_shaping_active or abs(lane_change_blend) > LANE_CHANGE_BLEND_EPSILON:
    return LateralMode.LANE_CHANGE.value, 1.0
  if path_quality < LOW_QUALITY_PATH_THRESHOLD or (path_reason and path_reason != "ok"):
    return LateralMode.LOW_QUALITY_PATH.value, 1.0
  target_decreasing_to_zero = abs(target) < abs(previous_target) and target != 0.0
  collapse_per_frame = abs(previous_target) - abs(target)
  if (
    target_decreasing_to_zero
    and _signs_stable(target, previous_target)
    and abs(target) < TURN_EXIT_MAX_ABS_TARGET
    and collapse_per_frame > TURN_EXIT_COLLAPSE_PER_FRAME
  ):
    return LateralMode.TURN_EXIT_RECENTER.value, 0.9
  if abs(target) > TURN_IN_MIN_LAT_ACCEL and abs(target_rate) > TURN_IN_MIN_ABS_TARGET_RATE:
    return LateralMode.TURN_IN.value, 0.9
  if abs(target) > STEADY_CURVE_MIN_LAT_ACCEL:
    return LateralMode.STEADY_CURVE.value, 0.8
  return LateralMode.STRAIGHT_STABLE.value, 0.8


def _signs_stable(a: float, b: float) -> bool:
  if a == 0.0 and b == 0.0:
    return True
  if a == 0.0 or b == 0.0:
    return False
  return (a > 0.0) == (b > 0.0)


def _safe_float(value, default: float) -> float:
  try:
    result = float(value)
  except (TypeError, ValueError):
    return float(default)
  if not math.isfinite(result):
    return float(default)
  return result
