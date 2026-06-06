from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class BrakingProfile:
  target_speed: float
  distance: float
  required_accel: float
  required_jerk: float
  stopping_distance: float
  comfortable: bool
  urgent: bool
  finite: bool
  input_valid: bool
  reason: str


def required_decel_to_target_speed(v_initial: float, target_speed: float, distance: float,
                                   max_decel: float = -3.5) -> float:
  v_initial = _safe_nonnegative(v_initial)
  target_speed = min(_safe_nonnegative(target_speed), v_initial)
  distance = _safe_distance(distance)
  if distance <= 0.0:
    return float(max_decel)
  return max(float(max_decel), (target_speed * target_speed - v_initial * v_initial) / (2.0 * distance))


def jerk_limited_braking_profile(v_initial: float, target_speed: float, distance: float, *,
                                 current_accel: float = 0.0, comfortable_decel: float = -1.5,
                                 urgent_decel: float = -2.5, max_decel: float = -3.5,
                                 jerk_limit: float = 2.0) -> BrakingProfile:
  finite_input = _finite(v_initial) and _finite(target_speed) and _finite(distance) and _finite(current_accel)
  if not finite_input:
    reason = "invalid_input"
  elif distance <= 0.0:
    reason = "invalid_distance"
  elif target_speed >= max(0.0, v_initial):
    reason = "target_not_slower"
  else:
    reason = "braking_profile"

  v_initial = _safe_nonnegative(v_initial)
  target_speed = min(_safe_nonnegative(target_speed), v_initial)
  distance = max(0.0, _safe_nonnegative(distance))
  current_accel = float(current_accel) if _finite(current_accel) else 0.0
  required_accel = required_decel_to_target_speed(v_initial, target_speed, distance, max_decel=max_decel)
  duration = _time_to_target(v_initial, target_speed, distance)
  required_jerk = (required_accel - current_accel) / max(duration, 0.1)
  stopping_distance = stopping_distance_with_jerk_limit(
    v_initial,
    target_speed=target_speed,
    current_accel=current_accel,
    max_decel=max_decel,
    jerk_limit=jerk_limit,
  )
  comfortable = bool(required_accel >= comfortable_decel and abs(required_jerk) <= abs(jerk_limit))
  urgent = bool(required_accel <= urgent_decel or distance < stopping_distance)
  return BrakingProfile(
    target_speed=float(target_speed),
    distance=float(distance),
    required_accel=float(required_accel),
    required_jerk=float(required_jerk) if math.isfinite(required_jerk) else 0.0,
    stopping_distance=float(stopping_distance),
    comfortable=comfortable,
    urgent=urgent,
    finite=True,
    input_valid=finite_input and distance > 0.0,
    reason=reason,
  )


def stopping_distance_with_jerk_limit(v_initial: float, *, target_speed: float = 0.0,
                                      current_accel: float = 0.0, max_decel: float = -3.5,
                                      jerk_limit: float = 2.0) -> float:
  v_initial = _safe_nonnegative(v_initial)
  target_speed = min(_safe_nonnegative(target_speed), v_initial)
  current_accel = float(current_accel) if _finite(current_accel) else 0.0
  max_decel = min(-0.1, float(max_decel) if _finite(max_decel) else -3.5)
  jerk = -max(0.1, abs(float(jerk_limit) if _finite(jerk_limit) else 2.0))

  accel_delta = max_decel - current_accel
  t_ramp = max(0.0, accel_delta / jerk)
  v_after_ramp = max(target_speed, v_initial + current_accel * t_ramp + 0.5 * jerk * t_ramp * t_ramp)
  d_ramp = max(0.0, v_initial * t_ramp + 0.5 * current_accel * t_ramp * t_ramp + jerk * t_ramp ** 3 / 6.0)
  if v_after_ramp <= target_speed:
    return d_ramp
  d_const = (v_after_ramp * v_after_ramp - target_speed * target_speed) / (2.0 * abs(max_decel))
  return max(0.0, d_ramp + d_const)


def speed_reachable_with_profile(v_initial: float, target_speed: float, distance: float, *,
                                 max_decel: float = -3.5, jerk_limit: float = 2.0) -> bool:
  profile = jerk_limited_braking_profile(v_initial, target_speed, distance, max_decel=max_decel, jerk_limit=jerk_limit)
  return bool(profile.finite and not profile.urgent)


def _time_to_target(v_initial: float, target_speed: float, distance: float) -> float:
  average_speed = max(0.1, (max(0.0, v_initial) + max(0.0, target_speed)) * 0.5)
  return max(0.1, max(0.0, distance) / average_speed)


def _safe_nonnegative(value: float) -> float:
  try:
    value = float(value)
  except (TypeError, ValueError):
    return 0.0
  return max(0.0, value) if math.isfinite(value) else 0.0


def _safe_distance(value: float) -> float:
  try:
    value = float(value)
  except (TypeError, ValueError):
    return 0.0
  return value if math.isfinite(value) else 0.0


def _finite(value: float) -> bool:
  try:
    return math.isfinite(float(value))
  except (TypeError, ValueError):
    return False
