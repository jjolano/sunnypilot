from __future__ import annotations

from dataclasses import dataclass, field
import math


@dataclass(frozen=True)
class TorqueV4RouteFrame:
  target_lateral_accel: float
  actual_lateral_accel: float
  raw_torque: float
  governed_torque: float
  governor_reason: int
  same_direction_actuator_limited: bool = False
  driver_override: bool = False
  steering_pressed: bool = False
  v_ego: float = 0.0


@dataclass(frozen=True)
class TorqueV4RouteMetrics:
  target_lateral_accel_rms: float
  actual_lateral_accel_rms: float
  phase_lag_proxy: float
  overshoot: float
  wrong_sign_response_duration: float
  governor_reason_fraction: dict[int, float] = field(default_factory=dict)
  same_direction_actuator_limit_fraction: float = 0.0
  driver_override_recovery: float = 0.0
  straight_road_wander_energy: float = 0.0


def compute_torque_v4_route_metrics(frames: tuple[TorqueV4RouteFrame, ...] | list[TorqueV4RouteFrame], *,
                                    dt: float = 0.05) -> TorqueV4RouteMetrics:
  samples = [frame for frame in frames if _frame_finite(frame)]
  if not samples:
    return TorqueV4RouteMetrics(0.0, 0.0, 0.0, 0.0, 0.0, {}, 0.0, 0.0, 0.0)
  dt = float(dt) if _finite(dt) and dt > 0.0 else 0.05
  target_values = [frame.target_lateral_accel for frame in samples]
  actual_values = [frame.actual_lateral_accel for frame in samples]
  error_values = [target - actual for target, actual in zip(target_values, actual_values, strict=True)]
  wrong_sign = [
    frame for frame in samples
    if _sign(frame.target_lateral_accel, 0.05) != 0 and
    _sign(frame.actual_lateral_accel, 0.05) == -_sign(frame.target_lateral_accel, 0.05)
  ]
  reason_counts: dict[int, int] = {}
  for frame in samples:
    reason_counts[int(frame.governor_reason)] = reason_counts.get(int(frame.governor_reason), 0) + 1
  reason_fraction = {reason: count / len(samples) for reason, count in reason_counts.items()}
  straight = [frame for frame in samples if abs(frame.target_lateral_accel) <= 0.05 and frame.v_ego >= 5.0]
  return TorqueV4RouteMetrics(
    target_lateral_accel_rms=_rms(target_values),
    actual_lateral_accel_rms=_rms(actual_values),
    phase_lag_proxy=_phase_lag_proxy(target_values, actual_values, dt),
    overshoot=max((abs(actual) - abs(target) for target, actual in zip(target_values, actual_values, strict=True)), default=0.0),
    wrong_sign_response_duration=len(wrong_sign) * dt,
    governor_reason_fraction=reason_fraction,
    same_direction_actuator_limit_fraction=sum(frame.same_direction_actuator_limited for frame in samples) / len(samples),
    driver_override_recovery=_driver_override_recovery(samples, dt),
    straight_road_wander_energy=sum(frame.actual_lateral_accel ** 2 * dt for frame in straight),
  )


def _driver_override_recovery(samples: list[TorqueV4RouteFrame], dt: float) -> float:
  recovery_frames = 0
  recovering = False
  for frame in samples:
    if frame.driver_override or frame.steering_pressed:
      recovering = True
      recovery_frames = 0
      continue
    if recovering and abs(frame.actual_lateral_accel - frame.target_lateral_accel) > 0.1:
      recovery_frames += 1
      continue
    if recovering:
      return recovery_frames * dt
  return recovery_frames * dt if recovering else 0.0


def _phase_lag_proxy(target_values: list[float], actual_values: list[float], dt: float) -> float:
  if len(target_values) < 2:
    return 0.0
  target_peak = max(range(len(target_values)), key=lambda idx: abs(target_values[idx]))
  actual_peak = max(range(len(actual_values)), key=lambda idx: abs(actual_values[idx]))
  return max(0.0, (actual_peak - target_peak) * dt)


def _rms(values: list[float]) -> float:
  return math.sqrt(sum(value * value for value in values) / len(values)) if values else 0.0


def _sign(value: float, threshold: float = 0.0) -> int:
  return 1 if value > threshold else (-1 if value < -threshold else 0)


def _frame_finite(frame: TorqueV4RouteFrame) -> bool:
  return _finite(
    frame.target_lateral_accel,
    frame.actual_lateral_accel,
    frame.raw_torque,
    frame.governed_torque,
    frame.v_ego,
  )


def _finite(*values: float) -> bool:
  try:
    return all(math.isfinite(float(value)) for value in values)
  except (TypeError, ValueError):
    return False
