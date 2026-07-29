"""Shadow-only classifier for near-zero desired-lateral-accel recenter conflicts."""
from __future__ import annotations

import math
from dataclasses import dataclass


_NZ_V_EGO_MIN = 10.0
_NZ_DESIRED_MAX = 0.07
_NZ_ACTUAL_MIN = 0.05
_NZ_ACTUAL_MAX = 0.15
_NZ_ERROR_MAX = 0.18
_NZ_STEERING_RATE_MAX = 20.0
_NZ_TORQUE_FRACTION = 0.5
_NZ_EPS = 1e-4


@dataclass(frozen=True)
class NearZeroRecenterDebug:
  conflict: bool = False
  error: float = 0.0
  closingRate: float = 0.0
  duration: float = 0.0


class NearZeroRecenterObserver:
  def __init__(self, dt: float):
    self.dt = max(float(dt), 1e-3)
    self._prev_abs_error: float | None = None
    self._duration = 0.0

  def reset(self) -> None:
    self._prev_abs_error = None
    self._duration = 0.0

  @staticmethod
  def _sign(value: float) -> int:
    if value > _NZ_EPS:
      return 1
    if value < -_NZ_EPS:
      return -1
    return 0

  @staticmethod
  def _finite(*values: float) -> bool:
    try:
      return all(math.isfinite(float(v)) for v in values)
    except (TypeError, ValueError):
      return False

  def update(self, *, active: bool, v_ego: float, steering_pressed: bool,
             steer_limited_by_safety: bool, curvature_limited: bool,
             desired_lateral_accel: float, actual_lateral_accel: float,
             steering_rate_deg: float, output_torque: float, steer_max: float) -> NearZeroRecenterDebug:
    if not active:
      self.reset()
      return NearZeroRecenterDebug()

    if not self._finite(v_ego, desired_lateral_accel, actual_lateral_accel, steering_rate_deg, output_torque, steer_max):
      self.reset()
      return NearZeroRecenterDebug()

    v = float(v_ego)
    desired = float(desired_lateral_accel)
    actual = float(actual_lateral_accel)
    rate = float(steering_rate_deg)
    torque = float(output_torque)
    smax = float(steer_max)
    if smax <= 0.0:
      self.reset()
      return NearZeroRecenterDebug()

    error = desired - actual
    desired_sign = self._sign(desired)
    actual_sign = self._sign(actual)
    torque_sign = self._sign(torque)
    error_sign = self._sign(error)

    gated = (
      not steering_pressed and
      not bool(steer_limited_by_safety) and
      not bool(curvature_limited) and
      v >= _NZ_V_EGO_MIN and
      abs(desired) <= _NZ_DESIRED_MAX and
      _NZ_ACTUAL_MIN < abs(actual) <= _NZ_ACTUAL_MAX and
      desired_sign != 0 and
      actual_sign != 0 and
      desired_sign != actual_sign and
      abs(error) <= _NZ_ERROR_MAX and
      abs(rate) <= _NZ_STEERING_RATE_MAX and
      abs(torque) < _NZ_TORQUE_FRACTION * smax and
      torque_sign == error_sign
    )
    if not gated:
      self.reset()
      return NearZeroRecenterDebug()

    abs_error = abs(error)
    closing_rate = 0.0 if self._prev_abs_error is None else (self._prev_abs_error - abs_error) / self.dt
    self._duration += self.dt
    self._prev_abs_error = abs_error
    return NearZeroRecenterDebug(True, float(error), float(closing_rate), float(self._duration))
