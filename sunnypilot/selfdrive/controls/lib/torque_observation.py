from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class TorqueObservation:
  active: bool
  v_ego: float
  steering_pressed: bool
  steer_limited_by_safety: bool
  curvature_limited: bool
  saturated: bool
  lateral_maneuver: bool
  target_lateral_accel: float
  target_lateral_accel_rate: float
  actual_lateral_accel: float
  actual_lateral_jerk: float

  @property
  def finite(self) -> bool:
    return all(math.isfinite(float(value)) for value in (
      self.v_ego,
      self.target_lateral_accel,
      self.target_lateral_accel_rate,
      self.actual_lateral_accel,
      self.actual_lateral_jerk,
    ))


def torque_direction(value: float, threshold: float = 0.0) -> int:
  return 1 if value > threshold else (-1 if value < -threshold else 0)
