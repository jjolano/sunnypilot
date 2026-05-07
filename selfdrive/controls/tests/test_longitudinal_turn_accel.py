import math
from types import SimpleNamespace

import pytest

from openpilot.selfdrive.controls.lib.longitudinal_planner import limit_accel_in_turns


class FakeVehicleModel:
  def __init__(self, curvature):
    self.curvature = curvature
    self.calls = []

  def calc_curvature(self, steering_angle_rad, v_ego, roll):
    self.calls.append((steering_angle_rad, v_ego, roll))
    return self.curvature


def test_turn_accel_limit_preserves_legacy_steering_angle_math_by_default():
  cp = SimpleNamespace(steerRatio=15.0, wheelbase=2.7)
  a_target = [-1.0, 1.5]
  vehicle_model = FakeVehicleModel(curvature=0.004)

  limited = limit_accel_in_turns(20.0, 0.0, a_target, cp, vehicle_model=vehicle_model)

  assert limited == a_target
  assert vehicle_model.calls == []


def test_turn_accel_limit_uses_vehicle_model_when_accurate_lateral_accel_enabled():
  cp = SimpleNamespace(steerRatio=15.0, wheelbase=2.7)
  a_target = [-1.0, 1.5]
  vehicle_model = FakeVehicleModel(curvature=0.004)
  steering_angle_deg = 2.0
  roll = 0.0

  limited = limit_accel_in_turns(
    20.0,
    steering_angle_deg,
    a_target,
    cp,
    vehicle_model=vehicle_model,
    roll=roll,
    accurate_lateral_accel=True,
  )

  assert limited[0] == a_target[0]
  assert limited[1] == pytest.approx(math.sqrt(1.7**2 - 1.6**2))
  assert vehicle_model.calls == [(math.radians(steering_angle_deg), 20.0, roll)]
