import math

import pytest

from openpilot.common.constants import ACCELERATION_DUE_TO_GRAVITY
from openpilot.selfdrive.controls.lib.lateral_accel import (
  lateral_accel_from_curvature,
  lateral_accel_from_steering_angle,
  roll_lateral_accel,
)


class FakeVehicleModel:
  def __init__(self, curvature):
    self.curvature = curvature
    self.calls = []

  def calc_curvature(self, steering_angle_rad, v_ego, roll):
    self.calls.append((steering_angle_rad, v_ego, roll))
    return self.curvature


def test_roll_lateral_accel_uses_exact_sine_compensation():
  roll = math.radians(30.0)

  assert roll_lateral_accel(roll) == pytest.approx(math.sin(roll) * ACCELERATION_DUE_TO_GRAVITY)
  assert roll_lateral_accel(roll) != pytest.approx(roll * ACCELERATION_DUE_TO_GRAVITY)


def test_lateral_accel_from_curvature_removes_roll_gravity():
  v_ego = 20.0
  curvature = 0.01
  roll = math.asin(0.25 / ACCELERATION_DUE_TO_GRAVITY)

  assert lateral_accel_from_curvature(v_ego, curvature, roll) == pytest.approx(3.75)


@pytest.mark.parametrize("roll", [math.radians(6.0), math.radians(-6.0)])
def test_lateral_accel_from_curvature_roll_sign_matches_roll_lateral_accel(roll):
  v_ego = 15.0
  curvature = 0.02
  base = curvature * v_ego**2

  assert lateral_accel_from_curvature(v_ego, curvature, roll) == pytest.approx(base - roll_lateral_accel(roll))


def test_lateral_accel_from_steering_angle_uses_vehicle_model_curvature():
  vehicle_model = FakeVehicleModel(curvature=0.02)
  v_ego = 15.0
  steering_angle_rad = math.radians(4.0)
  roll = math.asin(0.5 / ACCELERATION_DUE_TO_GRAVITY)

  assert lateral_accel_from_steering_angle(v_ego, steering_angle_rad, vehicle_model, roll) == pytest.approx(4.0)
  assert vehicle_model.calls == [(steering_angle_rad, v_ego, roll)]
