import math

import pytest

from openpilot.selfdrive.controls.lib.vehicle_math import (
  required_decel_to_target_speed,
  smooth_speed_floor,
  speed_for_lateral_accel,
  stopping_decel,
)


def test_smooth_speed_floor_uses_quadrature_floor():
  assert smooth_speed_floor(0.0, 1.0) == pytest.approx(1.0)
  assert smooth_speed_floor(3.0, 4.0) == pytest.approx(5.0)
  assert math.isfinite(smooth_speed_floor(-2.0, 1.0))


def test_smooth_speed_floor_is_continuous_around_zero():
  floor = 1.0
  left = smooth_speed_floor(-1e-6, floor)
  center = smooth_speed_floor(0.0, floor)
  right = smooth_speed_floor(1e-6, floor)

  assert left == pytest.approx(center, abs=1e-9)
  assert right == pytest.approx(center, abs=1e-9)


def test_required_decel_to_target_speed_sign_convention():
  assert required_decel_to_target_speed(20.0, 10.0, 100.0) < 0.0
  assert required_decel_to_target_speed(10.0, 20.0, 100.0) > 0.0
  assert required_decel_to_target_speed(10.0, 10.0, 100.0) == pytest.approx(0.0)


def test_stopping_decel_matches_target_speed_zero_case():
  assert stopping_decel(12.0, 40.0) == pytest.approx(required_decel_to_target_speed(12.0, 0.0, 40.0))


def test_stopping_decel_gets_stronger_with_speed():
  slow = stopping_decel(5.0, 30.0)
  fast = stopping_decel(10.0, 30.0)

  assert fast < slow < 0.0


def test_stopping_decel_gets_weaker_with_distance():
  close = stopping_decel(10.0, 10.0)
  far = stopping_decel(10.0, 40.0)

  assert close < far < 0.0


def test_required_decel_uses_min_distance_floor():
  assert stopping_decel(10.0, 0.0, min_distance=1.0) == pytest.approx(stopping_decel(10.0, 1.0, min_distance=1.0))


def test_required_decel_outputs_are_finite_for_finite_inputs():
  values = (
    smooth_speed_floor(12.0, 1.0),
    required_decel_to_target_speed(20.0, 10.0, 100.0),
    stopping_decel(20.0, 100.0),
    speed_for_lateral_accel(2.0, 0.01),
  )

  assert all(math.isfinite(value) for value in values)


def test_speed_for_lateral_accel_uses_curvature_magnitude():
  assert speed_for_lateral_accel(2.0, -0.02) == pytest.approx(speed_for_lateral_accel(2.0, 0.02))
  assert speed_for_lateral_accel(2.0, -0.02) == pytest.approx(10.0)


@pytest.mark.parametrize("lateral_accel,curvature", [
  (2.0, 0.0),
  (2.0, 1e-12),
  (math.nan, 0.01),
  (-1.0, 0.01),
  (math.inf, 0.01),
])
def test_speed_for_lateral_accel_invalid_inputs_return_inf(lateral_accel, curvature):
  assert math.isinf(speed_for_lateral_accel(lateral_accel, curvature))


def test_speed_for_lateral_accel_never_returns_nan():
  cases = ((math.nan, 0.1), (2.0, math.nan), (-1.0, 0.1), (2.0, 0.0))

  for lateral_accel, curvature in cases:
    assert not math.isnan(speed_for_lateral_accel(lateral_accel, curvature))
