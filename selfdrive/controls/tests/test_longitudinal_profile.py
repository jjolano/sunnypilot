import math

import pytest

from openpilot.selfdrive.controls.lib.longitudinal_profile import (
  jerk_limited_braking_profile,
  required_decel_to_target_speed,
  speed_reachable_with_profile,
  stopping_distance_with_jerk_limit,
)


def test_required_decel_monotonic_by_distance():
  near = required_decel_to_target_speed(20.0, 10.0, 40.0)
  far = required_decel_to_target_speed(20.0, 10.0, 80.0)

  assert near < far < 0.0


def test_required_decel_monotonic_by_initial_speed():
  slower = required_decel_to_target_speed(15.0, 10.0, 60.0)
  faster = required_decel_to_target_speed(25.0, 10.0, 60.0)

  assert faster < slower < 0.0


def test_required_decel_monotonic_by_target_speed():
  lower_target = required_decel_to_target_speed(20.0, 5.0, 60.0)
  higher_target = required_decel_to_target_speed(20.0, 12.0, 60.0)

  assert lower_target < higher_target < 0.0


def test_jerk_limit_affects_profile_smoothly():
  gentle_jerk = jerk_limited_braking_profile(20.0, 0.0, 90.0, jerk_limit=0.8)
  sharp_jerk = jerk_limited_braking_profile(20.0, 0.0, 90.0, jerk_limit=4.0)

  assert gentle_jerk.required_accel == pytest.approx(sharp_jerk.required_accel)
  assert abs(gentle_jerk.required_jerk) == pytest.approx(abs(sharp_jerk.required_jerk))
  assert gentle_jerk.stopping_distance > sharp_jerk.stopping_distance


@pytest.mark.parametrize(("v_initial", "target_speed", "distance"), [
  (math.nan, 0.0, 30.0),
  (20.0, math.inf, 30.0),
  (20.0, 0.0, math.nan),
  (20.0, 0.0, -1.0),
])
def test_invalid_inputs_return_finite_safe_profile(v_initial, target_speed, distance):
  profile = jerk_limited_braking_profile(v_initial, target_speed, distance)

  assert profile.finite
  assert math.isfinite(profile.required_accel)
  assert math.isfinite(profile.required_jerk)
  assert math.isfinite(profile.stopping_distance)


def test_urgent_flag_set_for_short_runway_cases():
  short = jerk_limited_braking_profile(25.0, 0.0, 20.0)
  long = jerk_limited_braking_profile(25.0, 0.0, 160.0)

  assert short.urgent
  assert not long.urgent
  assert speed_reachable_with_profile(25.0, 0.0, 20.0) is False


def test_stopping_distance_with_jerk_limit_is_finite_and_positive():
  distance = stopping_distance_with_jerk_limit(15.0, target_speed=0.0, jerk_limit=1.5)

  assert math.isfinite(distance)
  assert distance > 0.0
