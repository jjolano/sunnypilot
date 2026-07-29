"""Property tests for jerk-limited trajectory synthesis."""
from __future__ import annotations

import math

import numpy as np

from openpilot.sunnypilot.custom.longitudinal.trajectory import (
  NORMAL_NEGATIVE_RETREAT_JERK,
  POSITIVE_PROGRESS_JERK,
  preserve_seed_trajectory,
  synth_trajectory,
  synth_trajectory_dts,
)


def test_dts_are_positive_and_control_n_long():
  dts = synth_trajectory_dts()
  assert len(dts) == 17  # CONTROL_N
  assert all(math.isfinite(dt) and dt > 0 for dt in dts)


def test_jerk_limited_respects_asymmetric_budget():
  dts = synth_trajectory_dts()
  # large positive target step from rest accel -> rise bounded by POSITIVE_PROGRESS_JERK
  speeds, accels, jerks = synth_trajectory([10.0], [0.0], 10.0, a_target=3.0, limit_jerk=True)
  for j in jerks:
    assert NORMAL_NEGATIVE_RETREAT_JERK - 1e-9 <= j <= POSITIVE_PROGRESS_JERK + 1e-9
  # first accel step bounded by jerk * dt
  assert accels[0] <= POSITIVE_PROGRESS_JERK * dts[0] + 1e-9


def test_negative_retreat_uses_steeper_budget():
  _, accels, jerks = synth_trajectory([20.0], [0.0], 20.0, a_target=-4.0, limit_jerk=True)
  assert min(jerks) >= NORMAL_NEGATIVE_RETREAT_JERK - 1e-9
  assert accels[-1] < 0.0  # converges toward the decel target


def test_unlimited_jumps_to_target():
  _, accels, _ = synth_trajectory([15.0], [0.0], 15.0, a_target=2.0, limit_jerk=False)
  assert all(a == 2.0 for a in accels)


def test_speeds_never_negative():
  speeds, _, _ = synth_trajectory([2.0], [-3.0], 2.0, a_target=-3.0, limit_jerk=True)
  assert all(s >= 0.0 for s in speeds)


def test_speed_integrates_accel_over_grid():
  v0 = 12.0
  dts = synth_trajectory_dts()
  speeds, accels, _ = synth_trajectory([v0], [0.0], v0, a_target=1.0, limit_jerk=True)
  expected = max(0.0, v0)
  for i, (s, a, dt) in enumerate(zip(speeds, accels, dts)):
    assert s == np.float64(expected) or abs(s - expected) < 1e-9, f"step {i}"
    expected = max(0.0, expected + a * dt)


def test_preserve_seed_trajectory():
  # seed scalar present -> never preserve
  assert preserve_seed_trajectory(1.0, planner_seed_scalar=True, a_target=1.0) is False
  # a_target unchanged by stack -> preserve
  assert preserve_seed_trajectory(1.0, planner_seed_scalar=False, a_target=1.0) is True
  # a_target overridden -> do not preserve
  assert preserve_seed_trajectory(1.0, planner_seed_scalar=False, a_target=-2.0) is False
