"""Tests for the lead-following cushion and lead-aware speedup guard."""
from __future__ import annotations

import pytest

from openpilot.sunnypilot.custom.longitudinal.coast_horizon import CoastAction
from openpilot.sunnypilot.custom.longitudinal.lead_cushion import (
  lead_catchup_accel_cap,
  lead_following_cushion,
  lead_speedup_guard,
)


def test_cushion_coasts_to_slower_lead_with_runway():
  # closing 5 m/s; bleeding 20->15 at -0.25 needs ~350 m, so put the lead far enough that the
  # runway lands in the lift window -> coast, no brake
  r = lead_following_cushion(v_ego=20.0, v_lead=15.0, d_rel=375.0, follow_gap=20.0, coast_decel=-0.25)
  assert r.coast_first is True
  assert -0.5 <= r.a_target <= 0.0  # gentle coast, never hard braking
  assert r.action in (CoastAction.COAST, CoastAction.CRUISE)


def test_cushion_brakes_when_runway_short():
  # closing hard with little runway -> must brake
  r = lead_following_cushion(v_ego=25.0, v_lead=10.0, d_rel=30.0, follow_gap=20.0, coast_decel=-0.25)
  assert r.coast_first is False
  assert r.action is CoastAction.BRAKE
  assert r.a_target < 0.0


def test_cushion_inactive_when_lead_not_slower():
  r = lead_following_cushion(v_ego=18.0, v_lead=18.0, d_rel=50.0, follow_gap=20.0, coast_decel=-0.25)
  assert r.coast_first is False
  assert r.action is CoastAction.CRUISE


def test_cushion_reacts_to_small_stable_speed_drop():
  r = lead_following_cushion(v_ego=18.0, v_lead=17.75, d_rel=70.0, follow_gap=27.0, coast_decel=-0.25)
  assert r.coast_first is True
  assert r.a_target <= 0.0


def test_speedup_guard_allows_when_gap_large():
  # huge gap -> a modest speed-up stays comfortable -> unchanged
  out = lead_speedup_guard(v_ego=18.0, v_lead=20.0, d_rel=120.0, follow_gap=20.0, proposed_accel=0.8)
  assert out == pytest.approx(0.8)


def test_speedup_guard_caps_when_gap_tight():
  # small excess gap (2 m) + a strong speed-up that would overtake -> capped below proposed
  capped = lead_speedup_guard(v_ego=20.0, v_lead=20.5, d_rel=22.0, follow_gap=20.0, proposed_accel=3.0)
  assert 0.0 <= capped < 3.0


def test_speedup_guard_zero_when_no_excess_gap():
  assert lead_speedup_guard(v_ego=20.0, v_lead=20.0, d_rel=18.0, follow_gap=20.0, proposed_accel=1.0) == 0.0


def test_speedup_guard_allows_inside_gap_launch_that_stays_slower_than_lead():
  assert lead_speedup_guard(
    v_ego=0.0, v_lead=0.44, d_rel=4.84, follow_gap=5.0, proposed_accel=0.18,
  ) == pytest.approx(0.18)


def test_speedup_guard_passes_through_non_positive():
  assert lead_speedup_guard(v_ego=20.0, v_lead=15.0, d_rel=30.0, follow_gap=20.0, proposed_accel=-0.5) == -0.5


def test_speedup_guard_keeps_required_decel_comfortable():
  # after capping, the implied required decel must be >= the comfort floor
  excess = 24.0 - 20.0
  capped = lead_speedup_guard(v_ego=20.0, v_lead=20.5, d_rel=24.0, follow_gap=20.0, proposed_accel=3.0,
                              dt_lookahead=1.0, max_required_decel=-1.2)
  v_next = 20.0 + capped * 1.0
  closing = max(0.0, v_next - 20.5)
  required = -(closing ** 2) / (2 * excess)
  assert required >= -1.2 - 1e-6


def test_catchup_cap_uses_net_accel_not_natural_coast():
  out = lead_catchup_accel_cap(
    v_ego=0.6, v_lead=0.6, a_lead=0.0,
    d_rel=7.0, follow_gap=5.8, proposed_accel=0.4,
  )
  assert out == pytest.approx(0.12)


def test_catchup_cap_lets_lead_recover_inside_target_gap():
  out = lead_catchup_accel_cap(
    v_ego=0.6, v_lead=0.6, a_lead=0.5,
    d_rel=4.0, follow_gap=5.0, proposed_accel=0.5,
  )
  assert out == pytest.approx(0.4)
