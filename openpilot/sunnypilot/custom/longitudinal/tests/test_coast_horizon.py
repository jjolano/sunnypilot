"""Physics/property tests for coast-horizon anticipation and drag estimation."""
from __future__ import annotations

import math

import pytest

from openpilot.sunnypilot.custom.longitudinal.coast_horizon import (
  DEFAULT_COAST_DECEL,
  MAX_COAST_DECEL,
  MIN_COAST_DECEL,
  CoastAction,
  CoastHorizonInputs,
  DragEstimator,
  coast_decel_from_grade,
  coast_horizon,
)


def ch(v_ego, v_target, dist, a_coast=-0.25, comfort=-1.0):
  return coast_horizon(CoastHorizonInputs(v_ego, v_target, dist, a_coast, comfort))


def test_cruise_when_target_not_slower():
  assert ch(20.0, 22.0, 500.0).action is CoastAction.CRUISE
  assert ch(20.0, 20.0, 500.0).action is CoastAction.CRUISE


def test_cruise_when_constraint_far():
  r = ch(25.0, 15.0, 5000.0, a_coast=-0.25)
  assert r.action is CoastAction.CRUISE
  assert r.slack > 0.0


def test_coast_distance_matches_kinematics():
  # x = (v_t^2 - v0^2) / (2a)
  r = ch(25.0, 15.0, 10000.0, a_coast=-0.25)
  assert r.coast_distance == pytest.approx((15.0**2 - 25.0**2) / (2 * -0.25))  # 800 m


def test_coast_when_inside_lift_window():
  r = ch(25.0, 15.0, 805.0, a_coast=-0.25)  # coast_distance=800, lift_off=815
  assert r.action is CoastAction.COAST
  assert r.recommended_accel == pytest.approx(-0.25)  # command the natural coast decel


def test_brake_when_too_close():
  r = ch(25.0, 15.0, 100.0, a_coast=-0.25)
  assert r.action is CoastAction.BRAKE
  assert r.recommended_accel < 0.0
  # required decel to make the target over 100 m
  assert r.recommended_accel == pytest.approx((15.0**2 - 25.0**2) / (2 * 100.0))


def test_brake_when_slowing_with_almost_no_runway():
  # Need to slow (v_target < v_ego) but within MIN_USEFUL_DISTANCE -> must BRAKE, not CRUISE.
  # Regression: the old guard returned CRUISE here, zeroing braking in the last metre of a stop.
  for dist in (1.0, 0.5, 0.1):
    r = ch(1.55, 0.0, dist, a_coast=-0.25)
    assert r.action is CoastAction.BRAKE, f"dist={dist} should brake"
    assert r.recommended_accel < 0.0


def test_cruise_when_close_but_no_need_to_slow():
  # Short distance but target not slower -> still CRUISE (don't invent braking).
  assert ch(10.0, 12.0, 0.5).action is CoastAction.CRUISE


def test_weaker_coast_lifts_off_earlier():
  weak = ch(25.0, 15.0, 10000.0, a_coast=-0.15).lift_off_distance
  strong = ch(25.0, 15.0, 10000.0, a_coast=-0.8).lift_off_distance
  assert weak > strong  # gentler coast needs more runway -> lift earlier


def test_grade_changes_coast_decel():
  flat = coast_decel_from_grade(-0.25, 0.0)
  uphill = coast_decel_from_grade(-0.25, 0.05)    # positive pitch -> more decel
  downhill = coast_decel_from_grade(-0.25, -0.05)  # negative pitch -> less decel
  assert uphill < flat < downhill
  assert flat == pytest.approx(-0.25)


def test_drag_estimator_converges_to_observed_coast():
  est = DragEstimator(alpha=0.1)
  assert est.coast_decel == DEFAULT_COAST_DECEL
  for _ in range(300):
    est.update(v_ego=20.0, a_ego=-0.35, pitch_rad=0.0, on_throttle=False, on_brake=False)
  assert est.coast_decel == pytest.approx(-0.35, abs=1e-2)


def test_drag_estimator_ignores_throttle_and_brake():
  est = DragEstimator(alpha=0.1)
  before = est.coast_decel
  est.update(v_ego=20.0, a_ego=1.0, pitch_rad=0.0, on_throttle=True, on_brake=False)
  est.update(v_ego=20.0, a_ego=-3.0, pitch_rad=0.0, on_throttle=False, on_brake=True)
  assert est.coast_decel == before


def test_drag_estimate_stays_bounded():
  est = DragEstimator(alpha=0.5)
  for _ in range(50):
    est.update(v_ego=20.0, a_ego=-5.0, pitch_rad=0.0, on_throttle=False, on_brake=False)
  assert MIN_COAST_DECEL <= est.coast_decel <= MAX_COAST_DECEL
