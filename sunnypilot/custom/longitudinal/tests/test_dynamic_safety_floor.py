"""Tests for dynamic safety-floor shadow telemetry."""
from __future__ import annotations

import math

import pytest

from openpilot.sunnypilot.custom.longitudinal.dynamic_safety_floor import (
  compute_dynamic_safety_floor,
  get_safe_obstacle_distance,
  follow_offset,
  debug_dict,
)


TAU = 1.5


def test_follow_offset_is_monotonic_and_bounded():
  assert follow_offset(0.0) == pytest.approx(4.5)
  assert follow_offset(4.0) == pytest.approx(2.875)
  assert follow_offset(20.0) == pytest.approx(1.375)
  assert follow_offset(100.0) > 1.25
  assert follow_offset(0.0) > follow_offset(4.0) > follow_offset(20.0) > follow_offset(100.0)


def test_get_safe_obstacle_distance_finite_and_grows_with_speed():
  d0 = get_safe_obstacle_distance(0.0, TAU)
  d10 = get_safe_obstacle_distance(10.0, TAU)
  d20 = get_safe_obstacle_distance(20.0, TAU)
  assert math.isfinite(d0)
  assert math.isfinite(d10)
  assert math.isfinite(d20)
  assert d0 < d10 < d20


def test_compute_result_is_finite_and_fail_closed():
  r = compute_dynamic_safety_floor(15.0, TAU)
  assert math.isfinite(r.current_safe_distance)
  assert math.isfinite(r.proposed_safe_distance)
  assert math.isfinite(r.delta_safe_distance)
  assert math.isfinite(r.dynamic_floor_value)
  assert math.isfinite(r.comfort_brake_effective)
  # Proposed is never shorter than current fork distance.
  assert r.proposed_safe_distance >= r.current_safe_distance


def test_dynamic_floor_grows_with_speed_and_latency():
  r_slow = compute_dynamic_safety_floor(5.0, TAU)
  r_fast = compute_dynamic_safety_floor(25.0, TAU)
  assert r_fast.dynamic_floor_value > r_slow.dynamic_floor_value
  assert r_fast.dynamic_floor_value >= 4.0


def test_downhill_lengthens_proposed_distance():
  flat = compute_dynamic_safety_floor(15.0, TAU, pitch=0.0)
  downhill = compute_dynamic_safety_floor(15.0, TAU, pitch=-0.08)
  assert downhill.proposed_safe_distance > flat.proposed_safe_distance
  assert downhill.comfort_brake_effective < flat.comfort_brake_effective


def test_uphill_does_not_shorten_distance():
  flat = compute_dynamic_safety_floor(15.0, TAU, pitch=0.0)
  uphill = compute_dynamic_safety_floor(15.0, TAU, pitch=0.08)
  assert uphill.proposed_safe_distance >= flat.proposed_safe_distance
  assert uphill.comfort_brake_effective == flat.comfort_brake_effective


def test_lateral_accel_lengthens_proposed_distance():
  straight = compute_dynamic_safety_floor(15.0, TAU, a_lat=0.0)
  turning = compute_dynamic_safety_floor(15.0, TAU, a_lat=2.0)
  assert turning.proposed_safe_distance > straight.proposed_safe_distance
  assert turning.comfort_brake_effective < straight.comfort_brake_effective


def test_invalid_inputs_do_not_shorten_distance():
  base = compute_dynamic_safety_floor(15.0, TAU)
  missing_lat = compute_dynamic_safety_floor(15.0, TAU, a_lat=None)
  missing_pitch = compute_dynamic_safety_floor(15.0, TAU, pitch=None)
  missing_both = compute_dynamic_safety_floor(15.0, TAU, a_lat=None, pitch=None)
  invalid_both = compute_dynamic_safety_floor(15.0, TAU, a_lat=math.nan, pitch=math.inf)
  assert missing_lat.proposed_safe_distance >= base.proposed_safe_distance
  assert missing_pitch.proposed_safe_distance >= base.proposed_safe_distance
  assert missing_both.proposed_safe_distance >= base.proposed_safe_distance
  assert invalid_both.proposed_safe_distance >= base.proposed_safe_distance
  assert missing_both.block_reason != ""
  assert invalid_both.block_reason != ""


def test_kinematic_floor_violation_detected():
  r = compute_dynamic_safety_floor(15.0, TAU, lead_d_rel=3.0)
  assert r.kinematic_floor_violation is True


def test_no_violation_when_lead_beyond_floor():
  r = compute_dynamic_safety_floor(15.0, TAU, lead_d_rel=20.0)
  assert r.kinematic_floor_violation is False


def test_debug_dict_has_required_prefix_keys():
  r = compute_dynamic_safety_floor(15.0, TAU)
  d = debug_dict(r)
  required = {
    "dynamic_safety_floor_active",
    "dynamic_safety_floor_block_reason",
    "dynamic_safety_floor_current_safe_distance",
    "dynamic_safety_floor_proposed_safe_distance",
    "dynamic_safety_floor_delta_safe_distance",
    "dynamic_safety_floor_dynamic_floor_value",
    "dynamic_safety_floor_kinematic_floor_violation",
    "dynamic_safety_floor_comfort_brake_effective",
    "dynamic_safety_floor_latency_s",
    "dynamic_safety_floor_lat_accel",
    "dynamic_safety_floor_pitch",
  }
  assert required.issubset(d.keys())
  assert d["dynamic_safety_floor_active"] is True
