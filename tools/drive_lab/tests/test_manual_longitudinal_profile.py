import numpy as np

from openpilot.tools.drive_lab.manual_longitudinal_profile import (
  ManualSample,
  ProfileRange,
  SmoothAssertiveEnvelope,
  build_route_profile,
  classify_style,
  percentile_range,
  summarize_manual_style,
)


def sample(t, v, a, active=False, gas=False, brake=False, lead=False, d_rel=0.0, v_rel=0.0):
  return ManualSample(
    route="route-a",
    t=t,
    v_ego=v,
    a_ego=a,
    active=active,
    gas_pressed=gas,
    brake_pressed=brake,
    lead_status=lead,
    lead_d_rel=d_rel,
    lead_v_rel=v_rel,
  )


def test_percentile_range_uses_requested_percentiles():
  result = percentile_range([0.0, 1.0, 2.0, 3.0, 4.0], low_pct=25.0, high_pct=75.0)

  assert result == ProfileRange(low=1.0, high=3.0)


def test_percentile_range_accepts_numpy_float_scalars():
  result = percentile_range([np.float32(1.0), np.float32(2.0), np.float32(3.0)], 0.0, 100.0)

  assert result == ProfileRange(1.0, 3.0)


def test_percentile_range_accepts_numpy_int_scalars():
  result = percentile_range([np.int64(1), np.int64(2), np.int64(3)], 0.0, 100.0)

  assert result == ProfileRange(1.0, 3.0)


def test_percentile_range_ignores_non_finite_values():
  result = percentile_range([float("nan"), 1.0, float("inf"), 3.0], 0.0, 100.0)

  assert result == ProfileRange(1.0, 3.0)


def test_classifies_smooth_assertive_profile_inside_envelope():
  style = classify_style(
    accel=ProfileRange(-0.815, 0.917),
    launch_mean=ProfileRange(0.687, 0.932),
    stop_mean=ProfileRange(-0.890, -0.409),
    coast_accel=ProfileRange(-0.336, -0.294),
    envelope=SmoothAssertiveEnvelope(),
  )

  assert style == "smooth_assertive"


def test_classifies_unknown_when_profile_is_too_aggressive():
  style = classify_style(
    accel=ProfileRange(-2.5, 2.8),
    launch_mean=ProfileRange(1.6, 2.4),
    stop_mean=ProfileRange(-2.2, -1.6),
    coast_accel=ProfileRange(-0.8, -0.6),
    envelope=SmoothAssertiveEnvelope(),
  )

  assert style == "unknown"


def test_route_profile_includes_mostly_manual_route():
  samples = [sample(float(i), 8.0, 0.1, active=False) for i in range(20)]
  samples += [sample(20.0, 8.0, 0.1, active=True)]

  profile = build_route_profile("route-a", samples, min_manual_moving_samples=10, max_active_ratio=0.25)

  assert profile.include
  assert profile.manual_moving_samples == 20
  assert profile.active_ratio == 1 / 21


def test_route_profile_excludes_routes_with_too_much_active_control():
  samples = [sample(float(i), 8.0, 0.1, active=False) for i in range(10)]
  samples += [sample(float(i + 10), 8.0, 0.1, active=True) for i in range(10)]

  profile = build_route_profile("route-a", samples, min_manual_moving_samples=5, max_active_ratio=0.25)

  assert not profile.include
  assert profile.active_ratio == 0.5


def test_route_profile_ignores_stopped_samples_for_manual_moving_count():
  samples = [sample(float(i), 0.2, 0.0, active=False) for i in range(20)]
  samples += [sample(float(i + 20), 6.0, 0.1, active=False) for i in range(6)]

  profile = build_route_profile("route-a", samples, min_manual_moving_samples=10, max_active_ratio=0.25)

  assert not profile.include
  assert profile.manual_moving_samples == 6


def test_manual_style_summary_separates_lead_and_clear_launches():
  samples = [
    sample(0.0, 0.5, 1.4, gas=True, lead=True, d_rel=4.0),
    sample(1.0, 3.0, 0.8, gas=True, lead=True, d_rel=4.5),
    sample(2.0, 6.0, 0.4, gas=False, lead=True, d_rel=5.0),
    sample(10.0, 0.3, 1.8, gas=True, lead=False),
    sample(11.0, 4.0, 1.0, gas=True, lead=False),
    sample(12.0, 7.0, 0.2, gas=False, lead=False),
  ]

  summary = summarize_manual_style(samples)

  assert summary.lead_launch_count == 1
  assert summary.clear_launch_count == 1
  assert summary.lead_launch_mean_accel.low == summary.lead_launch_mean_accel.high == 1.1
  assert summary.clear_launch_peak_accel.low == summary.clear_launch_peak_accel.high == 1.8


def test_manual_style_summary_counts_stop_approaches_and_coast():
  samples = [
    sample(0.0, 10.0, -0.6, brake=True, lead=True, d_rel=12.0, v_rel=-1.0),
    sample(1.0, 6.0, -0.4, brake=True, lead=True, d_rel=10.0, v_rel=-0.5),
    sample(2.0, 0.5, -0.1, brake=False, lead=True, d_rel=8.0, v_rel=0.0),
    sample(10.0, 14.0, -0.3, gas=False, brake=False),
    sample(11.0, 15.0, -0.4, gas=False, brake=False),
    sample(12.0, 16.0, -0.2, gas=True, brake=False),
  ]

  summary = summarize_manual_style(samples)

  assert summary.lead_stop_count == 1
  assert summary.clear_stop_count == 0
  assert summary.stop_mean_accel.low == summary.stop_mean_accel.high == -0.5
  assert summary.coast_accel.low == -0.4
  assert summary.coast_accel.high == -0.3
