from dataclasses import asdict

import numpy as np
import pytest

from openpilot.tools.drive_lab.manual_longitudinal_profile import (
  ManualSample,
  ProfileRange,
  SmoothAssertiveEnvelope,
  build_route_profile,
  classify_stop_tier,
  classify_style,
  lead_crawl_gap_excess,
  lead_stop_presentation_distance,
  percentile_range,
  render_manual_style_summary,
  summarize_manual_style,
)


def sample(t, v, a, active=False, gas=False, brake=False, lead=False, d_rel=0.0, v_rel=0.0,
           route="route-a", lead_v=None, lead_a=0.0, model_prob=1.0,
           model_should_stop=False, model_desired_accel=None, model_stop_distance=None):
  return ManualSample(
    route=route,
    t=t,
    v_ego=v,
    a_ego=a,
    active=active,
    gas_pressed=gas,
    brake_pressed=brake,
    lead_status=lead,
    lead_d_rel=d_rel,
    lead_v_rel=v_rel,
    lead_v_lead=lead_v,
    lead_a_lead=lead_a,
    lead_model_prob=model_prob,
    model_should_stop=model_should_stop,
    model_desired_accel=model_desired_accel,
    model_stop_distance=model_stop_distance,
  )


def crawl_sample(t, gap_excess, v=0.3, a=0.0, lead_v=0.2, lead_a=0.0, model_prob=1.0,
                 gas=False, brake=False, route="route-a", active=False):
  stop_target = lead_stop_presentation_distance(v, lead_v, lead_a, model_prob)
  return sample(
    t=t,
    v=v,
    a=a,
    active=active,
    gas=gas,
    brake=brake,
    lead=True,
    d_rel=stop_target + gap_excess,
    v_rel=lead_v - v,
    route=route,
    lead_v=lead_v,
    lead_a=lead_a,
    model_prob=model_prob,
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


def test_classify_stop_tier_uses_required_decel_thresholds():
  assert classify_stop_tier(None) == "ambiguous"
  assert classify_stop_tier(0.5) == "routine"
  assert classify_stop_tier(1.0) == "ambiguous"
  assert classify_stop_tier(1.5) == "urgent"


def test_lead_crawl_gap_excess_uses_stop_presentation_distance():
  crawl = crawl_sample(0.0, gap_excess=2.0, v=0.25, lead_v=0.15, lead_a=-0.05, model_prob=0.9)

  assert lead_crawl_gap_excess(crawl) == pytest.approx(2.0)


def test_lead_crawl_gap_excess_falls_back_to_relative_speed():
  stop_target = lead_stop_presentation_distance(0.3, 0.1, 0.0, 1.0)
  crawl = sample(0.0, 0.3, 0.0, lead=True, d_rel=stop_target + 1.0, v_rel=-0.2, lead_v=None)

  assert lead_crawl_gap_excess(crawl) == pytest.approx(1.0)


def test_lead_crawl_gap_excess_ignores_missing_confirmed_lead():
  no_lead = sample(0.0, 0.3, 0.0, lead=False, d_rel=10.0, v_rel=0.0)
  missing_distance = sample(1.0, 0.3, 0.0, lead=True, d_rel=None, v_rel=0.0)
  missing_speed = sample(2.0, 0.3, 0.0, lead=True, d_rel=8.0, v_rel=None, lead_v=None)

  assert lead_crawl_gap_excess(no_lead) is None
  assert lead_crawl_gap_excess(missing_distance) is None
  assert lead_crawl_gap_excess(missing_speed) is None


def test_manual_style_summary_includes_lead_crawl_bins():
  samples = [
    crawl_sample(0.0, 2.0, v=0.4, a=0.10, lead_v=0.2, gas=True),
    crawl_sample(1.0, 1.0, v=0.5, a=-0.10, lead_v=0.2, brake=True),
    crawl_sample(2.0, 0.0, v=0.2, a=-0.20, lead_v=0.0, brake=True),
    crawl_sample(3.0, -0.2, v=0.1, a=-0.30, lead_v=0.0, brake=True),
    crawl_sample(5.0, 3.0, v=4.0, a=0.20, lead_v=4.0, gas=True),
  ]

  summary = summarize_manual_style(samples)
  bins = {bucket.label: bucket for bucket in summary.lead_crawl_bins}

  assert bins["open_to_crawl"].sample_count == 1
  assert bins["crawl_to_follow"].sample_count == 1
  assert bins["soft_stop"].sample_count == 1
  assert bins["inside_stop_target"].sample_count == 1
  assert bins["open_to_crawl"].gas_ratio == 1.0
  assert bins["open_to_crawl"].brake_ratio == 0.0
  assert bins["open_to_crawl"].coast_ratio == 0.0
  assert bins["crawl_to_follow"].gas_ratio == 0.0
  assert bins["crawl_to_follow"].brake_ratio == 1.0
  assert bins["crawl_to_follow"].coast_ratio == 0.0
  assert bins["soft_stop"].gas_ratio == 0.0
  assert bins["soft_stop"].brake_ratio == 1.0
  assert bins["soft_stop"].coast_ratio == 0.0
  assert bins["open_to_crawl"].gap_excess.low == bins["open_to_crawl"].gap_excess.high == 2.0
  assert bins["crawl_to_follow"].gap_excess.low == bins["crawl_to_follow"].gap_excess.high == 1.0
  assert bins["soft_stop"].gap_excess.low == bins["soft_stop"].gap_excess.high == 0.0
  assert bins["inside_stop_target"].gap_excess.low < 0.0
  assert bins["open_to_crawl"].closing_ratio == 1.0
  assert bins["open_to_crawl"].closing_speed.low > 0.0
  assert bins["open_to_crawl"].closing_speed.high > 0.0
  assert "open_to_crawl" in {bucket.label for bucket in summary.lead_crawl_bins}


def test_manual_style_summary_lead_crawl_bins_are_low_speed_only():
  samples = [
    crawl_sample(0.0, 2.5, v=0.3, a=0.10, lead_v=0.2, gas=True),
    crawl_sample(1.0, 2.5, v=4.0, a=0.10, lead_v=0.2, gas=True),
    crawl_sample(2.0, 2.5, v=0.3, a=0.10, lead_v=4.2, gas=True),
    crawl_sample(3.0, 2.5, v=4.0, a=0.10, lead_v=4.2, gas=True),
  ]

  summary = summarize_manual_style(samples)

  assert summary.lead_crawl_bins[0].label == "open_to_crawl"
  assert summary.lead_crawl_bins[0].sample_count == 1


def test_manual_style_summary_lead_crawl_closing_threshold_is_strict():
  samples = [
    crawl_sample(0.0, 2.5, v=0.3, a=0.0, lead_v=0.2),
    crawl_sample(1.0, 2.6, v=0.4, a=0.0, lead_v=0.2),
  ]

  summary = summarize_manual_style(samples)
  open_to_crawl = next(bucket for bucket in summary.lead_crawl_bins if bucket.label == "open_to_crawl")

  assert open_to_crawl.closing_ratio == 0.5
  assert open_to_crawl.closing_speed.low == pytest.approx(0.2)
  assert open_to_crawl.closing_speed.high == pytest.approx(0.2)


def test_manual_style_summary_extracts_lead_crawl_and_soft_stop_episodes():
  samples = [
    crawl_sample(0.0, 2.4, v=0.2, a=0.10, lead_v=0.3),
    crawl_sample(1.0, 1.6, v=0.4, a=0.05, lead_v=0.2),
    crawl_sample(2.0, 0.9, v=0.3, a=-0.10, lead_v=0.0),
    crawl_sample(3.0, 0.4, v=0.2, a=-0.20, lead_v=0.0),
    crawl_sample(4.0, 0.0, v=0.0, a=-0.10, lead_v=0.0),
  ]

  summary = summarize_manual_style(samples)
  episodes = {episode.label: episode for episode in summary.lead_crawl_episodes}

  assert episodes["crawl_to_follow"].count == 1
  assert episodes["crawl_to_follow"].start_gap_excess.low == pytest.approx(2.4)
  assert episodes["crawl_to_follow"].end_gap_excess.high == pytest.approx(0.9)
  assert episodes["soft_stop"].count == 1
  assert episodes["soft_stop"].min_gap_excess.low == pytest.approx(0.0)


def test_manual_style_summary_does_not_merge_crawl_episodes_across_routes():
  samples = [
    crawl_sample(0.0, 2.4, route="route-a"),
    crawl_sample(1.0, 1.6, route="route-b"),
    crawl_sample(2.0, 0.9, route="route-b"),
  ]

  summary = summarize_manual_style(samples)

  assert summary.lead_crawl_episodes == []


def test_manual_style_summary_breaks_crawl_episodes_on_invalid_intervening_samples():
  samples = [
    crawl_sample(0.0, 2.4, v=0.2, lead_v=0.2),
    crawl_sample(1.0, 1.6, v=3.0, lead_v=0.2),
    crawl_sample(2.0, 0.9, v=0.3, lead_v=0.0),
  ]

  summary = summarize_manual_style(samples)

  assert summary.lead_crawl_episodes == []


def test_manual_style_summary_breaks_crawl_episodes_on_active_intervening_samples():
  samples = [
    crawl_sample(0.0, 2.4, v=0.2, lead_v=0.2),
    crawl_sample(1.0, 1.6, v=0.2, lead_v=0.2, active=True),
    crawl_sample(2.0, 0.9, v=0.3, lead_v=0.0),
  ]

  summary = summarize_manual_style(samples)

  assert summary.lead_crawl_episodes == []


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
  assert round(summary.coast_accel.low, 2) == -0.35
  assert round(summary.coast_accel.high, 2) == -0.31


def test_manual_style_summary_stop_mean_ignores_stopped_brake_hold_samples():
  samples = [
    sample(0.0, 10.0, -0.6, brake=True, lead=True, d_rel=12.0, v_rel=-1.0),
    sample(1.0, 5.0, -0.4, brake=True, lead=True, d_rel=10.0, v_rel=-0.5),
    sample(2.0, 0.5, 0.0, brake=True, lead=True, d_rel=8.0, v_rel=0.0),
  ]
  samples += [sample(3.0 + i, 0.0, 0.0, brake=True, lead=True, d_rel=8.0, v_rel=0.0) for i in range(20)]
  samples.append(sample(24.0, 0.0, 0.0, brake=False, lead=True, d_rel=8.0, v_rel=0.0))

  summary = summarize_manual_style(samples)

  assert summary.lead_stop_count == 1
  assert summary.stop_mean_accel.low == summary.stop_mean_accel.high == -0.5
  assert summary.stop_peak_decel.low == summary.stop_peak_decel.high == -0.6


def test_manual_style_summary_separates_stop_approach_tiers_by_runway():
  samples = [
    sample(0.0, 8.0, -0.4, brake=True, model_stop_distance=60.0),
    sample(1.0, 4.0, -0.4, brake=True, model_stop_distance=50.0),
    sample(2.0, 0.5, -0.1, brake=False, model_stop_distance=45.0),
    sample(10.0, 10.0, -1.5, brake=True, model_stop_distance=20.0),
    sample(11.0, 5.0, -1.3, brake=True, model_stop_distance=10.0),
    sample(12.0, 0.5, -0.2, brake=False, model_stop_distance=5.0),
    sample(20.0, 8.0, -0.8, brake=True),
    sample(21.0, 4.0, -0.7, brake=True),
    sample(22.0, 0.5, -0.1, brake=False),
  ]

  summary = summarize_manual_style(samples)
  bins = {stop_bin.label: stop_bin for stop_bin in summary.stop_approach_bins}

  assert bins["routine"].count == 1
  assert bins["routine"].required_decel.low == pytest.approx(8.0 ** 2 / (2.0 * 60.0))
  assert bins["urgent"].count == 1
  assert bins["urgent"].required_decel.low == pytest.approx(10.0 ** 2 / (2.0 * 20.0))
  assert bins["ambiguous"].count == 1


def test_manual_style_classification_uses_routine_stop_when_urgent_stop_is_present():
  samples = [
    sample(0.0, 0.5, 0.8, gas=True, model_stop_distance=80.0),
    sample(1.1, 3.0, 0.8, gas=True, model_stop_distance=80.0),
    sample(2.2, 6.0, 0.2, gas=False, model_stop_distance=80.0),
    sample(10.0, 10.0, -0.6, brake=True, model_stop_distance=80.0),
    sample(11.1, 5.0, -0.5, brake=True, model_stop_distance=70.0),
    sample(12.2, 0.5, -0.2, brake=False, model_stop_distance=60.0),
    sample(30.0, 10.0, -2.0, brake=True, model_stop_distance=20.0),
    sample(31.1, 5.0, -1.8, brake=True, model_stop_distance=10.0),
    sample(32.2, 0.5, -0.2, brake=False, model_stop_distance=5.0),
  ]
  samples += [sample(40.0 + i, 12.0, 0.8, gas=True) for i in range(8)]
  samples += [sample(50.0 + i, 12.0, -0.7, brake=True) for i in range(8)]
  samples += [sample(60.0 + i, 12.0, -0.3, gas=False, brake=False) for i in range(20)]

  summary = summarize_manual_style(samples)

  assert {stop_bin.label for stop_bin in summary.stop_approach_bins} == {"routine", "urgent"}
  assert summary.style == "smooth_assertive"


def test_manual_style_summary_coast_outlier_does_not_dominate_classification():
  samples = [
    sample(0.0, 0.5, 0.8, gas=True, lead=True, d_rel=4.0),
    sample(1.1, 3.0, 0.8, gas=True, lead=True, d_rel=4.5),
    sample(2.2, 6.0, 0.2, gas=False, lead=True, d_rel=5.0),
    sample(10.0, 10.0, -0.7, brake=True, lead=True, d_rel=12.0, v_rel=-1.0),
    sample(11.1, 5.0, -0.5, brake=True, lead=True, d_rel=10.0, v_rel=-0.5),
    sample(12.2, 0.5, -0.2, brake=False, lead=True, d_rel=8.0, v_rel=0.0),
  ]
  samples += [sample(20.0 + i, 12.0, 0.8, gas=True) for i in range(8)]
  samples += [sample(40.0 + i, 12.0, -0.7, brake=True) for i in range(8)]
  samples += [sample(60.0 + i, 12.0, -0.3, gas=False, brake=False) for i in range(20)]
  samples.append(sample(90.0, 12.0, 5.0, gas=False, brake=False))

  summary = summarize_manual_style(samples)

  assert summary.coast_accel.low == summary.coast_accel.high == -0.3
  assert summary.style == "smooth_assertive"


def test_manual_style_summary_does_not_merge_pedal_episodes_across_routes():
  samples = [
    sample(0.0, 0.5, 1.4, gas=True, route="route-a"),
    sample(0.5, 4.0, 1.0, gas=True, route="route-b"),
    sample(2.0, 6.0, 0.4, gas=False, route="route-b"),
  ]

  summary = summarize_manual_style(samples)

  assert summary.clear_launch_count == 0


def test_manual_style_summary_accel_ignores_stopped_idle_samples():
  samples = [sample(float(i), 0.0, 0.0) for i in range(100)]
  samples += [sample(100.0 + i, 12.0, accel) for i, accel in enumerate([-0.8, -0.4, 0.0, 0.4, 0.8])]

  summary = summarize_manual_style(samples)

  assert summary.sample_count == 5
  assert summary.accel.low == -0.64
  assert round(summary.accel.high, 2) == 0.64


def test_manual_style_summary_includes_speed_and_following_bins():
  samples = [
    sample(0.0, 5.0, 0.5, gas=True, lead=True, d_rel=10.0, v_rel=-1.0),
    sample(1.0, 6.0, -0.5, brake=True, lead=True, d_rel=12.0, v_rel=-2.0),
    sample(2.0, 8.0, -0.3, gas=False, brake=False, lead=True, d_rel=16.0, v_rel=-1.0),
  ]

  summary = summarize_manual_style(samples)

  assert summary.speed_bins[0].label == "3-7 m/s"
  assert summary.speed_bins[0].sample_count == 2
  assert summary.speed_bins[0].gas_ratio == 0.5
  assert summary.speed_bins[0].brake_ratio == 0.5
  assert summary.following_bins[0].label == "3-7 m/s"
  assert summary.following_bins[0].sample_count == 2
  assert summary.following_bins[0].closing_ratio == 1.0


def test_manual_style_summary_uses_clear_launch_for_classification_when_no_lead_launches():
  samples = [
    sample(0.0, 0.5, 0.8, gas=True),
    sample(1.1, 3.0, 0.8, gas=True),
    sample(2.2, 6.0, 0.2, gas=False),
    sample(10.0, 10.0, -0.7, brake=True),
    sample(11.1, 5.0, -0.5, brake=True),
    sample(12.2, 0.5, -0.2, brake=False),
  ]
  samples += [sample(14.0 + i, 12.0, 0.8, gas=True) for i in range(8)]
  samples += [sample(16.0 + i, 12.0, -0.7, brake=True) for i in range(8)]
  samples += [sample(20.0 + i, 12.0, -0.3, gas=False, brake=False) for i in range(20)]

  summary = summarize_manual_style(samples)

  assert summary.lead_launch_count == 0
  assert summary.clear_launch_count == 1
  assert summary.style == "smooth_assertive"


def test_render_manual_style_summary_includes_core_values():
  summary = summarize_manual_style([
    sample(0.0, 0.5, 1.2, gas=True, lead=True, d_rel=4.0),
    sample(1.0, 5.5, 0.7, gas=False, lead=True, d_rel=5.0),
    sample(10.0, 10.0, -0.4, brake=True, lead=True, d_rel=12.0, v_rel=-0.8),
    sample(11.0, 0.5, -0.2, brake=False, lead=True, d_rel=8.0, v_rel=0.0),
    sample(20.0, 12.0, -0.3, gas=False, brake=False),
    sample(21.0, 13.0, -0.2, gas=False, brake=False),
  ])

  text = render_manual_style_summary(summary)

  assert "Manual longitudinal style" in text
  assert "style:" in text
  assert "lead launches:" in text
  assert "stop approaches:" in text
  assert "coast accel:" in text
  assert "Speed bins:" in text
  assert "Following bins:" in text
  assert "Stop approach tiers:" in text


def test_render_manual_style_summary_includes_lead_crawl_sections():
  summary = summarize_manual_style([
    crawl_sample(0.0, 2.4, v=0.2, a=0.10, lead_v=0.3, gas=True),
    crawl_sample(1.0, 1.5, v=0.4, a=0.00, lead_v=0.2),
    crawl_sample(2.0, 0.9, v=0.3, a=-0.10, lead_v=0.0, brake=True),
    crawl_sample(3.0, 0.0, v=0.0, a=-0.10, lead_v=0.0, brake=True),
  ])

  text = render_manual_style_summary(summary)
  payload = asdict(summary)

  assert "Lead crawl bins:" in text
  assert "open_to_crawl" in text
  assert "closing speed" in text
  assert "Lead crawl episodes:" in text
  assert payload["lead_crawl_bins"][0]["label"] == "open_to_crawl"
  assert payload["lead_crawl_episodes"][0]["label"] == "crawl_to_follow"
