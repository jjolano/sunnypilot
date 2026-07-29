from openpilot.sunnypilot.custom.localization.quality import (
  LocalizationQualityHealth,
  LocalizationQualityThresholds,
  freshness_summary,
  heading_error_deg,
  nearest_value,
  vector_norm3,
)


def test_heading_wrap_and_vector_norm_invalid():
  assert heading_error_deg(179.0, -179.0) == 2.0
  assert vector_norm3([3.0, 4.0, 0.0]) == 5.0
  assert vector_norm3([3.0, None, 0.0]) is None


def test_freshness_large_gaps_and_nearest_lookup():
  s = freshness_summary([0.0, 0.1, 0.8, 0.9], thresholds=LocalizationQualityThresholds(large_gap_s=0.5))
  assert s.large_gap_count == 1
  assert nearest_value([(0.0, "a"), (1.0, "b")], 0.9, 0.2, times=[0.0, 1.0]) == "b"


def test_health_is_conservative():
  health = LocalizationQualityHealth.from_signals(
    camera_fresh=freshness_summary([0.0, 1.0]),
    live_fresh=freshness_summary([0.0, 0.1]),
    high_trans_std_count=1,
    yaw_pair_count=0,
    gps_pair_count=0,
    gps_p95_abs_error_deg=30.0,
  )
  assert not health.ok
  assert any("missing" in reason or "degraded" in reason or "mismatch" in reason or "pairs" in reason for reason in health.degraded_reasons)
