import pytest

from openpilot.tools.drive_lab.lateral_disturbance_profile import (
  build_lateral_disturbance_profile,
  render_lateral_disturbance_profile,
  save_lateral_disturbance_profile,
  load_lateral_disturbance_profile,
)
from openpilot.tools.drive_lab.tests.test_lateral_torque_event_report import (
  sample_msgs,
)


def test_profile_empty_route():
  profile = build_lateral_disturbance_profile([], source="empty")
  assert profile.sample_count == 0
  assert profile.eligible_sample_count == 0
  assert profile.duration_s == 0.0


def test_profile_counts_lane_change_as_rejected():
  msgs = []
  for i in range(20):
    msgs.extend(sample_msgs(
      i * 0.1,
      output=0.1,
      lane_change_state="laneChangeStarting",
      desired_accel=0.1,
      actual_accel=0.1,
    ))

  profile = build_lateral_disturbance_profile(msgs, source="synthetic", already_sorted=True)

  assert profile.sample_count == 20
  assert profile.eligible_sample_count == 20
  assert profile.shadow_rejected == 20
  assert profile.shadow_reject_percent == pytest.approx(100.0)
  assert profile.shadow_accepted == 0
  assert profile.shadow_quarantined == 0
  assert profile.decision_counts.get("reject_shadow", 0) == 20
  rendered = render_lateral_disturbance_profile(profile)
  assert "shadow rejected: 20" in rendered


def test_profile_excludes_inactive_samples_from_shadow_percentages():
  msgs = []
  for i in range(10):
    sample = sample_msgs(
      i * 0.1,
      output=0.1,
      desired_accel=0.1,
      actual_accel=0.1,
    )
    sample[1].carControl.latActive = False
    msgs.extend(sample)

  profile = build_lateral_disturbance_profile(msgs, source="synthetic", already_sorted=True)
  rendered = render_lateral_disturbance_profile(profile)

  assert profile.sample_count == 10
  assert profile.eligible_sample_count == 0
  assert profile.inactive_excluded == 10
  assert profile.shadow_accepted == 0
  assert profile.shadow_quarantined == 0
  assert profile.shadow_rejected == 0
  assert profile.shadow_quarantine_percent == 0.0
  assert profile.shadow_reject_percent == 0.0
  assert "inactive excluded: 10" in rendered


def test_profile_counts_low_quality_path_as_quarantined():
  msgs = []
  for i in range(20):
    msgs.extend(sample_msgs(
      i * 0.1,
      output=0.1,
      desired_accel=0.1,
      actual_accel=0.1,
      path_gated=True,
      path_quality=0.5,
      path_reason="highPathStd",
    ))

  profile = build_lateral_disturbance_profile(msgs, source="synthetic", already_sorted=True)

  assert profile.sample_count == 20
  assert profile.eligible_sample_count == 20
  assert profile.shadow_quarantined == 20
  assert profile.shadow_quarantine_percent == pytest.approx(100.0)
  assert profile.shadow_rejected == 0
  assert profile.shadow_accepted == 0
  assert profile.reason_counts.get("MODEL_PATH_LOW_QUALITY", 0) == 20
  rendered = render_lateral_disturbance_profile(profile)
  assert "shadow quarantined: 20" in rendered


def test_profile_counts_clean_accepted():
  msgs = []
  for i in range(20):
    msgs.extend(sample_msgs(
      i * 0.1,
      output=0.1,
      desired_accel=0.1,
      actual_accel=0.1,
      path_quality=1.0,
    ))

  profile = build_lateral_disturbance_profile(msgs, source="synthetic", already_sorted=True)

  assert profile.sample_count == 20
  assert profile.eligible_sample_count == 20
  assert profile.shadow_accepted == 20
  assert profile.shadow_quarantined == 0
  assert profile.shadow_rejected == 0
  assert profile.reason_counts.get("CLEAN", 0) == 20


def test_profile_round_trip_json(tmp_path):
  msgs = []
  for i in range(10):
    msgs.extend(sample_msgs(
      i * 0.1,
      output=0.1,
      desired_accel=0.1,
      actual_accel=0.1,
      path_gated=True,
      path_quality=0.5,
    ))

  profile = build_lateral_disturbance_profile(msgs, source="synthetic", already_sorted=True)
  path = tmp_path / "disturbance-profile.json"
  save_lateral_disturbance_profile(profile, path)
  loaded = load_lateral_disturbance_profile(path)

  assert loaded.sample_count == profile.sample_count
  assert loaded.eligible_sample_count == profile.eligible_sample_count
  assert loaded.shadow_quarantined == profile.shadow_quarantined
  assert loaded.shadow_reject_percent == profile.shadow_reject_percent
