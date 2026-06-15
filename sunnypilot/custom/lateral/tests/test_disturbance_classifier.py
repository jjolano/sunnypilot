import pytest

from openpilot.sunnypilot.custom.lateral.disturbance_classifier import (
  DisturbanceClassifier,
  DisturbanceReason,
  LateralSample,
  LearningDecision,
  decision_name,
  reason_names,
)


def _sample(
  t: float = 0.0,
  *,
  lat_active: bool = True,
  steering_pressed: bool = False,
  lane_change_active: bool = False,
  output_sat: bool = False,
  steer_limited: bool = False,
  actual_lateral_accel: float | None = 0.1,
  steering_rate_deg: float | None = 0.0,
  output: float | None = 0.05,
  unshaped_output: float | None = 0.05,
  desired_lateral_accel: float | None = 0.1,
  model_path_quality: float | None = 1.0,
  model_path_gated: bool = False,
) -> LateralSample:
  return LateralSample(
    t=t,
    v_ego=20.0,
    lat_active=lat_active,
    steering_pressed=steering_pressed,
    lane_change_active=lane_change_active,
    output_sat=output_sat,
    steer_limited=steer_limited,
    actual_lateral_accel=actual_lateral_accel,
    steering_rate_deg=steering_rate_deg,
    output=output,
    unshaped_output=unshaped_output,
    desired_lateral_accel=desired_lateral_accel,
    model_path_quality=model_path_quality,
    model_path_gated=model_path_gated,
  )


def test_clean_tracking_accepted_with_high_confidence():
  clf = DisturbanceClassifier()
  sample = _sample(t=0.0)
  result = clf.classify(sample)

  assert result.decision == LearningDecision.ACCEPT
  assert result.confidence == 1.0
  assert DisturbanceReason.CLEAN in result.reasons
  assert not result.reasons & DisturbanceReason.MISSING_CONTEXT


def test_clean_tracking_does_not_start_cooldown():
  clf = DisturbanceClassifier(cooldown_s=1.0)
  first = clf.classify(_sample(t=0.0))
  second = clf.classify(_sample(t=0.1))

  assert first.decision == LearningDecision.ACCEPT
  assert second.decision == LearningDecision.ACCEPT
  assert DisturbanceReason.COOLDOWN_ACTIVE not in second.reasons


def test_missing_optional_context_lowers_confidence_but_accepts():
  clf = DisturbanceClassifier()
  sample = LateralSample(
    t=0.0,
    v_ego=20.0,
    lat_active=True,
    actual_lateral_accel=0.1,
  )
  result = clf.classify(sample)

  assert result.decision == LearningDecision.ACCEPT
  assert result.confidence < 1.0
  assert DisturbanceReason.MISSING_CONTEXT in result.reasons
  assert DisturbanceReason.CLEAN in result.reasons


def test_driver_override_rejects_shadow():
  clf = DisturbanceClassifier()
  sample = _sample(t=0.0, steering_pressed=True)
  result = clf.classify(sample)

  assert result.decision == LearningDecision.REJECT_SHADOW
  assert result.confidence == 1.0
  assert DisturbanceReason.DRIVER_OVERRIDE in result.reasons


def test_inactive_sample_rejects_shadow_not_clean():
  clf = DisturbanceClassifier()
  sample = _sample(t=0.0, lat_active=False)
  result = clf.classify(sample)

  assert result.decision == LearningDecision.REJECT_SHADOW
  assert DisturbanceReason.LAT_INACTIVE in result.reasons
  assert DisturbanceReason.CLEAN not in result.reasons


def test_lane_change_rejects_shadow():
  clf = DisturbanceClassifier()
  sample = _sample(t=0.0, lane_change_active=True)
  result = clf.classify(sample)

  assert result.decision == LearningDecision.REJECT_SHADOW
  assert DisturbanceReason.LANE_CHANGE in result.reasons


def test_control_limit_rejects_shadow():
  clf = DisturbanceClassifier()
  for flag in ("output_sat", "steer_limited"):
    sample = _sample(t=0.0, **{flag: True})
    result = clf.classify(sample)
    assert result.decision == LearningDecision.REJECT_SHADOW, flag
    assert DisturbanceReason.CONTROL_LIMIT in result.reasons


def test_measurement_spike_quarantines():
  clf = DisturbanceClassifier()
  sample = _sample(t=0.0, actual_lateral_accel=3.0)
  result = clf.classify(sample)

  assert result.decision == LearningDecision.QUARANTINE
  assert DisturbanceReason.MEASUREMENT_SPIKE in result.reasons


def test_bump_quarantines_based_on_rate_and_output_delta():
  clf = DisturbanceClassifier(cooldown_s=0.0)
  prev = _sample(t=0.0, output=0.0, steering_rate_deg=0.0)
  sample = _sample(t=0.05, output=0.15, steering_rate_deg=90.0)
  result = clf.classify(sample, prev_sample=prev, dt=0.05)

  assert result.decision == LearningDecision.QUARANTINE
  assert DisturbanceReason.BUMP in result.reasons


def test_model_demand_jitter_quarantines():
  clf = DisturbanceClassifier(cooldown_s=0.0)
  prev = _sample(t=0.0, desired_lateral_accel=0.0)
  sample = _sample(t=0.05, desired_lateral_accel=1.5)
  result = clf.classify(sample, prev_sample=prev, dt=0.05)

  assert result.decision == LearningDecision.QUARANTINE
  assert DisturbanceReason.MODEL_DEMAND_JITTER in result.reasons


def test_model_path_low_quality_quarantines():
  clf = DisturbanceClassifier()
  sample = _sample(t=0.0, model_path_quality=0.5)
  result = clf.classify(sample)

  assert result.decision == LearningDecision.QUARANTINE
  assert DisturbanceReason.MODEL_PATH_LOW_QUALITY in result.reasons


def test_model_path_gated_quarantines():
  clf = DisturbanceClassifier()
  sample = _sample(t=0.0, model_path_gated=True)
  result = clf.classify(sample)

  assert result.decision == LearningDecision.QUARANTINE
  assert DisturbanceReason.MODEL_PATH_LOW_QUALITY in result.reasons


def test_cooldown_hysteresis_sticks_after_rejection():
  clf = DisturbanceClassifier(cooldown_s=1.0)
  reject = clf.classify(_sample(t=0.0, steering_pressed=True))
  assert reject.decision == LearningDecision.REJECT_SHADOW
  assert reject.cooldown_remaining == pytest.approx(1.0)

  clean = clf.classify(_sample(t=0.3, steering_pressed=False))
  # Still inside cooldown of a high-confidence reject event.
  assert DisturbanceReason.COOLDOWN_ACTIVE in clean.reasons
  assert clean.cooldown_remaining == pytest.approx(0.7, abs=1e-6)


def test_cooldown_hysteresis_sticks_after_quarantine():
  clf = DisturbanceClassifier(cooldown_s=1.0)
  quarantine = clf.classify(_sample(t=0.0, model_path_gated=True))
  assert quarantine.decision == LearningDecision.QUARANTINE
  assert quarantine.cooldown_remaining == pytest.approx(1.0)

  clean = clf.classify(_sample(t=0.4, model_path_gated=False))
  assert DisturbanceReason.COOLDOWN_ACTIVE in clean.reasons
  assert clean.cooldown_remaining == pytest.approx(0.6, abs=1e-6)


def test_uncertain_does_not_restrict_output_or_reject():
  clf = DisturbanceClassifier()
  sample = _sample(t=0.0, model_path_quality=0.5, output=None, desired_lateral_accel=None)
  result = clf.classify(sample)

  assert result.decision == LearningDecision.QUARANTINE
  assert result.confidence < 1.0
  assert DisturbanceReason.MISSING_CONTEXT in result.reasons
  assert DisturbanceReason.MODEL_PATH_LOW_QUALITY in result.reasons


def test_reason_names_and_decision_helpers():
  reasons = DisturbanceReason.CLEAN | DisturbanceReason.MISSING_CONTEXT
  names = reason_names(reasons)
  assert "CLEAN" in names
  assert "MISSING_CONTEXT" in names
  assert decision_name(LearningDecision.QUARANTINE) == "quarantine"
