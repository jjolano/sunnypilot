import pytest

from openpilot.sunnypilot.selfdrive.controls.lib.torque_v3_estimator import (
  AdaptiveTorqueEstimator,
  CONFIDENCE_BUILD_RATE,
  EstimatorRejectReason,
  TorqueObservation,
)


def make_observation(**overrides):
  values = dict(
    active=True,
    v_ego=16.0,
    steering_pressed=False,
    steer_limited_by_safety=False,
    curvature_limited=False,
    saturated=False,
    commanded_torque=0.25,
    desired_lateral_accel=0.65,
    actual_lateral_accel=0.60,
    actual_lateral_jerk=0.1,
    roll_compensation=0.0,
    model_age=0.0,
  )
  values.update(overrides)
  return TorqueObservation(**values)


def test_estimator_accepts_clean_sample_and_updates_params():
  estimator = AdaptiveTorqueEstimator(dt=0.01)

  result = estimator.update(make_observation())

  assert result.sample_accepted
  assert result.reject_reason == EstimatorRejectReason.NONE
  assert result.params.lat_accel_factor > 2.0
  assert result.confidence > 0.0
  assert result.positive_coverage > 0.0


def test_estimator_rejects_driver_override_without_learning():
  estimator = AdaptiveTorqueEstimator(dt=0.01)

  result = estimator.update(make_observation(steering_pressed=True))

  assert not result.sample_accepted
  assert result.reject_reason & EstimatorRejectReason.STEERING_PRESSED
  assert result.confidence == 0.0


def test_estimator_separates_positive_and_negative_coverage():
  estimator = AdaptiveTorqueEstimator(dt=0.01)
  estimator.update(make_observation(commanded_torque=0.25, actual_lateral_accel=0.60, desired_lateral_accel=0.65))
  result = estimator.update(make_observation(commanded_torque=-0.25, actual_lateral_accel=-0.60, desired_lateral_accel=-0.65))

  assert result.positive_coverage > 0.0
  assert result.negative_coverage > 0.0


def test_residual_spike_demotes_confidence():
  estimator = AdaptiveTorqueEstimator(dt=0.01)
  for _ in range(40):
    estimator.update(make_observation())
  before = estimator.state.confidence

  result = estimator.update(make_observation(actual_lateral_accel=-0.7))

  assert not result.sample_accepted
  assert result.reject_reason & EstimatorRejectReason.SIGN_CONFLICT
  assert result.confidence < before


def test_estimator_rejects_command_desired_sign_conflict():
  estimator = AdaptiveTorqueEstimator(dt=0.01)

  result = estimator.update(make_observation(commanded_torque=0.25, desired_lateral_accel=-0.65, actual_lateral_accel=-0.60))

  assert not result.sample_accepted
  assert result.reject_reason & EstimatorRejectReason.SIGN_CONFLICT


def test_estimator_ignores_tiny_lateral_sign_conflict():
  estimator = AdaptiveTorqueEstimator(dt=0.01)

  result = estimator.update(make_observation(commanded_torque=0.25, desired_lateral_accel=0.04, actual_lateral_accel=-0.04))

  assert not result.reject_reason & EstimatorRejectReason.SIGN_CONFLICT


def test_same_sign_residual_spike_demotes_confidence_without_sign_conflict():
  estimator = AdaptiveTorqueEstimator(dt=0.01)
  for _ in range(40):
    estimator.update(make_observation())
  before = estimator.state.confidence

  result = estimator.update(make_observation(actual_lateral_accel=1.8))

  assert not result.sample_accepted
  assert result.reject_reason & EstimatorRejectReason.RESIDUAL_SPIKE
  assert not result.reject_reason & EstimatorRejectReason.SIGN_CONFLICT
  assert result.confidence < before


def test_clipped_residual_spike_uses_slow_confidence_decay():
  estimator = AdaptiveTorqueEstimator(dt=0.01)
  for _ in range(40):
    estimator.update(make_observation())
  before = estimator.state.confidence

  result = estimator.update(make_observation(steer_limited_by_safety=True, actual_lateral_accel=1.8))

  assert not result.sample_accepted
  assert result.reject_reason & EstimatorRejectReason.STEER_LIMITED
  assert result.reject_reason & EstimatorRejectReason.RESIDUAL_SPIKE
  assert result.confidence == pytest.approx(before - CONFIDENCE_BUILD_RATE)


def test_steer_limited_saturated_sign_conflict_uses_slow_confidence_decay():
  estimator = AdaptiveTorqueEstimator(dt=0.01)
  for _ in range(40):
    estimator.update(make_observation())
  before = estimator.state.confidence

  result = estimator.update(make_observation(steer_limited_by_safety=True, saturated=True, actual_lateral_accel=-0.7))

  assert not result.sample_accepted
  assert result.reject_reason & EstimatorRejectReason.STEER_LIMITED
  assert result.reject_reason & EstimatorRejectReason.SATURATED
  assert result.reject_reason & EstimatorRejectReason.SIGN_CONFLICT
  assert result.confidence == pytest.approx(before - CONFIDENCE_BUILD_RATE)


def test_steer_limited_sign_conflict_without_saturation_uses_fast_confidence_decay():
  estimator = AdaptiveTorqueEstimator(dt=0.01)
  for _ in range(40):
    estimator.update(make_observation())
  before = estimator.state.confidence

  result = estimator.update(make_observation(steer_limited_by_safety=True, actual_lateral_accel=-0.7))

  assert not result.sample_accepted
  assert result.reject_reason & EstimatorRejectReason.STEER_LIMITED
  assert result.reject_reason & EstimatorRejectReason.SIGN_CONFLICT
  assert result.confidence < before - CONFIDENCE_BUILD_RATE


def test_result_params_are_snapshot_not_live_state():
  estimator = AdaptiveTorqueEstimator(dt=0.01)

  first = estimator.update(make_observation())
  first_factor = first.params.lat_accel_factor
  estimator.update(make_observation(commanded_torque=0.25, actual_lateral_accel=0.30, desired_lateral_accel=0.30))

  assert first.params.lat_accel_factor == first_factor
