"""Tests for the session-only LateralVehicleHealthEstimator."""

import math

import pytest

from openpilot.selfdrive.controls.lib.lateral_vehicle_health_estimator import (
  HEALTH_EST_BIAS_ALPHA,
  HEALTH_EST_BIAS_MAX,
  HEALTH_EST_BIAS_WARNING,
  HEALTH_EST_MIN_PERSISTENCE_FRAMES,
  HEALTH_EST_MIN_SPEED,
  LateralVehicleHealthEstimate,
  LateralVehicleHealthEstimator,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def estimator():
  return LateralVehicleHealthEstimator(dt=0.01)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _valid_frame(*, v_ego=20.0, target_lat=0.0, actual_lat=0.0,
                 path_quality=1.0, demand_source="model_path",
                 lane_change_active=False, steering_pressed=False,
                 curvature_limited=False, saturated=False):
  """Return a dictionary of keyword arguments for estimator.update()."""
  return dict(
    v_ego=v_ego,
    target_lateral_accel=target_lat,
    actual_lateral_accel=actual_lat,
    path_quality=path_quality,
    demand_source=demand_source,
    lane_change_active=lane_change_active,
    steering_pressed=steering_pressed,
    curvature_limited=curvature_limited,
    saturated=saturated,
  )


def _feed_frames(estimator, n, *, bias=0.0, **kw):
  """Feed *n* frames into the estimator, optionally adding a constant bias
  to the actual lateral accel.  Returns the last estimate."""
  est = None
  for _ in range(n):
    args = _valid_frame(**kw)
    args["actual_lateral_accel"] = args.get("actual_lateral_accel", 0.0) + bias
    est = estimator.update(**args)
  return est


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLateralVehicleHealthEstimator:

  def test_bias_estimation_on_straight_road(self, estimator):
    """Feed constant positive lateral accel bias at high speed with good
    path quality, verify bias estimate converges toward the bias."""
    true_bias = 0.03  # m/s²
    n_frames = HEALTH_EST_MIN_PERSISTENCE_FRAMES + 50

    est = _feed_frames(estimator, n_frames, bias=true_bias)

    # The EMA should have converged close to the true bias
    assert est.learning_active
    assert est.bias_confidence == pytest.approx(1.0, abs=0.01)
    assert abs(est.bias_estimate - true_bias) < 0.01, (
      f"Bias {est.bias_estimate:.4f} should be close to {true_bias:.4f}"
    )
    assert not est.bias_warning, "0.03 should be below HEALTH_EST_BIAS_WARNING"

  def test_bias_warning_at_large_bias(self, estimator):
    """Feed large bias (>0.04 m/s²), verify bias_warning is True."""
    large_bias = 0.05
    n_frames = HEALTH_EST_MIN_PERSISTENCE_FRAMES

    est = _feed_frames(estimator, n_frames, bias=large_bias)
    assert est.bias_warning, f"Bias {large_bias} should trigger warning"
    assert abs(est.bias_estimate) > HEALTH_EST_BIAS_WARNING or est.bias_estimate >= 0.04

  def test_no_learning_at_low_speed(self, estimator):
    """Feed data below 15 m/s, verify learning_active is False and
    bias does not update."""
    n_frames = HEALTH_EST_MIN_PERSISTENCE_FRAMES
    est = _feed_frames(estimator, n_frames, v_ego=10.0, bias=0.03)
    assert not est.learning_active
    # Bias should still be 0 since we never learned
    assert est.bias_estimate == 0.0
    assert est.persistence_frames == 0

  def test_no_learning_during_lane_change(self, estimator):
    """Feed data with lane_change_active=True, verify learning_active is False."""
    n_frames = HEALTH_EST_MIN_PERSISTENCE_FRAMES
    est = _feed_frames(estimator, n_frames, bias=0.02,
                        lane_change_active=True)
    assert not est.learning_active
    assert est.bias_estimate == 0.0

  def test_no_learning_with_steering_input(self, estimator):
    """Feed data with steering_pressed=True, verify learning_active is False."""
    n_frames = HEALTH_EST_MIN_PERSISTENCE_FRAMES
    est = _feed_frames(estimator, n_frames, bias=0.02,
                        steering_pressed=True)
    assert not est.learning_active
    assert est.bias_estimate == 0.0

  def test_response_asymmetry_detection(self, estimator):
    """Feed left turns with higher response than right turns, verify
    response_asymmetry > 0."""
    n_frames = HEALTH_EST_MIN_PERSISTENCE_FRAMES
    # Left turns (target_lat > 0): actual is 90% of target (under-response)
    # Right turns (target_lat < 0): actual is 60% of target (more under-response)
    # This creates an asymmetry.

    for i in range(n_frames):
      if i % 2 == 0:
        # Left turn: target 2.0 m/s², actual 1.8 m/s² (ratio 0.9)
        est = estimator.update(**_valid_frame(target_lat=2.0, actual_lat=1.8))
      else:
        # Right turn: target -2.0 m/s², actual -1.2 m/s² (ratio 0.6)
        est = estimator.update(**_valid_frame(target_lat=-2.0, actual_lat=-1.2))

    assert est.learning_active
    assert est.response_asymmetry > 0.05, (
      f"Asymmetry {est.response_asymmetry:.4f} should be > 0.05"
    )
    assert est.left_response_estimate > est.right_response_estimate

  def test_recenter_lag_detection(self, estimator):
    """Feed target crossing zero followed by actual crossing zero after
    a few frames, verify recenter_lag_frames is detected."""
    # Phase 1: positive target and positive actual
    for _ in range(10):
      estimator.update(**_valid_frame(target_lat=1.0, actual_lat=1.0))

    # Phase 2: target crosses zero via a transition frame within the
    # deadband [-0.05, 0.05] so that current_target_sign becomes 0,
    # triggering the zero-crossing recording.
    estimator.update(**_valid_frame(target_lat=0.01, actual_lat=1.0))

    # Phase 3: target now negative, actual still positive (lag)
    for _ in range(5):
      estimator.update(**_valid_frame(target_lat=-0.5, actual_lat=0.5))

    # Phase 4: actual crosses zero too (within deadband)
    for _ in range(3):
      estimator.update(**_valid_frame(target_lat=-0.5, actual_lat=0.01))

    est = estimator.update(**_valid_frame(target_lat=-0.5, actual_lat=0.01))

    # recenter_lag should be > 0 (we expect at least a few frames lag)
    assert est.recenter_lag_frames > 0, (
      f"Recenter lag should be > 0, got {est.recenter_lag_frames}"
    )

  def test_reset_clears_all_estimates(self, estimator):
    """Feed data, call reset(), verify all estimates return to defaults."""
    _feed_frames(estimator, HEALTH_EST_MIN_PERSISTENCE_FRAMES, bias=0.03)
    # Verify some state was accumulated
    est_before = estimator.update(**_valid_frame())
    assert est_before.bias_estimate != 0.0 or est_before.persistence_frames > 0

    estimator.reset()
    default = LateralVehicleHealthEstimate()
    est_after = estimator.update(**_valid_frame())
    # After reset, first update has learning conditions met but no history.
    # bias_confidence = 1 / HEALTH_EST_MIN_PERSISTENCE_FRAMES = 0.01
    assert est_after.bias_estimate == pytest.approx(0.0)
    assert est_after.bias_confidence == pytest.approx(1.0 / HEALTH_EST_MIN_PERSISTENCE_FRAMES)
    assert not est_after.bias_warning
    assert est_after.left_response_estimate == 0.0
    assert est_after.right_response_estimate == 0.0
    assert est_after.response_asymmetry == 0.0
    assert est_after.recenter_lag_frames == 0
    assert est_after.persistence_frames == 1  # one update after reset
    assert est_after.learning_active

  def test_bias_bounds_clamp(self, estimator):
    """Feed very large bias, verify bias_estimate is clamped to ±0.06."""
    huge_bias = 10.0
    n_frames = HEALTH_EST_MIN_PERSISTENCE_FRAMES
    est = _feed_frames(estimator, n_frames, bias=huge_bias)
    assert abs(est.bias_estimate) <= HEALTH_EST_BIAS_MAX, (
      f"Bias {est.bias_estimate} should be clamped to ±{HEALTH_EST_BIAS_MAX}"
    )
    assert est.bias_estimate == pytest.approx(HEALTH_EST_BIAS_MAX, abs=0.01)
