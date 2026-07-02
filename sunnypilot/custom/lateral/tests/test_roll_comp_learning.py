import numpy as np
import pytest

from types import SimpleNamespace

from openpilot.sunnypilot.custom.lateral.roll_comp_learning import (
  ROLL_GAIN_MIN,
  ROLL_GAIN_MAX,
  MIN_CONFIDENCE,
  MIN_POINTS,
  RollCompBuckets,
  fit_roll_comp_profile,
  format_roll_comp_profile,
  parse_roll_comp_profile,
)


def cp():
  torque = SimpleNamespace(latAccelFactor=2.0, friction=0.2)
  return SimpleNamespace(carFingerprint="test", lateralTuning=SimpleNamespace(which=lambda: 'torque', torque=torque))


def _fill_buckets(buckets, n, gain=0.55, roll_min=-0.12, roll_max=0.12):
  rng = np.random.default_rng(0)
  rolls = roll_min + (roll_max - roll_min) * (0.5 + 0.5 * np.linspace(-1, 1, n) ** 3)
  for roll in rolls:
    x = -np.sin(roll) * 9.81
    torque_lat = gain * x
    # add small noise so SVD is well-conditioned
    torque_lat += rng.normal(scale=0.02)
    buckets.add_point(roll, torque_lat, 20.0)


def test_synthetic_slope_recovery():
  buckets = RollCompBuckets(points_per_bucket=MIN_POINTS * 2)
  _fill_buckets(buckets, MIN_POINTS * 2, gain=0.55)
  profile = fit_roll_comp_profile(cp(), buckets)
  assert profile is not None
  assert profile['gain'] == pytest.approx(0.55, abs=0.03)
  assert profile['points'] >= MIN_POINTS
  assert profile['span'] >= 0.25
  assert 0 <= profile['confidence'] <= 1


def test_slope_recovery_under_control_noise():
  # Regression for the TLS→OLS estimator fix: with y-noise comparable to the x span
  # (route-246 straight frames: y std 0.149 vs x std 0.142), the previous TLS fit
  # read the 0.55 vehicle response as ~1.15 and would have learned a wrong gain.
  buckets = RollCompBuckets(points_per_bucket=MIN_POINTS * 2)
  rng = np.random.default_rng(1)
  rolls = np.linspace(-0.06, 0.06, MIN_POINTS * 2)
  for roll in rolls:
    x = -np.sin(roll) * 9.81
    buckets.add_point(roll, 0.55 * x + rng.normal(scale=0.13), 20.0)
  profile = fit_roll_comp_profile(cp(), buckets)
  assert profile is not None
  assert profile['gain'] == pytest.approx(0.55, abs=0.05)


def test_clamp_low_gain():
  buckets = RollCompBuckets(points_per_bucket=MIN_POINTS * 2)
  _fill_buckets(buckets, MIN_POINTS * 2, gain=0.1)
  profile = fit_roll_comp_profile(cp(), buckets)
  assert profile is not None
  assert profile['gain'] == ROLL_GAIN_MIN


def test_clamp_high_gain():
  buckets = RollCompBuckets(points_per_bucket=MIN_POINTS * 2)
  _fill_buckets(buckets, MIN_POINTS * 2, gain=2.0)
  profile = fit_roll_comp_profile(cp(), buckets)
  assert profile is not None
  assert profile['gain'] == ROLL_GAIN_MAX


def test_span_rejection():
  buckets = RollCompBuckets(points_per_bucket=MIN_POINTS * 2)
  _fill_buckets(buckets, MIN_POINTS * 2, gain=0.55, roll_min=-0.01, roll_max=0.01)
  assert fit_roll_comp_profile(cp(), buckets) is None


def test_one_crown_rejection():
  buckets = RollCompBuckets(points_per_bucket=MIN_POINTS * 2)
  # only positive roll -> only negative x values
  rolls = np.linspace(0.02, 0.12, MIN_POINTS * 2)
  for roll in rolls:
    x = -np.sin(roll) * 9.81
    buckets.add_point(roll, 0.55 * x, 20.0)
  assert fit_roll_comp_profile(cp(), buckets) is None


def test_sparse_opposite_crown_outlier_rejection():
  buckets = RollCompBuckets(points_per_bucket=MIN_POINTS * 2)
  rolls = np.linspace(0.02, 0.12, MIN_POINTS * 2 - 1)
  for roll in rolls:
    x = -np.sin(roll) * 9.81
    buckets.add_point(roll, 0.55 * x, 20.0)
  buckets.add_point(-0.12, 0.55 * (-np.sin(-0.12) * 9.81), 20.0)
  assert fit_roll_comp_profile(cp(), buckets) is None


def test_low_points_rejection():
  buckets = RollCompBuckets(points_per_bucket=500)
  _fill_buckets(buckets, 100, gain=0.55)
  assert fit_roll_comp_profile(cp(), buckets) is None


def test_parse_rejects_wrong_restore_key():
  buckets = RollCompBuckets(points_per_bucket=MIN_POINTS * 2)
  _fill_buckets(buckets, MIN_POINTS * 2, gain=0.55)
  profile = fit_roll_comp_profile(cp(), buckets)
  assert profile is not None
  parsed = parse_roll_comp_profile(cp(), profile)
  assert parsed is not None

  bad_cp = SimpleNamespace(
    carFingerprint="other",
    lateralTuning=SimpleNamespace(which=lambda: 'torque', torque=SimpleNamespace(latAccelFactor=2.0, friction=0.2)),
  )
  assert parse_roll_comp_profile(bad_cp, profile) is None


def test_parse_rejects_malformed_and_nan():
  buckets = RollCompBuckets(points_per_bucket=MIN_POINTS * 2)
  _fill_buckets(buckets, MIN_POINTS * 2, gain=0.55)
  profile = fit_roll_comp_profile(cp(), buckets)
  assert profile is not None

  assert parse_roll_comp_profile(cp(), {**profile, 'gain': float('nan')}) is None
  assert parse_roll_comp_profile(cp(), {**profile, 'gain': 2.0}) is None
  assert parse_roll_comp_profile(cp(), {**profile, 'gain': 0.1}) is None
  assert parse_roll_comp_profile(cp(), {**profile, 'confidence': 1.5}) is None
  assert parse_roll_comp_profile(cp(), {**profile, 'confidence': MIN_CONFIDENCE - 1e-3}) is None
  assert parse_roll_comp_profile(cp(), {**profile, 'confidence': 0.0}) is None
  assert parse_roll_comp_profile(cp(), {**profile, 'points': -1}) is None
  assert parse_roll_comp_profile(cp(), {**profile, 'points': MIN_POINTS - 1}) is None
  assert parse_roll_comp_profile(cp(), {**profile, 'span': 0.1}) is None
  assert parse_roll_comp_profile(cp(), {**profile, 'version': 2}) is None


def test_format_rejects_non_finite_json():
  with pytest.raises(ValueError):
    format_roll_comp_profile({'bad': float('nan')})


def test_parse_rejects_missing_fields():
  buckets = RollCompBuckets(points_per_bucket=MIN_POINTS * 2)
  _fill_buckets(buckets, MIN_POINTS * 2, gain=0.55)
  profile = fit_roll_comp_profile(cp(), buckets)
  assert profile is not None
  for key in ('gain', 'points', 'span', 'confidence'):
    incomplete = {k: v for k, v in profile.items() if k != key}
    incomplete['version'] = profile['version']
    incomplete['restoreKey'] = profile['restoreKey']
    assert parse_roll_comp_profile(cp(), incomplete) is None
