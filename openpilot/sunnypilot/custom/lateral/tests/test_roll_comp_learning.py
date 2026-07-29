import json
import numpy as np
import pytest

from types import SimpleNamespace

from openpilot.sunnypilot.custom.lateral.roll_comp_learning import (
  blend_roll_comp_profile,
  ROLL_GAIN_MIN,
  ROLL_GAIN_MAX,
  MIN_CONFIDENCE,
  MIN_POINTS,
  RollCompBuckets,
  fit_roll_comp_profile,
  format_roll_comp_profile,
  parse_roll_comp_profile,
  roll_gain_at,
)


def cp():
  torque = SimpleNamespace(latAccelFactor=2.0, friction=0.2)
  return SimpleNamespace(carFingerprint="test", lateralTuning=SimpleNamespace(which=lambda: 'torque', torque=torque))


def _fill_buckets(buckets, n, gain=0.55, roll_min=-0.12, roll_max=0.12, v_ego=20.0):
  rng = np.random.default_rng(0)
  rolls = roll_min + (roll_max - roll_min) * (0.5 + 0.5 * np.linspace(-1, 1, n) ** 3)
  for roll in rolls:
    x = -np.sin(roll) * 9.81
    torque_lat = gain * x
    # add small noise so SVD is well-conditioned
    torque_lat += rng.normal(scale=0.02)
    buckets.add_point(roll, torque_lat, v_ego)


def _restore_key():
  car = cp()
  torque = car.lateralTuning.torque
  return {
    'carFingerprint': car.carFingerprint,
    'lateralTuning': car.lateralTuning.which(),
    'latAccelFactor': float(torque.latAccelFactor),
    'friction': float(torque.friction),
  }


def _profile(gain, points, span, confidence):
  return {
    'version': 1,
    'restoreKey': _restore_key(),
    'gain': gain,
    'points': points,
    'span': span,
    'confidence': confidence,
  }


def _banded_profile(bands):
  return {
    'version': 1,
    'restoreKey': _restore_key(),
    'bands': [{'vLo': lo, 'vHi': hi, 'gain': g, 'points': 6000, 'span': 0.5, 'confidence': 1.0}
              for lo, hi, g in bands],
  }


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


def test_band_routing():
  buckets = RollCompBuckets()
  buckets.add_point(0.05, 0.3, 7.0)
  buckets.add_point(0.05, 0.3, 12.0)
  buckets.add_point(0.05, 0.3, 20.0)
  buckets.add_point(0.05, 0.3, 4.0)    # below the collection floor: dropped
  buckets.add_point(0.05, 0.3, 150.0)  # beyond the top band: dropped
  assert len(buckets.band_points((5.0, 10.0))) == 1
  assert len(buckets.band_points((10.0, 15.0))) == 1
  assert len(buckets.band_points((15.0, 100.0))) == 1
  assert len(buckets.get_points()) == 3


def test_city_only_profile_fits_and_pins_highway_to_base():
  buckets = RollCompBuckets(points_per_bucket=MIN_POINTS * 2)
  _fill_buckets(buckets, MIN_POINTS * 2, gain=0.4, v_ego=7.0)
  profile = fit_roll_comp_profile(cp(), buckets)
  assert profile is not None
  assert 'gain' not in profile  # no primary band -> no top-level mirror

  parsed = parse_roll_comp_profile(cp(), json.loads(format_roll_comp_profile(profile)))
  assert parsed is not None
  assert [(b['vLo'], b['vHi']) for b in parsed['bands']] == [(5.0, 10.0)]
  low_gain = parsed['bands'][0]['gain']
  assert low_gain == pytest.approx(0.4, abs=0.03)
  assert roll_gain_at(parsed, 7.5, 0.55) == pytest.approx(low_gain)
  # the unfitted highway band is pinned to the base gain, never flat-extended
  assert roll_gain_at(parsed, 25.0, 0.55) == pytest.approx(0.55)


def test_full_profile_interpolates_between_bands():
  profile = _banded_profile([(5.0, 10.0, 0.4), (15.0, 100.0, 0.7)])
  assert roll_gain_at(profile, 3.0, 0.55) == pytest.approx(0.4)   # flat below the lowest anchor
  assert roll_gain_at(profile, 7.5, 0.55) == pytest.approx(0.4)
  assert roll_gain_at(profile, 20.0, 0.55) == pytest.approx(0.7)
  assert roll_gain_at(profile, 35.0, 0.55) == pytest.approx(0.7)  # flat above the top anchor
  # linear between the 7.5 and 20.0 anchors
  assert roll_gain_at(profile, 13.75, 0.55) == pytest.approx(0.55)


def test_gain_at_is_continuous_over_speed():
  profile = _banded_profile([(5.0, 10.0, 0.35), (10.0, 15.0, 0.5), (15.0, 100.0, 0.8)])
  vs = np.linspace(0.0, 45.0, 4501)
  gains = np.array([roll_gain_at(profile, v, 0.55) for v in vs])
  assert np.max(np.abs(np.diff(gains))) < 0.005  # no step anywhere on the speed axis


def test_gain_at_scalar_profile_reproduces_constant_gain():
  parsed = parse_roll_comp_profile(cp(), _profile(0.62, 6000, 0.5, 1.0))
  assert parsed is not None
  for v in (0.0, 7.0, 14.0, 25.0, 40.0):
    assert roll_gain_at(parsed, v, 0.55) == pytest.approx(0.62)


def test_blend_carries_forward_unrefit_bands():
  old = _banded_profile([(5.0, 10.0, 0.4)])
  new = _banded_profile([(15.0, 100.0, 0.7)])
  blended = blend_roll_comp_profile(old, new)
  assert [(b['vLo'], b['vHi']) for b in blended['bands']] == [(5.0, 10.0), (15.0, 100.0)]
  assert blended['bands'][0]['gain'] == pytest.approx(0.4)
  assert blended['gain'] == pytest.approx(0.7)  # mirror follows the primary band


def test_parse_legacy_scalar_payload_maps_to_primary_band():
  parsed = parse_roll_comp_profile(cp(), _profile(0.55, 6000, 0.5, 1.0))
  assert parsed is not None
  assert [(b['vLo'], b['vHi']) for b in parsed['bands']] == [(15.0, 100.0)]
  assert parsed['gain'] == pytest.approx(0.55)


def test_parse_rejects_malformed_band_entries():
  assert parse_roll_comp_profile(cp(), _banded_profile([(5.0, 10.0, 0.4)])) is not None
  assert parse_roll_comp_profile(cp(), _banded_profile([(5.0, 10.0, float('nan'))])) is None
  assert parse_roll_comp_profile(cp(), _banded_profile([(10.0, 5.0, 0.4)])) is None
  assert parse_roll_comp_profile(cp(), _banded_profile([(5.0, 10.0, 0.4), (5.0, 10.0, 0.5)])) is None

  missing_vlo = _banded_profile([(5.0, 10.0, 0.4)])
  del missing_vlo['bands'][0]['vLo']
  assert parse_roll_comp_profile(cp(), missing_vlo) is None

  low_points = _banded_profile([(5.0, 10.0, 0.4)])
  low_points['bands'][0]['points'] = MIN_POINTS - 1
  assert parse_roll_comp_profile(cp(), low_points) is None


def test_blend_profile_caps_and_round_trips():
  old = _profile(0.4, 6000, 0.4, 1.0)
  new = _profile(0.8, 3000, 0.6, 0.75)
  blended = blend_roll_comp_profile(old, new)
  old_weight = min(old['points'], 2 * MIN_POINTS)
  new_weight = min(new['points'], 2 * MIN_POINTS)

  assert blended['gain'] == pytest.approx((old_weight * old['gain'] + new_weight * new['gain']) / (old_weight + new_weight))
  assert blended['points'] == 4 * MIN_POINTS
  assert blended['span'] == pytest.approx(max(old['span'], new['span']))
  assert blended['confidence'] == pytest.approx(1.0)

  parsed = parse_roll_comp_profile(cp(), json.loads(format_roll_comp_profile(blended)))
  assert parsed is not None
  assert parsed['gain'] == pytest.approx(blended['gain'])
  assert parsed['points'] == blended['points']
  assert parsed['span'] == pytest.approx(blended['span'])
  assert parsed['confidence'] == pytest.approx(blended['confidence'])
