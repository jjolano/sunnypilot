import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from openpilot.sunnypilot.custom.lateral.block_jackknife import MAX_BLOCK_REL_SE, fit_block_slope
from openpilot.sunnypilot.custom.lateral.roll_comp_learning import (
  MIN_BLOCK_X_SPAN,
  MIN_POINTS,
  MIN_POINTS_PER_BLOCK,
  MIN_X_SPAN,
  ROLL_COMP_PARAMS_VERSION,
  ROLL_COMP_PRIMARY_BAND,
  ROLL_COMP_SPEED_BANDS,
  ROLL_GAIN_MAX,
  ROLL_GAIN_MIN,
  RollCompBuckets,
  fit_roll_comp_profile,
  format_roll_comp_profile,
  parse_roll_comp_profile,
  roll_gain_at,
  replace_roll_comp_profile,
)
from openpilot.sunnypilot.custom.lateral.speed_aware_torque import _restore_key


def cp():
  torque = SimpleNamespace(latAccelFactor=2.0, friction=0.2)
  return SimpleNamespace(carFingerprint="test", lateralTuning=SimpleNamespace(which=lambda: 'torque', torque=torque))


def _roll_for_x(x):
  return -float(np.arcsin(x / 9.81))


def _add_roll_block(buckets, block_id, slope=0.55, count=200, xs=None, v_ego=20.0):
  if xs is None:
    xs = np.linspace(-0.4, 0.4, count)
  for i in range(count):
    x = float(xs[i % len(xs)])
    buckets.add_point(_roll_for_x(x), slope * x + 0.25 * block_id, v_ego, block_id)


def _clean_roll_buckets(blocks=12):
  buckets = RollCompBuckets()
  for block_id in range(blocks):
    _add_roll_block(buckets, block_id)
  buckets.set_completed_through(blocks - 1)
  return buckets


def _band(lo, hi, gain=0.55, *, points=MIN_POINTS, block_count=12, slope_rel_se=0.05):
  return {
    'vLo': lo,
    'vHi': hi,
    'gain': gain,
    'points': points,
    'span': 0.5,
    'confidence': min(1.0, block_count / 24),
    'blockCount': block_count,
    'slopeRelSe': slope_rel_se,
  }


def _profile(bands):
  profile = {
    'version': ROLL_COMP_PARAMS_VERSION,
    'restoreKey': _restore_key(cp()),
    'bands': copy.deepcopy(bands),
  }
  primary = next((band for band in bands if (band['vLo'], band['vHi']) == ROLL_COMP_PRIMARY_BAND), None)
  if primary is not None:
    for field in ('gain', 'points', 'span', 'confidence', 'blockCount', 'slopeRelSe'):
      profile[field] = primary[field]
  return profile


def test_clean_twelve_completed_blocks_fit_and_incomplete_block_is_ignored():
  buckets = _clean_roll_buckets()
  _add_roll_block(buckets, 12, slope=0.9, count=400)

  profile = fit_roll_comp_profile(cp(), buckets)

  assert profile is not None
  assert profile['gain'] == pytest.approx(0.55)
  assert profile['points'] == 12 * 200
  assert profile['blockCount'] == 12
  assert profile['slopeRelSe'] == pytest.approx(0.0)
  assert profile['gain'] == profile['bands'][0]['gain']


def test_repeated_samples_in_one_block_do_not_pass_block_gate():
  buckets = RollCompBuckets()
  _add_roll_block(buckets, 0, count=MIN_POINTS)
  buckets.set_completed_through(0)
  points = buckets.band_points(ROLL_COMP_PRIMARY_BAND)

  assert len(points) >= MIN_POINTS
  assert float(np.percentile(points[:, 0], 95) - np.percentile(points[:, 0], 5)) >= MIN_X_SPAN
  assert fit_roll_comp_profile(cp(), buckets) is None


def test_low_total_points_are_rejected_even_with_informative_blocks():
  buckets = RollCompBuckets()
  for block_id in range(12):
    _add_roll_block(buckets, block_id, count=100)
  buckets.set_completed_through(11)
  assert len(buckets.band_points(ROLL_COMP_PRIMARY_BAND)) < MIN_POINTS
  assert fit_roll_comp_profile(cp(), buckets) is None


def test_correlated_block_variation_rejects_despite_old_point_and_span_gates():
  buckets = RollCompBuckets()
  _add_roll_block(buckets, 0, slope=0.3, count=1000,
                  xs=np.array([-0.45, -0.30, -0.10, 0.10, 0.30, 0.45]))
  for block_id in range(1, 12):
    _add_roll_block(buckets, block_id, slope=1.0, count=100,
                    xs=np.array([-0.05, 0.05]))
  buckets.set_completed_through(11)
  points = buckets.band_points(ROLL_COMP_PRIMARY_BAND)
  fit = fit_block_slope(points)

  assert len(points) >= MIN_POINTS
  assert float(np.percentile(points[:, 0], 95) - np.percentile(points[:, 0], 5)) >= MIN_X_SPAN
  assert fit is not None and fit.rel_se > MAX_BLOCK_REL_SE
  assert fit_roll_comp_profile(cp(), buckets) is None


def test_deterministic_measurement_noise_is_handled_by_centered_fit():
  buckets = RollCompBuckets()
  xs = np.linspace(-0.4, 0.4, 200)
  for block_id in range(12):
    for index, x in enumerate(xs):
      noise = 0.01 * ((index % 11) - 5)
      buckets.add_point(_roll_for_x(float(x)), 0.55 * x + noise + block_id, 20.0, block_id)
  buckets.set_completed_through(11)

  profile = fit_roll_comp_profile(cp(), buckets)

  assert profile is not None
  assert profile['gain'] == pytest.approx(0.55, abs=0.02)


def test_discarded_blocks_cannot_supply_informative_count():
  buckets = RollCompBuckets()
  for block_id in range(11):
    _add_roll_block(buckets, block_id)
  _add_roll_block(buckets, 11, count=MIN_POINTS, xs=np.zeros(MIN_POINTS))
  buckets.set_completed_through(11)
  points = buckets.band_points(ROLL_COMP_PRIMARY_BAND)

  assert len(points) >= MIN_POINTS
  assert fit_roll_comp_profile(cp(), buckets) is None


def test_discarded_blocks_cannot_supply_span_or_opposite_crown_sign():
  narrow = RollCompBuckets()
  for block_id in range(12):
    _add_roll_block(narrow, block_id, count=20, xs=np.linspace(0.02, 0.08, 20))
  _add_roll_block(narrow, 12, count=MIN_POINTS, xs=np.linspace(-0.45, 0.45, MIN_POINTS))
  narrow.set_completed_through(12)
  assert len(narrow.band_points(ROLL_COMP_PRIMARY_BAND)) >= MIN_POINTS
  assert np.ptp(narrow.band_points(ROLL_COMP_PRIMARY_BAND)[:, 0]) >= MIN_X_SPAN
  assert fit_roll_comp_profile(cp(), narrow) is None

  one_crown = RollCompBuckets()
  for block_id in range(12):
    _add_roll_block(one_crown, block_id, count=200, xs=np.linspace(0.3, 0.45, 200))
  _add_roll_block(one_crown, 12, count=MIN_POINTS, xs=np.linspace(-0.45, 0.45, MIN_POINTS))
  one_crown.set_completed_through(11)
  raw = one_crown.band_points(ROLL_COMP_PRIMARY_BAND)
  assert raw[:, 0].min() < 0.0 < raw[:, 0].max()
  assert fit_roll_comp_profile(cp(), one_crown) is None


def test_persisted_roll_counts_include_only_informative_blocks():
  buckets = _clean_roll_buckets()
  _add_roll_block(buckets, 12, count=MIN_POINTS, xs=np.zeros(MIN_POINTS))
  buckets.set_completed_through(12)

  profile = fit_roll_comp_profile(cp(), buckets)

  assert profile is not None
  assert profile['points'] == 12 * 200
  assert profile['blockCount'] == 12


def test_negative_informative_block_is_rejected():
  buckets = _clean_roll_buckets()
  _add_roll_block(buckets, 11, slope=-0.55)
  assert fit_roll_comp_profile(cp(), buckets) is None


def test_full_slope_out_of_range_is_rejected():
  buckets = RollCompBuckets()
  for block_id in range(12):
    _add_roll_block(buckets, block_id, slope=ROLL_GAIN_MAX + 0.1)
  buckets.set_completed_through(11)
  assert fit_roll_comp_profile(cp(), buckets) is None


def test_leave_one_out_slope_out_of_range_is_rejected():
  buckets = RollCompBuckets()
  _add_roll_block(buckets, 0, slope=0.55, count=1000,
                  xs=np.array([-0.45, -0.30, -0.10, 0.10, 0.30, 0.45]))
  for block_id in range(1, 12):
    _add_roll_block(buckets, block_id, slope=1.2, count=100,
                    xs=np.array([-0.05, 0.05]))
  buckets.set_completed_through(11)
  points = buckets.band_points(ROLL_COMP_PRIMARY_BAND)
  fit = fit_block_slope(points)

  assert fit is not None
  assert ROLL_GAIN_MIN <= fit.slope <= ROLL_GAIN_MAX
  assert fit.loo_slopes[0] > ROLL_GAIN_MAX
  assert fit_roll_comp_profile(cp(), buckets) is None


def test_parser_accepts_exact_configured_subset_and_rejects_unknown_band():
  low = _profile([_band(5.0, 10.0, gain=0.4)])
  assert parse_roll_comp_profile(cp(), json.loads(format_roll_comp_profile(low))) is not None

  unknown = _profile([_band(5.0, 11.0, gain=0.4)])
  assert parse_roll_comp_profile(cp(), unknown) is None


def test_roll_band_routing_preserves_all_configured_speed_bands():
  buckets = RollCompBuckets()
  buckets.add_point(0.05, 0.3, 7.0)
  buckets.add_point(0.05, 0.3, 12.0)
  buckets.add_point(0.05, 0.3, 20.0)
  buckets.add_point(0.05, 0.3, 4.0)
  buckets.add_point(0.05, 0.3, 150.0)
  assert [len(buckets.band_points(band)) for band in ROLL_COMP_SPEED_BANDS] == [1, 1, 1]
  assert len(buckets.get_points()) == 3


@pytest.mark.parametrize('field', ['blockCount', 'slopeRelSe', 'points', 'span', 'confidence'])
def test_parser_rejects_missing_nonfinite_or_inconsistent_fields(field):
  payload = _profile([_band(*ROLL_COMP_PRIMARY_BAND)])
  missing = copy.deepcopy(payload)
  del missing['bands'][0][field]
  assert parse_roll_comp_profile(cp(), missing) is None

  bad = copy.deepcopy(payload)
  bad['bands'][0][field] = float('nan') if field not in ('blockCount', 'points') else -1
  assert parse_roll_comp_profile(cp(), bad) is None


def test_parser_rejects_old_version_and_mismatched_top_level_mirrors():
  payload = _profile([_band(*ROLL_COMP_PRIMARY_BAND)])
  assert parse_roll_comp_profile(cp(), {**payload, 'version': 1}) is None

  mismatch = copy.deepcopy(payload)
  mismatch['gain'] += 0.01
  assert parse_roll_comp_profile(cp(), mismatch) is None

  extra = copy.deepcopy(payload)
  extra['blockCount'] += 1
  assert parse_roll_comp_profile(cp(), extra) is None

  foreign = copy.deepcopy(payload)
  foreign['restoreKey'] = {**foreign['restoreKey'], 'carFingerprint': 'other'}
  assert parse_roll_comp_profile(cp(), foreign) is None


def test_roll_parser_rejects_points_below_block_point_floor():
  payload = _profile([_band(*ROLL_COMP_PRIMARY_BAND, points=MIN_POINTS, block_count=101)])
  payload['confidence'] = payload['bands'][0]['confidence'] = 1.0
  assert parse_roll_comp_profile(cp(), payload) is None


def test_replace_roll_snapshot_carries_unrefitted_band_and_rejects_large_delta():
  old = _profile([_band(5.0, 10.0, gain=0.4), _band(*ROLL_COMP_PRIMARY_BAND, gain=0.6)])
  new = _profile([_band(*ROLL_COMP_PRIMARY_BAND, gain=0.7)])

  replaced = replace_roll_comp_profile(old, new)

  assert {(band['vLo'], band['vHi']) for band in replaced['bands']} == {
    (5.0, 10.0), ROLL_COMP_PRIMARY_BAND,
  }
  assert replaced['bands'][0]['gain'] == pytest.approx(0.4)
  assert replaced['gain'] == pytest.approx(0.6)


def test_roll_payload_is_finite_and_round_trips():
  profile = _profile([_band(*ROLL_COMP_PRIMARY_BAND)])
  encoded = format_roll_comp_profile(profile)
  assert 'NaN' not in encoded
  assert parse_roll_comp_profile(cp(), json.loads(encoded)) == profile


def test_parser_rejects_malformed_band_entries():
  valid = _profile([_band(5.0, 10.0, gain=0.4)])
  assert parse_roll_comp_profile(cp(), valid) is not None

  nan_gain = copy.deepcopy(valid)
  nan_gain['bands'][0]['gain'] = float('nan')
  assert parse_roll_comp_profile(cp(), nan_gain) is None

  reversed_band = copy.deepcopy(valid)
  reversed_band['bands'][0]['vLo'], reversed_band['bands'][0]['vHi'] = 10.0, 5.0
  assert parse_roll_comp_profile(cp(), reversed_band) is None

  duplicate = _profile([_band(5.0, 10.0, gain=0.4), _band(5.0, 10.0, gain=0.5)])
  assert parse_roll_comp_profile(cp(), duplicate) is None

  missing_vlo = copy.deepcopy(valid)
  del missing_vlo['bands'][0]['vLo']
  assert parse_roll_comp_profile(cp(), missing_vlo) is None

  low_points = copy.deepcopy(valid)
  low_points['bands'][0]['points'] = MIN_POINTS - 1
  assert parse_roll_comp_profile(cp(), low_points) is None


def test_format_roll_profile_rejects_nonfinite_json():
  with pytest.raises(ValueError):
    format_roll_comp_profile({'bad': float('nan')})


def test_city_only_fit_does_not_flat_extend_to_unlearned_highway():
  buckets = RollCompBuckets()
  for block_id in range(12):
    _add_roll_block(buckets, block_id, v_ego=7.0)
  buckets.set_completed_through(11)
  profile = fit_roll_comp_profile(cp(), buckets)

  assert profile is not None and 'gain' not in profile
  assert profile['bands'][0]['gain'] == pytest.approx(0.55)
  assert roll_gain_at(profile, 7.5, 0.55) == pytest.approx(0.55)
  assert roll_gain_at(profile, 25.0, 0.55) == pytest.approx(0.55)


@pytest.mark.parametrize('bands, expected', [
  ([_band(*ROLL_COMP_SPEED_BANDS[1], gain=0.65)], (0.55, 0.65, 0.55, 0.55)),
  ([_band(*ROLL_COMP_PRIMARY_BAND, gain=0.75)], (0.55, 0.55, 0.75, 0.75)),
  ([_band(*ROLL_COMP_SPEED_BANDS[0], gain=0.4)], (0.4, 0.55, 0.55, 0.55)),
  ([_band(*ROLL_COMP_SPEED_BANDS[0], gain=0.4), _band(*ROLL_COMP_SPEED_BANDS[1], gain=0.65),
    _band(*ROLL_COMP_PRIMARY_BAND, gain=0.8)], (0.4, 0.65, 0.8, 0.8)),
])
def test_partial_roll_profiles_use_fixed_base_anchors(bands, expected):
  profile = _profile(bands)

  assert roll_gain_at(profile, 7.5, 0.55) == pytest.approx(expected[0])
  assert roll_gain_at(profile, 12.5, 0.55) == pytest.approx(expected[1])
  assert roll_gain_at(profile, 20.0, 0.55) == pytest.approx(expected[2])
  assert roll_gain_at(profile, 30.0, 0.55) == pytest.approx(expected[3])


def test_roll_speed_interpolation_has_no_threshold_step():
  profile = _profile([
    _band(*ROLL_COMP_SPEED_BANDS[0], gain=0.4),
    _band(*ROLL_COMP_SPEED_BANDS[1], gain=0.5),
    _band(*ROLL_COMP_PRIMARY_BAND, gain=0.8),
  ])
  speeds = np.linspace(0.0, 45.0, 4501)
  gains = np.array([roll_gain_at(profile, speed, 0.55) for speed in speeds])
  assert np.max(np.abs(np.diff(gains))) < 0.005
