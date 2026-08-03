import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from openpilot.sunnypilot.custom.lateral.block_jackknife import MAX_BLOCK_REL_SE, fit_ratio_jackknife
from openpilot.sunnypilot.custom.lateral.direction_gain_learning import (
  BAND_AGREEMENT_MAX,
  DIRECTION_GAIN_PARAMS_VERSION,
  DIRECTION_GAIN_SPEED_BANDS,
  MIN_PAIRS_PER_DIRECTION,
  DirectionGainBuckets,
  _fit_band_ratio,
  direction_scales,
  fit_direction_gain_profile,
  format_direction_gain_profile,
  parse_direction_gain_profile,
  replace_direction_gain_profile,
)
from openpilot.sunnypilot.custom.lateral.speed_aware_torque import _restore_key


def cp():
  torque = SimpleNamespace(latAccelFactor=1.94, friction=0.126)
  return SimpleNamespace(carFingerprint='test', lateralTuning=SimpleNamespace(which=lambda: 'torque', torque=torque))


def _add_pair(buckets, band, direction, block_id, slope, pair_index, timestamp, offset=0.0):
  speed = (band[0] + min(band[1] - band[0], 2.0))
  magnitude = 0.08 + 0.005 * (pair_index % 5)
  delta = 0.06 + 0.01 * (pair_index % 10)
  x0 = direction * magnitude
  x1 = direction * (magnitude + delta)
  intercept = offset + 0.3 * block_id
  buckets.add_point(x0, slope * x0 + intercept, speed, timestamp, block_id)
  buckets.add_point(x1, slope * x1 + intercept, speed, timestamp + 1.0, block_id)
  buckets.clear_history()  # keep each analytic pair independent


def _add_block(buckets, band, direction, block_id, slope=1.0, count=34, same_delta=False, offset=0.0):
  for pair_index in range(count):
    if same_delta:
      # A block with repeated excursion size is deliberately non-informative.
      pair_index = 0
    _add_pair(buckets, band, direction, block_id, slope, pair_index, block_id * 1000 + pair_index * 2, offset)


def _clean_buckets(blocks=12, slopes=None):
  buckets = DirectionGainBuckets(pairs_per_cell=10000)
  slopes = slopes or {}
  for band in DIRECTION_GAIN_SPEED_BANDS:
    for direction in (-1, 1):
      for block_id in range(blocks):
        slope = slopes.get((band, direction, block_id), 1.1 if direction == -1 else 1.0)
        _add_block(buckets, band, direction, block_id, slope=slope)
  buckets.set_completed_through(blocks - 1)
  return buckets


def _band(ratio=1.0, *, left_se=0.05, right_se=0.05, ratio_se=0.05):
  return {
    'vLo': 8.0,
    'vHi': 15.0,
    'ratio': ratio,
    'pointsLeft': MIN_PAIRS_PER_DIRECTION,
    'pointsRight': MIN_PAIRS_PER_DIRECTION,
    'blocksLeft': 12,
    'blocksRight': 12,
    'ratioBlocks': 12,
    'leftSlopeRelSe': left_se,
    'rightSlopeRelSe': right_se,
    'ratioRelSe': ratio_se,
  }


def _profile(bands=None):
  bands = bands or [
    {**_band(1.0), 'vLo': DIRECTION_GAIN_SPEED_BANDS[0][0], 'vHi': DIRECTION_GAIN_SPEED_BANDS[0][1]},
    {**_band(1.0), 'vLo': DIRECTION_GAIN_SPEED_BANDS[1][0], 'vHi': DIRECTION_GAIN_SPEED_BANDS[1][1]},
  ]
  return {
    'version': DIRECTION_GAIN_PARAMS_VERSION,
    'restoreKey': _restore_key(cp()),
    'ratio': float(np.average([band['ratio'] for band in bands], weights=[12, 12])),
    'points': sum(band['pointsLeft'] + band['pointsRight'] for band in bands),
    'blockCount': 12,
    'maxRelSe': max(value for band in bands for value in (
      band['leftSlopeRelSe'], band['rightSlopeRelSe'], band['ratioRelSe'])),
    'bands': copy.deepcopy(bands),
  }


def test_clean_twelve_block_profile_excludes_incomplete_block():
  buckets = _clean_buckets(blocks=13)
  profile = fit_direction_gain_profile(cp(), buckets, range(12))

  assert profile is not None
  assert profile['blockCount'] == 12
  assert profile['points'] == 12 * 34 * 4
  assert profile['ratio'] == pytest.approx(1.1)
  assert all(band['blocksLeft'] == band['blocksRight'] == 12 for band in profile['bands'])
  assert all(band['ratioBlocks'] == 12 for band in profile['bands'])
  assert all(band['ratioRelSe'] == pytest.approx(0.0) for band in profile['bands'])


def test_crown_offset_does_not_bias_excursion_ratio():
  def build(offset):
    buckets = DirectionGainBuckets(pairs_per_cell=10000)
    for band in DIRECTION_GAIN_SPEED_BANDS:
      for direction, slope in ((-1, 1.6), (1, 1.5)):
        for block_id in range(12):
          _add_block(buckets, band, direction, block_id, slope=slope, offset=offset)
    buckets.set_completed_through(11)
    return fit_direction_gain_profile(cp(), buckets)

  clean = build(0.0)
  crowned = build(-0.3)

  assert clean is not None and crowned is not None
  assert clean['ratio'] == pytest.approx(crowned['ratio'])
  assert clean['ratio'] == pytest.approx(1.6 / 1.5)


def test_direction_fit_requires_every_configured_band():
  buckets = DirectionGainBuckets(pairs_per_cell=10000)
  for direction in (-1, 1):
    for block_id in range(12):
      _add_block(buckets, DIRECTION_GAIN_SPEED_BANDS[0], direction, block_id)
  buckets.set_completed_through(11)
  assert fit_direction_gain_profile(cp(), buckets) is None


def test_repeated_samples_in_one_block_do_not_pass_direction_gate():
  buckets = _clean_buckets(blocks=1)
  assert fit_direction_gain_profile(cp(), buckets) is None


def test_pairing_gates_keep_old_safety_rejections():
  band = DIRECTION_GAIN_SPEED_BANDS[1]

  buckets = DirectionGainBuckets()
  buckets.add_point(-0.2, -0.3, 20.0, 0.0)
  buckets.add_point(0.2, 0.3, 20.0, 1.0)
  assert len(buckets.direction_pairs(band, 1)) == 0

  buckets = DirectionGainBuckets()
  buckets.add_point(0.1, 0.15, 20.0, 0.0)
  buckets.add_point(0.2, 0.3, 20.0, 1.0)
  assert len(buckets.direction_pairs(band, 1)) == 1

  for end, delta in ((0.12, 'dither'), (0.09, 'center'), (0.2, 'gap')):
    buckets = DirectionGainBuckets()
    buckets.add_point(0.1 if delta != 'center' else 0.01, 0.15, 20.0, 0.0)
    buckets.add_point(end, 0.18 if delta != 'gap' else 0.3, 20.0, 5.0 if delta == 'gap' else 1.0)
    assert len(buckets.direction_pairs(band, 1)) == 0


def test_direction_scales_preserve_mean_and_are_clamped():
  scales = direction_scales({'ratio': 1.10})
  assert scales[-1] < 1.0 < scales[1]
  assert scales[-1] * 1.10 == pytest.approx(scales[1])
  clamped = direction_scales({'ratio': 1.3})
  assert all(0.85 - 1e-9 <= scale <= 1.15 + 1e-9 for scale in clamped.values())


def test_pairing_clears_on_block_or_guard_boundary_and_completed_filter_excludes_current():
  buckets = DirectionGainBuckets()
  band = DIRECTION_GAIN_SPEED_BANDS[1]
  buckets.add_point(0.1, 0.1, 20.0, 0.0, 0)
  buckets.add_point(0.2, 0.2, 20.0, 1.0, 1)  # block change clears the candidate
  assert len(buckets.direction_pairs(band, 1)) == 0

  buckets.add_point(0.1, 0.1, 20.0, 2.0, 0)
  buckets.add_point(0.2, 0.2, 20.0, 3.0, None)  # type: ignore[arg-type]  # guard clears the candidate
  assert len(buckets.direction_pairs(band, 1)) == 0

  _add_pair(buckets, band, 1, 0, 1.0, 0, 4.0)
  _add_pair(buckets, band, 1, 1, 1.0, 1, 6.0)
  assert len(buckets.direction_pairs(band, 1, range(1))) == 1
  assert len(buckets.direction_pairs(band, 1, range(2))) == 2


@pytest.mark.parametrize('bad_field', ['leftSlopeRelSe', 'rightSlopeRelSe', 'ratioRelSe'])
def test_parser_rejects_each_direction_uncertainty_field_independently(bad_field):
  payload = _profile()
  payload['bands'][0][bad_field] = MAX_BLOCK_REL_SE + 0.01
  payload['maxRelSe'] = MAX_BLOCK_REL_SE + 0.01
  assert parse_direction_gain_profile(cp(), payload) is None


def test_negative_and_degenerate_informative_blocks_are_rejected():
  band = DIRECTION_GAIN_SPEED_BANDS[0]
  negative = _clean_buckets(slopes={(band, -1, 0): -1.0})
  assert fit_direction_gain_profile(cp(), negative) is None

  degenerate = DirectionGainBuckets(pairs_per_cell=10000)
  for band in DIRECTION_GAIN_SPEED_BANDS:
    for direction in (-1, 1):
      for block_id in range(12):
        _add_block(degenerate, band, direction, block_id, same_delta=(block_id == 0))
  degenerate.set_completed_through(11)
  assert fit_direction_gain_profile(cp(), degenerate) is None


def test_informative_padding_does_not_supply_direction_count_or_span():
  buckets = DirectionGainBuckets(pairs_per_cell=10000)
  band = DIRECTION_GAIN_SPEED_BANDS[1]
  for direction in (-1, 1):
    for block_id in range(11):
      _add_block(buckets, band, direction, block_id)
    # This uncompleted block has enough raw pairs and all the global delta span,
    # but it cannot add an informative completed block to the fit.
    _add_block(buckets, band, direction, 11, count=400, slope=1.0)
  buckets.set_completed_through(10)

  raw = buckets.direction_pairs(band, 1)
  assert len(raw) >= MIN_PAIRS_PER_DIRECTION
  assert np.percentile(np.abs(raw[:, 0]), 95) - np.percentile(np.abs(raw[:, 0]), 5) >= 0.06
  assert fit_direction_gain_profile(cp(), buckets) is None


def test_positive_block_correlated_direction_noise_fails_robust_uncertainty():
  buckets = DirectionGainBuckets(pairs_per_cell=10000)
  band = DIRECTION_GAIN_SPEED_BANDS[1]
  for block_id in range(12):
    _add_block(buckets, band, -1, block_id, slope=0.75 if block_id == 0 else 1.2,
               count=1000 if block_id == 0 else 34)
    _add_block(buckets, band, 1, block_id, slope=1.0, count=34)
  buckets.set_completed_through(11)

  left = buckets.direction_pairs(band, -1, range(12))
  right = buckets.direction_pairs(band, 1, range(12))
  assert len(left) >= MIN_PAIRS_PER_DIRECTION and len(right) >= MIN_PAIRS_PER_DIRECTION
  assert np.percentile(np.abs(left[:, 0]), 95) - np.percentile(np.abs(left[:, 0]), 5) >= 0.06
  result = fit_ratio_jackknife(left, right)
  assert result is not None and result['left'].rel_se > MAX_BLOCK_REL_SE
  assert _fit_band_ratio(buckets, band, range(12)) is None


def test_influential_block_ratio_loo_out_of_range_rejects():
  buckets = DirectionGainBuckets(pairs_per_cell=10000)
  band = DIRECTION_GAIN_SPEED_BANDS[1]
  for block_id in range(12):
    _add_block(buckets, band, -1, block_id, slope=1.0, count=34)
    _add_block(buckets, band, 1, block_id, slope=1.0 if block_id == 0 else 0.7,
               count=1000 if block_id == 0 else 34)
  buckets.set_completed_through(11)
  left = buckets.direction_pairs(band, -1, range(12))
  right = buckets.direction_pairs(band, 1, range(12))
  result = fit_ratio_jackknife(left, right)

  assert result is not None
  assert 0.7 <= result['ratio'] <= 1.3
  assert result['ratio_loo'][0] > 1.3
  assert _fit_band_ratio(buckets, band, range(12)) is None


def test_ratio_jackknife_union_has_shared_block_covariance():
  def rows(slopes):
    return [
      (x, slope * x + block, block)
      for block, slope in enumerate(slopes)
      for x in (0.1, 0.2, 0.3)
    ]

  left = rows((1.0, 1.1, 1.2))
  right = rows((0.8, 0.9, 1.0))
  result = fit_ratio_jackknife(left, right)

  assert result is not None
  assert set(result['ratio_loo']) == {0, 1, 2}
  loo_result = fit_ratio_jackknife(
    [row for row in left if row[2] != 1],
    [row for row in right if row[2] != 1],
  )
  assert loo_result is not None
  assert result['ratio_loo'][1] == pytest.approx(loo_result['ratio'])


def test_cross_band_disagreement_is_rejected_by_fit():
  slopes = {}
  for direction in (-1, 1):
    for block_id in range(12):
      slopes[((8.0, 15.0), direction, block_id)] = 1.2 if direction == -1 else 1.0
      slopes[((15.0, 100.0), direction, block_id)] = 1.0
  assert fit_direction_gain_profile(cp(), _clean_buckets(slopes=slopes)) is None


def test_parser_rejects_cross_band_disagreement():
  bands = [
    {**_band(1.0), 'vLo': 8.0, 'vHi': 15.0},
    {**_band(1.2), 'vLo': 15.0, 'vHi': 100.0},
  ]
  payload = _profile(bands)
  # Parser and fitter must enforce the same cross-band agreement contract.
  assert max(band['ratio'] for band in bands) - min(band['ratio'] for band in bands) > BAND_AGREEMENT_MAX
  assert parse_direction_gain_profile(cp(), payload) is None


def test_direction_parser_versions_fields_and_derived_totals():
  payload = _profile()
  assert parse_direction_gain_profile(cp(), json.loads(format_direction_gain_profile(payload))) == payload
  assert parse_direction_gain_profile(cp(), {**payload, 'version': 2}) is None

  for field in ('points', 'blockCount', 'maxRelSe', 'ratio'):
    bad = copy.deepcopy(payload)
    bad[field] = float('nan') if field != 'points' else 1
    assert parse_direction_gain_profile(cp(), bad) is None

  for field in ('points', 'blockCount', 'maxRelSe', 'ratio'):
    bad = copy.deepcopy(payload)
    bad[field] += 1
    assert parse_direction_gain_profile(cp(), bad) is None

  missing = copy.deepcopy(payload)
  del missing['bands'][0]['ratioBlocks']
  assert parse_direction_gain_profile(cp(), missing) is None

  unknown_band = copy.deepcopy(payload)
  unknown_band['bands'][0]['vHi'] = 16.0
  assert parse_direction_gain_profile(cp(), unknown_band) is None

  foreign = copy.deepcopy(payload)
  foreign['restoreKey'] = 'foreign'
  assert parse_direction_gain_profile(cp(), foreign) is None
  malformed = copy.deepcopy(payload)
  malformed['bands'][0]['ratio'] = float('nan')
  assert parse_direction_gain_profile(cp(), malformed) is None


@pytest.mark.parametrize('field', ['ratioBlocksTooLow', 'ratioBlocksTooHigh', 'leftPointsTooLow', 'rightPointsTooLow'])
def test_direction_parser_rejects_union_and_block_point_invariants(field):
  payload = _profile()
  band = payload['bands'][0]
  if field == 'ratioBlocksTooLow':
    band['ratioBlocks'] = max(band['blocksLeft'], band['blocksRight']) - 1
  elif field == 'ratioBlocksTooHigh':
    band['ratioBlocks'] = band['blocksLeft'] + band['blocksRight'] + 1
  elif field == 'leftPointsTooLow':
    band['blocksLeft'] = 51
    band['pointsLeft'] = 400
  else:
    band['blocksRight'] = 51
    band['pointsRight'] = 400
  assert parse_direction_gain_profile(cp(), payload) is None


def test_direction_snapshot_replacement_is_whole_profile():
  old = _profile()
  close = copy.deepcopy(old)
  close['bands'][0]['ratio'] = 1.04
  close['ratio'] = 1.02
  assert replace_direction_gain_profile(old, close) is close

  far = copy.deepcopy(old)
  far['bands'][1]['ratio'] = 1.06
  far['ratio'] = 1.03
  assert replace_direction_gain_profile(old, far) is old


def test_invalid_direction_profile_is_identity():
  assert direction_scales(None) == {1: 1.0, -1: 1.0}
  assert direction_scales({'ratio': 2.0}) == {1: 1.0, -1: 1.0}
