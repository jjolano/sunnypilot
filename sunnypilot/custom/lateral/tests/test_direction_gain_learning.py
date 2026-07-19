import json
from types import SimpleNamespace

import numpy as np
import pytest

from openpilot.sunnypilot.custom.lateral.direction_gain_learning import (
  DIRECTION_GAIN_PARAMS_VERSION,
  MIN_POINTS_PER_DIRECTION,
  SCALE_CLAMP,
  DirectionGainBuckets,
  blend_direction_gain_profile,
  direction_scales,
  fit_direction_gain_profile,
  format_direction_gain_profile,
  parse_direction_gain_profile,
)
from openpilot.sunnypilot.custom.lateral.speed_aware_torque import _restore_key


def _cp():
  torque = SimpleNamespace(latAccelFactor=1.94, friction=0.126)
  return SimpleNamespace(carFingerprint="TOYOTA_RAV4_TSS2", MAX_LAT_ACCEL=3.0,
                         lateralTuning=SimpleNamespace(which=lambda: 'torque', torque=torque))


def _fill_buckets(slope_left=1.65, slope_right=1.50, n=2000, speeds=(10.0, 20.0), noise=0.02, seed=0):
  rng = np.random.default_rng(seed)
  buckets = DirectionGainBuckets()
  for v in speeds:
    for _ in range(n):
      x = rng.uniform(0.06, 0.4)
      buckets.add_point(x, slope_right * x + rng.normal(0, noise), v)      # rightward
      buckets.add_point(-x, -slope_left * x + rng.normal(0, noise), v)     # leftward
  return buckets


def test_fit_recovers_known_ratio():
  profile = fit_direction_gain_profile(_cp(), _fill_buckets())
  assert profile is not None
  assert abs(profile['ratio'] - 1.65 / 1.50) < 0.02
  assert len(profile['bands']) == 2


def test_fit_requires_min_points():
  profile = fit_direction_gain_profile(_cp(), _fill_buckets(n=MIN_POINTS_PER_DIRECTION // 2))
  assert profile is None


def test_fit_rejects_band_disagreement():
  rng = np.random.default_rng(1)
  buckets = DirectionGainBuckets()
  for v, (sl, sr) in ((10.0, (1.9, 1.5)), (20.0, (1.5, 1.9))):  # ratios 1.27 vs 0.79
    for _ in range(2000):
      x = rng.uniform(0.06, 0.4)
      buckets.add_point(x, sr * x, v)
      buckets.add_point(-x, -sl * x, v)
  assert fit_direction_gain_profile(_cp(), buckets) is None


def test_direction_scales_mapping_and_neutrality():
  # left responds stronger (ratio > 1) => leftward torque scales DOWN, rightward UP
  scales = direction_scales({'ratio': 1.10})
  assert scales[-1] < 1.0 < scales[1]
  assert abs((scales[1] + scales[-1] * 1.10 / 1.0) / 2 - scales[1]) < 1.0  # sanity: finite
  # neutrality: scaled slopes equalize — scale_left * slope_left == scale_right * slope_right
  assert abs(scales[-1] * 1.10 - scales[1] * 1.0) < 1e-9
  assert direction_scales(None) == {1: 1.0, -1: 1.0}


def test_direction_scales_clamped():
  scales = direction_scales({'ratio': 1.25})
  for s in scales.values():
    assert 1.0 - SCALE_CLAMP - 1e-9 <= s <= 1.0 + SCALE_CLAMP + 1e-9


def test_profile_roundtrip_and_parser_fails_closed():
  cp = _cp()
  profile = fit_direction_gain_profile(cp, _fill_buckets())
  payload = json.loads(format_direction_gain_profile(profile))
  assert parse_direction_gain_profile(cp, payload) is not None
  assert parse_direction_gain_profile(cp, {**payload, 'version': 99}) is None
  assert parse_direction_gain_profile(cp, {**payload, 'restoreKey': 'foreign'}) is None
  assert parse_direction_gain_profile(cp, {**payload, 'ratio': 3.0}) is None
  assert parse_direction_gain_profile(cp, {**payload, 'ratio': float('nan')}) is None
  assert parse_direction_gain_profile(cp, {**payload, 'points': 10}) is None
  assert parse_direction_gain_profile(cp, {**payload, 'bands': []}) is None
  assert parse_direction_gain_profile(cp, "junk") is None
  pid_cp = SimpleNamespace(carFingerprint="TOYOTA_RAV4_TSS2", MAX_LAT_ACCEL=3.0,
                           lateralTuning=SimpleNamespace(which=lambda: 'pid'))
  assert parse_direction_gain_profile(pid_cp, payload) is None


def test_blend_weighted_by_points():
  old = {'version': DIRECTION_GAIN_PARAMS_VERSION, 'restoreKey': 'k', 'ratio': 1.0, 'points': 6000, 'bands': []}
  new = {'version': DIRECTION_GAIN_PARAMS_VERSION, 'restoreKey': 'k', 'ratio': 1.2, 'points': 6000, 'bands': []}
  blended = blend_direction_gain_profile(old, new)
  assert blended['ratio'] == pytest.approx(1.1, abs=1e-6)
  assert blend_direction_gain_profile(None, new) is new


def test_restore_key_binds_profile_to_car():
  profile = fit_direction_gain_profile(_cp(), _fill_buckets())
  assert profile['restoreKey'] == _restore_key(_cp())
