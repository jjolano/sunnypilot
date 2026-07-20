import json
from types import SimpleNamespace

import numpy as np
import pytest

from openpilot.sunnypilot.custom.lateral.direction_gain_learning import (
  DIRECTION_GAIN_PARAMS_VERSION,
  SCALE_CLAMP,
  DirectionGainBuckets,
  blend_direction_gain_profile,
  direction_scales,
  fit_direction_gain_profile,
  format_direction_gain_profile,
  parse_direction_gain_profile,
)
from openpilot.sunnypilot.custom.lateral.speed_aware_torque import _restore_key

DT = 0.05


def _cp():
  torque = SimpleNamespace(latAccelFactor=1.94, friction=0.126)
  return SimpleNamespace(carFingerprint="TOYOTA_RAV4_TSS2", MAX_LAT_ACCEL=3.0,
                         lateralTuning=SimpleNamespace(which=lambda: 'torque', torque=torque))


def _feed_sine_drive(buckets, gain_left=1.5, gain_right=1.5, speeds=(10.0, 20.0),
                     duration_s=600.0, crown_hold=0.0, seed=0):
  """Synthetic drive: steer oscillates on each side of center around a possible
  crown-hold level; lat accel responds with per-direction gain plus offset."""
  rng = np.random.default_rng(seed)
  t = 0.0
  for v in speeds:
    for k in range(int(duration_s / DT)):
      # alternate blocks: work the left side, then the right side
      side = -1 if (k // 400) % 2 == 0 else 1
      wobble = 0.15 * np.sin(2 * np.pi * 0.25 * k * DT)
      level = side * (0.15 + abs(crown_hold) if side < 0 else 0.15)
      x = level + wobble * side * -1 if False else side * 0.15 + wobble
      if (x > 0) != (side > 0):
        x = side * 0.06  # keep the block on its side of center
      gain = gain_left if x < 0 else gain_right
      y = gain * x + crown_hold + rng.normal(0, 0.01)
      buckets.add_point(x, y, v, t)
      t += DT
    t += 10.0  # gap between speed blocks empties the pairing window
  return buckets


def test_fit_recovers_known_ratio():
  buckets = _feed_sine_drive(DirectionGainBuckets(), gain_left=1.65, gain_right=1.50)
  profile = fit_direction_gain_profile(_cp(), buckets)
  assert profile is not None
  assert abs(profile['ratio'] - 1.65 / 1.50) < 0.03
  assert len(profile['bands']) == 2


def test_crown_offset_does_not_bias_ratio():
  # the v1 failure mode: a constant crown hold shifted levels; differencing kills it
  clean = fit_direction_gain_profile(_cp(), _feed_sine_drive(DirectionGainBuckets(), 1.6, 1.5, crown_hold=0.0))
  crowned = fit_direction_gain_profile(_cp(), _feed_sine_drive(DirectionGainBuckets(), 1.6, 1.5, crown_hold=-0.3))
  assert clean is not None and crowned is not None
  assert abs(clean['ratio'] - crowned['ratio']) < 0.02


def test_fit_requires_every_band():
  buckets = _feed_sine_drive(DirectionGainBuckets(), speeds=(20.0,))  # highway only
  assert fit_direction_gain_profile(_cp(), buckets) is None


def test_fit_rejects_band_disagreement():
  buckets = DirectionGainBuckets()
  _feed_sine_drive(buckets, gain_left=1.9, gain_right=1.5, speeds=(10.0,))
  _feed_sine_drive(buckets, gain_left=1.5, gain_right=1.9, speeds=(20.0,), seed=1)
  assert fit_direction_gain_profile(_cp(), buckets) is None


def _total_pairs(buckets):
  return sum(len(buckets.direction_pairs(band, d)) for band in buckets.speed_bands for d in (1, -1))


def test_pairing_gates():
  # opposite-side endpoints never pair
  b = DirectionGainBuckets()
  b.add_point(-0.2, -0.3, 20.0, 0.0)
  b.add_point(0.2, 0.3, 20.0, 1.0)
  assert _total_pairs(b) == 0
  # same-side real excursion pairs, keyed by loaded side (positive = right)
  b = DirectionGainBuckets()
  b.add_point(0.1, 0.15, 20.0, 0.0)
  b.add_point(0.2, 0.3, 20.0, 1.0)
  assert len(b.direction_pairs((15.0, 100.0), 1)) == 1
  assert _total_pairs(b) == 1
  # below the excursion threshold: dither, no pair
  b = DirectionGainBuckets()
  b.add_point(0.10, 0.15, 20.0, 0.0)
  b.add_point(0.12, 0.18, 20.0, 1.0)
  assert _total_pairs(b) == 0
  # endpoints too close to center: ambiguous side, no pair
  b = DirectionGainBuckets()
  b.add_point(0.01, 0.02, 20.0, 0.0)
  b.add_point(0.09, 0.14, 20.0, 1.0)
  assert _total_pairs(b) == 0
  # gap larger than the window: no partner, no pair
  b = DirectionGainBuckets()
  b.add_point(0.1, 0.15, 20.0, 0.0)
  b.add_point(0.2, 0.3, 20.0, 5.0)
  assert _total_pairs(b) == 0


def test_direction_scales_mapping_and_neutrality():
  scales = direction_scales({'ratio': 1.10})
  assert scales[-1] < 1.0 < scales[1]
  assert abs(scales[-1] * 1.10 - scales[1] * 1.0) < 1e-9  # scaled gains equalize
  assert direction_scales(None) == {1: 1.0, -1: 1.0}


def test_direction_scales_clamped():
  scales = direction_scales({'ratio': 1.3})
  for s in scales.values():
    assert 1.0 - SCALE_CLAMP - 1e-9 <= s <= 1.0 + SCALE_CLAMP + 1e-9


def test_profile_roundtrip_and_parser_fails_closed():
  cp = _cp()
  profile = fit_direction_gain_profile(cp, _feed_sine_drive(DirectionGainBuckets()))
  payload = json.loads(format_direction_gain_profile(profile))
  assert parse_direction_gain_profile(cp, payload) is not None
  assert parse_direction_gain_profile(cp, {**payload, 'version': 1}) is None  # v1 rejected
  assert parse_direction_gain_profile(cp, {**payload, 'restoreKey': 'foreign'}) is None
  assert parse_direction_gain_profile(cp, {**payload, 'ratio': 3.0}) is None
  assert parse_direction_gain_profile(cp, {**payload, 'ratio': float('nan')}) is None
  assert parse_direction_gain_profile(cp, {**payload, 'points': 10}) is None
  assert parse_direction_gain_profile(cp, {**payload, 'bands': payload['bands'][:1]}) is None
  assert parse_direction_gain_profile(cp, "junk") is None


def test_blend_weighted_by_points():
  old = {'version': DIRECTION_GAIN_PARAMS_VERSION, 'restoreKey': 'k', 'ratio': 1.0, 'points': 3200, 'bands': []}
  new = {'version': DIRECTION_GAIN_PARAMS_VERSION, 'restoreKey': 'k', 'ratio': 1.2, 'points': 3200, 'bands': []}
  blended = blend_direction_gain_profile(old, new)
  assert blended['ratio'] == pytest.approx(1.1, abs=1e-6)
  assert blend_direction_gain_profile(None, new) is new


def test_restore_key_binds_profile_to_car():
  profile = fit_direction_gain_profile(_cp(), _feed_sine_drive(DirectionGainBuckets()))
  assert profile['restoreKey'] == _restore_key(_cp())
