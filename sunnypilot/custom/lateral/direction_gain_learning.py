"""Direction-gain asymmetry learner — per-direction torque->lat-accel slope ratio.

Route 2a1/2b0 offline fits showed the rack responding ~9% stronger per unit torque
leftward than rightward (slope 1.645 vs 1.506), confounded by speed mix (the active
latAccelFactor is speed-adaptive) and hysteresis-branch occupancy. This learner
resolves the confounds by fitting the slope per direction *within* each speed band
and requiring the bands to agree on the ratio before publishing.

Only the asymmetry ratio is learned; the symmetric magnitude stays owned by the
base torqued learner. Apply mode scales the controller's torque conversion per
direction, normalized so the mean scale is 1 (no net gain change).

Sign convention: torqued's ``steer`` (= -actuatorsOutput.torque) positive means a
rightward lateral-accel demand, so the positive-steer cell is the *right* slope.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np

from openpilot.selfdrive.locationd.helpers import PointBuckets
from openpilot.sunnypilot.custom.lateral.speed_aware_torque import _restore_key

DIRECTION_GAIN_PARAMS_VERSION = 1
DIRECTION_GAIN_SPEED_BANDS = ((8.0, 15.0), (15.0, 100.0))
MIN_POINTS_PER_DIRECTION = 1500
MIN_TORQUE_MAGNITUDE = 0.05   # below this hysteresis dominates the fit
MIN_TORQUE_SPAN = 0.08        # p5-p95 |torque| spread needed for slope leverage
RATIO_MIN = 0.8               # left/right slope ratio sanity bounds
RATIO_MAX = 1.25
BAND_AGREEMENT_MAX = 0.12     # fitted bands must agree on the ratio within this
SCALE_CLAMP = 0.15            # applied per-direction scale stays within 1 +/- this


def _fit_slope_ols(points: np.ndarray):
  if points.shape[0] < 2:
    return None
  try:
    if float(np.var(points[:, 0].astype(float))) <= 1e-12:
      return None
    coef, *_ = np.linalg.lstsq(points[:, :2].astype(float), points[:, 2].astype(float), rcond=None)
    slope = float(coef[0])
    return slope if np.isfinite(slope) and slope > 0 else None
  except Exception:
    return None


def _finite_float(value):
  try:
    f = float(value)
  except (TypeError, ValueError):
    return None
  return f if np.isfinite(f) else None


class _DirectionPointBuckets(PointBuckets):
  def add_point(self, x, y):
    if not np.isfinite(x) or not np.isfinite(y):
      return
    for bound_min, bound_max in self.x_bounds:
      if (x >= bound_min) and (x < bound_max):
        self.buckets[(bound_min, bound_max)].append([x, 1.0, y])
        break


class DirectionGainBuckets:
  """(steer torque, lat accel) points, split by torque sign within speed bands."""

  # x = steer torque, torqued convention (positive = rightward)
  X_BOUNDS = [(-1.0, -MIN_TORQUE_MAGNITUDE), (MIN_TORQUE_MAGNITUDE, 1.0)]

  def __init__(self, points_per_bucket=3000, speed_bands=None):
    self.speed_bands = tuple(speed_bands) if speed_bands is not None else DIRECTION_GAIN_SPEED_BANDS
    self._bands = {band: _DirectionPointBuckets(x_bounds=self.X_BOUNDS, min_points=[1.0, 1.0], min_points_total=1,
                                                points_per_bucket=points_per_bucket, rowsize=3)
                   for band in self.speed_bands}

  def add_point(self, steer, lateral_acc, v_ego):
    if not np.isfinite(steer) or not np.isfinite(lateral_acc) or not np.isfinite(v_ego):
      return
    for v_lo, v_hi in self.speed_bands:
      if v_lo <= v_ego < v_hi:
        self._bands[(v_lo, v_hi)].add_point(steer, lateral_acc)
        break

  def direction_points(self, band, direction):
    # direction: +1 = rightward (positive steer), -1 = leftward
    bounds = self.X_BOUNDS[1] if direction > 0 else self.X_BOUNDS[0]
    return self._bands[band].buckets[bounds].arr


def _fit_band_ratio(buckets: DirectionGainBuckets, band):
  slopes = {}
  points = {}
  for direction in (1, -1):
    pts = buckets.direction_points(band, direction)
    if len(pts) < MIN_POINTS_PER_DIRECTION:
      return None
    mags = np.abs(pts[:, 0].astype(float))
    if float(np.percentile(mags, 95) - np.percentile(mags, 5)) < MIN_TORQUE_SPAN:
      return None
    slope = _fit_slope_ols(pts)
    if slope is None:
      return None
    slopes[direction] = slope
    points[direction] = len(pts)
  ratio = slopes[-1] / slopes[1]  # left over right
  if not (RATIO_MIN <= ratio <= RATIO_MAX):
    return None
  return {'ratio': float(ratio), 'pointsLeft': int(points[-1]), 'pointsRight': int(points[1])}


def fit_direction_gain_profile(CP: Any, buckets: DirectionGainBuckets):
  if CP.lateralTuning.which() != 'torque':
    return None
  bands = []
  for v_lo, v_hi in buckets.speed_bands:
    fitted = _fit_band_ratio(buckets, (v_lo, v_hi))
    if fitted is not None:
      bands.append({'vLo': float(v_lo), 'vHi': float(v_hi), **fitted})
  # every configured band must fit: a single band bypasses the agreement check,
  # which is the whole confound control (route 2b5 published highway-only at 0.814)
  if len(bands) < len(buckets.speed_bands):
    return None
  ratios = [b['ratio'] for b in bands]
  if max(ratios) - min(ratios) > BAND_AGREEMENT_MAX:
    return None
  weights = [min(b['pointsLeft'], b['pointsRight']) for b in bands]
  ratio = float(np.average(ratios, weights=weights))
  return {
    'version': DIRECTION_GAIN_PARAMS_VERSION,
    'restoreKey': _restore_key(CP),
    'ratio': ratio,
    'points': int(sum(b['pointsLeft'] + b['pointsRight'] for b in bands)),
    'bands': bands,
  }


def direction_scales(profile) -> dict[int, float]:
  """Per-direction torque scale keyed by internal torque sign (+1 = rightward).

  Higher left slope (ratio > 1) means less torque needed leftward, so leftward
  torque scales down and rightward up; mean stays 1 (pure asymmetry).
  """
  if not profile:
    return {1: 1.0, -1: 1.0}
  ratio = float(profile['ratio'])
  mean_slope_scale = (1.0 + ratio) / 2.0
  left = float(np.clip(mean_slope_scale / ratio, 1.0 - SCALE_CLAMP, 1.0 + SCALE_CLAMP))
  right = float(np.clip(mean_slope_scale, 1.0 - SCALE_CLAMP, 1.0 + SCALE_CLAMP))
  return {1: right, -1: left}


def blend_direction_gain_profile(old_profile, new_profile):
  if old_profile is None:
    return new_profile
  old_w = float(min(int(old_profile['points']), 4 * MIN_POINTS_PER_DIRECTION))
  new_w = float(min(int(new_profile['points']), 4 * MIN_POINTS_PER_DIRECTION))
  if old_w + new_w <= 0:
    return new_profile
  blended = dict(new_profile)
  blended['ratio'] = float((old_w * float(old_profile['ratio']) + new_w * float(new_profile['ratio'])) / (old_w + new_w))
  blended['points'] = int(min(int(old_profile['points']) + int(new_profile['points']), 8 * MIN_POINTS_PER_DIRECTION))
  return blended


def format_direction_gain_profile(profile: dict) -> str:
  return json.dumps(profile, separators=(',', ':'), sort_keys=True, allow_nan=False)


def parse_direction_gain_profile(CP: Any, payload):
  """Fail closed: any missing/invalid/foreign field distrusts the whole payload."""
  if CP.lateralTuning.which() != 'torque' or not isinstance(payload, dict):
    return None
  if payload.get('version') != DIRECTION_GAIN_PARAMS_VERSION:
    return None
  if payload.get('restoreKey') != _restore_key(CP):
    return None
  ratio = _finite_float(payload.get('ratio'))
  if ratio is None or not (RATIO_MIN <= ratio <= RATIO_MAX):
    return None
  try:
    points = int(payload['points'])
  except (KeyError, TypeError, ValueError):
    return None
  if points < 2 * MIN_POINTS_PER_DIRECTION:
    return None
  raw_bands = payload.get('bands')
  if not isinstance(raw_bands, list) or len(raw_bands) < len(DIRECTION_GAIN_SPEED_BANDS):
    return None
  bands = []
  for entry in raw_bands:
    if not isinstance(entry, dict):
      return None
    v_lo = _finite_float(entry.get('vLo'))
    v_hi = _finite_float(entry.get('vHi'))
    band_ratio = _finite_float(entry.get('ratio'))
    try:
      pl, pr = int(entry['pointsLeft']), int(entry['pointsRight'])
    except (KeyError, TypeError, ValueError):
      return None
    if v_lo is None or v_hi is None or v_lo >= v_hi or band_ratio is None:
      return None
    if not (RATIO_MIN <= band_ratio <= RATIO_MAX) or pl < MIN_POINTS_PER_DIRECTION or pr < MIN_POINTS_PER_DIRECTION:
      return None
    bands.append({'vLo': v_lo, 'vHi': v_hi, 'ratio': band_ratio, 'pointsLeft': pl, 'pointsRight': pr})
  if len({(b['vLo'], b['vHi']) for b in bands}) != len(bands):
    return None
  return {'version': DIRECTION_GAIN_PARAMS_VERSION, 'restoreKey': payload['restoreKey'],
          'ratio': ratio, 'points': points, 'bands': bands}
