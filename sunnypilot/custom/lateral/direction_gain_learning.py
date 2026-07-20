"""Direction-gain asymmetry learner — excursion-based per-direction gain ratio.

v1 fitted torque *levels* against lat-accel levels per direction. On crowned
right-hand-traffic roads that is unidentifiable: the left bucket is dominated by
a narrow crown-holding torque cluster, so regression dilution crushes the left
slope at any sample size (route 2bc: apparent ratio 0.22-0.57 depending on gates,
while the wide-range highway fit read a stable 0.80). v2 fits *excursions*:
delta lat-accel against delta steer over ~1 s maneuver windows, with both window
endpoints loaded on the same side of center. Differencing removes the crown/offset
levels entirely, and symmetric attenuation (actuation + tire lag) cancels in the
left/right ratio.

Only the asymmetry ratio is learned; the symmetric magnitude stays owned by the
base torqued learner. Apply mode scales the controller's torque conversion per
direction, normalized so the mean scale is 1 (no net gain change).

Sign convention: torqued's ``steer`` (= -actuatorsOutput.torque) positive means a
rightward lateral-accel demand, so positive-level windows measure the *right* gain.
"""
from __future__ import annotations

import json
from collections import deque
from typing import Any

import numpy as np

from openpilot.sunnypilot.custom.lateral.speed_aware_torque import _restore_key

DIRECTION_GAIN_PARAMS_VERSION = 2  # v1 (level-based) profiles are rejected wholesale
DIRECTION_GAIN_SPEED_BANDS = ((8.0, 15.0), (15.0, 100.0))

EXCURSION_WINDOW_S = 1.0       # pair each sample against one ~this far back
EXCURSION_WINDOW_TOL_S = 0.35
EXCURSION_MIN_DELTA = 0.04     # |dsteer| below this is dither, not a maneuver
EXCURSION_MAX_DY = 2.5         # m/s^2; larger jumps are events, not tracking
LEVEL_MIN = 0.03               # both endpoints this far from center, same side
MIN_PAIRS_PER_DIRECTION = 400
MIN_DELTA_SPAN = 0.06          # p5-p95 |dsteer| spread needed for slope leverage
PAIRS_PER_CELL = 4000

RATIO_MIN = 0.7                # left/right gain ratio sanity bounds
RATIO_MAX = 1.3
BAND_AGREEMENT_MAX = 0.12      # fitted bands must agree on the ratio within this
SCALE_CLAMP = 0.15             # applied per-direction scale stays within 1 +/- this


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


class DirectionGainBuckets:
  """Excursion pairs (dsteer, dlat_accel), split by loaded side within speed bands."""

  def __init__(self, pairs_per_cell=PAIRS_PER_CELL, speed_bands=None):
    self.speed_bands = tuple(speed_bands) if speed_bands is not None else DIRECTION_GAIN_SPEED_BANDS
    self._pairs = {(band, direction): [] for band in self.speed_bands for direction in (1, -1)}
    self._pairs_per_cell = pairs_per_cell
    self._history: deque = deque()  # (t, steer, lat_acc, v_ego)

  def add_point(self, steer, lateral_acc, v_ego, t):
    if not (np.isfinite(steer) and np.isfinite(lateral_acc) and np.isfinite(v_ego) and np.isfinite(t)):
      return
    horizon = EXCURSION_WINDOW_S + EXCURSION_WINDOW_TOL_S
    while self._history and t - self._history[0][0] > horizon:
      self._history.popleft()

    # pair against the oldest in-window sample; a gap in gated samples (override,
    # inactive) empties the window naturally, so pairs never straddle an event
    partner = None
    for old in self._history:
      if EXCURSION_WINDOW_S - EXCURSION_WINDOW_TOL_S <= t - old[0] <= horizon:
        partner = old
        break
    self._history.append((t, steer, lateral_acc, v_ego))
    if partner is None:
      return

    t0, x0, y0, v0 = partner
    dx = steer - x0
    dy = lateral_acc - y0
    if abs(dx) < EXCURSION_MIN_DELTA or abs(dy) > EXCURSION_MAX_DY:
      return
    # both endpoints loaded on the same side of center
    if abs(x0) < LEVEL_MIN or abs(steer) < LEVEL_MIN or (x0 > 0) != (steer > 0):
      return
    direction = 1 if steer > 0 else -1
    v_mid = (v_ego + v0) / 2.0
    for v_lo, v_hi in self.speed_bands:
      if v_lo <= v_mid < v_hi:
        cell = self._pairs[((v_lo, v_hi), direction)]
        cell.append((dx, dy))
        del cell[:-self._pairs_per_cell]
        break

  def direction_pairs(self, band, direction) -> np.ndarray:
    pairs = self._pairs[(band, direction)]
    if not pairs:
      return np.empty((0, 3))
    arr = np.array(pairs, dtype=float)
    return np.column_stack([arr[:, 0], np.ones(len(arr)), arr[:, 1]])


def _fit_band_ratio(buckets: DirectionGainBuckets, band):
  slopes = {}
  counts = {}
  for direction in (1, -1):
    pts = buckets.direction_pairs(band, direction)
    if len(pts) < MIN_PAIRS_PER_DIRECTION:
      return None
    mags = np.abs(pts[:, 0])
    if float(np.percentile(mags, 95) - np.percentile(mags, 5)) < MIN_DELTA_SPAN:
      return None
    slope = _fit_slope_ols(pts)
    if slope is None:
      return None
    slopes[direction] = slope
    counts[direction] = len(pts)
  ratio = slopes[-1] / slopes[1]  # left over right
  if not (RATIO_MIN <= ratio <= RATIO_MAX):
    return None
  return {'ratio': float(ratio), 'pointsLeft': int(counts[-1]), 'pointsRight': int(counts[1])}


def fit_direction_gain_profile(CP: Any, buckets: DirectionGainBuckets):
  if CP.lateralTuning.which() != 'torque':
    return None
  bands = []
  for v_lo, v_hi in buckets.speed_bands:
    fitted = _fit_band_ratio(buckets, (v_lo, v_hi))
    if fitted is not None:
      bands.append({'vLo': float(v_lo), 'vHi': float(v_hi), **fitted})
  # every configured band must fit: a single band bypasses the agreement check,
  # which is the whole confound control
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

  Higher left gain (ratio > 1) means less torque needed leftward, so leftward
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
  old_w = float(min(int(old_profile['points']), 8 * MIN_PAIRS_PER_DIRECTION))
  new_w = float(min(int(new_profile['points']), 8 * MIN_PAIRS_PER_DIRECTION))
  if old_w + new_w <= 0:
    return new_profile
  blended = dict(new_profile)
  blended['ratio'] = float((old_w * float(old_profile['ratio']) + new_w * float(new_profile['ratio'])) / (old_w + new_w))
  blended['points'] = int(min(int(old_profile['points']) + int(new_profile['points']), 16 * MIN_PAIRS_PER_DIRECTION))
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
  if points < 2 * MIN_PAIRS_PER_DIRECTION:
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
    if not (RATIO_MIN <= band_ratio <= RATIO_MAX) or pl < MIN_PAIRS_PER_DIRECTION or pr < MIN_PAIRS_PER_DIRECTION:
      return None
    bands.append({'vLo': v_lo, 'vHi': v_hi, 'ratio': band_ratio, 'pointsLeft': pl, 'pointsRight': pr})
  if len({(b['vLo'], b['vHi']) for b in bands}) != len(bands):
    return None
  return {'version': DIRECTION_GAIN_PARAMS_VERSION, 'restoreKey': payload['restoreKey'],
          'ratio': ratio, 'points': points, 'bands': bands}
