"""Direction-gain asymmetry learner using block-independent excursion evidence."""

from __future__ import annotations

import json
from collections import deque
from typing import Any

import numpy as np

from openpilot.sunnypilot.custom.lateral.block_jackknife import (
  MAX_BLOCK_REL_SE,
  MIN_EVIDENCE_BLOCKS,
  fit_block_slope,
  fit_ratio_jackknife,
)
from openpilot.sunnypilot.custom.lateral.speed_aware_torque import _restore_key


DIRECTION_GAIN_PARAMS_VERSION = 3
DIRECTION_GAIN_SPEED_BANDS = ((8.0, 15.0), (15.0, 100.0))

EXCURSION_WINDOW_S = 1.0
EXCURSION_WINDOW_TOL_S = 0.35
EXCURSION_MIN_DELTA = 0.04
EXCURSION_MAX_DY = 2.5
LEVEL_MIN = 0.03
MIN_PAIRS_PER_DIRECTION = 400
MIN_DELTA_SPAN = 0.06
MIN_PAIRS_PER_BLOCK = 8
MIN_BLOCK_DELTA_SPAN = 0.03
PAIRS_PER_CELL = 4000

RATIO_MIN = 0.7
RATIO_MAX = 1.3
BAND_AGREEMENT_MAX = 0.12
SCALE_CLAMP = 0.15
DIRECTION_BAND_REPLACEMENT_MAX_DELTA = 0.05


def _finite_float(value):
  try:
    f = float(value)
  except (TypeError, ValueError):
    return None
  return f if np.isfinite(f) else None


def _finite_int(value):
  if isinstance(value, bool):
    return None
  try:
    f = float(value)
    i = int(value)
  except (TypeError, ValueError, OverflowError):
    return None
  return i if np.isfinite(f) and f == i else None


class DirectionGainBuckets:
  """Excursion pairs ``(dx, dy, block_id)`` split by loaded side and speed band."""

  def __init__(self, pairs_per_cell=PAIRS_PER_CELL, speed_bands=None):
    self.speed_bands = tuple(speed_bands) if speed_bands is not None else DIRECTION_GAIN_SPEED_BANDS
    self._pairs = {(band, direction): [] for band in self.speed_bands for direction in (1, -1)}
    self._pairs_per_cell = pairs_per_cell
    self._history: deque = deque()  # (t, steer, lat_acc, v_ego, block_id)
    self._completed_through = -1

  @property
  def completed_block_ids(self) -> range:
    return range(self._completed_through + 1)

  def set_completed_through(self, block_id: int):
    self._completed_through = max(self._completed_through, int(block_id))

  def clear_history(self):
    self._history.clear()

  def add_point(self, steer, lateral_acc, v_ego, t, block_id=0):
    if not (np.isfinite(steer) and np.isfinite(lateral_acc) and np.isfinite(v_ego) and np.isfinite(t)):
      return
    if block_id is None:
      self.clear_history()
      return
    try:
      block_id = int(block_id)
    except (TypeError, ValueError, OverflowError):
      self.clear_history()
      return
    if block_id < 0:
      self.clear_history()
      return
    if self._history and self._history[-1][4] != block_id:
      self.clear_history()

    horizon = EXCURSION_WINDOW_S + EXCURSION_WINDOW_TOL_S
    while self._history and t - self._history[0][0] > horizon:
      self._history.popleft()

    # Pair against the oldest in-window sample; a gap in gated samples (override,
    # inactive) empties the window naturally, so pairs never straddle an event.
    partner = None
    for old in self._history:
      if old[4] == block_id and EXCURSION_WINDOW_S - EXCURSION_WINDOW_TOL_S <= t - old[0] <= horizon:
        partner = old
        break
    self._history.append((t, steer, lateral_acc, v_ego, block_id))
    if partner is None:
      return

    _, x0, y0, v0, partner_block = partner
    if partner_block != block_id:
      return
    dx = steer - x0
    dy = lateral_acc - y0
    if abs(dx) < EXCURSION_MIN_DELTA or abs(dy) > EXCURSION_MAX_DY:
      return
    if abs(x0) < LEVEL_MIN or abs(steer) < LEVEL_MIN or (x0 > 0) != (steer > 0):
      return
    direction = 1 if steer > 0 else -1
    v_mid = (v_ego + v0) / 2.0
    for v_lo, v_hi in self.speed_bands:
      if v_lo <= v_mid < v_hi:
        cell = self._pairs[((v_lo, v_hi), direction)]
        cell.append((dx, dy, block_id))
        del cell[:-self._pairs_per_cell]
        break

  def direction_pairs(self, band, direction, completed_block_ids=None) -> np.ndarray:
    pairs = self._pairs[(band, direction)]
    if completed_block_ids is not None:
      if isinstance(completed_block_ids, range):
        pairs = [pair for pair in pairs if pair[2] in completed_block_ids]
      else:
        completed = set(completed_block_ids)
        pairs = [pair for pair in pairs if pair[2] in completed]
    if not pairs:
      return np.empty((0, 3), dtype=float)
    return np.asarray(pairs, dtype=float)


def _informative_pairs(points):
  informative = []
  for block_id in sorted(set(points[:, 2].astype(int))):
    block_points = points[points[:, 2].astype(int) == block_id]
    magnitudes = np.abs(block_points[:, 0])
    span = float(np.percentile(magnitudes, 95) - np.percentile(magnitudes, 5))
    if len(block_points) >= MIN_PAIRS_PER_BLOCK and span >= MIN_BLOCK_DELTA_SPAN:
      informative.append(block_id)
  return informative


def _fit_band_ratio(buckets: DirectionGainBuckets, band, completed_block_ids=None):
  points = {}
  informative = {}
  informative_points = {}
  for direction in (1, -1):
    points[direction] = buckets.direction_pairs(band, direction, completed_block_ids)
    informative[direction] = _informative_pairs(points[direction])
    if len(informative[direction]) < MIN_EVIDENCE_BLOCKS:
      return None
    informative_points[direction] = points[direction][
      np.isin(points[direction][:, 2].astype(int), informative[direction])
    ]
    if len(informative_points[direction]) < MIN_PAIRS_PER_DIRECTION:
      return None
    magnitudes = np.abs(informative_points[direction][:, 0])
    if float(np.percentile(magnitudes, 95) - np.percentile(magnitudes, 5)) < MIN_DELTA_SPAN:
      return None

  left_fit = fit_block_slope(informative_points[-1])
  right_fit = fit_block_slope(informative_points[1])
  if left_fit is None or right_fit is None:
    return None
  for fit, block_count in ((left_fit, len(informative[-1])), (right_fit, len(informative[1]))):
    if len(fit.block_slopes) != block_count:
      return None
    if not np.isfinite(fit.slope) or fit.slope <= 0:
      return None
    if any(not np.isfinite(slope) or slope <= 0 for slope in fit.block_slopes.values()):
      return None
    if any(not np.isfinite(slope) or slope <= 0 for slope in fit.loo_slopes.values()):
      return None

  ratio_fit = fit_ratio_jackknife(informative_points[-1], informative_points[1])
  if ratio_fit is None:
    return None
  if any(not np.isfinite(value) or not (RATIO_MIN <= value <= RATIO_MAX)
         for value in ratio_fit['ratio_loo'].values()):
    return None
  ratio = ratio_fit['ratio']
  if not (np.isfinite(ratio) and RATIO_MIN <= ratio <= RATIO_MAX):
    return None
  if (not np.isfinite(left_fit.rel_se) or left_fit.rel_se > MAX_BLOCK_REL_SE or
      not np.isfinite(right_fit.rel_se) or right_fit.rel_se > MAX_BLOCK_REL_SE or
      not np.isfinite(ratio_fit['ratio_rel_se']) or ratio_fit['ratio_rel_se'] > MAX_BLOCK_REL_SE):
    return None

  return {
    'ratio': float(ratio),
    'pointsLeft': int(len(informative_points[-1])),
    'pointsRight': int(len(informative_points[1])),
    'blocksLeft': int(len(informative[-1])),
    'blocksRight': int(len(informative[1])),
    'ratioBlocks': int(len(ratio_fit['ratio_loo'])),
    'leftSlopeRelSe': float(left_fit.rel_se),
    'rightSlopeRelSe': float(right_fit.rel_se),
    'ratioRelSe': float(ratio_fit['ratio_rel_se']),
  }


def fit_direction_gain_profile(CP: Any, buckets: DirectionGainBuckets, completed_block_ids=None):
  if CP.lateralTuning.which() != 'torque':
    return None
  if completed_block_ids is None:
    completed_block_ids = buckets.completed_block_ids
  bands = []
  for v_lo, v_hi in buckets.speed_bands:
    fitted = _fit_band_ratio(buckets, (v_lo, v_hi), completed_block_ids)
    if fitted is None:
      return None
    bands.append({'vLo': float(v_lo), 'vHi': float(v_hi), **fitted})

  ratios = [b['ratio'] for b in bands]
  if max(ratios) - min(ratios) > BAND_AGREEMENT_MAX:
    return None
  weights = [min(b['blocksLeft'], b['blocksRight']) for b in bands]
  ratio = float(np.average(ratios, weights=weights))
  return {
    'version': DIRECTION_GAIN_PARAMS_VERSION,
    'restoreKey': _restore_key(CP),
    'ratio': ratio,
    'points': int(sum(b['pointsLeft'] + b['pointsRight'] for b in bands)),
    'blockCount': int(min(min(b['blocksLeft'], b['blocksRight']) for b in bands)),
    'maxRelSe': float(max(
      value
      for band in bands
      for value in (band['leftSlopeRelSe'], band['rightSlopeRelSe'], band['ratioRelSe'])
    )),
    'bands': bands,
  }


def direction_scales(profile) -> dict[int, float]:
  """Per-direction torque scale; malformed profiles fail closed to identity."""
  if not isinstance(profile, dict):
    return {1: 1.0, -1: 1.0}
  ratio = _finite_float(profile.get('ratio'))
  if ratio is None or not (RATIO_MIN <= ratio <= RATIO_MAX):
    return {1: 1.0, -1: 1.0}
  mean_slope_scale = (1.0 + ratio) / 2.0
  left = float(np.clip(mean_slope_scale / ratio, 1.0 - SCALE_CLAMP, 1.0 + SCALE_CLAMP))
  right = float(np.clip(mean_slope_scale, 1.0 - SCALE_CLAMP, 1.0 + SCALE_CLAMP))
  return {1: right, -1: left}


def replace_direction_gain_profile(old_profile, new_profile):
  """Replace a whole snapshot only when every configured band moves little."""
  if old_profile is None:
    return new_profile
  old_bands = {(b['vLo'], b['vHi']): b for b in old_profile.get('bands', [])}
  new_bands = {(b['vLo'], b['vHi']): b for b in new_profile.get('bands', [])}
  if set(old_bands) != set(new_bands):
    return old_profile
  if any(abs(float(new_bands[key]['ratio']) - float(old_bands[key]['ratio'])) > DIRECTION_BAND_REPLACEMENT_MAX_DELTA
         for key in old_bands):
    return old_profile
  return new_profile


def format_direction_gain_profile(profile: dict) -> str:
  return json.dumps(profile, separators=(',', ':'), sort_keys=True, allow_nan=False)


def _validate_band(entry):
  expected_keys = {
    'vLo', 'vHi', 'ratio', 'pointsLeft', 'pointsRight', 'blocksLeft', 'blocksRight',
    'ratioBlocks', 'leftSlopeRelSe', 'rightSlopeRelSe', 'ratioRelSe',
  }
  if set(entry) != expected_keys:
    return None
  v_lo = _finite_float(entry.get('vLo'))
  v_hi = _finite_float(entry.get('vHi'))
  ratio = _finite_float(entry.get('ratio'))
  points_left = _finite_int(entry.get('pointsLeft'))
  points_right = _finite_int(entry.get('pointsRight'))
  blocks_left = _finite_int(entry.get('blocksLeft'))
  blocks_right = _finite_int(entry.get('blocksRight'))
  ratio_blocks = _finite_int(entry.get('ratioBlocks'))
  left_rel_se = _finite_float(entry.get('leftSlopeRelSe'))
  right_rel_se = _finite_float(entry.get('rightSlopeRelSe'))
  ratio_rel_se = _finite_float(entry.get('ratioRelSe'))
  if any(value is None for value in (
    v_lo, v_hi, ratio, points_left, points_right, blocks_left, blocks_right, ratio_blocks,
    left_rel_se, right_rel_se, ratio_rel_se,
  )):
    return None
  assert (v_lo is not None and v_hi is not None and ratio is not None and
          points_left is not None and points_right is not None and blocks_left is not None and
          blocks_right is not None and ratio_blocks is not None and left_rel_se is not None and
          right_rel_se is not None and ratio_rel_se is not None)
  if (not (v_lo < v_hi) or not (RATIO_MIN <= ratio <= RATIO_MAX) or
      points_left < MIN_PAIRS_PER_DIRECTION or points_right < MIN_PAIRS_PER_DIRECTION or
      blocks_left < MIN_EVIDENCE_BLOCKS or blocks_right < MIN_EVIDENCE_BLOCKS or
      ratio_blocks < max(blocks_left, blocks_right) or ratio_blocks > blocks_left + blocks_right or
      points_left < blocks_left * MIN_PAIRS_PER_BLOCK or points_right < blocks_right * MIN_PAIRS_PER_BLOCK or
      any(not (0.0 <= value <= MAX_BLOCK_REL_SE) for value in (left_rel_se, right_rel_se, ratio_rel_se))):
    return None
  return {
    'vLo': v_lo,
    'vHi': v_hi,
    'ratio': ratio,
    'pointsLeft': points_left,
    'pointsRight': points_right,
    'blocksLeft': blocks_left,
    'blocksRight': blocks_right,
    'ratioBlocks': ratio_blocks,
    'leftSlopeRelSe': left_rel_se,
    'rightSlopeRelSe': right_rel_se,
    'ratioRelSe': ratio_rel_se,
  }


def parse_direction_gain_profile(CP: Any, payload):
  """Fail closed on missing, foreign, non-finite, or inconsistent profile fields."""
  if CP.lateralTuning.which() != 'torque' or not isinstance(payload, dict):
    return None
  if payload.get('version') != DIRECTION_GAIN_PARAMS_VERSION:
    return None
  if payload.get('restoreKey') != _restore_key(CP):
    return None
  expected_keys = {'version', 'restoreKey', 'ratio', 'points', 'blockCount', 'maxRelSe', 'bands'}
  if set(payload) != expected_keys or not isinstance(payload.get('bands'), list):
    return None

  configured = set(DIRECTION_GAIN_SPEED_BANDS)
  bands = []
  for entry in payload['bands']:
    if not isinstance(entry, dict):
      return None
    band = _validate_band(entry)
    if band is None or (band['vLo'], band['vHi']) not in configured:
      return None
    bands.append(band)
  if len(bands) != len(configured) or len({(b['vLo'], b['vHi']) for b in bands}) != len(bands):
    return None
  if max(band['ratio'] for band in bands) - min(band['ratio'] for band in bands) > BAND_AGREEMENT_MAX:
    return None

  ratio = _finite_float(payload.get('ratio'))
  points = _finite_int(payload.get('points'))
  block_count = _finite_int(payload.get('blockCount'))
  max_rel_se = _finite_float(payload.get('maxRelSe'))
  if ratio is None or points is None or block_count is None or max_rel_se is None:
    return None
  if not (RATIO_MIN <= ratio <= RATIO_MAX and 0.0 <= max_rel_se <= MAX_BLOCK_REL_SE):
    return None

  expected_points = sum(b['pointsLeft'] + b['pointsRight'] for b in bands)
  expected_block_count = min(min(b['blocksLeft'], b['blocksRight']) for b in bands)
  expected_max_rel_se = max(
    value for band in bands for value in (band['leftSlopeRelSe'], band['rightSlopeRelSe'], band['ratioRelSe'])
  )
  weights = [min(b['blocksLeft'], b['blocksRight']) for b in bands]
  expected_ratio = float(np.average([b['ratio'] for b in bands], weights=weights))
  if (points != expected_points or block_count != expected_block_count or
      abs(max_rel_se - expected_max_rel_se) > 1e-9 or abs(ratio - expected_ratio) > 1e-9):
    return None

  bands.sort(key=lambda b: b['vLo'])
  return {
    'version': DIRECTION_GAIN_PARAMS_VERSION,
    'restoreKey': payload['restoreKey'],
    'ratio': ratio,
    'points': points,
    'blockCount': block_count,
    'maxRelSe': max_rel_se,
    'bands': bands,
  }
