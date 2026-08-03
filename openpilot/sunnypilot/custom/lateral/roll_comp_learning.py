import json
from typing import Any

import numpy as np

from openpilot.selfdrive.locationd.helpers import PointBuckets
from openpilot.sunnypilot.custom.lateral.block_jackknife import (
  MAX_BLOCK_REL_SE,
  MIN_EVIDENCE_BLOCKS,
  fit_block_slope,
)
from openpilot.sunnypilot.custom.lateral.speed_aware_torque import _restore_key


ROLL_COMP_PARAMS_VERSION = 2
ROLL_GAIN_MIN = 0.3
ROLL_GAIN_MAX = 1.0
MIN_POINTS = 2000
MIN_X_SPAN = 0.25
MIN_CONFIDENCE = 0.5
MIN_POINTS_PER_BLOCK = 20
MIN_BLOCK_X_SPAN = 0.10
ROLL_BAND_REPLACEMENT_MAX_DELTA = 0.05

# x = -sin(roll)*g in lat-accel units (m/s^2); bounds straddle zero so both
# crown directions are represented.
ROLL_COMP_BUCKET_BOUNDS = [(-1.0, -0.5), (-0.5, -0.25), (-0.25, 0.0), (0.0, 0.25), (0.25, 0.5), (0.5, 1.0)]

# Speed bands for the speed-resolved gain (docs/adr/2026-07-17-speed-resolved-roll-comp-gain.md).
# The gain is fitted per band and applied as a continuous interp over v_ego — never a
# hard speed gate, which would step the torque on every threshold crossing of a crowned
# road. Below 5 m/s the integrator is frozen and the measurement smoother resets, so
# nothing is collected there. 100.0 stands in for +inf because
# format_roll_comp_profile forbids non-finite JSON.
ROLL_COMP_SPEED_BANDS = ((5.0, 10.0), (10.0, 15.0), (15.0, 100.0))
ROLL_COMP_PRIMARY_BAND = (15.0, 100.0)

_BAND_FIELDS = ('gain', 'points', 'span', 'confidence', 'blockCount', 'slopeRelSe')


def _fit_slope_ols(points: np.ndarray):
  """Scalar slope-only compatibility helper for Drive Lab."""
  try:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 2:
      return None
    x = points[:, 0]
    if points.shape[1] >= 3 and np.allclose(points[:, 1], 1.0):
      y = points[:, 2]
    else:
      y = points[:, 1]
    x_bar, y_bar = np.mean(x), np.mean(y)
    sxx = float(np.sum((x - x_bar) ** 2))
    if sxx <= 0.0:
      return None
    slope = float(np.sum((x - x_bar) * (y - y_bar)) / sxx)
    return slope if np.isfinite(slope) else None
  except (TypeError, ValueError):
    return None


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


class _RollCompPointBuckets(PointBuckets):
  def add_point(self, x, y, block_id=0):
    if block_id is None or not np.isfinite(x) or not np.isfinite(y) or not np.isfinite(block_id):
      return
    try:
      block_id = int(block_id)
    except (TypeError, ValueError, OverflowError):
      return
    if block_id < 0:
      return
    for bound_min, bound_max in self.x_bounds:
      if (x >= bound_min) and (x < bound_max):
        self.buckets[(bound_min, bound_max)].append([x, y, block_id])
        break


class RollCompBuckets:
  """Roll-comp points, bounded by the existing x-bucket ceilings."""

  def __init__(self, x_bounds=None, points_per_bucket=5000, speed_bands=None):
    self.x_bounds = x_bounds if x_bounds is not None else ROLL_COMP_BUCKET_BOUNDS
    self.speed_bands = tuple(speed_bands) if speed_bands is not None else ROLL_COMP_SPEED_BANDS
    min_points = [1.0] * len(self.x_bounds)
    self._bands = {band: _RollCompPointBuckets(x_bounds=self.x_bounds, min_points=min_points, min_points_total=1,
                                               points_per_bucket=points_per_bucket, rowsize=3)
                   for band in self.speed_bands}
    self._completed_through = -1

  @property
  def completed_block_ids(self) -> range:
    return range(self._completed_through + 1)

  def set_completed_through(self, block_id: int):
    self._completed_through = max(self._completed_through, int(block_id))

  def add_point(self, roll, torque_lat_accel, v_ego, block_id=0):
    if not np.isfinite(roll) or not np.isfinite(torque_lat_accel) or not np.isfinite(v_ego):
      return
    if block_id is None:
      return
    x = -np.sin(roll) * 9.81
    if not np.isfinite(x):
      return
    for v_lo, v_hi in self.speed_bands:
      if v_lo <= v_ego < v_hi:
        self._bands[(v_lo, v_hi)].add_point(x, torque_lat_accel, block_id)
        break

  def band_points(self, band, completed_block_ids=None):
    points = self._bands[band].get_points()
    if completed_block_ids is None:
      return points
    if isinstance(completed_block_ids, range):
      completed = completed_block_ids
    else:
      completed = set(completed_block_ids)
    if not completed:
      return np.empty((0, 3), dtype=float)
    block_ids = points[:, 2].astype(int)
    mask = ((block_ids >= completed.start) & (block_ids < completed.stop)
            if isinstance(completed, range) else np.isin(block_ids, list(completed)))
    return points[mask]

  def get_points(self, completed_block_ids=None):
    points = [self.band_points(band, completed_block_ids) for band in self.speed_bands]
    points = [point for point in points if len(point)]
    return np.vstack(points) if points else np.empty((0, 3), dtype=float)


def _populated_extremes(points):
  if len(points) < 2:
    return None, None
  xs = points[:, 0].astype(float)
  return float(np.percentile(xs, 5)), float(np.percentile(xs, 95))


def _informative_blocks(points):
  informative = []
  for block_id in sorted(set(points[:, 2].astype(int))):
    block_points = points[points[:, 2].astype(int) == block_id]
    lo_x, hi_x = _populated_extremes(block_points)
    if (len(block_points) >= MIN_POINTS_PER_BLOCK and lo_x is not None and hi_x is not None
        and hi_x - lo_x >= MIN_BLOCK_X_SPAN):
      informative.append(block_id)
  return informative


def _fit_band(points):
  informative = _informative_blocks(points)
  if len(informative) < MIN_EVIDENCE_BLOCKS:
    return None
  informative_points = points[np.isin(points[:, 2].astype(int), informative)]
  if len(informative_points) < MIN_POINTS:
    return None
  lo_x, hi_x = _populated_extremes(informative_points)
  if lo_x is None or hi_x is None:
    return None
  span = float(hi_x - lo_x)
  if lo_x >= 0 or hi_x <= 0 or span < MIN_X_SPAN:
    return None

  fit = fit_block_slope(informative_points)
  if fit is None or len(fit.block_slopes) != len(informative):
    return None
  if not np.isfinite(fit.slope) or fit.slope <= 0:
    return None
  if any(not np.isfinite(slope) or slope <= 0 for slope in fit.block_slopes.values()):
    return None
  if any(not np.isfinite(slope) or slope <= 0 for slope in fit.loo_slopes.values()):
    return None
  if not (ROLL_GAIN_MIN <= fit.slope <= ROLL_GAIN_MAX):
    return None
  if any(not (ROLL_GAIN_MIN <= slope <= ROLL_GAIN_MAX) for slope in fit.loo_slopes.values()):
    return None
  if not np.isfinite(fit.rel_se) or fit.rel_se > MAX_BLOCK_REL_SE:
    return None

  return {
    'gain': float(fit.slope),
    'points': int(len(informative_points)),
    'span': span,
    'confidence': float(min(1.0, len(informative) / (2 * MIN_EVIDENCE_BLOCKS))),
    'blockCount': int(len(informative)),
    'slopeRelSe': float(fit.rel_se),
  }


def _band_anchor(v_lo, v_hi):
  # midpoint for narrow bands; the wide (open-ended) top band anchors at v_lo + 5
  return v_lo + min((v_hi - v_lo) / 2.0, 5.0)


def _as_bands(profile):
  bands = profile.get('bands', []) if isinstance(profile, dict) else []
  return [dict(band) for band in bands]


def _assemble_profile(restore_key, bands):
  """Build a canonical configured-subset, banded profile."""
  profile = {
    'version': ROLL_COMP_PARAMS_VERSION,
    'restoreKey': restore_key,
    'bands': sorted((dict(band) for band in bands), key=lambda b: b['vLo']),
  }
  primary = next((b for b in profile['bands'] if (b['vLo'], b['vHi']) == ROLL_COMP_PRIMARY_BAND), None)
  if primary is not None:
    profile.update({k: primary[k] for k in _BAND_FIELDS})
  return profile


def fit_roll_comp_profile(CP: Any, buckets: RollCompBuckets, completed_block_ids=None):
  if CP.lateralTuning.which() != 'torque':
    return None
  if completed_block_ids is None:
    completed_block_ids = buckets.completed_block_ids
  bands = []
  for v_lo, v_hi in buckets.speed_bands:
    fitted = _fit_band(buckets.band_points((v_lo, v_hi), completed_block_ids))
    if fitted is not None:
      bands.append({'vLo': float(v_lo), 'vHi': float(v_hi), **fitted})
  if not bands:
    return None
  return _assemble_profile(_restore_key(CP), bands)


def roll_gain_at(profile, v_ego, base_gain):
  """Continuous speed-resolved gain with the base gain as the highway fallback."""
  try:
    bands = _as_bands(profile) if profile else []
    anchors = {_band_anchor(v_lo, v_hi): float(base_gain) for v_lo, v_hi in ROLL_COMP_SPEED_BANDS}
    for band in bands:
      gain = float(band['gain'])
      if not np.isfinite(gain) or not (ROLL_GAIN_MIN <= gain <= ROLL_GAIN_MAX):
        return float(base_gain)
      anchors[_band_anchor(float(band['vLo']), float(band['vHi']))] = gain
    xs = sorted(anchors)
    return float(np.interp(v_ego, xs, [anchors[x] for x in xs]))
  except (KeyError, TypeError, ValueError):
    return float(base_gain)


def replace_roll_comp_profile(old_profile, new_profile):
  """Replace a snapshot band-wise without averaging evidence or uncertainty."""
  if old_profile is None:
    return _assemble_profile(new_profile['restoreKey'], _as_bands(new_profile))

  old_bands = {(b['vLo'], b['vHi']): b for b in _as_bands(old_profile)}
  bands = []
  for band in _as_bands(new_profile):
    key = (band['vLo'], band['vHi'])
    old = old_bands.pop(key, None)
    if old is not None and abs(float(band['gain']) - float(old['gain'])) > ROLL_BAND_REPLACEMENT_MAX_DELTA:
      bands.append(dict(old))
    else:
      bands.append(dict(band))
  bands.extend(old_bands.values())
  return _assemble_profile(new_profile['restoreKey'], bands)


def format_roll_comp_profile(profile: dict) -> str:
  return json.dumps(profile, separators=(',', ':'), sort_keys=True, allow_nan=False)


def _validate_band_fields(entry):
  if set(entry) != {'vLo', 'vHi', *_BAND_FIELDS}:
    return None
  gain = _finite_float(entry.get('gain'))
  span = _finite_float(entry.get('span'))
  confidence = _finite_float(entry.get('confidence'))
  slope_rel_se = _finite_float(entry.get('slopeRelSe'))
  points = _finite_int(entry.get('points'))
  block_count = _finite_int(entry.get('blockCount'))
  if (gain is None or span is None or confidence is None or slope_rel_se is None or
      points is None or block_count is None):
    return None
  if points < MIN_POINTS or block_count < MIN_EVIDENCE_BLOCKS:
    return None
  if not (ROLL_GAIN_MIN <= gain <= ROLL_GAIN_MAX):
    return None
  expected_confidence = min(1.0, block_count / (2 * MIN_EVIDENCE_BLOCKS))
  if not (MIN_CONFIDENCE <= confidence <= 1.0) or abs(confidence - expected_confidence) > 1e-9:
    return None
  if (span < MIN_X_SPAN or points < block_count * MIN_POINTS_PER_BLOCK or
      not (0.0 <= slope_rel_se <= MAX_BLOCK_REL_SE)):
    return None
  return {
    'gain': gain,
    'points': points,
    'span': span,
    'confidence': confidence,
    'blockCount': block_count,
    'slopeRelSe': slope_rel_se,
  }


def parse_roll_comp_profile(CP: Any, payload):
  if CP.lateralTuning.which() != 'torque' or not isinstance(payload, dict):
    return None
  if payload.get('version') != ROLL_COMP_PARAMS_VERSION:
    return None
  if payload.get('restoreKey') != _restore_key(CP):
    return None
  raw_bands = payload.get('bands')
  if not isinstance(raw_bands, list) or not raw_bands:
    return None

  configured = set(ROLL_COMP_SPEED_BANDS)
  bands = []
  for entry in raw_bands:
    if not isinstance(entry, dict):
      return None
    v_lo = _finite_float(entry.get('vLo'))
    v_hi = _finite_float(entry.get('vHi'))
    if v_lo is None or v_hi is None or (v_lo, v_hi) not in configured:
      return None
    fields = _validate_band_fields(entry)
    if fields is None:
      return None
    bands.append({'vLo': v_lo, 'vHi': v_hi, **fields})
  if len({(b['vLo'], b['vHi']) for b in bands}) != len(bands):
    return None

  primary_present = ROLL_COMP_PRIMARY_BAND in {(b['vLo'], b['vHi']) for b in bands}
  top_fields_present = any(field in payload for field in _BAND_FIELDS)
  if top_fields_present != primary_present:
    return None
  expected_keys = {'version', 'restoreKey', 'bands'} | (set(_BAND_FIELDS) if primary_present else set())
  if set(payload) != expected_keys:
    return None
  if primary_present:
    top = _validate_band_fields({
      'vLo': ROLL_COMP_PRIMARY_BAND[0],
      'vHi': ROLL_COMP_PRIMARY_BAND[1],
      **{field: payload[field] for field in _BAND_FIELDS},
    })
    primary = next(b for b in bands if (b['vLo'], b['vHi']) == ROLL_COMP_PRIMARY_BAND)
    if top is None or any(top[field] != primary[field] for field in _BAND_FIELDS):
      return None

  return _assemble_profile(_restore_key(CP), bands)
