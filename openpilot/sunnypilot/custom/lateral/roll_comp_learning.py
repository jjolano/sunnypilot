import json
from typing import Any

import numpy as np

from openpilot.selfdrive.locationd.helpers import PointBuckets
from openpilot.sunnypilot.custom.lateral.speed_aware_torque import _restore_key

ROLL_COMP_PARAMS_VERSION = 1
ROLL_GAIN_MIN = 0.3
ROLL_GAIN_MAX = 1.0
MIN_POINTS = 2000
MIN_X_SPAN = 0.25
MIN_CONFIDENCE = 0.5

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

_BAND_FIELDS = ('gain', 'points', 'span', 'confidence')


# OLS, not the TLS fit speed_aware_torque uses: here x (-sin(roll)*g from the filtered
# localizer roll) is near noise-free while y carries control-activity noise of comparable
# magnitude to the x span, and TLS attributes that y-noise to the line — on route-246
# rlogs it read slope 1.15 where OLS reads 0.55 (the diagnosed vehicle response).
def _fit_slope_ols(points: np.ndarray):
  if points.shape[0] < 2:
    return None
  try:
    if float(np.var(points[:, 0].astype(float))) <= 1e-12:
      return None
    coef, *_ = np.linalg.lstsq(points[:, :2].astype(float), points[:, 2].astype(float), rcond=None)
    slope = float(coef[0])
    return slope if np.isfinite(slope) else None
  except Exception:
    return None


def _finite_float(value):
  try:
    f = float(value)
  except (TypeError, ValueError):
    return None
  return f if np.isfinite(f) else None


class _RollCompPointBuckets(PointBuckets):
  def add_point(self, x, y):
    if not np.isfinite(x) or not np.isfinite(y):
      return
    for bound_min, bound_max in self.x_bounds:
      if (x >= bound_min) and (x < bound_max):
        self.buckets[(bound_min, bound_max)].append([x, 1.0, y])
        break


class RollCompBuckets:
  """Roll-comp learning points, bucketed by roll magnitude within each speed band."""

  def __init__(self, x_bounds=None, points_per_bucket=5000, speed_bands=None):
    self.x_bounds = x_bounds if x_bounds is not None else ROLL_COMP_BUCKET_BOUNDS
    self.speed_bands = tuple(speed_bands) if speed_bands is not None else ROLL_COMP_SPEED_BANDS
    min_points = [1.0] * len(self.x_bounds)
    self._bands = {band: _RollCompPointBuckets(x_bounds=self.x_bounds, min_points=min_points, min_points_total=1,
                                               points_per_bucket=points_per_bucket, rowsize=3)
                   for band in self.speed_bands}

  def add_point(self, roll, torque_lat_accel, v_ego):
    if not np.isfinite(roll) or not np.isfinite(torque_lat_accel) or not np.isfinite(v_ego):
      return
    x = -np.sin(roll) * 9.81
    if not np.isfinite(x):
      return
    for v_lo, v_hi in self.speed_bands:
      if v_lo <= v_ego < v_hi:
        self._bands[(v_lo, v_hi)].add_point(x, torque_lat_accel)
        break

  def band_points(self, band):
    return self._bands[band].get_points()

  def get_points(self):
    return np.vstack([self._bands[band].get_points() for band in self.speed_bands])


def _populated_extremes(points):
  if len(points) < 2:
    return None, None
  xs = points[:, 0].astype(float)
  return float(np.percentile(xs, 5)), float(np.percentile(xs, 95))


def _fit_band(points):
  if len(points) < MIN_POINTS:
    return None
  lo_x, hi_x = _populated_extremes(points)
  if lo_x is None or hi_x is None:
    return None
  span = float(hi_x - lo_x)
  if lo_x >= 0 or hi_x <= 0 or span < MIN_X_SPAN:
    return None
  slope = _fit_slope_ols(points)
  if slope is None or not np.isfinite(slope) or slope <= 0:
    return None
  return {
    'gain': float(np.clip(slope, ROLL_GAIN_MIN, ROLL_GAIN_MAX)),
    'points': int(len(points)),
    'span': span,
    'confidence': float(min(1.0, len(points) / (MIN_POINTS * 2))),
  }


def _band_anchor(v_lo, v_hi):
  # midpoint for narrow bands; the wide (open-ended) top band anchors at v_lo + 5
  return v_lo + min((v_hi - v_lo) / 2.0, 5.0)


def _as_bands(profile):
  """Canonical band list from a profile dict (banded, or legacy top-level-only)."""
  if profile.get('bands'):
    return [dict(band) for band in profile['bands']]
  return [{'vLo': ROLL_COMP_PRIMARY_BAND[0], 'vHi': ROLL_COMP_PRIMARY_BAND[1],
           **{k: profile[k] for k in _BAND_FIELDS}}]


def _assemble_profile(restore_key, bands):
  """Build the canonical profile: banded, with top-level fields mirroring the primary
  (>=15 m/s) band when it is fitted — legacy readers and telemetry use the mirror."""
  profile = {
    'version': ROLL_COMP_PARAMS_VERSION,
    'restoreKey': restore_key,
    'bands': sorted(bands, key=lambda b: b['vLo']),
  }
  primary = next((b for b in profile['bands'] if (b['vLo'], b['vHi']) == ROLL_COMP_PRIMARY_BAND), None)
  if primary is not None:
    profile.update({k: primary[k] for k in _BAND_FIELDS})
  return profile


def fit_roll_comp_profile(CP: Any, buckets: RollCompBuckets):
  if CP.lateralTuning.which() != 'torque':
    return None
  bands = []
  for v_lo, v_hi in buckets.speed_bands:
    fitted = _fit_band(buckets.band_points((v_lo, v_hi)))
    if fitted is not None:
      bands.append({'vLo': float(v_lo), 'vHi': float(v_hi), **fitted})
  if not bands:
    return None
  return _assemble_profile(_restore_key(CP), bands)


def roll_gain_at(profile, v_ego, base_gain):
  """Continuous speed-resolved gain: interp over the fitted bands' anchor speeds.

  If the primary (highway) band is unfitted, its anchor is pinned to ``base_gain`` so
  a city-learned gain never flat-extends to highway speeds; conversely a lone primary
  band reproduces today's constant-gain behavior at every speed.
  """
  bands = _as_bands(profile) if profile else []
  anchors = {_band_anchor(b['vLo'], b['vHi']): float(b['gain']) for b in bands}
  primary_anchor = _band_anchor(*ROLL_COMP_PRIMARY_BAND)
  if primary_anchor not in anchors:
    anchors[primary_anchor] = float(base_gain)
  xs = sorted(anchors)
  return float(np.interp(v_ego, xs, [anchors[x] for x in xs]))


def _blend_band(old, new):
  old_weight = float(min(int(old['points']), 2 * MIN_POINTS))
  new_weight = float(min(int(new['points']), 2 * MIN_POINTS))
  weight_sum = old_weight + new_weight
  if weight_sum <= 0:
    return dict(new)
  blended = dict(new)
  blended['gain'] = float((old_weight * float(old['gain']) + new_weight * float(new['gain'])) / weight_sum)
  blended['points'] = int(min(int(old['points']) + int(new['points']), 4 * MIN_POINTS))
  blended['span'] = float(max(float(old['span']), float(new['span'])))
  blended['confidence'] = float(min(1.0, blended['points'] / (MIN_POINTS * 2)))
  return blended


def blend_roll_comp_profile(old_profile, new_profile):
  if old_profile is None:
    return _assemble_profile(new_profile['restoreKey'], _as_bands(new_profile))

  old_bands = {(b['vLo'], b['vHi']): b for b in _as_bands(old_profile)}
  bands = []
  for band in _as_bands(new_profile):
    key = (band['vLo'], band['vHi'])
    bands.append(_blend_band(old_bands.pop(key), band) if key in old_bands else band)
  # bands learned earlier but not refit this cycle carry forward untouched, so a
  # highway-only drive never discards city-band evidence (and vice versa)
  bands.extend(old_bands.values())
  return _assemble_profile(new_profile['restoreKey'], bands)


def format_roll_comp_profile(profile: dict) -> str:
  return json.dumps(profile, separators=(',', ':'), sort_keys=True, allow_nan=False)


def _validate_band_fields(entry):
  gain = _finite_float(entry.get('gain'))
  span = _finite_float(entry.get('span'))
  confidence = _finite_float(entry.get('confidence'))
  raw_points = entry.get('points')
  if gain is None or span is None or confidence is None or raw_points is None:
    return None
  try:
    points = int(raw_points)
  except (TypeError, ValueError):
    return None
  if points < MIN_POINTS:
    return None
  if gain < ROLL_GAIN_MIN or gain > ROLL_GAIN_MAX:
    return None
  if confidence < MIN_CONFIDENCE or confidence > 1:
    return None
  if span < MIN_X_SPAN:
    return None
  return {'gain': gain, 'points': points, 'span': span, 'confidence': confidence}


def parse_roll_comp_profile(CP: Any, payload):
  if CP.lateralTuning.which() != 'torque' or not isinstance(payload, dict):
    return None
  if payload.get('version') != ROLL_COMP_PARAMS_VERSION:
    return None
  if payload.get('restoreKey') != _restore_key(CP):
    return None

  # Fail closed on inconsistency: if any top-level mirror field is present they must
  # all be present and valid, and every band entry must be fully valid — a partial or
  # corrupt payload is distrusted wholesale rather than repaired.
  top = None
  if any(k in payload for k in _BAND_FIELDS):
    top = _validate_band_fields(payload)
    if top is None:
      return None

  raw_bands = payload.get('bands')
  if raw_bands is not None:
    if not isinstance(raw_bands, list):
      return None
    bands = []
    for entry in raw_bands:
      if not isinstance(entry, dict):
        return None
      v_lo = _finite_float(entry.get('vLo'))
      v_hi = _finite_float(entry.get('vHi'))
      fields = _validate_band_fields(entry)
      if v_lo is None or v_hi is None or v_lo >= v_hi or fields is None:
        return None
      bands.append({'vLo': v_lo, 'vHi': v_hi, **fields})
    if len({(b['vLo'], b['vHi']) for b in bands}) != len(bands):
      return None
  elif top is not None:
    # legacy scalar payload: the learned gain was fitted from >=15 m/s points only
    bands = [{'vLo': ROLL_COMP_PRIMARY_BAND[0], 'vHi': ROLL_COMP_PRIMARY_BAND[1], **top}]
  else:
    return None

  if not bands:
    return None
  return _assemble_profile(_restore_key(CP), bands)
