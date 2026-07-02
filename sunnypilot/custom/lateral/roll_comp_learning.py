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
  def __init__(self, x_bounds=None, points_per_bucket=5000):
    x_bounds = x_bounds if x_bounds is not None else ROLL_COMP_BUCKET_BOUNDS
    min_points = [1.0] * len(x_bounds)
    self.x_bounds = x_bounds
    self.buckets = _RollCompPointBuckets(x_bounds=x_bounds, min_points=min_points, min_points_total=1,
                                         points_per_bucket=points_per_bucket, rowsize=3)

  def add_point(self, roll, torque_lat_accel, v_ego):
    if not np.isfinite(roll) or not np.isfinite(torque_lat_accel) or not np.isfinite(v_ego):
      return
    x = -np.sin(roll) * 9.81
    if not np.isfinite(x):
      return
    self.buckets.add_point(x, torque_lat_accel)

  def get_points(self):
    return self.buckets.get_points()


def _populated_extremes(points):
  if len(points) < 2:
    return None, None
  xs = points[:, 0].astype(float)
  return float(np.percentile(xs, 5)), float(np.percentile(xs, 95))


def fit_roll_comp_profile(CP: Any, buckets: RollCompBuckets):
  if CP.lateralTuning.which() != 'torque':
    return None
  points = buckets.get_points()
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
  gain = float(np.clip(slope, ROLL_GAIN_MIN, ROLL_GAIN_MAX))
  confidence = float(min(1.0, len(points) / (MIN_POINTS * 2)))
  return {
    'version': ROLL_COMP_PARAMS_VERSION,
    'restoreKey': _restore_key(CP),
    'gain': gain,
    'points': int(len(points)),
    'span': span,
    'confidence': confidence,
  }


def format_roll_comp_profile(profile: dict) -> str:
  return json.dumps(profile, separators=(',', ':'), sort_keys=True, allow_nan=False)


def parse_roll_comp_profile(CP: Any, payload):
  if CP.lateralTuning.which() != 'torque' or not isinstance(payload, dict):
    return None
  if payload.get('version') != ROLL_COMP_PARAMS_VERSION:
    return None
  if payload.get('restoreKey') != _restore_key(CP):
    return None
  gain = _finite_float(payload.get('gain'))
  raw_points = payload.get('points')
  span = _finite_float(payload.get('span'))
  confidence = _finite_float(payload.get('confidence'))
  if raw_points is None or gain is None or span is None or confidence is None:
    return None
  try:
    points = int(raw_points)
  except (TypeError, ValueError):
    return None
  if points < 0:
    return None
  if points < MIN_POINTS:
    return None
  if gain < ROLL_GAIN_MIN or gain > ROLL_GAIN_MAX:
    return None
  if confidence < MIN_CONFIDENCE or confidence > 1:
    return None
  if span < MIN_X_SPAN:
    return None
  return {
    'version': ROLL_COMP_PARAMS_VERSION,
    'restoreKey': _restore_key(CP),
    'gain': gain,
    'points': points,
    'span': span,
    'confidence': confidence,
  }
