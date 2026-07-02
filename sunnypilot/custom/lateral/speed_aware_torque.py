import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from openpilot.selfdrive.locationd.helpers import PointBuckets

SPEED_AWARE_TORQUE_PARAMS_VERSION = 1
SPEED_BUCKET_BP = [15.0, 20.0, 25.0, 30.0, 40.0]
LOW_SPEED_BUCKET_BP = [5.0, 10.0]
SPEED_BUCKET_LABELS = [f"{int(bp)}_plus" for bp in SPEED_BUCKET_BP]
MIN_RATIO = 0.75
MAX_RATIO = 1.25
MIN_BIN_POINTS = 200
MIN_GLOBAL_POINTS = 500
MIN_CONFIDENCE = 0.6
MAX_SPEED_ANCHORS = 16


def _finite_float(value):
  try:
    f = float(value)
  except (TypeError, ValueError):
    return None
  return f if np.isfinite(f) else None


def _restore_key(CP: Any):
  tuning = CP.lateralTuning.which()
  torque = CP.lateralTuning.torque if tuning == 'torque' else None
  return {
    'carFingerprint': CP.carFingerprint,
    'lateralTuning': tuning,
    'latAccelFactor': float(torque.latAccelFactor) if torque is not None else None,
    'friction': float(torque.friction) if torque is not None else None,
  }


def _fit_slope(points: np.ndarray):
  if points.shape[0] < 2:
    return None
  try:
    if float(np.var(points[:, 0].astype(float))) <= 1e-12:
      return None
    _, _, v = np.linalg.svd(points, full_matrices=False)
    denom = float(v.T[2, 2])
    if abs(denom) <= 1e-12:
      return None
    slope, _ = -v.T[0:2, 2] / denom
    slope = float(slope)
    return slope if np.isfinite(slope) else None
  except Exception:
    return None


class _TorqueBuckets(PointBuckets):
  def add_point(self, x, y):
    if not np.isfinite(x) or not np.isfinite(y):
      return
    for bound_min, bound_max in self.x_bounds:
      if (x >= bound_min) and (x < bound_max):
        self.buckets[(bound_min, bound_max)].append([x, 1.0, y])
        break


class SpeedAwareTorqueBuckets:
  def __init__(self, x_bounds, speed_bp, min_points, min_points_total, points_per_bucket, rowsize=3):
    self.x_bounds = x_bounds
    self.speed_bp = list(speed_bp)
    self.min_points = ([min_points] * len(x_bounds)) if np.isscalar(min_points) else min_points
    self.min_points_total = min_points_total
    self.points_per_bucket = points_per_bucket
    self.rowsize = rowsize
    self.buckets = {i: _TorqueBuckets(x_bounds=x_bounds, min_points=self.min_points, min_points_total=min_points_total,
                                      points_per_bucket=points_per_bucket, rowsize=rowsize)
                    for i in range(len(self.speed_bp))}

  def _bucket_idx(self, v_ego):
    if v_ego < self.speed_bp[0]:
      return None
    for i in range(len(self.speed_bp) - 1):
      if self.speed_bp[i] <= v_ego < self.speed_bp[i + 1]:
        return i
    return len(self.speed_bp) - 1

  def add_point(self, x, y, v_ego):
    if not np.isfinite(x) or not np.isfinite(y) or not np.isfinite(v_ego):
      return
    idx = self._bucket_idx(v_ego)
    if idx is None:
      return
    self.buckets[idx].add_point(x, y)

  def bucket_items(self):
    return list(self.buckets.items())


def _fit_speed_aware_section(buckets: SpeedAwareTorqueBuckets, global_slope: float, *, clamp_ratios: bool = True):
  anchors = []
  ratios = []
  slopes = []
  confidence = []
  points = []
  for idx, (_speed, bucket) in enumerate(buckets.bucket_items()):
    pts = bucket.get_points()
    n = int(len(pts))
    slope = _fit_slope(pts) if n >= MIN_BIN_POINTS else None
    if slope is None or not np.isfinite(slope) or slope <= 0:
      ratio = 1.0
      conf = 0.0
    else:
      ratio = float(slope / global_slope)
      if clamp_ratios:
        ratio = float(np.clip(ratio, MIN_RATIO, MAX_RATIO))
      conf = float(min(1.0, n / (MIN_BIN_POINTS * 2)))
    anchors.append(float(buckets.speed_bp[idx]))
    ratios.append(float(ratio))
    slopes.append(slope)
    confidence.append(float(conf))
    points.append(int(n))
  return {'anchors': anchors, 'ratios': ratios, 'slopes': slopes, 'confidence': confidence, 'points': points}


def fit_low_speed_section(CP: Any, buckets: SpeedAwareTorqueBuckets, global_slope: float | None = None):
  """Fit an evidence-oriented low-speed section from a separate bucket set.

  Runtime/apply paths ignore this section by construction; it is reported only.
  """
  if CP.lateralTuning.which() != 'torque':
    return None
  if global_slope is None:
    global_points = []
    for _, bucket in buckets.bucket_items():
      pts = bucket.get_points()
      if len(pts) > 0:
        global_points.append(pts)
    if not global_points:
      return None
    global_points = np.vstack(global_points)
    if len(global_points) < MIN_GLOBAL_POINTS:
      return None
    global_slope = _fit_slope(global_points)
    if global_slope is None or not np.isfinite(global_slope) or global_slope <= 0:
      return None
  return _fit_speed_aware_section(buckets, global_slope, clamp_ratios=False)


def fit_speed_aware_torque_profile(CP: Any, buckets: SpeedAwareTorqueBuckets, low_speed_buckets: SpeedAwareTorqueBuckets | None = None):
  if CP.lateralTuning.which() != 'torque':
    return None
  global_points = []
  for _, bucket in buckets.bucket_items():
    pts = bucket.get_points()
    if len(pts) > 0:
      global_points.append(pts)
  if not global_points:
    return None
  global_points = np.vstack(global_points)
  if len(global_points) < MIN_GLOBAL_POINTS:
    return None
  global_slope = _fit_slope(global_points)
  if global_slope is None or not np.isfinite(global_slope) or global_slope <= 0:
    return None

  section = _fit_speed_aware_section(buckets, global_slope)
  profile = {
    'version': SPEED_AWARE_TORQUE_PARAMS_VERSION,
    'restoreKey': _restore_key(CP),
    'anchors': section['anchors'],
    'ratios': section['ratios'],
    'confidence': section['confidence'],
    'points': section['points'],
    'globalLatAccelFactor': float(CP.lateralTuning.torque.latAccelFactor),
    'globalFriction': float(CP.lateralTuning.torque.friction),
  }
  if low_speed_buckets is not None:
    low_section = fit_low_speed_section(CP, low_speed_buckets, global_slope=global_slope)
    if low_section is not None:
      profile['lowSpeed'] = low_section
  return profile


def format_speed_aware_torque_profile(profile: dict) -> str:
  return json.dumps(profile, separators=(',', ':'), sort_keys=True, allow_nan=False)


def parse_speed_aware_torque_profile(CP: Any, payload):
  if CP.lateralTuning.which() != 'torque' or not isinstance(payload, dict):
    return None
  if payload.get('version') != SPEED_AWARE_TORQUE_PARAMS_VERSION:
    return None
  if payload.get('restoreKey') != _restore_key(CP):
    return None
  anchors = payload.get('anchors')
  ratios = payload.get('ratios')
  confidence = payload.get('confidence')
  points = payload.get('points')
  if not all(isinstance(x, list) for x in [anchors, ratios, confidence, points]):
    return None
  if not (len(anchors) == len(ratios) == len(confidence) == len(points) and len(anchors) > 0):
    return None
  parsed = {}
  for k in ('globalLatAccelFactor', 'globalFriction'):
    v = _finite_float(payload.get(k))
    if v is None or v < 0:
      return None
    parsed[k] = v
  if parsed['globalLatAccelFactor'] <= 0:
    return None
  parsed['anchors'] = []
  parsed['ratios'] = []
  parsed['confidence'] = []
  parsed['points'] = []
  last_anchor = None
  if len(anchors) > MAX_SPEED_ANCHORS:
    return None
  for a, r, c, n in zip(anchors, ratios, confidence, points, strict=True):
    a = _finite_float(a)
    r = _finite_float(r)
    c = _finite_float(c)
    try:
      n = int(n)
    except (TypeError, ValueError):
      return None
    if a is None or r is None or c is None or n < 0 or r < MIN_RATIO or r > MAX_RATIO or c < 0 or c > 1:
      return None
    if last_anchor is not None and a <= last_anchor:
      return None
    last_anchor = a
    parsed['anchors'].append(a)
    parsed['ratios'].append(r)
    parsed['confidence'].append(c)
    parsed['points'].append(n)
  return parsed


@dataclass
class SpeedAwareTorqueRuntime:
  profile: Any = None

  def ratio(self, v_ego):
    if self.profile is None:
      return 1.0
    v = _finite_float(v_ego)
    if v is None:
      return 1.0
    anchors = self.profile['anchors']
    if v < anchors[0]:
      return 1.0
    for i, a in enumerate(anchors):
      if v == a:
        return float(self.profile['ratios'][i]) if self.profile['confidence'][i] >= MIN_CONFIDENCE else 1.0
      if v < a:
        lo = i - 1
        if lo < 0:
          return 1.0
        if self.profile['confidence'][lo] < MIN_CONFIDENCE or self.profile['confidence'][i] < MIN_CONFIDENCE:
          return 1.0
        t = (v - anchors[lo]) / (anchors[i] - anchors[lo])
        return float(np.clip((1 - t) * self.profile['ratios'][lo] + t * self.profile['ratios'][i], MIN_RATIO, MAX_RATIO))
    last = len(anchors) - 1
    return float(self.profile['ratios'][last]) if self.profile['confidence'][last] >= MIN_CONFIDENCE else 1.0
