import numpy as np

from cereal import car

from openpilot.selfdrive.locationd.helpers import PointBuckets


SPEED_BUCKET_BP = [0, 10, 20, 30, 40]  # m/s
SPEED_BUCKET_LABELS = ["0_10", "10_20", "20_30", "30_40", "40_plus"]
SPEED_AWARE_PARAMS_VERSION = 1


def _payload_float_matches(payload: dict, key: str, expected: float) -> bool:
  try:
    return bool(np.isclose(float(payload.get(key)), expected))
  except (TypeError, ValueError):
    return False


def format_speed_aware_params(CP: car.CarParams, buckets: dict) -> dict | None:
  if CP.lateralTuning.which() != 'torque':
    return None
  return {
    "version": SPEED_AWARE_PARAMS_VERSION,
    "carFingerprint": CP.carFingerprint,
    "lateralTuning": CP.lateralTuning.which(),
    "torqueLatAccelFactor": float(CP.lateralTuning.torque.latAccelFactor),
    "torqueFriction": float(CP.lateralTuning.torque.friction),
    "buckets": buckets,
  }


def parse_speed_aware_params(CP: car.CarParams, payload: dict) -> dict | None:
  if CP.lateralTuning.which() != 'torque':
    return None
  if not isinstance(payload, dict):
    return None
  if payload.get("version") != SPEED_AWARE_PARAMS_VERSION:
    return None
  if payload.get("carFingerprint") != CP.carFingerprint:
    return None
  if payload.get("lateralTuning") != CP.lateralTuning.which():
    return None
  if not _payload_float_matches(payload, "torqueLatAccelFactor", float(CP.lateralTuning.torque.latAccelFactor)):
    return None
  if not _payload_float_matches(payload, "torqueFriction", float(CP.lateralTuning.torque.friction)):
    return None

  buckets = payload.get("buckets")
  if not isinstance(buckets, dict):
    return None

  parsed_buckets = {}
  for label, value in buckets.items():
    if label not in SPEED_BUCKET_LABELS or not isinstance(value, (tuple, list)) or len(value) != 3:
      return None
    try:
      bucket_values = tuple(float(v) for v in value)
    except (TypeError, ValueError):
      return None
    if any(not np.isfinite(v) for v in bucket_values):
      return None
    parsed_buckets[label] = bucket_values

  return parsed_buckets if parsed_buckets else None


class _TorqueBuckets(PointBuckets):
  def add_point(self, x, y):
    for bound_min, bound_max in self.x_bounds:
      if (x >= bound_min) and (x < bound_max):
        self.buckets[(bound_min, bound_max)].append([x, 1.0, y])
        break


class SpeedAwareTorqueBuckets:
  def __init__(self, x_bounds, speed_bp, min_points, min_points_total, points_per_bucket, rowsize=3):
    self.x_bounds = x_bounds
    self.speed_bp = list(speed_bp)
    self.min_points = min_points
    self.min_points_total = min_points_total
    self.points_per_bucket = points_per_bucket
    self.rowsize = rowsize
    self.buckets = {}
    self._init_buckets()

  def _init_buckets(self):
    for i in range(len(self.speed_bp)):
      self.buckets[i] = _TorqueBuckets(
        x_bounds=self.x_bounds,
        min_points=self.min_points,
        min_points_total=self.min_points_total,
        points_per_bucket=self.points_per_bucket,
        rowsize=self.rowsize
      )

  def _bucket_idx(self, v_ego):
    for i in range(len(self.speed_bp) - 1):
      if self.speed_bp[i] <= v_ego < self.speed_bp[i + 1]:
        return i
    return len(self.speed_bp) - 1

  def add_point(self, x, y, v_ego):
    idx = self._bucket_idx(v_ego)
    self.buckets[idx].add_point(x, y)

  def buckets_for_speed(self, v_ego):
    return self.buckets[self._bucket_idx(v_ego)]

  def is_calculable(self):
    return any(b.is_calculable() for b in self.buckets.values())

  def is_valid(self):
    return any(b.is_valid() for b in self.buckets.values())

  def get_points(self, n=None):
    all_pts = []
    for b in self.buckets.values():
      pts = b.get_points(n)
      if len(pts) > 0:
        all_pts.append(pts)
    if not all_pts:
      return np.empty((0, self.rowsize))
    return np.vstack(all_pts)

  def get_valid_percent(self):
    vals = [b.get_valid_percent() for b in self.buckets.values() if b.is_calculable()]
    return float(np.mean(vals)) if vals else 0.0

  def total_points(self):
    return sum(len(b) for b in self.buckets.values())
