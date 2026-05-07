"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

import numpy as np

from cereal import car

from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
from openpilot.sunnypilot import PARAMS_UPDATE_PERIOD
from openpilot.sunnypilot.selfdrive.locationd.speed_aware_torque import (
  SPEED_BUCKET_BP,
  SPEED_BUCKET_LABELS,
  SPEED_AWARE_PARAMS_VERSION,
  SpeedAwareTorqueBuckets,
  format_speed_aware_params,
  parse_speed_aware_params,
)

RELAXED_MIN_BUCKET_POINTS = np.array([1, 200, 300, 500, 500, 300, 200, 1])


ALLOWED_CARS = ['toyota', 'hyundai', 'rivian', 'honda']


class TorqueEstimatorExt:
  def __init__(self, CP: car.CarParams):
    self.CP = CP
    self._params = Params()
    self.frame = -1

    self.enforce_torque_control_toggle = self._params.get_bool("EnforceTorqueControl")  # only during init
    self.use_params = self.CP.brand in ALLOWED_CARS and self.CP.lateralTuning.which() == 'torque'
    self.use_live_torque_params = self._params.get_bool("LiveTorqueParamsToggle")
    self.custom_torque_params = self._params.get_bool("CustomTorqueParams")
    self.torque_override_enabled = self._params.get_bool("TorqueParamsOverrideEnabled")
    self.min_bucket_points = RELAXED_MIN_BUCKET_POINTS
    self.factor_sanity = 0.0
    self.friction_sanity = 0.0
    self.offline_latAccelFactor = 0.0
    self.offline_friction = 0.0

  def initialize_custom_params(self, decimated=False):
    self.update_use_params()

    if self.enforce_torque_control_toggle:
      if self._params.get_bool("LiveTorqueParamsRelaxedToggle"):
        self.min_bucket_points = RELAXED_MIN_BUCKET_POINTS / (10 if decimated else 1)
        self.factor_sanity = 0.5 if decimated else 1.0
        self.friction_sanity = 0.8 if decimated else 1.0

      if self._params.get_bool("CustomTorqueParams"):
        self.offline_latAccelFactor = float(self._params.get("TorqueParamsOverrideLatAccelFactor", return_default=True))
        self.offline_friction = float(self._params.get("TorqueParamsOverrideFriction", return_default=True))

    self.speed_adaptive_enabled = self._params.get_bool("LiveTorqueSpeedAdaptiveToggle")
    self.speed_bucket_params = {}
    self.speed_bucket_filters = {}
    self._init_speed_buckets()

  def _init_speed_buckets(self):
    from openpilot.selfdrive.locationd.torqued import STEER_BUCKET_BOUNDS, POINTS_PER_BUCKET
    self.speed_buckets = SpeedAwareTorqueBuckets(
      x_bounds=STEER_BUCKET_BOUNDS,
      speed_bp=SPEED_BUCKET_BP,
      min_points=self.min_bucket_points,
      min_points_total=self.min_points_total,
      points_per_bucket=POINTS_PER_BUCKET,
      rowsize=3
    )

  def add_speed_aware_point(self, steer, lateral_acc, v_ego):
    if not self.speed_adaptive_enabled:
      return
    self.speed_buckets.add_point(steer, lateral_acc, v_ego)

  def estimate_speed_aware_params(self):
    """Returns dict of bucket_label -> (latAccelFactor, latAccelOffset, frictionCoeff)"""
    from openpilot.selfdrive.locationd.torqued import slope2rot, FRICTION_FACTOR
    result = {}
    for idx, label in enumerate(SPEED_BUCKET_LABELS):
      bucket = self.speed_buckets.buckets[idx]
      if not bucket.is_valid():
        continue
      points = bucket.get_points(self.fit_points)
      try:
        _, _, v = np.linalg.svd(points, full_matrices=False)
        slope, offset = -v.T[0:2, 2] / v.T[2, 2]
        _, spread = np.matmul(points[:, [0, 2]], slope2rot(slope)).T
        friction_coeff = np.std(spread) * FRICTION_FACTOR
        if any(not np.isfinite(val) for val in (slope, offset, friction_coeff)):
          continue
        slope = np.clip(slope, self.min_lataccel_factor, self.max_lataccel_factor)
        friction_coeff = np.clip(friction_coeff, self.min_friction, self.max_friction)
        result[label] = (float(slope), float(offset), float(friction_coeff))
      except np.linalg.LinAlgError:
        continue
    return result

  def _update_params(self):
    if self.frame % int(PARAMS_UPDATE_PERIOD / DT_MDL) == 0:
      self.use_live_torque_params = self._params.get_bool("LiveTorqueParamsToggle")
      self.custom_torque_params = self._params.get_bool("CustomTorqueParams")
      self.torque_override_enabled = self._params.get_bool("TorqueParamsOverrideEnabled")

  def update_use_params(self):
    self._update_params()

    if self.enforce_torque_control_toggle:
      if self.custom_torque_params and self.torque_override_enabled:
        self.use_params = False
      else:
        self.use_params = self.use_live_torque_params

    self.frame += 1
