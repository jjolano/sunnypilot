#!/usr/bin/env python3
import os
import numpy as np
from collections import deque, defaultdict

import openpilot.cereal.messaging as messaging
from openpilot.cereal import log
from opendbc.car.structs import car
from openpilot.common.constants import ACCELERATION_DUE_TO_GRAVITY
from openpilot.common.params import Params
from openpilot.common.realtime import config_realtime_process, DT_MDL
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.locationd.helpers import PointBuckets, ParameterEstimator, PoseCalibrator, Pose
from openpilot.sunnypilot.livedelay.helpers import get_lat_delay
from openpilot.sunnypilot.selfdrive.locationd.torqued_ext import TorqueEstimatorExt

HISTORY = 5  # secs
POINTS_PER_BUCKET = 1500
MIN_POINTS_TOTAL = 4000
MIN_POINTS_TOTAL_QLOG = 600
FIT_POINTS_TOTAL = 2000
FIT_POINTS_TOTAL_QLOG = 600
MIN_VEL = 15  # m/s
FRICTION_FACTOR = 1.5  # ~85% of data coverage
FACTOR_SANITY = 0.3
FACTOR_SANITY_QLOG = 0.5
FRICTION_SANITY = 0.5
FRICTION_SANITY_QLOG = 0.8
MAX_LATACCEL_OFFSET = 1.0  # m/s^2
STEER_MIN_THRESHOLD = 0.02
MIN_FILTER_DECAY = 50
MAX_FILTER_DECAY = 250
LAT_ACC_THRESHOLD = 1
STEER_BUCKET_BOUNDS = [(-0.5, -0.3), (-0.3, -0.2), (-0.2, -0.1), (-0.1, 0), (0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.5)]
MIN_BUCKET_POINTS = np.array([100, 300, 500, 500, 500, 500, 300, 100])
MIN_ENGAGE_BUFFER = 2  # secs

VERSION = 1  # bump this to invalidate old parameter caches
ALLOWED_CARS = ['toyota', 'hyundai', 'rivian', 'honda', 'volkswagen']
EPS_TORQUE_CARS = ['chrysler', 'gm', 'hyundai', 'mazda', 'psa', 'subaru', 'toyota']
# steeringTorqueEps is in brand-native units with the carcontroller's sign convention;
# torqued's steer points are -actuatorsOutput.torque (normalized). Divide by the brand
# STEER_MAX and negate to compare them. ponytail: toyota-only, add brands when observed
EPS_TORQUE_STEER_MAX = {'toyota': 1500.0}


def slope2rot(slope):
  sin = np.sqrt(slope ** 2 / (slope ** 2 + 1))
  cos = np.sqrt(1 / (slope ** 2 + 1))
  return np.array([[cos, -sin], [sin, cos]])


class TorqueBuckets(PointBuckets):
  def add_point(self, x, y):
    for bound_min, bound_max in self.x_bounds:
      if (x >= bound_min) and (x < bound_max):
        self.buckets[(bound_min, bound_max)].append([x, 1.0, y])
        break


class TorqueEstimator(ParameterEstimator, TorqueEstimatorExt):
  def __init__(self, CP, decimated=False, track_all_points=False):
    ParameterEstimator.__init__(self)
    TorqueEstimatorExt.__init__(self, CP)
    self.CP = CP
    self.hist_len = int(HISTORY / DT_MDL)
    self.lag = 0.0
    self.track_all_points = track_all_points  # for offline analysis, without max lateral accel or max steer torque filters
    if decimated:
      self.min_bucket_points: list[float] = (MIN_BUCKET_POINTS / 10).tolist()
      self.min_points_total = MIN_POINTS_TOTAL_QLOG
      self.fit_points = FIT_POINTS_TOTAL_QLOG
      self.factor_sanity = FACTOR_SANITY_QLOG
      self.friction_sanity = FRICTION_SANITY_QLOG

    else:
      self.min_bucket_points = MIN_BUCKET_POINTS.tolist()
      self.min_points_total = MIN_POINTS_TOTAL
      self.fit_points = FIT_POINTS_TOTAL
      self.factor_sanity = FACTOR_SANITY
      self.friction_sanity = FRICTION_SANITY

    self.offline_friction = 0.0
    self.offline_latAccelFactor = 0.0
    self.resets = 0.0
    self.use_params = CP.brand in ALLOWED_CARS and CP.lateralTuning.which() == 'torque'
    self.eps_shadow_stats_enabled = CP.brand in EPS_TORQUE_CARS

    if CP.lateralTuning.which() == 'torque':
      self.offline_friction = CP.lateralTuning.torque.friction
      self.offline_latAccelFactor = CP.lateralTuning.torque.latAccelFactor

    self.calibrator = PoseCalibrator()

    TorqueEstimatorExt.initialize_custom_params(self, decimated)

    self.reset()

    initial_params = {
      'latAccelFactor': self.offline_latAccelFactor,
      'latAccelOffset': 0.0,
      'frictionCoefficient': self.offline_friction,
      'points': []
    }
    self.decay = MIN_FILTER_DECAY
    self.min_lataccel_factor = (1.0 - self.factor_sanity) * self.offline_latAccelFactor
    self.max_lataccel_factor = (1.0 + self.factor_sanity) * self.offline_latAccelFactor
    self.min_friction = (1.0 - self.friction_sanity) * self.offline_friction
    self.max_friction = (1.0 + self.friction_sanity) * self.offline_friction

    # try to restore cached params
    params = Params()
    self.params = params
    params_cache = params.get("CarParamsPrevRoute")
    torque_cache = params.get("LiveTorqueParameters")
    if params_cache is not None and torque_cache is not None:
      try:
        with log.Event.from_bytes(torque_cache) as log_evt:
          cache_ltp = log_evt.liveTorqueParameters
        with car.CarParams.from_bytes(params_cache) as msg:
          cache_CP = msg
        if self.get_restore_key(cache_CP, cache_ltp.version) == self.get_restore_key(CP, VERSION):
          filtered_fields = (cache_ltp.latAccelFactorFiltered, cache_ltp.latAccelOffsetFiltered,
                             cache_ltp.frictionCoefficientFiltered)
          if cache_ltp.liveValid and not all(np.isfinite(v) for v in filtered_fields):
            cloudlog.exception("cached LiveTorqueParameters has non-finite filtered fields; discarding cache")
            params.remove("LiveTorqueParameters")
          else:
            if cache_ltp.liveValid:
              initial_params = {
                'latAccelFactor': cache_ltp.latAccelFactorFiltered,
                'latAccelOffset': float(np.clip(cache_ltp.latAccelOffsetFiltered, -MAX_LATACCEL_OFFSET, MAX_LATACCEL_OFFSET)),
                'frictionCoefficient': cache_ltp.frictionCoefficientFiltered
              }
            cached_points: list[list[float]] = [list(point) for point in cache_ltp.points]
            initial_params['points'] = cached_points
            self.decay = cache_ltp.decay
            self.filtered_points.load_points(initial_params['points'])
            cloudlog.info("restored torque params from cache")
      except Exception:
        cloudlog.exception("failed to restore cached torque params")
        params.remove("LiveTorqueParameters")

    self.filtered_params = {}
    for param in initial_params:
      self.filtered_params[param] = FirstOrderFilter(initial_params[param], self.decay, DT_MDL)

  @staticmethod
  def get_restore_key(CP, version):
    a, b = None, None
    if CP.lateralTuning.which() == 'torque':
      a = CP.lateralTuning.torque.friction
      b = CP.lateralTuning.torque.latAccelFactor
    return (CP.carFingerprint, CP.lateralTuning.which(), a, b, version)

  def reset(self):
    self.resets += 1.0
    self.decay = MIN_FILTER_DECAY
    self.raw_points = defaultdict(lambda: deque(maxlen=self.hist_len))
    self.filtered_points = TorqueBuckets(x_bounds=STEER_BUCKET_BOUNDS,
                                         min_points=self.min_bucket_points,
                                         min_points_total=self.min_points_total,
                                         points_per_bucket=POINTS_PER_BUCKET,
                                         rowsize=3)
    self.all_torque_points = []

    # Phase 0b shadow-only EPS torque observability. These metrics never affect
    # live parameter fitting, useParams, or control in this phase.
    self.eps_observed = False
    self.eps_sample_count = 0
    self.eps_torque_latest = 0.0
    self.eps_command_torque_latest = 0.0
    self.eps_delta_sum = 0.0
    self.eps_delta_max = 0.0

  def update_eps_shadow_stats(self, t, steer):
    # Interpolate EPS torque at the learning-point timestamp, ignoring any
    # missing or non-finite samples so they do not contaminate the stats.
    if not getattr(self, 'eps_shadow_stats_enabled', False):
      self.eps_command_torque_latest = float(steer)
      return

    eps_torque = None
    if len(self.raw_points.get('steering_torque_eps', [])):
      times = []
      values = []
      for tt, vv in zip(self.raw_points['carState_t'], self.raw_points['steering_torque_eps'], strict=True):
        if np.isfinite(vv):
          times.append(tt)
          values.append(vv)
      if times:
        eps_torque = float(np.interp(t, times, values))

    self.eps_command_torque_latest = float(steer)
    eps_steer_max = EPS_TORQUE_STEER_MAX.get(self.CP.brand)
    if eps_steer_max is not None and eps_torque is not None and np.isfinite(eps_torque):
      eps_torque = -eps_torque / eps_steer_max
      delta = abs(float(steer) - eps_torque)
      self.eps_observed = True
      self.eps_torque_latest = eps_torque
      self.eps_sample_count += 1
      self.eps_delta_sum += delta
      self.eps_delta_max = max(self.eps_delta_max, delta)

  def estimate_params(self):
    points = self.filtered_points.get_points(self.fit_points)
    # total least square solution as both x and y are noisy observations
    # this is empirically the slope of the hysteresis parallelogram as opposed to the line through the diagonals
    try:
      _, _, v = np.linalg.svd(points, full_matrices=False)
      slope, offset = -v.T[0:2, 2] / v.T[2, 2]
      _, spread = np.matmul(points[:, [0, 2]], slope2rot(slope)).T
      friction_coeff = np.std(spread) * FRICTION_FACTOR
    except np.linalg.LinAlgError as e:
      cloudlog.exception(f"Error computing live torque params: {e}")
      slope = offset = friction_coeff = np.nan
    return slope, offset, friction_coeff

  def update_params(self, params):
    self.decay = min(self.decay + DT_MDL, MAX_FILTER_DECAY)
    for param, value in params.items():
      self.filtered_params[param].update(value)
      self.filtered_params[param].update_alpha(self.decay)

  def handle_log(self, t, which, msg):
    if which == "carControl":
      self.raw_points["carControl_t"].append(t + self.lag)
      self.raw_points["lat_active"].append(msg.latActive)
    elif which == "carOutput":
      self.raw_points["carOutput_t"].append(t + self.lag)
      self.raw_points["steer_torque"].append(-msg.actuatorsOutput.torque)
    elif which == "carState":
      self.raw_points["carState_t"].append(t + self.lag)
      # TODO: check if high aEgo affects resulting lateral accel
      self.raw_points["vego"].append(msg.vEgo)
      self.raw_points["steer_override"].append(msg.steeringPressed)
      self.raw_points["steering_rate_deg"].append(msg.steeringRateDeg)
      self.raw_points["steering_torque_eps"].append(float(msg.steeringTorqueEps))
      eps_steer_max = EPS_TORQUE_STEER_MAX.get(self.CP.brand)
      if eps_steer_max is not None:
        self.update_breakaway_observer(t, msg.steeringRateDeg, float(msg.steeringTorqueEps) / eps_steer_max,
                                       msg.vEgo, msg.steeringTorque)
    elif which == "liveCalibration":
      self.calibrator.feed_live_calib(msg)
    elif which == "liveDelay":
      self.lag = get_lat_delay(self.params, msg.lateralDelay)
    # calculate lateral accel from past steering torque
    elif which == "livePose":
      is_valid = msg.angularVelocityDevice.valid and msg.orientationNED.valid and msg.inputsOK and msg.sensorsOK and msg.posenetOK
      if len(self.raw_points['steer_torque']) == self.hist_len and is_valid:
        t = msg.timestamp * 1e-9
        device_pose = Pose.from_live_pose(msg)
        calibrated_pose = self.calibrator.build_calibrated_pose(device_pose)
        angular_velocity_calibrated = calibrated_pose.angular_velocity

        yaw_rate = angular_velocity_calibrated.yaw
        roll = device_pose.orientation.roll
        speed_shadow_mode = self.speed_adaptive_mode in ('shadow', 'apply')
        roll_comp_mode = self.roll_comp_mode in ('shadow', 'apply')
        direction_gain_mode = self.direction_gain_mode in ('shadow', 'apply')
        shadow_collection_mode = roll_comp_mode or direction_gain_mode or (speed_shadow_mode and self.low_speed_shadow)
        # check lat active up to now (without lag compensation)
        lat_active = np.interp(np.arange(t - MIN_ENGAGE_BUFFER, t + self.lag, DT_MDL),
                               self.raw_points['carControl_t'], self.raw_points['lat_active']).astype(bool)
        steer_override = np.interp(np.arange(t - MIN_ENGAGE_BUFFER, t + self.lag, DT_MDL),
                                   self.raw_points['carState_t'], self.raw_points['steer_override']).astype(bool)
        vego = np.interp(t, self.raw_points['carState_t'], self.raw_points['vego'])
        steer = np.interp(t, self.raw_points['carOutput_t'], self.raw_points['steer_torque']).item()
        lateral_acc = (vego * yaw_rate) - (np.sin(roll) * ACCELERATION_DUE_TO_GRAVITY).item()
        if all(lat_active) and not any(steer_override):
          if shadow_collection_mode:
            # Phase 3 shadow learning only needs steering-rate interpolation when
            # one of the shadow/apply modes is active.
            steering_rate = None
            if len(self.raw_points['steering_rate_deg']):
              steering_rate = np.interp(t, self.raw_points['carState_t'], self.raw_points['steering_rate_deg']).item()
            self.collect_shadow_learning_points(steer, lateral_acc, vego, roll, yaw_rate, steering_rate, t)

          if (vego > MIN_VEL) and (abs(steer) > STEER_MIN_THRESHOLD):
            if abs(lateral_acc) <= LAT_ACC_THRESHOLD:
              if speed_shadow_mode:
                self.add_torque_learning_point(steer, lateral_acc, vego)
              self.filtered_points.add_point(steer, lateral_acc)
              self.eps_command_torque_latest = float(steer)
              if self.eps_shadow_stats_enabled:
                self.update_eps_shadow_stats(t, steer)

            if self.track_all_points:
              self.all_torque_points.append([steer, lateral_acc])

  def get_msg(self, valid=True, with_points=False):
    msg = messaging.new_message('liveTorqueParameters')
    msg.valid = valid
    liveTorqueParameters = msg.liveTorqueParameters
    liveTorqueParameters.version = VERSION
    liveTorqueParameters.useParams = self.use_params

    # Calculate raw estimates when possible, only update filters when enough points are gathered
    if self.filtered_points.is_calculable():
      latAccelFactor, latAccelOffset, frictionCoeff = self.estimate_params()
      liveTorqueParameters.latAccelFactorRaw = float(latAccelFactor)
      liveTorqueParameters.latAccelOffsetRaw = float(latAccelOffset)
      liveTorqueParameters.frictionCoefficientRaw = float(frictionCoeff)

      if self.filtered_points.is_valid():
        raw_estimates = [latAccelFactor, latAccelOffset, frictionCoeff]
        if any(val is None or not np.isfinite(val) for val in raw_estimates):
          cloudlog.exception("Live torque parameters are non-finite.")
          liveTorqueParameters.liveValid = False
          self.reset()
        elif abs(latAccelOffset) > MAX_LATACCEL_OFFSET:
          cloudlog.exception("Live torque offset exceeds sane bound.")
          liveTorqueParameters.liveValid = False
          self.reset()
        else:
          liveTorqueParameters.liveValid = True
          latAccelFactor = np.clip(latAccelFactor, self.min_lataccel_factor, self.max_lataccel_factor)
          latAccelOffset = float(np.clip(latAccelOffset, -MAX_LATACCEL_OFFSET, MAX_LATACCEL_OFFSET))
          frictionCoeff = np.clip(frictionCoeff, self.min_friction, self.max_friction)
          self.update_params({'latAccelFactor': latAccelFactor, 'latAccelOffset': latAccelOffset, 'frictionCoefficient': frictionCoeff})

    if with_points:
      liveTorqueParameters.points = self.filtered_points.get_points()[:, [0, 2]].tolist()

    if not all(np.isfinite(self.filtered_params[p].x) for p in ('latAccelFactor', 'latAccelOffset', 'frictionCoefficient')):
      cloudlog.exception("Live torque filtered parameters are non-finite.")
      liveTorqueParameters.liveValid = False
      self.reset()
      self.filtered_params['latAccelFactor'].x = self.offline_latAccelFactor
      self.filtered_params['latAccelOffset'].x = 0.0
      self.filtered_params['frictionCoefficient'].x = self.offline_friction

    liveTorqueParameters.latAccelFactorFiltered = float(self.filtered_params['latAccelFactor'].x)
    liveTorqueParameters.latAccelOffsetFiltered = float(self.filtered_params['latAccelOffset'].x)
    liveTorqueParameters.frictionCoefficientFiltered = float(self.filtered_params['frictionCoefficient'].x)
    liveTorqueParameters.totalBucketPoints = len(self.filtered_points)
    liveTorqueParameters.calPerc = self.filtered_points.get_valid_percent()
    liveTorqueParameters.decay = self.decay
    liveTorqueParameters.maxResets = self.resets

    # Phase 0b shadow-only EPS torque observability.
    liveTorqueParameters.epsObserved = self.eps_observed
    liveTorqueParameters.epsSampleCount = self.eps_sample_count
    liveTorqueParameters.epsTorqueLatest = float(self.eps_torque_latest)
    liveTorqueParameters.epsCommandTorqueLatest = float(self.eps_command_torque_latest)
    liveTorqueParameters.epsDeltaMean = float(self.eps_delta_sum / self.eps_sample_count if self.eps_sample_count > 0 else 0.0)
    liveTorqueParameters.epsDeltaMax = float(self.eps_delta_max)

    # Phase 3 shadow-only roll-compensation gain telemetry.
    liveTorqueParameters.rollCompGainLearned = self.roll_comp_profile['gain']
    liveTorqueParameters.rollCompGainPoints = self.roll_comp_profile['points']
    liveTorqueParameters.rollCompGainSpan = self.roll_comp_profile['span']
    liveTorqueParameters.rollCompGainValid = self.roll_comp_profile['valid']
    liveTorqueParameters.rollCompBandGains = self.roll_comp_profile['bandGains']
    liveTorqueParameters.rollCompBandPoints = self.roll_comp_profile['bandPoints']

    # Shadow-only rack breakaway observer telemetry.
    breakaway = self.breakaway_telemetry()
    liveTorqueParameters.breakawayLeftMedian = breakaway['left']
    liveTorqueParameters.breakawayRightMedian = breakaway['right']
    liveTorqueParameters.breakawayEvents = breakaway['events']
    band_medians, band_counts = self.breakaway_band_telemetry()
    liveTorqueParameters.breakawayBandMedians = band_medians
    liveTorqueParameters.breakawayBandCounts = band_counts

    # Direction-gain asymmetry learner telemetry.
    liveTorqueParameters.directionGainRatio = self.direction_gain_telemetry['ratio']
    liveTorqueParameters.directionGainPoints = self.direction_gain_telemetry['points']
    liveTorqueParameters.directionGainValid = self.direction_gain_telemetry['valid']
    return msg


def main(demo=False):
  config_realtime_process([0, 1, 2, 3], 5)

  DEBUG = bool(int(os.getenv("DEBUG", "0")))

  pm = messaging.PubMaster(['liveTorqueParameters'])
  sm = messaging.SubMaster(['carControl', 'carOutput', 'carState', 'liveCalibration', 'livePose', 'liveDelay'], poll='livePose')

  params = Params()
  estimator = TorqueEstimator(messaging.log_from_bytes(params.get("CarParams", block=True), car.CarParams))

  while True:
    sm.update()
    if sm.all_checks():
      for which in sm.updated.keys():
        if sm.updated[which]:
          t = sm.logMonoTime[which] * 1e-9
          estimator.handle_log(t, which, sm[which])

    TorqueEstimatorExt.update_use_params(estimator)

    # 4Hz driven by livePose
    if sm.frame % 5 == 0:
      pm.send('liveTorqueParameters', estimator.get_msg(valid=sm.all_checks(), with_points=DEBUG))

    # Cache points every 60 seconds while onroad
    if sm.frame % 240 == 0:
      msg = estimator.get_msg(valid=sm.all_checks(), with_points=True)
      params.put("LiveTorqueParameters", msg.to_bytes())
      if estimator.speed_adaptive_mode in ('shadow', 'apply') or estimator.roll_comp_mode in ('shadow', 'apply'):
        estimator.maybe_persist_speed_profile(cache_write=True)


if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser(description='Process the --demo argument.')
  parser.add_argument('--demo', action='store_true', help='A boolean for demo mode.')
  args = parser.parse_args()
  main(demo=args.demo)
