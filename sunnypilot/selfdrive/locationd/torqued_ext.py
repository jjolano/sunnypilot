"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import numpy as np
import json

from cereal import car

from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
from openpilot.sunnypilot import PARAMS_UPDATE_PERIOD
from openpilot.sunnypilot.custom.lateral.roll_comp_learning import (
  ROLL_COMP_SPEED_BANDS, RollCompBuckets, blend_roll_comp_profile, fit_roll_comp_profile, format_roll_comp_profile,
  parse_roll_comp_profile,
)
from openpilot.sunnypilot.custom.lateral.speed_aware_torque import (
  SpeedAwareTorqueBuckets, blend_speed_aware_torque_profile, fit_speed_aware_torque_profile,
  format_speed_aware_torque_profile, parse_speed_aware_torque_profile, SpeedAwareTorqueRuntime,
  SPEED_BUCKET_BP, LOW_SPEED_BUCKET_BP,
)
from openpilot.sunnypilot.custom.lateral.torque_safety import (
  validate_friction_breakaway_mode,
  validate_live_torque_speed_adaptive_mode,
  validate_roll_comp_gain_mode,
  validate_torque_override_friction,
  validate_torque_override_lat_accel_factor,
)

RELAXED_MIN_BUCKET_POINTS = np.array([1, 200, 300, 500, 500, 300, 200, 1])
ROLL_COMP_MIN_V_EGO = 15.0  # Matches torqued.MIN_VEL without creating an import cycle.
# Roll-comp collection floor for the speed-resolved bands: below 5 m/s the controller
# freezes the integrator and the measurement smoother resets, so points there are junk.
ROLL_COMP_LEARN_MIN_V_EGO = 5.0

ALLOWED_CARS = ['toyota', 'hyundai', 'rivian', 'honda']

# Rack breakaway observer thresholds — mirror the offline analysis of route
# 000002a1 (analyze_stiction / stiction_lab): a dwell of >=0.3s followed by a
# >=1.5 deg/s jump is a stick-slip breakout; the EPS torque held during the
# dwell approximates the static breakaway for that direction.
BREAKAWAY_DWELL_S = 0.3
BREAKAWAY_STILL_DEG_S = 0.5
BREAKAWAY_JUMP_DEG_S = 1.5
BREAKAWAY_MIN_V_EGO = 8.0
BREAKAWAY_WINDOW = 25  # rolling breakout samples per direction


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
    self.speed_adaptive_mode = 'off'
    self.speed_adaptive_runtime = SpeedAwareTorqueRuntime()
    self.speed_learning_buckets = SpeedAwareTorqueBuckets(
      x_bounds=[(-0.5, -0.3), (-0.3, -0.2), (-0.2, -0.1), (-0.1, 0), (0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.5)],
      speed_bp=SPEED_BUCKET_BP, min_points=1, min_points_total=1, points_per_bucket=1500, rowsize=3)
    self.low_speed_shadow = False
    self.low_speed_buckets = SpeedAwareTorqueBuckets(
      x_bounds=[(-0.5, -0.3), (-0.3, -0.2), (-0.2, -0.1), (-0.1, 0), (0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.5), (-1.0, -0.5), (0.5, 1.0)],
      speed_bp=LOW_SPEED_BUCKET_BP, min_points=1, min_points_total=1, points_per_bucket=1500, rowsize=3)
    self.speed_profile_cache = None
    self._speed_profile_blend_base = None
    self._last_speed_profile_write = -1

    # Phase 3 shadow-only roll-compensation gain learner. No steering changes
    # in this phase; data is collected and persisted for later apply wiring.
    self.roll_comp_mode = 'off'
    self.roll_comp_buckets = RollCompBuckets()
    self.roll_comp_profile_cache = None
    self._roll_comp_blend_base = None
    self.update_roll_comp_telemetry()  # seeds the full default dict incl. band fields
    self._profile_blend_base_frozen = False

    # Shadow-only rack breakaway observer. Never affects control.
    self.friction_breakaway_mode = 'off'
    self._ba_last_t = None
    self._ba_dwell = 0.0
    self._ba_eps_hold = 0.0
    self._ba_samples = {1: [], -1: []}
    self.breakaway_events = 0

  def initialize_custom_params(self, decimated=False):
    self.update_use_params()

    if self.enforce_torque_control_toggle:
      if self._params.get_bool("LiveTorqueParamsRelaxedToggle"):
        self.min_bucket_points = RELAXED_MIN_BUCKET_POINTS / (10 if decimated else 1)
        self.factor_sanity = 0.5 if decimated else 1.0
        self.friction_sanity = 0.8 if decimated else 1.0

      if self._params.get_bool("CustomTorqueParams"):
        lat_accel_factor = validate_torque_override_lat_accel_factor(self._params.get("TorqueParamsOverrideLatAccelFactor", return_default=True))
        friction = validate_torque_override_friction(self._params.get("TorqueParamsOverrideFriction", return_default=True))
        if lat_accel_factor is not None and friction is not None:
          self.offline_latAccelFactor = lat_accel_factor
          self.offline_friction = friction
        else:
          self.custom_torque_params = False

  def _update_params(self):
    did_update = self.frame % int(PARAMS_UPDATE_PERIOD / DT_MDL) == 0
    if did_update:
      self.use_live_torque_params = self._params.get_bool("LiveTorqueParamsToggle")
      self.custom_torque_params = self._params.get_bool("CustomTorqueParams")
      self.torque_override_enabled = self._params.get_bool("TorqueParamsOverrideEnabled")
      self.speed_adaptive_mode = validate_live_torque_speed_adaptive_mode(self._params.get("LiveTorqueSpeedAdaptiveMode", return_default=True))
      self.low_speed_shadow = self._params.get_bool("LiveTorqueLowSpeedShadow")
      payload = self._params.get("LiveTorqueSpeedAdaptiveParams", return_default=True)
      if payload:
        try:
          parsed = json.loads(payload)
          self.speed_profile_cache = parse_speed_aware_torque_profile(self.CP, parsed)
          self.speed_adaptive_runtime.profile = self.speed_profile_cache
          if not self._profile_blend_base_frozen and self.speed_profile_cache is not None:
            self._speed_profile_blend_base = self.speed_profile_cache
        except Exception:
          self.speed_profile_cache = None
          self.speed_adaptive_runtime.profile = None

      self.friction_breakaway_mode = validate_friction_breakaway_mode(self._params.get("LatFrictionBreakawayMode", return_default=True))
      self.roll_comp_mode = validate_roll_comp_gain_mode(self._params.get("RollCompGainMode", return_default=True))
      roll_payload = self._params.get("RollCompGainParams", return_default=True)
      if roll_payload:
        try:
          parsed = json.loads(roll_payload)
          self.roll_comp_profile_cache = parse_roll_comp_profile(self.CP, parsed)
          if not self._profile_blend_base_frozen and self.roll_comp_profile_cache is not None:
            self._roll_comp_blend_base = self.roll_comp_profile_cache
        except Exception:
          self.roll_comp_profile_cache = None
      else:
        self.roll_comp_profile_cache = None
      self.update_roll_comp_telemetry()
    return did_update

  def update_use_params(self):
    did_update = self._update_params()

    if self.enforce_torque_control_toggle:
      if self.custom_torque_params and self.torque_override_enabled:
        self.use_params = False
      else:
        self.use_params = self.use_live_torque_params

    if did_update:
      self._profile_blend_base_frozen = True
    self.frame += 1

  def add_torque_learning_point(self, steer, lateral_acc, v_ego):
    if self.speed_adaptive_mode in ('shadow', 'apply'):
      self.speed_learning_buckets.add_point(steer, lateral_acc, v_ego)

  def collect_shadow_learning_points(self, steer, lateral_acc, v_ego, roll, yaw_rate, steering_rate_deg):
    """Shadow-only learning hooks. No steering changes in these phases.

    - Phase 6 low-speed shadow buckets: collect city/low-speed cornering evidence
      when the user has enabled low-speed shadow collection.
    - Phase 3 roll-compensation gain learning: gated on base torque learner validity
      and straight-road steady-state conditions so the learned gain maps
      roll-induced lateral acceleration to the torque controller's predicted
      lateral acceleration response.
    """
    # Phase 6: collection-only low-speed shadow buckets.
    if self.speed_adaptive_mode in ('shadow', 'apply') and self.low_speed_shadow:
      if v_ego < ROLL_COMP_MIN_V_EGO and abs(steer) > 0.02 and abs(lateral_acc) <= 3.0:
        self.low_speed_buckets.add_point(steer, lateral_acc, v_ego)

    if self.roll_comp_mode not in ('shadow', 'apply'):
      return
    if not self.filtered_points.is_valid():
      return
    if v_ego <= ROLL_COMP_LEARN_MIN_V_EGO:
      return
    if abs(v_ego * yaw_rate) >= 0.15:
      return
    if steering_rate_deg is None or abs(steering_rate_deg) >= 3.0:
      return
    if abs(roll) > 0.1:
      return
    torque_lat_accel = (self.filtered_params['latAccelFactor'].x * steer +
                        self.filtered_params['latAccelOffset'].x)
    self.roll_comp_buckets.add_point(roll, torque_lat_accel, v_ego)

  def update_breakaway_observer(self, t, steering_rate_deg, eps_norm, v_ego):
    """Shadow-only dwell->jump breakout detector, fed per carState frame.

    ``eps_norm`` is EPS torque normalized by the brand STEER_MAX (sign
    convention irrelevant here — magnitudes are recorded per motion direction).
    """
    if self.friction_breakaway_mode == 'off' or eps_norm is None:
      self._ba_last_t = None
      self._ba_dwell = 0.0
      return
    dt = 0.0 if self._ba_last_t is None else min(max(t - self._ba_last_t, 0.0), 0.1)
    self._ba_last_t = t

    lat_active = bool(self.raw_points['lat_active'][-1]) if len(self.raw_points.get('lat_active', [])) else False
    pressed = bool(self.raw_points['steer_override'][-1]) if len(self.raw_points.get('steer_override', [])) else True
    if not lat_active or pressed or v_ego < BREAKAWAY_MIN_V_EGO or not np.isfinite(eps_norm):
      self._ba_dwell = 0.0
      return

    rate = abs(steering_rate_deg)
    if rate < BREAKAWAY_STILL_DEG_S:
      self._ba_dwell += dt
      self._ba_eps_hold = eps_norm
    else:
      if rate >= BREAKAWAY_JUMP_DEG_S and self._ba_dwell >= BREAKAWAY_DWELL_S:
        direction = 1 if steering_rate_deg > 0 else -1
        samples = self._ba_samples[direction]
        samples.append(abs(self._ba_eps_hold))
        del samples[:-BREAKAWAY_WINDOW]
        self.breakaway_events += 1
      self._ba_dwell = 0.0

  def breakaway_telemetry(self):
    left, right = self._ba_samples[1], self._ba_samples[-1]
    return {
      'left': float(np.median(left)) if left else 0.0,
      'right': float(np.median(right)) if right else 0.0,
      'events': int(self.breakaway_events),
    }

  def update_roll_comp_telemetry(self):
    cache = self.roll_comp_profile_cache
    if self.roll_comp_mode in ('shadow', 'apply') and cache is not None:
      band_map = {(b['vLo'], b['vHi']): b for b in cache.get('bands', [])}
      # scalar fields keep their legacy meaning: the primary (>=15 m/s) band's fit
      self.roll_comp_profile = {
        'gain': float(cache.get('gain', 0.0)),
        'points': int(cache.get('points', 0)),
        'span': float(cache.get('span', 0.0)),
        'valid': 'gain' in cache,
        'bandGains': [float(band_map[b]['gain']) if b in band_map else 0.0 for b in ROLL_COMP_SPEED_BANDS],
        'bandPoints': [int(band_map[b]['points']) if b in band_map else 0 for b in ROLL_COMP_SPEED_BANDS],
      }
    else:
      self.roll_comp_profile = {'gain': 0.0, 'points': 0, 'span': 0.0, 'valid': False,
                                'bandGains': [0.0] * len(ROLL_COMP_SPEED_BANDS),
                                'bandPoints': [0] * len(ROLL_COMP_SPEED_BANDS)}

  def maybe_persist_speed_profile(self, cache_write=False):
    if not cache_write:
      return
    if self.speed_adaptive_mode in ('shadow', 'apply'):
      low_speed_buckets = self.low_speed_buckets if self.low_speed_shadow else None
      profile = fit_speed_aware_torque_profile(self.CP, self.speed_learning_buckets, low_speed_buckets=low_speed_buckets)
      if profile is not None:
        if self._speed_profile_blend_base is not None:
          profile = blend_speed_aware_torque_profile(self._speed_profile_blend_base, profile)
        self.speed_profile_cache = profile
        self.speed_adaptive_runtime.profile = profile
        self._params.put("LiveTorqueSpeedAdaptiveParams", format_speed_aware_torque_profile(profile), block=True)

    if self.roll_comp_mode in ('shadow', 'apply'):
      profile = fit_roll_comp_profile(self.CP, self.roll_comp_buckets)
      if profile is not None:
        if self._roll_comp_blend_base is not None:
          profile = blend_roll_comp_profile(self._roll_comp_blend_base, profile)
        self.roll_comp_profile_cache = profile
        self.update_roll_comp_telemetry()
        self._params.put("RollCompGainParams", format_roll_comp_profile(profile), block=True)
