import ast
import math
from dataclasses import dataclass
from enum import IntEnum, IntFlag

import numpy as np

from cereal import log
from opendbc.car.lateral import get_friction
from openpilot.common.params import Params, UnknownKeyName
from openpilot.selfdrive.controls.lib.lateral_demand import DEMAND_SOURCE_MODEL_PATH, ProcessedLateralDemand
from openpilot.selfdrive.controls.lib.lateral_accel import roll_lateral_accel
from openpilot.selfdrive.controls.lib.lateral_demand_profile import LateralDemandProfile, lateral_mode_to_uint8
from openpilot.selfdrive.controls.lib.lateral_oscillation_classifier import (
  LateralOscillationClassifier,
  WobbleResponse,
  compute_wobble_response,
  is_wobble_active,
  lateral_oscillation_to_uint8,
)
from openpilot.selfdrive.controls.lib.lateral_turn_exit_controller import (
  LateralTurnExitController,
  TurnExitDecision,
  TurnExitMode,
)
from openpilot.selfdrive.controls.lib.lateral_vehicle_health_estimator import (
  HEALTH_EST_BIAS_MAX,
  HEALTH_EST_BIAS_WARNING,
  LateralVehicleHealthEstimate,
  LateralVehicleHealthEstimator,
)
from openpilot.selfdrive.controls.lib.latcontrol import LatControl
from openpilot.sunnypilot.selfdrive.controls.lib.steering_actuator_feedback import classify_steering_limit_context
from openpilot.sunnypilot.selfdrive.locationd.speed_aware_torque import (
  SPEED_BUCKET_LABELS,
  parse_speed_aware_params,
)


VERSION = 4
FRICTION_THRESHOLD = 0.3

RESPONSE_SCALE_MIN = 0.80
RESPONSE_SCALE_MAX = 1.20
TRIM_LAT_ACCEL_MIN = -0.16
TRIM_LAT_ACCEL_MAX = 0.16
RESPONSE_DELAY_MIN = 0.10
RESPONSE_DELAY_MAX = 0.35

LEAD_GAIN_BP = [0.0, 3.0, 10.0, 20.0, 30.0, 40.0]
LEAD_GAIN_V = [0.20, 0.25, 0.45, 0.55, 0.45, 0.35]
LEAD_DELTA_CAP_BP = [0.0, 3.0, 10.0, 20.0, 30.0, 40.0]
LEAD_DELTA_CAP_V = [0.08, 0.10, 0.35, 0.55, 0.45, 0.35]
FEEDBACK_GAIN_BP = [0.0, 3.0, 10.0, 20.0, 30.0, 40.0]
FEEDBACK_GAIN_V = [0.12, 0.15, 0.22, 0.18, 0.12, 0.10]
DAMPING_GAIN_BP = [0.0, 3.0, 10.0, 20.0, 30.0, 40.0]
DAMPING_GAIN_V = [0.04, 0.05, 0.075, 0.060, 0.040, 0.035]
BREAKAWAY_SCALE_BP = [0.0, 3.0, 10.0, 20.0, 30.0, 40.0]
BREAKAWAY_SCALE_V = [0.30, 0.42, 0.68, 0.64, 0.50, 0.45]
BREAKAWAY_FULL_DEMAND = 0.40
MEASUREMENT_RATE_CAP = 20.0
MEASUREMENT_RATE_FILTER_ALPHA = 0.25

OUTPUT_SLEW_RATE_BP = [0.0, 3.0, 10.0, 20.0, 30.0, 40.0]
OUTPUT_SLEW_RATE_V = [0.80, 1.10, 2.40, 3.60, 4.00, 4.00]
SIGN_CHANGE_SLEW_RATE_BP = [0.0, 3.0, 10.0, 20.0, 30.0, 40.0]
SIGN_CHANGE_SLEW_RATE_V = [0.40, 0.60, 1.40, 2.00, 2.20, 2.00]
SAME_DIRECTION_LIMIT_CAP = 0.72
SAME_DIRECTION_LIMIT_RATE = 1.20
HIGH_RATE_START_DEG = 70.0
HIGH_RATE_FULL_DEG = 100.0
HIGH_RATE_MIN_CAP = 0.60
HIGH_RATE_SLEW_SCALE = 0.65
STALE_ACTUATOR_ERROR_THRESHOLD = 0.15
STALE_ACTUATOR_CAP = 0.35
LOW_SPEED_UNDER_RESPONSE_MARGIN = 0.12
LOW_SPEED_UNDER_RESPONSE_FADE_SPEED = 12.0
LOW_SPEED_UNDER_RESPONSE_CAP = 0.90
LOW_SPEED_UNDER_RESPONSE_MAX_ACTUAL_LAT_ACCEL = 2.6
LOW_SPEED_UNDER_RESPONSE_SIGN_THRESHOLD = 0.05
LOW_SPEED_UNDER_RESPONSE_MIN_PATH_QUALITY = 0.64
UNDER_RESPONSE_RECOVERY_FULL_DEFICIT = 0.30
UNDER_RESPONSE_RECOVERY_CAP_BP = [0.0, 9.0, 15.0, 25.0, 40.0]
UNDER_RESPONSE_RECOVERY_CAP_V = [0.90, 0.90, 0.88, 0.84, 0.80]
UNDER_RESPONSE_RECOVERY_RATE_SCALE_BP = [0.0, 15.0, 25.0, 40.0]
UNDER_RESPONSE_RECOVERY_RATE_SCALE_V = [1.0, 1.0, 0.85, 0.75]
UNDER_RESPONSE_LEAD_GAIN_BOOST_BP = [0.0, 10.0, 20.0, 40.0]
UNDER_RESPONSE_LEAD_GAIN_BOOST_V = [0.18, 0.15, 0.10, 0.06]
UNDER_RESPONSE_LEAD_CAP_BOOST_BP = [0.0, 10.0, 20.0, 40.0]
UNDER_RESPONSE_LEAD_CAP_BOOST_V = [0.15, 0.12, 0.08, 0.05]
V41_UNDER_RESPONSE_CATCHUP_GAIN_BP = [0.0, 10.0, 20.0, 40.0]
V41_UNDER_RESPONSE_CATCHUP_GAIN_V = [0.35, 0.30, 0.22, 0.14]
V41_UNDER_RESPONSE_CATCHUP_CAP_BP = [0.0, 10.0, 20.0, 40.0]
V41_UNDER_RESPONSE_CATCHUP_CAP_V = [0.16, 0.15, 0.12, 0.08]

LEARN_MIN_SPEED = 10.0
LEARN_MAX_SPEED = 35.0
LEARN_MIN_TARGET = 0.08
LEARN_SIGN_THRESHOLD = 0.05
LEARN_MAX_JERK = 8.0
LEARN_MAX_GOVERNOR_REASON = 0
LEARN_MIN_PATH_QUALITY = 0.75
LEARN_PATH_REASON_OK = "ok"
UNDER_RESPONSE_LOW_SPEED_ALLOWED_PATH_REASONS = frozenset((LEARN_PATH_REASON_OK, "low_lane_confidence"))

SPEED_BUCKET_CENTERS = [5.0, 15.0, 25.0, 35.0, 45.0]

# Recenter mode: faster torque unwind and relaxed sign-change when target collapses toward zero.
# Broader recenter: target can be larger and persistence is shorter so the
# turn-exit feel matches v2.1's "on rails" unwind without globally
# increasing aggression or causing straight-road hunting. The early
# release guard in _build_target fires even before recenter mode
# activates, so turn-exit feels immediate regardless of these bounds.
RECENTER_MIN_SPEED = 10.0  # m/s — only activate above this speed
RECENTER_TARGET_COLLAPSE_RATE = 0.3  # m/s² per frame — target must be decreasing faster than this
RECENTER_MAX_ABS_TARGET = 0.85  # m/s² — target must be small enough to be "near center" (loosened from 0.5)
RECENTER_MIN_PATH_QUALITY = 0.5  # — path quality must be reasonable
RECENTER_PERSISTENCE_FRAMES = 3  # — must be recentering for this many consecutive frames (loosened from 5)
RECENTER_LEAD_REDUCTION = 0.6  # — reduce response-delay lead by this fraction when recentering
RECENTER_SLEW_BOOST = 1.5  # — multiply sign-change slew rate by this when recentering
# Recenter lead_reduction has a non-zero floor so the very first frame
# the recenter activates already trims some response-delay lead, instead
# of waiting a full persistence window. This makes turn-exit feel
# immediate even on a single "collapse" frame.
RECENTER_LEAD_REDUCTION_FLOOR = 0.3
# Recenter slew boost floor for same-direction unwind. Smaller than
# RECENTER_SLEW_BOOST because same-direction unwind doesn't need the
# full sign-change boost; we just want a bit more headroom.
RECENTER_SAME_DIRECTION_SLEW_BOOST = 1.2


class TorqueV4LearnerRejectReason(IntFlag):
  NONE = 0
  INACTIVE = 1 << 0
  STEERING_PRESSED = 1 << 1
  STEER_LIMITED = 1 << 2
  CURVATURE_LIMITED = 1 << 3
  SATURATED = 1 << 4
  LOW_DEMAND = 1 << 5
  NON_FINITE = 1 << 6
  HIGH_JERK = 1 << 7
  SIGN_CONFLICT = 1 << 8
  SPEED_RANGE = 1 << 9
  GOVERNOR_ACTIVE = 1 << 10
  NON_MODEL_DEMAND = 1 << 11
  LOW_PATH_QUALITY = 1 << 12
  PATH_REASON = 1 << 13
  LANE_CHANGE_SHAPING = 1 << 14
  LANE_CENTERING_ASSIST = 1 << 15


class TorqueV4GovernorReason(IntFlag):
  NONE = 0
  CLIPPED = 1 << 0
  SLEW_LIMITED = 1 << 1
  SIGN_CHANGE_LIMITED = 1 << 2
  DRIVER_OVERRIDE = 1 << 3
  SAME_DIRECTION_LIMIT = 1 << 4
  HIGH_STEERING_RATE = 1 << 5
  INVALID = 1 << 6
  STALE_ACTUATOR_MISMATCH = 1 << 7
  LOW_SPEED_UNDER_RESPONSE_RECOVERY = 1 << 8
  RECENTER_MODE = 1 << 9


class TorqueV4Phase(IntEnum):
  idle = 0
  engage = 1
  hold = 2
  release = 3


PHASE_TO_CAPNP = {
  TorqueV4Phase.idle: log.ControlsState.LateralTorqueState.AdaptiveTorqueState.Phase.idle,
  TorqueV4Phase.engage: log.ControlsState.LateralTorqueState.AdaptiveTorqueState.Phase.engage,
  TorqueV4Phase.hold: log.ControlsState.LateralTorqueState.AdaptiveTorqueState.Phase.hold,
  TorqueV4Phase.release: log.ControlsState.LateralTorqueState.AdaptiveTorqueState.Phase.release,
}


def _finite(*values: float) -> bool:
  try:
    return all(math.isfinite(float(value)) for value in values)
  except (TypeError, ValueError):
    return False


def _finite_float(value) -> float | None:
  try:
    result = float(value)
  except (TypeError, ValueError):
    return None
  return result if math.isfinite(result) else None


def _clip(value: float, lower: float, upper: float) -> float:
  return float(np.clip(value, lower, upper))


def _interp(value: float, bp: list[float], vals: list[float]) -> float:
  return float(np.interp(float(value), bp, vals))


def _sign(value: float, threshold: float = 0.0) -> int:
  return 1 if value > threshold else (-1 if value < -threshold else 0)


def turn_exit_mode_to_uint8(mode: str) -> int:
  return TURN_EXIT_MODE_TO_UINT8.get(mode, 0)


TURN_EXIT_MODE_TO_UINT8 = {
  TurnExitMode.INACTIVE.value: 0,
  TurnExitMode.TURN_IN.value: 1,
  TurnExitMode.STEADY_CURVE.value: 2,
  TurnExitMode.TURN_EXIT.value: 3,
  TurnExitMode.EARLY_RELEASE.value: 4,
}


def _approach(value: float, target: float, step: float) -> float:
  if target > value:
    return min(target, value + step)
  return max(target, value - step)


def _under_response_strength(target_lateral_accel: float, actual_lateral_accel: float) -> float:
  if not _finite(target_lateral_accel, actual_lateral_accel):
    return 0.0
  target_sign = _sign(target_lateral_accel, LOW_SPEED_UNDER_RESPONSE_SIGN_THRESHOLD)
  actual_sign = _sign(actual_lateral_accel, LOW_SPEED_UNDER_RESPONSE_SIGN_THRESHOLD)
  if target_sign == 0 or actual_sign not in (0, target_sign):
    return 0.0
  if abs(actual_lateral_accel) > LOW_SPEED_UNDER_RESPONSE_MAX_ACTUAL_LAT_ACCEL:
    return 0.0

  under_response = target_sign * (target_lateral_accel - actual_lateral_accel)
  if under_response <= LOW_SPEED_UNDER_RESPONSE_MARGIN:
    return 0.0
  span = UNDER_RESPONSE_RECOVERY_FULL_DEFICIT - LOW_SPEED_UNDER_RESPONSE_MARGIN
  return _clip((under_response - LOW_SPEED_UNDER_RESPONSE_MARGIN) / max(span, 1e-3), 0.0, 1.0)


def finite_difference_curvature_rate_from_steering_rate(VM, steering_angle_rad: float, steering_rate_rad_s: float,
                                                        v_ego: float, roll: float) -> float:
  if not _finite(steering_angle_rad, steering_rate_rad_s, v_ego, roll):
    return 0.0
  eps = math.radians(0.1)
  try:
    k_plus = VM.calc_curvature(float(steering_angle_rad) + eps, float(v_ego), float(roll))
    k_minus = VM.calc_curvature(float(steering_angle_rad) - eps, float(v_ego), float(roll))
  except (ArithmeticError, ValueError, TypeError):
    return 0.0
  if not _finite(k_plus, k_minus):
    return 0.0
  dk_dangle = (k_plus - k_minus) / (2.0 * eps)
  actual_lateral_jerk = -float(v_ego) ** 2 * dk_dangle * float(steering_rate_rad_s)
  return actual_lateral_jerk if math.isfinite(actual_lateral_jerk) else 0.0


@dataclass(frozen=True)
class TorqueV4Target:
  raw_lateral_accel: float
  target_rate: float
  delay_lead_lateral_accel: float
  lead_delta: float
  lead_gain: float
  lead_delta_cap: float


@dataclass(frozen=True)
class TorqueV5Target(TorqueV4Target):
  """v5.0 target with profile-aware shaping fields.

  All shaping fields are computed/applied metadata. They are
  populated by the v5 _build_target() wrapper in later commits;
  this commit only adds the dataclass so callers can pass the
  richer object around without behavior change.
  """
  base_lead_delta: float = 0.0
  preview_boost_computed: float = 0.0
  preview_boost_applied: float = 0.0
  preview_reason: str = ""
  turn_exit_lead_gain_multiplier: float = 1.0
  turn_exit_lead_delta_cap_multiplier: float = 1.0
  turn_exit_early_release: bool = False
  v5_active: bool = False
  v5_reason: str = ""


@dataclass(frozen=True)
class TorqueV4Observation:
  active: bool
  v_ego: float
  steering_pressed: bool
  steer_limited_by_safety: bool
  curvature_limited: bool
  saturated: bool
  raw_target_lateral_accel: float
  delay_lead_lateral_accel: float
  target_lateral_accel_rate: float
  actual_lateral_accel: float
  actual_lateral_jerk: float
  measurement_rate: float
  finite: bool
  demand_source: str = DEMAND_SOURCE_MODEL_PATH
  path_quality: float = 1.0
  path_reason: str = LEARN_PATH_REASON_OK
  lane_change_shaping_active: bool = False
  lane_change_blend: float = 0.0
  lane_centering_assist_active: bool = False


@dataclass(frozen=True)
class TorqueV4SpeedModelResult:
  response_scale: float
  trim_lateral_accel: float
  response_delay: float
  lead_gain: float
  lead_delta_cap: float
  feedback_gain: float
  damping_gain: float
  breakaway_scale: float
  output_slew_rate: float
  sign_change_slew_rate: float
  speed_aware_confidence: float
  speed_aware_factor: float
  effective_lat_accel_factor: float
  effective_lat_accel_offset: float


@dataclass(frozen=True)
class TorqueV4GovernorProfile:
  output_slew_rate_bp: list[float]
  output_slew_rate_v: list[float]
  sign_change_slew_rate_bp: list[float]
  sign_change_slew_rate_v: list[float]
  same_direction_limit_cap: float
  same_direction_limit_rate: float
  high_rate_start_deg: float
  high_rate_full_deg: float
  high_rate_min_cap: float
  high_rate_slew_scale: float
  same_direction_limit_rate_bp: list[float] | None = None
  same_direction_limit_rate_v: list[float] | None = None
  same_direction_decrease_bypass: bool = False


@dataclass(frozen=True)
class TorqueV4AdaptationUpdate:
  sample_accepted: bool
  reject_reason: TorqueV4LearnerRejectReason
  residual_error: float


@dataclass(frozen=True)
class TorqueV4GovernorResult:
  output_torque: float
  reason: TorqueV4GovernorReason
  output_cap: float


@dataclass(frozen=True)
class TorqueV5GovernorContext:
  """Per-frame v5 context passed to the governor.

  Lets the governor distinguish a clean profile-driven frame
  from a degraded one (low path quality, lane change, wobble,
  turn-exit). The initial 5.0 implementation uses this for
  telemetry and a bounded turn-exit unwind slew boost. Safety
  caps, output caps, and sign-change slew rates stay at v4.1
  values regardless of context.
  """
  profile_available: bool
  demand_mode: str
  demand_mode_confidence: float
  preview_active: bool
  turn_exit_active: bool
  wobble_active: bool
  v5_active: bool
  v5_reason: str


@dataclass(frozen=True)
class TorqueV4RecenterMode:
  active: bool = False
  persistence_frames: int = 0
  lead_reduction: float = 0.0
  slew_boost: float = 1.0


class TorqueV4SpeedModel:
  """Speed-aware corrections stay opt-in and bounded; never apply raw bucket params directly."""

  def __init__(self, governor_profile: TorqueV4GovernorProfile | None = None):
    self.governor_profile = governor_profile or LatControlTorqueV4.GOVERNOR_PROFILE

  def update(self, v_ego: float, torque_params, speed_aware_params: dict | None, speed_aware_apply_enabled: bool,
             adaptation: "TorqueV4SessionAdaptation") -> TorqueV4SpeedModelResult:
    response_delay = _clip(adaptation.response_delay, RESPONSE_DELAY_MIN, RESPONSE_DELAY_MAX)
    lead_gain = _interp(v_ego, LEAD_GAIN_BP, LEAD_GAIN_V)
    lead_delta_cap = _interp(v_ego, LEAD_DELTA_CAP_BP, LEAD_DELTA_CAP_V)
    feedback_gain = _interp(v_ego, FEEDBACK_GAIN_BP, FEEDBACK_GAIN_V)
    damping_gain = _interp(v_ego, DAMPING_GAIN_BP, DAMPING_GAIN_V)
    breakaway_scale = _interp(v_ego, BREAKAWAY_SCALE_BP, BREAKAWAY_SCALE_V)
    output_slew_rate = _interp(v_ego, self.governor_profile.output_slew_rate_bp, self.governor_profile.output_slew_rate_v)
    sign_change_slew_rate = _interp(v_ego, self.governor_profile.sign_change_slew_rate_bp,
                                    self.governor_profile.sign_change_slew_rate_v)

    speed_factor, speed_offset, confidence = self._speed_aware_values(v_ego, torque_params, speed_aware_params, speed_aware_apply_enabled)
    global_factor = max(float(torque_params.latAccelFactor), 1e-3)
    global_offset = float(torque_params.latAccelOffset)
    speed_response_scale = _clip(global_factor / max(speed_factor, 1e-3), RESPONSE_SCALE_MIN, RESPONSE_SCALE_MAX)
    response_scale = _clip(adaptation.response_scale * speed_response_scale, RESPONSE_SCALE_MIN, RESPONSE_SCALE_MAX)
    trim_lateral_accel = _clip(adaptation.trim_lateral_accel + (global_offset - speed_offset), TRIM_LAT_ACCEL_MIN, TRIM_LAT_ACCEL_MAX)
    return TorqueV4SpeedModelResult(
      response_scale=response_scale,
      trim_lateral_accel=trim_lateral_accel,
      response_delay=response_delay,
      lead_gain=lead_gain,
      lead_delta_cap=lead_delta_cap,
      feedback_gain=feedback_gain,
      damping_gain=damping_gain,
      breakaway_scale=breakaway_scale,
      output_slew_rate=output_slew_rate,
      sign_change_slew_rate=sign_change_slew_rate,
      speed_aware_confidence=confidence,
      speed_aware_factor=speed_factor,
      effective_lat_accel_factor=speed_factor,
      effective_lat_accel_offset=speed_offset,
    )

  def _speed_aware_values(self, v_ego: float, torque_params, speed_aware_params: dict | None,
                          speed_aware_apply_enabled: bool) -> tuple[float, float, float]:
    global_factor = max(float(torque_params.latAccelFactor), 1e-3)
    global_offset = float(torque_params.latAccelOffset)
    if not speed_aware_apply_enabled or v_ego < 15.0 or not speed_aware_params:
      return global_factor, global_offset, 0.0

    label = self._label_for_speed(v_ego)
    bucket = speed_aware_params.get(label)
    if not self._valid_bucket(bucket, global_factor):
      return global_factor, global_offset, 0.0

    local_factor = self._interpolate_factor(v_ego, speed_aware_params, global_factor)
    local_offset = self._interpolate_offset(v_ego, speed_aware_params, global_factor, global_offset)
    if not _finite(local_factor, local_offset):
      return global_factor, global_offset, 0.0

    base_confidence = 0.30 if v_ego < 30.0 else 0.22
    speed_gate = _clip((v_ego - 15.0) / 5.0, 0.0, 1.0)
    confidence = base_confidence * speed_gate
    if confidence <= 0.0:
      return global_factor, global_offset, 0.0
    effective_factor = confidence * local_factor + (1.0 - confidence) * global_factor
    effective_offset = confidence * local_offset + (1.0 - confidence) * global_offset
    effective_factor = _clip(effective_factor, 0.5 * global_factor, 2.0 * global_factor)
    effective_offset = _clip(effective_offset, global_offset + TRIM_LAT_ACCEL_MIN, global_offset + TRIM_LAT_ACCEL_MAX)
    return effective_factor, effective_offset, confidence

  @staticmethod
  def _label_for_speed(v_ego: float) -> str:
    if v_ego < 10.0:
      return "0_10"
    if v_ego < 20.0:
      return "10_20"
    if v_ego < 30.0:
      return "20_30"
    if v_ego < 40.0:
      return "30_40"
    return "40_plus"

  @staticmethod
  def _valid_bucket(bucket, global_factor: float) -> bool:
    if bucket is None or not isinstance(bucket, (tuple, list)) or len(bucket) < 2:
      return False
    try:
      factor = float(bucket[0])
      offset = float(bucket[1])
    except (TypeError, ValueError):
      return False
    return _finite(factor, offset) and 0.5 * global_factor <= factor <= 2.0 * global_factor

  def _interpolate_factor(self, v_ego: float, speed_aware_params: dict, global_factor: float) -> float:
    valid_points = []
    for center, label in zip(SPEED_BUCKET_CENTERS, SPEED_BUCKET_LABELS):
      bucket = speed_aware_params.get(label)
      if self._valid_bucket(bucket, global_factor):
        valid_points.append((center, float(bucket[0])))
    if not valid_points:
      return global_factor
    if len(valid_points) == 1:
      return valid_points[0][1]
    return float(np.interp(v_ego, [point[0] for point in valid_points], [point[1] for point in valid_points]))

  def _interpolate_offset(self, v_ego: float, speed_aware_params: dict, global_factor: float, global_offset: float) -> float:
    valid_points = []
    for center, label in zip(SPEED_BUCKET_CENTERS, SPEED_BUCKET_LABELS):
      bucket = speed_aware_params.get(label)
      if self._valid_bucket(bucket, global_factor):
        valid_points.append((center, float(bucket[1])))
    if not valid_points:
      return global_offset
    if len(valid_points) == 1:
      return valid_points[0][1]
    return float(np.interp(v_ego, [point[0] for point in valid_points], [point[1] for point in valid_points]))


class TorqueV4SessionAdaptation:
  """Session-only learner scaffold. It never writes Params or persists state."""

  def __init__(self, response_delay: float):
    self.base_response_delay = _clip(response_delay, RESPONSE_DELAY_MIN, RESPONSE_DELAY_MAX)
    self.response_scale = 1.0
    self.trim_lateral_accel = 0.0
    self.response_delay = self.base_response_delay

  def reset(self) -> None:
    self.response_scale = 1.0
    self.trim_lateral_accel = 0.0
    self.response_delay = self.base_response_delay

  def update_lateral_lag(self, lag: float) -> None:
    if _finite(lag):
      self.base_response_delay = _clip(lag, RESPONSE_DELAY_MIN, RESPONSE_DELAY_MAX)
      self.response_delay = self.base_response_delay

  def update(self, observation: TorqueV4Observation, governor_reason: TorqueV4GovernorReason) -> TorqueV4AdaptationUpdate:
    reject_reason = self._reject_reason(observation, governor_reason)
    response_residual = observation.delay_lead_lateral_accel - observation.actual_lateral_accel if observation.finite else 0.0
    trim_residual = observation.raw_target_lateral_accel - observation.actual_lateral_accel if observation.finite else 0.0
    if reject_reason != TorqueV4LearnerRejectReason.NONE:
      return TorqueV4AdaptationUpdate(False, reject_reason, response_residual)

    if abs(observation.actual_lateral_accel) > LEARN_SIGN_THRESHOLD:
      target_scale = _clip(abs(observation.delay_lead_lateral_accel) / max(abs(observation.actual_lateral_accel), 1e-3),
                           RESPONSE_SCALE_MIN, RESPONSE_SCALE_MAX)
      self.response_scale = _clip(self.response_scale + 0.002 * (target_scale - self.response_scale),
                                  RESPONSE_SCALE_MIN, RESPONSE_SCALE_MAX)

    if abs(observation.target_lateral_accel_rate) < 0.12:
      self.trim_lateral_accel = _clip(self.trim_lateral_accel + 0.0005 * trim_residual,
                                      TRIM_LAT_ACCEL_MIN, TRIM_LAT_ACCEL_MAX)

    return TorqueV4AdaptationUpdate(True, TorqueV4LearnerRejectReason.NONE, response_residual)

  @staticmethod
  def _reject_reason(observation: TorqueV4Observation,
                     governor_reason: TorqueV4GovernorReason) -> TorqueV4LearnerRejectReason:
    reason = TorqueV4LearnerRejectReason.NONE
    if not observation.active:
      reason |= TorqueV4LearnerRejectReason.INACTIVE
    if observation.steering_pressed:
      reason |= TorqueV4LearnerRejectReason.STEERING_PRESSED
    if observation.steer_limited_by_safety:
      reason |= TorqueV4LearnerRejectReason.STEER_LIMITED
    if observation.curvature_limited:
      reason |= TorqueV4LearnerRejectReason.CURVATURE_LIMITED
    if observation.saturated:
      reason |= TorqueV4LearnerRejectReason.SATURATED
    if abs(observation.delay_lead_lateral_accel) < LEARN_MIN_TARGET:
      reason |= TorqueV4LearnerRejectReason.LOW_DEMAND
    if not observation.finite:
      reason |= TorqueV4LearnerRejectReason.NON_FINITE
    if abs(observation.actual_lateral_jerk) > LEARN_MAX_JERK:
      reason |= TorqueV4LearnerRejectReason.HIGH_JERK
    target_sign = _sign(observation.delay_lead_lateral_accel, LEARN_SIGN_THRESHOLD)
    actual_sign = _sign(observation.actual_lateral_accel, LEARN_SIGN_THRESHOLD)
    if target_sign != 0 and actual_sign != 0 and target_sign != actual_sign:
      reason |= TorqueV4LearnerRejectReason.SIGN_CONFLICT
    if observation.v_ego < LEARN_MIN_SPEED or observation.v_ego > LEARN_MAX_SPEED:
      reason |= TorqueV4LearnerRejectReason.SPEED_RANGE
    if governor_reason != TorqueV4GovernorReason(LEARN_MAX_GOVERNOR_REASON):
      reason |= TorqueV4LearnerRejectReason.GOVERNOR_ACTIVE
    if observation.demand_source != DEMAND_SOURCE_MODEL_PATH:
      reason |= TorqueV4LearnerRejectReason.NON_MODEL_DEMAND
    if not _finite(observation.path_quality) or observation.path_quality < LEARN_MIN_PATH_QUALITY:
      reason |= TorqueV4LearnerRejectReason.LOW_PATH_QUALITY
    if observation.path_reason != LEARN_PATH_REASON_OK:
      reason |= TorqueV4LearnerRejectReason.PATH_REASON
    lane_change_blend = _finite_float(observation.lane_change_blend)
    if observation.lane_change_shaping_active or lane_change_blend is None or abs(lane_change_blend) > 1e-3:
      reason |= TorqueV4LearnerRejectReason.LANE_CHANGE_SHAPING
    if observation.lane_centering_assist_active:
      reason |= TorqueV4LearnerRejectReason.LANE_CENTERING_ASSIST
    return reason


class TorqueV4OutputGovernor:
  def __init__(self, dt: float, profile: TorqueV4GovernorProfile):
    self.dt = max(float(dt), 1e-3)
    self.profile = profile
    self.previous_output = 0.0

  def reset(self) -> None:
    self.previous_output = 0.0

  def update(self, *, active: bool, v_ego: float, steering_pressed: bool, steering_rate_deg: float,
             same_direction_limit: bool, steer_limit_unwind: bool, actuator_mismatch: bool, actuator_error: float,
             raw_output_torque: float, max_output: float, speed_model: TorqueV4SpeedModelResult,
             recovery_target_lateral_accel: float = 0.0, actual_lateral_accel: float = 0.0,
             under_response_recovery_allowed: bool = False,
             recenter: TorqueV4RecenterMode | None = None,
             v5_context: TorqueV5GovernorContext | None = None) -> TorqueV4GovernorResult:
    reason = TorqueV4GovernorReason.NONE
    if not active:
      self.reset()
      return TorqueV4GovernorResult(0.0, reason, max_output)
    if not _finite(v_ego, steering_rate_deg, actuator_error, raw_output_torque, max_output) or max_output <= 0.0:
      self.reset()
      return TorqueV4GovernorResult(0.0, TorqueV4GovernorReason.INVALID, 0.0)

    if steering_pressed:
      reason |= TorqueV4GovernorReason.DRIVER_OVERRIDE

    output_cap = max_output
    high_rate_blend = _clip((abs(steering_rate_deg) - self.profile.high_rate_start_deg) / max(self.profile.high_rate_full_deg - self.profile.high_rate_start_deg, 1e-3), 0.0, 1.0)
    if high_rate_blend > 0.0:
      output_cap = min(output_cap, max_output * (1.0 + high_rate_blend * (self.profile.high_rate_min_cap - 1.0)))
      reason |= TorqueV4GovernorReason.HIGH_STEERING_RATE
    if same_direction_limit and not steer_limit_unwind:
      output_cap = min(output_cap, max_output * self.profile.same_direction_limit_cap)
      reason |= TorqueV4GovernorReason.SAME_DIRECTION_LIMIT
    stale_actuator_mismatch = actuator_mismatch and same_direction_limit and not steer_limit_unwind and abs(actuator_error) > STALE_ACTUATOR_ERROR_THRESHOLD
    if stale_actuator_mismatch:
      output_cap = min(output_cap, max_output * STALE_ACTUATOR_CAP)
      reason |= TorqueV4GovernorReason.STALE_ACTUATOR_MISMATCH
    under_response_recovery = self._under_response_recovery_strength(
      allowed=under_response_recovery_allowed,
      v_ego=v_ego,
      steering_pressed=steering_pressed,
      recovery_target_lateral_accel=recovery_target_lateral_accel,
      actual_lateral_accel=actual_lateral_accel,
      raw_output_torque=raw_output_torque,
      high_rate_blend=high_rate_blend,
      stale_actuator_mismatch=stale_actuator_mismatch,
    )
    if same_direction_limit and not steer_limit_unwind and under_response_recovery > 0.0:
      recovery_target_cap = _interp(v_ego, UNDER_RESPONSE_RECOVERY_CAP_BP, UNDER_RESPONSE_RECOVERY_CAP_V)
      recovery_cap = self.profile.same_direction_limit_cap + under_response_recovery * (recovery_target_cap - self.profile.same_direction_limit_cap)
      output_cap = max(output_cap, max_output * recovery_cap)
      reason |= TorqueV4GovernorReason.LOW_SPEED_UNDER_RESPONSE_RECOVERY

    clipped = _clip(raw_output_torque, -output_cap, output_cap)
    if abs(clipped - raw_output_torque) > 1e-6:
      reason |= TorqueV4GovernorReason.CLIPPED

    previous_sign = _sign(self.previous_output, 1e-4)
    target_sign = _sign(clipped, 1e-4)
    sign_change = previous_sign != 0 and target_sign != 0 and previous_sign != target_sign
    target_decreases_same_direction = previous_sign != 0 and target_sign == previous_sign and abs(clipped) <= abs(self.previous_output)
    slew_rate = speed_model.sign_change_slew_rate if sign_change else speed_model.output_slew_rate
    if sign_change:
      reason |= TorqueV4GovernorReason.SIGN_CHANGE_LIMITED
    if recenter is not None and recenter.active and (sign_change or target_decreases_same_direction):
      # Apply recenter slew boost: allow faster unwind (sign-change or
      # same-direction unwind) when recentering. The same-direction
      # path uses a smaller boost (RECENTER_SAME_DIRECTION_SLEW_BOOST)
      # because we just want a bit more headroom, not the full
      # sign-change boost. This is the "apply recenter boost to
      # same-direction unwind too" change.
      if sign_change:
        slew_rate *= recenter.slew_boost
      else:
        slew_rate *= RECENTER_SAME_DIRECTION_SLEW_BOOST
      reason |= TorqueV4GovernorReason.RECENTER_MODE
    # v5.0 turn-exit unwind boost. Only fires when the v5 context
    # reports turn_exit_active, only on unwind frames (sign change
    # or same-direction decrease), and only as a bounded slew-rate
    # multiplier. Output caps, sign-change slew rates, and safety
    # caps remain at v4.1 values.
    if (
      v5_context is not None
      and v5_context.turn_exit_active
      and (sign_change or target_decreases_same_direction)
    ):
      if sign_change:
        slew_rate *= TORQUE_V5_TURN_EXIT_SIGN_CHANGE_SLEW_BOOST
      else:
        slew_rate *= TORQUE_V5_TURN_EXIT_SAME_DIRECTION_SLEW_BOOST
    if high_rate_blend > 0.0:
      slew_rate *= self.profile.high_rate_slew_scale
    if same_direction_limit and not steer_limit_unwind:
      same_direction_rate = self._same_direction_limit_rate(v_ego)
      if under_response_recovery > 0.0:
        recovery_target_rate = speed_model.sign_change_slew_rate if sign_change else speed_model.output_slew_rate
        recovery_target_rate *= _interp(v_ego, UNDER_RESPONSE_RECOVERY_RATE_SCALE_BP, UNDER_RESPONSE_RECOVERY_RATE_SCALE_V)
        same_direction_rate = max(
          same_direction_rate,
          same_direction_rate + under_response_recovery * (recovery_target_rate - same_direction_rate),
        )
      slew_rate = min(slew_rate, same_direction_rate)
    if stale_actuator_mismatch:
      slew_rate = min(slew_rate, STALE_ACTUATOR_CAP)

    output = clipped if self.profile.same_direction_decrease_bypass and target_decreases_same_direction else \
      _approach(self.previous_output, clipped, slew_rate * self.dt)
    if abs(output - clipped) > 1e-6:
      reason |= TorqueV4GovernorReason.SLEW_LIMITED
    self.previous_output = output
    return TorqueV4GovernorResult(output, reason, output_cap)

  def _same_direction_limit_rate(self, v_ego: float) -> float:
    if self.profile.same_direction_limit_rate_bp is not None and self.profile.same_direction_limit_rate_v is not None:
      return _interp(v_ego, self.profile.same_direction_limit_rate_bp, self.profile.same_direction_limit_rate_v)
    return self.profile.same_direction_limit_rate

  @staticmethod
  def _under_response_recovery_strength(*, allowed: bool, v_ego: float, steering_pressed: bool,
                                         recovery_target_lateral_accel: float, actual_lateral_accel: float,
                                         raw_output_torque: float, high_rate_blend: float,
                                         stale_actuator_mismatch: bool) -> float:
    if not allowed or steering_pressed or high_rate_blend > 0.0 or stale_actuator_mismatch:
      return 0.0
    if not _finite(v_ego, recovery_target_lateral_accel, actual_lateral_accel, raw_output_torque):
      return 0.0

    desired_sign = _sign(recovery_target_lateral_accel, LOW_SPEED_UNDER_RESPONSE_SIGN_THRESHOLD)
    output_sign = _sign(raw_output_torque, LOW_SPEED_UNDER_RESPONSE_SIGN_THRESHOLD)
    if desired_sign == 0 or output_sign != desired_sign:
      return 0.0
    return _under_response_strength(recovery_target_lateral_accel, actual_lateral_accel)


class LatControlTorqueV4(LatControl):
  CONTROL_STATE = "torque"
  VERSION = 4
  UNDER_RESPONSE_RELEASE_HOLD = False
  UNDER_RESPONSE_CATCHUP_ENABLED = False
  UNDER_RESPONSE_CATCHUP_GAIN_BP = []
  UNDER_RESPONSE_CATCHUP_GAIN_V = []
  UNDER_RESPONSE_CATCHUP_CAP_BP = []
  UNDER_RESPONSE_CATCHUP_CAP_V = []
  UNDER_RESPONSE_CATCHUP_MAX_STEERING_RATE_DEG = HIGH_RATE_START_DEG
  # Gate for the active learned-bias term that compensates the
  # controller command lateral accel using the LateralVehicleHealthEstimator
  # output. The estimator itself is kept active (learning and telemetry
  # only); the gate controls whether the learned bias is allowed to
  # shape the command. Off in 4.1 so the 4.1 torque path is unchanged
  # by learned bias. A future LatControlTorqueV5 can set this True
  # intentionally to start applying the bias term to the command.
  ACTIVE_VEHICLE_BIAS_COMPENSATION = False
  GOVERNOR_PROFILE = TorqueV4GovernorProfile(
    output_slew_rate_bp=OUTPUT_SLEW_RATE_BP,
    output_slew_rate_v=OUTPUT_SLEW_RATE_V,
    sign_change_slew_rate_bp=SIGN_CHANGE_SLEW_RATE_BP,
    sign_change_slew_rate_v=SIGN_CHANGE_SLEW_RATE_V,
    same_direction_limit_cap=SAME_DIRECTION_LIMIT_CAP,
    same_direction_limit_rate=SAME_DIRECTION_LIMIT_RATE,
    high_rate_start_deg=HIGH_RATE_START_DEG,
    high_rate_full_deg=HIGH_RATE_FULL_DEG,
    high_rate_min_cap=HIGH_RATE_MIN_CAP,
    high_rate_slew_scale=HIGH_RATE_SLEW_SCALE,
  )

  def __init__(self, CP, CP_SP, CI, dt):
    super().__init__(CP, CP_SP, CI, dt)
    del CP_SP
    if CP.lateralTuning.which() != "torque":
      raise ValueError("Torque v4 requires native torque lateral tuning")
    self.CP = CP
    self.params = Params()
    self.torque_params = CP.lateralTuning.torque.as_builder()
    self.torque_from_lateral_accel = CI.torque_from_lateral_accel()
    self.steering_angle_deadzone_deg = self.torque_params.steeringAngleDeadzoneDeg
    initial_delay = max(float(getattr(CP, "steerActuatorDelay", 0.2)), self.dt)
    self.speed_model = TorqueV4SpeedModel(self.GOVERNOR_PROFILE)
    self.session_adaptation = TorqueV4SessionAdaptation(initial_delay)
    self.governor = TorqueV4OutputGovernor(self.dt, self.GOVERNOR_PROFILE)
    self.speed_aware_params = None
    self.speed_adaptive_apply_enabled = False
    self.previous_target_lateral_accel = 0.0
    self.previous_measurement = 0.0
    self.filtered_measurement_rate = 0.0
    self.last_v_ego = 0.0
    self.processed_lateral_demand = None
    self.processed_lateral_demand: ProcessedLateralDemand | None = None
    self.lateral_demand_profile: LateralDemandProfile | None = None
    self.oscillation_classifier = LateralOscillationClassifier()
    self._last_oscillation_classification: str = "none"
    self._last_oscillation_confidence: float = 0.0
    self._previous_output_torque: float = 0.0
    self._wobble_active: bool = False
    self._last_straight_road: bool = False
    self._last_wobble_response: WobbleResponse = compute_wobble_response("none", 0.0)
    self.vehicle_health_estimator = LateralVehicleHealthEstimator()
    self._last_health_estimate: LateralVehicleHealthEstimate = LateralVehicleHealthEstimate()
    self._previous_saturated: bool = False
    self.turn_exit_controller = LateralTurnExitController()
    self._last_turn_exit_mode: str = TurnExitMode.INACTIVE.value
    self._last_turn_exit_persistence: int = 0
    self._last_turn_exit_preview_boost: float = 0.0
    self._last_turn_exit_early_release: bool = False
    # v5.0 pre-target turn-exit seam. When set, the post-command
    # telemetry path uses self._v5_cached_turn_exit_decision instead of
    # re-calling turn_exit_controller.update(). v4 always leaves
    # this False, so the post-command call runs every frame.
    self._v5_turn_exit_decided: bool = False
    self._v5_cached_turn_exit_decision: TurnExitDecision | None = None
    # v5.0 last-frame shaping telemetry for the governor context.
    self._v5_last_preview_active: bool = False
    self._v5_last_turn_exit_active: bool = False
    self._v5_last_v5_active: bool = False
    self._v5_last_v5_reason: str = ""
    # v5.0 route-validation counters and last-frame values.
    self._v5_output_sign_flips: int = 0
    self._v5_straight_road_torque_flips: int = 0
    self._v5_last_output_sign: int = 0
    self._v5_last_final_lead_delta: float = 0.0
    self._v5_last_preview_applied_value: float = 0.0
    self._recenter_persistence_frames: int = 0

  def update_live_torque_params(self, latAccelFactor, latAccelOffset, friction) -> None:
    self.torque_params.latAccelFactor = latAccelFactor
    self.torque_params.latAccelOffset = latAccelOffset
    self.torque_params.friction = friction

  def update_speed_aware_params(self, params_str) -> None:
    self._refresh_speed_adaptive_apply_enabled()
    if not params_str:
      self.speed_aware_params = None
      return
    try:
      if isinstance(params_str, bytes):
        params_str = params_str.decode("utf-8")
      self.speed_aware_params = parse_speed_aware_params(self.CP, ast.literal_eval(params_str))
    except (TypeError, UnicodeDecodeError, ValueError, SyntaxError):
      self.speed_aware_params = None

  def update_lateral_lag(self, lag) -> None:
    try:
      lag = float(lag)
    except (TypeError, ValueError):
      lag = self.dt
    self.session_adaptation.update_lateral_lag(lag if math.isfinite(lag) else self.dt)

  def set_processed_lateral_demand(self, demand: ProcessedLateralDemand) -> None:
    self.processed_lateral_demand = demand

  def set_lateral_demand_profile(self, profile: LateralDemandProfile) -> None:
    self.lateral_demand_profile = profile

  def reset(self) -> None:
    super().reset()
    self.governor.reset()
    self.session_adaptation.reset()
    self.previous_target_lateral_accel = 0.0
    self.previous_measurement = 0.0
    self.filtered_measurement_rate = 0.0
    self._recenter_persistence_frames = 0
    self.lateral_demand_profile = None
    self.oscillation_classifier = LateralOscillationClassifier()
    self._last_oscillation_classification = "none"
    self._last_oscillation_confidence = 0.0
    self._previous_output_torque = 0.0
    self._wobble_active = False
    self._last_straight_road = False
    self._last_wobble_response = compute_wobble_response("none", 0.0)
    self.vehicle_health_estimator = LateralVehicleHealthEstimator()
    self._last_health_estimate = LateralVehicleHealthEstimate()
    self._previous_saturated = False
    self.turn_exit_controller = LateralTurnExitController()
    self._last_turn_exit_mode = TurnExitMode.INACTIVE.value
    self._last_turn_exit_persistence = 0
    self._last_turn_exit_preview_boost = 0.0
    self._last_turn_exit_early_release = False
    self._v5_turn_exit_decided = False
    self._v5_cached_turn_exit_decision = None
    self._v5_last_preview_active = False
    self._v5_last_turn_exit_active = False
    self._v5_last_v5_active = False
    self._v5_last_v5_reason = ""
    self._v5_output_sign_flips = 0
    self._v5_straight_road_torque_flips = 0
    self._v5_last_output_sign = 0
    self._v5_last_final_lead_delta = 0.0
    self._v5_last_preview_applied_value = 0.0

  def _under_response_recovery_allowed(self) -> bool:
    if self._wobble_active:
      return False
    demand = self.processed_lateral_demand
    if demand is None:
      return False
    path_quality = _finite_float(getattr(demand, "path_quality", None))
    lane_change_blend = _finite_float(getattr(demand, "lane_change_blend", None))
    path_reason = getattr(demand, "path_reason", None)
    low_speed_usable_path = (
      path_reason in UNDER_RESPONSE_LOW_SPEED_ALLOWED_PATH_REASONS
      and path_quality is not None
      and path_quality >= LOW_SPEED_UNDER_RESPONSE_MIN_PATH_QUALITY
      and self.last_v_ego < LOW_SPEED_UNDER_RESPONSE_FADE_SPEED
    )
    return (
      getattr(demand, "demand_source", None) == DEMAND_SOURCE_MODEL_PATH
      and path_quality is not None
      and ((path_reason == LEARN_PATH_REASON_OK and path_quality >= LEARN_MIN_PATH_QUALITY) or low_speed_usable_path)
      and not getattr(demand, "lane_change_shaping_active", True)
      and lane_change_blend is not None
      and abs(lane_change_blend) <= 1e-3
      and not getattr(demand, "lane_centering_assist_active", False)
    )

  def _detect_recenter_mode(self, *, target_lateral_accel: float, previous_target_lateral_accel: float,
                            v_ego: float, path_quality: float, lane_change_active: bool,
                            steering_pressed: bool, saturated: bool, curvature_limited: bool) -> TorqueV4RecenterMode:
    """Detect recenter mode: target lateral accel collapsing toward zero on a straight-ish road."""
    # Conditions that disqualify recenter mode
    if (not _finite(target_lateral_accel, previous_target_lateral_accel, v_ego)
        or v_ego < RECENTER_MIN_SPEED
        or lane_change_active
        or steering_pressed
        or saturated
        or curvature_limited
        or path_quality < RECENTER_MIN_PATH_QUALITY):
      self._recenter_persistence_frames = 0
      return TorqueV4RecenterMode()

    # Target must be decreasing in magnitude (collapsing toward zero)
    target_decreasing = abs(target_lateral_accel) < abs(previous_target_lateral_accel)
    # Target must be small enough to be "near center"
    target_near_center = abs(target_lateral_accel) < RECENTER_MAX_ABS_TARGET
    # Rate of decrease must be significant
    decrease_rate = abs(previous_target_lateral_accel) - abs(target_lateral_accel)
    target_collapsing = decrease_rate > RECENTER_TARGET_COLLAPSE_RATE * self.dt

    if target_decreasing and target_near_center and target_collapsing:
      self._recenter_persistence_frames = min(self._recenter_persistence_frames + 1, RECENTER_PERSISTENCE_FRAMES * 3)
    else:
      self._recenter_persistence_frames = max(0, self._recenter_persistence_frames - 1)

    # Only activate after persistence threshold
    if self._recenter_persistence_frames >= RECENTER_PERSISTENCE_FRAMES:
      # Blend in the recenter effect based on how persistent the recentering is.
      # The blend starts at RECENTER_LEAD_REDUCTION_FLOOR (non-zero) so the
      # first active frame already trims some response-delay lead, instead
      # of waiting the full RECENTER_PERSISTENCE_FRAMES ramp. This is the
      # "partial lead_reduction from first active frame" behavior: the
      # turn-exit feel is immediate even on a short recenter.
      ramp = (self._recenter_persistence_frames - RECENTER_PERSISTENCE_FRAMES) / RECENTER_PERSISTENCE_FRAMES
      ramp_blend = max(0.0, min(1.0, float(ramp)))
      persistence_blend = RECENTER_LEAD_REDUCTION_FLOOR + (1.0 - RECENTER_LEAD_REDUCTION_FLOOR) * ramp_blend
      return TorqueV4RecenterMode(
        active=True,
        persistence_frames=self._recenter_persistence_frames,
        lead_reduction=RECENTER_LEAD_REDUCTION * persistence_blend,
        slew_boost=1.0 + (RECENTER_SLEW_BOOST - 1.0) * persistence_blend,
      )

    return TorqueV4RecenterMode()

  def _refresh_speed_adaptive_apply_enabled(self) -> None:
    try:
      self.speed_adaptive_apply_enabled = self.params.get_bool("LiveTorqueSpeedAdaptiveApplyToggle")
    except UnknownKeyName:
      self.speed_adaptive_apply_enabled = False

  def update(self, active, CS, VM, params, steer_limited_by_safety, desired_curvature, calibrated_pose, curvature_limited, lat_delay):
    # Torque v4 follows only the processed controller-facing curvature passed in by controlsd.
    del calibrated_pose
    self.last_v_ego = CS.vEgo if _finite(CS.vEgo) else 0.0
    if _finite(lat_delay):
      self.update_lateral_lag(lat_delay)

    pid_log = log.ControlsState.LateralTorqueState.new_message()
    pid_log.version = self.VERSION

    input_invalid = not _finite(desired_curvature, CS.vEgo, CS.steeringAngleDeg, CS.steeringRateDeg,
                                params.angleOffsetDeg, params.roll)
    speed_result = self.speed_model.update(CS.vEgo if _finite(CS.vEgo) else 0.0, self.torque_params,
                                           self.speed_aware_params, self.speed_adaptive_apply_enabled,
                                           self.session_adaptation)
    recenter = self._detect_recenter_mode(
      target_lateral_accel=0.0 if input_invalid else desired_curvature * CS.vEgo ** 2,
      previous_target_lateral_accel=self.previous_target_lateral_accel,
      v_ego=CS.vEgo if _finite(CS.vEgo) else 0.0,
      path_quality=(_finite_float(getattr(self.processed_lateral_demand, "path_quality", None)) if self.processed_lateral_demand is not None else 1.0) or 1.0,
      lane_change_active=getattr(self.processed_lateral_demand, "lane_change_shaping_active", False) if self.processed_lateral_demand is not None else False,
      steering_pressed=CS.steeringPressed,
      # Recentering is active command shaping. It must not fire
      # immediately after a saturated frame; gate it on the
      # previous frame's saturation.
      saturated=self._previous_saturated,
      curvature_limited=curvature_limited,
    )
    target = self._build_target(0.0 if input_invalid else desired_curvature, CS.vEgo, speed_result, input_invalid, recenter=recenter,
                                curvature_limited=curvature_limited, cs=CS,
                                active=active, steer_limited_by_safety=steer_limited_by_safety)
    steering_angle_rad = math.radians(CS.steeringAngleDeg - params.angleOffsetDeg) if not input_invalid else 0.0
    measured_curvature = -VM.calc_curvature(steering_angle_rad, CS.vEgo, params.roll) if not input_invalid else 0.0
    actual_lateral_accel = measured_curvature * CS.vEgo ** 2 if not input_invalid else 0.0
    actual_lateral_jerk = finite_difference_curvature_rate_from_steering_rate(
      VM,
      steering_angle_rad,
      math.radians(CS.steeringRateDeg) if not input_invalid else 0.0,
      CS.vEgo if not input_invalid else 0.0,
      params.roll if not input_invalid else 0.0,
    )
    measurement_rate = self._filtered_measurement_rate(active, input_invalid, actual_lateral_accel)
    raw_desired_curvature = float(desired_curvature) if _finite(desired_curvature) else 0.0
    processed_curvature_for_classifier = _finite_float(getattr(self.processed_lateral_demand, "processed_curvature", None))
    if processed_curvature_for_classifier is None:
      processed_curvature_for_classifier = raw_desired_curvature
    classifier_result = self.oscillation_classifier.update(
      raw_curvature=raw_desired_curvature,
      processed_curvature=float(processed_curvature_for_classifier),
      target_lateral_accel=target.raw_lateral_accel,
      actual_lateral_accel=actual_lateral_accel,
      # Pass the previous frame's post-governor output torque so the
      # classifier can detect real torque flips. Using 0.0 here would
      # mean every input looks like a sign change on the first frame
      # and zero sign changes afterwards, which masks the wobble the
      # controller is supposed to detect.
      torque_output=self._previous_output_torque,
      path_quality=getattr(self.processed_lateral_demand, "path_quality", 1.0),
      lane_change_active=getattr(self.processed_lateral_demand, "lane_change_shaping_active", False),
      v_ego=CS.vEgo,
      curvature_limited=curvature_limited,
      steering_pressed=CS.steeringPressed,
    )
    self._last_oscillation_classification = classifier_result.classification
    self._last_oscillation_confidence = float(classifier_result.confidence)
    self._wobble_active = is_wobble_active(classifier_result.classification, classifier_result.confidence)
    self._last_straight_road = bool(classifier_result.straight_road)
    self._last_wobble_response = compute_wobble_response(classifier_result.classification, float(classifier_result.confidence))
    self._last_health_estimate = self.vehicle_health_estimator.update(
      v_ego=CS.vEgo if not input_invalid else 0.0,
      target_lateral_accel=target.raw_lateral_accel,
      actual_lateral_accel=actual_lateral_accel,
      path_quality=_finite_float(getattr(self.processed_lateral_demand, "path_quality", 0.0)) or 0.0,
      demand_source=getattr(self.processed_lateral_demand, "demand_source", DEMAND_SOURCE_MODEL_PATH),
      lane_change_active=bool(getattr(self.processed_lateral_demand, "lane_change_shaping_active", False)),
      steering_pressed=CS.steeringPressed,
      curvature_limited=curvature_limited,
      # The estimator cannot know this frame's saturation status yet
      # (it is determined after the command path runs), so feed the
      # previous frame's saturation. Hard-coding False here meant
      # bias learning ignored saturation entirely.
      saturated=self._previous_saturated,
    )
    target = self._apply_under_response_lead_boost(
      target,
      speed_result,
      CS.vEgo,
      active=active,
      steering_pressed=CS.steeringPressed,
      steering_rate_deg=CS.steeringRateDeg,
      actual_lateral_accel=actual_lateral_accel,
      invalid=input_invalid,
    )

    roll_compensation = roll_lateral_accel(params.roll) if not input_invalid else 0.0
    lateral_accel_deadzone = 0.0
    if not input_invalid:
      curvature_deadzone = abs(VM.calc_curvature(math.radians(self.steering_angle_deadzone_deg), CS.vEgo, 0.0))
      lateral_accel_deadzone = curvature_deadzone * CS.vEgo ** 2

    control_error = target.delay_lead_lateral_accel - actual_lateral_accel
    under_response_catchup = self._under_response_catchup_correction(
      target.raw_lateral_accel,
      actual_lateral_accel,
      v_ego=CS.vEgo,
      steering_rate_deg=CS.steeringRateDeg,
      active=active,
      steering_pressed=CS.steeringPressed,
      invalid=input_invalid,
    )
    feedback_correction = (speed_result.feedback_gain * self._last_wobble_response.feedback_gain_multiplier) * control_error + under_response_catchup / max(
      speed_result.response_scale, 1e-3,
    )
    damping_correction = -(speed_result.damping_gain * self._last_wobble_response.damping_gain_multiplier) * measurement_rate
    breakaway_compensation = self._breakaway_lateral_accel(control_error, lateral_accel_deadzone, target.raw_lateral_accel,
                                                            actual_lateral_accel, speed_result.breakaway_scale)
    # 4.1 keeps the learned bias diagnostic-only. Estimator state
    # still updates (and telemetry still surfaces it) but the
    # compensation term is forced to 0.0 so the 4.1 torque command is
    # bit-equivalent to the pre-bias-gate behavior. A future
    # LatControlTorqueV5 can flip ACTIVE_VEHICLE_BIAS_COMPENSATION
    # to True to start applying the term.
    bias_compensation = (
      self._vehicle_bias_compensation(self._last_health_estimate, wobble_active=self._wobble_active)
      if self.ACTIVE_VEHICLE_BIAS_COMPENSATION
      else 0.0
    )
    command_lateral_accel = (
      target.delay_lead_lateral_accel
      - roll_compensation
      - float(self.torque_params.latAccelOffset)
      + speed_result.response_scale * feedback_correction
      + damping_correction
      + speed_result.trim_lateral_accel
      + breakaway_compensation
      - bias_compensation
    )
    invalid = input_invalid or not _finite(command_lateral_accel, actual_lateral_accel, actual_lateral_jerk, measurement_rate)
    effective_torque_params = self._effective_torque_params(speed_result)
    raw_output_torque = 0.0 if invalid else self.torque_from_lateral_accel(command_lateral_accel, effective_torque_params)
    raw_output_torque = _clip(raw_output_torque, -self.steer_max, self.steer_max) if _finite(raw_output_torque) else 0.0

    steer_limit_feedback = self.steering_actuator_feedback
    steer_limit_context = classify_steering_limit_context(steer_limit_feedback, -raw_output_torque)
    same_direction_limit = bool(steer_limited_by_safety and (steer_limit_context.same_direction_limited if steer_limit_feedback.valid else True))
    governor_result = self.governor.update(
      active=active and not invalid,
      v_ego=CS.vEgo,
      steering_pressed=CS.steeringPressed,
      steering_rate_deg=CS.steeringRateDeg,
      same_direction_limit=same_direction_limit,
      steer_limit_unwind=steer_limit_context.unwind_allowed if steer_limit_feedback.valid else False,
      actuator_mismatch=bool(steer_limit_feedback.valid and steer_limit_feedback.limited),
      actuator_error=float(steer_limit_feedback.error if steer_limit_feedback.valid else 0.0),
      raw_output_torque=raw_output_torque,
      max_output=self.steer_max,
      speed_model=speed_result,
      recovery_target_lateral_accel=target.delay_lead_lateral_accel,
      actual_lateral_accel=actual_lateral_accel,
      under_response_recovery_allowed=self._under_response_recovery_allowed(),
      recenter=recenter,
      v5_context=self._v5_governor_context(target=target),
    )
    if invalid:
      governor_result = TorqueV4GovernorResult(0.0, governor_result.reason | TorqueV4GovernorReason.INVALID, governor_result.output_cap)
    output_torque = governor_result.output_torque
    saturated = self.steer_max - abs(output_torque) < 1e-3 or bool(governor_result.reason & TorqueV4GovernorReason.CLIPPED)
    self._previous_output_torque = float(output_torque)
    self._previous_saturated = bool(saturated)
    # v5.0 route-validation counters. Track sign flips of the
    # finalized output torque and how many of those happened on
    # straight road. Reset happens in reset().
    self._v5_record_output_torque_sign_flips(output_torque)

    observation = TorqueV4Observation(
      active=active and not invalid,
      v_ego=CS.vEgo,
      steering_pressed=CS.steeringPressed,
      steer_limited_by_safety=same_direction_limit,
      curvature_limited=curvature_limited,
      saturated=saturated,
      raw_target_lateral_accel=target.raw_lateral_accel,
      delay_lead_lateral_accel=target.delay_lead_lateral_accel,
      target_lateral_accel_rate=target.target_rate,
      actual_lateral_accel=actual_lateral_accel,
      actual_lateral_jerk=actual_lateral_jerk,
      measurement_rate=measurement_rate,
      finite=not invalid,
      demand_source=getattr(self.processed_lateral_demand, "demand_source", DEMAND_SOURCE_MODEL_PATH),
      path_quality=getattr(self.processed_lateral_demand, "path_quality", 1.0),
      path_reason=getattr(self.processed_lateral_demand, "path_reason", LEARN_PATH_REASON_OK),
      lane_change_shaping_active=getattr(self.processed_lateral_demand, "lane_change_shaping_active", False),
      lane_change_blend=getattr(self.processed_lateral_demand, "lane_change_blend", 0.0),
      lane_centering_assist_active=getattr(self.processed_lateral_demand, "lane_centering_assist_active", False),
    )
    sample_update = self.session_adaptation.update(observation, governor_result.reason)

    turn_exit_decision = self._v5_record_turn_exit_telemetry(
      target=target, active=active, CS=CS, curvature_limited=curvature_limited,
      saturated=saturated,
    )
    self._last_turn_exit_mode = turn_exit_decision.mode
    self._last_turn_exit_persistence = int(turn_exit_decision.persistence_frames)
    self._last_turn_exit_preview_boost = float(turn_exit_decision.preview_boost)
    self._last_turn_exit_early_release = bool(turn_exit_decision.early_release_lead_zero)

    pid_log.active = bool(active and not invalid)
    pid_log.error = float(control_error if _finite(control_error) else 0.0)
    pid_log.errorRate = float(-measurement_rate if _finite(measurement_rate) else 0.0)
    pid_log.p = float(feedback_correction if _finite(feedback_correction) else 0.0)
    pid_log.i = float(speed_result.trim_lateral_accel if _finite(speed_result.trim_lateral_accel) else 0.0)
    pid_log.d = float(damping_correction if _finite(damping_correction) else 0.0)
    pid_log.f = float(command_lateral_accel if _finite(command_lateral_accel) else 0.0)
    pid_log.output = float(-output_torque)
    pid_log.actualLateralAccel = float(actual_lateral_accel if _finite(actual_lateral_accel) else 0.0)
    pid_log.desiredLateralAccel = float(target.raw_lateral_accel if _finite(target.raw_lateral_accel) else 0.0)
    pid_log.desiredLateralJerk = float(target.target_rate if _finite(target.target_rate) else 0.0)
    pid_log.saturated = bool(self._check_saturation(saturated, CS, steer_limited_by_safety, curvature_limited))
    self._fill_adaptive_log(pid_log, active, target, feedback_correction, damping_correction, raw_output_torque,
                            governor_result, sample_update, speed_result, steer_limit_feedback,
                            steer_limit_context.same_direction_limited, steer_limit_context.unwind_allowed,
                            actual_lateral_jerk)
    if not active:
      self.session_adaptation.reset()

    return -output_torque, 0.0, pid_log

  def _build_target_base(self, desired_curvature: float, v_ego: float, speed_result: TorqueV4SpeedModelResult,
                         invalid: bool, recenter: TorqueV4RecenterMode | None = None) -> TorqueV4Target:
    """Base target builder shared by v4.0, v4.1, and v5.0.

    Computes raw target, target rate, lead gain / cap, the early-
    release guard, and the recenter lead reduction. The seam lets
    v5.0 wrap this method to add profile-aware shaping without
    duplicating the base behavior.
    """
    raw_target = 0.0 if invalid else desired_curvature * v_ego ** 2
    target_rate = 0.0 if invalid else (raw_target - self.previous_target_lateral_accel) / self.dt
    previous_target = self.previous_target_lateral_accel
    self.previous_target_lateral_accel = raw_target
    lead_gain = speed_result.lead_gain
    lead_delta_cap = speed_result.lead_delta_cap
    # Early release guard: when the target is collapsing toward zero and
    # the lead delta would push the controller output away from zero,
    # zero the lead delta. This fires before abs(target) < RECENTER_MAX_ABS_TARGET
    # so turn exit feels immediate, not just near the center threshold.
    # Conditions:
    # - raw target is decreasing in magnitude (|raw_target| < |previous|)
    # - lead delta would push away from zero (sign(lead_delta) != sign(raw_target))
    # - target sign is stable (sign(raw_target) == sign(previous), or both zero)
    target_decreasing_to_zero = abs(raw_target) < abs(previous_target)
    target_sign_stable = (raw_target == 0.0 and previous_target == 0.0) or (
      raw_target != 0.0 and previous_target != 0.0 and
      (raw_target > 0.0) == (previous_target > 0.0)
    )
    # Reduce lead when recentering — the target is collapsing toward zero, so lead overshoots
    if recenter is not None and recenter.active:
      lead_gain *= (1.0 - recenter.lead_reduction)
      lead_delta_cap *= (1.0 - recenter.lead_reduction)
    lead_delta = _clip(target_rate * speed_result.response_delay * lead_gain,
                       -lead_delta_cap, lead_delta_cap)
    if (
      target_decreasing_to_zero
      and target_sign_stable
      and raw_target != 0.0
      and lead_delta != 0.0
      and (lead_delta > 0.0) != (raw_target > 0.0)
    ):
      # Early release guard: zero the lead delta so the response-delay
      # overshoot doesn't pull the controller output away from zero
      # when the target is collapsing. This complements the recenter
      # lead reduction (which only fires after RECENTER_PERSISTENCE_FRAMES)
      # by making turn-exit feel immediate on the very first frame the
      # target starts collapsing.
      lead_delta = 0.0
    return TorqueV4Target(raw_target, target_rate, raw_target + lead_delta, lead_delta,
                          lead_gain, lead_delta_cap)

  def _build_target(self, desired_curvature: float, v_ego: float, speed_result: TorqueV4SpeedModelResult,
                    invalid: bool, recenter: TorqueV4RecenterMode | None = None,
                    curvature_limited: bool = False, cs=None,
                    active: bool = True,
                    steer_limited_by_safety: bool = False) -> TorqueV4Target:
    """Public target build seam. v4.0/v4.1 forward to the base
    implementation; v5.0 overrides this to add profile shaping.

    `curvature_limited`, `cs`, `active`, and
    `steer_limited_by_safety` are unused by the base; v5.0 uses
    them when routing the pre-target turn-exit decision and the
    preview gate.
    """
    return self._build_target_base(desired_curvature, v_ego, speed_result, invalid, recenter)

  def _v5_turn_exit_decision(self, target: TorqueV4Target, *, active: bool,
                             CS, curvature_limited: bool) -> None:
    """Pre-target turn-exit decision seam. v4 inherits as a no-op;
    v5 overrides to call turn_exit_controller.update() before the
    target is built and store the decision in self._v5_cached_turn_exit_decision.

    The decision is not yet used to shape the target; that lands in
    later commits. This commit only establishes the wiring.
    """
    return None

  def _v5_record_turn_exit_telemetry(self, *, target: TorqueV4Target, active: bool,
                                     CS, curvature_limited: bool, saturated: bool) -> TurnExitDecision:
    """Post-command turn-exit telemetry seam. v4 calls
    turn_exit_controller.update() and stores the result for
    telemetry. v5 with ACTIVE_TURN_EXIT_CONTROLLER=True uses the
    pre-target cached decision instead.
    """
    decision = self.turn_exit_controller.update(
      target=target.raw_lateral_accel,
      profile=self.lateral_demand_profile,
      active=bool(active and not self._v5_turn_exit_decided),
      v_ego=CS.vEgo,
      path_quality=_finite_float(getattr(self.processed_lateral_demand, "path_quality", 1.0)) or 1.0,
      lane_change_active=bool(getattr(self.processed_lateral_demand, "lane_change_shaping_active", False)),
      steering_pressed=CS.steeringPressed,
      curvature_limited=curvature_limited,
      # The actual saturation is known at this point in the
      # update flow. The turn-exit controller must receive it
      # so an early-release decision is suppressed when the
      # previous frame's output was clipped.
      saturated=saturated,
    )
    return decision

  def _v5_governor_context(self, *, target: TorqueV4Target) -> TorqueV5GovernorContext | None:
    """v4 always returns None: the governor sees no v5 context
    and applies the v4.1 slew rates and caps unchanged. v5
    overrides this to return a populated context.
    """
    return None

  def _v5_record_output_torque_sign_flips(self, output_torque: float) -> None:
    """v4 no-op. v5 overrides to maintain the
    output_sign_flips and straight_road_torque_flips counters
    that power route validation.
    """
    return None

  def _apply_under_response_lead_boost(self, target: TorqueV4Target, speed_result: TorqueV4SpeedModelResult, v_ego: float,
                                       *, active: bool, steering_pressed: bool, actual_lateral_accel: float,
                                       invalid: bool, steering_rate_deg: float = 0.0) -> TorqueV4Target:
    if invalid or not active or steering_pressed or not self._under_response_recovery_allowed():
      return target
    strength = _under_response_strength(target.delay_lead_lateral_accel, actual_lateral_accel)
    release_hold_allowed = (
      self.UNDER_RESPONSE_RELEASE_HOLD
      and self._under_response_enhancement_allowed()
      and _finite(steering_rate_deg)
      and abs(steering_rate_deg) < self.UNDER_RESPONSE_CATCHUP_MAX_STEERING_RATE_DEG
    )
    raw_strength = _under_response_strength(target.raw_lateral_accel, actual_lateral_accel) if release_hold_allowed else 0.0
    if strength <= 0.0 and raw_strength <= 0.0:
      return target

    lead_gain = target.lead_gain * (1.0 + strength * _interp(v_ego, UNDER_RESPONSE_LEAD_GAIN_BOOST_BP, UNDER_RESPONSE_LEAD_GAIN_BOOST_V))
    lead_delta_cap = target.lead_delta_cap * (1.0 + strength * _interp(v_ego, UNDER_RESPONSE_LEAD_CAP_BOOST_BP, UNDER_RESPONSE_LEAD_CAP_BOOST_V))
    lead_delta = _clip(target.target_rate * speed_result.response_delay * lead_gain, -lead_delta_cap, lead_delta_cap)
    if release_hold_allowed:
      raw_sign = _sign(target.raw_lateral_accel, LOW_SPEED_UNDER_RESPONSE_SIGN_THRESHOLD)
      lead_sign = _sign(lead_delta, LOW_SPEED_UNDER_RESPONSE_SIGN_THRESHOLD)
      actual_sign = _sign(actual_lateral_accel, LOW_SPEED_UNDER_RESPONSE_SIGN_THRESHOLD)
      lagging_raw_target = abs(actual_lateral_accel) < abs(target.raw_lateral_accel)
      if raw_strength > 0.0 and raw_sign != 0 and actual_sign in (0, raw_sign) and lagging_raw_target and lead_sign == -raw_sign:
        lead_delta = 0.0
    return TorqueV4Target(
      target.raw_lateral_accel,
      target.target_rate,
      target.raw_lateral_accel + lead_delta,
      lead_delta,
      lead_gain,
      lead_delta_cap,
    )

  def _under_response_catchup_correction(self, raw_target_lateral_accel: float, actual_lateral_accel: float, *,
                                         v_ego: float, steering_rate_deg: float, active: bool,
                                         steering_pressed: bool, invalid: bool) -> float:
    if not self.UNDER_RESPONSE_CATCHUP_ENABLED or invalid or not active or steering_pressed:
      return 0.0
    if not _finite(raw_target_lateral_accel, actual_lateral_accel, v_ego, steering_rate_deg):
      return 0.0
    if abs(steering_rate_deg) >= self.UNDER_RESPONSE_CATCHUP_MAX_STEERING_RATE_DEG:
      return 0.0
    if not self._under_response_enhancement_allowed():
      return 0.0

    strength = _under_response_strength(raw_target_lateral_accel, actual_lateral_accel)
    if strength <= 0.0:
      return 0.0
    target_sign = _sign(raw_target_lateral_accel, LOW_SPEED_UNDER_RESPONSE_SIGN_THRESHOLD)
    actual_sign = _sign(actual_lateral_accel, LOW_SPEED_UNDER_RESPONSE_SIGN_THRESHOLD)
    if target_sign == 0 or actual_sign not in (0, target_sign):
      return 0.0
    deficit = abs(raw_target_lateral_accel) - abs(actual_lateral_accel)
    if deficit <= LOW_SPEED_UNDER_RESPONSE_MARGIN:
      return 0.0

    gain = _interp(v_ego, self.UNDER_RESPONSE_CATCHUP_GAIN_BP, self.UNDER_RESPONSE_CATCHUP_GAIN_V)
    cap = _interp(v_ego, self.UNDER_RESPONSE_CATCHUP_CAP_BP, self.UNDER_RESPONSE_CATCHUP_CAP_V)
    correction = target_sign * min(max(deficit * gain * strength, 0.0), cap)
    return correction if _finite(correction) else 0.0

  def _under_response_enhancement_allowed(self) -> bool:
    if not self._under_response_recovery_allowed():
      return False
    return not getattr(self.processed_lateral_demand, "curvature_limited", True)

  def _filtered_measurement_rate(self, active: bool, invalid: bool, actual_lateral_accel: float) -> float:
    if invalid or not active or not _finite(actual_lateral_accel, self.previous_measurement):
      self.previous_measurement = actual_lateral_accel if _finite(actual_lateral_accel) else 0.0
      self.filtered_measurement_rate = 0.0
      return 0.0
    raw_rate = (actual_lateral_accel - self.previous_measurement) / self.dt
    self.previous_measurement = actual_lateral_accel
    bounded_rate = _clip(raw_rate, -MEASUREMENT_RATE_CAP, MEASUREMENT_RATE_CAP)
    self.filtered_measurement_rate += MEASUREMENT_RATE_FILTER_ALPHA * (bounded_rate - self.filtered_measurement_rate)
    return self.filtered_measurement_rate if _finite(self.filtered_measurement_rate) else 0.0

  def _effective_torque_params(self, speed_result: TorqueV4SpeedModelResult):
    effective_torque_params = self.CP.lateralTuning.torque.as_builder()
    effective_torque_params.friction = float(self.torque_params.friction)
    effective_torque_params.steeringAngleDeadzoneDeg = float(self.torque_params.steeringAngleDeadzoneDeg)
    effective_torque_params.latAccelFactor = float(speed_result.effective_lat_accel_factor)
    effective_torque_params.latAccelOffset = float(speed_result.effective_lat_accel_offset)
    return effective_torque_params

  def _breakaway_lateral_accel(self, error: float, lateral_accel_deadzone: float, target: float, measurement: float,
                               breakaway_scale: float) -> float:
    demand = max(abs(target), abs(measurement), abs(error))
    scale = _clip(demand / BREAKAWAY_FULL_DEMAND, 0.0, breakaway_scale)
    if _sign(target, LEARN_SIGN_THRESHOLD) != 0 and _sign(measurement, LEARN_SIGN_THRESHOLD) != 0 and _sign(target) != _sign(measurement):
      scale *= 0.5
    return get_friction(error * scale, lateral_accel_deadzone, FRICTION_THRESHOLD, self.torque_params) * scale

  def _vehicle_bias_compensation(self, estimate: LateralVehicleHealthEstimate, *, wobble_active: bool) -> float:
    if wobble_active:
      return 0.0
    if not _finite(estimate.bias_estimate, estimate.bias_confidence):
      return 0.0
    raw = float(estimate.bias_estimate) * float(estimate.bias_confidence)
    return _clip(raw, -HEALTH_EST_BIAS_MAX, HEALTH_EST_BIAS_MAX)

  @staticmethod
  def _phase(active: bool, target_rate: float, governor_reason: TorqueV4GovernorReason) -> TorqueV4Phase:
    if not active:
      return TorqueV4Phase.idle
    if governor_reason & (TorqueV4GovernorReason.DRIVER_OVERRIDE | TorqueV4GovernorReason.SIGN_CHANGE_LIMITED):
      return TorqueV4Phase.release
    if abs(target_rate) > 0.25:
      return TorqueV4Phase.engage if target_rate > 0.0 else TorqueV4Phase.release
    return TorqueV4Phase.hold

  def _fill_adaptive_log(self, pid_log, active: bool, target: TorqueV4Target, feedback_correction: float,
                          damping_correction: float, raw_output_torque: float, governor_result: TorqueV4GovernorResult,
                          sample_update: TorqueV4AdaptationUpdate, speed_result: TorqueV4SpeedModelResult,
                          steer_limit_feedback, steer_limit_same_direction: bool, steer_limit_unwind: bool,
                          actual_lateral_jerk: float) -> None:
    adaptive_log = pid_log.init('adaptiveTorqueState')
    log_active = bool(active and pid_log.active)
    adaptive_log.active = log_active
    adaptive_log.phase = PHASE_TO_CAPNP[self._phase(log_active, target.target_rate, governor_result.reason)]
    adaptive_log.releaseActive = bool(governor_result.reason & (TorqueV4GovernorReason.DRIVER_OVERRIDE | TorqueV4GovernorReason.SIGN_CHANGE_LIMITED))
    adaptive_log.phaseGain = float(target.lead_gain)
    adaptive_log.nominalOutput = float(-raw_output_torque)
    adaptive_log.assistOutput = float(feedback_correction)
    adaptive_log.biasOutput = float(speed_result.trim_lateral_accel)
    adaptive_log.responseDeficit = float(pid_log.error)
    adaptive_log.learningFrozen = bool(sample_update.reject_reason != TorqueV4LearnerRejectReason.NONE)
    adaptive_log.freezeReason = int(sample_update.reject_reason)
    adaptive_log.blockReason = int(sample_update.reject_reason)
    adaptive_log.shapingActive = bool(governor_result.reason != TorqueV4GovernorReason.NONE)
    adaptive_log.shapingReason = int(governor_result.reason)
    adaptive_log.shapingConfidence = float(speed_result.speed_aware_confidence)
    adaptive_log.unshapedOutput = float(-raw_output_torque)
    adaptive_log.outputCap = float(governor_result.output_cap)
    adaptive_log.modelMode = 0
    adaptive_log.modelConfidence = float(speed_result.speed_aware_confidence)
    adaptive_log.authorityBand = 0
    adaptive_log.authorityScale = float(speed_result.response_scale)
    adaptive_log.fallbackActive = False
    adaptive_log.learnedLatAccelFactor = float(speed_result.effective_lat_accel_factor)
    adaptive_log.learnedFriction = float(self.torque_params.friction)
    adaptive_log.learnedLatAccelOffset = float(speed_result.effective_lat_accel_offset)
    adaptive_log.learnedResponseDelay = float(speed_result.response_delay)
    adaptive_log.residualError = float(sample_update.residual_error)
    adaptive_log.sampleAccepted = bool(sample_update.sample_accepted)
    adaptive_log.sampleRejectReason = int(sample_update.reject_reason)
    adaptive_log.disturbanceState = 0
    adaptive_log.disturbanceReason = 0
    adaptive_log.disturbanceConfidence = 0.0
    adaptive_log.steerLimitValid = bool(steer_limit_feedback.valid)
    adaptive_log.steerLimitLimited = bool(steer_limit_feedback.limited)
    adaptive_log.steerLimitReason = int(steer_limit_feedback.reason)
    adaptive_log.steerLimitRequested = float(steer_limit_feedback.requested)
    adaptive_log.steerLimitApplied = float(steer_limit_feedback.applied)
    adaptive_log.steerLimitError = float(steer_limit_feedback.error)
    adaptive_log.steerLimitSameDirection = bool(steer_limit_same_direction)
    adaptive_log.steerLimitUnwind = bool(steer_limit_unwind)
    adaptive_log.rawTargetLateralAccel = float(target.raw_lateral_accel)
    adaptive_log.delayLeadLateralAccel = float(target.delay_lead_lateral_accel)
    adaptive_log.feedbackCorrection = float(feedback_correction)
    adaptive_log.trimCorrection = float(speed_result.trim_lateral_accel)
    adaptive_log.learnerResponseScale = float(speed_result.response_scale)
    adaptive_log.governorReason = int(governor_result.reason)
    adaptive_log.actualLateralJerk = float(actual_lateral_jerk if _finite(actual_lateral_jerk) else 0.0)
    if self.lateral_demand_profile is not None:
      adaptive_log.demandMode = int(lateral_mode_to_uint8(self.lateral_demand_profile.mode))
      adaptive_log.demandModeConfidence = float(self.lateral_demand_profile.mode_confidence) if _finite(self.lateral_demand_profile.mode_confidence) else 0.0
    adaptive_log.oscillationClassification = int(lateral_oscillation_to_uint8(self._last_oscillation_classification))
    adaptive_log.wobbleActive = bool(self._wobble_active)
    adaptive_log.vehicleBiasEstimate = float(self._last_health_estimate.bias_estimate) if _finite(self._last_health_estimate.bias_estimate) else 0.0
    adaptive_log.vehicleBiasConfidence = float(self._last_health_estimate.bias_confidence) if _finite(self._last_health_estimate.bias_confidence) else 0.0
    adaptive_log.vehicleBiasWarning = bool(self._last_health_estimate.bias_warning)
    adaptive_log.vehicleHealthActive = bool(self._last_health_estimate.learning_active)
    adaptive_log.turnExitMode = int(turn_exit_mode_to_uint8(self._last_turn_exit_mode))
    adaptive_log.previewBoost = float(self._last_turn_exit_preview_boost)
    adaptive_log.earlyReleaseActive = bool(self._last_turn_exit_early_release)
    adaptive_log.wobbleFeedbackGainMult = float(self._last_wobble_response.feedback_gain_multiplier)
    adaptive_log.wobbleDampingGainMult = float(self._last_wobble_response.damping_gain_multiplier)
    # v5.0 route-validation telemetry. Always present on the log
    # so route tooling has a single stable place to read these
    # fields regardless of whether v5 was active this frame.
    adaptive_log.v5Active = bool(self._v5_last_v5_active)
    adaptive_log.previewBoostApplied = float(self._v5_last_preview_applied_value)
    adaptive_log.finalLeadDelta = float(self._v5_last_final_lead_delta)
    adaptive_log.outputSignFlips = int(self._v5_output_sign_flips)
    adaptive_log.straightRoadTorqueFlips = int(self._v5_straight_road_torque_flips)


class LatControlTorqueV41(LatControlTorqueV4):
  VERSION = 41
  UNDER_RESPONSE_RELEASE_HOLD = True
  UNDER_RESPONSE_CATCHUP_ENABLED = True
  UNDER_RESPONSE_CATCHUP_GAIN_BP = V41_UNDER_RESPONSE_CATCHUP_GAIN_BP
  UNDER_RESPONSE_CATCHUP_GAIN_V = V41_UNDER_RESPONSE_CATCHUP_GAIN_V
  UNDER_RESPONSE_CATCHUP_CAP_BP = V41_UNDER_RESPONSE_CATCHUP_CAP_BP
  UNDER_RESPONSE_CATCHUP_CAP_V = V41_UNDER_RESPONSE_CATCHUP_CAP_V
  UNDER_RESPONSE_CATCHUP_MAX_STEERING_RATE_DEG = 80.0
  GOVERNOR_PROFILE = TorqueV4GovernorProfile(
    output_slew_rate_bp=[0.0, 5.0, 10.0, 20.0, 30.0, 40.0],
    output_slew_rate_v=[1.40, 2.00, 3.00, 4.20, 5.00, 5.60],
    sign_change_slew_rate_bp=[0.0, 5.0, 10.0, 20.0, 30.0, 40.0],
    sign_change_slew_rate_v=[0.90, 1.20, 1.80, 2.40, 3.00, 3.40],
    same_direction_limit_cap=0.85,
    same_direction_limit_rate=1.30,
    high_rate_start_deg=80.0,
    high_rate_full_deg=100.0,
    high_rate_min_cap=0.62,
    high_rate_slew_scale=0.70,
    same_direction_limit_rate_bp=[0.0, 10.0, 20.0, 30.0, 40.0],
    same_direction_limit_rate_v=[1.30, 1.30, 2.10, 3.20, 3.60],
    same_direction_decrease_bypass=True,
  )


# 5.0 active deltas. All flags are False in the initial commit, so
# the v5 subclass is bit-equivalent to v4.1. Subsequent commits will
# flip these one at a time, each with its own gate and tests.
V5_PREVIEW_BOOST_CAP_BP = [0.0, 10.0, 20.0, 40.0]
V5_PREVIEW_BOOST_CAP_V = [0.00, 0.08, 0.12, 0.10]


# Bounded turn-exit slew boost. Used by the governor only when the
# v5 context reports turn_exit_active, only on unwind frames, and
# only as a multiplier on the v4.1 slew rate. Safety caps and
# output caps remain at v4.1 values.
TORQUE_V5_TURN_EXIT_SIGN_CHANGE_SLEW_BOOST = 1.20
TORQUE_V5_TURN_EXIT_SAME_DIRECTION_SLEW_BOOST = 1.10

# 5.1 vehicle-bias compensation bounds. ACTIVE_VEHICLE_BIAS_COMPENSATION
# stays False in initial 5.0; flipping the flag in 5.1 applies the
# bias term with a ±V5_VEHICLE_BIAS_INITIAL_CAP envelope, hard-capped
# at ±V5_VEHICLE_BIAS_HARD_CAP. The flag flip must wait for
# route validation per the v5 plan.
V5_VEHICLE_BIAS_INITIAL_CAP = 0.03
V5_VEHICLE_BIAS_HARD_CAP = 0.06
V5_VEHICLE_BIAS_MIN_CONFIDENCE = 0.80


class LatControlTorqueV5(LatControlTorqueV41):
  """Torque 5.0: profile-aware active command shaping.

  Active delta flags (5.0):
    ACTIVE_PROFILE_PREVIEW_LEAD     - profile-aware turn-in preview boost
    ACTIVE_TURN_EXIT_CONTROLLER     - turn-exit controller as source of truth
    ACTIVE_VEHICLE_BIAS_COMPENSATION - learned vehicle-bias applied to command
                                       (5.1 territory; stays False in 5.0)
  """
  VERSION = 50
  ACTIVE_PROFILE_PREVIEW_LEAD = True
  ACTIVE_TURN_EXIT_CONTROLLER = True
  ACTIVE_VEHICLE_BIAS_COMPENSATION = False
  PREVIEW_BOOST_CAP_BP = V5_PREVIEW_BOOST_CAP_BP
  PREVIEW_BOOST_CAP_V = V5_PREVIEW_BOOST_CAP_V
  PREVIEW_MIN_PATH_QUALITY = 0.75
  PREVIEW_MIN_MODE_CONFIDENCE = 0.75

  def _v5_governor_context(self, *, target: TorqueV4Target) -> TorqueV5GovernorContext | None:
    """v5 builds a populated governor context every frame.

    The context lets the governor apply the bounded turn-exit
    unwind slew boost and surface v5 telemetry. The governor's
    safety caps, sign-change slew rates, and output caps stay at
    v4.1 values regardless of the context.
    """
    profile = self.lateral_demand_profile
    profile_available = profile is not None
    demand_mode = getattr(profile, "mode", "straight_stable") if profile is not None else "straight_stable"
    demand_mode_confidence = float(getattr(profile, "mode_confidence", 0.0)) if profile is not None else 0.0
    return TorqueV5GovernorContext(
      profile_available=profile_available,
      demand_mode=demand_mode,
      demand_mode_confidence=demand_mode_confidence,
      preview_active=self._v5_last_preview_active,
      turn_exit_active=self._v5_last_turn_exit_active,
      wobble_active=self._wobble_active,
      v5_active=self._v5_last_v5_active,
      v5_reason=self._v5_last_v5_reason,
    )

  def _v5_record_output_torque_sign_flips(self, output_torque: float) -> None:
    """Maintain the v5 route-validation counters.

    output_sign_flips counts every time the output torque's
    sign changes (with a small deadband to ignore numerical
    noise). straight_road_torque_flips counts only the flips
    that happen while the controller classifies the road as
    straight. Both counters are cumulative since the last reset
    and are surfaced through the adaptive log for route tooling.
    """
    if not _finite(output_torque):
      return
    new_sign = _sign(float(output_torque), 1e-4)
    prev_sign = getattr(self, "_v5_last_output_sign", 0)
    if new_sign != 0 and prev_sign != 0 and new_sign != prev_sign:
      self._v5_output_sign_flips += 1
      if getattr(self, "_last_straight_road", False):
        self._v5_straight_road_torque_flips += 1
    if new_sign != 0:
      self._v5_last_output_sign = new_sign

  def _v5_route_summary(self) -> dict:
    """Snapshot of the v5 route-validation telemetry. Useful for
    tests and route tools; the same values also flow into the
    adaptive log.
    """
    return {
      "v5_active": self._v5_last_v5_active,
      "preview_boost_applied": float(self._v5_last_preview_applied_value),
      "turn_exit_mode": int(turn_exit_mode_to_uint8(self._last_turn_exit_mode)),
      "wobble_active": bool(self._wobble_active),
      "final_lead_delta": float(self._v5_last_final_lead_delta),
      "output_sign_flips": int(self._v5_output_sign_flips),
      "straight_road_torque_flips": int(self._v5_straight_road_torque_flips),
    }

  def _v5_turn_exit_decision(self, target: TorqueV4Target, *, active: bool,
                             CS, curvature_limited: bool) -> None:
    """Pre-target turn-exit decision. v5 hooks this so the
    decision is computed before target building. ACTIVE_TURN_EXIT_CONTROLLER
    is True in v5.0, so this method drives the seam and the
    post-command path uses the cached decision. The seam is
    still a no-op when ACTIVE_TURN_EXIT_CONTROLLER is False
    (e.g. unit tests that explicitly opt out).
    """
    if not self.ACTIVE_TURN_EXIT_CONTROLLER:
      self._v5_turn_exit_decided = False
      self._v5_cached_turn_exit_decision = None
      return None
    # 5.0 v5 contract: missing profile = full 4.1 fallback. The
    # preview gate already requires a profile to allow a boost;
    # the turn-exit source-of-truth must do the same so a frame
    # without a profile produces a target that exactly matches
    # the v4.1 base.
    if self.lateral_demand_profile is None:
      self._v5_turn_exit_decided = False
      self._v5_cached_turn_exit_decision = None
      return None
    # Path enabled by a later commit. Not exercised in the skeleton.
    cs_v_ego = getattr(CS, "vEgo", 0.0) if CS is not None else 0.0
    cs_steering_pressed = bool(getattr(CS, "steeringPressed", False)) if CS is not None else False
    self._v5_turn_exit_decided = True
    self._v5_cached_turn_exit_decision = self.turn_exit_controller.update(
      target=target.raw_lateral_accel,
      profile=self.lateral_demand_profile,
      active=bool(active),
      v_ego=cs_v_ego,
      path_quality=_finite_float(getattr(self.processed_lateral_demand, "path_quality", 1.0)) or 1.0,
      lane_change_active=bool(getattr(self.processed_lateral_demand, "lane_change_shaping_active", False)),
      steering_pressed=cs_steering_pressed,
      curvature_limited=curvature_limited,
      saturated=bool(self._previous_saturated),
    )
    return None

  def _v5_record_turn_exit_telemetry(self, *, target: TorqueV4Target, active: bool,
                                     CS, curvature_limited: bool, saturated: bool) -> TurnExitDecision:
    """Post-command turn-exit telemetry. v5 with the active flag
    on uses the cached pre-target decision; otherwise it falls
    through to the v4 base which calls turn_exit_controller.update.
    The fallback is what keeps parity in the skeleton commit.
    """
    if self.ACTIVE_TURN_EXIT_CONTROLLER and self._v5_turn_exit_decided and self._v5_cached_turn_exit_decision is not None:
      # Clear the latch so the next frame starts fresh.
      self._v5_turn_exit_decided = False
      return self._v5_cached_turn_exit_decision
    return super()._v5_record_turn_exit_telemetry(
      target=target, active=active, CS=CS, curvature_limited=curvature_limited, saturated=saturated,
    )

  def _build_target(self, desired_curvature: float, v_ego: float, speed_result: TorqueV4SpeedModelResult,
                    invalid: bool, recenter: TorqueV4RecenterMode | None = None,
                    curvature_limited: bool = False, cs=None,
                    active: bool = True,
                    steer_limited_by_safety: bool = False) -> TorqueV4Target:
    """v5 build target orchestrator.

    Calls _build_target_base for the v4-compatible math, runs
    the pre-target turn-exit seam, gates the preview boost, then
    delegates to _build_v5_target to apply the v5 shaping. The
    orchestrator stays thin; the v5 shaping math lives in
    _build_v5_target and is unit-testable on its own.

    `active` and `steer_limited_by_safety` are propagated into
    the v5 gates so an inactive or steer-limited controller does
    not silently produce v5 target/telemetry pollution.
    """
    base = self._build_target_base(desired_curvature, v_ego, speed_result, invalid, recenter)
    v5_effective_active = bool(active and not invalid)
    if not self.ACTIVE_TURN_EXIT_CONTROLLER or not v5_effective_active:
      self._clear_v5_telemetry()
      self._v5_last_final_lead_delta = float(base.lead_delta)
      self._v5_last_preview_applied_value = 0.0
      return base

    # Pre-target decision seam. Stores self._v5_cached_turn_exit_decision
    # or leaves it None if the controller was inactive.
    self._v5_turn_exit_decision(
      base, active=v5_effective_active, CS=cs, curvature_limited=curvature_limited,
    )
    decision = self._v5_cached_turn_exit_decision
    if decision is None or not self._v5_turn_exit_decided:
      self._clear_v5_telemetry()
      self._v5_last_final_lead_delta = float(base.lead_delta)
      self._v5_last_preview_applied_value = 0.0
      return base

    # Gate the preview boost before handing the decision off.
    preview_allowed, preview_reason = self._v5_preview_allowed(
      active=v5_effective_active, invalid=False, CS=cs, v_ego=v_ego,
      profile=self.lateral_demand_profile, demand=self.processed_lateral_demand,
      curvature_limited=curvature_limited, saturated=self._previous_saturated,
      steer_limited_by_safety=steer_limited_by_safety,
    )
    return self._build_v5_target(
      base=base, v_ego=v_ego, speed_result=speed_result, decision=decision,
      preview_allowed=preview_allowed, preview_reason=preview_reason,
    )

  def _clear_v5_telemetry(self) -> None:
    """Reset v5 telemetry flags to their neutral state. Called on
    early-return paths where v5 shaping did not run.
    """
    self._v5_last_preview_active = False
    self._v5_last_turn_exit_active = False
    self._v5_last_v5_active = False
    self._v5_last_v5_reason = ""

  def _build_v5_target(self, *, base: TorqueV4Target,
                       v_ego: float,
                       speed_result: TorqueV4SpeedModelResult,
                       decision: TurnExitDecision,
                       preview_allowed: bool,
                       preview_reason: str) -> TorqueV5Target:
    """Pure v5 target builder. Applies the turn-exit decision's
    lead gain / cap multipliers, the early-release guard, and the
    preview boost. The result is a TorqueV5Target that carries
    the v5 metadata fields so downstream code can see exactly
    which shaping the v5 path applied.

    The v4 base target math is not re-run here; the orchestrator
    passed in the already-built base. The lead math is the same
    as the v4 base, with the decision's multipliers applied and
    lead_delta re-clipped to the scaled cap.
    """
    # Start from the base target's lead math. v5 must not
    # reintroduce a lead delta that the base immediate early-
    # release guard already zeroed on a turn-exit collapse frame.
    lead_gain = base.lead_gain
    lead_delta_cap = base.lead_delta_cap
    lead_delta = base.lead_delta
    # Apply lead gain / cap multipliers from the decision.
    lead_gain = lead_gain * float(decision.lead_gain_multiplier)
    lead_delta_cap = lead_delta_cap * float(decision.lead_delta_cap_multiplier)
    # Only re-derive lead_delta from the scaled gain/cap when the
    # base did not already early-release (base.lead_delta == 0).
    # If the base zeroed it, v5 must preserve that. This keeps
    # "turn-exit feels immediate" intact on the first collapse
    # frame instead of v5 reintroducing a nonzero lead.
    if base.lead_delta != 0.0:
      lead_delta = _clip(
        base.target_rate * speed_result.response_delay * lead_gain,
        -lead_delta_cap, lead_delta_cap,
      )
    # Early release. The decision's classification is the
    # source of truth; no persistence floor is required for
    # the immediate early-release guard. The persistence check
    # was an over-conservative gate that let v5 undo the base
    # guard for the first RECENTER_PERSISTENCE_FRAMES frames.
    early_release_active = bool(decision.early_release_lead_zero)
    if early_release_active:
      lead_delta = 0.0

    # Preview boost. Applied only when the gate passes AND the
    # active flag is on. The boost math is delegated to
    # _v5_preview_boost so it stays unit-testable.
    preview_boost_computed, preview_boost_applied = self._v5_preview_boost(
      v_ego=v_ego, decision=decision, allowed=preview_allowed,
    )
    # Re-clip after the boost so the final lead delta is bounded
    # by the scaled cap, never by a wider envelope.
    if preview_boost_applied != 0.0:
      lead_delta = _clip(lead_delta + preview_boost_applied, -lead_delta_cap, lead_delta_cap)
    delay_lead = base.raw_lateral_accel + lead_delta
    # Telemetry flags for the governor context and the route
    # summary. v5_active means "v5 actually changed behavior",
    # not "v5 path ran". Compute the delta against the base
    # target so route validation isn't polluted by frames
    # where v5 ran but the final target equals the v4.1 base.
    turn_exit_changed = (
      lead_gain != base.lead_gain
      or lead_delta_cap != base.lead_delta_cap
      or early_release_active
    )
    preview_changed = bool(preview_boost_applied != 0.0)
    lead_changed = abs(lead_delta - base.lead_delta) > 1e-9
    v5_active = bool(turn_exit_changed or preview_changed or lead_changed)
    if preview_changed:
      v5_reason = "preview_boost_applied"
    elif turn_exit_changed:
      v5_reason = "turn_exit_source_of_truth"
    else:
      v5_reason = "inactive"
    self._v5_last_preview_active = bool(preview_boost_applied != 0.0)
    self._v5_last_turn_exit_active = bool(early_release_active or decision.mode in ("turn_exit", "early_release"))
    self._v5_last_v5_active = v5_active
    self._v5_last_v5_reason = v5_reason
    self._v5_last_final_lead_delta = float(lead_delta)
    self._v5_last_preview_applied_value = float(preview_boost_applied)
    return TorqueV5Target(
      raw_lateral_accel=base.raw_lateral_accel,
      target_rate=base.target_rate,
      delay_lead_lateral_accel=delay_lead,
      lead_delta=lead_delta,
      lead_gain=lead_gain,
      lead_delta_cap=lead_delta_cap,
      base_lead_delta=base.lead_delta,
      preview_boost_computed=preview_boost_computed,
      preview_boost_applied=preview_boost_applied,
      preview_reason=preview_reason,
      turn_exit_lead_gain_multiplier=float(decision.lead_gain_multiplier),
      turn_exit_lead_delta_cap_multiplier=float(decision.lead_delta_cap_multiplier),
      turn_exit_early_release=early_release_active,
      v5_active=v5_active,
      v5_reason=v5_reason,
    )

  def _v5_preview_allowed(self, *, active: bool, invalid: bool, CS,
                          v_ego: float = 0.0,
                          profile: "LateralDemandProfile | None",
                          demand: "ProcessedLateralDemand | None",
                          curvature_limited: bool, saturated: bool,
                          steer_limited_by_safety: bool) -> tuple[bool, str]:
    """Pure gate: returns (allowed, reason). No boost math, no
    telemetry side effects. The reason string is one of a fixed
    set so route tooling can group disable causes.
    """
    if not active:
      return False, "inactive"
    if invalid:
      return False, "invalid"
    if profile is None:
      return False, "missing_profile"
    if demand is None:
      return False, "missing_demand"
    if getattr(profile, "mode", None) != "turn_in":
      return False, "wrong_mode"
    if float(getattr(profile, "mode_confidence", 0.0)) < self.PREVIEW_MIN_MODE_CONFIDENCE:
      return False, "low_mode_confidence"
    if float(getattr(profile, "path_quality", 1.0)) < self.PREVIEW_MIN_PATH_QUALITY:
      return False, "low_path_quality"
    if getattr(profile, "path_reason", LEARN_PATH_REASON_OK) != LEARN_PATH_REASON_OK:
      return False, "bad_path_reason"
    if getattr(profile, "demand_source", DEMAND_SOURCE_MODEL_PATH) != DEMAND_SOURCE_MODEL_PATH:
      return False, "non_model_profile"
    if getattr(demand, "demand_source", DEMAND_SOURCE_MODEL_PATH) != DEMAND_SOURCE_MODEL_PATH:
      return False, "non_model_demand"
    if bool(getattr(demand, "lane_change_shaping_active", False)):
      return False, "lane_change"
    # Fail closed: a missing or non-finite lane_change_blend is
    # treated as if a lane change is in progress. Better to skip
    # the boost than to apply it without the missing input.
    lcb = _finite_float(getattr(demand, "lane_change_blend", None))
    if lcb is None:
      return False, "lane_change"
    if abs(lcb) > 1e-3:
      return False, "lane_change"
    if curvature_limited:
      return False, "curvature_limited"
    if saturated:
      return False, "saturated"
    if steer_limited_by_safety:
      return False, "steer_limited"
    cs_steering_pressed = bool(getattr(CS, "steeringPressed", False)) if CS is not None else False
    if cs_steering_pressed:
      return False, "steering_pressed"
    if self._wobble_active:
      return False, "wobble_active"
    cs_v_ego = float(getattr(CS, "vEgo", 0.0)) if CS is not None else 0.0
    effective_v_ego = cs_v_ego if cs_v_ego > 0.0 else float(v_ego)
    if effective_v_ego < 5.0:
      return False, "low_speed"
    return True, "allowed"

  def _v5_preview_boost(self, *, v_ego: float, decision,
                         allowed: bool) -> tuple[float, float]:
    """Compute preview_boost_computed and preview_boost_applied.

    Per the v5 plan:
      preview_boost_computed = decision.preview_boost  (raw)
      preview_boost_applied  = 0.0 if the active flag is off
                              else the speed-capped value
                              when the gate is allowed

    The cap is on the applied value, not the computed value:
    computed is the raw decision so telemetry can show the
    unbounded intent, while applied is what actually lands
    in lead_delta.
    """
    raw_boost = float(decision.preview_boost) if decision is not None else 0.0
    cap = _interp(v_ego, self.PREVIEW_BOOST_CAP_BP, self.PREVIEW_BOOST_CAP_V)
    capped = _clip(raw_boost, -cap, cap)
    applied = capped if (self.ACTIVE_PROFILE_PREVIEW_LEAD and allowed) else 0.0
    return float(raw_boost), float(applied)

LatControlTorque = LatControlTorqueV4
