import ast
import math
from dataclasses import dataclass
from enum import IntEnum, IntFlag

import numpy as np

from cereal import log
from opendbc.car.lateral import get_friction
from openpilot.selfdrive.controls.lib.lateral_accel import roll_lateral_accel
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

OUTPUT_SLEW_RATE_BP = [0.0, 3.0, 10.0, 20.0, 30.0, 40.0]
OUTPUT_SLEW_RATE_V = [0.80, 1.10, 2.40, 3.60, 4.00, 4.00]
SIGN_CHANGE_SLEW_RATE_BP = [0.0, 3.0, 10.0, 20.0, 30.0, 40.0]
SIGN_CHANGE_SLEW_RATE_V = [0.40, 0.60, 1.40, 2.00, 2.20, 2.00]
OVERRIDE_RELEASE_RATE = 4.0
SAME_DIRECTION_LIMIT_CAP = 0.72
SAME_DIRECTION_LIMIT_RATE = 1.20
HIGH_RATE_START_DEG = 70.0
HIGH_RATE_FULL_DEG = 100.0
HIGH_RATE_MIN_CAP = 0.60
HIGH_RATE_SLEW_SCALE = 0.65
STALE_ACTUATOR_ERROR_THRESHOLD = 0.15
STALE_ACTUATOR_CAP = 0.35

LEARN_MIN_SPEED = 10.0
LEARN_MAX_SPEED = 35.0
LEARN_MIN_TARGET = 0.08
LEARN_SIGN_THRESHOLD = 0.05
LEARN_MAX_JERK = 8.0
LEARN_MAX_GOVERNOR_REASON = 0

SPEED_BUCKET_CENTERS = [5.0, 15.0, 25.0, 35.0, 45.0]


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
  return all(math.isfinite(float(value)) for value in values)


def _clip(value: float, lower: float, upper: float) -> float:
  return float(np.clip(value, lower, upper))


def _interp(value: float, bp: list[float], vals: list[float]) -> float:
  return float(np.interp(float(value), bp, vals))


def _sign(value: float, threshold: float = 0.0) -> int:
  return 1 if value > threshold else (-1 if value < -threshold else 0)


def _approach(value: float, target: float, step: float) -> float:
  if target > value:
    return min(target, value + step)
  return max(target, value - step)


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
  actual_lateral_jerk = float(v_ego) ** 2 * dk_dangle * float(steering_rate_rad_s)
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
class TorqueV4Observation:
  active: bool
  v_ego: float
  steering_pressed: bool
  steer_limited_by_safety: bool
  curvature_limited: bool
  saturated: bool
  target_lateral_accel: float
  target_lateral_accel_rate: float
  actual_lateral_accel: float
  actual_lateral_jerk: float
  measurement_rate: float
  finite: bool


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


class TorqueV4SpeedModel:
  """Speed-aware corrections stay bounded; feedforward remains lateral-accel based."""

  def update(self, v_ego: float, torque_params, speed_aware_params: dict | None,
             adaptation: "TorqueV4SessionAdaptation") -> TorqueV4SpeedModelResult:
    response_delay = _clip(adaptation.response_delay, RESPONSE_DELAY_MIN, RESPONSE_DELAY_MAX)
    lead_gain = _interp(v_ego, LEAD_GAIN_BP, LEAD_GAIN_V)
    lead_delta_cap = _interp(v_ego, LEAD_DELTA_CAP_BP, LEAD_DELTA_CAP_V)
    feedback_gain = _interp(v_ego, FEEDBACK_GAIN_BP, FEEDBACK_GAIN_V)
    damping_gain = _interp(v_ego, DAMPING_GAIN_BP, DAMPING_GAIN_V)
    breakaway_scale = _interp(v_ego, BREAKAWAY_SCALE_BP, BREAKAWAY_SCALE_V)
    output_slew_rate = _interp(v_ego, OUTPUT_SLEW_RATE_BP, OUTPUT_SLEW_RATE_V)
    sign_change_slew_rate = _interp(v_ego, SIGN_CHANGE_SLEW_RATE_BP, SIGN_CHANGE_SLEW_RATE_V)

    speed_factor, speed_offset, confidence = self._speed_aware_values(v_ego, torque_params, speed_aware_params)
    global_factor = max(float(torque_params.latAccelFactor), 1e-3)
    speed_response_scale = _clip(global_factor / max(speed_factor, 1e-3), RESPONSE_SCALE_MIN, RESPONSE_SCALE_MAX)
    response_scale = _clip(adaptation.response_scale * speed_response_scale, RESPONSE_SCALE_MIN, RESPONSE_SCALE_MAX)
    trim_lateral_accel = _clip(adaptation.trim_lateral_accel + (float(torque_params.latAccelOffset) - speed_offset),
                              TRIM_LAT_ACCEL_MIN, TRIM_LAT_ACCEL_MAX)
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
    )

  def _speed_aware_values(self, v_ego: float, torque_params, speed_aware_params: dict | None) -> tuple[float, float, float]:
    global_factor = max(float(torque_params.latAccelFactor), 1e-3)
    global_offset = float(torque_params.latAccelOffset)
    if v_ego < 10.0 or not speed_aware_params:
      return global_factor, global_offset, 0.0

    label = self._label_for_speed(v_ego)
    bucket = speed_aware_params.get(label)
    if not self._valid_bucket(bucket, global_factor):
      return global_factor, global_offset, 0.0

    local_factor = self._interpolate_factor(v_ego, speed_aware_params, global_factor)
    local_offset = float(bucket[1])
    if not _finite(local_factor, local_offset):
      return global_factor, global_offset, 0.0

    confidence = 0.30 if v_ego < 30.0 else 0.22
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
    residual = observation.target_lateral_accel - observation.actual_lateral_accel if observation.finite else 0.0
    if reject_reason != TorqueV4LearnerRejectReason.NONE:
      return TorqueV4AdaptationUpdate(False, reject_reason, residual)

    if abs(observation.actual_lateral_accel) > LEARN_SIGN_THRESHOLD:
      target_scale = _clip(abs(observation.target_lateral_accel) / max(abs(observation.actual_lateral_accel), 1e-3),
                           RESPONSE_SCALE_MIN, RESPONSE_SCALE_MAX)
      self.response_scale = _clip(self.response_scale + 0.002 * (target_scale - self.response_scale),
                                  RESPONSE_SCALE_MIN, RESPONSE_SCALE_MAX)

    if abs(observation.target_lateral_accel_rate) < 0.12:
      self.trim_lateral_accel = _clip(self.trim_lateral_accel + 0.0005 * residual,
                                      TRIM_LAT_ACCEL_MIN, TRIM_LAT_ACCEL_MAX)

    return TorqueV4AdaptationUpdate(True, TorqueV4LearnerRejectReason.NONE, residual)

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
    if abs(observation.target_lateral_accel) < LEARN_MIN_TARGET:
      reason |= TorqueV4LearnerRejectReason.LOW_DEMAND
    if not observation.finite:
      reason |= TorqueV4LearnerRejectReason.NON_FINITE
    if abs(observation.actual_lateral_jerk) > LEARN_MAX_JERK:
      reason |= TorqueV4LearnerRejectReason.HIGH_JERK
    target_sign = _sign(observation.target_lateral_accel, LEARN_SIGN_THRESHOLD)
    actual_sign = _sign(observation.actual_lateral_accel, LEARN_SIGN_THRESHOLD)
    if target_sign != 0 and actual_sign != 0 and target_sign != actual_sign:
      reason |= TorqueV4LearnerRejectReason.SIGN_CONFLICT
    if observation.v_ego < LEARN_MIN_SPEED or observation.v_ego > LEARN_MAX_SPEED:
      reason |= TorqueV4LearnerRejectReason.SPEED_RANGE
    if governor_reason != TorqueV4GovernorReason(LEARN_MAX_GOVERNOR_REASON):
      reason |= TorqueV4LearnerRejectReason.GOVERNOR_ACTIVE
    return reason


class TorqueV4OutputGovernor:
  def __init__(self, dt: float):
    self.dt = max(float(dt), 1e-3)
    self.previous_output = 0.0

  def reset(self) -> None:
    self.previous_output = 0.0

  def update(self, *, active: bool, v_ego: float, steering_pressed: bool, steering_rate_deg: float,
             same_direction_limit: bool, steer_limit_unwind: bool, actuator_mismatch: bool, actuator_error: float,
             raw_output_torque: float, max_output: float, speed_model: TorqueV4SpeedModelResult) -> TorqueV4GovernorResult:
    reason = TorqueV4GovernorReason.NONE
    if not active:
      self.reset()
      return TorqueV4GovernorResult(0.0, reason, max_output)
    if not _finite(v_ego, steering_rate_deg, actuator_error, raw_output_torque, max_output) or max_output <= 0.0:
      self.reset()
      return TorqueV4GovernorResult(0.0, TorqueV4GovernorReason.INVALID, 0.0)

    if steering_pressed:
      output = _approach(self.previous_output, 0.0, OVERRIDE_RELEASE_RATE * self.dt)
      self.previous_output = output
      return TorqueV4GovernorResult(output, TorqueV4GovernorReason.DRIVER_OVERRIDE, max_output)

    output_cap = max_output
    high_rate_blend = _clip((abs(steering_rate_deg) - HIGH_RATE_START_DEG) / max(HIGH_RATE_FULL_DEG - HIGH_RATE_START_DEG, 1e-3), 0.0, 1.0)
    if high_rate_blend > 0.0:
      output_cap = min(output_cap, max_output * (1.0 + high_rate_blend * (HIGH_RATE_MIN_CAP - 1.0)))
      reason |= TorqueV4GovernorReason.HIGH_STEERING_RATE
    if same_direction_limit and not steer_limit_unwind:
      output_cap = min(output_cap, max_output * SAME_DIRECTION_LIMIT_CAP)
      reason |= TorqueV4GovernorReason.SAME_DIRECTION_LIMIT
    stale_actuator_mismatch = actuator_mismatch and same_direction_limit and not steer_limit_unwind and abs(actuator_error) > STALE_ACTUATOR_ERROR_THRESHOLD
    if stale_actuator_mismatch:
      output_cap = min(output_cap, max_output * STALE_ACTUATOR_CAP)
      reason |= TorqueV4GovernorReason.STALE_ACTUATOR_MISMATCH

    clipped = _clip(raw_output_torque, -output_cap, output_cap)
    if abs(clipped - raw_output_torque) > 1e-6:
      reason |= TorqueV4GovernorReason.CLIPPED

    previous_sign = _sign(self.previous_output, 1e-4)
    target_sign = _sign(clipped, 1e-4)
    sign_change = previous_sign != 0 and target_sign != 0 and previous_sign != target_sign
    slew_rate = speed_model.sign_change_slew_rate if sign_change else speed_model.output_slew_rate
    if sign_change:
      reason |= TorqueV4GovernorReason.SIGN_CHANGE_LIMITED
    if high_rate_blend > 0.0:
      slew_rate *= HIGH_RATE_SLEW_SCALE
    if same_direction_limit and not steer_limit_unwind:
      slew_rate = min(slew_rate, SAME_DIRECTION_LIMIT_RATE)
    if stale_actuator_mismatch:
      slew_rate = min(slew_rate, STALE_ACTUATOR_CAP)

    output = _approach(self.previous_output, clipped, slew_rate * self.dt)
    if abs(output - clipped) > 1e-6:
      reason |= TorqueV4GovernorReason.SLEW_LIMITED
    self.previous_output = output
    return TorqueV4GovernorResult(output, reason, output_cap)


class LatControlTorqueV4(LatControl):
  CONTROL_STATE = "torque"

  def __init__(self, CP, CP_SP, CI, dt):
    super().__init__(CP, CP_SP, CI, dt)
    del CP_SP
    if CP.lateralTuning.which() != "torque":
      raise ValueError("Torque v4 requires native torque lateral tuning")
    self.CP = CP
    self.torque_params = CP.lateralTuning.torque.as_builder()
    self.torque_from_lateral_accel = CI.torque_from_lateral_accel()
    self.steering_angle_deadzone_deg = self.torque_params.steeringAngleDeadzoneDeg
    initial_delay = max(float(getattr(CP, "steerActuatorDelay", 0.2)), self.dt)
    self.speed_model = TorqueV4SpeedModel()
    self.session_adaptation = TorqueV4SessionAdaptation(initial_delay)
    self.governor = TorqueV4OutputGovernor(self.dt)
    self.speed_aware_params = None
    self.previous_target_lateral_accel = 0.0
    self.previous_measurement = 0.0

  def update_live_torque_params(self, latAccelFactor, latAccelOffset, friction) -> None:
    self.torque_params.latAccelFactor = latAccelFactor
    self.torque_params.latAccelOffset = latAccelOffset
    self.torque_params.friction = friction

  def update_speed_aware_params(self, params_str) -> None:
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

  def reset(self) -> None:
    super().reset()
    self.governor.reset()
    self.session_adaptation.reset()
    self.previous_target_lateral_accel = 0.0
    self.previous_measurement = 0.0

  def update(self, active, CS, VM, params, steer_limited_by_safety, desired_curvature, calibrated_pose, curvature_limited, lat_delay):
    # Torque v4 follows only the processed controller-facing curvature passed in by controlsd.
    del calibrated_pose
    if _finite(lat_delay):
      self.update_lateral_lag(lat_delay)

    pid_log = log.ControlsState.LateralTorqueState.new_message()
    pid_log.version = VERSION

    input_invalid = not _finite(desired_curvature, CS.vEgo, CS.steeringAngleDeg, CS.steeringRateDeg,
                                params.angleOffsetDeg, params.roll)
    speed_result = self.speed_model.update(CS.vEgo if _finite(CS.vEgo) else 0.0, self.torque_params,
                                           self.speed_aware_params, self.session_adaptation)
    target = self._build_target(0.0 if input_invalid else desired_curvature, CS.vEgo, speed_result, input_invalid)
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
    measurement_rate = 0.0 if input_invalid else (actual_lateral_accel - self.previous_measurement) / self.dt
    self.previous_measurement = actual_lateral_accel

    roll_compensation = roll_lateral_accel(params.roll) if not input_invalid else 0.0
    lateral_accel_deadzone = 0.0
    if not input_invalid:
      curvature_deadzone = abs(VM.calc_curvature(math.radians(self.steering_angle_deadzone_deg), CS.vEgo, 0.0))
      lateral_accel_deadzone = curvature_deadzone * CS.vEgo ** 2

    control_error = target.delay_lead_lateral_accel - actual_lateral_accel
    feedback_correction = speed_result.feedback_gain * control_error
    damping_correction = -speed_result.damping_gain * measurement_rate
    breakaway_compensation = self._breakaway_lateral_accel(control_error, lateral_accel_deadzone, target.raw_lateral_accel,
                                                           actual_lateral_accel, speed_result.breakaway_scale)
    command_lateral_accel = (
      target.delay_lead_lateral_accel
      - roll_compensation
      - float(self.torque_params.latAccelOffset)
      + speed_result.response_scale * feedback_correction
      + damping_correction
      + speed_result.trim_lateral_accel
      + breakaway_compensation
    )
    invalid = input_invalid or not _finite(command_lateral_accel, actual_lateral_accel, actual_lateral_jerk, measurement_rate)
    raw_output_torque = 0.0 if invalid else self.torque_from_lateral_accel(command_lateral_accel, self.torque_params)
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
    )
    if invalid:
      governor_result = TorqueV4GovernorResult(0.0, governor_result.reason | TorqueV4GovernorReason.INVALID, governor_result.output_cap)
    output_torque = governor_result.output_torque
    saturated = self.steer_max - abs(output_torque) < 1e-3 or bool(governor_result.reason & TorqueV4GovernorReason.CLIPPED)

    observation = TorqueV4Observation(
      active=active and not invalid,
      v_ego=CS.vEgo,
      steering_pressed=CS.steeringPressed,
      steer_limited_by_safety=same_direction_limit,
      curvature_limited=curvature_limited,
      saturated=saturated,
      target_lateral_accel=target.raw_lateral_accel,
      target_lateral_accel_rate=target.target_rate,
      actual_lateral_accel=actual_lateral_accel,
      actual_lateral_jerk=actual_lateral_jerk,
      measurement_rate=measurement_rate,
      finite=not invalid,
    )
    sample_update = self.session_adaptation.update(observation, governor_result.reason)

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

  def _build_target(self, desired_curvature: float, v_ego: float, speed_result: TorqueV4SpeedModelResult,
                    invalid: bool) -> TorqueV4Target:
    raw_target = 0.0 if invalid else desired_curvature * v_ego ** 2
    target_rate = 0.0 if invalid else (raw_target - self.previous_target_lateral_accel) / self.dt
    self.previous_target_lateral_accel = raw_target
    lead_delta = _clip(target_rate * speed_result.response_delay * speed_result.lead_gain,
                       -speed_result.lead_delta_cap, speed_result.lead_delta_cap)
    return TorqueV4Target(raw_target, target_rate, raw_target + lead_delta, lead_delta,
                          speed_result.lead_gain, speed_result.lead_delta_cap)

  def _breakaway_lateral_accel(self, error: float, lateral_accel_deadzone: float, target: float, measurement: float,
                               breakaway_scale: float) -> float:
    demand = max(abs(target), abs(measurement), abs(error))
    scale = _clip(demand / BREAKAWAY_FULL_DEMAND, 0.0, breakaway_scale)
    if _sign(target, LEARN_SIGN_THRESHOLD) != 0 and _sign(measurement, LEARN_SIGN_THRESHOLD) != 0 and _sign(target) != _sign(measurement):
      scale *= 0.5
    return get_friction(error * scale, lateral_accel_deadzone, FRICTION_THRESHOLD, self.torque_params) * scale

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
    adaptive_log.learnedLatAccelFactor = float(self.torque_params.latAccelFactor)
    adaptive_log.learnedFriction = float(self.torque_params.friction)
    adaptive_log.learnedLatAccelOffset = float(self.torque_params.latAccelOffset)
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


LatControlTorque = LatControlTorqueV4
