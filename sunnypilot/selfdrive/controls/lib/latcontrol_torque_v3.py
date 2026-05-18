import math
from dataclasses import dataclass
from enum import IntEnum, IntFlag

import numpy as np

from cereal import log
from opendbc.car.lateral import get_friction
from openpilot.common.constants import ACCELERATION_DUE_TO_GRAVITY
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.selfdrive.controls.lib.latcontrol import LatControl
from openpilot.sunnypilot.selfdrive.controls.lib.latcontrol_torque_ext import LatControlTorqueExt
from openpilot.sunnypilot.selfdrive.controls.lib.steering_actuator_feedback import classify_steering_limit_direction


VERSION = 3
FRICTION_THRESHOLD = 0.3
LAT_ACCEL_REQUEST_BUFFER_SECONDS = 1.0

MEASUREMENT_FILTER_CUTOFF_HZ = 1.2
MEASUREMENT_FILTER_MIN_SPEED = 0.1
MEASUREMENT_FILTER_MAX_RAW_ERROR = 1.0
MEASUREMENT_FILTER_MAX_PREDICTIVE_JERK = 5.0
MEASUREMENT_FILTER_IMPLAUSIBLE_JERK = 80.0

LEAD_GAIN_BP = [0.0, 10.0, 20.0, 30.0, 40.0]
LEAD_GAIN_V = [0.45, 0.55, 0.65, 0.58, 0.50]
LEAD_DELTA_CAP_BP = [0.0, 10.0, 20.0, 30.0, 40.0]
LEAD_DELTA_CAP_V = [0.20, 0.35, 0.55, 0.65, 0.70]
FEEDBACK_GAIN_BP = [0.0, 5.0, 10.0, 20.0, 30.0, 40.0]
FEEDBACK_GAIN_V = [0.34, 0.30, 0.24, 0.18, 0.14, 0.12]
DAMPING_GAIN_BP = [0.0, 5.0, 10.0, 20.0, 30.0, 40.0]
DAMPING_GAIN_V = [0.10, 0.09, 0.075, 0.060, 0.045, 0.040]

BREAKAWAY_FULL_DEMAND = 0.40
BREAKAWAY_MAX_SCALE = 0.80
TRIM_MAX_LAT_ACCEL = 0.16

OUTPUT_SLEW_RATE_BP = [0.0, 5.0, 10.0, 20.0, 30.0, 40.0]
OUTPUT_SLEW_RATE_V = [1.40, 2.00, 3.00, 4.20, 5.00, 5.60]
SIGN_CHANGE_SLEW_RATE_BP = [0.0, 5.0, 10.0, 20.0, 30.0, 40.0]
SIGN_CHANGE_SLEW_RATE_V = [0.90, 1.20, 1.80, 2.40, 3.00, 3.40]
OVERRIDE_RELEASE_RATE = 4.0
SAME_DIRECTION_LIMIT_RATE = 1.3
SAME_DIRECTION_LIMIT_CAP = 0.72
TOYOTA_HIGH_RATE_START_DEG = 80.0
TOYOTA_HIGH_RATE_FULL_DEG = 100.0
TOYOTA_HIGH_RATE_MIN_CAP = 0.62
TOYOTA_HIGH_RATE_SLEW_SCALE = 0.70

LEARN_MIN_COMMAND = 0.035
LEARN_SIGN_THRESHOLD = 0.05
LEARN_MAX_JERK = 8.0


class V3LearnerRejectReason(IntFlag):
  NONE = 0
  INACTIVE = 1 << 0
  STEERING_PRESSED = 1 << 1
  STEER_LIMITED = 1 << 2
  CURVATURE_LIMITED = 1 << 3
  SATURATED = 1 << 4
  LOW_COMMAND = 1 << 5
  NON_FINITE = 1 << 6
  HIGH_JERK = 1 << 7
  SIGN_CONFLICT = 1 << 8
  LATERAL_MANEUVER = 1 << 9


class V3GovernorReason(IntFlag):
  NONE = 0
  CLIPPED = 1 << 0
  SLEW_LIMITED = 1 << 1
  SIGN_CHANGE_LIMITED = 1 << 2
  DRIVER_OVERRIDE = 1 << 3
  SAME_DIRECTION_LIMIT = 1 << 4
  TOYOTA_HIGH_RATE = 1 << 5
  INVALID = 1 << 6


class V3Phase(IntEnum):
  idle = 0
  engage = 1
  hold = 2
  release = 3


PHASE_TO_CAPNP = {
  V3Phase.idle: log.ControlsState.LateralTorqueState.AdaptiveTorqueState.Phase.idle,
  V3Phase.engage: log.ControlsState.LateralTorqueState.AdaptiveTorqueState.Phase.engage,
  V3Phase.hold: log.ControlsState.LateralTorqueState.AdaptiveTorqueState.Phase.hold,
  V3Phase.release: log.ControlsState.LateralTorqueState.AdaptiveTorqueState.Phase.release,
}


def _finite(*values: float) -> bool:
  return all(math.isfinite(float(value)) for value in values)


def _sign(value: float, threshold: float = 0.0) -> int:
  return 1 if value > threshold else (-1 if value < -threshold else 0)


def _clip(value: float, lower: float, upper: float) -> float:
  return float(np.clip(value, lower, upper))


def _interp(value: float, bp: list[float], vals: list[float]) -> float:
  return float(np.interp(float(value), bp, vals))


def _approach(value: float, target: float, step: float) -> float:
  if target > value:
    return min(target, value + step)
  return max(target, value - step)


@dataclass
class ResponseSampleCorrection:
  response_scale: float
  trim_lat_accel: float
  confidence: float


@dataclass
class ResponseSampleUpdate:
  correction: ResponseSampleCorrection
  sample_accepted: bool
  reject_reason: V3LearnerRejectReason
  residual_error: float


@dataclass
class ResponseSample:
  active: bool
  steering_pressed: bool
  same_direction_limit: bool
  curvature_limited: bool
  saturated: bool
  lateral_maneuver: bool
  v_ego: float
  commanded_torque: float
  target_lateral_accel: float
  target_lateral_accel_rate: float
  actual_lateral_accel: float
  actual_lateral_jerk: float


IDENTITY_RESPONSE_SAMPLE_CORRECTION = ResponseSampleCorrection(1.0, 0.0, 0.0)


def evaluate_response_sample(sample: ResponseSample) -> ResponseSampleUpdate:
  direction = _sign(sample.commanded_torque, LEARN_MIN_COMMAND)
  target_direction = _sign(sample.target_lateral_accel, LEARN_SIGN_THRESHOLD)
  if direction == 0 and target_direction != 0:
    direction = target_direction
  if direction == 0:
    direction = 1

  reject_reason = _sample_reject_reason(sample, direction)
  residual = (
    sample.target_lateral_accel - sample.actual_lateral_accel
    if _finite(sample.target_lateral_accel, sample.actual_lateral_accel) else 0.0
  )
  return ResponseSampleUpdate(IDENTITY_RESPONSE_SAMPLE_CORRECTION, reject_reason == V3LearnerRejectReason.NONE, reject_reason, residual)


def _sample_reject_reason(sample: ResponseSample, command_direction: int) -> V3LearnerRejectReason:
  reason = V3LearnerRejectReason.NONE
  if not sample.active:
    reason |= V3LearnerRejectReason.INACTIVE
  if sample.steering_pressed:
    reason |= V3LearnerRejectReason.STEERING_PRESSED
  if sample.same_direction_limit:
    reason |= V3LearnerRejectReason.STEER_LIMITED
  if sample.curvature_limited:
    reason |= V3LearnerRejectReason.CURVATURE_LIMITED
  if sample.saturated:
    reason |= V3LearnerRejectReason.SATURATED
  if sample.lateral_maneuver:
    reason |= V3LearnerRejectReason.LATERAL_MANEUVER
  if abs(sample.commanded_torque) < LEARN_MIN_COMMAND:
    reason |= V3LearnerRejectReason.LOW_COMMAND
  if not _finite(sample.v_ego, sample.commanded_torque, sample.target_lateral_accel, sample.target_lateral_accel_rate,
                 sample.actual_lateral_accel, sample.actual_lateral_jerk):
    reason |= V3LearnerRejectReason.NON_FINITE
  if abs(sample.actual_lateral_jerk) > LEARN_MAX_JERK:
    reason |= V3LearnerRejectReason.HIGH_JERK
  target_sign = _sign(sample.target_lateral_accel, LEARN_SIGN_THRESHOLD)
  actual_sign = _sign(sample.actual_lateral_accel, LEARN_SIGN_THRESHOLD)
  if target_sign != 0 and actual_sign != 0 and command_direction != 0 and len({target_sign, actual_sign, command_direction}) > 1:
    reason |= V3LearnerRejectReason.SIGN_CONFLICT
  return reason


class LateralAccelMeasurementFilter:
  def __init__(self, dt: float):
    self.dt = max(float(dt), 1e-3)
    self.initialized = False
    self.was_reset = False
    self.value = 0.0

  def reset(self, raw_lateral_accel: float) -> float:
    self.initialized = False
    self.was_reset = True
    self.value = raw_lateral_accel
    return raw_lateral_accel

  def update(self, active: bool, v_ego: float, steering_pressed: bool, raw_lateral_accel: float,
             actual_lateral_jerk: float) -> float:
    self.was_reset = False
    if not _finite(v_ego, raw_lateral_accel, actual_lateral_jerk):
      return self.reset(0.0)
    if not active or steering_pressed or v_ego < MEASUREMENT_FILTER_MIN_SPEED:
      return self.reset(raw_lateral_accel)
    if abs(actual_lateral_jerk) > MEASUREMENT_FILTER_IMPLAUSIBLE_JERK:
      return self.reset(raw_lateral_accel)
    if not self.initialized:
      self.initialized = True
      self.value = raw_lateral_accel
      return raw_lateral_accel

    bounded_jerk = _clip(actual_lateral_jerk, -MEASUREMENT_FILTER_MAX_PREDICTIVE_JERK, MEASUREMENT_FILTER_MAX_PREDICTIVE_JERK)
    predicted = self.value + bounded_jerk * self.dt
    alpha = _clip(2.0 * math.pi * MEASUREMENT_FILTER_CUTOFF_HZ * self.dt, 0.0, 1.0)
    self.value = predicted + alpha * (raw_lateral_accel - predicted)
    self.value = _clip(self.value, raw_lateral_accel - MEASUREMENT_FILTER_MAX_RAW_ERROR,
                       raw_lateral_accel + MEASUREMENT_FILTER_MAX_RAW_ERROR)
    return self.value


@dataclass
class GovernorResult:
  output_torque: float
  reason: V3GovernorReason
  output_cap: float


class TorqueV3OutputGovernor:
  def __init__(self, dt: float):
    self.dt = max(float(dt), 1e-3)
    self.previous_output = 0.0

  def reset(self) -> None:
    self.previous_output = 0.0

  def update(self, active: bool, v_ego: float, steering_pressed: bool, steering_rate_deg: float,
             same_direction_limit: bool, raw_output_torque: float, max_output: float) -> GovernorResult:
    reason = V3GovernorReason.NONE
    if not active:
      self.reset()
      return GovernorResult(0.0, reason, max_output)
    if not _finite(v_ego, steering_rate_deg, raw_output_torque, max_output) or max_output <= 0.0:
      self.reset()
      return GovernorResult(0.0, V3GovernorReason.INVALID, 0.0)

    if steering_pressed:
      output = _approach(self.previous_output, 0.0, OVERRIDE_RELEASE_RATE * self.dt)
      self.previous_output = output
      return GovernorResult(output, V3GovernorReason.DRIVER_OVERRIDE, max_output)

    output_cap = max_output
    high_rate_blend = _clip((abs(steering_rate_deg) - TOYOTA_HIGH_RATE_START_DEG) /
                            max(TOYOTA_HIGH_RATE_FULL_DEG - TOYOTA_HIGH_RATE_START_DEG, 1e-3), 0.0, 1.0)
    if high_rate_blend > 0.0:
      output_cap = min(output_cap, max_output * (1.0 + high_rate_blend * (TOYOTA_HIGH_RATE_MIN_CAP - 1.0)))
      reason |= V3GovernorReason.TOYOTA_HIGH_RATE
    if same_direction_limit:
      output_cap = min(output_cap, max_output * SAME_DIRECTION_LIMIT_CAP)
      reason |= V3GovernorReason.SAME_DIRECTION_LIMIT

    clipped = _clip(raw_output_torque, -output_cap, output_cap)
    if abs(clipped - raw_output_torque) > 1e-6:
      reason |= V3GovernorReason.CLIPPED

    previous_sign = _sign(self.previous_output, 1e-4)
    target_sign = _sign(clipped, 1e-4)
    sign_change = previous_sign != 0 and target_sign != 0 and previous_sign != target_sign
    slew_rate = _interp(v_ego, SIGN_CHANGE_SLEW_RATE_BP, SIGN_CHANGE_SLEW_RATE_V) if sign_change else _interp(v_ego, OUTPUT_SLEW_RATE_BP, OUTPUT_SLEW_RATE_V)
    if sign_change:
      reason |= V3GovernorReason.SIGN_CHANGE_LIMITED
    if high_rate_blend > 0.0:
      slew_rate *= TOYOTA_HIGH_RATE_SLEW_SCALE
    if same_direction_limit:
      slew_rate = min(slew_rate, SAME_DIRECTION_LIMIT_RATE)

    output = _approach(self.previous_output, clipped, slew_rate * self.dt)
    if abs(output - clipped) > 1e-6:
      reason |= V3GovernorReason.SLEW_LIMITED
    self.previous_output = output
    return GovernorResult(output, reason, output_cap)


class LatControlTorque(LatControl):
  CONTROL_STATE = "torque"

  def __init__(self, CP, CP_SP, CI, dt):
    super().__init__(CP, CP_SP, CI, dt)
    if CP.lateralTuning.which() != 'torque':
      raise ValueError("Torque v3 requires native torque lateral tuning")
    self.torque_params = CP.lateralTuning.torque.as_builder()
    self.torque_from_lateral_accel = CI.torque_from_lateral_accel()
    self.lateral_accel_from_torque = CI.lateral_accel_from_torque()
    self.steering_angle_deadzone_deg = self.torque_params.steeringAngleDeadzoneDeg
    self.lat_accel_request_buffer_len = int(LAT_ACCEL_REQUEST_BUFFER_SECONDS / self.dt)
    self.lat_accel_request_buffer = [0.0] * self.lat_accel_request_buffer_len
    self.previous_target_lateral_accel = 0.0
    self.previous_measurement = 0.0
    self.measurement_rate_filter = FirstOrderFilter(0.0, 1 / (2 * math.pi * MEASUREMENT_FILTER_CUTOFF_HZ), self.dt)
    self.measurement_filter = LateralAccelMeasurementFilter(self.dt)
    self.governor = TorqueV3OutputGovernor(self.dt)
    self.extension = LatControlTorqueExt(self, CP, CP_SP, CI)
    self.lat_delay = max(float(getattr(CP, "steerActuatorDelay", 0.2)), self.dt)

  def update_live_torque_params(self, latAccelFactor, latAccelOffset, friction):
    self.torque_params.latAccelFactor = latAccelFactor
    self.torque_params.latAccelOffset = latAccelOffset
    self.torque_params.friction = friction

  def update_lateral_lag(self, lag):
    try:
      lag = float(lag)
    except (TypeError, ValueError):
      lag = self.dt
    self.lat_delay = max(lag if math.isfinite(lag) else self.dt, self.dt)

  def reset(self):
    super().reset()
    self.governor.reset()
    self.previous_target_lateral_accel = 0.0
    self.previous_measurement = 0.0
    self.lat_accel_request_buffer = [0.0] * self.lat_accel_request_buffer_len

  def update(self, active, CS, VM, params, steer_limited_by_safety, desired_curvature, calibrated_pose, curvature_limited, lat_delay):
    del calibrated_pose
    if _finite(lat_delay):
      self.update_lateral_lag(lat_delay)
    self.extension.last_v_ego = CS.vEgo
    self.extension.update_override_torque_params(self.torque_params)

    pid_log = log.ControlsState.LateralTorqueState.new_message()
    pid_log.version = VERSION

    input_invalid = not _finite(desired_curvature, CS.vEgo, CS.steeringAngleDeg, CS.steeringRateDeg, params.angleOffsetDeg, params.roll)
    raw_target_lateral_accel = desired_curvature * CS.vEgo ** 2 if not input_invalid else 0.0
    target_lateral_accel_rate = (raw_target_lateral_accel - self.previous_target_lateral_accel) / self.dt
    self.previous_target_lateral_accel = raw_target_lateral_accel
    self.lat_accel_request_buffer.append(raw_target_lateral_accel)
    self.lat_accel_request_buffer = self.lat_accel_request_buffer[-self.lat_accel_request_buffer_len:]

    measured_curvature = -VM.calc_curvature(math.radians(CS.steeringAngleDeg - params.angleOffsetDeg), CS.vEgo, params.roll)
    raw_measurement = measured_curvature * CS.vEgo ** 2
    raw_actual_lateral_jerk = -VM.calc_curvature(math.radians(CS.steeringRateDeg), CS.vEgo, 0.0) * CS.vEgo ** 2
    measurement = self.measurement_filter.update(active, CS.vEgo, CS.steeringPressed, raw_measurement, raw_actual_lateral_jerk)
    if self.measurement_filter.was_reset:
      self.measurement_rate_filter.x = 0.0
      measurement_rate = 0.0
    else:
      measurement_rate = self.measurement_rate_filter.update((measurement - self.previous_measurement) / self.dt)
    self.previous_measurement = measurement

    lead_gain = _interp(CS.vEgo, LEAD_GAIN_BP, LEAD_GAIN_V)
    lead_delta_cap = _interp(CS.vEgo, LEAD_DELTA_CAP_BP, LEAD_DELTA_CAP_V)
    lead_delta = _clip(target_lateral_accel_rate * self.lat_delay * lead_gain, -lead_delta_cap, lead_delta_cap)
    lead_lateral_accel = raw_target_lateral_accel + lead_delta
    roll_compensation = params.roll * ACCELERATION_DUE_TO_GRAVITY
    curvature_deadzone = abs(VM.calc_curvature(math.radians(self.steering_angle_deadzone_deg), CS.vEgo, 0.0))
    lateral_accel_deadzone = curvature_deadzone * CS.vEgo ** 2
    control_error = lead_lateral_accel - measurement
    feedback_correction = _interp(CS.vEgo, FEEDBACK_GAIN_BP, FEEDBACK_GAIN_V) * control_error
    damping_correction = -_interp(CS.vEgo, DAMPING_GAIN_BP, DAMPING_GAIN_V) * measurement_rate
    breakaway = self._breakaway_lateral_accel(control_error, lateral_accel_deadzone, raw_target_lateral_accel, measurement)
    trim_correction = 0.0
    commanded_lateral_accel = (
      lead_lateral_accel
      - roll_compensation
      - self.torque_params.latAccelOffset
      + feedback_correction
      + damping_correction
      + breakaway
      + trim_correction
    )

    invalid = input_invalid or not _finite(commanded_lateral_accel, lead_lateral_accel, measurement, raw_actual_lateral_jerk, measurement_rate)
    raw_output_torque = 0.0 if invalid else self.torque_from_lateral_accel(commanded_lateral_accel, self.torque_params)
    raw_output_torque = _clip(raw_output_torque, -self.steer_max, self.steer_max) if _finite(raw_output_torque) else 0.0

    steer_limit_feedback = self.steering_actuator_feedback
    steer_limit_same_direction, steer_limit_unwind = classify_steering_limit_direction(steer_limit_feedback, -raw_output_torque)
    if steer_limit_feedback.valid:
      steer_limit_same_direction = steer_limit_same_direction or bool(steer_limit_feedback.same_direction_limited)
      steer_limit_unwind = steer_limit_unwind or bool(steer_limit_feedback.unwind_allowed)
    same_direction_limit = bool(steer_limited_by_safety and (steer_limit_same_direction if steer_limit_feedback.valid else True))
    governor_result = self.governor.update(
      active and not invalid,
      CS.vEgo,
      CS.steeringPressed,
      CS.steeringRateDeg,
      same_direction_limit,
      raw_output_torque,
      self.steer_max,
    )
    output_torque = governor_result.output_torque
    saturated = self.steer_max - abs(output_torque) < 1e-3 or bool(governor_result.reason & V3GovernorReason.CLIPPED)
    delayed_target = self._delayed_target()
    response_sample = evaluate_response_sample(
      ResponseSample(
        active=active and not invalid,
        steering_pressed=CS.steeringPressed,
        same_direction_limit=same_direction_limit,
        curvature_limited=curvature_limited,
        saturated=saturated,
        lateral_maneuver=bool(getattr(CS, "leftBlinker", False) or getattr(CS, "rightBlinker", False)),
        v_ego=CS.vEgo,
        commanded_torque=output_torque,
        target_lateral_accel=delayed_target,
        target_lateral_accel_rate=target_lateral_accel_rate,
        actual_lateral_accel=measurement,
        actual_lateral_jerk=raw_actual_lateral_jerk,
      )
    )
    if invalid:
      governor_result.reason |= V3GovernorReason.INVALID

    pid_log.active = bool(active and not invalid)
    pid_log.error = float(control_error if _finite(control_error) else 0.0)
    pid_log.errorRate = float(-measurement_rate if _finite(measurement_rate) else 0.0)
    pid_log.p = float(feedback_correction if _finite(feedback_correction) else 0.0)
    pid_log.i = float(trim_correction if _finite(trim_correction) else 0.0)
    pid_log.d = float(damping_correction if _finite(damping_correction) else 0.0)
    pid_log.f = float(commanded_lateral_accel if _finite(commanded_lateral_accel) else 0.0)
    pid_log.output = float(-output_torque)
    pid_log.actualLateralAccel = float(measurement if _finite(measurement) else 0.0)
    pid_log.desiredLateralAccel = float(raw_target_lateral_accel if _finite(raw_target_lateral_accel) else 0.0)
    pid_log.desiredLateralJerk = float(target_lateral_accel_rate if _finite(target_lateral_accel_rate) else 0.0)
    pid_log.saturated = bool(self._check_saturation(saturated, CS, steer_limited_by_safety, curvature_limited))
    self._fill_adaptive_log(
      pid_log,
      active,
      raw_target_lateral_accel,
      lead_lateral_accel,
      feedback_correction,
      trim_correction,
      raw_output_torque,
      governor_result,
      response_sample,
      steer_limit_feedback,
      steer_limit_same_direction,
      steer_limit_unwind,
      raw_actual_lateral_jerk,
      target_lateral_accel_rate,
      lead_gain,
    )

    return -output_torque, 0.0, pid_log

  def _breakaway_lateral_accel(self, error: float, lateral_accel_deadzone: float, target: float, measurement: float) -> float:
    demand = max(abs(target), abs(measurement), abs(error))
    scale = _clip(demand / BREAKAWAY_FULL_DEMAND, 0.0, BREAKAWAY_MAX_SCALE)
    if _sign(target, LEARN_SIGN_THRESHOLD) != 0 and _sign(measurement, LEARN_SIGN_THRESHOLD) != 0 and _sign(target) != _sign(measurement):
      scale *= 0.5
    return get_friction(error * scale, lateral_accel_deadzone, FRICTION_THRESHOLD, self.torque_params) * scale

  def _delayed_target(self) -> float:
    delay_frames = int(np.clip(self.lat_delay / self.dt, 1, self.lat_accel_request_buffer_len))
    return float(self.lat_accel_request_buffer[-delay_frames])

  @staticmethod
  def _phase(active: bool, target_rate: float, governor_reason: V3GovernorReason) -> V3Phase:
    if not active:
      return V3Phase.idle
    if governor_reason & (V3GovernorReason.DRIVER_OVERRIDE | V3GovernorReason.SIGN_CHANGE_LIMITED):
      return V3Phase.release
    if abs(target_rate) > 0.25:
      return V3Phase.engage if target_rate > 0.0 else V3Phase.release
    return V3Phase.hold

  def _fill_adaptive_log(self, pid_log, active: bool, raw_target_lateral_accel: float, lead_lateral_accel: float,
                          feedback_correction: float, trim_correction: float, raw_output_torque: float,
                          governor_result: GovernorResult, response_sample: ResponseSampleUpdate, steer_limit_feedback,
                          steer_limit_same_direction: bool, steer_limit_unwind: bool, actual_lateral_jerk: float,
                          target_lateral_accel_rate: float, lead_gain: float) -> None:
    adaptive_log = pid_log.init('adaptiveTorqueState')
    log_active = bool(active and pid_log.active)
    adaptive_log.active = log_active
    adaptive_log.phase = PHASE_TO_CAPNP[self._phase(log_active, target_lateral_accel_rate, governor_result.reason)]
    adaptive_log.releaseActive = bool(governor_result.reason & (V3GovernorReason.DRIVER_OVERRIDE | V3GovernorReason.SIGN_CHANGE_LIMITED))
    adaptive_log.phaseGain = float(lead_gain if _finite(lead_gain) else 0.0)
    adaptive_log.nominalOutput = float(-raw_output_torque)
    adaptive_log.assistOutput = float(feedback_correction)
    adaptive_log.biasOutput = float(trim_correction)
    adaptive_log.responseDeficit = float(pid_log.error)
    adaptive_log.learningFrozen = False
    adaptive_log.freezeReason = int(V3LearnerRejectReason.NONE)
    adaptive_log.blockReason = int(V3LearnerRejectReason.NONE)
    adaptive_log.shapingActive = bool(governor_result.reason != V3GovernorReason.NONE)
    adaptive_log.shapingReason = int(governor_result.reason)
    adaptive_log.shapingConfidence = float(response_sample.correction.confidence)
    adaptive_log.unshapedOutput = float(-raw_output_torque)
    adaptive_log.outputCap = float(governor_result.output_cap)
    adaptive_log.modelMode = 1
    adaptive_log.modelConfidence = float(response_sample.correction.confidence)
    adaptive_log.authorityBand = 0
    adaptive_log.authorityScale = float(response_sample.correction.response_scale)
    adaptive_log.fallbackActive = False
    adaptive_log.learnedLatAccelFactor = float(self.torque_params.latAccelFactor)
    adaptive_log.learnedFriction = float(self.torque_params.friction)
    adaptive_log.learnedLatAccelOffset = float(self.torque_params.latAccelOffset)
    adaptive_log.learnedResponseDelay = float(self.lat_delay)
    adaptive_log.residualError = float(response_sample.residual_error)
    adaptive_log.sampleAccepted = bool(response_sample.sample_accepted)
    adaptive_log.sampleRejectReason = int(response_sample.reject_reason)
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
    adaptive_log.rawTargetLateralAccel = float(raw_target_lateral_accel)
    adaptive_log.delayLeadLateralAccel = float(lead_lateral_accel)
    adaptive_log.feedbackCorrection = float(feedback_correction)
    adaptive_log.trimCorrection = float(trim_correction)
    adaptive_log.learnerResponseScale = float(response_sample.correction.response_scale)
    adaptive_log.governorReason = int(governor_result.reason)
    adaptive_log.actualLateralJerk = float(actual_lateral_jerk if _finite(actual_lateral_jerk) else 0.0)


LatControlTorqueV3 = LatControlTorque
