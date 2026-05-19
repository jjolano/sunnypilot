import math
import numpy as np
from collections import deque
from dataclasses import dataclass
from enum import IntFlag

from cereal import log
from opendbc.car.lateral import get_friction
from openpilot.common.constants import ACCELERATION_DUE_TO_GRAVITY
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.pid import PIDController
from openpilot.selfdrive.controls.lib.latcontrol import LatControl

from openpilot.sunnypilot.selfdrive.controls.lib.latcontrol_torque_ext import LatControlTorqueExt
from openpilot.sunnypilot.selfdrive.controls.lib.torque_conservative_output_shaper import ConservativeOutputShaperInputs, TorqueConservativeOutputShaper
from openpilot.sunnypilot.selfdrive.controls.lib.torque_disturbance import TorqueDisturbanceInputs, classify_torque_disturbance
from openpilot.sunnypilot.selfdrive.controls.lib.torque_guarded_response_assist import GuardedResponseAssistInputs, TorqueGuardedResponseAssist
from openpilot.sunnypilot.selfdrive.controls.lib.steering_actuator_feedback import classify_steering_limit_direction
from openpilot.sunnypilot.selfdrive.controls.lib.torque_low_speed import low_speed_pid_gain_speed
from openpilot.sunnypilot.selfdrive.controls.lib.torque_observation import TorqueObservation
from openpilot.sunnypilot.selfdrive.controls.lib.torque_over_response_attenuator import attenuate_same_direction_over_response


KP = 1.0
KI = 0.2
KD = 0.0
INTERP_SPEEDS = [1, 1.5, 2.0, 3.0, 5, 7.5, 10, 15, 30]
# V2 is now the retained selector for the tested conservative torque controller.
KP_INTERP = [165, 90, 52, 26, 9.0, 5.5, 3.5, 2.0, KP]

LP_FILTER_CUTOFF_HZ = 1.2
LAT_ACCEL_REQUEST_BUFFER_SECONDS = 1.0
FRICTION_THRESHOLD = 0.3
LOW_DEMAND_FRICTION_FULL_LAT_ACCEL = 0.35
VERSION_V2 = 2
VERSION_V21 = 21
LOW_SPEED_UNWIND_VEGO = 8.0
LOW_SPEED_UNWIND_SETPOINT = 0.2
LOW_SPEED_UNWIND_MARGIN = 0.08
LOW_SPEED_UNWIND_JERK = 0.5
LOW_SPEED_UNWIND_GAIN_SPEED = 8.0
MEASUREMENT_SMOOTHER_MIN_VEGO = 5.0
MEASUREMENT_SMOOTHER_CORRECTION_GAIN = 0.35
MEASUREMENT_SMOOTHER_MAX_PREDICTIVE_JERK = 5.0
MEASUREMENT_SMOOTHER_IMPLAUSIBLE_JERK = 80.0
MEASUREMENT_SMOOTHER_MAX_RAW_ERROR = 1.0

V21_OUTPUT_SLEW_RATE_BP = [0.0, 5.0, 10.0, 20.0, 30.0, 40.0]
V21_OUTPUT_SLEW_RATE_V = [1.40, 2.00, 3.00, 4.20, 5.00, 5.60]
V21_SIGN_CHANGE_SLEW_RATE_BP = [0.0, 5.0, 10.0, 20.0, 30.0, 40.0]
V21_SIGN_CHANGE_SLEW_RATE_V = [0.90, 1.20, 1.80, 2.40, 3.00, 3.40]
V21_OVERRIDE_RELEASE_RATE = 6.0
V21_SAME_DIRECTION_LIMIT_RATE_BP = [0.0, 10.0, 20.0, 30.0, 40.0]
V21_SAME_DIRECTION_LIMIT_RATE_V = [1.30, 1.30, 2.10, 3.20, 3.60]
V21_SAME_DIRECTION_LIMIT_CAP = 0.85
V21_HIGH_RATE_START_DEG = 80.0
V21_HIGH_RATE_FULL_DEG = 100.0
V21_HIGH_RATE_MIN_CAP = 0.62
V21_HIGH_RATE_SLEW_SCALE = 0.70
V21_UNDER_RESPONSE_MARGIN = 0.12
V21_UNDER_RESPONSE_FULL_SPEED = 9.0
V21_UNDER_RESPONSE_FADE_SPEED = 12.0
V21_SIGN_THRESHOLD = 0.05

ADAPTIVE_PHASE_MAP = {
  0: log.ControlsState.LateralTorqueState.AdaptiveTorqueState.Phase.idle,
  1: log.ControlsState.LateralTorqueState.AdaptiveTorqueState.Phase.engage,
  2: log.ControlsState.LateralTorqueState.AdaptiveTorqueState.Phase.hold,
  3: log.ControlsState.LateralTorqueState.AdaptiveTorqueState.Phase.release,
}


def sign(value: float) -> float:
  return 1.0 if value > 0.0 else (-1.0 if value < 0.0 else 0.0)


def approach(value: float, target: float, step: float) -> float:
  if target > value:
    return min(target, value + step)
  return max(target, value - step)


def low_demand_friction_scale(setpoint: float, measurement: float) -> float:
  demand = max(abs(setpoint), abs(measurement))
  return float(np.clip(demand / LOW_DEMAND_FRICTION_FULL_LAT_ACCEL, 0.0, 1.0))


class RefinedOutputGovernorReason(IntFlag):
  NONE = 0
  CLIPPED = 1 << 0
  SLEW_LIMITED = 1 << 1
  SIGN_CHANGE_LIMITED = 1 << 2
  DRIVER_OVERRIDE = 1 << 3
  SAME_DIRECTION_LIMIT = 1 << 4
  HIGH_STEERING_RATE = 1 << 5
  INVALID = 1 << 6
  UNDER_RESPONSE_FLOOR = 1 << 7


@dataclass(frozen=True)
class RefinedOutputGovernorInputs:
  active: bool
  v_ego: float
  steering_pressed: bool
  steering_rate_deg: float
  same_direction_limit: bool
  output_torque: float
  max_output: float
  desired_lateral_accel: float
  actual_lateral_accel: float


@dataclass(frozen=True)
class RefinedOutputGovernorResult:
  output_torque: float
  active: bool
  reason: RefinedOutputGovernorReason


class TorqueV21RefinedOutputGovernor:
  def __init__(self, dt: float):
    self.dt = max(float(dt), 1e-3)
    self.previous_output = 0.0

  def reset(self) -> None:
    self.previous_output = 0.0

  def update(self, inputs: RefinedOutputGovernorInputs) -> RefinedOutputGovernorResult:
    reason = RefinedOutputGovernorReason.NONE
    if not inputs.active:
      self.reset()
      return RefinedOutputGovernorResult(0.0, False, reason)
    if not self._finite(inputs.v_ego, inputs.steering_rate_deg, inputs.output_torque, inputs.max_output,
                        inputs.desired_lateral_accel, inputs.actual_lateral_accel) or inputs.max_output <= 0.0:
      self.reset()
      return RefinedOutputGovernorResult(0.0, True, RefinedOutputGovernorReason.INVALID)

    if inputs.steering_pressed:
      output = approach(self.previous_output, 0.0, V21_OVERRIDE_RELEASE_RATE * self.dt)
      self.previous_output = output
      return RefinedOutputGovernorResult(output, abs(output - inputs.output_torque) > 1e-6,
                                         RefinedOutputGovernorReason.DRIVER_OVERRIDE)

    under_response_floor = self._under_response_floor(inputs)
    if under_response_floor > 0.0:
      reason |= RefinedOutputGovernorReason.UNDER_RESPONSE_FLOOR

    output_cap = inputs.max_output
    high_rate_blend = float(np.clip((abs(inputs.steering_rate_deg) - V21_HIGH_RATE_START_DEG) /
                                    max(V21_HIGH_RATE_FULL_DEG - V21_HIGH_RATE_START_DEG, 1e-3), 0.0, 1.0))
    if high_rate_blend > 0.0:
      output_cap = min(output_cap, inputs.max_output * (1.0 + high_rate_blend * (V21_HIGH_RATE_MIN_CAP - 1.0)))
      reason |= RefinedOutputGovernorReason.HIGH_STEERING_RATE
    if inputs.same_direction_limit:
      output_cap = min(output_cap, inputs.max_output * V21_SAME_DIRECTION_LIMIT_CAP)
      reason |= RefinedOutputGovernorReason.SAME_DIRECTION_LIMIT
    if under_response_floor > 0.0:
      output_cap += under_response_floor * (inputs.max_output - output_cap)

    clipped = float(np.clip(inputs.output_torque, -output_cap, output_cap))
    if abs(clipped - inputs.output_torque) > 1e-6:
      reason |= RefinedOutputGovernorReason.CLIPPED

    previous_sign = sign(self.previous_output)
    target_sign = sign(clipped)
    sign_change = previous_sign != 0.0 and target_sign != 0.0 and previous_sign != target_sign
    slew_rate = float(np.interp(inputs.v_ego, V21_SIGN_CHANGE_SLEW_RATE_BP, V21_SIGN_CHANGE_SLEW_RATE_V)) if sign_change else \
      float(np.interp(inputs.v_ego, V21_OUTPUT_SLEW_RATE_BP, V21_OUTPUT_SLEW_RATE_V))
    if sign_change:
      reason |= RefinedOutputGovernorReason.SIGN_CHANGE_LIMITED
    if high_rate_blend > 0.0:
      slew_rate *= V21_HIGH_RATE_SLEW_SCALE
    if inputs.same_direction_limit:
      same_direction_rate = float(np.interp(inputs.v_ego, V21_SAME_DIRECTION_LIMIT_RATE_BP, V21_SAME_DIRECTION_LIMIT_RATE_V))
      slew_rate = min(slew_rate, same_direction_rate)

    target_decreases_same_direction = previous_sign != 0.0 and target_sign == previous_sign and abs(clipped) <= abs(self.previous_output)
    limited = clipped if target_decreases_same_direction else approach(self.previous_output, clipped, slew_rate * self.dt)
    output = limited + under_response_floor * (clipped - limited)
    if abs(output - clipped) > 1e-6:
      reason |= RefinedOutputGovernorReason.SLEW_LIMITED
    self.previous_output = output
    active = abs(output - inputs.output_torque) > 1e-6 or reason != RefinedOutputGovernorReason.NONE
    return RefinedOutputGovernorResult(output, active, reason)

  @staticmethod
  def _finite(*values: float) -> bool:
    return all(math.isfinite(float(value)) for value in values)

  @staticmethod
  def _under_response_floor(inputs: RefinedOutputGovernorInputs) -> float:
    if inputs.v_ego >= V21_UNDER_RESPONSE_FADE_SPEED:
      return 0.0
    desired_sign = sign(inputs.desired_lateral_accel)
    actual_sign = 0.0 if abs(inputs.actual_lateral_accel) <= V21_SIGN_THRESHOLD else sign(inputs.actual_lateral_accel)
    output_sign = sign(inputs.output_torque)
    under_response = desired_sign * (inputs.desired_lateral_accel - inputs.actual_lateral_accel)
    same_sign_lag = desired_sign != 0.0 and output_sign == desired_sign and actual_sign in (0.0, desired_sign)
    corrective_reversal = desired_sign != 0.0 and actual_sign != 0.0 and actual_sign != desired_sign and output_sign == desired_sign
    if under_response <= V21_UNDER_RESPONSE_MARGIN or not (same_sign_lag or corrective_reversal):
      return 0.0
    if inputs.v_ego <= V21_UNDER_RESPONSE_FULL_SPEED:
      return 1.0
    span = V21_UNDER_RESPONSE_FADE_SPEED - V21_UNDER_RESPONSE_FULL_SPEED
    return float(np.clip((V21_UNDER_RESPONSE_FADE_SPEED - inputs.v_ego) / max(span, 1e-3), 0.0, 1.0))


class LateralAccelMeasurementSmoother:
  """Condition angle-derived lateral acceleration for the 100 Hz control loop."""

  def __init__(self, dt: float, min_v_ego: float = MEASUREMENT_SMOOTHER_MIN_VEGO,
               correction_gain: float = MEASUREMENT_SMOOTHER_CORRECTION_GAIN):
    self.dt = dt
    self.min_v_ego = min_v_ego
    self.correction_gain = correction_gain
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
    if not math.isfinite(raw_lateral_accel):
      return self.reset(0.0)
    if not active or steering_pressed or v_ego < self.min_v_ego:
      return self.reset(raw_lateral_accel)
    if not math.isfinite(actual_lateral_jerk) or abs(actual_lateral_jerk) > MEASUREMENT_SMOOTHER_IMPLAUSIBLE_JERK:
      return self.reset(raw_lateral_accel)

    if not self.initialized:
      self.initialized = True
      self.value = raw_lateral_accel
      return raw_lateral_accel

    bounded_jerk = float(np.clip(actual_lateral_jerk, -MEASUREMENT_SMOOTHER_MAX_PREDICTIVE_JERK,
                                 MEASUREMENT_SMOOTHER_MAX_PREDICTIVE_JERK))
    predicted = self.value + bounded_jerk * self.dt
    self.value = predicted + self.correction_gain * (raw_lateral_accel - predicted)
    self.value = float(np.clip(self.value, raw_lateral_accel - MEASUREMENT_SMOOTHER_MAX_RAW_ERROR,
                               raw_lateral_accel + MEASUREMENT_SMOOTHER_MAX_RAW_ERROR))
    return self.value


class LatControlTorque(LatControl):
  CONTROLLER_VERSION = VERSION_V2
  USE_REFINED_OUTPUT_GOVERNOR = False

  def __init__(self, CP, CP_SP, CI, dt):
    super().__init__(CP, CP_SP, CI, dt)
    self.torque_params = CP.lateralTuning.torque.as_builder()
    self.torque_from_lateral_accel = CI.torque_from_lateral_accel()
    self.lateral_accel_from_torque = CI.lateral_accel_from_torque()
    self.pid = PIDController([INTERP_SPEEDS, KP_INTERP], KI, KD, rate=1 / self.dt)
    self.update_limits()
    self.steering_angle_deadzone_deg = self.torque_params.steeringAngleDeadzoneDeg
    self.lat_accel_request_buffer_len = int(LAT_ACCEL_REQUEST_BUFFER_SECONDS / self.dt)
    self.lat_accel_request_buffer = deque([0.0] * self.lat_accel_request_buffer_len, maxlen=self.lat_accel_request_buffer_len)
    self.previous_measurement = 0.0
    self.measurement_rate_filter = FirstOrderFilter(0.0, 1 / (2 * np.pi * LP_FILTER_CUTOFF_HZ), self.dt)
    self.measurement_smoother = LateralAccelMeasurementSmoother(self.dt)

    self.extension = LatControlTorqueExt(self, CP, CP_SP, CI)
    self.response_assist = TorqueGuardedResponseAssist(self.dt)
    self.output_shaper = TorqueConservativeOutputShaper(self.dt)
    self.refined_output_governor = TorqueV21RefinedOutputGovernor(self.dt) if self.USE_REFINED_OUTPUT_GOVERNOR else None

  def update_live_torque_params(self, latAccelFactor, latAccelOffset, friction):
    self.torque_params.latAccelFactor = latAccelFactor
    self.torque_params.latAccelOffset = latAccelOffset
    self.torque_params.friction = friction
    self.update_limits()

  def update_limits(self):
    self.pid.set_limits(self.lateral_accel_from_torque(self.steer_max, self.torque_params), self.lateral_accel_from_torque(-self.steer_max, self.torque_params))

  def update(self, active, CS, VM, params, steer_limited_by_safety, desired_curvature, calibrated_pose, curvature_limited, lat_delay):
    if self.extension.update_override_torque_params(self.torque_params):
      self.update_limits()

    pid_log = log.ControlsState.LateralTorqueState.new_message()
    pid_log.version = self.CONTROLLER_VERSION

    measured_curvature = -VM.calc_curvature(math.radians(CS.steeringAngleDeg - params.angleOffsetDeg), CS.vEgo, params.roll)
    raw_measurement = measured_curvature * CS.vEgo**2
    roll_compensation = params.roll * ACCELERATION_DUE_TO_GRAVITY
    curvature_deadzone = abs(VM.calc_curvature(math.radians(self.steering_angle_deadzone_deg), CS.vEgo, 0.0))
    lateral_accel_deadzone = curvature_deadzone * CS.vEgo**2
    raw_actual_lateral_jerk = -VM.calc_curvature(math.radians(CS.steeringRateDeg), CS.vEgo, 0.0) * CS.vEgo**2
    measurement = self.measurement_smoother.update(active, CS.vEgo, CS.steeringPressed, raw_measurement, raw_actual_lateral_jerk)

    future_desired_lateral_accel = desired_curvature * CS.vEgo**2
    self.lat_accel_request_buffer.append(future_desired_lateral_accel)
    effective_lat_delay = max(lat_delay, self.dt)
    delay_frames = int(np.clip(effective_lat_delay / self.dt, 1, self.lat_accel_request_buffer_len))
    expected_lateral_accel = self.lat_accel_request_buffer[-delay_frames]
    gravity_adjusted_future_lateral_accel = future_desired_lateral_accel - roll_compensation
    desired_lateral_jerk = (future_desired_lateral_accel - expected_lateral_accel) / effective_lat_delay

    if self.measurement_smoother.was_reset:
      self.measurement_rate_filter.x = 0.0
      measurement_rate = 0.0
    else:
      measurement_rate = self.measurement_rate_filter.update((measurement - self.previous_measurement) / self.dt)
    self.previous_measurement = measurement

    setpoint = effective_lat_delay * desired_lateral_jerk + expected_lateral_accel
    error = setpoint - measurement
    same_sign_unwind = (
      CS.vEgo < LOW_SPEED_UNWIND_VEGO
      and abs(setpoint) < LOW_SPEED_UNWIND_SETPOINT
      and abs(desired_lateral_jerk) > LOW_SPEED_UNWIND_JERK
      and sign(setpoint) != 0.0
      and sign(measurement) == sign(setpoint)
      and abs(measurement) > abs(setpoint) + LOW_SPEED_UNWIND_MARGIN
      and desired_lateral_jerk * setpoint < 0.0
    )

    ff = setpoint if same_sign_unwind else gravity_adjusted_future_lateral_accel
    ff -= self.torque_params.latAccelOffset
    friction_scale = low_demand_friction_scale(setpoint, measurement)
    ff += get_friction(error * friction_scale, lateral_accel_deadzone, FRICTION_THRESHOLD, self.torque_params) * friction_scale

    setpoint_sign = sign(setpoint)
    raw_sign = sign(raw_measurement)
    measurement_sign = sign(measurement)
    raw_sign_conflict = setpoint_sign != 0.0 and raw_sign != 0.0 and raw_sign != setpoint_sign
    measurement_sign_conflict = setpoint_sign != 0.0 and measurement_sign != 0.0 and measurement_sign != setpoint_sign
    shaping_measurement = measurement
    if raw_sign_conflict and not measurement_sign_conflict:
      shaping_measurement = raw_measurement
    elif raw_sign_conflict == measurement_sign_conflict and abs(raw_measurement) > abs(measurement):
      shaping_measurement = raw_measurement

    output_torque = 0.0
    if not active:
      pid_log.active = False
      self.pid.reset()
    else:
      pid_log.error = float(error)
      if same_sign_unwind:
        self.pid.i *= 0.5
      freeze_integrator = steer_limited_by_safety or CS.steeringPressed or CS.vEgo < 5 or same_sign_unwind
      control_speed = low_speed_pid_gain_speed(CS.vEgo, LOW_SPEED_UNWIND_GAIN_SPEED if same_sign_unwind else None)
      output_lataccel = self.pid.update(pid_log.error, -measurement_rate, feedforward=ff, speed=control_speed, freeze_integrator=freeze_integrator)
      output_torque = self.torque_from_lateral_accel(output_lataccel, self.torque_params)

      pid_log, output_torque = self.extension.update(
        CS,
        VM,
        self.pid,
        params,
        ff,
        pid_log,
        setpoint,
        measurement,
        calibrated_pose,
        roll_compensation,
        future_desired_lateral_accel,
        measurement,
        lateral_accel_deadzone,
        gravity_adjusted_future_lateral_accel,
        desired_curvature,
        measured_curvature,
        steer_limited_by_safety,
        output_torque,
      )

      output_torque = attenuate_same_direction_over_response(output_torque, setpoint, measurement)
      pid_log.active = True

    saturated = self.steer_max - abs(output_torque) < 1e-3
    tracking_torque_error = error / max(float(self.torque_params.latAccelFactor), 1e-3)
    lane_change_active = bool(self.extension.model_valid and self.extension.model_v2.meta.laneChangeState != log.LaneChangeState.off)
    steer_limit_feedback = self.steering_actuator_feedback
    steer_limit_same_direction, steer_limit_unwind = classify_steering_limit_direction(steer_limit_feedback, -output_torque)
    response_steer_limited = steer_limited_by_safety and (steer_limit_same_direction if steer_limit_feedback.valid else True)
    torque_observation = TorqueObservation(
      active=active,
      v_ego=CS.vEgo,
      steering_pressed=CS.steeringPressed,
      steer_limited_by_safety=response_steer_limited,
      curvature_limited=curvature_limited,
      saturated=saturated,
      lateral_maneuver=lane_change_active,
      target_lateral_accel=setpoint,
      target_lateral_accel_rate=desired_lateral_jerk,
      actual_lateral_accel=measurement,
      actual_lateral_jerk=raw_actual_lateral_jerk,
    )
    assist_result = self.response_assist.update(
      GuardedResponseAssistInputs(
        active=torque_observation.active,
        v_ego=torque_observation.v_ego,
        steering_pressed=torque_observation.steering_pressed,
        steer_limited_by_safety=torque_observation.steer_limited_by_safety,
        curvature_limited=torque_observation.curvature_limited,
        saturated=torque_observation.saturated,
        max_output=self.steer_max,
        nominal_torque=output_torque,
        desired_lateral_accel=torque_observation.target_lateral_accel,
        actual_lateral_accel=torque_observation.actual_lateral_accel,
        desired_lateral_jerk=torque_observation.target_lateral_accel_rate,
        actual_lateral_jerk=torque_observation.actual_lateral_jerk,
        lookahead_lateral_jerk=self.extension.lookahead_lateral_jerk,
        desired_curvature=desired_curvature,
        tracking_torque_error=tracking_torque_error,
        lane_change_active=torque_observation.lateral_maneuver,
        same_sign_unwind=same_sign_unwind,
      )
    )
    output_torque = assist_result.output_torque if active else 0.0
    same_sign_unwind_release = same_sign_unwind and sign(measurement) != 0.0 and sign(-output_torque) == sign(measurement)
    steer_limit_same_direction, steer_limit_unwind = classify_steering_limit_direction(steer_limit_feedback, -output_torque)
    shaping_result = self.output_shaper.update(
      ConservativeOutputShaperInputs(
        active=torque_observation.active,
        v_ego=torque_observation.v_ego,
        steering_pressed=torque_observation.steering_pressed,
        steer_limited_by_safety=steer_limited_by_safety,
        release_active=assist_result.release_active,
        max_output=self.steer_max,
        unshaped_output=output_torque,
        desired_lateral_accel=torque_observation.target_lateral_accel,
        actual_lateral_accel=shaping_measurement,
        desired_lateral_jerk=torque_observation.target_lateral_accel_rate,
        actual_lateral_jerk=torque_observation.actual_lateral_jerk,
        lookahead_lateral_jerk=self.extension.lookahead_lateral_jerk,
        same_sign_unwind_release=same_sign_unwind_release,
        # The shaper sees pre-actuator torque; the actuator command is negated on return.
        steering_rate_deg=-CS.steeringRateDeg,
        steer_limit_same_direction=steer_limit_same_direction if steer_limit_feedback.valid else True,
        steer_limit_unwind=steer_limit_unwind if steer_limit_feedback.valid else False,
        steer_limit_requested_output=-steer_limit_feedback.requested if steer_limit_feedback.valid else 0.0,
        steer_limit_applied_output=-steer_limit_feedback.applied if steer_limit_feedback.valid else 0.0,
      )
    )
    disturbance_result = classify_torque_disturbance(
      TorqueDisturbanceInputs(
        active=torque_observation.active,
        v_ego=torque_observation.v_ego,
        steering_pressed=torque_observation.steering_pressed,
        steer_limited_by_safety=steer_limited_by_safety,
        curvature_limited=torque_observation.curvature_limited,
        saturated=saturated or self.steer_max - abs(shaping_result.unshaped_output) < 1e-3,
        desired_lateral_accel=torque_observation.target_lateral_accel,
        actual_lateral_accel=shaping_measurement,
        desired_lateral_jerk=torque_observation.target_lateral_accel_rate,
        actual_lateral_jerk=torque_observation.actual_lateral_jerk,
        lookahead_lateral_jerk=self.extension.lookahead_lateral_jerk,
        output_torque=shaping_result.unshaped_output,
        response_deficit=assist_result.response_deficit,
        same_sign_unwind=same_sign_unwind,
        measurement_reset=self.measurement_smoother.was_reset,
        measurement_valid=math.isfinite(raw_measurement) and math.isfinite(raw_actual_lateral_jerk) and math.isfinite(measurement),
      )
    )
    output_torque = shaping_result.output_torque
    governor_result = RefinedOutputGovernorResult(output_torque, False, RefinedOutputGovernorReason.NONE)
    if self.refined_output_governor is not None:
      governor_same_direction_limit = steer_limited_by_safety and (steer_limit_same_direction if steer_limit_feedback.valid else True) and not steer_limit_unwind
      governor_result = self.refined_output_governor.update(
        RefinedOutputGovernorInputs(
          active=torque_observation.active,
          v_ego=torque_observation.v_ego,
          steering_pressed=torque_observation.steering_pressed,
          steering_rate_deg=-CS.steeringRateDeg,
          same_direction_limit=governor_same_direction_limit,
          output_torque=output_torque,
          max_output=self.steer_max,
          desired_lateral_accel=torque_observation.target_lateral_accel,
          actual_lateral_accel=shaping_measurement,
        )
      )
      output_torque = governor_result.output_torque

    pid_log.p = float(self.pid.p)
    pid_log.i = float(self.pid.i)
    pid_log.d = float(self.pid.d)
    pid_log.f = float(self.pid.f)
    pid_log.output = float(-output_torque)
    pid_log.actualLateralAccel = float(measurement)
    pid_log.desiredLateralAccel = float(setpoint)
    pid_log.desiredLateralJerk = float(desired_lateral_jerk)
    adaptive_log = pid_log.init('adaptiveTorqueState')
    adaptive_active = active and (
      shaping_result.active or governor_result.active or assist_result.phase_id != 0
      or abs(assist_result.assist_torque) > 1e-3 or abs(assist_result.bias_torque) > 1e-3
    )
    adaptive_log.active = bool(adaptive_active)
    adaptive_log.phase = ADAPTIVE_PHASE_MAP[assist_result.phase_id]
    adaptive_log.releaseActive = bool(assist_result.release_active)
    adaptive_log.phaseGain = float(assist_result.phase_gain)
    adaptive_log.nominalOutput = float(-assist_result.nominal_torque)
    adaptive_log.assistOutput = float(-assist_result.assist_torque)
    adaptive_log.biasOutput = float(-assist_result.bias_torque)
    adaptive_log.responseDeficit = float(assist_result.response_deficit)
    adaptive_log.learningFrozen = bool(assist_result.learning_frozen)
    adaptive_log.freezeReason = int(assist_result.freeze_reason)
    adaptive_log.blockReason = int(assist_result.block_reason)
    adaptive_log.shapingActive = bool(shaping_result.active)
    adaptive_log.shapingReason = int(shaping_result.reason)
    adaptive_log.shapingConfidence = float(shaping_result.confidence)
    adaptive_log.unshapedOutput = float(-shaping_result.unshaped_output)
    adaptive_log.outputCap = float(shaping_result.output_cap)
    adaptive_log.governorReason = int(governor_result.reason)
    adaptive_log.disturbanceState = int(disturbance_result.state)
    adaptive_log.disturbanceReason = int(disturbance_result.reason)
    adaptive_log.disturbanceConfidence = float(disturbance_result.confidence)
    adaptive_log.steerLimitValid = bool(steer_limit_feedback.valid)
    adaptive_log.steerLimitLimited = bool(steer_limit_feedback.limited)
    adaptive_log.steerLimitReason = int(steer_limit_feedback.reason)
    adaptive_log.steerLimitRequested = float(steer_limit_feedback.requested)
    adaptive_log.steerLimitApplied = float(steer_limit_feedback.applied)
    adaptive_log.steerLimitError = float(steer_limit_feedback.error)
    adaptive_log.steerLimitSameDirection = bool(steer_limit_same_direction)
    adaptive_log.steerLimitUnwind = bool(steer_limit_unwind)
    pid_log.saturated = bool(self._check_saturation(self.steer_max - abs(output_torque) < 1e-3, CS, steer_limited_by_safety, curvature_limited))

    return -output_torque, 0.0, pid_log


class LatControlTorqueV21(LatControlTorque):
  CONTROLLER_VERSION = VERSION_V21
  USE_REFINED_OUTPUT_GOVERNOR = True


LatControlTorqueV2 = LatControlTorque
