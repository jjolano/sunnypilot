import math
import numpy as np
from collections import deque

from cereal import log
from opendbc.car.lateral import get_friction
from openpilot.common.constants import ACCELERATION_DUE_TO_GRAVITY
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.pid import PIDController
from openpilot.selfdrive.controls.lib.latcontrol import LatControl

from openpilot.sunnypilot.selfdrive.controls.lib.latcontrol_torque_ext import LatControlTorqueExt
from openpilot.sunnypilot.selfdrive.controls.lib.torque_guarded_response_assist import GuardedResponseAssistInputs, TorqueGuardedResponseAssist
from openpilot.sunnypilot.selfdrive.controls.lib.torque_over_response_attenuator import attenuate_same_direction_over_response
from openpilot.sunnypilot.selfdrive.controls.lib.torque_v3_authority import AuthorityManager, authority_fault_active
from openpilot.sunnypilot.selfdrive.controls.lib.torque_v3_estimator import AdaptiveTorqueEstimator, EstimatorRejectReason, TorqueObservation
from openpilot.sunnypilot.selfdrive.controls.lib.torque_v3_model import TorqueModelAdapter, TorqueModelMode, TorqueModelParams
from openpilot.sunnypilot.selfdrive.controls.lib.torque_v3_safety import TorqueV3SafetyEnvelope, TorqueV3SafetyInputs


KP = 1.0
KI = 0.2
KD = 0.0
INTERP_SPEEDS = [1, 1.5, 2.0, 3.0, 5, 7.5, 10, 15, 30]
KP_INTERP = [165, 90, 52, 26, 9.0, 5.5, 3.5, 2.0, KP]

LP_FILTER_CUTOFF_HZ = 1.2
LAT_ACCEL_REQUEST_BUFFER_SECONDS = 1.0
FRICTION_THRESHOLD = 0.3
VERSION = 3
LEARNED_MODEL_MIN_CONFIDENCE = 0.95
LEARNED_MODEL_MIN_COVERAGE = 0.5
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

ADAPTIVE_PHASE_MAP = {
  0: log.ControlsState.LateralTorqueState.AdaptiveTorqueState.Phase.idle,
  1: log.ControlsState.LateralTorqueState.AdaptiveTorqueState.Phase.engage,
  2: log.ControlsState.LateralTorqueState.AdaptiveTorqueState.Phase.hold,
  3: log.ControlsState.LateralTorqueState.AdaptiveTorqueState.Phase.release,
}


def sign(value: float) -> float:
  return 1.0 if value > 0.0 else (-1.0 if value < 0.0 else 0.0)


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
  CONTROL_STATE = "torque"

  def __init__(self, CP, CP_SP, CI, dt):
    super().__init__(CP, CP_SP, CI, dt)
    self.native_torque = CP.lateralTuning.which() == 'torque'
    if self.native_torque:
      self.torque_params = CP.lateralTuning.torque.as_builder()
      self.torque_from_lateral_accel = CI.torque_from_lateral_accel()
      self.lateral_accel_from_torque = CI.lateral_accel_from_torque()
      self.model_adapter = TorqueModelAdapter.native(self.torque_params, self.torque_from_lateral_accel, self.lateral_accel_from_torque)
    else:
      self.model_adapter = TorqueModelAdapter.synthetic()
      self.torque_params = self.model_adapter.params
      self.torque_from_lateral_accel = lambda lat_accel, _params: self.model_adapter.torque_from_lateral_accel(lat_accel)
      self.lateral_accel_from_torque = lambda torque, _params: self.model_adapter.lateral_accel_from_torque(torque)
    self.pid = PIDController([INTERP_SPEEDS, KP_INTERP], KI, KD, rate=1 / self.dt)
    self.update_limits()
    self.steering_angle_deadzone_deg = getattr(self.torque_params, "steeringAngleDeadzoneDeg", 0.0)
    self.lat_accel_request_buffer_len = int(LAT_ACCEL_REQUEST_BUFFER_SECONDS / self.dt)
    self.lat_accel_request_buffer = deque([0.0] * self.lat_accel_request_buffer_len, maxlen=self.lat_accel_request_buffer_len)
    self.previous_measurement = 0.0
    self.measurement_rate_filter = FirstOrderFilter(0.0, 1 / (2 * np.pi * LP_FILTER_CUTOFF_HZ), self.dt)
    self.measurement_smoother = LateralAccelMeasurementSmoother(self.dt)

    self.extension = LatControlTorqueExt(self, CP, CP_SP, CI)
    self._extension_update_model_v2 = self.extension.update_model_v2
    self.extension.update_model_v2 = self.update_model_v2
    self.response_assist = TorqueGuardedResponseAssist(self.dt)
    self.estimator = AdaptiveTorqueEstimator(self.dt)
    if self.native_torque:
      self.estimator.state.confidence = self.model_adapter.confidence
      self.estimator.state.params = TorqueModelParams(
        float(self.torque_params.latAccelFactor),
        float(self.torque_params.latAccelOffset),
        float(self.torque_params.friction),
      )
    self.authority_manager = AuthorityManager()
    self.safety_envelope = TorqueV3SafetyEnvelope(self.dt)
    self.lat_delay = 0.2
    self.model_age = 0.0

  def update_model_v2(self, model_v2):
    self._extension_update_model_v2(model_v2)
    self.model_age = 0.0

  def update_lateral_lag(self, lag):
    self.lat_delay = max(float(lag), self.dt)
    self.extension.update_lateral_lag(self.lat_delay)

  def update_live_torque_params(self, latAccelFactor, latAccelOffset, friction):
    if not self.native_torque or self.model_adapter.mode != TorqueModelMode.native:
      return
    self.torque_params.latAccelFactor = latAccelFactor
    self.torque_params.latAccelOffset = latAccelOffset
    self.torque_params.friction = friction
    self.update_limits()

  def update_limits(self):
    self.pid.set_limits(self.lateral_accel_from_torque(self.steer_max, self.torque_params), self.lateral_accel_from_torque(-self.steer_max, self.torque_params))

  def learned_model_ready(self, estimator_result) -> bool:
    return (
      estimator_result.confidence >= LEARNED_MODEL_MIN_CONFIDENCE
      and estimator_result.positive_coverage >= LEARNED_MODEL_MIN_COVERAGE
      and estimator_result.negative_coverage >= LEARNED_MODEL_MIN_COVERAGE
    )

  def authority_confidence(self, confidence: float, reject_reason: EstimatorRejectReason) -> float:
    if self.native_torque and not authority_fault_active(reject_reason):
      return max(float(confidence), float(self.model_adapter.confidence))
    return float(confidence)

  def update(self, active, CS, VM, params, steer_limited_by_safety, desired_curvature, calibrated_pose, curvature_limited, lat_delay):
    if self.native_torque and self.extension.update_override_torque_params(self.torque_params):
      self.update_limits()

    pid_log = log.ControlsState.LateralTorqueState.new_message()
    pid_log.version = VERSION

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
    ff += get_friction(error, lateral_accel_deadzone, FRICTION_THRESHOLD, self.torque_params)

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
      control_speed = max(CS.vEgo, LOW_SPEED_UNWIND_GAIN_SPEED) if same_sign_unwind else CS.vEgo
      output_lataccel = self.pid.update(pid_log.error, -measurement_rate, feedforward=ff, speed=control_speed, freeze_integrator=freeze_integrator)
      baseline_torque = self.torque_from_lateral_accel(output_lataccel, self.torque_params)
      learned_torque = self.model_adapter.torque_from_lateral_accel(output_lataccel)
      output_torque = 0.5 * baseline_torque + 0.5 * learned_torque

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
    assist_result = self.response_assist.update(
      GuardedResponseAssistInputs(
        active=active,
        v_ego=CS.vEgo,
        steering_pressed=CS.steeringPressed,
        steer_limited_by_safety=steer_limited_by_safety,
        curvature_limited=curvature_limited,
        saturated=saturated,
        max_output=self.steer_max,
        nominal_torque=output_torque,
        desired_lateral_accel=setpoint,
        actual_lateral_accel=measurement,
        desired_lateral_jerk=desired_lateral_jerk,
        actual_lateral_jerk=raw_actual_lateral_jerk,
        lookahead_lateral_jerk=self.extension.lookahead_lateral_jerk,
        desired_curvature=desired_curvature,
        tracking_torque_error=tracking_torque_error,
        lane_change_active=lane_change_active,
        same_sign_unwind=same_sign_unwind,
      )
    )
    output_torque = assist_result.output_torque if active else 0.0
    saturated = self.steer_max - abs(output_torque) < 1e-3
    authority_state = self.authority_manager.update(
      self.model_adapter.mode,
      self.authority_confidence(self.estimator.state.confidence, EstimatorRejectReason.NONE),
      self.estimator.state.positive_coverage,
      self.estimator.state.negative_coverage,
      EstimatorRejectReason.NONE,
    )
    same_sign_unwind_release = same_sign_unwind and sign(measurement) != 0.0 and sign(-output_torque) == sign(measurement)
    safety_result = self.safety_envelope.update(
      TorqueV3SafetyInputs(
        active=active,
        v_ego=CS.vEgo,
        steering_pressed=CS.steeringPressed,
        steer_limited_by_safety=steer_limited_by_safety,
        release_active=assist_result.release_active or authority_state.fallback_active,
        max_output=self.steer_max,
        unshaped_output=output_torque,
        desired_lateral_accel=setpoint,
        actual_lateral_accel=shaping_measurement,
        desired_lateral_jerk=desired_lateral_jerk,
        actual_lateral_jerk=raw_actual_lateral_jerk,
        lookahead_lateral_jerk=self.extension.lookahead_lateral_jerk,
        same_sign_unwind_release=same_sign_unwind_release,
        authority_scale=authority_state.scale,
        # The shaper sees pre-actuator torque; the actuator command is negated on return.
        steering_rate_deg=-CS.steeringRateDeg,
      )
    )
    shaping_result = safety_result.shaping_result
    output_torque = safety_result.output_torque
    estimator_result = self.estimator.update(
      TorqueObservation(
        active=active,
        v_ego=CS.vEgo,
        steering_pressed=CS.steeringPressed,
        steer_limited_by_safety=steer_limited_by_safety,
        curvature_limited=curvature_limited,
        saturated=saturated or safety_result.authority_limited or shaping_result.active,
        commanded_torque=output_torque,
        desired_lateral_accel=setpoint,
        actual_lateral_accel=measurement,
        actual_lateral_jerk=raw_actual_lateral_jerk,
        roll_compensation=roll_compensation,
        model_age=self.model_age,
      )
    )
    if self.learned_model_ready(estimator_result) and self.model_adapter.update_learned_params(estimator_result.params, estimator_result.confidence):
      self.torque_params = self.model_adapter.params
      self.extension.torque_params = self.torque_params
      self.model_adapter.set_residual(estimator_result.residual_error / max(float(estimator_result.params.lat_accel_factor), 1e-3))
      self.update_limits()
    authority_confidence = self.authority_confidence(estimator_result.confidence, estimator_result.reject_reason)
    authority_state = self.authority_manager.update(
      self.model_adapter.mode,
      authority_confidence,
      estimator_result.positive_coverage,
      estimator_result.negative_coverage,
      estimator_result.reject_reason,
    )
    same_frame_authority_limited = False
    if authority_state.scale < safety_result.authority_cap / max(self.steer_max, 1e-6):
      authority_cap = max(0.0, min(float(authority_state.scale), 1.0)) * self.steer_max
      recapped_output_torque = float(np.clip(output_torque, -authority_cap, authority_cap))
      same_frame_authority_limited = abs(recapped_output_torque) < abs(output_torque) - 1e-6
      output_torque = recapped_output_torque

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
      shaping_result.active or assist_result.phase_id != 0 or abs(assist_result.assist_torque) > 1e-3 or abs(assist_result.bias_torque) > 1e-3
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
    adaptive_log.modelMode = int(self.model_adapter.mode)
    adaptive_log.modelConfidence = float(authority_confidence)
    adaptive_log.authorityBand = int(authority_state.band)
    adaptive_log.authorityScale = float(authority_state.scale)
    adaptive_log.fallbackActive = bool(authority_state.fallback_active)
    adaptive_log.learnedLatAccelFactor = float(estimator_result.params.lat_accel_factor)
    adaptive_log.learnedFriction = float(estimator_result.params.friction)
    adaptive_log.learnedLatAccelOffset = float(estimator_result.params.lat_accel_offset)
    adaptive_log.learnedResponseDelay = float(estimator_result.response_delay)
    adaptive_log.residualError = float(estimator_result.residual_error)
    adaptive_log.sampleAccepted = bool(estimator_result.sample_accepted)
    adaptive_log.sampleRejectReason = int(estimator_result.reject_reason)
    saturation_check = safety_result.authority_limited or same_frame_authority_limited or self.steer_max - abs(output_torque) < 1e-3
    pid_log.saturated = bool(self._check_saturation(saturation_check, CS, steer_limited_by_safety, curvature_limited))
    self.model_age += self.dt

    return -output_torque, 0.0, pid_log


LatControlTorqueV3 = LatControlTorque
