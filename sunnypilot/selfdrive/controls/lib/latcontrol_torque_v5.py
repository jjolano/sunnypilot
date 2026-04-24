import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from cereal import log
from opendbc.car.lateral import get_friction
from openpilot.common.constants import ACCELERATION_DUE_TO_GRAVITY
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.pid import PIDController
from openpilot.selfdrive.controls.lib.latcontrol import LatControl

from openpilot.sunnypilot.selfdrive.controls.lib.latcontrol_torque_ext import LatControlTorqueExt
from openpilot.sunnypilot.selfdrive.controls.lib.torque_residual_adapter import ResidualAdapterInputs, TorqueResidualAdapter


KP = 0.9
KI = 0.16
KD = 0.035
INTERP_SPEEDS = [1, 1.5, 2.0, 3.0, 5, 7.5, 10, 15, 30]
KP_INTERP = [130, 74, 42, 22, 8.0, 5.0, 3.0, 1.8, KP]

LP_FILTER_CUTOFF_HZ = 1.2
LAT_ACCEL_REQUEST_BUFFER_SECONDS = 1.2
FRICTION_THRESHOLD = 0.3
FRICTION_JERK_GAIN = 0.08
VERSION = 5

REFERENCE_BLEND_SPEED_BP = [3.0, 8.0, 15.0]
REFERENCE_BLEND_V = [0.45, 0.75, 1.0]
LOW_SPEED_UNWIND_VEGO = 8.0
LOW_SPEED_UNWIND_SETPOINT = 0.2
LOW_SPEED_UNWIND_MARGIN = 0.08
LOW_SPEED_UNWIND_JERK = 0.5
LOW_SPEED_UNWIND_GAIN_SPEED = 8.0
SIGN_CHANGE_INTEGRATOR_DECAY = 0.6
UNWIND_INTEGRATOR_DECAY = 0.45

ADAPTIVE_PHASE_MAP = {
  0: log.ControlsState.LateralTorqueState.AdaptiveTorqueState.Phase.idle,
  1: log.ControlsState.LateralTorqueState.AdaptiveTorqueState.Phase.engage,
  2: log.ControlsState.LateralTorqueState.AdaptiveTorqueState.Phase.hold,
  3: log.ControlsState.LateralTorqueState.AdaptiveTorqueState.Phase.release,
}


def sign(value: float) -> float:
  return 1.0 if value > 0.0 else (-1.0 if value < 0.0 else 0.0)


@dataclass
class TorqueMeasurement:
  measured_curvature: float
  lateral_accel: float
  lateral_accel_rate: float
  roll_compensation: float
  lateral_accel_deadzone: float


@dataclass
class TorqueReference:
  future_lateral_accel: float
  delayed_lateral_accel: float
  setpoint: float
  feedforward_lateral_accel: float
  desired_lateral_jerk: float
  same_sign_unwind: bool


class LatControlTorque(LatControl):
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
    self.previous_setpoint_sign = 0.0
    self.measurement_rate_filter = FirstOrderFilter(0.0, 1 / (2 * np.pi * LP_FILTER_CUTOFF_HZ), self.dt)

    self.extension = LatControlTorqueExt(self, CP, CP_SP, CI)
    self.residual_adapter = TorqueResidualAdapter(self.dt)

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
    pid_log.version = VERSION

    measurement = self._estimate_measurement(CS, VM, params)
    reference = self._build_reference(CS, desired_curvature, measurement, lat_delay)
    output_torque = 0.0

    if not active:
      pid_log.active = False
      self.pid.reset()
    else:
      output_torque = self._update_nominal_feedback(CS, reference, measurement, steer_limited_by_safety, pid_log)
      pid_log, output_torque = self.extension.update(
        CS,
        VM,
        self.pid,
        params,
        self.pid.f,
        pid_log,
        reference.setpoint,
        measurement.lateral_accel,
        calibrated_pose,
        measurement.roll_compensation,
        reference.setpoint,
        measurement.lateral_accel,
        measurement.lateral_accel_deadzone,
        reference.feedforward_lateral_accel,
        desired_curvature,
        measurement.measured_curvature,
        steer_limited_by_safety,
        output_torque,
      )
      pid_log.active = True

    output_torque, residual_result = self._apply_residual(
      active,
      CS,
      steer_limited_by_safety,
      curvature_limited,
      desired_curvature,
      reference,
      measurement,
      output_torque,
    )

    self._write_log(pid_log, reference, measurement, output_torque, residual_result, CS, steer_limited_by_safety, curvature_limited)
    return -output_torque, 0.0, pid_log

  def _estimate_measurement(self, CS, VM, params) -> TorqueMeasurement:
    measured_curvature = -VM.calc_curvature(math.radians(CS.steeringAngleDeg - params.angleOffsetDeg), CS.vEgo, params.roll)
    lateral_accel = measured_curvature * CS.vEgo**2
    lateral_accel_rate = self.measurement_rate_filter.update((lateral_accel - self.previous_measurement) / self.dt)
    self.previous_measurement = lateral_accel

    curvature_deadzone = abs(VM.calc_curvature(math.radians(self.steering_angle_deadzone_deg), CS.vEgo, 0.0))
    return TorqueMeasurement(
      measured_curvature=measured_curvature,
      lateral_accel=lateral_accel,
      lateral_accel_rate=lateral_accel_rate,
      roll_compensation=params.roll * ACCELERATION_DUE_TO_GRAVITY,
      lateral_accel_deadzone=curvature_deadzone * CS.vEgo**2,
    )

  def _build_reference(self, CS, desired_curvature: float, measurement: TorqueMeasurement, lat_delay: float) -> TorqueReference:
    future_lateral_accel = desired_curvature * CS.vEgo**2
    self.lat_accel_request_buffer.append(future_lateral_accel)

    effective_lat_delay = max(lat_delay, self.dt)
    delay_frames = int(np.clip(effective_lat_delay / self.dt, 1, self.lat_accel_request_buffer_len))
    delayed_lateral_accel = self.lat_accel_request_buffer[-delay_frames]
    desired_lateral_jerk = (future_lateral_accel - delayed_lateral_accel) / effective_lat_delay

    reference_blend = float(np.interp(CS.vEgo, REFERENCE_BLEND_SPEED_BP, REFERENCE_BLEND_V))
    setpoint = delayed_lateral_accel + reference_blend * (future_lateral_accel - delayed_lateral_accel)
    same_sign_unwind = self._same_sign_unwind(CS, future_lateral_accel, measurement.lateral_accel, desired_lateral_jerk)
    if same_sign_unwind:
      setpoint = future_lateral_accel

    return TorqueReference(
      future_lateral_accel=future_lateral_accel,
      delayed_lateral_accel=delayed_lateral_accel,
      setpoint=setpoint,
      feedforward_lateral_accel=setpoint - measurement.roll_compensation,
      desired_lateral_jerk=desired_lateral_jerk,
      same_sign_unwind=same_sign_unwind,
    )

  @staticmethod
  def _same_sign_unwind(CS, future_lateral_accel: float, actual_lateral_accel: float, desired_lateral_jerk: float) -> bool:
    return (
      CS.vEgo < LOW_SPEED_UNWIND_VEGO
      and abs(future_lateral_accel) < LOW_SPEED_UNWIND_SETPOINT
      and abs(desired_lateral_jerk) > LOW_SPEED_UNWIND_JERK
      and sign(future_lateral_accel) != 0.0
      and sign(actual_lateral_accel) == sign(future_lateral_accel)
      and abs(actual_lateral_accel) > abs(future_lateral_accel) + LOW_SPEED_UNWIND_MARGIN
      and desired_lateral_jerk * future_lateral_accel < 0.0
    )

  def _update_nominal_feedback(self, CS, reference: TorqueReference, measurement: TorqueMeasurement, steer_limited_by_safety: bool, pid_log) -> float:
    error = reference.setpoint - measurement.lateral_accel
    pid_log.error = float(error)

    setpoint_sign = sign(reference.setpoint)
    if reference.same_sign_unwind:
      self.pid.i *= UNWIND_INTEGRATOR_DECAY
    elif setpoint_sign != 0.0 and self.previous_setpoint_sign != 0.0 and setpoint_sign != self.previous_setpoint_sign:
      self.pid.i *= SIGN_CHANGE_INTEGRATOR_DECAY
    if setpoint_sign != 0.0:
      self.previous_setpoint_sign = setpoint_sign

    friction_input = error + FRICTION_JERK_GAIN * reference.desired_lateral_jerk
    ff = reference.feedforward_lateral_accel
    ff -= self.torque_params.latAccelOffset
    ff += get_friction(friction_input, measurement.lateral_accel_deadzone, FRICTION_THRESHOLD, self.torque_params)

    freeze_integrator = steer_limited_by_safety or CS.steeringPressed or CS.vEgo < 5 or reference.same_sign_unwind
    control_speed = max(CS.vEgo, LOW_SPEED_UNWIND_GAIN_SPEED) if reference.same_sign_unwind else CS.vEgo
    output_lataccel = self.pid.update(error, -measurement.lateral_accel_rate, feedforward=ff, speed=control_speed, freeze_integrator=freeze_integrator)
    return self.torque_from_lateral_accel(output_lataccel, self.torque_params)

  def _apply_residual(
    self,
    active: bool,
    CS,
    steer_limited_by_safety: bool,
    curvature_limited: bool,
    desired_curvature: float,
    reference: TorqueReference,
    measurement: TorqueMeasurement,
    output_torque: float,
  ):
    saturated = self.steer_max - abs(output_torque) < 1e-3
    tracking_torque_error = (reference.setpoint - measurement.lateral_accel) / max(float(self.torque_params.latAccelFactor), 1e-3)
    lane_change_active = bool(self.extension.model_valid and self.extension.model_v2.meta.laneChangeState != log.LaneChangeState.off)
    residual_result = self.residual_adapter.update(
      ResidualAdapterInputs(
        active=active,
        v_ego=CS.vEgo,
        steering_pressed=CS.steeringPressed,
        steer_limited_by_safety=steer_limited_by_safety,
        curvature_limited=curvature_limited,
        saturated=saturated,
        max_output=self.steer_max,
        nominal_torque=output_torque,
        desired_lateral_accel=reference.setpoint,
        actual_lateral_accel=measurement.lateral_accel,
        desired_lateral_jerk=reference.desired_lateral_jerk,
        actual_lateral_jerk=self.extension.actual_lateral_jerk,
        lookahead_lateral_jerk=self.extension.lookahead_lateral_jerk,
        desired_curvature=desired_curvature,
        tracking_torque_error=tracking_torque_error,
        lane_change_active=lane_change_active,
        same_sign_unwind=reference.same_sign_unwind,
      )
    )
    return (residual_result.output_torque if active else 0.0), residual_result

  def _write_log(
    self,
    pid_log,
    reference: TorqueReference,
    measurement: TorqueMeasurement,
    output_torque: float,
    residual_result,
    CS,
    steer_limited_by_safety,
    curvature_limited,
  ):
    pid_log.p = float(self.pid.p)
    pid_log.i = float(self.pid.i)
    pid_log.d = float(self.pid.d)
    pid_log.f = float(self.pid.f)
    pid_log.output = float(-output_torque)
    pid_log.actualLateralAccel = float(measurement.lateral_accel)
    pid_log.desiredLateralAccel = float(reference.setpoint)
    pid_log.desiredLateralJerk = float(reference.desired_lateral_jerk)

    adaptive_log = pid_log.init('adaptiveTorqueState')
    adaptive_log.active = bool(
      pid_log.active and (residual_result.phase_id != 0 or abs(residual_result.assist_torque) > 1e-3 or abs(residual_result.bias_torque) > 1e-3)
    )
    adaptive_log.phase = ADAPTIVE_PHASE_MAP[residual_result.phase_id]
    adaptive_log.releaseActive = bool(residual_result.release_active)
    adaptive_log.phaseGain = float(residual_result.phase_gain)
    adaptive_log.nominalOutput = float(-residual_result.nominal_torque)
    adaptive_log.assistOutput = float(-residual_result.assist_torque)
    adaptive_log.biasOutput = float(-residual_result.bias_torque)
    adaptive_log.responseDeficit = float(residual_result.response_deficit)
    adaptive_log.learningFrozen = bool(residual_result.learning_frozen)
    pid_log.saturated = bool(self._check_saturation(self.steer_max - abs(output_torque) < 1e-3, CS, steer_limited_by_safety, curvature_limited))
