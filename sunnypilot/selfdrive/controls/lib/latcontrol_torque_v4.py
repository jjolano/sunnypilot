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
from openpilot.sunnypilot.selfdrive.controls.lib.torque_residual_adapter import ResidualAdapterInputs, TorqueResidualAdapter


KP = 1.0
KI = 0.2
KD = 0.0
INTERP_SPEEDS = [1, 1.5, 2.0, 3.0, 5, 7.5, 10, 15, 30]
KP_INTERP = [165, 90, 52, 26, 9.0, 5.5, 3.5, 2.0, KP]

LP_FILTER_CUTOFF_HZ = 1.2
LAT_ACCEL_REQUEST_BUFFER_SECONDS = 1.0
FRICTION_THRESHOLD = 0.3
VERSION = 4
LOW_SPEED_UNWIND_VEGO = 8.0
LOW_SPEED_UNWIND_SETPOINT = 0.2
LOW_SPEED_UNWIND_MARGIN = 0.08
LOW_SPEED_UNWIND_JERK = 0.5
LOW_SPEED_UNWIND_GAIN_SPEED = 8.0

ADAPTIVE_PHASE_MAP = {
  0: log.ControlsState.LateralTorqueState.AdaptiveTorqueState.Phase.idle,
  1: log.ControlsState.LateralTorqueState.AdaptiveTorqueState.Phase.engage,
  2: log.ControlsState.LateralTorqueState.AdaptiveTorqueState.Phase.hold,
  3: log.ControlsState.LateralTorqueState.AdaptiveTorqueState.Phase.release,
}


def sign(value: float) -> float:
  return 1.0 if value > 0.0 else (-1.0 if value < 0.0 else 0.0)


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

    measured_curvature = -VM.calc_curvature(math.radians(CS.steeringAngleDeg - params.angleOffsetDeg), CS.vEgo, params.roll)
    measurement = measured_curvature * CS.vEgo**2
    roll_compensation = params.roll * ACCELERATION_DUE_TO_GRAVITY
    curvature_deadzone = abs(VM.calc_curvature(math.radians(self.steering_angle_deadzone_deg), CS.vEgo, 0.0))
    lateral_accel_deadzone = curvature_deadzone * CS.vEgo**2

    future_desired_lateral_accel = desired_curvature * CS.vEgo**2
    self.lat_accel_request_buffer.append(future_desired_lateral_accel)
    effective_lat_delay = max(lat_delay, self.dt)
    delay_frames = int(np.clip(effective_lat_delay / self.dt, 1, self.lat_accel_request_buffer_len))
    expected_lateral_accel = self.lat_accel_request_buffer[-delay_frames]
    gravity_adjusted_future_lateral_accel = future_desired_lateral_accel - roll_compensation
    desired_lateral_jerk = (future_desired_lateral_accel - expected_lateral_accel) / effective_lat_delay

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

      pid_log.active = True

    saturated = self.steer_max - abs(output_torque) < 1e-3
    tracking_torque_error = error / max(float(self.torque_params.latAccelFactor), 1e-3)
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
        desired_lateral_accel=setpoint,
        actual_lateral_accel=measurement,
        desired_lateral_jerk=desired_lateral_jerk,
        actual_lateral_jerk=self.extension.actual_lateral_jerk,
        lookahead_lateral_jerk=self.extension.lookahead_lateral_jerk,
        desired_curvature=desired_curvature,
        tracking_torque_error=tracking_torque_error,
        lane_change_active=lane_change_active,
        same_sign_unwind=same_sign_unwind,
      )
    )
    output_torque = residual_result.output_torque if active else 0.0

    pid_log.p = float(self.pid.p)
    pid_log.i = float(self.pid.i)
    pid_log.d = float(self.pid.d)
    pid_log.f = float(self.pid.f)
    pid_log.output = float(-output_torque)
    pid_log.actualLateralAccel = float(measurement)
    pid_log.desiredLateralAccel = float(setpoint)
    pid_log.desiredLateralJerk = float(desired_lateral_jerk)
    adaptive_log = pid_log.init('adaptiveTorqueState')
    adaptive_log.active = bool(
      active and (residual_result.phase_id != 0 or abs(residual_result.assist_torque) > 1e-3 or abs(residual_result.bias_torque) > 1e-3)
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

    return -output_torque, 0.0, pid_log
