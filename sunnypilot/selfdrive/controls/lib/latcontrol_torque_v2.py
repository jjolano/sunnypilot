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
from openpilot.sunnypilot.selfdrive.controls.lib.torque_authority_envelope import EnvelopeInputs, TorqueAuthorityEnvelope


KP = 1.0
KI = 0.3
KD = 0.0
INTERP_SPEEDS = [1, 1.5, 2.0, 3.0, 5, 7.5, 10, 15, 30]
KP_INTERP = [250, 120, 65, 30, 11.5, 5.5, 3.5, 2.0, KP]

LP_FILTER_CUTOFF_HZ = 1.2
LAT_ACCEL_REQUEST_BUFFER_SECONDS = 1.0
FRICTION_THRESHOLD = 0.3
VERSION = 2

V2_PHASE_MAP = {
  0: log.ControlsState.LateralTorqueState.V2Phase.idle,
  1: log.ControlsState.LateralTorqueState.V2Phase.rampIn,
  2: log.ControlsState.LateralTorqueState.V2Phase.hold,
  3: log.ControlsState.LateralTorqueState.V2Phase.taperOut,
}


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
    self.authority_envelope = TorqueAuthorityEnvelope(self.dt)

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
    ff = gravity_adjusted_future_lateral_accel
    ff -= self.torque_params.latAccelOffset
    ff += get_friction(error, lateral_accel_deadzone, FRICTION_THRESHOLD, self.torque_params)

    output_torque = 0.0
    if not active:
      pid_log.active = False
      self.pid.reset()
    else:
      pid_log.error = float(error)
      freeze_integrator = steer_limited_by_safety or CS.steeringPressed or CS.vEgo < 5
      output_lataccel = self.pid.update(pid_log.error, -measurement_rate, feedforward=ff, speed=CS.vEgo, freeze_integrator=freeze_integrator)
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

    lane_change_active = bool(self.extension.model_valid and self.extension.model_v2.meta.laneChangeState != log.LaneChangeState.off)
    saturated = self.steer_max - abs(output_torque) < 1e-3
    tracking_torque_error = error / max(float(self.torque_params.latAccelFactor), 1e-3)
    envelope_result = self.authority_envelope.update(
      EnvelopeInputs(
        active=active,
        v_ego=CS.vEgo,
        steering_pressed=CS.steeringPressed,
        steer_limited_by_safety=steer_limited_by_safety,
        curvature_limited=curvature_limited,
        saturated=saturated,
        nominal_torque=output_torque,
        desired_lateral_accel=setpoint,
        actual_lateral_accel=measurement,
        desired_lateral_jerk=desired_lateral_jerk,
        actual_lateral_jerk=self.extension.actual_lateral_jerk,
        lookahead_lateral_jerk=self.extension.lookahead_lateral_jerk,
        tracking_torque_error=tracking_torque_error,
        lane_change_active=lane_change_active,
      )
    )
    output_torque = envelope_result.output_torque if active else 0.0

    pid_log.p = float(self.pid.p)
    pid_log.i = float(self.pid.i)
    pid_log.d = float(self.pid.d)
    pid_log.f = float(self.pid.f)
    pid_log.output = float(-output_torque)
    pid_log.actualLateralAccel = float(measurement)
    pid_log.desiredLateralAccel = float(setpoint)
    pid_log.desiredLateralJerk = float(desired_lateral_jerk)
    pid_log.v2Phase = V2_PHASE_MAP[envelope_result.phase_id]
    pid_log.v2PhaseGain = float(envelope_result.phase_gain)
    pid_log.v2AuthorityFloor = float(envelope_result.authority_floor)
    pid_log.v2DisturbanceBias = float(envelope_result.disturbance_bias)
    pid_log.v2NominalOutput = float(-envelope_result.nominal_torque)
    pid_log.v2LearningFrozen = bool(envelope_result.learning_frozen)
    pid_log.saturated = bool(self._check_saturation(self.steer_max - abs(output_torque) < 1e-3, CS, steer_limited_by_safety, curvature_limited))

    return -output_torque, 0.0, pid_log
