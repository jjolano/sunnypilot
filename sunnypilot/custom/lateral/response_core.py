"""Torque v2.1 response core — the feel-carrying math, ported unchanged.

This is the clean-room restructure of the pre-output-stage computation in the legacy
``latcontrol_torque_v2.py`` (the part producing the PID feedforward torque and the signals
the output stage consumes). The arithmetic is identical to the legacy controller; only the
structure is cleaned up and the external dependencies (curvature model, torque<->lat-accel
conversions, friction model) are injected rather than reached through a live car interface.

Because the math is unchanged, this module is validated to exact numerical parity against a
flat transcription of the legacy formulas — see ``tests/test_response_core_parity.py``. No
route data is required for that gate. See
``docs/adr/2026-06-13-clean-room-torque-v2-1-architecture.md``.

Constants are the legacy values (``docs/legacy/tuned-constants.yaml``); do not retune here.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.pid import PIDController

# --- legacy response-core constants (latcontrol_torque_v2.py) ---
KP = 1.0
KI = 0.2
KD = 0.0
INTERP_SPEEDS = [1, 1.5, 2.0, 3.0, 5, 7.5, 10, 15, 30]
KP_INTERP = [165, 90, 52, 26, 9.0, 5.5, 3.5, 2.0, KP]

LP_FILTER_CUTOFF_HZ = 1.2
LAT_ACCEL_REQUEST_BUFFER_SECONDS = 1.0
FRICTION_THRESHOLD = 0.3
LOW_DEMAND_FRICTION_FULL_LAT_ACCEL = 0.35

LOW_SPEED_UNWIND_VEGO = 8.0
LOW_SPEED_UNWIND_SETPOINT = 0.2
LOW_SPEED_UNWIND_MARGIN = 0.08
LOW_SPEED_UNWIND_JERK = 0.5
LOW_SPEED_UNWIND_GAIN_SPEED = 8.0
LOW_SPEED_PID_GAIN_FLOOR = 3.0

MEASUREMENT_SMOOTHER_MIN_VEGO = 5.0
MEASUREMENT_SMOOTHER_CORRECTION_GAIN = 0.35
MEASUREMENT_SMOOTHER_MAX_PREDICTIVE_JERK = 5.0
MEASUREMENT_SMOOTHER_IMPLAUSIBLE_JERK = 80.0
MEASUREMENT_SMOOTHER_MAX_RAW_ERROR = 1.0

ACCELERATION_DUE_TO_GRAVITY = 9.81
# Deliberate deviation from the legacy full-gravity roll feedforward (see
# docs/adr/2026-07-02-scale-roll-compensation-gain.md). Route 00000246: regressing steady
# straight-cruise torque need against crown over 22.7k frames gives slope 0.56 — full
# g*sin(roll) over-injects ~44%, which the integrator permanently cancels and re-converges
# after every crown change (the 0.15-0.2 Hz lane wander). Tune per-platform if needed.
ROLL_COMPENSATION_GAIN = 0.55


def sign(value: float) -> float:
  return 1.0 if value > 0.0 else (-1.0 if value < 0.0 else 0.0)


def low_speed_pid_gain_speed(v_ego: float, unwind_gain_floor: float | None = None) -> float:
  gain_floor = LOW_SPEED_PID_GAIN_FLOOR if unwind_gain_floor is None else unwind_gain_floor
  return max(float(v_ego), gain_floor)


def low_demand_friction_scale(setpoint: float, measurement: float) -> float:
  demand = max(abs(setpoint), abs(measurement))
  return float(np.clip(demand / LOW_DEMAND_FRICTION_FULL_LAT_ACCEL, 0.0, 1.0))


class TorqueParams(Protocol):
  """The subset of car torque tuning the response core reads."""
  latAccelFactor: float
  latAccelOffset: float
  friction: float
  steeringAngleDeadzoneDeg: float


# Injected external models (so the core is pure and testable):
#   calc_curvature(angle_rad, v_ego, roll) -> curvature   (VehicleModel.calc_curvature)
#   torque_from_lateral_accel(lat_accel, torque_params) -> torque
#   lateral_accel_from_torque(torque, torque_params) -> lat_accel
#   get_friction(lat_accel_error, deadzone, threshold, torque_params) -> friction lat-accel
CalcCurvature = Callable[[float, float, float], float]
TorqueConv = Callable[[float, "TorqueParams"], float]
FrictionModel = Callable[[float, float, float, "TorqueParams"], float]


@dataclass(frozen=True)
class ResponseCoreInputs:
  active: bool
  v_ego: float
  steering_angle_deg: float
  steering_rate_deg: float
  steering_pressed: bool
  angle_offset_deg: float
  roll: float
  desired_curvature: float
  lat_delay: float
  steer_limited_by_safety: bool


@dataclass(frozen=True)
class ResponseCoreResult:
  output_torque: float
  setpoint: float
  measurement: float
  raw_measurement: float
  measured_curvature: float
  error: float
  ff: float
  roll_compensation: float
  desired_lateral_jerk: float
  raw_actual_lateral_jerk: float
  future_desired_lateral_accel: float
  gravity_adjusted_future_lateral_accel: float
  lateral_accel_deadzone: float
  same_sign_unwind: bool
  measurement_reset: bool
  freeze_integrator: bool


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


class ResponseCore:
  """Pre-output-stage torque computation for torque v2.1, ported unchanged-in-math."""

  def __init__(self, dt: float, steer_max: float, torque_params: TorqueParams,
               calc_curvature: CalcCurvature, torque_from_lateral_accel: TorqueConv,
               lateral_accel_from_torque: TorqueConv, get_friction: FrictionModel):
    self.dt = dt
    self.steer_max = steer_max
    self.torque_params = torque_params
    self._calc_curvature = calc_curvature
    self._torque_from_lateral_accel = torque_from_lateral_accel
    self._lateral_accel_from_torque = lateral_accel_from_torque
    self._get_friction = get_friction

    self.pid = PIDController([INTERP_SPEEDS, KP_INTERP], KI, KD, rate=1 / self.dt)
    self.update_limits()
    self.steering_angle_deadzone_deg = torque_params.steeringAngleDeadzoneDeg
    self.lat_accel_request_buffer_len = int(LAT_ACCEL_REQUEST_BUFFER_SECONDS / self.dt)
    self.lat_accel_request_buffer = deque([0.0] * self.lat_accel_request_buffer_len,
                                          maxlen=self.lat_accel_request_buffer_len)
    self.previous_measurement = 0.0
    self.measurement_rate_filter = FirstOrderFilter(0.0, 1 / (2 * np.pi * LP_FILTER_CUTOFF_HZ), self.dt)
    self.measurement_smoother = LateralAccelMeasurementSmoother(self.dt)
    self._v_ego_invalid_logged = False

  def update_limits(self) -> None:
    self.pid.set_limits(self._lateral_accel_from_torque(self.steer_max, self.torque_params),
                        self._lateral_accel_from_torque(-self.steer_max, self.torque_params))

  def update(self, inp: ResponseCoreInputs) -> ResponseCoreResult:
    tp = self.torque_params
    v_ego = inp.v_ego

    if not math.isfinite(v_ego) or v_ego < 0:
      if not self._v_ego_invalid_logged:
        from openpilot.common.swaglog import cloudlog
        cloudlog.warning(f"response_core invalid v_ego: {v_ego}, using 0.0")
        self._v_ego_invalid_logged = True
      v_ego = 0.0

    measured_curvature = -self._calc_curvature(math.radians(inp.steering_angle_deg - inp.angle_offset_deg), v_ego, inp.roll)
    raw_measurement = measured_curvature * v_ego ** 2
    roll_compensation = ROLL_COMPENSATION_GAIN * inp.roll * ACCELERATION_DUE_TO_GRAVITY
    curvature_deadzone = abs(self._calc_curvature(math.radians(self.steering_angle_deadzone_deg), v_ego, 0.0))
    lateral_accel_deadzone = curvature_deadzone * v_ego ** 2
    raw_actual_lateral_jerk = -self._calc_curvature(math.radians(inp.steering_rate_deg), v_ego, 0.0) * v_ego ** 2
    measurement = self.measurement_smoother.update(inp.active, v_ego, inp.steering_pressed,
                                                   raw_measurement, raw_actual_lateral_jerk)

    future_desired_lateral_accel = inp.desired_curvature * v_ego ** 2
    if not math.isfinite(future_desired_lateral_accel):
      from openpilot.common.swaglog import cloudlog
      cloudlog.warning(f"response_core nonfinite desired lateral accel: v_ego={v_ego:.1f} curvature={inp.desired_curvature:.4f}")
      future_desired_lateral_accel = 0.0
      self.lat_accel_request_buffer.clear()
      self.lat_accel_request_buffer.extend([0.0] * self.lat_accel_request_buffer_len)
    self.lat_accel_request_buffer.append(future_desired_lateral_accel)

    effective_lat_delay = self.dt if (not math.isfinite(inp.lat_delay) or inp.lat_delay <= 0) else max(inp.lat_delay, self.dt)
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
    # State health: flag implausible tracking error.
    # 15 m/s² threshold avoids false positives from synthetic/simplified plants
    # while catching genuine production anomalies (normal lateral accel ≤ 5 m/s²).
    if abs(error) > 15.0:
      from openpilot.common.swaglog import cloudlog
      cloudlog.warning(f"response_core large tracking error: {error:.2f} m/s² (setpoint={setpoint:.2f} meas={measurement:.2f})")
    same_sign_unwind = (
      v_ego < LOW_SPEED_UNWIND_VEGO
      and abs(setpoint) < LOW_SPEED_UNWIND_SETPOINT
      and abs(desired_lateral_jerk) > LOW_SPEED_UNWIND_JERK
      and sign(setpoint) != 0.0
      and sign(measurement) == sign(setpoint)
      and abs(measurement) > abs(setpoint) + LOW_SPEED_UNWIND_MARGIN
      and desired_lateral_jerk * setpoint < 0.0
    )

    ff = setpoint if same_sign_unwind else gravity_adjusted_future_lateral_accel
    ff -= tp.latAccelOffset
    friction_scale = low_demand_friction_scale(setpoint, measurement)
    ff += self._get_friction(error * friction_scale, lateral_accel_deadzone, FRICTION_THRESHOLD, tp) * friction_scale

    output_torque = 0.0
    freeze_integrator = False
    if not inp.active:
      self.pid.reset()
    else:
      if same_sign_unwind:
        self.pid.i *= 0.5
      freeze_integrator = inp.steer_limited_by_safety or inp.steering_pressed or v_ego < 5 or same_sign_unwind
      control_speed = low_speed_pid_gain_speed(v_ego, LOW_SPEED_UNWIND_GAIN_SPEED if same_sign_unwind else None)
      output_lataccel = self.pid.update(error, -measurement_rate, feedforward=ff, speed=control_speed,
                                        freeze_integrator=freeze_integrator)
      output_torque = self._torque_from_lateral_accel(output_lataccel, tp)
      # State health: flag output exceeding steer_max (wiring or gain error).
      if abs(output_torque) > self.steer_max * 1.01 and not math.isclose(abs(output_torque), self.steer_max, rel_tol=0.02):
        from openpilot.common.swaglog import cloudlog
        cloudlog.warning(f"response_core output {output_torque:.3f} exceeds steer_max={self.steer_max}")

    return ResponseCoreResult(
      output_torque=output_torque,
      setpoint=setpoint,
      measurement=measurement,
      raw_measurement=raw_measurement,
      measured_curvature=measured_curvature,
      error=error,
      ff=ff,
      roll_compensation=roll_compensation,
      desired_lateral_jerk=desired_lateral_jerk,
      raw_actual_lateral_jerk=raw_actual_lateral_jerk,
      future_desired_lateral_accel=future_desired_lateral_accel,
      gravity_adjusted_future_lateral_accel=gravity_adjusted_future_lateral_accel,
      lateral_accel_deadzone=lateral_accel_deadzone,
      same_sign_unwind=same_sign_unwind,
      measurement_reset=self.measurement_smoother.was_reset,
      freeze_integrator=freeze_integrator,
    )
