from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass


MIN_V_EGO = 10.0                    # m/s
MIN_ABS_SETPOINT = 0.45             # m/s^2
MIN_ABS_ERROR = 0.25                # m/s^2
ERROR_FRACTION = 0.25               # ratio of abs(setpoint)
EWMA_TAU = 0.25                     # s
SOFT_DECAY_TAU = 0.30               # s
DESIRED_PERSISTENCE_TIME = 0.25     # s
TRIGGER_TIME = 0.20                 # s
MAX_DESIRED_JERK_FOR_STABLE = 4.0   # m/s^3
MAX_ABS_ROLL = 0.08                 # rad
MAX_ROLL_RATE = 0.08                # rad/s
MAX_OUTPUT_TORQUE_FRAC = 0.90       # normalized torque fraction
MIN_CLOSING_RATE = 0.50             # m/s^3
SHADOW_CORR_GAIN = 0.50
MAX_SHADOW_LAT_ACCEL = 0.35         # m/s^2
ACTUAL_OPPOSING_MIN_ABS = 0.20      # m/s^2

BLOCK_INACTIVE = 1 << 0
BLOCK_LOW_SPEED = 1 << 1
BLOCK_STEERING_PRESSED = 1 << 2
BLOCK_STEER_LIMITED = 1 << 3
BLOCK_CURVATURE_LIMITED = 1 << 4
BLOCK_TORQUE_SATURATED = 1 << 5
BLOCK_ROLL_TOO_HIGH = 1 << 6
BLOCK_ROLL_UNSTABLE = 1 << 7
BLOCK_DESIRED_TOO_SMALL = 1 << 8
BLOCK_DESIRED_NOT_PERSISTENT = 1 << 9
BLOCK_ACTUAL_OPPOSING = 1 << 10
BLOCK_SIGN_MISMATCH = 1 << 11
BLOCK_ERROR_TOO_SMALL = 1 << 12
BLOCK_FAST_CLOSING = 1 << 13
BLOCK_INVALID_INPUT = 1 << 14

BLOCK_NAMES = {
  BLOCK_INACTIVE: "inactive",
  BLOCK_LOW_SPEED: "low_speed",
  BLOCK_STEERING_PRESSED: "steering_pressed",
  BLOCK_STEER_LIMITED: "steer_limited",
  BLOCK_CURVATURE_LIMITED: "curvature_limited",
  BLOCK_TORQUE_SATURATED: "torque_saturated",
  BLOCK_ROLL_TOO_HIGH: "roll_too_high",
  BLOCK_ROLL_UNSTABLE: "roll_unstable",
  BLOCK_DESIRED_TOO_SMALL: "desired_too_small",
  BLOCK_DESIRED_NOT_PERSISTENT: "desired_not_persistent",
  BLOCK_ACTUAL_OPPOSING: "actual_opposing",
  BLOCK_SIGN_MISMATCH: "sign_mismatch",
  BLOCK_ERROR_TOO_SMALL: "error_too_small",
  BLOCK_FAST_CLOSING: "fast_closing",
  BLOCK_INVALID_INPUT: "invalid_input",
}


@dataclass
class UnderresponseDebug:
  active: bool = False
  eligible: bool = False
  block_mask: int = 0
  error: float = 0.0                 # m/s^2, setpoint - measurement
  error_filtered: float = 0.0        # m/s^2, EWMA of error
  duration: float = 0.0              # s, consecutive above-threshold duration
  closing_rate: float = 0.0          # m/s^3, positive means deficit is shrinking
  shadow_lat_accel: float = 0.0      # m/s^2, hypothetical correction, not applied
  severity: float = 0.0              # 0..1 heuristic


def _sign(x: float, eps: float = 1e-6) -> int:
  if x > eps:
    return 1
  if x < -eps:
    return -1
  return 0


def _clip(x: float, lo: float, hi: float) -> float:
  return min(max(x, lo), hi)


def _finite(*vals: float) -> bool:
  return all(math.isfinite(float(v)) for v in vals)


class UnderresponseSentinel:
  def __init__(self, dt: float):
    self.dt = max(float(dt), 1e-6)
    self.history_len = max(2, int(DESIRED_PERSISTENCE_TIME / self.dt))
    self.desired_history: deque[float] = deque(maxlen=self.history_len)
    self.error_filtered = 0.0
    self.above_threshold_frames = 0
    self.prev_deficit: float | None = None
    self.prev_roll: float | None = None
    self.last_debug = UnderresponseDebug()

  def _reset_tracking(self) -> None:
    self.desired_history.clear()
    self.error_filtered = 0.0
    self.above_threshold_frames = 0
    self.prev_deficit = None

  def reset(self) -> UnderresponseDebug:
    self._reset_tracking()
    self.prev_roll = None
    self.last_debug = UnderresponseDebug(block_mask=BLOCK_INACTIVE)
    return self.last_debug

  def _soft_decay(self) -> None:
    alpha = math.exp(-self.dt / SOFT_DECAY_TAU)
    self.error_filtered *= alpha
    self.above_threshold_frames = 0

  def _desired_persistent(self, desired_sign: int) -> bool:
    if desired_sign == 0 or len(self.desired_history) < self.history_len:
      return False
    vals = list(self.desired_history)
    if any(_sign(v) != desired_sign for v in vals):
      return False
    if min(abs(v) for v in vals) < MIN_ABS_SETPOINT:
      return False
    max_jerk = max(abs(vals[i] - vals[i - 1]) / self.dt for i in range(1, len(vals)))
    return max_jerk <= MAX_DESIRED_JERK_FOR_STABLE

  def update(self, *, active: bool, v_ego: float, steering_pressed: bool, steer_limited_by_safety: bool,
             curvature_limited: bool, setpoint: float, measurement: float, lateral_accel_deadzone: float,
             output_torque: float, steer_max: float, roll: float) -> UnderresponseDebug:
    if not _finite(v_ego, setpoint, measurement, lateral_accel_deadzone, output_torque, steer_max, roll):
      self.reset()
      self.last_debug = UnderresponseDebug(block_mask=BLOCK_INVALID_INPUT)
      return self.last_debug

    error = float(setpoint - measurement)
    if not active:
      self.reset()
      self.last_debug = UnderresponseDebug(block_mask=BLOCK_INACTIVE, error=error)
      return self.last_debug

    block_mask = 0
    desired_sign = _sign(setpoint)
    measurement_sign = _sign(measurement)
    abs_deadzone = abs(float(lateral_accel_deadzone))
    threshold = max(MIN_ABS_ERROR, ERROR_FRACTION * abs(setpoint), 2.0 * abs_deadzone)
    deficit = max(0.0, desired_sign * error) if desired_sign != 0 else 0.0
    closing_rate = 0.0 if self.prev_deficit is None else (self.prev_deficit - deficit) / self.dt

    roll_rate = 0.0
    if self.prev_roll is not None:
      roll_rate = abs(roll - self.prev_roll) / self.dt
    self.prev_roll = float(roll)

    if v_ego < MIN_V_EGO:
      block_mask |= BLOCK_LOW_SPEED
    if steering_pressed:
      block_mask |= BLOCK_STEERING_PRESSED
    if steer_limited_by_safety:
      block_mask |= BLOCK_STEER_LIMITED
    if curvature_limited:
      block_mask |= BLOCK_CURVATURE_LIMITED
    if steer_max <= 0.0 or abs(output_torque) >= MAX_OUTPUT_TORQUE_FRAC * abs(steer_max):
      block_mask |= BLOCK_TORQUE_SATURATED
    if abs(roll) > MAX_ABS_ROLL:
      block_mask |= BLOCK_ROLL_TOO_HIGH
    if roll_rate > MAX_ROLL_RATE:
      block_mask |= BLOCK_ROLL_UNSTABLE

    if block_mask != 0:
      self._reset_tracking()
      self.last_debug = UnderresponseDebug(block_mask=block_mask, error=error, closing_rate=closing_rate)
      return self.last_debug

    self.desired_history.append(float(setpoint))

    if abs(setpoint) < MIN_ABS_SETPOINT or desired_sign == 0:
      block_mask |= BLOCK_DESIRED_TOO_SMALL
    if not self._desired_persistent(desired_sign):
      block_mask |= BLOCK_DESIRED_NOT_PERSISTENT
    if measurement_sign != 0 and measurement_sign != desired_sign and abs(measurement) > max(ACTUAL_OPPOSING_MIN_ABS, 2.0 * abs_deadzone):
      block_mask |= BLOCK_ACTUAL_OPPOSING
    if desired_sign != 0 and desired_sign * error <= 0.0:
      block_mask |= BLOCK_SIGN_MISMATCH
    if deficit <= threshold:
      block_mask |= BLOCK_ERROR_TOO_SMALL
    if closing_rate > MIN_CLOSING_RATE:
      block_mask |= BLOCK_FAST_CLOSING

    self.prev_deficit = deficit

    if block_mask != 0:
      self._soft_decay()
      self.last_debug = UnderresponseDebug(block_mask=block_mask, error=error, error_filtered=self.error_filtered,
                                           closing_rate=closing_rate)
      return self.last_debug

    alpha = math.exp(-self.dt / EWMA_TAU)
    self.error_filtered = alpha * self.error_filtered + (1.0 - alpha) * error
    filtered_deficit = max(0.0, desired_sign * self.error_filtered)

    if filtered_deficit > threshold:
      self.above_threshold_frames += 1
    else:
      self.above_threshold_frames = 0
      block_mask |= BLOCK_ERROR_TOO_SMALL

    duration = self.above_threshold_frames * self.dt
    active_trigger = block_mask == 0 and duration >= TRIGGER_TIME
    if active_trigger:
      shadow_lat_accel = desired_sign * min(MAX_SHADOW_LAT_ACCEL, SHADOW_CORR_GAIN * filtered_deficit)
      severity = _clip((filtered_deficit - threshold) / max(threshold, 1e-6), 0.0, 1.0)
      severity *= _clip((MIN_CLOSING_RATE - closing_rate) / max(MIN_CLOSING_RATE, 1e-6), 0.0, 1.0)
    else:
      shadow_lat_accel = 0.0
      severity = 0.0

    self.last_debug = UnderresponseDebug(active=active_trigger, eligible=(block_mask == 0), block_mask=block_mask,
                                         error=error, error_filtered=self.error_filtered, duration=duration,
                                         closing_rate=closing_rate, shadow_lat_accel=shadow_lat_accel,
                                         severity=severity)
    return self.last_debug


def write_underresponse_debug(pid_log, debug: UnderresponseDebug) -> None:
  pid_log.underresponseActive = bool(debug.active)
  pid_log.underresponseEligible = bool(debug.eligible)
  pid_log.underresponseBlockMask = int(debug.block_mask)
  pid_log.underresponseError = float(debug.error)
  pid_log.underresponseErrorFiltered = float(debug.error_filtered)
  pid_log.underresponseDuration = float(debug.duration)
  pid_log.underresponseClosingRate = float(debug.closing_rate)
  pid_log.underresponseShadowLatAccel = float(debug.shadow_lat_accel)
  pid_log.underresponseSeverity = float(debug.severity)
