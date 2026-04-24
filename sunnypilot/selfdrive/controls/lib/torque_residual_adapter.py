from dataclasses import dataclass
from enum import IntEnum

import numpy as np


RESPONSE_DEFICIT_THRESHOLD = 0.04
STEADY_HOLD_LAT_ACCEL_THRESHOLD = 0.12
STEADY_HOLD_JERK_THRESHOLD = 0.35
LOW_DEMAND_CURVATURE_THRESHOLD = 0.02
PLANNED_UNWIND_JERK_THRESHOLD = 0.2
SIGN_THRESHOLD = 0.05
MIN_VEGO = 4.0
BUMP_JERK_THRESHOLD = 2.0
BUMP_LOOKAHEAD_DELTA_THRESHOLD = 1.4
FREEZE_TIME = 0.30
ASSIST_GAIN = 1.7
ASSIST_BUILD_RATE = 0.9
ASSIST_DECAY_RATE = 1.8
BIAS_TARGET_GAIN = 0.55
BIAS_BUILD_RATE = 0.18
BIAS_DECAY_RATE = 0.08
BIAS_APPLY_RATE = 0.24
RELEASE_DECAY_RATE = 0.24
ASSIST_CAP_BP = [0.0, 5.0, 10.0, 20.0, 30.0]
ASSIST_CAP_V = [0.04, 0.06, 0.10, 0.14, 0.18]
BIAS_CAP_BP = [0.0, 5.0, 10.0, 20.0, 30.0]
BIAS_CAP_V = [0.02, 0.03, 0.05, 0.07, 0.08]
LANE_CHANGE_ASSIST_SCALE = 0.75
LANE_CHANGE_BIAS_SCALE = 0.6
ACTIVE_RELEASE_THRESHOLD = 1e-4


def clamp(value: float, lower: float, upper: float) -> float:
  return max(lower, min(upper, value))


def sign(value: float) -> float:
  return 1.0 if value > 0.0 else (-1.0 if value < 0.0 else 0.0)


class Phase(IntEnum):
  IDLE = 0
  ENGAGE = 1
  HOLD = 2
  RELEASE = 3


@dataclass
class ResidualAdapterInputs:
  active: bool
  v_ego: float
  steering_pressed: bool
  steer_limited_by_safety: bool
  curvature_limited: bool
  saturated: bool
  max_output: float
  nominal_torque: float
  desired_lateral_accel: float
  actual_lateral_accel: float
  desired_lateral_jerk: float
  actual_lateral_jerk: float
  lookahead_lateral_jerk: float
  desired_curvature: float
  tracking_torque_error: float
  lane_change_active: bool


@dataclass
class ResidualAdapterResult:
  output_torque: float
  phase: str
  phase_id: int
  phase_gain: float
  assist_torque: float
  bias_torque: float
  nominal_torque: float
  release_active: bool
  response_deficit: float
  learning_frozen: bool


@dataclass
class ContextState:
  bias_torque: float = 0.0


class TorqueResidualAdapter:
  def __init__(self, dt: float):
    self.dt = dt
    self.phase = Phase.IDLE
    self.assist_torque = 0.0
    self.bias_torque = 0.0
    self.freeze_timer = 0.0
    self.contexts: dict[tuple, ContextState] = {}

  def update(self, inputs: ResidualAdapterInputs) -> ResidualAdapterResult:
    self.freeze_timer = max(0.0, self.freeze_timer - self.dt)
    if self._is_bump_disturbance(inputs):
      self.freeze_timer = FREEZE_TIME

    response_deficit = inputs.tracking_torque_error
    nominal_sign = sign(inputs.nominal_torque)
    desired_sign = sign(inputs.desired_lateral_accel)
    actual_sign = sign(inputs.actual_lateral_accel)
    sign_conflict = desired_sign != 0.0 and actual_sign != 0.0 and desired_sign != actual_sign and abs(inputs.actual_lateral_accel) > SIGN_THRESHOLD
    low_demand = (
      abs(inputs.desired_lateral_accel) < STEADY_HOLD_LAT_ACCEL_THRESHOLD
      and abs(inputs.desired_lateral_jerk) < STEADY_HOLD_JERK_THRESHOLD
      and abs(inputs.lookahead_lateral_jerk) < STEADY_HOLD_JERK_THRESHOLD
      and abs(inputs.desired_curvature) < LOW_DEMAND_CURVATURE_THRESHOLD
    )
    planned_unwind = (
      abs(inputs.lookahead_lateral_jerk) < PLANNED_UNWIND_JERK_THRESHOLD and abs(inputs.desired_lateral_jerk) < PLANNED_UNWIND_JERK_THRESHOLD and low_demand
    )
    residual_active = abs(self.assist_torque) > ACTIVE_RELEASE_THRESHOLD or abs(self.bias_torque) > ACTIVE_RELEASE_THRESHOLD
    release_active = bool(inputs.steering_pressed or sign_conflict or (planned_unwind and residual_active))
    nominal_aligned = nominal_sign != 0.0 and desired_sign != 0.0 and nominal_sign == desired_sign
    same_sign_response = desired_sign != 0.0 and (actual_sign == 0.0 or desired_sign == actual_sign)
    shared_learning_frozen = bool(
      self.freeze_timer > 0.0 or inputs.v_ego < MIN_VEGO or inputs.steering_pressed or inputs.steer_limited_by_safety or inputs.curvature_limited
    )
    stable_hold = bool(
      same_sign_response
      and not low_demand
      and abs(inputs.desired_lateral_accel) >= STEADY_HOLD_LAT_ACCEL_THRESHOLD
      and abs(inputs.desired_lateral_jerk) < STEADY_HOLD_JERK_THRESHOLD
      and abs(inputs.lookahead_lateral_jerk) < STEADY_HOLD_JERK_THRESHOLD
    )

    max_assist = self._get_speed_cap(inputs.v_ego, ASSIST_CAP_BP, ASSIST_CAP_V, inputs.lane_change_active, LANE_CHANGE_ASSIST_SCALE)
    max_bias = self._get_speed_cap(inputs.v_ego, BIAS_CAP_BP, BIAS_CAP_V, inputs.lane_change_active, LANE_CHANGE_BIAS_SCALE)
    bucket = self._get_context(inputs, desired_sign if desired_sign != 0.0 else nominal_sign)

    if not inputs.active:
      self.phase = Phase.IDLE
      self.assist_torque = self._approach(self.assist_torque, 0.0, RELEASE_DECAY_RATE)
      self.bias_torque = self._approach(self.bias_torque, 0.0, RELEASE_DECAY_RATE)
      return self._result(inputs.nominal_torque, response_deficit, False, False, inputs.max_output, max_assist, max_bias)

    bias_learning_blocked = False
    if bucket is not None and stable_hold and not release_active and not shared_learning_frozen:
      target_bias = clamp(inputs.tracking_torque_error * BIAS_TARGET_GAIN, -max_bias, max_bias)
      same_direction_bias = nominal_sign != 0.0 and sign(target_bias) == nominal_sign and abs(target_bias) > ACTIVE_RELEASE_THRESHOLD
      bias_learning_blocked = bool(inputs.saturated and same_direction_bias)
      if not bias_learning_blocked:
        bucket.bias_torque = self._approach(bucket.bias_torque, target_bias, BIAS_BUILD_RATE)
    elif bucket is not None and release_active:
      bucket.bias_torque = self._approach(bucket.bias_torque, 0.0, RELEASE_DECAY_RATE)

    assist_deficit = nominal_sign * response_deficit
    assist_requested = (
      nominal_aligned
      and same_sign_response
      and not low_demand
      and not release_active
      and not shared_learning_frozen
      and assist_deficit > RESPONSE_DEFICIT_THRESHOLD
    )
    assist_learning_blocked = bool(assist_requested and inputs.saturated)
    if assist_requested and not assist_learning_blocked:
      target_assist = nominal_sign * min(max_assist, ASSIST_GAIN * (assist_deficit - RESPONSE_DEFICIT_THRESHOLD))
      self.assist_torque = self._approach(self.assist_torque, target_assist, ASSIST_BUILD_RATE)
    else:
      decay_rate = RELEASE_DECAY_RATE if release_active else ASSIST_DECAY_RATE
      self.assist_torque = self._approach(self.assist_torque, 0.0, decay_rate)

    target_applied_bias = 0.0
    if bucket is not None and same_sign_response and not low_demand and not release_active:
      target_applied_bias = clamp(bucket.bias_torque, -max_bias, max_bias)
      blocked_bias_direction = nominal_sign != 0.0 and sign(target_applied_bias) == nominal_sign and abs(target_applied_bias) > ACTIVE_RELEASE_THRESHOLD
      if inputs.saturated and blocked_bias_direction:
        target_applied_bias = 0.0
    bias_rate = RELEASE_DECAY_RATE if release_active else BIAS_APPLY_RATE
    self.bias_torque = self._approach(self.bias_torque, target_applied_bias, bias_rate)

    learning_frozen = shared_learning_frozen or bias_learning_blocked or assist_learning_blocked

    if release_active:
      self.phase = Phase.RELEASE
    elif abs(self.assist_torque) > 1e-4:
      self.phase = Phase.ENGAGE
    elif stable_hold or abs(self.bias_torque) > 1e-4:
      self.phase = Phase.HOLD
    else:
      self.phase = Phase.IDLE

    return self._result(inputs.nominal_torque, response_deficit, release_active, learning_frozen, inputs.max_output, max_assist, max_bias)

  def _result(
    self,
    nominal_torque: float,
    response_deficit: float,
    release_active: bool,
    learning_frozen: bool,
    max_output: float,
    max_assist: float,
    max_bias: float,
  ) -> ResidualAdapterResult:
    unclamped_output = nominal_torque + self.assist_torque + self.bias_torque
    output_torque = clamp(unclamped_output, -max_output, max_output)
    clipped_delta = output_torque - unclamped_output
    applied_assist = self.assist_torque + clipped_delta
    phase_gain = self._phase_gain(applied_assist, max_assist, max_bias)
    return ResidualAdapterResult(
      output_torque=output_torque,
      phase=self.phase.name,
      phase_id=int(self.phase),
      phase_gain=phase_gain,
      assist_torque=applied_assist,
      bias_torque=self.bias_torque,
      nominal_torque=nominal_torque,
      release_active=release_active or self.phase == Phase.RELEASE,
      response_deficit=response_deficit,
      learning_frozen=learning_frozen,
    )

  def _get_context(self, inputs: ResidualAdapterInputs, command_sign: float) -> ContextState | None:
    if command_sign == 0.0:
      return None

    jerk_mag = max(abs(inputs.desired_lateral_jerk), abs(inputs.lookahead_lateral_jerk))
    key = (
      command_sign,
      self._bucket_value(inputs.v_ego, (8.0, 18.0)),
      self._bucket_value(abs(inputs.desired_lateral_accel), (0.35, 0.9)),
      self._bucket_value(jerk_mag, (0.35, 1.0)),
      int(inputs.lane_change_active),
    )
    return self.contexts.setdefault(key, ContextState())

  @staticmethod
  def _bucket_value(value: float, thresholds: tuple[float, float]) -> int:
    if value < thresholds[0]:
      return 0
    if value < thresholds[1]:
      return 1
    return 2

  @staticmethod
  def _get_speed_cap(v_ego: float, breakpoints: list[float], values: list[float], lane_change_active: bool, lane_change_scale: float) -> float:
    cap = float(np.interp(v_ego, breakpoints, values))
    return cap * lane_change_scale if lane_change_active else cap

  @staticmethod
  def _is_bump_disturbance(inputs: ResidualAdapterInputs) -> bool:
    jerk_delta = abs(inputs.actual_lateral_jerk - inputs.lookahead_lateral_jerk)
    return (
      abs(inputs.actual_lateral_jerk) > BUMP_JERK_THRESHOLD
      and jerk_delta > BUMP_LOOKAHEAD_DELTA_THRESHOLD
      and abs(inputs.desired_lateral_jerk) < BUMP_JERK_THRESHOLD
    )

  def _phase_gain(self, assist_torque: float, max_assist: float, max_bias: float) -> float:
    if self.phase == Phase.IDLE:
      return 0.0
    if self.phase == Phase.HOLD:
      return 1.0
    if self.phase == Phase.ENGAGE:
      magnitude = abs(assist_torque)
      basis = max(max_assist, 1e-3)
      return clamp(magnitude / basis, 0.0, 1.0)

    magnitude = abs(assist_torque + self.bias_torque)
    basis = max(max_assist, max_bias, 1e-3)
    return clamp(magnitude / basis, 0.0, 1.0)

  def _approach(self, current: float, target: float, rate: float) -> float:
    max_step = rate * self.dt
    return current + clamp(target - current, -max_step, max_step)
