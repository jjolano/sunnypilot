from dataclasses import dataclass
from enum import IntEnum, IntFlag

import numpy as np


RESPONSE_DEFICIT_THRESHOLD = 0.04
STEADY_HOLD_LAT_ACCEL_THRESHOLD = 0.12
STEADY_HOLD_JERK_THRESHOLD = 0.35
LOW_DEMAND_CURVATURE_THRESHOLD = 0.02
PLANNED_UNWIND_JERK_THRESHOLD = 0.2
ASSIST_GAIN = 1.6
ASSIST_BUILD_RATE = 0.8
ASSIST_DECAY_RATE = 1.6
CURVE_EXIT_RESPONSE_DEFICIT_THRESHOLD = 0.02
CURVE_EXIT_MIN_LAT_ACCEL = 0.6
CURVE_EXIT_MIN_CURVATURE = 0.02
CURVE_EXIT_MIN_UNDER_RESPONSE = 0.18
CURVE_EXIT_MIN_UNWIND_JERK = 0.15
CURVE_EXIT_MAX_LOOKAHEAD_JERK = 0.05
CURVE_EXIT_ASSIST_GAIN = 0.8
CURVE_EXIT_ASSIST_CAP_SCALE = 0.4
CURVE_EXIT_ASSIST_BUILD_RATE = 0.4
CURVE_PREPOSITION_RESPONSE_DEFICIT_THRESHOLD = 0.02
CURVE_PREPOSITION_MIN_LAT_ACCEL = 0.8
CURVE_PREPOSITION_MIN_CURVATURE = 0.02
CURVE_PREPOSITION_MIN_UNDER_RESPONSE = 0.2
CURVE_PREPOSITION_MIN_DESIRED_JERK = 0.25
CURVE_PREPOSITION_MIN_JERK_DEFICIT = 0.25
CURVE_PREPOSITION_MIN_OUTPUT_FRACTION = 0.55
CURVE_PREPOSITION_MAX_OUTPUT_FRACTION = 0.9
CURVE_PREPOSITION_ASSIST_DEFICIT_GAIN = 0.5
CURVE_PREPOSITION_ASSIST_JERK_GAIN = 0.02
CURVE_PREPOSITION_ASSIST_CAP_SCALE = 0.35
CURVE_PREPOSITION_ASSIST_BUILD_RATE = 0.35
BIAS_TARGET_GAIN = 0.6
BIAS_BUILD_RATE = 0.18
BIAS_DECAY_RATE = 0.08
RELEASE_DECAY_RATE = 0.24
UNWIND_TRIM_RATE = 1.2
SIGN_THRESHOLD = 0.05
MIN_VEGO = 4.0
BUMP_JERK_THRESHOLD = 2.0
BUMP_LOOKAHEAD_DELTA_THRESHOLD = 1.4
FREEZE_TIME = 0.30
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
  ASSIST = 1
  HOLD = 2
  RELEASE = 3


class GuardedResponseReason(IntFlag):
  NONE = 0
  INACTIVE = 1 << 0
  BUMP = 1 << 1
  LOW_SPEED = 1 << 2
  STEERING_PRESSED = 1 << 3
  STEER_LIMITED = 1 << 4
  CURVATURE_LIMITED = 1 << 5
  SATURATED = 1 << 6
  SIGN_CONFLICT = 1 << 7
  LOW_DEMAND = 1 << 8
  SAME_SIGN_UNWIND = 1 << 9
  LANE_CHANGE = 1 << 10
  NO_NOMINAL = 1 << 11
  SIGN_MISMATCH = 1 << 12
  BELOW_DEFICIT = 1 << 13
  HIGH_JERK = 1 << 14
  RELEASE = 1 << 15


@dataclass
class GuardedResponseAssistInputs:
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
  same_sign_unwind: bool


@dataclass
class GuardedResponseAssistResult:
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
  freeze_reason: int
  block_reason: int


class TorqueGuardedResponseAssist:
  def __init__(self, dt: float):
    self.dt = dt
    self.phase = Phase.IDLE
    self.assist_torque = 0.0
    self.bias_torque = 0.0
    self.freeze_timer = 0.0

  def update(self, inputs: GuardedResponseAssistInputs) -> GuardedResponseAssistResult:
    self.freeze_timer = max(0.0, self.freeze_timer - self.dt)
    if self._is_bump_disturbance(inputs):
      self.freeze_timer = FREEZE_TIME

    response_deficit = inputs.tracking_torque_error
    nominal_sign = sign(inputs.nominal_torque)
    desired_sign = sign(inputs.desired_lateral_accel)
    actual_sign = sign(inputs.actual_lateral_accel)
    sign_conflict = desired_sign != 0.0 and actual_sign != 0.0 and desired_sign != actual_sign and abs(inputs.actual_lateral_accel) > SIGN_THRESHOLD
    low_demand = self._low_demand(inputs)
    residual_active = abs(self.assist_torque) > ACTIVE_RELEASE_THRESHOLD or abs(self.bias_torque) > ACTIVE_RELEASE_THRESHOLD
    planned_unwind = (
      abs(inputs.lookahead_lateral_jerk) < PLANNED_UNWIND_JERK_THRESHOLD
      and abs(inputs.desired_lateral_jerk) < PLANNED_UNWIND_JERK_THRESHOLD
      and low_demand
    )
    release_active = bool(inputs.steering_pressed or sign_conflict or (planned_unwind and residual_active))
    same_sign_hold = desired_sign != 0.0 and (actual_sign == 0.0 or desired_sign == actual_sign)
    freeze_reason = self._freeze_reason(inputs)
    shared_learning_frozen = freeze_reason != GuardedResponseReason.NONE
    max_assist = self._get_speed_cap(inputs.v_ego, ASSIST_CAP_BP, ASSIST_CAP_V, inputs.lane_change_active, LANE_CHANGE_ASSIST_SCALE)
    max_bias = self._get_speed_cap(inputs.v_ego, BIAS_CAP_BP, BIAS_CAP_V, inputs.lane_change_active, LANE_CHANGE_BIAS_SCALE)

    if not inputs.active:
      self.phase = Phase.IDLE
      self.assist_torque = 0.0
      self.bias_torque = 0.0
      self.freeze_timer = 0.0
      return self._result(inputs.nominal_torque, response_deficit, False, False, GuardedResponseReason.NONE, GuardedResponseReason.INACTIVE,
                          inputs.max_output, max_assist, max_bias)

    if release_active:
      block_reason = freeze_reason | GuardedResponseReason.RELEASE
      if inputs.steering_pressed:
        block_reason |= GuardedResponseReason.STEERING_PRESSED
      if sign_conflict:
        block_reason |= GuardedResponseReason.SIGN_CONFLICT
      if planned_unwind and residual_active:
        block_reason |= GuardedResponseReason.LOW_DEMAND

      self.phase = Phase.RELEASE
      self.assist_torque = self._approach(self.assist_torque, 0.0, RELEASE_DECAY_RATE)
      self.bias_torque = self._approach(self.bias_torque, 0.0, RELEASE_DECAY_RATE)
      return self._result(inputs.nominal_torque, response_deficit, True, shared_learning_frozen, freeze_reason, block_reason,
                          inputs.max_output, max_assist, max_bias)

    learning_frozen = shared_learning_frozen
    if inputs.same_sign_unwind:
      self.phase = Phase.HOLD
      self.assist_torque = self._approach(self.assist_torque, 0.0, ASSIST_DECAY_RATE)
      target_bias = 0.0
      if nominal_sign != 0.0 and not shared_learning_frozen:
        trim_magnitude = min(max_bias, abs(response_deficit) * BIAS_TARGET_GAIN)
        target_bias = -nominal_sign * trim_magnitude if trim_magnitude > ACTIVE_RELEASE_THRESHOLD else 0.0
      self.bias_torque = self._approach(self.bias_torque, target_bias, UNWIND_TRIM_RATE)
      block_reason = freeze_reason | GuardedResponseReason.SAME_SIGN_UNWIND
      return self._result(inputs.nominal_torque, response_deficit, False, learning_frozen, freeze_reason, block_reason,
                          inputs.max_output, max_assist, max_bias)

    assist_deficit = nominal_sign * response_deficit
    assist_block_reason = self._assist_block_reason(inputs, nominal_sign, same_sign_hold, low_demand, assist_deficit, freeze_reason)
    assist_allowed = (
      nominal_sign != 0.0
      and not shared_learning_frozen
      and not inputs.saturated
      and same_sign_hold
      and not low_demand
    )
    assist_learning_blocked = bool(
      nominal_sign != 0.0 and not shared_learning_frozen and inputs.saturated and same_sign_hold and not low_demand and assist_deficit > RESPONSE_DEFICIT_THRESHOLD
    )
    if assist_learning_blocked:
      freeze_reason |= GuardedResponseReason.SATURATED
    learning_frozen = learning_frozen or assist_learning_blocked

    curve_exit_under_response = assist_allowed and self._is_curve_exit_under_response(inputs, nominal_sign, desired_sign, same_sign_hold, assist_deficit)
    curve_preposition_under_response = assist_allowed and self._is_curve_preposition_under_response(inputs, nominal_sign, desired_sign,
                                                                                                     same_sign_hold, assist_deficit)
    if curve_exit_under_response:
      assist_block_reason &= ~GuardedResponseReason.BELOW_DEFICIT
    if curve_preposition_under_response:
      assist_block_reason &= ~GuardedResponseReason.BELOW_DEFICIT

    target_assist = 0.0
    assist_build_rate = ASSIST_BUILD_RATE
    if assist_allowed and assist_deficit > RESPONSE_DEFICIT_THRESHOLD:
      target_assist = nominal_sign * min(max_assist, ASSIST_GAIN * (assist_deficit - RESPONSE_DEFICIT_THRESHOLD))
    if curve_exit_under_response:
      curve_exit_max_assist = max_assist * CURVE_EXIT_ASSIST_CAP_SCALE
      curve_exit_target = nominal_sign * min(curve_exit_max_assist, CURVE_EXIT_ASSIST_GAIN * (assist_deficit - CURVE_EXIT_RESPONSE_DEFICIT_THRESHOLD))
      if abs(curve_exit_target) > abs(target_assist):
        target_assist = curve_exit_target
        assist_build_rate = CURVE_EXIT_ASSIST_BUILD_RATE
    if curve_preposition_under_response:
      curve_preposition_max_assist = max_assist * CURVE_PREPOSITION_ASSIST_CAP_SCALE
      jerk_deficit = desired_sign * (inputs.desired_lateral_jerk - inputs.actual_lateral_jerk)
      curve_preposition_target = nominal_sign * min(
        curve_preposition_max_assist,
        CURVE_PREPOSITION_ASSIST_DEFICIT_GAIN * (assist_deficit - CURVE_PREPOSITION_RESPONSE_DEFICIT_THRESHOLD)
        + CURVE_PREPOSITION_ASSIST_JERK_GAIN * jerk_deficit,
      )
      if abs(curve_preposition_target) > abs(target_assist):
        target_assist = curve_preposition_target
        assist_build_rate = CURVE_PREPOSITION_ASSIST_BUILD_RATE

    if abs(target_assist) > ACTIVE_RELEASE_THRESHOLD:
      self.phase = Phase.ASSIST
      self.assist_torque = self._approach(self.assist_torque, target_assist, assist_build_rate)
    else:
      self.assist_torque = self._approach(self.assist_torque, 0.0, ASSIST_DECAY_RATE)
      if same_sign_hold and not low_demand and abs(inputs.desired_lateral_accel) >= STEADY_HOLD_LAT_ACCEL_THRESHOLD:
        self.phase = Phase.HOLD
      else:
        self.phase = Phase.IDLE

    bias_learning_blocked = False
    bias_block_reason = self._bias_block_reason(inputs, nominal_sign, same_sign_hold, low_demand, freeze_reason)
    if self.phase == Phase.HOLD and same_sign_hold and not low_demand and not shared_learning_frozen and abs(inputs.desired_lateral_jerk) < STEADY_HOLD_JERK_THRESHOLD:
      target_bias = clamp(response_deficit * BIAS_TARGET_GAIN, -max_bias, max_bias)
      blocked_bias_direction = nominal_sign != 0.0 and sign(target_bias) == nominal_sign and abs(target_bias) > ACTIVE_RELEASE_THRESHOLD
      bias_learning_blocked = bool(inputs.saturated and blocked_bias_direction)
      if not bias_learning_blocked:
        self.bias_torque = self._approach(self.bias_torque, target_bias, BIAS_BUILD_RATE)
    else:
      self.bias_torque = self._approach(self.bias_torque, 0.0, BIAS_DECAY_RATE)
    if bias_learning_blocked:
      freeze_reason |= GuardedResponseReason.SATURATED
    learning_frozen = learning_frozen or bias_learning_blocked

    return self._result(inputs.nominal_torque, response_deficit, False, learning_frozen, freeze_reason, assist_block_reason | bias_block_reason,
                        inputs.max_output, max_assist, max_bias)

  @staticmethod
  def _low_demand(inputs: GuardedResponseAssistInputs) -> bool:
    return (
      abs(inputs.desired_lateral_accel) < STEADY_HOLD_LAT_ACCEL_THRESHOLD
      and abs(inputs.desired_lateral_jerk) < STEADY_HOLD_JERK_THRESHOLD
      and abs(inputs.lookahead_lateral_jerk) < STEADY_HOLD_JERK_THRESHOLD
      and abs(inputs.desired_curvature) < LOW_DEMAND_CURVATURE_THRESHOLD
    )

  @staticmethod
  def _get_speed_cap(v_ego: float, breakpoints: list[float], values: list[float], lane_change_active: bool, lane_change_scale: float) -> float:
    cap = float(np.interp(v_ego, breakpoints, values))
    return cap * lane_change_scale if lane_change_active else cap

  @staticmethod
  def _is_bump_disturbance(inputs: GuardedResponseAssistInputs) -> bool:
    jerk_delta = abs(inputs.actual_lateral_jerk - inputs.lookahead_lateral_jerk)
    return (
      abs(inputs.actual_lateral_jerk) > BUMP_JERK_THRESHOLD
      and jerk_delta > BUMP_LOOKAHEAD_DELTA_THRESHOLD
      and abs(inputs.desired_lateral_jerk) < BUMP_JERK_THRESHOLD
    )

  @staticmethod
  def _is_curve_exit_under_response(inputs: GuardedResponseAssistInputs, nominal_sign: float, desired_sign: float,
                                    same_sign_hold: bool, assist_deficit: float) -> bool:
    under_response = desired_sign * (inputs.desired_lateral_accel - inputs.actual_lateral_accel)
    return (
      same_sign_hold
      and nominal_sign == desired_sign
      and assist_deficit > CURVE_EXIT_RESPONSE_DEFICIT_THRESHOLD
      and abs(inputs.desired_lateral_accel) >= CURVE_EXIT_MIN_LAT_ACCEL
      and abs(inputs.desired_curvature) >= CURVE_EXIT_MIN_CURVATURE
      and under_response > CURVE_EXIT_MIN_UNDER_RESPONSE
      and desired_sign * inputs.desired_lateral_jerk < -CURVE_EXIT_MIN_UNWIND_JERK
      and desired_sign * inputs.lookahead_lateral_jerk <= CURVE_EXIT_MAX_LOOKAHEAD_JERK
    )

  @staticmethod
  def _is_curve_preposition_under_response(inputs: GuardedResponseAssistInputs, nominal_sign: float, desired_sign: float,
                                           same_sign_hold: bool, assist_deficit: float) -> bool:
    under_response = desired_sign * (inputs.desired_lateral_accel - inputs.actual_lateral_accel)
    desired_jerk = desired_sign * inputs.desired_lateral_jerk
    jerk_deficit = desired_sign * (inputs.desired_lateral_jerk - inputs.actual_lateral_jerk)
    output_fraction = abs(inputs.nominal_torque) / max(inputs.max_output, 1e-3)
    return (
      same_sign_hold
      and not inputs.lane_change_active
      and nominal_sign == desired_sign
      and assist_deficit > CURVE_PREPOSITION_RESPONSE_DEFICIT_THRESHOLD
      and abs(inputs.desired_lateral_accel) >= CURVE_PREPOSITION_MIN_LAT_ACCEL
      and abs(inputs.desired_curvature) >= CURVE_PREPOSITION_MIN_CURVATURE
      and under_response > CURVE_PREPOSITION_MIN_UNDER_RESPONSE
      and desired_jerk > CURVE_PREPOSITION_MIN_DESIRED_JERK
      and jerk_deficit > CURVE_PREPOSITION_MIN_JERK_DEFICIT
      and CURVE_PREPOSITION_MIN_OUTPUT_FRACTION <= output_fraction < CURVE_PREPOSITION_MAX_OUTPUT_FRACTION
    )

  def _freeze_reason(self, inputs: GuardedResponseAssistInputs) -> GuardedResponseReason:
    reason = GuardedResponseReason.NONE
    if self.freeze_timer > 0.0:
      reason |= GuardedResponseReason.BUMP
    if inputs.v_ego < MIN_VEGO:
      reason |= GuardedResponseReason.LOW_SPEED
    if inputs.steering_pressed:
      reason |= GuardedResponseReason.STEERING_PRESSED
    if inputs.steer_limited_by_safety:
      reason |= GuardedResponseReason.STEER_LIMITED
    if inputs.curvature_limited:
      reason |= GuardedResponseReason.CURVATURE_LIMITED
    return reason

  @staticmethod
  def _assist_block_reason(inputs: GuardedResponseAssistInputs, nominal_sign: float, same_sign_hold: bool, low_demand: bool,
                           assist_deficit: float, freeze_reason: GuardedResponseReason) -> GuardedResponseReason:
    reason = freeze_reason
    if inputs.lane_change_active:
      reason |= GuardedResponseReason.LANE_CHANGE
    if nominal_sign == 0.0:
      reason |= GuardedResponseReason.NO_NOMINAL
    if inputs.saturated:
      reason |= GuardedResponseReason.SATURATED
    if not same_sign_hold:
      reason |= GuardedResponseReason.SIGN_MISMATCH
    if low_demand:
      reason |= GuardedResponseReason.LOW_DEMAND
    if assist_deficit <= RESPONSE_DEFICIT_THRESHOLD:
      reason |= GuardedResponseReason.BELOW_DEFICIT
    return reason

  @staticmethod
  def _bias_block_reason(inputs: GuardedResponseAssistInputs, nominal_sign: float, same_sign_hold: bool, low_demand: bool,
                         freeze_reason: GuardedResponseReason) -> GuardedResponseReason:
    reason = freeze_reason
    if inputs.lane_change_active:
      reason |= GuardedResponseReason.LANE_CHANGE
    if nominal_sign == 0.0:
      reason |= GuardedResponseReason.NO_NOMINAL
    if inputs.saturated:
      reason |= GuardedResponseReason.SATURATED
    if not same_sign_hold:
      reason |= GuardedResponseReason.SIGN_MISMATCH
    if low_demand:
      reason |= GuardedResponseReason.LOW_DEMAND
    if abs(inputs.desired_lateral_jerk) >= STEADY_HOLD_JERK_THRESHOLD:
      reason |= GuardedResponseReason.HIGH_JERK
    return reason

  def _result(
    self,
    nominal_torque: float,
    response_deficit: float,
    release_active: bool,
    learning_frozen: bool,
    freeze_reason: GuardedResponseReason,
    block_reason: GuardedResponseReason,
    max_output: float,
    max_assist: float,
    max_bias: float,
  ) -> GuardedResponseAssistResult:
    unclamped_output = nominal_torque + self.assist_torque + self.bias_torque
    output_torque = clamp(unclamped_output, -max_output, max_output)
    clipped_delta = output_torque - unclamped_output
    applied_assist = self.assist_torque + clipped_delta
    phase_gain = self._phase_gain(applied_assist, max_assist, max_bias)
    return GuardedResponseAssistResult(
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
      freeze_reason=int(freeze_reason),
      block_reason=int(block_reason),
    )

  def _phase_gain(self, assist_torque: float, max_assist: float, max_bias: float) -> float:
    if self.phase == Phase.IDLE:
      return 0.0
    if self.phase == Phase.HOLD:
      return 1.0
    basis = max_assist if self.phase == Phase.ASSIST else max(max_assist, max_bias)
    magnitude = abs(assist_torque if self.phase == Phase.ASSIST else assist_torque + self.bias_torque)
    return clamp(magnitude / max(basis, 1e-3), 0.0, 1.0)

  def _approach(self, current: float, target: float, rate: float) -> float:
    max_step = rate * self.dt
    return current + clamp(target - current, -max_step, max_step)
