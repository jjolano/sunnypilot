from dataclasses import dataclass
from enum import IntEnum


RESPONSE_DEFICIT_THRESHOLD = 0.04
STEADY_HOLD_LAT_ACCEL_THRESHOLD = 0.12
STEADY_HOLD_JERK_THRESHOLD = 0.35
LOW_DEMAND_CURVATURE_THRESHOLD = 0.02
ASSIST_GAIN = 1.6
MAX_ASSIST = 0.18
ASSIST_BUILD_RATE = 0.8
ASSIST_DECAY_RATE = 1.6
BIAS_TARGET_GAIN = 0.6
MAX_BIAS = 0.08
BIAS_BUILD_RATE = 0.18
BIAS_DECAY_RATE = 0.08
RELEASE_DECAY_RATE = 0.24
SIGN_THRESHOLD = 0.05


def clamp(value: float, lower: float, upper: float) -> float:
  return max(lower, min(upper, value))


def sign(value: float) -> float:
  return 1.0 if value > 0.0 else (-1.0 if value < 0.0 else 0.0)


class Phase(IntEnum):
  IDLE = 0
  ASSIST = 1
  HOLD = 2
  RELEASE = 3


@dataclass
class ResponseAssistInputs:
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


@dataclass
class ResponseAssistResult:
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


class TorqueResponseAssist:
  def __init__(self, dt: float):
    self.dt = dt
    self.phase = Phase.IDLE
    self.assist_torque = 0.0
    self.bias_torque = 0.0

  def update(self, inputs: ResponseAssistInputs) -> ResponseAssistResult:
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
    planned_unwind = abs(inputs.lookahead_lateral_jerk) < 0.2 and abs(inputs.desired_lateral_jerk) < 0.2 and low_demand
    release_active = bool(inputs.steering_pressed or sign_conflict or planned_unwind)

    if not inputs.active:
      self.phase = Phase.IDLE
      self.assist_torque = self._approach(self.assist_torque, 0.0, RELEASE_DECAY_RATE)
      self.bias_torque = self._approach(self.bias_torque, 0.0, RELEASE_DECAY_RATE)
      return self._result(inputs.nominal_torque, response_deficit, False, inputs.max_output)

    same_sign_hold = desired_sign != 0.0 and (actual_sign == 0.0 or desired_sign == actual_sign)
    assist_deficit = nominal_sign * response_deficit
    assist_allowed = (
      nominal_sign != 0.0
      and not release_active
      and not inputs.steer_limited_by_safety
      and not inputs.curvature_limited
      and not inputs.saturated
      and same_sign_hold
      and not low_demand
    )
    target_assist = 0.0
    if assist_allowed and assist_deficit > RESPONSE_DEFICIT_THRESHOLD:
      target_assist = nominal_sign * min(MAX_ASSIST, ASSIST_GAIN * (assist_deficit - RESPONSE_DEFICIT_THRESHOLD))
      self.phase = Phase.ASSIST
      self.assist_torque = self._approach(self.assist_torque, target_assist, ASSIST_BUILD_RATE)
    else:
      self.assist_torque = self._approach(self.assist_torque, 0.0, ASSIST_DECAY_RATE)
      if release_active:
        self.phase = Phase.RELEASE
      elif same_sign_hold and not low_demand and abs(inputs.desired_lateral_accel) >= STEADY_HOLD_LAT_ACCEL_THRESHOLD:
        self.phase = Phase.HOLD
      else:
        self.phase = Phase.IDLE

    if self.phase == Phase.HOLD and not release_active and not inputs.steering_pressed and abs(inputs.desired_lateral_jerk) < STEADY_HOLD_JERK_THRESHOLD:
      target_bias = clamp(inputs.tracking_torque_error * BIAS_TARGET_GAIN, -MAX_BIAS, MAX_BIAS)
      self.bias_torque = self._approach(self.bias_torque, target_bias, BIAS_BUILD_RATE)
    else:
      decay_rate = RELEASE_DECAY_RATE if release_active else BIAS_DECAY_RATE
      self.bias_torque = self._approach(self.bias_torque, 0.0, decay_rate)

    return self._result(inputs.nominal_torque, response_deficit, release_active, inputs.max_output)

  def _result(self, nominal_torque: float, response_deficit: float, release_active: bool, max_output: float) -> ResponseAssistResult:
    unclamped_output = nominal_torque + self.assist_torque + self.bias_torque
    output_torque = clamp(unclamped_output, -max_output, max_output)
    clipped_delta = output_torque - unclamped_output
    applied_assist = self.assist_torque + clipped_delta
    phase_gain = self._phase_gain(applied_assist)
    return ResponseAssistResult(
      output_torque=output_torque,
      phase=self.phase.name,
      phase_id=int(self.phase),
      phase_gain=phase_gain,
      assist_torque=applied_assist,
      bias_torque=self.bias_torque,
      nominal_torque=nominal_torque,
      release_active=release_active or self.phase == Phase.RELEASE,
      response_deficit=response_deficit,
      learning_frozen=False,
    )

  def _phase_gain(self, assist_torque: float) -> float:
    if self.phase == Phase.IDLE:
      return 0.0
    if self.phase == Phase.HOLD:
      return 1.0
    basis = MAX_ASSIST if self.phase == Phase.ASSIST else max(MAX_ASSIST, MAX_BIAS)
    magnitude = abs(assist_torque if self.phase == Phase.ASSIST else assist_torque + self.bias_torque)
    return clamp(magnitude / max(basis, 1e-3), 0.0, 1.0)

  def _approach(self, current: float, target: float, rate: float) -> float:
    max_step = rate * self.dt
    return current + clamp(target - current, -max_step, max_step)
