from dataclasses import dataclass
from enum import Enum, auto


START_TORQUE_THRESHOLD = 0.015
START_LAT_ACCEL_THRESHOLD = 0.15
END_LAT_ACCEL_THRESHOLD = 0.08
MIN_VEGO = 5.0
RAMP_IN_TIME = 0.35
TAPER_OUT_TIME = 0.25
STABLE_SIGN_FRAMES = 5
BUMP_FREEZE_TIME = 0.30
DEFAULT_AUTHORITY_FLOOR = 0.02
MIN_AUTHORITY_FLOOR = 0.01
MAX_AUTHORITY_FLOOR = 0.12
MAX_DISTURBANCE_BIAS = 0.06
GOOD_TRACKING_THRESHOLD = 0.12
UNDERSHOOT_THRESHOLD = 0.20
BUMP_JERK_THRESHOLD = 2.0
BUMP_LOOKAHEAD_DELTA_THRESHOLD = 1.4
STEADY_JERK_THRESHOLD = 0.8
FLOOR_LEARN_UP_RATE = 0.08
FLOOR_LEARN_DOWN_RATE = 0.02
BIAS_LEARN_RATE = 0.3
BIAS_DECAY_RATE = 0.04


def clamp(val: float, lower: float, upper: float) -> float:
  return max(lower, min(upper, val))


def sign(val: float) -> float:
  return 1.0 if val > 0.0 else (-1.0 if val < 0.0 else 0.0)


class Phase(Enum):
  IDLE = auto()
  RAMP_IN = auto()
  HOLD = auto()
  TAPER_OUT = auto()


@dataclass
class EnvelopeInputs:
  active: bool
  v_ego: float
  steering_pressed: bool
  steer_limited_by_safety: bool
  curvature_limited: bool
  saturated: bool
  nominal_torque: float
  desired_lateral_accel: float
  actual_lateral_accel: float
  desired_lateral_jerk: float
  actual_lateral_jerk: float
  lookahead_lateral_jerk: float
  tracking_torque_error: float
  lane_change_active: bool


@dataclass
class EnvelopeResult:
  output_torque: float
  phase: str
  phase_gain: float
  authority_floor: float
  disturbance_bias: float
  learning_frozen: bool


@dataclass
class BucketState:
  authority_floor: float = DEFAULT_AUTHORITY_FLOOR
  confidence: float = 0.0
  stable_frames: int = 0
  last_used_frame: int = 0


class TorqueAuthorityEnvelope:
  def __init__(self, dt: float):
    self.dt = dt
    self.phase = Phase.IDLE
    self.phase_gain = 0.0
    self.freeze_timer = 0.0
    self.sign_latch = 0.0
    self.last_nonzero_sign = 0.0
    self.sign_stable_frames = 0
    self.frame = 0
    self.disturbance_bias = 0.0
    self.buckets: dict[tuple, BucketState] = {}

  def update(self, inputs: EnvelopeInputs) -> EnvelopeResult:
    self.frame += 1
    self.freeze_timer = max(0.0, self.freeze_timer - self.dt)

    nominal_sign = sign(inputs.nominal_torque)
    self._update_sign_tracking(nominal_sign)

    if self._is_bump_disturbance(inputs):
      self.freeze_timer = BUMP_FREEZE_TIME

    self._advance_phase(inputs, nominal_sign)

    learning_frozen = self._learning_frozen(inputs, nominal_sign)
    bucket = self._get_bucket(inputs)
    authority_floor = bucket.authority_floor if bucket is not None else 0.0

    if bucket is not None and self.phase == Phase.HOLD and not learning_frozen:
      self._update_authority_floor(bucket, inputs)
      bucket.last_used_frame = self.frame
      authority_floor = bucket.authority_floor

    self._update_disturbance_bias(inputs, learning_frozen)

    command_core = inputs.nominal_torque + self.disturbance_bias
    output_torque = command_core
    if inputs.active and self.sign_latch != 0.0 and self.phase != Phase.IDLE:
      floor_torque = self.phase_gain * authority_floor
      output_torque = self.sign_latch * max(abs(command_core), floor_torque)

    return EnvelopeResult(
      output_torque=output_torque,
      phase=self.phase.name,
      phase_gain=self.phase_gain,
      authority_floor=authority_floor,
      disturbance_bias=self.disturbance_bias,
      learning_frozen=learning_frozen,
    )

  def _update_sign_tracking(self, nominal_sign: float) -> None:
    if nominal_sign == 0.0:
      self.sign_stable_frames = 0
      return

    if nominal_sign == self.last_nonzero_sign:
      self.sign_stable_frames += 1
    else:
      self.last_nonzero_sign = nominal_sign
      self.sign_stable_frames = 1

  def _advance_phase(self, inputs: EnvelopeInputs, nominal_sign: float) -> None:
    if not inputs.active:
      self.phase = Phase.IDLE
      self.phase_gain = 0.0
      self.sign_latch = 0.0
      return

    start_ready = self._start_ready(inputs, nominal_sign)
    taper_ready = self._taper_ready(inputs, nominal_sign)

    if self.phase == Phase.IDLE:
      self.phase_gain = 0.0
      if start_ready:
        self.phase = Phase.RAMP_IN
        self.sign_latch = nominal_sign
      return

    if self.phase == Phase.RAMP_IN:
      if taper_ready:
        self.phase = Phase.TAPER_OUT
      else:
        self.phase_gain = min(1.0, self.phase_gain + self.dt / RAMP_IN_TIME)
        if self.phase_gain >= 1.0:
          self.phase = Phase.HOLD
      return

    if self.phase == Phase.HOLD:
      self.phase_gain = 1.0
      if taper_ready:
        self.phase = Phase.TAPER_OUT
      return

    self.phase_gain = max(0.0, self.phase_gain - self.dt / TAPER_OUT_TIME)
    if self.phase_gain <= 0.0:
      self.phase = Phase.IDLE
      self.sign_latch = 0.0
      if start_ready:
        self.phase = Phase.RAMP_IN
        self.sign_latch = nominal_sign

  def _start_ready(self, inputs: EnvelopeInputs, nominal_sign: float) -> bool:
    has_demand = abs(inputs.nominal_torque) > START_TORQUE_THRESHOLD or abs(inputs.desired_lateral_accel) > START_LAT_ACCEL_THRESHOLD
    return inputs.v_ego >= MIN_VEGO and has_demand and nominal_sign != 0.0 and self.sign_stable_frames >= STABLE_SIGN_FRAMES

  def _taper_ready(self, inputs: EnvelopeInputs, nominal_sign: float) -> bool:
    demand_is_low = abs(inputs.nominal_torque) < START_TORQUE_THRESHOLD and abs(inputs.desired_lateral_accel) < END_LAT_ACCEL_THRESHOLD
    sign_flipped = self.sign_latch != 0.0 and nominal_sign != 0.0 and nominal_sign != self.sign_latch and self.sign_stable_frames >= STABLE_SIGN_FRAMES
    planned_unwind = (
      abs(inputs.lookahead_lateral_jerk) < 0.25 and abs(inputs.desired_lateral_jerk) < 0.25 and abs(inputs.desired_lateral_accel) < START_LAT_ACCEL_THRESHOLD
    )
    return demand_is_low or sign_flipped or planned_unwind

  def _get_bucket(self, inputs: EnvelopeInputs) -> BucketState | None:
    bucket_sign = self.sign_latch if self.sign_latch != 0.0 else self.last_nonzero_sign
    if not inputs.active or bucket_sign == 0.0:
      return None

    key = (
      bucket_sign,
      self._bucket_value(inputs.v_ego, (8.0, 18.0)),
      self._bucket_value(abs(inputs.desired_lateral_accel), (0.4, 1.0)),
      self._bucket_value(abs(inputs.desired_lateral_jerk), (0.5, 1.5)),
      int(inputs.lane_change_active),
    )
    return self.buckets.setdefault(key, BucketState())

  @staticmethod
  def _bucket_value(value: float, thresholds: tuple[float, float]) -> int:
    if value < thresholds[0]:
      return 0
    if value < thresholds[1]:
      return 1
    return 2

  def _learning_frozen(self, inputs: EnvelopeInputs, nominal_sign: float) -> bool:
    sign_unstable = nominal_sign == 0.0 or self.sign_stable_frames < STABLE_SIGN_FRAMES
    return (
      self.freeze_timer > 0.0
      or inputs.v_ego < MIN_VEGO
      or inputs.steering_pressed
      or inputs.steer_limited_by_safety
      or inputs.curvature_limited
      or inputs.saturated
      or sign_unstable
    )

  def _update_authority_floor(self, bucket: BucketState, inputs: EnvelopeInputs) -> None:
    command_abs = abs(inputs.nominal_torque + self.disturbance_bias)
    if command_abs < START_TORQUE_THRESHOLD:
      return

    tracking_error = abs(inputs.desired_lateral_accel - inputs.actual_lateral_accel)
    if tracking_error > UNDERSHOOT_THRESHOLD:
      target_floor = clamp(command_abs * 0.95, MIN_AUTHORITY_FLOOR, MAX_AUTHORITY_FLOOR)
      bucket.authority_floor = self._approach(bucket.authority_floor, target_floor, FLOOR_LEARN_UP_RATE, FLOOR_LEARN_DOWN_RATE)
      bucket.confidence = min(1.0, bucket.confidence + self.dt)
      return

    if tracking_error < GOOD_TRACKING_THRESHOLD:
      target_floor = clamp(command_abs * 0.75, MIN_AUTHORITY_FLOOR, MAX_AUTHORITY_FLOOR)
      bucket.authority_floor = self._approach(bucket.authority_floor, target_floor, FLOOR_LEARN_UP_RATE * 0.5, FLOOR_LEARN_DOWN_RATE)
      bucket.confidence = min(1.0, bucket.confidence + self.dt)

  def _update_disturbance_bias(self, inputs: EnvelopeInputs, learning_frozen: bool) -> None:
    should_adapt = (
      self.phase == Phase.HOLD
      and not learning_frozen
      and abs(inputs.desired_lateral_accel) > END_LAT_ACCEL_THRESHOLD
      and abs(inputs.desired_lateral_jerk) < STEADY_JERK_THRESHOLD
      and abs(inputs.lookahead_lateral_jerk) < STEADY_JERK_THRESHOLD
    )
    if should_adapt:
      bias_step = clamp(inputs.tracking_torque_error * BIAS_LEARN_RATE * self.dt, -0.002, 0.002)
      self.disturbance_bias = clamp(self.disturbance_bias + bias_step, -MAX_DISTURBANCE_BIAS, MAX_DISTURBANCE_BIAS)
      return

    self.disturbance_bias = self._approach(self.disturbance_bias, 0.0, BIAS_DECAY_RATE, BIAS_DECAY_RATE)

  @staticmethod
  def _is_bump_disturbance(inputs: EnvelopeInputs) -> bool:
    jerk_delta = abs(inputs.actual_lateral_jerk - inputs.lookahead_lateral_jerk)
    return (
      abs(inputs.actual_lateral_jerk) > BUMP_JERK_THRESHOLD
      and jerk_delta > BUMP_LOOKAHEAD_DELTA_THRESHOLD
      and abs(inputs.desired_lateral_jerk) < BUMP_JERK_THRESHOLD
    )

  def _approach(self, current: float, target: float, up_rate: float, down_rate: float) -> float:
    delta = target - current
    max_step = up_rate * self.dt if delta >= 0.0 else down_rate * self.dt
    return current + clamp(delta, -abs(max_step), abs(max_step))
