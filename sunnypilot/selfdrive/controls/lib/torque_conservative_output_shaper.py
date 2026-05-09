from dataclasses import dataclass
from enum import IntFlag


ISO_LATERAL_ACCEL = 3.0
ISO_ACCEL_MARGIN = 2.6
SIGN_THRESHOLD = 0.05
OVER_RESPONSE_MARGIN = 0.12
UNDER_RESPONSE_MARGIN = 0.12
OVER_RESPONSE_DYNAMIC_START = 0.10
OVER_RESPONSE_MODERATE_EXCESS = 0.35
OVER_RESPONSE_SEVERE_EXCESS = 0.60
BUMP_JERK_THRESHOLD = 2.0
BUMP_LOOKAHEAD_DELTA_THRESHOLD = 1.4
LOW_SPEED_THRESHOLD = 8.0
HIGH_OUTPUT_FRACTION = 0.70
OUTPUT_RATE_RECOVERY_WINDOW = 0.40
OUTPUT_RECOVERY_RATE = 2.50
OUTPUT_SIGN_TRANSITION_RATE = 4.00
AUTHORITY_RECOVERY_LOW_SPEED = 8.0
AUTHORITY_RECOVERY_MID_SPEED = 15.0
AUTHORITY_RECOVERY_HIGH_SPEED = 25.0
AUTHORITY_RECOVERY_LOW_SPEED_RATE = 1.20
AUTHORITY_RECOVERY_HIGH_SPEED_RATE = 4.00
AUTHORITY_RECOVERY_BYPASS_UNDER_RESPONSE = 0.50
STEERING_RATE_COMFORT_START = 15.0
STEERING_RATE_COMFORT_FULL = 80.0
STEERING_RATE_COMFORT_RATE = 0.75
ACTUATOR_LAG_COMFORT_START = 15.0
ACTUATOR_LAG_COMFORT_RATE = 0.35
ACTUATOR_LAG_COMFORT_LOW_SPEED = 8.0
ACTUATOR_LAG_COMFORT_MID_SPEED = 15.0
ACTUATOR_LAG_COMFORT_HIGH_SPEED = 25.0
STALE_ACTUATOR_REVERSAL_THRESHOLD = 0.05
SAFETY_LIMITED_RAMP_ERROR_THRESHOLD = 0.10
SAFETY_LIMITED_RAMP_FOLLOW_MARGIN = 0.15
SAFETY_LIMITED_RAMP_APPLIED_RECOVERY_THRESHOLD = 0.05
SAFETY_LIMITED_RAMP_UNDER_RESPONSE_FLOOR = 0.45
SAFETY_LIMITED_RAMP_UNDER_RESPONSE_FULL_SPEED = 9.0
SAFETY_LIMITED_RAMP_UNDER_RESPONSE_FADE_SPEED = 12.0
HIGH_SPEED_ACTUATOR_LAG_UNWIND_SPEED = 16.0
HIGH_SPEED_ACTUATOR_LAG_UNWIND_MARGIN = 0.15
HIGH_SPEED_ACTUATOR_LAG_UNWIND_GAP = 0.25
HIGH_SPEED_ACTUATOR_LAG_UNWIND_CAP = 0.70
DEFAULT_DT = 0.01

NORMAL_CAP = 1.00
LOW_SPEED_STEER_LIMITED_CAP = 0.92
BUMP_CAP = 0.90
NEAR_ISO_ACCEL_CAP = 0.85
OVER_ISO_ACCEL_CAP = 0.80
OVER_RESPONSE_CAP = 0.85
OVER_RESPONSE_MODERATE_CAP = 0.65
OVER_RESPONSE_SEVERE_CAP = 0.45
SIGN_CONFLICT_CAP = 0.80
OVERRIDE_RELEASE_CAP = 0.80
SAME_SIGN_UNWIND_CAP = 0.30
STEERING_RATE_COMFORT_MIN_CAP = 0.80
ACTUATOR_LAG_COMFORT_LOW_SPEED_CAP = 0.55
ACTUATOR_LAG_COMFORT_MID_SPEED_CAP = 0.70
ACTUATOR_LAG_COMFORT_HIGH_SPEED_CAP = 0.85
STALE_ACTUATOR_REVERSAL_CAP = 0.35
STALE_ACTUATOR_REVERSAL_RATE = 0.20


def clamp(value: float, lower: float, upper: float) -> float:
  return max(lower, min(upper, value))


def sign(value: float) -> float:
  return 1.0 if value > 0.0 else (-1.0 if value < 0.0 else 0.0)


class ConservativeOutputShapingReason(IntFlag):
  NONE = 0
  STEERING_PRESSED = 1 << 0
  RELEASE = 1 << 1
  SIGN_CONFLICT = 1 << 2
  OVER_RESPONSE = 1 << 3
  NEAR_ISO_ACCEL = 1 << 4
  BUMP = 1 << 5
  LOW_SPEED_STEER_LIMITED = 1 << 6
  OUTPUT_RATE_LIMITED = 1 << 7
  SAME_SIGN_UNWIND = 1 << 8
  STEERING_RATE_COMFORT = 1 << 9
  ACTUATOR_LAG_COMFORT = 1 << 10
  STALE_ACTUATOR_REVERSAL = 1 << 11
  SAFETY_LIMITED_RAMP = 1 << 12
  HIGH_SPEED_ACTUATOR_LAG_UNWIND = 1 << 13


@dataclass
class ConservativeOutputShaperInputs:
  active: bool
  v_ego: float
  steering_pressed: bool
  steer_limited_by_safety: bool
  release_active: bool
  max_output: float
  unshaped_output: float
  desired_lateral_accel: float
  actual_lateral_accel: float
  desired_lateral_jerk: float
  actual_lateral_jerk: float
  lookahead_lateral_jerk: float
  same_sign_unwind_release: bool
  steering_rate_deg: float = 0.0
  steer_limit_same_direction: bool = True
  steer_limit_unwind: bool = False
  steer_limit_requested_output: float = 0.0
  steer_limit_applied_output: float = 0.0


@dataclass
class ConservativeOutputShaperResult:
  output_torque: float
  active: bool
  reason: int
  confidence: float
  unshaped_output: float
  output_cap: float


class TorqueConservativeOutputShaper:
  def __init__(self, dt: float = DEFAULT_DT):
    self.dt: float = max(float(dt), 1e-3)
    self._previous_output: float | None = None
    self._recent_shaping_time: float = 0.0
    self._recent_hard_shaping_time: float = 0.0
    self._recent_over_response_time: float = 0.0
    self._recent_actuator_lag_comfort_time: float = 0.0

  def update(self, inputs: ConservativeOutputShaperInputs) -> ConservativeOutputShaperResult:
    output_cap = NORMAL_CAP
    confidence = 0.0
    reason = ConservativeOutputShapingReason.NONE

    if not inputs.active:
      self._reset()
      return self._result(inputs.unshaped_output, inputs.unshaped_output, False, reason, confidence, output_cap)
    if abs(inputs.unshaped_output) < 1e-6:
      self._previous_output = inputs.unshaped_output
      self._recent_shaping_time = max(0.0, self._recent_shaping_time - self.dt)
      self._recent_hard_shaping_time = max(0.0, self._recent_hard_shaping_time - self.dt)
      self._recent_over_response_time = max(0.0, self._recent_over_response_time - self.dt)
      self._recent_actuator_lag_comfort_time = max(0.0, self._recent_actuator_lag_comfort_time - self.dt)
      return self._result(inputs.unshaped_output, inputs.unshaped_output, False, reason, confidence, output_cap)

    desired_sign = sign(inputs.desired_lateral_accel)
    actual_sign = sign(inputs.actual_lateral_accel)
    output_sign = sign(inputs.unshaped_output)
    actual_abs = abs(inputs.actual_lateral_accel)
    desired_abs = abs(inputs.desired_lateral_accel)
    output_reinforces_actual = output_sign != 0.0 and output_sign == actual_sign
    steering_rate_sign = sign(inputs.steering_rate_deg)
    steering_rate_abs = abs(inputs.steering_rate_deg)
    output_reinforces_steering_rate = output_sign != 0.0 and steering_rate_sign != 0.0 and output_sign == steering_rate_sign
    output_opposes_steering_rate = output_sign != 0.0 and steering_rate_sign != 0.0 and output_sign != steering_rate_sign
    sign_conflict = desired_sign != 0.0 and actual_sign != 0.0 and desired_sign != actual_sign and actual_abs > SIGN_THRESHOLD
    over_response = desired_sign != 0.0 and desired_sign == actual_sign and output_reinforces_actual and actual_abs > desired_abs + OVER_RESPONSE_MARGIN
    jerk_delta = abs(inputs.actual_lateral_jerk - inputs.lookahead_lateral_jerk)
    bump_response = (
      abs(inputs.actual_lateral_jerk) > BUMP_JERK_THRESHOLD
      and jerk_delta > BUMP_LOOKAHEAD_DELTA_THRESHOLD
      and abs(inputs.desired_lateral_jerk) < BUMP_JERK_THRESHOLD
    )
    high_output = abs(inputs.unshaped_output) > max(inputs.max_output, 1e-3) * HIGH_OUTPUT_FRACTION
    same_direction_steer_limited = inputs.steer_limited_by_safety and inputs.steer_limit_same_direction and not inputs.steer_limit_unwind
    low_speed_steer_limited = inputs.v_ego < LOW_SPEED_THRESHOLD and same_direction_steer_limited and high_output and not output_opposes_steering_rate
    actuator_lag_comfort = (
      not inputs.steering_pressed and same_direction_steer_limited
      and output_reinforces_steering_rate and steering_rate_abs > ACTUATOR_LAG_COMFORT_START
    )
    stale_actuator_reversal = self._stale_actuator_reversal(inputs, output_sign)
    safety_limited_ramp_cap = self._safety_limited_ramp_cap(inputs, output_sign, actual_sign) if not stale_actuator_reversal else NORMAL_CAP
    same_sign_unwind_release = inputs.same_sign_unwind_release
    clear_under_response_catchup = (
      desired_sign != 0.0
      and output_sign == desired_sign
      and desired_sign * (inputs.desired_lateral_accel - inputs.actual_lateral_accel) > UNDER_RESPONSE_MARGIN
      and actual_abs <= ISO_ACCEL_MARGIN
      and not inputs.steering_pressed
      and not inputs.release_active
      and not same_sign_unwind_release
      and not sign_conflict
      and not bump_response
    )
    strong_under_response_catchup = (
      clear_under_response_catchup
      and desired_sign * (inputs.desired_lateral_accel - inputs.actual_lateral_accel) > AUTHORITY_RECOVERY_BYPASS_UNDER_RESPONSE
    )

    if inputs.steering_pressed:
      output_cap, confidence, reason = self._apply(output_cap, confidence, reason, OVERRIDE_RELEASE_CAP, 1.0,
                                                   ConservativeOutputShapingReason.STEERING_PRESSED)
    if inputs.release_active:
      output_cap, confidence, reason = self._apply(output_cap, confidence, reason, OVERRIDE_RELEASE_CAP, 1.0,
                                                   ConservativeOutputShapingReason.RELEASE)
    if same_sign_unwind_release:
      output_cap, confidence, reason = self._apply(output_cap, confidence, reason, SAME_SIGN_UNWIND_CAP, 1.0,
                                                   ConservativeOutputShapingReason.SAME_SIGN_UNWIND)
    if sign_conflict:
      output_cap, confidence, reason = self._apply(output_cap, confidence, reason, SIGN_CONFLICT_CAP, 1.0,
                                                   ConservativeOutputShapingReason.SIGN_CONFLICT)
    if over_response:
      over_excess = actual_abs - desired_abs - OVER_RESPONSE_MARGIN
      over_confidence = clamp(over_excess / max(0.4, desired_abs * 0.5), 0.0, 1.0)
      over_cap = OVER_RESPONSE_CAP if inputs.steering_pressed else self._over_response_cap(over_excess)
      output_cap, confidence, reason = self._apply(output_cap, confidence, reason, over_cap, over_confidence,
                                                   ConservativeOutputShapingReason.OVER_RESPONSE)
    if output_reinforces_actual and actual_abs > ISO_ACCEL_MARGIN:
      iso_confidence = clamp((actual_abs - ISO_ACCEL_MARGIN) / max(ISO_LATERAL_ACCEL - ISO_ACCEL_MARGIN, 1e-3), 0.0, 1.0)
      iso_cap = OVER_ISO_ACCEL_CAP if actual_abs > ISO_LATERAL_ACCEL else NEAR_ISO_ACCEL_CAP
      output_cap, confidence, reason = self._apply(output_cap, confidence, reason, iso_cap, iso_confidence,
                                                   ConservativeOutputShapingReason.NEAR_ISO_ACCEL)
    if bump_response:
      bump_confidence = clamp(max(abs(inputs.actual_lateral_jerk) - BUMP_JERK_THRESHOLD,
                                  jerk_delta - BUMP_LOOKAHEAD_DELTA_THRESHOLD) / BUMP_JERK_THRESHOLD, 0.0, 1.0)
      output_cap, confidence, reason = self._apply(output_cap, confidence, reason, BUMP_CAP, bump_confidence,
                                                   ConservativeOutputShapingReason.BUMP)
    if low_speed_steer_limited and not clear_under_response_catchup:
      output_fraction = abs(inputs.unshaped_output) / max(inputs.max_output, 1e-3)
      steer_limit_confidence = clamp((output_fraction - HIGH_OUTPUT_FRACTION) / max(1.0 - HIGH_OUTPUT_FRACTION, 1e-3), 0.0, 1.0)
      output_cap, confidence, reason = self._apply(output_cap, confidence, reason, LOW_SPEED_STEER_LIMITED_CAP, steer_limit_confidence,
                                                   ConservativeOutputShapingReason.LOW_SPEED_STEER_LIMITED)
    if output_reinforces_steering_rate and steering_rate_abs > STEERING_RATE_COMFORT_START and not clear_under_response_catchup:
      steering_rate_confidence = clamp(
        (steering_rate_abs - STEERING_RATE_COMFORT_START) / max(STEERING_RATE_COMFORT_FULL - STEERING_RATE_COMFORT_START, 1e-3),
        0.0,
        1.0,
      )
      steering_rate_cap = NORMAL_CAP + steering_rate_confidence * (STEERING_RATE_COMFORT_MIN_CAP - NORMAL_CAP)
      output_cap, confidence, reason = self._apply(output_cap, confidence, reason, steering_rate_cap, steering_rate_confidence,
                                                   ConservativeOutputShapingReason.STEERING_RATE_COMFORT)
    if actuator_lag_comfort and not clear_under_response_catchup:
      actuator_lag_confidence = clamp(
        (steering_rate_abs - ACTUATOR_LAG_COMFORT_START) / max(STEERING_RATE_COMFORT_FULL - ACTUATOR_LAG_COMFORT_START, 1e-3),
        0.0,
        1.0,
      )
      output_cap, confidence, reason = self._apply(output_cap, confidence, reason, self._actuator_lag_comfort_cap(inputs.v_ego),
                                                    actuator_lag_confidence, ConservativeOutputShapingReason.ACTUATOR_LAG_COMFORT)
    if stale_actuator_reversal:
      output_cap, confidence, reason = self._apply(output_cap, confidence, reason, STALE_ACTUATOR_REVERSAL_CAP, 1.0,
                                                   ConservativeOutputShapingReason.STALE_ACTUATOR_REVERSAL)
    if safety_limited_ramp_cap < NORMAL_CAP:
      output_cap, confidence, reason = self._apply(output_cap, confidence, reason, safety_limited_ramp_cap, 1.0,
                                                   ConservativeOutputShapingReason.SAFETY_LIMITED_RAMP)
    high_speed_actuator_lag_unwind = (
      inputs.v_ego >= HIGH_SPEED_ACTUATOR_LAG_UNWIND_SPEED
      and not inputs.steering_pressed
      and inputs.steer_limited_by_safety
      and inputs.steer_limit_same_direction
      and not inputs.steer_limit_unwind
      and output_reinforces_actual
      and actual_abs > desired_abs + HIGH_SPEED_ACTUATOR_LAG_UNWIND_MARGIN
      and abs(inputs.steer_limit_requested_output - inputs.steer_limit_applied_output) > HIGH_SPEED_ACTUATOR_LAG_UNWIND_GAP
    )
    if high_speed_actuator_lag_unwind:
      output_cap, confidence, reason = self._apply(output_cap, confidence, reason, HIGH_SPEED_ACTUATOR_LAG_UNWIND_CAP, 1.0,
                                                   ConservativeOutputShapingReason.HIGH_SPEED_ACTUATOR_LAG_UNWIND)

    base_active = reason != ConservativeOutputShapingReason.NONE and output_cap < NORMAL_CAP
    same_sign_unwind_shaping = bool(reason & ConservativeOutputShapingReason.SAME_SIGN_UNWIND)
    over_response_shaping = bool(reason & ConservativeOutputShapingReason.OVER_RESPONSE)
    hard_shaping = bool(reason & (
      ConservativeOutputShapingReason.STEERING_PRESSED
      | ConservativeOutputShapingReason.RELEASE
      | ConservativeOutputShapingReason.SIGN_CONFLICT
      | ConservativeOutputShapingReason.OVER_RESPONSE
      | ConservativeOutputShapingReason.NEAR_ISO_ACCEL
      | ConservativeOutputShapingReason.BUMP
      | ConservativeOutputShapingReason.HIGH_SPEED_ACTUATOR_LAG_UNWIND
    ))
    steering_rate_comfort_shaping = bool(reason & ConservativeOutputShapingReason.STEERING_RATE_COMFORT)
    actuator_lag_comfort_shaping = bool(reason & ConservativeOutputShapingReason.ACTUATOR_LAG_COMFORT)
    stale_actuator_reversal_shaping = bool(reason & ConservativeOutputShapingReason.STALE_ACTUATOR_REVERSAL)
    recently_shaped = self._recent_shaping_time > 0.0
    recently_hard_shaped = self._recent_hard_shaping_time > 0.0
    recently_over_response = self._recent_over_response_time > 0.0
    recently_actuator_lag_comfort = self._recent_actuator_lag_comfort_time > 0.0
    shaped_output = inputs.unshaped_output * output_cap if base_active else inputs.unshaped_output
    if abs(shaped_output) > abs(inputs.unshaped_output):
      shaped_output = inputs.unshaped_output
    shaped_output, rate_limited = self._apply_output_rate_limit(inputs, shaped_output, recently_shaped, recently_over_response,
                                                                steering_rate_comfort_shaping, actuator_lag_comfort_shaping,
                                                                stale_actuator_reversal_shaping, recently_actuator_lag_comfort,
                                                                clear_under_response_catchup, recently_hard_shaped,
                                                                strong_under_response_catchup)
    if rate_limited:
      reason |= ConservativeOutputShapingReason.OUTPUT_RATE_LIMITED
      confidence = max(confidence, 1.0)
      output_cap = min(output_cap, abs(shaped_output) / max(abs(inputs.unshaped_output), 1e-6))

    if base_active and not same_sign_unwind_shaping:
      self._recent_shaping_time = OUTPUT_RATE_RECOVERY_WINDOW
    else:
      self._recent_shaping_time = max(0.0, self._recent_shaping_time - self.dt)
    if base_active and over_response_shaping:
      self._recent_over_response_time = OUTPUT_RATE_RECOVERY_WINDOW
    else:
      self._recent_over_response_time = max(0.0, self._recent_over_response_time - self.dt)
    if base_active and hard_shaping:
      self._recent_hard_shaping_time = OUTPUT_RATE_RECOVERY_WINDOW
    else:
      self._recent_hard_shaping_time = max(0.0, self._recent_hard_shaping_time - self.dt)
    if base_active and actuator_lag_comfort_shaping:
      self._recent_actuator_lag_comfort_time = OUTPUT_RATE_RECOVERY_WINDOW
    else:
      self._recent_actuator_lag_comfort_time = max(0.0, self._recent_actuator_lag_comfort_time - self.dt)

    self._previous_output = shaped_output
    active = base_active or rate_limited
    return self._result(shaped_output, inputs.unshaped_output, active, reason, confidence if active else 0.0, output_cap)

  def _apply_output_rate_limit(self, inputs: ConservativeOutputShaperInputs, target_output: float,
                               recently_shaped: bool, recently_over_response: bool,
                               steering_rate_comfort_shaping: bool, actuator_lag_comfort_shaping: bool,
                               stale_actuator_reversal_shaping: bool, recently_actuator_lag_comfort: bool,
                               clear_under_response_catchup: bool, recently_hard_shaped: bool,
                               strong_under_response_catchup: bool) -> tuple[float, bool]:
    if (
      self._previous_output is None
      or (not recently_shaped and not steering_rate_comfort_shaping and not actuator_lag_comfort_shaping and not stale_actuator_reversal_shaping)
      or abs(target_output) < 1e-6
    ):
      return target_output, False

    target_sign = sign(target_output)
    previous_sign = sign(self._previous_output)
    target_abs = abs(target_output)
    previous_abs = abs(self._previous_output)
    if strong_under_response_catchup and not stale_actuator_reversal_shaping and target_sign == previous_sign:
      return target_output, False
    if clear_under_response_catchup and not recently_hard_shaped and not stale_actuator_reversal_shaping:
      return target_output, False

    actual_abs = abs(inputs.actual_lateral_accel)
    actual_sign = sign(inputs.actual_lateral_accel)
    desired_sign = sign(inputs.desired_lateral_accel)
    steering_rate_sign = sign(inputs.steering_rate_deg)
    steering_rate_abs = abs(inputs.steering_rate_deg)
    opposes_steering_rate = (
      target_sign != 0.0 and steering_rate_sign != 0.0 and target_sign != steering_rate_sign
      and (steering_rate_abs > STEERING_RATE_COMFORT_START or recently_actuator_lag_comfort)
    )
    corrective_near_iso = (
      target_sign != 0.0 and actual_sign != 0.0 and target_sign != actual_sign
      and actual_abs > ISO_ACCEL_MARGIN
    )
    corrective_over_response = (
      target_sign != 0.0 and actual_sign != 0.0 and target_sign != actual_sign
      and desired_sign == actual_sign
      and actual_abs > abs(inputs.desired_lateral_accel) + OVER_RESPONSE_MARGIN
    )
    corrective_recent_over_response = (
      recently_over_response and target_sign != 0.0 and actual_sign != 0.0 and target_sign != actual_sign
      and actual_abs > SIGN_THRESHOLD
    )
    if corrective_near_iso or corrective_over_response or corrective_recent_over_response or opposes_steering_rate:
      return target_output, False

    if target_sign != 0.0 and previous_sign != 0.0 and target_sign != previous_sign:
      limited_abs = min(target_abs, OUTPUT_SIGN_TRANSITION_RATE * self.dt)
      limited_output = target_sign * limited_abs
      return limited_output, limited_abs < target_abs

    if target_abs <= previous_abs:
      return target_output, False

    reinforces_steering_rate = target_sign != 0.0 and steering_rate_sign != 0.0 and target_sign == steering_rate_sign
    if stale_actuator_reversal_shaping:
      recovery_rate = STALE_ACTUATOR_REVERSAL_RATE
    elif actuator_lag_comfort_shaping and reinforces_steering_rate:
      recovery_rate = ACTUATOR_LAG_COMFORT_RATE
    elif steering_rate_comfort_shaping and reinforces_steering_rate:
      recovery_rate = STEERING_RATE_COMFORT_RATE
    elif recently_hard_shaped:
      recovery_rate = self._authority_recovery_rate(inputs.v_ego)
    else:
      recovery_rate = OUTPUT_RECOVERY_RATE
    limited_abs = min(target_abs, previous_abs + recovery_rate * self.dt)
    limited_output = target_sign * limited_abs
    return limited_output, limited_abs < target_abs

  @staticmethod
  def _stale_actuator_reversal(inputs: ConservativeOutputShaperInputs, output_sign: float) -> bool:
    requested_sign = sign(inputs.steer_limit_requested_output)
    applied_sign = sign(inputs.steer_limit_applied_output)
    return (
      inputs.steer_limited_by_safety
      and inputs.v_ego < ACTUATOR_LAG_COMFORT_MID_SPEED
      and inputs.steer_limit_same_direction
      and not inputs.steer_limit_unwind
      and not inputs.steering_pressed
      and output_sign != 0.0
      and requested_sign == output_sign
      and applied_sign == -output_sign
      and abs(inputs.steer_limit_applied_output) > STALE_ACTUATOR_REVERSAL_THRESHOLD
    )

  @staticmethod
  def _safety_limited_ramp_cap(inputs: ConservativeOutputShaperInputs, output_sign: float, actual_sign: float) -> float:
    requested_sign = sign(inputs.steer_limit_requested_output)
    applied_sign = sign(inputs.steer_limit_applied_output)
    actuator_error = abs(inputs.steer_limit_requested_output - inputs.steer_limit_applied_output)
    if not (
      inputs.steer_limited_by_safety
      and inputs.steer_limit_same_direction
      and not inputs.steer_limit_unwind
      and not inputs.steering_pressed
      and output_sign != 0.0
      and actual_sign == output_sign
      and requested_sign == output_sign
      and applied_sign in (0.0, output_sign)
      and actuator_error > SAFETY_LIMITED_RAMP_ERROR_THRESHOLD
    ):
      return NORMAL_CAP

    applied_follow_cap = abs(inputs.steer_limit_applied_output) + SAFETY_LIMITED_RAMP_FOLLOW_MARGIN
    cap = applied_follow_cap / max(abs(inputs.unshaped_output), 1e-6)
    if TorqueConservativeOutputShaper._low_speed_under_response_recovery_allowed(inputs, output_sign, applied_sign):
      cap = max(cap, TorqueConservativeOutputShaper._low_speed_under_response_cap_floor(inputs.v_ego))
    return clamp(cap, 0.0, NORMAL_CAP)

  @staticmethod
  def _low_speed_under_response_recovery_allowed(inputs: ConservativeOutputShaperInputs, output_sign: float, applied_sign: float) -> bool:
    desired_sign = sign(inputs.desired_lateral_accel)
    if desired_sign == 0.0 or output_sign != desired_sign:
      return False

    under_response = desired_sign * (inputs.desired_lateral_accel - inputs.actual_lateral_accel)
    jerk_delta = abs(inputs.actual_lateral_jerk - inputs.lookahead_lateral_jerk)
    bump_response = (
      abs(inputs.actual_lateral_jerk) > BUMP_JERK_THRESHOLD
      and jerk_delta > BUMP_LOOKAHEAD_DELTA_THRESHOLD
      and abs(inputs.desired_lateral_jerk) < BUMP_JERK_THRESHOLD
    )
    return (
      inputs.v_ego < SAFETY_LIMITED_RAMP_UNDER_RESPONSE_FADE_SPEED
      and applied_sign == output_sign
      and abs(inputs.steer_limit_applied_output) > SAFETY_LIMITED_RAMP_APPLIED_RECOVERY_THRESHOLD
      and under_response > UNDER_RESPONSE_MARGIN
      and abs(inputs.actual_lateral_accel) <= ISO_ACCEL_MARGIN
      and not inputs.release_active
      and not inputs.same_sign_unwind_release
      and not bump_response
    )

  @staticmethod
  def _low_speed_under_response_cap_floor(v_ego: float) -> float:
    if v_ego <= SAFETY_LIMITED_RAMP_UNDER_RESPONSE_FULL_SPEED:
      return SAFETY_LIMITED_RAMP_UNDER_RESPONSE_FLOOR
    span = SAFETY_LIMITED_RAMP_UNDER_RESPONSE_FADE_SPEED - SAFETY_LIMITED_RAMP_UNDER_RESPONSE_FULL_SPEED
    fade = (SAFETY_LIMITED_RAMP_UNDER_RESPONSE_FADE_SPEED - v_ego) / max(span, 1e-3)
    return SAFETY_LIMITED_RAMP_UNDER_RESPONSE_FLOOR * clamp(fade, 0.0, 1.0)

  @staticmethod
  def _apply(output_cap: float, confidence: float, reason: ConservativeOutputShapingReason, cap: float, reason_confidence: float,
             reason_flag: ConservativeOutputShapingReason) -> tuple[float, float, ConservativeOutputShapingReason]:
    return min(output_cap, cap), max(confidence, reason_confidence), reason | reason_flag

  @staticmethod
  def _over_response_cap(over_excess: float) -> float:
    if over_excess <= OVER_RESPONSE_DYNAMIC_START:
      return OVER_RESPONSE_CAP
    if over_excess >= OVER_RESPONSE_SEVERE_EXCESS:
      return OVER_RESPONSE_SEVERE_CAP
    if over_excess <= OVER_RESPONSE_MODERATE_EXCESS:
      span = OVER_RESPONSE_MODERATE_EXCESS - OVER_RESPONSE_DYNAMIC_START
      ratio = (over_excess - OVER_RESPONSE_DYNAMIC_START) / max(span, 1e-3)
      return OVER_RESPONSE_CAP + ratio * (OVER_RESPONSE_MODERATE_CAP - OVER_RESPONSE_CAP)
    span = OVER_RESPONSE_SEVERE_EXCESS - OVER_RESPONSE_MODERATE_EXCESS
    ratio = (over_excess - OVER_RESPONSE_MODERATE_EXCESS) / max(span, 1e-3)
    return OVER_RESPONSE_MODERATE_CAP + ratio * (OVER_RESPONSE_SEVERE_CAP - OVER_RESPONSE_MODERATE_CAP)

  @staticmethod
  def _actuator_lag_comfort_cap(v_ego: float) -> float:
    if v_ego <= ACTUATOR_LAG_COMFORT_LOW_SPEED:
      return ACTUATOR_LAG_COMFORT_LOW_SPEED_CAP
    if v_ego <= ACTUATOR_LAG_COMFORT_MID_SPEED:
      span = ACTUATOR_LAG_COMFORT_MID_SPEED - ACTUATOR_LAG_COMFORT_LOW_SPEED
      ratio = (v_ego - ACTUATOR_LAG_COMFORT_LOW_SPEED) / max(span, 1e-3)
      return ACTUATOR_LAG_COMFORT_LOW_SPEED_CAP + ratio * (ACTUATOR_LAG_COMFORT_MID_SPEED_CAP - ACTUATOR_LAG_COMFORT_LOW_SPEED_CAP)
    if v_ego >= ACTUATOR_LAG_COMFORT_HIGH_SPEED:
      return ACTUATOR_LAG_COMFORT_HIGH_SPEED_CAP

    span = ACTUATOR_LAG_COMFORT_HIGH_SPEED - ACTUATOR_LAG_COMFORT_MID_SPEED
    ratio = (v_ego - ACTUATOR_LAG_COMFORT_MID_SPEED) / max(span, 1e-3)
    return ACTUATOR_LAG_COMFORT_MID_SPEED_CAP + ratio * (ACTUATOR_LAG_COMFORT_HIGH_SPEED_CAP - ACTUATOR_LAG_COMFORT_MID_SPEED_CAP)

  @staticmethod
  def _authority_recovery_rate(v_ego: float) -> float:
    if v_ego <= AUTHORITY_RECOVERY_LOW_SPEED:
      return AUTHORITY_RECOVERY_LOW_SPEED_RATE
    if v_ego <= AUTHORITY_RECOVERY_MID_SPEED:
      span = AUTHORITY_RECOVERY_MID_SPEED - AUTHORITY_RECOVERY_LOW_SPEED
      ratio = (v_ego - AUTHORITY_RECOVERY_LOW_SPEED) / max(span, 1e-3)
      return AUTHORITY_RECOVERY_LOW_SPEED_RATE + ratio * (OUTPUT_RECOVERY_RATE - AUTHORITY_RECOVERY_LOW_SPEED_RATE)
    if v_ego >= AUTHORITY_RECOVERY_HIGH_SPEED:
      return AUTHORITY_RECOVERY_HIGH_SPEED_RATE

    span = AUTHORITY_RECOVERY_HIGH_SPEED - AUTHORITY_RECOVERY_MID_SPEED
    ratio = (v_ego - AUTHORITY_RECOVERY_MID_SPEED) / max(span, 1e-3)
    return OUTPUT_RECOVERY_RATE + ratio * (AUTHORITY_RECOVERY_HIGH_SPEED_RATE - OUTPUT_RECOVERY_RATE)

  def _reset(self) -> None:
    self._previous_output = None
    self._recent_shaping_time = 0.0
    self._recent_hard_shaping_time = 0.0
    self._recent_over_response_time = 0.0
    self._recent_actuator_lag_comfort_time = 0.0

  @staticmethod
  def _result(output_torque: float, unshaped_output: float, active: bool, reason: ConservativeOutputShapingReason, confidence: float,
              output_cap: float) -> ConservativeOutputShaperResult:
    return ConservativeOutputShaperResult(
      output_torque=output_torque,
      active=active,
      reason=int(reason),
      confidence=float(clamp(confidence, 0.0, 1.0)),
      unshaped_output=unshaped_output,
      output_cap=float(output_cap),
    )
