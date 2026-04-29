from dataclasses import dataclass
from enum import IntFlag


ISO_LATERAL_ACCEL = 3.0
ISO_ACCEL_MARGIN = 2.6
SIGN_THRESHOLD = 0.05
OVER_RESPONSE_MARGIN = 0.12
BUMP_JERK_THRESHOLD = 2.0
BUMP_LOOKAHEAD_DELTA_THRESHOLD = 1.4
LOW_SPEED_THRESHOLD = 8.0
HIGH_OUTPUT_FRACTION = 0.70
OUTPUT_RATE_RECOVERY_WINDOW = 0.40
OUTPUT_RECOVERY_RATE = 2.50
OUTPUT_SIGN_TRANSITION_RATE = 4.00
DEFAULT_DT = 0.01

NORMAL_CAP = 1.00
LOW_SPEED_STEER_LIMITED_CAP = 0.92
BUMP_CAP = 0.90
NEAR_ISO_ACCEL_CAP = 0.85
OVER_ISO_ACCEL_CAP = 0.80
OVER_RESPONSE_CAP = 0.85
SIGN_CONFLICT_CAP = 0.80
OVERRIDE_RELEASE_CAP = 0.80
SAME_SIGN_UNWIND_CAP = 0.30


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
    self.dt = max(float(dt), 1e-3)
    self._previous_output: float | None = None
    self._recent_shaping_time = 0.0

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
      return self._result(inputs.unshaped_output, inputs.unshaped_output, False, reason, confidence, output_cap)

    desired_sign = sign(inputs.desired_lateral_accel)
    actual_sign = sign(inputs.actual_lateral_accel)
    output_sign = sign(inputs.unshaped_output)
    actual_abs = abs(inputs.actual_lateral_accel)
    desired_abs = abs(inputs.desired_lateral_accel)
    output_reinforces_actual = output_sign != 0.0 and output_sign == actual_sign
    sign_conflict = desired_sign != 0.0 and actual_sign != 0.0 and desired_sign != actual_sign and actual_abs > SIGN_THRESHOLD
    over_response = desired_sign != 0.0 and desired_sign == actual_sign and output_reinforces_actual and actual_abs > desired_abs + OVER_RESPONSE_MARGIN
    jerk_delta = abs(inputs.actual_lateral_jerk - inputs.lookahead_lateral_jerk)
    bump_response = (
      abs(inputs.actual_lateral_jerk) > BUMP_JERK_THRESHOLD
      and jerk_delta > BUMP_LOOKAHEAD_DELTA_THRESHOLD
      and abs(inputs.desired_lateral_jerk) < BUMP_JERK_THRESHOLD
    )
    high_output = abs(inputs.unshaped_output) > max(inputs.max_output, 1e-3) * HIGH_OUTPUT_FRACTION
    low_speed_steer_limited = inputs.v_ego < LOW_SPEED_THRESHOLD and inputs.steer_limited_by_safety and high_output
    same_sign_unwind_release = inputs.same_sign_unwind_release

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
      over_confidence = clamp((actual_abs - desired_abs - OVER_RESPONSE_MARGIN) / max(0.4, desired_abs * 0.5), 0.0, 1.0)
      output_cap, confidence, reason = self._apply(output_cap, confidence, reason, OVER_RESPONSE_CAP, over_confidence,
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
    if low_speed_steer_limited:
      output_fraction = abs(inputs.unshaped_output) / max(inputs.max_output, 1e-3)
      steer_limit_confidence = clamp((output_fraction - HIGH_OUTPUT_FRACTION) / max(1.0 - HIGH_OUTPUT_FRACTION, 1e-3), 0.0, 1.0)
      output_cap, confidence, reason = self._apply(output_cap, confidence, reason, LOW_SPEED_STEER_LIMITED_CAP, steer_limit_confidence,
                                                   ConservativeOutputShapingReason.LOW_SPEED_STEER_LIMITED)

    base_active = reason != ConservativeOutputShapingReason.NONE and output_cap < NORMAL_CAP
    same_sign_unwind_shaping = bool(reason & ConservativeOutputShapingReason.SAME_SIGN_UNWIND)
    recently_shaped = self._recent_shaping_time > 0.0
    shaped_output = inputs.unshaped_output * output_cap if base_active else inputs.unshaped_output
    if abs(shaped_output) > abs(inputs.unshaped_output):
      shaped_output = inputs.unshaped_output
    shaped_output, rate_limited = self._apply_output_rate_limit(inputs, shaped_output, recently_shaped)
    if rate_limited:
      reason |= ConservativeOutputShapingReason.OUTPUT_RATE_LIMITED
      confidence = max(confidence, 1.0)
      output_cap = min(output_cap, abs(shaped_output) / max(abs(inputs.unshaped_output), 1e-6))

    if base_active and not same_sign_unwind_shaping:
      self._recent_shaping_time = OUTPUT_RATE_RECOVERY_WINDOW
    else:
      self._recent_shaping_time = max(0.0, self._recent_shaping_time - self.dt)

    self._previous_output = shaped_output
    active = base_active or rate_limited
    return self._result(shaped_output, inputs.unshaped_output, active, reason, confidence if active else 0.0, output_cap)

  def _apply_output_rate_limit(self, inputs: ConservativeOutputShaperInputs, target_output: float,
                               recently_shaped: bool) -> tuple[float, bool]:
    if self._previous_output is None or not recently_shaped or abs(target_output) < 1e-6:
      return target_output, False

    target_sign = sign(target_output)
    previous_sign = sign(self._previous_output)
    target_abs = abs(target_output)
    previous_abs = abs(self._previous_output)
    actual_sign = sign(inputs.actual_lateral_accel)
    corrective_near_iso = (
      target_sign != 0.0 and actual_sign != 0.0 and target_sign != actual_sign
      and abs(inputs.actual_lateral_accel) > ISO_ACCEL_MARGIN
    )
    if corrective_near_iso:
      return target_output, False

    if target_sign != 0.0 and previous_sign != 0.0 and target_sign != previous_sign:
      limited_abs = min(target_abs, OUTPUT_SIGN_TRANSITION_RATE * self.dt)
      limited_output = target_sign * limited_abs
      return limited_output, limited_abs < target_abs

    if target_abs <= previous_abs:
      return target_output, False

    limited_abs = min(target_abs, previous_abs + OUTPUT_RECOVERY_RATE * self.dt)
    limited_output = target_sign * limited_abs
    return limited_output, limited_abs < target_abs

  @staticmethod
  def _apply(output_cap: float, confidence: float, reason: ConservativeOutputShapingReason, cap: float, reason_confidence: float,
             reason_flag: ConservativeOutputShapingReason) -> tuple[float, float, ConservativeOutputShapingReason]:
    return min(output_cap, cap), max(confidence, reason_confidence), reason | reason_flag

  def _reset(self) -> None:
    self._previous_output = None
    self._recent_shaping_time = 0.0

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
