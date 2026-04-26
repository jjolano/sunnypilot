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

NORMAL_CAP = 1.00
LOW_SPEED_STEER_LIMITED_CAP = 0.92
BUMP_CAP = 0.90
NEAR_ISO_ACCEL_CAP = 0.85
OVER_ISO_ACCEL_CAP = 0.80
OVER_RESPONSE_CAP = 0.85
SIGN_CONFLICT_CAP = 0.80
OVERRIDE_RELEASE_CAP = 0.80


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


@dataclass
class ConservativeOutputShaperResult:
  output_torque: float
  active: bool
  reason: int
  confidence: float
  unshaped_output: float
  output_cap: float


class TorqueConservativeOutputShaper:
  def update(self, inputs: ConservativeOutputShaperInputs) -> ConservativeOutputShaperResult:
    output_cap = NORMAL_CAP
    confidence = 0.0
    reason = ConservativeOutputShapingReason.NONE

    if not inputs.active or abs(inputs.unshaped_output) < 1e-6:
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

    if inputs.steering_pressed:
      output_cap, confidence, reason = self._apply(output_cap, confidence, reason, OVERRIDE_RELEASE_CAP, 1.0,
                                                   ConservativeOutputShapingReason.STEERING_PRESSED)
    if inputs.release_active:
      output_cap, confidence, reason = self._apply(output_cap, confidence, reason, OVERRIDE_RELEASE_CAP, 1.0,
                                                   ConservativeOutputShapingReason.RELEASE)
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

    active = reason != ConservativeOutputShapingReason.NONE and output_cap < NORMAL_CAP
    shaped_output = inputs.unshaped_output * output_cap if active else inputs.unshaped_output
    if abs(shaped_output) > abs(inputs.unshaped_output):
      shaped_output = inputs.unshaped_output
    return self._result(shaped_output, inputs.unshaped_output, active, reason, confidence if active else 0.0, output_cap)

  @staticmethod
  def _apply(output_cap: float, confidence: float, reason: ConservativeOutputShapingReason, cap: float, reason_confidence: float,
             reason_flag: ConservativeOutputShapingReason) -> tuple[float, float, ConservativeOutputShapingReason]:
    return min(output_cap, cap), max(confidence, reason_confidence), reason | reason_flag

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
