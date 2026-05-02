from dataclasses import dataclass
from enum import IntEnum, IntFlag

from openpilot.sunnypilot.selfdrive.controls.lib.torque_conservative_output_shaper import (
  BUMP_JERK_THRESHOLD,
  BUMP_LOOKAHEAD_DELTA_THRESHOLD,
  ISO_ACCEL_MARGIN,
  ISO_LATERAL_ACCEL,
  OVER_RESPONSE_MARGIN,
  SIGN_THRESHOLD,
  clamp,
  sign,
)
from openpilot.sunnypilot.selfdrive.controls.lib.torque_guarded_response_assist import RESPONSE_DEFICIT_THRESHOLD


MEASUREMENT_RESET_MIN_VEGO = 5.0


class TorqueDisturbanceReason(IntFlag):
  NONE = 0
  BUMP_JERK = 1 << 0
  SIGN_CONFLICT = 1 << 1
  OVER_RESPONSE = 1 << 2
  HIGH_LATERAL_ACCEL = 1 << 3
  OUTPUT_SATURATED = 1 << 4
  SAFETY_LIMITED = 1 << 5
  CURVATURE_LIMITED = 1 << 6
  LOW_SPEED_UNWIND = 1 << 7
  RESPONSE_DEFICIT = 1 << 8
  MEASUREMENT_RESET_OR_INVALID = 1 << 9


class TorqueDisturbanceState(IntEnum):
  NONE = 0
  SUSPECTED = 1
  ACTIVE = 2


@dataclass
class TorqueDisturbanceInputs:
  active: bool
  v_ego: float
  steering_pressed: bool
  steer_limited_by_safety: bool
  curvature_limited: bool
  saturated: bool
  desired_lateral_accel: float
  actual_lateral_accel: float
  desired_lateral_jerk: float
  actual_lateral_jerk: float
  lookahead_lateral_jerk: float
  output_torque: float
  response_deficit: float
  same_sign_unwind: bool
  measurement_reset: bool
  measurement_valid: bool


@dataclass
class TorqueDisturbanceResult:
  state: TorqueDisturbanceState
  reason: TorqueDisturbanceReason
  confidence: float


def classify_torque_disturbance(inputs: TorqueDisturbanceInputs) -> TorqueDisturbanceResult:
  if not inputs.active:
    return TorqueDisturbanceResult(TorqueDisturbanceState.NONE, TorqueDisturbanceReason.NONE, 0.0)

  state = TorqueDisturbanceState.NONE
  reason = TorqueDisturbanceReason.NONE
  confidence = 0.0

  def add(reason_flag: TorqueDisturbanceReason, new_state: TorqueDisturbanceState, new_confidence: float) -> None:
    nonlocal state, reason, confidence
    reason |= reason_flag
    state = max(state, new_state)
    confidence = max(confidence, clamp(new_confidence, 0.0, 1.0))

  desired_sign = sign(inputs.desired_lateral_accel)
  actual_sign = sign(inputs.actual_lateral_accel)
  output_sign = sign(inputs.output_torque)
  actual_abs = abs(inputs.actual_lateral_accel)
  desired_abs = abs(inputs.desired_lateral_accel)
  output_reinforces_actual = output_sign != 0.0 and output_sign == actual_sign

  jerk_delta = abs(inputs.actual_lateral_jerk - inputs.lookahead_lateral_jerk)
  if (
    abs(inputs.actual_lateral_jerk) > BUMP_JERK_THRESHOLD
    and jerk_delta > BUMP_LOOKAHEAD_DELTA_THRESHOLD
    and abs(inputs.desired_lateral_jerk) < BUMP_JERK_THRESHOLD
  ):
    add(
      TorqueDisturbanceReason.BUMP_JERK,
      TorqueDisturbanceState.ACTIVE,
      max(abs(inputs.actual_lateral_jerk) - BUMP_JERK_THRESHOLD, jerk_delta - BUMP_LOOKAHEAD_DELTA_THRESHOLD) / BUMP_JERK_THRESHOLD,
    )

  if desired_sign != 0.0 and actual_sign != 0.0 and desired_sign != actual_sign and actual_abs > SIGN_THRESHOLD:
    add(TorqueDisturbanceReason.SIGN_CONFLICT, TorqueDisturbanceState.ACTIVE, 1.0)

  if desired_sign != 0.0 and desired_sign == actual_sign and output_reinforces_actual and actual_abs > desired_abs + OVER_RESPONSE_MARGIN:
    over_confidence = (actual_abs - desired_abs - OVER_RESPONSE_MARGIN) / max(0.4, desired_abs * 0.5)
    add(TorqueDisturbanceReason.OVER_RESPONSE, TorqueDisturbanceState.ACTIVE, over_confidence)

  if output_reinforces_actual and actual_abs > ISO_ACCEL_MARGIN:
    high_accel_confidence = (actual_abs - ISO_ACCEL_MARGIN) / max(ISO_LATERAL_ACCEL - ISO_ACCEL_MARGIN, 1e-3)
    add(TorqueDisturbanceReason.HIGH_LATERAL_ACCEL, TorqueDisturbanceState.ACTIVE, high_accel_confidence)

  if inputs.saturated:
    add(TorqueDisturbanceReason.OUTPUT_SATURATED, TorqueDisturbanceState.SUSPECTED, 0.5)
  if inputs.steer_limited_by_safety:
    add(TorqueDisturbanceReason.SAFETY_LIMITED, TorqueDisturbanceState.SUSPECTED, 0.5)
  if inputs.curvature_limited:
    add(TorqueDisturbanceReason.CURVATURE_LIMITED, TorqueDisturbanceState.SUSPECTED, 0.5)
  if inputs.same_sign_unwind:
    add(TorqueDisturbanceReason.LOW_SPEED_UNWIND, TorqueDisturbanceState.ACTIVE, 1.0)

  if abs(inputs.response_deficit) > RESPONSE_DEFICIT_THRESHOLD:
    deficit_confidence = (abs(inputs.response_deficit) - RESPONSE_DEFICIT_THRESHOLD) / 0.2
    add(TorqueDisturbanceReason.RESPONSE_DEFICIT, TorqueDisturbanceState.SUSPECTED, deficit_confidence)

  measurement_problem = inputs.measurement_reset or not inputs.measurement_valid
  if measurement_problem and not inputs.steering_pressed and inputs.v_ego >= MEASUREMENT_RESET_MIN_VEGO:
    add(TorqueDisturbanceReason.MEASUREMENT_RESET_OR_INVALID, TorqueDisturbanceState.SUSPECTED, 1.0)

  if reason == TorqueDisturbanceReason.NONE:
    return TorqueDisturbanceResult(TorqueDisturbanceState.NONE, reason, 0.0)
  return TorqueDisturbanceResult(state, reason, confidence)
