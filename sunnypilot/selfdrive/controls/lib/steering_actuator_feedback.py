from __future__ import annotations

from dataclasses import dataclass
from enum import IntFlag
import math

from cereal import car


TORQUE_LIMIT_THRESHOLD = 1e-2
ANGLE_LIMIT_THRESHOLD_DEG = 2.5
CURVATURE_LIMIT_THRESHOLD = 1e-4
SIGN_EPS = 1e-6


class SteeringLimitReason(IntFlag):
  NONE = 0
  ACTUATOR_MISMATCH = 1 << 0
  INACTIVE = 1 << 1
  UNAVAILABLE = 1 << 2


@dataclass(frozen=True)
class SteeringActuatorRequest:
  torque: float
  steering_angle_deg: float
  curvature: float

  @classmethod
  def from_actuators(cls, actuators) -> SteeringActuatorRequest:
    return cls(
      torque=float(actuators.torque),
      steering_angle_deg=float(actuators.steeringAngleDeg),
      curvature=float(actuators.curvature),
    )


@dataclass(frozen=True)
class SteeringActuatorFeedback:
  valid: bool
  limited: bool
  reason: SteeringLimitReason
  requested: float
  applied: float
  error: float
  same_direction_limited: bool
  unwind_allowed: bool

  @classmethod
  def invalid(cls, reason: SteeringLimitReason = SteeringLimitReason.UNAVAILABLE) -> SteeringActuatorFeedback:
    return cls(False, False, reason, 0.0, 0.0, 0.0, False, False)


def sign(value: float) -> int:
  if value > SIGN_EPS:
    return 1
  if value < -SIGN_EPS:
    return -1
  return 0


def build_steering_actuator_feedback(
  requested: SteeringActuatorRequest | None,
  applied_actuators,
  steer_control_type: car.CarParams.SteerControlType,
  current_command: float | None = None,
  lat_active: bool = True,
) -> SteeringActuatorFeedback:
  if not lat_active:
    return SteeringActuatorFeedback.invalid(SteeringLimitReason.INACTIVE)
  if requested is None or applied_actuators is None:
    return SteeringActuatorFeedback.invalid()

  requested_value, applied_value, threshold = _values_for_control_type(requested, applied_actuators, steer_control_type)
  if not math.isfinite(requested_value) or not math.isfinite(applied_value):
    return SteeringActuatorFeedback.invalid()

  error = requested_value - applied_value
  limited = abs(error) > threshold
  reason = SteeringLimitReason.ACTUATOR_MISMATCH if limited else SteeringLimitReason.NONE

  same_direction_limited, unwind_allowed = classify_steering_limit_direction(
    SteeringActuatorFeedback(True, limited, reason, requested_value, applied_value, error, False, False),
    current_command,
  )

  return SteeringActuatorFeedback(
    valid=True,
    limited=limited,
    reason=reason,
    requested=requested_value,
    applied=applied_value,
    error=error,
    same_direction_limited=same_direction_limited,
    unwind_allowed=unwind_allowed,
  )


def _values_for_control_type(requested: SteeringActuatorRequest, applied_actuators, steer_control_type: car.CarParams.SteerControlType) -> tuple[float, float, float]:
  if steer_control_type == car.CarParams.SteerControlType.angle:
    return requested.steering_angle_deg, float(applied_actuators.steeringAngleDeg), ANGLE_LIMIT_THRESHOLD_DEG
  if steer_control_type == car.CarParams.SteerControlType.curvatureDEPRECATED:
    return requested.curvature, float(applied_actuators.curvature), CURVATURE_LIMIT_THRESHOLD
  return requested.torque, float(applied_actuators.torque), TORQUE_LIMIT_THRESHOLD


def classify_steering_limit_direction(feedback: SteeringActuatorFeedback, current_command: float | None) -> tuple[bool, bool]:
  if current_command is None or not feedback.valid or not feedback.limited:
    return False, False

  error_sign = sign(feedback.error)
  command_sign = sign(float(current_command))
  if error_sign == 0 or command_sign == 0:
    return False, False

  return command_sign == error_sign, command_sign == -error_sign
