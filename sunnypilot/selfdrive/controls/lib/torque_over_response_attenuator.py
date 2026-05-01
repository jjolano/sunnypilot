OVER_RESPONSE_ATTENUATION_MARGIN = 0.12
OVER_RESPONSE_ATTENUATION_FULL_EXCESS = 0.60
OVER_RESPONSE_ATTENUATION_MIN_SCALE = 0.30


def _clamp(value: float, lower: float, upper: float) -> float:
  return max(lower, min(upper, value))


def _sign(value: float) -> float:
  return 1.0 if value > 0.0 else (-1.0 if value < 0.0 else 0.0)


def attenuate_same_direction_over_response(nominal_torque: float, desired_lateral_accel: float, actual_lateral_accel: float) -> float:
  desired_sign = _sign(desired_lateral_accel)
  actual_sign = _sign(actual_lateral_accel)
  torque_sign = _sign(nominal_torque)
  if desired_sign == 0.0 or actual_sign != desired_sign or torque_sign != actual_sign:
    return nominal_torque

  over_response = desired_sign * (actual_lateral_accel - desired_lateral_accel)
  if over_response <= OVER_RESPONSE_ATTENUATION_MARGIN:
    return nominal_torque

  attenuation_span = OVER_RESPONSE_ATTENUATION_FULL_EXCESS - OVER_RESPONSE_ATTENUATION_MARGIN
  ratio = _clamp((over_response - OVER_RESPONSE_ATTENUATION_MARGIN) / max(attenuation_span, 1e-3), 0.0, 1.0)
  scale = 1.0 + ratio * (OVER_RESPONSE_ATTENUATION_MIN_SCALE - 1.0)
  return nominal_torque * scale
