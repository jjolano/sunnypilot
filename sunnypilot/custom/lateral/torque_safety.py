import math


TORQUE_OVERRIDE_LAT_ACCEL_FACTOR_MIN = 0.1
TORQUE_OVERRIDE_LAT_ACCEL_FACTOR_MAX = 5.0
TORQUE_OVERRIDE_LAT_ACCEL_FACTOR_DEFAULT = 2.5

TORQUE_OVERRIDE_FRICTION_MIN = 0.0
TORQUE_OVERRIDE_FRICTION_MAX = 1.0
TORQUE_OVERRIDE_FRICTION_DEFAULT = 0.1

LIVE_TORQUE_SPEED_ADAPTIVE_MODES = ("off", "shadow", "apply")


def finite_float(value) -> float | None:
  try:
    parsed = float(value)
  except (TypeError, ValueError):
    return None
  return parsed if math.isfinite(parsed) else None


def validate_torque_override_lat_accel_factor(value) -> float | None:
  parsed = finite_float(value)
  if parsed is None:
    return None
  if not (TORQUE_OVERRIDE_LAT_ACCEL_FACTOR_MIN <= parsed <= TORQUE_OVERRIDE_LAT_ACCEL_FACTOR_MAX):
    return None
  return parsed


def validate_torque_override_friction(value) -> float | None:
  parsed = finite_float(value)
  if parsed is None:
    return None
  if not (TORQUE_OVERRIDE_FRICTION_MIN <= parsed <= TORQUE_OVERRIDE_FRICTION_MAX):
    return None
  return parsed


def validate_live_torque_speed_adaptive_mode(value) -> str:
  value = value or "off"
  return value if value in LIVE_TORQUE_SPEED_ADAPTIVE_MODES else "off"
