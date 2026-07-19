import math


TORQUE_OVERRIDE_LAT_ACCEL_FACTOR_MIN = 0.1
TORQUE_OVERRIDE_LAT_ACCEL_FACTOR_MAX = 5.0
TORQUE_OVERRIDE_LAT_ACCEL_FACTOR_DEFAULT = 2.5

TORQUE_OVERRIDE_FRICTION_MIN = 0.0
TORQUE_OVERRIDE_FRICTION_MAX = 1.0
TORQUE_OVERRIDE_FRICTION_DEFAULT = 0.1

# Runtime guardrails for manual overrides. The absolute schema range remains broad enough
# for platform variation, but a value applied onroad must stay close to the car's learned
# or configured baseline to avoid accidental large steering-response changes.
TORQUE_OVERRIDE_LAT_ACCEL_FACTOR_REL_MIN = 0.5
TORQUE_OVERRIDE_LAT_ACCEL_FACTOR_REL_MAX = 1.75
TORQUE_OVERRIDE_FRICTION_ABS_DELTA_MAX = 0.3

LIVE_TORQUE_SPEED_ADAPTIVE_MODES = ("off", "shadow", "apply")
ROLL_COMP_GAIN_MODES = ("off", "shadow", "apply")
FRICTION_BREAKAWAY_MODES = ("off", "shadow", "apply")
DIRECTION_GAIN_MODES = ("off", "shadow", "apply")


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


def validate_manual_torque_override_against_base(lat_accel_factor, friction, base_lat_accel_factor, base_friction) -> bool:
  lat_accel_factor = finite_float(lat_accel_factor)
  friction = finite_float(friction)
  base_lat_accel_factor = finite_float(base_lat_accel_factor)
  base_friction = finite_float(base_friction)
  if lat_accel_factor is None or friction is None or base_lat_accel_factor is None or base_friction is None:
    return False
  if base_lat_accel_factor <= 0.0:
    return False
  min_factor = base_lat_accel_factor * TORQUE_OVERRIDE_LAT_ACCEL_FACTOR_REL_MIN
  max_factor = base_lat_accel_factor * TORQUE_OVERRIDE_LAT_ACCEL_FACTOR_REL_MAX
  if not (min_factor <= lat_accel_factor <= max_factor):
    return False
  if abs(friction - base_friction) > TORQUE_OVERRIDE_FRICTION_ABS_DELTA_MAX:
    return False
  return True


def validate_live_torque_speed_adaptive_mode(value) -> str:
  value = value or "off"
  return value if value in LIVE_TORQUE_SPEED_ADAPTIVE_MODES else "off"


def validate_roll_comp_gain_mode(value) -> str:
  value = value or "off"
  return value if value in ROLL_COMP_GAIN_MODES else "off"


def validate_friction_breakaway_mode(value) -> str:
  value = value or "off"
  return value if value in FRICTION_BREAKAWAY_MODES else "off"


def validate_direction_gain_mode(value) -> str:
  value = value or "off"
  return value if value in DIRECTION_GAIN_MODES else "off"


# Breakaway profile: per-direction rack breakaway medians learned by the shadow
# observer in torqued_ext, persisted for the direction-aware friction floor.
BREAKAWAY_PROFILE_VERSION = 1
BREAKAWAY_PROFILE_MIN_EVENTS = 30
BREAKAWAY_MEDIAN_MIN = 0.02   # normalized EPS torque; below this the data is junk
BREAKAWAY_MEDIAN_MAX = 0.8


def format_breakaway_profile(CP, left: float, right: float, events: int) -> str:
  import json
  return json.dumps({
    'version': BREAKAWAY_PROFILE_VERSION,
    'car': str(CP.carFingerprint),
    'left': float(left),
    'right': float(right),
    'events': int(events),
  })


def parse_breakaway_profile(CP, payload) -> dict | None:
  """Fail closed: any missing/invalid/foreign field distrusts the whole payload."""
  if CP.lateralTuning.which() != 'torque' or not isinstance(payload, dict):
    return None
  if payload.get('version') != BREAKAWAY_PROFILE_VERSION:
    return None
  if payload.get('car') != str(CP.carFingerprint):
    return None
  try:
    left = float(payload['left'])
    right = float(payload['right'])
    events = int(payload['events'])
  except (KeyError, TypeError, ValueError):
    return None
  import math
  if not (math.isfinite(left) and math.isfinite(right)):
    return None
  if events < BREAKAWAY_PROFILE_MIN_EVENTS:
    return None
  if not (BREAKAWAY_MEDIAN_MIN <= left <= BREAKAWAY_MEDIAN_MAX and BREAKAWAY_MEDIAN_MIN <= right <= BREAKAWAY_MEDIAN_MAX):
    return None
  return {'left': left, 'right': right, 'events': events}
