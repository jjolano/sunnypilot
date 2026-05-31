import math


MIN_CURVATURE_FOR_SPEED = 1e-9
MIN_DISTANCE_FOR_DECEL = 0.1


def smooth_speed_floor(v_ego: float, floor: float = 1.0) -> float:
  """Smoothly lower-bound speed magnitude as sqrt(v^2 + floor^2)."""
  return math.hypot(float(v_ego), float(floor))


def required_decel_to_target_speed(v_initial: float, v_target: float, distance: float,
                                   min_distance: float = MIN_DISTANCE_FOR_DECEL) -> float:
  """Signed constant acceleration required to reach target speed over distance."""
  effective_distance = max(float(distance), float(min_distance))
  return (float(v_target) ** 2 - float(v_initial) ** 2) / (2.0 * effective_distance)


def stopping_decel(v_ego: float, distance: float, min_distance: float = MIN_DISTANCE_FOR_DECEL) -> float:
  """Signed constant acceleration required to stop over distance."""
  return required_decel_to_target_speed(v_ego, 0.0, distance, min_distance)


def speed_for_lateral_accel(lateral_accel: float, curvature: float) -> float:
  """Return speed for a lateral acceleration budget and curvature magnitude.

  Returns math.inf when curvature is too small or the lateral acceleration budget
  is invalid. This generic helper does not encode planner/UI sentinel values.
  """
  try:
    lateral_accel = float(lateral_accel)
    curvature = abs(float(curvature))
  except (TypeError, ValueError):
    return math.inf
  if not math.isfinite(lateral_accel) or lateral_accel < 0.0:
    return math.inf
  if not math.isfinite(curvature) or curvature <= MIN_CURVATURE_FOR_SPEED:
    return math.inf
  return math.sqrt(lateral_accel / curvature)
