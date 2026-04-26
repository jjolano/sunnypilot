from openpilot.selfdrive.car.cruise import V_CRUISE_UNSET

SUPPORTED_TRAFFIC_CONTROLS = {
  "stop",
  "stop_sign",
  "traffic_light",
  "traffic_lights",
  "traffic_signal",
  "traffic_signals",
}

TRAFFIC_CONTROL_CAUTION_SPEED = 8.33  # m/s, ~30 kph. Never command a map-only stop.
TRAFFIC_CONTROL_MAX_EGO_SPEED = 22.35  # m/s, ~50 mph. Avoid high-speed OSM false positives.
TRAFFIC_CONTROL_MAX_DISTANCE = 80.0  # m
TRAFFIC_CONTROL_MIN_DISTANCE = 1.0  # m
TRAFFIC_CONTROL_MODEL_STOP_SPEED = 1.0  # m/s
TRAFFIC_CONTROL_MODEL_SLOW_SPEED = 2.0  # m/s
TRAFFIC_CONTROL_MAX_ACCEL = -0.15  # m/s^2
TRAFFIC_CONTROL_MIN_ACCEL = -0.8  # m/s^2


def normalize_traffic_control(control_type: str) -> str:
  return str(control_type or "").strip().lower().replace("-", "_").replace(" ", "_")


def _model_action_should_stop(model_data) -> bool:
  try:
    return bool(model_data.action.shouldStop)
  except AttributeError:
    return False


def model_stop_distance(model_data) -> float | None:
  positions = list(getattr(model_data.position, "x", []))
  velocities = list(getattr(model_data.velocity, "x", []))
  if not positions or not velocities:
    return None

  for x, v in zip(positions, velocities, strict=False):
    if x > 0. and v <= TRAFFIC_CONTROL_MODEL_STOP_SPEED:
      return float(x)

  if _model_action_should_stop(model_data) and positions[-1] > 0. and velocities[-1] <= TRAFFIC_CONTROL_MODEL_SLOW_SPEED:
    return float(positions[-1])

  return None


def model_stop_matches_map_distance(model_distance: float, map_distance: float) -> bool:
  if map_distance <= TRAFFIC_CONTROL_MIN_DISTANCE:
    return model_distance <= TRAFFIC_CONTROL_MAX_DISTANCE


  allowed_error = max(12.0, map_distance * 0.35)
  return abs(model_distance - map_distance) <= allowed_error


class OsmTrafficControlPrior:
  output_v_target: float = V_CRUISE_UNSET
  output_a_target: float = 0.0

  def __init__(self):
    self.active = False
    self.control_type = ""
    self.distance = 0.0

  @staticmethod
  def _traffic_control_candidate(map_data) -> tuple[str, float]:
    candidates = (
      (getattr(map_data, "trafficControlAheadValid", False), getattr(map_data, "trafficControlAhead", ""),
       getattr(map_data, "trafficControlAheadDistance", 0.0)),
      (getattr(map_data, "trafficControlValid", False), getattr(map_data, "trafficControl", ""),
       getattr(map_data, "trafficControlDistance", 0.0)),
    )

    for valid, control_type, distance in candidates:
      normalized = normalize_traffic_control(control_type)
      if valid and normalized in SUPPORTED_TRAFFIC_CONTROLS:
        return normalized, max(0.0, float(distance))

    return "", 0.0

  def _reset(self, a_ego: float) -> None:
    self.active = False
    self.control_type = ""
    self.distance = 0.0
    self.output_v_target = V_CRUISE_UNSET
    self.output_a_target = a_ego

  def update(self, sm, long_enabled: bool, long_override: bool, v_ego: float, a_ego: float) -> None:
    self._reset(a_ego)

    if not long_enabled or long_override or v_ego <= TRAFFIC_CONTROL_CAUTION_SPEED or v_ego > TRAFFIC_CONTROL_MAX_EGO_SPEED:
      return

    control_type, map_distance = self._traffic_control_candidate(sm["liveMapDataSP"])
    if not control_type or map_distance > TRAFFIC_CONTROL_MAX_DISTANCE:
      return

    stop_distance = model_stop_distance(sm["modelV2"])
    if stop_distance is None or not model_stop_matches_map_distance(stop_distance, map_distance):
      return

    target_v = TRAFFIC_CONTROL_CAUTION_SPEED
    target_a = (target_v ** 2 - v_ego ** 2) / (2.0 * max(map_distance, TRAFFIC_CONTROL_MIN_DISTANCE))

    self.active = True
    self.control_type = control_type
    self.distance = map_distance
    self.output_v_target = target_v
    self.output_a_target = max(TRAFFIC_CONTROL_MIN_ACCEL, min(TRAFFIC_CONTROL_MAX_ACCEL, target_a))
