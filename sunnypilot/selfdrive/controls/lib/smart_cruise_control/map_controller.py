import json
import math
import platform
import time

import numpy as np

from cereal import custom
from openpilot.common.constants import CV
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.car.cruise import V_CRUISE_UNSET
from openpilot.selfdrive.controls.lib.vehicle_math import speed_for_lateral_accel
from openpilot.sunnypilot import PARAMS_UPDATE_PERIOD
from openpilot.sunnypilot.mapd.param_helpers import (
  MAP_ADVISORY_UPDATED_AT_PARAM,
  MAP_TARGET_VELOCITIES_UPDATED_AT_PARAM,
  get_first_mapd_json,
  get_mapd_json,
  mapd_section_float,
)
from openpilot.sunnypilot.navd.helpers import coordinate_from_param, Coordinate
from openpilot.sunnypilot.selfdrive.controls.lib.smart_cruise_control import MIN_V

MapState = VisionState = custom.LongitudinalPlanSP.SmartCruiseControl.MapState

ACTIVE_STATES = (MapState.turning, )
ENABLED_STATES = (MapState.enabled, MapState.overriding, *ACTIVE_STATES)

R = 6373000.0  # approximate radius of earth in meters
TO_RADIANS = math.pi / 180
TO_DEGREES = 180 / math.pi
TARGET_JERK = -0.6  # m/s^3 There's some jounce limits that are not consistent so we're fudging this some
TARGET_ACCEL = -1.2  # m/s^2 should match up with the long planner limit
TARGET_OFFSET = 1.0  # seconds - This controls how soon before the curve you reach the target velocity. It also helps
                      # reach the target velocity when inaccuracies in the distance modeling logic would cause overshoot.
                      # The value is multiplied against the target velocity to determine the additional distance. This is
                      # done to keep the distance calculations consistent but results in the offset actually being less
                      # time than specified depending on how much of a speed differential there is between v_ego and the
                      # target velocity.
MAX_MAP_SPEED = V_CRUISE_UNSET * CV.KPH_TO_MS
ADVISORY_LIMIT_KEYS = ("MapAdvisorySpeedLimit", "MapAdvisoryLimit")
NEXT_ADVISORY_LIMIT_KEYS = ("NextMapAdvisorySpeedLimit", "NextMapAdvisoryLimit")
MODEL_CURVE_DISTANCE_WINDOW = 20.0  # m, match map target points to nearby model path samples.
MODEL_CURVE_MIN_LAT_ACCEL = 1.3  # m/s^2, ignore weak/noisy curvature predictions.
MODEL_CURVE_TARGET_LAT_ACCEL = 2.0  # m/s^2, same comfort target used by SCC vision.
MODEL_CURVE_MIN_SPEED = 1.0  # m/s, avoid unstable curvature estimates at near-zero speed.
MODEL_CURVE_OVERSLOWDOWN_DELTA = 5.0  # m/s, require model confirmation for large map slowdowns.
MODEL_CURVE_OVERSLOWDOWN_RATIO = 0.8  # fraction of v_ego, require model confirmation for relative map slowdowns.
MODEL_CURVE_OVERSLOWDOWN_MARGIN = 2.0  # m/s, allow small map/model target mismatch.
PARAM_CACHE_MISS = object()
VALID_TRUE_VALUES = ("1", "true", "True", b"1", b"true", b"True")
MAP_DATA_HEARTBEAT_TTL = 5.0


def valid_map_coordinate(latitude: float | None, longitude: float | None) -> bool:
  return latitude is not None and longitude is not None and math.isfinite(latitude) and math.isfinite(longitude)


def velocities_from_param(param: str, params: Params):
  if params is None:
    params = Params()

  velocities = get_mapd_json(params, param)
  if not isinstance(velocities, list):
    return []

  valid_velocities = []
  for target_velocity in velocities:
    if not isinstance(target_velocity, dict):
      continue

    tlat = mapd_section_float(target_velocity, "latitude", None)
    tlon = mapd_section_float(target_velocity, "longitude", None)
    tv = mapd_section_float(target_velocity, "velocity", None)

    if not valid_map_coordinate(tlat, tlon) or not valid_map_speed(tv):
      continue

    valid_velocities.append({"latitude": tlat, "longitude": tlon, "velocity": tv})

  return valid_velocities


def sunnypilot_current_velocities_from_param(param: str, params: Params):
  if params is None:
    params = Params()

  json_str = params.get(param)
  if json_str is None:
    return None

  return json.loads(json_str)


def valid_map_speed(speed: float | None) -> bool:
  return speed is not None and math.isfinite(speed) and 0. < speed < MAX_MAP_SPEED


def calculate_accel(t, target_jerk, a_ego):
  return a_ego + target_jerk * t


def calculate_velocity(t, target_jerk, a_ego, v_ego):
  return v_ego + a_ego * t + target_jerk/2 * (t ** 2)


def calculate_distance(t, target_jerk, a_ego, v_ego):
  return t * v_ego + a_ego/2 * (t ** 2) + target_jerk/6 * (t ** 3)


# points should be in radians
# output is meters
def distance_to_point(ax, ay, bx, by):
  if not all(math.isfinite(value) for value in (ax, ay, bx, by)):
    return float("inf")

  a = math.sin((bx-ax)/2)*math.sin((bx-ax)/2) + math.cos(ax) * math.cos(bx)*math.sin((by-ay)/2)*math.sin((by-ay)/2)
  a = min(1.0, max(0.0, a))
  c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

  return R * c  # in meters


def point_distance(a: Coordinate, b: Coordinate) -> float:
  return distance_to_point(a.latitude * TO_RADIANS, a.longitude * TO_RADIANS,
                           b.latitude * TO_RADIANS, b.longitude * TO_RADIANS)


def target_velocity_coordinate(target_velocity: dict) -> Coordinate:
  return Coordinate(target_velocity["latitude"], target_velocity["longitude"])


class SmartCruiseControlMap:
  v_target: float = 0
  a_target: float = 0.
  v_ego: float = 0.
  a_ego: float = 0.
  output_v_target: float = V_CRUISE_UNSET
  output_a_target: float = 0.

  def __init__(self):
    self.params = Params()
    self.mem_params = Params("/dev/shm/params") if platform.system() != "Darwin" else self.params
    self.enabled = self.params.get_bool("SmartCruiseControlMap")
    self.long_enabled = False
    self.long_override = False
    self.is_enabled = False
    self.is_active = False
    self.state = MapState.disabled
    self.v_cruise = 0
    self.target_lat = 0.0
    self.target_lon = 0.0
    self.target_prediction_advanced = False
    self.frame = -1
    self._last_position_raw = PARAM_CACHE_MISS
    self._target_velocities_raw = PARAM_CACHE_MISS
    self._advisory_raw_cache = {}
    self._advisory_value_cache = {}

    self.last_position = Coordinate(0.0, 0.0)
    self.target_velocities = []
    self._update_cached_map_params()

  def get_v_target_from_control(self) -> float:
    if self.is_active:
      return max(self.v_target, MIN_V)

    return V_CRUISE_UNSET

  def get_a_target_from_control(self) -> float:
    return 0.0

  def update_params(self):
    if self.frame % int(PARAMS_UPDATE_PERIOD / DT_MDL) == 0:
      self.enabled = self.params.get_bool("SmartCruiseControlMap")

  def _update_cached_position(self) -> None:
    last_position_raw = self.mem_params.get("LastGPSPosition")
    if last_position_raw != self._last_position_raw:
      self._last_position_raw = last_position_raw
      self.last_position = coordinate_from_param("LastGPSPosition", self.mem_params) or Coordinate(0.0, 0.0)

  def _update_cached_target_velocities(self) -> None:
    target_velocities_raw = self.mem_params.get("MapTargetVelocities")
    if target_velocities_raw != self._target_velocities_raw:
      self._target_velocities_raw = target_velocities_raw
      self.target_velocities = velocities_from_param("MapTargetVelocities", self.mem_params) or []

  def _clear_cached_target_velocities(self) -> None:
    self._target_velocities_raw = PARAM_CACHE_MISS
    self.target_velocities = []

  def _update_cached_map_params(self) -> None:
    self._update_cached_position()
    if self._target_velocity_params_valid():
      self._update_cached_target_velocities()
    else:
      self._clear_cached_target_velocities()

  def _valid_param_flag(self, key: str) -> bool:
    value = self.mem_params.get(key)
    return value is True or value in (None, *VALID_TRUE_VALUES)

  def _heartbeat_fresh(self, key: str) -> bool:
    heartbeat_raw = self.mem_params.get(key)
    if heartbeat_raw in (None, ""):
      return True

    try:
      heartbeat = float(heartbeat_raw)
    except (TypeError, ValueError):
      return False

    now = time.monotonic()
    if not math.isfinite(heartbeat) or heartbeat < 0.0 or heartbeat > now:
      return False

    return now - heartbeat <= MAP_DATA_HEARTBEAT_TTL

  def _position_params_valid(self) -> bool:
    return self._valid_param_flag("LastGPSPositionValid")

  def _target_velocity_params_valid(self) -> bool:
    return self._valid_param_flag("MapTargetVelocitiesValid") and self._heartbeat_fresh(MAP_TARGET_VELOCITIES_UPDATED_AT_PARAM)

  def _advisory_params_valid(self) -> bool:
    return self._heartbeat_fresh(MAP_ADVISORY_UPDATED_AT_PARAM)

  def _clear_map_target(self) -> None:
    self.v_target = 0.0
    self.target_lat = 0.0
    self.target_lon = 0.0
    self.target_prediction_advanced = False

  def _cached_first_mapd_json(self, keys: tuple[str, ...]):
    raw_values = tuple(self.mem_params.get(key) for key in keys)
    if raw_values != self._advisory_raw_cache.get(keys, PARAM_CACHE_MISS):
      self._advisory_raw_cache[keys] = raw_values
      self._advisory_value_cache[keys] = get_first_mapd_json(self.mem_params, keys)
    return self._advisory_value_cache.get(keys)

  def update_calculations(self, model_msg=None) -> None:
    self._update_cached_map_params()

    if not self._position_params_valid():
      self._clear_map_target()
      return

    if self.last_position is None or self.target_velocities is None:
      return

    forward_points, forward_distances = self._forward_target_velocity_distances()

    # find velocities that we are within the distance we need to adjust for
    valid_velocities = self._advisory_targets(model_msg)
    for i in range(len(forward_points)):
      target_velocity = forward_points[i]
      tlat = target_velocity["latitude"]
      tlon = target_velocity["longitude"]
      tv = float(target_velocity["velocity"])
      if tv > self.v_ego:
        continue

      d = forward_distances[i]

      in_range, prediction_advanced, control_tv = self._target_range_state(tv, d, model_msg)
      if in_range:
        valid_velocities.append((float(control_tv), tlat, tlon, prediction_advanced))

    # Find the smallest velocity we need to adjust for
    min_v = 100.0
    target_lat = 0.0
    target_lon = 0.0
    target_prediction_advanced = False
    for tv, lat, lon, prediction_advanced in valid_velocities:
      if tv < min_v:
        min_v = tv
        target_lat = lat
        target_lon = lon
        target_prediction_advanced = prediction_advanced

    if self.v_target < min_v and not (self.target_lat == 0 and self.target_lon == 0):
      if not self.target_prediction_advanced:
        for i in range(len(forward_points)):
          target_velocity = forward_points[i]
          tlat = target_velocity["latitude"]
          tlon = target_velocity["longitude"]
          tv = float(target_velocity["velocity"])
          if tv > self.v_ego:
            continue

          d = forward_distances[i]
          if (tlat == self.target_lat and tlon == self.target_lon and tv == self.v_target and
              self._model_confirms_large_slowdown(self.v_ego, tv, d, model_msg)):
            return

      # not found so let's reset
      self._clear_map_target()

    self.v_target = min_v
    self.target_lat = target_lat
    self.target_lon = target_lon
    self.target_prediction_advanced = target_prediction_advanced

  def _forward_target_velocity_distances(self) -> tuple[list[dict], list[float]]:
    if not self.target_velocities:
      return [], []

    min_idx = 0
    min_dist = float("inf")
    for i, target_velocity in enumerate(self.target_velocities):
      d = point_distance(self.last_position, target_velocity_coordinate(target_velocity))
      if d < min_dist:
        min_dist = d
        min_idx = i

    forward_points = self.target_velocities[min_idx:]
    forward_distances = []
    last_position = self.last_position
    distance = 0.0
    for target_velocity in forward_points:
      current_position = target_velocity_coordinate(target_velocity)
      distance += point_distance(last_position, current_position)
      forward_distances.append(distance)
      last_position = current_position

    return forward_points, forward_distances

  def _target_control_distance(self, target_v: float) -> float | None:
    if target_v > self.v_ego:
      return None

    a_diff = (self.a_ego - TARGET_ACCEL)
    accel_t = abs(a_diff / TARGET_JERK)
    min_accel_v = calculate_velocity(accel_t, TARGET_JERK, self.a_ego, self.v_ego)

    max_d = 0.
    if target_v > min_accel_v:
      # calculate time needed based on target jerk
      a = 0.5 * TARGET_JERK
      b = self.a_ego
      c = self.v_ego - target_v
      discriminant = b**2 - 4 * a * c
      if discriminant < 0:
        return None
      t_a = -1 * (discriminant ** 0.5 + b) / (2 * a)
      t_b = (discriminant ** 0.5 - b) / (2 * a)
      t = t_a if t_a > 0 else t_b
      if t <= 0:
        return None

      max_d = max_d + calculate_distance(t, TARGET_JERK, self.a_ego, self.v_ego)
    else:
      t = accel_t
      max_d = calculate_distance(t, TARGET_JERK, self.a_ego, self.v_ego)

      # calculate additional time needed based on target accel
      t = abs((min_accel_v - target_v) / TARGET_ACCEL)
      max_d += calculate_distance(t, 0, TARGET_ACCEL, min_accel_v)

    return max_d + target_v * TARGET_OFFSET

  def _target_in_range(self, target_v: float, distance: float) -> bool:
    control_distance = self._target_control_distance(target_v)
    return control_distance is not None and distance < control_distance

  def _target_range_state(self, target_v: float, distance: float, model_msg) -> tuple[bool, bool, float]:
    if not self._model_confirms_large_slowdown(self.v_ego, target_v, distance, model_msg):
      soft_target = self._soft_model_confirmed_target(self.v_ego, target_v, distance, model_msg)
      if soft_target is not None and self._target_in_range(soft_target, distance):
        return True, True, soft_target
      return False, False, target_v

    if self._target_in_range(target_v, distance):
      return True, False, target_v

    control_target_v = self._prediction_control_target(target_v, distance, model_msg)
    prediction_advanced = control_target_v < target_v and self._target_in_range(control_target_v, distance)
    return prediction_advanced, prediction_advanced, target_v

  @staticmethod
  def _model_covers_distance(model_msg, distance: float) -> bool:
    if model_msg is None:
      return False

    positions = np.asarray(getattr(getattr(model_msg, "position", None), "x", []), dtype=float)
    return positions.ndim == 1 and positions.size > 0 and np.all(np.isfinite(positions)) and distance <= positions[-1] + MODEL_CURVE_DISTANCE_WINDOW

  @classmethod
  def _model_confirms_large_slowdown(cls, v_ego: float, target_v: float, distance: float, model_msg) -> bool:
    slowdown_requires_confirmation = (
      v_ego - target_v > MODEL_CURVE_OVERSLOWDOWN_DELTA or
      target_v < v_ego * MODEL_CURVE_OVERSLOWDOWN_RATIO
    )
    if model_msg is None or not slowdown_requires_confirmation:
      return True

    if not cls._model_covers_distance(model_msg, distance):
      return True

    prediction_target = cls._prediction_curve_target(model_msg, distance)
    return prediction_target is not None and prediction_target <= target_v + MODEL_CURVE_OVERSLOWDOWN_MARGIN

  @classmethod
  def _soft_model_confirmed_target(cls, v_ego: float, target_v: float, distance: float, model_msg) -> float | None:
    if model_msg is None or not cls._model_covers_distance(model_msg, distance):
      return None

    prediction_target = cls._prediction_curve_target(model_msg, distance)
    if prediction_target is None:
      return None

    soft_target = max(target_v, min(v_ego, prediction_target))
    return soft_target if soft_target < v_ego else None

  @staticmethod
  def _prediction_curve_target(model_msg, distance: float) -> float | None:
    if model_msg is None:
      return None

    positions = np.asarray(getattr(getattr(model_msg, "position", None), "x", []), dtype=float)
    velocities = np.asarray(getattr(getattr(model_msg, "velocity", None), "x", []), dtype=float)
    yaw_rates = np.abs(np.asarray(getattr(getattr(model_msg, "orientationRate", None), "z", []), dtype=float))
    if positions.ndim != 1 or positions.size == 0 or positions.size != velocities.size or positions.size != yaw_rates.size:
      return None
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(velocities)) or not np.all(np.isfinite(yaw_rates)):
      return None
    if distance > positions[-1] + MODEL_CURVE_DISTANCE_WINDOW:
      return None

    if distance <= MODEL_CURVE_DISTANCE_WINDOW:
      sample_mask = positions <= distance + MODEL_CURVE_DISTANCE_WINDOW
    else:
      sample_mask = np.abs(positions - distance) <= MODEL_CURVE_DISTANCE_WINDOW
    sample_mask &= velocities >= MODEL_CURVE_MIN_SPEED
    if not np.any(sample_mask):
      return None

    sample_velocities = velocities[sample_mask]
    sample_yaw_rates = yaw_rates[sample_mask]
    sample_lat_accels = sample_yaw_rates * sample_velocities
    if float(np.max(sample_lat_accels)) < MODEL_CURVE_MIN_LAT_ACCEL:
      return None

    curvatures = sample_yaw_rates / sample_velocities
    valid_curvatures = curvatures[curvatures > 1e-6]
    if valid_curvatures.size == 0:
      return None

    target = speed_for_lateral_accel(MODEL_CURVE_TARGET_LAT_ACCEL, float(np.max(valid_curvatures)))
    return float(target) if math.isfinite(target) else None

  @classmethod
  def _prediction_control_target(cls, target_v: float, distance: float, model_msg) -> float:
    prediction_target = cls._prediction_curve_target(model_msg, distance)
    if prediction_target is None:
      return target_v
    return max(MIN_V, min(target_v, prediction_target))

  @staticmethod
  def _advisory_target(section) -> tuple[float, float, float] | None:
    if not isinstance(section, dict):
      return None

    target_v = mapd_section_float(section, "speedlimit", None)
    if not valid_map_speed(target_v):
      return None

    lat = mapd_section_float(section, "start_latitude", 0.)
    lon = mapd_section_float(section, "start_longitude", 0.)
    if not valid_map_coordinate(lat, lon):
      return None
    return float(target_v), float(lat or 0.), float(lon or 0.)

  def _distance_to_advisory_start(self, section) -> float | None:
    if not isinstance(section, dict):
      return None

    distance = mapd_section_float(section, "distance", None)
    if distance is not None:
      return max(0., distance)

    lat = mapd_section_float(section, "start_latitude", None)
    lon = mapd_section_float(section, "start_longitude", None)
    if not valid_map_coordinate(lat, lon):
      return None

    return distance_to_point(self.last_position.latitude * TO_RADIANS, self.last_position.longitude * TO_RADIANS,
                             lat * TO_RADIANS, lon * TO_RADIANS)

  def _advisory_targets(self, model_msg=None) -> list[tuple[float, float, float, bool]]:
    targets = []
    if not self._advisory_params_valid():
      return targets

    current_advisory = self._cached_first_mapd_json(ADVISORY_LIMIT_KEYS)
    current_target = self._advisory_target(current_advisory)
    if current_target is not None:
      in_range, prediction_advanced, control_target_v = self._target_range_state(current_target[0], 0., model_msg)
      if in_range:
        targets.append((control_target_v, current_target[1], current_target[2], prediction_advanced))

    next_advisory = self._cached_first_mapd_json(NEXT_ADVISORY_LIMIT_KEYS)
    next_target = self._advisory_target(next_advisory)
    next_distance = self._distance_to_advisory_start(next_advisory)
    if next_target is not None and next_distance is not None:
      in_range, prediction_advanced, control_target_v = self._target_range_state(next_target[0], next_distance, model_msg)
      if in_range:
        targets.append((control_target_v, next_target[1], next_target[2], prediction_advanced))

    return targets

  def _update_state_machine(self) -> tuple[bool, bool]:
    # ENABLED, TURNING
    if self.state != MapState.disabled:
      if not self.long_enabled or not self.enabled:
        self.state = MapState.disabled
      elif self.long_override:
        self.state = MapState.overriding

      else:
        # ENABLED
        if self.state == MapState.enabled:
          if self.v_cruise > self.v_target != 0:
            self.state = MapState.turning

        # TURNING
        elif self.state == MapState.turning:
          if self.v_cruise <= self.v_target or self.v_target == 0:
            self.state = MapState.enabled

        # OVERRIDING
        elif self.state == MapState.overriding:
          if not self.long_override:
            if self.v_cruise > self.v_target != 0:
              self.state = MapState.turning
            else:
              self.state = MapState.enabled

    # DISABLED
    elif self.state == MapState.disabled:
      if self.long_enabled and self.enabled:
        if self.long_override:
          self.state = MapState.overriding
        else:
          self.state = MapState.enabled

    enabled = self.state in ENABLED_STATES
    active = self.state in ACTIVE_STATES

    return enabled, active

  def update(self, long_enabled: bool, long_override: bool, v_ego, a_ego, v_cruise, model_msg=None) -> None:
    self.long_enabled = long_enabled
    self.long_override = long_override
    self.v_ego = v_ego
    self.a_ego = a_ego
    self.v_cruise = v_cruise

    self.update_params()
    self.update_calculations(model_msg)

    self.is_enabled, self.is_active = self._update_state_machine()

    self.output_v_target = self.get_v_target_from_control()
    self.output_a_target = self.get_a_target_from_control()

    self.frame += 1


class SunnypilotCurrentSmartCruiseControlMap(SmartCruiseControlMap):
  def __init__(self):
    super().__init__()
    self.last_position = coordinate_from_param("LastGPSPosition", self.mem_params) or Coordinate(0.0, 0.0)
    self.target_velocities = sunnypilot_current_velocities_from_param("MapTargetVelocities", self.mem_params) or []

  def get_a_target_from_control(self) -> float:
    return self.a_ego

  def update_calculations(self, model_msg=None) -> None:
    self._update_cached_position()
    self.last_position = self.last_position or Coordinate(0.0, 0.0)
    lat = self.last_position.latitude
    lon = self.last_position.longitude

    if not self._position_params_valid() or not self._target_velocity_params_valid():
      self._clear_map_target()
      return

    self.target_velocities = sunnypilot_current_velocities_from_param("MapTargetVelocities", self.mem_params) or []

    if self.last_position is None or self.target_velocities is None:
      return

    min_dist = 1000
    min_idx = 0
    distances = []

    for i in range(len(self.target_velocities)):
      target_velocity = self.target_velocities[i]
      tlat = target_velocity["latitude"]
      tlon = target_velocity["longitude"]
      d = distance_to_point(lat * TO_RADIANS, lon * TO_RADIANS, tlat * TO_RADIANS, tlon * TO_RADIANS)
      distances.append(d)
      if d < min_dist:
        min_dist = d
        min_idx = i

    forward_points = self.target_velocities[min_idx:]
    forward_distances = distances[min_idx:]

    valid_velocities = []
    for i in range(len(forward_points)):
      target_velocity = forward_points[i]
      tlat = target_velocity["latitude"]
      tlon = target_velocity["longitude"]
      tv = target_velocity["velocity"]
      if tv > self.v_ego:
        continue

      d = forward_distances[i]

      a_diff = (self.a_ego - TARGET_ACCEL)
      accel_t = abs(a_diff / TARGET_JERK)
      min_accel_v = calculate_velocity(accel_t, TARGET_JERK, self.a_ego, self.v_ego)

      max_d = 0
      if tv > min_accel_v:
        a = 0.5 * TARGET_JERK
        b = self.a_ego
        c = self.v_ego - tv
        t_a = -1 * ((b**2 - 4 * a * c) ** 0.5 + b) / 2 * a
        t_b = ((b**2 - 4 * a * c) ** 0.5 - b) / 2 * a
        if not isinstance(t_a, complex) and t_a > 0:
          t = t_a
        else:
          t = t_b
        if isinstance(t, complex):
          continue

        max_d = max_d + calculate_distance(t, TARGET_JERK, self.a_ego, self.v_ego)
      else:
        t = accel_t
        max_d = calculate_distance(t, TARGET_JERK, self.a_ego, self.v_ego)

        t = abs((min_accel_v - tv) / TARGET_ACCEL)
        max_d += calculate_distance(t, 0, TARGET_ACCEL, min_accel_v)

      if d < max_d + tv * TARGET_OFFSET:
        valid_velocities.append((float(tv), tlat, tlon))

    min_v = 100.0
    target_lat = 0.0
    target_lon = 0.0
    for tv, lat, lon in valid_velocities:
      if tv < min_v:
        min_v = tv
        target_lat = lat
        target_lon = lon

    if self.v_target < min_v and not (self.target_lat == 0 and self.target_lon == 0):
      for i in range(len(forward_points)):
        target_velocity = forward_points[i]
        tlat = target_velocity["latitude"]
        tlon = target_velocity["longitude"]
        tv = target_velocity["velocity"]
        if tv > self.v_ego:
          continue

        if tlat == self.target_lat and tlon == self.target_lon and tv == self.v_target:
          return

      self.v_target = 0.0
      self.target_lat = 0.0
      self.target_lon = 0.0

    self.v_target = min_v
    self.target_lat = target_lat
    self.target_lon = target_lon

  def update(self, long_enabled: bool, long_override: bool, v_ego, a_ego, v_cruise, model_msg=None) -> None:
    self.long_enabled = long_enabled
    self.long_override = long_override
    self.v_ego = v_ego
    self.a_ego = a_ego
    self.v_cruise = v_cruise

    self.update_params()
    self.update_calculations(model_msg)

    self.is_enabled, self.is_active = self._update_state_machine()

    self.output_v_target = self.get_v_target_from_control()
    self.output_a_target = self.get_a_target_from_control()

    self.frame += 1
