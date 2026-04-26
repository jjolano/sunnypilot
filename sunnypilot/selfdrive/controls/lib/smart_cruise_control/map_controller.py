import math
import platform

from cereal import custom
from openpilot.common.constants import CV
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.car.cruise import V_CRUISE_UNSET
from openpilot.sunnypilot import PARAMS_UPDATE_PERIOD
from openpilot.sunnypilot.mapd.param_helpers import get_first_mapd_json, get_mapd_json, mapd_section_float
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

    if tlat is None or tlon is None or not valid_map_speed(tv):
      continue

    valid_velocities.append({"latitude": tlat, "longitude": tlon, "velocity": tv})

  return valid_velocities


def valid_map_speed(speed: float | None) -> bool:
  return speed is not None and 0. < speed < MAX_MAP_SPEED


def calculate_accel(t, target_jerk, a_ego):
  return a_ego + target_jerk * t


def calculate_velocity(t, target_jerk, a_ego, v_ego):
  return v_ego + a_ego * t + target_jerk/2 * (t ** 2)


def calculate_distance(t, target_jerk, a_ego, v_ego):
  return t * v_ego + a_ego/2 * (t ** 2) + target_jerk/6 * (t ** 3)


# points should be in radians
# output is meters
def distance_to_point(ax, ay, bx, by):
  a = math.sin((bx-ax)/2)*math.sin((bx-ax)/2) + math.cos(ax) * math.cos(bx)*math.sin((by-ay)/2)*math.sin((by-ay)/2)
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
    self.frame = -1

    self.last_position = coordinate_from_param("LastGPSPosition", self.mem_params) or Coordinate(0.0, 0.0)
    self.target_velocities = velocities_from_param("MapTargetVelocities", self.mem_params) or []

  def get_v_target_from_control(self) -> float:
    if self.is_active:
      return max(self.v_target, MIN_V)

    return V_CRUISE_UNSET

  def get_a_target_from_control(self) -> float:
    return self.a_ego

  def update_params(self):
    if self.frame % int(PARAMS_UPDATE_PERIOD / DT_MDL) == 0:
      self.enabled = self.params.get_bool("SmartCruiseControlMap")

  def update_calculations(self) -> None:
    self.last_position = coordinate_from_param("LastGPSPosition", self.mem_params) or Coordinate(0.0, 0.0)
    self.target_velocities = velocities_from_param("MapTargetVelocities", self.mem_params) or []

    if self.last_position is None or self.target_velocities is None:
      return

    forward_points, forward_distances = self._forward_target_velocity_distances()

    # find velocities that we are within the distance we need to adjust for
    valid_velocities = self._advisory_targets()
    for i in range(len(forward_points)):
      target_velocity = forward_points[i]
      tlat = target_velocity["latitude"]
      tlon = target_velocity["longitude"]
      tv = target_velocity["velocity"]
      if tv > self.v_ego:
        continue

      d = forward_distances[i]

      if self._target_in_range(tv, d):
        valid_velocities.append((float(tv), tlat, tlon))

    # Find the smallest velocity we need to adjust for
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

      # not found so let's reset
      self.v_target = 0.0
      self.target_lat = 0.0
      self.target_lon = 0.0

    self.v_target = min_v
    self.target_lat = target_lat
    self.target_lon = target_lon

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
      t_a = -1 * (discriminant ** 0.5 + b) / 2 * a
      t_b = (discriminant ** 0.5 - b) / 2 * a
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

  @staticmethod
  def _advisory_target(section) -> tuple[float, float, float] | None:
    if not isinstance(section, dict):
      return None

    target_v = mapd_section_float(section, "speedlimit", None)
    if not valid_map_speed(target_v):
      return None

    lat = mapd_section_float(section, "start_latitude", 0.)
    lon = mapd_section_float(section, "start_longitude", 0.)
    return float(target_v), float(lat or 0.), float(lon or 0.)

  def _distance_to_advisory_start(self, section) -> float | None:
    if not isinstance(section, dict):
      return None

    distance = mapd_section_float(section, "distance", None)
    if distance is not None:
      return max(0., distance)

    lat = mapd_section_float(section, "start_latitude", None)
    lon = mapd_section_float(section, "start_longitude", None)
    if lat is None or lon is None:
      return None

    return distance_to_point(self.last_position.latitude * TO_RADIANS, self.last_position.longitude * TO_RADIANS,
                             lat * TO_RADIANS, lon * TO_RADIANS)

  def _advisory_targets(self) -> list[tuple[float, float, float]]:
    targets = []

    current_advisory = get_first_mapd_json(self.mem_params, ADVISORY_LIMIT_KEYS)
    current_target = self._advisory_target(current_advisory)
    if current_target is not None and self._target_in_range(current_target[0], 0.):
      targets.append(current_target)

    next_advisory = get_first_mapd_json(self.mem_params, NEXT_ADVISORY_LIMIT_KEYS)
    next_target = self._advisory_target(next_advisory)
    next_distance = self._distance_to_advisory_start(next_advisory)
    if next_target is not None and next_distance is not None and self._target_in_range(next_target[0], next_distance):
      targets.append(next_target)

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

  def update(self, long_enabled: bool, long_override: bool, v_ego, a_ego, v_cruise) -> None:
    self.long_enabled = long_enabled
    self.long_override = long_override
    self.v_ego = v_ego
    self.a_ego = a_ego
    self.v_cruise = v_cruise

    self.update_params()
    self.update_calculations()

    self.is_enabled, self.is_active = self._update_state_machine()

    self.output_v_target = self.get_v_target_from_control()
    self.output_a_target = self.get_a_target_from_control()

    self.frame += 1
