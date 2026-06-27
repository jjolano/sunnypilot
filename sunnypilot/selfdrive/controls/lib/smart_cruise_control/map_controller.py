import json
import math
import platform

import numpy as np

from cereal import custom
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.car.cruise import V_CRUISE_UNSET
from openpilot.sunnypilot import PARAMS_UPDATE_PERIOD
from openpilot.sunnypilot.navd.helpers import coordinate_from_param, Coordinate

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
MAX_ROUTE_TARGET_DISTANCE = 20.0
MAX_SHORT_DROP_DISTANCE = 30.0
MAX_SHORT_DROP_DELTA_V = 6.0
SHORT_DROP_CONFIRM_POINTS = 2
MATERIAL_DROP_DELTA_V = 2.5
MIN_CURRENT_LAT_ACCEL_CORROBORATION = 0.5
MIN_MODEL_PRED_LAT_ACCEL_CORROBORATION = 0.9
_ROUTE_POINT_EPS = 1e-6


def velocities_from_param(param: str, params: Params):
  if params is None:
    params = Params()

  json_str = params.get(param)
  if json_str is None:
    return None

  try:
    velocities = json.loads(json_str)
  except (json.JSONDecodeError, TypeError):
    return None

  return velocities


def _is_finite_number(value) -> bool:
  return isinstance(value, (int, float)) and math.isfinite(value)


def _valid_velocity_entry(entry) -> bool:
  return (isinstance(entry, dict) and all(_is_finite_number(entry.get(k)) for k in ("latitude", "longitude", "velocity"))
          and entry["velocity"] > 0.0)


def _dedupe_route_points(route_points: list[dict]) -> list[dict]:
  deduped = []
  for point in route_points:
    if deduped:
      prev = deduped[-1]
      if abs(point["latitude"] - prev["latitude"]) <= _ROUTE_POINT_EPS and abs(point["longitude"] - prev["longitude"]) <= _ROUTE_POINT_EPS:
        if point["velocity"] > prev["velocity"]:
          deduped[-1] = point
        continue
    deduped.append(point)
  return deduped


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

    try:
      self.last_position = coordinate_from_param("LastGPSPosition", self.mem_params) or Coordinate(0.0, 0.0)
    except (json.JSONDecodeError, TypeError, ValueError):
      self.last_position = Coordinate(0.0, 0.0)
    self.target_velocities = velocities_from_param("MapTargetVelocities", self.mem_params) or []

  def get_v_target_from_control(self) -> float:
    # SCC-M is evidence-only in this fork: keep state/v_target for telemetry and curve
    # confidence, but never let it win planner arbitration until a bounded apply tier exists.
    return V_CRUISE_UNSET

  def get_a_target_from_control(self) -> float:
    # Return a neutral acceleration request; map evidence must not directly slow the car.
    return 0.0

  def _clear_targets(self) -> None:
    self.v_target = 0.0
    self.target_lat = 0.0
    self.target_lon = 0.0

  def _material_drop_needs_corroboration(self, tv: float) -> bool:
    return (self.v_ego - tv) >= MATERIAL_DROP_DELTA_V

  def _model_corroborates(self, sm) -> bool:
    if sm is None:
      return False

    try:
      rate_plan = np.asarray(np.abs(sm['modelV2'].orientationRate.z), dtype=np.float64)
      vel_plan = np.asarray(sm['modelV2'].velocity.x, dtype=np.float64)
    except Exception:
      return False

    if rate_plan.size == 0 or vel_plan.size == 0 or rate_plan.shape != vel_plan.shape:
      return False

    pred_lat_accels = rate_plan * vel_plan
    if not np.all(np.isfinite(pred_lat_accels)):
      return False

    return float(np.percentile(pred_lat_accels, 97)) >= MIN_MODEL_PRED_LAT_ACCEL_CORROBORATION

  def _current_lateral_accel_corroborates(self, sm) -> bool:
    if sm is None:
      return False

    try:
      curvature = float(sm['controlsState'].curvature)
    except Exception:
      return False

    if not math.isfinite(curvature) or not math.isfinite(self.v_ego):
      return False

    return abs(self.v_ego ** 2 * curvature) >= MIN_CURRENT_LAT_ACCEL_CORROBORATION

  def update_params(self):
    if self.frame % int(PARAMS_UPDATE_PERIOD / DT_MDL) == 0:
      self.enabled = self.params.get_bool("SmartCruiseControlMap")

  def update_calculations(self, sm=None) -> None:
    try:
      self.last_position = coordinate_from_param("LastGPSPosition", self.mem_params)
    except (json.JSONDecodeError, TypeError, ValueError):
      self.last_position = None
    if self.last_position is None:
      self.v_target = 0.0
      self.target_lat = 0.0
      self.target_lon = 0.0
      return
    lat = self.last_position.latitude
    lon = self.last_position.longitude

    self.target_velocities = velocities_from_param("MapTargetVelocities", self.mem_params) or []

    if not _is_finite_number(lat) or not _is_finite_number(lon):
      self.v_target = 0.0
      self.target_lat = 0.0
      self.target_lon = 0.0
      return

    if not isinstance(self.target_velocities, list) or not self.target_velocities:
      self.v_target = 0.0
      self.target_lat = 0.0
      self.target_lon = 0.0
      return

    route_points = _dedupe_route_points([tv for tv in self.target_velocities if _valid_velocity_entry(tv)])
    if not route_points:
      self.v_target = 0.0
      self.target_lat = 0.0
      self.target_lon = 0.0
      return

    nearest_idx = None
    nearest_dist = float("inf")
    distances = []

    # find our location in the path
    for i, target_velocity in enumerate(route_points):
      tlat = target_velocity["latitude"]
      tlon = target_velocity["longitude"]
      d = distance_to_point(lat * TO_RADIANS, lon * TO_RADIANS, tlat * TO_RADIANS, tlon * TO_RADIANS)
      distances.append(d)
      if d < nearest_dist:
        nearest_dist = d
        nearest_idx = i

    if nearest_idx is None or nearest_dist > MAX_ROUTE_TARGET_DISTANCE:
      self.v_target = 0.0
      self.target_lat = 0.0
      self.target_lon = 0.0
      return

    # only look at values from our current position forward
    forward_points = route_points[nearest_idx:]
    forward_distances = distances[nearest_idx:]

    # Require route progression/bracketing evidence around the nearest point.
    # Boundary slices are ambiguous and can be entirely stale behind/ahead segments.
    if nearest_idx == 0 or nearest_idx == len(route_points) - 1:
      self.v_target = 0.0
      self.target_lat = 0.0
      self.target_lon = 0.0
      return

    if not (distances[nearest_idx - 1] >= nearest_dist and distances[nearest_idx + 1] >= nearest_dist):
      self.v_target = 0.0
      self.target_lat = 0.0
      self.target_lon = 0.0
      return

    if any(forward_distances[i] + _ROUTE_POINT_EPS < forward_distances[i - 1] for i in range(1, len(forward_distances))):
      self.v_target = 0.0
      self.target_lat = 0.0
      self.target_lon = 0.0
      return

    # find velocities that we are within the distance we need to adjust for
    valid_velocities = []
    for i in range(len(forward_points)):
      target_velocity = forward_points[i]
      tv = target_velocity["velocity"]
      if tv > self.v_ego:
        continue

      d = forward_distances[i]
      # Reject abrupt short-range drops from noisy or stale map data. A >6 m/s drop within
      # ~30 m is already a strong brake request; without model/route corroboration, fail closed.
      if d < MAX_SHORT_DROP_DISTANCE and (self.v_ego - tv) > MAX_SHORT_DROP_DELTA_V and tv < 0.7 * self.v_ego:
        confirm_count = 1
        for j in range(i + 1, len(forward_points)):
          next_tv = forward_points[j]["velocity"]
          next_d = forward_distances[j]
          if next_d >= MAX_SHORT_DROP_DISTANCE or next_tv > self.v_ego or next_tv >= 0.7 * self.v_ego:
            break
          confirm_count += 1
        if confirm_count < SHORT_DROP_CONFIRM_POINTS and not (self.v_target > 0 and self.target_lat == target_velocity["latitude"] and self.target_lon == target_velocity["longitude"]):
          continue

      if self._material_drop_needs_corroboration(tv) and not (self._current_lateral_accel_corroborates(sm) or self._model_corroborates(sm)):
        continue

      a_diff = (self.a_ego - TARGET_ACCEL)
      accel_t = abs(a_diff / TARGET_JERK)
      min_accel_v = calculate_velocity(accel_t, TARGET_JERK, self.a_ego, self.v_ego)

      max_d = 0
      if tv > min_accel_v:
        # calculate time needed based on target jerk
        a = 0.5 * TARGET_JERK
        b = self.a_ego
        c = self.v_ego - tv
        disc = b**2 - 4 * a * c
        if disc < 0:
          continue
        sqrt_disc = disc ** 0.5
        den = 2.0 * a
        t_a = (-b + sqrt_disc) / den
        t_b = (-b - sqrt_disc) / den
        roots = [t for t in (t_a, t_b) if _is_finite_number(t) and t > 0]
        if not roots:
          continue
        t = min(roots)

        max_d = max_d + calculate_distance(t, TARGET_JERK, self.a_ego, self.v_ego)
      else:
        t = accel_t
        if t <= 0:
          continue
        max_d = calculate_distance(t, TARGET_JERK, self.a_ego, self.v_ego)

        # calculate additional time needed based on target accel
        t = abs((min_accel_v - tv) / TARGET_ACCEL)
        max_d += calculate_distance(t, 0, TARGET_ACCEL, min_accel_v)

      if d < max_d + tv * TARGET_OFFSET:
        valid_velocities.append((d, float(tv), target_velocity["latitude"], target_velocity["longitude"]))

    if not valid_velocities:
      self.v_target = 0.0
      self.target_lat = 0.0
      self.target_lon = 0.0
      return

    # Prefer the nearest valid upcoming target, not the lowest global speed.
    valid_velocities.sort(key=lambda x: (x[0], x[1]))
    best_d, min_v, target_lat, target_lon = valid_velocities[0]

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

  def update(self, long_enabled: bool, long_override: bool, v_ego, a_ego, v_cruise, sm=None) -> None:
    self.long_enabled = long_enabled
    self.long_override = long_override
    self.v_ego = v_ego
    self.a_ego = a_ego
    self.v_cruise = v_cruise

    self.update_params()
    if not (self.long_enabled and self.enabled):
      self._clear_targets()
    else:
      self.update_calculations(sm)

    self.is_enabled, self.is_active = self._update_state_machine()

    self.output_v_target = self.get_v_target_from_control()
    self.output_a_target = self.get_a_target_from_control()

    self.frame += 1
