"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import json
import math
import platform

from cereal import log
from openpilot.common.params import Params
from openpilot.sunnypilot.mapd.live_map_data.base_map_data import BaseMapData
from openpilot.sunnypilot.mapd.param_helpers import get_mapd_json, mapd_section_float, mapd_section_int
from openpilot.sunnypilot.navd.helpers import Coordinate


class OsmMapData(BaseMapData):
  def __init__(self):
    super().__init__()
    self.mem_params = Params("/dev/shm/params") if platform.system() != "Darwin" else self.params

  def update_location(self) -> None:
    location = self.sm['liveLocationKalman']
    self.localizer_valid = (location.status == log.LiveLocationKalman.Status.valid) and location.positionGeodetic.valid

    if self.localizer_valid:
      self.last_bearing = math.degrees(location.calibratedOrientationNED.value[2])
      self.last_position = Coordinate(location.positionGeodetic.value[0], location.positionGeodetic.value[1])

    if self.last_position is None:
      return

    params = {
      "latitude": self.last_position.latitude,
      "longitude": self.last_position.longitude,
    }

    if self.last_bearing is not None:
      params['bearing'] = self.last_bearing

    self.mem_params.put("LastGPSPosition", json.dumps(params))

  def get_current_speed_limit(self) -> float:
    return float(self.mem_params.get("MapSpeedLimit") or 0.0)

  def get_current_road_name(self) -> str:
    return str(self.mem_params.get("RoadName") or "")

  @staticmethod
  def _lanes_from_value(value) -> int:
    if isinstance(value, dict):
      lanes = mapd_section_int(value, "lanes")
    else:
      try:
        lanes = int(value or 0)
      except (TypeError, ValueError):
        lanes = 0

    return lanes if 0 < lanes <= 255 else 0

  def get_current_lanes(self) -> int:
    return self._lanes_from_value(get_mapd_json(self.mem_params, "MapLanes"))

  def get_next_lanes_and_distance(self) -> tuple[int, float]:
    next_lanes_section = get_mapd_json(self.mem_params, "NextMapLanes") or {}
    if not isinstance(next_lanes_section, dict):
      return 0, 0.0

    lanes = self._lanes_from_value(next_lanes_section)
    if lanes <= 0:
      return 0, 0.0

    return lanes, self._distance_to_section_start(next_lanes_section)

  def get_road_curvatures(self) -> tuple[list[float], list[float]]:
    return [], []

  def get_road_context(self) -> str:
    try:
      context = self.mem_params.get("MapRoadContext")
    except Exception:
      return ""

    if isinstance(context, bytes):
      context = context.decode("utf-8", errors="ignore")

    context = str(context or "").strip().lower()
    return context if context in ("freeway", "city", "unknown") else ""

  def _distance_to_section_start(self, section: dict) -> float:
    distance = mapd_section_float(section, 'distance', None)
    if distance is not None:
      return max(0.0, distance)

    latitude = mapd_section_float(section, 'start_latitude', None)
    longitude = mapd_section_float(section, 'start_longitude', None)
    if latitude is None or longitude is None:
      return 0.0

    return (self.last_position or Coordinate(0, 0)).distance_to(Coordinate(latitude, longitude))

  def _get_hazard_and_distance(self, key: str) -> tuple[str, float]:
    hazard_section = get_mapd_json(self.mem_params, key) or {}
    if not isinstance(hazard_section, dict):
      return "", 0.0

    hazard = str(hazard_section.get('hazard') or "")
    if not hazard:
      return "", 0.0

    return hazard, self._distance_to_section_start(hazard_section)

  def get_current_hazard_and_distance(self) -> tuple[str, float]:
    return self._get_hazard_and_distance("MapHazard")

  def get_next_hazard_and_distance(self) -> tuple[str, float]:
    return self._get_hazard_and_distance("NextMapHazard")

  @staticmethod
  def _traffic_control_type(section: dict) -> str:
    for key in ("type", "trafficControl", "traffic_control", "control"):
      value = str(section.get(key) or "")
      if value:
        return value
    return ""

  def _get_traffic_control_and_distance(self, key: str) -> tuple[str, float]:
    control_section = get_mapd_json(self.mem_params, key) or {}
    if not isinstance(control_section, dict):
      return "", 0.0

    control_type = self._traffic_control_type(control_section)
    if not control_type:
      return "", 0.0

    return control_type, self._distance_to_section_start(control_section)

  def get_current_traffic_control_and_distance(self) -> tuple[str, float]:
    return self._get_traffic_control_and_distance("MapTrafficControl")

  def get_next_traffic_control_and_distance(self) -> tuple[str, float]:
    return self._get_traffic_control_and_distance("NextMapTrafficControl")

  def get_next_speed_limit_and_distance(self) -> tuple[float, float]:
    next_speed_limit_section = get_mapd_json(self.mem_params, "NextMapSpeedLimit") or {}
    if not isinstance(next_speed_limit_section, dict):
      return 0.0, 0.0

    next_speed_limit = mapd_section_float(next_speed_limit_section, 'speedlimit') or 0.0
    next_speed_limit_latitude = mapd_section_float(next_speed_limit_section, 'latitude', None)
    next_speed_limit_longitude = mapd_section_float(next_speed_limit_section, 'longitude', None)
    next_speed_limit_distance = mapd_section_float(next_speed_limit_section, 'distance') or 0.0

    if next_speed_limit_distance <= 0.0 and next_speed_limit_latitude is not None and next_speed_limit_longitude is not None:
      next_speed_limit_coordinates = Coordinate(next_speed_limit_latitude, next_speed_limit_longitude)
      next_speed_limit_distance = (self.last_position or Coordinate(0, 0)).distance_to(next_speed_limit_coordinates)

    return next_speed_limit, next_speed_limit_distance
