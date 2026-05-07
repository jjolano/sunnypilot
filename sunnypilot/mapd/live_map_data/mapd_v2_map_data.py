import json
import math
import platform

from cereal import log
from openpilot.common.params import Params
from openpilot.sunnypilot.mapd.live_map_data.base_map_data import BaseMapData
from openpilot.sunnypilot.navd.helpers import Coordinate


ROAD_CONTEXT_BY_VALUE = ("freeway", "city", "unknown")
TRAFFIC_CONTROL_HAZARDS = {
  "stop",
  "stop_sign",
  "traffic_light",
  "traffic_lights",
  "traffic_signal",
  "traffic_signals",
}
MAP_CURVATURE_MAX_ROUTE_DISTANCE = 20.0


def _getattr_or_default(msg, name: str, default=None):
  try:
    return getattr(msg, name)
  except Exception:
    return default


def _text(value) -> str:
  if isinstance(value, bytes):
    value = value.decode("utf-8", errors="ignore")
  return str(value or "")


def _float(value, default: float = 0.0) -> float:
  try:
    return float(value)
  except (TypeError, ValueError):
    return default


def _int(value, default: int = 0) -> int:
  try:
    return int(value)
  except (TypeError, ValueError):
    return default


def _road_context_name(context) -> str:
  if isinstance(context, str):
    name = context.split(".")[-1].strip().lower()
  elif isinstance(context, int) and 0 <= context < len(ROAD_CONTEXT_BY_VALUE):
    name = ROAD_CONTEXT_BY_VALUE[context]
  else:
    name = str(context or "").split(".")[-1].strip().lower()

  return name if name in ROAD_CONTEXT_BY_VALUE else ""


def _traffic_control_hazard(hazard) -> str:
  hazard_text = _text(hazard)
  normalized = hazard_text.strip().lower().replace("-", "_").replace(" ", "_")
  return hazard_text if normalized in TRAFFIC_CONTROL_HAZARDS else ""


class MapdV2MapData(BaseMapData):
  def __init__(self):
    super().__init__(["mapdOut", "mapdExtendedOut"])
    self.mem_params = Params("/dev/shm/params") if platform.system() != "Darwin" else self.params
    self._download_progress_started = False
    self.mem_params.remove("OSMDownloadLocations")

  @staticmethod
  def _put_if_changed(params, key: str, value) -> None:
    current = params.get(key)
    if isinstance(current, bytes) and isinstance(value, str):
      current = current.decode("utf-8", errors="ignore")
    if current != value:
      params.put(key, value)

  @property
  def mapd_out(self):
    return self.sm["mapdOut"]

  @property
  def mapd_extended_out(self):
    return self.sm["mapdExtendedOut"]

  def update_location(self) -> None:
    self._write_last_gps_position()
    self._write_map_compat_params()
    self._write_download_progress()

  def get_current_speed_limit(self) -> float:
    return _float(_getattr_or_default(self.mapd_out, "speedLimit"))

  def get_next_speed_limit_and_distance(self) -> tuple[float, float]:
    return (_float(_getattr_or_default(self.mapd_out, "nextSpeedLimit")),
            max(0.0, _float(_getattr_or_default(self.mapd_out, "nextSpeedLimitDistance"))))

  def get_current_road_name(self) -> str:
    return _text(_getattr_or_default(self.mapd_out, "roadName"))

  def get_current_lanes(self) -> int:
    lanes = _int(_getattr_or_default(self.mapd_out, "lanes"))
    return lanes if 0 < lanes <= 255 else 0

  def get_next_lanes_and_distance(self) -> tuple[int, float]:
    return 0, 0.0

  @staticmethod
  def _path_point_coordinate(point) -> Coordinate | None:
    latitude = _float(_getattr_or_default(point, "latitude"), math.nan)
    longitude = _float(_getattr_or_default(point, "longitude"), math.nan)
    if not math.isfinite(latitude) or not math.isfinite(longitude):
      return None
    return Coordinate(latitude, longitude)

  @staticmethod
  def _segment_projection_fraction(start: Coordinate, end: Coordinate, position: Coordinate) -> float:
    mean_lat = math.radians((start.latitude + end.latitude + position.latitude) / 3.0)
    ab_x = (end.longitude - start.longitude) * math.cos(mean_lat)
    ab_y = end.latitude - start.latitude
    ap_x = (position.longitude - start.longitude) * math.cos(mean_lat)
    ap_y = position.latitude - start.latitude
    denom = ab_x * ab_x + ab_y * ab_y
    if denom <= 0.0:
      return 0.0
    return max(0.0, min(1.0, (ap_x * ab_x + ap_y * ab_y) / denom))

  @staticmethod
  def _interpolate_coordinate(start: Coordinate, end: Coordinate, fraction: float) -> Coordinate:
    return Coordinate(
      start.latitude + (end.latitude - start.latitude) * fraction,
      start.longitude + (end.longitude - start.longitude) * fraction,
    )

  @classmethod
  def _project_position_on_path(cls, path_points: list[tuple[Coordinate, float]], position: Coordinate) -> tuple[float, float, list[float]] | None:
    cumulative_distances = [0.0]
    for i in range(1, len(path_points)):
      cumulative_distances.append(cumulative_distances[-1] + path_points[i - 1][0].distance_to(path_points[i][0]))

    if len(path_points) == 1:
      return 0.0, position.distance_to(path_points[0][0]), cumulative_distances

    closest_route_distance = 0.0
    closest_distance = float("inf")
    for i in range(len(path_points) - 1):
      start = path_points[i][0]
      end = path_points[i + 1][0]
      segment_length = start.distance_to(end)
      fraction = cls._segment_projection_fraction(start, end, position)
      projected = cls._interpolate_coordinate(start, end, fraction)
      distance_to_segment = position.distance_to(projected)
      if distance_to_segment < closest_distance:
        closest_distance = distance_to_segment
        closest_route_distance = cumulative_distances[i] + segment_length * fraction

    return closest_route_distance, closest_distance, cumulative_distances

  def get_road_curvatures(self) -> tuple[list[float], list[float]]:
    all_checks = getattr(self.sm, "all_checks", None)
    if all_checks is not None and not all_checks(["mapdExtendedOut"]):
      return [], []

    path_points = []
    for point in _getattr_or_default(self.mapd_extended_out, "path", []) or []:
      coordinate = self._path_point_coordinate(point)
      curvature = _float(_getattr_or_default(point, "curvature"), math.nan)
      if coordinate is None or not math.isfinite(curvature):
        continue
      path_points.append((coordinate, curvature))

    if not path_points:
      return [], []

    projection = self._project_position_on_path(path_points, self.last_position) if self.last_position is not None else None
    if projection is None:
      distances = [0.0]
      curvatures = [float(path_points[0][1])]
      for coordinate, curvature in path_points[1:]:
        distances.append(distances[-1] + path_points[len(distances) - 1][0].distance_to(coordinate))
        curvatures.append(float(curvature))
      return distances, curvatures

    current_route_distance, distance_from_route, cumulative_distances = projection
    if distance_from_route > MAP_CURVATURE_MAX_ROUTE_DISTANCE:
      return [], []

    distances = []
    curvatures = []
    for i, (_, curvature) in enumerate(path_points):
      distance = cumulative_distances[i] - current_route_distance
      if distance >= 0.0:
        distances.append(distance)
        curvatures.append(float(curvature))

    return distances, curvatures

  def get_road_context(self) -> str:
    return _road_context_name(_getattr_or_default(self.mapd_out, "roadContext"))

  def get_current_hazard_and_distance(self) -> tuple[str, float]:
    return _text(_getattr_or_default(self.mapd_out, "hazard")), 0.0

  def get_next_hazard_and_distance(self) -> tuple[str, float]:
    return (_text(_getattr_or_default(self.mapd_out, "nextHazard")),
            max(0.0, _float(_getattr_or_default(self.mapd_out, "nextHazardDistance"))))

  def get_current_traffic_control_and_distance(self) -> tuple[str, float]:
    return _traffic_control_hazard(_getattr_or_default(self.mapd_out, "hazard")), 0.0

  def get_next_traffic_control_and_distance(self) -> tuple[str, float]:
    traffic_control = _traffic_control_hazard(_getattr_or_default(self.mapd_out, "nextHazard"))
    return traffic_control, max(0.0, _float(_getattr_or_default(self.mapd_out, "nextHazardDistance"))) if traffic_control else 0.0

  def _write_map_compat_params(self) -> None:
    target_velocities = []
    for point in _getattr_or_default(self.mapd_extended_out, "path", []) or []:
      target_velocity = _float(_getattr_or_default(point, "targetVelocity"))
      latitude = _float(_getattr_or_default(point, "latitude"))
      longitude = _float(_getattr_or_default(point, "longitude"))
      if target_velocity > 0.0:
        target_velocities.append({"latitude": latitude, "longitude": longitude, "velocity": target_velocity})

    self._put_if_changed(self.mem_params, "MapTargetVelocities", json.dumps(target_velocities))

    advisory_speed = _float(_getattr_or_default(self.mapd_out, "advisorySpeed"))
    self._put_if_changed(self.mem_params, "MapAdvisoryLimit", {"speedlimit": advisory_speed, "distance": 0.0} if advisory_speed > 0.0 else {})

    next_advisory_speed = _float(_getattr_or_default(self.mapd_out, "nextAdvisorySpeed"))
    next_advisory_distance = max(0.0, _float(_getattr_or_default(self.mapd_out, "nextAdvisorySpeedDistance")))
    self._put_if_changed(self.mem_params, "NextMapAdvisoryLimit", {
      "speedlimit": next_advisory_speed,
      "distance": next_advisory_distance,
    } if next_advisory_speed > 0.0 else {})

  def _write_last_gps_position(self) -> None:
    location = self.sm['liveLocationKalman']
    self.localizer_valid = (location.status == log.LiveLocationKalman.Status.valid) and location.positionGeodetic.valid

    if self.localizer_valid:
      self.last_bearing = math.degrees(location.calibratedOrientationNED.value[2])
      self.last_position = Coordinate(location.positionGeodetic.value[0], location.positionGeodetic.value[1])

    if self.last_position is None:
      return

    position = {
      "latitude": self.last_position.latitude,
      "longitude": self.last_position.longitude,
    }
    if self.last_bearing is not None:
      position["bearing"] = self.last_bearing

    self._put_if_changed(self.mem_params, "LastGPSPosition", json.dumps(position))

  def _download_marker_pending(self) -> bool:
    marker = self.mem_params.get("OSMDownloadLocations")
    return isinstance(marker, dict) and bool(marker.get("pending"))

  def _write_download_progress(self) -> None:
    progress = _getattr_or_default(self.mapd_extended_out, "downloadProgress")
    if progress is None:
      return

    active = bool(_getattr_or_default(progress, "active", False))
    locations = [_text(location) for location in (_getattr_or_default(progress, "locations", []) or []) if _text(location)]
    if active:
      self._download_progress_started = True
      self._put_if_changed(self.mem_params, "OSMDownloadLocations", {"paths": locations, "active": True})
    else:
      total_files = _int(_getattr_or_default(progress, "totalFiles"))
      cancelled = bool(_getattr_or_default(progress, "cancelled", False))
      if not self._download_marker_pending() and (total_files > 0 or cancelled or self._download_progress_started):
        self._download_progress_started = False
        self.mem_params.remove("OSMDownloadLocations")

    location_details = {}
    for detail in _getattr_or_default(progress, "locationDetails", []) or []:
      location = _text(_getattr_or_default(detail, "location"))
      if not location:
        continue
      location_details[location] = {
        "location_total_files": _int(_getattr_or_default(detail, "totalFiles")),
        "location_downloaded_files": _int(_getattr_or_default(detail, "downloadedFiles")),
      }

    self._put_if_changed(self.params, "OSMDownloadProgress", {
      "active": active,
      "cancelled": bool(_getattr_or_default(progress, "cancelled", False)),
      "total_files": _int(_getattr_or_default(progress, "totalFiles")),
      "downloaded_files": _int(_getattr_or_default(progress, "downloadedFiles")),
      "locations_to_download": locations,
      "location_details": location_details,
    })
