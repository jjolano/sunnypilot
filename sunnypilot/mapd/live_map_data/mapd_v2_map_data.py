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

    self.mem_params.put("MapTargetVelocities", json.dumps(target_velocities))

    advisory_speed = _float(_getattr_or_default(self.mapd_out, "advisorySpeed"))
    self.mem_params.put("MapAdvisoryLimit", json.dumps({"speedlimit": advisory_speed, "distance": 0.0} if advisory_speed > 0.0 else {}))

    next_advisory_speed = _float(_getattr_or_default(self.mapd_out, "nextAdvisorySpeed"))
    next_advisory_distance = max(0.0, _float(_getattr_or_default(self.mapd_out, "nextAdvisorySpeedDistance")))
    self.mem_params.put("NextMapAdvisoryLimit", json.dumps({
      "speedlimit": next_advisory_speed,
      "distance": next_advisory_distance,
    } if next_advisory_speed > 0.0 else {}))

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

    self.mem_params.put("LastGPSPosition", json.dumps(position))

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
      self.mem_params.put("OSMDownloadLocations", {"paths": locations, "active": True})
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

    self.params.put("OSMDownloadProgress", {
      "active": active,
      "cancelled": bool(_getattr_or_default(progress, "cancelled", False)),
      "total_files": _int(_getattr_or_default(progress, "totalFiles")),
      "downloaded_files": _int(_getattr_or_default(progress, "downloadedFiles")),
      "locations_to_download": locations,
      "location_details": location_details,
    })
