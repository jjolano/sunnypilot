import json
import math
from types import SimpleNamespace

import pytest
from cereal import log

from openpilot.sunnypilot.mapd.live_map_data.mapd_v2_map_data import MapdV2MapData
from openpilot.sunnypilot.mapd.param_helpers import (
  MAP_ADVISORY_UPDATED_AT_PARAM,
  MAP_TARGET_VELOCITIES_UPDATED_AT_PARAM,
)


class FakeParams:
  def __init__(self):
    self.values = {}
    self.put_counts = {}

  def get(self, key):
    return self.values.get(key)

  def put(self, key, value):
    self.put_counts[key] = self.put_counts.get(key, 0) + 1
    self.values[key] = value

  def remove(self, key):
    self.values.pop(key, None)


class FakePubMaster:
  def __init__(self):
    self.messages = []

  def send(self, service, msg):
    self.messages.append((service, msg))


def make_mapd_out(**overrides):
  values = {
    "roadName": "Main St",
    "speedLimit": 22.0,
    "nextSpeedLimit": 15.0,
    "nextSpeedLimitDistance": 120.0,
    "hazard": "animal_crossing",
    "nextHazard": "curve",
    "nextHazardDistance": 80.0,
    "advisorySpeed": 12.0,
    "nextAdvisorySpeed": 10.0,
    "nextAdvisorySpeedDistance": 60.0,
    "lanes": 3,
    "roadContext": "freeway",
  }
  values.update(overrides)
  return SimpleNamespace(**values)


def make_mapd_extended(**overrides):
  values = {
    "path": [],
    "downloadProgress": SimpleNamespace(
      active=False,
      cancelled=False,
      totalFiles=0,
      downloadedFiles=0,
      locations=[],
      locationDetails=[],
    ),
  }
  values.update(overrides)
  return SimpleNamespace(**values)


def make_location(**overrides):
  values = {
    "gpsOK": True,
    "status": log.LiveLocationKalman.Status.valid,
    "positionGeodetic": SimpleNamespace(valid=True, value=[39.0, -84.0, 0.0]),
    "calibratedOrientationNED": SimpleNamespace(value=[0.0, 0.0, 0.5]),
  }
  values.update(overrides)
  return SimpleNamespace(**values)


def build_map_data(mapd_out=None, mapd_extended=None):
  data = MapdV2MapData.__new__(MapdV2MapData)
  data.sm = {
    "mapdOut": mapd_out or make_mapd_out(),
    "mapdExtendedOut": mapd_extended or make_mapd_extended(),
    "liveLocationKalman": make_location(),
  }
  data.params = FakeParams()
  data.mem_params = FakeParams()
  data.pm = FakePubMaster()
  data.last_position = None
  data.last_bearing = None
  data.localizer_valid = False
  data._download_progress_started = False
  return data


def build_map_data_with_update_state(updated: dict[str, bool]):
  data = build_map_data()

  class FakeSubMaster(dict):
    pass

  sm = FakeSubMaster(data.sm)
  sm.updated = updated
  data.sm = sm
  return data


def test_mapd_v2_getters_populate_live_map_fields():
  data = build_map_data()

  assert data.get_current_speed_limit() == 22.0
  assert data.get_next_speed_limit_and_distance() == (15.0, 120.0)
  assert data.get_current_road_name() == "Main St"
  assert data.get_current_hazard_and_distance() == ("animal_crossing", 0.0)
  assert data.get_next_hazard_and_distance() == ("curve", 80.0)
  assert data.get_current_lanes() == 3
  assert data.get_next_lanes_and_distance() == (0, 0.0)
  assert data.get_road_context() == "freeway"
  assert data.get_current_traffic_control_and_distance() == ("", 0.0)
  assert data.get_next_traffic_control_and_distance() == ("", 0.0)


def test_mapd_v2_publish_populates_supported_live_map_fields():
  data = build_map_data()

  data.publish()

  assert data.pm.messages[0][0] == "liveMapDataSP"
  live_map_data = data.pm.messages[0][1].liveMapDataSP
  assert live_map_data.speedLimitValid
  assert live_map_data.speedLimit == pytest.approx(22.0)
  assert live_map_data.speedLimitAheadValid
  assert live_map_data.speedLimitAhead == pytest.approx(15.0)
  assert live_map_data.speedLimitAheadDistance == pytest.approx(120.0)
  assert live_map_data.roadName == "Main St"
  assert live_map_data.hazardValid
  assert live_map_data.hazard == "animal_crossing"
  assert live_map_data.hazardAheadValid
  assert live_map_data.hazardAhead == "curve"
  assert live_map_data.hazardAheadDistance == pytest.approx(80.0)
  assert live_map_data.lanesValid
  assert live_map_data.lanes == 3
  assert live_map_data.roadContext == "freeway"
  assert not live_map_data.trafficControlValid
  assert not live_map_data.trafficControlAheadValid


def test_mapd_v2_publish_exposes_supported_hazards_as_traffic_controls():
  data = build_map_data(make_mapd_out(hazard="stop_sign", nextHazard="traffic_signal", nextHazardDistance=35.0))

  data.publish()

  live_map_data = data.pm.messages[0][1].liveMapDataSP
  assert live_map_data.trafficControlValid
  assert live_map_data.trafficControl == "stop_sign"
  assert live_map_data.trafficControlDistance == pytest.approx(0.0)
  assert live_map_data.trafficControlAheadValid
  assert live_map_data.trafficControlAhead == "traffic_signal"
  assert live_map_data.trafficControlAheadDistance == pytest.approx(35.0)


def test_mapd_v2_compat_params_write_path_and_advisory_data():
  path = [
    SimpleNamespace(latitude=1.0, longitude=2.0, targetVelocity=14.0),
    SimpleNamespace(latitude=3.0, longitude=4.0, targetVelocity=0.0),
  ]
  data = build_map_data(mapd_extended=make_mapd_extended(path=path))

  data.update_location()

  assert json.loads(data.mem_params.values["MapTargetVelocities"]) == [
    {"latitude": 1.0, "longitude": 2.0, "velocity": 14.0},
  ]
  assert data.mem_params.values["MapAdvisoryLimit"] == {"speedlimit": 12.0, "distance": 0.0}
  assert data.mem_params.values["NextMapAdvisoryLimit"] == {"speedlimit": 10.0, "distance": 60.0}


def test_mapd_v2_update_location_writes_advisory_heartbeat_for_mapd_out(monkeypatch):
  monkeypatch.setattr("openpilot.sunnypilot.mapd.param_helpers.time.monotonic", lambda: 123.0)
  data = build_map_data_with_update_state({"mapdOut": True, "mapdExtendedOut": False})

  data.update_location()

  assert data.mem_params.values[MAP_ADVISORY_UPDATED_AT_PARAM] == "123.0"
  assert MAP_TARGET_VELOCITIES_UPDATED_AT_PARAM not in data.mem_params.values


def test_mapd_v2_update_location_writes_target_velocity_heartbeat_for_extended_out(monkeypatch):
  monkeypatch.setattr("openpilot.sunnypilot.mapd.param_helpers.time.monotonic", lambda: 123.0)
  data = build_map_data_with_update_state({"mapdOut": False, "mapdExtendedOut": True})

  data.update_location()

  assert data.mem_params.values[MAP_TARGET_VELOCITIES_UPDATED_AT_PARAM] == "123.0"
  assert MAP_ADVISORY_UPDATED_AT_PARAM not in data.mem_params.values


def test_mapd_v2_update_location_does_not_refresh_target_heartbeat_from_mapd_out_only(monkeypatch):
  monkeypatch.setattr("openpilot.sunnypilot.mapd.param_helpers.time.monotonic", lambda: 123.0)
  data = build_map_data_with_update_state({"mapdOut": True, "mapdExtendedOut": False})
  data.mem_params.put(MAP_TARGET_VELOCITIES_UPDATED_AT_PARAM, "100.0")

  data.update_location()

  assert data.mem_params.values[MAP_ADVISORY_UPDATED_AT_PARAM] == "123.0"
  assert data.mem_params.values[MAP_TARGET_VELOCITIES_UPDATED_AT_PARAM] == "100.0"


def test_mapd_v2_update_location_skips_unchanged_legacy_param_writes():
  path = [SimpleNamespace(latitude=1.0, longitude=2.0, targetVelocity=14.0)]
  progress = SimpleNamespace(
    active=True,
    cancelled=False,
    totalFiles=10,
    downloadedFiles=4,
    locations=["nation.US"],
    locationDetails=[SimpleNamespace(location="nation.US", totalFiles=10, downloadedFiles=4)],
  )
  data = build_map_data(mapd_extended=make_mapd_extended(path=path, downloadProgress=progress))

  data.update_location()
  mem_counts = dict(data.mem_params.put_counts)
  param_counts = dict(data.params.put_counts)

  data.update_location()

  assert data.mem_params.put_counts == mem_counts
  assert data.params.put_counts == param_counts


def test_mapd_v2_updates_legacy_last_gps_position_for_map_distance_users():
  data = build_map_data()

  data.update_location()

  assert json.loads(data.mem_params.values["LastGPSPosition"]) == {
    "latitude": 39.0,
    "longitude": -84.0,
    "bearing": pytest.approx(math.degrees(0.5)),
  }


def test_mapd_v2_download_progress_updates_legacy_ui_params():
  progress = SimpleNamespace(
    active=True,
    cancelled=False,
    totalFiles=10,
    downloadedFiles=4,
    locations=["nation.US"],
    locationDetails=[SimpleNamespace(location="nation.US", totalFiles=10, downloadedFiles=4)],
  )
  data = build_map_data(mapd_extended=make_mapd_extended(downloadProgress=progress))

  data.update_location()

  assert data.mem_params.values["OSMDownloadLocations"] == {"paths": ["nation.US"], "active": True}
  assert data.params.values["OSMDownloadProgress"] == {
    "active": True,
    "cancelled": False,
    "total_files": 10,
    "downloaded_files": 4,
    "locations_to_download": ["nation.US"],
    "location_details": {
      "nation.US": {"location_total_files": 10, "location_downloaded_files": 4},
    },
  }


def test_mapd_v2_download_progress_keeps_pending_marker_until_progress_starts():
  data = build_map_data()
  data.mem_params.put("OSMDownloadLocations", {"paths": ["nation.US"], "pending": True})

  data.update_location()

  assert data.mem_params.values["OSMDownloadLocations"] == {"paths": ["nation.US"], "pending": True}


def test_mapd_v2_download_progress_keeps_pending_marker_after_stale_complete_frame():
  progress = SimpleNamespace(
    active=False,
    cancelled=False,
    totalFiles=10,
    downloadedFiles=10,
    locations=["nation.US"],
    locationDetails=[],
  )
  data = build_map_data(mapd_extended=make_mapd_extended(downloadProgress=progress))
  data.mem_params.put("OSMDownloadLocations", {"paths": ["nation.US"], "pending": True})

  data.update_location()

  assert data.mem_params.values["OSMDownloadLocations"] == {"paths": ["nation.US"], "pending": True}


def test_mapd_v2_download_progress_keeps_pending_marker_after_stale_cancelled_frame():
  progress = SimpleNamespace(
    active=False,
    cancelled=True,
    totalFiles=10,
    downloadedFiles=4,
    locations=["nation.US"],
    locationDetails=[],
  )
  data = build_map_data(mapd_extended=make_mapd_extended(downloadProgress=progress))
  data.mem_params.put("OSMDownloadLocations", {"paths": ["nation.US"], "pending": True})

  data.update_location()

  assert data.mem_params.values["OSMDownloadLocations"] == {"paths": ["nation.US"], "pending": True}


def test_mapd_v2_download_progress_clears_legacy_active_marker_when_complete():
  progress = SimpleNamespace(
    active=False,
    cancelled=False,
    totalFiles=10,
    downloadedFiles=10,
    locations=["nation.US"],
    locationDetails=[],
  )
  data = build_map_data(mapd_extended=make_mapd_extended(downloadProgress=progress))
  data.mem_params.put("OSMDownloadLocations", {"paths": ["nation.US"]})

  data.update_location()

  assert "OSMDownloadLocations" not in data.mem_params.values
