"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

import pytest
import json

from openpilot.sunnypilot.mapd.live_map_data.osm_map_data import OsmMapData
from openpilot.sunnypilot.navd.helpers import Coordinate


class MockParams:
  def __init__(self, values):
    self.values = values

  def get(self, key):
    return self.values.get(key)


def build_osm_map_data(next_speed_limit=None, map_hazard=None, next_map_hazard=None,
                       map_traffic_control=None, next_map_traffic_control=None,
                       map_lanes=None, next_map_lanes=None, map_road_context=None):
  osm_map_data = OsmMapData.__new__(OsmMapData)
  osm_map_data.mem_params = MockParams({
    "NextMapSpeedLimit": next_speed_limit,
    "MapHazard": map_hazard,
    "NextMapHazard": next_map_hazard,
    "MapTrafficControl": map_traffic_control,
    "NextMapTrafficControl": next_map_traffic_control,
    "MapLanes": map_lanes,
    "NextMapLanes": next_map_lanes,
    "MapRoadContext": map_road_context,
  })
  osm_map_data.last_position = Coordinate(0., 0.)
  return osm_map_data


def test_next_speed_limit_prefers_mapd_distance():
  osm_map_data = build_osm_map_data({
    "speedlimit": 20.0,
    "latitude": 1.0,
    "longitude": 1.0,
    "distance": 123.0,
  })

  next_speed_limit, next_distance = osm_map_data.get_next_speed_limit_and_distance()

  assert next_speed_limit == 20.0
  assert next_distance == 123.0


def test_next_speed_limit_accepts_json_string():
  osm_map_data = build_osm_map_data(json.dumps({
    "speedlimit": 20.0,
    "distance": 123.0,
  }))

  next_speed_limit, next_distance = osm_map_data.get_next_speed_limit_and_distance()

  assert next_speed_limit == 20.0
  assert next_distance == 123.0


def test_next_speed_limit_ignores_malformed_json():
  osm_map_data = build_osm_map_data("not-json")

  next_speed_limit, next_distance = osm_map_data.get_next_speed_limit_and_distance()

  assert next_speed_limit == 0.0
  assert next_distance == 0.0


def test_next_speed_limit_falls_back_to_coordinate_distance():
  osm_map_data = build_osm_map_data({
    "speedlimit": 20.0,
    "latitude": 1.0,
    "longitude": 0.0,
  })

  next_speed_limit, next_distance = osm_map_data.get_next_speed_limit_and_distance()

  assert next_speed_limit == 20.0
  assert next_distance == pytest.approx(Coordinate(0., 0.).distance_to(Coordinate(1., 0.)))


def test_current_hazard_falls_back_to_coordinate_distance():
  osm_map_data = build_osm_map_data(map_hazard=json.dumps({
    "hazard": "animal_crossing",
    "start_latitude": 1.0,
    "start_longitude": 0.0,
  }))

  hazard, distance = osm_map_data.get_current_hazard_and_distance()

  assert hazard == "animal_crossing"
  assert distance == pytest.approx(Coordinate(0., 0.).distance_to(Coordinate(1., 0.)))


def test_next_hazard_prefers_mapd_distance():
  osm_map_data = build_osm_map_data(next_map_hazard=json.dumps({
    "hazard": "curve",
    "start_latitude": 1.0,
    "start_longitude": 1.0,
    "distance": 123.0,
  }))

  hazard, distance = osm_map_data.get_next_hazard_and_distance()

  assert hazard == "curve"
  assert distance == 123.0


def test_hazard_ignores_malformed_json():
  osm_map_data = build_osm_map_data(map_hazard="not-json")

  hazard, distance = osm_map_data.get_current_hazard_and_distance()

  assert hazard == ""
  assert distance == 0.0


def test_current_traffic_control_falls_back_to_coordinate_distance():
  osm_map_data = build_osm_map_data(map_traffic_control=json.dumps({
    "type": "stop_sign",
    "start_latitude": 1.0,
    "start_longitude": 0.0,
  }))

  control_type, distance = osm_map_data.get_current_traffic_control_and_distance()

  assert control_type == "stop_sign"
  assert distance == pytest.approx(Coordinate(0., 0.).distance_to(Coordinate(1., 0.)))


def test_next_traffic_control_accepts_traffic_control_alias_and_distance():
  osm_map_data = build_osm_map_data(next_map_traffic_control=json.dumps({
    "traffic_control": "traffic_signal",
    "start_latitude": 1.0,
    "start_longitude": 1.0,
    "distance": 123.0,
  }))

  control_type, distance = osm_map_data.get_next_traffic_control_and_distance()

  assert control_type == "traffic_signal"
  assert distance == 123.0


def test_traffic_control_ignores_missing_type():
  osm_map_data = build_osm_map_data(map_traffic_control=json.dumps({
    "start_latitude": 1.0,
    "start_longitude": 0.0,
  }))

  control_type, distance = osm_map_data.get_current_traffic_control_and_distance()

  assert control_type == ""
  assert distance == 0.0


def test_current_lanes_accepts_numeric_param():
  osm_map_data = build_osm_map_data(map_lanes="3")

  assert osm_map_data.get_current_lanes() == 3


def test_current_lanes_accepts_int_param_value():
  osm_map_data = build_osm_map_data(map_lanes=3)

  assert osm_map_data.get_current_lanes() == 3


def test_current_lanes_ignores_invalid_param():
  osm_map_data = build_osm_map_data(map_lanes="not-json")

  assert osm_map_data.get_current_lanes() == 0


def test_next_lanes_prefers_mapd_distance():
  osm_map_data = build_osm_map_data(next_map_lanes=json.dumps({
    "lanes": 2,
    "start_latitude": 1.0,
    "start_longitude": 1.0,
    "distance": 123.0,
  }))

  lanes, distance = osm_map_data.get_next_lanes_and_distance()

  assert lanes == 2
  assert distance == 123.0


def test_next_lanes_falls_back_to_coordinate_distance():
  osm_map_data = build_osm_map_data(next_map_lanes=json.dumps({
    "lanes": 1,
    "start_latitude": 1.0,
    "start_longitude": 0.0,
  }))

  lanes, distance = osm_map_data.get_next_lanes_and_distance()

  assert lanes == 1
  assert distance == pytest.approx(Coordinate(0., 0.).distance_to(Coordinate(1., 0.)))


def test_map_context_defaults_when_unpatched():
  osm_map_data = build_osm_map_data()

  lanes, lanes_distance = osm_map_data.get_next_lanes_and_distance()

  assert osm_map_data.get_current_lanes() == 0
  assert lanes == 0
  assert lanes_distance == 0.0
  assert osm_map_data.get_road_context() == ""


def test_road_context_accepts_known_context():
  osm_map_data = build_osm_map_data(map_road_context=b"freeway")

  assert osm_map_data.get_road_context() == "freeway"


def test_road_context_ignores_unknown_context():
  osm_map_data = build_osm_map_data(map_road_context="parking_lot")

  assert osm_map_data.get_road_context() == ""
