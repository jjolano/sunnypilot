"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

import pytest

from openpilot.sunnypilot.mapd.live_map_data.osm_map_data import OsmMapData
from openpilot.sunnypilot.navd.helpers import Coordinate


class MockParams:
  def __init__(self, next_speed_limit):
    self.next_speed_limit = next_speed_limit

  def get(self, key):
    if key == "NextMapSpeedLimit":
      return self.next_speed_limit
    return None


def build_osm_map_data(next_speed_limit):
  osm_map_data = OsmMapData.__new__(OsmMapData)
  osm_map_data.mem_params = MockParams(next_speed_limit)
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


def test_next_speed_limit_falls_back_to_coordinate_distance():
  osm_map_data = build_osm_map_data({
    "speedlimit": 20.0,
    "latitude": 1.0,
    "longitude": 0.0,
  })

  next_speed_limit, next_distance = osm_map_data.get_next_speed_limit_and_distance()

  assert next_speed_limit == 20.0
  assert next_distance == pytest.approx(Coordinate(0., 0.).distance_to(Coordinate(1., 0.)))
