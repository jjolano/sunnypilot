"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import json

import pytest

from cereal import custom
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.car.cruise import V_CRUISE_UNSET
from openpilot.sunnypilot.navd.helpers import Coordinate
from openpilot.sunnypilot.selfdrive.controls.lib.smart_cruise_control.map_controller import SmartCruiseControlMap, point_distance, velocities_from_param

MapState = VisionState = custom.LongitudinalPlanSP.SmartCruiseControl.MapState


class MockParams:
  def __init__(self):
    self.values = {}

  def get(self, key):
    return self.values.get(key)

  def put(self, key, value):
    self.values[key] = value


class TestSmartCruiseControlMap:

  def setup_method(self):
    self.params = Params()
    self.mem_params = MockParams()
    self.reset_params()
    self.scc_m = SmartCruiseControlMap()
    self.scc_m.mem_params = self.mem_params

  def reset_params(self):
    self.params.put_bool("SmartCruiseControlMap", True)

    # TODO-SP: mock data from gpsLocation
    self.mem_params.put("LastGPSPosition", json.dumps({"latitude": 0.0, "longitude": 0.0}))
    self.mem_params.put("MapTargetVelocities", "[]")
    self.mem_params.put("MapAdvisorySpeedLimit", "{}")
    self.mem_params.put("MapAdvisoryLimit", "{}")
    self.mem_params.put("NextMapAdvisorySpeedLimit", "{}")
    self.mem_params.put("NextMapAdvisoryLimit", "{}")

  def test_initial_state(self):
    assert self.scc_m.state == VisionState.disabled
    assert not self.scc_m.is_active
    assert self.scc_m.output_v_target == V_CRUISE_UNSET
    assert self.scc_m.output_a_target == 0.

  def test_system_disabled(self):
    self.params.put_bool("SmartCruiseControlMap", False)
    self.scc_m.enabled = self.params.get_bool("SmartCruiseControlMap")

    for _ in range(int(10. / DT_MDL)):
      self.scc_m.update(True, False, 0., 0., 0.)
    assert self.scc_m.state == VisionState.disabled
    assert not self.scc_m.is_active

  def test_disabled(self):
    for _ in range(int(10. / DT_MDL)):
      self.scc_m.update(False, False, 0., 0., 0.)
    assert self.scc_m.state == VisionState.disabled

  def test_transition_disabled_to_enabled(self):
    for _ in range(int(10. / DT_MDL)):
      self.scc_m.update(True, False, 0., 0., 0.)
    assert self.scc_m.state == VisionState.enabled

  # TODO-SP: mock data from modelV2 to test other states

  def test_velocities_from_param_ignores_malformed_data(self):
    self.mem_params.put("MapTargetVelocities", "not-json")

    assert velocities_from_param("MapTargetVelocities", self.mem_params) == []

  def test_forward_target_velocity_distances_follow_ordered_path(self):
    first = Coordinate(0.0, 0.001)
    second = Coordinate(0.001, 0.001)
    third = Coordinate(0.001, 0.002)
    self.scc_m.last_position = Coordinate(0.0, 0.0)
    self.scc_m.target_velocities = [
      {"latitude": first.latitude, "longitude": first.longitude, "velocity": 25.0},
      {"latitude": second.latitude, "longitude": second.longitude, "velocity": 15.0},
      {"latitude": third.latitude, "longitude": third.longitude, "velocity": 20.0},
    ]

    forward_points, forward_distances = self.scc_m._forward_target_velocity_distances()

    assert forward_points == self.scc_m.target_velocities
    assert forward_distances[0] == pytest.approx(point_distance(Coordinate(0.0, 0.0), first))
    assert forward_distances[1] == pytest.approx(forward_distances[0] + point_distance(first, second))
    assert forward_distances[2] == pytest.approx(forward_distances[1] + point_distance(second, third))

  def test_current_advisory_speed_limit_controls_scc_map_target(self):
    self.mem_params.put("MapAdvisorySpeedLimit", json.dumps({
      "start_latitude": 0.0,
      "start_longitude": 0.0,
      "end_latitude": 0.0,
      "end_longitude": 0.001,
      "speedlimit": 15.0,
    }))

    for _ in range(2):
      self.scc_m.update(True, False, 25.0, 0.0, 30.0)

    assert self.scc_m.state == VisionState.turning
    assert self.scc_m.output_v_target == 15.0

  def test_current_advisory_limit_alias_controls_scc_map_target(self):
    self.mem_params.put("MapAdvisoryLimit", json.dumps({
      "start_latitude": 0.0,
      "start_longitude": 0.0,
      "end_latitude": 0.0,
      "end_longitude": 0.001,
      "speedlimit": 14.0,
    }))

    for _ in range(2):
      self.scc_m.update(True, False, 25.0, 0.0, 30.0)

    assert self.scc_m.state == VisionState.turning
    assert self.scc_m.output_v_target == 14.0

  def test_faster_advisory_speed_limit_is_ignored(self):
    self.mem_params.put("MapAdvisorySpeedLimit", json.dumps({
      "start_latitude": 0.0,
      "start_longitude": 0.0,
      "end_latitude": 0.0,
      "end_longitude": 0.001,
      "speedlimit": 30.0,
    }))

    for _ in range(2):
      self.scc_m.update(True, False, 20.0, 0.0, 25.0)

    assert self.scc_m.state == VisionState.enabled
    assert self.scc_m.output_v_target == V_CRUISE_UNSET

  def test_next_advisory_speed_limit_controls_inside_adapt_distance(self):
    self.mem_params.put("NextMapAdvisorySpeedLimit", json.dumps({
      "start_latitude": 0.0,
      "start_longitude": 0.0,
      "end_latitude": 0.0,
      "end_longitude": 0.001,
      "speedlimit": 15.0,
      "distance": 10.0,
    }))

    for _ in range(2):
      self.scc_m.update(True, False, 25.0, 0.0, 30.0)

    assert self.scc_m.state == VisionState.turning
    assert self.scc_m.output_v_target == 15.0

  def test_next_advisory_speed_limit_waits_until_adapt_distance(self):
    self.mem_params.put("NextMapAdvisorySpeedLimit", json.dumps({
      "start_latitude": 0.0,
      "start_longitude": 0.0,
      "end_latitude": 0.0,
      "end_longitude": 0.001,
      "speedlimit": 15.0,
      "distance": 10000.0,
    }))

    for _ in range(2):
      self.scc_m.update(True, False, 25.0, 0.0, 30.0)

    assert self.scc_m.state == VisionState.enabled
    assert self.scc_m.output_v_target == V_CRUISE_UNSET
