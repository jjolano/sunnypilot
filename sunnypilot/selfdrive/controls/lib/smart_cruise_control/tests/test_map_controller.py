"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import json
import math
from types import SimpleNamespace

import pytest

from cereal import custom
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.car.cruise import V_CRUISE_UNSET
from openpilot.selfdrive.controls.lib.vehicle_math import speed_for_lateral_accel
from openpilot.sunnypilot.selfdrive.controls.lib.smart_cruise_control import map_controller
from openpilot.sunnypilot.navd.helpers import Coordinate
from openpilot.sunnypilot.selfdrive.controls.lib.smart_cruise_control.map_controller import (
  R,
  TO_DEGREES,
  SmartCruiseControlMap,
  SunnypilotCurrentSmartCruiseControlMap,
  distance_to_point,
  point_distance,
  sunnypilot_current_velocities_from_param,
  velocities_from_param,
)

MapState = VisionState = custom.LongitudinalPlanSP.SmartCruiseControl.MapState


def make_model_prediction(distance: float, yaw_rate: float, speed: float = 25.0):
  positions = [0.0, max(0.0, distance - 10.0), distance, distance + 10.0]
  velocities = [speed for _ in positions]
  yaw_rates = [0.0, yaw_rate, yaw_rate, 0.0]
  return SimpleNamespace(
    position=SimpleNamespace(x=positions),
    velocity=SimpleNamespace(x=velocities),
    orientationRate=SimpleNamespace(z=yaw_rates),
  )


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
    self.mem_params.put("LastGPSPositionValid", "1")
    self.mem_params.put("MapTargetVelocitiesValid", "1")

  def test_initial_state(self):
    assert self.scc_m.state == VisionState.disabled
    assert not self.scc_m.is_active
    assert self.scc_m.output_v_target == V_CRUISE_UNSET
    assert self.scc_m.output_a_target == 0.

  def test_target_control_distance_uses_full_quadratic_denominator(self):
    self.scc_m.v_ego = 25.0
    self.scc_m.a_ego = 0.0
    target_v = 24.0
    a = 0.5 * map_controller.TARGET_JERK
    b = self.scc_m.a_ego
    c = self.scc_m.v_ego - target_v
    expected_t = (-b - (b**2 - 4 * a * c) ** 0.5) / (2 * a)
    expected_distance = map_controller.calculate_distance(
      expected_t, map_controller.TARGET_JERK, self.scc_m.a_ego, self.scc_m.v_ego,
    ) + target_v * map_controller.TARGET_OFFSET

    assert self.scc_m._target_control_distance(target_v) == pytest.approx(expected_distance)

  def test_validity_params_are_registered(self):
    assert self.params.get("LastGPSPositionValid") is None
    assert self.params.get("MapTargetVelocitiesValid") is None

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

  def test_velocities_from_param_rejects_non_finite_coordinates(self):
    self.mem_params.put("MapTargetVelocities", json.dumps([
      {"latitude": float("inf"), "longitude": 0.001, "velocity": 15.0},
      {"latitude": 0.0, "longitude": float("nan"), "velocity": 15.0},
    ]))

    assert velocities_from_param("MapTargetVelocities", self.mem_params) == []

  def test_distance_to_point_returns_infinity_for_non_finite_inputs(self):
    assert math.isinf(distance_to_point(0.0, 0.0, float("inf"), 0.0))

  def test_distance_to_point_clamps_haversine_roundoff(self, monkeypatch):
    monkeypatch.setattr(map_controller.math, "sin", lambda _value: 1.0)
    monkeypatch.setattr(map_controller.math, "cos", lambda _value: 1.0)

    assert distance_to_point(0.0, 0.0, 0.0, 0.0) == pytest.approx(math.pi * R)

  def test_advisory_distance_rejects_non_finite_coordinates(self):
    section = {"start_latitude": float("inf"), "start_longitude": 0.0}

    assert self.scc_m._distance_to_advisory_start(section) is None

  def test_advisory_target_rejects_non_finite_coordinates(self):
    section = {"speedlimit": 15.0, "start_latitude": float("inf"), "start_longitude": 0.0}

    assert self.scc_m._advisory_target(section) is None

  def test_update_calculations_reuses_cached_target_velocity_parse(self, monkeypatch):
    calls = 0
    base_velocities_from_param = velocities_from_param

    def counting_velocities_from_param(param, params):
      nonlocal calls
      calls += 1
      return base_velocities_from_param(param, params)

    monkeypatch.setattr(map_controller, "velocities_from_param", counting_velocities_from_param)
    self.mem_params.put("MapTargetVelocities", json.dumps([
      {"latitude": 0.0, "longitude": 0.001, "velocity": 15.0},
    ]))

    self.scc_m.update_calculations()
    self.scc_m.update_calculations()

    assert calls == 1

  def test_update_calculations_reuses_cached_advisory_parse(self, monkeypatch):
    calls = 0
    base_get_first_mapd_json = map_controller.get_first_mapd_json

    def counting_get_first_mapd_json(params, keys):
      nonlocal calls
      calls += 1
      return base_get_first_mapd_json(params, keys)

    monkeypatch.setattr(map_controller, "get_first_mapd_json", counting_get_first_mapd_json)
    self.mem_params.put("MapAdvisoryLimit", json.dumps({"speedlimit": 15.0, "distance": 0.0}))
    self.mem_params.put("NextMapAdvisoryLimit", json.dumps({"speedlimit": 14.0, "distance": 10.0}))

    self.scc_m.update_calculations()
    self.scc_m.update_calculations()

    assert calls == 2

  def test_stale_map_params_clear_active_target(self, monkeypatch):
    self.mem_params.put("MapAdvisorySpeedLimit", json.dumps({
      "start_latitude": 0.0,
      "start_longitude": 0.0,
      "speedlimit": 15.0,
    }))
    monkeypatch.setattr(map_controller.time, "monotonic", lambda: 106.5)
    self.mem_params.put(map_controller.MAP_ADVISORY_UPDATED_AT_PARAM, "105.0")

    for _ in range(2):
      self.scc_m.update(True, False, 25.0, 0.0, 30.0)
    assert self.scc_m.state == VisionState.turning

    self.mem_params.put(map_controller.MAP_ADVISORY_UPDATED_AT_PARAM, "100.0")
    self.scc_m.update(True, False, 25.0, 0.0, 30.0)

    assert self.scc_m.state == VisionState.enabled
    assert self.scc_m.output_v_target == V_CRUISE_UNSET

  def test_invalid_map_params_clear_active_target(self, monkeypatch):
    self.mem_params.put("MapAdvisorySpeedLimit", json.dumps({
      "start_latitude": 0.0,
      "start_longitude": 0.0,
      "speedlimit": 15.0,
    }))
    monkeypatch.setattr(map_controller.time, "monotonic", lambda: 106.5)
    self.mem_params.put(map_controller.MAP_ADVISORY_UPDATED_AT_PARAM, "105.0")

    for _ in range(2):
      self.scc_m.update(True, False, 25.0, 0.0, 30.0)
    assert self.scc_m.state == VisionState.turning

    self.mem_params.put("LastGPSPositionValid", "0")
    self.scc_m.update(True, False, 25.0, 0.0, 30.0)

    assert self.scc_m.state == VisionState.enabled
    assert self.scc_m.output_v_target == V_CRUISE_UNSET

  def test_missing_heartbeat_keeps_compatibility(self, monkeypatch):
    self.mem_params.values.pop(map_controller.MAP_ADVISORY_UPDATED_AT_PARAM, None)
    self.mem_params.put("MapAdvisorySpeedLimit", json.dumps({
      "start_latitude": 0.0,
      "start_longitude": 0.0,
      "speedlimit": 15.0,
    }))

    monkeypatch.setattr(map_controller.time, "monotonic", lambda: 106.5)

    for _ in range(2):
      self.scc_m.update(True, False, 25.0, 0.0, 30.0)

    assert self.scc_m.state == VisionState.turning
    assert self.scc_m.output_v_target == 15.0

  def test_future_heartbeat_invalidates_advisory_params(self, monkeypatch):
    monkeypatch.setattr(map_controller.time, "monotonic", lambda: 106.5)
    self.mem_params.put(map_controller.MAP_ADVISORY_UPDATED_AT_PARAM, "107.0")

    assert not self.scc_m._advisory_params_valid()

  def test_stale_target_velocity_heartbeat_ignores_cached_map_target(self, monkeypatch):
    self.scc_m.v_ego = 25.0
    self.scc_m.a_ego = 0.0
    distance = self.scc_m._target_control_distance(20.0) - 1.0
    target_lon = distance / R * TO_DEGREES
    self.mem_params.put("MapTargetVelocities", json.dumps([
      {"latitude": 0.0, "longitude": target_lon, "velocity": 20.0},
    ]))
    self.mem_params.put(map_controller.MAP_ADVISORY_UPDATED_AT_PARAM, "105.0")
    self.mem_params.put(map_controller.MAP_TARGET_VELOCITIES_UPDATED_AT_PARAM, "100.0")
    monkeypatch.setattr(map_controller.time, "monotonic", lambda: 106.5)

    self.scc_m.update(True, False, 25.0, 0.0, 30.0)

    assert self.scc_m.state == VisionState.enabled
    assert self.scc_m.output_v_target == V_CRUISE_UNSET

  def test_stale_heartbeat_clears_current_stack_target(self, monkeypatch):
    current_map = SunnypilotCurrentSmartCruiseControlMap()
    current_map.mem_params = self.mem_params
    monkeypatch.setattr(map_controller.time, "monotonic", lambda: 106.5)
    self.mem_params.put(map_controller.MAP_TARGET_VELOCITIES_UPDATED_AT_PARAM, "105.0")
    self.mem_params.put("MapTargetVelocities", json.dumps([
      {"latitude": 0.0, "longitude": 0.001, "velocity": 15.0},
    ]))

    for _ in range(2):
      current_map.update(True, False, 25.0, 0.0, 30.0)

    self.mem_params.put(map_controller.MAP_TARGET_VELOCITIES_UPDATED_AT_PARAM, "100.0")
    current_map.update(True, False, 25.0, 0.0, 30.0)

    assert current_map.state == VisionState.enabled
    assert current_map.output_v_target == V_CRUISE_UNSET

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

  def test_current_advisory_speed_limit_does_not_hold_negative_accel(self):
    self.mem_params.put("MapAdvisorySpeedLimit", json.dumps({
      "start_latitude": 0.0,
      "start_longitude": 0.0,
      "end_latitude": 0.0,
      "end_longitude": 0.001,
      "speedlimit": 15.0,
    }))

    for _ in range(2):
      self.scc_m.update(True, False, 25.0, -0.65, 30.0)

    assert self.scc_m.state == VisionState.turning
    assert self.scc_m.output_v_target == 15.0
    assert self.scc_m.output_a_target == 0.0

  def test_current_advisory_speed_limit_ignores_large_slowdown_without_model_curve(self):
    self.mem_params.put("MapAdvisorySpeedLimit", json.dumps({
      "start_latitude": 0.0,
      "start_longitude": 0.0,
      "end_latitude": 0.0,
      "end_longitude": 0.001,
      "speedlimit": 15.0,
    }))
    model_msg = make_model_prediction(distance=20.0, yaw_rate=0.02)

    for _ in range(2):
      self.scc_m.update(True, False, 25.0, 0.0, 30.0, model_msg)

    assert self.scc_m.state == VisionState.enabled
    assert self.scc_m.output_v_target == V_CRUISE_UNSET

  def test_current_advisory_speed_limit_allows_model_confirmed_curve(self):
    self.mem_params.put("MapAdvisorySpeedLimit", json.dumps({
      "start_latitude": 0.0,
      "start_longitude": 0.0,
      "end_latitude": 0.0,
      "end_longitude": 0.001,
      "speedlimit": 15.0,
    }))
    model_msg = make_model_prediction(distance=20.0, yaw_rate=0.22)

    for _ in range(2):
      self.scc_m.update(True, False, 25.0, 0.0, 30.0, model_msg)

    assert self.scc_m.state == VisionState.turning
    assert self.scc_m.output_v_target == 15.0

  def test_current_advisory_uses_model_intermediate_when_full_map_target_unconfirmed(self):
    self.scc_m.v_ego = 25.0
    self.scc_m.a_ego = 0.0
    target_v = 13.0
    model_target_v = 18.0
    yaw_rate = self.scc_m.v_ego * map_controller.MODEL_CURVE_TARGET_LAT_ACCEL / model_target_v**2
    self.mem_params.put("MapAdvisorySpeedLimit", json.dumps({
      "start_latitude": 0.0,
      "start_longitude": 0.0,
      "end_latitude": 0.0,
      "end_longitude": 0.001,
      "speedlimit": target_v,
    }))
    model_msg = make_model_prediction(distance=0.0, yaw_rate=yaw_rate, speed=self.scc_m.v_ego)
    expected_target = self.scc_m._prediction_curve_target(model_msg, 0.0)

    assert expected_target is not None
    assert target_v < expected_target < self.scc_m.v_ego

    for _ in range(2):
      self.scc_m.update(True, False, self.scc_m.v_ego, 0.0, 30.0, model_msg)

    assert self.scc_m.state == VisionState.turning
    assert self.scc_m.output_v_target == pytest.approx(expected_target)

    weak_model_msg = make_model_prediction(distance=0.0, yaw_rate=0.02, speed=self.scc_m.v_ego)
    self.scc_m.update(True, False, self.scc_m.v_ego, 0.0, 30.0, weak_model_msg)

    assert self.scc_m.state == VisionState.enabled
    assert self.scc_m.output_v_target == V_CRUISE_UNSET

  def test_prediction_curve_target_matches_shared_lateral_accel_speed(self):
    target_v = 18.0
    yaw_rate = 25.0 * map_controller.MODEL_CURVE_TARGET_LAT_ACCEL / target_v**2
    model_msg = make_model_prediction(distance=0.0, yaw_rate=yaw_rate, speed=25.0)

    assert self.scc_m._prediction_curve_target(model_msg, 0.0) == pytest.approx(
      speed_for_lateral_accel(map_controller.MODEL_CURVE_TARGET_LAT_ACCEL, yaw_rate / 25.0)
    )

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

  def test_next_advisory_speed_limit_uses_map_when_model_horizon_does_not_cover_target(self):
    self.scc_m.v_ego = 25.0
    self.scc_m.a_ego = 0.0
    distance = self.scc_m._target_control_distance(15.0) - 1.0
    self.mem_params.put("NextMapAdvisorySpeedLimit", json.dumps({
      "start_latitude": 0.0,
      "start_longitude": 0.0,
      "end_latitude": 0.0,
      "end_longitude": 0.001,
      "speedlimit": 15.0,
      "distance": distance,
    }))
    model_msg = make_model_prediction(distance=5.0, yaw_rate=0.02)

    for _ in range(2):
      self.scc_m.update(True, False, 25.0, 0.0, 30.0, model_msg)

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

  def test_target_velocity_ignores_large_slowdown_without_model_curve(self):
    self.scc_m.v_ego = 25.0
    self.scc_m.a_ego = 0.0
    distance = self.scc_m._target_control_distance(15.0) - 1.0
    target_lon = distance / R * TO_DEGREES
    self.mem_params.put("MapTargetVelocities", json.dumps([
      {"latitude": 0.0, "longitude": target_lon, "velocity": 15.0},
    ]))
    model_msg = make_model_prediction(distance=distance, yaw_rate=0.02)

    for _ in range(2):
      self.scc_m.update(True, False, 25.0, 0.0, 30.0, model_msg)

    assert self.scc_m.state == VisionState.enabled
    assert self.scc_m.output_v_target == V_CRUISE_UNSET

  def test_target_velocity_ignores_relative_slowdown_without_model_curve(self):
    self.scc_m.v_ego = 16.0
    self.scc_m.a_ego = 0.0
    target_v = 12.24
    distance = self.scc_m._target_control_distance(target_v) - 1.0
    target_lon = distance / R * TO_DEGREES
    self.mem_params.put("MapTargetVelocities", json.dumps([
      {"latitude": 0.0, "longitude": target_lon, "velocity": target_v},
    ]))
    model_msg = make_model_prediction(distance=distance, yaw_rate=0.02, speed=self.scc_m.v_ego)

    for _ in range(2):
      self.scc_m.update(True, False, self.scc_m.v_ego, 0.0, 18.61, model_msg)

    assert self.scc_m.state == VisionState.enabled
    assert self.scc_m.output_v_target == V_CRUISE_UNSET

  def test_target_velocity_uses_model_intermediate_when_full_map_target_unconfirmed(self):
    self.scc_m.v_ego = 25.0
    self.scc_m.a_ego = 0.0
    target_v = 13.0
    model_target_v = 18.0
    yaw_rate = self.scc_m.v_ego * map_controller.MODEL_CURVE_TARGET_LAT_ACCEL / model_target_v**2
    distance = self.scc_m._target_control_distance(model_target_v) - 1.0
    target_lon = distance / R * TO_DEGREES
    self.mem_params.put("MapTargetVelocities", json.dumps([
      {"latitude": 0.0, "longitude": target_lon, "velocity": target_v},
    ]))
    model_msg = make_model_prediction(distance=distance, yaw_rate=yaw_rate, speed=self.scc_m.v_ego)
    expected_target = self.scc_m._prediction_curve_target(model_msg, distance)

    assert expected_target is not None
    assert target_v < expected_target < self.scc_m.v_ego
    assert expected_target > target_v + map_controller.MODEL_CURVE_OVERSLOWDOWN_MARGIN

    for _ in range(2):
      self.scc_m.update(True, False, self.scc_m.v_ego, 0.0, 30.0, model_msg)

    assert self.scc_m.state == VisionState.turning
    assert self.scc_m.output_v_target == pytest.approx(expected_target)

  def test_model_intermediate_target_releases_when_prediction_drops(self):
    self.scc_m.v_ego = 25.0
    self.scc_m.a_ego = 0.0
    target_v = 13.0
    model_target_v = 18.0
    yaw_rate = self.scc_m.v_ego * map_controller.MODEL_CURVE_TARGET_LAT_ACCEL / model_target_v**2
    distance = self.scc_m._target_control_distance(model_target_v) - 1.0
    target_lon = distance / R * TO_DEGREES
    self.mem_params.put("MapTargetVelocities", json.dumps([
      {"latitude": 0.0, "longitude": target_lon, "velocity": target_v},
    ]))
    model_msg = make_model_prediction(distance=distance, yaw_rate=yaw_rate, speed=self.scc_m.v_ego)

    for _ in range(2):
      self.scc_m.update(True, False, self.scc_m.v_ego, 0.0, 30.0, model_msg)
    assert self.scc_m.state == VisionState.turning
    assert target_v < self.scc_m.output_v_target < self.scc_m.v_ego

    weak_model_msg = make_model_prediction(distance=distance, yaw_rate=0.02, speed=self.scc_m.v_ego)
    self.scc_m.update(True, False, self.scc_m.v_ego, 0.0, 30.0, weak_model_msg)

    assert self.scc_m.state == VisionState.enabled
    assert self.scc_m.output_v_target == V_CRUISE_UNSET

  def test_full_model_confirmed_target_releases_when_prediction_drops(self):
    self.scc_m.v_ego = 25.0
    self.scc_m.a_ego = 0.0
    target_v = 13.0
    model_target_v = 14.0
    yaw_rate = self.scc_m.v_ego * map_controller.MODEL_CURVE_TARGET_LAT_ACCEL / model_target_v**2
    distance = self.scc_m._target_control_distance(target_v) - 1.0
    target_lon = distance / R * TO_DEGREES
    self.mem_params.put("MapTargetVelocities", json.dumps([
      {"latitude": 0.0, "longitude": target_lon, "velocity": target_v},
    ]))
    model_msg = make_model_prediction(distance=distance, yaw_rate=yaw_rate, speed=self.scc_m.v_ego)

    for _ in range(2):
      self.scc_m.update(True, False, self.scc_m.v_ego, 0.0, 30.0, model_msg)
    assert self.scc_m.state == VisionState.turning
    assert self.scc_m.output_v_target == target_v

    weak_model_msg = make_model_prediction(distance=distance, yaw_rate=0.02, speed=self.scc_m.v_ego)
    self.scc_m.update(True, False, self.scc_m.v_ego, 0.0, 30.0, weak_model_msg)

    assert self.scc_m.state == VisionState.enabled
    assert self.scc_m.output_v_target == V_CRUISE_UNSET

  def test_model_curve_prediction_can_advance_map_target(self):
    self.scc_m.v_ego = 25.0
    self.scc_m.a_ego = 0.0
    target_v = 20.0
    distance = self.scc_m._target_control_distance(target_v) + 5.0
    assert not self.scc_m._target_in_range(target_v, distance)

    target_lon = distance / R * TO_DEGREES
    self.mem_params.put("MapTargetVelocities", json.dumps([
      {"latitude": 0.0, "longitude": target_lon, "velocity": target_v},
    ]))
    model_msg = make_model_prediction(distance, yaw_rate=0.22)

    for _ in range(2):
      self.scc_m.update(True, False, 25.0, 0.0, 30.0, model_msg)

    assert self.scc_m.state == VisionState.turning
    assert self.scc_m.output_v_target == target_v

  def test_model_advanced_map_target_releases_when_prediction_drops(self):
    self.scc_m.v_ego = 25.0
    self.scc_m.a_ego = 0.0
    target_v = 20.0
    distance = self.scc_m._target_control_distance(target_v) + 5.0
    target_lon = distance / R * TO_DEGREES
    self.mem_params.put("MapTargetVelocities", json.dumps([
      {"latitude": 0.0, "longitude": target_lon, "velocity": target_v},
    ]))
    model_msg = make_model_prediction(distance, yaw_rate=0.22)

    for _ in range(2):
      self.scc_m.update(True, False, 25.0, 0.0, 30.0, model_msg)
    assert self.scc_m.state == VisionState.turning

    self.scc_m.update(True, False, 25.0, 0.0, 30.0)

    assert self.scc_m.state == VisionState.enabled
    assert self.scc_m.output_v_target == V_CRUISE_UNSET

  def test_model_curve_prediction_does_not_relax_map_target(self):
    model_msg = make_model_prediction(distance=50.0, yaw_rate=0.02)

    assert self.scc_m._prediction_control_target(15.0, 50.0, model_msg) == 15.0


class TestSunnypilotCurrentSmartCruiseControlMap:
  def setup_method(self):
    self.params = Params()
    self.mem_params = MockParams()
    self.params.put_bool("SmartCruiseControlMap", True)
    self.mem_params.put("LastGPSPosition", json.dumps({"latitude": 0.0, "longitude": 0.0}))
    self.mem_params.put("MapTargetVelocities", "[]")
    self.mem_params.put("MapAdvisorySpeedLimit", "{}")
    self.mem_params.put("MapAdvisoryLimit", "{}")
    self.scc_m = SunnypilotCurrentSmartCruiseControlMap()
    self.scc_m.mem_params = self.mem_params

  def test_current_map_uses_master_json_parser(self):
    self.mem_params.put("MapTargetVelocities", "not-json")

    with pytest.raises(json.JSONDecodeError):
      sunnypilot_current_velocities_from_param("MapTargetVelocities", self.mem_params)

  def test_current_map_holds_ego_accel_for_active_target(self):
    self.mem_params.put("MapTargetVelocities", json.dumps([
      {"latitude": 0.0, "longitude": 0.0, "velocity": 15.0},
    ]))

    for _ in range(2):
      self.scc_m.update(True, False, 25.0, -0.65, 30.0)

    assert self.scc_m.state == VisionState.turning
    assert self.scc_m.output_v_target == pytest.approx(15.0)
    assert self.scc_m.output_a_target == pytest.approx(-0.65)

  def test_current_map_ignores_osm_advisory_speed_params(self):
    self.mem_params.put("MapAdvisoryLimit", json.dumps({"speedlimit": 15.0, "distance": 0.0}))

    for _ in range(2):
      self.scc_m.update(True, False, 25.0, 0.0, 30.0)

    assert self.scc_m.state == VisionState.enabled
    assert self.scc_m.output_v_target == V_CRUISE_UNSET
