"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import json
import platform

import numpy as np
import pytest

import cereal.messaging as messaging
from cereal import custom, log
from openpilot.common.params import Params
from openpilot.selfdrive.car.cruise import V_CRUISE_UNSET
from openpilot.selfdrive.modeld.constants import ModelConstants

# Canonical (v2) package modules
from openpilot.sunnypilot.custom.longitudinal.curve_evidence import constants as v2_constants
from openpilot.sunnypilot.custom.longitudinal.curve_evidence import vision_controller as v2_vision
from openpilot.sunnypilot.custom.longitudinal.curve_evidence import map_controller as v2_map
from openpilot.sunnypilot.custom.longitudinal.curve_evidence import smart_cruise_control as v2_scc

# Legacy v1 backup modules
from openpilot.sunnypilot.selfdrive.controls.lib.smart_cruise_control import constants_v1 as v1_constants
from openpilot.sunnypilot.selfdrive.controls.lib.smart_cruise_control import vision_controller_v1 as v1_vision
from openpilot.sunnypilot.selfdrive.controls.lib.smart_cruise_control import map_controller_v1 as v1_map
from openpilot.sunnypilot.selfdrive.controls.lib.smart_cruise_control import smart_cruise_control_v1 as v1_scc

# Legacy facades (import sanity)
from openpilot.sunnypilot.selfdrive.controls.lib.smart_cruise_control import MIN_V as facade_MIN_V
from openpilot.sunnypilot.selfdrive.controls.lib.smart_cruise_control.vision_controller import (
  SmartCruiseControlVision as FacadeVision,
)
from openpilot.sunnypilot.selfdrive.controls.lib.smart_cruise_control.map_controller import (
  SmartCruiseControlMap as FacadeMap,
  distance_to_point as facade_distance_to_point,
)
from openpilot.sunnypilot.selfdrive.controls.lib.smart_cruise_control.smart_cruise_control import (
  SmartCruiseControl as FacadeSCC,
)

VisionState = custom.LongitudinalPlanSP.SmartCruiseControl.VisionState
MapState = custom.LongitudinalPlanSP.SmartCruiseControl.MapState


VISION_CONSTANTS = [
  "_ENTERING_PRED_LAT_ACC_TH",
  "_ABORT_ENTERING_PRED_LAT_ACC_TH",
  "_TURNING_LAT_ACC_TH",
  "_CURRENT_LAT_ACC_BLEED_TH",
  "_LEAVING_LAT_ACC_TH",
  "_FINISH_LAT_ACC_TH",
  "_A_LAT_REG_MAX",
  "_NO_OVERSHOOT_TIME_HORIZON",
  "_ENTERING_SMOOTH_DECEL_V",
  "_ENTERING_SMOOTH_DECEL_BP",
  "_PRE_ENTRY_PRED_LAT_ACC_TH",
  "_PRE_ENTRY_MIN_FRAMES",
  "_PRE_ENTRY_GENTLE_DECEL",
  "_TURNING_ACC_V",
  "_TURNING_ACC_BP",
  "_LEAVING_ACC",
  "_EPS",
]

MAP_CONSTANTS = [
  "R",
  "TO_RADIANS",
  "TO_DEGREES",
  "TARGET_JERK",
  "TARGET_ACCEL",
  "TARGET_OFFSET",
  "MAX_ROUTE_TARGET_DISTANCE",
  "MAX_SHORT_DROP_DISTANCE",
  "MAX_SHORT_DROP_DELTA_V",
  "SHORT_DROP_CONFIRM_POINTS",
  "MATERIAL_DROP_DELTA_V",
  "MIN_CURRENT_LAT_ACCEL_CORROBORATION",
  "MIN_MODEL_PRED_LAT_ACCEL_CORROBORATION",
  "_ROUTE_POINT_EPS",
]


def _generate_modelV2():
  model = messaging.new_message('modelV2')
  position = log.XYZTData.new_message()
  speed = 30
  position.x = [float(x) for x in (speed + 0.5) * np.array(ModelConstants.T_IDXS)]
  model.modelV2.position = position
  orientation = log.XYZTData.new_message()
  curvature = 0.05
  orientation.x = [float(curvature) for _ in ModelConstants.T_IDXS]
  orientation.y = [0.0 for _ in ModelConstants.T_IDXS]
  model.modelV2.orientation = orientation
  orientationRate = log.XYZTData.new_message()
  orientationRate.z = [float(z) for z in ModelConstants.T_IDXS]
  model.modelV2.orientationRate = orientationRate
  velocity = log.XYZTData.new_message()
  velocity.x = [float(x) for x in (speed + 0.5) * np.ones_like(ModelConstants.T_IDXS)]
  velocity.x[0] = float(speed)
  model.modelV2.velocity = velocity
  acceleration = log.XYZTData.new_message()
  acceleration.x = [float(x) for x in np.zeros_like(ModelConstants.T_IDXS)]
  acceleration.y = [float(y) for y in np.zeros_like(ModelConstants.T_IDXS)]
  model.modelV2.acceleration = acceleration
  return model


def _set_modelV2_lat_acc(mdl, lat_acc_arr, v_model: float):
  n = len(ModelConstants.T_IDXS)
  lat_acc_arr = np.asarray(lat_acc_arr, dtype=np.float64)
  if lat_acc_arr.shape != (n,):
    lat_acc_arr = np.full(n, lat_acc_arr, dtype=np.float64)
  yaw_rate = lat_acc_arr / max(v_model, 0.1)
  mdl.modelV2.velocity.x = [float(v_model)] * n
  mdl.modelV2.orientationRate.z = [float(z) for z in yaw_rate]
  return mdl


def _generate_controls_state(curvature=0.001):
  controls_state = messaging.new_message('controlsState')
  controls_state.controlsState.curvature = float(curvature)
  return controls_state


def _make_sm(curvature=0.001):
  mdl = _generate_modelV2()
  cs = messaging.new_message('carState')
  cs.carState.vEgo = 30.0
  cs.carState.standstill = False
  cs.carState.vCruise = 50.0 * 3.6
  controls_state = _generate_controls_state(curvature)
  return {
    'modelV2': mdl.modelV2,
    'carState': cs.carState,
    'controlsState': controls_state.controlsState,
  }


class _FakeSM:
  def __init__(self, curvature=None, pred_lat_accels=None):
    class _Vec:
      def __init__(self, values):
        self.z = values
        self.x = values

    class _ControlsState:
      def __init__(self, curvature):
        self.curvature = curvature

    class _ModelV2:
      def __init__(self, pred_lat_accels):
        self.orientationRate = _Vec(pred_lat_accels)
        self.velocity = _Vec([1.0 for _ in pred_lat_accels])

    self._items = {
      'controlsState': _ControlsState(curvature),
      'modelV2': _ModelV2(pred_lat_accels),
    }

  def __getitem__(self, key):
    return self._items[key]


@pytest.fixture
def params():
  return Params()


@pytest.fixture
def mem_params():
  return Params("/dev/shm/params") if platform.system() != "Darwin" else Params()


class TestConstantsAndHelpersParity:

  def test_min_v_parity(self):
    assert v1_constants.MIN_V == v2_constants.MIN_V
    assert v1_constants.MIN_V == facade_MIN_V

  @pytest.mark.parametrize("name", VISION_CONSTANTS)
  def test_vision_constants_parity(self, name):
    assert getattr(v1_vision, name) == getattr(v2_vision, name)

  @pytest.mark.parametrize("name", MAP_CONSTANTS)
  def test_map_constants_parity(self, name):
    assert getattr(v1_map, name) == getattr(v2_map, name)

  def test_state_constants_parity(self):
    assert v1_vision.ACTIVE_STATES == v2_vision.ACTIVE_STATES
    assert v1_vision.ENABLED_STATES == v2_vision.ENABLED_STATES
    assert v1_map.ACTIVE_STATES == v2_map.ACTIVE_STATES
    assert v1_map.ENABLED_STATES == v2_map.ENABLED_STATES
    assert v1_vision.VisionState == v2_vision.VisionState
    assert v1_map.MapState == v2_map.MapState

  def test_velocities_from_param_parity(self, mem_params):
    mem_params.put("MapTargetVelocities", json.dumps([{"latitude": 1.0, "longitude": 2.0, "velocity": 10.0}]), block=True)
    assert v1_map.velocities_from_param("MapTargetVelocities", mem_params) == v2_map.velocities_from_param("MapTargetVelocities", mem_params)

  def test_dedupe_route_points_parity(self):
    points = [
      {"latitude": 1.0, "longitude": 2.0, "velocity": 10.0},
      {"latitude": 1.0, "longitude": 2.0, "velocity": 12.0},
      {"latitude": 1.0000001, "longitude": 2.0000001, "velocity": 8.0},
    ]
    assert v1_map._dedupe_route_points(points) == v2_map._dedupe_route_points(points)

  def test_distance_to_point_parity(self):
    a = v1_map.distance_to_point(0.0, 0.0, 0.001, 0.001)
    b = v2_map.distance_to_point(0.0, 0.0, 0.001, 0.001)
    assert a == pytest.approx(b)
    assert a == pytest.approx(facade_distance_to_point(0.0, 0.0, 0.001, 0.001))

  def test_kinematic_helpers_parity(self):
    assert v1_map.calculate_accel(1.0, -0.6, 0.0) == v2_map.calculate_accel(1.0, -0.6, 0.0)
    assert v1_map.calculate_velocity(1.0, -0.6, 0.0, 20.0) == v2_map.calculate_velocity(1.0, -0.6, 0.0, 20.0)
    assert v1_map.calculate_distance(1.0, -0.6, 0.0, 20.0) == v2_map.calculate_distance(1.0, -0.6, 0.0, 20.0)


class TestWrapperContractAndOrder:

  def test_wrapper_has_vision_and_map(self):
    scc_v1 = v1_scc.SmartCruiseControl()
    scc_v2 = v2_scc.SmartCruiseControl()
    facade = FacadeSCC()

    assert isinstance(scc_v1.vision, v1_vision.SmartCruiseControlVision)
    assert isinstance(scc_v1.map, v1_map.SmartCruiseControlMap)
    assert isinstance(scc_v2.vision, v2_vision.SmartCruiseControlVision)
    assert isinstance(scc_v2.map, v2_map.SmartCruiseControlMap)
    assert isinstance(facade.vision, FacadeVision)
    assert isinstance(facade.map, FacadeMap)

  def test_update_order_map_then_vision(self):
    scc_v2 = v2_scc.SmartCruiseControl()
    order = []
    scc_v2.map.update = lambda *args, **kwargs: order.append("map")
    scc_v2.vision.update = lambda *args, **kwargs: order.append("vision")

    sm = _make_sm()
    scc_v2.update(sm, True, False, 20.0, 0.0, 30.0)
    assert order == ["map", "vision"]

  def test_wrapper_equivalent_snapshot(self, params, mem_params):
    params.put_bool("SmartCruiseControlVision", True, block=True)
    params.put_bool("SmartCruiseControlMap", True, block=True)
    mem_params.put("LastGPSPosition", "{}", block=True)
    mem_params.put("MapTargetVelocities", "{}", block=True)

    scc_v1 = v1_scc.SmartCruiseControl()
    scc_v2 = v2_scc.SmartCruiseControl()

    sm = _make_sm(curvature=0.001)
    for _ in range(10):
      scc_v1.update(sm, True, False, 20.0, 0.0, 30.0)
      scc_v2.update(sm, True, False, 20.0, 0.0, 30.0)

    self._assert_scc_snapshots_equal(scc_v1, scc_v2)

  def _assert_scc_snapshots_equal(self, scc_v1, scc_v2):
    assert scc_v1.vision.state == scc_v2.vision.state
    assert scc_v1.vision.is_enabled == scc_v2.vision.is_enabled
    assert scc_v1.vision.is_active == scc_v2.vision.is_active
    assert scc_v1.vision.output_v_target == pytest.approx(scc_v2.vision.output_v_target)
    assert scc_v1.vision.output_a_target == pytest.approx(scc_v2.vision.output_a_target)
    assert scc_v1.map.state == scc_v2.map.state
    assert scc_v1.map.is_enabled == scc_v2.map.is_enabled
    assert scc_v1.map.is_active == scc_v2.map.is_active
    assert scc_v1.map.output_v_target == scc_v2.map.output_v_target
    assert scc_v1.map.output_a_target == scc_v2.map.output_a_target


class TestVisionSequencesParity:

  def setup_method(self):
    self.params = Params()
    self.params.put_bool("SmartCruiseControlVision", True, block=True)

  def _make_controllers(self):
    v1 = v1_vision.SmartCruiseControlVision()
    v2 = v2_vision.SmartCruiseControlVision()
    v1.enabled = True
    v2.enabled = True
    return v1, v2

  def _vision_snapshot(self, c):
    return {
      'state': c.state,
      'is_enabled': c.is_enabled,
      'is_active': c.is_active,
      'output_v_target': c.output_v_target,
      'output_a_target': c.output_a_target,
      'max_pred_lat_acc': c.max_pred_lat_acc,
      'current_lat_acc': c.current_lat_acc,
      'v_target': c.v_target,
      'a_target': c.a_target,
      'pre_entry_active': c.pre_entry_active,
    }

  def _assert_snapshots_equal(self, c1, c2):
    s1 = self._vision_snapshot(c1)
    s2 = self._vision_snapshot(c2)
    assert s1['state'] == s2['state']
    assert s1['is_enabled'] == s2['is_enabled']
    assert s1['is_active'] == s2['is_active']
    assert s1['output_v_target'] == pytest.approx(s2['output_v_target'], abs=1e-6)
    assert s1['output_a_target'] == pytest.approx(s2['output_a_target'], abs=1e-6)
    assert s1['max_pred_lat_acc'] == pytest.approx(s2['max_pred_lat_acc'], abs=1e-6)
    assert s1['current_lat_acc'] == pytest.approx(s2['current_lat_acc'], abs=1e-6)
    assert s1['v_target'] == pytest.approx(s2['v_target'], abs=1e-6)
    assert s1['a_target'] == pytest.approx(s2['a_target'], abs=1e-6)
    assert s1['pre_entry_active'] == s2['pre_entry_active']

  def test_disabled_sequence(self):
    v1, v2 = self._make_controllers()
    sm = _make_sm()
    for _ in range(5):
      v1.update(sm, False, False, 20.0, 0.0, 0.0)
      v2.update(sm, False, False, 20.0, 0.0, 0.0)
      self._assert_snapshots_equal(v1, v2)

  def test_pre_entry_sequence(self):
    v1, v2 = self._make_controllers()
    n = len(ModelConstants.T_IDXS)
    pred_lat_accels = np.full(n, np.float32(1.05), dtype=np.float32)
    v_ego = float(v1_vision.MIN_V + 5.0)
    mdl = _generate_modelV2()
    mdl = _set_modelV2_lat_acc(mdl, pred_lat_accels, v_ego)
    sm = _make_sm()
    sm["modelV2"] = mdl.modelV2

    for _ in range(5):
      v1.update(sm, True, False, v_ego, 0.0, 0.0)
      v2.update(sm, True, False, v_ego, 0.0, 0.0)
      self._assert_snapshots_equal(v1, v2)

    assert v1.state == VisionState.enabled
    assert v1.is_active
    assert v1.output_a_target == pytest.approx(v1_vision._PRE_ENTRY_GENTLE_DECEL, abs=1e-6)

  def test_entering_sequence(self):
    v1, v2 = self._make_controllers()
    n = len(ModelConstants.T_IDXS)
    pred_lat_accels = np.full(n, np.float32(2.0), dtype=np.float32)
    v_ego = 30.0
    mdl = _generate_modelV2()
    mdl = _set_modelV2_lat_acc(mdl, pred_lat_accels, v_ego)
    sm = _make_sm()
    sm["modelV2"] = mdl.modelV2

    for _ in range(5):
      v1.update(sm, True, False, v_ego, 0.0, 0.0)
      v2.update(sm, True, False, v_ego, 0.0, 0.0)
      self._assert_snapshots_equal(v1, v2)

    assert v1.state == VisionState.entering
    assert v1.is_active

  def test_override_sequence(self):
    v1, v2 = self._make_controllers()
    sm = _make_sm()
    for _ in range(3):
      v1.update(sm, True, False, 20.0, 0.0, 0.0)
      v2.update(sm, True, False, 20.0, 0.0, 0.0)
    v1.update(sm, True, True, 20.0, 0.0, 0.0)
    v2.update(sm, True, True, 20.0, 0.0, 0.0)
    self._assert_snapshots_equal(v1, v2)
    assert v1.state == VisionState.overriding
    assert not v1.is_active

  def test_feature_disabled_sequence(self):
    self.params.put_bool("SmartCruiseControlVision", False, block=True)
    v1, v2 = self._make_controllers()
    v1.enabled = False
    v2.enabled = False
    sm = _make_sm()
    for _ in range(5):
      v1.update(sm, True, False, 20.0, 0.0, 0.0)
      v2.update(sm, True, False, 20.0, 0.0, 0.0)
      self._assert_snapshots_equal(v1, v2)


class TestMapSequencesParity:

  def setup_method(self):
    self.params = Params()
    self.params.put_bool("SmartCruiseControlMap", True, block=True)
    self.mem_params = Params("/dev/shm/params") if platform.system() != "Darwin" else self.params
    self.mem_params.put("LastGPSPosition", "{}", block=True)
    self.mem_params.put("MapTargetVelocities", "{}", block=True)

  def _make_controllers(self):
    v1 = v1_map.SmartCruiseControlMap()
    v2 = v2_map.SmartCruiseControlMap()
    v1.enabled = True
    v2.enabled = True
    return v1, v2

  def _set_gps(self, lat, lon):
    self.mem_params.put("LastGPSPosition", f'{{"latitude": {lat}, "longitude": {lon}}}', block=True)

  def _set_targets(self, targets):
    self.mem_params.put("MapTargetVelocities", json.dumps(targets), block=True)

  def _map_snapshot(self, c):
    return {
      'state': c.state,
      'is_enabled': c.is_enabled,
      'is_active': c.is_active,
      'output_v_target': c.output_v_target,
      'output_a_target': c.output_a_target,
      'v_target': c.v_target,
      'target_lat': c.target_lat,
      'target_lon': c.target_lon,
    }

  def _assert_snapshots_equal(self, c1, c2):
    s1 = self._map_snapshot(c1)
    s2 = self._map_snapshot(c2)
    assert s1['state'] == s2['state']
    assert s1['is_enabled'] == s2['is_enabled']
    assert s1['is_active'] == s2['is_active']
    assert s1['output_v_target'] == s2['output_v_target']
    assert s1['output_a_target'] == s2['output_a_target']
    assert s1['v_target'] == pytest.approx(s2['v_target'], abs=1e-6)
    assert s1['target_lat'] == pytest.approx(s2['target_lat'], abs=1e-8)
    assert s1['target_lon'] == pytest.approx(s2['target_lon'], abs=1e-8)

  def test_disabled_sequence(self):
    v1, v2 = self._make_controllers()
    for _ in range(5):
      v1.update(False, False, 20.0, 0.0, 30.0)
      v2.update(False, False, 20.0, 0.0, 30.0)
      self._assert_snapshots_equal(v1, v2)

  def test_reasonable_nearby_target_sequence(self):
    v1, v2 = self._make_controllers()
    self._set_gps(37.00008, -122.00008)
    self._set_targets([
      {"latitude": 37.00000, "longitude": -122.00008, "velocity": 24.0},
      {"latitude": 37.00004, "longitude": -122.00004, "velocity": 19.0},
      {"latitude": 37.00008, "longitude": -122.00008, "velocity": 18.5},
      {"latitude": 37.00012, "longitude": -122.00012, "velocity": 18.0},
    ])
    sm = _FakeSM(curvature=0.0018, pred_lat_accels=[0.18, 0.18, 0.18, 0.18])

    for _ in range(3):
      v1.update(True, False, 23.8, 0.0, 30.0, sm)
      v2.update(True, False, 23.8, 0.0, 30.0, sm)
      self._assert_snapshots_equal(v1, v2)

    assert v1.v_target == pytest.approx(18.5, abs=1e-6)
    assert v1.is_active

  def test_evidence_only_output_contract(self):
    v1, v2 = self._make_controllers()
    self._set_gps(37.00008, -122.00008)
    self._set_targets([
      {"latitude": 37.00000, "longitude": -122.00008, "velocity": 24.0},
      {"latitude": 37.00004, "longitude": -122.00004, "velocity": 19.0},
      {"latitude": 37.00008, "longitude": -122.00008, "velocity": 18.5},
      {"latitude": 37.00012, "longitude": -122.00012, "velocity": 18.0},
    ])
    sm = _FakeSM(curvature=0.0018, pred_lat_accels=[0.18, 0.18, 0.18, 0.18])

    for _ in range(3):
      v1.update(True, False, 23.8, 0.0, 30.0, sm)
      v2.update(True, False, 23.8, 0.0, 30.0, sm)

    for c in (v1, v2):
      assert c.is_active
      assert c.output_v_target == V_CRUISE_UNSET
      assert c.output_a_target == 0.0

    facade = FacadeMap()
    facade.enabled = True
    facade.update(True, False, 23.8, 0.0, 30.0, sm)
    assert facade.output_v_target == V_CRUISE_UNSET
    assert facade.output_a_target == 0.0

  def test_material_slowdown_rejected_without_corroboration(self):
    v1, v2 = self._make_controllers()
    self._set_gps(37.00008, -122.00008)
    self._set_targets([
      {"latitude": 37.00000, "longitude": -122.00008, "velocity": 17.0},
      {"latitude": 37.00004, "longitude": -122.00004, "velocity": 14.3},
      {"latitude": 37.00008, "longitude": -122.00008, "velocity": 12.2},
      {"latitude": 37.00012, "longitude": -122.00012, "velocity": 12.0},
    ])
    sm = _FakeSM(curvature=0.0006, pred_lat_accels=[0.18, 0.18, 0.18, 0.18])

    for _ in range(3):
      v1.update(True, False, 17.1, 0.0, 30.0, sm)
      v2.update(True, False, 17.1, 0.0, 30.0, sm)
      self._assert_snapshots_equal(v1, v2)

    assert v1.v_target == 0.0
    assert not v1.is_active

  def test_material_slowdown_accepted_with_corroboration(self):
    v1, v2 = self._make_controllers()
    self._set_gps(37.00008, -122.00008)
    self._set_targets([
      {"latitude": 37.00000, "longitude": -122.00008, "velocity": 17.0},
      {"latitude": 37.00004, "longitude": -122.00004, "velocity": 14.3},
      {"latitude": 37.00008, "longitude": -122.00008, "velocity": 12.2},
      {"latitude": 37.00012, "longitude": -122.00012, "velocity": 12.0},
    ])
    sm = _FakeSM(curvature=0.0018, pred_lat_accels=[0.18, 0.18, 0.18, 0.18])

    for _ in range(3):
      v1.update(True, False, 17.1, 0.0, 30.0, sm)
      v2.update(True, False, 17.1, 0.0, 30.0, sm)
      self._assert_snapshots_equal(v1, v2)

    assert v1.v_target == pytest.approx(12.2, abs=1e-6)
