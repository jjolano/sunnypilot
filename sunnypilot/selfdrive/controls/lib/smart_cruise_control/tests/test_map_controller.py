"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import platform

from cereal import custom
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.car.cruise import V_CRUISE_UNSET
from openpilot.sunnypilot.selfdrive.controls.lib.smart_cruise_control.map_controller import SmartCruiseControlMap

MapState = VisionState = custom.LongitudinalPlanSP.SmartCruiseControl.MapState


class TestSmartCruiseControlMap:

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

  def setup_method(self):
    self.params = Params()
    self.mem_params = Params("/dev/shm/params") if platform.system() != "Darwin" else self.params
    self.reset_params()
    self.scc_m = SmartCruiseControlMap()

  def reset_params(self):
    self.params.put_bool("SmartCruiseControlMap", True, block=True)
    self.params.put("LastGPSPosition", "{}", block=True)
    self.params.put("MapTargetVelocities", "{}", block=True)

  def set_gps(self, lat, lon):
    self.mem_params.put("LastGPSPosition", f'{{"latitude": {lat}, "longitude": {lon}}}', block=True)

  def set_targets(self, targets):
    import json
    self.mem_params.put("MapTargetVelocities", json.dumps(targets), block=True)

  def test_initial_state(self):
    assert self.scc_m.state == VisionState.disabled
    assert not self.scc_m.is_active
    assert self.scc_m.output_v_target == V_CRUISE_UNSET
    assert self.scc_m.output_a_target == 0.

  def test_system_disabled(self):
    self.params.put_bool("SmartCruiseControlMap", False, block=True)
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

  def test_malformed_targets_fail_closed(self):
    self.set_gps(37.0, -122.0)
    self.mem_params.put("MapTargetVelocities", '{"bad": true}', block=True)
    self.scc_m.update(True, False, 10., 0., 20.)
    assert self.scc_m.v_target == 0.0
    assert self.scc_m.target_lat == 0.0
    assert self.scc_m.target_lon == 0.0
    assert not self.scc_m.is_active

    self.mem_params.put("MapTargetVelocities", '{bad', block=True)
    self.scc_m.update(True, False, 10., 0., 20.)
    assert self.scc_m.v_target == 0.0

  def test_malformed_gps_before_constructor_fails_closed(self):
    self.mem_params.put("LastGPSPosition", '{bad', block=True)
    self.set_targets([{"latitude": 37.0, "longitude": -122.0, "velocity": 10.0}])

    controller = SmartCruiseControlMap()
    controller.update(True, False, 10., 0., 20.)

    assert controller.v_target == 0.0
    assert not controller.is_active

  def test_invalid_or_far_position_fail_closed(self):
    self.mem_params.put("LastGPSPosition", '{"latitude": "nan", "longitude": -122.0}', block=True)
    self.set_targets([{"latitude": 37.0, "longitude": -122.0, "velocity": 10.0}])
    self.scc_m.update(True, False, 10., 0., 20.)
    assert self.scc_m.v_target == 0.0

    self.set_gps(0.0, 0.0)
    self.set_targets([{"latitude": 37.0, "longitude": -122.0, "velocity": 10.0}])
    self.scc_m.update(True, False, 10., 0., 20.)
    assert self.scc_m.v_target == 0.0

  def test_abrupt_short_distance_drop_rejected(self):
    self.set_gps(37.0, -122.0)
    self.set_targets([
      {"latitude": 37.00015, "longitude": -122.00015, "velocity": 12.3},
    ])
    self.scc_m.update(True, False, 23.8, 0., 30.)
    assert self.scc_m.v_target == 0.0
    assert not self.scc_m.is_active

    # Live-like short-range drop: ~24 m/s ego to ~16 m/s target at ~14 m should fail closed.
    self.set_targets([
      {"latitude": 37.00009, "longitude": -122.00009, "velocity": 16.0},
    ])
    self.scc_m.update(True, False, 23.8, 0., 30.)
    assert self.scc_m.v_target == 0.0

  def test_jerk_only_time_uses_correct_quadratic_roots(self):
    self.set_gps(37.00010, -122.0)
    self.set_targets([
      {"latitude": 37.00000, "longitude": -122.0, "velocity": 24.0},
      {"latitude": 37.00010, "longitude": -122.0, "velocity": 19.5},
      {"latitude": 37.00020, "longitude": -122.0, "velocity": 19.0},
    ])

    self.scc_m.update(True, False, 20.0, 0.0, 30.0)
    self.scc_m.update(True, False, 20.0, 0.0, 30.0)
    assert self.scc_m.v_target == 19.5

  def test_duplicate_same_location_keeps_faster_target(self):
    self.set_gps(37.00008, -122.00008)
    self.set_targets([
      {"latitude": 37.00000, "longitude": -122.00008, "velocity": 24.0},
      {"latitude": 37.00008, "longitude": -122.00008, "velocity": 18.0},
      {"latitude": 37.00008, "longitude": -122.00008, "velocity": 12.0},
      {"latitude": 37.00011, "longitude": -122.00011, "velocity": 11.5},
      {"latitude": 37.00014, "longitude": -122.00014, "velocity": 11.0},
    ])

    sm = self._FakeSM(curvature=0.0018, pred_lat_accels=[0.18, 0.18, 0.18, 0.18])
    self.scc_m.update(True, False, 22.0, 0.0, 30.0, sm)
    assert self.scc_m.v_target == 18.0

  def test_out_of_order_route_point_fails_closed(self):
    self.set_gps(37.0, -122.0)
    self.set_targets([
      {"latitude": 37.00012, "longitude": -122.00012, "velocity": 18.0},
      {"latitude": 37.00005, "longitude": -122.00005, "velocity": 10.0},
    ])

    self.scc_m.update(True, False, 22.0, 0.0, 30.0)
    assert self.scc_m.v_target == 0.0

  def test_terminal_stale_selected_point_clears(self):
    self.set_gps(37.0, -122.0)
    self.set_targets([
      {"latitude": 37.00008, "longitude": -122.0, "velocity": 19.0},
      {"latitude": 37.00012, "longitude": -122.0, "velocity": 18.6},
      {"latitude": 37.00016, "longitude": -122.0, "velocity": 18.4},
    ])

    self.scc_m.update(True, False, 20.0, 0.0, 30.0)
    assert self.scc_m.v_target == 0.0

    self.set_gps(37.00010, -122.0)
    self.set_targets([
      {"latitude": 37.00008, "longitude": -122.0, "velocity": 19.0},
      {"latitude": 37.00012, "longitude": -122.0, "velocity": 18.5},
    ])

    self.scc_m.update(True, False, 20.0, 0.0, 30.0)
    assert self.scc_m.v_target == 0.0
    assert not self.scc_m.is_active

  def test_clustered_short_drop_is_accepted(self):
    self.set_gps(37.00008, -122.00008)
    self.set_targets([
      {"latitude": 37.00000, "longitude": -122.00008, "velocity": 24.0},
      {"latitude": 37.00005, "longitude": -122.00005, "velocity": 18.5},
      {"latitude": 37.00008, "longitude": -122.00008, "velocity": 18.0},
      {"latitude": 37.00011, "longitude": -122.00011, "velocity": 17.5},
      {"latitude": 37.00014, "longitude": -122.00014, "velocity": 17.0},
    ])

    sm = self._FakeSM(curvature=0.0018, pred_lat_accels=[0.18, 0.18, 0.18, 0.18])
    self.scc_m.update(True, False, 24.0, 0.0, 30.0, sm)
    self.scc_m.update(True, False, 24.0, 0.0, 30.0, sm)
    assert self.scc_m.v_target > 0.0

  def test_reasonable_nearby_lower_target_accepted(self):
    self.set_gps(37.00008, -122.00008)
    self.set_targets([
      {"latitude": 37.00000, "longitude": -122.00008, "velocity": 24.0},
      {"latitude": 37.00004, "longitude": -122.00004, "velocity": 19.0},
      {"latitude": 37.00008, "longitude": -122.00008, "velocity": 18.5},
      {"latitude": 37.00012, "longitude": -122.00012, "velocity": 18.0},
    ])
    sm = self._FakeSM(curvature=0.0018, pred_lat_accels=[0.18, 0.18, 0.18, 0.18])
    self.scc_m.update(True, False, 23.8, 0., 30., sm)
    self.scc_m.update(True, False, 23.8, 0., 30., sm)
    assert self.scc_m.v_target == 18.5
    assert self.scc_m.is_active

  def test_stale_terminal_single_point_rejected(self):
    self.set_gps(37.00010, -122.0)
    self.set_targets([
      {"latitude": 37.00008, "longitude": -122.0, "velocity": 19.0},
    ])

    self.scc_m.update(True, False, 23.8, 0., 30.)
    assert self.scc_m.v_target == 0.0
    assert not self.scc_m.is_active

  def test_three_point_behind_slice_clears(self):
    self.set_gps(37.00010, -122.0)
    self.set_targets([
      {"latitude": 37.00008, "longitude": -122.0, "velocity": 19.0},
      {"latitude": 37.00006, "longitude": -122.0, "velocity": 18.5},
      {"latitude": 37.00004, "longitude": -122.0, "velocity": 18.0},
    ])

    self.scc_m.update(True, False, 23.8, 0.0, 30.0)
    assert self.scc_m.v_target == 0.0
    assert not self.scc_m.is_active

  def test_nearer_valid_target_wins_over_farther_global_minimum(self):
    self.set_gps(37.00003, -122.00003)
    self.set_targets([
      {"latitude": 37.00000, "longitude": -122.00003, "velocity": 24.0},
      {"latitude": 37.00003, "longitude": -122.00003, "velocity": 18.0},
      {"latitude": 37.00018, "longitude": -122.00018, "velocity": 8.0},
      {"latitude": 37.00022, "longitude": -122.00022, "velocity": 7.5},
    ])
    sm = self._FakeSM(curvature=0.0018, pred_lat_accels=[0.18, 0.18, 0.18, 0.18])
    self.scc_m.update(True, False, 23.8, 0., 30., sm)
    assert self.scc_m.v_target == 18.0

  def test_material_slowdown_rejected_on_straight_segment_without_corroboration(self):
    self.set_gps(37.00008, -122.00008)
    self.set_targets([
      {"latitude": 37.00000, "longitude": -122.00008, "velocity": 17.0},
      {"latitude": 37.00004, "longitude": -122.00004, "velocity": 14.3},
      {"latitude": 37.00008, "longitude": -122.00008, "velocity": 12.2},
      {"latitude": 37.00012, "longitude": -122.00012, "velocity": 12.0},
    ])

    sm = self._FakeSM(curvature=0.0006, pred_lat_accels=[0.18, 0.18, 0.18, 0.18])
    self.scc_m.update(True, False, 17.1, 0.0, 30.0, sm)
    self.scc_m.update(True, False, 17.1, 0.0, 30.0, sm)
    assert self.scc_m.v_target == 0.0

  def test_material_slowdown_above_old_target_cutoff_rejected_without_corroboration(self):
    self.set_gps(37.00008, -122.00008)
    self.set_targets([
      {"latitude": 37.00000, "longitude": -122.00008, "velocity": 17.0},
      {"latitude": 37.00008, "longitude": -122.00008, "velocity": 14.3},
      {"latitude": 37.00012, "longitude": -122.00012, "velocity": 14.1},
      {"latitude": 37.00016, "longitude": -122.00016, "velocity": 13.9},
    ])

    sm = self._FakeSM(curvature=0.0006, pred_lat_accels=[0.18, 0.18, 0.18, 0.18])
    self.scc_m.update(True, False, 17.1, 0.0, 30.0, sm)
    self.scc_m.update(True, False, 17.1, 0.0, 30.0, sm)
    assert self.scc_m.v_target == 0.0

  def test_material_slowdown_single_model_spike_does_not_corroborate(self):
    self.set_gps(37.00008, -122.00008)
    self.set_targets([
      {"latitude": 37.00000, "longitude": -122.00008, "velocity": 17.0},
      {"latitude": 37.00008, "longitude": -122.00008, "velocity": 12.2},
      {"latitude": 37.00012, "longitude": -122.00012, "velocity": 12.0},
      {"latitude": 37.00016, "longitude": -122.00016, "velocity": 11.8},
    ])

    pred_lat_accels = [0.18] * 32 + [5.0]
    sm = self._FakeSM(curvature=0.0006, pred_lat_accels=pred_lat_accels)
    self.scc_m.update(True, False, 17.1, 0.0, 30.0, sm)
    self.scc_m.update(True, False, 17.1, 0.0, 30.0, sm)
    assert self.scc_m.v_target == 0.0

  def test_material_slowdown_threshold_is_inclusive(self):
    self.set_gps(37.00008, -122.00008)
    self.set_targets([
      {"latitude": 37.00000, "longitude": -122.00008, "velocity": 17.1},
      {"latitude": 37.00008, "longitude": -122.00008, "velocity": 14.6},
      {"latitude": 37.00012, "longitude": -122.00012, "velocity": 14.5},
      {"latitude": 37.00016, "longitude": -122.00016, "velocity": 14.4},
    ])

    sm = self._FakeSM(curvature=0.0006, pred_lat_accels=[0.18, 0.18, 0.18, 0.18])
    self.scc_m.update(True, False, 17.1, 0.0, 30.0, sm)
    self.scc_m.update(True, False, 17.1, 0.0, 30.0, sm)
    assert self.scc_m.v_target == 0.0

  def test_material_slowdown_accepted_when_model_corroborates(self):
    self.set_gps(37.00008, -122.00008)
    self.set_targets([
      {"latitude": 37.00000, "longitude": -122.00008, "velocity": 17.0},
      {"latitude": 37.00004, "longitude": -122.00004, "velocity": 14.3},
      {"latitude": 37.00008, "longitude": -122.00008, "velocity": 12.2},
      {"latitude": 37.00012, "longitude": -122.00012, "velocity": 12.0},
    ])

    sm = self._FakeSM(curvature=0.0006, pred_lat_accels=[0.2, 0.3, 0.95, 1.0])
    self.scc_m.update(True, False, 17.1, 0.0, 30.0, sm)
    self.scc_m.update(True, False, 17.1, 0.0, 30.0, sm)
    assert self.scc_m.v_target == 12.2

  def test_material_slowdown_accepted_when_current_lateral_accel_corroborates(self):
    self.set_gps(37.00008, -122.00008)
    self.set_targets([
      {"latitude": 37.00000, "longitude": -122.00008, "velocity": 17.0},
      {"latitude": 37.00004, "longitude": -122.00004, "velocity": 14.3},
      {"latitude": 37.00008, "longitude": -122.00008, "velocity": 12.2},
      {"latitude": 37.00012, "longitude": -122.00012, "velocity": 12.0},
    ])

    sm = self._FakeSM(curvature=0.0018, pred_lat_accels=[0.18, 0.18, 0.18, 0.18])
    self.scc_m.update(True, False, 17.1, 0.0, 30.0, sm)
    self.scc_m.update(True, False, 17.1, 0.0, 30.0, sm)
    assert self.scc_m.v_target == 12.2

  def test_small_non_material_slowdown_still_accepted_without_corroboration(self):
    self.set_gps(37.00008, -122.00008)
    self.set_targets([
      {"latitude": 37.00000, "longitude": -122.00008, "velocity": 24.0},
      {"latitude": 37.00004, "longitude": -122.00004, "velocity": 23.0},
      {"latitude": 37.00008, "longitude": -122.00008, "velocity": 22.8},
      {"latitude": 37.00012, "longitude": -122.00012, "velocity": 22.5},
    ])

    self.scc_m.update(True, False, 24.0, 0.0, 30.0, None)
    self.scc_m.update(True, False, 24.0, 0.0, 30.0, None)
    assert self.scc_m.v_target == 22.8

  def test_material_slowdown_missing_or_malformed_model_fails_closed(self):
    self.set_gps(37.00008, -122.00008)
    self.set_targets([
      {"latitude": 37.00000, "longitude": -122.00008, "velocity": 17.0},
      {"latitude": 37.00004, "longitude": -122.00004, "velocity": 14.3},
      {"latitude": 37.00008, "longitude": -122.00008, "velocity": 12.2},
    ])

    self.scc_m.update(True, False, 17.1, 0.0, 30.0, None)
    self.scc_m.update(True, False, 17.1, 0.0, 30.0, None)
    assert self.scc_m.v_target == 0.0

    bad_sm = self._FakeSM(curvature=0.0006, pred_lat_accels=[])
    self.scc_m.update(True, False, 17.1, 0.0, 30.0, bad_sm)
    assert self.scc_m.v_target == 0.0

  # TODO-SP: mock data from modelV2 to test other states
