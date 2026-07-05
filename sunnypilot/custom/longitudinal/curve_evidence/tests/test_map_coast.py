"""Coast-tier target extraction from the SCC-Map controller.

The coast pass exposes the kinematically most binding material slowdown within
MAX_COAST_LOOKAHEAD for lift-off-only shaping. It shares the braking pass's route-hygiene
gates (GPS proximity, bracketing, monotonic forward distances) but not its short-drop /
lateral-corroboration gates, because it can never request braking.
"""
from __future__ import annotations

import json
import platform

import pytest

from openpilot.common.params import Params
from openpilot.sunnypilot.custom.longitudinal.curve_evidence import map_controller


LON = -122.0
# ~111.2 m per 0.001 deg latitude at the test coordinates.
EGO_LAT = 37.0001


def _route(*points):
  return [{"latitude": lat, "longitude": LON, "velocity": v} for lat, v in points]


class TestMapCoastTargets:

  def setup_method(self):
    self.params = Params()
    self.params.put_bool("SmartCruiseControlMap", True, block=True)
    self.mem_params = Params("/dev/shm/params") if platform.system() != "Darwin" else self.params
    self._set_gps(EGO_LAT, LON)

  def _set_gps(self, lat, lon):
    self.mem_params.put("LastGPSPosition", f'{{"latitude": {lat}, "longitude": {lon}}}', block=True)

  def _set_targets(self, targets):
    self.mem_params.put("MapTargetVelocities", json.dumps(targets), block=True)

  def _controller(self):
    c = map_controller.SmartCruiseControlMap()
    c.enabled = True
    return c

  def _update(self, c, v_ego=25.0):
    c.update(True, False, v_ego, 0.0, 30.0)

  def test_coast_target_beyond_braking_window_no_corroboration_needed(self):
    self._set_targets(_route((37.0000, 30.0), (EGO_LAT, 30.0), (37.0010, 30.0), (37.0020, 12.0)))
    c = self._controller()
    self._update(c)
    assert c.coast_v_target == pytest.approx(12.0)
    assert 200.0 < c.coast_distance < 225.0
    # The braking pass stays inert for this uncorroborated material drop; coast decouples.
    assert c.v_target == 0.0

  def test_coast_ignores_targets_beyond_lookahead_and_non_material(self):
    # Slow point ~656 m ahead: beyond MAX_COAST_LOOKAHEAD.
    self._set_targets(_route((37.0000, 30.0), (EGO_LAT, 30.0), (37.0010, 30.0), (37.0060, 12.0)))
    c = self._controller()
    self._update(c)
    assert c.coast_v_target == 0.0 and c.coast_distance == 0.0
    # Non-material slowdown (25 -> 23.5 m/s < MATERIAL_DROP_DELTA_V): ignored.
    self._set_targets(_route((37.0000, 30.0), (EGO_LAT, 30.0), (37.0010, 30.0), (37.0020, 23.5)))
    self._update(c)
    assert c.coast_v_target == 0.0

  def test_coast_picks_most_binding_target(self):
    # 18 m/s at ~211 m needs ~-0.71; 10 m/s at ~333 m needs ~-0.79 -> the farther one binds.
    self._set_targets(_route((37.0000, 30.0), (EGO_LAT, 30.0), (37.0010, 30.0),
                             (37.0020, 18.0), (37.0031, 10.0)))
    c = self._controller()
    self._update(c)
    assert c.coast_v_target == pytest.approx(10.0)
    assert 320.0 < c.coast_distance < 345.0

  def test_coast_cleared_on_hygiene_failure_and_disable(self):
    self._set_targets(_route((37.0000, 30.0), (EGO_LAT, 30.0), (37.0010, 30.0), (37.0020, 12.0)))
    c = self._controller()
    self._update(c)
    assert c.coast_v_target > 0.0
    # GPS off-route (> MAX_ROUTE_TARGET_DISTANCE from every point): stale targets cleared.
    self._set_gps(38.0, LON)
    self._update(c)
    assert c.coast_v_target == 0.0 and c.coast_distance == 0.0
    # Long-disabled path clears via _clear_targets without running calculations.
    self._set_gps(EGO_LAT, LON)
    self._update(c)
    assert c.coast_v_target > 0.0
    c.update(False, False, 25.0, 0.0, 30.0)
    assert c.coast_v_target == 0.0 and c.coast_distance == 0.0
