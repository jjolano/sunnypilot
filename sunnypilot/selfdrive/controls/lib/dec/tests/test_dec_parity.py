from __future__ import annotations

from types import SimpleNamespace

import pytest

from openpilot.sunnypilot.selfdrive.controls.lib.dec import constants, constants_v1
from openpilot.sunnypilot.selfdrive.controls.lib.dec.dec import DynamicExperimentalController
from openpilot.sunnypilot.selfdrive.controls.lib.dec.dec_v1 import DynamicExperimentalController as DynamicExperimentalControllerV1


class FakeParams:
  def __init__(self, enabled: bool = True):
    self.enabled = bool(enabled)

  def get_bool(self, name: str) -> bool:
    assert name == "DynamicExperimentalControl"
    return self.enabled


def _cp(*, radar_unavailable: bool = False):
  return SimpleNamespace(radarUnavailable=radar_unavailable)


def _mpc(crash_cnt: int = 0):
  return SimpleNamespace(crash_cnt=crash_cnt)


def _model(*, endpoint_x: float = 120.0, valid: bool = True):
  size = 33 if valid else 10
  position_x = [float(endpoint_x)] * size
  orientation_x = [0.0] * size
  return SimpleNamespace(position=SimpleNamespace(x=position_x), orientation=SimpleNamespace(x=orientation_x))


def _sm(*, v_ego: float = 10.0, v_cruise: float = 20.0, standstill: bool = False,
        lead_status: float = 0.0, experimental: bool = True, model=None):
  return {
    'carState': SimpleNamespace(vEgo=v_ego, vCruise=v_cruise, standstill=standstill),
    'radarState': SimpleNamespace(leadOne=SimpleNamespace(status=lead_status)),
    'modelV2': model if model is not None else _model(),
    'selfdriveState': SimpleNamespace(experimentalMode=experimental),
  }


def _controllers(*, radar_unavailable: bool = False, enabled: bool = True, crash_cnt: int = 0):
  cp1 = _cp(radar_unavailable=radar_unavailable)
  cp2 = _cp(radar_unavailable=radar_unavailable)
  mpc1 = _mpc(crash_cnt)
  mpc2 = _mpc(crash_cnt)
  return (
    DynamicExperimentalControllerV1(cp1, mpc1, params=FakeParams(enabled)),
    DynamicExperimentalController(cp2, mpc2, params=FakeParams(enabled)),
    mpc1,
    mpc2,
  )


def _assert_controller_state_equal(v1, v2):
  assert v1.mode() == v2.mode()
  assert v1.enabled() == v2.enabled()
  assert v1.active() == v2.active()
  for attr in (
    '_frame', '_urgency', '_has_lead_filtered', '_has_slow_down', '_has_slowness', '_has_mpc_fcw',
    '_v_ego_kph', '_v_cruise_kph', '_has_standstill', '_mpc_fcw_crash_cnt', '_standstill_count',
    '_endpoint_x', '_expected_distance', '_trajectory_valid',
  ):
    v1_value = getattr(v1, attr)
    v2_value = getattr(v2, attr)
    if isinstance(v1_value, float):
      assert v2_value == pytest.approx(v1_value)
    else:
      assert v2_value == v1_value
  assert v2._mode_manager.__dict__ == v1._mode_manager.__dict__


def test_constants_match_v1_backup():
  assert constants.WMACConstants.LEAD_PROB == constants_v1.WMACConstants.LEAD_PROB
  assert constants.WMACConstants.SLOW_DOWN_BP == constants_v1.WMACConstants.SLOW_DOWN_BP
  assert constants.WMACConstants.SLOW_DOWN_DIST == constants_v1.WMACConstants.SLOW_DOWN_DIST
  assert constants.WMACConstants.SLOWNESS_CRUISE_OFFSET == constants_v1.WMACConstants.SLOWNESS_CRUISE_OFFSET


def test_initial_public_api_parity():
  v1, v2, _, _ = _controllers()
  _assert_controller_state_equal(v1, v2)
  assert hasattr(v2, 'set_mpc_fcw_crash_cnt')


@pytest.mark.parametrize("enabled,experimental", [(True, True), (True, False), (False, True)])
def test_active_enabled_parity(enabled: bool, experimental: bool):
  v1, v2, _, _ = _controllers(enabled=enabled)
  frame = _sm(experimental=experimental)
  for _ in range(5):
    v1.update(frame)
    v2.update(frame)
    _assert_controller_state_equal(v1, v2)


def test_radar_standstill_and_lead_sequence_parity():
  v1, v2, _, _ = _controllers(radar_unavailable=False)
  frames = [
    _sm(v_ego=12.0, v_cruise=25.0, lead_status=1.0),
    _sm(v_ego=0.0, v_cruise=25.0, standstill=True, lead_status=1.0),
    _sm(v_ego=0.0, v_cruise=25.0, standstill=True, lead_status=1.0),
    _sm(v_ego=3.0, v_cruise=25.0, standstill=False, lead_status=1.0),
  ]
  for frame in frames * 5:
    v1.update(frame)
    v2.update(frame)
    _assert_controller_state_equal(v1, v2)


def test_radarless_slowdown_sequence_parity():
  v1, v2, _, _ = _controllers(radar_unavailable=True)
  frames = [
    _sm(v_ego=25.0, v_cruise=30.0, model=_model(valid=False)),
    _sm(v_ego=25.0, v_cruise=30.0, model=_model(endpoint_x=10.0)),
    _sm(v_ego=8.0, v_cruise=30.0, model=_model(endpoint_x=140.0)),
  ]
  for frame in frames * 8:
    v1.update(frame)
    v2.update(frame)
    _assert_controller_state_equal(v1, v2)


def test_fcw_setter_and_update_parity():
  v1, v2, mpc1, mpc2 = _controllers(crash_cnt=0)
  mpc1.crash_cnt = 1
  mpc2.crash_cnt = 1
  v1.set_mpc_fcw_crash_cnt()
  v2.set_mpc_fcw_crash_cnt()
  _assert_controller_state_equal(v1, v2)
  for _ in range(4):
    v1.update(_sm())
    v2.update(_sm())
    _assert_controller_state_equal(v1, v2)
