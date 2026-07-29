from types import SimpleNamespace

import pytest

from openpilot.sunnypilot.selfdrive.controls import controlsd_ext


class DummyLatControl:
  pass


class DummyTorqueV0:
  def __init__(self, CP, CP_SP, CI, dt):
    self.CP = CP


class DummyTorqueV21:
  def __init__(self, CP, CP_SP, CI, dt):
    self.CP = CP


class FakeParams:
  """Stand-in for Params that only participates in torque selection."""
  def __init__(self, tune=None, enforce=False):
    self._tune = tune
    self._enforce = enforce

  def get(self, key):
    assert key == 'TorqueControlTune'
    return self._tune

  def get_bool(self, key):
    assert key == 'EnforceTorqueControl'
    return self._enforce


def cp(lateral_tuning='torque'):
  return SimpleNamespace(lateralTuning=SimpleNamespace(which=lambda: lateral_tuning))


@pytest.fixture(autouse=True)
def patch_torque_classes(monkeypatch):
  monkeypatch.setattr(controlsd_ext, 'LatControlTorqueV0', DummyTorqueV0)
  monkeypatch.setattr(controlsd_ext, 'LatControlTorqueV21', DummyTorqueV21)


def select(tune, lateral_tuning='torque', enforce=False):
  lac = DummyLatControl()
  params = FakeParams(tune=tune, enforce=enforce)
  return controlsd_ext.select_torque_controller(cp(lateral_tuning), SimpleNamespace(), SimpleNamespace(), 0.01, lac, tune, params), lac


def test_missing_or_v0_tune_with_enforce_selects_v0():
  selected, _ = select(0.0, enforce=True)
  assert isinstance(selected, DummyTorqueV0)


def test_v1_tune_with_enforce_returns_existing_lac():
  selected, lac = select(1.0, enforce=True)
  assert selected is lac


def test_v21_tune_with_enforce_selects_v21():
  selected, _ = select(2.1, enforce=True)
  assert isinstance(selected, DummyTorqueV21)


def test_enforce_false_ignores_torque_tune_and_returns_platform_default():
  for tune in (0.0, 1.0, 2.1, 9.9):
    selected, lac = select(tune, enforce=False)
    assert selected is lac


def test_initialize_lateral_control_gates_on_enforce_torque_control():
  params = FakeParams(tune=2.1, enforce=True)
  ext = SimpleNamespace(params=params, CP=cp(), CP_SP=SimpleNamespace())
  selected = controlsd_ext.ControlsExt.initialize_lateral_control(ext, DummyLatControl(), SimpleNamespace(), 0.01)
  assert isinstance(selected, DummyTorqueV21)


def test_invalid_tune_with_enforce_falls_back_to_v0():
  selected, _ = select(9.9, enforce=True)
  assert isinstance(selected, DummyTorqueV0)


def test_non_torque_lateral_tuning_ignores_tune_and_enforce():
  lac = DummyLatControl()
  params = FakeParams(tune=2.1, enforce=True)
  selected = controlsd_ext.select_torque_controller(cp('pid'), SimpleNamespace(), SimpleNamespace(), 0.01, lac, 2.1, params)
  assert selected is lac


def test_read_torque_control_tune_missing_or_invalid_defaults_to_v0():
  class P:
    def __init__(self, value):
      self.value = value

    def get(self, key):
      assert key == 'TorqueControlTune'
      return self.value

  assert controlsd_ext.read_torque_control_tune(P(None)) == 0.0
  assert controlsd_ext.read_torque_control_tune(P('bad')) == 0.0
  assert controlsd_ext.read_torque_control_tune(P('2.1')) == 2.1


def test_live_lateral_delay_is_consumed_each_control_tick():
  ext = SimpleNamespace(lat_delay=0.2, _live_lat_delay_enabled=True)
  live_delay = SimpleNamespace(lateralDelay=0.12)

  class FakeSm:
    alive = {"liveDelay": True}
    valid = {"liveDelay": True}

    def __getitem__(self, key):
      assert key == "liveDelay"
      return live_delay

  sm = FakeSm()
  assert controlsd_ext.ControlsExt.current_lateral_delay(ext, sm) == pytest.approx(0.12)
  live_delay.lateralDelay = 0.31
  assert controlsd_ext.ControlsExt.current_lateral_delay(ext, sm) == pytest.approx(0.31)
