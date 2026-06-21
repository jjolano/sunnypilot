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


def cp(lateral_tuning='torque'):
  return SimpleNamespace(lateralTuning=SimpleNamespace(which=lambda: lateral_tuning))


@pytest.fixture(autouse=True)
def patch_torque_classes(monkeypatch):
  monkeypatch.setattr(controlsd_ext, 'LatControlTorqueV0', DummyTorqueV0)
  monkeypatch.setattr(controlsd_ext, 'LatControlTorqueV21', DummyTorqueV21)


def select(tune, lateral_tuning='torque'):
  lac = DummyLatControl()
  return controlsd_ext.select_torque_controller(cp(lateral_tuning), SimpleNamespace(), SimpleNamespace(), 0.01, lac, tune), lac


def test_missing_or_v0_tune_selects_v0():
  selected, _ = select(0.0)
  assert isinstance(selected, DummyTorqueV0)


def test_v1_tune_returns_existing_lac():
  selected, lac = select(1.0)
  assert selected is lac


def test_v21_tune_selects_v21_independent_of_enforce_gate():
  selected, _ = select(2.1)
  assert isinstance(selected, DummyTorqueV21)


def test_initialize_lateral_control_does_not_read_enforce_torque_control():
  class P:
    def get(self, key):
      assert key == 'TorqueControlTune'
      return 2.1

    def get_bool(self, key):
      raise AssertionError(f'{key} must not participate in controller selection')

  ext = SimpleNamespace(params=P(), CP=cp(), CP_SP=SimpleNamespace())
  selected = controlsd_ext.ControlsExt.initialize_lateral_control(ext, DummyLatControl(), SimpleNamespace(), 0.01)
  assert isinstance(selected, DummyTorqueV21)


def test_invalid_tune_falls_back_to_v0():
  selected, _ = select(9.9)
  assert isinstance(selected, DummyTorqueV0)


def test_non_torque_lateral_tuning_ignores_tune():
  selected, lac = select(2.1, lateral_tuning='pid')
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
