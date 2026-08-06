from dataclasses import FrozenInstanceError, replace

import pytest

from openpilot.common.realtime import Priority
from openpilot.sunnypilot.custom.longitudinal.modes import LongitudinalMode
from openpilot.selfdrive.selfdrived.selfdrived import SelfdriveConfigSnapshot, SelfdriveD, main


def test_selfdrive_config_snapshot_swaps_atomically():
  s = SelfdriveD.__new__(SelfdriveD)
  s.config = SelfdriveConfigSnapshot(
    is_metric=True,
    is_ldw_enabled=False,
    disengage_on_accelerator=True,
    custom_longitudinal_enabled=True,
    custom_longitudinal_mode=LongitudinalMode.E2E,
    experimental_mode=False,
    personality=1,
  )

  assert s.is_metric is True
  assert s.is_ldw_enabled is False
  assert s.disengage_on_accelerator is True
  assert s.custom_longitudinal_enabled is True
  assert s.custom_longitudinal_mode is LongitudinalMode.E2E
  assert s.experimental_mode is False
  assert s.personality == 1

  with pytest.raises(FrozenInstanceError):
    setattr(s.config, "personality", 2)  # noqa: B010

  old_config = s.config
  s.config = replace(old_config, experimental_mode=True, personality=2)

  assert s.config is not old_config
  assert s.is_metric is True
  assert s.custom_longitudinal_enabled is True
  assert s.custom_longitudinal_mode is LongitudinalMode.E2E
  assert s.experimental_mode is True
  assert s.personality == 2


def test_params_thread_refreshes_custom_longitudinal_enabled(monkeypatch):
  class FakeParams:
    def get_bool(self, key):
      return {"CustomLongitudinalEnabled": False, "IsMetric": True,
              "IsLdwEnabled": False, "DisengageOnAccelerator": True}.get(key, False)

    def get(self, key, return_default=False):
      return {"CustomLongitudinalMode": "acc", "LongitudinalPersonality": 1}.get(key)

  class Once:
    calls = 0

    def is_set(self):
      self.calls += 1
      return self.calls > 1

  s = SelfdriveD.__new__(SelfdriveD)
  s.params = FakeParams()
  s.CP = type("CP", (), {"openpilotLongitudinalControl": True})()
  s.active_custom_longitudinal_mode = LongitudinalMode.SCC
  s.mads = type("Mads", (), {"read_params": lambda self: None})()
  s.config = SelfdriveConfigSnapshot(True, False, True, True, LongitudinalMode.SCC, False, 1)
  monkeypatch.setattr("openpilot.selfdrive.selfdrived.selfdrived.time.sleep", lambda _: None)

  s.params_thread(Once())

  assert s.config.custom_longitudinal_enabled is False
  assert s.config.custom_longitudinal_mode is LongitudinalMode.ACC


def test_main_configures_cpu5_ctrl_high(monkeypatch):
  calls = []

  def fake_config_realtime_process(cpu, priority):
    calls.append((cpu, priority))

  class FakeSelfdriveD:
    def __init__(self, *args, **kwargs):
      pass

    def run(self):
      pass

  monkeypatch.setattr("openpilot.selfdrive.selfdrived.selfdrived.config_realtime_process", fake_config_realtime_process)
  monkeypatch.setattr("openpilot.selfdrive.selfdrived.selfdrived.SelfdriveD", FakeSelfdriveD)

  main()

  assert calls == [(5, Priority.CTRL_HIGH)]
