from dataclasses import FrozenInstanceError, replace

import pytest

from openpilot.sunnypilot.custom.longitudinal.modes import LongitudinalMode
from openpilot.selfdrive.selfdrived.selfdrived import SelfdriveConfigSnapshot, SelfdriveD


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
    setattr(s.config, "personality", 2)

  old_config = s.config
  s.config = replace(old_config, experimental_mode=True, personality=2)

  assert s.config is not old_config
  assert s.is_metric is True
  assert s.custom_longitudinal_enabled is True
  assert s.custom_longitudinal_mode is LongitudinalMode.E2E
  assert s.experimental_mode is True
  assert s.personality == 2
