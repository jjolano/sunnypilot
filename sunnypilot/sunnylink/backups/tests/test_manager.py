from openpilot.common.params import ParamKeyFlag
from openpilot.selfdrive.controls.lib.longitudinal_modes import (
  LONGITUDINAL_MODE_MIGRATION_PARAM,
  LONGITUDINAL_MODE_MIGRATION_VERSION,
  LONGITUDINAL_MODE_PARAM,
  LongitudinalMode,
)
from openpilot.sunnypilot.sunnylink.backups import manager


class FakeParams:
  def __init__(self):
    self.values = {
      LONGITUDINAL_MODE_PARAM: str(int(LongitudinalMode.ACC)),
      LONGITUDINAL_MODE_MIGRATION_PARAM: LONGITUDINAL_MODE_MIGRATION_VERSION,
    }

  def get(self, key, *args, **kwargs):
    return self.values.get(key)

  def all_keys(self, flag):
    assert flag == ParamKeyFlag.BACKUP
    return [
      b"DynamicExperimentalControl",
      b"SmartCruiseControlVision",
      b"SmartCruiseControlMap",
      b"ExperimentalMode",
      b"SpeedLimitMode",
    ]


def test_backup_restore_after_migration_skips_legacy_mode_params(monkeypatch):
  restored: list[tuple[str, str]] = []
  monkeypatch.setattr(
    manager,
    "save_param_from_base64_encoded_string",
    lambda key, value: restored.append((key, value)),
  )
  backup_manager = manager.BackupManagerSP.__new__(manager.BackupManagerSP)
  backup_manager.params = FakeParams()

  backup_manager._apply_config({
    "DynamicExperimentalControl": "legacy-dec",
    "SmartCruiseControlVision": "legacy-vision",
    "SmartCruiseControlMap": "legacy-map",
    "ExperimentalMode": "legacy-experimental",
    "SpeedLimitMode": "speed-limit-mode",
  })

  assert restored == [("SpeedLimitMode", "speed-limit-mode")]
  assert backup_manager.params.values[LONGITUDINAL_MODE_PARAM] == str(int(LongitudinalMode.ACC))


def test_backup_restore_keeps_non_mode_params_after_migration(monkeypatch):
  restored: list[tuple[str, str]] = []
  monkeypatch.setattr(
    manager,
    "save_param_from_base64_encoded_string",
    lambda key, value: restored.append((key, value)),
  )
  backup_manager = manager.BackupManagerSP.__new__(manager.BackupManagerSP)
  backup_manager.params = FakeParams()

  backup_manager._apply_config({"SpeedLimitMode": "normal-param"})

  assert restored == [("SpeedLimitMode", "normal-param")]
