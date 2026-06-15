from __future__ import annotations

from types import SimpleNamespace

from openpilot.sunnypilot.sunnylink.backups.manager import BackupManagerSP


class DummyParams:
  def __init__(self, keys): self._keys = keys
  def all_keys(self, *_): return [k.encode() for k in self._keys]


def test_custom_longitudinal_backup_skips_legacy_mode_keys(monkeypatch):
  mgr = object.__new__(BackupManagerSP)
  mgr.params = DummyParams(["CustomLongitudinalEnabled", "CustomLongitudinalMode", "ExperimentalMode", "DynamicExperimentalControl", "OtherParam"])

  restored = []
  monkeypatch.setattr("openpilot.sunnypilot.sunnylink.backups.manager.save_param_from_base64_encoded_string",
                      lambda key, value: restored.append(key))
  mgr._apply_config({
    "CustomLongitudinalEnabled": "MQ==",
    "CustomLongitudinalMode": "c2Nj",
    "ExperimentalMode": "MQ==",
    "DynamicExperimentalControl": "MA==",
    "OtherParam": "QQ==",
  })

  assert restored == ["CustomLongitudinalEnabled", "CustomLongitudinalMode", "OtherParam"]


def test_generic_backup_restores_legacy_mode_keys_when_custom_absent(monkeypatch):
  mgr = object.__new__(BackupManagerSP)
  mgr.params = DummyParams(["ExperimentalMode", "DynamicExperimentalControl"])

  restored = []
  monkeypatch.setattr("openpilot.sunnypilot.sunnylink.backups.manager.save_param_from_base64_encoded_string",
                      lambda key, value: restored.append(key))
  mgr._apply_config({"ExperimentalMode": "MQ==", "DynamicExperimentalControl": "MA=="})

  assert restored == ["ExperimentalMode", "DynamicExperimentalControl"]


def test_custom_disabled_backup_restores_legacy_mode_keys(monkeypatch):
  mgr = object.__new__(BackupManagerSP)
  mgr.params = DummyParams(["CustomLongitudinalEnabled", "CustomLongitudinalMode", "ExperimentalMode", "DynamicExperimentalControl"])

  restored = []
  monkeypatch.setattr("openpilot.sunnypilot.sunnylink.backups.manager.save_param_from_base64_encoded_string",
                      lambda key, value: restored.append(key))
  mgr._apply_config({
    "CustomLongitudinalEnabled": "MA==",
    "CustomLongitudinalMode": "c2Nj",
    "ExperimentalMode": "MQ==",
    "DynamicExperimentalControl": "MQ==",
  })

  assert restored == ["CustomLongitudinalEnabled", "CustomLongitudinalMode", "ExperimentalMode", "DynamicExperimentalControl"]
