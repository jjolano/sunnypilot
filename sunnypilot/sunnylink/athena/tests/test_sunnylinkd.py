"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from openpilot.sunnypilot.sunnylink.athena import sunnylinkd
from openpilot.selfdrive.controls.lib.longitudinal_modes import LONGITUDINAL_MODE_MIGRATION_PARAM, LONGITUDINAL_MODE_MIGRATION_VERSION


class FakeParams:
  def __init__(self, values=None):
    self.values = {"ParamsVersion": "0"} if values is None else dict(values)

  def get(self, key, *args, **kwargs):
    return self.values.get(key)

  def put(self, key, value):
    self.values[key] = str(value)


class TestSunnylinkdMethods:
  def setup_method(self):
    self.saved_params = []

    self.original_save = sunnylinkd.save_param_from_base64_encoded_string
    self.original_params = sunnylinkd.params

    def mock_save_param(key, value, compression=False):
      self.saved_params.append((key, value, compression))

    sunnylinkd.save_param_from_base64_encoded_string = mock_save_param

  def teardown_method(self):
    sunnylinkd.save_param_from_base64_encoded_string = self.original_save
    sunnylinkd.params = self.original_params

  def test_saveParams_blocked(self):
    blocked_params = {
      "GithubUsername": "attacker",
      "GithubSshKeys": "ssh-rsa attacker_key",
    }

    sunnylinkd.saveParams(blocked_params)

    assert len(self.saved_params) == 0

  def test_saveParams_allowed(self):
    allowed_params = {
      "SpeedLimitOffset": "5",
      "MyCustomParam": "123"
    }

    sunnylinkd.saveParams(allowed_params)

    # verify content
    assert len(self.saved_params) == 2
    keys_saved = [p[0] for p in self.saved_params]
    assert "SpeedLimitOffset" in keys_saved
    assert "MyCustomParam" in keys_saved

  def test_saveParams_mixed(self):
    mixed_params = {
      "GithubUsername": "attacker",
      "SpeedLimitOffset": "10"
    }

    sunnylinkd.saveParams(mixed_params)

    # should save allowed one
    assert len(self.saved_params) == 1
    assert self.saved_params[0][0] == "SpeedLimitOffset"
    assert self.saved_params[0][1] == "10"

  def test_saveParams_after_migration_drops_legacy_longitudinal_mode_params(self):
    sunnylinkd.params = FakeParams({
      LONGITUDINAL_MODE_MIGRATION_PARAM: LONGITUDINAL_MODE_MIGRATION_VERSION,
      "ParamsVersion": "0",
    })
    params_to_update = {
      "DynamicExperimentalControl": "1",
      "SmartCruiseControlVision": "1",
      "SmartCruiseControlMap": "1",
      "ExperimentalMode": "1",
      "SpeedLimitMode": "2",
    }

    sunnylinkd.saveParams(params_to_update)

    assert [param[0] for param in self.saved_params] == ["SpeedLimitMode"]
    assert self.saved_params[0][1] == "2"
