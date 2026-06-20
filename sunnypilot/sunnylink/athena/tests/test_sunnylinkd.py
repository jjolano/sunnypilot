"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import base64

from openpilot.sunnypilot.sunnylink.athena import sunnylinkd


def _b64(value: bytes) -> str:
  return base64.b64encode(value).decode()


class _FakeParams:
  def __init__(self):
    self._bools: dict[str, bool] = {}
    self._strings: dict[str, str | bytes] = {}

  def get_bool(self, key: str) -> bool:
    return bool(self._bools.get(key, False))

  def put_bool(self, key: str, value: bool):
    self._bools[key] = value

  def get(self, key: str, return_default: bool = False):
    if key in ("CarParamsPersistent", "CarPlatformBundle"):
      return None
    return self._strings.get(key)

  def put(self, key: str, value: str | bytes, block: bool = False):
    self._strings[key] = value


class TestSunnylinkdMethods:
  def setup_method(self):
    self.saved_params = []

    self.original_save = sunnylinkd.save_param_from_base64_encoded_string
    self.original_params = sunnylinkd.params

    def mock_save_param(key, value, compression=False):
      self.saved_params.append((key, value, compression))

    sunnylinkd.save_param_from_base64_encoded_string = mock_save_param
    sunnylinkd.params = _FakeParams()
    sunnylinkd.params.put_bool("IsOffroad", True)

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

  def test_saveParams_metadata_blocked(self):
    sunnylinkd.saveParams({"SshEnabled": base64.b64encode(b"1").decode()})

    assert len(self.saved_params) == 0

  def test_saveParams_static_safety_blocked(self):
    blocked_params = {
      "JoystickDebugMode": base64.b64encode(b"1").decode(),
      "AlphaLongitudinalEnabled": base64.b64encode(b"1").decode(),
      "LateralManeuverMode": base64.b64encode(b"1").decode(),
      "LongitudinalManeuverMode": base64.b64encode(b"1").decode(),
      "LiveTorqueSpeedAdaptiveParams": base64.b64encode(b"{}").decode(),
    }

    sunnylinkd.saveParams(blocked_params, attested_params=list(blocked_params))

    assert len(self.saved_params) == 0

  def test_saveParams_remote_configurable_false_panel_blocked(self):
    sunnylinkd.saveParams({"CameraOffset": base64.b64encode(b"0.1").decode()}, attested_params=["CameraOffset"])

    assert len(self.saved_params) == 0

  def test_saveParams_attestation_required(self):
    sunnylinkd.params.put_bool("CustomTorqueParams", True)
    params = {"TorqueParamsOverrideEnabled": base64.b64encode(b"1").decode()}

    sunnylinkd.saveParams(params)
    assert len(self.saved_params) == 0

    sunnylinkd.saveParams(params, attested_params=["TorqueParamsOverrideEnabled"])
    assert self.saved_params[-1][0] == "TorqueParamsOverrideEnabled"

  def test_saveParams_range_validation(self):
    sunnylinkd.params.put_bool("CustomTorqueParams", True)
    key = "TorqueParamsOverrideLatAccelFactor"
    out_of_range = {key: base64.b64encode(b"10.0").decode()}
    in_range = {key: base64.b64encode(b"1.5").decode()}

    sunnylinkd.saveParams(out_of_range, attested_params={key: True})
    assert len(self.saved_params) == 0

    sunnylinkd.saveParams(in_range, attested_params={key: True})
    assert self.saved_params[-1][0] == key

  def test_saveParams_nested_sub_item_range_validation(self):
    key = "BlinkerMinLateralControlSpeed"
    out_of_range = {key: base64.b64encode(b"300").decode()}
    in_range = {key: base64.b64encode(b"55").decode()}

    sunnylinkd.saveParams(out_of_range, attested_params=[key])
    assert len(self.saved_params) == 0

    sunnylinkd.saveParams(in_range, attested_params=[key])
    assert self.saved_params[-1][0] == key

  def test_saveParams_option_validation(self):
    key = "LiveTorqueSpeedAdaptiveMode"
    invalid = {key: base64.b64encode(b"bad").decode()}
    valid = {key: base64.b64encode(b"apply").decode()}

    sunnylinkd.saveParams(invalid, attested_params=[key])
    assert len(self.saved_params) == 0

    sunnylinkd.saveParams(valid, attested_params=[key])
    assert self.saved_params[-1][0] == key

  def test_saveParams_torque_settings_rejected_onroad(self):
    sunnylinkd.params.put_bool("IsOffroad", False)
    params = {
      "TorqueParamsOverrideEnabled": _b64(b"1"),
      "TorqueParamsOverrideLatAccelFactor": _b64(b"2.0"),
      "TorqueParamsOverrideFriction": _b64(b"0.5"),
      "LiveTorqueParamsToggle": _b64(b"1"),
      "LiveTorqueParamsRelaxedToggle": _b64(b"1"),
      "LiveTorqueSpeedAdaptiveMode": _b64(b"apply"),
      "EnforceTorqueControl": _b64(b"1"),
      "TorqueControlTune": _b64(b"1.0"),
      "CustomTorqueParams": _b64(b"1"),
      "NeuralNetworkLateralControl": _b64(b"1"),
    }

    sunnylinkd.saveParams(params, attested_params=list(params.keys()))
    assert len(self.saved_params) == 0

  def test_saveParams_lateral_demand_settings_rejected_onroad(self):
    sunnylinkd.params.put_bool("IsOffroad", False)
    params = {
      "CustomLateralDemandEnabled": _b64(b"1"),
      "CurveMemoryEnabled": _b64(b"1"),
      "LaneCenteringAssistEnabled": _b64(b"1"),
    }

    sunnylinkd.saveParams(params, attested_params=list(params.keys()))

    assert len(self.saved_params) == 0

  def test_saveParams_lateral_demand_settings_allowed_offroad(self):
    params = {
      "CustomLateralDemandEnabled": _b64(b"1"),
      "CurveMemoryEnabled": _b64(b"1"),
      "LaneCenteringAssistEnabled": _b64(b"1"),
    }

    sunnylinkd.saveParams(params, attested_params=list(params.keys()))

    saved_keys = {p[0] for p in self.saved_params}
    assert saved_keys == set(params.keys())

  def test_saveParams_torque_settings_allowed_offroad(self):
    params = {
      "TorqueParamsOverrideEnabled": _b64(b"1"),
      "TorqueParamsOverrideLatAccelFactor": _b64(b"2.0"),
      "TorqueParamsOverrideFriction": _b64(b"0.5"),
      "LiveTorqueParamsToggle": _b64(b"1"),
      "LiveTorqueParamsRelaxedToggle": _b64(b"1"),
      "LiveTorqueSpeedAdaptiveMode": _b64(b"apply"),
      "EnforceTorqueControl": _b64(b"1"),
      "TorqueControlTune": _b64(b"1.0"),
      "CustomTorqueParams": _b64(b"1"),
      "NeuralNetworkLateralControl": _b64(b"1"),
    }

    sunnylinkd.saveParams(params, attested_params=list(params.keys()))
    saved_keys = {p[0] for p in self.saved_params}
    assert saved_keys == set(params.keys())

  def test_saveParams_torque_override_requires_custom_torque_params(self):
    sunnylinkd.params.put_bool("CustomTorqueParams", False)
    params = {"TorqueParamsOverrideEnabled": _b64(b"1")}

    sunnylinkd.saveParams(params, attested_params=list(params.keys()))
    assert len(self.saved_params) == 0

    params["CustomTorqueParams"] = _b64(b"1")
    sunnylinkd.saveParams(params, attested_params=list(params.keys()))
    saved_keys = {p[0] for p in self.saved_params}
    assert "TorqueParamsOverrideEnabled" in saved_keys

  def test_saveParams_torque_manual_values_require_custom_torque_params(self):
    params = {
      "TorqueParamsOverrideLatAccelFactor": _b64(b"2.0"),
      "TorqueParamsOverrideFriction": _b64(b"0.5"),
    }

    sunnylinkd.saveParams(params, attested_params=list(params.keys()))
    assert len(self.saved_params) == 0

    sunnylinkd.params.put_bool("CustomTorqueParams", True)
    sunnylinkd.saveParams(params, attested_params=list(params.keys()))
    saved_keys = {p[0] for p in self.saved_params}
    assert "TorqueParamsOverrideLatAccelFactor" in saved_keys
    assert "TorqueParamsOverrideFriction" in saved_keys

  def test_saveParams_torque_custom_false_in_transaction_rejected(self):
    params = {
      "TorqueParamsOverrideEnabled": _b64(b"1"),
      "CustomTorqueParams": _b64(b"0"),
    }

    sunnylinkd.saveParams(params, attested_params=list(params.keys()))
    saved_keys = {p[0] for p in self.saved_params}
    assert "TorqueParamsOverrideEnabled" not in saved_keys

  def test_saveParams_invalid_custom_torque_transaction_does_not_unlock_override(self):
    params = {
      "CustomTorqueParams": _b64(b"1"),
      "TorqueParamsOverrideEnabled": _b64(b"1"),
    }

    sunnylinkd.saveParams(params, attested_params=["TorqueParamsOverrideEnabled"])

    saved_keys = {p[0] for p in self.saved_params}
    assert "CustomTorqueParams" not in saved_keys
    assert "TorqueParamsOverrideEnabled" not in saved_keys
