"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import base64
import gzip
import json

import pytest

from openpilot.common.params import ParamKeyType
from openpilot.sunnypilot.sunnylink.athena import sunnylinkd


def _b64(value: bytes) -> str:
  return base64.b64encode(value).decode()


class _FakeParams:
  def __init__(self):
    self._bools: dict[str, bool] = {}
    self._strings: dict[str, str | bytes] = {}
    self._registered_keys: set[str] = set()
    self._types: dict[str, ParamKeyType] = {}

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

  def all_keys(self, flag=None):
    return [k.encode("utf-8") for k in self._registered_keys]

  def get_type(self, key: str):
    return self._types.get(key, ParamKeyType.STRING)

  def get_default_value(self, key: str):
    return None

  def register_for_remote_read(self, key: str, param_type: ParamKeyType):
    self._registered_keys.add(key)
    self._types[key] = param_type


class TestSunnylinkdMethods:
  def setup_method(self):
    self.saved_params = []
    self._param_bytes: dict[str, bytes] = {}

    self.original_save = sunnylinkd.save_param_from_base64_encoded_string
    self.original_params = sunnylinkd.params
    self.original_params_class = sunnylinkd.Params
    self.original_get_param_as_byte = sunnylinkd.get_param_as_byte

    def mock_save_param(key, value, compression=False):
      self.saved_params.append((key, value, compression))

    def mock_get_param_as_byte(param_name, params=None, get_default=False):
      return self._param_bytes.get(param_name)

    sunnylinkd.save_param_from_base64_encoded_string = mock_save_param
    sunnylinkd.get_param_as_byte = mock_get_param_as_byte
    sunnylinkd.params = _FakeParams()
    sunnylinkd.params.put_bool("IsOffroad", True)
    sunnylinkd.Params = lambda: sunnylinkd.params

  def teardown_method(self):
    sunnylinkd.save_param_from_base64_encoded_string = self.original_save
    sunnylinkd.params = self.original_params
    sunnylinkd.Params = self.original_params_class
    sunnylinkd.get_param_as_byte = self.original_get_param_as_byte

  def _register_remote_read(self, key: str, value: bytes, param_type: ParamKeyType):
    sunnylinkd.params.register_for_remote_read(key, param_type)
    self._param_bytes[key] = value

  def test_saveParams_blocked(self):
    blocked_params = {
      "GithubUsername": "attacker",
      "GithubSshKeys": "ssh-rsa attacker_key",
    }

    sunnylinkd.saveParams(blocked_params)

    assert len(self.saved_params) == 0

  def test_saveParams_allowed(self):
    allowed_params = {
      "ExperimentalMode": _b64(b"1"),
      "LongitudinalPersonality": _b64(b"1"),
    }

    sunnylinkd.saveParams(allowed_params)

    assert len(self.saved_params) == 2
    saved_keys = {p[0] for p in self.saved_params}
    assert saved_keys == {"ExperimentalMode", "LongitudinalPersonality"}

  def test_saveParams_unknown_unlisted_key_denied(self):
    sunnylinkd.saveParams({"MyCustomParam": "123"})

    assert len(self.saved_params) == 0

  def test_saveParams_unknown_speedlimitoffset_denied(self):
    # Not a real param key and not in the schema; should be default-denied.
    sunnylinkd.saveParams({"SpeedLimitOffset": "5"})

    assert len(self.saved_params) == 0

  def test_saveParams_mixed(self):
    mixed_params = {
      "GithubUsername": "attacker",
      "ExperimentalMode": _b64(b"1"),
    }

    sunnylinkd.saveParams(mixed_params)

    assert len(self.saved_params) == 1
    assert self.saved_params[0][0] == "ExperimentalMode"
    assert self.saved_params[0][1] == _b64(b"1")

  def test_saveParams_mixed_transaction_ignores_denied_and_increments_version(self):
    sunnylinkd.saveParams({
      "GithubUsername": "attacker",
      "MyCustomParam": "123",
      "ExperimentalMode": _b64(b"1"),
      "EnableCopyparty": _b64(b"1"),
    })

    assert len(self.saved_params) == 1
    assert self.saved_params[0][0] == "ExperimentalMode"
    assert sunnylinkd.params.get("ParamsVersion") == "1"

  def test_saveParams_all_denied_does_not_increment_version(self):
    sunnylinkd.saveParams({
      "GithubUsername": "attacker",
      "MyCustomParam": "123",
    })

    assert len(self.saved_params) == 0
    assert sunnylinkd.params.get("ParamsVersion") is None

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

  @pytest.mark.parametrize("key", [
    "EnableCopyparty",
    "EnableGithubRunner",
    "EnableSunnylinkUploader",
    "EnableTailscale",
    "TailscaleLoginRequested",
    "TailscaleLogoutRequested",
    "TailscaleInstallRequested",
  ])
  def test_saveParams_service_action_params_denied(self, key):
    sunnylinkd.saveParams({key: _b64(b"1")})

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
      "LaneCenteringAssistEnabled": _b64(b"1"),
    }

    sunnylinkd.saveParams(params, attested_params=list(params.keys()))

    assert len(self.saved_params) == 0

  def test_saveParams_lateral_demand_settings_allowed_offroad(self):
    params = {
      "CustomLateralDemandEnabled": _b64(b"1"),
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

  def test_getParams_omits_unlisted_non_sensitive_key(self):
    # The key exists on the device and is not sensitive, but it is not in the
    # settings schema, so remote read must default-deny and omit it.
    sunnylinkd.params.register_for_remote_read("MyCustomParam", ParamKeyType.STRING)
    self._param_bytes["MyCustomParam"] = b"secret"

    response = sunnylinkd.getParams(["MyCustomParam"])
    assert isinstance(response, dict)

    assert "MyCustomParam" not in response
    params_json = json.loads(response["params"])
    assert params_json == []

  def test_getParamsAllKeys_filters_by_remote_read_policy(self):
    self._register_remote_read("ExperimentalMode", b"1", ParamKeyType.BOOL)
    # registered on device but not remotely readable:
    sunnylinkd.params.register_for_remote_read("SshEnabled", ParamKeyType.BOOL)
    sunnylinkd.params.register_for_remote_read("TailscaleAuthURL", ParamKeyType.STRING)
    sunnylinkd.params.register_for_remote_read("EnableTailscale", ParamKeyType.BOOL)
    sunnylinkd.params.register_for_remote_read("MyCustomParam", ParamKeyType.STRING)

    keys = sunnylinkd.getParamsAllKeys()

    assert "ExperimentalMode" in keys
    assert "SshEnabled" not in keys
    assert "TailscaleAuthURL" not in keys
    assert "EnableTailscale" not in keys
    assert "MyCustomParam" not in keys

  def test_getParams_denies_blocked_and_service_action_keys(self):
    self._register_remote_read("ExperimentalMode", b"1", ParamKeyType.BOOL)
    self._register_remote_read("GithubUsername", b"attacker", ParamKeyType.STRING)
    self._register_remote_read("SshEnabled", b"1", ParamKeyType.BOOL)
    self._register_remote_read("EnableCopyparty", b"1", ParamKeyType.BOOL)
    self._register_remote_read("EnableTailscale", b"1", ParamKeyType.BOOL)

    response = sunnylinkd.getParams(["ExperimentalMode", "GithubUsername", "SshEnabled", "EnableCopyparty", "EnableTailscale"])
    assert isinstance(response, dict)

    assert "ExperimentalMode" in response
    assert "GithubUsername" not in response
    assert "SshEnabled" not in response
    assert "EnableCopyparty" not in response
    assert "EnableTailscale" not in response
    params_json = json.loads(response["params"])
    assert len(params_json) == 1
    assert params_json[0]["key"] == "ExperimentalMode"

  def test_getParams_denies_sensitive_keys(self):
    self._register_remote_read("ExperimentalMode", b"1", ParamKeyType.BOOL)
    self._register_remote_read("LiveTorqueSpeedAdaptiveParams", b"{}", ParamKeyType.STRING)
    self._register_remote_read("TailscaleAuthURL", b"https://login.tailscale.com/a/abc123", ParamKeyType.STRING)

    response = sunnylinkd.getParams(["ExperimentalMode", "LiveTorqueSpeedAdaptiveParams", "TailscaleAuthURL"])
    assert isinstance(response, dict)

    assert "ExperimentalMode" in response
    assert "LiveTorqueSpeedAdaptiveParams" not in response
    assert "TailscaleAuthURL" not in response
    params_json = json.loads(response["params"])
    assert len(params_json) == 1
    assert params_json[0]["key"] == "ExperimentalMode"

  def test_getParams_allowed_key_preserves_response_shape(self):
    self._register_remote_read("ExperimentalMode", b"1", ParamKeyType.BOOL)

    response = sunnylinkd.getParams(["ExperimentalMode"], compression=False)
    assert isinstance(response, dict)

    assert set(response.keys()) == {"ExperimentalMode", "params"}
    assert response["ExperimentalMode"] == _b64(b"1")
    params_json = json.loads(response["params"])
    assert params_json == [{
      "key": "ExperimentalMode",
      "value": _b64(b"1"),
      "type": ParamKeyType.BOOL.value,
      "is_compressed": False,
    }]

  def test_getParams_compressed_matches_uncompressed_shape(self):
    self._register_remote_read("ExperimentalMode", b"1", ParamKeyType.BOOL)

    response_uncompressed = sunnylinkd.getParams(["ExperimentalMode"], compression=False)
    response_compressed = sunnylinkd.getParams(["ExperimentalMode"], compression=True)
    assert isinstance(response_uncompressed, dict)
    assert isinstance(response_compressed, dict)

    assert response_uncompressed["ExperimentalMode"] == _b64(b"1")
    assert response_compressed["ExperimentalMode"] != response_uncompressed["ExperimentalMode"]

    uncompressed_entry = json.loads(response_uncompressed["params"])[0]
    compressed_entry = json.loads(response_compressed["params"])[0]

    assert uncompressed_entry["is_compressed"] is False
    assert compressed_entry["is_compressed"] is True
    assert base64.b64decode(compressed_entry["value"]) == gzip.compress(b"1")
    assert base64.b64decode(uncompressed_entry["value"]) == b"1"
