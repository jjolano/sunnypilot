from pathlib import Path

import pytest

from openpilot.selfdrive.controls.lib.controls_profile import (
  CONTROLS_PROFILE_MAPPINGS,
  DEFAULT_CONTROLS_PROFILE,
  DEFAULT_LATERAL_DEMAND_STACK,
  DEFAULT_TORQUE_CONTROL_TUNE,
  ControlsProfileId,
  TorqueControlTuneId,
  controls_profile_id_for_name,
  controls_profile_mapping_for,
  lateral_demand_stack_id_for_name,
  resolve_controls_profile,
  resolve_controls_profile_from_params,
  resolve_torque_control_tune,
  torque_control_tune_id_for_name,
)
from openpilot.selfdrive.controls.lib.lateral_demand_stacks.selector import (
  CUSTOM_EXPERIMENTAL as LATERAL_CUSTOM_EXPERIMENTAL,
  CUSTOM_RECOMMENDED as LATERAL_CUSTOM_RECOMMENDED,
  CUSTOM_V2 as LATERAL_CUSTOM_V2,
  SUNNYPILOT_CURRENT as LATERAL_SUNNYPILOT_CURRENT,
)
from openpilot.selfdrive.controls.lib.longitudinal_stacks.selector import (
  CUSTOM_EXPERIMENTAL as LONG_CUSTOM_EXPERIMENTAL,
  CUSTOM_V2 as LONG_CUSTOM_V2,
  SUNNYPILOT_CURRENT as LONG_SUNNYPILOT_CURRENT,
  resolve_longitudinal_stack,
)


class FakeParams:
  def __init__(self, values=None, bools=None):
    self.values = values or {}
    self.bools = bools or {}
    self.writes = {}

  def get(self, key, *args, **kwargs):
    if key in self.values:
      return self.values[key]
    if kwargs.get("return_default", False):
      return {
        "ControlsProfile": b"custom-2.0",
        "LateralDemandStack": b"custom-2.0",
        "LongitudinalStack": b"sunnypilot-current",
        "TorqueControlTune": b"4.1",
      }.get(key)
    return None

  def get_bool(self, key):
    return bool(self.bools.get(key, False))

  def put(self, key, value):
    self.writes[key] = value


def _resolved_torque(params: FakeParams) -> str:
  torque = resolve_controls_profile_from_params(params).controls_profile_resolution.torque_control_tune
  assert torque is not None
  return torque.value


def test_missing_controls_profile_does_not_select_torque_50():
  res = resolve_controls_profile(None)
  assert res.torque_control_tune == TorqueControlTuneId.V41
  assert res.lateral_demand_stack == DEFAULT_LATERAL_DEMAND_STACK


def test_unknown_controls_profile_does_not_select_torque_50():
  res = resolve_controls_profile("not-a-profile")
  assert res.resolved_profile == DEFAULT_CONTROLS_PROFILE
  assert res.torque_control_tune == TorqueControlTuneId.V41


def test_controls_profile_default_is_custom_2():
  assert DEFAULT_CONTROLS_PROFILE == ControlsProfileId.CUSTOM_2
  assert DEFAULT_LATERAL_DEMAND_STACK == LATERAL_CUSTOM_V2
  assert DEFAULT_TORQUE_CONTROL_TUNE == TorqueControlTuneId.V41


def test_controls_profile_custom_2_maps_to_41():
  res = resolve_controls_profile("custom-2.0")
  assert res.resolved_profile == ControlsProfileId.CUSTOM_2
  assert res.lateral_demand_stack == LATERAL_CUSTOM_V2
  assert res.torque_control_tune == TorqueControlTuneId.V41


def test_controls_profile_experimental_maps_to_50():
  res = resolve_controls_profile("custom-experimental")
  assert res.resolved_profile == ControlsProfileId.CUSTOM_EXPERIMENTAL
  assert res.lateral_demand_stack == LATERAL_CUSTOM_EXPERIMENTAL
  assert res.torque_control_tune == TorqueControlTuneId.V50_EXPERIMENTAL


def test_controls_profile_sunnypilot_current_maps_to_sunnypilot_current_stack():
  res = resolve_controls_profile("sunnypilot-current")
  assert res.lateral_demand_stack == LATERAL_SUNNYPILOT_CURRENT
  assert res.torque_control_tune is None
  assert res.torque_control_tune_resolution is None
  assert res.longitudinal_stack == LONG_SUNNYPILOT_CURRENT


def test_missing_controls_profile_preserves_existing_torque_control_tune_20():
  assert _resolved_torque(FakeParams({"TorqueControlTune": b"2.0"})) == "2.0"


def test_missing_controls_profile_preserves_existing_torque_control_tune_21():
  assert _resolved_torque(FakeParams({"TorqueControlTune": "2.1"})) == "2.1"


def test_missing_controls_profile_preserves_existing_torque_control_tune_41():
  assert _resolved_torque(FakeParams({"TorqueControlTune": "4.1"})) == "4.1"


def test_missing_controls_profile_without_existing_tune_defaults_to_41():
  assert _resolved_torque(FakeParams()) == "4.1"


def test_advanced_override_ignored_when_show_advanced_controls_false():
  params = FakeParams({"ControlsProfile": b"custom-2.0", "TorqueControlTune": b"5.0"})
  state = resolve_controls_profile_from_params(params)
  assert state.controls_profile_explicit is True
  assert state.torque_control_tune_explicit is True
  assert state.controls_profile_resolution.torque_control_tune == TorqueControlTuneId.V41


def test_advanced_override_honored_when_show_advanced_controls_true():
  params = FakeParams(
    {"ControlsProfile": b"custom-2.0", "TorqueControlTune": b"5.0", "LateralDemandStack": b"custom-experimental"},
    {"ShowAdvancedControls": True},
  )
  state = resolve_controls_profile_from_params(params)
  assert state.controls_profile_resolution.torque_control_tune == TorqueControlTuneId.V50_EXPERIMENTAL
  assert state.controls_profile_resolution.lateral_demand_stack == LATERAL_CUSTOM_EXPERIMENTAL


def test_controls_profile_custom_2_maps_longitudinal_stack():
  res = resolve_controls_profile("custom-2.0")
  assert res.longitudinal_stack == LONG_CUSTOM_V2


def test_controls_profile_experimental_maps_longitudinal_stack_or_fallback():
  res = resolve_controls_profile("custom-experimental")
  assert res.longitudinal_stack == LONG_CUSTOM_V2
  resolved_long = resolve_longitudinal_stack(res.longitudinal_stack)
  assert resolved_long.resolved_stack in {LONG_CUSTOM_EXPERIMENTAL, LONG_CUSTOM_V2, LONG_SUNNYPILOT_CURRENT}


def test_controls_profile_applies_longitudinal_stack_when_explicit():
  params = FakeParams({"ControlsProfile": b"custom-2.0", "LongitudinalStack": b"sunnypilot-current"})
  state = resolve_controls_profile_from_params(params)
  assert state.controls_profile_explicit is True
  assert state.controls_profile_resolution.longitudinal_stack == LONG_CUSTOM_V2
  repo_root = Path(__file__).parents[3]
  controlsd_source = (repo_root / "selfdrive/controls/controlsd.py").read_text()
  planner_source = (repo_root / "sunnypilot/selfdrive/controls/lib/longitudinal_planner.py").read_text()
  assert 'self.params.put("LongitudinalStack", self.resolved_longitudinal_stack)' in controlsd_source
  assert 'self.params.put("LongitudinalStack", requested_longitudinal_stack)' in planner_source


def test_missing_controls_profile_does_not_overwrite_existing_longitudinal_stack():
  params = FakeParams({"LongitudinalStack": b"sunnypilot-current"})
  state = resolve_controls_profile_from_params(params)
  assert state.controls_profile_explicit is False
  assert state.controls_profile_resolution.longitudinal_stack == LONG_SUNNYPILOT_CURRENT
  assert params.writes == {}
  planner_source = (Path(__file__).parents[3] / "sunnypilot/selfdrive/controls/lib/longitudinal_planner.py").read_text()
  assert 'requested_longitudinal_stack = self.params.get("LongitudinalStack", return_default=True)' in planner_source


def test_missing_controls_profile_longitudinal_stack_matches_param_default():
  params = FakeParams()
  state = resolve_controls_profile_from_params(params)

  assert state.controls_profile_explicit is False
  assert state.longitudinal_stack_explicit is False
  assert state.controls_profile_resolution.longitudinal_stack == LONG_SUNNYPILOT_CURRENT


def test_controls_profile_id_for_name_resolves_known_values():
  assert controls_profile_id_for_name("sunnypilot-current") == ControlsProfileId.SUNNYPILOT_CURRENT
  assert controls_profile_id_for_name("custom-recommended") == ControlsProfileId.CUSTOM_RECOMMENDED
  assert controls_profile_id_for_name("custom-2.0") == ControlsProfileId.CUSTOM_2
  assert controls_profile_id_for_name("custom-experimental") == ControlsProfileId.CUSTOM_EXPERIMENTAL


def test_torque_control_tune_id_for_name_resolves_supported_values():
  assert torque_control_tune_id_for_name("2.0") == TorqueControlTuneId.V20
  assert torque_control_tune_id_for_name(b"2.1") == TorqueControlTuneId.V21
  assert torque_control_tune_id_for_name(5.0) == TorqueControlTuneId.V50_EXPERIMENTAL
  assert resolve_torque_control_tune("not-a-tune").resolved_tune == TorqueControlTuneId.V41


def test_unknown_lateral_demand_stack_persists_safe_fallback():
  assert lateral_demand_stack_id_for_name("not-a-stack") == LATERAL_CUSTOM_V2


def test_all_profile_mappings_cover_every_id():
  seen = {m.profile_id for m in CONTROLS_PROFILE_MAPPINGS}
  assert seen == set(ControlsProfileId)
  for mapping in CONTROLS_PROFILE_MAPPINGS:
    assert mapping.lateral_demand_stack in {
      LATERAL_SUNNYPILOT_CURRENT,
      LATERAL_CUSTOM_RECOMMENDED,
      LATERAL_CUSTOM_V2,
      LATERAL_CUSTOM_EXPERIMENTAL,
    }
    assert mapping.torque_control_tune is None or mapping.torque_control_tune in set(TorqueControlTuneId)
    assert mapping.longitudinal_stack


def test_controls_profile_uses_runtime_lateral_stack_ids():
  runtime_ids = {LATERAL_SUNNYPILOT_CURRENT, LATERAL_CUSTOM_RECOMMENDED, LATERAL_CUSTOM_V2, LATERAL_CUSTOM_EXPERIMENTAL}
  assert {mapping.lateral_demand_stack for mapping in CONTROLS_PROFILE_MAPPINGS} <= runtime_ids


def test_no_duplicate_lateral_demand_stack_id_enums():
  repo_root = Path(__file__).parents[3]
  lib_files = list((repo_root / "selfdrive/controls/lib").rglob("*.py"))
  lib_files += list((repo_root / "sunnypilot/selfdrive/controls/lib").rglob("*.py"))
  definitions = [path for path in lib_files if "class LateralDemandStackId" in path.read_text()]
  assert definitions == []


def test_no_duplicate_lateral_demand_stack_output_definitions():
  repo_root = Path(__file__).parents[3]
  lib_files = list((repo_root / "selfdrive/controls/lib").rglob("*.py"))
  lib_files += list((repo_root / "sunnypilot/selfdrive/controls/lib").rglob("*.py"))
  definitions = [path for path in lib_files if "class LateralDemandStackOutput" in path.read_text()]
  assert definitions == [repo_root / "selfdrive/controls/lib/lateral_demand_stacks/interface.py"]


@pytest.mark.parametrize("value", [None, b"", "", "not-a-profile"])
def test_missing_or_unknown_controls_profile_safe_default_is_non_v5(value):
  res = resolve_controls_profile(value)
  assert res.torque_control_tune != TorqueControlTuneId.V50_EXPERIMENTAL
