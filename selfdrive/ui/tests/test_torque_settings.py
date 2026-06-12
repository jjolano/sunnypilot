from pathlib import Path
import json


TORQUE_SETTINGS = Path(__file__).parents[1] / "sunnypilot/layouts/settings/steering_sub_layouts/torque_settings.py"
SUNNYLINK_METADATA = Path(__file__).parents[3] / "sunnypilot/sunnylink/params_metadata.json"


def test_control_calculation_hardening_toggle_is_in_torque_settings():
  source = TORQUE_SETTINGS.read_text()

  assert "_control_calculation_hardening_toggle = toggle_item_sp" in source
  assert 'param="ControlCalculationHardening"' in source
  assert "self._control_calculation_hardening_toggle," in source


def test_controls_profile_ui_values_match_resolver():
  source = TORQUE_SETTINGS.read_text()

  for value in ("sunnypilot-current", "custom-recommended", "custom-2.0", "custom-experimental"):
    assert value in source
  assert "Controls Profile" in source
  assert "torque 2.1" in source
  assert "torque 4.1." not in source
  assert "CONTROLS_PROFILE_LABELS" in source
  assert "CONTROLS_PROFILE_DESCRIPTIONS" in source


def test_torque_controller_ui_values_include_50_experimental():
  source = TORQUE_SETTINGS.read_text()

  assert "TORQUE_CONTROLLER_LABELS" in source
  for label in ('"2.0"', '"2.1"', '"3.0"', '"4.0"', '"4.1"'):
    assert label in source
  assert "5.0 Experimental" in source
  assert "Torque Controller" in source


def test_lateral_demand_stack_ui_values_match_selector():
  source = TORQUE_SETTINGS.read_text()

  assert "SUNNYPILOT_CURRENT" in source
  assert "CUSTOM_RECOMMENDED" in source
  assert "CUSTOM_V2" in source
  assert "CUSTOM_EXPERIMENTAL" in source
  assert "Lateral Demand Stack" in source
  assert "LATERAL_DEMAND_STACK_LABELS" in source


def test_no_shadow_option_in_controls_ui():
  source = TORQUE_SETTINGS.read_text().lower()

  assert "shadow" not in source


def test_torque_controller_sunnylink_values_are_strings():
  metadata = json.loads(SUNNYLINK_METADATA.read_text())
  options = metadata["TorqueControlTune"]["options"]

  assert {option["label"] for option in options} >= {"2.0", "2.1", "3.0", "4.0", "4.1", "5.0 Experimental"}
  for option in options:
    assert isinstance(option["value"], str)
  assert "Default uses 2.1" in metadata["TorqueControlTune"]["description"]
  assert "shadow" not in metadata["TorqueControlTune"]["description"].lower()


def test_controls_profile_sunnylink_metadata_exists():
  metadata = json.loads(SUNNYLINK_METADATA.read_text())

  for key in ("ControlsProfile", "ControlsProfileMigrationVersion", "LateralDemandStack"):
    assert key in metadata

  assert metadata["ControlsProfile"]["title"] == "Controls Profile"
  assert metadata["LateralDemandStack"]["title"] == "Lateral Demand Stack"
  assert {option["label"] for option in metadata["ControlsProfile"]["options"]} >= {
    "Sunnypilot Current",
    "Custom Recommended",
    "Custom 2.0",
    "Custom Experimental",
  }
  assert {option["label"] for option in metadata["LateralDemandStack"]["options"]} >= {
    "Sunnypilot Current",
    "Custom Recommended",
    "Custom 2.0",
    "Custom Experimental",
  }
