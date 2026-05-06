from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
TORQUE_SETTINGS = REPO_ROOT / "selfdrive" / "ui" / "sunnypilot" / "layouts" / "settings" / "steering_sub_layouts" / "torque_settings.py"
CRUISE_SETTINGS = REPO_ROOT / "selfdrive" / "ui" / "sunnypilot" / "layouts" / "settings" / "cruise.py"
REMOVED_MASS_DRAG_PARAMS = (
  "LongLearnedMassDragToggle",
  "LongLearnedMassDragApplyToggle",
  "LongLearnedKForce",
  "LongLearnedCDrag",
)
REMOVED_RESPONSE_CURVE_PARAMS = (
  "LongLearnedResponseCurveToggle",
  "LongLearnedResponseCurveApplyToggle",
  "LongLearnedResponseOffsets",
)
REMOVED_RESPONSE_CURVE_ATTRIBUTES = (
  "long_learned_response_curve_toggle",
  "long_learned_response_curve_apply_toggle",
)


def _assert_toggle_param(source: str, param: str):
  assert f'param="{param}"' in source, f"{param} is not exposed as a settings toggle"


def test_lateral_live_learning_toggles_are_exposed_in_torque_settings():
  source = TORQUE_SETTINGS.read_text(encoding="utf-8")

  _assert_toggle_param(source, "LiveTorqueSpeedAdaptiveToggle")
  _assert_toggle_param(source, "LiveTorqueSpeedAdaptiveApplyToggle")


def test_mass_drag_learning_toggles_are_not_exposed_in_cruise_settings():
  source = CRUISE_SETTINGS.read_text(encoding="utf-8")

  for param in REMOVED_MASS_DRAG_PARAMS:
    assert param not in source


def test_response_curve_learning_toggles_are_not_exposed():
  source = CRUISE_SETTINGS.read_text(encoding="utf-8")

  for param in REMOVED_RESPONSE_CURVE_PARAMS:
    assert param not in source

  for attr in REMOVED_RESPONSE_CURVE_ATTRIBUTES:
    assert attr not in source
