from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
TORQUE_SETTINGS = REPO_ROOT / "selfdrive" / "ui" / "sunnypilot" / "layouts" / "settings" / "steering_sub_layouts" / "torque_settings.py"
CRUISE_SETTINGS = REPO_ROOT / "selfdrive" / "ui" / "sunnypilot" / "layouts" / "settings" / "cruise.py"
LONGITUDINAL_LIVE_LEARNING_TOGGLES = (
  "LongLearnedMassDragToggle",
  "LongLearnedMassDragApplyToggle",
  "LongLearnedResponseCurveToggle",
  "LongLearnedResponseCurveApplyToggle",
)


def _assert_toggle_param(source: str, param: str):
  assert f'param="{param}"' in source, f"{param} is not exposed as a settings toggle"


def test_lateral_live_learning_toggles_are_exposed_in_torque_settings():
  source = TORQUE_SETTINGS.read_text(encoding="utf-8")

  _assert_toggle_param(source, "LiveTorqueSpeedAdaptiveToggle")
  _assert_toggle_param(source, "LiveTorqueSpeedAdaptiveApplyToggle")


def test_longitudinal_live_learning_toggles_are_exposed_in_cruise_settings():
  source = CRUISE_SETTINGS.read_text(encoding="utf-8")

  for param in LONGITUDINAL_LIVE_LEARNING_TOGGLES:
    _assert_toggle_param(source, param)


def test_disabled_longitudinal_live_learning_toggles_preserve_saved_preferences():
  source = CRUISE_SETTINGS.read_text(encoding="utf-8")

  for param in LONGITUDINAL_LIVE_LEARNING_TOGGLES:
    assert f'ui_state.params.remove("{param}")' not in source
