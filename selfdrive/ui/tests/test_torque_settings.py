from pathlib import Path


TORQUE_SETTINGS = Path(__file__).parents[1] / "sunnypilot/layouts/settings/steering_sub_layouts/torque_settings.py"


def test_control_calculation_hardening_toggle_is_in_torque_settings():
  source = TORQUE_SETTINGS.read_text()

  assert "_control_calculation_hardening_toggle = toggle_item_sp" in source
  assert 'param="ControlCalculationHardening"' in source
  assert "self._control_calculation_hardening_toggle," in source
