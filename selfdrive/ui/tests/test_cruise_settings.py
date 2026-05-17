from pathlib import Path


CRUISE_SETTINGS = Path(__file__).parents[1] / "sunnypilot/layouts/settings/cruise.py"


def test_longitudinal_stack_selector_is_in_cruise_settings():
  source = CRUISE_SETTINGS.read_text()

  assert "self.longitudinal_stack_item = ListItemSP" in source
  assert "LongitudinalStack" in source
  assert "resolve_longitudinal_stack" in source
  assert "TreeOptionDialog" in source
  assert "self.longitudinal_stack_item," in source


def test_longitudinal_stack_selector_is_offroad_gated_and_requests_cycle():
  source = CRUISE_SETTINGS.read_text()

  assert "self.longitudinal_stack_item.action_item.set_enabled(has_long and ui_state.is_offroad())" in source
  assert "ui_state.params.put(\"LongitudinalStack\", selected_ref)" in source
  assert "ui_state.params.put_bool(\"OnroadCycleRequested\", True)" in source


def test_decision_layer_is_not_user_toggle():
  source = CRUISE_SETTINGS.read_text()

  assert "LongitudinalDecisionLayer" not in source
  assert "longitudinal_decision_layer_toggle" not in source
