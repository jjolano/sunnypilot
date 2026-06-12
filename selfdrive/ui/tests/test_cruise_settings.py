from pathlib import Path


CRUISE_SETTINGS = Path(__file__).parents[1] / "sunnypilot/layouts/settings/cruise.py"
SPEED_LIMIT_SETTINGS = Path(__file__).parents[1] / "sunnypilot/layouts/settings/cruise_sub_layouts/speed_limit_settings.py"
EXP_BUTTON = Path(__file__).parents[1] / "onroad/exp_button.py"
TOGGLES_SETTINGS = Path(__file__).parents[1] / "layouts/settings/toggles.py"
UI_STATE = Path(__file__).parents[1] / "sunnypilot/ui_state.py"
SELFDRIVED = Path(__file__).parents[1] / "../selfdrived/selfdrived.py"
CAR_INTERFACES = Path(__file__).parents[3] / "sunnypilot/selfdrive/car/interfaces.py"


def test_longitudinal_stack_selector_is_in_cruise_settings():
  source = CRUISE_SETTINGS.read_text()

  assert "self.longitudinal_stack_item = ListItemSP" in source
  assert "LongitudinalStack" in source
  assert "resolve_longitudinal_stack" in source
  assert "TreeOptionDialog" in source
  assert "self.longitudinal_stack_item," in source

def test_longitudinal_stack_selection_clears_controls_profile():
  source = CRUISE_SETTINGS.read_text()

  assert 'ui_state.params.remove("ControlsProfile")' in source


def test_planner_stack_selector_is_in_cruise_settings_and_validation_gated():
  source = CRUISE_SETTINGS.read_text()

  assert "self.planner_stack_item = ListItemSP" in source
  assert "PlannerStack" in source
  assert "PLANNER_STACK_VALIDATION_GATE_PARAM" in source
  assert "resolve_planner_stack" in source
  assert "SCENE_MEMORY_V1 not in resolution.available_stacks" in source
  assert "self.planner_stack_item," in source


def test_longitudinal_mode_selector_replaces_legacy_toggles():
  source = CRUISE_SETTINGS.read_text()

  assert "self.longitudinal_mode_item = multiple_button_item_sp" in source
  assert "LongitudinalMode" in source
  assert "buttons=[tr(\"ACC\"), tr(\"E2E\"), tr(\"SCC\")]" in source
  assert "param=\"LongitudinalMode\"" not in source
  assert "ExperimentalModeConfirmed" in source
  assert "ConfirmDialog" in source
  assert "SccCurveVisionEnabled" in source
  assert "SccCurveMapEnabled" in source
  assert "current_mode == LongitudinalMode.SCC" in source
  assert "param=\"DynamicExperimentalControl\"" not in source
  assert "param=\"SmartCruiseControlVision\"" not in source
  assert "param=\"SmartCruiseControlMap\"" not in source


def test_one_pedal_mode_is_custom_v2_gated():
  source = CRUISE_SETTINGS.read_text()

  assert "self.one_pedal_longitudinal_item = multiple_button_item_sp" in source
  assert "OnePedalLongitudinalMode" in source
  assert "resolution.resolved_stack == CUSTOM_V2" in source
  assert "ui_state.params.put_bool(\"OnroadCycleRequested\", True)" in source


def test_fast_lead_motion_is_custom_v2_offroad_cycle_toggle():
  source = CRUISE_SETTINGS.read_text()

  assert "self.fast_lead_motion_toggle = toggle_item_sp" in source
  assert "FastLeadMotionEvidenceEnabled" in source
  assert "resolution.resolved_stack == CUSTOM_V2" in source
  assert "self.fast_lead_motion_toggle.action_item.set_enabled(enabled)" in source
  assert "def _on_fast_lead_motion_changed" in source
  assert "ui_state.params.put_bool(\"OnroadCycleRequested\", True)" in source


def test_longitudinal_stack_selector_is_offroad_gated_and_requests_cycle():
  source = CRUISE_SETTINGS.read_text()

  assert "show_advanced = ui_state.params.get_bool(\"ShowAdvancedControls\")" in source
  assert "self.longitudinal_stack_item.set_visible(show_advanced)" in source
  assert "self.longitudinal_stack_item.action_item.set_enabled(has_long and ui_state.is_offroad())" in source
  assert "ui_state.params.put(\"LongitudinalStack\", selected_ref)" in source
  assert "ui_state.params.put_bool(\"OnroadCycleRequested\", True)" in source


def test_planner_stack_selector_is_offroad_gated_and_requests_cycle():
  source = CRUISE_SETTINGS.read_text()

  assert "self.planner_stack_item.action_item.set_enabled(has_long and ui_state.is_offroad())" in source
  assert "ui_state.params.put(PLANNER_STACK_PARAM, selected_ref)" in source
  assert "ui_state.params.put_bool(\"OnroadCycleRequested\", True)" in source


def test_decision_layer_is_not_user_toggle():
  source = CRUISE_SETTINGS.read_text()

  assert "LongitudinalDecisionLayer" not in source
  assert "longitudinal_decision_layer_toggle" not in source


def test_onroad_experimental_button_writes_longitudinal_mode_only():
  source = EXP_BUTTON.read_text()

  assert "put(\"LongitudinalMode\"" in source
  assert "str(int(longitudinal_mode))" not in source
  assert "ExperimentalModeConfirmed" in source
  assert "put_bool(\"ExperimentalMode\"" not in source


def test_speed_limit_assist_is_unavailable_in_acc_mode():
  source = SPEED_LIMIT_SETTINGS.read_text()

  assert "requested_mode_from_params" in source
  assert "has_icbm or (has_long and longitudinal_mode != LongitudinalMode.ACC)" in source


def test_generic_toggles_do_not_expose_experimental_mode():
  source = TOGGLES_SETTINGS.read_text()

  assert "\"ExperimentalMode\": (" not in source
  assert "_handle_experimental_mode_toggle" not in source


def test_ui_constraints_clear_scc_curve_params_without_longitudinal():
  source = UI_STATE.read_text()

  assert "self.params.remove(\"LongitudinalMode\")" in source
  assert "self.params.remove(\"SccCurveVisionEnabled\")" in source
  assert "self.params.remove(\"SccCurveMapEnabled\")" in source
  assert "if CP is not None:\n        self.params.remove(\"LongitudinalMode\")" not in source


def test_runtime_cleanup_clears_longitudinal_mode_params_without_longitudinal():
  sources = (SELFDRIVED.read_text(), CAR_INTERFACES.read_text())

  for source in sources:
    assert "remove(\"LongitudinalMode\")" in source
    assert "remove(\"SccCurveVisionEnabled\")" in source
    assert "remove(\"SccCurveMapEnabled\")" in source
