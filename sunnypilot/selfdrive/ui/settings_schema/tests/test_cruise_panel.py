"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Parity for the cruise panel rendered via SchemaNavLayout: the top-level controls
(including the consolidated longitudinal settings the owner chose to keep here)
render inline, the custom-ACC increments are inlined sub_items, and the speed-limit
settings stay a delegated sub-panel. Pure/headless.
"""
from __future__ import annotations

from openpilot.sunnypilot.selfdrive.ui.settings_schema.encoding import value_mapped_option
from openpilot.sunnypilot.selfdrive.ui.settings_schema.schema_loader import find_item, get_panel, load_schema, plan_page

CRUISE = get_panel(load_schema(), "cruise")
PLAN = plan_page(CRUISE)
CONTROLS = {e["item"]["key"]: e["item"]["widget"] for e in PLAN if e["kind"] == "control"}
SUBPANELS = [e["id"] for e in PLAN if e["kind"] == "subpanel"]


def test_cruise_top_level_controls_render_inline():
  for key in ("ExperimentalMode", "DynamicExperimentalControl", "DisengageOnAccelerator",
              "IntelligentCruiseButtonManagement", "CustomAccIncrementsEnabled",
              "CustomLongitudinalEnabled", "SmartCruiseControlVision", "SmartCruiseControlMap"):
    assert CONTROLS.get(key) == "toggle", f"{key} missing/wrong widget"
  assert CONTROLS.get("LongitudinalPersonality") == "multiple_button"
  assert CONTROLS.get("CustomLongitudinalMode") == "multiple_button"
  assert CONTROLS.get("LeadAnticipationMode") == "multiple_button"
  assert CONTROLS.get("LeadPathClearanceMode") == "multiple_button"
  assert CONTROLS.get("LongitudinalDebugTraceMode") == "multiple_button"
  assert CONTROLS.get("CutInBrakeAssistMode") == "multiple_button"
  assert CONTROLS.get("CurveSpeedConfidenceMode") == "multiple_button"
  assert CONTROLS.get("StandstillReleaseConfidenceMode") == "multiple_button"


def test_custom_acc_increments_inlined_as_subitems():
  # Inlined to match the device (not behind a sub-panel button).
  assert CONTROLS.get("CustomAccShortPressIncrement") == "option"
  assert CONTROLS.get("CustomAccLongPressIncrement") == "option"


def test_custom_longitudinal_mode_is_string_multiple_button():
  enabled = find_item(CRUISE, "CustomLongitudinalEnabled")
  assert enabled is not None and enabled["widget"] == "toggle"
  item = find_item(CRUISE, "CustomLongitudinalMode")
  assert item is not None
  assert "options" in item
  assert [opt["value"] for opt in item["options"]] == ["acc", "e2e", "scc"]


def test_lead_anticipation_mode_is_string_multiple_button():
  item = find_item(CRUISE, "LeadAnticipationMode")
  assert item is not None
  assert item["widget"] == "multiple_button"
  assert [opt["value"] for opt in item["options"]] == ["off", "shadow", "apply"]
  assert [opt["label"] for opt in item["options"]] == ["Off", "Monitor only", "Apply lead smoothing"]


def test_lead_path_clearance_mode_is_string_multiple_button():
  item = find_item(CRUISE, "LeadPathClearanceMode")
  assert item is not None
  assert item["widget"] == "multiple_button"
  assert [opt["value"] for opt in item["options"]] == ["off", "shadow"]
  assert [opt["label"] for opt in item["options"]] == ["Off", "Monitor only"]


def test_longitudinal_debug_trace_mode_is_string_multiple_button():
  item = find_item(CRUISE, "LongitudinalDebugTraceMode")
  assert item is not None
  assert item["widget"] == "multiple_button"
  assert [opt["value"] for opt in item["options"]] == ["off", "log"]
  assert [opt["label"] for opt in item["options"]] == ["Off", "Log"]


def test_shadow_observability_modes_are_off_shadow_only():
  for key in ("CutInBrakeAssistMode", "CurveSpeedConfidenceMode", "StandstillReleaseConfidenceMode"):
    item = find_item(CRUISE, key)
    assert item is not None
    assert item["widget"] == "multiple_button"
    assert [opt["value"] for opt in item["options"]] == ["off", "shadow"]
    assert [opt["label"] for opt in item["options"]] == ["Off", "Monitor only"]


def test_speed_limit_is_a_delegated_subpanel():
  assert "speed_limit_settings" in SUBPANELS
  # its controls are reached via the sub-layout, not inlined at the top level
  assert "SpeedLimitMode" not in CONTROLS
  assert "SpeedLimitValueOffset" not in CONTROLS


def test_increment_encodings():
  short = find_item(CRUISE, "CustomAccShortPressIncrement")
  assert "options" not in short and short["min"] == 1 and short["max"] == 10  # plain numeric stepper

  long = value_mapped_option(find_item(CRUISE, "CustomAccLongPressIncrement"))
  assert long is not None
  assert long.value_map == {1: 1, 2: 5, 3: 10}  # 3-step stepper storing 1/5/10
