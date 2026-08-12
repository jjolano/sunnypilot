"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Parity for flat (sub-panel-free) panels converted to the schema-driven renderer
and mounted via SchemaPanelLayout. Each test asserts the schema panel declares
EXACTLY the controls the retired hand-coded panel had, with device-appropriate
widgets — so a conversion can't silently drop or re-type a control (the NNLC
class of regression). Pure/headless.
"""
from __future__ import annotations

from openpilot.sunnypilot.selfdrive.ui.settings_schema.encoding import (
  contiguous_int_options, sequential_int_labels, value_mapped_option,
)
from openpilot.sunnypilot.selfdrive.ui.settings_schema.schema_loader import find_item, get_panel, iter_items, load_schema

VISUALS = get_panel(load_schema(), "visuals")
DISPLAY = get_panel(load_schema(), "display")
TOGGLES = get_panel(load_schema(), "toggles")

# Transcribed from the retired visuals.py: 11 toggles + 2 enum button-rows.
VISUALS_EXPECTED = {
  "BlindSpot": "toggle", "TorqueBar": "toggle", "RainbowMode": "toggle",
  "StandstillTimer": "toggle", "RoadNameToggle": "toggle", "GreenLightAlert": "toggle",
  "LeadDepartAlert": "toggle", "TrueVEgoUI": "toggle", "HideVEgoUI": "toggle",
  "ShowTurnSignals": "toggle", "RocketFuel": "toggle",
  "ChevronInfo": "multiple_button", "DevUIInfo": "multiple_button",
}


def test_visuals_parity_exact_keys_and_widgets():
  got = {it["key"]: it["widget"] for it in iter_items(VISUALS) if "key" in it}
  assert got == VISUALS_EXPECTED


def test_visuals_enums_render_as_button_rows():
  # multiple_button with values 0..n-1 -> a real button row, not the escape hatch.
  for key in ("ChevronInfo", "DevUIInfo"):
    assert sequential_int_labels(find_item(VISUALS, key)) is not None


def test_visuals_chevron_gated_on_longitudinal():
  # Faithful to visuals.py: the chevron metric selector is enabled only with
  # sunnypilot longitudinal control.
  chevron = find_item(VISUALS, "ChevronInfo")
  rules = chevron.get("enablement") or []
  assert any(r.get("type") == "capability" and r.get("field") == "has_longitudinal_control" for r in rules)


# --- display: three steppers + screensaver (merged upstream) -----------------

def test_display_controls_are_option_steppers():
  got = {it["key"]: it["widget"] for it in iter_items(DISPLAY) if "key" in it}
  assert got == {
    "OnroadScreenOffBrightness": "option",
    "OnroadScreenOffTimer": "option",
    "InteractivityTimeout": "option",
    "ScreenSaverEnabled": "toggle",
    "ScreenSaverTimeout": "multiple_button",
  }


def test_display_brightness_contiguous_with_labels():
  enc = contiguous_int_options(find_item(DISPLAY, "OnroadScreenOffBrightness"))
  assert enc is not None
  assert (enc.min_value, enc.max_value) == (0, 22)
  assert enc.labels_by_value[0] == "Auto (Default)"
  assert enc.labels_by_value[2] == "Screen Off"
  assert enc.labels_by_value[22] == "100 %"


def test_display_timer_value_mapped_to_valid_seconds_only():
  # Matches ONROAD_BRIGHTNESS_TIMER_VALUES exactly — no invalid 0/"Always On".
  enc = value_mapped_option(find_item(DISPLAY, "OnroadScreenOffTimer"))
  assert enc is not None
  assert set(enc.value_map.values()) == {3, 5, 7, 10, 15, 30, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600}


def test_display_timer_gated_on_brightness_not_auto():
  # display.py:88 — timer enabled only when brightness is not AUTO(0)/AUTO_DARK(1).
  enc = value_mapped_option(find_item(DISPLAY, "InteractivityTimeout"))
  assert enc is not None and enc.value_map[1] == 0  # "Default"
  timer = find_item(DISPLAY, "OnroadScreenOffTimer")
  assert timer.get("enablement"), "timer must carry the brightness gate"


# --- toggles: core openpilot feature toggles ------------------------------

TOGGLES_EXPECTED = {
  "OpenpilotEnabledToggle": "toggle",
  "IsLdwEnabled": "toggle",
  "AlwaysOnDM": "toggle",
  "IsMetric": "toggle",
  "RecordFront": "toggle",
  "RecordAudio": "toggle",
}


def test_toggles_parity_exact_keys_and_widgets():
  got = {it["key"]: it["widget"] for it in iter_items(TOGGLES) if "key" in it}
  assert got == TOGGLES_EXPECTED


def test_toggles_onroad_cycle_items_marked():
  # These toggles need an onroad cycle to take effect — the schema must declare
  # needs_onroad_cycle so the renderer can show the restart warning and block
  # while engaged.
  for key in ("OpenpilotEnabledToggle", "RecordFront", "RecordAudio"):
    item = find_item(TOGGLES, key)
    assert item.get("needs_onroad_cycle") is True, f"{key} must declare needs_onroad_cycle"


def test_toggles_openpilot_enabled_gated_offroad():
  # Enable sunnypilot toggle is only adjustable offroad.
  item = find_item(TOGGLES, "OpenpilotEnabledToggle")
  rules = item.get("enablement") or []
  assert any(r.get("type") == "offroad_only" for r in rules), \
      "OpenpilotEnabledToggle must be gated offroad"
