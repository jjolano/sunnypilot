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

from openpilot.sunnypilot.selfdrive.ui.settings_schema.encoding import sequential_int_labels
from openpilot.sunnypilot.selfdrive.ui.settings_schema.schema_loader import find_item, get_panel, iter_items, load_schema

VISUALS = get_panel(load_schema(), "visuals")

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
