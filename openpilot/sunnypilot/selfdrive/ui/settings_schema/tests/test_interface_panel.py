"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

The consolidated Interface page: Visuals (On-Road Display) + Display (Screen) under
two group headers, flat (no sub-panels), all controls preserved. Pure/headless.
"""
from __future__ import annotations

from openpilot.sunnypilot.selfdrive.ui.settings_schema.interface_panel import interface_panel_dict
from openpilot.sunnypilot.selfdrive.ui.settings_schema.schema_loader import plan_page

INTERFACE = interface_panel_dict()
PLAN = plan_page(INTERFACE, with_sections=True)
CONTROLS = {e["item"]["key"] for e in PLAN if e["kind"] == "control"}
HEADERS = [e["title"] for e in PLAN if e["kind"] == "section"]


def test_exactly_two_group_headers():
  assert HEADERS == ["On-Road Display", "Screen"]


def test_visuals_controls_present():
  assert {"BlindSpot", "ChevronInfo", "DevUIInfo", "RocketFuel", "ShowTurnSignals"} <= CONTROLS


def test_display_controls_present():
  assert {"OnroadScreenOffBrightness", "OnroadScreenOffTimer", "InteractivityTimeout"} <= CONTROLS


def test_flat_no_subpanels():
  assert not [e for e in PLAN if e["kind"] == "subpanel"]
