"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

The consolidated Driving page: steering (Lateral) + cruise (Longitudinal) under two
group headers, with sub-panels and all controls preserved. Pure/headless.
"""
from __future__ import annotations

from openpilot.sunnypilot.selfdrive.ui.settings_schema.driving_panel import driving_panel_dict
from openpilot.sunnypilot.selfdrive.ui.settings_schema.schema_loader import plan_page

DRIVING = driving_panel_dict()
PLAN = plan_page(DRIVING, with_sections=True)
CONTROLS = {e["item"]["key"] for e in PLAN if e["kind"] == "control"}
SUBPANELS = {e["id"] for e in PLAN if e["kind"] == "subpanel"}
HEADERS = [e["title"] for e in PLAN if e["kind"] == "section"]


def test_exactly_two_group_headers():
  assert HEADERS == ["Lateral Control", "Longitudinal Control"]


def test_lateral_controls_present():
  assert {"Mads", "BlinkerPauseLateralControl", "EnforceTorqueControl",
          "NeuralNetworkLateralControl", "AutoLaneChangeTimer"} <= CONTROLS


def test_longitudinal_controls_present():
  assert {"ExperimentalMode", "DisengageOnAccelerator", "LongitudinalPersonality",
          "CustomAccIncrementsEnabled", "CustomLongitudinalEnabled", "CustomLongitudinalMode",
          "LeadAnticipationMode", "LeadPathClearanceMode", "LongitudinalDebugTraceMode",
          "SmartCruiseControlVision"} <= CONTROLS


def test_custom_longitudinal_mode_is_multiple_button():
  item = next(e["item"] for e in PLAN if e["kind"] == "control" and e["item"].get("key") == "CustomLongitudinalMode")
  assert item["widget"] == "multiple_button"
  assert [opt["value"] for opt in item["options"]] == ["acc", "e2e", "scc"]


def test_subpanels_preserved():
  assert {"mads_settings", "torque_settings", "speed_limit_settings"} <= SUBPANELS


def test_enable_prefix_dropped_from_toggle_titles():
  mads = next(e["item"] for e in PLAN
              if e["kind"] == "control" and e["item"].get("key") == "Mads")
  assert not mads["title"].startswith("Enable ")
  assert "Modular Assistive Driving System" in mads["title"]
