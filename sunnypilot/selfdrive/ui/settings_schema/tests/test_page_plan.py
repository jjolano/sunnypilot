"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Proof that plan_page() flattens the steering panel into the right top-level
render order — controls interleaved with sub-panel nav buttons — without
inlining sub-panel contents. This is the structure the live SchemaSteeringLayout
consumes; verifying it headless de-risks the (display-only) pyray glue.
"""
from __future__ import annotations

from openpilot.sunnypilot.selfdrive.ui.settings_schema.schema_loader import get_panel, load_schema, plan_page

STEERING = get_panel(load_schema(), "steering")
PLAN = plan_page(STEERING)


def _keys(plan):
  return [(e["kind"], e.get("item", {}).get("key") or e.get("id")) for e in plan]


def test_plan_order_matches_device_top_level():
  assert _keys(PLAN) == [
    ("control", "Mads"),
    ("subpanel", "mads_settings"),
    ("control", "BlinkerPauseLateralControl"),
    ("control", "BlinkerMinLateralControlSpeed"),
    ("control", "BlinkerLateralReengageDelay"),
    ("control", "EnforceTorqueControl"),
    ("control", "NeuralNetworkLateralControl"),
    ("subpanel", "torque_settings"),
    ("control", "AutoLaneChangeTimer"),
    ("control", "AutoLaneChangeBsmDelay"),
    ("control", "CustomLateralDemandEnabled"),
    ("control", "LaneCenteringAssistEnabled"),
    ("control", "StraightPathStabilizationMode"),
    ("control", "LaneRateDampingMode"),
    ("control", "CurveMemoryEnabled"),       # sub_items, inlined after their parent toggle
  ]


def test_subpanels_carry_trigger_conditions():
  subpanels = {e["id"]: e for e in PLAN if e["kind"] == "subpanel"}
  # mads_settings shows when Mads is on; torque_settings when EnforceTorqueControl is on.
  assert subpanels["mads_settings"]["trigger"] == {"type": "param", "key": "Mads", "equals": True}
  assert subpanels["torque_settings"]["trigger"] == \
      {"type": "param", "key": "EnforceTorqueControl", "equals": True}


def test_subpanel_contents_not_inlined():
  # Controls that live *inside* sub-panels must not appear in the top-level plan.
  control_keys = {e["item"]["key"] for e in PLAN if e["kind"] == "control"}
  assert "LiveTorqueParamsToggle" not in control_keys      # inside torque_settings
  assert "MadsMainCruiseAllowed" not in control_keys       # inside mads_settings


def test_blinker_subitems_are_inlined_after_parent():
  # sub_items (not sub_panels) DO render inline, right after their parent toggle.
  keys = [e["item"]["key"] for e in PLAN if e["kind"] == "control"]
  i = keys.index("BlinkerPauseLateralControl")
  assert keys[i + 1:i + 3] == ["BlinkerMinLateralControlSpeed", "BlinkerLateralReengageDelay"]
