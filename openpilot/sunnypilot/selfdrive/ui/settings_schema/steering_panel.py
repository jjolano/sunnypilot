"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Schema-driven steering panel — the production steering settings panel.

The top-level controls and their enable/visible behavior are driven entirely by
the compiled settings_ui.json: there is no hand-written _initialize_items() or
_update_state() rule logic here. Sub-panels (MADS, Torque) reuse the existing
hand-coded layouts, reached via nav buttons gated by the schema's
trigger_condition. Any control the renderer can't build yet appears as a visible
"pending" placeholder; the steering top level currently has none.

Structurally a near-mirror of the now-retired hand-coded steering layout so
rendering matches by construction; the difference is where the panel's shape
comes from (schema vs hand-coded).
"""
from enum import IntEnum

import pyray as rl

from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.selfdrive.ui.sunnypilot.layouts.settings.steering_sub_layouts.mads_settings import MadsSettingsLayout
from openpilot.selfdrive.ui.sunnypilot.layouts.settings.steering_sub_layouts.torque_settings import TorqueSettingsLayout
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.sunnypilot.widgets.list_view import simple_button_item_sp
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.scroller_tici import Scroller

from openpilot.sunnypilot.selfdrive.ui.settings_schema.rules import evaluate_rule, rules_pass
from openpilot.sunnypilot.selfdrive.ui.settings_schema.schema_loader import get_panel, load_schema, plan_page
from openpilot.sunnypilot.selfdrive.ui.settings_schema.widgets import (
  build_control, live_rule_context, placeholder_item,
)


class PanelType(IntEnum):
  STEERING = 0
  MADS = 1
  TORQUE_CONTROL = 2


# Schema sub_panel id -> the local panel it opens (reusing the hand-coded layout).
SUB_PANEL_TARGETS = {
  "mads_settings": PanelType.MADS,
  "torque_settings": PanelType.TORQUE_CONTROL,
}


class SchemaSteeringLayout(Widget):
  def __init__(self):
    super().__init__()
    self._current_panel = PanelType.STEERING
    self._sub_layouts = {
      PanelType.MADS: MadsSettingsLayout(lambda: self._set_current_panel(PanelType.STEERING)),
      PanelType.TORQUE_CONTROL: TorqueSettingsLayout(lambda: self._set_current_panel(PanelType.STEERING)),
    }

    self._controls: list[tuple[dict, object]] = []      # (item, ListItemSP) — rule-driven
    self._nav_gates: list[tuple[dict, object]] = []     # (trigger_rule, button)
    self.unsupported: list[dict] = []

    items = []
    panel = get_panel(load_schema(), "steering")
    for entry in plan_page(panel):
      if entry["kind"] == "control":
        item = entry["item"]
        control = build_control(item, self.unsupported, lambda: ui_state.is_metric)
        if control is None:
          control = placeholder_item(item)        # visible escape-hatch marker
        else:
          self._controls.append((item, control))  # only real controls get rule-driven
        items.append(control)
      elif entry["kind"] == "subpanel":
        target = SUB_PANEL_TARGETS.get(entry["id"])
        if target is None:
          continue
        button = simple_button_item_sp(
          button_text=lambda label=entry["label"]: tr(label),
          button_width=850,
          callback=lambda t=target: self._set_current_panel(t),
        )
        items.append(button)
        if entry.get("trigger"):
          self._nav_gates.append((entry["trigger"], button))

    self._scroller = Scroller(items, line_separator=False, spacing=0)

  def _set_current_panel(self, panel: PanelType):
    self._current_panel = panel

  def _update_state(self):
    super()._update_state()
    ctx = live_rule_context()
    for item, control in self._controls:
      control.set_visible(rules_pass(item.get("visibility"), ctx))
      if control.action_item is not None:
        control.action_item.set_enabled(rules_pass(item.get("enablement"), ctx))
    for trigger, button in self._nav_gates:
      if button.action_item is not None:
        button.action_item.set_enabled(evaluate_rule(trigger, ctx))

  def _render(self, rect: rl.Rectangle):
    sub_layout = self._sub_layouts.get(self._current_panel)
    if sub_layout is not None:
      sub_layout.render(rect)
    else:
      self._scroller.render(rect)

  def show_event(self):
    self._set_current_panel(PanelType.STEERING)
    self._scroller.show_event()
