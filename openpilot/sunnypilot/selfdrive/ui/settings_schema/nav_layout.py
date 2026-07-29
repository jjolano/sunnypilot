"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Generic schema-driven panel with sub-panel navigation.

Renders a panel's top-level controls from the schema (same path as the flat
SchemaPanel) and turns each schema sub_panel into a nav button that opens a
supplied layout (typically the existing hand-coded sub-layout), gated by the
sub_panel's trigger_condition. This is the generalization of SchemaSteeringLayout;
panels with sub-panels (cruise, ...) mount it with their sub-layout factories.
"""
from collections.abc import Callable

import pyray as rl

from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.sunnypilot.widgets.list_view import simple_button_item_sp
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.scroller_tici import Scroller

from openpilot.sunnypilot.selfdrive.ui.settings_schema.rules import evaluate_rule, rules_pass
from openpilot.sunnypilot.selfdrive.ui.settings_schema.schema_loader import get_panel, load_schema, plan_page
from openpilot.sunnypilot.selfdrive.ui.settings_schema.widgets import (
  SectionHeaderSP, build_control, live_rule_context, placeholder_item,
)

_TOP_LEVEL = 0


class SchemaNavLayout(Widget):
  """Schema-driven panel whose sub_panels open supplied (hand-coded) sub-layouts.

  `subpanel_factories` maps schema sub_panel id -> a callable taking a back
  callback and returning the sub-layout Widget. Sub_panels without an entry get
  no nav button (their controls live only in the sub-layout, or are omitted).
  """
  def __init__(self, panel: str | dict, subpanel_factories: dict[str, Callable[[Callable], Widget]]):
    super().__init__()
    if isinstance(panel, str):
      panel = get_panel(load_schema(), panel)
      if panel is None:
        raise ValueError("schema panel not found")

    self._current = _TOP_LEVEL
    self._sub_layouts: dict[int, Widget] = {}
    self._sub_index: dict[str, int] = {}
    for i, (sp_id, factory) in enumerate(subpanel_factories.items(), start=1):
      self._sub_index[sp_id] = i
      self._sub_layouts[i] = factory(lambda: self._set_current(_TOP_LEVEL))

    self._controls: list[tuple[dict, object]] = []   # (item, ListItemSP) — rule-driven
    self._nav_gates: list[tuple[dict, object]] = []  # (trigger_rule, button)
    self.unsupported: list[dict] = []

    items = []
    for entry in plan_page(panel, with_sections=True):
      if entry["kind"] == "section":
        items.append(SectionHeaderSP(entry["title"]))
        continue
      if entry["kind"] == "control":
        item = entry["item"]
        control = build_control(item, self.unsupported, lambda: ui_state.is_metric)
        if control is None:
          control = placeholder_item(item)
        else:
          self._controls.append((item, control))
        items.append(control)
      elif entry["kind"] == "subpanel":
        target = self._sub_index.get(entry["id"])
        if target is None:
          continue
        button = simple_button_item_sp(
          button_text=lambda label=entry["label"]: tr(label),
          button_width=850,
          callback=lambda t=target: self._set_current(t),
        )
        items.append(button)
        if entry.get("trigger"):
          self._nav_gates.append((entry["trigger"], button))

    self._scroller = Scroller(items, line_separator=False, spacing=0)

  def _set_current(self, idx: int):
    self._current = idx

  def _update_state(self):
    super()._update_state()
    ctx = live_rule_context()
    for item, control in self._controls:
      control.set_visible(rules_pass(item.get("visibility"), ctx))
      if control.action_item is not None:
        control.action_item.set_enabled(rules_pass(item.get("enablement"), ctx))
      sync_hook = getattr(control, "sync_hook", None)
      if callable(sync_hook):
        sync_hook()
    for trigger, button in self._nav_gates:
      if button.action_item is not None:
        button.action_item.set_enabled(evaluate_rule(trigger, ctx))

  def _render(self, rect: rl.Rectangle):
    sub_layout = self._sub_layouts.get(self._current)
    if sub_layout is not None:
      sub_layout.render(rect)
    else:
      self._scroller.render(rect)

  def show_event(self):
    self._set_current(_TOP_LEVEL)
    self._scroller.show_event()
