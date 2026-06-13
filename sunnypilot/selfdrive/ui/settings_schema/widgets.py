"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Render a schema panel into device widgets.

`SchemaPanel` is a generic, panel-agnostic Widget: it builds the ListItemSP
toolkit widgets from a compiled schema panel and, every frame, drives each
control's visible/enabled state from that control's declared visibility/
enablement rules. It is the whole device-side replacement for a hand-coded
panel's _initialize_items() + _update_state(): one ~40-line class instead of
~160 lines per panel.

Scope (prototype): toggle, numeric option, and sequential-int multiple_button
are rendered fully. Enumerated string/float selectors (e.g. the speed-aware
curve, the dialog-backed tune-version picker) and custom widgets are the
declared escape-hatch residue — they are collected in `unsupported` rather than
mis-rendered, matching the "named provider / custom widget registry" design.
"""
from __future__ import annotations

from collections.abc import Callable

import pyray as rl

from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.sunnypilot.widgets.list_view import (
  ListItemSP,
  multiple_button_item_sp,
  option_item_sp,
  toggle_item_sp,
)
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.scroller_tici import Scroller

from openpilot.sunnypilot.selfdrive.ui.settings_schema.encoding import (
  contiguous_int_options, encode_numeric_option, sequential_int_labels,
)
from openpilot.sunnypilot.selfdrive.ui.settings_schema.registry import custom_widget_factory, resolve_options
from openpilot.sunnypilot.selfdrive.ui.settings_schema.rules import RuleContext, rules_pass
from openpilot.sunnypilot.selfdrive.ui.settings_schema.schema_loader import iter_items


def _t(text: str) -> Callable[[], str]:
  """Defer translation to render time (titles/descriptions are callables)."""
  return lambda: tr(text)


def _unit_suffix(unit, is_metric_fn: Callable[[], bool]) -> Callable[[], str]:
  """Resolve a schema `unit` (str or {metric, imperial}) to a label suffix."""
  if isinstance(unit, dict):
    return lambda: " " + (unit["metric"] if is_metric_fn() else unit["imperial"])
  if unit:
    return lambda: " " + str(unit)
  return lambda: ""


def _numeric_option(item: dict, is_metric_fn: Callable[[], bool]) -> ListItemSP:
  """Build an OptionControlSP from a numeric `option` item (min/max/step).

  The int encoding is derived from the schema by encode_numeric_option() — one
  place, unit-tested, instead of the per-panel hand-encoding that drifted.
  """
  enc = encode_numeric_option(item)
  suffix = _unit_suffix(item.get("unit"), is_metric_fn)
  scale = enc.scale

  def label(v: int) -> str:
    real = v / scale if enc.use_float_scaling else v
    return f"{real:g}{suffix()}"

  return option_item_sp(
    title=_t(item.get("title", item["key"])), param=item["key"],
    description=item.get("description", ""),
    min_value=enc.min_value, max_value=enc.max_value, value_change_step=enc.value_change_step,
    use_float_scaling=enc.use_float_scaling, label_callback=label,
  )


def build_control(item: dict, unsupported: list[dict], is_metric_fn: Callable[[], bool]) -> ListItemSP | None:
  """Build one device widget from a schema item, or record it as unsupported."""
  widget = item.get("widget")
  key = item.get("key", "")
  title = _t(item.get("title", key))
  desc = item.get("description", "")

  if widget == "toggle":
    return toggle_item_sp(title=title, description=desc, param=key)

  if widget == "custom":
    factory = custom_widget_factory(item.get("component", ""))
    if factory is None:
      unsupported.append(item)  # no registered component for this id
      return None
    return factory(item)

  if widget == "multiple_button":
    options = resolve_options(item)
    labels = sequential_int_labels({**item, "options": options}) if options else None
    if labels is None:
      unsupported.append(item)  # string/float/non-sequential enum -> custom selector
      return None
    return multiple_button_item_sp(
      title=title, description=desc, param=key,
      buttons=[_t(label) for label in labels],
    )

  if widget == "option":
    options = resolve_options(item)
    if options is not None:
      enum = contiguous_int_options({**item, "options": options})
      if enum is None:
        unsupported.append(item)  # string/float/gapped values -> value-mapped or custom selector
        return None
      labels = enum.labels_by_value
      return option_item_sp(
        title=title, param=key, description=desc,
        min_value=enum.min_value, max_value=enum.max_value, value_change_step=1,
        label_callback=lambda x: tr(labels.get(x, str(x))),
      )
    return _numeric_option(item, is_metric_fn)

  # info / unknown widgets -> custom registry territory.
  unsupported.append(item)
  return None


def placeholder_item(item: dict) -> ListItemSP:
  """A visible, inert marker for a control whose widget can't be built yet.

  Rendered on-device instead of silently dropping the control, so the
  escape-hatch residue (string/float enums, dialog pickers, custom widgets) is
  obvious during validation rather than looking like a complete panel.
  """
  widget = item.get("widget", "?")
  return ListItemSP(
    title=_t(item.get("title", item.get("key", ""))),
    description=f"Pending schema-driven renderer (widget: {widget}).",
    description_visible=True,
  )


class SchemaPanel(Widget):
  """Generic device panel rendered from a compiled schema panel.

  Replaces a hand-coded layout's item creation, _update_state() rule logic, and
  render plumbing. `ctx_provider` returns a fresh RuleContext each frame (see
  live_rule_context()); inject a stub in tests.
  """
  def __init__(self, panel: dict, ctx_provider: Callable[[], RuleContext],
               is_metric_fn: Callable[[], bool] = lambda: False):
    super().__init__()
    self._ctx_provider = ctx_provider
    self._controls: list[tuple[dict, ListItemSP]] = []
    self.unsupported: list[dict] = []

    widgets: list[Widget] = []
    for item in iter_items(panel):
      control = build_control(item, self.unsupported, is_metric_fn)
      if control is None:
        continue
      self._controls.append((item, control))
      widgets.append(control)
    self._scroller = Scroller(widgets, line_separator=True, spacing=0)

  def _update_state(self):
    super()._update_state()
    ctx = self._ctx_provider()
    for item, control in self._controls:
      control.set_visible(rules_pass(item.get("visibility"), ctx))
      if control.action_item is not None:
        control.action_item.set_enabled(rules_pass(item.get("enablement"), ctx))

  def _render(self, rect: rl.Rectangle):
    self._scroller.render(rect)

  def show_event(self):
    self._scroller.show_event()


def live_rule_context() -> RuleContext:
  """Build a RuleContext from live device state.

  Reuses generate_capabilities() — the *same* capability computation the cloud
  schema is evaluated against (sunnylinkd.getParamsMetadata) — so device and
  cloud can never disagree on a capability value.
  """
  from openpilot.selfdrive.ui.ui_state import ui_state
  from openpilot.sunnypilot.sunnylink.capabilities import generate_capabilities
  return RuleContext(
    params=ui_state.params,
    capabilities=generate_capabilities(ui_state.params),
    is_offroad=ui_state.is_offroad(),
    is_engaged=ui_state.is_onroad(),
  )
