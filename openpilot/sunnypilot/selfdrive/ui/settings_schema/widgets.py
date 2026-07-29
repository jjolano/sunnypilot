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

Scope (prototype): toggle, numeric option, sequential-int multiple_button, and
homogeneous string multiple_button selectors are rendered fully. Float/mixed/
dynamic selectors and custom widgets remain the declared escape-hatch residue —
they are collected in `unsupported` rather than mis-rendered, matching the
"named provider / custom widget registry" design.
"""
from __future__ import annotations

from collections.abc import Callable

import pyray as rl

from openpilot.system.ui.lib.application import FontWeight, gui_app
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.sunnypilot.lib.styles import style
from openpilot.system.ui.sunnypilot.widgets.list_view import (
  ListItemSP,
  button_item_sp,
  multiple_button_item_sp,
  option_item_sp,
  toggle_item_sp,
)
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.scroller_tici import Scroller

from openpilot.sunnypilot.selfdrive.ui.settings_schema.encoding import (
  contiguous_int_options, encode_numeric_option, homogeneous_string_options, sequential_int_labels, string_option_index, value_mapped_option,
)
from openpilot.sunnypilot.selfdrive.ui.settings_schema.registry import custom_widget_factory, resolve_options
from openpilot.sunnypilot.selfdrive.ui.settings_schema.rules import RuleContext, rules_pass
from openpilot.sunnypilot.selfdrive.ui.settings_schema.schema_loader import get_panel, iter_items, load_schema


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

  # Key-based override: if a custom widget factory is registered for this
  # item's key, it takes precedence over the declared widget type. This lets
  # device-specific side effects (file creation, complex state) override a
  # toggle/button without changing the schema's declared widget type.
  key_override = custom_widget_factory(key)
  if key_override is not None:
    return key_override(item)

  if widget == "toggle":
    needs_cycle = bool(item.get("needs_onroad_cycle"))

    if needs_cycle:
      # needs_onroad_cycle: augment description, block while engaged, write
      # OnroadCycleRequested on change — matching the retired hand-coded toggles.
      def _cycle_desc(base=desc):
        warning = tr("Changing this setting will restart sunnypilot if the car is powered on.")
        return (base + " " + warning) if base else warning

      def _on_cycle_change(_state: bool):
        from openpilot.common.params import Params
        Params().put_bool("OnroadCycleRequested", True)

      control = toggle_item_sp(title=title, description=_cycle_desc, param=key, callback=_on_cycle_change)

      def _cycle_sync(action=control.action_item, pkey: str = key):
        from openpilot.common.params import Params, UnknownKeyName
        from openpilot.selfdrive.ui.ui_state import ui_state
        params = Params()
        try:
          locked = params.get_bool(pkey + "Lock")
        except UnknownKeyName:
          locked = False
        action.set_enabled(not locked and not ui_state.engaged)

      control.sync_hook = _cycle_sync  # type: ignore[attr-defined]
      return control

    control = toggle_item_sp(title=title, description=desc, param=key)

    def _lock_sync(action=control.action_item, pkey: str = key):
      from openpilot.common.params import Params, UnknownKeyName
      params = Params()
      try:
        locked = params.get_bool(pkey + "Lock")
      except UnknownKeyName:
        locked = False
      if locked:
        action.set_enabled(False)

    control.sync_hook = _lock_sync  # type: ignore[attr-defined]
    return control

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
      string_enum = homogeneous_string_options({**item, "options": options}) if options else None
      if string_enum is not None:
        values = string_enum.values
        from openpilot.common.params import Params
        selected_index = string_option_index(Params().get(key, return_default=True), string_enum, key)

        def on_change(index: int, pkey: str = key, vals: list[str] = values):
          if 0 <= index < len(vals):
            from openpilot.common.params import Params
            Params().put(pkey, vals[index])

        def sync_selected(action, pkey: str = key, vals: list[str] = values):
          if action is not None:
            from openpilot.common.params import Params
            action.set_selected_button(string_option_index(Params().get(pkey, return_default=True), string_enum, pkey))

        item_out = multiple_button_item_sp(
          title=title, description=desc,
          buttons=[_t(label) for label in string_enum.labels],
          selected_index=selected_index,
          callback=on_change,
        )
        item_out.sync_hook = lambda: sync_selected(item_out.action_item)  # type: ignore[attr-defined]
        return item_out
      unsupported.append(item)  # float/mixed/dynamic enum -> custom selector
      return None
    return multiple_button_item_sp(
      title=title, description=desc, param=key,
      buttons=[_t(label) for label in labels],
    )

  if widget == "option":
    options = resolve_options(item)
    if options is not None:
      norm = {**item, "options": options}
      enum = contiguous_int_options(norm)
      if enum is not None:
        labels = enum.labels_by_value
        return option_item_sp(
          title=title, param=key, description=desc,
          min_value=enum.min_value, max_value=enum.max_value, value_change_step=1,
          label_callback=lambda x: tr(labels.get(x, str(x))),
        )
      vmap = value_mapped_option(norm)
      if vmap is not None:
        by_value = vmap.labels_by_value
        return option_item_sp(
          title=title, param=key, description=desc,
          min_value=vmap.min_value, max_value=vmap.max_value, value_change_step=1,
          value_map=vmap.value_map, label_callback=lambda v: tr(by_value.get(v, str(v))),
        )
      unsupported.append(item)  # non-integer values -> custom selector
      return None
    return _numeric_option(item, is_metric_fn)

  if widget == "button":
    # Button widgets route through the custom-widget registry using the
    # `action` field as the component id. Pages register factories for
    # their button actions (reboot, download, clear cache, etc.).
    action_id = item.get("action", "")
    factory = custom_widget_factory(action_id)
    if factory is None:
      unsupported.append(item)
      return None
    return factory(item)

  if widget == "info":
    # Read-only display of a param value. Uses ButtonActionSP in disabled
    # mode with a sync_hook that reads the param each frame.
    from openpilot.common.params import Params

    def _info_value(pkey: str = key):
      val = Params().get(pkey, return_default=True)
      if isinstance(val, bytes):
        val = val.decode("utf-8", "replace")
      return val or ""

    control = button_item_sp(title=title, button_text=_info_value, description=desc, enabled=False)

    def _info_sync(action=control.action_item, pkey: str = key):
      val = Params().get(pkey, return_default=True)
      if isinstance(val, bytes):
        val = val.decode("utf-8", "replace")
      action.set_value(val or "")

    control.sync_hook = _info_sync  # type: ignore[attr-defined]
    return control

  # unknown widgets -> custom registry territory.
  unsupported.append(item)
  return None


class SectionHeaderSP(Widget):
  """A non-interactive section header: an accent, upper-cased group title.

  Lets the schema's section titles structure a panel visually — essential for
  consolidated pages (e.g. Driving's Lateral / Longitudinal grouping).
  """
  HEIGHT = 120

  def __init__(self, title: str | Callable[[], str]):
    super().__init__()
    self._title = title
    self.set_rect(rl.Rectangle(0, 0, 0, self.HEIGHT))

  def _render(self, rect: rl.Rectangle):
    title = self._title() if callable(self._title) else self._title
    font = gui_app.font(FontWeight.BOLD)
    rl.draw_text_ex(font, title.upper(), rl.Vector2(rect.x + style.ITEM_PADDING, rect.y + 55), 42, 2, style.BLUE)
    line_y = int(rect.y + self.HEIGHT - 14)
    rl.draw_line(int(rect.x + style.ITEM_PADDING), line_y,
                 int(rect.x + rect.width - style.ITEM_PADDING), line_y, rl.Color(45, 45, 45, 255))


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
      sync_hook = getattr(control, "sync_hook", None)
      if callable(sync_hook):
        sync_hook()

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


class SchemaPanelLayout(Widget):
  """A flat schema-driven settings panel mounted directly in the sidebar.

  For panels with no sub-panels (visuals, display, toggles, ...): builds every
  control from the named compiled-schema panel and drives enable/visible from
  the rules each frame. Panels that have sub-panels use a dedicated layout
  (e.g. SchemaSteeringLayout). `unsupported` exposes any control the renderer
  could not build (escape-hatch residue) for callers that want to assert on it.
  """
  def __init__(self, panel_id: str):
    super().__init__()
    from openpilot.selfdrive.ui.ui_state import ui_state
    panel = get_panel(load_schema(), panel_id)
    if panel is None:
      raise ValueError(f"schema panel {panel_id!r} not found")
    self._panel = SchemaPanel(panel, live_rule_context, lambda: ui_state.is_metric)
    self.unsupported = self._panel.unsupported

  def _render(self, rect: rl.Rectangle):
    self._panel.render(rect)

  def show_event(self):
    self._panel.show_event()
