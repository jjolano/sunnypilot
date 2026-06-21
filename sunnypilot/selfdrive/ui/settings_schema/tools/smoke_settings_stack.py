#!/usr/bin/env python3
"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Xvfb smoke test for the feature-flagged settings stack.

Usage:

  xvfb-run -a env LIBGL_ALWAYS_SOFTWARE=1 OFFSCREEN=1 \
    uv run python -m openpilot.sunnypilot.selfdrive.ui.settings_schema.tools.smoke_settings_stack
"""
from __future__ import annotations

from dataclasses import dataclass

import pyray as rl

from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.widgets import Widget

from openpilot.sunnypilot.selfdrive.ui.settings_schema.search import SearchRecord
from openpilot.sunnypilot.selfdrive.ui.settings_schema.search_view import SearchLayout
from openpilot.selfdrive.ui.sunnypilot.layouts.settings.settings_stack import SettingsStackLayout
from openpilot.selfdrive.ui.sunnypilot.layouts.settings.settings_stack import search_result_context_label
from openpilot.selfdrive.ui.sunnypilot.layouts.settings.settings_stack import SettingsRowItem
from openpilot.selfdrive.ui.sunnypilot.layouts.settings.settings_stack import _load_icon_texture
from openpilot.sunnypilot.selfdrive.ui.settings_schema.schema_loader import load_schema


class DummyWidget(Widget):
  def __init__(self, name: str):
    super().__init__()
    self.name = name

  def _render(self, rect: rl.Rectangle):
    pass


@dataclass
class DummyTexture:
  width: int = 64
  height: int = 64


class DummyRegistry:
  def __init__(self):
    self.requested: list[str] = []

  def get(self, key: str) -> Widget:
    self.requested.append(key)
    return DummyWidget(key)


def _render_stack(stack: SettingsStackLayout):
  rect = rl.Rectangle(0, 0, 2200, 1400)
  rl.begin_drawing()
  rl.clear_background(rl.Color(0, 0, 0, 255))
  stack.render(rect)
  rl.end_drawing()


def _click(stack: SettingsStackLayout, rect: rl.Rectangle):
  stack._handle_mouse_release(rl.Vector2(rect.x + rect.width / 2, rect.y + rect.height / 2))


def main() -> None:
  gui_app.init_window("settings stack smoke")
  original_texture = gui_app.texture
  original_row_render = SettingsRowItem._render
  original_header_button = SettingsStackLayout._render_header_button
  original_load_icon = _load_icon_texture
  try:
    gui_app.texture = lambda *args, **kwargs: DummyTexture()  # type: ignore[method-assign]
    SettingsRowItem._render = lambda self, rect: None  # type: ignore[method-assign]
    SettingsStackLayout._render_header_button = lambda self, rect, icon, pressed: None  # type: ignore[method-assign]
    import openpilot.selfdrive.ui.sunnypilot.layouts.settings.settings_stack as _ssmod
    _ssmod._load_icon_texture = lambda icon_name, size: None  # type: ignore[method-assign]
    schema = load_schema()
    registry = DummyRegistry()
    stack = SettingsStackLayout(schema, registry=registry)  # type: ignore[arg-type]
    stack.show_event()
    _render_stack(stack)

    # root -> search -> back returns root.
    _click(stack, stack._search_btn_rect)
    assert stack._history == ["__search__"]
    _render_stack(stack)
    _click(stack, stack._back_btn_rect)
    assert stack._history == []

    # category -> search -> back returns category.
    stack._navigate_to("interface", replace=True)
    assert stack._history == ["interface"]
    _render_stack(stack)
    _click(stack, stack._search_btn_rect)
    assert stack._history == ["interface", "__search__"]
    _render_stack(stack)
    _click(stack, stack._back_btn_rect)
    assert stack._history == ["interface"]

    # Selecting route-aware search result navigates by route id.
    stack._navigate_to("__search__", replace=False)
    assert search_result_context_label(SearchRecord(
      key="OnroadScreenOffBrightness",
      title="Brightness",
      description="",
      panel_id="display",
      live_panel_id="interface",
      live_panel_label="Interface",
      route_id="interface.display",
      panel_label="Display",
      breadcrumbs=("Interface", "Display"),
    )) == "Interface / Display"
    stack._handle_search_record(SearchRecord(
      key="OnroadScreenOffBrightness",
      title="Brightness",
      description="",
      panel_id="display",
      live_panel_id="interface",
      live_panel_label="Interface",
      route_id="interface.display",
      panel_label="Display",
      breadcrumbs=("Interface", "Display"),
    ))
    assert stack._history == ["interface", "interface.display"]

    # Synthetic record without route stays on search.
    stack._navigate_to("__search__", replace=False)
    before = list(stack._history)
    stack._handle_search_record(SearchRecord(
      key="Synthetic",
      title="Synthetic",
      description="",
      panel_id="display",
      live_panel_id="interface",
      live_panel_label="Interface",
      route_id=None,
      panel_label="Display",
      breadcrumbs=(),
    ))
    assert stack._history == before

    # Category shortcut strip: deep in one category, tap another root category to jump.
    stack._navigate_to("driving.steering", replace=True)
    assert stack._history == ["driving", "driving.steering"]
    _render_stack(stack)
    system_rect = next((btn_rect for pid, btn_rect in stack._category_btn_rects if pid == "system"), None)
    assert system_rect is not None, "system category button not found in strip"
    stack._handle_mouse_release(rl.Vector2(system_rect.x + system_rect.width / 2, system_rect.y + system_rect.height / 2))
    assert stack._history == ["system"], f"expected ['system'], got {stack._history}"

    # Icon metadata coverage: every page icon in the schema has a path mapping.
    import openpilot.selfdrive.ui.sunnypilot.layouts.settings.settings_stack as _ssmod
    schema_icon_names: set[str] = set()
    for p in schema.get("pages", []):
      if isinstance(p, dict):
        icon = p.get("icon")
        if isinstance(icon, str):
          schema_icon_names.add(icon)
    missing_paths = schema_icon_names - set(_ssmod._ICON_PATHS)
    assert not missing_paths, f"schema icon names missing from _ICON_PATHS: {sorted(missing_paths)}"

    # Graceful fallback for unknown icon names.
    assert _ssmod._load_icon_texture("nonexistent_icon", 32) is None
    assert _ssmod._load_icon_texture(None, 32) is None

    # Legacy navigate callback still works.
    calls: list[tuple[str, str]] = []
    legacy = SearchLayout(navigate_callback=lambda live_panel_id, key: calls.append((live_panel_id, key)), query="brightness")
    legacy._navigate_to(SearchRecord(
      key="OnroadScreenOffBrightness",
      title="Brightness",
      description="",
      panel_id="display",
      live_panel_id="interface",
      live_panel_label="Interface",
      route_id="interface.display",
      panel_label="Display",
      breadcrumbs=("Interface", "Display"),
    ))
    assert calls == [("interface", "OnroadScreenOffBrightness")]

    # record_callback takes precedence.
    record_calls: list[str] = []
    legacy_calls: list[tuple[str, str]] = []
    preferred = SearchLayout(
      navigate_callback=lambda live_panel_id, key: legacy_calls.append((live_panel_id, key)),
      record_callback=lambda rec: record_calls.append(rec.key),
      query="brightness",
    )
    preferred._navigate_to(SearchRecord(
      key="OnroadScreenOffBrightness",
      title="Brightness",
      description="",
      panel_id="display",
      live_panel_id="interface",
      live_panel_label="Interface",
      route_id="interface.display",
      panel_label="Display",
      breadcrumbs=("Interface", "Display"),
    ))
    assert record_calls == ["OnroadScreenOffBrightness"]
    assert legacy_calls == []

    unexpected_registry_requests = set(registry.requested) - {"display"}
    assert not unexpected_registry_requests, f"unexpected registry requests: {sorted(unexpected_registry_requests)}"

    print("settings stack smoke: PASS")
  finally:
    gui_app.texture = original_texture  # type: ignore[method-assign]
    SettingsRowItem._render = original_row_render  # type: ignore[method-assign]
    SettingsStackLayout._render_header_button = original_header_button  # type: ignore[method-assign]
    import openpilot.selfdrive.ui.sunnypilot.layouts.settings.settings_stack as _ssmod
    _ssmod._load_icon_texture = original_load_icon  # type: ignore[method-assign]
    rl.close_window()


if __name__ == "__main__":
  main()
