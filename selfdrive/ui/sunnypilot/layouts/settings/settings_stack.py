"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Feature-flagged settings shell that drives schema navigation but still mounts the
existing imperative/custom widgets for every leaf page.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import cast

import pyray as rl  # type: ignore[import-not-found]

from openpilot.selfdrive.ui.layouts.settings.settings import PanelType as LegacyPanelType
from openpilot.selfdrive.ui.sunnypilot.layouts.settings.always_offroad_toggle import AlwaysOffroadToggle
from openpilot.selfdrive.ui.sunnypilot.layouts.settings.custom_page_registry import SettingsPageRegistry, get_settings_page_registry
from openpilot.sunnypilot.selfdrive.ui.settings_schema.schema_loader import (
  breadcrumbs_for,
  get_page,
  get_root_navigation,
  resolve_page_content,
)
from openpilot.sunnypilot.selfdrive.ui.settings_schema.search import SearchRecord, build_index
from openpilot.sunnypilot.selfdrive.ui.settings_schema.search_view import SearchLayout
from openpilot.system.ui.lib.application import FontWeight, MousePos, gui_app
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.sunnypilot.lib.styles import style
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.scroller_tici import Scroller
from openpilot.system.ui.widgets.label import gui_label


BACKGROUND = rl.Color(8, 8, 8, 255)
SURFACE = rl.Color(20, 20, 20, 255)
SURFACE_PRESSED = rl.Color(34, 34, 34, 255)
TEXT_MUTED = rl.Color(160, 160, 160, 255)
TEXT_SUBTLE = rl.Color(120, 120, 120, 255)
DIVIDER = rl.Color(34, 34, 34, 255)
ACCENT = style.BLUE

HEADER_HEIGHT = 248
HEADER_PAD = 36
HEADER_BTN_SIZE = 150
ROW_HEIGHT = 132
ROW_GAP = 16
ROW_PAD_X = 30
ROW_PAD_Y = 20
HEADER_TEXT_GAP = 24

ROOT_PAGE_ID = "__root__"
SEARCH_PAGE_ID = "__search__"

SchemaRendererFactory = Callable[[], Widget]


def _flat_schema_renderer(panel_id: str) -> SchemaRendererFactory:
  def build() -> Widget:
    from openpilot.sunnypilot.selfdrive.ui.settings_schema.widgets import SchemaPanelLayout
    # Import developer_actions so button/toggle factories are registered before
    # build_control is called for the developer panel.
    if panel_id == "developer":
      import openpilot.sunnypilot.selfdrive.ui.settings_schema.developer_actions  # noqa: F401
    if panel_id == "software":
      import openpilot.sunnypilot.selfdrive.ui.settings_schema.software_actions  # noqa: F401
    if panel_id == "models":
      import openpilot.sunnypilot.selfdrive.ui.settings_schema.models_actions  # noqa: F401
    if panel_id == "osm":
      import openpilot.sunnypilot.selfdrive.ui.settings_schema.osm_actions  # noqa: F401
    return SchemaPanelLayout(panel_id)
  return build


def _steering_schema_renderer() -> Widget:
  from openpilot.sunnypilot.selfdrive.ui.settings_schema.steering_panel import SchemaSteeringLayout
  return SchemaSteeringLayout()


def _cruise_schema_renderer() -> Widget:
  from openpilot.selfdrive.ui.sunnypilot.layouts.settings.cruise_sub_layouts.speed_limit_settings import (
    SpeedLimitSettingsLayout,
  )
  from openpilot.sunnypilot.selfdrive.ui.settings_schema.nav_layout import SchemaNavLayout
  return SchemaNavLayout("cruise", {"speed_limit_settings": SpeedLimitSettingsLayout})


def _sunnylink_schema_renderer() -> Widget:
  from openpilot.sunnypilot.selfdrive.ui.settings_schema.sunnylink_panel import SunnylinkSchemaPanelLayout
  return SunnylinkSchemaPanelLayout()


def _vehicle_schema_renderer() -> Widget:
  from openpilot.sunnypilot.selfdrive.ui.settings_schema.vehicle_panel import VehicleSchemaPanelLayout
  return VehicleSchemaPanelLayout()


_SCHEMA_PANEL_RENDERERS: dict[str, SchemaRendererFactory] = {
  "cruise": _cruise_schema_renderer,
  "developer": _flat_schema_renderer("developer"),
  "display": _flat_schema_renderer("display"),
  "models": _flat_schema_renderer("models"),
  "osm": _flat_schema_renderer("osm"),
  "software": _flat_schema_renderer("software"),
  "steering": _steering_schema_renderer,
  "sunnylink": _sunnylink_schema_renderer,
  "toggles": _flat_schema_renderer("toggles"),
  "vehicle": _vehicle_schema_renderer,
  "visuals": _flat_schema_renderer("visuals"),
}

_PANEL_TO_PAGE_ID = {
  "DEVICE": "system.device",
  "NETWORK": "system.network",
  "TOGGLES": "driving.toggles",
  "SOFTWARE": "system.software",
  "FIREHOSE": "system.firehose",
  "DEVELOPER": "system.developer",
  "SUNNYLINK": "system.sunnylink",
  "MODELS": "system.models",
  "DRIVING": "driving",
  "INTERFACE": "interface",
  "OSM": "interface.osm",
  "TRIPS": "system.trips",
  "VEHICLE": "vehicle",
}


def search_result_context_label(rec: SearchRecord) -> str:
  return " / ".join(rec.breadcrumbs) if rec.breadcrumbs else rec.live_panel_label


def _resolve_text(text: str | Callable[[], str] | None) -> str:
  if text is None:
    return ""
  return text() if callable(text) else text


def _page_path_ids(schema: dict, target_page_id: str) -> list[str]:
  def walk(page: dict, trail: list[str]) -> list[str] | None:
    pid = page.get("id")
    if not isinstance(pid, str):
      return None

    next_trail = trail + [pid]
    if pid == target_page_id:
      return next_trail

    children = page.get("children")
    if not isinstance(children, list):
      return None

    for child_id in children:
      if not isinstance(child_id, str):
        continue
      child = get_page(schema, child_id)
      if child is None:
        continue
      path = walk(child, next_trail)
      if path is not None:
        return path
    return None

  for root in get_root_navigation(schema):
    path = walk(root, [])
    if path is not None:
      return path
  return []


def _is_page_hidden(schema: dict, page_id: str) -> bool:
  page = get_page(schema, page_id)
  return isinstance(page, dict) and page.get("new_shell_hidden") is True


def _page_row_visible(schema: dict, page_id: str, memo: dict[str, bool] | None = None) -> bool:
  """Return whether a navigation page should render as a row in the new shell.

  Leaf pages are visible unless marked new_shell_hidden. Category pages are
  visible only when they are not hidden and have at least one visible descendant.
  """
  if memo is None:
    memo = {}
  if page_id in memo:
    return memo[page_id]
  if _is_page_hidden(schema, page_id):
    memo[page_id] = False
    return False
  page = get_page(schema, page_id)
  if not isinstance(page, dict):
    memo[page_id] = False
    return False
  children = page.get("children")
  if not isinstance(children, list) or not children:
    memo[page_id] = True
    return True
  visible = any(_page_row_visible(schema, child_id, memo) for child_id in children)
  memo[page_id] = visible
  return visible


def _visible_child_pages(schema: dict, page_id: str) -> list[dict]:
  page = get_page(schema, page_id)
  if not isinstance(page, dict):
    return []
  children = page.get("children")
  if not isinstance(children, list):
    return []
  return [child for child in (get_page(schema, child_id) for child_id in children)
          if child is not None and _page_row_visible(schema, child["id"])]


def _stack_route_visible(schema: dict, page_id: str) -> bool:
  return page_id == ROOT_PAGE_ID or _page_row_visible(schema, page_id)


class SettingsRowItem(Widget):
  HEIGHT = ROW_HEIGHT

  def __init__(self, title: str | Callable[[], str], subtitle: str | Callable[[], str] | None, callback: Callable[[], None]):
    super().__init__()
    self._title = title
    self._subtitle = subtitle
    self.set_click_callback(callback)
    self.set_rect(rl.Rectangle(0, 0, 0, self.HEIGHT))
    self._chevron = gui_app.texture("icons/arrow-right.png", 52, 52, keep_aspect_ratio=True)

  def _render(self, rect: rl.Rectangle):
    bg = SURFACE_PRESSED if self.is_pressed else SURFACE
    rl.draw_rectangle_rounded(rect, 0.14, 18, bg)
    rl.draw_rectangle_lines(int(rect.x), int(rect.y), int(rect.width), int(rect.height), DIVIDER)

    if self.is_pressed:
      rl.draw_rectangle(int(rect.x), int(rect.y + 10), 7, int(rect.height - 20), ACCENT)

    title = _resolve_text(self._title)
    subtitle = _resolve_text(self._subtitle)

    content_left = rect.x + ROW_PAD_X
    content_right = rect.x + rect.width - ROW_PAD_X - self._chevron.width - 24
    content_width = max(0, content_right - content_left)

    title_y = rect.y + 24 if subtitle else rect.y + (rect.height - 50) / 2 - 6
    title_rect = rl.Rectangle(content_left, title_y, content_width, 50)
    gui_label(title_rect, title, font_size=52, color=rl.WHITE, font_weight=FontWeight.BOLD,
              alignment=rl.GuiTextAlignment.TEXT_ALIGN_LEFT, alignment_vertical=rl.GuiTextAlignmentVertical.TEXT_ALIGN_TOP)

    if subtitle:
      subtitle_rect = rl.Rectangle(content_left, rect.y + 74, content_width, 34)
      gui_label(subtitle_rect, subtitle, font_size=33, color=TEXT_MUTED, font_weight=FontWeight.NORMAL,
                alignment=rl.GuiTextAlignment.TEXT_ALIGN_LEFT, alignment_vertical=rl.GuiTextAlignmentVertical.TEXT_ALIGN_TOP)

    chevron_y = rect.y + (rect.height - self._chevron.height) / 2
    chevron_color = rl.WHITE if self.is_pressed else TEXT_SUBTLE
    rl.draw_texture_ex(self._chevron, rl.Vector2(rect.x + rect.width - ROW_PAD_X - self._chevron.width, chevron_y), 0.0, 1.0, chevron_color)


class PageRowsView(Widget):
  def __init__(self, rows: list[Widget]):
    super().__init__()
    self._rows = rows
    self._scroller = Scroller(rows, spacing=ROW_GAP, line_separator=False, pad_end=False)

  def _render(self, rect: rl.Rectangle):
    for row in self._rows:
      row.set_rect(rl.Rectangle(0, 0, rect.width, row.rect.height))
    self._scroller.render(rect)

  def show_event(self):
    self._scroller.show_event()

  def hide_event(self):
    self._scroller.hide_event()


class SettingsStackLayout(Widget):
  def __init__(self, schema: dict, registry: SettingsPageRegistry | None = None):
    super().__init__()
    self._schema = schema
    self._registry = registry or get_settings_page_registry()
    self._close_callback: Callable | None = None
    self._history: list[str] = []
    self._shown = False
    self._header_rect = rl.Rectangle(0, 0, 0, HEADER_HEIGHT)
    self._back_btn_rect = rl.Rectangle(0, 0, 0, 0)
    self._close_btn_rect = rl.Rectangle(0, 0, 0, 0)
    self._search_btn_rect = rl.Rectangle(0, 0, 0, 0)
    self._page_views: dict[str, Widget] = {}
    self._failed_schema_panels: set[str] = set()
    self._always_offroad_toggle = AlwaysOffroadToggle()
    self._active_view: Widget = self._root_view()
    self._close_icon = gui_app.texture("icons/close2.png", 68, 68, keep_aspect_ratio=True)
    self._back_icon = gui_app.texture("icons/arrow-right.png", 68, 68, keep_aspect_ratio=True)

  def set_callbacks(self, on_close: Callable):
    self._close_callback = on_close

  def set_current_panel(self, panel_type: LegacyPanelType):
    page_id = _PANEL_TO_PAGE_ID.get(panel_type.name, ROOT_PAGE_ID)
    if not _stack_route_visible(self._schema, page_id):
      page_id = ROOT_PAGE_ID
    self._navigate_to(page_id, replace=True)

  def _root_view(self) -> Widget:
    return self._page_view_for(None)

  def _page_view_for(self, page_id: str | None) -> Widget:
    key = page_id or ROOT_PAGE_ID
    if key in self._page_views:
      return self._page_views[key]

    if page_id is None:
      pages = [p for p in get_root_navigation(self._schema) if _page_row_visible(self._schema, p["id"])]
    else:
      pages = _visible_child_pages(self._schema, page_id)

    rows: list[Widget] = []
    for page in pages:
      pid = page.get("id")
      if not isinstance(pid, str):
        continue
      title = page.get("title", pid)
      subtitle = page.get("description")
      rows.append(SettingsRowItem(title=tr(title), subtitle=tr(subtitle) if isinstance(subtitle, str) and subtitle else None,
                                  callback=lambda target=pid: self._navigate_to(target, replace=False)))

    view = PageRowsView(rows)
    self._page_views[key] = view
    return view

  def _current_page_id(self) -> str | None:
    return self._history[-1] if self._history else None

  def _is_search_page(self, page_id: str | None) -> bool:
    return page_id == SEARCH_PAGE_ID

  def _search_view(self) -> Widget:
    key = SEARCH_PAGE_ID
    if key not in self._page_views:
      self._page_views[key] = SearchLayout(
        index=build_index(self._schema, include_new_shell_hidden=False),
        record_callback=self._handle_search_record,
        result_context_label=search_result_context_label,
      )
    return self._page_views[key]

  def _handle_search_record(self, record: SearchRecord):
    if record.route_id:
      self._navigate_to(record.route_id, replace=True)
      return
    print(f"[settings_stack] search record '{record.key}' has no route_id; staying on search")

  def _open_search(self):
    if self._is_search_page(self._current_page_id()):
      return
    self._history = self._history + [SEARCH_PAGE_ID]
    self._swap_active_view(self._current_view())

  def _schema_panel_view(self, panel_id: str) -> Widget | None:
    factory = _SCHEMA_PANEL_RENDERERS.get(panel_id)
    if factory is None or panel_id in self._failed_schema_panels:
      return None

    key = f"schema_panel:{panel_id}"
    if key in self._page_views:
      return self._page_views[key]

    try:
      view = factory()
      unsupported = getattr(view, "unsupported", [])
      if unsupported:
        raise ValueError(f"schema-rendered panel {panel_id!r} has unsupported controls: {unsupported}")
      self._page_views[key] = view
    except Exception as e:
      self._failed_schema_panels.add(panel_id)
      print(f"[settings_stack] schema-rendered panel '{panel_id}' failed: {e}; falling back to registry")
      return None
    return self._page_views[key]

  def _current_view(self) -> Widget:
    page_id = self._current_page_id()
    if page_id is None:
      return self._root_view()

    if self._is_search_page(page_id):
      return self._search_view()

    page = get_page(self._schema, page_id)
    if page is None:
      return self._root_view()

    if isinstance(page.get("children"), list):
      return self._page_view_for(page_id)

    content = resolve_page_content(self._schema, page_id)
    if content is None:
      return self._root_view()
    if content.get("kind") == "panel_ref":
      panel_id = content.get("panel")
      if isinstance(panel_id, str):
        schema_view = self._schema_panel_view(panel_id)
        if schema_view is not None:
          return schema_view
        try:
          return cast(Widget, self._registry.get(panel_id))
        except KeyError:
          return self._root_view()
    if content.get("kind") == "custom_page":
      component = content.get("component")
      if isinstance(component, str):
        # Check schema renderers first (for custom_page components that have
        # been migrated to schema-driven rendering, like vehicle).
        schema_view = self._schema_panel_view(component)
        if schema_view is not None:
          return schema_view
        try:
          return cast(Widget, self._registry.get(component))
        except KeyError:
          return self._root_view()
    return self._root_view()

  def _swap_active_view(self, view: Widget):
    if view is self._active_view:
      return
    if self._shown:
      self._active_view.hide_event()
    self._active_view = view
    if self._shown:
      self._active_view.show_event()

  def _navigate_to(self, page_id: str, replace: bool):
    if not _stack_route_visible(self._schema, page_id):
      page_id = ROOT_PAGE_ID
      replace = True

    if replace:
      next_history = [] if page_id == ROOT_PAGE_ID else _page_path_ids(self._schema, page_id)
    else:
      current = self._current_page_id()
      if current == page_id:
        return
      next_history = self._history + [page_id]

    if not next_history and page_id != ROOT_PAGE_ID:
      next_history = []

    if next_history == self._history:
      return

    self._history = next_history
    self._swap_active_view(self._current_view())

  def _go_back(self):
    if not self._history:
      return
    next_history = self._history[:-1]
    if next_history == self._history:
      return
    self._history = next_history
    self._swap_active_view(self._current_view())

  def _header_text_geometry(self, rect: rl.Rectangle) -> tuple[float, float]:
    leading_width = HEADER_PAD + (HEADER_BTN_SIZE + HEADER_TEXT_GAP if self._history else 0)
    trailing_width = HEADER_PAD + (HEADER_BTN_SIZE * 2) + (HEADER_TEXT_GAP * 2)
    text_x = rect.x + leading_width
    text_width = rect.width - leading_width - trailing_width
    return text_x, text_width

  def _layout_header_buttons(self, rect: rl.Rectangle) -> tuple[rl.Rectangle, rl.Rectangle, rl.Rectangle]:
    back_btn_rect = rl.Rectangle(rect.x + HEADER_PAD, rect.y + 34, HEADER_BTN_SIZE, HEADER_BTN_SIZE) if self._history else rl.Rectangle(0, 0, 0, 0)
    close_btn_rect = rl.Rectangle(rect.x + rect.width - HEADER_PAD - HEADER_BTN_SIZE, rect.y + 34, HEADER_BTN_SIZE, HEADER_BTN_SIZE)
    search_btn_rect = rl.Rectangle(close_btn_rect.x - HEADER_TEXT_GAP - HEADER_BTN_SIZE, rect.y + 34, HEADER_BTN_SIZE, HEADER_BTN_SIZE)
    return back_btn_rect, search_btn_rect, close_btn_rect

  def _header_texts(self, rect: rl.Rectangle) -> tuple[str, str, float, float]:
    page_id = self._current_page_id()
    text_x, text_width = self._header_text_geometry(rect)
    if page_id is None:
      return "", tr("Settings"), text_x, text_width

    if self._is_search_page(page_id):
      return "", tr("Search settings"), text_x, text_width

    crumbs = breadcrumbs_for(self._schema, page_id)
    if not crumbs:
      page = get_page(self._schema, page_id)
      return "", tr(page.get("title", "Settings")) if isinstance(page, dict) else tr("Settings"), text_x, text_width

    if len(crumbs) == 1:
      return "", tr(crumbs[-1]), text_x, text_width
    return " / ".join(tr(c) for c in crumbs[:-1]), tr(crumbs[-1]), text_x, text_width

  def _content_rect(self, rect: rl.Rectangle) -> rl.Rectangle:
    footer_space = self._always_offroad_toggle.HEIGHT + HEADER_PAD
    return rl.Rectangle(rect.x + HEADER_PAD, rect.y + HEADER_HEIGHT, rect.width - (HEADER_PAD * 2), rect.height - HEADER_HEIGHT - footer_space)

  def _render_header_button(self, rect: rl.Rectangle, icon: str, pressed: bool):
    bg = SURFACE_PRESSED if pressed else SURFACE
    rl.draw_rectangle_rounded(rect, 0.18, 16, bg)
    color = rl.Color(230, 230, 230, 255) if pressed else rl.WHITE
    if icon == "close":
      texture = self._close_icon
      dest = rl.Rectangle(rect.x + (rect.width - texture.width) / 2, rect.y + (rect.height - texture.height) / 2, texture.width, texture.height)
      rl.draw_texture_pro(texture, rl.Rectangle(0, 0, texture.width, texture.height), dest, rl.Vector2(0, 0), 0, color)
    elif icon == "back":
      texture = self._back_icon
      dest = rl.Rectangle(rect.x + (rect.width - texture.width) / 2, rect.y + (rect.height - texture.height) / 2, texture.width, texture.height)
      rl.draw_texture_pro(texture, rl.Rectangle(0, 0, texture.width, texture.height), dest, rl.Vector2(0, 0), 180, color)
    elif icon == "search":
      cx = rect.x + rect.width * 0.43
      cy = rect.y + rect.height * 0.43
      radius = rect.width * 0.16
      for i in range(4):
        rl.draw_circle_lines(int(cx), int(cy), max(int(radius) - i, 1), color)
      rl.draw_line_ex(
        rl.Vector2(cx + radius * 0.7, cy + radius * 0.7),
        rl.Vector2(cx + radius * 1.55, cy + radius * 1.55),
        7,
        color,
      )

  def _header_button_pressed(self, rect: rl.Rectangle) -> bool:
    return rl.is_mouse_button_down(rl.MouseButton.MOUSE_BUTTON_LEFT) and rl.check_collision_point_rec(rl.get_mouse_position(), rect)

  def _render(self, rect: rl.Rectangle):
    rl.draw_rectangle_rec(rect, BACKGROUND)
    self._header_rect = rl.Rectangle(rect.x, rect.y, rect.width, HEADER_HEIGHT)

    breadcrumb, title, text_x, text_width = self._header_texts(rect)
    self._back_btn_rect, self._search_btn_rect, self._close_btn_rect = self._layout_header_buttons(rect)

    if self._history:
      self._render_header_button(self._back_btn_rect, "back", self._header_button_pressed(self._back_btn_rect))

    self._render_header_button(self._search_btn_rect, "search", self._header_button_pressed(self._search_btn_rect))
    self._render_header_button(self._close_btn_rect, "close", self._header_button_pressed(self._close_btn_rect))

    if breadcrumb:
      gui_label(rl.Rectangle(text_x, rect.y + 34, text_width, 40), breadcrumb,
                font_size=30, color=TEXT_SUBTLE, font_weight=FontWeight.MEDIUM,
                alignment=rl.GuiTextAlignment.TEXT_ALIGN_LEFT, alignment_vertical=rl.GuiTextAlignmentVertical.TEXT_ALIGN_TOP)

    title_y = rect.y + 76 if breadcrumb else rect.y + 64
    gui_label(rl.Rectangle(text_x, title_y, text_width, 90), title,
              font_size=72, color=rl.WHITE, font_weight=FontWeight.BOLD,
              alignment=rl.GuiTextAlignment.TEXT_ALIGN_LEFT, alignment_vertical=rl.GuiTextAlignmentVertical.TEXT_ALIGN_TOP)

    content_rect = self._content_rect(rect)
    self._active_view.render(content_rect)
    footer_width = min(self._always_offroad_toggle.WIDTH, rect.width - (HEADER_PAD * 2))
    footer_rect = rl.Rectangle(rect.x + HEADER_PAD, rect.y + rect.height - self._always_offroad_toggle.HEIGHT - HEADER_PAD,
                               footer_width, self._always_offroad_toggle.HEIGHT)
    self._always_offroad_toggle.render(footer_rect)

  def _handle_mouse_release(self, mouse_pos: MousePos) -> None:
    if rl.check_collision_point_rec(mouse_pos, self._always_offroad_toggle.rect):
      return
    if rl.check_collision_point_rec(mouse_pos, self._search_btn_rect):
      self._open_search()
      return
    if self._history and rl.check_collision_point_rec(mouse_pos, self._back_btn_rect):
      self._go_back()
      return
    if rl.check_collision_point_rec(mouse_pos, self._close_btn_rect):
      if self._close_callback:
        self._close_callback()
      return

  def show_event(self):
    super().show_event()
    self._shown = True
    self._active_view.show_event()

  def hide_event(self):
    super().hide_event()
    self._shown = False
    self._active_view.hide_event()
