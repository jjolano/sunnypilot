"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import os
from dataclasses import dataclass
from enum import IntEnum

import pyray as rl  # type: ignore[import-not-found]
from openpilot.selfdrive.ui.layouts.settings import settings as OP
from openpilot.selfdrive.ui.layouts.settings.firehose import FirehoseLayout
from openpilot.selfdrive.ui.layouts.settings.toggles import TogglesLayout
from openpilot.selfdrive.ui.sunnypilot.layouts.settings.developer import DeveloperLayoutSP
from openpilot.selfdrive.ui.sunnypilot.layouts.settings.device import DeviceLayoutSP
from openpilot.selfdrive.ui.sunnypilot.layouts.settings.models import ModelsLayout
from openpilot.selfdrive.ui.sunnypilot.layouts.settings.network import NetworkUISP
from openpilot.selfdrive.ui.sunnypilot.layouts.settings.osm import OSMLayout
from openpilot.selfdrive.ui.sunnypilot.layouts.settings.software import SoftwareLayoutSP
from openpilot.selfdrive.ui.sunnypilot.layouts.settings.sunnylink import SunnylinkLayout
from openpilot.selfdrive.ui.sunnypilot.layouts.settings.trips import TripsLayout
from openpilot.selfdrive.ui.sunnypilot.layouts.settings.vehicle import VehicleLayout
from openpilot.sunnypilot.selfdrive.ui.settings_schema.driving_panel import build_driving_layout
from openpilot.sunnypilot.selfdrive.ui.settings_schema.interface_panel import build_interface_layout
from openpilot.sunnypilot.selfdrive.ui.settings_schema.search_view import SearchLayout
from openpilot.system.ui.lib.application import gui_app, MousePos
from openpilot.system.ui.lib.multilang import tr_noop
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.lib.wifi_manager import WifiManager
from openpilot.system.ui.sunnypilot.lib.styles import style
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.scroller_tici import Scroller

# from openpilot.selfdrive.ui.sunnypilot.layouts.settings.navigation import NavigationLayout

OP.PANEL_COLOR = rl.Color(10, 10, 10, 255)
ICON_SIZE = 70

OP.PanelType = IntEnum(
  "PanelType",
  [es.name for es in OP.PanelType] + [
    "SEARCH",
    "SUNNYLINK",
    "MODELS",
    "DRIVING",
    "INTERFACE",
    "OSM",
    "NAVIGATION",
    "TRIPS",
    "VEHICLE",
  ],
  start=0,
)

# Consolidated-panel id (from a search record's live_panel_id) -> the PanelType to open.
_PANEL_BY_ID = {
  "driving": OP.PanelType.DRIVING,
  "interface": OP.PanelType.INTERFACE,
  "models": OP.PanelType.MODELS,
  "software": OP.PanelType.SOFTWARE,
  "developer": OP.PanelType.DEVELOPER,
  "device": OP.PanelType.DEVICE,
  "vehicle": OP.PanelType.VEHICLE,
  "toggles": OP.PanelType.TOGGLES,
}


@dataclass
class PanelInfo(OP.PanelInfo):
  icon: str = ""


class NavButton(Widget):
  def __init__(self, parent, p_type, p_info):
    super().__init__()
    self.parent = parent
    self.panel_type = p_type
    self.panel_info = p_info

  def _render(self, rect):
    is_selected = self.panel_type == self.parent._current_panel
    text_color = OP.TEXT_SELECTED if is_selected else OP.TEXT_NORMAL
    content_x = rect.x + 90
    text_size = measure_text_cached(self.parent._font_medium, self.panel_info.name, 65)

    # Draw background if selected
    if is_selected:
      self.container_rect = rl.Rectangle(
        content_x - 50, rect.y, OP.SIDEBAR_WIDTH - 50, OP.NAV_BTN_HEIGHT
      )
      rl.draw_rectangle_rounded(self.container_rect, 0.2, 5, OP.CLOSE_BTN_COLOR)

    if self.panel_info.icon:
      icon_texture = gui_app.texture(self.panel_info.icon, ICON_SIZE, ICON_SIZE, keep_aspect_ratio=True)
      rl.draw_texture_ex(icon_texture, rl.Vector2(content_x, rect.y + (OP.NAV_BTN_HEIGHT - icon_texture.height) / 2), 0.0, 1.0, rl.WHITE)
      content_x += ICON_SIZE + 20

    # Draw button text (right-aligned)
    text_pos = rl.Vector2(
      content_x,
      rect.y + (OP.NAV_BTN_HEIGHT - text_size.y) / 2
    )
    rl.draw_text_ex(self.parent._font_medium, self.panel_info.name, text_pos, 55, 0, text_color)

    # Store button rect for click detection
    self.panel_info.button_rect = rect


class SettingsLayoutSP(OP.SettingsLayout):
  def __init__(self):
    OP.SettingsLayout.__init__(self)
    self._nav_items: list[Widget] = []
    self._close_btn_rect = rl.Rectangle(0, 0, 0, 0)
    self._search_btn_rect = rl.Rectangle(0, 0, 0, 0)

    # Create sidebar scroller
    self._sidebar_scroller = Scroller([], spacing=0, line_separator=False, pad_end=False)

    # Panel configuration
    wifi_manager = WifiManager()
    wifi_manager.set_active(False)

    self._panels = {
      OP.PanelType.SEARCH: PanelInfo(tr_noop("Search"), SearchLayout(self._navigate_to_setting)),
      OP.PanelType.DEVICE: PanelInfo(tr_noop("Device"), DeviceLayoutSP(), icon="../../sunnypilot/selfdrive/assets/offroad/icon_home.png"),
      OP.PanelType.NETWORK: PanelInfo(tr_noop("Network"), NetworkUISP(wifi_manager), icon="icons/network.png"),
      OP.PanelType.SUNNYLINK: PanelInfo(tr_noop("sunnylink"), SunnylinkLayout(), icon="icons/wifi_strength_full.png"),
      OP.PanelType.TOGGLES: PanelInfo(tr_noop("Toggles"), TogglesLayout(), icon="../../sunnypilot/selfdrive/assets/offroad/icon_toggle.png"),
      OP.PanelType.SOFTWARE: PanelInfo(tr_noop("Software"), SoftwareLayoutSP(), icon="../../sunnypilot/selfdrive/assets/offroad/icon_software.png"),
      OP.PanelType.MODELS: PanelInfo(tr_noop("Models"), ModelsLayout(), icon="../../sunnypilot/selfdrive/assets/offroad/icon_models.png"),
      OP.PanelType.DRIVING: PanelInfo(tr_noop("Driving"), build_driving_layout(), icon="../../sunnypilot/selfdrive/assets/offroad/icon_lateral.png"),
      OP.PanelType.INTERFACE: PanelInfo(tr_noop("Interface"), build_interface_layout(), icon="../../sunnypilot/selfdrive/assets/offroad/icon_visuals.png"),
      OP.PanelType.OSM: PanelInfo(tr_noop("OSM"), OSMLayout(), icon="../../sunnypilot/selfdrive/assets/offroad/icon_map.png"),
      # OP.PanelType.NAVIGATION: PanelInfo(tr_noop("Navigation"), NavigationLayout(), icon="../../sunnypilot/selfdrive/assets/offroad/icon_map.png"),
      OP.PanelType.TRIPS: PanelInfo(tr_noop("Trips"), TripsLayout(), icon="../../sunnypilot/selfdrive/assets/offroad/icon_trips.png"),
      OP.PanelType.VEHICLE: PanelInfo(tr_noop("Vehicle"), VehicleLayout(), icon="../../sunnypilot/selfdrive/assets/offroad/icon_vehicle.png"),
      OP.PanelType.FIREHOSE: PanelInfo(tr_noop("Firehose"), FirehoseLayout(), icon="../../sunnypilot/selfdrive/assets/offroad/icon_firehose.png"),
      OP.PanelType.DEVELOPER: PanelInfo(tr_noop("Developer"), DeveloperLayoutSP(), icon="icons/shell.png"),
    }

  def _navigate_to_setting(self, live_panel_id: str, key: str) -> None:
    """Jump from a global-search result to the panel the setting lives in."""
    panel_type = _PANEL_BY_ID.get(live_panel_id)
    if panel_type is not None:
      self.set_current_panel(panel_type)

  def _draw_top_button(self, rect: rl.Rectangle, active: bool, icon: str):
    bg_color = OP.CLOSE_BTN_PRESSED if active else OP.CLOSE_BTN_COLOR
    rl.draw_rectangle_rounded(rect, 1.0, 20, bg_color)

    icon_color = rl.Color(255, 255, 255, 255) if not active else rl.Color(220, 220, 220, 255)
    if icon == "close":
      icon_dest = rl.Rectangle(
        rect.x + (rect.width - self._close_icon.width) / 2,
        rect.y + (rect.height - self._close_icon.height) / 2,
        self._close_icon.width,
        self._close_icon.height,
      )
      rl.draw_texture_pro(
        self._close_icon,
        rl.Rectangle(0, 0, self._close_icon.width, self._close_icon.height),
        icon_dest,
        rl.Vector2(0, 0),
        0,
        icon_color,
      )
      return

    # Search icon: a simple magnifier, drawn to match the close button's weight.
    cx = rect.x + rect.width * 0.43
    cy = rect.y + rect.height * 0.43
    radius = rect.width * 0.16
    for i in range(4):
      rl.draw_circle_lines(int(cx), int(cy), max(int(radius) - i, 1), icon_color)
    rl.draw_line_ex(
      rl.Vector2(cx + radius * 0.7, cy + radius * 0.7),
      rl.Vector2(cx + radius * 1.55, cy + radius * 1.55),
      7,
      icon_color,
    )

  def _draw_sidebar(self, rect: rl.Rectangle):
    rl.draw_rectangle_rec(rect, OP.SIDEBAR_COLOR)

    # Top action buttons: close + search, centered as a pair.
    top_btn_y = rect.y + style.ITEM_PADDING * 2
    top_btn_gap = style.ITEM_PADDING
    top_btn_total_width = style.CLOSE_BTN_SIZE * 2 + top_btn_gap
    top_btn_x = rect.x + (rect.width - top_btn_total_width) / 2

    close_btn_rect = rl.Rectangle(top_btn_x, top_btn_y, style.CLOSE_BTN_SIZE, style.CLOSE_BTN_SIZE)
    search_btn_rect = rl.Rectangle(
      top_btn_x + style.CLOSE_BTN_SIZE + top_btn_gap, top_btn_y, style.CLOSE_BTN_SIZE, style.CLOSE_BTN_SIZE
    )

    mouse_pos = rl.get_mouse_position()
    close_pressed = rl.is_mouse_button_down(rl.MouseButton.MOUSE_BUTTON_LEFT) and rl.check_collision_point_rec(mouse_pos, close_btn_rect)
    search_pressed = rl.is_mouse_button_down(rl.MouseButton.MOUSE_BUTTON_LEFT) and rl.check_collision_point_rec(mouse_pos, search_btn_rect)

    self._draw_top_button(close_btn_rect, close_pressed, "close")
    self._draw_top_button(search_btn_rect, search_pressed or self._current_panel == OP.PanelType.SEARCH, "search")

    # Store close button rect for click detection
    self._close_btn_rect = close_btn_rect
    self._search_btn_rect = search_btn_rect

    # Navigation buttons with scroller
    if not self._nav_items:
      for panel_type, panel_info in self._panels.items():
        if panel_type == OP.PanelType.SEARCH:
          continue
        nav_button = NavButton(self, panel_type, panel_info)
        nav_button.rect.width = rect.width - 100  # Full width minus padding
        nav_button.rect.height = OP.NAV_BTN_HEIGHT
        self._nav_items.append(nav_button)
        self._sidebar_scroller.add_widget(nav_button)

    # Draw navigation section with scroller
    nav_rect = rl.Rectangle(
      rect.x,
      self._search_btn_rect.y + self._search_btn_rect.height + style.ITEM_PADDING * 2,
      rect.width,
      rect.height - (self._search_btn_rect.y + self._search_btn_rect.height + style.ITEM_PADDING * 2 - rect.y)
    )

    if self._nav_items:
      self._sidebar_scroller.render(nav_rect)
      return

  def _handle_mouse_release(self, mouse_pos: MousePos) -> bool:
    # Check close button
    if rl.check_collision_point_rec(mouse_pos, self._close_btn_rect):
      if self._close_callback:
        self._close_callback()
      return True

    # Check search button
    if rl.check_collision_point_rec(mouse_pos, self._search_btn_rect):
      self.set_current_panel(OP.PanelType.SEARCH)
      return True

    # Check navigation buttons
    for panel_type, panel_info in self._panels.items():
      if panel_type == OP.PanelType.SEARCH:
        continue
      if rl.check_collision_point_rec(mouse_pos, panel_info.button_rect) and self._sidebar_scroller.scroll_panel.is_touch_valid():
        self.set_current_panel(panel_type)
        return True

    return False

  def show_event(self):
    super().show_event()
    self._panels[self._current_panel].instance.show_event()
    self._sidebar_scroller.show_event()


def make_settings_layout():
  if os.getenv("SP_SETTINGS_STACK_NAV") != "1":
    return SettingsLayoutSP()

  try:
    from openpilot.sunnypilot.selfdrive.ui.settings_schema.schema_loader import load_schema, navigation_available, navigation_errors

    schema = load_schema()
    if not navigation_available(schema):
      print(f"SP_SETTINGS_STACK_NAV=1 disabled: {'; '.join(navigation_errors(schema))}")
      return SettingsLayoutSP()

    from openpilot.selfdrive.ui.sunnypilot.layouts.settings.settings_stack import SettingsStackLayout
    return SettingsStackLayout(schema)
  except Exception as exc:
    print(f"SP_SETTINGS_STACK_NAV=1 disabled: {exc}")
    return SettingsLayoutSP()
