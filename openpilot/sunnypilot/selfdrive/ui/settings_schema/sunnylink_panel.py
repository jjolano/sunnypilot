"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Schema-driven sunnylink settings panel with the legacy header widget rendered
above the compiled schema controls.
"""
import pyray as rl

from openpilot.sunnypilot.selfdrive.ui.settings_schema.widgets import SchemaPanelLayout


class SunnylinkSchemaPanelLayout(SchemaPanelLayout):
  """Schema panel for sunnylink with header widget and lifecycle management."""

  def __init__(self, panel_id: str = "sunnylink"):
    super().__init__(panel_id)
    from openpilot.selfdrive.ui.sunnypilot.layouts.settings.sunnylink import SunnylinkHeader
    self._header = SunnylinkHeader()

  def _render(self, rect: rl.Rectangle):
    self._header.set_parent_rect(rl.Rectangle(rect.x, rect.y, rect.width, 0))
    header_height = self._header._rect.height

    header_rect = rl.Rectangle(rect.x, rect.y, rect.width, header_height)
    self._header.render(header_rect)

    panel_rect = rl.Rectangle(rect.x, rect.y + header_height, rect.width, rect.height - header_height)
    self._panel.render(panel_rect)

  def show_event(self):
    super().show_event()
    self._header.show_event()
    from openpilot.selfdrive.ui.ui_state import ui_state
    ui_state.sunnylink_state.set_settings_open(True)

  def hide_event(self):
    super().hide_event()
    self._header.hide_event()
    from openpilot.selfdrive.ui.ui_state import ui_state
    ui_state.sunnylink_state.set_settings_open(False)
