"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Persistent Always Offroad action button for settings shells.
"""
from __future__ import annotations

import pyray as rl  # type: ignore[import-not-found]

from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import FontWeight, gui_app
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.sunnypilot.lib.styles import style
from openpilot.system.ui.widgets import DialogResult, Widget
from openpilot.system.ui.widgets.confirm_dialog import ConfirmDialog, alert_dialog
from openpilot.system.ui.widgets.label import gui_label


class AlwaysOffroadToggle(Widget):
  WIDTH = 460
  HEIGHT = style.BUTTON_HEIGHT

  def __init__(self):
    super().__init__()
    self.set_rect(rl.Rectangle(0, 0, self.WIDTH, self.HEIGHT))

  @staticmethod
  def _always_offroad_enabled() -> bool:
    current = getattr(ui_state, "always_offroad", None)
    return ui_state.params.get_bool("OffroadMode") if current is None else current

  def _render(self, rect: rl.Rectangle):
    active = self._always_offroad_enabled()
    bg = style.BUTTON_PRIMARY_COLOR if active else style.BUTTON_NEUTRAL_GRAY
    accent = rl.Color(224, 224, 224, 255) if active else rl.Color(255, 116, 94, 255)
    text_color = rl.WHITE
    subtitle_color = style.ITEM_DESC_TEXT_COLOR if active else rl.Color(214, 188, 184, 255)

    rl.draw_rectangle_rounded(rect, 0.18, 18, bg)
    rl.draw_rectangle(int(rect.x + 10), int(rect.y + 10), 8, int(rect.height - 20), accent)
    rl.draw_rectangle_lines(int(rect.x), int(rect.y), int(rect.width), int(rect.height), rl.Color(0, 0, 0, 90))

    title = tr("Exit Offroad") if active else tr("Always Offroad")
    subtitle = tr("Always offroad is on") if active else tr("Force offroad mode")

    text_left = rect.x + 34
    text_width = rect.width - 68
    gui_label(rl.Rectangle(text_left, rect.y + 20, text_width, 44), title,
              font_size=48, color=text_color, font_weight=FontWeight.BOLD,
              alignment=rl.GuiTextAlignment.TEXT_ALIGN_LEFT, alignment_vertical=rl.GuiTextAlignmentVertical.TEXT_ALIGN_TOP)
    gui_label(rl.Rectangle(text_left, rect.y + 68, text_width, 30), subtitle,
              font_size=30, color=subtitle_color, font_weight=FontWeight.NORMAL,
              alignment=rl.GuiTextAlignment.TEXT_ALIGN_LEFT, alignment_vertical=rl.GuiTextAlignmentVertical.TEXT_ALIGN_TOP)

  def _prompt_toggle(self):
    target_state = not self._always_offroad_enabled()

    if ui_state.engaged:
      gui_app.push_widget(alert_dialog(tr("Disengage to change Always Offroad mode")))
      return

    confirm_text = tr("Are you sure you want to exit Always Offroad mode?") if not target_state else tr("Are you sure you want to enter Always Offroad mode?")
    confirm_label = tr("Exit") if not target_state else tr("Enable")

    def _set_offroad_mode(result: int):
      if result == DialogResult.CONFIRM and not ui_state.engaged:
        ui_state.params.put_bool("OffroadMode", target_state)
        ui_state.always_offroad = target_state
        ui_state.preserve_settings_on_next_onroad_transition = not target_state

    gui_app.push_widget(ConfirmDialog(confirm_text, confirm_label, callback=_set_offroad_mode))

  def _handle_mouse_release(self, mouse_pos):
    self._prompt_toggle()
