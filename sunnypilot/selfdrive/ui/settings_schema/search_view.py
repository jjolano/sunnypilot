"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Global settings search panel.

A search field (magnifier + query text) that opens the on-screen keyboard on tap,
over a list of ranked result rows. Each result shows the setting title + the panel
it now lives in, and tapping it jumps there via the supplied navigate callback.
`set_query` drives the results, so the screenshot harness can render a fixed query
without typing.
"""
from collections.abc import Callable

import pyray as rl

from openpilot.system.ui.lib.application import FontWeight, MousePos, gui_app
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.sunnypilot.lib.styles import style
from openpilot.system.ui.sunnypilot.widgets.input_dialog import InputDialogSP
from openpilot.system.ui.sunnypilot.widgets.list_view import simple_button_item_sp
from openpilot.system.ui.widgets import DialogResult, Widget
from openpilot.system.ui.widgets.scroller_tici import Scroller

from openpilot.sunnypilot.selfdrive.ui.settings_schema.search import build_index, search

_BOX_HEIGHT = 150
_RESULT_WIDTH = 1900


class SearchLayout(Widget):
  def __init__(self, navigate_callback: Callable[[str, str], None] | None = None,
               query: str = "", index=None):
    super().__init__()
    self._navigate = navigate_callback
    self._index = index if index is not None else build_index()
    self._query = ""
    self._box_rect = rl.Rectangle(0, 0, 0, 0)
    self._results = Scroller([], line_separator=True, spacing=0)
    self.set_query(query)

  def _open_keyboard(self):
    InputDialogSP(tr("Search settings"), current_text=self._query, callback=self._on_query).show()

  def _on_query(self, result: DialogResult, text: str):
    if result == DialogResult.CONFIRM:
      self.set_query(text)

  def set_query(self, query: str):
    self._query = query
    rows: list[Widget] = []
    for rec in search(query, self._index):
      label = f"{rec.title}     ·     {rec.live_panel_label}"
      rows.append(simple_button_item_sp(
        button_text=label, button_width=_RESULT_WIDTH,
        callback=lambda r=rec: self._navigate(r.live_panel_id, r.key) if self._navigate else None))
    self._results = Scroller(rows, line_separator=True, spacing=0)

  def _draw_search_field(self, rect: rl.Rectangle):
    self._box_rect = rl.Rectangle(rect.x, rect.y, rect.width, _BOX_HEIGHT)
    rl.draw_rectangle_rounded(self._box_rect, 0.3, 16, style.BASE_BG_COLOR)
    # magnifier: lens ring + handle
    cx, cy, r = rect.x + 58, rect.y + _BOX_HEIGHT / 2, 24
    icon_color = style.ITEM_DESC_TEXT_COLOR
    rl.draw_ring(rl.Vector2(cx, cy), r - 5, r, 0, 360, 32, icon_color)
    rl.draw_line_ex(rl.Vector2(cx + r * 0.7, cy + r * 0.7), rl.Vector2(cx + r * 1.4, cy + r * 1.4), 7, icon_color)
    # query / placeholder
    text = self._query or tr("Search settings")
    color = style.ITEM_TEXT_COLOR if self._query else style.ITEM_DESC_TEXT_COLOR
    rl.draw_text_ex(gui_app.font(FontWeight.NORMAL), text,
                    rl.Vector2(rect.x + 115, rect.y + _BOX_HEIGHT / 2 - 28), 52, 0, color)

  def _render(self, rect: rl.Rectangle):
    self._draw_search_field(rect)
    self._results.render(rl.Rectangle(rect.x, rect.y + _BOX_HEIGHT + 20,
                                      rect.width, rect.height - _BOX_HEIGHT - 20))

  def _handle_mouse_release(self, mouse_pos: MousePos):
    if rl.check_collision_point_rec(mouse_pos, self._box_rect):
      self._open_keyboard()

  def show_event(self):
    self._results.show_event()
