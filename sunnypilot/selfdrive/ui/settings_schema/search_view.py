"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Global settings search panel.

A search field (magnifier + query text) that opens the on-screen keyboard on tap,
over a list of ranked result rows. Each row shows the setting title (left) and the
panel it now lives in (right, muted), and tapping it jumps there via the supplied
navigate callback. `set_query` drives the results, so the screenshot harness can
render a fixed query without typing.
"""
from collections.abc import Callable

import pyray as rl

from openpilot.system.ui.lib.application import FontWeight, MousePos, gui_app
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.sunnypilot.lib.styles import style
from openpilot.system.ui.sunnypilot.widgets.input_dialog import InputDialogSP
from openpilot.system.ui.widgets import DialogResult, Widget
from openpilot.system.ui.widgets.scroller_tici import Scroller

from openpilot.sunnypilot.selfdrive.ui.settings_schema.search import SearchRecord, build_index, search

_BOX_HEIGHT = 150
_ROW_HEIGHT = 150


class _SearchResultRow(Widget):
  def __init__(self, rec: SearchRecord, on_click: Callable[[], None], context_label: str):
    super().__init__()
    self._rec = rec
    self._on_click = on_click
    self._context_label = context_label
    self.set_rect(rl.Rectangle(0, 0, 0, _ROW_HEIGHT))

  def set_parent_rect(self, parent_rect: rl.Rectangle) -> None:
    super().set_parent_rect(parent_rect)
    self._rect.width = parent_rect.width  # the Scroller sets position but not width

  def _render(self, rect: rl.Rectangle):
    font = gui_app.font(FontWeight.NORMAL)
    label = self._context_label
    lsize = measure_text_cached(font, label, 40)
    rl.draw_text_ex(font, label, rl.Vector2(rect.x + rect.width - lsize.x - 30, rect.y + _ROW_HEIGHT / 2 - 20),
                    40, 0, style.ITEM_DESC_TEXT_COLOR)
    rl.draw_text_ex(font, self._rec.title, rl.Vector2(rect.x + 30, rect.y + _ROW_HEIGHT / 2 - 26),
                    50, 0, style.ITEM_TEXT_COLOR)

  def _handle_mouse_release(self, mouse_pos: MousePos):
    if self._on_click:
      self._on_click()


class SearchLayout(Widget):
  def __init__(self, navigate_callback: Callable[[str, str], None] | None = None,
               query: str = "", index=None,
               record_callback: Callable[[SearchRecord], None] | None = None,
               result_context_label: Callable[[SearchRecord], str] | None = None):
    super().__init__()
    self._navigate = navigate_callback
    self._record_callback = record_callback
    self._result_context_label = result_context_label or (lambda rec: rec.live_panel_label)
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

  def _navigate_to(self, rec: SearchRecord):
    if self._record_callback:
      self._record_callback(rec)
      return
    if self._navigate:
      self._navigate(rec.live_panel_id, rec.key)

  def set_query(self, query: str):
    self._query = query
    rows: list[Widget] = [_SearchResultRow(rec, lambda r=rec: self._navigate_to(r), self._result_context_label(rec))
                          for rec in search(query, self._index)]
    self._results = Scroller(rows, line_separator=True, spacing=0)

  def _draw_search_field(self, rect: rl.Rectangle):
    self._box_rect = rl.Rectangle(rect.x, rect.y, rect.width, _BOX_HEIGHT)
    rl.draw_rectangle_rounded(self._box_rect, 0.3, 16, style.BASE_BG_COLOR)
    cx, cy, r = rect.x + 58, rect.y + _BOX_HEIGHT / 2, 24
    icon_color = style.ITEM_DESC_TEXT_COLOR
    rl.draw_ring(rl.Vector2(cx, cy), r - 5, r, 0, 360, 32, icon_color)
    rl.draw_line_ex(rl.Vector2(cx + r * 0.7, cy + r * 0.7), rl.Vector2(cx + r * 1.4, cy + r * 1.4), 7, icon_color)
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
