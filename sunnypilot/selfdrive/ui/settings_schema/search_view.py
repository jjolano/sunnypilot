"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Global settings search overlay (results view).

Renders a search box plus the ranked results for the current query — each result
shows the setting title and the panel it lives in. Text entry and jump-to-setting
navigation wire in on top of this; `set_query` drives the result list.
"""
import pyray as rl

from openpilot.system.ui.lib.application import FontWeight, gui_app
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.sunnypilot.lib.styles import style
from openpilot.system.ui.sunnypilot.widgets.list_view import ListItemSP
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.scroller_tici import Scroller

from openpilot.sunnypilot.selfdrive.ui.settings_schema.search import build_index, search

_BOX_HEIGHT = 120


class SearchView(Widget):
  def __init__(self, query: str = "", index=None):
    super().__init__()
    self._index = index if index is not None else build_index()
    self._query = ""
    self._scroller = Scroller([], line_separator=True, spacing=0)
    self.set_query(query)

  def set_query(self, query: str):
    self._query = query
    rows: list[Widget] = []
    for rec in search(query, self._index):
      row = ListItemSP(title=tr(rec.title), description="")
      row.set_right_value(rec.panel_label)  # which panel the setting lives in
      rows.append(row)
    self._scroller = Scroller(rows, line_separator=True, spacing=0)

  def _render(self, rect: rl.Rectangle):
    box = rl.Rectangle(rect.x, rect.y, rect.width, _BOX_HEIGHT)
    rl.draw_rectangle_rounded(box, 0.25, 16, style.BASE_BG_COLOR)
    font = gui_app.font(FontWeight.NORMAL)
    shown = self._query or "Search settings"
    color = style.ITEM_TEXT_COLOR if self._query else style.ITEM_DESC_TEXT_COLOR
    rl.draw_text_ex(font, f"⚲  {shown}", rl.Vector2(rect.x + 40, rect.y + 34), 52, 0, color)

    results = rl.Rectangle(rect.x, rect.y + _BOX_HEIGHT + 24, rect.width, rect.height - _BOX_HEIGHT - 24)
    self._scroller.render(results)
