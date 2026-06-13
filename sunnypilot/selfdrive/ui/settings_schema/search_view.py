"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Global settings search panel.

A tappable search box (opens the on-screen keyboard) over a list of ranked result
rows; each result shows the setting title + the panel it now lives in, and tapping
it jumps there via the supplied navigate callback. `set_query` drives the results,
so the screenshot harness can render a fixed query without typing.
"""
from collections.abc import Callable

import pyray as rl

from openpilot.system.ui.lib.multilang import tr
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
    self._search_button = simple_button_item_sp(
      button_text=lambda: self._query or tr("Search settings…"),
      callback=self._open_keyboard, button_width=_RESULT_WIDTH)
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

  def _render(self, rect: rl.Rectangle):
    self._search_button.set_parent_rect(rect)
    self._search_button.render(rl.Rectangle(rect.x, rect.y, rect.width, _BOX_HEIGHT))
    self._results.render(rl.Rectangle(rect.x, rect.y + _BOX_HEIGHT + 20,
                                      rect.width, rect.height - _BOX_HEIGHT - 20))

  def show_event(self):
    self._results.show_event()
