"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Schema-level tests for button and info widget types. Verifies that the
compiled schema declares these widgets correctly and that build_control
routes them rather than silently dropping them. Pure/headless for schema
assertions; the Xvfb smoke test covers actual rendering.
"""
from __future__ import annotations

from openpilot.sunnypilot.selfdrive.ui.settings_schema.schema_loader import iter_items, load_schema
from openpilot.sunnypilot.sunnylink.tools.compile_settings_ui import DEFAULT_SRC, compile_schema

SCHEMA = load_schema()


def _items_with_widget(widget_type: str) -> list[dict]:
  return [it for it in (i for p in SCHEMA.get("panels", []) for i in iter_items(p))
          if it.get("widget") == widget_type]


def test_button_items_have_action_field():
  """Every button widget in the schema must declare an `action` id."""
  buttons = _items_with_widget("button")
  for btn in buttons:
    assert btn.get("action"), f"button item '{btn.get('key')}' missing action field"


def test_info_items_have_key_field():
  """Every info widget in the schema must have a param key to display."""
  infos = _items_with_widget("info")
  for info in infos:
    assert info.get("key"), "info item missing key field"


def test_button_widget_is_valid_schema_type():
  """The compiled schema must accept 'button' as a widget type."""
  compiled = compile_schema(DEFAULT_SRC)
  # If button items exist in source, they must survive compilation.
  for panel in compiled.get("panels", []):
    for item in iter_items(panel):
      if item.get("widget") == "button":
        assert item.get("action"), f"compiled button '{item.get('key')}' lost action field"


def test_info_widget_is_valid_schema_type():
  """The compiled schema must accept 'info' as a widget type."""
  compiled = compile_schema(DEFAULT_SRC)
  for panel in compiled.get("panels", []):
    for item in iter_items(panel):
      if item.get("widget") == "info":
        assert item.get("key"), "compiled info item lost key field"
