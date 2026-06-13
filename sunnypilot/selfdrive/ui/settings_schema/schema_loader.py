"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Load and walk the compiled settings_ui.json. No rendering dependencies.

The compiled schema is the canonical artifact (settings_ui_src/*.yaml compiled
by tools/compile_settings_ui.py); macros are already expanded, so the rule
evaluator never sees a {$ref}. We deliberately read the static file here rather
than generate_settings_schema.generate_schema() so this stays import-light and
headless; the renderer can swap in the dynamic-options variant when wired live.
"""
from __future__ import annotations

import json
import os
from collections.abc import Iterator

from openpilot.common.basedir import BASEDIR

SETTINGS_UI_JSON = os.path.join(BASEDIR, "sunnypilot", "sunnylink", "settings_ui.json")


def load_schema(path: str | None = None) -> dict:
  with open(path or SETTINGS_UI_JSON) as f:
    return json.load(f)


def get_panel(schema: dict, panel_id: str) -> dict | None:
  return next((p for p in schema.get("panels", []) if p.get("id") == panel_id), None)


def _walk_item(item: dict) -> Iterator[dict]:
  """Yield an item and, recursively, its sub_items."""
  yield item
  for sub in item.get("sub_items", []):
    yield from _walk_item(sub)


def iter_items(panel: dict) -> Iterator[dict]:
  """Yield every control item in a panel, recursing sections, sub_panels, sub_items.

  Mirrors generate_settings_schema._walk_all_items so the renderer sees exactly
  the same item set the schema tooling validates.
  """
  for section in panel.get("sections", []):
    for item in section.get("items", []):
      yield from _walk_item(item)
    for sp in section.get("sub_panels", []):
      for item in sp.get("items", []):
        yield from _walk_item(item)
  for item in panel.get("items", []):
    yield from _walk_item(item)
  for sp in panel.get("sub_panels", []):
    for item in sp.get("items", []):
      yield from _walk_item(item)


def find_item(panel: dict, key: str) -> dict | None:
  return next((it for it in iter_items(panel) if it.get("key") == key), None)


def plan_page(panel: dict, with_sections: bool = False) -> list[dict]:
  """Flatten a panel into an ordered render plan for the device top level.

  Returns a list of entries in display order:
    {"kind": "section", "title", "id"}          a group header (only if with_sections)
    {"kind": "control", "item": <item>}         a rendered control (+ its sub_items)
    {"kind": "subpanel", "id", "label", "trigger"}  a nav button into a sub-panel

  Sub-panel *contents* are NOT inlined — they are reached via the nav button and
  rendered by their own layout. With `with_sections`, each titled section emits a
  header entry before its controls (used by consolidated pages like Driving).
  """
  plan: list[dict] = []

  def add_item(item: dict) -> None:
    plan.append({"kind": "control", "item": item})
    for sub in item.get("sub_items", []):
      add_item(sub)

  def add_subpanel(sp: dict) -> None:
    plan.append({
      "kind": "subpanel",
      "id": sp.get("id"),
      "label": sp.get("label", ""),
      "trigger": sp.get("trigger_condition"),
    })

  for section in panel.get("sections", []):
    title = section.get("title")
    if with_sections and title:
      plan.append({"kind": "section", "title": title, "id": section.get("id")})
    for item in section.get("items", []):
      add_item(item)
    for sp in section.get("sub_panels", []):
      add_subpanel(sp)
  for item in panel.get("items", []):
    add_item(item)
  for sp in panel.get("sub_panels", []):
    add_subpanel(sp)

  return plan


def iter_rules(schema_node: dict | list) -> Iterator[dict]:
  """Yield every rule node (recursing not/any/all) under an enablement/visibility list."""
  nodes = schema_node if isinstance(schema_node, list) else [schema_node]
  for rule in nodes:
    if not isinstance(rule, dict):
      continue
    yield rule
    if rule.get("type") == "not" and "condition" in rule:
      yield from iter_rules([rule["condition"]])
    elif rule.get("type") in ("any", "all"):
      yield from iter_rules(rule.get("conditions", []))
