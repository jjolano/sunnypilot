"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Shared helper for folding several schema panels into one consolidated page.

Each source panel becomes a titled group (its sub-section titles suppressed so
controls stay interleaved), and redundant "Enable " prefixes are dropped from
toggle titles. Pure (no pyray) — unit-testable headless. Device-side only: the
source schema panels and the cloud frontend are untouched.
"""
from openpilot.sunnypilot.selfdrive.ui.settings_schema.schema_loader import get_panel, load_schema


def _shorten(item: dict) -> dict:
  out = dict(item)
  title = out.get("title", "")
  if isinstance(title, str) and title.startswith("Enable "):
    out["title"] = title[len("Enable "):]
  if "sub_items" in out:
    out["sub_items"] = [_shorten(s) for s in out["sub_items"]]
  return out


def _untitled(sections: list) -> list:
  return [{**s, "title": "", "items": [_shorten(i) for i in s.get("items", [])]} for s in sections]


def _group(title: str) -> dict:
  return {"id": title.lower().replace(" ", "_"), "title": title, "items": [], "sub_panels": []}


def combined_panel(panel_id: str, label: str, groups: list[tuple[str, str]],
                   schema: dict | None = None) -> dict:
  """Build a consolidated panel dict.

  `groups` is an ordered list of (group title, source panel id). Each source
  panel's sections are inlined under a group header.
  """
  schema = schema if schema is not None else load_schema()
  sections: list[dict] = []
  for group_title, src_id in groups:
    src = get_panel(schema, src_id)
    if src is None:
      raise ValueError(f"source panel {src_id!r} not found")
    sections.append(_group(group_title))
    sections.extend(_untitled(src.get("sections", [])))
  return {"id": panel_id, "label": label, "sections": sections}
