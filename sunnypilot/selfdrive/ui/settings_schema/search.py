"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Global settings search.

Settings are data, so a search index over every schema item (title, description,
param key) is cheap. This is the engine — pure, headless-testable; the search
overlay UI consumes `build_index` + `search`. Records carry the panel a setting
lives in so the UI can jump straight to it.
"""
from dataclasses import dataclass

from openpilot.sunnypilot.selfdrive.ui.settings_schema.schema_loader import iter_items, load_schema, routes_for_panel


# Source schema panel -> the consolidated panel it now lives in (device IA).
# Anything not listed lives in a panel of its own id.
_CONSOLIDATION = {
  "steering": ("driving", "Driving"),
  "cruise": ("driving", "Driving"),
  "visuals": ("interface", "Interface"),
  "display": ("interface", "Interface"),
}


@dataclass(frozen=True)
class SearchRecord:
  key: str
  title: str
  description: str
  panel_id: str         # source schema panel
  panel_label: str
  route_id: str | None  # navigation page id when available
  breadcrumbs: tuple[str, ...]
  live_panel_id: str    # consolidated panel the setting is reached through
  live_panel_label: str


def _route_for_panel(schema: dict, panel_id: str) -> tuple[str | None, tuple[str, ...]]:
  routes = routes_for_panel(schema, panel_id)
  if not routes:
    return None, ()
  route = routes[0]
  return route.page_id, route.breadcrumbs


def build_index(schema: dict | None = None) -> list[SearchRecord]:
  schema = schema if schema is not None else load_schema()
  records: list[SearchRecord] = []
  for panel in schema.get("panels", []):
    pid = panel.get("id", "")
    plabel = panel.get("label", pid)
    route_id, breadcrumbs = _route_for_panel(schema, pid)
    live_id, live_label = _CONSOLIDATION.get(pid, (pid, plabel))
    for item in iter_items(panel):
      if "key" not in item:
        continue
      records.append(SearchRecord(
        key=item["key"], title=item.get("title", ""), description=item.get("description", ""),
        panel_id=pid, panel_label=plabel, route_id=route_id, breadcrumbs=breadcrumbs,
        live_panel_id=live_id, live_panel_label=live_label,
      ))
  return records


def _score(query: str, rec: SearchRecord) -> int:
  """Higher is better. Title hit > key hit > description hit; word-prefix bonus."""
  score = 0
  title = rec.title.lower()
  if query in title:
    score += 4
    if title.startswith(query) or f" {query}" in title:  # word-boundary match
      score += 2
  if query in rec.key.lower():
    score += 3
  if query in rec.description.lower():
    score += 1
  return score


def search(query: str, index: list[SearchRecord], limit: int = 12) -> list[SearchRecord]:
  q = query.lower().strip()
  if not q:
    return []
  scored = [(_score(q, r), r) for r in index]
  hits = [(s, r) for s, r in scored if s > 0]
  hits.sort(key=lambda sr: (-sr[0], sr[1].title))
  return [r for _, r in hits[:limit]]
