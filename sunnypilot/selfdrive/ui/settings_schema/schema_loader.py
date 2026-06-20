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
from dataclasses import dataclass
from collections.abc import Iterator

from openpilot.common.basedir import BASEDIR

SETTINGS_UI_JSON = os.path.join(BASEDIR, "sunnypilot", "sunnylink", "settings_ui.json")
MAX_NAV_DEPTH = 3


@dataclass(frozen=True)
class FocusAnchor:
  panel: str | None = None
  section: str | None = None
  sub_panel: str | None = None
  key: str | None = None


@dataclass(frozen=True)
class SettingsRoute:
  page_id: str
  breadcrumbs: tuple[str, ...] = ()
  focus: FocusAnchor | None = None


def load_schema(path: str | None = None) -> dict:
  with open(path or SETTINGS_UI_JSON) as f:
    return json.load(f)


def _pages(schema: dict) -> list[dict]:
  pages = schema.get("pages", [])
  return pages if isinstance(pages, list) else []


def get_page(schema: dict, page_id: str) -> dict | None:
  return next((p for p in _pages(schema) if p.get("id") == page_id), None)


def get_root_navigation(schema: dict) -> list[dict]:
  if navigation_errors(schema):
    return []
  nav = schema.get("navigation")
  if not isinstance(nav, dict):
    return []
  root = nav.get("root")
  if not isinstance(root, list):
    return []
  pages = []
  for page_id in root:
    if not isinstance(page_id, str):
      return []
    page = get_page(schema, page_id)
    if page is None:
      return []
    pages.append(page)
  return pages


def _page_title(page: dict) -> str | None:
  title = page.get("title")
  return title if isinstance(title, str) and title else None


def _page_content(page: dict) -> dict | None:
  content = page.get("content")
  return content if isinstance(content, dict) else None


def navigation_errors(schema: dict) -> list[str]:
  errors: list[str] = []
  nav = schema.get("navigation")
  pages = schema.get("pages")
  if not isinstance(nav, dict):
    errors.append("missing navigation")
    nav = {}
  if not isinstance(pages, list) or not pages:
    errors.append("missing pages")
    return errors

  page_by_id: dict[str, dict] = {}
  for page in pages:
    if not isinstance(page, dict):
      errors.append("invalid page")
      continue
    pid = page.get("id")
    if not isinstance(pid, str) or not pid:
      errors.append("invalid page id")
      continue
    if pid in page_by_id:
      errors.append(f"duplicate page id: {pid}")
    page_by_id[pid] = page
    if not isinstance(page.get("title"), str) or not page.get("title"):
      errors.append(f"invalid page title: {pid}")
    has_children = "children" in page
    has_content = "content" in page
    if has_children == has_content:
      errors.append(f"invalid page shape: {pid}")
    if has_children:
      children = page.get("children")
      if not isinstance(children, list) or not children or not all(isinstance(child, str) and child for child in children):
        errors.append(f"invalid category children: {pid}")
      elif len(set(children)) != len(children):
        errors.append(f"duplicate children: {pid}")
    elif has_content:
      content = _page_content(page)
      if content is None:
        errors.append(f"invalid leaf shape: {pid}")
      else:
        kind = content.get("kind")
        if kind == "panel_ref":
          panel = content.get("panel")
          if not isinstance(panel, str) or not panel or get_panel(schema, panel) is None:
            errors.append(f"unknown panel_ref panel: {pid}")
        elif kind == "custom_page":
          component = content.get("component")
          if not isinstance(component, str) or not component:
            errors.append(f"invalid custom_page: {pid}")
        else:
          errors.append(f"invalid content kind: {pid}")

  roots = nav.get("root", []) if isinstance(nav, dict) else []
  if not isinstance(roots, list) or not roots or not all(isinstance(root, str) and root for root in roots):
    errors.append("missing roots")
    return errors
  if len(set(roots)) != len(roots):
    errors.append("duplicate roots")
  parents: dict[str, str] = {}
  for root in roots:
    if root not in page_by_id:
      errors.append(f"unknown root: {root}")

  for page in pages:
    pid = page.get("id") if isinstance(page, dict) else None
    if not isinstance(pid, str) or pid not in page_by_id:
      continue
    children = page.get("children")
    if not isinstance(children, list):
      continue
    for child in children:
      if not isinstance(child, str):
        continue
      if child in parents:
        errors.append(f"multiple parents: {child}")
      parents[child] = pid
      if child not in page_by_id:
        errors.append(f"unknown child: {child}")
  root_set = set(roots)
  for child in parents:
    if child in root_set:
      errors.append(f"root-as-child: {child}")
  for pid, page in page_by_id.items():
    if pid not in roots and pid not in parents:
      errors.append(f"orphan page: {pid}")

  visited: set[str] = set()
  visiting: set[str] = set()

  def walk(page_id: str, depth: int) -> None:
    if depth > MAX_NAV_DEPTH:
      errors.append(f"max depth > {MAX_NAV_DEPTH}: {page_id}")
      return
    if page_id in visiting:
      errors.append(f"cycle: {page_id}")
      return
    if page_id in visited:
      return
    page = page_by_id.get(page_id)
    if page is None:
      return
    visiting.add(page_id)
    children = page.get("children")
    if isinstance(children, list):
      for child in children:
        if isinstance(child, str):
          walk(child, depth + 1)
    visiting.remove(page_id)
    visited.add(page_id)

  for root in roots:
    if isinstance(root, str):
      walk(root, 1)
  return errors


def navigation_available(schema: dict) -> bool:
  return not navigation_errors(schema)


def _walk_routes(schema: dict, page_id: str, breadcrumbs: tuple[str, ...], seen: set[str]) -> list[SettingsRoute]:
  page = get_page(schema, page_id)
  if page is None or page_id in seen:
    return []
  seen = set(seen)
  seen.add(page_id)
  title = _page_title(page)
  next_breadcrumbs = breadcrumbs + ((title,) if title and title != "Settings" else ())
  routes = [SettingsRoute(page_id=page_id, breadcrumbs=next_breadcrumbs)]
  for child in page.get("children", []) if isinstance(page.get("children"), list) else []:
    if isinstance(child, str):
      routes.extend(_walk_routes(schema, child, next_breadcrumbs, seen))
  return routes


def breadcrumbs_for(schema: dict, page_id: str) -> tuple[str, ...]:
  for route in flatten_routes(schema):
    if route.page_id == page_id:
      return route.breadcrumbs
  return ()


def resolve_page_content(schema: dict, page_id: str) -> dict | None:
  if navigation_errors(schema):
    return None
  page = get_page(schema, page_id)
  if page is None:
    return None
  content = _page_content(page)
  if content is None:
    return None
  kind = content.get("kind")
  if kind in ("panel_ref", "custom_page"):
    return content
  return None


def panel_for_page(schema: dict, page_id: str) -> dict | None:
  content = resolve_page_content(schema, page_id)
  if not content or content.get("kind") != "panel_ref":
    return None
  panel = content.get("panel")
  return get_panel(schema, panel) if isinstance(panel, str) else None


def flatten_routes(schema: dict) -> list[SettingsRoute]:
  if not navigation_available(schema):
    return []
  routes: list[SettingsRoute] = []
  for root in get_root_navigation(schema):
    routes.extend(_walk_routes(schema, root["id"], (), set()))
  return routes


def routes_for_panel(schema: dict, panel_id: str) -> list[SettingsRoute]:
  return [route for route in flatten_routes(schema) if (content := resolve_page_content(schema, route.page_id)) and content.get("kind") == "panel_ref" and content.get("panel") == panel_id]


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
