"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Route-loader coverage for schema navigation helpers: root ordering, breadcrumbs,
panel resolution, route flattening, and defensive fallbacks.
"""
from __future__ import annotations

from copy import deepcopy

from openpilot.sunnypilot.selfdrive.ui.settings_schema.schema_loader import (
  breadcrumbs_for,
  flatten_routes,
  get_page,
  get_root_navigation,
  navigation_available,
  navigation_errors,
  panel_for_page,
  resolve_page_content,
  routes_for_panel,
  load_schema,
)


SCHEMA = load_schema()


def test_navigation_available_and_root_order():
  assert navigation_available(SCHEMA)
  assert [page["id"] for page in get_root_navigation(SCHEMA)] == ["driving", "interface", "vehicle", "system"]


def test_breadcrumbs_for_representative_leaves():
  assert breadcrumbs_for(SCHEMA, "driving.steering") == ("Driving", "Lateral Control")
  assert breadcrumbs_for(SCHEMA, "driving.cruise") == ("Driving", "Longitudinal Control")
  assert breadcrumbs_for(SCHEMA, "interface.display") == ("Interface", "Display")
  assert breadcrumbs_for(SCHEMA, "interface.osm") == ("Interface", "OpenStreetMap")
  assert breadcrumbs_for(SCHEMA, "system.network") == ("System", "Network")


def test_panel_for_page_resolves_panel_refs_only():
  steering = panel_for_page(SCHEMA, "driving.steering")
  cruise = panel_for_page(SCHEMA, "driving.cruise")
  assert steering is not None and steering["id"] == "steering"
  assert cruise is not None and cruise["id"] == "cruise"
  assert panel_for_page(SCHEMA, "driving.toggles") is None
  assert panel_for_page(SCHEMA, "interface.osm") is None


def test_flatten_routes_depth_first_and_breadcrumbed():
  routes = flatten_routes(SCHEMA)
  assert [r.page_id for r in routes][:8] == [
    "driving",
    "driving.steering",
    "driving.cruise",
    "driving.toggles",
    "interface",
    "interface.display",
    "interface.visuals",
    "interface.osm",
  ]
  assert routes[1].breadcrumbs == ("Driving", "Lateral Control")
  assert routes[3].breadcrumbs == ("Driving", "Core Toggles")


def test_routes_for_panel_filters_panel_refs():
  assert [route.page_id for route in routes_for_panel(SCHEMA, "steering")] == ["driving.steering"]


def test_invalid_navigation_falls_back_safely():
  broken = deepcopy(SCHEMA)
  broken.pop("navigation", None)
  broken.pop("pages", None)
  assert navigation_available(broken) is False
  assert get_root_navigation(broken) == []
  assert flatten_routes(broken) == []
  assert breadcrumbs_for(broken, "driving") == ()
  assert resolve_page_content(broken, "driving") is None
  assert panel_for_page(broken, "driving.steering") is None
  assert routes_for_panel(broken, "steering") == []


def test_malformed_pages_fall_back_safely():
  broken = deepcopy(SCHEMA)
  broken["pages"] = [123]
  assert navigation_available(broken) is False
  assert get_page(broken, "driving") is None
  assert flatten_routes(broken) == []


def test_navigation_errors_cover_common_tree_issues():
  broken = deepcopy(SCHEMA)
  broken["navigation"] = {"root": ["driving", "driving"]}
  broken["pages"] = [
    {"id": "driving", "title": "Driving", "children": ["shared", "missing", "driving"]},
    {"id": "interface", "title": "Interface", "children": ["shared"]},
    {"id": "shared", "title": "Shared", "content": {"kind": "panel_ref", "panel": "not-a-panel"}},
  ]
  errors = navigation_errors(broken)
  assert any("duplicate roots" in e for e in errors)
  assert any("multiple parents" in e for e in errors)
  assert any("root-as-child" in e for e in errors)
  assert any("unknown child" in e for e in errors)
  assert any("unknown panel_ref panel" in e for e in errors)
  assert navigation_available(broken) is False
  assert get_root_navigation(broken) == []
  assert flatten_routes(broken) == []
  assert resolve_page_content(broken, "shared") is None
