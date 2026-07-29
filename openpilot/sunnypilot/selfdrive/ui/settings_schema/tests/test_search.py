"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Global settings search engine. Pure/headless.
"""
from __future__ import annotations

from openpilot.sunnypilot.selfdrive.ui.settings_schema.search import build_index, search


INDEX = build_index()


def _record(key: str):
  return next(r for r in INDEX if r.key == key)


def test_index_covers_settings_across_panels():
  keys = {r.key for r in INDEX}
  assert {"Mads", "EnforceTorqueControl", "BlindSpot",
          "OnroadScreenOffBrightness", "ExperimentalMode"} <= keys


def test_records_carry_panel_and_title():
  mads = _record("Mads")
  assert mads.panel_id == "steering"
  assert mads.title


def test_records_carry_route_metadata():
  expected = {
    "Mads": ("driving.steering", ("Driving", "Lateral Control")),
    "ExperimentalMode": ("driving.cruise", ("Driving", "Longitudinal Control")),
    "OnroadScreenOffBrightness": ("interface.display", ("Interface", "Display")),
    "BlindSpot": ("interface.visuals", ("Interface", "Visuals")),
  }
  for key, (route_id, breadcrumbs) in expected.items():
    rec = _record(key)
    assert rec.route_id == route_id
    assert rec.breadcrumbs == breadcrumbs


def test_legacy_live_panel_metadata_preserved():
  assert _record("Mads").live_panel_id == "driving"
  assert _record("OnroadScreenOffBrightness").live_panel_id == "interface"


def test_missing_route_metadata_falls_back_to_none_and_empty_breadcrumbs():
  schema = {
    "panels": [{
      "id": "orphan",
      "label": "Orphan",
      "sections": [{"items": [{"key": "OrphanKey", "title": "Orphan setting", "description": ""}]}],
    }],
  }
  rec = build_index(schema)[0]
  assert rec.route_id is None
  assert rec.breadcrumbs == ()


def test_search_torque():
  keys = [r.key for r in search("torque", INDEX)]
  assert "EnforceTorqueControl" in keys


def test_search_brightness():
  assert "OnroadScreenOffBrightness" in [r.key for r in search("brightness", INDEX)]


def test_search_phrase_in_title():
  assert "BlindSpot" in [r.key for r in search("blind spot", INDEX)]


def test_empty_query_returns_nothing():
  assert search("", INDEX) == []
  assert search("   ", INDEX) == []


def test_title_match_outranks_description_only():
  results = search("personality", INDEX)
  assert results
  assert "Personality" in results[0].title


def test_limit_respected():
  assert len(search("control", INDEX, limit=3)) <= 3


def test_build_index_includes_navigation_routes():
  schema = {
    "navigation": {"root": ["visible", "other"]},
    "pages": [
      {"id": "visible", "title": "Visible", "content": {"kind": "panel_ref", "panel": "visible_panel"}},
      {"id": "other", "title": "Other", "content": {"kind": "panel_ref", "panel": "other_panel"}},
    ],
    "panels": [
      {"id": "visible_panel", "label": "Visible", "icon": "a", "order": 0,
        "sections": [{"id": "s1", "title": "Section", "items": [{"key": "VisibleKey", "title": "Visible"}]}]},
      {"id": "other_panel", "label": "Other", "icon": "b", "order": 1,
       "sections": [{"id": "s2", "title": "Section", "items": [{"key": "OtherKey", "title": "Other"}]}]},
    ],
  }
  keys = {r.key for r in build_index(schema)}
  assert {"VisibleKey", "OtherKey"} <= keys


def test_search_includes_schema_pages():
  from openpilot.sunnypilot.selfdrive.ui.settings_schema.schema_loader import load_schema
  schema = load_schema()
  keys = {r.key for r in build_index(schema)}
  assert "Mads" in keys
  assert "ExperimentalMode" in keys
  assert "OnroadScreenOffBrightness" in keys
  assert "BlindSpot" in keys
