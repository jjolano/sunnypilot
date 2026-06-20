"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Tests for the settings_ui_src/ -> settings_ui.json compiler. Covers:
  - Roundtrip: compiled output matches the checked-in settings_ui.json
  - $ref macro resolution semantics (list-splice, scalar-substitute, depth, cycles)
  - Passive navigation tree integrity and additive-only emission

Does not cover device-side generator (test_settings_schema.py) or per-bug
regression (test_settings_changes.py); those continue to validate the
compiled output once the compiler has produced it.
"""
from __future__ import annotations

import difflib
import json
import os

import pytest
import yaml

from openpilot.sunnypilot.sunnylink.tools.compile_settings_ui import (
  CompileError,
  DEFAULT_OUT,
  DEFAULT_SRC,
  _canon_navigation,
  _resolve_refs,
  compile_schema,
)


SCHEMA_PATH = os.path.join(os.path.dirname(DEFAULT_OUT), "settings_ui.schema.json")


@pytest.fixture(scope="module")
def compiled() -> dict:
  return compile_schema(DEFAULT_SRC)


@pytest.fixture(scope="module")
def committed() -> dict:
  with open(DEFAULT_OUT) as f:
    return json.load(f)


class TestRoundtrip:
  def test_compiled_matches_committed(self, compiled, committed):
    """Compiled output must match the checked-in JSON."""
    if compiled == committed:
      return
    diff = "\n".join(difflib.unified_diff(
      json.dumps(committed, indent=2).splitlines(),
      json.dumps(compiled, indent=2).splitlines(),
      fromfile="settings_ui.json (committed)",
      tofile="settings_ui.json (freshly compiled)",
      lineterm="",
    ))
    pytest.fail(f"settings_ui.json schema mismatch — run compile_settings_ui.py\n\n{diff}")

  def test_additive_top_level_shape(self, compiled, committed):
    assert compiled["panels"] == committed["panels"]
    assert compiled["vehicle_settings"] == committed["vehicle_settings"]
    assert compiled["navigation"]
    assert compiled["pages"]

  def test_committed_file_validates_against_json_schema(self, committed):
    jsonschema = pytest.importorskip("jsonschema")
    with open(SCHEMA_PATH) as f:
      validator = json.load(f)
    jsonschema.validate(instance=committed, schema=validator)

  def test_committed_file_is_canonical(self):
    """Compiled output must byte-match the checked-in file (including trailing newline).
    Drift means someone edited settings_ui.json by hand instead of editing settings_ui_src/."""
    schema = compile_schema(DEFAULT_SRC)
    rendered = json.dumps(schema, indent=2) + "\n"
    with open(DEFAULT_OUT) as f:
      current = f.read()
    if current == rendered:
      return
    diff = "\n".join(difflib.unified_diff(
      current.splitlines(),
      rendered.splitlines(),
      fromfile="settings_ui.json (on disk)",
      tofile="settings_ui.json (freshly compiled)",
      lineterm="",
    ))
    pytest.fail(f"settings_ui.json out of sync — run compile_settings_ui.py\n\n{diff}")


class TestRefResolution:
  def test_list_context_splices(self):
    macros = {"a": [{"type": "offroad_only"}], "b": [{"type": "not_engaged"}]}
    out = _resolve_refs([{"$ref": "#/macros/a"}, {"$ref": "#/macros/b"}], macros)
    assert out == [{"type": "offroad_only"}, {"type": "not_engaged"}]

  def test_scalar_context_substitutes(self):
    macros = {"x": {"type": "capability", "field": "brand", "equals": "tesla"}}
    out = _resolve_refs({"condition": {"$ref": "#/macros/x"}}, macros)
    assert out == {"condition": {"type": "capability", "field": "brand", "equals": "tesla"}}

  def test_chained_ref_resolves(self):
    macros = {
      "leaf": [{"type": "offroad_only"}],
      "middle": [{"$ref": "#/macros/leaf"}],
    }
    out = _resolve_refs([{"$ref": "#/macros/middle"}], macros)
    assert out == [{"type": "offroad_only"}]

  def test_unknown_macro_raises(self):
    with pytest.raises(CompileError, match="unknown macro"):
      _resolve_refs([{"$ref": "#/macros/missing"}], {})

  def test_cycle_raises(self):
    macros = {"a": [{"$ref": "#/macros/b"}], "b": [{"$ref": "#/macros/a"}]}
    with pytest.raises(CompileError, match="cycle"):
      _resolve_refs([{"$ref": "#/macros/a"}], macros)

  def test_depth_limit(self):
    # Depth 4 chain should fail (limit is 3).
    macros = {
      "l1": [{"$ref": "#/macros/l2"}],
      "l2": [{"$ref": "#/macros/l3"}],
      "l3": [{"$ref": "#/macros/l4"}],
      "l4": [{"type": "offroad_only"}],
    }
    with pytest.raises(CompileError, match="depth"):
      _resolve_refs([{"$ref": "#/macros/l1"}], macros)

  def test_invalid_ref_scheme(self):
    with pytest.raises(CompileError, match="unsupported"):
      _resolve_refs([{"$ref": "https://example.com/x"}], {})

  def test_scalar_macro_in_list_context_raises(self):
    macros = {"x": {"type": "offroad_only"}}  # macro is a single rule (dict), not a list
    with pytest.raises(CompileError, match="must resolve to a list"):
      _resolve_refs([{"$ref": "#/macros/x"}], macros)


class TestCompiledShape:
  def test_panels_present(self, compiled):
    assert isinstance(compiled["panels"], list)
    assert len(compiled["panels"]) == 9
    panel_ids = {p["id"] for p in compiled["panels"]}
    assert {"steering", "cruise", "display", "visuals", "toggles",
            "device", "software", "developer", "models"} <= panel_ids

  def test_lead_path_clearance_mode_is_off_shadow_only(self, compiled):
    cruise = next(p for p in compiled["panels"] if p["id"] == "cruise")
    item = next(i for i in cruise["sections"][0]["items"] if i["key"] == "LeadPathClearanceMode")
    assert [opt["value"] for opt in item["options"]] == ["off", "shadow"]

  def test_longitudinal_debug_trace_mode_is_off_log_only(self, compiled):
    cruise = next(p for p in compiled["panels"] if p["id"] == "cruise")
    item = next(i for i in cruise["sections"][0]["items"] if i["key"] == "LongitudinalDebugTraceMode")
    assert [opt["value"] for opt in item["options"]] == ["off", "log"]

  def test_shadow_observability_modes_are_off_shadow_only(self, compiled):
    cruise = next(p for p in compiled["panels"] if p["id"] == "cruise")
    for key in ("CutInBrakeAssistMode", "CurveSpeedConfidenceMode", "StandstillReleaseConfidenceMode"):
      item = next(i for i in cruise["sections"][0]["items"] if i["key"] == key)
      assert [opt["value"] for opt in item["options"]] == ["off", "shadow"]

  def test_vehicle_settings_consistent_shape(self, compiled):
    """Each brand in vehicle_settings must have {title, description, items}."""
    for brand, data in compiled["vehicle_settings"].items():
      assert isinstance(data, dict), f"{brand}: expected object, got {type(data).__name__}"
      assert "title" in data, f"{brand}: missing title"
      assert "description" in data, f"{brand}: missing description"
      assert "items" in data, f"{brand}: missing items"

  def test_no_dangling_refs_after_compile(self, compiled):
    """All $ref objects must be resolved during compilation."""
    def walk(node):
      if isinstance(node, dict):
        if "$ref" in node:
          pytest.fail(f"unresolved $ref: {node}")
        for v in node.values():
          walk(v)
      elif isinstance(node, list):
        for x in node:
          walk(x)
    walk(compiled)

  def test_navigation_tree_is_valid(self, compiled):
    nav = compiled["navigation"]
    pages = compiled["pages"]
    page_ids = [p["id"] for p in pages]
    assert len(page_ids) == len(set(page_ids))
    assert nav["root"] == ["driving", "interface", "vehicle", "system"]
    assert set(nav["root"]).issubset(page_ids)

    page_map = {p["id"]: p for p in pages}
    panel_ids = {p["id"] for p in compiled["panels"]}
    allowed = {"device", "network", "sunnylink", "toggles", "software", "models", "osm", "trips", "vehicle", "firehose", "developer"}

    def walk(pid, stack):
      assert pid not in stack, f"cycle at {pid}"
      stack = stack + [pid]
      page = page_map[pid]
      has_children = "children" in page
      has_content = "content" in page
      assert has_children ^ has_content
      if has_children:
        assert len(stack) <= 3
        for child in page["children"]:
          assert child in page_map
          walk(child, stack)
      else:
        content = page["content"]
        if content["kind"] == "custom_page":
          assert content["component"] in allowed
        elif content["kind"] == "panel_ref":
          assert content["panel"] in panel_ids
        else:
          pytest.fail(f"unknown page content kind: {content['kind']}")

    for root in nav["root"]:
      walk(root, [])

    reachable = set()
    def collect(pid):
      if pid in reachable:
        return
      reachable.add(pid)
      for child in page_map[pid].get("children", []):
        collect(child)
    for root in nav["root"]:
      collect(root)
    assert reachable == set(page_ids)


def _nav_leaf(pid: str = "root", panel: str = "steering") -> dict:
  return {"id": pid, "title": pid, "content": {"kind": "panel_ref", "panel": panel}}


class TestNavigationValidation:
  PANELS = {"steering", "cruise", "display", "visuals", "toggles"}

  def test_navigation_pages_are_canonicalized(self):
    nav, pages = _canon_navigation({
      "root": ["root"],
      "pages": [{
        "id": "root",
        "title": "Root",
        "source_only": "ignored",
        "content": {"kind": "panel_ref", "panel": "steering", "source_only": "ignored"},
      }],
    }, self.PANELS)
    assert nav == {"root": ["root"]}
    assert pages == [{"id": "root", "title": "Root", "content": {"kind": "panel_ref", "panel": "steering"}}]

  @pytest.mark.parametrize("nav_doc, match", [
    ({"root": ["root", "root"], "pages": [_nav_leaf()]}, "duplicate"),
    ({"root": ["root"], "pages": [{"id": "root", "title": "Root", "children": ["child", "child"]}, _nav_leaf("child")]}, "duplicate"),
    ({"root": ["root"], "pages": [{"id": "root", "title": "Root", "children": ["missing"]}]}, "unknown child"),
    ({"root": ["root"], "pages": [_nav_leaf(panel="missing")]}, "unknown panel"),
    ({"root": ["root"], "pages": [{"id": "root", "title": "Root", "content": {"kind": "custom_page", "component": "missing"}}]}, "unknown custom"),
    ({"root": ["a", "b"], "pages": [
      {"id": "a", "title": "A", "children": ["shared"]},
      {"id": "b", "title": "B", "children": ["shared"]},
      _nav_leaf("shared"),
    ]}, "multiple parents"),
    ({"root": ["a"], "pages": [
      {"id": "a", "title": "A", "children": ["b"]},
      {"id": "b", "title": "B", "children": ["c"]},
      {"id": "c", "title": "C", "children": ["d"]},
      _nav_leaf("d"),
    ]}, "depth"),
  ])
  def test_invalid_navigation_raises(self, nav_doc, match):
    with pytest.raises(CompileError, match=match):
      _canon_navigation(nav_doc, self.PANELS)


class TestSourceTreeIntegrity:
  def test_macros_yaml_well_formed(self):
    with open(os.path.join(DEFAULT_SRC, "_macros.yaml")) as f:
      doc = yaml.safe_load(f)
    assert "macros" in doc
    for name, body in doc["macros"].items():
      assert name.replace("_", "").isalnum(), f"macro name '{name}' must be alphanumeric_"
      assert body, f"macro '{name}' empty"

  def test_pages_dir_well_formed(self):
    pages_dir = os.path.join(DEFAULT_SRC, "pages")
    assert os.path.isdir(pages_dir), "pages/ directory missing"
    page_files = sorted(fn for fn in os.listdir(pages_dir) if fn.endswith(".yaml"))
    # 9 panels + 1 vehicle = 10
    assert len(page_files) == 10, f"expected 10 pages, found {len(page_files)}: {page_files}"

  def test_every_page_has_id(self):
    pages_dir = os.path.join(DEFAULT_SRC, "pages")
    for fn in sorted(os.listdir(pages_dir)):
      if not fn.endswith(".yaml"):
        continue
      path = os.path.join(pages_dir, fn)
      with open(path) as f:
        doc = yaml.safe_load(f)
      assert isinstance(doc, dict), f"{path}: top-level must be a mapping"
      assert "id" in doc, f"{path}: page missing 'id'"
      # File basename should match page id (modulo .yaml extension).
      expected_id = os.path.splitext(fn)[0]
      assert doc["id"] == expected_id, (
        f"{path}: page id '{doc['id']}' must match filename '{expected_id}'"
      )

  def test_vehicle_page_kind(self):
    """vehicle.yaml must declare kind: vehicle so it routes to vehicle_settings."""
    path = os.path.join(DEFAULT_SRC, "pages", "vehicle.yaml")
    with open(path) as f:
      doc = yaml.safe_load(f)
    assert doc.get("kind") == "vehicle", "vehicle.yaml must declare kind: vehicle"
