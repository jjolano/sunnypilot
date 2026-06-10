import openpilot.selfdrive.controls.lib.feature_registry_entries  # noqa: F401
from openpilot.selfdrive.controls.lib.ui_metadata import (
  get_controls_profile_metadata,
  get_feature_registry_summary,
  get_lateral_manifest_summary,
  get_longitudinal_manifest_summary,
)


def test_lateral_manifest_summary_lists_all_four_stacks():
  summary = get_lateral_manifest_summary()
  assert summary["default_stack"] == "custom-2.0"
  assert "sunnypilot-current" in summary["stacks"]
  assert "custom-recommended" in summary["stacks"]
  assert "custom-2.0" in summary["stacks"]
  assert "custom-experimental" in summary["stacks"]
  assert "custom-experimental" in summary["availability"]


def test_longitudinal_manifest_summary_lists_all_four_stacks():
  summary = get_longitudinal_manifest_summary()
  assert "sunnypilot-current" in summary["stacks"]
  assert "custom-2.0" in summary["stacks"]
  assert "custom-experimental" in summary["stacks"]


def test_feature_registry_summary_exposes_capabilities():
  summary = get_feature_registry_summary()
  assert "custom-2.0" in summary
  assert "sunnypilot-current" in summary
  assert summary["custom-2.0"]["features"]["one_pedal_longitudinal"] is True
  assert summary["sunnypilot-current"]["features"]["one_pedal_longitudinal"] is False


def test_full_metadata_has_three_sections():
  md = get_controls_profile_metadata()
  assert "lateral_manifest" in md
  assert "longitudinal_manifest" in md
  assert "feature_registry" in md


def test_lateral_manifest_custom_recommendations_section():
  summary = get_lateral_manifest_summary()
  assert "default" in summary["custom_recommendations"]
  assert "brands" in summary["custom_recommendations"]
  assert "fingerprints" in summary["custom_recommendations"]
