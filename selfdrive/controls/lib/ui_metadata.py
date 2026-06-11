from __future__ import annotations

from typing import Any

import openpilot.selfdrive.controls.lib.feature_registry_entries  # noqa: F401  (registers feature entries on import)
from openpilot.selfdrive.controls.lib.feature_registry import feature_registry_snapshot
from openpilot.selfdrive.controls.lib.lateral_demand_stacks.selector import (
  load_lateral_demand_stack_manifest,
)
from openpilot.selfdrive.controls.lib.longitudinal_stacks.selector import (
  load_stack_manifest,
)


def get_lateral_manifest_summary() -> dict[str, Any]:
  manifest = load_lateral_demand_stack_manifest()
  return {
    "default_stack": manifest.get("defaultStack"),
    "custom_recommended_fallback": manifest.get("customRecommendedFallback"),
    "stacks": manifest.get("stacks", {}),
    "availability": manifest.get("availability", {}),
    "custom_recommendations": manifest.get("customRecommendations", {}),
  }


def get_longitudinal_manifest_summary() -> dict[str, Any]:
  manifest = load_stack_manifest()
  return {
    "default_stack": manifest.get("defaultStack"),
    "custom_recommended_fallback": manifest.get("customRecommendedFallback"),
    "stacks": manifest.get("stacks", {}),
    "availability": manifest.get("availability", {}),
    "custom_recommendations": manifest.get("customRecommendations", {}),
  }


def get_feature_registry_summary() -> dict[str, dict[str, Any]]:
  return feature_registry_snapshot()


def get_controls_profile_metadata() -> dict[str, Any]:
  return {
    "lateral_manifest": get_lateral_manifest_summary(),
    "longitudinal_manifest": get_longitudinal_manifest_summary(),
    "feature_registry": get_feature_registry_summary(),
  }
