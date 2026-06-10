import pytest

from openpilot.selfdrive.controls.lib.lateral_demand_stacks.selector import (
  CUSTOM_EXPERIMENTAL,
  CUSTOM_RECOMMENDED,
  CUSTOM_V2,
  DEFAULT_STACK,
  MANIFEST_DEFAULT_STACK,
  SUNNYPILOT_CURRENT,
  LateralDemandPlatformCapabilities,
  LateralDemandStackCatalog,
  LateralDemandStackResolution,
  get_available_lateral_demand_stacks,
  is_lateral_demand_custom_stack,
  load_lateral_demand_stack_manifest,
  normalize_stack_value,
  resolve_lateral_demand_stack,
)


class TestLateralDemandStackCatalog:

  def setup_method(self):
    self.manifest = load_lateral_demand_stack_manifest()
    self.catalog = LateralDemandStackCatalog(self.manifest)
    self.capabilities = LateralDemandPlatformCapabilities()

  def test_default_stack_is_custom_2_0(self):
    assert self.catalog.default_stack == CUSTOM_V2

  def test_stack_names_match_manifest(self):
    assert set(self.catalog.stack_names) == {
      SUNNYPILOT_CURRENT, CUSTOM_RECOMMENDED, CUSTOM_V2, CUSTOM_EXPERIMENTAL,
    }

  def test_stack_definition_metadata(self):
    custom_v2 = self.catalog.stack_definition(CUSTOM_V2)
    assert custom_v2.label == "Custom v2.0"
    assert custom_v2.family == "custom"
    assert custom_v2.version == "2.0"
    assert custom_v2.implemented is True

  def test_experimental_definition(self):
    experimental = self.catalog.stack_definition(CUSTOM_EXPERIMENTAL)
    assert experimental.family == "custom"
    assert experimental.version == "experimental"
    assert experimental.implemented is True

  def test_available_stacks_when_all_always(self):
    available = self.catalog.available_stacks(self.capabilities)
    assert set(available) == {
      SUNNYPILOT_CURRENT, CUSTOM_RECOMMENDED, CUSTOM_V2, CUSTOM_EXPERIMENTAL,
    }

  def test_is_known(self):
    assert self.catalog.is_known(CUSTOM_V2) is True
    assert self.catalog.is_known("nonexistent") is False
    assert self.catalog.is_known("") is False


class TestLateralDemandStackResolution:

  def setup_method(self):
    self.manifest = load_lateral_demand_stack_manifest()
    self.catalog = LateralDemandStackCatalog(self.manifest)
    self.capabilities = LateralDemandPlatformCapabilities()

  def test_resolve_known_stack_returns_requested(self):
    resolution = self.catalog.resolve(CUSTOM_V2, self.capabilities)
    assert resolution.requested_stack == CUSTOM_V2
    assert resolution.resolved_stack == CUSTOM_V2
    assert resolution.custom_version == "2.0"
    assert resolution.fallback_reason == ""

  def test_resolve_unknown_stack_falls_back_to_default(self):
    resolution = self.catalog.resolve("nonexistent-stack", self.capabilities)
    assert resolution.requested_stack == "nonexistent-stack"
    assert resolution.resolved_stack == CUSTOM_V2
    assert resolution.fallback_reason == "unknown_stack"

  def test_resolve_none_uses_default(self):
    resolution = self.catalog.resolve(None, self.capabilities)
    assert resolution.requested_stack == CUSTOM_V2
    assert resolution.resolved_stack == CUSTOM_V2
    assert resolution.fallback_reason == ""

  def test_resolve_sunnypilot_current(self):
    resolution = self.catalog.resolve(SUNNYPILOT_CURRENT, self.capabilities)
    assert resolution.resolved_stack == SUNNYPILOT_CURRENT
    assert resolution.fallback_reason == ""

  def test_resolve_experimental_when_available(self):
    resolution = self.catalog.resolve(CUSTOM_EXPERIMENTAL, self.capabilities)
    assert resolution.resolved_stack == CUSTOM_EXPERIMENTAL
    assert resolution.fallback_reason == ""

  def test_resolve_custom_recommended_default_fallback(self):
    no_recommendation_manifest = {
      **self.manifest,
      "customRecommendations": {
        "default": "",
        "brands": {},
        "fingerprints": {},
      },
    }
    catalog = LateralDemandStackCatalog(no_recommendation_manifest)
    resolution = catalog.resolve(CUSTOM_RECOMMENDED, self.capabilities)
    assert resolution.requested_stack == CUSTOM_RECOMMENDED
    assert resolution.resolved_stack == CUSTOM_V2
    assert resolution.recommended_stack == ""
    assert resolution.fallback_reason == "custom_recommended_unresolved"

  def test_resolve_experimental_unavailable_falls_back_to_custom_v2(self):
    unavailable_manifest = {
      **self.manifest,
      "availability": {
        SUNNYPILOT_CURRENT: {"always": True},
        CUSTOM_RECOMMENDED: {"always": True},
        CUSTOM_V2: {"always": True},
        CUSTOM_EXPERIMENTAL: {"always": False},
      },
    }
    catalog = LateralDemandStackCatalog(unavailable_manifest)
    resolution = catalog.resolve(CUSTOM_EXPERIMENTAL, self.capabilities)
    assert resolution.requested_stack == CUSTOM_EXPERIMENTAL
    assert resolution.resolved_stack == CUSTOM_V2
    assert resolution.fallback_reason == "experimental_unavailable"

  def test_resolve_unavailable_stack_falls_back_to_default(self):
    unavailable_manifest = {
      **self.manifest,
      "availability": {
        SUNNYPILOT_CURRENT: {"always": True},
        CUSTOM_RECOMMENDED: {"always": True},
        CUSTOM_V2: {"always": False},
        CUSTOM_EXPERIMENTAL: {"always": True},
      },
    }
    catalog = LateralDemandStackCatalog(unavailable_manifest)
    resolution = catalog.resolve(CUSTOM_V2, self.capabilities)
    assert resolution.requested_stack == CUSTOM_V2
    assert resolution.resolved_stack == CUSTOM_V2
    assert resolution.fallback_reason == "unavailable_stack"

  def test_resolve_unimplemented_stack_reports_unimplemented(self):
    unimplemented_manifest = {
      **self.manifest,
      "stacks": {
        **self.manifest["stacks"],
        CUSTOM_EXPERIMENTAL: {
          "label": "Experimental", "family": "custom",
          "version": "experimental", "implemented": False,
        },
      },
    }
    catalog = LateralDemandStackCatalog(unimplemented_manifest)
    resolution = catalog.resolve(CUSTOM_EXPERIMENTAL, self.capabilities)
    assert resolution.fallback_reason in {"unimplemented_stack", "experimental_unavailable"}


class TestLateralDemandStackNormalization:

  def test_normalize_none_returns_default(self):
    assert normalize_stack_value(None) == MANIFEST_DEFAULT_STACK

  def test_normalize_bytes_returns_decoded(self):
    assert normalize_stack_value(b"custom-2.0") == "custom-2.0"

  def test_normalize_strips_whitespace(self):
    assert normalize_stack_value("  custom-2.0  ") == "custom-2.0"

  def test_normalize_empty_returns_default(self):
    assert normalize_stack_value("") == MANIFEST_DEFAULT_STACK


class TestLateralDemandStackHelpers:

  def test_is_custom_stack(self):
    assert is_lateral_demand_custom_stack(CUSTOM_V2) is True
    assert is_lateral_demand_custom_stack(CUSTOM_EXPERIMENTAL) is True
    assert is_lateral_demand_custom_stack(CUSTOM_RECOMMENDED) is True
    assert is_lateral_demand_custom_stack(SUNNYPILOT_CURRENT) is False
    assert is_lateral_demand_custom_stack("") is False

  def test_get_available_lateral_demand_stacks(self):
    manifest = load_lateral_demand_stack_manifest()
    capabilities = LateralDemandPlatformCapabilities()
    available = get_available_lateral_demand_stacks(manifest, capabilities)
    assert CUSTOM_V2 in available
    assert SUNNYPILOT_CURRENT in available

  def test_resolve_lateral_demand_stack_with_no_car_params(self):
    resolution = resolve_lateral_demand_stack(CUSTOM_V2, CP=None, CP_SP=None)
    assert resolution.resolved_stack == CUSTOM_V2

  def test_resolve_lateral_demand_stack_with_override_manifest(self):
    custom_manifest = {
      "defaultStack": "sunnypilot-current",
      "customRecommendedFallback": "sunnypilot-current",
      "stacks": {
        "sunnypilot-current": {
          "label": "Sunny Current", "family": "baseline", "version": "", "implemented": True,
        },
        "custom-recommended": {
          "label": "Custom Rec", "family": "custom-alias", "version": "", "implemented": True,
        },
        "custom-2.0": {
          "label": "Custom v2", "family": "custom", "version": "2.0", "implemented": True,
        },
        "custom-experimental": {
          "label": "Experimental", "family": "custom", "version": "experimental", "implemented": True,
        },
      },
      "availability": {
        "sunnypilot-current": {"always": True},
        "custom-recommended": {"always": True},
        "custom-2.0": {"always": True},
        "custom-experimental": {"always": True},
      },
      "customRecommendations": {
        "default": "custom-2.0",
        "brands": {},
        "fingerprints": {},
      },
    }
    resolution = resolve_lateral_demand_stack(
      "sunnypilot-current", CP=None, CP_SP=None, manifest=custom_manifest,
    )
    assert resolution.resolved_stack == "sunnypilot-current"

  def test_default_constants_align(self):
    assert MANIFEST_DEFAULT_STACK == CUSTOM_V2
    assert DEFAULT_STACK == SUNNYPILOT_CURRENT
