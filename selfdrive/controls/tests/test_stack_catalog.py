from openpilot.selfdrive.controls.lib.stack_catalog import (
  StackCatalog,
  StackDefinition,
  StackResolution,
  normalize_stack_value,
)


def test_normalize_stack_value_handles_blank_and_bytes():
  assert normalize_stack_value(None, "default") == "default"
  assert normalize_stack_value("", "default") == "default"
  assert normalize_stack_value(b"custom", "default") == "custom"


def test_stack_catalog_resolves_using_callbacks_and_manifest_fields():
  manifest = {
    "defaultStack": "base",
    "customRecommendedFallback": "base",
    "stacks": {
      "base": {"label": "Base", "family": "baseline"},
      "custom-recommended": {"label": "Recommended", "family": "custom"},
      "custom": {"label": "Custom", "family": "custom", "version": "1.2"},
    },
    "availability": {
      "base": {"always": True},
      "custom-recommended": {"always": True},
      "custom": {"always": True},
    },
  }
  catalog = StackCatalog(manifest, lambda rule: bool(rule.get("always")), lambda: "custom")

  assert catalog.default_stack == "base"
  assert catalog.custom_recommended_fallback == "base"
  assert catalog.stack_names == ("base", "custom-recommended", "custom")
  assert catalog.stack_definition("custom") == StackDefinition("custom", "Custom", "custom", "1.2", True)
  assert catalog.is_known("custom")
  assert catalog.available_stacks() == ("base", "custom-recommended", "custom")

  resolved = catalog.resolve("custom-recommended")
  assert resolved == StackResolution(
    requested_stack="custom-recommended",
    resolved_stack="custom",
    available_stacks=("base", "custom-recommended", "custom"),
    recommended_stack="custom",
    custom_version="1.2",
    fallback_reason="",
  )


def test_stack_catalog_uses_fallback_and_reasons_for_unknown_and_unavailable():
  manifest = {
    "defaultStack": "base",
    "customRecommendedFallback": "base",
    "stacks": {
      "base": {"label": "Base", "family": "baseline"},
      "custom-recommended": {"label": "Recommended", "family": "custom"},
      "custom": {"label": "Custom", "family": "custom", "version": "1.2"},
    },
    "availability": {
      "base": {"always": True},
    },
  }
  catalog = StackCatalog(manifest, lambda rule: bool(rule.get("always")), lambda: "custom")

  assert catalog.custom_recommended_stack() == "custom"
  assert catalog.unavailable_reason("custom") == "unavailable_stack"
  assert catalog.unavailable_reason("unknown") == "unimplemented_stack"

  assert catalog.resolve("missing").fallback_reason == "unknown_stack"
  unresolved = catalog.resolve("custom-recommended")
  assert unresolved.resolved_stack == "base"
  assert unresolved.fallback_reason == "custom_recommended_unresolved"


def test_blank_recommendation_remains_unresolved_instead_of_defaulting():
  manifest = {
    "defaultStack": "base",
    "customRecommendedFallback": "base",
    "stacks": {
      "base": {"label": "Base", "family": "baseline"},
      "custom-recommended": {"label": "Recommended", "family": "custom"},
    },
    "availability": {
      "base": {"always": True},
      "custom-recommended": {"always": True},
    },
  }
  catalog = StackCatalog(manifest, lambda rule: bool(rule.get("always")), lambda: "")

  unresolved = catalog.resolve("custom-recommended")

  assert catalog.custom_recommended_stack() == ""
  assert unresolved.resolved_stack == "base"
  assert unresolved.recommended_stack == ""
  assert unresolved.fallback_reason == "custom_recommended_unresolved"


def test_catalog_can_use_domain_default_when_manifest_omits_default():
  catalog = StackCatalog({"stacks": {}, "availability": {}}, lambda _: False, default_stack="domain-default")

  assert catalog.default_stack == "domain-default"
