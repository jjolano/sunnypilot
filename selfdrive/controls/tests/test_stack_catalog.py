from openpilot.selfdrive.controls.lib.stack_catalog import (
  StackCatalog,
  normalize_stack_value,
)


def test_load_manifest_and_normalize_default_values():
  catalog = StackCatalog({"defaultStack": "sunnypilot-current", "stacks": {}})

  assert catalog.default_stack == "sunnypilot-current"
  assert normalize_stack_value(None, catalog.default_stack) == "sunnypilot-current"
  assert normalize_stack_value(b"custom-2.0", catalog.default_stack) == "custom-2.0"


def test_stack_catalog_shared_resolution_mechanics():
  manifest = {
    "defaultStack": "sunnypilot-current",
    "customRecommendedFallback": "sunnypilot-current",
    "stacks": {
      "sunnypilot-current": {"label": "Sunny", "family": "baseline", "implemented": True},
      "custom-recommended": {"label": "Rec", "family": "custom", "implemented": True},
      "custom-2.0": {"label": "V2", "family": "custom", "version": "2.0", "implemented": True},
    },
    "availability": {
      "sunnypilot-current": {"always": True},
      "custom-recommended": {"always": True},
      "custom-2.0": {"always": True},
    },
    "customRecommendations": {
      "default": "custom-2.0",
      "brands": {},
      "fingerprints": {},
    },
  }
  catalog = StackCatalog(manifest)
  capabilities = type("Cap", (), {"brand": "hyundai", "car_fingerprint": "TEST"})()

  assert catalog.stack_names == ("sunnypilot-current", "custom-recommended", "custom-2.0")
  assert catalog.available_stacks(capabilities) == ("sunnypilot-current", "custom-recommended", "custom-2.0")
  resolution = catalog.resolve("custom-recommended", capabilities)
  assert resolution.resolved_stack == "custom-2.0"
  assert resolution.recommended_stack == "custom-2.0"
  assert resolution.custom_version == "2.0"
