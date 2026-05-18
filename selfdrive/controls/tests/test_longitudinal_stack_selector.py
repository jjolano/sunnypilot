from types import SimpleNamespace

from openpilot.selfdrive.controls.lib.longitudinal_stacks.selector import (
  CUSTOM_RECOMMENDED,
  CUSTOM_V2,
  DEFAULT_STACK,
  SUNNYPILOT_CURRENT,
  load_stack_manifest,
  normalize_stack_value,
  resolve_longitudinal_stack,
)


def make_cp(**kwargs):
  values = {
    "brand": "hyundai",
    "carFingerprint": "HYUNDAI_TEST",
    "openpilotLongitudinalControl": False,
    "alphaLongitudinalAvailable": False,
    "pcmCruise": False,
    "radarUnavailable": False,
  }
  values.update(kwargs)
  return SimpleNamespace(**values)


def test_normalize_stack_value_defaults_to_sunnypilot_current():
  assert normalize_stack_value(None) == DEFAULT_STACK
  assert normalize_stack_value("") == DEFAULT_STACK
  assert normalize_stack_value(b"custom-2.0") == CUSTOM_V2


def test_unset_stack_resolves_to_sunnypilot_current():
  resolution = resolve_longitudinal_stack(None, make_cp())

  assert resolution.requested_stack == SUNNYPILOT_CURRENT
  assert resolution.resolved_stack == SUNNYPILOT_CURRENT
  assert resolution.fallback_reason == ""


def test_openpilot_current_is_not_a_supported_stack():
  resolution = resolve_longitudinal_stack("openpilot-current", make_cp(openpilotLongitudinalControl=True))

  assert resolution.resolved_stack == SUNNYPILOT_CURRENT
  assert resolution.fallback_reason == "unknown_stack"
  assert "openpilot-current" not in resolution.available_stacks


def test_custom_recommended_resolves_global_manifest_default():
  resolution = resolve_longitudinal_stack(CUSTOM_RECOMMENDED, make_cp(alphaLongitudinalAvailable=True))

  assert resolution.resolved_stack == CUSTOM_V2
  assert resolution.recommended_stack == CUSTOM_V2
  assert resolution.custom_version == "2.0"
  assert resolution.fallback_reason == ""


def test_custom_recommended_falls_back_when_custom_unavailable():
  resolution = resolve_longitudinal_stack(CUSTOM_RECOMMENDED, make_cp())

  assert resolution.resolved_stack == SUNNYPILOT_CURRENT
  assert resolution.recommended_stack == ""
  assert resolution.fallback_reason == "custom_recommended_unresolved"


def test_custom_recommended_resolves_per_platform_manifest():
  manifest = load_stack_manifest()
  manifest["customRecommendations"] = {
    "default": "",
    "brands": {"hyundai": CUSTOM_V2},
    "fingerprints": {},
  }

  resolution = resolve_longitudinal_stack(CUSTOM_RECOMMENDED, make_cp(alphaLongitudinalAvailable=True), manifest=manifest)

  assert resolution.resolved_stack == CUSTOM_V2
  assert resolution.recommended_stack == CUSTOM_V2
  assert resolution.custom_version == "2.0"
  assert resolution.fallback_reason == ""


def test_literal_available_custom_v2_can_be_forced_without_promoting_recommended():
  resolution = resolve_longitudinal_stack(CUSTOM_V2, make_cp(alphaLongitudinalAvailable=True))
  recommended = resolve_longitudinal_stack(CUSTOM_RECOMMENDED, make_cp(alphaLongitudinalAvailable=True))

  assert resolution.resolved_stack == CUSTOM_V2
  assert resolution.custom_version == "2.0"
  assert resolution.fallback_reason == ""
  assert recommended.resolved_stack == CUSTOM_V2
  assert recommended.recommended_stack == CUSTOM_V2
  assert recommended.fallback_reason == ""


def test_literal_removed_custom_1_0_falls_back_to_default():
  resolution = resolve_longitudinal_stack("custom-1.0", make_cp(alphaLongitudinalAvailable=True))

  assert resolution.resolved_stack == SUNNYPILOT_CURRENT
  assert resolution.fallback_reason == "unknown_stack"
  assert "custom-1.0" not in resolution.available_stacks


def test_literal_unavailable_custom_v2_falls_back_to_default():
  resolution = resolve_longitudinal_stack(CUSTOM_V2, make_cp())

  assert resolution.resolved_stack == SUNNYPILOT_CURRENT
  assert resolution.fallback_reason == "unavailable_stack"
  assert CUSTOM_V2 not in resolution.available_stacks


def test_unknown_stack_falls_back_to_default():
  resolution = resolve_longitudinal_stack("custom-9.9", make_cp(alphaLongitudinalAvailable=True))

  assert resolution.resolved_stack == SUNNYPILOT_CURRENT
  assert resolution.fallback_reason == "unknown_stack"
