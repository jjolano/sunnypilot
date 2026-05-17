from types import SimpleNamespace

from openpilot.selfdrive.controls.lib.longitudinal_stacks.selector import (
  CUSTOM_RECOMMENDED,
  CUSTOM_V1,
  DEFAULT_STACK,
  OPENPILOT_CURRENT,
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
  assert normalize_stack_value(b"custom-1.0") == CUSTOM_V1


def test_unset_stack_resolves_to_sunnypilot_current():
  resolution = resolve_longitudinal_stack(None, make_cp())

  assert resolution.requested_stack == SUNNYPILOT_CURRENT
  assert resolution.resolved_stack == SUNNYPILOT_CURRENT
  assert resolution.fallback_reason == ""


def test_openpilot_current_is_hidden_until_adapter_is_implemented():
  resolution = resolve_longitudinal_stack(OPENPILOT_CURRENT, make_cp(openpilotLongitudinalControl=True))

  assert resolution.resolved_stack == SUNNYPILOT_CURRENT
  assert resolution.fallback_reason == "unimplemented_stack"
  assert OPENPILOT_CURRENT not in resolution.available_stacks


def test_openpilot_current_requires_openpilot_longitudinal_control_when_implemented():
  manifest = load_stack_manifest()
  manifest["stacks"][OPENPILOT_CURRENT]["implemented"] = True

  unavailable = resolve_longitudinal_stack(OPENPILOT_CURRENT, make_cp(alphaLongitudinalAvailable=True), manifest=manifest)
  available = resolve_longitudinal_stack(OPENPILOT_CURRENT, make_cp(openpilotLongitudinalControl=True), manifest=manifest)

  assert unavailable.resolved_stack == SUNNYPILOT_CURRENT
  assert unavailable.fallback_reason == "unavailable_stack"
  assert OPENPILOT_CURRENT not in unavailable.available_stacks
  assert available.resolved_stack == OPENPILOT_CURRENT
  assert OPENPILOT_CURRENT in available.available_stacks


def test_custom_recommended_falls_back_to_sunnypilot_when_unresolved():
  resolution = resolve_longitudinal_stack(CUSTOM_RECOMMENDED, make_cp(alphaLongitudinalAvailable=True))

  assert resolution.resolved_stack == SUNNYPILOT_CURRENT
  assert resolution.recommended_stack == ""
  assert resolution.fallback_reason == "custom_recommended_unresolved"


def test_custom_recommended_resolves_per_platform_manifest():
  manifest = load_stack_manifest()
  manifest["customRecommendations"] = {
    "default": "",
    "brands": {"hyundai": CUSTOM_V1},
    "fingerprints": {},
  }

  resolution = resolve_longitudinal_stack(CUSTOM_RECOMMENDED, make_cp(alphaLongitudinalAvailable=True), manifest=manifest)

  assert resolution.resolved_stack == CUSTOM_V1
  assert resolution.recommended_stack == CUSTOM_V1
  assert resolution.custom_version == "1.0"
  assert resolution.fallback_reason == ""


def test_literal_available_custom_version_can_be_forced():
  resolution = resolve_longitudinal_stack(CUSTOM_V1, make_cp(alphaLongitudinalAvailable=True))

  assert resolution.resolved_stack == CUSTOM_V1
  assert resolution.custom_version == "1.0"
  assert resolution.fallback_reason == ""


def test_literal_unavailable_custom_version_falls_back_to_default():
  resolution = resolve_longitudinal_stack(CUSTOM_V1, make_cp())

  assert resolution.resolved_stack == SUNNYPILOT_CURRENT
  assert resolution.fallback_reason == "unavailable_stack"
  assert CUSTOM_V1 not in resolution.available_stacks


def test_unknown_stack_falls_back_to_default():
  resolution = resolve_longitudinal_stack("custom-9.9", make_cp(alphaLongitudinalAvailable=True))

  assert resolution.resolved_stack == SUNNYPILOT_CURRENT
  assert resolution.fallback_reason == "unknown_stack"
