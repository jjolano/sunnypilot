from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, cast

from openpilot.selfdrive.controls.lib.stack_catalog import StackCatalog as _StackCatalog, StackDefinition, StackResolution, normalize_stack_value as _normalize_stack_value


SUNNYPILOT_CURRENT = "sunnypilot-current"
CUSTOM_RECOMMENDED = "custom-recommended"
CUSTOM_V2 = "custom-2.0"
DEFAULT_STACK = SUNNYPILOT_CURRENT

MANIFEST_PATH = Path(__file__).with_name("longitudinal_stack_versions.json")


@dataclass(frozen=True)
class PlatformCapabilities:
  brand: str = ""
  car_fingerprint: str = ""
  openpilot_longitudinal_control: bool = False
  alpha_longitudinal_available: bool = False
  pcm_cruise: bool = False
  radar_unavailable: bool = False
  cp_sp_flags: int = 0


class StackCatalog:
  def __init__(self, manifest: dict[str, Any]):
    self.manifest = manifest
    self._capabilities = PlatformCapabilities()
    self._catalog = _StackCatalog(
      manifest,
      lambda rule: _availability_rule_matches(rule, self._capabilities),
      lambda: self._custom_recommended_stack(),
      default_stack=DEFAULT_STACK,
    )

  @property
  def default_stack(self) -> str:
    return self._catalog.default_stack

  @property
  def custom_recommended_fallback(self) -> str:
    return self._catalog.custom_recommended_fallback

  @property
  def stack_names(self) -> tuple[str, ...]:
    return self._catalog.stack_names

  def stack_definition(self, stack: str) -> StackDefinition:
    return self._catalog.stack_definition(stack)

  def is_known(self, stack: str) -> bool:
    return self._catalog.is_known(stack)

  def available_stacks(self, capabilities: PlatformCapabilities) -> tuple[str, ...]:
    self._capabilities = capabilities
    return self._catalog.available_stacks()

  def _custom_recommended_stack(self) -> str:
    recommendations = self.manifest.get("customRecommendations", {})
    fingerprints = recommendations.get("fingerprints", {})
    capabilities = self._capabilities
    fingerprint_keys = (
      f"{capabilities.brand}:{capabilities.car_fingerprint}",
      capabilities.car_fingerprint,
    )
    for key in fingerprint_keys:
      if key and key in fingerprints:
        return str(fingerprints[key] or "")

    brands = recommendations.get("brands", {})
    if capabilities.brand and capabilities.brand in brands:
      return str(brands[capabilities.brand] or "")
    return str(recommendations.get("default", "") or "")

  def custom_recommended_stack(self, capabilities: PlatformCapabilities) -> str:
    self._capabilities = capabilities
    return self._catalog.custom_recommended_stack()

  def unavailable_reason(self, stack: str) -> str:
    return self._catalog.unavailable_reason(stack)

  def resolve(self, requested_stack: object, capabilities: PlatformCapabilities) -> StackResolution:
    self._capabilities = capabilities
    return self._catalog.resolve(requested_stack)


def load_stack_manifest(path: str | Path = MANIFEST_PATH) -> dict[str, Any]:
  with open(path, encoding="utf-8") as f:
    return json.load(f)


def normalize_stack_value(value: object, default_stack: str = DEFAULT_STACK) -> str:
  return _normalize_stack_value(value, default_stack)


def platform_capabilities_from_car_params(CP: object | None, CP_SP: object | None = None) -> PlatformCapabilities:
  return PlatformCapabilities(
    brand=str(_get_attr(CP, "brand", "") or ""),
    car_fingerprint=str(_get_attr(CP, "carFingerprint", "") or ""),
    openpilot_longitudinal_control=bool(_get_attr(CP, "openpilotLongitudinalControl", False)),
    alpha_longitudinal_available=bool(_get_attr(CP, "alphaLongitudinalAvailable", False)),
    pcm_cruise=bool(_get_attr(CP, "pcmCruise", False)),
    radar_unavailable=bool(_get_attr(CP, "radarUnavailable", False)),
    cp_sp_flags=_safe_int(_get_attr(CP_SP, "flags", 0)),
  )


def get_available_stacks(manifest: dict[str, Any], capabilities: PlatformCapabilities) -> tuple[str, ...]:
  return StackCatalog(manifest).available_stacks(capabilities)


def is_custom_stack(stack: str) -> bool:
  return str(stack or "").startswith("custom-")


def resolve_longitudinal_stack(
    requested_stack: object,
    CP: object | None = None,
    CP_SP: object | None = None,
    manifest: dict[str, Any] | None = None) -> StackResolution:
  manifest = manifest if manifest is not None else load_stack_manifest()
  catalog = StackCatalog(manifest)
  capabilities = platform_capabilities_from_car_params(CP, CP_SP)
  return catalog.resolve(requested_stack, capabilities)


def stack_id_for_name(name: str):
  from cereal import custom

  StackId = custom.LongitudinalPlanSP.Stack.StackId
  stack_id_by_name = {
    SUNNYPILOT_CURRENT: StackId.sunnypilotCurrent,
    CUSTOM_RECOMMENDED: StackId.customRecommended,
    CUSTOM_V2: StackId.customV2,
  }
  return stack_id_by_name.get(str(name or ""), StackId.unknown)


def _get_attr(obj: object | None, name: str, default: object) -> object:
  return getattr(obj, name, default) if obj is not None else default


def _safe_int(value: object) -> int:
  try:
    return int(cast(Any, value))
  except (TypeError, ValueError):
    return 0


def _capability_value(capabilities: PlatformCapabilities, key: str) -> bool:
  mapping = {
    "openpilotLongitudinalControl": capabilities.openpilot_longitudinal_control,
    "alphaLongitudinalAvailable": capabilities.alpha_longitudinal_available,
    "pcmCruise": capabilities.pcm_cruise,
    "radarUnavailable": capabilities.radar_unavailable,
  }
  return bool(mapping.get(key, False))


def _availability_rule_matches(rule: dict[str, Any], capabilities: PlatformCapabilities) -> bool:
  if not rule:
    return False
  if rule.get("always"):
    return True

  required = tuple(cast(Any, rule.get("requires") or ()))
  if required and not all(_capability_value(capabilities, key) for key in required):
    return False

  requires_any = tuple(cast(Any, rule.get("requiresAny") or ()))
  if requires_any and not any(_capability_value(capabilities, key) for key in requires_any):
    return False

  blocked = tuple(cast(Any, rule.get("blockedBy") or ()))
  if blocked and any(_capability_value(capabilities, key) for key in blocked):
    return False

  return True
