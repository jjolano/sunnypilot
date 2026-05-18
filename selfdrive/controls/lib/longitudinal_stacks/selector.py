from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


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


@dataclass(frozen=True)
class StackResolution:
  requested_stack: str
  resolved_stack: str
  available_stacks: tuple[str, ...]
  recommended_stack: str = ""
  custom_version: str = ""
  fallback_reason: str = ""


@dataclass(frozen=True)
class StackDefinition:
  name: str
  label: str
  family: str
  version: str = ""
  implemented: bool = True


class StackCatalog:
  def __init__(self, manifest: dict[str, Any]):
    self.manifest = manifest

  @property
  def default_stack(self) -> str:
    return str(self.manifest.get("defaultStack") or DEFAULT_STACK)

  @property
  def custom_recommended_fallback(self) -> str:
    return str(self.manifest.get("customRecommendedFallback") or self.default_stack)

  @property
  def stack_names(self) -> tuple[str, ...]:
    return tuple(str(stack) for stack in self.manifest.get("stacks", {}))

  def stack_definition(self, stack: str) -> StackDefinition:
    info = self.manifest.get("stacks", {}).get(stack, {})
    return StackDefinition(
      name=str(stack or ""),
      label=str(info.get("label") or stack or ""),
      family=str(info.get("family") or ""),
      version=str(info.get("version") or ""),
      implemented=bool(info.get("implemented", True)),
    )

  def is_known(self, stack: str) -> bool:
    return stack in self.manifest.get("stacks", {})

  def available_stacks(self, capabilities: PlatformCapabilities) -> tuple[str, ...]:
    available: list[str] = []
    for stack in self.stack_names:
      definition = self.stack_definition(stack)
      if not definition.implemented:
        continue
      rule = self.manifest.get("availability", {}).get(stack, {})
      if _availability_rule_matches(rule, capabilities):
        available.append(stack)
    return tuple(available)

  def custom_recommended_stack(self, capabilities: PlatformCapabilities) -> str:
    recommendations = self.manifest.get("customRecommendations", {})
    fingerprints = recommendations.get("fingerprints", {})
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

  def unavailable_reason(self, stack: str) -> str:
    if not self.stack_definition(stack).implemented:
      return "unimplemented_stack"
    return "unavailable_stack"

  def resolve(self, requested_stack: object, capabilities: PlatformCapabilities) -> StackResolution:
    requested = normalize_stack_value(requested_stack, self.default_stack)
    available_stacks = self.available_stacks(capabilities)

    if requested == CUSTOM_RECOMMENDED:
      recommended = self.custom_recommended_stack(capabilities)
      if recommended and recommended in available_stacks:
        return StackResolution(
          requested_stack=requested,
          resolved_stack=recommended,
          available_stacks=available_stacks,
          recommended_stack=recommended,
          custom_version=self.stack_definition(recommended).version,
        )
      fallback = self.custom_recommended_fallback
      return StackResolution(
        requested_stack=requested,
        resolved_stack=fallback,
        available_stacks=available_stacks,
        recommended_stack="",
        custom_version=self.stack_definition(fallback).version,
        fallback_reason="custom_recommended_unresolved",
      )

    if not self.is_known(requested):
      return StackResolution(
        requested_stack=requested,
        resolved_stack=self.default_stack,
        available_stacks=available_stacks,
        fallback_reason="unknown_stack",
      )

    if requested not in available_stacks:
      return StackResolution(
        requested_stack=requested,
        resolved_stack=self.default_stack,
        available_stacks=available_stacks,
        custom_version=self.stack_definition(self.default_stack).version,
        fallback_reason=self.unavailable_reason(requested),
      )

    return StackResolution(
      requested_stack=requested,
      resolved_stack=requested,
      available_stacks=available_stacks,
      custom_version=self.stack_definition(requested).version,
    )


def load_stack_manifest(path: str | Path = MANIFEST_PATH) -> dict[str, Any]:
  with open(path, encoding="utf-8") as f:
    return json.load(f)


def normalize_stack_value(value: object, default_stack: str = DEFAULT_STACK) -> str:
  if value is None:
    return default_stack
  if isinstance(value, bytes):
    value = value.decode(errors="ignore")
  value = str(value).strip()
  return value or default_stack


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
    return int(value)
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

  required = tuple(rule.get("requires", ()))
  if required and not all(_capability_value(capabilities, key) for key in required):
    return False

  requires_any = tuple(rule.get("requiresAny", ()))
  if requires_any and not any(_capability_value(capabilities, key) for key in requires_any):
    return False

  blocked = tuple(rule.get("blockedBy", ()))
  if blocked and any(_capability_value(capabilities, key) for key in blocked):
    return False

  return True
