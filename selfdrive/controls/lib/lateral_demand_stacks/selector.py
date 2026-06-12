from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, cast

from openpilot.selfdrive.controls.lib.stack_catalog import (
  StackCatalog as _StackCatalog,
  StackDefinition as _StackDefinition,
  StackResolution as _StackResolution,
  normalize_stack_value as _normalize_stack_value,
)


SUNNYPILOT_CURRENT = "sunnypilot-current"
CUSTOM_RECOMMENDED = "custom-recommended"
CUSTOM_V2 = "custom-2.0"
CUSTOM_EXPERIMENTAL = "custom-experimental"

DEFAULT_STACK = SUNNYPILOT_CURRENT
MANIFEST_DEFAULT_STACK = CUSTOM_V2

MANIFEST_PATH = Path(__file__).with_name("lateral_demand_stack_versions.json")


@dataclass(frozen=True)
class LateralDemandPlatformCapabilities:
  brand: str = ""
  car_fingerprint: str = ""
  lateral_controller_stack: str = ""
  torque_version: int = 0
  model_v2_available: bool = True
  lane_lines_available: bool = True
  lane_centering_assist_enabled: bool = False
  cp_sp_flags: int = 0


LateralDemandStackResolution = _StackResolution
LateralDemandStackDefinition = _StackDefinition


class LateralDemandStackCatalog:
  def __init__(self, manifest: dict[str, Any]):
    self.manifest = manifest
    self._capabilities = LateralDemandPlatformCapabilities()
    self._catalog = _StackCatalog(
      manifest,
      lambda rule: _availability_rule_matches(rule, self._capabilities),
      lambda: self._custom_recommended_stack(),
      default_stack=MANIFEST_DEFAULT_STACK,
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

  def stack_definition(self, stack: str) -> LateralDemandStackDefinition:
    return self._catalog.stack_definition(stack)

  def is_known(self, stack: str) -> bool:
    return self._catalog.is_known(stack)

  def available_stacks(self, capabilities: LateralDemandPlatformCapabilities) -> tuple[str, ...]:
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

  def custom_recommended_stack(self, capabilities: LateralDemandPlatformCapabilities) -> str:
    self._capabilities = capabilities
    return self._catalog.custom_recommended_stack()

  def unavailable_reason(self, stack: str) -> str:
    return self._catalog.unavailable_reason(stack)

  def resolve(self, requested_stack: object, capabilities: LateralDemandPlatformCapabilities) -> LateralDemandStackResolution:
    requested = normalize_stack_value(requested_stack, self.default_stack)
    self._capabilities = capabilities
    available_stacks = self._catalog.available_stacks()

    if requested == CUSTOM_RECOMMENDED:
      recommended = self.custom_recommended_stack(capabilities)
      if recommended and recommended in available_stacks:
        return LateralDemandStackResolution(
          requested_stack=requested,
          resolved_stack=recommended,
          available_stacks=available_stacks,
          recommended_stack=recommended,
          custom_version=self.stack_definition(recommended).version,
        )
      fallback = self.custom_recommended_fallback
      return LateralDemandStackResolution(
        requested_stack=requested,
        resolved_stack=fallback,
        available_stacks=available_stacks,
        recommended_stack="",
        custom_version=self.stack_definition(fallback).version,
        fallback_reason="custom_recommended_unresolved",
      )

    if requested == CUSTOM_EXPERIMENTAL and CUSTOM_EXPERIMENTAL not in available_stacks:
      fallback = CUSTOM_V2 if CUSTOM_V2 in available_stacks else self.default_stack
      return LateralDemandStackResolution(
        requested_stack=requested,
        resolved_stack=fallback,
        available_stacks=available_stacks,
        custom_version=self.stack_definition(fallback).version,
        fallback_reason="experimental_unavailable",
      )

    return self._catalog.resolve(requested_stack)


def load_lateral_demand_stack_manifest(path: str | Path = MANIFEST_PATH) -> dict[str, Any]:
  with open(path, encoding="utf-8") as f:
    return json.load(f)


def normalize_stack_value(value: object, default_stack: str = MANIFEST_DEFAULT_STACK) -> str:
  return _normalize_stack_value(value, default_stack)


def lateral_demand_platform_capabilities_from_car_params(CP: object | None,
                                                         CP_SP: object | None = None) -> LateralDemandPlatformCapabilities:
  return LateralDemandPlatformCapabilities(
    brand=str(_get_attr(CP, "brand", "") or ""),
    car_fingerprint=str(_get_attr(CP, "carFingerprint", "") or ""),
    model_v2_available=True,
    lane_lines_available=True,
    lane_centering_assist_enabled=bool(_get_attr(CP_SP, "laneCenteringAssistEnabled", False)),
    cp_sp_flags=_safe_int(_get_attr(CP_SP, "flags", 0)),
  )


def get_available_lateral_demand_stacks(manifest: dict[str, Any], capabilities: LateralDemandPlatformCapabilities) -> tuple[str, ...]:
  return LateralDemandStackCatalog(manifest).available_stacks(capabilities)


def is_lateral_demand_custom_stack(stack: str) -> bool:
  return str(stack or "").startswith("custom-")


def resolve_lateral_demand_stack(requested_stack: object, CP: object | None = None,
                                 CP_SP: object | None = None, manifest: dict | None = None) -> LateralDemandStackResolution:
  manifest = manifest if manifest is not None else load_lateral_demand_stack_manifest()
  catalog = LateralDemandStackCatalog(manifest)
  capabilities = lateral_demand_platform_capabilities_from_car_params(CP, CP_SP)
  return catalog.resolve(requested_stack, capabilities)


def _get_attr(obj: object | None, name: str, default: object) -> object:
  return getattr(obj, name, default) if obj is not None else default


def _safe_int(value: object) -> int:
  try:
    return int(value)  # type: ignore[arg-type]
  except (TypeError, ValueError):
    return 0


def _capability_value(capabilities: LateralDemandPlatformCapabilities, key: str) -> bool:
  mapping = {
    "modelV2Available": capabilities.model_v2_available,
    "laneLinesAvailable": capabilities.lane_lines_available,
    "laneCenteringAssistEnabled": capabilities.lane_centering_assist_enabled,
  }
  return bool(mapping.get(key, False))


def _availability_rule_matches(rule: dict[str, Any], capabilities: LateralDemandPlatformCapabilities) -> bool:
  if not rule:
    return False
  if "always" in rule:
    return bool(rule.get("always"))

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
