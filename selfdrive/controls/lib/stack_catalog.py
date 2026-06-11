from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, cast


@dataclass(frozen=True)
class StackDefinition:
  name: str
  label: str
  family: str
  version: str = ""
  implemented: bool = True


@dataclass(frozen=True)
class StackResolution:
  requested_stack: str
  resolved_stack: str
  available_stacks: tuple[str, ...]
  recommended_stack: str = ""
  custom_version: str = ""
  fallback_reason: str = ""


def load_stack_manifest(path: str | Path) -> dict[str, Any]:
  with open(path, encoding="utf-8") as f:
    return json.load(f)


def normalize_stack_value(value: object, default_stack: str = "") -> str:
  if value is None:
    return default_stack
  if isinstance(value, bytes):
    value = value.decode(errors="ignore")
  value = str(value).strip()
  return value or default_stack


class StackCatalog:
  def __init__(
    self,
    manifest: dict[str, Any],
    availability_rule_matches: Callable[[dict[str, Any]], bool] | None = None,
    custom_recommendation_resolver: Callable[[], str] | None = None,
    default_stack: str = "",
  ):
    self.manifest = manifest
    self._availability_rule_matches = availability_rule_matches
    self._custom_recommendation_resolver = custom_recommendation_resolver
    self._default_stack = default_stack

  @property
  def default_stack(self) -> str:
    return str(self.manifest.get("defaultStack") or self._default_stack)

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

  def available_stacks(self, capabilities: object | None = None) -> tuple[str, ...]:
    available: list[str] = []
    for stack in self.stack_names:
      definition = self.stack_definition(stack)
      if not definition.implemented:
        continue
      rule = self.manifest.get("availability", {}).get(stack, {})
      if self._rule_matches(rule, capabilities):
        available.append(stack)
    return tuple(available)

  def custom_recommended_stack(self, capabilities: object | None = None) -> str:
    if self._custom_recommendation_resolver is not None:
      return str(self._custom_recommendation_resolver() or "")

    recommendations = self.manifest.get("customRecommendations", {})
    fingerprints = recommendations.get("fingerprints", {})
    brand = str(_get_attr(capabilities, "brand", "") or "")
    car_fingerprint = str(_get_attr(capabilities, "car_fingerprint", "") or "")
    fingerprint_keys = (
      f"{brand}:{car_fingerprint}",
      car_fingerprint,
    )
    for key in fingerprint_keys:
      if key and key in fingerprints:
        return str(fingerprints[key] or "")

    brands = recommendations.get("brands", {})
    if brand and brand in brands:
      return str(brands[brand] or "")
    return str(recommendations.get("default", "") or "")

  def unavailable_reason(self, stack: str) -> str:
    if not self.is_known(stack) or not self.stack_definition(stack).implemented:
      return "unimplemented_stack"
    return "unavailable_stack"

  def resolve(self, requested_stack: object, capabilities: object | None = None) -> StackResolution:
    requested = normalize_stack_value(requested_stack, self.default_stack)
    available_stacks = self.available_stacks(capabilities)

    if requested == self.manifest.get("customRecommendedName", "custom-recommended"):
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

  def _rule_matches(self, rule: dict[str, Any], capabilities: object | None) -> bool:
    if self._availability_rule_matches is not None:
      return self._availability_rule_matches(rule)
    return _availability_rule_matches(rule, capabilities, self._capability_value)

  def _capability_value(self, capabilities: object | None, key: str) -> bool:
    return False


def _get_attr(obj: object | None, name: str, default: object) -> object:
  return getattr(obj, name, default) if obj is not None else default


def _availability_rule_matches(
    rule: dict[str, Any],
    capabilities: object | None,
    capability_value_fn: Callable[[object | None, str], bool]) -> bool:
  if not rule:
    return False
  if "always" in rule:
    return bool(rule.get("always"))

  required = tuple(cast(Any, rule.get("requires") or ()))
  if required and not all(capability_value_fn(capabilities, key) for key in required):
    return False

  requires_any = tuple(cast(Any, rule.get("requiresAny") or ()))
  if requires_any and not any(capability_value_fn(capabilities, key) for key in requires_any):
    return False

  blocked = tuple(cast(Any, rule.get("blockedBy") or ()))
  if blocked and any(capability_value_fn(capabilities, key) for key in blocked):
    return False

  return True
