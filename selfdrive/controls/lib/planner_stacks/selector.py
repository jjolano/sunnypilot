from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, cast

from openpilot.selfdrive.controls.lib.stack_catalog import (
  StackCatalog as _StackCatalog,
  StackDefinition,
  StackResolution,
  normalize_stack_value as _normalize_stack_value,
)


PLANNER_CURRENT = "planner-current"
SCENE_MEMORY_V1 = "scene-memory-v1"
DEFAULT_STACK = PLANNER_CURRENT
PLANNER_STACK_PARAM = "PlannerStack"
PLANNER_STACK_VALIDATION_GATE_PARAM = "PlannerStackValidationGate"

MANIFEST_PATH = Path(__file__).with_name("planner_stack_versions.json")


@dataclass(frozen=True)
class PlannerCapabilities:
  brand: str = ""
  car_fingerprint: str = ""
  openpilot_longitudinal_control: bool = False
  alpha_longitudinal_available: bool = False
  pcm_cruise: bool = False
  radar_unavailable: bool = False
  cp_sp_flags: int = 0
  planner_validation_gate: bool = False


def normalize_stack_value(value: object, default_stack: str = DEFAULT_STACK) -> str:
  return _normalize_stack_value(value, default_stack)


class PlannerStackCatalog:
  def __init__(self, manifest: dict[str, Any]):
    self.manifest = manifest
    self._capabilities = PlannerCapabilities()
    self._catalog = _StackCatalog(
      manifest,
      lambda rule: _availability_rule_matches(rule, self._capabilities),
      default_stack=DEFAULT_STACK,
    )

  @property
  def default_stack(self) -> str:
    return self._catalog.default_stack

  @property
  def stack_names(self) -> tuple[str, ...]:
    return self._catalog.stack_names

  def stack_definition(self, stack: str) -> StackDefinition:
    return self._catalog.stack_definition(stack)

  def is_known(self, stack: str) -> bool:
    return self._catalog.is_known(stack)

  def available_stacks(self, capabilities: PlannerCapabilities) -> tuple[str, ...]:
    self._capabilities = capabilities
    return self._catalog.available_stacks()

  def unavailable_reason(self, stack: str) -> str:
    if stack == SCENE_MEMORY_V1 and not self._capabilities.planner_validation_gate:
      return "validation_gate_unmet"
    return self._catalog.unavailable_reason(stack)

  def resolve(self, requested_stack: object, capabilities: PlannerCapabilities) -> StackResolution:
    self._capabilities = capabilities
    requested = normalize_stack_value(requested_stack, self.default_stack)
    available_stacks = self.available_stacks(capabilities)

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


def planner_capabilities_from_car_params(
    CP: object | None,
    CP_SP: object | None = None,
    *,
    validation_gate: bool = False) -> PlannerCapabilities:
  return PlannerCapabilities(
    brand=str(_get_attr(CP, "brand", "") or ""),
    car_fingerprint=str(_get_attr(CP, "carFingerprint", "") or ""),
    openpilot_longitudinal_control=bool(_get_attr(CP, "openpilotLongitudinalControl", False)),
    alpha_longitudinal_available=bool(_get_attr(CP, "alphaLongitudinalAvailable", False)),
    pcm_cruise=bool(_get_attr(CP, "pcmCruise", False)),
    radar_unavailable=bool(_get_attr(CP, "radarUnavailable", False)),
    cp_sp_flags=_safe_int(_get_attr(CP_SP, "flags", 0)),
    planner_validation_gate=bool(validation_gate),
  )


def get_available_stacks(manifest: dict[str, Any], capabilities: PlannerCapabilities) -> tuple[str, ...]:
  return PlannerStackCatalog(manifest).available_stacks(capabilities)


def resolve_planner_stack(
    requested_stack: object,
    CP: object | None = None,
    CP_SP: object | None = None,
    *,
    validation_gate: bool = False,
    manifest: dict[str, Any] | None = None) -> StackResolution:
  manifest = manifest if manifest is not None else load_stack_manifest()
  catalog = PlannerStackCatalog(manifest)
  capabilities = planner_capabilities_from_car_params(CP, CP_SP, validation_gate=validation_gate)
  return catalog.resolve(requested_stack, capabilities)


def planner_stack_id_for_name(name: str):
  from cereal import custom

  StackId = custom.LongitudinalPlanSP.PlannerStack.PlannerStackId
  stack_id_by_name = {
    PLANNER_CURRENT: StackId.plannerCurrent,
    SCENE_MEMORY_V1: StackId.sceneMemoryV1,
  }
  return stack_id_by_name.get(str(name or ""), StackId.unknown)


def _get_attr(obj: object | None, name: str, default: object) -> object:
  return getattr(obj, name, default) if obj is not None else default


def _safe_int(value: object) -> int:
  try:
    return int(cast(Any, value))
  except (TypeError, ValueError):
    return 0


def _capability_value(capabilities: PlannerCapabilities, key: str) -> bool:
  mapping = {
    "openpilotLongitudinalControl": capabilities.openpilot_longitudinal_control,
    "alphaLongitudinalAvailable": capabilities.alpha_longitudinal_available,
    "pcmCruise": capabilities.pcm_cruise,
    "radarUnavailable": capabilities.radar_unavailable,
    "plannerValidationGate": capabilities.planner_validation_gate,
  }
  return bool(mapping.get(key, False))


def _availability_rule_matches(rule: dict[str, Any], capabilities: PlannerCapabilities) -> bool:
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
