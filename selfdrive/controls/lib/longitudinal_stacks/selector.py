from __future__ import annotations

from pathlib import Path

from openpilot.selfdrive.controls.lib.stack_catalog import (
  StackCatalog as _StackCatalog,
  StackDefinition as _StackDefinition,
  StackResolution as _StackResolution,
  load_stack_manifest as _load_stack_manifest,
  normalize_stack_value as _normalize_stack_value,
)


SUNNYPILOT_CURRENT = "sunnypilot-current"
CUSTOM_RECOMMENDED = "custom-recommended"
CUSTOM_V2 = "custom-2.0"
CUSTOM_EXPERIMENTAL = "custom-experimental"
DEFAULT_STACK = SUNNYPILOT_CURRENT

MANIFEST_PATH = Path(__file__).with_name("longitudinal_stack_versions.json")


class PlatformCapabilities:
  def __init__(self, brand: str = "", car_fingerprint: str = "", openpilot_longitudinal_control: bool = False,
               alpha_longitudinal_available: bool = False, pcm_cruise: bool = False, radar_unavailable: bool = False,
               cp_sp_flags: int = 0):
    self.brand = brand
    self.car_fingerprint = car_fingerprint
    self.openpilot_longitudinal_control = openpilot_longitudinal_control
    self.alpha_longitudinal_available = alpha_longitudinal_available
    self.pcm_cruise = pcm_cruise
    self.radar_unavailable = radar_unavailable
    self.cp_sp_flags = cp_sp_flags


StackResolution = _StackResolution
StackDefinition = _StackDefinition


class StackCatalog(_StackCatalog):
  @property
  def default_stack(self) -> str:
    return str(self.manifest.get("defaultStack") or DEFAULT_STACK)

  def _capability_value(self, capabilities: object, key: str) -> bool:
    capabilities = capabilities if isinstance(capabilities, PlatformCapabilities) else PlatformCapabilities()
    mapping = {
      "openpilotLongitudinalControl": capabilities.openpilot_longitudinal_control,
      "alphaLongitudinalAvailable": capabilities.alpha_longitudinal_available,
      "pcmCruise": capabilities.pcm_cruise,
      "radarUnavailable": capabilities.radar_unavailable,
    }
    return bool(mapping.get(key, False))


def load_stack_manifest(path: str | Path = MANIFEST_PATH) -> dict:
  return _load_stack_manifest(path)


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


def get_available_stacks(manifest: dict, capabilities: PlatformCapabilities) -> tuple[str, ...]:
  return StackCatalog(manifest).available_stacks(capabilities)


def is_custom_stack(stack: str) -> bool:
  return str(stack or "").startswith("custom-")


def resolve_longitudinal_stack(requested_stack: object, CP: object | None = None,
                               CP_SP: object | None = None, manifest: dict | None = None) -> StackResolution:
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
    return int(value)  # type: ignore[arg-type]
  except (TypeError, ValueError):
    return 0
