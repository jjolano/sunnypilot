from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from openpilot.selfdrive.controls.lib.lateral_demand_stacks.selector import (
  CUSTOM_EXPERIMENTAL as LATERAL_CUSTOM_EXPERIMENTAL,
  CUSTOM_RECOMMENDED as LATERAL_CUSTOM_RECOMMENDED,
  CUSTOM_V2 as LATERAL_CUSTOM_V2,
  SUNNYPILOT_CURRENT as LATERAL_SUNNYPILOT_CURRENT,
  LateralDemandStackResolution,
  resolve_lateral_demand_stack,
)
from openpilot.selfdrive.controls.lib.longitudinal_stacks.selector import (
  CUSTOM_RECOMMENDED as LONG_CUSTOM_RECOMMENDED,
  CUSTOM_V2 as LONG_CUSTOM_V2,
  SUNNYPILOT_CURRENT as LONG_SUNNYPILOT_CURRENT,
)


class ControlsProfileId(str, Enum):
  SUNNYPILOT_CURRENT = "sunnypilot-current"
  CUSTOM_RECOMMENDED = "custom-recommended"
  CUSTOM_2 = "custom-2.0"
  CUSTOM_EXPERIMENTAL = "custom-experimental"


class TorqueControlTuneId(str, Enum):
  V20 = "2.0"
  V21 = "2.1"
  V30 = "3.0"
  V40 = "4.0"
  V41 = "4.1"
  V50_EXPERIMENTAL = "5.0"


DEFAULT_CONTROLS_PROFILE = ControlsProfileId.CUSTOM_2
DEFAULT_LATERAL_DEMAND_STACK = LATERAL_CUSTOM_V2
DEFAULT_TORQUE_CONTROL_TUNE = TorqueControlTuneId.V41
SAFE_TORQUE_TUNE_FALLBACK = TorqueControlTuneId.V41

KNOWN_LATERAL_DEMAND_STACKS = frozenset({
  LATERAL_SUNNYPILOT_CURRENT,
  LATERAL_CUSTOM_RECOMMENDED,
  LATERAL_CUSTOM_V2,
  LATERAL_CUSTOM_EXPERIMENTAL,
})
KNOWN_LONGITUDINAL_STACKS = frozenset({
  LONG_SUNNYPILOT_CURRENT,
  LONG_CUSTOM_RECOMMENDED,
  LONG_CUSTOM_V2,
})


def _decode_param_value(value):
  if isinstance(value, bytes):
    try:
      return value.decode("utf-8")
    except UnicodeDecodeError:
      return ""
  return value


def _param_has_value(params, key: str) -> bool:
  try:
    value = params.get(key)
  except Exception:
    return False
  if value is None:
    return False
  if isinstance(value, bytes):
    return len(value.strip()) > 0
  if isinstance(value, str):
    return len(value.strip()) > 0
  return bool(value)


def _param_value(params, key: str):
  try:
    return params.get(key)
  except Exception:
    return None


def _param_bool(params, key: str) -> bool:
  try:
    return bool(params.get_bool(key))
  except Exception:
    return False


def torque_control_tune_id_for_name(value) -> TorqueControlTuneId:
  if isinstance(value, TorqueControlTuneId):
    return value
  value = _decode_param_value(value)
  if value is None:
    return DEFAULT_TORQUE_CONTROL_TUNE
  if isinstance(value, (int, float)):
    value = f"{float(value):.1f}"
  if not isinstance(value, str):
    return DEFAULT_TORQUE_CONTROL_TUNE
  value = value.strip()
  for member in TorqueControlTuneId:
    if member.value == value:
      return member
  return DEFAULT_TORQUE_CONTROL_TUNE


@dataclass(frozen=True)
class TorqueControlTuneResolution:
  requested_tune: TorqueControlTuneId
  resolved_tune: TorqueControlTuneId
  fallback_reason: str = ""


def resolve_torque_control_tune(value) -> TorqueControlTuneResolution:
  requested = torque_control_tune_id_for_name(value)
  return TorqueControlTuneResolution(requested, requested)


def lateral_demand_stack_id_for_name(value) -> str:
  value = _decode_param_value(value)
  if value is None:
    return DEFAULT_LATERAL_DEMAND_STACK
  value = str(value).strip()
  if value in KNOWN_LATERAL_DEMAND_STACKS:
    return value
  return DEFAULT_LATERAL_DEMAND_STACK


def longitudinal_stack_id_for_name(value, default_stack: str = LONG_CUSTOM_V2) -> str:
  value = _decode_param_value(value)
  if value is None:
    return default_stack
  value = str(value).strip()
  if value in KNOWN_LONGITUDINAL_STACKS:
    return value
  return default_stack


@dataclass(frozen=True)
class ControlsProfileMapping:
  profile_id: ControlsProfileId
  longitudinal_stack: str
  lateral_demand_stack: str
  torque_control_tune: TorqueControlTuneId


def _build_controls_profile_mappings() -> tuple[ControlsProfileMapping, ...]:
  return (
    ControlsProfileMapping(
      ControlsProfileId.SUNNYPILOT_CURRENT,
      LONG_SUNNYPILOT_CURRENT,
      LATERAL_SUNNYPILOT_CURRENT,
      SAFE_TORQUE_TUNE_FALLBACK,
    ),
    ControlsProfileMapping(
      ControlsProfileId.CUSTOM_RECOMMENDED,
      LONG_CUSTOM_RECOMMENDED,
      LATERAL_CUSTOM_RECOMMENDED,
      SAFE_TORQUE_TUNE_FALLBACK,
    ),
    ControlsProfileMapping(
      ControlsProfileId.CUSTOM_2,
      LONG_CUSTOM_V2,
      LATERAL_CUSTOM_V2,
      TorqueControlTuneId.V41,
    ),
    ControlsProfileMapping(
      ControlsProfileId.CUSTOM_EXPERIMENTAL,
      # No separate experimental longitudinal stack is implemented yet;
      # keep profile-owned longitudinal selection on the stable custom-v2
      # stack instead of persisting an unknown LongitudinalStack value.
      LONG_CUSTOM_V2,
      LATERAL_CUSTOM_EXPERIMENTAL,
      TorqueControlTuneId.V50_EXPERIMENTAL,
    ),
  )


CONTROLS_PROFILE_MAPPINGS: tuple[ControlsProfileMapping, ...] = _build_controls_profile_mappings()


def controls_profile_id_for_name(value) -> ControlsProfileId:
  if isinstance(value, ControlsProfileId):
    return value
  value = _decode_param_value(value)
  if value is None:
    return DEFAULT_CONTROLS_PROFILE
  if not isinstance(value, str):
    return DEFAULT_CONTROLS_PROFILE
  value = value.strip()
  for mapping in CONTROLS_PROFILE_MAPPINGS:
    if mapping.profile_id.value == value:
      return mapping.profile_id
  return DEFAULT_CONTROLS_PROFILE


def controls_profile_mapping_for(profile_id: ControlsProfileId) -> ControlsProfileMapping:
  for mapping in CONTROLS_PROFILE_MAPPINGS:
    if mapping.profile_id == profile_id:
      return mapping
  return controls_profile_mapping_for(DEFAULT_CONTROLS_PROFILE)


@dataclass(frozen=True)
class ControlsProfileResolution:
  requested_profile: ControlsProfileId
  resolved_profile: ControlsProfileId
  longitudinal_stack: str
  lateral_demand_stack: str
  torque_control_tune: TorqueControlTuneId
  fallback_reason: str = ""
  lateral_demand_stack_resolution: LateralDemandStackResolution | None = None
  torque_control_tune_resolution: TorqueControlTuneResolution | None = None


@dataclass(frozen=True)
class ControlsProfileParamResolution:
  controls_profile_resolution: ControlsProfileResolution
  controls_profile_explicit: bool
  torque_control_tune_explicit: bool
  lateral_demand_stack_explicit: bool
  longitudinal_stack_explicit: bool
  advanced_overrides_enabled: bool


def resolve_controls_profile(
  value,
  *,
  advanced_lateral_demand_stack: object = None,
  advanced_torque_control_tune: object = None,
  advanced_longitudinal_stack: object = None,
  advanced_overrides_enabled: bool = False,
) -> ControlsProfileResolution:
  requested = controls_profile_id_for_name(value)
  mapping = controls_profile_mapping_for(requested)

  fallback_reasons: list[str] = []

  effective_lateral = mapping.lateral_demand_stack
  if advanced_overrides_enabled and advanced_lateral_demand_stack is not None:
    override = lateral_demand_stack_id_for_name(advanced_lateral_demand_stack)
    if override != effective_lateral:
      fallback_reasons.append("advanced_lateral_demand_stack_override")
    effective_lateral = override
  lateral_resolution = resolve_lateral_demand_stack(effective_lateral)
  effective_lateral = lateral_resolution.resolved_stack

  effective_torque = mapping.torque_control_tune
  if advanced_overrides_enabled and advanced_torque_control_tune is not None:
    override_torque = torque_control_tune_id_for_name(advanced_torque_control_tune)
    if override_torque != effective_torque:
      fallback_reasons.append("advanced_torque_control_tune_override")
    effective_torque = override_torque
  torque_resolution = resolve_torque_control_tune(effective_torque)
  effective_torque = torque_resolution.resolved_tune

  effective_longitudinal = mapping.longitudinal_stack
  if advanced_overrides_enabled and advanced_longitudinal_stack is not None:
    override_longitudinal = longitudinal_stack_id_for_name(advanced_longitudinal_stack, mapping.longitudinal_stack)
    if override_longitudinal != effective_longitudinal:
      fallback_reasons.append("advanced_longitudinal_stack_override")
    effective_longitudinal = override_longitudinal

  return ControlsProfileResolution(
    requested_profile=requested,
    resolved_profile=requested,
    longitudinal_stack=effective_longitudinal,
    lateral_demand_stack=effective_lateral,
    torque_control_tune=effective_torque,
    fallback_reason=",".join(fallback_reasons),
    lateral_demand_stack_resolution=lateral_resolution,
    torque_control_tune_resolution=torque_resolution,
  )


def resolve_controls_profile_from_params(params) -> ControlsProfileParamResolution:
  controls_profile_explicit = _param_has_value(params, "ControlsProfile")
  torque_tune_explicit = _param_has_value(params, "TorqueControlTune")
  lateral_stack_explicit = _param_has_value(params, "LateralDemandStack")
  longitudinal_stack_explicit = _param_has_value(params, "LongitudinalStack")
  advanced_overrides_enabled = _param_bool(params, "ShowAdvancedControls")

  if controls_profile_explicit:
    resolution = resolve_controls_profile(
      _param_value(params, "ControlsProfile"),
      advanced_lateral_demand_stack=(
        _param_value(params, "LateralDemandStack")
        if advanced_overrides_enabled and lateral_stack_explicit else None
      ),
      advanced_torque_control_tune=(
        _param_value(params, "TorqueControlTune")
        if advanced_overrides_enabled and torque_tune_explicit else None
      ),
      advanced_longitudinal_stack=(
        _param_value(params, "LongitudinalStack")
        if advanced_overrides_enabled and longitudinal_stack_explicit else None
      ),
      advanced_overrides_enabled=advanced_overrides_enabled,
    )
  else:
    default_resolution = resolve_controls_profile(None)
    lateral_value = default_resolution.lateral_demand_stack
    torque_value = SAFE_TORQUE_TUNE_FALLBACK
    longitudinal_value = default_resolution.longitudinal_stack
    fallback_reasons: list[str] = []

    if lateral_stack_explicit:
      lateral_value = lateral_demand_stack_id_for_name(_param_value(params, "LateralDemandStack"))
      fallback_reasons.append("existing_lateral_demand_stack_preserved")
    if torque_tune_explicit:
      torque_value = resolve_torque_control_tune(_param_value(params, "TorqueControlTune")).resolved_tune
      fallback_reasons.append("existing_torque_control_tune_preserved")
    if longitudinal_stack_explicit:
      longitudinal_value = longitudinal_stack_id_for_name(_param_value(params, "LongitudinalStack"), default_resolution.longitudinal_stack)
      fallback_reasons.append("existing_longitudinal_stack_preserved")

    lateral_resolution = resolve_lateral_demand_stack(lateral_value)
    torque_resolution = resolve_torque_control_tune(torque_value)
    resolution = ControlsProfileResolution(
      requested_profile=DEFAULT_CONTROLS_PROFILE,
      resolved_profile=DEFAULT_CONTROLS_PROFILE,
      longitudinal_stack=longitudinal_value,
      lateral_demand_stack=lateral_resolution.resolved_stack,
      torque_control_tune=torque_resolution.resolved_tune,
      fallback_reason=",".join(fallback_reasons),
      lateral_demand_stack_resolution=lateral_resolution,
      torque_control_tune_resolution=torque_resolution,
    )

  return ControlsProfileParamResolution(
    controls_profile_resolution=resolution,
    controls_profile_explicit=controls_profile_explicit,
    torque_control_tune_explicit=torque_tune_explicit,
    lateral_demand_stack_explicit=lateral_stack_explicit,
    longitudinal_stack_explicit=longitudinal_stack_explicit,
    advanced_overrides_enabled=advanced_overrides_enabled,
  )
