"""
Lateral demand stack abstraction.

The lateral demand stack wraps a LateralDemandProfileBuilder
with a per-frame `update()` that returns both the legacy
ProcessedLateralDemand (forwarded to the controller as today)
and a LateralDemandProfile the controller reads inside
_build_target for v5 profile-aware preview gating, turn-exit
source-of-truth, and demand-mode telemetry.

Three stacks are defined for 5.0:

- SunnypilotCurrentLateralDemandStack:  migration default.  The
  profile is the existing one and is consumed only by
  telemetry.  No v5 active delta.  Behavior is bit-equivalent
  to today.
- CustomV2LateralDemandStack:  profile-aware custom-2.0 lateral
  demand.  Same builder output, but the stack is the contract
  surface for future custom-2.0 shaping.  Pairs with
  TorqueControlTune=4.1.
- CustomExperimentalLateralDemandStack:  v5-shaped profile.  The
  preview_lateral_accel_* fields and mode flags are populated
  for the v5 preview boost.  Pairs with TorqueControlTune=5.0.

Selection is via the LateralDemandStack param
(string).  Unknown / missing values resolve to
LateralDemandStackId.SUNNYPILOT_CURRENT for migration safety.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from openpilot.selfdrive.controls.lib.lateral_demand import ProcessedLateralDemand
from openpilot.selfdrive.controls.lib.lateral_demand_profile import (
  LateralDemandProfile,
  LateralDemandProfileBuilder,
)


class LateralDemandStackId(str, Enum):
  SUNNYPILOT_CURRENT = "sunnypilot-current"
  CUSTOM_RECOMMENDED = "custom-recommended"
  CUSTOM_V2 = "custom-2.0"
  CUSTOM_EXPERIMENTAL = "custom-experimental"


DEFAULT_LATERAL_DEMAND_STACK = LateralDemandStackId.CUSTOM_V2


class TorqueControlTuneId(str, Enum):
  V20 = "2.0"
  V21 = "2.1"
  V30 = "3.0"
  V40 = "4.0"
  V41 = "4.1"
  V50_EXPERIMENTAL = "5.0"


DEFAULT_TORQUE_CONTROL_TUNE = TorqueControlTuneId.V41
SAFE_TORQUE_TUNE_FALLBACK = TorqueControlTuneId.V41


def torque_control_tune_id_for_name(value) -> TorqueControlTuneId:
  if isinstance(value, TorqueControlTuneId):
    return value
  if value is None:
    return DEFAULT_TORQUE_CONTROL_TUNE
  if isinstance(value, bytes):
    try:
      value = value.decode("utf-8")
    except UnicodeDecodeError:
      return DEFAULT_TORQUE_CONTROL_TUNE
  if isinstance(value, (int, float)):
    value = f"{float(value):.1f}"
  if not isinstance(value, str):
    return DEFAULT_TORQUE_CONTROL_TUNE
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
  """Resolve a raw param value to a TorqueControlTuneResolution.
  Unknown / missing values resolve to the safe fallback (4.1),
  never 5.0. 5.0 is only selected when the user explicitly
  asked for it."""
  requested = torque_control_tune_id_for_name(value)
  if requested == TorqueControlTuneId.V50_EXPERIMENTAL and value not in (
    TorqueControlTuneId.V50_EXPERIMENTAL.value, "5.0", b"5.0", 5.0,
  ):
    return TorqueControlTuneResolution(
      requested_tune=requested,
      resolved_tune=SAFE_TORQUE_TUNE_FALLBACK,
      fallback_reason="experimental_requires_explicit_selection",
    )
  return TorqueControlTuneResolution(
    requested_tune=requested,
    resolved_tune=requested,
  )


class ControlsProfileId(str, Enum):
  """User-facing driving-profile alias. The profile auto-couples
  LateralDemandStack and TorqueControlTune so the user can pick
  one value instead of two. Advanced selectors break the
  coupling."""
  SUNNYPILOT_CURRENT = "sunnypilot-current"
  CUSTOM_RECOMMENDED = "custom-recommended"
  CUSTOM_2 = "custom-2.0"
  CUSTOM_EXPERIMENTAL = "custom-experimental"


DEFAULT_CONTROLS_PROFILE = ControlsProfileId.CUSTOM_2


@dataclass(frozen=True)
class ControlsProfileMapping:
  profile_id: ControlsProfileId
  longitudinal_stack: str
  lateral_demand_stack: LateralDemandStackId
  torque_control_tune: TorqueControlTuneId


def _build_controls_profile_mappings() -> tuple[ControlsProfileMapping, ...]:
  return (
    ControlsProfileMapping(
      ControlsProfileId.SUNNYPILOT_CURRENT,
      "sunnypilot-current",
      LateralDemandStackId.SUNNYPILOT_CURRENT,
      SAFE_TORQUE_TUNE_FALLBACK,
    ),
    ControlsProfileMapping(
      ControlsProfileId.CUSTOM_RECOMMENDED,
      "custom-recommended",
      LateralDemandStackId.CUSTOM_RECOMMENDED,
      SAFE_TORQUE_TUNE_FALLBACK,
    ),
    ControlsProfileMapping(
      ControlsProfileId.CUSTOM_2,
      "custom-2.0",
      LateralDemandStackId.CUSTOM_V2,
      TorqueControlTuneId.V41,
    ),
    ControlsProfileMapping(
      ControlsProfileId.CUSTOM_EXPERIMENTAL,
      "custom-experimental",
      LateralDemandStackId.CUSTOM_EXPERIMENTAL,
      TorqueControlTuneId.V50_EXPERIMENTAL,
    ),
  )


CONTROLS_PROFILE_MAPPINGS: tuple[ControlsProfileMapping, ...] = _build_controls_profile_mappings()


def controls_profile_id_for_name(value) -> ControlsProfileId:
  if isinstance(value, ControlsProfileId):
    return value
  if value is None:
    return DEFAULT_CONTROLS_PROFILE
  if isinstance(value, bytes):
    try:
      value = value.decode("utf-8")
    except UnicodeDecodeError:
      return DEFAULT_CONTROLS_PROFILE
  if not isinstance(value, str):
    return DEFAULT_CONTROLS_PROFILE
  for mapping in CONTROLS_PROFILE_MAPPINGS:
    if mapping.profile_id.value == value:
      return mapping.profile_id
  return DEFAULT_CONTROLS_PROFILE


def controls_profile_mapping_for(profile_id: ControlsProfileId) -> ControlsProfileMapping:
  for mapping in CONTROLS_PROFILE_MAPPINGS:
    if mapping.profile_id == profile_id:
      return mapping
  return CONTROLS_PROFILE_MAPPINGS[0]


@dataclass(frozen=True)
class ControlsProfileResolution:
  """Full resolution of the controls profile system.

  requested_profile is what the user asked for (or
  DEFAULT_CONTROLS_PROFILE if missing).  resolved_profile is
  the profile that was actually applied (after the fallback
  pass for missing / unknown values).  longitudinal_stack,
  lateral_demand_stack, and torque_control_tune are the
  per-layer values the resolver chose.  fallback_reason is
  non-empty when any of the resolved fields were changed
  from the requested profile's natural mapping.
  """
  requested_profile: ControlsProfileId
  resolved_profile: ControlsProfileId
  longitudinal_stack: str
  lateral_demand_stack: LateralDemandStackId
  torque_control_tune: TorqueControlTuneId
  fallback_reason: str = ""
  lateral_demand_stack_resolution: Optional[object] = None
  torque_control_tune_resolution: Optional[TorqueControlTuneResolution] = None


def resolve_controls_profile(
  value,
  *,
  advanced_lateral_demand_stack: object = None,
  advanced_torque_control_tune: object = None,
  advanced_overrides_enabled: bool = False,
) -> ControlsProfileResolution:
  """Resolve the user-facing driving profile into the full
  ControlsProfileResolution.

  - Missing / unknown profile → DEFAULT_CONTROLS_PROFILE
  - advanced_lateral_demand_stack / advanced_torque_control_tune
    replace the per-layer fields when the user has explicitly
    set them AND advanced_overrides_enabled is True.
  - 5.0 is never selected for an existing user by a missing or
    default profile; it is only reached via the
    custom-experimental profile or an explicit TorqueControlTune
    override.
  """
  requested = controls_profile_id_for_name(value)
  mapping = controls_profile_mapping_for(requested)

  effective_lat_demand = mapping.lateral_demand_stack
  fallback_reasons: list[str] = []
  if advanced_overrides_enabled and advanced_lateral_demand_stack is not None:
    override = lateral_demand_stack_id_for_name(advanced_lateral_demand_stack)
    if override != effective_lat_demand:
      fallback_reasons.append("advanced_lateral_demand_stack_override")
    effective_lat_demand = override
  lateral_resolution = resolve_lateral_demand_stack(effective_lat_demand)
  effective_lat_demand = lateral_resolution.resolved_stack

  effective_torque = mapping.torque_control_tune
  if advanced_overrides_enabled and advanced_torque_control_tune is not None:
    override_torque = torque_control_tune_id_for_name(advanced_torque_control_tune)
    if override_torque != effective_torque:
      fallback_reasons.append("advanced_torque_control_tune_override")
    effective_torque = override_torque
  torque_resolution = TorqueControlTuneResolution(
    requested_tune=effective_torque,
    resolved_tune=effective_torque,
  )

  return ControlsProfileResolution(
    requested_profile=requested,
    resolved_profile=requested,
    longitudinal_stack=mapping.longitudinal_stack,
    lateral_demand_stack=effective_lat_demand,
    torque_control_tune=effective_torque,
    fallback_reason=",".join(fallback_reasons),
    lateral_demand_stack_resolution=lateral_resolution,
    torque_control_tune_resolution=torque_resolution,
  )


@dataclass(frozen=True)
class LateralDemandStackOutput:
  legacy: ProcessedLateralDemand
  profile: LateralDemandProfile


@dataclass
class LateralDemandStack:
  """Base class for lateral demand stacks.

  Concrete stacks override _id and (optionally) the builder
  class.  update() builds the same-frame profile and bundles
  it with the legacy demand into a LateralDemandStackOutput.
  """

  dt: float = 0.05

  def __post_init__(self) -> None:
    self._builder: LateralDemandProfileBuilder = self._builder_factory()

  def _builder_factory(self) -> LateralDemandProfileBuilder:
    return LateralDemandProfileBuilder(dt=self.dt)

  @property
  def stack_id(self) -> LateralDemandStackId:
    raise NotImplementedError

  def reset(self) -> None:
    self._builder.reset()

  def update(
    self,
    demand: ProcessedLateralDemand,
    v_ego: float,
    *,
    curvature_limited: bool = False,
    saturated: bool = False,
    steer_limited_by_safety: bool = False,
    steering_pressed: bool = False,
  ) -> LateralDemandStackOutput:
    profile = self._builder.update(
      demand,
      v_ego,
      curvature_limited=curvature_limited,
      saturated=saturated,
      steer_limited_by_safety=steer_limited_by_safety,
      steering_pressed=steering_pressed,
    )
    return LateralDemandStackOutput(legacy=demand, profile=profile)


@dataclass
class SunnypilotCurrentLateralDemandStack(LateralDemandStack):
  """Migration default.  Profile output is the existing one and
  is consumed only by telemetry.  No v5 active delta.  Behavior
  is bit-equivalent to today."""

  @property
  def stack_id(self) -> LateralDemandStackId:
    return LateralDemandStackId.SUNNYPILOT_CURRENT


@dataclass
class CustomV2LateralDemandStack(LateralDemandStack):
  """custom-2.0 lateral demand stack.  Profile-aware but
  conservative; pairs with TorqueControlTune=4.1.  Same profile
  shape as the current builder; reserved as the contract surface
  for future custom-2.0 shaping."""

  @property
  def stack_id(self) -> LateralDemandStackId:
    return LateralDemandStackId.CUSTOM_V2


@dataclass
class CustomExperimentalLateralDemandStack(LateralDemandStack):
  """custom-experimental (v5) lateral demand stack.  The profile
  carries the preview_lateral_accel_* and mode fields the v5
  preview boost reads inside _build_target.  Pairs with
  TorqueControlTune=5.0."""

  @property
  def stack_id(self) -> LateralDemandStackId:
    return LateralDemandStackId.CUSTOM_EXPERIMENTAL


@dataclass
class CustomRecommendedLateralDemandStack(LateralDemandStack):
  """Recommended custom lateral demand stack.

  Not yet implemented as a distinct class.  The class exists
  so the user-facing 'custom-recommended' value resolves to a
  real object instead of failing; runtime behavior is
  identical to CustomV2LateralDemandStack and the resolution
  carries fallback_reason='not_implemented' so the manifest
  can mark the entry as a stub."""

  @property
  def stack_id(self) -> LateralDemandStackId:
    return LateralDemandStackId.CUSTOM_RECOMMENDED


@dataclass(frozen=True)
class LateralDemandStackDefinition:
  stack_id: LateralDemandStackId
  factory: type


LATERAL_DEMAND_STACK_REGISTRY = LateralDemandStackDefinition  # type alias for clarity


def _lateral_demand_stack_definitions() -> tuple[LateralDemandStackDefinition, ...]:
  return (
    LateralDemandStackDefinition(
      LateralDemandStackId.SUNNYPILOT_CURRENT, SunnypilotCurrentLateralDemandStack,
    ),
    LateralDemandStackDefinition(
      LateralDemandStackId.CUSTOM_RECOMMENDED, CustomRecommendedLateralDemandStack,
    ),
    LateralDemandStackDefinition(
      LateralDemandStackId.CUSTOM_V2, CustomV2LateralDemandStack,
    ),
    LateralDemandStackDefinition(
      LateralDemandStackId.CUSTOM_EXPERIMENTAL, CustomExperimentalLateralDemandStack,
    ),
  )


def lateral_demand_stack_id_for_name(value) -> LateralDemandStackId:
  """Resolve a raw param value (str | bytes | LateralDemandStackId
  | None) to a known LateralDemandStackId.  Unknown / missing
  values resolve to the migration default (SunnypilotCurrent)
  for safety."""
  if isinstance(value, LateralDemandStackId):
    return value
  if value is None:
    return DEFAULT_LATERAL_DEMAND_STACK
  if isinstance(value, bytes):
    try:
      value = value.decode("utf-8")
    except UnicodeDecodeError:
      return DEFAULT_LATERAL_DEMAND_STACK
  if not isinstance(value, str):
    return DEFAULT_LATERAL_DEMAND_STACK
  for definition in _lateral_demand_stack_definitions():
    if definition.stack_id.value == value:
      return definition.stack_id
  return DEFAULT_LATERAL_DEMAND_STACK


@dataclass(frozen=True)
class LateralDemandStackResolution:
  """Resolution object for the lateral demand stack selector.

  requested_stack is the value the user (or upstream caller)
  asked for.  resolved_stack is the value the resolver chose
  (post-fallback).  fallback_reason is non-empty when the
  resolved_stack differs from a non-empty requested_stack or
  when the resolved_stack is a documented fallback
  (custom-recommended with no real implementation,
  custom-experimental unavailable on this platform, etc.).
  """
  requested_stack: LateralDemandStackId
  resolved_stack: LateralDemandStackId
  fallback_reason: str = ""


def resolve_lateral_demand_stack(
  value, *, dt: float = 0.05,
) -> LateralDemandStackResolution:
  """Resolve a raw param value to a LateralDemandStackResolution.
  Unknown / missing values resolve to the safe default
  (custom-2.0).  custom-recommended carries
  fallback_reason='not_implemented'."""
  requested = lateral_demand_stack_id_for_name(value)
  fallback_reason = ""
  if requested == LateralDemandStackId.CUSTOM_RECOMMENDED:
    fallback_reason = "not_implemented"
  if requested not in {d.stack_id for d in _lateral_demand_stack_definitions()}:
    fallback_reason = "unknown"
  return LateralDemandStackResolution(
    requested_stack=requested,
    resolved_stack=requested,
    fallback_reason=fallback_reason,
  )


def build_lateral_demand_stack(stack_id, dt: float = 0.05) -> LateralDemandStack:
  """Construct a concrete LateralDemandStack instance by id.
  Pass a LateralDemandStackResolution to use the resolved_stack."""
  if isinstance(stack_id, LateralDemandStackResolution):
    target = stack_id.resolved_stack
  else:
    target = lateral_demand_stack_id_for_name(stack_id)
  for definition in _lateral_demand_stack_definitions():
    if definition.stack_id == target:
      return definition.factory(dt=dt)
  return CustomV2LateralDemandStack(dt=dt)
