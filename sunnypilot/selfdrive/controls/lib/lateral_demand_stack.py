"""
Lateral demand stack and controls profile abstraction.

The lateral demand stack wraps a LateralDemandProfileBuilder
with a per-frame `update()` that returns both the legacy
ProcessedLateralDemand (forwarded to the controller as today)
and a LateralDemandProfile the controller reads inside
_build_target for v5 profile-aware preview gating, turn-exit
source-of-truth, and demand-mode telemetry.

Four stacks are defined for 5.0:

- SunnypilotCurrentLateralDemandStack   migration default.  The
                                        profile is the existing
                                        one and is consumed only
                                        by telemetry.  No v5
                                        active delta.  Behavior is
                                        bit-equivalent to today.
- CustomRecommendedLateralDemandStack   contract surface for
                                        the recommended custom
                                        stack.  Not yet
                                        implemented; falls back
                                        to CustomV2 with
                                        fallback_reason metadata.
- CustomV2LateralDemandStack            custom-2.0 lateral
                                        demand.  Profile-aware but
                                        conservative; pairs with
                                        TorqueControlTune=4.1.
- ExperimentalLateralDemandStack        v5-shaped profile.  The
                                        preview_lateral_accel_*
                                        fields and mode flags are
                                        populated for the v5
                                        preview boost.  Pairs with
                                        TorqueControlTune=5.0.

Selection is via the LateralDemandStack param (string).
Unknown / missing values resolve to
LateralDemandStackId.CUSTOM_V2 for safety.

The ControlsProfile param is the user-facing driving-profile
alias that auto-couples LateralDemandStack and
TorqueControlTune.  The user picks one value; the per-layer
advanced selectors break the coupling when ShowAdvancedControls
is on.  Custom-experimental does NOT silently expose
unavailable stacks: CustomRecommendedLateralDemandStack is
explicitly marked with a fallback_reason so callers know they
got the CustomV2 implementation under that label.
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


# ---------------------------------------------------------------------------
# Lateral Demand Stack
# ---------------------------------------------------------------------------


class LateralDemandStackId(str, Enum):
  SUNNYPILOT_CURRENT = "sunnypilot-current"
  CUSTOM_RECOMMENDED = "custom-recommended"
  CUSTOM_V2 = "custom-2.0"
  CUSTOM_EXPERIMENTAL = "custom-experimental"


DEFAULT_LATERAL_DEMAND_STACK = LateralDemandStackId.CUSTOM_V2


@dataclass(frozen=True)
class LateralDemandStackResolution:
  """The output of resolve_lateral_demand_stack. Carries the
  requested id, the resolved id (post-fallback), and a
  fallback_reason when the resolved id differs from the
  requested id (or when a non-implemented class delegated to
  CustomV2)."""
  requested_stack: LateralDemandStackId
  resolved_stack: LateralDemandStackId
  fallback_reason: str = ""
  available: bool = True


@dataclass(frozen=True)
class LateralDemandStackInputs:
  """Per-frame bundle the lateral demand stack consumes.

  Carries the ProcessedLateralDemand the stack passes through
  to its output.legacy, plus the v_ego and gates the stack's
  profile builder needs to classify the mode.
  """
  processed_lateral_demand: ProcessedLateralDemand
  v_ego: float
  curvature_limited: bool = False
  saturated: bool = False
  steer_limited_by_safety: bool = False
  steering_pressed: bool = False


@dataclass(frozen=True)
class LateralDemandStackOutput:
  legacy: ProcessedLateralDemand
  profile: LateralDemandProfile


# ---------------------------------------------------------------------------
# Stack base + concrete stacks
# ---------------------------------------------------------------------------


@dataclass
class LateralDemandStack:
  """Base class for lateral demand stacks."""

  dt: float = 0.05

  def __post_init__(self) -> None:
    self._builder: LateralDemandProfileBuilder = LateralDemandProfileBuilder(dt=self.dt)

  @property
  def stack_id(self) -> LateralDemandStackId:
    raise NotImplementedError

  def reset(self) -> None:
    self._builder.reset()

  def update(self, inputs: LateralDemandStackInputs) -> LateralDemandStackOutput:
    profile = self._builder.update(
      inputs.processed_lateral_demand,
      inputs.v_ego,
      curvature_limited=inputs.curvature_limited,
      saturated=inputs.saturated,
      steer_limited_by_safety=inputs.steer_limited_by_safety,
      steering_pressed=inputs.steering_pressed,
    )
    return LateralDemandStackOutput(legacy=inputs.processed_lateral_demand, profile=profile)


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
class CustomRecommendedLateralDemandStack(LateralDemandStack):
  """Recommended custom lateral demand stack.

  Not yet implemented as a distinct implementation.  The class
  exists so the user-facing 'custom-recommended' value resolves
  to a real object instead of failing; runtime behavior is
  identical to CustomV2LateralDemandStack and the resolution
  carries fallback_reason='not_implemented' so the manifest
  can mark the entry as a stub."""

  @property
  def stack_id(self) -> LateralDemandStackId:
    return LateralDemandStackId.CUSTOM_RECOMMENDED


@dataclass
class ExperimentalLateralDemandStack(LateralDemandStack):
  """v5-shaped profile.  The preview_lateral_accel_* and mode
  fields are populated for the v5 preview boost.  Pairs with
  TorqueControlTune=5.0."""

  @property
  def stack_id(self) -> LateralDemandStackId:
    return LateralDemandStackId.CUSTOM_EXPERIMENTAL


# Backward-compat alias: the previous class was named
# CustomExperimentalLateralDemandStack.  Keep it as an alias so
# external code that imported the old name still works.
CustomExperimentalLateralDemandStack = ExperimentalLateralDemandStack


@dataclass(frozen=True)
class LateralDemandStackDefinition:
  stack_id: LateralDemandStackId
  factory: type
  available: bool = True
  fallback_reason: str = ""


def _lateral_demand_stack_definitions() -> tuple[LateralDemandStackDefinition, ...]:
  return (
    LateralDemandStackDefinition(
      LateralDemandStackId.SUNNYPILOT_CURRENT, SunnypilotCurrentLateralDemandStack,
      available=True,
    ),
    LateralDemandStackDefinition(
      LateralDemandStackId.CUSTOM_RECOMMENDED, CustomRecommendedLateralDemandStack,
      available=False,
      fallback_reason="not_implemented",
    ),
    LateralDemandStackDefinition(
      LateralDemandStackId.CUSTOM_V2, CustomV2LateralDemandStack,
      available=True,
    ),
    LateralDemandStackDefinition(
      LateralDemandStackId.CUSTOM_EXPERIMENTAL, ExperimentalLateralDemandStack,
      available=True,
    ),
  )


def lateral_demand_stack_id_for_name(value) -> LateralDemandStackId:
  """Resolve a raw param value (str | bytes |
  LateralDemandStackId | None) to a known LateralDemandStackId.
  Unknown / missing values resolve to the migration default
  (CustomV2) for safety."""
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


def resolve_lateral_demand_stack(
  value, *, CP=None, CP_SP=None,
) -> LateralDemandStackResolution:
  """Resolve a raw param value into a LateralDemandStackResolution.

  Unknown / missing values resolve to the migration default
  (CustomV2).  Stack classes that are not yet implemented
  (CustomRecommended) carry fallback_reason='not_implemented'
  so callers know to treat the entry as a stub.
  """
  requested = lateral_demand_stack_id_for_name(value)
  for definition in _lateral_demand_stack_definitions():
    if definition.stack_id == requested:
      if not definition.available:
        return LateralDemandStackResolution(
          requested_stack=requested,
          resolved_stack=LateralDemandStackId.CUSTOM_V2,
          fallback_reason=definition.fallback_reason,
          available=False,
        )
      return LateralDemandStackResolution(
        requested_stack=requested,
        resolved_stack=definition.stack_id,
      )
  return LateralDemandStackResolution(
    requested_stack=requested,
    resolved_stack=DEFAULT_LATERAL_DEMAND_STACK,
    fallback_reason="unknown",
  )


def build_lateral_demand_stack(stack_id, dt: float = 0.05) -> LateralDemandStack:
  """Construct a concrete LateralDemandStack instance by id.

  Unknown ids fall back to CustomV2LateralDemandStack.  Use
  resolve_lateral_demand_stack first to get a resolution with
  fallback metadata; this builder is the post-resolution
  construction step.
  """
  if isinstance(stack_id, LateralDemandStackResolution):
    target = stack_id.resolved_stack
  else:
    target = lateral_demand_stack_id_for_name(stack_id)
  for definition in _lateral_demand_stack_definitions():
    if definition.stack_id == target:
      return definition.factory(dt=dt)
  return CustomV2LateralDemandStack(dt=dt)


# ---------------------------------------------------------------------------
# Torque tune (string form)
# ---------------------------------------------------------------------------


class TorqueControlTuneId(str, Enum):
  V20 = "2.0"
  V21 = "2.1"
  V30 = "3.0"
  V40 = "4.0"
  V41 = "4.1"
  V50_EXPERIMENTAL = "5.0"


DEFAULT_TORQUE_CONTROL_TUNE = TorqueControlTuneId.V41
SAFE_TORQUE_TUNE_FALLBACK = TorqueControlTuneId.V41  # 4.1 is the current stable default


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


def resolve_torque_control_tune(value) -> "TorqueControlTuneResolution":
  """Resolve a raw param value to a TorqueControlTuneResolution.
  Unknown / missing values resolve to the safe fallback (4.1),
  never 5.0. 5.0 is only selected when the user explicitly
  asked for it."""
  requested = torque_control_tune_id_for_name(value)
  return TorqueControlTuneResolution(
    requested_tune=requested,
    resolved_tune=requested if requested != TorqueControlTuneId.V50_EXPERIMENTAL or value in (
      TorqueControlTuneId.V50_EXPERIMENTAL.value, "5.0", b"5.0", 5.0,
    ) else SAFE_TORQUE_TUNE_FALLBACK,
    fallback_reason="" if requested != TorqueControlTuneId.V50_EXPERIMENTAL or value in (
      TorqueControlTuneId.V50_EXPERIMENTAL.value, "5.0", b"5.0", 5.0,
    ) else "experimental_requires_explicit_selection",
  )


@dataclass(frozen=True)
class TorqueControlTuneResolution:
  requested_tune: TorqueControlTuneId
  resolved_tune: TorqueControlTuneId
  fallback_reason: str = ""


# ---------------------------------------------------------------------------
# Controls Profile
# ---------------------------------------------------------------------------


class ControlsProfileId(str, Enum):
  """User-facing driving-profile alias. The profile auto-couples
  LateralDemandStack and TorqueControlTune so the user can pick
  one value instead of two. Advanced selectors break the
  coupling when ShowAdvancedControls is on."""
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
  """The full resolution of the controls profile system.

  `requested_profile` is what the user asked for (or
  DEFAULT_CONTROLS_PROFILE if missing).  `resolved_profile` is
  the profile that was actually applied (after the fallback
  pass for missing / unknown values).  `longitudinal_stack`,
  `lateral_demand_stack`, and `torque_control_tune` are the
  per-layer values the resolver chose.  The advanced
  per-layer override params (LateralDemandStack, TorqueControlTune)
  replace the corresponding field when they are explicitly set
  by the user and ShowAdvancedControls is on.  fallback_reason
  is non-empty when any of the resolved fields were changed
  from the requested profile's natural mapping.
  """
  requested_profile: ControlsProfileId
  resolved_profile: ControlsProfileId
  longitudinal_stack: str
  lateral_demand_stack: LateralDemandStackId
  torque_control_tune: TorqueControlTuneId
  fallback_reason: str = ""
  lateral_demand_stack_resolution: Optional[LateralDemandStackResolution] = None
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

  lat_demand_resolution = resolve_lateral_demand_stack(mapping.lateral_demand_stack.value)

  # Apply per-layer advanced overrides when the user opted in.
  effective_lat_demand = lat_demand_resolution.resolved_stack
  fallback_reasons: list[str] = []
  if advanced_overrides_enabled and advanced_lateral_demand_stack is not None:
    override_resolution = resolve_lateral_demand_stack(advanced_lateral_demand_stack)
    effective_lat_demand = override_resolution.resolved_stack
    if override_resolution.resolved_stack != lat_demand_resolution.resolved_stack:
      fallback_reasons.append("advanced_lateral_demand_stack_override")
    lat_demand_resolution = override_resolution

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
    lateral_demand_stack_resolution=lat_demand_resolution,
    torque_control_tune_resolution=torque_resolution,
  )
