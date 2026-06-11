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

from openpilot.selfdrive.controls.lib.lateral_demand import ProcessedLateralDemand
from openpilot.selfdrive.controls.lib.lateral_demand_profile import (
  LateralDemandProfile,
  LateralDemandProfileBuilder,
)


class LateralDemandStackId(str, Enum):
  SUNNYPILOT_CURRENT = "sunnypilot-current"
  CUSTOM_V2 = "custom-2.0"
  CUSTOM_EXPERIMENTAL = "custom-experimental"


DEFAULT_LATERAL_DEMAND_STACK = LateralDemandStackId.SUNNYPILOT_CURRENT


class ControlsProfileId(str, Enum):
  """User-facing driving-profile alias. The profile auto-couples
  LateralDemandStack and TorqueControlTune so the user can pick
  one value instead of two. Advanced selectors break the
  coupling."""
  STANDARD = "standard"
  CUSTOM_2 = "custom-2.0"
  EXPERIMENTAL = "experimental"


DEFAULT_CONTROLS_PROFILE = ControlsProfileId.STANDARD


@dataclass(frozen=True)
class ControlsProfileMapping:
  profile_id: ControlsProfileId
  lateral_demand_stack: LateralDemandStackId
  torque_tune: float


CONTROLS_PROFILE_MAPPINGS: tuple[ControlsProfileMapping, ...] = (
  ControlsProfileMapping(
    ControlsProfileId.STANDARD,
    LateralDemandStackId.SUNNYPILOT_CURRENT, 4.1,
  ),
  ControlsProfileMapping(
    ControlsProfileId.CUSTOM_2,
    LateralDemandStackId.CUSTOM_V2, 4.1,
  ),
  ControlsProfileMapping(
    ControlsProfileId.EXPERIMENTAL,
    LateralDemandStackId.CUSTOM_EXPERIMENTAL, 5.0,
  ),
)


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


def resolve_lateral_demand_stack(
  value, *, dt: float = 0.05,
) -> LateralDemandStack:
  """Resolve a raw param value to a concrete LateralDemandStack
  instance.  Unknown / missing values resolve to the
  SunnypilotCurrent migration default."""
  stack_id = lateral_demand_stack_id_for_name(value)
  for definition in _lateral_demand_stack_definitions():
    if definition.stack_id == stack_id:
      return definition.factory(dt=dt)
  return SunnypilotCurrentLateralDemandStack(dt=dt)
