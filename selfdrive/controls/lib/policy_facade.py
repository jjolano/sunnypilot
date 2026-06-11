from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from openpilot.selfdrive.controls.lib.feature_registry import (
  ResolvedFeatureSet,
  get_features,
  resolve_feature_set,
)
from openpilot.selfdrive.controls.lib.lateral_demand_stacks.selector import (
  CUSTOM_EXPERIMENTAL as LATERAL_CUSTOM_EXPERIMENTAL,
  CUSTOM_RECOMMENDED as LATERAL_CUSTOM_RECOMMENDED,
  CUSTOM_V2 as LATERAL_CUSTOM_V2,
  SUNNYPILOT_CURRENT as LATERAL_SUNNYPILOT_CURRENT,
)
from openpilot.selfdrive.controls.lib.longitudinal_stacks.selector import (
  CUSTOM_EXPERIMENTAL as LONG_CUSTOM_EXPERIMENTAL,
  CUSTOM_RECOMMENDED as LONG_CUSTOM_RECOMMENDED,
  CUSTOM_V2 as LONG_CUSTOM_V2,
  SUNNYPILOT_CURRENT as LONG_SUNNYPILOT_CURRENT,
  resolve_longitudinal_stack,
  StackResolution as LongitudinalStackResolution,
)
from openpilot.selfdrive.controls.lib.lateral_demand_stacks.selector import (
  resolve_lateral_demand_stack,
  LateralDemandStackResolution,
)


@dataclass(frozen=True)
class ControlsProfile:
  requested_lateral: str
  resolved_lateral: str
  requested_longitudinal: str
  resolved_longitudinal: str
  lateral_fallback_active: bool
  lateral_fallback_reason: str
  longitudinal_fallback_active: bool
  longitudinal_fallback_reason: str
  lateral_features: ResolvedFeatureSet
  longitudinal_features: ResolvedFeatureSet

  @property
  def is_custom(self) -> bool:
    return self.lateral_features.is_custom or self.longitudinal_features.is_custom

  @property
  def is_pure_baseline(self) -> bool:
    return (self.resolved_lateral == LATERAL_SUNNYPILOT_CURRENT and
            self.resolved_longitudinal == LONG_SUNNYPILOT_CURRENT)


def _side_features(stack: str, requested: str, fallback_active: bool, fallback_reason: str) -> ResolvedFeatureSet:
  return resolve_feature_set(
    requested_stack=requested,
    resolved_stack=stack,
    fallback_active=fallback_active,
    fallback_reason=fallback_reason,
  )


def resolve_controls_profile(
    requested_lateral: str,
    requested_longitudinal: str,
    *,
    CP: object | None = None,
    CP_SP: object | None = None,
    manifest: dict[str, Any] | None = None) -> ControlsProfile:
  lateral: LateralDemandStackResolution = resolve_lateral_demand_stack(
    requested_lateral, CP=CP, CP_SP=CP_SP, manifest=manifest,
  )
  longitudinal: LongitudinalStackResolution = resolve_longitudinal_stack(
    requested_longitudinal, CP=CP, CP_SP=CP_SP, manifest=manifest,
  )
  lateral_fallback_active = bool(lateral.fallback_reason) or lateral.resolved_stack != lateral.requested_stack
  longitudinal_fallback_active = bool(longitudinal.fallback_reason) or longitudinal.resolved_stack != longitudinal.requested_stack
  lateral_features = _side_features(
    lateral.resolved_stack, lateral.requested_stack,
    lateral_fallback_active, lateral.fallback_reason,
  )
  longitudinal_features = _side_features(
    longitudinal.resolved_stack, longitudinal.requested_stack,
    longitudinal_fallback_active, longitudinal.fallback_reason,
  )
  return ControlsProfile(
    requested_lateral=lateral.requested_stack,
    resolved_lateral=lateral.resolved_stack,
    requested_longitudinal=longitudinal.requested_stack,
    resolved_longitudinal=longitudinal.resolved_stack,
    lateral_fallback_active=lateral_fallback_active,
    lateral_fallback_reason=lateral.fallback_reason,
    longitudinal_fallback_active=longitudinal_fallback_active,
    longitudinal_fallback_reason=longitudinal.fallback_reason,
    lateral_features=lateral_features,
    longitudinal_features=longitudinal_features,
  )


def _lateral_capabilities_from_car_params(CP: object) -> Any:  # noqa: ARG001  (kept for backwards compat)
  return None


def policy_summary(profile: ControlsProfile) -> dict[str, Any]:
  return {
    "lateral": {
      "requested": profile.requested_lateral,
      "resolved": profile.resolved_lateral,
      "fallback_active": profile.lateral_fallback_active,
      "fallback_reason": profile.lateral_fallback_reason,
      "features": profile.lateral_features.features.__dict__,
    },
    "longitudinal": {
      "requested": profile.requested_longitudinal,
      "resolved": profile.resolved_longitudinal,
      "fallback_active": profile.longitudinal_fallback_active,
      "fallback_reason": profile.longitudinal_fallback_reason,
      "features": profile.longitudinal_features.features.__dict__,
    },
    "is_custom": profile.is_custom,
    "is_pure_baseline": profile.is_pure_baseline,
  }
