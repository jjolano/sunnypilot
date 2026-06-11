from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class StackFeatures:
  one_pedal_longitudinal: bool = False
  lead_confirmed_progress: bool = False
  lane_change_path_shaping: bool = False
  lane_centering_assist: bool = False
  straight_road_damping: bool = False
  lateral_oscillation_supervisor: bool = False
  lateral_turn_exit_controller: bool = False
  experimental_stage: str = "stable"


@dataclass(frozen=True)
class FeatureRegistryEntry:
  stack: str
  family: str
  features: StackFeatures


@dataclass(frozen=True)
class ResolvedFeatureSet:
  requested_stack: str
  resolved_stack: str
  features: StackFeatures
  fallback_active: bool
  fallback_reason: str

  @property
  def is_custom(self) -> bool:
    return str(self.resolved_stack or "").startswith("custom-")


_REGISTRY: dict[str, FeatureRegistryEntry] = {}


def register_features(entry: FeatureRegistryEntry) -> None:
  _REGISTRY[entry.stack] = entry


def get_features(stack: str) -> StackFeatures:
  entry = _REGISTRY.get(str(stack or ""))
  if entry is None:
    return StackFeatures()
  return entry.features


def known_stacks() -> tuple[str, ...]:
  return tuple(sorted(_REGISTRY))


def _entry(stack: str) -> FeatureRegistryEntry | None:
  return _REGISTRY.get(str(stack or ""))


def resolve_feature_set(
    requested_stack: str,
    resolved_stack: str,
    *,
    fallback_active: bool = False,
    fallback_reason: str = "",
) -> ResolvedFeatureSet:
  features = get_features(resolved_stack)
  return ResolvedFeatureSet(
    requested_stack=str(requested_stack or ""),
    resolved_stack=str(resolved_stack or ""),
    features=features,
    fallback_active=bool(fallback_active),
    fallback_reason=str(fallback_reason or ""),
  )


def merge_features(*stacks: str) -> StackFeatures:
  """Return the union of features supported by any of the named stacks."""
  if not stacks:
    return StackFeatures()
  base = StackFeatures(
    one_pedal_longitudinal=False,
    lead_confirmed_progress=False,
    lane_change_path_shaping=False,
    lane_centering_assist=False,
    straight_road_damping=False,
    lateral_oscillation_supervisor=False,
    lateral_turn_exit_controller=False,
    experimental_stage="stable",
  )
  out = StackFeatures(**{**base.__dict__, **{k: False for k in base.__dict__ if isinstance(base.__dict__[k], bool)}})
  for stack in stacks:
    f = get_features(stack)
    merged = {**out.__dict__}
    for k, v in f.__dict__.items():
      if isinstance(v, bool):
        merged[k] = bool(out.__dict__[k]) or bool(v)
      else:
        merged[k] = v
    out = StackFeatures(**merged)
  return out


def feature_registry_snapshot() -> dict[str, dict[str, Any]]:
  return {
    stack: {
      "family": entry.family,
      "features": {k: v for k, v in entry.features.__dict__.items()},
    }
    for stack, entry in sorted(_REGISTRY.items())
  }
