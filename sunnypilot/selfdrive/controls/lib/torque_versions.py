from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

DEFAULT_TORQUE_TUNE_VERSION = 4.1
SUPPORTED_TORQUE_TUNE_VERSIONS = frozenset({0.0, 2.0, 2.1, 3.0, 4.0, 4.1, 5.0})
REMOVED_TORQUE_TUNE_FALLBACKS = {}


@dataclass(frozen=True)
class TorqueTuneResolution:
  requested_version: float | None
  resolved_version: float | None
  persist_value: str | None = None


@dataclass(frozen=True)
class TorqueControllerDefinition:
  version: float
  factory: Callable[..., Any]


class TorqueControllerRegistry:
  def __init__(self, definitions: tuple[TorqueControllerDefinition, ...]):
    self._factories = {definition.version: definition.factory for definition in definitions}

  def factory_for(self, version: float | None):
    if version is None:
      return None
    return self._factories.get(version)


def normalize_torque_tune_version(value) -> float | None:
  if value is None:
    return None

  if isinstance(value, (int, float)):
    return float(value)

  if isinstance(value, bytes):
    value = value.decode()

  try:
    return float(value)
  except (TypeError, ValueError):
    return None


def resolve_torque_tune_version(value) -> TorqueTuneResolution:
  requested = normalize_torque_tune_version(value)
  fallback = REMOVED_TORQUE_TUNE_FALLBACKS.get(requested) if requested is not None else None
  if fallback is not None:
    return TorqueTuneResolution(requested, fallback, persist_value=f"{fallback:.1f}")
  if requested not in SUPPORTED_TORQUE_TUNE_VERSIONS:
    return TorqueTuneResolution(requested, DEFAULT_TORQUE_TUNE_VERSION)
  return TorqueTuneResolution(requested, requested)
