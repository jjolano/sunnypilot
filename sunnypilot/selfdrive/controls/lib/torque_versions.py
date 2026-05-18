from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

DEFAULT_TORQUE_TUNE_VERSION = 2.0
REMOVED_TORQUE_TUNE_FALLBACKS = {
  4.0: DEFAULT_TORQUE_TUNE_VERSION,
}


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
  fallback = REMOVED_TORQUE_TUNE_FALLBACKS.get(requested)
  if fallback is not None:
    return TorqueTuneResolution(requested, fallback, persist_value=f"{fallback:.1f}")
  return TorqueTuneResolution(requested, requested)
