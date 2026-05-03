from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any


class CandidateRole(Enum):
  DRIVER_INTENT = "driver_intent"
  PHYSICAL_HAZARD = "physical_hazard"
  ADVISORY_CAP = "advisory_cap"
  COMFORT_SHAPING = "comfort_shaping"
  FALLBACK = "fallback"


class DecisionSource(Enum):
  CRUISE = "cruise"
  LEAD_MPC = "lead_mpc"
  E2E_STOP = "e2e_stop"
  SPEED_LIMIT = "speed_limit"
  SCC_VISION = "scc_vision"
  SCC_MAP = "scc_map"
  OSM_TRAFFIC_CONTROL = "osm_traffic_control"
  CRUISE_COAST = "cruise_coast"
  STOP_LAUNCH = "stop_launch"
  LEGACY_FALLBACK = "legacy_fallback"


def _clamp01(value: float) -> float:
  if not math.isfinite(value):
    return 0.0
  return min(1.0, max(0.0, float(value)))


@dataclass(frozen=True)
class LongitudinalCandidate:
  source: DecisionSource
  role: CandidateRole
  v_target: float
  a_target: float
  confidence: float
  urgency: float
  active_reason: str
  should_stop: bool = False
  comfort_bounds: tuple[float | None, float | None] = (None, None)
  safety_bounds: tuple[float | None, float | None] = (None, None)
  debug: dict[str, Any] = field(default_factory=dict)

  def __post_init__(self) -> None:
    object.__setattr__(self, "v_target", float(self.v_target))
    object.__setattr__(self, "a_target", float(self.a_target))
    object.__setattr__(self, "confidence", _clamp01(float(self.confidence)))
    object.__setattr__(self, "urgency", _clamp01(float(self.urgency)))
    object.__setattr__(self, "active_reason", str(self.active_reason))

  @property
  def invalid_reason(self) -> str:
    if not math.isfinite(self.v_target) or not math.isfinite(self.a_target):
      return "non_finite_target"
    if self.v_target < 0.0:
      return "negative_v_target"
    if not self.active_reason:
      return "missing_active_reason"
    return ""

  @property
  def valid(self) -> bool:
    return self.invalid_reason == ""
