from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


@dataclass(frozen=True)
class CurveSpeedEvidence:
  active: bool
  v_target: float
  a_target: float
  distance_to_target: float
  confidence: float
  urgency: float
  source: str
  reason: str
  model_confirmed: bool = False
  model_horizon_distance: float = 0.0

  @property
  def valid(self) -> bool:
    return bool(
      self.active and str(self.source) and str(self.reason) and
      _finite(self.v_target) and self.v_target >= 0.0 and
      _finite(self.a_target) and _finite(self.distance_to_target) and self.distance_to_target >= 0.0
    )


@dataclass(frozen=True)
class CurveSpeedPolicyResult:
  active: bool
  v_target: float
  a_target: float
  distance_to_target: float
  confidence: float
  urgency: float
  source: str
  reason: str


def select_curve_speed_policy(*, mode: str, current_speed: float,
                              vision: CurveSpeedEvidence | None = None,
                              map_advisory: CurveSpeedEvidence | None = None) -> CurveSpeedPolicyResult:
  if str(mode).upper() != "SCC":
    return _inactive("mode_boundary")

  candidates: list[CurveSpeedEvidence] = []
  if vision is not None and vision.valid:
    candidates.append(vision)
  if map_advisory is not None and map_advisory.valid and _map_curve_allowed(map_advisory, current_speed):
    candidates.append(map_advisory)
  if not candidates:
    return _inactive("no_curve_evidence")

  selected = min(candidates, key=lambda evidence: (
    float(evidence.a_target),
    float(evidence.distance_to_target),
    -_clamp01(evidence.confidence),
    -_clamp01(evidence.urgency),
  ))
  return CurveSpeedPolicyResult(
    active=True,
    v_target=float(selected.v_target),
    a_target=float(selected.a_target),
    distance_to_target=float(selected.distance_to_target),
    confidence=_clamp01(selected.confidence),
    urgency=_clamp01(selected.urgency),
    source=str(selected.source),
    reason=str(selected.reason),
  )


def _map_curve_allowed(evidence: CurveSpeedEvidence, current_speed: float) -> bool:
  current_speed = float(current_speed) if _finite(current_speed) else 0.0
  large_slowdown = current_speed - float(evidence.v_target) > 5.0
  model_horizon_covers = evidence.model_horizon_distance >= evidence.distance_to_target > 0.0
  return bool(evidence.model_confirmed or not (large_slowdown and model_horizon_covers))


def _inactive(reason: str) -> CurveSpeedPolicyResult:
  return CurveSpeedPolicyResult(
    active=False,
    v_target=0.0,
    a_target=0.0,
    distance_to_target=0.0,
    confidence=0.0,
    urgency=0.0,
    source="",
    reason=reason,
  )


def _finite(value: Any) -> bool:
  try:
    value = float(value)
  except (TypeError, ValueError):
    return False
  return math.isfinite(value)


def _clamp01(value: Any) -> float:
  try:
    value = float(value)
  except (TypeError, ValueError):
    return 0.0
  return min(1.0, max(0.0, value)) if math.isfinite(value) else 0.0
