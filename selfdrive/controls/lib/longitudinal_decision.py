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


PHYSICAL_CONFIDENCE_MIN = 0.55
ADVISORY_CONFIDENCE_MIN = 0.75
COMFORT_CONFIDENCE_MIN = 0.50


@dataclass(frozen=True)
class LongitudinalDecision:
  enabled: bool
  winner: DecisionSource
  v_target: float
  a_target: float
  should_stop: bool
  candidates: tuple[LongitudinalCandidate, ...] = ()
  suppressed: tuple[tuple[DecisionSource, str], ...] = ()
  fallback_reason: str = ""

  def inside_accel_limits(self, accel_limits: tuple[float, float]) -> bool:
    lo, hi = accel_limits
    return math.isfinite(self.a_target) and lo <= self.a_target <= hi


def _fallback_decision(v_target: float, a_target: float, should_stop: bool, reason: str) -> LongitudinalDecision:
  return LongitudinalDecision(
    enabled=False,
    winner=DecisionSource.LEGACY_FALLBACK,
    v_target=float(v_target),
    a_target=float(a_target),
    should_stop=bool(should_stop),
    fallback_reason=reason,
  )


class LongitudinalArbiter:
  def decide(self, candidates: list[LongitudinalCandidate] | tuple[LongitudinalCandidate, ...]) -> LongitudinalDecision:
    valid = [candidate for candidate in candidates if candidate.valid]
    suppressed: list[tuple[DecisionSource, str]] = [
      (candidate.source, candidate.invalid_reason) for candidate in candidates if not candidate.valid
    ]

    driver = next((candidate for candidate in valid if candidate.role == CandidateRole.DRIVER_INTENT), None)
    if driver is None:
      driver = LongitudinalCandidate(
        source=DecisionSource.LEGACY_FALLBACK,
        role=CandidateRole.FALLBACK,
        v_target=0.0,
        a_target=0.0,
        confidence=1.0,
        urgency=0.0,
        active_reason="missing_driver_intent",
      )

    physical = [
      candidate for candidate in valid
      if candidate.role == CandidateRole.PHYSICAL_HAZARD and candidate.confidence >= PHYSICAL_CONFIDENCE_MIN
    ]
    low_confidence = [
      candidate for candidate in valid
      if candidate.role in (CandidateRole.PHYSICAL_HAZARD, CandidateRole.ADVISORY_CAP)
      and candidate.confidence < (PHYSICAL_CONFIDENCE_MIN if candidate.role == CandidateRole.PHYSICAL_HAZARD else ADVISORY_CONFIDENCE_MIN)
    ]
    suppressed.extend((candidate.source, "low_confidence") for candidate in low_confidence)

    if physical:
      winner = min(physical, key=lambda candidate: (candidate.a_target, candidate.v_target))
      suppressed.extend(
        (candidate.source, "physical_hazard_active") for candidate in valid
        if candidate is not winner and candidate.role != CandidateRole.PHYSICAL_HAZARD
      )
    else:
      advisory = [
        candidate for candidate in valid
        if candidate.role == CandidateRole.ADVISORY_CAP
        and candidate.confidence >= ADVISORY_CONFIDENCE_MIN
        and candidate.v_target < driver.v_target
      ]
      if advisory:
        winner = min(advisory, key=lambda candidate: (candidate.v_target, candidate.a_target))
        suppressed.extend((candidate.source, "higher_advisory_target") for candidate in advisory if candidate is not winner)
      else:
        comfort = [
          candidate for candidate in valid
          if candidate.role == CandidateRole.COMFORT_SHAPING
          and candidate.confidence >= COMFORT_CONFIDENCE_MIN
          and candidate.v_target <= driver.v_target
          and candidate.a_target > driver.a_target
        ]
        winner = max(comfort, key=lambda candidate: candidate.a_target) if comfort else driver

    return LongitudinalDecision(
      enabled=True,
      winner=winner.source,
      v_target=winner.v_target,
      a_target=winner.a_target,
      should_stop=winner.should_stop,
      candidates=tuple(valid),
      suppressed=tuple(dict.fromkeys(suppressed)),
    )


def resolve_longitudinal_decision(enabled: bool, candidates: list[LongitudinalCandidate] | tuple[LongitudinalCandidate, ...],
                                  fallback_v_target: float, fallback_a_target: float, fallback_should_stop: bool,
                                  accel_limits: tuple[float, float], arbiter: LongitudinalArbiter) -> LongitudinalDecision:
  if not enabled:
    return _fallback_decision(fallback_v_target, fallback_a_target, fallback_should_stop, "feature_flag_disabled")
  if not any(candidate.valid and candidate.role == CandidateRole.DRIVER_INTENT for candidate in candidates):
    return _fallback_decision(fallback_v_target, fallback_a_target, fallback_should_stop, "missing_driver_intent")

  try:
    decision = arbiter.decide(candidates)
  except Exception:
    return _fallback_decision(fallback_v_target, fallback_a_target, fallback_should_stop, "arbiter_exception")

  if not math.isfinite(decision.v_target) or not math.isfinite(decision.a_target):
    return _fallback_decision(fallback_v_target, fallback_a_target, fallback_should_stop, "decision_non_finite")
  if decision.winner == DecisionSource.LEGACY_FALLBACK:
    return _fallback_decision(fallback_v_target, fallback_a_target, fallback_should_stop, "missing_driver_intent")
  if not decision.inside_accel_limits(accel_limits):
    return _fallback_decision(fallback_v_target, fallback_a_target, fallback_should_stop, "decision_outside_accel_limits")

  return decision
