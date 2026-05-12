from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any

from cereal import log


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
COMFORT_MAX_DRIVER_ACCEL_MARGIN = 0.5


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
      return LongitudinalDecision(
        enabled=True,
        winner=driver.source,
        v_target=driver.v_target,
        a_target=driver.a_target,
        should_stop=driver.should_stop,
        candidates=tuple(valid),
        suppressed=tuple(dict.fromkeys(suppressed)),
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

    if winner is not driver and winner.role == CandidateRole.COMFORT_SHAPING:
      max_allowed = driver.a_target + COMFORT_MAX_DRIVER_ACCEL_MARGIN
      if winner.a_target > max_allowed:
        suppressed.append((winner.source, "comfort_accel_exceeds_margin"))
        winner = driver

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


def get_active_lead_confidence(*leads: Any) -> float:
  return max(
    (float(getattr(lead, "modelProb", 1.0)) for lead in leads if getattr(lead, "status", False)),
    default=0.0,
  )


def apply_longitudinal_decision_output(decision: LongitudinalDecision, legacy_a_target: float,
                                       legacy_should_stop: bool, prev_a_target: float | None = None,
                                       personality: int = log.LongitudinalPersonality.standard,
                                       dt: float = 0.0, comfort_active: bool = True) -> tuple[float, bool]:
  legacy_should_stop = bool(legacy_should_stop)
  if not decision.enabled or decision.winner in (DecisionSource.LEGACY_FALLBACK, DecisionSource.CRUISE):
    output_a_target = float(legacy_a_target)
    output_should_stop = legacy_should_stop
  elif decision.winner in (DecisionSource.LEAD_MPC, DecisionSource.E2E_STOP, DecisionSource.STOP_LAUNCH):
    output_a_target = float(legacy_a_target)
    output_should_stop = legacy_should_stop or decision.should_stop
  elif decision.winner in (
    DecisionSource.SPEED_LIMIT,
    DecisionSource.SCC_VISION,
    DecisionSource.SCC_MAP,
    DecisionSource.OSM_TRAFFIC_CONTROL,
  ):
    output_a_target = min(float(decision.a_target), float(legacy_a_target))
    output_should_stop = legacy_should_stop or decision.should_stop
  elif decision.winner == DecisionSource.CRUISE_COAST:
    output_a_target = float(decision.a_target)
    output_should_stop = legacy_should_stop or decision.should_stop
  else:
    output_a_target = float(legacy_a_target)
    output_should_stop = legacy_should_stop

  if comfort_active and not output_should_stop and prev_a_target is not None:
    output_a_target = apply_personality_accel_comfort(decision, output_a_target, prev_a_target, personality, dt)
  return output_a_target, output_should_stop


ACCEL_COMFORT_POSITIVE_RATE = {
  log.LongitudinalPersonality.relaxed: 1.2,
  log.LongitudinalPersonality.standard: 2.0,
  log.LongitudinalPersonality.aggressive: 4.0,
}
ACCEL_COMFORT_MILD_DECEL_RATE = {
  log.LongitudinalPersonality.relaxed: 1.6,
  log.LongitudinalPersonality.standard: 2.4,
  log.LongitudinalPersonality.aggressive: 4.0,
}
ACCEL_COMFORT_URGENT_DECEL_DELTA = 0.6
ACCEL_COMFORT_URGENT_DECEL_TARGET = -0.7
ACCEL_COMFORT_BYPASS_SOURCES = frozenset((
  DecisionSource.CRUISE,
  DecisionSource.LEAD_MPC,
  DecisionSource.E2E_STOP,
  DecisionSource.STOP_LAUNCH,
  DecisionSource.LEGACY_FALLBACK,
))


def _personality_rate(rates: dict[log.LongitudinalPersonality, float], personality: int) -> float:
  return rates.get(personality, rates[log.LongitudinalPersonality.standard])


def apply_personality_accel_comfort(decision: LongitudinalDecision, a_target: float, prev_a_target: float,
                                   personality: int, dt: float) -> float:
  a_target = float(a_target)
  prev_a_target = float(prev_a_target)
  if (
    not decision.enabled
    or decision.should_stop
    or not math.isfinite(a_target)
    or not math.isfinite(prev_a_target)
    or dt <= 0.0
  ):
    return a_target

  delta = a_target - prev_a_target
  if delta >= 0.0:
    max_delta = _personality_rate(ACCEL_COMFORT_POSITIVE_RATE, personality) * dt
    return min(a_target, prev_a_target + max_delta)

  if delta <= -ACCEL_COMFORT_URGENT_DECEL_DELTA or a_target <= ACCEL_COMFORT_URGENT_DECEL_TARGET:
    return a_target

  max_delta = _personality_rate(ACCEL_COMFORT_MILD_DECEL_RATE, personality) * dt
  return max(a_target, prev_a_target - max_delta)


def build_core_longitudinal_candidates(has_lead: bool, lead_confidence: float, v_cruise: float, a_cruise: float,
                                       output_a_target_mpc: float, output_should_stop_mpc: bool,
                                       e2e_active: bool, output_a_target_e2e: float, output_should_stop_e2e: bool,
                                       e2e_stop_approach_a_target: float,
                                       cruise_coast_applied: bool, cruise_coast_a_target: float) -> list[LongitudinalCandidate]:
  candidates: list[LongitudinalCandidate] = []

  if has_lead:
    candidates.append(LongitudinalCandidate(
      source=DecisionSource.LEAD_MPC,
      role=CandidateRole.PHYSICAL_HAZARD,
      v_target=max(0.0, v_cruise),
      a_target=output_a_target_mpc,
      confidence=lead_confidence,
      urgency=0.70 if output_a_target_mpc < -0.3 or output_should_stop_mpc else 0.45,
      active_reason="confirmed_radar_lead",
      should_stop=output_should_stop_mpc,
    ))

  if e2e_active or output_should_stop_e2e or e2e_stop_approach_a_target < 0.0:
    e2e_accel = min(output_a_target_e2e, e2e_stop_approach_a_target) if e2e_stop_approach_a_target < 0.0 else output_a_target_e2e
    candidates.append(LongitudinalCandidate(
      source=DecisionSource.E2E_STOP,
      role=CandidateRole.PHYSICAL_HAZARD,
      v_target=max(0.0, v_cruise),
      a_target=e2e_accel,
      confidence=0.85 if output_should_stop_e2e else 0.65,
      urgency=0.80 if output_should_stop_e2e else 0.55,
      active_reason="model_stop_or_slowdown",
      should_stop=output_should_stop_e2e,
    ))

  if cruise_coast_applied:
    candidates.append(LongitudinalCandidate(
      source=DecisionSource.CRUISE_COAST,
      role=CandidateRole.COMFORT_SHAPING,
      v_target=max(0.0, v_cruise),
      a_target=cruise_coast_a_target,
      confidence=0.80,
      urgency=0.20,
      active_reason="context_efficient_overspeed_coast",
      debug={"legacy_cruise_accel": float(a_cruise)},
    ))

  return candidates
