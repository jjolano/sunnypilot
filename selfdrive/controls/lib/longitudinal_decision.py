from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from numbers import Real
from typing import Any

from cereal import log


class CandidateRole(Enum):
  DRIVER_INTENT = "driver_intent"
  PHYSICAL_HAZARD = "physical_hazard"
  ADVISORY_CAP = "advisory_cap"
  RELAXATION = "relaxation"
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


DECISION_SOURCE_PRIORITY: dict[DecisionSource, int] = {
  DecisionSource.CRUISE: 0,
  DecisionSource.LEAD_MPC: 10,
  DecisionSource.E2E_STOP: 20,
  DecisionSource.OSM_TRAFFIC_CONTROL: 30,
  DecisionSource.SPEED_LIMIT: 40,
  DecisionSource.SCC_VISION: 50,
  DecisionSource.SCC_MAP: 60,
  DecisionSource.CRUISE_COAST: 70,
  DecisionSource.STOP_LAUNCH: 80,
  DecisionSource.LEGACY_FALLBACK: 90,
}


def _source_priority(source: DecisionSource) -> int:
  return DECISION_SOURCE_PRIORITY[source]


def _bounds_invalid_reason(name: str, bounds: object) -> str:
  if not isinstance(bounds, tuple) or len(bounds) != 2:
    return f"invalid_{name}_bounds"

  lower, upper = bounds
  for value in (lower, upper):
    if value is None:
      continue
    if not isinstance(value, Real):
      return f"invalid_{name}_bounds"
    if not math.isfinite(float(value)):
      return f"non_finite_{name}_bounds"

  if lower is not None and upper is not None and float(lower) > float(upper):
    return f"inverted_{name}_bounds"
  return ""


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
    if not isinstance(self.source, DecisionSource):
      return "invalid_source"
    if not isinstance(self.role, CandidateRole):
      return "invalid_role"
    if not math.isfinite(self.v_target) or not math.isfinite(self.a_target):
      return "non_finite_target"
    if self.v_target < 0.0:
      return "negative_v_target"
    if not self.active_reason:
      return "missing_active_reason"
    for name, bounds in (("comfort", self.comfort_bounds), ("safety", self.safety_bounds)):
      bounds_reason = _bounds_invalid_reason(name, bounds)
      if bounds_reason:
        return bounds_reason
    return ""

  @property
  def valid(self) -> bool:
    return self.invalid_reason == ""


@dataclass(frozen=True)
class SuppressedLongitudinalCandidate:
  source: DecisionSource
  role: CandidateRole
  active_reason: str
  suppression_reason: str
  debug: dict[str, Any] = field(default_factory=dict)
  v_target: float = 0.0
  a_target: float = 0.0
  should_stop: bool = False


PHYSICAL_CONFIDENCE_MIN = 0.55
ADVISORY_CONFIDENCE_MIN = 0.75
COMFORT_CONFIDENCE_MIN = 0.50
COMFORT_MAX_DRIVER_ACCEL_MARGIN = 0.5
SOURCE_STABILITY_MAX_V_EGO = 2.5
SOURCE_STABILITY_RELEASE_FRAMES = 3
SOURCE_STABILITY_ACCEL_EPS = 1e-3
SOURCE_STABILITY_SPEED_EPS = 1e-3
SOURCE_STABILITY_HOLD_REASON = "source_stability_hold"
SOURCE_STABILITY_HOLD_SOURCES = frozenset((
  DecisionSource.LEAD_MPC,
  DecisionSource.E2E_STOP,
  DecisionSource.SPEED_LIMIT,
  DecisionSource.SCC_VISION,
  DecisionSource.SCC_MAP,
  DecisionSource.OSM_TRAFFIC_CONTROL,
  DecisionSource.STOP_LAUNCH,
))


@dataclass(frozen=True)
class LongitudinalDecision:
  enabled: bool
  winner: DecisionSource
  v_target: float
  a_target: float
  should_stop: bool
  candidates: tuple[LongitudinalCandidate, ...] = ()
  suppressed: tuple[tuple[DecisionSource, str], ...] = ()
  suppressed_candidates: tuple[SuppressedLongitudinalCandidate, ...] = ()
  fallback_reason: str = ""
  active_reason: str = ""

  def inside_accel_limits(self, accel_limits: tuple[float, float]) -> bool:
    lo, hi = accel_limits
    return math.isfinite(self.a_target) and lo <= self.a_target <= hi


@dataclass(frozen=True)
class LongitudinalDecisionTelemetry:
  raw_source: DecisionSource
  raw_v_target: float
  raw_a_target: float
  raw_should_stop: bool
  raw_active_reason: str
  legacy_a_target: float
  legacy_should_stop: bool
  applied_a_target: float
  applied_should_stop: bool
  applied_reason: str
  accel_delta: float


@dataclass(frozen=True)
class LongitudinalDecisionOutput:
  a_target: float
  should_stop: bool
  applied_reason: str
  held_by_source_stability: bool = False


def _fallback_decision(v_target: float, a_target: float, should_stop: bool, reason: str) -> LongitudinalDecision:
  return LongitudinalDecision(
    enabled=False,
    winner=DecisionSource.LEGACY_FALLBACK,
    v_target=float(v_target),
    a_target=float(a_target),
    should_stop=bool(should_stop),
    fallback_reason=reason,
    active_reason=reason,
  )


def _internal_contract_fallback(valid: list[LongitudinalCandidate], suppressed: list[tuple[DecisionSource, str]],
                                reason: str,
                                suppressed_candidates: list[SuppressedLongitudinalCandidate] | None = None) -> LongitudinalDecision:
  return LongitudinalDecision(
    enabled=True,
    winner=DecisionSource.LEGACY_FALLBACK,
    v_target=0.0,
    a_target=0.0,
    should_stop=False,
    candidates=tuple(valid),
    suppressed=tuple(dict.fromkeys(suppressed)),
    suppressed_candidates=_dedupe_suppressed_candidates(suppressed_candidates or []),
    fallback_reason=reason,
    active_reason=reason,
  )


def _driver_intent_contract_reason(candidates: list[LongitudinalCandidate] | tuple[LongitudinalCandidate, ...]) -> str:
  valid_drivers = [
    candidate for candidate in candidates
    if candidate.valid and candidate.role == CandidateRole.DRIVER_INTENT
  ]
  if not valid_drivers:
    return "missing_driver_intent"
  if len(valid_drivers) > 1:
    return "duplicate_driver_intent"
  return ""


def _suppressed_candidate_safe(candidate: LongitudinalCandidate) -> bool:
  return bool(
    isinstance(candidate.source, DecisionSource) and
    isinstance(candidate.role, CandidateRole) and
    isinstance(candidate.active_reason, str) and candidate.active_reason
  )


def _suppressed_candidate(candidate: LongitudinalCandidate, reason: str) -> SuppressedLongitudinalCandidate | None:
  if not _suppressed_candidate_safe(candidate):
    return None
  return SuppressedLongitudinalCandidate(
    source=candidate.source,
    role=candidate.role,
    active_reason=candidate.active_reason,
    suppression_reason=str(reason),
    debug=dict(candidate.debug),
    v_target=float(candidate.v_target),
    a_target=float(candidate.a_target),
    should_stop=bool(candidate.should_stop),
  )


def _append_suppression(suppressed: list[tuple[DecisionSource, str]],
                        suppressed_candidates: list[SuppressedLongitudinalCandidate],
                        candidate: LongitudinalCandidate, reason: str) -> None:
  suppressed.append((candidate.source, str(reason)))
  rich = _suppressed_candidate(candidate, reason)
  if rich is not None:
    suppressed_candidates.append(rich)


def _append_suppressions(suppressed: list[tuple[DecisionSource, str]],
                         suppressed_candidates: list[SuppressedLongitudinalCandidate],
                         candidates, reason: str) -> None:
  for candidate in candidates:
    _append_suppression(suppressed, suppressed_candidates, candidate, reason)


def _suppressed_candidate_key(candidate: SuppressedLongitudinalCandidate) -> tuple[object, ...]:
  return (
    candidate.source,
    candidate.role,
    candidate.active_reason,
    candidate.suppression_reason,
    candidate.v_target,
    candidate.a_target,
    candidate.should_stop,
  )


def _dedupe_suppressed_candidates(candidates: list[SuppressedLongitudinalCandidate] |
                                  tuple[SuppressedLongitudinalCandidate, ...]) -> tuple[SuppressedLongitudinalCandidate, ...]:
  deduped: list[SuppressedLongitudinalCandidate] = []
  seen: set[tuple[object, ...]] = set()
  for candidate in candidates:
    key = _suppressed_candidate_key(candidate)
    if key in seen:
      continue
    seen.add(key)
    deduped.append(candidate)
  return tuple(deduped)


def _candidate_identity(candidate: LongitudinalCandidate) -> tuple[object, ...]:
  return (
    candidate.source,
    candidate.role,
    candidate.active_reason,
    candidate.v_target,
    candidate.a_target,
    candidate.should_stop,
  )


def _prepend_candidate(candidate: LongitudinalCandidate | None,
                       candidates: tuple[LongitudinalCandidate, ...]) -> tuple[LongitudinalCandidate, ...]:
  if candidate is None:
    return candidates
  held_key = _candidate_identity(candidate)
  return (candidate, *(current for current in candidates if _candidate_identity(current) != held_key))


def _selected_candidate_for_decision(decision: LongitudinalDecision) -> LongitudinalCandidate | None:
  for candidate in decision.candidates:
    if (
      candidate.source == decision.winner and
      candidate.active_reason == decision.active_reason and
      math.isclose(candidate.a_target, decision.a_target, abs_tol=1e-6)
    ):
      return candidate
  for candidate in decision.candidates:
    if candidate.source == decision.winner and candidate.active_reason == decision.active_reason:
      return candidate
  for candidate in decision.candidates:
    if candidate.source == decision.winner:
      return candidate
  return None


class LongitudinalDecisionCore:
  def decide(self, candidates: list[LongitudinalCandidate] | tuple[LongitudinalCandidate, ...]) -> LongitudinalDecision:
    valid = [candidate for candidate in candidates if candidate.valid]
    suppressed: list[tuple[DecisionSource, str]] = []
    suppressed_candidates: list[SuppressedLongitudinalCandidate] = []
    for candidate in candidates:
      if candidate.valid:
        continue
      _append_suppression(suppressed, suppressed_candidates, candidate, candidate.invalid_reason)

    drivers = [candidate for candidate in valid if candidate.role == CandidateRole.DRIVER_INTENT]
    driver = min(
      drivers,
      key=lambda candidate: (_source_priority(candidate.source), candidate.v_target, candidate.a_target),
    ) if drivers else None
    if driver is None:
      return _internal_contract_fallback(valid, suppressed, "missing_driver_intent", suppressed_candidates)
    if len(drivers) > 1:
      _append_suppressions(suppressed, suppressed_candidates, drivers, "duplicate_driver_intent")
      return _internal_contract_fallback(valid, suppressed, "duplicate_driver_intent", suppressed_candidates)

    physical = [
      candidate for candidate in valid
      if candidate.role == CandidateRole.PHYSICAL_HAZARD and candidate.confidence >= PHYSICAL_CONFIDENCE_MIN and
      _physical_candidate_active(candidate, driver)
    ]
    low_confidence = [
      candidate for candidate in valid
      if candidate.role in (CandidateRole.PHYSICAL_HAZARD, CandidateRole.ADVISORY_CAP)
      and candidate.confidence < (PHYSICAL_CONFIDENCE_MIN if candidate.role == CandidateRole.PHYSICAL_HAZARD else ADVISORY_CONFIDENCE_MIN)
    ]
    _append_suppressions(suppressed, suppressed_candidates, low_confidence, "low_confidence")

    if physical:
      winner = min(physical, key=lambda candidate: (
        candidate.a_target,
        candidate.v_target,
        -candidate.confidence,
        -candidate.urgency,
        _source_priority(candidate.source),
        candidate.active_reason,
      ))
      _append_suppressions(
        suppressed, suppressed_candidates,
        (candidate for candidate in valid if candidate is not winner and candidate.role != CandidateRole.PHYSICAL_HAZARD),
        "physical_hazard_active",
      )
    else:
      advisory = [
        candidate for candidate in valid
        if candidate.role == CandidateRole.ADVISORY_CAP
        and candidate.confidence >= ADVISORY_CONFIDENCE_MIN
        and candidate.v_target < driver.v_target
      ]
      if advisory:
        winner = min(advisory, key=lambda candidate: (
          candidate.v_target,
          candidate.a_target,
          -candidate.confidence,
          -candidate.urgency,
          _source_priority(candidate.source),
          candidate.active_reason,
        ))
        _append_suppressions(
          suppressed, suppressed_candidates,
          (candidate for candidate in advisory if candidate is not winner),
          "higher_advisory_target",
        )
      else:
        comfort = [
          candidate for candidate in valid
          if candidate.role == CandidateRole.RELAXATION
          and candidate.confidence >= COMFORT_CONFIDENCE_MIN
          and candidate.v_target <= driver.v_target
          and candidate.a_target > driver.a_target
        ]
        winner = min(comfort, key=lambda candidate: (
          -candidate.a_target,
          candidate.v_target,
          -candidate.confidence,
          -candidate.urgency,
          _source_priority(candidate.source),
          candidate.active_reason,
        )) if comfort else driver

    if (
      winner is not driver and winner.role == CandidateRole.RELAXATION and winner.a_target > 0.0 and
      winner.source != DecisionSource.STOP_LAUNCH
    ):
      max_allowed = driver.a_target + COMFORT_MAX_DRIVER_ACCEL_MARGIN
      if winner.a_target > max_allowed:
        _append_suppression(suppressed, suppressed_candidates, winner, "comfort_accel_exceeds_margin")
        winner = driver

    return LongitudinalDecision(
      enabled=True,
      winner=winner.source,
      v_target=winner.v_target,
      a_target=winner.a_target,
      should_stop=winner.should_stop,
      candidates=tuple(valid),
      suppressed=tuple(dict.fromkeys(suppressed)),
      suppressed_candidates=_dedupe_suppressed_candidates(suppressed_candidates),
      active_reason=winner.active_reason,
    )


class SourceStabilityHold:
  def __init__(self) -> None:
    self._source_stability_decision: LongitudinalDecision | None = None
    self._source_stability_candidate: LongitudinalCandidate | None = None
    self._source_stability_release_frames: int = 0

  def reset(self) -> None:
    self._source_stability_decision = None
    self._source_stability_candidate = None
    self._source_stability_release_frames = 0

  def apply(self, decision: LongitudinalDecision, v_ego: float | None) -> LongitudinalDecision:
    if not _source_stability_active(v_ego):
      self.reset()
      return decision

    previous = self._source_stability_decision
    if previous is None or not previous.enabled or previous.winner == DecisionSource.LEGACY_FALLBACK:
      self._record_source_stability_decision(decision)
      return decision
    if decision.winner == previous.winner or previous.winner not in SOURCE_STABILITY_HOLD_SOURCES:
      self._record_source_stability_decision(decision)
      return decision
    if _decision_more_restrictive(decision, previous):
      self._record_source_stability_decision(decision)
      return decision
    if self._source_stability_release_frames <= 0:
      self._record_source_stability_decision(decision)
      return decision

    self._source_stability_release_frames -= 1
    held_suppressed = tuple(dict.fromkeys((*decision.suppressed, (decision.winner, SOURCE_STABILITY_HOLD_REASON))))
    current_candidate = _selected_candidate_for_decision(decision)
    held_suppressed_candidates = list(decision.suppressed_candidates)
    if current_candidate is not None:
      rich = _suppressed_candidate(current_candidate, SOURCE_STABILITY_HOLD_REASON)
      if rich is not None:
        held_suppressed_candidates.append(rich)
    held_candidate = self._source_stability_candidate or _selected_candidate_for_decision(previous)
    held = LongitudinalDecision(
      enabled=True,
      winner=previous.winner,
      v_target=previous.v_target,
      a_target=previous.a_target,
      should_stop=previous.should_stop,
      candidates=_prepend_candidate(held_candidate, decision.candidates),
      suppressed=held_suppressed,
      suppressed_candidates=_dedupe_suppressed_candidates(held_suppressed_candidates),
      active_reason=previous.active_reason,
    )
    self._source_stability_decision = held
    self._source_stability_candidate = held_candidate
    return held

  def _record_source_stability_decision(self, decision: LongitudinalDecision) -> None:
    self._source_stability_decision = decision
    self._source_stability_candidate = _selected_candidate_for_decision(decision)
    self._source_stability_release_frames = SOURCE_STABILITY_RELEASE_FRAMES


class LongitudinalArbiter:
  def __init__(self) -> None:
    self.core = LongitudinalDecisionCore()
    self.source_stability = SourceStabilityHold()

  def decide(self, candidates: list[LongitudinalCandidate] | tuple[LongitudinalCandidate, ...]) -> LongitudinalDecision:
    return self.core.decide(candidates)

  def reset_source_stability(self) -> None:
    self.source_stability.reset()

  def apply_source_stability(self, decision: LongitudinalDecision, v_ego: float | None) -> LongitudinalDecision:
    return self.source_stability.apply(decision, v_ego)


def _source_stability_active(v_ego: float | None) -> bool:
  return v_ego is not None and math.isfinite(v_ego) and v_ego <= SOURCE_STABILITY_MAX_V_EGO


def _decision_more_restrictive(decision: LongitudinalDecision, previous: LongitudinalDecision) -> bool:
  return (
    (decision.should_stop and not previous.should_stop) or
    decision.a_target < previous.a_target - SOURCE_STABILITY_ACCEL_EPS or
    decision.v_target < previous.v_target - SOURCE_STABILITY_SPEED_EPS
  )


def _physical_candidate_active(candidate: LongitudinalCandidate, driver: LongitudinalCandidate) -> bool:
  return bool(
    candidate.source == DecisionSource.STOP_LAUNCH or
    (candidate.should_stop and not driver.should_stop) or
    candidate.a_target < driver.a_target - SOURCE_STABILITY_ACCEL_EPS
  )


def _decision_held_by_source_stability(decision: LongitudinalDecision) -> bool:
  return any(reason == SOURCE_STABILITY_HOLD_REASON for _, reason in decision.suppressed)


def _decision_raw_reason(decision: LongitudinalDecision) -> str:
  return decision.active_reason or decision.fallback_reason or decision.winner.value


def resolve_longitudinal_decision(enabled: bool, candidates: list[LongitudinalCandidate] | tuple[LongitudinalCandidate, ...],
                                  fallback_v_target: float, fallback_a_target: float, fallback_should_stop: bool,
                                  accel_limits: tuple[float, float], arbiter: LongitudinalArbiter,
                                  v_ego: float | None = None) -> LongitudinalDecision:
  if not enabled:
    arbiter.reset_source_stability()
    return _fallback_decision(fallback_v_target, fallback_a_target, fallback_should_stop, "feature_flag_disabled")
  contract_reason = _driver_intent_contract_reason(candidates)
  if contract_reason:
    arbiter.reset_source_stability()
    return _fallback_decision(fallback_v_target, fallback_a_target, fallback_should_stop, contract_reason)

  try:
    decision = arbiter.decide(candidates)
  except Exception:
    arbiter.reset_source_stability()
    return _fallback_decision(fallback_v_target, fallback_a_target, fallback_should_stop, "arbiter_exception")

  if not math.isfinite(decision.v_target) or not math.isfinite(decision.a_target):
    arbiter.reset_source_stability()
    return _fallback_decision(fallback_v_target, fallback_a_target, fallback_should_stop, "decision_non_finite")
  if decision.winner == DecisionSource.LEGACY_FALLBACK:
    arbiter.reset_source_stability()
    contract_fallback_reason = decision.fallback_reason or decision.active_reason or "missing_driver_intent"
    return _fallback_decision(fallback_v_target, fallback_a_target, fallback_should_stop, contract_fallback_reason)
  if not decision.inside_accel_limits(accel_limits):
    arbiter.reset_source_stability()
    return _fallback_decision(fallback_v_target, fallback_a_target, fallback_should_stop, "decision_outside_accel_limits")

  decision = arbiter.apply_source_stability(decision, v_ego)
  if not decision.inside_accel_limits(accel_limits):
    arbiter.reset_source_stability()
    return _fallback_decision(fallback_v_target, fallback_a_target, fallback_should_stop, "decision_outside_accel_limits")

  return decision


def get_active_lead_confidence(*leads: Any) -> float:
  return max(
    (float(getattr(lead, "modelProb", 1.0)) for lead in leads if getattr(lead, "status", False)),
    default=0.0,
  )


def apply_decision_source_output(decision: LongitudinalDecision, legacy_a_target: float,
                                 legacy_should_stop: bool) -> LongitudinalDecisionOutput:
  held_by_source_stability = _decision_held_by_source_stability(decision)
  if not decision.enabled or decision.winner == DecisionSource.LEGACY_FALLBACK:
    return LongitudinalDecisionOutput(
      legacy_a_target,
      legacy_should_stop,
      decision.fallback_reason or "legacy_fallback",
    )
  if held_by_source_stability:
    return LongitudinalDecisionOutput(
      min(float(decision.a_target), legacy_a_target),
      legacy_should_stop or decision.should_stop,
      SOURCE_STABILITY_HOLD_REASON,
      held_by_source_stability=True,
    )
  if decision.winner == DecisionSource.CRUISE:
    return LongitudinalDecisionOutput(legacy_a_target, legacy_should_stop, "cruise_preserve_legacy")
  if decision.winner in (DecisionSource.LEAD_MPC, DecisionSource.E2E_STOP, DecisionSource.STOP_LAUNCH):
    return LongitudinalDecisionOutput(
      legacy_a_target,
      legacy_should_stop or decision.should_stop,
      "physical_hazard_preserve_legacy",
    )
  if decision.winner in (
    DecisionSource.SPEED_LIMIT,
    DecisionSource.SCC_VISION,
    DecisionSource.SCC_MAP,
    DecisionSource.OSM_TRAFFIC_CONTROL,
  ):
    return LongitudinalDecisionOutput(
      min(float(decision.a_target), legacy_a_target),
      legacy_should_stop or decision.should_stop,
      "advisory_min_legacy",
    )
  if decision.winner == DecisionSource.CRUISE_COAST:
    return LongitudinalDecisionOutput(
      float(decision.a_target),
      legacy_should_stop or decision.should_stop,
      "cruise_coast_applied",
    )
  return LongitudinalDecisionOutput(legacy_a_target, legacy_should_stop, "unknown_preserve_legacy")


def apply_longitudinal_decision_output(decision: LongitudinalDecision, legacy_a_target: float,
                                       legacy_should_stop: bool, prev_a_target: float | None = None,
                                       personality: int = log.LongitudinalPersonality.standard,
                                       dt: float = 0.0, comfort_active: bool = True) -> tuple[float, bool]:
  telemetry = apply_longitudinal_decision_output_with_telemetry(
    decision,
    legacy_a_target,
    legacy_should_stop,
    prev_a_target=prev_a_target,
    personality=personality,
    dt=dt,
    comfort_active=comfort_active,
  )
  return telemetry.applied_a_target, telemetry.applied_should_stop


def apply_longitudinal_decision_output_with_telemetry(
    decision: LongitudinalDecision, legacy_a_target: float,
    legacy_should_stop: bool, prev_a_target: float | None = None,
    personality: int = log.LongitudinalPersonality.standard,
    dt: float = 0.0, comfort_active: bool = True) -> LongitudinalDecisionTelemetry:
  legacy_a_target = float(legacy_a_target)
  legacy_should_stop = bool(legacy_should_stop)
  output = apply_decision_source_output(decision, legacy_a_target, legacy_should_stop)
  output_a_target = output.a_target
  output_should_stop = output.should_stop
  applied_reason = output.applied_reason

  if comfort_active and not output.held_by_source_stability and not output_should_stop and prev_a_target is not None:
    pre_comfort_a_target = output_a_target
    output_a_target = apply_personality_accel_comfort(decision, output_a_target, prev_a_target, personality, dt)
    if output_a_target != pre_comfort_a_target:
      applied_reason = f"{applied_reason}+personality_comfort"

  raw_a_target = float(decision.a_target)
  return LongitudinalDecisionTelemetry(
    raw_source=decision.winner,
    raw_v_target=float(decision.v_target),
    raw_a_target=raw_a_target,
    raw_should_stop=bool(decision.should_stop),
    raw_active_reason=_decision_raw_reason(decision),
    legacy_a_target=legacy_a_target,
    legacy_should_stop=legacy_should_stop,
    applied_a_target=output_a_target,
    applied_should_stop=output_should_stop,
    applied_reason=applied_reason,
    accel_delta=output_a_target - raw_a_target,
  )


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


def _personality_rate(rates: dict[int, float], personality: int) -> float:
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
                                        cruise_coast_applied: bool, cruise_coast_a_target: float,
                                        lead_mpc_allowed: bool = True) -> list[LongitudinalCandidate]:
  candidates: list[LongitudinalCandidate] = []

  if has_lead and lead_mpc_allowed:
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
      role=CandidateRole.RELAXATION,
      v_target=max(0.0, v_cruise),
      a_target=cruise_coast_a_target,
      confidence=0.80,
      urgency=0.20,
      active_reason="context_efficient_overspeed_coast",
      debug={"legacy_cruise_accel": float(a_cruise)},
    ))

  return candidates
