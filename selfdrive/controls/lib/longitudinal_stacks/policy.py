from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any

from cereal import log

from openpilot.selfdrive.controls.lib.longitudinal_decision import (
  CandidateRole,
  DecisionSource,
  LongitudinalCandidate,
  LongitudinalDecision,
)
from openpilot.selfdrive.controls.lib.longitudinal_stacks.interface import LongitudinalStackOutput
from openpilot.selfdrive.controls.lib.longitudinal_stacks.planner_seed import (
  PLANNER_SEED_INTENT_DRIVER_CRUISE,
  PLANNER_SEED_INTENT_LAUNCH,
  PLANNER_SEED_INTENT_LEAD_FOLLOW,
  PLANNER_SEED_INTENT_SAFETY_CAP,
  PLANNER_SEED_INTENT_STOP_APPROACH,
  PLANNER_SEED_FLOOR,
  PLANNER_SEED_MPC_REASON,
  PlannerSeedCandidate,
  planner_seed_intent_for_reason,
)

CUSTOM_V2_DEBUG_INTENT = "custom_v2_intent"
CUSTOM_V2_DEBUG_REASON = "custom_v2_reason"
CUSTOM_V2_DEBUG_SEED_CONTEXT = "custom_v2_seed_context"
CUSTOM_V2_DEBUG_SEED_CANDIDATE = "custom_v2_seed_candidate"
CUSTOM_V2_DEBUG_STACK_OUTPUT = "custom_v2_stack_output"
CUSTOM_V2_DEBUG_DISABLE_JERK_LIMIT = "custom_v2_disable_jerk_limit"

LONGITUDINAL_PLAN_SOURCE = log.LongitudinalPlan.LongitudinalPlanSource
LEAD_MPC_SOURCE_VALUES = {int(LONGITUDINAL_PLAN_SOURCE.lead0), int(LONGITUDINAL_PLAN_SOURCE.lead1)}
E2E_SOURCE_VALUES = {int(LONGITUDINAL_PLAN_SOURCE.e2e)}


@dataclass(frozen=True)
class SignalProviderCandidate:
  source: DecisionSource
  role: CandidateRole
  target: tuple[float, float]
  active: bool
  confidence: float
  urgency: float
  active_reason: str

  def to_longitudinal_candidate(self) -> LongitudinalCandidate:
    v_target, a_target = self.target
    return LongitudinalCandidate(
      source=self.source,
      role=self.role,
      v_target=v_target,
      a_target=a_target,
      confidence=self.confidence,
      urgency=self.urgency,
      active_reason=self.active_reason,
      required_a_target=a_target if self.role == CandidateRole.ADVISORY_CAP else None,
    )


def build_sp_candidates_from_signal_providers(providers: tuple[SignalProviderCandidate, ...]) -> list[LongitudinalCandidate]:
  return [provider.to_longitudinal_candidate() for provider in providers if provider.active]


def build_sp_longitudinal_candidates(speed_limit_active, cruise, scc_vision, scc_vision_active, scc_map, scc_map_active,
                                     speed_limit_assist, osm_traffic_control, osm_traffic_control_active):
  return build_sp_candidates_from_signal_providers((
    SignalProviderCandidate(
      source=DecisionSource.CRUISE,
      role=CandidateRole.DRIVER_INTENT,
      target=cruise,
      active=True,
      confidence=1.0,
      urgency=0.1,
      active_reason="driver_cruise_target",
    ),
    SignalProviderCandidate(
      source=DecisionSource.SPEED_LIMIT,
      role=CandidateRole.ADVISORY_CAP,
      target=speed_limit_assist,
      active=speed_limit_active,
      confidence=0.85,
      urgency=0.35,
      active_reason="speed_limit_assist_active",
    ),
    SignalProviderCandidate(
      source=DecisionSource.SCC_VISION,
      role=CandidateRole.ADVISORY_CAP,
      target=scc_vision,
      active=scc_vision_active,
      confidence=0.80,
      urgency=0.45,
      active_reason="confident_vision_curve",
    ),
    SignalProviderCandidate(
      source=DecisionSource.SCC_MAP,
      role=CandidateRole.ADVISORY_CAP,
      target=scc_map,
      active=scc_map_active,
      confidence=0.80,
      urgency=0.40,
      active_reason="confident_map_curve",
    ),
    SignalProviderCandidate(
      source=DecisionSource.OSM_TRAFFIC_CONTROL,
      role=CandidateRole.ADVISORY_CAP,
      target=osm_traffic_control,
      active=osm_traffic_control_active,
      confidence=0.75,
      urgency=0.55,
      active_reason="model_confirmed_map_caution",
    ),
  ))


def replace_driver_intent(candidates: tuple[LongitudinalCandidate, ...],
                          driver: LongitudinalCandidate) -> tuple[LongitudinalCandidate, ...]:
  return (driver, *(candidate for candidate in candidates if candidate.role != CandidateRole.DRIVER_INTENT))


def ensure_driver_intent(candidates: tuple[LongitudinalCandidate, ...], fallback_output: LongitudinalStackOutput,
                         v_target: float) -> tuple[LongitudinalCandidate, ...]:
  if any(candidate.role == CandidateRole.DRIVER_INTENT for candidate in candidates):
    return candidates
  return (
    custom_v2_candidate_with_debug(
      LongitudinalCandidate(
        source=DecisionSource.CRUISE,
        role=CandidateRole.DRIVER_INTENT,
        v_target=max(0.0, float(v_target)),
        a_target=float(fallback_output.a_target),
        confidence=1.0,
        urgency=0.1,
        active_reason="driver_cruise_target",
        should_stop=bool(fallback_output.should_stop),
      ),
      intent="driver_cruise",
      reason="sunnypilot_current_seed",
      output=fallback_output,
    ),
    *candidates,
  )


def planner_seed_candidate_to_longitudinal_candidate(candidate: PlannerSeedCandidate,
                                                     v_target: float) -> LongitudinalCandidate:
  output = candidate.output
  intent = _seed_intent(candidate)
  reason = str(candidate.reason or output.seed_reason or output.debug.get("planner_seed_candidate_reason", "") or intent)
  source = _decision_source_for_seed(intent, reason, output)
  role = _role_for_seed(intent, reason, candidate.selection)
  confidence, urgency = _confidence_urgency_for_seed(intent, role, output)

  return custom_v2_candidate_with_debug(
    LongitudinalCandidate(
      source=source,
      role=role,
      v_target=max(0.0, float(v_target)),
      a_target=float(output.a_target),
      confidence=confidence,
      urgency=urgency,
      active_reason=reason,
      should_stop=bool(output.should_stop),
    ),
    intent=_custom_v2_intent_for_seed(intent),
    reason=reason,
    output=output,
    seed_context="planner",
    seed_candidate=candidate.name,
  )


def planner_seed_candidates_to_longitudinal_candidates(candidates: tuple[PlannerSeedCandidate, ...] | list[PlannerSeedCandidate],
                                                       v_target: float) -> tuple[LongitudinalCandidate, ...]:
  return tuple(planner_seed_candidate_to_longitudinal_candidate(candidate, v_target) for candidate in candidates)


def custom_v2_candidate_with_debug(candidate: LongitudinalCandidate, intent: str, reason: str,
                                   output: LongitudinalStackOutput | None = None,
                                   seed_context: str = "", seed_candidate: str = "",
                                   disable_jerk_limit: bool = False,
                                   extra_rejected: tuple[tuple[str, str], ...] = ()) -> LongitudinalCandidate:
  debug: dict[str, Any] = dict(candidate.debug)
  debug.update({
    CUSTOM_V2_DEBUG_INTENT: str(intent),
    CUSTOM_V2_DEBUG_REASON: str(reason),
    CUSTOM_V2_DEBUG_SEED_CONTEXT: str(seed_context),
    CUSTOM_V2_DEBUG_SEED_CANDIDATE: str(seed_candidate),
  })
  if output is not None:
    debug[CUSTOM_V2_DEBUG_STACK_OUTPUT] = output
  if disable_jerk_limit:
    debug[CUSTOM_V2_DEBUG_DISABLE_JERK_LIMIT] = True
  if extra_rejected:
    debug["custom_v2_extra_rejected"] = extra_rejected
  return replace(candidate, debug=debug)


def fallback_physical_candidates(converted_candidates: tuple[LongitudinalCandidate, ...],
                                 raw_candidates: tuple[LongitudinalCandidate, ...] | list[LongitudinalCandidate],
                                 fallback_output: LongitudinalStackOutput) -> tuple[LongitudinalCandidate, ...]:
  represented = {
    _physical_candidate_identity(candidate) for candidate in converted_candidates
    if candidate.role == CandidateRole.PHYSICAL_HAZARD
  }
  planner_seed_launch_release = any(
    candidate.source == DecisionSource.STOP_LAUNCH and
    candidate.role == CandidateRole.RELAXATION and
    _candidate_custom_v2_intent(candidate) == "launch" and
    not candidate.should_stop
    for candidate in converted_candidates
  )
  fallback_source_is_lead = _source_matches(fallback_output.source, LEAD_MPC_SOURCE_VALUES, {"lead0", "lead1"})
  fallbacks: list[LongitudinalCandidate] = []
  for candidate in raw_candidates:
    if candidate.role != CandidateRole.PHYSICAL_HAZARD:
      continue
    intent = custom_v2_intent_for_source(candidate.source)
    if (candidate.source, intent, candidate.active_reason) in represented:
      continue
    if candidate.source == DecisionSource.E2E_STOP and (
      planner_seed_launch_release or (fallback_source_is_lead and not fallback_output.should_stop)
    ):
      continue
    fallbacks.append(custom_v2_candidate_with_debug(
      candidate,
      intent=intent,
      reason=candidate.active_reason,
      output=fallback_output,
      seed_context="core_physical",
      seed_candidate="",
    ))
  return tuple(fallbacks)


def selected_candidate_for_decision(decision: LongitudinalDecision) -> LongitudinalCandidate | None:
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


def custom_v2_rejections_from_decision(decision: LongitudinalDecision) -> tuple[tuple[str, str], ...]:
  rich_suppressed = getattr(decision, "suppressed_candidates", ())
  rich_keys = {
    (getattr(candidate, "source", None), _suppressed_candidate_reason(candidate))
    for candidate in rich_suppressed
  }
  rejected: list[tuple[str, str]] = [
    (_suppressed_candidate_custom_v2_intent(candidate), _suppressed_candidate_reason(candidate))
    for candidate in rich_suppressed
  ]

  candidates_by_source = {candidate.source: candidate for candidate in decision.candidates}
  for source, reason in decision.suppressed:
    if (source, str(reason)) in rich_keys:
      continue
    candidate = candidates_by_source.get(source)
    intent = _candidate_custom_v2_intent(candidate) if candidate is not None else custom_v2_intent_for_source(source)
    rejected.append((intent, str(reason)))
  return tuple(dict.fromkeys(rejected))


def custom_v2_intent_for_source(source: DecisionSource) -> str:
  if source == DecisionSource.SPEED_LIMIT:
    return "speed_policy"
  if source in (DecisionSource.SCC_VISION, DecisionSource.SCC_MAP):
    return "curve_policy"
  if source == DecisionSource.OSM_TRAFFIC_CONTROL:
    return "map_caution"
  if source == DecisionSource.LEAD_MPC:
    return "lead_follow"
  if source == DecisionSource.E2E_STOP:
    return "stop_approach"
  if source == DecisionSource.STOP_LAUNCH:
    return "launch"
  if source == DecisionSource.CRUISE_COAST:
    return "comfort_relax"
  return "driver_cruise"


def _candidate_custom_v2_intent(candidate: LongitudinalCandidate | None) -> str:
  if candidate is None:
    return "driver_cruise"
  return str(candidate.debug.get(CUSTOM_V2_DEBUG_INTENT) or custom_v2_intent_for_source(candidate.source))


def _suppressed_candidate_custom_v2_intent(candidate: Any) -> str:
  debug = getattr(candidate, "debug", {})
  intent = debug.get(CUSTOM_V2_DEBUG_INTENT) if isinstance(debug, dict) else ""
  if intent:
    return str(intent)
  return custom_v2_intent_for_source(getattr(candidate, "source", DecisionSource.CRUISE))


def _suppressed_candidate_reason(candidate: Any) -> str:
  reason = getattr(candidate, "suppression_reason", "")
  if reason:
    return str(reason)
  debug = getattr(candidate, "debug", {})
  if isinstance(debug, dict):
    debug_reason = debug.get(CUSTOM_V2_DEBUG_REASON)
    if debug_reason:
      return str(debug_reason)
  return str(getattr(candidate, "active_reason", ""))


def _physical_candidate_identity(candidate: LongitudinalCandidate) -> tuple[DecisionSource, str, str]:
  return (candidate.source, _candidate_custom_v2_intent(candidate), candidate.active_reason)


def _seed_intent(candidate: PlannerSeedCandidate) -> str:
  output = candidate.output
  intent = str(candidate.intent or output.seed_intent or "")
  if intent:
    return intent
  reason = str(candidate.reason or output.seed_reason or output.debug.get("planner_seed_candidate_reason", ""))
  if reason:
    return planner_seed_intent_for_reason(reason, output.has_lead, output.should_stop, output.source)
  if output.has_lead or _source_matches(output.source, LEAD_MPC_SOURCE_VALUES, {"lead0", "lead1"}):
    return PLANNER_SEED_INTENT_LEAD_FOLLOW
  if output.should_stop:
    return PLANNER_SEED_INTENT_STOP_APPROACH
  if float(output.a_target) < 0.0:
    return PLANNER_SEED_INTENT_SAFETY_CAP
  return PLANNER_SEED_INTENT_DRIVER_CRUISE


def _custom_v2_intent_for_seed(intent: str) -> str:
  if intent == PLANNER_SEED_INTENT_DRIVER_CRUISE:
    return "driver_cruise"
  if intent == PLANNER_SEED_INTENT_LEAD_FOLLOW:
    return "lead_follow"
  if intent == PLANNER_SEED_INTENT_STOP_APPROACH:
    return "stop_approach"
  if intent == PLANNER_SEED_INTENT_LAUNCH:
    return "launch"
  if intent == PLANNER_SEED_INTENT_SAFETY_CAP:
    return "safety_cap"
  return "driver_cruise"


def _role_for_seed(intent: str, reason: str, selection: str = "") -> CandidateRole:
  if intent in (PLANNER_SEED_INTENT_LEAD_FOLLOW, PLANNER_SEED_INTENT_STOP_APPROACH, PLANNER_SEED_INTENT_SAFETY_CAP):
    return CandidateRole.PHYSICAL_HAZARD
  if intent == PLANNER_SEED_INTENT_LAUNCH and selection != PLANNER_SEED_FLOOR:
    return CandidateRole.PHYSICAL_HAZARD
  if intent == PLANNER_SEED_INTENT_LAUNCH or reason == "plain_cruise_overspeed_coast":
    return CandidateRole.RELAXATION
  return CandidateRole.RELAXATION


def _decision_source_for_seed(intent: str, reason: str, output: LongitudinalStackOutput) -> DecisionSource:
  if intent == PLANNER_SEED_INTENT_LEAD_FOLLOW:
    return DecisionSource.LEAD_MPC
  if intent == PLANNER_SEED_INTENT_STOP_APPROACH:
    return DecisionSource.E2E_STOP
  if intent == PLANNER_SEED_INTENT_SAFETY_CAP:
    return DecisionSource.STOP_LAUNCH
  if intent == PLANNER_SEED_INTENT_LAUNCH:
    return DecisionSource.STOP_LAUNCH
  if reason == "plain_cruise_overspeed_coast":
    return DecisionSource.CRUISE_COAST
  if reason == PLANNER_SEED_MPC_REASON and _source_matches(output.source, E2E_SOURCE_VALUES, {"e2e"}):
    return DecisionSource.E2E_STOP
  return DecisionSource.CRUISE


def _confidence_urgency_for_seed(intent: str, role: CandidateRole, output: LongitudinalStackOutput) -> tuple[float, float]:
  if role == CandidateRole.PHYSICAL_HAZARD:
    if intent == PLANNER_SEED_INTENT_SAFETY_CAP:
      return 1.0, 1.0
    return 0.90, 0.80 if output.should_stop or output.a_target < -0.3 else 0.60
  if intent == PLANNER_SEED_INTENT_LAUNCH:
    return 0.90, 0.55
  return 0.80, 0.20


def _source_matches(source: object, values: set[int], names: set[str]) -> bool:
  source_name = str(source or "")
  if source_name in names:
    return True
  try:
    return int(source_name) in values
  except (TypeError, ValueError):
    return False
