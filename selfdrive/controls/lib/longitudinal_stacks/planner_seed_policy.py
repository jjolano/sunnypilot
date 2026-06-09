from __future__ import annotations

from cereal import log

from openpilot.selfdrive.controls.lib.longitudinal_decision import (
  CandidateRole,
  DecisionSource,
  LongitudinalCandidate,
)
from openpilot.selfdrive.controls.lib.longitudinal_stacks.custom_v2_debug import (
  _candidate_custom_v2_intent,
  _physical_candidate_identity,
  custom_v2_candidate_with_debug,
  custom_v2_intent_for_source,
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


LONGITUDINAL_PLAN_SOURCE = log.LongitudinalPlan.LongitudinalPlanSource
LEAD_MPC_SOURCE_VALUES = {int(LONGITUDINAL_PLAN_SOURCE.lead0), int(LONGITUDINAL_PLAN_SOURCE.lead1)}
E2E_SOURCE_VALUES = {int(LONGITUDINAL_PLAN_SOURCE.e2e)}
PROGRESS_RELAXATION_SEED_REASONS = {"confirmed_lead_pullaway_pulse", "excess_gap_closure"}
ROUTINE_COMFORT_SEED_REASONS = {"routine_slower_lead_approach"}


def _is_routine_comfort_relaxation(converted_candidates: tuple[LongitudinalCandidate, ...]) -> bool:
  """Detect whether a routine comfort RELAXATION seed owns non-urgent comfort shape.

  A routine comfort seed is present when:
  - role == RELAXATION
  - reason == routine_slower_lead_approach
  - debug says routine can own non-urgent shape
  - debug says existing target is not safety-relevant
  - debug says no urgent bypass
  """
  for candidate in converted_candidates:
    if candidate.role != CandidateRole.RELAXATION:
      continue
    reason = str(candidate.active_reason or "")
    if reason not in ROUTINE_COMFORT_SEED_REASONS:
      continue
    debug = getattr(candidate, "debug", None) or {}
    if isinstance(debug, dict):
      can_own = bool(debug.get("routine_lead_can_own_nonurgent_shape", False))
      safety_relevant = bool(debug.get("routine_lead_existing_target_safety_relevant", False))
      urgent_bypass = bool(debug.get("routine_lead_urgent_bypass", False))
    else:
      can_own = bool(getattr(debug, "routine_lead_can_own_nonurgent_shape", False))
      safety_relevant = bool(getattr(debug, "routine_lead_existing_target_safety_relevant", False))
      urgent_bypass = bool(getattr(debug, "routine_lead_urgent_bypass", False))
    if can_own and not safety_relevant and not urgent_bypass:
      return True
  return False


PROGRESS_ACCEL_CAP_SEED_REASONS = {
  "creep_pullaway_launch_accel_cap",
  "low_speed_pullaway_accel_step_cap",
  "lead_pullaway_pulse_accel_cap",
  "excess_gap_closure_accel_cap",
}
STOP_LAUNCH_SEED_REASONS = {
  "confirmed_lead_pullaway_pulse",
  "lead_pullaway_pulse_accel_cap",
  "excess_gap_closure",
  "excess_gap_closure_accel_cap",
}


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
  routine_comfort_owns_shape = _is_routine_comfort_relaxation(converted_candidates)
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
    # When routine comfort owns non-urgent shape, skip raw LEAD_MPC fallback
    # that would merely reintroduce the same non-urgent lead baseline.
    # True physical hazards still pass through:
    # - should_stop: stop-threat handling requires this candidate
    # - hard braking (a_target < -1.0 m/s²): decel strong enough to be
    #   safety-relevant even if should_stop is not yet set. This threshold
    #   is intentionally more conservative than the lead-stop slew threshold
    #   (LEAD_STOP_APPROACH_DECEL_SLEW_MIN_LEAD_DECEL = 0.6 m/s²) because
    #   fallback suppression should err on the side of preserving safety-relevant
    #   decel, while the lead-stop slew can apply its own rate limit.
    if routine_comfort_owns_shape and candidate.source == DecisionSource.LEAD_MPC:
      if not candidate.should_stop and not (float(candidate.a_target) < -1.0):
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
  if reason in PROGRESS_ACCEL_CAP_SEED_REASONS:
    return CandidateRole.ADVISORY_CAP
  if reason in ROUTINE_COMFORT_SEED_REASONS and selection == PLANNER_SEED_FLOOR:
    return CandidateRole.RELAXATION
  if reason in PROGRESS_RELAXATION_SEED_REASONS and selection == PLANNER_SEED_FLOOR:
    return CandidateRole.RELAXATION
  if intent in (PLANNER_SEED_INTENT_LEAD_FOLLOW, PLANNER_SEED_INTENT_STOP_APPROACH, PLANNER_SEED_INTENT_SAFETY_CAP):
    return CandidateRole.PHYSICAL_HAZARD
  if intent == PLANNER_SEED_INTENT_LAUNCH and selection != PLANNER_SEED_FLOOR:
    return CandidateRole.PHYSICAL_HAZARD
  if intent == PLANNER_SEED_INTENT_LAUNCH or reason == "plain_cruise_overspeed_coast":
    return CandidateRole.RELAXATION
  return CandidateRole.RELAXATION


def _decision_source_for_seed(intent: str, reason: str, output: LongitudinalStackOutput) -> DecisionSource:
  if reason in STOP_LAUNCH_SEED_REASONS:
    return DecisionSource.STOP_LAUNCH
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
