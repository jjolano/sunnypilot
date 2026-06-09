from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace

from cereal import log

from openpilot.selfdrive.controls.lib.longitudinal_stacks.interface import LongitudinalStackOutput

PLANNER_SEED_CAP = "cap"
PLANNER_SEED_FLOOR = "floor"
POST_CAP_FLOOR_GROUPS = {"lead_stop_approach_slew", "low_speed_pullaway_accel_step"}
STOP_INTENT_RELEASE_GROUPS = {"creep_pullaway_launch", "lead_pullaway_pulse", "excess_gap_closure"}
PLANNER_SEED_INTENT_DRIVER_CRUISE = "driver_cruise"
PLANNER_SEED_INTENT_LEAD_FOLLOW = "lead_follow"
PLANNER_SEED_INTENT_STOP_APPROACH = "stop_approach"
PLANNER_SEED_INTENT_LAUNCH = "launch"
PLANNER_SEED_INTENT_SAFETY_CAP = "safety_cap"
PLANNER_SEED_MPC_REASON = "planner_seed_mpc"
LONGITUDINAL_PLAN_SOURCE = log.LongitudinalPlan.LongitudinalPlanSource
LEAD_MPC_SOURCE_VALUES = {int(LONGITUDINAL_PLAN_SOURCE.lead0), int(LONGITUDINAL_PLAN_SOURCE.lead1)}
E2E_SOURCE_VALUES = {int(LONGITUDINAL_PLAN_SOURCE.e2e)}

STOP_APPROACH_SEED_REASONS = {
  "engage_model_stop_bootstrap",
  "no_lead_close_stop_settle",
  "no_lead_model_runway_comfort",
  "no_lead_model_stop_approach",
  "low_speed_model_runway_positive_cap",
}
LEAD_FOLLOW_SEED_REASONS = {
  "stopped_lead_stop_gap_guard",
  "creep_to_stop_gap",
  "creep_to_stop_gap_accel_cap",
  "stopped_lead_gap_fill",
  "stopped_lead_gap_fill_accel_cap",
  "lead_crawl_accel_cap",
  "stopped_lead_creep_hold",
  "moving_lead_stop_gap_guard",
  "lead_accel_recovery",
  "lead_stop_approach_slew",
  "lead_loss_e2e_guard",
  "excess_gap_closure",
  "excess_gap_closure_accel_cap",
  "routine_slower_lead_approach",
}
LAUNCH_SEED_REASONS = {
  "creep_pullaway_launch",
  "creep_pullaway_launch_accel_cap",
  "low_speed_pullaway_accel_step_floor",
  "low_speed_pullaway_accel_step_cap",
  "confirmed_lead_pullaway_pulse",
  "lead_pullaway_pulse_accel_cap",
}
DRIVER_CRUISE_SEED_REASONS = {"plain_cruise_overspeed_coast"}
SAFETY_CAP_SEED_REASONS = {"lead_flicker_speedup_cap"}


@dataclass(frozen=True)
class PlannerSeedCandidate:
  name: str
  output: LongitudinalStackOutput
  selection: str = PLANNER_SEED_CAP
  group: str = ""
  intent: str = ""
  reason: str = ""

  def __post_init__(self) -> None:
    reason = self.reason or self.output.seed_reason
    intent = self.intent or self.output.seed_intent
    if reason and not intent:
      intent = planner_seed_intent_for_reason(reason, self.output.has_lead, self.output.should_stop, self.output.source)
    object.__setattr__(self, "reason", str(reason or ""))
    object.__setattr__(self, "intent", str(intent or ""))
    if (intent and self.output.seed_intent != intent) or (reason and self.output.seed_reason != reason):
      object.__setattr__(
        self,
        "output",
        replace(self.output, seed_intent=str(intent or ""), seed_reason=str(reason or "")),
      )


def select_planner_seed_candidate(candidates: Iterable[PlannerSeedCandidate]) -> PlannerSeedCandidate:
  candidate_list = tuple(candidates)
  if not candidate_list:
    raise ValueError("no_planner_seed_candidates")
  baseline = candidate_list[0]
  floor_candidate = _select_floor_candidate(baseline, candidate_list[1:])
  cap_candidate = _select_cap_candidate(baseline, candidate_list[1:], floor_candidate)
  post_cap_floor = _select_post_cap_floor(cap_candidate, candidate_list[1:])
  if post_cap_floor is not None:
    selected = post_cap_floor
  elif cap_candidate is not None:
    selected = cap_candidate
  elif floor_candidate is not None:
    selected = floor_candidate
  else:
    selected = baseline
  return _merge_stop_intent(selected, candidate_list[1:])


def planner_seed_intent_for_reason(reason: str, has_lead: bool = False, should_stop: bool = False,
                                   source: object = "") -> str:
  reason = str(reason or "")
  if reason == PLANNER_SEED_MPC_REASON:
    if _source_matches(source, LEAD_MPC_SOURCE_VALUES, {"lead0", "lead1"}):
      return PLANNER_SEED_INTENT_LEAD_FOLLOW
    if should_stop or _source_matches(source, E2E_SOURCE_VALUES, {"e2e"}):
      return PLANNER_SEED_INTENT_STOP_APPROACH
    return PLANNER_SEED_INTENT_DRIVER_CRUISE
  if reason in STOP_APPROACH_SEED_REASONS:
    return PLANNER_SEED_INTENT_STOP_APPROACH
  if reason in LEAD_FOLLOW_SEED_REASONS:
    return PLANNER_SEED_INTENT_LEAD_FOLLOW
  if reason in LAUNCH_SEED_REASONS:
    return PLANNER_SEED_INTENT_LAUNCH
  if reason in SAFETY_CAP_SEED_REASONS:
    return PLANNER_SEED_INTENT_SAFETY_CAP
  if reason in DRIVER_CRUISE_SEED_REASONS:
    return PLANNER_SEED_INTENT_DRIVER_CRUISE
  return PLANNER_SEED_INTENT_DRIVER_CRUISE


def _source_matches(source: object, values: set[int], names: set[str]) -> bool:
  if str(source or "") in names:
    return True
  try:
    return int(source) in values
  except (TypeError, ValueError):
    return False


def _select_floor_candidate(baseline: PlannerSeedCandidate,
                            candidates: Iterable[PlannerSeedCandidate]) -> PlannerSeedCandidate | None:
  floors = [candidate for candidate in candidates if candidate.selection == PLANNER_SEED_FLOOR]
  applicable = [
    candidate for candidate in floors
    if candidate.output.a_target > baseline.output.a_target or
       (baseline.output.should_stop and not candidate.output.should_stop)
  ]
  return max(applicable, key=lambda candidate: candidate.output.a_target, default=None)


def _select_cap_candidate(baseline: PlannerSeedCandidate, candidates: Iterable[PlannerSeedCandidate],
                          selected_floor: PlannerSeedCandidate | None) -> PlannerSeedCandidate | None:
  caps = [candidate for candidate in candidates if candidate.selection != PLANNER_SEED_FLOOR]
  applicable = [
    candidate for candidate in caps
    if candidate.output.a_target < baseline.output.a_target or
       _candidate_caps_floor(candidate, selected_floor) or
       _candidate_is_equal_safety_cap(candidate, baseline)
  ]
  return min(applicable, key=lambda candidate: candidate.output.a_target, default=None)


def _candidate_caps_floor(candidate: PlannerSeedCandidate, selected_floor: PlannerSeedCandidate | None) -> bool:
  if selected_floor is None or candidate.output.a_target >= selected_floor.output.a_target:
    return False
  return not candidate.group or candidate.group == selected_floor.group


def _candidate_is_equal_safety_cap(candidate: PlannerSeedCandidate, baseline: PlannerSeedCandidate) -> bool:
  return candidate.intent == PLANNER_SEED_INTENT_SAFETY_CAP and candidate.output.a_target <= baseline.output.a_target


def _select_post_cap_floor(cap_candidate: PlannerSeedCandidate | None,
                           candidates: Iterable[PlannerSeedCandidate]) -> PlannerSeedCandidate | None:
  if cap_candidate is None:
    return None
  floors = [
    candidate for candidate in candidates
    if _candidate_is_post_cap_floor(candidate) and
       candidate.output.a_target > cap_candidate.output.a_target
  ]
  return max(floors, key=lambda candidate: candidate.output.a_target, default=None)


def _candidate_is_post_cap_floor(candidate: PlannerSeedCandidate) -> bool:
  return (
    candidate.selection == PLANNER_SEED_FLOOR and candidate.group in POST_CAP_FLOOR_GROUPS
  ) or candidate.group == "lead_stop_approach_slew"


def _merge_stop_intent(selected: PlannerSeedCandidate, candidates: Iterable[PlannerSeedCandidate]) -> PlannerSeedCandidate:
  if selected.output.should_stop:
    return selected
  if selected.group in STOP_INTENT_RELEASE_GROUPS:
    return selected
  stop_candidates = [candidate for candidate in candidates if candidate.output.should_stop]
  if not stop_candidates:
    return selected
  stop_candidate = min(stop_candidates, key=lambda candidate: candidate.output.a_target)
  debug = dict(selected.output.debug)
  debug.update(stop_candidate.output.debug)
  return PlannerSeedCandidate(
    stop_candidate.name if selected.name == "sunnypilot-current" else selected.name,
    replace(
      selected.output,
      should_stop=True,
      debug=debug,
      seed_intent=selected.intent or stop_candidate.intent,
      seed_reason=selected.reason or stop_candidate.reason,
    ),
    selection=selected.selection,
    group=selected.group,
    intent=selected.intent or stop_candidate.intent,
    reason=selected.reason or stop_candidate.reason,
  )
