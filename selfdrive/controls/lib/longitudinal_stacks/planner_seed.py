from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace

from openpilot.selfdrive.controls.lib.longitudinal_stacks.interface import LongitudinalStackOutput

PLANNER_SEED_CAP = "cap"
PLANNER_SEED_FLOOR = "floor"
POST_CAP_FLOOR_GROUPS = {"lead_stop_approach_slew", "low_speed_pullaway_accel_step"}
STOP_INTENT_RELEASE_GROUPS = {"creep_pullaway_launch"}


@dataclass(frozen=True)
class PlannerSeedCandidate:
  name: str
  output: LongitudinalStackOutput
  selection: str = PLANNER_SEED_CAP
  group: str = ""


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
       _candidate_caps_floor(candidate, selected_floor)
  ]
  return min(applicable, key=lambda candidate: candidate.output.a_target, default=None)


def _candidate_caps_floor(candidate: PlannerSeedCandidate, selected_floor: PlannerSeedCandidate | None) -> bool:
  if selected_floor is None or candidate.output.a_target >= selected_floor.output.a_target:
    return False
  return not candidate.group or candidate.group == selected_floor.group


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
    replace(selected.output, should_stop=True, debug=debug),
    selection=selected.selection,
    group=selected.group,
  )
