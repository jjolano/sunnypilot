from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace

from openpilot.selfdrive.controls.lib.longitudinal_stacks.interface import LongitudinalStackOutput
from openpilot.selfdrive.controls.lib.longitudinal_stacks.selector import CUSTOM_V1

CUSTOM_V1_CAP = "cap"
CUSTOM_V1_FLOOR = "floor"
POST_CAP_FLOOR_GROUPS = {"lead_stop_approach_slew", "low_speed_pullaway_accel_step"}


@dataclass(frozen=True)
class CustomV1Candidate:
  name: str
  output: LongitudinalStackOutput
  selection: str = CUSTOM_V1_CAP
  group: str = ""


def select_custom_v1_candidate(candidates: Iterable[CustomV1Candidate]) -> CustomV1Candidate:
  candidate_list = tuple(candidates)
  if not candidate_list:
    raise ValueError("no_custom_v1_candidates")
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


def _select_floor_candidate(baseline: CustomV1Candidate,
                            candidates: Iterable[CustomV1Candidate]) -> CustomV1Candidate | None:
  floors = [candidate for candidate in candidates if candidate.selection == CUSTOM_V1_FLOOR]
  applicable = [
    candidate for candidate in floors
    if candidate.output.a_target > baseline.output.a_target or
       (baseline.output.should_stop and not candidate.output.should_stop)
  ]
  return max(applicable, key=lambda candidate: candidate.output.a_target, default=None)


def _select_cap_candidate(baseline: CustomV1Candidate, candidates: Iterable[CustomV1Candidate],
                          selected_floor: CustomV1Candidate | None) -> CustomV1Candidate | None:
  caps = [candidate for candidate in candidates if candidate.selection != CUSTOM_V1_FLOOR]
  applicable = [
    candidate for candidate in caps
    if candidate.output.a_target < baseline.output.a_target or
       _candidate_caps_floor(candidate, selected_floor)
  ]
  return min(applicable, key=lambda candidate: candidate.output.a_target, default=None)


def _candidate_caps_floor(candidate: CustomV1Candidate, selected_floor: CustomV1Candidate | None) -> bool:
  if selected_floor is None or candidate.output.a_target >= selected_floor.output.a_target:
    return False
  return not candidate.group or candidate.group == selected_floor.group


def _select_post_cap_floor(cap_candidate: CustomV1Candidate | None,
                           candidates: Iterable[CustomV1Candidate]) -> CustomV1Candidate | None:
  if cap_candidate is None:
    return None
  floors = [
    candidate for candidate in candidates
    if _candidate_is_post_cap_floor(candidate) and
       candidate.output.a_target > cap_candidate.output.a_target
  ]
  return max(floors, key=lambda candidate: candidate.output.a_target, default=None)


def _candidate_is_post_cap_floor(candidate: CustomV1Candidate) -> bool:
  return (
    candidate.selection == CUSTOM_V1_FLOOR and candidate.group in POST_CAP_FLOOR_GROUPS
  ) or candidate.group == "lead_stop_approach_slew"


def _merge_stop_intent(selected: CustomV1Candidate, candidates: Iterable[CustomV1Candidate]) -> CustomV1Candidate:
  if selected.output.should_stop:
    return selected
  stop_candidates = [candidate for candidate in candidates if candidate.output.should_stop]
  if not stop_candidates:
    return selected
  stop_candidate = min(stop_candidates, key=lambda candidate: candidate.output.a_target)
  debug = dict(selected.output.debug)
  debug.update(stop_candidate.output.debug)
  return CustomV1Candidate(
    stop_candidate.name if selected.name == "sunnypilot-current" else selected.name,
    replace(selected.output, should_stop=True, debug=debug),
    selection=selected.selection,
    group=selected.group,
  )


class CustomLongitudinalStackV1:
  stack_name = CUSTOM_V1

  def update(self, sunnypilot_output: LongitudinalStackOutput,
             candidates: Iterable[CustomV1Candidate] = ()) -> LongitudinalStackOutput:
    primary_candidate = CustomV1Candidate("sunnypilot-current", sunnypilot_output)
    selected = select_custom_v1_candidate((primary_candidate, *tuple(candidates)))

    debug = dict(sunnypilot_output.debug)
    debug.update(selected.output.debug)
    debug["custom_stack"] = self.stack_name
    debug["custom_v1_candidate"] = selected.name
    debug["custom_v1_mode"] = "candidate_arbitration"
    return replace(selected.output, debug=debug)
