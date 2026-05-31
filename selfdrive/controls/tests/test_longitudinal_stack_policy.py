from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N
from openpilot.selfdrive.controls.lib.longitudinal_decision import (
  CandidateRole,
  DecisionSource,
  LongitudinalArbiter,
  LongitudinalCandidate,
)
from openpilot.selfdrive.controls.lib.longitudinal_stacks.custom_v2 import (
  CustomV2Scene,
  build_custom_v2_progress_candidates,
)
from openpilot.selfdrive.controls.lib.longitudinal_stacks.interface import LongitudinalStackOutput
from openpilot.selfdrive.controls.lib.longitudinal_stacks.planner_seed import PlannerSeedCandidate
from openpilot.selfdrive.controls.lib.longitudinal_stacks.policy import (
  CUSTOM_V2_DEBUG_INTENT,
  CUSTOM_V2_DEBUG_REASON,
  CUSTOM_V2_DEBUG_SEED_CANDIDATE,
  CUSTOM_V2_DEBUG_SEED_CONTEXT,
  custom_v2_candidate_with_debug,
  custom_v2_rejections_from_decision,
  fallback_physical_candidates,
  planner_seed_candidate_to_longitudinal_candidate,
  replace_driver_intent,
)


def make_output(a_target=0.0, should_stop=False, has_lead=False, seed_intent="", seed_reason=""):
  return LongitudinalStackOutput(
    a_target=a_target,
    should_stop=should_stop,
    has_lead=has_lead,
    source="cruise",
    allow_throttle=True,
    allow_brake=True,
    speeds=tuple(0.0 for _ in range(CONTROL_N)),
    accels=tuple(a_target for _ in range(CONTROL_N)),
    jerks=tuple(0.0 for _ in range(CONTROL_N)),
    seed_intent=seed_intent,
    seed_reason=seed_reason,
  )


def make_candidate(source, role, a_target, reason, v_target=8.0, confidence=1.0, urgency=0.5, should_stop=False):
  return LongitudinalCandidate(
    source=source,
    role=role,
    v_target=v_target,
    a_target=a_target,
    confidence=confidence,
    urgency=urgency,
    active_reason=reason,
    should_stop=should_stop,
  )


def test_duplicate_stop_launch_rejections_keep_actual_custom_v2_intents():
  cruise = custom_v2_candidate_with_debug(
    make_candidate(DecisionSource.CRUISE, CandidateRole.DRIVER_INTENT, 0.0, "driver_cruise_target"),
    intent="driver_cruise",
    reason="driver_cruise_target",
  )
  safety_cap = custom_v2_candidate_with_debug(
    make_candidate(DecisionSource.STOP_LAUNCH, CandidateRole.PHYSICAL_HAZARD, 0.0, "lead_flicker_speedup_cap"),
    intent="safety_cap",
    reason="lead_flicker_speedup_cap",
  )
  launch = custom_v2_candidate_with_debug(
    make_candidate(DecisionSource.STOP_LAUNCH, CandidateRole.RELAXATION, 1.35, "no_lead_stop_clear"),
    intent="launch",
    reason="no_lead_stop_clear",
  )
  excess_gap = custom_v2_candidate_with_debug(
    make_candidate(DecisionSource.STOP_LAUNCH, CandidateRole.RELAXATION, 0.4, "excess_gap_progress"),
    intent="lead_follow",
    reason="excess_gap_progress",
  )

  decision = LongitudinalArbiter().decide([cruise, safety_cap, launch, excess_gap])
  rejected = custom_v2_rejections_from_decision(decision)

  assert decision.winner == DecisionSource.STOP_LAUNCH
  assert ("launch", "physical_hazard_active") in rejected
  assert ("lead_follow", "physical_hazard_active") in rejected
  assert ("driver_cruise", "physical_hazard_active") in rejected


def test_cruise_one_pedal_replaces_driver_intent_without_duplicate_cruise_driver():
  driver = custom_v2_candidate_with_debug(
    make_candidate(DecisionSource.CRUISE, CandidateRole.DRIVER_INTENT, 0.3, "driver_cruise_target"),
    intent="driver_cruise",
    reason="driver_cruise_target",
  )
  one_pedal = custom_v2_candidate_with_debug(
    make_candidate(DecisionSource.CRUISE, CandidateRole.DRIVER_INTENT, 0.0, "lift_off_coast"),
    intent="one_pedal",
    reason="lift_off_coast",
  )

  candidates = replace_driver_intent((driver,), one_pedal)

  assert candidates == (one_pedal,)
  assert candidates[0].debug[CUSTOM_V2_DEBUG_INTENT] == "one_pedal"
  assert candidates[0].debug[CUSTOM_V2_DEBUG_REASON] == "lift_off_coast"


def test_fallback_physical_candidates_keep_same_source_different_hazards():
  converted_seed = custom_v2_candidate_with_debug(
    make_candidate(DecisionSource.LEAD_MPC, CandidateRole.PHYSICAL_HAZARD, -0.4, "planner_seed_mpc"),
    intent="lead_follow",
    reason="planner_seed_mpc",
  )
  duplicate_raw = make_candidate(DecisionSource.LEAD_MPC, CandidateRole.PHYSICAL_HAZARD, -0.4, "planner_seed_mpc")
  distinct_raw = make_candidate(DecisionSource.LEAD_MPC, CandidateRole.PHYSICAL_HAZARD, -0.7, "confirmed_radar_lead")

  fallbacks = fallback_physical_candidates((converted_seed,), [duplicate_raw, distinct_raw], make_output(-0.1, has_lead=True))

  assert len(fallbacks) == 1
  assert fallbacks[0].active_reason == "confirmed_radar_lead"
  assert fallbacks[0].debug[CUSTOM_V2_DEBUG_INTENT] == "lead_follow"
  assert fallbacks[0].debug[CUSTOM_V2_DEBUG_SEED_CONTEXT] == "core_physical"


def test_planner_seed_conversion_preserves_custom_v2_metadata():
  seed = PlannerSeedCandidate(
    "lead_crawl_accel_cap",
    make_output(0.0, has_lead=True, seed_intent="lead_follow", seed_reason="lead_crawl_accel_cap"),
  )

  candidate = planner_seed_candidate_to_longitudinal_candidate(seed, v_target=6.0)

  assert candidate.source == DecisionSource.LEAD_MPC
  assert candidate.role == CandidateRole.PHYSICAL_HAZARD
  assert candidate.active_reason == "lead_crawl_accel_cap"
  assert candidate.debug[CUSTOM_V2_DEBUG_INTENT] == "lead_follow"
  assert candidate.debug[CUSTOM_V2_DEBUG_REASON] == "lead_crawl_accel_cap"
  assert candidate.debug[CUSTOM_V2_DEBUG_SEED_CONTEXT] == "planner"
  assert candidate.debug[CUSTOM_V2_DEBUG_SEED_CANDIDATE] == "lead_crawl_accel_cap"


def test_scene_derived_progress_candidates_are_relaxation_only_and_lose_to_physical():
  output = make_output(0.0)
  scene = CustomV2Scene(v_ego=0.2, v_cruise=8.0, model_stop_distance=30.0, model_desired_accel=0.0)
  progress_candidates, rejected = build_custom_v2_progress_candidates(output, scene, (-2.0, 2.0))
  cruise = custom_v2_candidate_with_debug(
    make_candidate(DecisionSource.CRUISE, CandidateRole.DRIVER_INTENT, 0.0, "driver_cruise_target"),
    intent="driver_cruise",
    reason="driver_cruise_target",
  )
  e2e_stop = custom_v2_candidate_with_debug(
    make_candidate(DecisionSource.E2E_STOP, CandidateRole.PHYSICAL_HAZARD, -0.6, "model_stop", should_stop=True),
    intent="stop_approach",
    reason="model_stop",
  )

  decision = LongitudinalArbiter().decide([cruise, e2e_stop, *progress_candidates])
  custom_rejected = custom_v2_rejections_from_decision(decision)

  assert rejected == ()
  assert len(progress_candidates) == 1
  assert progress_candidates[0].role == CandidateRole.RELAXATION
  assert progress_candidates[0].debug[CUSTOM_V2_DEBUG_INTENT] == "launch"
  assert progress_candidates[0].debug[CUSTOM_V2_DEBUG_SEED_CONTEXT] == ""
  assert decision.winner == DecisionSource.E2E_STOP
  assert ("launch", "physical_hazard_active") in custom_rejected


def test_scene_derived_confirmed_lead_pullaway_candidate_is_standard_relaxation():
  output = make_output(0.0, has_lead=True)
  scene = CustomV2Scene(
    v_ego=0.2,
    v_cruise=8.0,
    has_lead=True,
    lead_v=1.3,
    lead_v_rel=1.1,
    lead_gap_excess=2.5,
    lead_progress_allowed=True,
    lead_confirmed_pullaway=True,
  )

  candidates, rejected = build_custom_v2_progress_candidates(output, scene, (-2.0, 2.0))
  launch = [candidate for candidate in candidates if candidate.active_reason == "confirmed_lead_pullaway"]

  assert rejected == ()
  assert len(launch) == 1
  assert launch[0].source == DecisionSource.STOP_LAUNCH
  assert launch[0].role == CandidateRole.RELAXATION
  assert launch[0].a_target == 1.30
  assert launch[0].should_stop is False
  assert launch[0].debug[CUSTOM_V2_DEBUG_INTENT] == "launch"
  assert launch[0].debug[CUSTOM_V2_DEBUG_REASON] == "confirmed_lead_pullaway"


def test_scene_derived_progress_candidates_are_blocked_by_overrides_and_stop_threats():
  output = make_output(0.0)
  blocked_scene = CustomV2Scene(v_ego=0.2, v_cruise=8.0, force_slow_decel=True, model_stop_distance=30.0)
  stop_scene = CustomV2Scene(v_ego=0.2, v_cruise=8.0, stop_threat=True, model_stop_distance=30.0)
  independent_stop_scene = CustomV2Scene(
    v_ego=0.2,
    v_cruise=8.0,
    has_lead=True,
    lead_v=0.5,
    lead_confirmed_pullaway=True,
    independent_stop_threat=True,
  )

  blocked_candidates, blocked_rejected = build_custom_v2_progress_candidates(output, blocked_scene, (-2.0, 2.0))
  stop_candidates, stop_rejected = build_custom_v2_progress_candidates(output, stop_scene, (-2.0, 2.0))
  independent_candidates, _independent_rejected = build_custom_v2_progress_candidates(output, independent_stop_scene, (-2.0, 2.0))

  assert blocked_candidates == ()
  assert blocked_rejected == (("launch", "driver_or_force_blocked"),)
  assert stop_candidates == ()
  assert stop_rejected == ()
  assert independent_candidates == ()


def test_scene_derived_excess_gap_progress_loses_to_close_physical_lead():
  output = make_output(-0.2, has_lead=True)
  scene = CustomV2Scene(
    v_ego=15.5, v_cruise=17.0, has_lead=True, lead_v=9.0, lead_v_rel=-6.5,
    lead_gap_excess=40.0, lead_progress_allowed=True,
  )
  progress_candidates, _rejected = build_custom_v2_progress_candidates(output, scene, (-2.0, 2.0))
  cruise = custom_v2_candidate_with_debug(
    make_candidate(DecisionSource.CRUISE, CandidateRole.DRIVER_INTENT, -0.2, "driver_cruise_target", v_target=17.0),
    intent="driver_cruise",
    reason="driver_cruise_target",
  )
  lead = custom_v2_candidate_with_debug(
    make_candidate(DecisionSource.LEAD_MPC, CandidateRole.PHYSICAL_HAZARD, -0.7, "confirmed_radar_lead", v_target=17.0),
    intent="lead_follow",
    reason="confirmed_radar_lead",
  )

  decision = LongitudinalArbiter().decide([cruise, lead, *progress_candidates])

  assert any(candidate.debug[CUSTOM_V2_DEBUG_INTENT] == "lead_follow" for candidate in progress_candidates)
  assert decision.winner == DecisionSource.LEAD_MPC
  assert decision.a_target == -0.7
