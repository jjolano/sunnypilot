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
from openpilot.selfdrive.controls.lib.longitudinal_stacks.planner_seed import PLANNER_SEED_CAP, PLANNER_SEED_FLOOR, PlannerSeedCandidate, select_planner_seed_candidate
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


def test_excess_gap_closure_seed_converts_to_relaxation_not_lead_mpc_physics():
  seed = PlannerSeedCandidate(
    "excess_gap_closure",
    make_output(0.24, has_lead=True, seed_intent="lead_follow", seed_reason="excess_gap_closure"),
    selection=PLANNER_SEED_FLOOR,
  )

  candidate = planner_seed_candidate_to_longitudinal_candidate(seed, v_target=8.0)

  assert candidate.source == DecisionSource.STOP_LAUNCH
  assert candidate.role == CandidateRole.RELAXATION
  assert candidate.active_reason == "excess_gap_closure"
  assert candidate.debug[CUSTOM_V2_DEBUG_INTENT] == "lead_follow"
  assert candidate.debug[CUSTOM_V2_DEBUG_SEED_CANDIDATE] == "excess_gap_closure"


def test_lead_pullaway_seed_relaxation_loses_to_physical_hazard():
  seed = PlannerSeedCandidate(
    "lead_pullaway_pulse",
    make_output(0.3, has_lead=True, seed_intent="launch", seed_reason="confirmed_lead_pullaway_pulse"),
    selection=PLANNER_SEED_FLOOR,
  )
  launch = planner_seed_candidate_to_longitudinal_candidate(seed, v_target=8.0)
  driver = custom_v2_candidate_with_debug(
    make_candidate(DecisionSource.CRUISE, CandidateRole.DRIVER_INTENT, 0.0, "driver_cruise_target", v_target=8.0),
    intent="driver_cruise",
    reason="driver_cruise_target",
  )
  physical = custom_v2_candidate_with_debug(
    make_candidate(DecisionSource.LEAD_MPC, CandidateRole.PHYSICAL_HAZARD, -0.5, "confirmed_radar_lead", v_target=8.0),
    intent="lead_follow",
    reason="confirmed_radar_lead",
  )

  decision = LongitudinalArbiter().decide((driver, physical, launch))
  rejected = custom_v2_rejections_from_decision(decision)

  assert decision.winner == DecisionSource.LEAD_MPC
  assert ("launch", "physical_hazard_active") in rejected


def test_excess_gap_closure_seed_relaxation_loses_to_advisory_cap():
  seed = PlannerSeedCandidate(
    "excess_gap_closure",
    make_output(0.24, has_lead=True, seed_intent="lead_follow", seed_reason="excess_gap_closure"),
    selection=PLANNER_SEED_FLOOR,
  )
  gap_closure = planner_seed_candidate_to_longitudinal_candidate(seed, v_target=8.0)
  driver = custom_v2_candidate_with_debug(
    make_candidate(DecisionSource.CRUISE, CandidateRole.DRIVER_INTENT, 0.0, "driver_cruise_target", v_target=8.0),
    intent="driver_cruise",
    reason="driver_cruise_target",
  )
  advisory = custom_v2_candidate_with_debug(
    make_candidate(DecisionSource.SPEED_LIMIT, CandidateRole.ADVISORY_CAP, -0.2, "coast_biased_speed_reduction", v_target=6.0,
                   confidence=1.0, urgency=0.7),
    intent="speed_policy",
    reason="coast_biased_speed_reduction",
  )

  decision = LongitudinalArbiter().decide((driver, advisory, gap_closure))
  rejected = custom_v2_rejections_from_decision(decision)

  assert decision.winner == DecisionSource.SPEED_LIMIT
  assert ("lead_follow", "advisory_cap_active") in rejected


def test_low_speed_pullaway_cap_seed_is_restrictive_cap_not_physics():
  seed = PlannerSeedCandidate(
    "low_speed_pullaway_accel_step_cap",
    make_output(0.55, has_lead=True, seed_intent="launch", seed_reason="low_speed_pullaway_accel_step_cap"),
  )

  candidate = planner_seed_candidate_to_longitudinal_candidate(seed, v_target=8.0)

  assert candidate.source == DecisionSource.STOP_LAUNCH
  assert candidate.role == CandidateRole.ADVISORY_CAP
  assert candidate.debug[CUSTOM_V2_DEBUG_INTENT] == "launch"


def test_lead_pullaway_pulse_cap_seed_is_restrictive_cap_not_progress_authority():
  seed = PlannerSeedCandidate(
    "lead_pullaway_pulse_accel_cap",
    make_output(0.42, has_lead=True, seed_intent="launch", seed_reason="lead_pullaway_pulse_accel_cap"),
  )

  candidate = planner_seed_candidate_to_longitudinal_candidate(seed, v_target=8.0)

  assert candidate.source == DecisionSource.STOP_LAUNCH
  assert candidate.role == CandidateRole.ADVISORY_CAP
  assert candidate.debug[CUSTOM_V2_DEBUG_INTENT] == "launch"
  assert candidate.active_reason == "lead_pullaway_pulse_accel_cap"


def test_low_speed_pullaway_cap_seed_loses_to_more_restrictive_advisory():
  seed = PlannerSeedCandidate(
    "low_speed_pullaway_accel_step_cap",
    make_output(0.55, has_lead=True, seed_intent="launch", seed_reason="low_speed_pullaway_accel_step_cap"),
  )
  pullaway_cap = planner_seed_candidate_to_longitudinal_candidate(seed, v_target=8.0)
  driver = custom_v2_candidate_with_debug(
    make_candidate(DecisionSource.CRUISE, CandidateRole.DRIVER_INTENT, 1.0, "driver_cruise_target", v_target=8.0),
    intent="driver_cruise",
    reason="driver_cruise_target",
  )
  advisory = custom_v2_candidate_with_debug(
    make_candidate(DecisionSource.SPEED_LIMIT, CandidateRole.ADVISORY_CAP, -0.2, "coast_biased_speed_reduction", v_target=6.0,
                   confidence=1.0, urgency=0.7),
    intent="speed_policy",
    reason="coast_biased_speed_reduction",
  )

  decision = LongitudinalArbiter().decide((driver, advisory, pullaway_cap))
  rejected = custom_v2_rejections_from_decision(decision)

  assert decision.winner == DecisionSource.SPEED_LIMIT
  assert ("launch", "higher_advisory_target") in rejected


def test_low_speed_pullaway_cap_seed_loses_to_physical_braking():
  seed = PlannerSeedCandidate(
    "low_speed_pullaway_accel_step_cap",
    make_output(0.55, has_lead=True, seed_intent="launch", seed_reason="low_speed_pullaway_accel_step_cap"),
  )
  pullaway_cap = planner_seed_candidate_to_longitudinal_candidate(seed, v_target=8.0)
  driver = custom_v2_candidate_with_debug(
    make_candidate(DecisionSource.CRUISE, CandidateRole.DRIVER_INTENT, 1.0, "driver_cruise_target", v_target=8.0),
    intent="driver_cruise",
    reason="driver_cruise_target",
  )
  physical = custom_v2_candidate_with_debug(
    make_candidate(DecisionSource.LEAD_MPC, CandidateRole.PHYSICAL_HAZARD, -0.5, "confirmed_radar_lead", v_target=8.0),
    intent="lead_follow",
    reason="confirmed_radar_lead",
  )

  decision = LongitudinalArbiter().decide((driver, physical, pullaway_cap))
  rejected = custom_v2_rejections_from_decision(decision)

  assert decision.winner == DecisionSource.LEAD_MPC
  assert ("launch", "physical_hazard_active") in rejected


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


def test_scene_derived_confirmed_lead_pullaway_no_longer_creates_custom_physics():
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

  assert candidates == ()
  assert ("lead_follow", "planner_seed_required") in rejected


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

  assert progress_candidates == ()
  assert decision.winner == DecisionSource.LEAD_MPC
  assert decision.a_target == -0.7


def test_routine_lead_approach_seed_is_relaxation_not_physical_hazard():
  from openpilot.selfdrive.controls.lib.longitudinal_stacks.planner_seed import (
    PlannerSeedCandidate, PLANNER_SEED_FLOOR, PLANNER_SEED_INTENT_LEAD_FOLLOW,
  )
  from openpilot.selfdrive.controls.lib.longitudinal_stacks.planner_seed_policy import (
    planner_seed_candidate_to_longitudinal_candidate, ROUTINE_COMFORT_SEED_REASONS,
  )
  from openpilot.selfdrive.controls.lib.longitudinal_decision import CandidateRole
  from openpilot.selfdrive.controls.lib.longitudinal_planner import ROUTINE_LEAD_APPROACH_SEED_REASON

  output = make_output(a_target=-0.15, has_lead=True)
  candidate = PlannerSeedCandidate(
    name="routine_lead_approach",
    output=output,
    selection=PLANNER_SEED_FLOOR,
    intent=PLANNER_SEED_INTENT_LEAD_FOLLOW,
    reason=ROUTINE_LEAD_APPROACH_SEED_REASON,
  )
  converted = planner_seed_candidate_to_longitudinal_candidate(candidate, v_target=18.0)
  assert converted.role == CandidateRole.RELAXATION, f"routine lead approach floor should be RELAXATION, got {converted.role}"
  assert ROUTINE_LEAD_APPROACH_SEED_REASON in ROUTINE_COMFORT_SEED_REASONS


def test_routine_lead_approach_seed_does_not_become_progress_authority():
  from openpilot.selfdrive.controls.lib.longitudinal_stacks.planner_seed_policy import (
    _custom_v2_intent_for_seed, _role_for_seed,
  )
  from openpilot.selfdrive.controls.lib.longitudinal_decision import CandidateRole
  from openpilot.selfdrive.controls.lib.longitudinal_stacks.planner_seed import PLANNER_SEED_FLOOR
  from openpilot.selfdrive.controls.lib.longitudinal_planner import ROUTINE_LEAD_APPROACH_SEED_REASON

  role = _role_for_seed("lead_follow", ROUTINE_LEAD_APPROACH_SEED_REASON, PLANNER_SEED_FLOOR)
  assert role == CandidateRole.RELAXATION
  intent = _custom_v2_intent_for_seed("lead_follow")
  assert intent == "lead_follow", f"routine lead approach intent should be lead_follow, got {intent}"
  assert intent != "launch", "routine lead approach must not be launch intent"


def test_safety_cap_beats_routine_comfort_floor():
  from openpilot.selfdrive.controls.lib.longitudinal_stacks.planner_seed import (
    PLANNER_SEED_CAP, PLANNER_SEED_FLOOR, select_planner_seed_candidate,
  )
  from openpilot.selfdrive.controls.lib.longitudinal_planner import ROUTINE_LEAD_APPROACH_SEED_REASON

  baseline = PlannerSeedCandidate(
    name="baseline",
    output=make_output(a_target=-1.5, has_lead=True),
    selection=PLANNER_SEED_CAP,
  )
  routine_floor = PlannerSeedCandidate(
    name="routine_lead_approach",
    output=make_output(a_target=-0.15, has_lead=True),
    selection=PLANNER_SEED_FLOOR,
    reason=ROUTINE_LEAD_APPROACH_SEED_REASON,
  )
  safety_cap = PlannerSeedCandidate(
    name="moving_lead_stop_gap_guard",
    output=make_output(a_target=-2.0, has_lead=True),
    selection=PLANNER_SEED_CAP,
    reason="moving_lead_stop_gap_guard",
  )
  selected = select_planner_seed_candidate([baseline, routine_floor, safety_cap])
  # Safety cap should win (most restrictive cap)
  assert selected.output.a_target <= -1.5, f"safety cap should win over routine floor, got {selected.output.a_target}"


def test_routine_comfort_relaxation_survives_nonurgent_lead_mpc_fallback():
  from openpilot.selfdrive.controls.lib.longitudinal_decision import CandidateRole, DecisionSource, LongitudinalCandidate
  from openpilot.selfdrive.controls.lib.longitudinal_stacks.planner_seed_policy import fallback_physical_candidates
  from openpilot.selfdrive.controls.lib.longitudinal_planner import ROUTINE_LEAD_APPROACH_SEED_REASON

  # Routine comfort RELAXATION seed that owns non-urgent shape
  routine_seed = LongitudinalCandidate(
    source=DecisionSource.LEAD_MPC,
    role=CandidateRole.RELAXATION,
    v_target=18.0,
    a_target=-0.15,
    confidence=0.8,
    urgency=0.2,
    active_reason=ROUTINE_LEAD_APPROACH_SEED_REASON,
    should_stop=False,
    debug={
      "routine_lead_can_own_nonurgent_shape": True,
      "routine_lead_existing_target_safety_relevant": False,
      "routine_lead_urgent_bypass": False,
    },
  )
  # Raw non-urgent LEAD_MPC physical hazard that would suppress routine comfort
  raw_lead_mpc = LongitudinalCandidate(
    source=DecisionSource.LEAD_MPC,
    role=CandidateRole.PHYSICAL_HAZARD,
    v_target=18.0,
    a_target=-0.8,
    confidence=0.9,
    urgency=0.6,
    active_reason="lead0",
    should_stop=False,
  )
  fallback_output = make_output(a_target=-0.8, has_lead=True)
  result = fallback_physical_candidates((routine_seed,), (raw_lead_mpc,), fallback_output)
  # Non-urgent LEAD_MPC fallback should be skipped when routine comfort owns shape
  assert len(result) == 0, f"non-urgent LEAD_MPC fallback should be skipped, got {len(result)} candidates"


def test_routine_comfort_does_not_block_urgent_lead_mpc_fallback():
  from openpilot.selfdrive.controls.lib.longitudinal_decision import CandidateRole, DecisionSource, LongitudinalCandidate
  from openpilot.selfdrive.controls.lib.longitudinal_stacks.planner_seed_policy import fallback_physical_candidates
  from openpilot.selfdrive.controls.lib.longitudinal_planner import ROUTINE_LEAD_APPROACH_SEED_REASON

  routine_seed = LongitudinalCandidate(
    source=DecisionSource.LEAD_MPC,
    role=CandidateRole.RELAXATION,
    v_target=18.0,
    a_target=-0.15,
    confidence=0.8,
    urgency=0.2,
    active_reason=ROUTINE_LEAD_APPROACH_SEED_REASON,
    should_stop=False,
    debug={
      "routine_lead_can_own_nonurgent_shape": True,
      "routine_lead_existing_target_safety_relevant": False,
      "routine_lead_urgent_bypass": False,
    },
  )
  # Urgent LEAD_MPC with hard braking
  urgent_lead_mpc = LongitudinalCandidate(
    source=DecisionSource.LEAD_MPC,
    role=CandidateRole.PHYSICAL_HAZARD,
    v_target=18.0,
    a_target=-2.5,
    confidence=0.9,
    urgency=0.8,
    active_reason="lead0",
    should_stop=False,
  )
  fallback_output = make_output(a_target=-2.5, has_lead=True)
  result = fallback_physical_candidates((routine_seed,), (urgent_lead_mpc,), fallback_output)
  assert len(result) == 1, f"urgent LEAD_MPC fallback should survive, got {len(result)} candidates"
  assert result[0].a_target == -2.5


def test_routine_comfort_does_not_block_should_stop_fallback():
  from openpilot.selfdrive.controls.lib.longitudinal_decision import CandidateRole, DecisionSource, LongitudinalCandidate
  from openpilot.selfdrive.controls.lib.longitudinal_stacks.planner_seed_policy import fallback_physical_candidates
  from openpilot.selfdrive.controls.lib.longitudinal_planner import ROUTINE_LEAD_APPROACH_SEED_REASON

  routine_seed = LongitudinalCandidate(
    source=DecisionSource.LEAD_MPC,
    role=CandidateRole.RELAXATION,
    v_target=18.0,
    a_target=-0.15,
    confidence=0.8,
    urgency=0.2,
    active_reason=ROUTINE_LEAD_APPROACH_SEED_REASON,
    should_stop=False,
    debug={
      "routine_lead_can_own_nonurgent_shape": True,
      "routine_lead_existing_target_safety_relevant": False,
      "routine_lead_urgent_bypass": False,
    },
  )
  stop_lead_mpc = LongitudinalCandidate(
    source=DecisionSource.LEAD_MPC,
    role=CandidateRole.PHYSICAL_HAZARD,
    v_target=0.5,
    a_target=-1.5,
    confidence=0.95,
    urgency=0.9,
    active_reason="lead0",
    should_stop=True,
  )
  fallback_output = make_output(a_target=-1.5, has_lead=True)
  result = fallback_physical_candidates((routine_seed,), (stop_lead_mpc,), fallback_output)
  assert len(result) == 1, f"should_stop LEAD_MPC fallback should survive, got {len(result)} candidates"
  assert result[0].should_stop is True


def test_routine_comfort_without_own_shape_allows_nonurgent_fallback():
  from openpilot.selfdrive.controls.lib.longitudinal_decision import CandidateRole, DecisionSource, LongitudinalCandidate
  from openpilot.selfdrive.controls.lib.longitudinal_stacks.planner_seed_policy import fallback_physical_candidates
  from openpilot.selfdrive.controls.lib.longitudinal_planner import ROUTINE_LEAD_APPROACH_SEED_REASON

  # Routine seed where safety_relevant=True (routine should NOT own shape)
  routine_seed = LongitudinalCandidate(
    source=DecisionSource.LEAD_MPC,
    role=CandidateRole.RELAXATION,
    v_target=18.0,
    a_target=-0.15,
    confidence=0.8,
    urgency=0.2,
    active_reason=ROUTINE_LEAD_APPROACH_SEED_REASON,
    should_stop=False,
    debug={
      "routine_lead_can_own_nonurgent_shape": True,
      "routine_lead_existing_target_safety_relevant": True,  # safety relevant
      "routine_lead_urgent_bypass": False,
    },
  )
  raw_lead_mpc = LongitudinalCandidate(
    source=DecisionSource.LEAD_MPC,
    role=CandidateRole.PHYSICAL_HAZARD,
    v_target=18.0,
    a_target=-0.8,
    confidence=0.9,
    urgency=0.6,
    active_reason="lead0",
    should_stop=False,
  )
  fallback_output = make_output(a_target=-0.8, has_lead=True)
  result = fallback_physical_candidates((routine_seed,), (raw_lead_mpc,), fallback_output)
  assert len(result) == 1, f"non-urgent LEAD_MPC should pass when routine does NOT own shape, got {len(result)}"


def test_advisory_cap_still_wins_over_routine_comfort_floor():
  from openpilot.selfdrive.controls.lib.longitudinal_planner import ROUTINE_LEAD_APPROACH_SEED_REASON

  baseline = PlannerSeedCandidate(
    name="baseline",
    output=make_output(a_target=-0.5, has_lead=True),
  )
  routine_floor = PlannerSeedCandidate(
    name="routine_lead_approach",
    output=make_output(a_target=-0.15, has_lead=True),
    selection=PLANNER_SEED_FLOOR,
    reason=ROUTINE_LEAD_APPROACH_SEED_REASON,
  )
  advisory_cap = PlannerSeedCandidate(
    name="lead_flicker_speedup_cap",
    output=make_output(a_target=0.0, has_lead=True),
    selection=PLANNER_SEED_CAP,
    reason="lead_flicker_speedup_cap",
  )
  selected = select_planner_seed_candidate([baseline, routine_floor, advisory_cap])
  # Advisory cap should clamp positive accel
  assert selected.output.a_target <= 0.0, f"advisory cap should win, got {selected.output.a_target}"


def test_select_planner_seed_candidate_routine_floor_selected():
  from openpilot.selfdrive.controls.lib.longitudinal_planner import ROUTINE_LEAD_APPROACH_SEED_REASON

  baseline = PlannerSeedCandidate(
    name="baseline",
    output=make_output(a_target=-1.0, has_lead=True),
  )
  routine_floor = PlannerSeedCandidate(
    name="routine_lead_approach",
    output=make_output(a_target=-0.2, has_lead=True),
    selection=PLANNER_SEED_FLOOR,
    reason=ROUTINE_LEAD_APPROACH_SEED_REASON,
  )
  selected = select_planner_seed_candidate([baseline, routine_floor])
  # Routine floor should raise the target from -1.0 to -0.2
  assert selected.output.a_target == -0.2, f"routine floor should be selected, got {selected.output.a_target}"
