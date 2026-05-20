import pytest

from cereal import log
from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N
from openpilot.selfdrive.controls.lib.longitudinal_stacks.custom_v2 import (
  COMFORT_RELAX_ACCEL_MIN,
  CUSTOM_V2_INTENTS,
  CustomLongitudinalStackV2,
  CustomV2SceneValidationError,
  CustomV2Scene,
  MPH_TO_MS,
  NORMAL_NEGATIVE_RETREAT_JERK,
  NO_LEAD_LAUNCH_ACCEL_MAX,
  ONE_PEDAL_CREEP_ACCEL_MAX,
  ONE_PEDAL_CREEP_TARGET_SPEED,
  ONE_PEDAL_FULL_STOP_DECEL,
  ONE_PEDAL_FULL_STOP_HOLD_SPEED,
  ONE_PEDAL_MODE_CREEP,
  ONE_PEDAL_MODE_FULL_STOP,
  POSITIVE_PROGRESS_JERK,
  SYNTH_TRAJECTORY_DT,
  dynamic_cruise_overspeed_leeway,
  lead_evidence_releases_stop,
  no_lead_stop_clear,
)
from openpilot.selfdrive.controls.lib.longitudinal_stacks.interface import LongitudinalStackOutput, validate_stack_output
from openpilot.selfdrive.controls.lib.longitudinal_stacks.planner_seed import PlannerSeedCandidate, select_planner_seed_candidate


def make_output(a_target=0.0, should_stop=False, has_lead=False, debug=None, speeds=None, accels=None, jerks=None,
                seed_intent="", seed_reason="", source="cruise"):
  return LongitudinalStackOutput(
    a_target=a_target,
    should_stop=should_stop,
    has_lead=has_lead,
    source=source,
    allow_throttle=True,
    allow_brake=True,
    speeds=tuple(0.0 for _ in range(CONTROL_N)) if speeds is None else speeds,
    accels=tuple(a_target for _ in range(CONTROL_N)) if accels is None else accels,
    jerks=tuple(0.0 for _ in range(CONTROL_N)) if jerks is None else jerks,
    debug={} if debug is None else debug,
    seed_intent=seed_intent,
    seed_reason=seed_reason,
  )


def test_custom_v2_intent_taxonomy_is_complete():
  assert CUSTOM_V2_INTENTS == (
    "driver_cruise",
    "lead_follow",
    "stop_approach",
    "launch",
    "one_pedal",
    "speed_policy",
    "curve_policy",
    "map_caution",
    "comfort_relax",
    "safety_cap",
  )


def test_no_lead_launch_requires_stop_clear():
  clear_scene = CustomV2Scene(v_ego=0.2, v_cruise=5.0, model_stop_distance=30.0, model_desired_accel=0.0)
  blocked_scene = CustomV2Scene(v_ego=0.2, v_cruise=5.0, model_stop_distance=10.0, model_desired_accel=0.0)
  stack = CustomLongitudinalStackV2()

  clear_output = stack.update(make_output(), clear_scene, accel_limits=(-2.0, 2.0))
  blocked_output = stack.update(make_output(), blocked_scene, accel_limits=(-2.0, 2.0))

  assert no_lead_stop_clear(clear_scene)
  assert not no_lead_stop_clear(blocked_scene)
  assert clear_output.a_target == NO_LEAD_LAUNCH_ACCEL_MAX
  assert clear_output.debug["custom_v2_selected_intent"] == "launch"
  assert blocked_output.a_target == 0.0
  assert "model_stop_not_clear" in blocked_output.debug["custom_v2_rejected_reasons"]


def test_zero_safety_cap_blocks_custom_progress_floors():
  scene = CustomV2Scene(v_ego=0.2, v_cruise=5.0, model_stop_distance=30.0, model_desired_accel=0.0)

  output = CustomLongitudinalStackV2().update(
    make_output(0.0, seed_intent="safety_cap", seed_reason="lead_flicker_speedup_cap"),
    scene,
    accel_limits=(-2.0, 2.0),
  )

  assert output.a_target == 0.0
  assert output.debug["custom_v2_selected_intent"] == "safety_cap"
  assert output.debug["custom_v2_selected_reason"] == "lead_flicker_speedup_cap"


def test_equal_zero_safety_cap_seed_wins_over_baseline_seed():
  baseline = PlannerSeedCandidate("sunnypilot-current", make_output(0.0))
  safety_cap = PlannerSeedCandidate(
    "lead_flicker_speedup_cap",
    make_output(0.0, seed_intent="safety_cap", seed_reason="lead_flicker_speedup_cap"),
  )

  selected = select_planner_seed_candidate((baseline, safety_cap))

  assert selected.name == "lead_flicker_speedup_cap"
  assert selected.intent == "safety_cap"


def test_direct_lead_scene_does_not_create_lead_progress():
  scene = CustomV2Scene(
    v_ego=0.3,
    v_cruise=6.0,
    has_lead=True,
    lead_v=0.2,
    lead_confirmed_pullaway=True,
    stop_threat=True,
    model_should_stop=True,
    model_stop_distance=5.0,
    model_desired_accel=-1.0,
  )

  output = CustomLongitudinalStackV2().update(make_output(-0.4, should_stop=True), scene, accel_limits=(-2.0, 2.0))

  assert lead_evidence_releases_stop(scene)
  assert output.a_target == -0.4
  assert output.should_stop
  assert output.debug["custom_v2_selected_intent"] == "driver_cruise"
  assert output.debug["custom_v2_selected_reason"] == "sunnypilot_current_seed"


def test_excess_gap_progress_softens_far_fast_lead_approach_braking():
  scene = CustomV2Scene(
    v_ego=15.5,
    v_cruise=16.7,
    has_lead=True,
    lead_v=9.0,
    lead_v_rel=-6.5,
    lead_gap_excess=40.0,
  )

  output = CustomLongitudinalStackV2().update(make_output(-0.3, has_lead=True), scene, accel_limits=(-2.0, 2.0))

  assert -0.05 <= output.a_target <= 0.05
  assert output.debug["custom_v2_selected_intent"] == "lead_follow"
  assert output.debug["custom_v2_selected_reason"] == "excess_gap_progress"


def test_excess_gap_progress_remains_blocked_for_close_closing_lead():
  scene = CustomV2Scene(
    v_ego=15.0,
    v_cruise=17.0,
    has_lead=True,
    lead_v_rel=-1.0,
    lead_gap_excess=4.0,
  )

  output = CustomLongitudinalStackV2().update(make_output(-0.3, has_lead=True), scene, accel_limits=(-2.0, 2.0))

  assert output.a_target == -0.3
  assert "closing_speed_guard" in output.debug["custom_v2_rejected_reasons"]


def test_excess_gap_progress_ignores_stopped_lead():
  scene = CustomV2Scene(
    v_ego=0.0,
    v_cruise=8.0,
    has_lead=True,
    lead_v=0.0,
    lead_v_rel=0.0,
    lead_gap_excess=9.0,
  )

  output = CustomLongitudinalStackV2().update(make_output(0.0, has_lead=True), scene, accel_limits=(-2.0, 2.0))

  assert output.a_target == 0.0
  assert "closing_speed_guard" in output.debug["custom_v2_rejected_reasons"]


def test_planner_seed_classification_preserves_seed_trajectory():
  speeds = tuple(float(idx) for idx in range(CONTROL_N))
  accels = tuple(-0.7 + 0.01 * idx for idx in range(CONTROL_N))
  jerks = tuple(-0.1 for _ in range(CONTROL_N))
  seed = make_output(
    -0.7,
    should_stop=True,
    has_lead=True,
    speeds=speeds,
    accels=accels,
    jerks=jerks,
    debug={"planner_seed_candidate_reason": "stopped_lead_stop_gap_guard"},
  )
  scene = CustomV2Scene(v_ego=0.3, v_cruise=6.0, has_lead=True)

  output = CustomLongitudinalStackV2().update(seed, scene, accel_limits=(-2.0, 2.0))

  assert output.a_target == -0.7
  assert output.should_stop
  assert output.speeds == speeds
  assert output.accels == accels
  assert output.jerks == jerks
  assert output.debug["custom_v2_selected_intent"] == "lead_follow"
  assert output.debug["custom_v2_selected_reason"] == "stopped_lead_stop_gap_guard"


def test_structured_planner_seed_classification_does_not_require_debug_reason():
  seed = make_output(
    -0.7,
    should_stop=True,
    has_lead=True,
    seed_intent="lead_follow",
    seed_reason="stopped_lead_stop_gap_guard",
  )
  scene = CustomV2Scene(v_ego=0.3, v_cruise=6.0, has_lead=True)

  output = CustomLongitudinalStackV2().update(seed, scene, accel_limits=(-2.0, 2.0))

  assert output.a_target == -0.7
  assert output.debug["custom_v2_selected_intent"] == "lead_follow"
  assert output.debug["custom_v2_selected_reason"] == "stopped_lead_stop_gap_guard"


def test_planner_lead_seed_caps_are_not_relaxed_by_pullaway_progress():
  seed = make_output(
    0.55,
    has_lead=True,
    seed_intent="lead_follow",
    seed_reason="lead_crawl_accel_cap",
  )
  scene = CustomV2Scene(
    v_ego=0.8,
    v_cruise=8.0,
    has_lead=True,
    lead_v=0.5,
    lead_v_rel=-0.3,
    lead_gap_excess=1.4,
    lead_confirmed_pullaway=True,
  )

  output = CustomLongitudinalStackV2().update(seed, scene, accel_limits=(-2.0, 2.0))

  assert lead_evidence_releases_stop(scene)
  assert output.a_target == 0.55
  assert output.debug["custom_v2_selected_intent"] == "lead_follow"
  assert output.debug["custom_v2_selected_reason"] == "lead_crawl_accel_cap"
  assert "planner_seed_preserved" in output.debug["custom_v2_rejected_reasons"]


def test_planner_seed_mpc_classifies_by_context():
  output = CustomLongitudinalStackV2().update(
    make_output(-0.6, should_stop=False, has_lead=True, debug={"planner_seed_candidate_reason": "planner_seed_mpc"}),
    CustomV2Scene(v_ego=10.0, v_cruise=10.0, has_lead=True),
    accel_limits=(-2.0, 2.0),
  )

  assert output.debug["custom_v2_selected_intent"] == "lead_follow"
  assert output.debug["custom_v2_selected_reason"] == "planner_seed_mpc"


def test_lead_mpc_source_is_preserved_without_independent_progress():
  output = CustomLongitudinalStackV2().update(
    make_output(
      -0.6,
      has_lead=True,
      source=log.LongitudinalPlan.LongitudinalPlanSource.lead0,
    ),
    CustomV2Scene(v_ego=10.0, v_cruise=12.0, has_lead=True, lead_gap_excess=10.0),
    accel_limits=(-2.0, 2.0),
  )

  assert output.a_target == -0.6
  assert output.debug["custom_v2_selected_intent"] == "lead_follow"
  assert output.debug["custom_v2_selected_reason"] == "lead_mpc_seed"


def test_e2e_source_decel_is_preserved_with_lead_pullaway_evidence():
  scene = CustomV2Scene(
    v_ego=2.2,
    v_cruise=8.0,
    has_lead=True,
    lead_v=4.2,
    lead_v_rel=2.0,
    lead_gap_excess=10.0,
    lead_opening_prediction=True,
    lead_confirmed_pullaway=True,
    model_desired_accel=-0.8,
  )

  output = CustomLongitudinalStackV2().update(
    make_output(
      -0.8,
      has_lead=True,
      source=log.LongitudinalPlan.LongitudinalPlanSource.e2e,
    ),
    scene,
    accel_limits=(-2.0, 2.0),
  )

  assert lead_evidence_releases_stop(scene)
  assert output.a_target == -0.8
  assert output.debug["custom_v2_selected_intent"] == "stop_approach"
  assert output.debug["custom_v2_selected_reason"] == "e2e_seed"


def test_opening_lead_progress_preserves_planner_decel():
  scene = CustomV2Scene(
    v_ego=0.0,
    v_cruise=8.0,
    has_lead=True,
    lead_v=0.8,
    lead_v_rel=0.8,
    lead_gap_excess=4.5,
    lead_confirmed_pullaway=True,
  )

  output = CustomLongitudinalStackV2().update(make_output(-1.0, has_lead=True), scene, accel_limits=(-2.0, 2.0))

  assert lead_evidence_releases_stop(scene)
  assert output.a_target == -1.0
  assert "closing_speed_guard" in output.debug["custom_v2_rejected_reasons"]


def test_speed_policy_is_coast_biased_for_speed_reductions():
  scene = CustomV2Scene(
    v_ego=22.0,
    v_cruise=24.0,
    accel_coast=-0.25,
    speed_limit_active=True,
    speed_limit_v_target=18.0,
    speed_limit_a_target=-1.2,
  )

  output = CustomLongitudinalStackV2().update(make_output(0.8), scene, accel_limits=(-2.0, 2.0))

  assert output.a_target == -0.25
  assert output.debug["custom_v2_selected_intent"] == "speed_policy"
  assert output.debug["custom_v2_selected_reason"] == "coast_biased_speed_reduction"


def test_map_only_caution_is_ignored_for_control():
  scene = CustomV2Scene(
    v_ego=12.0,
    v_cruise=18.0,
    map_caution_active=True,
    map_caution_confirmed=False,
    map_caution_a_target=-1.0,
  )

  output = CustomLongitudinalStackV2().update(make_output(0.5), scene, accel_limits=(-2.0, 2.0))

  assert output.a_target == 0.5
  assert not output.should_stop
  assert output.debug["custom_v2_selected_intent"] == "driver_cruise"
  assert "map_only_ignored" in output.debug["custom_v2_rejected_reasons"]


def test_comfort_relax_only_softens_advisory_decel():
  scene = CustomV2Scene(v_ego=12.0, v_cruise=18.0, curve_active=True, curve_a_target=-1.0)

  output = CustomLongitudinalStackV2().update(make_output(0.4), scene, accel_limits=(-2.0, 2.0))

  assert output.a_target == COMFORT_RELAX_ACCEL_MIN
  assert output.debug["custom_v2_selected_intent"] == "comfort_relax"
  assert output.debug["custom_v2_selected_reason"] == "clear_margin_advisory_softening"


def test_comfort_relax_does_not_soften_confirmed_map_caution():
  scene = CustomV2Scene(v_ego=12.0, v_cruise=18.0, map_caution_active=True, map_caution_confirmed=True, map_caution_a_target=-1.0)

  output = CustomLongitudinalStackV2().update(make_output(0.4), scene, accel_limits=(-2.0, 2.0))

  assert output.a_target == -1.0
  assert output.debug["custom_v2_selected_intent"] == "map_caution"
  assert output.debug["custom_v2_selected_reason"] == "confirmed_map_caution"


def test_dynamic_plain_cruise_leeway_allows_downhill_coasting():
  scene = CustomV2Scene(v_ego=20.0 + 6.0 * MPH_TO_MS, v_cruise=20.0, accel_coast=0.25)

  output = CustomLongitudinalStackV2().update(make_output(-0.8), scene, accel_limits=(-2.0, 2.0))

  assert dynamic_cruise_overspeed_leeway(scene.accel_coast) == 10.0 * MPH_TO_MS
  assert output.a_target == 0.0
  assert output.debug["custom_v2_selected_reason"] == "dynamic_overspeed_coast_leeway"


def test_force_slow_decel_is_safety_cap():
  scene = CustomV2Scene(v_ego=10.0, v_cruise=20.0, force_slow_decel=True)

  output = CustomLongitudinalStackV2().update(make_output(0.5), scene, accel_limits=(-2.0, 2.0))

  assert output.a_target == -0.2
  assert output.should_stop
  assert output.debug["custom_v2_selected_intent"] == "safety_cap"
  assert validate_stack_output(output, accel_limits=(-2.0, 2.0)).valid


def test_optional_invalid_advisory_source_is_ignored():
  scene = CustomV2Scene(
    v_ego=15.0,
    v_cruise=15.0,
    speed_limit_active=True,
    speed_limit_v_target=10.0,
    speed_limit_a_target=float("nan"),
  )

  output = CustomLongitudinalStackV2().update(make_output(0.2), scene, accel_limits=(-2.0, 2.0))

  assert output.a_target == 0.2
  assert output.debug["custom_v2_selected_intent"] == "driver_cruise"


def test_invalid_core_scene_field_fails_closed():
  with pytest.raises(CustomV2SceneValidationError) as error:
    CustomLongitudinalStackV2().update(make_output(), CustomV2Scene(v_ego=float("nan")), accel_limits=(-2.0, 2.0))

  assert error.value.reason == "invalid_scene_v_ego"


def test_hard_model_stop_uses_required_decel_clipped_to_limits():
  scene = CustomV2Scene(
    v_ego=12.0,
    v_cruise=12.0,
    stop_threat=True,
    model_should_stop=True,
    model_stop_distance=10.0,
    model_desired_accel=-0.2,
  )

  output = CustomLongitudinalStackV2().update(make_output(0.0), scene, accel_limits=(-3.0, 2.0))

  assert output.a_target == -3.0
  assert output.should_stop
  assert output.accels[0] == -3.0
  assert output.debug["custom_v2_selected_intent"] == "stop_approach"
  assert output.debug["custom_v2_selected_reason"] == "hard_model_stop_threat"


def test_normal_stop_approach_remains_comfort_bounded():
  scene = CustomV2Scene(
    v_ego=12.0,
    v_cruise=12.0,
    stop_threat=True,
    model_should_stop=False,
    model_stop_distance=10.0,
    model_desired_accel=-2.0,
  )

  output = CustomLongitudinalStackV2().update(make_output(0.0), scene, accel_limits=(-3.0, 2.0))

  assert output.a_target == -1.5
  assert not output.should_stop
  assert output.debug["custom_v2_selected_intent"] == "stop_approach"
  assert output.debug["custom_v2_selected_reason"] == "comfort_early_stop_threat"


def test_normal_v2_accel_changes_are_jerk_limited():
  scene = CustomV2Scene(v_ego=0.2, v_cruise=5.0, model_stop_distance=30.0, model_desired_accel=0.0)

  output = CustomLongitudinalStackV2().update(make_output(0.0), scene, accel_limits=(-2.0, 2.0))

  assert output.a_target == NO_LEAD_LAUNCH_ACCEL_MAX
  assert output.accels[0] == POSITIVE_PROGRESS_JERK * SYNTH_TRAJECTORY_DT
  assert output.jerks[0] == POSITIVE_PROGRESS_JERK


def test_normal_negative_v2_changes_are_jerk_limited():
  scene = CustomV2Scene(
    v_ego=22.0,
    v_cruise=24.0,
    accel_coast=-0.25,
    speed_limit_active=True,
    speed_limit_v_target=18.0,
    speed_limit_a_target=-1.2,
  )

  output = CustomLongitudinalStackV2().update(make_output(0.8), scene, accel_limits=(-2.0, 2.0))

  assert output.a_target == -0.25
  assert output.accels[0] == 0.8 + NORMAL_NEGATIVE_RETREAT_JERK * SYNTH_TRAJECTORY_DT
  assert output.jerks[0] == NORMAL_NEGATIVE_RETREAT_JERK


def test_one_pedal_lift_off_coasts_instead_of_accelerating_to_cruise():
  scene = CustomV2Scene(v_ego=12.0, v_cruise=20.0, one_pedal_mode=ONE_PEDAL_MODE_CREEP)

  output = CustomLongitudinalStackV2().update(make_output(0.8), scene, accel_limits=(-2.0, 2.0))

  assert output.a_target == 0.0
  assert not output.should_stop
  assert output.debug["custom_v2_selected_intent"] == "one_pedal"
  assert output.debug["custom_v2_selected_reason"] == "lift_off_coast"


def test_one_pedal_ignores_no_hazard_advisory_braking():
  scene = CustomV2Scene(
    v_ego=22.0,
    v_cruise=24.0,
    speed_limit_active=True,
    speed_limit_v_target=18.0,
    speed_limit_a_target=-1.2,
    curve_active=True,
    curve_a_target=-0.8,
    map_caution_active=True,
    map_caution_confirmed=True,
    map_caution_a_target=-1.0,
    one_pedal_mode=ONE_PEDAL_MODE_CREEP,
  )

  output = CustomLongitudinalStackV2().update(make_output(0.6), scene, accel_limits=(-2.0, 2.0))

  assert output.a_target == 0.0
  assert output.debug["custom_v2_selected_intent"] == "one_pedal"
  assert output.debug["custom_v2_selected_reason"] == "lift_off_coast"


def test_one_pedal_preserves_physical_lead_mpc_braking():
  output = CustomLongitudinalStackV2().update(
    make_output(-0.7, has_lead=True, source=log.LongitudinalPlan.LongitudinalPlanSource.lead0),
    CustomV2Scene(v_ego=14.0, v_cruise=20.0, has_lead=True, one_pedal_mode=ONE_PEDAL_MODE_CREEP),
    accel_limits=(-2.0, 2.0),
  )

  assert output.a_target == -0.7
  assert output.debug["custom_v2_selected_intent"] == "lead_follow"
  assert "physical_hazard_preserved" in output.debug["custom_v2_rejected_reasons"]


def test_one_pedal_does_not_create_lead_braking_for_nonhazard_lead():
  scene = CustomV2Scene(v_ego=14.0, v_cruise=20.0, has_lead=True, lead_v=14.0, lead_v_rel=0.0, one_pedal_mode=ONE_PEDAL_MODE_CREEP)

  output = CustomLongitudinalStackV2().update(make_output(0.5, has_lead=True), scene, accel_limits=(-2.0, 2.0))

  assert output.a_target == 0.0
  assert output.debug["custom_v2_selected_intent"] == "one_pedal"
  assert output.debug["custom_v2_selected_reason"] == "lift_off_coast"


def test_one_pedal_creep_holds_crawl_with_clear_evidence():
  scene = CustomV2Scene(
    v_ego=0.8,
    v_cruise=10.0,
    model_stop_distance=30.0,
    model_desired_accel=0.0,
    one_pedal_mode=ONE_PEDAL_MODE_CREEP,
  )

  output = CustomLongitudinalStackV2().update(make_output(0.0), scene, accel_limits=(-2.0, 2.0))

  assert 0.0 < output.a_target <= ONE_PEDAL_CREEP_ACCEL_MAX
  assert output.a_target == (ONE_PEDAL_CREEP_TARGET_SPEED - scene.v_ego) * 0.5
  assert output.debug["custom_v2_selected_intent"] == "one_pedal"
  assert output.debug["custom_v2_selected_reason"] == "terminal_creep"


def test_one_pedal_creep_from_standstill_requires_clear_evidence():
  scene = CustomV2Scene(v_ego=0.0, v_cruise=10.0, model_stop_distance=10.0, model_desired_accel=0.0, one_pedal_mode=ONE_PEDAL_MODE_CREEP)

  output = CustomLongitudinalStackV2().update(make_output(0.0), scene, accel_limits=(-2.0, 2.0))

  assert output.a_target == 0.0
  assert output.debug["custom_v2_selected_reason"] == "lift_off_coast"
  assert "terminal_creep_not_authorized" in output.debug["custom_v2_rejected_reasons"]


def test_one_pedal_full_stop_gently_stops_and_holds_near_standstill():
  rolling_scene = CustomV2Scene(v_ego=2.0, v_cruise=10.0, one_pedal_mode=ONE_PEDAL_MODE_FULL_STOP)
  hold_scene = CustomV2Scene(v_ego=ONE_PEDAL_FULL_STOP_HOLD_SPEED, v_cruise=10.0, one_pedal_mode=ONE_PEDAL_MODE_FULL_STOP)

  rolling_output = CustomLongitudinalStackV2().update(make_output(0.4), rolling_scene, accel_limits=(-2.0, 2.0))
  hold_output = CustomLongitudinalStackV2().update(make_output(0.4), hold_scene, accel_limits=(-2.0, 2.0))

  assert rolling_output.a_target == ONE_PEDAL_FULL_STOP_DECEL
  assert not rolling_output.should_stop
  assert hold_output.a_target == ONE_PEDAL_FULL_STOP_DECEL
  assert hold_output.should_stop
  assert hold_output.debug["custom_v2_selected_reason"] == "low_speed_terminal_stop"


def test_one_pedal_cruise_hold_escape_uses_normal_custom_v2_policy():
  scene = CustomV2Scene(
    v_ego=0.2,
    v_cruise=5.0,
    model_stop_distance=30.0,
    model_desired_accel=0.0,
    one_pedal_mode=ONE_PEDAL_MODE_CREEP,
    one_pedal_cruise_hold=True,
  )

  output = CustomLongitudinalStackV2().update(make_output(0.0), scene, accel_limits=(-2.0, 2.0))

  assert output.a_target == NO_LEAD_LAUNCH_ACCEL_MAX
  assert output.debug["custom_v2_selected_intent"] == "launch"
  assert "temporary_cruise_hold" in output.debug["custom_v2_rejected_reasons"]
