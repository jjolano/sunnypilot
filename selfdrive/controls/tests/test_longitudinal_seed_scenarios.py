from types import SimpleNamespace

from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N
from openpilot.selfdrive.controls.lib.longitudinal_planner import (
  build_cruise_coast_seed_candidates,
  build_lead_loss_seed_candidates,
  build_lead_pullaway_seed_candidates,
  build_moving_lead_seed_candidates,
  build_no_lead_stop_seed_candidates,
  build_stopped_lead_seed_candidates,
)
from openpilot.selfdrive.controls.lib.longitudinal_stacks.planner_seed import PLANNER_SEED_FLOOR
from openpilot.selfdrive.controls.lib.longitudinal_stacks.policy import planner_seed_candidates_to_longitudinal_candidates


def make_planner(output_a_target=0.0, output_should_stop=False):
  return SimpleNamespace(
    output_a_target=output_a_target,
    output_should_stop=output_should_stop,
    allow_throttle=True,
    fcw=False,
    source="cruise",
    mpc=SimpleNamespace(source="cruise"),
    v_desired_trajectory=tuple(10.0 for _ in range(CONTROL_N)),
    a_desired_trajectory=tuple(output_a_target for _ in range(CONTROL_N)),
    j_desired_trajectory=tuple(0.0 for _ in range(CONTROL_N)),
  )


def assert_seed(candidate, *, name, intent, reason, selection, group="", should_stop=False):
  assert candidate.name == name
  assert candidate.intent == intent
  assert candidate.reason == reason
  assert candidate.selection == selection
  assert candidate.group == group
  assert candidate.output.should_stop is should_stop
  assert candidate.output.seed_intent == intent
  assert candidate.output.seed_reason == reason
  assert candidate.output.debug["planner_seed_candidate_reason"] == reason


def test_no_lead_stop_helper_emits_stable_metadata():
  seeds = build_no_lead_stop_seed_candidates(
    make_planner(0.2), False, (-2.0, 2.0),
    engage_bootstrap_active=True,
    engage_bootstrap_a_target=-1.0,
    engage_bootstrap_should_stop=True,
  )

  assert len(seeds) == 1
  assert_seed(
    seeds[0], name="engage_stop_bootstrap", intent="stop_approach", reason="engage_model_stop_bootstrap",
    selection="cap", should_stop=True,
  )


def test_stopped_lead_helper_preserves_group_and_floor_selection():
  seeds = build_stopped_lead_seed_candidates(
    make_planner(-0.5), True, (-2.0, 2.0),
    creep_to_stop_gap_a_target=0.1,
    creep_to_stop_gap_should_stop=False,
    creep_to_stop_gap_selection=PLANNER_SEED_FLOOR,
    creep_to_stop_gap_accel_max=0.18,
  )

  assert [seed.name for seed in seeds] == ["creep_to_stop_gap", "creep_to_stop_gap_accel_cap"]
  assert_seed(seeds[0], name="creep_to_stop_gap", intent="lead_follow", reason="creep_to_stop_gap",
              selection=PLANNER_SEED_FLOOR, group="creep_to_stop_gap")
  assert_seed(seeds[1], name="creep_to_stop_gap_accel_cap", intent="lead_follow", reason="creep_to_stop_gap_accel_cap",
              selection="cap", group="creep_to_stop_gap")


def test_lead_pullaway_helper_preserves_stop_release_cap_suppression():
  seeds = build_lead_pullaway_seed_candidates(
    make_planner(-0.5), True, (-2.0, 2.0),
    creep_pullaway_launch_floor=0.7,
    pullaway_accel_step_floor=0.3,
    pullaway_accel_step_cap=0.6,
    pullaway_step_cap_suppressed_for_stop_release=True,
  )

  assert [seed.name for seed in seeds] == ["creep_pullaway_launch", "low_speed_pullaway_accel_step_floor"]
  assert_seed(seeds[0], name="creep_pullaway_launch", intent="launch", reason="creep_pullaway_launch",
              selection=PLANNER_SEED_FLOOR, group="creep_pullaway_launch")


def test_moving_lead_helper_marks_slew_floor_group():
  seeds = build_moving_lead_seed_candidates(
    make_planner(-1.0), True, (-2.0, 2.0),
    lead_stop_approach_slewed_a_target=-0.5,
    lead_stop_approach_base_a_target=-1.0,
  )

  assert len(seeds) == 1
  assert_seed(seeds[0], name="lead_stop_approach_slew", intent="lead_follow", reason="lead_stop_approach_slew",
              selection=PLANNER_SEED_FLOOR, group="lead_stop_approach_slew")


def test_lead_loss_and_cruise_helpers_keep_policy_conversion_debug():
  lead_loss = build_lead_loss_seed_candidates(
    make_planner(-1.0), True, (-2.0, 2.0), lead_loss_e2e_guard_a_target=-0.4,
  )
  cruise = build_cruise_coast_seed_candidates(
    make_planner(-1.0), False, (-2.0, 2.0), active=True, a_target=-0.3,
  )
  converted = planner_seed_candidates_to_longitudinal_candidates((*lead_loss, *cruise), v_target=12.0)

  assert [seed.name for seed in (*lead_loss, *cruise)] == ["lead_loss_e2e_guard", "cruise_coast"]
  assert {candidate.debug["custom_v2_seed_candidate"] for candidate in converted} == {"lead_loss_e2e_guard", "cruise_coast"}
  assert {candidate.debug["custom_v2_seed_context"] for candidate in converted} == {"planner"}
