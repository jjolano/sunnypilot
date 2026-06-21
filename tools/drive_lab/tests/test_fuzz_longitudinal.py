import contextlib
import io
import sys

import numpy as np

from openpilot.tools.drive_lab import fuzz_longitudinal
from openpilot.tools.drive_lab.fuzz_longitudinal import (
  Scenario,
  aggregate_mpc_solution_status_counts,
  capture_commanded_accel,
  evaluate_collision_response,
  evaluate_invariants,
  evaluate_lead_pullaway_start,
  generate_openpilot_acc_scenarios,
  generate_scenarios,
  generate_udacity_acc_scenarios,
  render_maneuver_snippet,
  run_scenario,
  scenario_maneuver_kwargs,
  scenario_to_spec,
  shipped_longitudinal_config,
)
from openpilot.tools.drive_lab.log_profile import LongitudinalProfile, ProfileRange
from openpilot.tools.drive_lab.longitudinal_scenarios import (
  SCENARIO_PRESETS,
  generate_preset_scenarios,
  PresetRequest,
)
from openpilot.tools.drive_lab.ncap_acc_scenarios import generate_ncap_acc_scenarios
from openpilot.tools.drive_lab.commonroad_acc import generate_commonroad_acc_scenarios


def test_generate_scenarios_is_seeded():
  assert generate_scenarios(seed=42, cases=5) == generate_scenarios(seed=42, cases=5)


def test_comfort_stopped_lead_decel_is_plausible():
  scenarios = [s for s in generate_scenarios(seed=4, cases=100, mode="comfort") if s.kind == "stopped_lead_approach"]

  assert scenarios
  for scenario in scenarios:
    speeds = scenario.kwargs["speed_lead_values"]
    breakpoints = scenario.kwargs["breakpoints"]
    lead_decel = (speeds[1] - speeds[0]) / (breakpoints[1] - breakpoints[0])
    assert -3.6 <= lead_decel <= -1.4


def test_comfort_cut_in_is_not_already_impossible_when_detected():
  scenarios = [s for s in generate_scenarios(seed=5, cases=100, mode="comfort") if s.kind == "slower_cut_in"]

  assert scenarios
  for scenario in scenarios:
    v_ego = scenario.kwargs["initial_speed"]
    v_lead = scenario.kwargs["speed_lead_values"][0]
    cut_in_time = scenario.kwargs["breakpoints"][1]
    detected_gap = scenario.kwargs["initial_distance_lead"] - max(0.0, v_ego - v_lead) * cut_in_time
    assert detected_gap >= max(24.0, v_ego * 1.45)


def test_profile_biases_generated_ranges():
  profile = LongitudinalProfile(
    source="test",
    sample_count=100,
    ego_speed=ProfileRange(12.0, 13.0),
    cruise_speed=ProfileRange(6.0, 7.0),
    lead_gap=ProfileRange(35.0, 40.0),
    closing_speed=ProfileRange(1.0, 1.5),
    lead_decel=ProfileRange(2.0, 2.5),
    stopped_lead_gap=ProfileRange(5.0, 6.0),
    lead_pullaway_speed=ProfileRange(1.5, 2.0),
  )
  scenarios = generate_scenarios(seed=1, cases=100, mode="comfort", profile=profile)

  pullaways = [s for s in scenarios if s.kind == "lead_pullaway"]
  assert pullaways
  for scenario in pullaways:
    assert 5.0 <= scenario.kwargs["initial_distance_lead"] <= 6.0
    assert 1.5 <= scenario.kwargs["speed_lead_values"][2] <= 2.0


def test_udacity_acc_scenarios_cover_all_cases():
  scenarios = generate_udacity_acc_scenarios()

  assert len(scenarios) == 15
  kinds = {scenario.kind for scenario in scenarios}
  assert "udacity_acc_cruise_speed_decrease" in kinds
  assert "udacity_acc_grade_downhill" in kinds
  assert "udacity_acc_approach_from_stop" in kinds
  assert "udacity_acc_accel_while_lead_decel_hard" in kinds
  for scenario in scenarios:
    assert scenario.mode == "comfort"
    assert scenario.title.startswith("udacity acc inspired")
    assert scenario.duration > 0.0
    if "breakpoints" in scenario.kwargs:
      assert scenario.kwargs["breakpoints"] == sorted(scenario.kwargs["breakpoints"])


def test_udacity_lead_decel_stop_uses_regression_oracle():
  scenarios = {scenario.kind: scenario for scenario in generate_udacity_acc_scenarios()}

  assert scenarios["udacity_acc_lead_decel_to_stop"].oracle_profile == "regression"
  assert scenarios["udacity_acc_lead_decel_to_stop_2ms2"].oracle_profile == "regression"
  assert scenarios["udacity_acc_stopped_lead"].oracle_profile == "comfort"


def test_udacity_lead_decel_stop_regression_oracle_passes_comfort_gate():
  scenarios = [
    scenario for scenario in generate_udacity_acc_scenarios()
    if scenario.kind in {"udacity_acc_lead_decel_to_stop", "udacity_acc_lead_decel_to_stop_2ms2"}
  ]

  with shipped_longitudinal_config():
    results = [run_scenario(scenario, max_normal_jerk=8.0) for scenario in scenarios]

  assert [(result.scenario.kind, result.failures) for result in results] == [
    ("udacity_acc_lead_decel_to_stop", []),
    ("udacity_acc_lead_decel_to_stop_2ms2", []),
  ]


def test_openpilot_acc_preset_count():
  scenarios = generate_openpilot_acc_scenarios()
  assert len(scenarios) == 15
  assert any(s.kind == "openpilot_resume_from_stop" for s in scenarios)


def test_openpilot_stopped_lead_variants_use_regression_oracle():
  scenarios = {scenario.title: scenario for scenario in generate_openpilot_acc_scenarios()}

  for title in (
    "approach stopped car at 25m/s, initial distance: 120m",
    "approach stopped car at 20m/s, initial distance 90m",
    "approach stopped car at 20m/s, with prob_lead_values",
    "approach stopped car at 20m/s, with prob_throttle_values and pitch = -0.1",
    "approach stopped car at 20m/s, with prob_throttle_values and pitch = +0.1",
  ):
    assert scenarios[title].oracle_profile == "regression"


def test_openpilot_stopped_lead_prob_variant_passes_regression_gate():
  scenario = next(
    scenario for scenario in generate_openpilot_acc_scenarios()
    if scenario.title == "approach stopped car at 20m/s, with prob_lead_values"
  )

  with shipped_longitudinal_config():
    result = run_scenario(scenario, max_normal_jerk=8.0)

  assert result.failures == []


def test_ncap_acc_curated_count():
  scenarios = generate_ncap_acc_scenarios()
  assert len(scenarios) == 18
  assert all(s.oracle_profile == "safety" for s in scenarios)


def test_ncap_ccrm_distance_scales_with_closing_speed():
  scenarios = [scenario for scenario in generate_ncap_acc_scenarios() if scenario.kind.startswith("ncap_ccrm_")]

  for scenario in scenarios:
    ego_speed = scenario.kwargs["initial_speed"]
    target_speed = scenario.kwargs["speed_lead_values"][0]
    relative_speed = max(0.0, ego_speed - target_speed)
    assert scenario.kwargs["initial_distance_lead"] >= max(80.0, relative_speed * 4.5)


def test_ncap_fast_ccrm_cases_pass_safety_gate():
  scenarios = [
    scenario for scenario in generate_ncap_acc_scenarios()
    if scenario.kind in {"ncap_ccrm_110_20", "ncap_ccrm_120_20", "ncap_ccrm_130_20"}
  ]

  with shipped_longitudinal_config():
    results = [run_scenario(scenario, max_normal_jerk=8.0) for scenario in scenarios]

  assert [(result.scenario.kind, result.failures) for result in results] == [
    ("ncap_ccrm_110_20", []),
    ("ncap_ccrm_120_20", []),
    ("ncap_ccrm_130_20", []),
  ]


def test_ncap_acc_sample_family():
  scenarios = generate_ncap_acc_scenarios(mode="comfort", family="CCRs", sample=2, seed=1)
  assert len(scenarios) == 2
  assert all(s.kind.startswith("ncap_ccrs_") for s in scenarios)


def test_commonroad_acc_fixtures():
  scenarios = generate_commonroad_acc_scenarios()
  assert len(scenarios) == 4
  assert scenarios[0].kind.startswith("commonroad_")


def test_all_presets_registered():
  assert set(SCENARIO_PRESETS) == {"fuzz", "udacity-acc", "openpilot-acc", "ncap-acc", "commonroad-acc",
                                     "nuscenes-acc", "iso15622-acc", "unr157-alks", "nhtsa-fcw", "cncap-ccrh", "iihs-acc"}


def test_main_lists_udacity_acc_preset():
  stdout = io.StringIO()
  previous_argv = sys.argv
  try:
    sys.argv = ["fuzz_longitudinal.py", "--preset", "udacity-acc", "--list-only"]
    with contextlib.redirect_stdout(stdout):
      fuzz_longitudinal.main()
  finally:
    sys.argv = previous_argv

  output = stdout.getvalue()
  assert "udacity acc inspired green light lead launch" in output
  assert "fuzz stopped lead approach" not in output


def test_evaluate_invariants_catches_collision_and_nan():
  output = np.array([
    [0.0, 0.0, 0.0, 10.0, 9.0, 0.0, 10.0],
    [0.1, 1.0, 1.0, 10.0, 9.0, np.nan, 0.2],
  ])

  failures = evaluate_invariants(True, output)

  assert [f.check for f in failures] == ["finite"]


def test_evaluate_invariants_reports_malformed_output_shape():
  failures = evaluate_invariants(True, np.zeros((2, 3)))
  assert any(f.check == "output" for f in failures)


def test_evaluate_lead_pullaway_start_detects_no_launch():
  output = np.array([
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0],
    [1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 9.0],
    [2.0, 2.0, 2.0, 0.0, 1.0, 0.0, 8.0],
  ])
  failures = evaluate_lead_pullaway_start(output)
  assert failures and failures[0].check == "launch"


def test_evaluate_collision_response_accepts_best_effort_brake():
  output = np.array([
    [0.0, 0.0, 0.0, 8.0, 0.0, -3.0, 10.0],
    [0.1, 1.0, 1.0, 4.0, 0.0, -3.0, 0.3],
  ])
  commanded = np.array([-3.0, -3.0])
  prob_lead = np.array([1.0, 1.0])
  assert not evaluate_collision_response(output, commanded, prob_lead)


def test_evaluate_collision_response_safety_profile_fails_high_impact():
  output = np.array([
    [0.0, 0.0, 0.0, 10.0, 0.0, -1.0, 10.0],
    [0.1, 1.0, 1.0, 8.0, 0.0, -1.0, 0.3],
  ])
  failures = evaluate_collision_response(output, np.array([-1.0, -1.0]), np.array([1.0, 1.0]), max_impact_speed_ms=1.0, use_best_effort=False)
  assert failures


def test_scenario_maneuver_kwargs_disables_ensure_start_for_launch_oracle():
  scenario = generate_udacity_acc_scenarios()[next(i for i, s in enumerate(generate_udacity_acc_scenarios()) if s.kind == "udacity_acc_green_light_launch")]
  kwargs = scenario_maneuver_kwargs(scenario)
  assert kwargs["ensure_start"] is False


def test_scenario_to_spec_includes_oracle_profile_in_tags():
  scenario = generate_udacity_acc_scenarios()[0]
  spec = scenario_to_spec(scenario, source="udacity-acc")
  assert "udacity-acc" in spec.tags


def test_preset_request_openpilot():
  scenarios = generate_preset_scenarios(PresetRequest(preset="openpilot-acc", cases=1))
  assert len(scenarios) == 15


def test_run_scenario_smoke():
  scenario = generate_scenarios(seed=1, cases=1)[0]
  with shipped_longitudinal_config():
    result = run_scenario(scenario)
  assert isinstance(result.valid, bool)


def test_aggregate_mpc_solution_status_counts_sums_per_status():
  dummy = Scenario(mode="comfort", kind="test", title="dummy", duration=1.0, kwargs={})
  results = [
    fuzz_longitudinal.ScenarioResult(scenario=dummy, valid=True, failures=[], mpc_solution_status_counts={4: 2, 5: 1}),
    fuzz_longitudinal.ScenarioResult(scenario=dummy, valid=True, failures=[], mpc_solution_status_counts={4: 1}),
    fuzz_longitudinal.ScenarioResult(scenario=dummy, valid=True, failures=[], mpc_solution_status_counts={}),
  ]
  assert aggregate_mpc_solution_status_counts(results) == {4: 3, 5: 1}


def test_capture_commanded_accel_records_mpc_reset_status():
  from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import LongitudinalMpc
  with capture_commanded_accel() as capture:
    mpc = LongitudinalMpc()
    mpc.solution_status = 4
    mpc.reset()
    mpc.solution_status = 4
    mpc.reset()
    mpc.solution_status = 2
    mpc.reset()
  assert capture.mpc_solution_status_counts == {4: 2, 2: 1}


def test_main_json_includes_mpc_resets(monkeypatch):
  import json
  scenarios = [Scenario(mode="comfort", kind="test", title="test scenario", duration=1.0, kwargs={})]

  def fake_run_scenario(scenario, max_normal_jerk=8.0):
    return fuzz_longitudinal.ScenarioResult(
      scenario=scenario,
      valid=True,
      failures=[],
      mpc_solution_status_counts={4: 3, 2: 1},
    )

  monkeypatch.setattr(fuzz_longitudinal, "generate_preset_scenarios", lambda request: scenarios)
  monkeypatch.setattr(fuzz_longitudinal, "run_scenario", fake_run_scenario)

  stdout = io.StringIO()
  previous_argv = sys.argv
  try:
    sys.argv = ["fuzz_longitudinal.py", "--preset", "fuzz", "--cases", "1", "--json"]
    with contextlib.redirect_stdout(stdout):
      fuzz_longitudinal.main()
  finally:
    sys.argv = previous_argv

  payload = json.loads(stdout.getvalue())
  assert payload["mpcSolutionStatusCounts"] == {"4": 3, "2": 1}
  assert payload["totalMpcResets"] == 4
  assert payload["scenarioResults"][0]["mpcSolutionStatusCounts"] == {"4": 3, "2": 1}
  assert payload["failures"] == []
