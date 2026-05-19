import contextlib
import io
import sys

import numpy as np

from openpilot.tools.drive_lab import fuzz_longitudinal
from openpilot.tools.drive_lab.fuzz_longitudinal import (
  Scenario,
  evaluate_lead_pullaway_start,
  evaluate_invariants,
  generate_scenarios,
  generate_udacity_acc_scenarios,
  render_maneuver_snippet,
  scenario_maneuver_kwargs,
  scenario_to_spec,
)
from openpilot.tools.drive_lab.log_profile import LongitudinalProfile, ProfileRange


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


def test_udacity_acc_scenarios_cover_legacy_drive_lab_cases():
  scenarios = generate_udacity_acc_scenarios()

  assert len(scenarios) == 8
  assert {scenario.kind for scenario in scenarios} == {
    "udacity_acc_cruise_speed_step",
    "udacity_acc_grade_change",
    "udacity_acc_slower_lead",
    "udacity_acc_stopped_lead",
    "udacity_acc_lead_decel_to_stop",
    "udacity_acc_oscillating_lead",
    "udacity_acc_stop_and_go",
    "udacity_acc_green_light_launch",
  }
  for scenario in scenarios:
    assert scenario.mode == "comfort"
    assert scenario.title.startswith("udacity acc inspired")
    assert scenario.duration > 0.0
    assert scenario.kwargs["breakpoints"] == sorted(scenario.kwargs["breakpoints"])
    for key, value in scenario.kwargs.items():
      if key.endswith("_values"):
        assert len(value) == len(scenario.kwargs["breakpoints"])


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
  output = np.zeros((2, 6))

  failures = evaluate_invariants(True, output)

  assert [f.check for f in failures] == ["output"]
  assert "expected maneuver output" in failures[0].detail


def test_lead_pullaway_start_check_accepts_started_then_settled_follow():
  output = np.array([
    [5.8, 0.0, 6.4, 0.00, 0.0, 0.00, 6.4],
    [6.0, 0.0, 6.5, 0.10, 0.5, 0.55, 6.5],
    [7.0, 0.5, 7.4, 0.60, 1.3, 0.55, 6.9],
    [12.5, 6.8, 13.8, 1.29, 1.3, -0.07, 7.0],
  ])

  assert evaluate_lead_pullaway_start(output) == []


def test_lead_pullaway_start_check_flags_never_starting():
  output = np.array([
    [0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 6.0],
    [1.0, 0.0, 7.0, 0.0, 1.0, 0.0, 7.0],
    [2.0, 0.0, 8.0, 0.0, 1.0, 0.0, 8.0],
  ])

  failures = evaluate_lead_pullaway_start(output)

  assert [failure.check for failure in failures] == ["launch"]


def test_lead_pullaway_fuzzer_uses_bounded_start_oracle_instead_of_legacy_ensure_start():
  scenario = Scenario(
    "comfort",
    "lead_pullaway",
    "lead pullaway",
    10.0,
    {
      "lead_relevancy": True,
      "ensure_start": True,
      "initial_speed": 0.0,
    },
  )

  kwargs = scenario_maneuver_kwargs(scenario)

  assert kwargs["ensure_start"] is False


def test_render_maneuver_snippet_contains_replayable_fields():
  scenario = generate_scenarios(seed=1, cases=1)[0]

  snippet = render_maneuver_snippet(scenario)

  assert "Maneuver(" in snippet
  assert "duration=" in snippet
  assert "initial_speed" in snippet


def test_scenario_to_spec_preserves_fuzzer_context():
  scenario = generate_scenarios(seed=1, cases=1, mode="comfort")[0]

  spec = scenario_to_spec(scenario, source="fuzz", seed=1, index=0)

  assert spec.scenario_id == f"fuzz:comfort:{scenario.kind}:1:0"
  assert spec.kind == scenario.kind
  assert spec.title == scenario.title
  assert spec.mode == scenario.mode
  assert spec.duration == scenario.duration
  assert spec.maneuver_kwargs == scenario.kwargs


def test_scenario_to_dict_can_include_spec_metadata():
  scenario = generate_scenarios(seed=1, cases=1, mode="comfort")[0]

  payload = fuzz_longitudinal.scenario_to_dict(scenario, source="fuzz", seed=1, index=0)

  assert payload["mode"] == scenario.mode
  assert payload["kind"] == scenario.kind
  assert payload["kwargs"] == scenario.kwargs
  assert payload["scenarioId"] == f"fuzz:comfort:{scenario.kind}:1:0"
  assert payload["spec"]["events"] == [scenario.kind]
  assert payload["spec"]["oracle"]["checks"] == ["valid", "finite", "speed", "collision", "jerk"]
