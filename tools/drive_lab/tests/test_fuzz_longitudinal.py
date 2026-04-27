import numpy as np

from openpilot.tools.drive_lab.fuzz_longitudinal import evaluate_invariants, generate_scenarios, render_maneuver_snippet
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


def test_render_maneuver_snippet_contains_replayable_fields():
  scenario = generate_scenarios(seed=1, cases=1)[0]

  snippet = render_maneuver_snippet(scenario)

  assert "Maneuver(" in snippet
  assert "duration=" in snippet
  assert "initial_speed" in snippet
