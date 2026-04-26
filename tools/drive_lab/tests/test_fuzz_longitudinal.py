import numpy as np

from openpilot.tools.drive_lab.fuzz_longitudinal import evaluate_invariants, generate_scenarios, render_maneuver_snippet


def test_generate_scenarios_is_seeded():
  assert generate_scenarios(seed=42, cases=5) == generate_scenarios(seed=42, cases=5)


def test_evaluate_invariants_catches_collision_and_nan():
  output = np.array([
    [0.0, 0.0, 0.0, 10.0, 9.0, 0.0, 10.0],
    [0.1, 1.0, 1.0, 10.0, 9.0, np.nan, 0.2],
  ])

  failures = evaluate_invariants(True, output)

  assert [f.check for f in failures] == ["finite"]


def test_render_maneuver_snippet_contains_replayable_fields():
  scenario = generate_scenarios(seed=1, cases=1)[0]

  snippet = render_maneuver_snippet(scenario)

  assert "Maneuver(" in snippet
  assert "duration=" in snippet
  assert "initial_speed" in snippet
