import numpy as np

from openpilot.tools.drive_lab.metrics import evaluate_maneuver_output
from openpilot.tools.drive_lab.scenario_spec import ScenarioSpec


def test_scenario_spec_round_trips_maneuver_kwargs():
  spec = ScenarioSpec.from_maneuver_kwargs(
    kind="lead_pullaway",
    title="fuzz lead pullaway #3",
    mode="comfort",
    duration=12.5,
    kwargs={
      "initial_speed": 0.0,
      "lead_relevancy": True,
      "initial_distance_lead": 7.5,
      "speed_lead_values": [0.0, 0.0, 2.5],
      "prob_lead_values": [1.0, 1.0, 1.0],
      "cruise_values": [12.0, 12.0, 12.0],
      "breakpoints": [0.0, 2.0, 8.0],
    },
    source="fuzz",
    seed=7,
    index=3,
  )

  payload = spec.to_dict()
  restored = ScenarioSpec.from_dict(payload)

  assert restored == spec
  assert spec.scenario_id == "fuzz:comfort:lead_pullaway:7:3"
  assert spec.ego["initial_speed"] == 0.0
  assert spec.actors["lead"]["initial_distance"] == 7.5
  assert spec.events == ("lead_pullaway",)
  assert spec.oracle["checks"] == ("valid", "finite", "speed", "collision", "jerk")


def test_scenario_spec_round_trips_provenance_and_legacy_payloads():
  legacy = {
    "scenario_id": "s:mode:kind:x",
    "kind": "kind",
    "title": "title",
    "mode": "mode",
    "duration": 1.0,
    "source": "src",
    "maneuver_kwargs": {},
  }
  restored_legacy = ScenarioSpec.from_dict(legacy)
  spec = ScenarioSpec.from_maneuver_kwargs("kind", "title", "mode", 1.0, {}, source="src", provenance={"route_id": "r1"})

  assert restored_legacy.provenance == {}
  assert ScenarioSpec.from_dict(spec.to_dict()) == spec
  assert spec.provenance == {"route_id": "r1"}


def test_evaluate_maneuver_output_returns_metrics_and_failures():
  output = np.array([
    [0.0, 0.0, 0.0, 4.0, 4.0, 0.0, 10.0],
    [0.1, 0.4, 0.0, 3.0, 3.0, -0.5, 0.2],
    [0.2, 0.7, 0.0, -0.1, 2.0, -2.0, 0.5],
  ])

  result = evaluate_maneuver_output("fuzz:comfort:cut_in:1:0", True, output, max_normal_jerk=4.0)

  assert result.scenario_id == "fuzz:comfort:cut_in:1:0"
  assert not result.valid
  assert [failure.check for failure in result.failures] == ["speed", "collision", "jerk"]
  assert result.metric_value("min_speed") == -0.1
  assert result.metric_value("min_lead_gap") == 0.2
  assert result.metric_value("max_abs_jerk") == 15.0
  assert result.to_dict()["metrics"]["min_speed"]["unit"] == "m/s"


def test_evaluate_maneuver_output_reports_available_metrics_for_nonfinite_output():
  output = np.array([
    [0.0, 0.0, 0.0, 4.0, 4.0, 0.0, 10.0],
    [0.1, 0.4, 0.0, 3.0, 3.0, -0.2, 5.0],
    [0.2, 0.7, 0.0, 2.0, 2.0, np.nan, 3.0],
    [0.3, 0.9, 0.0, 2.5, 2.5, -0.2, 4.0],
  ])

  result = evaluate_maneuver_output("fuzz:comfort:cut_in:1:0", True, output)

  assert [failure.check for failure in result.failures] == ["finite"]
  assert result.metric_value("min_speed") == 2.0
  assert result.metric_value("min_lead_gap") == 3.0
  assert result.metric_value("max_abs_jerk") == 2.0
