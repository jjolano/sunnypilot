import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path

import pytest

from openpilot.tools.drive_lab import fuzz_lateral_closed_loop
from openpilot.tools.drive_lab.fuzz_lateral_closed_loop import (
  ClosedLoopResult,
  ClosedLoopThresholds,
  _run_closed_loop,
  _unexpected_plant_failures,
  generate_closed_loop_scenarios,
  main,
  replay_artifact,
  write_artifact,
)


# ---------- determinism ----------


def test_generate_closed_loop_scenarios_is_seeded():
  assert generate_closed_loop_scenarios(seed=42, cases=10) == generate_closed_loop_scenarios(seed=42, cases=10)


def test_run_closed_loop_is_deterministic():
  scenarios = generate_closed_loop_scenarios(seed=7, cases=1, kind="high_quality_path")
  scenario = scenarios[0]
  first = _run_closed_loop(scenario)
  second = _run_closed_loop(scenario)
  assert first.valid == second.valid
  assert [f["check"] for f in first.demand_failures] == [f["check"] for f in second.demand_failures]
  assert [f.check for f in first.plant_failures] == [f.check for f in second.plant_failures]


# ---------- default kinds ----------


def test_default_kinds_exclude_lateral_maneuver_override():
  scenarios = generate_closed_loop_scenarios(seed=1, cases=100)
  kinds = {s.kind for s in scenarios}
  assert "lateral_maneuver_override" not in kinds
  assert kinds <= set(fuzz_lateral_closed_loop.DEFAULT_KINDS)


# ---------- high quality path ----------


def test_high_quality_path_closed_loop_passes():
  scenarios = generate_closed_loop_scenarios(seed=1, cases=1, kind="high_quality_path", duration_s=2.0)
  result = _run_closed_loop(scenarios[0])
  assert result.valid
  assert not result.plant_skipped
  assert result.plant_result is not None
  assert result.plant_evaluation is not None


def test_lateral_maneuver_override_allows_expected_tracking_transient():
  scenarios = generate_closed_loop_scenarios(seed=1, cases=1, kind="lateral_maneuver_override", duration_s=2.0)
  result = _run_closed_loop(scenarios[0])
  assert result.valid
  assert not result.plant_skipped
  assert result.plant_evaluation is not None
  assert any(metric.name == "max_abs_tracking_error" and not metric.passed for metric in result.plant_evaluation.metrics)


def test_lateral_maneuver_override_keeps_structural_plant_failures_hard():
  from openpilot.tools.drive_lab.metrics import ScenarioFailure

  failures = [
    ScenarioFailure("tracking", "expected override step lag"),
    ScenarioFailure("lateral_jerk", "structural instability"),
  ]

  filtered = _unexpected_plant_failures("lateral_maneuver_override", failures)

  assert [failure.check for failure in filtered] == ["lateral_jerk"]


def test_iso_3888_lane_change_allows_expected_tracking_transient():
  from openpilot.tools.drive_lab.metrics import ScenarioFailure

  failures = [ScenarioFailure("tracking", "expected ISO 3888 plant lag")]
  filtered = _unexpected_plant_failures("iso_3888_lane_change", failures)

  assert filtered == []


def test_iso_3888_lane_change_keeps_structural_failures_hard():
  from openpilot.tools.drive_lab.metrics import ScenarioFailure

  failures = [
    ScenarioFailure("tracking", "expected ISO 3888 plant lag"),
    ScenarioFailure("lateral_jerk", "structural instability"),
  ]

  filtered = _unexpected_plant_failures("iso_3888_lane_change", failures)

  assert [failure.check for failure in filtered] == ["lateral_jerk"]


def test_non_iso_tracking_failure_still_fails():
  from openpilot.tools.drive_lab.metrics import ScenarioFailure

  failures = [ScenarioFailure("tracking", "should not be filtered for non-ISO kind")]
  filtered = _unexpected_plant_failures("high_quality_path", failures)

  assert [failure.check for failure in filtered] == ["tracking"]


def test_long_low_lane_confidence_does_not_fail_only_from_duration_scaled_oscillation_count():
  scenarios = generate_closed_loop_scenarios(seed=25, cases=1, duration_s=12.0)
  scenario = scenarios[0]

  result = _run_closed_loop(scenario)

  assert scenario.kind == "low_lane_confidence"
  assert result.valid


# ---------- demand failure skips plant ----------


def test_demand_failure_skips_plant_scoring():
  from openpilot.tools.drive_lab.fuzz_lateral_demand import DemandScenario

  # Empty frames should make evaluate_scenario return no outputs and fail.
  scenario = DemandScenario(kind="high_quality_path", title="empty", duration_s=0.0, frames=())
  result = _run_closed_loop(scenario)
  assert not result.valid
  assert result.plant_skipped
  assert result.plant_result is None
  assert result.plant_failures
  assert result.plant_failures[0]["check"] == "plant_skipped_due_to_demand_failure"


def test_length_mismatch_skips_plant():
  from openpilot.tools.drive_lab.fuzz_lateral_demand import DemandScenario

  # Construct a scenario whose frame count won't match a fabricated demand result.
  scenarios = generate_closed_loop_scenarios(seed=1, cases=1, kind="high_quality_path", duration_s=0.5)
  scenario = scenarios[0]
  # Monkeypatch evaluate_scenario to return a result with mismatched output length.
  original_evaluate = fuzz_lateral_closed_loop.evaluate_scenario
  base_result = original_evaluate(scenario)
  fake_demand = base_result.__class__(
    scenario=base_result.scenario,
    outputs=base_result.outputs[:5],
    valid=True,
    failures=[],
    metrics=base_result.metrics,
  )

  def _fake_evaluate(scenario):
    return fake_demand

  fuzz_lateral_closed_loop.evaluate_scenario = _fake_evaluate
  try:
    result = _run_closed_loop(scenario)
  finally:
    fuzz_lateral_closed_loop.evaluate_scenario = original_evaluate

  assert result.plant_skipped
  assert result.plant_failures
  assert result.plant_failures[0]["check"] == "plant_skipped_due_to_demand_failure"


def test_one_frame_scenario_skips_plant_without_resampling():
  from openpilot.tools.drive_lab.fuzz_lateral_demand import DemandScenario

  scenario = generate_closed_loop_scenarios(seed=1, cases=1, kind="high_quality_path", duration_s=0.5)[0]
  one_frame = DemandScenario(scenario.kind, "one frame", 0.0, scenario.frames[:1], scenario.thresholds)

  result = _run_closed_loop(one_frame)

  assert not result.valid
  assert result.plant_skipped
  assert result.plant_result is None
  assert result.plant_failures[0]["check"] == "plant_skipped_due_to_demand_failure"


# ---------- layer-separated failure reporting ----------


def test_plant_failure_reported_separately():
  scenarios = generate_closed_loop_scenarios(seed=1, cases=1, kind="high_quality_path", duration_s=2.0)
  scenario = scenarios[0]
  base_result = _run_closed_loop(scenario)
  assert base_result.valid

  # Inject a plant failure while leaving demand valid.
  fake_plant_evaluation = base_result.plant_evaluation
  from openpilot.tools.drive_lab.metrics import ScenarioFailure
  fake_evaluation = fake_plant_evaluation.__class__(
    scenario_id=fake_plant_evaluation.scenario_id,
    valid=False,
    failures=[ScenarioFailure("injected_plant", "injected plant failure")],
    metrics=fake_plant_evaluation.metrics,
  )

  original_evaluate_lateral_trace = fuzz_lateral_closed_loop.evaluate_lateral_trace
  fuzz_lateral_closed_loop.evaluate_lateral_trace = lambda *a, **k: fake_evaluation
  try:
    result = _run_closed_loop(scenario)
  finally:
    fuzz_lateral_closed_loop.evaluate_lateral_trace = original_evaluate_lateral_trace

  assert not result.valid
  assert not result.demand_failures
  assert [f.check for f in result.plant_failures] == ["injected_plant"]


# ---------- artifact / replay ----------


def test_artifact_replay_reports_equivalent_classification():
  scenarios = generate_closed_loop_scenarios(seed=9, cases=1, kind="curvature_jump", duration_s=2.0)
  scenario = scenarios[0]
  result = _run_closed_loop(scenario)
  thresholds = ClosedLoopThresholds()
  with tempfile.TemporaryDirectory() as tmpdir:
    path = write_artifact(result, thresholds, Path(tmpdir), seed=9, index=0)
    replayed = replay_artifact(path)
    assert replayed.valid == result.valid
    assert [f["check"] for f in replayed.demand_failures] == [f["check"] for f in result.demand_failures]
    assert [f.check for f in replayed.plant_failures] == [f.check for f in result.plant_failures]


def test_artifact_contains_full_demand_frames_and_is_strict_json_safe():
  scenarios = generate_closed_loop_scenarios(seed=2, cases=1, kind="high_quality_path", duration_s=1.0)
  scenario = scenarios[0]
  result = _run_closed_loop(scenario)
  thresholds = ClosedLoopThresholds()
  with tempfile.TemporaryDirectory() as tmpdir:
    path = write_artifact(result, thresholds, Path(tmpdir), seed=2, index=0)
    raw = path.read_text()
    payload = json.loads(raw)
    assert payload["schema"] == "drive-lab-lateral-closed-loop-fuzzer-artifact"
    assert len(payload["scenario"]["frames"]) == len(scenario.frames)
    assert "nan" not in raw.lower()
    assert "inf" not in raw.lower()


# ---------- CLI smoke ----------


def test_main_text_output_runs_and_reports_zero_failures():
  stdout = io.StringIO()
  previous_argv = sys.argv
  try:
    sys.argv = ["fuzz_lateral_closed_loop.py", "--seed", "1", "--cases", "5", "--kind", "high_quality_path"]
    with contextlib.redirect_stdout(stdout):
      main()
  finally:
    sys.argv = previous_argv

  output = stdout.getvalue()
  assert "Drive Lab lateral closed-loop fuzz" in output
  assert "failures=0" in output


def test_main_json_output_is_stable():
  stdout = io.StringIO()
  previous_argv = sys.argv
  try:
    sys.argv = ["fuzz_lateral_closed_loop.py", "--seed", "2", "--cases", "3", "--json"]
    with contextlib.redirect_stdout(stdout):
      main()
  finally:
    sys.argv = previous_argv

  payload = json.loads(stdout.getvalue())
  assert payload["seed"] == 2
  assert payload["cases"] == 3
  assert "failures" in payload


def test_iso_3888_preset_closed_loop_passes():
  stdout = io.StringIO()
  previous_argv = sys.argv
  try:
    sys.argv = ["fuzz_lateral_closed_loop.py", "--preset", "iso-3888", "--seed", "42", "--cases", "50"]
    with contextlib.redirect_stdout(stdout):
      main()
  finally:
    sys.argv = previous_argv

  output = stdout.getvalue()
  assert "preset=iso-3888" in output
  assert "failures=0" in output


def test_main_exits_nonzero_on_injected_failure():
  scenarios = generate_closed_loop_scenarios(seed=1, cases=1, kind="high_quality_path")
  baseline = _run_closed_loop(scenarios[0])
  assert baseline.valid
  failed = ClosedLoopResult(
    scenario=baseline.scenario,
    demand_result=baseline.demand_result,
    plant_result=baseline.plant_result,
    plant_evaluation=baseline.plant_evaluation,
    demand_failures=baseline.demand_failures,
    plant_failures=[{"check": "injected", "detail": "injected closed-loop failure"}],
    plant_skipped=False,
  )

  original_run = fuzz_lateral_closed_loop._run_closed_loop
  fuzz_lateral_closed_loop._run_closed_loop = lambda scenario, thresholds=None: failed
  previous_argv = sys.argv
  try:
    sys.argv = ["fuzz_lateral_closed_loop.py", "--seed", "1", "--cases", "1"]
    with pytest.raises(SystemExit) as exc:
      main()
    assert exc.value.code == 1
  finally:
    fuzz_lateral_closed_loop._run_closed_loop = original_run
    sys.argv = previous_argv
