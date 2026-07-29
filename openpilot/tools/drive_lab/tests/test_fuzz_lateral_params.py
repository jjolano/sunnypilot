import contextlib
import io
import json
import math
import sys
import tempfile
from pathlib import Path

import pytest

from openpilot.tools.drive_lab import fuzz_lateral_params
from openpilot.tools.drive_lab.fuzz_lateral_params import (
  ParamFuzzerConfig,
  ParamFrame,
  ParamResult,
  ParamScenario,
  ParamThresholds,
  evaluate_scenario,
  generate_scenarios,
  main,
  replay_artifact,
  scenario_from_dict,
  scenario_to_dict,
  write_artifact,
)


# ---------- determinism ----------


def test_generate_scenarios_is_seeded():
  assert generate_scenarios(ParamFuzzerConfig(seed=42, cases=20)) == generate_scenarios(ParamFuzzerConfig(seed=42, cases=20))


def test_evaluate_scenario_is_deterministic():
  config = ParamFuzzerConfig(seed=7, cases=1, kind="demand_enable_cycle")
  scenario = generate_scenarios(config)[0]
  first = evaluate_scenario(scenario)
  second = evaluate_scenario(scenario)
  assert first.valid == second.valid
  assert [f["check"] for f in first.structural_failures] == [f["check"] for f in second.structural_failures]
  assert [f["check"] for f in first.event_failures] == [f["check"] for f in second.event_failures]


# ---------- demand enable cycle ----------


def test_demand_enable_cycle_disables_and_reenables():
  config = ParamFuzzerConfig(seed=1, cases=1, kind="demand_enable_cycle")
  scenario = generate_scenarios(config)[0]
  result = evaluate_scenario(scenario)
  assert result.valid
  assert result.outputs[0].enabled
  disabled = [i for i, o in enumerate(result.outputs) if not o.enabled]
  assert disabled
  assert all(math.isclose(result.outputs[i].output_curvature, result.outputs[i].raw_curvature, abs_tol=1e-9) for i in disabled)
  reenabled = [i for i, o in enumerate(result.outputs) if o.enabled and i > disabled[0]]
  assert reenabled


# ---------- params refresh cadence ----------


def test_params_refresh_cadence_changes_at_frame_99():
  config = ParamFuzzerConfig(seed=1, cases=1, kind="params_refresh_cadence")
  scenario = generate_scenarios(config)[0]
  result = evaluate_scenario(scenario)
  assert result.valid
  refresh_frames = [i for i, o in enumerate(result.outputs) if o.refresh_frame]
  assert refresh_frames
  first_refresh = refresh_frames[0]
  assert first_refresh == 99
  assert all(o.enabled for o in result.outputs[:first_refresh])
  assert not any(o.lane_centering_assist_enabled for o in result.outputs[:first_refresh])
  assert any(o.lane_centering_assist_enabled for o in result.outputs[first_refresh:])


def test_refresh_scenarios_auto_extend_short_duration():
  for kind in ("demand_enable_cycle", "params_refresh_cadence", "params_read_fault"):
    scenario = generate_scenarios(ParamFuzzerConfig(seed=1, cases=1, kind=kind, duration_s=0.2))[0]
    result = evaluate_scenario(scenario)
    assert result.valid, f"{kind} failed: {result.structural_failures} {result.event_failures}"
    assert any(o.refresh_frame for o in result.outputs)


# ---------- params read fault ----------


def test_params_read_fault_fail_closed():
  config = ParamFuzzerConfig(seed=1, cases=1, kind="params_read_fault")
  scenario = generate_scenarios(config)[0]
  result = evaluate_scenario(scenario)
  assert result.valid
  fault_frames = [i for i, f in enumerate(scenario.frames) if f.fault_on_refresh]
  assert fault_frames
  fault_frame = fault_frames[0]
  assert not result.outputs[fault_frame].enabled
  assert math.isclose(result.outputs[fault_frame].output_curvature, result.outputs[fault_frame].raw_curvature, abs_tol=1e-9)
  after_fault = result.outputs[fault_frame:]
  assert after_fault
  assert all(not o.enabled for o in after_fault[:20])
  assert all(math.isclose(o.output_curvature, o.raw_curvature, abs_tol=1e-9) for o in after_fault[:20])


# ---------- missing model fail closed ----------


def test_missing_model_fail_closed_passthrough():
  config = ParamFuzzerConfig(seed=1, cases=1, kind="missing_model_fail_closed")
  scenario = generate_scenarios(config)[0]
  result = evaluate_scenario(scenario)
  assert result.valid
  missing = [i for i, f in enumerate(scenario.frames) if f.model_data is None]
  assert missing
  assert all(math.isclose(result.outputs[i].output_curvature, result.outputs[i].raw_curvature, abs_tol=1e-9) for i in missing)


# ---------- toggle matrix ----------


def test_toggle_matrix_covers_all_combos_and_passthrough_when_disabled():
  config = ParamFuzzerConfig(seed=1, cases=1, kind="toggle_matrix")
  scenario = generate_scenarios(config)[0]
  result = evaluate_scenario(scenario)
  assert result.valid
  desired_combos = {tuple(sorted(f.desired_params.items())) for f in scenario.frames}
  assert len(desired_combos) == 4
  observed_combos = {(o.enabled, o.lane_centering_assist_enabled) for o in result.outputs}
  assert len(observed_combos) == 4
  observed_disabled = [i for i, o in enumerate(result.outputs) if not o.enabled]
  assert observed_disabled
  assert all(math.isclose(result.outputs[i].output_curvature, result.outputs[i].raw_curvature, abs_tol=1e-9) for i in observed_disabled)


# ---------- replay / artifact ----------


def test_scenario_round_trips_through_dict():
  config = ParamFuzzerConfig(seed=3, cases=1)
  scenario = generate_scenarios(config)[0]
  restored = scenario_from_dict(scenario_to_dict(scenario))
  assert restored.kind == scenario.kind
  assert restored.title == scenario.title
  assert restored.index == scenario.index
  assert len(restored.frames) == len(scenario.frames)


def test_artifact_replay_reports_equivalent_classification():
  config = ParamFuzzerConfig(seed=9, cases=1, kind="params_refresh_cadence")
  scenario = generate_scenarios(config)[0]
  result = evaluate_scenario(scenario)
  with tempfile.TemporaryDirectory() as tmpdir:
    path = write_artifact(result, Path(tmpdir), seed=9, index=0)
    replayed = replay_artifact(path)
    assert replayed.valid == result.valid
    assert [f["check"] for f in replayed.structural_failures] == [f["check"] for f in result.structural_failures]
    assert [f["check"] for f in replayed.event_failures] == [f["check"] for f in result.event_failures]


def test_artifact_contains_full_frames_and_is_strict_json_safe():
  config = ParamFuzzerConfig(seed=2, cases=1, kind="toggle_matrix")
  scenario = generate_scenarios(config)[0]
  result = evaluate_scenario(scenario)
  with tempfile.TemporaryDirectory() as tmpdir:
    path = write_artifact(result, Path(tmpdir), seed=2, index=0)
    raw = path.read_text()
    payload = json.loads(raw)
    assert payload["schema"] == "drive-lab-lateral-params-fuzzer-artifact"
    assert len(payload["scenario"]["frames"]) == len(scenario.frames)
    assert "nan" not in raw.lower()
    assert "inf" not in raw.lower()


def test_failure_artifact_writes_and_replays_equivalent_classification():
  config = ParamFuzzerConfig(seed=1, cases=1, kind="demand_enable_cycle")
  scenario = generate_scenarios(config)[0]
  failed_scenario = ParamScenario(
    kind=scenario.kind,
    title=scenario.title,
    index=scenario.index,
    frames=scenario.frames,
    event_windows=scenario.event_windows,
    thresholds=ParamThresholds(max_abs_output_curvature=1e-6),
  )
  result = evaluate_scenario(failed_scenario)
  assert result.structural_failures
  with tempfile.TemporaryDirectory() as tmpdir:
    path = write_artifact(result, Path(tmpdir), seed=1, index=0)
    replayed = replay_artifact(path)
    assert not replayed.valid
    assert [f["check"] for f in replayed.structural_failures] == [f["check"] for f in result.structural_failures]


# ---------- empty scenario ----------


def test_empty_scenario_returns_structured_failure():
  scenario = ParamScenario("empty", "empty", 0, (), event_windows=())
  result = evaluate_scenario(scenario)
  assert not result.valid
  assert [f["check"] for f in result.structural_failures] == ["output"]


# ---------- CLI smoke ----------


def test_main_text_output_runs_and_reports_zero_failures():
  stdout = io.StringIO()
  previous_argv = sys.argv
  try:
    sys.argv = ["fuzz_lateral_params.py", "--seed", "1", "--cases", "5", "--kind", "missing_model_fail_closed"]
    with contextlib.redirect_stdout(stdout):
      main()
  finally:
    sys.argv = previous_argv

  output = stdout.getvalue()
  assert "Drive Lab lateral params fuzz" in output
  assert "failures=0" in output


def test_main_json_output_is_stable():
  stdout = io.StringIO()
  previous_argv = sys.argv
  try:
    sys.argv = ["fuzz_lateral_params.py", "--seed", "2", "--cases", "3", "--json"]
    with contextlib.redirect_stdout(stdout):
      main()
  finally:
    sys.argv = previous_argv

  payload = json.loads(stdout.getvalue())
  assert payload["seed"] == 2
  assert payload["cases"] == 3
  assert "failures" in payload


def test_main_exits_nonzero_on_injected_failure():
  original_evaluate = fuzz_lateral_params.evaluate_scenario
  config = ParamFuzzerConfig(seed=1, cases=1, kind="demand_enable_cycle")
  scenario = generate_scenarios(config)[0]
  baseline = evaluate_scenario(scenario)
  failed = ParamResult(
    scenario=baseline.scenario,
    outputs=baseline.outputs,
    structural_failures=baseline.structural_failures,
    event_failures=baseline.event_failures + [{"check": "injected", "detail": "injected failure"}],
    metrics=baseline.metrics,
  )

  def _forced_evaluate(scenario):
    return failed

  fuzz_lateral_params.evaluate_scenario = _forced_evaluate
  previous_argv = sys.argv
  try:
    sys.argv = ["fuzz_lateral_params.py", "--seed", "1", "--cases", "1"]
    with pytest.raises(SystemExit) as exc:
      main()
    assert exc.value.code == 1
  finally:
    fuzz_lateral_params.evaluate_scenario = original_evaluate
    sys.argv = previous_argv
