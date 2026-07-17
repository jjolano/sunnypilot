import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path

import pytest

from openpilot.sunnypilot.custom.lateral.demand.types import DEMAND_SOURCE_FALLBACK_MEASURED
from openpilot.tools.drive_lab import fuzz_lateral_transitions
from openpilot.tools.drive_lab.fuzz_lateral_transitions import (
  TransitionFuzzerConfig,
  TransitionResult,
  TransitionScenario,
  TransitionThresholds,
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
  assert generate_scenarios(TransitionFuzzerConfig(seed=42, cases=20)) == generate_scenarios(TransitionFuzzerConfig(seed=42, cases=20))


def test_evaluate_scenario_is_deterministic():
  config = TransitionFuzzerConfig(seed=7, cases=1, kind="clean_baseline")
  scenario = generate_scenarios(config)[0]
  first = evaluate_scenario(scenario)
  second = evaluate_scenario(scenario)
  assert first.valid == second.valid
  assert [f["check"] for f in first.structural_failures] == [f["check"] for f in second.structural_failures]
  assert [f["check"] for f in first.event_failures] == [f["check"] for f in second.event_failures]


# ---------- clean baseline ----------


def test_clean_baseline_passes_ungated():
  config = TransitionFuzzerConfig(seed=1, cases=1, kind="clean_baseline")
  scenario = generate_scenarios(config)[0]
  result = evaluate_scenario(scenario)
  assert result.valid
  assert all(not o.gated for o in result.outputs)


# ---------- lat_active toggle ----------


def test_lat_active_toggle_uses_fallback_measured():
  config = TransitionFuzzerConfig(seed=1, cases=1, kind="lat_active_toggle")
  scenario = generate_scenarios(config)[0]
  result = evaluate_scenario(scenario)
  inactive = [(o, f) for o, f in zip(result.outputs, scenario.frames) if not f.lat_active]
  assert inactive
  assert all(o.demand_source == DEMAND_SOURCE_FALLBACK_MEASURED and o.path_reason == "inactive" for o, _ in inactive)


# ---------- driver override pulse ----------


def test_driver_override_pulse_passes():
  config = TransitionFuzzerConfig(seed=1, cases=1, kind="driver_override_pulse")
  scenario = generate_scenarios(config)[0]
  result = evaluate_scenario(scenario)
  assert result.valid
  press = [o for o, f in zip(result.outputs, scenario.frames) if f.steering_pressed]
  assert press


# ---------- lane change session ----------


def test_lane_change_session_activates_shaping():
  config = TransitionFuzzerConfig(seed=1, cases=1, kind="lane_change_session")
  scenario = generate_scenarios(config)[0]
  result = evaluate_scenario(scenario)
  lc = [o for o, f in zip(result.outputs, scenario.frames) if f.lane_change_state != 0]
  assert lc
  assert any(o.lane_change_shaping_active for o in result.outputs)


# ---------- gating recovery ----------


def test_gating_recovery_gates_and_recovers():
  config = TransitionFuzzerConfig(seed=1, cases=1, kind="gating_recovery")
  scenario = generate_scenarios(config)[0]
  result = evaluate_scenario(scenario)
  gated = [o for o in result.outputs if o.gated]
  assert gated
  assert any(not o.gated for o in result.outputs[-20:])


# ---------- explicit control/jitter follow-ups ----------


def test_control_limit_flag_passes():
  config = TransitionFuzzerConfig(seed=1, cases=1, kind="control_limit_flag")
  scenario = generate_scenarios(config)[0]
  result = evaluate_scenario(scenario)
  assert result.valid
  limited = [o for o, f in zip(result.outputs, scenario.frames) if f.curvature_limited]
  assert limited


def test_model_demand_jitter_pulse_passes_without_gating():
  config = TransitionFuzzerConfig(seed=1, cases=1, kind="model_demand_jitter_pulse")
  scenario = generate_scenarios(config)[0]
  result = evaluate_scenario(scenario)
  assert result.valid
  base_curvature = scenario.frames[0].raw_curvature
  pulse = [o for o, f in zip(result.outputs, scenario.frames) if abs(f.raw_curvature - base_curvature) > 1e-12]
  assert pulse
  assert all(not o.gated for o in pulse)


def test_default_random_generation_excludes_explicit_only_kinds():
  config = TransitionFuzzerConfig(seed=1, cases=200)
  scenarios = generate_scenarios(config)
  explicit_kinds = {"control_limit_flag", "model_demand_jitter_pulse"}
  generated_kinds = {s.kind for s in scenarios}
  assert not (generated_kinds & explicit_kinds)


# ---------- replay / artifact ----------


def test_scenario_round_trips_through_dict():
  config = TransitionFuzzerConfig(seed=3, cases=1)
  scenario = generate_scenarios(config)[0]
  restored = scenario_from_dict(scenario_to_dict(scenario))
  assert restored.kind == scenario.kind
  assert restored.title == scenario.title
  assert restored.index == scenario.index
  assert len(restored.frames) == len(scenario.frames)


def test_artifact_replay_reports_equivalent_results():
  config = TransitionFuzzerConfig(seed=9, cases=1, kind="gating_recovery")
  scenario = generate_scenarios(config)[0]
  result = evaluate_scenario(scenario)
  with tempfile.TemporaryDirectory() as tmpdir:
    path = write_artifact(result, Path(tmpdir), seed=9, index=0)
    replayed = replay_artifact(path)
    assert replayed.valid == result.valid
    assert [f["check"] for f in replayed.structural_failures] == [f["check"] for f in result.structural_failures]
    assert [f["check"] for f in replayed.event_failures] == [f["check"] for f in result.event_failures]


def test_artifact_contains_full_frames_and_is_strict_json_safe():
  config = TransitionFuzzerConfig(seed=2, cases=1, kind="lane_change_session")
  scenario = generate_scenarios(config)[0]
  result = evaluate_scenario(scenario)
  with tempfile.TemporaryDirectory() as tmpdir:
    path = write_artifact(result, Path(tmpdir), seed=2, index=0)
    raw = path.read_text()
    payload = json.loads(raw)
    assert payload["schema"] == "drive-lab-lateral-transition-fuzzer-artifact"
    assert len(payload["scenario"]["frames"]) == len(scenario.frames)
    assert "nan" not in raw.lower()
    assert "inf" not in raw.lower()


def test_failure_artifact_writes_and_replays_equivalent_results():
  config = TransitionFuzzerConfig(seed=1, cases=1, kind="clean_baseline")
  scenario = generate_scenarios(config)[0]
  failed_scenario = TransitionScenario(
    kind=scenario.kind,
    title=scenario.title,
    index=scenario.index,
    frames=scenario.frames,
    event_windows=scenario.event_windows,
    thresholds=TransitionThresholds(max_abs_processed_curvature=1e-6),
  )
  result = evaluate_scenario(failed_scenario)
  assert result.structural_failures
  with tempfile.TemporaryDirectory() as tmpdir:
    path = write_artifact(result, Path(tmpdir), seed=1, index=0)
    replayed = replay_artifact(path)
    assert not replayed.valid
    assert [f["check"] for f in replayed.structural_failures] == [f["check"] for f in result.structural_failures]


def test_empty_scenario_returns_structured_failure():
  scenario = TransitionScenario(
    kind="clean_baseline",
    title="empty",
    index=0,
    frames=(),
    event_windows=(),
  )
  result = evaluate_scenario(scenario)
  assert not result.valid
  assert [f["check"] for f in result.structural_failures] == ["output"]


# ---------- CLI smoke ----------


def test_main_text_output_runs_and_reports_zero_failures():
  stdout = io.StringIO()
  previous_argv = sys.argv
  try:
    sys.argv = ["fuzz_lateral_transitions.py", "--seed", "1", "--cases", "5", "--kind", "clean_baseline"]
    with contextlib.redirect_stdout(stdout):
      main()
  finally:
    sys.argv = previous_argv

  output = stdout.getvalue()
  assert "Drive Lab lateral transition fuzz" in output
  assert "failures=0" in output


def test_main_json_output_is_stable():
  stdout = io.StringIO()
  previous_argv = sys.argv
  try:
    sys.argv = ["fuzz_lateral_transitions.py", "--seed", "2", "--cases", "3", "--json"]
    with contextlib.redirect_stdout(stdout):
      main()
  finally:
    sys.argv = previous_argv

  payload = json.loads(stdout.getvalue())
  assert payload["seed"] == 2
  assert payload["cases"] == 3
  assert "failures" in payload


def test_cli_explicit_kinds_report_zero_failures():
  for kind in ("control_limit_flag", "model_demand_jitter_pulse"):
    stdout = io.StringIO()
    previous_argv = sys.argv
    try:
      sys.argv = ["fuzz_lateral_transitions.py", "--seed", "1", "--cases", "5", "--kind", kind]
      with contextlib.redirect_stdout(stdout):
        main()
    finally:
      sys.argv = previous_argv
    output = stdout.getvalue()
    assert "failures=0" in output, f"{kind} produced failures: {output[:500]}"


def test_main_exits_nonzero_on_injected_failure():
  original_evaluate = fuzz_lateral_transitions.evaluate_scenario
  config = TransitionFuzzerConfig(seed=1, cases=1, kind="clean_baseline")
  scenario = generate_scenarios(config)[0]
  baseline = evaluate_scenario(scenario)
  failed = TransitionResult(
    scenario=baseline.scenario,
    outputs=baseline.outputs,
    structural_failures=baseline.structural_failures,
    event_failures=baseline.event_failures + [{"check": "injected", "detail": "injected failure"}],
    metrics=baseline.metrics,
  )

  def _forced_evaluate(scenario):
    return failed

  fuzz_lateral_transitions.evaluate_scenario = _forced_evaluate
  previous_argv = sys.argv
  try:
    sys.argv = ["fuzz_lateral_transitions.py", "--seed", "1", "--cases", "1"]
    with pytest.raises(SystemExit) as exc:
      main()
    assert exc.value.code == 1
  finally:
    fuzz_lateral_transitions.evaluate_scenario = original_evaluate
    sys.argv = previous_argv
