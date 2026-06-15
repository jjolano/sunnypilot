import contextlib
import io
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from openpilot.sunnypilot.custom.lateral.demand.types import DEMAND_SOURCE_LATERAL_MANEUVER, DEMAND_SOURCE_MODEL_PATH
from openpilot.tools.drive_lab import fuzz_lateral_demand
from openpilot.tools.drive_lab.fuzz_lateral_demand import (
  DemandFuzzerConfig,
  DemandScenario,
  evaluate_scenario,
  generate_scenarios,
  main,
  replay_artifact,
  scenario_from_dict,
  scenario_to_dict,
  write_artifact,
)


# ---------- helpers ----------


def _coherent_frames(curvature: float, v_ego: float = 20.0, duration_s: float = 1.0):
  from openpilot.tools.drive_lab.fuzz_lateral_demand import DT, _base_frame, _time_array
  return tuple(_base_frame(t=float(i * DT), v_ego=v_ego, curvature=curvature) for i, _ in enumerate(_time_array(duration_s)))


# ---------- determinism ----------


def test_generate_scenarios_is_seeded():
  config = DemandFuzzerConfig(seed=42, cases=20)
  assert generate_scenarios(config) == generate_scenarios(config)


def test_evaluate_scenario_is_deterministic():
  config = DemandFuzzerConfig(seed=7, cases=1, kind="high_quality_path")
  scenario = generate_scenarios(config)[0]
  first = evaluate_scenario(scenario)
  second = evaluate_scenario(scenario)
  assert first.valid == second.valid
  assert [f["check"] for f in first.failures] == [f["check"] for f in second.failures]


# ---------- high quality path ----------


def test_high_quality_path_passes():
  config = DemandFuzzerConfig(seed=1, cases=1, kind="high_quality_path", duration_s=1.0)
  scenario = generate_scenarios(config)[0]
  result = evaluate_scenario(scenario)
  assert result.valid
  assert all(out.gated is False for out in result.outputs[-10:])
  assert all(out.path_quality >= 0.9 for out in result.outputs[-10:])


# ---------- invalid path recovery ----------


def test_invalid_path_recovery_is_gated_and_finite():
  config = DemandFuzzerConfig(seed=1, cases=1, kind="invalid_path_recovery", duration_s=2.0)
  scenario = generate_scenarios(config)[0]
  result = evaluate_scenario(scenario)
  assert all(math.isfinite(out.processed_curvature) for out in result.outputs)
  reasons = {out.path_reason for out in result.outputs}
  assert "invalid_path" in reasons


# ---------- curvature jump ----------


def test_curvature_jump_reports_expected_behavior_or_bounded():
  config = DemandFuzzerConfig(seed=1, cases=5, kind="curvature_jump", duration_s=2.0)
  scenarios = generate_scenarios(config)
  for scenario in scenarios:
    result = evaluate_scenario(scenario)
    assert all(math.isfinite(out.processed_curvature) for out in result.outputs)
    assert max(abs(out.processed_curvature) for out in result.outputs) <= 0.5
    # The generator ensures a hard jump, so the pipeline should flag it.
    reasons = {out.path_reason for out in result.outputs}
    assert "curvature_jump" in reasons


# ---------- low lane confidence ----------


def test_low_lane_confidence_reacts_and_stays_bounded():
  config = DemandFuzzerConfig(seed=1, cases=3, kind="low_lane_confidence", duration_s=2.0)
  for scenario in generate_scenarios(config):
    result = evaluate_scenario(scenario)
    assert all(math.isfinite(out.processed_curvature) for out in result.outputs)
    reasons = {out.path_reason for out in result.outputs}
    assert "low_lane_confidence" in reasons or min(out.path_quality for out in result.outputs) < 0.9


# ---------- path disagreement ----------


def test_path_disagreement_reacts_and_stays_bounded():
  config = DemandFuzzerConfig(seed=1, cases=3, kind="path_disagreement", duration_s=2.0)
  for scenario in generate_scenarios(config):
    result = evaluate_scenario(scenario)
    assert all(math.isfinite(out.processed_curvature) for out in result.outputs)
    reasons = {out.path_reason for out in result.outputs}
    assert "path_disagreement" in reasons or min(out.path_quality for out in result.outputs) < 0.9


# ---------- lateral maneuver override ----------


def test_lateral_maneuver_override_source():
  config = DemandFuzzerConfig(seed=1, cases=1, kind="lateral_maneuver_override", duration_s=2.0)
  scenario = generate_scenarios(config)[0]
  result = evaluate_scenario(scenario)
  override_frames = [i for i, frame in enumerate(scenario.frames) if frame.get("lateral_maneuver_curvature") is not None]
  assert override_frames
  for i in override_frames:
    assert result.outputs[i].demand_source == DEMAND_SOURCE_LATERAL_MANEUVER
    assert result.outputs[i].processed_curvature == pytest.approx(scenario.frames[i]["lateral_maneuver_curvature"], abs=1e-6)


# ---------- replay / artifact ----------


def test_scenario_round_trips_through_dict():
  config = DemandFuzzerConfig(seed=3, cases=1)
  scenario = generate_scenarios(config)[0]
  restored = scenario_from_dict(scenario_to_dict(scenario))
  assert restored.kind == scenario.kind
  assert restored.title == scenario.title
  assert len(restored.frames) == len(scenario.frames)


def test_artifact_replay_reports_equivalent_classification():
  config = DemandFuzzerConfig(seed=9, cases=1, kind="invalid_path_recovery")
  scenario = generate_scenarios(config)[0]
  result = evaluate_scenario(scenario)
  with tempfile.TemporaryDirectory() as tmpdir:
    path = write_artifact(result, Path(tmpdir), seed=9, index=0)
    replayed = replay_artifact(path)
    assert replayed.valid == result.valid
    assert [f["check"] for f in replayed.failures] == [f["check"] for f in result.failures]


def test_artifact_contains_full_per_frame_inputs_and_is_strict_json_safe():
  config = DemandFuzzerConfig(seed=2, cases=1, kind="curvature_jump")
  scenario = generate_scenarios(config)[0]
  result = evaluate_scenario(scenario)
  with tempfile.TemporaryDirectory() as tmpdir:
    path = write_artifact(result, Path(tmpdir), seed=2, index=0)
    raw = path.read_text()
    payload = json.loads(raw)
    assert payload["schema"] == "drive-lab-lateral-demand-fuzzer-artifact"
    assert len(payload["scenario"]["frames"]) == len(scenario.frames)
    # Strict JSON: no NaN/Inf should have been emitted.
    assert "nan" not in raw.lower()
    assert "inf" not in raw.lower()


def test_empty_replay_scenario_returns_structured_failure():
  scenario = DemandScenario("empty", "empty", 0.0, ())

  result = evaluate_scenario(scenario)

  assert not result.valid
  assert [failure["check"] for failure in result.failures] == ["output"]


# ---------- CLI smoke ----------


def test_main_text_output_runs_and_reports_zero_failures():
  stdout = io.StringIO()
  previous_argv = sys.argv
  try:
    sys.argv = ["fuzz_lateral_demand.py", "--seed", "1", "--cases", "5", "--kind", "high_quality_path"]
    with contextlib.redirect_stdout(stdout):
      main()
  finally:
    sys.argv = previous_argv

  output = stdout.getvalue()
  assert "Drive Lab lateral demand fuzz" in output
  assert "failures=0" in output


def test_main_json_output_is_stable():
  stdout = io.StringIO()
  previous_argv = sys.argv
  try:
    sys.argv = ["fuzz_lateral_demand.py", "--seed", "2", "--cases", "3", "--json"]
    with contextlib.redirect_stdout(stdout):
      main()
  finally:
    sys.argv = previous_argv

  payload = json.loads(stdout.getvalue())
  assert payload["seed"] == 2
  assert payload["cases"] == 3
  assert "failures" in payload


def test_main_exits_nonzero_on_injected_failure():
  original_evaluate = fuzz_lateral_demand.evaluate_scenario
  config = DemandFuzzerConfig(seed=1, cases=1, kind="high_quality_path")
  scenario = generate_scenarios(config)[0]
  baseline = evaluate_scenario(scenario)
  failed = baseline.__class__(
    scenario=baseline.scenario,
    outputs=baseline.outputs,
    valid=False,
    failures=[{"check": "injected", "detail": "injected failure"}],
    metrics=baseline.metrics,
  )

  def _forced_evaluate(scenario):
    return failed

  fuzz_lateral_demand.evaluate_scenario = _forced_evaluate
  previous_argv = sys.argv
  try:
    sys.argv = ["fuzz_lateral_demand.py", "--seed", "1", "--cases", "1"]
    with pytest.raises(SystemExit) as exc:
      main()
    assert exc.value.code == 1
  finally:
    fuzz_lateral_demand.evaluate_scenario = original_evaluate
    sys.argv = previous_argv
