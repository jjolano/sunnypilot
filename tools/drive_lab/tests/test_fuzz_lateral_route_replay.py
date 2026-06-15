import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path

import pytest

from openpilot.tools.drive_lab import fuzz_lateral_route_replay
from openpilot.tools.drive_lab.fuzz_lateral_route_replay import (
  LateralRouteFrame,
  PerturbationRecipe,
  RouteReplayFuzzerConfig,
  RouteReplayResult,
  RouteReplayScenario,
  RouteReplayThresholds,
  _apply_recipe,
  evaluate_scenario,
  generate_scenarios,
  main,
  replay_artifact,
  write_artifact,
)


# ---------- helpers ----------


def _coherent_frame(t: float, v_ego: float = 20.0, curvature: float = 0.001):
  from openpilot.tools.drive_lab.fuzz_lateral_route_replay import N_PATH_POINTS
  xs = tuple(float(x) for x in range(N_PATH_POINTS))
  ys = tuple(0.5 * curvature * x * x for x in range(N_PATH_POINTS))
  ystd = tuple(0.1 for _ in range(N_PATH_POINTS))
  yaw = tuple(curvature * x for x in range(N_PATH_POINTS))
  yaw_rate = tuple(curvature * v_ego for _ in range(N_PATH_POINTS))
  return LateralRouteFrame(
    t=t,
    v_ego=v_ego,
    lat_active=True,
    raw_curvature=curvature,
    measured_curvature=curvature,
    roll=0.0,
    steering_pressed=False,
    left_blinker=False,
    right_blinker=False,
    lane_change_state=0,
    lane_change_direction=0,
    position_x=xs,
    position_y=ys,
    position_y_std=ystd,
    orientation_z=yaw,
    orientation_rate_z=yaw_rate,
    lane_line_probs=(0.9, 0.9, 0.9, 0.9),
    frame_drop_perc=0.0,
  )


# ---------- determinism ----------


def test_generate_scenarios_is_seeded():
  assert generate_scenarios(RouteReplayFuzzerConfig(seed=42, cases=10)) == generate_scenarios(RouteReplayFuzzerConfig(seed=42, cases=10))


def test_evaluate_scenario_is_deterministic():
  scenario = generate_scenarios(RouteReplayFuzzerConfig(seed=7, cases=1, preset="synthetic_curve"))[0]
  first = evaluate_scenario(scenario)
  second = evaluate_scenario(scenario)
  assert first.valid == second.valid
  assert [f["check"] for f in first.baseline_failures] == [f["check"] for f in second.baseline_failures]
  assert [f["check"] for f in first.perturbation_failures] == [f["check"] for f in second.perturbation_failures]


# ---------- baseline == perturbed without perturbation ----------


def test_identity_recipe_produces_identical_outputs():
  scenario = generate_scenarios(RouteReplayFuzzerConfig(seed=1, cases=1, preset="synthetic_curve"))[0]
  no_perturb = scenario.__class__(
    preset=scenario.preset,
    title=scenario.title,
    frames=scenario.frames,
    recipe=PerturbationRecipe(kind="none", start_frame=0, end_frame=0, description="no perturbation"),
  )
  result = evaluate_scenario(no_perturb)
  base_processed = [o.processed_curvature for o in result.baseline_outputs]
  pert_processed = [o.processed_curvature for o in result.perturbed_outputs]
  assert base_processed == pert_processed
  assert result.valid


# ---------- default perturbation coverage ----------


def test_all_presets_and_perturbations_pass_mild_cases():
  for preset in ("synthetic_straight", "synthetic_curve", "synthetic_sine", "synthetic_reversal"):
    for pert in ("noise", "dropout", "delay", "stale", "scale", "offset"):
      config = RouteReplayFuzzerConfig(seed=1, cases=3, preset=preset, perturbation=pert)
      for scenario in generate_scenarios(config):
        result = evaluate_scenario(scenario)
        assert result.valid, f"{preset}/{pert} failed: {result.baseline_failures} {result.perturbation_failures} {result.comparison_failures}"


# ---------- dropout targeted behavior ----------


def test_dropout_produces_invalid_path_or_reduced_quality():
  config = RouteReplayFuzzerConfig(seed=1, cases=5, preset="synthetic_curve", perturbation="dropout")
  found = False
  for scenario in generate_scenarios(config):
    result = evaluate_scenario(scenario)
    if not result.valid:
      continue
    window = result.perturbed_outputs[scenario.recipe.start_frame:scenario.recipe.end_frame]
    reasons = {o.path_reason for o in window}
    if "invalid_path" in reasons or any(o.path_quality < 0.9 for o in window):
      found = True
      break
  assert found, "dropout did not produce expected invalid_path or reduced quality in any case"


# ---------- delay/stale causality ----------


def test_delay_is_causal():
  config = RouteReplayFuzzerConfig(seed=1, cases=1, preset="synthetic_curve", perturbation="delay")
  scenario = generate_scenarios(config)[0]
  delay_frames = scenario.recipe.params.get("delay_frames", 3)
  perturbed = _apply_recipe(scenario.recipe, scenario.frames)
  start = scenario.recipe.start_frame
  end = scenario.recipe.end_frame
  for i in range(start, end):
    assert perturbed[i].raw_curvature == scenario.frames[i - delay_frames].raw_curvature


def test_stale_uses_only_start_frame_value():
  config = RouteReplayFuzzerConfig(seed=1, cases=1, preset="synthetic_curve", perturbation="stale")
  scenario = generate_scenarios(config)[0]
  perturbed = _apply_recipe(scenario.recipe, scenario.frames)
  start = scenario.recipe.start_frame
  stale_value = scenario.frames[start].raw_curvature
  for i in range(start, scenario.recipe.end_frame):
    assert perturbed[i].raw_curvature == stale_value


# ---------- baseline failure skips comparison ----------


def test_baseline_failure_skips_comparison():
  frames = (_coherent_frame(0.0), _coherent_frame(0.01))
  # Non-finite curvature triggers a baseline input validation failure.
  bad_frames = tuple(
    frame.__class__(**{**frame.to_dict(), "raw_curvature": float("nan"), "measured_curvature": float("nan")})
    for frame in frames
  )
  scenario = RouteReplayScenario(
    preset="synthetic_straight",
    title="bad baseline",
    frames=bad_frames,
    recipe=PerturbationRecipe(kind="none", start_frame=0, end_frame=0, description="no perturbation"),
  )
  result = evaluate_scenario(scenario)
  assert result.baseline_failures
  assert any(f["check"] == "comparison_skipped" for f in result.comparison_failures)


def test_non_monotonic_or_non_fixed_dt_frames_fail_validation():
  frames = (_coherent_frame(0.0), _coherent_frame(0.02))
  scenario = RouteReplayScenario(
    preset="synthetic_straight",
    title="bad timing",
    frames=frames,
    recipe=PerturbationRecipe(kind="none", start_frame=0, end_frame=0, description="no perturbation"),
  )
  result = evaluate_scenario(scenario)
  assert any(f["check"] == "input_timing" for f in result.baseline_failures)


def test_perturbed_length_mismatch_reports_comparison_failure():
  frames = (_coherent_frame(0.0), _coherent_frame(0.01))
  scenario = RouteReplayScenario(
    preset="synthetic_straight",
    title="short perturbed replay",
    frames=frames,
    recipe=PerturbationRecipe(kind="none", start_frame=0, end_frame=0, description="no perturbation"),
    perturbed_frames=(frames[0],),
  )
  result = evaluate_scenario(scenario)
  assert any(f["check"] == "comparison_length" for f in result.comparison_failures)


# ---------- artifact / replay ----------


def test_artifact_replay_reports_equivalent_classification():
  scenario = generate_scenarios(RouteReplayFuzzerConfig(seed=9, cases=1, preset="synthetic_curve", perturbation="noise"))[0]
  result = evaluate_scenario(scenario)
  with tempfile.TemporaryDirectory() as tmpdir:
    path = write_artifact(result, Path(tmpdir), seed=9, index=0)
    replayed = replay_artifact(path)
    assert replayed.valid == result.valid
    assert [f["check"] for f in replayed.baseline_failures] == [f["check"] for f in result.baseline_failures]
    assert [f["check"] for f in replayed.perturbation_failures] == [f["check"] for f in result.perturbation_failures]


def test_artifact_contains_baseline_and_perturbed_frames_and_is_strict_json_safe():
  scenario = generate_scenarios(RouteReplayFuzzerConfig(seed=2, cases=1, preset="synthetic_curve", perturbation="delay"))[0]
  result = evaluate_scenario(scenario)
  with tempfile.TemporaryDirectory() as tmpdir:
    path = write_artifact(result, Path(tmpdir), seed=2, index=0)
    raw = path.read_text()
    payload = json.loads(raw)
    assert payload["schema"] == "drive-lab-lateral-route-replay-fuzzer-artifact"
    assert "baseline_frames" in payload
    assert "perturbed_frames" in payload
    assert len(payload["baseline_frames"]) == len(scenario.frames)
    assert "nan" not in raw.lower()
    assert "inf" not in raw.lower()


# ---------- CLI smoke ----------


def test_main_text_output_runs_and_reports_zero_failures():
  stdout = io.StringIO()
  previous_argv = sys.argv
  try:
    sys.argv = ["fuzz_lateral_route_replay.py", "--seed", "1", "--cases", "5", "--preset", "synthetic_curve", "--perturbation", "noise"]
    with contextlib.redirect_stdout(stdout):
      main()
  finally:
    sys.argv = previous_argv

  output = stdout.getvalue()
  assert "Drive Lab lateral route replay fuzz" in output
  assert "failures=0" in output


def test_main_accepts_none_perturbation():
  stdout = io.StringIO()
  previous_argv = sys.argv
  try:
    sys.argv = ["fuzz_lateral_route_replay.py", "--seed", "1", "--cases", "2", "--preset", "synthetic_curve", "--perturbation", "none"]
    with contextlib.redirect_stdout(stdout):
      main()
  finally:
    sys.argv = previous_argv

  assert "failures=0" in stdout.getvalue()


def test_main_json_output_is_stable():
  stdout = io.StringIO()
  previous_argv = sys.argv
  try:
    sys.argv = ["fuzz_lateral_route_replay.py", "--seed", "2", "--cases", "3", "--json"]
    with contextlib.redirect_stdout(stdout):
      main()
  finally:
    sys.argv = previous_argv

  payload = json.loads(stdout.getvalue())
  assert payload["seed"] == 2
  assert payload["cases"] == 3
  assert "failures" in payload


def test_failure_artifact_writes_and_replays_equivalent_classification():
  scenarios = generate_scenarios(RouteReplayFuzzerConfig(seed=1, cases=1, preset="synthetic_sine"))
  scenario = scenarios[0]
  # Tighten jerk threshold so the sinusoidal baseline legitimately fails.
  failed_scenario = RouteReplayScenario(
    preset=scenario.preset,
    title=scenario.title,
    frames=scenario.frames,
    recipe=scenario.recipe,
    thresholds=RouteReplayThresholds(max_abs_lat_jerk=1e-6),
  )
  result = evaluate_scenario(failed_scenario)
  assert result.baseline_failures
  with tempfile.TemporaryDirectory() as tmpdir:
    path = write_artifact(result, Path(tmpdir), seed=1, index=0)
    replayed = replay_artifact(path)
    assert not replayed.valid
    assert [f["check"] for f in replayed.baseline_failures] == [f["check"] for f in result.baseline_failures]


def test_main_exits_nonzero_on_injected_failure():
  scenarios = generate_scenarios(RouteReplayFuzzerConfig(seed=1, cases=1, preset="synthetic_curve"))
  baseline = evaluate_scenario(scenarios[0])
  failed = RouteReplayResult(
    scenario=baseline.scenario,
    baseline_outputs=baseline.baseline_outputs,
    perturbed_outputs=baseline.perturbed_outputs,
    baseline_failures=baseline.baseline_failures,
    perturbation_failures=baseline.perturbation_failures,
    comparison_failures=baseline.comparison_failures + [{"check": "injected", "detail": "injected failure"}],
    metrics=baseline.metrics,
  )

  original_run = fuzz_lateral_route_replay.evaluate_scenario
  fuzz_lateral_route_replay.evaluate_scenario = lambda scenario: failed
  previous_argv = sys.argv
  try:
    sys.argv = ["fuzz_lateral_route_replay.py", "--seed", "1", "--cases", "1"]
    with pytest.raises(SystemExit) as exc:
      main()
    assert exc.value.code == 1
  finally:
    fuzz_lateral_route_replay.evaluate_scenario = original_run
    sys.argv = previous_argv
