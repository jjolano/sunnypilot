import contextlib
import io
import json
import random
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

import openpilot.tools.drive_lab.route_io as route_io_module
from openpilot.tools.drive_lab import fuzz_lateral_route_replay
from openpilot.tools.drive_lab.fuzz_lateral_route_replay import (
  DT,
  LateralRouteFrame,
  N_PATH_POINTS,
  PerturbationRecipe,
  RouteReplayFuzzerConfig,
  RouteReplayResult,
  RouteReplayScenario,
  RouteReplayThresholds,
  ROUTE_EXTRACTED_PRESET,
  _apply_recipe,
  _coherent_path,
  _frame_to_inputs,
  _generate_recipe,
  evaluate_scenario,
  extract_lateral_route_frames,
  extract_lateral_route_frames_with_summary,
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
    perturbations = ["noise", "dropout", "delay", "stale", "model_age_stale", "scale", "offset"]
    if preset in ("synthetic_sine", "synthetic_reversal"):
      perturbations.append("model_age_delay")
    for pert in perturbations:
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


def test_dropout_recovery_boundary_is_not_scored_as_structural_jerk_failure():
  scenario = generate_scenarios(RouteReplayFuzzerConfig(seed=20, cases=61, duration_s=3.0, perturbation="dropout"))[60]
  assert scenario.recipe.kind == "dropout"

  result = evaluate_scenario(scenario)

  assert result.valid


def test_noise_window_sign_flips_are_not_scored_as_residual_oscillation():
  scenario = generate_scenarios(RouteReplayFuzzerConfig(seed=29, cases=35, duration_s=30.0, perturbation="noise"))[34]
  assert scenario.preset == "synthetic_sine"
  assert scenario.recipe.kind == "noise"

  result = evaluate_scenario(scenario)

  assert result.valid


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


def test_model_age_stale_marks_window_without_changing_curvature():
  config = RouteReplayFuzzerConfig(seed=1, cases=1, preset="synthetic_curve", perturbation="model_age_stale")
  scenario = generate_scenarios(config)[0]
  perturbed = _apply_recipe(scenario.recipe, scenario.frames)
  start = scenario.recipe.start_frame
  end = scenario.recipe.end_frame
  stale_age_s = float(scenario.recipe.params["model_age_s"])
  for i in range(start, end):
    assert perturbed[i].model_age_s == pytest.approx(stale_age_s)
    assert perturbed[i].raw_curvature == scenario.frames[i].raw_curvature


def test_model_age_stale_produces_model_stale_reason():
  config = RouteReplayFuzzerConfig(seed=1, cases=3, preset="synthetic_curve", perturbation="model_age_stale")
  for scenario in generate_scenarios(config):
    result = evaluate_scenario(scenario)
    if not result.valid:
      continue
    window = result.perturbed_outputs[scenario.recipe.start_frame:scenario.recipe.end_frame]
    assert window
    assert all(o.path_reason == "model_stale" for o in window)
    return
  raise AssertionError("model_age_stale did not produce a valid scenario")


def test_model_age_delay_combines_delayed_raw_with_stale_age():
  config = RouteReplayFuzzerConfig(seed=1, cases=1, preset="synthetic_sine", perturbation="model_age_delay")
  scenario = generate_scenarios(config)[0]
  perturbed = _apply_recipe(scenario.recipe, scenario.frames)
  start = scenario.recipe.start_frame
  end = scenario.recipe.end_frame
  delay_frames = int(scenario.recipe.params["delay_frames"])
  stale_age_s = float(scenario.recipe.params["model_age_s"])
  for i in range(start, end):
    assert perturbed[i].model_age_s == pytest.approx(stale_age_s)
    assert perturbed[i].raw_curvature == scenario.frames[i - delay_frames].raw_curvature
    assert perturbed[i].measured_curvature == scenario.frames[i].measured_curvature


def test_model_age_delay_generation_never_starts_before_delayable_frame():
  for seed in range(50):
    recipe = _generate_recipe(random.Random(seed), 30, "model_age_delay")
    assert recipe.start_frame >= 2
    assert int(recipe.params["delay_frames"]) >= 1
    assert int(recipe.params["delay_frames"]) <= recipe.start_frame


def test_model_age_delay_bridges_toward_measured_curvature():
  config = RouteReplayFuzzerConfig(seed=1, cases=3, preset="synthetic_sine", perturbation="model_age_delay")
  for scenario in generate_scenarios(config):
    result = evaluate_scenario(scenario)
    if not result.valid:
      continue
    window_outputs = result.perturbed_outputs[scenario.recipe.start_frame:scenario.recipe.end_frame]
    window_frames = scenario.perturbed_frames[scenario.recipe.start_frame:scenario.recipe.end_frame]
    assert window_outputs
    assert all(o.path_reason == "model_stale" for o in window_outputs)
    assert any(
      abs(o.processed_curvature - f.measured_curvature) < abs(f.raw_curvature - f.measured_curvature)
      for f, o in zip(window_frames, window_outputs, strict=False)
    )
    return
  raise AssertionError("model_age_delay did not produce a valid scenario")


def test_perturbations_preserve_model_age_unless_explicitly_stale():
  frames = tuple(_coherent_frame(i * DT).__class__(**{**_coherent_frame(i * DT).to_dict(), "model_age_s": 0.07}) for i in range(20))
  for kind in ("noise", "dropout", "delay", "stale", "scale", "offset"):
    recipe = PerturbationRecipe(kind=kind, start_frame=5, end_frame=12, description=kind,
                                params={"delay_frames": 2, "noise_seed": 1, "scale_seed": 1, "offset_seed": 1})
    perturbed = _apply_recipe(recipe, frames)
    assert all(frame.model_age_s == pytest.approx(0.07) for frame in perturbed)


def test_stale_exit_boundary_is_not_scored_as_structural_jerk_failure():
  scenario = generate_scenarios(RouteReplayFuzzerConfig(seed=19, cases=14, duration_s=3.0, perturbation="stale"))[13]
  assert scenario.recipe.kind == "stale"

  result = evaluate_scenario(scenario)

  assert result.valid


def test_stale_window_divergence_is_not_scored_inside_materialized_window():
  scenario = generate_scenarios(RouteReplayFuzzerConfig(seed=20, cases=115, duration_s=3.0, perturbation="stale"))[114]
  assert scenario.recipe.kind == "stale"

  result = evaluate_scenario(scenario)

  assert result.valid


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


# ---------- route extraction helpers ----------


class _FakeMsg(SimpleNamespace):
  def which(self):
    return self.kind


def _route_msg(kind: str, t_s: float, payload: SimpleNamespace) -> _FakeMsg:
  return _FakeMsg(kind=kind, logMonoTime=int(t_s * 1e9), **{kind: payload})


def _car_state(t_s: float, v_ego: float = 20.0, steering_pressed: bool = False) -> SimpleNamespace:
  return _route_msg("carState", t_s, SimpleNamespace(vEgo=v_ego, steeringPressed=steering_pressed, leftBlinker=False, rightBlinker=False))


def _car_control(t_s: float, lat_active: bool = True, actuator_curvature: float = 0.001) -> SimpleNamespace:
  return _route_msg(
    "carControl",
    t_s,
    SimpleNamespace(latActive=lat_active, actuators=SimpleNamespace(curvature=actuator_curvature)),
  )


def _live_parameters(t_s: float, roll: float = 0.0) -> SimpleNamespace:
  return _route_msg("liveParameters", t_s, SimpleNamespace(roll=roll))


def _model_v2(t_s: float, desired_curvature: float, v_ego: float = 20.0, frame_drop_perc: float = 0.0) -> SimpleNamespace:
  path = _coherent_path(desired_curvature, v_ego)
  return _route_msg(
    "modelV2",
    t_s,
    SimpleNamespace(
      action=SimpleNamespace(desiredCurvature=desired_curvature),
      position=SimpleNamespace(
        x=path["position_x"],
        y=path["position_y"],
        yStd=path["position_y_std"],
      ),
      orientation=SimpleNamespace(z=path["orientation_z"]),
      orientationRate=SimpleNamespace(z=path["orientation_rate_z"]),
      laneLineProbs=[0.9, 0.9, 0.9, 0.9],
      frameDropPerc=frame_drop_perc,
      meta=SimpleNamespace(laneChangeState=0, laneChangeDirection=0),
    ),
  )


def _controls_state(t_s: float, curvature: float = 0.001, desired_curvature: float = 0.001) -> SimpleNamespace:
  return _route_msg(
    "controlsState",
    t_s,
    SimpleNamespace(curvature=curvature, desiredCurvature=desired_curvature),
  )


def _controls_state_with_model_path(t_s: float, raw_desired_curvature: float) -> SimpleNamespace:
  return _route_msg(
    "controlsState",
    t_s,
    SimpleNamespace(
      curvature=0.0,
      desiredCurvature=0.0,
      modelPathState=SimpleNamespace(rawDesiredCurvature=raw_desired_curvature),
    ),
  )


def _model_v2_no_action(t_s: float, desired_curvature: float, v_ego: float = 20.0) -> SimpleNamespace:
  path = _coherent_path(desired_curvature, v_ego)
  return _route_msg(
    "modelV2",
    t_s,
    SimpleNamespace(
      position=SimpleNamespace(
        x=path["position_x"],
        y=path["position_y"],
        yStd=path["position_y_std"],
      ),
      orientation=SimpleNamespace(z=path["orientation_z"]),
      orientationRate=SimpleNamespace(z=path["orientation_rate_z"]),
      laneLineProbs=[0.9, 0.9, 0.9, 0.9],
      meta=SimpleNamespace(laneChangeState=0, laneChangeDirection=0),
    ),
  )


def _model_v2_empty_arrays(t_s: float) -> SimpleNamespace:
  return _route_msg(
    "modelV2",
    t_s,
    SimpleNamespace(
      position=SimpleNamespace(x=[], y=[], yStd=[]),
      orientation=SimpleNamespace(z=[]),
      orientationRate=SimpleNamespace(z=[]),
      laneLineProbs=[],
      meta=SimpleNamespace(laneChangeState=0, laneChangeDirection=0),
    ),
  )


def _fake_route_messages(n: int, desired_curvature: float = 0.001, start_t: float = 0.0) -> list[SimpleNamespace]:
  msgs: list[SimpleNamespace] = []
  for i in range(n):
    t = start_t + i * DT
    msgs.extend([
      _car_state(t),
      _car_control(t),
      _live_parameters(t),
      _model_v2(t, desired_curvature),
      _controls_state(t, curvature=desired_curvature, desired_curvature=desired_curvature),
    ])
  return msgs


# ---------- route extraction unit tests ----------


def test_extract_lateral_route_frames_counts_and_normalizes_time():
  frames = extract_lateral_route_frames(_fake_route_messages(5, desired_curvature=0.001))
  assert len(frames) == 5
  for i, frame in enumerate(frames):
    assert frame.t == pytest.approx(i * DT)
    assert frame.v_ego == pytest.approx(20.0)
    assert frame.lat_active is True
    assert len(frame.position_x) == N_PATH_POINTS


def test_extract_lateral_route_frames_summary_preserves_original_time_span():
  frames, summary = extract_lateral_route_frames_with_summary(_fake_route_messages(5, desired_curvature=0.001), route="fake/route", qlog=True)
  assert len(frames) == 5
  assert summary.route == "fake/route"
  assert summary.qlog is True
  assert summary.extracted_count == 5
  assert summary.original_time_span_s == pytest.approx(0.04)
  assert summary.selected_original_start_s == pytest.approx(0.0)
  assert summary.selected_original_end_s == pytest.approx(0.04)
  assert summary.source_dt_stats == {"min": pytest.approx(DT), "median": pytest.approx(DT), "max": pytest.approx(DT)}


def test_extract_lateral_route_frames_applies_window_and_max_frames():
  msgs = _fake_route_messages(10, desired_curvature=0.001, start_t=0.0)
  frames = extract_lateral_route_frames(msgs, start_s=0.02, end_s=0.07, max_frames=3)
  assert len(frames) == 3
  assert frames[0].t == pytest.approx(0.0)


def test_extract_lateral_route_frames_prefers_model_v2_action_curvature():
  k_model = 0.002
  k_controls = 0.001
  msgs = [
    _car_state(0.0),
    _car_control(0.0),
    _live_parameters(0.0),
    _model_v2(0.0, desired_curvature=k_model),
    _controls_state(0.0, curvature=k_controls, desired_curvature=k_controls),
  ]
  frames = extract_lateral_route_frames(msgs)
  assert len(frames) == 1
  assert frames[0].raw_curvature == pytest.approx(k_model)
  assert frames[0].measured_curvature == pytest.approx(k_controls)


def test_extract_lateral_route_frames_preserves_frame_drop_perc():
  msgs = [
    _car_state(0.0),
    _car_control(0.0),
    _live_parameters(0.0),
    _model_v2(0.0, desired_curvature=0.001, frame_drop_perc=12.5),
    _controls_state(0.0, curvature=0.001, desired_curvature=0.001),
  ]
  frames = extract_lateral_route_frames(msgs)
  assert len(frames) == 1
  assert frames[0].frame_drop_perc == pytest.approx(12.5)


def test_extract_lateral_route_frames_uses_measured_when_inactive_and_preserves_steering_pressed():
  k_model = 0.002
  k_controls = 0.001
  msgs = [
    _car_state(0.0, steering_pressed=True),
    _car_control(0.0, lat_active=False),
    _live_parameters(0.0),
    _model_v2(0.0, desired_curvature=k_model),
    _controls_state(0.0, curvature=k_controls, desired_curvature=k_model),
  ]
  frames = extract_lateral_route_frames(msgs)
  assert len(frames) == 1
  assert frames[0].raw_curvature == pytest.approx(k_controls)
  assert frames[0].measured_curvature == pytest.approx(k_controls)
  assert frames[0].steering_pressed is True
  assert _frame_to_inputs(frames[0]).steering_pressed is True
  result = evaluate_scenario(RouteReplayScenario(
    preset=ROUTE_EXTRACTED_PRESET,
    title="inactive route frame",
    frames=frames,
    recipe=PerturbationRecipe(kind="none", start_frame=0, end_frame=0, description="no perturbation"),
  ))
  assert result.valid


def test_extract_lateral_route_frames_falls_back_when_model_v2_missing():
  k = 0.0015
  msgs = [
    _car_state(0.0),
    _car_control(0.0),
    _live_parameters(0.0),
    _controls_state(0.0, curvature=k, desired_curvature=k),
  ]
  frames = extract_lateral_route_frames(msgs)
  assert len(frames) == 1
  assert frames[0].raw_curvature == pytest.approx(k)


def test_extract_lateral_route_frames_returns_empty_for_missing_controls_state():
  msgs = [
    _car_state(0.0),
    _car_control(0.0),
    _live_parameters(0.0),
    _model_v2(0.0, desired_curvature=0.001),
  ]
  assert extract_lateral_route_frames(msgs) == ()


def test_extract_lateral_route_frames_skips_without_car_state():
  msgs = [
    _car_control(0.0),
    _live_parameters(0.0),
    _model_v2(0.0, desired_curvature=0.001),
    _controls_state(0.0, curvature=0.001, desired_curvature=0.001),
  ]
  assert extract_lateral_route_frames(msgs) == ()


def test_extract_lateral_route_frames_uses_controls_state_model_path_raw_desired():
  k_raw = 0.0025
  msgs = [
    _car_state(0.0),
    _car_control(0.0),
    _live_parameters(0.0),
    _model_v2_no_action(0.0, desired_curvature=0.001),
    _controls_state_with_model_path(0.0, raw_desired_curvature=k_raw),
  ]
  frames = extract_lateral_route_frames(msgs)
  assert len(frames) == 1
  assert frames[0].raw_curvature == pytest.approx(k_raw)


def test_extract_lateral_route_frames_active_fallback_uses_actuator_curvature():
  k = 0.0018
  msgs = [
    _car_state(0.0),
    _car_control(0.0, actuator_curvature=k),
    _live_parameters(0.0),
    _model_v2_no_action(0.0, desired_curvature=0.001),
    _route_msg("controlsState", 0.0, SimpleNamespace()),
  ]
  frames = extract_lateral_route_frames(msgs)
  assert len(frames) == 1
  assert frames[0].raw_curvature == pytest.approx(k)
  assert frames[0].measured_curvature == pytest.approx(k)


def test_extract_lateral_route_frames_inactive_fallback_uses_actuator_curvature():
  k = 0.0016
  msgs = [
    _car_state(0.0),
    _car_control(0.0, lat_active=False, actuator_curvature=k),
    _live_parameters(0.0),
    _route_msg("controlsState", 0.0, SimpleNamespace()),
  ]
  frames = extract_lateral_route_frames(msgs)
  assert len(frames) == 1
  assert frames[0].raw_curvature == pytest.approx(k)
  assert frames[0].lat_active is False


def test_extraction_summary_quality_counts_input_and_controls_state():
  msgs = _fake_route_messages(3, desired_curvature=0.001)
  frames, summary = extract_lateral_route_frames_with_summary(msgs)
  assert len(frames) == 3
  assert summary.quality is not None
  assert summary.quality.input_message_count == len(msgs)
  assert summary.quality.controls_state_seen == 3
  assert summary.quality.controls_state_in_window == 3
  assert summary.quality.skipped_missing_car_state == 0


def test_extraction_summary_quality_counts_missing_context():
  msgs = [
    _car_state(0.0),
    _controls_state(0.0),
    _car_state(0.01),
    _controls_state(0.01),
  ]
  frames, summary = extract_lateral_route_frames_with_summary(msgs)
  assert len(frames) == 2
  assert summary.quality is not None
  assert summary.quality.missing_model_v2 == 2
  assert summary.quality.missing_car_control == 2
  assert summary.quality.missing_live_parameters == 2


def test_extraction_summary_quality_counts_skipped_missing_car_state():
  msgs = [
    _controls_state(0.0),
    _car_state(0.01),
    _controls_state(0.01),
  ]
  frames, summary = extract_lateral_route_frames_with_summary(msgs)
  assert len(frames) == 1
  assert summary.quality is not None
  assert summary.quality.controls_state_seen == 2
  assert summary.quality.controls_state_in_window == 2
  assert summary.quality.skipped_missing_car_state == 1


def test_extraction_summary_quality_counts_empty_arrays():
  msgs = [
    _car_state(0.0),
    _car_control(0.0),
    _live_parameters(0.0),
    _model_v2_empty_arrays(0.0),
    _controls_state(0.0),
  ]
  frames, summary = extract_lateral_route_frames_with_summary(msgs)
  assert len(frames) == 1
  assert summary.quality is not None
  assert summary.quality.empty_position_path == 1
  assert summary.quality.empty_orientation == 1
  assert summary.quality.empty_orientation_rate == 1
  assert summary.quality.empty_lane_line_probs == 1
  assert summary.quality.missing_model_v2 == 0


def test_extracted_frames_preserve_source_t():
  msgs = _fake_route_messages(5, desired_curvature=0.001, start_t=2.0)
  frames = extract_lateral_route_frames(msgs)
  assert len(frames) == 5
  for i, frame in enumerate(frames):
    assert frame.t == pytest.approx(i * DT)
    assert frame.source_t == pytest.approx(i * DT)


def test_perturbations_preserve_source_t():
  frames = extract_lateral_route_frames(_fake_route_messages(8, desired_curvature=0.001))
  for kind, params in (
    ("noise", {"noise_seed": 1}),
    ("dropout", {}),
    ("delay", {"delay_frames": 1}),
    ("stale", {}),
    ("scale", {"scale_seed": 1}),
    ("offset", {"offset_seed": 1}),
  ):
    perturbed = _apply_recipe(PerturbationRecipe(kind=kind, start_frame=2, end_frame=6, description=kind, params=params), frames)
    assert [frame.source_t for frame in perturbed] == [frame.source_t for frame in frames]


# ---------- route-derived scenario tests ----------


def test_generate_route_scenarios_attaches_metadata(monkeypatch):
  baseline = _fake_route_messages(10, desired_curvature=0.001)
  original_load_route_msgs = route_io_module.load_route_msgs
  route_io_module.load_route_msgs = lambda route, qlog=False: baseline
  try:
    scenarios = generate_scenarios(
      RouteReplayFuzzerConfig(seed=5, cases=3, route="fake/route", qlog=True, window_start_s=0.0, window_end_s=0.05, max_frames=10)
    )
    assert len(scenarios) == 3
    assert all(s.preset == ROUTE_EXTRACTED_PRESET for s in scenarios)
    assert all(s.route_metadata is not None for s in scenarios)
    meta = scenarios[0].route_metadata
    assert meta is not None
    assert meta.route == "fake/route"
    assert meta.qlog is True
    assert meta.extracted_count == 5
    assert meta.original_time_span_s == pytest.approx(0.04)
  finally:
    route_io_module.load_route_msgs = original_load_route_msgs


def test_route_scenario_evaluates_valid(monkeypatch):
  baseline = _fake_route_messages(20, desired_curvature=0.0005)
  original_load_route_msgs = route_io_module.load_route_msgs
  route_io_module.load_route_msgs = lambda route, qlog=False: baseline
  try:
    scenarios = generate_scenarios(RouteReplayFuzzerConfig(seed=1, cases=1, route="fake/route", perturbation="none"))
    result = evaluate_scenario(scenarios[0])
    assert result.valid
  finally:
    route_io_module.load_route_msgs = original_load_route_msgs


def test_route_replay_artifact_round_trips_metadata(monkeypatch):
  baseline = _fake_route_messages(10, desired_curvature=0.0005)
  original_load_route_msgs = route_io_module.load_route_msgs
  route_io_module.load_route_msgs = lambda route, qlog=False: baseline
  try:
    scenarios = generate_scenarios(RouteReplayFuzzerConfig(seed=2, cases=1, route="fake/route", qlog=True, max_frames=10))
    result = evaluate_scenario(scenarios[0])
    with tempfile.TemporaryDirectory() as tmpdir:
      path = write_artifact(result, Path(tmpdir), seed=2, index=0)
      data = json.loads(path.read_text())
      assert data["route_metadata"]["route"] == "fake/route"
      assert data["route_metadata"]["qlog"] is True
      replayed = replay_artifact(path)
      assert replayed.scenario.route_metadata is not None
      assert replayed.scenario.route_metadata.route == "fake/route"
  finally:
    route_io_module.load_route_msgs = original_load_route_msgs


# ---------- CLI route tests ----------


def test_main_exits_nonzero_when_route_has_no_frames(monkeypatch):
  original_load_route_msgs = route_io_module.load_route_msgs
  route_io_module.load_route_msgs = lambda route, qlog=False: []
  previous_argv = sys.argv
  try:
    sys.argv = ["fuzz_lateral_route_replay.py", "--route", "fake/route", "--cases", "1"]
    with pytest.raises(SystemExit) as exc:
      main()
    assert exc.value.code != 0
  finally:
    route_io_module.load_route_msgs = original_load_route_msgs
    sys.argv = previous_argv


def test_main_list_only_outputs_extraction_summary(monkeypatch):
  baseline = _fake_route_messages(5, desired_curvature=0.0005)
  original_load_route_msgs = route_io_module.load_route_msgs
  route_io_module.load_route_msgs = lambda route, qlog=False: baseline
  previous_argv = sys.argv
  stdout = io.StringIO()
  try:
    sys.argv = ["fuzz_lateral_route_replay.py", "--route", "fake/route", "--list-only", "--json"]
    with contextlib.redirect_stdout(stdout):
      main()
  finally:
    route_io_module.load_route_msgs = original_load_route_msgs
    sys.argv = previous_argv

  payload = json.loads(stdout.getvalue())
  assert payload["route"] == "fake/route"
  assert payload["extracted_count"] == 5


def test_main_route_and_preset_are_mutually_exclusive(monkeypatch):
  original_load_route_msgs = route_io_module.load_route_msgs
  route_io_module.load_route_msgs = lambda route, qlog=False: []
  previous_argv = sys.argv
  try:
    sys.argv = [
      "fuzz_lateral_route_replay.py",
      "--route", "fake/route",
      "--preset", "synthetic_curve",
      "--cases", "1",
    ]
    with pytest.raises(SystemExit):
      main()
  finally:
    route_io_module.load_route_msgs = original_load_route_msgs
    sys.argv = previous_argv


def test_list_only_json_includes_quality(monkeypatch):
  baseline = _fake_route_messages(5, desired_curvature=0.0005)
  original_load_route_msgs = route_io_module.load_route_msgs
  route_io_module.load_route_msgs = lambda route, qlog=False: baseline
  previous_argv = sys.argv
  stdout = io.StringIO()
  try:
    sys.argv = ["fuzz_lateral_route_replay.py", "--route", "fake/route", "--list-only", "--json"]
    with contextlib.redirect_stdout(stdout):
      main()
  finally:
    route_io_module.load_route_msgs = original_load_route_msgs
    sys.argv = previous_argv

  payload = json.loads(stdout.getvalue())
  assert "quality" in payload
  assert payload["quality"]["input_message_count"] == len(baseline)
  assert payload["quality"]["controls_state_seen"] == 5


def test_route_mode_json_preserves_quality_and_timing_metadata(monkeypatch):
  baseline = _fake_route_messages(6, desired_curvature=0.0005)
  original_load_route_msgs = route_io_module.load_route_msgs
  route_io_module.load_route_msgs = lambda route, qlog=False: baseline
  previous_argv = sys.argv
  stdout = io.StringIO()
  try:
    sys.argv = [
      "fuzz_lateral_route_replay.py",
      "--route", "fake/route",
      "--qlog",
      "--cases", "1",
      "--perturbation", "none",
      "--json",
    ]
    with contextlib.redirect_stdout(stdout):
      main()
  finally:
    route_io_module.load_route_msgs = original_load_route_msgs
    sys.argv = previous_argv

  payload = json.loads(stdout.getvalue())
  assert payload["duration"] is None
  meta = payload["route_metadata"]
  assert meta["route"] == "fake/route"
  assert meta["qlog"] is True
  assert meta["timing_mode"] == "fixed-dt"
  assert "quality" in meta
  assert meta["quality_scope"] == "extracted_frames"
  assert meta["quality"]["controls_state_seen"] == 6


def test_route_artifact_preserves_quality_and_metadata(monkeypatch):
  baseline = _fake_route_messages(8, desired_curvature=0.0005)
  original_load_route_msgs = route_io_module.load_route_msgs
  route_io_module.load_route_msgs = lambda route, qlog=False: baseline
  try:
    scenarios = generate_scenarios(RouteReplayFuzzerConfig(seed=3, cases=1, route="fake/route", qlog=True, perturbation="none"))
    result = evaluate_scenario(scenarios[0])
    with tempfile.TemporaryDirectory() as tmpdir:
      path = write_artifact(result, Path(tmpdir), seed=3, index=0)
      data = json.loads(path.read_text())
      assert data["route_metadata"]["quality"]["controls_state_seen"] == 8
      assert data["route_metadata"]["timing_mode"] == "fixed-dt"
  finally:
    route_io_module.load_route_msgs = original_load_route_msgs


def test_default_prefix_sampling_matches_current_extraction(monkeypatch):
  baseline = _fake_route_messages(12, desired_curvature=0.0005)
  original_load_route_msgs = route_io_module.load_route_msgs
  route_io_module.load_route_msgs = lambda route, qlog=False: baseline
  try:
    scenarios = generate_scenarios(RouteReplayFuzzerConfig(seed=4, cases=2, route="fake/route", max_frames=8))
    assert len(scenarios) == 2
    assert all(len(s.frames) == 8 for s in scenarios)
    assert all(s.route_metadata.sampling_mode == "prefix" for s in scenarios)
    assert all(s.route_metadata.sampling_window_index is None for s in scenarios)
  finally:
    route_io_module.load_route_msgs = original_load_route_msgs


def test_window_applies_before_sampling(monkeypatch):
  baseline = _fake_route_messages(20, desired_curvature=0.0005)
  original_load_route_msgs = route_io_module.load_route_msgs
  route_io_module.load_route_msgs = lambda route, qlog=False: baseline
  try:
    scenarios = generate_scenarios(
      RouteReplayFuzzerConfig(
        seed=5,
        cases=1,
        route="fake/route",
        window_start_s=0.02,
        window_end_s=0.08,
        sample_mode="uniform-windows",
        sample_window_duration_s=0.03,
        sample_window_count=2,
      )
    )
    assert len(scenarios) == 1
    meta = scenarios[0].route_metadata
    assert meta.window_start_s == pytest.approx(0.02)
    assert meta.window_end_s == pytest.approx(0.08)
    assert meta.selected_original_start_s >= 0.02 - 1e-9
    assert meta.selected_original_end_s <= 0.08 + 1e-9
  finally:
    route_io_module.load_route_msgs = original_load_route_msgs


def test_random_window_sampling_is_deterministic(monkeypatch):
  baseline = _fake_route_messages(30, desired_curvature=0.0005)
  original_load_route_msgs = route_io_module.load_route_msgs
  route_io_module.load_route_msgs = lambda route, qlog=False: baseline
  try:
    def run():
      return generate_scenarios(
        RouteReplayFuzzerConfig(
          seed=6,
          cases=1,
          route="fake/route",
          sample_mode="random-window",
          sample_window_duration_s=0.05,
          sample_window_count=2,
          sample_seed=42,
        )
      )
    first = run()
    second = run()
    assert len(first) == len(second) == 1
    assert first[0].route_metadata.selected_original_start_s == second[0].route_metadata.selected_original_start_s
    assert first[0].route_metadata.selected_original_end_s == second[0].route_metadata.selected_original_end_s
  finally:
    route_io_module.load_route_msgs = original_load_route_msgs


def test_uniform_windows_produces_expected_metadata(monkeypatch):
  baseline = _fake_route_messages(20, desired_curvature=0.0005)
  original_load_route_msgs = route_io_module.load_route_msgs
  route_io_module.load_route_msgs = lambda route, qlog=False: baseline
  try:
    scenarios = generate_scenarios(
      RouteReplayFuzzerConfig(
        seed=7,
        cases=4,
        route="fake/route",
        sample_mode="uniform-windows",
        sample_window_duration_s=0.05,
        sample_window_count=2,
      )
    )
    assert len(scenarios) == 4
    window_indices = [s.route_metadata.sampling_window_index for s in scenarios]
    assert window_indices == [0, 1, 0, 1]
    metas = [s.route_metadata for s in scenarios[:2]]
    assert metas[0].selected_original_start_s < metas[1].selected_original_start_s
    assert metas[0].sampling_window_count == 2
    assert metas[1].sampling_window_count == 2
    assert metas[0].quality_scope == "base_extraction"
    assert metas[1].quality_scope == "base_extraction"
  finally:
    route_io_module.load_route_msgs = original_load_route_msgs


def test_route_mode_json_includes_sampled_windows(monkeypatch):
  baseline = _fake_route_messages(25, desired_curvature=0.0005)
  original_load_route_msgs = route_io_module.load_route_msgs
  route_io_module.load_route_msgs = lambda route, qlog=False: baseline
  previous_argv = sys.argv
  stdout = io.StringIO()
  try:
    sys.argv = [
      "fuzz_lateral_route_replay.py",
      "--route", "fake/route",
      "--cases", "2",
      "--perturbation", "none",
      "--sample-mode", "uniform-windows",
      "--sample-window-duration", "0.05",
      "--sample-window-count", "2",
      "--json",
    ]
    with contextlib.redirect_stdout(stdout):
      main()
  finally:
    route_io_module.load_route_msgs = original_load_route_msgs
    sys.argv = previous_argv

  payload = json.loads(stdout.getvalue())
  assert len(payload["sampled_windows"]) == 2
  assert [window["sampling_window_index"] for window in payload["sampled_windows"]] == [0, 1]
  assert {window["quality_scope"] for window in payload["sampled_windows"]} == {"base_extraction"}


def test_list_only_summarizes_sampled_windows(monkeypatch):
  baseline = _fake_route_messages(25, desired_curvature=0.0005)
  original_load_route_msgs = route_io_module.load_route_msgs
  route_io_module.load_route_msgs = lambda route, qlog=False: baseline
  previous_argv = sys.argv
  stdout = io.StringIO()
  try:
    sys.argv = [
      "fuzz_lateral_route_replay.py",
      "--route", "fake/route",
      "--list-only",
      "--json",
      "--sample-mode", "uniform-windows",
      "--sample-window-duration", "0.05",
      "--sample-window-count", "2",
    ]
    with contextlib.redirect_stdout(stdout):
      main()
  finally:
    route_io_module.load_route_msgs = original_load_route_msgs
    sys.argv = previous_argv

  payload = json.loads(stdout.getvalue())
  assert "windows" in payload
  assert len(payload["windows"]) == 2
  for window in payload["windows"]:
    assert window["sampling_mode"] == "uniform-windows"
    assert "sampling_window_index" in window
    assert "selected_original_start_s" in window
    assert "selected_original_end_s" in window


def test_invalid_sampling_args_fail_before_loading():
  loaded: list[bool] = []
  original_load_route_msgs = route_io_module.load_route_msgs
  def capture_load(route, qlog=False):
    loaded.append(True)
    return []
  route_io_module.load_route_msgs = capture_load
  previous_argv = sys.argv
  try:
    sys.argv = [
      "fuzz_lateral_route_replay.py",
      "--route", "fake/route",
      "--sample-mode", "random-window",
      "--cases", "1",
    ]
    with pytest.raises(SystemExit):
      main()
    assert not loaded

    loaded.clear()
    sys.argv = [
      "fuzz_lateral_route_replay.py",
      "--route", "fake/route",
      "--sample-mode", "uniform-windows",
      "--sample-window-duration", "0",
      "--cases", "1",
    ]
    with pytest.raises(SystemExit):
      main()
    assert not loaded

    loaded.clear()
    sys.argv = [
      "fuzz_lateral_route_replay.py",
      "--route", "fake/route",
      "--sample-window-count", "0",
      "--cases", "1",
    ]
    with pytest.raises(SystemExit):
      main()
    assert not loaded
  finally:
    route_io_module.load_route_msgs = original_load_route_msgs
    sys.argv = previous_argv


def test_timing_original_fails_before_loading():
  loaded: list[bool] = []
  original_load_route_msgs = route_io_module.load_route_msgs
  def capture_load(route, qlog=False):
    loaded.append(True)
    return []
  route_io_module.load_route_msgs = capture_load
  previous_argv = sys.argv
  try:
    sys.argv = [
      "fuzz_lateral_route_replay.py",
      "--route", "fake/route",
      "--timing", "original",
      "--cases", "1",
    ]
    with pytest.raises(SystemExit):
      main()
    assert not loaded
  finally:
    route_io_module.load_route_msgs = original_load_route_msgs
    sys.argv = previous_argv


def test_api_rejects_original_timing_before_fake_fixed_dt_metadata():
  with pytest.raises(ValueError, match="original"):
    extract_lateral_route_frames_with_summary(_fake_route_messages(2), timing_mode="original")


def test_api_rejects_unknown_extraction_sampling_metadata():
  with pytest.raises(ValueError, match="sampling_mode"):
    extract_lateral_route_frames_with_summary(_fake_route_messages(2), sampling_mode="invalid")


def test_generate_scenarios_rejects_route_original_timing_before_loading(monkeypatch):
  loaded: list[bool] = []
  original_load_route_msgs = route_io_module.load_route_msgs
  def capture_load(route, qlog=False):
    loaded.append(True)
    return []
  route_io_module.load_route_msgs = capture_load
  try:
    with pytest.raises(ValueError, match="original"):
      generate_scenarios(RouteReplayFuzzerConfig(route="fake/route", timing_mode="original"))
    assert not loaded
  finally:
    route_io_module.load_route_msgs = original_load_route_msgs


def test_generate_scenarios_rejects_invalid_route_sampling_before_loading(monkeypatch):
  loaded: list[bool] = []
  original_load_route_msgs = route_io_module.load_route_msgs
  def capture_load(route, qlog=False):
    loaded.append(True)
    return []
  route_io_module.load_route_msgs = capture_load
  try:
    with pytest.raises(ValueError, match="sample_mode"):
      generate_scenarios(RouteReplayFuzzerConfig(route="fake/route", sample_mode="invalid"))
    with pytest.raises(ValueError, match="requires"):
      generate_scenarios(RouteReplayFuzzerConfig(route="fake/route", sample_mode="random-window"))
    with pytest.raises(ValueError, match="> 0"):
      generate_scenarios(RouteReplayFuzzerConfig(route="fake/route", sample_window_count=0))
    assert not loaded
  finally:
    route_io_module.load_route_msgs = original_load_route_msgs


def test_generate_scenarios_returns_no_sampled_windows_for_empty_route(monkeypatch):
  original_load_route_msgs = route_io_module.load_route_msgs
  route_io_module.load_route_msgs = lambda route, qlog=False: []
  try:
    scenarios = generate_scenarios(
      RouteReplayFuzzerConfig(
        route="fake/route",
        sample_mode="uniform-windows",
        sample_window_duration_s=0.05,
        sample_window_count=2,
      )
    )
    assert scenarios == []
  finally:
    route_io_module.load_route_msgs = original_load_route_msgs


def test_list_only_sampled_empty_route_exits_cleanly_without_format_traceback(monkeypatch):
  original_load_route_msgs = route_io_module.load_route_msgs
  route_io_module.load_route_msgs = lambda route, qlog=False: []
  previous_argv = sys.argv
  stdout = io.StringIO()
  try:
    sys.argv = [
      "fuzz_lateral_route_replay.py",
      "--route", "fake/route",
      "--list-only",
      "--sample-mode", "uniform-windows",
      "--sample-window-duration", "0.05",
      "--sample-window-count", "2",
    ]
    with pytest.raises(SystemExit) as exc, contextlib.redirect_stdout(stdout):
      main()
    assert exc.value.code == 1
    assert "sampled windows" in stdout.getvalue()
  finally:
    route_io_module.load_route_msgs = original_load_route_msgs
    sys.argv = previous_argv


def test_main_bad_window_format_exits_before_route_loading():
  loaded: list[bool] = []
  original_load_route_msgs = route_io_module.load_route_msgs
  def capture_load(route, qlog=False):
    loaded.append(True)
    return []
  route_io_module.load_route_msgs = capture_load
  previous_argv = sys.argv
  try:
    sys.argv = ["fuzz_lateral_route_replay.py", "--route", "fake/route", "--window", "bad", "--cases", "1"]
    with pytest.raises(SystemExit):
      main()
    assert not loaded
  finally:
    route_io_module.load_route_msgs = original_load_route_msgs
    sys.argv = previous_argv


def test_main_window_start_greater_or_equal_end_exits():
  previous_argv = sys.argv
  try:
    sys.argv = ["fuzz_lateral_route_replay.py", "--route", "fake/route", "--window", "1.0,1.0", "--cases", "1"]
    with pytest.raises(SystemExit):
      main()
    sys.argv = ["fuzz_lateral_route_replay.py", "--route", "fake/route", "--window", "2.0,1.0", "--cases", "1"]
    with pytest.raises(SystemExit):
      main()
  finally:
    sys.argv = previous_argv


def test_main_max_frames_zero_exits():
  previous_argv = sys.argv
  try:
    sys.argv = ["fuzz_lateral_route_replay.py", "--route", "fake/route", "--max-frames", "0", "--cases", "1"]
    with pytest.raises(SystemExit):
      main()
  finally:
    sys.argv = previous_argv


def test_main_route_cases_zero_exits():
  previous_argv = sys.argv
  try:
    sys.argv = ["fuzz_lateral_route_replay.py", "--route", "fake/route", "--cases", "0"]
    with pytest.raises(SystemExit):
      main()
  finally:
    sys.argv = previous_argv


def test_main_route_json_output_has_null_duration_and_route_metadata(monkeypatch):
  baseline = _fake_route_messages(8, desired_curvature=0.0005)
  original_load_route_msgs = route_io_module.load_route_msgs
  route_io_module.load_route_msgs = lambda route, qlog=False: baseline
  previous_argv = sys.argv
  stdout = io.StringIO()
  try:
    sys.argv = [
      "fuzz_lateral_route_replay.py",
      "--route", "fake/route",
      "--qlog",
      "--window", "0.0,0.1",
      "--max-frames", "8",
      "--cases", "1",
      "--perturbation", "none",
      "--json",
    ]
    with contextlib.redirect_stdout(stdout):
      main()
  finally:
    route_io_module.load_route_msgs = original_load_route_msgs
    sys.argv = previous_argv

  payload = json.loads(stdout.getvalue())
  assert payload["duration"] is None
  assert payload["route_metadata"]["route"] == "fake/route"
  assert payload["route_metadata"]["qlog"] is True
  assert payload["route_metadata"]["window_start_s"] == pytest.approx(0.0)
  assert payload["route_metadata"]["window_end_s"] == pytest.approx(0.1)
  assert payload["route_metadata"]["max_frames"] == 8
  assert payload["route_metadata"]["extracted_count"] == 8
  assert payload["route_metadata"]["dt"] == DT


def test_main_route_text_output_uses_fixed_dt_span(monkeypatch):
  baseline = _fake_route_messages(8, desired_curvature=0.0005)
  original_load_route_msgs = route_io_module.load_route_msgs
  route_io_module.load_route_msgs = lambda route, qlog=False: baseline
  previous_argv = sys.argv
  stdout = io.StringIO()
  try:
    sys.argv = [
      "fuzz_lateral_route_replay.py",
      "--route", "fake/route",
      "--cases", "1",
      "--perturbation", "none",
    ]
    with contextlib.redirect_stdout(stdout):
      main()
  finally:
    route_io_module.load_route_msgs = original_load_route_msgs
    sys.argv = previous_argv

  output = stdout.getvalue()
  assert "route_fixed_dt_span=" in output
  assert " duration=" not in output
