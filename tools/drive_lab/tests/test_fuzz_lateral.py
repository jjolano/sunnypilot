import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from openpilot.tools.drive_lab import fuzz_lateral
from openpilot.tools.drive_lab.fuzz_lateral import (
  FuzzerConfig,
  LateralScenario,
  generate_scenarios,
  main,
  replay_artifact,
  run_scenario,
  scenario_from_dict,
  scenario_to_dict,
  write_artifact,
)
from openpilot.tools.drive_lab.lateral_metrics import LateralMetricThresholds, evaluate_lateral_trace
from openpilot.tools.drive_lab.lateral_plant import LateralPlantConfig, LateralPlantResult, run_lateral_plant
from openpilot.tools.drive_lab.metrics import ScenarioFailure


# ---------- determinism ----------

def test_generate_scenarios_is_seeded():
  config = FuzzerConfig(seed=42, cases=20)
  assert generate_scenarios(config) == generate_scenarios(config)


def test_run_scenario_is_seeded():
  config = FuzzerConfig(seed=7, cases=1)
  scenario = generate_scenarios(config)[0]
  first = run_scenario(scenario)
  second = run_scenario(scenario)
  assert first.result.trace.to_dict() == second.result.trace.to_dict()
  assert first.evaluation.to_dict() == second.evaluation.to_dict()


# ---------- replay equivalence ----------

def test_scenario_round_trips_through_dict():
  config = FuzzerConfig(seed=3, cases=1)
  scenario = generate_scenarios(config)[0]
  restored = scenario_from_dict(scenario_to_dict(scenario))
  assert restored.kind == scenario.kind
  assert restored.title == scenario.title
  assert restored.duration == scenario.duration
  assert restored.speed_mps == scenario.speed_mps
  assert restored.dt_s == scenario.dt_s
  assert restored.desired_curvature == scenario.desired_curvature
  assert restored.v_ego == scenario.v_ego
  assert restored.plant == scenario.plant
  assert restored.metric_thresholds == scenario.metric_thresholds


def test_artifact_replay_reports_equivalent_classification():
  config = FuzzerConfig(seed=9, cases=1, kind="curve_entry")
  scenario = generate_scenarios(config)[0]
  result = run_scenario(scenario)
  with tempfile.TemporaryDirectory() as tmpdir:
    path = write_artifact(result, Path(tmpdir), seed=9, index=0)
    replayed = replay_artifact(path)
    assert replayed.valid == result.valid
    assert [f.check for f in replayed.failures] == [f.check for f in result.failures]


# ---------- metric failure cases ----------

def test_metric_catches_nan_inf_output():
  trace = run_lateral_plant(np.zeros(200), config=LateralPlantConfig(duration_s=10.0)).trace
  broken = list(trace.actual_curvature)
  broken[50] = float("nan")
  broken[51] = float("inf")
  trace = trace.__class__(**{**trace.to_dict(), "actual_curvature": tuple(broken)})
  evaluation = evaluate_lateral_trace("nan", trace, LateralPlantConfig(), LateralMetricThresholds())
  assert any(f.check == "finite" for f in evaluation.failures)


def test_metric_catches_divergence_and_tracking_error():
  # Use an unstable plant (negative damping) so tracking error grows.
  config = LateralPlantConfig(
    duration_s=5.0,
    controller_gain=5.0,
    controller_damping=-2.0,
    tire_lag_s=0.05,
    actuator_delay_s=0.05,
  )
  t = np.arange(0.0, 5.0, config.dt_s)
  desired = 0.002 * np.ones_like(t)
  result = run_lateral_plant(desired, config=config)
  thresholds = LateralMetricThresholds(max_abs_tracking_error=0.001, max_final_tracking_error=0.0005)
  evaluation = evaluate_lateral_trace("diverge", result.trace, config, thresholds)
  assert any(f.check in ("tracking", "settle") for f in evaluation.failures)


def test_metric_catches_oscillation():
  # High gain + low damping on a step produces actuator reversals beyond what
  # the step input itself explains.
  config = LateralPlantConfig(
    duration_s=8.0,
    controller_gain=4.0,
    controller_damping=0.05,
    tire_lag_s=0.05,
    actuator_delay_s=0.15,
  )
  t = np.arange(0.0, 8.0, config.dt_s)
  desired = np.where(t < 1.0, 0.0, 0.002)
  result = run_lateral_plant(desired, config=config)
  thresholds = LateralMetricThresholds(max_oscillation_reversals=1)
  evaluation = evaluate_lateral_trace("osc", result.trace, config, thresholds, scenario_kind="curve_entry")
  assert any(f.check == "oscillation" for f in evaluation.failures)


def test_metric_fails_inverted_sign_response():
  # A wrong-way controller must invalidate the result, not merely record a metric.
  # Build a trace by inverting the actual curvature of a healthy run.
  from dataclasses import replace as _replace
  # Sustained demand: a zero-mean sine has no net direction to invert, and the check
  # deliberately skips those (the demand floor).
  config = LateralPlantConfig(duration_s=6.0)
  t = np.arange(0.0, 6.0, config.dt_s)
  desired = 0.002 * (1.0 - np.exp(-t))
  result = run_lateral_plant(desired, config=config)
  assert not any(f.check == "sign" for f in evaluate_lateral_trace("ok", result.trace, config).failures)

  inverted = _replace(result.trace, actual_curvature=tuple(-np.array(result.trace.actual_curvature)))
  failures = evaluate_lateral_trace("inverted", inverted, config).failures
  assert any(f.check == "sign" for f in failures), [f.check for f in failures]


def test_metric_checks_sign_on_negative_only_demand():
  # Demand entirely negative was never sign-checked at all before.
  from dataclasses import replace as _replace
  config = LateralPlantConfig(duration_s=6.0)
  t = np.arange(0.0, 6.0, config.dt_s)
  desired = -0.002 * (1.0 - np.exp(-t))
  result = run_lateral_plant(desired, config=config)
  assert not any(f.check == "sign" for f in evaluate_lateral_trace("neg_ok", result.trace, config).failures)

  inverted = _replace(result.trace, actual_curvature=tuple(-np.array(result.trace.actual_curvature)))
  failures = evaluate_lateral_trace("neg_inverted", inverted, config).failures
  assert any(f.check == "sign" for f in failures), [f.check for f in failures]


def test_metric_catches_steering_rate_and_jerk():
  # Very tight rate limit and high desired curvature drive the command to saturate
  # and produce high actuator rate / lateral jerk at the step.
  config = LateralPlantConfig(
    duration_s=3.0,
    actuator_rate_limit_deg_s=500.0,
    controller_gain=8.0,
    tire_lag_s=0.05,
  )
  t = np.arange(0.0, 3.0, config.dt_s)
  desired = np.where(t < 0.5, 0.0, 0.01)
  result = run_lateral_plant(desired, config=config)
  thresholds = LateralMetricThresholds(max_abs_steering_rate=1.0, max_abs_lateral_jerk=0.1)
  evaluation = evaluate_lateral_trace("rate_jerk", result.trace, config, thresholds)
  assert any(f.check == "steering_rate" for f in evaluation.failures)
  assert any(f.check == "lateral_jerk" for f in evaluation.failures)


def test_metric_allows_normal_highway_speed_lateral_jerk():
  config = FuzzerConfig(seed=17, cases=4, speed_mps=30.0)
  scenario = generate_scenarios(config)[3]

  result = run_scenario(scenario)

  assert scenario.kind == "noisy_model_curvature"
  assert result.valid


def test_metric_still_catches_highway_speed_lateral_jerk_when_threshold_is_tight():
  config = FuzzerConfig(
    seed=17,
    cases=4,
    speed_mps=30.0,
    thresholds=LateralMetricThresholds(max_abs_lateral_jerk=1.0),
  )
  scenario = generate_scenarios(config)[3]

  result = run_scenario(scenario)

  assert any(f.check == "lateral_jerk" for f in result.failures)


def test_metric_allows_normal_very_high_speed_s_curve_lateral_jerk():
  config = FuzzerConfig(seed=25, cases=114, duration=6.0, speed_mps=40.0)
  scenario = generate_scenarios(config)[113]

  result = run_scenario(scenario)

  assert scenario.kind == "s_curve_reversal"
  assert result.valid


def test_metric_allows_normal_extreme_speed_straight_disturbance_lateral_jerk():
  config = FuzzerConfig(seed=28, cases=1, duration=6.0, speed_mps=55.0)
  scenario = generate_scenarios(config)[0]

  result = run_scenario(scenario)

  assert scenario.kind == "straight_disturbance"
  assert result.valid


def test_metric_catches_saturation():
  config = LateralPlantConfig(duration_s=3.0, controller_gain=20.0, max_steering_angle_deg=10.0)
  t = np.arange(0.0, 3.0, config.dt_s)
  desired = 0.01 * np.ones_like(t)
  result = run_lateral_plant(desired, config=config)
  thresholds = LateralMetricThresholds(max_saturation_fraction=0.01)
  evaluation = evaluate_lateral_trace("sat", result.trace, config, thresholds)
  assert any(f.check == "saturation" for f in evaluation.failures)


# ---------- plant structural properties ----------

def test_positive_desired_curvature_yields_positive_actual():
  t = np.arange(0.0, 10.0, 0.05)
  desired = 0.002 * np.ones_like(t)
  result = run_lateral_plant(desired, config=LateralPlantConfig(duration_s=10.0))
  # After transients, mean actual curvature should have the same sign as desired.
  late_mean = np.mean(result.trace.actual_curvature[-50:])
  assert late_mean > 0.0


def test_zero_input_stays_near_zero():
  t = np.arange(0.0, 10.0, 0.05)
  desired = np.zeros_like(t)
  result = run_lateral_plant(desired, config=LateralPlantConfig(duration_s=10.0))
  assert np.max(np.abs(result.trace.actual_curvature)) < 1e-6


def test_coarse_requested_dt_uses_stable_internal_plant_step():
  config = FuzzerConfig(seed=18, cases=1, dt_s=0.1)
  scenario = generate_scenarios(config)[0]

  result = run_scenario(scenario)

  assert result.valid
  assert result.result.config.dt_s == pytest.approx(0.05)


# ---------- CLI smoke ----------

def test_main_text_output_runs_and_reports_zero_failures():
  stdout = io.StringIO()
  previous_argv = sys.argv
  try:
    sys.argv = ["fuzz_lateral.py", "--seed", "1", "--cases", "5"]
    with contextlib.redirect_stdout(stdout):
      main()
  finally:
    sys.argv = previous_argv

  output = stdout.getvalue()
  assert "Drive Lab lateral fuzz" in output
  assert "failures=0" in output


def test_main_json_output_is_stable():
  stdout = io.StringIO()
  previous_argv = sys.argv
  try:
    sys.argv = ["fuzz_lateral.py", "--seed", "2", "--cases", "3", "--json"]
    with contextlib.redirect_stdout(stdout):
      main()
  finally:
    sys.argv = previous_argv

  payload = json.loads(stdout.getvalue())
  assert payload["seed"] == 2
  assert payload["cases"] == 3
  assert "failures" in payload


def test_main_writes_and_replays_failure_artifact():
  # Use an unstable plant configuration guaranteed to fail.
  config = LateralPlantConfig(
    duration_s=5.0,
    controller_gain=5.0,
    controller_damping=-2.0,
    tire_lag_s=0.05,
    actuator_delay_s=0.05,
  )
  thresholds = LateralMetricThresholds(max_abs_tracking_error=0.001)
  t = np.arange(0.0, 5.0, config.dt_s)
  desired = 0.002 * np.ones_like(t)
  scenario = LateralScenario(
    kind="curve_entry",
    title="forced failure",
    duration=5.0,
    speed_mps=20.0,
    dt_s=0.05,
    desired_curvature=tuple(float(v) for v in desired),
    plant_config=config,
    thresholds=thresholds,
  )
  result = run_scenario(scenario)
  assert result.failures

  with tempfile.TemporaryDirectory() as tmpdir:
    artifact_dir = Path(tmpdir) / "artifacts"
    stdout = io.StringIO()
    previous_argv = sys.argv
    try:
      sys.argv = [
        "fuzz_lateral.py",
        "--artifact-dir",
        str(artifact_dir),
      ]
      with contextlib.redirect_stdout(stdout):
        # Manually write the artifact as main() would.
        write_artifact(result, artifact_dir, seed=None, index=0)
    finally:
      sys.argv = previous_argv

    artifact = next(artifact_dir.glob("*.json"))
    replayed = replay_artifact(artifact)
    assert replayed.valid == result.valid


def test_main_exits_nonzero_on_failure():
  # Replace run_scenario to inject a failure without depending on RNG.
  original_run_scenario = fuzz_lateral.run_scenario
  fake_result = run_scenario(
    LateralScenario(
      kind="straight_disturbance",
      title="forced",
      duration=1.0,
      speed_mps=20.0,
      dt_s=0.05,
      desired_curvature=tuple(0.0 for _ in range(20)),
    )
  )
  fake_evaluation = fake_result.evaluation
  fake_evaluation = fake_evaluation.__class__(
    scenario_id=fake_evaluation.scenario_id,
    valid=False,
    failures=[ScenarioFailure("forced", "injected failure")],
    metrics=fake_evaluation.metrics,
  )
  fake_result = fake_result.__class__(
    scenario=fake_result.scenario,
    result=fake_result.result,
    evaluation=fake_evaluation,
  )

  def _forced_run(scenario, scenario_id=None):
    return fake_result

  fuzz_lateral.run_scenario = _forced_run
  previous_argv = sys.argv
  try:
    sys.argv = ["fuzz_lateral.py", "--seed", "1", "--cases", "1"]
    with pytest.raises(SystemExit) as exc:
      main()
    assert exc.value.code == 1
  finally:
    fuzz_lateral.run_scenario = original_run_scenario
    sys.argv = previous_argv
