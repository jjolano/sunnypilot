#!/usr/bin/env python3
"""Seeded lateral structural stability fuzzer.

Runs synthetic closed-loop lateral scenarios through a deterministic test plant
and checks for structural failures: NaN/inf, divergence, excessive steering
rate/jerk, saturation, and oscillation. Designed to mirror the CLI and
report style of fuzz_longitudinal.py.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from openpilot.tools.drive_lab.lateral_metrics import LateralMetricThresholds, evaluate_lateral_trace
from openpilot.tools.drive_lab.lateral_plant import LateralPlantConfig, LateralPlantResult, run_lateral_plant
from openpilot.tools.drive_lab.metrics import EvaluationResult, ScenarioFailure


ARTIFACT_SCHEMA = "drive-lab-lateral-fuzzer-artifact"
ARTIFACT_VERSION = 1
SCENARIO_KINDS = (
  "straight_disturbance",
  "curve_entry",
  "s_curve_reversal",
  "step_correction",
  "noisy_model_curvature",
)


@dataclass(frozen=True)
class LateralScenario:
  kind: str
  title: str
  duration: float
  speed_mps: float
  dt_s: float
  desired_curvature: tuple[float, ...]
  v_ego: tuple[float, ...] = ()
  plant_config: LateralPlantConfig | None = None
  thresholds: LateralMetricThresholds | None = None

  @property
  def plant(self) -> LateralPlantConfig:
    return self.plant_config or LateralPlantConfig(dt_s=self.dt_s, speed_mps=self.speed_mps, duration_s=self.duration)

  @property
  def metric_thresholds(self) -> LateralMetricThresholds:
    return self.thresholds or LateralMetricThresholds()


@dataclass(frozen=True)
class LateralScenarioResult:
  scenario: LateralScenario
  result: LateralPlantResult
  evaluation: EvaluationResult

  @property
  def failures(self) -> list[ScenarioFailure]:
    return list(self.evaluation.failures)

  @property
  def valid(self) -> bool:
    return self.evaluation.valid


def _time_array(duration: float, dt: float) -> np.ndarray:
  return np.arange(0.0, max(duration, dt) + dt * 0.5, dt, dtype=float)


def _generate_straight_disturbance(rng: random.Random, idx: int, speed: float, dt: float, duration: float) -> LateralScenario:
  t = _time_array(duration, dt)
  desired = np.zeros_like(t)
  # Single short curvature impulse to excite the plant, then return to straight.
  impulse_start = rng.uniform(1.0, max(1.5, duration * 0.3))
  impulse_width = rng.uniform(0.2, 0.6)
  impulse_mag = rng.choice([-1.0, 1.0]) * rng.uniform(0.0005, 0.0020)
  mask = (t >= impulse_start) & (t < impulse_start + impulse_width)
  desired[mask] = impulse_mag
  return LateralScenario(
    kind="straight_disturbance",
    title=f"fuzz straight disturbance #{idx}",
    duration=duration,
    speed_mps=speed,
    dt_s=dt,
    desired_curvature=tuple(float(v) for v in desired),
  )


def _generate_curve_entry(rng: random.Random, idx: int, speed: float, dt: float, duration: float) -> LateralScenario:
  t = _time_array(duration, dt)
  desired = np.zeros_like(t)
  entry_start = rng.uniform(1.0, max(1.5, duration * 0.25))
  entry_rise = rng.uniform(0.5, 2.0)
  curvature = rng.uniform(0.0010, 0.0040) * rng.choice([-1.0, 1.0])
  desired = curvature / (1.0 + np.exp(-(t - entry_start) / max(entry_rise * 0.4, 0.1)))
  return LateralScenario(
    kind="curve_entry",
    title=f"fuzz curve entry #{idx}",
    duration=duration,
    speed_mps=speed,
    dt_s=dt,
    desired_curvature=tuple(float(v) for v in desired),
  )


def _generate_s_curve_reversal(rng: random.Random, idx: int, speed: float, dt: float, duration: float) -> LateralScenario:
  t = _time_array(duration, dt)
  curvature = rng.uniform(0.0010, 0.0030)
  t1 = rng.uniform(1.0, max(1.5, duration * 0.25))
  t2 = t1 + rng.uniform(1.0, max(1.5, duration * 0.25))
  desired = np.where(t < t1, curvature, np.where(t < t2, -curvature, curvature))
  # Smooth the transitions with a short ramp.
  ramp = rng.uniform(0.2, 0.5)
  for tp in (t1, t2):
    left = desired.copy()
    left[t >= tp] = 0.0
    right = desired.copy()
    right[t < tp] = 0.0
    blend = 0.5 * (1.0 + np.tanh((t - tp) / max(ramp, dt)))
    desired = left * (1.0 - blend) + right * blend
  return LateralScenario(
    kind="s_curve_reversal",
    title=f"fuzz s-curve reversal #{idx}",
    duration=duration,
    speed_mps=speed,
    dt_s=dt,
    desired_curvature=tuple(float(v) for v in desired),
  )


def _generate_step_correction(rng: random.Random, idx: int, speed: float, dt: float, duration: float) -> LateralScenario:
  t = _time_array(duration, dt)
  base_curvature = rng.uniform(0.0010, 0.0030) * rng.choice([-1.0, 1.0])
  step_time = rng.uniform(1.0, max(1.5, duration * 0.4))
  step_mag = rng.uniform(0.0005, 0.0020) * rng.choice([-1.0, 1.0])
  desired = np.full_like(t, base_curvature)
  desired[t >= step_time] = base_curvature + step_mag
  return LateralScenario(
    kind="step_correction",
    title=f"fuzz step correction #{idx}",
    duration=duration,
    speed_mps=speed,
    dt_s=dt,
    desired_curvature=tuple(float(v) for v in desired),
  )


def _generate_noisy_model_curvature(rng: random.Random, idx: int, speed: float, dt: float, duration: float) -> LateralScenario:
  t = _time_array(duration, dt)
  base = rng.uniform(-0.0010, 0.0010)
  amplitude = rng.uniform(0.0005, 0.0020)
  freq = rng.uniform(0.1, 0.5)
  phase = rng.uniform(0.0, 2.0 * math.pi)
  noise = rng.choice([-1.0, 1.0]) * amplitude * np.sin(2.0 * math.pi * freq * t + phase)
  desired = base + noise
  return LateralScenario(
    kind="noisy_model_curvature",
    title=f"fuzz noisy model curvature #{idx}",
    duration=duration,
    speed_mps=speed,
    dt_s=dt,
    desired_curvature=tuple(float(v) for v in desired),
  )


SCENARIO_GENERATORS: dict[str, Any] = {
  "straight_disturbance": _generate_straight_disturbance,
  "curve_entry": _generate_curve_entry,
  "s_curve_reversal": _generate_s_curve_reversal,
  "step_correction": _generate_step_correction,
  "noisy_model_curvature": _generate_noisy_model_curvature,
}


@dataclass
class FuzzerConfig:
  seed: int = 1
  cases: int = 100
  kind: str | None = None
  duration: float = 10.0
  dt_s: float = 0.05
  speed_mps: float = 20.0
  plant_config: LateralPlantConfig = field(default_factory=LateralPlantConfig)
  thresholds: LateralMetricThresholds = field(default_factory=LateralMetricThresholds)

  def __post_init__(self):
    self.plant_config = self.plant_config or LateralPlantConfig()
    self.thresholds = self.thresholds or LateralMetricThresholds()


def generate_scenarios(fuzzer_config: FuzzerConfig) -> list[LateralScenario]:
  rng = random.Random(fuzzer_config.seed)
  kinds = [fuzzer_config.kind] if fuzzer_config.kind else list(SCENARIO_KINDS)
  generators = [SCENARIO_GENERATORS[k] for k in kinds]
  scenarios: list[LateralScenario] = []
  for idx in range(fuzzer_config.cases):
    gen = rng.choice(generators)
    scenario = gen(rng, idx, fuzzer_config.speed_mps, fuzzer_config.dt_s, fuzzer_config.duration)
    plant_config = replace(
      fuzzer_config.plant_config,
      dt_s=scenario.dt_s,
      speed_mps=scenario.speed_mps,
      duration_s=scenario.duration,
    )
    scenarios.append(
      LateralScenario(
        kind=scenario.kind,
        title=scenario.title,
        duration=scenario.duration,
        speed_mps=scenario.speed_mps,
        dt_s=scenario.dt_s,
        desired_curvature=scenario.desired_curvature,
        v_ego=scenario.v_ego,
        plant_config=plant_config,
        thresholds=fuzzer_config.thresholds,
      )
    )
  return scenarios


def run_scenario(scenario: LateralScenario, scenario_id: str | None = None) -> LateralScenarioResult:
  plant_config = scenario.plant
  desired = np.array(scenario.desired_curvature, dtype=float)
  v_ego = np.array(scenario.v_ego, dtype=float) if scenario.v_ego else None
  result = run_lateral_plant(desired_curvature=desired, v_ego=v_ego, config=plant_config)
  sid = scenario_id or f"lateral:{scenario.kind}"
  evaluation = evaluate_lateral_trace(sid, result.trace, result.config, scenario.metric_thresholds, scenario.kind)
  return LateralScenarioResult(scenario, result, evaluation)


def scenario_to_dict(scenario: LateralScenario, seed: int | None = None, index: int | None = None) -> dict[str, Any]:
  payload: dict[str, Any] = {
    "kind": scenario.kind,
    "title": scenario.title,
    "duration": scenario.duration,
    "speed_mps": scenario.speed_mps,
    "dt_s": scenario.dt_s,
    "desired_curvature": scenario.desired_curvature,
    "v_ego": scenario.v_ego,
  }
  if scenario.plant_config is not None:
    payload["plant_config"] = scenario.plant_config.to_dict()
  if scenario.thresholds is not None:
    payload["thresholds"] = scenario.thresholds.to_dict()
  if seed is not None:
    payload["seed"] = seed
  if index is not None:
    payload["index"] = index
  return payload


def scenario_from_dict(data: dict[str, Any]) -> LateralScenario:
  return LateralScenario(
    kind=str(data["kind"]),
    title=str(data["title"]),
    duration=float(data["duration"]),
    speed_mps=float(data["speed_mps"]),
    dt_s=float(data["dt_s"]),
    desired_curvature=tuple(float(v) for v in data["desired_curvature"]),
    v_ego=tuple(float(v) for v in data.get("v_ego", ())),
    plant_config=LateralPlantConfig.from_dict(data.get("plant_config", {})),
    thresholds=LateralMetricThresholds.from_dict(data.get("thresholds", {})),
  )


def artifact_to_dict(result: LateralScenarioResult, seed: int | None, index: int | None) -> dict[str, Any]:
  return {
    "schema": ARTIFACT_SCHEMA,
    "version": ARTIFACT_VERSION,
    "seed": seed,
    "index": index,
    "kind": result.scenario.kind,
    "scenario": scenario_to_dict(result.scenario, seed=seed, index=index),
    "plant_config": result.result.config.to_dict(),
    "thresholds": result.scenario.metric_thresholds.to_dict(),
    "failures": [failure.__dict__ for failure in result.failures],
    "metrics": {metric.name: metric.__dict__ for metric in result.evaluation.metrics},
    "valid": result.valid,
  }


def write_artifact(result: LateralScenarioResult, artifact_dir: Path, seed: int | None, index: int | None) -> Path:
  artifact_dir.mkdir(parents=True, exist_ok=True)
  filename = f"lateral_failure_{result.scenario.kind}_seed{seed}_idx{index}.json"
  path = artifact_dir / filename
  path.write_text(json.dumps(_json_safe(artifact_to_dict(result, seed, index)), allow_nan=False, indent=2, sort_keys=True))
  return path


def load_artifact(path: str | Path) -> dict[str, Any]:
  return json.loads(Path(path).read_text())


def _json_safe(value: Any) -> Any:
  if isinstance(value, float):
    if math.isnan(value):
      return "NaN"
    if math.isinf(value):
      return "Infinity" if value > 0.0 else "-Infinity"
    return value
  if isinstance(value, dict):
    return {key: _json_safe(item) for key, item in value.items()}
  if isinstance(value, (list, tuple)):
    return [_json_safe(item) for item in value]
  return value


def replay_artifact(path: str | Path) -> LateralScenarioResult:
  data = load_artifact(path)
  scenario = scenario_from_dict(data["scenario"])
  # The artifact stores the effective plant config and thresholds both inside
  # the scenario (when explicitly provided) and at the top level. Always prefer
  # the top-level values so replay matches the original run exactly.
  scenario = scenario.__class__(
    kind=scenario.kind,
    title=scenario.title,
    duration=scenario.duration,
    speed_mps=scenario.speed_mps,
    dt_s=scenario.dt_s,
    desired_curvature=scenario.desired_curvature,
    v_ego=scenario.v_ego,
    plant_config=LateralPlantConfig.from_dict(data.get("plant_config", {})),
    thresholds=LateralMetricThresholds.from_dict(data.get("thresholds", {})),
  )
  return run_scenario(scenario, scenario_id=f"replay:{scenario.kind}")


def _render_scenario_snippet(scenario: LateralScenario) -> str:
  lines = [
    f"# kind: {scenario.kind}",
    f"LateralScenario(",
    f"    title={scenario.title!r},",
    f"    duration={scenario.duration!r},",
    f"    speed_mps={scenario.speed_mps!r},",
    f"    dt_s={scenario.dt_s!r},",
    f"    desired_curvature={scenario.desired_curvature!r},",
    f")",
  ]
  return "\n".join(lines)


def main() -> None:
  parser = argparse.ArgumentParser(description="Seeded lateral structural stability fuzzer.")
  parser.add_argument("--seed", type=int, default=1)
  parser.add_argument("--cases", type=int, default=100)
  parser.add_argument("--kind", choices=SCENARIO_KINDS, help="Run only one scenario kind")
  parser.add_argument("--duration", type=float, default=10.0, help="Scenario duration in seconds")
  parser.add_argument("--dt", type=float, default=0.05, help="Simulation timestep in seconds")
  parser.add_argument("--speed", type=float, default=20.0, help="Ego speed in m/s")
  parser.add_argument("--json", action="store_true", help="Emit JSON instead of text")
  parser.add_argument("--fail-fast", action="store_true", help="Stop after the first failure")
  parser.add_argument("--artifact-dir", type=str, default=None, help="Directory to write failure artifacts")
  parser.add_argument("--replay", type=str, default=None, help="Replay a failure artifact JSON file")
  args = parser.parse_args()

  if args.cases < 0:
    parser.error("--cases must be >= 0")
  if args.duration <= 0.0:
    parser.error("--duration must be > 0")
  if args.dt <= 0.0:
    parser.error("--dt must be > 0")
  if args.speed < 0.0:
    parser.error("--speed must be >= 0")

  if args.replay:
    result = replay_artifact(args.replay)
    if args.json:
      print(json.dumps(_json_safe(artifact_to_dict(result, seed=None, index=None)), allow_nan=False, indent=2, sort_keys=True))
    else:
      print(f"Replayed {args.replay}: valid={result.valid} failures={len(result.failures)}")
      for failure in result.failures:
        print(f"  {failure.check}: {failure.detail}")
    sys.exit(0 if result.valid else 1)

  fuzzer_config = FuzzerConfig(
    seed=args.seed,
    cases=args.cases,
    kind=args.kind,
    duration=args.duration,
    dt_s=args.dt,
    speed_mps=args.speed,
  )
  scenarios = generate_scenarios(fuzzer_config)
  results: list[tuple[int, LateralScenarioResult]] = []
  for idx, scenario in enumerate(scenarios):
    result = run_scenario(scenario, scenario_id=f"lateral:{scenario.kind}:{args.seed}:{idx}")
    results.append((idx, result))
    if result.failures and args.fail_fast:
      break

  failures = [(idx, result) for idx, result in results if result.failures]
  artifact_paths: list[str] = []
  if args.artifact_dir and failures:
    artifact_dir = Path(args.artifact_dir)
    for idx, result in failures:
      path = write_artifact(result, artifact_dir, args.seed, idx)
      artifact_paths.append(str(path))

  if args.json:
    payload = {
      "seed": args.seed,
      "cases": len(results),
      "kind": args.kind,
      "duration": args.duration,
      "dt": args.dt,
      "speed": args.speed,
      "failures": [
        {
          "scenario": scenario_to_dict(result.scenario, seed=args.seed, index=idx),
          "checks": [failure.__dict__ for failure in result.failures],
        }
        for idx, result in failures
      ],
    }
    print(json.dumps(_json_safe(payload), allow_nan=False, indent=2, sort_keys=True))
  else:
    print(
      f"Drive Lab lateral fuzz seed={args.seed} cases={len(results)} "
      f"kind={args.kind or 'all'} duration={args.duration}s dt={args.dt}s speed={args.speed}m/s "
      f"failures={len(failures)}"
    )
    for idx, result in failures[:10]:
      print(f"\nFAILED: {result.scenario.title} [{result.scenario.kind}]")
      for failure in result.failures:
        print(f"  {failure.check}: {failure.detail}")
      print(_render_scenario_snippet(result.scenario))
    if artifact_paths:
      print(f"\nWrote {len(artifact_paths)} failure artifact(s) to {args.artifact_dir}")

  if failures:
    raise SystemExit(1)


if __name__ == "__main__":
  main()
