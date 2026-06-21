#!/usr/bin/env python3
"""Synthetic demand-to-plant closed-loop lateral fuzzer.

This is a composition fuzzer: it feeds the per-frame processed curvature output
of the lateral demand pipeline (Phase B) into the synthetic structural stability
plant (Phase A). It is *not* a high-fidelity representation of the production
lateral control loop; it checks only that the demand pipeline's output, when
used as a desired-curvature sequence, does not drive the simple test plant into
obvious structural instability.

Layer separation is explicit:
  - demand failures come from evaluate_scenario (Phase B)
  - plant failures come from evaluate_lateral_trace (Phase A)
  - overall valid = demand valid AND plant valid
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from openpilot.tools.drive_lab.fuzz_lateral_demand import (
  DemandFuzzerConfig,
  DemandScenario,
  DemandScenarioResult,
  SCENARIO_GENERATORS,
  evaluate_scenario,
  generate_scenarios,
  scenario_from_dict,
  scenario_to_dict,
)
from openpilot.tools.drive_lab.lateral_scenarios import (
  LATERAL_PRESETS,
  LateralPresetRequest,
  generate_preset_scenarios,
)
from openpilot.tools.drive_lab.lateral_metrics import LateralMetricThresholds, evaluate_lateral_trace
from openpilot.tools.drive_lab.lateral_plant import LateralPlantConfig, LateralPlantResult, run_lateral_plant


ARTIFACT_SCHEMA = "drive-lab-lateral-closed-loop-fuzzer-artifact"
ARTIFACT_VERSION = 1
DT = 0.01
DEFAULT_KINDS = (
  "high_quality_path",
  "invalid_path_recovery",
  "curvature_jump",
  "low_lane_confidence",
  "path_disagreement",
)
ALL_KINDS = DEFAULT_KINDS + ("lateral_maneuver_override",)
EXPECTED_PLANT_TRANSIENT_FAILURES_BY_KIND = {
  # Lateral maneuver override is an explicit source-authority transition: the demand layer must
  # pass the override curvature through exactly, so the simple plant's steady tracking checks are
  # expected to lag the step. Keep the structural plant checks hard.
  "lateral_maneuver_override": {"tracking", "settle"},
  # ISO 3888-1 double lane change is an aggressive open-loop maneuver: the simple test plant is
  # expected to lag the reference path during the transient, so tracking error is informational.
  # Structural plant checks (jerk, saturation, oscillation, finite outputs) remain hard failures.
  "iso_3888_lane_change": {"tracking"},
}


@dataclass(frozen=True)
class ClosedLoopThresholds:
  """Thresholds for the closed-loop composition.

  Demand thresholds are embedded in the DemandScenario. Plant thresholds are
  loosened slightly versus the standalone Phase A defaults because the demand
  pipeline runs at 100 Hz, which can produce sharper curvature transients.
  """

  plant: LateralMetricThresholds = LateralMetricThresholds(
    max_abs_tracking_error=0.008,
    max_abs_steering_rate=200.0,
    max_abs_lateral_jerk=25.0,
    max_saturation_fraction=0.25,
    max_zero_drift_curvature=2e-4,
    max_oscillation_reversals=25,
    max_final_tracking_error=0.005,
  )

  def to_dict(self) -> dict[str, Any]:
    return {"plant": self.plant.to_dict()}

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> ClosedLoopThresholds:
    return cls(plant=LateralMetricThresholds.from_dict(data.get("plant", {})))


@dataclass(frozen=True)
class ClosedLoopResult:
  """Result of running one demand scenario through demand+plant layers."""

  scenario: DemandScenario
  demand_result: DemandScenarioResult
  plant_result: LateralPlantResult | None
  plant_evaluation: Any | None
  demand_failures: list[dict[str, Any]]
  plant_failures: list[Any]
  plant_skipped: bool

  @property
  def valid(self) -> bool:
    return self.demand_result.valid and not self.plant_skipped and not self.plant_failures


# ---------- helpers ----------


def _sanitize(value: Any) -> Any:
  """Recursively sanitize nonfinite floats for strict JSON output."""
  if isinstance(value, np.generic):
    return _sanitize(value.item())
  if isinstance(value, float):
    return value if math.isfinite(value) else None
  if isinstance(value, dict):
    return {k: _sanitize(v) for k, v in value.items()}
  if isinstance(value, (list, tuple)):
    return [_sanitize(v) for v in value]
  return value


def _run_closed_loop(scenario: DemandScenario, thresholds: ClosedLoopThresholds | None = None) -> ClosedLoopResult:
  thresholds = thresholds or ClosedLoopThresholds()
  demand_result = evaluate_scenario(scenario)
  demand_failures = list(demand_result.failures)

  # Skip plant if demand layer failed or produced unusable outputs.
  if not demand_result.valid:
    return ClosedLoopResult(
      scenario=scenario,
      demand_result=demand_result,
      plant_result=None,
      plant_evaluation=None,
      demand_failures=demand_failures,
      plant_failures=[{
        "check": "plant_skipped_due_to_demand_failure",
        "detail": "demand layer reported failure(s); plant scoring skipped",
      }],
      plant_skipped=True,
    )

  if len(demand_result.outputs) != len(scenario.frames):
    return ClosedLoopResult(
      scenario=scenario,
      demand_result=demand_result,
      plant_result=None,
      plant_evaluation=None,
      demand_failures=demand_failures,
      plant_failures=[{
        "check": "plant_skipped_due_to_demand_failure",
        "detail": f"demand output length {len(demand_result.outputs)} != frame count {len(scenario.frames)}",
      }],
      plant_skipped=True,
    )

  processed = np.array([out.processed_curvature for out in demand_result.outputs], dtype=float)
  v_ego = np.array([frame["v_ego"] for frame in scenario.frames], dtype=float)
  if not np.all(np.isfinite(processed)) or not np.all(np.isfinite(v_ego)):
    return ClosedLoopResult(
      scenario=scenario,
      demand_result=demand_result,
      plant_result=None,
      plant_evaluation=None,
      demand_failures=demand_failures,
      plant_failures=[{
        "check": "plant_skipped_due_to_demand_failure",
        "detail": "nonfinite processed_curvature or v_ego from demand layer",
      }],
      plant_skipped=True,
    )

  n = len(processed)
  if n < 2:
    return ClosedLoopResult(
      scenario=scenario,
      demand_result=demand_result,
      plant_result=None,
      plant_evaluation=None,
      demand_failures=demand_failures,
      plant_failures=[{
        "check": "plant_skipped_due_to_demand_failure",
        "detail": f"need at least 2 demand output frames for plant scoring, got {n}",
      }],
      plant_skipped=True,
    )

  plant_config = LateralPlantConfig(
    dt_s=DT,
    duration_s=max((n - 1) * DT, DT),
    speed_mps=float(np.mean(v_ego)),
  )

  plant_result = run_lateral_plant(desired_curvature=processed, v_ego=v_ego, config=plant_config)
  plant_evaluation = evaluate_lateral_trace(
    scenario_id=f"closed_loop:{scenario.kind}",
    trace=plant_result.trace,
    config=plant_result.config,
    thresholds=thresholds.plant,
    scenario_kind=scenario.kind,
  )
  plant_failures = _unexpected_plant_failures(scenario.kind, plant_evaluation.failures)

  return ClosedLoopResult(
    scenario=scenario,
    demand_result=demand_result,
    plant_result=plant_result,
    plant_evaluation=plant_evaluation,
    demand_failures=demand_failures,
    plant_failures=plant_failures,
    plant_skipped=False,
  )


def _unexpected_plant_failures(scenario_kind: str, failures: Any) -> list[Any]:
  expected = EXPECTED_PLANT_TRANSIENT_FAILURES_BY_KIND.get(scenario_kind, set())
  return [failure for failure in failures if _failure_check(failure) not in expected]


def _failure_check(failure: Any) -> str:
  return str(failure.check if hasattr(failure, "check") else failure.get("check", ""))


# ---------- scenario generation ----------


def generate_closed_loop_scenarios(seed: int, cases: int, kind: str | None = None, duration_s: float = 2.0) -> list[DemandScenario]:
  kinds = [kind] if kind else list(DEFAULT_KINDS)
  config = DemandFuzzerConfig(seed=seed, cases=cases, kind=kinds[0] if len(kinds) == 1 else None, duration_s=duration_s)
  if kind is None:
    rng = random.Random(seed)
    generators = [SCENARIO_GENERATORS[k] for k in kinds]
    return [generators[rng.randrange(len(generators))](rng, idx, duration_s) for idx in range(cases)]
  config = DemandFuzzerConfig(seed=seed, cases=cases, kind=kind, duration_s=duration_s)
  return generate_scenarios(config)


# ---------- serialization / CLI ----------


def _demand_summary(result: DemandScenarioResult) -> dict[str, Any]:
  return {
    "valid": result.valid,
    "failure_checks": [f["check"] for f in result.failures],
    "metrics": _sanitize(result.metrics),
  }


def _plant_summary(result: ClosedLoopResult) -> dict[str, Any]:
  if result.plant_skipped:
    return {"skipped": True, "skipped_reason": result.plant_failures[0]["detail"] if result.plant_failures else "demand layer failed"}
  if result.plant_evaluation is None:
    return {"skipped": True, "skipped_reason": "unknown"}
  return {
    "skipped": False,
    "valid": not result.plant_failures,
    "failure_checks": [f.check for f in result.plant_failures],
    "metrics": _sanitize({m.name: m.__dict__ for m in result.plant_evaluation.metrics}),
  }


def artifact_to_dict(result: ClosedLoopResult, thresholds: ClosedLoopThresholds, seed: int | None, index: int | None) -> dict[str, Any]:
  return {
    "schema": ARTIFACT_SCHEMA,
    "version": ARTIFACT_VERSION,
    "seed": seed,
    "index": index,
    "kind": result.scenario.kind,
    "scenario": scenario_to_dict(result.scenario, seed=seed, index=index),
    "thresholds": thresholds.to_dict(),
    "plant_config": result.plant_result.config.to_dict() if result.plant_result else None,
    "plant_thresholds": thresholds.plant.to_dict(),
    "demand_summary": _demand_summary(result.demand_result),
    "plant_summary": _plant_summary(result),
    "overall_valid": result.valid,
  }


def write_artifact(result: ClosedLoopResult, thresholds: ClosedLoopThresholds, artifact_dir: Path, seed: int | None, index: int | None) -> Path:
  artifact_dir.mkdir(parents=True, exist_ok=True)
  filename = f"lateral_closed_loop_failure_{result.scenario.kind}_seed{seed}_idx{index}.json"
  path = artifact_dir / filename
  path.write_text(json.dumps(_sanitize(artifact_to_dict(result, thresholds, seed, index)), indent=2, sort_keys=True, allow_nan=False))
  return path


def load_artifact(path: str | Path) -> dict[str, Any]:
  return json.loads(Path(path).read_text())


def replay_artifact(path: str | Path) -> ClosedLoopResult:
  data = load_artifact(path)
  scenario = scenario_from_dict(data["scenario"])
  thresholds = ClosedLoopThresholds.from_dict(data.get("thresholds", {"plant": data.get("plant_thresholds", {})}))
  return _run_closed_loop(scenario, thresholds)


def scenario_summary_to_dict(scenario: DemandScenario, seed: int | None = None, index: int | None = None) -> dict[str, Any]:
  payload: dict[str, Any] = {
    "kind": scenario.kind,
    "title": scenario.title,
    "duration_s": scenario.duration_s,
    "frames": len(scenario.frames),
  }
  if seed is not None:
    payload["seed"] = seed
  if index is not None:
    payload["index"] = index
  return payload


def _render_scenario_snippet(scenario: DemandScenario) -> str:
  return (
    f"# kind: {scenario.kind}\n"
    f"DemandScenario(title={scenario.title!r}, duration_s={scenario.duration_s!r}, frames=[...{len(scenario.frames)} frames...])"
  )


def main() -> None:
  parser = argparse.ArgumentParser(description="Synthetic demand-to-plant closed-loop lateral fuzzer.")
  parser.add_argument("--seed", type=int, default=1)
  parser.add_argument("--cases", type=int, default=100)
  parser.add_argument("--kind", choices=ALL_KINDS, help="Run only one scenario kind")
  parser.add_argument("--preset", choices=LATERAL_PRESETS, help="Public lateral benchmark preset")
  parser.add_argument("--nhtsa-family", choices=("primary", "secondary"), help="NHTSA LKA test family filter")
  parser.add_argument("--nhtsa-line-type", help="NHTSA LKA line type filter")
  parser.add_argument("--nhtsa-drift-rate", type=float, help="NHTSA LKA drift rate filter (m/s)")
  parser.add_argument("--euroncap-family", choices=("lka", "elk", "sbend", "alc"), help="Euro NCAP LSS family filter")
  parser.add_argument("--nuplan-focus", choices=("error", "jerk", "oscillation"), help="nuPlan lateral focus filter")
  parser.add_argument("--stress-grid-sample", type=int, default=None, help="Number of random stress-grid cells (None=full grid)")
  parser.add_argument("--duration", type=float, default=2.0, help="Scenario duration in seconds")
  parser.add_argument("--json", action="store_true", help="Emit JSON instead of text")
  parser.add_argument("--fail-fast", action="store_true", help="Stop after the first failure")
  parser.add_argument("--artifact-dir", type=str, default=None, help="Directory to write failure artifacts")
  parser.add_argument("--replay", type=str, default=None, help="Replay a closed-loop artifact JSON file")
  args = parser.parse_args()

  if args.cases < 0:
    parser.error("--cases must be >= 0")
  if args.duration <= 0.0:
    parser.error("--duration must be > 0")
  if args.preset and args.kind:
    parser.error("--preset and --kind are mutually exclusive")

  thresholds = ClosedLoopThresholds()

  if args.replay:
    result = replay_artifact(args.replay)
    if args.json:
      print(json.dumps(_sanitize(artifact_to_dict(result, thresholds, seed=None, index=None)), indent=2, sort_keys=True, allow_nan=False))
    else:
      print(f"Replayed {args.replay}: valid={result.valid} demand_failures={len(result.demand_failures)} plant_failures={len(result.plant_failures)} plant_skipped={result.plant_skipped}")
      for failure in result.demand_failures:
        print(f"  demand: {failure['check']}: {failure['detail']}")
      for failure in result.plant_failures:
        print(f"  plant: {failure.check if hasattr(failure, 'check') else failure['check']}: {failure.detail if hasattr(failure, 'detail') else failure['detail']}")
    sys.exit(0 if result.valid else 1)

  if args.preset:
    request = LateralPresetRequest(
      preset=args.preset,
      seed=args.seed,
      cases=args.cases,
      duration_s=args.duration,
      nhtsa_family=args.nhtsa_family,
      nhtsa_line_type=args.nhtsa_line_type,
      nhtsa_drift_rate=args.nhtsa_drift_rate,
      euroncap_family=args.euroncap_family,
      nuplan_focus=args.nuplan_focus,
      stress_grid_sample=args.stress_grid_sample,
    )
    scenarios = generate_preset_scenarios(request)
  else:
    scenarios = generate_closed_loop_scenarios(args.seed, args.cases, args.kind, args.duration)
  results: list[tuple[int, ClosedLoopResult]] = []
  for idx, scenario in enumerate(scenarios):
    result = _run_closed_loop(scenario, thresholds)
    results.append((idx, result))
    if not result.valid and args.fail_fast:
      break

  failures = [(idx, result) for idx, result in results if not result.valid]
  artifact_paths: list[str] = []
  if args.artifact_dir and failures:
    artifact_dir = Path(args.artifact_dir)
    for idx, result in failures:
      path = write_artifact(result, thresholds, artifact_dir, args.seed, idx)
      artifact_paths.append(str(path))

  if args.json:
    payload = {
      "seed": args.seed,
      "cases": len(results),
      "kind": args.kind,
      "preset": args.preset,
      "nhtsa_family": args.nhtsa_family,
      "nhtsa_line_type": args.nhtsa_line_type,
      "nhtsa_drift_rate": args.nhtsa_drift_rate,
      "euroncap_family": args.euroncap_family,
      "nuplan_focus": args.nuplan_focus,
      "duration": args.duration,
      "dt": DT,
      "failures": [
        {
          "scenario": scenario_summary_to_dict(result.scenario, seed=args.seed, index=result_idx),
          "artifact_hint": "rerun with --artifact-dir for full per-frame replay input",
          "demand_checks": [f["check"] for f in result.demand_failures],
          "plant_checks": [f.check if hasattr(f, "check") else f["check"] for f in result.plant_failures],
          "plant_skipped": result.plant_skipped,
        }
        for result_idx, result in failures
      ],
    }
    print(json.dumps(_sanitize(payload), indent=2, sort_keys=True, allow_nan=False))
  else:
    print(
      f"Drive Lab lateral closed-loop fuzz seed={args.seed} cases={len(results)} "
      f"kind={args.kind or ('n/a' if args.preset else 'default')} "
      f"preset={args.preset or 'none'} duration={args.duration}s dt={DT}s failures={len(failures)}"
    )
    for idx, result in failures[:10]:
      print(f"\nFAILED: {result.scenario.title} [{result.scenario.kind}]")
      for failure in result.demand_failures:
        print(f"  demand: {failure['check']}: {failure['detail']}")
      for failure in result.plant_failures:
        print(f"  plant: {failure.check if hasattr(failure, 'check') else failure['check']}: {failure.detail if hasattr(failure, 'detail') else failure['detail']}")
      if result.plant_skipped and not result.plant_failures:
        print("  plant: skipped due to demand failure")
      print(_render_scenario_snippet(result.scenario))
    if artifact_paths:
      print(f"\nWrote {len(artifact_paths)} failure artifact(s) to {args.artifact_dir}")

  if failures:
    raise SystemExit(1)


if __name__ == "__main__":
  main()
