#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import random
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Scenario:
  kind: str
  title: str
  duration: float
  kwargs: dict[str, Any]


@dataclass(frozen=True)
class ScenarioFailure:
  check: str
  detail: str


@dataclass(frozen=True)
class ScenarioResult:
  scenario: Scenario
  valid: bool
  failures: list[ScenarioFailure]


def generate_scenarios(seed: int, cases: int) -> list[Scenario]:
  rng = random.Random(seed)
  generators = [
    _stopped_lead_approach,
    _slower_cut_in,
    _lead_occlusion,
    _lead_pullaway,
    _cruise_coast,
  ]
  return [rng.choice(generators)(rng, idx) for idx in range(cases)]


def evaluate_invariants(valid: bool, output: np.ndarray, max_normal_jerk: float = 8.0) -> list[ScenarioFailure]:
  failures = []
  if not valid:
    failures.append(ScenarioFailure("valid", "maneuver reported invalid"))
  if output.size == 0:
    return [*failures, ScenarioFailure("output", "maneuver produced no output")]
  if not np.all(np.isfinite(output)):
    failures.append(ScenarioFailure("finite", "output contains NaN or infinite values"))
    return failures

  time_s = output[:, 0]
  speed = output[:, 3]
  accel = output[:, 5]
  d_rel = output[:, 6]

  if np.min(speed) < -1e-3:
    failures.append(ScenarioFailure("speed", f"negative speed {np.min(speed):.3f} m/s"))
  if np.min(d_rel) < 0.4:
    failures.append(ScenarioFailure("collision", f"minimum lead gap {np.min(d_rel):.3f} m"))
  if len(accel) > 2:
    dt = np.diff(time_s)
    valid_dt = dt > 1e-6
    if np.any(valid_dt):
      jerk = np.diff(accel)[valid_dt] / dt[valid_dt]
      max_abs_jerk = float(np.max(np.abs(jerk)))
      if max_abs_jerk > max_normal_jerk:
        failures.append(ScenarioFailure("jerk", f"maximum absolute jerk {max_abs_jerk:.3f} m/s^3"))
  return failures


def run_scenario(scenario: Scenario, max_normal_jerk: float = 8.0) -> ScenarioResult:
  from openpilot.selfdrive.test.longitudinal_maneuvers.maneuver import Maneuver

  maneuver = Maneuver(scenario.title, scenario.duration, **scenario.kwargs)
  with contextlib.redirect_stdout(io.StringIO()):
    valid, output = maneuver.evaluate()
  failures = evaluate_invariants(valid, output, max_normal_jerk)
  return ScenarioResult(scenario, valid and not failures, failures)


def render_maneuver_snippet(scenario: Scenario) -> str:
  kwargs = ",\n".join(f"    {key}={repr(value)}" for key, value in scenario.kwargs.items())
  return f"Maneuver(\n    {scenario.title!r},\n    duration={scenario.duration!r},\n{kwargs}\n)"


def scenario_to_dict(scenario: Scenario) -> dict[str, Any]:
  return {
    "kind": scenario.kind,
    "title": scenario.title,
    "duration": scenario.duration,
    "kwargs": scenario.kwargs,
  }


def main() -> None:
  parser = argparse.ArgumentParser(description="Seeded longitudinal maneuver fuzzer.")
  parser.add_argument("--seed", type=int, default=1)
  parser.add_argument("--cases", type=int, default=100)
  parser.add_argument("--max-normal-jerk", type=float, default=8.0)
  parser.add_argument("--max-failures", type=int, default=10)
  parser.add_argument("--list-only", action="store_true", help="Print generated scenarios without running the simulator")
  parser.add_argument("--json", action="store_true", help="Emit JSON instead of text")
  args = parser.parse_args()

  scenarios = generate_scenarios(args.seed, args.cases)
  if args.list_only:
    payload = [scenario_to_dict(s) for s in scenarios]
    print(json.dumps(payload, indent=2) if args.json else "\n\n".join(render_maneuver_snippet(s) for s in scenarios))
    return

  results = [run_scenario(s, args.max_normal_jerk) for s in scenarios]
  failures = [r for r in results if r.failures]
  if args.json:
    print(json.dumps({
      "seed": args.seed,
      "cases": args.cases,
      "failures": [
        {
          "scenario": scenario_to_dict(result.scenario),
          "checks": [failure.__dict__ for failure in result.failures],
        }
        for result in failures
      ],
    }, indent=2))
  else:
    print(f"Drive Lab fuzz seed={args.seed} cases={args.cases} failures={len(failures)}")
    for result in failures[:args.max_failures]:
      print(f"\nFAILED: {result.scenario.title}")
      for failure in result.failures:
        print(f"  {failure.check}: {failure.detail}")
      print(render_maneuver_snippet(result.scenario))

  if failures:
    raise SystemExit(1)


def _stopped_lead_approach(rng: random.Random, idx: int) -> Scenario:
  v_ego = rng.uniform(8.0, 28.0)
  lead_distance = rng.uniform(max(35.0, v_ego * 2.5), max(75.0, v_ego * 5.5))
  lead_stop_time = rng.uniform(0.2, 2.5)
  return Scenario(
    "stopped_lead_approach",
    f"fuzz stopped lead approach #{idx}",
    rng.uniform(12.0, 28.0),
    {
      "initial_speed": round(v_ego, 3),
      "lead_relevancy": True,
      "initial_distance_lead": round(lead_distance, 3),
      "speed_lead_values": [round(v_ego, 3), 0.0, 0.0],
      "prob_lead_values": [1.0, 1.0, 1.0],
      "cruise_values": [round(max(v_ego, 12.0), 3)] * 3,
      "breakpoints": [0.0, round(lead_stop_time, 3), round(lead_stop_time + 0.01, 3)],
    },
  )


def _slower_cut_in(rng: random.Random, idx: int) -> Scenario:
  v_ego = rng.uniform(10.0, 25.0)
  v_lead = rng.uniform(max(0.0, v_ego - 8.0), v_ego + 1.0)
  cut_in_time = rng.uniform(1.0, 4.0)
  return Scenario(
    "slower_cut_in",
    f"fuzz slower cut-in #{idx}",
    rng.uniform(8.0, 18.0),
    {
      "initial_speed": round(v_ego, 3),
      "lead_relevancy": True,
      "initial_distance_lead": round(rng.uniform(18.0, 55.0), 3),
      "speed_lead_values": [round(v_lead, 3)] * 3,
      "prob_lead_values": [0.0, 0.0, 1.0],
      "cruise_values": [round(v_ego, 3)] * 3,
      "breakpoints": [0.0, round(cut_in_time, 3), round(cut_in_time + 0.01, 3)],
    },
  )


def _lead_occlusion(rng: random.Random, idx: int) -> Scenario:
  v_ego = rng.uniform(8.0, 22.0)
  occlusion_start = rng.uniform(2.0, 5.0)
  occlusion_end = occlusion_start + rng.uniform(0.2, 1.2)
  return Scenario(
    "lead_occlusion",
    f"fuzz lead occlusion #{idx}",
    rng.uniform(10.0, 20.0),
    {
      "initial_speed": round(v_ego, 3),
      "lead_relevancy": True,
      "initial_distance_lead": round(rng.uniform(25.0, 70.0), 3),
      "speed_lead_values": [round(v_ego - rng.uniform(0.0, 4.0), 3)] * 4,
      "prob_lead_values": [1.0, 1.0, 0.0, 1.0],
      "cruise_values": [round(v_ego, 3)] * 4,
      "breakpoints": [0.0, round(occlusion_start, 3), round(occlusion_end, 3), round(occlusion_end + 0.01, 3)],
    },
  )


def _lead_pullaway(rng: random.Random, idx: int) -> Scenario:
  pullaway_time = rng.uniform(3.0, 8.0)
  v_lead = rng.uniform(1.0, 5.0)
  return Scenario(
    "lead_pullaway",
    f"fuzz lead pullaway #{idx}",
    rng.uniform(10.0, 18.0),
    {
      "initial_speed": 0.0,
      "lead_relevancy": True,
      "initial_distance_lead": round(rng.uniform(4.0, 8.0), 3),
      "speed_lead_values": [0.0, 0.0, round(v_lead, 3), round(v_lead, 3)],
      "prob_lead_values": [1.0, 1.0, 1.0, 1.0],
      "cruise_values": [round(rng.uniform(5.0, 15.0), 3)] * 4,
      "breakpoints": [0.0, round(pullaway_time, 3), round(pullaway_time + 1.0, 3), 18.0],
      "ensure_start": True,
    },
  )


def _cruise_coast(rng: random.Random, idx: int) -> Scenario:
  v_ego = rng.uniform(10.0, 25.0)
  cruise = max(0.0, v_ego - rng.uniform(0.0, 5.0))
  pitch = rng.uniform(-0.08, 0.08)
  if math.isclose(cruise, 0.0):
    cruise = 1.0
  return Scenario(
    "cruise_coast",
    f"fuzz cruise coast #{idx}",
    rng.uniform(8.0, 18.0),
    {
      "initial_speed": round(v_ego, 3),
      "lead_relevancy": False,
      "cruise_values": [round(cruise, 3), round(cruise, 3)],
      "pitch_values": [round(pitch, 3), round(pitch, 3)],
      "breakpoints": [0.0, 18.0],
    },
  )


if __name__ == "__main__":
  main()
