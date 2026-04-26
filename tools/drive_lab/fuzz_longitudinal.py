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

from openpilot.tools.drive_lab.log_profile import LongitudinalProfile, load_profile


@dataclass(frozen=True)
class Scenario:
  mode: str
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


REALISM_MODES = ("comfort", "emergency", "adversarial")
MODE_DEFAULT_JERK = {
  "comfort": 8.0,
  "emergency": 12.0,
  "adversarial": 100.0,
}


def generate_scenarios(seed: int, cases: int, mode: str = "comfort", profile: LongitudinalProfile | None = None) -> list[Scenario]:
  if mode not in REALISM_MODES:
    raise ValueError(f"unknown mode {mode!r}; expected one of {REALISM_MODES}")

  rng = random.Random(seed)
  generators = [
    _stopped_lead_approach,
    _slower_cut_in,
    _lead_occlusion,
    _lead_pullaway,
    _cruise_coast,
  ]
  return [rng.choice(generators)(rng, idx, mode, profile) for idx in range(cases)]


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
  return f"# mode: {scenario.mode}\nManeuver(\n    {scenario.title!r},\n    duration={scenario.duration!r},\n{kwargs}\n)"


def scenario_to_dict(scenario: Scenario) -> dict[str, Any]:
  return {
    "mode": scenario.mode,
    "kind": scenario.kind,
    "title": scenario.title,
    "duration": scenario.duration,
    "kwargs": scenario.kwargs,
  }


def main() -> None:
  parser = argparse.ArgumentParser(description="Seeded longitudinal maneuver fuzzer.")
  parser.add_argument("--seed", type=int, default=1)
  parser.add_argument("--cases", type=int, default=100)
  parser.add_argument("--mode", choices=REALISM_MODES, default="comfort", help="Scenario realism profile")
  parser.add_argument("--profile", help="Optional JSON profile from profile_route.py to bias generated ranges")
  parser.add_argument("--max-normal-jerk", type=float, help="Override the mode's jerk threshold")
  parser.add_argument("--max-failures", type=int, default=10)
  parser.add_argument("--list-only", action="store_true", help="Print generated scenarios without running the simulator")
  parser.add_argument("--json", action="store_true", help="Emit JSON instead of text")
  args = parser.parse_args()

  profile = load_profile(args.profile) if args.profile else None
  scenarios = generate_scenarios(args.seed, args.cases, args.mode, profile)
  if args.list_only:
    payload = [scenario_to_dict(s) for s in scenarios]
    print(json.dumps(payload, indent=2) if args.json else "\n\n".join(render_maneuver_snippet(s) for s in scenarios))
    return

  max_normal_jerk = args.max_normal_jerk if args.max_normal_jerk is not None else MODE_DEFAULT_JERK[args.mode]
  results = [run_scenario(s, max_normal_jerk) for s in scenarios]
  failures = [r for r in results if r.failures]
  if args.json:
    print(json.dumps({
      "seed": args.seed,
      "cases": args.cases,
      "mode": args.mode,
      "profile": profile.source if profile is not None else None,
      "maxNormalJerk": max_normal_jerk,
      "failures": [
        {
          "scenario": scenario_to_dict(result.scenario),
          "checks": [failure.__dict__ for failure in result.failures],
        }
        for result in failures
      ],
    }, indent=2))
  else:
    profile_text = f" profile={profile.source}" if profile is not None else ""
    print(f"Drive Lab fuzz seed={args.seed} mode={args.mode}{profile_text} cases={args.cases} max_normal_jerk={max_normal_jerk:g} failures={len(failures)}")
    for result in failures[:args.max_failures]:
      print(f"\nFAILED: {result.scenario.title} [{result.scenario.mode}/{result.scenario.kind}]")
      for failure in result.failures:
        print(f"  {failure.check}: {failure.detail}")
      print(render_maneuver_snippet(result.scenario))

  if failures:
    raise SystemExit(1)


def _stopped_lead_approach(rng: random.Random, idx: int, mode: str, profile: LongitudinalProfile | None) -> Scenario:
  if mode == "comfort":
    v_ego = _sample_profile_range(rng, profile, "ego_speed", (8.0, 24.0), (5.0, 26.0))
    lead_decel = _sample_profile_range(rng, profile, "lead_decel", (1.5, 3.5), (1.0, 3.8))
    lead_distance = rng.uniform(max(55.0, v_ego * 3.5), max(110.0, v_ego * 6.0))
  elif mode == "emergency":
    v_ego = _sample_profile_range(rng, profile, "ego_speed", (8.0, 28.0), (5.0, 32.0))
    lead_decel = _sample_profile_range(rng, profile, "lead_decel", (3.5, 7.0), (3.0, 8.0))
    lead_distance = rng.uniform(max(35.0, v_ego * 2.2), max(90.0, v_ego * 4.5))
  else:
    v_ego = rng.uniform(8.0, 28.0)
    lead_decel = v_ego / rng.uniform(0.2, 2.5)
    lead_distance = rng.uniform(max(35.0, v_ego * 2.5), max(75.0, v_ego * 5.5))

  lead_stop_time = v_ego / lead_decel
  return Scenario(
    mode,
    "stopped_lead_approach",
    f"fuzz stopped lead approach #{idx}",
    max(rng.uniform(12.0, 28.0), lead_stop_time + 8.0),
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


def _slower_cut_in(rng: random.Random, idx: int, mode: str, profile: LongitudinalProfile | None) -> Scenario:
  v_ego = _sample_profile_range(rng, profile, "ego_speed", (10.0, 25.0), (5.0, 30.0))
  cut_in_time = rng.uniform(1.0, 4.0)
  if mode == "comfort":
    closing_speed = _sample_profile_range(rng, profile, "closing_speed", (0.0, 4.0), (0.0, 4.5))
    v_lead = max(0.0, v_ego - closing_speed)
    closing_speed = max(0.0, v_ego - v_lead)
    detected_gap = rng.uniform(max(25.0, v_ego * 1.5, closing_speed * 4.0 + 10.0), max(65.0, v_ego * 3.0, closing_speed * 6.0 + 20.0))
    initial_distance_lead = detected_gap + closing_speed * cut_in_time
  elif mode == "emergency":
    closing_speed = _sample_profile_range(rng, profile, "closing_speed", (3.0, 9.0), (2.0, 11.0))
    v_lead = max(0.0, v_ego - closing_speed)
    closing_speed = max(0.0, v_ego - v_lead)
    detected_gap = rng.uniform(max(12.0, v_ego * 0.8, closing_speed * 1.5 + 6.0), max(45.0, v_ego * 1.8, closing_speed * 3.0 + 12.0))
    initial_distance_lead = detected_gap + closing_speed * cut_in_time
  else:
    v_lead = rng.uniform(max(0.0, v_ego - 8.0), v_ego + 1.0)
    initial_distance_lead = rng.uniform(18.0, 55.0)

  return Scenario(
    mode,
    "slower_cut_in",
    f"fuzz slower cut-in #{idx}",
    rng.uniform(8.0, 18.0),
    {
      "initial_speed": round(v_ego, 3),
      "lead_relevancy": True,
      "initial_distance_lead": round(initial_distance_lead, 3),
      "speed_lead_values": [round(v_lead, 3)] * 3,
      "prob_lead_values": [0.0, 0.0, 1.0],
      "cruise_values": [round(v_ego, 3)] * 3,
      "breakpoints": [0.0, round(cut_in_time, 3), round(cut_in_time + 0.01, 3)],
    },
  )


def _lead_occlusion(rng: random.Random, idx: int, mode: str, profile: LongitudinalProfile | None) -> Scenario:
  v_ego = _sample_profile_range(rng, profile, "ego_speed", (8.0, 22.0), (5.0, 28.0))
  occlusion_start = rng.uniform(2.0, 5.0)
  occlusion_end = occlusion_start + rng.uniform(0.2, 1.2)
  if mode == "comfort":
    initial_distance_lead = _sample_profile_range(
      rng, profile, "lead_gap", (max(35.0, v_ego * 2.0), max(80.0, v_ego * 4.0)), (max(30.0, v_ego * 1.5), max(90.0, v_ego * 5.0))
    )
    lead_delta = _sample_profile_range(rng, profile, "closing_speed", (0.0, 2.5), (0.0, 3.0))
  elif mode == "emergency":
    initial_distance_lead = _sample_profile_range(
      rng, profile, "lead_gap", (max(20.0, v_ego * 1.2), max(60.0, v_ego * 3.0)), (max(15.0, v_ego), max(70.0, v_ego * 4.0))
    )
    lead_delta = _sample_profile_range(rng, profile, "closing_speed", (1.0, 5.0), (0.5, 7.0))
  else:
    initial_distance_lead = rng.uniform(25.0, 70.0)
    lead_delta = rng.uniform(0.0, 4.0)

  return Scenario(
    mode,
    "lead_occlusion",
    f"fuzz lead occlusion #{idx}",
    rng.uniform(10.0, 20.0),
    {
      "initial_speed": round(v_ego, 3),
      "lead_relevancy": True,
      "initial_distance_lead": round(initial_distance_lead, 3),
      "speed_lead_values": [round(max(0.0, v_ego - lead_delta), 3)] * 4,
      "prob_lead_values": [1.0, 1.0, 0.0, 1.0],
      "cruise_values": [round(v_ego, 3)] * 4,
      "breakpoints": [0.0, round(occlusion_start, 3), round(occlusion_end, 3), round(occlusion_end + 0.01, 3)],
    },
  )


def _lead_pullaway(rng: random.Random, idx: int, mode: str, profile: LongitudinalProfile | None) -> Scenario:
  pullaway_time = rng.uniform(3.0, 8.0)
  if mode == "comfort":
    v_lead = _sample_profile_range(rng, profile, "lead_pullaway_speed", (1.0, 3.5), (0.5, 4.0))
    initial_distance_lead = _sample_profile_range(rng, profile, "stopped_lead_gap", (4.5, 8.0), (3.5, 10.0))
    cruise = _sample_profile_range(rng, profile, "cruise_speed", (5.0, 12.0), (3.0, 15.0))
  elif mode == "emergency":
    v_lead = _sample_profile_range(rng, profile, "lead_pullaway_speed", (2.5, 5.0), (1.0, 6.0))
    initial_distance_lead = _sample_profile_range(rng, profile, "stopped_lead_gap", (4.0, 8.0), (3.0, 10.0))
    cruise = _sample_profile_range(rng, profile, "cruise_speed", (5.0, 15.0), (3.0, 20.0))
  else:
    v_lead = rng.uniform(1.0, 5.0)
    initial_distance_lead = rng.uniform(4.0, 8.0)
    cruise = rng.uniform(5.0, 15.0)

  return Scenario(
    mode,
    "lead_pullaway",
    f"fuzz lead pullaway #{idx}",
    rng.uniform(10.0, 18.0),
    {
      "initial_speed": 0.0,
      "lead_relevancy": True,
      "initial_distance_lead": round(initial_distance_lead, 3),
      "speed_lead_values": [0.0, 0.0, round(v_lead, 3), round(v_lead, 3)],
      "prob_lead_values": [1.0, 1.0, 1.0, 1.0],
      "cruise_values": [round(cruise, 3)] * 4,
      "breakpoints": [0.0, round(pullaway_time, 3), round(pullaway_time + 1.0, 3), 18.0],
      "ensure_start": True,
    },
  )


def _cruise_coast(rng: random.Random, idx: int, mode: str, profile: LongitudinalProfile | None) -> Scenario:
  v_ego = _sample_profile_range(rng, profile, "ego_speed", (10.0, 25.0), (5.0, 30.0))
  cruise = min(v_ego, _sample_profile_range(rng, profile, "cruise_speed", (max(1.0, v_ego - 5.0), v_ego), (1.0, 32.0)))
  pitch = rng.uniform(-0.08, 0.08)
  if math.isclose(cruise, 0.0):
    cruise = 1.0
  return Scenario(
    mode,
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


def _sample_profile_range(
  rng: random.Random, profile: LongitudinalProfile | None, attr: str, fallback: tuple[float, float], clamp: tuple[float, float]
) -> float:
  low, high = fallback
  if profile is not None:
    profile_range = getattr(profile, attr)
    profiled_low = max(float(profile_range.low), clamp[0])
    profiled_high = min(float(profile_range.high), clamp[1])
    if profiled_low <= profiled_high:
      low, high = profiled_low, profiled_high

  low = max(low, clamp[0])
  high = min(high, clamp[1])
  if high < low:
    low, high = high, low
  if math.isclose(low, high):
    return low
  return rng.uniform(low, high)


if __name__ == "__main__":
  main()
