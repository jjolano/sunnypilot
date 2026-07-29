#!/usr/bin/env python3
"""Route-extracted longitudinal replay perturbation fuzzer.

Extracts engaged longitudinal context from route logs, optionally perturbs lead
signals, and replays through the openpilot Maneuver plant with drive_lab oracles.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass, field
from typing import Any

from openpilot.tools.drive_lab.fuzz_longitudinal import (
  MODE_DEFAULT_JERK,
  REALISM_MODES,
  Scenario,
  run_scenario,
  scenario_to_dict,
  shipped_longitudinal_config,
)
from openpilot.tools.drive_lab.metrics import ScenarioFailure
from openpilot.tools.drive_lab.longitudinal_route_extract import (
  DT,
  ROUTE_EXTRACTED_PRESET,
  LongitudinalRouteFrame,
  frames_to_maneuver_kwargs,
  load_route_frames,
  max_d_rel_error,
)
from openpilot.tools.drive_lab.timeline import select_event_time

ARTIFACT_SCHEMA = "drive-lab-longitudinal-route-replay-fuzzer-artifact"
ARTIFACT_VERSION = 1
PERTURBATION_KINDS = ("none", "dropout", "delay", "stale", "noise", "gap_scale")
CLI_PERTURBATION_KINDS = PERTURBATION_KINDS


@dataclass(frozen=True)
class PerturbationRecipe:
  kind: str
  start_frame: int
  end_frame: int
  description: str
  params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RouteReplayScenario:
  preset: str
  title: str
  frames: tuple[LongitudinalRouteFrame, ...]
  recipe: PerturbationRecipe
  perturbed_frames: tuple[LongitudinalRouteFrame, ...]
  route_metadata: dict[str, Any] | None = None

  @property
  def duration(self) -> float:
    if not self.perturbed_frames:
      return 0.0
    return self.perturbed_frames[-1].t + DT


# A replay that has wandered this far from the recorded d_rel is no longer
# reproducing the route, so its oracle verdict is not evidence about it.
MAX_D_REL_DRIFT_M = 15.0


@dataclass(frozen=True)
class RouteReplayResult:
  scenario: RouteReplayScenario
  valid: bool
  failures: list[Any]
  max_d_rel_error: float | None = None


@dataclass
class RouteReplayFuzzerConfig:
  seed: int = 1
  cases: int = 10
  mode: str = "comfort"
  perturbation: str | None = None
  route: str | None = None
  qlog: bool = False
  window_start_s: float | None = None
  window_end_s: float | None = None
  max_frames: int | None = None
  engaged_only: bool = True
  nearest_bookmark: bool = False
  event_time: float | None = None
  before_s: float = 30.0
  after_s: float = 30.0


def _generate_recipe(rng: random.Random, n: int, kind: str | None) -> PerturbationRecipe:
  kind = kind or rng.choice([k for k in PERTURBATION_KINDS if k != "none"])
  if kind == "none":
    return PerturbationRecipe(kind="none", start_frame=0, end_frame=0, description="no perturbation")
  window = max(5, int(n * rng.uniform(0.15, 0.35)))
  start = rng.randint(0, max(0, n - window - 1))
  end = min(n, start + window)
  if kind == "noise":
    params = {"noise_std": rng.uniform(0.2, 1.0)}
    desc = f"v_lead noise frames {start}-{end}"
  elif kind == "dropout":
    params = {}
    desc = f"prob_lead dropout frames {start}-{end}"
  elif kind == "delay":
    params = {"delay_frames": rng.randint(1, min(10, max(1, start)))}
    desc = f"lead delay {params['delay_frames']} frames {start}-{end}"
  elif kind == "stale":
    params = {}
    desc = f"stale lead freeze frames {start}-{end}"
  elif kind == "gap_scale":
    params = {"scale": rng.uniform(0.7, 1.3)}
    desc = f"gap scale {params['scale']:.2f} frames {start}-{end}"
  else:
    raise ValueError(f"unknown perturbation {kind!r}")
  return PerturbationRecipe(kind=kind, start_frame=start, end_frame=end, description=desc, params=params)


def _apply_recipe(recipe: PerturbationRecipe, frames: tuple[LongitudinalRouteFrame, ...]) -> tuple[LongitudinalRouteFrame, ...]:
  if recipe.kind == "none" or not frames:
    return frames
  out = list(frames)
  start, end = recipe.start_frame, recipe.end_frame
  if recipe.kind == "dropout":
    for i in range(start, end):
      f = out[i]
      out[i] = LongitudinalRouteFrame(**{**f.to_dict(), "prob_lead": 0.0})
  elif recipe.kind == "delay":
    delay = int(recipe.params.get("delay_frames", 1))
    for i in range(start, end):
      src = max(0, i - delay)
      f = out[i]
      src_f = out[src]
      out[i] = LongitudinalRouteFrame(**{
        **f.to_dict(),
        "v_lead": src_f.v_lead,
        "prob_lead": src_f.prob_lead,
        "d_rel": src_f.d_rel,
      })
  elif recipe.kind == "stale":
    hold = out[max(0, start - 1)]
    for i in range(start, end):
      f = out[i]
      out[i] = LongitudinalRouteFrame(**{
        **f.to_dict(),
        "v_lead": hold.v_lead,
        "prob_lead": hold.prob_lead,
      })
  elif recipe.kind == "noise":
    rng = random.Random(recipe.params.get("noise_seed", 0))
    std = float(recipe.params.get("noise_std", 0.5))
    for i in range(start, end):
      f = out[i]
      if f.v_lead is None:
        continue
      out[i] = LongitudinalRouteFrame(**{
        **f.to_dict(),
        "v_lead": max(0.0, float(f.v_lead) + rng.gauss(0.0, std)),
      })
  elif recipe.kind == "gap_scale":
    scale = float(recipe.params.get("scale", 1.0))
    for i in range(start, end):
      f = out[i]
      if f.d_rel is None or f.v_lead is None:
        continue
      out[i] = LongitudinalRouteFrame(**{
        **f.to_dict(),
        "v_lead": max(0.0, float(f.v_ego) + (float(f.v_lead) - float(f.v_ego)) * scale),
      })
  return tuple(out)


def generate_route_replay_scenarios(
  frames: tuple[LongitudinalRouteFrame, ...],
  config: RouteReplayFuzzerConfig,
  *,
  route_metadata: dict[str, Any] | None = None,
) -> list[RouteReplayScenario]:
  if not frames:
    return []
  rng = random.Random(config.seed)
  scenarios: list[RouteReplayScenario] = []
  for idx in range(config.cases):
    recipe = _generate_recipe(rng, len(frames), config.perturbation)
    perturbed = _apply_recipe(recipe, frames)
    title = f"route replay {ROUTE_EXTRACTED_PRESET} with {recipe.kind} #{idx}"
    scenarios.append(RouteReplayScenario(
      preset=ROUTE_EXTRACTED_PRESET,
      title=title,
      frames=frames,
      recipe=recipe,
      perturbed_frames=perturbed,
      route_metadata=route_metadata,
    ))
  return scenarios


def route_replay_to_scenario(replay: RouteReplayScenario, mode: str) -> Scenario:
  kwargs = frames_to_maneuver_kwargs(replay.perturbed_frames)
  kind = f"route_replay_{replay.recipe.kind}"
  return Scenario(
    mode,
    kind,
    replay.title,
    replay.duration,
    kwargs,
    oracle_profile="comfort",
    provenance={"preset": replay.preset, "recipe": replay.recipe.description},
  )


def run_route_replay_scenario(replay: RouteReplayScenario, mode: str, max_normal_jerk: float,
                              max_d_rel_drift: float = MAX_D_REL_DRIFT_M) -> RouteReplayResult:
  scenario = route_replay_to_scenario(replay, mode)
  result = run_scenario(scenario, max_normal_jerk)
  from openpilot.selfdrive.test.longitudinal_maneuvers.maneuver import Maneuver
  from openpilot.tools.drive_lab.fuzz_longitudinal import scenario_maneuver_kwargs

  maneuver = Maneuver(scenario.title, scenario.duration, **scenario_maneuver_kwargs(scenario))
  _, output = maneuver.evaluate()
  drift = max_d_rel_error(replay.perturbed_frames, output)
  # Drift was reported but never gated, so a replay that had wandered far from the
  # recorded geometry still scored its oracle checks as if it were the real route.
  # Past this bound the replay is no longer the route it claims to replay.
  failures = list(result.failures)
  valid = result.valid
  if drift is not None and drift > max_d_rel_drift:
    failures.append(ScenarioFailure(
      check="route_replay_drift",
      detail=f"max |d_rel - recorded| {drift:.1f} m exceeds {max_d_rel_drift:.1f} m; replay left the recorded geometry",
    ))
    valid = False
  return RouteReplayResult(replay, valid, failures, drift)


def main() -> None:
  parser = argparse.ArgumentParser(description="Longitudinal route replay fuzzer.")
  parser.add_argument("--route", required=True, help="Route identifier or log file")
  parser.add_argument("--seed", type=int, default=1)
  parser.add_argument("--cases", type=int, default=10)
  parser.add_argument("--mode", choices=REALISM_MODES, default="comfort")
  parser.add_argument("--perturbation", choices=CLI_PERTURBATION_KINDS, help="Perturbation kind")
  parser.add_argument("--qlog", action="store_true")
  parser.add_argument("--start-s", type=float, dest="window_start_s")
  parser.add_argument("--end-s", type=float, dest="window_end_s")
  parser.add_argument("--max-frames", type=int)
  parser.add_argument("--time", type=float, dest="event_time", help="Event time for bookmark window")
  parser.add_argument("--nearest-bookmark", action="store_true")
  parser.add_argument("--before", type=float, default=30.0)
  parser.add_argument("--after", type=float, default=30.0)
  parser.add_argument("--list-only", action="store_true")
  parser.add_argument("--json", action="store_true")
  parser.add_argument("--max-failures", type=int, default=10)
  args = parser.parse_args()

  start_s = args.window_start_s
  end_s = args.window_end_s
  if args.nearest_bookmark or args.event_time is not None:
    from openpilot.tools.drive_lab.route_io import load_route_msgs
    msgs = load_route_msgs(args.route, qlog=args.qlog)
    event_t = select_event_time(msgs, args.event_time, args.nearest_bookmark)
    start_s = event_t - args.before
    end_s = event_t + args.after

  frames, summary = load_route_frames(
    args.route,
    qlog=args.qlog,
    start_s=start_s,
    end_s=end_s,
    max_frames=args.max_frames,
    engaged_only=True,
  )
  if args.list_only:
    payload = {
      "extracted_count": summary.extracted_count,
      "summary": summary.to_dict(),
      "frames": [f.to_dict() for f in frames[:5]],
    }
    print(json.dumps(payload, indent=2) if args.json else f"extracted_frames={summary.extracted_count}")
    return

  if not frames:
    parser.error(f"no route frames extracted from {args.route}")

  config = RouteReplayFuzzerConfig(
    seed=args.seed,
    cases=args.cases,
    mode=args.mode,
    perturbation=args.perturbation,
    route=args.route,
    qlog=args.qlog,
    window_start_s=start_s,
    window_end_s=end_s,
    max_frames=args.max_frames,
  )
  scenarios = generate_route_replay_scenarios(frames, config, route_metadata=summary.to_dict())
  max_jerk = MODE_DEFAULT_JERK[args.mode]
  with shipped_longitudinal_config():
    results = [run_route_replay_scenario(s, args.mode, max_jerk) for s in scenarios]
  failures = [r for r in results if r.failures]

  if args.json:
    print(json.dumps({
      "route": args.route,
      "failures": [
        {
          "title": r.scenario.title,
          "checks": [f.__dict__ for f in r.failures],
          "maxDRelError": r.max_d_rel_error,
        }
        for r in failures
      ],
    }, indent=2))
  else:
    print(f"Drive Lab longitudinal route replay route={args.route} cases={len(scenarios)} failures={len(failures)}")
    for result in failures[:args.max_failures]:
      print(f"\nFAILED: {result.scenario.title}")
      for failure in result.failures:
        print(f"  {failure.check}: {failure.detail}")
      if result.max_d_rel_error is not None:
        print(f"  max_d_rel_error: {result.max_d_rel_error:.3f} m")

  if failures:
    raise SystemExit(1)


if __name__ == "__main__":
  main()
