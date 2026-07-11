#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import io

import numpy as np

from openpilot.common.realtime import DT_MDL

from openpilot.tools.drive_lab.fuzz_longitudinal import (
  capture_commanded_accel,
  scenario_maneuver_kwargs,
  shipped_longitudinal_config,
)
from openpilot.tools.drive_lab.log_profile import load_profile
from openpilot.tools.drive_lab.longitudinal_scenarios import (
  REALISM_MODES,
  SCENARIO_PRESETS,
  PresetRequest,
  generate_preset_scenarios,
)
from openpilot.tools.drive_lab.manual_longitudinal_baseline import (
  ScenarioComparison,
  compare_scenario_output,
  render_behavior_outline,
  render_comparison_table,
)


def main() -> None:
  parser = argparse.ArgumentParser(description="Compare synthetic longitudinal scenarios with manual-style baseline envelopes.")
  parser.add_argument("--seed", type=int, default=1)
  parser.add_argument("--cases", type=int, default=25)
  parser.add_argument("--mode", choices=REALISM_MODES, default="comfort")
  parser.add_argument("--preset", choices=SCENARIO_PRESETS, default="fuzz")
  parser.add_argument("--profile", help="Optional JSON profile from profile_route.py or openacc_segments.py")
  parser.add_argument("--e2e", action="store_true")
  parser.add_argument("--force-decel", action="store_true")
  parser.add_argument("--ncap-family", choices=("CCRs", "CCRm", "CCRb"))
  parser.add_argument("--ncap-sample", type=int)
  parser.add_argument("--strict", action="store_true", help="Exit non-zero when any baseline comparison fails")
  args = parser.parse_args()

  profile = load_profile(args.profile) if args.profile else None
  scenarios = generate_preset_scenarios(PresetRequest(
    preset=args.preset,
    mode=args.mode,
    seed=args.seed,
    cases=args.cases,
    profile=profile,
    e2e=args.e2e,
    force_decel=args.force_decel,
    ncap_family=args.ncap_family,
    ncap_sample=args.ncap_sample,
  ))
  with shipped_longitudinal_config():
    results = [evaluate_scenario(scenario) for scenario in scenarios]
  print(render_report(results, args.seed, args.mode, args.preset))
  if args.strict and any(not result.passed for result in results):
    raise SystemExit(1)


def evaluate_scenario(scenario) -> ScenarioComparison:
  from openpilot.selfdrive.test.longitudinal_maneuvers.maneuver import Maneuver

  maneuver = Maneuver(scenario.title, scenario.duration, **scenario_maneuver_kwargs(scenario))
  with contextlib.redirect_stdout(io.StringIO()), capture_commanded_accel() as capture:
    valid, output = maneuver.evaluate()
  commanded_accel = np.asarray(capture.commanded) if len(capture.commanded) == len(output) else None
  action_horizon = (capture.actuator_delay or 0.0) + DT_MDL
  jerk_window = max(1, round(action_horizon / DT_MDL))
  comparisons = compare_scenario_output(
    scenario.kind, output, commanded_accel=commanded_accel, jerk_window=jerk_window,
  )
  if scenario.oracle_profile == "safety" and not comparisons:
    return ScenarioComparison(scenario.title, scenario.kind, bool(valid), comparisons)
  return ScenarioComparison(scenario.title, scenario.kind, bool(valid), comparisons)


def render_report(results: list[ScenarioComparison], seed: int, mode: str, preset: str) -> str:
  failed = sum(1 for result in results if not result.passed)
  lines = [
    "Drive Lab manual longitudinal baseline",
    f"seed={seed} mode={mode} preset={preset} scenarios={len(results)} failures={failed}",
    "",
    "Behavior outline",
    render_behavior_outline(),
    "",
    "Current vs expected",
  ]
  for result in results:
    status = "pass" if result.passed else "fail"
    valid = "valid" if result.valid else "invalid"
    lines.extend([
      "",
      f"Scenario: {result.title} [{result.kind}] {status} ({valid})",
      render_comparison_table(result.comparisons),
    ])
  return "\n".join(lines)


if __name__ == "__main__":
  main()
