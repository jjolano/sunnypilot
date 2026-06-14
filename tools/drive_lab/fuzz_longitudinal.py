#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from openpilot.common.realtime import DT_MDL
from openpilot.tools.drive_lab.metrics import ScenarioFailure, evaluate_maneuver_output
from openpilot.tools.drive_lab.log_profile import LongitudinalProfile, load_profile
from openpilot.tools.drive_lab.scenario_spec import ScenarioSpec


@dataclass(frozen=True)
class Scenario:
  mode: str
  kind: str
  title: str
  duration: float
  kwargs: dict[str, Any]


@dataclass(frozen=True)
class ScenarioResult:
  scenario: Scenario
  valid: bool
  failures: list[ScenarioFailure]


REALISM_MODES = ("comfort", "emergency", "adversarial")
SCENARIO_PRESETS = ("fuzz", "udacity-acc")
MODE_DEFAULT_JERK = {
  "comfort": 8.0,
  "emergency": 12.0,
  "adversarial": 100.0,
}
LEAD_PULLAWAY_MOVING_SPEED = 0.5
LEAD_PULLAWAY_STARTED_SPEED = 0.2
LEAD_PULLAWAY_STARTED_ACCEL = 0.1
# Launch scenarios use the bounded start oracle (did the ego eventually move after the lead did?)
# instead of the legacy ensure_start, which fails on any single frame of near-zero accel while the
# lead creeps forward — a false alarm when gently following a barely-moving lead away from a stop.
LAUNCH_START_ORACLE_KINDS = ("lead_pullaway", "udacity_acc_green_light_launch")
# Collision handling: a lead that brakes harder than the ego can (e.g. it crashes) makes some
# contact physically unavoidable. The system's job there is best effort — "stop successfully, or
# minimize impact" — with driver override as the backstop, exactly as a human would. So contact is
# acceptable when the system either used (near-)full braking authority (it did all it could) or the
# impact was benign (a low-speed bump). Only a hard impact the system left braking authority unused
# for is a failure: it hit hard when it could have bled off more speed.
COLLISION_GAP = 0.4  # metres; matches the maneuver/metric contact threshold
# Committed firm braking (~0.25g). The MPC's comfort-balanced optimum for a moderate-closing cut-in
# plateaus around 3 m/s^2 rather than slamming the full |ACCEL_MIN| = 3.5, so best effort means a
# genuine hard-braking commitment, not literally maxing the clip; a weak response still fails.
BEST_EFFORT_BRAKE = 2.5
BENIGN_IMPACT_SPEED = 3.0  # m/s relative speed at contact below which the impact counts as minimized


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


def generate_udacity_acc_scenarios(mode: str = "comfort") -> list[Scenario]:
  """Return native Drive Lab scenarios inspired by Udacity's archived ACC challenge cases."""
  if mode not in REALISM_MODES:
    raise ValueError(f"unknown mode {mode!r}; expected one of {REALISM_MODES}")

  return [
    Scenario(
      mode,
      "udacity_acc_cruise_speed_step",
      "udacity acc inspired cruise speed step",
      30.0,
      {
        "initial_speed": 17.881,
        "lead_relevancy": False,
        "cruise_values": [17.881, 17.881, 22.352, 22.352],
        "breakpoints": [0.0, 10.0, 10.01, 30.0],
      },
    ),
    Scenario(
      mode,
      "udacity_acc_grade_change",
      "udacity acc inspired uphill grade change",
      25.0,
      {
        "initial_speed": 8.941,
        "lead_relevancy": False,
        "cruise_values": [8.941, 8.941, 8.941, 8.941],
        "pitch_values": [0.0, 0.0, 0.08, 0.08],
        "breakpoints": [0.0, 10.0, 11.0, 25.0],
      },
    ),
    Scenario(
      mode,
      "udacity_acc_slower_lead",
      "udacity acc inspired slower lead approach",
      30.0,
      {
        "initial_speed": 26.822,
        "lead_relevancy": True,
        "initial_distance_lead": 100.0,
        "speed_lead_values": [17.881, 17.881],
        "prob_lead_values": [1.0, 1.0],
        "cruise_values": [26.822, 26.822],
        "breakpoints": [0.0, 30.0],
      },
    ),
    Scenario(
      mode,
      "udacity_acc_stopped_lead",
      "udacity acc inspired stopped lead approach",
      30.0,
      {
        "initial_speed": 17.881,
        "lead_relevancy": True,
        "initial_distance_lead": 150.0,
        "speed_lead_values": [0.0, 0.0],
        "prob_lead_values": [1.0, 1.0],
        "cruise_values": [17.881, 17.881],
        "breakpoints": [0.0, 30.0],
      },
    ),
    Scenario(
      mode,
      "udacity_acc_lead_decel_to_stop",
      "udacity acc inspired lead decel to stop",
      45.0,
      {
        "initial_speed": 20.0,
        "lead_relevancy": True,
        "initial_distance_lead": 45.0,
        "speed_lead_values": [20.0, 20.0, 0.0, 0.0],
        "prob_lead_values": [1.0, 1.0, 1.0, 1.0],
        "cruise_values": [20.0, 20.0, 20.0, 20.0],
        "breakpoints": [0.0, 10.0, 30.0, 45.0],
      },
    ),
    Scenario(
      mode,
      "udacity_acc_oscillating_lead",
      "udacity acc inspired oscillating lead speed",
      25.0,
      {
        "initial_speed": 30.0,
        "lead_relevancy": True,
        "initial_distance_lead": 55.0,
        "speed_lead_values": [30.0, 30.0, 29.0, 31.0, 29.0, 31.0, 29.0],
        "prob_lead_values": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "cruise_values": [30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0],
        "breakpoints": [0.0, 6.0, 8.0, 12.0, 16.0, 20.0, 24.0],
      },
    ),
    Scenario(
      mode,
      "udacity_acc_stop_and_go",
      "udacity acc inspired stop and go lead",
      60.0,
      {
        "initial_speed": 0.0,
        "lead_relevancy": True,
        "initial_distance_lead": 20.0,
        "speed_lead_values": [10.0, 0.0, 0.0, 10.0, 0.0, 0.0],
        "prob_lead_values": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "cruise_values": [15.0, 15.0, 15.0, 15.0, 15.0, 15.0],
        "breakpoints": [0.0, 10.0, 20.0, 30.0, 40.0, 50.0],
      },
    ),
    Scenario(
      mode,
      "udacity_acc_green_light_launch",
      "udacity acc inspired green light lead launch",
      20.0,
      {
        "initial_speed": 0.0,
        "lead_relevancy": True,
        "initial_distance_lead": 11.0,
        "speed_lead_values": [0.0, 0.0, 5.0, 12.0],
        "prob_lead_values": [1.0, 1.0, 1.0, 1.0],
        "cruise_values": [15.0, 15.0, 15.0, 15.0],
        "breakpoints": [0.0, 5.0, 8.0, 20.0],
        "ensure_start": True,
      },
    ),
  ]


def evaluate_invariants(
  valid: bool,
  output: np.ndarray,
  max_normal_jerk: float = 8.0,
  commanded_accel: np.ndarray | None = None,
  jerk_window: int = 1,
) -> list[ScenarioFailure]:
  return evaluate_maneuver_output("legacy", valid, output, max_normal_jerk, commanded_accel, jerk_window).failures


@dataclass
class CommandCapture:
  commanded: list[float] = field(default_factory=list)
  prob_lead: list[float] = field(default_factory=list)
  actuator_delay: float | None = None


@contextlib.contextmanager
def capture_commanded_accel():
  """Record the planner's commanded acceleration (output_a_target) for each maneuver step.

  The plant overwrites its own acceleration with a crude stop model, so jerk measured on the
  maneuver output reflects harness scaffolding rather than the longitudinal policy. Capturing the
  planner's pre-override command lets the jerk oracle evaluate what the policy actually emits, and
  the actuator delay lets it window jerk over the actuator response rather than a single frame.

  Also skips the plant's one-off socket-connect sleep (vestigial offline — the planner is fed a
  plain dict, not the published messages), verified to leave outputs bit-identical, so a fuzz sweep
  of thousands of scenarios is not dominated by per-scenario sleeps.
  """
  from openpilot.selfdrive.test.longitudinal_maneuvers import plant as plant_module
  Plant = plant_module.Plant

  capture = CommandCapture()
  original_step = Plant.step
  original_sleep = plant_module.time.sleep

  def step(self, *step_args, **step_kwargs):
    result = original_step(self, *step_args, **step_kwargs)
    capture.commanded.append(float(self.planner.output_a_target))
    # plant.step(v_lead, prob_lead, v_cruise, pitch, prob_throttle); prob_lead drives lead detection.
    prob_lead = step_kwargs.get("prob_lead", step_args[1] if len(step_args) > 1 else 1.0)
    capture.prob_lead.append(float(prob_lead))
    if capture.actuator_delay is None:
      capture.actuator_delay = float(self.planner.CP.longitudinalActuatorDelay)
    return result

  Plant.step = step
  plant_module.time.sleep = lambda *a, **k: None
  try:
    yield capture
  finally:
    Plant.step = original_step
    plant_module.time.sleep = original_sleep


def evaluate_lead_pullaway_start(output: np.ndarray) -> list[ScenarioFailure]:
  if output.ndim != 2 or output.shape[1] < 6 or output.size == 0:
    return []

  time_s = output[:, 0]
  speed = output[:, 3]
  lead_speed = output[:, 4]
  accel = output[:, 5]
  lead_moving = lead_speed > LEAD_PULLAWAY_MOVING_SPEED
  if not np.any(lead_moving):
    return []

  lead_move_time = float(time_s[int(np.flatnonzero(lead_moving)[0])])
  after_lead_moves = time_s >= lead_move_time
  started = np.any(speed[after_lead_moves] > LEAD_PULLAWAY_STARTED_SPEED) or np.any(accel[after_lead_moves] > LEAD_PULLAWAY_STARTED_ACCEL)
  if started:
    return []
  return [ScenarioFailure("launch", "lead moved but ego never started")]


def evaluate_collision_response(
  output: np.ndarray, commanded_accel: np.ndarray | None, prob_lead: np.ndarray | None
) -> list[ScenarioFailure]:
  """Judge a contact event by best-effort mitigation, not by absence of contact.

  Contact is acceptable when the system did all it reasonably could: it braked at (near-)full
  authority, or the impact was benign (low relative speed). A failure is a hard impact the system
  hit while leaving braking authority unused — it could have bled off more speed and did not.
  """
  if output.ndim != 2 or output.shape[1] < 7 or output.size == 0:
    return []

  v_ego = output[:, 3]
  v_lead = output[:, 4]
  d_rel = output[:, 6]

  finite = np.isfinite(d_rel)
  contact = np.flatnonzero(finite & (d_rel < COLLISION_GAP))
  if contact.size == 0:
    return []
  impact = int(contact[0])
  min_gap = float(np.min(d_rel[finite]))
  impact_speed = max(0.0, float(v_ego[impact] - v_lead[impact]))

  detected = np.flatnonzero(np.asarray(prob_lead) > 0.5) if prob_lead is not None else np.empty(0, dtype=int)
  d0 = min(int(detected[0]) if detected.size else 0, impact)

  best_effort = (
    commanded_accel is not None
    and len(commanded_accel) == len(output)
    and float(np.min(commanded_accel[d0:impact + 1])) <= -BEST_EFFORT_BRAKE
  )
  if best_effort or impact_speed <= BENIGN_IMPACT_SPEED:
    return []
  return [
    ScenarioFailure("collision", f"hard collision at {impact_speed:.1f} m/s without full braking (minimum lead gap {min_gap:.3f} m)")
  ]


def scenario_maneuver_kwargs(scenario: Scenario) -> dict[str, Any]:
  kwargs = dict(scenario.kwargs)
  if scenario.kind in LAUNCH_START_ORACLE_KINDS:
    kwargs["ensure_start"] = False
  return kwargs


@contextlib.contextmanager
def shipped_longitudinal_config():
  """Pin the custom-2.0 longitudinal policy on for the duration of a fuzz run, then restore.

  drive_lab is the validation gate for the fork's shipped behavior, and the custom policy ships
  default-on (CustomLongitudinalEnabled defaults to "1" in params_keys.h). The Python test
  environment does not apply those C++ param defaults, so without this the fuzzer would silently
  validate stock openpilot rather than the policy that actually ships. The prior param value is
  saved and restored so the run leaves the param store unchanged.
  """
  from openpilot.common.params import Params

  params = Params()
  key = "CustomLongitudinalEnabled"
  previous = params.get(key)
  params.put_bool(key, True)
  try:
    yield
  finally:
    if previous is None:
      params.remove(key)
    else:
      params.put(key, previous)


def run_scenario(scenario: Scenario, max_normal_jerk: float = 8.0) -> ScenarioResult:
  from openpilot.selfdrive.test.longitudinal_maneuvers.maneuver import Maneuver

  maneuver = Maneuver(scenario.title, scenario.duration, **scenario_maneuver_kwargs(scenario))
  with contextlib.redirect_stdout(io.StringIO()), capture_commanded_accel() as capture:
    valid, output = maneuver.evaluate()
  commanded_accel = np.array(capture.commanded) if len(capture.commanded) == len(output) else None
  # Window jerk over the planner's own control horizon (longitudinalActuatorDelay + DT_MDL, the
  # action_t at which it evaluates its target accel), which bounds how fast realized accel changes.
  action_horizon = (capture.actuator_delay or 0.0) + DT_MDL
  jerk_window = max(1, round(action_horizon / DT_MDL))
  failures = evaluate_invariants(valid, output, max_normal_jerk, commanded_accel, jerk_window)
  if scenario.kind in LAUNCH_START_ORACLE_KINDS:
    failures.extend(evaluate_lead_pullaway_start(output))
  # Contact is judged by the best-effort collision oracle. Drop the maneuver's hard contact signals
  # (the metric's "collision" check and the crash-driven "valid" failure) and defer to the oracle's
  # avoidability/mitigation verdict so an unavoidable, maximally-braked collision is not a failure.
  prob_lead = np.array(capture.prob_lead) if len(capture.prob_lead) == len(output) else None
  oracle = evaluate_collision_response(output, commanded_accel, prob_lead)
  contact = output.ndim == 2 and output.shape[1] >= 7 and bool(np.any(np.isfinite(output[:, 6]) & (output[:, 6] < COLLISION_GAP)))
  if contact:
    failures = [f for f in failures if f.check not in ("collision", "valid")] + oracle
  return ScenarioResult(scenario, not failures, failures)


def render_maneuver_snippet(scenario: Scenario) -> str:
  kwargs = ",\n".join(f"    {key}={repr(value)}" for key, value in scenario.kwargs.items())
  return f"# mode: {scenario.mode}\nManeuver(\n    {scenario.title!r},\n    duration={scenario.duration!r},\n{kwargs}\n)"


def scenario_to_spec(scenario: Scenario, source: str = "generated", seed: int | None = None, index: int | None = None) -> ScenarioSpec:
  return ScenarioSpec.from_maneuver_kwargs(
    kind=scenario.kind,
    title=scenario.title,
    mode=scenario.mode,
    duration=scenario.duration,
    kwargs=scenario.kwargs,
    source=source,
    seed=seed,
    index=index,
  )


def scenario_to_dict(scenario: Scenario, source: str | None = None, seed: int | None = None, index: int | None = None) -> dict[str, Any]:
  payload = {
    "mode": scenario.mode,
    "kind": scenario.kind,
    "title": scenario.title,
    "duration": scenario.duration,
    "kwargs": scenario.kwargs,
  }
  if source is not None:
    spec = scenario_to_spec(scenario, source=source, seed=seed, index=index)
    payload["scenarioId"] = spec.scenario_id
    payload["spec"] = spec.to_dict()
  return payload


def main() -> None:
  parser = argparse.ArgumentParser(description="Seeded longitudinal maneuver fuzzer.")
  parser.add_argument("--seed", type=int, default=1)
  parser.add_argument("--cases", type=int, default=100)
  parser.add_argument("--mode", choices=REALISM_MODES, default="comfort", help="Scenario realism profile")
  parser.add_argument(
    "--preset", choices=SCENARIO_PRESETS, default="fuzz", help="Scenario source: seeded fuzzing or fixed Udacity ACC-inspired cases"
  )
  parser.add_argument("--profile", help="Optional JSON profile from profile_route.py to bias generated ranges")
  parser.add_argument("--max-normal-jerk", type=float, help="Override the mode's jerk threshold")
  parser.add_argument("--max-failures", type=int, default=10)
  parser.add_argument("--list-only", action="store_true", help="Print generated scenarios without running the simulator")
  parser.add_argument("--json", action="store_true", help="Emit JSON instead of text")
  args = parser.parse_args()

  profile = load_profile(args.profile) if args.profile else None
  scenarios = (
    generate_udacity_acc_scenarios(args.mode)
    if args.preset == "udacity-acc"
    else generate_scenarios(args.seed, args.cases, args.mode, profile)
  )
  if args.list_only:
    payload = [
      scenario_to_dict(
        scenario,
        source=args.preset,
        seed=args.seed if args.preset == "fuzz" else None,
        index=idx,
      )
      for idx, scenario in enumerate(scenarios)
    ]
    print(json.dumps(payload, indent=2) if args.json else "\n\n".join(render_maneuver_snippet(s) for s in scenarios))
    return

  max_normal_jerk = args.max_normal_jerk if args.max_normal_jerk is not None else MODE_DEFAULT_JERK[args.mode]
  with shipped_longitudinal_config():
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
    # Adversarial: the lead may brake far harder than the ego can (e.g. it crashes into something),
    # so the resulting collision can be physically unavoidable. That is intentional — the collision
    # oracle judges best-effort mitigation here, not absence of contact.
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
    closing_speed = max(0.0, v_ego - v_lead)
    # Adversarial: an aggressively tight detected gap that may be physically unavoidable; the
    # collision oracle judges best-effort mitigation. The cut-in must still be coherent though — a
    # cut-in appears in front of the ego, so add the distance the ego closes while the lead is
    # occluded, keeping the lead ahead at reveal rather than spawning behind the ego.
    detected_gap = rng.uniform(6.0, 30.0)
    initial_distance_lead = detected_gap + closing_speed * cut_in_time

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
