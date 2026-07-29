#!/usr/bin/env python3
"""Offline corner-demand jerk comparison. Never changes live-control code."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

# Permit analysis from a clean worktree while the shared checkout is mid-migration.
REPO_ROOT = Path(os.environ.get("SUNNYPILOT_REPO_ROOT", Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "openpilot"))

from openpilot.sunnypilot.custom.lateral.demand.model_path_processor import (
  DEMAND_JERK_SMOOTH_MAX_FRAME_DROP_PERC,
  DEMAND_JERK_SMOOTH_MAX_LAT_JERK,
  DEMAND_JERK_SMOOTH_MAX_SPEED,
  DEMAND_JERK_SMOOTH_MIN_SPEED,
  DEMAND_JERK_SMOOTH_SPEED_BP,
  ModelPathProcessor,
  ModelPathProcessorInputs,
)
from openpilot.sunnypilot.custom.lateral.demand.pipeline import LANE_CHANGE_STATE_OFF, LateralDemandPipeline
from openpilot.tools.drive_lab.fuzz_lateral_route_replay import _frame_from_controls_state, _frame_to_inputs
from openpilot.tools.drive_lab.route_analysis import build_route_messages
from openpilot.tools.lib.logreader import LogReader


DT = 0.01
ALLOWED_REASONS = frozenset(("ok", "low_lane_confidence", "high_path_std"))
MIN_QUALITY = 0.55
MAX_PATH_Y_STD = 1.8
MAX_PATH_DISAGREEMENT = 0.75
MAX_CORNER_LAT_ACCEL = 2.5
MIN_METRIC_LAT_ACCEL = 0.3
EXCURSION_LAT_JERK = 3.0
SIGN_MISMATCH_LAT_ACCEL = 0.05
VARIANTS = {
  "pre_lag08": ("pre_gate", 0.08),
  "pre_lag20": ("pre_gate", 0.20),
  "post_lag08": ("post_gate", 0.08),
  "post_lag20": ("post_gate", 0.20),
}


class CornerSafeProcessor(ModelPathProcessor):
  """Spike-only eligibility expansion around the existing slew/clamp operation."""

  def __init__(self, placement: str, lag_lat_accel: float) -> None:
    self.placement = placement
    self.lag_lat_accel = float(lag_lat_accel)
    self._post_smoothed_curvature: float | None = None
    super().__init__()

  def reset(self) -> None:
    super().reset()
    self._post_smoothed_curvature = None

  def update(self, inputs: ModelPathProcessorInputs):
    if self.placement == "pre_gate":
      return super().update(inputs)

    # The current pre-gate position never runs on quality < 0.75. Disable it there and
    # reuse the same bounded operation once, after the processor has chosen its fallback.
    result = super().update(replace(inputs, demand_jerk_smoothing_enabled=False))
    target = float(result.desired_curvature)
    if not self._demand_jerk_smoothing_eligible(
      inputs, target, target, result.quality, result.reason, None,
    ):
      self._post_smoothed_curvature = None
      return result
    candidate, active, step, lag = self._shape_target(
      self._post_smoothed_curvature, target, float(inputs.v_ego),
    )
    self._post_smoothed_curvature = candidate
    return replace(
      result,
      desired_curvature=candidate,
      demand_jerk_smoothing_active=active,
      demand_jerk_smoothing_step=step,
      demand_jerk_smoothing_lag=lag,
    )

  def _demand_jerk_smoothing_eligible(
    self,
    inputs: ModelPathProcessorInputs,
    raw_base: float,
    target: float,
    quality: float,
    reason: str,
    path_disagreement: float | None,
  ) -> bool:
    if not self._corner_gates_ok(inputs, raw_base, target, quality, reason, path_disagreement):
      return False
    v_sq = float(inputs.v_ego) ** 2
    return max(abs(raw_base), abs(target)) * v_sq <= MAX_CORNER_LAT_ACCEL

  def _corner_gates_ok(
    self,
    inputs: ModelPathProcessorInputs,
    raw_base: float,
    target: float,
    quality: float,
    reason: str,
    path_disagreement: float | None,
  ) -> bool:
    if not inputs.demand_jerk_smoothing_enabled or not inputs.demand_jerk_smoothing_allowed:
      return False
    if not inputs.smooth_model_path_curvature or inputs.lane_change_active:
      return False
    if reason not in ALLOWED_REASONS or quality < MIN_QUALITY:
      return False
    v_ego = float(inputs.v_ego)
    if not math.isfinite(v_ego) or not DEMAND_JERK_SMOOTH_MIN_SPEED <= v_ego <= DEMAND_JERK_SMOOTH_MAX_SPEED:
      return False
    if not math.isfinite(raw_base) or not math.isfinite(target):
      return False
    if math.isfinite(inputs.frame_drop_perc) and inputs.frame_drop_perc > DEMAND_JERK_SMOOTH_MAX_FRAME_DROP_PERC:
      return False
    if path_disagreement is not None and path_disagreement > MAX_PATH_DISAGREEMENT:
      return False
    core_std = tuple(float(v) for v in inputs.position_y_std[:17])
    return len(core_std) == 17 and all(math.isfinite(v) for v in core_std) and max(core_std) <= MAX_PATH_Y_STD

  def _apply_demand_jerk_smoothing(
    self,
    inputs: ModelPathProcessorInputs,
    raw_base: float,
    target: float,
    quality: float,
    reason: str,
    path_disagreement: float | None,
  ) -> tuple[float, bool, float, float]:
    if not self._demand_jerk_smoothing_eligible(inputs, raw_base, target, quality, reason, path_disagreement):
      self._reset_demand_jerk_smoothing()
      return float(target), False, 0.0, 0.0

    candidate, active, max_step, lag = self._shape_target(
      self._demand_jerk_smoothed_curvature, float(target), float(inputs.v_ego),
    )
    self._demand_jerk_smoothed_curvature = candidate
    self._demand_jerk_smoothing_active = active
    self._last_demand_jerk_smoothing_step = max_step
    self._last_demand_jerk_smoothing_lag = lag
    return candidate, active, max_step, lag

  def _shape_target(
    self,
    previous: float | None,
    target: float,
    v_ego: float,
  ) -> tuple[float, bool, float, float]:
    v_ego = max(v_ego, 1.0)
    v_sq = v_ego * v_ego
    max_lat_jerk = float(np.interp(v_ego, DEMAND_JERK_SMOOTH_SPEED_BP, DEMAND_JERK_SMOOTH_MAX_LAT_JERK))
    max_step = max_lat_jerk * DT / v_sq
    lag_limit = self.lag_lat_accel / v_sq

    if previous is None or not math.isfinite(previous):
      return target, False, max_step, 0.0
    previous = float(previous)

    # A material direction reversal is not a comfort event. Follow it immediately.
    if previous * target < 0.0 and min(abs(previous), abs(target)) * v_sq > SIGN_MISMATCH_LAT_ACCEL:
      return target, False, max_step, 0.0

    delta = target - previous
    candidate = target if abs(delta) <= max_step else previous + math.copysign(max_step, delta)
    candidate = self._clamp_demand_jerk_candidate(
      previous, candidate, target, target, lag_limit, raw_cap_active=False,
    )
    lag = abs(candidate - target)
    return candidate, lag > 1e-9, max_step, lag


@dataclass(frozen=True)
class ReplayFrame:
  frame: Any
  steering_torque: float
  model_frame_id: int


@dataclass(frozen=True)
class Event:
  route: str
  segment: int
  t: float
  phase: str
  reason: str
  baseline_jerk: float
  candidate_jerk: float
  lag: float

  def to_dict(self) -> dict[str, Any]:
    return {
      "route": self.route,
      "segment": self.segment,
      "t": round(self.t, 3),
      "phase": self.phase,
      "reason": self.reason,
      "baseline_jerk": round(self.baseline_jerk, 3),
      "candidate_jerk": round(self.candidate_jerk, 3),
      "lag": round(self.lag, 3),
    }


def _segment_number(path: Path) -> int:
  try:
    return int(path.parent.name)
  except ValueError:
    return int(path.parent.name.rsplit("--", 1)[-1])


def _segment_paths(route_dir: Path) -> list[Path]:
  paths = list(route_dir.glob("*/rlog.zst"))
  return sorted(paths, key=_segment_number)


def _frames(path: Path) -> list[ReplayFrame]:
  latest: dict[str, Any] = {}
  out: list[ReplayFrame] = []
  for message in build_route_messages(LogReader(str(path))):
    if message.typ in ("carState", "carControl", "modelV2", "liveParameters", "controlsState"):
      latest[message.typ] = message.payload
    if message.typ != "controlsState" or latest.get("carState") is None:
      continue
    frame = _frame_from_controls_state(message.t, latest, source_t=message.t)
    model = latest.get("modelV2")
    frame_id = int(getattr(model, "frameId", 0)) if model is not None else 0
    out.append(ReplayFrame(frame, float(getattr(latest["carState"], "steeringTorque", 0.0)), frame_id))
  return out


def _pipeline(variant: tuple[str, float] | None) -> LateralDemandPipeline:
  pipeline = LateralDemandPipeline(DT)
  if variant is not None:
    pipeline._model_path_processor = CornerSafeProcessor(*variant)
  return pipeline


def _inputs(replay: ReplayFrame, smoothing: bool):
  return replace(
    _frame_to_inputs(replay.frame),
    smooth_model_path_curvature=True,
    demand_jerk_smoothing_enabled=smoothing,
    lane_change_state_valid=True,
    model_frame_id=replay.model_frame_id,
  )


def _percentile(values: list[float], q: float) -> float:
  return float(np.percentile(values, q)) if values else 0.0


def _jerk_stats(values: list[float]) -> dict[str, float | int]:
  absolute = [abs(v) for v in values]
  return {
    "n": len(values),
    "p95": round(_percentile(absolute, 95), 3),
    "p99": round(_percentile(absolute, 99), 3),
    "max": round(max(absolute, default=0.0), 3),
    "excursions_gt_3": sum(v > EXCURSION_LAT_JERK for v in absolute),
  }


def _phase(previous_ay: float, current_ay: float) -> str:
  delta = abs(current_ay) - abs(previous_ay)
  if delta > 0.005:
    return "turn_in"
  if delta < -0.005:
    return "unwind"
  return "settle"


def _clean(frame: ReplayFrame, reason: str) -> bool:
  f = frame.frame
  return (
    f.lat_active
    and not f.steering_pressed
    and abs(frame.steering_torque) < 20.0
    and not f.left_blinker
    and not f.right_blinker
    and f.lane_change_state == LANE_CHANGE_STATE_OFF
    and reason in ALLOWED_REASONS
  )


def analyze(routes: list[Path]) -> dict[str, Any]:
  collected: dict[str, dict[str, list[float]]] = {
    name: {"baseline": [], "candidate": [], "turn_in_baseline": [], "turn_in_candidate": [],
           "unwind_baseline": [], "unwind_candidate": [], "lags": [], "delay": []}
    for name in VARIANTS
  }
  events: dict[str, list[Event]] = {name: [] for name in VARIANTS}
  counters: dict[str, dict[str, int]] = {
    name: {"changed_frames": 0, "new_excursions": 0, "sign_mismatches": 0, "metric_frames": 0}
    for name in VARIANTS
  }
  coverage = {"routes": len(routes), "segments": 0, "frames": 0, "clean_corner_frames": 0}

  for route_dir in routes:
    route_name = route_dir.name
    for path in _segment_paths(route_dir):
      coverage["segments"] += 1
      segment = _segment_number(path)
      baseline = _pipeline(None)
      candidates = {name: _pipeline(variant) for name, variant in VARIANTS.items()}
      previous: dict[str, float] = {}
      previous_t: float | None = None
      previous_clean = False

      for replay in _frames(path):
        coverage["frames"] += 1
        base_result = baseline.update(_inputs(replay, False))
        v_sq = replay.frame.v_ego ** 2
        base_ay = float(base_result.demand.processed_curvature) * v_sq
        reason = base_result.model_path_result.reason
        clean = _clean(replay, reason)

        candidate_ay: dict[str, float] = {}
        for name, pipeline in candidates.items():
          result = pipeline.update(_inputs(replay, True))
          candidate_ay[name] = float(result.demand.processed_curvature) * v_sq

        t = float(replay.frame.source_t if replay.frame.source_t is not None else replay.frame.t)
        dt = t - previous_t if previous_t is not None else 0.0
        in_corner = max(abs(base_ay), abs(previous.get("baseline", 0.0))) >= MIN_METRIC_LAT_ACCEL
        in_corner = in_corner and max(abs(base_ay), abs(previous.get("baseline", 0.0))) <= MAX_CORNER_LAT_ACCEL
        metric_frame = clean and previous_clean and in_corner and 0.005 <= dt <= 0.03
        if metric_frame:
          coverage["clean_corner_frames"] += 1
          base_jerk = (base_ay - previous["baseline"]) / dt
          phase = _phase(previous["baseline"], base_ay)
          for name, ay in candidate_ay.items():
            cand_jerk = (ay - previous[name]) / dt
            lag = abs(ay - base_ay)
            max_jerk = float(np.interp(replay.frame.v_ego, DEMAND_JERK_SMOOTH_SPEED_BP, DEMAND_JERK_SMOOTH_MAX_LAT_JERK))
            values = collected[name]
            values["baseline"].append(base_jerk)
            values["candidate"].append(cand_jerk)
            values["lags"].append(lag)
            values["delay"].append(lag / max_jerk)
            if phase in ("turn_in", "unwind"):
              values[f"{phase}_baseline"].append(base_jerk)
              values[f"{phase}_candidate"].append(cand_jerk)
            counters[name]["metric_frames"] += 1
            counters[name]["changed_frames"] += int(lag > 1e-6)
            counters[name]["new_excursions"] += int(abs(cand_jerk) > EXCURSION_LAT_JERK >= abs(base_jerk))
            counters[name]["sign_mismatches"] += int(
              ay * base_ay < 0.0 and min(abs(ay), abs(base_ay)) > SIGN_MISMATCH_LAT_ACCEL
            )
            if abs(base_jerk) > EXCURSION_LAT_JERK or abs(cand_jerk) > EXCURSION_LAT_JERK:
              events[name].append(Event(route_name, segment, t, phase, reason, base_jerk, cand_jerk, lag))

        previous = {"baseline": base_ay, **candidate_ay}
        previous_t = t
        previous_clean = clean

  variants: dict[str, Any] = {}
  for name, (placement, cap) in VARIANTS.items():
    values = collected[name]
    base_all = _jerk_stats(values["baseline"])
    cand_all = _jerk_stats(values["candidate"])
    base_in = _jerk_stats(values["turn_in_baseline"])
    cand_in = _jerk_stats(values["turn_in_candidate"])
    base_out = _jerk_stats(values["unwind_baseline"])
    cand_out = _jerk_stats(values["unwind_candidate"])
    max_lag = max(values["lags"], default=0.0)
    max_delay = max(values["delay"], default=0.0)

    def reduction(before: float | int, after: float | int) -> float:
      return round(1.0 - float(after) / float(before), 4) if before else 0.0

    variants[name] = {
      "placement": placement,
      "lag_cap": cap,
      "baseline": {"all": base_all, "turn_in": base_in, "unwind": base_out},
      "candidate": {"all": cand_all, "turn_in": cand_in, "unwind": cand_out},
      "reduction": {
        "all_excursions": reduction(base_all["excursions_gt_3"], cand_all["excursions_gt_3"]),
        "turn_in_p99": reduction(base_in["p99"], cand_in["p99"]),
        "unwind_p99": reduction(base_out["p99"], cand_out["p99"]),
      },
      "max_lag": round(max_lag, 4),
      "p95_lag": round(_percentile(values["lags"], 95), 4),
      "max_equivalent_delay_s": round(max_delay, 4),
      **counters[name],
      "preliminary_safety_gate": (
        max_lag <= cap + 1e-3
        and max_delay <= 0.201
        and counters[name]["sign_mismatches"] == 0
        and counters[name]["new_excursions"] == 0
      ),
      "worst_events": [
        event.to_dict()
        for event in sorted(events[name], key=lambda e: abs(e.baseline_jerk) - abs(e.candidate_jerk), reverse=True)[:12]
      ],
    }

  return {"coverage": coverage, "variants": variants}


def self_check() -> None:
  proc = CornerSafeProcessor("pre_gate", 0.20)
  base = dict(
    lat_active=True, v_ego=10.0, desired_curvature=0.0, measured_curvature=0.0,
    previous_desired_curvature=0.0, position_x=tuple(range(33)), position_y=(0.0,) * 33,
    position_y_std=(0.1,) * 33, orientation_z=(0.0,) * 33,
    orientation_rate_z=(0.0,) * 33, lane_line_probs=(0.9,) * 4,
    smooth_model_path_curvature=True, demand_jerk_smoothing_enabled=True,
    demand_jerk_smoothing_allowed=True, steering_pressed=False,
  )
  first = proc._apply_demand_jerk_smoothing(ModelPathProcessorInputs(**base), 0.0, 0.0, 1.0, "ok", 0.0)
  step_k = 0.004  # 0.4 m/s² at 10 m/s
  shaped = proc._apply_demand_jerk_smoothing(
    ModelPathProcessorInputs(**{**base, "desired_curvature": step_k}), step_k, step_k, 1.0, "ok", 0.0,
  )
  assert first[0] == 0.0
  assert 0.0 < shaped[0] < step_k
  assert abs(shaped[0] - step_k) * 100.0 <= 0.20 + 1e-9

  # Material sign reversals bypass comfort shaping.
  proc._demand_jerk_smoothed_curvature = 0.004
  reversed_out = proc._apply_demand_jerk_smoothing(
    ModelPathProcessorInputs(**{**base, "desired_curvature": -0.004}), -0.004, -0.004, 1.0, "ok", 0.0,
  )
  assert reversed_out[0] == -0.004 and not reversed_out[1]

  # Driver input fails closed to passthrough.
  blocked = proc._apply_demand_jerk_smoothing(
    ModelPathProcessorInputs(**{**base, "desired_curvature": step_k, "steering_pressed": True,
                                "demand_jerk_smoothing_allowed": False}),
    step_k, step_k, 1.0, "ok", 0.0,
  )
  assert blocked[0] == step_k and not blocked[1]
  print("self-check passed")


def render(report: dict[str, Any]) -> str:
  coverage = report["coverage"]
  lines = [
    f"coverage: {coverage['routes']} routes, {coverage['segments']} segments, "
    + f"{coverage['frames']} frames, {coverage['clean_corner_frames']} clean corner frames"
  ]
  for name, result in report["variants"].items():
    lines.append(
      f"{name}: excursions {result['baseline']['all']['excursions_gt_3']} -> "
      + f"{result['candidate']['all']['excursions_gt_3']}; "
      + f"turn-in p99 {result['baseline']['turn_in']['p99']:.2f} -> {result['candidate']['turn_in']['p99']:.2f}; "
      + f"unwind p99 {result['baseline']['unwind']['p99']:.2f} -> {result['candidate']['unwind']['p99']:.2f}; "
      + f"max lag {result['max_lag']:.3f} m/s²; delay {result['max_equivalent_delay_s']:.3f}s; "
      + f"safety={'PASS' if result['preliminary_safety_gate'] else 'FAIL'}"
    )
  return "\n".join(lines)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("routes", nargs="*", type=Path)
  parser.add_argument("--output", type=Path)
  parser.add_argument("--self-check", action="store_true")
  args = parser.parse_args()
  if args.self_check:
    self_check()
    return
  if not args.routes:
    parser.error("provide at least one cached route directory")
  missing = [str(path) for path in args.routes if not path.is_dir()]
  if missing:
    parser.error(f"missing route directories: {', '.join(missing)}")
  report = analyze(args.routes)
  if args.output:
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  print(render(report))


if __name__ == "__main__":
  main()
