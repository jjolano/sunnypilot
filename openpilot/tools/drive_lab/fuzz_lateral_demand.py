#!/usr/bin/env python3
"""Seeded lateral demand / model-path structural fuzzer.

Directly exercises ``LateralDemandPipeline`` with synthetic per-frame inputs
and checks structural invariants: finite outputs, bounded processed curvature,
sane rate/jerk, quality in [0, 1], and expected gating/reason/source behavior
for targeted scenario kinds. Each scenario gets a fresh pipeline instance so
there is no state leakage across cases.

This is intentionally not a lateral-control fuzzer; it validates the demand
pipeline's model-path processing, gating, and fallback behavior.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from openpilot.sunnypilot.custom.lateral.demand.pipeline import (
  LateralDemandPipeline,
  LateralDemandPipelineInputs,
  LateralDemandPipelineResult,
)
from openpilot.sunnypilot.custom.lateral.demand.types import DEMAND_SOURCE_LATERAL_MANEUVER
from openpilot.tools.drive_lab.lateral_scenarios import (
  LATERAL_PRESETS,
  LateralPresetRequest,
  generate_preset_scenarios,
)


ARTIFACT_SCHEMA = "drive-lab-lateral-demand-fuzzer-artifact"
ARTIFACT_VERSION = 1
DT = 0.01
N_PATH_POINTS = 33
SCENARIO_KINDS = (
  "high_quality_path",
  "invalid_path_recovery",
  "curvature_jump",
  "low_lane_confidence",
  "path_disagreement",
  "lateral_maneuver_override",
)


@dataclass(frozen=True)
class DemandFuzzThresholds:
  """Structural thresholds for the demand pipeline fuzzer."""

  max_abs_processed_curvature: float = 0.5  # 1/m
  max_abs_step_lat_accel: float = 5.0  # m/s^2 per 0.01s step
  max_abs_lat_jerk: float = 20.0  # m/s^3
  max_gated_curvature_lat_accel_delta: float = 3.0  # m/s^2 vs measured/previous
  path_quality_min: float = 0.0
  path_quality_max: float = 1.0

  def to_dict(self) -> dict[str, Any]:
    return asdict(self)

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> DemandFuzzThresholds:
    fields = cls.__dataclass_fields__
    return cls(**{key: data[key] for key in fields if key in data})


@dataclass(frozen=True)
class DemandScenario:
  """A single fuzz scenario with a full per-frame input sequence."""

  kind: str
  title: str
  duration_s: float
  frames: tuple[dict[str, Any], ...]
  thresholds: DemandFuzzThresholds | None = None

  @property
  def metric_thresholds(self) -> DemandFuzzThresholds:
    return self.thresholds or DemandFuzzThresholds()


@dataclass(frozen=True)
class DemandFrameOutput:
  """Per-frame output record from the demand pipeline."""

  t: float
  raw_curvature: float
  processed_curvature: float
  measured_curvature: float
  path_quality: float
  path_reason: str
  gated: bool
  demand_source: str
  curvature_limited: bool
  lane_change_shaping_active: bool
  lane_change_blend: float
  lane_centering_assist_active: bool
  lane_centering_reason: str
  lane_centering_curvature_nudge: float
  lane_centering_confidence: float


@dataclass(frozen=True)
class DemandScenarioResult:
  """Result of running a demand scenario through the pipeline."""

  scenario: DemandScenario
  outputs: tuple[DemandFrameOutput, ...]
  valid: bool
  failures: list[dict[str, Any]]
  metrics: dict[str, Any]


# ---------- helpers ----------


def _coherent_path(curvature: float, v_ego: float, n: int = N_PATH_POINTS) -> dict[str, Any]:
  """Return a coherent 33-point parabolic path for the given curvature."""
  xs = [float(x) for x in range(n)]
  ys = [0.5 * curvature * x * x for x in range(n)]
  y_std = [0.1] * n
  yaw = [curvature * x for x in range(n)]
  yaw_rate = [curvature * v_ego] * n
  return {
    "position_x": xs,
    "position_y": ys,
    "position_y_std": y_std,
    "orientation_z": yaw,
    "orientation_rate_z": yaw_rate,
  }


def _base_frame(
  t: float,
  v_ego: float = 20.0,
  curvature: float = 0.001,
  lat_active: bool = True,
  lane_line_probs: tuple[float, ...] = (0.9, 0.9, 0.9, 0.9),
  **kwargs: Any,
) -> dict[str, Any]:
  path = _coherent_path(curvature, v_ego)
  frame: dict[str, Any] = {
    "lat_active": lat_active,
    "v_ego": v_ego,
    "roll": 0.0,
    "desired_curvature": curvature,
    "measured_curvature": curvature,
    "lane_line_probs": lane_line_probs,
    "frame_drop_perc": 0.0,
    "model_data_v2_sp_valid": True,
    "turn_direction": 0,
    "lane_change_state": 0,
    "lane_change_direction": 0,
    "left_blinker": False,
    "right_blinker": False,
    "steering_pressed": False,
    "left_lane_y0": None,
    "right_lane_y0": None,
    "lateral_maneuver_curvature": None,
    "smooth_model_path_curvature": False,
    "lane_centering_assist_enabled": False,
    "curvature_limited": False,
  }
  frame.update(path)
  frame.update(kwargs)
  return frame


def _time_array(duration_s: float, dt: float = DT) -> np.ndarray:
  return np.arange(0.0, max(duration_s, dt) + dt * 0.5, dt, dtype=float)


def _sanitize(value: Any) -> Any:
  """Recursively sanitize floats for strict JSON output (allow_nan=False)."""
  if isinstance(value, np.generic):
    return _sanitize(value.item())
  if isinstance(value, float):
    if math.isfinite(value):
      return value
    return None
  if isinstance(value, dict):
    return {k: _sanitize(v) for k, v in value.items()}
  if isinstance(value, (list, tuple)):
    return [_sanitize(v) for v in value]
  return value


# ---------- scenario generators ----------


def _generate_high_quality_path(rng: random.Random, idx: int, duration_s: float) -> DemandScenario:
  t = _time_array(duration_s)
  v_ego = rng.uniform(15.0, 25.0)
  curvature = rng.choice([-1.0, 1.0]) * rng.uniform(0.0005, 0.0030)
  frames: list[dict[str, Any]] = []
  for i, _ in enumerate(t):
    frames.append(_base_frame(t=float(i * DT), v_ego=v_ego, curvature=curvature))
  return DemandScenario(
    kind="high_quality_path",
    title=f"fuzz high quality path #{idx}",
    duration_s=duration_s,
    frames=tuple(frames),
  )


def _generate_invalid_path_recovery(rng: random.Random, idx: int, duration_s: float) -> DemandScenario:
  t = _time_array(duration_s)
  v_ego = rng.uniform(15.0, 25.0)
  curvature = rng.choice([-1.0, 1.0]) * rng.uniform(0.0010, 0.0030)
  invalid_start = int(len(t) * rng.uniform(0.35, 0.55))
  invalid_kind = rng.choice(["short", "nonmonotonic", "steep"])
  frames: list[dict[str, Any]] = []
  for i, _ in enumerate(t):
    if i < invalid_start:
      frames.append(_base_frame(t=float(i * DT), v_ego=v_ego, curvature=curvature))
    else:
      path: dict[str, Any]
      if invalid_kind == "short":
        path = {"position_x": [], "position_y": [], "position_y_std": [], "orientation_z": [], "orientation_rate_z": []}
      elif invalid_kind == "nonmonotonic":
        n = N_PATH_POINTS
        xs = [float(x) for x in range(n)]
        ys = [0.5 * curvature * x * x for x in range(n)]
        ys[10] = ys[9] - 5.0  # huge backward lateral step -> steep slope
        path = {
          "position_x": xs,
          "position_y": ys,
          "position_y_std": [0.1] * n,
          "orientation_z": [curvature * x for x in range(n)],
          "orientation_rate_z": [curvature * v_ego] * n,
        }
      else:  # steep
        n = N_PATH_POINTS
        xs = [float(x) for x in range(n)]
        ys = [10.0 * x for x in range(n)]  # lateral slope far above limit
        path = {
          "position_x": xs,
          "position_y": ys,
          "position_y_std": [0.1] * n,
          "orientation_z": [curvature * x for x in range(n)],
          "orientation_rate_z": [curvature * v_ego] * n,
        }
      frame = _base_frame(t=float(i * DT), v_ego=v_ego, curvature=curvature)
      frame.update(path)
      frames.append(frame)
  return DemandScenario(
    kind="invalid_path_recovery",
    title=f"fuzz invalid path recovery #{idx}",
    duration_s=duration_s,
    frames=tuple(frames),
  )


def _generate_curvature_jump(rng: random.Random, idx: int, duration_s: float) -> DemandScenario:
  t = _time_array(duration_s)
  v_ego = rng.uniform(15.0, 25.0)
  base_curvature = rng.choice([-1.0, 1.0]) * rng.uniform(0.0005, 0.0015)
  # Ensure the jump exceeds the pipeline's hard lateral-accel jump threshold so
  # the curvature_jump reason is reliably reported.
  min_jump_mag = 6.0 / max(v_ego ** 2, 1.0)
  jump_mag = rng.choice([-1.0, 1.0]) * rng.uniform(max(0.010, min_jump_mag), max(0.025, min_jump_mag * 1.5))
  jump_frame = int(len(t) * rng.uniform(0.40, 0.60))
  frames: list[dict[str, Any]] = []
  for i, _ in enumerate(t):
    k = base_curvature + (jump_mag if i >= jump_frame else 0.0)
    frames.append(_base_frame(t=float(i * DT), v_ego=v_ego, curvature=k))
  return DemandScenario(
    kind="curvature_jump",
    title=f"fuzz curvature jump #{idx}",
    duration_s=duration_s,
    frames=tuple(frames),
  )


def _generate_low_lane_confidence(rng: random.Random, idx: int, duration_s: float) -> DemandScenario:
  t = _time_array(duration_s)
  v_ego = rng.uniform(5.0, 12.0)  # low-speed so sustained low lane confidence gates
  curvature = rng.choice([-1.0, 1.0]) * rng.uniform(0.0010, 0.0040)
  low_start = int(len(t) * rng.uniform(0.35, 0.55))
  frames: list[dict[str, Any]] = []
  for i, _ in enumerate(t):
    probs = (0.05, 0.05, 0.05, 0.05) if i >= low_start else (0.9, 0.9, 0.9, 0.9)
    frames.append(_base_frame(t=float(i * DT), v_ego=v_ego, curvature=curvature, lane_line_probs=probs))
  return DemandScenario(
    kind="low_lane_confidence",
    title=f"fuzz low lane confidence #{idx}",
    duration_s=duration_s,
    frames=tuple(frames),
  )


def _generate_path_disagreement(rng: random.Random, idx: int, duration_s: float) -> DemandScenario:
  t = _time_array(duration_s)
  v_ego = rng.uniform(15.0, 25.0)
  curvature = rng.choice([-1.0, 1.0]) * rng.uniform(0.0010, 0.0040)
  disagree_start = int(len(t) * rng.uniform(0.35, 0.55))
  frames: list[dict[str, Any]] = []
  for i, _ in enumerate(t):
    frame = _base_frame(t=float(i * DT), v_ego=v_ego, curvature=curvature)
    if i >= disagree_start:
      # Path orientation implies a curvature far enough from desired_curvature to
      # exceed the pipeline's path-disagreement lateral-accel threshold.
      n = N_PATH_POINTS
      sign_k = 1.0 if curvature >= 0 else -1.0
      opposite_k = curvature - sign_k * rng.uniform(0.012, 0.025)
      frame["orientation_z"] = [opposite_k * x for x in range(n)]
      frame["orientation_rate_z"] = [opposite_k * v_ego] * n
    frames.append(frame)
  return DemandScenario(
    kind="path_disagreement",
    title=f"fuzz path disagreement #{idx}",
    duration_s=duration_s,
    frames=tuple(frames),
  )


def _generate_lateral_maneuver_override(rng: random.Random, idx: int, duration_s: float) -> DemandScenario:
  t = _time_array(duration_s)
  v_ego = rng.uniform(15.0, 25.0)
  base_curvature = rng.choice([-1.0, 1.0]) * rng.uniform(0.0005, 0.0015)
  override_curvature = rng.choice([-1.0, 1.0]) * rng.uniform(0.005, 0.020)
  override_start = int(len(t) * rng.uniform(0.35, 0.55))
  override_end = int(len(t) * rng.uniform(0.65, 0.80))
  frames: list[dict[str, Any]] = []
  for i, _ in enumerate(t):
    override = override_curvature if override_start <= i < override_end else None
    frame = _base_frame(t=float(i * DT), v_ego=v_ego, curvature=base_curvature, lateral_maneuver_curvature=override)
    frames.append(frame)
  return DemandScenario(
    kind="lateral_maneuver_override",
    title=f"fuzz lateral maneuver override #{idx}",
    duration_s=duration_s,
    frames=tuple(frames),
  )


SCENARIO_GENERATORS: dict[str, Any] = {
  "high_quality_path": _generate_high_quality_path,
  "invalid_path_recovery": _generate_invalid_path_recovery,
  "curvature_jump": _generate_curvature_jump,
  "low_lane_confidence": _generate_low_lane_confidence,
  "path_disagreement": _generate_path_disagreement,
  "lateral_maneuver_override": _generate_lateral_maneuver_override,
}


# ---------- runner / evaluation ----------


@dataclass
class DemandFuzzerConfig:
  seed: int = 1
  cases: int = 100
  kind: str | None = None
  duration_s: float = 2.0


def generate_scenarios(config: DemandFuzzerConfig) -> list[DemandScenario]:
  rng = random.Random(config.seed)
  kinds = [config.kind] if config.kind else list(SCENARIO_KINDS)
  generators = [SCENARIO_GENERATORS[k] for k in kinds]
  scenarios: list[DemandScenario] = []
  for idx in range(config.cases):
    gen = rng.choice(generators)
    scenario = gen(rng, idx, config.duration_s)
    scenarios.append(scenario)
  return scenarios


def _frame_to_inputs(frame: dict[str, Any]) -> LateralDemandPipelineInputs:
  return LateralDemandPipelineInputs(**frame)


def _output_from_result(t: float, result: LateralDemandPipelineResult) -> DemandFrameOutput:
  d = result.demand
  return DemandFrameOutput(
    t=t,
    raw_curvature=d.raw_curvature,
    processed_curvature=d.processed_curvature,
    measured_curvature=d.measured_curvature,
    path_quality=d.path_quality,
    path_reason=result.model_path_result.reason,
    gated=result.model_path_result.gated,
    demand_source=d.demand_source,
    curvature_limited=d.curvature_limited,
    lane_change_shaping_active=d.lane_change_shaping_active,
    lane_change_blend=d.lane_change_blend,
    lane_centering_assist_active=d.lane_centering_assist_active,
    lane_centering_reason=d.lane_centering_reason,
    lane_centering_curvature_nudge=d.lane_centering_curvature_nudge,
    lane_centering_confidence=d.lane_centering_confidence,
  )


def _lat_accel(v_ego: float, curvature: float) -> float:
  return (v_ego * v_ego) * curvature


def evaluate_scenario(scenario: DemandScenario) -> DemandScenarioResult:
  thresholds = scenario.metric_thresholds
  failures: list[dict[str, Any]] = []
  metrics: dict[str, Any] = {}
  outputs: list[DemandFrameOutput] = []

  pipeline = LateralDemandPipeline(dt=DT)
  for idx, frame in enumerate(scenario.frames):
    try:
      inputs = _frame_to_inputs(frame)
      result = pipeline.update(inputs)
      outputs.append(_output_from_result(idx * DT, result))
    except Exception as exc:
      failures.append({
        "check": "exception",
        "detail": f"frame {idx} t={idx * DT:.2f} raised {type(exc).__name__}: {exc}",
        "traceback": traceback.format_exc(),
      })
      break

  if failures:
    return DemandScenarioResult(scenario, tuple(outputs), False, failures, metrics)
  if not outputs:
    failures.append({"check": "output", "detail": "scenario produced no output frames"})
    return DemandScenarioResult(scenario, tuple(outputs), False, failures, metrics)

  t_arr = np.array([o.t for o in outputs], dtype=float)
  raw = np.array([o.raw_curvature for o in outputs], dtype=float)
  processed = np.array([o.processed_curvature for o in outputs], dtype=float)
  measured = np.array([o.measured_curvature for o in outputs], dtype=float)
  quality = np.array([o.path_quality for o in outputs], dtype=float)
  v_ego_arr = np.array([frame["v_ego"] for frame in scenario.frames], dtype=float)

  # Finite outputs.
  if not np.all(np.isfinite(processed)) or not np.all(np.isfinite(raw)) or not np.all(np.isfinite(measured)) or not np.all(np.isfinite(quality)):
    failures.append({"check": "finite", "detail": "non-finite raw/processed/measured/quality output"})

  # Path quality bounds.
  min_q = float(np.min(quality))
  max_q = float(np.max(quality))
  metrics["min_path_quality"] = min_q
  metrics["max_path_quality"] = max_q
  if min_q < thresholds.path_quality_min - 1e-6 or max_q > thresholds.path_quality_max + 1e-6:
    failures.append({"check": "path_quality_range", "detail": f"path_quality {min_q:.3f}..{max_q:.3f} outside [0,1]"})

  # Processed curvature cap.
  max_abs_k = float(np.max(np.abs(processed)))
  metrics["max_abs_processed_curvature"] = max_abs_k
  if max_abs_k > thresholds.max_abs_processed_curvature:
    failures.append({"check": "curvature_cap", "detail": f"processed curvature |{max_abs_k:.3f}| exceeds {thresholds.max_abs_processed_curvature}"})

  # Step/rate checks via lateral accel/jerk.
  if len(processed) > 1:
    lat_accel = _lat_accel(v_ego_arr, processed)
    lat_accel_step = np.diff(lat_accel)
    # Lateral-maneuver override is an intentional bypass/source transition. The
    # source check below owns that scenario; don't let structural model-path
    # continuity checks report the deliberate handoff as demand instability.
    override = np.array([frame.get("lateral_maneuver_curvature") is not None for frame in scenario.frames], dtype=bool)
    if scenario.kind == "lateral_maneuver_override":
      step_mask = np.zeros_like(lat_accel_step, dtype=bool)
    else:
      step_mask = ~(override[1:] | override[:-1]) if override.size == len(processed) else np.ones_like(lat_accel_step, dtype=bool)
    dt = float(np.mean(np.diff(t_arr))) if len(t_arr) > 1 else DT
    valid_dt = dt > 1e-6
    if valid_dt:
      checked_steps = lat_accel_step[step_mask]
      lat_jerk = checked_steps / dt
      max_step = float(np.max(np.abs(checked_steps))) if checked_steps.size else 0.0
      max_jerk = float(np.max(np.abs(lat_jerk))) if lat_jerk.size else 0.0
      metrics["max_abs_lat_accel_step"] = max_step
      metrics["max_abs_lat_jerk"] = max_jerk
      if max_step > thresholds.max_abs_step_lat_accel:
        failures.append({"check": "lat_accel_step", "detail": f"lateral accel step {max_step:.2f} m/s^2 exceeds {thresholds.max_abs_step_lat_accel}"})
      if max_jerk > thresholds.max_abs_lat_jerk:
        failures.append({"check": "lat_jerk", "detail": f"lateral jerk {max_jerk:.2f} m/s^3 exceeds {thresholds.max_abs_lat_jerk}"})

  # Gated/low-quality frames should not drift far from measured/previous.
  for i, out in enumerate(outputs):
    if out.gated or out.path_quality < 0.5:
      # Skip drift checks around lateral-maneuver override transitions; the
      # pipeline resets on entry and the curvature handoff on exit is expected
      # to be a large intentional step.
      prev_override = i > 0 and scenario.frames[i - 1].get("lateral_maneuver_curvature") is not None
      curr_override = scenario.frames[i].get("lateral_maneuver_curvature") is not None
      if prev_override or curr_override:
        continue
      fallback = measured[i] if i == 0 else processed[i - 1]
      if not math.isfinite(fallback):
        fallback = measured[i]
      v = v_ego_arr[i]
      delta_lat_accel = abs(_lat_accel(v, out.processed_curvature) - _lat_accel(v, fallback))
      if delta_lat_accel > thresholds.max_gated_curvature_lat_accel_delta:
        failures.append({
          "check": "gated_drift",
          "detail": f"frame {i} gated/q={out.path_quality:.2f} drifted {delta_lat_accel:.2f} m/s^2 from fallback",
        })
        break  # one representative failure is enough

  # Scenario-kind-specific strict checks (look in the expected window).
  reasons = [out.path_reason for out in outputs]
  sources = [out.demand_source for out in outputs]

  if scenario.kind == "invalid_path_recovery":
    if not any(r == "invalid_path" for r in reasons):
      failures.append({"check": "expected_invalid_path", "detail": "invalid_path reason not observed during invalid path window"})

  if scenario.kind == "curvature_jump":
    if not any(r == "curvature_jump" for r in reasons):
      failures.append({"check": "expected_curvature_jump", "detail": "curvature_jump reason not observed during jump window"})

  if scenario.kind == "low_lane_confidence":
    if not (any(r == "low_lane_confidence" for r in reasons) or min_q < 0.9):
      failures.append({"check": "expected_low_lane_confidence", "detail": "low_lane_confidence reason or reduced quality not observed"})

  if scenario.kind == "path_disagreement":
    if not (any(r == "path_disagreement" for r in reasons) or min_q < 0.9):
      failures.append({"check": "expected_path_disagreement", "detail": "path_disagreement reason or reduced quality not observed"})

  if scenario.kind == "lateral_maneuver_override":
    override_frames = [i for i, f in enumerate(scenario.frames) if f.get("lateral_maneuver_curvature") is not None]
    if override_frames and not all(sources[i] == DEMAND_SOURCE_LATERAL_MANEUVER for i in override_frames):
      failures.append({"check": "expected_maneuver_source", "detail": "override frames did not report lateral_maneuver demand source"})

  return DemandScenarioResult(scenario, tuple(outputs), not failures, failures, metrics)


# ---------- serialization / CLI ----------


def scenario_to_dict(scenario: DemandScenario, seed: int | None = None, index: int | None = None) -> dict[str, Any]:
  payload: dict[str, Any] = {
    "kind": scenario.kind,
    "title": scenario.title,
    "duration_s": scenario.duration_s,
    "frames": [_sanitize(frame) for frame in scenario.frames],
    "thresholds": scenario.metric_thresholds.to_dict(),
  }
  if seed is not None:
    payload["seed"] = seed
  if index is not None:
    payload["index"] = index
  return payload


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


def scenario_from_dict(data: dict[str, Any]) -> DemandScenario:
  frames = tuple(dict(frame) for frame in data.get("frames", []))
  # Restore None for nullable fields that may have been sanitized.
  nullable = ("left_lane_y0", "right_lane_y0", "lateral_maneuver_curvature")
  for frame in frames:
    for key in nullable:
      if key in frame and frame[key] is None:
        pass  # keep None
  return DemandScenario(
    kind=str(data["kind"]),
    title=str(data["title"]),
    duration_s=float(data["duration_s"]),
    frames=frames,
    thresholds=DemandFuzzThresholds.from_dict(data.get("thresholds", {})),
  )


def artifact_to_dict(result: DemandScenarioResult, seed: int | None, index: int | None) -> dict[str, Any]:
  return {
    "schema": ARTIFACT_SCHEMA,
    "version": ARTIFACT_VERSION,
    "seed": seed,
    "index": index,
    "kind": result.scenario.kind,
    "scenario": scenario_to_dict(result.scenario, seed=seed, index=index),
    "thresholds": result.scenario.metric_thresholds.to_dict(),
    "valid": result.valid,
    "failures": result.failures,
    "metrics": _sanitize(result.metrics),
    "summary": {
      "frames": len(result.outputs),
      "failure_checks": [f["check"] for f in result.failures],
    },
  }


def write_artifact(result: DemandScenarioResult, artifact_dir: Path, seed: int | None, index: int | None) -> Path:
  artifact_dir.mkdir(parents=True, exist_ok=True)
  filename = f"lateral_demand_failure_{result.scenario.kind}_seed{seed}_idx{index}.json"
  path = artifact_dir / filename
  path.write_text(json.dumps(_sanitize(artifact_to_dict(result, seed, index)), indent=2, sort_keys=True, allow_nan=False))
  return path


def load_artifact(path: str | Path) -> dict[str, Any]:
  return json.loads(Path(path).read_text())


def replay_artifact(path: str | Path) -> DemandScenarioResult:
  data = load_artifact(path)
  scenario = scenario_from_dict(data["scenario"])
  return evaluate_scenario(scenario)


def _render_scenario_snippet(scenario: DemandScenario) -> str:
  lines = [
    f"# kind: {scenario.kind}",
    f"DemandScenario(",
    f"    title={scenario.title!r},",
    f"    duration_s={scenario.duration_s!r},",
    f"    frames=[...{len(scenario.frames)} frames...],",
    f")",
  ]
  return "\n".join(lines)


def main() -> None:
  parser = argparse.ArgumentParser(description="Seeded lateral demand / model-path structural fuzzer.")
  parser.add_argument("--seed", type=int, default=1)
  parser.add_argument("--cases", type=int, default=100)
  parser.add_argument("--kind", choices=SCENARIO_KINDS, help="Run only one scenario kind")
  parser.add_argument("--preset", choices=LATERAL_PRESETS, help="Public lateral benchmark preset")
  parser.add_argument("--nhtsa-family", choices=("primary", "secondary"), help="NHTSA LKA test family filter")
  parser.add_argument("--nhtsa-line-type", help="NHTSA LKA line type filter")
  parser.add_argument("--nhtsa-drift-rate", type=float, help="NHTSA LKA drift rate filter (m/s)")
  parser.add_argument("--euroncap-family", choices=("lka", "elk", "sbend", "alc"), help="Euro NCAP LSS family filter")
  parser.add_argument("--nuplan-focus", choices=("error", "jerk", "oscillation"), help="nuPlan lateral focus filter")
  parser.add_argument("--stress-grid-sample", type=int, default=None, help="Number of random stress-grid cells (None=full grid)")
  parser.add_argument("--profile", type=str, default=None, help="LateralProfile JSON for profile-guided fuzzing")
  parser.add_argument("--export-specs", type=str, default=None, help="Export scenarios as ScenarioSpec JSON for behavior_change_gate")
  parser.add_argument("--duration", type=float, default=2.0, help="Scenario duration in seconds")
  parser.add_argument("--json", action="store_true", help="Emit JSON instead of text")
  parser.add_argument("--fail-fast", action="store_true", help="Stop after the first failure")
  parser.add_argument("--artifact-dir", type=str, default=None, help="Directory to write failure artifacts")
  parser.add_argument("--replay", type=str, default=None, help="Replay a failure artifact JSON file")
  args = parser.parse_args()

  if args.cases < 0:
    parser.error("--cases must be >= 0")
  if args.duration <= 0.0:
    parser.error("--duration must be > 0")

  if args.replay:
    result = replay_artifact(args.replay)
    if args.json:
      print(json.dumps(_sanitize(artifact_to_dict(result, seed=None, index=None)), indent=2, sort_keys=True, allow_nan=False))
    else:
      print(f"Replayed {args.replay}: valid={result.valid} failures={len(result.failures)}")
      for failure in result.failures:
        print(f"  {failure['check']}: {failure['detail']}")
    sys.exit(0 if result.valid else 1)

  if args.preset:
    profile = None
    if args.profile:
        from openpilot.tools.drive_lab.log_profile import load_lateral_profile
        profile = load_lateral_profile(args.profile)
    request = LateralPresetRequest(
      preset=args.preset,
      seed=args.seed,
      cases=args.cases,
      duration_s=args.duration,
      profile=profile,
      nhtsa_family=args.nhtsa_family,
      nhtsa_line_type=args.nhtsa_line_type,
      nhtsa_drift_rate=args.nhtsa_drift_rate,
      euroncap_family=args.euroncap_family,
      nuplan_focus=args.nuplan_focus,
      stress_grid_sample=args.stress_grid_sample,
    )
    scenarios = generate_preset_scenarios(request)
  else:
    config = DemandFuzzerConfig(seed=args.seed, cases=args.cases, kind=args.kind, duration_s=args.duration)
    scenarios = generate_scenarios(config)

  if args.export_specs:
    from openpilot.tools.drive_lab.lateral_scenarios import lateral_scenario_to_spec
    source = args.preset or "fuzz"
    specs = [lateral_scenario_to_spec(s, source=source, seed=args.seed, index=i)
             for i, s in enumerate(scenarios)]
    payload: dict[str, Any] = {"scenarios": [spec.to_dict() for spec in specs]}
    path = Path(args.export_specs)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_sanitize(payload), indent=2, sort_keys=True, allow_nan=False))
    print(f"Exported {len(specs)} ScenarioSpec(s) to {args.export_specs}")
    return

  results: list[tuple[int, DemandScenarioResult]] = []
  for idx, scenario in enumerate(scenarios):
    result = evaluate_scenario(scenario)
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
      "dt": DT,
      "failures": [
        {
          "scenario": scenario_summary_to_dict(result.scenario, seed=args.seed, index=result_idx),
          "artifact_hint": "rerun with --artifact-dir for full per-frame replay input",
          "checks": [f["check"] for f in result.failures],
          "metrics": _sanitize(result.metrics),
        }
        for result_idx, result in failures
      ],
    }
    if args.preset:
      payload["preset"] = args.preset
    print(json.dumps(_sanitize(payload), indent=2, sort_keys=True, allow_nan=False))
  else:
    print(
      f"Drive Lab lateral demand fuzz seed={args.seed} cases={len(results)} "
      f"kind={args.kind or 'all'} preset={args.preset or 'none'} "
      f"duration={args.duration}s dt={DT}s failures={len(failures)}"
    )
    for idx, result in failures[:10]:
      print(f"\nFAILED: {result.scenario.title} [{result.scenario.kind}]")
      for failure in result.failures:
        print(f"  {failure['check']}: {failure['detail']}")
      print(_render_scenario_snippet(result.scenario))
    if artifact_paths:
      print(f"\nWrote {len(artifact_paths)} failure artifact(s) to {args.artifact_dir}")

  if failures:
    raise SystemExit(1)


if __name__ == "__main__":
  main()
