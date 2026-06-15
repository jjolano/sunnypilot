#!/usr/bin/env python3
"""Synthetic / serialized route-like lateral replay perturbation fuzzer.

This tool generates deterministic synthetic lateral "route" sequences, perturbs
them with causal, materialized perturbations, and replays both the baseline and
perturbed sequences through ``LateralDemandPipeline``. It is *not* real route
loading (that is deferred); everything is synthesized and fully serialized so
replay needs no RNG or generator behavior.

Each scenario stores:
  - baseline frames
  - a materialized perturbation recipe
  - perturbed frames derived from the recipe
Replay reconstructs both frame sequences directly from the artifact.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from openpilot.sunnypilot.custom.lateral.demand.pipeline import (
  LateralDemandPipeline,
  LateralDemandPipelineInputs,
)
from openpilot.sunnypilot.custom.lateral.demand.types import DEMAND_SOURCE_MODEL_PATH


ARTIFACT_SCHEMA = "drive-lab-lateral-route-replay-fuzzer-artifact"
ARTIFACT_VERSION = 1
DT = 0.01
N_PATH_POINTS = 33
PRESETS = ("synthetic_straight", "synthetic_curve", "synthetic_sine", "synthetic_reversal")
PERTURBATION_KINDS = ("noise", "dropout", "delay", "stale", "scale", "offset")
CLI_PERTURBATION_KINDS = PERTURBATION_KINDS + ("none",)


@dataclass(frozen=True)
class LateralRouteFrame:
  """One synthetic lateral route frame (sufficient to build pipeline inputs)."""

  t: float
  v_ego: float
  lat_active: bool
  raw_curvature: float
  measured_curvature: float
  roll: float
  steering_pressed: bool
  left_blinker: bool
  right_blinker: bool
  lane_change_state: int
  lane_change_direction: int
  position_x: tuple[float, ...]
  position_y: tuple[float, ...]
  position_y_std: tuple[float, ...]
  orientation_z: tuple[float, ...]
  orientation_rate_z: tuple[float, ...]
  lane_line_probs: tuple[float, ...]
  frame_drop_perc: float

  def to_dict(self) -> dict[str, Any]:
    return {
      "t": self.t,
      "v_ego": self.v_ego,
      "lat_active": self.lat_active,
      "raw_curvature": self.raw_curvature,
      "measured_curvature": self.measured_curvature,
      "roll": self.roll,
      "steering_pressed": self.steering_pressed,
      "left_blinker": self.left_blinker,
      "right_blinker": self.right_blinker,
      "lane_change_state": self.lane_change_state,
      "lane_change_direction": self.lane_change_direction,
      "position_x": list(self.position_x),
      "position_y": list(self.position_y),
      "position_y_std": list(self.position_y_std),
      "orientation_z": list(self.orientation_z),
      "orientation_rate_z": list(self.orientation_rate_z),
      "lane_line_probs": list(self.lane_line_probs),
      "frame_drop_perc": self.frame_drop_perc,
    }

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> LateralRouteFrame:
    return cls(
      t=float(data["t"]),
      v_ego=float(data["v_ego"]),
      lat_active=bool(data["lat_active"]),
      raw_curvature=float(data["raw_curvature"]),
      measured_curvature=float(data["measured_curvature"]),
      roll=float(data["roll"]),
      steering_pressed=bool(data["steering_pressed"]),
      left_blinker=bool(data["left_blinker"]),
      right_blinker=bool(data["right_blinker"]),
      lane_change_state=int(data["lane_change_state"]),
      lane_change_direction=int(data["lane_change_direction"]),
      position_x=tuple(float(v) for v in data["position_x"]),
      position_y=tuple(float(v) for v in data["position_y"]),
      position_y_std=tuple(float(v) for v in data["position_y_std"]),
      orientation_z=tuple(float(v) for v in data["orientation_z"]),
      orientation_rate_z=tuple(float(v) for v in data["orientation_rate_z"]),
      lane_line_probs=tuple(float(v) for v in data["lane_line_probs"]),
      frame_drop_perc=float(data["frame_drop_perc"]),
    )


@dataclass(frozen=True)
class PerturbationRecipe:
  """Fully materialized perturbation description."""

  kind: str
  start_frame: int
  end_frame: int
  description: str
  params: dict[str, Any] = field(default_factory=dict)

  def to_dict(self) -> dict[str, Any]:
    return {
      "kind": self.kind,
      "start_frame": self.start_frame,
      "end_frame": self.end_frame,
      "description": self.description,
      "params": _sanitize(self.params),
    }

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> PerturbationRecipe:
    return cls(
      kind=str(data["kind"]),
      start_frame=int(data["start_frame"]),
      end_frame=int(data["end_frame"]),
      description=str(data["description"]),
      params=dict(data.get("params", {})),
    )


@dataclass(frozen=True)
class RouteReplayThresholds:
  """Thresholds for route-replay structural and comparison checks."""

  max_abs_processed_curvature: float = 0.5
  max_abs_step_lat_accel: float = 5.0
  max_abs_lat_jerk: float = 150.0
  path_quality_min: float = 0.0
  path_quality_max: float = 1.0
  max_baseline_perturbed_lat_accel_delta: float = 3.0
  oscillation_lat_accel_eps: float = 0.5
  max_extra_sign_flips: int = 4

  def to_dict(self) -> dict[str, Any]:
    return {
      "max_abs_processed_curvature": self.max_abs_processed_curvature,
      "max_abs_step_lat_accel": self.max_abs_step_lat_accel,
      "max_abs_lat_jerk": self.max_abs_lat_jerk,
      "path_quality_min": self.path_quality_min,
      "path_quality_max": self.path_quality_max,
      "max_baseline_perturbed_lat_accel_delta": self.max_baseline_perturbed_lat_accel_delta,
      "oscillation_lat_accel_eps": self.oscillation_lat_accel_eps,
      "max_extra_sign_flips": self.max_extra_sign_flips,
    }

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> RouteReplayThresholds:
    return cls(**{k: data[k] for k in cls.__dataclass_fields__ if k in data})


@dataclass(frozen=True)
class RouteReplayScenario:
  preset: str
  title: str
  frames: tuple[LateralRouteFrame, ...]
  recipe: PerturbationRecipe
  thresholds: RouteReplayThresholds | None = None
  perturbed_frames: tuple[LateralRouteFrame, ...] = ()

  @property
  def metric_thresholds(self) -> RouteReplayThresholds:
    return self.thresholds or RouteReplayThresholds()


@dataclass(frozen=True)
class RouteFrameOutput:
  """Per-frame pipeline output."""

  t: float
  v_ego: float
  raw_curvature: float
  processed_curvature: float
  measured_curvature: float
  path_quality: float
  path_reason: str
  gated: bool
  demand_source: str


@dataclass(frozen=True)
class RouteReplayResult:
  scenario: RouteReplayScenario
  baseline_outputs: tuple[RouteFrameOutput, ...]
  perturbed_outputs: tuple[RouteFrameOutput, ...]
  baseline_failures: list[dict[str, Any]]
  perturbation_failures: list[dict[str, Any]]
  comparison_failures: list[dict[str, Any]]
  metrics: dict[str, Any]

  @property
  def valid(self) -> bool:
    return not (self.baseline_failures or self.perturbation_failures or self.comparison_failures)


# ---------- helpers ----------


def _sanitize(value: Any) -> Any:
  if isinstance(value, np.generic):
    return _sanitize(value.item())
  if isinstance(value, float):
    return value if math.isfinite(value) else None
  if isinstance(value, dict):
    return {k: _sanitize(v) for k, v in value.items()}
  if isinstance(value, (list, tuple)):
    return [_sanitize(v) for v in value]
  return value


def _time_array(duration_s: float, dt: float = DT) -> np.ndarray:
  return np.arange(0.0, max(duration_s, dt) + dt * 0.5, dt, dtype=float)


def _coherent_path(curvature: float, v_ego: float, n: int = N_PATH_POINTS) -> dict[str, tuple[float, ...]]:
  xs = tuple(float(x) for x in range(n))
  ys = tuple(0.5 * curvature * x * x for x in range(n))
  ystd = tuple(0.1 for _ in range(n))
  yaw = tuple(curvature * x for x in range(n))
  yaw_rate = tuple(curvature * v_ego for _ in range(n))
  return {"position_x": xs, "position_y": ys, "position_y_std": ystd, "orientation_z": yaw, "orientation_rate_z": yaw_rate}


def _frame_to_inputs(frame: LateralRouteFrame) -> LateralDemandPipelineInputs:
  return LateralDemandPipelineInputs(
    lat_active=frame.lat_active,
    v_ego=frame.v_ego,
    roll=frame.roll,
    desired_curvature=frame.raw_curvature,
    measured_curvature=frame.measured_curvature,
    position_x=frame.position_x,
    position_y=frame.position_y,
    position_y_std=frame.position_y_std,
    orientation_z=frame.orientation_z,
    orientation_rate_z=frame.orientation_rate_z,
    lane_line_probs=frame.lane_line_probs,
    frame_drop_perc=frame.frame_drop_perc,
    left_blinker=frame.left_blinker,
    right_blinker=frame.right_blinker,
    lane_change_state=frame.lane_change_state,
    lane_change_direction=frame.lane_change_direction,
  )


def _output_from_result(frame: LateralRouteFrame, pipeline_result: Any) -> RouteFrameOutput:
  d = pipeline_result.demand
  return RouteFrameOutput(
    t=frame.t,
    v_ego=frame.v_ego,
    raw_curvature=d.raw_curvature,
    processed_curvature=d.processed_curvature,
    measured_curvature=d.measured_curvature,
    path_quality=d.path_quality,
    path_reason=pipeline_result.model_path_result.reason,
    gated=pipeline_result.model_path_result.gated,
    demand_source=d.demand_source,
  )


# ---------- presets ----------


def _generate_preset_frames(preset: str, duration_s: float, v_ego: float, curvature_scale: float) -> tuple[LateralRouteFrame, ...]:
  t = _time_array(duration_s)
  n = len(t)
  base_k: float
  if preset == "synthetic_straight":
    base_k = 0.0
    desired = np.zeros(n)
  elif preset == "synthetic_curve":
    base_k = curvature_scale
    desired = np.full(n, base_k)
  elif preset == "synthetic_sine":
    base_k = curvature_scale
    desired = base_k * np.sin(2.0 * math.pi * 0.25 * t)
  elif preset == "synthetic_reversal":
    base_k = curvature_scale
    # Smooth reversals over ~0.2 s to keep jerk realistic.
    ramp = 0.2
    desired = base_k * np.tanh(np.sin(2.0 * math.pi * t / 4.0) / max(ramp, 1e-3))
  else:
    raise ValueError(f"unknown preset {preset!r}")

  frames: list[LateralRouteFrame] = []
  for i, ti in enumerate(t):
    k = float(desired[i])
    path = _coherent_path(k, v_ego)
    frames.append(LateralRouteFrame(
      t=float(ti),
      v_ego=v_ego,
      lat_active=True,
      raw_curvature=k,
      measured_curvature=k,
      roll=0.0,
      steering_pressed=False,
      left_blinker=False,
      right_blinker=False,
      lane_change_state=0,
      lane_change_direction=0,
      position_x=path["position_x"],
      position_y=path["position_y"],
      position_y_std=path["position_y_std"],
      orientation_z=path["orientation_z"],
      orientation_rate_z=path["orientation_rate_z"],
      lane_line_probs=(0.9, 0.9, 0.9, 0.9),
      frame_drop_perc=0.0,
    ))
  return tuple(frames)


# ---------- perturbations ----------


def _apply_noise(rng: random.Random, frames: tuple[LateralRouteFrame, ...], start: int, end: int) -> tuple[LateralRouteFrame, ...]:
  noise_std = rng.uniform(1e-4, 3e-4)
  noise = [rng.gauss(0.0, noise_std) for _ in range(start, end)]
  out: list[LateralRouteFrame] = []
  for i, frame in enumerate(frames):
    if start <= i < end:
      k = frame.raw_curvature + noise[i - start]
      path = _coherent_path(k, frame.v_ego)
      out.append(frame.__class__(
        t=frame.t,
        v_ego=frame.v_ego,
        lat_active=frame.lat_active,
        raw_curvature=k,
        measured_curvature=frame.measured_curvature,
        roll=frame.roll,
        steering_pressed=frame.steering_pressed,
        left_blinker=frame.left_blinker,
        right_blinker=frame.right_blinker,
        lane_change_state=frame.lane_change_state,
        lane_change_direction=frame.lane_change_direction,
        position_x=path["position_x"],
        position_y=path["position_y"],
        position_y_std=frame.position_y_std,
        orientation_z=path["orientation_z"],
        orientation_rate_z=path["orientation_rate_z"],
        lane_line_probs=frame.lane_line_probs,
        frame_drop_perc=frame.frame_drop_perc,
      ))
    else:
      out.append(frame)
  return tuple(out)


def _apply_dropout(frames: tuple[LateralRouteFrame, ...], start: int, end: int) -> tuple[LateralRouteFrame, ...]:
  out: list[LateralRouteFrame] = []
  for i, frame in enumerate(frames):
    if start <= i < end:
      out.append(frame.__class__(
        t=frame.t,
        v_ego=frame.v_ego,
        lat_active=frame.lat_active,
        raw_curvature=frame.raw_curvature,
        measured_curvature=frame.measured_curvature,
        roll=frame.roll,
        steering_pressed=frame.steering_pressed,
        left_blinker=frame.left_blinker,
        right_blinker=frame.right_blinker,
        lane_change_state=frame.lane_change_state,
        lane_change_direction=frame.lane_change_direction,
        position_x=(),
        position_y=(),
        position_y_std=(),
        orientation_z=(),
        orientation_rate_z=(),
        lane_line_probs=frame.lane_line_probs,
        frame_drop_perc=frame.frame_drop_perc,
      ))
    else:
      out.append(frame)
  return tuple(out)


def _apply_delay(frames: tuple[LateralRouteFrame, ...], start: int, end: int, delay_frames: int) -> tuple[LateralRouteFrame, ...]:
  out = list(frames)
  delay_frames = min(delay_frames, start)
  for i in range(start, end):
    src = frames[i - delay_frames]
    k = src.raw_curvature
    path = _coherent_path(k, frames[i].v_ego)
    out[i] = frames[i].__class__(
      t=frames[i].t,
      v_ego=frames[i].v_ego,
      lat_active=frames[i].lat_active,
      raw_curvature=k,
      measured_curvature=frames[i].measured_curvature,
      roll=frames[i].roll,
      steering_pressed=frames[i].steering_pressed,
      left_blinker=frames[i].left_blinker,
      right_blinker=frames[i].right_blinker,
      lane_change_state=frames[i].lane_change_state,
      lane_change_direction=frames[i].lane_change_direction,
      position_x=path["position_x"],
      position_y=path["position_y"],
      position_y_std=frames[i].position_y_std,
      orientation_z=path["orientation_z"],
      orientation_rate_z=path["orientation_rate_z"],
      lane_line_probs=frames[i].lane_line_probs,
      frame_drop_perc=frames[i].frame_drop_perc,
    )
  return tuple(out)


def _apply_stale(frames: tuple[LateralRouteFrame, ...], start: int, end: int) -> tuple[LateralRouteFrame, ...]:
  src = frames[start]
  k = src.raw_curvature
  out = list(frames)
  for i in range(start, end):
    path = _coherent_path(k, frames[i].v_ego)
    out[i] = frames[i].__class__(
      t=frames[i].t,
      v_ego=frames[i].v_ego,
      lat_active=frames[i].lat_active,
      raw_curvature=k,
      measured_curvature=frames[i].measured_curvature,
      roll=frames[i].roll,
      steering_pressed=frames[i].steering_pressed,
      left_blinker=frames[i].left_blinker,
      right_blinker=frames[i].right_blinker,
      lane_change_state=frames[i].lane_change_state,
      lane_change_direction=frames[i].lane_change_direction,
      position_x=path["position_x"],
      position_y=path["position_y"],
      position_y_std=frames[i].position_y_std,
      orientation_z=path["orientation_z"],
      orientation_rate_z=path["orientation_rate_z"],
      lane_line_probs=frames[i].lane_line_probs,
      frame_drop_perc=frames[i].frame_drop_perc,
    )
  return tuple(out)


def _apply_scale(rng: random.Random, frames: tuple[LateralRouteFrame, ...], start: int, end: int) -> tuple[LateralRouteFrame, ...]:
  factor = rng.uniform(0.9, 1.1)
  out: list[LateralRouteFrame] = []
  for i, frame in enumerate(frames):
    if start <= i < end:
      k = frame.raw_curvature * factor
      path = _coherent_path(k, frame.v_ego)
      out.append(frame.__class__(
        t=frame.t,
        v_ego=frame.v_ego,
        lat_active=frame.lat_active,
        raw_curvature=k,
        measured_curvature=frame.measured_curvature,
        roll=frame.roll,
        steering_pressed=frame.steering_pressed,
        left_blinker=frame.left_blinker,
        right_blinker=frame.right_blinker,
        lane_change_state=frame.lane_change_state,
        lane_change_direction=frame.lane_change_direction,
        position_x=path["position_x"],
        position_y=path["position_y"],
        position_y_std=frame.position_y_std,
        orientation_z=path["orientation_z"],
        orientation_rate_z=path["orientation_rate_z"],
        lane_line_probs=frame.lane_line_probs,
        frame_drop_perc=frame.frame_drop_perc,
      ))
    else:
      out.append(frame)
  return tuple(out)


def _apply_offset(rng: random.Random, frames: tuple[LateralRouteFrame, ...], start: int, end: int) -> tuple[LateralRouteFrame, ...]:
  offset = rng.choice([-1.0, 1.0]) * rng.uniform(2e-4, 6e-4)
  out: list[LateralRouteFrame] = []
  for i, frame in enumerate(frames):
    if start <= i < end:
      k = frame.raw_curvature + offset
      path = _coherent_path(k, frame.v_ego)
      out.append(frame.__class__(
        t=frame.t,
        v_ego=frame.v_ego,
        lat_active=frame.lat_active,
        raw_curvature=k,
        measured_curvature=frame.measured_curvature,
        roll=frame.roll,
        steering_pressed=frame.steering_pressed,
        left_blinker=frame.left_blinker,
        right_blinker=frame.right_blinker,
        lane_change_state=frame.lane_change_state,
        lane_change_direction=frame.lane_change_direction,
        position_x=path["position_x"],
        position_y=path["position_y"],
        position_y_std=frame.position_y_std,
        orientation_z=path["orientation_z"],
        orientation_rate_z=path["orientation_rate_z"],
        lane_line_probs=frame.lane_line_probs,
        frame_drop_perc=frame.frame_drop_perc,
      ))
    else:
      out.append(frame)
  return tuple(out)


def _apply_recipe(recipe: PerturbationRecipe, frames: tuple[LateralRouteFrame, ...]) -> tuple[LateralRouteFrame, ...]:
  if recipe.kind == "none":
    return frames
  start = max(0, recipe.start_frame)
  end = min(len(frames), recipe.end_frame)
  if end <= start:
    return frames
  if recipe.kind == "noise":
    rng = random.Random(recipe.params.get("noise_seed", 0))
    return _apply_noise(rng, frames, start, end)
  if recipe.kind == "dropout":
    return _apply_dropout(frames, start, end)
  if recipe.kind == "delay":
    return _apply_delay(frames, start, end, int(recipe.params.get("delay_frames", 3)))
  if recipe.kind == "stale":
    return _apply_stale(frames, start, end)
  if recipe.kind == "scale":
    rng = random.Random(recipe.params.get("scale_seed", 0))
    return _apply_scale(rng, frames, start, end)
  if recipe.kind == "offset":
    rng = random.Random(recipe.params.get("offset_seed", 0))
    return _apply_offset(rng, frames, start, end)
  raise ValueError(f"unknown perturbation kind {recipe.kind!r}")


def _generate_recipe(rng: random.Random, n: int, kind: str | None) -> PerturbationRecipe:
  kind = kind or rng.choice(PERTURBATION_KINDS)
  if kind == "none":
    return PerturbationRecipe(kind="none", start_frame=0, end_frame=0, description="no perturbation")
  window = max(5, int(n * rng.uniform(0.15, 0.35)))
  start = rng.randint(0, max(0, n - window - 1))
  end = min(n, start + window)
  params: dict[str, Any]
  if kind == "noise":
    params = {"noise_seed": rng.randint(0, 1_000_000)}
    desc = f"coherent curvature noise frames {start}-{end}"
  elif kind == "dropout":
    params = {}
    desc = f"invalid path dropout frames {start}-{end}"
  elif kind == "delay":
    if start < 2:
      start = min(max(2, start), max(0, n - window - 1))
      end = min(n, start + window)
    max_delay = max(1, min(10, start))
    params = {"delay_frames": rng.randint(1, max_delay)}
    desc = f"causal delay {params['delay_frames']} frames {start}-{end}"
  elif kind == "stale":
    params = {}
    desc = f"stale curvature freeze frames {start}-{end}"
  elif kind == "scale":
    params = {"scale_seed": rng.randint(0, 1_000_000)}
    desc = f"coherent scale factor frames {start}-{end}"
  elif kind == "offset":
    params = {"offset_seed": rng.randint(0, 1_000_000)}
    desc = f"coherent curvature offset frames {start}-{end}"
  else:
    raise ValueError(f"unknown perturbation kind {kind!r}")
  return PerturbationRecipe(kind=kind, start_frame=start, end_frame=end, description=desc, params=params)


# ---------- scenario generation ----------


@dataclass
class RouteReplayFuzzerConfig:
  seed: int = 1
  cases: int = 100
  preset: str | None = None
  perturbation: str | None = None
  duration_s: float = 2.0


def generate_scenarios(config: RouteReplayFuzzerConfig) -> list[RouteReplayScenario]:
  rng = random.Random(config.seed)
  presets = [config.preset] if config.preset else list(PRESETS)
  scenarios: list[RouteReplayScenario] = []
  for idx in range(config.cases):
    preset = rng.choice(presets)
    v_ego = rng.uniform(15.0, 25.0)
    curvature_scale = rng.uniform(0.0005, 0.0030) * rng.choice([-1.0, 1.0])
    baseline = _generate_preset_frames(preset, config.duration_s, v_ego, curvature_scale)
    recipe = _generate_recipe(rng, len(baseline), config.perturbation)
    perturbed = _apply_recipe(recipe, baseline)
    title = f"route replay {preset} with {recipe.kind} #{idx}"
    scenarios.append(RouteReplayScenario(preset=preset, title=title, frames=baseline, recipe=recipe, perturbed_frames=perturbed))
  return scenarios


# ---------- runner / evaluation ----------


def _run_frames(frames: tuple[LateralRouteFrame, ...]) -> tuple[RouteFrameOutput, ...]:
  pipeline = LateralDemandPipeline(dt=DT)
  outputs: list[RouteFrameOutput] = []
  for frame in frames:
    result = pipeline.update(_frame_to_inputs(frame))
    outputs.append(_output_from_result(frame, result))
  return tuple(outputs)


def _lat_accel(v_ego: float, curvature: float) -> float:
  return (v_ego * v_ego) * curvature


def _validate_input_frames(frames: tuple[LateralRouteFrame, ...], label: str) -> list[dict[str, Any]]:
  failures: list[dict[str, Any]] = []
  for i, frame in enumerate(frames):
    if not math.isfinite(frame.t) or not math.isfinite(frame.raw_curvature) or not math.isfinite(frame.measured_curvature) or not math.isfinite(frame.v_ego):
      failures.append({"check": "input_finite", "detail": f"{label}: frame {i} has non-finite curvature/v_ego"})
    if i > 0:
      dt = frame.t - frames[i - 1].t
      if dt <= 0.0 or abs(dt - DT) > 1e-6:
        failures.append({"check": "input_timing", "detail": f"{label}: frame {i} dt={dt:.6f}s, expected {DT:.6f}s"})
        break
  return failures


def _evaluate_outputs(outputs: tuple[RouteFrameOutput, ...], thresholds: RouteReplayThresholds, label: str) -> list[dict[str, Any]]:
  failures: list[dict[str, Any]] = []
  if not outputs:
    failures.append({"check": "output", "detail": f"{label}: produced no output frames"})
    return failures

  processed = np.array([o.processed_curvature for o in outputs], dtype=float)
  raw = np.array([o.raw_curvature for o in outputs], dtype=float)
  measured = np.array([o.measured_curvature for o in outputs], dtype=float)
  quality = np.array([o.path_quality for o in outputs], dtype=float)
  v_ego = np.array([o.v_ego for o in outputs], dtype=float)

  if not (np.all(np.isfinite(processed)) and np.all(np.isfinite(raw)) and np.all(np.isfinite(measured)) and np.all(np.isfinite(quality))):
    failures.append({"check": "finite", "detail": f"{label}: nonfinite raw/processed/measured/quality output"})

  min_q = float(np.min(quality))
  max_q = float(np.max(quality))
  if min_q < thresholds.path_quality_min - 1e-6 or max_q > thresholds.path_quality_max + 1e-6:
    failures.append({"check": "path_quality_range", "detail": f"{label}: path_quality {min_q:.3f}..{max_q:.3f} outside [0,1]"})

  max_abs_k = float(np.max(np.abs(processed)))
  if max_abs_k > thresholds.max_abs_processed_curvature:
    failures.append({"check": "curvature_cap", "detail": f"{label}: processed curvature |{max_abs_k:.3f}| exceeds {thresholds.max_abs_processed_curvature}"})

  if len(processed) > 1:
    lat_accel = _lat_accel(v_ego, processed)
    steps = np.diff(lat_accel)
    dt = DT
    jerks = steps / dt
    max_step = float(np.max(np.abs(steps)))
    max_jerk = float(np.max(np.abs(jerks)))
    if max_step > thresholds.max_abs_step_lat_accel:
      failures.append({"check": "lat_accel_step", "detail": f"{label}: lateral accel step {max_step:.2f} m/s^2 exceeds {thresholds.max_abs_step_lat_accel}"})
    if max_jerk > thresholds.max_abs_lat_jerk:
      failures.append({"check": "lat_jerk", "detail": f"{label}: lateral jerk {max_jerk:.2f} m/s^3 exceeds {thresholds.max_abs_lat_jerk}"})

  return failures


def evaluate_scenario(scenario: RouteReplayScenario) -> RouteReplayResult:
  thresholds = scenario.metric_thresholds
  baseline_failures: list[dict[str, Any]] = _validate_input_frames(scenario.frames, "baseline")
  perturbation_failures: list[dict[str, Any]] = []
  comparison_failures: list[dict[str, Any]] = []
  metrics: dict[str, Any] = {}

  try:
    baseline_outputs = _run_frames(scenario.frames) if not baseline_failures else ()
  except Exception as exc:
    baseline_failures.append({"check": "exception", "detail": f"baseline raised {type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
    baseline_outputs = ()

  baseline_failures.extend(_evaluate_outputs(baseline_outputs, thresholds, "baseline"))

  try:
    perturbed_frames = scenario.perturbed_frames or _apply_recipe(scenario.recipe, scenario.frames)
    perturbation_failures.extend(_validate_input_frames(perturbed_frames, "perturbed"))
    perturbed_outputs = _run_frames(perturbed_frames) if not perturbation_failures else ()
  except Exception as exc:
    perturbation_failures.append({"check": "exception", "detail": f"perturbed raised {type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
    perturbed_outputs = ()

  if perturbed_outputs:
    perturbation_failures.extend(_evaluate_outputs(perturbed_outputs, thresholds, "perturbed"))

  if baseline_failures:
    comparison_failures.append({
      "check": "comparison_skipped",
      "detail": "baseline failed; perturbed comparison skipped",
    })
  elif baseline_outputs and perturbed_outputs:
    if len(baseline_outputs) != len(perturbed_outputs):
      comparison_failures.append({
        "check": "comparison_length",
        "detail": f"baseline output length {len(baseline_outputs)} != perturbed output length {len(perturbed_outputs)}",
      })
      return RouteReplayResult(
        scenario=scenario,
        baseline_outputs=baseline_outputs,
        perturbed_outputs=perturbed_outputs,
        baseline_failures=baseline_failures,
        perturbation_failures=perturbation_failures,
        comparison_failures=comparison_failures,
        metrics=metrics,
      )

    v_ego = np.array([o.v_ego for o in baseline_outputs], dtype=float)
    base_lat_accel = _lat_accel(v_ego, np.array([o.processed_curvature for o in baseline_outputs]))
    pert_lat_accel = _lat_accel(v_ego, np.array([o.processed_curvature for o in perturbed_outputs]))
    delta = np.abs(base_lat_accel - pert_lat_accel)
    max_delta = float(np.max(delta))
    metrics["max_baseline_perturbed_lat_accel_delta"] = max_delta
    if max_delta > thresholds.max_baseline_perturbed_lat_accel_delta:
      comparison_failures.append({
        "check": "perturbed_divergence",
        "detail": f"perturbed diverged from baseline by {max_delta:.2f} m/s^2 lateral accel",
      })

    base_flips = _sign_flip_count(base_lat_accel, thresholds.oscillation_lat_accel_eps)
    pert_flips = _sign_flip_count(pert_lat_accel, thresholds.oscillation_lat_accel_eps)
    metrics["baseline_sign_flips"] = base_flips
    metrics["perturbed_sign_flips"] = pert_flips
    metrics["oscillation_lat_accel_eps"] = thresholds.oscillation_lat_accel_eps
    if pert_flips > base_flips + thresholds.max_extra_sign_flips:
      comparison_failures.append({
        "check": "oscillation",
        "detail": f"perturbed sign flips {pert_flips} exceeds baseline {base_flips} by more than {thresholds.max_extra_sign_flips}",
      })

    # Targeted dropout check.
    if scenario.recipe.kind == "dropout":
      reasons = {o.path_reason for o in perturbed_outputs[scenario.recipe.start_frame:scenario.recipe.end_frame]}
      if not ("invalid_path" in reasons or any(o.path_quality < 0.9 for o in perturbed_outputs[scenario.recipe.start_frame:scenario.recipe.end_frame])):
        comparison_failures.append({
          "check": "expected_dropout_reaction",
          "detail": "dropout window did not produce invalid_path or reduced quality",
        })

  return RouteReplayResult(
    scenario=scenario,
    baseline_outputs=baseline_outputs,
    perturbed_outputs=perturbed_outputs,
    baseline_failures=baseline_failures,
    perturbation_failures=perturbation_failures,
    comparison_failures=comparison_failures,
    metrics=metrics,
  )


# ---------- serialization / CLI ----------


def scenario_to_dict(scenario: RouteReplayScenario, seed: int | None = None, index: int | None = None) -> dict[str, Any]:
  payload: dict[str, Any] = {
    "preset": scenario.preset,
    "title": scenario.title,
    "baseline_frames": [frame.to_dict() for frame in scenario.frames],
    "perturbed_frames": [frame.to_dict() for frame in (scenario.perturbed_frames or _apply_recipe(scenario.recipe, scenario.frames))],
    "recipe": scenario.recipe.to_dict(),
    "thresholds": scenario.metric_thresholds.to_dict(),
  }
  if seed is not None:
    payload["seed"] = seed
  if index is not None:
    payload["index"] = index
  return payload


def _sign_flip_count(values: np.ndarray, eps: float) -> int:
  ok = np.isfinite(values) & (np.abs(values) > eps)
  signs = np.sign(values[ok])
  return int(np.sum(signs[1:] != signs[:-1])) if signs.size > 1 else 0


def scenario_from_dict(data: dict[str, Any]) -> RouteReplayScenario:
  return RouteReplayScenario(
    preset=str(data["preset"]),
    title=str(data["title"]),
    frames=tuple(LateralRouteFrame.from_dict(frame) for frame in data["baseline_frames"]),
    recipe=PerturbationRecipe.from_dict(data["recipe"]),
    thresholds=RouteReplayThresholds.from_dict(data.get("thresholds", {})),
    perturbed_frames=tuple(LateralRouteFrame.from_dict(frame) for frame in data.get("perturbed_frames", ())),
  )


def scenario_summary_to_dict(scenario: RouteReplayScenario, seed: int | None = None, index: int | None = None) -> dict[str, Any]:
  payload: dict[str, Any] = {
    "preset": scenario.preset,
    "title": scenario.title,
    "frames": len(scenario.frames),
    "perturbation": scenario.recipe.kind,
    "recipe_window": [scenario.recipe.start_frame, scenario.recipe.end_frame],
  }
  if seed is not None:
    payload["seed"] = seed
  if index is not None:
    payload["index"] = index
  return payload


def artifact_to_dict(result: RouteReplayResult, seed: int | None, index: int | None) -> dict[str, Any]:
  perturbed_frames = result.scenario.perturbed_frames or _apply_recipe(result.scenario.recipe, result.scenario.frames)
  return {
    "schema": ARTIFACT_SCHEMA,
    "version": ARTIFACT_VERSION,
    "seed": seed,
    "index": index,
    "preset": result.scenario.preset,
    "title": result.scenario.title,
    "recipe": result.scenario.recipe.to_dict(),
    "thresholds": result.scenario.metric_thresholds.to_dict(),
    "baseline_frames": [frame.to_dict() for frame in result.scenario.frames],
    "perturbed_frames": [frame.to_dict() for frame in perturbed_frames],
    "baseline_summary": {
      "valid": not result.baseline_failures,
      "failure_checks": [f["check"] for f in result.baseline_failures],
    },
    "perturbed_summary": {
      "valid": not result.perturbation_failures,
      "failure_checks": [f["check"] for f in result.perturbation_failures],
    },
    "comparison_summary": {
      "valid": not result.comparison_failures,
      "failure_checks": [f["check"] for f in result.comparison_failures],
    },
    "metrics": _sanitize(result.metrics),
    "overall_valid": result.valid,
  }


def write_artifact(result: RouteReplayResult, artifact_dir: Path, seed: int | None, index: int | None) -> Path:
  artifact_dir.mkdir(parents=True, exist_ok=True)
  filename = f"lateral_route_replay_failure_{result.scenario.preset}_{result.scenario.recipe.kind}_seed{seed}_idx{index}.json"
  path = artifact_dir / filename
  path.write_text(json.dumps(_sanitize(artifact_to_dict(result, seed, index)), indent=2, sort_keys=True, allow_nan=False))
  return path


def load_artifact(path: str | Path) -> dict[str, Any]:
  return json.loads(Path(path).read_text())


def replay_artifact(path: str | Path) -> RouteReplayResult:
  data = load_artifact(path)
  scenario = scenario_from_dict(data)
  return evaluate_scenario(scenario)


def _render_scenario_snippet(scenario: RouteReplayScenario) -> str:
  return f"# preset: {scenario.preset} perturbation: {scenario.recipe.kind}\nRouteReplayScenario(title={scenario.title!r}, frames=[...{len(scenario.frames)} frames...])"


def main() -> None:
  parser = argparse.ArgumentParser(description="Synthetic / serialized route-like lateral replay perturbation fuzzer.")
  parser.add_argument("--seed", type=int, default=1)
  parser.add_argument("--cases", type=int, default=100)
  parser.add_argument("--preset", choices=PRESETS, help="Route preset")
  parser.add_argument("--perturbation", choices=CLI_PERTURBATION_KINDS, help="Perturbation kind (default random)")
  parser.add_argument("--duration", type=float, default=2.0, help="Scenario duration in seconds")
  parser.add_argument("--json", action="store_true", help="Emit JSON instead of text")
  parser.add_argument("--fail-fast", action="store_true", help="Stop after the first failure")
  parser.add_argument("--artifact-dir", type=str, default=None, help="Directory to write failure artifacts")
  parser.add_argument("--replay", type=str, default=None, help="Replay a route-replay artifact JSON file")
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
      print(f"Replayed {args.replay}: valid={result.valid} baseline_failures={len(result.baseline_failures)} perturbation_failures={len(result.perturbation_failures)} comparison_failures={len(result.comparison_failures)}")
      for failure in result.baseline_failures:
        print(f"  baseline: {failure['check']}: {failure['detail']}")
      for failure in result.perturbation_failures:
        print(f"  perturbed: {failure['check']}: {failure['detail']}")
      for failure in result.comparison_failures:
        print(f"  comparison: {failure['check']}: {failure['detail']}")
    sys.exit(0 if result.valid else 1)

  config = RouteReplayFuzzerConfig(seed=args.seed, cases=args.cases, preset=args.preset, perturbation=args.perturbation, duration_s=args.duration)
  scenarios = list(generate_scenarios(config))
  results: list[tuple[int, RouteReplayResult]] = []
  for idx, scenario in enumerate(scenarios):
    result = evaluate_scenario(scenario)
    results.append((idx, result))
    if not result.valid and args.fail_fast:
      break

  failures = [(idx, result) for idx, result in results if not result.valid]
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
      "preset": args.preset,
      "perturbation": args.perturbation,
      "duration": args.duration,
      "dt": DT,
      "failures": [
        {
          "scenario": scenario_summary_to_dict(result.scenario, seed=args.seed, index=result_idx),
          "artifact_hint": "rerun with --artifact-dir for full baseline/perturbed replay frames",
          "baseline_checks": [f["check"] for f in result.baseline_failures],
          "perturbed_checks": [f["check"] for f in result.perturbation_failures],
          "comparison_checks": [f["check"] for f in result.comparison_failures],
        }
        for result_idx, result in failures
      ],
    }
    print(json.dumps(_sanitize(payload), indent=2, sort_keys=True, allow_nan=False))
  else:
    print(
      f"Drive Lab lateral route replay fuzz seed={args.seed} cases={len(results)} "
      f"preset={args.preset or 'all'} perturbation={args.perturbation or 'random'} "
      f"duration={args.duration}s dt={DT}s failures={len(failures)}"
    )
    for idx, result in failures[:10]:
      print(f"\nFAILED: {result.scenario.title}")
      for failure in result.baseline_failures:
        print(f"  baseline: {failure['check']}: {failure['detail']}")
      for failure in result.perturbation_failures:
        print(f"  perturbed: {failure['check']}: {failure['detail']}")
      for failure in result.comparison_failures:
        print(f"  comparison: {failure['check']}: {failure['detail']}")
      print(_render_scenario_snippet(result.scenario))
    if artifact_paths:
      print(f"\nWrote {len(artifact_paths)} failure artifact(s) to {args.artifact_dir}")

  if failures:
    raise SystemExit(1)


if __name__ == "__main__":
  main()
