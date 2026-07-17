#!/usr/bin/env python3
"""Synthetic transition structural fuzzer for LateralDemandPipeline.

This tool generates deterministic synthetic transition scenarios (lat_active toggles,
driver override pulses, lane-change sessions, gating recovery, and explicit
control-limit / demand-jitter pulses) and replays them through
``LateralDemandPipeline``. It is a structural fuzzer for demand behavior, not a
proof of production lateral control. Each scenario stores its full frame sequence
so replay needs no RNG.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cereal import log
from openpilot.sunnypilot.custom.lateral.demand.pipeline import (
  LateralDemandPipeline,
  LateralDemandPipelineInputs,
)
from openpilot.sunnypilot.custom.lateral.demand.types import (
  DEMAND_SOURCE_FALLBACK_MEASURED,
)
from openpilot.tools.drive_lab.lateral_scenarios import (
  LATERAL_PRESETS,
  LateralPresetRequest,  # noqa: F401
  generate_preset_scenarios,  # noqa: F401
)


ARTIFACT_SCHEMA = "drive-lab-lateral-transition-fuzzer-artifact"
ARTIFACT_VERSION = 1
DT = 0.01
N_PATH_POINTS = 33
DEFAULT_KINDS = (
  "clean_baseline",
  "lat_active_toggle",
  "driver_override_pulse",
  "lane_change_session",
  "gating_recovery",
)
EXPLICIT_KINDS = (
  "control_limit_flag",
  "model_demand_jitter_pulse",
)
KINDS = DEFAULT_KINDS + EXPLICIT_KINDS
LANE_CHANGE_STATE_OFF = int(log.LaneChangeState.off)
LANE_CHANGE_STATE_STARTING = int(log.LaneChangeState.laneChangeStarting)
LANE_CHANGE_STATE_FINISHING = int(log.LaneChangeState.laneChangeFinishing)
LANE_CHANGE_DIRECTION_LEFT = int(log.LaneChangeDirection.left)
LANE_CHANGE_DIRECTION_RIGHT = int(log.LaneChangeDirection.right)


@dataclass(frozen=True)
class TransitionFrame:
  """One synthetic transition frame with all inputs needed by the pipeline."""

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
  left_lane_y0: float | None = None
  right_lane_y0: float | None = None
  lateral_maneuver_curvature: float | None = None
  smooth_model_path_curvature: bool = False
  lane_centering_assist_enabled: bool = False
  curve_memory_enabled: bool = False
  curvature_limited: bool = False
  turn_direction: int = 0
  model_data_v2_sp_valid: bool = True

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
      "left_lane_y0": self.left_lane_y0,
      "right_lane_y0": self.right_lane_y0,
      "lateral_maneuver_curvature": self.lateral_maneuver_curvature,
      "smooth_model_path_curvature": self.smooth_model_path_curvature,
      "lane_centering_assist_enabled": self.lane_centering_assist_enabled,
      "curve_memory_enabled": self.curve_memory_enabled,
      "curvature_limited": self.curvature_limited,
      "turn_direction": self.turn_direction,
      "model_data_v2_sp_valid": self.model_data_v2_sp_valid,
    }

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> TransitionFrame:
    def _tuple_floats(key: str) -> tuple[float, ...]:
      return tuple(float(v) for v in data[key])
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
      position_x=_tuple_floats("position_x"),
      position_y=_tuple_floats("position_y"),
      position_y_std=_tuple_floats("position_y_std"),
      orientation_z=_tuple_floats("orientation_z"),
      orientation_rate_z=_tuple_floats("orientation_rate_z"),
      lane_line_probs=tuple(float(v) for v in data["lane_line_probs"]),
      frame_drop_perc=float(data["frame_drop_perc"]),
      left_lane_y0=float(data["left_lane_y0"]) if data.get("left_lane_y0") is not None else None,
      right_lane_y0=float(data["right_lane_y0"]) if data.get("right_lane_y0") is not None else None,
      lateral_maneuver_curvature=float(data["lateral_maneuver_curvature"]) if data.get("lateral_maneuver_curvature") is not None else None,
      smooth_model_path_curvature=bool(data.get("smooth_model_path_curvature", False)),
      lane_centering_assist_enabled=bool(data.get("lane_centering_assist_enabled", False)),
      curve_memory_enabled=bool(data.get("curve_memory_enabled", False)),
      curvature_limited=bool(data.get("curvature_limited", False)),
      turn_direction=int(data.get("turn_direction", 0)),
      model_data_v2_sp_valid=bool(data.get("model_data_v2_sp_valid", True)),
    )


@dataclass(frozen=True)
class TransitionThresholds:
  """Thresholds for structural and event checks."""

  max_abs_processed_curvature: float = 0.5
  max_abs_step_lat_accel: float = 5.0
  max_abs_lat_jerk: float = 150.0
  path_quality_min: float = 0.0
  path_quality_max: float = 1.0
  max_lane_change_blend: float = 1.0

  def to_dict(self) -> dict[str, Any]:
    return {
      "max_abs_processed_curvature": self.max_abs_processed_curvature,
      "max_abs_step_lat_accel": self.max_abs_step_lat_accel,
      "max_abs_lat_jerk": self.max_abs_lat_jerk,
      "path_quality_min": self.path_quality_min,
      "path_quality_max": self.path_quality_max,
      "max_lane_change_blend": self.max_lane_change_blend,
    }

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> TransitionThresholds:
    return cls(**{k: data[k] for k in cls.__dataclass_fields__ if k in data})


@dataclass(frozen=True)
class TransitionScenario:
  """A transition scenario with its event windows (for permitted discontinuities)."""

  kind: str
  title: str
  index: int
  frames: tuple[TransitionFrame, ...]
  event_windows: tuple[tuple[int, int], ...]
  thresholds: TransitionThresholds | None = None

  @property
  def metric_thresholds(self) -> TransitionThresholds:
    return self.thresholds or TransitionThresholds()


@dataclass(frozen=True)
class TransitionFrameOutput:
  """Per-frame pipeline output."""

  t: float
  v_ego: float
  raw_curvature: float
  processed_curvature: float
  measured_curvature: float
  desired_lateral_accel: float
  actual_lateral_accel: float
  path_quality: float
  path_reason: str
  gated: bool
  demand_source: str
  lane_change_shaping_active: bool
  lane_change_blend: float


@dataclass(frozen=True)
class TransitionResult:
  scenario: TransitionScenario
  outputs: tuple[TransitionFrameOutput, ...]
  structural_failures: list[dict[str, Any]]
  event_failures: list[dict[str, Any]]
  metrics: dict[str, Any]

  @property
  def valid(self) -> bool:
    return not (self.structural_failures or self.event_failures)


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


def _base_frame(
  t: float,
  v_ego: float,
  curvature: float,
  measured_curvature: float | None = None,
  lat_active: bool = True,
  steering_pressed: bool = False,
  left_blinker: bool = False,
  right_blinker: bool = False,
  lane_change_state: int = LANE_CHANGE_STATE_OFF,
  lane_change_direction: int = 0,
  lane_line_probs: tuple[float, ...] = (0.9, 0.9, 0.9, 0.9),
  path_curvature: float | None = None,
  curvature_limited: bool = False,
) -> TransitionFrame:
  k = float(curvature)
  mk = float(measured_curvature) if measured_curvature is not None else k
  path = _coherent_path(k if path_curvature is None else float(path_curvature), v_ego)
  return TransitionFrame(
    t=float(t),
    v_ego=float(v_ego),
    lat_active=bool(lat_active),
    raw_curvature=k,
    measured_curvature=mk,
    roll=0.0,
    steering_pressed=bool(steering_pressed),
    left_blinker=bool(left_blinker),
    right_blinker=bool(right_blinker),
    lane_change_state=int(lane_change_state),
    lane_change_direction=int(lane_change_direction),
    position_x=path["position_x"],
    position_y=path["position_y"],
    position_y_std=path["position_y_std"],
    orientation_z=path["orientation_z"],
    orientation_rate_z=path["orientation_rate_z"],
    lane_line_probs=tuple(float(p) for p in lane_line_probs),
    frame_drop_perc=0.0,
    left_lane_y0=-1.8,
    right_lane_y0=1.8,
    smooth_model_path_curvature=False,
    curvature_limited=curvature_limited,
  )


def _frame_to_inputs(frame: TransitionFrame) -> LateralDemandPipelineInputs:
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
    left_lane_y0=frame.left_lane_y0,
    right_lane_y0=frame.right_lane_y0,
    lateral_maneuver_curvature=frame.lateral_maneuver_curvature,
    smooth_model_path_curvature=frame.smooth_model_path_curvature,
    lane_centering_assist_enabled=frame.lane_centering_assist_enabled,
    curve_memory_enabled=frame.curve_memory_enabled,
    curvature_limited=frame.curvature_limited,
    turn_direction=frame.turn_direction,
    model_data_v2_sp_valid=frame.model_data_v2_sp_valid,
  )


def _lateral_accel(v_ego: float, curvature: float) -> float:
  return (v_ego * v_ego) * curvature


def _merge_windows(windows: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
  if not windows:
    return ()
  sorted_windows = sorted(windows)
  merged: list[tuple[int, int]] = [sorted_windows[0]]
  for start, end in sorted_windows[1:]:
    prev_start, prev_end = merged[-1]
    if start <= prev_end + 1:
      merged[-1] = (prev_start, max(prev_end, end))
    else:
      merged.append((start, end))
  return tuple(merged)


def _inside_any_window(idx: int, windows: tuple[tuple[int, int], ...]) -> bool:
  return any(start <= idx < end for start, end in windows)


# ---------- scenario generators ----------


def _generate_clean_baseline(
  rng: random.Random,
  duration_s: float,
  v_ego: float,
  curvature: float,
  index: int,
) -> TransitionScenario:
  t = _time_array(duration_s)
  frames = tuple(_base_frame(float(ti), v_ego, curvature) for ti in t)
  return TransitionScenario(
    kind="clean_baseline",
    title=f"clean baseline #{index}",
    index=index,
    frames=frames,
    event_windows=(),
  )


def _generate_lat_active_toggle(
  rng: random.Random,
  duration_s: float,
  v_ego: float,
  curvature: float,
  index: int,
) -> TransitionScenario:
  t = _time_array(duration_s)
  n = len(t)
  off_start = rng.randint(int(0.25 * n), int(0.45 * n))
  off_end = rng.randint(int(0.55 * n), int(0.75 * n))
  frames: list[TransitionFrame] = []
  for i, ti in enumerate(t):
    lat_active = not (off_start <= i < off_end)
    frames.append(_base_frame(float(ti), v_ego, curvature, lat_active=lat_active))
  return TransitionScenario(
    kind="lat_active_toggle",
    title=f"lat active toggle #{index}",
    index=index,
    frames=tuple(frames),
    event_windows=_merge_windows([(max(0, off_start - 5), min(n, off_end + 5))]),
  )


def _generate_driver_override_pulse(
  rng: random.Random,
  duration_s: float,
  v_ego: float,
  curvature: float,
  index: int,
) -> TransitionScenario:
  t = _time_array(duration_s)
  n = len(t)
  press_start = rng.randint(int(0.25 * n), int(0.45 * n))
  press_end = rng.randint(int(0.55 * n), int(0.75 * n))
  frames: list[TransitionFrame] = []
  for i, ti in enumerate(t):
    steering_pressed = press_start <= i < press_end
    frames.append(_base_frame(float(ti), v_ego, curvature, steering_pressed=steering_pressed))
  return TransitionScenario(
    kind="driver_override_pulse",
    title=f"driver override pulse #{index}",
    index=index,
    frames=tuple(frames),
    event_windows=_merge_windows([(max(0, press_start - 5), min(n, press_end + 10))]),
  )


def _generate_lane_change_session(
  rng: random.Random,
  duration_s: float,
  v_ego: float,
  curvature: float,
  index: int,
) -> TransitionScenario:
  t = _time_array(duration_s)
  n = len(t)
  session_start = rng.randint(int(0.2 * n), int(0.35 * n))
  session_end = rng.randint(int(0.65 * n), int(0.85 * n))
  direction = rng.choice([LANE_CHANGE_DIRECTION_LEFT, LANE_CHANGE_DIRECTION_RIGHT])
  left_blinker = direction == LANE_CHANGE_DIRECTION_LEFT
  right_blinker = direction == LANE_CHANGE_DIRECTION_RIGHT
  # Use a near-straight road so the lane-change path shaper plans and activates.
  # High entry curvature prevents planning because prev_desired_curvature is capped.
  road_curvature = 0.0
  frames: list[TransitionFrame] = []
  for i, ti in enumerate(t):
    if session_start <= i < session_end:
      # Half starting, half finishing to exercise full session.
      mid = (session_start + session_end) // 2
      state = LANE_CHANGE_STATE_STARTING if i < mid else LANE_CHANGE_STATE_FINISHING
      frames.append(_base_frame(
        float(ti), v_ego, road_curvature,
        left_blinker=left_blinker,
        right_blinker=right_blinker,
        lane_change_state=state,
        lane_change_direction=direction,
      ))
    else:
      frames.append(_base_frame(float(ti), v_ego, road_curvature))
  return TransitionScenario(
    kind="lane_change_session",
    title=f"lane change session #{index}",
    index=index,
    frames=tuple(frames),
    event_windows=_merge_windows([(max(0, session_start - 5), min(n, session_end + 5))]),
  )


def _generate_gating_recovery(
  rng: random.Random,
  duration_s: float,
  v_ego: float,
  curvature: float,
  index: int,
) -> TransitionScenario:
  t = _time_array(duration_s)
  n = len(t)
  gate_start = rng.randint(int(0.25 * n), int(0.45 * n))
  gate_end = rng.randint(int(0.55 * n), int(0.75 * n))
  frames: list[TransitionFrame] = []
  for i, ti in enumerate(t):
    if gate_start <= i < gate_end:
      # Drop path arrays to force invalid_path / gated.
      base = _base_frame(float(ti), v_ego, curvature)
      frames.append(base.__class__(
        **{
          **base.to_dict(),
          "position_x": (),
          "position_y": (),
          "position_y_std": (),
          "orientation_z": (),
          "orientation_rate_z": (),
        }
      ))
    else:
      frames.append(_base_frame(float(ti), v_ego, curvature))
  return TransitionScenario(
    kind="gating_recovery",
    title=f"gating recovery #{index}",
    index=index,
    frames=tuple(frames),
    event_windows=_merge_windows([(max(0, gate_start - 5), min(n, gate_end + 15))]),
  )


def _generate_control_limit_flag(
  rng: random.Random,
  duration_s: float,
  v_ego: float,
  curvature: float,
  index: int,
) -> TransitionScenario:
  t = _time_array(duration_s)
  n = len(t)
  limit_start = rng.randint(int(0.25 * n), int(0.45 * n))
  limit_end = rng.randint(int(0.55 * n), int(0.75 * n))
  frames: list[TransitionFrame] = []
  for i, ti in enumerate(t):
    curvature_limited = limit_start <= i < limit_end
    frames.append(_base_frame(float(ti), v_ego, curvature, curvature_limited=curvature_limited))
  return TransitionScenario(
    kind="control_limit_flag",
    title=f"control limit flag #{index}",
    index=index,
    frames=tuple(frames),
    event_windows=_merge_windows([(max(0, limit_start - 5), min(n, limit_end + 5))]),
  )


def _generate_model_demand_jitter_pulse(
  rng: random.Random,
  duration_s: float,
  v_ego: float,
  curvature: float,
  index: int,
) -> TransitionScenario:
  t = _time_array(duration_s)
  n = len(t)
  pulse_start = rng.randint(int(0.25 * n), int(0.40 * n))
  pulse_end = rng.randint(int(0.45 * n), int(0.60 * n))
  # Curvature bump mild enough to avoid path gating.
  # At v=20, delta_k=8e-5 -> jerk ~ 400 * 8e-5 / 0.01 = 3.2 m/s^3.
  delta_k = rng.uniform(7e-5, 1.3e-4) * (1.0 if curvature >= 0.0 else -1.0)
  if curvature == 0.0:
    delta_k = abs(delta_k)
  frames: list[TransitionFrame] = []
  for i, ti in enumerate(t):
    k = curvature + (delta_k if pulse_start <= i < pulse_end else 0.0)
    frames.append(_base_frame(float(ti), v_ego, k))
  return TransitionScenario(
    kind="model_demand_jitter_pulse",
    title=f"model demand jitter pulse #{index}",
    index=index,
    frames=tuple(frames),
    event_windows=_merge_windows([(max(0, pulse_start - 5), min(n, pulse_end + 5))]),
  )


_GENERATORS = {
  "clean_baseline": _generate_clean_baseline,
  "lat_active_toggle": _generate_lat_active_toggle,
  "driver_override_pulse": _generate_driver_override_pulse,
  "lane_change_session": _generate_lane_change_session,
  "gating_recovery": _generate_gating_recovery,
  "control_limit_flag": _generate_control_limit_flag,
  "model_demand_jitter_pulse": _generate_model_demand_jitter_pulse,
}


@dataclass
class TransitionFuzzerConfig:
  seed: int = 1
  cases: int = 100
  kind: str | None = None
  duration_s: float = 2.0


def generate_scenarios(config: TransitionFuzzerConfig) -> list[TransitionScenario]:
  rng = random.Random(config.seed)
  kinds = [config.kind] if config.kind else list(DEFAULT_KINDS)
  scenarios: list[TransitionScenario] = []
  for idx in range(config.cases):
    kind = rng.choice(kinds)
    v_ego = rng.uniform(15.0, 25.0)
    curvature = rng.uniform(0.0005, 0.0030) * rng.choice([-1.0, 1.0])
    scenario = _GENERATORS[kind](rng, config.duration_s, v_ego, curvature, idx)
    scenarios.append(scenario)
  return scenarios


# ---------- runner / evaluation ----------


def _run_scenario(scenario: TransitionScenario) -> tuple[TransitionFrameOutput, ...]:
  pipeline = LateralDemandPipeline(dt=DT)
  outputs: list[TransitionFrameOutput] = []
  for frame in scenario.frames:
    pipeline_result = pipeline.update(_frame_to_inputs(frame))
    demand = pipeline_result.demand
    processed_curvature = demand.processed_curvature
    outputs.append(TransitionFrameOutput(
      t=frame.t,
      v_ego=frame.v_ego,
      raw_curvature=demand.raw_curvature,
      processed_curvature=processed_curvature,
      measured_curvature=demand.measured_curvature,
      desired_lateral_accel=_lateral_accel(frame.v_ego, processed_curvature),
      actual_lateral_accel=_lateral_accel(frame.v_ego, frame.measured_curvature),
      path_quality=demand.path_quality,
      path_reason=demand.path_reason,
      gated=pipeline_result.model_path_result.gated,
      demand_source=demand.demand_source,
      lane_change_shaping_active=demand.lane_change_shaping_active,
      lane_change_blend=demand.lane_change_blend,
    ))
  return tuple(outputs)


def _evaluate_structural(
  outputs: tuple[TransitionFrameOutput, ...],
  scenario: TransitionScenario,
) -> list[dict[str, Any]]:
  thresholds = scenario.metric_thresholds
  failures: list[dict[str, Any]] = []
  if not outputs:
    failures.append({"check": "output", "detail": "scenario produced no output frames"})
    return failures

  n = len(outputs)
  raw = np.array([o.raw_curvature for o in outputs], dtype=float)
  processed = np.array([o.processed_curvature for o in outputs], dtype=float)
  measured = np.array([o.measured_curvature for o in outputs], dtype=float)
  quality = np.array([o.path_quality for o in outputs], dtype=float)
  blend = np.array([o.lane_change_blend for o in outputs], dtype=float)
  desired_lat_accel = np.array([o.desired_lateral_accel for o in outputs], dtype=float)
  actual_lat_accel = np.array([o.actual_lateral_accel for o in outputs], dtype=float)

  if not (np.all(np.isfinite(raw)) and np.all(np.isfinite(processed)) and np.all(np.isfinite(measured))):
    failures.append({"check": "finite_curvature", "detail": "non-finite raw/processed/measured curvature"})
  if not (np.all(np.isfinite(desired_lat_accel)) and np.all(np.isfinite(actual_lat_accel))):
    failures.append({"check": "finite_lat_accel", "detail": "non-finite desired/actual lateral accel"})

  min_q = float(np.min(quality))
  max_q = float(np.max(quality))
  if min_q < thresholds.path_quality_min - 1e-6 or max_q > thresholds.path_quality_max + 1e-6:
    failures.append({"check": "path_quality_range", "detail": f"path_quality {min_q:.3f}..{max_q:.3f} outside [0,1]"})

  max_abs_k = float(np.max(np.abs(processed)))
  if max_abs_k > thresholds.max_abs_processed_curvature:
    failures.append({"check": "curvature_cap", "detail": f"processed curvature |{max_abs_k:.3f}| exceeds {thresholds.max_abs_processed_curvature}"})

  min_blend = float(np.min(blend))
  max_blend = float(np.max(blend))
  if min_blend < -1e-6 or max_blend > thresholds.max_lane_change_blend + 1e-6:
    failures.append({"check": "lane_change_blend_range", "detail": f"lane_change_blend {min_blend:.3f}..{max_blend:.3f} outside [0,{thresholds.max_lane_change_blend}]"})

  if n > 1:
    v_ego = np.array([o.v_ego for o in outputs], dtype=float)
    lat_accel = _lateral_accel(v_ego, processed)
    steps = np.diff(lat_accel)
    jerks = steps / DT
    for i in range(1, n):
      if _inside_any_window(i, scenario.event_windows):
        continue
      if abs(steps[i - 1]) > thresholds.max_abs_step_lat_accel:
        failures.append({"check": "lat_accel_step", "detail": f"frame {i} lateral accel step {steps[i - 1]:.2f} outside event window exceeds {thresholds.max_abs_step_lat_accel}"})
      if abs(jerks[i - 1]) > thresholds.max_abs_lat_jerk:
        failures.append({"check": "lat_jerk", "detail": f"frame {i} lateral jerk {jerks[i - 1]:.2f} outside event window exceeds {thresholds.max_abs_lat_jerk}"})

  return failures


def _evaluate_events(
  outputs: tuple[TransitionFrameOutput, ...],
  scenario: TransitionScenario,
) -> list[dict[str, Any]]:
  failures: list[dict[str, Any]] = []
  if not outputs:
    return failures

  kind = scenario.kind
  frames = scenario.frames

  if kind == "clean_baseline":
    gated = [o.gated for o in outputs]
    if any(gated):
      failures.append({"check": "unexpected_gating", "detail": f"clean baseline had {sum(gated)} gated frame(s)"})

  elif kind == "lat_active_toggle":
    inactive_outputs = [o for o, f in zip(outputs, frames) if not f.lat_active]
    if not inactive_outputs:
      failures.append({"check": "missing_inactive", "detail": "lat_active_toggle never produced an inactive frame"})
    elif not all(o.demand_source == DEMAND_SOURCE_FALLBACK_MEASURED for o in inactive_outputs):
      failures.append({"check": "fallback_source", "detail": "not all inactive frames sourced fallback_measured"})
    elif not all(o.path_reason == "inactive" for o in inactive_outputs):
      failures.append({"check": "inactive_reason", "detail": "not all inactive frames reported path_reason=inactive"})

  elif kind == "driver_override_pulse":
    press_outputs = [o for o, f in zip(outputs, frames) if f.steering_pressed]
    if not press_outputs:
      failures.append({"check": "missing_override", "detail": "driver_override_pulse had no steering_pressed frames"})

  elif kind == "lane_change_session":
    lc_outputs = [o for o, f in zip(outputs, frames) if f.lane_change_state != LANE_CHANGE_STATE_OFF]
    if not lc_outputs:
      failures.append({"check": "missing_lane_change", "detail": "lane_change_session had no lane-change frames"})
    if not any(o.lane_change_shaping_active for o in outputs):
      failures.append({"check": "shaping_active", "detail": "lane_change_session never activated lane_change_shaping_active"})

  elif kind == "gating_recovery":
    gate_outputs = [o for o in outputs if o.gated]
    if not gate_outputs:
      failures.append({"check": "missing_gating", "detail": "gating_recovery had no gated frames"})
    post_recovery = outputs[-20:]
    if post_recovery and not any(not o.gated for o in post_recovery):
      failures.append({"check": "recovery_ungated", "detail": "post-recovery frames did not return to ungated"})

  elif kind == "control_limit_flag":
    limited_outputs = [o for o, f in zip(outputs, frames) if f.curvature_limited]
    if not limited_outputs:
      failures.append({"check": "missing_limit", "detail": "control_limit_flag had no curvature_limited frames"})

  elif kind == "model_demand_jitter_pulse":
    base_curvature = frames[0].raw_curvature if frames else 0.0
    pulse_outputs = [o for o, f in zip(outputs, frames) if abs(f.raw_curvature - base_curvature) > 1e-12]
    if not pulse_outputs:
      failures.append({"check": "missing_pulse", "detail": "model_demand_jitter_pulse had no perturbed-curvature frames"})
    if any(o.gated for o in pulse_outputs):
      failures.append({"check": "jitter_not_path_quality", "detail": "model_demand_jitter_pulse tripped path gating instead of passing demand jitter through"})

  return failures


def evaluate_scenario(scenario: TransitionScenario) -> TransitionResult:
  structural_failures: list[dict[str, Any]] = []
  event_failures: list[dict[str, Any]] = []
  metrics: dict[str, Any] = {}

  try:
    outputs = _run_scenario(scenario)
  except Exception as exc:
    structural_failures.append({"check": "exception", "detail": f"scenario raised {type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
    outputs = ()

  structural_failures.extend(_evaluate_structural(outputs, scenario))
  if outputs:
    event_failures.extend(_evaluate_events(outputs, scenario))
    metrics["gated_frames"] = int(sum(o.gated for o in outputs))
    metrics["max_lane_change_blend"] = float(max(o.lane_change_blend for o in outputs))

  return TransitionResult(
    scenario=scenario,
    outputs=outputs,
    structural_failures=structural_failures,
    event_failures=event_failures,
    metrics=_sanitize(metrics),
  )


# ---------- serialization / CLI ----------


def scenario_to_dict(scenario: TransitionScenario, seed: int | None = None) -> dict[str, Any]:
  payload: dict[str, Any] = {
    "kind": scenario.kind,
    "title": scenario.title,
    "index": scenario.index,
    "frames": [frame.to_dict() for frame in scenario.frames],
    "event_windows": list(scenario.event_windows),
    "thresholds": scenario.metric_thresholds.to_dict(),
  }
  if seed is not None:
    payload["seed"] = seed
  return payload


def scenario_from_dict(data: dict[str, Any]) -> TransitionScenario:
  return TransitionScenario(
    kind=str(data["kind"]),
    title=str(data["title"]),
    index=int(data["index"]),
    frames=tuple(TransitionFrame.from_dict(frame) for frame in data["frames"]),
    event_windows=tuple((int(w[0]), int(w[1])) for w in data.get("event_windows", [])),
    thresholds=TransitionThresholds.from_dict(data.get("thresholds", {})),
  )


def artifact_to_dict(result: TransitionResult, seed: int | None, index: int | None) -> dict[str, Any]:
  return {
    "schema": ARTIFACT_SCHEMA,
    "version": ARTIFACT_VERSION,
    "seed": seed,
    "index": index,
    "kind": result.scenario.kind,
    "title": result.scenario.title,
    "thresholds": result.scenario.metric_thresholds.to_dict(),
    "scenario": scenario_to_dict(result.scenario, seed=seed),
    "outputs": [
      {
        "t": o.t,
        "v_ego": o.v_ego,
        "raw_curvature": o.raw_curvature,
        "processed_curvature": o.processed_curvature,
        "measured_curvature": o.measured_curvature,
        "desired_lateral_accel": o.desired_lateral_accel,
        "actual_lateral_accel": o.actual_lateral_accel,
        "path_quality": o.path_quality,
        "path_reason": o.path_reason,
        "gated": o.gated,
        "demand_source": o.demand_source,
        "lane_change_shaping_active": o.lane_change_shaping_active,
        "lane_change_blend": o.lane_change_blend,
      }
      for o in result.outputs
    ],
    "structural_failure_checks": [f["check"] for f in result.structural_failures],
    "event_failure_checks": [f["check"] for f in result.event_failures],
    "metrics": _sanitize(result.metrics),
    "overall_valid": result.valid,
  }


def write_artifact(result: TransitionResult, artifact_dir: Path, seed: int | None, index: int | None) -> Path:
  artifact_dir.mkdir(parents=True, exist_ok=True)
  filename = f"lateral_transition_failure_{result.scenario.kind}_seed{seed}_idx{index}.json"
  path = artifact_dir / filename
  path.write_text(json.dumps(_sanitize(artifact_to_dict(result, seed, index)), indent=2, sort_keys=True, allow_nan=False))
  return path


def load_artifact(path: str | Path) -> dict[str, Any]:
  return json.loads(Path(path).read_text())


def replay_artifact(path: str | Path) -> TransitionResult:
  data = load_artifact(path)
  scenario = scenario_from_dict(data["scenario"])
  return evaluate_scenario(scenario)


def _render_scenario_snippet(scenario: TransitionScenario) -> str:
  return f"# kind: {scenario.kind}\nTransitionScenario(title={scenario.title!r}, frames=[...{len(scenario.frames)} frames...])"


def main() -> None:
  parser = argparse.ArgumentParser(description="Synthetic transition structural fuzzer for LateralDemandPipeline.")
  parser.add_argument("--seed", type=int, default=1)
  parser.add_argument("--cases", type=int, default=100)
  parser.add_argument("--kind", choices=KINDS, help="Transition kind")
  parser.add_argument("--preset", choices=LATERAL_PRESETS, help="Public lateral benchmark preset (fuzz mode only)")
  parser.add_argument("--duration", type=float, default=2.0, help="Scenario duration in seconds")
  parser.add_argument("--json", action="store_true", help="Emit JSON instead of text")
  parser.add_argument("--fail-fast", action="store_true", help="Stop after the first failure")
  parser.add_argument("--artifact-dir", type=str, default=None, help="Directory to write failure artifacts")
  parser.add_argument("--replay", type=str, default=None, help="Replay a transition artifact JSON file")
  args = parser.parse_args()

  if args.preset and args.preset != "fuzz":
    print(f"Preset '{args.preset}' is designed for fuzz_lateral_demand / fuzz_lateral_closed_loop.")
    print("The transition fuzzer currently supports --preset fuzz only.")
    raise SystemExit(1)

  if args.cases < 0:
    parser.error("--cases must be >= 0")
  if args.duration <= 0.0:
    parser.error("--duration must be > 0")

  if args.replay:
    result = replay_artifact(args.replay)
    if args.json:
      print(json.dumps(_sanitize(artifact_to_dict(result, seed=None, index=None)), indent=2, sort_keys=True, allow_nan=False))
    else:
      print(f"Replayed {args.replay}: valid={result.valid} structural={len(result.structural_failures)} event={len(result.event_failures)}")
      for failure in result.structural_failures:
        print(f"  structural: {failure['check']}: {failure['detail']}")
      for failure in result.event_failures:
        print(f"  event: {failure['check']}: {failure['detail']}")
    sys.exit(0 if result.valid else 1)

  config = TransitionFuzzerConfig(seed=args.seed, cases=args.cases, kind=args.kind, duration_s=args.duration)
  scenarios = list(generate_scenarios(config))
  results: list[TransitionResult] = []
  for scenario in scenarios:
    result = evaluate_scenario(scenario)
    results.append(result)
    if not result.valid and args.fail_fast:
      break

  failures = [r for r in results if not r.valid]
  artifact_paths: list[str] = []
  if args.artifact_dir and failures:
    artifact_dir = Path(args.artifact_dir)
    for result in failures:
      path = write_artifact(result, artifact_dir, args.seed, result.scenario.index)
      artifact_paths.append(str(path))

  if args.json:
    payload = {
      "seed": args.seed,
      "cases": len(results),
      "kind": args.kind,
      "preset": args.preset,
      "duration": args.duration,
      "dt": DT,
      "failures": [
        {
          "scenario": scenario_to_dict(result.scenario, seed=args.seed),
          "structural_checks": [f["check"] for f in result.structural_failures],
          "event_checks": [f["check"] for f in result.event_failures],
          "metrics": _sanitize(result.metrics),
        }
        for result in failures
      ],
    }
    print(json.dumps(_sanitize(payload), indent=2, sort_keys=True, allow_nan=False))
  else:
    preset_str = f"preset={args.preset} " if args.preset else ""
    print(
      f"Drive Lab lateral transition fuzz seed={args.seed} cases={len(results)} "
      f"{preset_str}kind={args.kind or 'default'} duration={args.duration}s dt={DT}s failures={len(failures)}"
    )
    for result in failures[:10]:
      print(f"\nFAILED: {result.scenario.title}")
      for failure in result.structural_failures:
        print(f"  structural: {failure['check']}: {failure['detail']}")
      for failure in result.event_failures:
        print(f"  event: {failure['check']}: {failure['detail']}")
      print(_render_scenario_snippet(result.scenario))
    if artifact_paths:
      print(f"\nWrote {len(artifact_paths)} failure artifact(s) to {args.artifact_dir}")

  if failures:
    raise SystemExit(1)


if __name__ == "__main__":
  main()
