#!/usr/bin/env python3
"""Synthetic or route-extracted lateral replay perturbation fuzzer.

This tool defaults to deterministic synthetic lateral "route" sequences. It can
also optionally extract route-derived frames from logs, serialize those frames,
perturb them with causal materialized perturbations, and replay both baseline
and perturbed sequences through ``LateralDemandPipeline``. Route extraction uses
latest available context and fixed-DT normalization; it is not faithful timing
replay or proof of production lateral-control safety. Serialized replay needs no
route file, RNG, or generator behavior.

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
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from openpilot.sunnypilot.custom.lateral.demand.pipeline import (
  LateralDemandPipeline,
  LateralDemandPipelineInputs,
)


ARTIFACT_SCHEMA = "drive-lab-lateral-route-replay-fuzzer-artifact"
ARTIFACT_VERSION = 1
DT = 0.01
N_PATH_POINTS = 33
PRESETS = ("synthetic_straight", "synthetic_curve", "synthetic_sine", "synthetic_reversal")
PERTURBATION_KINDS = ("noise", "dropout", "delay", "stale", "model_age_stale", "model_age_delay", "scale", "offset")
CLI_PERTURBATION_KINDS = PERTURBATION_KINDS + ("none",)
ROUTE_EXTRACTED_PRESET = "route_extracted"
SAMPLE_MODES = ("prefix", "random-window", "uniform-windows")
TIMING_MODES = ("fixed-dt", "original")

# model_age_delay windows are expected to fall back toward measured curvature,
# but the controller may damp aggressively. Measure lateral-accel errors over
# frames where the delayed raw input is materially different from measured.
_MODEL_AGE_MATERIAL_LAT_ACCEL_EPS = 0.02  # m/s^2
_MODEL_AGE_DELAY_MAX_EXTRA_LAT_ACCEL_ERROR = 0.30      # m/s^2
_MODEL_AGE_DELAY_MAX_PROCESSED_LAT_ACCEL_ERROR = 1.00  # m/s^2


@dataclass
class RouteExtractionQuality:
  """Quality/context counters for route extraction."""

  input_message_count: int = 0
  controls_state_seen: int = 0
  controls_state_in_window: int = 0
  skipped_missing_car_state: int = 0
  missing_model_v2: int = 0
  missing_car_control: int = 0
  missing_live_parameters: int = 0
  empty_position_path: int = 0
  empty_orientation: int = 0
  empty_orientation_rate: int = 0
  empty_lane_line_probs: int = 0

  def to_dict(self) -> dict[str, Any]:
    return {
      "input_message_count": self.input_message_count,
      "controls_state_seen": self.controls_state_seen,
      "controls_state_in_window": self.controls_state_in_window,
      "skipped_missing_car_state": self.skipped_missing_car_state,
      "missing_model_v2": self.missing_model_v2,
      "missing_car_control": self.missing_car_control,
      "missing_live_parameters": self.missing_live_parameters,
      "empty_position_path": self.empty_position_path,
      "empty_orientation": self.empty_orientation,
      "empty_orientation_rate": self.empty_orientation_rate,
      "empty_lane_line_probs": self.empty_lane_line_probs,
    }

  @classmethod
  def from_dict(cls, data: dict[str, Any] | None) -> RouteExtractionQuality | None:
    if data is None:
      return None
    return cls(
      input_message_count=int(data.get("input_message_count", 0)),
      controls_state_seen=int(data.get("controls_state_seen", 0)),
      controls_state_in_window=int(data.get("controls_state_in_window", 0)),
      skipped_missing_car_state=int(data.get("skipped_missing_car_state", 0)),
      missing_model_v2=int(data.get("missing_model_v2", 0)),
      missing_car_control=int(data.get("missing_car_control", 0)),
      missing_live_parameters=int(data.get("missing_live_parameters", 0)),
      empty_position_path=int(data.get("empty_position_path", 0)),
      empty_orientation=int(data.get("empty_orientation", 0)),
      empty_orientation_rate=int(data.get("empty_orientation_rate", 0)),
      empty_lane_line_probs=int(data.get("empty_lane_line_probs", 0)),
    )


@dataclass(frozen=True)
class RouteExtractionSummary:
  """Metadata describing how route-derived frames were extracted."""

  route: str | None
  qlog: bool
  window_start_s: float | None
  window_end_s: float | None
  max_frames: int | None
  extracted_count: int
  original_time_span_s: float | None
  dt: float
  quality: RouteExtractionQuality | None = None
  quality_scope: str = "extracted_frames"
  timing_mode: str = "fixed-dt"
  sampling_mode: str | None = None
  sampling_window_index: int | None = None
  sampling_window_count: int | None = None
  sampling_seed: int | None = None
  sampling_window_duration_s: float | None = None
  selected_original_start_s: float | None = None
  selected_original_end_s: float | None = None
  source_dt_stats: dict[str, float] | None = None

  def to_dict(self) -> dict[str, Any]:
    payload: dict[str, Any] = {
      "route": self.route,
      "qlog": self.qlog,
      "window_start_s": self.window_start_s,
      "window_end_s": self.window_end_s,
      "max_frames": self.max_frames,
      "extracted_count": self.extracted_count,
      "original_time_span_s": self.original_time_span_s,
      "dt": self.dt,
      "quality_scope": self.quality_scope,
      "timing_mode": self.timing_mode,
    }
    if self.quality is not None:
      payload["quality"] = self.quality.to_dict()
    if self.sampling_mode is not None:
      payload["sampling_mode"] = self.sampling_mode
    if self.sampling_window_index is not None:
      payload["sampling_window_index"] = self.sampling_window_index
    if self.sampling_window_count is not None:
      payload["sampling_window_count"] = self.sampling_window_count
    if self.sampling_seed is not None:
      payload["sampling_seed"] = self.sampling_seed
    if self.sampling_window_duration_s is not None:
      payload["sampling_window_duration_s"] = self.sampling_window_duration_s
    if self.selected_original_start_s is not None:
      payload["selected_original_start_s"] = self.selected_original_start_s
    if self.selected_original_end_s is not None:
      payload["selected_original_end_s"] = self.selected_original_end_s
    if self.source_dt_stats is not None:
      payload["source_dt_stats"] = self.source_dt_stats
    return payload

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> RouteExtractionSummary:
    return cls(
      route=data.get("route"),
      qlog=bool(data.get("qlog", False)),
      window_start_s=_float_or_none(data.get("window_start_s")),
      window_end_s=_float_or_none(data.get("window_end_s")),
      max_frames=int(data["max_frames"]) if data.get("max_frames") is not None else None,
      extracted_count=int(data.get("extracted_count", 0)),
      original_time_span_s=_float_or_none(data.get("original_time_span_s")),
      dt=float(data.get("dt", DT)),
      quality=RouteExtractionQuality.from_dict(data.get("quality")),
      quality_scope=str(data.get("quality_scope", "extracted_frames")),
      timing_mode=str(data.get("timing_mode", "fixed-dt")),
      sampling_mode=data.get("sampling_mode"),
      sampling_window_index=int(data["sampling_window_index"]) if data.get("sampling_window_index") is not None else None,
      sampling_window_count=int(data["sampling_window_count"]) if data.get("sampling_window_count") is not None else None,
      sampling_seed=int(data["sampling_seed"]) if data.get("sampling_seed") is not None else None,
      sampling_window_duration_s=_float_or_none(data.get("sampling_window_duration_s")),
      selected_original_start_s=_float_or_none(data.get("selected_original_start_s")),
      selected_original_end_s=_float_or_none(data.get("selected_original_end_s")),
      source_dt_stats=data.get("source_dt_stats"),
    )


def _float_or_none(value: Any) -> float | None:
  if value is None:
    return None
  try:
    f = float(value)
    return f if math.isfinite(f) else None
  except (TypeError, ValueError):
    return None


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
  model_age_s: float = 0.0
  source_t: float | None = None

  def to_dict(self) -> dict[str, Any]:
    payload: dict[str, Any] = {
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
      "model_age_s": self.model_age_s,
    }
    if self.source_t is not None:
      payload["source_t"] = self.source_t
    return payload

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
      model_age_s=float(data.get("model_age_s", 0.0)),
      source_t=_float_or_none(data.get("source_t")),
    )


def _copy_frame(frame: LateralRouteFrame, **overrides: Any) -> LateralRouteFrame:
  return replace(frame, **overrides)


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
  route_metadata: RouteExtractionSummary | None = None

  @property
  def metric_thresholds(self) -> RouteReplayThresholds:
    return self.thresholds or RouteReplayThresholds()


@dataclass(frozen=True)
class RouteFrameOutput:
  """Per-frame pipeline output."""

  t: float
  source_t: float | None
  v_ego: float
  raw_curvature: float
  processed_curvature: float
  command_curvature: float
  measured_curvature: float
  path_quality: float
  path_reason: str
  gated: bool
  demand_source: str
  demand_enabled: bool


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


@dataclass(frozen=True)
class LateralDemandABVariant:
  """Measurement for one demand-enabled/disabled replay stream."""

  name: str
  demand_enabled: bool
  frame_count: int
  normalized_start_s: float | None
  normalized_end_s: float | None
  source_start_s: float | None
  source_end_s: float | None
  source_timing_used: bool
  metrics: dict[str, float | int | None]

  def to_dict(self) -> dict[str, Any]:
    return {
      "name": self.name,
      "demand_enabled": self.demand_enabled,
      "frame_count": self.frame_count,
      "normalized_start_s": self.normalized_start_s,
      "normalized_end_s": self.normalized_end_s,
      "source_start_s": self.source_start_s,
      "source_end_s": self.source_end_s,
      "source_timing_used": self.source_timing_used,
      "metrics": _sanitize(self.metrics),
    }


@dataclass(frozen=True)
class LateralDemandABReport:
  """Measurement-only A/B report for one already-extracted route frame window."""

  source: str
  route_metadata: RouteExtractionSummary | None
  variants: dict[str, LateralDemandABVariant]
  deltas_enabled_minus_disabled: dict[str, float | None]
  notes: list[str]

  def to_dict(self) -> dict[str, Any]:
    payload: dict[str, Any] = {
      "source": self.source,
      "variants": {name: variant.to_dict() for name, variant in self.variants.items()},
      "deltas_enabled_minus_disabled": _sanitize(self.deltas_enabled_minus_disabled),
      "notes": list(self.notes),
    }
    if self.route_metadata is not None:
      payload["route_metadata"] = self.route_metadata.to_dict()
    return payload


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
    model_age_s=frame.model_age_s,
    left_blinker=frame.left_blinker,
    right_blinker=frame.right_blinker,
    steering_pressed=frame.steering_pressed,
    lane_change_state=frame.lane_change_state,
    lane_change_direction=frame.lane_change_direction,
  )


def _output_from_result(frame: LateralRouteFrame, pipeline_result: Any, *, demand_enabled: bool = True) -> RouteFrameOutput:
  d = pipeline_result.demand
  processed_curvature = float(d.processed_curvature) if demand_enabled else float(frame.raw_curvature)
  return RouteFrameOutput(
    t=frame.t,
    source_t=frame.source_t,
    v_ego=frame.v_ego,
    raw_curvature=d.raw_curvature,
    processed_curvature=processed_curvature,
    command_curvature=processed_curvature,
    measured_curvature=d.measured_curvature,
    path_quality=d.path_quality,
    path_reason=pipeline_result.model_path_result.reason,
    gated=pipeline_result.model_path_result.gated,
    demand_source=d.demand_source if demand_enabled else "disabled_raw_passthrough",
    demand_enabled=demand_enabled,
  )


# ---------- route extraction ----------


def _safe_float(value: Any, default: float = 0.0) -> float:
  if value is None:
    return default
  try:
    f = float(value)
    return f if math.isfinite(f) else default
  except (TypeError, ValueError):
    return default


def _safe_bool(value: Any, default: bool = False) -> bool:
  if value is None:
    return default
  return bool(value)


def _safe_int(value: Any, default: int = 0) -> int:
  if value is None:
    return default
  try:
    return int(value)
  except (TypeError, ValueError):
    return default


def _as_tuple(values: Any, n: int | None = None) -> tuple[float, ...]:
  if values is None:
    return ()
  try:
    iterable = list(values)
  except TypeError:
    return ()
  out_values: list[float] = []
  for value in iterable:
    try:
      finite = float(value)
    except (TypeError, ValueError):
      return ()
    if not math.isfinite(finite):
      return ()
    out_values.append(finite)
  out = tuple(out_values)
  if n is not None and len(out) != n:
    return ()
  return out


def _extract_raw_curvature(lat_active: bool, state: dict[str, Any]) -> float:
  if not lat_active:
    return _extract_measured_curvature(state)
  model_v2 = state.get("modelV2")
  if model_v2 is not None:
    action = getattr(model_v2, "action", None)
    if action is not None:
      desired = _safe_float(getattr(action, "desiredCurvature", None))
      if desired != 0.0 or getattr(action, "desiredCurvature", None) is not None:
        return desired
  controls_state = state.get("controlsState")
  if controls_state is not None:
    model_path_state = getattr(controls_state, "modelPathState", None)
    if model_path_state is not None:
      raw_desired = _safe_float(getattr(model_path_state, "rawDesiredCurvature", None))
      if raw_desired != 0.0 or getattr(model_path_state, "rawDesiredCurvature", None) is not None:
        return raw_desired
    desired = _safe_float(getattr(controls_state, "desiredCurvature", None))
    if desired != 0.0 or getattr(controls_state, "desiredCurvature", None) is not None:
      return desired
  return _extract_measured_curvature(state)


def _extract_measured_curvature(state: dict[str, Any]) -> float:
  controls_state = state.get("controlsState")
  if controls_state is not None:
    curvature = _safe_float(getattr(controls_state, "curvature", None))
    if curvature != 0.0 or getattr(controls_state, "curvature", None) is not None:
      return curvature
  car_control = state.get("carControl")
  if car_control is not None:
    actuators = getattr(car_control, "actuators", None)
    if actuators is not None:
      return _safe_float(getattr(actuators, "curvature", None))
  return 0.0


def _frame_from_controls_state(t: float, state: dict[str, Any], source_t: float | None = None) -> LateralRouteFrame:
  car_state = state.get("carState")
  car_control = state.get("carControl")
  model_v2 = state.get("modelV2")
  live_parameters = state.get("liveParameters")

  v_ego = _safe_float(getattr(car_state, "vEgo", None) if car_state is not None else None)
  lat_active = _safe_bool(getattr(car_control, "latActive", None) if car_control is not None else None, default=True)
  measured_curvature = _extract_measured_curvature(state)
  raw_curvature = _extract_raw_curvature(lat_active, state)
  roll = _safe_float(getattr(live_parameters, "roll", None) if live_parameters is not None else None)

  steering_pressed = _safe_bool(getattr(car_state, "steeringPressed", None) if car_state is not None else None)
  left_blinker = _safe_bool(getattr(car_state, "leftBlinker", None) if car_state is not None else None)
  right_blinker = _safe_bool(getattr(car_state, "rightBlinker", None) if car_state is not None else None)

  lane_change_state = 0
  lane_change_direction = 0
  if model_v2 is not None:
    meta = getattr(model_v2, "meta", None)
    if meta is not None:
      lane_change_state = _safe_int(getattr(meta, "laneChangeState", None))
      lane_change_direction = _safe_int(getattr(meta, "laneChangeDirection", None))

  position_x = ()
  position_y = ()
  position_y_std = ()
  orientation_z = ()
  orientation_rate_z = ()
  lane_line_probs = ()
  if model_v2 is not None:
    position = getattr(model_v2, "position", None)
    if position is not None:
      position_x = _as_tuple(getattr(position, "x", None), N_PATH_POINTS)
      position_y = _as_tuple(getattr(position, "y", None), N_PATH_POINTS)
      position_y_std = _as_tuple(getattr(position, "yStd", None), N_PATH_POINTS)
    orientation = getattr(model_v2, "orientation", None)
    if orientation is not None:
      orientation_z = _as_tuple(getattr(orientation, "z", None), N_PATH_POINTS)
    orientation_rate = getattr(model_v2, "orientationRate", None)
    if orientation_rate is not None:
      orientation_rate_z = _as_tuple(getattr(orientation_rate, "z", None), N_PATH_POINTS)
    lane_line_probs = _as_tuple(getattr(model_v2, "laneLineProbs", None))

  frame_drop_perc = _safe_float(getattr(model_v2, "frameDropPerc", None) if model_v2 is not None else None)

  return LateralRouteFrame(
    t=t,
    v_ego=v_ego,
    lat_active=lat_active,
    raw_curvature=raw_curvature,
    measured_curvature=measured_curvature,
    roll=roll,
    steering_pressed=steering_pressed,
    left_blinker=left_blinker,
    right_blinker=right_blinker,
    lane_change_state=lane_change_state,
    lane_change_direction=lane_change_direction,
    position_x=position_x,
    position_y=position_y,
    position_y_std=position_y_std,
    orientation_z=orientation_z,
    orientation_rate_z=orientation_rate_z,
    lane_line_probs=lane_line_probs,
    frame_drop_perc=frame_drop_perc,
    source_t=source_t,
  )


def _source_dt_stats(frames: tuple[LateralRouteFrame, ...]) -> dict[str, float] | None:
  times = [frame.source_t for frame in frames if frame.source_t is not None]
  if len(times) < 2:
    return None
  dts = [times[i] - times[i - 1] for i in range(1, len(times))]
  if not dts:
    return None
  return {
    "min": float(np.min(dts)),
    "median": float(np.median(dts)),
    "max": float(np.max(dts)),
  }


def _count_frame_quality(frame: LateralRouteFrame, state: dict[str, Any], quality: RouteExtractionQuality) -> None:
  if state.get("modelV2") is None:
    quality.missing_model_v2 += 1
  else:
    if len(frame.position_x) == 0 or len(frame.position_y) == 0 or len(frame.position_y_std) == 0:
      quality.empty_position_path += 1
    if len(frame.orientation_z) == 0:
      quality.empty_orientation += 1
    if len(frame.orientation_rate_z) == 0:
      quality.empty_orientation_rate += 1
    if len(frame.lane_line_probs) == 0:
      quality.empty_lane_line_probs += 1
  if state.get("carControl") is None:
    quality.missing_car_control += 1
  if state.get("liveParameters") is None:
    quality.missing_live_parameters += 1


def _validate_route_timing_mode(timing_mode: str) -> None:
  if timing_mode != "fixed-dt":
    raise ValueError("timing_mode 'original' is not yet supported; fixed-DT route replay only")


def _validate_route_sampling_mode(sampling_mode: str) -> None:
  if sampling_mode not in SAMPLE_MODES:
    raise ValueError(f"unknown sampling_mode {sampling_mode!r}")


def extract_lateral_route_frames_with_summary(
  messages: Any,
  *,
  route: str | None = None,
  qlog: bool = False,
  start_s: float | None = None,
  end_s: float | None = None,
  max_frames: int | None = None,
  timing_mode: str = "fixed-dt",
  sampling_mode: str = "prefix",
) -> tuple[tuple[LateralRouteFrame, ...], RouteExtractionSummary]:
  """Extract fixed-DT lateral replay frames from raw route messages.

  Iterates raw messages (with ``logMonoTime`` and ``which()``), maintains latest
  carState/carControl/modelV2/liveParameters, and emits one LateralRouteFrame per
  controlsState. Applies the original-time window, truncates to ``max_frames``,
  then normalizes ``frame.t = index * DT``.
  """
  _validate_route_timing_mode(timing_mode)
  _validate_route_sampling_mode(sampling_mode)
  from openpilot.tools.drive_lab.route_analysis import build_route_messages

  route_messages = build_route_messages(messages)
  latest_state: dict[str, Any] = {}
  extracted: list[LateralRouteFrame] = []
  original_times: list[float] = []
  quality = RouteExtractionQuality(input_message_count=len(route_messages))

  for rm in route_messages:
    if rm.typ in ("carState", "carControl", "modelV2", "liveParameters", "controlsState"):
      latest_state[rm.typ] = rm.payload
    if rm.typ != "controlsState":
      continue
    quality.controls_state_seen += 1
    if start_s is not None and rm.t < start_s:
      continue
    if end_s is not None and rm.t >= end_s:
      continue
    quality.controls_state_in_window += 1
    if latest_state.get("carState") is None:
      quality.skipped_missing_car_state += 1
      continue
    frame = _frame_from_controls_state(rm.t, latest_state, source_t=rm.t)
    _count_frame_quality(frame, latest_state, quality)
    extracted.append(frame)
    original_times.append(rm.t)
    if max_frames is not None and len(extracted) >= max_frames:
      break

  normalized = tuple(
    LateralRouteFrame.from_dict({**frame.to_dict(), "t": float(i) * DT})
    for i, frame in enumerate(extracted)
  )
  original_span = original_times[-1] - original_times[0] if len(original_times) > 1 else None
  summary = RouteExtractionSummary(
    route=route,
    qlog=qlog,
    window_start_s=start_s,
    window_end_s=end_s,
    max_frames=max_frames,
    extracted_count=len(normalized),
    original_time_span_s=original_span,
    dt=DT,
    quality=quality,
    timing_mode=timing_mode,
    sampling_mode=sampling_mode,
    selected_original_start_s=original_times[0] if original_times else None,
    selected_original_end_s=original_times[-1] if original_times else None,
    source_dt_stats=_source_dt_stats(normalized),
  )
  return normalized, summary


def extract_lateral_route_frames(
  messages: Any,
  *,
  start_s: float | None = None,
  end_s: float | None = None,
  max_frames: int | None = None,
) -> tuple[LateralRouteFrame, ...]:
  frames, _ = extract_lateral_route_frames_with_summary(messages, start_s=start_s, end_s=end_s, max_frames=max_frames)
  return frames


def _load_route_frames_with_summary(
  route: str,
  *,
  qlog: bool = False,
  start_s: float | None = None,
  end_s: float | None = None,
  max_frames: int | None = None,
  timing_mode: str = "fixed-dt",
  sampling_mode: str = "prefix",
) -> tuple[tuple[LateralRouteFrame, ...], RouteExtractionSummary]:
  from openpilot.tools.drive_lab.route_io import load_route_msgs

  messages = load_route_msgs(route, qlog=qlog)
  return extract_lateral_route_frames_with_summary(
    messages,
    route=route,
    qlog=qlog,
    start_s=start_s,
    end_s=end_s,
    max_frames=max_frames,
    timing_mode=timing_mode,
    sampling_mode=sampling_mode,
  )


def _load_route_frames(
  route: str,
  *,
  qlog: bool = False,
  start_s: float | None = None,
  end_s: float | None = None,
  max_frames: int | None = None,
) -> tuple[LateralRouteFrame, ...]:
  frames, _ = _load_route_frames_with_summary(
    route,
    qlog=qlog,
    start_s=start_s,
    end_s=end_s,
    max_frames=max_frames,
  )
  return frames


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
      out.append(_copy_frame(
        frame,
        raw_curvature=k,
        position_x=path["position_x"],
        position_y=path["position_y"],
        orientation_z=path["orientation_z"],
        orientation_rate_z=path["orientation_rate_z"],
      ))
    else:
      out.append(frame)
  return tuple(out)


def _apply_dropout(frames: tuple[LateralRouteFrame, ...], start: int, end: int) -> tuple[LateralRouteFrame, ...]:
  out: list[LateralRouteFrame] = []
  for i, frame in enumerate(frames):
    if start <= i < end:
      out.append(_copy_frame(
        frame,
        position_x=(),
        position_y=(),
        position_y_std=(),
        orientation_z=(),
        orientation_rate_z=(),
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
    out[i] = _copy_frame(
      frames[i],
      raw_curvature=k,
      position_x=path["position_x"],
      position_y=path["position_y"],
      orientation_z=path["orientation_z"],
      orientation_rate_z=path["orientation_rate_z"],
    )
  return tuple(out)


def _apply_stale(frames: tuple[LateralRouteFrame, ...], start: int, end: int) -> tuple[LateralRouteFrame, ...]:
  src = frames[start]
  k = src.raw_curvature
  out = list(frames)
  for i in range(start, end):
    path = _coherent_path(k, frames[i].v_ego)
    out[i] = _copy_frame(
      frames[i],
      raw_curvature=k,
      position_x=path["position_x"],
      position_y=path["position_y"],
      orientation_z=path["orientation_z"],
      orientation_rate_z=path["orientation_rate_z"],
    )
  return tuple(out)


def _apply_scale(rng: random.Random, frames: tuple[LateralRouteFrame, ...], start: int, end: int) -> tuple[LateralRouteFrame, ...]:
  factor = rng.uniform(0.9, 1.1)
  out: list[LateralRouteFrame] = []
  for i, frame in enumerate(frames):
    if start <= i < end:
      k = frame.raw_curvature * factor
      path = _coherent_path(k, frame.v_ego)
      out.append(_copy_frame(
        frame,
        raw_curvature=k,
        position_x=path["position_x"],
        position_y=path["position_y"],
        orientation_z=path["orientation_z"],
        orientation_rate_z=path["orientation_rate_z"],
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
      out.append(_copy_frame(
        frame,
        raw_curvature=k,
        position_x=path["position_x"],
        position_y=path["position_y"],
        orientation_z=path["orientation_z"],
        orientation_rate_z=path["orientation_rate_z"],
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
  if recipe.kind == "model_age_stale":
    stale_age_s = float(recipe.params.get("model_age_s", 0.30))
    return tuple(
      _copy_frame(frame, model_age_s=stale_age_s) if start <= i < end else frame
      for i, frame in enumerate(frames)
    )
  if recipe.kind == "model_age_delay":
    delay_frames = min(int(recipe.params.get("delay_frames", 3)), start)
    stale_age_s = float(recipe.params.get("model_age_s", 0.30))
    out = list(frames)
    for i in range(start, end):
      src = frames[i - delay_frames]
      k = src.raw_curvature
      path = _coherent_path(k, frames[i].v_ego)
      out[i] = _copy_frame(
        frames[i],
        raw_curvature=k,
        position_x=path["position_x"],
        position_y=path["position_y"],
        orientation_z=path["orientation_z"],
        orientation_rate_z=path["orientation_rate_z"],
        model_age_s=stale_age_s,
      )
    return tuple(out)
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
  elif kind == "model_age_stale":
    params = {"model_age_s": rng.uniform(0.25, 0.45)}
    desc = f"stale model age {params['model_age_s']:.2f}s frames {start}-{end}"
  elif kind == "model_age_delay":
    if start < 2:
      start = min(max(2, start), max(0, n - window - 1))
      end = min(n, start + window)
    max_delay = max(1, min(10, start))
    params = {"delay_frames": rng.randint(1, max_delay), "model_age_s": rng.uniform(0.25, 0.45)}
    desc = f"stale model age + causal delay {params['delay_frames']} frames {start}-{end}"
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
  route: str | None = None
  qlog: bool = False
  window_start_s: float | None = None
  window_end_s: float | None = None
  max_frames: int | None = None
  sample_mode: str = "prefix"
  sample_window_duration_s: float | None = None
  sample_window_count: int = 1
  sample_seed: int | None = None
  timing_mode: str = "fixed-dt"


def _validate_route_config(config: RouteReplayFuzzerConfig) -> None:
  _validate_route_timing_mode(config.timing_mode)
  if config.sample_mode not in SAMPLE_MODES:
    raise ValueError(f"unknown sample_mode {config.sample_mode!r}")
  if config.sample_mode != "prefix" and config.sample_window_duration_s is None:
    raise ValueError(f"sample_mode {config.sample_mode!r} requires sample_window_duration_s")
  if config.sample_window_duration_s is not None and config.sample_window_duration_s <= 0.0:
    raise ValueError("sample_window_duration_s must be > 0")
  if config.sample_window_count <= 0:
    raise ValueError("sample_window_count must be > 0")


def _generate_route_scenarios(
  baseline: tuple[LateralRouteFrame, ...],
  metadata: RouteExtractionSummary,
  config: RouteReplayFuzzerConfig,
) -> list[RouteReplayScenario]:
  if not baseline:
    return []
  rng = random.Random(config.seed)
  n = len(baseline)
  scenarios: list[RouteReplayScenario] = []
  for idx in range(config.cases):
    recipe = _generate_recipe(rng, n, config.perturbation)
    perturbed = _apply_recipe(recipe, baseline)
    title = f"route replay {ROUTE_EXTRACTED_PRESET} with {recipe.kind} #{idx}"
    scenarios.append(RouteReplayScenario(
      preset=ROUTE_EXTRACTED_PRESET,
      title=title,
      frames=baseline,
      recipe=recipe,
      perturbed_frames=perturbed,
      route_metadata=metadata,
    ))
  return scenarios


def _sample_route_windows(
  frames: tuple[LateralRouteFrame, ...],
  base_summary: RouteExtractionSummary,
  config: RouteReplayFuzzerConfig,
) -> list[tuple[tuple[LateralRouteFrame, ...], RouteExtractionSummary]]:
  """Split extracted frames into sampled replay windows.

  ``prefix`` sampling returns the whole input unchanged. ``random-window`` and
  ``uniform-windows`` select ``sample_window_count`` sub-windows of
  ``sample_window_duration_s`` seconds from the original route time span and
  re-normalize each window to ``frame.t = index * DT``.
  """
  if config.sample_mode == "prefix":
    return [(frames, base_summary)]
  if not frames:
    return []

  source_times = [frame.source_t for frame in frames if frame.source_t is not None]
  if len(source_times) < 2:
    return []

  start_t = source_times[0]
  end_t = source_times[-1]
  available = end_t - start_t
  duration = config.sample_window_duration_s
  if duration is None or duration <= 0.0:
    return [(frames, base_summary)]

  rng = random.Random(config.sample_seed if config.sample_seed is not None else config.seed)
  window_count = config.sample_window_count
  windows: list[tuple[float, float]] = []

  if config.sample_mode == "random-window":
    if duration >= available:
      windows = [(start_t, end_t)]
    else:
      starts = sorted(start_t + rng.random() * (available - duration) for _ in range(window_count))
      windows = [(s, s + duration) for s in starts]
  elif config.sample_mode == "uniform-windows":
    if duration >= available:
      windows = [(start_t, end_t)]
    elif window_count == 1:
      windows = [(start_t, min(end_t, start_t + duration))]
    else:
      last_start = end_t - duration
      step = (last_start - start_t) / float(window_count - 1)
      windows = [(start_t + i * step, min(end_t, start_t + i * step + duration)) for i in range(window_count)]
  else:
    return [(frames, base_summary)]

  results: list[tuple[tuple[LateralRouteFrame, ...], RouteExtractionSummary]] = []
  sampling_seed = config.sample_seed if config.sample_seed is not None else config.seed
  for window_idx, (w_start, w_end) in enumerate(windows):
    selected_raw = [
      frame for frame in frames
      if frame.source_t is not None and w_start <= frame.source_t < w_end
    ]
    selected = tuple(
      LateralRouteFrame.from_dict({**frame.to_dict(), "t": float(i) * DT})
      for i, frame in enumerate(selected_raw)
    )
    if config.max_frames is not None:
      selected = selected[:config.max_frames]
    if not selected:
      continue
    original_times = [frame.source_t for frame in selected if frame.source_t is not None]
    original_span = original_times[-1] - original_times[0] if len(original_times) > 1 else None
    summary = RouteExtractionSummary(
      route=base_summary.route,
      qlog=base_summary.qlog,
      window_start_s=base_summary.window_start_s,
      window_end_s=base_summary.window_end_s,
      max_frames=config.max_frames,
      extracted_count=len(selected),
      original_time_span_s=original_span,
      dt=DT,
      quality=base_summary.quality,
      quality_scope="base_extraction",
      timing_mode=base_summary.timing_mode,
      sampling_mode=config.sample_mode,
      sampling_window_index=window_idx,
      sampling_window_count=len(windows),
      sampling_seed=sampling_seed,
      sampling_window_duration_s=duration,
      selected_original_start_s=w_start,
      selected_original_end_s=w_end,
      source_dt_stats=_source_dt_stats(selected),
    )
    results.append((selected, summary))
  return results


def _route_window_config(config: RouteReplayFuzzerConfig) -> RouteReplayFuzzerConfig:
  """Return a config suitable for loading the base route before sampling."""
  return RouteReplayFuzzerConfig(
    seed=config.seed,
    cases=config.cases,
    preset=config.preset,
    perturbation=config.perturbation,
    duration_s=config.duration_s,
    route=config.route,
    qlog=config.qlog,
    window_start_s=config.window_start_s,
    window_end_s=config.window_end_s,
    max_frames=None if config.sample_mode != "prefix" else config.max_frames,
    sample_mode="prefix",
    sample_window_duration_s=None,
    sample_window_count=1,
    sample_seed=None,
    timing_mode=config.timing_mode,
  )


def generate_scenarios(config: RouteReplayFuzzerConfig) -> list[RouteReplayScenario]:
  if config.route is not None:
    _validate_route_config(config)
    load_config = _route_window_config(config)
    assert load_config.route is not None
    baseline, metadata = _load_route_frames_with_summary(
      load_config.route,
      qlog=load_config.qlog,
      start_s=load_config.window_start_s,
      end_s=load_config.window_end_s,
      max_frames=load_config.max_frames,
      timing_mode=load_config.timing_mode,
      sampling_mode=load_config.sample_mode,
    )
    if config.sample_mode == "prefix":
      return _generate_route_scenarios(baseline, metadata, config)

    windows = _sample_route_windows(baseline, metadata, config)
    if not windows:
      return []
    rng = random.Random(config.seed)
    scenarios: list[RouteReplayScenario] = []
    for idx in range(config.cases):
      window_idx = idx % len(windows)
      window_frames, window_metadata = windows[window_idx]
      recipe = _generate_recipe(rng, len(window_frames), config.perturbation)
      perturbed = _apply_recipe(recipe, window_frames)
      title = f"route replay {ROUTE_EXTRACTED_PRESET} with {recipe.kind} #{idx} window={window_idx}"
      scenarios.append(RouteReplayScenario(
        preset=ROUTE_EXTRACTED_PRESET,
        title=title,
        frames=window_frames,
        recipe=recipe,
        perturbed_frames=perturbed,
        route_metadata=window_metadata,
      ))
    return scenarios

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


def _run_frames(
  frames: tuple[LateralRouteFrame, ...], *, demand_enabled: bool = True, use_enabled_demand_config: bool = False,
) -> tuple[RouteFrameOutput, ...]:
  pipeline = LateralDemandPipeline(dt=DT)
  outputs: list[RouteFrameOutput] = []
  for frame in frames:
    inputs = _frame_to_inputs(frame)
    if use_enabled_demand_config:
      inputs = replace(inputs, smooth_model_path_curvature=True, demand_jerk_smoothing_enabled=True)
    result = pipeline.update(inputs)
    outputs.append(_output_from_result(frame, result, demand_enabled=demand_enabled))
  return tuple(outputs)


def _lat_accel(v_ego: float | np.ndarray, curvature: float | np.ndarray) -> float | np.ndarray:
  return (v_ego * v_ego) * curvature


DEMAND_AB_CURVATURE_EPS = 1e-6


def _metric_span(values: np.ndarray) -> float | None:
  finite = values[np.isfinite(values)]
  if finite.size < 2:
    return None
  return float(np.percentile(finite, 95.0) - np.percentile(finite, 5.0))


def _metric_p95_abs(values: np.ndarray) -> float | None:
  finite = np.abs(values[np.isfinite(values)])
  return float(np.percentile(finite, 95.0)) if finite.size else None


def _metric_jerk_p95(values: np.ndarray, times: np.ndarray) -> float | None:
  if values.size < 2 or times.size != values.size:
    return None
  dt = np.diff(times)
  dv = np.diff(values)
  valid = (dt > 1e-6) & np.isfinite(dt) & np.isfinite(dv)
  return _metric_p95_abs(dv[valid] / dt[valid]) if np.any(valid) else None


def _metric_times(frames: tuple[LateralRouteFrame, ...]) -> tuple[np.ndarray, bool]:
  source_times = [frame.source_t for frame in frames]
  if len(source_times) == len(frames) and all(time is not None and math.isfinite(float(time)) for time in source_times):
    times = np.asarray([float(time) for time in source_times if time is not None], dtype=float)
    if len(times) < 2 or np.all(np.diff(times) > 0.0):
      return times, True
  return np.asarray([frame.t for frame in frames], dtype=float), False


def _fmt(value: Any, ndigits: int = 6) -> str:
  if value is None:
    return "n/a"
  try:
    value_f = float(value)
  except (TypeError, ValueError):
    return "n/a"
  return "n/a" if not math.isfinite(value_f) else f"{value_f:.{ndigits}f}"


def _demand_ab_metrics(frames: tuple[LateralRouteFrame, ...], outputs: tuple[RouteFrameOutput, ...]) -> dict[str, float | int | None]:
  raw = np.asarray([frame.raw_curvature for frame in frames], dtype=float)
  processed = np.asarray([output.processed_curvature for output in outputs], dtype=float)
  command = np.asarray([output.command_curvature for output in outputs], dtype=float)
  measured = np.asarray([frame.measured_curvature for frame in frames], dtype=float)
  v_ego = np.asarray([frame.v_ego for frame in frames], dtype=float)
  times, _ = _metric_times(frames)
  path_error = _lat_accel(v_ego, command - measured)
  metrics: dict[str, float | int | None] = {}
  for name, values in (("raw", raw), ("processed", processed), ("command", command)):
    metrics[f"{name}_curvature_oscillation_pp"] = _metric_span(values)
    metrics[f"{name}_curvature_reversals"] = _sign_flip_count(values, DEMAND_AB_CURVATURE_EPS)
    metrics[f"{name}_curvature_jerk_p95"] = _metric_jerk_p95(values, times)
  metrics["path_wander_evidence"] = _metric_p95_abs(path_error)
  return metrics


def _demand_ab_variant(
  name: str, demand_enabled: bool, frames: tuple[LateralRouteFrame, ...], outputs: tuple[RouteFrameOutput, ...],
) -> LateralDemandABVariant:
  source_times = [frame.source_t for frame in frames if frame.source_t is not None]
  _, source_timing_used = _metric_times(frames)
  return LateralDemandABVariant(
    name=name,
    demand_enabled=demand_enabled,
    frame_count=len(frames),
    normalized_start_s=frames[0].t if frames else None,
    normalized_end_s=frames[-1].t if frames else None,
    source_start_s=float(source_times[0]) if source_times else None,
    source_end_s=float(source_times[-1]) if source_times else None,
    source_timing_used=source_timing_used,
    metrics=_demand_ab_metrics(frames, outputs),
  )


def analyze_lateral_demand_ab(
  frames: tuple[LateralRouteFrame, ...] | list[LateralRouteFrame],
  *,
  source: str = "unknown",
  route_metadata: RouteExtractionSummary | None = None,
) -> LateralDemandABReport:
  """Replay one extracted frame window with demand processing enabled and disabled.

  Both variants execute the same ``LateralDemandPipeline`` with identical inputs and the
  production enabled smoothing flags. The disabled stream reports the adapter's raw
  pass-through output; no pass/fail threshold or control tuning is applied.
  """
  selected = tuple(frames)
  enabled_outputs = _run_frames(selected, demand_enabled=True, use_enabled_demand_config=True)
  disabled_outputs = _run_frames(selected, demand_enabled=False, use_enabled_demand_config=True)
  enabled = _demand_ab_variant("enabled", True, selected, enabled_outputs)
  disabled = _demand_ab_variant("disabled", False, selected, disabled_outputs)
  deltas: dict[str, float | None] = {}
  for key in enabled.metrics:
    enabled_value = enabled.metrics[key]
    disabled_value = disabled.metrics[key]
    if enabled_value is None or disabled_value is None:
      deltas[key] = None
    else:
      deltas[key] = float(enabled_value) - float(disabled_value)
  return LateralDemandABReport(
    source=source,
    route_metadata=route_metadata,
    variants={"disabled": disabled, "enabled": enabled},
    deltas_enabled_minus_disabled=deltas,
    notes=[
      "measurement-only: no pass/fail threshold or tuning change",
      "command curvature is the replayed pre-clip processed demand output",
      "path_wander_evidence is p95 absolute command-versus-measured lateral-acceleration error; vehicle dynamics are not re-simulated",
      "both variants re-execute LateralDemandPipeline with identical inputs/config; only demand output enable/pass-through differs",
    ],
  )


def render_lateral_demand_ab(report: LateralDemandABReport) -> str:
  lines = [f"Lateral demand replay A/B: {report.source}"]
  for name in ("disabled", "enabled"):
    variant = report.variants[name]
    lines.append(
      f"  {name}: demand_enabled={variant.demand_enabled} frames={variant.frame_count} "
      + f"source={variant.source_start_s}..{variant.source_end_s} normalized={variant.normalized_start_s}..{variant.normalized_end_s} "
      + f"metric_timing={'source_t' if variant.source_timing_used else 'normalized_t'}"
    )
    for key, value in sorted(variant.metrics.items()):
      lines.append(f"    {key}={_fmt(value)}")
  lines.append("  deltas (enabled - disabled):")
  for key, value in sorted(report.deltas_enabled_minus_disabled.items()):
    lines.append(f"    {key}={_fmt(value)}")
  lines.extend(f"  note: {note}" for note in report.notes)
  return "\n".join(lines)


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


def _evaluate_outputs(
  outputs: tuple[RouteFrameOutput, ...],
  thresholds: RouteReplayThresholds,
  label: str,
  skip_step_end_frames: tuple[int, ...] = (),
) -> list[dict[str, Any]]:
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
    step_mask = ~(quality[1:] < 0.9) & ~(quality[:-1] < 0.9)
    gated = np.array([o.gated for o in outputs], dtype=bool)
    step_mask &= ~(gated[1:] | gated[:-1])
    if skip_step_end_frames:
      for frame_idx in skip_step_end_frames:
        step_idx = int(frame_idx) - 1
        if 0 <= step_idx < step_mask.size:
          step_mask[step_idx] = False
    steps = steps[step_mask]
    dt = DT
    jerks = steps / dt
    max_step = float(np.max(np.abs(steps))) if steps.size else 0.0
    max_jerk = float(np.max(np.abs(jerks))) if jerks.size else 0.0
    if max_step > thresholds.max_abs_step_lat_accel:
      failures.append({"check": "lat_accel_step", "detail": f"{label}: lateral accel step {max_step:.2f} m/s^2 exceeds {thresholds.max_abs_step_lat_accel}"})
    if max_jerk > thresholds.max_abs_lat_jerk:
      failures.append({"check": "lat_jerk", "detail": f"{label}: lateral jerk {max_jerk:.2f} m/s^3 exceeds {thresholds.max_abs_lat_jerk}"})

  return failures


def _expected_perturbed_step_end_frames(recipe: PerturbationRecipe) -> tuple[int, ...]:
  if recipe.kind in ("stale", "model_age_stale", "model_age_delay") and recipe.end_frame > recipe.start_frame:
    return (recipe.end_frame,)
  return ()


def _comparison_delta_mask(scenario: RouteReplayScenario, size: int) -> np.ndarray:
  mask = np.ones(size, dtype=bool)
  if scenario.recipe.kind in ("stale", "model_age_stale", "model_age_delay"):
    start = max(0, min(size, scenario.recipe.start_frame))
    end = max(start, min(size, scenario.recipe.end_frame))
    mask[start:end] = False
  return mask


def _comparison_oscillation_mask(scenario: RouteReplayScenario, size: int) -> np.ndarray:
  mask = np.ones(size, dtype=bool)
  if scenario.recipe.kind != "none":
    start = max(0, min(size, scenario.recipe.start_frame))
    end = max(start, min(size, scenario.recipe.end_frame))
    mask[start:end] = False
  return mask


def _sign_flip_count_masked(values: np.ndarray, eps: float, mask: np.ndarray) -> int:
  flips = 0
  previous_sign: float | None = None
  for value, include in zip(values, mask, strict=False):
    if not include or not np.isfinite(value) or abs(value) <= eps:
      previous_sign = None if not include else previous_sign
      continue
    sign = float(np.sign(value))
    if previous_sign is not None and sign != previous_sign:
      flips += 1
    previous_sign = sign
  return flips


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

  perturbed_frames = scenario.perturbed_frames or ()
  try:
    perturbed_frames = scenario.perturbed_frames or _apply_recipe(scenario.recipe, scenario.frames)
    perturbation_failures.extend(_validate_input_frames(perturbed_frames, "perturbed"))
    perturbed_outputs = _run_frames(perturbed_frames) if not perturbation_failures else ()
  except Exception as exc:
    perturbation_failures.append({"check": "exception", "detail": f"perturbed raised {type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
    perturbed_outputs = ()

  if perturbed_outputs:
    perturbation_failures.extend(
      _evaluate_outputs(
        perturbed_outputs,
        thresholds,
        "perturbed",
        skip_step_end_frames=_expected_perturbed_step_end_frames(scenario.recipe),
      )
    )

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
    delta_mask = _comparison_delta_mask(scenario, delta.size)
    checked_delta = delta[delta_mask]
    max_delta = float(np.max(checked_delta)) if checked_delta.size else 0.0
    metrics["max_baseline_perturbed_lat_accel_delta"] = max_delta
    if max_delta > thresholds.max_baseline_perturbed_lat_accel_delta:
      comparison_failures.append({
        "check": "perturbed_divergence",
        "detail": f"perturbed diverged from baseline by {max_delta:.2f} m/s^2 lateral accel",
      })

    oscillation_mask = _comparison_oscillation_mask(scenario, base_lat_accel.size)
    base_flips = _sign_flip_count_masked(base_lat_accel, thresholds.oscillation_lat_accel_eps, oscillation_mask)
    pert_flips = _sign_flip_count_masked(pert_lat_accel, thresholds.oscillation_lat_accel_eps, oscillation_mask)
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

    # Targeted stale-age check.
    if scenario.recipe.kind in ("model_age_stale", "model_age_delay"):
      window = perturbed_outputs[scenario.recipe.start_frame:scenario.recipe.end_frame]
      active_window = [o for o in window if o.v_ego > 0.1]
      stale_count = sum(1 for o in active_window if o.path_reason == "model_stale")
      required = max(1, int(0.9 * len(active_window))) if active_window else 0
      if stale_count < required:
        comparison_failures.append({
          "check": "expected_model_age_stale_reaction",
          "detail": f"stale model-age window produced {stale_count}/{len(active_window)} model_stale frames",
        })
      if scenario.recipe.kind == "model_age_delay" and window:
        window_start = scenario.recipe.start_frame
        window_end = scenario.recipe.end_frame
        material_frames = 0
        max_raw_lat_error = 0.0
        max_processed_lat_error = 0.0
        for frame, output in zip(perturbed_frames[window_start:window_end], window, strict=False):
          v_sq = float(output.v_ego) ** 2
          raw_lat_error = abs(float(frame.raw_curvature) - float(frame.measured_curvature)) * v_sq
          if raw_lat_error <= _MODEL_AGE_MATERIAL_LAT_ACCEL_EPS:
            continue
          material_frames += 1
          processed_lat_error = abs(float(output.processed_curvature) - float(frame.measured_curvature)) * v_sq
          max_raw_lat_error = max(max_raw_lat_error, raw_lat_error)
          max_processed_lat_error = max(max_processed_lat_error, processed_lat_error)
        metrics["model_age_delay_material_frames"] = material_frames
        metrics["model_age_delay_max_raw_lat_error"] = max_raw_lat_error
        metrics["model_age_delay_max_processed_lat_error"] = max_processed_lat_error
        if material_frames > 0:
          if (max_processed_lat_error > max_raw_lat_error + _MODEL_AGE_DELAY_MAX_EXTRA_LAT_ACCEL_ERROR and
              max_processed_lat_error > _MODEL_AGE_DELAY_MAX_PROCESSED_LAT_ACCEL_ERROR):
            comparison_failures.append({
              "check": "expected_stale_age_bridge_toward_measured",
              "detail": (
                f"stale-age delayed max processed lateral error {max_processed_lat_error:.3f} m/s^2 "
                + f"exceeds raw {max_raw_lat_error:.3f} m/s^2 + {_MODEL_AGE_DELAY_MAX_EXTRA_LAT_ACCEL_ERROR:.3f} "
                + f"and absolute cap {_MODEL_AGE_DELAY_MAX_PROCESSED_LAT_ACCEL_ERROR:.3f} m/s^2 "
                + f"({material_frames} material frames)"
              ),
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
  if scenario.route_metadata is not None:
    payload["route_metadata"] = scenario.route_metadata.to_dict()
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
  route_metadata = data.get("route_metadata")
  return RouteReplayScenario(
    preset=str(data["preset"]),
    title=str(data["title"]),
    frames=tuple(LateralRouteFrame.from_dict(frame) for frame in data["baseline_frames"]),
    recipe=PerturbationRecipe.from_dict(data["recipe"]),
    thresholds=RouteReplayThresholds.from_dict(data.get("thresholds", {})),
    perturbed_frames=tuple(LateralRouteFrame.from_dict(frame) for frame in data.get("perturbed_frames", ())),
    route_metadata=RouteExtractionSummary.from_dict(route_metadata) if route_metadata else None,
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
  payload: dict[str, Any] = {
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
  if result.scenario.route_metadata is not None:
    payload["route_metadata"] = result.scenario.route_metadata.to_dict()
  return payload


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
  return (
    f"# preset: {scenario.preset} perturbation: {scenario.recipe.kind}\n"
    + f"RouteReplayScenario(title={scenario.title!r}, frames=[...{len(scenario.frames)} frames...])"
  )


def _parse_window(window: str | None) -> tuple[float | None, float | None]:
  if window is None:
    return (None, None)
  parts = window.split(",")
  if len(parts) != 2:
    raise argparse.ArgumentTypeError("--window must be START,END")
  try:
    start = float(parts[0])
    end = float(parts[1])
  except ValueError as exc:
    raise argparse.ArgumentTypeError(f"--window values must be numeric: {exc}") from exc
  if not (math.isfinite(start) and math.isfinite(end)):
    raise argparse.ArgumentTypeError("--window values must be finite")
  if end <= start:
    raise argparse.ArgumentTypeError("--window START must be less than END")
  return (start, end)


def main() -> None:
  parser = argparse.ArgumentParser(description="Synthetic / serialized route-like lateral replay perturbation fuzzer.")
  parser.add_argument("--seed", type=int, default=1)
  parser.add_argument("--cases", type=int, default=100)
  parser.add_argument("--preset", choices=PRESETS, help="Route preset")
  parser.add_argument("--perturbation", choices=CLI_PERTURBATION_KINDS, help="Perturbation kind (default random)")
  parser.add_argument("--duration", type=float, default=2.0, help="Scenario duration in seconds")
  parser.add_argument("--json", action="store_true", help="Emit JSON instead of text")
  parser.add_argument("--output", type=str, default=None, help="Write demand A/B JSON to this path")
  parser.add_argument("--fail-fast", action="store_true", help="Stop after the first failure")
  parser.add_argument("--artifact-dir", type=str, default=None, help="Directory to write failure artifacts")
  parser.add_argument("--replay", type=str, default=None, help="Replay a route-replay artifact JSON file")
  parser.add_argument("--route", type=str, default=None, help="Route base/segment, local rlog file, or URL accepted by LogReader")
  parser.add_argument("--qlog", action="store_true", help="Use qlog when loading route")
  parser.add_argument("--window", type=str, default=None, help="Original-time window START,END in seconds")
  parser.add_argument("--max-frames", type=int, default=None, help="Maximum route frames to extract")
  parser.add_argument("--sample-mode", choices=SAMPLE_MODES, default="prefix", help="Route sampling mode (default prefix)")
  parser.add_argument("--sample-window-duration", type=float, default=None, help="Sampled window duration in seconds")
  parser.add_argument("--sample-window-count", type=int, default=1, help="Number of sampled windows")
  parser.add_argument("--sample-seed", type=int, default=None, help="Seed for random-window sampling (defaults to --seed)")
  parser.add_argument("--timing", choices=TIMING_MODES, default="fixed-dt", help="Timing mode for route replay (default fixed-dt)")
  parser.add_argument("--demand-ab", action="store_true", help="Compare demand processing enabled vs raw pass-through on one route window")
  parser.add_argument("--list-only", action="store_true", help="Only extract and summarize route frames; skip fuzzing")
  args = parser.parse_args()

  if args.cases < 0:
    parser.error("--cases must be >= 0")
  if args.duration <= 0.0:
    parser.error("--duration must be > 0")
  if args.max_frames is not None and args.max_frames <= 0:
    parser.error("--max-frames must be > 0")
  try:
    window_start_s, window_end_s = _parse_window(args.window)
  except argparse.ArgumentTypeError as exc:
    parser.error(str(exc))

  if args.route and args.preset:
    parser.error("--route and --preset are mutually exclusive")
  if args.list_only and not args.route:
    parser.error("--list-only requires --route")
  if args.route and args.cases <= 0:
    parser.error("--route requires --cases > 0")
  if args.sample_mode != "prefix" and args.sample_window_duration is None:
    parser.error(f"--sample-mode {args.sample_mode} requires --sample-window-duration")
  if args.sample_window_duration is not None and args.sample_window_duration <= 0.0:
    parser.error("--sample-window-duration must be > 0")
  if args.sample_window_count <= 0:
    parser.error("--sample-window-count must be > 0")
  if args.route and args.timing == "original":
    parser.error("--timing original is not yet supported; use fixed-dt")
  if args.demand_ab and not args.route:
    parser.error("--demand-ab requires --route")
  if args.demand_ab and (args.replay or args.list_only):
    parser.error("--demand-ab cannot be combined with --replay or --list-only")
  if args.demand_ab and args.sample_mode != "prefix":
    parser.error("--demand-ab uses the explicit --window; --sample-mode is not supported")
  sample_seed = args.sample_seed if args.sample_seed is not None else args.seed

  if args.demand_ab:
    assert args.route is not None
    frames, metadata = _load_route_frames_with_summary(
      args.route,
      qlog=args.qlog,
      start_s=window_start_s,
      end_s=window_end_s,
      max_frames=args.max_frames,
      timing_mode=args.timing,
      sampling_mode="prefix",
    )
    if not frames:
      parser.error(f"no route frames extracted from {args.route}")
    report = analyze_lateral_demand_ab(frames, source=args.route, route_metadata=metadata)
    if args.output:
      Path(args.output).write_text(json.dumps(_sanitize(report.to_dict()), indent=2, sort_keys=True, allow_nan=False) + "\n")
    if args.json:
      print(json.dumps(_sanitize(report.to_dict()), indent=2, sort_keys=True, allow_nan=False))
    else:
      print(render_lateral_demand_ab(report))
    return

  if args.route and args.list_only:
    list_config = RouteReplayFuzzerConfig(
      seed=args.seed,
      route=args.route,
      qlog=args.qlog,
      window_start_s=window_start_s,
      window_end_s=window_end_s,
      max_frames=None if args.sample_mode != "prefix" else args.max_frames,
      sample_mode=args.sample_mode,
      sample_window_duration_s=args.sample_window_duration,
      sample_window_count=args.sample_window_count,
      sample_seed=sample_seed,
      timing_mode=args.timing,
    )
    load_config = _route_window_config(list_config)
    assert load_config.route is not None
    baseline, base_summary = _load_route_frames_with_summary(
      load_config.route,
      qlog=load_config.qlog,
      start_s=load_config.window_start_s,
      end_s=load_config.window_end_s,
      max_frames=load_config.max_frames,
      timing_mode=load_config.timing_mode,
      sampling_mode=load_config.sample_mode,
    )
    windows = _sample_route_windows(baseline, base_summary, list_config)
    summaries = [window_summary for _, window_summary in windows]
    if args.json:
      if args.sample_mode == "prefix":
        payload = _sanitize(summaries[0].to_dict()) if summaries else {}
      else:
        payload = {"windows": [_sanitize(summary.to_dict()) for summary in summaries]}
      print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    else:
      if args.sample_mode == "prefix":
        summary = summaries[0] if summaries else base_summary
        print(f"Route extraction summary for {args.route}:")
        print(f"  extracted_frames={summary.extracted_count}")
        print(f"  qlog={summary.qlog}")
        print(f"  window={summary.window_start_s},{summary.window_end_s}")
        print(f"  max_frames={summary.max_frames}")
        print(f"  original_time_span_s={summary.original_time_span_s}")
        print(f"  dt={summary.dt}")
        if summary.quality is not None:
          print(f"  quality={summary.quality.to_dict()}")
      else:
        print(f"Route extraction summary for {args.route} ({len(summaries)} sampled windows):")
        for idx, summary in enumerate(summaries):
          print(
            f"  window[{idx}]: extracted={summary.extracted_count} "
            + f"original_start={summary.selected_original_start_s:.3f} "
            + f"original_end={summary.selected_original_end_s:.3f} "
            + f"dt={summary.dt}"
          )
          if summary.quality is not None:
            print(f"    quality={summary.quality.to_dict()}")
    if not summaries or not any(summary.extracted_count > 0 for summary in summaries):
      raise SystemExit(1)
    return

  if args.replay:
    result = replay_artifact(args.replay)
    if args.json:
      print(json.dumps(_sanitize(artifact_to_dict(result, seed=None, index=None)), indent=2, sort_keys=True, allow_nan=False))
    else:
      print(
        f"Replayed {args.replay}: valid={result.valid} baseline_failures={len(result.baseline_failures)} "
        + f"perturbation_failures={len(result.perturbation_failures)} "
        + f"comparison_failures={len(result.comparison_failures)}"
      )
      for failure in result.baseline_failures:
        print(f"  baseline: {failure['check']}: {failure['detail']}")
      for failure in result.perturbation_failures:
        print(f"  perturbed: {failure['check']}: {failure['detail']}")
      for failure in result.comparison_failures:
        print(f"  comparison: {failure['check']}: {failure['detail']}")
    sys.exit(0 if result.valid else 1)

  config = RouteReplayFuzzerConfig(
    seed=args.seed,
    cases=args.cases,
    preset=args.preset,
    perturbation=args.perturbation,
    duration_s=args.duration,
    route=args.route,
    qlog=args.qlog,
    window_start_s=window_start_s,
    window_end_s=window_end_s,
    max_frames=args.max_frames,
    sample_mode=args.sample_mode,
    sample_window_duration_s=args.sample_window_duration,
    sample_window_count=args.sample_window_count,
    sample_seed=sample_seed,
    timing_mode=args.timing,
  )
  scenarios = list(generate_scenarios(config))
  if args.route and not scenarios:
    parser.error(f"no route frames extracted from {args.route}")
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
    payload: dict[str, Any] = {
      "seed": args.seed,
      "cases": len(results),
      "preset": args.preset,
      "perturbation": args.perturbation,
      "duration": None if args.route else args.duration,
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
    if args.route:
      if results and results[0][1].scenario.route_metadata is not None:
        payload["route_metadata"] = results[0][1].scenario.route_metadata.to_dict()
        if args.sample_mode != "prefix":
          sampled_windows: list[dict[str, Any]] = []
          seen_windows: set[int | None] = set()
          for _, result in results:
            meta = result.scenario.route_metadata
            if meta is None or meta.sampling_window_index in seen_windows:
              continue
            seen_windows.add(meta.sampling_window_index)
            sampled_windows.append(meta.to_dict())
          payload["sampled_windows"] = sampled_windows
      else:
        payload["route"] = args.route
        payload["qlog"] = args.qlog
        payload["window"] = args.window
        payload["max_frames"] = args.max_frames
    print(json.dumps(_sanitize(payload), indent=2, sort_keys=True, allow_nan=False))
  else:
    source = args.route or (args.preset or "all")
    route_span = None
    if args.route and results and results[0][1].scenario.route_metadata is not None:
      meta = results[0][1].scenario.route_metadata
      route_span = max(0.0, (meta.extracted_count - 1) * DT)
    span_text = f"route_fixed_dt_span={route_span}s" if args.route else f"duration={args.duration}s"
    print(
      f"Drive Lab lateral route replay fuzz seed={args.seed} cases={len(results)} "
      + f"preset={source} perturbation={args.perturbation or 'random'} "
      + f"{span_text} dt={DT}s failures={len(failures)}"
    )
    for _idx, result in failures[:10]:
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
