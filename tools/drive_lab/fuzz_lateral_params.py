#!/usr/bin/env python3
"""Synthetic lateral demand params/config structural fuzzer.

This tool exercises ``sunnypilot.custom.lateral.demand.wiring.LateralDemandAdapter``
with a fake params backend and deterministic serialized model inputs. It verifies
fail-closed behavior, params refresh cadence, and feature toggles without touching
real Params, device state, or production LatControl wiring.
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
from types import SimpleNamespace
from typing import Any

import numpy as np

from openpilot.sunnypilot.custom.lateral.demand.wiring import (
  LateralDemandAdapter,
  PARAMS_REFRESH_PERIOD,
)


ARTIFACT_SCHEMA = "drive-lab-lateral-params-fuzzer-artifact"
ARTIFACT_VERSION = 1
DT = 0.01
N_PATH_POINTS = 33
DEFAULT_KINDS = (
  "demand_enable_cycle",
  "params_refresh_cadence",
  "params_read_fault",
  "missing_model_fail_closed",
  "toggle_matrix",
)
PARAM_KEYS = (
  "CustomLateralDemandEnabled",
  "LaneCenteringAssistEnabled",
  "CurveMemoryEnabled",
)


class FakeParams:
  """In-memory params backend supporting get_bool/put_bool and fault injection."""

  def __init__(self, initial: dict[str, bool] | None = None) -> None:
    self._values: dict[str, bool] = dict(initial) if initial else {}
    self._inject_fault = False

  def get_bool(self, key: str) -> bool:
    if self._inject_fault:
      raise RuntimeError("injected params read fault")
    return bool(self._values.get(key, False))

  def get(self, key: str, default: Any = None) -> Any:
    if self._inject_fault:
      raise RuntimeError("injected params read fault")
    return self._values.get(key, default)

  def put_bool(self, key: str, value: bool) -> None:
    self._values[key] = bool(value)

  def set_fault(self, inject: bool) -> None:
    self._inject_fault = bool(inject)


@dataclass(frozen=True)
class ParamFrame:
  """One synthetic params fuzzer frame."""

  t: float
  v_ego: float
  lat_active: bool
  raw_curvature: float
  measured_curvature: float
  model_data: dict[str, Any] | None
  desired_params: dict[str, bool]
  fault_on_refresh: bool = False

  def to_dict(self) -> dict[str, Any]:
    return {
      "t": self.t,
      "v_ego": self.v_ego,
      "lat_active": self.lat_active,
      "raw_curvature": self.raw_curvature,
      "measured_curvature": self.measured_curvature,
      "model_data": _sanitize(self.model_data),
      "desired_params": dict(self.desired_params),
      "fault_on_refresh": self.fault_on_refresh,
    }

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> ParamFrame:
    return cls(
      t=float(data["t"]),
      v_ego=float(data["v_ego"]),
      lat_active=bool(data["lat_active"]),
      raw_curvature=float(data["raw_curvature"]),
      measured_curvature=float(data["measured_curvature"]),
      model_data=data.get("model_data"),
      desired_params=dict(data.get("desired_params", {})),
      fault_on_refresh=bool(data.get("fault_on_refresh", False)),
    )


@dataclass(frozen=True)
class ParamThresholds:
  """Thresholds for structural checks."""

  max_abs_output_curvature: float = 0.5
  max_abs_step_lat_accel: float = 5.0
  max_abs_lat_jerk: float = 150.0
  raw_curvature_max: float = 0.5

  def to_dict(self) -> dict[str, Any]:
    return {
      "max_abs_output_curvature": self.max_abs_output_curvature,
      "max_abs_step_lat_accel": self.max_abs_step_lat_accel,
      "max_abs_lat_jerk": self.max_abs_lat_jerk,
      "raw_curvature_max": self.raw_curvature_max,
    }

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> ParamThresholds:
    return cls(**{k: data[k] for k in cls.__dataclass_fields__ if k in data})


@dataclass(frozen=True)
class ParamScenario:
  """A params fuzzer scenario."""

  kind: str
  title: str
  index: int
  frames: tuple[ParamFrame, ...]
  event_windows: tuple[tuple[int, int], ...]
  thresholds: ParamThresholds | None = None

  @property
  def metric_thresholds(self) -> ParamThresholds:
    return self.thresholds or ParamThresholds()


@dataclass(frozen=True)
class ParamFrameOutput:
  """Per-frame adapter output."""

  t: float
  v_ego: float
  raw_curvature: float
  output_curvature: float
  measured_curvature: float
  desired_params: dict[str, bool]
  enabled: bool
  lane_centering_assist_enabled: bool
  curve_memory_enabled: bool
  refresh_frame: bool
  params_fault: bool


@dataclass(frozen=True)
class ParamResult:
  scenario: ParamScenario
  outputs: tuple[ParamFrameOutput, ...]
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


def _time_array_with_min_frames(duration_s: float, min_frames: int, dt: float = DT) -> np.ndarray:
  min_duration = max(1, min_frames) - 1
  return _time_array(max(duration_s, min_duration * dt), dt)


def _coherent_path(curvature: float, v_ego: float, n: int = N_PATH_POINTS) -> dict[str, tuple[float, ...]]:
  xs = tuple(float(x) for x in range(n))
  ys = tuple(0.5 * curvature * x * x for x in range(n))
  ystd = tuple(0.1 for _ in range(n))
  yaw = tuple(curvature * x for x in range(n))
  yaw_rate = tuple(curvature * v_ego for _ in range(n))
  return {"x": xs, "y": ys, "yStd": ystd, "z": yaw, "rate_z": yaw_rate}


def _model_data_dict(curvature: float, v_ego: float) -> dict[str, Any]:
  path = _coherent_path(curvature, v_ego)
  return {
    "position": {"x": list(path["x"]), "y": list(path["y"]), "yStd": list(path["yStd"])},
    "orientation": {"z": list(path["z"])},
    "orientationRate": {"z": list(path["rate_z"])},
    "laneLineProbs": [0.9, 0.9, 0.9, 0.9],
    "frameDropPerc": 0.0,
  }


def _model_v2_from_dict(data: dict[str, Any] | None) -> Any:
  if data is None:
    return None
  position = SimpleNamespace(**data.get("position", {}))
  orientation = SimpleNamespace(**data.get("orientation", {}))
  orientation_rate = SimpleNamespace(**data.get("orientationRate", {}))
  return SimpleNamespace(
    position=position,
    orientation=orientation,
    orientationRate=orientation_rate,
    laneLineProbs=tuple(data.get("laneLineProbs", ())),
    frameDropPerc=float(data.get("frameDropPerc", 0.0)),
  )


def _base_frame(
  t: float,
  v_ego: float,
  curvature: float,
  desired_params: dict[str, bool],
  model_data: dict[str, Any] | None = None,
  fault_on_refresh: bool = False,
  lat_active: bool = True,
) -> ParamFrame:
  mk = curvature
  return ParamFrame(
    t=float(t),
    v_ego=float(v_ego),
    lat_active=bool(lat_active),
    raw_curvature=float(curvature),
    measured_curvature=float(mk),
    model_data=model_data,
    desired_params=dict(desired_params),
    fault_on_refresh=bool(fault_on_refresh),
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


def _refresh_frames(n: int) -> list[int]:
  # process() increments _tick then refreshes when _tick % PARAMS_REFRESH_PERIOD == 0.
  # Frame 0 => _tick=1, so first refresh at frame 99 (_tick=100).
  return [i for i in range(n) if (i + 1) % PARAMS_REFRESH_PERIOD == 0]


# ---------- scenario generators ----------


def _all_enabled_params(enabled: bool = True) -> dict[str, bool]:
  return {k: enabled for k in PARAM_KEYS}


def _generate_demand_enable_cycle(
  rng: random.Random,
  duration_s: float,
  v_ego: float,
  curvature: float,
  index: int,
) -> ParamScenario:
  t = _time_array_with_min_frames(duration_s, 2 * PARAMS_REFRESH_PERIOD + 20)
  n = len(t)
  refresh = _refresh_frames(n)
  # Use first two refresh frames as toggle points.
  off_start = refresh[0] if refresh else int(0.3 * n)
  on_start = refresh[1] if len(refresh) > 1 else int(0.7 * n)
  frames: list[ParamFrame] = []
  model_data = _model_data_dict(curvature, v_ego)
  for i, ti in enumerate(t):
    if i < off_start:
      desired = _all_enabled_params(True)
    elif i < on_start:
      desired = _all_enabled_params(False)
    else:
      desired = _all_enabled_params(True)
    frames.append(_base_frame(float(ti), v_ego, curvature, desired, model_data=model_data))
  return ParamScenario(
    kind="demand_enable_cycle",
    title=f"demand enable cycle #{index}",
    index=index,
    frames=tuple(frames),
    event_windows=_merge_windows([
      (max(0, off_start - 10), min(n, off_start + 15)),
      (max(0, on_start - 10), min(n, on_start + 15)),
    ]),
  )


def _generate_params_refresh_cadence(
  rng: random.Random,
  duration_s: float,
  v_ego: float,
  curvature: float,
  index: int,
) -> ParamScenario:
  t = _time_array_with_min_frames(duration_s, PARAMS_REFRESH_PERIOD + 20)
  n = len(t)
  refresh = _refresh_frames(n)
  first_refresh = refresh[0] if refresh else 99
  frames: list[ParamFrame] = []
  model_data = _model_data_dict(curvature, v_ego)
  for i, ti in enumerate(t):
    desired = {
      "CustomLateralDemandEnabled": True,
      "LaneCenteringAssistEnabled": i >= first_refresh // 2,
      "CurveMemoryEnabled": False,
    }
    frames.append(_base_frame(float(ti), v_ego, curvature, desired, model_data=model_data))
  return ParamScenario(
    kind="params_refresh_cadence",
    title=f"params refresh cadence #{index}",
    index=index,
    frames=tuple(frames),
    event_windows=_merge_windows([(max(0, first_refresh - 10), min(n, first_refresh + 15))]),
  )


def _generate_params_read_fault(
  rng: random.Random,
  duration_s: float,
  v_ego: float,
  curvature: float,
  index: int,
) -> ParamScenario:
  t = _time_array_with_min_frames(duration_s, PARAMS_REFRESH_PERIOD + 20)
  n = len(t)
  refresh = _refresh_frames(n)
  fault_frame = refresh[0] if refresh else 99
  frames: list[ParamFrame] = []
  model_data = _model_data_dict(curvature, v_ego)
  for i, ti in enumerate(t):
    frames.append(_base_frame(
      float(ti), v_ego, curvature,
      desired_params=_all_enabled_params(True),
      model_data=model_data,
      fault_on_refresh=(i == fault_frame),
    ))
  return ParamScenario(
    kind="params_read_fault",
    title=f"params read fault #{index}",
    index=index,
    frames=tuple(frames),
    event_windows=_merge_windows([(max(0, fault_frame - 10), min(n, fault_frame + 15))]),
  )


def _generate_missing_model_fail_closed(
  rng: random.Random,
  duration_s: float,
  v_ego: float,
  curvature: float,
  index: int,
) -> ParamScenario:
  t = _time_array(duration_s)
  n = len(t)
  frames: list[ParamFrame] = []
  for i, ti in enumerate(t):
    model_data = _model_data_dict(curvature, v_ego) if i % 2 == 0 else None
    frames.append(_base_frame(
      float(ti), v_ego, curvature,
      desired_params=_all_enabled_params(True),
      model_data=model_data,
    ))
  return ParamScenario(
    kind="missing_model_fail_closed",
    title=f"missing model fail closed #{index}",
    index=index,
    frames=tuple(frames),
    event_windows=(),
  )


def _generate_toggle_matrix(
  rng: random.Random,
  duration_s: float,
  v_ego: float,
  curvature: float,
  index: int,
) -> ParamScenario:
  # Ensure enough frames for all 8 combos and at least one refresh per combo.
  min_frames = 8 * PARAMS_REFRESH_PERIOD + 10
  t = _time_array(max(duration_s, min_frames * DT))
  n = len(t)
  combos: list[dict[str, bool]] = []
  for a in (False, True):
    for b in (False, True):
      for c in (False, True):
        combos.append({
          "CustomLateralDemandEnabled": a,
          "LaneCenteringAssistEnabled": b,
          "CurveMemoryEnabled": c,
        })
  rng.shuffle(combos)
  segment = n // len(combos)
  frames: list[ParamFrame] = []
  model_data = _model_data_dict(curvature, v_ego)
  for i, ti in enumerate(t):
    combo_idx = min(i // segment, len(combos) - 1)
    frames.append(_base_frame(float(ti), v_ego, curvature, combos[combo_idx], model_data=model_data))
  return ParamScenario(
    kind="toggle_matrix",
    title=f"toggle matrix #{index}",
    index=index,
    frames=tuple(frames),
    event_windows=_merge_windows([(r - 5, r + 10) for r in _refresh_frames(n)]),
  )


_GENERATORS = {
  "demand_enable_cycle": _generate_demand_enable_cycle,
  "params_refresh_cadence": _generate_params_refresh_cadence,
  "params_read_fault": _generate_params_read_fault,
  "missing_model_fail_closed": _generate_missing_model_fail_closed,
  "toggle_matrix": _generate_toggle_matrix,
}


@dataclass
class ParamFuzzerConfig:
  seed: int = 1
  cases: int = 100
  kind: str | None = None
  duration_s: float = 2.0


def generate_scenarios(config: ParamFuzzerConfig) -> list[ParamScenario]:
  rng = random.Random(config.seed)
  kinds = [config.kind] if config.kind else list(DEFAULT_KINDS)
  scenarios: list[ParamScenario] = []
  for idx in range(config.cases):
    kind = rng.choice(kinds)
    v_ego = rng.uniform(15.0, 25.0)
    curvature = rng.uniform(0.0005, 0.0030) * rng.choice([-1.0, 1.0])
    scenario = _GENERATORS[kind](rng, config.duration_s, v_ego, curvature, idx)
    scenarios.append(scenario)
  return scenarios


# ---------- runner / evaluation ----------


def _run_scenario(scenario: ParamScenario) -> tuple[ParamFrameOutput, ...]:
  initial_params = scenario.frames[0].desired_params if scenario.frames else {}
  params = FakeParams(initial_params)
  adapter = LateralDemandAdapter(params=params)
  outputs: list[ParamFrameOutput] = []
  for i, frame in enumerate(scenario.frames):
    params.set_fault(frame.fault_on_refresh)
    for key, value in frame.desired_params.items():
      params.put_bool(key, value)
    refresh_frame = (i + 1) % PARAMS_REFRESH_PERIOD == 0
    try:
      output_curvature = adapter.process(
        lat_active=frame.lat_active,
        v_ego=frame.v_ego,
        roll=0.0,
        raw_curvature=frame.raw_curvature,
        measured_curvature=frame.measured_curvature,
        model_v2=_model_v2_from_dict(frame.model_data),
      )
    except Exception as exc:
      # Adapter is supposed to be fail-closed, but surface unexpected exceptions.
      raise RuntimeError(f"adapter.process raised {type(exc).__name__} at frame {i}: {exc}") from exc
    outputs.append(ParamFrameOutput(
      t=frame.t,
      v_ego=frame.v_ego,
      raw_curvature=frame.raw_curvature,
      output_curvature=float(output_curvature),
      measured_curvature=frame.measured_curvature,
      desired_params=dict(frame.desired_params),
      enabled=bool(adapter.enabled),
      lane_centering_assist_enabled=bool(adapter.lane_centering_assist_enabled),
      curve_memory_enabled=bool(adapter.curve_memory_enabled),
      refresh_frame=refresh_frame,
      params_fault=frame.fault_on_refresh,
    ))
  return tuple(outputs)


def _evaluate_structural(
  outputs: tuple[ParamFrameOutput, ...],
  scenario: ParamScenario,
) -> list[dict[str, Any]]:
  thresholds = scenario.metric_thresholds
  failures: list[dict[str, Any]] = []
  if not outputs:
    failures.append({"check": "output", "detail": "scenario produced no output frames"})
    return failures

  n = len(outputs)
  raw = np.array([o.raw_curvature for o in outputs], dtype=float)
  out = np.array([o.output_curvature for o in outputs], dtype=float)
  measured = np.array([o.measured_curvature for o in outputs], dtype=float)
  v_ego = np.array([o.v_ego for o in outputs], dtype=float)

  if not (np.all(np.isfinite(raw)) and np.all(np.isfinite(out)) and np.all(np.isfinite(measured))):
    failures.append({"check": "finite_curvature", "detail": "non-finite raw/output/measured curvature"})
  if not np.all(np.isfinite(v_ego)):
    failures.append({"check": "finite_v_ego", "detail": "non-finite v_ego"})

  lat_accel = _lateral_accel(v_ego, out)
  if not np.all(np.isfinite(lat_accel)):
    failures.append({"check": "finite_lat_accel", "detail": "non-finite output lateral acceleration"})

  if np.any(np.abs(raw) > thresholds.raw_curvature_max):
    failures.append({"check": "raw_curvature_cap", "detail": f"raw curvature exceeds {thresholds.raw_curvature_max}"})

  if np.any(np.abs(out) > thresholds.max_abs_output_curvature):
    failures.append({"check": "output_curvature_cap", "detail": f"output curvature exceeds {thresholds.max_abs_output_curvature}"})

  if n > 1:
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
  outputs: tuple[ParamFrameOutput, ...],
  scenario: ParamScenario,
) -> list[dict[str, Any]]:
  failures: list[dict[str, Any]] = []
  if not outputs:
    return failures

  kind = scenario.kind
  frames = scenario.frames

  if kind == "demand_enable_cycle":
    # Find first disabled and re-enabled observed transitions.
    disabled = [i for i, o in enumerate(outputs) if not o.enabled]
    if not disabled:
      failures.append({"check": "missing_disabled", "detail": "demand_enable_cycle never observed adapter disabled"})
    else:
      # Disabled frames should passthrough raw curvature.
      if not all(math.isclose(outputs[i].output_curvature, outputs[i].raw_curvature, abs_tol=1e-9) for i in disabled):
        failures.append({"check": "disabled_passthrough", "detail": "disabled frames did not pass through raw curvature"})
    reenabled = [i for i, o in enumerate(outputs) if o.enabled and i > (disabled[0] if disabled else 0)]
    if not reenabled:
      failures.append({"check": "missing_reenable", "detail": "demand_enable_cycle never re-enabled"})

  elif kind == "params_refresh_cadence":
    refresh_frames = [i for i, o in enumerate(outputs) if o.refresh_frame]
    if not refresh_frames:
      failures.append({"check": "missing_refresh", "detail": "params_refresh_cadence had no refresh frames"})
    else:
      first_refresh = refresh_frames[0]
      before = outputs[:first_refresh]
      after = outputs[first_refresh:]
      if before and any(o.lane_centering_assist_enabled for o in before):
        failures.append({"check": "premature_refresh", "detail": "lane_centering assist enabled before first refresh frame"})
      if after and not any(o.lane_centering_assist_enabled for o in after):
        failures.append({"check": "refresh_no_change", "detail": "lane_centering assist did not enable after first refresh frame"})

  elif kind == "params_read_fault":
    fault_frames = [i for i, f in enumerate(frames) if f.fault_on_refresh]
    if not fault_frames:
      failures.append({"check": "missing_fault", "detail": "params_read_fault had no fault_on_refresh frame"})
    else:
      fault_frame = fault_frames[0]
      after_fault = outputs[fault_frame:]
      if not after_fault:
        failures.append({"check": "no_post_fault", "detail": "no frames after injected fault"})
      elif not all(not o.enabled for o in after_fault[:20]):
        failures.append({"check": "fault_not_disabled", "detail": "adapter did not stay disabled after params read fault"})
      if not all(math.isclose(o.output_curvature, o.raw_curvature, abs_tol=1e-9) for o in after_fault[:20]):
        failures.append({"check": "fault_passthrough", "detail": "post-fault disabled frames did not pass through raw curvature"})

  elif kind == "missing_model_fail_closed":
    missing_model_frames = [i for i, f in enumerate(frames) if f.model_data is None]
    if not missing_model_frames:
      failures.append({"check": "missing_model_window", "detail": "missing_model_fail_closed had no None model frames"})
    else:
      if not all(math.isclose(outputs[i].output_curvature, outputs[i].raw_curvature, abs_tol=1e-9) for i in missing_model_frames):
        failures.append({"check": "missing_model_passthrough", "detail": "model_data=None frames did not pass through raw curvature"})

  elif kind == "toggle_matrix":
    # All 8 desired combos should appear.
    desired_combos = {tuple(sorted(f.desired_params.items())) for f in frames}
    if len(desired_combos) < 8:
      failures.append({"check": "combo_coverage", "detail": f"toggle_matrix covered only {len(desired_combos)}/8 desired combos"})
    observed_combos = {(o.enabled, o.lane_centering_assist_enabled, o.curve_memory_enabled) for o in outputs}
    if len(observed_combos) < 8:
      failures.append({"check": "observed_combo_coverage", "detail": f"toggle_matrix observed only {len(observed_combos)}/8 adapter combos"})
    # When adapter observes master disabled, output should be raw passthrough.
    observed_disabled_frames = [i for i, o in enumerate(outputs) if not o.enabled]
    if observed_disabled_frames:
      if not all(math.isclose(outputs[i].output_curvature, outputs[i].raw_curvature, abs_tol=1e-9) for i in observed_disabled_frames):
        failures.append({"check": "toggle_disabled_passthrough", "detail": "observed disabled frames did not pass through raw curvature"})
    else:
      failures.append({"check": "no_observed_disabled", "detail": "toggle_matrix never observed adapter disabled"})

  return failures


def evaluate_scenario(scenario: ParamScenario) -> ParamResult:
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
    observed_enabled = [o.enabled for o in outputs]
    metrics["enabled_frames"] = sum(observed_enabled)
    metrics["disabled_frames"] = len(observed_enabled) - sum(observed_enabled)
    metrics["refresh_frames"] = sum(1 for o in outputs if o.refresh_frame)
    metrics["observed_param_combos"] = len({(o.enabled, o.lane_centering_assist_enabled, o.curve_memory_enabled) for o in outputs})

  return ParamResult(
    scenario=scenario,
    outputs=outputs,
    structural_failures=structural_failures,
    event_failures=event_failures,
    metrics=_sanitize(metrics),
  )


# ---------- serialization / CLI ----------


def scenario_to_dict(scenario: ParamScenario, seed: int | None = None) -> dict[str, Any]:
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


def scenario_from_dict(data: dict[str, Any]) -> ParamScenario:
  return ParamScenario(
    kind=str(data["kind"]),
    title=str(data["title"]),
    index=int(data["index"]),
    frames=tuple(ParamFrame.from_dict(frame) for frame in data["frames"]),
    event_windows=tuple((int(w[0]), int(w[1])) for w in data.get("event_windows", [])),
    thresholds=ParamThresholds.from_dict(data.get("thresholds", {})),
  )


def artifact_to_dict(result: ParamResult, seed: int | None, index: int | None) -> dict[str, Any]:
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
        "output_curvature": o.output_curvature,
        "measured_curvature": o.measured_curvature,
        "desired_params": o.desired_params,
        "enabled": o.enabled,
        "lane_centering_assist_enabled": o.lane_centering_assist_enabled,
        "curve_memory_enabled": o.curve_memory_enabled,
        "refresh_frame": o.refresh_frame,
        "params_fault": o.params_fault,
      }
      for o in result.outputs
    ],
    "structural_failure_checks": [f["check"] for f in result.structural_failures],
    "event_failure_checks": [f["check"] for f in result.event_failures],
    "metrics": _sanitize(result.metrics),
    "overall_valid": result.valid,
  }


def write_artifact(result: ParamResult, artifact_dir: Path, seed: int | None, index: int | None) -> Path:
  artifact_dir.mkdir(parents=True, exist_ok=True)
  filename = f"lateral_params_failure_{result.scenario.kind}_seed{seed}_idx{index}.json"
  path = artifact_dir / filename
  path.write_text(json.dumps(_sanitize(artifact_to_dict(result, seed, index)), indent=2, sort_keys=True, allow_nan=False))
  return path


def load_artifact(path: str | Path) -> dict[str, Any]:
  return json.loads(Path(path).read_text())


def replay_artifact(path: str | Path) -> ParamResult:
  data = load_artifact(path)
  scenario = scenario_from_dict(data["scenario"])
  return evaluate_scenario(scenario)


def _render_scenario_snippet(scenario: ParamScenario) -> str:
  return f"# kind: {scenario.kind}\nParamScenario(title={scenario.title!r}, frames=[...{len(scenario.frames)} frames...])"


def main() -> None:
  parser = argparse.ArgumentParser(description="Synthetic lateral demand params/config structural fuzzer.")
  parser.add_argument("--seed", type=int, default=1)
  parser.add_argument("--cases", type=int, default=100)
  parser.add_argument("--kind", choices=DEFAULT_KINDS, help="Params fuzzer kind")
  parser.add_argument("--duration", type=float, default=2.0, help="Scenario duration in seconds")
  parser.add_argument("--json", action="store_true", help="Emit JSON instead of text")
  parser.add_argument("--fail-fast", action="store_true", help="Stop after the first failure")
  parser.add_argument("--artifact-dir", type=str, default=None, help="Directory to write failure artifacts")
  parser.add_argument("--replay", type=str, default=None, help="Replay a params fuzzer artifact JSON file")
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
      print(f"Replayed {args.replay}: valid={result.valid} structural={len(result.structural_failures)} event={len(result.event_failures)}")
      for failure in result.structural_failures:
        print(f"  structural: {failure['check']}: {failure['detail']}")
      for failure in result.event_failures:
        print(f"  event: {failure['check']}: {failure['detail']}")
    sys.exit(0 if result.valid else 1)

  config = ParamFuzzerConfig(seed=args.seed, cases=args.cases, kind=args.kind, duration_s=args.duration)
  scenarios = list(generate_scenarios(config))
  results: list[ParamResult] = []
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
    print(
      f"Drive Lab lateral params fuzz seed={args.seed} cases={len(results)} "
      f"kind={args.kind or 'default'} duration={args.duration}s dt={DT}s failures={len(failures)}"
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
