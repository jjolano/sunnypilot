#!/usr/bin/env python3
"""Synthetic closed-loop lateral structural stability plant.

This module is a deliberately simplified, deterministic test plant for fuzzing
lateral controller structure. It is *not* a high-fidelity model of the production
steering stack; it captures only enough steering mechanics (actuator delay, rate
limit, stiction, backlash, tire lag/saturation) to expose synthetic instability,
sign errors, divergence, and excessive oscillation.

Positive desired curvature should produce a positive actual curvature response
for the default tuning; zero desired curvature with no disturbance should stay
near zero.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np


MAX_SIMULATION_DT_S = 0.05


@dataclass(frozen=True)
class LateralPlantConfig:
  """Configuration for the synthetic structural stability plant."""

  dt_s: float = 0.05
  speed_mps: float = 20.0
  duration_s: float = 10.0
  steering_gain: float = 3200.0  # deg steering per 1/m curvature (open-loop scale)
  controller_gain: float = 1.5  # closed-loop error gain
  controller_damping: float = 0.75  # damping on curvature rate (steering command)
  actuator_delay_s: float = 0.15
  actuator_rate_limit_deg_s: float = 180.0
  steering_stiction_deg: float = 0.0
  steering_backlash_deg: float = 0.0
  tire_gain: float = 1.0
  tire_lag_s: float = 0.25
  tire_saturation_curvature: float = 0.02
  max_steering_angle_deg: float = 360.0
  initial_curvature: float = 0.0
  initial_steering_deg: float = 0.0

  def to_dict(self) -> dict[str, Any]:
    return asdict(self)

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> LateralPlantConfig:
    fields = cls.__dataclass_fields__
    return cls(**{key: data[key] for key in fields if key in data})


@dataclass(frozen=True)
class LateralPlantTrace:
  """Time-series output of a synthetic plant run."""

  t: tuple[float, ...]
  v_ego: tuple[float, ...]
  desired_curvature: tuple[float, ...]
  measured_curvature: tuple[float, ...]
  steering_command_deg: tuple[float, ...]
  actuator_steering_deg: tuple[float, ...]
  actual_curvature: tuple[float, ...]

  def to_dict(self) -> dict[str, Any]:
    return asdict(self)

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> LateralPlantTrace:
    return cls(
      t=_float_tuple(data.get("t", ())),
      v_ego=_float_tuple(data.get("v_ego", data.get("vEgo", ()))),
      desired_curvature=_float_tuple(data.get("desired_curvature", data.get("desiredCurvature", ()))),
      measured_curvature=_float_tuple(data.get("measured_curvature", data.get("measuredCurvature", ()))),
      steering_command_deg=_float_tuple(data.get("steering_command_deg", data.get("steeringCommandDeg", ()))),
      actuator_steering_deg=_float_tuple(data.get("actuator_steering_deg", data.get("actuatorSteeringDeg", ()))),
      actual_curvature=_float_tuple(data.get("actual_curvature", data.get("actualCurvature", ()))),
    )


@dataclass(frozen=True)
class LateralPlantSample:
  """Single time-step plant state."""

  t: float
  v_ego: float
  desired_curvature: float
  measured_curvature: float
  steering_command_deg: float
  actuator_steering_deg: float
  actual_curvature: float


@dataclass(frozen=True)
class LateralPlantResult:
  """Result of running a scenario through the synthetic plant."""

  config: LateralPlantConfig
  trace: LateralPlantTrace
  samples: tuple[LateralPlantSample, ...]

  def to_dict(self) -> dict[str, Any]:
    return {
      "config": self.config.to_dict(),
      "trace": self.trace.to_dict(),
      "samples": [sample.__dict__ for sample in self.samples],
    }

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> LateralPlantResult:
    return cls(
      config=LateralPlantConfig.from_dict(data["config"]),
      trace=LateralPlantTrace.from_dict(data["trace"]),
      samples=tuple(LateralPlantSample(**sample) for sample in data.get("samples", [])),
    )


def run_lateral_plant(
  desired_curvature: np.ndarray,
  v_ego: np.ndarray | None = None,
  config: LateralPlantConfig | None = None,
) -> LateralPlantResult:
  """Run the synthetic closed-loop lateral plant.

  The controller is a damped proportional feedback on curvature error:
    error = desired - actual
    command_deg = steering_gain * (controller_gain * error - controller_damping * curvature_rate)
  where curvature_rate is a one-sample backward difference of actual curvature.

  The command then passes through delay, rate-limit, stiction, and backlash to
  become the actuator steering angle. Finally a first-order tire lag plus
  saturation converts actuator angle back to actual curvature.
  """
  config = config or LateralPlantConfig()
  dt = min(max(config.dt_s, 1e-6), MAX_SIMULATION_DT_S)
  if dt != config.dt_s:
    config = replace(config, dt_s=dt)
  duration = max(config.duration_s, dt)
  t = np.arange(0.0, duration + dt * 0.5, dt, dtype=float)
  n = t.size

  if v_ego is None:
    v_ego_arr = np.full(n, config.speed_mps, dtype=float)
  else:
    v_ego_arr = np.asarray(v_ego, dtype=float)
    if v_ego_arr.size != n:
      v_ego_arr = _resample_or_fill(tuple(v_ego_arr), n, config.speed_mps)

  desired = np.asarray(desired_curvature, dtype=float)
  if desired.size != n:
    desired = _resample_or_fill(tuple(desired), n, 0.0)

  delay_frames = max(0, int(round(config.actuator_delay_s / dt)))
  command = np.zeros(n, dtype=float)

  actual = np.zeros(n, dtype=float)
  actual[0] = config.initial_curvature
  actuator = np.zeros(n, dtype=float)
  actuator[0] = config.initial_steering_deg

  steering_gain = max(config.steering_gain, 1e-6)
  max_steering = max(config.max_steering_angle_deg, 1e-3)

  backlash = max(config.steering_backlash_deg, 0.0)
  stiction = max(config.steering_stiction_deg, 0.0)
  rate_limit = max(config.actuator_rate_limit_deg_s, 1e-3)
  lag = max(config.tire_lag_s, 1e-3)
  saturation = max(config.tire_saturation_curvature, 1e-6)

  # Interleaved closed-loop simulation: command, actuator, and tire state are
  # updated one timestep at a time so the controller sees the previous actual
  # curvature.
  for i in range(1, n):
    dt_step = max(float(t[i] - t[i - 1]), 1e-6)
    curvature_rate = (actual[i - 1] - actual[max(0, i - 2)]) / dt_step
    error = desired[i - 1] - actual[i - 1]
    raw_command = steering_gain * (config.controller_gain * error - config.controller_damping * curvature_rate)
    command[i] = float(np.clip(raw_command, -max_steering, max_steering))

    # Actuator/perception delay: look back `delay_frames` in the command history.
    delayed_idx = max(0, i - delay_frames)
    delayed_command = command[delayed_idx]

    # Steering mechanics: stiction, backlash, rate limit.
    delta = delayed_command - actuator[i - 1]
    if abs(delta) <= stiction:
      target = actuator[i - 1]
    else:
      target = delayed_command - math.copysign(min(backlash, abs(delta)), delta)
    max_step = rate_limit * dt_step
    actuator[i] = actuator[i - 1] + float(np.clip(target - actuator[i - 1], -max_step, max_step))

    # Tire response with lag and saturation.
    target = config.tire_gain * actuator[i] / steering_gain
    target = float(np.clip(target, -saturation, saturation))
    alpha = dt_step / (dt_step + lag)
    actual[i] = actual[i - 1] + alpha * (target - actual[i - 1])

  # Measured curvature mirrors actual curvature in this deterministic synthetic
  # plant; later phases can add explicit sensor-noise scenarios if needed.
  measured = actual.copy()

  trace = LateralPlantTrace(
    t=tuple(float(v) for v in t),
    v_ego=tuple(float(v) for v in v_ego_arr),
    desired_curvature=tuple(float(v) for v in desired),
    measured_curvature=tuple(float(v) for v in measured),
    steering_command_deg=tuple(float(v) for v in command),
    actuator_steering_deg=tuple(float(v) for v in actuator),
    actual_curvature=tuple(float(v) for v in actual),
  )
  samples = tuple(
    LateralPlantSample(
      t=trace.t[i],
      v_ego=trace.v_ego[i],
      desired_curvature=trace.desired_curvature[i],
      measured_curvature=trace.measured_curvature[i],
      steering_command_deg=trace.steering_command_deg[i],
      actuator_steering_deg=trace.actuator_steering_deg[i],
      actual_curvature=trace.actual_curvature[i],
    )
    for i in range(n)
  )
  return LateralPlantResult(config=config, trace=trace, samples=samples)


def _resample_or_fill(values: tuple[float, ...], size: int, fill: float) -> np.ndarray:
  arr = np.array(values, dtype=float)
  if arr.size == size:
    return arr
  if arr.size == 0:
    return np.full(size, fill, dtype=float)
  return np.interp(np.linspace(0.0, arr.size - 1, size), np.arange(arr.size), arr)


def _float_tuple(values: Any) -> tuple[float, ...]:
  return tuple(float(v) for v in values)
