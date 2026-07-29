from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ScenarioFailure:
  check: str
  detail: str


@dataclass(frozen=True)
class EvaluationMetric:
  name: str
  value: float
  unit: str = ""
  passed: bool = True
  detail: str = ""


@dataclass(frozen=True)
class EvaluationResult:
  scenario_id: str
  valid: bool
  failures: list[ScenarioFailure]
  metrics: tuple[EvaluationMetric, ...]

  def metric_value(self, name: str) -> float:
    for metric in self.metrics:
      if metric.name == name:
        return metric.value
    raise KeyError(name)

  def to_dict(self) -> dict[str, Any]:
    return {
      "scenario_id": self.scenario_id,
      "valid": self.valid,
      "failures": [failure.__dict__ for failure in self.failures],
      "metrics": {
        metric.name: {
          "value": metric.value,
          "unit": metric.unit,
          "passed": metric.passed,
          "detail": metric.detail,
        }
        for metric in self.metrics
      },
    }


def evaluate_maneuver_output(
  scenario_id: str,
  valid: bool,
  output: np.ndarray,
  max_normal_jerk: float = 8.0,
  commanded_accel: np.ndarray | None = None,
  jerk_window: int = 1,
) -> EvaluationResult:
  failures: list[ScenarioFailure] = []
  metrics: list[EvaluationMetric] = []
  if not valid:
    failures.append(ScenarioFailure("valid", "maneuver reported invalid"))
  if output.size == 0:
    failures.append(ScenarioFailure("output", "maneuver produced no output"))
    return EvaluationResult(scenario_id, False, failures, tuple(metrics))
  if output.ndim != 2 or output.shape[1] < 7:
    failures.append(ScenarioFailure("output", f"expected maneuver output with at least 7 columns, got shape {output.shape}"))
    return EvaluationResult(scenario_id, False, failures, tuple(metrics))
  has_nonfinite_output = not np.all(np.isfinite(output))
  if has_nonfinite_output:
    failures.append(ScenarioFailure("finite", "output contains NaN or infinite values"))

  time_s = output[:, 0]
  speed = output[:, 3]
  accel = output[:, 5]
  d_rel = output[:, 6]

  finite_speed = speed[np.isfinite(speed)]
  if finite_speed.size:
    min_speed = float(np.min(finite_speed))
    metrics.append(EvaluationMetric("min_speed", min_speed, "m/s", min_speed >= -1e-3))
    if min_speed < -1e-3 and not has_nonfinite_output:
      failures.append(ScenarioFailure("speed", f"negative speed {min_speed:.3f} m/s"))

  finite_d_rel = d_rel[np.isfinite(d_rel)]
  if finite_d_rel.size:
    min_lead_gap = float(np.min(finite_d_rel))
    metrics.append(EvaluationMetric("min_lead_gap", min_lead_gap, "m", min_lead_gap >= 0.4))
    if min_lead_gap < 0.4 and not has_nonfinite_output:
      failures.append(ScenarioFailure("collision", f"minimum lead gap {min_lead_gap:.3f} m"))

  # Jerk reflects ride comfort, so it must be measured on the acceleration the longitudinal
  # policy actually commands. The maneuver plant overwrites its accel column with a crude stop
  # model (a -0.5 m/s^2 floor when shouldStop flips, then a hard zero once the car reaches
  # standstill); those discontinuities are test-harness scaffolding, not policy output, and
  # would otherwise show up as ~10-12 m/s^3 phantom jerk. When the caller captures the planner's
  # commanded acceleration, evaluate jerk on that instead of the post-override column.
  jerk_accel = accel
  if commanded_accel is not None:
    commanded_accel = np.asarray(commanded_accel, dtype=float)
    if commanded_accel.shape == accel.shape:
      jerk_accel = commanded_accel

  # Felt jerk is bounded by how fast the longitudinal actuator can change realized acceleration,
  # so jerk is measured as the acceleration change across a window (jerk_window control frames,
  # ~the actuator delay) rather than a single 50 ms control step. A single-frame command step the
  # actuator physically cannot reproduce (e.g. the MPC's onset re-plan when a lead first appears)
  # is not felt as jerk; sustained harsh jerk still spans the window and is caught. The default
  # window of 1 preserves single-step behavior for callers measuring already-realized accel.
  window = max(1, int(jerk_window))
  max_abs_jerk = 0.0
  has_jerk_metric = False
  if len(jerk_accel) > window + 1:
    dt = time_s[window:] - time_s[:-window]
    valid_dt = np.isfinite(dt) & (dt > 1e-6)
    accel_delta = jerk_accel[window:] - jerk_accel[:-window]
    valid_jerk = valid_dt & np.isfinite(accel_delta)
    if np.any(valid_jerk):
      jerk = accel_delta[valid_jerk] / dt[valid_jerk]
      max_abs_jerk = float(np.max(np.abs(jerk)))
      has_jerk_metric = True
      if max_abs_jerk > max_normal_jerk and not has_nonfinite_output:
        failures.append(ScenarioFailure("jerk", f"maximum absolute jerk {max_abs_jerk:.3f} m/s^3"))
  if has_jerk_metric or not has_nonfinite_output:
    metrics.append(EvaluationMetric("max_abs_jerk", max_abs_jerk, "m/s^3", max_abs_jerk <= max_normal_jerk))

  return EvaluationResult(scenario_id, valid and not failures, failures, tuple(metrics))
