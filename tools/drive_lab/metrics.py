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


def evaluate_maneuver_output(scenario_id: str, valid: bool, output: np.ndarray, max_normal_jerk: float = 8.0) -> EvaluationResult:
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

  max_abs_jerk = 0.0
  has_jerk_metric = False
  if len(accel) > 2:
    dt = np.diff(time_s)
    valid_dt = np.isfinite(dt) & (dt > 1e-6)
    accel_delta = np.diff(accel)
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
