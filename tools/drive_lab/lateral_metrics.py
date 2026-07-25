#!/usr/bin/env python3
"""Structural stability metrics for the synthetic lateral fuzzer.

These checks are intentionally conservative and scenario/speed-aware. They are
meant to catch obvious synthetic instability (NaN/inf, divergence, excess
oscillation, impossible steering rate/jerk, saturation) while keeping false
positives low on the deliberately simple test plant.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from openpilot.tools.drive_lab.lateral_plant import LateralPlantConfig, LateralPlantTrace
from openpilot.tools.drive_lab.metrics import EvaluationMetric, EvaluationResult, ScenarioFailure


OSCILLATION_THRESHOLD_WINDOW_S = 5.0


# Sign is only meaningful when the demand actually goes somewhere. Straight-line
# disturbance traces average |mean desired| <= 7.8e-5 1/m, where the sign of the mean is
# noise; real directional scenarios are an order of magnitude above this.
SIGN_MIN_MEAN_DEMAND_CURVATURE = 2e-4
# ...and only when the response commits to the wrong side. A mean response near zero is
# under-response, not inversion, and the tracking-error checks own that.
SIGN_COMMITTED_RESPONSE_FRAC = 0.25

@dataclass(frozen=True)
class LateralMetricThresholds:
  """Thresholds for lateral structural checks.

  Defaults are chosen to flag obvious synthetic instability without rejecting
  the normal lag/overshoot of the simple closed-loop plant.
  """

  max_abs_tracking_error: float = 0.008  # 1/m
  max_abs_steering_rate: float = 200.0  # deg/s
  max_abs_lateral_jerk: float = 15.0  # m/s^3
  max_saturation_fraction: float = 0.25
  max_zero_drift_curvature: float = 2e-4  # 1/m
  max_oscillation_reversals: int = 25
  max_final_tracking_error: float = 0.005  # 1/m

  def to_dict(self) -> dict[str, Any]:
    return asdict(self)

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> LateralMetricThresholds:
    fields = cls.__dataclass_fields__
    return cls(**{key: data[key] for key in fields if key in data})


def _derivative(t: np.ndarray, y: np.ndarray) -> np.ndarray:
  out = np.zeros_like(y)
  dt = np.diff(t)
  dy = np.diff(y)
  valid = np.isfinite(dt) & (dt > 1e-6) & np.isfinite(dy)
  vals = np.zeros_like(dy)
  vals[valid] = dy[valid] / dt[valid]
  out[1:] = vals
  return out


def _sign_flip_count(x: np.ndarray, eps: float) -> int:
  ok = np.isfinite(x) & (np.abs(x) > eps)
  signs = np.sign(x[ok])
  return int(np.sum(signs[1:] != signs[:-1])) if signs.size > 1 else 0


def _lateral_acceleration(v_ego: np.ndarray, curvature: np.ndarray) -> np.ndarray:
  """Lateral acceleration a_lat = v^2 * kappa."""
  return v_ego * v_ego * curvature


def _lateral_jerk_threshold(base_threshold: float, mean_speed_mps: float) -> float:
  # Jerk is measured on lateral acceleration, so the same curvature response scales roughly
  # with v^2. Keep a floor for low-speed numerical noise; other metrics catch tracking,
  # steering-rate, saturation, and oscillation failures independently.
  speed_ratio = max(float(mean_speed_mps), 0.0) / 20.0
  speed_factor = max(0.8, speed_ratio * speed_ratio)
  return float(base_threshold) * speed_factor


def _oscillation_reversal_limit(base_limit: int, evaluation_span_s: float) -> int:
  if base_limit <= 0:
    return 0
  span_factor = max(1.0, float(evaluation_span_s) / OSCILLATION_THRESHOLD_WINDOW_S)
  return int(math.ceil(float(base_limit) * span_factor))


def evaluate_lateral_trace(
  scenario_id: str,
  trace: LateralPlantTrace,
  config: LateralPlantConfig,
  thresholds: LateralMetricThresholds | None = None,
  scenario_kind: str = "unknown",
) -> EvaluationResult:
  """Evaluate a synthetic lateral plant trace for structural stability.

  Returns an EvaluationResult with metrics and any ScenarioFailure items.
  """
  thresholds = thresholds or LateralMetricThresholds()
  failures: list[ScenarioFailure] = []
  metrics: list[EvaluationMetric] = []

  t = np.array(trace.t, dtype=float)
  desired = np.array(trace.desired_curvature, dtype=float)
  actual = np.array(trace.actual_curvature, dtype=float)
  measured = np.array(trace.measured_curvature, dtype=float)
  command = np.array(trace.steering_command_deg, dtype=float)
  actuator = np.array(trace.actuator_steering_deg, dtype=float)
  v_ego = np.array(trace.v_ego, dtype=float)

  if t.size == 0:
    failures.append(ScenarioFailure("output", "plant produced no output"))
    return EvaluationResult(scenario_id, False, failures, tuple(metrics))

  sizes = {
    "t": t.size,
    "desired_curvature": desired.size,
    "actual_curvature": actual.size,
    "measured_curvature": measured.size,
    "steering_command_deg": command.size,
    "actuator_steering_deg": actuator.size,
    "v_ego": v_ego.size,
  }
  if len(set(sizes.values())) != 1:
    failures.append(ScenarioFailure("output", f"inconsistent trace lengths {sizes}"))
    return EvaluationResult(scenario_id, False, failures, tuple(metrics))

  # Finite output check.
  has_nonfinite = not (
    np.all(np.isfinite(t))
    and np.all(np.isfinite(desired))
    and np.all(np.isfinite(actual))
    and np.all(np.isfinite(measured))
    and np.all(np.isfinite(command))
    and np.all(np.isfinite(actuator))
    and np.all(np.isfinite(v_ego))
  )
  metrics.append(EvaluationMetric("has_nonfinite_output", float(has_nonfinite), "", not has_nonfinite))
  if has_nonfinite:
    failures.append(ScenarioFailure("finite", "output contains NaN or infinite values"))

  # Sign response: a controller that steers the wrong way is a structural failure, not
  # a note. This previously recorded sign_ok on the metric and returned a passing result
  # anyway, and only examined desired > 0 — so a trace whose demand was entirely negative
  # was never sign-checked at all.
  #
  # Compare mean directions over the whole trace. A per-sample desired>0 mask cannot
  # tell inversion from lag — in an alternating maneuver (ISO 3888) the plant is still
  # unwinding the previous direction when demand flips, and in a noisy trace with a
  # dominant direction the brief opposite blips never move the plant. Correlation is no
  # better here: these traces are lag-dominated and only weakly correlated (median 0.04
  # over 180 seeded scenarios), so inverting one stays well above any usable threshold.
  #
  # Mean direction is robust to both, and is polarity-symmetric so negative-only demand
  # is covered. Two floors keep it honest: the demand must actually go somewhere, and
  # the response must be committed to the wrong side rather than merely sitting near
  # zero (which is under-response, caught by the tracking-error checks instead).
  if t.size >= 5 and not has_nonfinite:
    mean_desired = float(np.mean(desired))
    mean_actual = float(np.mean(actual))
    if abs(mean_desired) >= SIGN_MIN_MEAN_DEMAND_CURVATURE:
      committed = abs(mean_actual) >= SIGN_COMMITTED_RESPONSE_FRAC * abs(mean_desired)
      inverted = committed and (mean_desired * mean_actual) < 0.0
      metrics.append(EvaluationMetric("mean_actual_curvature", mean_actual, "1/m", not inverted))
      if inverted:
        failures.append(ScenarioFailure(
          "sign", f"mean desired {mean_desired:.6f} 1/m produced mean actual {mean_actual:.6f} 1/m"))

  # Zero-input drift: when desired curvature is near zero for the whole trace,
  # actual curvature must stay near zero.
  desired_span = float(np.max(desired) - np.min(desired))
  desired_mag = float(np.max(np.abs(desired)))
  if desired_span < 1e-5 and desired_mag < 1e-5:
    max_drift = float(np.max(np.abs(actual)))
    drift_ok = max_drift <= thresholds.max_zero_drift_curvature
    metrics.append(EvaluationMetric("max_zero_drift_curvature", max_drift, "1/m", drift_ok))
    if not drift_ok and not has_nonfinite:
      failures.append(ScenarioFailure("drift", f"zero-input drift {max_drift:.6f} 1/m exceeds threshold"))

  # Tracking error / divergence.
  if t.size > 1:
    tracking_error = actual - desired
    abs_error = np.abs(tracking_error)
    max_abs_error = float(np.max(abs_error))
    mean_abs_error = float(np.mean(abs_error))
    final_error = float(abs_error[-1])
    metrics.extend([
      EvaluationMetric("max_abs_tracking_error", max_abs_error, "1/m", max_abs_error <= thresholds.max_abs_tracking_error),
      EvaluationMetric("mean_abs_tracking_error", mean_abs_error, "1/m", True),
      EvaluationMetric("final_abs_tracking_error", final_error, "1/m", final_error <= thresholds.max_final_tracking_error),
    ])
    if max_abs_error > thresholds.max_abs_tracking_error and not has_nonfinite:
      failures.append(ScenarioFailure("tracking", f"max tracking error {max_abs_error:.5f} 1/m"))
    if final_error > thresholds.max_final_tracking_error and not has_nonfinite:
      failures.append(ScenarioFailure("settle", f"final tracking error {final_error:.5f} 1/m"))

  # Steering rate.
  steering_rate = _derivative(t, actuator)
  max_abs_steering_rate = float(np.max(np.abs(steering_rate))) if t.size > 1 else 0.0
  rate_ok = max_abs_steering_rate <= thresholds.max_abs_steering_rate
  metrics.append(EvaluationMetric("max_abs_steering_rate", max_abs_steering_rate, "deg/s", rate_ok))
  if not rate_ok and not has_nonfinite:
    failures.append(ScenarioFailure("steering_rate", f"max steering rate {max_abs_steering_rate:.1f} deg/s"))

  # Lateral jerk.
  lateral_accel = _lateral_acceleration(v_ego, actual)
  lateral_jerk = _derivative(t, lateral_accel)
  max_abs_lateral_jerk = float(np.max(np.abs(lateral_jerk))) if t.size > 1 else 0.0
  jerk_threshold = _lateral_jerk_threshold(thresholds.max_abs_lateral_jerk, float(np.mean(v_ego)))
  jerk_ok = max_abs_lateral_jerk <= jerk_threshold
  metrics.append(EvaluationMetric("max_abs_lateral_jerk", max_abs_lateral_jerk, "m/s^3", jerk_ok))
  if not jerk_ok and not has_nonfinite:
    failures.append(ScenarioFailure("lateral_jerk", f"max lateral jerk {max_abs_lateral_jerk:.2f} m/s^3"))

  # Saturation fraction: fraction of samples clipped at max steering angle.
  max_steering = max(config.max_steering_angle_deg, 1e-3)
  saturated = np.abs(command) >= max_steering - 1e-3
  saturation_fraction = float(np.mean(saturated)) if command.size else 0.0
  saturation_ok = saturation_fraction <= thresholds.max_saturation_fraction
  metrics.append(EvaluationMetric("saturation_fraction", saturation_fraction, "", saturation_ok))
  if not saturation_ok and not has_nonfinite:
    failures.append(ScenarioFailure("saturation", f"command saturated {saturation_fraction:.2%} of scenario"))

  # Excess oscillation: count actuator reversals in the latter half of the
  # scenario that are not explained by desired-curvature reversals. This avoids
  # flagging normal settling transients and sinusoidal/S-curve tracking while
  # still catching instability-induced hunting.
  if t.size > 20:
    late = t >= 0.5 * float(np.max(t))
    actuator_reversals = _sign_flip_count(actuator[late], 0.5)
    desired_reversals = _sign_flip_count(desired[late], 1e-5)
    excess_reversals = max(0, actuator_reversals - desired_reversals)
    late_t = t[late]
    evaluation_span_s = float(late_t[-1] - late_t[0]) if late_t.size > 1 else 0.0
    oscillation_limit = _oscillation_reversal_limit(thresholds.max_oscillation_reversals, evaluation_span_s)
    oscillation_ok = excess_reversals <= oscillation_limit
    metrics.append(EvaluationMetric("excess_steering_reversals", float(excess_reversals), "count", oscillation_ok))
    if not oscillation_ok and not has_nonfinite:
      failures.append(
        ScenarioFailure(
          "oscillation",
          f"{excess_reversals} excess steering reversals in second half "
          f"(limit={oscillation_limit}, span={evaluation_span_s:.1f}s, actuator={actuator_reversals}, desired={desired_reversals})",
        )
      )

  return EvaluationResult(scenario_id, not failures, failures, tuple(metrics))
