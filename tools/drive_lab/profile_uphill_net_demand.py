#!/usr/bin/env python3
"""Profile relative grade and net demand from an extracted route NPZ.

This report intentionally cannot mint an apply profile: a shift/no-shift-labeled
steep corpus is required before `UphillNetDemandGradeProfile` may be marked calibrated.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from openpilot.sunnypilot.custom.longitudinal.net_demand_cap import GRAVITY, fit_coast_samples


def _array(data: Any, *names: str, default: float | None = None) -> np.ndarray:
  for name in names:
    if name in data:
      return np.asarray(data[name], dtype=np.float64)
  if default is None:
    raise KeyError(f"missing any of {names}")
  return np.asarray(default, dtype=np.float64)


def _nearest(source_t: np.ndarray, values: np.ndarray, target_t: np.ndarray) -> np.ndarray:
  if len(source_t) == 0:
    return np.full(len(target_t), np.nan)
  right = np.searchsorted(source_t, target_t, side="left")
  right = np.clip(right, 0, len(source_t) - 1)
  left = np.maximum(0, right - 1)
  choose_left = np.abs(target_t - source_t[left]) <= np.abs(source_t[right] - target_t)
  return values[np.where(choose_left, left, right)]


def _decimate(t: np.ndarray, mask: np.ndarray, period_s: float = 0.5) -> np.ndarray:
  selected = []
  last = -np.inf
  for index in np.flatnonzero(mask):
    if t[index] - last >= period_s:
      selected.append(index)
      last = t[index]
  return np.asarray(selected, dtype=np.int64)


def _longest_duration(t: np.ndarray, mask: np.ndarray) -> float:
  longest = current = 0.0
  for i in range(1, len(t)):
    if mask[i] and mask[i - 1]:
      current += max(0.0, t[i] - t[i - 1])
      longest = max(longest, current)
    else:
      current = 0.0
  return longest


def _stats(values: np.ndarray) -> dict[str, float | int]:
  finite = values[np.isfinite(values)]
  if len(finite) == 0:
    return {"count": 0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "over_1_2_percent": 0.0}
  p50, p90, p95 = np.percentile(finite, [50, 90, 95])
  return {
    "count": int(len(finite)),
    "p50": float(p50),
    "p90": float(p90),
    "p95": float(p95),
    "over_1_2_percent": float(100.0 * np.mean(finite > 1.2)),
  }


def analyze_npz(path: str | Path) -> dict[str, Any]:
  with np.load(path) as data:
    cs_t = _array(data, "cs_t")
    v_ego = _array(data, "cs_vEgo", "cs_v")
    a_ego = _array(data, "cs_aEgo", "cs_a")
    gas = _array(data, "cs_gasPressed", "cs_gp") > 0.5
    brake = _array(data, "cs_brakePressed", "cs_bp") > 0.5
    cc_t = _array(data, "cc_t")
    pitch = _nearest(cc_t, _array(data, "cc_pitch"), cs_t)
    long_active = _nearest(cc_t, _array(data, "cc_longActive", "cc_la"), cs_t) > 0.5
    accel_request = _nearest(cc_t, _array(data, "cc_accel", "cc_ac"), cs_t)
    planner_request = (
      _nearest(_array(data, "sp_t"), _array(data, "sp_upBefore"), cs_t)
      if "sp_upBefore" in data else accel_request
    )
    toyota_request = (
      _nearest(_array(data, "co_t"), _array(data, "co_accel"), cs_t)
      if "co_accel" in data else accel_request
    )
    lead_present = (
      _nearest(_array(data, "rs_t"), _array(data, "rs_status"), cs_t) > 0.5
      if "rs_status" in data else np.zeros(len(cs_t), dtype=bool)
    )
    trace_counts = {
      "would_cap": int(np.count_nonzero(_array(data, "sp_upWouldCap"))) if "sp_upWouldCap" in data else 0,
      "applied": int(np.count_nonzero(_array(data, "sp_upApplied"))) if "sp_upApplied" in data else 0,
    }

  eligible = (~long_active) & (~gas) & (~brake) & (v_ego >= 3.0) & np.isfinite(pitch) & np.isfinite(a_ego)
  indexes = _decimate(cs_t, eligible)
  fit = fit_coast_samples((float(pitch[i]), float(a_ego[i]), float(v_ego[i])) for i in indexes)
  if not fit.ready:
    return {
      "route": str(path),
      "fit": fit.__dict__,
      "apply_go": False,
      "no_go_reasons": ["manual coast fit is not ready", "no labeled steep-climb shift/no-shift corpus"],
    }

  relative_pitch = pitch - fit.pitch_zero_rad
  grade_percent = 100.0 * np.tan(relative_pitch)
  grade_accel = GRAVITY * np.sin(relative_pitch)
  measured_net = a_ego + grade_accel
  planner_net = planner_request + grade_accel
  actuator_net = accel_request + grade_accel
  toyota_net = toyota_request + grade_accel
  steep = grade_percent > 3.0
  engaged = long_active & ~gas & ~brake
  report = {
    "route": str(path),
    "fit": fit.__dict__,
    "raw_pitch_median_rad": float(np.nanmedian(pitch)),
    "grade_percent_p95": float(np.nanpercentile(grade_percent, 95)),
    "longest_6_percent_climb_s": float(_longest_duration(cs_t, grade_percent >= 6.0)),
    "manual_steep_measured_net": _stats(measured_net[steep & ~long_active]),
    "engaged_steep_measured_net": _stats(measured_net[steep & engaged]),
    "engaged_leadless_steep_measured_net": _stats(measured_net[steep & engaged & ~lead_present]),
    "engaged_steep_planner_net": _stats(planner_net[steep & engaged]),
    "engaged_steep_car_control_net": _stats(actuator_net[steep & engaged]),
    "engaged_steep_toyota_request_net": _stats(toyota_net[steep & engaged]),
    "trace_counts": trace_counts,
    "profile_evidence": {
      "version": 1,
      "calibrated": False,
      "pitch_zero_rad": fit.pitch_zero_rad,
      "fit_slope": fit.slope,
      "fit_score": fit.score,
      "fit_pitch_span": fit.pitch_span,
      "fit_residual_mad": fit.residual_mad,
      "fit_sample_count": fit.sample_count,
      "fit_speed_band_spread": fit.speed_band_spread,
    },
    "apply_go": False,
    "no_go_reasons": ["no labeled steep-climb shift/no-shift corpus"],
  }
  if report["longest_6_percent_climb_s"] < 10.0:
    report["no_go_reasons"].append("no sustained 6% climb")
  return report


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("npz", nargs="+", help="route NPZ files produced by extract_route_npz.py")
  parser.add_argument("--output", type=Path)
  args = parser.parse_args()
  result = {"routes": [analyze_npz(path) for path in args.npz], "apply_go": False}
  text = json.dumps(result, indent=2, sort_keys=True)
  if args.output:
    args.output.write_text(text + "\n")
  print(text)


if __name__ == "__main__":
  main()
