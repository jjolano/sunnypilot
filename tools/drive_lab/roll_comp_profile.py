#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path
from typing import Any

import numpy as np

from openpilot.tools.drive_lab.route_io import load_route_msgs, output_report
from openpilot.tools.drive_lab.timeline import msg_payload, msg_time_s, msg_type, safe_get
from sunnypilot.custom.lateral.speed_aware_torque import _fit_slope

GRAVITY = 9.81
MIN_V_EGO = 15.0
MAX_DESIRED_LATERAL_ACCEL = 0.15
MAX_DESIRED_LATERAL_ACCEL_DELTA = 0.05
MIN_X_SPAN = 1e-6


@dataclass(frozen=True)
class RollCompProfileReport:
  source: str
  slope: float | None
  integrator_mean: float | None
  integrator_std: float | None
  point_count: int
  roll_span: float

  def to_dict(self) -> dict[str, Any]:
    return asdict(self)


@dataclass(frozen=True)
class _Frame:
  t_s: float
  lat_active: bool
  steering_pressed: bool
  v_ego: float
  roll: float
  p: float
  i: float
  f: float
  desired_lateral_accel: float
  saturated: bool
  torque_active: bool


def _extract_frames(msgs: list[Any]) -> list[_Frame]:
  if not msgs:
    return []

  latest: dict[str, Any] = {}
  frames: list[_Frame] = []

  for msg in msgs:
    typ = msg_type(msg)
    payload = msg_payload(msg)
    if typ in ("carState", "carControl", "liveParameters"):
      latest[typ] = payload
    if typ != "controlsState":
      continue

    lateral_control = safe_get(payload, "lateralControlState")
    if lateral_control is None:
      continue
    which = safe_get(lateral_control, "which")
    if callable(which):
      which = which()
    if which != "torqueState":
      continue

    torque_state = safe_get(lateral_control, "torqueState")
    if torque_state is None:
      continue

    car_state = latest.get("carState")
    car_control = latest.get("carControl")
    live_params = latest.get("liveParameters")
    if car_state is None or car_control is None or live_params is None:
      continue

    roll = safe_get(live_params, "roll")
    v_ego = safe_get(car_state, "vEgo")
    p = safe_get(torque_state, "p")
    i = safe_get(torque_state, "i")
    f_val = safe_get(torque_state, "f")
    desired = safe_get(torque_state, "desiredLateralAccel")

    if not all(isinstance(v, int | float) and isfinite(float(v)) for v in (roll, v_ego, p, i, f_val, desired)):
      continue

    frames.append(_Frame(
      t_s=msg_time_s(msg),
      lat_active=bool(safe_get(car_control, "latActive", False)),
      steering_pressed=bool(safe_get(car_state, "steeringPressed", False)),
      v_ego=float(v_ego),
      roll=float(roll),
      p=float(p),
      i=float(i),
      f=float(f_val),
      desired_lateral_accel=float(desired),
      saturated=bool(safe_get(torque_state, "saturated", False)),
      torque_active=bool(safe_get(torque_state, "active", True)),
    ))

  return frames


def _select_straight_frames(frames: list[_Frame]) -> list[_Frame]:
  selected: list[_Frame] = []
  previous_desired: float | None = None

  for frame in frames:
    delta_ok = previous_desired is None or abs(frame.desired_lateral_accel - previous_desired) <= MAX_DESIRED_LATERAL_ACCEL_DELTA
    previous_desired = frame.desired_lateral_accel
    if not frame.lat_active or not frame.torque_active:
      continue
    if frame.steering_pressed:
      continue
    if frame.v_ego <= MIN_V_EGO:
      continue
    if abs(frame.desired_lateral_accel) > MAX_DESIRED_LATERAL_ACCEL:
      continue
    if not delta_ok:
      continue
    if frame.saturated:
      continue
    selected.append(frame)

  return selected


def build_roll_comp_profile(msgs: list[Any], source: str = "unknown", already_sorted: bool = False) -> RollCompProfileReport:
  if not already_sorted:
    msgs = sorted(msgs, key=lambda m: int(getattr(m, "logMonoTime", 0)))

  frames = _extract_frames(msgs)
  straight = _select_straight_frames(frames)

  if not straight:
    return RollCompProfileReport(
      source=source,
      slope=None,
      integrator_mean=None,
      integrator_std=None,
      point_count=0,
      roll_span=0.0,
    )

  xs = np.array([-np.sin(f.roll) * GRAVITY for f in straight], dtype=float)
  ys = np.array([f.p + f.i + f.f for f in straight], dtype=float)
  integrators = np.array([f.i for f in straight], dtype=float)

  points = np.column_stack([xs, np.ones_like(xs), ys])
  span = float(np.max(xs) - np.min(xs))
  slope = _fit_slope(points) if span >= MIN_X_SPAN and len(points) >= 2 else None

  return RollCompProfileReport(
    source=source,
    slope=slope,
    integrator_mean=float(np.mean(integrators)),
    integrator_std=float(np.std(integrators)),
    point_count=len(straight),
    roll_span=span,
  )


def render_roll_comp_profile(report: RollCompProfileReport) -> str:
  slope_str = f"{report.slope:.4f}" if report.slope is not None else "n/a"
  mean_str = f"{report.integrator_mean:.4f}" if report.integrator_mean is not None else "n/a"
  std_str = f"{report.integrator_std:.4f}" if report.integrator_std is not None else "n/a"
  lines = [
    f"Roll compensation profile for {report.source}",
    f"  slope:            {slope_str}",
    f"  integrator mean:  {mean_str}",
    f"  integrator std:   {std_str}",
    f"  point count:      {report.point_count}",
    f"  roll span (m/s^2): {report.roll_span:.4f}",
  ]
  return "\n".join(lines)


def save_roll_comp_profile(report: RollCompProfileReport, path: str | Path) -> None:
  Path(path).write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")


def load_roll_comp_profile(path: str | Path) -> RollCompProfileReport:
  data = json.loads(Path(path).read_text())
  return RollCompProfileReport(
    source=str(data.get("source", "unknown")),
    slope=float(data["slope"]) if data.get("slope") is not None else None,
    integrator_mean=float(data["integrator_mean"]) if data.get("integrator_mean") is not None else None,
    integrator_std=float(data["integrator_std"]) if data.get("integrator_std") is not None else None,
    point_count=int(data.get("point_count", 0)),
    roll_span=float(data.get("roll_span", 0.0)),
  )


def main() -> None:
  parser = argparse.ArgumentParser(description="Profile roll-compensation gain from route logs.")
  parser.add_argument("route", help="Route, segment range, log file, or URL accepted by LogReader")
  parser.add_argument("--output", help="Write report JSON to this path")
  parser.add_argument("--json", action="store_true", help="Print JSON instead of a text summary")
  parser.add_argument("--qlog", action="store_true", help="Prefer qlogs instead of rlogs")
  args = parser.parse_args()

  msgs = load_route_msgs(args.route, qlog=args.qlog)
  report = build_roll_comp_profile(msgs, source=args.route, already_sorted=True)

  print(output_report(
    report,
    json_output=args.json,
    renderer=render_roll_comp_profile,
    output_path=args.output,
    save=save_roll_comp_profile,
  ))


if __name__ == "__main__":
  main()
