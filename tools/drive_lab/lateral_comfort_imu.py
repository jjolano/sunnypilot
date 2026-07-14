#!/usr/bin/env python3
"""IMU-measured lateral comfort survey.

The lateral stack's own "actual lateral jerk" is inferred from steering rate
(response_core), i.e. what the controller commanded — not what the car body did. This tool
scores felt lateral comfort from the fused IMU (livePose yaw rate, calibrated to the
vehicle frame) and attributes the worst jerk events to control (commanded curvature moved
with the body) or road/disturbance (body moved on its own), so tuning effort goes to what
occupants actually feel.

Per engaged-at-speed frame:
  measured lateral accel  = calibrated yaw_rate * v_ego     (vehicle response)
  commanded lateral accel = carControl.actuators.curvature * v_ego^2  (controller intent)
Both are smoothed identically and differentiated for jerk, so the pair is comparable.

Usage (route names resolve like the other drive_lab tools):
  uv run python -m openpilot.tools.drive_lab.lateral_comfort_imu 0000028c \\
      --log-root /tmp/opencode/sunnypilot-route-logs --top 5
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from openpilot.selfdrive.locationd.helpers import Pose, PoseCalibrator
from openpilot.tools.drive_lab.analyze_longitudinal_lateral_route import DEFAULT_LOG_ROOTS, resolve_inputs
from openpilot.tools.lib.logreader import LogReader, ReadMode

SMOOTH_WINDOW_S = 0.25          # kills mount vibration, keeps the 0-3 Hz comfort band
MASK_MIN_V_EGO = 3.0            # score comfort only at speed
EVENT_JERK_MIN = 0.8            # m/s^3 measured jerk to count as an event
EVENT_HALF_WINDOW_S = 1.0       # attribution window around an event peak
EVENT_SPACING_S = 1.5           # non-max suppression between events
CONTROL_CORR_MIN = 0.6          # commanded/measured jerk correlation to call it control-caused
CONTROL_RATIO_MIN = 0.5         # commanded peak must be a real fraction of the measured peak
# ponytail: band split via moving-average differences, not a real filter bank; swap in
# scipy band-passes if the wander/comfort split ever needs sharp edges.
WANDER_SLOW_WINDOW_S = 8.0
WANDER_FAST_WINDOW_S = 1.2


@dataclass(frozen=True)
class ComfortEvent:
  t: float
  v_ego: float
  measured_jerk: float
  commanded_jerk: float
  correlation: float
  label: str  # "control" | "road/disturbance"


@dataclass(frozen=True)
class ComfortSeries:
  t: np.ndarray
  measured_lat_accel: np.ndarray   # yaw_rate * v (vehicle frame)
  commanded_lat_accel: np.ndarray  # commanded curvature * v^2
  v_ego: np.ndarray
  mask: np.ndarray                 # engaged, at speed, hands off


def smooth(x: np.ndarray, t: np.ndarray, window_s: float) -> np.ndarray:
  if len(t) < 3:
    return x.copy()
  dt = float(np.median(np.diff(t)))
  n = max(1, int(round(window_s / max(dt, 1e-3))))
  n += 1 - n % 2  # odd
  if n <= 1:
    return x.copy()
  return np.convolve(x, np.ones(n) / n, mode="same")


def jerks(series: ComfortSeries) -> tuple[np.ndarray, np.ndarray]:
  meas = smooth(series.measured_lat_accel, series.t, SMOOTH_WINDOW_S)
  cmd = smooth(series.commanded_lat_accel, series.t, SMOOTH_WINDOW_S)
  return np.gradient(meas, series.t), np.gradient(cmd, series.t)


def find_events(series: ComfortSeries, top: int = 10, jerk_min: float = EVENT_JERK_MIN) -> list[ComfortEvent]:
  meas_jerk, cmd_jerk = jerks(series)
  candidate = np.where(series.mask & (np.abs(meas_jerk) >= jerk_min))[0]
  order = candidate[np.argsort(-np.abs(meas_jerk[candidate]))]
  events: list[ComfortEvent] = []
  taken: list[float] = []
  for i in order:
    t_i = float(series.t[i])
    if any(abs(t_i - t_e) < EVENT_SPACING_S for t_e in taken):
      continue
    w = (series.t >= t_i - EVENT_HALF_WINDOW_S) & (series.t <= t_i + EVENT_HALF_WINDOW_S)
    corr = 0.0
    if w.sum() >= 4 and np.std(cmd_jerk[w]) > 1e-6 and np.std(meas_jerk[w]) > 1e-6:
      corr = float(np.corrcoef(cmd_jerk[w], meas_jerk[w])[0, 1])
    cmd_peak = float(np.max(np.abs(cmd_jerk[w]))) if w.any() else 0.0
    meas_peak = float(abs(meas_jerk[i]))
    control = corr >= CONTROL_CORR_MIN and cmd_peak >= CONTROL_RATIO_MIN * meas_peak
    events.append(ComfortEvent(t=t_i, v_ego=float(series.v_ego[i]),
                               measured_jerk=float(meas_jerk[i]), commanded_jerk=cmd_peak,
                               correlation=corr, label="control" if control else "road/disturbance"))
    taken.append(t_i)
    if len(events) >= top:
      break
  return events


def summarize(series: ComfortSeries) -> dict:
  meas_jerk, cmd_jerk = jerks(series)
  m = series.mask
  if not m.any():
    return {"masked_duration_s": 0.0}
  la = series.measured_lat_accel
  wander = smooth(la, series.t, WANDER_FAST_WINDOW_S) - smooth(la, series.t, WANDER_SLOW_WINDOW_S)
  comfort = smooth(la, series.t, SMOOTH_WINDOW_S) - smooth(la, series.t, WANDER_FAST_WINDOW_S)
  dt = float(np.median(np.diff(series.t)))
  abs_meas, abs_cmd = np.abs(meas_jerk[m]), np.abs(cmd_jerk[m])
  return {
    "masked_duration_s": float(m.sum() * dt),
    "measured_jerk_p50": float(np.percentile(abs_meas, 50)),
    "measured_jerk_p95": float(np.percentile(abs_meas, 95)),
    "measured_jerk_p99": float(np.percentile(abs_meas, 99)),
    "measured_jerk_max": float(abs_meas.max()),
    "commanded_jerk_p95": float(np.percentile(abs_cmd, 95)),
    "commanded_jerk_p99": float(np.percentile(abs_cmd, 99)),
    "wander_band_rms": float(np.sqrt(np.mean(wander[m] ** 2))),
    "comfort_band_rms": float(np.sqrt(np.mean(comfort[m] ** 2))),
  }


def extract_series(identifiers: list[str]) -> ComfortSeries:
  calib = PoseCalibrator()
  v_ego = 0.0
  steering_pressed = False
  lat_active = False
  cmd_curv = 0.0
  t0 = None
  rows: list[tuple[float, float, float, float, bool]] = []
  for msg in LogReader(identifiers, sort_by_time=True):
    w = msg.which()
    if t0 is None:
      t0 = msg.logMonoTime
    if w == "liveCalibration":
      calib.feed_live_calib(msg.liveCalibration)
    elif w == "carState":
      v_ego = float(msg.carState.vEgo)
      steering_pressed = bool(msg.carState.steeringPressed)
    elif w == "carControl":
      lat_active = bool(msg.carControl.latActive)
      cmd_curv = float(msg.carControl.actuators.curvature)
    elif w == "livePose":
      pose = calib.build_calibrated_pose(Pose.from_live_pose(msg.livePose))
      yaw_rate = float(pose.angular_velocity.z)
      if not math.isfinite(yaw_rate):
        continue
      rows.append(((msg.logMonoTime - t0) / 1e9, yaw_rate, cmd_curv, v_ego,
                   lat_active and not steering_pressed and v_ego > MASK_MIN_V_EGO))
  if not rows:
    raise ValueError("no livePose frames found (qlog-only input?)")
  a = np.array(rows, dtype=float)
  return ComfortSeries(t=a[:, 0], measured_lat_accel=a[:, 1] * a[:, 3],
                       commanded_lat_accel=a[:, 2] * a[:, 3] ** 2,
                       v_ego=a[:, 3], mask=a[:, 4] > 0.5)


def main() -> None:
  parser = argparse.ArgumentParser(description="IMU-measured lateral comfort survey")
  parser.add_argument("inputs", nargs="+", help="Route id/name, local route dir, or log files")
  parser.add_argument("--log-root", action="append", default=[], help="Extra root for local short routes")
  parser.add_argument("--top", type=int, default=8, help="Top jerk events to report per route")
  parser.add_argument("--json", dest="json_path", help="Write per-route summaries + events as JSON")
  args = parser.parse_args()

  log_roots = tuple(Path(p) for p in args.log_root) + DEFAULT_LOG_ROOTS
  out = {}
  for route in args.inputs:
    identifiers = resolve_inputs([route], segment=None, read_mode=ReadMode.RLOG, log_roots=log_roots)
    series = extract_series(identifiers)
    stats = summarize(series)
    events = find_events(series, top=args.top)
    out[route] = {"summary": stats, "events": [e.__dict__ for e in events]}
    print(f"\n=== {route}  engaged-at-speed {stats.get('masked_duration_s', 0.0):.0f}s ===")
    if stats.get("masked_duration_s", 0.0) <= 0.0:
      print("  no engaged-at-speed frames")
      continue
    print(f"  measured |jerk| p50={stats['measured_jerk_p50']:.2f} p95={stats['measured_jerk_p95']:.2f} "
          f"p99={stats['measured_jerk_p99']:.2f} max={stats['measured_jerk_max']:.2f} m/s^3")
    print(f"  commanded |jerk| p95={stats['commanded_jerk_p95']:.2f} p99={stats['commanded_jerk_p99']:.2f} m/s^3")
    print(f"  lat-accel RMS: wander band(~0.1-0.8Hz)={stats['wander_band_rms']:.3f} "
          f"comfort band(~0.8-3Hz)={stats['comfort_band_rms']:.3f} m/s^2")
    for e in events:
      print(f"    t={e.t:7.1f}s v={e.v_ego:4.1f} measured={e.measured_jerk:+5.2f} "
            f"cmd_peak={e.commanded_jerk:5.2f} corr={e.correlation:+.2f}  {e.label}")
  if args.json_path:
    Path(args.json_path).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.json_path}")


if __name__ == "__main__":
  main()
