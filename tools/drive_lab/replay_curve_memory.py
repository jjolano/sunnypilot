#!/usr/bin/env python3
"""Replay A/B for the curve-memory fix (corner amnesia): run the lateral demand pipeline over a
route's logged model inputs twice — curve memory OFF vs ON — and score each stream with the
committed corner-recovery metric (``profile_corner_recovery``). This is the default-on gate: does
curve memory lift the launch curvature X toward the pre-stop curve Z on real corner-stops, WITHOUT
over-steering when a corner actually ended during the stop?

Faithful to the controlsd hook (``controlsd.py``): the pipeline sees ``carControl.latActive``,
``carState.vEgo``, ``liveParameters.roll``, raw curvature = ``modelV2.action.desiredCurvature``
(or the measured curvature while inactive), and measured = ``controlsState.curvature`` (==
``self.curvature``, the VM path curvature). The two runs differ ONLY in ``curve_memory_enabled``;
output is pre-clip (clip_curvature would apply equally to both, so it cancels in the A/B).

Run:
  uv run python -m openpilot.tools.drive_lab.replay_curve_memory ROUTE [ROUTE ...]
  uv run python -m openpilot.tools.drive_lab.replay_curve_memory ROUTE --json --output /tmp/x.json
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Any

from openpilot.sunnypilot.custom.lateral.demand.pipeline import LateralDemandPipeline
from openpilot.sunnypilot.custom.lateral.demand.wiring import build_pipeline_inputs
from openpilot.tools.drive_lab.profile_corner_recovery import (
  DetectorParams,
  Sample,
  _analyze_stop,
  _f,
  _lane_change_active,
  _stop_intervals,
)
from openpilot.tools.drive_lab.route_analysis import build_route_messages
from openpilot.tools.drive_lab.route_io import load_route_msgs, output_report
from openpilot.tools.drive_lab.timeline import safe_get

OVERSTEER_MARGIN = 0.004  # 1/m; ON adding more than this beyond raw vision on a curve that ended = over-steer


def replay_samples(msgs: list[Any]) -> tuple[list[Sample], list[Sample]]:
  """Two Sample streams (curve memory off, on), each with cmd_curv = the replayed pipeline output,
  resampled on the controlsState clock (the control rate the pipeline runs at)."""
  pipe_off, pipe_on = LateralDemandPipeline(), LateralDemandPipeline()
  latest: dict[str, Any] = {}
  off: list[Sample] = []
  on: list[Sample] = []
  for record in build_route_messages(msgs):
    typ, payload = record.typ, record.payload
    if typ in ("carState", "modelV2", "carControl", "liveParameters"):
      latest[typ] = payload
      continue
    if typ != "controlsState":
      continue
    cs = latest.get("carState")
    if cs is None:
      continue
    mv = latest.get("modelV2")
    cc = latest.get("carControl")
    lp = latest.get("liveParameters")
    v_ego = _f(safe_get(cs, "vEgo"))
    lat_active = bool(safe_get(cc, "latActive", False)) if cc is not None else True
    measured = _f(safe_get(payload, "curvature"))            # controlsState.curvature == self.curvature
    roll = _f(safe_get(lp, "roll")) if lp is not None else 0.0
    model_curv = _f(safe_get(mv, "action.desiredCurvature"))
    raw = model_curv if lat_active else measured             # controlsd: raw is measured while inactive
    standstill = bool(safe_get(cs, "standstill", False))
    steering_pressed = bool(safe_get(cs, "steeringPressed", False))
    lane_change_active = _lane_change_active(mv)
    if mv is None or not math.isfinite(raw):
      proc_off = proc_on = raw
    else:
      proc_off = _run(pipe_off, lat_active, v_ego, roll, raw, measured, mv, curve_memory=False,
                      steering_pressed=steering_pressed)
      proc_on = _run(pipe_on, lat_active, v_ego, roll, raw, measured, mv, curve_memory=True,
                     steering_pressed=steering_pressed)
    off.append(Sample(t=record.t, v_ego=v_ego, standstill=standstill, steering_pressed=steering_pressed,
                      lane_change_active=lane_change_active, lat_active=lat_active, cmd_curv=proc_off,
                      act_curv=measured, model_curv=model_curv))
    on.append(Sample(t=record.t, v_ego=v_ego, standstill=standstill, steering_pressed=steering_pressed,
                     lane_change_active=lane_change_active, lat_active=lat_active, cmd_curv=proc_on,
                     act_curv=measured, model_curv=model_curv))
  return off, on


def _run(pipe: LateralDemandPipeline, lat_active: bool, v_ego: float, roll: float, raw: float,
         measured: float, model_v2: Any, curve_memory: bool, steering_pressed: bool) -> float:
  inputs = build_pipeline_inputs(
    lat_active=lat_active, v_ego=v_ego, roll=roll, raw_curvature=raw, measured_curvature=measured,
    model_v2=model_v2, lane_centering_assist_enabled=False, curve_memory_enabled=curve_memory,
    steering_pressed=steering_pressed,
  )
  return float(pipe.update(inputs).demand.processed_curvature)


@dataclass
class EpisodeAB:
  stop_start_s: float
  z: float | None              # pre-stop curve (off stream; ~same on both)
  x_off: float | None
  x_on: float | None
  x_vis: float | None
  continuation: float | None
  cls_off: str
  cls_on: str
  lift: float | None           # |x_on| - |x_off| in Z's direction (>0 = on resumes more corner)
  oversteer: bool              # on adds curvature on a stop whose curve did NOT continue

  def to_dict(self) -> dict[str, Any]:
    return {k: (round(v, 6) if isinstance(v, float) and math.isfinite(v) else v)
            for k, v in self.__dict__.items()}


@dataclass
class ReplayABReport:
  source: str
  corner_stops: int
  amnesia_off: int
  amnesia_on: int
  resumed: int                 # episodes amnesia-off -> not-amnesia-on
  oversteer_events: int
  median_lift: float | None
  episodes: list[EpisodeAB]
  notes: list[str]

  def to_dict(self) -> dict[str, Any]:
    return {
      "source": self.source,
      "corner_stops": self.corner_stops,
      "amnesia_off": self.amnesia_off,
      "amnesia_on": self.amnesia_on,
      "resumed": self.resumed,
      "oversteer_events": self.oversteer_events,
      "median_lift": round(self.median_lift, 6) if self.median_lift is not None else None,
      "episodes": [e.to_dict() for e in self.episodes],
      "notes": list(self.notes),
    }


def analyze_route(msgs: list[Any], source: str = "unknown", params: DetectorParams | None = None) -> ReplayABReport:
  p = params or DetectorParams()
  off, on = replay_samples(msgs)
  notes: list[str] = []
  if not off:
    notes.append("no controlsState samples (need rlogs with controlsState + carState + modelV2)")
    return ReplayABReport(source, 0, 0, 0, 0, 0, None, [], notes)

  episodes: list[EpisodeAB] = []
  for a, b in _stop_intervals(off, p):           # off/on share v_ego/standstill -> identical intervals
    e_off = _analyze_stop(off, a, b, p)
    e_on = _analyze_stop(on, a, b, p)
    if e_off is None or e_on is None or e_off.classification == "straight_stop":
      continue
    z = e_off.pre_stop_curv
    zsign = 1.0 if (z or 0.0) > 0 else -1.0
    x_off, x_on, x_vis = e_off.launch_curv, e_on.launch_curv, e_off.launch_curv_vision
    lift = (abs(x_on) - abs(x_off)) if (x_off is not None and x_on is not None) else None
    continued = e_off.classification not in ("curve_ended",)
    # over-steer: the curve did NOT continue, yet ON pushed launch curvature in Z's direction
    # materially beyond the raw vision (held a corner that had ended).
    oversteer = bool(
      not continued and x_on is not None and x_vis is not None
      and (x_on * zsign) - (x_vis * zsign) > OVERSTEER_MARGIN
    )
    episodes.append(EpisodeAB(
      stop_start_s=e_off.stop_start_s, z=z, x_off=x_off, x_on=x_on, x_vis=x_vis,
      continuation=e_off.continuation_curv, cls_off=e_off.classification, cls_on=e_on.classification,
      lift=lift, oversteer=oversteer,
    ))

  amnesia_off = sum(1 for e in episodes if e.cls_off == "amnesia")
  amnesia_on = sum(1 for e in episodes if e.cls_on == "amnesia")
  resumed = sum(1 for e in episodes if e.cls_off == "amnesia" and e.cls_on != "amnesia")
  lifts = [e.lift for e in episodes if e.lift is not None and e.cls_off == "amnesia"]
  if not episodes:
    notes.append("no corner-stops detected on this route")
  return ReplayABReport(
    source=source,
    corner_stops=len(episodes),
    amnesia_off=amnesia_off,
    amnesia_on=amnesia_on,
    resumed=resumed,
    oversteer_events=sum(1 for e in episodes if e.oversteer),
    median_lift=(float(sorted(lifts)[len(lifts) // 2]) if lifts else None),
    episodes=episodes,
    notes=notes,
  )


def render_report(report: ReplayABReport) -> str:
  lines = [
    f"Curve-memory replay A/B: {report.source}",
    f"  corner-stops {report.corner_stops}:  amnesia off={report.amnesia_off} -> on={report.amnesia_on}  "
    + f"(resumed {report.resumed})   over-steer events: {report.oversteer_events}",
  ]
  if report.median_lift is not None:
    lines.append(f"  median launch-curvature lift on amnesia stops: {report.median_lift:+.4f} 1/m")
  for e in report.episodes:
    tag = "  OVER-STEER" if e.oversteer else ""
    lines.append(
      f"    t={e.stop_start_s:.1f}s  Z={_fmt(e.z)} Xvis={_fmt(e.x_vis)} "
      + f"X_off={_fmt(e.x_off)} X_on={_fmt(e.x_on)} cont={_fmt(e.continuation)}  "
      + f"{e.cls_off}->{e.cls_on}{tag}"
    )
  for note in report.notes:
    lines.append(f"  note: {note}")
  return "\n".join(lines)


def _fmt(value: float | None, ndigits: int = 4) -> str:
  return "n/a" if value is None or not math.isfinite(float(value)) else f"{float(value):.{ndigits}f}"


def main() -> None:
  parser = argparse.ArgumentParser(description="Replay A/B for the curve-memory corner-amnesia fix.")
  parser.add_argument("routes", nargs="+", help="Routes, rlog files, or URLs accepted by LogReader")
  parser.add_argument("--output", help="Write the report JSON to this path")
  parser.add_argument("--json", action="store_true", help="Print JSON instead of the text summary")
  parser.add_argument("--qlog", action="store_true", help="Prefer qlogs instead of rlogs (lower rate)")
  args = parser.parse_args()

  for route in args.routes:
    msgs = load_route_msgs(route, qlog=args.qlog)
    report = analyze_route(msgs, source=route)
    print(output_report(report, json_output=args.json, renderer=render_report, output_path=args.output))


if __name__ == "__main__":
  main()
