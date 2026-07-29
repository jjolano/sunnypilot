#!/usr/bin/env python3
"""Trace the MPC cut-in timing chain to find where the delay occurs.

For each cut-in event (new radarTrackId appearing in radarState with present=True),
measures the timing from:

1. t_lead_first_seen: radarTrackId first appears in leadOne/leadTwo with present=True
2. t_source_switch: longitudinalPlanSource switches from cruise to lead0/lead1
3. t_mpc_brake: longitudinalPlan.aTarget goes below -0.1 m/s²
4. t_sp_brake: longitudinalPlanSP.aTarget goes below -0.1 m/s²
5. t_ego_brake: carState.aEgo goes below -0.1 m/s²

The delay between steps tells us where the 2.7s cut-in brake delay comes from:
- lead_first_seen → source_switch: vision model confirmation + MPC source selection
- source_switch → mpc_brake: MPC solve + trajectory extrapolation
- mpc_brake → ego_brake: actuator lag

Run:
  uv run python -m openpilot.tools.drive_lab.trace_cut_in_timing ROUTE --qlog
  uv run python -m openpilot.tools.drive_lab.trace_cut_in_timing ROUTE --qlog --json --output /tmp/x.json
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from openpilot.tools.drive_lab.route_analysis import build_route_messages
from openpilot.tools.drive_lab.route_io import load_route_msgs, output_report
from openpilot.tools.drive_lab.timeline import safe_get


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TimingChainParams:
  # Cut-in detection
  max_d_rel: float = 30.0             # m; only consider leads within this distance
  min_closing_speed: float = 1.0      # m/s; -vRel must exceed this
  min_v_ego: float = 1.0             # m/s; ignore below this speed
  track_swap_d_rel: float = 4.0       # m; new lead within this of old = swap, not cut-in
  track_swap_v_rel: float = 2.0       # m/s

  # Timing thresholds
  mpc_brake_threshold: float = -0.1   # m/s²; aTarget below this = braking
  ego_brake_threshold: float = -0.1   # m/s²; aEgo below this = braking
  timing_window_s: float = 10.0       # s; max window to look for each step

  # Lead tracking
  min_persistence_frames: int = 2     # frames before counting as a real cut-in


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CutInTimingEvent:
  t_lead_first_seen: float            # when radarTrackId first appears with present=True
  lead_id: int
  d_rel: float
  v_rel: float
  y_rel: float
  v_lead: float
  v_ego: float
  model_prob: float
  op_engaged: bool

  t_source_switch: float | None = None       # when planSource switches to lead0/lead1
  source_before: str = ""
  source_after: str = ""
  t_mpc_brake: float | None = None           # when longitudinalPlan.aTarget < -0.1
  mpc_brake_a: float | None = None
  t_sp_brake: float | None = None            # when longitudinalPlanSP.aTarget < -0.1
  sp_brake_a: float | None = None
  t_ego_brake: float | None = None           # when carState.aEgo < -0.1
  ego_brake_a: float | None = None

  @property
  def lead_to_source_s(self) -> float | None:
    if self.t_source_switch is not None:
      return self.t_source_switch - self.t_lead_first_seen
    return None

  @property
  def source_to_mpc_brake_s(self) -> float | None:
    if self.t_source_switch is not None and self.t_mpc_brake is not None:
      return self.t_mpc_brake - self.t_source_switch
    return None

  @property
  def mpc_brake_to_ego_brake_s(self) -> float | None:
    if self.t_mpc_brake is not None and self.t_ego_brake is not None:
      return self.t_ego_brake - self.t_mpc_brake
    return None

  @property
  def total_lead_to_ego_s(self) -> float | None:
    if self.t_ego_brake is not None:
      return self.t_ego_brake - self.t_lead_first_seen
    return None


@dataclass
class TimingChainReport:
  source: str
  duration_s: float
  op_engaged_s: float
  manual_moving_s: float
  total_cut_ins: int
  events: list[CutInTimingEvent] = field(default_factory=list)
  notes: list[str] = field(default_factory=list)

  def to_dict(self) -> dict[str, Any]:
    lead_to_source = [e.lead_to_source_s for e in self.events if e.lead_to_source_s is not None]
    source_to_mpc = [e.source_to_mpc_brake_s for e in self.events if e.source_to_mpc_brake_s is not None]
    mpc_to_ego = [e.mpc_brake_to_ego_brake_s for e in self.events if e.mpc_brake_to_ego_brake_s is not None]
    total = [e.total_lead_to_ego_s for e in self.events if e.total_lead_to_ego_s is not None]

    op_events = [e for e in self.events if e.op_engaged]
    manual_events = [e for e in self.events if not e.op_engaged]

    op_total = [e.total_lead_to_ego_s for e in op_events if e.total_lead_to_ego_s is not None]
    manual_total = [e.total_lead_to_ego_s for e in manual_events if e.total_lead_to_ego_s is not None]

    return {
      "source": self.source,
      "duration_s": _r(self.duration_s, 2),
      "op_engaged_s": _r(self.op_engaged_s, 2),
      "manual_moving_s": _r(self.manual_moving_s, 2),
      "total_cut_ins": self.total_cut_ins,
      "summary": {
        "events": len(self.events),
        "op_events": len(op_events),
        "manual_events": len(manual_events),
        "lead_to_source_switch_median_s": _r(_opt_median(lead_to_source), 3),
        "source_switch_to_mpc_brake_median_s": _r(_opt_median(source_to_mpc), 3),
        "mpc_brake_to_ego_brake_median_s": _r(_opt_median(mpc_to_ego), 3),
        "total_lead_to_ego_median_s": _r(_opt_median(total), 3),
        "op_total_median_s": _r(_opt_median(op_total), 3),
        "manual_total_median_s": _r(_opt_median(manual_total), 3),
      },
      "events": [_event_to_dict(e) for e in self.events],
      "notes": list(self.notes),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _r(value: Any, ndigits: int = 6) -> float | None:
  if value is None or not isinstance(value, int | float) or not math.isfinite(float(value)):
    return None
  return round(float(value), ndigits)


def _f(value: Any, default: float = 0.0) -> float:
  try:
    out = float(value)
  except (TypeError, ValueError):
    return default
  return out if math.isfinite(out) else default


def _opt_median(values: list[float]) -> float | None:
  return float(np.median(values)) if values else None


def _event_to_dict(e: CutInTimingEvent) -> dict[str, Any]:
  return {
    "t_lead_first_seen": _r(e.t_lead_first_seen, 3),
    "lead_id": e.lead_id,
    "d_rel": _r(e.d_rel, 2), "v_rel": _r(e.v_rel, 3), "y_rel": _r(e.y_rel, 2),
    "v_lead": _r(e.v_lead, 3), "v_ego": _r(e.v_ego, 3),
    "model_prob": _r(e.model_prob, 3), "op_engaged": e.op_engaged,
    "t_source_switch": _r(e.t_source_switch, 3),
    "source_before": e.source_before, "source_after": e.source_after,
    "t_mpc_brake": _r(e.t_mpc_brake, 3), "mpc_brake_a": _r(e.mpc_brake_a, 3),
    "t_sp_brake": _r(e.t_sp_brake, 3), "sp_brake_a": _r(e.sp_brake_a, 3),
    "t_ego_brake": _r(e.t_ego_brake, 3), "ego_brake_a": _r(e.ego_brake_a, 3),
    "lead_to_source_s": _r(e.lead_to_source_s, 3),
    "source_to_mpc_brake_s": _r(e.source_to_mpc_brake_s, 3),
    "mpc_brake_to_ego_brake_s": _r(e.mpc_brake_to_ego_brake_s, 3),
    "total_lead_to_ego_s": _r(e.total_lead_to_ego_s, 3),
  }


# ---------------------------------------------------------------------------
# Core collection
# ---------------------------------------------------------------------------

@dataclass
class _Frame:
  """Combined state at a single timestamp."""
  t: float
  v_ego: float
  a_ego: float
  long_active: bool
  lead_one_id: int
  lead_one_status: bool
  lead_one_d_rel: float
  lead_one_v_rel: float
  lead_one_y_rel: float
  lead_one_v_lead: float
  lead_one_model_prob: float
  lead_two_id: int
  lead_two_status: bool
  plan_source: str
  plan_a_target: float
  sp_a_target: float | None


def _collect_frames(msgs: list[Any], p: TimingChainParams) -> tuple[list[_Frame], float, float, float]:
  frames: list[_Frame] = []
  op_engaged_s = 0.0
  manual_moving_s = 0.0
  last_t: float | None = None
  latest: dict[str, Any] = {}

  for rec in build_route_messages(msgs):
    if rec.typ in ("carState", "carControl", "radarState", "longitudinalPlan", "longitudinalPlanSP"):
      latest[rec.typ] = rec.payload

    # Trigger on longitudinalPlan (the MPC output)
    if rec.typ != "longitudinalPlan":
      continue

    cs = latest.get("carState")
    cc = latest.get("carControl")
    radar = latest.get("radarState")
    lp = rec.payload
    sp = latest.get("longitudinalPlanSP")

    if cs is None or radar is None:
      continue

    v_ego = _f(safe_get(cs, "vEgo"))
    a_ego = _f(safe_get(cs, "aEgo"))
    long_active = bool(safe_get(cc, "longActive", False))

    if last_t is not None and math.isfinite(v_ego):
      dt = max(0.0, min(2.0, rec.t - last_t))
      if long_active:
        op_engaged_s += dt
      elif v_ego > 0.1:
        manual_moving_s += dt
    last_t = rec.t

    lead_one = safe_get(radar, "leadOne")
    lead_two = safe_get(radar, "leadTwo")

    def _lead_field(lead, field, default=0.0):
      if lead is None:
        return default
      return _f(safe_get(lead, field, default))

    l1 = lead_one if lead_one is not None else type('X', (), {'present': False, 'dRel': 0, 'vRel': 0, 'yRel': 0, 'vLead': 0, 'modelProb': 0, 'radarTrackId': -1})()
    l2 = lead_two if lead_two is not None else type('X', (), {'present': False, 'radarTrackId': -1})()

    raw_source = safe_get(lp, "longitudinalPlanSource", "unknown")
    if hasattr(raw_source, 'name'):
      plan_source = raw_source.name
    else:
      plan_source = str(raw_source)
    plan_a = _f(safe_get(lp, "aTarget"))
    sp_a = _f(safe_get(sp, "aTarget")) if sp is not None else float("nan")

    frames.append(_Frame(
      t=rec.t, v_ego=v_ego, a_ego=a_ego, long_active=long_active,
      lead_one_id=int(_lead_field(l1, "radarTrackId", -1)),
      lead_one_status=bool(safe_get(l1, "present", False)),
      lead_one_d_rel=_lead_field(l1, "dRel"),
      lead_one_v_rel=_lead_field(l1, "vRel"),
      lead_one_y_rel=_lead_field(l1, "yRel"),
      lead_one_v_lead=_lead_field(l1, "vLead"),
      lead_one_model_prob=_lead_field(l1, "modelProb"),
      lead_two_id=int(_lead_field(l2, "radarTrackId", -1)),
      lead_two_status=bool(safe_get(l2, "present", False)),
      plan_source=plan_source,
      plan_a_target=plan_a,
      sp_a_target=sp_a if math.isfinite(sp_a) else None,
    ))

  duration = frames[-1].t - frames[0].t if frames else 0.0
  return frames, op_engaged_s, manual_moving_s, duration


def _detect_cut_ins(frames: list[_Frame], p: TimingChainParams) -> list[CutInTimingEvent]:
  events: list[CutInTimingEvent] = []
  if len(frames) < 5:
    return events

  prev_lead_id: int | None = None
  prev_d_rel: float = 0.0
  prev_v_rel: float = 0.0
  pending_new_ids: dict[int, int] = {}  # lead_id → frames seen

  for i, frame in enumerate(frames):
    if not frame.lead_one_status or frame.lead_one_id < 0:
      prev_lead_id = None
      continue

    if frame.v_ego < p.min_v_ego:
      prev_lead_id = None
      continue

    lead_id = frame.lead_one_id
    is_new = (prev_lead_id is not None and lead_id != prev_lead_id and lead_id > 0)

    if is_new:
      # Track-swap suppression
      if (abs(frame.lead_one_d_rel - prev_d_rel) < p.track_swap_d_rel and
          abs(frame.lead_one_v_rel - prev_v_rel) < p.track_swap_v_rel):
        prev_lead_id = lead_id
        prev_d_rel = frame.lead_one_d_rel
        prev_v_rel = frame.lead_one_v_rel
        continue

      # Check if it's a real cut-in (closing speed, distance)
      closing_speed = max(0.0, -frame.lead_one_v_rel)
      if closing_speed < p.min_closing_speed or frame.lead_one_d_rel > p.max_d_rel:
        prev_lead_id = lead_id
        prev_d_rel = frame.lead_one_d_rel
        prev_v_rel = frame.lead_one_v_rel
        continue

      # New cut-in detected
      events.append(CutInTimingEvent(
        t_lead_first_seen=frame.t,
        lead_id=lead_id,
        d_rel=frame.lead_one_d_rel,
        v_rel=frame.lead_one_v_rel,
        y_rel=frame.lead_one_y_rel,
        v_lead=frame.lead_one_v_lead,
        v_ego=frame.v_ego,
        model_prob=frame.lead_one_model_prob,
        op_engaged=frame.long_active,
        source_before=frame.plan_source,
      ))

    prev_lead_id = lead_id
    prev_d_rel = frame.lead_one_d_rel
    prev_v_rel = frame.lead_one_v_rel

  # Now trace each event through the timing chain
  for event in events:
    end_t = event.t_lead_first_seen + p.timing_window_s

    # Find source switch
    source_switched = False
    for i in range(len(frames)):
      if frames[i].t < event.t_lead_first_seen:
        continue
      if frames[i].t > end_t:
        break
      source = frames[i].plan_source
      if source in ("lead0", "lead1") and not source_switched:
        event.t_source_switch = frames[i].t
        event.source_after = source
        source_switched = True
        break

    # Find MPC brake
    for i in range(len(frames)):
      if frames[i].t < event.t_lead_first_seen:
        continue
      if frames[i].t > end_t:
        break
      if frames[i].plan_a_target < p.mpc_brake_threshold:
        event.t_mpc_brake = frames[i].t
        event.mpc_brake_a = frames[i].plan_a_target
        break

    # Find SP brake
    for i in range(len(frames)):
      if frames[i].t < event.t_lead_first_seen:
        continue
      if frames[i].t > end_t:
        break
      sp_a = frames[i].sp_a_target
      if sp_a is not None and math.isfinite(sp_a) and sp_a < p.mpc_brake_threshold:
        event.t_sp_brake = frames[i].t
        event.sp_brake_a = sp_a
        break

    # Find ego brake
    for i in range(len(frames)):
      if frames[i].t < event.t_lead_first_seen:
        continue
      if frames[i].t > end_t:
        break
      if frames[i].a_ego < p.ego_brake_threshold:
        event.t_ego_brake = frames[i].t
        event.ego_brake_a = frames[i].a_ego
        break

  return events


def analyze_route(msgs: list[Any], source: str = "unknown",
                  params: TimingChainParams | None = None) -> TimingChainReport:
  p = params or TimingChainParams()
  frames, op_engaged_s, manual_moving_s, duration = _collect_frames(msgs, p)
  notes: list[str] = []

  if not frames:
    notes.append("no longitudinalPlan frames found")

  events = _detect_cut_ins(frames, p)

  return TimingChainReport(
    source=source,
    duration_s=duration,
    op_engaged_s=op_engaged_s,
    manual_moving_s=manual_moving_s,
    total_cut_ins=len(events),
    events=events,
    notes=notes,
  )


def render_report(report: TimingChainReport) -> str:
  lines = [
    f"Cut-in timing chain: {report.source}",
    f"  duration {report.duration_s:.1f}s, OP engaged {report.op_engaged_s:.1f}s, "
    f"manual moving {report.manual_moving_s:.1f}s",
    f"  total cut-ins detected: {report.total_cut_ins}",
  ]

  s = report.to_dict()["summary"]
  lines += [
    "",
    "  timing chain (median):",
    f"    lead first seen → source switch:  {s['lead_to_source_switch_median_s']} s",
    f"    source switch → MPC brake:        {s['source_switch_to_mpc_brake_median_s']} s",
    f"    MPC brake → ego brake:            {s['mpc_brake_to_ego_brake_median_s']} s",
    f"    total (lead seen → ego brake):     {s['total_lead_to_ego_median_s']} s",
    "",
    f"    OP total median:     {s['op_total_median_s']} s",
    f"    manual total median:  {s['manual_total_median_s']} s",
  ]

  # Per-event detail
  op_events = [e for e in report.events if e.op_engaged]
  if len(op_events) <= 20:
    lines += ["", "  OP events:"]
    for e in op_events:
      lines.append(
        f"    t={e.t_lead_first_seen:.1f} id={e.lead_id} d={e.d_rel:.1f}m vR={e.v_rel:.1f} "
        f"prob={e.model_prob:.2f} "
        f"src_switch={_fmt(e.t_source_switch, 1)} ({e.source_before}→{e.source_after}) "
        f"mpc_brake={_fmt(e.t_mpc_brake, 1)} "
        f"ego_brake={_fmt(e.t_ego_brake, 1)} "
        f"total={_fmt(e.total_lead_to_ego_s, 2)}s"
      )

  manual_events = [e for e in report.events if not e.op_engaged]
  if len(manual_events) <= 20:
    lines += ["", "  manual events:"]
    for e in manual_events:
      lines.append(
        f"    t={e.t_lead_first_seen:.1f} id={e.lead_id} d={e.d_rel:.1f}m vR={e.v_rel:.1f} "
        f"prob={e.model_prob:.2f} "
        f"src_switch={_fmt(e.t_source_switch, 1)} ({e.source_before}→{e.source_after}) "
        f"mpc_brake={_fmt(e.t_mpc_brake, 1)} "
        f"ego_brake={_fmt(e.t_ego_brake, 1)} "
        f"total={_fmt(e.total_lead_to_ego_s, 2)}s"
      )

  for note in report.notes:
    lines.append(f"  note: {note}")
  return "\n".join(lines)


def _fmt(value: float | None, ndigits: int = 3) -> str:
  return "n/a" if value is None or not math.isfinite(float(value)) else f"{float(value):.{ndigits}f}"


def main() -> None:
  parser = argparse.ArgumentParser(description="Trace MPC cut-in timing chain from route logs.")
  parser.add_argument("route", help="Route, segment range, log file, or URL accepted by LogReader")
  parser.add_argument("--output", help="Write the report JSON to this path")
  parser.add_argument("--json", action="store_true", help="Print JSON instead of the text summary")
  parser.add_argument("--qlog", action="store_true", help="Prefer qlogs instead of rlogs (lower rate)")
  args = parser.parse_args()

  msgs = load_route_msgs(args.route, qlog=args.qlog)
  report = analyze_route(msgs, source=args.route)
  print(output_report(report, json_output=args.json, renderer=render_report, output_path=args.output))


if __name__ == "__main__":
  main()
