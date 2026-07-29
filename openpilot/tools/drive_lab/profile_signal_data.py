#!/usr/bin/env python3
"""Catalog drive data available on a device for longitudinal signal scoring.

Phase 1 of the system-vs-manual longitudinal disagreement pipeline.

For each route, records:
- which segments have rlog/qlog and modelV2 availability (shadow-eligible)
- engaged vs manual moving time
- override events (gas/brake override while system is engaged, or engaged-then-disengaged)
- leadless slow/stop candidate windows (the ground-truth episodes the scorer will use)
- whether the route has any engaged stops (the highest-confidence category)

Output is a JSON census that the signal-scorer, route-to-regression, and triage
phases consume. The census is read-only; it never moves or deletes files.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from openpilot.tools.drive_lab.timeline import msg_payload, safe_get
from openpilot.tools.lib.logreader import LogReader, ReadMode


SPEED_MOVING = 1.0
SLOWING_PEAK_ACCEL = -0.3
SLOWING_DURATION_S = 0.6
DECEL_RATIO_FOR_STOP = 0.7
STOP_SPEED = 0.5
ENGAGED_PLANNER_SOURCES = {"cruise", "e2e", "lead0", "lead1"}


@dataclass(frozen=True)
class OverrideEvent:
  route: str
  route_id: str
  segment: int | None
  start_time_s: float
  end_time_s: float
  kind: str
  active_before: bool
  reason_text: str = ""


@dataclass(frozen=True)
class LeadlessWindow:
  route: str
  route_id: str
  segment: int | None
  start_time_s: float
  end_time_s: float
  peak_decel: float
  min_speed: float
  end_speed: float
  active_ratio: float
  long_active_ratio: float


@dataclass(frozen=True)
class SegmentCensus:
  route: str
  route_id: str
  segment: int | None
  has_rlog: bool
  has_qlog: bool
  has_modelv2: bool
  has_longplan: bool
  has_radar: bool
  sample_count: int
  moving_samples: int
  active_samples: int
  long_active_samples: int
  long_control_state_counts: dict[str, int]
  engaged_source_counts: dict[str, int]
  override_events: list[OverrideEvent] = field(default_factory=list)
  leadless_windows: list[LeadlessWindow] = field(default_factory=list)
  engaged_stops: int = 0


@dataclass(frozen=True)
class CensusSummary:
  total_segments: int
  shadow_eligible_segments: int
  total_samples: int
  moving_samples: int
  active_samples: int
  long_active_samples: int
  override_event_count: int
  leadless_window_count: int
  engaged_stop_count: int
  segments: list[SegmentCensus] = field(default_factory=list)


def _is_file(p: Path) -> bool:
  return p.is_file() and p.stat().st_size > 0


def list_route_segments(roots: Iterable[Path]) -> list[tuple[Path, str, int | None]]:
  out: list[tuple[Path, str, int | None]] = []
  for root in roots:
    if not root.exists():
      continue
    for entry in sorted(root.iterdir()):
      if not entry.is_dir() or entry.name == "boot":
        continue
      name = entry.name
      if "--" not in name:
        continue
      prefix, seg_str = name.rsplit("--", 1)
      try:
        seg = int(seg_str)
      except ValueError:
        continue
      qlog = entry / "qlog.zst"
      rlog = entry / "rlog.zst"
      if _is_file(qlog) or _is_file(rlog):
        out.append((entry, prefix, seg))
  return out


def route_id_for(path: Path) -> str:
  parent = path.parent
  if parent.name in {"qlog.zst", "rlog.zst"} or parent.name.startswith("qlog") or parent.name.startswith("rlog"):
    name = parent.parent.name
  else:
    name = parent.name
  if "--" not in name:
    return name
  return name.rsplit("--", 1)[0]


def detect_segment_files(seg_dir: Path) -> tuple[bool, bool]:
  return _is_file(seg_dir / "rlog.zst"), _is_file(seg_dir / "qlog.zst")


def choose_log_path(seg_dir: Path, prefer_qlog: bool) -> Path | None:
  qlog = seg_dir / "qlog.zst"
  rlog = seg_dir / "rlog.zst"
  if prefer_qlog and _is_file(qlog):
    return qlog
  if _is_file(rlog):
    return rlog
  if _is_file(qlog):
    return qlog
  return None


def classify_override(prev_active: bool, now_active: bool, gas: bool, brake: bool, v_ego: float) -> tuple[str, str] | None:
  if now_active and (gas or brake) and v_ego > STOP_SPEED:
    if gas and not prev_active:
      return ("gas_override_engaged", "driver gas override while system engaged")
    if gas:
      return ("gas_override_engaged", "driver gas override while system engaged")
    if brake:
      return ("brake_override_engaged", "driver brake override while system engaged")
  if prev_active and not now_active and brake:
    return ("brake_disengage", "system disengaged by driver brake press")
  return None


def detect_slowing_runs(states: list[dict], min_duration: float) -> list[LeadlessWindow]:
  out: list[LeadlessWindow] = []
  if not states:
    return out
  run_start = None
  run_min_a = 0.0
  run_min_v = float("inf")
  for i, st in enumerate(states):
    a = st["a_ego"]
    v = st["v_ego"]
    if a <= SLOWING_PEAK_ACCEL:
      if run_start is None:
        run_start = i
        run_min_a = a
        run_min_v = v
      else:
        run_min_a = min(run_min_a, a)
        run_min_v = min(run_min_v, v)
    else:
      if run_start is not None and i - run_start >= max(1, int(min_duration / 0.1)):
        window = states[run_start:i]
        end_v = window[-1]["v_ego"]
        active_ratio = sum(1 for s in window if s["active"]) / max(1, len(window))
        long_active_ratio = sum(1 for s in window if s["long_active"]) / max(1, len(window))
        has_lead = any(s["lead_status"] for s in window)
        if not has_lead and run_min_v <= window[0]["v_ego"] * 0.7 + 0.5:
          out.append(LeadlessWindow(
            route=window[0]["route"],
            route_id=window[0]["route_id"],
            segment=window[0]["segment"],
            start_time_s=window[0]["t"],
            end_time_s=window[-1]["t"],
            peak_decel=run_min_a,
            min_speed=run_min_v,
            end_speed=end_v,
            active_ratio=active_ratio,
            long_active_ratio=long_active_ratio,
          ))
      run_start = None
      run_min_a = 0.0
      run_min_v = float("inf")
  if run_start is not None and len(states) - run_start >= max(1, int(min_duration / 0.1)):
    window = states[run_start:]
    has_lead = any(s["lead_status"] for s in window)
    if not has_lead:
      out.append(LeadlessWindow(
        route=window[0]["route"],
        route_id=window[0]["route_id"],
        segment=window[0]["segment"],
        start_time_s=window[0]["t"],
        end_time_s=window[-1]["t"],
        peak_decel=run_min_a,
        min_speed=run_min_v,
        end_speed=window[-1]["v_ego"],
        active_ratio=sum(1 for s in window if s["active"]) / max(1, len(window)),
        long_active_ratio=sum(1 for s in window if s["long_active"]) / max(1, len(window)),
      ))
  return out


def census_segment(seg_dir: Path, route_id: str, segment: int | None, prefer_qlog: bool) -> SegmentCensus:
  has_rlog, has_qlog = detect_segment_files(seg_dir)
  log_path = choose_log_path(seg_dir, prefer_qlog)
  base_census = SegmentCensus(
    route=str(seg_dir),
    route_id=route_id,
    segment=segment,
    has_rlog=has_rlog,
    has_qlog=has_qlog,
    has_modelv2=False,
    has_longplan=False,
    has_radar=False,
    sample_count=0,
    moving_samples=0,
    active_samples=0,
    long_active_samples=0,
    long_control_state_counts={},
    engaged_source_counts={},
  )
  if log_path is None:
    return base_census

  read_mode = ReadMode.QLOG if (prefer_qlog or not has_rlog) else ReadMode.AUTO
  state: dict = {
    "selfdrive_active": False,
    "long_active": False,
    "long_control_state": "unknown",
    "plan_source": "",
    "v_ego": 0.0,
    "a_ego": 0.0,
    "gas": False,
    "brake": False,
    "lead_status": False,
  }
  states: list[dict] = []
  override_events: list[OverrideEvent] = []
  has_modelv2 = False
  has_longplan = False
  has_radar = False
  long_state_counts: Counter = Counter()
  engaged_source_counts: Counter = Counter()
  sample_count = 0
  moving_samples = 0
  active_samples = 0
  long_active_samples = 0
  engaged_stops = 0
  prev_active = False

  for msg in LogReader(str(log_path), default_mode=read_mode, sort_by_time=True):
    typ = msg.which()
    payload = msg_payload(msg)
    if typ == "selfdriveState":
      state["selfdrive_active"] = bool(safe_get(payload, "active", False))
    elif typ == "carControl":
      state["long_active"] = bool(safe_get(payload, "longActive", False))
    elif typ == "controlsState":
      state["long_control_state"] = str(safe_get(payload, "longControlState", "unknown"))
    elif typ == "modelV2":
      has_modelv2 = True
    elif typ == "longitudinalPlan":
      has_longplan = True
      state["plan_source"] = str(safe_get(payload, "longitudinalPlanSource", ""))
    elif typ == "radarState":
      has_radar = True
      lead = safe_get(payload, "leadOne", {})
      state["lead_status"] = bool(safe_get(lead, "present", False))
    elif typ == "carState":
      v_ego = safe_get(payload, "vEgo")
      a_ego = safe_get(payload, "aEgo")
      if not isinstance(v_ego, (int, float)) or not isinstance(a_ego, (int, float)):
        continue
      state["v_ego"] = float(v_ego)
      state["a_ego"] = float(a_ego)
      state["gas"] = bool(safe_get(payload, "gasPressed", False))
      state["brake"] = bool(safe_get(payload, "brakePressed", False))
      now_active = bool(state["selfdrive_active"])
      long_active = bool(state["long_active"])
      long_state = str(state["long_control_state"])
      long_state_counts[long_state] += 1
      if now_active and long_active and state["plan_source"] in ENGAGED_PLANNER_SOURCES:
        engaged_source_counts[state["plan_source"]] += 1
      override = classify_override(prev_active, now_active, state["gas"], state["brake"], state["v_ego"])
      if override:
        kind, text = override
        override_events.append(OverrideEvent(
          route=str(seg_dir),
          route_id=route_id,
          segment=segment,
          start_time_s=float(safe_get(payload, "logMonoTime", 0)) * 1e-9,
          end_time_s=float(safe_get(payload, "logMonoTime", 0)) * 1e-9,
          kind=kind,
          active_before=prev_active,
          reason_text=text,
        ))
      prev_active = now_active
      if now_active and state["v_ego"] < STOP_SPEED:
        engaged_stops += 1
      sample_count += 1
      if state["v_ego"] > SPEED_MOVING:
        moving_samples += 1
        if now_active:
          active_samples += 1
        if long_active:
          long_active_samples += 1
        states.append({
          "route": str(seg_dir),
          "route_id": route_id,
          "segment": segment,
          "t": float(safe_get(payload, "logMonoTime", 0)) * 1e-9,
          "v_ego": state["v_ego"],
          "a_ego": state["a_ego"],
          "gas": state["gas"],
          "brake": state["brake"],
          "active": now_active,
          "long_active": long_active,
          "lead_status": state["lead_status"],
        })

  leadless = detect_slowing_runs(states, SLOWING_DURATION_S)
  return SegmentCensus(
    route=str(seg_dir),
    route_id=route_id,
    segment=segment,
    has_rlog=has_rlog,
    has_qlog=has_qlog,
    has_modelv2=has_modelv2,
    has_longplan=has_longplan,
    has_radar=has_radar,
    sample_count=sample_count,
    moving_samples=moving_samples,
    active_samples=active_samples,
    long_active_samples=long_active_samples,
    long_control_state_counts=dict(long_state_counts),
    engaged_source_counts=dict(engaged_source_counts),
    override_events=override_events,
    leadless_windows=leadless,
    engaged_stops=engaged_stops,
  )


def build_census(roots: list[Path], prefer_qlog: bool = False) -> CensusSummary:
  segments: list[SegmentCensus] = []
  for seg_dir, route_id, segment in list_route_segments(roots):
    segments.append(census_segment(seg_dir, route_id, segment, prefer_qlog))
  total_samples = sum(s.sample_count for s in segments)
  total_moving = sum(s.moving_samples for s in segments)
  total_active = sum(s.active_samples for s in segments)
  total_long_active = sum(s.long_active_samples for s in segments)
  total_overrides = sum(len(s.override_events) for s in segments)
  total_leadless = sum(len(s.leadless_windows) for s in segments)
  total_engaged_stops = sum(s.engaged_stops for s in segments)
  shadow_eligible = sum(1 for s in segments if s.has_rlog and s.has_modelv2 and s.has_radar)
  return CensusSummary(
    total_segments=len(segments),
    shadow_eligible_segments=shadow_eligible,
    total_samples=total_samples,
    moving_samples=total_moving,
    active_samples=total_active,
    long_active_samples=total_long_active,
    override_event_count=total_overrides,
    leadless_window_count=total_leadless,
    engaged_stop_count=total_engaged_stops,
    segments=segments,
  )


def summary_to_dict(summary: CensusSummary) -> dict:
  return asdict(summary)


def render_census(summary: CensusSummary, max_segments: int = 30) -> str:
  lines = [
    "Drive Lab data census",
    f"segments={summary.total_segments} shadow_eligible={summary.shadow_eligible_segments} "
    f"samples={summary.total_samples} moving={summary.moving_samples} "
    f"active={summary.active_samples} long_active={summary.long_active_samples}",
    f"overrides={summary.override_event_count} leadless_windows={summary.leadless_window_count} "
    f"engaged_stops={summary.engaged_stop_count}",
    "Per-segment summary:",
  ]
  for seg in summary.segments[:max_segments]:
    seg_label = f"{seg.route_id}--{seg.segment}" if seg.segment is not None else seg.route_id
    lines.append(
      f"  {seg_label}: rlog={seg.has_rlog} qlog={seg.has_qlog} modelv2={seg.has_modelv2} "
      f"radar={seg.has_radar} moving={seg.moving_samples} active={seg.active_samples} "
      f"long_active={seg.long_active_samples} overrides={len(seg.override_events)} "
      f"leadless={len(seg.leadless_windows)} engaged_stops={seg.engaged_stops} "
      f"engaged_sources={seg.engaged_source_counts or 'none'}"
    )
  if len(summary.segments) > max_segments:
    lines.append(f"  ... and {len(summary.segments) - max_segments} more segments")
  return "\n".join(lines)


def main() -> None:
  parser = argparse.ArgumentParser(description="Catalog drive data available for longitudinal signal scoring.")
  parser.add_argument("roots", nargs="+", type=Path, help="Realdata root directories (e.g. /data/media/0/realdata)")
  parser.add_argument("--qlog", action="store_true", help="Prefer qlogs over rlogs for sample extraction")
  parser.add_argument("--json", action="store_true", help="Emit JSON instead of text")
  parser.add_argument("--output", type=Path, help="Write JSON census to this path")
  parser.add_argument("--max-segments", type=int, default=30)
  args = parser.parse_args()
  summary = build_census(args.roots, prefer_qlog=args.qlog)
  if args.output:
    args.output.write_text(json.dumps(summary_to_dict(summary), indent=2))
  if args.json:
    print(json.dumps(summary_to_dict(summary), indent=2))
  else:
    print(render_census(summary, max_segments=args.max_segments))


if __name__ == "__main__":
  main()
