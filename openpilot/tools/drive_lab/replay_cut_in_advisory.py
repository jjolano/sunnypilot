#!/usr/bin/env python3
"""Replay a cut-in advisory candidate over route logs to validate timing advantage and false-positive rate.

Simulates the two-level cut-in advisory (SUSPECT → CONFIRMED) with all gates from the high-reasoning
agent's design, and measures:

1. When the advisory would fire vs when MPC actually brakes (timing advantage)
2. False-positive rate (advisory fires but MPC never brakes within 5s)
3. Peak decel with vs without advisory (estimated)
4. Cut-in detection breakdown by gate (why candidates were rejected)

Does NOT modify any production code. Pure offline replay over qlog/rlog data.

Run:
  uv run python -m openpilot.tools.drive_lab.replay_cut_in_advisory ROUTE --qlog
  uv run python -m openpilot.tools.drive_lab.replay_cut_in_advisory ROUTE --qlog --json --output /tmp/x.json
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
class CutInAdvisoryParams:
  # Risk gate
  min_closing_speed: float = 2.0       # m/s; -vRel must exceed this
  max_d_rel: float = 30.0             # m; only consider cut-ins within this distance
  max_time_gap: float = 1.8          # s; dRel/vEgo must be below this (alternative to dRel)
  max_ttc: float = 8.0               # s; TTC must be below this to be a candidate

  # Path plausibility gate
  max_y_rel_strict: float = 1.2       # m; abs(yRel) below this = same lane
  max_y_rel_loose: float = 2.0       # m; abs(yRel) below this + lateral motion toward ego lane
  min_model_prob: float = 0.5        # minimum modelProb for vision corroboration

  # Persistence gate (in frames)
  suspect_frames: int = 2            # frames to reach SUSPECT
  confirmed_frames: int = 4          # frames to reach CONFIRMED

  # Kinematic consistency gate
  max_kinematic_error: float = 3.0   # m/s; max |(dRel_now - dRel_prev)/dt - vRel_now|

  # Track-swap suppression
  swap_max_d_rel: float = 4.0        # m; new lead within this of old lead's dRel = swap
  swap_max_v_rel: float = 2.0        # m/s; new lead within this of old lead's vRel = swap
  swap_max_y_rel: float = 1.0        # m; new lead within this of old lead's yRel = swap

  # Moving-object gate
  min_v_lead: float = 2.0           # m/s; lead must be moving (unless high modelProb)

  # Advisory ramp
  suspect_advisory: float = -0.20    # m/s²; SUSPECT level cap
  confirmed_advisory_low: float = -0.25  # m/s²; CONFIRMED low-risk
  confirmed_advisory_high: float = -0.50  # m/s²; CONFIRMED high-risk (TTC < 5)
  confirmed_ramp_ttc: float = 5.0    # s; TTC below this ramps to high advisory
  confirmed_ramp_closing: float = 3.0  # m/s; closing speed above this ramps to high advisory
  min_advisory: float = -0.15       # m/s²; floor (never lighter than this)
  max_advisory: float = -0.50       # m/s²; ceiling (never stronger than this)

  # Slew limiting
  max_slew_per_frame: float = 0.10   # m/s² per frame; limits jerk

  # MPC brake detection (for timing comparison)
  mpc_brake_threshold: float = -0.1  # m/s²; MPC aTarget below this = "MPC braking"
  timing_window_s: float = 5.0       # s; window to look for MPC brake after advisory fires

  # Veto gates
  min_v_ego: float = 1.0            # m/s; ignore below this speed


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CutInTrack:
  """Tracks a potential cut-in candidate across frames."""
  lead_id: int
  first_seen_t: float
  frames: int = 0
  d_rel_history: list[float] = field(default_factory=list)
  v_rel_history: list[float] = field(default_factory=list)
  y_rel_history: list[float] = field(default_factory=list)
  t_history: list[float] = field(default_factory=list)
  last_advisory: float = 0.0
  stage: str = "none"                # "none", "suspect", "confirmed"
  fired_t: float | None = None       # when advisory first fired
  fired_advisory: float = 0.0


@dataclass
class CutInAdvisoryEvent:
  t: float
  lead_id: int
  stage: str                          # "suspect" or "confirmed"
  advisory_a: float
  d_rel: float
  v_rel: float
  y_rel: float
  v_lead: float
  v_ego: float
  ttc: float | None
  model_prob: float
  op_engaged: bool
  mpc_brake_t: float | None          # when MPC first brakes after advisory
  mpc_brake_a: float | None          # MPC aTarget at brake onset
  timing_advantage_s: float | None   # mpc_brake_t - advisory_t (positive = advisory was earlier)
  false_positive: bool               # MPC never braked within timing_window


@dataclass
class CutInReplayReport:
  source: str
  duration_s: float
  op_engaged_s: float
  manual_moving_s: float
  total_cut_in_candidates: int
  suspect_events: int
  confirmed_events: int
  events: list[CutInAdvisoryEvent] = field(default_factory=list)
  rejection_reasons: dict[str, int] = field(default_factory=dict)
  notes: list[str] = field(default_factory=list)

  def to_dict(self) -> dict[str, Any]:
    timing_advantages = [e.timing_advantage_s for e in self.events if e.timing_advantage_s is not None]
    false_positives = [e for e in self.events if e.false_positive]
    op_events = [e for e in self.events if e.op_engaged]
    manual_events = [e for e in self.events if not e.op_engaged]

    return {
      "source": self.source,
      "duration_s": _r(self.duration_s, 2),
      "op_engaged_s": _r(self.op_engaged_s, 2),
      "manual_moving_s": _r(self.manual_moving_s, 2),
      "total_cut_in_candidates": self.total_cut_in_candidates,
      "suspect_events": self.suspect_events,
      "confirmed_events": self.confirmed_events,
      "summary": {
        "total_advisory_events": len(self.events),
        "op_events": len(op_events),
        "manual_events": len(manual_events),
        "timing_advantage_median_s": _r(_opt_median(timing_advantages), 3),
        "timing_advantage_p10_s": _r(_opt_percentile(timing_advantages, 10), 3),
        "timing_advantage_p90_s": _r(_opt_percentile(timing_advantages, 90), 3),
        "false_positive_count": len(false_positives),
        "false_positive_rate": _r(len(false_positives) / len(self.events) if self.events else 0.0, 3),
        "op_timing_advantage_median_s": _r(_opt_median([e.timing_advantage_s for e in op_events if e.timing_advantage_s is not None]), 3),
        "manual_timing_advantage_median_s": _r(_opt_median([e.timing_advantage_s for e in manual_events if e.timing_advantage_s is not None]), 3),
      },
      "rejection_reasons": dict(self.rejection_reasons),
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


def _opt_percentile(values: list[float], p: float) -> float | None:
  return float(np.percentile(values, p)) if values else None


def _event_to_dict(e: CutInAdvisoryEvent) -> dict[str, Any]:
  return {
    "t": _r(e.t, 3), "lead_id": e.lead_id, "stage": e.stage,
    "advisory_a": _r(e.advisory_a, 3),
    "d_rel": _r(e.d_rel, 2), "v_rel": _r(e.v_rel, 3), "y_rel": _r(e.y_rel, 2),
    "v_lead": _r(e.v_lead, 3), "v_ego": _r(e.v_ego, 3),
    "ttc": _r(e.ttc, 2), "model_prob": _r(e.model_prob, 3),
    "op_engaged": e.op_engaged,
    "mpc_brake_t": _r(e.mpc_brake_t, 3), "mpc_brake_a": _r(e.mpc_brake_a, 3),
    "timing_advantage_s": _r(e.timing_advantage_s, 3),
    "false_positive": e.false_positive,
  }


# ---------------------------------------------------------------------------
# Core replay
# ---------------------------------------------------------------------------

@dataclass
class _RadarLead:
  t: float
  d_rel: float
  y_rel: float
  v_rel: float
  v_lead: float
  v_lead_k: float
  a_lead: float
  status: bool
  lead_id: int
  model_prob: float
  radar: bool


@dataclass
class _EgoState:
  t: float
  v: float
  a: float
  a_target: float | None
  long_active: bool


def _collect(msgs: list[Any]) -> tuple[list[_RadarLead], list[_EgoState], float, float]:
  """Collect radar lead data (both leadOne and leadTwo) and ego state."""
  radar_leads: list[_RadarLead] = []
  ego: list[_EgoState] = []
  op_engaged_s = 0.0
  manual_moving_s = 0.0
  last_t: float | None = None
  latest: dict[str, Any] = {}

  for rec in build_route_messages(msgs):
    if rec.typ in ("carState", "carControl", "radarState", "longitudinalPlanSP"):
      latest[rec.typ] = rec.payload

    if rec.typ == "carState":
      cs = rec.payload
      cc = latest.get("carControl")
      sp = latest.get("longitudinalPlanSP")
      v = _f(safe_get(cs, "vEgo"))
      a = _f(safe_get(cs, "aEgo"))
      long_active = bool(safe_get(cc, "longActive", False))
      a_target = _f(safe_get(sp, "aTarget"), float("nan")) if sp is not None else float("nan")

      if last_t is not None and math.isfinite(v):
        dt = max(0.0, min(2.0, rec.t - last_t))
        if long_active:
          op_engaged_s += dt
        elif v > 0.1:
          manual_moving_s += dt
      last_t = rec.t

      if math.isfinite(v) and math.isfinite(a):
        ego.append(_EgoState(t=rec.t, v=v, a=a,
                             a_target=a_target if math.isfinite(a_target) else None,
                             long_active=long_active))

    elif rec.typ == "radarState":
      for lead_name in ("leadOne", "leadTwo"):
        lead = safe_get(rec.payload, lead_name)
        if lead is None:
          continue
        status = bool(safe_get(lead, "present", False))
        d_rel = _f(safe_get(lead, "dRel"))
        y_rel = _f(safe_get(lead, "yRel"))
        v_rel = _f(safe_get(lead, "vRel"))
        v_lead = _f(safe_get(lead, "vLead"))
        v_lead_k = _f(safe_get(lead, "vLeadK"))
        a_lead = _f(safe_get(lead, "aLeadK"))
        lead_id = safe_get(lead, "radarTrackId", -1)
        model_prob = _f(safe_get(lead, "modelProb"))
        is_radar = bool(safe_get(lead, "radar", False))
        if not (math.isfinite(d_rel) and math.isfinite(v_rel) and math.isfinite(v_lead)):
          continue
        radar_leads.append(_RadarLead(t=rec.t, d_rel=d_rel, y_rel=y_rel, v_rel=v_rel,
                                     v_lead=v_lead, v_lead_k=v_lead_k, a_lead=a_lead,
                                     status=status, lead_id=int(lead_id) if lead_id is not None else -1,
                                     model_prob=model_prob if math.isfinite(model_prob) else 0.0,
                                     radar=is_radar))

  return radar_leads, ego, op_engaged_s, manual_moving_s


def _nearest_ego(ego: list[_EgoState], t: float) -> _EgoState | None:
  if not ego:
    return None
  lo, hi = 0, len(ego) - 1
  if t <= ego[0].t:
    return ego[0]
  if t >= ego[-1].t:
    return ego[-1]
  while lo < hi:
    mid = (lo + hi) // 2
    if ego[mid].t < t:
      lo = mid + 1
    else:
      hi = mid
  return ego[lo]


def _check_track_swap(new: _RadarLead, old_lead_id: int, old_d_rel: float,
                       old_v_rel: float, old_y_rel: float, p: CutInAdvisoryParams) -> bool:
  """Return True if this looks like a track swap (same object, new ID)."""
  if old_lead_id < 0:
    return False
  return (abs(new.d_rel - old_d_rel) < p.swap_max_d_rel and
          abs(new.v_rel - old_v_rel) < p.swap_max_v_rel and
          abs(new.y_rel - old_y_rel) < p.swap_max_y_rel)


def _compute_advisory(track: CutInTrack, p: CutInAdvisoryParams) -> float:
  """Compute the advisory acceleration for a track at its current stage."""
  if track.stage == "none":
    return 0.0

  d_rel = track.d_rel_history[-1]
  v_rel = track.v_rel_history[-1]
  closing_speed = max(0.0, -v_rel)
  ttc = d_rel / max(closing_speed, 0.1) if closing_speed > 0.1 else None

  if track.stage == "suspect":
    target = p.suspect_advisory
  else:
    # Confirmed: ramp based on risk
    target = p.confirmed_advisory_low
    if ttc is not None and ttc < p.confirmed_ramp_ttc:
      target = p.confirmed_advisory_high
    if closing_speed > p.confirmed_ramp_closing:
      target = p.confirmed_advisory_high

  # Slew limit
  slewed = max(target, track.last_advisory - p.max_slew_per_frame)
  return max(p.max_advisory, min(p.min_advisory, slewed))


def _evaluate_gates(lead: _RadarLead, ego: _EgoState | None, p: CutInAdvisoryParams,
                     rejection_reasons: dict[str, int]) -> bool:
  """Evaluate all gates for a radar lead. Returns True if candidate passes all gates."""
  if ego is None:
    rejection_reasons["no_ego"] = rejection_reasons.get("no_ego", 0) + 1
    return False
  if ego.v < p.min_v_ego:
    rejection_reasons["low_speed"] = rejection_reasons.get("low_speed", 0) + 1
    return False
  if not lead.status:
    rejection_reasons["no_status"] = rejection_reasons.get("no_status", 0) + 1
    return False
  if lead.d_rel <= 0 or lead.d_rel > p.max_d_rel:
    rejection_reasons["d_rel_out_of_range"] = rejection_reasons.get("d_rel_out_of_range", 0) + 1
    return False

  closing_speed = max(0.0, -lead.v_rel)
  if closing_speed < p.min_closing_speed:
    rejection_reasons["low_closing_speed"] = rejection_reasons.get("low_closing_speed", 0) + 1
    return False

  time_gap = lead.d_rel / max(ego.v, 0.1)
  if time_gap > p.max_time_gap and lead.d_rel > p.max_d_rel:
    rejection_reasons["far_time_gap"] = rejection_reasons.get("far_time_gap", 0) + 1
    return False

  ttc = lead.d_rel / max(closing_speed, 0.1) if closing_speed > 0.1 else None
  if ttc is not None and ttc > p.max_ttc:
    rejection_reasons["high_ttc"] = rejection_reasons.get("high_ttc", 0) + 1
    return False

  # Path plausibility
  if abs(lead.y_rel) > p.max_y_rel_loose:
    rejection_reasons["off_path"] = rejection_reasons.get("off_path", 0) + 1
    return False
  if abs(lead.y_rel) > p.max_y_rel_strict and lead.model_prob < p.min_model_prob:
    rejection_reasons["lateral_no_vision"] = rejection_reasons.get("lateral_no_vision", 0) + 1
    return False

  # Moving-object gate
  if lead.v_lead < p.min_v_lead and lead.model_prob < p.min_model_prob:
    rejection_reasons["stationary_no_vision"] = rejection_reasons.get("stationary_no_vision", 0) + 1
    return False

  return True


def _check_kinematic_consistency(track: CutInTrack, lead: _RadarLead, dt: float,
                                  p: CutInAdvisoryParams) -> bool:
  """Check that range change matches relative speed."""
  if len(track.d_rel_history) < 1 or dt <= 0:
    return True
  prev_d = track.d_rel_history[-1]
  expected_change = lead.v_rel * dt
  actual_change = lead.d_rel - prev_d
  error = abs(actual_change - expected_change)
  return error < p.max_kinematic_error


def analyze_route(msgs: list[Any], source: str = "unknown",
                  params: CutInAdvisoryParams | None = None) -> CutInReplayReport:
  p = params or CutInAdvisoryParams()
  radar, ego, op_engaged_s, manual_moving_s = _collect(msgs)
  notes: list[str] = []

  if not radar:
    notes.append("no radarState samples with valid lead data found")
  if not ego:
    notes.append("no carState samples found")

  duration = ego[-1].t - ego[0].t if ego else 0.0

  # Track active cut-in candidates by lead_id
  active_tracks: dict[int, CutInTrack] = {}
  events: list[CutInAdvisoryEvent] = []
  rejection_reasons: dict[str, int] = {}
  total_candidates = 0

  # Previous lead state for track-swap detection
  prev_lead_id: int | None = None
  prev_d_rel: float = 0.0
  prev_v_rel: float = 0.0
  prev_y_rel: float = 0.0
  prev_t: float | None = None

  # Group radar samples by time (leadOne + leadTwo at same timestamp)
  radar_by_time: dict[float, list[_RadarLead]] = {}
  for r in radar:
    radar_by_time.setdefault(r.t, []).append(r)

  for t in sorted(radar_by_time.keys()):
    leads_at_t = radar_by_time[t]
    ego_state = _nearest_ego(ego, t)
    dt = (t - prev_t) if prev_t is not None else 0.1
    dt = max(0.01, min(0.5, dt))

    # Track which lead_ids are seen this frame
    seen_ids: set[int] = set()

    for lead in leads_at_t:
      if not lead.status or lead.lead_id < 0:
        continue
      seen_ids.add(lead.lead_id)

      # Check if this is a new cut-in candidate
      is_new = lead.lead_id not in active_tracks

      if is_new:
        # Track-swap suppression
        if prev_lead_id is not None and _check_track_swap(lead, prev_lead_id, prev_d_rel,
                                                           prev_v_rel, prev_y_rel, p):
          rejection_reasons["track_swap"] = rejection_reasons.get("track_swap", 0) + 1
          continue

        # Evaluate gates
        if not _evaluate_gates(lead, ego_state, p, rejection_reasons):
          continue

        total_candidates += 1
        active_tracks[lead.lead_id] = CutInTrack(
          lead_id=lead.lead_id, first_seen_t=t, frames=1,
          d_rel_history=[lead.d_rel], v_rel_history=[lead.v_rel],
          y_rel_history=[lead.y_rel], t_history=[t],
          stage="none",
        )
      else:
        track = active_tracks[lead.lead_id]

        # Kinematic consistency check
        if not _check_kinematic_consistency(track, lead, dt, p):
          rejection_reasons["kinematic_inconsistency"] = rejection_reasons.get("kinematic_inconsistency", 0) + 1
          del active_tracks[lead.lead_id]
          continue

        # Re-evaluate gates
        if not _evaluate_gates(lead, ego_state, p, rejection_reasons):
          if track.stage != "none":
            # Was active, now fails gates — close it
            pass
          del active_tracks[lead.lead_id]
          continue

        track.frames += 1
        track.d_rel_history.append(lead.d_rel)
        track.v_rel_history.append(lead.v_rel)
        track.y_rel_history.append(lead.y_rel)
        track.t_history.append(t)

        # Stage transitions
        if track.stage == "none" and track.frames >= p.suspect_frames:
          track.stage = "suspect"
        if track.stage == "suspect" and track.frames >= p.confirmed_frames:
          track.stage = "confirmed"

      # Update track advisory
      if lead.lead_id in active_tracks:
        track = active_tracks[lead.lead_id]
        advisory = _compute_advisory(track, p)
        track.last_advisory = advisory

        # Fire event when advisory first becomes non-zero
        if advisory != 0.0 and track.fired_t is None:
          track.fired_t = t
          track.fired_advisory = advisory

          closing_speed = max(0.0, -lead.v_rel)
          ttc = lead.d_rel / max(closing_speed, 0.1) if closing_speed > 0.1 else None

          events.append(CutInAdvisoryEvent(
            t=t, lead_id=lead.lead_id, stage=track.stage,
            advisory_a=advisory,
            d_rel=lead.d_rel, v_rel=lead.v_rel, y_rel=lead.y_rel,
            v_lead=lead.v_lead, v_ego=ego_state.v if ego_state else 0.0,
            ttc=ttc, model_prob=lead.model_prob,
            op_engaged=ego_state.long_active if ego_state else False,
            mpc_brake_t=None, mpc_brake_a=None,
            timing_advantage_s=None, false_positive=True,
          ))

    # Remove tracks not seen this frame (lead disappeared)
    stale_ids = set(active_tracks.keys()) - seen_ids
    for sid in stale_ids:
      del active_tracks[sid]

    # Update previous lead state
    if leads_at_t:
      primary = leads_at_t[0]
      prev_lead_id = primary.lead_id
      prev_d_rel = primary.d_rel
      prev_v_rel = primary.v_rel
      prev_y_rel = primary.y_rel
    prev_t = t

  # Post-process: find MPC brake onset for each event
  for event in events:
    end_t = event.t + p.timing_window_s
    for e in ego:
      if e.t < event.t:
        continue
      if e.t > end_t:
        break
      a_target = e.a_target if e.a_target is not None else e.a
      if a_target is not None and math.isfinite(a_target) and a_target < p.mpc_brake_threshold:
        event.mpc_brake_t = e.t
        event.mpc_brake_a = a_target
        event.timing_advantage_s = float(e.t - event.t)
        event.false_positive = False
        break

  suspect_count = sum(1 for e in events if e.stage == "suspect")
  confirmed_count = sum(1 for e in events if e.stage == "confirmed")

  return CutInReplayReport(
    source=source,
    duration_s=duration,
    op_engaged_s=op_engaged_s,
    manual_moving_s=manual_moving_s,
    total_cut_in_candidates=total_candidates,
    suspect_events=suspect_count,
    confirmed_events=confirmed_count,
    events=events,
    rejection_reasons=dict(sorted(rejection_reasons.items())),
    notes=notes,
  )


def render_report(report: CutInReplayReport) -> str:
  lines = [
    f"Cut-in advisory replay: {report.source}",
    f"  duration {report.duration_s:.1f}s, OP engaged {report.op_engaged_s:.1f}s, "
    f"manual moving {report.manual_moving_s:.1f}s",
    f"  total cut-in candidates: {report.total_cut_in_candidates}",
    f"  suspect events: {report.suspect_events}, confirmed events: {report.confirmed_events}",
  ]

  s = report.to_dict()["summary"]
  lines += [
    "",
    "  advisory timing vs MPC brake onset:",
    f"    total events:     {s['total_advisory_events']} ({s['op_events']} OP, {s['manual_events']} manual)",
    f"    timing advantage: median {s['timing_advantage_median_s']} s  "
    f"(p10 {s['timing_advantage_p10_s']}, p90 {s['timing_advantage_p90_s']})",
    f"    OP advantage:     median {s['op_timing_advantage_median_s']} s",
    f"    manual advantage: median {s['manual_timing_advantage_median_s']} s",
    f"    false positives:  {s['false_positive_count']} ({s['false_positive_rate'] * 100:.1f}%)",
  ]

  if report.rejection_reasons:
    lines += ["", "  rejection reasons:"]
    for reason, count in sorted(report.rejection_reasons.items(), key=lambda x: -x[1]):
      lines.append(f"    {reason}: {count}")

  # Per-event detail for small counts
  op_events = [e for e in report.events if e.op_engaged]
  if len(op_events) <= 15:
    lines += ["", "  OP events:"]
    for e in op_events:
      lines.append(
        f"    t={e.t:.1f} id={e.lead_id} stage={e.stage} d={e.d_rel:.1f}m vR={e.v_rel:.1f} "
        f"ttc={_fmt(e.ttc, 1)} advisory={e.advisory_a:.2f} "
        f"mpc_brake={_fmt(e.mpc_brake_t, 1)} advantage={_fmt(e.timing_advantage_s, 2)}s fp={e.false_positive}"
      )

  for note in report.notes:
    lines.append(f"  note: {note}")
  return "\n".join(lines)


def _fmt(value: float | None, ndigits: int = 3) -> str:
  return "n/a" if value is None or not math.isfinite(float(value)) else f"{float(value):.{ndigits}f}"


def main() -> None:
  parser = argparse.ArgumentParser(description="Replay cut-in advisory candidate over route logs.")
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
