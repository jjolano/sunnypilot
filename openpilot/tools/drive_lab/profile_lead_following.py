#!/usr/bin/env python3
"""Profile engaged lead-following: steady headway, approach harshness, and reactive braking.

Quantifies the "hangs back / brakes reactively" behaviour for the hypermile-following work, and is
the regression gate those longitudinal-policy changes must beat. Three views, all on engaged
(`carControl.longActive`) cruise-following above ~8 m/s with a real lead:

- **Steady following** (|vRel| < 1 m/s): time headway dRel/vEgo and absolute gap — how far back it sits.
- **Approach to a slower lead** (vRel < −1 m/s): peak decel per closing episode, headway at the start
  of the approach (early vs late), and how often the lead then *sped back up* (braked for nothing).
- **Braking while following**: how much of the braking happens while the lead is actually accelerating.

Reference (fork `T_FOLLOW`): ~1.20 s aggressive / ~1.45 s normal / ~1.75 s relaxed.

Ported from a `/btw` analysis script onto drive_lab conventions.
Signals: carState.vEgo/aEgo, carControl.longActive, radarState.leadOne.{dRel,vRel,vLead,aLeadK,present,modelProb}.

Run:
  uv run python -m openpilot.tools.drive_lab.profile_lead_following ROUTE
  uv run python -m openpilot.tools.drive_lab.profile_lead_following ROUTE --json --output /tmp/x.json
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from openpilot.tools.drive_lab.route_analysis import build_route_messages
from openpilot.tools.drive_lab.route_io import load_route_msgs, output_report
from openpilot.tools.drive_lab.timeline import safe_get


@dataclass(frozen=True)
class LeadFollowParams:
  v_min: float = 8.0         # m/s; only consider cruise-following at/above this speed
  steady_vrel: float = 1.0   # m/s; |vRel| below this is steady following
  close_vrel: float = -1.0   # m/s; vRel below this starts a closing/approach episode
  close_end_vrel: float = -0.5  # m/s; episode continues while vRel below this
  brake_a: float = -0.4      # m/s^2; aEgo below this counts as braking
  lead_accel: float = 0.2    # m/s^2; aLeadK above this means the lead is accelerating
  min_close_samples: int = 3  # minimum samples for a valid approach episode
  lead_model_prob: float = 0.5  # minimum lead modelProb to trust the lead


@dataclass
class LeadFollowReport:
  source: str
  duration_s: float
  follow_samples: int
  follow_minutes: float
  steady_samples: int
  thw_median: float | None
  thw_p10: float | None
  thw_p90: float | None
  thw_share_above_2s: float | None
  gap_median: float | None
  gap_p90: float | None
  approach_events: int
  peak_decel_median: float | None
  peak_decel_harshest: float | None
  approach_start_thw_median: float | None
  lead_resumed: int
  lead_resumed_frac: float | None
  brake_samples: int
  brake_lead_accel: int
  brake_lead_accel_frac: float | None
  notes: list[str]

  def to_dict(self) -> dict[str, Any]:
    return {
      "source": self.source,
      "duration_s": _r(self.duration_s, 2),
      "follow_samples": self.follow_samples,
      "follow_minutes": _r(self.follow_minutes, 1),
      "steady_samples": self.steady_samples,
      "thw_median": _r(self.thw_median, 2),
      "thw_p10": _r(self.thw_p10, 2),
      "thw_p90": _r(self.thw_p90, 2),
      "thw_share_above_2s": _r(self.thw_share_above_2s, 3),
      "gap_median": _r(self.gap_median, 1),
      "gap_p90": _r(self.gap_p90, 1),
      "approach_events": self.approach_events,
      "peak_decel_median": _r(self.peak_decel_median, 3),
      "peak_decel_harshest": _r(self.peak_decel_harshest, 3),
      "approach_start_thw_median": _r(self.approach_start_thw_median, 2),
      "lead_resumed": self.lead_resumed,
      "lead_resumed_frac": _r(self.lead_resumed_frac, 3),
      "brake_samples": self.brake_samples,
      "brake_lead_accel": self.brake_lead_accel,
      "brake_lead_accel_frac": _r(self.brake_lead_accel_frac, 3),
      "notes": list(self.notes),
    }


def _r(value: Any, ndigits: int = 6) -> float | None:
  if value is None or not isinstance(value, int | float) or not math.isfinite(float(value)):
    return None
  return round(float(value), ndigits)


def _f(value: Any) -> float:
  try:
    out = float(value)
  except (TypeError, ValueError):
    return math.nan
  return out if math.isfinite(out) else math.nan


def _pct(values: list[float], q: float) -> float | None:
  return float(np.percentile(values, q)) if values else None


@dataclass(frozen=True)
class _FollowSample:
  v: float
  a: float
  d_rel: float
  v_rel: float
  v_lead: float
  a_lead: float


def _collect(msgs: list[Any], p: LeadFollowParams) -> list[_FollowSample]:
  """Engaged cruise-following samples (triggered on radarState, zero-order-holding carState/carControl)."""
  latest: dict[str, Any] = {}
  out: list[_FollowSample] = []
  for record in build_route_messages(msgs):
    typ, payload = record.typ, record.payload
    if typ in ("carState", "carControl"):
      latest[typ] = payload
      continue
    if typ != "radarState":
      continue
    cs, cc = latest.get("carState"), latest.get("carControl")
    if cs is None or cc is None or not bool(safe_get(cc, "longActive", False)):
      continue
    v = _f(safe_get(cs, "vEgo"))
    if not (math.isfinite(v) and v >= p.v_min):
      continue
    lead = safe_get(payload, "leadOne")
    if lead is None or not bool(safe_get(lead, "present", False)) or _f(safe_get(lead, "modelProb")) < p.lead_model_prob:
      continue
    d_rel = _f(safe_get(lead, "dRel"))
    if not (math.isfinite(d_rel) and d_rel > 0.5):
      continue
    out.append(_FollowSample(v, _f(safe_get(cs, "aEgo")), d_rel, _f(safe_get(lead, "vRel")),
                             _f(safe_get(lead, "vLead")), _f(safe_get(lead, "aLeadK"))))
  return out


def analyze_route(msgs: list[Any], source: str = "unknown", params: LeadFollowParams | None = None) -> LeadFollowReport:
  p = params or LeadFollowParams()
  records = build_route_messages(msgs)
  duration = (records[-1].t - records[0].t) if records else 0.0
  series = _collect(msgs, p)
  notes: list[str] = []

  thw_steady, gap_steady = [], []
  brake_total = brake_lead_accel = 0
  for s in series:
    if math.isfinite(s.v_rel) and abs(s.v_rel) < p.steady_vrel and s.v > 0:
      thw_steady.append(s.d_rel / s.v)
      gap_steady.append(s.d_rel)
    if math.isfinite(s.a) and s.a < p.brake_a:
      brake_total += 1
      if math.isfinite(s.a_lead) and s.a_lead > p.lead_accel:
        brake_lead_accel += 1

  peak_decel, start_thw, lead_resumed = [], [], 0
  i, n = 0, len(series)
  while i < n:
    if not (math.isfinite(series[i].v_rel) and series[i].v_rel < p.close_vrel):
      i += 1
      continue
    j, min_a, vlead_lo, vlead_end = i, math.inf, math.inf, series[i].v_lead
    thw0 = series[i].d_rel / series[i].v if series[i].v > 0 else math.nan
    while j < n and math.isfinite(series[j].v_rel) and series[j].v_rel < p.close_end_vrel:
      if math.isfinite(series[j].a):
        min_a = min(min_a, series[j].a)
      if math.isfinite(series[j].v_lead):
        vlead_lo = min(vlead_lo, series[j].v_lead)
        vlead_end = series[j].v_lead
      j += 1
    if math.isfinite(min_a) and (j - i) >= p.min_close_samples:
      peak_decel.append(min_a)
      if math.isfinite(thw0):
        start_thw.append(thw0)
      if math.isfinite(vlead_end) and math.isfinite(vlead_lo) and vlead_end > vlead_lo + 1.0:
        lead_resumed += 1
    i = max(j, i + 1)

  if not series:
    notes.append(f"no engaged cruise-following with a lead above {p.v_min} m/s (need rlogs w/ carControl+radarState)")

  return LeadFollowReport(
    source=source,
    duration_s=duration,
    follow_samples=len(series),
    follow_minutes=len(series) * 0.05 / 60.0,
    steady_samples=len(thw_steady),
    thw_median=_pct(thw_steady, 50),
    thw_p10=_pct(thw_steady, 10),
    thw_p90=_pct(thw_steady, 90),
    thw_share_above_2s=(sum(1 for x in thw_steady if x > 2.0) / len(thw_steady)) if thw_steady else None,
    gap_median=_pct(gap_steady, 50),
    gap_p90=_pct(gap_steady, 90),
    approach_events=len(peak_decel),
    peak_decel_median=_pct(peak_decel, 50),
    peak_decel_harshest=_pct(peak_decel, 10),
    approach_start_thw_median=_pct(start_thw, 50),
    lead_resumed=lead_resumed,
    lead_resumed_frac=(lead_resumed / len(peak_decel)) if peak_decel else None,
    brake_samples=brake_total,
    brake_lead_accel=brake_lead_accel,
    brake_lead_accel_frac=(brake_lead_accel / brake_total) if brake_total else None,
    notes=notes,
  )


def render_report(report: LeadFollowReport) -> str:
  lines = [
    f"Lead-following profile: {report.source}",
    f"  duration {report.duration_s/60:.1f} min, engaged cruise-following ~{report.follow_minutes:.0f} min "
    + f"({report.follow_samples} samples)",
  ]
  if report.steady_samples:
    lines += [
      "",
      f"  steady following ({report.steady_samples} samples):",
      f"    time headway dRel/v:  median {_fmt(report.thw_median, 2)} s  "
      + f"(p10 {_fmt(report.thw_p10, 2)} / p90 {_fmt(report.thw_p90, 2)})   ref ~1.45 s normal",
      f"    share above 2.0 s:    {_fmt((report.thw_share_above_2s or 0)*100, 0)}%",
      f"    gap:                  median {_fmt(report.gap_median, 1)} m (p90 {_fmt(report.gap_p90, 1)} m)",
    ]
  if report.approach_events:
    lines += [
      "",
      f"  approach to slower lead ({report.approach_events} episodes):",
      f"    peak decel:           median {_fmt(report.peak_decel_median, 2)}  "
      + f"harshest-decile {_fmt(report.peak_decel_harshest, 2)} m/s^2",
      f"    headway at start:     median {_fmt(report.approach_start_thw_median, 2)} s",
      f"    lead then sped up:    {report.lead_resumed}/{report.approach_events} "
      + f"({_fmt((report.lead_resumed_frac or 0)*100, 0)}%)  (braked for a lead that didn't stop)",
    ]
  if report.brake_samples:
    lines.append("")
    lines.append(f"  braking while following: {report.brake_samples} samples; lead was accelerating in "
                 + f"{_fmt((report.brake_lead_accel_frac or 0)*100, 0)}%")
  for note in report.notes:
    lines.append(f"  note: {note}")
  return "\n".join(lines)


def _fmt(value: float | None, ndigits: int = 2) -> str:
  return "n/a" if value is None or not math.isfinite(float(value)) else f"{float(value):.{ndigits}f}"


def main() -> None:
  parser = argparse.ArgumentParser(description="Profile engaged lead-following (headway, approach, reactive braking).")
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
