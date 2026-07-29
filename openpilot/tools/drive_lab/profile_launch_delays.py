#!/usr/bin/env python3
"""Profile longitudinal launch behaviour: how the car accelerates away from a near-stop.

Finds speed-recovery events — the car slows below ~5 m/s, dips to a near-stop, then accelerates
away — and for each measures recovery time, acceleration, and jerk, split by openpilot-engaged vs
manual and lead vs no-lead. The lead/no-lead split exposes whether launches are sluggish only when
following (a longitudinal-policy question), independent of the lateral corner-amnesia that
``profile_corner_recovery`` measures.

Ported from a device-side script (scripts/analyze_launch_delays.py); reworked onto the drive_lab
conventions with time-based (rate-independent) windows.

Signals: carState.vEgo / aEgo, selfdriveState.enabled, radarState.leadOne.{dRel,vLead,status}.

Run:
  uv run python -m openpilot.tools.drive_lab.profile_launch_delays ROUTE
  uv run python -m openpilot.tools.drive_lab.profile_launch_delays ROUTE --json --output /tmp/x.json
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
class LaunchParams:
  speed_threshold: float = 5.0  # m/s; below this we may be approaching a stop
  near_stop_speed: float = 2.0  # m/s; min speed must dip below this to count as a launch
  recovery_delta: float = 0.5  # m/s; "recovered" once speed rises this far above the dip
  recovery_max_s: float = 20.0  # give up looking for recovery after this long
  match_window_s: float = 1.0  # how close (s) an enabled/lead sample must be to the dip time
  lead_d_rel_max: float = 199.0  # m; a lead counts as present only within this distance


@dataclass
class LaunchEvent:
  time: float
  min_speed: float
  recovery_speed: float
  recovery_time: float
  lead_move_time: float | None
  lead_wait_time: float | None
  reaction_time: float | None
  planner_release_time: float | None
  planner_release_delay: float | None
  ego_start_time: float | None
  ego_start_delay: float | None
  op_engaged: bool
  lead_present: bool
  lead_d_rel: float | None
  accel_mean: float
  accel_peak: float
  jerk_peak: float
  long_active: bool
  driver_override: bool
  enabled: bool

  def to_dict(self) -> dict[str, Any]:
    return {
      "time": _r(self.time, 2),
      "min_speed": _r(self.min_speed, 3),
      "recovery_speed": _r(self.recovery_speed, 3),
      "recovery_time": _r(self.recovery_time, 3),
      "lead_move_time": _r(self.lead_move_time, 3),
      "lead_wait_time": _r(self.lead_wait_time, 3),
      "reaction_time": _r(self.reaction_time, 3),
      "planner_release_time": _r(self.planner_release_time, 3),
      "planner_release_delay": _r(self.planner_release_delay, 3),
      "ego_start_time": _r(self.ego_start_time, 3),
      "ego_start_delay": _r(self.ego_start_delay, 3),
      "op_engaged": self.op_engaged,
      "lead_present": self.lead_present,
      "lead_d_rel": _r(self.lead_d_rel, 1),
      "accel_mean": _r(self.accel_mean, 3),
      "accel_peak": _r(self.accel_peak, 3),
      "jerk_peak": _r(self.jerk_peak, 2),
      "long_active": self.long_active,
      "driver_override": self.driver_override,
      "enabled": self.enabled,
    }


@dataclass
class LaunchReport:
  source: str
  duration_s: float
  total_events: int
  op_engaged_events: int
  op_lead_events: int
  op_nolead_events: int
  manual_events: int
  op_lead_median_recovery_s: float | None
  op_nolead_median_recovery_s: float | None
  op_lead_median_reaction_s: float | None
  op_lead_median_planner_release_s: float | None
  op_lead_median_ego_start_s: float | None
  op_lead_median_accel: float | None
  op_nolead_median_accel: float | None
  events: list[LaunchEvent]
  notes: list[str]

  def to_dict(self) -> dict[str, Any]:
    return {
      "source": self.source,
      "duration_s": _r(self.duration_s, 2),
      "total_events": self.total_events,
      "op_engaged_events": self.op_engaged_events,
      "op_lead_events": self.op_lead_events,
      "op_nolead_events": self.op_nolead_events,
      "manual_events": self.manual_events,
      "op_lead_median_recovery_s": _r(self.op_lead_median_recovery_s, 3),
      "op_nolead_median_recovery_s": _r(self.op_nolead_median_recovery_s, 3),
      "op_lead_median_reaction_s": _r(self.op_lead_median_reaction_s, 3),
      "op_lead_median_planner_release_s": _r(self.op_lead_median_planner_release_s, 3),
      "op_lead_median_ego_start_s": _r(self.op_lead_median_ego_start_s, 3),
      "op_lead_median_accel": _r(self.op_lead_median_accel, 3),
      "op_nolead_median_accel": _r(self.op_nolead_median_accel, 3),
      "events": [event.to_dict() for event in self.events],
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


def _nearest(series: list[tuple[float, Any]], t: float, window_s: float) -> Any:
  """Value from (time, value) pairs nearest to t, or None if none within window_s."""
  best = None
  best_dt = window_s
  for st, value in series:
    dt = abs(st - t)
    if dt <= best_dt:
      best_dt = dt
      best = value
  return best


def _any_true_in_window(series: list[tuple[float, bool]], start_t: float, end_t: float) -> bool:
  """Return True if any sample in [start_t, end_t] is True."""
  for st, value in series:
    if st < start_t:
      continue
    if st > end_t:
      break
    if value:
      return True
  return False


def _collect(msgs: list[Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list, list, list, list, list]:
  cs_t, cs_v, cs_a = [], [], []
  radar: list[tuple[float, tuple[float, float, float, bool]]] = []
  enabled: list[tuple[float, bool]] = []
  long_active: list[tuple[float, bool]] = []
  driver_override: list[tuple[float, bool]] = []
  plan_sp: list[tuple[float, tuple[float, bool | None]]] = []
  for record in build_route_messages(msgs):
    typ, payload, t = record.typ, record.payload, record.t
    if typ == "carState":
      cs_t.append(t)
      cs_v.append(_f(safe_get(payload, "vEgo")))
      cs_a.append(_f(safe_get(payload, "aEgo")))
      driver_override.append(
        (t, bool(safe_get(payload, "gasPressed", False)) or bool(safe_get(payload, "brakePressed", False)))
      )
    elif typ == "carControl":
      long_active.append((t, bool(safe_get(payload, "longActive", False))))
    elif typ == "radarState":
      lead = safe_get(payload, "leadOne")
      if lead is not None:
        radar.append((t, (_f(safe_get(lead, "dRel")), _f(safe_get(lead, "vLead")), _f(safe_get(lead, "vRel")), bool(safe_get(lead, "status", False)))))
    elif typ == "selfdriveState":
      enabled.append((t, bool(safe_get(payload, "enabled", False))))
    elif typ == "longitudinalPlanSP":
      should_stop = safe_get(payload, "customLongitudinal.shouldStop")
      plan_sp.append((t, (_f(safe_get(payload, "aTarget")), bool(should_stop) if should_stop is not None else None)))
  return np.array(cs_t), np.array(cs_v), np.array(cs_a), radar, enabled, long_active, driver_override, plan_sp


def _make_event(t, v, a, min_idx, rec_idx, radar, enabled, long_active, driver_override, plan_sp, p) -> LaunchEvent:
  dip_t = float(t[min_idx])
  enabled_at_dip = bool(_nearest(enabled, dip_t, p.match_window_s) or False)
  long_active_at_dip = bool(_nearest(long_active, dip_t, p.match_window_s) or False)
  # Prefer carControl.longActive over selfdriveState.enabled when available.
  has_long_active = bool(long_active)
  driver_override_any = _any_true_in_window(driver_override, dip_t, float(t[rec_idx])) if driver_override else False
  op_engaged = (long_active_at_dip if has_long_active else enabled_at_dip) and not driver_override_any
  lead = _nearest(radar, dip_t, p.match_window_s)
  lead_present = bool(lead is not None and lead[3] and math.isfinite(lead[0]) and lead[0] < p.lead_d_rel_max)
  accel_win = a[min_idx : rec_idx + 1]
  accel_win = accel_win[np.isfinite(accel_win)]
  positive = accel_win[accel_win > 0]
  dt = np.diff(t[min_idx : rec_idx + 1])
  da = np.diff(a[min_idx : rec_idx + 1])
  jerk = np.abs(da / np.maximum(dt, 1e-3))
  jerk = jerk[np.isfinite(jerk)]
  lead_move_time = None
  lead_wait_time = None
  reaction_time = None
  planner_release_time = None
  planner_release_delay = None
  ego_start_time = None
  ego_start_delay = None
  prev_v_lead: float | None = None
  if lead_present:
    for st, (d_rel, v_lead, v_rel, status) in radar:
      if st <= dip_t or st > float(t[rec_idx]):
        continue
      if not status or not math.isfinite(v_lead) or not math.isfinite(d_rel):
        continue
      moved = v_lead >= 0.5 or (math.isfinite(v_rel) and v_rel >= 0.5) or (prev_v_lead is not None and v_lead - prev_v_lead >= 0.5)
      if moved:
        lead_move_time = float(st)
        lead_wait_time = float(lead_move_time - dip_t)
        break
      prev_v_lead = float(v_lead)
  if lead_move_time is not None:
    for st, v_ego, a_ego in zip(t[min_idx : rec_idx + 1], v[min_idx : rec_idx + 1], a[min_idx : rec_idx + 1], strict=True):
      if float(st) <= lead_move_time:
        continue
      if (math.isfinite(v_ego) and v_ego > float(v[min_idx]) + 0.1) or (math.isfinite(a_ego) and a_ego > 0.2):
        ego_start_time = float(st)
        ego_start_delay = float(st - lead_move_time)
        reaction_time = ego_start_delay
        break
    rec_t = float(t[rec_idx])
    for st, (a_target, should_stop) in plan_sp:
      if st <= lead_move_time or st > rec_t:
        continue
      has_positive_target = math.isfinite(a_target) and a_target > 0.05
      has_non_holding_target = math.isfinite(a_target) and a_target > -0.05
      released = has_positive_target or (should_stop is False and has_non_holding_target)
      if released:
        planner_release_time = float(st)
        planner_release_delay = float(st - lead_move_time)
        break
  return LaunchEvent(
    time=dip_t,
    min_speed=float(v[min_idx]),
    recovery_speed=float(v[rec_idx]),
    recovery_time=float(t[rec_idx] - t[min_idx]),
    lead_move_time=lead_move_time,
    lead_wait_time=lead_wait_time,
    reaction_time=reaction_time,
    planner_release_time=planner_release_time,
    planner_release_delay=planner_release_delay,
    ego_start_time=ego_start_time,
    ego_start_delay=ego_start_delay,
    op_engaged=op_engaged,
    lead_present=lead_present,
    lead_d_rel=(float(lead[0]) if lead is not None and math.isfinite(lead[0]) else None),
    accel_mean=float(np.mean(positive)) if positive.size else 0.0,
    accel_peak=float(np.max(accel_win)) if accel_win.size else 0.0,
    jerk_peak=float(np.max(jerk)) if jerk.size else 0.0,
    long_active=long_active_at_dip,
    driver_override=driver_override_any,
    enabled=enabled_at_dip,
  )


def analyze_route(msgs: list[Any], source: str = "unknown", params: LaunchParams | None = None) -> LaunchReport:
  p = params or LaunchParams()
  t, v, a, radar, enabled, long_active, driver_override, plan_sp = _collect(msgs)
  notes: list[str] = []
  if t.size == 0:
    notes.append("no carState samples found (need rlogs with carState)")
    return LaunchReport(source, 0.0, 0, 0, 0, 0, 0, None, None, None, None, None, None, None, [], notes)

  events: list[LaunchEvent] = []
  i, n = 0, len(v)
  while i < n:
    if not (math.isfinite(v[i]) and v[i] < p.speed_threshold):
      i += 1
      continue
    min_idx, min_v, j = i, v[i], i + 1
    while j < n and math.isfinite(v[j]) and v[j] < p.speed_threshold:
      if v[j] < min_v:
        min_v, min_idx = v[j], j
      j += 1
    rec_idx = None
    k = min_idx + 1
    while k < n and (t[k] - t[min_idx]) <= p.recovery_max_s:
      if math.isfinite(v[k]) and v[k] > min_v + p.recovery_delta:
        rec_idx = k
        break
      k += 1
    if rec_idx is not None and min_v < p.near_stop_speed:
      events.append(_make_event(t, v, a, min_idx, rec_idx, radar, enabled, long_active, driver_override, plan_sp, p))
    i = max(j, min_idx + 1)

  op = [e for e in events if e.op_engaged]
  op_lead = [e for e in op if e.lead_present]
  op_nolead = [e for e in op if not e.lead_present]
  return LaunchReport(
    source=source,
    duration_s=float(t[-1] - t[0]),
    total_events=len(events),
    op_engaged_events=len(op),
    op_lead_events=len(op_lead),
    op_nolead_events=len(op_nolead),
    manual_events=len(events) - len(op),
    op_lead_median_recovery_s=_opt_median([e.recovery_time for e in op_lead]),
    op_nolead_median_recovery_s=_opt_median([e.recovery_time for e in op_nolead]),
    op_lead_median_reaction_s=_opt_median([e.reaction_time for e in op_lead if e.reaction_time is not None]),
    op_lead_median_planner_release_s=_opt_median([e.planner_release_delay for e in op_lead if e.planner_release_delay is not None]),
    op_lead_median_ego_start_s=_opt_median([e.ego_start_delay for e in op_lead if e.ego_start_delay is not None]),
    op_lead_median_accel=_opt_median([e.accel_mean for e in op_lead]),
    op_nolead_median_accel=_opt_median([e.accel_mean for e in op_nolead]),
    events=sorted(events, key=lambda e: e.time),
    notes=notes,
  )


def _opt_median(values: list[float]) -> float | None:
  return float(np.median(values)) if values else None


def render_report(report: LaunchReport) -> str:
  lines = [
    f"Launch-delay profile: {report.source}",
    (
      f"  duration {report.duration_s:.1f}s, {report.total_events} near-stop launches "
      + f"({report.op_engaged_events} engaged: {report.op_lead_events} lead, {report.op_nolead_events} no-lead; "
      + f"{report.manual_events} manual)"
    ),
  ]
  if report.op_lead_events or report.op_nolead_events:
    lines += [
      "",
      "  engaged launch recovery (dip → +0.5 m/s):",
      (f"    median recovery:  lead {_fmt(report.op_lead_median_recovery_s, 2)} s   " + f"no-lead {_fmt(report.op_nolead_median_recovery_s, 2)} s"),
      f"    median reaction:  lead {_fmt(report.op_lead_median_reaction_s, 2)} s",
      f"    median planner release: lead {_fmt(report.op_lead_median_planner_release_s, 2)} s",
      f"    median ego start:       lead {_fmt(report.op_lead_median_ego_start_s, 2)} s",
      (f"    median accel:     lead {_fmt(report.op_lead_median_accel, 3)} m/s^2   " + f"no-lead {_fmt(report.op_nolead_median_accel, 3)} m/s^2"),
    ]
  for note in report.notes:
    lines.append(f"  note: {note}")
  return "\n".join(lines)


def _fmt(value: float | None, ndigits: int = 3) -> str:
  return "n/a" if value is None or not math.isfinite(float(value)) else f"{float(value):.{ndigits}f}"


def main() -> None:
  parser = argparse.ArgumentParser(description="Profile longitudinal launch behaviour after near-stops.")
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
