#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from openpilot.sunnypilot.custom.longitudinal.lead_anticipation import LeadAnticipation
from openpilot.tools.drive_lab.replay_lead_anticipation import analyze_route as analyze_lead_replay
from openpilot.tools.drive_lab.route_analysis import build_route_messages
from openpilot.tools.drive_lab.route_io import load_route_msgs
from openpilot.tools.drive_lab.timeline import safe_get


class _AlwaysOnLeadAnticipationParams:
  """Force §3 shaping on for offline attribution without depending on device params."""

  def get_bool(self, key: str) -> bool:
    return key == "CustomLongitudinalEnabled"

  def get(self, key: str, default: Any = None, return_default: bool = False) -> Any:
    if key == "LeadAnticipationMode":
      return "apply"
    return default


def _finite(v: Any) -> float | None:
  try:
    f = float(v)
  except (TypeError, ValueError):
    return None
  return f if math.isfinite(f) else None


def _round(v: Any, ndigits: int = 3) -> float | None:
  f = _finite(v)
  return None if f is None else round(f, ndigits)


def _counts(values: list[str]) -> dict[str, int]:
  return dict(sorted(Counter(values).items()))


def _percentile(values: list[float], p: float) -> float | None:
  clean = sorted(v for v in values if math.isfinite(v))
  if not clean:
    return None
  if len(clean) == 1:
    return clean[0]
  idx = (len(clean) - 1) * p / 100.0
  lo = math.floor(idx)
  hi = math.ceil(idx)
  if lo == hi:
    return clean[int(idx)]
  return clean[lo] * (hi - idx) + clean[hi] * (idx - lo)


def _stats(values: list[float]) -> dict[str, float | None]:
  return {
    "min": _round(min(values)) if values else None,
    "p05": _round(_percentile(values, 5.0)),
    "median": _round(_percentile(values, 50.0)),
    "p95": _round(_percentile(values, 95.0)),
  }


def _flag_duration(samples: list[tuple[float, bool]]) -> float:
  if len(samples) < 2:
    return 0.0
  samples = sorted(samples)
  total = 0.0
  for (t, active), (t_next, _) in zip(samples, samples[1:]):
    dt = max(0.0, min(2.0, t_next - t))
    if active:
      total += dt
  return total


def _sample_value(msg: Any, path: str) -> Any:
  return safe_get(msg, path)


def _fmt_msg_value(msg: Any, path: str) -> float | None:
  return _round(_sample_value(msg, path))


def _lead_anticipation_delta(raw_radar: Any, shaped_radar: Any) -> dict[str, Any]:
  raw = _finite(safe_get(raw_radar, "leadOne.aLeadK"))
  shaped = _finite(safe_get(shaped_radar, "leadOne.aLeadK"))
  delta = None if raw is None or shaped is None else shaped - raw
  return {
    "leadOneRawALeadK": _round(raw),
    "leadOneShapedALeadK": _round(shaped),
    "leadOneDelta": _round(delta),
    "softened": bool(delta is not None and delta > 0.02),
  }


def _radar_dt(t: float, last_t: float | None) -> float:
  return 0.05 if last_t is None else max(0.0, min(0.5, t - last_t))


def _build_lead_anticipation_summary(msgs: list[Any]) -> dict[str, Any]:
  la = LeadAnticipation(_AlwaysOnLeadAnticipationParams())
  radar_samples = 0
  braking_samples = 0
  softened_samples = 0
  deltas: list[float] = []
  softened_deltas: list[float] = []
  last_radar_t: float | None = None
  for rec in build_route_messages(msgs):
    if rec.typ != "radarState":
      continue
    radar = rec.payload
    shaped = la.shape(radar, _radar_dt(rec.t, last_radar_t))
    last_radar_t = rec.t
    lead = safe_get(radar, "leadOne")
    if lead is None or not bool(safe_get(lead, "status", False)):
      continue
    radar_samples += 1
    raw = _finite(safe_get(lead, "aLeadK"))
    shp = _finite(safe_get(shaped, "leadOne.aLeadK"))
    if raw is None or shp is None:
      continue
    delta = shp - raw
    deltas.append(delta)
    if raw < -0.1:
      braking_samples += 1
    if delta > 0.02:
      softened_samples += 1
      softened_deltas.append(delta)
  if not deltas:
    return {"radar_samples": radar_samples, "lead_braking_samples": braking_samples, "softened_samples": softened_samples, "note": "no lead-anticipation deltas available"}
  return {
    "radar_samples": radar_samples,
    "lead_braking_samples": braking_samples,
    "softened_samples": softened_samples,
    "max_delta": _round(max(deltas)),
    "median_delta": _round(_percentile(deltas, 50.0)),
    "median_soften_delta": _round(_percentile(softened_deltas, 50.0)),
  }


def _current_state(latest: dict[str, Any], rec: Any, lead_anticipation: dict[str, Any] | None) -> dict[str, Any]:
  cs = latest.get("carState")
  cc = latest.get("carControl")
  lp = latest.get("longitudinalPlan")
  sp = latest.get("longitudinalPlanSP")
  lead = safe_get(latest.get("radarState"), "leadOne")
  return {
    "time_s": _round(rec.t, 3),
    "vEgo": _fmt_msg_value(cs, "vEgo"),
    "aEgo": _fmt_msg_value(cs, "aEgo"),
    "dRel": _fmt_msg_value(lead, "dRel"),
    "vRel": _fmt_msg_value(lead, "vRel"),
    "vLead": _round(safe_get(lead, "vLeadK", safe_get(lead, "vLead"))),
    "aLeadK": _fmt_msg_value(lead, "aLeadK"),
    "THW": (_round(float(safe_get(lead, "dRel")) / float(safe_get(cs, "vEgo")), 3) if lead and _finite(safe_get(lead, "dRel")) is not None and (v_ego := _finite(safe_get(cs, "vEgo"))) is not None and v_ego > 0.1 else None),
    "longitudinalPlan": {
      "source": str(safe_get(lp, "longitudinalPlanSource", "unknown")),
      "aTarget": _fmt_msg_value(lp, "aTarget"),
    },
    "longitudinalPlanSP": {
      "source": str(safe_get(sp, "longitudinalPlanSource", "unknown")),
      "aTarget": _fmt_msg_value(sp, "aTarget"),
      "customLongitudinal": {
        "enabled": bool(safe_get(sp, "customLongitudinal.enabled", False)),
        "active": bool(safe_get(sp, "customLongitudinal.active", False)),
        "selectedIntent": str(safe_get(sp, "customLongitudinal.selectedIntent", "") or ""),
        "reason": str(safe_get(sp, "customLongitudinal.reason", "") or ""),
        "shouldStop": bool(safe_get(sp, "customLongitudinal.shouldStop", False)),
      },
    },
    "leadAnticipation": lead_anticipation or {},
    "carControl": {"longActive": bool(safe_get(cc, "longActive", False))},
  }


def analyze_route(msgs: list[Any], source: str = "unknown", include_replay: bool = True) -> dict[str, Any]:
  records = build_route_messages(msgs)
  duration_s = records[-1].t - records[0].t if records else 0.0
  plan_sources: list[str] = []
  sp_sources: list[str] = []
  custom_intents: list[str] = []
  custom_reasons: list[str] = []
  plan_a: list[float] = []
  sp_a: list[float] = []
  enabled_samples: list[tuple[float, bool]] = []
  long_active_samples: list[tuple[float, bool]] = []
  custom_active_samples: list[tuple[float, bool]] = []
  latest: dict[str, Any] = {}
  lead_anticipation = LeadAnticipation(_AlwaysOnLeadAnticipationParams())
  latest_lead_anticipation: dict[str, Any] | None = None
  last_radar_t: float | None = None
  decel_samples: list[dict[str, Any]] = []
  for rec in records:
    latest[rec.typ] = rec.payload
    if rec.typ == "radarState":
      shaped = lead_anticipation.shape(rec.payload, _radar_dt(rec.t, last_radar_t))
      last_radar_t = rec.t
      latest_lead_anticipation = _lead_anticipation_delta(rec.payload, shaped)
    elif rec.typ == "selfdriveState":
      enabled_samples.append((rec.t, bool(safe_get(rec.payload, "enabled", False))))
    elif rec.typ == "carControl":
      long_active_samples.append((rec.t, bool(safe_get(rec.payload, "longActive", False))))
    if rec.typ == "carState":
      cs = rec.payload
      cc = latest.get("carControl")
      lead = safe_get(latest.get("radarState"), "leadOne")
      if bool(safe_get(cc, "longActive", False)) and lead and bool(safe_get(lead, "status", False)):
        a_ego = _finite(safe_get(cs, "aEgo"))
        if a_ego is not None and a_ego < -0.5:
          decel_samples.append(_current_state(latest, rec, latest_lead_anticipation))
    if rec.typ == "longitudinalPlan":
      source_name = str(safe_get(rec.payload, "longitudinalPlanSource", "unknown"))
      plan_sources.append(source_name)
      if (a := _finite(safe_get(rec.payload, "aTarget"))) is not None:
        plan_a.append(a)
    elif rec.typ == "longitudinalPlanSP":
      source_name = str(safe_get(rec.payload, "longitudinalPlanSource", "unknown"))
      sp_sources.append(source_name)
      if (a := _finite(safe_get(rec.payload, "aTarget"))) is not None:
        sp_a.append(a)
      custom_intents.append(str(safe_get(rec.payload, "customLongitudinal.selectedIntent", "") or ""))
      custom_reasons.append(str(safe_get(rec.payload, "customLongitudinal.reason", "") or ""))
      custom_active_samples.append((rec.t, bool(safe_get(rec.payload, "customLongitudinal.active", False))))

  episodes: list[dict[str, Any]] = []
  if decel_samples:
    current = [decel_samples[0]]
    for sample in decel_samples[1:]:
      if float(sample["time_s"]) - float(current[-1]["time_s"]) > 2.0:
        episodes.append(_episode(current))
        current = [sample]
      else:
        current.append(sample)
    episodes.append(_episode(current))

  report = {
    "source": source,
    "duration_s": _round(duration_s, 3),
    "activity": {
      "enabled_s": _round(_flag_duration(enabled_samples), 3),
      "longActive_s": _round(_flag_duration(long_active_samples), 3),
      "customActive_s": _round(_flag_duration(custom_active_samples), 3),
    },
    "plan_source_counts": _counts(plan_sources),
    "sp_plan_source_counts": _counts(sp_sources),
    "custom_intent_counts": _counts([v for v in custom_intents if v]),
    "custom_reason_counts": _counts([v for v in custom_reasons if v]),
    "a_target": {"longitudinalPlan": _stats(plan_a), "longitudinalPlanSP": _stats(sp_a)},
    "lead_anticipation": _build_lead_anticipation_summary(msgs),
    "strong_decel_episodes": episodes,
  }
  if include_replay:
    report["mpc_replay"] = analyze_lead_replay(msgs, source=source)
  return report


def _episode(samples: list[dict[str, Any]]) -> dict[str, Any]:
  worst = min(samples, key=lambda s: float(s.get("aEgo") or 0.0))
  return {
    "start_s": samples[0]["time_s"],
    "end_s": samples[-1]["time_s"],
    "duration_s": _round(float(samples[-1]["time_s"]) - float(samples[0]["time_s"]), 3),
    "sample_count": len(samples),
    "worst_sample": worst,
  }


def render_report(report: dict[str, Any]) -> str:
  lines = [f"Longitudinal attribution: {report['source']}", f"  duration {report.get('duration_s', 0):.1f}s"]
  act = report.get("activity", {})
  lines.append(f"  activity: enabled {act.get('enabled_s')}s, longActive {act.get('longActive_s')}s, custom {act.get('customActive_s')}s")
  lines.append(f"  plan sources: {report.get('plan_source_counts', {})}")
  lines.append(f"  SP sources: {report.get('sp_plan_source_counts', {})}")
  lines.append(f"  custom intents: {report.get('custom_intent_counts', {})}")
  lines.append(f"  custom reasons: {report.get('custom_reason_counts', {})}")
  lines.append(f"  aTarget: {report.get('a_target', {})}")
  lines.append(f"  lead anticipation: {report.get('lead_anticipation', {})}")
  if report.get("mpc_replay") is not None:
    lines.append(f"  mpc replay: {report['mpc_replay']}")
  for idx, ep in enumerate(report.get("strong_decel_episodes", []), start=1):
    lines.append(f"  episode {idx}: {ep['start_s']}s->{ep['end_s']}s ({ep['sample_count']} samples)")
  return "\n".join(lines)


def main() -> None:
  p = argparse.ArgumentParser(description="Read-only longitudinal attribution report for route logs.")
  p.add_argument("route")
  p.add_argument("--qlog", action="store_true")
  p.add_argument("--json", action="store_true")
  p.add_argument("--output")
  p.add_argument("--no-replay", action="store_true")
  args = p.parse_args()
  msgs = load_route_msgs(args.route, qlog=args.qlog)
  report = analyze_route(msgs, source=args.route, include_replay=not args.no_replay)
  rendered = json.dumps(report, indent=2, sort_keys=True) if args.json else render_report(report)
  if args.output:
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  print(rendered)


if __name__ == "__main__":
  main()
