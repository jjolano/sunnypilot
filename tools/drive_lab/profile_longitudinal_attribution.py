#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from openpilot.tools.drive_lab.route_analysis import build_route_messages
from openpilot.tools.drive_lab.route_io import load_route_msgs
from openpilot.tools.drive_lab.timeline import safe_get


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


def _current_state(latest: dict[str, Any], rec: Any) -> dict[str, Any]:
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
    "carControl": {"longActive": bool(safe_get(cc, "longActive", False))},
  }


def _churn(intents: list[str], duration_s: float) -> dict[str, Any]:
  """Intent transition rate. Occupancy counts hide churn: an owner that holds for 60 s and
  one that flips 60 times both show up as 'present'. Route 2dc ran 94 changes/min with each
  handoff worth 0.4-0.6 m/s^2 of commanded step, which is what the jerk p95 was made of."""
  seq = [v for v in intents if v]
  transitions: dict[str, int] = {}
  for prev, cur in zip(seq, seq[1:], strict=False):
    if prev != cur:
      transitions[f"{prev} -> {cur}"] = transitions.get(f"{prev} -> {cur}", 0) + 1
  changes = sum(transitions.values())
  top = dict(sorted(transitions.items(), key=lambda kv: -kv[1])[:6])
  return {
    "changes": changes,
    "changes_per_min": _round(changes / (duration_s / 60.0), 1) if duration_s > 0 else None,
    "top_transitions": top,
  }


def analyze_route(msgs: list[Any], source: str = "unknown") -> dict[str, Any]:
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
  decel_samples: list[dict[str, Any]] = []
  for rec in records:
    latest[rec.typ] = rec.payload
    if rec.typ == "selfdriveState":
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
          decel_samples.append(_current_state(latest, rec))
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
    "custom_intent_churn": _churn(custom_intents, duration_s),
    "custom_reason_counts": _counts([v for v in custom_reasons if v]),
    "a_target": {"longitudinalPlan": _stats(plan_a), "longitudinalPlanSP": _stats(sp_a)},
    "strong_decel_episodes": episodes,
  }
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
  churn = report.get("custom_intent_churn", {})
  lines.append(f"  intent churn: {churn.get('changes_per_min')}/min ({churn.get('changes')} changes) top: {churn.get('top_transitions', {})}")
  lines.append(f"  custom reasons: {report.get('custom_reason_counts', {})}")
  lines.append(f"  aTarget: {report.get('a_target', {})}")
  for idx, ep in enumerate(report.get("strong_decel_episodes", []), start=1):
    lines.append(f"  episode {idx}: {ep['start_s']}s->{ep['end_s']}s ({ep['sample_count']} samples)")
  return "\n".join(lines)


def main() -> None:
  p = argparse.ArgumentParser(description="Read-only longitudinal attribution report for route logs.")
  p.add_argument("route")
  p.add_argument("--qlog", action="store_true")
  p.add_argument("--json", action="store_true")
  p.add_argument("--output")
  args = p.parse_args()
  msgs = load_route_msgs(args.route, qlog=args.qlog)
  report = analyze_route(msgs, source=args.route)
  rendered = json.dumps(report, indent=2, sort_keys=True) if args.json else render_report(report)
  if args.output:
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  print(rendered)


if __name__ == "__main__":
  main()
