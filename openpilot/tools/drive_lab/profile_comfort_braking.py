#!/usr/bin/env python3
"""Comfort-braking opportunity profiler (offline / shadow analysis only).

Finds engaged closing-lead episodes where earlier mild decel might have reduced later
hard braking. This tool does not command, shadow, or recommend live control changes; it is
purely an offline diagnostic to answer "can we improve comfort braking?" safely.

Signals:
- carState.{vEgo,aEgo}
- carControl.longActive
- radarState.leadOne.{present,dRel,vRel}
- longitudinalPlanSP.aTarget

Run:
  uv run python -m openpilot.tools.drive_lab.profile_comfort_braking ROUTE
  uv run python -m openpilot.tools.drive_lab.profile_comfort_braking ROUTE --json --output /tmp/x.json
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from openpilot.tools.drive_lab.route_analysis import build_route_messages
from openpilot.tools.drive_lab.route_io import load_route_msgs, output_report
from openpilot.tools.drive_lab.timeline import safe_get


@dataclass(frozen=True)
class ComfortBrakingParams:
  v_ego_min: float = 3.0  # m/s; ignore very low-speed parking lots
  d_rel_min: float = 0.5  # m; require a real positive gap
  close_v_rel: float = -2.0  # m/s; start/maintain a closing episode
  high_close_v_rel: float = -3.0  # m/s; "strong closing" start for opportunity flag
  mild_a_target: float = -0.5  # m/s^2; first planned mild braking threshold
  firm_a_target: float = -1.5  # m/s^2; first planned firm braking threshold
  firm_a_ego: float = -1.5  # m/s^2; first measured firm braking threshold
  hard_a_ego: float = -2.0  # m/s^2; hard measured decel used in opportunity flag
  max_gap_s: float = 1.0  # s; merge closing samples across gaps up to this
  min_episode_s: float = 0.5  # s; drop trivially short episodes
  lead_model_prob_min: float = 0.0  # minimum lead modelProb to trust the lead
  candidate_model_prob_min: float = 0.8  # minimum lead confidence for review candidates
  candidate_d_rel_max: float = 120.0  # m; farthest lead distance for review candidates
  candidate_min_lead_time_s: float = 0.5  # s; minimum lead time before existing mild plan
  candidate_sustain_s: float = 0.25  # s; candidate condition must persist this long
  candidate_path_y_abs_max: float = 2.0  # m; optional path-confidence gate when lead yRel exists


@dataclass
class ComfortBrakingEpisode:
  start_s: float
  end_s: float
  duration_s: float
  start_v_ego: float
  start_d_rel: float
  start_v_rel: float
  min_d_rel: float
  peak_closing_v_rel: float
  worst_a_ego: float
  worst_a_target: float | None
  time_to_mild_plan_s: float | None
  time_to_firm_plan_s: float | None
  time_to_firm_a_ego_s: float | None
  opportunity: bool
  start_model_prob: float | None
  min_model_prob: float | None
  max_model_prob: float | None
  candidate_start_s: float | None
  candidate_start_delay_s: float | None
  candidate_start_d_rel: float | None
  candidate_start_v_rel: float | None
  candidate_start_ttc_s: float | None
  candidate_start_model_prob: float | None
  candidate_lead_time_to_mild_plan_s: float | None
  candidate_quality: str
  candidate_block_reasons: list[str]
  samples: int

  def to_dict(self) -> dict[str, Any]:
    return {
      "start_s": _r(self.start_s, 2),
      "end_s": _r(self.end_s, 2),
      "duration_s": _r(self.duration_s, 2),
      "start_v_ego_m_s": _r(self.start_v_ego, 2),
      "start_d_rel_m": _r(self.start_d_rel, 1),
      "start_v_rel_m_s": _r(self.start_v_rel, 2),
      "min_d_rel_m": _r(self.min_d_rel, 1),
      "peak_closing_v_rel_m_s": _r(self.peak_closing_v_rel, 2),
      "worst_a_ego_m_s2": _r(self.worst_a_ego, 3),
      "worst_a_target_m_s2": _r(self.worst_a_target, 3),
      "time_to_mild_plan_s": _r(self.time_to_mild_plan_s, 2),
      "time_to_firm_plan_s": _r(self.time_to_firm_plan_s, 2),
      "time_to_firm_a_ego_s": _r(self.time_to_firm_a_ego_s, 2),
      "opportunity": bool(self.opportunity),
      "start_model_prob": _r(self.start_model_prob, 3),
      "min_model_prob": _r(self.min_model_prob, 3),
      "max_model_prob": _r(self.max_model_prob, 3),
      "candidate_start_s": _r(self.candidate_start_s, 2),
      "candidate_start_delay_s": _r(self.candidate_start_delay_s, 2),
      "candidate_start_d_rel_m": _r(self.candidate_start_d_rel, 1),
      "candidate_start_v_rel_m_s": _r(self.candidate_start_v_rel, 2),
      "candidate_start_ttc_s": _r(self.candidate_start_ttc_s, 2),
      "candidate_start_model_prob": _r(self.candidate_start_model_prob, 3),
      "candidate_lead_time_to_mild_plan_s": _r(self.candidate_lead_time_to_mild_plan_s, 2),
      "candidate_quality": self.candidate_quality,
      "candidate_block_reasons": list(self.candidate_block_reasons),
      "samples": int(self.samples),
    }


@dataclass
class ComfortBrakingReport:
  source: str
  duration_s: float
  episode_count: int
  opportunity_count: int
  candidate_count: int
  path_confident_candidate_count: int
  closing_samples: int
  worst_a_ego_min: float | None
  median_worst_a_ego: float | None
  median_worst_a_target: float | None
  median_time_to_mild_plan_s: float | None
  median_firm_measured_decel: float | None
  median_firm_commanded_decel: float | None
  params: dict[str, float]
  notes: list[str]
  episodes: list[ComfortBrakingEpisode]

  def to_dict(self) -> dict[str, Any]:
    return {
      "source": self.source,
      "duration_s": _r(self.duration_s, 2),
      "episode_count": self.episode_count,
      "opportunity_count": self.opportunity_count,
      "candidate_count": self.candidate_count,
      "path_confident_candidate_count": self.path_confident_candidate_count,
      "closing_samples": self.closing_samples,
      "worst_a_ego_min_m_s2": _r(self.worst_a_ego_min, 3),
      "median_worst_a_ego_m_s2": _r(self.median_worst_a_ego, 3),
      "median_worst_a_target_m_s2": _r(self.median_worst_a_target, 3),
      "median_time_to_mild_plan_s": _r(self.median_time_to_mild_plan_s, 2),
      "median_firm_measured_decel_m_s2": _r(self.median_firm_measured_decel, 3),
      "median_firm_commanded_decel_m_s2": _r(self.median_firm_commanded_decel, 3),
      "params": {k: _r(v, 3) for k, v in self.params.items()},
      "notes": list(self.notes),
      "episodes": [e.to_dict() for e in self.episodes],
    }


@dataclass(frozen=True)
class _ClosingSample:
  t: float
  v_ego: float
  a_ego: float
  d_rel: float
  v_rel: float
  a_target: float | None
  model_prob: float | None
  y_rel: float | None


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


def _median(values: Sequence[float | None]) -> float | None:
  finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
  return float(np.percentile(finite, 50)) if finite else None


def _collect(msgs: list[Any], p: ComfortBrakingParams) -> list[_ClosingSample]:
  """Engaged closing-lead samples (zero-order-hold on carState, carControl, longitudinalPlanSP)."""
  latest: dict[str, Any] = {}
  out: list[_ClosingSample] = []
  for record in build_route_messages(msgs):
    typ, payload = record.typ, record.payload
    if typ in ("carState", "carControl", "longitudinalPlanSP"):
      latest[typ] = payload
      continue
    if typ != "radarState":
      continue
    cs = latest.get("carState")
    cc = latest.get("carControl")
    sp = latest.get("longitudinalPlanSP")
    if cs is None or cc is None or not bool(safe_get(cc, "longActive", False)):
      continue
    v_ego = _f(safe_get(cs, "vEgo"))
    if not (math.isfinite(v_ego) and v_ego >= p.v_ego_min):
      continue
    lead = safe_get(payload, "leadOne")
    if lead is None or not bool(safe_get(lead, "present", False)):
      continue
    if _f(safe_get(lead, "modelProb")) < p.lead_model_prob_min:
      continue
    d_rel = _f(safe_get(lead, "dRel"))
    v_rel = _f(safe_get(lead, "vRel"))
    model_prob = _f(safe_get(lead, "modelProb"))
    y_rel = _f(safe_get(lead, "yRel"))
    if not (math.isfinite(d_rel) and d_rel > p.d_rel_min):
      continue
    if not (math.isfinite(v_rel) and v_rel <= p.close_v_rel):
      continue
    a_ego = _f(safe_get(cs, "aEgo"))
    a_target = None
    if sp is not None:
      a_target = _f(safe_get(sp, "aTarget"))
      if not math.isfinite(a_target):
        a_target = None
    out.append(
      _ClosingSample(
        record.t,
        v_ego,
        a_ego,
        d_rel,
        v_rel,
        a_target,
        model_prob if math.isfinite(model_prob) else None,
        y_rel if math.isfinite(y_rel) else None,
      )
    )
  return out


def _cluster_episodes(samples: list[_ClosingSample], p: ComfortBrakingParams) -> list[list[_ClosingSample]]:
  clusters: list[list[_ClosingSample]] = []
  current: list[_ClosingSample] = []
  for s in samples:
    if not current or s.t - current[-1].t <= p.max_gap_s:
      current.append(s)
    else:
      if current and current[-1].t - current[0].t >= p.min_episode_s:
        clusters.append(current)
      current = [s]
  if current and current[-1].t - current[0].t >= p.min_episode_s:
    clusters.append(current)
  return clusters


def _analyze_episode(cluster: list[_ClosingSample], p: ComfortBrakingParams) -> ComfortBrakingEpisode:
  start = cluster[0]
  end = cluster[-1]

  min_d_rel = min((s.d_rel for s in cluster if math.isfinite(s.d_rel)), default=math.nan)
  peak_v_rel = min((s.v_rel for s in cluster if math.isfinite(s.v_rel)), default=math.nan)
  worst_a_ego = min((s.a_ego for s in cluster if math.isfinite(s.a_ego)), default=math.nan)

  a_targets = [s.a_target for s in cluster if s.a_target is not None]
  worst_a_target = min(a_targets) if a_targets else None
  model_probs = [s.model_prob for s in cluster if s.model_prob is not None]

  def _first_time(predicate: Any) -> float | None:
    match = next((s for s in cluster if predicate(s)), None)
    return None if match is None else max(0.0, match.t - start.t)

  time_to_mild = _first_time(lambda s: s.a_target is not None and s.a_target <= p.mild_a_target)
  time_to_firm = _first_time(lambda s: s.a_target is not None and s.a_target <= p.firm_a_target)
  time_to_firm_ae = _first_time(lambda s: math.isfinite(s.a_ego) and s.a_ego <= p.firm_a_ego)

  has_later_firm = time_to_firm is not None or (time_to_firm_ae is not None and (time_to_mild is None or time_to_firm_ae > 0.0))
  has_hard_ae = any(math.isfinite(s.a_ego) and s.a_ego <= p.hard_a_ego for s in cluster)

  opportunity = (
    math.isfinite(start.v_rel)
    and start.v_rel <= p.high_close_v_rel
    and time_to_mild is not None
    and time_to_mild > 0.75
    and has_later_firm
    and (time_to_firm is not None or has_hard_ae)
  )

  candidate_sample, candidate_block_reasons = _find_candidate_sample(cluster, time_to_mild, opportunity, p)
  candidate_start_delay_s = None if candidate_sample is None else max(0.0, candidate_sample.t - start.t)
  candidate_lead_time = None
  if candidate_start_delay_s is not None and time_to_mild is not None:
    candidate_lead_time = max(0.0, time_to_mild - candidate_start_delay_s)
  candidate_quality = "none"
  if candidate_sample is not None and candidate_lead_time is not None and candidate_lead_time >= p.candidate_min_lead_time_s:
    if candidate_sample.y_rel is not None and abs(candidate_sample.y_rel) <= p.candidate_path_y_abs_max:
      candidate_quality = "path_confident"
    else:
      candidate_quality = "kinematic"
      if candidate_sample.y_rel is None:
        candidate_block_reasons.append("path_confidence_unknown")
      else:
        candidate_block_reasons.append("not_path_confident")
  elif candidate_sample is not None:
    candidate_block_reasons.append("lead_time_below_min")

  return ComfortBrakingEpisode(
    start_s=start.t,
    end_s=end.t,
    duration_s=end.t - start.t,
    start_v_ego=start.v_ego,
    start_d_rel=start.d_rel,
    start_v_rel=start.v_rel,
    min_d_rel=min_d_rel,
    peak_closing_v_rel=peak_v_rel,
    worst_a_ego=worst_a_ego,
    worst_a_target=worst_a_target,
    time_to_mild_plan_s=time_to_mild,
    time_to_firm_plan_s=time_to_firm,
    time_to_firm_a_ego_s=time_to_firm_ae,
    opportunity=opportunity,
    start_model_prob=start.model_prob,
    min_model_prob=min(model_probs) if model_probs else None,
    max_model_prob=max(model_probs) if model_probs else None,
    candidate_start_s=None if candidate_sample is None else candidate_sample.t,
    candidate_start_delay_s=candidate_start_delay_s,
    candidate_start_d_rel=None if candidate_sample is None else candidate_sample.d_rel,
    candidate_start_v_rel=None if candidate_sample is None else candidate_sample.v_rel,
    candidate_start_ttc_s=_ttc(candidate_sample) if candidate_sample is not None else None,
    candidate_start_model_prob=None if candidate_sample is None else candidate_sample.model_prob,
    candidate_lead_time_to_mild_plan_s=candidate_lead_time,
    candidate_quality=candidate_quality,
    candidate_block_reasons=sorted(set(candidate_block_reasons)),
    samples=len(cluster),
  )


def _ttc(sample: _ClosingSample) -> float | None:
  if not (math.isfinite(sample.d_rel) and math.isfinite(sample.v_rel) and sample.v_rel < -1e-3):
    return None
  return sample.d_rel / -sample.v_rel


def _find_candidate_sample(
  cluster: list[_ClosingSample],
  time_to_mild: float | None,
  opportunity: bool,
  p: ComfortBrakingParams,
) -> tuple[_ClosingSample | None, list[str]]:
  if not opportunity:
    return None, ["not_opportunity"]
  if time_to_mild is None:
    return None, ["no_existing_mild_plan"]
  start = cluster[0]

  def eligible(sample: _ClosingSample) -> bool:
    if sample.t - start.t >= time_to_mild:
      return False
    if sample.a_target is None:
      return False
    return (
      sample.a_target > p.mild_a_target
      and math.isfinite(sample.v_rel)
      and sample.v_rel <= p.high_close_v_rel
      and sample.model_prob is not None
      and sample.model_prob >= p.candidate_model_prob_min
      and math.isfinite(sample.d_rel)
      and sample.d_rel <= p.candidate_d_rel_max
    )

  reasons: list[str] = []
  saw_plan_missing = any(s.a_target is None for s in cluster if s.t - start.t < time_to_mild)
  saw_model_prob = any(s.model_prob is not None and s.model_prob >= p.candidate_model_prob_min for s in cluster)
  saw_distance = any(math.isfinite(s.d_rel) and s.d_rel <= p.candidate_d_rel_max for s in cluster)
  for idx, sample in enumerate(cluster):
    if not eligible(sample):
      continue
    end_t = sample.t + p.candidate_sustain_s
    sustained = [s for s in cluster[idx:] if sample.t <= s.t <= end_t]
    if sustained and sustained[-1].t - sample.t >= p.candidate_sustain_s and all(eligible(s) for s in sustained):
      return sample, []
  if saw_plan_missing:
    reasons.append("plan_missing")
  if not saw_model_prob:
    reasons.append("model_prob_below_min")
  if not saw_distance:
    reasons.append("distance_over_max")
  reasons.append("no_sustained_candidate")
  return None, reasons


def analyze_route(
  msgs: list[Any],
  source: str = "unknown",
  params: ComfortBrakingParams | None = None,
) -> ComfortBrakingReport:
  p = params or ComfortBrakingParams()
  records = build_route_messages(msgs)
  duration = (records[-1].t - records[0].t) if records else 0.0
  samples = _collect(msgs, p)
  clusters = _cluster_episodes(samples, p)
  episodes = [_analyze_episode(c, p) for c in clusters]

  notes: list[str] = []
  if not samples:
    notes.append(f"no engaged closing-lead samples above {p.v_ego_min} m/s with vRel <= {p.close_v_rel} m/s")
  elif not episodes:
    notes.append(f"found {len(samples)} closing samples but no episodes longer than {p.min_episode_s} s")

  worst_a_ego_list = [e.worst_a_ego for e in episodes if math.isfinite(e.worst_a_ego)]
  worst_a_target_list = [e.worst_a_target for e in episodes if e.worst_a_target is not None]
  firm_ae_list = [e.worst_a_ego for e in episodes if e.time_to_firm_a_ego_s is not None]
  firm_target_list = [e.worst_a_target for e in episodes if e.time_to_firm_plan_s is not None]
  opportunity_times = [e.time_to_mild_plan_s for e in episodes if e.opportunity and e.time_to_mild_plan_s is not None]

  return ComfortBrakingReport(
    source=source,
    duration_s=duration,
    episode_count=len(episodes),
    opportunity_count=sum(1 for e in episodes if e.opportunity),
    candidate_count=sum(1 for e in episodes if e.candidate_quality != "none"),
    path_confident_candidate_count=sum(1 for e in episodes if e.candidate_quality == "path_confident"),
    closing_samples=len(samples),
    worst_a_ego_min=min(worst_a_ego_list) if worst_a_ego_list else None,
    median_worst_a_ego=_median(worst_a_ego_list),
    median_worst_a_target=_median(worst_a_target_list),
    median_time_to_mild_plan_s=_median(opportunity_times),
    median_firm_measured_decel=_median(firm_ae_list),
    median_firm_commanded_decel=_median(firm_target_list),
    params={
      "v_ego_min": p.v_ego_min,
      "close_v_rel": p.close_v_rel,
      "high_close_v_rel": p.high_close_v_rel,
      "mild_a_target": p.mild_a_target,
      "firm_a_target": p.firm_a_target,
      "firm_a_ego": p.firm_a_ego,
      "max_gap_s": p.max_gap_s,
      "min_episode_s": p.min_episode_s,
      "candidate_model_prob_min": p.candidate_model_prob_min,
      "candidate_d_rel_max": p.candidate_d_rel_max,
      "candidate_min_lead_time_s": p.candidate_min_lead_time_s,
      "candidate_sustain_s": p.candidate_sustain_s,
      "candidate_path_y_abs_max": p.candidate_path_y_abs_max,
    },
    notes=notes,
    episodes=episodes,
  )


def render_report(report: ComfortBrakingReport) -> str:
  lines = [
    f"Comfort-braking profile: {report.source}",
    "",
    "NOTE: offline shadow analysis only. This report does not command, shadow, or recommend",
    "live control changes. Use it to identify routes/episodes worth deeper review before",
    "changing longitudinal policy.",
    "Candidate means the offline heuristic found sustained closing-lead evidence before",
    "the existing planner reached mild decel. It is not proof earlier braking was safe.",
    "",
    f"  route duration:       {report.duration_s / 60:.1f} min",
    f"  closing samples:      {report.closing_samples}",
    f"  episodes:             {report.episode_count}",
    f"  opportunities:        {report.opportunity_count}",
    f"  candidates:           {report.candidate_count} ({report.path_confident_candidate_count} path-confident)",
  ]

  if report.episode_count:
    lines += [
      "",
      f"  worst aEgo:           {_fmt(report.worst_a_ego_min, 3)} m/s^2",
      f"  median worst aEgo:    {_fmt(report.median_worst_a_ego, 3)} m/s^2",
      f"  median worst aTarget: {_fmt(report.median_worst_a_target, 3)} m/s^2",
      f"  median firm aEgo*:    {_fmt(report.median_firm_measured_decel, 3)} m/s^2  " + "(*episodes where aEgo crossed firm threshold)",
      f"  median firm aTarget*: {_fmt(report.median_firm_commanded_decel, 3)} m/s^2  " + "(*episodes where aTarget crossed firm threshold)",
    ]
  if report.opportunity_count:
    lines.append(f"  median time to mild plan for opportunities: {_fmt(report.median_time_to_mild_plan_s, 2)} s")

  if report.episodes:
    lines.append("")
    lines.append(f"  episodes ({len(report.episodes)}):")
    for i, e in enumerate(report.episodes, start=1):
      marker = " [OPPORTUNITY]" if e.opportunity else ""
      candidate = ""
      if e.candidate_quality != "none":
        candidate = f"  candidate={e.candidate_quality}@{_fmt(e.candidate_start_delay_s, 2)}s"
      elif e.candidate_block_reasons:
        candidate = f"  candidate_block={','.join(e.candidate_block_reasons)}"
      lines.append(
        f"    #{i:02d}  {e.start_s:.1f}s-{e.end_s:.1f}s  dur={e.duration_s:.2f}s  "
        + f"vRel0={e.start_v_rel:.1f}  dRel0={e.start_d_rel:.1f}m  "
        + f"aWorst={e.worst_a_ego:.2f}  mild@t={_fmt(e.time_to_mild_plan_s, 2)}s{candidate}{marker}"
      )

  for note in report.notes:
    lines.append(f"  note: {note}")
  return "\n".join(lines)


def _fmt(value: float | None, ndigits: int = 2) -> str:
  return "n/a" if value is None or not math.isfinite(float(value)) else f"{float(value):.{ndigits}f}"


def main() -> None:
  parser = argparse.ArgumentParser(description="Find comfort-braking opportunities in an engaged route log (offline analysis).")
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
