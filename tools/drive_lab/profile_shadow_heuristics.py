#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openpilot.tools.drive_lab.route_analysis import build_route_messages
from openpilot.tools.drive_lab.route_io import load_route_msgs
from openpilot.tools.drive_lab.timeline import safe_get


CONTROL_INVARIANT_PATHS = (
  "controlsState.desiredCurvature",
  "carControl.actuators.accel",
  "carControl.actuators.steer",
  "carControl.actuators.steeringAngleDeg",
  "longitudinalPlan.aTarget",
)


@dataclass(frozen=True)
class ShadowHeuristicsReport:
  source: str
  lateral: dict[str, Any]
  grade: dict[str, Any]
  invariants: dict[str, Any]

  def to_dict(self) -> dict[str, Any]:
    return {"source": self.source, "lateral": self.lateral, "grade": self.grade, "invariants": self.invariants}


def build_shadow_heuristics_report(msgs: list[Any], *, source: str = "unknown", baseline_msgs: list[Any] | None = None) -> ShadowHeuristicsReport:
  records = build_route_messages(msgs)
  lateral = _lateral_summary(records)
  grade = _grade_summary(records)
  invariants = {"baselineCompared": baseline_msgs is not None}
  if baseline_msgs is not None:
    invariants.update(_compare_control_invariants(build_route_messages(baseline_msgs), records))
  return ShadowHeuristicsReport(source=source, lateral=lateral, grade=grade, invariants=invariants)


def render_shadow_heuristics_report(report: ShadowHeuristicsReport) -> str:
  data = report.to_dict()
  lat = data["lateral"]
  grade = data["grade"]
  inv = data["invariants"]
  lines = [f"Shadow heuristics: {data['source']}"]
  lines.append(f"  lateral samples={lat['samples']} available={lat['availableSamples']} suppress={lat['suppressCandidates']}")
  lines.append(f"  lateral response={lat['responseClassCounts']} disagreement={lat['disagreementCounts']}")
  lines.append(f"  lateral deltas={lat['latAccelDeltaStats']}")
  lines.append(f"  grade samples={grade['samples']} scenarios={grade['scenarioCounts']} grades={grade['roadGradeCounts']}")
  lines.append(f"  grade confidence={grade['gradeConfidenceStats']} proposed={grade['proposedCompensationStats']}")
  if inv.get("baselineCompared"):
    lines.append(f"  invariants pass={inv['pass']} diffs={inv['diffCounts']}")
  else:
    lines.append("  invariants: provide --baseline to compare actuation/target outputs")
  return "\n".join(lines)


def _lateral_summary(records: list[Any]) -> dict[str, Any]:
  available = 0
  suppress = 0
  disagreement: list[str] = []
  response: list[str] = []
  block_reasons: list[str] = []
  model_measured: list[float] = []
  model_yaw: list[float] = []
  steering_yaw: list[float] = []
  model_yaw_signed: list[float] = []
  steering_yaw_signed: list[float] = []
  dtle: list[float] = []

  for rec in records:
    if rec.typ != "controlsState":
      continue
    mps = safe_get(rec.payload, "modelPathState")
    if mps is None:
      continue
    if bool(safe_get(mps, "sensorConfidenceAvailable", False)):
      available += 1
    if bool(safe_get(mps, "sensorSuppressCandidate", False)):
      suppress += 1
    disagreement.append(str(safe_get(mps, "sensorDisagreementLevel", "") or ""))
    response.append(str(safe_get(mps, "sensorResponseClassification", "") or ""))
    reason = str(safe_get(mps, "sensorConfidenceBlockReason", "") or "")
    if reason and reason != "ok":
      block_reasons.append(reason)
    _append_finite(model_measured, safe_get(mps, "sensorModelMeasuredLatAccelDelta"))
    _append_finite(model_yaw, safe_get(mps, "sensorModelYawLatAccelDelta"))
    _append_finite(steering_yaw, safe_get(mps, "sensorSteeringYawLatAccelDelta"))
    _append_finite(model_yaw_signed, safe_get(mps, "sensorModelYawLatAccelSignedDelta"))
    _append_finite(steering_yaw_signed, safe_get(mps, "sensorSteeringYawLatAccelSignedDelta"))
    _append_finite(dtle, safe_get(mps, "dtleEstimate"))

  return {
    "samples": sum(1 for rec in records if rec.typ == "controlsState"),
    "availableSamples": available,
    "suppressCandidates": suppress,
    "disagreementCounts": _counts(disagreement),
    "responseClassCounts": _counts(response),
    "blockReasonCounts": _counts(block_reasons),
    "latAccelDeltaStats": {
      "modelMeasured": _stats(model_measured),
      "modelYaw": _stats(model_yaw),
      "steeringYaw": _stats(steering_yaw),
      "modelYawSigned": _stats(model_yaw_signed),
      "steeringYawSigned": _stats(steering_yaw_signed),
      "dtle": _stats(dtle),
    },
  }


def _grade_summary(records: list[Any]) -> dict[str, Any]:
  scenarios: list[str] = []
  grades: list[str] = []
  blocks: list[str] = []
  coast: list[float] = []
  confidence: list[float] = []
  bias: list[float] = []
  proposed: list[float] = []

  for rec in records:
    if rec.typ != "longitudinalPlanSP":
      continue
    ctx = safe_get(rec.payload, "longitudinalDebug.scenarioContext")
    if ctx is None:
      continue
    scenarios.append(str(safe_get(ctx, "scenario", "") or ""))
    grades.append(str(safe_get(ctx, "roadGrade", "") or ""))
    block = str(safe_get(ctx, "blockReason", "") or "")
    if block:
      blocks.append(block)
    _append_finite(coast, safe_get(ctx, "accelCoast"))
    _append_finite(confidence, safe_get(ctx, "gradeConfidence"))
    _append_finite(bias, safe_get(ctx, "estimatedAccelBias"))
    _append_finite(proposed, safe_get(ctx, "proposedCompensation"))

  return {
    "samples": sum(1 for rec in records if rec.typ == "longitudinalPlanSP"),
    "scenarioCounts": _counts(scenarios),
    "roadGradeCounts": _counts(grades),
    "blockReasonCounts": _counts(blocks),
    "accelCoastStats": _stats(coast),
    "gradeConfidenceStats": _stats(confidence),
    "estimatedAccelBiasStats": _stats(bias),
    "proposedCompensationStats": _stats(proposed),
  }


def _compare_control_invariants(baseline: list[Any], candidate: list[Any]) -> dict[str, Any]:
  base_by_type = _messages_by_type(baseline)
  cand_by_type = _messages_by_type(candidate)
  diff_counts: dict[str, int] = {}
  max_abs_delta: dict[str, float | None] = {}
  time_aligned: dict[str, bool] = {}
  for path in CONTROL_INVARIANT_PATHS:
    typ, field = path.split(".", 1)
    base_msgs = base_by_type.get(typ, [])
    cand_msgs = cand_by_type.get(typ, [])
    time_aligned[path] = [r.log_mono_time for r in base_msgs] == [r.log_mono_time for r in cand_msgs]
    count = abs(len(base_msgs) - len(cand_msgs))
    max_delta = 0.0
    for b, c in zip(base_msgs, cand_msgs, strict=False):
      bv = _finite(safe_get(b.payload, field))
      cv = _finite(safe_get(c.payload, field))
      if bv is None or cv is None:
        if bv != cv:
          count += 1
        continue
      delta = abs(bv - cv)
      if delta > 1e-6:
        count += 1
        max_delta = max(max_delta, delta)
    diff_counts[path] = count
    max_abs_delta[path] = _round(max_delta)
  return {
    "pass": all(v == 0 for v in diff_counts.values()) and all(time_aligned.values()),
    "comparison": "same_type_order_and_logMonoTime",
    "timeAligned": time_aligned,
    "diffCounts": diff_counts,
    "maxAbsDelta": max_abs_delta,
  }


def _messages_by_type(records: list[Any]) -> dict[str, list[Any]]:
  by_type: dict[str, list[Any]] = {}
  for rec in records:
    by_type.setdefault(rec.typ, []).append(rec)
  return by_type


def _finite(value: Any) -> float | None:
  try:
    f = float(value)
  except (TypeError, ValueError):
    return None
  return f if math.isfinite(f) else None


def _append_finite(values: list[float], value: Any) -> None:
  if (f := _finite(value)) is not None:
    values.append(f)


def _round(value: float | None, ndigits: int = 4) -> float | None:
  return None if value is None else round(float(value), ndigits)


def _percentile(values: list[float], p: float) -> float | None:
  if not values:
    return None
  ordered = sorted(values)
  idx = (len(ordered) - 1) * p / 100.0
  lo = math.floor(idx)
  hi = math.ceil(idx)
  if lo == hi:
    return ordered[int(idx)]
  return ordered[lo] * (hi - idx) + ordered[hi] * (idx - lo)


def _stats(values: list[float]) -> dict[str, float | None]:
  return {
    "count": len(values),
    "p50": _round(_percentile(values, 50.0)),
    "p95": _round(_percentile(values, 95.0)),
    "p99": _round(_percentile(values, 99.0)),
    "maxAbs": _round(max((abs(v) for v in values), default=0.0)),
  }


def _counts(values: list[str]) -> dict[str, int]:
  return dict(sorted(Counter(v for v in values if v).items()))


def main() -> None:
  parser = argparse.ArgumentParser(description="Profile shadow-only lateral confidence and grade telemetry from route logs.")
  parser.add_argument("route", help="Candidate route/log accepted by LogReader")
  parser.add_argument("--baseline", help="Optional baseline route/log for actuation/target invariant comparison")
  parser.add_argument("--qlog", action="store_true", help="Prefer qlogs instead of rlogs")
  parser.add_argument("--json", action="store_true")
  parser.add_argument("--output")
  args = parser.parse_args()

  msgs = load_route_msgs(args.route, qlog=args.qlog)
  baseline_msgs = load_route_msgs(args.baseline, qlog=args.qlog) if args.baseline else None
  report = build_shadow_heuristics_report(msgs, source=args.route, baseline_msgs=baseline_msgs)
  rendered = json.dumps(report.to_dict(), indent=2, sort_keys=True) if args.json else render_shadow_heuristics_report(report)
  if args.output:
    Path(args.output).write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")
  print(rendered)


if __name__ == "__main__":
  main()
