#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any

from openpilot.tools.drive_lab.compare_manual_planner_targets import PlannerTargetSample, extract_planner_target_samples
from openpilot.tools.drive_lab.route_io import output_report
from openpilot.tools.lib.logreader import ReadMode


def _fmt(value: float | None, ndigits: int = 2) -> float | None:
  if value is None or not math.isfinite(float(value)):
    return None
  return round(float(value), ndigits)


def _finite(value: Any) -> float | None:
  try:
    numeric = float(value)
  except (TypeError, ValueError):
    return None
  return numeric if math.isfinite(numeric) else None


@dataclass(frozen=True)
class LeadPolicyBinSpec:
  distance_bins: tuple[float, ...] = (45.0, 80.0)
  plan_brake_thresholds: tuple[float, ...] = (-0.3, -0.5, -0.7)
  low_required_decel: float = 0.3
  examples: int = 20


@dataclass(frozen=True)
class LeadPolicySampleView:
  route: str
  route_id: str
  segment: int | None
  t: float
  source: str
  sp_source: str
  sp_stack: str
  d_rel: float
  v_rel: float
  closing_speed: float
  headway: float
  required_decel: float
  plan_a_target: float
  a_ego: float
  model_desired_accel: float | None
  plan_should_stop: bool
  fcw: bool


@dataclass
class LeadPolicyBucketStats:
  label: str
  count: int = 0
  d_rel: list[float] = field(default_factory=list)
  headway: list[float] = field(default_factory=list)
  v_rel: list[float] = field(default_factory=list)
  closing_speed: list[float] = field(default_factory=list)
  required_decel: list[float] = field(default_factory=list)
  plan_a_target: list[float] = field(default_factory=list)
  a_ego: list[float] = field(default_factory=list)
  planner_sources: Counter[str] = field(default_factory=Counter)
  sp_sources: Counter[str] = field(default_factory=Counter)
  sp_stacks: Counter[str] = field(default_factory=Counter)
  plan_brake_counts: dict[str, int] = field(default_factory=dict)
  a_ego_brake_counts: dict[str, int] = field(default_factory=dict)
  non_closing_plan_brake_count: int = 0
  low_required_decel_plan_brake_count: int = 0
  samples: list[LeadPolicySampleView] = field(default_factory=list)

  def add(self, sample: LeadPolicySampleView, plan_thresholds: tuple[float, ...], low_required_decel: float) -> None:
    self.count += 1
    self.d_rel.append(sample.d_rel)
    self.headway.append(sample.headway)
    self.v_rel.append(sample.v_rel)
    self.closing_speed.append(sample.closing_speed)
    self.required_decel.append(sample.required_decel)
    self.plan_a_target.append(sample.plan_a_target)
    self.a_ego.append(sample.a_ego)
    self.planner_sources[sample.source] += 1
    self.sp_sources[sample.sp_source] += 1
    self.sp_stacks[sample.sp_stack] += 1
    self.samples.append(sample)
    for threshold in plan_thresholds:
      key = f"plan_a_target<={threshold}"
      self.plan_brake_counts[key] = self.plan_brake_counts.get(key, 0) + int(sample.plan_a_target <= threshold)
      key2 = f"a_ego<={threshold}"
      self.a_ego_brake_counts[key2] = self.a_ego_brake_counts.get(key2, 0) + int(sample.a_ego <= threshold)
    if sample.closing_speed <= 0 and sample.plan_a_target <= plan_thresholds[0]:
      self.non_closing_plan_brake_count += 1
    elif sample.required_decel <= low_required_decel and sample.plan_a_target <= plan_thresholds[0]:
      self.low_required_decel_plan_brake_count += 1

  def to_dict(self, *, examples: int = 20, plan_thresholds: tuple[float, ...] = (-0.3,), low_required_decel: float = 0.3) -> dict[str, Any]:
    return {
      "label": self.label,
      "count": self.count,
      "d_rel": _stats(self.d_rel),
      "headway": _stats(self.headway),
      "v_rel": _stats(self.v_rel),
      "closing_speed": _stats(self.closing_speed),
      "required_decel": _stats(self.required_decel),
      "plan_a_target": _stats(self.plan_a_target),
      "a_ego": _stats(self.a_ego),
      "planner_sources": dict(self.planner_sources),
      "sp_sources": dict(self.sp_sources),
      "sp_stacks": dict(self.sp_stacks),
      "plan_brake_counts": dict(self.plan_brake_counts),
      "plan_brake_fractions": _fractions(self.plan_brake_counts, self.count),
      "a_ego_brake_counts": dict(self.a_ego_brake_counts),
      "a_ego_brake_fractions": _fractions(self.a_ego_brake_counts, self.count),
      "non_closing_plan_brake_count": self.non_closing_plan_brake_count,
      "low_required_decel_plan_brake_count": self.low_required_decel_plan_brake_count,
      "source_buckets": _source_bucket_dict(self.samples, plan_thresholds, low_required_decel),
      "examples": [asdict(sample) for sample in _worst_examples(self.samples, examples, plan_thresholds[0], low_required_decel)],
    }


@dataclass
class LeadPolicyReport:
  spec: LeadPolicyBinSpec
  total_samples: int
  routes: list[str]
  bin_names: list[str]
  buckets: dict[str, LeadPolicyBucketStats]
  per_route_bins: dict[str, dict[str, LeadPolicyBucketStats]]

  def to_dict(self) -> dict[str, Any]:
    return {
      "spec": {
        "distance_bins": list(self.spec.distance_bins),
        "plan_brake_thresholds": list(self.spec.plan_brake_thresholds),
        "low_required_decel": self.spec.low_required_decel,
        "examples": self.spec.examples,
      },
      "total_samples": self.total_samples,
      "routes": self.routes,
      "bin_names": self.bin_names,
      "buckets": {k: v.to_dict(examples=self.spec.examples, plan_thresholds=self.spec.plan_brake_thresholds, low_required_decel=self.spec.low_required_decel)
                  for k, v in self.buckets.items()},
      "per_route_bins": {
        route: {k: v.to_dict(examples=self.spec.examples, plan_thresholds=self.spec.plan_brake_thresholds, low_required_decel=self.spec.low_required_decel)
                for k, v in buckets.items()}
        for route, buckets in self.per_route_bins.items()
      },
    }


def assign_distance_bin(d_rel: float, distance_bins: tuple[float, ...]) -> str:
  if not distance_bins:
    return "all"
  sorted_bins = tuple(sorted(distance_bins))
  if d_rel < sorted_bins[0]:
    return f"near_<{int(sorted_bins[0])}m"
  for lower, upper in zip((sorted_bins[0],) + sorted_bins[:-1], sorted_bins):
    if lower <= d_rel < upper:
      return f"mid_{int(lower)}_{int(upper)}m"
  return f"far_>={int(sorted_bins[-1])}m"


def _stats(values: list[float]) -> dict[str, float | None]:
  clean = [value for value in values if math.isfinite(float(value))]
  return {
    "count": len(clean),
    "median": _fmt(_percentile(clean, 50.0) if clean else None),
    "p10": _fmt(_percentile(clean, 10.0) if clean else None),
    "p90": _fmt(_percentile(clean, 90.0) if clean else None),
  }


def _percentile(values: list[float], percentile: float) -> float:
  if not values:
    return 0.0
  ordered = sorted(values)
  if len(ordered) == 1:
    return ordered[0]
  rank = (len(ordered) - 1) * percentile / 100.0
  lo = math.floor(rank)
  hi = math.ceil(rank)
  if lo == hi:
    return ordered[int(rank)]
  return ordered[lo] * (hi - rank) + ordered[hi] * (rank - lo)


def _fractions(counts: dict[str, int], total: int) -> dict[str, float]:
  return {key: round(value / total, 3) if total else 0.0 for key, value in counts.items()}


def _source_bucket_dict(samples: list[LeadPolicySampleView], plan_thresholds: tuple[float, ...], low_required_decel: float) -> dict[str, dict[str, Any]]:
  grouped: dict[str, list[LeadPolicySampleView]] = defaultdict(list)
  for sample in samples:
    grouped[sample.source].append(sample)
  return {source: _sample_group_summary(source_samples, plan_thresholds, low_required_decel) for source, source_samples in sorted(grouped.items())}


def _sample_group_summary(samples: list[LeadPolicySampleView], plan_thresholds: tuple[float, ...], low_required_decel: float) -> dict[str, Any]:
  counts = {f"plan_a_target<={threshold}": sum(1 for sample in samples if sample.plan_a_target <= threshold) for threshold in plan_thresholds}
  actual_counts = {f"a_ego<={threshold}": sum(1 for sample in samples if sample.a_ego <= threshold) for threshold in plan_thresholds}
  primary_threshold = plan_thresholds[0]
  non_closing = sum(1 for sample in samples if sample.closing_speed <= 0 and sample.plan_a_target <= primary_threshold)
  low_required = sum(1 for sample in samples if sample.closing_speed > 0 and sample.required_decel <= low_required_decel and sample.plan_a_target <= primary_threshold)
  return {
    "count": len(samples),
    "d_rel": _stats([sample.d_rel for sample in samples]),
    "headway": _stats([sample.headway for sample in samples]),
    "v_rel": _stats([sample.v_rel for sample in samples]),
    "closing_speed": _stats([sample.closing_speed for sample in samples]),
    "required_decel": _stats([sample.required_decel for sample in samples]),
    "plan_a_target": _stats([sample.plan_a_target for sample in samples]),
    "a_ego": _stats([sample.a_ego for sample in samples]),
    "plan_brake_counts": counts,
    "plan_brake_fractions": _fractions(counts, len(samples)),
    "a_ego_brake_counts": actual_counts,
    "a_ego_brake_fractions": _fractions(actual_counts, len(samples)),
    "non_closing_plan_brake_count": non_closing,
    "low_required_decel_plan_brake_count": low_required,
  }


def _worst_examples(samples: list[LeadPolicySampleView], limit: int, primary_threshold: float, low_required_decel: float) -> list[LeadPolicySampleView]:
  if limit <= 0:
    return []

  def rank(sample: LeadPolicySampleView) -> tuple[int, float, float]:
    low_risk_brake = sample.plan_a_target <= primary_threshold and (sample.closing_speed <= 0 or sample.required_decel <= low_required_decel)
    return (0 if low_risk_brake else 1, sample.plan_a_target, sample.required_decel)

  return sorted(samples, key=rank)[:limit]


def is_actuation_applicable(sample: PlannerTargetSample) -> bool:
  d_rel = _finite(sample.lead_d_rel)
  v_rel = _finite(sample.lead_v_rel)
  plan_a = _finite(sample.plan_a_target)
  a_ego = _finite(sample.a_ego)
  v_ego = _finite(sample.v_ego)
  return (
    sample.long_active
    and sample.selfdrive_active
    and not sample.gas_pressed
    and not sample.brake_pressed
    and sample.lead_status
    and v_ego is not None and v_ego >= 8.0
    and d_rel is not None and d_rel > 0.0
    and v_rel is not None
    and plan_a is not None
    and a_ego is not None
  )


def _view(sample: PlannerTargetSample) -> LeadPolicySampleView:
  d_rel = float(sample.lead_d_rel) if sample.lead_d_rel is not None else math.nan
  v_rel = float(sample.lead_v_rel) if sample.lead_v_rel is not None else math.nan
  closing_speed = max(0.0, -v_rel) if math.isfinite(v_rel) else math.nan
  headway = d_rel / sample.v_ego if sample.v_ego > 0 and math.isfinite(d_rel) else math.nan
  required_decel = (closing_speed ** 2) / (2.0 * max(d_rel, 0.1)) if math.isfinite(closing_speed) and math.isfinite(d_rel) else math.nan
  return LeadPolicySampleView(
    route=sample.route,
    route_id=sample.route_id,
    segment=sample.segment,
    t=sample.t,
    source=sample.plan_source,
    sp_source=sample.sp_source,
    sp_stack=sample.sp_stack,
    d_rel=d_rel,
    v_rel=v_rel,
    closing_speed=closing_speed,
    headway=headway,
    required_decel=required_decel,
    plan_a_target=sample.plan_a_target,
    a_ego=sample.a_ego,
    model_desired_accel=sample.model_desired_accel,
    plan_should_stop=sample.plan_should_stop,
    fcw=sample.plan_fcw,
  )


def build_lead_policy_report(samples_by_route: dict[str, list[PlannerTargetSample]], spec: LeadPolicyBinSpec) -> LeadPolicyReport:
  buckets: dict[str, LeadPolicyBucketStats] = {}
  per_route_bins: dict[str, dict[str, LeadPolicyBucketStats]] = defaultdict(dict)
  total = 0
  routes = list(samples_by_route)
  bin_names = []
  for route, samples in samples_by_route.items():
    for sample in samples:
      if not is_actuation_applicable(sample):
        continue
      total += 1
      view = _view(sample)
      bin_name = assign_distance_bin(view.d_rel, spec.distance_bins)
      if bin_name not in buckets:
        buckets[bin_name] = LeadPolicyBucketStats(label=bin_name)
        bin_names.append(bin_name)
      if bin_name not in per_route_bins[route]:
        per_route_bins[route][bin_name] = LeadPolicyBucketStats(label=bin_name)
      buckets[bin_name].add(view, spec.plan_brake_thresholds, spec.low_required_decel)
      per_route_bins[route][bin_name].add(view, spec.plan_brake_thresholds, spec.low_required_decel)
  return LeadPolicyReport(spec=spec, total_samples=total, routes=routes, bin_names=bin_names, buckets=buckets, per_route_bins=per_route_bins)


def render_report(report: LeadPolicyReport) -> str:
  lines = [f"Lead policy profile: {report.total_samples} samples across {len(report.routes)} routes"]
  for name in report.bin_names:
    bucket = report.buckets[name]
    lines.append(f"  {name}: n={bucket.count} sources={dict(bucket.planner_sources)}")
    bucket_dict = bucket.to_dict(examples=0, plan_thresholds=report.spec.plan_brake_thresholds, low_required_decel=report.spec.low_required_decel)
    lines.append(f"    d_rel median {bucket_dict['d_rel']['median']}m, headway median {bucket_dict['headway']['median']}s")
    lines.append(f"    plan brake {bucket.plan_brake_counts}; low-required={bucket.low_required_decel_plan_brake_count} non-closing={bucket.non_closing_plan_brake_count}")
  return "\n".join(lines)


def _parse_float_csv(text: str) -> tuple[float, ...]:
  values = tuple(float(part.strip()) for part in text.split(",") if part.strip())
  if any(not math.isfinite(value) for value in values):
    raise ValueError(f"non-finite value in {text!r}")
  return values


def output_lead_policy_report(report: LeadPolicyReport, *, json_output: bool = False, output_path: str | None = None) -> str:
  return output_report(report, json_output=json_output, renderer=render_report, output_path=output_path)


def main() -> None:
  parser = argparse.ArgumentParser(description="Profile lead policy behavior by distance and source.")
  parser.add_argument("routes", nargs="+", help="Routes, segment ranges, log files, or URLs accepted by LogReader")
  parser.add_argument("--qlog", action="store_true", help="Prefer qlogs instead of rlogs")
  parser.add_argument("--json", action="store_true", help="Print JSON instead of text")
  parser.add_argument("--output", help="Write JSON summary to this path")
  parser.add_argument("--distance-bins", default="45,80")
  parser.add_argument("--plan-brake-thresholds", default="-0.3,-0.5,-0.7")
  parser.add_argument("--low-required-decel", type=float, default=0.3)
  parser.add_argument("--examples", type=int, default=20)
  args = parser.parse_args(sys.argv[1:])

  spec = LeadPolicyBinSpec(
    distance_bins=_parse_float_csv(args.distance_bins),
    plan_brake_thresholds=_parse_float_csv(args.plan_brake_thresholds),
    low_required_decel=args.low_required_decel,
    examples=args.examples,
  )
  read_mode = ReadMode.QLOG if args.qlog else ReadMode.AUTO
  samples_by_route = {route: extract_planner_target_samples(route, read_mode) for route in args.routes}
  report = build_lead_policy_report(samples_by_route, spec)
  print(output_lead_policy_report(report, json_output=args.json, output_path=args.output))


if __name__ == "__main__":
  main()
