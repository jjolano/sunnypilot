#!/usr/bin/env python3
"""Multi-route lateral corpus runner.

Feeds real route frames through ``LateralDemandPipeline`` and produces
aggregate metrics: gating rates, fallback frequencies, path quality
distributions, curvature tracking error, and demand-source time budgets.

Example::

    uv run python tools/drive_lab/corpus_lateral.py ROUTE1 ROUTE2 --json
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from openpilot.sunnypilot.custom.lateral.demand.pipeline import LateralDemandPipeline
from openpilot.tools.drive_lab.fuzz_lateral_route_replay import (
  DT,
  LateralRouteFrame,
  _frame_to_inputs,
  extract_lateral_route_frames_with_summary,
)
from openpilot.tools.drive_lab.log_profile import LateralProfile, build_lateral_profile
from openpilot.tools.drive_lab.route_io import load_route_msgs


@dataclass(frozen=True)
class CorpusFrameOutput:
  """Per-frame pipeline output captured by the corpus runner."""

  t: float
  v_ego: float
  raw_curvature: float
  processed_curvature: float
  measured_curvature: float
  path_quality: float
  path_reason: str
  gated: bool
  demand_source: str
  curvature_limited: bool


@dataclass(frozen=True)
class RouteLateralMetrics:
  """Aggregate lateral metrics for a single route."""

  gating_rate: float
  fallback_rate: float
  path_quality_p5: float
  path_quality_p50: float
  path_quality_p95: float
  curvature_rmse: float
  source_distribution: dict[str, float]
  curvature_limited_rate: float

  def to_dict(self) -> dict[str, Any]:
    return {
      "gating_rate": self.gating_rate,
      "fallback_rate": self.fallback_rate,
      "path_quality_p5": self.path_quality_p5,
      "path_quality_p50": self.path_quality_p50,
      "path_quality_p95": self.path_quality_p95,
      "curvature_rmse": self.curvature_rmse,
      "source_distribution": dict(self.source_distribution),
      "curvature_limited_rate": self.curvature_limited_rate,
    }

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> RouteLateralMetrics:
    return cls(
      gating_rate=float(data.get("gating_rate", 0.0)),
      fallback_rate=float(data.get("fallback_rate", 0.0)),
      path_quality_p5=float(data.get("path_quality_p5", 0.0)),
      path_quality_p50=float(data.get("path_quality_p50", 0.0)),
      path_quality_p95=float(data.get("path_quality_p95", 0.0)),
      curvature_rmse=float(data.get("curvature_rmse", 0.0)),
      source_distribution=dict(data.get("source_distribution", {})),
      curvature_limited_rate=float(data.get("curvature_limited_rate", 0.0)),
    )


@dataclass(frozen=True)
class RouteCorpusResult:
  """Lateral corpus result for one route."""

  route: str
  frame_count: int
  metrics: RouteLateralMetrics
  profile: LateralProfile
  error: str | None = None

  def to_dict(self) -> dict[str, Any]:
    payload: dict[str, Any] = {
      "route": self.route,
      "frame_count": self.frame_count,
      "metrics": self.metrics.to_dict(),
      "profile": self.profile.to_dict(),
    }
    if self.error is not None:
      payload["error"] = self.error
    return payload

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> RouteCorpusResult:
    return cls(
      route=str(data["route"]),
      frame_count=int(data.get("frame_count", 0)),
      metrics=RouteLateralMetrics.from_dict(data["metrics"]),
      profile=LateralProfile.from_dict(data["profile"]),
      error=data.get("error"),
    )


@dataclass(frozen=True)
class CorpusReport:
  """Aggregate lateral corpus report across routes."""

  routes: tuple[RouteCorpusResult, ...]
  aggregate_metrics: dict[str, float]

  def to_dict(self) -> dict[str, Any]:
    return {
      "routes": [route.to_dict() for route in self.routes],
      "aggregate_metrics": dict(self.aggregate_metrics),
    }

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> CorpusReport:
    return cls(
      routes=tuple(RouteCorpusResult.from_dict(r) for r in data.get("routes", [])),
      aggregate_metrics=dict(data.get("aggregate_metrics", {})),
    )


def _run_frames(frames: tuple[LateralRouteFrame, ...]) -> tuple[CorpusFrameOutput, ...]:
  """Run lateral route frames through a fresh ``LateralDemandPipeline``."""

  pipeline = LateralDemandPipeline(dt=DT)
  outputs: list[CorpusFrameOutput] = []
  for frame in frames:
    inputs = _frame_to_inputs(frame)
    result = pipeline.update(inputs)
    demand = result.demand
    outputs.append(CorpusFrameOutput(
      t=frame.t,
      v_ego=frame.v_ego,
      raw_curvature=demand.raw_curvature,
      processed_curvature=demand.processed_curvature,
      measured_curvature=demand.measured_curvature,
      path_quality=demand.path_quality,
      path_reason=result.model_path_result.reason,
      gated=result.model_path_result.gated,
      demand_source=demand.demand_source,
      curvature_limited=demand.curvature_limited,
    ))
  return tuple(outputs)


def _empty_metrics() -> RouteLateralMetrics:
  return RouteLateralMetrics(
    gating_rate=0.0,
    fallback_rate=0.0,
    path_quality_p5=0.0,
    path_quality_p50=0.0,
    path_quality_p95=0.0,
    curvature_rmse=0.0,
    source_distribution={},
    curvature_limited_rate=0.0,
  )


def _compute_metrics(outputs: tuple[CorpusFrameOutput, ...]) -> RouteLateralMetrics:
  """Compute per-route lateral metrics from pipeline outputs."""

  if not outputs:
    return _empty_metrics()

  n = len(outputs)
  gated = sum(1 for o in outputs if o.gated)
  fallback = sum(1 for o in outputs if o.path_reason != "valid")
  curvature_limited = sum(1 for o in outputs if o.curvature_limited)

  quality = np.array([o.path_quality for o in outputs], dtype=float)
  raw = np.array([o.raw_curvature for o in outputs], dtype=float)
  processed = np.array([o.processed_curvature for o in outputs], dtype=float)

  diff = processed - raw
  finite_diff = diff[np.isfinite(diff)]
  rmse = float(np.sqrt(np.mean(finite_diff * finite_diff))) if finite_diff.size else 0.0

  source_counts: dict[str, int] = {}
  for o in outputs:
    source_counts[o.demand_source] = source_counts.get(o.demand_source, 0) + 1
  source_distribution = {source: count / n for source, count in source_counts.items()}

  def _pct(values: np.ndarray, pct: float) -> float:
    if values.size == 0:
      return 0.0
    return float(np.percentile(values, pct))

  return RouteLateralMetrics(
    gating_rate=gated / n,
    fallback_rate=fallback / n,
    path_quality_p5=_pct(quality, 5.0),
    path_quality_p50=_pct(quality, 50.0),
    path_quality_p95=_pct(quality, 95.0),
    curvature_rmse=rmse,
    source_distribution=source_distribution,
    curvature_limited_rate=curvature_limited / n,
  )


def _mean_median_p95(values: list[float]) -> dict[str, float]:
  arr = np.array([v for v in values if math.isfinite(v)], dtype=float)
  if arr.size == 0:
    return {"mean": 0.0, "median": 0.0, "p95": 0.0}
  return {
    "mean": float(np.mean(arr)),
    "median": float(np.median(arr)),
    "p95": float(np.percentile(arr, 95.0)),
  }


def _aggregate_metrics(results: list[RouteCorpusResult]) -> dict[str, float]:
  """Aggregate per-route metrics into corpus-level summaries."""

  if not results:
    return {}

  aggregate: dict[str, float] = {}

  for metric_name in (
    "gating_rate",
    "fallback_rate",
    "path_quality_p5",
    "path_quality_p50",
    "path_quality_p95",
    "curvature_rmse",
    "curvature_limited_rate",
  ):
    values = [getattr(r.metrics, metric_name) for r in results if r.error is None]
    stats = _mean_median_p95(values)
    for stat_name, value in stats.items():
      aggregate[f"{metric_name}_{stat_name}"] = value

  # Average source distribution across routes (missing sources count as 0).
  all_sources = set()
  for r in results:
    if r.error is None:
      all_sources.update(r.metrics.source_distribution.keys())

  for source in sorted(all_sources):
    values = [r.metrics.source_distribution.get(source, 0.0) for r in results if r.error is None]
    stats = _mean_median_p95(values)
    for stat_name, value in stats.items():
      aggregate[f"source_distribution_{source}_{stat_name}"] = value

  aggregate["route_count"] = float(len(results))
  aggregate["total_frames"] = float(sum(r.frame_count for r in results))

  return aggregate


def run_route(
  route: str,
  *,
  qlog: bool = False,
  max_frames: int | None = None,
) -> RouteCorpusResult:
  """Extract, run, and profile one route."""

  try:
    msgs = load_route_msgs(route, qlog=qlog)
    frames, _summary = extract_lateral_route_frames_with_summary(
      msgs,
      route=route,
      qlog=qlog,
      max_frames=max_frames,
    )
    outputs = _run_frames(frames)
    metrics = _compute_metrics(outputs)
    profile = build_lateral_profile(msgs, source=route)
    return RouteCorpusResult(
      route=route,
      frame_count=len(frames),
      metrics=metrics,
      profile=profile,
    )
  except Exception as exc:
    empty_profile = build_lateral_profile([], source=route)
    return RouteCorpusResult(
      route=route,
      frame_count=0,
      metrics=_empty_metrics(),
      profile=empty_profile,
      error=f"{type(exc).__name__}: {exc}",
    )


def run_corpus(
  routes: list[str],
  *,
  qlog: bool = False,
  max_frames: int | None = None,
) -> CorpusReport:
  """Run the lateral corpus across a list of routes."""

  results = [run_route(route, qlog=qlog, max_frames=max_frames) for route in routes]
  aggregate = _aggregate_metrics(results)
  return CorpusReport(routes=tuple(results), aggregate_metrics=aggregate)


def _sanitize(value: Any) -> Any:
  """Recursively sanitize floats for strict JSON output."""

  if isinstance(value, np.generic):
    return _sanitize(value.item())
  if isinstance(value, float):
    return value if math.isfinite(value) else None
  if isinstance(value, dict):
    return {k: _sanitize(v) for k, v in value.items()}
  if isinstance(value, (list, tuple)):
    return [_sanitize(v) for v in value]
  return value


def _render_report(report: CorpusReport) -> str:
  lines: list[str] = [
    "Drive Lab lateral corpus report",
    f"Routes: {len(report.routes)}",
    "",
  ]

  for result in report.routes:
    if result.error:
      lines.append(f"{result.route}: ERROR {result.error}")
      continue
    m = result.metrics
    lines.append(f"{result.route}: {result.frame_count} frames")
    lines.append(f"  gating_rate={m.gating_rate:.4f} fallback_rate={m.fallback_rate:.4f} curvature_limited_rate={m.curvature_limited_rate:.4f}")
    lines.append(f"  quality p5/p50/p95={m.path_quality_p5:.3f}/{m.path_quality_p50:.3f}/{m.path_quality_p95:.3f}")
    lines.append(f"  curvature_rmse={m.curvature_rmse:.6f}")
    if m.source_distribution:
      sources = " ".join(f"{k}={v:.3f}" for k, v in sorted(m.source_distribution.items()))
      lines.append(f"  sources: {sources}")

  lines.append("")
  lines.append("Aggregate metrics:")
  for key, value in sorted(report.aggregate_metrics.items()):
    lines.append(f"  {key}={value:.6f}")

  return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description="Multi-route lateral corpus runner.")
  parser.add_argument("routes", nargs="+", help="Route(s) or log files")
  parser.add_argument("--output", help="Write report JSON")
  parser.add_argument("--json", action="store_true", help="Emit JSON instead of text")
  parser.add_argument("--qlog", action="store_true", help="Load qlog files")
  parser.add_argument("--max-frames", type=int, default=None, help="Max frames per route")
  args = parser.parse_args(argv)

  if args.max_frames is not None and args.max_frames <= 0:
    parser.error("--max-frames must be > 0")

  report = run_corpus(args.routes, qlog=args.qlog, max_frames=args.max_frames)
  sanitized = _sanitize(report.to_dict())

  if args.output:
    Path(args.output).write_text(json.dumps(sanitized, indent=2, sort_keys=True, allow_nan=False))

  if args.json:
    print(json.dumps(sanitized, indent=2, sort_keys=True, allow_nan=False))
  else:
    print(_render_report(report))

  return 0 if all(r.error is None for r in report.routes) else 1


if __name__ == "__main__":
  raise SystemExit(main())
