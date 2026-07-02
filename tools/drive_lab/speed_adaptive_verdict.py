#!/usr/bin/env python3
"""§5 validation gate — replay engaged routes through the real torque estimator with
speed-adaptive shadow collection forced, fit the speed-aware profile from the collected
buckets, and decide whether the cross-route evidence supports promotion to apply.

Run: uv run python -m openpilot.tools.drive_lab.speed_adaptive_verdict ROUTE [ROUTE ...]
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from cereal import car

from openpilot.common.params import Params
from openpilot.common.prefix import OpenpilotPrefix
from openpilot.selfdrive.locationd.torqued import TorqueEstimator
from openpilot.sunnypilot.custom.lateral.speed_aware_torque import (
  SPEED_BUCKET_BP,
  MIN_BIN_POINTS,
  MIN_CONFIDENCE,
  SpeedAwareTorqueRuntime,
  _fit_slope,
  fit_low_speed_section,
  fit_speed_aware_torque_profile,
  parse_speed_aware_torque_profile,
)
from openpilot.tools.drive_lab.route_io import load_route_msgs, output_report
from openpilot.tools.drive_lab.timeline import msg_payload, msg_time_s, msg_type, safe_get


@dataclass(frozen=True)
class SpeedAdaptiveRouteProfile:
  source: str
  anchors: list[float]
  ratios: list[float]
  confidence: list[float]
  points: list[int]
  global_slope: float | None
  bin_slopes: list[float | None]
  engaged_frames: int
  ratio_active_frames: int
  ratio_active_percent: float
  base_lat_accel_factor: float | None
  lat_accel_deltas: list[float | None]
  fitted: bool
  profile_source: str
  low_speed: dict[str, Any] | None = None

  def to_dict(self) -> dict[str, Any]:
    return asdict(self)


@dataclass(frozen=True)
class SpeedAdaptiveVerdictReport:
  routes: list[SpeedAdaptiveRouteProfile]
  verdict: str
  verdict_reason: str
  cross_route_ratio_spread: float | None
  confident_route_count: int

  def to_dict(self) -> dict[str, Any]:
    return asdict(self)


def _cp_lat_accel_factor(CP: car.CarParams) -> float | None:
  if CP.lateralTuning.which() != 'torque':
    return None
  val = safe_get(CP.lateralTuning.torque, "latAccelFactor")
  if isinstance(val, int | float) and float(val) > 0 and np.isfinite(float(val)):
    return float(val)
  return None


def _default_torque_cp() -> car.CarParams:
  cp = car.CarParams.new_message()
  cp.brand = "toyota"
  cp.carFingerprint = "TOYOTA_CAMRY"
  cp.lateralTuning.init('torque')
  cp.lateralTuning.torque.latAccelFactor = 2.0
  cp.lateralTuning.torque.friction = 0.2
  return cp


def _extract_cp(msgs: list[Any]) -> car.CarParams | None:
  for msg in msgs:
    if msg_type(msg) == "carParams":
      return msg_payload(msg)
  return None


def _extract_live_torque_factor(msgs: list[Any]) -> float | None:
  latest = None
  for msg in msgs:
    if msg_type(msg) == "liveTorqueParameters":
      latest = msg_payload(msg)
  if latest is None:
    return None
  val = safe_get(latest, "latAccelFactorFiltered")
  if isinstance(val, int | float) and float(val) > 0 and np.isfinite(float(val)):
    return float(val)
  return None


def _load_profile_json(CP: car.CarParams, path: str | Path) -> dict:
  data = json.loads(Path(path).read_text())
  if not isinstance(data, dict):
    raise ValueError(f"profile JSON must be an object: {path}")
  # Accept either a raw speed-aware profile or a single route profile wrapper.
  if "routes" in data:
    raise ValueError(f"--profile-json expects a single speed-aware profile, not a verdict report: {path}")
  profile = parse_speed_aware_torque_profile(CP, data)
  if profile is None:
    raise ValueError(f"profile JSON does not match CP restore key or is invalid: {path}")
  if isinstance(data.get("lowSpeed"), dict):
    profile["lowSpeed"] = data["lowSpeed"]
  return profile


def _collect_ratios(msgs: list[Any], runtime: SpeedAwareTorqueRuntime) -> tuple[int, int]:
  v_ego = float('nan')
  lat_active = False
  engaged = 0
  active = 0
  for msg in msgs:
    which = msg_type(msg)
    payload = msg_payload(msg)
    if which == "carState":
      v_ego = safe_get(payload, "vEgo", float('nan'))
    elif which == "carControl":
      lat_active = bool(safe_get(payload, "latActive", False))
    elif which == "livePose":
      if lat_active and np.isfinite(v_ego) and v_ego > 0.0:
        engaged += 1
        if runtime.ratio(v_ego) != 1.0:
          active += 1
  return engaged, active


def _empty_route_profile(source: str, fitted: bool, profile_source: str) -> SpeedAdaptiveRouteProfile:
  return SpeedAdaptiveRouteProfile(
    source=source,
    anchors=list(SPEED_BUCKET_BP),
    ratios=[1.0] * len(SPEED_BUCKET_BP),
    confidence=[0.0] * len(SPEED_BUCKET_BP),
    points=[0] * len(SPEED_BUCKET_BP),
    global_slope=None,
    bin_slopes=[None] * len(SPEED_BUCKET_BP),
    engaged_frames=0,
    ratio_active_frames=0,
    ratio_active_percent=0.0,
    base_lat_accel_factor=None,
    lat_accel_deltas=[None] * len(SPEED_BUCKET_BP),
    fitted=fitted,
    profile_source=profile_source,
    low_speed=None,
  )


def _bucket_slopes(buckets) -> list[float | None]:
  slopes: list[float | None] = []
  for _, bucket in buckets.bucket_items():
    pts = bucket.get_points()
    if len(pts) >= MIN_BIN_POINTS:
      slopes.append(_fit_slope(pts))
    else:
      slopes.append(None)
  return slopes


def _global_slope(buckets) -> float | None:
  all_points: list[Any] = []
  for _, bucket in buckets.bucket_items():
    pts = bucket.get_points()
    if len(pts) > 0:
      all_points.append(pts)
  if not all_points:
    return None
  return _fit_slope(np.vstack(all_points))


def analyze_route(
  msgs: list[Any],
  source: str = "unknown",
  *,
  profile_json: str | Path | None = None,
  already_sorted: bool = False,
) -> SpeedAdaptiveRouteProfile:
  if not already_sorted:
    msgs = sorted(msgs, key=lambda m: int(getattr(m, "logMonoTime", 0)))

  CP = _extract_cp(msgs) or _default_torque_cp()
  if CP.lateralTuning.which() != 'torque':
    return _empty_route_profile(source, fitted=profile_json is None, profile_source=str(profile_json) if profile_json else "fit")

  with OpenpilotPrefix():
    params = Params()
    # Force shadow collection so the replay populates speed_learning_buckets fresh.
    params.put("LiveTorqueSpeedAdaptiveMode", "shadow", block=True)
    params.put_bool("LiveTorqueLowSpeedShadow", True)
    estimator = TorqueEstimator(CP)
    # _update_params does not run on the init frame, so set the mode directly.
    estimator.speed_adaptive_mode = 'shadow'
    estimator.low_speed_shadow = True
    estimator.speed_adaptive_runtime.profile = None

    for msg in msgs:
      which = msg_type(msg)
      if which not in ("carControl", "carOutput", "carState", "liveCalibration", "liveDelay", "livePose"):
        continue
      payload = msg_payload(msg)
      t = msg_time_s(msg)
      estimator.handle_log(t, which, payload)

    base_factor = _extract_live_torque_factor(msgs)
    if base_factor is None:
      filtered = estimator.filtered_params.get('latAccelFactor')
      if filtered is not None:
        val = float(filtered.x)
        if np.isfinite(val) and val > 0.0:
          base_factor = val
    if base_factor is None:
      base_factor = _cp_lat_accel_factor(CP)

    if profile_json is not None:
      profile = _load_profile_json(CP, profile_json)
      fitted = False
      profile_source = str(profile_json)
    else:
      profile = fit_speed_aware_torque_profile(CP, estimator.speed_learning_buckets, low_speed_buckets=estimator.low_speed_buckets)
      fitted = True
      profile_source = "fit"

    if profile is None:
      return _empty_route_profile(source, fitted=fitted, profile_source=profile_source)

    runtime = SpeedAwareTorqueRuntime(profile=profile)
    engaged, active = _collect_ratios(msgs, runtime)
    ratio_active_percent = 100.0 * active / engaged if engaged > 0 else 0.0

    lat_accel_deltas = [(r - 1.0) * base_factor if base_factor is not None else None for r in profile['ratios']]

    low_speed = profile.get('lowSpeed')
    if low_speed is None and not fitted:
      low_speed = fit_low_speed_section(CP, estimator.low_speed_buckets, global_slope=_global_slope(estimator.speed_learning_buckets))

    return SpeedAdaptiveRouteProfile(
      source=source,
      anchors=list(profile['anchors']),
      ratios=list(profile['ratios']),
      confidence=list(profile['confidence']),
      points=list(profile['points']),
      global_slope=_global_slope(estimator.speed_learning_buckets),
      bin_slopes=_bucket_slopes(estimator.speed_learning_buckets),
      engaged_frames=engaged,
      ratio_active_frames=active,
      ratio_active_percent=ratio_active_percent,
      base_lat_accel_factor=base_factor,
      lat_accel_deltas=lat_accel_deltas,
      fitted=fitted,
      profile_source=profile_source,
      low_speed=low_speed,
    )


def build_speed_adaptive_verdict_report(route_profiles: list[SpeedAdaptiveRouteProfile]) -> SpeedAdaptiveVerdictReport:
  confident_routes = [
    r for r in route_profiles
    if any(c >= MIN_CONFIDENCE for c in r.confidence)
  ]
  confident_route_count = len(confident_routes)

  ratios_by_anchor: dict[float, list[float]] = {}
  for r in confident_routes:
    for anchor, conf, ratio in zip(r.anchors, r.confidence, r.ratios, strict=True):
      if conf >= MIN_CONFIDENCE:
        ratios_by_anchor.setdefault(float(anchor), []).append(float(ratio))

  shared_anchor_values = {
    anchor: vals for anchor, vals in ratios_by_anchor.items()
    if len(vals) == confident_route_count
  }
  spreads = [max(vals) - min(vals) for vals in shared_anchor_values.values()]
  max_spread = max(spreads) if spreads else None
  partial_overlap = any(len(vals) != confident_route_count for vals in ratios_by_anchor.values())

  if confident_route_count < 3:
    verdict = "insufficient_evidence"
    reason = f"only {confident_route_count} route(s) with confident anchors (need >=3)"
  elif not shared_anchor_values:
    verdict = "insufficient_evidence"
    reason = f"no shared confident anchors across {confident_route_count} confident routes"
  elif partial_overlap:
    verdict = "insufficient_evidence"
    reason = f"partial confident-anchor overlap across {confident_route_count} routes; need shared anchors to compare spread"
  elif max_spread is not None and max_spread < 0.05:
    verdict = "promote"
    reason = f"{confident_route_count} routes with confident anchors and max cross-route ratio spread {max_spread:.4f} < 0.05"
  else:
    verdict = "park"
    reason = f"cross-route ratio spread {max_spread:.4f} >= 0.05"

  return SpeedAdaptiveVerdictReport(
    routes=route_profiles,
    verdict=verdict,
    verdict_reason=reason,
    cross_route_ratio_spread=max_spread,
    confident_route_count=confident_route_count,
  )


def _fmt_optional(value: float | None, precision: int = 4) -> str:
  return "n/a" if value is None else f"{value:.{precision}f}"


def _fmt_slope_list(slopes: list[float | None]) -> str:
  return "[" + ", ".join(_fmt_optional(s) for s in slopes) + "]"


def render_speed_adaptive_route_profile(profile: SpeedAdaptiveRouteProfile) -> str:
  lines = [
    f"Speed-aware torque profile: {profile.source}",
    f"  fitted: {profile.fitted} ({profile.profile_source})",
    f"  anchors:          {profile.anchors}",
    f"  ratios:           {profile.ratios}",
    f"  confidence:       {profile.confidence}",
    f"  points:           {profile.points}",
    f"  global slope:     {_fmt_optional(profile.global_slope)}",
    f"  bin slopes:       {_fmt_slope_list(profile.bin_slopes)}",
    f"  engaged frames:   {profile.engaged_frames}",
    f"  ratio active:     {profile.ratio_active_frames} ({profile.ratio_active_percent:.1f}%)",
    f"  base latAccelFactor: {_fmt_optional(profile.base_lat_accel_factor)}",
    f"  latAccelFactor deltas: {_fmt_slope_list(profile.lat_accel_deltas)}",
  ]
  if profile.low_speed is not None:
    lines.extend([
      "  lowSpeed section:",
      f"    anchors:    {profile.low_speed.get('anchors')}",
      f"    ratios:     {profile.low_speed.get('ratios')}",
      f"    slopes:     {_fmt_slope_list(profile.low_speed.get('slopes', []))}",
      f"    confidence: {profile.low_speed.get('confidence')}",
      f"    points:     {profile.low_speed.get('points')}",
    ])
  return "\n".join(lines)


def render_speed_adaptive_verdict_report(report: SpeedAdaptiveVerdictReport) -> str:
  lines = [
    "Speed-aware torque verdict",
    f"  routes: {len(report.routes)}",
    f"  confident routes: {report.confident_route_count}",
    f"  max cross-route ratio spread: {_fmt_optional(report.cross_route_ratio_spread)}",
    f"  verdict: {report.verdict}",
    f"  reason: {report.verdict_reason}",
    "",
  ]
  for route in report.routes:
    lines.append(render_speed_adaptive_route_profile(route))
  return "\n\n".join(lines)


def save_speed_adaptive_verdict_report(report: SpeedAdaptiveVerdictReport, path: str | Path) -> None:
  Path(path).write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")


def load_speed_adaptive_verdict_report(path: str | Path) -> SpeedAdaptiveVerdictReport:
  data = json.loads(Path(path).read_text())
  routes = [SpeedAdaptiveRouteProfile(**r) for r in data["routes"]]
  return SpeedAdaptiveVerdictReport(
    routes=routes,
    verdict=str(data["verdict"]),
    verdict_reason=str(data["verdict_reason"]),
    cross_route_ratio_spread=data.get("cross_route_ratio_spread"),
    confident_route_count=int(data["confident_route_count"]),
  )


def main() -> None:
  parser = argparse.ArgumentParser(description="Speed-aware torque verdict analyzer.")
  parser.add_argument("routes", nargs="+", help="Routes, segments, log files, or URLs accepted by LogReader")
  parser.add_argument("--profile-json", help="Apply this speed-aware profile JSON instead of fitting from the log")
  parser.add_argument("--qlog", action="store_true", help="Prefer qlogs instead of rlogs")
  parser.add_argument("--json", action="store_true", help="Print JSON instead of text summary")
  parser.add_argument("--output", help="Write report JSON to this path")
  args = parser.parse_args()

  route_profiles = [
    analyze_route(load_route_msgs(route, qlog=args.qlog), source=route, profile_json=args.profile_json, already_sorted=True)
    for route in args.routes
  ]
  report = build_speed_adaptive_verdict_report(route_profiles)

  print(output_report(
    report,
    json_output=args.json,
    renderer=render_speed_adaptive_verdict_report,
    output_path=args.output,
    save=save_speed_adaptive_verdict_report,
  ))


if __name__ == "__main__":
  main()
