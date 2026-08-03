#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path
from typing import Any

import numpy as np

from openpilot.tools.drive_lab.route_io import load_route_msgs, output_report
from openpilot.tools.drive_lab.timeline import msg_payload, msg_time_s, msg_type, safe_get
from openpilot.sunnypilot.custom.lateral.block_jackknife import (
  MAX_BLOCK_REL_SE,
  MIN_EVIDENCE_BLOCKS,
  EvidenceBlockClock,
  fit_block_slope,
)
from openpilot.sunnypilot.custom.lateral.roll_comp_learning import (
  MIN_POINTS,
  MIN_X_SPAN as LIVE_MIN_X_SPAN,
  ROLL_COMP_SPEED_BANDS,
  ROLL_GAIN_MAX,
  ROLL_GAIN_MIN,
  _informative_blocks,
  _populated_extremes,
)

GRAVITY = 9.81
MIN_V_EGO = 15.0
MAX_DESIRED_LATERAL_ACCEL = 0.15
MAX_DESIRED_LATERAL_ACCEL_DELTA = 0.05
# Keep the public name aligned with the live learner's global p5-p95 gate.
MIN_X_SPAN = LIVE_MIN_X_SPAN
EVIDENCE_OPPORTUNITY_S = 0.25
ROLL_COMP_VERDICT_MIN_ROLL_SPAN = 0.3
ROLL_COMP_VERDICT_MAX_GAIN_SPREAD = 0.05
ROLL_COMP_VERDICT_MIN_ROUTE_COUNT = 3


@dataclass(frozen=True)
class RollCompProfileReport:
  source: str
  slope: float | None
  integrator_mean: float | None
  integrator_std: float | None
  point_count: int
  roll_span: float
  slope_rel_se: float | None = None
  block_count: int = 0
  quality_valid: bool = False
  quality_reason: str = "temporal-block quality not evaluated"

  def to_dict(self) -> dict[str, Any]:
    return asdict(self)


@dataclass(frozen=True)
class RollCompVerdictReport:
  routes: list[RollCompProfileReport]
  qualifying_route_count: int
  slope_spread: float | None
  verdict: str
  verdict_reason: str

  def to_dict(self) -> dict[str, Any]:
    return asdict(self)


@dataclass(frozen=True)
class _Frame:
  t_s: float
  lat_active: bool
  steering_pressed: bool
  v_ego: float
  roll: float
  p: float
  i: float
  f: float
  desired_lateral_accel: float
  saturated: bool
  torque_active: bool


def _extract_frames(msgs: list[Any]) -> list[_Frame]:
  if not msgs:
    return []

  latest: dict[str, Any] = {}
  frames: list[_Frame] = []

  for msg in msgs:
    typ = msg_type(msg)
    payload = msg_payload(msg)
    if typ in ("carState", "carControl", "liveParameters"):
      latest[typ] = payload
    if typ != "controlsState":
      continue

    lateral_control = safe_get(payload, "lateralControlState")
    if lateral_control is None:
      continue
    which = safe_get(lateral_control, "which")
    if callable(which):
      which = which()
    if which != "torqueState":
      continue

    torque_state = safe_get(lateral_control, "torqueState")
    if torque_state is None:
      continue

    car_state = latest.get("carState")
    car_control = latest.get("carControl")
    live_params = latest.get("liveParameters")
    if car_state is None or car_control is None or live_params is None:
      continue

    roll = _float_or_nan(safe_get(live_params, "roll"))
    v_ego = _float_or_nan(safe_get(car_state, "vEgo"))
    p = _float_or_nan(safe_get(torque_state, "p"))
    i = _float_or_nan(safe_get(torque_state, "i"))
    f_val = _float_or_nan(safe_get(torque_state, "f"))
    desired = _float_or_nan(safe_get(torque_state, "desiredLateralAccel"))

    frames.append(_Frame(
      t_s=msg_time_s(msg),
      lat_active=bool(safe_get(car_control, "latActive", False)),
      steering_pressed=bool(safe_get(car_state, "steeringPressed", False)),
      v_ego=v_ego,
      roll=roll,
      p=p,
      i=i,
      f=f_val,
      desired_lateral_accel=desired,
      saturated=bool(safe_get(torque_state, "saturated", False)),
      torque_active=bool(safe_get(torque_state, "active", True)),
    ))

  return frames


def _float_or_nan(value: Any) -> float:
  try:
    return float(value)
  except (TypeError, ValueError, OverflowError):
    return float("nan")


def _passes_straight_gates(frame: _Frame, delta_ok: bool, strict_straight: bool,
                           v_lo: float, v_hi: float | None) -> bool:
  max_desired = MAX_DESIRED_LATERAL_ACCEL_DELTA if strict_straight else MAX_DESIRED_LATERAL_ACCEL
  if not all(isfinite(value) for value in (
    frame.v_ego, frame.roll, frame.p, frame.i, frame.f, frame.desired_lateral_accel,
  )):
    return False
  if not frame.lat_active or not frame.torque_active:
    return False
  if frame.steering_pressed:
    return False
  if frame.v_ego < v_lo:
    return False
  if v_hi is not None and frame.v_ego >= v_hi:
    return False
  if abs(frame.desired_lateral_accel) > max_desired:
    return False
  if not delta_ok:
    return False
  return not frame.saturated


def _select_straight_frames(frames: list[_Frame], strict_straight: bool = False,
                            v_lo: float = MIN_V_EGO, v_hi: float | None = None) -> list[_Frame]:
  selected: list[_Frame] = []
  previous_desired: float | None = None

  for frame in frames:
    delta_ok = previous_desired is None or abs(frame.desired_lateral_accel - previous_desired) <= MAX_DESIRED_LATERAL_ACCEL_DELTA
    previous_desired = frame.desired_lateral_accel
    if _passes_straight_gates(frame, delta_ok, strict_straight, v_lo, v_hi):
      selected.append(frame)

  return selected


def _collect_evidence(frames: list[_Frame], strict_straight: bool, v_lo: float,
                      v_hi: float | None) -> tuple[list[tuple[float, float, int, float]], EvidenceBlockClock]:
  """Collect one timestamp-spaced opportunity at most every quarter second."""
  clock = EvidenceBlockClock()
  next_opportunity: float | None = None
  previous_desired: float | None = None
  rows: list[tuple[float, float, int, float]] = []

  for frame in frames:
    block_id = clock.advance(frame.t_s)
    delta_ok = previous_desired is None or abs(frame.desired_lateral_accel - previous_desired) <= MAX_DESIRED_LATERAL_ACCEL_DELTA
    previous_desired = frame.desired_lateral_accel
    if not isfinite(frame.t_s):
      continue
    if next_opportunity is None:
      next_opportunity = frame.t_s

    if next_opportunity is None or frame.t_s < next_opportunity:
      continue

    # Consume exactly one opportunity.  Missing timestamp slots are skipped rather
    # than replayed on later frames.
    next_opportunity = frame.t_s + EVIDENCE_OPPORTUNITY_S
    if block_id is None or not _passes_straight_gates(frame, delta_ok, strict_straight, v_lo, v_hi):
      continue

    x = -float(np.sin(frame.roll)) * GRAVITY
    y = frame.p + frame.i + frame.f
    if isfinite(x) and isfinite(y):
      rows.append((x, y, int(block_id), frame.i))

  return rows, clock


def _quality_reason(points: np.ndarray, informative_blocks: list[int], fit: Any,
                    roll_span: float) -> tuple[bool, str]:
  block_count = len(informative_blocks)
  point_count = len(points)
  if block_count < MIN_EVIDENCE_BLOCKS:
    return False, f"only {block_count} informative completed block(s); need >= {MIN_EVIDENCE_BLOCKS}"
  if point_count < MIN_POINTS:
    return False, f"only {point_count} fitted informative point(s); need >= {MIN_POINTS}"

  lo_x, hi_x = _populated_extremes(points)
  if lo_x is None or hi_x is None:
    return False, "global p5-p95 x leverage is unavailable"
  global_span = hi_x - lo_x
  if not isfinite(global_span) or global_span < MIN_X_SPAN:
    return False, f"global p5-p95 x span {global_span:.4f} < {MIN_X_SPAN:.2f}"
  if lo_x >= 0.0 or hi_x <= 0.0:
    return False, f"global x sign gate failed (p5={lo_x:.4f}, p95={hi_x:.4f})"
  if fit is None:
    return False, "centered block slope is not computable"

  expected_blocks = set(informative_blocks)
  if not isfinite(float(fit.slope)) or fit.slope <= 0.0:
    return False, "full slope is non-finite or non-positive"
  if set(fit.block_slopes) != expected_blocks:
    return False, "a fitted informative block slope is unavailable"
  if any(not isfinite(float(slope)) or slope <= 0.0 for slope in fit.block_slopes.values()):
    return False, "an informative block slope is non-finite or non-positive"
  if set(fit.loo_slopes) != expected_blocks:
    return False, "a leave-one-block-out slope is unavailable"
  if any(not isfinite(float(slope)) or slope <= 0.0 for slope in fit.loo_slopes.values()):
    return False, "a leave-one-block-out slope is non-finite or non-positive"
  if not (ROLL_GAIN_MIN <= fit.slope <= ROLL_GAIN_MAX):
    return False, f"full slope {fit.slope:.4f} is outside [{ROLL_GAIN_MIN:.1f}, {ROLL_GAIN_MAX:.1f}]"
  if any(not (ROLL_GAIN_MIN <= slope <= ROLL_GAIN_MAX) for slope in fit.loo_slopes.values()):
    return False, f"a leave-one-block-out slope is outside [{ROLL_GAIN_MIN:.1f}, {ROLL_GAIN_MAX:.1f}]"
  if not isfinite(float(fit.rel_se)):
    return False, "block jackknife relative SE is non-finite"
  if fit.rel_se > MAX_BLOCK_REL_SE:
    return False, f"block jackknife relative SE {fit.rel_se:.4f} > {MAX_BLOCK_REL_SE:.4f}"
  if not isfinite(roll_span) or roll_span < ROLL_COMP_VERDICT_MIN_ROLL_SPAN:
    return False, f"roll span {roll_span:.4f} < {ROLL_COMP_VERDICT_MIN_ROLL_SPAN:.1f}"
  return True, "all temporal-block quality gates passed"


def build_roll_comp_profile(
  msgs: list[Any],
  source: str = "unknown",
  already_sorted: bool = False,
  strict_straight: bool = False,
  v_lo: float = MIN_V_EGO,
  v_hi: float | None = None,
) -> RollCompProfileReport:
  if not already_sorted:
    msgs = sorted(msgs, key=lambda m: int(getattr(m, "logMonoTime", 0)))

  frames = sorted(_extract_frames(msgs), key=lambda frame: frame.t_s)
  rows, clock = _collect_evidence(frames, strict_straight, v_lo, v_hi)
  completed_rows = [row for row in rows if clock.is_completed(row[2])]
  points = np.asarray([row[:3] for row in completed_rows], dtype=float)
  if not len(points):
    points = np.empty((0, 3), dtype=float)

  informative_blocks = _informative_blocks(points)
  if informative_blocks:
    informative_points = points[np.isin(points[:, 2].astype(int), informative_blocks)]
  else:
    informative_points = np.empty((0, 3), dtype=float)

  if len(informative_points):
    xs = informative_points[:, 0]
    roll_span = float(np.max(xs) - np.min(xs))
  else:
    roll_span = 0.0
  integrators = np.asarray([row[3] for row in completed_rows], dtype=float)
  fit = fit_block_slope(informative_points) if len(informative_points) else None
  slope = float(fit.slope) if fit is not None and isfinite(float(fit.slope)) else None
  slope_rel_se = float(fit.rel_se) if fit is not None and isfinite(float(fit.rel_se)) else None
  quality_valid, quality_reason = _quality_reason(informative_points, informative_blocks, fit, roll_span)

  return RollCompProfileReport(
    source=source,
    slope=slope,
    integrator_mean=float(np.mean(integrators)) if len(integrators) else None,
    integrator_std=float(np.std(integrators)) if len(integrators) else None,
    point_count=len(informative_points),
    roll_span=roll_span,
    slope_rel_se=slope_rel_se,
    block_count=len(informative_blocks),
    quality_valid=quality_valid,
    quality_reason=quality_reason,
  )


def _report_is_quality_qualified(report: RollCompProfileReport) -> bool:
  try:
    return bool(
      type(report.quality_valid) is bool
      and report.quality_valid
      and type(report.quality_reason) is str
      and bool(report.quality_reason.strip())
      and type(report.point_count) is int
      and report.point_count >= MIN_POINTS
      and type(report.block_count) is int
      and report.block_count >= MIN_EVIDENCE_BLOCKS
      and type(report.roll_span) in (int, float)
      and not isinstance(report.roll_span, bool)
      and isfinite(float(report.roll_span))
      and report.roll_span >= ROLL_COMP_VERDICT_MIN_ROLL_SPAN
      and report.slope is not None
      and type(report.slope) in (int, float)
      and not isinstance(report.slope, bool)
      and isfinite(float(report.slope))
      and ROLL_GAIN_MIN <= float(report.slope) <= ROLL_GAIN_MAX
      and report.slope_rel_se is not None
      and type(report.slope_rel_se) in (int, float)
      and not isinstance(report.slope_rel_se, bool)
      and isfinite(float(report.slope_rel_se))
      and float(report.slope_rel_se) >= 0.0
      and float(report.slope_rel_se) <= MAX_BLOCK_REL_SE
    )
  except (TypeError, ValueError, OverflowError):
    return False


def _canonical_route_source(source: Any) -> str:
  return source.strip() if type(source) is str else ""


def build_roll_comp_verdict_report(route_reports: list[RollCompProfileReport]) -> RollCompVerdictReport:
  distinct_routes: dict[str, RollCompProfileReport] = {}
  for report in route_reports:
    if not _report_is_quality_qualified(report):
      continue
    source = _canonical_route_source(report.source)
    if source and source not in distinct_routes:
      distinct_routes[source] = report
  qualifying_routes = list(distinct_routes.values())
  qualifying_route_count = len(qualifying_routes)
  finite_slopes = [float(report.slope) for report in qualifying_routes if report.slope is not None and isfinite(report.slope)]
  slope_spread = max(finite_slopes) - min(finite_slopes) if len(finite_slopes) >= 2 else None

  if qualifying_route_count < ROLL_COMP_VERDICT_MIN_ROUTE_COUNT:
    verdict = "insufficient-data"
    reason = f"only {qualifying_route_count} distinct quality route source(s) (need >=3)"
  elif len(finite_slopes) != qualifying_route_count:
    verdict = "insufficient-data"
    reason = f"{qualifying_route_count} distinct quality route source(s), but only {len(finite_slopes)} have a finite slope"
  elif slope_spread is not None and slope_spread < ROLL_COMP_VERDICT_MAX_GAIN_SPREAD:
    verdict = "promote"
    reason = (
      f"{qualifying_route_count} distinct quality route sources with roll span >= "
      + f"{ROLL_COMP_VERDICT_MIN_ROLL_SPAN:.1f} m/s^2 and learned gain spread "
      + f"{slope_spread:.4f} < {ROLL_COMP_VERDICT_MAX_GAIN_SPREAD:.2f}"
    )
  else:
    verdict = "park"
    reason = (
      f"{qualifying_route_count} distinct quality route sources with roll span >= "
      + f"{ROLL_COMP_VERDICT_MIN_ROLL_SPAN:.1f} m/s^2 but learned gain spread "
      + f"{slope_spread:.4f} >= {ROLL_COMP_VERDICT_MAX_GAIN_SPREAD:.2f}"
    )

  return RollCompVerdictReport(
    routes=route_reports,
    qualifying_route_count=qualifying_route_count,
    slope_spread=slope_spread,
    verdict=verdict,
    verdict_reason=reason,
  )


def render_roll_comp_profile(report: RollCompProfileReport) -> str:
  slope_str = f"{report.slope:.4f}" if report.slope is not None else "n/a"
  mean_str = f"{report.integrator_mean:.4f}" if report.integrator_mean is not None else "n/a"
  std_str = f"{report.integrator_std:.4f}" if report.integrator_std is not None else "n/a"
  rel_se_str = _fmt_optional(report.slope_rel_se)
  lines = [
    f"Roll compensation profile for {report.source}",
    f"  slope:            {slope_str}",
    f"  slope rel SE:     {rel_se_str}",
    f"  integrator mean:  {mean_str}",
    f"  integrator std:   {std_str}",
    f"  point count:      {report.point_count}",
    f"  block count:      {report.block_count}",
    f"  roll span (m/s^2): {report.roll_span:.4f}",
    f"  quality valid:    {report.quality_valid}",
    f"  quality reason:   {report.quality_reason}",
  ]
  return "\n".join(lines)


def _fmt_optional(value: float | None, precision: int = 4) -> str:
  return "n/a" if value is None else f"{value:.{precision}f}"


def render_roll_comp_profile_brief(report: RollCompProfileReport) -> str:
  slope_str = _fmt_optional(report.slope)
  rel_se_str = _fmt_optional(report.slope_rel_se)
  lines = [
    f"Roll compensation profile for {report.source}",
    f"  slope:            {slope_str}",
    f"  slope rel SE:     {rel_se_str}",
    f"  roll span (m/s^2): {report.roll_span:.4f}",
    f"  point count:      {report.point_count}",
    f"  block count:      {report.block_count}",
    f"  quality valid:    {report.quality_valid}",
    f"  quality reason:   {report.quality_reason}",
  ]
  return "\n".join(lines)


def render_roll_comp_verdict_report(report: RollCompVerdictReport) -> str:
  lines = [
    "Roll compensation verdict (Phase 2 route gate)",
    f"  routes: {len(report.routes)}",
    f"  distinct quality route sources with roll span >= {ROLL_COMP_VERDICT_MIN_ROLL_SPAN:.1f} m/s^2: {report.qualifying_route_count}",
    f"  slope spread: {_fmt_optional(report.slope_spread)}",
    f"  verdict: {report.verdict}",
    f"  reason: {report.verdict_reason}",
    "",
  ]
  for route in report.routes:
    lines.append(render_roll_comp_profile_brief(route))
  return "\n\n".join(lines)


def save_roll_comp_profile(report: RollCompProfileReport, path: str | Path) -> None:
  Path(path).write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")


def save_roll_comp_verdict_report(report: RollCompVerdictReport, path: str | Path) -> None:
  Path(path).write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")


def _malformed_saved_report(reason: str, source: str = "unknown") -> RollCompProfileReport:
  detail = str(reason).strip() or "invalid payload"
  return RollCompProfileReport(
    source=source if type(source) is str else "unknown",
    slope=None,
    integrator_mean=None,
    integrator_std=None,
    point_count=0,
    roll_span=0.0,
    slope_rel_se=None,
    block_count=0,
    quality_valid=False,
    quality_reason=f"malformed saved report: {detail}",
  )


def _saved_string(data: dict[str, Any], field: str, *, nonempty: bool = False) -> str:
  if field not in data or type(data[field]) is not str:
    raise ValueError(f"{field} must be a string")
  value = data[field]
  if nonempty and not value.strip():
    raise ValueError(f"{field} must be nonempty")
  return value


def _saved_real(data: dict[str, Any], field: str, *, allow_none: bool = False,
                nonnegative: bool = False) -> float | None:
  if field not in data:
    raise ValueError(f"missing {field}")
  value = data[field]
  if value is None:
    if allow_none:
      return None
    raise ValueError(f"{field} cannot be null")
  if type(value) not in (int, float):
    raise ValueError(f"{field} must be a real number")
  try:
    value = float(value)
  except (TypeError, ValueError, OverflowError) as exc:
    raise ValueError(f"{field} must be a finite real number") from exc
  if not isfinite(value):
    raise ValueError(f"{field} must be a finite real number")
  if nonnegative and value < 0.0:
    raise ValueError(f"{field} cannot be negative")
  return value


def _saved_count(data: dict[str, Any], field: str) -> int:
  if field not in data or type(data[field]) is not int or data[field] < 0:
    raise ValueError(f"{field} must be a nonnegative integer")
  return data[field]


def load_roll_comp_profile(path: str | Path) -> RollCompProfileReport:
  source = "unknown"
  try:
    data = json.loads(Path(path).read_text())
    if type(data) is not dict:
      raise ValueError("JSON root must be an object")

    raw_source = data.get("source")
    if type(raw_source) is str:
      source = raw_source
    source = _saved_string(data, "source", nonempty=True)
    slope = _saved_real(data, "slope", allow_none=True)
    integrator_mean = _saved_real(data, "integrator_mean", allow_none=True)
    integrator_std = _saved_real(data, "integrator_std", allow_none=True)
    point_count = _saved_count(data, "point_count")
    roll_span = _saved_real(data, "roll_span", nonnegative=True)
    if roll_span is None:
      raise ValueError("roll_span cannot be null")

    quality_fields = ("slope_rel_se", "block_count", "quality_valid", "quality_reason")
    present_quality_fields = [field for field in quality_fields if field in data]
    if not present_quality_fields:
      return RollCompProfileReport(
        source=source,
        slope=slope,
        integrator_mean=integrator_mean,
        integrator_std=integrator_std,
        point_count=point_count,
        roll_span=roll_span,
        slope_rel_se=None,
        block_count=0,
        quality_valid=False,
        quality_reason="legacy report missing temporal-block quality fields; cannot promote",
      )
    if len(present_quality_fields) != len(quality_fields):
      raise ValueError("quality fields are incomplete")

    if type(data["quality_valid"]) is not bool:
      raise ValueError("quality_valid must be a JSON boolean")
    quality_reason = _saved_string(data, "quality_reason", nonempty=True)
    slope_rel_se = _saved_real(data, "slope_rel_se", allow_none=True, nonnegative=True)
    block_count = _saved_count(data, "block_count")
    return RollCompProfileReport(
      source=source,
      slope=slope,
      integrator_mean=integrator_mean,
      integrator_std=integrator_std,
      point_count=point_count,
      roll_span=roll_span,
      slope_rel_se=slope_rel_se,
      block_count=block_count,
      quality_valid=data["quality_valid"],
      quality_reason=quality_reason,
    )
  except (OSError, UnicodeError, TypeError, ValueError, OverflowError, json.JSONDecodeError) as exc:
    return _malformed_saved_report(str(exc), source)


def build_roll_comp_band_reports(
  msgs: list[Any],
  source: str = "unknown",
  already_sorted: bool = False,
  strict_straight: bool = False,
) -> list[RollCompProfileReport]:
  """One report per device speed band — the shadow gate compares these against the
  device learner's rollCompBandGains (agreement within +-0.1)."""
  if not already_sorted:
    msgs = sorted(msgs, key=lambda m: int(getattr(m, "logMonoTime", 0)))
  return [
    build_roll_comp_profile(msgs, source=f"{source} [{v_lo:g}-{v_hi:g} m/s]", already_sorted=True,
                            strict_straight=strict_straight, v_lo=v_lo, v_hi=v_hi)
    for v_lo, v_hi in ROLL_COMP_SPEED_BANDS
  ]


def main() -> None:
  parser = argparse.ArgumentParser(description="Profile roll-compensation gain from route logs.")
  parser.add_argument("routes", nargs="+", help="Route, segment range, log file, or URLs accepted by LogReader")
  parser.add_argument("--output", help="Write report JSON to this path")
  parser.add_argument("--json", action="store_true", help="Print JSON instead of a text summary")
  parser.add_argument("--qlog", action="store_true", help="Prefer qlogs instead of rlogs")
  parser.add_argument("--strict-straight", action="store_true", help="Use the tighter 0.05 m/s^2 straightness gate")
  parser.add_argument("--speed-bands", action="store_true",
                      help="Fit per device speed band (ROLL_COMP_SPEED_BANDS) instead of the single >15 m/s gate")
  args = parser.parse_args()

  if args.speed_bands:
    band_reports = [
      report
      for route in args.routes
      for report in build_roll_comp_band_reports(
        load_route_msgs(route, qlog=args.qlog), source=route, already_sorted=True, strict_straight=args.strict_straight,
      )
    ]
    payload = [report.to_dict() for report in band_reports]
    if args.output:
      Path(args.output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if args.json:
      print(json.dumps(payload, indent=2, sort_keys=True))
    else:
      print("\n\n".join(render_roll_comp_profile(report) for report in band_reports))
    return

  route_reports = [
    build_roll_comp_profile(
      load_route_msgs(route, qlog=args.qlog),
      source=route,
      already_sorted=True,
      strict_straight=args.strict_straight,
    )
    for route in args.routes
  ]

  if len(route_reports) == 1:
    print(output_report(
      route_reports[0],
      json_output=args.json,
      renderer=render_roll_comp_profile,
      output_path=args.output,
      save=save_roll_comp_profile,
    ))
    return

  report = build_roll_comp_verdict_report(route_reports)

  print(output_report(
    report,
    json_output=args.json,
    renderer=render_roll_comp_verdict_report,
    output_path=args.output,
    save=save_roll_comp_verdict_report,
  ))


if __name__ == "__main__":
  main()
