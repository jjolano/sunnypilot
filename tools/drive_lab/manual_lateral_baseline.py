from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path
from typing import Any

import numpy as np

from openpilot.tools.drive_lab.route_analysis import (
  LATERAL_DEMAND_SCHEMA_LEGACY,
  conditioned_desired_curvature,
  finite_or_none,
  lateral_demand_schema,
)
from openpilot.tools.drive_lab.route_io import output_report
from openpilot.tools.drive_lab.timeline import format_enum, safe_get


SPEED_BINS = ((3.0, 8.0), (8.0, 12.0), (12.0, 18.0), (18.0, 30.0), (30.0, float("inf")))
LAT_ACCEL_BINS = ((0.0, 0.3), (0.3, 0.8), (0.8, 1.5), (1.5, float("inf")))


@dataclass(frozen=True)
class ManualLateralSample:
  route: str
  t: float
  mode: str
  v_ego: float
  steering_angle_deg: float
  steering_torque: float | None
  current_curvature: float
  desired_curvature: float | None
  processed_desired_curvature: float | None
  raw_desired_curvature: float | None
  current_lat_accel: float
  desired_lat_accel: float | None
  lat_error: float | None
  lat_jerk: float | None
  steering_rate_deg_s: float | None
  curve_side: str
  speed_bin: str
  accel_bin: str
  phase: str
  model_path_quality: float | None = None
  model_path_gated: bool | None = None
  model_path_state: str | None = None


@dataclass(frozen=True)
class LateralBucketSummary:
  label: str
  sample_count: int
  duration_s: float
  median_abs_actual_lat_accel: float
  p95_abs_actual_lat_accel: float
  median_abs_lat_jerk: float
  p95_abs_lat_jerk: float
  median_abs_steering_rate: float
  p95_abs_steering_rate: float
  median_abs_lat_error: float | None = None
  p95_abs_lat_error: float | None = None


@dataclass(frozen=True)
class ManualLateralBaselineSummary:
  source: str
  sample_count: int
  manual_sample_count: int
  engaged_sample_count: int
  excluded_sample_count: int
  low_speed_excluded_count: int
  unknown_model_path_quality_count: int
  speed_bins: list[LateralBucketSummary]
  accel_bins: list[LateralBucketSummary]
  curve_side_means: dict[str, dict[str, float | None]]
  phase_counts: dict[str, int]
  notes: list[str]

  def to_dict(self) -> dict[str, Any]:
    return asdict(self)


@dataclass
class _LateralState:
  car_state: Any | None = None
  car_control: Any | None = None
  controls_state: Any | None = None
  model_v2: Any | None = None


def build_manual_lateral_samples(route: str, msgs: list[Any], *, already_sorted: bool = False) -> list[ManualLateralSample]:
  ordered = list(msgs) if already_sorted else sorted(msgs, key=lambda m: int(getattr(m, "logMonoTime", 0)))
  demand_schema = lateral_demand_schema(ordered)
  state = _LateralState()
  samples: list[ManualLateralSample] = []
  last_t_by_mode: dict[str, float] = {}
  last_current_lat_accel: dict[str, float] = {}
  last_sample_steering_angle: float | None = None
  last_t: float | None = None

  for msg in iter_route_messages_from_msgs(ordered):
    typ = msg.typ
    payload = msg.payload
    if typ == "carState":
      state.car_state = payload
    elif typ == "carControl":
      state.car_control = payload
    elif typ == "modelV2":
      state.model_v2 = payload
    elif typ == "controlsState":
      state.controls_state = payload
      sample = _sample_from_state(route, msg.t, state, last_t_by_mode, last_current_lat_accel, last_sample_steering_angle, last_t, demand_schema)
      if sample is not None:
        samples.append(sample)
        last_t_by_mode[sample.mode] = sample.t
        last_current_lat_accel[sample.mode] = sample.current_lat_accel
        last_sample_steering_angle = sample.steering_angle_deg
        last_t = sample.t
  return samples


def summarize_manual_lateral_baseline(samples: list[ManualLateralSample], *, source: str = "unknown") -> ManualLateralBaselineSummary:
  accepted = [sample for sample in samples if sample.mode in {"manual", "engaged"}]
  manual = [sample for sample in accepted if sample.mode == "manual"]
  engaged = [sample for sample in accepted if sample.mode == "engaged"]
  excluded = len(samples) - len(accepted)
  low_speed_excluded = sum(1 for sample in samples if sample.v_ego < 3.0)
  unknown_quality = sum(1 for sample in samples if sample.model_path_quality is None)

  return ManualLateralBaselineSummary(
    source=source,
    sample_count=len(samples),
    manual_sample_count=len(manual),
    engaged_sample_count=len(engaged),
    excluded_sample_count=excluded,
    low_speed_excluded_count=low_speed_excluded,
    unknown_model_path_quality_count=unknown_quality,
    speed_bins=[
      _summarize_bucket(samples, f"{mode} {_speed_bin_label(low, high)} m/s", lambda s, mode=mode, lo=low, hi=high: s.mode == mode and lo <= s.v_ego < hi)
      for mode in ("manual", "engaged") for low, high in SPEED_BINS
    ],
    accel_bins=[
      _summarize_bucket(samples, f"{mode} {_accel_bin_label(low, high)} m/s^2", lambda s, mode=mode, lo=low, hi=high: s.mode == mode and lo <= _bucket_lat_accel(s) < hi)
      for mode in ("manual", "engaged") for low, high in LAT_ACCEL_BINS
    ],
    curve_side_means=_curve_side_means(accepted),
    phase_counts=dict(Counter(sample.phase for sample in samples)),
    notes=["manual lateral is a descriptive envelope, not ground truth", "no live tuning conclusions should be drawn"],
  )


def render_manual_lateral_baseline(summary: ManualLateralBaselineSummary) -> str:
  lines = [
    f"Manual lateral baseline: {summary.source}",
    f"accepted samples: {summary.sample_count} manual={summary.manual_sample_count} engaged={summary.engaged_sample_count}",
    f"low-speed excluded (<3 m/s): {summary.low_speed_excluded_count}",
    "speed bins:",
  ]
  for bucket in summary.speed_bins:
    err = "" if bucket.median_abs_lat_error is None else f" err_med={bucket.median_abs_lat_error:.3f} err_p95={bucket.p95_abs_lat_error:.3f}"
    lines.append(f"  {bucket.label}: n={bucket.sample_count} act_med={bucket.median_abs_actual_lat_accel:.3f} act_p95={bucket.p95_abs_actual_lat_accel:.3f} jerk_p95={bucket.p95_abs_lat_jerk:.3f} steer_rate_p95={bucket.p95_abs_steering_rate:.3f}{err}")
  lines.append("accel bins:")
  for bucket in summary.accel_bins:
    err = "" if bucket.median_abs_lat_error is None else f" err_med={bucket.median_abs_lat_error:.3f} err_p95={bucket.p95_abs_lat_error:.3f}"
    lines.append(f"  {bucket.label}: n={bucket.sample_count} act_med={bucket.median_abs_actual_lat_accel:.3f} act_p95={bucket.p95_abs_actual_lat_accel:.3f} jerk_p95={bucket.p95_abs_lat_jerk:.3f}{err}")
  lines.append(f"curve side means: {summary.curve_side_means}")
  lines.append(f"phase counts: {summary.phase_counts}")
  lines.extend(f"note: {note}" for note in summary.notes)
  return "\n".join(lines)


def iter_route_messages_from_msgs(msgs: list[Any]):
  from openpilot.tools.drive_lab.route_analysis import build_route_messages
  return build_route_messages(msgs)


def _sample_from_state(
  route: str,
  t: float,
  state: _LateralState,
  last_t_by_mode: dict[str, float],
  last_current_lat_accel: dict[str, float],
  last_steering_angle: float | None,
  last_t: float | None,
  demand_schema: str = LATERAL_DEMAND_SCHEMA_LEGACY,
) -> ManualLateralSample | None:
  car_state = state.car_state
  controls_state = state.controls_state
  car_control = state.car_control
  if car_state is None or controls_state is None:
    return None
  v_ego = finite_or_none(safe_get(car_state, "vEgo"))
  steering_angle = finite_or_none(safe_get(car_state, "steeringAngleDeg"))
  if v_ego is None or steering_angle is None or not isfinite(v_ego) or v_ego < 3.0:
    return None
  blinker = bool(safe_get(car_state, "leftBlinker", False) or safe_get(car_state, "rightBlinker", False))
  lane_change_state = format_enum(safe_get(state.model_v2, "meta.laneChangeState", safe_get(controls_state, "laneChangeState", "")))
  standstill = bool(safe_get(car_state, "standstill", False))
  steering_pressed = bool(safe_get(car_state, "steeringPressed", False))
  lat_active = bool(safe_get(car_control, "latActive", safe_get(controls_state, "active", False)))
  lateral_state = _lateral_control_payload(controls_state)
  active_lat_state = bool(safe_get(lateral_state, "active", False))
  if blinker or lane_change_state not in ("", "off", "unknown", "preLaneChange") or standstill:
    return None
  mode = "engaged" if (lat_active and active_lat_state and not steering_pressed) else "manual"

  current_curvature = finite_or_none(safe_get(car_state, "curvature"))
  if current_curvature is None:
    current_curvature = finite_or_none(safe_get(controls_state, "curvature")) or 0.0
  desired_curvature = finite_or_none(safe_get(controls_state, "desiredCurvature"))
  model_path = safe_get(controls_state, "modelPathState")
  processed_desired = finite_or_none(conditioned_desired_curvature(model_path, demand_schema))
  raw_desired = finite_or_none(safe_get(model_path, "rawDesiredCurvature"))
  model_quality = finite_or_none(safe_get(model_path, "quality", safe_get(state.model_v2, "path.prob")))
  model_gated = safe_get(model_path, "gated")
  model_state = safe_get(model_path, "reason")
  actual_lat_accel_log = finite_or_none(safe_get(lateral_state, "actualLateralAccel"))
  desired_lat_accel_log = finite_or_none(safe_get(lateral_state, "desiredLateralAccel"))
  model_path_usable = bool(safe_get(model_path, "active", False)) or (model_quality is not None and model_quality > 0.0)
  desired_source = (
    processed_desired if processed_desired is not None and model_path_usable else
    desired_curvature if desired_curvature is not None else
    raw_desired
  )
  curvature_lat_accel = current_curvature * v_ego * v_ego
  current_lat_accel = actual_lat_accel_log if mode == "engaged" and actual_lat_accel_log is not None else curvature_lat_accel
  desired_lat_accel = desired_lat_accel_log if mode == "engaged" and desired_lat_accel_log is not None else desired_source * v_ego * v_ego if desired_source is not None else None
  lat_error = (desired_lat_accel - current_lat_accel) if (desired_lat_accel is not None and mode == "engaged") else None
  prev_t = last_t_by_mode.get(mode)
  prev_a = last_current_lat_accel.get(mode)
  lat_jerk = (current_lat_accel - prev_a) / (t - prev_t) if prev_t is not None and prev_a is not None and t > prev_t else None
  steering_rate = (steering_angle - last_steering_angle) / (t - last_t) if last_t is not None and last_steering_angle is not None and t > last_t else None
  return ManualLateralSample(
    route=route, t=t, mode=mode, v_ego=v_ego, steering_angle_deg=steering_angle, steering_torque=finite_or_none(safe_get(car_state, "steeringTorque")),
    current_curvature=current_curvature, desired_curvature=desired_curvature, processed_desired_curvature=processed_desired, raw_desired_curvature=raw_desired,
    current_lat_accel=current_lat_accel, desired_lat_accel=desired_lat_accel, lat_error=lat_error, lat_jerk=lat_jerk, steering_rate_deg_s=steering_rate,
    curve_side=_curve_side(current_curvature), speed_bin=_speed_bin(v_ego), accel_bin=_accel_bin(_bucket_lat_accel_values(mode, current_lat_accel, desired_lat_accel)), phase=_phase(lat_jerk),
    model_path_quality=model_quality, model_path_gated=bool(model_gated) if model_gated is not None else None, model_path_state=str(model_state) if model_state is not None else None,
  )


def _curve_side(curvature: float) -> str:
  if curvature > 1e-4:
    return "left"
  if curvature < -1e-4:
    return "right"
  return "straight"


def _speed_bin(v_ego: float) -> str:
  for low, high in SPEED_BINS:
    if low <= v_ego < high:
      return _speed_bin_label(low, high)
  return ">=30"


def _accel_bin(a: float) -> str:
  for low, high in LAT_ACCEL_BINS:
    if low <= a < high:
      return _accel_bin_label(low, high)
  return ">1.5"


def _bucket_lat_accel(sample: ManualLateralSample) -> float:
  return _bucket_lat_accel_values(sample.mode, sample.current_lat_accel, sample.desired_lat_accel)


def _bucket_lat_accel_values(mode: str, current_lat_accel: float, desired_lat_accel: float | None) -> float:
  if mode == "engaged" and desired_lat_accel is not None:
    return abs(desired_lat_accel)
  return abs(current_lat_accel)


def _speed_bin_label(low: float, high: float) -> str:
  return f">={low:.0f}" if high == float("inf") else f"{low:.0f}-{high:.0f}"


def _accel_bin_label(low: float, high: float) -> str:
  return f">={low:.1f}" if high == float("inf") else f"{low:.1f}-{high:.1f}"


def _phase(lat_jerk: float | None) -> str:
  if lat_jerk is None:
    return "steady"
  if abs(lat_jerk) < 0.15:
    return "steady"
  return "entry" if lat_jerk > 0 else "exit"


def _summarize_bucket(samples: list[ManualLateralSample], label: str, predicate) -> LateralBucketSummary:
  bucket = [sample for sample in samples if predicate(sample)]
  return LateralBucketSummary(
    label=label,
    sample_count=len(bucket),
    duration_s=_duration(bucket),
    median_abs_actual_lat_accel=_stat(bucket, lambda s: abs(s.current_lat_accel)),
    p95_abs_actual_lat_accel=_stat(bucket, lambda s: abs(s.current_lat_accel), 95),
    median_abs_lat_jerk=_stat(bucket, lambda s: abs(s.lat_jerk) if s.lat_jerk is not None else None),
    p95_abs_lat_jerk=_stat(bucket, lambda s: abs(s.lat_jerk) if s.lat_jerk is not None else None, 95),
    median_abs_steering_rate=_stat(bucket, lambda s: abs(s.steering_rate_deg_s) if s.steering_rate_deg_s is not None else None),
    p95_abs_steering_rate=_stat(bucket, lambda s: abs(s.steering_rate_deg_s) if s.steering_rate_deg_s is not None else None, 95),
    median_abs_lat_error=_stat_optional(bucket, lambda s: abs(s.lat_error) if s.lat_error is not None else None),
    p95_abs_lat_error=_stat_optional(bucket, lambda s: abs(s.lat_error) if s.lat_error is not None else None, 95),
  )


def _duration(samples: list[ManualLateralSample]) -> float:
  return max((s.t for s in samples), default=0.0) - min((s.t for s in samples), default=0.0)


def _stat(samples: list[ManualLateralSample], getter, pct: float | None = None) -> float:
  values = [value for sample in samples if (value := getter(sample)) is not None and isfinite(value)]
  if not values:
    return 0.0
  if pct is None or pct == 50:
    return float(np.median(values))
  return float(np.percentile(values, pct))


def _stat_optional(samples: list[ManualLateralSample], getter, pct: float | None = None) -> float | None:
  values = [value for sample in samples if (value := getter(sample)) is not None and isfinite(value)]
  if not values:
    return None
  if pct is None or pct == 50:
    return float(np.median(values))
  return float(np.percentile(values, pct))


def _curve_side_means(samples: list[ManualLateralSample]) -> dict[str, dict[str, float | None]]:
  out: dict[str, dict[str, float | None]] = {}
  for mode in ("manual", "engaged"):
    for side in ("left", "right", "straight"):
      side_samples = [s for s in samples if s.mode == mode and s.curve_side == side]
      out[f"{mode}:{side}"] = {
        "current_lat_accel": float(np.mean([s.current_lat_accel for s in side_samples])) if side_samples else None,
        "lat_error": float(np.mean([s.lat_error for s in side_samples if s.lat_error is not None])) if any(s.lat_error is not None for s in side_samples) else None,
      }
  return out


def _lateral_control_payload(controls_state: Any) -> Any | None:
  lateral_state = safe_get(controls_state, "lateralControlState")
  if lateral_state is None:
    return None
  which = getattr(lateral_state, "which", None)
  if callable(which):
    try:
      return safe_get(lateral_state, format_enum(which()))
    except Exception:
      pass
  for name in ("torqueState", "pidState", "angleState", "debugState"):
    payload = safe_get(lateral_state, name)
    if payload is not None:
      return payload
  return lateral_state


def main() -> None:
  import argparse
  from openpilot.tools.lib.logreader import LogReader, ReadMode
  from openpilot.tools.drive_lab.analyze_longitudinal_lateral_route import resolve_inputs, DEFAULT_LOG_ROOTS

  parser = argparse.ArgumentParser(description="Build a conservative manual-vs-engaged lateral route baseline.")
  parser.add_argument("inputs", nargs="+", help="Route ids, local dirs, files, or LogReader route strings")
  parser.add_argument("--qlog", action="store_true", help="Prefer qlogs")
  parser.add_argument("--json", action="store_true", help="Print JSON instead of text")
  parser.add_argument("--log-root", action="append", default=[], help="Extra local route search roots")
  args = parser.parse_args()
  read_mode = ReadMode.QLOG if args.qlog else ReadMode.AUTO
  log_roots = tuple(Path(p) for p in args.log_root) + DEFAULT_LOG_ROOTS
  identifiers = resolve_inputs(args.inputs, segment=None, read_mode=read_mode, log_roots=log_roots)
  msgs = list(LogReader(identifiers, default_mode=read_mode, sort_by_time=True))
  samples = build_manual_lateral_samples(", ".join(identifiers), msgs, already_sorted=True)
  summary = summarize_manual_lateral_baseline(samples, source=", ".join(identifiers))
  print(output_report(summary, json_output=args.json, renderer=render_manual_lateral_baseline))


if __name__ == "__main__":
  main()
