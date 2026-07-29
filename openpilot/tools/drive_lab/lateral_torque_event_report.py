from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from openpilot.tools.drive_lab.timeline import format_enum, msg_payload, msg_time_s, msg_type, safe_get


SHAPING_REASON_NAMES = {
  1 << 0: "STEERING_PRESSED",
  1 << 1: "RELEASE",
  1 << 2: "SIGN_CONFLICT",
  1 << 3: "OVER_RESPONSE",
  1 << 4: "NEAR_ISO_ACCEL",
  1 << 5: "BUMP",
  1 << 6: "LOW_SPEED_STEER_LIMITED",
  1 << 7: "OUTPUT_RATE_LIMITED",
  1 << 8: "SAME_SIGN_UNWIND",
  1 << 9: "STEERING_RATE_COMFORT",
  1 << 10: "ACTUATOR_LAG_COMFORT",
  1 << 11: "STALE_ACTUATOR_REVERSAL",
  1 << 12: "SAFETY_LIMITED_RAMP",
  1 << 13: "HIGH_SPEED_ACTUATOR_LAG_UNWIND",
  1 << 14: "SAFETY_LIMITED_SIGN_HOLD",
}

GOVERNOR_REASON_NAMES = {
  1 << 0: "CLIPPED",
  1 << 1: "SLEW_LIMITED",
  1 << 2: "SIGN_CHANGE_LIMITED",
  1 << 3: "DRIVER_OVERRIDE",
  1 << 4: "SAME_DIRECTION_LIMIT",
  1 << 5: "HIGH_STEERING_RATE",
  1 << 6: "INVALID",
  1 << 7: "UNDER_RESPONSE_FLOOR",
}

V21_GOVERNOR_REASON_NAMES = {
  1 << 0: "CLIPPED",
  1 << 1: "SLEW_LIMITED",
  1 << 2: "SIGN_CHANGE_LIMITED",
  1 << 3: "SAME_DIRECTION_LIMIT",
  1 << 4: "HIGH_STEERING_RATE",
  1 << 5: "OVER_RESPONSE",
  1 << 6: "SIGN_CONFLICT",
  1 << 7: "NEAR_ISO_ACCEL",
  1 << 8: "OVERRIDE_RELEASE",
  1 << 9: "UNDER_RESPONSE_FLOOR",
  1 << 10: "INVALID",
  1 << 11: "UNDER_RESPONSE_GUARDED",
  1 << 12: "STEERING_RATE_COMFORT",
  1 << 13: "TARGET_ARRIVAL",
  # Telemetry-only condition marker (LateralSlewScaleMode apply); counted under its own
  # name for per-route condition verification, never a governor limit event.
  1 << 14: "SLEW_SCALE_APPLIED",
}

V3_GOVERNOR_REASON_NAMES = {
  **GOVERNOR_REASON_NAMES,
  1 << 5: "TOYOTA_HIGH_RATE",
}

V4_GOVERNOR_REASON_NAMES = {
  **GOVERNOR_REASON_NAMES,
  1 << 7: "STALE_ACTUATOR_MISMATCH",
  1 << 8: "LOW_SPEED_UNDER_RESPONSE_RECOVERY",
}

LOW_SPEED_TIER_BOUNDS = ((0.0, 3.0), (3.0, 5.0), (5.0, 8.0), (8.0, 12.0))
LOW_SPEED_REPORT_MAX_SPEED = LOW_SPEED_TIER_BOUNDS[-1][1]
LOW_SPEED_TURN_MIN_LAT_ACCEL = 0.035
LOW_SPEED_TURN_MIN_CURVATURE = 0.0015
LOW_SPEED_TURN_MIN_STEERING_ANGLE_DEG = 3.0


@dataclass(frozen=True)
class LateralTorqueEvent:
  start_s: float
  end_s: float
  score: float
  sample_count: int
  likely_source: str
  speed_mps_median: float
  steering_angle_pp: float
  steering_rate_p95: float
  steering_rate_reversals: int
  output_pp: float
  unshaped_output_pp: float
  applied_torque_pp: float
  output_reversals: int
  unshaped_output_reversals: int
  applied_torque_reversals: int
  desired_lateral_accel_pp: float
  actual_lateral_accel_pp: float
  desired_lateral_accel_reversals: int
  actual_lateral_accel_reversals: int
  desired_actual_corr: float | None
  shaping_active_percent: float
  release_active_percent: float
  steer_limited_percent: float
  output_cap_median: float
  steer_limit_error_pp: float
  shaping_reason_counts: dict[str, int]
  governor_reason_counts: dict[str, int]


@dataclass(frozen=True)
class LateralTorqueEventReport:
  source: str
  sample_count: int
  duration_s: float
  active_percent: float
  fast_torque_event_count: int
  top_events: list[LateralTorqueEvent]

  def to_dict(self) -> dict[str, Any]:
    return asdict(self)

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> LateralTorqueEventReport:
    return cls(
      source=str(data.get("source", "unknown")),
      sample_count=int(data.get("sample_count", data.get("sampleCount", 0))),
      duration_s=float(data.get("duration_s", data.get("durationS", 0.0))),
      active_percent=float(data.get("active_percent", data.get("activePercent", 0.0))),
      fast_torque_event_count=int(data.get("fast_torque_event_count", data.get("fastTorqueEventCount", 0))),
      top_events=[_event_from_dict(event) for event in data.get("top_events", data.get("topEvents", ()))],
    )


@dataclass(frozen=True)
class LateralTorqueLagMetrics:
  segment: str
  sample_count: int
  best_lag_s: float | None
  desired_actual_corr: float | None
  abs_error_mean: float
  abs_error_p95: float
  steering_rate_p95: float
  output_reversals: int
  steer_limited_percent: float
  high_steering_rate_percent: float
  desired_lateral_jerk_p95: float
  actual_lateral_jerk_p95: float
  desired_lateral_accel_residual_abs_p95: float
  actual_lateral_accel_residual_abs_p95: float
  model_path_low_quality_percent: float
  model_path_reason_counts: dict[str, int]
  shaping_reason_counts: dict[str, int]
  governor_reason_counts: dict[str, int]


@dataclass(frozen=True)
class LateralLowSpeedTierMetrics:
  segment: str
  speed_lower_mps: float
  speed_upper_mps: float
  sample_count: int
  best_lag_s: float | None
  desired_actual_corr: float | None
  abs_error_mean: float
  abs_error_p95: float
  output_reversals: int
  unshaped_output_reversals: int
  desired_lateral_accel_reversals: int
  actual_lateral_accel_reversals: int
  steering_rate_p95: float
  steer_limited_percent: float
  high_steering_rate_percent: float
  raw_processed_curvature_delta_p95: float
  desired_processed_curvature_delta_p95: float
  model_path_gated_percent: float
  model_path_quality_median: float
  model_path_reason_counts: dict[str, int]


@dataclass(frozen=True)
class LateralLowSpeedReport:
  source: str
  sample_count: int
  duration_s: float
  lane_change_excluded_count: int
  signal_tagged_category_counts: dict[str, int]
  signal_tagged_state_counts: dict[str, int]
  tiers: list[LateralLowSpeedTierMetrics]
  signal_tagged_tiers: list[LateralLowSpeedTierMetrics]

  def to_dict(self) -> dict[str, Any]:
    return asdict(self)


@dataclass(frozen=True)
class LateralTorqueLagReport:
  source: str
  sample_count: int
  duration_s: float
  metrics: list[LateralTorqueLagMetrics]

  def to_dict(self) -> dict[str, Any]:
    return asdict(self)


@dataclass(frozen=True)
class LateralTorqueABReport:
  baseline: LateralTorqueLagReport
  candidate: LateralTorqueLagReport
  deltas: dict[str, float | None]

  def to_dict(self) -> dict[str, Any]:
    return asdict(self)


@dataclass(frozen=True)
class _TorqueSample:
  t: float
  v_ego: float
  lat_active: bool
  steering_pressed: bool
  blinker_active: bool
  lane_change_state: str
  steering_angle_deg: float
  steering_rate_deg: float
  output: float
  unshaped_output: float
  applied_torque: float
  desired_lateral_accel: float
  actual_lateral_accel: float
  torque_version: int
  shaping_active: bool
  shaping_reason: int
  governor_reason: int
  release_active: bool
  steer_limited: bool
  output_cap: float
  steer_limit_error: float
  learner_confidence: float
  current_curvature: float
  desired_curvature: float
  raw_desired_curvature: float
  processed_desired_curvature: float
  model_path_quality: float
  model_path_gated: bool
  model_path_reason: str


def build_lateral_torque_event_report(
  msgs: list[Any],
  source: str = "unknown",
  already_sorted: bool = False,
  max_events: int = 12,
) -> LateralTorqueEventReport:
  ordered_msgs = list(msgs) if already_sorted else sorted(msgs, key=lambda m: int(getattr(m, "logMonoTime", 0)))
  samples = _extract_torque_samples(ordered_msgs)
  if not samples:
    return LateralTorqueEventReport(source, 0, 0.0, 0.0, 0, [])
  cols = _columns(samples)
  events = _select_events(_fast_torque_events(cols), max_events)
  return LateralTorqueEventReport(
    source=source,
    sample_count=len(samples),
    duration_s=float(cols["t"][-1] - cols["t"][0]) if len(samples) > 1 else 0.0,
    active_percent=_percent(cols["lat_active"] > 0.5),
    fast_torque_event_count=len(events),
    top_events=events,
  )


def render_lateral_torque_event_report(report: LateralTorqueEventReport) -> str:
  lines = [
    f"Lateral torque event report: {report.source}",
    f"samples: {report.sample_count}",
    f"duration: {report.duration_s:.1f} s",
    f"active: {report.active_percent:.1f}%",
    f"fast torque events: {report.fast_torque_event_count}",
  ]
  if report.top_events:
    lines.append("Top events:")
    for event in report.top_events:
      shaping_reasons = ",".join(f"{name}:{count}" for name, count in sorted(event.shaping_reason_counts.items())) or "none"
      governor_reasons = ",".join(f"{name}:{count}" for name, count in sorted(event.governor_reason_counts.items())) or "none"
      lines.append(
        f"  {event.start_s:.1f}-{event.end_s:.1f}s source={event.likely_source} score={event.score:.2f} "
        f"steer_pp={event.steering_angle_pp:.3f}deg rate95={event.steering_rate_p95:.3f}deg/s "
        f"out_pp={event.output_pp:.3f} unshaped_pp={event.unshaped_output_pp:.3f} "
        f"out_flips={event.output_reversals} unshaped_flips={event.unshaped_output_reversals} "
        f"shaping={event.shaping_active_percent:.1f}% limited={event.steer_limited_percent:.1f}% "
        f"shaper_reasons={shaping_reasons} governor_reasons={governor_reasons}"
      )
  return "\n".join(lines)


def save_lateral_torque_event_report(report: LateralTorqueEventReport, path: str | Path) -> None:
  Path(path).write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")


def load_lateral_torque_event_report(path: str | Path) -> LateralTorqueEventReport:
  return LateralTorqueEventReport.from_dict(json.loads(Path(path).read_text()))


def build_lateral_torque_lag_report(
  msgs: list[Any],
  source: str = "unknown",
  already_sorted: bool = False,
) -> LateralTorqueLagReport:
  ordered_msgs = list(msgs) if already_sorted else sorted(msgs, key=lambda m: int(getattr(m, "logMonoTime", 0)))
  samples = _extract_torque_samples(ordered_msgs)
  if not samples:
    return LateralTorqueLagReport(source, 0, 0.0, [])
  cols = _columns(samples)
  base = _base_mask(cols)
  desired_rate = _derivative(cols["t"], cols["desired_lateral_accel"])
  desired_magnitude_rate = np.sign(cols["desired_lateral_accel"]) * desired_rate
  processed_curvature = _finite_or_fallback(cols["processed_desired_curvature"], cols["desired_curvature"])
  desired_lateral_accel_residual = cols["desired_lateral_accel"] - np.square(cols["v_ego"]) * processed_curvature
  actual_lateral_accel_residual = cols["actual_lateral_accel"] - np.square(cols["v_ego"]) * cols["current_curvature"]
  curve = np.abs(cols["desired_lateral_accel"]) > 0.08
  masks = {
    "curve": base & curve,
    "entry": base & curve & (desired_magnitude_rate > 0.15),
    "exit": base & curve & (desired_magnitude_rate < -0.15),
    "cold": base & curve & (cols["learner_confidence"] < 0.3),
    "warm": base & curve & (cols["learner_confidence"] >= 0.6),
    "high_curvature": base & (np.abs(processed_curvature) >= 0.0015),
    "low_path_quality": base & ((cols["model_path_gated"] > 0.5) | ((cols["model_path_quality"] < 0.7) & np.isfinite(cols["model_path_quality"]))),
    "steering_limited": base & (cols["steer_limited"] > 0.5),
  }
  return LateralTorqueLagReport(
    source=source,
    sample_count=len(samples),
    duration_s=float(cols["t"][-1] - cols["t"][0]) if len(samples) > 1 else 0.0,
    metrics=[_lag_metrics(cols, name, mask, desired_rate, desired_lateral_accel_residual, actual_lateral_accel_residual) for name, mask in masks.items()],
  )


def build_lateral_low_speed_report(
  msgs: list[Any],
  source: str = "unknown",
  already_sorted: bool = False,
) -> LateralLowSpeedReport:
  ordered_msgs = list(msgs) if already_sorted else sorted(msgs, key=lambda m: int(getattr(m, "logMonoTime", 0)))
  samples = _extract_torque_samples(ordered_msgs)
  if not samples:
    return LateralLowSpeedReport(source, 0, 0.0, 0, {}, {}, [], [])
  cols = _columns(samples)
  base = _low_speed_primary_mask(cols)
  signal_tagged = _low_speed_signal_tagged_mask(cols)
  turn = _low_speed_turn_mask(cols)
  tiers = [
    _low_speed_tier_metrics(cols, _tier_label(lower, upper), lower, upper, base & turn & _speed_tier_mask(cols, lower, upper))
    for lower, upper in LOW_SPEED_TIER_BOUNDS
  ]
  signal_tagged_tiers = [
    _low_speed_tier_metrics(cols, _tier_label(lower, upper), lower, upper, signal_tagged & turn & _speed_tier_mask(cols, lower, upper))
    for lower, upper in LOW_SPEED_TIER_BOUNDS
  ]
  return LateralLowSpeedReport(
    source=source,
    sample_count=len(samples),
    duration_s=float(cols["t"][-1] - cols["t"][0]) if len(samples) > 1 else 0.0,
    lane_change_excluded_count=_lane_change_excluded_count(cols),
    signal_tagged_category_counts=_signal_tagged_category_counts(cols, turn),
    signal_tagged_state_counts=_signal_tagged_state_counts(cols, turn),
    tiers=tiers,
    signal_tagged_tiers=signal_tagged_tiers,
  )


def build_lateral_torque_ab_report(
  baseline_msgs: list[Any],
  candidate_msgs: list[Any],
  baseline_source: str = "baseline",
  candidate_source: str = "candidate",
  already_sorted: bool = False,
) -> LateralTorqueABReport:
  baseline = build_lateral_torque_lag_report(baseline_msgs, baseline_source, already_sorted)
  candidate = build_lateral_torque_lag_report(candidate_msgs, candidate_source, already_sorted)
  return LateralTorqueABReport(baseline, candidate, _lag_deltas(baseline, candidate))


def render_lateral_torque_lag_report(report: LateralTorqueLagReport) -> str:
  lines = [
    f"Lateral torque lag report: {report.source}",
    f"samples: {report.sample_count}",
    f"duration: {report.duration_s:.1f} s",
  ]
  for metric in report.metrics:
    lag = "n/a" if metric.best_lag_s is None else f"{metric.best_lag_s:.3f}s"
    corr = "n/a" if metric.desired_actual_corr is None else f"{metric.desired_actual_corr:.3f}"
    reasons = _format_counts(metric.model_path_reason_counts)
    shaper = _format_counts(metric.shaping_reason_counts)
    governor = _format_counts(metric.governor_reason_counts)
    lines.append(
      f"{metric.segment}: samples={metric.sample_count} lag={lag} corr={corr} "
      f"err_mean={metric.abs_error_mean:.3f} err95={metric.abs_error_p95:.3f} "
      f"rate95={metric.steering_rate_p95:.2f} jerk95={metric.desired_lateral_jerk_p95:.2f}/{metric.actual_lateral_jerk_p95:.2f} "
      f"aresid95={metric.desired_lateral_accel_residual_abs_p95:.4f}/{metric.actual_lateral_accel_residual_abs_p95:.4f} "
      f"out_flips={metric.output_reversals} limited={metric.steer_limited_percent:.1f}% "
      f"high_rate={metric.high_steering_rate_percent:.1f}% path_low={metric.model_path_low_quality_percent:.1f}% "
      f"path_reasons={reasons} shaper_reasons={shaper} governor_reasons={governor}"
    )
  return "\n".join(lines)


def render_lateral_low_speed_report(report: LateralLowSpeedReport) -> str:
  lines = [
    f"Low-speed lateral report: {report.source}",
    f"samples: {report.sample_count}",
    f"duration: {report.duration_s:.1f} s",
    f"lane-change excluded samples: {report.lane_change_excluded_count}",
    f"signal-tagged categories: {_format_counts(report.signal_tagged_category_counts)}",
    f"signal-tagged states: {_format_counts(report.signal_tagged_state_counts)}",
    "Primary tiers:",
  ]
  for metric in report.tiers:
    lines.append(_render_low_speed_tier_metric(metric))
  lines.append("Signal-tagged tiers:")
  for metric in report.signal_tagged_tiers:
    lines.append(_render_low_speed_tier_metric(metric))
  return "\n".join(lines)


def _render_low_speed_tier_metric(metric: LateralLowSpeedTierMetrics) -> str:
  lag = "n/a" if metric.best_lag_s is None else f"{metric.best_lag_s:.3f}s"
  corr = "n/a" if metric.desired_actual_corr is None else f"{metric.desired_actual_corr:.3f}"
  reasons = ",".join(f"{name}:{count}" for name, count in sorted(metric.model_path_reason_counts.items())) or "none"
  return (
    f"{metric.segment}: samples={metric.sample_count} lag={lag} corr={corr} "
    f"err_mean={metric.abs_error_mean:.3f} err95={metric.abs_error_p95:.3f} "
    f"out_flips={metric.output_reversals} unshaped_flips={metric.unshaped_output_reversals} "
    f"desired_flips={metric.desired_lateral_accel_reversals} actual_flips={metric.actual_lateral_accel_reversals} "
    f"rate95={metric.steering_rate_p95:.2f} limited={metric.steer_limited_percent:.1f}% "
    f"high_rate={metric.high_steering_rate_percent:.1f}% path_gated={metric.model_path_gated_percent:.1f}% "
    f"path_quality={metric.model_path_quality_median:.2f} raw_proc_k95={metric.raw_processed_curvature_delta_p95:.5f} "
    f"desired_proc_k95={metric.desired_processed_curvature_delta_p95:.5f} reasons={reasons}"
  )


def render_lateral_torque_ab_report(report: LateralTorqueABReport) -> str:
  lines = [
    "Lateral torque A/B report",
    render_lateral_torque_lag_report(report.baseline),
    render_lateral_torque_lag_report(report.candidate),
    "Deltas candidate-baseline:",
  ]
  for key, value in sorted(report.deltas.items()):
    rendered = "n/a" if value is None else f"{value:.3f}"
    lines.append(f"{key}: {rendered}")
  return "\n".join(lines)


def _extract_torque_samples(msgs: list[Any]) -> list[_TorqueSample]:
  if not msgs:
    return []
  base_mono_time = int(getattr(msgs[0], "logMonoTime", 0))
  latest: dict[str, Any] = {}
  samples: list[_TorqueSample] = []
  for msg in msgs:
    typ = msg_type(msg)
    payload = msg_payload(msg)
    if typ in ("carState", "carControl", "carOutput", "modelV2"):
      latest[typ] = payload
    if typ != "controlsState":
      continue

    car_state = latest.get("carState")
    car_control = latest.get("carControl")
    car_output = latest.get("carOutput")
    model_v2 = latest.get("modelV2")
    lateral_state = safe_get(payload, "lateralControlState")
    lateral_payload = safe_get(lateral_state, format_enum(lateral_state.which()) if lateral_state is not None and hasattr(lateral_state, "which") else "torqueState", lateral_state)
    adaptive = safe_get(lateral_payload, "adaptiveTorqueState")
    model_path_state = safe_get(payload, "modelPathState")
    torque_version = int(safe_get(lateral_payload, "version", 0) or 0)
    shaping_reason = int(safe_get(adaptive, "shapingReason", 0) or 0)
    governor_reason = int(safe_get(adaptive, "governorReason", 0) or 0)
    if torque_version == 3:
      governor_reason = governor_reason or shaping_reason
      shaping_reason = 0
    samples.append(_TorqueSample(
      t=msg_time_s(msg, base_mono_time),
      v_ego=_finite_float(safe_get(car_state, "vEgo")),
      lat_active=bool(safe_get(lateral_payload, "active", False)) and bool(safe_get(car_control, "latActive", False)),
      steering_pressed=bool(safe_get(car_state, "steeringPressed", False)),
      blinker_active=bool(safe_get(car_state, "leftBlinker", False)) or bool(safe_get(car_state, "rightBlinker", False)),
      lane_change_state=format_enum(safe_get(model_v2, "meta.laneChangeState")),
      steering_angle_deg=_finite_float(safe_get(car_state, "steeringAngleDeg")),
      steering_rate_deg=_finite_float(safe_get(car_state, "steeringRateDeg")),
      output=_finite_float(safe_get(lateral_payload, "output")),
      unshaped_output=_finite_float(safe_get(adaptive, "unshapedOutput")),
      applied_torque=_finite_float(safe_get(car_output, "actuatorsOutput.torque")),
      desired_lateral_accel=_finite_float(safe_get(lateral_payload, "desiredLateralAccel")),
      actual_lateral_accel=_finite_float(safe_get(lateral_payload, "actualLateralAccel")),
      torque_version=torque_version,
      shaping_active=bool(safe_get(adaptive, "shapingActive", False)),
      shaping_reason=shaping_reason,
      governor_reason=governor_reason,
      release_active=bool(safe_get(adaptive, "releaseActive", False)),
      steer_limited=bool(safe_get(adaptive, "steerLimitLimited", False)),
      output_cap=_finite_float(safe_get(adaptive, "outputCap"), 1.0),
      steer_limit_error=_finite_float(safe_get(adaptive, "steerLimitError")),
      learner_confidence=_finite_float(safe_get(adaptive, "modelConfidence")),
      current_curvature=_finite_float(safe_get(payload, "curvature"), float("nan")),
      desired_curvature=_finite_float(safe_get(payload, "desiredCurvature"), float("nan")),
      raw_desired_curvature=_finite_float(safe_get(model_path_state, "rawDesiredCurvature"), float("nan")),
      processed_desired_curvature=_finite_float(safe_get(payload, "desiredCurvature"), float("nan")),
      model_path_quality=_finite_float(safe_get(model_path_state, "quality"), float("nan")),
      model_path_gated=bool(safe_get(model_path_state, "gated", False)),
      model_path_reason=format_enum(safe_get(model_path_state, "reason")),
    ))
  return samples


def _columns(samples: list[_TorqueSample]) -> dict[str, np.ndarray]:
  return {
    "t": np.array([sample.t for sample in samples], dtype=float),
    "v_ego": np.array([sample.v_ego for sample in samples], dtype=float),
    "lat_active": np.array([float(sample.lat_active) for sample in samples], dtype=float),
    "steering_pressed": np.array([float(sample.steering_pressed) for sample in samples], dtype=float),
    "blinker_active": np.array([float(sample.blinker_active) for sample in samples], dtype=float),
    "lane_change_state": np.array([sample.lane_change_state for sample in samples], dtype=object),
    "lane_change_active": np.array([float(_lane_change_state_active(sample.lane_change_state)) for sample in samples], dtype=float),
    "steering_angle_deg": np.array([sample.steering_angle_deg for sample in samples], dtype=float),
    "steering_rate_deg": np.array([sample.steering_rate_deg for sample in samples], dtype=float),
    "output": np.array([sample.output for sample in samples], dtype=float),
    "unshaped_output": np.array([sample.unshaped_output for sample in samples], dtype=float),
    "applied_torque": np.array([sample.applied_torque for sample in samples], dtype=float),
    "desired_lateral_accel": np.array([sample.desired_lateral_accel for sample in samples], dtype=float),
    "actual_lateral_accel": np.array([sample.actual_lateral_accel for sample in samples], dtype=float),
    "torque_version": np.array([sample.torque_version for sample in samples], dtype=int),
    "shaping_active": np.array([float(sample.shaping_active) for sample in samples], dtype=float),
    "shaping_reason": np.array([sample.shaping_reason for sample in samples], dtype=int),
    "governor_reason": np.array([sample.governor_reason for sample in samples], dtype=int),
    "release_active": np.array([float(sample.release_active) for sample in samples], dtype=float),
    "steer_limited": np.array([float(sample.steer_limited) for sample in samples], dtype=float),
    "output_cap": np.array([sample.output_cap for sample in samples], dtype=float),
    "steer_limit_error": np.array([sample.steer_limit_error for sample in samples], dtype=float),
    "learner_confidence": np.array([sample.learner_confidence for sample in samples], dtype=float),
    "current_curvature": np.array([sample.current_curvature for sample in samples], dtype=float),
    "desired_curvature": np.array([sample.desired_curvature for sample in samples], dtype=float),
    "raw_desired_curvature": np.array([sample.raw_desired_curvature for sample in samples], dtype=float),
    "processed_desired_curvature": np.array([sample.processed_desired_curvature for sample in samples], dtype=float),
    "model_path_quality": np.array([sample.model_path_quality for sample in samples], dtype=float),
    "model_path_gated": np.array([float(sample.model_path_gated) for sample in samples], dtype=float),
    "model_path_reason": np.array([sample.model_path_reason for sample in samples], dtype=object),
  }


def _fast_torque_events(cols: dict[str, np.ndarray]) -> list[LateralTorqueEvent]:
  base = _base_mask(cols)
  steering_rate_from_angle = _derivative(cols["t"], cols["steering_angle_deg"])
  events: list[LateralTorqueEvent] = []
  for mask in _window_masks(cols["t"], 4.0, 0.5):
    idx = mask & base
    if int(np.sum(idx)) < 20:
      continue
    output_reversals = _sign_flip_count(cols["output"][idx] - _median(cols["output"][idx]), 0.01)
    unshaped_reversals = _sign_flip_count(cols["unshaped_output"][idx] - _median(cols["unshaped_output"][idx]), 0.01)
    applied_reversals = _sign_flip_count(cols["applied_torque"][idx] - _median(cols["applied_torque"][idx]), 0.01)
    steering_rate_reversals = _sign_flip_count(steering_rate_from_angle[idx], 4.0)
    desired_reversals = _sign_flip_count(cols["desired_lateral_accel"][idx] - _median(cols["desired_lateral_accel"][idx]), 0.03)
    actual_reversals = _sign_flip_count(cols["actual_lateral_accel"][idx] - _median(cols["actual_lateral_accel"][idx]), 0.03)
    steering_rate_p95 = _p95_abs(steering_rate_from_angle[idx])
    steering_pp = _percentile_span(cols["steering_angle_deg"][idx])
    output_pp = _percentile_span(cols["output"][idx])
    if max(output_reversals, unshaped_reversals, steering_rate_reversals) < 6:
      continue
    if steering_pp < 0.5 and steering_rate_p95 < 8.0 and output_pp < 0.20:
      continue
    score = (
      steering_rate_reversals * 2.0
      + output_reversals * 1.4
      + unshaped_reversals * 1.0
      + applied_reversals * 0.7
      + steering_rate_p95 * 0.35
      + output_pp * 8.0
    )
    events.append(_make_event(cols, idx, score, output_reversals, unshaped_reversals, applied_reversals,
                              steering_rate_reversals, desired_reversals, actual_reversals))
  return events


def _make_event(cols: dict[str, np.ndarray], idx: np.ndarray, score: float, output_reversals: int,
                unshaped_reversals: int, applied_reversals: int, steering_rate_reversals: int,
                desired_reversals: int, actual_reversals: int) -> LateralTorqueEvent:
  shaping_reason_counts = _reason_counts(cols["shaping_reason"][idx], SHAPING_REASON_NAMES)
  governor_reason_counts = _governor_reason_counts(cols["governor_reason"][idx], cols["torque_version"][idx])
  shaping_active_percent = _percent(cols["shaping_active"][idx] > 0.5)
  steer_limited_percent = _percent(cols["steer_limited"][idx] > 0.5)
  return LateralTorqueEvent(
    start_s=float(np.min(cols["t"][idx])),
    end_s=float(np.max(cols["t"][idx])),
    score=float(score),
    sample_count=int(np.sum(idx)),
    likely_source=_likely_source(output_reversals, unshaped_reversals, desired_reversals, shaping_active_percent, steer_limited_percent),
    speed_mps_median=_median(cols["v_ego"][idx]),
    steering_angle_pp=_percentile_span(cols["steering_angle_deg"][idx]),
    steering_rate_p95=_p95_abs(_derivative(cols["t"], cols["steering_angle_deg"])[idx]),
    steering_rate_reversals=steering_rate_reversals,
    output_pp=_percentile_span(cols["output"][idx]),
    unshaped_output_pp=_percentile_span(cols["unshaped_output"][idx]),
    applied_torque_pp=_percentile_span(cols["applied_torque"][idx]),
    output_reversals=output_reversals,
    unshaped_output_reversals=unshaped_reversals,
    applied_torque_reversals=applied_reversals,
    desired_lateral_accel_pp=_percentile_span(cols["desired_lateral_accel"][idx]),
    actual_lateral_accel_pp=_percentile_span(cols["actual_lateral_accel"][idx]),
    desired_lateral_accel_reversals=desired_reversals,
    actual_lateral_accel_reversals=actual_reversals,
    desired_actual_corr=_correlation(cols["desired_lateral_accel"][idx], cols["actual_lateral_accel"][idx]),
    shaping_active_percent=shaping_active_percent,
    release_active_percent=_percent(cols["release_active"][idx] > 0.5),
    steer_limited_percent=steer_limited_percent,
    output_cap_median=_median(cols["output_cap"][idx]),
    steer_limit_error_pp=_percentile_span(cols["steer_limit_error"][idx]),
    shaping_reason_counts=shaping_reason_counts,
    governor_reason_counts=governor_reason_counts,
  )


def _select_events(events: list[LateralTorqueEvent], max_events: int) -> list[LateralTorqueEvent]:
  selected: list[LateralTorqueEvent] = []
  for event in sorted(events, key=lambda item: item.score, reverse=True):
    if all(abs(event.start_s - seen.start_s) >= 2.0 for seen in selected):
      selected.append(event)
    if len(selected) >= max_events:
      break
  return selected


def _window_masks(t: np.ndarray, window_s: float, step_s: float) -> list[np.ndarray]:
  masks: list[np.ndarray] = []
  if t.size == 0:
    return masks
  cur = float(t[0])
  end = float(t[-1])
  while cur + window_s <= end:
    masks.append((t >= cur) & (t < cur + window_s))
    cur += step_s
  return masks


def _base_mask(cols: dict[str, np.ndarray]) -> np.ndarray:
  return (
    (cols["lat_active"] > 0.5)
    & (cols["v_ego"] > 8.0)
    & (cols["steering_pressed"] < 0.5)
    & (cols["blinker_active"] < 0.5)
    & (cols["lane_change_active"] < 0.5)
    & np.isfinite(cols["output"])
    & np.isfinite(cols["unshaped_output"])
    & np.isfinite(cols["steering_angle_deg"])
  )


def _low_speed_primary_mask(cols: dict[str, np.ndarray]) -> np.ndarray:
  return _low_speed_common_mask(cols) & (cols["blinker_active"] < 0.5) & (cols["lane_change_active"] < 0.5)


def _low_speed_signal_tagged_mask(cols: dict[str, np.ndarray]) -> np.ndarray:
  signal_tagged = (cols["blinker_active"] > 0.5) | (cols["lane_change_active"] > 0.5)
  return _low_speed_common_mask(cols) & signal_tagged


def _low_speed_common_mask(cols: dict[str, np.ndarray]) -> np.ndarray:
  return (
    (cols["lat_active"] > 0.5)
    & (cols["v_ego"] >= 0.0)
    & (cols["v_ego"] < LOW_SPEED_REPORT_MAX_SPEED)
    & (cols["steering_pressed"] < 0.5)
    & np.isfinite(cols["output"])
    & np.isfinite(cols["unshaped_output"])
    & np.isfinite(cols["desired_lateral_accel"])
    & np.isfinite(cols["actual_lateral_accel"])
  )


def _low_speed_turn_mask(cols: dict[str, np.ndarray]) -> np.ndarray:
  processed_curvature = _finite_or_fallback(cols["processed_desired_curvature"], cols["desired_curvature"])
  return (
    (np.abs(cols["desired_lateral_accel"]) >= LOW_SPEED_TURN_MIN_LAT_ACCEL)
    | (np.abs(processed_curvature) >= LOW_SPEED_TURN_MIN_CURVATURE)
    | (np.abs(cols["steering_angle_deg"]) >= LOW_SPEED_TURN_MIN_STEERING_ANGLE_DEG)
  )


def _speed_tier_mask(cols: dict[str, np.ndarray], lower: float, upper: float) -> np.ndarray:
  return (cols["v_ego"] >= lower) & (cols["v_ego"] < upper)


def _lane_change_excluded_count(cols: dict[str, np.ndarray]) -> int:
  low_speed_active = (cols["lat_active"] > 0.5) & (cols["v_ego"] >= 0.0) & (cols["v_ego"] < LOW_SPEED_REPORT_MAX_SPEED)
  lane_change = (cols["blinker_active"] > 0.5) | (cols["lane_change_active"] > 0.5)
  return int(np.sum(low_speed_active & lane_change))


def _signal_tagged_category_counts(cols: dict[str, np.ndarray], turn: np.ndarray) -> dict[str, int]:
  signal_tagged = _low_speed_signal_tagged_mask(cols) & turn
  blinker = cols["blinker_active"] > 0.5
  lane_change = cols["lane_change_active"] > 0.5
  return {
    "blinker_only": int(np.sum(signal_tagged & blinker & ~lane_change)),
    "lane_change_state_only": int(np.sum(signal_tagged & ~blinker & lane_change)),
    "both": int(np.sum(signal_tagged & blinker & lane_change)),
  }


def _signal_tagged_state_counts(cols: dict[str, np.ndarray], turn: np.ndarray) -> dict[str, int]:
  signal_tagged = _low_speed_signal_tagged_mask(cols) & turn
  return _string_counts(cols["lane_change_state"][signal_tagged])


def _tier_label(lower: float, upper: float) -> str:
  return f"{lower:g}-{upper:g}mps"


def _lane_change_state_active(state: str) -> bool:
  return state not in ("off", "unknown")


def _low_speed_tier_metrics(cols: dict[str, np.ndarray], segment: str, lower: float, upper: float,
                            idx: np.ndarray) -> LateralLowSpeedTierMetrics:
  sample_count = int(np.sum(idx))
  if sample_count == 0:
    return LateralLowSpeedTierMetrics(segment, lower, upper, 0, None, None, 0.0, 0.0, 0, 0, 0, 0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, {})

  desired = cols["desired_lateral_accel"][idx]
  actual = cols["actual_lateral_accel"][idx]
  error = desired - actual
  raw_processed_curvature_delta = cols["raw_desired_curvature"][idx] - cols["processed_desired_curvature"][idx]
  desired_processed_curvature_delta = cols["desired_curvature"][idx] - cols["processed_desired_curvature"][idx]
  return LateralLowSpeedTierMetrics(
    segment=segment,
    speed_lower_mps=lower,
    speed_upper_mps=upper,
    sample_count=sample_count,
    best_lag_s=_best_lag_s(cols["t"][idx], desired, actual),
    desired_actual_corr=_correlation(desired, actual),
    abs_error_mean=float(np.nanmean(np.abs(error[np.isfinite(error)]))) if np.any(np.isfinite(error)) else 0.0,
    abs_error_p95=_p95_abs(error),
    output_reversals=_sign_flip_count(cols["output"][idx] - _median(cols["output"][idx]), 0.01),
    unshaped_output_reversals=_sign_flip_count(cols["unshaped_output"][idx] - _median(cols["unshaped_output"][idx]), 0.01),
    desired_lateral_accel_reversals=_sign_flip_count(desired - _median(desired), 0.03),
    actual_lateral_accel_reversals=_sign_flip_count(actual - _median(actual), 0.03),
    steering_rate_p95=_p95_abs(cols["steering_rate_deg"][idx]),
    steer_limited_percent=_percent(cols["steer_limited"][idx] > 0.5),
    high_steering_rate_percent=_percent(np.abs(cols["steering_rate_deg"][idx]) >= 80.0),
    raw_processed_curvature_delta_p95=_p95_abs(raw_processed_curvature_delta),
    desired_processed_curvature_delta_p95=_p95_abs(desired_processed_curvature_delta),
    model_path_gated_percent=_percent(cols["model_path_gated"][idx] > 0.5),
    model_path_quality_median=_median(cols["model_path_quality"][idx]),
    model_path_reason_counts=_string_counts(cols["model_path_reason"][idx]),
  )


def _reason_counts(reasons: np.ndarray, reason_names: dict[int, str]) -> dict[str, int]:
  counts: dict[str, int] = {}
  for reason in reasons.astype(int):
    for bit, name in reason_names.items():
      if reason & bit:
        counts[name] = counts.get(name, 0) + 1
  return counts


def _governor_reason_counts(reasons: np.ndarray, versions: np.ndarray) -> dict[str, int]:
  counts: dict[str, int] = {}
  for reason, version in zip(reasons.astype(int), versions.astype(int), strict=False):
    if version == 3:
      names = V3_GOVERNOR_REASON_NAMES
    elif version == 4:
      names = V4_GOVERNOR_REASON_NAMES
    elif version == 21:
      names = V21_GOVERNOR_REASON_NAMES
    else:
      names = GOVERNOR_REASON_NAMES
    for bit, name in names.items():
      if reason & bit:
        counts[name] = counts.get(name, 0) + 1
  return counts


def _string_counts(values: np.ndarray) -> dict[str, int]:
  counts: dict[str, int] = {}
  for value in values:
    key = str(value) if str(value) else "unknown"
    counts[key] = counts.get(key, 0) + 1
  return counts


def _format_counts(counts: dict[str, int]) -> str:
  return ",".join(f"{name}:{count}" for name, count in sorted(counts.items())) or "none"


def _finite_or_fallback(values: np.ndarray, fallback: np.ndarray) -> np.ndarray:
  return np.where(np.isfinite(values), values, fallback)


def _likely_source(output_reversals: int, unshaped_reversals: int, desired_reversals: int,
                   shaping_active_percent: float, steer_limited_percent: float) -> str:
  if shaping_active_percent >= 45.0 or steer_limited_percent >= 60.0:
    return "safety_shaping_or_actuator_limit"
  if output_reversals >= 6 and unshaped_reversals >= output_reversals - 1 and desired_reversals <= max(2, output_reversals // 3):
    return "controller_unshaped_reversal"
  if desired_reversals >= max(3, output_reversals // 2):
    return "demand_driven"
  return "mixed"


def _event_from_dict(data: dict[str, Any]) -> LateralTorqueEvent:
  raw_corr = data.get("desired_actual_corr", data.get("desiredActualCorr"))
  return LateralTorqueEvent(
    start_s=float(data.get("start_s", data.get("startS", 0.0))),
    end_s=float(data.get("end_s", data.get("endS", 0.0))),
    score=float(data.get("score", 0.0)),
    sample_count=int(data.get("sample_count", data.get("sampleCount", 0))),
    likely_source=str(data.get("likely_source", data.get("likelySource", "mixed"))),
    speed_mps_median=float(data.get("speed_mps_median", data.get("speedMpsMedian", 0.0))),
    steering_angle_pp=float(data.get("steering_angle_pp", data.get("steeringAnglePp", 0.0))),
    steering_rate_p95=float(data.get("steering_rate_p95", data.get("steeringRateP95", 0.0))),
    steering_rate_reversals=int(data.get("steering_rate_reversals", data.get("steeringRateReversals", 0))),
    output_pp=float(data.get("output_pp", data.get("outputPp", 0.0))),
    unshaped_output_pp=float(data.get("unshaped_output_pp", data.get("unshapedOutputPp", 0.0))),
    applied_torque_pp=float(data.get("applied_torque_pp", data.get("appliedTorquePp", 0.0))),
    output_reversals=int(data.get("output_reversals", data.get("outputReversals", 0))),
    unshaped_output_reversals=int(data.get("unshaped_output_reversals", data.get("unshapedOutputReversals", 0))),
    applied_torque_reversals=int(data.get("applied_torque_reversals", data.get("appliedTorqueReversals", 0))),
    desired_lateral_accel_pp=float(data.get("desired_lateral_accel_pp", data.get("desiredLateralAccelPp", 0.0))),
    actual_lateral_accel_pp=float(data.get("actual_lateral_accel_pp", data.get("actualLateralAccelPp", 0.0))),
    desired_lateral_accel_reversals=int(data.get("desired_lateral_accel_reversals", data.get("desiredLateralAccelReversals", 0))),
    actual_lateral_accel_reversals=int(data.get("actual_lateral_accel_reversals", data.get("actualLateralAccelReversals", 0))),
    desired_actual_corr=None if raw_corr is None else float(raw_corr),
    shaping_active_percent=float(data.get("shaping_active_percent", data.get("shapingActivePercent", 0.0))),
    release_active_percent=float(data.get("release_active_percent", data.get("releaseActivePercent", 0.0))),
    steer_limited_percent=float(data.get("steer_limited_percent", data.get("steerLimitedPercent", 0.0))),
    output_cap_median=float(data.get("output_cap_median", data.get("outputCapMedian", 0.0))),
    steer_limit_error_pp=float(data.get("steer_limit_error_pp", data.get("steerLimitErrorPp", 0.0))),
    shaping_reason_counts={str(k): int(v) for k, v in data.get("shaping_reason_counts", data.get("shapingReasonCounts", {})).items()},
    governor_reason_counts={str(k): int(v) for k, v in data.get("governor_reason_counts", data.get("governorReasonCounts", {})).items()},
  )


def _lag_metrics(
  cols: dict[str, np.ndarray],
  segment: str,
  idx: np.ndarray,
  desired_rate: np.ndarray,
  desired_lateral_accel_residual: np.ndarray,
  actual_lateral_accel_residual: np.ndarray,
) -> LateralTorqueLagMetrics:
  if int(np.sum(idx)) < 8:
    return LateralTorqueLagMetrics(segment, int(np.sum(idx)), None, None, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, {}, {}, {})
  desired = cols["desired_lateral_accel"][idx]
  actual = cols["actual_lateral_accel"][idx]
  error = desired - actual
  return LateralTorqueLagMetrics(
    segment=segment,
    sample_count=int(np.sum(idx)),
    best_lag_s=_best_lag_s(cols["t"][idx], desired, actual),
    desired_actual_corr=_correlation(desired, actual),
    abs_error_mean=float(np.nanmean(np.abs(error[np.isfinite(error)]))) if np.any(np.isfinite(error)) else 0.0,
    abs_error_p95=_p95_abs(error),
    steering_rate_p95=_p95_abs(cols["steering_rate_deg"][idx]),
    output_reversals=_sign_flip_count(cols["output"][idx] - _median(cols["output"][idx]), 0.01),
    steer_limited_percent=_percent(cols["steer_limited"][idx] > 0.5),
    high_steering_rate_percent=_percent(np.abs(cols["steering_rate_deg"][idx]) >= 80.0),
    desired_lateral_jerk_p95=_p95_abs(desired_rate[idx]),
    actual_lateral_jerk_p95=_p95_abs(_derivative(cols["t"][idx], actual)),
    desired_lateral_accel_residual_abs_p95=_p95_abs(desired_lateral_accel_residual[idx]),
    actual_lateral_accel_residual_abs_p95=_p95_abs(actual_lateral_accel_residual[idx]),
    model_path_low_quality_percent=_percent((cols["model_path_gated"][idx] > 0.5) | ((cols["model_path_quality"][idx] < 0.7) & np.isfinite(cols["model_path_quality"][idx]))),
    model_path_reason_counts=_string_counts(cols["model_path_reason"][idx]),
    shaping_reason_counts=_reason_counts(cols["shaping_reason"][idx], SHAPING_REASON_NAMES),
    governor_reason_counts=_governor_reason_counts(cols["governor_reason"][idx], cols["torque_version"][idx]),
  )


def _lag_deltas(baseline: LateralTorqueLagReport, candidate: LateralTorqueLagReport) -> dict[str, float | None]:
  baseline_by_segment = {metric.segment: metric for metric in baseline.metrics}
  deltas: dict[str, float | None] = {}
  for candidate_metric in candidate.metrics:
    baseline_metric = baseline_by_segment.get(candidate_metric.segment)
    if baseline_metric is None:
      continue
    prefix = candidate_metric.segment
    deltas[f"{prefix}.best_lag_s"] = _delta(candidate_metric.best_lag_s, baseline_metric.best_lag_s)
    deltas[f"{prefix}.abs_error_p95"] = candidate_metric.abs_error_p95 - baseline_metric.abs_error_p95
    deltas[f"{prefix}.output_reversals"] = float(candidate_metric.output_reversals - baseline_metric.output_reversals)
    deltas[f"{prefix}.steer_limited_percent"] = candidate_metric.steer_limited_percent - baseline_metric.steer_limited_percent
    deltas[f"{prefix}.high_steering_rate_percent"] = candidate_metric.high_steering_rate_percent - baseline_metric.high_steering_rate_percent
  return deltas


def _delta(candidate: float | None, baseline: float | None) -> float | None:
  if candidate is None or baseline is None:
    return None
  return candidate - baseline


def _best_lag_s(t: np.ndarray, desired: np.ndarray, actual: np.ndarray, max_lag_s: float = 0.6) -> float | None:
  ok = np.isfinite(t) & np.isfinite(desired) & np.isfinite(actual)
  t = t[ok]
  desired = desired[ok]
  actual = actual[ok]
  if t.size < 8 or np.nanstd(desired) < 1e-9 or np.nanstd(actual) < 1e-9:
    return None
  dt = float(np.nanmedian(np.diff(t))) if t.size > 1 else 0.0
  if not np.isfinite(dt) or dt <= 1e-3:
    return None
  max_shift = max(1, int(max_lag_s / dt))
  best_lag = 0.0
  best_corr = -2.0
  for shift in range(-max_shift, max_shift + 1):
    if shift > 0:
      a = desired[:-shift]
      b = actual[shift:]
    elif shift < 0:
      a = desired[-shift:]
      b = actual[:shift]
    else:
      a = desired
      b = actual
    corr = _correlation(a, b)
    if corr is not None and corr > best_corr:
      best_corr = corr
      best_lag = shift * dt
  return float(best_lag) if best_corr > -2.0 else None


def _derivative(t: np.ndarray, y: np.ndarray) -> np.ndarray:
  out = np.zeros_like(y)
  dt = np.diff(t)
  dy = np.diff(y)
  valid = np.isfinite(dt) & (dt > 1e-3) & np.isfinite(dy)
  vals = np.zeros_like(dy)
  vals[valid] = dy[valid] / dt[valid]
  out[1:] = vals
  return out


def _sign_flip_count(x: np.ndarray, eps: float) -> int:
  ok = np.isfinite(x) & (np.abs(x) > eps)
  signs = np.sign(x[ok])
  return int(np.sum(signs[1:] != signs[:-1])) if signs.size > 1 else 0


def _percentile_span(a: np.ndarray) -> float:
  a = a[np.isfinite(a)]
  return float(np.nanpercentile(a, 95) - np.nanpercentile(a, 5)) if a.size else 0.0


def _p95_abs(a: np.ndarray) -> float:
  a = a[np.isfinite(a)]
  return float(np.nanpercentile(np.abs(a), 95)) if a.size else 0.0


def _median(a: np.ndarray) -> float:
  a = a[np.isfinite(a)]
  return float(np.nanmedian(a)) if a.size else 0.0


def _percent(mask: np.ndarray) -> float:
  return float(np.mean(mask) * 100.0) if mask.size else 0.0


def _correlation(a: np.ndarray, b: np.ndarray) -> float | None:
  ok = np.isfinite(a) & np.isfinite(b)
  if int(np.sum(ok)) < 5 or np.nanstd(a[ok]) < 1e-9 or np.nanstd(b[ok]) < 1e-9:
    return None
  return float(np.corrcoef(a[ok], b[ok])[0, 1])


def _finite_float(value: Any, default: float = 0.0) -> float:
  try:
    converted = float(value)
  except (TypeError, ValueError):
    return default
  return converted if np.isfinite(converted) else default
