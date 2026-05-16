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
  shaping_active: bool
  shaping_reason: int
  release_active: bool
  steer_limited: bool
  output_cap: float
  steer_limit_error: float


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
      reasons = ",".join(f"{name}:{count}" for name, count in sorted(event.shaping_reason_counts.items())) or "none"
      lines.append(
        f"  {event.start_s:.1f}-{event.end_s:.1f}s source={event.likely_source} score={event.score:.2f} "
        f"steer_pp={event.steering_angle_pp:.3f}deg rate95={event.steering_rate_p95:.3f}deg/s "
        f"out_pp={event.output_pp:.3f} unshaped_pp={event.unshaped_output_pp:.3f} "
        f"out_flips={event.output_reversals} unshaped_flips={event.unshaped_output_reversals} "
        f"shaping={event.shaping_active_percent:.1f}% limited={event.steer_limited_percent:.1f}% "
        f"reasons={reasons}"
      )
  return "\n".join(lines)


def save_lateral_torque_event_report(report: LateralTorqueEventReport, path: str | Path) -> None:
  Path(path).write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")


def load_lateral_torque_event_report(path: str | Path) -> LateralTorqueEventReport:
  return LateralTorqueEventReport.from_dict(json.loads(Path(path).read_text()))


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
      shaping_active=bool(safe_get(adaptive, "shapingActive", False)),
      shaping_reason=int(safe_get(adaptive, "shapingReason", 0) or 0),
      release_active=bool(safe_get(adaptive, "releaseActive", False)),
      steer_limited=bool(safe_get(adaptive, "steerLimitLimited", False)),
      output_cap=_finite_float(safe_get(adaptive, "outputCap"), 1.0),
      steer_limit_error=_finite_float(safe_get(adaptive, "steerLimitError")),
    ))
  return samples


def _columns(samples: list[_TorqueSample]) -> dict[str, np.ndarray]:
  return {
    "t": np.array([sample.t for sample in samples], dtype=float),
    "v_ego": np.array([sample.v_ego for sample in samples], dtype=float),
    "lat_active": np.array([float(sample.lat_active) for sample in samples], dtype=float),
    "steering_pressed": np.array([float(sample.steering_pressed) for sample in samples], dtype=float),
    "blinker_active": np.array([float(sample.blinker_active) for sample in samples], dtype=float),
    "lane_change_off": np.array([float(sample.lane_change_state == "off") for sample in samples], dtype=float),
    "steering_angle_deg": np.array([sample.steering_angle_deg for sample in samples], dtype=float),
    "steering_rate_deg": np.array([sample.steering_rate_deg for sample in samples], dtype=float),
    "output": np.array([sample.output for sample in samples], dtype=float),
    "unshaped_output": np.array([sample.unshaped_output for sample in samples], dtype=float),
    "applied_torque": np.array([sample.applied_torque for sample in samples], dtype=float),
    "desired_lateral_accel": np.array([sample.desired_lateral_accel for sample in samples], dtype=float),
    "actual_lateral_accel": np.array([sample.actual_lateral_accel for sample in samples], dtype=float),
    "shaping_active": np.array([float(sample.shaping_active) for sample in samples], dtype=float),
    "shaping_reason": np.array([sample.shaping_reason for sample in samples], dtype=int),
    "release_active": np.array([float(sample.release_active) for sample in samples], dtype=float),
    "steer_limited": np.array([float(sample.steer_limited) for sample in samples], dtype=float),
    "output_cap": np.array([sample.output_cap for sample in samples], dtype=float),
    "steer_limit_error": np.array([sample.steer_limit_error for sample in samples], dtype=float),
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
  reason_counts = _reason_counts(cols["shaping_reason"][idx])
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
    shaping_reason_counts=reason_counts,
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
    & (cols["lane_change_off"] > 0.5)
    & np.isfinite(cols["output"])
    & np.isfinite(cols["unshaped_output"])
    & np.isfinite(cols["steering_angle_deg"])
  )


def _reason_counts(reasons: np.ndarray) -> dict[str, int]:
  counts: dict[str, int] = {}
  for reason in reasons.astype(int):
    for bit, name in SHAPING_REASON_NAMES.items():
      if reason & bit:
        counts[name] = counts.get(name, 0) + 1
  return counts


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
  )


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
