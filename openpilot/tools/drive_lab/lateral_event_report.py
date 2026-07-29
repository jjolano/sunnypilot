from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from openpilot.tools.drive_lab.lateral_oscillation_profile import _columns, _extract_lateral_samples, _straight_mask


@dataclass(frozen=True)
class LateralEvent:
  kind: str
  start_s: float
  end_s: float
  score: float
  sample_count: int
  speed_mps_median: float
  steering_angle_pp: float
  steering_rate_p95: float
  raw_curvature_pp: float
  desired_curvature_pp: float
  actual_curvature_pp: float
  output_pp: float
  raw_actual_corr: float | None
  gated_percent: float
  quality_median: float
  steering_reversals: int = 0
  raw_reversals: int = 0
  pre_soft_percent: float = 0.0
  pre_attenuation_median: float = 0.0
  requested_torque_pp: float = 0.0
  applied_torque_pp: float = 0.0
  eps_torque_pp: float = 0.0
  driver_torque_p95: float = 0.0
  command_eps_corr: float | None = None
  command_applied_corr: float | None = None


@dataclass(frozen=True)
class LateralEventReport:
  source: str
  sample_count: int
  duration_s: float
  active_percent: float
  slow_wander_count: int
  rebound_count: int
  fast_reversal_count: int
  top_events: list[LateralEvent]

  def to_dict(self) -> dict[str, Any]:
    return asdict(self)

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> LateralEventReport:
    return cls(
      source=str(data.get("source", "unknown")),
      sample_count=int(data.get("sample_count", data.get("sampleCount", 0))),
      duration_s=float(data.get("duration_s", data.get("durationS", 0.0))),
      active_percent=float(data.get("active_percent", data.get("activePercent", 0.0))),
      slow_wander_count=int(data.get("slow_wander_count", data.get("slowWanderCount", 0))),
      rebound_count=int(data.get("rebound_count", data.get("reboundCount", 0))),
      fast_reversal_count=int(data.get("fast_reversal_count", data.get("fastReversalCount", 0))),
      top_events=[_event_from_dict(event) for event in data.get("top_events", data.get("topEvents", ()))],
    )


def build_lateral_event_report(
  msgs: list[Any],
  source: str = "unknown",
  already_sorted: bool = False,
  max_events: int = 15,
) -> LateralEventReport:
  ordered_msgs = list(msgs) if already_sorted else sorted(msgs, key=lambda m: int(getattr(m, "logMonoTime", 0)))
  samples = _extract_lateral_samples(ordered_msgs)
  if not samples:
    return LateralEventReport(source, 0, 0.0, 0.0, 0, 0, 0, [])
  cols = _columns(samples)
  duration_s = float(cols["t"][-1] - cols["t"][0]) if len(samples) > 1 else 0.0
  active = cols["lat_active"] > 0.5
  events = [
    *_slow_wander_events(cols),
    *_rebound_events(cols),
    *_fast_reversal_events(cols),
  ]
  events = _select_events(events, max_events)
  return LateralEventReport(
    source=source,
    sample_count=len(samples),
    duration_s=duration_s,
    active_percent=_percent(active),
    slow_wander_count=sum(1 for event in events if event.kind == "slow_wander"),
    rebound_count=sum(1 for event in events if event.kind == "rebound"),
    fast_reversal_count=sum(1 for event in events if event.kind == "fast_reversal"),
    top_events=events,
  )


def render_lateral_event_report(report: LateralEventReport) -> str:
  lines = [
    f"Lateral event report: {report.source}",
    f"samples: {report.sample_count}",
    f"duration: {report.duration_s:.1f} s",
    f"active: {report.active_percent:.1f}%",
    f"slow wander events: {report.slow_wander_count}",
    f"rebound events: {report.rebound_count}",
    f"fast reversal events: {report.fast_reversal_count}",
  ]
  if report.top_events:
    lines.append("Top events:")
    for event in report.top_events:
      lines.append(
        f"  {event.kind} {event.start_s:.1f}-{event.end_s:.1f}s score={event.score:.2f} "
        f"steer_pp={event.steering_angle_pp:.3f}deg rate95={event.steering_rate_p95:.3f}deg/s "
        f"raw_pp={event.raw_curvature_pp:.6f} actual_pp={event.actual_curvature_pp:.6f} "
        f"cmd_pp={event.requested_torque_pp:.3f} eps_pp={event.eps_torque_pp:.3f} "
        f"driver95={event.driver_torque_p95:.3f} cmd_eps={_format_optional(event.command_eps_corr)} "
        f"gated={event.gated_percent:.1f}% q={event.quality_median:.2f}"
      )
  return "\n".join(lines)


def save_lateral_event_report(report: LateralEventReport, path: str | Path) -> None:
  Path(path).write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True))


def load_lateral_event_report(path: str | Path) -> LateralEventReport:
  return LateralEventReport.from_dict(json.loads(Path(path).read_text()))


def _slow_wander_events(cols: dict[str, np.ndarray]) -> list[LateralEvent]:
  straight = _straight_mask(cols, min_speed=8.0, max_raw_curvature=0.002)
  events: list[LateralEvent] = []
  for mask in _window_masks(cols["t"], 30.0, 5.0):
    if not np.any(mask) or float(np.mean(straight[mask])) < 0.75:
      continue
    idx = mask & straight
    steering_pp = _percentile_span(cols["steering_angle_deg"][idx])
    raw_actual_corr = _correlation(cols["raw_desired_curvature"][idx], cols["curvature"][idx])
    if steering_pp < 5.0 or raw_actual_corr is None or raw_actual_corr < 0.75:
      continue
    raw_pp = _percentile_span(cols["raw_desired_curvature"][idx])
    actual_pp = _percentile_span(cols["curvature"][idx])
    score = steering_pp * 2.0 + raw_actual_corr * 10.0 + actual_pp * 1500.0
    events.append(_make_event("slow_wander", cols, idx, score, raw_actual_corr=raw_actual_corr))
  return events


def _rebound_events(cols: dict[str, np.ndarray]) -> list[LateralEvent]:
  t = cols["t"]
  base = _base_mask(cols)
  low_curve = (np.abs(cols["raw_desired_curvature"]) < 0.0028) & (np.abs(cols["curvature"]) < 0.0040)
  attenuation = _attenuation(cols)
  soft = base & low_curve & (
    ((attenuation > 0.15) & (attenuation < 0.85))
    | (cols["model_path_gated"] > 0.5)
    | (cols["model_path_quality"] < 0.8)
  )
  steering_rate = _derivative(t, cols["steering_angle_deg"])
  events: list[LateralEvent] = []
  start = float(t[0]) + 2.0
  end = float(t[-1]) - 4.0
  cur = start
  while cur <= end:
    pre = (t >= cur - 2.0) & (t < cur) & base & low_curve
    post = (t >= cur + 0.25) & (t < cur + 3.0) & base
    if int(np.sum(pre)) < 20 or int(np.sum(post)) < 20:
      cur += 0.5
      continue
    pre_soft = _percent(soft[pre])
    pre_att = _median(np.maximum(attenuation[pre], 0.0))
    if pre_soft < 35.0 and pre_att < 0.10:
      cur += 0.5
      continue
    post_steer_pp = _percentile_span(cols["steering_angle_deg"][post])
    post_rate = _p95_abs(steering_rate[post])
    post_actual_pp = _percentile_span(cols["curvature"][post])
    if post_steer_pp < 2.0 or post_rate < 1.5 or post_actual_pp < _percentile_span(cols["curvature"][pre]) + 4e-4:
      cur += 0.5
      continue
    score = post_rate * 1.2 + post_steer_pp * 1.4 + post_actual_pp * 3000.0 + pre_soft * 0.05 + pre_att * 10.0
    event = _make_event("rebound", cols, post, score, pre_soft_percent=pre_soft, pre_attenuation_median=pre_att)
    events.append(event)
    cur += 0.5
  return events


def _fast_reversal_events(cols: dict[str, np.ndarray]) -> list[LateralEvent]:
  t = cols["t"]
  base = _base_mask(cols) & (cols["model_path_gated"] < 0.5) & (cols["model_path_quality"] >= 0.95)
  steering_rate = _derivative(t, cols["steering_angle_deg"])
  events: list[LateralEvent] = []
  for mask in _window_masks(t, 4.0, 0.5):
    idx = mask & base
    if int(np.sum(idx)) < 20:
      continue
    steering = cols["steering_angle_deg"][idx]
    raw = cols["raw_desired_curvature"][idx]
    output = cols["lat_output"][idx]
    reversals = _sign_flip_count(steering - _median(steering), 0.08)
    rate_reversals = _sign_flip_count(steering_rate[idx], 4.0)
    raw_reversals = _sign_flip_count(raw - _median(raw), 2e-5)
    output_reversals = _sign_flip_count(output - _median(output), 0.01)
    steering_rate_p95 = _p95_abs(steering_rate[idx])
    steering_pp = _percentile_span(steering)
    if rate_reversals < 6 and reversals < 4:
      continue
    if steering_pp < 0.5 and steering_rate_p95 < 10.0:
      continue
    score = rate_reversals * 2.0 + reversals * 3.0 + raw_reversals * 1.2 + output_reversals * 1.2 + steering_rate_p95 * 0.4
    events.append(_make_event("fast_reversal", cols, idx, score, steering_reversals=reversals, raw_reversals=raw_reversals))
  return events


def _make_event(kind: str, cols: dict[str, np.ndarray], idx: np.ndarray, score: float,
                raw_actual_corr: float | None = None, steering_reversals: int = 0, raw_reversals: int = 0,
                pre_soft_percent: float = 0.0, pre_attenuation_median: float = 0.0) -> LateralEvent:
  if raw_actual_corr is None:
    raw_actual_corr = _correlation(cols["raw_desired_curvature"][idx], cols["curvature"][idx])
  steering_rate = _derivative(cols["t"], cols["steering_angle_deg"])
  return LateralEvent(
    kind=kind,
    start_s=float(np.min(cols["t"][idx])),
    end_s=float(np.max(cols["t"][idx])),
    score=float(score),
    sample_count=int(np.sum(idx)),
    speed_mps_median=_median(cols["v_ego"][idx]),
    steering_angle_pp=_percentile_span(cols["steering_angle_deg"][idx]),
    steering_rate_p95=_p95_abs(steering_rate[idx]),
    raw_curvature_pp=_percentile_span(cols["raw_desired_curvature"][idx]),
    desired_curvature_pp=_percentile_span(cols["desired_curvature"][idx]),
    actual_curvature_pp=_percentile_span(cols["curvature"][idx]),
    output_pp=_percentile_span(cols["lat_output"][idx]),
    raw_actual_corr=raw_actual_corr,
    gated_percent=_percent(cols["model_path_gated"][idx] > 0.5),
    quality_median=_median(cols["model_path_quality"][idx]),
    steering_reversals=steering_reversals,
    raw_reversals=raw_reversals,
    pre_soft_percent=float(pre_soft_percent),
    pre_attenuation_median=float(pre_attenuation_median),
    requested_torque_pp=_percentile_span(cols["requested_torque"][idx]),
    applied_torque_pp=_percentile_span(cols["applied_torque"][idx]),
    eps_torque_pp=_percentile_span(cols["eps_torque"][idx]),
    driver_torque_p95=_p95_abs(cols["driver_torque"][idx]),
    command_eps_corr=_correlation(cols["requested_torque"][idx], cols["eps_torque"][idx]),
    command_applied_corr=_correlation(cols["requested_torque"][idx], cols["applied_torque"][idx]),
  )


def _select_events(events: list[LateralEvent], max_events: int) -> list[LateralEvent]:
  selected: list[LateralEvent] = []
  selected_by_kind: dict[str, int] = {}
  max_per_kind = max(1, max_events // 3)
  for event in sorted(events, key=lambda item: item.score, reverse=True):
    if selected_by_kind.get(event.kind, 0) >= max_per_kind:
      continue
    if all(event.kind != seen.kind or abs(event.start_s - seen.start_s) >= 5.0 for seen in selected):
      selected.append(event)
      selected_by_kind[event.kind] = selected_by_kind.get(event.kind, 0) + 1
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
    & ((cols["lane_change_off"] > 0.5) | (cols["lane_change_unknown"] > 0.5))
    & np.isfinite(cols["raw_desired_curvature"])
    & np.isfinite(cols["processed_desired_curvature"])
    & np.isfinite(cols["curvature"])
    & np.isfinite(cols["steering_angle_deg"])
  )


def _attenuation(cols: dict[str, np.ndarray]) -> np.ndarray:
  raw = cols["raw_desired_curvature"]
  processed = cols["processed_desired_curvature"]
  attenuation = np.zeros_like(raw)
  valid = np.isfinite(raw) & np.isfinite(processed) & (np.abs(raw) > 2e-4)
  attenuation[valid] = 1.0 - (np.abs(processed[valid]) / np.abs(raw[valid]))
  return attenuation


def _event_from_dict(data: dict[str, Any]) -> LateralEvent:
  raw_actual_corr = data.get("raw_actual_corr", data.get("rawActualCorr"))
  return LateralEvent(
    kind=str(data.get("kind", "unknown")),
    start_s=float(data.get("start_s", data.get("startS", 0.0))),
    end_s=float(data.get("end_s", data.get("endS", 0.0))),
    score=float(data.get("score", 0.0)),
    sample_count=int(data.get("sample_count", data.get("sampleCount", 0))),
    speed_mps_median=float(data.get("speed_mps_median", data.get("speedMpsMedian", 0.0))),
    steering_angle_pp=float(data.get("steering_angle_pp", data.get("steeringAnglePp", 0.0))),
    steering_rate_p95=float(data.get("steering_rate_p95", data.get("steeringRateP95", 0.0))),
    raw_curvature_pp=float(data.get("raw_curvature_pp", data.get("rawCurvaturePp", 0.0))),
    desired_curvature_pp=float(data.get("desired_curvature_pp", data.get("desiredCurvaturePp", 0.0))),
    actual_curvature_pp=float(data.get("actual_curvature_pp", data.get("actualCurvaturePp", 0.0))),
    output_pp=float(data.get("output_pp", data.get("outputPp", 0.0))),
    raw_actual_corr=None if raw_actual_corr is None else float(raw_actual_corr),
    gated_percent=float(data.get("gated_percent", data.get("gatedPercent", 0.0))),
    quality_median=float(data.get("quality_median", data.get("qualityMedian", 0.0))),
    steering_reversals=int(data.get("steering_reversals", data.get("steeringReversals", 0))),
    raw_reversals=int(data.get("raw_reversals", data.get("rawReversals", 0))),
    pre_soft_percent=float(data.get("pre_soft_percent", data.get("preSoftPercent", 0.0))),
    pre_attenuation_median=float(data.get("pre_attenuation_median", data.get("preAttenuationMedian", 0.0))),
    requested_torque_pp=float(data.get("requested_torque_pp", data.get("requestedTorquePp", 0.0))),
    applied_torque_pp=float(data.get("applied_torque_pp", data.get("appliedTorquePp", 0.0))),
    eps_torque_pp=float(data.get("eps_torque_pp", data.get("epsTorquePp", 0.0))),
    driver_torque_p95=float(data.get("driver_torque_p95", data.get("driverTorqueP95", 0.0))),
    command_eps_corr=_optional_float(data.get("command_eps_corr", data.get("commandEpsCorr"))),
    command_applied_corr=_optional_float(data.get("command_applied_corr", data.get("commandAppliedCorr"))),
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


def _optional_float(value: Any) -> float | None:
  return None if value is None else float(value)


def _format_optional(value: float | None) -> str:
  return "n/a" if value is None else f"{value:.3f}"
