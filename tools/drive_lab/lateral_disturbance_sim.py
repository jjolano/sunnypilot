from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from openpilot.tools.drive_lab.scenario_spec import ScenarioSpec, route_window_provenance


@dataclass(frozen=True)
class LateralDisturbanceConfig:
  seed: int = 1
  duration_s: float = 60.0
  dt_s: float = 0.05
  speed_mps: float = 20.0
  steering_gain: float = 3200.0
  controller_gain: float = 1.0
  controller_damping: float = 0.25
  crown_curvature: float = 0.0
  crosswind_curvature: float = 0.0
  crosswind_gust_curvature: float = 0.0
  crosswind_gust_period_s: float = 12.0
  model_noise_std: float = 0.0
  sensor_noise_std: float = 0.0
  steering_stiction_deg: float = 0.0
  steering_backlash_deg: float = 0.0
  actuator_delay_s: float = 0.0
  actuator_rate_limit_deg_s: float = 180.0
  tire_gain: float = 1.0
  tire_lag_s: float = 0.25
  tire_saturation_curvature: float = 0.02
  authority_attenuation: float = 0.0
  authority_start_s: float = 0.0
  authority_end_s: float = 0.0

  def to_dict(self) -> dict[str, Any]:
    return asdict(self)

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> LateralDisturbanceConfig:
    fields = cls.__dataclass_fields__
    return cls(**{key: data[key] for key in fields if key in data})


@dataclass(frozen=True)
class LateralTrace:
  t: tuple[float, ...]
  v_ego: tuple[float, ...]
  desired_curvature: tuple[float, ...]
  steering_angle_deg: tuple[float, ...] = ()

  def to_dict(self) -> dict[str, Any]:
    return asdict(self)

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> LateralTrace:
    return cls(
      t=tuple(float(v) for v in data.get("t", ())),
      v_ego=tuple(float(v) for v in data.get("v_ego", data.get("vEgo", ()))),
      desired_curvature=tuple(float(v) for v in data.get("desired_curvature", data.get("desiredCurvature", ()))),
      steering_angle_deg=tuple(float(v) for v in data.get("steering_angle_deg", data.get("steeringAngleDeg", ()))),
    )


@dataclass(frozen=True)
class LateralSimSample:
  t: float
  v_ego: float
  desired_curvature: float
  disturbed_curvature: float
  commanded_steering_deg: float
  actuator_steering_deg: float
  actual_curvature: float
  authority_attenuation: float


@dataclass(frozen=True)
class LateralEventWindow:
  kind: str
  start_s: float
  end_s: float
  score: float
  steering_angle_pp: float
  steering_rate_p95: float
  desired_curvature_pp: float
  actual_curvature_pp: float
  steering_reversals: int
  desired_reversals: int
  authority_attenuation_median: float


@dataclass(frozen=True)
class LateralDisturbanceReport:
  source: str
  config_hash: str
  config: LateralDisturbanceConfig
  sample_count: int
  duration_s: float
  steering_angle_pp: float
  steering_rate_p95: float
  desired_actual_lag_s: float | None
  desired_actual_corr: float | None
  fast_reversal_count: int
  rebound_count: int
  top_events: list[LateralEventWindow]

  def to_dict(self) -> dict[str, Any]:
    return {
      **asdict(self),
      "config": self.config.to_dict(),
      "top_events": [event.__dict__ for event in self.top_events],
    }


def synthetic_lateral_trace(config: LateralDisturbanceConfig, kind: str = "straight") -> LateralTrace:
  t = np.arange(0.0, max(config.duration_s, config.dt_s), config.dt_s, dtype=float)
  v = np.full_like(t, config.speed_mps)
  if kind == "straight":
    desired = np.zeros_like(t)
  elif kind == "sine":
    desired = 0.0010 * np.sin(2.0 * np.pi * t / 8.0)
  elif kind == "reversal":
    desired = np.where((np.floor(t / 1.0).astype(int) % 2) == 0, 0.0010, -0.0010)
  elif kind == "curve_entry":
    desired = 0.0025 / (1.0 + np.exp(-(t - config.duration_s * 0.45) * 2.5))
  else:
    raise ValueError(f"unknown synthetic profile {kind!r}")
  return LateralTrace(tuple(t), tuple(v), tuple(desired))


def route_lateral_trace(msgs: Iterable[Any], source: str = "route") -> LateralTrace:
  from openpilot.tools.drive_lab.lateral_oscillation_profile import _columns, _extract_lateral_samples

  samples = _extract_lateral_samples(list(msgs))
  cols = _columns(samples)
  active = (
    (cols["lat_active"] > 0.5)
    & (cols["steering_pressed"] < 0.5)
    & (cols["blinker_active"] < 0.5)
    & (cols["lane_change_off"] > 0.5)
  )
  if not np.any(active):
    return LateralTrace((), (), (), ())
  t = cols["t"][active]
  t = t - float(t[0])
  return LateralTrace(
    tuple(float(v) for v in t),
    tuple(float(v) for v in cols["v_ego"][active]),
    tuple(float(v) for v in cols["desired_curvature"][active]),
    tuple(float(v) for v in cols["steering_angle_deg"][active]),
  )


def simulate_lateral_disturbance(trace: LateralTrace, config: LateralDisturbanceConfig, source: str = "synthetic") -> LateralDisturbanceReport:
  t = np.array(trace.t, dtype=float)
  if t.size == 0:
    return _empty_report(source, config)
  v_ego = _resample_or_fill(trace.v_ego, t.size, config.speed_mps)
  desired = np.array(trace.desired_curvature, dtype=float)
  rng = np.random.default_rng(config.seed)

  model_noise = rng.normal(0.0, config.model_noise_std, size=t.size) if config.model_noise_std > 0.0 else np.zeros_like(t)
  gust = config.crosswind_gust_curvature * np.sin(2.0 * np.pi * t / max(config.crosswind_gust_period_s, config.dt_s))
  disturbance = config.crown_curvature + config.crosswind_curvature + gust
  authority = _authority_attenuation(t, config)
  disturbed = desired * (1.0 - authority) + model_noise

  command = disturbed * config.steering_gain * config.controller_gain
  actuator = _apply_steering_mechanics(command, t, config)
  actual = _simulate_tire_response(actuator, disturbance, t, config)
  if config.sensor_noise_std > 0.0:
    actual = actual + rng.normal(0.0, config.sensor_noise_std, size=t.size)
  steering_rate = _derivative(t, actuator)
  events = _rank_events(t, disturbed, actual, actuator, steering_rate, authority)
  fast_count = sum(1 for event in events if event.kind == "fast_reversal")
  rebound_count = sum(1 for event in events if event.kind == "rebound")
  return LateralDisturbanceReport(
    source=source,
    config_hash=_config_hash(config),
    config=config,
    sample_count=int(t.size),
    duration_s=float(t[-1] - t[0]) if t.size > 1 else 0.0,
    steering_angle_pp=_percentile_span(actuator),
    steering_rate_p95=_p95_abs(steering_rate),
    desired_actual_lag_s=_best_lag(t, disturbed, actual),
    desired_actual_corr=_correlation(disturbed, actual),
    fast_reversal_count=fast_count,
    rebound_count=rebound_count,
    top_events=events[:12],
  )


def render_lateral_disturbance_report(report: LateralDisturbanceReport) -> str:
  lines = [
    f"Lateral disturbance simulation: {report.source}",
    f"samples: {report.sample_count}",
    f"duration: {report.duration_s:.1f} s",
    f"config hash: {report.config_hash}",
    f"steering angle pp: {report.steering_angle_pp:.3f} deg",
    f"steering rate p95: {report.steering_rate_p95:.3f} deg/s",
    f"desired->actual lag: {_format_optional(report.desired_actual_lag_s)} s",
    f"desired->actual corr: {_format_optional(report.desired_actual_corr)}",
    f"fast reversal events: {report.fast_reversal_count}",
    f"rebound events: {report.rebound_count}",
  ]
  if report.top_events:
    lines.append("Top events:")
    for event in report.top_events:
      lines.append(
        f"  {event.kind} {event.start_s:.1f}-{event.end_s:.1f}s "
        f"score={event.score:.2f} steer_pp={event.steering_angle_pp:.3f}deg "
        f"rate95={event.steering_rate_p95:.3f} reversals={event.steering_reversals} "
        f"atten={event.authority_attenuation_median:.2f}"
      )
  return "\n".join(lines)


def lateral_disturbance_event_to_spec(report: LateralDisturbanceReport, event: LateralEventWindow, index: int | None = None) -> ScenarioSpec:
  maneuvers = {
    "config": report.config.to_dict(),
    "event": {
      "kind": event.kind,
      "start_s": event.start_s,
      "end_s": event.end_s,
      "score": event.score,
    },
  }
  checks = ["finite", "lag"]
  if event.kind == "fast_reversal":
    checks.append("reversal")
  elif event.kind == "rebound":
    checks.append("rebound")
  return ScenarioSpec(
    scenario_id=f"{report.source}:{event.kind}:{report.config.seed}:{index if index is not None else int(round(event.start_s * 10))}",
    kind=event.kind,
    title=f"{report.source} {event.kind} {event.start_s:.1f}-{event.end_s:.1f}s",
    mode="lateral-disturbance",
    duration=event.end_s - event.start_s,
    source=report.source,
    maneuver_kwargs=maneuvers,
    ego={"speed_mps": report.config.speed_mps},
    events=(event.kind,),
    oracle={"checks": tuple(checks)},
    tags=("route-derived", "lateral", "lateral-disturbance", event.kind),
    seed=report.config.seed,
    index=index,
    provenance={
      **route_window_provenance(report.source, None, event.start_s, event.end_s, "lateral_disturbance_sim"),
      "config_hash": report.config_hash,
      "event_start_s": event.start_s,
      "event_end_s": event.end_s,
    },
  )


def lateral_disturbance_report_to_specs(report: LateralDisturbanceReport, max_events: int = 12) -> list[ScenarioSpec]:
  return [lateral_disturbance_event_to_spec(report, event, index=i) for i, event in enumerate(report.top_events[:max_events])]


def save_lateral_disturbance_report(report: LateralDisturbanceReport, path: str | Path) -> None:
  Path(path).write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True))


def _empty_report(source: str, config: LateralDisturbanceConfig) -> LateralDisturbanceReport:
  return LateralDisturbanceReport(source, _config_hash(config), config, 0, 0.0, 0.0, 0.0, None, None, 0, 0, [])


def _resample_or_fill(values: tuple[float, ...], size: int, fill: float) -> np.ndarray:
  arr = np.array(values, dtype=float)
  if arr.size == size:
    return arr
  if arr.size == 0:
    return np.full(size, fill, dtype=float)
  return np.interp(np.linspace(0.0, arr.size - 1, size), np.arange(arr.size), arr)


def _authority_attenuation(t: np.ndarray, config: LateralDisturbanceConfig) -> np.ndarray:
  authority = np.zeros_like(t)
  if config.authority_attenuation <= 0.0 or config.authority_end_s <= config.authority_start_s:
    return authority
  mask = (t >= config.authority_start_s) & (t <= config.authority_end_s)
  authority[mask] = min(max(config.authority_attenuation, 0.0), 0.95)
  return authority


def _apply_steering_mechanics(command: np.ndarray, t: np.ndarray, config: LateralDisturbanceConfig) -> np.ndarray:
  delay_frames = max(0, int(round(config.actuator_delay_s / max(config.dt_s, 1e-3))))
  delayed = np.concatenate((np.full(delay_frames, command[0]), command))[:command.size] if delay_frames else command.copy()
  actuator = np.zeros_like(delayed)
  actuator[0] = delayed[0]
  backlash = max(config.steering_backlash_deg, 0.0)
  stiction = max(config.steering_stiction_deg, 0.0)
  rate_limit = max(config.actuator_rate_limit_deg_s, 1e-3)
  for i in range(1, delayed.size):
    dt = max(float(t[i] - t[i - 1]), 1e-3)
    delta = delayed[i] - actuator[i - 1]
    if abs(delta) <= stiction:
      target = actuator[i - 1]
    else:
      target = delayed[i] - math.copysign(min(backlash, abs(delta)), delta)
    max_step = rate_limit * dt
    actuator[i] = actuator[i - 1] + float(np.clip(target - actuator[i - 1], -max_step, max_step))
  return actuator


def _simulate_tire_response(actuator: np.ndarray, disturbance: np.ndarray, t: np.ndarray, config: LateralDisturbanceConfig) -> np.ndarray:
  actual = np.zeros_like(actuator)
  lag = max(config.tire_lag_s, 1e-3)
  saturation = max(config.tire_saturation_curvature, 1e-4)
  for i in range(1, actuator.size):
    dt = max(float(t[i] - t[i - 1]), 1e-3)
    target = config.tire_gain * actuator[i] / max(config.steering_gain, 1e-3) + disturbance[i]
    target = float(np.clip(target, -saturation, saturation))
    alpha = dt / (dt + lag)
    actual[i] = actual[i - 1] + alpha * (target - actual[i - 1])
  return actual


def _rank_events(t: np.ndarray, desired: np.ndarray, actual: np.ndarray, steering: np.ndarray,
                 steering_rate: np.ndarray, authority: np.ndarray) -> list[LateralEventWindow]:
  events: list[LateralEventWindow] = []
  for start in np.arange(float(t[0]), float(t[-1]) - 4.0, 1.0):
    mask = (t >= start) & (t < start + 4.0)
    if int(np.sum(mask)) < 10:
      continue
    steering_reversals = _sign_flip_count(steering[mask] - _median(steering[mask]), 0.08)
    desired_reversals = _sign_flip_count(desired[mask] - _median(desired[mask]), 2e-5)
    steering_rate_p95 = _p95_abs(steering_rate[mask])
    steering_pp = _percentile_span(steering[mask])
    authority_med = _median(authority[mask])
    if steering_reversals >= 4 and steering_rate_p95 >= 6.0:
      score = steering_reversals * 3.0 + steering_rate_p95 * 0.5 + steering_pp
      events.append(_event("fast_reversal", start, start + 4.0, score, steering_pp, steering_rate_p95, desired, actual, steering, authority, mask, desired_reversals))
    post = (t >= start + 4.0) & (t < start + 7.0)
    if authority_med > 0.1 and int(np.sum(post)) >= 10:
      actual_jump = max(0.0, _percentile_span(actual[post]) - _percentile_span(actual[mask]))
      post_rate = _p95_abs(steering_rate[post])
      if actual_jump > 4e-4 or post_rate > 10.0:
        score = authority_med * 20.0 + actual_jump * 5000.0 + post_rate * 0.5
        post_mask = mask | post
        events.append(_event("rebound", start, start + 7.0, score, _percentile_span(steering[post_mask]), post_rate, desired, actual, steering, authority, post_mask, _sign_flip_count(desired[post_mask] - _median(desired[post_mask]), 2e-5)))
  selected: list[LateralEventWindow] = []
  for event in sorted(events, key=lambda e: e.score, reverse=True):
    if all(abs(event.start_s - seen.start_s) >= 3.0 or event.kind != seen.kind for seen in selected):
      selected.append(event)
  return selected


def _event(kind: str, start: float, end: float, score: float, steering_pp: float, steering_rate_p95: float,
           desired: np.ndarray, actual: np.ndarray, steering: np.ndarray, authority: np.ndarray,
           mask: np.ndarray, desired_reversals: int) -> LateralEventWindow:
  return LateralEventWindow(
    kind,
    float(start),
    float(end),
    float(score),
    float(steering_pp),
    float(steering_rate_p95),
    _percentile_span(desired[mask]),
    _percentile_span(actual[mask]),
    _sign_flip_count(steering[mask] - _median(steering[mask]), 0.08),
    desired_reversals,
    _median(authority[mask]),
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


def _best_lag(t: np.ndarray, desired: np.ndarray, actual: np.ndarray) -> float | None:
  if t.size < 10:
    return None
  dt = float(np.nanmedian(np.diff(t)))
  if not math.isfinite(dt) or dt <= 0.0:
    return None
  best_lag = 0
  best_corr = -2.0
  for lag in range(-40, 41):
    if lag < 0:
      a, b = desired[-lag:], actual[:lag]
    elif lag > 0:
      a, b = desired[:-lag], actual[lag:]
    else:
      a, b = desired, actual
    c = _correlation(a, b)
    if c is not None and c > best_corr:
      best_corr = c
      best_lag = lag
  return float(best_lag * dt)


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


def _correlation(a: np.ndarray, b: np.ndarray) -> float | None:
  ok = np.isfinite(a) & np.isfinite(b)
  if int(np.sum(ok)) < 5 or np.nanstd(a[ok]) < 1e-9 or np.nanstd(b[ok]) < 1e-9:
    return None
  return float(np.corrcoef(a[ok], b[ok])[0, 1])


def _config_hash(config: LateralDisturbanceConfig) -> str:
  raw = json.dumps(config.to_dict(), sort_keys=True).encode("utf-8")
  return hashlib.sha256(raw).hexdigest()[:12]


def _format_optional(value: float | None) -> str:
  return "n/a" if value is None else f"{value:.3f}"
