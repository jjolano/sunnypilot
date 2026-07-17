#!/usr/bin/env python3
"""Model-free, fixed-trace audit of the torque governor's reversal slew.

This deliberately does not claim a plant or causal effect.  Each route is replayed
independently through the validated OutputGovernor trace in ``replay_output_governor``.
Reversal events are kept at least 1.10 s apart; each event uses a 0.50 s post-reversal
horizon.  The intervention is the fixed-trace G0-vs-G2 impulse, while the angle and
body-lateral-acceleration outcomes are measured from the logged vehicle response.
"""
from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from openpilot.selfdrive.locationd.helpers import Pose, PoseCalibrator
from openpilot.tools.drive_lab.analyze_longitudinal_lateral_route import DEFAULT_LOG_ROOTS, resolve_inputs
from openpilot.tools.drive_lab.replay_output_governor import (
  DT,
  GovernorFrame,
  ReplaySample,
  _extract_frames,
  _replay,
  _safe_get,
  _sign,
  _sort_log_paths,
)
from openpilot.tools.lib.logreader import LogReader, ReadMode


# Max possible outcome start delay (0.50 s) + body delay + 0.50 s outcome horizon.
EVENT_SPACING_S = 1.10
OUTCOME_HORIZON_S = 0.50
OUTCOME_TICKS = int(round(OUTCOME_HORIZON_S / DT))
BODY_DELAY_S = 0.08
TORQUE_STEER_FALLBACK_S = 0.12
DELAY_MIN_S = 0.02
DELAY_MAX_S = 0.50
DELAY_MIN_CORRELATION = 0.10
BOOTSTRAP_COUNT = 1000
FEATURE_NAMES = ("impulse", "abs_error", "abs_steering_rate", "angle_accel", "nominal_step", "speed")


@dataclass(frozen=True)
class ReversalCandidate:
  index: int
  t: float
  old_sign: int
  nominal_step: float


@dataclass(frozen=True)
class ControlContext:
  steering_angle_deg: float
  angle_accel_deg_s2: float


@dataclass(frozen=True)
class ReversalEvent:
  route: str
  t: float
  impulse: float
  old_direction_hold_s: float
  abs_error: float
  abs_steering_rate: float
  angle_accel: float
  nominal_step: float
  speed: float
  steering_angle_deg: float
  angle_excursion_deg: float | None
  body_lat_accel_excursion: float | None
  placebo_angle_excursion_deg: float | None
  placebo_body_lat_accel_excursion: float | None


@dataclass
class RouteAudit:
  route: str
  identifiers: list[str]
  frames: list[GovernorFrame]
  samples: dict[str, list[ReplaySample]]
  events: list[ReversalEvent]
  delay_s: float
  delay_method: str
  delay_correlation: float | None
  g1_difference: dict[str, float]
  notes: list[str]

  def to_dict(self) -> dict[str, Any]:
    return {
      "route": self.route,
      "segments": len(self.identifiers),
      "events": [asdict(event) for event in self.events],
      "delay_s": self.delay_s,
      "delay_method": self.delay_method,
      "delay_correlation": self.delay_correlation,
      "g1_difference": self.g1_difference,
      "notes": self.notes,
    }


def _finite(value: Any) -> float:
  try:
    result = float(value)
  except (TypeError, ValueError):
    return math.nan
  return result if math.isfinite(result) else math.nan


def _correlation(x: np.ndarray, y: np.ndarray) -> float | None:
  mask = np.isfinite(x) & np.isfinite(y)
  if int(mask.sum()) < 3:
    return None
  x, y = x[mask], y[mask]
  if np.std(x) <= 1e-9 or np.std(y) <= 1e-9:
    return None
  return float(np.corrcoef(x, y)[0, 1])


def _rank(values: np.ndarray) -> np.ndarray:
  order = np.argsort(values, kind="mergesort")
  ranks = np.empty(len(values), dtype=float)
  i = 0
  while i < len(values):
    j = i + 1
    while j < len(values) and values[order[j]] == values[order[i]]:
      j += 1
    ranks[order[i:j]] = (i + j - 1) / 2.0
    i = j
  return ranks


def _spearman(x: Iterable[float], y: Iterable[float]) -> float | None:
  xa, ya = np.asarray(list(x), dtype=float), np.asarray(list(y), dtype=float)
  if xa.size != ya.size or xa.size < 3:
    return None
  mask = np.isfinite(xa) & np.isfinite(ya)
  if int(mask.sum()) < 3:
    return None
  return _correlation(_rank(xa[mask]), _rank(ya[mask]))


def _cluster_reversals(candidates: list[ReversalCandidate], spacing_s: float = EVENT_SPACING_S) -> list[ReversalCandidate]:
  """Keep the strongest candidate in each time cluster; selected windows cannot overlap."""
  if not candidates:
    return []
  clusters: list[list[ReversalCandidate]] = [[candidates[0]]]
  for candidate in candidates[1:]:
    if candidate.t - clusters[-1][-1].t < spacing_s:
      clusters[-1].append(candidate)
    else:
      clusters.append([candidate])
  return [max(cluster, key=lambda candidate: (candidate.nominal_step, -candidate.t)) for cluster in clusters]


def _nominal_reversals(samples: list[ReplaySample]) -> list[ReversalCandidate]:
  candidates: list[ReversalCandidate] = []
  previous_sign = 0
  previous_value = 0.0
  for index, sample in enumerate(samples):
    if not sample.active:
      previous_sign = 0
      previous_value = 0.0
      continue
    current_sign = _sign(sample.nominal_torque)
    if current_sign and previous_sign and current_sign != previous_sign:
      candidates.append(ReversalCandidate(index, sample.t, previous_sign,
                                          abs(sample.nominal_torque - previous_value)))
    if current_sign:
      previous_sign = current_sign
      previous_value = sample.nominal_torque
  return candidates


def _event_impulse(samples_g0: list[ReplaySample], samples_g2: list[ReplaySample], index: int,
                   horizon_ticks: int = OUTCOME_TICKS) -> float:
  end = min(len(samples_g0), len(samples_g2), index + horizon_ticks)
  return float(sum(abs(samples_g0[i].output_torque - samples_g2[i].output_torque) * DT
                   for i in range(index, end)))


def _old_direction_hold(samples: list[ReplaySample], candidate: ReversalCandidate,
                        horizon_ticks: int = OUTCOME_TICKS) -> float:
  end = min(len(samples), candidate.index + horizon_ticks)
  ticks = 0
  for sample in samples[candidate.index:end]:
    if _sign(sample.output_torque) != candidate.old_sign:
      break
    ticks += 1
  return ticks * DT


def _latest_series_value(times: np.ndarray, values: np.ndarray, t: float) -> float:
  if not len(times):
    return math.nan
  index = int(np.searchsorted(times, t, side="right") - 1)
  return float(values[index]) if index >= 0 else math.nan


def _window_excursion(times: np.ndarray, values: np.ndarray, start: float, end: float) -> float | None:
  if len(times) < 2 or start < times[0] or end > times[-1]:
    return None
  mask = (times >= start) & (times <= end) & np.isfinite(values)
  if int(mask.sum()) < 2:
    return None
  baseline = float(np.interp(start, times, values))
  return float(np.max(np.abs(values[mask] - baseline)))


def _route_signals(messages: list[Any], frames: list[GovernorFrame]) -> tuple[list[ControlContext], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Extract steering angle and calibrated body lateral acceleration without merging routes."""
  ordered = sorted(messages, key=lambda message: int(getattr(message, "logMonoTime", 0)))
  if not ordered:
    empty = np.empty(0, dtype=float)
    return [], empty, empty, empty, empty
  first = int(getattr(ordered[0], "logMonoTime", 0))
  car_t: list[float] = []
  car_angle: list[float] = []
  pose_t: list[float] = []
  body_lat_accel: list[float] = []
  calibrator = PoseCalibrator()
  latest_speed = math.nan
  for message in ordered:
    t = (int(getattr(message, "logMonoTime", 0)) - first) / 1e9
    which = message.which()
    if which == "liveCalibration":
      try:
        calibrator.feed_live_calib(message.liveCalibration)
      except Exception:
        pass
    elif which == "carState":
      state = message.carState
      latest_speed = _finite(_safe_get(state, "vEgo"))
      angle = _finite(_safe_get(state, "steeringAngleDeg"))
      if math.isfinite(angle):
        car_t.append(t)
        car_angle.append(angle)
    elif which == "livePose":
      try:
        pose = calibrator.build_calibrated_pose(Pose.from_live_pose(message.livePose))
        yaw_rate = _finite(pose.angular_velocity.z)
      except Exception:
        yaw_rate = math.nan
      if math.isfinite(yaw_rate) and math.isfinite(latest_speed):
        pose_t.append(t)
        body_lat_accel.append(yaw_rate * latest_speed)

  angle_t = np.asarray(car_t, dtype=float)
  angle_values = np.asarray(car_angle, dtype=float)
  pose_times = np.asarray(pose_t, dtype=float)
  body_values = np.asarray(body_lat_accel, dtype=float)
  angle_accel = np.full(len(angle_values), math.nan)
  if len(angle_values) >= 3 and np.all(np.diff(angle_t) > 0.0):
    angle_accel = np.gradient(np.gradient(angle_values, angle_t), angle_t)
  contexts = [ControlContext(
    steering_angle_deg=_latest_series_value(angle_t, angle_values, frame.t),
    angle_accel_deg_s2=_latest_series_value(angle_t, angle_accel, frame.t),
  ) for frame in frames]
  return contexts, angle_t, angle_values, pose_times, body_values


def _estimate_delay(frames: list[GovernorFrame], contexts: list[ControlContext]) -> tuple[float, str, float | None]:
  if len(frames) < 100 or len(contexts) != len(frames):
    return TORQUE_STEER_FALLBACK_S, "fixed fallback (insufficient angle trace)", None
  times = np.asarray([frame.t for frame in frames], dtype=float)
  torque = np.asarray([frame.nominal_torque for frame in frames], dtype=float)
  angle = np.asarray([context.steering_angle_deg for context in contexts], dtype=float)
  valid = np.isfinite(times) & np.isfinite(torque) & np.isfinite(angle)
  if int(valid.sum()) < 100:
    return TORQUE_STEER_FALLBACK_S, "fixed fallback (insufficient monotonic angle trace)", None
  times, torque, angle = times[valid], torque[valid], angle[valid]
  if len(times) < 100 or not np.all(np.diff(times) > 0.0):
    return TORQUE_STEER_FALLBACK_S, "fixed fallback (insufficient monotonic angle trace)", None
  torque_rate = np.gradient(torque, times)
  angle_rate = np.gradient(angle, times)
  dt = float(np.median(np.diff(times)))
  if not math.isfinite(dt) or dt <= 0.0:
    return TORQUE_STEER_FALLBACK_S, "fixed fallback (invalid sample interval)", None
  best: tuple[float, float] | None = None
  for delay in np.arange(DELAY_MIN_S, DELAY_MAX_S + DT / 2, DT):
    shift = max(1, int(round(delay / dt)))
    if shift >= len(times) - 3:
      continue
    corr = _correlation(torque_rate[:-shift], angle_rate[shift:])
    if corr is not None and (best is None or abs(corr) > abs(best[1])):
      best = (float(delay), corr)
  if best is None or abs(best[1]) < DELAY_MIN_CORRELATION:
    return TORQUE_STEER_FALLBACK_S, "fixed fallback (no reliable delay correlation)", None
  return best[0], f"data-derived torque-rate/angle-rate correlation={best[1]:+.3f}", best[1]


def _event_outcomes(route: str, frames: list[GovernorFrame], samples: dict[str, list[ReplaySample]],
                    contexts: list[ControlContext], angle_t: np.ndarray, angle_values: np.ndarray,
                    pose_t: np.ndarray, body_values: np.ndarray, delay_s: float) -> list[ReversalEvent]:
  candidates = _cluster_reversals(_nominal_reversals(samples["G0"]))
  events: list[ReversalEvent] = []
  for candidate in candidates:
    if candidate.index + OUTCOME_TICKS > len(frames) or candidate.index >= len(contexts):
      continue
    frame = frames[candidate.index]
    context = contexts[candidate.index]
    angle_start = frame.t + delay_s
    angle_end = angle_start + OUTCOME_HORIZON_S
    body_start = angle_start + BODY_DELAY_S
    body_end = body_start + OUTCOME_HORIZON_S
    pre_start = frame.t - OUTCOME_HORIZON_S
    pre_end = frame.t
    events.append(ReversalEvent(
      route=route,
      t=frame.t,
      impulse=_event_impulse(samples["G0"], samples["G2"], candidate.index),
      old_direction_hold_s=_old_direction_hold(samples["G0"], candidate),
      abs_error=abs(frame.desired_lateral_accel - frame.actual_lateral_accel),
      abs_steering_rate=abs(frame.steering_rate_deg),
      angle_accel=context.angle_accel_deg_s2,
      nominal_step=candidate.nominal_step,
      speed=frame.v_ego,
      steering_angle_deg=context.steering_angle_deg,
      angle_excursion_deg=_window_excursion(angle_t, angle_values, angle_start, angle_end),
      body_lat_accel_excursion=_window_excursion(pose_t, body_values, body_start, body_end),
      placebo_angle_excursion_deg=_window_excursion(angle_t, angle_values, pre_start, pre_end),
      placebo_body_lat_accel_excursion=_window_excursion(pose_t, body_values, pre_start, pre_end),
    ))
  return events


def _g1_difference(samples: dict[str, list[ReplaySample]], events: list[ReversalEvent] | None = None) -> dict[str, float]:
  ranges = [(event.t, event.t + OUTCOME_HORIZON_S) for event in events] if events else None
  differences = np.asarray([abs(g0.output_torque - g1.output_torque)
                            for g0, g1 in zip(samples["G0"], samples["G1"], strict=True)
                            if ranges is None or any(start <= g0.t < end for start, end in ranges)], dtype=float)
  return {
    "max": float(np.max(differences)) if len(differences) else 0.0,
    "p95": float(np.percentile(differences, 95)) if len(differences) else 0.0,
  }


def audit_route(route: str, log_roots: tuple[Path, ...]) -> RouteAudit:
  identifiers = _sort_log_paths(resolve_inputs([route], segment=None, read_mode=ReadMode.RLOG, log_roots=log_roots))
  messages = list(LogReader(identifiers, default_mode=ReadMode.RLOG, sort_by_time=True))
  frames = _extract_frames(messages)
  samples = _replay(frames)
  contexts, angle_t, angle_values, pose_t, body_values = _route_signals(messages, frames)
  delay_s, delay_method, delay_correlation = _estimate_delay(frames, contexts)
  events = _event_outcomes(route, frames, samples, contexts, angle_t, angle_values, pose_t, body_values, delay_s)
  notes = []
  if not len(angle_t):
    notes.append("steering-angle outcomes unavailable: no finite carState.steeringAngleDeg")
  if not len(pose_t):
    notes.append("body-lateral-acceleration outcomes unavailable: no calibrated livePose yaw-rate samples")
  notes.append("roll omitted: no reliably aligned roll signal is used")
  return RouteAudit(route, identifiers, frames, samples, events, delay_s, delay_method,
                    delay_correlation, _g1_difference(samples, events), notes)


def _event_matrix(events: list[ReversalEvent], outcome: str) -> tuple[np.ndarray, np.ndarray]:
  rows = []
  targets = []
  for event in events:
    values = [event.impulse, event.abs_error, event.abs_steering_rate, event.angle_accel,
              event.nominal_step, event.speed]
    target = getattr(event, outcome)
    if all(math.isfinite(value) for value in values) and target is not None and math.isfinite(target):
      rows.append(values)
      targets.append(target)
  return np.asarray(rows, dtype=float), np.asarray(targets, dtype=float)


def _bootstrap_rank_ci(impulse: np.ndarray, outcome: np.ndarray, seed: int) -> tuple[float, float] | None:
  if len(impulse) < 3:
    return None
  rng = np.random.default_rng(seed)
  values = []
  for _ in range(BOOTSTRAP_COUNT):
    index = rng.integers(0, len(impulse), len(impulse))
    statistic = _spearman(impulse[index], outcome[index])
    if statistic is not None:
      values.append(statistic)
  if not values:
    return None
  return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def _projection(train: RouteAudit, test: RouteAudit, outcome: str) -> dict[str, Any]:
  train_x, train_y = _event_matrix(train.events, outcome)
  test_x, test_y = _event_matrix(test.events, outcome)
  min_rows = len(FEATURE_NAMES) + 2
  if len(train_y) < min_rows or len(test_y) < 3:
    return {"status": "insufficient", "train_events": len(train_y), "test_events": len(test_y),
            "reason": f"need at least {min_rows} train and 3 held-out events"}
  mean = np.mean(train_x, axis=0)
  scale = np.std(train_x, axis=0)
  scale[scale < 1e-9] = 1.0
  y_mean = float(np.mean(train_y))
  y_scale = float(np.std(train_y))
  if y_scale < 1e-9:
    return {"status": "insufficient", "train_events": len(train_y), "test_events": len(test_y),
            "reason": "training outcome has no variance"}
  train_z = (train_x - mean) / scale
  test_z = (test_x - mean) / scale
  coefficients = np.linalg.lstsq(np.column_stack((np.ones(len(train_z)), train_z)),
                                 (train_y - y_mean) / y_scale, rcond=None)[0]
  predicted = (np.column_stack((np.ones(len(test_z)), test_z)) @ coefficients) * y_scale + y_mean
  residual = test_y - predicted
  total = np.sum((test_y - np.mean(test_y)) ** 2)
  r2 = float(1.0 - np.sum(residual ** 2) / total) if total > 1e-12 else None
  corr = _correlation(test_y, predicted)
  rank = _spearman(test_x[:, 0], test_y)
  seed = sum((index + 1) * ord(character) for index, character in enumerate(
    f"{train.route}|{test.route}|{outcome}")) & 0xffffffff
  ci = _bootstrap_rank_ci(test_x[:, 0], test_y, seed=seed)
  return {
    "status": "ok",
    "train_route": train.route,
    "test_route": test.route,
    "outcome": outcome,
    "train_events": len(train_y),
    "test_events": len(test_y),
    "standardized_impulse_coefficient": float(coefficients[1]),
    "holdout_r2": r2,
    "holdout_prediction_correlation": corr,
    "holdout_impulse_outcome_rank_correlation": rank,
    "rank_correlation_bootstrap_ci95": ci,
    "features": FEATURE_NAMES,
  }


def _placebo(route: RouteAudit, outcome: str) -> dict[str, Any]:
  values = [(event.impulse, getattr(event, f"placebo_{outcome}")) for event in route.events]
  values = [(impulse, value) for impulse, value in values if value is not None and math.isfinite(impulse) and math.isfinite(value)]
  return {"route": route.route, "outcome": outcome, "events": len(values),
          "impulse_outcome_rank_correlation": _spearman([v[0] for v in values], [v[1] for v in values])}


def _median_p95(values: Iterable[float]) -> tuple[float | None, float | None]:
  finite = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
  if not len(finite):
    return None, None
  return float(np.median(finite)), float(np.percentile(finite, 95))


def _render(audits: list[RouteAudit], projections: list[dict[str, Any]]) -> str:
  lines = [
    "Governor slew model-free audit",
    f"event spacing={EVENT_SPACING_S:.2f}s; post horizon={OUTCOME_HORIZON_S:.2f}s; body delay=+{BODY_DELAY_S:.2f}s",
    f"delay search={DELAY_MIN_S:.2f}-{DELAY_MAX_S:.2f}s; fixed fallback={TORQUE_STEER_FALLBACK_S:.2f}s when correlation is unreliable",
    "angle window starts at torque-to-steer delay; body window starts another +0.08 s later.",
    "per-event controls: abs accel error, abs steering-rate, angle acceleration, nominal step, speed, and steering angle; roll omitted.",
    "outcomes are excursions from the value at each window start; roll is explicitly omitted.",
  ]
  for audit in audits:
    hold_median, hold_p95 = _median_p95(event.old_direction_hold_s for event in audit.events)
    impulse_median, impulse_p95 = _median_p95(event.impulse for event in audit.events)
    lines.extend(("", f"Route {audit.route}: segments={len(audit.identifiers)} frames={len(audit.frames)} events={len(audit.events)}",
                  f"  torque-to-steer delay={audit.delay_s:.3f}s ({audit.delay_method})",
                  f"  G0 old-direction hold median/p95={hold_median}/{hold_p95}s; "
                  + f"G0-G2 impulse median/p95={impulse_median}/{impulse_p95}",
                  f"  negative control abs(G0-G1) in event windows: max={audit.g1_difference['max']:.6f} "
                  + f"p95={audit.g1_difference['p95']:.6f}"))
    if audit.g1_difference["max"] <= 1e-9:
      lines.append("  G1 conclusion: not inferred; the negative control is zero on this route.")
    for outcome in ("angle_excursion_deg", "body_lat_accel_excursion"):
      placebo = _placebo(audit, outcome)
      available = sum(getattr(event, outcome) is not None for event in audit.events)
      lines.append(f"  {outcome}: {available}/{len(audit.events)} events with measured outcome; "
                   + f"placebo impulse-rank={placebo['impulse_outcome_rank_correlation']}")
    lines.extend(f"  note: {note}" for note in audit.notes)
  lines.append("")
  lines.append("Leave-one-route-out local projections (standardized train features and outcome):")
  if not projections:
    lines.append("  insufficient: need at least two distinct route inputs.")
  for projection in projections:
    if projection["status"] != "ok":
      lines.append(f"  {projection.get('train_route')} -> {projection.get('test_route')}: insufficient ({projection['reason']})")
      continue
    lines.append(
      f"  {projection['train_route']} -> {projection['test_route']} {projection['outcome']}: "
      + f"beta_impulse={projection['standardized_impulse_coefficient']:+.3f} "
      + f"R2={projection['holdout_r2']} corr={projection['holdout_prediction_correlation']} "
      + f"rank={projection['holdout_impulse_outcome_rank_correlation']} "
      + f"bootstrap95={projection['rank_correlation_bootstrap_ci95']}"
    )
  lines.extend(("", "Conclusion gate: observational fixed-trace evidence only; this is not causal proof.",
                "A live slew change requires closed-course crossover or independent plant identification."))
  return "\n".join(lines)


def main() -> None:
  parser = argparse.ArgumentParser(description="Model-free fixed-trace audit of governor reversal slew.")
  parser.add_argument("inputs", nargs="+", help="Route ID/name, local route directory, or explicit local rlog")
  parser.add_argument("--log-root", action="append", default=[], help="Extra root for local short routes")
  parser.add_argument("--json", action="store_true", help="Print JSON instead of the text report")
  args = parser.parse_args()
  roots = tuple(Path(path) for path in args.log_root) + DEFAULT_LOG_ROOTS
  audits: list[RouteAudit] = []
  for route in args.inputs:
    try:
      audits.append(audit_route(route, roots))
    except Exception as exc:
      print(f"Route {route}: insufficient data ({type(exc).__name__}: {exc})")
  projections = [_projection(train, test, outcome)
                 for train in audits for test in audits if train.route != test.route
                 for outcome in ("angle_excursion_deg", "body_lat_accel_excursion")]
  if args.json:
    print(json.dumps({"routes": [audit.to_dict() for audit in audits], "projections": projections,
                      "constants": {"event_spacing_s": EVENT_SPACING_S, "outcome_horizon_s": OUTCOME_HORIZON_S,
                                     "body_delay_s": BODY_DELAY_S},
                      "conclusion_gate": "observational only; closed-course crossover or independent plant identification required"},
                   indent=2, default=str))
  else:
    print(_render(audits, projections))


if __name__ == "__main__":
  main()
