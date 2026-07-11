from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path
from typing import Any

import numpy as np

from openpilot.tools.drive_lab.lateral_torque_event_report import (
  LateralLowSpeedReport,
  LateralTorqueEventReport,
  build_lateral_low_speed_report,
  build_lateral_torque_event_report,
)
from openpilot.tools.drive_lab.timeline import format_enum, msg_payload, msg_time_s, msg_type, safe_get


TORQUE_EVENT_DOMINANT = "torque_event_dominant"
PATH_WANDER_DOMINANT = "path_wander_dominant"
LOW_SPEED_LATERAL_DOMINANT = "low_speed_lateral_dominant"
INSUFFICIENT_EVIDENCE = "insufficient_evidence"

DEMAND_DRIVEN_WANDER = "demand_driven_straight_path_wander"
ACTUATION_DRIVEN_WANDER = "actuation_driven_straight_path_wander"
MIXED_WANDER = "mixed_straight_path_wander"

BRANCH_RECOMMENDATIONS = {
  TORQUE_EVENT_DOMINANT: "feat/lateral-control",
  PATH_WANDER_DOMINANT: "feat/lateral-control",
  LOW_SPEED_LATERAL_DOMINANT: "feat/lateral-control",
  INSUFFICIENT_EVIDENCE: "none",
}

BROAD_STRAIGHT_MAX_RAW_CURVATURE = 0.004
BROAD_STRAIGHT_MIN_SPEED = 8.0
WANDER_MIN_STEERING_PP = 3.0
WANDER_MIN_ACTUAL_CURVATURE_PP = 8e-4
WANDER_MIN_DEMAND_CURVATURE_PP = 6e-4
WANDER_STRONG_CORR = 0.65
RECENTER_OFFSET_EPS = 0.05
RECENTER_CURVATURE_EPS = 2e-4


@dataclass(frozen=True)
class WanderCandidateWindow:
  start_s: float
  end_s: float
  sample_count: int
  confidence: str
  cause: str
  severity_score: float
  speed_mps_median: float
  steering_angle_pp: float
  actual_curvature_pp: float
  raw_curvature_pp: float
  processed_curvature_pp: float
  desired_curvature_pp: float
  raw_actual_corr: float | None
  processed_actual_corr: float | None
  gated_percent: float
  quality_median: float
  lane_state_unknown_percent: float


@dataclass(frozen=True)
class RecenterOvershootCandidate:
  start_s: float
  end_s: float
  sample_count: int
  confidence: str
  offset_agreement_percent: float
  model_path_offset_pp: float
  lane_center_offset_pp: float
  combined_offset_pp: float
  offset_crossings: int
  correction_reversals: int
  processed_curvature_pp: float
  lane_state_unknown_percent: float


@dataclass(frozen=True)
class LateralPerformanceGateReport:
  source: str
  sample_count: int
  duration_s: float
  active_percent: float
  lane_state_unknown_percent: float
  qlog_safe_lane_policy: bool
  dominant_failure_class: str
  branch_recommendation: str
  confidence: str
  torque_event_score: float
  path_wander_score: float
  low_speed_score: float
  torque_event_report: LateralTorqueEventReport
  low_speed_report: LateralLowSpeedReport
  wander_candidate_windows: list[WanderCandidateWindow]
  recenter_overshoot_candidates: list[RecenterOvershootCandidate]
  notes: list[str]

  def to_dict(self) -> dict[str, Any]:
    return asdict(self)

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> LateralPerformanceGateReport:
    return cls(
      source=str(data.get("source", "unknown")),
      sample_count=int(data.get("sample_count", data.get("sampleCount", 0))),
      duration_s=float(data.get("duration_s", data.get("durationS", 0.0))),
      active_percent=float(data.get("active_percent", data.get("activePercent", 0.0))),
      lane_state_unknown_percent=float(data.get("lane_state_unknown_percent", data.get("laneStateUnknownPercent", 0.0))),
      qlog_safe_lane_policy=bool(data.get("qlog_safe_lane_policy", data.get("qlogSafeLanePolicy", True))),
      dominant_failure_class=str(data.get("dominant_failure_class", data.get("dominantFailureClass", INSUFFICIENT_EVIDENCE))),
      branch_recommendation=str(data.get("branch_recommendation", data.get("branchRecommendation", "none"))),
      confidence=str(data.get("confidence", "low")),
      torque_event_score=float(data.get("torque_event_score", data.get("torqueEventScore", 0.0))),
      path_wander_score=float(data.get("path_wander_score", data.get("pathWanderScore", 0.0))),
      low_speed_score=float(data.get("low_speed_score", data.get("lowSpeedScore", 0.0))),
      torque_event_report=LateralTorqueEventReport.from_dict(data.get("torque_event_report", data.get("torqueEventReport", {}))),
      low_speed_report=_low_speed_report_from_dict(data.get("low_speed_report", data.get("lowSpeedReport", {}))),
      wander_candidate_windows=[_wander_window_from_dict(item) for item in data.get("wander_candidate_windows", data.get("wanderCandidateWindows", []))],
      recenter_overshoot_candidates=[
        _recenter_candidate_from_dict(item) for item in data.get("recenter_overshoot_candidates", data.get("recenterOvershootCandidates", []))
      ],
      notes=[str(note) for note in data.get("notes", [])],
    )


@dataclass(frozen=True)
class LateralPerformanceGateABReport:
  baseline: LateralPerformanceGateReport
  candidate: LateralPerformanceGateReport
  deltas: dict[str, float | None]

  def to_dict(self) -> dict[str, Any]:
    return asdict(self)


@dataclass(frozen=True)
class _GateSample:
  t: float
  v_ego: float
  lat_active: bool
  steering_pressed: bool
  blinker_active: bool
  lane_change_state: str
  steering_angle_deg: float
  curvature: float
  raw_desired_curvature: float
  processed_desired_curvature: float
  desired_curvature: float
  model_path_gated: bool
  model_path_quality: float
  model_path_offset_y: float
  lane_center_offset_y: float


def build_lateral_performance_gate(
  msgs: list[Any],
  source: str = "unknown",
  already_sorted: bool = False,
  qlog_safe_lane_policy: bool = True,
  window_s: float = 30.0,
  step_s: float = 5.0,
  max_wander_windows: int = 8,
  max_recenter_candidates: int = 8,
) -> LateralPerformanceGateReport:
  ordered_msgs = list(msgs) if already_sorted else sorted(msgs, key=lambda m: int(getattr(m, "logMonoTime", 0)))
  torque_event_report = build_lateral_torque_event_report(ordered_msgs, source=source, already_sorted=True)
  low_speed_report = build_lateral_low_speed_report(ordered_msgs, source=source, already_sorted=True)
  samples = _extract_gate_samples(ordered_msgs)
  if not samples:
    return LateralPerformanceGateReport(
      source, 0, 0.0, 0.0, 0.0, qlog_safe_lane_policy, INSUFFICIENT_EVIDENCE,
      BRANCH_RECOMMENDATIONS[INSUFFICIENT_EVIDENCE], "low", 0.0, 0.0, 0.0,
      torque_event_report, low_speed_report, [], [], ["no controlsState samples found"],
    )

  cols = _columns(samples)
  wander_windows = _rank_wander_windows(cols, qlog_safe_lane_policy, window_s, step_s, max_wander_windows)
  recenter_candidates = _rank_recenter_candidates(cols, qlog_safe_lane_policy, window_s, step_s, max_recenter_candidates)
  torque_score = _torque_event_score(torque_event_report)
  wander_score = max((window.severity_score for window in wander_windows), default=0.0)
  low_speed_score = _low_speed_score(low_speed_report)
  dominant, confidence = _dominant_failure_class(torque_score, wander_score, low_speed_score, wander_windows)
  notes = _notes(cols, qlog_safe_lane_policy, wander_windows, recenter_candidates)
  return LateralPerformanceGateReport(
    source=source,
    sample_count=len(samples),
    duration_s=float(cols["t"][-1] - cols["t"][0]) if len(samples) > 1 else 0.0,
    active_percent=_percent(cols["lat_active"] > 0.5),
    lane_state_unknown_percent=_percent(cols["lane_state_unknown"] > 0.5),
    qlog_safe_lane_policy=qlog_safe_lane_policy,
    dominant_failure_class=dominant,
    branch_recommendation=BRANCH_RECOMMENDATIONS[dominant],
    confidence=confidence,
    torque_event_score=torque_score,
    path_wander_score=wander_score,
    low_speed_score=low_speed_score,
    torque_event_report=torque_event_report,
    low_speed_report=low_speed_report,
    wander_candidate_windows=wander_windows,
    recenter_overshoot_candidates=recenter_candidates,
    notes=notes,
  )


def build_lateral_performance_gate_ab_report(
  baseline_msgs: list[Any],
  candidate_msgs: list[Any],
  baseline_source: str = "baseline",
  candidate_source: str = "candidate",
  already_sorted: bool = False,
  qlog_safe_lane_policy: bool = True,
) -> LateralPerformanceGateABReport:
  baseline = build_lateral_performance_gate(
    baseline_msgs, baseline_source, already_sorted=already_sorted, qlog_safe_lane_policy=qlog_safe_lane_policy,
  )
  candidate = build_lateral_performance_gate(
    candidate_msgs, candidate_source, already_sorted=already_sorted, qlog_safe_lane_policy=qlog_safe_lane_policy,
  )
  return LateralPerformanceGateABReport(baseline, candidate, _gate_deltas(baseline, candidate))


def render_lateral_performance_gate(report: LateralPerformanceGateReport) -> str:
  lines = [
    f"Lateral performance gate: {report.source}",
    f"samples: {report.sample_count}",
    f"duration: {report.duration_s:.1f} s",
    f"active: {report.active_percent:.1f}%",
    f"lane-state unknown: {report.lane_state_unknown_percent:.1f}%",
    f"dominant failure class: {report.dominant_failure_class} ({report.confidence})",
    f"branch recommendation: {report.branch_recommendation}",
    f"scores: torque={report.torque_event_score:.2f} wander={report.path_wander_score:.2f} low_speed={report.low_speed_score:.2f}",
  ]
  if report.wander_candidate_windows:
    lines.append("Wander candidate windows:")
    for window in report.wander_candidate_windows:
      lines.append(
        f"  {window.start_s:.1f}-{window.end_s:.1f}s cause={window.cause} confidence={window.confidence} "
        f"score={window.severity_score:.2f} steer_pp={window.steering_angle_pp:.3f}deg "
        f"actual_pp={window.actual_curvature_pp:.6f} raw_pp={window.raw_curvature_pp:.6f} "
        f"processed_pp={window.processed_curvature_pp:.6f} unknown_lane={window.lane_state_unknown_percent:.1f}%"
      )
  if report.recenter_overshoot_candidates:
    lines.append("Recenter overshoot candidates:")
    for candidate in report.recenter_overshoot_candidates:
      lines.append(
        f"  {candidate.start_s:.1f}-{candidate.end_s:.1f}s confidence={candidate.confidence} "
        f"offset_crossings={candidate.offset_crossings} correction_reversals={candidate.correction_reversals} "
        f"offset_agree={candidate.offset_agreement_percent:.1f}% combined_offset_pp={candidate.combined_offset_pp:.3f}"
      )
  if report.notes:
    lines.append("Notes:")
    lines.extend(f"  {note}" for note in report.notes)
  return "\n".join(lines)


def render_lateral_performance_gate_ab_report(report: LateralPerformanceGateABReport) -> str:
  lines = [
    "Lateral performance gate A/B report",
    render_lateral_performance_gate(report.baseline),
    render_lateral_performance_gate(report.candidate),
    "Deltas candidate-baseline:",
  ]
  for key, value in sorted(report.deltas.items()):
    rendered = "n/a" if value is None else f"{value:.3f}"
    lines.append(f"{key}: {rendered}")
  return "\n".join(lines)


def save_lateral_performance_gate(report: LateralPerformanceGateReport | LateralPerformanceGateABReport, path: str | Path) -> None:
  Path(path).write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")


def load_lateral_performance_gate(path: str | Path) -> LateralPerformanceGateReport:
  return LateralPerformanceGateReport.from_dict(json.loads(Path(path).read_text()))


def _extract_gate_samples(msgs: list[Any]) -> list[_GateSample]:
  if not msgs:
    return []
  base_mono_time = int(getattr(msgs[0], "logMonoTime", 0))
  latest: dict[str, Any] = {}
  samples: list[_GateSample] = []
  for msg in msgs:
    typ = msg_type(msg)
    payload = msg_payload(msg)
    if typ in ("carState", "carControl", "modelV2"):
      latest[typ] = payload
    if typ != "controlsState":
      continue

    car_state = latest.get("carState")
    car_control = latest.get("carControl")
    model_v2 = latest.get("modelV2")
    lateral_state = safe_get(payload, "lateralControlState")
    lateral_kind = format_enum(lateral_state.which()) if lateral_state is not None and hasattr(lateral_state, "which") else "torqueState"
    lateral_payload = safe_get(lateral_state, lateral_kind, lateral_state)
    model_path = safe_get(payload, "modelPathState")
    samples.append(_GateSample(
      t=msg_time_s(msg, base_mono_time),
      v_ego=_finite_float(safe_get(car_state, "vEgo")),
      lat_active=bool(safe_get(lateral_payload, "active", False)) and bool(safe_get(car_control, "latActive", False)),
      steering_pressed=bool(safe_get(car_state, "steeringPressed", False)),
      blinker_active=bool(safe_get(car_state, "leftBlinker", False)) or bool(safe_get(car_state, "rightBlinker", False)),
      lane_change_state=format_enum(safe_get(model_v2, "meta.laneChangeState")),
      steering_angle_deg=_finite_float(safe_get(car_state, "steeringAngleDeg")),
      curvature=_finite_float(safe_get(payload, "curvature")),
      raw_desired_curvature=_finite_float(safe_get(model_path, "rawDesiredCurvature")),
      processed_desired_curvature=_finite_float(safe_get(payload, "desiredCurvature")),
      desired_curvature=_finite_float(safe_get(payload, "desiredCurvature")),
      model_path_gated=bool(safe_get(model_path, "gated", False)),
      model_path_quality=_finite_float(safe_get(model_path, "quality")),
      model_path_offset_y=_model_path_offset_y(model_v2),
      lane_center_offset_y=_lane_center_offset_y(model_v2),
    ))
  return samples


def _columns(samples: list[_GateSample]) -> dict[str, np.ndarray]:
  return {
    "t": np.array([sample.t for sample in samples], dtype=float),
    "v_ego": np.array([sample.v_ego for sample in samples], dtype=float),
    "lat_active": np.array([float(sample.lat_active) for sample in samples], dtype=float),
    "steering_pressed": np.array([float(sample.steering_pressed) for sample in samples], dtype=float),
    "blinker_active": np.array([float(sample.blinker_active) for sample in samples], dtype=float),
    "lane_change_state": np.array([sample.lane_change_state for sample in samples], dtype=object),
    "lane_state_unknown": np.array([float(sample.lane_change_state == "unknown") for sample in samples], dtype=float),
    "steering_angle_deg": np.array([sample.steering_angle_deg for sample in samples], dtype=float),
    "curvature": np.array([sample.curvature for sample in samples], dtype=float),
    "raw_desired_curvature": np.array([sample.raw_desired_curvature for sample in samples], dtype=float),
    "processed_desired_curvature": np.array([sample.processed_desired_curvature for sample in samples], dtype=float),
    "desired_curvature": np.array([sample.desired_curvature for sample in samples], dtype=float),
    "model_path_gated": np.array([float(sample.model_path_gated) for sample in samples], dtype=float),
    "model_path_quality": np.array([sample.model_path_quality for sample in samples], dtype=float),
    "model_path_offset_y": np.array([sample.model_path_offset_y for sample in samples], dtype=float),
    "lane_center_offset_y": np.array([sample.lane_center_offset_y for sample in samples], dtype=float),
  }


def _rank_wander_windows(cols: dict[str, np.ndarray], qlog_safe_lane_policy: bool, window_s: float, step_s: float,
                         max_windows: int) -> list[WanderCandidateWindow]:
  candidates: list[WanderCandidateWindow] = []
  base = _broad_straight_mask(cols, qlog_safe_lane_policy)
  for mask in _window_masks(cols["t"], window_s, step_s):
    if not np.any(mask) or float(np.mean(base[mask])) < 0.70:
      continue
    idx = mask & base
    if int(np.sum(idx)) < 20:
      continue
    candidate = _wander_candidate(cols, idx)
    if candidate.severity_score <= 0.0:
      continue
    candidates.append(candidate)
  candidates.sort(key=lambda item: item.severity_score, reverse=True)
  return candidates[:max_windows]


def _rank_recenter_candidates(cols: dict[str, np.ndarray], qlog_safe_lane_policy: bool, window_s: float, step_s: float,
                              max_candidates: int) -> list[RecenterOvershootCandidate]:
  candidates: list[RecenterOvershootCandidate] = []
  base = _broad_straight_mask(cols, qlog_safe_lane_policy)
  for mask in _window_masks(cols["t"], window_s, step_s):
    if not np.any(mask) or float(np.mean(base[mask])) < 0.70:
      continue
    idx = mask & base
    if int(np.sum(idx)) < 20:
      continue
    candidate = _recenter_candidate(cols, idx)
    if candidate is not None:
      candidates.append(candidate)
  candidates.sort(key=lambda item: (item.confidence == "high", item.offset_crossings, item.combined_offset_pp), reverse=True)
  return candidates[:max_candidates]


def _broad_straight_mask(cols: dict[str, np.ndarray], qlog_safe_lane_policy: bool) -> np.ndarray:
  lane_state = cols["lane_change_state"]
  lane_ok = lane_state == "off"
  if qlog_safe_lane_policy:
    lane_ok = lane_ok | (lane_state == "unknown")
  return (
    (cols["lat_active"] > 0.5)
    & (cols["v_ego"] >= BROAD_STRAIGHT_MIN_SPEED)
    & (cols["steering_pressed"] < 0.5)
    & (cols["blinker_active"] < 0.5)
    & lane_ok
    & (np.abs(cols["raw_desired_curvature"]) <= BROAD_STRAIGHT_MAX_RAW_CURVATURE)
    & np.isfinite(cols["raw_desired_curvature"])
    & np.isfinite(cols["processed_desired_curvature"])
    & np.isfinite(cols["curvature"])
  )


def _wander_candidate(cols: dict[str, np.ndarray], idx: np.ndarray) -> WanderCandidateWindow:
  steering_pp = _percentile_span(cols["steering_angle_deg"][idx])
  actual_pp = _percentile_span(cols["curvature"][idx])
  raw_pp = _percentile_span(cols["raw_desired_curvature"][idx])
  processed_pp = _percentile_span(cols["processed_desired_curvature"][idx])
  desired_pp = _percentile_span(cols["desired_curvature"][idx])
  raw_actual_corr = _correlation(cols["raw_desired_curvature"][idx], cols["curvature"][idx])
  processed_actual_corr = _correlation(cols["processed_desired_curvature"][idx], cols["curvature"][idx])
  severity = 0.0
  if steering_pp >= WANDER_MIN_STEERING_PP or actual_pp >= WANDER_MIN_ACTUAL_CURVATURE_PP:
    severity = steering_pp * 1.2 + actual_pp * 3500.0 + min(raw_pp, processed_pp) * 1200.0
  cause = _wander_cause(raw_pp, processed_pp, actual_pp, raw_actual_corr, processed_actual_corr)
  unknown_percent = _percent(cols["lane_state_unknown"][idx] > 0.5)
  confidence = _wander_confidence(cause, severity, unknown_percent)
  return WanderCandidateWindow(
    start_s=float(np.min(cols["t"][idx])),
    end_s=float(np.max(cols["t"][idx])),
    sample_count=int(np.sum(idx)),
    confidence=confidence,
    cause=cause,
    severity_score=float(severity),
    speed_mps_median=_median(cols["v_ego"][idx]),
    steering_angle_pp=steering_pp,
    actual_curvature_pp=actual_pp,
    raw_curvature_pp=raw_pp,
    processed_curvature_pp=processed_pp,
    desired_curvature_pp=desired_pp,
    raw_actual_corr=raw_actual_corr,
    processed_actual_corr=processed_actual_corr,
    gated_percent=_percent(cols["model_path_gated"][idx] > 0.5),
    quality_median=_median(cols["model_path_quality"][idx]),
    lane_state_unknown_percent=unknown_percent,
  )


def _recenter_candidate(cols: dict[str, np.ndarray], idx: np.ndarray) -> RecenterOvershootCandidate | None:
  model_offset = cols["model_path_offset_y"][idx]
  lane_offset = cols["lane_center_offset_y"][idx]
  processed = cols["processed_desired_curvature"][idx]
  good = np.isfinite(model_offset) & np.isfinite(lane_offset) & np.isfinite(processed)
  if int(np.sum(good)) < 20:
    return None

  model_good = model_offset[good]
  lane_good = lane_offset[good]
  processed_good = processed[good]
  model_sign = np.sign(model_good[np.abs(model_good) > RECENTER_OFFSET_EPS])
  lane_sign = np.sign(lane_good[np.abs(lane_good) > RECENTER_OFFSET_EPS])
  if model_sign.size == 0 or lane_sign.size == 0:
    return None
  same_sign = np.sign(model_good) == np.sign(lane_good)
  strong_offset = (np.abs(model_good) > RECENTER_OFFSET_EPS) & (np.abs(lane_good) > RECENTER_OFFSET_EPS)
  strong_count = int(np.sum(strong_offset))
  if strong_count == 0:
    return None
  offset_agreement = 100.0 * float(np.sum(same_sign & strong_offset)) / strong_count
  combined = np.where(same_sign, (model_good + lane_good) / 2.0, np.nan)
  combined_good = combined[np.isfinite(combined)]
  if combined_good.size < 20:
    return None

  offset_crossings = _sign_flip_count(combined_good, RECENTER_OFFSET_EPS)
  correction_reversals = _sign_flip_count(processed_good, RECENTER_CURVATURE_EPS)
  model_pp = _percentile_span(model_good)
  lane_pp = _percentile_span(lane_good)
  combined_pp = _percentile_span(combined_good)
  processed_pp = _percentile_span(processed_good)
  if offset_crossings < 1 or correction_reversals < 1 or combined_pp < RECENTER_OFFSET_EPS * 2.0:
    return None
  unknown_percent = _percent(cols["lane_state_unknown"][idx] > 0.5)
  confidence = "high" if offset_agreement >= 70.0 and unknown_percent <= 20.0 else ("medium" if offset_agreement >= 50.0 else "low")
  return RecenterOvershootCandidate(
    start_s=float(np.min(cols["t"][idx])),
    end_s=float(np.max(cols["t"][idx])),
    sample_count=int(np.sum(good)),
    confidence=confidence,
    offset_agreement_percent=offset_agreement,
    model_path_offset_pp=model_pp,
    lane_center_offset_pp=lane_pp,
    combined_offset_pp=combined_pp,
    offset_crossings=offset_crossings,
    correction_reversals=correction_reversals,
    processed_curvature_pp=processed_pp,
    lane_state_unknown_percent=unknown_percent,
  )


def _wander_cause(raw_pp: float, processed_pp: float, actual_pp: float, raw_actual_corr: float | None,
                  processed_actual_corr: float | None) -> str:
  demand_moves = max(raw_pp, processed_pp) >= WANDER_MIN_DEMAND_CURVATURE_PP
  actual_moves = actual_pp >= WANDER_MIN_ACTUAL_CURVATURE_PP
  demand_tracks_actual = (
    (raw_actual_corr is not None and raw_actual_corr >= WANDER_STRONG_CORR)
    or (processed_actual_corr is not None and processed_actual_corr >= WANDER_STRONG_CORR)
  )
  if demand_moves and actual_moves and demand_tracks_actual:
    return DEMAND_DRIVEN_WANDER
  if not demand_moves and actual_moves:
    return ACTUATION_DRIVEN_WANDER
  return MIXED_WANDER


def _wander_confidence(cause: str, severity: float, unknown_percent: float) -> str:
  if severity <= 0.0:
    return "low"
  if cause == MIXED_WANDER:
    return "medium" if unknown_percent <= 50.0 else "low"
  if unknown_percent > 80.0:
    return "medium"
  if unknown_percent > 20.0:
    return "medium"
  return "high"


def _torque_event_score(report: LateralTorqueEventReport) -> float:
  if not report.top_events:
    return 0.0
  top = max(event.score for event in report.top_events)
  severe = sum(1 for event in report.top_events if event.score >= 40.0)
  return float(top + max(0, severe - 1) * 3.0)


def _low_speed_score(report: LateralLowSpeedReport) -> float:
  scores: list[float] = []
  for metric in report.tiers:
    if metric.sample_count < 30:
      continue
    scores.append(
      metric.abs_error_p95 * 80.0
      + metric.steer_limited_percent * 0.25
      + metric.output_reversals * 0.35
      + metric.high_steering_rate_percent * 0.6
    )
  return float(max(scores, default=0.0))


def _dominant_failure_class(torque_score: float, wander_score: float, low_speed_score: float,
                            wander_windows: list[WanderCandidateWindow]) -> tuple[str, str]:
  if torque_score >= 35.0 and torque_score >= wander_score * 1.35 and torque_score >= low_speed_score * 1.10:
    return TORQUE_EVENT_DOMINANT, "high" if torque_score >= 50.0 else "medium"
  if wander_score >= 18.0 and wander_score >= torque_score * 0.75 and wander_score >= low_speed_score * 1.10:
    confidence = wander_windows[0].confidence if wander_windows else "low"
    return PATH_WANDER_DOMINANT, confidence
  if low_speed_score >= 35.0 and low_speed_score >= torque_score * 0.80 and low_speed_score >= wander_score:
    return LOW_SPEED_LATERAL_DOMINANT, "medium"
  return INSUFFICIENT_EVIDENCE, "low"


def _notes(cols: dict[str, np.ndarray], qlog_safe_lane_policy: bool, wander_windows: list[WanderCandidateWindow],
           recenter_candidates: list[RecenterOvershootCandidate]) -> list[str]:
  notes: list[str] = []
  unknown_percent = _percent(cols["lane_state_unknown"] > 0.5)
  if qlog_safe_lane_policy and unknown_percent > 20.0:
    notes.append("qlog-safe lane policy used; unknown lane-change samples require no steering override and no blinkers")
  if any(window.confidence == "medium" and window.lane_state_unknown_percent > 20.0 for window in wander_windows):
    notes.append("path-wander classification from qlog-like lane-state evidence is medium confidence")
  if recenter_candidates:
    notes.append("recenter overshoot candidates are validation evidence only, not behavior authorization")
  return notes


def _gate_deltas(baseline: LateralPerformanceGateReport, candidate: LateralPerformanceGateReport) -> dict[str, float | None]:
  return {
    "torque_event_score": candidate.torque_event_score - baseline.torque_event_score,
    "path_wander_score": candidate.path_wander_score - baseline.path_wander_score,
    "low_speed_score": candidate.low_speed_score - baseline.low_speed_score,
    "active_percent": candidate.active_percent - baseline.active_percent,
    "lane_state_unknown_percent": candidate.lane_state_unknown_percent - baseline.lane_state_unknown_percent,
  }


def _window_masks(t: np.ndarray, window_s: float, step_s: float) -> list[np.ndarray]:
  if t.size == 0:
    return []
  masks: list[np.ndarray] = []
  cur = float(t[0])
  end = float(t[-1])
  while cur + window_s <= end:
    masks.append((t >= cur) & (t < cur + window_s))
    cur += step_s
  return masks


def _model_path_offset_y(model_v2: Any) -> float:
  values = safe_get(model_v2, "position.y")
  if values is None:
    return float("nan")
  try:
    idx = min(5, len(values) - 1)
    if idx < 0:
      return float("nan")
    return _finite_float(values[idx])
  except (TypeError, IndexError):
    return float("nan")


def _lane_center_offset_y(model_v2: Any) -> float:
  lane_lines = safe_get(model_v2, "laneLines")
  left = _lane_line_y(lane_lines, 1, 5)
  right = _lane_line_y(lane_lines, 2, 5)
  if not isfinite(left) or not isfinite(right):
    return float("nan")
  return (left + right) / 2.0


def _lane_line_y(lane_lines: Any, idx: int, horizon_idx: int) -> float:
  try:
    lane_line = lane_lines[idx]
    values = getattr(lane_line, "y")
    value_idx = min(horizon_idx, len(values) - 1)
    if value_idx < 0:
      return float("nan")
    return _finite_float(values[value_idx])
  except (TypeError, IndexError, AttributeError):
    return float("nan")


def _finite_float(value: Any, default: float = np.nan) -> float:
  try:
    candidate = float(value)
  except (TypeError, ValueError):
    return default
  return candidate if isfinite(candidate) else default


def _percentile_span(values: np.ndarray) -> float:
  finite = values[np.isfinite(values)]
  if finite.size == 0:
    return 0.0
  return float(np.percentile(finite, 95.0) - np.percentile(finite, 5.0))


def _median(values: np.ndarray) -> float:
  finite = values[np.isfinite(values)]
  return float(np.median(finite)) if finite.size else 0.0


def _percent(mask: np.ndarray) -> float:
  return float(100.0 * np.mean(mask)) if mask.size else 0.0


def _correlation(left: np.ndarray, right: np.ndarray) -> float | None:
  good = np.isfinite(left) & np.isfinite(right)
  if int(np.sum(good)) < 5:
    return None
  left_good = left[good]
  right_good = right[good]
  if float(np.std(left_good)) <= 1e-12 or float(np.std(right_good)) <= 1e-12:
    return None
  return float(np.corrcoef(left_good, right_good)[0, 1])


def _sign_flip_count(values: np.ndarray, eps: float) -> int:
  good = np.isfinite(values) & (np.abs(values) > eps)
  signs = np.sign(values[good])
  return int(np.sum(signs[1:] != signs[:-1])) if signs.size > 1 else 0


def _wander_window_from_dict(data: dict[str, Any]) -> WanderCandidateWindow:
  return WanderCandidateWindow(
    start_s=float(data.get("start_s", data.get("startS", 0.0))),
    end_s=float(data.get("end_s", data.get("endS", 0.0))),
    sample_count=int(data.get("sample_count", data.get("sampleCount", 0))),
    confidence=str(data.get("confidence", "low")),
    cause=str(data.get("cause", MIXED_WANDER)),
    severity_score=float(data.get("severity_score", data.get("severityScore", 0.0))),
    speed_mps_median=float(data.get("speed_mps_median", data.get("speedMpsMedian", 0.0))),
    steering_angle_pp=float(data.get("steering_angle_pp", data.get("steeringAnglePp", 0.0))),
    actual_curvature_pp=float(data.get("actual_curvature_pp", data.get("actualCurvaturePp", 0.0))),
    raw_curvature_pp=float(data.get("raw_curvature_pp", data.get("rawCurvaturePp", 0.0))),
    processed_curvature_pp=float(data.get("processed_curvature_pp", data.get("processedCurvaturePp", 0.0))),
    desired_curvature_pp=float(data.get("desired_curvature_pp", data.get("desiredCurvaturePp", 0.0))),
    raw_actual_corr=_optional_float(data.get("raw_actual_corr", data.get("rawActualCorr"))),
    processed_actual_corr=_optional_float(data.get("processed_actual_corr", data.get("processedActualCorr"))),
    gated_percent=float(data.get("gated_percent", data.get("gatedPercent", 0.0))),
    quality_median=float(data.get("quality_median", data.get("qualityMedian", 0.0))),
    lane_state_unknown_percent=float(data.get("lane_state_unknown_percent", data.get("laneStateUnknownPercent", 0.0))),
  )


def _recenter_candidate_from_dict(data: dict[str, Any]) -> RecenterOvershootCandidate:
  return RecenterOvershootCandidate(
    start_s=float(data.get("start_s", data.get("startS", 0.0))),
    end_s=float(data.get("end_s", data.get("endS", 0.0))),
    sample_count=int(data.get("sample_count", data.get("sampleCount", 0))),
    confidence=str(data.get("confidence", "low")),
    offset_agreement_percent=float(data.get("offset_agreement_percent", data.get("offsetAgreementPercent", 0.0))),
    model_path_offset_pp=float(data.get("model_path_offset_pp", data.get("modelPathOffsetPp", 0.0))),
    lane_center_offset_pp=float(data.get("lane_center_offset_pp", data.get("laneCenterOffsetPp", 0.0))),
    combined_offset_pp=float(data.get("combined_offset_pp", data.get("combinedOffsetPp", 0.0))),
    offset_crossings=int(data.get("offset_crossings", data.get("offsetCrossings", 0))),
    correction_reversals=int(data.get("correction_reversals", data.get("correctionReversals", 0))),
    processed_curvature_pp=float(data.get("processed_curvature_pp", data.get("processedCurvaturePp", 0.0))),
    lane_state_unknown_percent=float(data.get("lane_state_unknown_percent", data.get("laneStateUnknownPercent", 0.0))),
  )


def _optional_float(value: Any) -> float | None:
  if value is None:
    return None
  candidate = _finite_float(value)
  return candidate if isfinite(candidate) else None


def _low_speed_report_from_dict(data: dict[str, Any]) -> LateralLowSpeedReport:
  from openpilot.tools.drive_lab.lateral_torque_event_report import LateralLowSpeedTierMetrics

  def tier(item: dict[str, Any]) -> LateralLowSpeedTierMetrics:
    return LateralLowSpeedTierMetrics(
      segment=str(item.get("segment", "unknown")),
      speed_lower_mps=float(item.get("speed_lower_mps", item.get("speedLowerMps", 0.0))),
      speed_upper_mps=float(item.get("speed_upper_mps", item.get("speedUpperMps", 0.0))),
      sample_count=int(item.get("sample_count", item.get("sampleCount", 0))),
      best_lag_s=_optional_float(item.get("best_lag_s", item.get("bestLagS"))),
      desired_actual_corr=_optional_float(item.get("desired_actual_corr", item.get("desiredActualCorr"))),
      abs_error_mean=float(item.get("abs_error_mean", item.get("absErrorMean", 0.0))),
      abs_error_p95=float(item.get("abs_error_p95", item.get("absErrorP95", 0.0))),
      output_reversals=int(item.get("output_reversals", item.get("outputReversals", 0))),
      unshaped_output_reversals=int(item.get("unshaped_output_reversals", item.get("unshapedOutputReversals", 0))),
      desired_lateral_accel_reversals=int(item.get("desired_lateral_accel_reversals", item.get("desiredLateralAccelReversals", 0))),
      actual_lateral_accel_reversals=int(item.get("actual_lateral_accel_reversals", item.get("actualLateralAccelReversals", 0))),
      steering_rate_p95=float(item.get("steering_rate_p95", item.get("steeringRateP95", 0.0))),
      steer_limited_percent=float(item.get("steer_limited_percent", item.get("steerLimitedPercent", 0.0))),
      high_steering_rate_percent=float(item.get("high_steering_rate_percent", item.get("highSteeringRatePercent", 0.0))),
      raw_processed_curvature_delta_p95=float(item.get("raw_processed_curvature_delta_p95", item.get("rawProcessedCurvatureDeltaP95", 0.0))),
      desired_processed_curvature_delta_p95=float(item.get("desired_processed_curvature_delta_p95", item.get("desiredProcessedCurvatureDeltaP95", 0.0))),
      model_path_gated_percent=float(item.get("model_path_gated_percent", item.get("modelPathGatedPercent", 0.0))),
      model_path_quality_median=float(item.get("model_path_quality_median", item.get("modelPathQualityMedian", 0.0))),
      model_path_reason_counts={str(k): int(v) for k, v in item.get("model_path_reason_counts", item.get("modelPathReasonCounts", {})).items()},
    )

  return LateralLowSpeedReport(
    source=str(data.get("source", "unknown")),
    sample_count=int(data.get("sample_count", data.get("sampleCount", 0))),
    duration_s=float(data.get("duration_s", data.get("durationS", 0.0))),
    lane_change_excluded_count=int(data.get("lane_change_excluded_count", data.get("laneChangeExcludedCount", 0))),
    signal_tagged_category_counts={str(k): int(v) for k, v in data.get("signal_tagged_category_counts", data.get("signalTaggedCategoryCounts", {})).items()},
    signal_tagged_state_counts={str(k): int(v) for k, v in data.get("signal_tagged_state_counts", data.get("signalTaggedStateCounts", {})).items()},
    tiers=[tier(item) for item in data.get("tiers", [])],
    signal_tagged_tiers=[tier(item) for item in data.get("signal_tagged_tiers", data.get("signalTaggedTiers", []))],
  )
