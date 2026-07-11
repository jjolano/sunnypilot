#!/usr/bin/env python3
"""Compare manual-vs-model and engaged controller lateral timing.

Segments curve events from a reference lateral-accel signal and reports timing
and magnitude differences between the reference (model/controller desired) and
the actual lateral response. Manual events compare human steering to the model
plan; engaged events evaluate controller tracking.

Run:
  uv run python -m openpilot.tools.drive_lab.compare_manual_lateral_timing ROUTE --qlog
  uv run python -m openpilot.tools.drive_lab.compare_manual_lateral_timing FILES --rlog --json
"""
from __future__ import annotations

import argparse
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from openpilot.tools.drive_lab.route_analysis import build_route_messages, finite_or_none
from openpilot.tools.drive_lab.route_io import output_report
from openpilot.tools.drive_lab.timeline import format_enum, safe_get


SCHEMA = "drive-lab-compare-manual-lateral-timing"
SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EventDetectionParams:
  onset_threshold: float = 0.25            # m/s^2
  release_threshold: float = 0.12          # m/s^2
  min_duration_s: float = 0.30             # s
  max_gap_s: float = 0.20                  # s
  pre_window_s: float = 1.0                # s before reference onset
  post_window_s: float = 1.5               # s after reference release
  min_v_ego: float = 5.0                   # m/s
  stale_model_threshold_s: float = 0.25    # s
  max_roll_abs: float = 0.05               # rad
  underresponse_min_deficit: float = 0.10  # m/s^2
  underresponse_min_peak_deficit: float = 0.15  # m/s^2
  underresponse_max_peak_ratio: float = 0.95
  underresponse_max_area_ratio: float = 0.95
  overresponse_min_reference_peak: float = 0.50
  overresponse_min_mean_overshoot: float = 0.08
  overresponse_min_peak_overshoot: float = 0.15
  overresponse_min_peak_ratio: float = 1.10
  overresponse_min_area_ratio: float = 1.05
  tracking_rms_threshold: float = 0.30     # m/s^2

  def to_dict(self) -> dict[str, Any]:
    return asdict(self)


# ---------------------------------------------------------------------------
# Frame / event / report data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LateralTimingFrame:
  """Aligned state at a single controlsState timestamp.

  The frame is intentionally easy to construct from synthetic data for tests.
  """

  t: float
  route: str = "synthetic"
  mode: str = "unknown"  # "manual", "engaged", "inactive", "unknown"
  v_ego: float = 0.0
  steering_angle_deg: float | None = None
  steering_rate_deg_s: float | None = None
  steering_torque: float | None = None
  roll: float | None = None
  model_raw_lat_accel: float | None = None
  processed_lat_accel: float | None = None
  control_desired_lat_accel: float | None = None
  controller_desired_lat_accel: float | None = None
  controller_actual_lat_accel: float | None = None
  actual_lat_accel: float | None = None
  path_quality: float | None = None
  path_gated: bool | None = None
  path_reason: str | None = None
  model_age_s: float | None = None
  steering_pressed: bool = False
  left_blinker: bool = False
  right_blinker: bool = False
  lane_change_state: str = "off"
  standstill: bool = False
  lat_active: bool = False
  lateral_state_active: bool = False
  saturated: bool | None = None
  underresponse_active: bool | None = None
  underresponse_eligible: bool | None = None
  underresponse_block: bool | None = None
  underresponse_shadow: float | None = None
  governor_reason: int | None = None
  controller_output: float | None = None


@dataclass(frozen=True)
class LateralTimingEvent:
  route: str
  start_t: float
  end_t: float
  mode: str
  sign: float
  reference_source: str
  reference_onset_t: float | None
  reference_peak_t: float | None
  reference_release_t: float | None
  reference_peak: float | None
  reference_area: float | None
  actual_onset_t: float | None
  actual_peak_t: float | None
  actual_release_t: float | None
  actual_peak: float | None
  actual_area: float | None
  onset_delta_s: float | None
  peak_delta_s: float | None
  peak_ratio: float | None
  area_ratio: float | None
  rms_tracking_error: float | None
  classifications: list[str]
  gates_passed: bool
  gate_reasons: list[str]
  missing_model_v2: bool
  sample_count: int

  def to_dict(self) -> dict[str, Any]:
    return _sanitize({
      **asdict(self),
      "classifications": list(self.classifications),
      "gate_reasons": list(self.gate_reasons),
    })


@dataclass
class CoverageCounters:
  total_frames: int = 0
  manual_frames: int = 0
  engaged_frames: int = 0
  inactive_frames: int = 0
  unknown_frames: int = 0
  excluded_frames: int = 0
  missing_model_v2_frames: int = 0

  def to_dict(self) -> dict[str, Any]:
    return asdict(self)


@dataclass
class LateralTimingReport:
  schema: str
  version: int
  source: str
  parameters: EventDetectionParams
  coverage: CoverageCounters
  summary: dict[str, Any]
  events: list[LateralTimingEvent]
  notes: list[str]

  def to_dict(self) -> dict[str, Any]:
    return {
      "schema": self.schema,
      "version": self.version,
      "source": self.source,
      "parameters": self.parameters.to_dict(),
      "coverage": self.coverage.to_dict(),
      "summary": _sanitize(self.summary),
      "events": [e.to_dict() for e in self.events],
      "notes": list(self.notes),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sanitize(value: Any) -> Any:
  if isinstance(value, float):
    return None if not math.isfinite(value) else round(value, 9)
  if isinstance(value, dict):
    return {k: _sanitize(v) for k, v in value.items()}
  if isinstance(value, list | tuple):
    return [_sanitize(v) for v in value]
  return value


def _r(value: float | None, ndigits: int = 3) -> float | None:
  if value is None or not isinstance(value, int | float) or not math.isfinite(float(value)):
    return None
  return round(float(value), ndigits)


def _lat_accel(v_ego: float, curvature: float | None) -> float | None:
  if v_ego is None or curvature is None:
    return None
  return curvature * v_ego * v_ego


def _reference_value_and_source(frame: LateralTimingFrame) -> tuple[float | None, str, bool]:
  """Return (reference lat accel, source label, missing_model_v2 flag)."""
  if frame.mode == "engaged":
    if frame.controller_desired_lat_accel is not None and math.isfinite(frame.controller_desired_lat_accel):
      return frame.controller_desired_lat_accel, "controller_desired", False
    if frame.model_raw_lat_accel is not None and math.isfinite(frame.model_raw_lat_accel):
      return frame.model_raw_lat_accel, "model_raw", True
    if frame.processed_lat_accel is not None and math.isfinite(frame.processed_lat_accel):
      return frame.processed_lat_accel, "processed", True
    return None, "none", True

  # manual
  missing_model = frame.model_raw_lat_accel is None
  if frame.model_raw_lat_accel is not None and math.isfinite(frame.model_raw_lat_accel):
    return frame.model_raw_lat_accel, "model_raw", False
  if frame.processed_lat_accel is not None and math.isfinite(frame.processed_lat_accel):
    return frame.processed_lat_accel, "processed", missing_model
  if frame.control_desired_lat_accel is not None and math.isfinite(frame.control_desired_lat_accel):
    return frame.control_desired_lat_accel, "control_desired", missing_model
  return None, "none", missing_model


def _actual_value(frame: LateralTimingFrame) -> float | None:
  if frame.mode == "engaged" and frame.controller_actual_lat_accel is not None and math.isfinite(frame.controller_actual_lat_accel):
    return frame.controller_actual_lat_accel
  if frame.actual_lat_accel is not None and math.isfinite(frame.actual_lat_accel):
    return frame.actual_lat_accel
  return None


def _sign(value: float | None) -> float:
  if value is None:
    return 0.0
  return 1.0 if value > 0 else -1.0


def _segment_frames(frames: Sequence[LateralTimingFrame], params: EventDetectionParams) -> list[list[LateralTimingFrame]]:
  """Split frames into contiguous segments with same mode and sign, bounded by max_gap."""
  segments: list[list[LateralTimingFrame]] = []
  current: list[LateralTimingFrame] = []
  last_t: float | None = None

  for frame in frames:
    ref, _, _ = _reference_value_and_source(frame)
    if ref is None or not math.isfinite(ref):
      if current:
        segments.append(current)
        current = []
      last_t = None
      continue

    frame_sign = _sign(ref)
    if current:
      prev = current[-1]
      prev_ref, _, _ = _reference_value_and_source(prev)
      gap = frame.t - last_t if last_t is not None else 0.0
      route_changed = prev.route != frame.route
      mode_changed = prev.mode != frame.mode
      sign_changed = prev_ref is None or _sign(prev_ref) != frame_sign
      gap_too_big = gap > params.max_gap_s
      if route_changed or mode_changed or sign_changed or gap_too_big:
        segments.append(current)
        current = [frame]
      else:
        current.append(frame)
    else:
      current = [frame]
    last_t = frame.t

  if current:
    segments.append(current)
  return segments


def _detect_event_indices(segment: list[LateralTimingFrame], params: EventDetectionParams) -> list[tuple[int, int]]:
  """Return start/end indices within the segment for each event."""
  indices: list[tuple[int, int]] = []
  start_idx: int | None = None

  for i, frame in enumerate(segment):
    ref, _, _ = _reference_value_and_source(frame)
    if ref is None or not math.isfinite(ref):
      continue
    abs_ref = abs(ref)
    if start_idx is None:
      if abs_ref >= params.onset_threshold:
        start_idx = i
    else:
      if abs_ref < params.release_threshold:
        # active region is start_idx .. i-1
        if i - 1 >= start_idx:
          duration = segment[i - 1].t - segment[start_idx].t
          if duration >= params.min_duration_s:
            indices.append((start_idx, i - 1))
        start_idx = None

  if start_idx is not None and len(segment) - 1 >= start_idx:
    duration = segment[-1].t - segment[start_idx].t
    if duration >= params.min_duration_s:
      indices.append((start_idx, len(segment) - 1))

  return indices


def _trapezoid_area(ts: np.ndarray, ys: np.ndarray) -> float:
  if ts.size < 2 or ys.size < 2:
    return 0.0
  finite = np.isfinite(ts) & np.isfinite(ys)
  indices = np.flatnonzero(finite)
  if indices.size < 2:
    return 0.0
  finite_ts = ts[finite]
  finite_ys = ys[finite]
  first, last = int(indices[0]), int(indices[-1])
  ts = ts[first:last + 1]
  ys = np.interp(ts, finite_ts, finite_ys)
  dts = np.diff(ts)
  valid_dt = np.isfinite(dts) & (dts > 0.0)
  return float(np.sum(dts[valid_dt] * 0.5 * (ys[:-1][valid_dt] + ys[1:][valid_dt])))


def _gate_event(event_frames: list[LateralTimingFrame], params: EventDetectionParams, missing_model: bool) -> tuple[bool, list[str]]:
  reasons: list[str] = []

  for frame in event_frames:
    if frame.left_blinker or frame.right_blinker:
      reasons.append("blinker")
    if frame.lane_change_state not in ("", "off", "unknown", "preLaneChange"):
      reasons.append("lane_change")
    if frame.standstill:
      reasons.append("standstill")
    if frame.v_ego < params.min_v_ego:
      reasons.append("low_speed")
    if frame.model_age_s is not None and frame.model_age_s > params.stale_model_threshold_s:
      reasons.append("stale_model")
    if frame.path_gated:
      reasons.append("path_gated")
    if frame.path_reason is not None and frame.path_reason.lower() not in {"ok", "clean", "active", "none", "disabled", ""}:
      reasons.append("path_reason_non_ok")

  # `missing_model` is reported on the event/coverage, but it is not a hard gate
  # when a fallback reference signal is available. This keeps qlog/processed-curvature
  # analysis useful while making the weaker evidence explicit.

  if not event_frames:
    reasons.append("no_frames")

  # Unique while preserving order.
  unique = list(dict.fromkeys(reasons))
  return (not unique, unique)


def _classify_event(
  event_frames: list[LateralTimingFrame],
  metrics: dict[str, Any],
  params: EventDetectionParams,
) -> list[str]:
  mode = metrics["mode"]
  gates_passed = metrics["gates_passed"]
  labels: list[str] = []

  if not gates_passed:
    labels.append("invalid_gate")
    # Keep conservative: do not also apply under/overresponse classifications.
    return labels

  ref_onset = metrics["reference_onset_t"]
  actual_onset = metrics["actual_onset_t"]
  ref_peak = metrics["reference_peak"]
  actual_peak = metrics["actual_peak"]
  rms = metrics["rms_tracking_error"]
  peak_ratio = metrics.get("peak_ratio")
  area_ratio = metrics.get("area_ratio")

  if mode == "manual":
    if actual_onset is not None and ref_onset is not None:
      delta = actual_onset - ref_onset
      if delta > 0.05:
        labels.append("manual_human_later")
      elif delta < -0.05:
        labels.append("manual_human_earlier")
    if actual_peak is not None and ref_peak is not None and ref_peak != 0:
      ratio = abs(actual_peak) / abs(ref_peak)
      if ratio < 0.7:
        labels.append("manual_human_lower_peak")
      elif ratio > 1.3:
        labels.append("manual_human_higher_peak")
    if actual_peak is not None and abs(actual_peak) < params.release_threshold and (ref_peak is not None and abs(ref_peak) >= params.onset_threshold):
      labels.append("manual_model_only")
    return labels

  if mode != "engaged":
    labels.append("invalid_mode")
    return labels

  # Engaged diagnostics.
  deficits: list[float] = []
  overshoots: list[float] = []
  for frame in event_frames:
    ref, _, _ = _reference_value_and_source(frame)
    actual = _actual_value(frame)
    if ref is None or actual is None or not math.isfinite(ref) or not math.isfinite(actual):
      continue
    if _sign(ref) != _sign(actual):
      continue
    diff = ref - actual
    if diff > 0:
      deficits.append(diff)
    elif diff < 0:
      overshoots.append(-diff)

  mean_deficit = float(np.mean(deficits)) if deficits else 0.0
  peak_deficit = max(deficits) if deficits else 0.0
  mean_overshoot = float(np.mean(overshoots)) if overshoots else 0.0
  peak_overshoot = max(overshoots) if overshoots else 0.0

  # Underresponse candidate: conservative gate set.
  any_steering_pressed = any(f.steering_pressed for f in event_frames)
  any_saturated = any(f.saturated is True for f in event_frames)
  high_roll = any(f.roll is not None and abs(f.roll) >= params.max_roll_abs for f in event_frames)

  is_underresponse = (
    not any_steering_pressed
    and not any_saturated
    and not high_roll
    and mean_deficit > params.underresponse_min_deficit
    and peak_deficit > params.underresponse_min_peak_deficit
    and peak_ratio is not None
    and area_ratio is not None
    and peak_ratio <= params.underresponse_max_peak_ratio
    and area_ratio <= params.underresponse_max_area_ratio
  )
  is_overresponse = (
    ref_peak is not None
    and abs(ref_peak) >= params.overresponse_min_reference_peak
    and peak_ratio is not None
    and area_ratio is not None
    and mean_overshoot > params.overresponse_min_mean_overshoot
    and peak_overshoot > params.overresponse_min_peak_overshoot
    and peak_ratio >= params.overresponse_min_peak_ratio
    and area_ratio >= params.overresponse_min_area_ratio
  )

  if is_underresponse:
    labels.append("engaged_underresponse_candidate")
  elif is_overresponse:
    labels.append("engaged_overresponse_candidate")
  elif rms is not None and rms <= params.tracking_rms_threshold:
    labels.append("engaged_tracks")
  else:
    labels.append("engaged_uncertain")

  return labels


def _build_event(
  segment: list[LateralTimingFrame],
  start_idx: int,
  end_idx: int,
  params: EventDetectionParams,
) -> LateralTimingEvent | None:
  if start_idx > end_idx or not segment:
    return None

  route = segment[0].route
  ref_values = []
  actual_values = []
  ref_source = "none"
  missing_model_ref = True
  for i in range(start_idx, end_idx + 1):
    ref, src, missing = _reference_value_and_source(segment[i])
    ref_source = src
    missing_model_ref = missing
    actual = _actual_value(segment[i])
    ref_values.append(ref)
    actual_values.append(actual)

  refs = np.asarray(ref_values, dtype=float)
  actuals = np.asarray(actual_values, dtype=float)
  finite_ref = np.isfinite(refs)
  finite_actual = np.isfinite(actuals)

  if not finite_ref.any():
    return None

  event_sign = _sign(float(refs[finite_ref][0]))
  signed_refs = event_sign * refs

  # Reference timings over the active window.
  active_mask = finite_ref & (signed_refs >= params.onset_threshold)
  onset_idx_rel = int(np.argmax(active_mask)) if active_mask.any() else 0
  release_idx_rel = int(len(refs) - 1 - np.argmax(active_mask[::-1])) if active_mask.any() else len(refs) - 1

  ref_onset_idx = start_idx + onset_idx_rel
  ref_release_idx = start_idx + release_idx_rel
  ref_onset_t = segment[ref_onset_idx].t
  ref_release_t = segment[ref_release_idx].t

  peak_ref_idx_rel = int(np.argmax(np.where(finite_ref, signed_refs, -np.inf)))
  peak_ref_idx = start_idx + peak_ref_idx_rel
  ref_peak_t = segment[peak_ref_idx].t
  ref_peak = float(refs[peak_ref_idx_rel])

  # Build expanded window for actual response.
  t0 = max(segment[0].t, ref_onset_t - params.pre_window_s)
  t1 = min(segment[-1].t, ref_release_t + params.post_window_s)
  response_start_idx = 0
  for i in range(start_idx - 1, -1, -1):
    ref, _, _ = _reference_value_and_source(segment[i])
    if ref is not None and math.isfinite(ref) and abs(ref) >= params.onset_threshold:
      response_start_idx = i + 1
      break
  response_end_idx = len(segment) - 1
  for i in range(end_idx + 1, len(segment)):
    ref, _, _ = _reference_value_and_source(segment[i])
    if ref is not None and math.isfinite(ref) and abs(ref) >= params.onset_threshold:
      response_end_idx = i - 1
      break
  window_frames = []
  for i, frame in enumerate(segment):
    if i < response_start_idx or i > response_end_idx or not t0 <= frame.t <= t1:
      continue
    window_frames.append(frame)

  actual_onset_t: float | None = None
  actual_release_t: float | None = None
  actual_peak: float | None = None
  actual_peak_t: float | None = None

  if window_frames:
    win_refs = []
    win_actuals = []
    for f in window_frames:
      r, _, _ = _reference_value_and_source(f)
      a = _actual_value(f)
      win_refs.append(r)
      win_actuals.append(a)
    win_actuals_arr = np.asarray(win_actuals, dtype=float)
    win_finite_actual = np.isfinite(win_actuals_arr)
    signed_actuals = event_sign * win_actuals_arr

    actual_active = win_finite_actual & (signed_actuals >= params.onset_threshold)
    if actual_active.any():
      actual_onset_rel = int(np.argmax(actual_active))
      actual_onset_t = window_frames[actual_onset_rel].t

    actual_release_active = win_finite_actual & (signed_actuals >= params.release_threshold)
    if actual_release_active.any():
      actual_release_rel = int(len(actual_release_active) - 1 - np.argmax(actual_release_active[::-1]))
      actual_release_t = window_frames[actual_release_rel].t

    if win_finite_actual.any():
      peak_actual_rel = int(np.argmax(np.where(win_finite_actual, signed_actuals, -np.inf)))
      actual_peak = float(win_actuals_arr[peak_actual_rel])
      actual_peak_t = window_frames[peak_actual_rel].t

  # Areas and RMS over the active event window only.
  event_times = np.asarray([f.t for f in segment[start_idx:end_idx + 1]], dtype=float)
  ref_area = _trapezoid_area(event_times, event_sign * refs)
  actual_area = _trapezoid_area(event_times, event_sign * np.where(finite_actual, actuals, np.nan))

  if segment[start_idx].mode == "engaged":
    errors = []
    for i in range(start_idx, end_idx + 1):
      r, _, _ = _reference_value_and_source(segment[i])
      a = _actual_value(segment[i])
      if r is not None and a is not None and math.isfinite(r) and math.isfinite(a):
        errors.append((r - a) ** 2)
    rms_tracking_error = math.sqrt(sum(errors) / len(errors)) if errors else None
  else:
    rms_tracking_error = None

  peak_ratio = (abs(actual_peak) / abs(ref_peak)) if (actual_peak is not None and ref_peak is not None and ref_peak != 0) else None
  area_ratio = (actual_area / ref_area) if ref_area and ref_area != 0 else None

  onset_delta = (actual_onset_t - ref_onset_t) if (actual_onset_t is not None and ref_onset_t is not None) else None
  peak_delta = (actual_peak_t - ref_peak_t) if (actual_peak_t is not None and ref_peak_t is not None) else None

  gates_passed, gate_reasons = _gate_event(segment[start_idx:end_idx + 1], params, missing_model_ref)

  metrics = {
    "mode": segment[start_idx].mode,
    "gates_passed": gates_passed,
    "reference_onset_t": ref_onset_t,
    "actual_onset_t": actual_onset_t,
    "reference_peak": ref_peak,
    "actual_peak": actual_peak,
    "peak_ratio": peak_ratio,
    "area_ratio": area_ratio,
    "rms_tracking_error": rms_tracking_error,
  }
  classifications = _classify_event(segment[start_idx:end_idx + 1], metrics, params)

  return LateralTimingEvent(
    route=route,
    start_t=segment[start_idx].t,
    end_t=segment[end_idx].t,
    mode=segment[start_idx].mode,
    sign=event_sign,
    reference_source=ref_source,
    reference_onset_t=ref_onset_t,
    reference_peak_t=ref_peak_t,
    reference_release_t=ref_release_t,
    reference_peak=_r(ref_peak, 4),
    reference_area=_r(ref_area, 4),
    actual_onset_t=actual_onset_t,
    actual_peak_t=actual_peak_t,
    actual_release_t=actual_release_t,
    actual_peak=_r(actual_peak, 4),
    actual_area=_r(actual_area, 4),
    onset_delta_s=_r(onset_delta, 4),
    peak_delta_s=_r(peak_delta, 4),
    peak_ratio=_r(peak_ratio, 4),
    area_ratio=_r(area_ratio, 4),
    rms_tracking_error=_r(rms_tracking_error, 4),
    classifications=classifications,
    gates_passed=gates_passed,
    gate_reasons=gate_reasons,
    missing_model_v2=missing_model_ref,
    sample_count=end_idx - start_idx + 1,
  )


def detect_lateral_events(
  frames: Sequence[LateralTimingFrame],
  params: EventDetectionParams | None = None,
) -> list[LateralTimingEvent]:
  """Segment curve events from a sequence of frames."""
  p = params or EventDetectionParams()
  segments = _segment_frames(frames, p)
  events: list[LateralTimingEvent] = []
  for segment in segments:
    for start_idx, end_idx in _detect_event_indices(segment, p):
      event = _build_event(segment, start_idx, end_idx, p)
      if event is not None:
        events.append(event)
  return events


# ---------------------------------------------------------------------------
# Route message parsing
# ---------------------------------------------------------------------------

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


def _underresponse_blocked(lateral_state: Any | None) -> bool | None:
  if lateral_state is None:
    return None
  block_mask = safe_get(lateral_state, "underresponseBlockMask")
  if block_mask is None:
    return None
  try:
    return int(block_mask) > 0
  except (TypeError, ValueError):
    return None


def build_lateral_timing_frames(
  route: str,
  msgs: list[Any],
  *,
  already_sorted: bool = False,
) -> list[LateralTimingFrame]:
  ordered = list(msgs) if already_sorted else sorted(msgs, key=lambda m: int(getattr(m, "logMonoTime", 0)))
  frames: list[LateralTimingFrame] = []
  latest: dict[str, Any] = {}
  latest_mono: dict[str, int] = {}

  for msg in build_route_messages(ordered):
    if msg.typ in ("carState", "carControl", "modelV2", "liveParameters", "controlsState"):
      latest[msg.typ] = msg.payload
      latest_mono[msg.typ] = int(getattr(msg.raw, "logMonoTime", 0))

    if msg.typ != "controlsState":
      continue

    controls_state = latest.get("controlsState")
    car_state = latest.get("carState")
    car_control = latest.get("carControl")
    model_v2 = latest.get("modelV2")
    live_parameters = latest.get("liveParameters")

    if car_state is None or controls_state is None:
      continue

    v_ego = finite_or_none(safe_get(car_state, "vEgo"))
    if v_ego is None:
      continue

    t = msg.t

    steering_angle = finite_or_none(safe_get(car_state, "steeringAngleDeg"))
    steering_rate = finite_or_none(safe_get(car_state, "steeringRateDeg"))
    steering_torque = finite_or_none(safe_get(car_state, "steeringTorque"))
    roll = finite_or_none(safe_get(live_parameters, "roll"))

    lat_active = bool(safe_get(car_control, "latActive", safe_get(controls_state, "active", False)))
    lateral_state = _lateral_control_payload(controls_state)
    lateral_state_active = bool(safe_get(lateral_state, "active", False))
    steering_pressed = bool(safe_get(car_state, "steeringPressed", False))

    if lat_active and lateral_state_active and not steering_pressed:
      mode = "engaged"
    else:
      # Match manual_lateral_baseline.py: everything with usable car/control state
      # that is not cleanly engaged is descriptive manual/human-reference data.
      mode = "manual"

    curvature = finite_or_none(safe_get(controls_state, "curvature"))
    desired_curvature = finite_or_none(safe_get(controls_state, "desiredCurvature"))
    model_path = safe_get(controls_state, "modelPathState")
    processed_curvature = desired_curvature
    path_quality = finite_or_none(safe_get(model_path, "quality"))
    path_gated = safe_get(model_path, "gated")
    path_reason = safe_get(model_path, "reason")
    if path_reason is not None:
      path_reason = str(path_reason)

    model_action = safe_get(model_v2, "action")
    model_desired_curvature = finite_or_none(safe_get(model_action, "desiredCurvature"))

    actual_lat_accel_log = finite_or_none(safe_get(lateral_state, "actualLateralAccel"))
    desired_lat_accel_log = finite_or_none(safe_get(lateral_state, "desiredLateralAccel"))

    model_raw_lat_accel = _lat_accel(v_ego, model_desired_curvature)
    processed_lat_accel = _lat_accel(v_ego, processed_curvature)
    control_desired_lat_accel = _lat_accel(v_ego, desired_curvature)
    controller_desired_lat_accel = desired_lat_accel_log
    controller_actual_lat_accel = actual_lat_accel_log

    actual_lat_accel = _lat_accel(v_ego, curvature)

    lane_change_state = format_enum(safe_get(model_v2, "meta.laneChangeState", safe_get(controls_state, "laneChangeState", "")))
    controls_mono = latest_mono.get("controlsState")
    model_mono = latest_mono.get("modelV2")
    if controls_mono is not None and model_mono is not None:
      model_age = max(0.0, (controls_mono - model_mono) / 1e9)
    else:
      model_age = None

    adaptive = safe_get(lateral_state, "adaptiveTorqueState")
    governor_reason = safe_get(adaptive, "governorReason")
    if governor_reason is not None:
      try:
        governor_reason = int(governor_reason)
      except (TypeError, ValueError):
        governor_reason = None

    frames.append(LateralTimingFrame(
      t=t,
      route=route,
      mode=mode,
      v_ego=v_ego,
      steering_angle_deg=steering_angle,
      steering_rate_deg_s=steering_rate,
      steering_torque=steering_torque,
      roll=roll,
      model_raw_lat_accel=model_raw_lat_accel,
      processed_lat_accel=processed_lat_accel,
      control_desired_lat_accel=control_desired_lat_accel,
      controller_desired_lat_accel=controller_desired_lat_accel,
      controller_actual_lat_accel=controller_actual_lat_accel,
      actual_lat_accel=actual_lat_accel,
      path_quality=path_quality,
      path_gated=bool(path_gated) if path_gated is not None else False,
      path_reason=path_reason,
      model_age_s=model_age,
      steering_pressed=steering_pressed,
      left_blinker=bool(safe_get(car_state, "leftBlinker", False)),
      right_blinker=bool(safe_get(car_state, "rightBlinker", False)),
      lane_change_state=lane_change_state,
      standstill=bool(safe_get(car_state, "standstill", False)),
      lat_active=lat_active,
      lateral_state_active=lateral_state_active,
      saturated=bool(safe_get(lateral_state, "saturated")) if lateral_state is not None else None,
      underresponse_active=bool(safe_get(lateral_state, "underresponseActive")) if lateral_state is not None else None,
      underresponse_eligible=bool(safe_get(lateral_state, "underresponseEligible")) if lateral_state is not None else None,
      underresponse_block=_underresponse_blocked(lateral_state),
      underresponse_shadow=finite_or_none(safe_get(lateral_state, "underresponseShadowLatAccel")),
      governor_reason=governor_reason,
      controller_output=finite_or_none(safe_get(lateral_state, "output")),
    ))

  return frames


# ---------------------------------------------------------------------------
# Report construction
# ---------------------------------------------------------------------------

def _coverage(frames: list[LateralTimingFrame]) -> CoverageCounters:
  c = CoverageCounters()
  c.total_frames = len(frames)
  for f in frames:
    if f.mode == "manual":
      c.manual_frames += 1
    elif f.mode == "engaged":
      c.engaged_frames += 1
    elif f.mode == "inactive":
      c.inactive_frames += 1
    else:
      c.unknown_frames += 1
    if f.model_raw_lat_accel is None:
      c.missing_model_v2_frames += 1
  return c


def _summarize(events: list[LateralTimingEvent], coverage: CoverageCounters) -> dict[str, Any]:
  by_mode: Counter[str] = Counter()
  for e in events:
    by_mode[e.mode] += 1

  engaged_events = [e for e in events if e.mode == "engaged" and e.gates_passed]
  underresponse = [e for e in engaged_events if "engaged_underresponse_candidate" in e.classifications]
  tracks = [e for e in engaged_events if "engaged_tracks" in e.classifications]
  manual_events = [e for e in events if e.mode == "manual"]

  return {
    "events": len(events),
    "manual_events": len([e for e in events if e.mode == "manual"]),
    "engaged_events": len([e for e in events if e.mode == "engaged"]),
    "invalid_gate_events": len([e for e in events if not e.gates_passed]),
    "missing_model_v2_events": len([e for e in events if e.missing_model_v2]),
    "engaged_underresponse_candidates": len(underresponse),
    "engaged_tracks": len(tracks),
    "manual_human_later": len([e for e in manual_events if "manual_human_later" in e.classifications]),
    "manual_human_lower_peak": len([e for e in manual_events if "manual_human_lower_peak" in e.classifications]),
    "median_onset_delta_s": _r(float(np.median(deltas))) if (deltas := [e.onset_delta_s for e in events if e.onset_delta_s is not None]) else None,
    "median_peak_ratio": _r(float(np.median(ratios))) if (ratios := [e.peak_ratio for e in events if e.peak_ratio is not None]) else None,
    "mode_counts": dict(by_mode),
    "coverage": coverage.to_dict(),
  }


def analyze_lateral_timing(
  msgs: list[Any],
  source: str = "unknown",
  params: EventDetectionParams | None = None,
) -> LateralTimingReport:
  return analyze_lateral_timing_frames(
    build_lateral_timing_frames(source, msgs, already_sorted=True), source=source, params=params,
  )


def analyze_lateral_timing_frames(
  frames: list[LateralTimingFrame],
  source: str = "unknown",
  params: EventDetectionParams | None = None,
) -> LateralTimingReport:
  p = params or EventDetectionParams()
  coverage = _coverage(frames)
  events = detect_lateral_events(frames, p)

  notes = [
    "manual events compare human steering to the model plan; disagreement is not a tracking error",
    "engaged underresponse candidates use conservative gates and are not proof of a controller defect",
  ]
  if coverage.total_frames and coverage.missing_model_v2_frames == coverage.total_frames:
    notes.append("modelV2 is absent; manual event reference falls back to processed/control desired curvature")

  return LateralTimingReport(
    schema=SCHEMA,
    version=SCHEMA_VERSION,
    source=source,
    parameters=p,
    coverage=coverage,
    summary=_summarize(events, coverage),
    events=events,
    notes=notes,
  )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_report(report: LateralTimingReport) -> str:
  lines = [
    f"Manual/engaged lateral timing: {report.source}",
    f"  frames: {report.coverage.total_frames} (manual={report.coverage.manual_frames} engaged={report.coverage.engaged_frames})",
    f"  events: {report.summary['events']} manual={report.summary['manual_events']} engaged={report.summary['engaged_events']}",
    f"  invalid_gate_events: {report.summary['invalid_gate_events']}",
    f"  engaged underresponse candidates: {report.summary['engaged_underresponse_candidates']}",
    f"  engaged tracks: {report.summary['engaged_tracks']}",
    f"  median onset delta (actual - ref): {_fmt(report.summary.get('median_onset_delta_s'))} s",
    f"  median peak ratio: {_fmt(report.summary.get('median_peak_ratio'))}",
  ]

  for event in report.events[:20]:
    lines.append(
      f"  t={event.start_t:.2f}-{event.end_t:.2f} mode={event.mode} sign={event.sign:+.0f} "
      f"ref={event.reference_peak:.3f}@{_fmt(event.reference_peak_t, 2)} "
      f"actual={event.actual_peak:.3f}@{_fmt(event.actual_peak_t, 2)} "
      f"delta={_fmt(event.onset_delta_s, 3)}s rms={_fmt(event.rms_tracking_error, 3)} "
      f"gates={event.gates_passed} classes={','.join(event.classifications)}"
    )

  if len(report.events) > 20:
    lines.append(f"  ... and {len(report.events) - 20} more events")

  for note in report.notes:
    lines.append(f"  note: {note}")

  return "\n".join(lines)


def _fmt(value: float | None, ndigits: int = 3) -> str:
  if value is None or not isinstance(value, int | float) or not math.isfinite(float(value)):
    return "n/a"
  return f"{float(value):.{ndigits}f}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _resolve_inputs(inputs: list[str], read_mode: Any, log_roots: tuple[Path, ...]) -> list[str]:
  from openpilot.tools.drive_lab.analyze_longitudinal_lateral_route import resolve_inputs as upstream_resolve, DEFAULT_LOG_ROOTS
  roots = log_roots + DEFAULT_LOG_ROOTS
  return upstream_resolve(inputs, segment=None, read_mode=read_mode, log_roots=roots)


def main() -> None:
  import sys
  from openpilot.tools.lib.logreader import LogReader, ReadMode

  parser = argparse.ArgumentParser(description="Compare manual-vs-model and engaged controller lateral timing.")
  parser.add_argument("inputs", nargs="+", help="Route ids, local dirs, files, or LogReader route strings")
  parser.add_argument("--qlog", action="store_true", help="Prefer qlogs")
  parser.add_argument("--rlog", action="store_true", help="Prefer rlogs")
  parser.add_argument("--json", action="store_true", help="Print JSON instead of text")
  parser.add_argument("--output", help="Write the report JSON to this path")
  parser.add_argument("--log-root", action="append", default=[], help="Extra local route search roots")
  parser.add_argument("--onset", type=float, default=None, help="Onset threshold m/s^2")
  parser.add_argument("--release", type=float, default=None, help="Release threshold m/s^2")
  parser.add_argument("--min-duration", type=float, default=None, help="Minimum event duration s")
  parser.add_argument("--max-gap", type=float, default=None, help="Max allowed gap within an event s")
  args = parser.parse_args()

  if args.qlog and args.rlog:
    parser.error("choose at most one of --qlog or --rlog")

  read_mode = ReadMode.QLOG if args.qlog else ReadMode.RLOG if args.rlog else ReadMode.AUTO
  log_roots = tuple(Path(p) for p in args.log_root)

  params_kwargs: dict[str, float] = {}
  if args.onset is not None:
    params_kwargs["onset_threshold"] = args.onset
  if args.release is not None:
    params_kwargs["release_threshold"] = args.release
  if args.min_duration is not None:
    params_kwargs["min_duration_s"] = args.min_duration
  if args.max_gap is not None:
    params_kwargs["max_gap_s"] = args.max_gap
  params = EventDetectionParams(**params_kwargs)

  try:
    identifiers = _resolve_inputs(args.inputs, read_mode, log_roots)
  except Exception as exc:
    print(f"failed to resolve inputs: {exc}", file=sys.stderr)
    raise SystemExit(1)

  source_label = ", ".join(identifiers) if identifiers else ", ".join(args.inputs)
  frames = [
    frame
    for identifier in identifiers
    for frame in build_lateral_timing_frames(
      identifier,
      list(LogReader(identifier, default_mode=read_mode, sort_by_time=True)),
      already_sorted=True,
    )
  ]
  report = analyze_lateral_timing_frames(frames, source=source_label, params=params)
  print(output_report(report, json_output=args.json, renderer=render_report, output_path=args.output))


if __name__ == "__main__":
  main()
