"""Extract fixed-DT longitudinal replay frames from route logs.

Route extraction uses latest-available carState/radarState/liveParameters context per
controlsState step. Replay validates planner response to lead speed/confidence/cruise
trajectories; the Maneuver plant integrates gap internally so bit-identical d_rel replay
is not expected.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from openpilot.common.realtime import DT_MDL
from openpilot.tools.drive_lab.route_analysis import build_route_messages
from openpilot.tools.drive_lab.timeline import safe_get

DT = DT_MDL
SAMPLE_MODES = ("prefix", "random-window", "uniform-windows")
ROUTE_EXTRACTED_PRESET = "route_extracted"


@dataclass
class LongitudinalExtractionQuality:
  input_message_count: int = 0
  controls_state_seen: int = 0
  controls_state_in_window: int = 0
  skipped_missing_car_state: int = 0
  missing_radar_state: int = 0
  missing_live_parameters: int = 0

  def to_dict(self) -> dict[str, Any]:
    return {
      "input_message_count": self.input_message_count,
      "controls_state_seen": self.controls_state_seen,
      "controls_state_in_window": self.controls_state_in_window,
      "skipped_missing_car_state": self.skipped_missing_car_state,
      "missing_radar_state": self.missing_radar_state,
      "missing_live_parameters": self.missing_live_parameters,
    }


@dataclass(frozen=True)
class LongitudinalRouteFrame:
  t: float
  source_t: float | None
  v_ego: float
  v_cruise: float
  pitch: float
  lead_active: bool
  d_rel: float | None
  v_lead: float | None
  prob_lead: float
  prob_throttle: float = 1.0

  def to_dict(self) -> dict[str, Any]:
    return {
      "t": self.t,
      "source_t": self.source_t,
      "v_ego": self.v_ego,
      "v_cruise": self.v_cruise,
      "pitch": self.pitch,
      "lead_active": self.lead_active,
      "d_rel": self.d_rel,
      "v_lead": self.v_lead,
      "prob_lead": self.prob_lead,
      "prob_throttle": self.prob_throttle,
    }

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> LongitudinalRouteFrame:
    return cls(
      t=float(data["t"]),
      source_t=data.get("source_t"),
      v_ego=float(data["v_ego"]),
      v_cruise=float(data["v_cruise"]),
      pitch=float(data.get("pitch", 0.0)),
      lead_active=bool(data.get("lead_active", False)),
      d_rel=data.get("d_rel"),
      v_lead=data.get("v_lead"),
      prob_lead=float(data.get("prob_lead", 0.0)),
      prob_throttle=float(data.get("prob_throttle", 1.0)),
    )


@dataclass(frozen=True)
class LongitudinalExtractionSummary:
  route: str | None
  qlog: bool
  window_start_s: float | None
  window_end_s: float | None
  max_frames: int | None
  extracted_count: int
  original_time_span_s: float | None
  dt: float
  quality: LongitudinalExtractionQuality | None = None
  sampling_mode: str | None = None

  def to_dict(self) -> dict[str, Any]:
    return {
      "route": self.route,
      "qlog": self.qlog,
      "window_start_s": self.window_start_s,
      "window_end_s": self.window_end_s,
      "max_frames": self.max_frames,
      "extracted_count": self.extracted_count,
      "original_time_span_s": self.original_time_span_s,
      "dt": self.dt,
      "quality": self.quality.to_dict() if self.quality else None,
      "sampling_mode": self.sampling_mode,
    }


def extract_longitudinal_route_frames_with_summary(
  messages: Any,
  *,
  route: str | None = None,
  qlog: bool = False,
  start_s: float | None = None,
  end_s: float | None = None,
  max_frames: int | None = None,
  engaged_only: bool = False,
  sampling_mode: str = "prefix",
) -> tuple[tuple[LongitudinalRouteFrame, ...], LongitudinalExtractionSummary]:
  if sampling_mode not in SAMPLE_MODES:
    raise ValueError(f"unknown sampling_mode {sampling_mode!r}")

  route_messages = build_route_messages(messages)
  latest: dict[str, Any] = {}
  extracted: list[LongitudinalRouteFrame] = []
  original_times: list[float] = []
  quality = LongitudinalExtractionQuality(input_message_count=len(route_messages))

  for rm in route_messages:
    if rm.typ in ("carState", "radarState", "liveParameters", "carControl", "controlsState"):
      latest[rm.typ] = rm.payload
    if rm.typ != "controlsState":
      continue
    quality.controls_state_seen += 1
    if start_s is not None and rm.t < start_s:
      continue
    if end_s is not None and rm.t >= end_s:
      continue
    if engaged_only and not bool(safe_get(latest.get("carControl"), "longActive", False)):
      continue
    quality.controls_state_in_window += 1
    if latest.get("carState") is None:
      quality.skipped_missing_car_state += 1
      continue
    if latest.get("radarState") is None:
      quality.missing_radar_state += 1
    if latest.get("liveParameters") is None:
      quality.missing_live_parameters += 1
    frame = _frame_from_state(rm.t, latest, source_t=rm.t)
    extracted.append(frame)
    original_times.append(rm.t)
    if max_frames is not None and len(extracted) >= max_frames:
      break

  normalized = tuple(
    LongitudinalRouteFrame.from_dict({**frame.to_dict(), "t": float(i) * DT})
    for i, frame in enumerate(extracted)
  )
  original_span = original_times[-1] - original_times[0] if len(original_times) > 1 else None
  summary = LongitudinalExtractionSummary(
    route=route,
    qlog=qlog,
    window_start_s=start_s,
    window_end_s=end_s,
    max_frames=max_frames,
    extracted_count=len(normalized),
    original_time_span_s=original_span,
    dt=DT,
    quality=quality,
    sampling_mode=sampling_mode,
  )
  return normalized, summary


def extract_longitudinal_route_frames(
  messages: Any,
  *,
  start_s: float | None = None,
  end_s: float | None = None,
  max_frames: int | None = None,
  engaged_only: bool = False,
) -> tuple[LongitudinalRouteFrame, ...]:
  frames, _ = extract_longitudinal_route_frames_with_summary(
    messages,
    start_s=start_s,
    end_s=end_s,
    max_frames=max_frames,
    engaged_only=engaged_only,
  )
  return frames


def load_route_frames(
  route: str,
  *,
  qlog: bool = False,
  start_s: float | None = None,
  end_s: float | None = None,
  max_frames: int | None = None,
  engaged_only: bool = False,
) -> tuple[tuple[LongitudinalRouteFrame, ...], LongitudinalExtractionSummary]:
  from openpilot.tools.drive_lab.route_io import load_route_msgs

  messages = load_route_msgs(route, qlog=qlog)
  return extract_longitudinal_route_frames_with_summary(
    messages,
    route=route,
    qlog=qlog,
    start_s=start_s,
    end_s=end_s,
    max_frames=max_frames,
    engaged_only=engaged_only,
  )


def frames_to_maneuver_kwargs(frames: tuple[LongitudinalRouteFrame, ...]) -> dict[str, Any]:
  if not frames:
    raise ValueError("expected at least one longitudinal route frame")
  first = frames[0]
  lead_active = any(frame.lead_active for frame in frames)
  breakpoints = [frame.t for frame in frames]
  speed_lead_values = [
    float(frame.v_lead if frame.v_lead is not None else 0.0) for frame in frames
  ]
  prob_lead_values = [float(frame.prob_lead) for frame in frames]
  cruise_values = [float(frame.v_cruise) for frame in frames]
  pitch_values = [float(frame.pitch) for frame in frames]
  prob_throttle_values = [float(frame.prob_throttle) for frame in frames]
  kwargs: dict[str, Any] = {
    "initial_speed": float(first.v_ego),
    "lead_relevancy": lead_active,
    "breakpoints": breakpoints,
    "speed_lead_values": speed_lead_values,
    "prob_lead_values": prob_lead_values,
    "cruise_values": cruise_values,
    "pitch_values": pitch_values,
    "prob_throttle_values": prob_throttle_values,
  }
  # lead_active is any() over the route, so a route whose lead only appears later
  # (a cut-in) has no d_rel on frame zero. Falling through here left Maneuver's
  # 200 m default in place, so the cut-in replayed from 200 m away instead of from
  # where it actually appeared. Seed from the first frame that really has a lead.
  first_lead = next(
    (frame for frame in frames if frame.lead_active and frame.d_rel is not None), None
  )
  if lead_active and first_lead is not None:
    kwargs["initial_distance_lead"] = float(first_lead.d_rel)
  return kwargs


def max_d_rel_error(frames: tuple[LongitudinalRouteFrame, ...], output: Any) -> float | None:
  import numpy as np

  output = np.asarray(output, dtype=float)
  if output.ndim != 2 or output.shape[1] < 7:
    return None
  errors: list[float] = []
  for idx, frame in enumerate(frames):
    if frame.d_rel is None or idx >= len(output):
      continue
    errors.append(abs(float(output[idx, 6]) - float(frame.d_rel)))
  return max(errors) if errors else None


def _frame_from_state(t: float, latest: dict[str, Any], *, source_t: float) -> LongitudinalRouteFrame:
  car_state = latest.get("carState") or {}
  radar = latest.get("radarState") or {}
  live_params = latest.get("liveParameters") or {}
  lead = safe_get(radar, "leadOne") or {}

  v_ego = float(safe_get(car_state, "vEgo", 0.0))
  v_cruise_kph = float(safe_get(car_state, "vCruise", 255.0))
  v_cruise = v_cruise_kph / 3.6 if v_cruise_kph < 255.0 else v_ego
  pitch = float(safe_get(live_params, "pitch", 0.0))

  lead_status = bool(safe_get(lead, "present", False))
  d_rel = safe_get(lead, "dRel")
  v_lead = safe_get(lead, "vLead")
  model_prob = safe_get(lead, "modelProb", 0.0)
  prob_lead = float(model_prob) if lead_status else 0.0

  return LongitudinalRouteFrame(
    t=t,
    source_t=source_t,
    v_ego=v_ego,
    v_cruise=v_cruise,
    pitch=pitch,
    lead_active=lead_status,
    d_rel=float(d_rel) if d_rel is not None else None,
    v_lead=float(v_lead) if v_lead is not None else None,
    prob_lead=prob_lead,
  )
