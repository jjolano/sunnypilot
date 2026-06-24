import math
from typing import Any

import pytest

from openpilot.tools.drive_lab.compare_manual_lateral_timing import (
  EventDetectionParams,
  LateralTimingFrame,
  detect_lateral_events,
)


DEFAULT_PARAMS = EventDetectionParams()


def _frames(
  t_start: float,
  t_end: float,
  dt: float,
  *,
  v_ego: float = 10.0,
  mode: str = "manual",
  ref_signal: str = "model_raw",
  ref_func,
  actual_func,
  **kwargs,
):
  frames = []
  t = t_start
  while t <= t_end + dt * 0.5:
    ref = ref_func(t)
    actual = actual_func(t)
    kw: dict[str, Any] = dict(
      t=t,
      v_ego=v_ego,
      mode=mode,
      actual_lat_accel=actual,
      left_blinker=False,
      right_blinker=False,
      lane_change_state="off",
      standstill=False,
      lat_active=mode == "engaged",
      lateral_state_active=mode == "engaged",
      steering_pressed=False,
      saturated=False,
      path_gated=False,
      roll=0.0,
    )
    if ref_signal == "model_raw":
      kw["model_raw_lat_accel"] = ref
    elif ref_signal == "controller_desired":
      kw["controller_desired_lat_accel"] = ref
    elif ref_signal == "processed":
      kw["processed_lat_accel"] = ref
    elif ref_signal == "control_desired":
      kw["control_desired_lat_accel"] = ref
    kw.update(kwargs)
    frames.append(LateralTimingFrame(**kw))
    t += dt
  return frames


def _ramp_up_hold_down(t, t_onset=0.0, t_peak=0.5, t_release=1.0, peak=0.6):
  if t < t_onset:
    return 0.0
  if t < t_peak:
    return peak * (t - t_onset) / (t_peak - t_onset)
  if t < t_release:
    return peak
  return max(0.0, peak - peak * (t - t_release) / 0.4)


def test_manual_event_later_and_lower_peak():
  """Human steering starts later and peaks lower than the model plan."""
  dt = 0.05
  ref_frames = _frames(
    0.0, 1.5, dt,
    mode="manual",
    ref_signal="model_raw",
    ref_func=lambda t: _ramp_up_hold_down(t, t_onset=0.0, t_peak=0.4, t_release=0.8, peak=0.55),
    actual_func=lambda t: _ramp_up_hold_down(t, t_onset=0.5, t_peak=0.9, t_release=1.3, peak=0.30),
  )
  events = detect_lateral_events(ref_frames, DEFAULT_PARAMS)
  assert len(events) == 1
  ev = events[0]
  assert ev.mode == "manual"
  assert ev.rms_tracking_error is None
  assert "manual_human_later" in ev.classifications
  assert "manual_human_lower_peak" in ev.classifications
  assert "engaged_underresponse_candidate" not in ev.classifications
  assert ev.onset_delta_s is not None
  assert ev.onset_delta_s > 0.4
  assert ev.peak_ratio is not None
  assert ev.peak_ratio < 0.7


def test_engaged_event_tracks_reference():
  """Controller output closely follows the desired lateral accel."""
  dt = 0.05
  noise = lambda t: 0.01 * math.sin(2 * math.pi * 3 * t)
  frames = _frames(
    0.0, 1.2, dt,
    mode="engaged",
    ref_signal="controller_desired",
    ref_func=lambda t: _ramp_up_hold_down(t, t_onset=0.0, t_peak=0.4, t_release=0.8, peak=0.50),
    actual_func=lambda t: _ramp_up_hold_down(t, t_onset=0.0, t_peak=0.4, t_release=0.8, peak=0.50) + noise(t),
  )
  events = detect_lateral_events(frames, DEFAULT_PARAMS)
  assert len(events) == 1
  ev = events[0]
  assert ev.mode == "engaged"
  assert ev.gates_passed
  assert "engaged_tracks" in ev.classifications
  assert "engaged_underresponse_candidate" not in ev.classifications
  assert ev.rms_tracking_error is not None
  assert ev.rms_tracking_error < 0.05
  assert ev.peak_ratio is not None
  assert 0.9 <= ev.peak_ratio <= 1.1


def test_engaged_underresponse_candidate():
  """Controller actual lags and under-peaks the desired signal with clean gates."""
  dt = 0.05
  frames = _frames(
    0.0, 1.6, dt,
    mode="engaged",
    ref_signal="controller_desired",
    ref_func=lambda t: _ramp_up_hold_down(t, t_onset=0.0, t_peak=0.4, t_release=0.9, peak=0.60),
    actual_func=lambda t: _ramp_up_hold_down(t, t_onset=0.35, t_peak=1.0, t_release=1.4, peak=0.25),
  )
  events = detect_lateral_events(frames, DEFAULT_PARAMS)
  assert len(events) == 1
  ev = events[0]
  assert ev.mode == "engaged"
  assert ev.gates_passed
  assert "engaged_underresponse_candidate" in ev.classifications
  assert "engaged_tracks" not in ev.classifications
  assert ev.rms_tracking_error is not None
  assert ev.peak_ratio is not None
  assert ev.peak_ratio < 0.7


def test_engaged_lag_with_higher_peak_is_not_underresponse():
  """A transient early deficit with higher eventual response should not be underresponse."""
  dt = 0.05
  frames = _frames(
    0.0, 1.6, dt,
    mode="engaged",
    ref_signal="controller_desired",
    ref_func=lambda t: _ramp_up_hold_down(t, t_onset=0.0, t_peak=0.4, t_release=0.9, peak=0.60),
    actual_func=lambda t: _ramp_up_hold_down(t, t_onset=0.35, t_peak=0.8, t_release=1.3, peak=0.78),
  )
  events = detect_lateral_events(frames, DEFAULT_PARAMS)
  assert len(events) == 1
  ev = events[0]
  assert ev.mode == "engaged"
  assert ev.gates_passed
  assert "engaged_underresponse_candidate" not in ev.classifications
  assert ev.peak_ratio is not None
  assert ev.peak_ratio > 1.0


def test_engaged_sustained_overresponse_candidate():
  """Sustained larger actual response should be flagged for overresponse analysis."""
  dt = 0.05
  frames = _frames(
    0.0, 1.4, dt,
    mode="engaged",
    ref_signal="controller_desired",
    ref_func=lambda t: _ramp_up_hold_down(t, t_onset=0.0, t_peak=0.4, t_release=0.9, peak=0.70),
    actual_func=lambda t: _ramp_up_hold_down(t, t_onset=0.0, t_peak=0.4, t_release=0.9, peak=0.90),
  )
  events = detect_lateral_events(frames, DEFAULT_PARAMS)
  assert len(events) == 1
  ev = events[0]
  assert ev.gates_passed
  assert "engaged_overresponse_candidate" in ev.classifications
  assert "engaged_underresponse_candidate" not in ev.classifications
  assert ev.peak_ratio is not None and ev.peak_ratio > 1.1
  assert ev.area_ratio is not None and ev.area_ratio > 1.05


def test_engaged_tiny_reference_overshoot_is_not_overresponse():
  """Ratio explosions on tiny references should not be considered tune-worthy."""
  dt = 0.05
  params = EventDetectionParams(onset_threshold=0.10, release_threshold=0.05)
  frames = _frames(
    0.0, 1.0, dt,
    mode="engaged",
    ref_signal="controller_desired",
    ref_func=lambda t: _ramp_up_hold_down(t, t_onset=0.0, t_peak=0.3, t_release=0.7, peak=0.20),
    actual_func=lambda t: _ramp_up_hold_down(t, t_onset=0.0, t_peak=0.3, t_release=0.7, peak=0.35),
  )
  events = detect_lateral_events(frames, params)
  assert len(events) == 1
  ev = events[0]
  assert ev.gates_passed
  assert ev.peak_ratio is not None and ev.peak_ratio > 1.4
  assert "engaged_overresponse_candidate" not in ev.classifications


def test_path_gated_marks_invalid_gate():
  """A gated model path should produce an invalid_gate classification."""
  dt = 0.05
  frames = _frames(
    0.0, 1.2, dt,
    mode="engaged",
    ref_signal="controller_desired",
    ref_func=lambda t: _ramp_up_hold_down(t, t_onset=0.0, t_peak=0.4, t_release=0.8, peak=0.50),
    actual_func=lambda t: _ramp_up_hold_down(t, t_onset=0.0, t_peak=0.4, t_release=0.8, peak=0.50),
    path_gated=True,
    path_reason="gated",
  )
  events = detect_lateral_events(frames, DEFAULT_PARAMS)
  assert len(events) == 1
  ev = events[0]
  assert not ev.gates_passed
  assert "invalid_gate" in ev.classifications
  assert "path_gated" in ev.gate_reasons


def test_stale_model_marks_invalid_gate():
  """Model age above the stale threshold should invalidate the event."""
  dt = 0.05
  frames = _frames(
    0.0, 1.2, dt,
    mode="manual",
    ref_signal="model_raw",
    ref_func=lambda t: _ramp_up_hold_down(t, t_onset=0.0, t_peak=0.4, t_release=0.8, peak=0.50),
    actual_func=lambda t: _ramp_up_hold_down(t, t_onset=0.0, t_peak=0.4, t_release=0.8, peak=0.50),
    model_age_s=0.35,
  )
  events = detect_lateral_events(frames, DEFAULT_PARAMS)
  assert len(events) == 1
  ev = events[0]
  assert not ev.gates_passed
  assert "invalid_gate" in ev.classifications
  assert "stale_model" in ev.gate_reasons


def test_missing_model_v2_flag_falls_back_and_reports():
  """Manual mode without modelV2 should still segment from fallback signals and flag missing model."""
  dt = 0.05
  frames = _frames(
    0.0, 1.2, dt,
    mode="manual",
    ref_signal="processed",
    ref_func=lambda t: _ramp_up_hold_down(t, t_onset=0.0, t_peak=0.4, t_release=0.8, peak=0.50),
    actual_func=lambda t: _ramp_up_hold_down(t, t_onset=0.0, t_peak=0.4, t_release=0.8, peak=0.50),
  )
  events = detect_lateral_events(frames, DEFAULT_PARAMS)
  assert len(events) == 1
  ev = events[0]
  assert ev.reference_source == "processed"
  assert ev.missing_model_v2
