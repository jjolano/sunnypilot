"""Tests for bounded center-chase relaxation in the lane-centering assist layer.

Relaxation should only engage during repeated near-center lateral reversals and should
act as a temporary soft deadband on lane-centering error, never as a steering bias.
"""
from __future__ import annotations

import math

import pytest

from openpilot.sunnypilot.custom.lateral.demand.lane_centering_assist import (
  LANE_CENTERING_RELAX_REASON_CURVE,
  LANE_CENTERING_RELAX_REASON_DRIVER,
  LANE_CENTERING_RELAX_REASON_LARGE_ERROR,
  LANE_CENTERING_RELAX_REASON_LOW_SPEED,
  LANE_CENTERING_RELAX_REASON_QUALITY,
  LaneCenteringAssistInputs,
  LaneCenteringAssistTracker,
)

DT = 0.01
N = 33


def _tracker_inputs(lat_error: float = 0.0, pred_bias: float = 0.0, v_ego: float = 20.0,
                    path_quality: float = 1.0, **kwargs) -> LaneCenteringAssistInputs:
  xs = [float(x) for x in range(N)]
  ys = [lat_error + pred_bias * (x / max(1, N - 1)) for x in xs]
  yaws = [0.0] * N
  return LaneCenteringAssistInputs(
    lat_active=kwargs.get("lat_active", True),
    v_ego=kwargs.get("v_ego", v_ego),
    measured_curvature=kwargs.get("measured_curvature", 0.0),
    model_curvature=kwargs.get("model_curvature", 0.0),
    previous_processed_curvature=kwargs.get("previous_processed_curvature", 0.0),
    path_quality=kwargs.get("path_quality", path_quality),
    path_reason=kwargs.get("path_reason", "ok"),
    lane_change_shaping_active=kwargs.get("lane_change_shaping_active", False),
    lane_change_blend=kwargs.get("lane_change_blend", 0.0),
    curvature_limited=kwargs.get("curvature_limited", False),
    steering_pressed=kwargs.get("steering_pressed", False),
    left_blinker=kwargs.get("left_blinker", False),
    right_blinker=kwargs.get("right_blinker", False),
    position_x=kwargs.get("position_x", xs),
    position_y=kwargs.get("position_y", ys),
    orientation_z=kwargs.get("orientation_z", yaws),
    lane_line_probs=kwargs.get("lane_line_probs", [0.9, 0.9, 0.9, 0.9]),
    demand_source=kwargs.get("demand_source", "model_path"),
  )


def test_no_relaxation_for_steady_offset():
  tracker = LaneCenteringAssistTracker()
  for _ in range(200):
    r = tracker.update(_tracker_inputs(lat_error=0.10, pred_bias=0.02), DT)

  assert r.relax_active is False
  assert r.relax_envelope == pytest.approx(0.0)
  assert r.relax_nudge_flip_score == 0.0
  assert r.relax_error_cross_score == 0.0


def test_relaxation_triggers_on_repeated_near_center_flips():
  tracker = LaneCenteringAssistTracker()
  active_frames: list[bool] = []

  for i in range(200):
    lat_error = 0.10 if i % 2 == 0 else -0.10
    r = tracker.update(_tracker_inputs(lat_error=lat_error, pred_bias=0.02), DT)
    active_frames.append(r.relax_active)

  assert any(active_frames), "relaxation never became active"
  trigger_frame = active_frames.index(True)
  # Should trigger quickly once two sign changes have been observed in the window.
  assert trigger_frame < 10


def test_relaxation_zeros_nudge_within_envelope():
  tracker = LaneCenteringAssistTracker()
  active_nudges: list[float] = []

  for i in range(200):
    lat_error = 0.10 if i % 2 == 0 else -0.10
    r = tracker.update(_tracker_inputs(lat_error=lat_error, pred_bias=0.02), DT)
    if r.relax_active:
      active_nudges.append(abs(r.curvature_nudge))

  assert active_nudges, "no active relaxation frames observed"
  assert max(active_nudges) < 1e-6


def test_relaxation_reduces_lateral_error_in_result():
  tracker = LaneCenteringAssistTracker()

  for i in range(150):
    lat_error = 0.10 if i % 2 == 0 else -0.10
    r = tracker.update(_tracker_inputs(lat_error=lat_error, pred_bias=0.02), DT)
    if r.relax_active:
      # Effective (relaxed) errors should be pulled toward zero inside the envelope.
      assert abs(r.relaxed_lateral_error) <= abs(r.lateral_error)
      assert abs(r.relaxed_predicted_error) <= abs(r.predicted_lateral_error)
      assert r.relax_envelope >= 0.08


def test_relaxation_aborts_when_lateral_error_grows():
  tracker = LaneCenteringAssistTracker()

  # First trigger relaxation with rapid flips.
  for i in range(120):
    lat_error = 0.10 if i % 2 == 0 else -0.10
    tracker.update(_tracker_inputs(lat_error=lat_error, pred_bias=0.02), DT)

  # Then jump far enough past the abort threshold.
  for _ in range(20):
    r = tracker.update(_tracker_inputs(lat_error=0.35, pred_bias=0.0), DT)

  assert r.relax_active is False
  assert r.relax_reason_bits & LANE_CENTERING_RELAX_REASON_LARGE_ERROR


def test_relaxation_aborts_on_driver_steering():
  tracker = LaneCenteringAssistTracker()

  for i in range(120):
    lat_error = 0.10 if i % 2 == 0 else -0.10
    tracker.update(_tracker_inputs(lat_error=lat_error, pred_bias=0.02), DT)

  for _ in range(5):
    r = tracker.update(_tracker_inputs(lat_error=0.10, pred_bias=0.02, steering_pressed=True), DT)

  assert r.relax_active is False
  assert r.relax_reason_bits & LANE_CENTERING_RELAX_REASON_DRIVER


def test_relaxation_aborts_low_quality():
  tracker = LaneCenteringAssistTracker()

  for i in range(120):
    lat_error = 0.10 if i % 2 == 0 else -0.10
    tracker.update(_tracker_inputs(lat_error=lat_error, pred_bias=0.02), DT)

  for _ in range(5):
    r = tracker.update(_tracker_inputs(lat_error=0.10, pred_bias=0.02, path_quality=0.80), DT)

  assert r.relax_active is False
  assert r.relax_reason_bits & LANE_CENTERING_RELAX_REASON_QUALITY


def test_relaxation_aborts_low_speed():
  tracker = LaneCenteringAssistTracker()

  for i in range(120):
    lat_error = 0.10 if i % 2 == 0 else -0.10
    tracker.update(_tracker_inputs(lat_error=lat_error, pred_bias=0.02), DT)

  for _ in range(5):
    r = tracker.update(_tracker_inputs(lat_error=0.10, pred_bias=0.02, v_ego=5.0), DT)

  assert r.relax_active is False
  assert r.relax_reason_bits & LANE_CENTERING_RELAX_REASON_LOW_SPEED


def test_relaxation_decay_and_cooldown_are_reasonable():
  tracker = LaneCenteringAssistTracker()

  # Trigger briefly; the deadband then holds, decays, and eventually cools down.
  for i in range(60):
    lat_error = 0.10 if i % 2 == 0 else -0.10
    tracker.update(_tracker_inputs(lat_error=lat_error, pred_bias=0.02), DT)

  # Remove flips but keep near-center conditions valid.
  envelopes = []
  for _ in range(500):
    r = tracker.update(_tracker_inputs(lat_error=0.05, pred_bias=0.01), DT)
    envelopes.append(r.relax_envelope)

  max_env = max(envelopes)
  assert max_env > 0.0
  # Envelope should have decayed to essentially zero well within hold + max active + decay.
  assert envelopes[-1] < 1e-3
  # The maximum envelope was reached during the initial hold.
  assert max_env >= 0.08


def test_relaxation_does_not_add_steering_bias():
  tracker = LaneCenteringAssistTracker()

  for i in range(150):
    lat_error = 0.10 if i % 2 == 0 else -0.10
    r = tracker.update(_tracker_inputs(lat_error=lat_error, pred_bias=0.02), DT)

  # The relaxation is purely a soft deadband; when active with symmetric flips the
  # demanded curvature should stay near zero (no net steering bias).
  assert abs(r.curvature_nudge) < 1e-5
  # Base path/model curvature is preserved by checking the nudge alone.
  assert math.isfinite(r.curvature_nudge)


def test_safety_abort_immediately_zeros_envelope_and_resets_errors():
  tracker = LaneCenteringAssistTracker()

  # Trigger relaxation.
  for i in range(120):
    lat_error = 0.10 if i % 2 == 0 else -0.10
    tracker.update(_tracker_inputs(lat_error=lat_error, pred_bias=0.02), DT)

  # Safety abort on the very next frame: driver steering.
  r = tracker.update(_tracker_inputs(lat_error=0.05, pred_bias=0.01, steering_pressed=True), DT)

  assert r.relax_active is False
  assert r.relax_envelope == pytest.approx(0.0)
  assert r.relax_reason_bits & LANE_CENTERING_RELAX_REASON_DRIVER
  # Effective errors must equal raw errors immediately (no deadband applied).
  assert r.relaxed_lateral_error == pytest.approx(r.lateral_error)
  assert r.relaxed_predicted_error == pytest.approx(r.predicted_lateral_error)


def test_hard_block_invalid_path_prevents_stale_rearm():
  tracker = LaneCenteringAssistTracker()

  # Trigger relaxation.
  for i in range(120):
    lat_error = 0.10 if i % 2 == 0 else -0.10
    tracker.update(_tracker_inputs(lat_error=lat_error, pred_bias=0.02), DT)

  # Hard block via invalid path: should clear histories and reset relaxation.
  r_invalid = tracker.update(LaneCenteringAssistInputs(
    lat_active=True, v_ego=20.0, measured_curvature=0.0, model_curvature=0.0,
    previous_processed_curvature=0.0, path_quality=1.0, path_reason="ok",
    lane_change_shaping_active=False, lane_change_blend=0.0, curvature_limited=False,
    steering_pressed=False, left_blinker=False, right_blinker=False,
    position_x=(), position_y=(), orientation_z=(), lane_line_probs=[0.9, 0.9, 0.9, 0.9],
  ), DT)
  assert r_invalid.reason == "invalid_path"
  assert r_invalid.relax_active is False
  assert r_invalid.relax_envelope == pytest.approx(0.0)

  # A single sign flip right after recovery must not re-arm from stale history.
  r = tracker.update(_tracker_inputs(lat_error=-0.10, pred_bias=0.02), DT)
  assert r.relax_active is False
  assert r.relax_nudge_flip_score < 2
  assert r.relax_error_cross_score < 2


def test_predicted_error_above_trigger_threshold_does_not_trigger():
  tracker = LaneCenteringAssistTracker()

  # Lateral error is near center, but predicted error cycles between ~0.05 m and
  # ~0.25 m, sitting between the 0.20 m trigger threshold and the 0.35 m abort threshold.
  active_frames: list[bool] = []
  max_predicted = 0.0
  for i in range(200):
    lat_error = 0.10 if i % 2 == 0 else -0.10
    r = tracker.update(_tracker_inputs(lat_error=lat_error, pred_bias=0.20), DT)
    active_frames.append(r.relax_active)
    max_predicted = max(max_predicted, abs(r.predicted_lateral_error))

  assert not any(active_frames), "relaxation should not trigger with predicted error above trigger threshold"
  assert r.relax_reason_bits == 0
  assert 0.20 < max_predicted < 0.35


def test_curve_demand_abort_is_hard_abort():
  tracker = LaneCenteringAssistTracker()

  # Trigger relaxation on a straight road.
  for i in range(120):
    lat_error = 0.10 if i % 2 == 0 else -0.10
    tracker.update(_tracker_inputs(lat_error=lat_error, pred_bias=0.02), DT)

  # Inject enough curvature demand to exceed the 0.70 m/s^2 lateral-accel limit.
  # v_ego=20 m/s -> k > 0.70 / 400 = 0.00175 1/m.
  r = tracker.update(_tracker_inputs(
    lat_error=0.10, pred_bias=0.02,
    model_curvature=0.002, measured_curvature=0.002, previous_processed_curvature=0.002,
  ), DT)

  assert r.relax_active is False
  assert r.relax_envelope == pytest.approx(0.0)
  assert r.relax_reason_bits & LANE_CENTERING_RELAX_REASON_CURVE
  assert r.relaxed_lateral_error == pytest.approx(r.lateral_error)
  assert r.relaxed_predicted_error == pytest.approx(r.predicted_lateral_error)
