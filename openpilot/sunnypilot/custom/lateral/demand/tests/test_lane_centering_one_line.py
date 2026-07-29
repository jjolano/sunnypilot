"""Tests for one-confident-line lane centering and geometry-mode exit hysteresis.

Route 0000025a drift diagnosis: both-line geometry was unavailable (one inner line dead)
in exactly the regime where slow off-center drift happened, and geometry mode flipped
12-50x/min because disengage was instantaneous. Shadow mode must be exactly
non-actuating; apply must flow through the unchanged geometry-mode machinery.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from openpilot.sunnypilot.custom.lateral.demand.lane_geometry import (
  LANE_GEOMETRY_REASON_AMBIGUOUS_LINES,
  LANE_GEOMETRY_REASON_BAD_LINE_SIDE,
  LANE_GEOMETRY_REASON_LOW_PROB,
  LANE_GEOMETRY_REASON_NO_LEARNED_WIDTH,
  evaluate_single_line_geometry,
)
from openpilot.sunnypilot.custom.lateral.demand.lane_centering_assist import (
  LANE_CENTERING_ASSIST_GEOMETRY_EXIT_HOLD_FRAMES,
  LANE_CENTERING_ASSIST_GEOMETRY_PERSISTENCE_FRAMES,
  LaneCenteringAssistInputs,
  LaneCenteringAssistTracker,
  sanitize_one_line_centering_mode,
)

DT = 0.01
N = 33


def _lane_lines(center_offset: float = 0.0, width: float = 4.0):
  xs = [float(x) for x in range(N)]
  left_y = [center_offset - width * 0.5] * N
  right_y = [center_offset + width * 0.5] * N
  return [
    SimpleNamespace(x=xs, y=[y - width for y in left_y]),
    SimpleNamespace(x=xs, y=left_y),
    SimpleNamespace(x=xs, y=right_y),
    SimpleNamespace(x=xs, y=[y + width for y in right_y]),
  ]


def _path(offset: float = 0.0, growth: float = 0.0):
  xs = [float(x) for x in range(N)]
  ys = [offset + growth * (x / (N - 1)) for x in xs]
  return xs, ys


def _single(probs, *, width=4.0, offset=0.5, stds=None):
  xs, ys = _path(offset=offset)
  return evaluate_single_line_geometry(
    lane_lines=_lane_lines(0.0),
    lane_line_probs=probs,
    lane_line_stds=stds or [0.2, 0.1, 0.1, 0.2],
    position_x=xs,
    position_y=ys,
    near_x=10.0,
    preview_x=30.0,
    learned_width=width,
  )


def test_single_left_line_estimates_center():
  g = _single([0.5, 0.9, 0.2, 0.5], offset=0.5)
  assert g.valid is True
  assert g.source == "single_line"
  # Lane center is y=0 (left inner at -2 + learned half width 2); path at +0.5 is right of center.
  assert g.lateral_error == pytest.approx(-0.5, abs=1e-6)
  assert g.predicted_lateral_error == pytest.approx(-0.5, abs=1e-6)
  assert 0.0 < g.confidence < 0.9  # discounted below the raw line prob


def test_single_right_line_estimates_center():
  g = _single([0.5, 0.2, 0.9, 0.5], offset=-0.3)
  assert g.valid is True
  assert g.lateral_error == pytest.approx(0.3, abs=1e-6)


def test_two_confident_lines_are_ambiguous():
  g = _single([0.5, 0.9, 0.9, 0.5])
  assert g.valid is False
  assert g.reason == LANE_GEOMETRY_REASON_AMBIGUOUS_LINES


def test_no_confident_lines_fail_closed():
  g = _single([0.5, 0.5, 0.5, 0.5])
  assert g.valid is False
  assert g.reason == LANE_GEOMETRY_REASON_LOW_PROB


def test_missing_learned_width_fails_closed():
  g = _single([0.5, 0.9, 0.2, 0.5], width=float("nan"))
  assert g.valid is False
  assert g.reason == LANE_GEOMETRY_REASON_NO_LEARNED_WIDTH


def test_line_on_wrong_side_fails_closed():
  # "Left" inner line sitting at positive y is an adjacent-lane line.
  xs = [float(x) for x in range(N)]
  lanes = [
    SimpleNamespace(x=xs, y=[-6.0] * N),
    SimpleNamespace(x=xs, y=[2.0] * N),
    SimpleNamespace(x=xs, y=[6.0] * N),
    SimpleNamespace(x=xs, y=[9.0] * N),
  ]
  pxs, pys = _path()
  g = evaluate_single_line_geometry(
    lane_lines=lanes,
    lane_line_probs=[0.5, 0.9, 0.2, 0.5],
    lane_line_stds=[0.2, 0.1, 0.1, 0.2],
    position_x=pxs,
    position_y=pys,
    near_x=10.0,
    preview_x=30.0,
    learned_width=4.0,
  )
  assert g.valid is False
  assert g.reason == LANE_GEOMETRY_REASON_BAD_LINE_SIDE


def test_sanitize_one_line_mode_fails_closed():
  assert sanitize_one_line_centering_mode("shadow") == "shadow"
  assert sanitize_one_line_centering_mode("APPLY ") == "apply"
  assert sanitize_one_line_centering_mode("bogus") == "off"
  assert sanitize_one_line_centering_mode(None) == "off"


def _tracker_inputs(offset: float = 0.0, growth: float = 0.0, probs=None, mode: str = "off",
                    **kwargs) -> LaneCenteringAssistInputs:
  xs, ys = _path(offset=offset, growth=growth)
  return LaneCenteringAssistInputs(
    lat_active=True,
    v_ego=kwargs.get("v_ego", 20.0),
    measured_curvature=0.0,
    model_curvature=0.0,
    previous_processed_curvature=0.0,
    path_quality=1.0,
    path_reason="ok",
    lane_change_shaping_active=False,
    lane_change_blend=0.0,
    curvature_limited=False,
    steering_pressed=kwargs.get("steering_pressed", False),
    left_blinker=False,
    right_blinker=False,
    position_x=xs,
    position_y=ys,
    orientation_z=[0.0] * N,
    lane_line_probs=probs if probs is not None else [0.5, 0.9, 0.9, 0.5],
    demand_source="model_path",
    lane_lines=_lane_lines(0.0),
    lane_line_stds=[0.2, 0.1, 0.1, 0.2],
    one_line_mode=mode,
  )


BOTH = [0.5, 0.9, 0.9, 0.5]
LEFT_ONLY = [0.5, 0.9, 0.2, 0.5]
NONE_CONF = [0.5, 0.2, 0.2, 0.5]


def _engage_geometry(tracker: LaneCenteringAssistTracker, mode: str = "off", frames: int | None = None):
  frames = frames or (LANE_CENTERING_ASSIST_GEOMETRY_PERSISTENCE_FRAMES + 5)
  r = None
  for _ in range(frames):
    r = tracker.update(_tracker_inputs(offset=0.3, probs=BOTH, mode=mode), DT)
  assert r is not None and r.debug["lane_centering_geometry_mode"] is True
  return r


def test_geometry_hold_bridges_brief_dropout():
  tracker = LaneCenteringAssistTracker()
  _engage_geometry(tracker)
  # Validity flicker shorter than the hold: geometry mode must persist.
  for _ in range(LANE_CENTERING_ASSIST_GEOMETRY_EXIT_HOLD_FRAMES - 5):
    r = tracker.update(_tracker_inputs(offset=0.3, probs=NONE_CONF), DT)
    assert r.debug["lane_centering_geometry_mode"] is True
    assert r.debug["lane_centering_geometry_hold_active"] is True
  # Lines return: still engaged, hold released.
  r = tracker.update(_tracker_inputs(offset=0.3, probs=BOTH), DT)
  assert r.debug["lane_centering_geometry_mode"] is True
  assert r.debug["lane_centering_geometry_hold_active"] is False


def test_geometry_drops_after_hold_expires():
  tracker = LaneCenteringAssistTracker()
  _engage_geometry(tracker)
  for _ in range(LANE_CENTERING_ASSIST_GEOMETRY_EXIT_HOLD_FRAMES + 2):
    r = tracker.update(_tracker_inputs(offset=0.3, probs=NONE_CONF), DT)
  assert r.debug["lane_centering_geometry_mode"] is False
  assert r.debug["lane_centering_geometry_hold_active"] is False


def test_hard_gate_drops_geometry_instantly():
  tracker = LaneCenteringAssistTracker()
  _engage_geometry(tracker)
  r = tracker.update(_tracker_inputs(offset=0.3, probs=BOTH, steering_pressed=True), DT)
  assert r.debug["lane_centering_geometry_mode"] is False
  # And the hold cache must not resurrect geometry mode on an invalid frame afterwards.
  r = tracker.update(_tracker_inputs(offset=0.3, probs=NONE_CONF), DT)
  assert r.debug["lane_centering_geometry_mode"] is False


def test_shadow_is_exactly_non_actuating():
  shadow = LaneCenteringAssistTracker()
  off = LaneCenteringAssistTracker()
  # Learn width with both lines, then drop the right line; growing offset keeps the
  # model-path nudge nonzero so equality is a meaningful assertion.
  for _ in range(100):
    shadow.update(_tracker_inputs(offset=0.1, growth=0.2, probs=BOTH, mode="shadow"), DT)
    off.update(_tracker_inputs(offset=0.1, growth=0.2, probs=BOTH, mode="off"), DT)
  saw_candidate = False
  # Outlast the geometry exit hold so shadow-vs-off equality also covers the post-hold frames.
  for _ in range(LANE_CENTERING_ASSIST_GEOMETRY_EXIT_HOLD_FRAMES + 200):
    rs = shadow.update(_tracker_inputs(offset=0.1, growth=0.2, probs=LEFT_ONLY, mode="shadow"), DT)
    ro = off.update(_tracker_inputs(offset=0.1, growth=0.2, probs=LEFT_ONLY, mode="off"), DT)
    assert rs.curvature_nudge == pytest.approx(ro.curvature_nudge, abs=1e-12)
    if rs.debug["lane_centering_one_line_active"]:
      saw_candidate = True
      assert rs.debug["lane_centering_one_line_applied"] is False
      assert rs.debug["lane_centering_one_line_learned_width"] == pytest.approx(4.0, abs=0.1)
  assert saw_candidate


def test_shadow_candidate_sign_points_at_center():
  tracker = LaneCenteringAssistTracker()
  for _ in range(100):
    tracker.update(_tracker_inputs(offset=0.0, probs=BOTH, mode="shadow"), DT)
  r = None
  for _ in range(LANE_CENTERING_ASSIST_GEOMETRY_EXIT_HOLD_FRAMES + 50):
    # Path (and car) sitting 0.3 m right of center with only the left line confident.
    r = tracker.update(_tracker_inputs(offset=0.3, probs=LEFT_ONLY, mode="shadow"), DT)
  assert r.debug["lane_centering_one_line_active"] is True
  assert r.debug["lane_centering_one_line_lateral_error"] == pytest.approx(-0.3, abs=1e-6)
  assert r.debug["lane_centering_one_line_candidate_nudge"] < 0.0


def test_apply_feeds_geometry_mode_and_nudges():
  tracker = LaneCenteringAssistTracker()
  for _ in range(100):
    tracker.update(_tracker_inputs(offset=0.0, probs=BOTH, mode="apply"), DT)
  r = None
  for _ in range(300):
    r = tracker.update(_tracker_inputs(offset=0.3, probs=LEFT_ONLY, mode="apply"), DT)
  assert r.debug["lane_centering_one_line_applied"] is True
  assert r.debug["lane_centering_geometry_mode"] is True
  assert r.curvature_nudge < 0.0  # pulls back toward center from right of center


def test_one_line_off_never_evaluates():
  tracker = LaneCenteringAssistTracker()
  for _ in range(100):
    tracker.update(_tracker_inputs(offset=0.1, probs=BOTH, mode="off"), DT)
  r = None
  for _ in range(LANE_CENTERING_ASSIST_GEOMETRY_EXIT_HOLD_FRAMES + 50):
    r = tracker.update(_tracker_inputs(offset=0.3, probs=LEFT_ONLY, mode="off"), DT)
  assert r.debug["lane_centering_one_line_mode"] == "off"
  assert r.debug["lane_centering_one_line_active"] is False
  assert r.debug["lane_centering_one_line_reason"] == "not_evaluated"


def test_stale_width_blocks_one_line():
  tracker = LaneCenteringAssistTracker()
  # Never see both lines: width is never learned.
  r = None
  for _ in range(100):
    r = tracker.update(_tracker_inputs(offset=0.3, probs=LEFT_ONLY, mode="shadow"), DT)
  assert r.debug["lane_centering_one_line_active"] is False
  assert r.debug["lane_centering_one_line_reason"] == "no_learned_width"
  assert math.isclose(float(r.debug["lane_centering_one_line_learned_width"]), 0.0)
