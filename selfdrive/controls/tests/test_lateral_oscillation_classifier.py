import math

import pytest

from openpilot.selfdrive.controls.lib.lateral_oscillation_classifier import (
  STRAIGHT_ROAD_MIN_SPEED,
  LateralOscillationClassifier,
)


def _fill_classifier(clf, n, **kwargs):
  """Feed n frames of uniform values to fill the classifier window."""
  for _ in range(n):
    clf.update(**kwargs)


def _populate(clf, frames, **fixed_kwargs):
  """Feed multiple frames; each frame in `frames` is a dict of overrides."""
  for frame_kwargs in frames:
    kw = dict(fixed_kwargs)
    kw.update(frame_kwargs)
    clf.update(**kw)


class TestLateralOscillationClassifier:

  BASE_KWARGS = dict(
    raw_curvature=0.0,
    processed_curvature=0.0,
    target_lateral_accel=0.0,
    actual_lateral_accel=0.0,
    torque_output=0.0,
    path_quality=1.0,
    lane_change_active=False,
    v_ego=STRAIGHT_ROAD_MIN_SPEED,
    curvature_limited=False,
    steering_pressed=False,
  )

  # ---------- planner_oscillation ----------

  def test_planner_oscillation_detected(self):
    clf = LateralOscillationClassifier(window_frames=50)
    _fill_classifier(clf, 10, **self.BASE_KWARGS)

    # Feed alternating raw+processed curvature to trigger planner oscillations
    frames = []
    for i in range(30):
      val = 1e-3 if i % 2 == 0 else -1e-3
      frames.append(dict(raw_curvature=val, processed_curvature=val))
    _populate(clf, frames, **self.BASE_KWARGS)

    result = clf.update(**self.BASE_KWARGS)
    assert result.classification == "planner_oscillation", f"Expected planner_oscillation, got {result.classification}"
    assert result.raw_curvature_sign_flips > 3
    assert result.processed_curvature_sign_flips > 3

  # ---------- controller_oscillation ----------

  def test_controller_oscillation_detected(self):
    clf = LateralOscillationClassifier(window_frames=50)
    _fill_classifier(clf, 10, **self.BASE_KWARGS)

    # Stable processed curvature but alternating torque output
    frames = []
    for i in range(30):
      torque = 1.0 if i % 2 == 0 else -1.0
      frames.append(dict(processed_curvature=1e-6, torque_output=torque))
    _populate(clf, frames, **self.BASE_KWARGS)

    result = clf.update(**self.BASE_KWARGS)
    assert result.classification == "controller_oscillation", f"Expected controller_oscillation, got {result.classification}"
    assert result.processed_curvature_sign_flips <= 2
    assert result.torque_sign_flips > 4

  # ---------- vehicle_bias ----------

  def test_vehicle_bias_detected(self):
    clf = LateralOscillationClassifier(window_frames=50)
    # Fill with a constant offset on processed curvature while driving straight
    offset = 5e-5  # > 1e-5
    frames = []
    for i in range(40):
      frames.append(dict(
        raw_curvature=0.0,
        processed_curvature=offset,
        v_ego=STRAIGHT_ROAD_MIN_SPEED,
        lane_change_active=False,
      ))
    _populate(clf, frames, **self.BASE_KWARGS)

    result = clf.update(**dict(self.BASE_KWARGS, processed_curvature=offset))
    assert result.classification == "vehicle_bias", f"Expected vehicle_bias, got {result.classification}"
    assert abs(result.curvature_offset) > 1e-5
    assert result.curvature_offset_confidence > 0.7

  # ---------- none (stable driving) ----------

  def test_no_oscillation_stable_driving(self):
    clf = LateralOscillationClassifier(window_frames=50)
    frames = []
    for i in range(30):
      frames.append(dict(
        raw_curvature=1e-6,
        processed_curvature=1e-6,
        target_lateral_accel=0.1,
        actual_lateral_accel=0.1,
        torque_output=0.5,
      ))
    _populate(clf, frames, **self.BASE_KWARGS)

    result = clf.update(**dict(self.BASE_KWARGS, raw_curvature=1e-6, processed_curvature=1e-6,
                               target_lateral_accel=0.1, actual_lateral_accel=0.1, torque_output=0.5))
    assert result.classification == "none", f"Expected none, got {result.classification}"

  # ---------- straight_road_hunting ----------

  def test_straight_road_hunting_detected(self):
    clf = LateralOscillationClassifier(window_frames=50)

    # Build 40+ frames of straight-road hunting behavior.
    # processed_curvature must alternate enough for >2 flips but NOT >3 flips
    # (to avoid planner_oscillation taking priority).  That means exactly 3 flips.
    # raw_curvature stays constant (0 flips) so planner_oscillation won't trigger.
    # torque_output alternates aggressively to get >> 6 flips.
    frames = []
    # Block 1: 10 frames, processed=+1e-6, torque=+0.5
    for _ in range(10):
      frames.append(dict(raw_curvature=0.0, processed_curvature=1e-6, torque_output=0.5,
                         v_ego=STRAIGHT_ROAD_MIN_SPEED + 5.0))
    # Block 2: 10 frames, processed=-1e-6, torque alternates → creates flip #1 at boundary
    for i in range(10):
      frames.append(dict(raw_curvature=0.0, processed_curvature=-1e-6,
                         torque_output=0.5 if i % 2 == 0 else -0.5,
                         v_ego=STRAIGHT_ROAD_MIN_SPEED + 5.0))
    # Block 3: 10 frames, processed=+1e-6, torque alternates → creates flip #2 at boundary
    for i in range(10):
      frames.append(dict(raw_curvature=0.0, processed_curvature=1e-6,
                         torque_output=0.5 if i % 2 == 0 else -0.5,
                         v_ego=STRAIGHT_ROAD_MIN_SPEED + 5.0))
    # Block 4: 10 frames, processed=-1e-6, torque alternates → creates flip #3 at boundary
    for i in range(10):
      frames.append(dict(raw_curvature=0.0, processed_curvature=-1e-6,
                         torque_output=0.5 if i % 2 == 0 else -0.5,
                         v_ego=STRAIGHT_ROAD_MIN_SPEED + 5.0))
    _populate(clf, frames, **{k: v for k, v in self.BASE_KWARGS.items() if k != 'v_ego'})

    result = clf.update(**dict(self.BASE_KWARGS, v_ego=STRAIGHT_ROAD_MIN_SPEED + 5.0,
                               raw_curvature=0.0, processed_curvature=-1e-6, torque_output=-0.5))
    assert result.classification == "straight_road_hunting", f"Expected straight_road_hunting, got {result.classification}"
    assert result.straight_road
    assert result.torque_sign_flips > 6
    assert result.processed_curvature_sign_flips > 2

  # ---------- lane change suppresses oscillation ----------

  def test_lane_change_not_classified_as_oscillation(self):
    clf = LateralOscillationClassifier(window_frames=50)
    _fill_classifier(clf, 10, **self.BASE_KWARGS)

    # Even with oscillating signals, lane_change_active should yield "none"
    frames = []
    for i in range(30):
      val = 1e-3 if i % 2 == 0 else -1e-3
      torque = 1.0 if i % 2 == 0 else -1.0
      frames.append(dict(
        raw_curvature=val,
        processed_curvature=val,
        torque_output=torque,
        lane_change_active=True,
      ))
    _populate(clf, frames, **self.BASE_KWARGS)

    result = clf.update(**dict(self.BASE_KWARGS, lane_change_active=True))
    assert result.classification == "none", f"Expected none (lane change active), got {result.classification}"
    assert result.lane_change_active
