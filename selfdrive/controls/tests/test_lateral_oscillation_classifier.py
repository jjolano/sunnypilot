import math

import pytest

from openpilot.selfdrive.controls.lib.lateral_oscillation_classifier import (
  LATERAL_OSCILLATION_TO_UINT8,
  LATERAL_UINT8_TO_OSCILLATION,
  STRAIGHT_ROAD_MIN_SPEED,
  WOBBLE_ACTIVE_CLASSIFICATIONS,
  WOBBLE_CONFIDENCE_THRESHOLD,
  WobbleResponse,
  compute_wobble_response,
  is_wobble_active,
  lateral_oscillation_to_uint8,
  uint8_to_lateral_oscillation,
)
from openpilot.selfdrive.controls.lib.lateral_oscillation_classifier import LateralOscillationClassifier


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


class TestOscillationUint8Mapping:

  def test_all_classifications_have_unique_uint8(self):
    values = list(LATERAL_OSCILLATION_TO_UINT8.values())
    assert len(values) == len(LATERAL_UINT8_TO_OSCILLATION)
    assert len(set(values)) == len(values)

  def test_known_classifications(self):
    assert lateral_oscillation_to_uint8("none") == 0
    assert lateral_oscillation_to_uint8("planner_oscillation") == 1
    assert lateral_oscillation_to_uint8("controller_oscillation") == 2
    assert lateral_oscillation_to_uint8("vehicle_bias") == 3
    assert lateral_oscillation_to_uint8("recenter_lag") == 4
    assert lateral_oscillation_to_uint8("sign_change_lag") == 5
    assert lateral_oscillation_to_uint8("straight_road_hunting") == 6

  def test_unknown_returns_zero(self):
    assert lateral_oscillation_to_uint8("nope") == 0
    assert lateral_oscillation_to_uint8("") == 0

  def test_uint8_round_trip(self):
    for name, value in LATERAL_OSCILLATION_TO_UINT8.items():
      assert uint8_to_lateral_oscillation(value) == name
    assert uint8_to_lateral_oscillation(99) == "none"


class TestIsWobbleActive:

  def test_controller_oscillation_high_confidence_activates(self):
    assert is_wobble_active("controller_oscillation", 0.8) is True

  def test_planner_oscillation_high_confidence_activates(self):
    assert is_wobble_active("planner_oscillation", 0.7) is True

  def test_straight_road_hunting_high_confidence_activates(self):
    assert is_wobble_active("straight_road_hunting", 0.9) is True

  def test_low_confidence_does_not_activate(self):
    assert is_wobble_active("controller_oscillation", WOBBLE_CONFIDENCE_THRESHOLD) is False
    assert is_wobble_active("controller_oscillation", WOBBLE_CONFIDENCE_THRESHOLD - 0.1) is False

  def test_recenter_lag_does_not_activate(self):
    assert is_wobble_active("recenter_lag", 0.95) is False

  def test_sign_change_lag_does_not_activate(self):
    assert is_wobble_active("sign_change_lag", 0.95) is False

  def test_vehicle_bias_does_not_activate(self):
    assert is_wobble_active("vehicle_bias", 0.95) is False

  def test_none_does_not_activate(self):
    assert is_wobble_active("none", 1.0) is False

  def test_wobble_set_includes_only_oscillation_sources(self):
    assert WOBBLE_ACTIVE_CLASSIFICATIONS == frozenset({
      "planner_oscillation",
      "controller_oscillation",
      "straight_road_hunting",
    })


class TestComputeWobbleResponse:

  def test_none_returns_neutral(self):
    r = compute_wobble_response("none", 1.0)
    assert r.is_neutral is True
    assert r.source_active is False
    assert r.source == "none"
    assert r.feedback_gain_multiplier == 1.0
    assert r.damping_gain_multiplier == 1.0

  def test_planner_oscillation_reduces_feedback(self):
    r = compute_wobble_response("planner_oscillation", 0.8)
    assert r.source_active is True
    assert r.source == "planner"
    assert r.feedback_gain_multiplier < 1.0
    assert r.damping_gain_multiplier > 1.0

  def test_controller_oscillation_aggressive_reduction(self):
    r = compute_wobble_response("controller_oscillation", 0.9)
    assert r.source_active is True
    assert r.source == "controller"
    assert r.feedback_gain_multiplier <= 0.6
    assert r.damping_gain_multiplier >= 1.5

  def test_straight_road_hunting_same_as_controller(self):
    r = compute_wobble_response("straight_road_hunting", 0.8)
    assert r.source_active is True
    assert r.source == "straight_road"
    assert r.feedback_gain_multiplier == compute_wobble_response("controller_oscillation", 0.9).feedback_gain_multiplier
    assert r.damping_gain_multiplier == compute_wobble_response("controller_oscillation", 0.9).damping_gain_multiplier

  def test_low_confidence_returns_neutral(self):
    r = compute_wobble_response("planner_oscillation", WOBBLE_CONFIDENCE_THRESHOLD - 0.1)
    assert r.is_neutral is True
    assert r.source_active is False

  def test_vehicle_bias_returns_neutral(self):
    r = compute_wobble_response("vehicle_bias", 0.9)
    assert r.is_neutral is True
    assert r.source_active is False

  def test_recenter_lag_returns_neutral(self):
    r = compute_wobble_response("recenter_lag", 0.9)
    assert r.is_neutral is True

  def test_sign_change_lag_returns_neutral(self):
    r = compute_wobble_response("sign_change_lag", 0.9)
    assert r.is_neutral is True

  def test_returns_wobble_response_instance(self):
    r = compute_wobble_response("none", 0.0)
    assert isinstance(r, WobbleResponse)

  def test_is_neutral_property(self):
    assert compute_wobble_response("none", 0.0).is_neutral is True
    assert compute_wobble_response("planner_oscillation", 0.8).is_neutral is False
    assert compute_wobble_response("controller_oscillation", 0.9).is_neutral is False
