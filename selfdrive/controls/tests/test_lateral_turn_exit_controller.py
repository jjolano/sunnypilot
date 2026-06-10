import math

import pytest

from openpilot.selfdrive.controls.lib.lateral_demand import ProcessedLateralDemand
from openpilot.selfdrive.controls.lib.lateral_demand_profile import (
  LateralDemandProfile,
  LateralDemandProfileBuilder,
)
from openpilot.selfdrive.controls.lib.lateral_turn_exit_controller import (
  PREVIEW_HORIZON_S,
  RECENTER_PERSISTENCE_FRAMES,
  TURN_EXIT_COLLAPSE_PER_FRAME,
  TURN_EXIT_MAX_ABS_TARGET,
  TURN_IN_MIN_ABS_TARGET_RATE,
  TURN_IN_MIN_LAT_ACCEL,
  TURN_IN_PREVIEW_BOOST_CAP,
  TURN_IN_PREVIEW_BOOST_GAIN,
  LateralTurnExitController,
  TurnExitMode,
  _signs_stable,
)


def _make_demand(**overrides) -> ProcessedLateralDemand:
  defaults = dict(
    raw_curvature=0.0,
    processed_curvature=0.0,
    measured_curvature=0.0,
    curvature_limited=False,
    path_quality=1.0,
    path_reason="ok",
    lane_change_shaping_active=False,
    lane_change_blend=0.0,
    lateral_accel_limit=4.0,
    demand_source="model_path",
  )
  defaults.update(overrides)
  return ProcessedLateralDemand(**defaults)


def _make_profile(curvature=0.001, v_ego=20.0, path_quality=1.0) -> LateralDemandProfile:
  builder = LateralDemandProfileBuilder(dt=0.05)
  return builder.update(_make_demand(processed_curvature=curvature, path_quality=path_quality), v_ego=v_ego)


class TestLateralTurnExitController:

  def test_inactive_when_active_false(self):
    c = LateralTurnExitController(dt=0.05)
    d = c.update(target=0.4, profile=_make_profile(), active=False)
    assert d.mode == TurnExitMode.INACTIVE.value
    assert d.lead_gain_multiplier == 1.0
    assert d.early_release_lead_zero is False

  def test_steady_curve_when_target_stable(self):
    c = LateralTurnExitController(dt=0.05)
    d = c.update(target=0.4, profile=_make_profile(curvature=0.001), active=True, v_ego=20.0)
    assert d.mode == TurnExitMode.STEADY_CURVE.value

  def test_turn_in_when_target_building(self):
    c = LateralTurnExitController(dt=0.05)
    c.update(target=0.0, profile=None, active=True, v_ego=20.0)
    d = c.update(target=0.4, profile=_make_profile(curvature=0.001), active=True, v_ego=20.0)
    assert d.mode == TurnExitMode.TURN_IN.value

  def test_turn_in_requires_min_target_rate(self):
    c = LateralTurnExitController(dt=0.05)
    c.update(target=0.0, profile=None, active=True, v_ego=20.0)
    small_curvature = TURN_IN_MIN_LAT_ACCEL / (20.0 ** 2) * 0.5
    d = c.update(
      target=TURN_IN_MIN_LAT_ACCEL * 0.5, profile=_make_profile(curvature=small_curvature),
      active=True, v_ego=20.0,
    )
    assert d.mode != TurnExitMode.TURN_IN.value

  def test_early_release_on_first_collapse_frame(self):
    c = LateralTurnExitController(dt=0.05)
    c.update(target=0.5, profile=None, active=True, v_ego=20.0, path_quality=1.0)
    d = c.update(target=0.3, profile=_make_profile(curvature=0.0005), active=True, v_ego=20.0, path_quality=1.0)
    assert d.mode == TurnExitMode.EARLY_RELEASE.value
    assert d.early_release_lead_zero is True

  def test_early_release_requires_stable_sign(self):
    c = LateralTurnExitController(dt=0.05)
    c.update(target=-0.5, profile=None, active=True, v_ego=20.0, path_quality=1.0)
    d = c.update(target=0.3, profile=_make_profile(curvature=0.0005), active=True, v_ego=20.0, path_quality=1.0)
    assert d.early_release_lead_zero is False

  def test_turn_exit_after_persistence(self):
    c = LateralTurnExitController(dt=0.05)
    decreasing_targets = [0.5, 0.4, 0.3, 0.2, 0.15]
    for t in decreasing_targets:
      d = c.update(target=t, profile=_make_profile(curvature=t / (20.0 ** 2)),
                   active=True, v_ego=20.0, path_quality=1.0)
    assert d.mode == TurnExitMode.TURN_EXIT.value
    assert d.lead_gain_multiplier < 1.0
    assert d.slew_boost > 1.0

  def test_turn_exit_persistence_counter(self):
    c = LateralTurnExitController(dt=0.05)
    decreasing_targets = [0.5, 0.4, 0.3, 0.2, 0.15]
    for t in decreasing_targets:
      d = c.update(target=t, profile=_make_profile(curvature=t / (20.0 ** 2)),
                   active=True, v_ego=20.0, path_quality=1.0)
    assert d.persistence_frames >= RECENTER_PERSISTENCE_FRAMES

  def test_lead_reduction_in_recenter_mode(self):
    c = LateralTurnExitController(dt=0.05)
    decreasing_targets = [0.5, 0.4, 0.3, 0.2, 0.15]
    for t in decreasing_targets:
      d = c.update(target=t, profile=_make_profile(curvature=t / (20.0 ** 2)),
                   active=True, v_ego=20.0, path_quality=1.0)
    assert d.lead_gain_multiplier < 1.0
    assert d.lead_delta_cap_multiplier < 1.0

  def test_lane_change_disables_recenter(self):
    c = LateralTurnExitController(dt=0.05)
    c.update(target=0.5, profile=None, active=True, v_ego=20.0, path_quality=1.0)
    d = c.update(target=0.3, profile=_make_profile(curvature=0.0005),
                 active=True, v_ego=20.0, path_quality=1.0, lane_change_active=True)
    assert d.persistence_frames == 0

  def test_steering_pressed_disables_recenter(self):
    c = LateralTurnExitController(dt=0.05)
    c.update(target=0.5, profile=None, active=True, v_ego=20.0, path_quality=1.0)
    d = c.update(target=0.3, profile=_make_profile(curvature=0.0005),
                 active=True, v_ego=20.0, path_quality=1.0, steering_pressed=True)
    assert d.persistence_frames == 0

  def test_low_speed_disables_recenter(self):
    c = LateralTurnExitController(dt=0.05)
    c.update(target=0.5, profile=None, active=True, v_ego=5.0, path_quality=1.0)
    d = c.update(target=0.3, profile=_make_profile(curvature=0.0005, v_ego=5.0),
                 active=True, v_ego=5.0, path_quality=1.0)
    assert d.persistence_frames == 0

  def test_low_path_quality_disables_recenter(self):
    c = LateralTurnExitController(dt=0.05)
    c.update(target=0.5, profile=None, active=True, v_ego=20.0, path_quality=1.0)
    d = c.update(target=0.3, profile=_make_profile(curvature=0.0005, path_quality=0.3),
                 active=True, v_ego=20.0, path_quality=0.3)
    assert d.persistence_frames == 0

  def test_reset_clears_state(self):
    c = LateralTurnExitController(dt=0.05)
    c.update(target=0.5, profile=None, active=True, v_ego=20.0, path_quality=1.0)
    c.update(target=0.3, profile=_make_profile(), active=True, v_ego=20.0, path_quality=1.0)
    c.reset()
    d = c.update(target=0.0, profile=None, active=True, v_ego=20.0, path_quality=1.0)
    assert d.persistence_frames == 0
    assert d.early_release_lead_zero is False

  def test_preview_boost_computed_for_turn_in(self):
    c = LateralTurnExitController(dt=0.05)
    c.update(target=0.0, profile=None, active=True, v_ego=20.0, path_quality=1.0)
    d = c.update(target=0.4, profile=_make_profile(curvature=0.001), active=True, v_ego=20.0, path_quality=1.0)
    assert d.preview_boost != 0.0
    assert abs(d.preview_boost) <= TURN_IN_PREVIEW_BOOST_CAP

  def test_preview_boost_zero_on_early_release(self):
    c = LateralTurnExitController(dt=0.05)
    c.update(target=0.5, profile=None, active=True, v_ego=20.0, path_quality=1.0)
    d = c.update(target=0.3, profile=_make_profile(curvature=0.0005),
                 active=True, v_ego=20.0, path_quality=1.0)
    assert d.preview_boost == 0.0

  def test_preview_boost_zero_at_low_speed(self):
    c = LateralTurnExitController(dt=0.05)
    c.update(target=0.0, profile=None, active=True, v_ego=2.0, path_quality=1.0)
    d = c.update(target=0.4, profile=_make_profile(curvature=0.001, v_ego=2.0),
                 active=True, v_ego=2.0, path_quality=1.0)
    assert d.preview_boost == 0.0

  def test_preview_boost_zero_for_steady_curve(self):
    c = LateralTurnExitController(dt=0.05)
    d = c.update(target=0.4, profile=_make_profile(curvature=0.001), active=True, v_ego=20.0, path_quality=1.0)
    assert d.preview_boost == 0.0

  def test_preview_boost_bounded(self):
    c = LateralTurnExitController(dt=0.05)
    c.update(target=0.0, profile=None, active=True, v_ego=20.0, path_quality=1.0)
    d = c.update(target=2.0, profile=_make_profile(curvature=0.005), active=True, v_ego=20.0, path_quality=1.0)
    assert abs(d.preview_boost) <= TURN_IN_PREVIEW_BOOST_CAP

  def test_no_profile_uses_target_rate_preview(self):
    c = LateralTurnExitController(dt=0.05)
    c.update(target=0.0, profile=None, active=True, v_ego=20.0, path_quality=1.0)
    d = c.update(target=0.4, profile=None, active=True, v_ego=20.0, path_quality=1.0)
    assert d.preview_boost != 0.0
    assert abs(d.preview_boost) <= TURN_IN_PREVIEW_BOOST_CAP


class TestSignsStable:

  def test_both_zero(self):
    assert _signs_stable(0.0, 0.0) is True

  def test_both_positive(self):
    assert _signs_stable(0.5, 0.3) is True

  def test_both_negative(self):
    assert _signs_stable(-0.5, -0.3) is True

  def test_positive_to_negative(self):
    assert _signs_stable(0.5, -0.3) is False

  def test_one_zero(self):
    assert _signs_stable(0.0, 0.5) is False
    assert _signs_stable(0.5, 0.0) is False
