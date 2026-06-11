import math

import pytest

from openpilot.selfdrive.controls.lib.lateral_demand import ProcessedLateralDemand
from openpilot.selfdrive.controls.lib.lateral_demand_profile import (
  LOW_QUALITY_PATH_THRESHOLD,
  STEADY_CURVE_MIN_LAT_ACCEL,
  STRAIGHT_STABLE_MAX_LAT_ACCEL,
  TURN_EXIT_COLLAPSE_PER_FRAME,
  TURN_IN_MIN_ABS_TARGET_RATE,
  LATERAL_MODE_TO_UINT8,
  LATERAL_UINT8_TO_MODE,
  LateralDemandProfile,
  LateralDemandProfileBuilder,
  LateralMode,
  _classify_lateral_mode,
  _safe_float,
  lateral_mode_to_uint8,
  uint8_to_lateral_mode,
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


class TestLateralDemandProfileBuilder:

  def test_target_lateral_accel_from_curvature_and_speed(self):
    builder = LateralDemandProfileBuilder(dt=0.05)
    demand = _make_demand(processed_curvature=0.001)
    profile = builder.update(demand, v_ego=20.0)
    assert math.isclose(profile.desired_lateral_accel, 0.4, rel_tol=1e-9)

  def test_desired_lateral_accel_zero_for_zero_curvature(self):
    builder = LateralDemandProfileBuilder(dt=0.05)
    profile = builder.update(_make_demand(processed_curvature=0.0), v_ego=20.0)
    assert profile.desired_lateral_accel == 0.0

  def test_non_finite_curvature_yields_zero_target(self):
    builder = LateralDemandProfileBuilder(dt=0.05)
    profile = builder.update(_make_demand(processed_curvature=float("nan")), v_ego=20.0)
    assert profile.desired_lateral_accel == 0.0

  def test_jerk_is_per_frame_difference_over_dt(self):
    builder = LateralDemandProfileBuilder(dt=0.05)
    builder.update(_make_demand(processed_curvature=0.0005), v_ego=20.0)
    profile = builder.update(_make_demand(processed_curvature=0.0010), v_ego=20.0)
    assert math.isclose(profile.desired_lateral_jerk, 4.0, rel_tol=1e-9)

  def test_jerk_zero_on_first_frame(self):
    builder = LateralDemandProfileBuilder(dt=0.05)
    profile = builder.update(_make_demand(processed_curvature=0.001), v_ego=20.0)
    assert profile.desired_lateral_jerk == 0.0

  def test_reset_clears_previous_target(self):
    builder = LateralDemandProfileBuilder(dt=0.05)
    builder.update(_make_demand(processed_curvature=0.001), v_ego=20.0)
    builder.reset()
    profile = builder.update(_make_demand(processed_curvature=0.002), v_ego=20.0)
    assert profile.desired_lateral_jerk == 0.0

  def test_passthrough_fields_preserved(self):
    builder = LateralDemandProfileBuilder(dt=0.05)
    demand = _make_demand(
      raw_curvature=0.0012,
      path_quality=0.83,
      path_reason="low_lane_confidence",
      lane_change_shaping_active=True,
      lane_change_blend=0.4,
    )
    profile = builder.update(demand, v_ego=20.0)
    assert profile.raw_curvature == pytest.approx(0.0012)
    assert profile.path_quality == pytest.approx(0.83)
    assert profile.path_reason == "low_lane_confidence"
    assert profile.lane_change_shaping_active is True
    assert profile.lane_change_blend == pytest.approx(0.4)
    assert profile.demand_source == "model_path"

  def test_preview_fields_zero_jerk_equals_current(self):
    builder = LateralDemandProfileBuilder(dt=0.05)
    profile = builder.update(_make_demand(processed_curvature=0.001), v_ego=20.0)
    assert profile.preview_lateral_accel_0_2s == pytest.approx(0.4, rel=1e-9)
    assert profile.preview_lateral_accel_0_5s == pytest.approx(0.4, rel=1e-9)
    assert profile.preview_lateral_accel_1_0s == pytest.approx(0.4, rel=1e-9)

  def test_preview_fields_taylor_approximation(self):
    builder = LateralDemandProfileBuilder(dt=0.05)
    builder.update(_make_demand(processed_curvature=0.0005), v_ego=20.0)
    profile = builder.update(_make_demand(processed_curvature=0.0010), v_ego=20.0)
    assert profile.desired_lateral_jerk == pytest.approx(4.0, rel=1e-9)
    assert profile.preview_lateral_accel_0_2s == pytest.approx(0.4 + 4.0 * 0.2, rel=1e-9)
    assert profile.preview_lateral_accel_0_5s == pytest.approx(0.4 + 4.0 * 0.5, rel=1e-9)
    assert profile.preview_lateral_accel_1_0s == pytest.approx(0.4 + 4.0 * 1.0, rel=1e-9)

  def test_straight_road_damping_active_default_false(self):
    builder = LateralDemandProfileBuilder(dt=0.05)
    profile = builder.update(_make_demand(processed_curvature=0.0), v_ego=25.0)
    assert profile.straight_road_damping_active is False

  def test_non_finite_v_ego_clamped_to_zero(self):
    builder = LateralDemandProfileBuilder(dt=0.05)
    profile = builder.update(_make_demand(processed_curvature=0.001), v_ego=float("nan"))
    assert profile.desired_lateral_accel == 0.0

  def test_returns_demand_profile_instance(self):
    builder = LateralDemandProfileBuilder(dt=0.05)
    profile = builder.update(_make_demand(), v_ego=20.0)
    assert isinstance(profile, LateralDemandProfile)

  def test_classifier_field_populated(self):
    builder = LateralDemandProfileBuilder(dt=0.05)
    profile = builder.update(_make_demand(processed_curvature=0.001), v_ego=20.0)
    assert profile.mode in {m.value for m in LateralMode}
    assert 0.0 <= profile.mode_confidence <= 1.0


class TestClassifyLateralMode:

  def _kwargs(self, **overrides):
    base = dict(
      target=0.0,
      previous_target=0.0,
      target_rate=0.0,
      path_quality=1.0,
      path_reason="ok",
      lane_change_shaping_active=False,
      lane_change_blend=0.0,
      curvature_limited=False,
      saturated=False,
      steer_limited_by_safety=False,
      steering_pressed=False,
      v_ego=20.0,
    )
    base.update(overrides)
    return base

  def test_straight_stable_default(self):
    mode, conf = _classify_lateral_mode(**self._kwargs(target=0.01))
    assert mode == LateralMode.STRAIGHT_STABLE.value
    assert conf > 0.0

  def test_straight_stable_max_lat_accel_boundary(self):
    mode, _ = _classify_lateral_mode(**self._kwargs(target=STRAIGHT_STABLE_MAX_LAT_ACCEL - 0.01))
    assert mode == LateralMode.STRAIGHT_STABLE.value

  def test_turn_in_when_building_target(self):
    mode, conf = _classify_lateral_mode(**self._kwargs(
      target=0.5,
      previous_target=0.0,
      target_rate=TURN_IN_MIN_ABS_TARGET_RATE + 1.0,
    ))
    assert mode == LateralMode.TURN_IN.value
    assert conf >= 0.9

  def test_steady_curve_above_min_lat_accel(self):
    mode, _ = _classify_lateral_mode(**self._kwargs(
      target=STEADY_CURVE_MIN_LAT_ACCEL + 0.05,
      previous_target=STEADY_CURVE_MIN_LAT_ACCEL + 0.05,
      target_rate=0.0,
    ))
    assert mode == LateralMode.STEADY_CURVE.value

  def test_turn_exit_recenter_when_target_collapses_stable_sign(self):
    mode, _ = _classify_lateral_mode(**self._kwargs(
      target=0.3,
      previous_target=0.5,
      target_rate=-4.0,
    ))
    assert mode == LateralMode.TURN_EXIT_RECENTER.value

  def test_turn_exit_recenter_requires_stable_sign(self):
    mode, _ = _classify_lateral_mode(**self._kwargs(
      target=0.3,
      previous_target=-0.5,
      target_rate=-16.0,
    ))
    assert mode == LateralMode.SAFETY_LIMITED.value or mode != LateralMode.TURN_EXIT_RECENTER.value

  def test_turn_exit_recenter_below_collapse_threshold(self):
    mode, _ = _classify_lateral_mode(**self._kwargs(
      target=0.499,
      previous_target=0.5,
      target_rate=-(TURN_EXIT_COLLAPSE_PER_FRAME - 1e-4) * 20,
    ))
    assert mode != LateralMode.TURN_EXIT_RECENTER.value

  def test_lane_change_priority(self):
    mode, _ = _classify_lateral_mode(**self._kwargs(
      lane_change_shaping_active=True,
      target=0.5,
      target_rate=1.0,
    ))
    assert mode == LateralMode.LANE_CHANGE.value

  def test_lane_change_via_blend(self):
    mode, _ = _classify_lateral_mode(**self._kwargs(
      lane_change_blend=0.5,
    ))
    assert mode == LateralMode.LANE_CHANGE.value

  def test_low_quality_path_priority(self):
    mode, _ = _classify_lateral_mode(**self._kwargs(
      path_quality=LOW_QUALITY_PATH_THRESHOLD - 0.1,
      target=0.5,
      target_rate=1.0,
    ))
    assert mode == LateralMode.LOW_QUALITY_PATH.value

  def test_low_quality_path_reason_priority(self):
    mode, _ = _classify_lateral_mode(**self._kwargs(
      path_reason="high_path_std",
      target=0.5,
      target_rate=1.0,
    ))
    assert mode == LateralMode.LOW_QUALITY_PATH.value

  def test_safety_limited_curvature_limited(self):
    mode, _ = _classify_lateral_mode(**self._kwargs(
      curvature_limited=True,
      target=0.5,
      target_rate=1.0,
    ))
    assert mode == LateralMode.SAFETY_LIMITED.value

  def test_safety_limited_saturated(self):
    mode, _ = _classify_lateral_mode(**self._kwargs(
      saturated=True,
      target=0.5,
      target_rate=1.0,
    ))
    assert mode == LateralMode.SAFETY_LIMITED.value

  def test_safety_limited_steer_limited(self):
    mode, _ = _classify_lateral_mode(**self._kwargs(
      steer_limited_by_safety=True,
      target=0.5,
      target_rate=1.0,
    ))
    assert mode == LateralMode.SAFETY_LIMITED.value

  def test_driver_override_highest_priority(self):
    mode, _ = _classify_lateral_mode(**self._kwargs(
      steering_pressed=True,
      curvature_limited=True,
      lane_change_shaping_active=True,
      path_quality=0.1,
      target=0.5,
      target_rate=1.0,
    ))
    assert mode == LateralMode.DRIVER_OVERRIDE.value

  def test_safety_limited_beats_lane_change(self):
    mode, _ = _classify_lateral_mode(**self._kwargs(
      curvature_limited=True,
      lane_change_shaping_active=True,
    ))
    assert mode == LateralMode.SAFETY_LIMITED.value

  def test_lane_change_beats_low_quality_path(self):
    mode, _ = _classify_lateral_mode(**self._kwargs(
      lane_change_shaping_active=True,
      path_quality=0.1,
    ))
    assert mode == LateralMode.LANE_CHANGE.value

  def test_low_quality_path_beats_turn_in(self):
    mode, _ = _classify_lateral_mode(**self._kwargs(
      path_quality=0.1,
      target=0.5,
      target_rate=1.0,
    ))
    assert mode == LateralMode.LOW_QUALITY_PATH.value

  def test_straight_bias_correction_not_yet_implemented(self):
    mode, _ = _classify_lateral_mode(**self._kwargs(target=0.0))
    assert mode == LateralMode.STRAIGHT_STABLE.value


class TestSafeFloat:

  def test_passes_finite_value(self):
    assert _safe_float(1.5, 0.0) == 1.5

  def test_returns_default_for_none(self):
    assert _safe_float(None, 0.5) == 0.5

  def test_returns_default_for_nan(self):
    assert _safe_float(float("nan"), 0.5) == 0.5

  def test_returns_default_for_inf(self):
    assert _safe_float(float("inf"), 0.5) == 0.5

  def test_returns_default_for_string(self):
    assert _safe_float("bad", 0.5) == 0.5


class TestLateralModeUint8Mapping:

  def test_all_modes_have_unique_uint8(self):
    values = list(LATERAL_MODE_TO_UINT8.values())
    assert len(values) == len(LATERAL_UINT8_TO_MODE)
    assert len(set(values)) == len(values)
    assert all(v > 0 for v in values)
    assert all(v < 256 for v in values)

  def test_lateral_mode_to_uint8_known_modes(self):
    assert lateral_mode_to_uint8(LateralMode.STRAIGHT_STABLE.value) == 1
    assert lateral_mode_to_uint8(LateralMode.STRAIGHT_BIAS_CORRECTION.value) == 2
    assert lateral_mode_to_uint8(LateralMode.TURN_IN.value) == 3
    assert lateral_mode_to_uint8(LateralMode.STEADY_CURVE.value) == 4
    assert lateral_mode_to_uint8(LateralMode.TURN_EXIT_RECENTER.value) == 5
    assert lateral_mode_to_uint8(LateralMode.LANE_CHANGE.value) == 6
    assert lateral_mode_to_uint8(LateralMode.LOW_QUALITY_PATH.value) == 7
    assert lateral_mode_to_uint8(LateralMode.SAFETY_LIMITED.value) == 8
    assert lateral_mode_to_uint8(LateralMode.DRIVER_OVERRIDE.value) == 9

  def test_lateral_mode_to_uint8_unknown_returns_zero(self):
    assert lateral_mode_to_uint8("not_a_mode") == 0
    assert lateral_mode_to_uint8("") == 0

  def test_uint8_to_lateral_mode_known_values(self):
    assert uint8_to_lateral_mode(1) == LateralMode.STRAIGHT_STABLE.value
    assert uint8_to_lateral_mode(5) == LateralMode.TURN_EXIT_RECENTER.value
    assert uint8_to_lateral_mode(9) == LateralMode.DRIVER_OVERRIDE.value

  def test_uint8_to_lateral_mode_unknown_falls_back_to_straight_stable(self):
    assert uint8_to_lateral_mode(0) == LateralMode.STRAIGHT_STABLE.value
    assert uint8_to_lateral_mode(99) == LateralMode.STRAIGHT_STABLE.value
    assert uint8_to_lateral_mode(255) == LateralMode.STRAIGHT_STABLE.value

  def test_round_trip_for_all_modes(self):
    for mode in LateralMode:
      uint8 = lateral_mode_to_uint8(mode.value)
      assert uint8_to_lateral_mode(uint8) == mode.value
