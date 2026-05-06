import pytest

from openpilot.sunnypilot.selfdrive.controls.lib.torque_conservative_output_shaper import (
  ConservativeOutputShaperInputs,
  ConservativeOutputShapingReason,
  TorqueConservativeOutputShaper,
)


def make_inputs(**overrides):
  values = {
    "active": True,
    "v_ego": 15.0,
    "steering_pressed": False,
    "steer_limited_by_safety": False,
    "release_active": False,
    "max_output": 1.0,
    "unshaped_output": 0.5,
    "desired_lateral_accel": 0.8,
    "actual_lateral_accel": 0.4,
    "desired_lateral_jerk": 0.2,
    "actual_lateral_jerk": 0.05,
    "lookahead_lateral_jerk": 0.2,
    "same_sign_unwind_release": False,
    "steering_rate_deg": 0.0,
    "steer_limit_same_direction": True,
    "steer_limit_unwind": False,
  }
  values.update(overrides)
  return ConservativeOutputShaperInputs(**values)


def assert_cap_only(inputs):
  result = TorqueConservativeOutputShaper().update(inputs)

  assert abs(result.output_torque) <= abs(inputs.unshaped_output)
  assert result.output_torque * inputs.unshaped_output >= 0.0
  return result


def test_clean_under_response_is_not_shaped():
  result = assert_cap_only(make_inputs(desired_lateral_accel=0.8, actual_lateral_accel=0.4))

  assert not result.active
  assert result.reason == 0
  assert result.output_cap == 1.0
  assert result.output_torque == result.unshaped_output


def test_driver_override_caps_output():
  result = assert_cap_only(make_inputs(steering_pressed=True))

  assert result.active
  assert result.reason & ConservativeOutputShapingReason.STEERING_PRESSED
  assert result.output_cap == 0.8
  assert result.output_torque == 0.4


def test_release_caps_output():
  result = assert_cap_only(make_inputs(release_active=True))

  assert result.active
  assert result.reason & ConservativeOutputShapingReason.RELEASE
  assert result.output_cap == 0.8


def test_sign_conflict_caps_output():
  result = assert_cap_only(make_inputs(desired_lateral_accel=0.4, actual_lateral_accel=-0.2))

  assert result.active
  assert result.reason & ConservativeOutputShapingReason.SIGN_CONFLICT
  assert result.output_cap == 0.8


def test_sign_conflict_caps_negative_output():
  result = assert_cap_only(make_inputs(unshaped_output=-0.5, desired_lateral_accel=0.4, actual_lateral_accel=-0.2))

  assert result.active
  assert result.reason & ConservativeOutputShapingReason.SIGN_CONFLICT
  assert result.output_torque == -0.4


def test_over_response_caps_output():
  result = assert_cap_only(make_inputs(desired_lateral_accel=0.4, actual_lateral_accel=0.8))

  assert result.active
  assert result.reason & ConservativeOutputShapingReason.OVER_RESPONSE
  assert 0.65 < result.output_cap < 0.85
  assert result.output_torque == result.unshaped_output * result.output_cap


def test_mild_over_response_keeps_existing_cap():
  result = assert_cap_only(make_inputs(desired_lateral_accel=0.8, actual_lateral_accel=1.02))

  assert result.active
  assert result.reason & ConservativeOutputShapingReason.OVER_RESPONSE
  assert result.output_cap == 0.85


def test_severe_over_response_caps_output_more_strictly():
  result = assert_cap_only(make_inputs(desired_lateral_accel=2.5, actual_lateral_accel=3.3, unshaped_output=1.0))

  assert result.active
  assert result.reason & ConservativeOutputShapingReason.OVER_RESPONSE
  assert result.reason & ConservativeOutputShapingReason.NEAR_ISO_ACCEL
  assert result.output_cap == 0.45
  assert result.output_torque == 0.45


def test_over_response_does_not_cap_corrective_output():
  result = assert_cap_only(make_inputs(unshaped_output=-0.5, desired_lateral_accel=0.4, actual_lateral_accel=0.8))

  assert not result.active
  assert result.output_torque == result.unshaped_output


def test_over_response_cap_does_not_rate_limit_next_corrective_output():
  shaper = TorqueConservativeOutputShaper(dt=0.01)
  shaper.update(make_inputs(unshaped_output=1.0, desired_lateral_accel=0.4, actual_lateral_accel=1.2))

  result = shaper.update(make_inputs(unshaped_output=-1.0, desired_lateral_accel=0.4, actual_lateral_accel=1.2))

  assert not result.active
  assert result.output_torque == result.unshaped_output


def test_over_response_cap_allows_next_corrective_output_after_over_response_clears():
  shaper = TorqueConservativeOutputShaper(dt=0.01)
  shaper.update(make_inputs(unshaped_output=1.0, desired_lateral_accel=0.4, actual_lateral_accel=1.2))

  result = shaper.update(make_inputs(unshaped_output=-1.0, desired_lateral_accel=0.4, actual_lateral_accel=0.5))

  assert not result.active
  assert result.output_torque == result.unshaped_output


def test_recent_over_response_does_not_bypass_rate_limit_on_near_zero_actual_accel():
  shaper = TorqueConservativeOutputShaper(dt=0.01)
  shaper.update(make_inputs(unshaped_output=1.0, desired_lateral_accel=0.4, actual_lateral_accel=1.2))

  result = shaper.update(make_inputs(unshaped_output=-1.0, desired_lateral_accel=0.4, actual_lateral_accel=0.01))

  assert result.active
  assert result.reason & ConservativeOutputShapingReason.OUTPUT_RATE_LIMITED
  assert result.output_torque == -0.04


def test_recent_over_response_bypass_expires_during_zero_output_frames():
  shaper = TorqueConservativeOutputShaper(dt=0.01)
  shaper.update(make_inputs(unshaped_output=1.0, desired_lateral_accel=0.4, actual_lateral_accel=1.2))
  for _ in range(50):
    shaper.update(make_inputs(unshaped_output=0.0, desired_lateral_accel=0.4, actual_lateral_accel=0.5))
  shaper.update(make_inputs(unshaped_output=1.0, steering_pressed=True, desired_lateral_accel=0.4, actual_lateral_accel=0.5))

  result = shaper.update(make_inputs(unshaped_output=-1.0, desired_lateral_accel=0.4, actual_lateral_accel=0.5))

  assert result.active
  assert result.reason & ConservativeOutputShapingReason.OUTPUT_RATE_LIMITED
  assert result.output_torque == -0.04


def test_driver_override_keeps_release_cap_during_over_response():
  result = assert_cap_only(make_inputs(steering_pressed=True, desired_lateral_accel=2.5, actual_lateral_accel=3.3, unshaped_output=1.0))

  assert result.active
  assert result.reason & ConservativeOutputShapingReason.STEERING_PRESSED
  assert result.reason & ConservativeOutputShapingReason.OVER_RESPONSE
  assert result.output_cap == 0.8
  assert result.output_torque == 0.8


def test_near_iso_accel_caps_output():
  result = assert_cap_only(make_inputs(desired_lateral_accel=2.75, actual_lateral_accel=2.8))

  assert result.active
  assert result.reason & ConservativeOutputShapingReason.NEAR_ISO_ACCEL
  assert result.output_cap == 0.85


def test_over_iso_accel_caps_output_more_strictly():
  result = assert_cap_only(make_inputs(desired_lateral_accel=2.9, actual_lateral_accel=3.1))

  assert result.active
  assert result.reason & ConservativeOutputShapingReason.NEAR_ISO_ACCEL
  assert result.output_cap == 0.8


def test_near_iso_accel_does_not_cap_corrective_output():
  result = assert_cap_only(make_inputs(unshaped_output=-0.5, desired_lateral_accel=2.75, actual_lateral_accel=2.8))

  assert not result.active
  assert result.output_torque == result.unshaped_output


def test_over_iso_accel_does_not_cap_corrective_output():
  result = assert_cap_only(make_inputs(unshaped_output=-0.5, desired_lateral_accel=2.9, actual_lateral_accel=3.1))

  assert not result.active
  assert result.output_torque == result.unshaped_output


def test_negative_output_keeps_sign_when_capped():
  result = assert_cap_only(make_inputs(unshaped_output=-0.5, desired_lateral_accel=-0.4, actual_lateral_accel=-0.8))

  assert result.active
  assert result.reason & ConservativeOutputShapingReason.OVER_RESPONSE
  assert result.output_cap < 0.85
  assert result.output_torque == result.unshaped_output * result.output_cap


def test_bump_response_caps_output():
  result = assert_cap_only(make_inputs(actual_lateral_jerk=3.0, lookahead_lateral_jerk=0.0, desired_lateral_jerk=0.0))

  assert result.active
  assert result.reason & ConservativeOutputShapingReason.BUMP
  assert result.output_cap == 0.9


def test_low_speed_steer_limited_high_output_caps_output():
  result = assert_cap_only(make_inputs(v_ego=5.0, steer_limited_by_safety=True, unshaped_output=0.8,
                                       desired_lateral_accel=0.8, actual_lateral_accel=0.72))

  assert result.active
  assert result.reason & ConservativeOutputShapingReason.LOW_SPEED_STEER_LIMITED
  assert result.output_cap == 0.92


def test_low_speed_steer_limited_does_not_cap_clear_under_response_catchup():
  result = assert_cap_only(make_inputs(v_ego=5.0, steer_limited_by_safety=True, unshaped_output=0.8,
                                       desired_lateral_accel=0.8, actual_lateral_accel=0.4))

  assert not result.active
  assert not result.reason & ConservativeOutputShapingReason.LOW_SPEED_STEER_LIMITED
  assert result.output_cap == 1.0
  assert result.output_torque == result.unshaped_output


def test_low_speed_steer_limited_does_not_rate_limit_clear_under_response_after_soft_cap():
  shaper = TorqueConservativeOutputShaper(dt=0.01)
  shaper.update(make_inputs(v_ego=5.0, steer_limited_by_safety=True, unshaped_output=0.8,
                            desired_lateral_accel=0.8, actual_lateral_accel=0.72))

  result = shaper.update(make_inputs(v_ego=5.0, steer_limited_by_safety=True, unshaped_output=1.0,
                                     desired_lateral_accel=0.8, actual_lateral_accel=0.4))

  assert not result.active
  assert not result.reason & ConservativeOutputShapingReason.OUTPUT_RATE_LIMITED
  assert result.output_torque == result.unshaped_output


def test_steering_rate_comfort_caps_reinforcing_output():
  result = assert_cap_only(make_inputs(steering_rate_deg=40.0, unshaped_output=0.8,
                                       desired_lateral_accel=0.8, actual_lateral_accel=0.72))

  assert result.active
  assert result.reason & ConservativeOutputShapingReason.STEERING_RATE_COMFORT
  assert 0.8 <= result.output_cap < 1.0
  assert 0.0 < result.output_torque < result.unshaped_output


def test_steering_rate_comfort_does_not_cap_clear_under_response_catchup():
  result = assert_cap_only(make_inputs(steering_rate_deg=40.0, unshaped_output=0.8,
                                       desired_lateral_accel=0.8, actual_lateral_accel=0.4))

  assert not result.active
  assert not result.reason & ConservativeOutputShapingReason.STEERING_RATE_COMFORT
  assert result.output_cap == 1.0
  assert result.output_torque == result.unshaped_output


def test_steering_rate_comfort_does_not_rate_limit_clear_under_response_after_soft_cap():
  shaper = TorqueConservativeOutputShaper(dt=0.01)
  shaper.update(make_inputs(steering_rate_deg=40.0, unshaped_output=0.8,
                            desired_lateral_accel=0.8, actual_lateral_accel=0.72))

  result = shaper.update(make_inputs(steering_rate_deg=40.0, unshaped_output=1.0,
                                     desired_lateral_accel=0.8, actual_lateral_accel=0.4))

  assert not result.active
  assert not result.reason & ConservativeOutputShapingReason.OUTPUT_RATE_LIMITED
  assert result.output_torque == result.unshaped_output


def test_steering_rate_comfort_does_not_cap_corrective_output():
  result = assert_cap_only(make_inputs(steering_rate_deg=40.0, unshaped_output=-0.8))

  assert not result.active
  assert not result.reason & ConservativeOutputShapingReason.STEERING_RATE_COMFORT
  assert result.output_torque == result.unshaped_output


def test_steering_rate_comfort_ignores_low_steering_rate():
  result = assert_cap_only(make_inputs(steering_rate_deg=5.0, unshaped_output=0.8))

  assert not result.active
  assert not result.reason & ConservativeOutputShapingReason.STEERING_RATE_COMFORT
  assert result.output_torque == result.unshaped_output


def test_steering_rate_comfort_slews_reinforcing_output_growth():
  shaper = TorqueConservativeOutputShaper(dt=0.01)
  first = shaper.update(make_inputs(steering_rate_deg=40.0, unshaped_output=0.2,
                                    desired_lateral_accel=0.8, actual_lateral_accel=0.72))

  result = shaper.update(make_inputs(steering_rate_deg=40.0, unshaped_output=0.8,
                                     desired_lateral_accel=0.8, actual_lateral_accel=0.72))

  assert result.active
  assert result.reason & ConservativeOutputShapingReason.STEERING_RATE_COMFORT
  assert result.reason & ConservativeOutputShapingReason.OUTPUT_RATE_LIMITED
  assert 0.0 < result.output_torque - first.output_torque <= 0.0075 + 1e-6


def test_steering_rate_comfort_slews_first_reinforcing_growth_after_clean_tracking():
  shaper = TorqueConservativeOutputShaper(dt=0.01)
  first = shaper.update(make_inputs(steering_rate_deg=0.0, unshaped_output=0.2))

  result = shaper.update(make_inputs(steering_rate_deg=40.0, unshaped_output=0.8,
                                     desired_lateral_accel=0.8, actual_lateral_accel=0.72))

  assert result.active
  assert result.reason & ConservativeOutputShapingReason.STEERING_RATE_COMFORT
  assert result.reason & ConservativeOutputShapingReason.OUTPUT_RATE_LIMITED
  assert 0.0 < result.output_torque - first.output_torque <= 0.0075 + 1e-6


def test_steering_rate_comfort_does_not_rate_limit_next_opposing_output():
  shaper = TorqueConservativeOutputShaper(dt=0.01)
  shaper.update(make_inputs(steering_rate_deg=40.0, unshaped_output=0.8))

  result = shaper.update(make_inputs(steering_rate_deg=40.0, unshaped_output=-0.8))

  assert not result.active
  assert not result.reason & ConservativeOutputShapingReason.OUTPUT_RATE_LIMITED
  assert result.output_torque == result.unshaped_output


def test_actuator_lag_comfort_caps_low_speed_reinforcing_output_more_than_steering_rate_comfort():
  result = assert_cap_only(make_inputs(v_ego=5.0, steer_limited_by_safety=True, steering_rate_deg=40.0, unshaped_output=0.8,
                                       desired_lateral_accel=0.8, actual_lateral_accel=0.72))

  assert result.active
  assert result.reason & ConservativeOutputShapingReason.STEERING_RATE_COMFORT
  assert result.reason & ConservativeOutputShapingReason.ACTUATOR_LAG_COMFORT
  assert result.output_cap <= 0.55
  assert result.output_torque <= 0.44 + 1e-6


def test_actuator_lag_comfort_uses_signed_same_direction_limit():
  result = assert_cap_only(make_inputs(v_ego=5.0, steer_limited_by_safety=True, steer_limit_same_direction=True,
                                       steer_limit_unwind=False, steering_rate_deg=40.0, unshaped_output=0.8,
                                       desired_lateral_accel=0.8, actual_lateral_accel=0.72))

  assert result.active
  assert result.reason & ConservativeOutputShapingReason.ACTUATOR_LAG_COMFORT
  assert result.output_cap <= 0.55


def test_actuator_lag_comfort_allows_signed_unwind():
  result = assert_cap_only(make_inputs(v_ego=5.0, steer_limited_by_safety=True, steer_limit_same_direction=False,
                                       steer_limit_unwind=True, steering_rate_deg=40.0, unshaped_output=0.8,
                                       desired_lateral_accel=0.8, actual_lateral_accel=0.72))

  assert not result.reason & ConservativeOutputShapingReason.ACTUATOR_LAG_COMFORT
  assert result.output_cap > 0.55


def test_actuator_lag_comfort_uses_moderate_cap_at_mid_speed():
  result = assert_cap_only(make_inputs(v_ego=10.0, steer_limited_by_safety=True, steering_rate_deg=40.0, unshaped_output=0.8,
                                       desired_lateral_accel=0.8, actual_lateral_accel=0.72))

  assert result.active
  assert result.reason & ConservativeOutputShapingReason.ACTUATOR_LAG_COMFORT
  assert 0.55 < result.output_cap <= 0.70


def test_actuator_lag_comfort_stays_mild_at_high_speed():
  result = assert_cap_only(make_inputs(v_ego=20.0, steer_limited_by_safety=True, steering_rate_deg=40.0, unshaped_output=0.8,
                                       desired_lateral_accel=0.8, actual_lateral_accel=0.72))

  assert result.active
  assert result.reason & ConservativeOutputShapingReason.ACTUATOR_LAG_COMFORT
  assert 0.75 <= result.output_cap <= 0.85


def test_actuator_lag_comfort_does_not_cap_opposing_output():
  result = assert_cap_only(make_inputs(v_ego=5.0, steer_limited_by_safety=True, steering_rate_deg=40.0, unshaped_output=-0.8))

  assert not result.active
  assert not result.reason & ConservativeOutputShapingReason.ACTUATOR_LAG_COMFORT
  assert result.output_torque == result.unshaped_output


def test_actuator_lag_comfort_does_not_cap_clear_under_response_catchup():
  result = assert_cap_only(make_inputs(v_ego=5.0, steer_limited_by_safety=True, steering_rate_deg=40.0, unshaped_output=0.8,
                                       desired_lateral_accel=0.8, actual_lateral_accel=0.4))

  assert not result.active
  assert not result.reason & ConservativeOutputShapingReason.ACTUATOR_LAG_COMFORT
  assert result.output_torque == result.unshaped_output


def test_actuator_lag_comfort_ignores_low_steering_rate():
  result = assert_cap_only(make_inputs(v_ego=5.0, steer_limited_by_safety=True, steering_rate_deg=5.0, unshaped_output=0.8))

  assert not result.reason & ConservativeOutputShapingReason.ACTUATOR_LAG_COMFORT


def test_actuator_lag_comfort_slews_reinforcing_growth_more_than_steering_rate_comfort():
  shaper = TorqueConservativeOutputShaper(dt=0.01)
  first = shaper.update(make_inputs(v_ego=5.0, steer_limited_by_safety=True, steering_rate_deg=40.0, unshaped_output=0.2,
                                    desired_lateral_accel=0.8, actual_lateral_accel=0.72))

  result = shaper.update(make_inputs(v_ego=5.0, steer_limited_by_safety=True, steering_rate_deg=40.0, unshaped_output=0.8,
                                     desired_lateral_accel=0.8, actual_lateral_accel=0.72))

  assert result.active
  assert result.reason & ConservativeOutputShapingReason.ACTUATOR_LAG_COMFORT
  assert result.reason & ConservativeOutputShapingReason.OUTPUT_RATE_LIMITED
  assert 0.0 < result.output_torque - first.output_torque <= 0.0035 + 1e-6


def test_actuator_lag_comfort_does_not_rate_limit_next_opposing_output():
  shaper = TorqueConservativeOutputShaper(dt=0.01)
  shaper.update(make_inputs(v_ego=5.0, steer_limited_by_safety=True, steering_rate_deg=40.0, unshaped_output=0.8))

  result = shaper.update(make_inputs(v_ego=5.0, steer_limited_by_safety=True, steering_rate_deg=40.0, unshaped_output=-0.8))

  assert not result.active
  assert not result.reason & ConservativeOutputShapingReason.OUTPUT_RATE_LIMITED
  assert result.output_torque == result.unshaped_output


def test_actuator_lag_comfort_does_not_rate_limit_clear_under_response_after_soft_cap():
  shaper = TorqueConservativeOutputShaper(dt=0.01)
  shaper.update(make_inputs(v_ego=5.0, steer_limited_by_safety=True, steering_rate_deg=40.0, unshaped_output=0.8,
                            desired_lateral_accel=0.8, actual_lateral_accel=0.72))

  result = shaper.update(make_inputs(v_ego=5.0, steer_limited_by_safety=True, steering_rate_deg=40.0, unshaped_output=1.0,
                                     desired_lateral_accel=0.8, actual_lateral_accel=0.4))

  assert not result.active
  assert not result.reason & ConservativeOutputShapingReason.OUTPUT_RATE_LIMITED
  assert result.output_torque == result.unshaped_output


def test_stale_actuator_reversal_caps_low_speed_reversing_output():
  result = assert_cap_only(make_inputs(v_ego=5.0, steer_limited_by_safety=True, steer_limit_same_direction=True,
                                       steering_rate_deg=40.0, unshaped_output=0.8,
                                       desired_lateral_accel=0.8, actual_lateral_accel=0.72,
                                       steer_limit_requested_output=0.8, steer_limit_applied_output=-0.25))

  assert result.active
  assert result.reason & ConservativeOutputShapingReason.STALE_ACTUATOR_REVERSAL
  assert result.output_cap == 0.35
  assert result.output_torque == pytest.approx(0.28)


def test_stale_actuator_reversal_caps_negative_low_speed_reversing_output():
  result = assert_cap_only(make_inputs(v_ego=5.0, steer_limited_by_safety=True, steer_limit_same_direction=True,
                                       steering_rate_deg=-40.0, unshaped_output=-0.8,
                                       desired_lateral_accel=-0.8, actual_lateral_accel=-0.72,
                                       steer_limit_requested_output=-0.8, steer_limit_applied_output=0.25))

  assert result.active
  assert result.reason & ConservativeOutputShapingReason.STALE_ACTUATOR_REVERSAL
  assert result.output_cap == 0.35
  assert result.output_torque == pytest.approx(-0.28)


def test_stale_actuator_reversal_clears_when_applied_output_near_zero():
  result = assert_cap_only(make_inputs(v_ego=5.0, steer_limited_by_safety=True, steer_limit_same_direction=True,
                                       steering_rate_deg=40.0, unshaped_output=0.8,
                                       desired_lateral_accel=0.8, actual_lateral_accel=0.72,
                                       steer_limit_requested_output=0.8, steer_limit_applied_output=-0.02))

  assert result.active
  assert not result.reason & ConservativeOutputShapingReason.STALE_ACTUATOR_REVERSAL
  assert result.reason & ConservativeOutputShapingReason.ACTUATOR_LAG_COMFORT
  assert result.output_cap > 0.35


def test_stale_actuator_reversal_caps_even_when_tracking_under_responds():
  result = assert_cap_only(make_inputs(v_ego=5.0, steer_limited_by_safety=True, steer_limit_same_direction=True,
                                       steering_rate_deg=40.0, unshaped_output=0.8,
                                       desired_lateral_accel=0.8, actual_lateral_accel=0.4,
                                       steer_limit_requested_output=0.8, steer_limit_applied_output=-0.25))

  assert result.active
  assert result.reason & ConservativeOutputShapingReason.STALE_ACTUATOR_REVERSAL
  assert result.output_cap == 0.35


def test_stale_actuator_reversal_keeps_high_speed_actuator_lag_cap():
  result = assert_cap_only(make_inputs(v_ego=20.0, steer_limited_by_safety=True, steer_limit_same_direction=True,
                                       steering_rate_deg=40.0, unshaped_output=0.8,
                                       desired_lateral_accel=0.8, actual_lateral_accel=0.72,
                                       steer_limit_requested_output=0.8, steer_limit_applied_output=-0.25))

  assert result.active
  assert not result.reason & ConservativeOutputShapingReason.STALE_ACTUATOR_REVERSAL
  assert result.reason & ConservativeOutputShapingReason.ACTUATOR_LAG_COMFORT
  assert 0.75 <= result.output_cap <= 0.85


def test_stale_actuator_reversal_slews_growth_more_slowly():
  shaper = TorqueConservativeOutputShaper(dt=0.01)
  first = shaper.update(make_inputs(v_ego=5.0, steer_limited_by_safety=True, steer_limit_same_direction=True,
                                    steering_rate_deg=40.0, unshaped_output=0.1,
                                    desired_lateral_accel=0.8, actual_lateral_accel=0.72,
                                    steer_limit_requested_output=0.1, steer_limit_applied_output=-0.25))

  result = shaper.update(make_inputs(v_ego=5.0, steer_limited_by_safety=True, steer_limit_same_direction=True,
                                     steering_rate_deg=40.0, unshaped_output=0.8,
                                     desired_lateral_accel=0.8, actual_lateral_accel=0.72,
                                     steer_limit_requested_output=0.8, steer_limit_applied_output=-0.25))

  assert result.active
  assert result.reason & ConservativeOutputShapingReason.STALE_ACTUATOR_REVERSAL
  assert result.reason & ConservativeOutputShapingReason.OUTPUT_RATE_LIMITED
  assert 0.0 < result.output_torque - first.output_torque <= 0.002 + 1e-6


def test_stale_actuator_reversal_slews_without_steering_rate_comfort():
  shaper = TorqueConservativeOutputShaper(dt=0.01)
  first = shaper.update(make_inputs(v_ego=5.0, steer_limited_by_safety=True, steer_limit_same_direction=True,
                                    steering_rate_deg=0.0, unshaped_output=0.1,
                                    desired_lateral_accel=0.8, actual_lateral_accel=0.72,
                                    steer_limit_requested_output=0.1, steer_limit_applied_output=-0.25))

  result = shaper.update(make_inputs(v_ego=5.0, steer_limited_by_safety=True, steer_limit_same_direction=True,
                                     steering_rate_deg=0.0, unshaped_output=0.8,
                                     desired_lateral_accel=0.8, actual_lateral_accel=0.72,
                                     steer_limit_requested_output=0.8, steer_limit_applied_output=-0.25))

  assert result.active
  assert result.reason & ConservativeOutputShapingReason.STALE_ACTUATOR_REVERSAL
  assert result.reason & ConservativeOutputShapingReason.OUTPUT_RATE_LIMITED
  assert 0.0 < result.output_torque - first.output_torque <= 0.002 + 1e-6


def test_stale_actuator_reversal_slews_under_response_catchup():
  shaper = TorqueConservativeOutputShaper(dt=0.01)
  first = shaper.update(make_inputs(v_ego=5.0, steer_limited_by_safety=True, steer_limit_same_direction=True,
                                    steering_rate_deg=40.0, unshaped_output=0.1,
                                    desired_lateral_accel=0.8, actual_lateral_accel=0.4,
                                    steer_limit_requested_output=0.1, steer_limit_applied_output=-0.25))

  result = shaper.update(make_inputs(v_ego=5.0, steer_limited_by_safety=True, steer_limit_same_direction=True,
                                     steering_rate_deg=40.0, unshaped_output=0.8,
                                     desired_lateral_accel=0.8, actual_lateral_accel=0.4,
                                     steer_limit_requested_output=0.8, steer_limit_applied_output=-0.25))

  assert result.active
  assert result.reason & ConservativeOutputShapingReason.STALE_ACTUATOR_REVERSAL
  assert result.reason & ConservativeOutputShapingReason.OUTPUT_RATE_LIMITED
  assert 0.0 < result.output_torque - first.output_torque <= 0.002 + 1e-6


def test_same_direction_safety_limit_follows_applied_output():
  result = assert_cap_only(make_inputs(steer_limited_by_safety=True, steer_limit_same_direction=True,
                                       unshaped_output=1.0, desired_lateral_accel=0.8, actual_lateral_accel=0.72,
                                       steer_limit_requested_output=1.0, steer_limit_applied_output=0.2))

  assert result.active
  assert result.reason & ConservativeOutputShapingReason.SAFETY_LIMITED_RAMP
  assert result.output_cap == pytest.approx(0.35)
  assert result.output_torque == pytest.approx(0.35)


def test_same_direction_safety_limit_caps_under_response_catchup():
  result = assert_cap_only(make_inputs(steer_limited_by_safety=True, steer_limit_same_direction=True,
                                       unshaped_output=1.0, desired_lateral_accel=0.8, actual_lateral_accel=0.4,
                                       steer_limit_requested_output=1.0, steer_limit_applied_output=0.2))

  assert result.active
  assert result.reason & ConservativeOutputShapingReason.SAFETY_LIMITED_RAMP
  assert result.output_cap == pytest.approx(0.35)
  assert result.output_torque == pytest.approx(0.35)


def test_same_direction_safety_limit_does_not_cap_unwind():
  result = assert_cap_only(make_inputs(steer_limited_by_safety=True, steer_limit_same_direction=False, steer_limit_unwind=True,
                                       unshaped_output=1.0, desired_lateral_accel=0.8, actual_lateral_accel=0.72,
                                       steer_limit_requested_output=1.0, steer_limit_applied_output=0.2))

  assert not result.reason & ConservativeOutputShapingReason.SAFETY_LIMITED_RAMP
  assert result.output_torque == result.unshaped_output


def test_same_direction_safety_limit_does_not_cap_corrective_output():
  result = assert_cap_only(make_inputs(steer_limited_by_safety=True, steer_limit_same_direction=True,
                                       unshaped_output=-1.0, desired_lateral_accel=0.4, actual_lateral_accel=0.8,
                                       steer_limit_requested_output=-1.0, steer_limit_applied_output=-0.2))

  assert not result.reason & ConservativeOutputShapingReason.SAFETY_LIMITED_RAMP
  assert result.output_torque == result.unshaped_output


def test_same_direction_safety_limit_keeps_stale_reversal_cap():
  result = assert_cap_only(make_inputs(v_ego=5.0, steer_limited_by_safety=True, steer_limit_same_direction=True,
                                       steering_rate_deg=40.0, unshaped_output=0.8,
                                       desired_lateral_accel=0.8, actual_lateral_accel=0.72,
                                       steer_limit_requested_output=0.8, steer_limit_applied_output=-0.25))

  assert result.active
  assert result.reason & ConservativeOutputShapingReason.STALE_ACTUATOR_REVERSAL
  assert not result.reason & ConservativeOutputShapingReason.SAFETY_LIMITED_RAMP
  assert result.output_cap == 0.35


def test_recent_actuator_lag_comfort_does_not_rate_limit_next_low_rate_opposing_output():
  shaper = TorqueConservativeOutputShaper(dt=0.01)
  shaper.update(make_inputs(v_ego=5.0, steer_limited_by_safety=True, steering_rate_deg=40.0, unshaped_output=0.8))

  result = shaper.update(make_inputs(v_ego=5.0, steer_limited_by_safety=True, steering_rate_deg=5.0, unshaped_output=-0.8))

  assert not result.active
  assert not result.reason & ConservativeOutputShapingReason.OUTPUT_RATE_LIMITED
  assert result.output_torque == result.unshaped_output


def test_actuator_lag_comfort_does_not_reduce_driver_override_cap():
  result = assert_cap_only(
    make_inputs(v_ego=5.0, steering_pressed=True, steer_limited_by_safety=True, steering_rate_deg=40.0, unshaped_output=1.0)
  )

  assert result.active
  assert result.reason & ConservativeOutputShapingReason.STEERING_PRESSED
  assert not result.reason & ConservativeOutputShapingReason.ACTUATOR_LAG_COMFORT
  assert result.output_cap == 0.8
  assert result.output_torque == 0.8


def test_same_sign_unwind_release_clamps_output_toward_zero():
  result = assert_cap_only(
    make_inputs(
      v_ego=5.0,
      same_sign_unwind_release=True,
      unshaped_output=0.6,
      desired_lateral_accel=0.05,
      actual_lateral_accel=0.35,
      desired_lateral_jerk=-0.8,
      lookahead_lateral_jerk=-0.4,
    )
  )

  assert result.active
  assert result.reason & ConservativeOutputShapingReason.SAME_SIGN_UNWIND
  assert result.output_cap == 0.3
  assert result.output_torque == 0.18


def test_same_sign_unwind_does_not_clamp_when_release_flag_is_clear():
  result = assert_cap_only(
    make_inputs(
      v_ego=5.0,
      same_sign_unwind_release=False,
      unshaped_output=-0.6,
      desired_lateral_accel=0.05,
      actual_lateral_accel=0.35,
      desired_lateral_jerk=-0.8,
      lookahead_lateral_jerk=-0.4,
    )
  )

  assert not result.active
  assert result.reason == 0
  assert result.output_cap == 1.0
  assert result.output_torque == result.unshaped_output


def test_same_sign_unwind_cap_does_not_rate_limit_next_corrective_output():
  shaper = TorqueConservativeOutputShaper(dt=0.01)
  shaper.update(
    make_inputs(
      v_ego=5.0,
      same_sign_unwind_release=True,
      unshaped_output=0.6,
      desired_lateral_accel=0.05,
      actual_lateral_accel=0.35,
      desired_lateral_jerk=-0.8,
      lookahead_lateral_jerk=-0.4,
    )
  )

  result = shaper.update(
    make_inputs(
      v_ego=5.0,
      same_sign_unwind_release=False,
      unshaped_output=-0.6,
      desired_lateral_accel=0.05,
      actual_lateral_accel=0.35,
      desired_lateral_jerk=-0.8,
      lookahead_lateral_jerk=-0.4,
    )
  )

  assert not result.active
  assert result.reason == 0
  assert result.output_torque == result.unshaped_output


def test_high_speed_actuator_lag_unwind_caps_output():
  result = assert_cap_only(make_inputs(
    v_ego=20.0,
    desired_lateral_accel=0.4,
    actual_lateral_accel=0.8,
    steer_limited_by_safety=True,
    steer_limit_same_direction=True,
    steer_limit_unwind=False,
    steer_limit_requested_output=0.5,
    steer_limit_applied_output=-0.1,
  ))

  assert result.active
  assert result.reason & ConservativeOutputShapingReason.HIGH_SPEED_ACTUATOR_LAG_UNWIND
  assert result.output_cap == 0.70
  assert result.output_torque == pytest.approx(0.35)


def test_high_speed_actuator_lag_unwind_does_not_trigger_below_speed():
  result = assert_cap_only(make_inputs(
    v_ego=15.0,
    desired_lateral_accel=0.4,
    actual_lateral_accel=0.8,
    steer_limited_by_safety=True,
    steer_limit_same_direction=True,
    steer_limit_unwind=False,
    steer_limit_requested_output=0.5,
    steer_limit_applied_output=0.1,
  ))

  assert not result.reason & ConservativeOutputShapingReason.HIGH_SPEED_ACTUATOR_LAG_UNWIND


def test_high_speed_actuator_lag_unwind_does_not_trigger_without_gap():
  result = assert_cap_only(make_inputs(
    v_ego=20.0,
    desired_lateral_accel=0.4,
    actual_lateral_accel=0.8,
    steer_limited_by_safety=True,
    steer_limit_same_direction=True,
    steer_limit_unwind=False,
    steer_limit_requested_output=0.5,
    steer_limit_applied_output=0.35,
  ))

  assert not result.reason & ConservativeOutputShapingReason.HIGH_SPEED_ACTUATOR_LAG_UNWIND


def test_high_speed_actuator_lag_unwind_does_not_trigger_on_unwind():
  result = assert_cap_only(make_inputs(
    v_ego=20.0,
    desired_lateral_accel=0.4,
    actual_lateral_accel=0.8,
    steer_limited_by_safety=True,
    steer_limit_same_direction=False,
    steer_limit_unwind=True,
    steer_limit_requested_output=0.5,
    steer_limit_applied_output=0.1,
  ))

  assert not result.reason & ConservativeOutputShapingReason.HIGH_SPEED_ACTUATOR_LAG_UNWIND


def test_high_speed_actuator_lag_unwind_does_not_cap_corrective_output():
  result = assert_cap_only(make_inputs(
    v_ego=20.0,
    unshaped_output=-0.5,
    desired_lateral_accel=0.4,
    actual_lateral_accel=0.8,
    steer_limited_by_safety=True,
    steer_limit_same_direction=True,
    steer_limit_unwind=False,
    steer_limit_requested_output=-0.5,
    steer_limit_applied_output=-0.1,
  ))

  assert not result.active
  assert not result.reason & ConservativeOutputShapingReason.HIGH_SPEED_ACTUATOR_LAG_UNWIND
  assert result.output_torque == result.unshaped_output


def test_strongest_cap_wins_when_multiple_reasons_apply():
  result = assert_cap_only(
    make_inputs(steering_pressed=True, desired_lateral_accel=0.4, actual_lateral_accel=-0.3, actual_lateral_jerk=3.0, lookahead_lateral_jerk=0.0)
  )

  assert result.active
  assert result.reason & ConservativeOutputShapingReason.STEERING_PRESSED
  assert result.reason & ConservativeOutputShapingReason.SIGN_CONFLICT
  assert result.reason & ConservativeOutputShapingReason.BUMP
  assert result.output_cap == 0.8


def test_recovery_after_cap_ramps_upward():
  shaper = TorqueConservativeOutputShaper(dt=0.1)
  capped = shaper.update(make_inputs(steering_pressed=True))
  recovered = shaper.update(make_inputs(unshaped_output=1.0, desired_lateral_accel=1.0, actual_lateral_accel=0.6))

  assert capped.output_torque == 0.4
  assert recovered.active
  assert recovered.reason & ConservativeOutputShapingReason.OUTPUT_RATE_LIMITED
  assert abs(recovered.output_torque - 0.65) < 1e-6
  assert abs(recovered.output_torque) <= abs(recovered.unshaped_output)


def test_low_speed_recovery_after_driver_override_ramps_more_softly():
  shaper = TorqueConservativeOutputShaper(dt=0.1)
  capped = shaper.update(make_inputs(v_ego=5.0, steering_pressed=True))
  recovered = shaper.update(make_inputs(v_ego=5.0, unshaped_output=1.0, desired_lateral_accel=1.0, actual_lateral_accel=0.6))

  assert capped.output_torque == 0.4
  assert recovered.active
  assert recovered.reason & ConservativeOutputShapingReason.OUTPUT_RATE_LIMITED
  assert abs(recovered.output_torque - 0.52) < 1e-6
  assert abs(recovered.output_torque) <= abs(recovered.unshaped_output)


def test_strong_under_response_bypasses_low_speed_recovery_ramp():
  shaper = TorqueConservativeOutputShaper(dt=0.1)
  shaper.update(make_inputs(v_ego=5.0, steering_pressed=True))

  recovered = shaper.update(make_inputs(v_ego=5.0, unshaped_output=1.0, desired_lateral_accel=1.2, actual_lateral_accel=0.4))

  assert not recovered.active
  assert not recovered.reason & ConservativeOutputShapingReason.OUTPUT_RATE_LIMITED
  assert recovered.output_torque == recovered.unshaped_output


def test_lower_target_after_cap_applies_immediately():
  shaper = TorqueConservativeOutputShaper(dt=0.01)
  shaper.update(make_inputs(unshaped_output=1.0, steering_pressed=True))
  result = shaper.update(make_inputs(unshaped_output=0.3, desired_lateral_accel=0.4, actual_lateral_accel=0.2))

  assert not result.active
  assert result.output_torque == 0.3


def test_driver_override_cap_applies_immediately_after_clean_tracking():
  shaper = TorqueConservativeOutputShaper(dt=0.01)
  shaper.update(make_inputs(unshaped_output=1.0, desired_lateral_accel=1.0, actual_lateral_accel=0.2))
  result = shaper.update(make_inputs(unshaped_output=1.0, steering_pressed=True))

  assert result.active
  assert result.reason & ConservativeOutputShapingReason.STEERING_PRESSED
  assert result.output_cap == 0.8
  assert result.output_torque == 0.8


def test_near_iso_cap_applies_immediately_after_clean_tracking():
  shaper = TorqueConservativeOutputShaper(dt=0.01)
  shaper.update(make_inputs(unshaped_output=1.0, desired_lateral_accel=1.0, actual_lateral_accel=0.2))
  result = shaper.update(make_inputs(unshaped_output=1.0, desired_lateral_accel=2.75, actual_lateral_accel=2.8))

  assert result.active
  assert result.reason & ConservativeOutputShapingReason.NEAR_ISO_ACCEL
  assert result.output_cap == 0.85
  assert result.output_torque == 0.85


def test_near_iso_corrective_output_is_not_rate_limited_after_cap():
  shaper = TorqueConservativeOutputShaper(dt=0.01)
  shaper.update(
    make_inputs(unshaped_output=-0.5, steering_pressed=True, desired_lateral_accel=2.9, actual_lateral_accel=3.1)
  )
  result = shaper.update(make_inputs(unshaped_output=-1.0, desired_lateral_accel=2.9, actual_lateral_accel=3.1))

  assert not result.active
  assert result.output_torque == result.unshaped_output


def test_recent_hard_shaping_rate_limits_clear_under_response_recovery():
  hard_shaping_cases = (
    {"release_active": True},
    {"desired_lateral_accel": 0.4, "actual_lateral_accel": -0.2},
    {"actual_lateral_jerk": 3.0, "lookahead_lateral_jerk": 0.0, "desired_lateral_jerk": 0.0},
    {"desired_lateral_accel": 2.75, "actual_lateral_accel": 2.8},
    {"desired_lateral_accel": 0.4, "actual_lateral_accel": 0.8},
  )
  for previous_inputs in hard_shaping_cases:
    shaper = TorqueConservativeOutputShaper(dt=0.01)
    capped = shaper.update(make_inputs(unshaped_output=0.8, **previous_inputs))

    result = shaper.update(make_inputs(unshaped_output=1.0, desired_lateral_accel=0.8, actual_lateral_accel=0.4))

    assert capped.active
    assert result.active
    assert result.reason & ConservativeOutputShapingReason.OUTPUT_RATE_LIMITED
    assert 0.0 < result.output_torque < result.unshaped_output


def test_sign_flip_after_cap_does_not_hold_old_opposite_torque():
  shaper = TorqueConservativeOutputShaper(dt=0.01)
  shaper.update(make_inputs(steering_pressed=True))
  result = shaper.update(make_inputs(unshaped_output=-1.0, desired_lateral_accel=-0.8, actual_lateral_accel=-0.2))

  assert result.active
  assert result.reason & ConservativeOutputShapingReason.OUTPUT_RATE_LIMITED
  assert result.output_torque < 0.0
  assert abs(result.output_torque - -0.04) < 1e-6
  assert abs(result.output_torque) <= abs(result.unshaped_output)


def test_output_rate_limit_expires_after_short_window():
  shaper = TorqueConservativeOutputShaper(dt=0.2)
  shaper.update(make_inputs(steering_pressed=True))
  shaper.update(make_inputs(unshaped_output=0.2, desired_lateral_accel=0.4, actual_lateral_accel=0.2))
  shaper.update(make_inputs(unshaped_output=0.2, desired_lateral_accel=0.4, actual_lateral_accel=0.2))
  result = shaper.update(make_inputs(unshaped_output=1.0, desired_lateral_accel=1.0, actual_lateral_accel=0.2))

  assert not result.active
  assert result.output_torque == result.unshaped_output
