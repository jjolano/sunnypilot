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
  assert result.output_cap == 0.85


def test_over_response_does_not_cap_corrective_output():
  result = assert_cap_only(make_inputs(unshaped_output=-0.5, desired_lateral_accel=0.4, actual_lateral_accel=0.8))

  assert not result.active
  assert result.output_torque == result.unshaped_output


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
  assert result.output_torque == -0.425


def test_bump_response_caps_output():
  result = assert_cap_only(make_inputs(actual_lateral_jerk=3.0, lookahead_lateral_jerk=0.0, desired_lateral_jerk=0.0))

  assert result.active
  assert result.reason & ConservativeOutputShapingReason.BUMP
  assert result.output_cap == 0.9


def test_low_speed_steer_limited_high_output_caps_output():
  result = assert_cap_only(make_inputs(v_ego=5.0, steer_limited_by_safety=True, unshaped_output=0.8))

  assert result.active
  assert result.reason & ConservativeOutputShapingReason.LOW_SPEED_STEER_LIMITED
  assert result.output_cap == 0.92


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
  recovered = shaper.update(make_inputs(unshaped_output=1.0, desired_lateral_accel=1.0, actual_lateral_accel=0.2))

  assert capped.output_torque == 0.4
  assert recovered.active
  assert recovered.reason & ConservativeOutputShapingReason.OUTPUT_RATE_LIMITED
  assert abs(recovered.output_torque - 0.65) < 1e-6
  assert abs(recovered.output_torque) <= abs(recovered.unshaped_output)


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
