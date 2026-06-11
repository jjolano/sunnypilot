from openpilot.sunnypilot.selfdrive.controls.lib.torque_guarded_response_assist import (
  GuardedResponseAssistInputs,
  GuardedResponseReason,
  TorqueGuardedResponseAssist,
)


def make_inputs(**overrides):
  values = {
    "active": True,
    "v_ego": 15.0,
    "steering_pressed": False,
    "steer_limited_by_safety": False,
    "curvature_limited": False,
    "saturated": False,
    "max_output": 1.0,
    "nominal_torque": 0.2,
    "desired_lateral_accel": 0.8,
    "actual_lateral_accel": 0.4,
    "desired_lateral_jerk": 0.2,
    "actual_lateral_jerk": 0.05,
    "lookahead_lateral_jerk": 0.2,
    "desired_curvature": 0.03,
    "tracking_torque_error": 0.12,
    "lane_change_active": False,
    "same_sign_unwind": False,
  }
  values.update(overrides)
  return GuardedResponseAssistInputs(**values)


def test_assist_builds_for_same_sign_response_deficit():
  assist = TorqueGuardedResponseAssist(0.01)
  result = None
  for _ in range(40):
    result = assist.update(make_inputs())

  assert result is not None
  assert result.phase == "ASSIST"
  assert result.assist_torque > 0.0
  assert not result.release_active
  assert not result.learning_frozen
  assert result.freeze_reason == 0
  assert result.block_reason == 0


def test_curve_preposition_adds_positive_assist_before_saturation():
  assist = TorqueGuardedResponseAssist(0.01)
  result = None
  for _ in range(40):
    result = assist.update(
      make_inputs(
        v_ego=8.0,
        nominal_torque=0.72,
        desired_lateral_accel=1.2,
        actual_lateral_accel=0.72,
        desired_lateral_jerk=0.65,
        actual_lateral_jerk=0.08,
        lookahead_lateral_jerk=0.55,
        desired_curvature=0.035,
        tracking_torque_error=0.03,
      )
    )

  assert result is not None
  assert result.phase == "ASSIST"
  assert 0.0 < result.assist_torque <= 0.035
  assert result.output_torque > result.nominal_torque
  assert not result.learning_frozen
  assert result.freeze_reason == 0


def test_curve_preposition_adds_negative_assist_before_saturation():
  assist = TorqueGuardedResponseAssist(0.01)
  result = None
  for _ in range(40):
    result = assist.update(
      make_inputs(
        v_ego=8.0,
        nominal_torque=-0.72,
        desired_lateral_accel=-1.2,
        actual_lateral_accel=-0.72,
        desired_lateral_jerk=-0.65,
        actual_lateral_jerk=-0.08,
        lookahead_lateral_jerk=-0.55,
        desired_curvature=-0.035,
        tracking_torque_error=-0.03,
      )
    )

  assert result is not None
  assert result.phase == "ASSIST"
  assert -0.035 <= result.assist_torque < 0.0
  assert result.output_torque < result.nominal_torque
  assert not result.learning_frozen
  assert result.freeze_reason == 0


def test_curve_preposition_does_not_trigger_when_output_is_pinned():
  assist = TorqueGuardedResponseAssist(0.01)
  result = None
  for _ in range(40):
    result = assist.update(
      make_inputs(
        v_ego=8.0,
        nominal_torque=0.94,
        desired_lateral_accel=1.2,
        actual_lateral_accel=0.72,
        desired_lateral_jerk=0.65,
        actual_lateral_jerk=0.08,
        lookahead_lateral_jerk=0.55,
        desired_curvature=0.035,
        tracking_torque_error=0.03,
      )
    )

  assert result is not None
  assert result.assist_torque == 0.0
  assert result.bias_torque == 0.0


def test_curve_preposition_respects_existing_limit_blocks():
  for blocked_input in (
    {"saturated": True},
    {"steer_limited_by_safety": True},
    {"curvature_limited": True},
    {"steering_pressed": True},
    {"actual_lateral_accel": -0.72},
    {"actual_lateral_jerk": 3.0},
    {"v_ego": 3.5},
    {"lane_change_active": True},
  ):
    assist = TorqueGuardedResponseAssist(0.01)
    preposition_input = {
      "v_ego": 8.0,
      "nominal_torque": 0.72,
      "desired_lateral_accel": 1.2,
      "actual_lateral_accel": 0.72,
      "desired_lateral_jerk": 0.65,
      "actual_lateral_jerk": 0.08,
      "lookahead_lateral_jerk": 0.55,
      "desired_curvature": 0.035,
      "tracking_torque_error": 0.03,
      **blocked_input,
    }
    result = None
    for _ in range(40):
      result = assist.update(make_inputs(**preposition_input))

    assert result is not None
    assert abs(result.assist_torque) < 1e-6


def test_curve_preposition_does_not_trigger_when_response_rate_keeps_up():
  assist = TorqueGuardedResponseAssist(0.01)
  result = None
  for _ in range(40):
    result = assist.update(
      make_inputs(
        v_ego=8.0,
        nominal_torque=0.72,
        desired_lateral_accel=1.2,
        actual_lateral_accel=0.72,
        desired_lateral_jerk=0.65,
        actual_lateral_jerk=0.55,
        lookahead_lateral_jerk=0.55,
        desired_curvature=0.035,
        tracking_torque_error=0.03,
      )
    )

  assert result is not None
  assert result.assist_torque == 0.0
  assert result.bias_torque == 0.0


def test_curve_exit_under_response_adds_conservative_positive_assist():
  assist = TorqueGuardedResponseAssist(0.01)
  result = None
  for _ in range(40):
    result = assist.update(
      make_inputs(
        nominal_torque=0.35,
        desired_lateral_accel=1.0,
        actual_lateral_accel=0.68,
        desired_lateral_jerk=-0.45,
        lookahead_lateral_jerk=-0.10,
        desired_curvature=0.035,
        tracking_torque_error=0.03,
      )
    )

  assert result is not None
  assert result.phase == "ASSIST"
  assert 0.0 < result.assist_torque <= 0.04
  assert result.bias_torque == 0.0
  assert not result.learning_frozen
  assert result.freeze_reason == 0


def test_curve_exit_under_response_adds_conservative_negative_assist():
  assist = TorqueGuardedResponseAssist(0.01)
  result = None
  for _ in range(40):
    result = assist.update(
      make_inputs(
        nominal_torque=-0.35,
        desired_lateral_accel=-1.0,
        actual_lateral_accel=-0.68,
        desired_lateral_jerk=0.45,
        lookahead_lateral_jerk=0.10,
        desired_curvature=-0.035,
        tracking_torque_error=-0.03,
      )
    )

  assert result is not None
  assert result.phase == "ASSIST"
  assert -0.04 <= result.assist_torque < 0.0
  assert result.bias_torque == 0.0
  assert not result.learning_frozen
  assert result.freeze_reason == 0


def test_curve_exit_assist_does_not_trigger_on_turn_in():
  assist = TorqueGuardedResponseAssist(0.01)
  result = None
  for _ in range(40):
    result = assist.update(
      make_inputs(
        nominal_torque=0.35,
        desired_lateral_accel=1.0,
        actual_lateral_accel=0.68,
        desired_lateral_jerk=0.45,
        lookahead_lateral_jerk=0.10,
        desired_curvature=0.035,
        tracking_torque_error=0.03,
      )
    )

  assert result is not None
  assert result.assist_torque == 0.0
  assert result.bias_torque == 0.0


def test_curve_exit_assist_respects_existing_limit_blocks():
  for blocked_input in (
    {"saturated": True},
    {"steer_limited_by_safety": True},
    {"curvature_limited": True},
    {"steering_pressed": True},
    {"actual_lateral_accel": -0.68},
    {"actual_lateral_jerk": 3.0},
    {"v_ego": 3.5},
  ):
    assist = TorqueGuardedResponseAssist(0.01)
    curve_exit_input = {
      "nominal_torque": 0.35,
      "desired_lateral_accel": 1.0,
      "actual_lateral_accel": 0.68,
      "desired_lateral_jerk": -0.45,
      "lookahead_lateral_jerk": -0.10,
      "desired_curvature": 0.035,
      "tracking_torque_error": 0.08,
      **blocked_input,
    }
    result = None
    for _ in range(40):
      result = assist.update(make_inputs(**curve_exit_input))

    assert result is not None
    assert abs(result.assist_torque) < 1e-6


def test_steering_override_freezes_without_release():
  assist = TorqueGuardedResponseAssist(0.01)
  for _ in range(40):
    assist.update(make_inputs())

  result = assist.update(make_inputs(steering_pressed=True))

  assert result.phase != "RELEASE"
  assert not result.release_active
  assert result.learning_frozen
  assert result.freeze_reason & GuardedResponseReason.STEERING_PRESSED
  assert result.block_reason & GuardedResponseReason.STEERING_PRESSED
  assert not result.block_reason & GuardedResponseReason.RELEASE


def test_bump_disturbance_freezes_learning():
  assist = TorqueGuardedResponseAssist(0.01)
  result = assist.update(make_inputs(actual_lateral_jerk=3.0, lookahead_lateral_jerk=0.0, desired_lateral_jerk=0.0))

  assert result.learning_frozen
  assert result.freeze_reason & GuardedResponseReason.BUMP
  assert result.block_reason & GuardedResponseReason.BUMP
  assert abs(result.assist_torque) < 1e-6


def test_saturated_under_response_does_not_add_same_direction_assist():
  assist = TorqueGuardedResponseAssist(0.01)
  result = None
  for _ in range(40):
    result = assist.update(make_inputs(nominal_torque=1.0, saturated=True, tracking_torque_error=0.12))

  assert result is not None
  assert result.learning_frozen
  assert result.freeze_reason & GuardedResponseReason.SATURATED
  assert result.block_reason & GuardedResponseReason.SATURATED
  assert abs(result.assist_torque) < 1e-6
  assert abs(result.bias_torque) < 1e-6


def test_lane_change_scales_assist_lower():
  normal = TorqueGuardedResponseAssist(0.01)
  lane_change = TorqueGuardedResponseAssist(0.01)
  normal_result = None
  lane_change_result = None
  for _ in range(40):
    normal_result = normal.update(make_inputs())
    lane_change_result = lane_change.update(make_inputs(lane_change_active=True))

  assert normal_result is not None
  assert lane_change_result is not None
  assert lane_change_result.assist_torque < normal_result.assist_torque
  assert lane_change_result.block_reason & GuardedResponseReason.LANE_CHANGE


def test_hold_bias_does_not_build_inward_during_over_response():
  assist = TorqueGuardedResponseAssist(0.01)
  result = None
  for _ in range(40):
    result = assist.update(
      make_inputs(
        nominal_torque=0.2,
        desired_lateral_accel=0.8,
        actual_lateral_accel=1.05,
        desired_lateral_jerk=0.0,
        lookahead_lateral_jerk=0.0,
        tracking_torque_error=0.02,
      )
    )

  assert result is not None
  assert result.phase == "HOLD"
  assert result.bias_torque == 0.0


def test_hold_bias_does_not_build_negative_inward_during_over_response():
  assist = TorqueGuardedResponseAssist(0.01)
  result = None
  for _ in range(40):
    result = assist.update(
      make_inputs(
        nominal_torque=-0.2,
        desired_lateral_accel=-0.8,
        actual_lateral_accel=-1.05,
        desired_lateral_jerk=0.0,
        lookahead_lateral_jerk=0.0,
        tracking_torque_error=-0.02,
      )
    )

  assert result is not None
  assert result.phase == "HOLD"
  assert result.bias_torque == 0.0


def test_hold_bias_still_builds_for_clear_under_response():
  assist = TorqueGuardedResponseAssist(0.01)
  result = None
  for _ in range(40):
    result = assist.update(
      make_inputs(
        nominal_torque=0.2,
        desired_lateral_accel=0.8,
        actual_lateral_accel=0.55,
        desired_lateral_jerk=0.0,
        lookahead_lateral_jerk=0.0,
        tracking_torque_error=0.02,
      )
    )

  assert result is not None
  assert result.phase == "HOLD"
  assert result.bias_torque > 0.0


def test_same_sign_unwind_trims_opposite_nominal():
  assist = TorqueGuardedResponseAssist(0.01)
  result = assist.update(
    make_inputs(
      v_ego=5.0,
      nominal_torque=0.9,
      desired_lateral_accel=0.05,
      actual_lateral_accel=0.35,
      desired_lateral_jerk=0.6,
      lookahead_lateral_jerk=0.0,
      tracking_torque_error=-0.08,
      same_sign_unwind=True,
    )
  )

  assert result.phase == "HOLD"
  assert result.assist_torque <= 0.0
  assert result.bias_torque < 0.0
  assert result.output_torque < result.nominal_torque
  assert not result.release_active
  assert result.block_reason & GuardedResponseReason.SAME_SIGN_UNWIND


def test_inactive_reports_block_reason():
  assist = TorqueGuardedResponseAssist(0.01)
  result = assist.update(make_inputs(active=False))

  assert result.block_reason & GuardedResponseReason.INACTIVE
  assert result.freeze_reason == 0


def test_inactive_clears_stored_assist_and_bias():
  assist = TorqueGuardedResponseAssist(0.01)
  for _ in range(40):
    result = assist.update(make_inputs())

  assert result.assist_torque > 0.0

  result = assist.update(make_inputs(active=False))

  assert result.block_reason & GuardedResponseReason.INACTIVE
  assert result.assist_torque == 0.0
  assert result.bias_torque == 0.0
  assert assist.assist_torque == 0.0
  assert assist.bias_torque == 0.0


def test_inactive_clears_freeze_timer():
  assist = TorqueGuardedResponseAssist(0.01)

  inactive = assist.update(make_inputs(active=False, actual_lateral_jerk=3.0, lookahead_lateral_jerk=0.0, desired_lateral_jerk=0.0))
  resumed = assist.update(make_inputs(actual_lateral_jerk=0.0, lookahead_lateral_jerk=0.0, desired_lateral_jerk=0.0))

  assert inactive.block_reason & GuardedResponseReason.INACTIVE
  assert not inactive.learning_frozen
  assert not resumed.learning_frozen
  assert resumed.freeze_reason == 0
