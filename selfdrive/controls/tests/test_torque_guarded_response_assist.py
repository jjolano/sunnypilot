from openpilot.sunnypilot.selfdrive.controls.lib.torque_guarded_response_assist import GuardedResponseAssistInputs, TorqueGuardedResponseAssist


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


def test_release_on_override():
  assist = TorqueGuardedResponseAssist(0.01)
  for _ in range(40):
    assist.update(make_inputs())

  result = assist.update(
    make_inputs(steering_pressed=True, desired_lateral_accel=0.05, desired_lateral_jerk=0.05, lookahead_lateral_jerk=0.05, desired_curvature=0.005)
  )
  assert result.phase == "RELEASE"
  assert result.release_active
  assert abs(result.assist_torque) < 0.14


def test_bump_disturbance_freezes_learning():
  assist = TorqueGuardedResponseAssist(0.01)
  result = assist.update(make_inputs(actual_lateral_jerk=3.0, lookahead_lateral_jerk=0.0, desired_lateral_jerk=0.0))

  assert result.learning_frozen
  assert abs(result.assist_torque) < 1e-6


def test_saturated_under_response_does_not_add_same_direction_assist():
  assist = TorqueGuardedResponseAssist(0.01)
  result = None
  for _ in range(40):
    result = assist.update(make_inputs(nominal_torque=1.0, saturated=True, tracking_torque_error=0.12))

  assert result is not None
  assert result.learning_frozen
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
