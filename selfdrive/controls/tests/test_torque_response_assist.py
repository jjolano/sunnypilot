from openpilot.sunnypilot.selfdrive.controls.lib.torque_response_assist import ResponseAssistInputs, TorqueResponseAssist


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
  }
  values.update(overrides)
  return ResponseAssistInputs(**values)


def test_assist_builds_for_same_sign_response_deficit():
  assist = TorqueResponseAssist(0.01)
  result = None
  for _ in range(40):
    result = assist.update(make_inputs())

  assert result is not None
  assert result.phase == "ASSIST"
  assert result.assist_torque > 0.0
  assert not result.release_active


def test_release_on_override():
  assist = TorqueResponseAssist(0.01)
  for _ in range(40):
    assist.update(make_inputs())

  result = assist.update(
    make_inputs(steering_pressed=True, desired_lateral_accel=0.05, desired_lateral_jerk=0.05, lookahead_lateral_jerk=0.05, desired_curvature=0.005)
  )
  assert result.phase == "RELEASE"
  assert result.release_active
  assert abs(result.assist_torque) < 0.18


def test_hold_builds_bias_without_assist():
  assist = TorqueResponseAssist(0.01)
  result = None
  for _ in range(60):
    result = assist.update(make_inputs(tracking_torque_error=0.03, desired_lateral_jerk=0.05, lookahead_lateral_jerk=0.05))

  assert result is not None
  assert result.phase == "HOLD"
  assert abs(result.assist_torque) < 1e-6
  assert result.bias_torque > 0.0
