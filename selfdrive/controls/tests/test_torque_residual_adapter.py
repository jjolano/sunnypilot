from openpilot.sunnypilot.selfdrive.controls.lib.torque_residual_adapter import FREEZE_TIME, ResidualAdapterInputs, TorqueResidualAdapter


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
  return ResidualAdapterInputs(**values)


def prime_hold(adapter: TorqueResidualAdapter, **overrides):
  result = None
  for _ in range(60):
    result = adapter.update(make_inputs(tracking_torque_error=0.03, desired_lateral_jerk=0.05, lookahead_lateral_jerk=0.05, **overrides))
  return result


def test_assist_builds_for_same_sign_response_deficit():
  adapter = TorqueResidualAdapter(0.01)
  result = None
  for _ in range(40):
    result = adapter.update(make_inputs())

  assert result is not None
  assert result.phase == "ENGAGE"
  assert result.assist_torque > 0.0
  assert not result.release_active


def test_hold_builds_context_bias_without_assist():
  adapter = TorqueResidualAdapter(0.01)
  result = prime_hold(adapter)

  assert result is not None
  assert result.phase == "HOLD"
  assert abs(result.assist_torque) < 1e-6
  assert result.bias_torque > 0.0
  assert any(context.bias_torque > 0.0 for context in adapter.contexts.values())


def test_bump_freezes_context_learning():
  adapter = TorqueResidualAdapter(0.01)
  prime_hold(adapter)

  bucket = next(iter(adapter.contexts.values()))
  learned_bias = bucket.bias_torque
  result = None
  for _ in range(int(FREEZE_TIME / adapter.dt / 2)):
    result = adapter.update(make_inputs(actual_lateral_jerk=3.0, lookahead_lateral_jerk=0.0, desired_lateral_jerk=0.0, tracking_torque_error=0.06))

  assert result is not None
  assert result.learning_frozen
  assert next(iter(adapter.contexts.values())).bias_torque == learned_bias


def test_release_on_override():
  adapter = TorqueResidualAdapter(0.01)
  for _ in range(40):
    adapter.update(make_inputs())

  result = adapter.update(
    make_inputs(steering_pressed=True, desired_lateral_accel=0.05, desired_lateral_jerk=0.05, lookahead_lateral_jerk=0.05, desired_curvature=0.005)
  )
  assert result.phase == "RELEASE"
  assert result.release_active


def test_low_demand_without_residual_stays_idle():
  adapter = TorqueResidualAdapter(0.01)
  result = adapter.update(
    make_inputs(desired_lateral_accel=0.05, actual_lateral_accel=0.03, desired_lateral_jerk=0.05, lookahead_lateral_jerk=0.05, desired_curvature=0.005)
  )

  assert result.phase == "IDLE"
  assert not result.release_active


def test_saturated_over_response_can_build_negative_bias():
  adapter = TorqueResidualAdapter(0.01)
  result = None
  for _ in range(80):
    result = adapter.update(
      make_inputs(
        nominal_torque=1.0,
        saturated=True,
        desired_lateral_accel=0.8,
        actual_lateral_accel=1.0,
        desired_lateral_jerk=0.05,
        lookahead_lateral_jerk=0.05,
        tracking_torque_error=-0.08,
      )
    )

  assert result is not None
  assert result.bias_torque < 0.0
  assert result.output_torque < 1.0
  assert not result.learning_frozen


def test_same_sign_unwind_trims_over_response():
  adapter = TorqueResidualAdapter(0.01)
  result = adapter.update(
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


def test_same_sign_unwind_blocks_same_direction_residual():
  adapter = TorqueResidualAdapter(0.01)
  prime_hold(adapter)

  result = adapter.update(
    make_inputs(
      v_ego=5.0,
      nominal_torque=0.9,
      desired_lateral_accel=0.05,
      actual_lateral_accel=0.35,
      desired_lateral_jerk=0.6,
      lookahead_lateral_jerk=0.0,
      tracking_torque_error=0.08,
      same_sign_unwind=True,
    )
  )

  assert result.assist_torque <= 0.0
  assert result.bias_torque <= 0.0
  assert result.output_torque <= result.nominal_torque


def test_output_clamps_and_lane_changes_scale_assist():
  normal = TorqueResidualAdapter(0.01)
  lane_change = TorqueResidualAdapter(0.01)
  clamped = TorqueResidualAdapter(0.01)
  normal_result = None
  lane_change_result = None
  for _ in range(40):
    normal_result = normal.update(make_inputs())
    lane_change_result = lane_change.update(make_inputs(lane_change_active=True))

  clamped_result = None
  for _ in range(40):
    clamped_result = clamped.update(make_inputs(max_output=0.25, nominal_torque=0.24, tracking_torque_error=0.2))

  assert normal_result is not None and lane_change_result is not None and clamped_result is not None
  assert lane_change_result.assist_torque < normal_result.assist_torque
  assert abs(clamped_result.output_torque - 0.25) < 1e-6


def test_saturated_under_response_does_not_add_same_direction_residual():
  adapter = TorqueResidualAdapter(0.01)
  result = None
  for _ in range(40):
    result = adapter.update(make_inputs(nominal_torque=1.0, saturated=True, tracking_torque_error=0.12))

  assert result is not None
  assert abs(result.assist_torque) < 1e-6
  assert abs(result.bias_torque) < 1e-6
  assert result.learning_frozen
