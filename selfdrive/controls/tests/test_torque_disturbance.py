from openpilot.sunnypilot.selfdrive.controls.lib.torque_disturbance import (
  TorqueDisturbanceInputs,
  TorqueDisturbanceReason,
  TorqueDisturbanceState,
  classify_torque_disturbance,
)


def make_inputs(**overrides):
  values = {
    "active": True,
    "v_ego": 15.0,
    "steering_pressed": False,
    "steer_limited_by_safety": False,
    "curvature_limited": False,
    "saturated": False,
    "desired_lateral_accel": 0.8,
    "actual_lateral_accel": 0.78,
    "desired_lateral_jerk": 0.1,
    "actual_lateral_jerk": 0.1,
    "lookahead_lateral_jerk": 0.1,
    "output_torque": 0.2,
    "response_deficit": 0.0,
    "same_sign_unwind": False,
    "measurement_reset": False,
    "measurement_valid": True,
  }
  values.update(overrides)
  return TorqueDisturbanceInputs(**values)


def test_clean_inputs_report_no_disturbance():
  result = classify_torque_disturbance(make_inputs())

  assert result.state == TorqueDisturbanceState.NONE
  assert result.reason == TorqueDisturbanceReason.NONE
  assert result.confidence == 0.0


def test_bump_jerk_reports_active_disturbance():
  result = classify_torque_disturbance(
    make_inputs(actual_lateral_jerk=3.0, lookahead_lateral_jerk=0.0, desired_lateral_jerk=0.0)
  )

  assert result.state == TorqueDisturbanceState.ACTIVE
  assert result.reason & TorqueDisturbanceReason.BUMP_JERK
  assert result.confidence > 0.0


def test_sign_conflict_reports_active_disturbance():
  result = classify_torque_disturbance(make_inputs(desired_lateral_accel=0.4, actual_lateral_accel=-0.2))

  assert result.state == TorqueDisturbanceState.ACTIVE
  assert result.reason & TorqueDisturbanceReason.SIGN_CONFLICT
  assert result.confidence == 1.0


def test_over_response_reports_active_disturbance_when_output_reinforces_actual():
  result = classify_torque_disturbance(make_inputs(desired_lateral_accel=0.4, actual_lateral_accel=0.7, output_torque=0.5))

  assert result.state == TorqueDisturbanceState.ACTIVE
  assert result.reason & TorqueDisturbanceReason.OVER_RESPONSE
  assert result.confidence > 0.0


def test_response_deficit_is_suspected_only():
  result = classify_torque_disturbance(make_inputs(response_deficit=0.12))

  assert result.state == TorqueDisturbanceState.SUSPECTED
  assert result.reason & TorqueDisturbanceReason.RESPONSE_DEFICIT
  assert result.confidence > 0.0


def test_measurement_reset_is_logged_only_when_active_uncontrolled_and_moving():
  ignored = classify_torque_disturbance(make_inputs(measurement_reset=True, steering_pressed=True))
  logged = classify_torque_disturbance(make_inputs(measurement_reset=True, v_ego=15.0))

  assert not ignored.reason & TorqueDisturbanceReason.MEASUREMENT_RESET_OR_INVALID
  assert logged.state == TorqueDisturbanceState.SUSPECTED
  assert logged.reason & TorqueDisturbanceReason.MEASUREMENT_RESET_OR_INVALID
