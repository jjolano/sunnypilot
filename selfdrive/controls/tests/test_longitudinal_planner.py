from types import SimpleNamespace

from openpilot.selfdrive.controls.lib.longitudinal_planner import (
  E2E_STOP_APPROACH_DECEL_MAX,
  get_e2e_stop_approach_accel,
  has_model_stop_context,
  has_valid_radar_lead,
  should_run_engage_stop_bootstrap,
)


def make_radar_state(lead_one=False, lead_two=False):
  return SimpleNamespace(
    leadOne=SimpleNamespace(status=lead_one),
    leadTwo=SimpleNamespace(status=lead_two),
  )


def make_model_msg(desired_accel=0.0, should_stop=False, endpoint_x=200.0, positions=None, velocities=None):
  return SimpleNamespace(
    action=SimpleNamespace(desiredAcceleration=desired_accel, shouldStop=should_stop),
    position=SimpleNamespace(x=positions if positions is not None else [0.0, endpoint_x]),
    velocity=SimpleNamespace(x=velocities or []),
  )


def test_has_valid_radar_lead_checks_both_tracks():
  assert not has_valid_radar_lead(make_radar_state())
  assert has_valid_radar_lead(make_radar_state(lead_one=True))
  assert has_valid_radar_lead(make_radar_state(lead_two=True))


def test_engage_stop_bootstrap_requires_timer_speed_and_no_lead():
  model_msg = make_model_msg(desired_accel=-1.5)

  assert not should_run_engage_stop_bootstrap(0.0, 10.0, make_radar_state(), model_msg)
  assert not should_run_engage_stop_bootstrap(0.5, 2.0, make_radar_state(), model_msg)
  assert not should_run_engage_stop_bootstrap(0.5, 10.0, make_radar_state(lead_one=True), model_msg)


def test_engage_stop_bootstrap_requires_stop_context_for_negative_model_accel():
  assert not should_run_engage_stop_bootstrap(0.5, 10.0, make_radar_state(), make_model_msg(desired_accel=-1.5))


def test_engage_stop_bootstrap_activates_for_negative_model_accel_with_stop_endpoint():
  model_msg = make_model_msg(desired_accel=-1.5, positions=[0.0, 12.0, 20.0], velocities=[10.0, 4.0, 0.5])

  assert should_run_engage_stop_bootstrap(0.5, 10.0, make_radar_state(), model_msg)


def test_engage_stop_bootstrap_activates_for_model_should_stop_without_lead():
  assert should_run_engage_stop_bootstrap(0.5, 10.0, make_radar_state(), make_model_msg(should_stop=True))


def test_engage_stop_bootstrap_model_stop_context_uses_low_predicted_velocity():
  assert has_model_stop_context(make_model_msg(positions=[0.0, 20.0], velocities=[10.0, 0.5]))
  assert not has_model_stop_context(make_model_msg(positions=[0.0, 20.0], velocities=[10.0, 5.0]))


def test_engage_stop_bootstrap_ignores_weak_model_stop_signal():
  assert not should_run_engage_stop_bootstrap(0.5, 10.0, make_radar_state(), make_model_msg(desired_accel=-0.2))


def test_e2e_stop_approach_brakes_for_short_no_lead_endpoint():
  accel = get_e2e_stop_approach_accel(12.0, make_model_msg(endpoint_x=45.0), make_radar_state(), True)

  assert -E2E_STOP_APPROACH_DECEL_MAX <= accel < -0.5


def test_e2e_stop_approach_ignores_endpoint_with_sufficient_runway():
  accel = get_e2e_stop_approach_accel(12.0, make_model_msg(endpoint_x=70.0), make_radar_state(), True)

  assert accel == 0.0


def test_e2e_stop_approach_brakes_before_high_speed_max_decel_boundary():
  accel = get_e2e_stop_approach_accel(60.0 / 3.6, make_model_msg(endpoint_x=112.0), make_radar_state(), True)

  assert -E2E_STOP_APPROACH_DECEL_MAX <= accel < -0.5


def test_e2e_stop_approach_ignores_clear_endpoint():
  assert get_e2e_stop_approach_accel(12.0, make_model_msg(endpoint_x=200.0), make_radar_state(), True) == 0.0


def test_e2e_stop_approach_requires_no_lead_and_no_override():
  model_msg = make_model_msg(endpoint_x=45.0)

  assert get_e2e_stop_approach_accel(12.0, model_msg, make_radar_state(lead_one=True), True) == 0.0
  assert get_e2e_stop_approach_accel(12.0, model_msg, make_radar_state(), False) == 0.0
  assert get_e2e_stop_approach_accel(12.0, model_msg, make_radar_state(), True, brake_pressed=True) == 0.0
  assert get_e2e_stop_approach_accel(12.0, model_msg, make_radar_state(), True, gas_pressed=True) == 0.0


def test_e2e_stop_approach_leaves_hard_model_stop_to_model():
  accel = get_e2e_stop_approach_accel(12.0, make_model_msg(should_stop=True, endpoint_x=30.0), make_radar_state(), True)

  assert accel == 0.0
