from types import SimpleNamespace
import math

import numpy as np
from opendbc.car.interfaces import ACCEL_MAX

from openpilot.selfdrive.controls.lib.longitudinal_planner import (
  E2E_CLOSE_STOP_DECEL_MAX,
  E2E_CLOSE_STOP_MIN_ROLLING_V,
  E2E_STOP_APPROACH_DECEL_MAX,
  E2E_RUNWAY_FINAL_CRAWL_ACCEL_MAX,
  LongitudinalPlanner,
  get_e2e_close_stop_settle,
  get_max_accel,
  get_e2e_runway_comfort_accel,
  get_e2e_runway_positive_accel_cap,
  get_e2e_stop_approach_accel,
  get_model_stop_distance,
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


class FakeSubMaster(dict):
  logMonoTime = {'modelV2': 1.0}

  def all_checks(self, service_list):
    return True


def test_has_valid_radar_lead_checks_both_tracks():
  assert not has_valid_radar_lead(make_radar_state())
  assert has_valid_radar_lead(make_radar_state(lead_one=True))
  assert has_valid_radar_lead(make_radar_state(lead_two=True))


def test_publish_has_lead_checks_both_tracks(monkeypatch):
  planner = LongitudinalPlanner.__new__(LongitudinalPlanner)
  planner.mpc = SimpleNamespace(solve_time=0.0, source="lead1")
  planner.v_desired_trajectory = np.zeros(3)
  planner.a_desired_trajectory = np.zeros(3)
  planner.j_desired_trajectory = np.zeros(2)
  planner.fcw = False
  planner.output_a_target = 0.0
  planner.output_should_stop = False
  planner.allow_throttle = True
  planner.publish_longitudinal_plan_sp = lambda _sm, _pm: None
  sm = FakeSubMaster({
    'radarState': make_radar_state(lead_two=True),
  })
  plan_send = SimpleNamespace(logMonoTime=2_000_000_000, longitudinalPlan=SimpleNamespace())
  pm = SimpleNamespace(sent=None, send=lambda _service, msg: setattr(pm, "sent", msg))

  monkeypatch.setattr("openpilot.selfdrive.controls.lib.longitudinal_planner.messaging.new_message", lambda _service: plan_send)

  planner.publish(sm, pm)

  assert pm.sent.longitudinalPlan.hasLead


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


def test_model_stop_distance_uses_first_low_velocity_point():
  model_msg = make_model_msg(positions=[0.0, 0.8, 3.0], velocities=[1.0, 0.2, 0.0])

  assert get_model_stop_distance(model_msg) == 0.8


def test_engage_stop_bootstrap_ignores_weak_model_stop_signal():
  assert not should_run_engage_stop_bootstrap(0.5, 10.0, make_radar_state(), make_model_msg(desired_accel=-0.2))


def test_e2e_stop_approach_brakes_for_short_no_lead_endpoint():
  accel = get_e2e_stop_approach_accel(12.0, make_model_msg(endpoint_x=45.0), make_radar_state(), True)

  assert -E2E_STOP_APPROACH_DECEL_MAX <= accel < -0.5


def test_e2e_stop_approach_ignores_endpoint_with_sufficient_runway():
  accel = get_e2e_stop_approach_accel(12.0, make_model_msg(endpoint_x=70.0), make_radar_state(), True)

  assert accel == 0.0


def test_e2e_stop_approach_starts_mild_decel_for_route_like_runway():
  accel = get_e2e_stop_approach_accel(15.7, make_model_msg(endpoint_x=84.0), make_radar_state(), True)

  assert -0.5 < accel < -0.15


def test_e2e_stop_approach_brakes_before_high_speed_max_decel_boundary():
  accel = get_e2e_stop_approach_accel(60.0 / 3.6, make_model_msg(endpoint_x=90.0), make_radar_state(), True)

  assert -E2E_STOP_APPROACH_DECEL_MAX <= accel < -0.5


def test_e2e_stop_approach_caps_route_like_peak_decel():
  accel = get_e2e_stop_approach_accel(60.0 / 3.6, make_model_msg(endpoint_x=45.0), make_radar_state(), True)

  assert math.isclose(accel, -E2E_STOP_APPROACH_DECEL_MAX)


def test_e2e_stop_approach_ignores_clear_endpoint():
  assert get_e2e_stop_approach_accel(12.0, make_model_msg(endpoint_x=200.0), make_radar_state(), True) == 0.0


def test_e2e_stop_approach_requires_no_lead_and_no_override():
  model_msg = make_model_msg(endpoint_x=45.0)

  assert get_e2e_stop_approach_accel(12.0, model_msg, make_radar_state(lead_one=True), True) == 0.0
  assert get_e2e_stop_approach_accel(12.0, model_msg, make_radar_state(), False) == 0.0
  assert get_e2e_stop_approach_accel(12.0, model_msg, make_radar_state(), True, brake_pressed=True) == 0.0
  assert get_e2e_stop_approach_accel(12.0, model_msg, make_radar_state(), True, gas_pressed=True) == 0.0


def test_e2e_stop_approach_protects_close_endpoint_during_dec_acc_transition():
  accel = get_e2e_stop_approach_accel(
    2.6,
    make_model_msg(desired_accel=-1.19, endpoint_x=5.7),
    make_radar_state(),
    False,
    model_stop_protection_active=True,
  )

  assert -E2E_STOP_APPROACH_DECEL_MAX <= accel < -0.5


def test_e2e_stop_approach_protection_keeps_low_speed_floor():
  accel = get_e2e_stop_approach_accel(
    1.5,
    make_model_msg(desired_accel=-1.19, endpoint_x=4.0),
    make_radar_state(),
    False,
    model_stop_protection_active=True,
  )

  assert accel == 0.0


def test_e2e_stop_approach_leaves_hard_model_stop_to_model():
  accel = get_e2e_stop_approach_accel(12.0, make_model_msg(should_stop=True, endpoint_x=30.0), make_radar_state(), True)

  assert accel == 0.0


def test_e2e_close_stop_settle_holds_decel_at_route_like_stop_line():
  accel, should_stop, active = get_e2e_close_stop_settle(
    0.44,
    -0.26,
    make_model_msg(desired_accel=-0.26, positions=[0.0, 0.01, 20.0], velocities=[1.0, 0.2, 2.0]),
    make_radar_state(),
    True,
  )

  assert active
  assert should_stop
  assert -E2E_CLOSE_STOP_DECEL_MAX <= accel < -0.3


def test_e2e_close_stop_settle_keeps_stop_latch_below_rolling_speed():
  accel, should_stop, active = get_e2e_close_stop_settle(
    E2E_CLOSE_STOP_MIN_ROLLING_V - 0.01,
    -0.05,
    make_model_msg(desired_accel=-0.05, positions=[0.0, 0.1], velocities=[1.0, 0.0]),
    make_radar_state(),
    True,
    active=True,
  )

  assert accel == -0.05
  assert should_stop
  assert active


def test_e2e_close_stop_settle_requires_no_lead_e2e_and_no_override():
  model_msg = make_model_msg(desired_accel=-0.2, positions=[0.0, 0.2], velocities=[1.0, 0.0])

  assert get_e2e_close_stop_settle(0.5, -0.2, model_msg, make_radar_state(lead_one=True), True) == (-0.2, False, False)
  assert get_e2e_close_stop_settle(0.5, -0.2, model_msg, make_radar_state(), False) == (-0.2, False, False)
  assert get_e2e_close_stop_settle(0.5, -0.2, model_msg, make_radar_state(), True, brake_pressed=True) == (-0.2, False, False)
  assert get_e2e_close_stop_settle(0.5, -0.2, model_msg, make_radar_state(), True, gas_pressed=True) == (-0.2, False, False)


def test_e2e_close_stop_settle_ignores_positive_model_accel():
  model_msg = make_model_msg(desired_accel=0.01, positions=[0.0, 0.2], velocities=[1.0, 0.0])

  assert get_e2e_close_stop_settle(0.5, 0.01, model_msg, make_radar_state(), True) == (0.01, False, False)


def test_e2e_close_stop_settle_releases_after_stop_distance_clears():
  model_msg = make_model_msg(desired_accel=-0.2, positions=[0.0, 1.5], velocities=[1.0, 0.0])

  assert get_e2e_close_stop_settle(0.5, -0.2, model_msg, make_radar_state(), True, active=True) == (-0.2, False, False)


def test_e2e_runway_comfort_caps_long_runway_raw_model_braking():
  accel = get_e2e_runway_comfort_accel(
    v_ego=17.2,
    raw_e2e_accel=-1.2,
    coast_accel=-0.25,
    model_msg=make_model_msg(desired_accel=-1.2, should_stop=False, endpoint_x=145.0),
    e2e_active=True,
    prev_output_a_target=-0.2,
  )

  assert accel == -0.2175


def test_e2e_runway_comfort_prefers_coast_on_excessive_runway():
  accel = get_e2e_runway_comfort_accel(
    v_ego=17.2,
    raw_e2e_accel=-1.2,
    coast_accel=-0.25,
    model_msg=make_model_msg(desired_accel=-1.2, should_stop=False, endpoint_x=145.0),
    e2e_active=True,
    prev_output_a_target=-0.5,
  )

  assert math.isclose(accel, -0.30)


def test_e2e_runway_comfort_allows_short_runway_model_braking():
  accel = get_e2e_runway_comfort_accel(
    v_ego=17.2,
    raw_e2e_accel=-1.2,
    coast_accel=-0.25,
    model_msg=make_model_msg(desired_accel=-1.2, should_stop=False, endpoint_x=55.0),
    e2e_active=True,
    prev_output_a_target=-0.2,
  )

  assert accel == -1.2


def test_e2e_runway_comfort_leaves_model_stop_untouched():
  accel = get_e2e_runway_comfort_accel(
    v_ego=17.2,
    raw_e2e_accel=-1.2,
    coast_accel=-0.25,
    model_msg=make_model_msg(desired_accel=-1.2, should_stop=True, endpoint_x=145.0),
    e2e_active=True,
    prev_output_a_target=-0.2,
  )

  assert accel == -1.2


def test_e2e_runway_comfort_leaves_stop_context_bootstrap_untouched():
  accel = get_e2e_runway_comfort_accel(
    v_ego=17.2,
    raw_e2e_accel=-1.2,
    coast_accel=-0.25,
    model_msg=make_model_msg(
      desired_accel=-1.2,
      should_stop=False,
      endpoint_x=145.0,
      positions=[0.0, 35.0, 60.0],
      velocities=[17.0, 5.0, 0.5],
    ),
    e2e_active=True,
    prev_output_a_target=-0.2,
    engage_stop_bootstrap_active=True,
  )

  assert accel == -1.2


def test_e2e_runway_comfort_leaves_driver_override_untouched():
  model_msg = make_model_msg(desired_accel=-1.2, should_stop=False, endpoint_x=145.0)

  assert get_e2e_runway_comfort_accel(17.2, -1.2, -0.25, model_msg, True, -0.2, gas_pressed=True) == -1.2
  assert get_e2e_runway_comfort_accel(17.2, -1.2, -0.25, model_msg, True, -0.2, brake_pressed=True) == -1.2
  assert get_e2e_runway_comfort_accel(17.2, -1.2, -0.25, model_msg, True, -0.2, reset_state=True) == -1.2
  assert get_e2e_runway_comfort_accel(17.2, -1.2, -0.25, model_msg, True, -0.2, force_slow_decel=True) == -1.2


def test_e2e_runway_comfort_leaves_radar_lead_untouched():
  model_msg = make_model_msg(desired_accel=-1.2, should_stop=False, endpoint_x=145.0)

  accel = get_e2e_runway_comfort_accel(
    v_ego=17.2,
    raw_e2e_accel=-1.2,
    coast_accel=-0.25,
    model_msg=model_msg,
    e2e_active=True,
    prev_output_a_target=-0.2,
    has_radar_lead=True,
  )

  assert accel == -1.2


def test_e2e_runway_comfort_limits_negative_ramp():
  accel = get_e2e_runway_comfort_accel(
    v_ego=17.2,
    raw_e2e_accel=-0.8,
    coast_accel=-0.25,
    model_msg=make_model_msg(desired_accel=-0.8, should_stop=False, endpoint_x=90.0),
    e2e_active=True,
    prev_output_a_target=-0.2,
  )

  assert accel == -0.2175


def test_e2e_runway_comfort_does_not_block_stop_approach_shortage_braking():
  model_msg = make_model_msg(desired_accel=-0.4, should_stop=False, endpoint_x=45.0)
  governed = get_e2e_runway_comfort_accel(12.0, -0.4, -0.25, model_msg, True, -0.2)
  shortage_accel = get_e2e_stop_approach_accel(12.0, model_msg, make_radar_state(), True)

  assert shortage_accel < governed
  assert shortage_accel < -0.5


def test_e2e_runway_positive_accel_cap_limits_short_runway_at_crawl():
  cap = get_e2e_runway_positive_accel_cap(
    0.5,
    make_model_msg(desired_accel=1.0, should_stop=False, endpoint_x=2.0, positions=[0.0, 2.0], velocities=[0.5, 0.1]),
    True,
  )

  assert 0.0 < cap < 1.0


def test_e2e_runway_positive_accel_cap_limits_final_endpoint_crawl():
  cap = get_e2e_runway_positive_accel_cap(
    0.25,
    make_model_msg(desired_accel=0.5, should_stop=True, endpoint_x=0.2, positions=[0.0, 0.2], velocities=[0.2, 0.0]),
    True,
  )

  assert 0.0 < cap <= E2E_RUNWAY_FINAL_CRAWL_ACCEL_MAX


def test_e2e_runway_positive_accel_cap_caps_at_15m_crawl_example():
  cap = get_e2e_runway_positive_accel_cap(
    0.5,
    make_model_msg(desired_accel=0.0, should_stop=False, endpoint_x=15.0, positions=[0.0, 15.0], velocities=[0.5, 0.1]),
    True,
  )

  assert 0.0 < cap < get_max_accel(0.5)


def test_e2e_runway_positive_accel_cap_supports_model_stop_protection():
  cap = get_e2e_runway_positive_accel_cap(
    0.5,
    make_model_msg(desired_accel=0.0, should_stop=False, endpoint_x=15.0, positions=[0.0, 15.0], velocities=[0.5, 0.1]),
    False,
    model_stop_protection_active=True,
  )

  assert 0.0 < cap < get_max_accel(0.5)


def test_e2e_runway_positive_accel_cap_is_no_op_for_long_runway():
  cap = get_e2e_runway_positive_accel_cap(
    0.5,
    make_model_msg(desired_accel=1.0, should_stop=False, endpoint_x=40.0, positions=[0.0, 40.0], velocities=[0.5, 0.1]),
    True,
  )

  assert cap == ACCEL_MAX


def test_e2e_runway_positive_accel_cap_scales_with_runway_length():
  short_cap = get_e2e_runway_positive_accel_cap(
    0.5,
    make_model_msg(desired_accel=1.0, should_stop=False, endpoint_x=2.0, positions=[0.0, 2.0], velocities=[0.5, 0.1]),
    True,
  )
  mid_cap = get_e2e_runway_positive_accel_cap(
    0.5,
    make_model_msg(desired_accel=0.0, should_stop=False, endpoint_x=15.0, positions=[0.0, 15.0], velocities=[0.5, 0.1]),
    True,
  )
  long_cap = get_e2e_runway_positive_accel_cap(
    0.5,
    make_model_msg(desired_accel=1.0, should_stop=False, endpoint_x=40.0, positions=[0.0, 40.0], velocities=[0.5, 0.1]),
    True,
  )

  assert short_cap < mid_cap < long_cap == ACCEL_MAX


def test_e2e_runway_positive_accel_cap_disables_on_override_and_reset():
  model_msg = make_model_msg(desired_accel=1.0, should_stop=False, endpoint_x=2.0, positions=[0.0, 2.0], velocities=[0.5, 0.1])

  assert get_e2e_runway_positive_accel_cap(0.5, model_msg, True, brake_pressed=True) == ACCEL_MAX
  assert get_e2e_runway_positive_accel_cap(0.5, model_msg, True, gas_pressed=True) == ACCEL_MAX
  assert get_e2e_runway_positive_accel_cap(0.5, model_msg, True, reset_state=True) == ACCEL_MAX
  assert get_e2e_runway_positive_accel_cap(0.5, model_msg, True, force_slow_decel=True) == ACCEL_MAX
  assert get_e2e_runway_positive_accel_cap(0.5, model_msg, True, engage_stop_bootstrap_active=True) == ACCEL_MAX
  assert get_e2e_runway_positive_accel_cap(0.5, model_msg, True, has_radar_lead=True) == ACCEL_MAX
  assert get_e2e_runway_positive_accel_cap(0.5, model_msg, False) == ACCEL_MAX


def test_e2e_runway_positive_accel_cap_ignores_weak_model_signal_and_invalid_endpoint():
  weak_model_msg = make_model_msg(desired_accel=0.0, should_stop=False, endpoint_x=2.0, positions=[0.0, 2.0], velocities=[0.5, 2.0])
  invalid_endpoint_msg = make_model_msg(desired_accel=1.0, should_stop=False, endpoint_x=float('nan'), positions=[0.0, float('nan')], velocities=[0.5, 0.1])

  assert get_e2e_runway_positive_accel_cap(0.5, weak_model_msg, True) == ACCEL_MAX
  assert get_e2e_runway_positive_accel_cap(0.5, invalid_endpoint_msg, True) == ACCEL_MAX
