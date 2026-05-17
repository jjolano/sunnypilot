from types import SimpleNamespace
import math

import numpy as np
import pytest
from opendbc.car.car_helpers import interfaces
from opendbc.car.interfaces import ACCEL_MAX
from opendbc.car.toyota.values import CAR as TOYOTA
from opendbc.car.vehicle_model import VehicleModel
from openpilot.common.constants import CV
from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N
from openpilot.selfdrive.controls.lib.longitudinal_stacks.custom_v1 import CUSTOM_V1_FLOOR

from openpilot.selfdrive.controls.lib.longitudinal_planner import (
  build_custom_v1_accel_candidate,
  E2E_CLOSE_STOP_DECEL_MAX,
  E2E_CLOSE_STOP_MIN_ROLLING_V,
  E2E_STOP_APPROACH_DECEL_MAX,
  E2E_RUNWAY_FINAL_CRAWL_ACCEL_MAX,
  LongitudinalPlanner,
  _A_TOTAL_MAX_BP,
  _A_TOTAL_MAX_V,
  get_e2e_close_stop_settle,
  get_max_accel,
  get_e2e_runway_comfort_accel,
  get_e2e_runway_positive_accel_cap,
  get_e2e_stop_approach_accel,
  get_model_stop_distance,
  has_model_stop_context,
  has_valid_radar_lead,
  limit_accel_in_turns,
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


def get_test_cp():
  return interfaces[TOYOTA.TOYOTA_COROLLA_TSS2].get_non_essential_params(TOYOTA.TOYOTA_COROLLA_TSS2)


class FakeSubMaster(dict):
  logMonoTime = {'modelV2': 1.0}

  def all_checks(self, service_list):
    return True


def test_has_valid_radar_lead_checks_both_tracks():
  assert not has_valid_radar_lead(make_radar_state())
  assert has_valid_radar_lead(make_radar_state(lead_one=True))
  assert has_valid_radar_lead(make_radar_state(lead_two=True))


def test_limit_accel_in_turns_defaults_to_legacy_kinematic_calculation():
  CP = get_test_cp()
  v_ego = 30.0
  angle_steers = 5.0
  a_target = [-1.0, 1.2]

  limited = limit_accel_in_turns(v_ego, angle_steers, a_target, CP)

  a_total_max = np.interp(v_ego, _A_TOTAL_MAX_BP, _A_TOTAL_MAX_V)
  legacy_a_y = v_ego**2 * angle_steers * CV.DEG_TO_RAD / (CP.steerRatio * CP.wheelbase)
  expected_a_x_allowed = math.sqrt(max(a_total_max**2 - legacy_a_y**2, 0.0))
  assert limited == pytest.approx([a_target[0], min(a_target[1], expected_a_x_allowed)])


def test_limit_accel_in_turns_hardening_uses_vehicle_model_curvature():
  CP = get_test_cp()
  v_ego = 30.0
  angle_steers = 5.0
  a_target = [-1.0, 1.2]
  VM = VehicleModel(CP)

  try:
    limited = limit_accel_in_turns(v_ego, angle_steers, a_target, CP, control_calculation_hardening=True)
  except TypeError as exc:
    pytest.fail(f"limit_accel_in_turns rejected hardening toggle: {exc!r}")

  a_total_max = np.interp(v_ego, _A_TOTAL_MAX_BP, _A_TOTAL_MAX_V)
  vehicle_model_a_y = v_ego**2 * VM.calc_curvature(angle_steers * CV.DEG_TO_RAD, v_ego, 0.0)
  expected_a_x_allowed = math.sqrt(max(a_total_max**2 - vehicle_model_a_y**2, 0.0))
  assert limited == pytest.approx([a_target[0], min(a_target[1], expected_a_x_allowed)])


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


def test_engage_stop_bootstrap_custom_candidate_does_not_mutate_baseline_output():
  planner = SimpleNamespace(
    output_a_target=0.1,
    output_should_stop=False,
    allow_throttle=True,
    fcw=False,
    source="cruise",
    mpc=SimpleNamespace(source="cruise"),
    v_desired_trajectory=tuple(10.0 for _ in range(CONTROL_N)),
    a_desired_trajectory=tuple(0.1 for _ in range(CONTROL_N)),
    j_desired_trajectory=tuple(0.0 for _ in range(CONTROL_N)),
  )

  candidate = build_custom_v1_accel_candidate(
    planner, "engage_stop_bootstrap", -1.2, has_lead=False, reason="engage_model_stop_bootstrap",
    accel_limits=(-2.0, 2.0), should_stop=True,
  )

  assert candidate is not None
  assert candidate.name == "engage_stop_bootstrap"
  assert candidate.output.a_target == pytest.approx(-1.2)
  assert candidate.output.should_stop
  assert candidate.output.debug["custom_v1_candidate_reason"] == "engage_model_stop_bootstrap"
  assert planner.output_a_target == pytest.approx(0.1)
  assert not planner.output_should_stop


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


def test_e2e_stop_approach_custom_candidate_does_not_mutate_baseline_output():
  planner = SimpleNamespace(
    output_a_target=-0.2,
    output_should_stop=False,
    allow_throttle=True,
    fcw=False,
    source="cruise",
    mpc=SimpleNamespace(source="cruise"),
    v_desired_trajectory=tuple(10.0 for _ in range(CONTROL_N)),
    a_desired_trajectory=tuple(-0.2 for _ in range(CONTROL_N)),
    j_desired_trajectory=tuple(0.0 for _ in range(CONTROL_N)),
  )

  candidate = build_custom_v1_accel_candidate(
    planner, "e2e_stop_approach", -0.8, has_lead=False, reason="no_lead_model_stop_approach", accel_limits=(-2.0, 2.0),
  )

  assert candidate is not None
  assert candidate.name == "e2e_stop_approach"
  assert candidate.output.a_target == pytest.approx(-0.8)
  assert candidate.output.debug["custom_v1_candidate_reason"] == "no_lead_model_stop_approach"
  assert planner.output_a_target == pytest.approx(-0.2)


def test_custom_v1_accel_candidate_skips_non_restrictive_target():
  planner = SimpleNamespace(output_a_target=-0.2)

  candidate = build_custom_v1_accel_candidate(
    planner, "e2e_stop_approach", -0.1, has_lead=False, reason="no_lead_model_stop_approach", accel_limits=(-2.0, 2.0),
  )

  assert candidate is None


def test_custom_v1_accel_floor_candidate_can_relax_baseline_output():
  planner = SimpleNamespace(
    output_a_target=-1.0,
    output_should_stop=False,
    allow_throttle=True,
    fcw=False,
    source="cruise",
    mpc=SimpleNamespace(source="cruise"),
    v_desired_trajectory=tuple(10.0 for _ in range(CONTROL_N)),
    a_desired_trajectory=tuple(-1.0 for _ in range(CONTROL_N)),
    j_desired_trajectory=tuple(0.0 for _ in range(CONTROL_N)),
  )

  candidate = build_custom_v1_accel_candidate(
    planner, "cruise_coast", -0.3, has_lead=False, reason="plain_cruise_overspeed_coast",
    accel_limits=(-2.0, 2.0), selection=CUSTOM_V1_FLOOR,
  )

  assert candidate is not None
  assert candidate.name == "cruise_coast"
  assert candidate.selection == CUSTOM_V1_FLOOR
  assert candidate.output.a_target == pytest.approx(-0.3)
  assert planner.output_a_target == pytest.approx(-1.0)


def test_custom_v1_accel_floor_candidate_skips_non_relaxing_target():
  planner = SimpleNamespace(output_a_target=-0.2, output_should_stop=False)

  candidate = build_custom_v1_accel_candidate(
    planner, "cruise_coast", -0.4, has_lead=False, reason="plain_cruise_overspeed_coast",
    accel_limits=(-2.0, 2.0), selection=CUSTOM_V1_FLOOR,
  )

  assert candidate is None


def test_custom_v1_accel_candidate_force_keeps_cap_available_for_floor_conflicts():
  planner = SimpleNamespace(
    output_a_target=-0.2,
    output_should_stop=False,
    allow_throttle=True,
    fcw=False,
    source="cruise",
    mpc=SimpleNamespace(source="lead0"),
    v_desired_trajectory=tuple(10.0 for _ in range(CONTROL_N)),
    a_desired_trajectory=tuple(-0.2 for _ in range(CONTROL_N)),
    j_desired_trajectory=tuple(0.0 for _ in range(CONTROL_N)),
  )

  candidate = build_custom_v1_accel_candidate(
    planner, "creep_to_stop_gap_accel_cap", 0.18, has_lead=True,
    reason="creep_to_stop_gap_accel_cap", accel_limits=(-2.0, 2.0), force=True,
  )

  assert candidate is not None
  assert candidate.output.a_target == pytest.approx(0.18)
  assert planner.output_a_target == pytest.approx(-0.2)


def test_custom_v1_accel_candidate_can_carry_stop_intent_without_accel_delta():
  planner = SimpleNamespace(
    output_a_target=-0.2,
    output_should_stop=False,
    allow_throttle=True,
    fcw=False,
    source="cruise",
    mpc=SimpleNamespace(source="cruise"),
    v_desired_trajectory=tuple(10.0 for _ in range(CONTROL_N)),
    a_desired_trajectory=tuple(-0.2 for _ in range(CONTROL_N)),
    j_desired_trajectory=tuple(0.0 for _ in range(CONTROL_N)),
  )

  candidate = build_custom_v1_accel_candidate(
    planner, "e2e_close_stop_settle", -0.2, has_lead=False, reason="no_lead_close_stop_settle",
    accel_limits=(-2.0, 2.0), should_stop=True,
  )

  assert candidate is not None
  assert candidate.output.a_target == pytest.approx(-0.2)
  assert candidate.output.should_stop
  assert not planner.output_should_stop


def test_moving_lead_stop_gap_guard_custom_candidate_does_not_mutate_baseline_output():
  planner = SimpleNamespace(
    output_a_target=0.1,
    output_should_stop=False,
    allow_throttle=True,
    fcw=False,
    source="cruise",
    mpc=SimpleNamespace(source="lead0"),
    v_desired_trajectory=tuple(10.0 for _ in range(CONTROL_N)),
    a_desired_trajectory=tuple(0.1 for _ in range(CONTROL_N)),
    j_desired_trajectory=tuple(0.0 for _ in range(CONTROL_N)),
  )

  candidate = build_custom_v1_accel_candidate(
    planner, "moving_lead_stop_gap_guard", -0.7, has_lead=True, reason="moving_lead_stop_gap_guard", accel_limits=(-2.0, 2.0),
  )

  assert candidate is not None
  assert candidate.name == "moving_lead_stop_gap_guard"
  assert candidate.output.a_target == pytest.approx(-0.7)
  assert candidate.output.has_lead
  assert candidate.output.debug["custom_v1_candidate_reason"] == "moving_lead_stop_gap_guard"
  assert planner.output_a_target == pytest.approx(0.1)


def test_stopped_lead_stop_gap_guard_custom_candidate_carries_stop_intent_without_mutating_baseline():
  planner = SimpleNamespace(
    output_a_target=-0.1,
    output_should_stop=False,
    allow_throttle=True,
    fcw=False,
    source="cruise",
    mpc=SimpleNamespace(source="lead0"),
    v_desired_trajectory=tuple(10.0 for _ in range(CONTROL_N)),
    a_desired_trajectory=tuple(-0.1 for _ in range(CONTROL_N)),
    j_desired_trajectory=tuple(0.0 for _ in range(CONTROL_N)),
  )

  candidate = build_custom_v1_accel_candidate(
    planner, "stopped_lead_stop_gap_guard", -0.8, has_lead=True, reason="stopped_lead_stop_gap_guard",
    accel_limits=(-2.0, 2.0), should_stop=True,
  )

  assert candidate is not None
  assert candidate.name == "stopped_lead_stop_gap_guard"
  assert candidate.output.a_target == pytest.approx(-0.8)
  assert candidate.output.should_stop
  assert candidate.output.has_lead
  assert candidate.output.debug["custom_v1_candidate_reason"] == "stopped_lead_stop_gap_guard"
  assert planner.output_a_target == pytest.approx(-0.1)
  assert not planner.output_should_stop


def test_stopped_lead_creep_hold_custom_candidate_carries_stop_intent_without_mutating_baseline():
  planner = SimpleNamespace(
    output_a_target=0.0,
    output_should_stop=False,
    allow_throttle=True,
    fcw=False,
    source="cruise",
    mpc=SimpleNamespace(source="lead0"),
    v_desired_trajectory=tuple(10.0 for _ in range(CONTROL_N)),
    a_desired_trajectory=tuple(0.0 for _ in range(CONTROL_N)),
    j_desired_trajectory=tuple(0.0 for _ in range(CONTROL_N)),
  )

  candidate = build_custom_v1_accel_candidate(
    planner, "stopped_lead_creep_hold", -0.25, has_lead=True, reason="stopped_lead_creep_hold",
    accel_limits=(-2.0, 2.0), should_stop=True,
  )

  assert candidate is not None
  assert candidate.name == "stopped_lead_creep_hold"
  assert candidate.output.a_target == pytest.approx(-0.25)
  assert candidate.output.should_stop
  assert candidate.output.has_lead
  assert candidate.output.debug["custom_v1_candidate_reason"] == "stopped_lead_creep_hold"
  assert planner.output_a_target == pytest.approx(0.0)
  assert not planner.output_should_stop


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


def test_e2e_close_stop_settle_custom_candidate_does_not_mutate_baseline_output():
  planner = SimpleNamespace(
    output_a_target=-0.05,
    output_should_stop=False,
    allow_throttle=True,
    fcw=False,
    source="cruise",
    mpc=SimpleNamespace(source="cruise"),
    v_desired_trajectory=tuple(10.0 for _ in range(CONTROL_N)),
    a_desired_trajectory=tuple(-0.05 for _ in range(CONTROL_N)),
    j_desired_trajectory=tuple(0.0 for _ in range(CONTROL_N)),
  )
  accel, should_stop, _active = get_e2e_close_stop_settle(
    0.44,
    -0.26,
    make_model_msg(desired_accel=-0.26, positions=[0.0, 0.01, 20.0], velocities=[1.0, 0.2, 2.0]),
    make_radar_state(),
    True,
  )

  candidate = build_custom_v1_accel_candidate(
    planner, "e2e_close_stop_settle", accel, has_lead=False, reason="no_lead_close_stop_settle",
    accel_limits=(-2.0, 2.0), should_stop=should_stop,
  )

  assert candidate is not None
  assert candidate.name == "e2e_close_stop_settle"
  assert candidate.output.a_target < planner.output_a_target
  assert candidate.output.should_stop
  assert planner.output_a_target == pytest.approx(-0.05)
  assert not planner.output_should_stop


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


def test_e2e_runway_comfort_caps_route_like_far_no_stop_decel():
  accel = get_e2e_runway_comfort_accel(
    v_ego=17.26,
    raw_e2e_accel=-0.88,
    coast_accel=-0.49,
    model_msg=make_model_msg(desired_accel=-0.88, should_stop=False, endpoint_x=101.7),
    e2e_active=True,
    prev_output_a_target=-0.52,
  )

  assert math.isclose(accel, -0.51)


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


def test_e2e_runway_positive_cap_custom_candidate_does_not_mutate_baseline_output():
  planner = SimpleNamespace(
    output_a_target=0.5,
    output_should_stop=False,
    allow_throttle=True,
    fcw=False,
    source="cruise",
    mpc=SimpleNamespace(source="cruise"),
    v_desired_trajectory=tuple(10.0 for _ in range(CONTROL_N)),
    a_desired_trajectory=tuple(0.5 for _ in range(CONTROL_N)),
    j_desired_trajectory=tuple(0.0 for _ in range(CONTROL_N)),
  )

  candidate = build_custom_v1_accel_candidate(
    planner, "e2e_runway_positive_cap", 0.1, has_lead=False,
    reason="low_speed_model_runway_positive_cap", accel_limits=(-2.0, 2.0),
  )

  assert candidate is not None
  assert candidate.name == "e2e_runway_positive_cap"
  assert candidate.output.a_target == pytest.approx(0.1)
  assert candidate.output.debug["custom_v1_candidate_reason"] == "low_speed_model_runway_positive_cap"
  assert planner.output_a_target == pytest.approx(0.5)


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
