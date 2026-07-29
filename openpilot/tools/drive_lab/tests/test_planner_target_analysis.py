import pytest

from openpilot.tools.drive_lab.planner_target_analysis import (
  PlannerTargetSample,
  build_suspicious_episodes,
  high_plan_jerk_pairs,
  is_opposite_intent,
  is_strong_opposite_intent,
)


def sample(t, plan_a, a_ego, route="route-a", segment=None, gas=False, brake=False,
           source="cruise", lead=False, d_rel=None, v_rel=None, should_stop=False, plan_t=None):
  return PlannerTargetSample(
    route=route,
    route_id=route,
    segment=segment,
    t=t,
    v_ego=8.0,
    a_ego=a_ego,
    gas_pressed=gas,
    brake_pressed=brake,
    standstill=False,
    selfdrive_enabled=False,
    selfdrive_active=False,
    long_active=False,
    long_control_state="pid",
    v_cruise_kph=80.0,
    plan_a_target=plan_a,
    plan_source=source,
    plan_should_stop=should_stop,
    plan_fcw=False,
    sp_a_target=plan_a,
    sp_source=source,
    sp_stack="sunnypilotCurrent",
    lead_status=lead,
    lead_d_rel=d_rel,
    lead_v_rel=v_rel,
    model_desired_accel=None,
    model_should_stop=False,
    plan_time_s=plan_t,
  )


def test_opposite_intent_helpers_are_independent_of_cli_modules():
  hard_brake_driver_gas = sample(0.0, -1.4, 0.2, gas=True, source="lead0")
  accel_driver_brake = sample(1.0, 0.9, -0.7, brake=True)
  neutral_driver_brake = sample(2.0, 0.1, -0.7, brake=True)

  assert is_opposite_intent(hard_brake_driver_gas)
  assert is_strong_opposite_intent(hard_brake_driver_gas)
  assert is_opposite_intent(accel_driver_brake)
  assert is_strong_opposite_intent(accel_driver_brake)
  assert not is_opposite_intent(neutral_driver_brake)


def test_suspicious_episodes_and_jerk_group_by_route_segment():
  samples = [
    sample(0.0, 0.2, 0.2, lead=False, source="cruise"),
    sample(0.1, 0.3, 0.3, lead=True, d_rel=8.0, v_rel=-0.2, source="cruise"),
    sample(0.2, -1.4, 0.1, gas=True, lead=True, d_rel=7.5, v_rel=-0.4, source="lead0"),
    sample(0.3, -1.3, 0.1, gas=True, lead=True, d_rel=7.2, v_rel=-0.5, source="lead0"),
    sample(0.9, 0.2, 0.2, lead=False, source="cruise"),
  ]

  episodes = build_suspicious_episodes(samples, large_error_threshold=1.2, episode_gap_s=0.6, context_s=1.0, high_jerk_threshold=8.0)

  assert len(episodes) == 1
  assert episodes[0].sample_count == 2
  assert episodes[0].opposite_count == 2
  assert episodes[0].strong_opposite_count == 2
  assert episodes[0].lead_status_flips >= 2
  assert episodes[0].plan_source_flips >= 2
  assert episodes[0].plan_span == pytest.approx(1.7)
  assert episodes[0].duration_s == pytest.approx(0.1)
  assert len(high_plan_jerk_pairs(samples, threshold=8.0)) >= 1


def test_plan_jerk_uses_planner_publish_cadence_not_car_state_cadence():
  samples = [
    sample(0.001, 0.0, 0.0, plan_t=0.0),
    sample(0.011, 0.0, 0.0, plan_t=0.0),
    sample(0.051, 0.5, 0.0, plan_t=0.05),
    sample(0.061, 0.5, 0.0, plan_t=0.05),
  ]

  pairs = high_plan_jerk_pairs(samples, threshold=8.0)

  assert len(pairs) == 1
  assert pairs[0][2] == pytest.approx(10.0)
