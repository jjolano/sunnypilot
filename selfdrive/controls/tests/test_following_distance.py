import numpy as np
import pytest
import itertools
from opendbc.car.interfaces import ACCEL_MIN
from openpilot.common.parameterized import parameterized_class

from cereal import log

from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import (
  APPROACH_BRAKE,
  APPROACH_BRAKE_MIN,
  APPROACH_ENGAGE_OFFSET_MAX,
  APPROACH_MIN_GAP_BUFFER,
  COMFORT_BRAKE,
  LEAD_DEPARTURE_RELAXATION_MAX,
  LEAD_ACCEL_MATCH_MIN_POSITIVE_BLEND,
  LEAD_ACCEL_MATCH_DECEL_CAP,
  LEAD_ACCEL_MATCH_DECEL_TARGET_BLEND,
  LEAD_ACCEL_RECOVERY_ACCEL_MAX,
  MOVING_LEAD_STOP_RESERVE_MAX,
  LEAD_STOP_GAP_EXCESS_OFFSET_MAX,
  LEAD_STOP_GAP_TAPER_MAX,
  LEAD_GAP_COMFORT_LIGHT_DECEL,
  LEAD_STOP_RUNWAY_BRAKE,
  STOP_DISTANCE,
  STOP_DISTANCE_FADE_V,
  STOP_DISTANCE_MIN,
  STOPPED_LEAD_BUFFER,
  get_approach_available_runway,
  get_approach_brake,
  get_approach_engage_offset,
  get_approach_follow_distance,
  get_approach_runway_blend,
  get_desired_follow_distance,
  get_lead_accel_match_margin,
  get_lead_accel_match_target,
  get_lead_accel_recovery_a_min,
  get_lead_gap_comfort_a_min,
  get_lead_gap_comfort_floor,
  get_lead_gap_comfort_recovery_blend,
  get_lead_departure_available_runway,
  get_lead_departure_relaxation,
  get_lead_danger_distance,
  get_lead_stop_runway_blend,
  get_lead_stop_runway_gap,
  get_lead_stop_gap_excess_offset,
  get_lead_stop_gap_taper,
  get_lead_time_gap_target,
  get_moving_lead_stop_reserve,
  get_safe_obstacle_distance,
  get_stopped_lead_buffer,
  get_stopped_equivalence_factor,
  get_T_FOLLOW,
)
from openpilot.selfdrive.controls.lib.longitudinal_planner import (
  CREEP_TO_STOP_GAP_ACCEL_MAX,
  CREEP_TO_STOP_GAP_ACCEL_MIN,
  CREEP_TO_STOP_GAP_HOLD_EXCESS,
  CREEP_TO_STOP_GAP_MAX_EXCESS,
  CREEP_TO_STOP_GAP_PULLAWAY_ACCEL_MAX,
  CREEP_TO_STOP_GAP_PULLAWAY_ACCEL_MIN,
  CREEP_TO_STOP_GAP_PREDICT_MIN_GAP_OPENING,
  get_creep_to_stop_gap_accel,
  get_predicted_lead_pullaway,
  should_hold_creep_to_stop_gap,
)
from openpilot.selfdrive.test.longitudinal_maneuvers.maneuver import Maneuver


def stop_distance_buffer(v_ego):
  fade = (STOP_DISTANCE_FADE_V**2) / (v_ego**2 + STOP_DISTANCE_FADE_V**2)
  return STOP_DISTANCE_MIN + (STOP_DISTANCE - STOP_DISTANCE_MIN) * fade


@pytest.mark.parametrize("speed", [0.0, 5.0, 10.0, 35.0])
def test_safe_obstacle_distance_matches_explicit_formula(speed):
  t_follow = get_T_FOLLOW(log.LongitudinalPersonality.standard)
  expected = (speed**2) / (2 * COMFORT_BRAKE) + t_follow * speed + stop_distance_buffer(speed)
  assert get_safe_obstacle_distance(speed, t_follow) == pytest.approx(expected)


@pytest.mark.parametrize("speed", [0.0, 5.0, 10.0, 35.0])
def test_desired_follow_distance_matches_explicit_formula(speed):
  t_follow = get_T_FOLLOW(log.LongitudinalPersonality.standard)
  expected = (speed**2) / (2 * COMFORT_BRAKE) + t_follow * speed + stop_distance_buffer(speed) - get_stopped_equivalence_factor(speed)
  assert get_desired_follow_distance(speed, speed, t_follow) == pytest.approx(expected)


def test_stop_distance_buffer_fades_with_speed():
  buffer_speeds = [0.0, 5.0, 10.0, 35.0]
  buffers = [stop_distance_buffer(speed) for speed in buffer_speeds]
  assert buffers[0] == pytest.approx(STOP_DISTANCE)
  assert buffers[0] > buffers[1] > buffers[2] > buffers[3] > STOP_DISTANCE_MIN - 1e-6


def test_stopped_lead_buffer_only_applies_near_stop():
  assert get_stopped_lead_buffer(0.0, 0.0) == pytest.approx(0.0)
  assert get_stopped_lead_buffer(1.0, 0.0) == pytest.approx(STOPPED_LEAD_BUFFER)
  assert get_stopped_lead_buffer(1.0, 3.0) == pytest.approx(0.0)


def test_lead_stop_gap_taper_only_applies_for_low_speed_moving_leads():
  assert get_lead_stop_gap_taper(0.0, 0.0) == pytest.approx(0.0)
  assert 0.0 < get_lead_stop_gap_taper(0.0, 0.8) < LEAD_STOP_GAP_TAPER_MAX
  assert get_lead_stop_gap_taper(0.0, 2.0) == pytest.approx(LEAD_STOP_GAP_TAPER_MAX)
  assert get_lead_stop_gap_taper(1.5, 2.0) == pytest.approx(0.0)


def test_lead_stop_gap_excess_offset_requires_extra_runway():
  assert get_lead_stop_gap_excess_offset(0.0, STOP_DISTANCE + 0.9) == pytest.approx(0.0)
  assert 0.0 < get_lead_stop_gap_excess_offset(0.0, STOP_DISTANCE + 2.0) < LEAD_STOP_GAP_EXCESS_OFFSET_MAX
  assert get_lead_stop_gap_excess_offset(0.0, STOP_DISTANCE + 5.0) == pytest.approx(LEAD_STOP_GAP_EXCESS_OFFSET_MAX)
  assert get_lead_stop_gap_excess_offset(1.5, STOP_DISTANCE + 5.0) == pytest.approx(0.0)


def test_creep_to_stop_gap_release_arms_and_tapers_to_target_gap():
  active, accel = get_creep_to_stop_gap_accel(0.0, STOP_DISTANCE + 1.0, 0.0, 1.0, False)
  assert active
  assert 0.0 < accel <= CREEP_TO_STOP_GAP_ACCEL_MAX

  active, accel = get_creep_to_stop_gap_accel(0.0, STOP_DISTANCE + 0.5, 0.0, 1.0, active)
  assert active
  assert 0.0 < accel < CREEP_TO_STOP_GAP_ACCEL_MAX

  active, accel = get_creep_to_stop_gap_accel(0.2, STOP_DISTANCE + 0.3, 0.0, 1.0, active)
  assert active
  assert CREEP_TO_STOP_GAP_ACCEL_MIN < accel < 0.0

  active, accel = get_creep_to_stop_gap_accel(0.0, STOP_DISTANCE, 0.0, 1.0, active)
  assert not active
  assert accel == pytest.approx(0.0)


def test_creep_to_stop_gap_uses_stronger_accel_for_confirmed_pullaway():
  active, accel = get_creep_to_stop_gap_accel(0.0, STOP_DISTANCE + 0.6, 0.8, 1.0, False)

  assert active
  assert CREEP_TO_STOP_GAP_ACCEL_MAX < accel <= CREEP_TO_STOP_GAP_PULLAWAY_ACCEL_MAX


def test_creep_to_stop_gap_uses_predicted_pullaway_before_speed_threshold():
  active, accel = get_creep_to_stop_gap_accel(
    0.0, STOP_DISTANCE + 0.35, 0.05, 1.0, False, a_lead=1.0, a_lead_tau=0.0
  )

  assert active
  assert CREEP_TO_STOP_GAP_PULLAWAY_ACCEL_MIN <= accel <= CREEP_TO_STOP_GAP_PULLAWAY_ACCEL_MAX


def test_creep_to_stop_gap_prediction_requires_clear_gap_opening():
  predicted_v_lead, predicted_gap_opening = get_predicted_lead_pullaway(0.0, 0.1, 0.0)
  active, _ = get_creep_to_stop_gap_accel(0.0, STOP_DISTANCE + 0.35, 0.0, 1.0, False, a_lead=0.1, a_lead_tau=0.0)

  assert predicted_v_lead > 0.0
  assert predicted_gap_opening < CREEP_TO_STOP_GAP_PREDICT_MIN_GAP_OPENING
  assert not active


def test_creep_to_stop_gap_release_requires_confirmed_safe_lead():
  assert not get_creep_to_stop_gap_accel(0.0, STOP_DISTANCE + 1.0, 0.0, 0.4, False)[0]
  assert not get_creep_to_stop_gap_accel(0.0, STOP_DISTANCE + 1.0, -0.4, 1.0, False)[0]
  assert not get_creep_to_stop_gap_accel(0.0, STOP_DISTANCE + 1.0, 0.0, 1.0, False, brake_pressed=True)[0]
  assert not get_creep_to_stop_gap_accel(0.0, STOP_DISTANCE + 1.0, 0.0, 1.0, False, gas_pressed=True)[0]
  assert not get_creep_to_stop_gap_accel(0.4, STOP_DISTANCE + 1.0, 0.0, 1.0, False)[0]
  assert not get_creep_to_stop_gap_accel(0.0, STOP_DISTANCE + CREEP_TO_STOP_GAP_MAX_EXCESS + 0.1, 0.0, 1.0, False)[0]
  assert not get_creep_to_stop_gap_accel(0.0, STOP_DISTANCE, 0.0, 1.0, False, a_lead=1.0, a_lead_tau=0.0)[0]


def test_creep_to_stop_gap_hold_covers_near_target_gap_only():
  assert should_hold_creep_to_stop_gap(0.1, STOP_DISTANCE + 0.3, 0.0, 0.0)
  assert not should_hold_creep_to_stop_gap(0.1, STOP_DISTANCE + CREEP_TO_STOP_GAP_HOLD_EXCESS + 0.1, 0.0, 0.0)
  assert not should_hold_creep_to_stop_gap(0.1, STOP_DISTANCE + 0.3, 0.3, 0.0)
  assert not should_hold_creep_to_stop_gap(0.1, STOP_DISTANCE + 0.3, 0.0, 0.1)


def test_lead_departure_relaxation_requires_gap_growth_and_pullaway():
  assert get_lead_departure_relaxation(0.0, 0.5, 1.0) == pytest.approx(0.0)
  assert get_lead_departure_relaxation(0.0, 1.5, 0.2) == pytest.approx(0.0)
  assert get_lead_departure_relaxation(0.6, 0.7, 1.0) == pytest.approx(0.0)


def test_lead_departure_runway_uses_extra_stopped_gap():
  assert get_lead_departure_available_runway(0.0, STOP_DISTANCE - 0.1, 0.0) == pytest.approx(0.0)
  assert get_lead_departure_available_runway(0.0, STOP_DISTANCE + 0.6, 0.0) == pytest.approx(0.6)


def test_lead_departure_relaxation_can_use_extra_stopped_gap():
  runway = get_lead_departure_available_runway(0.0, STOP_DISTANCE + 0.6, 0.0)

  assert get_lead_departure_relaxation(0.0, 0.5, runway) == pytest.approx(0.0)
  assert 0.0 < get_lead_departure_relaxation(0.0, 0.9, runway) < LEAD_DEPARTURE_RELAXATION_MAX


def test_lead_departure_relaxation_grows_with_confirmed_departure():
  mild_departure = get_lead_departure_relaxation(0.0, 0.9, 0.6)
  strong_departure = get_lead_departure_relaxation(0.0, 2.0, 1.0)

  assert 0.0 < mild_departure < strong_departure <= LEAD_DEPARTURE_RELAXATION_MAX


def test_lead_departure_relaxation_fades_out_as_ego_starts_creeping():
  stopped_relaxation = get_lead_departure_relaxation(0.0, 2.0, 1.0)
  creeping_relaxation = get_lead_departure_relaxation(0.8, 2.0, 1.0)

  assert 0.0 < creeping_relaxation < stopped_relaxation
  assert get_lead_departure_relaxation(1.0, 2.0, 1.0) == pytest.approx(0.0)


def test_approach_brake_stays_stock_for_small_closure():
  assert get_approach_brake(0.0) == pytest.approx(APPROACH_BRAKE)
  assert get_approach_brake(1.5) == pytest.approx(APPROACH_BRAKE)


def test_approach_brake_ramps_down_for_stronger_closure():
  moderate_closure_brake = get_approach_brake(3.0)
  strong_closure_brake = get_approach_brake(5.0)

  assert APPROACH_BRAKE_MIN <= strong_closure_brake < moderate_closure_brake < APPROACH_BRAKE
  assert strong_closure_brake == pytest.approx(APPROACH_BRAKE_MIN)


@pytest.mark.parametrize("speed", [0.0, 5.0, 10.0, 35.0])
def test_approach_follow_distance_matches_steady_state_when_speeds_match(speed):
  t_follow = get_T_FOLLOW(log.LongitudinalPersonality.standard)
  x_lead = get_desired_follow_distance(speed, speed, t_follow)
  assert get_approach_follow_distance(x_lead, speed, speed, t_follow) == pytest.approx(get_desired_follow_distance(speed, speed, t_follow))


def test_approach_follow_distance_uses_runway_before_danger_zone():
  t_follow = get_T_FOLLOW(log.LongitudinalPersonality.standard)
  v_ego = 25.0
  v_lead = 20.0
  closing_speed = v_ego - v_lead
  x_lead = get_desired_follow_distance(v_ego, v_lead, t_follow) + 25.0

  approach_gap = get_approach_follow_distance(x_lead, v_ego, v_lead, t_follow)
  expected_gap = max(
    t_follow * v_lead + stop_distance_buffer(v_lead) + (closing_speed**2) / (2 * get_approach_brake(closing_speed)),
    get_lead_danger_distance(v_ego, v_lead, t_follow) + APPROACH_MIN_GAP_BUFFER,
  )

  assert approach_gap == pytest.approx(expected_gap)
  assert get_lead_danger_distance(v_ego, v_lead, t_follow) < approach_gap < get_desired_follow_distance(v_ego, v_lead, t_follow)


def test_approach_runway_blend_stays_off_near_legacy_gap():
  t_follow = get_T_FOLLOW(log.LongitudinalPersonality.standard)
  v_ego = 20.0
  v_lead = 20.0
  x_lead = get_desired_follow_distance(v_ego, v_lead, t_follow) + 1.0

  assert get_approach_runway_blend(x_lead, v_ego, v_lead, t_follow) == pytest.approx(0.0)


def test_approach_runway_blend_reaches_full_with_large_runway():
  t_follow = get_T_FOLLOW(log.LongitudinalPersonality.standard)
  v_ego = 25.0
  v_lead = 20.0
  x_lead = get_desired_follow_distance(v_ego, v_lead, t_follow) + 25.0

  assert get_approach_runway_blend(x_lead, v_ego, v_lead, t_follow) == pytest.approx(1.0)


def test_approach_available_runway_uses_stop_distance_for_slowing_lead():
  t_follow = get_T_FOLLOW(log.LongitudinalPersonality.standard)
  steady_runway = get_approach_available_runway(40.0, 20.0, 15.0, t_follow, a_lead=0.0)
  slowing_runway = get_approach_available_runway(40.0, 20.0, 15.0, t_follow, a_lead=-1.0)
  reserve = get_moving_lead_stop_reserve(20.0, 15.0, 5.0, -1.0)

  assert steady_runway == pytest.approx(max(40.0 - get_desired_follow_distance(20.0, 15.0, t_follow), 0.0))
  assert slowing_runway == pytest.approx(40.0 + get_stopped_equivalence_factor(15.0) - STOP_DISTANCE - reserve)
  assert slowing_runway > steady_runway


def test_approach_available_runway_uses_fixed_stop_gap_for_slowing_lead():
  t_follow = get_T_FOLLOW(log.LongitudinalPersonality.standard)
  v_ego = 25.0
  v_lead = 15.0
  x_lead = 35.0

  slowing_runway = get_approach_available_runway(x_lead, v_ego, v_lead, t_follow, a_lead=-1.0)
  reserve = get_moving_lead_stop_reserve(v_ego, v_lead, v_ego - v_lead, -1.0)

  assert slowing_runway == pytest.approx(x_lead + get_stopped_equivalence_factor(v_lead) - STOP_DISTANCE - reserve)
  assert slowing_runway != pytest.approx(x_lead + get_stopped_equivalence_factor(v_lead) - stop_distance_buffer(v_ego))


def test_moving_lead_stop_reserve_tapers_to_fixed_stop_gap():
  full_reserve = get_moving_lead_stop_reserve(v_ego=8.0, v_lead=2.0, closing_speed=2.0, a_lead=-1.0)

  assert full_reserve == pytest.approx(MOVING_LEAD_STOP_RESERVE_MAX)
  assert get_moving_lead_stop_reserve(8.0, 2.0, 0.0, -1.0) == pytest.approx(0.0)
  assert get_moving_lead_stop_reserve(0.0, 0.0, 2.0, 0.0) == pytest.approx(0.0)
  assert get_moving_lead_stop_reserve(8.0, 2.0, 2.0, 0.0) == pytest.approx(0.0)
  assert get_moving_lead_stop_reserve(8.0, 0.0, 2.0, 0.0) == pytest.approx(0.0)


def test_approach_runway_blend_uses_stop_runway_for_slowing_lead():
  t_follow = get_T_FOLLOW(log.LongitudinalPersonality.standard)
  assert get_approach_runway_blend(40.0, 20.0, 15.0, t_follow, a_lead=0.0) == pytest.approx(0.0)
  assert get_approach_runway_blend(40.0, 20.0, 15.0, t_follow, a_lead=-1.0) == pytest.approx(1.0)


def test_low_speed_stopped_lead_uses_stop_runway_gap():
  t_follow = get_T_FOLLOW(log.LongitudinalPersonality.standard)
  v_ego = 2.0
  runway_gap = STOP_DISTANCE + v_ego**2 / (2 * LEAD_STOP_RUNWAY_BRAKE)

  assert get_lead_stop_runway_blend(v_ego, 0.0, 0.0) == pytest.approx(1.0)
  assert get_lead_stop_runway_gap(v_ego, 0.0, v_ego, 0.0) == pytest.approx(runway_gap)
  assert get_approach_follow_distance(12.0, v_ego, 0.0, t_follow, a_lead=0.0) == pytest.approx(runway_gap)


def test_low_speed_slowing_lead_runway_accounts_for_lead_stop_point():
  v_ego = 3.0
  v_lead = 1.0
  a_lead = -0.8
  closing_speed = v_ego - v_lead
  reserve = get_moving_lead_stop_reserve(v_ego, v_lead, closing_speed, a_lead)
  expected_gap = STOP_DISTANCE + reserve + v_ego**2 / (2 * LEAD_STOP_RUNWAY_BRAKE) - get_stopped_equivalence_factor(v_lead)

  assert get_lead_stop_runway_blend(v_ego, v_lead, a_lead) > 0.0
  assert get_lead_stop_runway_gap(v_ego, v_lead, closing_speed, a_lead) == pytest.approx(max(STOP_DISTANCE, expected_gap))


def test_stop_runway_blend_stays_off_at_higher_speed():
  t_follow = get_T_FOLLOW(log.LongitudinalPersonality.standard)
  assert get_lead_stop_runway_blend(8.0, 0.0, 0.0) == pytest.approx(0.0)
  assert get_approach_runway_blend(12.0, 8.0, 0.0, t_follow, a_lead=0.0) == pytest.approx(0.0)


def test_lead_accel_match_tapers_positive_accel_under_time_gap():
  t_follow = get_T_FOLLOW(log.LongitudinalPersonality.standard)
  v_lead = 12.0
  a_lead = 1.0
  target_gap = get_lead_time_gap_target(v_lead, t_follow)

  near_stop_target, near_stop_cost = get_lead_accel_match_target(v_lead, STOP_DISTANCE + 0.5, a_lead, t_follow)
  mid_gap_target, mid_gap_cost = get_lead_accel_match_target(v_lead, 0.5 * (STOP_DISTANCE + target_gap), a_lead, t_follow)
  target_gap_target, target_gap_cost = get_lead_accel_match_target(v_lead, target_gap, a_lead, t_follow)

  assert 0.0 < near_stop_target < mid_gap_target < target_gap_target <= a_lead
  assert near_stop_target == pytest.approx(a_lead * LEAD_ACCEL_MATCH_MIN_POSITIVE_BLEND, abs=0.05)
  assert 0.0 < near_stop_cost < mid_gap_cost < target_gap_cost


def test_lead_accel_match_fades_far_decelerating_leads():
  t_follow = get_T_FOLLOW(log.LongitudinalPersonality.standard)
  v_ego = 25.0
  v_lead = 20.0
  a_lead = -1.0
  reserve = get_moving_lead_stop_reserve(v_ego, v_lead, v_ego - v_lead, a_lead)
  target_gap = get_lead_time_gap_target(v_lead, t_follow) + reserve
  far_gap = target_gap + get_lead_accel_match_margin(target_gap)

  near_target, near_cost = get_lead_accel_match_target(v_lead, target_gap, a_lead, t_follow, v_ego)
  far_target, far_cost = get_lead_accel_match_target(v_lead, far_gap, a_lead, t_follow, v_ego)

  assert near_target == pytest.approx(a_lead * LEAD_ACCEL_MATCH_DECEL_TARGET_BLEND)
  assert near_cost > 0.0
  assert far_target == pytest.approx(0.0)
  assert far_cost == pytest.approx(0.0)


def test_lead_accel_match_caps_soft_decel_target():
  t_follow = get_T_FOLLOW(log.LongitudinalPersonality.standard)
  v_ego = 15.0
  v_lead = 2.0
  reserve = get_moving_lead_stop_reserve(v_ego, v_lead, v_ego - v_lead, -4.0)
  target_gap = get_lead_time_gap_target(v_lead, t_follow) + reserve

  accel_target, _ = get_lead_accel_match_target(v_lead, target_gap, -4.0, t_follow, v_ego)

  assert accel_target == pytest.approx(-LEAD_ACCEL_MATCH_DECEL_CAP)


def test_approach_engage_offset_stays_off_without_closure_or_runway():
  t_follow = get_T_FOLLOW(log.LongitudinalPersonality.standard)
  assert get_approach_engage_offset(20.0, 40.0, 20.0, t_follow) == pytest.approx(0.0)
  assert get_approach_engage_offset(20.0, 20.0, 15.0, t_follow) == pytest.approx(0.0)


def test_approach_engage_offset_grows_for_large_closing_runway_cases():
  t_follow = get_T_FOLLOW(log.LongitudinalPersonality.standard)
  mild_offset = get_approach_engage_offset(20.0, 70.0, 15.0, t_follow, a_lead=-0.5)
  strong_offset = get_approach_engage_offset(20.0, 80.0, 0.0, t_follow, a_lead=-1.0)

  assert 0.0 < mild_offset < strong_offset <= APPROACH_ENGAGE_OFFSET_MAX


def test_lead_gap_comfort_uses_light_brake_for_steady_under_gap():
  t_follow = get_T_FOLLOW(log.LongitudinalPersonality.standard)
  v_ego = 20.0
  v_lead = 20.0
  comfort_floor = get_lead_gap_comfort_floor(v_ego, v_lead, t_follow)
  desired_gap = get_desired_follow_distance(v_ego, v_lead, t_follow)
  d_rel = 0.5 * (comfort_floor + desired_gap)

  comfort_a_min = get_lead_gap_comfort_a_min(v_ego, v_lead, d_rel, t_follow)

  assert -LEAD_GAP_COMFORT_LIGHT_DECEL <= comfort_a_min < 0.0


def test_lead_gap_comfort_prefers_coast_for_pullaway_lead():
  t_follow = get_T_FOLLOW(log.LongitudinalPersonality.standard)
  v_ego = 20.0
  v_lead = 20.5
  comfort_floor = get_lead_gap_comfort_floor(v_ego, v_lead, t_follow)
  desired_gap = get_desired_follow_distance(v_ego, v_lead, t_follow)
  d_rel = 0.5 * (comfort_floor + desired_gap)

  assert get_lead_gap_comfort_a_min(v_ego, v_lead, d_rel, t_follow) == pytest.approx(0.0)


def test_lead_gap_comfort_tapers_toward_coast_as_gap_recovers():
  t_follow = get_T_FOLLOW(log.LongitudinalPersonality.standard)
  v_ego = 20.0
  v_lead = 20.0
  comfort_floor = get_lead_gap_comfort_floor(v_ego, v_lead, t_follow)
  desired_gap = get_desired_follow_distance(v_ego, v_lead, t_follow)
  early_gap = comfort_floor + 0.2
  late_gap = desired_gap - 0.2

  early_a_min = get_lead_gap_comfort_a_min(v_ego, v_lead, early_gap, t_follow)
  late_a_min = get_lead_gap_comfort_a_min(v_ego, v_lead, late_gap, t_follow)

  assert early_a_min < late_a_min < 0.0
  assert get_lead_gap_comfort_recovery_blend(late_gap, comfort_floor, desired_gap) > get_lead_gap_comfort_recovery_blend(early_gap, comfort_floor, desired_gap)


def test_lead_gap_comfort_disables_near_danger_or_real_closure():
  t_follow = get_T_FOLLOW(log.LongitudinalPersonality.standard)
  v_ego = 20.0
  comfort_floor = get_lead_gap_comfort_floor(v_ego, v_ego, t_follow)
  desired_gap = get_desired_follow_distance(v_ego, v_ego, t_follow)
  d_rel = 0.5 * (comfort_floor + desired_gap)

  assert get_lead_gap_comfort_a_min(v_ego, v_ego - 1.0, d_rel, t_follow) == pytest.approx(ACCEL_MIN)
  assert get_lead_gap_comfort_a_min(v_ego, v_ego, comfort_floor - 0.1, t_follow) == pytest.approx(ACCEL_MIN)


def test_lead_accel_recovery_allows_accel_when_lead_pulls_away():
  t_follow = get_T_FOLLOW(log.LongitudinalPersonality.standard)
  recovery_a_min = get_lead_accel_recovery_a_min(2.2, 4.2, 15.0, 0.9, t_follow)

  assert 0.0 < recovery_a_min <= LEAD_ACCEL_RECOVERY_ACCEL_MAX


def test_lead_accel_recovery_requires_safe_opening_gap():
  t_follow = get_T_FOLLOW(log.LongitudinalPersonality.standard)
  assert get_lead_accel_recovery_a_min(2.2, 4.2, 5.0, 0.9, t_follow) == pytest.approx(ACCEL_MIN)
  assert get_lead_accel_recovery_a_min(2.2, 2.0, 15.0, 0.9, t_follow) == pytest.approx(ACCEL_MIN)
  assert get_lead_accel_recovery_a_min(2.2, 4.2, 15.0, 0.1, t_follow) == pytest.approx(ACCEL_MIN)


def run_following_distance_simulation(v_lead, t_end=100.0, e2e=False, personality=0):
  man = Maneuver(
    '',
    duration=t_end,
    initial_speed=float(v_lead),
    lead_relevancy=True,
    initial_distance_lead=100,
    speed_lead_values=[v_lead],
    breakpoints=[0.0],
    e2e=e2e,
    personality=personality,
  )
  valid, output = man.evaluate()
  assert valid
  return output[-1, 2] - output[-1, 1]


def run_lead_closing_simulation(v_ego, v_lead, initial_distance_lead, t_end=30.0, personality=0):
  man = Maneuver(
    '',
    duration=t_end,
    initial_speed=float(v_ego),
    lead_relevancy=True,
    initial_distance_lead=float(initial_distance_lead),
    speed_lead_values=[float(v_lead)],
    breakpoints=[0.0],
    personality=personality,
  )
  valid, output = man.evaluate()
  assert valid
  return output


@parameterized_class(
  ("e2e", "personality", "speed"),
  itertools.product(
    [True, False],  # e2e
    [
      log.LongitudinalPersonality.relaxed,  # personality
      log.LongitudinalPersonality.standard,
      log.LongitudinalPersonality.aggressive,
    ],
    [0, 10, 35],
  ),
)  # speed
class TestFollowingDistance:
  def test_following_distance(self):
    v_lead = float(self.speed)
    simulation_steady_state = run_following_distance_simulation(v_lead, e2e=self.e2e, personality=self.personality)
    correct_steady_state = get_desired_follow_distance(v_lead, v_lead, get_T_FOLLOW(self.personality))
    err_ratio = 0.2 if self.e2e else 0.1
    abs_err_margin = 0.5 if v_lead > 0.0 else 2.0
    assert simulation_steady_state == pytest.approx(correct_steady_state, abs=err_ratio * correct_steady_state + abs_err_margin)


def test_closing_lead_bleeds_off_speed_late_in_approach():
  t_follow = get_T_FOLLOW(log.LongitudinalPersonality.standard)
  output = run_lead_closing_simulation(v_ego=25.0, v_lead=20.0, initial_distance_lead=90.0)

  time = output[:, 0]
  closing_speed = output[:, 3] - output[:, 4]
  late_approach = time >= (time[-1] - 5.0)

  assert np.any(late_approach)
  assert np.max(closing_speed[late_approach]) < 1.5
  assert output[-1, 6] == pytest.approx(get_desired_follow_distance(20.0, 20.0, t_follow), abs=4.0)


def test_stopped_car_approach_settles_near_stop_gap():
  output = run_lead_closing_simulation(v_ego=20.0, v_lead=0.0, initial_distance_lead=90.0, t_end=20.0)

  assert output[-1, 6] == pytest.approx(5.75, abs=0.5)
