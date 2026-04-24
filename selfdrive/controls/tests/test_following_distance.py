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
  LEAD_STOP_GAP_EXCESS_OFFSET_MAX,
  LEAD_STOP_GAP_TAPER_MAX,
  LEAD_GAP_COMFORT_LIGHT_DECEL,
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
  get_lead_gap_comfort_a_min,
  get_lead_gap_comfort_floor,
  get_lead_gap_comfort_recovery_blend,
  get_lead_departure_available_runway,
  get_lead_departure_relaxation,
  get_lead_danger_distance,
  get_lead_stop_gap_excess_offset,
  get_lead_stop_gap_taper,
  get_safe_obstacle_distance,
  get_stopped_lead_buffer,
  get_stopped_equivalence_factor,
  get_T_FOLLOW,
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
  assert get_lead_stop_gap_excess_offset(0.0, STOP_DISTANCE + 4.0) == pytest.approx(LEAD_STOP_GAP_EXCESS_OFFSET_MAX)
  assert get_lead_stop_gap_excess_offset(1.5, STOP_DISTANCE + 4.0) == pytest.approx(0.0)


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

  assert steady_runway == pytest.approx(max(40.0 - get_desired_follow_distance(20.0, 15.0, t_follow), 0.0))
  assert slowing_runway == pytest.approx(40.0 + get_stopped_equivalence_factor(15.0) - STOP_DISTANCE)
  assert slowing_runway > steady_runway


def test_approach_runway_blend_uses_stop_runway_for_slowing_lead():
  t_follow = get_T_FOLLOW(log.LongitudinalPersonality.standard)
  assert get_approach_runway_blend(40.0, 20.0, 15.0, t_follow, a_lead=0.0) == pytest.approx(0.0)
  assert get_approach_runway_blend(40.0, 20.0, 15.0, t_follow, a_lead=-1.0) == pytest.approx(1.0)


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
