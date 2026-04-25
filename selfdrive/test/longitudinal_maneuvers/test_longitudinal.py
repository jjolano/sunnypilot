import itertools
import numpy as np
from openpilot.common.parameterized import parameterized_class

from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import (
  STOP_DISTANCE,
  get_desired_follow_distance,
  get_lead_gap_comfort_floor,
  get_T_FOLLOW,
)
from openpilot.selfdrive.test.longitudinal_maneuvers.maneuver import Maneuver


# TODO: make new FCW tests
def create_maneuvers(kwargs):
  maneuvers = [
    Maneuver(
      'approach stopped car at 25m/s, initial distance: 120m',
      duration=20.0,
      initial_speed=25.0,
      lead_relevancy=True,
      initial_distance_lead=120.0,
      speed_lead_values=[30.0, 0.0],
      breakpoints=[0.0, 1.0],
      **kwargs,
    ),
    Maneuver(
      'approach stopped car at 20m/s, initial distance 90m',
      duration=20.0,
      initial_speed=20.0,
      lead_relevancy=True,
      initial_distance_lead=90.0,
      speed_lead_values=[20.0, 0.0],
      breakpoints=[0.0, 1.0],
      **kwargs,
    ),
    Maneuver(
      'steady state following a car at 20m/s, then lead decel to 0mph at 1m/s^2',
      duration=50.0,
      initial_speed=20.0,
      lead_relevancy=True,
      initial_distance_lead=35.0,
      speed_lead_values=[20.0, 20.0, 0.0],
      breakpoints=[0.0, 15.0, 35.0],
      **kwargs,
    ),
    Maneuver(
      'steady state following a car at 20m/s, then lead decel to 0mph at 2m/s^2',
      duration=50.0,
      initial_speed=20.0,
      lead_relevancy=True,
      initial_distance_lead=35.0,
      speed_lead_values=[20.0, 20.0, 0.0],
      breakpoints=[0.0, 15.0, 25.0],
      **kwargs,
    ),
    Maneuver(
      'steady state following a car at 20m/s, then lead decel to 0mph at 3m/s^2',
      duration=50.0,
      initial_speed=20.0,
      lead_relevancy=True,
      initial_distance_lead=35.0,
      speed_lead_values=[20.0, 20.0, 0.0],
      breakpoints=[0.0, 15.0, 21.66],
      **kwargs,
    ),
    Maneuver(
      'steady state following a car at 20m/s, then lead decel to 0mph at 3+m/s^2',
      duration=40.0,
      initial_speed=20.0,
      lead_relevancy=True,
      initial_distance_lead=35.0,
      speed_lead_values=[20.0, 20.0, 0.0],
      prob_lead_values=[0.0, 1.0, 1.0],
      cruise_values=[20.0, 20.0, 20.0],
      breakpoints=[2.0, 2.01, 8.8],
      **kwargs,
    ),
    Maneuver(
      "approach stopped car at 20m/s, with prob_lead_values",
      duration=30.0,
      initial_speed=20.0,
      lead_relevancy=True,
      initial_distance_lead=120.0,
      speed_lead_values=[0.0, 0.0, 0.0],
      prob_lead_values=[0.0, 0.0, 1.0],
      cruise_values=[20.0, 20.0, 20.0],
      breakpoints=[0.0, 2.0, 2.01],
      **kwargs,
    ),
    Maneuver(
      "approach stopped car at 20m/s, with prob_throttle_values and pitch = -0.1",
      duration=30.0,
      initial_speed=20.0,
      lead_relevancy=True,
      initial_distance_lead=120.0,
      speed_lead_values=[0.0, 0.0, 0.0],
      prob_throttle_values=[1.0, 0.0, 0.0],
      cruise_values=[20.0, 20.0, 20.0],
      pitch_values=[0.0, -0.1, -0.1],
      breakpoints=[0.0, 2.0, 2.01],
      **kwargs,
    ),
    Maneuver(
      "approach stopped car at 20m/s, with prob_throttle_values and pitch = +0.1",
      duration=30.0,
      initial_speed=20.0,
      lead_relevancy=True,
      initial_distance_lead=120.0,
      speed_lead_values=[0.0, 0.0, 0.0],
      prob_throttle_values=[1.0, 0.0, 0.0],
      cruise_values=[20.0, 20.0, 20.0],
      pitch_values=[0.0, 0.1, 0.1],
      breakpoints=[0.0, 2.0, 2.01],
      **kwargs,
    ),
    Maneuver(
      "approach slower cut-in car at 20m/s",
      duration=20.0,
      initial_speed=20.0,
      lead_relevancy=True,
      initial_distance_lead=50.0,
      speed_lead_values=[15.0, 15.0],
      breakpoints=[1.0, 11.0],
      only_lead2=True,
      **kwargs,
    ),
    Maneuver(
      "stay stopped behind radar override lead",
      duration=20.0,
      initial_speed=0.0,
      lead_relevancy=True,
      initial_distance_lead=10.0,
      speed_lead_values=[0.0, 0.0],
      prob_lead_values=[0.0, 0.0],
      breakpoints=[1.0, 11.0],
      only_radar=True,
      **kwargs,
    ),
    Maneuver(
      "NaN recovery",
      duration=30.0,
      initial_speed=15.0,
      lead_relevancy=True,
      initial_distance_lead=60.0,
      speed_lead_values=[0.0, 0.0, 0.0],
      breakpoints=[1.0, 1.01, 11.0],
      cruise_values=[float("nan"), 15.0, 15.0],
      **kwargs,
    ),
    Maneuver(
      'cruising at 25 m/s while disabled',
      duration=20.0,
      initial_speed=25.0,
      lead_relevancy=False,
      enabled=False,
      **kwargs,
    ),
    Maneuver(
      "slow to 5m/s with allow_throttle = False and pitch = +0.1",
      duration=30.0,
      initial_speed=20.0,
      lead_relevancy=False,
      prob_throttle_values=[1.0, 0.0, 0.0],
      cruise_values=[20.0, 20.0, 20.0],
      pitch_values=[0.0, 0.1, 0.1],
      breakpoints=[0.0, 2.0, 2.01],
      ensure_slowdown=True,
      **kwargs,
    ),
  ]
  if not kwargs['force_decel']:
    maneuvers.append(
      Maneuver(
        "approach stopped car that resumes while still closing",
        duration=40.0,
        initial_speed=20.0,
        lead_relevancy=True,
        initial_distance_lead=120.0,
        speed_lead_values=[0.0, 0.0, 10.0, 10.0],
        breakpoints=[0.0, 2.0, 8.0, 40.0],
        **kwargs,
      )
    )
    # controls relies on planner commanding to move for stock-ACC resume spamming
    maneuvers.append(
      Maneuver(
        "resume from a stop",
        duration=20.0,
        initial_speed=0.0,
        lead_relevancy=True,
        initial_distance_lead=STOP_DISTANCE,
        speed_lead_values=[0.0, 0.0, 2.0],
        breakpoints=[1.0, 10.0, 15.0],
        ensure_start=True,
        **kwargs,
      )
    )
  return maneuvers


def evaluate_maneuver_output(maneuver):
  valid, output = maneuver.evaluate()
  assert valid
  return output


def run_under_gap_cut_in_simulation(v_ego, v_lead, duration=10.0):
  t_follow = get_T_FOLLOW()
  comfort_floor = get_lead_gap_comfort_floor(v_ego, v_lead, t_follow)
  desired_gap = get_desired_follow_distance(v_ego, v_lead, t_follow)
  initial_distance_lead = 0.5 * (comfort_floor + desired_gap)
  return evaluate_maneuver_output(
    Maneuver(
      "under-gap cut-in",
      duration=duration,
      initial_speed=v_ego,
      lead_relevancy=True,
      initial_distance_lead=initial_distance_lead,
      speed_lead_values=[v_lead, v_lead, v_lead],
      prob_lead_values=[0.0, 0.0, 1.0],
      cruise_values=[v_ego, v_ego, v_ego],
      breakpoints=[0.0, 2.0, 2.01],
    )
  )


def test_lead_creep_then_stop_does_not_launch_from_gap_noise():
  output = evaluate_maneuver_output(
    Maneuver(
      "lead creeps then stops",
      duration=20.0,
      initial_speed=0.0,
      lead_relevancy=True,
      initial_distance_lead=STOP_DISTANCE,
      speed_lead_values=[0.0, 0.0, 0.8, 0.0, 0.0],
      breakpoints=[0.0, 10.0, 10.3, 10.6, 20.0],
    )
  )

  assert np.max(output[:, 3]) < 0.35
  assert output[-1, 6] > STOP_DISTANCE - 0.8


def test_lead_departure_creeps_once_gap_opens():
  output = evaluate_maneuver_output(
    Maneuver(
      "lead departs from stop",
      duration=20.0,
      initial_speed=0.0,
      lead_relevancy=True,
      initial_distance_lead=STOP_DISTANCE,
      speed_lead_values=[0.0, 0.0, 1.5, 1.5],
      breakpoints=[0.0, 10.0, 11.0, 20.0],
    )
  )

  gap_opening = output[:, 6] - STOP_DISTANCE
  departure_idx = np.argmax(gap_opening >= 0.5)
  assert gap_opening[departure_idx] >= 0.5

  response_window = (output[:, 0] >= output[departure_idx, 0]) & (output[:, 0] <= output[departure_idx, 0] + 2.0)
  assert np.max(output[response_window, 5]) > 0.25
  assert np.max(output[response_window, 3]) > 0.25


def test_confirmed_pullaway_uses_uncapped_creep_accel():
  output = evaluate_maneuver_output(
    Maneuver(
      "lead pulls away from short stopped gap",
      duration=16.0,
      initial_speed=0.0,
      lead_relevancy=True,
      initial_distance_lead=STOP_DISTANCE + 0.6,
      speed_lead_values=[0.0, 0.0, 1.8, 4.0, 4.0],
      breakpoints=[0.0, 10.0, 11.0, 13.0, 16.0],
    )
  )

  pullaway_idx = np.argmax(output[:, 4] >= 1.5)
  assert output[pullaway_idx, 4] >= 1.5

  response_window = (output[:, 0] >= output[pullaway_idx, 0]) & (output[:, 0] <= output[pullaway_idx, 0] + 1.0)
  assert np.max(output[response_window, 5]) > 0.25
  assert np.min(output[response_window, 6]) > STOP_DISTANCE


def test_predicted_pullaway_releases_before_measured_speed_threshold():
  output = evaluate_maneuver_output(
    Maneuver(
      "lead accelerates from short stopped gap",
      duration=16.0,
      initial_speed=0.0,
      lead_relevancy=True,
      initial_distance_lead=STOP_DISTANCE + 0.35,
      speed_lead_values=[0.0, 0.0, 1.0, 2.0, 2.0],
      breakpoints=[0.0, 10.0, 11.0, 13.0, 16.0],
    )
  )

  early_pullaway = (output[:, 0] >= 10.0) & (output[:, 4] < 0.25)
  assert np.max(output[early_pullaway, 5]) >= 0.25
  assert np.min(output[:, 6]) > STOP_DISTANCE


def test_lead_creep_uses_extra_stopped_gap():
  output = evaluate_maneuver_output(
    Maneuver(
      "lead creeps while over stopped gap",
      duration=20.0,
      initial_speed=0.0,
      lead_relevancy=True,
      initial_distance_lead=STOP_DISTANCE + 1.0,
      speed_lead_values=[0.0, 0.0, 0.8, 0.8],
      breakpoints=[0.0, 10.0, 10.5, 20.0],
    )
  )

  lead_roll_idx = np.argmax(output[:, 4] >= 0.3)
  response_window = (output[:, 0] >= output[lead_roll_idx, 0]) & (output[:, 0] <= output[lead_roll_idx, 0] + 1.0)

  assert np.max(output[response_window, 5]) > 0.05
  assert np.max(output[response_window, 3]) > 0.1
  assert np.min(output[response_window, 6]) > STOP_DISTANCE


def test_stationary_lead_over_stopped_gap_creeps_toward_target_gap():
  output = evaluate_maneuver_output(
    Maneuver(
      "stationary lead while over stopped gap",
      duration=20.0,
      initial_speed=0.0,
      lead_relevancy=True,
      initial_distance_lead=STOP_DISTANCE + 1.0,
      speed_lead_values=[0.0, 0.0],
      breakpoints=[0.0, 20.0],
    )
  )

  early_window = output[:, 0] <= 3.0
  assert np.max(output[early_window, 3]) > 0.1
  assert output[-1, 6] > STOP_DISTANCE - 0.25


def test_near_target_stopped_lead_hold_does_not_roll_through_gap():
  output = evaluate_maneuver_output(
    Maneuver(
      "stationary lead near stopped gap",
      duration=8.0,
      initial_speed=0.0,
      lead_relevancy=True,
      initial_distance_lead=STOP_DISTANCE + 0.3,
      speed_lead_values=[0.0, 0.0],
      breakpoints=[0.0, 8.0],
    )
  )

  assert np.max(output[:, 3]) < 0.03
  assert np.min(output[:, 6]) > STOP_DISTANCE + 0.2


def test_rolling_lead_stop_does_not_stage_at_reserve_gap():
  output = evaluate_maneuver_output(
    Maneuver(
      "rolling lead stop",
      duration=25.0,
      initial_speed=5.0,
      lead_relevancy=True,
      initial_distance_lead=18.0,
      speed_lead_values=[5.0, 5.0, 0.0, 0.0],
      cruise_values=[5.0, 5.0, 5.0, 5.0],
      breakpoints=[0.0, 2.0, 10.0, 25.0],
    )
  )

  stopped_idxs = np.where(output[:, 3] < 0.03)[0]
  assert len(stopped_idxs) > 0

  first_stop_gap = output[stopped_idxs[0], 6]
  assert first_stop_gap < STOP_DISTANCE + 0.5
  assert abs(first_stop_gap - output[-1, 6]) < 0.2


def test_equal_speed_under_gap_cut_in_uses_only_light_brake():
  output = run_under_gap_cut_in_simulation(20.0, 20.0)

  response_window = (output[:, 0] >= 2.0) & (output[:, 0] <= 4.0)
  recovery_window = output[:, 0] >= 6.0
  initial_gap = output[np.argmax(output[:, 0] >= 2.0), 6]

  assert np.min(output[response_window, 5]) < -0.05
  assert np.min(output[response_window, 5]) > -0.7
  assert np.min(output[recovery_window, 5]) > np.min(output[response_window, 5])
  assert output[-1, 6] > initial_gap


def test_pullaway_under_gap_cut_in_prefers_coast():
  output = run_under_gap_cut_in_simulation(20.0, 20.5)

  response_window = (output[:, 0] >= 2.0) & (output[:, 0] <= 4.0)
  initial_gap = output[np.argmax(output[:, 0] >= 2.0), 6]

  assert np.min(output[response_window, 5]) > -0.2
  assert output[-1, 6] > initial_gap


def test_closing_under_gap_cut_in_still_brakes_normally():
  steady_output = run_under_gap_cut_in_simulation(20.0, 20.0)
  output = run_under_gap_cut_in_simulation(20.0, 18.0)

  response_window = (output[:, 0] >= 2.0) & (output[:, 0] <= 4.0)
  steady_response_window = (steady_output[:, 0] >= 2.0) & (steady_output[:, 0] <= 4.0)

  assert np.min(output[response_window, 5]) < np.min(steady_output[steady_response_window, 5]) - 0.2


def test_accelerating_lead_under_time_gap_allows_tapered_accel():
  v_ego = 15.0
  t_follow = get_T_FOLLOW()
  target_gap = get_desired_follow_distance(v_ego, v_ego, t_follow)
  initial_distance_lead = 0.5 * (STOP_DISTANCE + target_gap)
  output = evaluate_maneuver_output(
    Maneuver(
      "accelerating lead under time gap",
      duration=12.0,
      initial_speed=v_ego,
      lead_relevancy=True,
      initial_distance_lead=initial_distance_lead,
      speed_lead_values=[15.0, 15.0, 18.0, 18.0],
      cruise_values=[20.0, 20.0, 20.0, 20.0],
      breakpoints=[0.0, 2.0, 5.0, 12.0],
    )
  )

  lead_accel_window = (output[:, 0] >= 2.0) & (output[:, 0] <= 5.0)

  assert np.max(output[lead_accel_window, 5]) > 0.2
  assert np.min(output[lead_accel_window, 6]) > STOP_DISTANCE
  assert output[-1, 6] > initial_distance_lead


@parameterized_class(("e2e", "force_decel"), itertools.product([True, False], repeat=2))
class TestLongitudinalControl:
  e2e: bool
  force_decel: bool

  def test_maneuver(self, subtests):
    for maneuver in create_maneuvers({"e2e": self.e2e, "force_decel": self.force_decel}):
      with subtests.test(title=maneuver.title, e2e=maneuver.e2e, force_decel=maneuver.force_decel):
        print(maneuver.title, f'in {"e2e" if maneuver.e2e else "acc"} mode')
        valid, _ = maneuver.evaluate()
        assert valid
