import pytest
import itertools
from openpilot.common.parameterized import parameterized_class

from cereal import log

from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import (
  COMFORT_BRAKE,
  STOP_DISTANCE,
  STOP_DISTANCE_FADE_V,
  get_safe_obstacle_distance,
  get_stopped_equivalence_factor,
  get_T_FOLLOW,
)
from openpilot.selfdrive.test.longitudinal_maneuvers.maneuver import Maneuver


def stop_distance_buffer(v_ego):
  return STOP_DISTANCE * (STOP_DISTANCE_FADE_V**2) / (v_ego**2 + STOP_DISTANCE_FADE_V**2)


def desired_follow_distance(v_ego, v_lead, t_follow=None):
  if t_follow is None:
    t_follow = get_T_FOLLOW()
  return (v_ego**2) / (2 * COMFORT_BRAKE) + t_follow * v_ego + stop_distance_buffer(v_ego) - get_stopped_equivalence_factor(v_lead)


@pytest.mark.parametrize("speed", [0.0, 5.0, 10.0, 35.0])
def test_safe_obstacle_distance_matches_explicit_formula(speed):
  t_follow = get_T_FOLLOW(log.LongitudinalPersonality.standard)
  expected = (speed**2) / (2 * COMFORT_BRAKE) + t_follow * speed + stop_distance_buffer(speed)
  assert get_safe_obstacle_distance(speed, t_follow) == pytest.approx(expected)


def test_stop_distance_buffer_fades_with_speed():
  buffer_speeds = [0.0, 5.0, 10.0, 35.0]
  buffers = [stop_distance_buffer(speed) for speed in buffer_speeds]
  assert buffers[0] == pytest.approx(STOP_DISTANCE)
  assert buffers[0] > buffers[1] > buffers[2] > buffers[3] > 0.0


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
    correct_steady_state = desired_follow_distance(v_lead, v_lead, get_T_FOLLOW(self.personality))
    err_ratio = 0.2 if self.e2e else 0.1
    abs_err_margin = 0.5 if v_lead > 0.0 else 1.15
    assert simulation_steady_state == pytest.approx(correct_steady_state, abs=err_ratio * correct_steady_state + abs_err_margin)
