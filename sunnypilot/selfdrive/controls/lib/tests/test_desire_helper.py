import math

from cereal import log
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.desire_helper import DesireHelper, LaneChangeDirection, LaneChangeState
from openpilot.selfdrive.controls.lib.lane_change_path_shaper import LANE_CHANGE_DURATION


class DummyCarState:
  def __init__(self, vEgo=9.0, leftBlinker=True, rightBlinker=False):
    self.vEgo = vEgo
    self.leftBlinker = leftBlinker
    self.rightBlinker = rightBlinker
    self.leftBlindspot = False
    self.rightBlindspot = False
    self.steeringPressed = False
    self.steeringTorque = 0.0
    self.brakePressed = False


def test_path_shaped_lane_change_completes_from_elapsed_progress():
  dh = DesireHelper()
  dh.lane_change_state = LaneChangeState.laneChangeStarting
  dh.lane_change_direction = LaneChangeDirection.left
  carstate = DummyCarState()

  steps = math.ceil(LANE_CHANGE_DURATION / DT_MDL) + 1
  for _ in range(steps):
    dh.update(carstate, True, 1.0)

  assert dh.lane_change_state == LaneChangeState.laneChangeFinishing
  assert dh.desire == log.Desire.laneChangeLeft


def test_model_probability_still_finishes_lane_change_early():
  dh = DesireHelper()
  dh.lane_change_state = LaneChangeState.laneChangeStarting
  dh.lane_change_direction = LaneChangeDirection.left
  carstate = DummyCarState()

  steps = math.ceil(0.6 / DT_MDL)
  for _ in range(steps):
    dh.update(carstate, True, 0.0)

  assert dh.lane_change_state == LaneChangeState.laneChangeFinishing


def test_finishing_with_held_signal_does_not_queue_next_lane_change():
  dh = DesireHelper()
  dh.lane_change_state = LaneChangeState.laneChangeFinishing
  dh.lane_change_direction = LaneChangeDirection.left
  dh.lane_change_ll_prob = 0.99
  dh.prev_one_blinker = True

  dh.update(DummyCarState(leftBlinker=True, rightBlinker=False), True, 1.0)

  assert dh.lane_change_state == LaneChangeState.off
  assert dh.lane_change_direction == LaneChangeDirection.none


def test_finishing_reinit_with_signal_cycle_queues_next_lane_change():
  dh = DesireHelper()
  dh.lane_change_state = LaneChangeState.laneChangeFinishing
  dh.lane_change_direction = LaneChangeDirection.left
  dh.lane_change_ll_prob = 0.9
  dh.prev_one_blinker = True

  dh.update(DummyCarState(leftBlinker=False, rightBlinker=False), True, 1.0)
  dh.update(DummyCarState(leftBlinker=True, rightBlinker=False), True, 1.0)

  assert dh.lane_change_state == LaneChangeState.preLaneChange
  assert dh.lane_change_direction == LaneChangeDirection.left


def test_finishing_reinit_with_opposite_direction_queues_new_direction():
  dh = DesireHelper()
  dh.lane_change_state = LaneChangeState.laneChangeFinishing
  dh.lane_change_direction = LaneChangeDirection.left
  dh.lane_change_ll_prob = 0.95
  dh.prev_one_blinker = True

  dh.update(DummyCarState(leftBlinker=False, rightBlinker=True), True, 1.0)

  assert dh.lane_change_state == LaneChangeState.preLaneChange
  assert dh.lane_change_direction == LaneChangeDirection.right
