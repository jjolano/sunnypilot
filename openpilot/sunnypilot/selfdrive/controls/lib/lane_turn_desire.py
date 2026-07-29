"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from openpilot.cereal import custom

from openpilot.common.constants import CV
from openpilot.common.params import Params

TurnDirection = custom.ModelDataV2SP.TurnDirection

LANE_CHANGE_SPEED_MIN = 20 * CV.MPH_TO_MS


def _lane_turn_speed_from_param(value) -> float:
  try:
    speed_mph = float(value)
  except (TypeError, ValueError):
    return 0.0
  if not math.isfinite(speed_mph) or speed_mph <= 0.0:
    return 0.0
  return min(float(LANE_CHANGE_SPEED_MIN), speed_mph * CV.MPH_TO_MS)


class LaneTurnController:
  def __init__(self, desire_helper):
    self.DH = desire_helper
    self.turn_direction = TurnDirection.none
    self.params = Params()
    self.lane_turn_value = _lane_turn_speed_from_param(self.params.get("LaneTurnValue", return_default=True))
    self.param_read_counter = 0
    self.enabled = self.params.get_bool("LaneTurnDesire")

  def read_params(self):
    self.enabled = self.params.get_bool("LaneTurnDesire")
    self.lane_turn_value = _lane_turn_speed_from_param(self.params.get("LaneTurnValue", return_default=True))

  def update_params(self) -> None:
    if self.param_read_counter % 50 == 0:
      self.read_params()
    self.param_read_counter += 1

  def update_lane_turn(self, blindspot_left: bool, blindspot_right: bool, left_blinker: bool, right_blinker: bool, v_ego: float) -> None:
    if left_blinker and not right_blinker and v_ego < self.lane_turn_value and not blindspot_left:
      self.turn_direction = TurnDirection.turnLeft
    elif right_blinker and not left_blinker and v_ego < self.lane_turn_value and not blindspot_right:
      self.turn_direction = TurnDirection.turnRight
    else:
      self.turn_direction = TurnDirection.none

  def get_turn_direction(self):
    if not self.enabled:
      return TurnDirection.none
    return self.turn_direction
