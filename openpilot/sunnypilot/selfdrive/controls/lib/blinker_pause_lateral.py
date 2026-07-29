"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import math

from openpilot.cereal import car

from openpilot.common.constants import CV
from openpilot.common.params import Params


def _nonnegative_float(value, default: float = 0.0) -> float:
  try:
    v = float(value)
  except (TypeError, ValueError):
    return default
  return max(0.0, v) if math.isfinite(v) else default


class BlinkerPauseLateral:
  def __init__(self):
    self.params = Params()

    self.enabled = self.params.get_bool("BlinkerPauseLateralControl")
    self.is_metric = self.params.get_bool("IsMetric")
    self.min_speed = 0
    self.reengage_delay = 0
    self.blinker_off_timer = 0.0

  def get_params(self) -> None:
    self.enabled = self.params.get_bool("BlinkerPauseLateralControl")
    self.is_metric = self.params.get_bool("IsMetric")
    self.min_speed = _nonnegative_float(self.params.get("BlinkerMinLateralControlSpeed", return_default=True))
    self.reengage_delay = _nonnegative_float(self.params.get("BlinkerLateralReengageDelay", return_default=True))

  def update(self, CS: car.CarState, DT_CTRL: float = 0.01) -> bool:
    if not self.enabled:
      return False

    one_blinker = CS.leftBlinker != CS.rightBlinker
    speed_factor = CV.KPH_TO_MS if self.is_metric else CV.MPH_TO_MS
    min_speed_ms = _nonnegative_float(self.min_speed) * speed_factor

    below_speed = CS.vEgo < min_speed_ms

    if one_blinker and below_speed:
      self.blinker_off_timer = _nonnegative_float(self.reengage_delay)
    elif self.blinker_off_timer > 0:
      self.blinker_off_timer -= DT_CTRL

    return bool((one_blinker and below_speed) or self.blinker_off_timer > 0)
