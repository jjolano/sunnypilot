"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from abc import abstractmethod, ABC

import cereal.messaging as messaging
from openpilot.common.params import Params
from openpilot.common.constants import CV
from openpilot.selfdrive.car.cruise import V_CRUISE_UNSET
from openpilot.sunnypilot.navd.helpers import coordinate_from_param

MAX_SPEED_LIMIT = V_CRUISE_UNSET * CV.KPH_TO_MS


class BaseMapData(ABC):
  def __init__(self):
    self.params = Params()

    self.sm = messaging.SubMaster(['liveLocationKalman'])
    self.pm = messaging.PubMaster(['liveMapDataSP'])

    self.localizer_valid = False
    self.last_bearing = None
    self.last_position = coordinate_from_param("LastGPSPositionLLK", self.params)

  @abstractmethod
  def update_location(self) -> None:
    pass

  @abstractmethod
  def get_current_speed_limit(self) -> float:
    pass

  @abstractmethod
  def get_next_speed_limit_and_distance(self) -> tuple[float, float]:
    pass

  @abstractmethod
  def get_current_road_name(self) -> str:
    pass

  @abstractmethod
  def get_current_lanes(self) -> int:
    pass

  @abstractmethod
  def get_next_lanes_and_distance(self) -> tuple[int, float]:
    pass

  @abstractmethod
  def get_road_context(self) -> str:
    pass

  @abstractmethod
  def get_current_hazard_and_distance(self) -> tuple[str, float]:
    pass

  @abstractmethod
  def get_next_hazard_and_distance(self) -> tuple[str, float]:
    pass

  @abstractmethod
  def get_current_traffic_control_and_distance(self) -> tuple[str, float]:
    pass

  @abstractmethod
  def get_next_traffic_control_and_distance(self) -> tuple[str, float]:
    pass

  def publish(self) -> None:
    speed_limit = self.get_current_speed_limit()
    next_speed_limit, next_speed_limit_distance = self.get_next_speed_limit_and_distance()
    hazard, hazard_distance = self.get_current_hazard_and_distance()
    hazard_ahead, hazard_ahead_distance = self.get_next_hazard_and_distance()
    traffic_control, traffic_control_distance = self.get_current_traffic_control_and_distance()
    traffic_control_ahead, traffic_control_ahead_distance = self.get_next_traffic_control_and_distance()
    lanes = self.get_current_lanes()
    lanes_ahead, lanes_ahead_distance = self.get_next_lanes_and_distance()

    mapd_sp_send = messaging.new_message('liveMapDataSP')
    mapd_sp_send.valid = self.sm['liveLocationKalman'].gpsOK
    live_map_data = mapd_sp_send.liveMapDataSP

    live_map_data.speedLimitValid = bool(MAX_SPEED_LIMIT > speed_limit > 0)
    live_map_data.speedLimit = speed_limit
    live_map_data.speedLimitAheadValid = bool(MAX_SPEED_LIMIT > next_speed_limit > 0)
    live_map_data.speedLimitAhead = next_speed_limit
    live_map_data.speedLimitAheadDistance = next_speed_limit_distance
    live_map_data.roadName = self.get_current_road_name()
    live_map_data.hazardValid = bool(hazard)
    live_map_data.hazard = hazard
    live_map_data.hazardDistance = hazard_distance
    live_map_data.hazardAheadValid = bool(hazard_ahead)
    live_map_data.hazardAhead = hazard_ahead
    live_map_data.hazardAheadDistance = hazard_ahead_distance
    live_map_data.trafficControlValid = bool(traffic_control)
    live_map_data.trafficControl = traffic_control
    live_map_data.trafficControlDistance = traffic_control_distance
    live_map_data.trafficControlAheadValid = bool(traffic_control_ahead)
    live_map_data.trafficControlAhead = traffic_control_ahead
    live_map_data.trafficControlAheadDistance = traffic_control_ahead_distance
    live_map_data.lanesValid = lanes > 0
    live_map_data.lanes = min(255, max(0, lanes))
    live_map_data.lanesAheadValid = lanes_ahead > 0
    live_map_data.lanesAhead = min(255, max(0, lanes_ahead))
    live_map_data.lanesAheadDistance = lanes_ahead_distance
    live_map_data.roadContext = self.get_road_context()

    self.pm.send('liveMapDataSP', mapd_sp_send)

  def tick(self) -> None:
    self.sm.update(0)
    self.update_location()
    self.publish()
