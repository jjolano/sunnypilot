import bisect
from enum import IntEnum
from abc import abstractmethod
from collections.abc import Callable
from typing import cast

from openpilot.cereal import log
from opendbc.car.structs import car
import openpilot.cereal.messaging as messaging
from openpilot.common.realtime import DT_CTRL
from openpilot.common.hardware import HARDWARE

AlertSize = log.SelfdriveState.AlertSize
AlertStatus = log.SelfdriveState.AlertStatus
VisualAlert = car.CarControl.HUDControl.VisualAlert
AudibleAlert = log.SelfdriveState.AudibleAlert


# Alert priorities
class Priority(IntEnum):
  LOWEST = 0
  LOWER = 1
  LOW = 2
  MID = 3
  HIGH = 4
  HIGHEST = 5


# Event types
class ET:
  ENABLE = 'enable'
  PRE_ENABLE = 'preEnable'
  OVERRIDE_LATERAL = 'overrideLateral'
  OVERRIDE_LONGITUDINAL = 'overrideLongitudinal'
  NO_ENTRY = 'noEntry'
  WARNING = 'warning'
  USER_DISABLE = 'userDisable'
  SOFT_DISABLE = 'softDisable'
  IMMEDIATE_DISABLE = 'immediateDisable'
  PERMANENT = 'permanent'


class Alert:
  def __init__(self,
               alert_text_1: str,
               alert_text_2: str,
               alert_status: log.SelfdriveState.AlertStatus,
               alert_size: log.SelfdriveState.AlertSize,
               priority: Priority,
               visual_alert: car.CarControl.HUDControl.VisualAlert,
               audible_alert: log.SelfdriveState.AudibleAlert,
               duration: float,
               creation_delay: float = 0.):

    self.alert_text_1 = alert_text_1
    self.alert_text_2 = alert_text_2
    self.alert_status = alert_status
    self.alert_size = alert_size
    self.priority = priority
    self.visual_alert = visual_alert
    self.audible_alert = audible_alert

    self.duration = int(duration / DT_CTRL)

    self.creation_delay = creation_delay

    self.alert_type = ""
    self.event_type: str | None = None

  def __str__(self) -> str:
    return f"{self.alert_text_1}/{self.alert_text_2} {self.priority} {self.visual_alert} {self.audible_alert}"

  def __gt__(self, alert2) -> bool:
    if not isinstance(alert2, Alert):
      return False
    return self.priority > alert2.priority

class AlertBase(Alert):
  def __init__(self, alert_text_1: str, alert_text_2: str, alert_status: log.SelfdriveState.AlertStatus,
               alert_size: log.SelfdriveState.AlertSize, priority: Priority,
               visual_alert: car.CarControl.HUDControl.VisualAlert,
               audible_alert: log.SelfdriveState.AudibleAlert, duration: float):
    super().__init__(alert_text_1, alert_text_2, alert_status, alert_size, priority, visual_alert, audible_alert, duration)


AlertCallbackType = Callable[[car.CarParams, car.CarState, messaging.SubMaster, bool, int, log.ControlsState], Alert]


# ********** alert callback functions **********


def wrong_car_mode_alert(CP: car.CarParams, CS: car.CarState, sm: messaging.SubMaster, metric: bool, soft_disable_time: int, personality) -> Alert:
  text = "Enable Adaptive Cruise to Engage"
  if CP.brand == "honda":
    text = "Enable Main Switch to Engage"
  return NoEntryAlert(text)


class EventsBase:
  def __init__(self):
    self.events: list[int] = []
    self.static_events: list[int] = []
    self.event_counters = {}
    self._event_counts: dict[int, int] = {}
    self._static_event_counts: dict[int, int] = {}
    self._event_type_counts: dict[str, int] = {}
    self._static_event_type_counts: dict[str, int] = {}

  @property
  def names(self) -> list[int]:
    return self.events

  def __len__(self) -> int:
    return len(self.events)

  @staticmethod
  def _remove_sorted(items: list[int], event_name: int) -> bool:
    idx = bisect.bisect_left(items, event_name)
    if idx < len(items) and items[idx] == event_name:
      items.pop(idx)
      return True
    return False

  def _adjust_cache(self, counts: dict[int, int], type_counts: dict[str, int],
                    event_name: int, delta: int, event_types: dict[str, Alert | Callable[..., Alert]]) -> None:
    count = counts.get(event_name, 0) + delta
    if count > 0:
      counts[event_name] = count
    else:
      counts.pop(event_name, None)

    for event_type in event_types:
      type_count = type_counts.get(event_type, 0) + delta
      if type_count > 0:
        type_counts[event_type] = type_count
      else:
        type_counts.pop(event_type, None)

  def add(self, event_name: int, static: bool = False) -> None:
    event_types = self.get_events_mapping().get(event_name, {})
    self.event_counters.setdefault(event_name, 0)
    if static:
      bisect.insort(self.static_events, event_name)
      self._adjust_cache(self._static_event_counts, self._static_event_type_counts, event_name, 1, event_types)

    bisect.insort(self.events, event_name)
    self._adjust_cache(self._event_counts, self._event_type_counts, event_name, 1, event_types)

  def clear(self) -> None:
    current_counts = self._event_counts
    for event_name in current_counts:
      self.event_counters.setdefault(event_name, 0)
    self.event_counters = {k: (v + 1 if current_counts.get(k, 0) else 0) for k, v in self.event_counters.items()}
    self.events = self.static_events.copy()
    self._event_counts = self._static_event_counts.copy()
    self._event_type_counts = self._static_event_type_counts.copy()

  def contains(self, event_type: str) -> bool:
    return self._event_type_counts.get(event_type, 0) > 0

  def create_alerts(self, event_types: list[str], callback_args=None):
    if callback_args is None:
      callback_args = []

    ret = []
    mapping = self.get_events_mapping()
    for e in self.events:
      event_alerts = mapping[e]
      event_count = self.event_counters.get(e, 0)
      for et in event_types:
        if et in event_alerts:
          alert = event_alerts[et]
          if not isinstance(alert, Alert):
            alert = cast(Callable[..., Alert], alert)(*callback_args)

          if DT_CTRL * (event_count + 1) >= alert.creation_delay:
            alert.alert_type = f"{self.get_event_name(e)}/{et}"
            alert.event_type = et
            ret.append(alert)
    return ret

  def add_from_msg(self, events):
    for e in events:
      event_name = e.name.raw
      event_types = self.get_events_mapping().get(event_name, {})
      self.event_counters.setdefault(event_name, 0)
      bisect.insort(self.events, event_name)
      self._adjust_cache(self._event_counts, self._event_type_counts, event_name, 1, event_types)

  def to_msg(self):
    ret = []
    mapping = self.get_events_mapping()
    for event_name in self.events:
      event = self.get_event_msg_type().new_message()
      event.name = event_name
      for event_type in mapping.get(event_name, {}):
        setattr(event, event_type, True)
      ret.append(event)
    return ret

  def has(self, event_name: int) -> bool:
    return self._event_counts.get(event_name, 0) > 0

  def contains_in_list(self, events_list: list[int]) -> bool:
    return any(self._event_counts.get(event_name, 0) > 0 for event_name in events_list)

  def remove(self, event_name: int, static: bool = False) -> None:
    event_types = self.get_events_mapping().get(event_name, {})

    if static and self._remove_sorted(self.static_events, event_name):
      self._adjust_cache(self._static_event_counts, self._static_event_type_counts, event_name, -1, event_types)

    if self._remove_sorted(self.events, event_name):
      self.event_counters[event_name] = self.event_counters.get(event_name, 0) + 1
      self._adjust_cache(self._event_counts, self._event_type_counts, event_name, -1, event_types)

  @abstractmethod
  def get_events_mapping(self) -> dict[int, dict[str, Alert | Callable[..., Alert]]]:
    raise NotImplementedError

  @abstractmethod
  def get_event_name(self, event: int) -> str:
    raise NotImplementedError

  @abstractmethod
  def get_event_msg_type(self):
    raise NotImplementedError


EmptyAlert = Alert("" , "", AlertStatus.normal, AlertSize.none, Priority.LOWEST,
                   VisualAlert.none, AudibleAlert.none, 0)

class NoEntryAlert(Alert):
  def __init__(self, alert_text_2: str,
               alert_text_1: str = "openpilot Unavailable",
               visual_alert: car.CarControl.HUDControl.VisualAlert=VisualAlert.none,
               priority: Priority = Priority.LOW):
    if HARDWARE.get_device_type() == 'mici':
      alert_text_1, alert_text_2 = alert_text_2, alert_text_1
    super().__init__(alert_text_1, alert_text_2, AlertStatus.normal,
                     AlertSize.mid, priority, visual_alert,
                     AudibleAlert.refuse, 3.)


class SoftDisableAlert(Alert):
  def __init__(self, alert_text_2: str):
    super().__init__("TAKE CONTROL IMMEDIATELY", alert_text_2,
                     AlertStatus.userPrompt, AlertSize.full,
                     Priority.MID, VisualAlert.steerRequired,
                     AudibleAlert.warningSoft, 2.),


# less harsh version of SoftDisable, where the condition is user-triggered
class UserSoftDisableAlert(SoftDisableAlert):
  def __init__(self, alert_text_2: str):
    super().__init__(alert_text_2),
    self.alert_text_1 = "openpilot will disengage"


class ImmediateDisableAlert(Alert):
  def __init__(self, alert_text_2: str):
    super().__init__("TAKE CONTROL IMMEDIATELY", alert_text_2,
                     AlertStatus.critical, AlertSize.full,
                     Priority.HIGHEST, VisualAlert.steerRequired,
                     AudibleAlert.warningImmediate, 4.),


class EngagementAlert(Alert):
  def __init__(self, audible_alert: log.SelfdriveState.AudibleAlert):
    super().__init__("", "",
                     AlertStatus.normal, AlertSize.none,
                     Priority.MID, VisualAlert.none,
                     audible_alert, .2),


class NormalPermanentAlert(Alert):
  def __init__(self, alert_text_1: str, alert_text_2: str = "", duration: float = 0.2, priority: Priority = Priority.LOWER, creation_delay: float = 0.):
    super().__init__(alert_text_1, alert_text_2,
                     AlertStatus.normal, AlertSize.mid if len(alert_text_2) else AlertSize.small,
                     priority, VisualAlert.none, AudibleAlert.none, duration, creation_delay=creation_delay),


class StartupAlert(Alert):
  def __init__(self, alert_text_1: str, alert_text_2: str = "Always keep hands on wheel and eyes on road", alert_status=AlertStatus.normal):
    alert_size = AlertSize.mid
    if HARDWARE.get_device_type() == 'mici':
      if alert_text_2 == "Always keep hands on wheel and eyes on road":
        alert_text_2 = ""
      alert_size = AlertSize.small
    super().__init__(alert_text_1, alert_text_2,
                     alert_status, alert_size,
                     Priority.LOWER, VisualAlert.none, AudibleAlert.none, 5.),
