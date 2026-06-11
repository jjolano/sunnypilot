from cereal import car

from openpilot.common.params import UnknownKeyName

from openpilot.selfdrive.controls.lib.longitudinal_stacks.custom_v2 import (
  ONE_PEDAL_MODE_OFF,
  ONE_PEDAL_MODES,
)

ButtonType = car.CarState.ButtonEvent.Type
ONE_PEDAL_LONGITUDINAL_MODE_PARAM = "OnePedalLongitudinalMode"
ONE_PEDAL_CRUISE_HOLD_BUTTON_TYPES = frozenset((
  ButtonType.accelCruise,
  ButtonType.decelCruise,
  ButtonType.resumeCruise,
  ButtonType.setCruise,
))


def get_one_pedal_longitudinal_mode(params) -> int:
  try:
    mode = int(params.get(ONE_PEDAL_LONGITUDINAL_MODE_PARAM, return_default=True))
  except (TypeError, ValueError, UnknownKeyName):
    return ONE_PEDAL_MODE_OFF
  return mode if mode in ONE_PEDAL_MODES else ONE_PEDAL_MODE_OFF


def one_pedal_cruise_hold_requested(button_events) -> bool:
  return any(getattr(event, "type", None) in ONE_PEDAL_CRUISE_HOLD_BUTTON_TYPES for event in button_events)


def update_one_pedal_cruise_hold(active: bool, button_events, gas_pressed: bool, brake_pressed: bool, enabled: bool) -> bool:
  if not enabled or gas_pressed or brake_pressed:
    return False
  return bool(active or one_pedal_cruise_hold_requested(button_events))
