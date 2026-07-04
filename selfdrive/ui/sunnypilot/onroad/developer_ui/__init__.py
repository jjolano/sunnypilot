"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from enum import IntEnum
from collections import OrderedDict

import pyray as rl
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.selfdrive.ui.sunnypilot.onroad.developer_ui.elements import (
  UiElement, RelDistElement, RelSpeedElement, SteeringAngleElement,
  DesiredLateralAccelElement, ActualLateralAccelElement, DesiredSteeringAngleElement,
  AEgoElement, LeadSpeedElement, FrictionCoefficientElement, LatAccelFactorElement,
  SteeringTorqueEpsElement, BearingDegElement, AltitudeElement, DesiredSteeringPIDElement
)
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.widgets import Widget


def get_bottom_dev_ui_offset():
  if ui_state.developer_ui in (DeveloperUiState.BOTTOM, DeveloperUiState.BOTH):
    return 60
  return 0


class DeveloperUiState(IntEnum):
  OFF = 0
  BOTTOM = 1
  RIGHT = 2
  BOTH = 3


class DeveloperUiRenderer(Widget):
  def __init__(self):
    super().__init__()
    self._font_bold: rl.Font = gui_app.font(FontWeight.BOLD)
    self._font_semi_bold: rl.Font = gui_app.font(FontWeight.SEMI_BOLD)
    self.dev_ui_mode = DeveloperUiState.OFF
    self._text_width_cache: OrderedDict[tuple[int, str, int, float], float] = OrderedDict()
    self._element_measure_cache: dict[tuple[str, int], tuple[str, str, float, float, float, float]] = {}

    self.rel_dist_elem = RelDistElement()
    self.rel_speed_elem = RelSpeedElement()
    self.steering_angle_elem = SteeringAngleElement()
    self.desired_lat_accel_elem = DesiredLateralAccelElement()
    self.actual_lat_accel_elem = ActualLateralAccelElement()
    self.desired_steer_elem = DesiredSteeringAngleElement()
    self.desired_pid_steer_elem = DesiredSteeringPIDElement()
    self.a_ego_elem = AEgoElement()
    self.lead_speed_elem = LeadSpeedElement()
    self.friction_elem = FrictionCoefficientElement()
    self.lat_accel_factor_elem = LatAccelFactorElement()
    self.steering_torque_elem = SteeringTorqueEpsElement()
    self.bearing_elem = BearingDegElement()
    self.altitude_elem = AltitudeElement()

  def _measure_text_width(self, text: str, size: int, spacing: float = 0.0) -> float:
    key = (self._font_bold.texture.id, text, size, spacing)
    width = self._text_width_cache.get(key)
    if width is None:
      width = measure_text_cached(self._font_bold, text, size, spacing).x
      self._text_width_cache[key] = width
      if len(self._text_width_cache) > 128:
        self._text_width_cache.popitem(last=False)
    else:
      self._text_width_cache.move_to_end(key)
    return width

  def _measure_element(self, element: UiElement, font_size: int) -> UiElement:
    cache_key = (element.label, font_size)
    cached = self._element_measure_cache.get(cache_key)

    element.label_text = f"{element.label} "
    element.val_text = element.value
    element.unit_text = f" {element.unit}" if element.unit else ""

    if cached is not None and cached[0] == element.value and cached[1] == element.unit:
      element.label_width, element.val_width, element.unit_width, element.total_width = cached[2:]
      return element

    element.label_width = self._measure_text_width(element.label_text, font_size)
    element.val_width = self._measure_text_width(element.val_text, font_size)
    element.unit_width = self._measure_text_width(element.unit_text, font_size) if element.unit else 0
    element.total_width = element.label_width + element.val_width + element.unit_width
    self._element_measure_cache[cache_key] = (element.value, element.unit, element.label_width, element.val_width, element.unit_width, element.total_width)
    return element

  def _update_state(self) -> None:
    self.dev_ui_mode = ui_state.developer_ui

  def _render(self, rect: rl.Rectangle) -> None:
    if self.dev_ui_mode == DeveloperUiState.OFF:
      return

    sm = ui_state.sm
    if sm.recv_frame["carState"] < ui_state.started_frame:
      return

    if self.dev_ui_mode == DeveloperUiState.BOTTOM:
      self._draw_bottom_dev_ui(rect)
    elif self.dev_ui_mode == DeveloperUiState.RIGHT:
      self._draw_right_dev_ui(rect)
    elif self.dev_ui_mode == DeveloperUiState.BOTH:
      self._draw_right_dev_ui(rect)
      self._draw_bottom_dev_ui(rect)

  def _draw_right_dev_ui(self, rect: rl.Rectangle) -> None:
    sm = ui_state.sm
    controls_state = sm['controlsState']

    UI_BORDER_SIZE = 20
    container_width = 184
    x = int(rect.x + rect.width - container_width - UI_BORDER_SIZE * 2)
    y = int(rect.y + UI_BORDER_SIZE * 1.5)

    elements = [
      self.rel_dist_elem.update(sm, ui_state.is_metric),
      self.rel_speed_elem.update(sm, ui_state.is_metric),
      self.steering_angle_elem.update(sm, ui_state.is_metric),
    ]
    if controls_state.lateralControlState.which() == 'torqueState':
      elements.append(self.desired_lat_accel_elem.update(sm, ui_state.is_metric))
    elif controls_state.lateralControlState.which() == 'angleState':
      elements.append(self.desired_steer_elem.update(sm, ui_state.is_metric))
    elif controls_state.lateralControlState.which() == 'pidState':
      elements.append(self.desired_pid_steer_elem.update(sm, ui_state.is_metric))

    elements.append(self.actual_lat_accel_elem.update(sm, ui_state.is_metric))

    current_y = y
    for element in elements:
      current_y += self._draw_right_dev_ui_element(x, current_y, element)

  def _draw_right_dev_ui_element(self, x: int, y: int, element: UiElement) -> int:
    x += 0
    y += 230
    container_width = 184
    label_size = 28
    value_size = 60
    unit_size = 28
    label_width = self._measure_text_width(element.label, label_size, 0)
    centered_label_x = x + (container_width - label_width) / 2
    rl.draw_text_ex(self._font_bold, element.label, rl.Vector2(centered_label_x, y), label_size, 0, rl.WHITE)

    y += 45
    value_width = self._measure_text_width(element.value, value_size, 0)
    centered_value_x = x + (container_width - value_width) / 2
    rl.draw_text_ex(self._font_bold, element.value, rl.Vector2(centered_value_x, y), value_size, 0, element.color)

    if element.unit:
      units_height = self._measure_text_width(element.unit, unit_size, 0)

      units_x = x + container_width
      units_y = y + (value_size / 2) + (units_height / 2)

      rl.draw_text_pro(self._font_bold, element.unit, rl.Vector2(units_x, units_y), rl.Vector2(0, 0), -90.0, unit_size, 0, rl.WHITE)

    return 130

  def _draw_bottom_dev_ui(self, rect: rl.Rectangle) -> None:
    sm = ui_state.sm
    bar_height = 61
    y = int(rect.y + rect.height - bar_height)

    rl.draw_rectangle(int(rect.x), y, int(rect.width), bar_height,
                      rl.Color(0, 0, 0, 100))

    elements = [
      self.a_ego_elem.update(sm, ui_state.is_metric),
      self.lead_speed_elem.update(sm, ui_state.is_metric),
    ]

    # Add torque-specific elements if using torque control
    if sm['controlsState'].lateralControlState.which() == 'torqueState':
      override_active = ui_state.enforce_torque_control and ui_state.custom_torque_params and ui_state.torque_override_enabled
      if sm.valid['liveTorqueParameters'] or override_active:
        elements.extend([
          self.friction_elem.update(sm, ui_state.is_metric),
          self.lat_accel_factor_elem.update(sm, ui_state.is_metric),
        ])
    else:
      # Non-torque: show steering torque and GPS data
      elements.append(self.steering_torque_elem.update(sm, ui_state.is_metric))

      if sm.valid['gpsLocationExternal'] or sm.valid['gpsLocation']:
        elements.append(self.bearing_elem.update(sm, ui_state.is_metric))

    # Add altitude if GPS available
    if sm.valid['gpsLocationExternal'] or sm.valid['gpsLocation']:
      elements.append(self.altitude_elem.update(sm, ui_state.is_metric))

    if not elements:
      return

    font_size = 38
    element_widths = []
    for element in elements:
      self._measure_element(element, font_size)
      element_widths.append(element.total_width)

    total_element_width = sum(element_widths)
    num_gaps = len(elements) + 1
    available_width = rect.width
    gap_width = (available_width - total_element_width) / num_gaps

    center_y = y + bar_height // 2
    current_x = rect.x + gap_width

    for i, element in enumerate(elements):
      element_center_x = int(current_x + element_widths[i] / 2)
      self._draw_bottom_dev_ui_element(element_center_x, center_y, element)
      current_x += element_widths[i] + gap_width

  def _draw_bottom_dev_ui_element(self, center_x: int, y: int, element: UiElement) -> None:
    font_size = 38
    start_x = center_x - element.total_width / 2

    rl.draw_text_ex(self._font_bold, element.label_text, rl.Vector2(start_x, y - font_size // 2), font_size, 0, rl.WHITE)
    rl.draw_text_ex(self._font_bold, element.val_text, rl.Vector2(start_x + element.label_width, y - font_size // 2), font_size, 0, element.color)

    if element.unit:
      rl.draw_text_ex(self._font_bold, element.unit_text, rl.Vector2(start_x + element.label_width + element.val_width, y - font_size // 2),
                      font_size, 0, rl.WHITE)
