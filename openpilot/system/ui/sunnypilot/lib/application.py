"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import os
import time

import pyray as rl

from openpilot.common.hardware import HARDWARE

SHOW_MOUSE_COORDS = os.getenv("SHOW_MOUSE_COORDS") == "1"
SUNNYPILOT_UI = os.getenv("SUNNYPILOT_UI", "1") == "1"

IDLE_FPS = 20
# contexts that need a fixed frame rate: offscreen/recorded rendering, mici's vblank-paced display
_FIXED_RATE = os.getenv("OFFSCREEN") == "1" or os.getenv("RECORD") == "1"


class GuiApplicationExt:
  def __init__(self):
    self._show_mouse_coords = SHOW_MOUSE_COORDS
    self._full_fps = 0

  def set_fps_idle(self, idle: bool):
    """Throttle to IDLE_FPS while onroad and untouched; camera content is 20Hz anyway.

    Keeps big-core load (and thermals) at the pre-60fps baseline for the ~95% of
    drive time the screen isn't being interacted with.
    """
    if not SUNNYPILOT_UI or _FIXED_RATE or HARDWARE.get_device_type() == 'mici':
      return
    if self._full_fps == 0:
      self._full_fps = self._target_fps
    want = min(IDLE_FPS, self._full_fps) if idle else self._full_fps
    if want != self._target_fps:
      rl.set_target_fps(want)
      self._target_fps = want
      self._last_fps_log_time = time.monotonic()  # grace so the drop monitor skips the rate change

  @staticmethod
  def sunnypilot_ui() -> bool:
    return SUNNYPILOT_UI

  def _draw_mouse_coordinates(self, font):
    coords_text = f"X:{int(rl.get_mouse_x())}, Y:{int(rl.get_mouse_y())}"

    green_color = rl.Color(0, 159, 47, 255)  # Match the green color of FPS counter

    # Calculate text width to position it at the right edge; estimate width based on text length
    # Each character is approximately 10-12 pixels wide at font size 20
    estimated_text_width = len(coords_text) * 11

    # Position text at the top right corner, 10px from the top
    screen_width = self._scaled_width if self._scale != 1.0 else self._width
    text_pos = rl.Vector2(screen_width - estimated_text_width - 10, 6)

    # Draw the text
    rl.draw_text_ex(font, coords_text, text_pos, 20, 0, green_color)

  def set_show_mouse_coords(self, show: bool):
    self._show_mouse_coords = show
