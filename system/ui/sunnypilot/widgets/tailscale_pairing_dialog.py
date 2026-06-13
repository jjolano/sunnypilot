"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

import pyray as rl
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.ui.widgets.pairing_dialog import PairingDialog
from openpilot.system.ui.lib.application import FontWeight, gui_app
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.lib.wrap_text import wrap_text
from openpilot.sunnypilot.system.tailscale.auth import clear_tailscale_auth_url


class TailscalePairingDialog(PairingDialog):
  """Full-screen dialog showing a QR code for Tailscale login."""

  QR_REFRESH_INTERVAL = 30  # Re-read param every 30s in case URL changes

  def __init__(self):
    PairingDialog.__init__(self)
    self._auth_url: str = ""
    self._raw_url_for_display: str = ""

  def _get_pairing_url(self) -> str:
    """Read the auth URL from the TailscaleAuthURL param."""
    try:
      url = self.params.get("TailscaleAuthURL") or ""
      if url:
        self._auth_url = url
        self._raw_url_for_display = url
      return self._auth_url
    except Exception:
      cloudlog.exception("tailscale: failed to read auth URL param")
      return self._auth_url

  def _update_state(self):
    state = self.params.get("TailscaleState") or ""
    if state.startswith("Running"):
      gui_app.pop_widget()
      return

    auth_url = self.params.get("TailscaleAuthURL") or ""
    if not auth_url and not self._auth_url:
      gui_app.pop_widget()

  def hide_event(self):
    clear_tailscale_auth_url(self.params)
    PairingDialog.hide_event(self)

  def _render(self, rect: rl.Rectangle) -> int:
    rl.clear_background(rl.Color(224, 224, 224, 255))

    self._check_qr_refresh()

    margin = 70
    content_rect = rl.Rectangle(rect.x + margin, rect.y + margin, rect.width - 2 * margin, rect.height - 2 * margin)
    y = content_rect.y

    close_size = 80
    pad = 20
    close_rect = rl.Rectangle(content_rect.x - pad, y - pad, close_size + pad * 2, close_size + pad * 2)
    self._close_btn.render(close_rect)

    y += close_size + 40

    title = tr("Login to Tailscale")
    title_font = gui_app.font(FontWeight.NORMAL)
    left_width = int(content_rect.width * 0.5 - 15)

    title_wrapped = wrap_text(title_font, title, 75, left_width)
    rl.draw_text_ex(title_font, "\n".join(title_wrapped), rl.Vector2(content_rect.x, y), 75, 0.0, rl.BLACK)
    y += len(title_wrapped) * 75 + 60

    remaining_height = content_rect.height - (y - content_rect.y)
    right_width = content_rect.width // 2 - 20

    self._render_instructions(rl.Rectangle(content_rect.x, y, left_width, remaining_height))

    qr_size = min(right_width, content_rect.height) - 40
    qr_x = content_rect.x + left_width + 40 + (right_width - qr_size) // 2
    qr_y = content_rect.y
    self._render_qr_code(rl.Rectangle(qr_x, qr_y, qr_size, qr_size))

    return -1

  def _render_instructions(self, rect: rl.Rectangle) -> None:
    instructions = [
      tr("Scan the QR code with your phone to authenticate with Tailscale"),
      tr("Or visit the URL below on any device to log in"),
      tr("Once authenticated, this dialog will close automatically"),
    ]

    font = gui_app.font(FontWeight.BOLD)
    y = rect.y

    for i, text in enumerate(instructions):
      circle_radius = 25
      circle_x = rect.x + circle_radius + 15
      text_x = rect.x + circle_radius * 2 + 40
      text_width = rect.width - (circle_radius * 2 + 40)

      wrapped = wrap_text(font, text, 47, int(text_width))
      text_height = len(wrapped) * 47
      circle_y = y + text_height // 2

      rl.draw_circle(int(circle_x), int(circle_y), circle_radius, rl.Color(70, 70, 70, 255))
      number = str(i + 1)
      number_size = measure_text_cached(font, number, 30)
      rl.draw_text_ex(font, number, (int(circle_x - number_size.x // 2), int(circle_y - number_size.y // 2)), 30, 0, rl.WHITE)

      rl.draw_text_ex(font, "\n".join(wrapped), rl.Vector2(text_x, y), 47, 0.0, rl.BLACK)
      y += text_height + 50

    if self._raw_url_for_display:
      y += 20
      url_font = gui_app.font(FontWeight.NORMAL)
      url_wrapped = wrap_text(url_font, self._raw_url_for_display, 32, int(rect.width - 30))
      rl.draw_text_ex(url_font, "\n".join(url_wrapped), rl.Vector2(rect.x + 15, y), 32, 0.0, rl.Color(30, 121, 232, 255))
