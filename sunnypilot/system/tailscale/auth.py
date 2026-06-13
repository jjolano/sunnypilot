"""Small, UI-free helpers for Tailscale authentication state."""
from __future__ import annotations

from typing import Any

from openpilot.common.swaglog import cloudlog


def clear_tailscale_auth_url(params: Any) -> None:
  """Clear the one-shot Tailscale auth URL without importing the UI stack."""
  try:
    params.remove("TailscaleAuthURL")
  except Exception:
    cloudlog.exception("tailscale: failed to clear auth URL")
