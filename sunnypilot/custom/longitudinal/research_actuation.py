"""Research longitudinal actuation gate.

Computes the single default-off gate for non-baseline custom longitudinal research
actuation. All research apply paths route through this helper where CarParams is
available; shadow telemetry and the fork's baseline custom policy are unaffected.
"""
from __future__ import annotations

from typing import Any


def research_actuation_allowed(params: Any, CP: Any) -> bool:
  """True only when custom longitudinal is on, research actuation is enabled, and the car is
  under openpilot longitudinal control. Fail-closed on any param/CP read error."""
  try:
    if not getattr(CP, "openpilotLongitudinalControl", False):
      return False
    return bool(params.get_bool("CustomLongitudinalEnabled") and
                params.get_bool("AllowLongitudinalResearchActuation"))
  except Exception:
    return False
