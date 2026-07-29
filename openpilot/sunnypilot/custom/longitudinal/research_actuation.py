"""Research longitudinal actuation gate.

Computes the single default-off gate for non-baseline custom longitudinal research
actuation. All research apply paths route through this helper where CarParams is
available; shadow telemetry and the fork's baseline custom policy are unaffected.
"""
from __future__ import annotations

from typing import Any


def research_actuation_allowed(params: Any, CP: Any, *,
                               custom_long_enabled: bool | None = None,
                               allow_longitudinal_research_actuation: bool | None = None) -> bool:
  """True only when custom longitudinal is on, research actuation is enabled, and the car is
  under openpilot longitudinal control. Cached booleans may be supplied to avoid Params reads.
  Fail-closed on any param/CP read error."""
  try:
    if not getattr(CP, "openpilotLongitudinalControl", False):
      return False
    if custom_long_enabled is None:
      custom_long_enabled = bool(params.get_bool("CustomLongitudinalEnabled"))
    else:
      custom_long_enabled = bool(custom_long_enabled)
    if allow_longitudinal_research_actuation is None:
      allow_longitudinal_research_actuation = bool(params.get_bool("AllowLongitudinalResearchActuation"))
    else:
      allow_longitudinal_research_actuation = bool(allow_longitudinal_research_actuation)
    return bool(custom_long_enabled and allow_longitudinal_research_actuation)
  except Exception:
    return False
