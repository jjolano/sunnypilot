"""Legacy facade for radar cut-in override.

Re-exports the canonical implementation from ``sunnypilot.custom.longitudinal.radar_cut_in``
so existing call sites and tests that import from ``sunnypilot.selfdrive.controls.lib.cut_in_override``
continue to work unchanged.
"""
from __future__ import annotations

from openpilot.sunnypilot.custom.longitudinal.radar_cut_in.override import (
  apply_cut_in_override,
  _is_high_risk_cut_in,
  _track_ttc,
  _MIN_CLOSING_SPEED,
  _MAX_D_REL,
  _MAX_TTC,
  _MAX_Y_REL,
  _MIN_V_LEAD,
  _MIN_TRACK_CNT,
  _MIN_V_EGO,
)

__all__ = [
  "apply_cut_in_override",
  "_is_high_risk_cut_in",
  "_track_ttc",
  "_MIN_CLOSING_SPEED",
  "_MAX_D_REL",
  "_MAX_TTC",
  "_MAX_Y_REL",
  "_MIN_V_LEAD",
  "_MIN_TRACK_CNT",
  "_MIN_V_EGO",
]
