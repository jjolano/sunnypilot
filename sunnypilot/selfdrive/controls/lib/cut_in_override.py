"""Cut-in override: promote high-risk on-path radar tracks to leadOne when the vision model
hasn't confirmed them yet.

The vision gate in ``radard.get_lead()`` requires ``lead_prob > 0.5`` before a radar track
becomes ``leadOne``. For cut-ins, the radar sees the track immediately but the vision model
takes 0.5–2.0 s to confirm it. This override bridges that gap: when no lead is found and a
radar track has high closing speed, is on-path, moving, and persistent, promote it to
``leadOne`` so the MPC can brake immediately.

Safety gates (all must pass):
  - No existing leadOne (status=False) — never override a confirmed lead
  - Track persistence (cnt >= 2) — not a one-frame ghost
  - On-path (abs(yRel) < 1.2 m) — not an adjacent-lane target
  - Moving object (vLead > 2.0 m/s) — not a stationary object
  - High closing speed (vRel < -2.0 m/s) — genuine cut-in threat
  - Within distance (dRel <= 30 m) — close enough to matter
  - Low TTC (dRel / closing_speed <= 8.0 s) — urgent enough to override
  - Custom longitudinal enabled — only active when custom policy is on

The override is fail-closed: any exception returns the original lead dict unchanged.
"""
from __future__ import annotations

import math
from typing import Any

from openpilot.common.params import Params


# Gate constants
_MIN_CLOSING_SPEED = 2.0        # m/s
_MAX_D_REL = 30.0               # m
_MAX_TTC = 8.0                  # s
_MAX_Y_REL = 1.2                # m (strict on-path)
_MIN_V_LEAD = 2.0               # m/s (moving object)
_MIN_TRACK_CNT = 2              # frames (persistence)
_MIN_V_EGO = 1.0               # m/s


def _is_high_risk_cut_in(track: Any, v_ego: float) -> bool:
  """Return True if a radar track is a high-risk cut-in candidate."""
  if v_ego < _MIN_V_EGO:
    return False
  if track.cnt < _MIN_TRACK_CNT:
    return False
  d_rel = float(getattr(track, "dRel", 0.0))
  y_rel = float(getattr(track, "yRel", 0.0))
  v_rel = float(getattr(track, "vRel", 0.0))
  v_lead = float(getattr(track, "vLead", 0.0))
  if not all(math.isfinite(x) for x in (d_rel, y_rel, v_rel, v_lead)):
    return False
  if d_rel <= 0 or d_rel > _MAX_D_REL:
    return False
  if abs(y_rel) > _MAX_Y_REL:
    return False
  if v_lead < _MIN_V_LEAD:
    return False
  closing_speed = -v_rel
  if closing_speed < _MIN_CLOSING_SPEED:
    return False
  ttc = d_rel / max(closing_speed, 0.1)
  if ttc > _MAX_TTC:
    return False
  return True


def _track_ttc(track: Any) -> float:
  """Compute TTC for a track (lower = more dangerous)."""
  d_rel = float(getattr(track, "dRel", 0.0))
  v_rel = float(getattr(track, "vRel", 0.0))
  closing = max(0.1, -v_rel)
  return d_rel / closing


def apply_cut_in_override(lead_dict: dict[str, Any], tracks: dict[int, Any],
                          v_ego: float, CP: Any = None, CP_SP: Any = None,
                          *, custom_longitudinal_enabled: bool | None = None) -> dict[str, Any]:
  """Promote a high-risk on-path radar track to leadOne when vision hasn't confirmed.

  Called after ``get_lead()`` returns. If the result has ``status=False`` (no lead found),
  scans radar tracks for a high-risk cut-in candidate. If found, returns that track's
  RadarState with a low modelProb (vision hasn't confirmed it).

  Args:
    lead_dict: The lead dict from ``get_lead()``.
    tracks: The radar tracks dict from ``RadarD.tracks``.
    v_ego: Current ego speed.
    CP: CarParams (unused, for signature compatibility).
    CP_SP: CarParamsSP (unused, for signature compatibility).
    custom_longitudinal_enabled: Optional cached flag from RadarD. None falls back to
      reading ``CustomLongitudinalEnabled`` from Params; False returns the lead unchanged.

  Returns:
    The original lead_dict if no override applies, or a new lead dict with the
    high-risk track promoted.
  """
  # Fail-closed: never override an existing confirmed lead
  if lead_dict.get("status", False):
    return lead_dict

  # Only active when custom longitudinal is enabled
  if custom_longitudinal_enabled is None:
    try:
      if not Params().get_bool("CustomLongitudinalEnabled"):
        return lead_dict
    except Exception:
      return lead_dict
  elif not custom_longitudinal_enabled:
    return lead_dict

  if not tracks or v_ego < _MIN_V_EGO:
    return lead_dict

  # Find the most dangerous cut-in candidate (lowest TTC)
  candidates = [t for t in tracks.values() if _is_high_risk_cut_in(t, v_ego)]
  if not candidates:
    return lead_dict

  best = min(candidates, key=_track_ttc)

  # Promote the track to leadOne with a low modelProb
  # (vision hasn't confirmed it, but the radar threat is real)
  try:
    override = best.get_RadarState(model_prob=0.0)
    # Mark as radar-only (not vision-confirmed)
    override["modelProb"] = 0.0
    override["radar"] = True
    return override
  except Exception:
    return lead_dict
