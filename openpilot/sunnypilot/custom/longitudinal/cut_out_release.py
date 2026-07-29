"""Cut-out lead release: stop braking for a lead that has left the path.

Route 261: the MPC kept commanding -0.5..-0.9 m/s^2 for a lead already 2-3 m
laterally off the driving path (turning off / exiting the lane) until radard
finally dropped the track, while the model had long recovered to positive accel.
Radar tracks are sticky laterally; there is no declamp between "leadOne exists"
and "leadOne is no longer in front of us".

This filter sits between radarState and ``mpc.update`` only (the custom stack,
follow-gap scheduler and telemetry keep the raw lead). When leadOne has been
confidently exiting sideways for a sustained window and is not a braking threat,
the MPC sees no lead-one and plans against lead-two / cruise instead.

Fail-closed: any error, any gate not met, or research actuation off returns the
raw radarState unchanged. Cut-ins are never suppressed (|yRel| shrinking resets
the exit timer).
"""
from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

from openpilot.sunnypilot.custom.longitudinal.lead_context import _path_relative_y_or_none

# A lead genuinely in our lane never sits this far off center (half lane ~1.6-1.85 m).
CUT_OUT_MIN_Y_REL = 2.0
# Beyond this it is unambiguously gone regardless of outward motion.
CUT_OUT_FAR_Y_REL = 3.0
# Outward motion requirement over the persistence window (m).
CUT_OUT_MIN_OUTWARD_DELTA = 0.2
# Sustained-exit persistence before suppression (s).
CUT_OUT_PERSIST_S = 0.4
# Never suppress a lead we would need to brake for soon.
CUT_OUT_MIN_TTC_S = 4.0
CUT_OUT_MIN_D_REL = 15.0
CUT_OUT_MIN_V_EGO = 3.0


def _f(value: Any, default: float = 0.0) -> float:
  try:
    v = float(value)
  except (TypeError, ValueError):
    return default
  return v if math.isfinite(v) else default


class CutOutLeadRelease:
  """Per-frame stateful gate deciding whether the MPC should ignore leadOne."""

  def __init__(self):
    self._track_id: int | None = None
    self._exit_timer_s = 0.0
    self._prev_abs_y_rel = 0.0
    self.suppressing = False
    self.block_reason = "init"

  def _reset(self, reason: str) -> None:
    self._track_id = None
    self._exit_timer_s = 0.0
    self._prev_abs_y_rel = 0.0
    self.suppressing = False
    self.block_reason = reason

  def filtered(self, radar_state: Any, v_ego: float, dt: float, *,
               long_active: bool, custom_long_enabled: bool,
               research_actuation_allowed: bool, mode: str = "off",
               model_msg: Any | None = None) -> Any:
    try:
      if mode != "apply" or not (long_active and custom_long_enabled and research_actuation_allowed):
        self._reset("gates_off")
        return radar_state
      lead = getattr(radar_state, "leadOne", None)
      if lead is None or not bool(getattr(lead, "present", False)):
        self._reset("no_lead")
        return radar_state

      d_rel = _f(getattr(lead, "dRel", 0.0))
      y_rel = _f(getattr(lead, "yRel", 0.0))
      v_rel = _f(getattr(lead, "vRel", 0.0))
      track_id = int(_f(getattr(lead, "radarTrackId", -1), -1.0))
      if track_id < 0:
        self._reset("track_unknown")
        return radar_state
      path_y_rel = _path_relative_y_or_none(y_rel, d_rel, model_msg)
      if path_y_rel is None:
        self._reset("path_unavailable")
        return radar_state
      abs_y = abs(path_y_rel)

      if track_id != self._track_id:
        self._track_id = track_id
        self._exit_timer_s = 0.0
        self._prev_abs_y_rel = abs_y

      closing = max(0.0, -v_rel)
      ttc = d_rel / closing if closing > 0.05 else math.inf
      threat = ttc < CUT_OUT_MIN_TTC_S or d_rel < CUT_OUT_MIN_D_REL or _f(v_ego) < CUT_OUT_MIN_V_EGO

      outward = abs_y >= self._prev_abs_y_rel - 0.05
      exiting = (not threat) and (
        abs_y >= CUT_OUT_FAR_Y_REL or (abs_y >= CUT_OUT_MIN_Y_REL and outward)
      )
      self._prev_abs_y_rel = abs_y

      if exiting:
        self._exit_timer_s += max(0.0, _f(dt))
      else:
        self._exit_timer_s = 0.0

      self.suppressing = self._exit_timer_s >= CUT_OUT_PERSIST_S
      self.block_reason = "" if self.suppressing else ("threat" if threat else "not_exiting")
      if not self.suppressing:
        return radar_state
      return SimpleNamespace(leadOne=SimpleNamespace(present=False),
                             leadTwo=getattr(radar_state, "leadTwo", SimpleNamespace(present=False)))
    except Exception:
      self._reset("fault")
      return radar_state
