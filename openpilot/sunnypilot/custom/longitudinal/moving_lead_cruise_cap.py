"""Bounded moving-lead cruise cap candidate for the MPC cruise obstacle."""
from __future__ import annotations

import math
from typing import Any

from openpilot.sunnypilot.custom.longitudinal.lead_context import lead_present

CAP_ALLOWANCE_M_S = 0.5
MIN_V_EGO_FOR_APPLY = 8.0
MIN_V_LEAD = 3.0
MIN_CLOSING = 0.3
MAX_CLOSING = 4.0
MIN_A_LEAD = -1.5
MAX_A_LEAD = -0.3
RADAR_MAX_A_LEAD = 0.0
MAX_ABS_Y_REL = 1.2
MIN_D_REL_M = 10.0
MIN_TIME_GAP_S = 1.2
MIN_TTC_S = 8.0
PARAMS_REFRESH_PERIOD = 50

MODE_OFF = "off"
MODE_SHADOW = "shadow"
MODE_APPLY = "apply"
VALID_MODES = {MODE_OFF, MODE_SHADOW, MODE_APPLY}


def _param_string(params: Any, key: str) -> str | None:
  try:
    raw = params.get(key)
  except TypeError:
    raw = params.get(key, None)
  if raw is None:
    return None
  if isinstance(raw, bytes):
    try:
      raw = raw.decode()
    except Exception:
      return None
  return str(raw)


def _is_finite(value: Any) -> bool:
  if value is None:
    return False
  try:
    return math.isfinite(float(value))
  except (TypeError, ValueError):
    return False


def _finite_float(value: Any) -> float | None:
  try:
    v = float(value)
  except (TypeError, ValueError):
    return None
  return v if math.isfinite(v) else None


class MovingLeadCruiseCap:
  def __init__(self, params: Any = None):
    self._params = params
    self._tick = 0
    self.mode = MODE_OFF
    self.enabled = False
    self.last_result: dict[str, Any] | None = None
    if params is not None:
      self.refresh_params()

  def refresh_params(self) -> None:
    if self._params is None:
      return
    try:
      mode_raw = (_param_string(self._params, "MovingLeadCruiseCapMode") or "").strip().lower()
      if mode_raw == "":
        self.mode = MODE_SHADOW
        self.enabled = False
      elif mode_raw in VALID_MODES:
        self.mode = mode_raw
        self.enabled = self.mode == MODE_APPLY
      else:
        self.mode = MODE_OFF
        self.enabled = False
    except Exception:
      self.mode = MODE_OFF
      self.enabled = False

  def capped(self, radarstate: Any, v_ego: float, v_cruise: float, dt: float, *,
             long_active: bool = False, brake_pressed: bool = False, gas_pressed: bool = False,
             force_decel: bool = False, custom_long_enabled: bool | None = None,
             research_actuation_allowed: bool = False) -> float:
    self._tick += 1
    if self._params is not None and self._tick % PARAMS_REFRESH_PERIOD == 0:
      self.refresh_params()

    try:
      raw_v_cruise = float(v_cruise)
    except (TypeError, ValueError):
      self.last_result = None
      return v_cruise

    base = raw_v_cruise if _is_finite(raw_v_cruise) and raw_v_cruise > 0.0 else 0.0
    if base <= 0.0:
      self.last_result = None
      return raw_v_cruise

    if custom_long_enabled is None:
      custom_long_enabled = False
      if self._params is not None:
        try:
          custom_long_enabled = bool(self._params.get_bool("CustomLongitudinalEnabled"))
        except Exception:
          custom_long_enabled = False
    else:
      custom_long_enabled = bool(custom_long_enabled)

    should_apply = (
      self.mode == MODE_APPLY and self.enabled and custom_long_enabled and research_actuation_allowed and long_active
    )
    should_shadow = self.mode in (MODE_SHADOW, MODE_APPLY)
    if not should_shadow:
      self.last_result = None
      return raw_v_cruise

    try:
      eligible, block_reason, capped_v_cruise = self._candidate(
        radarstate, v_ego, long_active, brake_pressed, gas_pressed, force_decel, base)
      self.last_result = {
        "mode": self.mode,
        "apply": bool(should_apply and capped_v_cruise < base),
        "eligible": eligible,
        "block_reason": block_reason,
        "v_cruise": base,
        "capped_v_cruise": capped_v_cruise,
      }
      return capped_v_cruise if should_apply else raw_v_cruise
    except Exception:
      self.last_result = None
      return raw_v_cruise

  def _candidate(self, radarstate: Any, v_ego: float, long_active: bool,
                 brake_pressed: bool, gas_pressed: bool, force_decel: bool, base_v_cruise: float) -> tuple[bool, str, float]:
    if not long_active:
      return False, "long_inactive", base_v_cruise
    if brake_pressed:
      return False, "brake_pressed", base_v_cruise
    if gas_pressed:
      return False, "gas_pressed", base_v_cruise
    if force_decel:
      return False, "force_decel", base_v_cruise
    if not _is_finite(v_ego):
      return False, "low_speed", base_v_cruise
    v_ego_f = float(v_ego)
    if v_ego_f < MIN_V_EGO_FOR_APPLY:
      return False, "low_speed", base_v_cruise

    lead_one = getattr(radarstate, "leadOne", None)
    if not lead_present(lead_one):
      return False, "no_lead", base_v_cruise

    d_rel = _finite_float(getattr(lead_one, "dRel", None))
    y_rel = _finite_float(getattr(lead_one, "yRel", None))
    v_rel = _finite_float(getattr(lead_one, "vRel", None))
    v_lead = _finite_float(getattr(lead_one, "vLead", None))
    a_lead = _finite_float(getattr(lead_one, "aLeadK", None))
    if d_rel is None or y_rel is None or v_rel is None or v_lead is None or a_lead is None:
      return False, "lead_non_finite", base_v_cruise

    if abs(y_rel) > MAX_ABS_Y_REL:
      return False, "lead_off_path", base_v_cruise

    if v_lead < MIN_V_LEAD:
      return False, "lead_slow", base_v_cruise

    if a_lead < MIN_A_LEAD:
      return False, "lead_hard_braking", base_v_cruise
    radar_confirmed = bool(getattr(lead_one, "radar", False))
    max_a_lead = RADAR_MAX_A_LEAD if radar_confirmed else MAX_A_LEAD
    if a_lead > max_a_lead:
      return False, "lead_not_braking", base_v_cruise

    if d_rel < max(MIN_D_REL_M, MIN_TIME_GAP_S * v_ego_f):
      return False, "too_close", base_v_cruise

    closing = max(0.0, -v_rel)
    if closing < MIN_CLOSING:
      return False, "not_closing", base_v_cruise
    if closing > MAX_CLOSING:
      return False, "fast_closing", base_v_cruise
    if d_rel / closing < MIN_TTC_S:
      return False, "low_ttc", base_v_cruise

    return True, "", min(base_v_cruise, v_lead + CAP_ALLOWANCE_M_S)
