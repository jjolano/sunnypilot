from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

MODE_OFF = "off"
MODE_SHADOW = "shadow"

PATH_NEAR_Y_M = 1.7
MAX_CLOSE_DISTANCE_M = 45.0
MAX_TTC_S = 5.0
MIN_CLOSING_SPEED_MS = 0.4
MIN_CONFIDENCE = 0.25
MAX_PROPOSED_DECEL = 2.5


@dataclass(frozen=True)
class CutInBrakeAssistResult:
  mode: str = MODE_OFF
  effective_mode: str = MODE_OFF
  apply_supported: bool = False
  eligible: bool = False
  block_reason: str = "mode_off"
  lead_idx: int = -1
  path_y_rel: float = 0.0
  lateral_velocity: float = 0.0
  ttc: float = 0.0
  required_decel: float = 0.0
  proposed_cap: float = 0.0
  confidence: float = 0.0

  def debug_dict(self) -> dict[str, Any]:
    prefix = "cut_in_brake_assist"
    return {
      f"{prefix}_mode": self.mode,
      f"{prefix}_effective_mode": self.effective_mode,
      f"{prefix}_apply_supported": self.apply_supported,
      f"{prefix}_eligible": self.eligible,
      f"{prefix}_block_reason": self.block_reason,
      f"{prefix}_lead_idx": self.lead_idx,
      f"{prefix}_path_y_rel": self.path_y_rel,
      f"{prefix}_lateral_velocity": self.lateral_velocity,
      f"{prefix}_ttc": self.ttc,
      f"{prefix}_required_decel": self.required_decel,
      f"{prefix}_proposed_cap": self.proposed_cap,
      f"{prefix}_confidence": self.confidence,
    }


def _f(value: Any, default: float = 0.0) -> float:
  try:
    v = float(value)
  except (TypeError, ValueError):
    return default
  return v if math.isfinite(v) else default


def _time_value(value: Any) -> float:
  v = _f(value, 0.0)
  return v if v > 0.0 else 0.0


def _primary(ctx: Any) -> Any | None:
  if ctx is None:
    return None
  return getattr(ctx, "behavior", None) or getattr(ctx, "physical", None)


def predict_cut_in_brake_assist(mode: Any, actual_ctx: Any | None, shadow_ctx: Any | None,
                                v_ego: float) -> CutInBrakeAssistResult:
  """Compute debug-only close cut-in brake-assist evidence.

  This helper never changes lead authority, MPC input, stop bits, or accel targets. The
  proposed cap is a route-log/debug value only.
  """
  mode_s = str(mode or "").strip().lower()
  if mode_s not in (MODE_OFF, MODE_SHADOW):
    mode_s = MODE_OFF
  if mode_s == MODE_OFF:
    return CutInBrakeAssistResult(mode=MODE_OFF, effective_mode=MODE_OFF)

  state = _primary(shadow_ctx) or _primary(actual_ctx)
  if state is None or not bool(getattr(state, "status", False)):
    return CutInBrakeAssistResult(mode=MODE_SHADOW, effective_mode=MODE_SHADOW, block_reason="no_lead")

  path_y_rel = _f(getattr(state, "path_y_rel", getattr(state, "y_rel", 0.0)))
  d_rel = _f(getattr(state, "d_rel", 0.0))
  v_rel = _f(getattr(state, "v_rel", 0.0))
  closing_speed = max(0.0, -v_rel)
  confidence = _f(getattr(state, "confidence", 0.0))
  risk_model = getattr(state, "risk_model", None)
  ttc = _time_value(getattr(risk_model, "ttc", getattr(state, "ttc", 0.0)))
  required_decel = max(0.0, _f(getattr(risk_model, "required_decel", getattr(state, "required_decel", 0.0))))

  if abs(path_y_rel) > PATH_NEAR_Y_M:
    block = "not_near_path"
  elif d_rel <= 0.0 or d_rel > MAX_CLOSE_DISTANCE_M:
    block = "not_close"
  elif closing_speed < MIN_CLOSING_SPEED_MS and (ttc <= 0.0 or ttc > MAX_TTC_S):
    block = "not_closing"
  elif confidence < MIN_CONFIDENCE:
    block = "low_confidence"
  else:
    block = ""

  eligible = block == ""
  proposed_decel = max(required_decel + 0.2, min(MAX_PROPOSED_DECEL, closing_speed * 0.25)) if eligible else 0.0
  proposed_cap = -min(MAX_PROPOSED_DECEL, max(0.0, proposed_decel)) if eligible else 0.0
  return CutInBrakeAssistResult(
    mode=MODE_SHADOW, effective_mode=MODE_SHADOW, apply_supported=False, eligible=eligible,
    block_reason=block, lead_idx=int(_f(getattr(state, "lead_idx", -1), -1)),
    path_y_rel=path_y_rel, lateral_velocity=0.0, ttc=ttc, required_decel=required_decel,
    proposed_cap=proposed_cap, confidence=confidence,
  )
