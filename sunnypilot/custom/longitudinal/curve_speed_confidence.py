from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

MODE_OFF = "off"
MODE_SHADOW = "shadow"
MODE_APPLY_CONSERVATIVE = "apply_conservative"


@dataclass(frozen=True)
class CurveSpeedConfidenceInputs:
  vision_active: bool = False
  vision_a_target: float = 0.0
  vision_state: Any = None
  vision_current_lat_acc: float = 0.0
  vision_max_pred_lat_acc: float = 0.0
  vision_pre_entry_active: bool = False
  map_active: bool = False
  map_a_target: float = 0.0
  map_state: Any = None
  map_target_lat: float = 0.0
  map_target_lon: float = 0.0


@dataclass(frozen=True)
class CurveSpeedConfidenceResult:
  mode: str = MODE_OFF
  effective_mode: str = MODE_OFF
  apply_supported: bool = False
  eligible: bool = False
  block_reason: str = "mode_off"
  confidence: float = 0.0
  proposed_cap: float = 0.0
  source: str = ""
  active: bool = False
  current_lat_acc: float = 0.0
  max_pred_lat_acc: float = 0.0
  pre_entry_active: bool = False

  def debug_dict(self) -> dict[str, Any]:
    prefix = "curve_speed_confidence"
    return {
      f"{prefix}_mode": self.mode,
      f"{prefix}_effective_mode": self.effective_mode,
      f"{prefix}_apply_supported": self.apply_supported,
      f"{prefix}_eligible": self.eligible,
      f"{prefix}_block_reason": self.block_reason,
      f"{prefix}_confidence": self.confidence,
      f"{prefix}_proposed_cap": self.proposed_cap,
      f"{prefix}_source": self.source,
      f"{prefix}_active": self.active,
      f"{prefix}_current_lat_acc": self.current_lat_acc,
      f"{prefix}_max_pred_lat_acc": self.max_pred_lat_acc,
      f"{prefix}_pre_entry_active": self.pre_entry_active,
    }


def _f(value: Any, default: float = 0.0) -> float:
  try:
    v = float(value)
  except (TypeError, ValueError):
    return default
  return v if math.isfinite(v) else default


def _state_name(state: Any) -> str:
  return str(getattr(state, "name", state) or "")


def predict_curve_speed_confidence(mode: Any, data: CurveSpeedConfidenceInputs) -> CurveSpeedConfidenceResult:
  mode_s = str(mode or "").strip().lower()
  if mode_s not in (MODE_OFF, MODE_SHADOW, MODE_APPLY_CONSERVATIVE):
    mode_s = MODE_OFF
  if mode_s == MODE_OFF:
    return CurveSpeedConfidenceResult(mode=MODE_OFF, effective_mode=MODE_OFF)
  apply_supported = mode_s == MODE_APPLY_CONSERVATIVE

  vision_a = _f(data.vision_a_target)
  caps: list[tuple[str, float]] = []
  if data.vision_active and vision_a < 0.0:
    caps.append(("vision", vision_a))
  if not caps:
    active = bool(data.vision_active or data.map_active)
    return CurveSpeedConfidenceResult(mode=mode_s, effective_mode=mode_s,
                                      apply_supported=apply_supported,
                                      block_reason="no_negative_curve_cap" if active else "inactive",
                                      active=active,
                                      current_lat_acc=_f(data.vision_current_lat_acc),
                                      max_pred_lat_acc=_f(data.vision_max_pred_lat_acc),
                                      pre_entry_active=bool(data.vision_pre_entry_active))

  source, proposed_cap = min(caps, key=lambda item: item[1])
  confidence = 0.45
  if data.vision_active and _f(data.vision_max_pred_lat_acc) >= 1.0:
    confidence = max(confidence, 0.70)
  if data.vision_pre_entry_active:
    confidence = max(confidence, 0.65)
  if data.map_active and math.isfinite(_f(data.map_target_lat)) and math.isfinite(_f(data.map_target_lon)):
    confidence = max(confidence, 0.60)
  if data.vision_active and data.map_active:
    confidence = max(confidence, 0.85)

  return CurveSpeedConfidenceResult(
    mode=mode_s, effective_mode=mode_s, apply_supported=apply_supported, eligible=True,
    block_reason="", confidence=confidence, proposed_cap=proposed_cap,
    source="+".join(name for name, _ in caps) or source, active=True,
    current_lat_acc=_f(data.vision_current_lat_acc), max_pred_lat_acc=_f(data.vision_max_pred_lat_acc),
    pre_entry_active=bool(data.vision_pre_entry_active),
  )
