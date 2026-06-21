from __future__ import annotations

from dataclasses import dataclass
from typing import Any

MODE_OFF = "off"
MODE_SHADOW = "shadow"
MODE_GATE = "gate"


@dataclass(frozen=True)
class StandstillReleaseConfidenceResult:
  mode: str = MODE_OFF
  effective_mode: str = MODE_OFF
  apply_supported: bool = False
  eligible: bool = False
  block_reason: str = "mode_off"
  confidence: float = 0.0
  release_allowed: bool = False
  release_source: str = ""
  release_reason: str = ""
  release_a_target: float = 0.0

  def debug_dict(self) -> dict[str, Any]:
    prefix = "standstill_release_confidence"
    return {
      f"{prefix}_mode": self.mode,
      f"{prefix}_effective_mode": self.effective_mode,
      f"{prefix}_apply_supported": self.apply_supported,
      f"{prefix}_eligible": self.eligible,
      f"{prefix}_block_reason": self.block_reason,
      f"{prefix}_confidence": self.confidence,
      f"{prefix}_release_allowed": self.release_allowed,
      f"{prefix}_release_source": self.release_source,
      f"{prefix}_release_reason": self.release_reason,
      f"{prefix}_release_a_target": self.release_a_target,
    }


def predict_standstill_release_confidence(*, mode: Any, release_allowed: bool, release_source: str,
                                          release_reason: str, release_a_target: float,
                                          lead_progress_allowed: bool, lead_gap_excess: float,
                                          lead_shadow_active: bool, alternate_threat_active: bool,
                                          force_slow_decel: bool, brake_pressed: bool, gas_pressed: bool,
                                          model_should_stop: bool) -> StandstillReleaseConfidenceResult:
  """Score the release pulse the existing stack already selected.

  Shadow mode must never create, increase, or veto a release. It only explains whether the
  existing decision looks well-supported.
  """
  mode_s = str(mode or "").strip().lower()
  if mode_s not in (MODE_OFF, MODE_SHADOW, MODE_GATE):
    mode_s = MODE_OFF
  if mode_s == MODE_OFF:
    return StandstillReleaseConfidenceResult(mode=MODE_OFF, effective_mode=MODE_OFF)
  apply_supported = mode_s == MODE_GATE

  confidence = 0.0
  if release_allowed:
    confidence += 0.35
  if lead_progress_allowed:
    confidence += 0.25
  if lead_gap_excess > 0.0:
    confidence += 0.10
  if not lead_shadow_active and not alternate_threat_active:
    confidence += 0.10
  if not (force_slow_decel or brake_pressed or gas_pressed or model_should_stop):
    confidence += 0.20
  confidence = max(0.0, min(1.0, confidence))

  if not release_allowed:
    block = "release_not_allowed"
  elif force_slow_decel or brake_pressed or gas_pressed:
    block = "driver_or_force_block"
  elif model_should_stop:
    block = "model_stop"
  elif lead_shadow_active or alternate_threat_active:
    block = "lead_threat"
  elif confidence < 0.6:
    block = "low_confidence"
  else:
    block = ""

  return StandstillReleaseConfidenceResult(
    mode=mode_s, effective_mode=mode_s, apply_supported=apply_supported,
    eligible=block == "", block_reason=block, confidence=confidence,
    release_allowed=bool(release_allowed), release_source=str(release_source or ""),
    release_reason=str(release_reason or ""), release_a_target=float(release_a_target),
  )
