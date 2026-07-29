"""Typed evidence for the coast-only lead-departure prediction mode.

The lead context already builds the short-horizon prediction used by the longitudinal stack.
This module only validates and carries that prediction; it deliberately does not create a
second lead predictor.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from openpilot.sunnypilot.custom.longitudinal.lead_context import lead_present


MODE_OFF = "off"
MODE_SHADOW = "shadow"
MODE_APPLY = "apply"
VALID_MODES = frozenset((MODE_OFF, MODE_SHADOW, MODE_APPLY))

PHASE_INACTIVE = "inactive"
PHASE_ARMING = "arming"
PHASE_PREDICTED = "predicted"

PERSISTENCE_S = 0.20
TIMEOUT_S = 1.0
MIN_A_LEAD_K = 0.10
MIN_PREDICTED_GAP_GROWTH_M = 0.20


def sanitize_mode(value: Any) -> str:
  if isinstance(value, bytes):
    value = value.decode(errors="ignore")
  mode = str(value or "").strip().lower()
  return mode if mode in VALID_MODES else MODE_OFF


def effective_mode(mode: Any, research_actuation_allowed: bool) -> tuple[str, bool]:
  """Return the effective mode and whether its apply tier is supported."""
  mode_s = sanitize_mode(mode)
  if mode_s != MODE_APPLY or not bool(research_actuation_allowed):
    return (MODE_SHADOW if mode_s == MODE_APPLY and not research_actuation_allowed else mode_s, False)
  return MODE_APPLY, True


@dataclass(frozen=True)
class DeparturePredictionEvidence:
  """A validated snapshot of the selected physical lead's existing context prediction."""

  mode: str = MODE_OFF
  effective_mode: str = MODE_OFF
  apply_supported: bool = False
  research_actuation_allowed: bool = False
  eligible: bool = False
  block_reason: str = "mode_off"
  fault: bool = False
  lead_idx: int = -1
  track_id: int = -1
  stable: bool = False
  radar: bool = False
  shadow_active: bool = False
  alternate_threat_active: bool = False
  progress_authorized: bool = False
  prediction_valid: bool = False
  d_rel: float = 0.0
  v_lead: float = 0.0
  v_rel: float = 0.0
  a_lead_k: float = 0.0
  predicted_gap_1s: float = 0.0
  predicted_gap_growth_1s: float = 0.0
  predicted_gap_delta: float = 0.0

  def __post_init__(self) -> None:
    # Keep the concise trace vocabulary usable by callers while retaining the explicit
    # horizon-qualified field for runtime validation.
    if self.predicted_gap_growth_1s == 0.0 and self.predicted_gap_delta != 0.0:
      object.__setattr__(self, "predicted_gap_growth_1s", self.predicted_gap_delta)
    elif self.predicted_gap_delta == 0.0 and self.predicted_gap_growth_1s != 0.0:
      object.__setattr__(self, "predicted_gap_delta", self.predicted_gap_growth_1s)

  @property
  def predicted_gap_growth(self) -> float:
    return self.predicted_gap_growth_1s

  def debug_dict(self) -> dict[str, Any]:
    prefix = "departure_prediction"
    return {
      f"{prefix}_mode": self.mode,
      f"{prefix}_effective_mode": self.effective_mode,
      f"{prefix}_apply_supported": self.apply_supported,
      f"{prefix}_research_actuation_allowed": self.research_actuation_allowed,
      f"{prefix}_eligible": self.eligible,
      f"{prefix}_block_reason": self.block_reason,
      f"{prefix}_fault": self.fault,
      f"{prefix}_lead_idx": self.lead_idx,
      f"{prefix}_track_id": self.track_id,
      f"{prefix}_stable": self.stable,
      f"{prefix}_radar": self.radar,
      f"{prefix}_shadow_active": self.shadow_active,
      f"{prefix}_alternate_threat_active": self.alternate_threat_active,
      f"{prefix}_progress_authorized": self.progress_authorized,
      f"{prefix}_prediction_valid": self.prediction_valid,
      f"{prefix}_d_rel": self.d_rel,
      f"{prefix}_v_lead": self.v_lead,
      f"{prefix}_v_rel": self.v_rel,
      f"{prefix}_a_lead_k": self.a_lead_k,
      f"{prefix}_predicted_gap_1s": self.predicted_gap_1s,
      f"{prefix}_predicted_gap_growth_1s": self.predicted_gap_growth_1s,
      f"{prefix}_predicted_gap_delta": self.predicted_gap_delta,
    }


@dataclass(frozen=True)
class DeparturePredictionTrace:
  """Typed finalizer trace; this is never an actuation input."""

  mode: str = MODE_OFF
  effective_mode: str = MODE_OFF
  apply_supported: bool = False
  eligible: bool = False
  phase: str = PHASE_INACTIVE
  phase_s: float = 0.0
  evidence_s: float = 0.0
  age_s: float = 0.0
  lead_idx: int = -1
  track_id: int = -1
  pre_hold_active: bool = False
  post_hold_active: bool = False
  same_track: bool = False
  release_source: str = ""
  release_permission: bool = False
  release_mpc_stop: bool = False
  release_slew_provenance: bool = False
  measured_departure: bool = False
  threat_free: bool = False
  applied: bool = False
  predicted_gap_delta: float = 0.0
  would_coast: bool = False
  a_target_before: float = 0.0
  a_target_proposed: float = 0.0
  a_target_after: float = 0.0
  a_target_final: float = 0.0
  delta_a: float = 0.0
  research_actuation_allowed: bool = False
  block_reason: str = "mode_off"
  fault: bool = False

  def debug_dict(self) -> dict[str, Any]:
    prefix = "departure_prediction"
    return {
      f"{prefix}_mode": self.mode,
      f"{prefix}_effective_mode": self.effective_mode,
      f"{prefix}_apply_supported": self.apply_supported,
      f"{prefix}_eligible": self.eligible,
      f"{prefix}_phase": self.phase,
      f"{prefix}_phase_s": self.phase_s,
      f"{prefix}_evidence_s": self.evidence_s,
      f"{prefix}_age_s": self.age_s,
      f"{prefix}_lead_idx": self.lead_idx,
      f"{prefix}_track_id": self.track_id,
      f"{prefix}_pre_hold_active": self.pre_hold_active,
      f"{prefix}_post_hold_active": self.post_hold_active,
      f"{prefix}_same_track": self.same_track,
      f"{prefix}_release_source": self.release_source,
      f"{prefix}_release_permission": self.release_permission,
      f"{prefix}_release_mpc_stop": self.release_mpc_stop,
      f"{prefix}_release_slew_provenance": self.release_slew_provenance,
      f"{prefix}_measured_departure": self.measured_departure,
      f"{prefix}_threat_free": self.threat_free,
      f"{prefix}_applied": self.applied,
      f"{prefix}_predicted_gap_delta": self.predicted_gap_delta,
      f"{prefix}_would_coast": self.would_coast,
      f"{prefix}_a_target_before": self.a_target_before,
      f"{prefix}_a_target_proposed": self.a_target_proposed,
      f"{prefix}_a_target_after": self.a_target_after,
      f"{prefix}_a_target_final": self.a_target_final,
      f"{prefix}_delta_a": self.delta_a,
      f"{prefix}_research_actuation_allowed": self.research_actuation_allowed,
      f"{prefix}_block_reason": self.block_reason,
      f"{prefix}_fault": self.fault,
    }


def _finite(value: Any) -> bool:
  try:
    return math.isfinite(float(value))
  except (TypeError, ValueError):
    return False


def _int(value: Any, default: int = -1) -> int:
  try:
    return int(value)
  except (TypeError, ValueError):
    return default


def _prediction_finite(prediction: Any) -> bool:
  if prediction is None or not bool(getattr(prediction, "valid", False)):
    return False
  for name in ("x", "v", "a"):
    values = getattr(prediction, name, None)
    if values is None:
      return False
    try:
      if not values or any(not _finite(value) for value in values):
        return False
    except (TypeError, ValueError):
      return False
  return True


def build_departure_prediction_evidence(*, mode: Any, research_actuation_allowed: bool,
                                       physical_state: Any | None,
                                       physical_lead: Any | None,
                                       lead_context: Any | None) -> DeparturePredictionEvidence:
  """Validate one frame of existing physical-lead prediction/context evidence.

  The function is intentionally pure with respect to prediction: ``physical_state.prediction``
  is produced by ``LeadContextTracker`` and is the only prediction consumed here.
  """
  mode_s = sanitize_mode(mode)
  effective, apply_supported = effective_mode(mode_s, research_actuation_allowed)
  if mode_s == MODE_OFF:
    return DeparturePredictionEvidence()

  state = physical_state
  lead = physical_lead
  context = lead_context
  lead_idx = _int(getattr(state, "lead_idx", -1))
  state_track_id = _int(getattr(state, "track_id", -1))
  lead_track_id = _int(getattr(lead, "radarTrackId", -1)) if lead is not None else -1
  stable = bool(getattr(state, "stable", False)) if state is not None else False
  radar = bool(getattr(state, "radar", False)) if state is not None else False
  shadow_active = bool(getattr(context, "shadow_active", False)) if context is not None else False
  alternate_threat_active = bool(getattr(context, "alternate_threat_active", False)) if context is not None else False
  progress_authorized = bool(getattr(context, "lead_progress_allowed", False)) if context is not None else False

  values: dict[str, float] = {}
  try:
    if lead is not None:
      values = {
        "d_rel": float(lead.dRel),
        "v_lead": float(lead.vLead),
        "v_lead_k": float(lead.vLeadK),
        "v_rel": float(lead.vRel),
        "a_lead_k": float(lead.aLeadK),
      }
  except (AttributeError, TypeError, ValueError):
    values = {}

  d_rel = values.get("d_rel", 0.0)
  v_lead = values.get("v_lead", 0.0)
  v_rel = values.get("v_rel", 0.0)
  a_lead_k = values.get("a_lead_k", 0.0)
  prediction = getattr(state, "prediction", None) if state is not None else None
  prediction_valid = _prediction_finite(prediction)
  predicted_gap_1s = 0.0
  if prediction_valid:
    try:
      # LeadContextTracker's final preview sample is its fixed 1.0 s sample.
      prediction_x: Any = getattr(prediction, "x", ())
      predicted_gap_1s = float(prediction_x[-1])
    except (IndexError, TypeError, ValueError):
      prediction_valid = False
  predicted_gap_growth_1s = predicted_gap_1s - d_rel if prediction_valid and _finite(d_rel) else 0.0

  common: dict[str, Any] = dict(
    mode=mode_s,
    effective_mode=effective,
    apply_supported=apply_supported,
    research_actuation_allowed=bool(research_actuation_allowed),
    lead_idx=lead_idx,
    track_id=state_track_id if state_track_id >= 0 else lead_track_id,
    stable=stable,
    radar=radar,
    shadow_active=shadow_active,
    alternate_threat_active=alternate_threat_active,
    progress_authorized=progress_authorized,
    prediction_valid=prediction_valid,
    d_rel=d_rel if _finite(d_rel) else 0.0,
    v_lead=v_lead if _finite(v_lead) else 0.0,
    v_rel=v_rel if _finite(v_rel) else 0.0,
    a_lead_k=a_lead_k if _finite(a_lead_k) else 0.0,
    predicted_gap_1s=predicted_gap_1s if _finite(predicted_gap_1s) else 0.0,
    predicted_gap_growth_1s=predicted_gap_growth_1s if _finite(predicted_gap_growth_1s) else 0.0,
  )

  if state is None or not lead_present(lead):
    return DeparturePredictionEvidence(**common, block_reason="no_physical_lead")
  if not stable:
    return DeparturePredictionEvidence(**common, block_reason="unstable_lead")
  if not radar or state_track_id < 0 or lead_track_id < 0 or state_track_id != lead_track_id:
    return DeparturePredictionEvidence(**common, block_reason="unknown_radar_lead")
  if shadow_active or bool(getattr(state, "shadow", False)):
    return DeparturePredictionEvidence(**common, block_reason="shadow_threat")
  if alternate_threat_active:
    return DeparturePredictionEvidence(**common, block_reason="alternate_threat")
  if not progress_authorized:
    return DeparturePredictionEvidence(**common, block_reason="progress_not_authorized")
  if not values or not all(_finite(value) for value in values.values()) or d_rel <= 0.0:
    return DeparturePredictionEvidence(**common, block_reason="invalid_kinematics")
  if not prediction_valid:
    return DeparturePredictionEvidence(**common, block_reason="invalid_prediction")
  if a_lead_k <= MIN_A_LEAD_K:
    return DeparturePredictionEvidence(**common, block_reason="lead_accel_too_low")
  if v_rel < -0.05:
    return DeparturePredictionEvidence(**common, block_reason="lead_closing")
  if predicted_gap_growth_1s <= MIN_PREDICTED_GAP_GROWTH_M:
    return DeparturePredictionEvidence(**common, block_reason="insufficient_predicted_growth")
  return DeparturePredictionEvidence(**common, eligible=True, block_reason="")
