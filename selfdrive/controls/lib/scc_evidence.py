from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
import math


class SccEvidenceTier(IntEnum):
  NONE = 0
  SLOWDOWN = 1
  STOP = 2
  URGENT_STOP = 3

  # Lowercase aliases match the contract vocabulary while preserving existing
  # uppercase imports used by mode resolver tests.
  none = 0
  slowdown = 1
  stop = 2
  urgent_stop = 3

  @property
  def label(self) -> str:
    return self.name.lower()


@dataclass(frozen=True)
class SccAdvisoryFlags:
  speed_limit_cap: bool = False
  map_caution: bool = False
  curve_cap: bool = False
  traffic_control_prior: bool = False

  @property
  def status(self) -> tuple[str, ...]:
    active: list[str] = []
    if self.map_caution:
      active.append("map_caution")
    if self.speed_limit_cap:
      active.append("speed_limit_cap")
    if self.curve_cap:
      active.append("curve_cap")
    if self.traffic_control_prior:
      active.append("traffic_control_prior")
    return tuple(active)


# Compatibility name used by existing imports/tests.
SccEvidenceAdvisory = SccAdvisoryFlags


@dataclass(frozen=True)
class SccEvidenceResult:
  tier: SccEvidenceTier = SccEvidenceTier.NONE
  confidence: float = 0.0
  urgency: float = 0.0
  reason: str = "scc_no_evidence"
  independent_of_lead: bool = False
  confirmed_lead: bool = False
  advisory: SccAdvisoryFlags = field(default_factory=SccAdvisoryFlags)
  model_stop_distance: float | None = None
  associated_lead_idx: int | None = None

  @property
  def advisories(self) -> SccAdvisoryFlags:
    return self.advisory

  @property
  def tier_label(self) -> str:
    return self.tier.label

  @property
  def e2e_active(self) -> bool:
    if self.tier == SccEvidenceTier.NONE:
      return False
    if not self.confirmed_lead:
      return True
    return bool(self.tier == SccEvidenceTier.URGENT_STOP and self.independent_of_lead)

  @property
  def advisory_status(self) -> tuple[str, ...]:
    return self.advisory.status


@dataclass(frozen=True)
class SccModeEvidence:
  confirmed_lead: bool = False
  model_stop: bool = False
  curve_control: bool = False
  map_control: bool = False
  speed_limit_control: bool = False
  traffic_control: bool = False
  model_slowdown: bool = False
  urgent_stop: bool = False
  independent_of_lead: bool = False
  confidence: float | None = None
  urgency: float | None = None
  evidence_reason: str = ""
  model_stop_distance: float | None = None
  lead_distance: float | None = None
  lead_path_y_rel: float = 0.0
  lead_idx: int | None = None
  v_ego: float = 0.0

  def classify(self) -> SccEvidenceResult:
    advisories = SccAdvisoryFlags(
      map_caution=bool(self.traffic_control),
      traffic_control_prior=bool(self.traffic_control),
      speed_limit_cap=bool(self.speed_limit_control),
      curve_cap=bool(self.curve_control or self.map_control),
    )
    return classify_scc_evidence(
      confirmed_lead=bool(self.confirmed_lead),
      model_stop=bool(self.model_stop),
      model_slowdown=bool(self.model_slowdown),
      urgent_stop=bool(self.urgent_stop),
      independent_of_lead=bool(self.independent_of_lead),
      confidence=self.confidence,
      urgency=self.urgency,
      reason=self.evidence_reason,
      advisories=advisories,
      model_stop_distance=self.model_stop_distance,
      lead_distance=self.lead_distance,
      lead_path_y_rel=self.lead_path_y_rel,
      lead_idx=self.lead_idx,
      v_ego=self.v_ego,
    )

  @property
  def e2e_active(self) -> bool:
    return self.classify().e2e_active

  @property
  def reason(self) -> str:
    classification = self.classify()
    if classification.e2e_active:
      return classification.reason
    if self.confirmed_lead:
      return "scc_confirmed_lead"
    if self.traffic_control:
      return "scc_traffic_control"
    if self.speed_limit_control:
      return "scc_speed_limit"
    if self.map_control:
      return "scc_map"
    if self.curve_control:
      return "scc_curve"
    return "scc_cruise"


def classify_scc_evidence(*, confirmed_lead: bool = False, model_stop: bool = False,
                          model_slowdown: bool = False, urgent_stop: bool = False,
                          independent_of_lead: bool = False, confidence: float | None = None,
                          urgency: float | None = None, reason: str = "",
                          advisories: SccAdvisoryFlags | None = None,
                          model_stop_distance: float | None = None,
                          lead_distance: float | None = None,
                          lead_path_y_rel: float = 0.0,
                          lead_idx: int | None = None,
                          v_ego: float = 0.0) -> SccEvidenceResult:
  advisories = advisories or SccAdvisoryFlags()
  model_stop_distance = _finite_optional(model_stop_distance)
  lead_distance = _finite_optional(lead_distance)

  if urgent_stop:
    tier = SccEvidenceTier.URGENT_STOP
    default_confidence = 1.0
    default_urgency = 1.0
    default_reason = "scc_urgent_stop"
  elif model_stop:
    tier = SccEvidenceTier.STOP
    default_confidence = 0.90
    default_urgency = 0.80
    default_reason = "scc_model_stop"
  elif model_slowdown:
    tier = SccEvidenceTier.SLOWDOWN
    default_confidence = 0.65
    default_urgency = 0.45
    default_reason = "scc_model_slowdown"
  else:
    return SccEvidenceResult(
      tier=SccEvidenceTier.NONE,
      confidence=0.0,
      urgency=0.0,
      reason=reason or _scc_advisory_reason(advisories),
      independent_of_lead=False,
      confirmed_lead=bool(confirmed_lead),
      advisory=advisories,
      model_stop_distance=model_stop_distance,
      associated_lead_idx=None,
    )

  associated_lead_idx = associate_model_stop_with_lead(
    confirmed_lead=confirmed_lead,
    model_stop_distance=model_stop_distance,
    lead_distance=lead_distance,
    lead_path_y_rel=lead_path_y_rel,
    lead_idx=lead_idx,
    v_ego=v_ego,
    confidence=_bounded_unit(confidence, default_confidence),
  )
  independent = _independent_stop_evidence(
    bool(confirmed_lead), bool(independent_of_lead), model_stop_distance, lead_distance, associated_lead_idx,
  )
  return SccEvidenceResult(
    tier=tier,
    confidence=_bounded_unit(confidence, default_confidence),
    urgency=_bounded_unit(urgency, default_urgency),
    reason=reason or default_reason,
    independent_of_lead=independent,
    confirmed_lead=bool(confirmed_lead),
    advisory=advisories,
    model_stop_distance=model_stop_distance,
    associated_lead_idx=associated_lead_idx,
  )


def associate_model_stop_with_lead(*, confirmed_lead: bool, model_stop_distance: float | None,
                                   lead_distance: float | None, lead_path_y_rel: float = 0.0,
                                   lead_idx: int | None = None, v_ego: float = 0.0,
                                   confidence: float = 1.0) -> int | None:
  if not confirmed_lead or model_stop_distance is None or lead_distance is None:
    return None
  if abs(float(lead_path_y_rel)) > 1.2:
    return None
  margin = _lead_association_margin(v_ego, confidence)
  if abs(model_stop_distance - lead_distance) <= margin:
    return 0 if lead_idx is None else int(lead_idx)
  return None


def _independent_stop_evidence(confirmed_lead: bool, explicit_independent: bool,
                               model_stop_distance: float | None, lead_distance: float | None,
                               associated_lead_idx: int | None) -> bool:
  if not confirmed_lead:
    return True
  if explicit_independent:
    return True
  if model_stop_distance is not None and lead_distance is not None:
    return associated_lead_idx is None
  return False


def _lead_association_margin(v_ego: float, confidence: float) -> float:
  speed_margin = max(0.0, float(v_ego)) * 0.25
  confidence_margin = (1.0 - _bounded_unit(confidence, 1.0)) * 2.0
  return max(3.0, speed_margin + confidence_margin)


def _bounded_unit(value: float | None, default: float) -> float:
  if value is None:
    value = default
  else:
    try:
      value = float(value)
    except (TypeError, ValueError):
      value = default
  if not math.isfinite(value):
    value = default
  return max(0.0, min(1.0, value))


def _finite_optional(value: float | None) -> float | None:
  if value is None:
    return None
  try:
    result = float(value)
  except (TypeError, ValueError):
    return None
  return result if math.isfinite(result) and result >= 0.0 else None


def _scc_advisory_reason(advisory: SccAdvisoryFlags) -> str:
  if advisory.traffic_control_prior:
    return "scc_traffic_control_prior"
  if advisory.map_caution:
    return "scc_map_caution"
  if advisory.speed_limit_cap:
    return "scc_speed_limit_cap"
  if advisory.curve_cap:
    return "scc_curve_cap"
  return "scc_no_evidence"
