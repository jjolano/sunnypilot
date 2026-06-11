from __future__ import annotations

from dataclasses import dataclass

from openpilot.selfdrive.controls.lib.longitudinal_decision import (
  CandidateRole,
  DecisionSource,
  LongitudinalCandidate,
)
from openpilot.selfdrive.controls.lib.longitudinal_stacks.custom_v2_debug import custom_v2_candidate_with_debug
from openpilot.selfdrive.controls.lib.longitudinal_stacks.interface import LongitudinalStackOutput


@dataclass(frozen=True)
class SignalProviderCandidate:
  source: DecisionSource
  role: CandidateRole
  target: tuple[float, float]
  active: bool
  confidence: float
  urgency: float
  active_reason: str

  def to_longitudinal_candidate(self) -> LongitudinalCandidate:
    v_target, a_target = self.target
    return LongitudinalCandidate(
      source=self.source,
      role=self.role,
      v_target=v_target,
      a_target=a_target,
      confidence=self.confidence,
      urgency=self.urgency,
      active_reason=self.active_reason,
      required_a_target=a_target if self.role == CandidateRole.ADVISORY_CAP else None,
    )


def build_sp_candidates_from_signal_providers(providers: tuple[SignalProviderCandidate, ...]) -> list[LongitudinalCandidate]:
  return [provider.to_longitudinal_candidate() for provider in providers if provider.active]


def build_sp_longitudinal_candidates(speed_limit_active, cruise, scc_vision, scc_vision_active, scc_map, scc_map_active,
                                     speed_limit_assist, osm_traffic_control, osm_traffic_control_active):
  return build_sp_candidates_from_signal_providers((
    SignalProviderCandidate(
      source=DecisionSource.CRUISE,
      role=CandidateRole.DRIVER_INTENT,
      target=cruise,
      active=True,
      confidence=1.0,
      urgency=0.1,
      active_reason="driver_cruise_target",
    ),
    SignalProviderCandidate(
      source=DecisionSource.SPEED_LIMIT,
      role=CandidateRole.ADVISORY_CAP,
      target=speed_limit_assist,
      active=speed_limit_active,
      confidence=0.85,
      urgency=0.35,
      active_reason="speed_limit_assist_active",
    ),
    SignalProviderCandidate(
      source=DecisionSource.SCC_VISION,
      role=CandidateRole.ADVISORY_CAP,
      target=scc_vision,
      active=scc_vision_active,
      confidence=0.80,
      urgency=0.45,
      active_reason="confident_vision_curve",
    ),
    SignalProviderCandidate(
      source=DecisionSource.SCC_MAP,
      role=CandidateRole.ADVISORY_CAP,
      target=scc_map,
      active=scc_map_active,
      confidence=0.80,
      urgency=0.40,
      active_reason="confident_map_curve",
    ),
    SignalProviderCandidate(
      source=DecisionSource.OSM_TRAFFIC_CONTROL,
      role=CandidateRole.ADVISORY_CAP,
      target=osm_traffic_control,
      active=osm_traffic_control_active,
      confidence=0.75,
      urgency=0.55,
      active_reason="model_confirmed_map_caution",
    ),
  ))


def replace_driver_intent(candidates: tuple[LongitudinalCandidate, ...],
                          driver: LongitudinalCandidate) -> tuple[LongitudinalCandidate, ...]:
  return (driver, *(candidate for candidate in candidates if candidate.role != CandidateRole.DRIVER_INTENT))


def ensure_driver_intent(candidates: tuple[LongitudinalCandidate, ...], fallback_output: LongitudinalStackOutput,
                         v_target: float) -> tuple[LongitudinalCandidate, ...]:
  if any(candidate.role == CandidateRole.DRIVER_INTENT for candidate in candidates):
    return candidates
  return (
    custom_v2_candidate_with_debug(
      LongitudinalCandidate(
        source=DecisionSource.CRUISE,
        role=CandidateRole.DRIVER_INTENT,
        v_target=max(0.0, float(v_target)),
        a_target=float(fallback_output.a_target),
        confidence=1.0,
        urgency=0.1,
        active_reason="driver_cruise_target",
        should_stop=bool(fallback_output.should_stop),
      ),
      intent="driver_cruise",
      reason="sunnypilot_current_seed",
      output=fallback_output,
    ),
    *candidates,
  )
