from __future__ import annotations

from dataclasses import dataclass
import math
from types import SimpleNamespace

from openpilot.selfdrive.controls.lib.lead_confidence import LeadConfidenceState
from openpilot.selfdrive.controls.lib.lead_context import (
  LEAD_AUTHORITY_PHYSICAL,
  LEAD_AUTHORITY_SUPPRESS_ONLY,
  LeadContextTracker,
  PrimaryLeadContext,
)
from openpilot.selfdrive.controls.lib.longitudinal_decision import (
  CandidateRole,
  DecisionSource,
  LongitudinalArbiter,
  LongitudinalCandidate,
  LongitudinalDecision,
)
from openpilot.selfdrive.controls.lib.longitudinal_modes import (
  LONGITUDINAL_MODE_MIGRATION_PARAM,
  LONGITUDINAL_MODE_MIGRATION_VERSION,
  LONGITUDINAL_MODE_PARAM,
  LongitudinalMode,
  LongitudinalModeResolution,
  resolve_longitudinal_mode,
)
from openpilot.selfdrive.controls.lib.scc_evidence import SccModeEvidence


@dataclass(frozen=True)
class ScenarioLead:
  d_rel: float
  v_lead: float
  v_rel: float | None = None
  y_rel: float = 0.0
  a_lead: float = 0.0
  model_prob: float = 0.9
  radar: bool = True
  track_id: int = 1
  stable: bool = True
  new_lead: bool = False
  guard_timer: float = 0.0
  flicker_guard_timer: float = 0.0
  age: float = 1.0


@dataclass(frozen=True)
class PlannerScenarioFrame:
  name: str
  v_ego: float = 12.0
  dt: float = 0.2
  requested_mode: LongitudinalMode = LongitudinalMode.ACC
  driver_v_target: float = 20.0
  driver_a_target: float = 0.0
  lead0: ScenarioLead | None = None
  lead1: ScenarioLead | None = None
  model_stop: bool = False
  model_slowdown: bool = False
  urgent_stop: bool = False
  model_stop_distance: float | None = None
  independent_model_stop: bool = False
  evidence_confidence: float | None = None
  evidence_urgency: float | None = None
  speed_limit_v_target: float | None = None
  speed_limit_a_target: float | None = None
  curve_v_target: float | None = None
  curve_a_target: float | None = None
  map_v_target: float | None = None
  map_a_target: float | None = None
  traffic_control_v_target: float | None = None
  traffic_control_a_target: float | None = None
  lead_progress_a_target: float | None = None
  one_pedal_lift_off: bool = False
  reset_lead_context: bool = False


@dataclass(frozen=True)
class PlannerScenario:
  name: str
  frames: tuple[PlannerScenarioFrame, ...]


@dataclass(frozen=True)
class PlannerScenarioFrameResult:
  scenario: str
  frame: str
  frame_index: int
  requested_mode: str
  resolved_implementation: str
  actuation_type: str
  primary_physical_lead_idx: int | None
  primary_behavior_lead_idx: int | None
  alternate_threat_active: bool
  shadow_active: bool
  lead_progress_allowed: bool
  lead_release_blocked_reason: str
  mode_evidence_tier: str
  mode_evidence_reason: str
  selected_candidates: tuple[str, ...]
  rejected_candidates: tuple[str, ...]
  a_target: float
  should_stop: bool
  source: str
  stack_intent: str
  stack_reason: str


class _ScenarioParams:
  def __init__(self, mode: LongitudinalMode) -> None:
    self._values = {
      LONGITUDINAL_MODE_PARAM: int(mode),
      LONGITUDINAL_MODE_MIGRATION_PARAM: LONGITUDINAL_MODE_MIGRATION_VERSION,
    }

  def get(self, key: str, default: object | None = None) -> object | None:
    return self._values.get(key, default)

  def get_bool(self, key: str) -> bool:
    return bool(self._values.get(key, False))


def simulate_scenario(scenario: PlannerScenario) -> tuple[PlannerScenarioFrameResult, ...]:
  lead_tracker = LeadContextTracker()
  arbiter = LongitudinalArbiter()
  results: list[PlannerScenarioFrameResult] = []
  for idx, frame in enumerate(scenario.frames):
    if frame.reset_lead_context:
      lead_tracker.reset()
    lead_context = lead_tracker.update(
      (_lead_msg(frame.lead0, frame.v_ego), _lead_msg(frame.lead1, frame.v_ego)),
      (_confidence_state(frame.lead0), _confidence_state(frame.lead1)),
      frame.v_ego,
      frame.dt,
    )
    resolution = _mode_resolution(frame, lead_context)
    candidates = _candidates_for_frame(frame, lead_context, resolution)
    decision = arbiter.decide(candidates)
    results.append(_result_for_frame(scenario.name, idx, frame, lead_context, resolution, decision))
  return tuple(results)


def default_longitudinal_scenarios() -> tuple[PlannerScenario, ...]:
  return (
    PlannerScenario(
      "lead_cut_in_out_flicker",
      (
        PlannerScenarioFrame("clear", driver_a_target=0.2),
        PlannerScenarioFrame(
          "new_close_lead",
          v_ego=8.0,
          driver_a_target=0.2,
          lead0=ScenarioLead(d_rel=11.0, v_lead=1.0, v_rel=-7.0, new_lead=True, stable=False, guard_timer=0.25),
        ),
        PlannerScenarioFrame("shadow_after_drop", v_ego=8.0, driver_a_target=0.2),
      ),
    ),
    PlannerScenario(
      "stopped_lead_pullaway",
      (
        PlannerScenarioFrame(
          "opening_gap",
          v_ego=0.0,
          driver_a_target=0.0,
          lead0=ScenarioLead(d_rel=30.0, v_lead=2.0, v_rel=2.0, a_lead=0.2, track_id=2),
          lead_progress_a_target=0.35,
        ),
      ),
    ),
    PlannerScenario(
      "no_lead_model_stop",
      (
        PlannerScenarioFrame(
          "independent_stop",
          requested_mode=LongitudinalMode.E2E,
          model_stop=True,
          urgent_stop=True,
          independent_model_stop=True,
          model_stop_distance=18.0,
          driver_a_target=0.1,
        ),
      ),
    ),
    PlannerScenario(
      "false_model_stop_in_acc",
      (
        PlannerScenarioFrame(
          "acc_ignores_model_stop",
          requested_mode=LongitudinalMode.ACC,
          model_stop=True,
          model_stop_distance=22.0,
          driver_a_target=0.1,
        ),
      ),
    ),
    PlannerScenario(
      "speed_limit_drop",
      (
        PlannerScenarioFrame(
          "scc_speed_limit_cap",
          requested_mode=LongitudinalMode.SCC,
          speed_limit_v_target=12.0,
          speed_limit_a_target=-0.35,
          driver_a_target=0.0,
        ),
      ),
    ),
    PlannerScenario(
      "map_curve_false_positive",
      (
        PlannerScenarioFrame(
          "advisory_would_raise_accel",
          requested_mode=LongitudinalMode.SCC,
          map_v_target=10.0,
          map_a_target=0.25,
          driver_a_target=0.0,
        ),
      ),
    ),
    PlannerScenario(
      "vision_curve_true_positive",
      (
        PlannerScenarioFrame(
          "restrictive_curve",
          requested_mode=LongitudinalMode.SCC,
          curve_v_target=14.0,
          curve_a_target=-0.45,
          driver_a_target=0.0,
        ),
      ),
    ),
    PlannerScenario(
      "downhill_overspeed",
      (
        PlannerScenarioFrame(
          "overspeed_speed_limit_cap",
          requested_mode=LongitudinalMode.SCC,
          driver_a_target=-0.1,
          speed_limit_v_target=9.0,
          speed_limit_a_target=-0.6,
        ),
      ),
    ),
    PlannerScenario(
      "one_pedal_lift_off",
      (
        PlannerScenarioFrame(
          "lift_off_coast",
          requested_mode=LongitudinalMode.ACC,
          driver_a_target=0.25,
          one_pedal_lift_off=True,
        ),
      ),
    ),
  )


def _lead_msg(lead: ScenarioLead | None, v_ego: float) -> SimpleNamespace:
  if lead is None:
    return SimpleNamespace(status=False)
  v_rel = lead.v_lead - v_ego if lead.v_rel is None else lead.v_rel
  return SimpleNamespace(
    status=True,
    dRel=float(lead.d_rel),
    yRel=float(lead.y_rel),
    vRel=float(v_rel),
    aLeadK=float(lead.a_lead),
    vLead=float(lead.v_lead),
    vLeadK=float(lead.v_lead),
    modelProb=float(lead.model_prob),
    radar=bool(lead.radar),
    radarTrackId=int(lead.track_id),
  )


def _confidence_state(lead: ScenarioLead | None) -> LeadConfidenceState:
  if lead is None:
    return LeadConfidenceState(status=False)
  return LeadConfidenceState(
    status=True,
    new_lead=bool(lead.new_lead),
    stable=bool(lead.stable),
    speed_trusted=bool(lead.radar or lead.model_prob >= 0.5),
    radar=bool(lead.radar),
    age=float(lead.age),
    guard_timer=float(lead.guard_timer),
    flicker_guard_timer=float(lead.flicker_guard_timer),
    track_id=int(lead.track_id),
    d_rel=float(lead.d_rel),
    v_lead=float(lead.v_lead),
    y_rel=float(lead.y_rel),
  )


def _mode_resolution(frame: PlannerScenarioFrame, lead_context: PrimaryLeadContext) -> LongitudinalModeResolution:
  physical = lead_context.physical
  confirmed_lead = physical is not None and not physical.shadow
  evidence = SccModeEvidence(
    confirmed_lead=confirmed_lead,
    model_stop=bool(frame.model_stop),
    curve_control=frame.curve_a_target is not None,
    map_control=frame.map_a_target is not None,
    speed_limit_control=frame.speed_limit_a_target is not None,
    traffic_control=frame.traffic_control_a_target is not None,
    model_slowdown=bool(frame.model_slowdown),
    urgent_stop=bool(frame.urgent_stop),
    independent_of_lead=bool(frame.independent_model_stop),
    confidence=frame.evidence_confidence,
    urgency=frame.evidence_urgency,
    model_stop_distance=frame.model_stop_distance,
    lead_distance=None if physical is None else physical.d_rel,
    lead_path_y_rel=0.0 if physical is None else physical.path_y_rel,
    lead_idx=None if physical is None else physical.lead_idx,
    v_ego=frame.v_ego,
  )
  return resolve_longitudinal_mode(
    _ScenarioParams(frame.requested_mode),
    SimpleNamespace(openpilotLongitudinalControl=True, radarUnavailable=False),
    scc_evidence=evidence,
    restriction_status=evidence.classify().advisory_status,
  )


def _candidates_for_frame(frame: PlannerScenarioFrame, lead_context: PrimaryLeadContext,
                          resolution: LongitudinalModeResolution) -> tuple[LongitudinalCandidate, ...]:
  driver_a_target = 0.0 if frame.one_pedal_lift_off else frame.driver_a_target
  candidates = [
    LongitudinalCandidate(
      source=DecisionSource.CRUISE,
      role=CandidateRole.DRIVER_INTENT,
      v_target=frame.driver_v_target,
      a_target=driver_a_target,
      confidence=1.0,
      urgency=0.0,
      active_reason="one_pedal_lift_off" if frame.one_pedal_lift_off else "driver_cruise",
    )
  ]

  physical = lead_context.physical
  if physical is not None and physical.authority in (LEAD_AUTHORITY_PHYSICAL, LEAD_AUTHORITY_SUPPRESS_ONLY):
    candidates.append(LongitudinalCandidate(
      source=DecisionSource.LEAD_MPC,
      role=CandidateRole.PHYSICAL_HAZARD,
      v_target=min(frame.driver_v_target, max(0.0, physical.v_lead)),
      a_target=min(driver_a_target, -0.3 if physical.authority == LEAD_AUTHORITY_PHYSICAL else 0.0),
      confidence=max(0.60, physical.confidence),
      urgency=max(0.50, physical.risk_score),
      active_reason=physical.reason,
      should_stop=bool(physical.risk_model.stopped_or_crawling and physical.d_rel <= 15.0),
    ))

  if lead_context.lead_progress_allowed and frame.lead_progress_a_target is not None:
    candidates.append(LongitudinalCandidate(
      source=DecisionSource.STOP_LAUNCH,
      role=CandidateRole.RELAXATION,
      v_target=frame.driver_v_target,
      a_target=frame.lead_progress_a_target,
      confidence=0.85,
      urgency=0.25,
      active_reason="lead_progress_allowed",
      debug=lead_context.debug_dict(),
    ))

  if _model_stop_allowed(frame, resolution):
    candidates.append(LongitudinalCandidate(
      source=DecisionSource.E2E_STOP,
      role=CandidateRole.PHYSICAL_HAZARD,
      v_target=0.0,
      a_target=-1.8 if frame.urgent_stop else -0.8,
      confidence=0.95 if frame.urgent_stop else 0.85,
      urgency=1.0 if frame.urgent_stop else 0.75,
      active_reason="independent_model_stop" if frame.independent_model_stop else "model_stop",
      should_stop=True,
      horizon_distance=frame.model_stop_distance,
    ))

  if frame.requested_mode != LongitudinalMode.ACC:
    _append_advisory(candidates, DecisionSource.SPEED_LIMIT, frame.speed_limit_v_target, frame.speed_limit_a_target, "speed_limit_drop")
    _append_advisory(candidates, DecisionSource.SCC_VISION, frame.curve_v_target, frame.curve_a_target, "vision_curve_cap")
    _append_advisory(candidates, DecisionSource.SCC_MAP, frame.map_v_target, frame.map_a_target, "map_curve_cap")
    _append_advisory(
      candidates, DecisionSource.OSM_TRAFFIC_CONTROL, frame.traffic_control_v_target, frame.traffic_control_a_target,
      "traffic_control_prior",
    )

  return tuple(candidates)


def _append_advisory(candidates: list[LongitudinalCandidate], source: DecisionSource,
                     v_target: float | None, a_target: float | None, reason: str) -> None:
  if v_target is None or a_target is None:
    return
  candidates.append(LongitudinalCandidate(
    source=source,
    role=CandidateRole.ADVISORY_CAP,
    v_target=v_target,
    a_target=a_target,
    confidence=0.90,
    urgency=0.50,
    active_reason=reason,
    horizon_distance=max(1.0, v_target),
  ))


def _model_stop_allowed(frame: PlannerScenarioFrame, resolution: LongitudinalModeResolution) -> bool:
  if not frame.model_stop:
    return False
  if frame.requested_mode == LongitudinalMode.E2E:
    return True
  if frame.requested_mode != LongitudinalMode.SCC:
    return False
  return bool(resolution.scc_evidence.e2e_active)


def _result_for_frame(scenario: str, frame_index: int, frame: PlannerScenarioFrame,
                      lead_context: PrimaryLeadContext, resolution: LongitudinalModeResolution,
                      decision: LongitudinalDecision) -> PlannerScenarioFrameResult:
  selected = tuple(_candidate_label(candidate) for candidate in decision.candidates if _candidate_selected(candidate, decision))
  rejected = tuple(
    f"{suppressed.source.value}:{suppressed.suppression_reason}"
    for suppressed in decision.suppressed_candidates
  )
  selected_candidate = next((candidate for candidate in decision.candidates if _candidate_selected(candidate, decision)), None)
  return PlannerScenarioFrameResult(
    scenario=scenario,
    frame=frame.name,
    frame_index=frame_index,
    requested_mode=frame.requested_mode.name.lower(),
    resolved_implementation=resolution.resolved_implementation.value,
    actuation_type=resolution.actuation_type.value,
    primary_physical_lead_idx=lead_context.physical_idx,
    primary_behavior_lead_idx=lead_context.behavior_idx,
    alternate_threat_active=lead_context.alternate_threat_active,
    shadow_active=lead_context.shadow_active,
    lead_progress_allowed=lead_context.lead_progress_allowed,
    lead_release_blocked_reason=lead_context.lead_release_blocked_reason,
    mode_evidence_tier=resolution.scc_evidence.tier_label,
    mode_evidence_reason=resolution.scc_evidence.reason,
    selected_candidates=selected,
    rejected_candidates=rejected,
    a_target=_finite(decision.a_target),
    should_stop=bool(decision.should_stop),
    source=decision.winner.value,
    stack_intent=_stack_intent(selected_candidate),
    stack_reason=decision.active_reason or decision.fallback_reason,
  )


def _candidate_selected(candidate: LongitudinalCandidate, decision: LongitudinalDecision) -> bool:
  return bool(
    candidate.source == decision.winner and
    candidate.active_reason == decision.active_reason and
    math.isclose(candidate.a_target, decision.a_target, abs_tol=1e-6)
  )


def _candidate_label(candidate: LongitudinalCandidate) -> str:
  return f"{candidate.source.value}:{candidate.role.value}:{candidate.active_reason}"


def _stack_intent(candidate: LongitudinalCandidate | None) -> str:
  if candidate is None:
    return "fallback"
  if candidate.source == DecisionSource.LEAD_MPC:
    return "lead_follow"
  if candidate.source == DecisionSource.E2E_STOP:
    return "stop_approach"
  if candidate.source == DecisionSource.STOP_LAUNCH:
    return "launch"
  if candidate.source in (DecisionSource.SPEED_LIMIT, DecisionSource.SCC_MAP, DecisionSource.SCC_VISION, DecisionSource.OSM_TRAFFIC_CONTROL):
    return "advisory_cap"
  return "cruise"


def _finite(value: float) -> float:
  return float(value) if math.isfinite(float(value)) else 0.0
