"""CustomLongitudinalStack — composition entry point for the custom-2.0 longitudinal policy.

Ties the longitudinal components together behind one update():

    leads -> lead confidence -> lead context (risk/progress)
                                      |
    planner baseline + model/SCC/map/speed evidence + mode + personality
                                      v
                          policy.build_candidates -> decision.decide -> a_target

This is the longitudinal analog of the lateral ``torque_v2_1`` adapter: plannerd builds the
inputs (from the MPC baseline, radarState leads, modelV2, SCC/map/speed-limit providers, and
the Longitudinal Mode / personality params) and calls ``update``. The custom policy shapes
the planner's a_target within the MPC's physical envelope; the MPC keeps lead-follow physics
(ADR 0001). End-to-end feel is validated against the engaged corpus; the composition itself
is integration-tested with fakes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from openpilot.sunnypilot.custom.longitudinal.decision import Decision, decide
from openpilot.sunnypilot.custom.longitudinal.lead_confidence import LeadConfidenceState, LeadConfidenceTracker
from openpilot.sunnypilot.custom.longitudinal.lead_context import LeadContextTracker
from openpilot.sunnypilot.custom.longitudinal.modes import LongitudinalMode, SourceToggles
from openpilot.sunnypilot.custom.longitudinal.policy import LongitudinalScene, build_candidates
from openpilot.sunnypilot.custom.longitudinal.policy_tables import Personality


@dataclass(frozen=True)
class LongitudinalStackInputs:
  v_ego: float
  v_cruise: float
  seed_a_target: float                 # MPC/planner baseline accel
  accel_limits: tuple[float, float]
  accel_coast: float = 0.0
  leads: tuple[Any, Any] = (None, None)  # duck-typed radar/model leads (lead0, lead1)
  lead_a_target: float = 0.0           # MPC lead-follow accel (physics owned by MPC)
  lead_should_stop: bool = False
  # model stop (E2E)
  model_should_stop: bool = False
  model_stop_distance: float | None = None
  model_desired_accel: float = 0.0
  stop_threat: bool = False
  # advisory evidence
  speed_limit_active: bool = False
  speed_limit_v_target: float = 0.0
  speed_limit_a_target: float = 0.0
  map_caution_active: bool = False
  map_caution_confirmed: bool = False
  map_caution_a_target: float = 0.0
  curve_active: bool = False
  curve_a_target: float = 0.0
  # driver / safety
  force_slow_decel: bool = False
  brake_pressed: bool = False
  gas_pressed: bool = False
  # mode / personality
  mode: LongitudinalMode = LongitudinalMode.ACC
  sources: SourceToggles = SourceToggles()
  personality: Personality = Personality.STANDARD


@dataclass(frozen=True)
class LongitudinalStackResult:
  a_target: float
  should_stop: bool
  decision: Decision
  debug: dict[str, Any]


class CustomLongitudinalStack:
  def __init__(self) -> None:
    self._lead_confidence = (LeadConfidenceTracker(), LeadConfidenceTracker())
    self._lead_context = LeadContextTracker()

  def reset(self) -> None:
    self._lead_confidence = (LeadConfidenceTracker(), LeadConfidenceTracker())
    self._lead_context = LeadContextTracker()

  def update(self, inp: LongitudinalStackInputs, dt: float) -> LongitudinalStackResult:
    confidence_states = (
      self._lead_confidence[0].update(inp.leads[0], dt),
      self._lead_confidence[1].update(inp.leads[1], dt),
    )
    lead_ctx = self._lead_context.update(inp.leads, confidence_states, inp.v_ego, dt)
    has_lead = bool(getattr(lead_ctx, "has_lead", False) or _any_status(inp.leads))
    lead_progress_allowed = bool(getattr(lead_ctx, "lead_progress_allowed", False))
    lead_gap_excess = float(getattr(lead_ctx, "lead_gap_excess", 0.0) or 0.0)

    scene = LongitudinalScene(
      v_ego=inp.v_ego, v_cruise=inp.v_cruise, seed_a_target=inp.seed_a_target,
      accel_coast=inp.accel_coast, personality=inp.personality,
      has_lead=has_lead, lead_a_target=inp.lead_a_target, lead_should_stop=inp.lead_should_stop,
      lead_gap_excess=lead_gap_excess, lead_progress_allowed=lead_progress_allowed,
      model_should_stop=inp.model_should_stop, model_stop_distance=inp.model_stop_distance,
      model_desired_accel=inp.model_desired_accel, stop_threat=inp.stop_threat,
      speed_limit_active=inp.speed_limit_active, speed_limit_v_target=inp.speed_limit_v_target,
      speed_limit_a_target=inp.speed_limit_a_target,
      map_caution_active=inp.map_caution_active, map_caution_confirmed=inp.map_caution_confirmed,
      map_caution_a_target=inp.map_caution_a_target,
      curve_active=inp.curve_active, curve_a_target=inp.curve_a_target,
      force_slow_decel=inp.force_slow_decel, brake_pressed=inp.brake_pressed, gas_pressed=inp.gas_pressed,
    )
    candidates = build_candidates(scene)
    decision = decide(candidates, inp.mode, inp.accel_limits, inp.sources)

    # The custom policy never relaxes the MPC's physical envelope: clamp to the seed when the
    # seed is more conservative than a non-hazard policy choice would allow.
    a_target = decision.a_target
    return LongitudinalStackResult(
      a_target=float(a_target),
      should_stop=bool(decision.should_stop),
      decision=decision,
      debug={
        "intent": decision.selected_intent,
        "reason": decision.reason,
        "has_lead": has_lead,
        "lead_progress_allowed": lead_progress_allowed,
        "n_candidates": len(candidates),
        "rejected": decision.rejected,
      },
    )


def _any_status(leads: tuple[Any, Any]) -> bool:
  return any(lead is not None and bool(getattr(lead, "status", False)) for lead in leads)
