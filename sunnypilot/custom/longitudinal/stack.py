"""CustomLongitudinalStack — composition entry point for the custom-2.0 longitudinal policy.

Ties the longitudinal components together behind one update():

    leads -> lead confidence -> lead context (risk/progress)
                                      |
    planner baseline + model/SCC/map/speed evidence + mode + personality
                                      v
                          policy.build_candidates -> decision.decide -> a_target

This is the longitudinal analog of the lateral ``torque_v2_1`` adapter: plannerd builds the
pre-MPC target inputs (radarState leads, modelV2, SCC/map/speed-limit providers, and the
Longitudinal Mode / personality params) and calls ``update``. The custom policy shapes that
pre-MPC target; the MPC still owns final lead-follow physics and stop output downstream
(ADR 0001). End-to-end feel is validated against the engaged corpus; the composition itself is
integration-tested with fakes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from openpilot.sunnypilot.custom.longitudinal.acc_envelope import AccEnvelopeInputs, evaluate_acc_envelope
from openpilot.sunnypilot.custom.longitudinal.cut_in_brake_assist import predict_cut_in_brake_assist
from openpilot.sunnypilot.custom.longitudinal.curve_speed_confidence import CurveSpeedConfidenceInputs, predict_curve_speed_confidence
from openpilot.sunnypilot.custom.longitudinal.decision import Decision, decide
from openpilot.sunnypilot.custom.longitudinal.lead_confidence import LeadConfidenceState, LeadConfidenceTracker
from openpilot.sunnypilot.custom.longitudinal.lead_context import LeadContextTracker
from openpilot.sunnypilot.custom.longitudinal.lead_path_clearance import MODE_OFF as LEAD_PATH_CLEARANCE_MODE_OFF, predict_lead_path_clearance
from openpilot.sunnypilot.custom.longitudinal.standstill_release_confidence import predict_standstill_release_confidence
from openpilot.sunnypilot.custom.longitudinal.modes import EvidenceClass, LongitudinalMode, SourceToggles
from openpilot.sunnypilot.custom.longitudinal.policy import LongitudinalScene, build_candidates
from openpilot.sunnypilot.custom.longitudinal.policy_tables import Personality

import math

FOLLOW_TIME_GAP_S = 1.5   # steady-state follow time gap proxy
FOLLOW_GAP_MIN_M = 6.0


def _f(value: object, default: float = 0.0) -> float:
  try:
    v = float(value)  # type: ignore[arg-type]
  except (TypeError, ValueError):
    return default
  return v if math.isfinite(v) else default


def _raw_lead_kinematics_valid(lead0: Any) -> bool:
  """True only when the raw radarState lead0 kinematic fields are all present and finite."""
  for attr in ("dRel", "vLead", "vLeadK", "vRel"):
    v = getattr(lead0, attr, None)
    try:
      if v is None or not math.isfinite(float(v)):
        return False
    except (TypeError, ValueError):
      return False
  return True


@dataclass(frozen=True)
class LongitudinalStackInputs:
  v_ego: float
  a_ego: float = 0.0                   # current ego accel (wired for future smoothing; Phase 1 unused)
  v_cruise: float = 0.0
  seed_a_target: float = 0.0           # MPC/planner baseline accel
  accel_limits: tuple[float, float] = (-4.0, 2.0)
  accel_coast: float = 0.0
  leads: tuple[Any, Any] = (None, None)  # duck-typed radar/model leads (lead0, lead1)
  lead_a_target: float = 0.0           # lead-present pre-MPC seed accel; final MPC lead physics is downstream
  lead_should_stop: bool = False
  # model stop (E2E)
  model_should_stop: bool = False
  model_stop_distance: float | None = None
  model_desired_accel: float = 0.0
  model_stop_prob: float = 1.0   # model confidence in the stop (trust gate); 1.0 = fully trusted
  model_stale: bool = False
  stop_threat: bool = False
  # shadow path-relative lead context (telemetry only; not used for actuation)
  model_msg: Any | None = None
  lead_path_clearance_mode: Any = LEAD_PATH_CLEARANCE_MODE_OFF
  cut_in_brake_assist_mode: Any = "off"
  curve_speed_confidence_mode: Any = "off"
  standstill_release_confidence_mode: Any = "off"
  curve_confidence: CurveSpeedConfidenceInputs = field(default_factory=CurveSpeedConfidenceInputs)
  # advisory evidence
  speed_limit_active: bool = False
  speed_limit_v_target: float = 0.0
  speed_limit_a_target: float = 0.0
  curve_active: bool = False
  curve_a_target: float = 0.0
  curve_source: EvidenceClass = EvidenceClass.CURVE_VISION   # which SCC curve source bound the cap
  # driver / safety
  long_active: bool = False
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
  debug: dict[str, Any] = field(default_factory=dict)
  standstill_release_allowed: bool = False
  standstill_release_source: str = ""
  standstill_release_a_target: float = 0.0
  standstill_release_reason: str = ""


class CustomLongitudinalStack:
  def __init__(self) -> None:
    self._lead_confidence = (LeadConfidenceTracker(), LeadConfidenceTracker())
    self._lead_context = LeadContextTracker()
    self._shadow_lead_context = LeadContextTracker()

  def reset(self) -> None:
    self._lead_confidence = (LeadConfidenceTracker(), LeadConfidenceTracker())
    self._lead_context = LeadContextTracker()
    self._shadow_lead_context = LeadContextTracker()

  def update(self, inp: LongitudinalStackInputs, dt: float) -> LongitudinalStackResult:
    confidence_states = (
      self._lead_confidence[0].update(inp.leads[0], dt),
      self._lead_confidence[1].update(inp.leads[1], dt),
    )
    lead_ctx = self._lead_context.update(inp.leads, confidence_states, inp.v_ego, dt)

    # Shadow path-relative lead context is computed in an exception-isolated tracker for
    # telemetry/debug only. It must never change actuation or fail the adapter.
    path_shadow_model_path_available = False
    path_shadow_fault = False
    shadow_ctx = None
    shadow_debug: dict[str, Any] = {}
    try:
      path_shadow_model_path_available = _model_path_available(inp.model_msg)
      shadow_ctx = self._shadow_lead_context.update(inp.leads, confidence_states, inp.v_ego, dt, model_msg=inp.model_msg)
      shadow_debug = {f"path_shadow_{k}": v for k, v in shadow_ctx.debug_dict().items()}
    except Exception:
      path_shadow_fault = True

    # Lead path clearance is Phase 1 shadow-only telemetry. It must not feed lead
    # selection, stop commitment, or accel targets, and all failures are contained.
    lead_path_clearance_fault = False
    lead_path_clearance_debug: dict[str, Any] = {}
    try:
      clearance = predict_lead_path_clearance(inp.lead_path_clearance_mode, shadow_ctx, inp.model_msg, inp.v_ego)
      lead_path_clearance_debug = clearance.debug_dict()
    except Exception:
      lead_path_clearance_fault = True

    cut_in_brake_assist_fault = False
    cut_in_brake_assist_debug: dict[str, Any] = {}
    try:
      cut_in_brake_assist_debug = predict_cut_in_brake_assist(
        inp.cut_in_brake_assist_mode, lead_ctx, shadow_ctx, inp.v_ego,
        long_active=inp.long_active,
      ).debug_dict()
    except Exception:
      cut_in_brake_assist_fault = True

    curve_speed_confidence_fault = False
    curve_speed_confidence_debug: dict[str, Any] = {}
    try:
      curve_speed_confidence_debug = predict_curve_speed_confidence(
        inp.curve_speed_confidence_mode, inp.curve_confidence,
      ).debug_dict()
    except Exception:
      curve_speed_confidence_fault = True

    raw_lead_present = _any_status(inp.leads)
    lead_shadow_active = bool(getattr(lead_ctx, "shadow_active", False))
    alternate_threat_active = bool(getattr(lead_ctx, "alternate_threat_active", False))
    lead_threat_active = bool(getattr(lead_ctx, "has_physical_lead", False) or lead_shadow_active or alternate_threat_active)
    has_lead = bool(raw_lead_present or lead_threat_active)
    lead_progress_allowed = bool(getattr(lead_ctx, "lead_progress_allowed", False))
    lead_gap_excess = float(getattr(lead_ctx, "lead_gap_excess", 0.0) or 0.0)

    # Lead kinematics for the cushion / speedup guard / radar corroboration (from radarState).
    # Preserve raw validity *before* _f sanitizes missing/non-finite values so the policy can
    # reject softening on bad live data.
    lead0 = inp.leads[0]
    lead_kinematics_valid = True
    if has_lead and lead0 is not None:
      lead_kinematics_valid = _raw_lead_kinematics_valid(lead0)
    lead_v = _f(getattr(lead0, "vLeadK", getattr(lead0, "vLead", 0.0))) if has_lead else 0.0
    lead_d_rel = _f(getattr(lead0, "dRel", 0.0)) if has_lead else 0.0
    lead_v_rel = _f(getattr(lead0, "vRel", lead_v - inp.v_ego)) if has_lead else 0.0
    lead_a_k = _f(getattr(lead0, "aLeadK", 0.0)) if has_lead else 0.0
    follow_gap = max(FOLLOW_GAP_MIN_M, FOLLOW_TIME_GAP_S * max(0.0, inp.v_ego))
    primary_state = lead_ctx.behavior or lead_ctx.physical
    alignment_state = primary_state if primary_state is not None and primary_state.lead_idx == 0 else None
    lead_confidence = float(alignment_state.confidence) if alignment_state is not None else 0.0
    lead_stable = bool(alignment_state.stable) if alignment_state is not None else False

    scene = LongitudinalScene(
      v_ego=inp.v_ego, a_ego=inp.a_ego, v_cruise=inp.v_cruise, seed_a_target=inp.seed_a_target,
      accel_coast=inp.accel_coast, personality=inp.personality,
      has_lead=has_lead, lead_a_target=inp.lead_a_target, lead_should_stop=inp.lead_should_stop,
      lead_gap_excess=lead_gap_excess, lead_progress_allowed=lead_progress_allowed,
      lead_v=lead_v, lead_d_rel=lead_d_rel, lead_v_rel=lead_v_rel, lead_a_k=lead_a_k,
      follow_gap=follow_gap, lead_kinematics_valid=lead_kinematics_valid,
      lead_confidence=lead_confidence, lead_stable=lead_stable,
      lead_shadow_active=lead_shadow_active, alternate_threat_active=alternate_threat_active,
      model_should_stop=inp.model_should_stop, model_stop_distance=inp.model_stop_distance,
      model_desired_accel=inp.model_desired_accel, model_stop_prob=inp.model_stop_prob,
      model_stale=inp.model_stale,
      stop_threat=inp.stop_threat,
      speed_limit_active=inp.speed_limit_active, speed_limit_v_target=inp.speed_limit_v_target,
      speed_limit_a_target=inp.speed_limit_a_target,
      curve_active=inp.curve_active, curve_a_target=inp.curve_a_target, curve_source=inp.curve_source,
      force_slow_decel=inp.force_slow_decel, brake_pressed=inp.brake_pressed, gas_pressed=inp.gas_pressed,
    )
    candidates = build_candidates(scene)
    decision = decide(candidates, inp.mode, inp.accel_limits, inp.sources)
    acc_envelope_debug: dict[str, Any] = {}
    try:
      model_progress_candidate = str(decision.selected_intent) in ("no_lead_launch", "lead_pullaway", "lead_standstill_launch")
      acc_envelope_debug = evaluate_acc_envelope(AccEnvelopeInputs(
        v_ego=inp.v_ego,
        candidate_a_target=decision.a_target,
        previous_a_target=inp.seed_a_target,
        dt=dt,
        openpilot_longitudinal_control=True,
        has_lead=has_lead,
        lead_d_rel=lead_d_rel,
        lead_v_rel=lead_v_rel,
        lead_v_lead=lead_v,
        lead_a_lead_k=lead_a_k,
        lead_kinematics_valid=lead_kinematics_valid,
        model_stale=inp.model_stale,
        model_progress_candidate=model_progress_candidate,
        radar_stale=False,
        lead_required=has_lead,
      )).debug_dict()
    except Exception:
      acc_envelope_debug = {
        "acc_envelope_active": False,
        "acc_envelope_would_cap": True,
        "acc_envelope_cap_reason": "fault",
      }

    # Decision output is a pre-MPC target. Most lead-present braking seeds bind as hazards;
    # only explicitly approved low-risk soft cases raise the seed before the downstream MPC
    # solves final lead-follow physics.
    a_target = decision.a_target
    release_source = str(decision.selected_intent)
    lead_release_context = bool(release_source in ("lead_pullaway", "lead_standstill_launch")
                                and raw_lead_present and lead_progress_allowed
                                and not lead_shadow_active and not alternate_threat_active)
    clear_release_context = bool(release_source == "no_lead_launch" and not raw_lead_present and not lead_threat_active)
    standstill_release_allowed = bool(
      release_source in ("lead_pullaway", "lead_standstill_launch", "no_lead_launch")
      and (lead_release_context or clear_release_context)
      and decision.reason != "physical_hazard"
      and not decision.should_stop
      and a_target >= 0.15
      and not inp.force_slow_decel
      and not inp.brake_pressed
      and not inp.gas_pressed
      and not inp.model_should_stop
    )
    standstill_release_confidence_fault = False
    standstill_release_confidence_debug: dict[str, Any] = {}
    try:
      standstill_release_confidence_debug = predict_standstill_release_confidence(
        mode=inp.standstill_release_confidence_mode,
        release_allowed=standstill_release_allowed,
        release_source=str(decision.selected_intent if standstill_release_allowed else ""),
        release_reason=str(decision.reason if standstill_release_allowed else ""),
        release_a_target=float(max(a_target, 0.15)) if standstill_release_allowed else 0.0,
        lead_progress_allowed=lead_progress_allowed,
        lead_gap_excess=lead_gap_excess,
        lead_shadow_active=lead_shadow_active,
        alternate_threat_active=alternate_threat_active,
        force_slow_decel=inp.force_slow_decel,
        brake_pressed=inp.brake_pressed,
        gas_pressed=inp.gas_pressed,
        model_should_stop=inp.model_should_stop,
      ).debug_dict()
    except Exception:
      standstill_release_confidence_fault = True
    return LongitudinalStackResult(
      a_target=float(a_target),
      should_stop=bool(decision.should_stop),
      decision=decision,
      standstill_release_allowed=standstill_release_allowed,
      standstill_release_source=str(decision.selected_intent if standstill_release_allowed else ""),
      standstill_release_a_target=float(max(a_target, 0.15)) if standstill_release_allowed else 0.0,
      standstill_release_reason=str(decision.reason if standstill_release_allowed else ""),
      debug={
        "intent": decision.selected_intent,
        "reason": decision.reason,
        "has_lead": has_lead,
        "lead_progress_allowed": lead_progress_allowed,
        "lead_shadow_active": lead_shadow_active,
        "alternate_threat_active": alternate_threat_active,
        "n_candidates": len(candidates),
        "model_stale": bool(inp.model_stale),
        "rejected": decision.rejected,
        **{f"actual_{k}": v for k, v in lead_ctx.debug_dict().items()},
        "path_shadow_model_path_available": path_shadow_model_path_available,
        "path_shadow_fault": path_shadow_fault,
        **shadow_debug,
        "lead_path_clearance_fault": lead_path_clearance_fault,
        **lead_path_clearance_debug,
        "cut_in_brake_assist_fault": cut_in_brake_assist_fault,
        **cut_in_brake_assist_debug,
        "curve_speed_confidence_fault": curve_speed_confidence_fault,
        **curve_speed_confidence_debug,
        "standstill_release_confidence_fault": standstill_release_confidence_fault,
        **standstill_release_confidence_debug,
        **acc_envelope_debug,
      },
    )


def _any_status(leads: tuple[Any, Any]) -> bool:
  return any(lead is not None and bool(getattr(lead, "status", False)) for lead in leads)


def _model_path_available(model_msg: Any | None) -> bool:
  if model_msg is None:
    return False
  try:
    position = getattr(model_msg, "position", None)
    xs = getattr(position, "x", None) if position is not None else None
    ys = getattr(position, "y", None) if position is not None else None
    return bool(xs is not None and ys is not None and len(xs) >= 2 and len(xs) == len(ys))
  except Exception:
    return False
