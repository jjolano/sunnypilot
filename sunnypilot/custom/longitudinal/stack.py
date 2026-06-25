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

from dataclasses import dataclass, field, replace
from typing import Any

from openpilot.sunnypilot.custom.longitudinal.acc_envelope import AccEnvelopeInputs, evaluate_acc_envelope
from openpilot.sunnypilot.custom.longitudinal.cut_in_brake_assist import predict_cut_in_brake_assist
from openpilot.sunnypilot.custom.longitudinal.curve_speed_confidence import CurveSpeedConfidenceInputs, predict_curve_speed_confidence
from openpilot.sunnypilot.custom.longitudinal.decision import CandidateRole, Decision, decide
from openpilot.sunnypilot.custom.longitudinal.lead_confidence import LeadConfidenceState, LeadConfidenceTracker
from openpilot.sunnypilot.custom.longitudinal.lead_context import LeadContextTracker
from openpilot.sunnypilot.custom.longitudinal.lead_path_clearance import MODE_OFF as LEAD_PATH_CLEARANCE_MODE_OFF, predict_lead_path_clearance
from openpilot.sunnypilot.custom.longitudinal.standstill_release_confidence import predict_standstill_release_confidence
from openpilot.sunnypilot.custom.longitudinal.modes import EvidenceClass, LongitudinalMode, SourceToggles, admitted_evidence
from openpilot.sunnypilot.custom.longitudinal.policy import LongitudinalScene, build_candidates
from openpilot.sunnypilot.custom.longitudinal.policy_tables import Personality
from openpilot.sunnypilot.custom.longitudinal.scenario_context import predict_scenario_context

import math

FOLLOW_TIME_GAP_S = 1.5   # steady-state follow time gap proxy
FOLLOW_GAP_MIN_M = 6.0
UPWARD_TARGET_SLEW_MAX_DT_S = 0.20
UPWARD_TARGET_SLEW_MAX_LAG = 0.50
DOWNWARD_TARGET_SLEW_JERK = -4.0  # faster comfort decel smoothing; never used for hazards
DOWNWARD_TARGET_SMOOTH_MAX_DELTA = 0.30
DOWNWARD_TARGET_SMOOTH_MIN_RAW_A = -0.40
DOWNWARD_TARGET_SMOOTH_CLOSING_V_REL = -0.50
DOWNWARD_TARGET_SMOOTH_LEAD_A_K = -0.50
DOWNWARD_TARGET_SMOOTH_RISK_REASONS = frozenset((
  "inside_time_gap",
  "ttc_low",
  "closing_decel_high",
  "invalid_lead_kinematics",
  "invalid_lead_distance",
  "invalid_data",
  "fault",
))


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


def _sanitize_inputs_for_mode(inp: LongitudinalStackInputs) -> LongitudinalStackInputs:
  """Return a sanitized copy of inputs with mode-excluded evidence neutralized.

  Raw ``inp`` is left untouched for telemetry/shadow/debug paths.
  """
  admitted = admitted_evidence(inp.mode, inp.sources)
  updates: dict[str, Any] = {}

  if EvidenceClass.MODEL_STOP not in admitted:
    updates.update(
      model_should_stop=False,
      model_stop_distance=None,
      model_desired_accel=0.0,
      model_stale=False,
      stop_threat=False,
      model_stop_prob=1.0,  # neutral; not consulted when MODEL_STOP is excluded
    )

  if EvidenceClass.SPEED_LIMIT not in admitted:
    updates.update(
      speed_limit_active=False,
      speed_limit_v_target=0.0,
      speed_limit_a_target=0.0,
    )

  curve_confidence_updates: dict[str, Any] = {}
  if EvidenceClass.CURVE_VISION not in admitted:
    curve_confidence_updates.update(
      vision_active=False,
      vision_a_target=0.0,
      vision_current_lat_acc=0.0,
      vision_max_pred_lat_acc=0.0,
      vision_pre_entry_active=False,
    )
  if EvidenceClass.CURVE_MAP not in admitted:
    curve_confidence_updates.update(
      map_active=False,
      map_a_target=0.0,
      map_target_lat=0.0,
      map_target_lon=0.0,
    )
  if curve_confidence_updates:
    updates["curve_confidence"] = replace(inp.curve_confidence, **curve_confidence_updates)

  if inp.curve_source not in admitted:
    updates.update(
      curve_active=False,
      curve_a_target=0.0,
    )

  if updates:
    return replace(inp, **updates)
  return inp


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
  scenario_context_mode: Any = "off"
  standstill: bool = False
  steering_angle_deg: float = 0.0
  steering_torque: float = 0.0
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
    self._prev_smoothed_a_target: float | None = None

  def reset(self) -> None:
    self._lead_confidence = (LeadConfidenceTracker(), LeadConfidenceTracker())
    self._lead_context = LeadContextTracker()
    self._shadow_lead_context = LeadContextTracker()
    self._prev_smoothed_a_target = None

  def update(self, inp: LongitudinalStackInputs, dt: float) -> LongitudinalStackResult:
    confidence_states = (
      self._lead_confidence[0].update(inp.leads[0], dt),
      self._lead_confidence[1].update(inp.leads[1], dt),
    )
    lead_ctx = self._lead_context.update(inp.leads, confidence_states, inp.v_ego, dt)
    act_inp = _sanitize_inputs_for_mode(inp)

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
        inp.curve_speed_confidence_mode, act_inp.curve_confidence,
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
      v_ego=act_inp.v_ego, a_ego=act_inp.a_ego, v_cruise=act_inp.v_cruise, seed_a_target=act_inp.seed_a_target,
      accel_coast=act_inp.accel_coast, personality=act_inp.personality,
      has_lead=has_lead, lead_a_target=act_inp.lead_a_target, lead_should_stop=act_inp.lead_should_stop,
      lead_gap_excess=lead_gap_excess, lead_progress_allowed=lead_progress_allowed,
      lead_v=lead_v, lead_d_rel=lead_d_rel, lead_v_rel=lead_v_rel, lead_a_k=lead_a_k,
      follow_gap=follow_gap, lead_kinematics_valid=lead_kinematics_valid,
      lead_confidence=lead_confidence, lead_stable=lead_stable,
      lead_shadow_active=lead_shadow_active, alternate_threat_active=alternate_threat_active,
      model_should_stop=act_inp.model_should_stop, model_stop_distance=act_inp.model_stop_distance,
      model_desired_accel=act_inp.model_desired_accel, model_stop_prob=act_inp.model_stop_prob,
      model_stale=act_inp.model_stale,
      stop_threat=act_inp.stop_threat,
      speed_limit_active=act_inp.speed_limit_active, speed_limit_v_target=act_inp.speed_limit_v_target,
      speed_limit_a_target=act_inp.speed_limit_a_target,
      curve_active=act_inp.curve_active, curve_a_target=act_inp.curve_a_target, curve_source=act_inp.curve_source,
      force_slow_decel=act_inp.force_slow_decel, brake_pressed=act_inp.brake_pressed, gas_pressed=act_inp.gas_pressed,
    )
    candidates = build_candidates(scene)
    decision = decide(candidates, inp.mode, inp.accel_limits, inp.sources)
    acc_envelope_debug: dict[str, Any] = {}
    acc_envelope_result: Any | None = None
    try:
      model_progress_candidate = str(decision.selected_intent) in ("no_lead_launch", "lead_pullaway", "lead_standstill_launch")
      lead_compression_candidate = str(decision.selected_intent) == "lead_gap_compression"
      previous_for_envelope = self._prev_smoothed_a_target if self._prev_smoothed_a_target is not None else inp.seed_a_target
      acc_envelope_result = evaluate_acc_envelope(AccEnvelopeInputs(
        v_ego=inp.v_ego,
        candidate_a_target=decision.a_target,
        previous_a_target=previous_for_envelope,
        dt=dt,
        openpilot_longitudinal_control=True,
        has_lead=has_lead,
        lead_d_rel=lead_d_rel,
        lead_v_rel=lead_v_rel,
        lead_v_lead=lead_v,
        lead_a_lead_k=lead_a_k,
        lead_kinematics_valid=lead_kinematics_valid,
        model_stale=act_inp.model_stale,
        model_progress_candidate=model_progress_candidate,
        lead_compression_candidate=lead_compression_candidate,
        radar_stale=False,
        lead_required=has_lead,
      ))
      acc_envelope_debug = acc_envelope_result.debug_dict()
    except Exception:
      acc_envelope_debug = {
        "acc_envelope_active": False,
        "acc_envelope_would_cap": True,
        "acc_envelope_cap_reason": "fault",
      }

    # Decision output is a pre-MPC target. Most lead-present braking seeds bind as hazards;
    # only explicitly approved low-risk soft cases raise the seed before the downstream MPC
    # solves final lead-follow physics.
    raw_a_target = float(decision.a_target)
    a_target = raw_a_target
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
      and raw_a_target >= 0.15
      and not act_inp.force_slow_decel
      and not act_inp.brake_pressed
      and not act_inp.gas_pressed
      and not act_inp.model_should_stop
    )
    standstill_release_confidence_fault = False
    standstill_release_confidence_debug: dict[str, Any] = {}
    try:
      standstill_release_confidence_debug = predict_standstill_release_confidence(
        mode=inp.standstill_release_confidence_mode,
        release_allowed=standstill_release_allowed,
        release_source=str(decision.selected_intent if standstill_release_allowed else ""),
        release_reason=str(decision.reason if standstill_release_allowed else ""),
        release_a_target=float(max(raw_a_target, 0.15)) if standstill_release_allowed else 0.0,
        lead_progress_allowed=lead_progress_allowed,
        lead_gap_excess=lead_gap_excess,
        lead_shadow_active=lead_shadow_active,
        alternate_threat_active=alternate_threat_active,
        force_slow_decel=act_inp.force_slow_decel,
        brake_pressed=act_inp.brake_pressed,
        gas_pressed=act_inp.gas_pressed,
        model_should_stop=act_inp.model_should_stop,
      ).debug_dict()
    except Exception:
      standstill_release_confidence_fault = True

    # Scenario context is intentionally shadow-only: it classifies the situation for telemetry
    # and future phases but must not touch actuation, candidates, a_target, or should_stop.
    scenario_context_fault = False
    scenario_context_debug: dict[str, Any] = {}
    try:
      # Use mode-sanitized actuation inputs for evidence that is mode-gated, so the shadow
      # classifier does not report ACC-excluded model/curve/speed evidence as potential effects.
      scenario_context_debug = predict_scenario_context(
        mode=inp.scenario_context_mode,
        v_ego=act_inp.v_ego,
        a_ego=act_inp.a_ego,
        accel_coast=act_inp.accel_coast,
        standstill=inp.standstill,
        steering_angle_deg=inp.steering_angle_deg,
        steering_torque=inp.steering_torque,
        leads=inp.leads,
        model_should_stop=act_inp.model_should_stop,
        model_stop_distance=act_inp.model_stop_distance,
        speed_limit_active=act_inp.speed_limit_active,
        curve_active=act_inp.curve_active,
        gas_pressed=act_inp.gas_pressed,
        brake_pressed=act_inp.brake_pressed,
      ).debug_dict()
    except Exception:
      scenario_context_fault = True

    admitted_hazard_targets = [float(c.a_target) for c in candidates
                               if c.role is CandidateRole.PHYSICAL_HAZARD
                               and c.source in decision.admitted_sources
                               and math.isfinite(float(c.a_target))]
    strongest_admitted_hazard_a = min(admitted_hazard_targets) if admitted_hazard_targets else None
    target_smoothing_debug = self._apply_target_smoothing(raw_a_target, dt, act_inp, decision, acc_envelope_result,
                                                          strongest_admitted_hazard_a)
    a_target = float(target_smoothing_debug["target_smoothing_a_target"])
    return LongitudinalStackResult(
      a_target=float(a_target),
      should_stop=bool(decision.should_stop),
      decision=decision,
      standstill_release_allowed=standstill_release_allowed,
      standstill_release_source=str(decision.selected_intent if standstill_release_allowed else ""),
      standstill_release_a_target=float(max(raw_a_target, 0.15)) if standstill_release_allowed else 0.0,
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
        "scenario_context_fault": scenario_context_fault,
        **scenario_context_debug,
        **acc_envelope_debug,
        **target_smoothing_debug,
      },
    )

  def _apply_target_smoothing(self, raw_a_target: float, dt: float, inp: LongitudinalStackInputs,
                              decision: Decision, acc_envelope_result: Any | None,
                              strongest_admitted_hazard_a: float | None = None) -> dict[str, Any]:
    debug = {
      "target_smoothing_active": False,
      "target_smoothing_applied": False,
      "target_smoothing_direction": "none",
      "target_smoothing_reason": "inactive",
      "target_smoothing_raw_a_target": float(raw_a_target) if math.isfinite(raw_a_target) else 0.0,
      "target_smoothing_prev_a_target": float("nan"),
      "target_smoothing_hazard_floor": float(strongest_admitted_hazard_a) if strongest_admitted_hazard_a is not None else float("nan"),
      "target_smoothing_a_target": float(raw_a_target) if math.isfinite(raw_a_target) else 0.0,
    }

    reset_reason = ""
    try:
      a_min, a_max = float(inp.accel_limits[0]), float(inp.accel_limits[1])
      prev = self._prev_smoothed_a_target
      if not inp.long_active:
        reset_reason = "long_inactive"
      elif inp.brake_pressed or inp.gas_pressed:
        reset_reason = "driver_override"
      elif inp.force_slow_decel:
        reset_reason = "force_slow"
      elif bool(decision.should_stop):
        reset_reason = "should_stop"
      elif inp.model_should_stop:
        reset_reason = "model_stop"
      elif acc_envelope_result is None or not bool(getattr(acc_envelope_result, "active", False)):
        reset_reason = "acc_envelope_inactive"
      elif not math.isfinite(float(raw_a_target)):
        reset_reason = "nonfinite_raw"
      elif not (math.isfinite(float(dt)) and 0.0 < float(dt) <= UPWARD_TARGET_SLEW_MAX_DT_S):
        reset_reason = "invalid_dt"
      elif not (math.isfinite(a_min) and math.isfinite(a_max) and a_min <= a_max):
        reset_reason = "invalid_accel_limits"
      elif prev is not None and not math.isfinite(float(prev)):
        reset_reason = "nonfinite_previous"

      if reset_reason:
        self._prev_smoothed_a_target = None
        debug["target_smoothing_reason"] = reset_reason
        return debug

      raw = min(max(float(raw_a_target), a_min), a_max)
      if prev is None:
        self._prev_smoothed_a_target = raw
        debug.update({
          "target_smoothing_active": True,
          "target_smoothing_reason": "primed",
          "target_smoothing_a_target": raw,
        })
        return debug

      prev_f = min(max(float(prev), a_min), a_max)
      debug["target_smoothing_prev_a_target"] = prev_f
      debug["target_smoothing_active"] = True

      if raw < prev_f:
        if not self._downward_smoothing_allowed(raw, prev_f, inp, decision, acc_envelope_result):
          self._prev_smoothed_a_target = raw
          debug.update({
            "target_smoothing_direction": "downward",
            "target_smoothing_reason": "downward_passthrough",
            "target_smoothing_a_target": raw,
          })
          return debug

        max_down_step = abs(DOWNWARD_TARGET_SLEW_JERK) * float(dt)
        smoothed = max(raw, prev_f - max_down_step)
        if strongest_admitted_hazard_a is not None and math.isfinite(float(strongest_admitted_hazard_a)):
          smoothed = min(smoothed, float(strongest_admitted_hazard_a))
        smoothed = min(max(smoothed, a_min), a_max)
        self._prev_smoothed_a_target = smoothed
        debug.update({
          "target_smoothing_applied": bool(smoothed > raw + 1e-9),
          "target_smoothing_direction": "downward",
          "target_smoothing_reason": "downward_slew_limited" if smoothed > raw + 1e-9 else "downward_passthrough",
          "target_smoothing_a_target": smoothed,
        })
        return debug

      if raw == prev_f:
        self._prev_smoothed_a_target = raw
        debug.update({
          "target_smoothing_reason": "equal_passthrough",
          "target_smoothing_a_target": raw,
        })
        return debug

      jerk_limited = getattr(acc_envelope_result, "jerk_limited_a_target", raw)
      if not math.isfinite(float(jerk_limited)):
        jerk_limited = raw
      smoothed = min(raw, max(prev_f, float(jerk_limited)))
      if prev_f <= -0.5 and raw >= 0.0:
        smoothed = max(smoothed, raw - UPWARD_TARGET_SLEW_MAX_LAG)
      smoothed = min(max(smoothed, a_min), a_max)
      self._prev_smoothed_a_target = smoothed
      debug.update({
        "target_smoothing_applied": bool(smoothed < raw - 1e-9),
        "target_smoothing_direction": "upward",
        "target_smoothing_reason": "upward_slew_limited" if smoothed < raw - 1e-9 else "upward_passthrough",
        "target_smoothing_a_target": smoothed,
      })
      return debug
    except Exception:
      self._prev_smoothed_a_target = None
      debug["target_smoothing_reason"] = "fault"
      return debug

  def _downward_smoothing_allowed(self, raw: float, prev: float, inp: LongitudinalStackInputs,
                                  decision: Decision, acc_envelope_result: Any | None) -> bool:
    if decision.reason == "physical_hazard" or bool(decision.should_stop) or inp.model_should_stop:
      return False
    if raw < DOWNWARD_TARGET_SMOOTH_MIN_RAW_A:
      return False
    if prev - raw > DOWNWARD_TARGET_SMOOTH_MAX_DELTA:
      return False
    reasons = set(getattr(acc_envelope_result, "cap_reasons", ()) or ())
    if reasons & DOWNWARD_TARGET_SMOOTH_RISK_REASONS:
      return False
    lead0 = inp.leads[0] if inp.leads else None
    if lead0 is not None and bool(getattr(lead0, "status", False)):
      lead_v_rel = _f(getattr(lead0, "vRel", 0.0))
      lead_a_k = _f(getattr(lead0, "aLeadK", 0.0))
      if lead_v_rel < DOWNWARD_TARGET_SMOOTH_CLOSING_V_REL or lead_a_k < DOWNWARD_TARGET_SMOOTH_LEAD_A_K:
        return False
    return True


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
