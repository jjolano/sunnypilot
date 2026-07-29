"""Post-MPC custom longitudinal final arbitration.

``CustomLongitudinalFinalizer`` owns the stop-hold/release state and the helpers that
arbitrate the final ``(a_target, should_stop, e2e_source)`` tuple after the upstream MPC
solve.  The implementation is split into small single-concern stages for clarity, but the
public API and behavior remain unchanged from the original Phase-5B extraction.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any

from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import (
  STOP_DISTANCE,
  get_safe_obstacle_distance,
  get_stopped_equivalence_factor,
)
from openpilot.sunnypilot.custom.longitudinal.lead_cushion import (
  LOW_SPEED_GAP_CLOSURE_MAX_ACCEL,
  lead_catchup_accel_cap,
  low_speed_gap_closure_accel,
)
from openpilot.sunnypilot.custom.longitudinal.lead_confidence import close_stop_go_radar_id_churn_continuous
from openpilot.sunnypilot.custom.longitudinal.lead_context import lead_present
from openpilot.sunnypilot.custom.longitudinal.modes import LongitudinalMode
from openpilot.sunnypilot.custom.longitudinal.net_demand_cap import NetDemandCapFinalStage, NetDemandCapTrace
from openpilot.sunnypilot.custom.longitudinal.policy_tables import LEAD_CRAWL_BREAKOUT_MIN_OPENING
from openpilot.sunnypilot.custom.longitudinal.departure_prediction import (
  DeparturePredictionEvidence,
  DeparturePredictionTrace,
  PHASE_ARMING,
  PHASE_INACTIVE,
  PHASE_PREDICTED,
  PERSISTENCE_S,
  TIMEOUT_S,
)
from openpilot.sunnypilot.custom.longitudinal.wiring import FAULT_CLASS_INTERNAL, CustomLongitudinalOutput


@dataclass
class FinalizerResult:
  a_target: float
  should_stop: bool
  e2e_source: bool
  custom_long_output_telemetry: CustomLongitudinalOutput | None = None
  last_release_block_reason: str = ""
  departure_prediction_evidence: DeparturePredictionEvidence = field(default_factory=DeparturePredictionEvidence)
  departure_prediction_trace: DeparturePredictionTrace = field(default_factory=DeparturePredictionTrace)


def _valid_lead_id(lead: Any) -> int | None:
  """Return only radar-confirmed track IDs; -1 is the vision-only sentinel."""
  try:
    lead_id = int(getattr(lead, 'radarTrackId', -1))
  except (TypeError, ValueError):
    return None
  return lead_id if lead_id >= 0 else None


# ---------------------------------------------------------------------------
# Input snapshot
# ---------------------------------------------------------------------------

@dataclass
class _InputSnapshot:
  sm: Any
  custom_long: Any
  custom_long_output: Any
  is_e2e: bool
  model_stale: bool
  dt: float
  mpc_a_target: float
  mpc_should_stop: bool
  raw_model_a_target: float
  raw_model_should_stop: bool
  car_state: Any
  controls_state: Any
  radar_state: Any
  selected_lead: Any
  has_lead: bool
  lead_d_rel: float
  lead_v: float
  lead_v_rel: float
  lead_y_rel: float
  lead_a_k: float
  gas_pressed: bool
  brake_pressed: bool
  v_ego: float
  standstill: bool
  force_decel: bool
  lead_id: Any
  stopping_distance: float
  v_ego_stopping: float
  stop_accel: float
  mpc_a_target_valid: bool = True
  raw_model_a_target_valid: bool = True
  long_active: bool = False

  @classmethod
  def build(cls, finalizer: CustomLongitudinalFinalizer, sm: Any,
            custom_long: Any, custom_long_output: Any, is_e2e: bool,
            model_stale: bool, dt: float, mpc_a_target: float, mpc_should_stop: bool,
            raw_model_a_target: float, raw_model_should_stop: bool,
            mpc_a_target_valid: bool | None = None,
            raw_model_a_target_valid: bool | None = None,
            long_active: bool | None = None) -> _InputSnapshot:
    car_state = finalizer._sm_item(sm, 'carState')
    controls_state = finalizer._sm_item(sm, 'controlsState')
    car_control = finalizer._sm_item(sm, 'carControl')
    radar_state = finalizer._sm_item(sm, 'radarState')
    selected_lead = finalizer._select_stop_hold_lead(radar_state, finalizer.lead_stop_hold_lead_id) if radar_state is not None else None
    has_lead = selected_lead is not None
    lead_d_rel = float(getattr(selected_lead, 'dRel', 0.0) or 0.0) if selected_lead is not None else 0.0
    lead_v = float(getattr(selected_lead, 'vLead', 0.0) or 0.0) if selected_lead is not None else 0.0
    lead_v_rel = float(getattr(selected_lead, 'vRel', 0.0) or 0.0) if selected_lead is not None else 0.0
    lead_y_rel = float(getattr(selected_lead, 'yRel', 0.0) or 0.0) if selected_lead is not None else 0.0
    lead_a_k = float(getattr(selected_lead, 'aLeadK', 0.0) or 0.0) if selected_lead is not None else 0.0
    lead_id = _valid_lead_id(selected_lead)
    gas_pressed = bool(getattr(car_state, 'gasPressed', False)) if car_state is not None else False
    brake_pressed = bool(getattr(car_state, 'brakePressed', False)) if car_state is not None else False
    v_ego = float(getattr(car_state, 'vEgo', 0.0) or 0.0) if car_state is not None else 0.0
    standstill = bool(getattr(car_state, 'standstill', False)) if car_state is not None else False
    force_decel = bool(getattr(controls_state, 'forceDecel', False)) if controls_state is not None else False

    # CarParams has no stoppingDistance field — the old 6.0 fallback silently governed every
    # distance rule while the MPC actually lands at STOP_DISTANCE (route 000002b0 co-stop
    # diagnosis). The getattr stays as a test seam for fakes that set it explicitly.
    stopping_distance = float(getattr(finalizer.CP, 'stoppingDistance', STOP_DISTANCE) or STOP_DISTANCE)
    v_ego_stopping = finalizer._STOP_HOLD_V_EGO_STOPPING
    stop_accel = getattr(finalizer.CP, 'stopAccel', None)
    stop_accel = -0.5 if stop_accel is None else float(stop_accel)

    if mpc_a_target_valid is None:
      try:
        mpc_a_target_valid = math.isfinite(float(mpc_a_target))
      except (TypeError, ValueError):
        mpc_a_target_valid = False
    if raw_model_a_target_valid is None:
      try:
        raw_model_a_target_valid = math.isfinite(float(raw_model_a_target))
      except (TypeError, ValueError):
        raw_model_a_target_valid = False
    if long_active is None:
      long_active = bool(getattr(car_control, 'longActive', False)) if car_control is not None else False

    return cls(
      sm=sm,
      custom_long=custom_long,
      custom_long_output=custom_long_output,
      is_e2e=is_e2e,
      model_stale=model_stale,
      dt=dt,
      mpc_a_target=mpc_a_target,
      mpc_should_stop=mpc_should_stop,
      raw_model_a_target=raw_model_a_target,
      raw_model_should_stop=raw_model_should_stop,
      car_state=car_state,
      controls_state=controls_state,
      radar_state=radar_state,
      selected_lead=selected_lead,
      has_lead=has_lead,
      lead_d_rel=lead_d_rel,
      lead_v=lead_v,
      lead_v_rel=lead_v_rel,
      lead_y_rel=lead_y_rel,
      lead_a_k=lead_a_k,
      gas_pressed=gas_pressed,
      brake_pressed=brake_pressed,
      v_ego=v_ego,
      standstill=standstill,
      force_decel=force_decel,
      lead_id=lead_id,
      stopping_distance=stopping_distance,
      v_ego_stopping=v_ego_stopping,
      stop_accel=stop_accel,
      mpc_a_target_valid=bool(mpc_a_target_valid),
      raw_model_a_target_valid=bool(raw_model_a_target_valid),
      long_active=bool(long_active),
    )


def _model_stop_blocks_release(snapshot: _InputSnapshot) -> bool:
  """Apply raw model-stop evidence only in modes that admit it."""
  return bool(
    snapshot.custom_long.mode is not LongitudinalMode.ACC and
    not snapshot.model_stale and
    snapshot.raw_model_should_stop
  )


def _same_latched_lead(finalizer: CustomLongitudinalFinalizer, snapshot: _InputSnapshot) -> bool:
  """Match the latched lead, tolerating tightly bounded stop-go radar ID churn."""
  lead_id = snapshot.lead_id
  latched_id = finalizer.lead_stop_hold_lead_id
  if lead_id is None or latched_id is None:
    return False
  if lead_id == latched_id:
    return True
  if lead_id not in finalizer.lead_stop_hold_churn_ids or latched_id not in finalizer.lead_stop_hold_churn_ids:
    return False
  if finalizer.lead_stop_hold_prev_v is None or finalizer.lead_stop_hold_prev_y_rel is None:
    return False
  prev_d_rel = finalizer.lead_stop_hold_gap_prev_d_rel
  if prev_d_rel is None:
    return False
  return close_stop_go_radar_id_churn_continuous(
    int(latched_id), int(lead_id),
    float(prev_d_rel), snapshot.lead_d_rel,
    float(finalizer.lead_stop_hold_prev_v), snapshot.lead_v,
    float(finalizer.lead_stop_hold_prev_y_rel), snapshot.lead_y_rel,
  )


# ---------------------------------------------------------------------------
# Stop-hold lead selector
# ---------------------------------------------------------------------------

class _StopHoldLeadSelector:
  """Picks the closest stopped/crawling lead, otherwise the closest lead."""

  @staticmethod
  def select(radar_state: Any, latched_id: Any = None) -> Any:
    return CustomLongitudinalFinalizer._select_stop_hold_lead(radar_state, latched_id)


# ---------------------------------------------------------------------------
# Stop-hold latch lifecycle
# ---------------------------------------------------------------------------

class _StopHoldLatchLifecycle:
  """Arms, maintains, transfers, and drops out the stopped-lead hold latch."""

  @staticmethod
  def settle_arm_applies(finalizer: CustomLongitudinalFinalizer, snapshot: _InputSnapshot) -> bool:
    if not snapshot.has_lead or snapshot.gas_pressed:
      return False
    if not all(math.isfinite(v) for v in (snapshot.v_ego, snapshot.v_ego_stopping, snapshot.lead_v, snapshot.lead_v_rel, snapshot.lead_d_rel)):
      return False
    if snapshot.v_ego > max(snapshot.v_ego_stopping + 0.2, finalizer._STOP_HOLD_SETTLE_ARM_V_EGO_FLOOR):
      return False
    if snapshot.lead_v > finalizer._STOP_HOLD_SETTLE_ARM_MAX_LEAD_V:
      return False
    if snapshot.lead_v_rel > finalizer._STOP_HOLD_SETTLE_ARM_MAX_LEAD_V_REL:
      return False
    braking_dist = min(
      snapshot.v_ego ** 2 / (2.0 * finalizer._STOP_HOLD_SETTLE_ARM_BRAKE_DIST_DECEL),
      finalizer._STOP_HOLD_SETTLE_ARM_BRAKE_DIST_MAX,
    )
    settle_distance = snapshot.stopping_distance + finalizer._STOP_HOLD_SETTLE_ARM_DISTANCE_MARGIN + braking_dist
    return snapshot.lead_d_rel <= settle_distance

  @staticmethod
  def update(finalizer: CustomLongitudinalFinalizer, snapshot: _InputSnapshot,
             reset_lead_stop_hold: Any) -> bool:
    dt = snapshot.dt
    v_ego = snapshot.v_ego
    has_lead = snapshot.has_lead
    lead_d_rel = snapshot.lead_d_rel
    lead_v = snapshot.lead_v
    gas_pressed = snapshot.gas_pressed
    lead_id = snapshot.lead_id

    # Route 261: the old max(stopping_distance+2, 10) armed the hold at 8-10 m, freezing
    # whatever gap ego happened to rest at (9.3 m observed vs the driver's 1.6 m manual
    # median). Arm only inside the MPC stop buffer envelope so the MPC finishes creeping
    # to its stop gap before the hold latches.
    arm_distance = finalizer._STOP_HOLD_ARM_GAP_M
    v_ego_stopping = snapshot.v_ego_stopping

    if math.isfinite(dt) and dt > 0.0:
      finalizer.lead_stop_hold_release_carry_s = max(0.0, finalizer.lead_stop_hold_release_carry_s - dt)
    # Route 000002b2 t=281: the stop bit ran 3 consecutive frames with mpcA still POSITIVE
    # (+0.01..+0.08) mid-creep — launch-transition chatter, not a stop demand — and the
    # persistence cancel interrupted an in-progress creep at 0.4 m/s with a -2.0 re-latch.
    # A stop demand is the bit AND a non-positive accel request (non-finite fails closed).
    mpc_stop_demand = bool(snapshot.mpc_should_stop and
                           (not math.isfinite(snapshot.mpc_a_target) or snapshot.mpc_a_target <= 0.0))
    finalizer.mpc_stop_persist_frames = finalizer.mpc_stop_persist_frames + 1 if mpc_stop_demand else 0
    mpc_go = bool(math.isfinite(snapshot.mpc_a_target) and snapshot.mpc_a_target >= finalizer._STOP_HOLD_MPC_GO_MIN_A
                  and not snapshot.mpc_should_stop)
    finalizer.mpc_go_persist_frames = finalizer.mpc_go_persist_frames + 1 if mpc_go else 0
    lead_departing = bool(
      snapshot.has_lead and
      math.isfinite(snapshot.lead_a_k) and math.isfinite(snapshot.lead_v_rel) and
      snapshot.lead_a_k >= finalizer._DEPARTING_LEAD_MIN_A_K and
      snapshot.lead_v_rel >= finalizer._DEPARTING_LEAD_MIN_OPENING
    )
    finalizer.lead_accel_persist_frames = finalizer.lead_accel_persist_frames + 1 if lead_departing else 0

    # Hold owns a stopped state, not equal-speed crawl. During a release grace, a stale
    # MPC stop bit cannot re-arm it; real braking or another admitted stop still can.
    custom_stop = bool(getattr(snapshot.custom_long_output, "should_stop", False))
    hard_stop_requested = bool(
      snapshot.mpc_a_target < finalizer._STOP_HOLD_SAME_ID_MIN_MPC_A_TARGET or
      _model_stop_blocks_release(snapshot) or
      custom_stop
    )
    stop_requested = bool(snapshot.mpc_should_stop or hard_stop_requested)
    stopped_pair = bool(snapshot.standstill and snapshot.lead_v_rel <= finalizer._STOP_HOLD_SETTLE_ARM_MAX_LEAD_V_REL)
    settling_to_stop = bool(stop_requested and (
      snapshot.v_ego <= v_ego_stopping or snapshot.lead_v < snapshot.v_ego
    ))
    release_grace_blocks_rearm = bool(
      finalizer.launch_dip_grace_s > 0.0 and
      not hard_stop_requested and
      snapshot.lead_v_rel >= 0.0 and
      not snapshot.brake_pressed and
      not snapshot.force_decel
    )
    # A sustained crawl release must not be re-latched by the very stop flags it outranks
    # (route 000002ac: lead re-stops mid-creep with the gap still open); only an MPC stop
    # demand or the driver may re-arm the hold while the sustain window is alive.
    sustain_blocks_rearm = bool(
      finalizer.stop_hold_release_sustain_s > 0.0 and
      finalizer.mpc_stop_persist_frames < finalizer._STOP_HOLD_MPC_STOP_PERSIST_FRAMES and
      snapshot.mpc_a_target >= finalizer._STOP_HOLD_SAME_ID_MIN_MPC_A_TARGET and
      not snapshot.brake_pressed and
      not snapshot.force_decel
    )
    arm_evidence = bool((stopped_pair or settling_to_stop) and not release_grace_blocks_rearm and not sustain_blocks_rearm)

    stop_hold_set = bool(
      not finalizer.lead_stop_hold_active and
      arm_evidence and
      has_lead and
      v_ego < v_ego_stopping + 0.2 and
      lead_d_rel <= arm_distance and
      lead_v <= 0.3 and
      not gas_pressed,
    )
    settle_hold_set = bool(
      not finalizer.lead_stop_hold_active and
      arm_evidence and
      _StopHoldLatchLifecycle.settle_arm_applies(finalizer, snapshot)
    )
    if stop_hold_set or settle_hold_set:
      finalizer._clear_launch_floor_fade_state(clear_approach=True)
      finalizer.lead_stop_hold_active = True
      finalizer.stop_hold_release_sustain_s = 0.0
      finalizer.lead_stop_hold_gap_increasing_s = 0.0
      finalizer.lead_stop_hold_missing_s = 0.0
      finalizer.lead_stop_hold_gap_prev_d_rel = float(lead_d_rel)
      finalizer.lead_stop_hold_prev_v = float(lead_v)
      finalizer.lead_stop_hold_prev_y_rel = float(snapshot.lead_y_rel)
      finalizer.lead_stop_hold_gap_baseline_d_rel = min(float(lead_d_rel), finalizer._STOP_HOLD_MAX_BASELINE_D_REL)
      finalizer.lead_stop_hold_arm_d_rel = float(lead_d_rel)
      finalizer.lead_stop_hold_lead_id = lead_id
      finalizer.lead_stop_hold_churn_ids = {lead_id} if lead_id is not None else set()

    if finalizer.lead_stop_hold_active:
      same_latched_lead = _same_latched_lead(finalizer, snapshot)
      if gas_pressed:
        reset_lead_stop_hold()
      elif lead_id is not None and finalizer.lead_stop_hold_lead_id is not None and not same_latched_lead:
        if lead_v <= 0.3 and lead_d_rel <= arm_distance:
          finalizer.lead_stop_hold_churn_ids.update((finalizer.lead_stop_hold_lead_id, lead_id))
          finalizer.lead_stop_hold_lead_id = lead_id
          finalizer.lead_stop_hold_gap_prev_d_rel = float(lead_d_rel)
          finalizer.lead_stop_hold_prev_v = float(lead_v)
          finalizer.lead_stop_hold_prev_y_rel = float(snapshot.lead_y_rel)
          finalizer.lead_stop_hold_gap_baseline_d_rel = min(float(lead_d_rel), finalizer._STOP_HOLD_MAX_BASELINE_D_REL)
          finalizer.lead_stop_hold_arm_d_rel = float(lead_d_rel)
          finalizer.lead_stop_hold_missing_s = 0.0
          finalizer.lead_stop_hold_gap_increasing_s = 0.0
        else:
          finalizer.lead_stop_hold_missing_s += dt
          if not (finalizer.lead_stop_hold_missing_s < 0.5 and v_ego < v_ego_stopping + 0.2 and not gas_pressed):
            reset_lead_stop_hold()
      elif not has_lead:
        finalizer.lead_stop_hold_missing_s += dt
        if not (finalizer.lead_stop_hold_missing_s < 0.5 and v_ego < v_ego_stopping + 0.2 and not gas_pressed):
          reset_lead_stop_hold()
      else:
        finalizer.lead_stop_hold_missing_s = 0.0
        gap_increasing = finalizer.lead_stop_hold_gap_prev_d_rel is not None and float(lead_d_rel) > float(finalizer.lead_stop_hold_gap_prev_d_rel)
        if gap_increasing:
          finalizer.lead_stop_hold_gap_increasing_s += dt
        else:
          finalizer.lead_stop_hold_gap_increasing_s = 0.0
        finalizer.lead_stop_hold_gap_prev_d_rel = float(lead_d_rel)
        finalizer.lead_stop_hold_prev_v = float(lead_v)
        finalizer.lead_stop_hold_prev_y_rel = float(snapshot.lead_y_rel)
        if finalizer.lead_stop_hold_gap_baseline_d_rel is None:
          finalizer.lead_stop_hold_gap_baseline_d_rel = float(lead_d_rel)
    else:
      finalizer.lead_stop_hold_gap_increasing_s = 0.0
      finalizer.lead_stop_hold_gap_prev_d_rel = float(lead_d_rel) if has_lead else None
      finalizer.lead_stop_hold_prev_v = float(lead_v) if has_lead else None
      finalizer.lead_stop_hold_prev_y_rel = float(snapshot.lead_y_rel) if has_lead else None
      finalizer.lead_stop_hold_gap_baseline_d_rel = min(float(lead_d_rel), finalizer._STOP_HOLD_MAX_BASELINE_D_REL) if has_lead else None
      finalizer.lead_stop_hold_arm_d_rel = float(lead_d_rel) if has_lead else None
      finalizer.lead_stop_hold_missing_s = 0.0

    return finalizer.lead_stop_hold_active


# ---------------------------------------------------------------------------
# Release gate
# ---------------------------------------------------------------------------

class _ReleaseGate:
  """Decides whether a stop-hold latch may release, and which accel to emit."""

  @staticmethod
  def routine_breakout(lead_v_rel: float) -> bool:
    return CustomLongitudinalFinalizer._routine_lead_launch_breakout(lead_v_rel)

  @staticmethod
  def standstill_release_gate_enabled(finalizer: CustomLongitudinalFinalizer, custom_long: Any) -> bool:
    return bool(
      custom_long.enabled and
      custom_long.mode is LongitudinalMode.SCC and
      str(getattr(custom_long, "standstill_release_confidence_mode", "off") or "off") == "gate"
    )

  @staticmethod
  def standstill_release_request_valid(finalizer: CustomLongitudinalFinalizer, snapshot: _InputSnapshot,
                                       min_mpc_a_target: float = -0.03, carried: bool = False) -> bool:
    custom_long = snapshot.custom_long
    custom_long_output = snapshot.custom_long_output
    mpc_a_target = snapshot.mpc_a_target
    raw_model_a_target = snapshot.raw_model_a_target

    if not custom_long.enabled or custom_long_output is None:
      finalizer.last_release_block_reason = "no_release_permission"
      return False
    # Carried frames re-use a verdict granted moments ago on a model-stop-clear flicker
    # (route 000002ac t=252/t=1243); the lapsed permission and the re-asserted stop flags
    # are exactly what lapsed, so they are skipped — every live gate below still runs.
    if not carried:
      if not bool(getattr(custom_long_output, "standstill_release_allowed", False)):
        finalizer.last_release_block_reason = "no_release_permission"
        return False
      if str(getattr(custom_long_output, "standstill_release_source", "")) not in ("lead_pullaway", "lead_standstill_launch", "no_lead_launch"):
        finalizer.last_release_block_reason = "invalid_release_source"
        return False
      if bool(getattr(custom_long_output, "should_stop", False)):
        finalizer.last_release_block_reason = "custom_should_stop"
        return False
      if _model_stop_blocks_release(snapshot):
        finalizer.last_release_block_reason = "raw_model_stop"
        return False
    for value in (mpc_a_target, raw_model_a_target):
      if not math.isfinite(float(value)):
        finalizer.last_release_block_reason = "non_finite_target"
        return False
    if custom_long.mode is LongitudinalMode.E2E and float(raw_model_a_target) < 0.15:
      finalizer.last_release_block_reason = "e2e_model_accel_too_low"
      return False
    cs = snapshot.sm["carState"]
    controls_state = snapshot.sm["controlsState"]
    if bool(getattr(cs, "brakePressed", False)):
      finalizer.last_release_block_reason = "driver_brake"
      return False
    if bool(getattr(cs, "gasPressed", False)):
      finalizer.last_release_block_reason = "driver_gas"
      return False
    if bool(getattr(controls_state, "forceDecel", False)):
      finalizer.last_release_block_reason = "force_decel"
      return False
    if float(mpc_a_target) < min_mpc_a_target:
      finalizer.last_release_block_reason = "mpc_brake_veto"
      return False
    finalizer.last_release_block_reason = ""
    return True

  @staticmethod
  def static_gap_overshoot(finalizer: CustomLongitudinalFinalizer, snapshot: _InputSnapshot) -> bool:
    """Both cars fully at rest, latched gap well past the MPC stop buffer, MPC persistently
    demanding closure — the co-stop settle-freeze signature (route 000002b0 t=948)."""
    if not (math.isfinite(float(snapshot.lead_v)) and math.isfinite(float(snapshot.lead_v_rel)) and
            math.isfinite(float(snapshot.lead_d_rel))):
      return False
    if float(snapshot.lead_v) > 0.3 or abs(float(snapshot.lead_v_rel)) > 0.3:
      return False
    if finalizer.mpc_go_persist_frames < finalizer._STOP_HOLD_MPC_GO_PERSIST_FRAMES:
      return False
    return float(snapshot.lead_d_rel) - snapshot.stopping_distance >= finalizer._STOP_HOLD_STATIC_OVERSHOOT_MIN_M

  @staticmethod
  def crawl_fallback_applies(finalizer: CustomLongitudinalFinalizer, snapshot: _InputSnapshot,
                             same_id: bool) -> bool:
    custom_long = snapshot.custom_long
    custom_long_output = snapshot.custom_long_output
    mpc_a_target = snapshot.mpc_a_target
    raw_model_a_target = snapshot.raw_model_a_target
    selected_lead = snapshot.selected_lead
    lead_d_rel = snapshot.lead_d_rel
    lead_v = snapshot.lead_v
    lead_v_rel = snapshot.lead_v_rel

    # Route 261: the fallback originally accepted only still-stationary creeping leads; a
    # moving lead had to carry the stack's explicit release verdict (two authorities
    # disagreed otherwise). Route 00000288: leads crawling at 0.3-0.6 m/s sit between the
    # two authorities and stay held. A moving lead may now ride this fallback only on
    # cumulative displacement evidence (checked below with the baseline opening); the crawl
    # accel ramp keeps it the gentle authority either way.
    if not math.isfinite(float(snapshot.lead_v)):
      return False
    if _ReleaseGate.standstill_release_gate_enabled(finalizer, custom_long):
      if not bool(getattr(custom_long_output, "research_actuation_allowed", False)):
        return False
    if not custom_long.enabled or custom_long_output is None:
      return False
    if not bool(getattr(custom_long_output, "enabled", False)):
      return False
    # Route 000002ac: a stopped/crawling lead <10 m keeps the policy should_stop and the raw
    # model stop asserted for the entire crawl, so as binary vetoes they made this fallback
    # unreachable in its own target scenario. They now only cap the released accel
    # (_STOP_HOLD_CRAWL_MODEL_STOP_A_MAX in release_accepts). E2E keeps the model veto: there
    # the model is the sole longitudinal authority.
    if custom_long.mode is LongitudinalMode.E2E and not snapshot.model_stale and snapshot.raw_model_should_stop:
      return False
    cs = finalizer._sm_item(snapshot.sm, 'carState')
    controls_state = finalizer._sm_item(snapshot.sm, 'controlsState')
    if cs is None or controls_state is None:
      return False
    if bool(getattr(cs, "brakePressed", False)):
      return False
    if bool(getattr(cs, "gasPressed", False)):
      return False
    if bool(getattr(controls_state, "forceDecel", False)):
      return False
    if selected_lead is None or not same_id:
      return False
    lead_id = snapshot.lead_id
    if lead_id is None or finalizer.lead_stop_hold_lead_id is None or not same_id:
      return False
    if finalizer.lead_stop_hold_gap_baseline_d_rel is None:
      return False
    for value in (lead_d_rel, lead_v, lead_v_rel, mpc_a_target, raw_model_a_target):
      if not math.isfinite(float(value)):
        return False
    if custom_long.mode is not LongitudinalMode.ACC and float(raw_model_a_target) < 0.0:
      return False
    if float(mpc_a_target) < finalizer._STOP_HOLD_SAME_ID_MIN_MPC_A_TARGET:
      return False
    # Second evidence class: the lead never has to move to close a settle-frozen overshoot
    # gap — both at rest + persistent MPC closure demand is the release evidence.
    if _ReleaseGate.static_gap_overshoot(finalizer, snapshot):
      return True
    if float(lead_v_rel) < 0.05:
      return False
    arm_d_rel = finalizer.lead_stop_hold_arm_d_rel
    displacement = float(lead_d_rel) - float(arm_d_rel) if arm_d_rel is not None else 0.0
    # One distance rule (route 28b t=344, 2026-07-14): the lead giving >=0.5 m of cumulative
    # ground from the latched gap carries the creep at ANY lead speed — the gap ramp starts at
    # ~0.13 m/s^2 there and is self-limiting (ego can only close what the lead gives), so the
    # release gate needs only a noise floor, not a velocity ladder. Replaces the old 0.8 m
    # moving threshold and the 0.6 m stationary-creep branch.
    if displacement < finalizer._STOP_HOLD_CREEP_DISPLACEMENT_M:
      return False
    return True

  @staticmethod
  def release_accepts(finalizer: CustomLongitudinalFinalizer, snapshot: _InputSnapshot) -> tuple[bool, float]:
    custom_long_output = snapshot.custom_long_output
    mpc_a_target = snapshot.mpc_a_target
    raw_model_a_target = snapshot.raw_model_a_target
    selected_lead = snapshot.selected_lead
    lead_d_rel = snapshot.lead_d_rel
    lead_v = snapshot.lead_v
    lead_v_rel = snapshot.lead_v_rel

    if selected_lead is None:
      finalizer.last_release_block_reason = "no_lead"
      return False, float(lead_d_rel)
    release_source = str(getattr(custom_long_output, "standstill_release_source", ""))
    lead_id = snapshot.lead_id
    same_id = _same_latched_lead(finalizer, snapshot)
    source_valid = release_source in ("lead_pullaway", "lead_standstill_launch")
    if source_valid and same_id:
      finalizer.lead_stop_hold_release_carry_s = finalizer._STOP_HOLD_RELEASE_CARRY_S
      finalizer.lead_stop_hold_release_carry_a = float(getattr(custom_long_output, "standstill_release_a_target", 0.0))
    carried = bool(not source_valid and same_id and finalizer.lead_stop_hold_release_carry_s > 0.0)
    if lead_id is not None and finalizer.lead_stop_hold_lead_id is not None and not same_id:
      finalizer.last_release_block_reason = "different_lead_id"
      return False, float(lead_d_rel)

    crawl_fallback = bool(
      not source_valid and not carried and
      _ReleaseGate.crawl_fallback_applies(finalizer, snapshot, same_id)
    )
    if not source_valid and not carried and not crawl_fallback:
      finalizer.last_release_block_reason = "invalid_release_source"
      return False, float(lead_d_rel)

    for value in (lead_d_rel, lead_v, lead_v_rel, mpc_a_target, raw_model_a_target):
      if not math.isfinite(float(value)):
        finalizer.last_release_block_reason = "non_finite_values"
        return False, float(lead_d_rel)

    stopping_distance = snapshot.stopping_distance
    if not crawl_fallback:
      if float(lead_v) < 0.30 or float(lead_v_rel) < 0.15:
        finalizer.last_release_block_reason = "lead_not_moving"
        return False, float(lead_d_rel)

    min_d_rel = stopping_distance + finalizer._STOP_HOLD_SAME_ID_MIN_D_REL_MARGIN if same_id else stopping_distance + 0.1
    if same_id and finalizer.lead_stop_hold_gap_baseline_d_rel is not None:
      baseline_opening = (
        finalizer._STOP_HOLD_SAME_ID_VALID_BASELINE_OPENING_M if (source_valid or carried)
        else finalizer._STOP_HOLD_SAME_ID_MIN_D_REL_BASELINE_OPENING
      )
      baseline_min_d_rel = float(finalizer.lead_stop_hold_gap_baseline_d_rel) + baseline_opening
      # Route 000002ac t=763/t=1243: stops routinely latch at 3.3-3.8 m; the absolute 4.5 m
      # floor then demanded ~1 m of donated gap before even an authorized release could fire.
      # The floor never asks for more than the latch geometry plus the required opening.
      floor = min(finalizer._STOP_HOLD_SAME_ID_MIN_D_REL_FLOOR, baseline_min_d_rel)
      min_d_rel = max(floor, min(min_d_rel, baseline_min_d_rel))
    if float(lead_d_rel) <= min_d_rel:
      finalizer.last_release_block_reason = "distance_gate"
      return False, float(lead_d_rel)

    if same_id:
      if source_valid or carried:
        min_gap_increasing_s = finalizer._STOP_HOLD_SAME_ID_VALID_GAP_INCREASING_S
      elif _ReleaseGate.routine_breakout(float(lead_v_rel)):
        min_gap_increasing_s = finalizer._STOP_HOLD_SAME_ID_ROUTINE_PULLAWAY_S
      else:
        min_gap_increasing_s = finalizer._STOP_HOLD_SAME_ID_MIN_PULLAWAY_S
    else:
      min_gap_increasing_s = 0.15
    # Sub-resolution crawl motion (~2 cm/frame) resets the strictly-increasing streak on
    # flat/jitter frames; >=_STOP_HOLD_CREEP_DISPLACEMENT_M of cumulative opening from the
    # latched arm gap outranks any streak (route 00000288). A static-overshoot release has
    # no opening at all by definition — its evidence (rest + persistent MPC demand) carries.
    baseline_opening_carries = bool(
      same_id and finalizer.lead_stop_hold_arm_d_rel is not None and
      float(lead_d_rel) - float(finalizer.lead_stop_hold_arm_d_rel) >= finalizer._STOP_HOLD_CREEP_DISPLACEMENT_M
    )
    if crawl_fallback and not baseline_opening_carries:
      baseline_opening_carries = _ReleaseGate.static_gap_overshoot(finalizer, snapshot)
    if not baseline_opening_carries and finalizer.lead_stop_hold_gap_increasing_s < min_gap_increasing_s:
      finalizer.last_release_block_reason = "gap_increasing_time"
      return False, float(lead_d_rel)

    if same_id and finalizer.lead_stop_hold_gap_baseline_d_rel is not None:
      min_baseline_opening = 0.5 if crawl_fallback else finalizer._STOP_HOLD_SAME_ID_VALID_BASELINE_OPENING_M
      if float(lead_d_rel) - float(finalizer.lead_stop_hold_gap_baseline_d_rel) < min_baseline_opening:
        finalizer.last_release_block_reason = "baseline_opening"
        return False, float(lead_d_rel)

    if lead_id is None or finalizer.lead_stop_hold_lead_id is None:
      if finalizer.lead_stop_hold_gap_increasing_s < finalizer._STOP_HOLD_NEW_ID_GAP_INCREASING_S:
        finalizer.last_release_block_reason = "new_id_gap_increasing_time"
        return False, float(lead_d_rel)

    if crawl_fallback:
      finalizer.last_release_block_reason = ""
      requested_a = max(float(mpc_a_target), finalizer._STOP_HOLD_CRAWL_RELEASE_A_MIN)
      release_a = _ReleaseAccel.accel_for_gap(
        finalizer, requested_a, lead_d_rel, lead_v, lead_v_rel, same_id, valid_source=False
      )
      if release_a <= 0.0:
        finalizer.last_release_block_reason = "crawl_deadband"
        return False, float(lead_d_rel)
      if _model_stop_blocks_release(snapshot) or bool(getattr(custom_long_output, "should_stop", False)):
        release_a = min(release_a, finalizer._STOP_HOLD_CRAWL_MODEL_STOP_A_MAX)
      return True, release_a

    min_mpc = finalizer._STOP_HOLD_SAME_ID_MIN_MPC_A_TARGET if same_id else -0.03
    if not _ReleaseGate.standstill_release_request_valid(finalizer, snapshot, min_mpc, carried=carried):
      return False, float(lead_d_rel)

    finalizer.last_release_block_reason = ""
    requested_release_a = float(getattr(custom_long_output, "standstill_release_a_target", 0.0)) if custom_long_output is not None else 0.0
    if carried:
      requested_release_a = max(requested_release_a, float(finalizer.lead_stop_hold_release_carry_a))
    release_a = _ReleaseAccel.accel_for_gap(
      finalizer, requested_release_a, lead_d_rel, lead_v, lead_v_rel, same_id, valid_source=True
    )
    if release_a <= 0.0:
      finalizer.last_release_block_reason = "crawl_deadband"
      return False, float(lead_d_rel)
    if carried and (_model_stop_blocks_release(snapshot) or bool(getattr(custom_long_output, "should_stop", False))):
      release_a = min(release_a, finalizer._STOP_HOLD_CRAWL_MODEL_STOP_A_MAX)
    return True, release_a


# ---------------------------------------------------------------------------
# Release accel
# ---------------------------------------------------------------------------

class _ReleaseAccel:
  """Crawl-aware positive accel shaping for stop-hold release."""

  @staticmethod
  def accel_for_gap(finalizer: CustomLongitudinalFinalizer, requested_a: float,
                    lead_d_rel: float, lead_v: float, lead_v_rel: float,
                    same_id: bool, valid_source: bool = False) -> float:
    release_a = min(max(float(requested_a), finalizer._STOP_HOLD_RELEASE_A_MIN), finalizer._STOP_HOLD_RELEASE_A_MAX)
    if _ReleaseGate.routine_breakout(float(lead_v_rel)):
      return float(release_a)
    if not same_id or finalizer.lead_stop_hold_gap_baseline_d_rel is None:
      return float(release_a)

    gap_error = float(lead_d_rel) - float(finalizer.lead_stop_hold_gap_baseline_d_rel)
    crawl_release_a = float(min(release_a, finalizer._STOP_HOLD_CRAWL_RELEASE_A_MAX))
    if gap_error <= finalizer._STOP_HOLD_CRAWL_DEADBAND_M:
      if valid_source and float(lead_v) >= 0.30 and float(lead_v_rel) >= 0.15:
        return crawl_release_a
      return 0.0
    gap_limited_a = (gap_error - finalizer._STOP_HOLD_CRAWL_DEADBAND_M) / finalizer._STOP_HOLD_CRAWL_GAP_TAU
    if gap_limited_a < finalizer._STOP_HOLD_CRAWL_RELEASE_A_MIN:
      if valid_source and float(lead_v) >= 0.30 and float(lead_v_rel) >= 0.15:
        return crawl_release_a
      return 0.0
    return float(min(release_a, finalizer._STOP_HOLD_CRAWL_RELEASE_A_MAX, gap_limited_a))


# ---------------------------------------------------------------------------
# Hold command
# ---------------------------------------------------------------------------

class _HoldCommand:
  """Computes the hold accel and optional pre-release prep slew."""

  @staticmethod
  def prep_applies(finalizer: CustomLongitudinalFinalizer, snapshot: _InputSnapshot) -> bool:
    custom_long = snapshot.custom_long
    custom_long_output = snapshot.custom_long_output
    lead_d_rel = snapshot.lead_d_rel
    lead_v = snapshot.lead_v
    lead_v_rel = snapshot.lead_v_rel
    mpc_a_target = snapshot.mpc_a_target
    raw_model_a_target = snapshot.raw_model_a_target

    if not custom_long.enabled or custom_long_output is None:
      return False
    # Route 261: prep used to require the full release permission (pullaway source), so it
    # could never lead the release and the TSS2 PCM ate ~1 s of brake-hold unwind AFTER the
    # release fired. Prep is not a release — the car stays held at the prep target — so it
    # keys on its own early-lead-motion thresholds below and runs the PCM unwind in
    # parallel with the release decision. Any stop demand (model stop, custom should_stop,
    # driver input, lead re-stopping) drops the hold straight back to the harsh target.
    if not bool(getattr(custom_long_output, "enabled", False)):
      return False
    if bool(getattr(custom_long_output, "should_stop", False)):
      return False
    if _model_stop_blocks_release(snapshot):
      return False

    if snapshot.brake_pressed or snapshot.gas_pressed:
      return False
    if snapshot.force_decel:
      return False

    v_ego = snapshot.v_ego
    v_ego_stopping = snapshot.v_ego_stopping
    if v_ego >= v_ego_stopping + 0.2:
      return False

    selected_lead = snapshot.selected_lead
    if selected_lead is None:
      return False
    if not _same_latched_lead(finalizer, snapshot):
      return False

    for value in (lead_d_rel, lead_v, lead_v_rel, mpc_a_target, raw_model_a_target):
      if not math.isfinite(float(value)):
        return False

    if float(mpc_a_target) < finalizer._STOP_HOLD_RELEASE_PREP_MIN_MPC_A_TARGET:
      return False
    if float(lead_v) < finalizer._STOP_HOLD_RELEASE_PREP_MIN_LEAD_V or float(lead_v_rel) < finalizer._STOP_HOLD_RELEASE_PREP_MIN_LEAD_V_REL:
      return False
    if finalizer.lead_stop_hold_gap_increasing_s < finalizer._STOP_HOLD_RELEASE_PREP_MIN_GAP_INCREASING_S:
      return False
    if float(lead_d_rel) <= finalizer._STOP_HOLD_SAME_ID_MIN_D_REL_FLOOR:
      return False

    stopping_distance = snapshot.stopping_distance
    absolute_distance_ready = float(lead_d_rel) > stopping_distance + finalizer._STOP_HOLD_RELEASE_PREP_MIN_D_REL_MARGIN
    arm_d_rel = finalizer.lead_stop_hold_arm_d_rel
    sustained_opening_ready = (
      arm_d_rel is not None and math.isfinite(float(arm_d_rel)) and
      float(lead_d_rel) - float(arm_d_rel) >= finalizer._STOP_HOLD_RELEASE_PREP_MIN_OPENING_M
    )
    if not (absolute_distance_ready or sustained_opening_ready):
      return False

    return True

  @staticmethod
  def apply_prep(finalizer: CustomLongitudinalFinalizer, snapshot: _InputSnapshot,
                 raw_hold: float) -> float:
    dt = snapshot.dt
    state = finalizer.stop_hold_release_prep_a_target
    raw_prev = finalizer.stop_hold_release_prep_raw_prev

    clear = (
      not math.isfinite(raw_hold) or
      not math.isfinite(dt) or dt <= 0.0 or
      (state is not None and not math.isfinite(state)) or
      (raw_prev is not None and not math.isfinite(raw_prev)) or
      not _HoldCommand.prep_applies(finalizer, snapshot)
    )
    if clear:
      finalizer.stop_hold_release_prep_a_target = None
      finalizer.stop_hold_release_prep_raw_prev = None
      return float(raw_hold)

    prev_output = float(state) if state is not None else float(raw_hold)
    prev_raw = float(raw_prev) if raw_prev is not None else float(raw_hold)

    if raw_hold < prev_raw:
      finalizer.stop_hold_release_prep_a_target = float(raw_hold)
      finalizer.stop_hold_release_prep_raw_prev = float(raw_hold)
      return float(raw_hold)

    desired = max(float(raw_hold), finalizer._STOP_HOLD_RELEASE_PREP_A_TARGET)
    max_step = finalizer._STOP_HOLD_RELEASE_PREP_MAX_UP_JERK * dt
    if desired > prev_output + max_step:
      limited = prev_output + max_step
    else:
      limited = desired
    limited = max(limited, prev_output)
    finalizer.stop_hold_release_prep_a_target = float(limited)
    finalizer.stop_hold_release_prep_raw_prev = float(raw_hold)
    return float(limited)

  @staticmethod
  def compute(finalizer: CustomLongitudinalFinalizer, snapshot: _InputSnapshot) -> tuple[float, bool]:
    finalizer.stop_hold_release_slew_a_target = None
    stop_accel = snapshot.stop_accel
    hold_a_target = float(snapshot.mpc_a_target) if math.isfinite(float(snapshot.mpc_a_target)) else stop_accel
    if snapshot.is_e2e and not snapshot.model_stale:
      raw_a_target = float(snapshot.raw_model_a_target) if math.isfinite(float(snapshot.raw_model_a_target)) else stop_accel
      raw_hold = min(raw_a_target, hold_a_target, stop_accel)
    else:
      raw_hold = min(hold_a_target, stop_accel)
    e2e_source = bool(snapshot.is_e2e and not snapshot.model_stale and raw_hold < hold_a_target)

    selected_lead = snapshot.selected_lead
    same_id_sp = _same_latched_lead(finalizer, snapshot)
    if (
      (snapshot.standstill or snapshot.v_ego <= finalizer._STOP_HOLD_STANDSTILL_NORMALIZE_MAX_V_EGO) and
      selected_lead is not None and
      same_id_sp and
      math.isfinite(raw_hold) and
      not snapshot.brake_pressed and
      not snapshot.gas_pressed and
      not snapshot.force_decel and
      not _model_stop_blocks_release(snapshot)
    ):
      raw_hold = max(float(raw_hold), finalizer._STOP_HOLD_STANDSTILL_NORMALIZED_A_TARGET)

    a_target = _HoldCommand.apply_prep(finalizer, snapshot, raw_hold)
    return float(a_target), e2e_source


# ---------------------------------------------------------------------------
# Final arbitration
# ---------------------------------------------------------------------------

class _FinalArbitration:
  """Custom-disabled, SCC/ACC/E2E path selection, caps, and release slew."""

  @staticmethod
  def custom_longitudinal_should_stop(custom_long: Any, custom_long_output: Any,
                                      mpc_should_stop: bool, raw_model_should_stop: bool,
                                      model_stale: bool = False) -> bool | None:
    if not custom_long.enabled or custom_long_output is None:
      return None
    if custom_long.mode is LongitudinalMode.ACC:
      return bool(mpc_should_stop)
    if custom_long.mode is LongitudinalMode.E2E:
      return bool(mpc_should_stop or (raw_model_should_stop and not model_stale))
    return bool(mpc_should_stop or custom_long_output.should_stop)

  @staticmethod
  def scc_custom_stop_cap(base_a_target: float, custom_long: Any, custom_long_output: Any,
                          release_mpc_stop: bool = False) -> float:
    if release_mpc_stop:
      return float(base_a_target)
    if custom_long.mode is not LongitudinalMode.SCC or not custom_long.enabled or custom_long_output is None:
      return float(base_a_target)
    if not bool(getattr(custom_long_output, "enabled", False)):
      return float(base_a_target)
    stop_posture = (
      str(getattr(custom_long_output, "selected_intent", "") or "") == "stop_approach"
      or bool(getattr(custom_long_output, "model_stop_corroborated", False))
    )
    if not stop_posture:
      return float(base_a_target)
    raw_custom_a = getattr(custom_long_output, "a_target", None)
    if raw_custom_a is None:
      return float(base_a_target)
    try:
      custom_a = float(raw_custom_a)
    except (TypeError, ValueError):
      return float(base_a_target)
    if not math.isfinite(custom_a):
      return float(base_a_target)
    return float(min(float(base_a_target), custom_a))

  @staticmethod
  def lead_catchup_cap(finalizer: CustomLongitudinalFinalizer, base_a_target: float,
                       snapshot: _InputSnapshot, should_stop: bool) -> float:
    custom_long_output = snapshot.custom_long_output
    if should_stop or float(base_a_target) <= 0.0 or custom_long_output is None:
      return float(base_a_target)
    if not bool(getattr(custom_long_output, "enabled", False)):
      return float(base_a_target)
    if snapshot.brake_pressed or snapshot.gas_pressed or snapshot.force_decel:
      return float(base_a_target)
    if snapshot.v_ego > finalizer._LEAD_CATCHUP_MAX_V_EGO:
      return float(base_a_target)

    lead = getattr(snapshot.radar_state, "leadOne", None)
    if not lead_present(lead):
      return float(base_a_target)
    d_rel = finalizer._finite_float_or_none(getattr(lead, "dRel", None))
    v_lead = finalizer._finite_float_or_none(getattr(lead, "vLead", None))
    a_lead = finalizer._finite_float_or_none(getattr(lead, "aLeadK", 0.0))
    t_follow = finalizer._finite_float_or_none(getattr(custom_long_output, "t_follow", None))
    if None in (d_rel, v_lead, a_lead, t_follow) or d_rel <= 0.0 or t_follow <= 0.0:
      return float(base_a_target)

    follow_gap = max(0.0, float(
      get_safe_obstacle_distance(snapshot.v_ego, t_follow) - get_stopped_equivalence_factor(v_lead)
    ))
    return float(lead_catchup_accel_cap(
      snapshot.v_ego, v_lead, a_lead, d_rel, follow_gap, base_a_target,
    ))

  @staticmethod
  def _resolve_gap_closure_lead(finalizer: CustomLongitudinalFinalizer,
                                snapshot: _InputSnapshot, request: Any) -> Any | None:
    """Resolve the requested radar slot, allowing only known close-track continuity."""
    slots = (
      getattr(snapshot.radar_state, "leadOne", None),
      getattr(snapshot.radar_state, "leadTwo", None),
    )
    try:
      request_idx = int(getattr(request, "lead_idx", -1))
      request_track_id = int(getattr(request, "lead_track_id", -1))
    except (TypeError, ValueError):
      return None
    if request_idx not in (0, 1) or request_track_id < 0:
      return None

    request_d_rel = finalizer._finite_float_or_none(getattr(request, "lead_d_rel", None))
    request_v_lead = finalizer._finite_float_or_none(getattr(request, "lead_v_lead", None))
    request_v_rel = finalizer._finite_float_or_none(getattr(request, "lead_v_rel", None))
    request_y_rel = finalizer._finite_float_or_none(getattr(request, "lead_y_rel", None))

    def values(lead: Any) -> tuple[float, float, float, float] | None:
      if not lead_present(lead):
        return None
      current = tuple(
        finalizer._finite_float_or_none(getattr(lead, name, None))
        for name in ("dRel", "vLead", "vRel", "yRel")
      )
      if any(value is None for value in current):
        return None
      return (
        float(current[0]) if current[0] is not None else 0.0,
        float(current[1]) if current[1] is not None else 0.0,
        float(current[2]) if current[2] is not None else 0.0,
        float(current[3]) if current[3] is not None else 0.0,
      )

    def matches_requested(lead: Any) -> bool:
      current_values = values(lead)
      current_track_id = _valid_lead_id(lead)
      if current_values is None or current_track_id is None:
        return False
      if current_track_id == request_track_id:
        return True
      if any(value is None for value in (request_d_rel, request_v_lead, request_v_rel, request_y_rel)):
        return False
      return close_stop_go_radar_id_churn_continuous(
        request_track_id, current_track_id,
        float(request_d_rel or 0.0), current_values[0],
        float(request_v_lead or 0.0), current_values[1],
        float(request_y_rel or 0.0), current_values[3],
      )

    active = tuple(
      (idx, lead) for idx, lead in enumerate(slots)
      if lead_present(lead)
    )
    matching = tuple((idx, lead) for idx, lead in active if matches_requested(lead))
    requested = next((lead for idx, lead in matching if idx == request_idx), None)
    if requested is None and matching:
      requested = matching[0][1]
    if requested is None:
      return None

    # A second slot is harmless only when it is the same requested physical lead. Any
    # distinct close/closing lead is an alternate threat and vetoes the correction.
    for _, alternate in active:
      if alternate is requested or matches_requested(alternate):
        continue
      alternate_values = values(alternate)
      if alternate_values is None or alternate_values[0] <= 15.0 or alternate_values[2] < -0.05:
        return None
    return requested

  @staticmethod
  def scc_low_speed_gap_closure_floor(finalizer: CustomLongitudinalFinalizer, base_a_target: float,
                                      snapshot: _InputSnapshot, should_stop: bool,
                                      release_mpc_stop: bool) -> float:
    """Apply the one-tick, locally rechecked SCC crawl gap-closure floor."""
    finalizer.low_speed_gap_closure_applied = False
    output = snapshot.custom_long_output
    request = getattr(output, "low_speed_gap_closure", None) if output is not None else None
    if request is None:
      return float(base_a_target)
    if (snapshot.custom_long is None or snapshot.custom_long.mode is not LongitudinalMode.SCC or
        snapshot.is_e2e or not bool(getattr(snapshot.custom_long, "enabled", False)) or
        output is None or not bool(getattr(output, "enabled", False)) or
        bool(getattr(snapshot.custom_long, "fault_class", "")) or
        bool(getattr(output, "fault_class", ""))):
      return float(base_a_target)
    if (finalizer.lead_stop_hold_active or finalizer.stop_hold_release_slew_a_target is not None or
        should_stop or release_mpc_stop or snapshot.mpc_should_stop or snapshot.raw_model_should_stop or
        snapshot.model_stale or snapshot.standstill or not snapshot.long_active or
        snapshot.brake_pressed or snapshot.gas_pressed or snapshot.force_decel or
        bool(getattr(output, "should_stop", False)) or
        bool(getattr(output, "model_stop_corroborated", False)) or
        str(getattr(output, "selected_intent", "") or "") in ("stop_approach", "lead_stop_hold") or
        bool(getattr(output, "standstill_release_allowed", False)) or
        bool(str(getattr(output, "standstill_release_source", "") or ""))):
      return float(base_a_target)
    if not math.isfinite(float(base_a_target)) or float(base_a_target) < -0.10:
      return float(base_a_target)
    if (not snapshot.mpc_a_target_valid or not snapshot.raw_model_a_target_valid or
        not math.isfinite(snapshot.mpc_a_target) or not math.isfinite(snapshot.raw_model_a_target) or
        snapshot.mpc_a_target <= -0.10 or snapshot.raw_model_a_target <= -0.10):
      return float(base_a_target)
    if snapshot.v_ego <= 0.0:
      return float(base_a_target)
    lead = _FinalArbitration._resolve_gap_closure_lead(finalizer, snapshot, request)
    if lead is None:
      return float(base_a_target)
    if str(getattr(output, "reason", "") or "") == "advisory_capped":
      return float(base_a_target)
    actuation = getattr(output, "actuation", None)
    for verdict, cap_name in (
      (getattr(actuation, "cut_in_brake_assist", None), "proposed_cap"),
      (getattr(actuation, "curve_traffic_advisor", None), "a_curve_cap_proposed"),
    ):
      cap = finalizer._finite_float_or_none(getattr(verdict, cap_name, 0.0))
      if (verdict is not None and bool(getattr(verdict, "eligible", False)) and
          bool(getattr(verdict, "apply_supported", False)) and cap is not None and cap < 0.0):
        return float(base_a_target)

    request_accel = finalizer._finite_float_or_none(getattr(request, "requested_accel", None))
    confidence = finalizer._finite_float_or_none(getattr(request, "lead_confidence", None))
    try:
      request_track_id = int(getattr(request, "lead_track_id", -1))
    except (TypeError, ValueError):
      request_track_id = -1
    if (request_accel is None or not 0.0 < request_accel <= LOW_SPEED_GAP_CLOSURE_MAX_ACCEL + 1e-9 or
        not bool(getattr(request, "lead_stable", False)) or not bool(getattr(request, "lead_radar", False)) or
        confidence is None or confidence < 0.55 or request_track_id < 0 or
        not lead_present(lead) or not bool(getattr(lead, "radar", False))):
      return float(base_a_target)

    t_follow = finalizer._finite_float_or_none(getattr(output, "t_follow", None))
    if t_follow is None or t_follow <= 0.0:
      return float(base_a_target)
    d_rel = finalizer._finite_float_or_none(getattr(lead, "dRel", None))
    v_lead = finalizer._finite_float_or_none(getattr(lead, "vLead", None))
    v_rel = finalizer._finite_float_or_none(getattr(lead, "vRel", None))
    a_lead = finalizer._finite_float_or_none(getattr(lead, "aLeadK", None))
    if any(value is None for value in (d_rel, v_lead, v_rel, a_lead)):
      return float(base_a_target)
    d_rel_f = float(d_rel) if d_rel is not None else 0.0
    v_lead_f = float(v_lead) if v_lead is not None else 0.0
    v_rel_f = float(v_rel) if v_rel is not None else 0.0
    a_lead_f = float(a_lead) if a_lead is not None else 0.0
    follow_gap = max(0.0, float(
      get_safe_obstacle_distance(snapshot.v_ego, t_follow) - get_stopped_equivalence_factor(v_lead_f)
    ))
    local_request = low_speed_gap_closure_accel(
      snapshot.v_ego, v_lead_f, a_lead_f, d_rel_f, follow_gap, v_rel_f,
    )
    if local_request <= 0.0 or not math.isfinite(local_request):
      return float(base_a_target)

    floor = min(request_accel, local_request)
    raised = max(float(base_a_target), floor)
    if raised <= float(base_a_target):
      return float(base_a_target)
    if not math.isfinite(snapshot.dt) or snapshot.dt <= 0.0:
      return float(base_a_target)
    previous = finalizer.final_a_prev
    if previous is not None and math.isfinite(float(previous)):
      raised = max(float(base_a_target), min(raised, float(previous) + 0.8 * snapshot.dt))
    finalizer.low_speed_gap_closure_applied = raised > float(base_a_target)
    return float(raised)

  @staticmethod
  def scc_cut_in_brake_assist_final_cap(finalizer: CustomLongitudinalFinalizer, base_a_target: float,
                                        sm: Any, custom_long: Any, custom_long_output: Any,
                                        release_mpc_stop: bool = False) -> float:
    if release_mpc_stop:
      return float(base_a_target)
    if custom_long.mode is not LongitudinalMode.SCC or not custom_long.enabled or custom_long_output is None:
      return float(base_a_target)
    if not bool(getattr(custom_long_output, "enabled", False)):
      return float(base_a_target)
    if not bool(getattr(custom_long_output, "research_actuation_allowed", False)):
      return float(base_a_target)
    if str(getattr(custom_long, "cut_in_brake_assist_mode", "off") or "off") != "apply":
      return float(base_a_target)
    actuation = getattr(custom_long_output, "actuation", None)
    verdict = getattr(actuation, "cut_in_brake_assist", None)
    if verdict is None or not bool(getattr(verdict, "eligible", False)) or not bool(getattr(verdict, "apply_supported", False)):
      return float(base_a_target)
    if not bool(getattr(actuation, "model_path_available", False)):
      return float(base_a_target)
    confidence = finalizer._finite_float_or_none(getattr(verdict, "confidence", 0.0))
    if confidence is None or confidence < finalizer._CUT_IN_BRAKE_ASSIST_APPLY_MIN_CONFIDENCE:
      return float(base_a_target)
    path_y_rel = finalizer._finite_float_or_none(getattr(verdict, "path_y_rel", None))
    if path_y_rel is None or abs(path_y_rel) > finalizer._CUT_IN_BRAKE_ASSIST_PATH_NEAR_Y_M:
      return float(base_a_target)
    car_state = finalizer._sm_item(sm, 'carState')
    controls_state = finalizer._sm_item(sm, 'controlsState')
    if bool(getattr(car_state, 'brakePressed', False)) or bool(getattr(car_state, 'gasPressed', False)):
      return float(base_a_target)
    if bool(getattr(controls_state, 'forceDecel', False)):
      return float(base_a_target)
    proposed_cap = finalizer._finite_float_or_none(getattr(verdict, "proposed_cap", 0.0))
    if proposed_cap is None or proposed_cap >= 0.0:
      return float(base_a_target)
    gentle_cap = max(proposed_cap, finalizer._CUT_IN_BRAKE_ASSIST_APPLY_MAX_DECEL)
    if gentle_cap >= float(base_a_target):
      return float(base_a_target)
    return float(min(float(base_a_target), gentle_cap))

  @staticmethod
  def scc_curve_traffic_advisor_final_cap(finalizer: CustomLongitudinalFinalizer, base_a_target: float,
                                          sm: Any, custom_long: Any, custom_long_output: Any,
                                          release_mpc_stop: bool = False) -> float:
    if release_mpc_stop:
      return float(base_a_target)
    if custom_long.mode is not LongitudinalMode.SCC or not custom_long.enabled or custom_long_output is None:
      return float(base_a_target)
    if not bool(getattr(custom_long_output, "enabled", False)):
      return float(base_a_target)
    if not bool(getattr(custom_long_output, "research_actuation_allowed", False)):
      return float(base_a_target)
    if str(getattr(custom_long, "curve_traffic_advisor_mode", "off") or "off") != "apply_conservative":
      return float(base_a_target)
    actuation = getattr(custom_long_output, "actuation", None)
    verdict = getattr(actuation, "curve_traffic_advisor", None)
    if verdict is None or not bool(getattr(verdict, "eligible", False)) or not bool(getattr(verdict, "apply_supported", False)):
      return float(base_a_target)
    if str(getattr(verdict, "traffic_block_reason", "") or "") != "":
      return float(base_a_target)
    confidence = finalizer._finite_float_or_none(getattr(verdict, "confidence", 0.0))
    if confidence is None or confidence < finalizer._CURVE_TRAFFIC_APPLY_MIN_CONFIDENCE:
      return float(base_a_target)
    if bool(getattr(actuation, "model_stale", False)):
      return float(base_a_target)
    car_state = finalizer._sm_item(sm, 'carState')
    controls_state = finalizer._sm_item(sm, 'controlsState')
    if bool(getattr(car_state, 'brakePressed', False)) or bool(getattr(car_state, 'gasPressed', False)):
      return float(base_a_target)
    if bool(getattr(controls_state, 'forceDecel', False)):
      return float(base_a_target)
    proposed_cap = finalizer._finite_float_or_none(getattr(verdict, "a_curve_cap_proposed", 0.0))
    if proposed_cap is None or proposed_cap >= 0.0:
      return float(base_a_target)
    conservative_cap = max(proposed_cap, finalizer._CURVE_TRAFFIC_APPLY_MIN_CAP)
    if conservative_cap >= float(base_a_target):
      return float(base_a_target)
    return float(min(float(base_a_target), conservative_cap))

  @staticmethod
  def apply_release_slew(finalizer: CustomLongitudinalFinalizer, sm: Any, dt: float, a_target: float,
                         release_mpc_stop: bool, mpc_stop: bool, raw_model_should_stop: bool,
                         should_stop: bool) -> float:
    car_state = finalizer._sm_item(sm, 'carState')
    controls_state = finalizer._sm_item(sm, 'controlsState')
    brake_pressed = bool(getattr(car_state, 'brakePressed', False)) if car_state is not None else False
    gas_pressed = bool(getattr(car_state, 'gasPressed', False)) if car_state is not None else False
    force_decel = bool(getattr(controls_state, 'forceDecel', False)) if controls_state is not None else False

    clear = (
      not math.isfinite(a_target) or
      not math.isfinite(dt) or dt <= 0.0 or
      (finalizer.stop_hold_release_slew_a_target is not None and not math.isfinite(finalizer.stop_hold_release_slew_a_target)) or
      bool(should_stop) or
      bool(mpc_stop) or
      # Crawl/sustained releases run with the raw model stop still asserted (route 000002ac);
      # on a known release frame the flag must not defeat the up-jerk bound, or the
      # hold -> release step (-2.0 -> +0.25) reaches the actuator unslewed.
      bool(raw_model_should_stop and not release_mpc_stop) or
      brake_pressed or gas_pressed or force_decel or
      a_target <= 0.0
    )
    if clear:
      finalizer.stop_hold_release_slew_a_target = None
      return float(a_target)

    if release_mpc_stop and finalizer.stop_hold_release_slew_a_target is None:
      # Seed at the prior commanded accel + one slew step, not at the release accel:
      # seeding at the release accel let the hold->release step (-0.5 -> +0.5) bypass
      # the up-jerk slew entirely (fuzz comfort failures, lead pullaway seeds 3/7).
      # Upward steps only — a prior command above the release target passes through.
      prev = finalizer.final_a_prev
      if prev is not None and prev < a_target:
        a_target = min(a_target, prev + finalizer._STOP_HOLD_RELEASE_MAX_UP_JERK * dt)
      finalizer.stop_hold_release_slew_a_target = float(a_target)
      return float(a_target)

    if finalizer.stop_hold_release_slew_a_target is not None:
      max_step = finalizer._STOP_HOLD_RELEASE_MAX_UP_JERK * dt
      last_slew = finalizer.stop_hold_release_slew_a_target
      if a_target > last_slew + max_step:
        a_target = last_slew + max_step
        finalizer.stop_hold_release_slew_a_target = float(a_target)
      elif a_target > last_slew:
        finalizer.stop_hold_release_slew_a_target = None
      else:
        finalizer.stop_hold_release_slew_a_target = float(a_target)

    return float(a_target)

  @staticmethod
  def standstill_release_clears_mpc_stop(finalizer: CustomLongitudinalFinalizer, snapshot: _InputSnapshot) -> tuple[bool, float]:
    mpc_a_target = snapshot.mpc_a_target
    mpc_should_stop = snapshot.mpc_should_stop
    custom_long_output = snapshot.custom_long_output
    if not _ReleaseGate.standstill_release_request_valid(finalizer, snapshot, min_mpc_a_target=-0.03):
      return False, float(mpc_a_target)
    if not mpc_should_stop:
      return False, float(mpc_a_target)
    release_a = min(
      max(float(mpc_a_target), finalizer._STOP_HOLD_RELEASE_A_MIN, float(getattr(custom_long_output, "standstill_release_a_target", 0.0))),
      finalizer._STOP_HOLD_RELEASE_A_MAX,
    )
    return True, release_a

  @staticmethod
  def stop_hold_release_sustain(finalizer: CustomLongitudinalFinalizer, snapshot: _InputSnapshot) -> tuple[bool, float]:
    """Bridge a stop-hold release across frames whose per-frame stack verdict has lapsed.

    Route 000002ac: after a crawl/carried release the stack keeps publishing should_stop
    while the scene still shows a close slow lead, which re-pins longcontrol to stopping one
    frame later. Sustain the release while the MPC stays comfortable; the MPC's own decel
    demand is the exit and hands the stop back at the proper gap. SCC only — in E2E the
    model is the sole authority and in ACC the stack verdict never pins.
    """
    mpc_a_target = float(snapshot.mpc_a_target)
    if finalizer.stop_hold_release_sustain_s <= 0.0:
      return False, mpc_a_target
    dt = snapshot.dt if (math.isfinite(snapshot.dt) and snapshot.dt > 0.0) else 0.05
    finalizer.stop_hold_release_sustain_s = max(0.0, finalizer.stop_hold_release_sustain_s - dt)
    if snapshot.custom_long.mode is not LongitudinalMode.SCC:
      finalizer.stop_hold_release_sustain_s = 0.0
      return False, mpc_a_target
    if snapshot.brake_pressed or snapshot.gas_pressed or snapshot.force_decel:
      finalizer.stop_hold_release_sustain_s = 0.0
      return False, mpc_a_target
    mpc_stop_persistent = finalizer.mpc_stop_persist_frames >= finalizer._STOP_HOLD_MPC_STOP_PERSIST_FRAMES
    if not math.isfinite(mpc_a_target) or mpc_stop_persistent or mpc_a_target < finalizer._STOP_HOLD_SAME_ID_MIN_MPC_A_TARGET:
      finalizer.stop_hold_release_sustain_s = 0.0
      return False, mpc_a_target
    if not math.isfinite(snapshot.v_ego) or snapshot.v_ego > finalizer._STOP_HOLD_SUSTAIN_MAX_V_EGO:
      finalizer.stop_hold_release_sustain_s = 0.0
      return False, mpc_a_target
    # Nothing to bridge unless the stack posture would re-pin or re-clamp this frame; a
    # healthy launch (posture cleared) passes mpcA through untouched and the window just
    # expires. Route 000002b0 t=915/t=928: post-release the policy sits in stop_approach
    # (model-stop-driven, should_stop False) and scc_custom_stop_cap clamped the creep to
    # its -0.38 approach decel — the car braked at standstill and re-latched. That advisory
    # posture needs the bridge exactly like the pinned verdict does.
    output = snapshot.custom_long_output
    stack_stop_posture = bool(
      bool(getattr(output, "should_stop", False)) or
      str(getattr(output, "selected_intent", "") or "") == "stop_approach"
    )
    if not stack_stop_posture:
      return False, mpc_a_target
    # Rolling window: lead presence refreshes it so radar flicker cannot abort a creep;
    # true lead loss lets it expire within the window (bounded, gentle roll-out).
    if snapshot.has_lead:
      finalizer.stop_hold_release_sustain_s = finalizer._STOP_HOLD_RELEASE_SUSTAIN_S
    release_a = min(max(mpc_a_target, finalizer._STOP_HOLD_RELEASE_A_MIN), finalizer._STOP_HOLD_CRAWL_RELEASE_A_MAX)
    if _model_stop_blocks_release(snapshot) or bool(getattr(snapshot.custom_long_output, "should_stop", False)):
      release_a = min(release_a, finalizer._STOP_HOLD_CRAWL_MODEL_STOP_A_MAX)
    return True, release_a

  @staticmethod
  def scc_launch_floor(finalizer: CustomLongitudinalFinalizer, snapshot: _InputSnapshot,
                       a_target: float, should_stop: bool) -> float:
    """Floor the weak first seconds of a confirmed launch (see _STOP_HOLD_LAUNCH_FLOOR_A).

    Applies only inside the post-release launch grace, SCC mode, with a present departing
    lead (routine-breakout opening), no stop posture on any layer, and a non-objecting
    MPC. Runs before the cap chain so every cap still wins.
    """
    if should_stop or finalizer.launch_dip_grace_s <= 0.0:
      return float(a_target)
    if snapshot.custom_long.mode is not LongitudinalMode.SCC:
      return float(a_target)
    if snapshot.brake_pressed or snapshot.gas_pressed or snapshot.force_decel:
      return float(a_target)
    if not snapshot.has_lead:
      return float(a_target)
    if _model_stop_blocks_release(snapshot) or bool(getattr(snapshot.custom_long_output, "should_stop", False)):
      return float(a_target)
    for value in (a_target, snapshot.lead_v, snapshot.lead_v_rel, snapshot.mpc_a_target, snapshot.v_ego):
      if not math.isfinite(float(value)):
        return float(a_target)
    if snapshot.v_ego > finalizer._LAUNCH_DIP_MAX_V_EGO:
      return float(a_target)
    # Departure in progress = lead genuinely moving AND still pulling ahead. Route 000002b5
    # t=1264: gating on lead_v_rel alone faded the floor mid-launch as ego accelerated to
    # chase (vRel 0.7 -> 0.48 while the lead kept opening), sagging the command to 0.2 for
    # 1.5 s — the drive's only gas press. At standstill lead_v == v_rel, so entry is
    # unchanged; a crawling lead (< breakout speed) still never gets floored.
    if float(snapshot.lead_v) < LEAD_CRAWL_BREAKOUT_MIN_OPENING:
      return float(a_target)
    if float(snapshot.lead_v_rel) < finalizer._STOP_HOLD_LAUNCH_FLOOR_MIN_OPENING:
      return float(a_target)
    if float(snapshot.mpc_a_target) < 0.0:
      return float(a_target)
    # ponytail: flat floor — the breakout gate above already guarantees a >=0.7 m/s speed
    # deficit, so a deficit-scaled floor never scales below the constant before the gate
    # itself fades it out as ego catches the lead.
    return float(max(float(a_target), finalizer._STOP_HOLD_LAUNCH_FLOOR_A))

  @staticmethod
  def scc_departing_lead_coast(finalizer: CustomLongitudinalFinalizer, snapshot: _InputSnapshot,
                               a_target: float, should_stop: bool) -> float:
    """Green-light save: never brake into a gap the departing lead is re-opening.

    Lift-off-only authority (see _DEPARTING_LEAD_* constants): clamps shallow braking to
    coast while the lead is measurably accelerating away, sustained for a few frames.
    Deep demand is real physics and passes through; a model/policy stop posture, driver
    input, closing lead, or lead loss drops it instantly.
    """
    if should_stop or not math.isfinite(float(a_target)):
      return float(a_target)
    if float(a_target) >= 0.0 or float(a_target) < finalizer._DEPARTING_LEAD_MAX_CLAMP_DEPTH:
      return float(a_target)
    if snapshot.custom_long.mode is not LongitudinalMode.SCC:
      return float(a_target)
    if snapshot.brake_pressed or snapshot.gas_pressed or snapshot.force_decel:
      return float(a_target)
    if not snapshot.has_lead:
      return float(a_target)
    if _model_stop_blocks_release(snapshot) or bool(getattr(snapshot.custom_long_output, "should_stop", False)):
      return float(a_target)
    for value in (snapshot.lead_d_rel, snapshot.lead_v_rel, snapshot.v_ego):
      if not math.isfinite(float(value)):
        return float(a_target)
    if snapshot.v_ego > finalizer._DEPARTING_LEAD_MAX_V_EGO:
      return float(a_target)
    if float(snapshot.lead_d_rel) < snapshot.stopping_distance:
      return float(a_target)
    if finalizer.lead_accel_persist_frames < finalizer._DEPARTING_LEAD_PERSIST_FRAMES:
      return float(a_target)
    return 0.0


# ---------------------------------------------------------------------------
# Telemetry adapter
# ---------------------------------------------------------------------------

class _TelemetryAdapter:
  """Builds the final result and telemetry snapshot."""

  @staticmethod
  def build_hold_telemetry(finalizer: CustomLongitudinalFinalizer,
                           custom_long_output: Any) -> CustomLongitudinalOutput | None:
    if custom_long_output is None:
      return None
    return replace(custom_long_output, should_stop=True, selected_intent="lead_stop_hold", reason="stopped_lead_latch")

  @staticmethod
  def result(a_target: float, should_stop: bool, e2e_source: bool,
             telemetry: CustomLongitudinalOutput | None,
             block_reason: str) -> FinalizerResult:
    return FinalizerResult(
      a_target=float(a_target),
      should_stop=bool(should_stop),
      e2e_source=bool(e2e_source),
      custom_long_output_telemetry=telemetry,
      last_release_block_reason=str(block_reason),
    )


# ---------------------------------------------------------------------------
# Public finalizer
# ---------------------------------------------------------------------------

class CustomLongitudinalFinalizer:
  _STOP_HOLD_V_EGO_STOPPING = 0.25
  # mirrors long_mpc STOP_DISTANCE (4.5) + 0.5 settle slack; the crawl-release governor
  # closes any latched gap back toward this baseline
  _STOP_HOLD_MAX_BASELINE_D_REL = 5.0
  # latch arm envelope: MPC stop buffer (4.5) + 1.5 settle margin
  _STOP_HOLD_ARM_GAP_M = 6.0
  _STOP_HOLD_SAME_ID_MIN_D_REL_MARGIN = 0.2
  _STOP_HOLD_SAME_ID_MIN_D_REL_FLOOR = 4.5
  _STOP_HOLD_SAME_ID_MIN_D_REL_BASELINE_OPENING = 0.5
  _STOP_HOLD_SAME_ID_VALID_BASELINE_OPENING_M = 0.20
  _STOP_HOLD_SAME_ID_GAP_INCREASING_S = 0.10
  _STOP_HOLD_SAME_ID_VALID_GAP_INCREASING_S = 0.10
  _STOP_HOLD_SAME_ID_MIN_MPC_A_TARGET = -0.10
  _STOP_HOLD_NEW_ID_GAP_INCREASING_S = 0.30
  _STOP_HOLD_SAME_ID_MIN_PULLAWAY_S = 0.30
  _STOP_HOLD_SAME_ID_ROUTINE_PULLAWAY_S = 0.10
  # Route 00000288 t=398/t=429: leads crawling away at 0.3-0.6 m/s fell in a dead zone between
  # the stationary-only crawl fallback and the flickering 0.3/0.15 m/s release gates — holds
  # pinned -0.5..-2.0 for 9-16 s while the gap opened 2+ m. Cumulative opening from the latched
  # gap is displacement evidence velocity noise cannot fake: at/above this opening the crawl
  # fallback releases at ANY lead speed and the strictly-increasing gap streak (which
  # sub-resolution crawl motion resets on flat/jitter frames) is no longer required. Measured
  # from the unclamped arm-time dRel (lead_stop_hold_arm_d_rel), NOT the 5.0 m-clamped runway
  # baseline. 0.5 m sits just above the crawl ramp's own dead zone (deadband 0.35 + A_MIN*TAU
  # ~= 0.41 m) and 2-5x radar dRel jitter; the ramp's gap proportionality bounds the commanded
  # creep, so the gate needs only this noise floor (route 28b t=344: driver launched 1.5 s into
  # a creep at 0.5 m opening that the old 0.8 m gate was still holding).
  _STOP_HOLD_CREEP_DISPLACEMENT_M = 0.5
  # Breakout is only consulted while the stop-hold latch is active (ego <= ~0.7 m/s), where
  # v_rel ~= lead_v, so v_rel is the whole gate (a lead_v arm was subsumed and deleted).
  # Crawl-launch feel is tuned via _STOP_HOLD_CRAWL_GAP_TAU below, not here.
  # Route 00000274: standstill launches behind a departing lead were still too weak to hold
  # without a gas override — mpcA wanted ~+0.54 while the crawl cap trickled the command out.
  # Shrink the deadband so accel starts building sooner, ramp faster per metre, and raise the cap.
  _STOP_HOLD_CRAWL_DEADBAND_M = 0.35
  # Route 0000025a: below the breakout the release accel ramps as (gap_opened - deadband) / TAU; the old
  # 2.0 s TAU held the launch at ~0.05 m/s^2 well after the lead was clearly moving. 1.2 ramps faster
  # per metre of gap while the deadband + A_MAX cap still bound it. ponytail: knob, retune from logs.
  _STOP_HOLD_CRAWL_GAP_TAU = 1.2
  _STOP_HOLD_CRAWL_RELEASE_A_MIN = 0.05
  # Raised with _STOP_HOLD_RELEASE_A_MAX (route 00000246, then 00000274): the close-crawl cap still
  # stays below the breakout cap, and the gap governor above continues to bound crawl accel.
  _STOP_HOLD_CRAWL_RELEASE_A_MAX = 0.50
  # Route 000002ac (2026-07-18): the crawl fallback was structurally dead — a stopped/crawling
  # lead <10 m ahead keeps raw model stop and the policy stop verdict asserted the whole time
  # (modelStop 0.87-0.95 across every held stop), so the two binary vetoes blocked the
  # displacement rule in exactly its target scenario (holds pinned -2.0 for 10-20 s while the
  # lead opened 1.5-4.2 m; 2 of 4 stops ended in driver gas). The vetoes are now a cap: while
  # either stop flag is still asserted the crawl/sustain command stays at this gentle ceiling;
  # once the model clears, the normal crawl cap applies. The one modelStop-clear stop that
  # night released in 0.66 s — the flags, not the evidence gates, were the whole difference.
  # Route 000002b0 (first on-road): 0.25 sat in the Toyota dead zone — carcontroller
  # permit_braking only flips off above a 0.3 net request, and route 246 measured the PCM
  # barely rolling below ~0.35 — the released creep never moved the car and the driver
  # gassed. 0.40 clears the hysteresis with margin while staying under the 0.50 crawl cap.
  _STOP_HOLD_CRAWL_MODEL_STOP_A_MAX = 0.40
  # Route 000002ac t=252/t=1243: policy release verdicts (lead_pullaway/lead_standstill_launch)
  # surfaced for 1-3 frames on model-stop-clear flickers and lapsed before the finalizer's own
  # gates (lead-motion 0.30 m/s, distance floor) could pass. Carry the verdict for this window;
  # driver/mpc/lead-motion gates still run live on carried frames.
  _STOP_HOLD_RELEASE_CARRY_S = 2.0
  # Post-release bridge: the stack keeps publishing should_stop while the scene still shows a
  # close slow lead, which would re-pin longcontrol to stopping one frame after a crawl or
  # carried release (longcontrol: stopping_condition = should_stop, unconditionally). Sustain
  # the release while the MPC stays comfortable (no mpc stop, mpcA >= same-id min); the window
  # refreshes while a lead is present so radar flicker cannot abort a creep, and true lead
  # loss ends it within the window. MPC decel demand is the natural exit and hands the stop
  # back at the proper gap.
  _STOP_HOLD_RELEASE_SUSTAIN_S = 1.5
  _STOP_HOLD_SUSTAIN_MAX_V_EGO = 2.0
  # Route 000002ac t=1251: the MPC stop bit chatters 1<->0 frame-to-frame through launch
  # transitions while mpcA sits at ~0.0 — a single flicker frame must not cancel the
  # sustain or unblock re-arm. Only a persistently asserted bit counts as an MPC stop
  # demand (mpcA < the same-id min remains the instant, non-flickery exit).
  _STOP_HOLD_MPC_STOP_PERSIST_FRAMES = 3
  # Static-overshoot release (route 000002b0 co-stop diagnosis): when the lead's rest
  # catches ego mid-roll, the settle latch freezes the gap at 6.6-7.0 m — 1.5-2 m past the
  # MPC's STOP_DISTANCE landing — and with the lead never moving again (red light) nothing
  # closed it: the MPC demanded +0.65-0.67 the entire park. With both cars fully at rest,
  # the latched gap well past the stop buffer, and the MPC persistently demanding closure,
  # release into the capped crawl; the MPC folds its demand at its own buffer, which
  # re-latches the hold at the proper gap. The 0.75 m threshold is the hysteresis that
  # keeps the closed park (overshoot ~0-0.3) from re-firing.
  _STOP_HOLD_STATIC_OVERSHOOT_MIN_M = 0.75
  _STOP_HOLD_MPC_GO_MIN_A = 0.30
  _STOP_HOLD_MPC_GO_PERSIST_FRAMES = 10
  # Launch floor (route 000002b2 t=753): after a good release the policy intent flips to
  # lead_follow as the lead speeds up, the release verdict lapses, and the command
  # collapses to raw mpcA — 1-2 s of 0.07-0.3 while the driver sustains ~1.0 from the
  # first second (route 282 measurement; both route-2b2 gas presses landed in this
  # window). During the post-release launch grace, with the departure confirmed by the
  # routine-breakout opening and no stop posture, floor the command at this value — the
  # breakout gate fades it out as ego catches the lead and the MPC ramp takes over. Caps
  # still run after the floor, so any curve/stop cap wins. 0.60 sits between the
  # route-282 lever (0.5-0.65) and the driver-demonstrated 1.0 first-second mean.
  _STOP_HOLD_LAUNCH_FLOOR_A = 0.60
  # Same still-pulling-ahead margin as the lead-motion release gates.
  _STOP_HOLD_LAUNCH_FLOOR_MIN_OPENING = 0.15
  # Green-light save (routes 2b5 t=1110, 2b0 t=338, 296 t=791): a lead that slows for a
  # red and re-accelerates on the green leaves the MPC braking -0.6..-0.8 for 3-4.5 s to
  # restore the inflated time gap — a gap the departing lead is already re-opening for
  # free (hypermile: the runway rebuilds from the lead's side). With the lead measurably
  # accelerating away, sustained for a few frames, clamp shallow braking to coast; deep
  # demand (below the clamp depth) is real physics and is never reshaped, and a genuine
  # model/policy stop posture always wins.
  _DEPARTING_LEAD_MIN_A_K = 0.30
  _DEPARTING_LEAD_MIN_OPENING = 0.15
  _DEPARTING_LEAD_PERSIST_FRAMES = 4
  _DEPARTING_LEAD_MAX_V_EGO = 8.0
  _DEPARTING_LEAD_MAX_CLAMP_DEPTH = -1.0
  _STOP_HOLD_SETTLE_ARM_V_EGO_FLOOR = 0.7
  _STOP_HOLD_SETTLE_ARM_MAX_LEAD_V = 0.5
  _STOP_HOLD_SETTLE_ARM_MAX_LEAD_V_REL = 0.1
  # Raised 0.5 -> 1.5 with the stopping_distance correction (6.0 phantom -> STOP_DISTANCE
  # 5.0) so the settle envelope stays ~7.5 m: co-stop parks rest at 6.6-7.0 m (route
  # 000002b0) and MUST still latch — an un-latched far park loses the crawl/overshoot
  # release machinery entirely. The frozen gap is closed by the static-overshoot release
  # below, not by refusing to latch.
  _STOP_HOLD_SETTLE_ARM_DISTANCE_MARGIN = 1.5
  _STOP_HOLD_SETTLE_ARM_BRAKE_DIST_DECEL = 2.0
  _STOP_HOLD_SETTLE_ARM_BRAKE_DIST_MAX = 1.0
  _STOP_HOLD_RELEASE_A_MIN = 0.15
  # Route 00000246 t=4454: at the old 0.35 cap the TSS2 PCM took ~1.3 s of commanded
  # +0.25-0.30 m/s^2 before the car moved off brake hold; 0.50 cleared most of it. Route
  # 00000261: the driver's launches sustain ~1.0 m/s^2 mean from the first second, so 0.65
  # shaves the residual PCM lag while staying inside driver-demonstrated comfort. The gap
  # governor and release slew still bound the profile.
  _STOP_HOLD_RELEASE_A_MAX = 0.65
  _STOP_HOLD_RELEASE_MAX_UP_JERK = 6.0
  _STOP_HOLD_RELEASE_PREP_A_TARGET = -0.20
  _STOP_HOLD_RELEASE_PREP_MAX_UP_JERK = 6.0
  _STOP_HOLD_RELEASE_PREP_MIN_LEAD_V = 0.25
  _STOP_HOLD_RELEASE_PREP_MIN_LEAD_V_REL = 0.10
  _STOP_HOLD_RELEASE_PREP_MIN_MPC_A_TARGET = -0.10
  _STOP_HOLD_RELEASE_PREP_MIN_GAP_INCREASING_S = 0.10
  _STOP_HOLD_RELEASE_PREP_MIN_D_REL_MARGIN = 0.10
  # Prep is still a hold, not a release. Sustained same-lead motion can start unwinding
  # the PCM before the absolute release distance once displacement clears radar jitter.
  _STOP_HOLD_RELEASE_PREP_MIN_OPENING_M = 0.05
  _STOP_HOLD_STANDSTILL_NORMALIZED_A_TARGET = -0.50
  _STOP_HOLD_STANDSTILL_NORMALIZE_MAX_V_EGO = 0.70
  # Approach damping: on route 0000025a the final aTarget limit-cycled +/-0.3 m/s^2 at ~2-3 Hz on
  # steady lead approaches and reached the actuator. Primary cause was the policy coast-fallback
  # regression (bb1d135b2d, fixed in policy._usable_coast_decel) toggling candidate caps through the
  # MPC; this damp stays as a generic backstop against any in-band arbitration chatter. Rate-limit
  # aTarget only inside the gentle authority band; strong accel/decel, stops and releases pass
  # through untouched so brake authority is never delayed.
  _APPROACH_DAMP_BAND = 0.55
  _APPROACH_DAMP_MAX_JERK = 3.0
  _APPROACH_DAMP_MIN_V_EGO = 1.5
  # Follow coast band: routes 290/291 steady-follow finalA crossed zero 12-15x/min with a
  # median dither depth of only -0.08 m/s^2 (63-81% of below-zero excursions never deeper
  # than flat coast), and each crossing toggles the Toyota PCM between gas and brake
  # regimes — the felt follow jerk. In HOLD regime, negative demands shallower than the
  # grade-aware natural coast decel (floored at _FOLLOW_BAND_MAX_DEPTH) clamp to 0 like a
  # human ignoring a tiny gap compression; a demand past the band floor switches to DECEL
  # regime and passes through unchanged (braking is never reshaped, only its tiny
  # precursors), returning to HOLD only once the demand is genuinely positive again.
  _FOLLOW_BAND_MAX_DEPTH = -0.35   # m/s^2; deepest demand the band may ever clamp
  _FOLLOW_BAND_MIN_DEPTH = -0.02   # m/s^2; shallower coast collapses the band (downhill)
  _FOLLOW_BAND_EXIT_A = 0.02       # m/s^2; DECEL -> HOLD only past this (hysteresis)
  _FOLLOW_BAND_MIN_V_EGO = 3.0     # m/s; stay out of the creep/stop regime
  # Giveaway budget. HOLD clamps a real (if shallow) closing demand to 0, so every held
  # frame hands back gap at the closing rate. Ignoring a *tiny* compression is the whole
  # premise; ignoring an unbounded number of them is not. Without this the band held for
  # 10 s of steady 20 m/s following in openpilot_lead_decel_3ms2 while d_rel ratcheted
  # 32.5 -> 26.2 m, and the lead's stop then had 2.3 m less runway than upstream — enough
  # to turn a 1.11 m clearance into contact. Budgeted in headway so city gaps are not held
  # open on a highway-sized allowance; the band's own v_ego floor keeps it non-degenerate.
  _FOLLOW_BAND_MAX_GIVEN_S = 0.10  # s of headway the band may give away per HOLD episode
  # Launch dip damping: route 282 rlog (t=42244-42246) — radar/vision lead churn during
  # pullaway punched 1-2 frame mpcA dips (1.3 -> 0.2 m/s^2) through final arbitration.
  # Masked then by driver gas; unmasked it reads as launch surging. Bounded by the grace
  # window, a confirmed-departing radar lead, and positive-to-positive steps only.
  _LAUNCH_DIP_GRACE_S = 3.0
  _LAUNCH_DIP_MAX_V_EGO = 5.0
  _LEAD_CATCHUP_MAX_V_EGO = 8.0

  _CUT_IN_BRAKE_ASSIST_APPLY_MIN_CONFIDENCE = 0.60
  _CUT_IN_BRAKE_ASSIST_APPLY_MAX_DECEL = -0.60
  _CUT_IN_BRAKE_ASSIST_PATH_NEAR_Y_M = 1.70

  _CURVE_TRAFFIC_APPLY_MIN_CONFIDENCE = 0.45
  _CURVE_TRAFFIC_APPLY_MIN_CAP = -0.85

  lead_stop_hold_active: bool
  lead_stop_hold_gap_increasing_s: float
  lead_stop_hold_missing_s: float
  lead_stop_hold_lead_id: Any
  lead_stop_hold_gap_prev_d_rel: float | None
  lead_stop_hold_prev_v: float | None
  lead_stop_hold_prev_y_rel: float | None
  lead_stop_hold_churn_ids: set[int]
  lead_stop_hold_gap_baseline_d_rel: float | None
  lead_stop_hold_arm_d_rel: float | None
  lead_stop_hold_release_carry_s: float
  lead_stop_hold_release_carry_a: float
  stop_hold_release_sustain_s: float
  mpc_stop_persist_frames: int
  mpc_go_persist_frames: int
  lead_accel_persist_frames: int
  custom_long_output_telemetry: CustomLongitudinalOutput | None
  last_release_block_reason: str
  stop_hold_release_slew_a_target: float | None
  stop_hold_release_prep_a_target: float | None
  stop_hold_release_prep_raw_prev: float | None
  approach_damp_a_prev: float | None
  low_speed_gap_closure_applied: bool
  launch_dip_grace_s: float
  launch_floor_fade_pending: bool
  final_a_prev: float | None
  departure_prediction_phase: str
  departure_prediction_phase_s: float
  departure_prediction_track_id: int
  departure_prediction_lockout_track_id: int
  departure_prediction_frame_start_ready: bool
  departure_prediction_applied: bool
  departure_prediction_applied_track_id: int
  departure_prediction_trace: DeparturePredictionTrace

  def __init__(self, CP: Any):
    self.CP = CP

    self.lead_stop_hold_active = False
    self.lead_stop_hold_gap_increasing_s = 0.0
    self.lead_stop_hold_missing_s = 0.0
    self.lead_stop_hold_lead_id = None
    self.lead_stop_hold_gap_prev_d_rel = None
    self.lead_stop_hold_prev_v = None
    self.lead_stop_hold_prev_y_rel = None
    self.lead_stop_hold_churn_ids = set()
    self.lead_stop_hold_gap_baseline_d_rel = None
    self.lead_stop_hold_arm_d_rel = None
    self.lead_stop_hold_release_carry_s = 0.0
    self.lead_stop_hold_release_carry_a = 0.0
    self.stop_hold_release_sustain_s = 0.0
    self.mpc_stop_persist_frames = 0
    self.mpc_go_persist_frames = 0
    self.lead_accel_persist_frames = 0
    self.custom_long_output_telemetry = None
    self.last_release_block_reason = ""
    self.stop_hold_release_slew_a_target = None
    self.stop_hold_release_prep_a_target = None
    self.stop_hold_release_prep_raw_prev = None
    self.approach_damp_a_prev = None
    self.low_speed_gap_closure_applied = False
    self.follow_band_regime = None
    self.follow_band_given_m = 0.0
    self.launch_dip_grace_s = 0.0
    self.launch_floor_fade_pending = False
    # Last commanded a_target across ALL finalize paths, including hold frames (which
    # never route through the release slew). Deliberately NOT cleared in
    # reset_lead_stop_hold: that runs on the release frame itself, and the release-slew
    # seed needs the prior hold command to bound the release step.
    self.final_a_prev = None
    self.departure_prediction_phase = PHASE_INACTIVE
    self.departure_prediction_phase_s = 0.0
    self.departure_prediction_track_id = -1
    self.departure_prediction_lockout_track_id = -1
    self.departure_prediction_frame_start_ready = False
    self.departure_prediction_applied = False
    self.departure_prediction_applied_track_id = -1
    self.departure_prediction_trace = DeparturePredictionTrace()
    self.uphill_net_demand_cap = NetDemandCapFinalStage()

  @staticmethod
  def _sm_item(sm: Any, key: str) -> Any:
    if hasattr(sm, 'get'):
      return sm.get(key)
    try:
      return sm[key]
    except Exception:
      return None

  @staticmethod
  def _select_stop_hold_lead(radar_state: Any, latched_id: Any = None) -> Any:
    candidates = []
    for lead in (getattr(radar_state, 'leadOne', None), getattr(radar_state, 'leadTwo', None)):
      if not lead_present(lead):
        continue
      try:
        d_rel = float(getattr(lead, 'dRel', 0.0))
        v = float(getattr(lead, 'vLead', 0.0))
        v_rel = float(getattr(lead, 'vRel', 0.0))
      except (TypeError, ValueError):
        continue
      if not (math.isfinite(d_rel) and math.isfinite(v) and math.isfinite(v_rel)) or d_rel <= 0.0:
        continue
      candidates.append((d_rel, v, v_rel, lead))
    if not candidates:
      return None
    # Route 282: the stopped-preference below un-selected the latched lead the moment it
    # accelerated past 0.5 m/s, breaking same_id exactly when the launch gate needed it
    # (hold hardened to -2.0, gate blocked on different_lead_id, latch died of the 0.5 s
    # timeout). The lead we stopped behind keeps its identity while the latch is alive.
    if latched_id is not None:
      for c in candidates:
        if _valid_lead_id(c[3]) == latched_id:
          return c[3]
    stopped = [c for c in candidates if c[1] <= 0.5]
    if stopped:
      return min(stopped, key=lambda c: c[0])[3]
    return min(candidates, key=lambda c: c[0])[3]

  @staticmethod
  def _routine_lead_launch_breakout(lead_v_rel: float) -> bool:
    return float(lead_v_rel) >= LEAD_CRAWL_BREAKOUT_MIN_OPENING

  def reset_lead_stop_hold(self) -> None:
    self.lead_stop_hold_active = False
    self.lead_stop_hold_gap_increasing_s = 0.0
    self.lead_stop_hold_missing_s = 0.0
    self.lead_stop_hold_lead_id = None
    self.lead_stop_hold_gap_prev_d_rel = None
    self.lead_stop_hold_prev_v = None
    self.lead_stop_hold_prev_y_rel = None
    self.lead_stop_hold_churn_ids.clear()
    self.lead_stop_hold_gap_baseline_d_rel = None
    self.lead_stop_hold_arm_d_rel = None
    self.lead_stop_hold_release_carry_s = 0.0
    self.lead_stop_hold_release_carry_a = 0.0
    self.stop_hold_release_slew_a_target = None
    self.stop_hold_release_prep_a_target = None
    self.stop_hold_release_prep_raw_prev = None
    self._clear_launch_floor_fade_state(clear_approach=True)
    self.low_speed_gap_closure_applied = False

  @staticmethod
  def _launch_floor_fade_hard_bypass(snapshot: _InputSnapshot) -> bool:
    output = snapshot.custom_long_output
    return bool(
      output is not None and (
        not bool(getattr(output, "enabled", False)) or
        bool(getattr(snapshot.custom_long, "fault_class", "")) or
        bool(getattr(output, "fault_class", ""))
      )
    )

  def _clear_launch_floor_fade_state(self, *, clear_approach: bool = False) -> None:
    self.launch_dip_grace_s = 0.0
    self.launch_floor_fade_pending = False
    if clear_approach:
      self.approach_damp_a_prev = None

  def _settle_stop_hold_arm_applies(self, v_ego: float, v_ego_stopping: float,
                                    lead_v: float, lead_v_rel: float,
                                    lead_d_rel: float, gas_pressed: bool,
                                    has_lead: bool) -> bool:
    snapshot = _InputSnapshot.__new__(_InputSnapshot)
    snapshot.v_ego = v_ego
    snapshot.v_ego_stopping = v_ego_stopping
    snapshot.lead_v = lead_v
    snapshot.lead_v_rel = lead_v_rel
    snapshot.lead_d_rel = lead_d_rel
    snapshot.gas_pressed = gas_pressed
    snapshot.has_lead = has_lead
    snapshot.stopping_distance = float(getattr(self.CP, 'stoppingDistance', 6.0) or 6.0)
    return _StopHoldLatchLifecycle.settle_arm_applies(self, snapshot)

  def _update_lead_stop_hold(self, sm: Any, v_ego: float, has_lead: bool, selected_lead: Any,
                             lead_d_rel: float, lead_v: float, lead_v_rel: float,
                             gas_pressed: bool, dt: float, reset_lead_stop_hold: Any) -> bool:
    snapshot = _InputSnapshot.build(
      self, sm,
      custom_long=None, custom_long_output=None,
      is_e2e=False, model_stale=False, dt=dt,
      mpc_a_target=0.0, mpc_should_stop=False,
      raw_model_a_target=0.0, raw_model_should_stop=False,
    )
    # Override snapshot fields that are independent of the custom-long state.
    snapshot.v_ego = v_ego
    snapshot.has_lead = has_lead
    snapshot.selected_lead = selected_lead
    snapshot.lead_d_rel = lead_d_rel
    snapshot.lead_v = lead_v
    snapshot.lead_v_rel = lead_v_rel
    snapshot.gas_pressed = gas_pressed
    snapshot.lead_id = _valid_lead_id(selected_lead)
    return _StopHoldLatchLifecycle.update(self, snapshot, reset_lead_stop_hold)

  def custom_longitudinal_should_stop(self, custom_long: Any, custom_long_output: Any,
                                      mpc_should_stop: bool, raw_model_should_stop: bool,
                                      model_stale: bool = False) -> bool | None:
    return _FinalArbitration.custom_longitudinal_should_stop(
      custom_long, custom_long_output, mpc_should_stop, raw_model_should_stop, model_stale
    )

  def _scc_custom_stop_cap(self, base_a_target: float, custom_long: Any, custom_long_output: Any) -> float:
    return _FinalArbitration.scc_custom_stop_cap(base_a_target, custom_long, custom_long_output)

  def _apply_stop_hold_release_slew(self, sm: Any, dt: float, a_target: float, release_mpc_stop: bool,
                                    mpc_stop: bool, raw_model_should_stop: bool, should_stop: bool) -> float:
    return _FinalArbitration.apply_release_slew(
      self, sm, dt, a_target, release_mpc_stop, mpc_stop, raw_model_should_stop, should_stop
    )

  def _stop_hold_release_prep_applies(self, sm: Any, selected_lead: Any, custom_long: Any, custom_long_output: Any,
                                      lead_d_rel: float, lead_v: float, lead_v_rel: float,
                                      mpc_a_target: float, raw_model_a_target: float,
                                      raw_model_should_stop: bool) -> bool:
    snapshot = _InputSnapshot.build(
      self, sm,
      custom_long=custom_long, custom_long_output=custom_long_output,
      is_e2e=False, model_stale=False, dt=0.0,
      mpc_a_target=mpc_a_target, mpc_should_stop=False,
      raw_model_a_target=raw_model_a_target, raw_model_should_stop=raw_model_should_stop,
    )
    snapshot.selected_lead = selected_lead
    snapshot.lead_d_rel = lead_d_rel
    snapshot.lead_v = lead_v
    snapshot.lead_v_rel = lead_v_rel
    snapshot.lead_id = _valid_lead_id(selected_lead)
    return _HoldCommand.prep_applies(self, snapshot)

  def _apply_stop_hold_release_prep(self, sm: Any, dt: float, raw_hold: float, selected_lead: Any,
                                    custom_long: Any, custom_long_output: Any,
                                    lead_d_rel: float, lead_v: float, lead_v_rel: float,
                                    mpc_a_target: float, raw_model_a_target: float,
                                    raw_model_should_stop: bool) -> float:
    snapshot = _InputSnapshot.build(
      self, sm,
      custom_long=custom_long, custom_long_output=custom_long_output,
      is_e2e=False, model_stale=False, dt=dt,
      mpc_a_target=mpc_a_target, mpc_should_stop=False,
      raw_model_a_target=raw_model_a_target, raw_model_should_stop=raw_model_should_stop,
    )
    snapshot.selected_lead = selected_lead
    snapshot.lead_d_rel = lead_d_rel
    snapshot.lead_v = lead_v
    snapshot.lead_v_rel = lead_v_rel
    snapshot.lead_id = _valid_lead_id(selected_lead)
    return _HoldCommand.apply_prep(self, snapshot, raw_hold)

  def _standstill_release_gate_enabled(self, custom_long: Any) -> bool:
    return _ReleaseGate.standstill_release_gate_enabled(self, custom_long)

  def _standstill_release_clears_mpc_stop(self, sm: Any, custom_long: Any, custom_long_output: Any,
                                          mpc_a_target: float, mpc_should_stop: bool,
                                          raw_model_a_target: float, raw_model_should_stop: bool) -> tuple[bool, float]:
    snapshot = _InputSnapshot.build(
      self, sm,
      custom_long=custom_long, custom_long_output=custom_long_output,
      is_e2e=False, model_stale=False, dt=0.0,
      mpc_a_target=mpc_a_target, mpc_should_stop=mpc_should_stop,
      raw_model_a_target=raw_model_a_target, raw_model_should_stop=raw_model_should_stop,
    )
    return _FinalArbitration.standstill_release_clears_mpc_stop(self, snapshot)

  def _standstill_release_request_valid(self, sm: Any, custom_long: Any, custom_long_output: Any,
                                        mpc_a_target: float, raw_model_a_target: float, raw_model_should_stop: bool,
                                        min_mpc_a_target: float = -0.03) -> bool:
    snapshot = _InputSnapshot.build(
      self, sm,
      custom_long=custom_long, custom_long_output=custom_long_output,
      is_e2e=False, model_stale=False, dt=0.0,
      mpc_a_target=mpc_a_target, mpc_should_stop=False,
      raw_model_a_target=raw_model_a_target, raw_model_should_stop=raw_model_should_stop,
    )
    return _ReleaseGate.standstill_release_request_valid(self, snapshot, min_mpc_a_target)

  def _apply_follow_coast_band(self, a_target: float, snapshot: _InputSnapshot,
                               should_stop: bool, release_mpc_stop: bool) -> float:
    """Suppress accel<->decel sign chatter around the follow-gap equilibrium.

    Only shapes shallow negative demands while following a moving lead: HOLD clamps
    demands above the band floor to 0, DECEL passes everything through untouched. Any
    stop, release, pedal, or force-decel context drops the state and passes through, so
    real braking is never delayed by more than the band depth itself.
    """
    # Same tick-fresh coast source as the catch-up cap: degraded/fault/disabled outputs
    # leave accel_coast at 0.0, which collapses the band to passthrough (fail-closed).
    band_lo = max(min(float(getattr(snapshot.custom_long_output, "accel_coast", 0.0) or 0.0), 0.0),
                  self._FOLLOW_BAND_MAX_DEPTH)
    if (not math.isfinite(a_target) or not math.isfinite(band_lo)
        or band_lo > self._FOLLOW_BAND_MIN_DEPTH
        or should_stop or release_mpc_stop or self.stop_hold_release_slew_a_target is not None
        or snapshot.gas_pressed or snapshot.brake_pressed or snapshot.force_decel
        or not snapshot.has_lead or snapshot.v_ego < self._FOLLOW_BAND_MIN_V_EGO):
      self._set_follow_band_regime(None)
      return float(a_target)
    if self.follow_band_regime is None:
      # Joining mid-decel stays honest: any negative demand starts in DECEL, not clamped.
      self._set_follow_band_regime("decel" if a_target < 0.0 else "hold")
    if self.follow_band_regime == "decel":
      if a_target < self._FOLLOW_BAND_EXIT_A:
        return float(a_target)
      self._set_follow_band_regime("hold")
    if a_target <= band_lo:
      self._set_follow_band_regime("decel")
      return float(a_target)
    if a_target < 0.0:
      # Only clamped frames spend budget: a positive demand is not being suppressed.
      dt = float(getattr(snapshot, "dt", 0.0) or 0.0)
      v_rel = float(getattr(snapshot, "lead_v_rel", 0.0) or 0.0)
      if math.isfinite(dt) and dt > 0.0 and math.isfinite(v_rel):
        self.follow_band_given_m = max(0.0, self.follow_band_given_m - v_rel * dt)
      if self.follow_band_given_m > self._FOLLOW_BAND_MAX_GIVEN_S * snapshot.v_ego:
        # The gap is not recovering on its own; the demand is honest evidence, not dither.
        self._set_follow_band_regime("decel")
        return float(a_target)
    return max(float(a_target), 0.0)

  def _set_follow_band_regime(self, regime: str | None) -> None:
    """Set the band regime, resetting the giveaway budget on every regime change."""
    if regime != self.follow_band_regime:
      self.follow_band_given_m = 0.0
    self.follow_band_regime = regime

  def _apply_approach_damp(self, a_target: float, should_stop: bool, release_mpc_stop: bool,
                           dt: float, v_ego: float | None = None) -> float:
    """Jerk-limit aTarget inside the gentle authority band to kill the ACC-MPC approach-cusp limit cycle.

    Only active above crawl speed when the command is small (|a| <= band), not stopping,
    not releasing a stop hold, and no stop-hold release ramp is in progress. Crawl uses
    the gap governor directly so this state cannot carry an old owner's sign through it.
    """
    crawl = v_ego is not None and (not math.isfinite(v_ego) or v_ego < self._APPROACH_DAMP_MIN_V_EGO)
    if (not math.isfinite(a_target) or not math.isfinite(dt) or dt <= 0.0 or crawl
        or should_stop or release_mpc_stop or abs(a_target) > self._APPROACH_DAMP_BAND
        or self.stop_hold_release_slew_a_target is not None):
      self.approach_damp_a_prev = None
      return float(a_target)
    prev = self.approach_damp_a_prev
    if prev is None or not math.isfinite(prev) or abs(prev) > self._APPROACH_DAMP_BAND:
      self.approach_damp_a_prev = float(a_target)
      return float(a_target)
    max_step = self._APPROACH_DAMP_MAX_JERK * dt
    limited = min(max(float(a_target), prev - max_step), prev + max_step)
    self.approach_damp_a_prev = float(limited)
    return float(limited)

  def _apply_launch_dip_damp(self, a_target: float, snapshot: _InputSnapshot, should_stop: bool, dt: float) -> float:
    """Rate-limit transient positive-accel dips in the first seconds after a stop-hold release.

    Route 282 rlog (t=42244-42246): radar/vision lead churn during pullaway punched 1-2 frame
    mpcA dips (1.3 -> 0.2 m/s^2) through final arbitration — masked then by driver gas,
    unmasked it reads as launch surging. Only positive-to-positive downward steps are damped,
    only while a radar lead is confirmed departing at low ego speed inside the release grace
    window, and never when any stop, brake, gas, or force-decel flag is set — genuine braking
    demands pass through untouched. A persistent (real) reduction still arrives at
    _APPROACH_DAMP_MAX_JERK within ~0.4 s.
    """
    prev = self.final_a_prev
    if (self.launch_dip_grace_s <= 0.0
        or not math.isfinite(a_target) or not math.isfinite(dt) or dt <= 0.0
        or prev is None or not math.isfinite(prev)
        or should_stop or snapshot.mpc_should_stop or snapshot.raw_model_should_stop
        or snapshot.brake_pressed or snapshot.gas_pressed or snapshot.force_decel
        or a_target <= 0.0 or prev <= 0.0 or a_target >= prev
        or not snapshot.has_lead
        or snapshot.lead_v < 0.30 or snapshot.lead_v_rel < 0.15
        or snapshot.v_ego >= self._LAUNCH_DIP_MAX_V_EGO):
      return float(a_target)
    return float(max(a_target, prev - self._APPROACH_DAMP_MAX_JERK * dt))

  def _apply_launch_floor_fade(self, a_target: float, snapshot: _InputSnapshot,
                               should_stop: bool, release_mpc_stop: bool, dt: float) -> float:
    """Taper a bound launch floor into the first positive post-grace MPC target."""
    try:
      target = float(a_target)
    except (TypeError, ValueError):
      self._clear_launch_floor_fade_state(clear_approach=True)
      return a_target

    custom_long = snapshot.custom_long
    output = snapshot.custom_long_output
    lifecycle_active = bool(
      not snapshot.is_e2e and snapshot.long_active and
      custom_long is not None and getattr(custom_long, "mode", None) is LongitudinalMode.SCC and
      bool(getattr(custom_long, "enabled", False)) and
      output is not None and bool(getattr(output, "enabled", False)) and
      not bool(getattr(custom_long, "fault_class", "")) and
      not bool(getattr(output, "fault_class", ""))
    )
    if not lifecycle_active:
      self._clear_launch_floor_fade_state(clear_approach=self._launch_floor_fade_hard_bypass(snapshot))
      return target
    if not self.launch_floor_fade_pending:
      return target
    if self.launch_dip_grace_s > 0.0:
      return target

    custom_stop = bool(
      getattr(output, "should_stop", False) or
      getattr(output, "model_stop_corroborated", False) or
      str(getattr(output, "selected_intent", "") or "") == "stop_approach"
    )
    try:
      previous = float(self.final_a_prev)
      dt_f = float(dt)
      mpc_a = float(snapshot.mpc_a_target)
      raw_model_a = float(snapshot.raw_model_a_target)
      lead_d_rel = float(snapshot.lead_d_rel)
      lead_v = float(snapshot.lead_v)
      lead_v_rel = float(snapshot.lead_v_rel)
      v_ego = float(snapshot.v_ego)
      stopping_distance = float(snapshot.stopping_distance)
    except (TypeError, ValueError):
      self._clear_launch_floor_fade_state(clear_approach=True)
      return target

    if (
      not math.isfinite(target) or not math.isfinite(previous) or
      not math.isfinite(dt_f) or dt_f <= 0.0 or
      not snapshot.mpc_a_target_valid or not snapshot.raw_model_a_target_valid or
      not math.isfinite(mpc_a) or not math.isfinite(raw_model_a) or
      snapshot.model_stale or should_stop or release_mpc_stop or
      snapshot.mpc_should_stop or snapshot.raw_model_should_stop or custom_stop or
      mpc_a < 0.0 or raw_model_a < 0.0 or
      snapshot.brake_pressed or snapshot.gas_pressed or snapshot.force_decel or
      not snapshot.has_lead or not math.isfinite(lead_d_rel) or
      not math.isfinite(lead_v) or not math.isfinite(lead_v_rel) or
      not math.isfinite(v_ego) or not math.isfinite(stopping_distance) or
      lead_v < LEAD_CRAWL_BREAKOUT_MIN_OPENING or
      lead_v_rel < self._STOP_HOLD_LAUNCH_FLOOR_MIN_OPENING or
      lead_d_rel < stopping_distance or v_ego < 0.0 or
      v_ego > self._LAUNCH_DIP_MAX_V_EGO or target <= 0.0 or
      previous <= 0.0 or target >= previous
    ):
      self._clear_launch_floor_fade_state(clear_approach=True)
      return target

    faded_target = max(target, previous - self._APPROACH_DAMP_MAX_JERK * dt_f)
    if not math.isfinite(faded_target) or faded_target <= target:
      self._clear_launch_floor_fade_state(clear_approach=True)
      return target
    self.approach_damp_a_prev = None
    return float(faded_target)

  def _stop_hold_release_accel_for_gap(self, requested_a: float, lead_d_rel: float,
                                       lead_v: float, lead_v_rel: float, same_id: bool,
                                       valid_source: bool = False) -> float:
    return _ReleaseAccel.accel_for_gap(self, requested_a, lead_d_rel, lead_v, lead_v_rel, same_id, valid_source)

  def _lead_stop_hold_crawl_fallback_applies(self, sm: Any, custom_long: Any, custom_long_output: Any,
                                             mpc_a_target: float, raw_model_a_target: float,
                                             raw_model_should_stop: bool, selected_lead: Any,
                                             lead_d_rel: float, lead_v: float, lead_v_rel: float,
                                             same_id: bool) -> bool:
    snapshot = _InputSnapshot.build(
      self, sm,
      custom_long=custom_long, custom_long_output=custom_long_output,
      is_e2e=False, model_stale=False, dt=0.0,
      mpc_a_target=mpc_a_target, mpc_should_stop=False,
      raw_model_a_target=raw_model_a_target, raw_model_should_stop=raw_model_should_stop,
    )
    snapshot.selected_lead = selected_lead
    snapshot.lead_d_rel = lead_d_rel
    snapshot.lead_v = lead_v
    snapshot.lead_v_rel = lead_v_rel
    snapshot.lead_id = _valid_lead_id(selected_lead)
    return _ReleaseGate.crawl_fallback_applies(self, snapshot, same_id)

  def _lead_stop_hold_release_accepts(self, sm: Any, custom_long: Any, custom_long_output: Any,
                                      mpc_a_target: float, raw_model_a_target: float,
                                      raw_model_should_stop: bool, selected_lead: Any, lead_d_rel: float,
                                      lead_v: float, lead_v_rel: float) -> tuple[bool, float]:
    snapshot = _InputSnapshot.build(
      self, sm,
      custom_long=custom_long, custom_long_output=custom_long_output,
      is_e2e=False, model_stale=False, dt=0.0,
      mpc_a_target=mpc_a_target, mpc_should_stop=False,
      raw_model_a_target=raw_model_a_target, raw_model_should_stop=raw_model_should_stop,
    )
    snapshot.selected_lead = selected_lead
    snapshot.lead_d_rel = lead_d_rel
    snapshot.lead_v = lead_v
    snapshot.lead_v_rel = lead_v_rel
    snapshot.lead_id = _valid_lead_id(selected_lead)
    return _ReleaseGate.release_accepts(self, snapshot)

  @staticmethod
  def _finite_float_or_none(value) -> float | None:
    try:
      v = float(value)
    except (TypeError, ValueError):
      return None
    return v if math.isfinite(v) else None

  @staticmethod
  def _safe_float(value, default: float = 0.0) -> float:
    try:
      v = float(value)
    except (TypeError, ValueError):
      return default
    return v if math.isfinite(v) else default

  def _departure_prediction_evidence(self, custom_long_output: Any) -> DeparturePredictionEvidence:
    evidence = getattr(custom_long_output, "departure_prediction_evidence", None)
    return evidence if isinstance(evidence, DeparturePredictionEvidence) else DeparturePredictionEvidence()

  def _clear_departure_prediction_phase(self, *, clear_lockout: bool = False) -> None:
    self.departure_prediction_phase = PHASE_INACTIVE
    self.departure_prediction_phase_s = 0.0
    self.departure_prediction_track_id = -1
    self.departure_prediction_frame_start_ready = False
    if clear_lockout:
      self.departure_prediction_lockout_track_id = -1

  def _reset_departure_prediction_phase(self, *, clear_lockout: bool = True) -> None:
    self._clear_departure_prediction_phase(clear_lockout=clear_lockout)
    self.departure_prediction_trace = DeparturePredictionTrace()

  @staticmethod
  def _departure_prediction_int(value: Any, default: int = -1) -> int:
    try:
      return int(value)
    except (TypeError, ValueError):
      return default

  def _departure_prediction_trace_from_evidence(self, evidence: DeparturePredictionEvidence) -> DeparturePredictionTrace:
    return DeparturePredictionTrace(
      mode=str(getattr(evidence, "mode", "off") or "off"),
      effective_mode=str(getattr(evidence, "effective_mode", "off") or "off"),
      apply_supported=bool(getattr(evidence, "apply_supported", False)),
      lead_idx=self._departure_prediction_int(getattr(evidence, "lead_idx", -1)),
      track_id=self._departure_prediction_int(getattr(evidence, "track_id", -1)),
      predicted_gap_delta=self._safe_float(getattr(evidence, "predicted_gap_growth_1s", 0.0)),
      research_actuation_allowed=bool(getattr(evidence, "research_actuation_allowed", False)),
      block_reason=str(getattr(evidence, "block_reason", "") or ""),
      fault=bool(getattr(evidence, "fault", False)),
    )

  def _departure_prediction_exact_track(self, snapshot: _InputSnapshot, evidence: Any,
                                        latched_id: Any = None) -> bool:
    selected_id = snapshot.lead_id
    latched_id = self.lead_stop_hold_lead_id if latched_id is None else latched_id
    evidence_id = self._departure_prediction_int(getattr(evidence, "track_id", -1))
    selected = snapshot.selected_lead
    if (
      selected is None or not lead_present(selected) or
      selected_id is None or latched_id is None or evidence_id < 0
    ):
      return False
    if not bool(getattr(evidence, "radar", False)):
      return False
    try:
      return int(selected_id) == int(latched_id) == evidence_id
    except (TypeError, ValueError):
      return False

  def _departure_prediction_phase_gate(self, snapshot: _InputSnapshot, evidence: Any,
                                      pre_hold_active: bool, dt_s: float) -> tuple[bool, str]:
    custom_long = snapshot.custom_long
    if custom_long is None or not bool(getattr(custom_long, "enabled", False)):
      return False, "custom_long_disabled"
    if snapshot.custom_long_output is None or not bool(getattr(snapshot.custom_long_output, "enabled", False)):
      return False, "custom_long_disabled"
    if bool(getattr(custom_long, "fault_class", "")) or bool(getattr(snapshot.custom_long_output, "fault_class", "")):
      return False, "evidence_fault"
    if getattr(custom_long, "mode", None) is not LongitudinalMode.SCC:
      return False, "mode_not_scc"
    if not snapshot.long_active:
      return False, "long_inactive"
    if not bool(getattr(self.CP, "openpilotLongitudinalControl", False)):
      return False, "openpilot_longitudinal_disabled"
    if not pre_hold_active:
      return False, "no_stop_hold"
    if not math.isfinite(dt_s) or dt_s <= 0.0:
      return False, "invalid_dt"
    if snapshot.brake_pressed or snapshot.gas_pressed:
      return False, "driver_override"
    if snapshot.force_decel:
      return False, "force_decel"
    if bool(getattr(evidence, "fault", False)):
      return False, "evidence_fault"
    if not bool(getattr(evidence, "eligible", False)):
      return False, str(getattr(evidence, "block_reason", "ineligible") or "ineligible")
    if not self._departure_prediction_exact_track(snapshot, evidence):
      return False, "different_lead_id"
    if bool(snapshot.raw_model_should_stop) or self._departure_prediction_threat_active(snapshot, evidence):
      return False, "threat_active"
    return True, ""

  def _update_departure_prediction_phase(self, snapshot: _InputSnapshot,
                                         pre_hold_active: bool) -> DeparturePredictionEvidence:
    evidence = self._departure_prediction_evidence(snapshot.custom_long_output)
    trace = self._departure_prediction_trace_from_evidence(evidence)
    self.departure_prediction_trace = trace
    self.departure_prediction_frame_start_ready = False

    try:
      dt_f = float(snapshot.dt)
    except (TypeError, ValueError):
      dt_f = 0.0
    dt_s = dt_f if math.isfinite(dt_f) and dt_f > 0.0 else 0.0
    track_id = self._departure_prediction_int(getattr(evidence, "track_id", -1))
    raw_eligible = bool(getattr(evidence, "eligible", False)) and track_id >= 0

    if trace.effective_mode == "off" or trace.fault:
      self._reset_departure_prediction_phase()
      self.departure_prediction_trace = replace(trace, phase=PHASE_INACTIVE, phase_s=0.0)
      return evidence
    if not raw_eligible:
      # A real predictor-evidence drop is the event that releases a timeout lockout.
      self._clear_departure_prediction_phase(clear_lockout=True)
      self.departure_prediction_trace = replace(trace, phase=PHASE_INACTIVE, phase_s=0.0)
      return evidence

    if track_id != self.departure_prediction_track_id:
      self._clear_departure_prediction_phase(clear_lockout=False)
      self.departure_prediction_track_id = track_id
      if self.departure_prediction_lockout_track_id != track_id:
        self.departure_prediction_lockout_track_id = -1

    phase_ok, phase_reason = self._departure_prediction_phase_gate(snapshot, evidence, pre_hold_active, dt_s)
    if not phase_ok:
      # Pedals, force decel, and a threat clear readiness immediately. A timeout lockout is
      # deliberately retained while the same predictor track remains continuously eligible.
      self._clear_departure_prediction_phase(clear_lockout=False)
      self.departure_prediction_trace = replace(
        trace, phase=PHASE_INACTIVE, phase_s=0.0, block_reason=phase_reason,
      )
      return evidence

    if self.departure_prediction_lockout_track_id == track_id:
      self._clear_departure_prediction_phase(clear_lockout=False)
      self.departure_prediction_track_id = track_id
      self.departure_prediction_trace = replace(
        trace, phase=PHASE_INACTIVE, phase_s=0.0, block_reason="prediction_timeout",
      )
      return evidence

    frame_start_predicted = bool(
      self.departure_prediction_phase == PHASE_PREDICTED and
      self.departure_prediction_track_id == track_id
    )
    if self.departure_prediction_phase == PHASE_PREDICTED:
      self.departure_prediction_phase_s += dt_s
      if self.departure_prediction_phase_s >= TIMEOUT_S:
        self.departure_prediction_phase = PHASE_INACTIVE
        self.departure_prediction_phase_s = 0.0
        self.departure_prediction_lockout_track_id = track_id
        frame_start_predicted = False
        phase_reason = "prediction_timeout"
    elif self.departure_prediction_phase == PHASE_INACTIVE:
      self.departure_prediction_phase = PHASE_ARMING
      self.departure_prediction_phase_s = dt_s
      if self.departure_prediction_phase_s >= PERSISTENCE_S:
        self.departure_prediction_phase = PHASE_PREDICTED
        self.departure_prediction_phase_s = 0.0
    elif self.departure_prediction_phase == PHASE_ARMING:
      self.departure_prediction_phase_s += dt_s
      if self.departure_prediction_phase_s >= PERSISTENCE_S:
        self.departure_prediction_phase = PHASE_PREDICTED
        self.departure_prediction_phase_s = 0.0

    self.departure_prediction_frame_start_ready = frame_start_predicted
    self.departure_prediction_trace = replace(
      trace,
      phase=self.departure_prediction_phase,
      phase_s=float(self.departure_prediction_phase_s),
      evidence_s=(PERSISTENCE_S if self.departure_prediction_phase == PHASE_PREDICTED else
                  float(self.departure_prediction_phase_s) if self.departure_prediction_phase == PHASE_ARMING else 0.0),
      age_s=(float(self.departure_prediction_phase_s) if self.departure_prediction_phase == PHASE_PREDICTED else 0.0),
      block_reason=phase_reason,
    )
    return evidence

  @staticmethod
  def _departure_prediction_threat_active(snapshot: _InputSnapshot, evidence: Any) -> bool:
    if bool(getattr(evidence, "alternate_threat_active", False)):
      return True
    if bool(snapshot.raw_model_should_stop) or _model_stop_blocks_release(snapshot):
      return True
    output = snapshot.custom_long_output
    if (bool(getattr(output, "should_stop", False)) or
        bool(getattr(output, "model_stop_corroborated", False)) or
        str(getattr(output, "selected_intent", "") or "") == "stop_approach"):
      return True
    actuation = getattr(output, "actuation", None)
    if actuation is None:
      return False
    cut_in = getattr(actuation, "cut_in_brake_assist", None)
    if cut_in is not None:
      proposed = CustomLongitudinalFinalizer._finite_float_or_none(getattr(cut_in, "proposed_cap", None))
      if bool(getattr(cut_in, "eligible", False)) or (proposed is not None and proposed < 0.0):
        return True
    curve = getattr(actuation, "curve_traffic_advisor", None)
    if curve is not None:
      proposed = CustomLongitudinalFinalizer._finite_float_or_none(getattr(curve, "a_curve_cap_proposed", None))
      if (bool(getattr(curve, "eligible", False)) or bool(getattr(curve, "active", False)) or
          bool(getattr(curve, "suppress_accel", False)) or (proposed is not None and proposed < 0.0)):
        return True
    return False

  def _record_departure_prediction_context(self, snapshot: _InputSnapshot, a_target: float,
                                           should_stop: bool, release_mpc_stop: bool,
                                           pre_hold_active: bool, post_hold_active: bool,
                                           release_slew_provenance: bool,
                                           pre_hold_lead_id: Any, block_reason: str) -> float:
    before = self._safe_float(a_target)
    evidence = self._departure_prediction_evidence(snapshot.custom_long_output)
    if block_reason == "stop_hold_active" and self.departure_prediction_trace.block_reason:
      block_reason = self.departure_prediction_trace.block_reason
    same_track = bool(
      pre_hold_active and not post_hold_active and
      self._departure_prediction_exact_track(snapshot, evidence, pre_hold_lead_id) and
      self._departure_prediction_int(pre_hold_lead_id) == snapshot.lead_id
    )
    self.departure_prediction_trace = replace(
      self.departure_prediction_trace,
      pre_hold_active=bool(pre_hold_active),
      post_hold_active=bool(post_hold_active),
      same_track=same_track,
      release_source=str(getattr(snapshot.custom_long_output, "standstill_release_source", "") or ""),
      release_permission=bool(getattr(snapshot.custom_long_output, "standstill_release_allowed", False)),
      release_mpc_stop=bool(release_mpc_stop),
      release_slew_provenance=bool(release_slew_provenance),
      a_target_before=before,
      a_target_proposed=before,
      a_target_after=before,
      block_reason=block_reason,
    )
    return a_target

  def _apply_departure_prediction(self, snapshot: _InputSnapshot, a_target: float,
                                  should_stop: bool, release_mpc_stop: bool,
                                  pre_hold_active: bool, post_hold_active: bool,
                                  release_slew_provenance: bool = False,
                                  pre_hold_lead_id: Any = None,
                                  frame_start_predicted: bool = False,
                                  pre_slew_state: Any = None,
                                  pre_slew_input: Any = None,
                                  post_slew_state: Any = None,
                                  post_slew_target: Any = None) -> float:
    """Apply only the approved shallow-brake-to-coast transformation."""
    try:
      before = float(a_target)
      evidence = self._departure_prediction_evidence(snapshot.custom_long_output)
      input_target = self._finite_float_or_none(pre_slew_input)
      post_target = self._finite_float_or_none(post_slew_target)
      post_state = self._finite_float_or_none(post_slew_state)
      release_slew_provenance = bool(
        release_mpc_stop and pre_hold_active and not post_hold_active and
        pre_slew_state is None and input_target is not None and input_target >= 0.0 and
        post_target is not None and post_slew_state is not None and post_state is not None and
        post_state == post_target and post_target < input_target
      )
      evidence_track_id = self._departure_prediction_int(getattr(evidence, "track_id", -1))
      same_track = bool(
        pre_hold_active and not post_hold_active and snapshot.lead_id is not None and
        pre_hold_lead_id is not None and self._departure_prediction_exact_track(snapshot, evidence, pre_hold_lead_id) and
        self._departure_prediction_int(pre_hold_lead_id) == int(snapshot.lead_id) and
        evidence_track_id == int(snapshot.lead_id)
      )
      self.departure_prediction_trace = replace(
        self.departure_prediction_trace,
        pre_hold_active=bool(pre_hold_active),
        post_hold_active=bool(post_hold_active),
        same_track=same_track,
        release_source=str(getattr(snapshot.custom_long_output, "standstill_release_source", "") or ""),
        release_permission=bool(getattr(snapshot.custom_long_output, "standstill_release_allowed", False)),
        release_mpc_stop=bool(release_mpc_stop),
        release_slew_provenance=release_slew_provenance,
        a_target_before=before if math.isfinite(before) else 0.0,
        a_target_proposed=before if math.isfinite(before) else 0.0,
        a_target_after=before if math.isfinite(before) else 0.0,
      )

      def blocked(reason: str, *, fault: bool = False) -> float:
        self.departure_prediction_trace = replace(
          self.departure_prediction_trace, block_reason=reason, fault=fault,
        )
        return a_target

      if not math.isfinite(before):
        return blocked("nonfinite_target", fault=True)
      if self.departure_prediction_trace.effective_mode == "off":
        return blocked("non_actuating_mode")
      if bool(getattr(evidence, "fault", False)):
        return blocked("evidence_fault", fault=True)
      evidence_values = (
        getattr(evidence, "d_rel", 0.0), getattr(evidence, "v_lead", 0.0),
        getattr(evidence, "v_rel", 0.0), getattr(evidence, "a_lead_k", 0.0),
        getattr(evidence, "predicted_gap_1s", 0.0),
        getattr(evidence, "predicted_gap_growth_1s", 0.0),
      )
      if any(not math.isfinite(float(value)) for value in evidence_values):
        return blocked("nonfinite_evidence", fault=True)
      if not bool(getattr(evidence, "eligible", False)):
        return blocked(str(getattr(evidence, "block_reason", "ineligible") or "ineligible"))
      if snapshot.custom_long is None or getattr(snapshot.custom_long, "mode", None) is not LongitudinalMode.SCC:
        return blocked("mode_not_scc")
      if not bool(getattr(snapshot.custom_long, "enabled", False)) or not bool(getattr(snapshot.custom_long_output, "enabled", False)):
        return blocked("custom_long_disabled")
      if bool(getattr(snapshot.custom_long, "fault_class", "")) or bool(getattr(snapshot.custom_long_output, "fault_class", "")):
        return blocked("evidence_fault", fault=True)
      if not snapshot.long_active:
        return blocked("long_inactive")
      if not bool(getattr(self.CP, "openpilotLongitudinalControl", False)):
        return blocked("openpilot_longitudinal_disabled")
      if not frame_start_predicted:
        return blocked("prediction_not_persistent")
      if self.departure_prediction_applied:
        return blocked("already_applied")
      if not pre_hold_active or post_hold_active or not release_mpc_stop:
        return blocked("not_first_release_frame")
      if not release_slew_provenance:
        return blocked("release_slew_provenance")
      if not bool(getattr(snapshot.custom_long_output, "standstill_release_allowed", False)):
        return blocked("no_release_permission")
      if self.departure_prediction_trace.release_source not in ("lead_pullaway", "lead_standstill_launch"):
        return blocked("invalid_release_source")
      if should_stop:
        return blocked("should_stop")
      if snapshot.brake_pressed or snapshot.gas_pressed or snapshot.force_decel:
        return blocked("driver_or_force_block")
      if self._departure_prediction_threat_active(snapshot, evidence):
        return blocked("threat_active")
      if snapshot.model_stale or snapshot.mpc_should_stop or snapshot.raw_model_should_stop:
        return blocked("mpc_or_model_stop")
      if not snapshot.mpc_a_target_valid or not snapshot.raw_model_a_target_valid:
        return blocked("invalid_raw_target")
      if (not math.isfinite(float(snapshot.mpc_a_target)) or snapshot.mpc_a_target < 0.0 or
          not math.isfinite(float(snapshot.raw_model_a_target)) or snapshot.raw_model_a_target < 0.0):
        return blocked("mpc_brake_or_stop")
      if not same_track:
        return blocked("different_lead_id")
      measured_departure = bool(
        math.isfinite(float(snapshot.lead_v)) and math.isfinite(float(snapshot.lead_v_rel)) and
        math.isfinite(float(snapshot.lead_d_rel)) and math.isfinite(float(snapshot.stopping_distance)) and
        snapshot.lead_v >= 0.30 and snapshot.lead_v_rel >= 0.15 and
        snapshot.lead_d_rel >= snapshot.stopping_distance + 0.20
      )
      self.departure_prediction_trace = replace(self.departure_prediction_trace, measured_departure=measured_departure, threat_free=True)
      if not measured_departure:
        return blocked("measured_departure_not_confirmed")
      if not -0.20 <= before < 0.0:
        return blocked("target_outside_coast_band")

      self.departure_prediction_trace = replace(
        self.departure_prediction_trace,
        eligible=True,
        would_coast=True,
        a_target_proposed=0.0,
        delta_a=0.0,
      )
      research_allowed = bool(getattr(evidence, "research_actuation_allowed", False))
      if self.departure_prediction_trace.effective_mode == "apply" and not research_allowed:
        return blocked("research_actuation_gate")
      if (str(getattr(evidence, "mode", "") or "") != "apply" or
          self.departure_prediction_trace.effective_mode != "apply" or
          not self.departure_prediction_trace.apply_supported):
        return blocked("non_actuating_mode")
      self.departure_prediction_applied = True
      self.departure_prediction_applied_track_id = evidence_track_id
      self.departure_prediction_trace = replace(
        self.departure_prediction_trace,
        applied=True,
        a_target_after=0.0,
        delta_a=-before,
        block_reason="",
      )
      return 0.0
    except Exception:
      self.departure_prediction_trace = replace(self.departure_prediction_trace, block_reason="internal_error", fault=True)
      return a_target

  def finalize(self, sm: Any, custom_long: Any, custom_long_output: Any, is_e2e: bool,
               model_stale: bool, dt: float, mpc_a_target: float, mpc_should_stop: bool,
               raw_model_a_target: float, raw_model_should_stop: bool,
               apply_stop_hold_release_slew: Any, reset_lead_stop_hold: Any) -> FinalizerResult:
    """Run ``_finalize_impl`` and record the commanded accel for the next frame.

    ``final_a_prev`` must see every commanded a_target — hold frames included —
    so the release-slew seed can bound the hold->release step.
    """
    result = self._finalize_impl(
      sm, custom_long, custom_long_output, is_e2e, model_stale, dt,
      mpc_a_target, mpc_should_stop, raw_model_a_target, raw_model_should_stop,
      apply_stop_hold_release_slew, reset_lead_stop_hold,
    )
    result.departure_prediction_evidence = self._departure_prediction_evidence(custom_long_output)
    evidence = getattr(custom_long_output, "uphill_net_demand", None)
    trace = NetDemandCapTrace(a_target_before=result.a_target, a_target_cap=result.a_target, a_target_after=result.a_target)
    if evidence is not None:
      try:
        capped_target, trace = self.uphill_net_demand_cap.apply(
          result.a_target, evidence, should_stop=result.should_stop, dt=dt,
        )
        result.a_target = capped_target
      except Exception:
        if self.uphill_net_demand_cap.applied_last_tick:
          custom_long.fault_class = FAULT_CLASS_INTERNAL
        trace = replace(
          trace,
          mode=str(getattr(evidence, "mode", "off")),
          effective_mode="shadow",
          block_reason="internal_error",
        )

    if (bool(getattr(custom_long, "fault_class", "")) or
        bool(getattr(custom_long_output, "fault_class", ""))):
      self._clear_launch_floor_fade_state(clear_approach=True)

    telemetry = result.custom_long_output_telemetry or custom_long_output
    final_target = float(result.a_target) if math.isfinite(float(result.a_target)) else 0.0
    final_before = self.departure_prediction_trace.a_target_before
    departure_trace = replace(
      self.departure_prediction_trace,
      a_target_after=final_target if self.departure_prediction_trace.applied else self.departure_prediction_trace.a_target_after,
      a_target_final=final_target,
      delta_a=(final_target - final_before if self.departure_prediction_trace.applied else self.departure_prediction_trace.delta_a),
    )
    self.departure_prediction_trace = departure_trace
    result.departure_prediction_trace = departure_trace
    if isinstance(telemetry, CustomLongitudinalOutput):
      result.custom_long_output_telemetry = replace(
        telemetry,
        departure_prediction_evidence=result.departure_prediction_evidence,
        departure_prediction_trace=departure_trace,
        uphill_net_demand_trace=trace,
      )
    self.final_a_prev = float(result.a_target) if math.isfinite(result.a_target) else None
    return result

  def _finalize_impl(self, sm: Any, custom_long: Any, custom_long_output: Any, is_e2e: bool,
                     model_stale: bool, dt: float, mpc_a_target: float, mpc_should_stop: bool,
                     raw_model_a_target: float, raw_model_should_stop: bool,
                     apply_stop_hold_release_slew: Any, reset_lead_stop_hold: Any) -> FinalizerResult:
    """Return the final longitudinal arbitration tuple.

    ``apply_stop_hold_release_slew`` and ``reset_lead_stop_hold`` are supplied by the
    planner so that live instrumentation (e.g. ``tools/drive_lab`` monkeypatches against
    ``LongitudinalPlannerSP`` methods) remains in the loop.
    """
    # Preserve source validity before the baseline-compatible finite fallbacks below.
    try:
      mpc_a_target_valid = math.isfinite(float(mpc_a_target))
    except (TypeError, ValueError):
      mpc_a_target_valid = False
    try:
      raw_model_a_target_valid = math.isfinite(float(raw_model_a_target))
    except (TypeError, ValueError):
      raw_model_a_target_valid = False
    mpc_a_target = self._safe_float(mpc_a_target)
    raw_model_a_target = self._safe_float(raw_model_a_target, mpc_a_target)

    if not bool(getattr(custom_long, "enabled", False)):
      self._clear_launch_floor_fade_state(clear_approach=True)
      reset_lead_stop_hold()
      self.stop_hold_release_sustain_s = 0.0
      self.custom_long_output_telemetry = None
      self.last_release_block_reason = ""
      self._set_follow_band_regime(None)
      self._reset_departure_prediction_phase(clear_lockout=False)
      self.departure_prediction_applied = False
      self.departure_prediction_applied_track_id = -1
      if is_e2e and not model_stale:
        a_target = min(raw_model_a_target, mpc_a_target)
        return _TelemetryAdapter.result(
          a_target, mpc_should_stop or raw_model_should_stop, a_target < mpc_a_target, None, ""
        )
      return _TelemetryAdapter.result(
        mpc_a_target, mpc_should_stop, False, None, ""
      )

    snapshot = _InputSnapshot.build(
      self, sm, custom_long, custom_long_output, is_e2e, model_stale, dt,
      mpc_a_target, mpc_should_stop, raw_model_a_target, raw_model_should_stop,
      mpc_a_target_valid=mpc_a_target_valid,
      raw_model_a_target_valid=raw_model_a_target_valid,
    )
    fade_lifecycle_active = bool(
      not snapshot.is_e2e and snapshot.long_active and
      getattr(snapshot.custom_long, "mode", None) is LongitudinalMode.SCC and
      bool(getattr(snapshot.custom_long, "enabled", False)) and
      snapshot.custom_long_output is not None and
      bool(getattr(snapshot.custom_long_output, "enabled", False)) and
      not bool(getattr(snapshot.custom_long, "fault_class", "")) and
      not bool(getattr(snapshot.custom_long_output, "fault_class", ""))
    )
    fade_hard_bypass = self._launch_floor_fade_hard_bypass(snapshot)
    if not fade_lifecycle_active:
      self._clear_launch_floor_fade_state(clear_approach=fade_hard_bypass)
    pre_hold_active = bool(self.lead_stop_hold_active)
    pre_hold_lead_id = self.lead_stop_hold_lead_id
    self._update_departure_prediction_phase(snapshot, pre_hold_active)
    frame_start_predicted = bool(self.departure_prediction_frame_start_ready)
    model_stop_blocks_release = _model_stop_blocks_release(snapshot)

    lead_stop_hold_active = _StopHoldLatchLifecycle.update(self, snapshot, reset_lead_stop_hold)
    post_hold_active = bool(lead_stop_hold_active)
    if post_hold_active:
      self._clear_launch_floor_fade_state(clear_approach=True)
    if not pre_hold_active and post_hold_active:
      # A newly armed hold starts a new measured-motion release episode.
      self.departure_prediction_applied = False
      self.departure_prediction_applied_track_id = -1
    release_mpc_stop = False
    release_a_target = float(mpc_a_target)
    mpc_stop = bool(mpc_should_stop)

    if lead_stop_hold_active:
      latch_release_ok, latch_release_a = _ReleaseGate.release_accepts(self, snapshot)
      if latch_release_ok:
        reset_lead_stop_hold()
        lead_stop_hold_active = False
        post_hold_active = False
        mpc_stop = False
        release_mpc_stop = True
        release_a_target = latch_release_a
        self.stop_hold_release_sustain_s = self._STOP_HOLD_RELEASE_SUSTAIN_S
      else:
        mpc_stop = True
        release_mpc_stop = False
        release_a_target = float(mpc_a_target)
    else:
      self.stop_hold_release_prep_a_target = None
      self.stop_hold_release_prep_raw_prev = None
      release_mpc_stop, release_a_target = _FinalArbitration.standstill_release_clears_mpc_stop(
        self, snapshot
      )
      if not release_mpc_stop:
        release_mpc_stop, release_a_target = _FinalArbitration.stop_hold_release_sustain(self, snapshot)
      mpc_stop = bool(mpc_should_stop and not release_mpc_stop)

    if pre_hold_active and not post_hold_active:
      # Keep the frame-start verdict in the local variable for this release frame, but do not
      # carry predictor readiness into the next stop-hold episode.
      self._clear_departure_prediction_phase(clear_lockout=False)

    custom_should_stop = _FinalArbitration.custom_longitudinal_should_stop(
      custom_long, custom_long_output, mpc_stop, raw_model_should_stop, model_stale
    )
    # A release/sustain frame must not be re-pinned by the stack's still-latched stop verdict
    # (route 000002ac: longcontrol goes stopping on should_stop unconditionally, which made
    # every crawl release one frame long).
    if release_mpc_stop and custom_should_stop:
      custom_should_stop = False
    should_stop = bool(custom_should_stop if custom_should_stop is not None else (mpc_stop or (raw_model_should_stop and is_e2e and not model_stale)))

    if release_mpc_stop:
      self.launch_dip_grace_s = self._LAUNCH_DIP_GRACE_S
    elif math.isfinite(dt) and dt > 0.0:
      self.launch_dip_grace_s = max(0.0, self.launch_dip_grace_s - dt)

    if lead_stop_hold_active:
      a_target, e2e_source = _HoldCommand.compute(self, snapshot)
      self._record_departure_prediction_context(
        snapshot, a_target, True, release_mpc_stop, pre_hold_active, post_hold_active,
        False, pre_hold_lead_id, "stop_hold_active",
      )
      telemetry = _TelemetryAdapter.build_hold_telemetry(self, custom_long_output)
      self.custom_long_output_telemetry = telemetry
      return _TelemetryAdapter.result(a_target, True, e2e_source, telemetry, self.last_release_block_reason)

    if is_e2e and not model_stale:
      self._set_follow_band_regime(None)
      a_target = min(raw_model_a_target, release_a_target if release_mpc_stop else mpc_a_target)
      e2e_source = bool(a_target < mpc_a_target)
      a_target = apply_stop_hold_release_slew(sm, a_target, release_mpc_stop, mpc_stop, model_stop_blocks_release, should_stop)
      self._record_departure_prediction_context(
        snapshot, a_target, should_stop, release_mpc_stop, pre_hold_active, post_hold_active,
        False, pre_hold_lead_id, "mode_not_scc",
      )
      a_target = _FinalArbitration.lead_catchup_cap(self, a_target, snapshot, should_stop)
      a_target = self._apply_approach_damp(a_target, should_stop, release_mpc_stop, dt, snapshot.v_ego)
      if not fade_lifecycle_active:
        self._clear_launch_floor_fade_state(clear_approach=fade_hard_bypass)
      return _TelemetryAdapter.result(
        a_target, should_stop, e2e_source, self.custom_long_output_telemetry, self.last_release_block_reason
      )

    a_target = float(release_a_target if release_mpc_stop else mpc_a_target)
    pre_floor_target = a_target
    if fade_lifecycle_active:
      a_target = _FinalArbitration.scc_launch_floor(self, snapshot, a_target, should_stop)
    if self.launch_dip_grace_s > 0.0:
      self.launch_floor_fade_pending = bool(fade_lifecycle_active and a_target > pre_floor_target)
    a_target = _FinalArbitration.scc_departing_lead_coast(self, snapshot, a_target, should_stop)
    a_target = _FinalArbitration.scc_custom_stop_cap(a_target, custom_long, custom_long_output, release_mpc_stop=release_mpc_stop)
    a_target = _FinalArbitration.scc_cut_in_brake_assist_final_cap(
      self, a_target, sm, custom_long, custom_long_output, release_mpc_stop=release_mpc_stop
    )
    a_target = _FinalArbitration.scc_curve_traffic_advisor_final_cap(
      self, a_target, sm, custom_long, custom_long_output, release_mpc_stop=release_mpc_stop
    )
    # ponytail: SCC path only — the observed dips ride in on mpc_a_target; E2E arbitration
    # min()s against the model and damping that would delay legitimate model braking.
    a_target = self._apply_launch_dip_damp(a_target, snapshot, should_stop, dt)
    pre_slew_state = self.stop_hold_release_slew_a_target
    pre_slew_input = a_target
    a_target = apply_stop_hold_release_slew(sm, a_target, release_mpc_stop, mpc_stop, model_stop_blocks_release, should_stop)
    post_slew_state = self.stop_hold_release_slew_a_target
    if custom_long.mode is LongitudinalMode.SCC:
      a_target = self._apply_departure_prediction(
        snapshot, a_target, should_stop, release_mpc_stop, pre_hold_active, post_hold_active,
        release_slew_provenance=False, pre_hold_lead_id=pre_hold_lead_id,
        frame_start_predicted=frame_start_predicted,
        pre_slew_state=pre_slew_state,
        pre_slew_input=pre_slew_input,
        post_slew_state=post_slew_state,
        post_slew_target=a_target,
      )
    else:
      a_target = self._record_departure_prediction_context(
        snapshot, a_target, should_stop, release_mpc_stop, pre_hold_active, post_hold_active,
        False, pre_hold_lead_id, "mode_not_scc",
      )
    a_target = self._apply_launch_floor_fade(a_target, snapshot, should_stop, release_mpc_stop, dt)
    # ponytail: SCC path only — E2E min()s against deliberate model decel styling; the
    # measured chatter (routes 290/291) rides in on mpc_a_target. Approach damp runs last
    # and jerk-bounds shallow band regime transitions and catch-up cap engage/release
    # edges (lead status flicker, aLeadK steps, the v_ego gate); a DECEL entry straight
    # to a deep brake (|a| > damp band) passes unsmoothed on purpose — brake authority is
    # never delayed.
    a_target = _FinalArbitration.lead_catchup_cap(self, a_target, snapshot, should_stop)
    closure_was_applied = self.low_speed_gap_closure_applied
    a_target = _FinalArbitration.scc_low_speed_gap_closure_floor(
      self, a_target, snapshot, should_stop, release_mpc_stop,
    )
    if closure_was_applied and not self.low_speed_gap_closure_applied:
      # A failed gate must remove this feature's authority immediately; do not let the generic
      # above-crawl approach dam turn a stale positive correction into a release slew.
      self.approach_damp_a_prev = None
    a_target = self._apply_approach_damp(a_target, should_stop, release_mpc_stop, dt, snapshot.v_ego)
    if not fade_lifecycle_active:
      self._clear_launch_floor_fade_state(clear_approach=fade_hard_bypass)
    return _TelemetryAdapter.result(
      a_target, should_stop, False, self.custom_long_output_telemetry, self.last_release_block_reason
    )
