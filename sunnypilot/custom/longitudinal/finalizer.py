"""Post-MPC custom longitudinal final arbitration.

``CustomLongitudinalFinalizer`` owns the stop-hold/release state and the helpers that
arbitrate the final ``(a_target, should_stop, e2e_source)`` tuple after the upstream MPC
solve.  The implementation is split into small single-concern stages for clarity, but the
public API and behavior remain unchanged from the original Phase-5B extraction.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

from openpilot.sunnypilot.custom.longitudinal.modes import LongitudinalMode
from openpilot.sunnypilot.custom.longitudinal.wiring import CustomLongitudinalOutput


@dataclass
class FinalizerResult:
  a_target: float
  should_stop: bool
  e2e_source: bool
  custom_long_output_telemetry: CustomLongitudinalOutput | None = None
  last_release_block_reason: str = ""


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
  gas_pressed: bool
  brake_pressed: bool
  v_ego: float
  standstill: bool
  force_decel: bool
  lead_id: Any
  stopping_distance: float
  v_ego_stopping: float
  stop_accel: float

  @classmethod
  def build(cls, finalizer: CustomLongitudinalFinalizer, sm: Any,
            custom_long: Any, custom_long_output: Any, is_e2e: bool,
            model_stale: bool, dt: float, mpc_a_target: float, mpc_should_stop: bool,
            raw_model_a_target: float, raw_model_should_stop: bool) -> _InputSnapshot:
    car_state = finalizer._sm_item(sm, 'carState')
    controls_state = finalizer._sm_item(sm, 'controlsState')
    radar_state = finalizer._sm_item(sm, 'radarState')
    selected_lead = finalizer._select_stop_hold_lead(radar_state) if radar_state is not None else None
    has_lead = selected_lead is not None
    lead_d_rel = float(getattr(selected_lead, 'dRel', 0.0) or 0.0) if selected_lead is not None else 0.0
    lead_v = float(getattr(selected_lead, 'vLead', 0.0) or 0.0) if selected_lead is not None else 0.0
    lead_v_rel = float(getattr(selected_lead, 'vRel', 0.0) or 0.0) if selected_lead is not None else 0.0
    lead_id = getattr(selected_lead, 'radarTrackId', None) if selected_lead is not None else None
    gas_pressed = bool(getattr(car_state, 'gasPressed', False)) if car_state is not None else False
    brake_pressed = bool(getattr(car_state, 'brakePressed', False)) if car_state is not None else False
    v_ego = float(getattr(car_state, 'vEgo', 0.0) or 0.0) if car_state is not None else 0.0
    standstill = bool(getattr(car_state, 'standstill', False)) if car_state is not None else False
    force_decel = bool(getattr(controls_state, 'forceDecel', False)) if controls_state is not None else False

    stopping_distance = float(getattr(finalizer.CP, 'stoppingDistance', 6.0) or 6.0)
    v_ego_stopping = float(getattr(finalizer.CP, 'vEgoStopping', 0.0))
    stop_accel = getattr(finalizer.CP, 'stopAccel', None)
    stop_accel = -0.5 if stop_accel is None else float(stop_accel)

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
      gas_pressed=gas_pressed,
      brake_pressed=brake_pressed,
      v_ego=v_ego,
      standstill=standstill,
      force_decel=force_decel,
      lead_id=lead_id,
      stopping_distance=stopping_distance,
      v_ego_stopping=v_ego_stopping,
      stop_accel=stop_accel,
    )


# ---------------------------------------------------------------------------
# Stop-hold lead selector
# ---------------------------------------------------------------------------

class _StopHoldLeadSelector:
  """Picks the closest stopped/crawling lead, otherwise the closest lead."""

  @staticmethod
  def select(radar_state: Any) -> Any:
    return CustomLongitudinalFinalizer._select_stop_hold_lead(radar_state)


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

    stop_hold_set = bool(
      not finalizer.lead_stop_hold_active and
      has_lead and
      v_ego < v_ego_stopping + 0.2 and
      lead_d_rel <= arm_distance and
      lead_v <= 0.3 and
      not gas_pressed,
    )
    settle_hold_set = bool(
      not finalizer.lead_stop_hold_active and
      _StopHoldLatchLifecycle.settle_arm_applies(finalizer, snapshot)
    )
    if stop_hold_set or settle_hold_set:
      finalizer.lead_stop_hold_active = True
      finalizer.lead_stop_hold_gap_increasing_s = 0.0
      finalizer.lead_stop_hold_missing_s = 0.0
      finalizer.lead_stop_hold_gap_prev_d_rel = float(lead_d_rel)
      finalizer.lead_stop_hold_gap_baseline_d_rel = min(float(lead_d_rel), finalizer._STOP_HOLD_MAX_BASELINE_D_REL)
      finalizer.lead_stop_hold_lead_id = lead_id

    if finalizer.lead_stop_hold_active:
      if gas_pressed:
        reset_lead_stop_hold()
      elif lead_id is not None and finalizer.lead_stop_hold_lead_id is not None and lead_id != finalizer.lead_stop_hold_lead_id:
        if lead_v <= 0.3 and lead_d_rel <= arm_distance:
          finalizer.lead_stop_hold_lead_id = lead_id
          finalizer.lead_stop_hold_gap_prev_d_rel = float(lead_d_rel)
          finalizer.lead_stop_hold_gap_baseline_d_rel = min(float(lead_d_rel), finalizer._STOP_HOLD_MAX_BASELINE_D_REL)
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
        if finalizer.lead_stop_hold_gap_baseline_d_rel is None:
          finalizer.lead_stop_hold_gap_baseline_d_rel = float(lead_d_rel)
    else:
      finalizer.lead_stop_hold_gap_increasing_s = 0.0
      finalizer.lead_stop_hold_gap_prev_d_rel = float(lead_d_rel) if has_lead else None
      finalizer.lead_stop_hold_gap_baseline_d_rel = min(float(lead_d_rel), finalizer._STOP_HOLD_MAX_BASELINE_D_REL) if has_lead else None
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
                                       min_mpc_a_target: float = -0.03) -> bool:
    custom_long = snapshot.custom_long
    custom_long_output = snapshot.custom_long_output
    mpc_a_target = snapshot.mpc_a_target
    raw_model_a_target = snapshot.raw_model_a_target
    raw_model_should_stop = snapshot.raw_model_should_stop

    if not custom_long.enabled or custom_long_output is None or not bool(getattr(custom_long_output, "standstill_release_allowed", False)):
      finalizer.last_release_block_reason = "no_release_permission"
      return False
    if str(getattr(custom_long_output, "standstill_release_source", "")) not in ("lead_pullaway", "lead_standstill_launch", "no_lead_launch"):
      finalizer.last_release_block_reason = "invalid_release_source"
      return False
    if bool(getattr(custom_long_output, "should_stop", False)):
      finalizer.last_release_block_reason = "custom_should_stop"
      return False
    if raw_model_should_stop:
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
  def standstill_release_planner_gate_valid(finalizer: CustomLongitudinalFinalizer, snapshot: _InputSnapshot,
                                            same_id: bool) -> bool:
    custom_long = snapshot.custom_long
    custom_long_output = snapshot.custom_long_output
    mpc_a_target = snapshot.mpc_a_target
    raw_model_a_target = snapshot.raw_model_a_target
    raw_model_should_stop = snapshot.raw_model_should_stop
    lead_d_rel = snapshot.lead_d_rel
    lead_v = snapshot.lead_v
    lead_v_rel = snapshot.lead_v_rel

    if not _ReleaseGate.standstill_release_gate_enabled(finalizer, custom_long) or not same_id:
      return False
    if custom_long_output is None or not bool(getattr(custom_long_output, "enabled", False)):
      finalizer.last_release_block_reason = "custom_output_unavailable"
      return False
    if bool(getattr(custom_long_output, "should_stop", False)):
      finalizer.last_release_block_reason = "custom_should_stop"
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
    if raw_model_should_stop:
      finalizer.last_release_block_reason = "raw_model_stop"
      return False
    for value in (lead_d_rel, lead_v, lead_v_rel, mpc_a_target, raw_model_a_target):
      if not math.isfinite(float(value)):
        finalizer.last_release_block_reason = "non_finite_values"
        return False
    if float(lead_v) < 0.30 or float(lead_v_rel) < 0.25:
      finalizer.last_release_block_reason = "lead_not_moving"
      return False
    if float(mpc_a_target) < 0.05 or float(raw_model_a_target) < 0.0:
      finalizer.last_release_block_reason = "planner_accel_too_low"
      return False
    if finalizer.lead_stop_hold_gap_baseline_d_rel is None:
      finalizer.last_release_block_reason = "no_baseline_gap"
      return False
    if float(lead_d_rel) - float(finalizer.lead_stop_hold_gap_baseline_d_rel) < 0.5:
      finalizer.last_release_block_reason = "baseline_opening"
      return False
    finalizer.last_release_block_reason = ""
    return True

  @staticmethod
  def crawl_fallback_applies(finalizer: CustomLongitudinalFinalizer, snapshot: _InputSnapshot,
                             same_id: bool) -> bool:
    custom_long = snapshot.custom_long
    custom_long_output = snapshot.custom_long_output
    mpc_a_target = snapshot.mpc_a_target
    raw_model_a_target = snapshot.raw_model_a_target
    raw_model_should_stop = snapshot.raw_model_should_stop
    selected_lead = snapshot.selected_lead
    lead_d_rel = snapshot.lead_d_rel
    lead_v = snapshot.lead_v
    lead_v_rel = snapshot.lead_v_rel

    # Route 261: with the confidence gate on, a stopped lead that crept +1 m left the car
    # latched 9.3 m back with no crawl path at all (gate releases demand a moving lead).
    # The bounded crawl tier (deadband + gap tau + 0.35 cap) stays available under the
    # gate when research actuation is explicitly allowed; otherwise gate mode keeps the
    # old fail-closed behavior.
    if _ReleaseGate.standstill_release_gate_enabled(finalizer, custom_long):
      if not bool(getattr(custom_long_output, "research_actuation_allowed", False)):
        return False
      # Moving leads stay on the (faster) confidence-gate release path; the crawl tier
      # only handles the stationary-creep case the gate cannot (lead_v below its
      # lead-moving threshold).
      if math.isfinite(float(snapshot.lead_v)) and float(snapshot.lead_v) >= 0.30:
        return False
    if not custom_long.enabled or custom_long_output is None:
      return False
    if not bool(getattr(custom_long_output, "enabled", False)):
      return False
    if bool(getattr(custom_long_output, "should_stop", False)):
      return False
    if raw_model_should_stop:
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
    lead_id = getattr(selected_lead, 'radarTrackId', None)
    if lead_id is None or finalizer.lead_stop_hold_lead_id is None or lead_id != finalizer.lead_stop_hold_lead_id:
      return False
    if finalizer.lead_stop_hold_gap_baseline_d_rel is None:
      return False
    for value in (lead_d_rel, lead_v, lead_v_rel, mpc_a_target, raw_model_a_target):
      if not math.isfinite(float(value)):
        return False
    if float(raw_model_a_target) < 0.0:
      return False
    if float(mpc_a_target) < finalizer._STOP_HOLD_SAME_ID_MIN_MPC_A_TARGET:
      return False
    if float(lead_v_rel) < 0.05:
      return False
    baseline_opening = float(lead_d_rel) - float(finalizer.lead_stop_hold_gap_baseline_d_rel)
    if float(lead_v) < 0.20 and baseline_opening < 0.6:
      return False
    return True

  @staticmethod
  def release_accepts(finalizer: CustomLongitudinalFinalizer, snapshot: _InputSnapshot) -> tuple[bool, float]:
    custom_long = snapshot.custom_long
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
    lead_id = getattr(selected_lead, 'radarTrackId', None)
    same_id = lead_id is not None and finalizer.lead_stop_hold_lead_id is not None and lead_id == finalizer.lead_stop_hold_lead_id
    source_valid = release_source in ("lead_pullaway", "lead_standstill_launch")
    if lead_id is not None and finalizer.lead_stop_hold_lead_id is not None and lead_id != finalizer.lead_stop_hold_lead_id:
      finalizer.last_release_block_reason = "different_lead_id"
      return False, float(lead_d_rel)

    crawl_fallback = bool(
      not source_valid and
      _ReleaseGate.crawl_fallback_applies(finalizer, snapshot, same_id)
    )
    gate_fallback_candidate = bool(
      not source_valid and not crawl_fallback and
      _ReleaseGate.standstill_release_gate_enabled(finalizer, custom_long) and same_id and
      bool(getattr(custom_long_output, "research_actuation_allowed", False))
    )
    if not source_valid and not crawl_fallback and not gate_fallback_candidate:
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
      baseline_opening = finalizer._STOP_HOLD_SAME_ID_VALID_BASELINE_OPENING_M if source_valid else finalizer._STOP_HOLD_SAME_ID_MIN_D_REL_BASELINE_OPENING
      baseline_min_d_rel = float(finalizer.lead_stop_hold_gap_baseline_d_rel) + baseline_opening
      min_d_rel = max(finalizer._STOP_HOLD_SAME_ID_MIN_D_REL_FLOOR, min(min_d_rel, baseline_min_d_rel))
    if float(lead_d_rel) <= min_d_rel:
      finalizer.last_release_block_reason = "distance_gate"
      return False, float(lead_d_rel)

    if same_id:
      if source_valid:
        min_gap_increasing_s = finalizer._STOP_HOLD_SAME_ID_VALID_GAP_INCREASING_S
      elif _ReleaseGate.routine_breakout(float(lead_v_rel)):
        min_gap_increasing_s = finalizer._STOP_HOLD_SAME_ID_ROUTINE_PULLAWAY_S
      elif gate_fallback_candidate:
        min_gap_increasing_s = finalizer._STOP_HOLD_SAME_ID_GATE_MIN_PULLAWAY_S
      else:
        min_gap_increasing_s = finalizer._STOP_HOLD_SAME_ID_MIN_PULLAWAY_S
    else:
      min_gap_increasing_s = 0.15
    if finalizer.lead_stop_hold_gap_increasing_s < min_gap_increasing_s:
      finalizer.last_release_block_reason = "gap_increasing_time"
      return False, float(lead_d_rel)

    if same_id and finalizer.lead_stop_hold_gap_baseline_d_rel is not None:
      min_baseline_opening = 0.5 if crawl_fallback else (finalizer._STOP_HOLD_SAME_ID_VALID_BASELINE_OPENING_M if source_valid else 0.3)
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
      return True, release_a

    min_mpc = finalizer._STOP_HOLD_SAME_ID_MIN_MPC_A_TARGET if same_id else -0.03
    if not _ReleaseGate.standstill_release_request_valid(finalizer, snapshot, min_mpc):
      if not gate_fallback_candidate:
        return False, float(lead_d_rel)
      if not _ReleaseGate.standstill_release_planner_gate_valid(finalizer, snapshot, same_id):
        return False, float(lead_d_rel)
      finalizer.last_release_block_reason = ""
      release_a = _ReleaseAccel.accel_for_gap(
        finalizer, float(mpc_a_target), lead_d_rel, lead_v, lead_v_rel, same_id, valid_source=False
      )
      if release_a <= 0.0:
        finalizer.last_release_block_reason = "crawl_deadband"
        return False, float(lead_d_rel)
      return True, release_a

    finalizer.last_release_block_reason = ""
    requested_release_a = float(getattr(custom_long_output, "standstill_release_a_target", 0.0)) if custom_long_output is not None else 0.0
    release_a = _ReleaseAccel.accel_for_gap(
      finalizer, requested_release_a, lead_d_rel, lead_v, lead_v_rel, same_id, valid_source=True
    )
    if release_a <= 0.0:
      finalizer.last_release_block_reason = "crawl_deadband"
      return False, float(lead_d_rel)
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
    raw_model_should_stop = snapshot.raw_model_should_stop

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
    if raw_model_should_stop:
      return False

    car_state = finalizer._sm_item(snapshot.sm, 'carState')
    controls_state = finalizer._sm_item(snapshot.sm, 'controlsState')
    if car_state is None or controls_state is None:
      return False
    if bool(getattr(car_state, "brakePressed", False)) or bool(getattr(car_state, "gasPressed", False)):
      return False
    if bool(getattr(controls_state, "forceDecel", False)):
      return False

    v_ego = float(getattr(car_state, 'vEgo', 0.0))
    v_ego_stopping = float(getattr(finalizer.CP, 'vEgoStopping', 0.5))
    if v_ego >= v_ego_stopping + 0.2:
      return False

    selected_lead = snapshot.selected_lead
    if selected_lead is None:
      return False
    lead_id = getattr(selected_lead, 'radarTrackId', None)
    if lead_id is None or finalizer.lead_stop_hold_lead_id is None or lead_id != finalizer.lead_stop_hold_lead_id:
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

    stopping_distance = snapshot.stopping_distance
    if float(lead_d_rel) <= stopping_distance + finalizer._STOP_HOLD_RELEASE_PREP_MIN_D_REL_MARGIN:
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
    lead_id_sp = getattr(selected_lead, 'radarTrackId', None) if selected_lead is not None else None
    same_id_sp = lead_id_sp is not None and finalizer.lead_stop_hold_lead_id is not None and lead_id_sp == finalizer.lead_stop_hold_lead_id
    if (
      (snapshot.standstill or snapshot.v_ego <= finalizer._STOP_HOLD_STANDSTILL_NORMALIZE_MAX_V_EGO) and
      selected_lead is not None and
      same_id_sp and
      math.isfinite(raw_hold) and
      not snapshot.brake_pressed and
      not snapshot.gas_pressed and
      not snapshot.force_decel and
      not snapshot.raw_model_should_stop
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
  def scc_custom_stop_cap(base_a_target: float, custom_long: Any, custom_long_output: Any) -> float:
    if custom_long.mode is not LongitudinalMode.SCC or not custom_long.enabled or custom_long_output is None:
      return float(base_a_target)
    if not bool(getattr(custom_long_output, "enabled", False)):
      return float(base_a_target)
    if str(getattr(custom_long_output, "selected_intent", "") or "") != "stop_approach":
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
  def scc_curve_confidence_final_cap(finalizer: CustomLongitudinalFinalizer, base_a_target: float,
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
    if str(getattr(custom_long, "curve_speed_confidence_mode", "off") or "off") != "apply_conservative":
      return float(base_a_target)
    debug = dict(getattr(custom_long_output, "debug", {}) or {})
    prefix = "curve_speed_confidence_"
    if not bool(debug.get(prefix + "eligible", False)) or not bool(debug.get(prefix + "apply_supported", False)):
      return float(base_a_target)
    confidence = finalizer._finite_float_or_none(debug.get(prefix + "confidence", 0.0))
    if confidence is None or confidence < finalizer._CURVE_CONFIDENCE_APPLY_MIN_CONFIDENCE:
      return float(base_a_target)
    proposed_cap = finalizer._finite_float_or_none(debug.get(prefix + "proposed_cap", 0.0))
    if proposed_cap is None or proposed_cap >= float(base_a_target):
      return float(base_a_target)
    car_state = finalizer._sm_item(sm, 'carState')
    v_ego = finalizer._safe_float(getattr(car_state, 'vEgo', 0.0) if car_state is not None else 0.0)
    if v_ego < finalizer._CURVE_CONFIDENCE_APPLY_MIN_V_EGO:
      return float(base_a_target)
    conservative_cap = max(proposed_cap, finalizer._CURVE_CONFIDENCE_APPLY_MIN_CAP)
    return float(min(float(base_a_target), conservative_cap))

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
    debug = dict(getattr(custom_long_output, "debug", {}) or {})
    prefix = "cut_in_brake_assist_"
    if not bool(debug.get(prefix + "eligible", False)) or not bool(debug.get(prefix + "apply_supported", False)):
      return float(base_a_target)
    if not bool(debug.get("path_shadow_model_path_available", False)):
      return float(base_a_target)
    confidence = finalizer._finite_float_or_none(debug.get(prefix + "confidence", 0.0))
    if confidence is None or confidence < finalizer._CUT_IN_BRAKE_ASSIST_APPLY_MIN_CONFIDENCE:
      return float(base_a_target)
    path_y_rel = finalizer._finite_float_or_none(debug.get(prefix + "path_y_rel"))
    if path_y_rel is None or abs(path_y_rel) > finalizer._CUT_IN_BRAKE_ASSIST_PATH_NEAR_Y_M:
      return float(base_a_target)
    car_state = finalizer._sm_item(sm, 'carState')
    controls_state = finalizer._sm_item(sm, 'controlsState')
    if bool(getattr(car_state, 'brakePressed', False)) or bool(getattr(car_state, 'gasPressed', False)):
      return float(base_a_target)
    if bool(getattr(controls_state, 'forceDecel', False)):
      return float(base_a_target)
    proposed_cap = finalizer._finite_float_or_none(debug.get(prefix + "proposed_cap", 0.0))
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
    debug = dict(getattr(custom_long_output, "debug", {}) or {})
    prefix = "curve_traffic_"
    if not bool(debug.get(prefix + "eligible", False)) or not bool(debug.get(prefix + "apply_supported", False)):
      return float(base_a_target)
    if str(debug.get(prefix + "traffic_block_reason", "")) != "":
      return float(base_a_target)
    confidence = finalizer._finite_float_or_none(debug.get(prefix + "confidence", 0.0))
    if confidence is None or confidence < finalizer._CURVE_TRAFFIC_APPLY_MIN_CONFIDENCE:
      return float(base_a_target)
    if bool(debug.get("model_stale", False)):
      return float(base_a_target)
    car_state = finalizer._sm_item(sm, 'carState')
    controls_state = finalizer._sm_item(sm, 'controlsState')
    if bool(getattr(car_state, 'brakePressed', False)) or bool(getattr(car_state, 'gasPressed', False)):
      return float(base_a_target)
    if bool(getattr(controls_state, 'forceDecel', False)):
      return float(base_a_target)
    proposed_cap = finalizer._finite_float_or_none(debug.get(prefix + "a_curve_cap_proposed", 0.0))
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
      bool(raw_model_should_stop) or
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
  _STOP_HOLD_SAME_ID_GATE_MIN_PULLAWAY_S = 0.15
  # Breakout is only consulted while the stop-hold latch is active (ego <= ~0.7 m/s), where
  # v_rel ~= lead_v, so v_rel is the whole gate (a lead_v arm was subsumed and deleted).
  # Crawl-launch feel is tuned via _STOP_HOLD_CRAWL_GAP_TAU below, not here.
  _STOP_HOLD_ROUTINE_BREAKOUT_MIN_V_REL = 1.0
  _STOP_HOLD_CRAWL_DEADBAND_M = 0.50
  # Route 0000025a: below the breakout the release accel ramps as (gap_opened - deadband) / TAU; the old
  # 2.0 s TAU held the launch at ~0.05 m/s^2 well after the lead was clearly moving. 1.5 ramps 33% faster
  # per metre of gap while the deadband + A_MAX cap still bound it. ponytail: knob, retune from logs.
  _STOP_HOLD_CRAWL_GAP_TAU = 1.5
  _STOP_HOLD_CRAWL_RELEASE_A_MIN = 0.05
  # Raised with _STOP_HOLD_RELEASE_A_MAX (route 00000246): the close-crawl cap still stays
  # below the breakout cap, and the gap governor above continues to bound crawl accel.
  _STOP_HOLD_CRAWL_RELEASE_A_MAX = 0.35
  _STOP_HOLD_SETTLE_ARM_V_EGO_FLOOR = 0.7
  _STOP_HOLD_SETTLE_ARM_MAX_LEAD_V = 0.5
  _STOP_HOLD_SETTLE_ARM_MAX_LEAD_V_REL = 0.1
  _STOP_HOLD_SETTLE_ARM_DISTANCE_MARGIN = 0.5
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
  _CURVE_CONFIDENCE_APPLY_MIN_V_EGO = 8.0
  _CURVE_CONFIDENCE_APPLY_MIN_CONFIDENCE = 0.70
  _CURVE_CONFIDENCE_APPLY_MIN_CAP = -0.85

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
  lead_stop_hold_gap_baseline_d_rel: float | None
  custom_long_output_telemetry: CustomLongitudinalOutput | None
  last_release_block_reason: str
  stop_hold_release_slew_a_target: float | None
  stop_hold_release_prep_a_target: float | None
  stop_hold_release_prep_raw_prev: float | None
  approach_damp_a_prev: float | None
  final_a_prev: float | None

  def __init__(self, CP: Any):
    self.CP = CP

    self.lead_stop_hold_active = False
    self.lead_stop_hold_gap_increasing_s = 0.0
    self.lead_stop_hold_missing_s = 0.0
    self.lead_stop_hold_lead_id = None
    self.lead_stop_hold_gap_prev_d_rel = None
    self.lead_stop_hold_gap_baseline_d_rel = None
    self.custom_long_output_telemetry = None
    self.last_release_block_reason = ""
    self.stop_hold_release_slew_a_target = None
    self.stop_hold_release_prep_a_target = None
    self.stop_hold_release_prep_raw_prev = None
    self.approach_damp_a_prev = None
    # Last commanded a_target across ALL finalize paths, including hold frames (which
    # never route through the release slew). Deliberately NOT cleared in
    # reset_lead_stop_hold: that runs on the release frame itself, and the release-slew
    # seed needs the prior hold command to bound the release step.
    self.final_a_prev = None

  @staticmethod
  def _sm_item(sm: Any, key: str) -> Any:
    if hasattr(sm, 'get'):
      return sm.get(key)
    try:
      return sm[key]
    except Exception:
      return None

  @staticmethod
  def _select_stop_hold_lead(radar_state: Any) -> Any:
    candidates = []
    for lead in (getattr(radar_state, 'leadOne', None), getattr(radar_state, 'leadTwo', None)):
      if lead is None or not getattr(lead, 'status', False):
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
    stopped = [c for c in candidates if c[1] <= 0.5]
    if stopped:
      return min(stopped, key=lambda c: c[0])[3]
    return min(candidates, key=lambda c: c[0])[3]

  @staticmethod
  def _routine_lead_launch_breakout(lead_v_rel: float) -> bool:
    return float(lead_v_rel) >= CustomLongitudinalFinalizer._STOP_HOLD_ROUTINE_BREAKOUT_MIN_V_REL

  def reset_lead_stop_hold(self) -> None:
    self.lead_stop_hold_active = False
    self.lead_stop_hold_gap_increasing_s = 0.0
    self.lead_stop_hold_missing_s = 0.0
    self.lead_stop_hold_lead_id = None
    self.lead_stop_hold_gap_prev_d_rel = None
    self.lead_stop_hold_gap_baseline_d_rel = None
    self.stop_hold_release_slew_a_target = None
    self.stop_hold_release_prep_a_target = None
    self.stop_hold_release_prep_raw_prev = None
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
    snapshot.lead_id = getattr(selected_lead, 'radarTrackId', None) if selected_lead is not None else None
    return _StopHoldLatchLifecycle.update(self, snapshot, reset_lead_stop_hold)

  def custom_longitudinal_should_stop(self, custom_long: Any, custom_long_output: Any,
                                      mpc_should_stop: bool, raw_model_should_stop: bool,
                                      model_stale: bool = False) -> bool | None:
    return _FinalArbitration.custom_longitudinal_should_stop(
      custom_long, custom_long_output, mpc_should_stop, raw_model_should_stop, model_stale
    )

  def _scc_custom_stop_cap(self, base_a_target: float, custom_long: Any, custom_long_output: Any) -> float:
    return _FinalArbitration.scc_custom_stop_cap(base_a_target, custom_long, custom_long_output)

  def _scc_curve_confidence_final_cap(self, base_a_target: float, sm: Any, custom_long: Any, custom_long_output: Any,
                                      release_mpc_stop: bool = False) -> float:
    return _FinalArbitration.scc_curve_confidence_final_cap(
      self, base_a_target, sm, custom_long, custom_long_output, release_mpc_stop
    )

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
    snapshot.lead_id = getattr(selected_lead, 'radarTrackId', None) if selected_lead is not None else None
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
    snapshot.lead_id = getattr(selected_lead, 'radarTrackId', None) if selected_lead is not None else None
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

  def _apply_approach_damp(self, a_target: float, should_stop: bool, release_mpc_stop: bool, dt: float) -> float:
    """Jerk-limit aTarget inside the gentle authority band to kill the ACC-MPC approach-cusp limit cycle.

    Only active when the command is small (|a| <= band), not stopping, not releasing a stop hold, and no
    stop-hold release ramp is in progress, so a developing strong brake or accel (or any stop/launch)
    leaves the band and passes straight through with no added lag. In particular the launch ramp off a
    stop hold must never be damped. Outside those cases the filter state is dropped so it re-seeds cleanly.
    """
    if (not math.isfinite(a_target) or not math.isfinite(dt) or dt <= 0.0
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

  def _standstill_release_planner_gate_valid(self, sm: Any, custom_long: Any, custom_long_output: Any,
                                             mpc_a_target: float, raw_model_a_target: float, raw_model_should_stop: bool,
                                             selected_lead: Any, lead_d_rel: float, lead_v: float,
                                             lead_v_rel: float, same_id: bool) -> bool:
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
    snapshot.lead_id = getattr(selected_lead, 'radarTrackId', None) if selected_lead is not None else None
    return _ReleaseGate.standstill_release_planner_gate_valid(self, snapshot, same_id)

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
    snapshot.lead_id = getattr(selected_lead, 'radarTrackId', None) if selected_lead is not None else None
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
    snapshot.lead_id = getattr(selected_lead, 'radarTrackId', None) if selected_lead is not None else None
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

  def finalize(self, *args: Any, **kwargs: Any) -> FinalizerResult:
    """Run ``_finalize_impl`` and record the commanded accel for the next frame.

    ``final_a_prev`` must see every commanded a_target — hold frames included —
    so the release-slew seed can bound the hold->release step.
    """
    result = self._finalize_impl(*args, **kwargs)
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
    if not bool(getattr(custom_long, "enabled", False)):
      reset_lead_stop_hold()
      self.custom_long_output_telemetry = None
      self.last_release_block_reason = ""
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
    )

    lead_stop_hold_active = _StopHoldLatchLifecycle.update(self, snapshot, reset_lead_stop_hold)
    release_mpc_stop = False
    release_a_target = float(mpc_a_target)
    mpc_stop = bool(mpc_should_stop)

    if lead_stop_hold_active:
      latch_release_ok, latch_release_a = _ReleaseGate.release_accepts(self, snapshot)
      if latch_release_ok:
        reset_lead_stop_hold()
        lead_stop_hold_active = False
        mpc_stop = False
        release_mpc_stop = True
        release_a_target = latch_release_a
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
      mpc_stop = bool(mpc_should_stop and not release_mpc_stop)

    custom_should_stop = _FinalArbitration.custom_longitudinal_should_stop(
      custom_long, custom_long_output, mpc_stop, raw_model_should_stop, model_stale
    )
    should_stop = bool(custom_should_stop if custom_should_stop is not None else (mpc_stop or (raw_model_should_stop and is_e2e and not model_stale)))

    if lead_stop_hold_active:
      a_target, e2e_source = _HoldCommand.compute(self, snapshot)
      telemetry = _TelemetryAdapter.build_hold_telemetry(self, custom_long_output)
      self.custom_long_output_telemetry = telemetry
      return _TelemetryAdapter.result(a_target, True, e2e_source, telemetry, self.last_release_block_reason)

    if is_e2e and not model_stale:
      a_target = min(raw_model_a_target, release_a_target if release_mpc_stop else mpc_a_target)
      e2e_source = bool(a_target < mpc_a_target)
      a_target = apply_stop_hold_release_slew(sm, a_target, release_mpc_stop, mpc_stop, raw_model_should_stop, should_stop)
      a_target = self._apply_approach_damp(a_target, should_stop, release_mpc_stop, dt)
      return _TelemetryAdapter.result(
        a_target, should_stop, e2e_source, self.custom_long_output_telemetry, self.last_release_block_reason
      )

    a_target = float(release_a_target if release_mpc_stop else mpc_a_target)
    a_target = _FinalArbitration.scc_custom_stop_cap(a_target, custom_long, custom_long_output)
    a_target = _FinalArbitration.scc_curve_confidence_final_cap(
      self, a_target, sm, custom_long, custom_long_output, release_mpc_stop=release_mpc_stop
    )
    a_target = _FinalArbitration.scc_cut_in_brake_assist_final_cap(
      self, a_target, sm, custom_long, custom_long_output, release_mpc_stop=release_mpc_stop
    )
    a_target = _FinalArbitration.scc_curve_traffic_advisor_final_cap(
      self, a_target, sm, custom_long, custom_long_output, release_mpc_stop=release_mpc_stop
    )
    a_target = apply_stop_hold_release_slew(sm, a_target, release_mpc_stop, mpc_stop, raw_model_should_stop, should_stop)
    a_target = self._apply_approach_damp(a_target, should_stop, release_mpc_stop, dt)
    return _TelemetryAdapter.result(
      a_target, should_stop, False, self.custom_long_output_telemetry, self.last_release_block_reason
    )
