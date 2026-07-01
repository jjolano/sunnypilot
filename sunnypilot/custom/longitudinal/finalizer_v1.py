"""Post-MPC custom longitudinal final arbitration.

``CustomLongitudinalFinalizer`` owns the stop-hold/release state and the helpers that
arbitrate the final ``(a_target, should_stop, e2e_source)`` tuple after the upstream MPC
solve.  It is intentionally boring: constants, state, and helper logic were behavior-
preserving extracted from ``LongitudinalPlannerSP`` in Phase 5B.
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


class CustomLongitudinalFinalizer:
  _STOP_HOLD_MAX_BASELINE_D_REL = 6.0
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
  _STOP_HOLD_ROUTINE_BREAKOUT_MIN_LEAD_V = 5.0
  _STOP_HOLD_ROUTINE_BREAKOUT_MIN_V_REL = 1.0
  _STOP_HOLD_CRAWL_DEADBAND_M = 0.50
  _STOP_HOLD_CRAWL_GAP_TAU = 2.0
  _STOP_HOLD_CRAWL_RELEASE_A_MIN = 0.05
  _STOP_HOLD_CRAWL_RELEASE_A_MAX = 0.25
  # Settle-hold arm: catch nearly-stopped leads at very low speed before the
  # legacy vEgoStopping-based latch would fire. Prevents small crawl/brake
  # oscillations when stopping behind a stationary lead on vehicles with a low
  # vEgoStopping value.
  _STOP_HOLD_SETTLE_ARM_V_EGO_FLOOR = 0.7
  _STOP_HOLD_SETTLE_ARM_MAX_LEAD_V = 0.5
  _STOP_HOLD_SETTLE_ARM_MAX_LEAD_V_REL = 0.1
  _STOP_HOLD_SETTLE_ARM_DISTANCE_MARGIN = 0.5
  _STOP_HOLD_SETTLE_ARM_BRAKE_DIST_DECEL = 2.0
  _STOP_HOLD_SETTLE_ARM_BRAKE_DIST_MAX = 1.0
  _STOP_HOLD_RELEASE_A_MIN = 0.15
  _STOP_HOLD_RELEASE_A_MAX = 0.35
  _STOP_HOLD_RELEASE_MAX_UP_JERK = 6.0
  # Pre-release only: relax harsh stopped-lead hold when the same latched lead is pulling away.
  _STOP_HOLD_RELEASE_PREP_A_TARGET = -0.20
  _STOP_HOLD_RELEASE_PREP_MAX_UP_JERK = 6.0
  _STOP_HOLD_RELEASE_PREP_MIN_LEAD_V = 0.25
  _STOP_HOLD_RELEASE_PREP_MIN_LEAD_V_REL = 0.10
  _STOP_HOLD_RELEASE_PREP_MIN_MPC_A_TARGET = -0.10
  _STOP_HOLD_RELEASE_PREP_MIN_GAP_INCREASING_S = 0.10
  _STOP_HOLD_RELEASE_PREP_MIN_D_REL_MARGIN = 0.10
  _STOP_HOLD_STANDSTILL_NORMALIZED_A_TARGET = -0.50
  _STOP_HOLD_STANDSTILL_NORMALIZE_MAX_V_EGO = 0.70
  _CURVE_CONFIDENCE_APPLY_MIN_V_EGO = 8.0
  _CURVE_CONFIDENCE_APPLY_MIN_CONFIDENCE = 0.70
  _CURVE_CONFIDENCE_APPLY_MIN_CAP = -0.85

  _CUT_IN_BRAKE_ASSIST_APPLY_MIN_CONFIDENCE = 0.60
  _CUT_IN_BRAKE_ASSIST_APPLY_MAX_DECEL = -0.60
  _CUT_IN_BRAKE_ASSIST_PATH_NEAR_Y_M = 1.70

  _CURVE_TRAFFIC_APPLY_MIN_CONFIDENCE = 0.45
  _CURVE_TRAFFIC_APPLY_MIN_CAP = -0.85

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
    # Prefer closest stopped/crawling lead; otherwise closest lead
    stopped = [c for c in candidates if c[1] <= 0.5]
    if stopped:
      return min(stopped, key=lambda c: c[0])[3]
    return min(candidates, key=lambda c: c[0])[3]

  def _settle_stop_hold_arm_applies(self, v_ego: float, v_ego_stopping: float,
                                    lead_v: float, lead_v_rel: float,
                                    lead_d_rel: float, gas_pressed: bool,
                                    has_lead: bool) -> bool:
    if not has_lead or gas_pressed:
      return False
    if not all(math.isfinite(v) for v in (v_ego, v_ego_stopping, lead_v, lead_v_rel, lead_d_rel)):
      return False
    if v_ego > max(v_ego_stopping + 0.2, self._STOP_HOLD_SETTLE_ARM_V_EGO_FLOOR):
      return False
    if lead_v > self._STOP_HOLD_SETTLE_ARM_MAX_LEAD_V:
      return False
    if lead_v_rel > self._STOP_HOLD_SETTLE_ARM_MAX_LEAD_V_REL:
      return False
    stopping_distance = float(getattr(self.CP, 'stoppingDistance', 6.0) or 6.0)
    braking_dist = min(
      v_ego ** 2 / (2.0 * self._STOP_HOLD_SETTLE_ARM_BRAKE_DIST_DECEL),
      self._STOP_HOLD_SETTLE_ARM_BRAKE_DIST_MAX,
    )
    settle_distance = stopping_distance + self._STOP_HOLD_SETTLE_ARM_DISTANCE_MARGIN + braking_dist
    return lead_d_rel <= settle_distance

  @classmethod
  def _routine_lead_launch_breakout(cls, lead_v: float, lead_v_rel: float) -> bool:
    return bool(
      float(lead_v) >= cls._STOP_HOLD_ROUTINE_BREAKOUT_MIN_LEAD_V or
      float(lead_v_rel) >= cls._STOP_HOLD_ROUTINE_BREAKOUT_MIN_V_REL
    )

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

  def _update_lead_stop_hold(self, sm: Any, v_ego: float, has_lead: bool, selected_lead: Any,
                             lead_d_rel: float, lead_v: float, lead_v_rel: float,
                             gas_pressed: bool, dt: float, reset_lead_stop_hold: Any) -> bool:
    lead_id = getattr(selected_lead, 'radarTrackId', None) if selected_lead is not None else None

    stopping_distance = float(getattr(self.CP, 'stoppingDistance', 6.0) or 6.0)
    arm_distance = max(stopping_distance + 2.0, 10.0)
    v_ego_stopping = float(getattr(self.CP, 'vEgoStopping', 0.0))

    # Arm check (only when not already latched).
    stop_hold_set = bool(
      not self.lead_stop_hold_active and
      has_lead and
      v_ego < v_ego_stopping + 0.2 and
      lead_d_rel <= arm_distance and
      lead_v <= 0.3 and
      not gas_pressed,
    )
    settle_hold_set = bool(
      not self.lead_stop_hold_active and
      self._settle_stop_hold_arm_applies(
        v_ego, v_ego_stopping, lead_v, lead_v_rel, lead_d_rel, gas_pressed, has_lead,
      )
    )
    if stop_hold_set or settle_hold_set:
      self.lead_stop_hold_active = True
      self.lead_stop_hold_gap_increasing_s = 0.0
      self.lead_stop_hold_missing_s = 0.0
      self.lead_stop_hold_gap_prev_d_rel = float(lead_d_rel)
      self.lead_stop_hold_gap_baseline_d_rel = min(float(lead_d_rel), self._STOP_HOLD_MAX_BASELINE_D_REL)
      self.lead_stop_hold_lead_id = lead_id

    if self.lead_stop_hold_active:
      if gas_pressed:
        reset_lead_stop_hold()
      elif lead_id is not None and self.lead_stop_hold_lead_id is not None and lead_id != self.lead_stop_hold_lead_id:
        # A different lead appeared while latched.
        if lead_v <= 0.3 and lead_d_rel <= arm_distance:
          # Valid stopped hold candidate: transfer latch immediately (no one-cycle gap).
          self.lead_stop_hold_lead_id = lead_id
          self.lead_stop_hold_gap_prev_d_rel = float(lead_d_rel)
          self.lead_stop_hold_gap_baseline_d_rel = min(float(lead_d_rel), self._STOP_HOLD_MAX_BASELINE_D_REL)
          self.lead_stop_hold_missing_s = 0.0
          self.lead_stop_hold_gap_increasing_s = 0.0
        else:
          # Non-stopped transient: treat as dropout within the 0.5 s grace window.
          self.lead_stop_hold_missing_s += dt
          if not (self.lead_stop_hold_missing_s < 0.5 and v_ego < v_ego_stopping + 0.2 and not gas_pressed):
            reset_lead_stop_hold()
      elif not has_lead:
        self.lead_stop_hold_missing_s += dt
        if not (self.lead_stop_hold_missing_s < 0.5 and v_ego < v_ego_stopping + 0.2 and not gas_pressed):
          reset_lead_stop_hold()
      else:
        self.lead_stop_hold_missing_s = 0.0
        gap_increasing = self.lead_stop_hold_gap_prev_d_rel is not None and float(lead_d_rel) > float(self.lead_stop_hold_gap_prev_d_rel)
        if gap_increasing:
          self.lead_stop_hold_gap_increasing_s += dt
        else:
          self.lead_stop_hold_gap_increasing_s = 0.0
        self.lead_stop_hold_gap_prev_d_rel = float(lead_d_rel)
        if self.lead_stop_hold_gap_baseline_d_rel is None:
          self.lead_stop_hold_gap_baseline_d_rel = float(lead_d_rel)
    else:
      self.lead_stop_hold_gap_increasing_s = 0.0
      self.lead_stop_hold_gap_prev_d_rel = float(lead_d_rel) if has_lead else None
      self.lead_stop_hold_gap_baseline_d_rel = min(float(lead_d_rel), self._STOP_HOLD_MAX_BASELINE_D_REL) if has_lead else None
      self.lead_stop_hold_missing_s = 0.0

    return self.lead_stop_hold_active

  def custom_longitudinal_should_stop(self, custom_long: Any, custom_long_output: Any,
                                      mpc_should_stop: bool, raw_model_should_stop: bool,
                                      model_stale: bool = False) -> bool | None:
    if not custom_long.enabled or custom_long_output is None:
      return None
    if custom_long.mode is LongitudinalMode.ACC:
      return bool(mpc_should_stop)
    if custom_long.mode is LongitudinalMode.E2E:
      return bool(mpc_should_stop or (raw_model_should_stop and not model_stale))
    return bool(mpc_should_stop or custom_long_output.should_stop)

  def _scc_custom_stop_cap(self, base_a_target: float, custom_long: Any, custom_long_output: Any) -> float:
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

  def _scc_curve_confidence_final_cap(self, base_a_target: float, sm: Any, custom_long: Any, custom_long_output: Any,
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
    confidence = self._finite_float_or_none(debug.get(prefix + "confidence", 0.0))
    if confidence is None or confidence < self._CURVE_CONFIDENCE_APPLY_MIN_CONFIDENCE:
      return float(base_a_target)
    proposed_cap = self._finite_float_or_none(debug.get(prefix + "proposed_cap", 0.0))
    if proposed_cap is None or proposed_cap >= float(base_a_target):
      return float(base_a_target)
    car_state = self._sm_item(sm, 'carState')
    v_ego = self._safe_float(getattr(car_state, 'vEgo', 0.0) if car_state is not None else 0.0)
    if v_ego < self._CURVE_CONFIDENCE_APPLY_MIN_V_EGO:
      return float(base_a_target)
    conservative_cap = max(proposed_cap, self._CURVE_CONFIDENCE_APPLY_MIN_CAP)
    return float(min(float(base_a_target), conservative_cap))

  def _scc_cut_in_brake_assist_final_cap(self, base_a_target: float, sm: Any, custom_long: Any, custom_long_output: Any,
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
    confidence = self._finite_float_or_none(debug.get(prefix + "confidence", 0.0))
    if confidence is None or confidence < self._CUT_IN_BRAKE_ASSIST_APPLY_MIN_CONFIDENCE:
      return float(base_a_target)
    path_y_rel = self._finite_float_or_none(debug.get(prefix + "path_y_rel"))
    if path_y_rel is None or abs(path_y_rel) > self._CUT_IN_BRAKE_ASSIST_PATH_NEAR_Y_M:
      return float(base_a_target)
    car_state = self._sm_item(sm, 'carState')
    controls_state = self._sm_item(sm, 'controlsState')
    if bool(getattr(car_state, 'brakePressed', False)) or bool(getattr(car_state, 'gasPressed', False)):
      return float(base_a_target)
    if bool(getattr(controls_state, 'forceDecel', False)):
      return float(base_a_target)
    proposed_cap = self._finite_float_or_none(debug.get(prefix + "proposed_cap", 0.0))
    if proposed_cap is None or proposed_cap >= 0.0:
      return float(base_a_target)
    gentle_cap = max(proposed_cap, self._CUT_IN_BRAKE_ASSIST_APPLY_MAX_DECEL)
    if gentle_cap >= float(base_a_target):
      return float(base_a_target)
    return float(min(float(base_a_target), gentle_cap))

  def _scc_curve_traffic_advisor_final_cap(self, base_a_target: float, sm: Any, custom_long: Any, custom_long_output: Any,
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
    confidence = self._finite_float_or_none(debug.get(prefix + "confidence", 0.0))
    if confidence is None or confidence < self._CURVE_TRAFFIC_APPLY_MIN_CONFIDENCE:
      return float(base_a_target)
    if bool(debug.get("model_stale", False)):
      return float(base_a_target)
    car_state = self._sm_item(sm, 'carState')
    controls_state = self._sm_item(sm, 'controlsState')
    if bool(getattr(car_state, 'brakePressed', False)) or bool(getattr(car_state, 'gasPressed', False)):
      return float(base_a_target)
    if bool(getattr(controls_state, 'forceDecel', False)):
      return float(base_a_target)
    proposed_cap = self._finite_float_or_none(debug.get(prefix + "a_curve_cap_proposed", 0.0))
    if proposed_cap is None or proposed_cap >= 0.0:
      return float(base_a_target)
    conservative_cap = max(proposed_cap, self._CURVE_TRAFFIC_APPLY_MIN_CAP)
    if conservative_cap >= float(base_a_target):
      return float(base_a_target)
    return float(min(float(base_a_target), conservative_cap))

  def _apply_stop_hold_release_slew(self, sm: Any, dt: float, a_target: float, release_mpc_stop: bool,
                                    mpc_stop: bool, raw_model_should_stop: bool, should_stop: bool) -> float:
    """Narrow upward-only slew limiter for stop-hold release pullaways.

    Seeds on the first positive release tick with the actual final output returned, then
    caps only upward increases to reduce launch jerk. Downward/braking changes pass through
    unmodified so hazard responses are not delayed. Cleared on any stop/override/hazard.
    """
    car_state = self._sm_item(sm, 'carState')
    controls_state = self._sm_item(sm, 'controlsState')
    brake_pressed = bool(getattr(car_state, 'brakePressed', False)) if car_state is not None else False
    gas_pressed = bool(getattr(car_state, 'gasPressed', False)) if car_state is not None else False
    force_decel = bool(getattr(controls_state, 'forceDecel', False)) if controls_state is not None else False

    clear = (
      not math.isfinite(a_target) or
      not math.isfinite(dt) or dt <= 0.0 or
      (self.stop_hold_release_slew_a_target is not None and not math.isfinite(self.stop_hold_release_slew_a_target)) or
      bool(should_stop) or
      bool(mpc_stop) or
      bool(raw_model_should_stop) or
      brake_pressed or gas_pressed or force_decel or
      a_target <= 0.0
    )
    if clear:
      self.stop_hold_release_slew_a_target = None
      return float(a_target)

    # First positive release tick seeds the slew state with the actual final output returned.
    if release_mpc_stop and self.stop_hold_release_slew_a_target is None:
      self.stop_hold_release_slew_a_target = float(a_target)
      return float(a_target)

    # Active slew: cap only upward increases; downward/braking changes pass through.
    if self.stop_hold_release_slew_a_target is not None:
      max_step = self._STOP_HOLD_RELEASE_MAX_UP_JERK * dt
      last_slew = self.stop_hold_release_slew_a_target
      if a_target > last_slew + max_step:
        a_target = last_slew + max_step
        self.stop_hold_release_slew_a_target = float(a_target)
      elif a_target > last_slew:
        self.stop_hold_release_slew_a_target = None
      else:
        self.stop_hold_release_slew_a_target = float(a_target)

    return float(a_target)

  def _stop_hold_release_prep_applies(self, sm: Any, selected_lead: Any, custom_long: Any, custom_long_output: Any,
                                      lead_d_rel: float, lead_v: float, lead_v_rel: float,
                                      mpc_a_target: float, raw_model_a_target: float,
                                      raw_model_should_stop: bool) -> bool:
    """Return True when early release evidence justifies relaxing the stop-hold accel."""
    if not custom_long.enabled or custom_long_output is None:
      return False
    if not bool(getattr(custom_long_output, "standstill_release_allowed", False)):
      return False
    if str(getattr(custom_long_output, "standstill_release_source", "")) not in ("lead_pullaway", "lead_standstill_launch"):
      return False
    if bool(getattr(custom_long_output, "should_stop", False)):
      return False
    if raw_model_should_stop:
      return False

    car_state = self._sm_item(sm, 'carState')
    controls_state = self._sm_item(sm, 'controlsState')
    if car_state is None or controls_state is None:
      return False
    if bool(getattr(car_state, "brakePressed", False)) or bool(getattr(car_state, "gasPressed", False)):
      return False
    if bool(getattr(controls_state, "forceDecel", False)):
      return False

    v_ego = float(getattr(car_state, 'vEgo', 0.0))
    v_ego_stopping = float(getattr(self.CP, 'vEgoStopping', 0.5))
    if v_ego >= v_ego_stopping + 0.2:
      return False

    if selected_lead is None:
      return False
    lead_id = getattr(selected_lead, 'radarTrackId', None)
    if lead_id is None or self.lead_stop_hold_lead_id is None or lead_id != self.lead_stop_hold_lead_id:
      return False
    for value in (lead_d_rel, lead_v, lead_v_rel, mpc_a_target, raw_model_a_target):
      if not math.isfinite(float(value)):
        return False

    if float(mpc_a_target) < self._STOP_HOLD_RELEASE_PREP_MIN_MPC_A_TARGET:
      return False
    if float(lead_v) < self._STOP_HOLD_RELEASE_PREP_MIN_LEAD_V or float(lead_v_rel) < self._STOP_HOLD_RELEASE_PREP_MIN_LEAD_V_REL:
      return False
    if self.lead_stop_hold_gap_increasing_s < self._STOP_HOLD_RELEASE_PREP_MIN_GAP_INCREASING_S:
      return False

    stopping_distance = float(getattr(self.CP, 'stoppingDistance', 6.0) or 6.0)
    if float(lead_d_rel) <= stopping_distance + self._STOP_HOLD_RELEASE_PREP_MIN_D_REL_MARGIN:
      return False

    return True

  def _apply_stop_hold_release_prep(self, sm: Any, dt: float, raw_hold: float, selected_lead: Any,
                                    custom_long: Any, custom_long_output: Any,
                                    lead_d_rel: float, lead_v: float, lead_v_rel: float,
                                    mpc_a_target: float, raw_model_a_target: float,
                                    raw_model_should_stop: bool) -> float:
    """Relax a harsh stop-hold accel toward a mild negative target before the first positive release.

    Applied only while the lead stop-hold latch is active. Upward changes are limited by a
    jerk cap; downward/braking changes pass through immediately so hazard responses are not
    delayed.
    """
    state = self.stop_hold_release_prep_a_target
    raw_prev = self.stop_hold_release_prep_raw_prev

    clear = (
      not math.isfinite(raw_hold) or
      not math.isfinite(dt) or dt <= 0.0 or
      (state is not None and not math.isfinite(state)) or
      (raw_prev is not None and not math.isfinite(raw_prev)) or
      not self._stop_hold_release_prep_applies(
        sm, selected_lead, custom_long, custom_long_output, lead_d_rel, lead_v, lead_v_rel,
        mpc_a_target, raw_model_a_target, raw_model_should_stop,
      )
    )
    if clear:
      self.stop_hold_release_prep_a_target = None
      self.stop_hold_release_prep_raw_prev = None
      return float(raw_hold)

    prev_output = float(state) if state is not None else float(raw_hold)
    prev_raw = float(raw_prev) if raw_prev is not None else float(raw_hold)

    # Downward / braking change: raw hold became more negative than last cycle.
    if raw_hold < prev_raw:
      self.stop_hold_release_prep_a_target = float(raw_hold)
      self.stop_hold_release_prep_raw_prev = float(raw_hold)
      return float(raw_hold)

    desired = max(float(raw_hold), self._STOP_HOLD_RELEASE_PREP_A_TARGET)
    max_step = self._STOP_HOLD_RELEASE_PREP_MAX_UP_JERK * dt
    if desired > prev_output + max_step:
      limited = prev_output + max_step
    else:
      limited = desired
    limited = max(limited, prev_output)  # never drift downward
    self.stop_hold_release_prep_a_target = float(limited)
    self.stop_hold_release_prep_raw_prev = float(raw_hold)
    return float(limited)

  def _standstill_release_gate_enabled(self, custom_long: Any) -> bool:
    return bool(
      custom_long.enabled and
      custom_long.mode is LongitudinalMode.SCC and
      str(getattr(custom_long, "standstill_release_confidence_mode", "off") or "off") == "gate"
    )

  def _standstill_release_clears_mpc_stop(self, sm: Any, custom_long: Any, custom_long_output: Any,
                                          mpc_a_target: float, mpc_should_stop: bool,
                                          raw_model_a_target: float, raw_model_should_stop: bool) -> tuple[bool, float]:
    if not self._standstill_release_request_valid(sm, custom_long, custom_long_output, mpc_a_target, raw_model_a_target, raw_model_should_stop):
      return False, float(mpc_a_target)
    if not mpc_should_stop:
      return False, float(mpc_a_target)
    release_a = min(
      max(float(mpc_a_target), self._STOP_HOLD_RELEASE_A_MIN, float(getattr(custom_long_output, "standstill_release_a_target", 0.0))),
      self._STOP_HOLD_RELEASE_A_MAX,
    )
    return True, release_a

  def _standstill_release_request_valid(self, sm: Any, custom_long: Any, custom_long_output: Any,
                                        mpc_a_target: float, raw_model_a_target: float, raw_model_should_stop: bool,
                                        min_mpc_a_target: float = -0.03) -> bool:
    if not custom_long.enabled or custom_long_output is None or not bool(getattr(custom_long_output, "standstill_release_allowed", False)):
      self.last_release_block_reason = "no_release_permission"
      return False
    if str(getattr(custom_long_output, "standstill_release_source", "")) not in ("lead_pullaway", "lead_standstill_launch", "no_lead_launch"):
      self.last_release_block_reason = "invalid_release_source"
      return False
    if bool(getattr(custom_long_output, "should_stop", False)):
      self.last_release_block_reason = "custom_should_stop"
      return False
    if raw_model_should_stop:
      self.last_release_block_reason = "raw_model_stop"
      return False
    for value in (mpc_a_target, raw_model_a_target):
      if not math.isfinite(float(value)):
        self.last_release_block_reason = "non_finite_target"
        return False
    if custom_long.mode is LongitudinalMode.E2E and float(raw_model_a_target) < 0.15:
      self.last_release_block_reason = "e2e_model_accel_too_low"
      return False
    cs = sm["carState"]
    controls_state = sm["controlsState"]
    if bool(getattr(cs, "brakePressed", False)):
      self.last_release_block_reason = "driver_brake"
      return False
    if bool(getattr(cs, "gasPressed", False)):
      self.last_release_block_reason = "driver_gas"
      return False
    if bool(getattr(controls_state, "forceDecel", False)):
      self.last_release_block_reason = "force_decel"
      return False
    if float(mpc_a_target) < min_mpc_a_target:
      self.last_release_block_reason = "mpc_brake_veto"
      return False
    self.last_release_block_reason = ""
    return True

  def _standstill_release_planner_gate_valid(self, sm: Any, custom_long: Any, custom_long_output: Any,
                                             mpc_a_target: float, raw_model_a_target: float, raw_model_should_stop: bool,
                                             selected_lead: Any, lead_d_rel: float, lead_v: float,
                                             lead_v_rel: float, same_id: bool) -> bool:
    if not self._standstill_release_gate_enabled(custom_long) or not same_id:
      return False
    if custom_long_output is None or not bool(getattr(custom_long_output, "enabled", False)):
      self.last_release_block_reason = "custom_output_unavailable"
      return False
    if bool(getattr(custom_long_output, "should_stop", False)):
      self.last_release_block_reason = "custom_should_stop"
      return False
    cs = sm["carState"]
    controls_state = sm["controlsState"]
    if bool(getattr(cs, "brakePressed", False)):
      self.last_release_block_reason = "driver_brake"
      return False
    if bool(getattr(cs, "gasPressed", False)):
      self.last_release_block_reason = "driver_gas"
      return False
    if bool(getattr(controls_state, "forceDecel", False)):
      self.last_release_block_reason = "force_decel"
      return False
    if raw_model_should_stop:
      self.last_release_block_reason = "raw_model_stop"
      return False
    for value in (lead_d_rel, lead_v, lead_v_rel, mpc_a_target, raw_model_a_target):
      if not math.isfinite(float(value)):
        self.last_release_block_reason = "non_finite_values"
        return False
    if float(lead_v) < 0.30 or float(lead_v_rel) < 0.25:
      self.last_release_block_reason = "lead_not_moving"
      return False
    if float(mpc_a_target) < 0.05 or float(raw_model_a_target) < 0.0:
      self.last_release_block_reason = "planner_accel_too_low"
      return False
    if self.lead_stop_hold_gap_baseline_d_rel is None:
      self.last_release_block_reason = "no_baseline_gap"
      return False
    if float(lead_d_rel) - float(self.lead_stop_hold_gap_baseline_d_rel) < 0.5:
      self.last_release_block_reason = "baseline_opening"
      return False
    self.last_release_block_reason = ""
    return True

  def _stop_hold_release_accel_for_gap(self, requested_a: float, lead_d_rel: float,
                                       lead_v: float, lead_v_rel: float, same_id: bool,
                                       valid_source: bool = False) -> float:
    release_a = min(max(float(requested_a), self._STOP_HOLD_RELEASE_A_MIN), self._STOP_HOLD_RELEASE_A_MAX)
    if self._routine_lead_launch_breakout(float(lead_v), float(lead_v_rel)):
      return float(release_a)
    if not same_id or self.lead_stop_hold_gap_baseline_d_rel is None:
      return float(release_a)

    # Crawl mode maintains the original stopped gap with a deadband instead of chasing each
    # small lead pulse. Only the gap beyond the deadband can ask for positive crawl accel.
    gap_error = float(lead_d_rel) - float(self.lead_stop_hold_gap_baseline_d_rel)
    crawl_release_a = float(min(release_a, self._STOP_HOLD_CRAWL_RELEASE_A_MAX))
    if gap_error <= self._STOP_HOLD_CRAWL_DEADBAND_M:
      # Valid source with a clearly moving lead: do not let the deadband suppress the
      # release entirely, but still keep the crawl launch cap for proportionality.
      if valid_source and float(lead_v) >= 0.30 and float(lead_v_rel) >= 0.15:
        return crawl_release_a
      return 0.0
    gap_limited_a = (gap_error - self._STOP_HOLD_CRAWL_DEADBAND_M) / self._STOP_HOLD_CRAWL_GAP_TAU
    if gap_limited_a < self._STOP_HOLD_CRAWL_RELEASE_A_MIN:
      if valid_source and float(lead_v) >= 0.30 and float(lead_v_rel) >= 0.15:
        return crawl_release_a
      return 0.0
    return float(min(release_a, self._STOP_HOLD_CRAWL_RELEASE_A_MAX, gap_limited_a))

  def _lead_stop_hold_crawl_fallback_applies(self, sm: Any, custom_long: Any, custom_long_output: Any,
                                             mpc_a_target: float, raw_model_a_target: float,
                                             raw_model_should_stop: bool, selected_lead: Any,
                                             lead_d_rel: float, lead_v: float, lead_v_rel: float,
                                             same_id: bool) -> bool:
    """Finalizer-owned crawl fallback when the same latched lead is physically opening
    but the stack's selected release source is absent or invalid.

    This is intentionally narrow: it does not bypass driver/system/model/MPC brake vetoes
    and only emits a small, gap-limited crawl release. It does not override the explicit
    gate-mode planner path, which is preserved unchanged.
    """
    if self._standstill_release_gate_enabled(custom_long):
      return False
    if not custom_long.enabled or custom_long_output is None:
      return False
    if not bool(getattr(custom_long_output, "enabled", False)):
      return False
    if bool(getattr(custom_long_output, "should_stop", False)):
      return False
    if raw_model_should_stop:
      return False
    cs = self._sm_item(sm, 'carState')
    controls_state = self._sm_item(sm, 'controlsState')
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
    if lead_id is None or self.lead_stop_hold_lead_id is None or lead_id != self.lead_stop_hold_lead_id:
      return False
    if self.lead_stop_hold_gap_baseline_d_rel is None:
      return False
    for value in (lead_d_rel, lead_v, lead_v_rel, mpc_a_target, raw_model_a_target):
      if not math.isfinite(float(value)):
        return False
    if float(raw_model_a_target) < 0.0:
      return False
    if float(mpc_a_target) < self._STOP_HOLD_SAME_ID_MIN_MPC_A_TARGET:
      return False
    if float(lead_v_rel) < 0.05:
      return False
    baseline_opening = float(lead_d_rel) - float(self.lead_stop_hold_gap_baseline_d_rel)
    if float(lead_v) < 0.20 and baseline_opening < 0.6:
      return False
    return True

  def _lead_stop_hold_release_accepts(self, sm: Any, custom_long: Any, custom_long_output: Any,
                                      mpc_a_target: float, raw_model_a_target: float,
                                      raw_model_should_stop: bool, selected_lead: Any, lead_d_rel: float,
                                      lead_v: float, lead_v_rel: float) -> tuple[bool, float]:
    if selected_lead is None:
      self.last_release_block_reason = "no_lead"
      return False, float(lead_d_rel)
    release_source = str(getattr(custom_long_output, "standstill_release_source", ""))
    lead_id = getattr(selected_lead, 'radarTrackId', None)
    same_id = lead_id is not None and self.lead_stop_hold_lead_id is not None and lead_id == self.lead_stop_hold_lead_id
    source_valid = release_source in ("lead_pullaway", "lead_standstill_launch")
    if lead_id is not None and self.lead_stop_hold_lead_id is not None and lead_id != self.lead_stop_hold_lead_id:
      self.last_release_block_reason = "different_lead_id"
      return False, float(lead_d_rel)
    crawl_fallback = bool(
      not source_valid and
      self._lead_stop_hold_crawl_fallback_applies(
        sm, custom_long, custom_long_output, mpc_a_target, raw_model_a_target,
        raw_model_should_stop, selected_lead, lead_d_rel, lead_v, lead_v_rel, same_id,
      )
    )
    gate_fallback_candidate = bool(
      not source_valid and not crawl_fallback and
      self._standstill_release_gate_enabled(custom_long) and same_id and
      bool(getattr(custom_long_output, "research_actuation_allowed", False))
    )
    if not source_valid and not crawl_fallback and not gate_fallback_candidate:
      self.last_release_block_reason = "invalid_release_source"
      return False, float(lead_d_rel)
    for value in (lead_d_rel, lead_v, lead_v_rel, mpc_a_target, raw_model_a_target):
      if not math.isfinite(float(value)):
        self.last_release_block_reason = "non_finite_values"
        return False, float(lead_d_rel)
    stopping_distance = float(getattr(self.CP, 'stoppingDistance', 6.0) or 6.0)
    if not crawl_fallback:
      if float(lead_v) < 0.30 or float(lead_v_rel) < 0.15:
        self.last_release_block_reason = "lead_not_moving"
        return False, float(lead_d_rel)
    min_d_rel = stopping_distance + self._STOP_HOLD_SAME_ID_MIN_D_REL_MARGIN if same_id else stopping_distance + 0.1
    if same_id and self.lead_stop_hold_gap_baseline_d_rel is not None:
      baseline_opening = self._STOP_HOLD_SAME_ID_VALID_BASELINE_OPENING_M if source_valid else self._STOP_HOLD_SAME_ID_MIN_D_REL_BASELINE_OPENING
      baseline_min_d_rel = float(self.lead_stop_hold_gap_baseline_d_rel) + baseline_opening
      min_d_rel = max(self._STOP_HOLD_SAME_ID_MIN_D_REL_FLOOR, min(min_d_rel, baseline_min_d_rel))
    if float(lead_d_rel) <= min_d_rel:
      self.last_release_block_reason = "distance_gate"
      return False, float(lead_d_rel)
    if same_id:
      if source_valid:
        min_gap_increasing_s = self._STOP_HOLD_SAME_ID_VALID_GAP_INCREASING_S
      elif self._routine_lead_launch_breakout(float(lead_v), float(lead_v_rel)):
        min_gap_increasing_s = self._STOP_HOLD_SAME_ID_ROUTINE_PULLAWAY_S
      elif gate_fallback_candidate:
        min_gap_increasing_s = self._STOP_HOLD_SAME_ID_GATE_MIN_PULLAWAY_S
      else:
        min_gap_increasing_s = self._STOP_HOLD_SAME_ID_MIN_PULLAWAY_S
    else:
      min_gap_increasing_s = 0.15
    if self.lead_stop_hold_gap_increasing_s < min_gap_increasing_s:
      self.last_release_block_reason = "gap_increasing_time"
      return False, float(lead_d_rel)
    if same_id and self.lead_stop_hold_gap_baseline_d_rel is not None:
      min_baseline_opening = 0.5 if crawl_fallback else (self._STOP_HOLD_SAME_ID_VALID_BASELINE_OPENING_M if source_valid else 0.3)
      if float(lead_d_rel) - float(self.lead_stop_hold_gap_baseline_d_rel) < min_baseline_opening:
        self.last_release_block_reason = "baseline_opening"
        return False, float(lead_d_rel)
    if lead_id is None or self.lead_stop_hold_lead_id is None:
      if self.lead_stop_hold_gap_increasing_s < self._STOP_HOLD_NEW_ID_GAP_INCREASING_S:
        self.last_release_block_reason = "new_id_gap_increasing_time"
        return False, float(lead_d_rel)
    if crawl_fallback:
      self.last_release_block_reason = ""
      requested_a = max(float(mpc_a_target), self._STOP_HOLD_CRAWL_RELEASE_A_MIN)
      release_a = self._stop_hold_release_accel_for_gap(requested_a, lead_d_rel, lead_v, lead_v_rel, same_id, valid_source=False)
      if release_a <= 0.0:
        self.last_release_block_reason = "crawl_deadband"
        return False, float(lead_d_rel)
      return True, release_a
    if not self._standstill_release_request_valid(
      sm, custom_long, custom_long_output, mpc_a_target, raw_model_a_target, raw_model_should_stop,
      self._STOP_HOLD_SAME_ID_MIN_MPC_A_TARGET if same_id else -0.03,
    ):
      if not gate_fallback_candidate:
        # block_reason already set by one of the release validators
        return False, float(lead_d_rel)
      if not self._standstill_release_planner_gate_valid(
        sm, custom_long, custom_long_output, mpc_a_target, raw_model_a_target, raw_model_should_stop,
        selected_lead, lead_d_rel, lead_v, lead_v_rel, same_id,
      ):
        # block_reason already set by one of the release validators
        return False, float(lead_d_rel)
      self.last_release_block_reason = ""
      release_a = self._stop_hold_release_accel_for_gap(float(mpc_a_target), lead_d_rel, lead_v, lead_v_rel, same_id, valid_source=False)
      if release_a <= 0.0:
        self.last_release_block_reason = "crawl_deadband"
        return False, float(lead_d_rel)
      return True, release_a
    self.last_release_block_reason = ""
    requested_release_a = float(getattr(custom_long_output, "standstill_release_a_target", 0.0)) if custom_long_output is not None else 0.0
    release_a = self._stop_hold_release_accel_for_gap(requested_release_a, lead_d_rel, lead_v, lead_v_rel, same_id, valid_source=True)
    if release_a <= 0.0:
      self.last_release_block_reason = "crawl_deadband"
      return False, float(lead_d_rel)
    return True, release_a

  def finalize(self, sm: Any, custom_long: Any, custom_long_output: Any, is_e2e: bool,
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
        return FinalizerResult(
          a_target=float(a_target),
          should_stop=bool(mpc_should_stop or raw_model_should_stop),
          e2e_source=bool(a_target < mpc_a_target),
          custom_long_output_telemetry=None,
          last_release_block_reason="",
        )
      return FinalizerResult(
        a_target=float(mpc_a_target),
        should_stop=bool(mpc_should_stop),
        e2e_source=False,
        custom_long_output_telemetry=None,
        last_release_block_reason="",
      )

    car_state = self._sm_item(sm, 'carState')
    radar_state = self._sm_item(sm, 'radarState')
    selected_lead = self._select_stop_hold_lead(radar_state) if radar_state is not None else None
    has_lead = selected_lead is not None
    lead_d_rel = float(getattr(selected_lead, 'dRel', 0.0) or 0.0) if selected_lead is not None else 0.0
    lead_v = float(getattr(selected_lead, 'vLead', 0.0) or 0.0) if selected_lead is not None else 0.0
    lead_v_rel = float(getattr(selected_lead, 'vRel', 0.0) or 0.0) if selected_lead is not None else 0.0
    gas_pressed = bool(getattr(car_state, 'gasPressed', False)) if car_state is not None else False
    v_ego = float(getattr(car_state, 'vEgo', 0.0) or 0.0) if car_state is not None else 0.0

    lead_stop_hold_active = self._update_lead_stop_hold(
      sm, v_ego, has_lead, selected_lead, lead_d_rel, lead_v, lead_v_rel, gas_pressed, dt,
      reset_lead_stop_hold)
    release_mpc_stop = False
    release_a_target = float(mpc_a_target)
    mpc_stop = bool(mpc_should_stop)

    if lead_stop_hold_active:
      current_custom_long_output = custom_long_output
      latch_release_ok, latch_release_a = self._lead_stop_hold_release_accepts(
        sm, custom_long, current_custom_long_output, mpc_a_target, raw_model_a_target,
        raw_model_should_stop, selected_lead, lead_d_rel, lead_v, lead_v_rel)
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
      release_mpc_stop, release_a_target = self._standstill_release_clears_mpc_stop(
        sm, custom_long, custom_long_output, mpc_a_target, mpc_should_stop,
        raw_model_a_target, raw_model_should_stop)
      mpc_stop = bool(mpc_should_stop and not release_mpc_stop)

    custom_should_stop = self.custom_longitudinal_should_stop(
      custom_long, custom_long_output, mpc_stop, raw_model_should_stop, model_stale)
    should_stop = bool(custom_should_stop if custom_should_stop is not None else (mpc_stop or (raw_model_should_stop and is_e2e and not model_stale)))

    if lead_stop_hold_active:
      self.stop_hold_release_slew_a_target = None
      stop_accel = getattr(self.CP, 'stopAccel', None)
      stop_accel = -0.5 if stop_accel is None else float(stop_accel)
      hold_a_target = float(mpc_a_target) if math.isfinite(float(mpc_a_target)) else stop_accel
      if is_e2e and not model_stale:
        raw_a_target = float(raw_model_a_target) if math.isfinite(float(raw_model_a_target)) else stop_accel
        raw_hold = min(raw_a_target, hold_a_target, stop_accel)
      else:
        raw_hold = min(hold_a_target, stop_accel)
      e2e_source = bool(is_e2e and not model_stale and raw_hold < hold_a_target)

      # Stop-hold command normalization: clamp harsh hold commands up to a local mild
      # hold target when stopped or creeping into the final stop, avoiding an artificial
      # jump to the first positive release. Does not delay braking above creeping speed.
      controls_state_sp = self._sm_item(sm, 'controlsState')
      brake_pressed_sp = bool(getattr(car_state, 'brakePressed', False)) if car_state is not None else False
      force_decel_sp = bool(getattr(controls_state_sp, 'forceDecel', False)) if controls_state_sp is not None else False
      lead_id_sp = getattr(selected_lead, 'radarTrackId', None) if selected_lead is not None else None
      same_id_sp = lead_id_sp is not None and self.lead_stop_hold_lead_id is not None and lead_id_sp == self.lead_stop_hold_lead_id
      standstill_sp = bool(getattr(car_state, 'standstill', False)) if car_state is not None else False
      if (
        (standstill_sp or v_ego <= self._STOP_HOLD_STANDSTILL_NORMALIZE_MAX_V_EGO) and
        selected_lead is not None and
        same_id_sp and
        math.isfinite(raw_hold) and
        not brake_pressed_sp and
        not gas_pressed and
        not force_decel_sp and
        not raw_model_should_stop
      ):
        raw_hold = max(float(raw_hold), self._STOP_HOLD_STANDSTILL_NORMALIZED_A_TARGET)

      a_target = self._apply_stop_hold_release_prep(
        sm, dt, raw_hold, selected_lead, custom_long, custom_long_output, lead_d_rel, lead_v, lead_v_rel,
        mpc_a_target, raw_model_a_target, raw_model_should_stop,
      )
      if custom_long_output is not None:
        self.custom_long_output_telemetry = replace(custom_long_output, should_stop=True, selected_intent="lead_stop_hold", reason="stopped_lead_latch")
      return FinalizerResult(
        a_target=float(a_target),
        should_stop=True,
        e2e_source=e2e_source,
        custom_long_output_telemetry=self.custom_long_output_telemetry,
        last_release_block_reason=self.last_release_block_reason,
      )

    if is_e2e and not model_stale:
      a_target = min(raw_model_a_target, release_a_target if release_mpc_stop else mpc_a_target)
      e2e_source = bool(a_target < mpc_a_target)
      a_target = apply_stop_hold_release_slew(sm, a_target, release_mpc_stop, mpc_stop, raw_model_should_stop, should_stop)
      return FinalizerResult(
        a_target=float(a_target),
        should_stop=should_stop,
        e2e_source=e2e_source,
        custom_long_output_telemetry=self.custom_long_output_telemetry,
        last_release_block_reason=self.last_release_block_reason,
      )

    a_target = float(release_a_target if release_mpc_stop else mpc_a_target)
    a_target = self._scc_custom_stop_cap(a_target, custom_long, custom_long_output)
    a_target = self._scc_curve_confidence_final_cap(a_target, sm, custom_long, custom_long_output, release_mpc_stop=release_mpc_stop)
    a_target = self._scc_cut_in_brake_assist_final_cap(a_target, sm, custom_long, custom_long_output, release_mpc_stop=release_mpc_stop)
    a_target = self._scc_curve_traffic_advisor_final_cap(a_target, sm, custom_long, custom_long_output, release_mpc_stop=release_mpc_stop)
    a_target = apply_stop_hold_release_slew(sm, a_target, release_mpc_stop, mpc_stop, raw_model_should_stop, should_stop)
    return FinalizerResult(
      a_target=a_target,
      should_stop=bool(should_stop),
      e2e_source=False,
      custom_long_output_telemetry=self.custom_long_output_telemetry,
      last_release_block_reason=self.last_release_block_reason,
    )

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
