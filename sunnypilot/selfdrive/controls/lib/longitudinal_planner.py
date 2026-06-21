"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

import math
from dataclasses import replace

from cereal import messaging, custom
from openpilot.common.constants import CV
from openpilot.selfdrive.car.cruise import V_CRUISE_MAX
from openpilot.sunnypilot.selfdrive.controls.lib.dec.dec import DynamicExperimentalController
from openpilot.sunnypilot.selfdrive.controls.lib.e2e_alerts_helper import E2EAlertsHelper
from openpilot.sunnypilot.selfdrive.controls.lib.smart_cruise_control.smart_cruise_control import SmartCruiseControl
from openpilot.sunnypilot.selfdrive.controls.lib.speed_limit.speed_limit_assist import SpeedLimitAssist
from openpilot.sunnypilot.selfdrive.controls.lib.speed_limit.speed_limit_resolver import SpeedLimitResolver
from openpilot.sunnypilot.selfdrive.selfdrived.events import EventsSP
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.car.cruise import V_CRUISE_UNSET
from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState
from openpilot.sunnypilot.custom.longitudinal.lead_anticipation import LeadAnticipation
from openpilot.sunnypilot.custom.longitudinal.modes import EvidenceClass, LongitudinalMode, admitted_evidence
from openpilot.sunnypilot.custom.longitudinal.wiring import CustomLongitudinalAdapter, MODEL_STALE_AGE_S, _message_age_s

DecState = custom.LongitudinalPlanSP.DynamicExperimentalControl.DynamicExperimentalControlState
LongitudinalPlanSource = custom.LongitudinalPlanSP.LongitudinalPlanSource


class LongitudinalPlannerSP:
  _STOP_HOLD_SAME_ID_MIN_D_REL_MARGIN = 0.2
  _STOP_HOLD_SAME_ID_MIN_D_REL_FLOOR = 4.5
  _STOP_HOLD_SAME_ID_MIN_D_REL_BASELINE_OPENING = 0.5
  _STOP_HOLD_SAME_ID_GAP_INCREASING_S = 0.10
  _STOP_HOLD_SAME_ID_MIN_MPC_A_TARGET = -0.10
  _STOP_HOLD_NEW_ID_GAP_INCREASING_S = 0.30
  _STOP_HOLD_SAME_ID_MIN_PULLAWAY_S = 0.30
  _STOP_HOLD_SAME_ID_GATE_MIN_PULLAWAY_S = 0.15
  _STOP_HOLD_RELEASE_A_MIN = 0.15
  _STOP_HOLD_RELEASE_A_MAX = 0.35
  _STOP_HOLD_RELEASE_MAX_UP_JERK = 6.0
  # Pre-release only: relax harsh stopped-lead hold when the same latched lead is pulling away.
  _STOP_HOLD_RELEASE_PREP_A_TARGET = -0.20
  _STOP_HOLD_RELEASE_PREP_MAX_UP_JERK = 6.0
  _STOP_HOLD_RELEASE_PREP_MIN_LEAD_V = 0.25
  _STOP_HOLD_RELEASE_PREP_MIN_LEAD_V_REL = 0.10
  _STOP_HOLD_RELEASE_PREP_MIN_MPC_A_TARGET = -0.10
  _STOP_HOLD_RELEASE_PREP_MIN_GAP_INCREASING_S = 0.15
  _STOP_HOLD_RELEASE_PREP_MIN_D_REL_MARGIN = 0.20
  _STOP_HOLD_STANDSTILL_NORMALIZED_A_TARGET = -0.50
  _STOP_HOLD_STANDSTILL_NORMALIZE_MAX_V_EGO = 0.02
  _CURVE_CONFIDENCE_APPLY_MIN_V_EGO = 8.0
  _CURVE_CONFIDENCE_APPLY_MIN_CONFIDENCE = 0.70
  _CURVE_CONFIDENCE_APPLY_MIN_CAP = -0.85
  _stop_hold_release_slew_a_target: float | None
  _stop_hold_release_prep_a_target: float | None
  _stop_hold_release_prep_raw_prev: float | None
  def __init__(self, CP, CP_SP, mpc):
    self.CP = CP
    self.events_sp = EventsSP()
    self.dec = DynamicExperimentalController(CP, mpc)
    self.scc = SmartCruiseControl()
    self.resolver = SpeedLimitResolver()
    self.sla = SpeedLimitAssist(CP, CP_SP)
    self.source = LongitudinalPlanSource.cruise
    self.e2e_alerts_helper = E2EAlertsHelper()

    self.output_v_target = 0.
    self.output_a_target = 0.
    self._lead_stop_hold_active = False
    self._lead_stop_hold_gap_increasing_s = 0.0
    self._lead_stop_hold_missing_s = 0.0
    self._lead_stop_hold_lead_id = None
    self._lead_stop_hold_gap_prev_d_rel = None
    self._lead_stop_hold_gap_baseline_d_rel = None
    self._custom_long_output_telemetry = None
    self._last_release_block_reason = ""
    self._stop_hold_release_slew_a_target: float | None = None
    self._stop_hold_release_prep_a_target: float | None = None
    self._stop_hold_release_prep_raw_prev: float | None = None

    # Custom-2.0 longitudinal policy (default-on in this fork; fail-closed to stock output).
    self.custom_long = CustomLongitudinalAdapter(Params())
    self.custom_long_output = None
    self._custom_long_output_telemetry = None
    # §3 lead-motion anticipation: mode-gated shadow/apply shaping of lead accel fed to the MPC
    # (off/shadow/apply via LeadAnticipationMode; compatibility LeadAnticipationEnabled only when
    # mode is absent, fail-closed to the raw radarState).
    self.lead_anticipation = LeadAnticipation(Params())

  @staticmethod
  def _sm_item(sm, key):
    if hasattr(sm, 'get'):
      return sm.get(key)
    try:
      return sm[key]
    except Exception:
      return None

  @staticmethod
  def _select_stop_hold_lead(radar_state):
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

  def _reset_lead_stop_hold(self) -> None:
    self._lead_stop_hold_active = False
    self._lead_stop_hold_gap_increasing_s = 0.0
    self._lead_stop_hold_missing_s = 0.0
    self._lead_stop_hold_lead_id = None
    self._lead_stop_hold_gap_prev_d_rel = None
    self._lead_stop_hold_gap_baseline_d_rel = None
    self._stop_hold_release_slew_a_target = None
    self._stop_hold_release_prep_a_target = None
    self._stop_hold_release_prep_raw_prev = None

  def _update_lead_stop_hold(self, sm: messaging.SubMaster, v_ego: float, has_lead: bool, selected_lead,
                             lead_d_rel: float, lead_v: float, lead_v_rel: float, gas_pressed: bool) -> bool:
    dt = float(getattr(self, 'dt', DT_MDL))
    lead_id = getattr(selected_lead, 'radarTrackId', None) if selected_lead is not None else None

    stopping_distance = float(getattr(self.CP, 'stoppingDistance', 6.0) or 6.0)
    arm_distance = max(stopping_distance + 2.0, 10.0)
    release_distance = stopping_distance + 1.0
    v_ego_stopping = float(getattr(self.CP, 'vEgoStopping', 0.0))

    # Arm check (only when not already latched).
    stop_hold_set = bool(
      not self._lead_stop_hold_active and
      has_lead and
      v_ego < v_ego_stopping + 0.2 and
      lead_d_rel <= arm_distance and
      lead_v <= 0.3 and
      not gas_pressed,
    )
    if stop_hold_set:
      self._lead_stop_hold_active = True
      self._lead_stop_hold_gap_increasing_s = 0.0
      self._lead_stop_hold_missing_s = 0.0
      self._lead_stop_hold_gap_prev_d_rel = float(lead_d_rel)
      self._lead_stop_hold_gap_baseline_d_rel = float(lead_d_rel)
      self._lead_stop_hold_lead_id = lead_id

    if self._lead_stop_hold_active:
      if gas_pressed:
        self._reset_lead_stop_hold()
      elif lead_id is not None and self._lead_stop_hold_lead_id is not None and lead_id != self._lead_stop_hold_lead_id:
        # A different lead appeared while latched.
        if lead_v <= 0.3 and lead_d_rel <= arm_distance:
          # Valid stopped hold candidate: transfer latch immediately (no one-cycle gap).
          self._lead_stop_hold_lead_id = lead_id
          self._lead_stop_hold_gap_prev_d_rel = float(lead_d_rel)
          self._lead_stop_hold_gap_baseline_d_rel = float(lead_d_rel)
          self._lead_stop_hold_missing_s = 0.0
          self._lead_stop_hold_gap_increasing_s = 0.0
        else:
          # Non-stopped transient: treat as dropout within the 0.5 s grace window.
          self._lead_stop_hold_missing_s += dt
          if not (self._lead_stop_hold_missing_s < 0.5 and v_ego < v_ego_stopping + 0.2 and not gas_pressed):
            self._reset_lead_stop_hold()
      elif not has_lead:
        self._lead_stop_hold_missing_s += dt
        if not (self._lead_stop_hold_missing_s < 0.5 and v_ego < v_ego_stopping + 0.2 and not gas_pressed):
          self._reset_lead_stop_hold()
      else:
        self._lead_stop_hold_missing_s = 0.0
        gap_increasing = self._lead_stop_hold_gap_prev_d_rel is not None and float(lead_d_rel) > float(self._lead_stop_hold_gap_prev_d_rel)
        if gap_increasing:
          self._lead_stop_hold_gap_increasing_s += dt
        else:
          self._lead_stop_hold_gap_increasing_s = 0.0
        self._lead_stop_hold_gap_prev_d_rel = float(lead_d_rel)
        if self._lead_stop_hold_gap_baseline_d_rel is None:
          self._lead_stop_hold_gap_baseline_d_rel = float(lead_d_rel)
    else:
      self._lead_stop_hold_gap_increasing_s = 0.0
      self._lead_stop_hold_gap_prev_d_rel = float(lead_d_rel) if has_lead else None
      self._lead_stop_hold_gap_baseline_d_rel = float(lead_d_rel) if has_lead else None
      self._lead_stop_hold_missing_s = 0.0

    return self._lead_stop_hold_active

  def is_e2e(self, sm: messaging.SubMaster) -> bool:
    experimental_mode = sm['selfdriveState'].experimentalMode
    if self.custom_long.enabled:
      return self.custom_long.mode is LongitudinalMode.E2E

    # Custom off: preserve legacy ExperimentalMode + DEC behavior unchanged.
    if not self.dec.active():
      return experimental_mode

    return experimental_mode and self.dec.mode() == "blended"

  def update_targets(self, sm: messaging.SubMaster, v_ego: float, a_ego: float, v_cruise: float) -> tuple[float, float]:
    self.custom_long.maybe_refresh_params()

    CS = sm['carState']
    v_cruise_cluster_kph = min(CS.vCruiseCluster, V_CRUISE_MAX)
    v_cruise_cluster = v_cruise_cluster_kph * CV.KPH_TO_MS

    long_enabled = sm['carControl'].enabled
    long_override = sm['carControl'].cruiseControl.override

    # Smart Cruise Control
    self.scc.update(sm, long_enabled, long_override, v_ego, a_ego, v_cruise)

    # Speed Limit Resolver
    self.resolver.update(v_ego, sm)

    # Speed Limit Assist
    has_speed_limit = self.resolver.speed_limit_valid or self.resolver.speed_limit_last_valid
    self.sla.update(long_enabled, long_override, v_ego, a_ego, v_cruise_cluster, self.resolver.speed_limit,
                    self.resolver.speed_limit_final_last, has_speed_limit, self.resolver.distance, self.events_sp)

    targets = {
      LongitudinalPlanSource.cruise: (v_cruise, a_ego),
      LongitudinalPlanSource.sccVision: (self.scc.vision.output_v_target, self.scc.vision.output_a_target),
      LongitudinalPlanSource.sccMap: (self.scc.map.output_v_target, self.scc.map.output_a_target),
      LongitudinalPlanSource.speedLimitAssist: (self.sla.output_v_target, self.sla.output_a_target),
    }

    filtered_targets = self.custom_longitudinal_targets(targets)
    self.source = min(filtered_targets, key=lambda k: filtered_targets[k][0])
    self.output_v_target, self.output_a_target = filtered_targets[self.source]

    # Opt-in: shape the baseline a_target with the custom-2.0 policy (fail-closed; returns the
    # unchanged target when disabled or on any fault, so default behavior is never affected).
    self.custom_long_output = self.custom_long.evaluate(
      sm, v_ego, a_ego, v_cruise, self.output_a_target, self.scc, self.sla,
    )
    self.output_a_target = self.custom_long_output.a_target
    return self.output_v_target, self.output_a_target

  def custom_longitudinal_should_stop(self, mpc_should_stop: bool, raw_model_should_stop: bool,
                                      model_stale: bool = False) -> bool | None:
    if not self.custom_long.enabled or self.custom_long_output is None:
      return None
    if self.custom_long.mode is LongitudinalMode.ACC:
      return bool(mpc_should_stop)
    if self.custom_long.mode is LongitudinalMode.E2E:
      return bool(mpc_should_stop or (raw_model_should_stop and not model_stale))
    return bool(mpc_should_stop or self.custom_long_output.should_stop)

  def _scc_custom_stop_cap(self, base_a_target: float) -> float:
    if self.custom_long.mode is not LongitudinalMode.SCC or not self.custom_long.enabled or self.custom_long_output is None:
      return float(base_a_target)
    if not bool(getattr(self.custom_long_output, "enabled", False)):
      return float(base_a_target)
    if str(getattr(self.custom_long_output, "selected_intent", "") or "") != "stop_approach":
      return float(base_a_target)
    raw_custom_a = getattr(self.custom_long_output, "a_target", None)
    if raw_custom_a is None:
      return float(base_a_target)
    try:
      custom_a = float(raw_custom_a)
    except (TypeError, ValueError):
      return float(base_a_target)
    if not math.isfinite(custom_a):
      return float(base_a_target)
    return float(min(float(base_a_target), custom_a))

  def _scc_curve_confidence_final_cap(self, base_a_target: float, sm: messaging.SubMaster,
                                      release_mpc_stop: bool = False) -> float:
    if release_mpc_stop:
      return float(base_a_target)
    if self.custom_long.mode is not LongitudinalMode.SCC or not self.custom_long.enabled or self.custom_long_output is None:
      return float(base_a_target)
    if not bool(getattr(self.custom_long_output, "enabled", False)):
      return float(base_a_target)
    if str(getattr(self.custom_long, "curve_speed_confidence_mode", "off") or "off") != "apply_conservative":
      return float(base_a_target)
    debug = dict(getattr(self.custom_long_output, "debug", {}) or {})
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

  def _apply_stop_hold_release_slew(self, sm: messaging.SubMaster, a_target: float, release_mpc_stop: bool,
                                    mpc_stop: bool, raw_model_should_stop: bool, should_stop: bool) -> float:
    """Narrow upward-only slew limiter for stop-hold release pullaways.

    Seeds on the first positive release tick with the actual final output returned, then
    caps only upward increases to reduce launch jerk. Downward/braking changes pass through
    unmodified so hazard responses are not delayed. Cleared on any stop/override/hazard.
    """
    dt = float(getattr(self, 'dt', DT_MDL))
    car_state = self._sm_item(sm, 'carState')
    controls_state = self._sm_item(sm, 'controlsState')
    brake_pressed = bool(getattr(car_state, 'brakePressed', False)) if car_state is not None else False
    gas_pressed = bool(getattr(car_state, 'gasPressed', False)) if car_state is not None else False
    force_decel = bool(getattr(controls_state, 'forceDecel', False)) if controls_state is not None else False

    clear = (
      not math.isfinite(a_target) or
      not math.isfinite(dt) or dt <= 0.0 or
      (self._stop_hold_release_slew_a_target is not None and not math.isfinite(self._stop_hold_release_slew_a_target)) or
      bool(should_stop) or
      bool(mpc_stop) or
      bool(raw_model_should_stop) or
      brake_pressed or gas_pressed or force_decel or
      a_target <= 0.0
    )
    if clear:
      self._stop_hold_release_slew_a_target = None
      return float(a_target)

    # First positive release tick seeds the slew state with the actual final output returned.
    if release_mpc_stop and self._stop_hold_release_slew_a_target is None:
      self._stop_hold_release_slew_a_target = float(a_target)
      return float(a_target)

    # Active slew: cap only upward increases; downward/braking changes pass through.
    if self._stop_hold_release_slew_a_target is not None:
      max_step = self._STOP_HOLD_RELEASE_MAX_UP_JERK * dt
      last_slew = self._stop_hold_release_slew_a_target
      if a_target > last_slew + max_step:
        a_target = last_slew + max_step
        self._stop_hold_release_slew_a_target = float(a_target)
      elif a_target > last_slew:
        self._stop_hold_release_slew_a_target = None
      else:
        self._stop_hold_release_slew_a_target = float(a_target)

    return float(a_target)

  def _stop_hold_release_prep_applies(self, sm: messaging.SubMaster, selected_lead,
                                      lead_d_rel: float, lead_v: float, lead_v_rel: float,
                                      mpc_a_target: float, raw_model_a_target: float,
                                      raw_model_should_stop: bool) -> bool:
    """Return True when early release evidence justifies relaxing the stop-hold accel."""
    if not self.custom_long.enabled or self.custom_long_output is None:
      return False
    if not bool(getattr(self.custom_long_output, "standstill_release_allowed", False)):
      return False
    if str(getattr(self.custom_long_output, "standstill_release_source", "")) not in ("lead_pullaway", "lead_standstill_launch"):
      return False
    if bool(getattr(self.custom_long_output, "should_stop", False)):
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
    if lead_id is None or self._lead_stop_hold_lead_id is None or lead_id != self._lead_stop_hold_lead_id:
      return False
    for value in (lead_d_rel, lead_v, lead_v_rel, mpc_a_target, raw_model_a_target):
      if not math.isfinite(float(value)):
        return False

    if float(mpc_a_target) < self._STOP_HOLD_RELEASE_PREP_MIN_MPC_A_TARGET:
      return False
    if float(lead_v) < self._STOP_HOLD_RELEASE_PREP_MIN_LEAD_V or float(lead_v_rel) < self._STOP_HOLD_RELEASE_PREP_MIN_LEAD_V_REL:
      return False
    if self._lead_stop_hold_gap_increasing_s < self._STOP_HOLD_RELEASE_PREP_MIN_GAP_INCREASING_S:
      return False

    stopping_distance = float(getattr(self.CP, 'stoppingDistance', 6.0) or 6.0)
    if float(lead_d_rel) <= stopping_distance + self._STOP_HOLD_RELEASE_PREP_MIN_D_REL_MARGIN:
      return False

    return True

  def _apply_stop_hold_release_prep(self, sm: messaging.SubMaster, raw_hold: float, selected_lead,
                                    lead_d_rel: float, lead_v: float, lead_v_rel: float,
                                    mpc_a_target: float, raw_model_a_target: float,
                                    raw_model_should_stop: bool) -> float:
    """Relax a harsh stop-hold accel toward a mild negative target before the first positive release.

    Applied only while the lead stop-hold latch is active. Upward changes are limited by a
    jerk cap; downward/braking changes pass through immediately so hazard responses are not
    delayed.
    """
    dt = float(getattr(self, 'dt', DT_MDL))
    state = self._stop_hold_release_prep_a_target
    raw_prev = self._stop_hold_release_prep_raw_prev

    clear = (
      not math.isfinite(raw_hold) or
      not math.isfinite(dt) or dt <= 0.0 or
      (state is not None and not math.isfinite(state)) or
      (raw_prev is not None and not math.isfinite(raw_prev)) or
      not self._stop_hold_release_prep_applies(
        sm, selected_lead, lead_d_rel, lead_v, lead_v_rel,
        mpc_a_target, raw_model_a_target, raw_model_should_stop,
      )
    )
    if clear:
      self._stop_hold_release_prep_a_target = None
      self._stop_hold_release_prep_raw_prev = None
      return float(raw_hold)

    prev_output = float(state) if state is not None else float(raw_hold)
    prev_raw = float(raw_prev) if raw_prev is not None else float(raw_hold)

    # Downward / braking change: raw hold became more negative than last cycle.
    if raw_hold < prev_raw:
      self._stop_hold_release_prep_a_target = float(raw_hold)
      self._stop_hold_release_prep_raw_prev = float(raw_hold)
      return float(raw_hold)

    desired = max(float(raw_hold), self._STOP_HOLD_RELEASE_PREP_A_TARGET)
    max_step = self._STOP_HOLD_RELEASE_PREP_MAX_UP_JERK * dt
    if desired > prev_output + max_step:
      limited = prev_output + max_step
    else:
      limited = desired
    limited = max(limited, prev_output)  # never drift downward
    self._stop_hold_release_prep_a_target = float(limited)
    self._stop_hold_release_prep_raw_prev = float(raw_hold)
    return float(limited)

  def _standstill_release_gate_enabled(self) -> bool:
    return bool(
      self.custom_long.enabled and
      self.custom_long.mode is LongitudinalMode.SCC and
      str(getattr(self.custom_long, "standstill_release_confidence_mode", "off") or "off") == "gate"
    )

  def custom_longitudinal_targets(self, targets: dict) -> dict:
    if not self.custom_long.enabled:
      return targets
    admitted = admitted_evidence(self.custom_long.mode, self.custom_long.sources)
    source_to_evidence = {
      LongitudinalPlanSource.cruise: EvidenceClass.CRUISE,
      LongitudinalPlanSource.sccVision: EvidenceClass.CURVE_VISION,
      LongitudinalPlanSource.sccMap: EvidenceClass.CURVE_MAP,
      LongitudinalPlanSource.speedLimitAssist: EvidenceClass.SPEED_LIMIT,
    }
    filtered = {src: target for src, target in targets.items() if source_to_evidence[src] in admitted}
    filtered[LongitudinalPlanSource.cruise] = targets[LongitudinalPlanSource.cruise]
    return filtered

  def _standstill_release_clears_mpc_stop(self, sm: messaging.SubMaster, mpc_a_target: float, mpc_should_stop: bool,
                                          raw_model_a_target: float, raw_model_should_stop: bool) -> tuple[bool, float]:
    if not self._standstill_release_request_valid(sm, self.custom_long_output, mpc_a_target, raw_model_a_target, raw_model_should_stop):
      return False, float(mpc_a_target)
    if not mpc_should_stop:
      return False, float(mpc_a_target)
    release_a = min(
      max(float(mpc_a_target), self._STOP_HOLD_RELEASE_A_MIN, float(getattr(self.custom_long_output, "standstill_release_a_target", 0.0))),
      self._STOP_HOLD_RELEASE_A_MAX,
    )
    return True, release_a

  def _standstill_release_request_valid(self, sm: messaging.SubMaster, custom_long_output,
                                        mpc_a_target: float, raw_model_a_target: float, raw_model_should_stop: bool,
                                        min_mpc_a_target: float = -0.03) -> bool:
    if not self.custom_long.enabled or custom_long_output is None or not bool(getattr(custom_long_output, "standstill_release_allowed", False)):
      self._last_release_block_reason = "no_release_permission"
      return False
    if str(getattr(custom_long_output, "standstill_release_source", "")) not in ("lead_pullaway", "lead_standstill_launch", "no_lead_launch"):
      self._last_release_block_reason = "invalid_release_source"
      return False
    if bool(getattr(custom_long_output, "should_stop", False)):
      self._last_release_block_reason = "custom_should_stop"
      return False
    if raw_model_should_stop:
      self._last_release_block_reason = "raw_model_stop"
      return False
    for value in (mpc_a_target, raw_model_a_target):
      if not math.isfinite(float(value)):
        self._last_release_block_reason = "non_finite_target"
        return False
    if self.custom_long.mode is LongitudinalMode.E2E and float(raw_model_a_target) < 0.15:
      self._last_release_block_reason = "e2e_model_accel_too_low"
      return False
    cs = sm["carState"]
    controls_state = sm["controlsState"]
    if bool(getattr(cs, "brakePressed", False)):
      self._last_release_block_reason = "driver_brake"
      return False
    if bool(getattr(cs, "gasPressed", False)):
      self._last_release_block_reason = "driver_gas"
      return False
    if bool(getattr(controls_state, "forceDecel", False)):
      self._last_release_block_reason = "force_decel"
      return False
    if float(mpc_a_target) < min_mpc_a_target:
      self._last_release_block_reason = "mpc_brake_veto"
      return False
    self._last_release_block_reason = ""
    return True

  def _standstill_release_planner_gate_valid(self, sm: messaging.SubMaster, custom_long_output,
                                            mpc_a_target: float, raw_model_a_target: float, raw_model_should_stop: bool,
                                            selected_lead, lead_d_rel: float, lead_v: float,
                                            lead_v_rel: float, same_id: bool) -> bool:
    if not self._standstill_release_gate_enabled() or not same_id:
      return False
    if custom_long_output is None or not bool(getattr(custom_long_output, "enabled", False)):
      self._last_release_block_reason = "custom_output_unavailable"
      return False
    if bool(getattr(custom_long_output, "should_stop", False)):
      self._last_release_block_reason = "custom_should_stop"
      return False
    cs = sm["carState"]
    controls_state = sm["controlsState"]
    if bool(getattr(cs, "brakePressed", False)):
      self._last_release_block_reason = "driver_brake"
      return False
    if bool(getattr(cs, "gasPressed", False)):
      self._last_release_block_reason = "driver_gas"
      return False
    if bool(getattr(controls_state, "forceDecel", False)):
      self._last_release_block_reason = "force_decel"
      return False
    if raw_model_should_stop:
      self._last_release_block_reason = "raw_model_stop"
      return False
    for value in (lead_d_rel, lead_v, lead_v_rel, mpc_a_target, raw_model_a_target):
      if not math.isfinite(float(value)):
        self._last_release_block_reason = "non_finite_values"
        return False
    if float(lead_v) < 0.30 or float(lead_v_rel) < 0.25:
      self._last_release_block_reason = "lead_not_moving"
      return False
    if float(mpc_a_target) < 0.05 or float(raw_model_a_target) < 0.0:
      self._last_release_block_reason = "planner_accel_too_low"
      return False
    if self._lead_stop_hold_gap_baseline_d_rel is None:
      self._last_release_block_reason = "no_baseline_gap"
      return False
    if float(lead_d_rel) - float(self._lead_stop_hold_gap_baseline_d_rel) < 0.5:
      self._last_release_block_reason = "baseline_opening"
      return False
    self._last_release_block_reason = ""
    return True

  def _lead_stop_hold_release_accepts(self, sm: messaging.SubMaster, custom_long_output, mpc_a_target: float, raw_model_a_target: float,
                                      raw_model_should_stop: bool, selected_lead, lead_d_rel: float, lead_v: float,
                                      lead_v_rel: float) -> tuple[bool, float]:
    if selected_lead is None:
      self._last_release_block_reason = "no_lead"
      return False, float(lead_d_rel)
    release_source = str(getattr(custom_long_output, "standstill_release_source", ""))
    lead_id = getattr(selected_lead, 'radarTrackId', None)
    same_id = lead_id is not None and self._lead_stop_hold_lead_id is not None and lead_id == self._lead_stop_hold_lead_id
    gate_fallback_candidate = bool(self._standstill_release_gate_enabled() and same_id and release_source not in ("lead_pullaway", "lead_standstill_launch"))
    if release_source not in ("lead_pullaway", "lead_standstill_launch"):
      if not gate_fallback_candidate:
        self._last_release_block_reason = "invalid_release_source"
        return False, float(lead_d_rel)
    if lead_id is not None and self._lead_stop_hold_lead_id is not None and lead_id != self._lead_stop_hold_lead_id:
      self._last_release_block_reason = "different_lead_id"
      return False, float(lead_d_rel)
    for value in (lead_d_rel, lead_v, lead_v_rel, mpc_a_target, raw_model_a_target):
      if not math.isfinite(float(value)):
        self._last_release_block_reason = "non_finite_values"
        return False, float(lead_d_rel)
    stopping_distance = float(getattr(self.CP, 'stoppingDistance', 6.0) or 6.0)
    if float(lead_v) < 0.30 or float(lead_v_rel) < 0.15:
      self._last_release_block_reason = "lead_not_moving"
      return False, float(lead_d_rel)
    min_d_rel = stopping_distance + self._STOP_HOLD_SAME_ID_MIN_D_REL_MARGIN if same_id else stopping_distance + 0.1
    if same_id and self._lead_stop_hold_gap_baseline_d_rel is not None:
      baseline_min_d_rel = float(self._lead_stop_hold_gap_baseline_d_rel) + self._STOP_HOLD_SAME_ID_MIN_D_REL_BASELINE_OPENING
      min_d_rel = max(self._STOP_HOLD_SAME_ID_MIN_D_REL_FLOOR, min(min_d_rel, baseline_min_d_rel))
    if float(lead_d_rel) <= min_d_rel:
      self._last_release_block_reason = "distance_gate"
      return False, float(lead_d_rel)
    min_gap_increasing_s = self._STOP_HOLD_SAME_ID_MIN_PULLAWAY_S if same_id else 0.15
    if gate_fallback_candidate:
      min_gap_increasing_s = self._STOP_HOLD_SAME_ID_GATE_MIN_PULLAWAY_S
    if self._lead_stop_hold_gap_increasing_s < min_gap_increasing_s:
      self._last_release_block_reason = "gap_increasing_time"
      return False, float(lead_d_rel)
    if same_id and self._lead_stop_hold_gap_baseline_d_rel is not None:
      if float(lead_d_rel) - float(self._lead_stop_hold_gap_baseline_d_rel) < 0.3:
        self._last_release_block_reason = "baseline_opening"
        return False, float(lead_d_rel)
    if lead_id is None or self._lead_stop_hold_lead_id is None:
      if self._lead_stop_hold_gap_increasing_s < self._STOP_HOLD_NEW_ID_GAP_INCREASING_S:
        self._last_release_block_reason = "new_id_gap_increasing_time"
        return False, float(lead_d_rel)
    if not self._standstill_release_request_valid(
      sm, custom_long_output, mpc_a_target, raw_model_a_target, raw_model_should_stop,
      self._STOP_HOLD_SAME_ID_MIN_MPC_A_TARGET if same_id else -0.03,
    ):
      if not self._standstill_release_planner_gate_valid(
        sm, custom_long_output, mpc_a_target, raw_model_a_target, raw_model_should_stop,
        selected_lead, lead_d_rel, lead_v, lead_v_rel, same_id,
      ):
        # block_reason already set by one of the release validators
        return False, float(lead_d_rel)
      self._last_release_block_reason = ""
      return True, min(max(float(mpc_a_target), self._STOP_HOLD_RELEASE_A_MIN), self._STOP_HOLD_RELEASE_A_MAX)
    self._last_release_block_reason = ""
    requested_release_a = float(getattr(custom_long_output, "standstill_release_a_target", 0.0)) if custom_long_output is not None else 0.0
    return True, min(max(requested_release_a, self._STOP_HOLD_RELEASE_A_MIN), self._STOP_HOLD_RELEASE_A_MAX)

  def final_longitudinal_output(self, sm: messaging.SubMaster, mpc_a_target: float, mpc_should_stop: bool,
                                raw_model_a_target: float, raw_model_should_stop: bool) -> tuple[float, bool, bool]:
    car_state = self._sm_item(sm, 'carState')
    radar_state = self._sm_item(sm, 'radarState')
    selected_lead = self._select_stop_hold_lead(radar_state) if radar_state is not None else None
    has_lead = selected_lead is not None
    lead_d_rel = float(getattr(selected_lead, 'dRel', 0.0) or 0.0) if selected_lead is not None else 0.0
    lead_v = float(getattr(selected_lead, 'vLead', 0.0) or 0.0) if selected_lead is not None else 0.0
    lead_v_rel = float(getattr(selected_lead, 'vRel', 0.0) or 0.0) if selected_lead is not None else 0.0
    gas_pressed = bool(getattr(car_state, 'gasPressed', False)) if car_state is not None else False
    v_ego = float(getattr(car_state, 'vEgo', 0.0) or 0.0) if car_state is not None else 0.0
    lead_stop_hold_active = self._update_lead_stop_hold(sm, v_ego, has_lead, selected_lead, lead_d_rel, lead_v, lead_v_rel, gas_pressed)
    release_mpc_stop = False
    release_a_target = float(mpc_a_target)
    mpc_stop = bool(mpc_should_stop)

    if lead_stop_hold_active:
      current_custom_long_output = self.custom_long_output
      latch_release_ok, latch_release_a = self._lead_stop_hold_release_accepts(
        sm, current_custom_long_output, mpc_a_target, raw_model_a_target, raw_model_should_stop, selected_lead, lead_d_rel, lead_v, lead_v_rel)
      if latch_release_ok:
        self._reset_lead_stop_hold()
        lead_stop_hold_active = False
        mpc_stop = False
        release_mpc_stop = True
        release_a_target = latch_release_a
      else:
        mpc_stop = True
        release_mpc_stop = False
        release_a_target = float(mpc_a_target)
    else:
      self._stop_hold_release_prep_a_target = None
      self._stop_hold_release_prep_raw_prev = None
      release_mpc_stop, release_a_target = self._standstill_release_clears_mpc_stop(
        sm, mpc_a_target, mpc_should_stop, raw_model_a_target, raw_model_should_stop)
      mpc_stop = bool(mpc_should_stop and not release_mpc_stop)
    model_stale = _message_age_s(sm, 'modelV2') > MODEL_STALE_AGE_S
    custom_should_stop = self.custom_longitudinal_should_stop(mpc_stop, raw_model_should_stop, model_stale)
    is_e2e = self.is_e2e(sm)
    should_stop = bool(custom_should_stop if custom_should_stop is not None else (mpc_stop or (raw_model_should_stop and is_e2e and not model_stale)))
    if lead_stop_hold_active:
      self._stop_hold_release_slew_a_target = None
      stop_accel = getattr(self.CP, 'stopAccel', None)
      stop_accel = -0.5 if stop_accel is None else float(stop_accel)
      hold_a_target = float(mpc_a_target) if math.isfinite(float(mpc_a_target)) else stop_accel
      if is_e2e and not model_stale:
        raw_a_target = float(raw_model_a_target) if math.isfinite(float(raw_model_a_target)) else stop_accel
        raw_hold = min(raw_a_target, hold_a_target, stop_accel)
      else:
        raw_hold = min(hold_a_target, stop_accel)
      e2e_source = bool(is_e2e and not model_stale and raw_hold < hold_a_target)

      # Standstill stop-hold command normalization: clamp harsh hold commands up to a
      # local mild hold target when already stopped, avoiding an artificial jump to the
      # first positive release. Does not delay braking or affect rolling stops.
      controls_state_sp = self._sm_item(sm, 'controlsState')
      brake_pressed_sp = bool(getattr(car_state, 'brakePressed', False)) if car_state is not None else False
      force_decel_sp = bool(getattr(controls_state_sp, 'forceDecel', False)) if controls_state_sp is not None else False
      lead_id_sp = getattr(selected_lead, 'radarTrackId', None) if selected_lead is not None else None
      same_id_sp = lead_id_sp is not None and self._lead_stop_hold_lead_id is not None and lead_id_sp == self._lead_stop_hold_lead_id
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
        sm, raw_hold, selected_lead, lead_d_rel, lead_v, lead_v_rel,
        mpc_a_target, raw_model_a_target, raw_model_should_stop,
      )
      if self.custom_long_output is not None:
        self._custom_long_output_telemetry = replace(self.custom_long_output, should_stop=True, selected_intent="lead_stop_hold", reason="stopped_lead_latch")
      return float(a_target), True, e2e_source
    if is_e2e and not model_stale:
      a_target = min(raw_model_a_target, release_a_target if release_mpc_stop else mpc_a_target)
      e2e_source = bool(a_target < mpc_a_target)
      a_target = self._apply_stop_hold_release_slew(sm, a_target, release_mpc_stop, mpc_stop, raw_model_should_stop, should_stop)
      return float(a_target), should_stop, e2e_source
    a_target = float(release_a_target if release_mpc_stop else mpc_a_target)
    a_target = self._scc_custom_stop_cap(a_target)
    a_target = self._scc_curve_confidence_final_cap(a_target, sm, release_mpc_stop=release_mpc_stop)
    a_target = self._apply_stop_hold_release_slew(sm, a_target, release_mpc_stop, mpc_stop, raw_model_should_stop, should_stop)
    return a_target, bool(should_stop), False

  def update(self, sm: messaging.SubMaster) -> None:
    self.custom_long.refresh_params(mode_only=True)  # enabled/mode every tick; tuning on slow cadence via update_targets
    self.events_sp.clear()
    if not self.custom_long.enabled:   # custom SCC mode replaces DEC; keep DEC dormant when custom is on
      self.dec.update(sm)
    self.e2e_alerts_helper.update(sm, self.events_sp)
    controls_state = self._sm_item(sm, 'controlsState')
    car_state = self._sm_item(sm, 'carState')
    selfdrive_state = self._sm_item(sm, 'selfdriveState')
    long_control_off = bool(getattr(controls_state, 'longControlState', None) == LongCtrlState.off)
    v_cruise_initialized = True
    if car_state is not None:
      v_cruise_initialized = float(getattr(car_state, 'vCruise', V_CRUISE_UNSET)) != V_CRUISE_UNSET
    reset_state = long_control_off if getattr(self.CP, 'openpilotLongitudinalControl', False) else not bool(getattr(selfdrive_state, 'enabled', False))
    reset_state = reset_state or not v_cruise_initialized
    if reset_state:
      self._reset_lead_stop_hold()

  def publish_longitudinal_plan_sp(self, sm: messaging.SubMaster, pm: messaging.PubMaster) -> None:
    plan_sp_send = messaging.new_message('longitudinalPlanSP')

    plan_sp_send.valid = sm.all_checks(service_list=['carState', 'controlsState'])

    longitudinalPlanSP = plan_sp_send.longitudinalPlanSP
    longitudinalPlanSP.longitudinalPlanSource = self.source
    longitudinalPlanSP.vTarget = float(self.output_v_target)
    longitudinalPlanSP.aTarget = float(self.output_a_target)
    longitudinalPlanSP.events = self.events_sp.to_msg()

    # Dynamic Experimental Control compatibility alias. Custom longitudinal owns mode selection
    # when enabled, so DEC is reported inactive to avoid implying legacy DEC is driving actuation.
    dec = longitudinalPlanSP.dec
    if self.custom_long.enabled:
      dec.state = DecState.blended if self.custom_long.mode is LongitudinalMode.SCC else DecState.acc
      dec.enabled = False
      dec.active = False
    else:
      dec.state = DecState.blended if self.dec.mode() == 'blended' else DecState.acc
      dec.enabled = self.dec.enabled()
      dec.active = self.dec.active()

    # Smart Cruise Control
    smartCruiseControl = longitudinalPlanSP.smartCruiseControl
    # Vision Control
    sccVision = smartCruiseControl.vision
    sccVision.state = self.scc.vision.state
    sccVision.vTarget = float(self.scc.vision.output_v_target)
    sccVision.aTarget = float(self.scc.vision.output_a_target)
    sccVision.currentLateralAccel = float(self.scc.vision.current_lat_acc)
    sccVision.maxPredictedLateralAccel = float(self.scc.vision.max_pred_lat_acc)
    sccVision.enabled = self.scc.vision.is_enabled
    sccVision.active = self.scc.vision.is_active
    # Map Control
    sccMap = smartCruiseControl.map
    sccMap.state = self.scc.map.state
    sccMap.vTarget = float(self.scc.map.output_v_target)
    sccMap.aTarget = float(self.scc.map.output_a_target)
    sccMap.enabled = self.scc.map.is_enabled
    sccMap.active = self.scc.map.is_active

    # Speed Limit
    speedLimit = longitudinalPlanSP.speedLimit
    resolver = speedLimit.resolver
    resolver.speedLimit = float(self.resolver.speed_limit)
    resolver.speedLimitLast = float(self.resolver.speed_limit_last)
    resolver.speedLimitFinal = float(self.resolver.speed_limit_final)
    resolver.speedLimitFinalLast = float(self.resolver.speed_limit_final_last)
    resolver.speedLimitValid = self.resolver.speed_limit_valid
    resolver.speedLimitLastValid = self.resolver.speed_limit_last_valid
    resolver.speedLimitOffset = float(self.resolver.speed_limit_offset)
    resolver.distToSpeedLimit = float(self.resolver.distance)
    resolver.source = self.resolver.source
    assist = speedLimit.assist
    assist.state = self.sla.state
    assist.enabled = self.sla.is_enabled
    assist.active = self.sla.is_active
    assist.vTarget = float(self.sla.output_v_target)
    assist.aTarget = float(self.sla.output_a_target)

    # E2E Alerts
    e2eAlerts = longitudinalPlanSP.e2eAlerts
    e2eAlerts.greenLightAlert = self.e2e_alerts_helper.green_light_alert
    e2eAlerts.leadDepartAlert = self.e2e_alerts_helper.lead_depart_alert

    custom_long = longitudinalPlanSP.customLongitudinal
    custom_long.enabled = bool(self.custom_long.enabled)
    telemetry_custom_long_output = self._custom_long_output_telemetry if self._custom_long_output_telemetry is not None else self.custom_long_output
    custom_long.active = bool(telemetry_custom_long_output.enabled) if telemetry_custom_long_output is not None else bool(self.custom_long.enabled)
    custom_long.shouldStop = bool(telemetry_custom_long_output.should_stop) if telemetry_custom_long_output is not None else False
    custom_long.mode = self._custom_longitudinal_mode_to_telemetry()
    custom_long.selectedIntent = str(getattr(telemetry_custom_long_output, "selected_intent", "" ) or "")
    custom_long.reason = str(getattr(telemetry_custom_long_output, "reason", "" ) or "")
    if getattr(self.custom_long, "debug_trace_mode", "off") == "log":
      self._populate_longitudinal_debug_trace(longitudinalPlanSP.longitudinalDebug, sm, telemetry_custom_long_output)
    self._custom_long_output_telemetry = None
    self._last_release_block_reason = ""

    pm.send('longitudinalPlanSP', plan_sp_send)

  def _custom_longitudinal_mode_to_telemetry(self):
    if self.custom_long.mode is LongitudinalMode.ACC:
      return custom.LongitudinalPlanSP.CustomLongitudinal.CustomLongitudinalMode.acc
    if self.custom_long.mode is LongitudinalMode.E2E:
      return custom.LongitudinalPlanSP.CustomLongitudinal.CustomLongitudinalMode.e2e
    return custom.LongitudinalPlanSP.CustomLongitudinal.CustomLongitudinalMode.scc

  def _populate_longitudinal_debug_trace(self, msg, sm: messaging.SubMaster, custom_long_output) -> None:
    try:
      trace = dict(getattr(self, '_last_longitudinal_debug', {}) or {})
      debug = dict(getattr(custom_long_output, 'debug', {}) or {})
      # Inject planner-level release block reason into the standstill_release_confidence trace
      release_block_reason = str(getattr(self, '_last_release_block_reason', '') or '')
      if release_block_reason:
        debug['standstill_release_confidence_block_reason'] = release_block_reason
      msg.enabled = True
      msg.traceMode = str(getattr(self.custom_long, "debug_trace_mode", "off"))
      car_state = self._sm_item(sm, 'carState')
      msg.vEgo = self._safe_float(getattr(car_state, 'vEgo', 0.0) if car_state is not None else 0.0)
      msg.vCruise = self._safe_float(trace.get('v_cruise', 0.0))
      msg.customATarget = self._safe_float(getattr(custom_long_output, 'a_target', 0.0))
      msg.customShouldStop = bool(getattr(custom_long_output, 'should_stop', False))
      msg.customIntent = str(getattr(custom_long_output, 'selected_intent', '') or '')
      msg.customReason = str(getattr(custom_long_output, 'reason', '') or '')
      msg.mpcATarget = self._safe_float(trace.get('mpc_a_target', 0.0))
      msg.mpcShouldStop = bool(trace.get('mpc_should_stop', False))
      msg.modelATarget = self._safe_float(trace.get('model_a_target', 0.0))
      msg.modelShouldStop = bool(trace.get('model_should_stop', False))
      msg.finalATargetUnclipped = self._safe_float(trace.get('final_a_target_unclipped', 0.0))
      msg.finalATargetClipped = self._safe_float(trace.get('final_a_target_clipped', 0.0))
      msg.finalShouldStop = bool(trace.get('final_should_stop', False))
      msg.accelClipMin = self._safe_float(trace.get('accel_clip_min', 0.0))
      msg.accelClipMax = self._safe_float(trace.get('accel_clip_max', 0.0))
      msg.e2eSource = bool(trace.get('e2e_source', False))
      lpc = msg.leadPathClearance
      prefix = 'lead_path_clearance_'
      lpc.mode = str(debug.get(prefix + 'mode', 'off') or 'off')
      lpc.effectiveMode = str(debug.get(prefix + 'effective_mode', '') or ('shadow' if lpc.mode == 'apply' else lpc.mode))
      lpc.applySupported = bool(debug.get(prefix + 'apply_supported', False))
      lpc.shadowEligible = bool(debug.get(prefix + 'shadow_eligible', debug.get(prefix + 'eligible', False)))
      lpc.blockReason = str(debug.get(prefix + 'shadow_blocked_reason', debug.get(prefix + 'block_reason', '')) or '')
      lpc.leadIdx = int(self._safe_float(debug.get(prefix + 'lead_idx', -1), -1))
      lpc.pathYRel = self._safe_float(debug.get(prefix + 'path_y_rel', 0.0))
      lpc.lateralVelocity = self._safe_float(debug.get(prefix + 'lateral_velocity', 0.0))
      lpc.tClear = self._safe_float(debug.get(prefix + 't_clear', 0.0))
      lpc.tConflict = self._safe_float(debug.get(prefix + 't_conflict', 0.0))
      lpc.confidence = self._safe_float(debug.get(prefix + 'confidence', 0.0))
      lpc.modelProb = self._safe_float(debug.get(prefix + 'model_prob', 0.0))
      lpc.ttc = self._safe_float(debug.get(prefix + 'ttc', 0.0))
      lpc.requiredDecel = self._safe_float(debug.get(prefix + 'required_decel', 0.0))
      lpc.leadStatus = bool(debug.get('actual_primary_lead_authority', ''))
      lpc.leadDRel = self._safe_float(debug.get('actual_primary_lead_d_rel', 0.0))
      lpc.leadVRel = self._safe_float(debug.get('actual_primary_lead_v_rel', 0.0))
      lpc.leadYRel = self._safe_float(debug.get('actual_primary_lead_y_rel', 0.0))
      self._populate_cut_in_brake_assist_trace(msg.cutInBrakeAssist, debug)
      self._populate_curve_speed_confidence_trace(msg.curveSpeedConfidence, debug)
      self._populate_standstill_release_confidence_trace(msg.standstillReleaseConfidence, debug)
      self._populate_acc_envelope_trace(msg.accEnvelope, debug)
    except Exception:
      msg.enabled = False
      msg.traceMode = 'off'

  def _populate_feature_trace_common(self, msg, debug: dict, prefix: str) -> None:
    msg.mode = str(debug.get(prefix + 'mode', 'off') or 'off')
    msg.effectiveMode = str(debug.get(prefix + 'effective_mode', msg.mode) or msg.mode)
    msg.applySupported = bool(debug.get(prefix + 'apply_supported', False))
    msg.eligible = bool(debug.get(prefix + 'eligible', False))
    msg.blockReason = str(debug.get(prefix + 'block_reason', '') or '')

  def _populate_cut_in_brake_assist_trace(self, msg, debug: dict) -> None:
    prefix = 'cut_in_brake_assist_'
    self._populate_feature_trace_common(msg, debug, prefix)
    msg.leadIdx = int(self._safe_float(debug.get(prefix + 'lead_idx', -1), -1))
    msg.pathYRel = self._safe_float(debug.get(prefix + 'path_y_rel', 0.0))
    msg.lateralVelocity = self._safe_float(debug.get(prefix + 'lateral_velocity', 0.0))
    msg.ttc = self._safe_float(debug.get(prefix + 'ttc', 0.0))
    msg.requiredDecel = self._safe_float(debug.get(prefix + 'required_decel', 0.0))
    msg.proposedCap = self._safe_float(debug.get(prefix + 'proposed_cap', 0.0))
    msg.confidence = self._safe_float(debug.get(prefix + 'confidence', 0.0))

  def _populate_curve_speed_confidence_trace(self, msg, debug: dict) -> None:
    prefix = 'curve_speed_confidence_'
    self._populate_feature_trace_common(msg, debug, prefix)
    msg.confidence = self._safe_float(debug.get(prefix + 'confidence', 0.0))
    msg.proposedCap = self._safe_float(debug.get(prefix + 'proposed_cap', 0.0))
    msg.source = str(debug.get(prefix + 'source', '') or '')
    msg.active = bool(debug.get(prefix + 'active', False))
    msg.currentLatAccel = self._safe_float(debug.get(prefix + 'current_lat_acc', 0.0))
    msg.maxPredLatAccel = self._safe_float(debug.get(prefix + 'max_pred_lat_acc', 0.0))
    msg.preEntryActive = bool(debug.get(prefix + 'pre_entry_active', False))

  def _populate_standstill_release_confidence_trace(self, msg, debug: dict) -> None:
    prefix = 'standstill_release_confidence_'
    self._populate_feature_trace_common(msg, debug, prefix)
    msg.confidence = self._safe_float(debug.get(prefix + 'confidence', 0.0))
    msg.releaseAllowed = bool(debug.get(prefix + 'release_allowed', False))
    msg.releaseSource = str(debug.get(prefix + 'release_source', '') or '')
    msg.releaseReason = str(debug.get(prefix + 'release_reason', '') or '')
    msg.releaseATarget = self._safe_float(debug.get(prefix + 'release_a_target', 0.0))

  def _populate_acc_envelope_trace(self, msg, debug: dict) -> None:
    prefix = 'acc_envelope_'
    msg.active = bool(debug.get(prefix + 'active', False))
    msg.wouldCap = bool(debug.get(prefix + 'would_cap', False))
    msg.capReason = str(debug.get(prefix + 'cap_reason', '') or '')
    msg.allowedATarget = self._safe_float(debug.get(prefix + 'allowed_a_target', 0.0))
    msg.deltaA = self._safe_float(debug.get(prefix + 'delta_a', 0.0))
    msg.desiredGap = self._safe_float(debug.get(prefix + 'desired_gap', 0.0))
    msg.timeGap = self._safe_float(debug.get(prefix + 'time_gap', 0.0))
    msg.ttc = self._safe_float(debug.get(prefix + 'ttc', 0.0))
    msg.usableStoppingGap = self._safe_float(debug.get(prefix + 'usable_stopping_gap', 0.0))
    msg.requiredStoppingDecel = self._safe_float(debug.get(prefix + 'required_stopping_decel', 0.0))
    msg.closingSpeedDecel = self._safe_float(debug.get(prefix + 'closing_speed_decel', 0.0))
    msg.jerkLimitedATarget = self._safe_float(debug.get(prefix + 'jerk_limited_a_target', 0.0))

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
