"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

from cereal import messaging, custom
from opendbc.car import structs
from openpilot.common.constants import CV
from openpilot.selfdrive.car.cruise import V_CRUISE_MAX
from openpilot.selfdrive.controls.lib.longitudinal_decision import CandidateRole, DecisionSource, LongitudinalCandidate
from openpilot.sunnypilot.selfdrive.controls.lib.dec.dec import DynamicExperimentalController
from openpilot.sunnypilot.selfdrive.controls.lib.e2e_alerts_helper import E2EAlertsHelper
from openpilot.sunnypilot.selfdrive.controls.lib.smart_cruise_control.smart_cruise_control import SmartCruiseControl
from openpilot.sunnypilot.selfdrive.controls.lib.osm_traffic_control_prior import OsmTrafficControlPrior
from openpilot.sunnypilot.selfdrive.controls.lib.speed_limit.speed_limit_assist import SpeedLimitAssist, V_CRUISE_UNSET
from openpilot.sunnypilot.selfdrive.controls.lib.speed_limit.speed_limit_resolver import SpeedLimitResolver
from openpilot.sunnypilot.selfdrive.selfdrived.events import EventsSP
from openpilot.sunnypilot.models.helpers import get_active_bundle

DecState = custom.LongitudinalPlanSP.DynamicExperimentalControl.DynamicExperimentalControlState
LongitudinalPlanSource = custom.LongitudinalPlanSP.LongitudinalPlanSource
SPEED_LIMIT_HANDOFF_EXIT_MARGIN = 0.25  # m/s, near enough to manual cruise to return to cruise.
SPEED_LIMIT_HANDOFF_A_TARGET_MAX = 0.0  # m/s^2, coast instead of accelerating during handoff.
SPEED_LIMIT_SPEED_UP_ACCEL_CAP = 0.8  # m/s^2, driver-intent candidate speed-up governor.
SPEED_LIMIT_SPEED_UP_LOOKAHEAD = 2.0  # s, short candidate horizon for active speed-limit increases.
LEAD_SPEEDUP_GUARD_TIME_GAP = 2.2  # s, match the observed uncomfortable closing window.
LEAD_SPEEDUP_GUARD_MIN_DISTANCE = 25.0  # m, low-speed floor for close-lead gating.
LEAD_SPEEDUP_GUARD_CLOSING_V_REL = -0.2  # m/s, ignore noise around matched speed.
LEAD_SPEEDUP_GUARD_A_TARGET_MAX = 0.0  # m/s^2, coast instead of accelerating into the lead.
LEAD_SPEEDUP_GUARD_LATERAL_EXIT_Y_REL = 1.6
SOURCE_SELECTION_HYSTERESIS_V = 0.25


def _select_lower_target(selected_source, selected_v_target, selected_a_target, candidate_source, candidate):
  candidate_v_target, candidate_a_target = candidate
  if candidate_v_target < selected_v_target:
    return candidate_source, candidate_v_target, candidate_a_target
  return selected_source, selected_v_target, selected_a_target


def should_block_lead_speedup(v_ego: float, lead_status: bool, d_rel: float, v_rel: float, y_rel: float,
                              gas_pressed: bool, brake_pressed: bool) -> bool:
  if not lead_status or gas_pressed or brake_pressed:
    return False
  if abs(y_rel) >= LEAD_SPEEDUP_GUARD_LATERAL_EXIT_Y_REL:
    return False
  if v_rel > LEAD_SPEEDUP_GUARD_CLOSING_V_REL:
    return False

  close_distance = max(LEAD_SPEEDUP_GUARD_MIN_DISTANCE, v_ego * LEAD_SPEEDUP_GUARD_TIME_GAP)
  return d_rel < close_distance


def apply_lead_speedup_guard(active: bool, v_ego: float, target: tuple[float, float]) -> tuple[float, float]:
  if not active:
    return target

  v_target, a_target = target
  return min(v_target, v_ego), min(a_target, LEAD_SPEEDUP_GUARD_A_TARGET_MAX)


def select_lowest_longitudinal_target(speed_limit_active, cruise, scc_vision, scc_map, speed_limit_assist, osm_traffic_control,
                                      source_prev=None, v_target_prev=None):
  if speed_limit_active:
    selected_source = LongitudinalPlanSource.speedLimitAssist
    selected_v_target = speed_limit_assist[0]
    selected_a_target = cruise[1]
  else:
    selected_source = LongitudinalPlanSource.cruise
    selected_v_target, selected_a_target = cruise

  selected_source, selected_v_target, selected_a_target = _select_lower_target(
    selected_source, selected_v_target, selected_a_target, LongitudinalPlanSource.sccVision, scc_vision
  )

  selected_source, selected_v_target, selected_a_target = _select_lower_target(
    selected_source, selected_v_target, selected_a_target, LongitudinalPlanSource.sccMap, scc_map
  )
  if not speed_limit_active:
    selected_source, selected_v_target, selected_a_target = _select_lower_target(
      selected_source, selected_v_target, selected_a_target, LongitudinalPlanSource.speedLimitAssist, speed_limit_assist
    )
  selected_source, selected_v_target, selected_a_target = _select_lower_target(
    selected_source, selected_v_target, selected_a_target, LongitudinalPlanSource.osmTrafficControl, osm_traffic_control
  )

  if source_prev is not None and v_target_prev is not None:
    # Always allow switching to a more restrictive (slower) target for safety.
    if selected_v_target < v_target_prev:
      return selected_source, selected_v_target, selected_a_target

    # Only switch to a less restrictive (faster) target if it's significantly faster
    # OR if it's the same target but a different source (allows state machine transitions).
    if selected_v_target > v_target_prev + SOURCE_SELECTION_HYSTERESIS_V or \
       (abs(selected_v_target - v_target_prev) < 1e-4 and selected_source != source_prev):
      return selected_source, selected_v_target, selected_a_target
    else:
      return source_prev, v_target_prev, selected_a_target

  return selected_source, selected_v_target, selected_a_target


def build_sp_longitudinal_candidates(speed_limit_active, cruise, scc_vision, scc_vision_active, scc_map, scc_map_active,
                                     speed_limit_assist, osm_traffic_control, osm_traffic_control_active):
  cruise_v, cruise_a = cruise
  candidates = [LongitudinalCandidate(
    source=DecisionSource.CRUISE,
    role=CandidateRole.DRIVER_INTENT,
    v_target=cruise_v,
    a_target=cruise_a,
    confidence=1.0,
    urgency=0.1,
    active_reason="driver_cruise_target",
  )]

  if speed_limit_active:
    candidates.append(LongitudinalCandidate(
      source=DecisionSource.SPEED_LIMIT,
      role=CandidateRole.ADVISORY_CAP,
      v_target=speed_limit_assist[0],
      a_target=speed_limit_assist[1],
      confidence=0.85,
      urgency=0.35,
      active_reason="speed_limit_assist_active",
    ))
  if scc_vision_active:
    candidates.append(LongitudinalCandidate(
      source=DecisionSource.SCC_VISION,
      role=CandidateRole.ADVISORY_CAP,
      v_target=scc_vision[0],
      a_target=scc_vision[1],
      confidence=0.80,
      urgency=0.45,
      active_reason="confident_vision_curve",
    ))
  if scc_map_active:
    candidates.append(LongitudinalCandidate(
      source=DecisionSource.SCC_MAP,
      role=CandidateRole.ADVISORY_CAP,
      v_target=scc_map[0],
      a_target=scc_map[1],
      confidence=0.80,
      urgency=0.40,
      active_reason="confident_map_curve",
    ))
  if osm_traffic_control_active:
    candidates.append(LongitudinalCandidate(
      source=DecisionSource.OSM_TRAFFIC_CONTROL,
      role=CandidateRole.ADVISORY_CAP,
      v_target=osm_traffic_control[0],
      a_target=osm_traffic_control[1],
      confidence=0.75,
      urgency=0.55,
      active_reason="model_confirmed_map_caution",
    ))

  return candidates


class LongitudinalPlannerSP:
  def __init__(self, CP: structs.CarParams, CP_SP: structs.CarParamsSP, mpc):
    self.events_sp = EventsSP()
    self.resolver = SpeedLimitResolver()
    self.dec = DynamicExperimentalController(CP, mpc)
    self.scc = SmartCruiseControl()
    self.osm_traffic_control_prior = OsmTrafficControlPrior()
    self.resolver = SpeedLimitResolver()
    self.sla = SpeedLimitAssist(CP, CP_SP)
    self.generation = int(model_bundle.generation) if (model_bundle := get_active_bundle()) else None
    self.source = LongitudinalPlanSource.cruise
    self.e2e_alerts_helper = E2EAlertsHelper()

    self.output_v_target = 0.
    self.output_a_target = 0.
    self.decision_candidates_sp = []
    self._speed_limit_handoff_active = False
    self._speed_limit_active_prev = False

  def is_e2e(self, sm: messaging.SubMaster) -> bool:
    experimental_mode = sm['selfdriveState'].experimentalMode
    if not self.dec.active():
      return experimental_mode

    return experimental_mode and self.dec.mode() == "blended"

  def _update_speed_limit_handoff(self, long_enabled: bool, long_override: bool, v_ego: float, v_cruise: float) -> bool:
    has_limit_target = (self.resolver.speed_limit_valid or self.resolver.speed_limit_last_valid) and \
                       self.resolver.speed_limit_final_last > 0. and \
                       self.resolver.speed_limit_final_last != V_CRUISE_UNSET
    manual_cruise_below_limit = v_cruise < self.resolver.speed_limit_final_last
    above_manual_cruise = v_ego > v_cruise + SPEED_LIMIT_HANDOFF_EXIT_MARGIN

    if self.sla.is_active:
      self._speed_limit_handoff_active = False
    elif (self._speed_limit_active_prev and long_enabled and not long_override and has_limit_target and
          manual_cruise_below_limit and above_manual_cruise):
      self._speed_limit_handoff_active = True
    elif not long_enabled or long_override or not has_limit_target or not manual_cruise_below_limit or not above_manual_cruise:
      self._speed_limit_handoff_active = False

    self._speed_limit_active_prev = self.sla.is_active
    return self._speed_limit_handoff_active

  def _speed_limit_handoff_target(self, v_ego: float, a_ego: float) -> tuple[float, float]:
    return min(v_ego, self.resolver.speed_limit_final_last), min(a_ego, SPEED_LIMIT_HANDOFF_A_TARGET_MAX)

  def update_targets(self, sm: messaging.SubMaster, v_ego: float, a_ego: float, v_cruise: float,
                     coast_accel: float | None = None) -> tuple[float, float]:
    CS = sm['carState']
    v_cruise_cluster_kph = min(CS.vCruiseCluster, V_CRUISE_MAX)
    v_cruise_cluster = v_cruise_cluster_kph * CV.KPH_TO_MS

    long_enabled = sm['carControl'].enabled
    long_override = sm['carControl'].cruiseControl.override

    # Smart Cruise Control
    self.scc.update(sm, long_enabled, long_override, v_ego, a_ego, v_cruise)

    # Speed Limit Resolver
    self.resolver.update(v_ego, sm, coast_accel=coast_accel)

    # Speed Limit Assist
    has_speed_limit = self.resolver.speed_limit_valid or self.resolver.speed_limit_last_valid
    self.sla.update(long_enabled, long_override, v_ego, a_ego, v_cruise_cluster, self.resolver.speed_limit,
                    self.resolver.speed_limit_final_last, has_speed_limit, self.resolver.distance, self.events_sp,
                    coast_accel=coast_accel)

    self.osm_traffic_control_prior.update(sm, long_enabled, long_override, v_ego, a_ego)

    speed_limit_handoff_active = self._update_speed_limit_handoff(long_enabled, long_override, v_ego, v_cruise)
    speed_limit_active = self.sla.is_active or speed_limit_handoff_active
    speed_limit_assist_target = self._speed_limit_handoff_target(v_ego, a_ego) if speed_limit_handoff_active else (
      self.sla.output_v_target,
      a_ego,
    )
    cruise_target = (v_cruise, min(a_ego, SPEED_LIMIT_HANDOFF_A_TARGET_MAX)) if speed_limit_handoff_active else (v_cruise, a_ego)
    lead_one = sm['radarState'].leadOne
    lead_speedup_guard_active = should_block_lead_speedup(
      v_ego,
      bool(lead_one.status),
      float(lead_one.dRel),
      float(lead_one.vRel),
      float(lead_one.yRel),
      bool(CS.gasPressed),
      bool(CS.brakePressed),
    )
    speed_limit_assist_target = apply_lead_speedup_guard(lead_speedup_guard_active, v_ego, speed_limit_assist_target)
    cruise_target = apply_lead_speedup_guard(lead_speedup_guard_active, v_ego, cruise_target)
    decision_cruise_target = speed_limit_assist_target if speed_limit_active else cruise_target
    if speed_limit_active:
      decision_cruise_target = (
        min(decision_cruise_target[0], v_ego + SPEED_LIMIT_SPEED_UP_ACCEL_CAP * SPEED_LIMIT_SPEED_UP_LOOKAHEAD),
        decision_cruise_target[1],
      )
    decision_speed_limit_active = False  # Active SLA is represented as effective driver intent below.

    self.decision_candidates_sp = build_sp_longitudinal_candidates(
      decision_speed_limit_active,
      decision_cruise_target,
      (self.scc.vision.output_v_target, self.scc.vision.output_a_target),
      self.scc.vision.is_active,
      (self.scc.map.output_v_target, self.scc.map.output_a_target),
      self.scc.map.is_active,
      speed_limit_assist_target,
      (self.osm_traffic_control_prior.output_v_target, self.osm_traffic_control_prior.output_a_target),
      self.osm_traffic_control_prior.active,
    )

    self.source, self.output_v_target, self.output_a_target = select_lowest_longitudinal_target(
      speed_limit_active,
      cruise_target,
      (self.scc.vision.output_v_target, self.scc.vision.output_a_target),
      (self.scc.map.output_v_target, self.scc.map.output_a_target),
      speed_limit_assist_target,
      (self.osm_traffic_control_prior.output_v_target, self.osm_traffic_control_prior.output_a_target),
      source_prev=getattr(self, 'source', None),
      v_target_prev=getattr(self, 'output_v_target', None),
    )
    return self.output_v_target, self.output_a_target

  def update(self, sm: messaging.SubMaster) -> None:
    self.events_sp.clear()
    self.dec.update(sm)
    self.e2e_alerts_helper.update(sm, self.events_sp)

  def publish_longitudinal_plan_sp(self, sm: messaging.SubMaster, pm: messaging.PubMaster) -> None:
    plan_sp_send = messaging.new_message('longitudinalPlanSP')

    plan_sp_send.valid = sm.all_checks(service_list=['carState', 'controlsState'])

    longitudinalPlanSP = plan_sp_send.longitudinalPlanSP
    longitudinalPlanSP.longitudinalPlanSource = self.source
    longitudinalPlanSP.vTarget = float(self.output_v_target)
    longitudinalPlanSP.aTarget = float(self.output_a_target)
    longitudinalPlanSP.events = self.events_sp.to_msg()

    # Dynamic Experimental Control
    dec = longitudinalPlanSP.dec
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
    assist.autoCruiseEnabled = self.sla.auto_enabled
    assist.vTarget = float(self.sla.output_v_target)
    assist.aTarget = float(self.sla.output_a_target)

    # E2E Alerts
    e2eAlerts = longitudinalPlanSP.e2eAlerts
    e2eAlerts.greenLightAlert = self.e2e_alerts_helper.green_light_alert
    e2eAlerts.leadDepartAlert = self.e2e_alerts_helper.lead_depart_alert

    pm.send('longitudinalPlanSP', plan_sp_send)
