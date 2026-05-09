#!/usr/bin/env python3
import os
import time
import numpy as np
from cereal import log
from opendbc.car.interfaces import ACCEL_MIN, ACCEL_MAX
from openpilot.common.realtime import DT_MDL
from openpilot.common.swaglog import cloudlog

# WARNING: imports outside of constants will not trigger a rebuild
from openpilot.selfdrive.modeld.constants import index_function
from openpilot.selfdrive.controls.radard import _LEAD_ACCEL_TAU

if __name__ == '__main__':  # generating code
  from openpilot.third_party.acados.acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
else:
  from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.c_generated_code.acados_ocp_solver_pyx import AcadosOcpSolverCython

from casadi import SX, vertcat

MODEL_NAME = 'long'
LONG_MPC_DIR = os.path.dirname(os.path.abspath(__file__))
EXPORT_DIR = os.path.join(LONG_MPC_DIR, "c_generated_code")
JSON_FILE = os.path.join(LONG_MPC_DIR, "acados_ocp_long.json")

LongitudinalPlanSource = log.LongitudinalPlan.LongitudinalPlanSource
MPC_SOURCES = (LongitudinalPlanSource.lead0, LongitudinalPlanSource.lead1, LongitudinalPlanSource.cruise)

X_DIM = 3
U_DIM = 1
PARAM_DIM = 6
COST_E_DIM = 5
COST_DIM = COST_E_DIM + 1
CONSTR_DIM = 4

X_EGO_OBSTACLE_COST = 3.0
X_EGO_COST = 0.0
V_EGO_COST = 0.0
A_EGO_COST = 0.0
J_EGO_COST = 5.0
A_CHANGE_COST = 200.0
DANGER_ZONE_COST = 100.0
CRASH_DISTANCE = 0.25
LEAD_DANGER_FACTOR = 0.75
LIMIT_COST = 1e6
ACADOS_SOLVER_TYPE = 'SQP_RTI'

# Fewer timestamps don't hurt performance and lead to
# much better convergence of the MPC with low iterations
N = 12
MAX_T = 10.0
T_IDXS_LST = [index_function(idx, max_val=MAX_T, max_idx=N) for idx in range(N + 1)]

T_IDXS = np.array(T_IDXS_LST)
FCW_IDXS = T_IDXS < 5.0
T_DIFFS = np.diff(T_IDXS, prepend=[0.0])
COMFORT_BRAKE = 2.5
STOP_DISTANCE = 5.0
LEAD_STOP_PRESENTATION_DISTANCE = 5.0
LEAD_STOP_PRESENTATION_CONFIDENCE_MIN = 0.75
LEAD_STOP_PRESENTATION_V_EGO_BP = [0.0, 3.0]
LEAD_STOP_PRESENTATION_V_LEAD_BP = [0.2, 1.0]
LEAD_STOP_PRESENTATION_DECEL_BP = [0.0, 0.6]
STOP_DISTANCE_FADE_V = 3.0
STOP_DISTANCE_MIN = 2.0
STOPPED_LEAD_BUFFER = 1.75
STOPPED_LEAD_V_EGO_BP = [0.0, 0.5, 1.5]
STOPPED_LEAD_V_LEAD_BP = [0.0, 1.0, 2.0]
LEAD_STOP_GAP_TAPER_MAX = 1.0
LEAD_STOP_GAP_TAPER_V_EGO_BP = [0.0, 1.5]
LEAD_STOP_GAP_TAPER_V_LEAD_BP = [0.0, 0.6, 2.0]
LEAD_STOP_GAP_EXCESS_OFFSET_MAX = 4.0
LEAD_STOP_GAP_EXCESS_V_EGO_BP = [0.0, 1.5]
LEAD_STOP_GAP_EXCESS_BP = [1.0, 5.0]
LEAD_DEPARTURE_RELAXATION_MAX = 2.0
LEAD_DEPARTURE_ARM_V_EGO = 0.1
LEAD_DEPARTURE_V_EGO_BP = [0.0, 1.0]
LEAD_DEPARTURE_V_LEAD_BP = [0.6, 2.0]
LEAD_DEPARTURE_V_REL_BP = [0.2, 1.0]
LEAD_DEPARTURE_GAP_OPENING_BP = [0.3, 1.0]
APPROACH_BRAKE = 3.0
APPROACH_BRAKE_MIN = 2.5
APPROACH_BRAKE_CLOSING_BP = [1.5, 5.0]
APPROACH_MIN_GAP_BUFFER = 2.0
APPROACH_DECEL_BLEND_BP = [0.5, 2.0]
APPROACH_STOP_RUNWAY_DECEL_BP = [0.3, 1.0]
APPROACH_RUNWAY_BLEND_BP = [5.0, 20.0]
APPROACH_ENGAGE_OFFSET_MAX = 8.0
APPROACH_ENGAGE_CLOSING_BP = [3.0, 12.0]
APPROACH_ENGAGE_RUNWAY_BP = [25.0, 80.0]
LEAD_STOP_RUNWAY_V_EGO_BP = [2.0, 5.0]
LEAD_STOP_RUNWAY_V_LEAD_BP = [0.2, 1.0]
LEAD_STOP_RUNWAY_DECEL_BP = [0.1, 0.6]
LEAD_STOP_RUNWAY_MOVING_V_LEAD_BP = [0.1, 1.5]
LEAD_STOP_RUNWAY_BRAKE = 0.6
LEAD_STOP_RUNWAY_URGENCY_DECEL_BP = [LEAD_STOP_RUNWAY_BRAKE, 2.0]
LEAD_STOP_RUNWAY_URGENCY_CLOSING_BP = [0.5, 2.0]
LEAD_STOP_RUNWAY_URGENCY_DANGER_MARGIN = 2.0
LEAD_STOP_RUNWAY_STOPPED_BUFFER_FADE = 0.25
LEAD_CRAWL_V_EGO_BP = [6.0, 8.0]
LEAD_CRAWL_V_LEAD_BP = [0.2, 1.0]
LEAD_CRAWL_GAP_BP = [STOP_DISTANCE + 0.3, STOP_DISTANCE + 4.0, STOP_DISTANCE + 4.5]
LEAD_CRAWL_BRAKE_GAP_BP = [STOP_DISTANCE + 0.2, STOP_DISTANCE + 1.0, STOP_DISTANCE + 4.0]
LEAD_CRAWL_ACCEL_LIMIT_GAP_BP = [STOP_DISTANCE + 0.3, STOP_DISTANCE + 4.0, 20.0, 25.0]
LEAD_CRAWL_OPENING_BP = [0.2, 1.2]
LEAD_CRAWL_CLOSING_BP = [0.1, 1.0]
LEAD_CRAWL_DECEL_BP = [0.1, 1.0]
LEAD_CRAWL_REQUIRED_DECEL_BP = [0.15, LEAD_STOP_RUNWAY_BRAKE]
LEAD_CRAWL_ACCEL_MAX = 0.6
LEAD_CRAWL_ACCEL_LIMIT = 0.75
LEAD_CRAWL_BRAKE_MAX = 0.75
LEAD_CRAWL_COST = 1.2
LEAD_SURGE_DAMPING_V_EGO_BP = [0.5, 1.5, 6.0, 8.0]
LEAD_SURGE_DAMPING_V_LEAD_BP = [0.5, 1.0]
LEAD_SURGE_DAMPING_GAP_EXCESS_BP = [0.3, 2.0]
LEAD_SURGE_DAMPING_OPENING_BP = [0.2, 1.2]
LEAD_SURGE_DAMPING_DECEL_BP = [0.2, 0.8]
LEAD_SURGE_DAMPING_DECEL_MEMORY_MAX = 1.0
LEAD_SURGE_DAMPING_DECEL_MEMORY_TIME = 2.0
LEAD_SURGE_DAMPING_CLEAR_PULLAWAY_SPEED = 2.0
LEAD_SURGE_DAMPING_CLEAR_PULLAWAY_ACCEL = 0.6
LEAD_SURGE_DAMPING_ACCEL_MAX = 0.25
LEAD_SURGE_DAMPING_COST = 0.6
LEAD_STOP_APPROACH_V_EGO_BP = [4.0, 8.0]
LEAD_STOP_APPROACH_V_LEAD_BP = [0.3, 1.0]
LEAD_STOP_APPROACH_REQUIRED_DECEL_BP = [0.6, 1.4]
LEAD_STOP_APPROACH_DECEL_CAP = 1.2
LEAD_STOP_APPROACH_COST = 10.0
MOVING_LEAD_STOP_APPROACH_V_EGO_BP = [4.0, 12.0]
MOVING_LEAD_STOP_APPROACH_V_LEAD_BP = [1.0, 3.0, 18.0, 22.0]
MOVING_LEAD_STOP_APPROACH_DECEL_BP = [0.5, 1.0]
MOVING_LEAD_STOP_APPROACH_CLOSING_BP = [0.5, 2.0]
MOVING_LEAD_STOP_APPROACH_ANTICIPATORY_CLOSING_BP = [0.5, 2.0, 4.0]
MOVING_LEAD_STOP_APPROACH_ANTICIPATORY_CLOSING_V = [0.0, 1.0, 0.0]
MOVING_LEAD_STOP_APPROACH_REQUIRED_DECEL_BP = [0.35, 1.2]
MOVING_LEAD_STOP_APPROACH_GAP_EXCESS_BP = [0.0, 10.0]
MOVING_LEAD_STOP_APPROACH_DECEL_BLEND = 0.75
MOVING_LEAD_STOP_APPROACH_DECEL_MIN = 0.4
MOVING_LEAD_STOP_APPROACH_DECEL_CAP = 1.8
MOVING_LEAD_STOP_APPROACH_CUSHION_FACTOR = 0.75
MOVING_LEAD_STOP_APPROACH_LIGHT_CUSHION_FRACTION = 0.35
MOVING_LEAD_STOP_APPROACH_FULL_CUSHION_FRACTION = 0.75
MOVING_LEAD_STOP_APPROACH_LIGHT_DECEL_MAX = 0.65
MOVING_LEAD_STOP_APPROACH_URGENT_CLOSING_BP = [2.3, 2.8]
MOVING_LEAD_STOP_APPROACH_PRE_TARGET_MARGIN_BP = [4.0, 8.0]
MOVING_LEAD_STOP_APPROACH_COAST_RECOVERY_GAP_BP = [0.0, 0.25, 0.5]
MOVING_LEAD_STOP_APPROACH_SOFT_RAMP_DECEL = 0.65
MOVING_LEAD_STOP_APPROACH_SOFT_RAMP_EXCESS_BP = [0.0, 2.0]
MOVING_LEAD_STOP_APPROACH_COAST_FIRST_EXCESS = 0.25
MOVING_LEAD_STOP_APPROACH_SOFT_RAMP_LEAD_DECEL_BP = [0.2, 0.8, 1.5, 2.0]
MOVING_LEAD_STOP_APPROACH_DESIRED_TTC_FADE = 3.5
MOVING_LEAD_STOP_APPROACH_DESIRED_TTC_MAX_BLEND = 0.5
LEAD_APPROACH_CAUTION_GAP_FRACTION = 0.75
LEAD_APPROACH_DANGER_GAP_FRACTION = 0.50
LEAD_APPROACH_CAUTION_TTC_FADE = 3.0
LEAD_APPROACH_CAUTION_TTC_MAX_BLEND = 0.75
MOVING_LEAD_STOP_APPROACH_DANGER_TTC_FULL = 1.0
MOVING_LEAD_STOP_APPROACH_DANGER_TTC_FADE = 2.0
MOVING_LEAD_STOP_APPROACH_COST = 50.0
PRE_TARGET_RUNWAY_DECEL_THRESHOLD_RELAXED = 0.8
PRE_TARGET_RUNWAY_DECEL_THRESHOLD_STANDARD = 1.0
PRE_TARGET_RUNWAY_DECEL_THRESHOLD_AGGRESSIVE = 1.3
PRE_TARGET_RUNWAY_DECEL_BLEND_WIDTH = 0.4
MOVING_LEAD_STOP_RESERVE_MAX = 2.5
MOVING_LEAD_STOP_RESERVE_V_EGO_BP = [0.2, 3.0]
MOVING_LEAD_STOP_RESERVE_V_LEAD_BP = [0.1, 1.5]
MOVING_LEAD_STOP_RESERVE_CLOSING_BP = [0.1, 1.5]
MOVING_LEAD_STOP_RESERVE_DECEL_BP = [0.05, 0.5]
MOVING_LEAD_CLOSING_CUSHION_V_EGO_BP = [7.0, 13.0]
MOVING_LEAD_CLOSING_CUSHION_V_LEAD_BP = [2.0, 4.0]
MOVING_LEAD_CLOSING_CUSHION_CLOSING_BP = [0.3, 2.0, 3.5]
MOVING_LEAD_CLOSING_CUSHION_GAP_EXCESS_BP = [0.0, 6.0]
MOVING_LEAD_CLOSING_CUSHION_ACCEL_MIN = -0.15
MOVING_LEAD_CLOSING_CUSHION_DECEL_MAX = 0.55
MOVING_LEAD_CLOSING_CUSHION_COST = 1.5
PROGRESSIVE_LEAD_APPROACH_RUNWAY_BP = [8.0, 25.0]
PROGRESSIVE_LEAD_APPROACH_MIN_V_EGO = 4.0
PROGRESSIVE_LEAD_APPROACH_MIN_V_LEAD = 3.0
PROGRESSIVE_LEAD_APPROACH_FAR_FLOOR_FRACTION = 0.75
PROGRESSIVE_LEAD_APPROACH_CLOSING_BP = [0.5, 2.0]
PROGRESSIVE_LEAD_HARD_RELAXATION_RUNWAY_BP = [5.0, 25.0]
PROGRESSIVE_LEAD_HARD_RELAXATION_CLOSING_GUARD_BP = [4.0, 6.0]
PROGRESSIVE_LEAD_HARD_RELAXATION_DECEL_BP = [0.05, 0.3]
LEAD_CHASE_ACCEL_BP = [0.25, 0.9]
LEAD_CHASE_OPENING_BP = [0.0, 1.0]
LEAD_CHASE_CLOSING_GUARD_BP = [2.0, 4.0]
LEAD_CHASE_CLOSING_FLOOR_FRACTION = 0.75
LEAD_CHASE_OPENING_FLOOR_FRACTION = 0.50
SLOW_MOVING_LEAD_RUNWAY_RELAXATION_MAX = 1.0
SLOW_MOVING_LEAD_RUNWAY_RELAXATION_CAP = 1.0
SLOW_MOVING_LEAD_RUNWAY_RELAXATION_V_EGO_BP = [0.1, 0.5]
SLOW_MOVING_LEAD_RUNWAY_RELAXATION_V_LEAD_BP = [0.1, 0.8, 2.5, 4.0]
SLOW_MOVING_LEAD_RUNWAY_RELAXATION_CLOSING_BP = [1.0, 3.0]
SLOW_MOVING_LEAD_RUNWAY_RELAXATION_DECEL_BP = [0.0, 0.5]
LEAD_ACCEL_MATCH_COST = 2.0
LEAD_ACCEL_MATCH_MIN_ABS_ACCEL = 0.05
LEAD_ACCEL_MATCH_MIN_POSITIVE_BLEND = 0.25
LEAD_ACCEL_MATCH_MIN_POSITIVE_GAP_EXCESS = 1.0
LEAD_ACCEL_MATCH_DECEL_TARGET_BLEND = 0.65
LEAD_ACCEL_MATCH_DECEL_NEAR_STOP_BLEND = 0.35
LEAD_ACCEL_MATCH_DECEL_CLOSING_BP = [0.3, 2.0]
LEAD_ACCEL_MATCH_DECEL_ANTICIPATION_TIME = 1.0
LEAD_ACCEL_MATCH_DECEL_CAP = 1.2
LEAD_ACCEL_MATCH_GAP_MARGIN = 10.0
LEAD_ACCEL_MATCH_GAP_MARGIN_FACTOR = 0.5
CRUISE_MIN_ACCEL = -1.2
CRUISE_MAX_ACCEL = 1.6
MIN_X_LEAD_FACTOR = 0.5
LEAD_GAP_COMFORT_MIN_V_EGO = 2.0
LEAD_GAP_COMFORT_LIGHT_DECEL = 0.5
LEAD_GAP_COMFORT_DANGER_HEADROOM = 1.5
LEAD_GAP_COMFORT_CLOSING_ENTER = 0.05
LEAD_GAP_COMFORT_CLOSING_EXIT = 0.15
LEAD_GAP_COMFORT_OPENING_CLOSING_BP = [-0.3, 0.0]
LEAD_GAP_COMFORT_HORIZON_BP = [0.0, 1.5, 3.0]
LEAD_ACCEL_RECOVERY_MIN_V_EGO = 1.0
LEAD_ACCEL_RECOVERY_MIN_LEAD_ACCEL = 0.2
LEAD_ACCEL_RECOVERY_ACCEL_BP = [LEAD_ACCEL_RECOVERY_MIN_LEAD_ACCEL, 0.8]
LEAD_ACCEL_RECOVERY_OPENING_BP = [0.2, 1.2]
LEAD_ACCEL_RECOVERY_GAP_MARGIN = 1.0
LEAD_ACCEL_RECOVERY_ACCEL_MAX = 0.55
SHORT_GAP_PULLAWAY_RESPONSE_MIN_GAP = 0.15
SHORT_GAP_PULLAWAY_RESPONSE_FULL_GAP = 1.0
SHORT_GAP_PULLAWAY_RESPONSE_MAX_CLOSING = 0.5
SHORT_GAP_PULLAWAY_RESPONSE_MIN_LEAD_ACCEL = 0.5
SHORT_GAP_PULLAWAY_RESPONSE_ACCEL_MAX_RELAXED = 0.35
SHORT_GAP_PULLAWAY_RESPONSE_ACCEL_MAX_STANDARD = 0.45
SHORT_GAP_PULLAWAY_RESPONSE_ACCEL_MAX_AGGRESSIVE = 0.55
SHORT_GAP_PULLAWAY_RESPONSE_COST = 1.4


def get_jerk_factor(personality=log.LongitudinalPersonality.standard):
  if personality == log.LongitudinalPersonality.relaxed:
    return 1.0
  elif personality == log.LongitudinalPersonality.standard:
    return 1.0
  elif personality == log.LongitudinalPersonality.aggressive:
    return 0.5
  else:
    raise NotImplementedError("Longitudinal personality not supported")


def get_T_FOLLOW(personality=log.LongitudinalPersonality.standard):
  if personality == log.LongitudinalPersonality.relaxed:
    return 1.85
  elif personality == log.LongitudinalPersonality.standard:
    return 1.55
  elif personality == log.LongitudinalPersonality.aggressive:
    return 1.30
  else:
    raise NotImplementedError("Longitudinal personality not supported")


def get_pre_target_runway_decel_threshold(t_follow):
  return float(np.interp(
    t_follow,
    [
      get_T_FOLLOW(log.LongitudinalPersonality.aggressive),
      get_T_FOLLOW(log.LongitudinalPersonality.standard),
      get_T_FOLLOW(log.LongitudinalPersonality.relaxed),
    ],
    [
      PRE_TARGET_RUNWAY_DECEL_THRESHOLD_AGGRESSIVE,
      PRE_TARGET_RUNWAY_DECEL_THRESHOLD_STANDARD,
      PRE_TARGET_RUNWAY_DECEL_THRESHOLD_RELAXED,
    ],
  ))


def get_stopped_equivalence_factor(v_lead):
  return (v_lead**2) / (2 * COMFORT_BRAKE)


def get_lead_stop_presentation_distance(v_ego, v_lead, a_lead=0.0, model_prob=1.0):
  confidence_blend = np.interp(model_prob, [LEAD_STOP_PRESENTATION_CONFIDENCE_MIN, 1.0], [0.0, 1.0])
  ego_blend = 1.0 - np.interp(v_ego, LEAD_STOP_PRESENTATION_V_EGO_BP, [0.0, 1.0])
  stopped_blend = 1.0 - np.interp(v_lead, LEAD_STOP_PRESENTATION_V_LEAD_BP, [0.0, 1.0])
  decel_blend = 1.0 - np.interp(np.clip(-a_lead, 0.0, LEAD_STOP_PRESENTATION_DECEL_BP[-1]),
                                LEAD_STOP_PRESENTATION_DECEL_BP, [0.0, 1.0])
  presentation_blend = confidence_blend * ego_blend * stopped_blend * decel_blend
  return STOP_DISTANCE - presentation_blend * (STOP_DISTANCE - LEAD_STOP_PRESENTATION_DISTANCE)


def get_stop_distance_buffer(v_ego):
  # Preserve the full stopped gap at low speed, but keep a smaller floor at speed.
  fade = (STOP_DISTANCE_FADE_V**2) / (v_ego**2 + STOP_DISTANCE_FADE_V**2)
  return STOP_DISTANCE_MIN + (STOP_DISTANCE - STOP_DISTANCE_MIN) * fade


def get_stopped_lead_buffer(v_ego, v_lead):
  ego_blend = np.interp(v_ego, STOPPED_LEAD_V_EGO_BP, [0.0, 1.0, 1.0])
  lead_blend = np.interp(v_lead, STOPPED_LEAD_V_LEAD_BP, [1.0, 1.0, 0.0])
  return STOPPED_LEAD_BUFFER * ego_blend * lead_blend


def get_lead_stop_gap_taper(v_ego, v_lead):
  ego_blend = np.interp(v_ego, LEAD_STOP_GAP_TAPER_V_EGO_BP, [1.0, 0.0])
  lead_blend = np.interp(v_lead, LEAD_STOP_GAP_TAPER_V_LEAD_BP, [0.0, 0.5, 1.0])
  return LEAD_STOP_GAP_TAPER_MAX * ego_blend * lead_blend


def get_lead_stop_gap_excess_offset(v_ego, d_rel):
  gap_excess = np.maximum(0.0, d_rel - get_stop_distance_buffer(v_ego))
  ego_blend = np.interp(v_ego, LEAD_STOP_GAP_EXCESS_V_EGO_BP, [1.0, 0.0])
  excess_blend = np.interp(gap_excess, LEAD_STOP_GAP_EXCESS_BP, [0.0, 1.0])
  return LEAD_STOP_GAP_EXCESS_OFFSET_MAX * ego_blend * excess_blend


def get_lead_departure_relaxation_blend(v_ego, v_lead, gap_opening):
  ego_blend = np.interp(v_ego, LEAD_DEPARTURE_V_EGO_BP, [1.0, 0.0])
  lead_blend = np.interp(v_lead, LEAD_DEPARTURE_V_LEAD_BP, [0.0, 1.0])
  relative_blend = np.interp(v_lead - v_ego, LEAD_DEPARTURE_V_REL_BP, [0.0, 1.0])
  gap_blend = np.interp(gap_opening, LEAD_DEPARTURE_GAP_OPENING_BP, [0.0, 1.0])
  return ego_blend * min(lead_blend, relative_blend, gap_blend)


def get_lead_departure_relaxation(v_ego, v_lead, gap_opening):
  return LEAD_DEPARTURE_RELAXATION_MAX * get_lead_departure_relaxation_blend(v_ego, v_lead, gap_opening)


def get_lead_departure_available_runway(v_ego, d_rel, gap_opening):
  stopped_gap_excess = max(0.0, d_rel - get_stop_distance_buffer(v_ego))
  return max(gap_opening, stopped_gap_excess)


def get_safe_obstacle_distance(v_ego, t_follow):
  return (v_ego**2) / (2 * COMFORT_BRAKE) + t_follow * v_ego + get_stop_distance_buffer(v_ego)


def get_desired_follow_distance(v_ego, v_lead, t_follow):
  return get_safe_obstacle_distance(v_ego, t_follow) - get_stopped_equivalence_factor(v_lead)


def get_lead_danger_distance(v_ego, v_lead, t_follow):
  return LEAD_DANGER_FACTOR * get_safe_obstacle_distance(v_ego, t_follow) - get_stopped_equivalence_factor(v_lead)


def get_lead_approach_gaps(v_ego, v_lead, t_follow, stop_gap=STOP_DISTANCE):
  target_gap = get_desired_follow_distance(v_ego, v_lead, t_follow)
  stop_gap = np.maximum(np.asarray(stop_gap, dtype=float), STOP_DISTANCE)
  gap_span = np.maximum(target_gap - stop_gap, 0.0)
  caution_gap = stop_gap + LEAD_APPROACH_CAUTION_GAP_FRACTION * gap_span
  legacy_danger_gap = get_lead_danger_distance(v_ego, v_lead, t_follow)
  danger_gap = np.minimum(stop_gap + LEAD_APPROACH_DANGER_GAP_FRACTION * gap_span, legacy_danger_gap)
  danger_gap = np.maximum(stop_gap, danger_gap)
  return target_gap, caution_gap, danger_gap


def get_time_to_gap(d_rel, gap, closing_speed):
  gap_margin = np.maximum(np.asarray(d_rel, dtype=float) - np.asarray(gap, dtype=float), 0.0)
  closing_speed = np.asarray(closing_speed, dtype=float)
  gap_margin, closing_speed = np.broadcast_arrays(gap_margin, closing_speed)
  return np.divide(
    gap_margin,
    closing_speed,
    out=np.full_like(gap_margin, np.inf, dtype=float),
    where=closing_speed > 1e-3,
  )


def get_lead_gap_comfort_floor(v_ego, v_lead, t_follow):
  return get_lead_danger_distance(v_ego, v_lead, t_follow) + LEAD_GAP_COMFORT_DANGER_HEADROOM


def get_lead_gap_comfort_recovery_blend(d_rel, comfort_floor, desired_gap):
  if desired_gap <= comfort_floor:
    return 0.0
  return float(np.interp(d_rel, [comfort_floor, desired_gap], [0.0, 1.0]))


def get_lead_gap_comfort_a_min(v_ego, v_lead, d_rel, t_follow, closing_threshold=0.0):
  closing_speed = v_ego - v_lead
  comfort_floor = get_lead_gap_comfort_floor(v_ego, v_lead, t_follow)
  desired_gap = get_desired_follow_distance(v_ego, v_lead, t_follow)
  if v_ego < LEAD_GAP_COMFORT_MIN_V_EGO or closing_speed > closing_threshold or desired_gap <= comfort_floor or not comfort_floor < d_rel < desired_gap:
    return ACCEL_MIN

  recovery_blend = get_lead_gap_comfort_recovery_blend(d_rel, comfort_floor, desired_gap)
  opening_blend = float(np.interp(closing_speed, LEAD_GAP_COMFORT_OPENING_CLOSING_BP, [1.0, 0.0]))
  light_brake_cap = -LEAD_GAP_COMFORT_LIGHT_DECEL * (1.0 - recovery_blend)
  return float(np.clip((1.0 - opening_blend) * light_brake_cap, ACCEL_MIN, 0.0))


def get_lead_accel_recovery_a_min(v_ego, v_lead, d_rel, a_lead, t_follow):
  if v_ego < LEAD_ACCEL_RECOVERY_MIN_V_EGO or d_rel <= STOP_DISTANCE or a_lead < LEAD_ACCEL_RECOVERY_MIN_LEAD_ACCEL or v_lead <= v_ego:
    return ACCEL_MIN

  comfort_floor = get_lead_gap_comfort_floor(v_ego, v_lead, t_follow)
  if d_rel <= comfort_floor:
    return ACCEL_MIN

  desired_gap = get_desired_follow_distance(v_ego, v_lead, t_follow)
  full_recovery_gap = max(comfort_floor, desired_gap) + LEAD_ACCEL_RECOVERY_GAP_MARGIN
  gap_blend = float(np.interp(d_rel, [comfort_floor, full_recovery_gap], [0.0, 1.0]))
  opening_blend = float(np.interp(v_lead - v_ego, LEAD_ACCEL_RECOVERY_OPENING_BP, [0.0, 1.0]))
  accel_blend = float(np.interp(a_lead, LEAD_ACCEL_RECOVERY_ACCEL_BP, [0.0, 1.0]))
  return LEAD_ACCEL_RECOVERY_ACCEL_MAX * min(gap_blend, opening_blend, accel_blend)


def get_short_gap_pullaway_response_accel_max(t_follow):
  return float(np.interp(
    t_follow,
    [get_T_FOLLOW(log.LongitudinalPersonality.aggressive), get_T_FOLLOW(log.LongitudinalPersonality.standard),
     get_T_FOLLOW(log.LongitudinalPersonality.relaxed)],
    [SHORT_GAP_PULLAWAY_RESPONSE_ACCEL_MAX_AGGRESSIVE, SHORT_GAP_PULLAWAY_RESPONSE_ACCEL_MAX_STANDARD,
     SHORT_GAP_PULLAWAY_RESPONSE_ACCEL_MAX_RELAXED],
  ))


def get_short_gap_pullaway_response_target(v_ego, v_lead, d_rel, a_lead, t_follow, model_prob=1.0, blocked=False):
  presentation_distance = get_lead_stop_presentation_distance(v_ego, v_lead, a_lead, model_prob)
  cushion = np.asarray(d_rel, dtype=float) - presentation_distance
  closing_speed = np.asarray(v_ego, dtype=float) - np.asarray(v_lead, dtype=float)
  model_confirmed = np.asarray(model_prob, dtype=float) >= LEAD_STOP_PRESENTATION_CONFIDENCE_MIN

  active = (
    (cushion > SHORT_GAP_PULLAWAY_RESPONSE_MIN_GAP) &
    (cushion < SHORT_GAP_PULLAWAY_RESPONSE_FULL_GAP) &
    (closing_speed <= SHORT_GAP_PULLAWAY_RESPONSE_MAX_CLOSING) &
    (np.asarray(a_lead, dtype=float) >= SHORT_GAP_PULLAWAY_RESPONSE_MIN_LEAD_ACCEL) &
    model_confirmed &
    ~np.asarray(blocked, dtype=bool)
  )
  if np.all(~active):
    return np.zeros_like(cushion), np.zeros_like(cushion)

  gap_blend = np.interp(cushion, [SHORT_GAP_PULLAWAY_RESPONSE_MIN_GAP, SHORT_GAP_PULLAWAY_RESPONSE_FULL_GAP], [0.0, 1.0])
  closing_blend = 1.0 - np.interp(closing_speed, [0.0, SHORT_GAP_PULLAWAY_RESPONSE_MAX_CLOSING], [0.0, 1.0])
  accel_blend = np.interp(np.asarray(a_lead, dtype=float), [SHORT_GAP_PULLAWAY_RESPONSE_MIN_LEAD_ACCEL, 1.0], [0.0, 1.0])
  response_blend = np.clip(gap_blend, 0.0, 1.0) * np.clip(closing_blend, 0.0, 1.0) * np.clip(accel_blend, 0.0, 1.0)
  accel_max = get_short_gap_pullaway_response_accel_max(t_follow)
  target = np.where(active, accel_max * response_blend, 0.0)
  cost = np.where(active, SHORT_GAP_PULLAWAY_RESPONSE_COST * response_blend, 0.0)
  if np.ndim(target) == 0:
    return float(target), float(cost)
  return target, cost


def get_lead_launch_comfort_target(v_ego, v_lead, d_rel, a_lead, t_follow, model_prob=1.0, blocked=False):
  return get_short_gap_pullaway_response_target(v_ego, v_lead, d_rel, a_lead, t_follow, model_prob=model_prob, blocked=blocked)


def get_moving_lead_closing_cushion_target(d_rel, v_ego, v_lead, t_follow):
  d_rel = np.asarray(d_rel, dtype=float)
  v_lead = np.asarray(v_lead, dtype=float)
  closing_speed = np.maximum(v_ego - v_lead, 0.0)
  comfort_floor = get_lead_gap_comfort_floor(v_ego, v_lead, t_follow)
  desired_gap = get_desired_follow_distance(v_ego, v_lead, t_follow)
  cushion_range = np.maximum(desired_gap - comfort_floor, 1e-3)
  cushion_used = np.clip((desired_gap - d_rel) / cushion_range, 0.0, 1.0)

  speed_blend = np.interp(v_ego, MOVING_LEAD_CLOSING_CUSHION_V_EGO_BP, [0.0, 1.0])
  moving_blend = np.interp(v_lead, MOVING_LEAD_CLOSING_CUSHION_V_LEAD_BP, [0.0, 1.0])
  closing_blend = np.interp(closing_speed, MOVING_LEAD_CLOSING_CUSHION_CLOSING_BP, [0.0, 1.0, 0.0])
  gap_blend = np.where(
    d_rel > desired_gap,
    1.0 - np.interp(d_rel - desired_gap, MOVING_LEAD_CLOSING_CUSHION_GAP_EXCESS_BP, [0.0, 1.0]),
    1.0,
  )
  safety_blend = np.interp(d_rel - comfort_floor, [0.0, APPROACH_MIN_GAP_BUFFER], [0.0, 1.0])
  cushion_blend = speed_blend * moving_blend * closing_blend * gap_blend * safety_blend
  if np.all(cushion_blend <= 0.0):
    return np.zeros_like(d_rel), np.zeros_like(d_rel)

  coast_target = MOVING_LEAD_CLOSING_CUSHION_ACCEL_MIN
  decel_target = -MOVING_LEAD_CLOSING_CUSHION_DECEL_MAX
  coast_blend = np.clip(cushion_used / 0.25, 0.0, 1.0)
  decel_blend = np.clip((cushion_used - 0.25) / 0.75, 0.0, 1.0)
  target = coast_blend * coast_target + decel_blend * (decel_target - coast_target)
  target = np.where(cushion_blend > 0.0, target, 0.0)
  cost = MOVING_LEAD_CLOSING_CUSHION_COST * cushion_blend * np.maximum(0.25, cushion_used)
  return target, cost


def get_approach_available_runway(x_lead, v_ego, v_lead, t_follow, a_lead=0.0):
  legacy_runway = x_lead - get_desired_follow_distance(v_ego, v_lead, t_follow)
  closing_speed = np.maximum(v_ego - v_lead, 0.0)
  moving_stop_reserve = get_moving_lead_stop_reserve(v_ego, v_lead, closing_speed, a_lead)
  relaxation = get_slow_moving_lead_runway_relaxation(v_ego, v_lead, closing_speed, a_lead)
  stop_floor = get_lead_stop_presentation_distance(v_ego, v_lead, a_lead) - relaxation
  stop_runway = x_lead + get_stopped_equivalence_factor(v_lead) - stop_floor - moving_stop_reserve
  slowing_blend = np.interp(np.clip(-a_lead, 0.0, APPROACH_STOP_RUNWAY_DECEL_BP[-1]), APPROACH_STOP_RUNWAY_DECEL_BP, [0.0, 1.0])
  return np.clip((1.0 - slowing_blend) * legacy_runway + slowing_blend * stop_runway, 0.0, 1e8)


def get_lead_stop_runway_blend(v_ego, v_lead, a_lead, closing_speed=None):
  closing_speed = np.maximum(v_ego - v_lead, 0.0) if closing_speed is None else closing_speed
  low_speed_blend = np.interp(v_ego, LEAD_STOP_RUNWAY_V_EGO_BP, [1.0, 0.0])
  stopped_blend = np.interp(v_lead, LEAD_STOP_RUNWAY_V_LEAD_BP, [1.0, 0.0])
  moving_blend = np.interp(v_lead, LEAD_STOP_RUNWAY_MOVING_V_LEAD_BP, [0.0, 1.0])
  slowing_blend = np.interp(np.clip(-a_lead, 0.0, LEAD_STOP_RUNWAY_DECEL_BP[-1]), LEAD_STOP_RUNWAY_DECEL_BP, [0.0, 1.0])
  relaxation_norm = max(min(SLOW_MOVING_LEAD_RUNWAY_RELAXATION_MAX, SLOW_MOVING_LEAD_RUNWAY_RELAXATION_CAP), 1e-6)
  relaxation_blend = get_slow_moving_lead_runway_relaxation(v_ego, v_lead, closing_speed, a_lead) / relaxation_norm
  return low_speed_blend * np.maximum(stopped_blend, moving_blend * np.maximum(slowing_blend, relaxation_blend))


def get_slow_moving_lead_runway_relaxation(v_ego, v_lead, closing_speed, a_lead):
  ego_blend = np.interp(v_ego, SLOW_MOVING_LEAD_RUNWAY_RELAXATION_V_EGO_BP, [0.0, 1.0])
  lead_blend = np.interp(v_lead, SLOW_MOVING_LEAD_RUNWAY_RELAXATION_V_LEAD_BP, [0.0, 1.0, 1.0, 0.0])
  controlled_closure_blend = 1.0 - np.interp(closing_speed, SLOW_MOVING_LEAD_RUNWAY_RELAXATION_CLOSING_BP, [0.0, 1.0])
  lead_decel_blend = np.interp(np.clip(-a_lead, 0.0, SLOW_MOVING_LEAD_RUNWAY_RELAXATION_DECEL_BP[-1]),
                               SLOW_MOVING_LEAD_RUNWAY_RELAXATION_DECEL_BP, [0.5, 1.0])
  relaxation = SLOW_MOVING_LEAD_RUNWAY_RELAXATION_MAX * ego_blend * lead_blend * controlled_closure_blend * lead_decel_blend
  return np.clip(relaxation, 0.0, SLOW_MOVING_LEAD_RUNWAY_RELAXATION_CAP)


def get_lead_stop_runway_available(x_lead, v_ego, v_lead, closing_speed, a_lead):
  moving_stop_reserve = get_moving_lead_stop_reserve(v_ego, v_lead, closing_speed, a_lead)
  relaxation = get_slow_moving_lead_runway_relaxation(v_ego, v_lead, closing_speed, a_lead)
  stop_floor = get_lead_stop_presentation_distance(v_ego, v_lead, a_lead) - relaxation
  return np.maximum(0.0, x_lead + get_stopped_equivalence_factor(v_lead) - stop_floor - moving_stop_reserve)


def get_lead_stop_runway_required_decel(x_lead, v_ego, v_lead, closing_speed, a_lead):
  runway = get_lead_stop_runway_available(x_lead, v_ego, v_lead, closing_speed, a_lead)
  return np.where(runway > 0.0, v_ego**2 / (2 * np.maximum(runway, 1e-3)), 1e8)


def get_lead_stop_runway_urgency(x_lead, v_ego, v_lead, t_follow, a_lead):
  closing_speed = np.maximum(v_ego - v_lead, 0.0)
  required_decel = get_lead_stop_runway_required_decel(x_lead, v_ego, v_lead, closing_speed, a_lead)
  decel_urgency = np.interp(required_decel, LEAD_STOP_RUNWAY_URGENCY_DECEL_BP, [0.0, 1.0])
  closing_urgency = np.interp(closing_speed, LEAD_STOP_RUNWAY_URGENCY_CLOSING_BP, [0.0, 1.0])
  min_gap = get_lead_danger_distance(v_ego, v_lead, t_follow) + APPROACH_MIN_GAP_BUFFER * (closing_speed > 0.0)
  danger_margin = x_lead - min_gap
  danger_urgency = closing_urgency * np.interp(danger_margin, [0.0, LEAD_STOP_RUNWAY_URGENCY_DANGER_MARGIN], [1.0, 0.0])
  return np.maximum(danger_urgency, decel_urgency * closing_urgency)


def get_lead_stop_runway_preference(x_lead, v_ego, v_lead, t_follow, a_lead):
  closing_speed = np.maximum(v_ego - v_lead, 0.0)
  return get_lead_stop_runway_blend(v_ego, v_lead, a_lead, closing_speed) * (1.0 - get_lead_stop_runway_urgency(x_lead, v_ego, v_lead, t_follow, a_lead))


def get_lead_stop_runway_gap(v_ego, v_lead, closing_speed, a_lead):
  moving_stop_reserve = get_moving_lead_stop_reserve(v_ego, v_lead, closing_speed, a_lead)
  relaxation = get_slow_moving_lead_runway_relaxation(v_ego, v_lead, closing_speed, a_lead)
  stop_floor = get_lead_stop_presentation_distance(v_ego, v_lead, a_lead) - relaxation
  ego_stop_distance = v_ego**2 / (2 * LEAD_STOP_RUNWAY_BRAKE)
  lead_stop_distance = get_stopped_equivalence_factor(v_lead)
  return np.maximum(stop_floor, stop_floor + moving_stop_reserve + ego_stop_distance - lead_stop_distance)


def get_lead_chase_target_gap(v_ego, v_lead, a_lead, t_follow, normal_gap=None):
  v_lead = np.asarray(v_lead, dtype=float)
  a_lead = np.asarray(a_lead, dtype=float)
  normal_gap = get_desired_follow_distance(v_ego, v_lead, t_follow) if normal_gap is None else np.asarray(normal_gap, dtype=float)
  closing_speed = np.maximum(v_ego - v_lead, 0.0)
  opening_speed = np.maximum(v_lead - v_ego, 0.0)

  accel_blend = np.interp(np.clip(a_lead, 0.0, LEAD_CHASE_ACCEL_BP[-1]), LEAD_CHASE_ACCEL_BP, [0.0, 1.0])
  closing_guard = 1.0 - np.interp(closing_speed, LEAD_CHASE_CLOSING_GUARD_BP, [0.0, 1.0])
  opening_blend = np.interp(opening_speed, LEAD_CHASE_OPENING_BP, [0.0, 1.0])
  floor_fraction = (
    LEAD_CHASE_CLOSING_FLOOR_FRACTION +
    opening_blend * (LEAD_CHASE_OPENING_FLOOR_FRACTION - LEAD_CHASE_CLOSING_FLOOR_FRACTION)
  )
  safety_floor = np.maximum(
    STOP_DISTANCE,
    get_lead_danger_distance(v_ego, v_lead, t_follow) + APPROACH_MIN_GAP_BUFFER * (closing_speed > 0.0),
  )
  chase_floor = np.maximum(safety_floor, floor_fraction * normal_gap)
  chase_blend = accel_blend * closing_guard
  return normal_gap - chase_blend * (normal_gap - chase_floor)


def get_progressive_lead_approach_gap(x_lead, v_ego, v_lead, t_follow, a_lead=0.0):
  x_lead = np.asarray(x_lead, dtype=float)
  v_lead = np.asarray(v_lead, dtype=float)
  closing_speed = np.clip(v_ego - v_lead, 0.0, 1e8)
  approach_speed = np.minimum(v_ego, v_lead)
  steady_gap = t_follow * approach_speed + get_stop_distance_buffer(approach_speed)
  speed_blend = np.interp(v_ego, [PROGRESSIVE_LEAD_APPROACH_MIN_V_EGO, PROGRESSIVE_LEAD_APPROACH_MIN_V_EGO + 2.0], [0.0, 1.0])
  moving_blend = np.interp(v_lead, [PROGRESSIVE_LEAD_APPROACH_MIN_V_LEAD, PROGRESSIVE_LEAD_APPROACH_MIN_V_LEAD + 2.0], [0.0, 1.0])
  closing_blend = np.interp(closing_speed, PROGRESSIVE_LEAD_APPROACH_CLOSING_BP, [0.0, 1.0])
  runway_margin = np.maximum(x_lead - steady_gap, 0.0)
  late_ramp_blend = 1.0 - np.interp(runway_margin, PROGRESSIVE_LEAD_APPROACH_RUNWAY_BP, [0.0, 1.0])
  progressive_floor = np.maximum(STOP_DISTANCE, PROGRESSIVE_LEAD_APPROACH_FAR_FLOOR_FRACTION * steady_gap)
  chase_gap = get_lead_chase_target_gap(v_ego, v_lead, a_lead, t_follow, normal_gap=steady_gap)
  far_target_gap = np.minimum(progressive_floor + late_ramp_blend * (steady_gap - progressive_floor), chase_gap)
  steady_gap = steady_gap + closing_blend * (far_target_gap - steady_gap)
  closing_gap = (closing_speed**2) / (2 * get_approach_brake(closing_speed))

  lead_decel_blend = np.interp(np.clip(-np.asarray(a_lead, dtype=float), 0.0, APPROACH_DECEL_BLEND_BP[-1]), APPROACH_DECEL_BLEND_BP, [0.0, 1.0])
  ramp_blend = np.maximum(late_ramp_blend, lead_decel_blend)
  return steady_gap + speed_blend * moving_blend * ramp_blend * closing_gap


def get_progressive_lead_hard_obstacle_relaxation(x_lead, v_ego, v_lead, a_lead, t_follow, target_gap=None):
  x_lead = np.asarray(x_lead, dtype=float)
  v_lead = np.asarray(v_lead, dtype=float)
  a_lead = np.asarray(a_lead, dtype=float)
  target_gap = get_progressive_lead_approach_gap(x_lead, v_ego, v_lead, t_follow, a_lead) if target_gap is None else np.asarray(target_gap, dtype=float)
  hard_gap = get_lead_danger_distance(v_ego, v_lead, t_follow)

  speed_blend = np.interp(v_ego, [PROGRESSIVE_LEAD_APPROACH_MIN_V_EGO, PROGRESSIVE_LEAD_APPROACH_MIN_V_EGO + 2.0], [0.0, 1.0])
  moving_blend = np.interp(v_lead, [PROGRESSIVE_LEAD_APPROACH_MIN_V_LEAD, PROGRESSIVE_LEAD_APPROACH_MIN_V_LEAD + 2.0], [0.0, 1.0])
  closing_speed = np.maximum(v_ego - v_lead, 0.0)
  closing_blend = np.interp(closing_speed, PROGRESSIVE_LEAD_APPROACH_CLOSING_BP, [0.0, 1.0])
  closing_guard = 1.0 - np.interp(closing_speed, PROGRESSIVE_LEAD_HARD_RELAXATION_CLOSING_GUARD_BP, [0.0, 1.0])
  decel_guard = 1.0 - np.interp(np.clip(-a_lead, 0.0, PROGRESSIVE_LEAD_HARD_RELAXATION_DECEL_BP[-1]),
                                PROGRESSIVE_LEAD_HARD_RELAXATION_DECEL_BP, [0.0, 1.0])
  runway_blend = np.interp(x_lead - hard_gap, PROGRESSIVE_LEAD_HARD_RELAXATION_RUNWAY_BP, [0.0, 1.0])
  relaxation = np.maximum(hard_gap - target_gap, 0.0)
  return relaxation * speed_blend * moving_blend * closing_blend * closing_guard * decel_guard * runway_blend


def get_lead_crawl_comfort_target(x_lead, v_ego, v_lead, a_lead, t_follow, block_short_gap_pullaway_response=False, model_prob=1.0):
  x_lead = np.asarray(x_lead, dtype=float)
  v_lead = np.asarray(v_lead, dtype=float)
  a_lead = np.asarray(a_lead, dtype=float)
  closing_speed = np.maximum(v_ego - v_lead, 0.0)
  opening_speed = np.maximum(v_lead - v_ego, 0.0)
  model_confirmed = np.asarray(model_prob, dtype=float) >= LEAD_STOP_PRESENTATION_CONFIDENCE_MIN
  speed_blend = np.interp(v_ego, LEAD_CRAWL_V_EGO_BP, [1.0, 0.0])
  moving_blend = np.interp(v_lead, LEAD_CRAWL_V_LEAD_BP, [0.0, 1.0])
  gap_blend = np.interp(x_lead, LEAD_CRAWL_GAP_BP, [0.0, 1.0, 0.0])
  brake_gap_blend = 1.0 - np.interp(x_lead, LEAD_CRAWL_BRAKE_GAP_BP, [0.0, 0.0, 1.0])
  urgency_blend = 1.0 - get_lead_stop_runway_urgency(x_lead, v_ego, v_lead, t_follow, a_lead)
  crawl_blend = speed_blend * moving_blend * gap_blend * urgency_blend * model_confirmed
  short_gap_target, short_gap_cost = get_lead_launch_comfort_target(
    v_ego, v_lead, x_lead, a_lead, t_follow, model_prob=model_prob, blocked=block_short_gap_pullaway_response,
  )
  if np.all(crawl_blend <= 0.0) and np.all(short_gap_cost <= 0.0):
    return np.zeros_like(x_lead), np.zeros_like(x_lead)

  required_decel = get_lead_stop_runway_required_decel(x_lead, v_ego, v_lead, closing_speed, a_lead)
  lead_decel_blend = np.interp(np.clip(-a_lead, 0.0, LEAD_CRAWL_DECEL_BP[-1]), LEAD_CRAWL_DECEL_BP, [0.0, 1.0])
  closing_blend = np.interp(closing_speed, LEAD_CRAWL_CLOSING_BP, [0.0, 1.0])
  required_decel_blend = np.interp(required_decel, LEAD_CRAWL_REQUIRED_DECEL_BP, [0.0, 1.0])
  opening_blend = np.interp(opening_speed, LEAD_CRAWL_OPENING_BP, [0.0, 1.0])
  brake_blend = brake_gap_blend * np.maximum.reduce([
    lead_decel_blend,
    closing_blend,
    required_decel_blend * np.maximum(lead_decel_blend, closing_blend),
  ])
  lead_accel_blend = np.interp(np.clip(a_lead, 0.0, LEAD_ACCEL_RECOVERY_ACCEL_BP[-1]), LEAD_ACCEL_RECOVERY_ACCEL_BP, [0.0, 1.0])
  accel_blend = np.minimum(opening_blend, np.maximum(opening_blend * 0.5, lead_accel_blend))

  accel_target = LEAD_CRAWL_ACCEL_MAX * accel_blend
  brake_target = LEAD_CRAWL_BRAKE_MAX * brake_blend
  target = np.clip(accel_target - brake_target, -LEAD_CRAWL_BRAKE_MAX, LEAD_CRAWL_ACCEL_MAX)
  target = np.where(short_gap_cost > 0.0, np.maximum(target, short_gap_target), target)
  cost = np.maximum(LEAD_CRAWL_COST * crawl_blend * np.maximum(brake_blend, accel_blend), short_gap_cost)
  return target, cost


def get_lead_crawl_accel_max(x_lead, v_ego, v_lead, a_lead, t_follow):
  x_lead = np.asarray(x_lead, dtype=float)
  v_lead = np.asarray(v_lead, dtype=float)
  a_lead = np.asarray(a_lead, dtype=float)
  opening_speed = np.maximum(v_lead - v_ego, 0.0)
  speed_blend = np.interp(v_ego, LEAD_CRAWL_V_EGO_BP, [1.0, 0.0])
  moving_blend = np.interp(v_lead, LEAD_CRAWL_V_LEAD_BP, [0.0, 1.0])
  gap_blend = np.interp(x_lead, LEAD_CRAWL_ACCEL_LIMIT_GAP_BP, [0.0, 1.0, 1.0, 0.0])
  urgency_blend = 1.0 - get_lead_stop_runway_urgency(x_lead, v_ego, v_lead, t_follow, a_lead)
  opening_blend = np.interp(opening_speed, LEAD_CRAWL_OPENING_BP, [0.0, 1.0])
  limit_blend = speed_blend * moving_blend * gap_blend * urgency_blend * opening_blend
  return ACCEL_MAX - limit_blend * (ACCEL_MAX - LEAD_CRAWL_ACCEL_LIMIT)


def get_lead_surge_damping_target(x_lead, v_ego, v_lead, a_lead, t_follow, decel_memory):
  x_lead = np.asarray(x_lead, dtype=float)
  v_lead = np.asarray(v_lead, dtype=float)
  a_lead = np.asarray(a_lead, dtype=float)
  opening_speed = np.maximum(v_lead - v_ego, 0.0)
  gap_excess = x_lead - get_desired_follow_distance(v_ego, v_lead, t_follow)

  speed_blend = np.interp(v_ego, LEAD_SURGE_DAMPING_V_EGO_BP, [0.0, 1.0, 1.0, 0.0])
  moving_blend = np.interp(v_lead, LEAD_SURGE_DAMPING_V_LEAD_BP, [0.0, 1.0])
  gap_blend = np.interp(gap_excess, LEAD_SURGE_DAMPING_GAP_EXCESS_BP, [0.0, 1.0])
  opening_blend = np.interp(opening_speed, LEAD_SURGE_DAMPING_OPENING_BP, [0.0, 1.0])
  decel_blend = np.interp(np.clip(decel_memory, 0.0, LEAD_SURGE_DAMPING_DECEL_MEMORY_MAX), LEAD_SURGE_DAMPING_DECEL_BP, [0.0, 1.0])
  urgency_blend = 1.0 - get_lead_stop_runway_urgency(x_lead, v_ego, v_lead, t_follow, a_lead)
  clear_pullaway_blend = np.where(
    (opening_speed >= LEAD_SURGE_DAMPING_CLEAR_PULLAWAY_SPEED) | (a_lead >= LEAD_SURGE_DAMPING_CLEAR_PULLAWAY_ACCEL),
    0.0,
    1.0,
  )
  damping_blend = speed_blend * moving_blend * gap_blend * opening_blend * decel_blend * urgency_blend * clear_pullaway_blend
  cost = LEAD_SURGE_DAMPING_COST * damping_blend
  target = np.where(cost > 0.0, LEAD_SURGE_DAMPING_ACCEL_MAX, 0.0)
  return target, cost


def get_selected_lead_targets(lead_0_targets, lead_1_targets, lead_0_costs, lead_1_costs, dominant_obstacle):
  targets = np.zeros_like(lead_0_targets)
  costs = np.zeros_like(lead_0_costs)
  lead_0_dominant = dominant_obstacle == 0
  lead_1_dominant = dominant_obstacle == 1
  targets[lead_0_dominant] = lead_0_targets[lead_0_dominant]
  targets[lead_1_dominant] = lead_1_targets[lead_1_dominant]
  costs[lead_0_dominant] = lead_0_costs[lead_0_dominant]
  costs[lead_1_dominant] = lead_1_costs[lead_1_dominant]
  return targets, costs


def get_combined_accel_target(accel_match_targets, accel_match_costs,
                              lead_0_closing_cushion_targets, lead_1_closing_cushion_targets,
                              lead_0_closing_cushion_costs, lead_1_closing_cushion_costs,
                              dominant_obstacle,
                              crawl_targets, crawl_costs,
                              stop_targets, stop_costs,
                              moving_stop_targets, moving_stop_costs,
                              surge_targets, surge_costs):
  closing_cushion_targets, closing_cushion_costs = get_selected_lead_targets(
    lead_0_closing_cushion_targets, lead_1_closing_cushion_targets,
    lead_0_closing_cushion_costs, lead_1_closing_cushion_costs, dominant_obstacle
  )
  combined_accel_costs = accel_match_costs + closing_cushion_costs + crawl_costs + stop_costs + moving_stop_costs + surge_costs
  combined_accel_targets = np.divide(
    accel_match_targets * accel_match_costs + closing_cushion_targets * closing_cushion_costs + crawl_targets * crawl_costs +
    stop_targets * stop_costs + moving_stop_targets * moving_stop_costs + surge_targets * surge_costs,
    combined_accel_costs,
    out=np.zeros_like(combined_accel_costs),
    where=combined_accel_costs > 0.0,
  )
  return combined_accel_targets, combined_accel_costs


def get_lead_stop_approach_comfort_target(x_lead, v_ego, v_lead, a_lead, t_follow):
  x_lead = np.asarray(x_lead, dtype=float)
  v_lead = np.asarray(v_lead, dtype=float)
  a_lead = np.asarray(a_lead, dtype=float)
  closing_speed = np.maximum(v_ego - v_lead, 0.0)
  required_decel = get_lead_stop_runway_required_decel(x_lead, v_ego, v_lead, closing_speed, a_lead)

  speed_blend = np.interp(v_ego, LEAD_STOP_APPROACH_V_EGO_BP, [0.0, 1.0])
  stopped_blend = np.interp(v_lead, LEAD_STOP_APPROACH_V_LEAD_BP, [1.0, 0.0])
  decel_blend = np.interp(required_decel, LEAD_STOP_APPROACH_REQUIRED_DECEL_BP, [0.0, 1.0])
  urgency_blend = 1.0 - get_lead_stop_runway_urgency(x_lead, v_ego, v_lead, t_follow, a_lead)
  comfort_blend = speed_blend * stopped_blend * decel_blend * urgency_blend
  if np.all(comfort_blend <= 0.0):
    return np.zeros_like(x_lead), np.zeros_like(x_lead)

  target = -np.minimum(required_decel, LEAD_STOP_APPROACH_DECEL_CAP)
  cost = LEAD_STOP_APPROACH_COST * comfort_blend
  return target, cost


def get_moving_lead_stop_approach_gap_deficit_blend(d_rel, v_lead, t_follow, reference_gap):
  cushion = MOVING_LEAD_STOP_APPROACH_CUSHION_FACTOR * t_follow * np.maximum(v_lead, 0.0)
  gap_deficit = np.maximum(reference_gap - d_rel, 0.0)
  cushion_used = np.divide(
    gap_deficit,
    np.maximum(cushion, 1e-3),
    out=np.zeros_like(gap_deficit, dtype=float),
    where=cushion > 0.0,
  ).clip(0.0, 1.0)
  return np.interp(
    cushion_used,
    [MOVING_LEAD_STOP_APPROACH_LIGHT_CUSHION_FRACTION, MOVING_LEAD_STOP_APPROACH_FULL_CUSHION_FRACTION],
    [0.0, 1.0],
  )


def get_moving_lead_stop_approach_ttc_gate(x_lead, v_ego, v_lead, t_follow):
  x_lead = np.asarray(x_lead, dtype=float)
  v_lead = np.asarray(v_lead, dtype=float)
  closing_speed = np.maximum(v_ego - v_lead, 0.0)
  x_lead, v_lead, closing_speed = np.broadcast_arrays(x_lead, v_lead, closing_speed)
  if np.all(closing_speed <= 0.0):
    return np.zeros_like(x_lead)

  desired_gap, caution_gap, danger_gap = get_lead_approach_gaps(v_ego, v_lead, t_follow)
  danger_gap = danger_gap + APPROACH_MIN_GAP_BUFFER
  desired_ttc = get_time_to_gap(x_lead, desired_gap, closing_speed)
  caution_ttc = get_time_to_gap(x_lead, caution_gap, closing_speed)
  danger_ttc = get_time_to_gap(x_lead, danger_gap, closing_speed)
  desired_gate = MOVING_LEAD_STOP_APPROACH_DESIRED_TTC_MAX_BLEND * (1.0 - np.interp(
    desired_ttc,
    [0.0, MOVING_LEAD_STOP_APPROACH_DESIRED_TTC_FADE],
    [0.0, 1.0],
  ))
  caution_gate = LEAD_APPROACH_CAUTION_TTC_MAX_BLEND * (1.0 - np.interp(
    caution_ttc,
    [0.0, LEAD_APPROACH_CAUTION_TTC_FADE],
    [0.0, 1.0],
  ))
  danger_gate = 1.0 - np.interp(
    danger_ttc,
    [MOVING_LEAD_STOP_APPROACH_DANGER_TTC_FULL, MOVING_LEAD_STOP_APPROACH_DANGER_TTC_FADE],
    [0.0, 1.0],
  )
  return np.maximum.reduce([desired_gate, caution_gate, danger_gate])


def get_moving_lead_stop_approach_comfort_target(x_lead, v_ego, v_lead, a_lead, t_follow):
  x_lead = np.asarray(x_lead, dtype=float)
  v_lead = np.asarray(v_lead, dtype=float)
  a_lead = np.asarray(a_lead, dtype=float)
  closing_speed = np.maximum(v_ego - v_lead, 0.0)
  required_decel = get_lead_stop_runway_required_decel(x_lead, v_ego, v_lead, closing_speed, a_lead)

  speed_blend = np.interp(v_ego, MOVING_LEAD_STOP_APPROACH_V_EGO_BP, [0.0, 1.0])
  moving_blend = np.interp(v_lead, MOVING_LEAD_STOP_APPROACH_V_LEAD_BP, [0.0, 1.0, 1.0, 0.0])
  lead_decel_blend = np.interp(np.clip(-a_lead, 0.0, MOVING_LEAD_STOP_APPROACH_DECEL_BP[-1]),
                               MOVING_LEAD_STOP_APPROACH_DECEL_BP, [0.0, 1.0])
  closing_blend = np.interp(closing_speed, MOVING_LEAD_STOP_APPROACH_CLOSING_BP, [0.0, 1.0])
  required_decel_blend = np.interp(required_decel, MOVING_LEAD_STOP_APPROACH_REQUIRED_DECEL_BP, [0.0, 1.0])
  desired_gap, caution_gap, danger_gap = get_lead_approach_gaps(v_ego, v_lead, t_follow)
  min_gap = danger_gap + APPROACH_MIN_GAP_BUFFER * (closing_speed > 0.0)
  danger_margin = x_lead - min_gap
  danger_blend = 1.0 - closing_blend * np.interp(danger_margin, [0.0, LEAD_STOP_RUNWAY_URGENCY_DANGER_MARGIN], [1.0, 0.0])
  danger_floor_blend = 0.25 * closing_blend * (1.0 - np.interp(danger_margin, [0.0, LEAD_STOP_RUNWAY_URGENCY_DANGER_MARGIN], [0.0, 1.0]))
  danger_blend = np.maximum(danger_blend, danger_floor_blend)
  pre_target = x_lead > desired_gap
  pre_target_margin = np.maximum(x_lead - desired_gap, 0.0)
  safe_closing_speed = np.sqrt(2.0 * MOVING_LEAD_STOP_APPROACH_SOFT_RAMP_DECEL * pre_target_margin)
  excess_closing_blend = np.interp(
    closing_speed - safe_closing_speed,
    MOVING_LEAD_STOP_APPROACH_SOFT_RAMP_EXCESS_BP,
    [0.0, 1.0],
  )
  brake_excess_closing_blend = np.interp(
    closing_speed - safe_closing_speed - MOVING_LEAD_STOP_APPROACH_COAST_FIRST_EXCESS,
    MOVING_LEAD_STOP_APPROACH_SOFT_RAMP_EXCESS_BP,
    [0.0, 1.0],
  )
  ttc_gate = get_moving_lead_stop_approach_ttc_gate(x_lead, v_ego, v_lead, t_follow)
  excess_closing_blend *= ttc_gate
  brake_excess_closing_blend *= ttc_gate
  soft_ramp_decel_blend = np.interp(
    np.clip(-a_lead, 0.0, MOVING_LEAD_STOP_APPROACH_SOFT_RAMP_LEAD_DECEL_BP[-1]),
    MOVING_LEAD_STOP_APPROACH_SOFT_RAMP_LEAD_DECEL_BP,
    [0.0, 1.0, 1.0, 0.0],
  )
  soft_closing_ramp_blend = np.where(pre_target, excess_closing_blend * soft_ramp_decel_blend, 0.0)
  soft_brake_ramp_blend = np.where(pre_target, brake_excess_closing_blend * soft_ramp_decel_blend, 0.0)
  active_lead_decel_blend = np.maximum(lead_decel_blend, np.where(soft_closing_ramp_blend > 0.0, soft_ramp_decel_blend, 0.0))
  gap_runway_need_blend = 1.0 - np.interp(x_lead - desired_gap, MOVING_LEAD_STOP_APPROACH_GAP_EXCESS_BP, [0.0, 1.0])
  anticipatory_runway_blend = lead_decel_blend * np.interp(closing_speed, MOVING_LEAD_STOP_APPROACH_ANTICIPATORY_CLOSING_BP,
                                                           MOVING_LEAD_STOP_APPROACH_ANTICIPATORY_CLOSING_V)
  runway_need_blend = np.maximum.reduce([gap_runway_need_blend, anticipatory_runway_blend, soft_closing_ramp_blend])
  comfort_blend = speed_blend * moving_blend * active_lead_decel_blend * closing_blend * required_decel_blend * danger_blend * runway_need_blend
  if np.all(comfort_blend <= 0.0):
    return np.zeros_like(x_lead), np.zeros_like(x_lead)

  full_decel = np.clip(MOVING_LEAD_STOP_APPROACH_DECEL_BLEND * required_decel,
                       MOVING_LEAD_STOP_APPROACH_DECEL_MIN, MOVING_LEAD_STOP_APPROACH_DECEL_CAP)
  light_decel = np.minimum(full_decel, MOVING_LEAD_STOP_APPROACH_LIGHT_DECEL_MAX)
  gap_deficit_blend = get_moving_lead_stop_approach_gap_deficit_blend(x_lead, v_lead, t_follow, caution_gap)
  urgent_closing_blend = np.interp(closing_speed, MOVING_LEAD_STOP_APPROACH_URGENT_CLOSING_BP, [0.0, 1.0])
  urgent_required_blend = np.interp(required_decel, [LEAD_STOP_RUNWAY_BRAKE, MOVING_LEAD_STOP_APPROACH_DECEL_CAP], [0.0, 1.0])
  urgent_danger_blend = np.interp(danger_margin, [0.0, LEAD_STOP_RUNWAY_URGENCY_DANGER_MARGIN], [1.0, 0.0])
  hard_braking_blend = np.interp(-a_lead, [2.0, 3.0], [0.0, 1.0])
  required_runway_blend = urgent_required_blend * hard_braking_blend * gap_runway_need_blend
  moderated_blend = np.maximum(gap_deficit_blend * danger_blend, urgent_closing_blend * urgent_required_blend * urgent_danger_blend)
  moderated_blend = np.maximum(moderated_blend, required_runway_blend)
  runway_critical_blend = np.maximum(np.maximum(gap_deficit_blend, urgent_closing_blend), required_runway_blend)
  low_speed_route_blend = (1.0 - np.interp(v_ego, [11.5, 12.5], [0.0, 1.0])) * np.interp(-a_lead, [1.2, 1.6], [0.0, 1.0])
  low_speed_route_blend *= np.interp(danger_margin, [0.25 * LEAD_STOP_RUNWAY_URGENCY_DANGER_MARGIN, 0.5 * LEAD_STOP_RUNWAY_URGENCY_DANGER_MARGIN], [0.0, 1.0])
  gap_deficit_blend = (1.0 - low_speed_route_blend) * runway_critical_blend + low_speed_route_blend * moderated_blend
  target_decel = light_decel + gap_deficit_blend * (full_decel - light_decel)
  target = -target_decel

  pre_target_threshold = get_pre_target_runway_decel_threshold(t_follow)
  pre_target_safety_threshold = max(
    MOVING_LEAD_STOP_APPROACH_DECEL_CAP,
    pre_target_threshold + 2.0 * PRE_TARGET_RUNWAY_DECEL_BLEND_WIDTH,
  )
  runway_safety_blend = np.interp(
    required_decel,
    [pre_target_safety_threshold, pre_target_safety_threshold + PRE_TARGET_RUNWAY_DECEL_BLEND_WIDTH],
    [0.0, 1.0],
  )
  danger_safety_blend = closing_blend * np.interp(
    danger_margin,
    [0.0, LEAD_STOP_RUNWAY_URGENCY_DANGER_MARGIN],
    [1.0, 0.0],
  )
  pre_target_margin_brake_blend = 1.0 - np.interp(
    pre_target_margin,
    MOVING_LEAD_STOP_APPROACH_PRE_TARGET_MARGIN_BP,
    [0.0, 1.0],
  )
  pre_target_margin_brake_blend = np.maximum(pre_target_margin_brake_blend, soft_brake_ramp_blend)
  pre_target_margin_brake_blend *= np.interp(
    required_decel,
    [pre_target_threshold, pre_target_threshold + PRE_TARGET_RUNWAY_DECEL_BLEND_WIDTH],
    [0.0, 1.0],
  )
  pre_target_margin_brake_blend = np.where(pre_target, pre_target_margin_brake_blend, 0.0)
  pre_target_brake_blend = np.maximum.reduce([runway_safety_blend, danger_safety_blend, pre_target_margin_brake_blend])
  coast_limited_target = np.maximum(target, MOVING_LEAD_CLOSING_CUSHION_ACCEL_MIN)
  pre_target_safety_blend = np.maximum.reduce([runway_safety_blend, danger_safety_blend, required_runway_blend])
  pre_target_light_target = np.maximum(target, -MOVING_LEAD_STOP_APPROACH_LIGHT_DECEL_MAX)
  pre_target_target = pre_target_light_target + pre_target_safety_blend * (target - pre_target_light_target)
  near_desired_recovery_blend = np.interp(
    desired_gap - x_lead,
    MOVING_LEAD_STOP_APPROACH_COAST_RECOVERY_GAP_BP,
    [0.0, 1.0, 0.0],
  )
  near_desired_recovery_blend *= 1.0 - np.interp(
    closing_speed,
    MOVING_LEAD_STOP_APPROACH_URGENT_CLOSING_BP,
    [0.0, 1.0],
  )
  near_desired_recovery_blend *= 1.0 - required_runway_blend
  near_desired_recovery_blend *= 1.0 - pre_target_brake_blend
  target = np.where(
    pre_target,
    coast_limited_target + pre_target_brake_blend * (pre_target_target - coast_limited_target),
    target + near_desired_recovery_blend * (coast_limited_target - target),
  )
  return target, MOVING_LEAD_STOP_APPROACH_COST * comfort_blend


def get_approach_follow_distance(x_lead, v_ego, v_lead, t_follow, a_lead=0.0):
  closing_speed = np.clip(v_ego - v_lead, 0.0, 1e8)
  moving_stop_reserve = get_moving_lead_stop_reserve(v_ego, v_lead, closing_speed, a_lead)
  approach_gap = get_progressive_lead_approach_gap(x_lead, v_ego, v_lead, t_follow, a_lead)
  decel_blend = np.interp(np.clip(-a_lead, 0.0, APPROACH_DECEL_BLEND_BP[-1]), APPROACH_DECEL_BLEND_BP, [0.0, 1.0])
  decel_blend *= 1.0 - np.interp(get_approach_available_runway(x_lead, v_ego, v_lead, t_follow, a_lead), APPROACH_RUNWAY_BLEND_BP, [0.0, 1.0])
  approach_gap = (1.0 - decel_blend) * approach_gap + decel_blend * (get_desired_follow_distance(v_ego, v_lead, t_follow) + moving_stop_reserve)
  min_gap = get_lead_danger_distance(v_ego, v_lead, t_follow) + APPROACH_MIN_GAP_BUFFER * (closing_speed > 0.0)
  min_gap_runway = np.maximum(x_lead - min_gap, 0.0)
  min_gap_blend = 1.0 - np.interp(min_gap_runway, PROGRESSIVE_LEAD_APPROACH_RUNWAY_BP, [0.0, 1.0])
  min_gap_decel_blend = np.interp(np.clip(-a_lead, 0.0, PROGRESSIVE_LEAD_HARD_RELAXATION_DECEL_BP[-1]),
                                  PROGRESSIVE_LEAD_HARD_RELAXATION_DECEL_BP, [0.0, 1.0])
  min_gap_blend = np.maximum(min_gap_blend, min_gap_decel_blend)
  normal_gap = np.maximum(approach_gap, approach_gap + min_gap_blend * (min_gap - approach_gap))
  stop_runway_blend = get_lead_stop_runway_preference(x_lead, v_ego, v_lead, t_follow, a_lead)
  stop_runway_gap = get_lead_stop_runway_gap(v_ego, v_lead, closing_speed, a_lead)
  return (1.0 - stop_runway_blend) * normal_gap + stop_runway_blend * stop_runway_gap


def get_approach_runway_blend(x_lead, v_ego, v_lead, t_follow, a_lead=0.0):
  runway = get_approach_available_runway(x_lead, v_ego, v_lead, t_follow, a_lead)
  return np.maximum(np.interp(runway, APPROACH_RUNWAY_BLEND_BP, [0.0, 1.0]), get_lead_stop_runway_preference(x_lead, v_ego, v_lead, t_follow, a_lead))


def get_approach_engage_offset(v_ego, x_lead, v_lead, t_follow, a_lead=0.0):
  closing_speed = np.clip(v_ego - v_lead, 0.0, 1e8)
  runway = get_approach_available_runway(x_lead, v_ego, v_lead, t_follow, a_lead)
  speed_blend = np.interp(closing_speed, APPROACH_ENGAGE_CLOSING_BP, [0.0, 1.0])
  runway_blend = np.interp(runway, APPROACH_ENGAGE_RUNWAY_BP, [0.0, 1.0])
  return APPROACH_ENGAGE_OFFSET_MAX * min(speed_blend, runway_blend)


def get_approach_brake(closing_speed):
  return np.interp(closing_speed, APPROACH_BRAKE_CLOSING_BP, [APPROACH_BRAKE, APPROACH_BRAKE_MIN])


def get_lead_time_gap_target(v_lead, t_follow):
  return max(STOP_DISTANCE, get_desired_follow_distance(v_lead, v_lead, t_follow))


def get_moving_lead_stop_reserve(v_ego, v_lead, closing_speed, a_lead):
  ego_blend = np.interp(v_ego, MOVING_LEAD_STOP_RESERVE_V_EGO_BP, [0.0, 1.0])
  lead_blend = np.interp(v_lead, MOVING_LEAD_STOP_RESERVE_V_LEAD_BP, [0.0, 1.0])
  closing_blend = np.interp(closing_speed, MOVING_LEAD_STOP_RESERVE_CLOSING_BP, [0.0, 1.0])
  decel_blend = np.interp(np.clip(-a_lead, 0.0, MOVING_LEAD_STOP_RESERVE_DECEL_BP[-1]), MOVING_LEAD_STOP_RESERVE_DECEL_BP, [0.0, 1.0])
  reserve_blend = lead_blend * decel_blend
  return MOVING_LEAD_STOP_RESERVE_MAX * ego_blend * closing_blend * reserve_blend


def get_lead_accel_match_margin(target_gap):
  return max(LEAD_ACCEL_MATCH_GAP_MARGIN, LEAD_ACCEL_MATCH_GAP_MARGIN_FACTOR * target_gap)


def get_lead_accel_match_blend(v_lead, d_rel, a_lead, t_follow, v_ego=None):
  if d_rel <= STOP_DISTANCE or abs(a_lead) < LEAD_ACCEL_MATCH_MIN_ABS_ACCEL:
    return 0.0

  v_ego = v_lead if v_ego is None else v_ego
  closing_speed = max(v_ego - v_lead, 0.0)
  decel_closing_speed = closing_speed
  if a_lead < 0.0:
    anticipated_closing_speed = -a_lead * LEAD_ACCEL_MATCH_DECEL_ANTICIPATION_TIME
    decel_closing_speed = max(closing_speed, min(anticipated_closing_speed, LEAD_ACCEL_MATCH_DECEL_CLOSING_BP[-1]))
  reserve = get_moving_lead_stop_reserve(v_ego, v_lead, decel_closing_speed, a_lead)
  target_gap = get_lead_time_gap_target(v_lead, t_follow) + reserve
  margin = get_lead_accel_match_margin(target_gap)
  positive_match_gap = STOP_DISTANCE + LEAD_ACCEL_MATCH_MIN_POSITIVE_GAP_EXCESS
  if a_lead > 0.0 and d_rel < positive_match_gap:
    return 0.0

  if a_lead < 0.0:
    closing_blend = float(np.interp(decel_closing_speed, LEAD_ACCEL_MATCH_DECEL_CLOSING_BP, [0.0, 1.0]))
    if d_rel <= target_gap:
      distance_blend = float(np.interp(d_rel, [STOP_DISTANCE, target_gap], [LEAD_ACCEL_MATCH_DECEL_NEAR_STOP_BLEND, LEAD_ACCEL_MATCH_DECEL_TARGET_BLEND]))
    else:
      distance_blend = float(np.interp(d_rel, [target_gap, target_gap + margin], [LEAD_ACCEL_MATCH_DECEL_TARGET_BLEND, 0.0]))
    return distance_blend * closing_blend

  if d_rel <= target_gap:
    if a_lead > 0.0:
      return float(np.interp(d_rel, [positive_match_gap, target_gap], [LEAD_ACCEL_MATCH_MIN_POSITIVE_BLEND, 1.0]))
    return 1.0

  return float(np.interp(d_rel, [target_gap, target_gap + margin], [1.0, 0.0]))


def get_lead_accel_match_target(v_lead, d_rel, a_lead, t_follow, v_ego=None):
  blend = get_lead_accel_match_blend(v_lead, d_rel, a_lead, t_follow, v_ego)
  if blend <= 0.0:
    return 0.0, 0.0

  accel_target = float(np.clip(a_lead * blend, ACCEL_MIN, ACCEL_MAX))
  if a_lead < 0.0:
    accel_target = max(accel_target, -LEAD_ACCEL_MATCH_DECEL_CAP)
  return accel_target, LEAD_ACCEL_MATCH_COST * blend


def _interp_linear_clipped(x, x0, x1, y0, y1):
  blend = np.clip((x - x0) / np.maximum(x1 - x0, 1e-9), 0.0, 1.0)
  return y0 + blend * (y1 - y0)


def get_lead_accel_match_targets(v_lead, d_rel, a_lead, t_follow, v_ego=None, block_short_gap_pullaway_response=False, model_prob=1.0):
  v_lead = np.asarray(v_lead, dtype=float)
  d_rel = np.asarray(d_rel, dtype=float)
  a_lead = np.asarray(a_lead, dtype=float)
  v_ego_values = v_lead if v_ego is None else np.asarray(v_ego, dtype=float)

  closing_speed = np.maximum(v_ego_values - v_lead, 0.0)
  anticipated_closing_speed = -a_lead * LEAD_ACCEL_MATCH_DECEL_ANTICIPATION_TIME
  decel_closing_speed = np.where(
    a_lead < 0.0,
    np.maximum(closing_speed, np.minimum(anticipated_closing_speed, LEAD_ACCEL_MATCH_DECEL_CLOSING_BP[-1])),
    closing_speed,
  )
  reserve = get_moving_lead_stop_reserve(v_ego_values, v_lead, decel_closing_speed, a_lead)
  target_gap = np.maximum(STOP_DISTANCE, get_desired_follow_distance(v_lead, v_lead, t_follow)) + reserve
  margin = np.maximum(LEAD_ACCEL_MATCH_GAP_MARGIN, LEAD_ACCEL_MATCH_GAP_MARGIN_FACTOR * target_gap)
  positive_match_gap = STOP_DISTANCE + LEAD_ACCEL_MATCH_MIN_POSITIVE_GAP_EXCESS
  positive_floor = positive_match_gap
  if v_ego is not None:
    presentation_distance = get_lead_stop_presentation_distance(v_ego_values, v_lead, a_lead, model_prob)
    model_confirmed = np.asarray(model_prob, dtype=float) >= LEAD_STOP_PRESENTATION_CONFIDENCE_MIN
    low_speed_cushion = v_ego_values < LEAD_STOP_PRESENTATION_V_EGO_BP[-1]
    positive_floor = np.where(
      model_confirmed & low_speed_cushion,
      np.minimum(positive_match_gap, presentation_distance + SHORT_GAP_PULLAWAY_RESPONSE_MIN_GAP),
      positive_match_gap,
    )
  blocked = np.asarray(block_short_gap_pullaway_response, dtype=bool)

  blend = np.zeros_like(v_lead, dtype=float)
  decel_active = (d_rel > STOP_DISTANCE) & (np.abs(a_lead) >= LEAD_ACCEL_MATCH_MIN_ABS_ACCEL)
  positive_active = (d_rel > positive_floor) & (np.abs(a_lead) >= LEAD_ACCEL_MATCH_MIN_ABS_ACCEL)

  decel_mask = decel_active & (a_lead < 0.0)
  if np.any(decel_mask):
    closing_blend = np.interp(decel_closing_speed, LEAD_ACCEL_MATCH_DECEL_CLOSING_BP, [0.0, 1.0])
    near_stop_blend = _interp_linear_clipped(d_rel, STOP_DISTANCE, target_gap,
                                             LEAD_ACCEL_MATCH_DECEL_NEAR_STOP_BLEND, LEAD_ACCEL_MATCH_DECEL_TARGET_BLEND)
    far_blend = _interp_linear_clipped(d_rel, target_gap, target_gap + margin,
                                       LEAD_ACCEL_MATCH_DECEL_TARGET_BLEND, 0.0)
    distance_blend = np.where(d_rel <= target_gap, near_stop_blend, far_blend)
    blend = np.where(decel_mask, distance_blend * closing_blend, blend)

  positive_mask = positive_active & (a_lead > 0.0) & ~blocked
  if np.any(positive_mask):
    near_blend = _interp_linear_clipped(d_rel, positive_floor, target_gap,
                                        LEAD_ACCEL_MATCH_MIN_POSITIVE_BLEND, 1.0)
    far_blend = _interp_linear_clipped(d_rel, target_gap, target_gap + margin, 1.0, 0.0)
    distance_blend = np.where(d_rel <= target_gap, near_blend, far_blend)
    closing_blend = 1.0 - np.interp(closing_speed, [0.0, SHORT_GAP_PULLAWAY_RESPONSE_MAX_CLOSING], [0.0, 1.0])
    blend = np.where(positive_mask, distance_blend * closing_blend, blend)

  accel_targets = np.clip(a_lead * blend, ACCEL_MIN, ACCEL_MAX)
  accel_targets = np.where(a_lead < 0.0, np.maximum(accel_targets, -LEAD_ACCEL_MATCH_DECEL_CAP), accel_targets)
  accel_targets = np.where(blend > 0.0, accel_targets, 0.0)
  if v_ego is not None:
    short_gap_targets, short_gap_costs = get_short_gap_pullaway_response_target(
      v_ego_values, v_lead, d_rel, a_lead, t_follow, model_prob=model_prob, blocked=block_short_gap_pullaway_response,
    )
    short_gap_active = short_gap_costs > 0.0
    accel_targets = np.where(short_gap_active, np.maximum(accel_targets, short_gap_targets), accel_targets)
    blend = np.where(short_gap_active, np.maximum(blend, short_gap_costs / LEAD_ACCEL_MATCH_COST), blend)
  return accel_targets, LEAD_ACCEL_MATCH_COST * blend


def gen_long_model():
  model = AcadosModel()
  model.name = MODEL_NAME

  # states
  x_ego, v_ego, a_ego = SX.sym('x_ego'), SX.sym('v_ego'), SX.sym('a_ego')
  model.x = vertcat(x_ego, v_ego, a_ego)

  # controls
  j_ego = SX.sym('j_ego')
  model.u = vertcat(j_ego)

  # xdot
  x_ego_dot = SX.sym('x_ego_dot')
  v_ego_dot = SX.sym('v_ego_dot')
  a_ego_dot = SX.sym('a_ego_dot')
  model.xdot = vertcat(x_ego_dot, v_ego_dot, a_ego_dot)

  # live parameters
  a_min = SX.sym('a_min')
  a_max = SX.sym('a_max')
  x_obstacle = SX.sym('x_obstacle')
  a_prev = SX.sym('a_prev')
  lead_t_follow = SX.sym('lead_t_follow')
  lead_danger_factor = SX.sym('lead_danger_factor')
  model.p = vertcat(a_min, a_max, x_obstacle, a_prev, lead_t_follow, lead_danger_factor)

  # dynamics model
  f_expl = vertcat(v_ego, a_ego, j_ego)
  model.f_impl_expr = model.xdot - f_expl
  model.f_expl_expr = f_expl
  return model


def gen_long_ocp():
  ocp = AcadosOcp()
  ocp.model = gen_long_model()

  Tf = T_IDXS[-1]

  # set dimensions
  ocp.dims.N = N

  # set cost module
  ocp.cost.cost_type = 'NONLINEAR_LS'
  ocp.cost.cost_type_e = 'NONLINEAR_LS'

  QR = np.zeros((COST_DIM, COST_DIM))
  Q = np.zeros((COST_E_DIM, COST_E_DIM))

  ocp.cost.W = QR
  ocp.cost.W_e = Q

  x_ego, v_ego, a_ego = ocp.model.x[0], ocp.model.x[1], ocp.model.x[2]
  j_ego = ocp.model.u[0]

  a_min, a_max = ocp.model.p[0], ocp.model.p[1]
  x_obstacle = ocp.model.p[2]
  a_prev = ocp.model.p[3]
  lead_t_follow = ocp.model.p[4]
  lead_danger_factor = ocp.model.p[5]

  ocp.cost.yref = np.zeros((COST_DIM,))
  ocp.cost.yref_e = np.zeros((COST_E_DIM,))

  desired_dist_comfort = get_safe_obstacle_distance(v_ego, lead_t_follow)

  # The main cost in normal operation is how close you are to the "desired" distance
  # from an obstacle at every timestep. This obstacle can be a lead car
  # or other object. In e2e mode we can use x_position targets as a cost
  # instead.
  costs = [((x_obstacle - x_ego) - desired_dist_comfort) / (v_ego + 10.0), x_ego, v_ego, a_ego, a_ego - a_prev, j_ego]
  ocp.model.cost_y_expr = vertcat(*costs)
  ocp.model.cost_y_expr_e = vertcat(*costs[:-1])

  # Constraints on speed, acceleration and desired distance to
  # the obstacle, which is treated as a slack constraint so it
  # behaves like an asymmetrical cost.
  constraints = vertcat(v_ego, (a_ego - a_min), (a_max - a_ego), ((x_obstacle - x_ego) - lead_danger_factor * desired_dist_comfort) / (v_ego + 10.0))
  ocp.model.con_h_expr = constraints

  x0 = np.zeros(X_DIM)
  ocp.constraints.x0 = x0
  ocp.parameter_values = np.array([-1.2, 1.2, 0.0, 0.0, get_T_FOLLOW(), LEAD_DANGER_FACTOR])

  # We put all constraint cost weights to 0 and only set them at runtime
  cost_weights = np.zeros(CONSTR_DIM)
  ocp.cost.zl = cost_weights
  ocp.cost.Zl = cost_weights
  ocp.cost.Zu = cost_weights
  ocp.cost.zu = cost_weights

  ocp.constraints.lh = np.zeros(CONSTR_DIM)
  ocp.constraints.uh = 1e4 * np.ones(CONSTR_DIM)
  ocp.constraints.idxsh = np.arange(CONSTR_DIM)

  # The HPIPM solver can give decent solutions even when it is stopped early
  # Which is critical for our purpose where compute time is strictly bounded
  # We use HPIPM in the SPEED_ABS mode, which ensures fastest runtime. This
  # does not cause issues since the problem is well bounded.
  ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
  ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
  ocp.solver_options.integrator_type = 'ERK'
  ocp.solver_options.nlp_solver_type = ACADOS_SOLVER_TYPE
  ocp.solver_options.qp_solver_cond_N = 1

  # More iterations take too much time and less lead to inaccurate convergence in
  # some situations. Ideally we would run just 1 iteration to ensure fixed runtime.
  ocp.solver_options.qp_solver_iter_max = 10
  ocp.solver_options.qp_tol = 1e-3

  # set prediction horizon
  ocp.solver_options.tf = Tf
  ocp.solver_options.shooting_nodes = T_IDXS

  ocp.code_export_directory = EXPORT_DIR
  return ocp


class LongitudinalMpc:
  def __init__(self, dt=DT_MDL):
    self.dt = dt
    self.solver = AcadosOcpSolverCython(MODEL_NAME, ACADOS_SOLVER_TYPE, N)
    self.reset()
    self.source = LongitudinalPlanSource.cruise

  def reset(self):
    self.solver.reset()

    self.x_sol = np.zeros((N + 1, X_DIM))
    self.u_sol = np.zeros((N, 1))
    self.v_solution = np.zeros(N + 1)
    self.a_solution = np.zeros(N + 1)
    self.j_solution = np.zeros(N)
    self.a_prev = np.array(self.a_solution)
    self.yref = np.zeros((N + 1, COST_DIM))

    for i in range(N):
      self.solver.cost_set(i, "yref", self.yref[i])
    self.solver.cost_set(N, "yref", self.yref[N][:COST_E_DIM])

    self.params = np.zeros((N + 1, PARAM_DIM))
    for i in range(N + 1):
      self.solver.set(i, 'x', np.zeros(X_DIM))

    self.last_cloudlog_t = 0
    self.status = False
    self.crash_cnt = 0.0
    self.solution_status = 0
    # timers
    self.solve_time = 0.0
    self.time_qp_solution = 0.0
    self.time_linearization = 0.0
    self.time_integrator = 0.0
    self.x0 = np.zeros(X_DIM)
    self.lead_departure_anchors = np.full(2, np.nan)
    self.lead_gap_comfort_active = np.zeros(2, dtype=bool)
    self.lead_surge_decel_memories = np.zeros(2)
    self._last_set_weights_key = None
    self._last_cost_weight_key = None
    self._last_accel_match_costs = None
    self.set_weights()

  def set_cost_weights(self, cost_weights, constraint_cost_weights, accel_match_costs=None):
    accel_match_costs_array = None if accel_match_costs is None else np.asarray(accel_match_costs, dtype=float)
    cost_weight_key = (tuple(cost_weights), tuple(constraint_cost_weights))
    if (
      self._last_cost_weight_key == cost_weight_key and
      (
        (accel_match_costs_array is None and self._last_accel_match_costs is None) or
        (
          accel_match_costs_array is not None and self._last_accel_match_costs is not None and
          np.array_equal(accel_match_costs_array, self._last_accel_match_costs)
        )
      )
    ):
      return

    W = np.asfortranarray(np.diag(cost_weights))
    for i in range(N):
      # TODO don't hardcode A_CHANGE_COST idx
      # reduce the cost on (a-a_prev) later in the horizon.
      W[4, 4] = cost_weights[4] * np.interp(T_IDXS[i], [0.0, 1.0, 2.0], [1.0, 1.0, 0.0])
      if accel_match_costs_array is not None:
        W[3, 3] = accel_match_costs_array[i]
      self.solver.cost_set(i, 'W', W)
    # Setting the slice without the copy make the array not contiguous,
    # causing issues with the C interface.
    if accel_match_costs_array is not None:
      W[3, 3] = accel_match_costs_array[N]
    self.solver.cost_set(N, 'W', np.copy(W[:COST_E_DIM, :COST_E_DIM]))

    # Set L2 slack cost on lower bound constraints
    Zl = np.array(constraint_cost_weights)
    for i in range(N):
      self.solver.cost_set(i, 'Zl', Zl)
    self._last_cost_weight_key = cost_weight_key
    self._last_accel_match_costs = None if accel_match_costs_array is None else np.array(accel_match_costs_array, copy=True)

  def set_weights(self, prev_accel_constraint=True, personality=log.LongitudinalPersonality.standard):
    set_weights_key = (bool(prev_accel_constraint), str(personality))
    if self._last_set_weights_key == set_weights_key:
      return

    jerk_factor = get_jerk_factor(personality)
    a_change_cost = A_CHANGE_COST if prev_accel_constraint else 0
    self.cost_weights = [X_EGO_OBSTACLE_COST, X_EGO_COST, V_EGO_COST, A_EGO_COST, jerk_factor * a_change_cost, jerk_factor * J_EGO_COST]
    self.constraint_cost_weights = [LIMIT_COST, LIMIT_COST, LIMIT_COST, DANGER_ZONE_COST]
    self._last_set_weights_key = set_weights_key
    self.set_cost_weights(self.cost_weights, self.constraint_cost_weights)

  def set_cur_state(self, v, a):
    v_prev = self.x0[1]
    self.x0[1] = v
    self.x0[2] = a
    if abs(v_prev - v) > 2.0:  # probably only helps if v < v_prev
      for i in range(N + 1):
        self.solver.set(i, 'x', self.x0)

  @staticmethod
  def extrapolate_lead(x_lead, v_lead, a_lead, a_lead_tau):
    a_lead_traj = a_lead * np.exp(-a_lead_tau * (T_IDXS**2) / 2.0)
    v_lead_traj = np.clip(v_lead + np.cumsum(T_DIFFS * a_lead_traj), 0.0, 1e8)
    x_lead_traj = x_lead + np.cumsum(T_DIFFS * v_lead_traj)
    lead_xv = np.column_stack((x_lead_traj, v_lead_traj))
    return lead_xv, a_lead_traj

  def process_lead(self, lead):
    v_ego = self.x0[1]
    if lead is not None and lead.status:
      x_lead = float(lead.dRel)
      v_lead = float(lead.vLead)
      a_lead = float(lead.aLeadK)
      a_lead_tau = float(lead.aLeadTau)
      valid_lead = all(np.isfinite(value) for value in (x_lead, v_lead, a_lead, a_lead_tau))
    else:
      valid_lead = False

    if not valid_lead:
      # Fake a fast lead car, so mpc can keep running in the same mode
      x_lead = 50.0
      v_lead = v_ego + 10.0
      a_lead = 0.0
      a_lead_tau = _LEAD_ACCEL_TAU

    # MPC will not converge if immediate crash is expected
    # Clip lead distance to what is still possible to brake for
    min_x_lead = MIN_X_LEAD_FACTOR * (v_ego + v_lead) * (v_ego - v_lead) / (-ACCEL_MIN * 2)
    x_lead = np.clip(x_lead, min_x_lead, 1e8)
    v_lead = np.clip(v_lead, 0.0, 1e8)
    a_lead = np.clip(a_lead, -10.0, 5.0)
    lead_xv, a_lead_traj = self.extrapolate_lead(x_lead, v_lead, a_lead, a_lead_tau)
    return lead_xv, a_lead, a_lead_traj

  def get_lead_departure_state(self, lead_idx, lead):
    v_ego = self.x0[1]
    if lead is None or not lead.status:
      self.lead_departure_anchors[lead_idx] = np.nan
      return False, 0.0, 0.0

    d_rel = float(lead.dRel)
    if v_ego >= LEAD_DEPARTURE_V_EGO_BP[-1]:
      self.lead_departure_anchors[lead_idx] = np.nan
      return False, 0.0, 0.0

    if not np.isfinite(self.lead_departure_anchors[lead_idx]):
      if v_ego <= LEAD_DEPARTURE_ARM_V_EGO:
        self.lead_departure_anchors[lead_idx] = d_rel
      return np.isfinite(self.lead_departure_anchors[lead_idx]), 0.0, 0.0

    self.lead_departure_anchors[lead_idx] = min(self.lead_departure_anchors[lead_idx], d_rel)

    gap_opening = max(0.0, d_rel - self.lead_departure_anchors[lead_idx])
    # If we're already parked long behind the lead, use that extra runway to start creeping with it.
    available_runway = get_lead_departure_available_runway(v_ego, d_rel, gap_opening)
    blend = get_lead_departure_relaxation_blend(v_ego, float(lead.vLeadK), available_runway)
    return True, blend, LEAD_DEPARTURE_RELAXATION_MAX * blend

  def get_lead_gap_comfort_state(self, lead_idx, lead, t_follow):
    if lead is None or not lead.status:
      self.lead_gap_comfort_active[lead_idx] = False
      return np.full(N + 1, ACCEL_MIN)

    closing_threshold = LEAD_GAP_COMFORT_CLOSING_EXIT if self.lead_gap_comfort_active[lead_idx] else LEAD_GAP_COMFORT_CLOSING_ENTER
    comfort_a_min = get_lead_gap_comfort_a_min(self.x0[1], float(lead.vLeadK), float(lead.dRel), t_follow, closing_threshold=closing_threshold)
    self.lead_gap_comfort_active[lead_idx] = comfort_a_min > ACCEL_MIN
    if not self.lead_gap_comfort_active[lead_idx]:
      return np.full(N + 1, ACCEL_MIN)

    comfort_horizon_blend = np.interp(T_IDXS, LEAD_GAP_COMFORT_HORIZON_BP, [1.0, 1.0, 0.0])
    return ACCEL_MIN + comfort_horizon_blend * (comfort_a_min - ACCEL_MIN)

  def update_lead_surge_decel_memory(self, lead_idx, lead):
    if lead is None or not lead.status:
      self.lead_surge_decel_memories[lead_idx] = 0.0
      return 0.0

    decay = LEAD_SURGE_DAMPING_DECEL_MEMORY_MAX * self.dt / LEAD_SURGE_DAMPING_DECEL_MEMORY_TIME
    memory = max(0.0, self.lead_surge_decel_memories[lead_idx] - decay)
    self.lead_surge_decel_memories[lead_idx] = max(memory, min(max(-float(lead.aLeadK), 0.0), LEAD_SURGE_DAMPING_DECEL_MEMORY_MAX))
    return self.lead_surge_decel_memories[lead_idx]

  def update(self, radarstate, v_cruise, personality=log.LongitudinalPersonality.standard, block_short_gap_pullaway_response=False):
    t_follow = get_T_FOLLOW(personality)
    v_ego = self.x0[1]
    self.status = radarstate.leadOne.status or radarstate.leadTwo.status

    if not np.isfinite(v_cruise):
      v_cruise = 0.0

    lead_0_surge_decel_memory = self.update_lead_surge_decel_memory(0, radarstate.leadOne)
    lead_1_surge_decel_memory = self.update_lead_surge_decel_memory(1, radarstate.leadTwo)
    lead_xv_0, lead_0_a, lead_0_a_traj = self.process_lead(radarstate.leadOne)
    lead_xv_1, lead_1_a, lead_1_a_traj = self.process_lead(radarstate.leadTwo)

    # To estimate a safe distance from a moving lead, we calculate how much stopping
    # distance that lead needs as a minimum. We can add that to the current distance
    # and then treat that as a stopped car/obstacle at this new distance.
    lead_0_departure_armed, lead_0_departure_blend, lead_0_departure_relaxation = self.get_lead_departure_state(0, radarstate.leadOne)
    lead_1_departure_armed, lead_1_departure_blend, lead_1_departure_relaxation = self.get_lead_departure_state(1, radarstate.leadTwo)
    lead_0_gap_comfort_a_min = self.get_lead_gap_comfort_state(0, radarstate.leadOne, t_follow)
    lead_1_gap_comfort_a_min = self.get_lead_gap_comfort_state(1, radarstate.leadTwo, t_follow)
    lead_0_hard_v = lead_xv_0[:, 1] if not lead_0_departure_armed else lead_0_departure_blend * lead_xv_0[:, 1]
    lead_1_hard_v = lead_xv_1[:, 1] if not lead_1_departure_armed else lead_1_departure_blend * lead_xv_1[:, 1]
    lead_0_stopped_buffer = get_stopped_lead_buffer(v_ego, lead_xv_0[:, 1])
    lead_1_stopped_buffer = get_stopped_lead_buffer(v_ego, lead_xv_1[:, 1])
    lead_0_stop_gap_taper = get_lead_stop_gap_taper(v_ego, lead_xv_0[:, 1])
    lead_1_stop_gap_taper = get_lead_stop_gap_taper(v_ego, lead_xv_1[:, 1])
    lead_0_stop_gap_excess_offset = get_lead_stop_gap_excess_offset(v_ego, lead_xv_0[:, 0])
    lead_1_stop_gap_excess_offset = get_lead_stop_gap_excess_offset(v_ego, lead_xv_1[:, 0])
    if lead_0_departure_armed:
      lead_0_stopped_buffer *= lead_0_departure_blend
    if lead_1_departure_armed:
      lead_1_stopped_buffer *= lead_1_departure_blend
    lead_0_stopped_buffer *= 1.0 - LEAD_STOP_RUNWAY_STOPPED_BUFFER_FADE * get_lead_stop_runway_preference(
      lead_xv_0[:, 0], v_ego, lead_xv_0[:, 1], t_follow, lead_0_a_traj
    )
    lead_1_stopped_buffer *= 1.0 - LEAD_STOP_RUNWAY_STOPPED_BUFFER_FADE * get_lead_stop_runway_preference(
      lead_xv_1[:, 0], v_ego, lead_xv_1[:, 1], t_follow, lead_1_a_traj
    )
    lead_0_moving_stop_reserve = get_moving_lead_stop_reserve(v_ego, lead_xv_0[:, 1], np.maximum(v_ego - lead_xv_0[:, 1], 0.0), lead_0_a_traj)
    lead_1_moving_stop_reserve = get_moving_lead_stop_reserve(v_ego, lead_xv_1[:, 1], np.maximum(v_ego - lead_xv_1[:, 1], 0.0), lead_1_a_traj)
    lead_0_desired_gap = get_approach_follow_distance(lead_xv_0[:, 0], v_ego, lead_xv_0[:, 1], t_follow, lead_0_a)
    lead_1_desired_gap = get_approach_follow_distance(lead_xv_1[:, 0], v_ego, lead_xv_1[:, 1], t_follow, lead_1_a)
    lead_0_hard_relaxation = get_progressive_lead_hard_obstacle_relaxation(
      lead_xv_0[:, 0], v_ego, lead_xv_0[:, 1], lead_0_a_traj, t_follow, target_gap=lead_0_desired_gap,
    )
    lead_1_hard_relaxation = get_progressive_lead_hard_obstacle_relaxation(
      lead_xv_1[:, 0], v_ego, lead_xv_1[:, 1], lead_1_a_traj, t_follow, target_gap=lead_1_desired_gap,
    )
    lead_0_obstacle = (
      lead_xv_0[:, 0] + get_stopped_equivalence_factor(lead_0_hard_v) - lead_0_stopped_buffer - lead_0_moving_stop_reserve
      + lead_0_stop_gap_taper + lead_0_stop_gap_excess_offset + lead_0_hard_relaxation
    )
    lead_1_obstacle = (
      lead_xv_1[:, 0] + get_stopped_equivalence_factor(lead_1_hard_v) - lead_1_stopped_buffer - lead_1_moving_stop_reserve
      + lead_1_stop_gap_taper + lead_1_stop_gap_excess_offset + lead_1_hard_relaxation
    )
    lead_0_cost_obstacle_soft = (
      lead_xv_0[:, 0]
      + np.clip(get_safe_obstacle_distance(v_ego, t_follow) - lead_0_desired_gap, 0.0, 1e8)
      + lead_0_stop_gap_taper
      + lead_0_stop_gap_excess_offset
    )
    lead_1_cost_obstacle_soft = (
      lead_xv_1[:, 0]
      + np.clip(get_safe_obstacle_distance(v_ego, t_follow) - lead_1_desired_gap, 0.0, 1e8)
      + lead_1_stop_gap_taper
      + lead_1_stop_gap_excess_offset
    )
    lead_0_cost_obstacle_soft -= get_approach_engage_offset(v_ego, lead_xv_0[0, 0], lead_xv_0[0, 1], t_follow, lead_0_a)
    lead_1_cost_obstacle_soft -= get_approach_engage_offset(v_ego, lead_xv_1[0, 0], lead_xv_1[0, 1], t_follow, lead_1_a)
    # Only bias the preferred gap once the lead has both opened real space and clearly pulled away.
    lead_0_cost_obstacle_soft += lead_0_departure_relaxation
    lead_1_cost_obstacle_soft += lead_1_departure_relaxation
    lead_0_runway_blend = get_approach_runway_blend(lead_xv_0[0, 0], v_ego, lead_xv_0[0, 1], t_follow, lead_0_a)
    lead_1_runway_blend = get_approach_runway_blend(lead_xv_1[0, 0], v_ego, lead_xv_1[0, 1], t_follow, lead_1_a)
    lead_0_cost_obstacle = lead_0_obstacle + lead_0_runway_blend * (lead_0_cost_obstacle_soft - lead_0_obstacle)
    lead_1_cost_obstacle = lead_1_obstacle + lead_1_runway_blend * (lead_1_cost_obstacle_soft - lead_1_obstacle)

    # Fake an obstacle for cruise, this ensures smooth acceleration to set speed
    # when the leads are no factor.
    v_lower = v_ego + (T_IDXS * CRUISE_MIN_ACCEL * 1.05)
    # TODO does this make sense when max_a is negative?
    v_upper = v_ego + (T_IDXS * CRUISE_MAX_ACCEL * 1.05)
    v_cruise_clipped = np.clip(v_cruise * np.ones(N + 1), v_lower, v_upper)
    cruise_obstacle = np.cumsum(T_DIFFS * v_cruise_clipped) + get_safe_obstacle_distance(v_cruise_clipped, t_follow)

    cost_obstacles = np.column_stack([lead_0_cost_obstacle, lead_1_cost_obstacle, cruise_obstacle])
    x_obstacles = np.column_stack([lead_0_obstacle, lead_1_obstacle, cruise_obstacle])
    self.source = MPC_SOURCES[np.argmin(cost_obstacles[0])]
    dominant_obstacle = np.argmin(x_obstacles, axis=1)

    lead_0_model_prob = float(radarstate.leadOne.modelProb) if radarstate.leadOne.status else 1.0
    lead_1_model_prob = float(radarstate.leadTwo.modelProb) if radarstate.leadTwo.status else 1.0
    lead_0_accel_targets, lead_0_accel_costs = get_lead_accel_match_targets(
      lead_xv_0[:, 1], lead_xv_0[:, 0], lead_0_a_traj, t_follow, v_ego, block_short_gap_pullaway_response, lead_0_model_prob,
    )
    lead_1_accel_targets, lead_1_accel_costs = get_lead_accel_match_targets(
      lead_xv_1[:, 1], lead_xv_1[:, 0], lead_1_a_traj, t_follow, v_ego, block_short_gap_pullaway_response, lead_1_model_prob,
    )
    lead_0_closing_cushion_targets, lead_0_closing_cushion_costs = get_moving_lead_closing_cushion_target(
      lead_xv_0[:, 0], v_ego, lead_xv_0[:, 1], t_follow
    )
    lead_1_closing_cushion_targets, lead_1_closing_cushion_costs = get_moving_lead_closing_cushion_target(
      lead_xv_1[:, 0], v_ego, lead_xv_1[:, 1], t_follow
    )

    lead_0_crawl_targets, lead_0_crawl_costs = get_lead_crawl_comfort_target(
      lead_xv_0[:, 0], v_ego, lead_xv_0[:, 1], lead_0_a_traj, t_follow, block_short_gap_pullaway_response, lead_0_model_prob,
    )
    lead_1_crawl_targets, lead_1_crawl_costs = get_lead_crawl_comfort_target(
      lead_xv_1[:, 0], v_ego, lead_xv_1[:, 1], lead_1_a_traj, t_follow, block_short_gap_pullaway_response, lead_1_model_prob,
    )
    lead_0_crawl_accel_max = get_lead_crawl_accel_max(lead_xv_0[:, 0], v_ego, lead_xv_0[:, 1], lead_0_a_traj, t_follow)
    lead_1_crawl_accel_max = get_lead_crawl_accel_max(lead_xv_1[:, 0], v_ego, lead_xv_1[:, 1], lead_1_a_traj, t_follow)
    lead_0_stop_targets, lead_0_stop_costs = get_lead_stop_approach_comfort_target(lead_xv_0[:, 0], v_ego, lead_xv_0[:, 1], lead_0_a_traj, t_follow)
    lead_1_stop_targets, lead_1_stop_costs = get_lead_stop_approach_comfort_target(lead_xv_1[:, 0], v_ego, lead_xv_1[:, 1], lead_1_a_traj, t_follow)
    lead_0_moving_stop_targets, lead_0_moving_stop_costs = get_moving_lead_stop_approach_comfort_target(
      lead_xv_0[:, 0], v_ego, lead_xv_0[:, 1], lead_0_a_traj, t_follow
    )
    lead_1_moving_stop_targets, lead_1_moving_stop_costs = get_moving_lead_stop_approach_comfort_target(
      lead_xv_1[:, 0], v_ego, lead_xv_1[:, 1], lead_1_a_traj, t_follow
    )
    lead_0_surge_targets, lead_0_surge_costs = get_lead_surge_damping_target(
      lead_xv_0[:, 0], v_ego, lead_xv_0[:, 1], lead_0_a_traj, t_follow, lead_0_surge_decel_memory
    )
    lead_1_surge_targets, lead_1_surge_costs = get_lead_surge_damping_target(
      lead_xv_1[:, 0], v_ego, lead_xv_1[:, 1], lead_1_a_traj, t_follow, lead_1_surge_decel_memory
    )
    lead_0_crawl_selected = lead_0_crawl_costs >= lead_1_crawl_costs
    crawl_targets = np.where(lead_0_crawl_selected, lead_0_crawl_targets, lead_1_crawl_targets)
    crawl_costs = np.where(lead_0_crawl_selected, lead_0_crawl_costs, lead_1_crawl_costs)
    lead_0_stop_selected = lead_0_stop_costs >= lead_1_stop_costs
    stop_targets = np.where(lead_0_stop_selected, lead_0_stop_targets, lead_1_stop_targets)
    stop_costs = np.where(lead_0_stop_selected, lead_0_stop_costs, lead_1_stop_costs)
    accel_match_targets, accel_match_costs = get_selected_lead_targets(
      lead_0_accel_targets, lead_1_accel_targets, lead_0_accel_costs, lead_1_accel_costs, dominant_obstacle
    )
    lead_0_moving_stop_selected = lead_0_moving_stop_costs >= lead_1_moving_stop_costs
    moving_stop_targets = np.where(lead_0_moving_stop_selected, lead_0_moving_stop_targets, lead_1_moving_stop_targets)
    moving_stop_costs = np.where(lead_0_moving_stop_selected, lead_0_moving_stop_costs, lead_1_moving_stop_costs)
    surge_targets, surge_costs = get_selected_lead_targets(
      lead_0_surge_targets, lead_1_surge_targets, lead_0_surge_costs, lead_1_surge_costs, dominant_obstacle
    )
    combined_accel_targets, combined_accel_costs = get_combined_accel_target(
      accel_match_targets, accel_match_costs,
      lead_0_closing_cushion_targets, lead_1_closing_cushion_targets,
      lead_0_closing_cushion_costs, lead_1_closing_cushion_costs,
      dominant_obstacle,
      crawl_targets, crawl_costs,
      stop_targets, stop_costs,
      moving_stop_targets, moving_stop_costs,
      surge_targets, surge_costs,
    )
    accel_match_targets = combined_accel_targets
    accel_match_costs = combined_accel_costs
    self.set_cost_weights(self.cost_weights, self.constraint_cost_weights, accel_match_costs)

    self.yref[:, :] = 0.0
    self.yref[:, 0] = (np.min(x_obstacles, axis=1) - np.min(cost_obstacles, axis=1)) / (v_ego + 10.0)
    self.yref[:, 3] = accel_match_targets
    for i in range(N):
      self.solver.set(i, "yref", self.yref[i])
    self.solver.set(N, "yref", self.yref[N][:COST_E_DIM])

    self.params[:, 0] = ACCEL_MIN
    self.params[dominant_obstacle == 0, 0] = lead_0_gap_comfort_a_min[dominant_obstacle == 0]
    self.params[dominant_obstacle == 1, 0] = lead_1_gap_comfort_a_min[dominant_obstacle == 1]
    self.params[:, 1] = ACCEL_MAX
    self.params[:, 1] = np.minimum(self.params[:, 1], np.minimum(lead_0_crawl_accel_max, lead_1_crawl_accel_max))
    self.params[:, 2] = np.min(x_obstacles, axis=1)
    self.params[:, 3] = np.copy(self.a_prev)
    self.params[:, 4] = t_follow
    self.params[:, 5] = LEAD_DANGER_FACTOR

    self.run()
    if np.any(lead_xv_0[FCW_IDXS, 0] - self.x_sol[FCW_IDXS, 0] < CRASH_DISTANCE) and radarstate.leadOne.modelProb > 0.9:
      self.crash_cnt += 1
    else:
      self.crash_cnt = 0

  def run(self):
    for i in range(N + 1):
      self.solver.set(i, 'p', self.params[i])
    self.solver.constraints_set(0, "lbx", self.x0)
    self.solver.constraints_set(0, "ubx", self.x0)

    self.solution_status = self.solver.solve()
    self.solve_time = float(self.solver.get_stats('time_tot')[0])
    self.time_qp_solution = float(self.solver.get_stats('time_qp')[0])
    self.time_linearization = float(self.solver.get_stats('time_lin')[0])
    self.time_integrator = float(self.solver.get_stats('time_sim')[0])

    for i in range(N + 1):
      self.x_sol[i] = self.solver.get(i, 'x')
    for i in range(N):
      self.u_sol[i] = self.solver.get(i, 'u')

    self.v_solution = self.x_sol[:, 1]
    self.a_solution = self.x_sol[:, 2]
    self.j_solution = self.u_sol[:, 0]

    self.a_prev = np.interp(T_IDXS + self.dt, T_IDXS, self.a_solution)

    t = time.monotonic()
    if self.solution_status != 0:
      if t > self.last_cloudlog_t + 5.0:
        self.last_cloudlog_t = t
        cloudlog.warning(f"Long mpc reset, solution_status: {self.solution_status}")
      self.reset()


if __name__ == "__main__":
  ocp = gen_long_ocp()
  AcadosOcpSolver.generate(ocp, json_file=JSON_FILE)
