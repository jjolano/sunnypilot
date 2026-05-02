#!/usr/bin/env python3
import math
import numpy as np

from cereal import custom, log
import cereal.messaging as messaging
from opendbc.car.interfaces import ACCEL_MIN, ACCEL_MAX
from openpilot.common.constants import CV
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import LongitudinalMpc, LongitudinalPlanSource, STOP_DISTANCE
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import T_IDXS as T_IDXS_MPC
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import get_T_FOLLOW, get_lead_accel_recovery_a_min, get_lead_stop_presentation_distance
from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N, get_accel_from_plan
from openpilot.selfdrive.car.cruise import V_CRUISE_MAX, V_CRUISE_UNSET
from openpilot.common.swaglog import cloudlog

from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlannerSP

A_CRUISE_MAX_VALS = [1.6, 1.2, 0.8, 0.6]
A_CRUISE_MAX_BP = [0.0, 10.0, 25.0, 40.0]
CONTROL_N_T_IDX = ModelConstants.T_IDXS[:CONTROL_N]
ALLOW_THROTTLE_THRESHOLD = 0.4
MIN_ALLOW_THROTTLE_SPEED = 2.5
CREEP_TO_STOP_GAP_ARM_EXCESS = 0.5
CREEP_TO_STOP_GAP_STOP_EXCESS = 0.05
CREEP_TO_STOP_GAP_MAX_V_EGO_ARM = 0.3
CREEP_TO_STOP_GAP_MAX_V_EGO = 1.0
# Treat pullaway creep as a near stopped-gap behavior; farther leads return to normal MPC handling.
CREEP_TO_STOP_GAP_MAX_EXCESS = 4.0
CREEP_TO_STOP_GAP_MIN_LEAD_SPEED = -0.3
CREEP_TO_STOP_GAP_MIN_MODEL_PROB = 0.5
CREEP_TO_STOP_GAP_SPEED_MAX = 0.75
CREEP_TO_STOP_GAP_SPEED_BP = [CREEP_TO_STOP_GAP_STOP_EXCESS, 1.0, 5.0]
CREEP_TO_STOP_GAP_SPEED_V = [0.0, 0.18, CREEP_TO_STOP_GAP_SPEED_MAX]
CREEP_TO_STOP_GAP_ACCEL_GAIN = 0.8
CREEP_TO_STOP_GAP_ACCEL_MIN = -0.25
CREEP_TO_STOP_GAP_ACCEL_MAX = 0.18
CREEP_TO_STOP_GAP_HOLD_EXCESS = 0.3
CREEP_TO_STOP_GAP_REHOLD_EXCESS = 0.2
CREEP_TO_STOP_GAP_HOLD_RELEASE_EXCESS = 0.45
CREEP_TO_STOP_GAP_HOLD_RELEASE_MIN_LEAD_SPEED = 0.05
CREEP_TO_STOP_GAP_HOLD_RELEASE_MIN_LEAD_ACCEL = 0.15
CREEP_TO_STOP_GAP_PULLAWAY_MIN_LEAD_SPEED = 0.25
CREEP_TO_STOP_GAP_PULLAWAY_ARM_EXCESS = 0.5
CREEP_TO_STOP_GAP_PULLAWAY_SPEED_MAX = 1.2
CREEP_TO_STOP_GAP_PULLAWAY_ACCEL_MAX = 0.55
CREEP_TO_STOP_GAP_PULLAWAY_ACCEL_MIN = 0.30
CREEP_TO_STOP_GAP_PREDICT_T = 0.8
CREEP_TO_STOP_GAP_PREDICT_MIN_LEAD_SPEED = 0.35
CREEP_TO_STOP_GAP_PREDICT_MIN_LEAD_ACCEL = 0.25
CREEP_TO_STOP_GAP_PREDICT_MIN_GAP_OPENING = 0.2
CREEP_TO_STOP_GAP_MODEL_LEAD_MIN_PROB = 0.75
CREEP_TO_STOP_GAP_MODEL_LEAD_MAX_DIST_ERROR = 1.5
CREEP_TO_STOP_GAP_MODEL_LEAD_MAX_V_ERROR = 1.0
CREEP_TO_STOP_GAP_MODEL_LEAD_MAX_X_STD = 2.0
CREEP_TO_STOP_GAP_MODEL_LEAD_MAX_Y_STD = 1.0
CREEP_TO_STOP_GAP_MODEL_LEAD_MAX_V_STD = 2.0
CREEP_TO_STOP_GAP_MODEL_LEAD_MIN_Y_ERROR = 0.75
CREEP_TO_STOP_GAP_MODEL_LEAD_MAX_Y_ERROR = 1.25
CREEP_TO_STOP_GAP_MODEL_LEAD_HORIZON = 2.0
CREEP_TO_STOP_GAP_MODEL_LEAD_CAMERA_OFFSET = 1.52
STOPPED_LEAD_GAP_FILL_ARM_TIME = 8.0
STOPPED_LEAD_GAP_FILL_ARM_MAX_V_EGO = 0.3
STOPPED_LEAD_GAP_FILL_ARM_MAX_GAP_EXCESS = 0.6
STOPPED_LEAD_GAP_FILL_ARM_MAX_LEAD_SPEED = 0.25
# Start stopped-lead gap fill once near pullaway creep stops chasing the lead.
STOPPED_LEAD_GAP_FILL_MIN_EXCESS = CREEP_TO_STOP_GAP_MAX_EXCESS
STOPPED_LEAD_GAP_FILL_MAX_EXCESS = 35.0
STOPPED_LEAD_GAP_FILL_MAX_V_EGO = 2.5
STOPPED_LEAD_GAP_FILL_MAX_LEAD_SPEED = 1.0
STOPPED_LEAD_GAP_FILL_MIN_MODEL_PROB = 0.75
STOPPED_LEAD_GAP_FILL_SPEED_MAX = 1.5
STOPPED_LEAD_GAP_FILL_SPEED_BP = [STOPPED_LEAD_GAP_FILL_MIN_EXCESS, 20.0, STOPPED_LEAD_GAP_FILL_MAX_EXCESS]
STOPPED_LEAD_GAP_FILL_SPEED_V = [CREEP_TO_STOP_GAP_SPEED_MAX, 1.2, STOPPED_LEAD_GAP_FILL_SPEED_MAX]
STOPPED_LEAD_GAP_FILL_ACCEL_GAIN = 0.6
STOPPED_LEAD_GAP_FILL_ACCEL_MAX = 0.35
STOPPED_LEAD_GAP_FILL_ACCEL_MIN = -0.25
ENGAGE_STOP_BOOTSTRAP_TIME = 0.75
ENGAGE_STOP_BOOTSTRAP_MIN_SPEED = 5.0
ENGAGE_STOP_BOOTSTRAP_MODEL_ACCEL = -1.0
E2E_STOP_APPROACH_MIN_V_EGO = 3.0
E2E_STOP_APPROACH_MAX_MODEL_ACCEL = 0.2
E2E_STOP_APPROACH_MIN_ENDPOINT = 5.0
E2E_STOP_APPROACH_EXPECTED_DIST_BP = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 55.0, 60.0]
E2E_STOP_APPROACH_EXPECTED_DIST_V = [8.0, 18.0, 30.0, 48.0, 68.0, 88.0, 104.0, 122.0]
E2E_STOP_APPROACH_SHORTAGE_BP = [0.15, 0.5]
E2E_STOP_APPROACH_DECEL_BP = [0.35, 1.15]
E2E_STOP_APPROACH_REQUIRED_DECEL_BLEND = 0.65
E2E_STOP_APPROACH_DECEL_MAX = 1.2
ENGAGE_STOP_BOOTSTRAP_MODEL_STOP_SPEED = 1.0
LEAD_LOSS_E2E_GUARD_TIME = 3.0
LEAD_LOSS_E2E_GUARD_ACCEL_FLOOR = -0.45
LEAD_LOSS_E2E_GUARD_MIN_D_REL = 45.0
LEAD_LOSS_E2E_GUARD_MIN_MODEL_PROB = 0.8
STOPPED_LEAD_GAP_FILL_CONTINUITY_MAX_D_REL_DELTA = 3.0
STOPPED_LEAD_GAP_FILL_CONTINUITY_MAX_V_LEAD_DELTA = 1.0

# Lookup table for turns
_A_TOTAL_MAX_V = [1.7, 3.2]
_A_TOTAL_MAX_BP = [20.0, 40.0]


def get_max_accel(v_ego):
  return np.interp(v_ego, A_CRUISE_MAX_BP, A_CRUISE_MAX_VALS)


def get_coast_accel(pitch):
  return np.sin(pitch) * -5.65 - 0.3  # fitted from data using xx/projects/allow_throttle/compute_coast_accel.py


def has_valid_radar_lead(radar_state):
  return radar_state.leadOne.status or radar_state.leadTwo.status


def has_model_stop_context(model_msg):
  if model_msg.action.shouldStop:
    return True

  positions = list(getattr(model_msg.position, "x", []))
  velocities = list(getattr(model_msg.velocity, "x", []))
  return any(x > 0.0 and v <= ENGAGE_STOP_BOOTSTRAP_MODEL_STOP_SPEED for x, v in zip(positions, velocities, strict=False))


def should_run_engage_stop_bootstrap(timer, v_ego, radar_state, model_msg):
  if timer <= 0.0 or v_ego < ENGAGE_STOP_BOOTSTRAP_MIN_SPEED or has_valid_radar_lead(radar_state):
    return False

  return bool(
    model_msg.action.shouldStop or
    (model_msg.action.desiredAcceleration <= ENGAGE_STOP_BOOTSTRAP_MODEL_ACCEL and has_model_stop_context(model_msg))
  )


def get_e2e_stop_approach_accel(v_ego, model_msg, radar_state, e2e_active, force_slow_decel=False,
                                brake_pressed=False, gas_pressed=False):
  blocked = not e2e_active or force_slow_decel or brake_pressed or gas_pressed
  blocked = blocked or v_ego < E2E_STOP_APPROACH_MIN_V_EGO or has_valid_radar_lead(radar_state)
  blocked = blocked or model_msg.action.shouldStop or model_msg.action.desiredAcceleration > E2E_STOP_APPROACH_MAX_MODEL_ACCEL
  blocked = blocked or len(model_msg.position.x) == 0
  if blocked:
    return 0.0

  endpoint_x = float(model_msg.position.x[-1])
  if not np.isfinite(endpoint_x) or endpoint_x <= 0.0:
    return 0.0

  expected_distance = float(np.interp(v_ego * CV.MS_TO_KPH, E2E_STOP_APPROACH_EXPECTED_DIST_BP, E2E_STOP_APPROACH_EXPECTED_DIST_V))
  max_decel_distance = v_ego**2 / (2.0 * E2E_STOP_APPROACH_DECEL_MAX * (1.0 - E2E_STOP_APPROACH_SHORTAGE_BP[0]))
  expected_distance = max(expected_distance, max_decel_distance)
  if expected_distance <= 0.0:
    return 0.0

  shortage = max(0.0, expected_distance - endpoint_x) / expected_distance
  if shortage <= E2E_STOP_APPROACH_SHORTAGE_BP[0]:
    return 0.0

  shortage_decel = float(np.interp(shortage, E2E_STOP_APPROACH_SHORTAGE_BP, E2E_STOP_APPROACH_DECEL_BP))
  required_decel = E2E_STOP_APPROACH_REQUIRED_DECEL_BLEND * v_ego**2 / (2.0 * max(endpoint_x, E2E_STOP_APPROACH_MIN_ENDPOINT))
  target_decel = min(max(shortage_decel, required_decel), E2E_STOP_APPROACH_DECEL_MAX)
  return -target_decel


def is_lane_change_active(model_msg):
  return model_msg.meta.laneChangeState != log.LaneChangeState.off


def update_lead_loss_e2e_guard_timer(timer, dt, previous_lead_status, previous_d_rel, previous_model_prob,
                                    current_has_lead, lane_change_active, reset_state=False, force_slow_decel=False,
                                    brake_pressed=False, gas_pressed=False):
  blocked = reset_state or force_slow_decel or brake_pressed or gas_pressed or current_has_lead
  if blocked:
    return 0.0

  lost_far_confirmed_lead = (
    previous_lead_status and
    previous_d_rel >= LEAD_LOSS_E2E_GUARD_MIN_D_REL and
    previous_model_prob >= LEAD_LOSS_E2E_GUARD_MIN_MODEL_PROB
  )
  if lost_far_confirmed_lead and lane_change_active:
    return LEAD_LOSS_E2E_GUARD_TIME

  return max(0.0, timer - dt)


def apply_lead_loss_e2e_guard_accel(e2e_accel, e2e_should_stop, timer, has_lead):
  if timer <= 0.0 or has_lead or e2e_should_stop:
    return e2e_accel
  return max(e2e_accel, LEAD_LOSS_E2E_GUARD_ACCEL_FLOOR)


def limit_accel_in_turns(v_ego, angle_steers, a_target, CP):
  """
  This function returns a limited long acceleration allowed, depending on the existing lateral acceleration
  this should avoid accelerating when losing the target in turns
  """
  # FIXME: This function to calculate lateral accel is incorrect and should use the VehicleModel
  # The lookup table for turns should also be updated if we do this
  a_total_max = np.interp(v_ego, _A_TOTAL_MAX_BP, _A_TOTAL_MAX_V)
  a_y = v_ego**2 * angle_steers * CV.DEG_TO_RAD / (CP.steerRatio * CP.wheelbase)
  a_x_allowed = math.sqrt(max(a_total_max**2 - a_y**2, 0.0))

  return [a_target[0], min(a_target[1], a_x_allowed)]


def get_predicted_lead_pullaway(v_lead, a_lead, a_lead_tau, horizon=CREEP_TO_STOP_GAP_PREDICT_T):
  steps = 4
  ts = np.linspace(horizon / steps, horizon, steps)
  dt = horizon / steps
  a_traj = a_lead * np.exp(-max(a_lead_tau, 0.0) * (ts**2) / 2.0)
  v_traj = np.clip(v_lead + np.cumsum(dt * a_traj), 0.0, 1e8)
  return float(v_traj[-1]), float(np.sum(dt * v_traj))


def has_predicted_lead_pullaway(gap_excess, predicted_v_lead, predicted_gap_opening):
  return (
    predicted_v_lead >= CREEP_TO_STOP_GAP_PREDICT_MIN_LEAD_SPEED and
    predicted_gap_opening >= CREEP_TO_STOP_GAP_PREDICT_MIN_GAP_OPENING and
    gap_excess + predicted_gap_opening >= CREEP_TO_STOP_GAP_PULLAWAY_ARM_EXCESS
  )


def get_model_lead_pullaway(model_msg, radar_lead, v_ego, horizon=CREEP_TO_STOP_GAP_MODEL_LEAD_HORIZON):
  if v_ego >= CREEP_TO_STOP_GAP_MAX_V_EGO_ARM or not getattr(radar_lead, "status", False):
    return 0.0, 0.0

  leads_v3 = getattr(model_msg, "leadsV3", [])
  if len(leads_v3) == 0:
    return 0.0, 0.0

  lead_msg = leads_v3[0]
  if float(getattr(lead_msg, "prob", 0.0)) < CREEP_TO_STOP_GAP_MODEL_LEAD_MIN_PROB:
    return 0.0, 0.0
  if float(getattr(radar_lead, "modelProb", getattr(lead_msg, "prob", 0.0))) < CREEP_TO_STOP_GAP_MIN_MODEL_PROB:
    return 0.0, 0.0

  d_rel = float(getattr(radar_lead, "dRel", 0.0))
  if not np.isfinite(d_rel):
    return 0.0, 0.0
  radar_y_rel = float(getattr(radar_lead, "yRel", 0.0))
  if not np.isfinite(radar_y_rel):
    return 0.0, 0.0
  radar_v_lead = float(getattr(radar_lead, "vLeadK", getattr(radar_lead, "vLead", 0.0)))
  if not np.isfinite(radar_v_lead):
    return 0.0, 0.0

  ts = np.asarray(getattr(lead_msg, "t", []), dtype=float)
  xs = np.asarray(getattr(lead_msg, "x", []), dtype=float)
  ys = np.asarray(getattr(lead_msg, "y", []), dtype=float)
  vs = np.asarray(getattr(lead_msg, "v", []), dtype=float)
  x_stds = np.asarray(getattr(lead_msg, "xStd", []), dtype=float)
  y_stds = np.asarray(getattr(lead_msg, "yStd", []), dtype=float)
  v_stds = np.asarray(getattr(lead_msg, "vStd", []), dtype=float)
  if any(values.ndim != 1 for values in (ts, xs, ys, vs, x_stds, y_stds, v_stds)):
    return 0.0, 0.0
  if ts.size == 0 or any(values.size != ts.size for values in (xs, ys, vs, x_stds, y_stds, v_stds)):
    return 0.0, 0.0
  if any(not np.all(np.isfinite(values)) for values in (ts, xs, ys, vs, x_stds, y_stds, v_stds)):
    return 0.0, 0.0
  if ts[0] > 0.05 or ts[-1] < horizon or np.any(np.diff(ts) <= 0.0):
    return 0.0, 0.0

  horizon_mask = ts <= horizon
  x_std = max(float(np.max(x_stds[horizon_mask])), float(np.interp(horizon, ts, x_stds)))
  y_std = max(float(np.max(y_stds[horizon_mask])), float(np.interp(horizon, ts, y_stds)))
  v_std = max(float(np.max(v_stds[horizon_mask])), float(np.interp(horizon, ts, v_stds)))
  if x_std > CREEP_TO_STOP_GAP_MODEL_LEAD_MAX_X_STD or y_std > CREEP_TO_STOP_GAP_MODEL_LEAD_MAX_Y_STD:
    return 0.0, 0.0
  if v_std > CREEP_TO_STOP_GAP_MODEL_LEAD_MAX_V_STD:
    return 0.0, 0.0

  model_d_rel_now = float(xs[0] - CREEP_TO_STOP_GAP_MODEL_LEAD_CAMERA_OFFSET)
  if abs(model_d_rel_now - d_rel) > CREEP_TO_STOP_GAP_MODEL_LEAD_MAX_DIST_ERROR:
    return 0.0, 0.0
  model_y_rel_now = float(-ys[0])
  max_y_error = min(CREEP_TO_STOP_GAP_MODEL_LEAD_MAX_Y_ERROR,
                    max(CREEP_TO_STOP_GAP_MODEL_LEAD_MIN_Y_ERROR, 2.0 * float(y_std)))
  if abs(model_y_rel_now - radar_y_rel) > max_y_error:
    return 0.0, 0.0
  if abs(float(vs[0]) - radar_v_lead) > CREEP_TO_STOP_GAP_MODEL_LEAD_MAX_V_ERROR:
    return 0.0, 0.0

  model_d_rel_horizon = float(np.interp(horizon, ts, xs) - CREEP_TO_STOP_GAP_MODEL_LEAD_CAMERA_OFFSET)
  predicted_v_lead = float(np.interp(horizon, ts, vs))
  predicted_gap_opening = max(0.0, model_d_rel_horizon - d_rel)
  return predicted_v_lead, predicted_gap_opening


def creep_to_stop_gap_blocked(v_ego, d_rel, v_lead, model_prob, brake_pressed=False, gas_pressed=False, force_slow_decel=False,
                              a_lead=0.0):
  stop_target = get_lead_stop_presentation_distance(v_ego, v_lead, a_lead, model_prob)
  gap_excess = d_rel - stop_target
  blocked = brake_pressed or gas_pressed or force_slow_decel or model_prob < CREEP_TO_STOP_GAP_MIN_MODEL_PROB
  blocked = blocked or v_lead < CREEP_TO_STOP_GAP_MIN_LEAD_SPEED or v_ego >= CREEP_TO_STOP_GAP_MAX_V_EGO
  return blocked or gap_excess <= 0.0 or gap_excess > CREEP_TO_STOP_GAP_MAX_EXCESS


def get_creep_to_stop_gap_accel(v_ego, d_rel, v_lead, model_prob, active, brake_pressed=False, gas_pressed=False,
                                force_slow_decel=False, a_lead=0.0, a_lead_tau=0.0,
                                model_predicted_v_lead=0.0, model_predicted_gap_opening=0.0):
  stop_target = get_lead_stop_presentation_distance(v_ego, v_lead, a_lead, model_prob)
  gap_excess = d_rel - stop_target
  if creep_to_stop_gap_blocked(v_ego, d_rel, v_lead, model_prob, brake_pressed, gas_pressed, force_slow_decel, a_lead):
    return False, 0.0

  radar_predicted_v_lead, radar_predicted_gap_opening = get_predicted_lead_pullaway(v_lead, a_lead, a_lead_tau)
  lead_pullaway = v_lead >= CREEP_TO_STOP_GAP_PULLAWAY_MIN_LEAD_SPEED
  radar_predicted_pullaway = a_lead >= CREEP_TO_STOP_GAP_PREDICT_MIN_LEAD_ACCEL and has_predicted_lead_pullaway(
    gap_excess, radar_predicted_v_lead, radar_predicted_gap_opening
  )
  model_predicted_pullaway = has_predicted_lead_pullaway(gap_excess, model_predicted_v_lead, model_predicted_gap_opening)
  predicted_pullaway = radar_predicted_pullaway or model_predicted_pullaway
  should_arm = gap_excess >= CREEP_TO_STOP_GAP_ARM_EXCESS and v_ego < CREEP_TO_STOP_GAP_MAX_V_EGO_ARM
  should_arm = should_arm or (lead_pullaway and gap_excess >= CREEP_TO_STOP_GAP_PULLAWAY_ARM_EXCESS) or predicted_pullaway
  if not active and not should_arm:
    return False, 0.0

  target_speed = float(np.interp(gap_excess, CREEP_TO_STOP_GAP_SPEED_BP, CREEP_TO_STOP_GAP_SPEED_V))
  accel_max = CREEP_TO_STOP_GAP_ACCEL_MAX
  if lead_pullaway or predicted_pullaway:
    predicted_v_lead = max(
      radar_predicted_v_lead if radar_predicted_pullaway else 0.0,
      model_predicted_v_lead if model_predicted_pullaway else 0.0,
    )
    target_speed = max(target_speed, min(max(v_lead, predicted_v_lead), CREEP_TO_STOP_GAP_PULLAWAY_SPEED_MAX))
    accel_max = CREEP_TO_STOP_GAP_PULLAWAY_ACCEL_MAX
  accel = np.clip((target_speed - v_ego) * CREEP_TO_STOP_GAP_ACCEL_GAIN, CREEP_TO_STOP_GAP_ACCEL_MIN, accel_max)
  if (lead_pullaway or predicted_pullaway) and v_ego < CREEP_TO_STOP_GAP_MAX_V_EGO_ARM and accel > 0.0:
    accel = max(accel, min(CREEP_TO_STOP_GAP_PULLAWAY_ACCEL_MIN, accel_max))
  return True, float(accel)


def should_release_creep_stop_hold(release_active, v_ego, d_rel, v_lead, a_lead, predicted_pullaway=False, model_prob=1.0):
  stop_target = get_lead_stop_presentation_distance(v_ego, v_lead, a_lead, model_prob)
  if v_ego >= CREEP_TO_STOP_GAP_MAX_V_EGO or d_rel <= stop_target + CREEP_TO_STOP_GAP_REHOLD_EXCESS:
    return False
  if release_active:
    return True
  return (
    d_rel >= stop_target + CREEP_TO_STOP_GAP_HOLD_RELEASE_EXCESS and
    (predicted_pullaway or v_lead >= CREEP_TO_STOP_GAP_HOLD_RELEASE_MIN_LEAD_SPEED or a_lead >= CREEP_TO_STOP_GAP_HOLD_RELEASE_MIN_LEAD_ACCEL)
  )


def should_hold_creep_to_stop_gap(v_ego, d_rel, v_lead, a_lead, predicted_pullaway=False, release_active=False, model_prob=1.0):
  if should_release_creep_stop_hold(release_active, v_ego, d_rel, v_lead, a_lead, predicted_pullaway, model_prob):
    return False
  stop_target = get_lead_stop_presentation_distance(v_ego, v_lead, a_lead, model_prob)
  return (
    not predicted_pullaway and
    v_ego < CREEP_TO_STOP_GAP_MAX_V_EGO and
    v_lead < CREEP_TO_STOP_GAP_PULLAWAY_MIN_LEAD_SPEED and
    a_lead <= 0.05 and
    d_rel <= stop_target + CREEP_TO_STOP_GAP_HOLD_EXCESS
  )


def should_arm_stopped_lead_gap_fill(v_ego, d_rel, v_lead, model_prob, brake_pressed=False, gas_pressed=False, force_slow_decel=False,
                                     a_lead=0.0):
  stop_target = get_lead_stop_presentation_distance(v_ego, v_lead, a_lead, model_prob)
  gap_excess = d_rel - stop_target
  return (
    not brake_pressed and not gas_pressed and not force_slow_decel and
    model_prob >= CREEP_TO_STOP_GAP_MIN_MODEL_PROB and
    v_ego < STOPPED_LEAD_GAP_FILL_ARM_MAX_V_EGO and
    abs(v_lead) <= STOPPED_LEAD_GAP_FILL_ARM_MAX_LEAD_SPEED and
    0.0 <= gap_excess <= STOPPED_LEAD_GAP_FILL_ARM_MAX_GAP_EXCESS
  )


def get_stopped_lead_gap_fill_accel(v_ego, d_rel, v_lead, model_prob, armed, brake_pressed=False, gas_pressed=False, force_slow_decel=False,
                                    a_lead=0.0):
  stop_target = get_lead_stop_presentation_distance(v_ego, v_lead, a_lead, model_prob)
  gap_excess = d_rel - stop_target
  blocked = (
    not armed or brake_pressed or gas_pressed or force_slow_decel or
    model_prob < STOPPED_LEAD_GAP_FILL_MIN_MODEL_PROB or
    v_ego >= STOPPED_LEAD_GAP_FILL_MAX_V_EGO or
    v_lead < CREEP_TO_STOP_GAP_MIN_LEAD_SPEED or
    v_lead > STOPPED_LEAD_GAP_FILL_MAX_LEAD_SPEED or
    gap_excess <= STOPPED_LEAD_GAP_FILL_MIN_EXCESS or
    gap_excess > STOPPED_LEAD_GAP_FILL_MAX_EXCESS
  )
  if blocked:
    return False, 0.0

  target_speed = float(np.interp(gap_excess, STOPPED_LEAD_GAP_FILL_SPEED_BP, STOPPED_LEAD_GAP_FILL_SPEED_V))
  accel = np.clip((target_speed - v_ego) * STOPPED_LEAD_GAP_FILL_ACCEL_GAIN,
                  STOPPED_LEAD_GAP_FILL_ACCEL_MIN, STOPPED_LEAD_GAP_FILL_ACCEL_MAX)
  return True, float(accel)


def stopped_lead_gap_fill_lead_continuous(track_id, prev_track_id, d_rel, prev_d_rel, v_lead, prev_v_lead):
  if prev_track_id == -2:
    return False
  if track_id >= 0 and prev_track_id >= 0 and track_id != prev_track_id:
    return False
  return bool(
    abs(d_rel - prev_d_rel) <= STOPPED_LEAD_GAP_FILL_CONTINUITY_MAX_D_REL_DELTA and
    abs(v_lead - prev_v_lead) <= STOPPED_LEAD_GAP_FILL_CONTINUITY_MAX_V_LEAD_DELTA
  )


class LongitudinalPlanner(LongitudinalPlannerSP):
  def __init__(self, CP, CP_SP, init_v=0.0, init_a=0.0, dt=DT_MDL):
    self.CP = CP
    self.mpc = LongitudinalMpc(dt=dt)
    LongitudinalPlannerSP.__init__(self, self.CP, CP_SP, self.mpc)
    self.fcw = False
    self.dt = dt
    self.allow_throttle = True

    self.a_desired = init_a
    self.v_desired_filter = FirstOrderFilter(init_v, 2.0, self.dt)
    self.prev_accel_clip = [ACCEL_MIN, ACCEL_MAX]
    self.prev_reset_state = True
    self.engage_stop_bootstrap_timer = 0.0
    self.output_a_target = 0.0
    self.output_should_stop = False
    self.creep_to_stop_gap_active = False
    self.creep_stop_hold_released = False
    self.stopped_lead_gap_fill_timer = 0.0
    self.lead_loss_e2e_guard_timer = 0.0
    self.previous_lead_loss_status = False
    self.previous_lead_loss_d_rel = 0.0
    self.previous_lead_loss_model_prob = 0.0
    self.stopped_lead_gap_fill_track_id = -2
    self.stopped_lead_gap_fill_d_rel = 0.0
    self.stopped_lead_gap_fill_v_lead = 0.0

    self.v_desired_trajectory = np.zeros(CONTROL_N)
    self.a_desired_trajectory = np.zeros(CONTROL_N)
    self.j_desired_trajectory = np.zeros(CONTROL_N)

  @staticmethod
  def parse_model(model_msg):
    if (
      len(model_msg.position.x) == ModelConstants.IDX_N
      and len(model_msg.velocity.x) == ModelConstants.IDX_N
      and len(model_msg.acceleration.x) == ModelConstants.IDX_N
    ):
      x = np.interp(T_IDXS_MPC, ModelConstants.T_IDXS, model_msg.position.x)
      v = np.interp(T_IDXS_MPC, ModelConstants.T_IDXS, model_msg.velocity.x)
      a = np.interp(T_IDXS_MPC, ModelConstants.T_IDXS, model_msg.acceleration.x)
      j = np.zeros(len(T_IDXS_MPC))
    else:
      x = np.zeros(len(T_IDXS_MPC))
      v = np.zeros(len(T_IDXS_MPC))
      a = np.zeros(len(T_IDXS_MPC))
      j = np.zeros(len(T_IDXS_MPC))
    if len(model_msg.meta.disengagePredictions.gasPressProbs) > 1:
      throttle_prob = model_msg.meta.disengagePredictions.gasPressProbs[1]
    else:
      throttle_prob = 1.0
    return x, v, a, j, throttle_prob

  def update(self, sm):
    LongitudinalPlannerSP.update(self, sm)

    if len(sm['carControl'].orientationNED) == 3:
      accel_coast = get_coast_accel(sm['carControl'].orientationNED[1])
    else:
      accel_coast = ACCEL_MAX

    v_ego = sm['carState'].vEgo
    v_cruise_kph = min(sm['carState'].vCruise, V_CRUISE_MAX)
    v_cruise = v_cruise_kph * CV.KPH_TO_MS
    v_cruise_initialized = sm['carState'].vCruise != V_CRUISE_UNSET

    long_control_off = sm['controlsState'].longControlState == LongCtrlState.off
    force_slow_decel = sm['controlsState'].forceDecel

    # Reset current state when not engaged, or user is controlling the speed
    reset_state = long_control_off if self.CP.openpilotLongitudinalControl else not sm['selfdriveState'].enabled
    # PCM cruise speed may be updated a few cycles later, check if initialized
    reset_state = reset_state or not v_cruise_initialized

    if reset_state:
      self.engage_stop_bootstrap_timer = 0.0
    elif self.prev_reset_state:
      self.engage_stop_bootstrap_timer = ENGAGE_STOP_BOOTSTRAP_TIME
    else:
      self.engage_stop_bootstrap_timer = max(0.0, self.engage_stop_bootstrap_timer - self.dt)
    self.prev_reset_state = reset_state

    # No change cost when user is controlling the speed, or when standstill
    prev_accel_constraint = not (reset_state or sm['carState'].standstill)

    accel_clip = [ACCEL_MIN, get_max_accel(v_ego)]
    steer_angle_without_offset = sm['carState'].steeringAngleDeg - sm['liveParameters'].angleOffsetDeg
    accel_clip = limit_accel_in_turns(v_ego, steer_angle_without_offset, accel_clip, self.CP)

    if reset_state:
      self.v_desired_filter.x = v_ego
      # Clip aEgo to cruise limits to prevent large accelerations when becoming active
      self.a_desired = np.clip(sm['carState'].aEgo, accel_clip[0], accel_clip[1])

    # Prevent divergence, smooth in current v_ego
    self.v_desired_filter.x = max(0.0, self.v_desired_filter.update(v_ego))
    _, _, _, _, throttle_prob = self.parse_model(sm['modelV2'])
    # Don't clip at low speeds since throttle_prob doesn't account for creep
    self.allow_throttle = throttle_prob > ALLOW_THROTTLE_THRESHOLD or v_ego <= MIN_ALLOW_THROTTLE_SPEED

    if not self.allow_throttle:
      clipped_accel_coast = max(accel_coast, accel_clip[0])
      clipped_accel_coast_interp = np.interp(v_ego, [MIN_ALLOW_THROTTLE_SPEED, MIN_ALLOW_THROTTLE_SPEED * 2], [accel_clip[1], clipped_accel_coast])
      accel_clip[1] = min(accel_clip[1], clipped_accel_coast_interp)

    # Get new v_cruise and a_desired from Smart Cruise Control and Speed Limit Assist
    v_cruise, self.a_desired = LongitudinalPlannerSP.update_targets(self, sm, self.v_desired_filter.x, self.a_desired, v_cruise)

    if force_slow_decel:
      v_cruise = 0.0

    lead_one = sm['radarState'].leadOne
    has_radar_lead = has_valid_radar_lead(sm['radarState'])
    engage_stop_bootstrap_active = should_run_engage_stop_bootstrap(self.engage_stop_bootstrap_timer, v_ego, sm['radarState'], sm['modelV2'])
    if engage_stop_bootstrap_active:
      v_cruise = 0.0

    self.lead_loss_e2e_guard_timer = update_lead_loss_e2e_guard_timer(
      self.lead_loss_e2e_guard_timer, self.dt,
      self.previous_lead_loss_status, self.previous_lead_loss_d_rel, self.previous_lead_loss_model_prob,
      has_radar_lead, is_lane_change_active(sm['modelV2']),
      reset_state=reset_state,
      force_slow_decel=force_slow_decel,
      brake_pressed=sm['carState'].brakePressed,
      gas_pressed=sm['carState'].gasPressed,
    )

    self.mpc.set_weights(prev_accel_constraint, personality=sm['selfdriveState'].personality)
    self.mpc.set_cur_state(self.v_desired_filter.x, self.a_desired)
    self.mpc.update(sm['radarState'], v_cruise, personality=sm['selfdriveState'].personality)

    self.v_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC, self.mpc.v_solution)
    self.a_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC, self.mpc.a_solution)
    self.j_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC[:-1], self.mpc.j_solution)

    # TODO counter is only needed because radar is glitchy, remove once radar is gone
    self.fcw = self.mpc.crash_cnt > 2 and not sm['carState'].standstill
    if self.fcw:
      cloudlog.info("FCW triggered")

    # Interpolate 0.05 seconds and save as starting point for next iteration
    a_prev = self.a_desired
    self.a_desired = float(np.interp(self.dt, CONTROL_N_T_IDX, self.a_desired_trajectory))
    self.v_desired_filter.x = self.v_desired_filter.x + self.dt * (self.a_desired + a_prev) / 2.0

    action_t = self.CP.longitudinalActuatorDelay + DT_MDL
    output_a_target_mpc, output_should_stop_mpc = get_accel_from_plan(
      self.v_desired_trajectory, self.a_desired_trajectory, CONTROL_N_T_IDX, action_t=action_t, vEgoStopping=self.CP.vEgoStopping
    )
    output_a_target_e2e = sm['modelV2'].action.desiredAcceleration
    output_should_stop_e2e = sm['modelV2'].action.shouldStop

    e2e_active = self.is_e2e(sm)
    if e2e_active:
      output_a_target_e2e = apply_lead_loss_e2e_guard_accel(
        output_a_target_e2e, output_should_stop_e2e, self.lead_loss_e2e_guard_timer, has_radar_lead
      )
      output_a_target = min(output_a_target_e2e, output_a_target_mpc)
      self.output_should_stop = output_should_stop_e2e or output_should_stop_mpc
      if output_a_target < output_a_target_mpc:
        self.mpc.source = LongitudinalPlanSource.e2e
    else:
      output_a_target = output_a_target_mpc
      self.output_should_stop = output_should_stop_mpc

    e2e_stop_approach_a_target = get_e2e_stop_approach_accel(
      v_ego, sm['modelV2'], sm['radarState'], e2e_active,
      force_slow_decel=force_slow_decel or reset_state,
      brake_pressed=sm['carState'].brakePressed,
      gas_pressed=sm['carState'].gasPressed,
    )
    if e2e_stop_approach_a_target < 0.0 and e2e_stop_approach_a_target < output_a_target:
      output_a_target = e2e_stop_approach_a_target
      self.mpc.source = LongitudinalPlanSource.e2e

    lead_track_id = int(getattr(lead_one, "radarTrackId", -2)) if lead_one.status else -2
    lead_d_rel = float(lead_one.dRel) if lead_one.status else 0.0
    lead_v_lead = float(lead_one.vLeadK) if lead_one.status else 0.0
    if lead_one.status and should_arm_stopped_lead_gap_fill(
      v_ego, lead_d_rel, lead_v_lead, float(lead_one.modelProb),
      brake_pressed=sm['carState'].brakePressed,
      gas_pressed=sm['carState'].gasPressed,
      force_slow_decel=force_slow_decel or reset_state,
      a_lead=float(lead_one.aLeadK),
    ):
      self.stopped_lead_gap_fill_timer = STOPPED_LEAD_GAP_FILL_ARM_TIME
      self.stopped_lead_gap_fill_track_id = lead_track_id
      self.stopped_lead_gap_fill_d_rel = lead_d_rel
      self.stopped_lead_gap_fill_v_lead = lead_v_lead
    elif not (lead_one.status and stopped_lead_gap_fill_lead_continuous(
      lead_track_id, self.stopped_lead_gap_fill_track_id, lead_d_rel, self.stopped_lead_gap_fill_d_rel,
      lead_v_lead, self.stopped_lead_gap_fill_v_lead,
    )):
      self.stopped_lead_gap_fill_timer = 0.0
      self.stopped_lead_gap_fill_track_id = -2
      self.stopped_lead_gap_fill_d_rel = 0.0
      self.stopped_lead_gap_fill_v_lead = 0.0
    else:
      self.stopped_lead_gap_fill_timer = max(0.0, self.stopped_lead_gap_fill_timer - self.dt)
      self.stopped_lead_gap_fill_d_rel = lead_d_rel
      self.stopped_lead_gap_fill_v_lead = lead_v_lead

    model_predicted_v_lead, model_predicted_gap_opening = (
      get_model_lead_pullaway(sm['modelV2'], lead_one, v_ego) if lead_one.status else (0.0, 0.0)
    )
    self.creep_to_stop_gap_active, creep_a_target = get_creep_to_stop_gap_accel(
      v_ego, float(lead_one.dRel), float(lead_one.vLeadK), float(lead_one.modelProb),
      self.creep_to_stop_gap_active and not reset_state,
      brake_pressed=sm['carState'].brakePressed,
      gas_pressed=sm['carState'].gasPressed,
      force_slow_decel=force_slow_decel or reset_state,
      a_lead=float(lead_one.aLeadK),
      a_lead_tau=float(lead_one.aLeadTau),
      model_predicted_v_lead=model_predicted_v_lead,
      model_predicted_gap_opening=model_predicted_gap_opening,
    ) if lead_one.status else (False, 0.0)
    if self.creep_to_stop_gap_active:
      if creep_a_target >= 0.0:
        if not self.output_should_stop:
          output_a_target = max(output_a_target, creep_a_target)
      else:
        output_a_target = min(output_a_target, creep_a_target)
      creep_accel_max = CREEP_TO_STOP_GAP_PULLAWAY_ACCEL_MAX if creep_a_target > CREEP_TO_STOP_GAP_ACCEL_MAX else CREEP_TO_STOP_GAP_ACCEL_MAX
      output_a_target = min(output_a_target, creep_accel_max)
      self.output_should_stop = self.output_should_stop or (creep_a_target <= 0.0 and v_ego < self.CP.vEgoStopping)

    gap_fill_active, gap_fill_a_target = get_stopped_lead_gap_fill_accel(
      v_ego, float(lead_one.dRel), float(lead_one.vLeadK), float(lead_one.modelProb),
      lead_one.status and self.stopped_lead_gap_fill_timer > 0.0,
      brake_pressed=sm['carState'].brakePressed,
      gas_pressed=sm['carState'].gasPressed,
      force_slow_decel=force_slow_decel or reset_state,
      a_lead=float(lead_one.aLeadK),
    ) if lead_one.status else (False, 0.0)
    if gap_fill_active:
      if gap_fill_a_target >= 0.0:
        if not self.output_should_stop:
          output_a_target = max(output_a_target, gap_fill_a_target)
      else:
        output_a_target = min(output_a_target, gap_fill_a_target)
      output_a_target = min(output_a_target, STOPPED_LEAD_GAP_FILL_ACCEL_MAX)
      self.output_should_stop = self.output_should_stop or (gap_fill_a_target <= 0.0 and v_ego < self.CP.vEgoStopping)

    if lead_one.status and not self.output_should_stop and not reset_state and self.mpc.source != LongitudinalPlanSource.e2e and \
       self.source == custom.LongitudinalPlanSP.LongitudinalPlanSource.cruise:
      recovery_a_min = get_lead_accel_recovery_a_min(
        v_ego, float(lead_one.vLeadK), float(lead_one.dRel), float(lead_one.aLeadK), get_T_FOLLOW(sm['selfdriveState'].personality)
      )
      output_a_target = max(output_a_target, recovery_a_min)

    model_predicted_pullaway = not creep_to_stop_gap_blocked(
      v_ego, float(lead_one.dRel), float(lead_one.vLeadK), float(lead_one.modelProb),
      sm['carState'].brakePressed, sm['carState'].gasPressed, force_slow_decel or reset_state,
      float(lead_one.aLeadK),
    ) and has_predicted_lead_pullaway(
      float(lead_one.dRel) - get_lead_stop_presentation_distance(v_ego, float(lead_one.vLeadK), float(lead_one.aLeadK), float(lead_one.modelProb)),
      model_predicted_v_lead, model_predicted_gap_opening,
    )
    if lead_one.status and not sm['carState'].brakePressed and not sm['carState'].gasPressed and not force_slow_decel and not reset_state:
      self.creep_stop_hold_released = should_release_creep_stop_hold(
        self.creep_stop_hold_released, v_ego, float(lead_one.dRel), float(lead_one.vLeadK), float(lead_one.aLeadK),
        model_predicted_pullaway, float(lead_one.modelProb),
      )
    else:
      self.creep_stop_hold_released = False
    if lead_one.status and should_hold_creep_to_stop_gap(
      v_ego, float(lead_one.dRel), float(lead_one.vLeadK), float(lead_one.aLeadK), model_predicted_pullaway,
      self.creep_stop_hold_released, float(lead_one.modelProb),
    ):
      output_a_target = min(output_a_target, CREEP_TO_STOP_GAP_ACCEL_MIN)
      self.output_should_stop = True

    if engage_stop_bootstrap_active:
      output_a_target = min(output_a_target, output_a_target_e2e)
      self.output_should_stop = self.output_should_stop or output_should_stop_e2e
      if output_a_target_e2e < output_a_target_mpc:
        self.mpc.source = LongitudinalPlanSource.e2e

    self.previous_lead_loss_status = bool(lead_one.status)
    self.previous_lead_loss_d_rel = float(lead_one.dRel) if lead_one.status else 0.0
    self.previous_lead_loss_model_prob = float(lead_one.modelProb) if lead_one.status else 0.0

    for idx in range(2):
      accel_clip[idx] = np.clip(accel_clip[idx], self.prev_accel_clip[idx] - 0.05, self.prev_accel_clip[idx] + 0.05)
    self.output_a_target = np.clip(output_a_target, accel_clip[0], accel_clip[1])
    self.prev_accel_clip = accel_clip

  def publish(self, sm, pm):
    plan_send = messaging.new_message('longitudinalPlan')

    plan_send.valid = sm.all_checks(service_list=['carState', 'controlsState', 'selfdriveState', 'radarState'])

    longitudinalPlan = plan_send.longitudinalPlan
    longitudinalPlan.modelMonoTime = sm.logMonoTime['modelV2']
    longitudinalPlan.processingDelay = (plan_send.logMonoTime / 1e9) - sm.logMonoTime['modelV2']
    longitudinalPlan.solverExecutionTime = self.mpc.solve_time

    longitudinalPlan.speeds = self.v_desired_trajectory.tolist()
    longitudinalPlan.accels = self.a_desired_trajectory.tolist()
    longitudinalPlan.jerks = self.j_desired_trajectory.tolist()

    longitudinalPlan.hasLead = sm['radarState'].leadOne.status
    longitudinalPlan.longitudinalPlanSource = self.mpc.source
    longitudinalPlan.fcw = self.fcw

    longitudinalPlan.aTarget = float(self.output_a_target)
    longitudinalPlan.shouldStop = bool(self.output_should_stop)
    longitudinalPlan.allowBrake = True
    longitudinalPlan.allowThrottle = bool(self.allow_throttle)

    pm.send('longitudinalPlan', plan_send)

    self.publish_longitudinal_plan_sp(sm, pm)
