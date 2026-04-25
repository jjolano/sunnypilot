#!/usr/bin/env python3
import math
import numpy as np

import cereal.messaging as messaging
from opendbc.car.interfaces import ACCEL_MIN, ACCEL_MAX
from openpilot.common.constants import CV
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import LongitudinalMpc, LongitudinalPlanSource, STOP_DISTANCE
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import T_IDXS as T_IDXS_MPC
from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N, get_accel_from_plan
from openpilot.selfdrive.car.cruise import V_CRUISE_MAX, V_CRUISE_UNSET
from openpilot.common.swaglog import cloudlog

from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlannerSP

A_CRUISE_MAX_VALS = [1.6, 1.2, 0.8, 0.6]
A_CRUISE_MAX_BP = [0.0, 10.0, 25.0, 40.0]
CONTROL_N_T_IDX = ModelConstants.T_IDXS[:CONTROL_N]
ALLOW_THROTTLE_THRESHOLD = 0.4
MIN_ALLOW_THROTTLE_SPEED = 2.5
ENGAGE_STOP_BOOTSTRAP_TIME = 0.75
ENGAGE_STOP_BOOTSTRAP_MIN_SPEED = 5.0
ENGAGE_STOP_BOOTSTRAP_MODEL_ACCEL = -1.0
CREEP_TO_STOP_GAP_ARM_EXCESS = 0.5
CREEP_TO_STOP_GAP_STOP_EXCESS = 0.05
CREEP_TO_STOP_GAP_MAX_V_EGO_ARM = 0.3
CREEP_TO_STOP_GAP_MAX_V_EGO = 1.0
CREEP_TO_STOP_GAP_MAX_EXCESS = 10.0
CREEP_TO_STOP_GAP_MIN_LEAD_SPEED = -0.3
CREEP_TO_STOP_GAP_MIN_MODEL_PROB = 0.5
CREEP_TO_STOP_GAP_SPEED_MAX = 0.75
CREEP_TO_STOP_GAP_SPEED_BP = [CREEP_TO_STOP_GAP_STOP_EXCESS, 1.0, 5.0]
CREEP_TO_STOP_GAP_SPEED_V = [0.0, 0.25, CREEP_TO_STOP_GAP_SPEED_MAX]
CREEP_TO_STOP_GAP_ACCEL_GAIN = 0.8
CREEP_TO_STOP_GAP_ACCEL_MIN = -0.25
CREEP_TO_STOP_GAP_ACCEL_MAX = 0.18
CREEP_TO_STOP_GAP_HOLD_EXCESS = 0.1
CREEP_TO_STOP_GAP_PULLAWAY_MIN_LEAD_SPEED = 0.25
CREEP_TO_STOP_GAP_PULLAWAY_ARM_EXCESS = 0.5
CREEP_TO_STOP_GAP_PULLAWAY_SPEED_MAX = 1.2
CREEP_TO_STOP_GAP_PULLAWAY_ACCEL_MAX = 0.35

# Lookup table for turns
_A_TOTAL_MAX_V = [1.7, 3.2]
_A_TOTAL_MAX_BP = [20.0, 40.0]


def get_max_accel(v_ego):
  return np.interp(v_ego, A_CRUISE_MAX_BP, A_CRUISE_MAX_VALS)


def get_coast_accel(pitch):
  return np.sin(pitch) * -5.65 - 0.3  # fitted from data using xx/projects/allow_throttle/compute_coast_accel.py


def has_valid_radar_lead(radar_state):
  return radar_state.leadOne.status or radar_state.leadTwo.status


def should_run_engage_stop_bootstrap(timer, v_ego, radar_state, model_msg):
  if timer <= 0.0 or v_ego < ENGAGE_STOP_BOOTSTRAP_MIN_SPEED or has_valid_radar_lead(radar_state):
    return False

  return bool(model_msg.action.shouldStop or model_msg.action.desiredAcceleration <= ENGAGE_STOP_BOOTSTRAP_MODEL_ACCEL)


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


def get_creep_to_stop_gap_accel(v_ego, d_rel, v_lead, model_prob, active, brake_pressed=False, gas_pressed=False, force_slow_decel=False):
  gap_excess = d_rel - STOP_DISTANCE
  blocked = brake_pressed or gas_pressed or force_slow_decel or model_prob < CREEP_TO_STOP_GAP_MIN_MODEL_PROB
  blocked = blocked or v_lead < CREEP_TO_STOP_GAP_MIN_LEAD_SPEED or v_ego >= CREEP_TO_STOP_GAP_MAX_V_EGO
  blocked = blocked or gap_excess <= 0.0 or gap_excess > CREEP_TO_STOP_GAP_MAX_EXCESS
  if blocked:
    return False, 0.0

  lead_pullaway = v_lead >= CREEP_TO_STOP_GAP_PULLAWAY_MIN_LEAD_SPEED
  should_arm = gap_excess >= CREEP_TO_STOP_GAP_ARM_EXCESS and v_ego < CREEP_TO_STOP_GAP_MAX_V_EGO_ARM
  should_arm = should_arm or (lead_pullaway and gap_excess >= CREEP_TO_STOP_GAP_PULLAWAY_ARM_EXCESS)
  if not active and not should_arm:
    return False, 0.0

  target_speed = float(np.interp(gap_excess, CREEP_TO_STOP_GAP_SPEED_BP, CREEP_TO_STOP_GAP_SPEED_V))
  accel_max = CREEP_TO_STOP_GAP_ACCEL_MAX
  if lead_pullaway:
    target_speed = max(target_speed, min(v_lead, CREEP_TO_STOP_GAP_PULLAWAY_SPEED_MAX))
    accel_max = CREEP_TO_STOP_GAP_PULLAWAY_ACCEL_MAX
  accel = np.clip((target_speed - v_ego) * CREEP_TO_STOP_GAP_ACCEL_GAIN, CREEP_TO_STOP_GAP_ACCEL_MIN, accel_max)
  return True, float(accel)


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

    engage_stop_bootstrap_active = should_run_engage_stop_bootstrap(self.engage_stop_bootstrap_timer, v_ego, sm['radarState'], sm['modelV2'])
    if engage_stop_bootstrap_active:
      v_cruise = 0.0

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

    if self.is_e2e(sm):
      output_a_target = min(output_a_target_e2e, output_a_target_mpc)
      self.output_should_stop = output_should_stop_e2e or output_should_stop_mpc
      if output_a_target < output_a_target_mpc:
        self.mpc.source = LongitudinalPlanSource.e2e
    else:
      output_a_target = output_a_target_mpc
      self.output_should_stop = output_should_stop_mpc

    if engage_stop_bootstrap_active:
      output_a_target = min(output_a_target, output_a_target_e2e)
      self.output_should_stop = self.output_should_stop or output_should_stop_e2e
      if output_a_target_e2e < output_a_target_mpc:
        self.mpc.source = LongitudinalPlanSource.e2e

    lead_one = sm['radarState'].leadOne
    self.creep_to_stop_gap_active, creep_a_target = get_creep_to_stop_gap_accel(
      v_ego, float(lead_one.dRel), float(lead_one.vLeadK), float(lead_one.modelProb),
      self.creep_to_stop_gap_active and not reset_state,
      brake_pressed=sm['carState'].brakePressed,
      gas_pressed=sm['carState'].gasPressed,
      force_slow_decel=force_slow_decel or reset_state,
    ) if lead_one.status else (False, 0.0)
    if self.creep_to_stop_gap_active:
      if creep_a_target >= 0.0:
        output_a_target = max(output_a_target, creep_a_target)
      else:
        output_a_target = min(output_a_target, creep_a_target)
      creep_accel_max = CREEP_TO_STOP_GAP_PULLAWAY_ACCEL_MAX if creep_a_target > CREEP_TO_STOP_GAP_ACCEL_MAX else CREEP_TO_STOP_GAP_ACCEL_MAX
      output_a_target = min(output_a_target, creep_accel_max)
      self.output_should_stop = creep_a_target <= 0.0 and v_ego < self.CP.vEgoStopping

    if lead_one.status and v_ego < CREEP_TO_STOP_GAP_MAX_V_EGO and float(lead_one.vLeadK) < CREEP_TO_STOP_GAP_PULLAWAY_MIN_LEAD_SPEED and float(lead_one.aLeadK) <= 0.05:
      if float(lead_one.dRel) <= STOP_DISTANCE + CREEP_TO_STOP_GAP_HOLD_EXCESS:
        output_a_target = min(output_a_target, CREEP_TO_STOP_GAP_ACCEL_MIN)
        self.output_should_stop = True

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
