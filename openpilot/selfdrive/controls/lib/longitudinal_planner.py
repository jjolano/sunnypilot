#!/usr/bin/env python3
import math
import numpy as np

import openpilot.cereal.messaging as messaging
from opendbc.car.interfaces import ACCEL_MIN, ACCEL_MAX
from opendbc.car.vehicle_model import VehicleModel
from openpilot.common.constants import CV
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import LongitudinalMpc, LongitudinalPlanSource
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import T_IDXS as T_IDXS_MPC
from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N, get_accel_from_plan
from openpilot.selfdrive.car.cruise import V_CRUISE_MAX, V_CRUISE_UNSET
from openpilot.common.swaglog import cloudlog
from openpilot.common.params import Params

from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlannerSP
from openpilot.sunnypilot.custom.longitudinal.research_actuation import research_actuation_allowed

A_CRUISE_MAX_VALS = [1.6, 1.2, 0.8, 0.6]
A_CRUISE_MAX_BP = [0., 10.0, 25., 40.]
CONTROL_N_T_IDX = ModelConstants.T_IDXS[:CONTROL_N]
ALLOW_THROTTLE_THRESHOLD = 0.4
MIN_ALLOW_THROTTLE_SPEED = 2.5
GAS_OVERRIDE_COAST_EPS = 0.5  # m/s; hysteresis for coast-after-gas-override
GAS_OVERRIDE_COAST_MAX_S = 10.0  # maximum post-release coast window

# Lookup table for turns
_A_TOTAL_MAX_V = [1.7, 3.2]
_A_TOTAL_MAX_BP = [20., 40.]

def get_max_accel(v_ego):
  return np.interp(v_ego, A_CRUISE_MAX_BP, A_CRUISE_MAX_VALS)

def get_coast_accel(pitch):
  return np.sin(pitch) * -5.65 - 0.3  # fitted from data using xx/projects/allow_throttle/compute_coast_accel.py

def limit_accel_in_turns(v_ego, angle_steers, a_target, VM, roll):
  """
  This function returns a limited long acceleration allowed, depending on the existing lateral acceleration
  this should avoid accelerating when losing the target in turns
  """
  a_total_max = np.interp(v_ego, _A_TOTAL_MAX_BP, _A_TOTAL_MAX_V)
  steer_angle_rad = angle_steers * CV.DEG_TO_RAD
  curvature = VM.calc_curvature(steer_angle_rad, v_ego, roll)
  a_y = v_ego ** 2 * curvature
  a_x_allowed = math.sqrt(max(a_total_max ** 2 - a_y ** 2, 0.))

  return [a_target[0], min(a_target[1], a_x_allowed)]


class LongitudinalPlanner(LongitudinalPlannerSP):
  def __init__(self, CP, CP_SP, init_v=0.0, init_a=0.0, dt=DT_MDL):
    self.CP = CP
    self.VM = VehicleModel(self.CP)
    self.mpc = LongitudinalMpc(dt=dt)
    LongitudinalPlannerSP.__init__(self, self.CP, CP_SP, self.mpc)
    self.fcw = False
    self.dt = dt
    self.allow_throttle = True

    self.a_desired = init_a
    self.v_desired_filter = FirstOrderFilter(init_v, 2.0, self.dt)
    self.prev_accel_clip = [ACCEL_MIN, ACCEL_MAX]
    self.output_a_target = 0.0
    self.output_should_stop = False

    self.v_desired_trajectory = np.zeros(CONTROL_N)
    self.a_desired_trajectory = np.zeros(CONTROL_N)
    self.j_desired_trajectory = np.zeros(CONTROL_N)

    # Let speed coast back to the cruise target after driver gas override pushes ego above it,
    # instead of having the cruise virtual obstacle command active decel.
    self._gas_override_coast_active = False
    self._gas_override_coast_elapsed_s = 0.0

  def _update_gas_override_coast(self, gas_pressed, brake_pressed, force_slow_decel,
                                  v_ego, v_cruise, v_cruise_initialized):
    if not v_cruise_initialized or v_cruise <= 0.0 or force_slow_decel or brake_pressed:
      self._gas_override_coast_active = False
      self._gas_override_coast_elapsed_s = 0.0
      return
    if gas_pressed and v_ego > v_cruise:
      self._gas_override_coast_active = True
      self._gas_override_coast_elapsed_s = 0.0
    elif self._gas_override_coast_active:
      self._gas_override_coast_elapsed_s += max(float(self.dt), 0.0)
      if self._gas_override_coast_elapsed_s >= GAS_OVERRIDE_COAST_MAX_S:
        self._gas_override_coast_active = False
    if self._gas_override_coast_active and v_ego <= v_cruise + GAS_OVERRIDE_COAST_EPS:
      self._gas_override_coast_active = False
    if not self._gas_override_coast_active:
      self._gas_override_coast_elapsed_s = 0.0

  def _effective_v_cruise(self, v_cruise, v_ego):
    return max(v_cruise, v_ego) if self._gas_override_coast_active else v_cruise

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

    # No change cost when user is controlling the speed, or when standstill
    prev_accel_constraint = not (reset_state or sm['carState'].standstill)

    # Update VehicleModel with live parameters (mirrors controlsd)
    lp = sm['liveParameters']
    self.VM.update_params(max(lp.stiffnessFactor, 0.1), max(lp.steerRatio, 0.1))

    accel_clip = [ACCEL_MIN, get_max_accel(v_ego)]
    steer_angle_without_offset = sm['carState'].steeringAngleDeg - lp.angleOffsetDeg
    accel_clip = limit_accel_in_turns(v_ego, steer_angle_without_offset, accel_clip, self.VM, lp.roll)

    if reset_state:
      self.v_desired_filter.x = v_ego
      # Clip aEgo to cruise limits to prevent large accelerations when becoming active
      self.a_desired = np.clip(sm['carState'].aEgo, accel_clip[0], accel_clip[1])

    # Prevent divergence, smooth in current v_ego
    self.v_desired_filter.x = max(0.0, self.v_desired_filter.update(v_ego))
    throttle_probs = sm['modelV2'].meta.disengagePredictions.gasPressProbs
    throttle_prob = throttle_probs[1] if len(throttle_probs) > 1 else 1.0
    # Don't clip at low speeds since throttle_prob doesn't account for creep
    self.allow_throttle = throttle_prob > ALLOW_THROTTLE_THRESHOLD or v_ego <= MIN_ALLOW_THROTTLE_SPEED

    if not self.allow_throttle:
      clipped_accel_coast = max(accel_coast, accel_clip[0])
      clipped_accel_coast_interp = np.interp(v_ego, [MIN_ALLOW_THROTTLE_SPEED, MIN_ALLOW_THROTTLE_SPEED*2], [accel_clip[1], clipped_accel_coast])
      accel_clip[1] = min(accel_clip[1], clipped_accel_coast_interp)

    # Mode-gated helpers (cut-out release, moving-lead cruise cap) share these planner
    # booleans; cache them once so the helpers do not reread Params on every call.
    custom_long_enabled = bool(self.custom_long.enabled)
    try:
      allow_longitudinal_research_actuation = bool(
        Params().get_bool("AllowLongitudinalResearchActuation"))
    except Exception:
      allow_longitudinal_research_actuation = False
    research_allowed = research_actuation_allowed(
      None, self.CP,
      custom_long_enabled=custom_long_enabled,
      allow_longitudinal_research_actuation=allow_longitudinal_research_actuation,
    )
    self.custom_long.research_actuation_allowed = research_allowed

    long_active_for_follow_gap = sm['carControl'].longActive and not reset_state and not long_control_off
    radar_state = sm['radarState']

    # Get new v_cruise and a_desired from Smart Cruise Control and Speed Limit Assist
    v_cruise, self.a_desired = LongitudinalPlannerSP.update_targets(
      self, sm, self.v_desired_filter.x, self.a_desired, v_cruise,
      refresh_custom_long=False)

    if force_slow_decel:
      v_cruise = 0.0

    self._update_gas_override_coast(
      sm['carState'].gasPressed, sm['carState'].brakePressed, force_slow_decel,
      v_ego, v_cruise, v_cruise_initialized,
    )
    v_cruise = self._effective_v_cruise(v_cruise, v_ego)

    self.mpc.set_weights(prev_accel_constraint, personality=sm['selfdriveState'].personality)
    self.mpc.set_cur_state(self.v_desired_filter.x, self.a_desired)
    # Cut-out lead release: MPC-input-only filter dropping a lead that has confidently
    # exited the path sideways (fail-closed to the raw radarState; research-gated apply).
    radar_state_for_mpc = self.cut_out_release.filtered(
      radar_state, v_ego, self.dt,
      long_active=long_active_for_follow_gap,
      custom_long_enabled=custom_long_enabled,
      research_actuation_allowed=research_allowed,
      mode=self.custom_long.cut_out_lead_release_mode,
      model_msg=sm['modelV2'],
    )
    # Moving-lead cruise cap: lower the MPC cruise obstacle before the solve on a mildly braking lead.
    v_cruise_for_mpc = self.moving_lead_cruise_cap.capped(
      radar_state, v_ego, v_cruise, self.dt,
      long_active=long_active_for_follow_gap,
      brake_pressed=sm['carState'].brakePressed,
      gas_pressed=sm['carState'].gasPressed,
      force_decel=force_slow_decel,
      custom_long_enabled=custom_long_enabled,
      research_actuation_allowed=research_allowed,
    )
    self.mpc.update(radar_state_for_mpc, v_cruise_for_mpc, personality=sm['selfdriveState'].personality)

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

    action_t =  self.CP.longitudinalActuatorDelay + DT_MDL
    output_a_target_mpc, output_should_stop_mpc = get_accel_from_plan(self.v_desired_trajectory, self.a_desired_trajectory, CONTROL_N_T_IDX,
                                                                        action_t=action_t)
    output_a_target_e2e = sm['modelV2'].action.desiredAcceleration
    output_should_stop_e2e = sm['modelV2'].action.shouldStop
    self._last_longitudinal_debug = {
      "v_cruise": float(v_cruise),
      "mpc_a_target": float(output_a_target_mpc),
      "mpc_should_stop": bool(output_should_stop_mpc),
      "model_a_target": float(output_a_target_e2e),
      "model_should_stop": bool(output_should_stop_e2e),
    }
    output_a_target, self.output_should_stop, e2e_source = self.final_longitudinal_output(
      sm, output_a_target_mpc, output_should_stop_mpc, output_a_target_e2e, output_should_stop_e2e)
    self._last_longitudinal_debug.update({
      "final_a_target_unclipped": float(output_a_target),
      "final_should_stop": bool(self.output_should_stop),
      "e2e_source": e2e_source,
    })
    if e2e_source:
      self.mpc.source = LongitudinalPlanSource.e2e

    for idx in range(2):
      accel_clip[idx] = np.clip(accel_clip[idx], self.prev_accel_clip[idx] - 0.05, self.prev_accel_clip[idx] + 0.05)
    self.output_a_target = np.clip(output_a_target, accel_clip[0], accel_clip[1])
    self._last_longitudinal_debug.update({
      "final_a_target_clipped": float(self.output_a_target),
      "accel_clip_min": float(accel_clip[0]),
      "accel_clip_max": float(accel_clip[1]),
    })
    self.prev_accel_clip = accel_clip

  def publish(self, sm, pm):
    plan_send = messaging.new_message('longitudinalPlan')

    plan_send.valid = sm.all_checks(service_list=['carControl', 'carState', 'carStateSP', 'controlsState', 'liveParameters', 'modelV2', 'radarState', 'selfdriveState', 'selfdriveStateSP'])

    longitudinalPlan = plan_send.longitudinalPlan
    longitudinalPlan.modelMonoTime = sm.logMonoTime['modelV2']
    longitudinalPlan.processingDelay = (plan_send.logMonoTime / 1e9) - sm.logMonoTime['modelV2']
    longitudinalPlan.solverExecutionTime = self.mpc.solve_time

    longitudinalPlan.speeds = self.v_desired_trajectory.tolist()
    longitudinalPlan.accels = self.a_desired_trajectory.tolist()
    longitudinalPlan.jerks = self.j_desired_trajectory.tolist()

    longitudinalPlan.hasLead = sm['radarState'].leadOne.present
    longitudinalPlan.longitudinalPlanSource = self.mpc.source
    longitudinalPlan.fcw = self.fcw

    longitudinalPlan.aTarget = float(self.output_a_target)
    longitudinalPlan.shouldStop = bool(self.output_should_stop)
    longitudinalPlan.allowBrake = True
    longitudinalPlan.allowThrottle = bool(self.allow_throttle)

    pm.send('longitudinalPlan', plan_send)

    self.publish_longitudinal_plan_sp(sm, pm)
