import numpy as np
from cereal import car
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N
from openpilot.common.pid import PIDController
from openpilot.selfdrive.modeld.constants import ModelConstants

CONTROL_N_T_IDX = ModelConstants.T_IDXS[:CONTROL_N]
LAUNCH_ENVELOPE_MIN_ACCEL = 0.15
LAUNCH_ENVELOPE_MAX_ACCEL = 0.60
LAUNCH_BREAKAWAY_ACCEL = 0.70
LAUNCH_BREAKAWAY_BASE_ACCEL = 0.65
LAUNCH_BREAKAWAY_TARGET_ACCEL = 0.50
LAUNCH_BREAKAWAY_BASE_TARGET_ACCEL = 0.30
LAUNCH_BREAKAWAY_MIN_TIME = 0.25
LAUNCH_BREAKAWAY_MAX_TIME = 0.6
LAUNCH_BREAKAWAY_A_EGO = 0.05
LAUNCH_BREAKAWAY_V_EGO = 0.2
LAUNCH_SHOULD_STOP_HOLD_TIME = LAUNCH_BREAKAWAY_MAX_TIME
SOFT_STOP_ACCEL = -0.8
SOFT_STOP_V_EGO = 0.05
LAUNCH_ENVELOPE_TIME_BP = [0.0, 0.35]
LAUNCH_ENVELOPE_V_EGO_BP = [0.0, 0.6]
LAUNCH_BREAKAWAY_TARGET_BP = [LAUNCH_ENVELOPE_MIN_ACCEL, LAUNCH_BREAKAWAY_BASE_TARGET_ACCEL, LAUNCH_BREAKAWAY_TARGET_ACCEL]
LAUNCH_BREAKAWAY_ACCEL_BP = [LAUNCH_ENVELOPE_MIN_ACCEL, LAUNCH_BREAKAWAY_BASE_ACCEL, LAUNCH_BREAKAWAY_ACCEL]

LongCtrlState = car.CarControl.Actuators.LongControlState


def long_control_state_trans(CP, CP_SP, active, long_control_state, v_ego, should_stop, brake_pressed, cruise_standstill):
  # Gas Interceptor
  cruise_standstill = cruise_standstill and not CP_SP.enableGasInterceptor

  stopping_condition = should_stop
  starting_condition = not should_stop and not cruise_standstill and not brake_pressed
  started_condition = v_ego > CP.vEgoStarting

  if not active:
    long_control_state = LongCtrlState.off

  else:
    if long_control_state == LongCtrlState.off:
      if not starting_condition:
        long_control_state = LongCtrlState.stopping
      else:
        if starting_condition and CP.startingState:
          long_control_state = LongCtrlState.starting
        else:
          long_control_state = LongCtrlState.pid

    elif long_control_state == LongCtrlState.stopping:
      if starting_condition and CP.startingState:
        long_control_state = LongCtrlState.starting
      elif starting_condition:
        long_control_state = LongCtrlState.pid

    elif long_control_state in [LongCtrlState.starting, LongCtrlState.pid]:
      if stopping_condition:
        long_control_state = LongCtrlState.stopping
      elif started_condition:
        long_control_state = LongCtrlState.pid
  return long_control_state


def get_launch_envelope_blend(v_ego, launch_elapsed):
  speed_blend = np.interp(v_ego, LAUNCH_ENVELOPE_V_EGO_BP, [1.0, 0.0])
  time_blend = np.interp(launch_elapsed, LAUNCH_ENVELOPE_TIME_BP, [1.0, 0.0])
  return min(speed_blend, time_blend)


def launch_breakaway_active(v_ego, a_ego, launch_elapsed):
  if v_ego >= LAUNCH_BREAKAWAY_V_EGO or launch_elapsed >= LAUNCH_BREAKAWAY_MAX_TIME:
    return False
  return launch_elapsed < LAUNCH_BREAKAWAY_MIN_TIME or a_ego < LAUNCH_BREAKAWAY_A_EGO


def get_launch_breakaway_accel(a_target, accel_limits):
  if a_target < LAUNCH_ENVELOPE_MIN_ACCEL:
    return 0.0
  breakaway_accel = np.interp(max(a_target, 0.0), LAUNCH_BREAKAWAY_TARGET_BP, LAUNCH_BREAKAWAY_ACCEL_BP)
  return float(np.clip(breakaway_accel, 0.0, accel_limits[1]))


def launch_should_stop_hold_active(v_ego, a_ego, brake_pressed, launch_elapsed, a_target):
  return not brake_pressed and a_target > LAUNCH_ENVELOPE_MIN_ACCEL and v_ego < LAUNCH_BREAKAWAY_V_EGO and \
    a_ego < LAUNCH_BREAKAWAY_A_EGO and launch_elapsed < LAUNCH_SHOULD_STOP_HOLD_TIME


def apply_launch_envelope(output_accel, accel_limits, v_ego, launch_elapsed, blend=None):
  if output_accel <= 0.0:
    return float(output_accel)

  if blend is None:
    blend = get_launch_envelope_blend(v_ego, launch_elapsed)
  if blend <= 0.0:
    return float(output_accel)

  launch_cap = accel_limits[1] - (accel_limits[1] - min(accel_limits[1], LAUNCH_ENVELOPE_MAX_ACCEL)) * blend
  launch_floor = min(LAUNCH_ENVELOPE_MIN_ACCEL * blend, launch_cap)
  return float(np.clip(output_accel, launch_floor, launch_cap))


class LongControl:
  def __init__(self, CP, CP_SP):
    self.CP = CP
    self.CP_SP = CP_SP
    self.long_control_state = LongCtrlState.off
    self.pid = PIDController((CP.longitudinalTuning.kpBP, CP.longitudinalTuning.kpV), (CP.longitudinalTuning.kiBP, CP.longitudinalTuning.kiV), rate=int(1 / DT_CTRL))
    self.last_output_accel = 0.0
    self.launch_envelope_active = False
    self.launch_breakaway_elapsed = 0.0
    self.launch_taper_elapsed = 0.0
    self.launch_breakaway_done = False

  def reset_launch_envelope(self):
    self.launch_envelope_active = False
    self.launch_breakaway_elapsed = 0.0
    self.launch_taper_elapsed = 0.0
    self.launch_breakaway_done = False

  def reset(self):
    self.pid.reset()
    self.reset_launch_envelope()

  def update(self, active, CS, a_target, should_stop, accel_limits, has_lead=False):
    """Update longitudinal control. This updates the state machine and runs a PID loop"""
    self.pid.neg_limit = accel_limits[0]
    self.pid.pos_limit = accel_limits[1]
    launch_a_target = max(a_target, LAUNCH_ENVELOPE_MIN_ACCEL) if a_target >= 0.0 else a_target

    effective_should_stop = should_stop and not (
      has_lead and self.launch_envelope_active and
      launch_should_stop_hold_active(CS.vEgo, CS.aEgo, CS.brakePressed, self.launch_breakaway_elapsed, launch_a_target)
    )
    prev_state = self.long_control_state
    self.long_control_state = long_control_state_trans(
      self.CP, self.CP_SP, active, self.long_control_state, CS.vEgo, effective_should_stop, CS.brakePressed, CS.cruiseState.standstill
    )
    if not active or CS.brakePressed or self.long_control_state in (LongCtrlState.off, LongCtrlState.stopping):
      self.reset_launch_envelope()
    elif prev_state == LongCtrlState.stopping and self.long_control_state in (LongCtrlState.starting, LongCtrlState.pid) and \
         not effective_should_stop and a_target >= 0.0:
      self.launch_envelope_active = True
      self.launch_breakaway_elapsed = 0.0
      self.launch_taper_elapsed = 0.0
      self.launch_breakaway_done = False

    if self.long_control_state == LongCtrlState.off:
      self.reset()
      output_accel = 0.0

    elif self.long_control_state == LongCtrlState.stopping:
      output_accel = self.last_output_accel
      stop_accel = self.CP.stopAccel
      if CS.vEgo > SOFT_STOP_V_EGO and not CS.cruiseState.standstill:
        stop_accel = max(stop_accel, SOFT_STOP_ACCEL)
      if output_accel <= stop_accel:
        output_accel = stop_accel
      else:
        output_accel = min(output_accel, 0.0)
        output_accel = max(output_accel - self.CP.stoppingDecelRate * DT_CTRL, stop_accel)
      self.pid.reset()

    elif self.long_control_state == LongCtrlState.starting:
      output_accel = a_target if has_lead or a_target < LAUNCH_ENVELOPE_MIN_ACCEL else self.CP.startAccel
      self.pid.reset()

    else:  # LongCtrlState.pid
      error = a_target - CS.aEgo
      output_accel = self.pid.update(error, speed=CS.vEgo, feedforward=a_target)

    if self.launch_envelope_active:
      if a_target < 0.0:
        self.reset_launch_envelope()
      elif not self.launch_breakaway_done and launch_breakaway_active(CS.vEgo, CS.aEgo, self.launch_breakaway_elapsed):
        output_accel = get_launch_breakaway_accel(launch_a_target, accel_limits)
        self.launch_breakaway_elapsed += DT_CTRL
      else:
        self.launch_breakaway_done = True
        launch_blend = get_launch_envelope_blend(CS.vEgo, self.launch_taper_elapsed)
        output_accel = max(output_accel, LAUNCH_ENVELOPE_MIN_ACCEL * launch_blend)
        output_accel = apply_launch_envelope(output_accel, accel_limits, CS.vEgo, self.launch_taper_elapsed, launch_blend)
        if launch_blend <= 0.0:
          self.reset_launch_envelope()
        else:
          self.launch_taper_elapsed += DT_CTRL

    self.last_output_accel = np.clip(output_accel, accel_limits[0], accel_limits[1])
    return self.last_output_accel
