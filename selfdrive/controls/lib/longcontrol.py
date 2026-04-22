import numpy as np
from cereal import car
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N
from openpilot.common.pid import PIDController
from openpilot.selfdrive.modeld.constants import ModelConstants

CONTROL_N_T_IDX = ModelConstants.T_IDXS[:CONTROL_N]
LAUNCH_ENVELOPE_MIN_ACCEL = 0.15
LAUNCH_ENVELOPE_MAX_ACCEL = 0.35
LAUNCH_BREAKAWAY_ACCEL = 0.4
LAUNCH_BREAKAWAY_TIME = 0.25
LAUNCH_BREAKAWAY_V_EGO = 0.2
LAUNCH_ENVELOPE_TIME_BP = [0.0, 0.5]
LAUNCH_ENVELOPE_V_EGO_BP = [0.0, 0.6]

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
  taper_elapsed = max(launch_elapsed - LAUNCH_BREAKAWAY_TIME, 0.0)
  speed_blend = np.interp(v_ego, LAUNCH_ENVELOPE_V_EGO_BP, [1.0, 0.0])
  time_blend = np.interp(taper_elapsed, LAUNCH_ENVELOPE_TIME_BP, [1.0, 0.0])
  return min(speed_blend, time_blend)


def launch_breakaway_active(v_ego, launch_elapsed):
  return v_ego < LAUNCH_BREAKAWAY_V_EGO and launch_elapsed < LAUNCH_BREAKAWAY_TIME


def apply_launch_envelope(output_accel, accel_limits, v_ego, launch_elapsed):
  if output_accel <= 0.0:
    return float(output_accel)

  if launch_breakaway_active(v_ego, launch_elapsed):
    return float(np.clip(LAUNCH_BREAKAWAY_ACCEL, 0.0, accel_limits[1]))

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
    self.pid = PIDController((CP.longitudinalTuning.kpBP, CP.longitudinalTuning.kpV), (CP.longitudinalTuning.kiBP, CP.longitudinalTuning.kiV), rate=1 / DT_CTRL)
    self.last_output_accel = 0.0
    self.launch_envelope_active = False
    self.launch_elapsed = 0.0

  def reset_launch_envelope(self):
    self.launch_envelope_active = False
    self.launch_elapsed = 0.0

  def reset(self):
    self.pid.reset()
    self.reset_launch_envelope()

  def update(self, active, CS, a_target, should_stop, accel_limits):
    """Update longitudinal control. This updates the state machine and runs a PID loop"""
    self.pid.neg_limit = accel_limits[0]
    self.pid.pos_limit = accel_limits[1]

    prev_state = self.long_control_state
    self.long_control_state = long_control_state_trans(
      self.CP, self.CP_SP, active, self.long_control_state, CS.vEgo, should_stop, CS.brakePressed, CS.cruiseState.standstill
    )
    if not active or CS.brakePressed or self.long_control_state in (LongCtrlState.off, LongCtrlState.stopping):
      self.reset_launch_envelope()
    elif prev_state == LongCtrlState.stopping and self.long_control_state in (LongCtrlState.starting, LongCtrlState.pid) and not should_stop:
      self.launch_envelope_active = True
      self.launch_elapsed = 0.0

    if self.long_control_state == LongCtrlState.off:
      self.reset()
      output_accel = 0.0

    elif self.long_control_state == LongCtrlState.stopping:
      output_accel = self.last_output_accel
      if output_accel > self.CP.stopAccel:
        output_accel = min(output_accel, 0.0)
        output_accel -= self.CP.stoppingDecelRate * DT_CTRL
      self.pid.reset()

    elif self.long_control_state == LongCtrlState.starting:
      output_accel = self.CP.startAccel
      self.pid.reset()

    else:  # LongCtrlState.pid
      error = a_target - CS.aEgo
      output_accel = self.pid.update(error, speed=CS.vEgo, feedforward=a_target)

    if self.launch_envelope_active:
      output_accel = apply_launch_envelope(output_accel, accel_limits, CS.vEgo, self.launch_elapsed)
      if get_launch_envelope_blend(CS.vEgo, self.launch_elapsed) <= 0.0:
        self.reset_launch_envelope()
      else:
        self.launch_elapsed += DT_CTRL

    self.last_output_accel = np.clip(output_accel, accel_limits[0], accel_limits[1])
    return self.last_output_accel
