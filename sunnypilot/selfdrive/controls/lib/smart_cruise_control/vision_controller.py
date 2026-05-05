"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

import numpy as np

import cereal.messaging as messaging
from cereal import custom
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.car.cruise import V_CRUISE_UNSET
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.sunnypilot import PARAMS_UPDATE_PERIOD
from openpilot.sunnypilot.selfdrive.controls.lib.smart_cruise_control import MIN_V

VisionState = custom.LongitudinalPlanSP.SmartCruiseControl.VisionState

ACTIVE_STATES = (VisionState.entering, VisionState.turning, VisionState.leaving)
ENABLED_STATES = (VisionState.enabled, VisionState.overriding, *ACTIVE_STATES)

_ENTERING_PRED_LAT_ACC_TH = 1.3  # Predicted Lat Acc threshold to trigger entering turn state.
_ABORT_ENTERING_PRED_LAT_ACC_TH = 1.1  # Predicted Lat Acc threshold to abort entering state if speed drops.

_TURNING_LAT_ACC_TH = 1.6  # Lat Acc threshold to trigger turning state.

_LEAVING_LAT_ACC_TH = 1.3  # Lat Acc threshold to trigger leaving turn state.
_FINISH_LAT_ACC_TH = 1.1  # Lat Acc threshold to trigger the end of the turn cycle.

_A_LAT_REG_MAX = 2.0  # Maximum lateral acceleration
_IN_TURN_LAT_ACC_TARGET = 3.0  # ISO 11270 lateral accel budget for confirmed turns.
_IN_TURN_LAT_ACC_RAMP_RATE = 2.0  # m/s^2 per second. Reach the in-turn budget in about 0.5s.
_CURRENT_LAT_ACC_BLEED_TH = 2.8
_UPCOMING_PRED_LAT_ACC_MARGIN = 0.4  # Treat stronger predicted accel as the next bend, not the current curve.

_NO_OVERSHOOT_TIME_HORIZON = 2.5  # s. Time to use for velocity desired based on a_target when not overshooting.

# Lookup table for the minimum smooth deceleration during the ENTERING state
# depending on the actual maximum absolute lateral acceleration predicted on the turn ahead.
_ENTERING_SMOOTH_DECEL_V = [-0.15, -0.7]  # min decel value allowed on ENTERING state
_ENTERING_SMOOTH_DECEL_BP = [1.3, 3.0]  # absolute value of lat acc ahead

# Lookup table for the acceleration for the TURNING state
# depending on the current lateral acceleration of the vehicle.
_TURNING_ACC_V = [0.5, 0.15, -0.15]  # acc value
_TURNING_ACC_BP = [1.5, 2.3, 3.0]  # absolute value of current lat acc
_CURRENT_LAT_ACC_BLEED_V = [-0.05, -0.20, -0.40]
_CURRENT_LAT_ACC_BLEED_BP = [2.8, 3.0, 3.4]

_LEAVING_ACC = 0.5  # Conformable acceleration to regain speed while leaving a turn.


class SmartCruiseControlVision:
  v_target: float = 0
  a_target: float = 0.0
  v_ego: float = 0.0
  a_ego: float = 0.0
  output_v_target: float = V_CRUISE_UNSET
  output_a_target: float = 0.0

  def __init__(self):
    self.params = Params()
    self.frame = -1
    self.long_enabled = False
    self.long_override = False
    self.is_enabled = False
    self.is_active = False
    self.enabled = self.params.get_bool("SmartCruiseControlVision")
    self.v_cruise_setpoint = 0.0

    self.state = VisionState.disabled
    self.current_lat_acc = 0.0
    self.current_curvature = 0.0
    self.current_lat_acc_bleed = False
    self.max_pred_lat_acc = 0.0
    self.predicted_turn_time = 0.0
    self.entering_v_target_valid = False
    self.in_turn_lat_acc_budget = _A_LAT_REG_MAX

  def get_a_target_from_control(self) -> float:
    return self.a_target

  def get_v_target_from_control(self) -> float:
    if self.is_active:
      if self.state == VisionState.entering and self.entering_v_target_valid:
        return max(self.v_target, MIN_V)
      return max(self.v_target, MIN_V) + self.a_target * _NO_OVERSHOOT_TIME_HORIZON

    return V_CRUISE_UNSET

  def _update_params(self) -> None:
    if self.frame % int(PARAMS_UPDATE_PERIOD / DT_MDL) == 0:
      self.enabled = self.params.get_bool("SmartCruiseControlVision")

  def _update_calculations(self, sm: messaging.SubMaster) -> None:
    if not self.long_enabled:
      return
    else:
      rate_plan = np.array(np.abs(sm['modelV2'].orientationRate.z))
      vel_plan = np.array(sm['modelV2'].velocity.x)

      self.current_curvature = abs(sm['controlsState'].curvature)
      self.current_lat_acc = self.v_ego**2 * self.current_curvature
      self.current_lat_acc_bleed = self.current_lat_acc >= _CURRENT_LAT_ACC_BLEED_TH

      # get the maximum lat accel from the model
      predicted_lat_accels = rate_plan * vel_plan
      self.max_pred_lat_acc = np.percentile(predicted_lat_accels, 97)
      turn_idxs = np.nonzero(predicted_lat_accels >= _TURNING_LAT_ACC_TH)[0]
      self.predicted_turn_time = float(ModelConstants.T_IDXS[int(turn_idxs[0])]) if len(turn_idxs) > 0 else 0.0

  def _speed_for_lateral_accel(self, lateral_accel: float, curvature: float) -> float:
    if curvature <= 1e-6:
      return V_CRUISE_UNSET
    return (lateral_accel / curvature) ** 0.5

  def _update_in_turn_lat_acc_budget(self) -> None:
    if self.state == VisionState.turning:
      self.in_turn_lat_acc_budget = min(_IN_TURN_LAT_ACC_TARGET,
                                        self.in_turn_lat_acc_budget + _IN_TURN_LAT_ACC_RAMP_RATE * DT_MDL)
    elif self.state == VisionState.leaving and self.max_pred_lat_acc >= _ENTERING_PRED_LAT_ACC_TH:
      self.in_turn_lat_acc_budget = max(self.in_turn_lat_acc_budget, _A_LAT_REG_MAX)
    else:
      self.in_turn_lat_acc_budget = _A_LAT_REG_MAX

  def _update_v_target(self) -> None:
    self.entering_v_target_valid = False
    v_ego = max(self.v_ego, 0.1)  # ensure a value greater than 0 for calculations
    predicted_curve = self.max_pred_lat_acc / (v_ego**2)
    predicted_v_target = self._speed_for_lateral_accel(_A_LAT_REG_MAX, predicted_curve)

    if self.state == VisionState.entering and self.predicted_turn_time > DT_MDL and predicted_curve > 1e-6:
      entering_decel = self._entering_smooth_decel()
      conservative_v_target = self._speed_for_lateral_accel(_A_LAT_REG_MAX, predicted_curve)
      iso_v_target = min(self._speed_for_lateral_accel(_IN_TURN_LAT_ACC_TARGET, predicted_curve), self.v_ego)
      reachable_v_target = self.v_ego + entering_decel * self.predicted_turn_time

      self.v_target = float(np.clip(reachable_v_target, conservative_v_target, iso_v_target))
      self.entering_v_target_valid = True
    elif self.state == VisionState.turning and self.current_curvature > 1e-6:
      current_v_target = self._speed_for_lateral_accel(self.in_turn_lat_acc_budget, self.current_curvature)
      if self.current_lat_acc_bleed:
        current_v_target = min(current_v_target, self._speed_for_lateral_accel(_CURRENT_LAT_ACC_BLEED_TH, self.current_curvature))
      if self.max_pred_lat_acc > self.current_lat_acc + _UPCOMING_PRED_LAT_ACC_MARGIN:
        self.v_target = min(current_v_target, predicted_v_target)
      else:
        self.v_target = current_v_target
    elif self.state == VisionState.leaving and self.max_pred_lat_acc >= _ENTERING_PRED_LAT_ACC_TH:
      self.v_target = min(predicted_v_target, self.v_ego)
    else:
      self.v_target = predicted_v_target

  def _entering_smooth_decel(self) -> float:
    return float(np.interp(self.max_pred_lat_acc, _ENTERING_SMOOTH_DECEL_BP, _ENTERING_SMOOTH_DECEL_V))

  def _update_state_machine(self) -> tuple[bool, bool]:
    # ENABLED, ENTERING, TURNING, LEAVING, OVERRIDING
    if self.state != VisionState.disabled:
      # longitudinal and feature disable always have priority in a non-disabled state
      if not self.long_enabled or not self.enabled:
        self.state = VisionState.disabled
      elif self.long_override:
        self.state = VisionState.overriding

      else:
        # ENABLED
        if self.state == VisionState.enabled:
          # Do not enter a turn control cycle if the speed is low.
          if self.v_ego <= MIN_V:
            pass
          # If significant lateral acceleration is predicted ahead, then move to Entering turn state.
          elif self.max_pred_lat_acc >= _ENTERING_PRED_LAT_ACC_TH:
            self.state = VisionState.entering
          # If the current measured curve is already near the lateral limit, start shedding speed even if prediction missed it.
          elif self.current_lat_acc_bleed:
            self.state = VisionState.turning

        # OVERRIDING
        elif self.state == VisionState.overriding:
          if not self.long_override:
            self.state = VisionState.enabled

        # ENTERING
        elif self.state == VisionState.entering:
          # Transition to Turning if current lateral acceleration is over the threshold.
          if self.current_lat_acc >= _TURNING_LAT_ACC_TH:
            self.state = VisionState.turning
          # Abort if the predicted lateral acceleration drops
          elif self.max_pred_lat_acc < _ABORT_ENTERING_PRED_LAT_ACC_TH:
            self.state = VisionState.enabled

        # TURNING
        elif self.state == VisionState.turning:
          # Transition to Leaving if current lateral acceleration drops below a threshold.
          if self.current_lat_acc <= _LEAVING_LAT_ACC_TH:
            self.state = VisionState.leaving

        # LEAVING
        elif self.state == VisionState.leaving:
          # Transition back to Turning if current lateral acceleration goes back over the threshold.
          if self.current_lat_acc >= _TURNING_LAT_ACC_TH:
            self.state = VisionState.turning
          # Finish if current lateral acceleration goes below a threshold.
          elif self.current_lat_acc < _FINISH_LAT_ACC_TH:
            self.state = VisionState.enabled

    # DISABLED
    elif self.state == VisionState.disabled:
      if self.long_enabled and self.enabled:
        if self.long_override:
          self.state = VisionState.overriding
        else:
          self.state = VisionState.enabled

    enabled = self.state in ENABLED_STATES
    active = self.state in ACTIVE_STATES

    return enabled, active

  def _update_solution(self) -> float:
    # DISABLED, ENABLED, OVERRIDING
    if self.state not in ACTIVE_STATES:
      # when not overshooting, calculate v_turn as the speed at the prediction horizon when following
      # the smooth deceleration.
      a_target = self.a_ego
    # ENTERING
    elif self.state == VisionState.entering:
      # when not overshooting, target a smooth deceleration in preparation for a sharp turn to come.
      a_target = self._entering_smooth_decel()
      if self.entering_v_target_valid:
        a_target = max((self.v_target - self.v_ego) / self.predicted_turn_time, a_target)
    # TURNING
    elif self.state == VisionState.turning:
      # When turning, we provide a target acceleration that is comfortable for the lateral acceleration felt.
      a_target = np.interp(self.current_lat_acc, _TURNING_ACC_BP, _TURNING_ACC_V)
    # LEAVING
    elif self.state == VisionState.leaving:
      # When leaving, we provide a comfortable acceleration to regain speed.
      a_target = _LEAVING_ACC
      if self.max_pred_lat_acc >= _ENTERING_PRED_LAT_ACC_TH:
        a_target = min(0.0, np.interp(self.max_pred_lat_acc, _ENTERING_SMOOTH_DECEL_BP, _ENTERING_SMOOTH_DECEL_V))
    else:
      raise NotImplementedError(f"SCC-V state not supported: {self.state}")

    if self.current_lat_acc_bleed:
      a_target = min(a_target, np.interp(self.current_lat_acc, _CURRENT_LAT_ACC_BLEED_BP, _CURRENT_LAT_ACC_BLEED_V))

    return a_target

  def update(self, sm: messaging.SubMaster, long_enabled: bool, long_override: bool, v_ego: float, a_ego: float, v_cruise_setpoint: float) -> None:
    self.long_enabled = long_enabled
    self.long_override = long_override
    self.v_ego = v_ego
    self.a_ego = a_ego
    self.v_cruise_setpoint = v_cruise_setpoint

    self._update_params()
    self._update_calculations(sm)

    self.is_enabled, self.is_active = self._update_state_machine()
    self._update_in_turn_lat_acc_budget()
    self._update_v_target()
    self.a_target = self._update_solution()

    self.output_v_target = self.get_v_target_from_control()
    self.output_a_target = self.get_a_target_from_control()

    self.frame += 1
