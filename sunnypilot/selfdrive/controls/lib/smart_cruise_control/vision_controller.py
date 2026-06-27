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

_CURRENT_LAT_ACC_BLEED_TH = 2.8  # High current lateral acceleration forces turning even if prediction is weak.

_LEAVING_LAT_ACC_TH = 1.3  # Lat Acc threshold to trigger leaving turn state.
_FINISH_LAT_ACC_TH = 1.1  # Lat Acc threshold to trigger the end of the turn cycle.

_A_LAT_REG_MAX = 2.  # Maximum lateral acceleration

_NO_OVERSHOOT_TIME_HORIZON = 4.  # s. Time to use for velocity desired based on a_target when not overshooting.

# Lookup table for the minimum smooth deceleration during the ENTERING state
# depending on the actual maximum absolute lateral acceleration predicted on the turn ahead.
_ENTERING_SMOOTH_DECEL_V = [-0.2, -1.]  # min decel value allowed on ENTERING state
_ENTERING_SMOOTH_DECEL_BP = [1.3, 3.]  # absolute value of lat acc ahead

_PRE_ENTRY_PRED_LAT_ACC_TH = 1.0  # Mild predicted lat accel band for guarded pre-entry.
_PRE_ENTRY_MIN_FRAMES = 3  # Require persistence before pre-entry activates.
_PRE_ENTRY_GENTLE_DECEL = -0.25  # Gentle lift/coast warning, not a hard brake.

# Lookup table for the acceleration for the TURNING state
# depending on the current lateral acceleration of the vehicle.
_TURNING_ACC_V = [0.5, 0., -0.4]  # acc value
_TURNING_ACC_BP = [1.5, 2.3, 3.]  # absolute value of current lat acc

_LEAVING_ACC = 0.5  # Conformable acceleration to regain speed while leaving a turn.
_EPS = 1e-6


class SmartCruiseControlVision:
  v_target: float = 0
  a_target: float = 0.
  v_ego: float = 0.
  a_ego: float = 0.
  output_v_target: float = V_CRUISE_UNSET
  output_a_target: float = 0.

  def __init__(self):
    self.params = Params()
    self.frame = -1
    self.long_enabled = False
    self.long_override = False
    self.is_enabled = False
    self.is_active = False
    self.enabled = self.params.get_bool("SmartCruiseControlVision")
    self.v_cruise_setpoint = 0.

    self.state = VisionState.disabled
    self.current_lat_acc = 0.
    self.max_pred_lat_acc = 0.
    self._prev_max_pred_lat_acc = 0.
    self.pre_entry_frames = 0
    self.pre_entry_active = False
    self._required_decel = 0.
    self._t_risk = 0.
    self._current_curvature = 0.

  def get_a_target_from_control(self) -> float:
    return self.a_target

  def _fail_closed(self) -> None:
    self.max_pred_lat_acc = 0.
    self.current_lat_acc = 0.
    self.v_target = 0.
    self._required_decel = 0.
    self._t_risk = 0.
    self._current_curvature = 0.
    self.pre_entry_frames = 0
    self.pre_entry_active = False
    self.state = VisionState.enabled if self.long_enabled and self.enabled else VisionState.disabled

  def get_v_target_from_control(self) -> float:
    if self.is_active:
      if self.state == VisionState.leaving and self.max_pred_lat_acc < _ENTERING_PRED_LAT_ACC_TH:
        return V_CRUISE_UNSET

      if self.state == VisionState.turning:
        # When already in the turn, derive the target speed from the actual current curve
        # rather than a potentially stale or prediction-weak v_target.
        current_curvature = max(self._current_curvature, _EPS)
        v_target = float((_A_LAT_REG_MAX / current_curvature) ** 0.5)
      else:
        v_target = self.v_target
      if not np.isfinite(v_target):
        v_target = 0.
      v = max(v_target, MIN_V) + self.a_target * _NO_OVERSHOOT_TIME_HORIZON
      if not np.isfinite(v):
        return MIN_V
      return max(v, MIN_V)

    return V_CRUISE_UNSET

  def _update_params(self) -> None:
    if self.frame % int(PARAMS_UPDATE_PERIOD / DT_MDL) == 0:
      self.enabled = self.params.get_bool("SmartCruiseControlVision")

  def _update_calculations(self, sm: messaging.SubMaster) -> None:
    self._required_decel = 0.
    self._t_risk = float(ModelConstants.T_IDXS[-1])

    if not self.long_enabled:
      self.pre_entry_frames = 0
      self.pre_entry_active = False
      return

    try:
      rate_plan = np.asarray(sm['modelV2'].orientationRate.z, dtype=np.float64)
      vel_plan = np.asarray(sm['modelV2'].velocity.x, dtype=np.float64)
    except Exception:
      self._fail_closed()
      return

    # Fail closed unless model arrays are 1D, same length as the horizon, non-empty, and finite.
    if (rate_plan.ndim != 1 or vel_plan.ndim != 1 or
        rate_plan.size == 0 or rate_plan.shape != vel_plan.shape or
        rate_plan.shape[0] != len(ModelConstants.T_IDXS)):
      self._fail_closed()
      return

    if not np.isfinite(self.v_ego) or not np.isfinite(self.a_ego):
      self._fail_closed()
      return

    curvature = sm['controlsState'].curvature
    if not np.isfinite(curvature):
      self._fail_closed()
      return

    self._current_curvature = float(abs(curvature))
    self.current_lat_acc = self.v_ego ** 2 * self._current_curvature
    if not np.isfinite(self.current_lat_acc):
      self._fail_closed()
      return

    rate_plan = np.abs(rate_plan)
    vel_plan = np.abs(vel_plan)
    if not np.all(np.isfinite(rate_plan)) or not np.all(np.isfinite(vel_plan)):
      self._fail_closed()
      return

    # Predicted curvature from model yaw rate / model velocity, then re-projected at ego speed.
    predicted_curvatures = rate_plan / np.maximum(vel_plan, 0.1)
    if not np.all(np.isfinite(predicted_curvatures)):
      self._fail_closed()
      return

    v_ego = max(abs(self.v_ego), 0.1)
    predicted_lat_accels = predicted_curvatures * (v_ego ** 2)
    if not np.all(np.isfinite(predicted_lat_accels)):
      self._fail_closed()
      return

    self.max_pred_lat_acc = float(np.percentile(predicted_lat_accels, 97))
    if not np.isfinite(self.max_pred_lat_acc) or self.max_pred_lat_acc <= _EPS:
      self.max_pred_lat_acc = 0.
      self.v_target = 0.
      self._required_decel = 0.
      self._t_risk = float(ModelConstants.T_IDXS[-1])
      self.pre_entry_frames = 0
      self.pre_entry_active = False
      self._prev_max_pred_lat_acc = 0.
      return

    # Maximum predicted curvature aligned with the p97 risk accel source.
    max_predicted_curvature = self.max_pred_lat_acc / (v_ego ** 2)
    if not np.isfinite(max_predicted_curvature) or max_predicted_curvature <= _EPS:
      self._fail_closed()
      return

    # Binding point: among predicted points above the entering threshold, choose the
    # point whose own target speed and horizon time require the strongest decel now.
    # p97 severity remains available for state triggering, telemetry, and spike filtering.
    crossing = predicted_lat_accels >= _ENTERING_PRED_LAT_ACC_TH
    if np.any(crossing):
      indices = np.where(crossing)[0]
      t_cross = np.asarray(ModelConstants.T_IDXS, dtype=np.float64)[indices]
      curvatures_cross = predicted_curvatures[indices]
      valid = np.isfinite(curvatures_cross) & (curvatures_cross > _EPS) & np.isfinite(t_cross)
      if np.any(valid):
        indices = indices[valid]
        t_cross = t_cross[valid]
        curvatures_cross = curvatures_cross[valid]
        v_targets_cross = np.sqrt(_A_LAT_REG_MAX / curvatures_cross)
        required_decels_cross = (v_targets_cross - self.v_ego) / np.maximum(t_cross, DT_MDL)
        binding = int(np.argmin(required_decels_cross))
        binding_idx = indices[binding]
        self._t_risk = float(ModelConstants.T_IDXS[binding_idx])
        self.v_target = float(v_targets_cross[binding])
        self._required_decel = float(required_decels_cross[binding])
      else:
        self._t_risk = float(ModelConstants.T_IDXS[-1])
        self.v_target = float((_A_LAT_REG_MAX / max_predicted_curvature) ** 0.5)
        self._required_decel = 0.
    else:
      self._t_risk = float(ModelConstants.T_IDXS[-1])
      self.v_target = float((_A_LAT_REG_MAX / max_predicted_curvature) ** 0.5)
      self._required_decel = 0.

    if not np.isfinite(self.v_target) or self.v_target < 0.:
      self.v_target = 0.
    if not np.isfinite(self._required_decel):
      self._required_decel = 0.

    can_pre_entry = self.enabled and not self.long_override and self.v_ego > MIN_V
    if can_pre_entry and _PRE_ENTRY_PRED_LAT_ACC_TH <= self.max_pred_lat_acc < _ENTERING_PRED_LAT_ACC_TH:
      if self.pre_entry_frames == 0 or self.max_pred_lat_acc >= self._prev_max_pred_lat_acc:
        self.pre_entry_frames += 1
      else:
        self.pre_entry_frames = 1
    else:
      self.pre_entry_frames = 0

    self.pre_entry_active = can_pre_entry and self.pre_entry_frames >= _PRE_ENTRY_MIN_FRAMES
    self._prev_max_pred_lat_acc = self.max_pred_lat_acc

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
          # High current lateral acceleration forces turning even with weak prediction.
          elif self.current_lat_acc >= _CURRENT_LAT_ACC_BLEED_TH:
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
    # Pre-entry keeps the public state enabled, but makes the controller gently active early.
    # Disabled/overriding states must still fail closed even if the prediction buffer was mild.
    active = self.state in ACTIVE_STATES or (self.state == VisionState.enabled and self.pre_entry_active)

    return enabled, active

  def _update_solution(self) -> float:
    # DISABLED, ENABLED, OVERRIDING
    if self.pre_entry_active and self.state == VisionState.enabled:
      a_target = _PRE_ENTRY_GENTLE_DECEL
    elif self.state not in ACTIVE_STATES:
      # when not overshooting, calculate v_turn as the speed at the prediction horizon when following
      # the smooth deceleration.
      a_target = self.a_ego
    # ENTERING
    elif self.state == VisionState.entering:
      # Horizon-aware deceleration, bounded by the existing smooth-decel table so far curves stay gentle
      # and near curves do not command more decel than the table allows.
      smooth_decel = float(np.interp(self.max_pred_lat_acc, _ENTERING_SMOOTH_DECEL_BP, _ENTERING_SMOOTH_DECEL_V))
      a_target = min(0.0, max(self._required_decel, smooth_decel))
    # TURNING
    elif self.state == VisionState.turning:
      # When turning, we provide a target acceleration that is comfortable for the lateral acceleration felt.
      a_target = float(np.interp(self.current_lat_acc, _TURNING_ACC_BP, _TURNING_ACC_V))
    # LEAVING
    elif self.state == VisionState.leaving:
      # If another curve is predicted ahead, suppress positive acceleration and only allow braking
      # when the horizon still requires it.
      if self.max_pred_lat_acc >= _ENTERING_PRED_LAT_ACC_TH:
        a_target = min(0.0, self._required_decel)
      else:
        a_target = _LEAVING_ACC
    else:
      raise NotImplementedError(f"SCC-V state not supported: {self.state}")

    return a_target

  def update(self, sm: messaging.SubMaster, long_enabled: bool, long_override: bool, v_ego: float, a_ego: float,
             v_cruise_setpoint: float) -> None:
    self.long_enabled = long_enabled
    self.long_override = long_override
    self.v_ego = v_ego
    self.a_ego = a_ego
    self.v_cruise_setpoint = v_cruise_setpoint

    self._update_params()
    if not (self.long_enabled and self.enabled):
      self._fail_closed()
    else:
      self._update_calculations(sm)

    self.is_enabled, self.is_active = self._update_state_machine()
    self.a_target = self._update_solution()

    self.output_v_target = self.get_v_target_from_control()
    self.output_a_target = self.get_a_target_from_control()

    self.frame += 1
