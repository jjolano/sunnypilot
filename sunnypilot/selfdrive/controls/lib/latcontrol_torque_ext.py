"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

import ast

import numpy as np

from openpilot.common.params import UnknownKeyName
from openpilot.sunnypilot.selfdrive.controls.lib.nnlc.nnlc import NeuralNetworkLateralControl
from openpilot.sunnypilot.selfdrive.controls.lib.latcontrol_torque_ext_override import LatControlTorqueExtOverride
from openpilot.sunnypilot.selfdrive.locationd.torqued_ext import parse_speed_aware_params


class LatControlTorqueExt(NeuralNetworkLateralControl, LatControlTorqueExtOverride):
  def __init__(self, lac_torque, CP, CP_SP, CI):
    NeuralNetworkLateralControl.__init__(self, lac_torque, CP, CP_SP, CI)
    LatControlTorqueExtOverride.__init__(self, CP)
    self.last_v_ego = 0.0
    self.speed_aware_params = None
    self.speed_adaptive_apply_enabled = False
    self._speed_adaptive_base_factor = None
    self._speed_adaptive_applied_factor = None
    self.nominal_lat_accel_factor = float(CP.lateralTuning.torque.latAccelFactor) if CP.lateralTuning.which() == 'torque' else 0.0

  def update(self, CS, VM, pid, params, ff, pid_log, setpoint, measurement, calibrated_pose, roll_compensation,
             desired_lateral_accel, actual_lateral_accel, lateral_accel_deadzone, gravity_adjusted_lateral_accel,
             desired_curvature, actual_curvature, steer_limited_by_safety, output_torque):
    self.last_v_ego = CS.vEgo
    self._ff = ff
    self._pid = pid
    self._pid_log = pid_log
    self._setpoint = setpoint
    self._measurement = measurement
    self._roll_compensation = roll_compensation
    self._lateral_accel_deadzone = lateral_accel_deadzone
    self._desired_lateral_accel = desired_lateral_accel
    self._actual_lateral_accel = actual_lateral_accel
    self._desired_curvature = desired_curvature
    self._actual_curvature = actual_curvature
    self._gravity_adjusted_lateral_accel = gravity_adjusted_lateral_accel
    self._steer_limited_by_safety = steer_limited_by_safety
    self._output_torque = output_torque

    self.update_calculations(CS, VM, desired_lateral_accel)
    self.update_neural_network_feedforward(CS, params, calibrated_pose)

    return self._pid_log, self._output_torque

  def update_speed_aware_params(self, params_str):
    try:
      self.speed_adaptive_apply_enabled = self.params.get_bool("LiveTorqueSpeedAdaptiveApplyToggle")
    except UnknownKeyName:
      self.speed_adaptive_apply_enabled = False

    if not params_str:
      self.speed_aware_params = None
      return
    try:
      if isinstance(params_str, bytes):
        params_str = params_str.decode("utf-8")
      self.speed_aware_params = parse_speed_aware_params(self.CP, ast.literal_eval(params_str))
    except (TypeError, UnicodeDecodeError, ValueError, SyntaxError):
      self.speed_aware_params = None

  def _interpolate_speed_factor(self, v_ego):
    if not self.speed_aware_params:
      return None
    bp = [0, 10, 20, 30, 40]
    factors = []
    labels = ["0_10", "10_20", "20_30", "30_40", "40_plus"]
    for label in labels:
      if label in self.speed_aware_params:
        factor = float(self.speed_aware_params[label][0])  # latAccelFactor
        factors.append(factor if np.isfinite(factor) else None)
      else:
        factors.append(None)

    valid = [(b, f) for b, f in zip(bp, factors) if f is not None]
    if not valid:
      return None
    if len(valid) == 1:
      return valid[0][1]
    return float(np.interp(v_ego, [b for b, _ in valid], [f for _, f in valid]))

  def _valid_speed_factor(self, factor):
    if factor is None or not np.isfinite(factor) or factor <= 0.0:
      return False
    if self.nominal_lat_accel_factor <= 0.0:
      return False
    return 0.5 * self.nominal_lat_accel_factor <= factor <= 2.0 * self.nominal_lat_accel_factor

  def update_override_torque_params(self, torque_params) -> bool:
    restored_speed_adaptive = self._restore_speed_adaptive_base(torque_params)
    overridden = LatControlTorqueExtOverride.update_override_torque_params(self, torque_params)

    if self.speed_adaptive_apply_enabled and getattr(self, 'speed_aware_params', None) is not None:
      factor = self._interpolate_speed_factor(self.last_v_ego)
      if self._valid_speed_factor(factor):
        self._speed_adaptive_base_factor = float(torque_params.latAccelFactor)
        self._speed_adaptive_applied_factor = float(factor)
        torque_params.latAccelFactor = factor
        return True

    return overridden or restored_speed_adaptive

  def _restore_speed_adaptive_base(self, torque_params) -> bool:
    if self._speed_adaptive_applied_factor is None:
      return False

    restored = False
    if np.isclose(torque_params.latAccelFactor, self._speed_adaptive_applied_factor):
      torque_params.latAccelFactor = self._speed_adaptive_base_factor
      restored = True

    self._speed_adaptive_base_factor = None
    self._speed_adaptive_applied_factor = None
    return restored
