"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

import ast

import numpy as np

from openpilot.sunnypilot.selfdrive.controls.lib.nnlc.nnlc import NeuralNetworkLateralControl
from openpilot.sunnypilot.selfdrive.controls.lib.latcontrol_torque_ext_override import LatControlTorqueExtOverride


class LatControlTorqueExt(NeuralNetworkLateralControl, LatControlTorqueExtOverride):
  def __init__(self, lac_torque, CP, CP_SP, CI):
    NeuralNetworkLateralControl.__init__(self, lac_torque, CP, CP_SP, CI)
    LatControlTorqueExtOverride.__init__(self, CP)
    self.last_v_ego = 0.0
    self.speed_aware_params = None

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
    if not params_str:
      self.speed_aware_params = None
      return
    try:
      self.speed_aware_params = ast.literal_eval(params_str)
    except (ValueError, SyntaxError):
      self.speed_aware_params = None

  def _interpolate_speed_factor(self, v_ego):
    if not self.speed_aware_params:
      return None
    bp = [0, 10, 20, 30, 40]
    factors = []
    labels = ["0_10", "10_20", "20_30", "30_40", "40_plus"]
    for label in labels:
      if label in self.speed_aware_params:
        factors.append(self.speed_aware_params[label][0])  # latAccelFactor
      else:
        factors.append(None)

    valid = [(b, f) for b, f in zip(bp, factors) if f is not None]
    if not valid:
      return None
    if len(valid) == 1:
      return valid[0][1]
    return float(np.interp(v_ego, [b for b, _ in valid], [f for _, f in valid]))

  def update_override_torque_params(self, torque_params) -> bool:
    overridden = LatControlTorqueExtOverride.update_override_torque_params(self, torque_params)

    if getattr(self, 'speed_aware_params', None) is not None:
      factor = self._interpolate_speed_factor(self.last_v_ego)
      if factor is not None:
        torque_params.latAccelFactor = factor
        return True

    return overridden
