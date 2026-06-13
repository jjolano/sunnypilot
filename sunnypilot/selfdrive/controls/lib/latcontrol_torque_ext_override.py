"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import json

from openpilot.common.params import Params
from openpilot.sunnypilot.custom.lateral.speed_aware_torque import SpeedAwareTorqueRuntime, parse_speed_aware_torque_profile


class LatControlTorqueExtOverride:
  def __init__(self, CP):
    self.CP = CP
    self.params = Params()
    self.enforce_torque_control_toggle = self.params.get_bool("EnforceTorqueControl")
    self.torque_override_enabled = self.params.get_bool("TorqueParamsOverrideEnabled")
    self.frame = -1
    self.base_latAccelFactor = None
    self.base_friction = None
    self.last_speed_applied = None
    self.last_manual_applied = None
    self.last_manual_friction_applied = None
    self._speed_runtime = SpeedAwareTorqueRuntime()
    self._speed_mode = 'off'
    self._speed_profile_raw = None
    self._speed_profile = None
    self._live_torque_enabled = self.params.get_bool("LiveTorqueParamsToggle")
    self._manual_latAccelFactor = None
    self._manual_friction = None

  def _poll(self):
    self.torque_override_enabled = self.params.get_bool("TorqueParamsOverrideEnabled")
    self._live_torque_enabled = self.params.get_bool("LiveTorqueParamsToggle")
    if self.torque_override_enabled:
      self._manual_latAccelFactor = float(self.params.get("TorqueParamsOverrideLatAccelFactor", return_default=True))
      self._manual_friction = float(self.params.get("TorqueParamsOverrideFriction", return_default=True))
    mode = self.params.get("LiveTorqueSpeedAdaptiveMode", return_default=True) or 'off'
    self._speed_mode = mode if mode in ('off', 'shadow', 'apply') else 'off'
    self._speed_profile_raw = self.params.get("LiveTorqueSpeedAdaptiveParams", return_default=True) if self._speed_mode == 'apply' else None
    self._speed_profile = None
    if self._speed_profile_raw and self._live_torque_enabled:
      try:
        self._speed_profile = parse_speed_aware_torque_profile(self.CP, json.loads(self._speed_profile_raw))
      except Exception:
        self._speed_profile = None

  def _capture_base(self, torque_params):
    cur = float(torque_params.latAccelFactor)
    if self.base_latAccelFactor is None:
      self.base_latAccelFactor = cur
    elif self.last_speed_applied is not None and abs(cur - self.last_speed_applied) < 1e-9:
      torque_params.latAccelFactor = self.base_latAccelFactor
      cur = self.base_latAccelFactor
    else:
      self.base_latAccelFactor = cur
    self.base_friction = float(torque_params.friction)
    self.last_speed_applied = None
    return cur

  def _restore_manual_or_speed_base(self, torque_params):
    changed = False
    cur = float(torque_params.latAccelFactor)
    if self.last_speed_applied is not None and abs(cur - self.last_speed_applied) < 1e-9:
      changed = self._restore_base(torque_params) or changed
    elif self.last_manual_applied is not None and abs(cur - self.last_manual_applied) < 1e-9:
      changed = self._restore_base(torque_params) or changed
      if (self.last_manual_friction_applied is not None and self.base_friction is not None
          and abs(float(torque_params.friction) - self.last_manual_friction_applied) < 1e-9):
        torque_params.friction = self.base_friction
        changed = True
    self.last_manual_applied = None
    self.last_manual_friction_applied = None
    return changed

  def _restore_base(self, torque_params):
    if self.base_latAccelFactor is not None and abs(float(torque_params.latAccelFactor) - self.base_latAccelFactor) > 1e-9:
      torque_params.latAccelFactor = self.base_latAccelFactor
      self.last_speed_applied = None
      return True
    return False

  def update_override_torque_params(self, torque_params, v_ego=None) -> bool:
    if not self.enforce_torque_control_toggle:
      return False

    self.frame += 1
    if self.frame % 300 == 0:
      self._poll()

    if self.torque_override_enabled:
      if self.base_latAccelFactor is None:
        self.base_latAccelFactor = float(torque_params.latAccelFactor)
        self.base_friction = float(torque_params.friction)
      elif self.last_speed_applied is not None and abs(float(torque_params.latAccelFactor) - self.last_speed_applied) < 1e-9:
        self._restore_base(torque_params)
      manual_latAccelFactor = self._manual_latAccelFactor if self._manual_latAccelFactor is not None else float(torque_params.latAccelFactor)
      manual_friction = self._manual_friction if self._manual_friction is not None else float(torque_params.friction)
      torque_params.latAccelFactor = float(manual_latAccelFactor)
      torque_params.friction = float(manual_friction)
      self.last_speed_applied = None
      self.last_manual_applied = float(torque_params.latAccelFactor)
      self.last_manual_friction_applied = float(torque_params.friction)
      return True

    restored_manual_or_speed = self._restore_manual_or_speed_base(torque_params)
    self._capture_base(torque_params)

    if self._speed_mode != 'apply' or not self._live_torque_enabled or self._speed_profile is None:
      return self._restore_base(torque_params) or restored_manual_or_speed

    self._speed_runtime.profile = self._speed_profile
    ratio = self._speed_runtime.ratio(v_ego)
    if ratio == 1.0:
      return self._restore_base(torque_params) or restored_manual_or_speed

    base = self.base_latAccelFactor if self.base_latAccelFactor is not None else float(torque_params.latAccelFactor)
    torque_params.latAccelFactor = float(base * ratio)
    self.last_speed_applied = float(torque_params.latAccelFactor)
    return True
