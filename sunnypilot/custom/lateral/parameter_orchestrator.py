"""Lateral torque parameter orchestration.

This module consolidates the torque-v2.1 parameter-adjacent pieces that were previously
spread across three modules:

  - model jerk / lookahead lateral-jerk evidence (legacy LatControlTorqueExtBase logic)
  - manual torque-parameter override + speed-aware base capture/restore policy
  - underresponse sentinel / monitor

The original modules are kept as thin compatibility facades during Phase 1.
No numeric behavior has been changed; code blocks are moved verbatim except for
minor namespacing (e.g. disambiguating helper functions that collided when merged).
"""
from __future__ import annotations

import json
import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from openpilot.common.params import Params
from openpilot.common.pid import PIDController
from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.sunnypilot.custom.lateral.speed_aware_torque import (
  SpeedAwareTorqueRuntime,
  parse_speed_aware_torque_profile,
)
from openpilot.sunnypilot.custom.lateral.torque_safety import (
  validate_live_torque_speed_adaptive_mode,
  validate_manual_torque_override_against_base,
  validate_torque_override_friction,
  validate_torque_override_lat_accel_factor,
)


# -----------------------------------------------------------------------------
# Model jerk / lookahead lateral-jerk evidence (LatControlTorqueExtBase shape)
# -----------------------------------------------------------------------------
LAT_PLAN_MIN_IDX = 5
LATERAL_LAG_MOD = 0.0

KP = 1.0
KI = 0.3


def get_predicted_lateral_jerk(lat_accels: Sequence[float], t_diffs: np.ndarray) -> list[float]:
  lat_accel_diffs = np.diff(lat_accels)
  lat_jerk = lat_accel_diffs / t_diffs
  return lat_jerk.tolist()


def _base_sign(x: float) -> float:
  return 1.0 if x > 0.0 else (-1.0 if x < 0.0 else 0.0)


# Legacy alias used by callers that import ``sign`` from the base module.
sign = _base_sign


def get_lookahead_value(future_vals: Sequence[float], current_val: float) -> float:
  if len(future_vals) == 0:
    return current_val

  same_sign_vals = [v for v in future_vals if _base_sign(v) == _base_sign(current_val)]

  if len(same_sign_vals) < len(future_vals):
    return 0.0

  min_val = min(same_sign_vals + [current_val], key=lambda x: abs(x))
  return min_val


def _sanitize_finite_float(value: Any, default: float = 0.0) -> float:
  try:
    v = float(value)
  except (TypeError, ValueError):
    return default
  return v if math.isfinite(v) else default


class TorqueModelEvidence:
  """Lateral-jerk evidence derived from the model path and vehicle state."""

  def __init__(self, lac_torque: Any, CP: Any, CP_SP: Any, CI: Any) -> None:
    self.model_v2 = None
    self.model_valid = False
    self.lac_torque = lac_torque

    self.actual_lateral_jerk: float = 0.0
    self.lateral_jerk_setpoint: float = 0.0
    self.lateral_jerk_measurement: float = 0.0
    self.lookahead_lateral_jerk: float = 0.0

    self.torque_from_lateral_accel_in_torque_space = CI.torque_from_lateral_accel_in_torque_space()
    self.torque_params = lac_torque.torque_params

    self._ff = 0.0
    self._pid = PIDController(KP, KI)
    self._pid_log = None
    self._setpoint = 0.0
    self._measurement = 0.0
    self._roll_compensation = 0.0
    self._lateral_accel_deadzone = 0.0
    self._desired_lateral_accel = 0.0
    self._actual_lateral_accel = 0.0
    self._desired_curvature = 0.0
    self._actual_curvature = 0.0
    self._gravity_adjusted_lateral_accel = 0.0
    self._steer_limited_by_safety = False
    self._output_torque = 0.0

    self.friction_look_ahead_v = [1.4, 2.0]
    self.friction_look_ahead_bp = [9.0, 30.0]

    self.lat_jerk_friction_factor = 0.4
    self.lat_accel_friction_factor = 0.7

    self.t_diffs = np.diff(ModelConstants.T_IDXS)
    self.desired_lat_jerk_time = CP.steerActuatorDelay + LATERAL_LAG_MOD
    self._torque_model_evidence_initialized = True

  def update_model_v2(self, model_v2: Any) -> None:
    self.model_v2 = model_v2
    self.model_valid = self.model_v2 is not None and len(self.model_v2.orientation.x) >= CONTROL_N

  def update_lateral_lag(self, lag: float) -> None:
    self.desired_lat_jerk_time = max(0.01, lag) + LATERAL_LAG_MOD

  def update_friction_input(self, val_1: float, val_2: float) -> float:
    _error = val_1 - val_2
    _value = self.lat_accel_friction_factor * _error + self.lat_jerk_friction_factor * self.lookahead_lateral_jerk
    return _value

  def update_calculations(self, CS: Any, VM: Any, desired_lateral_accel: float) -> None:
    self.actual_lateral_jerk = 0.0
    self.lateral_jerk_setpoint = 0.0
    self.lateral_jerk_measurement = 0.0
    self.lookahead_lateral_jerk = 0.0

    actual_curvature_rate = -VM.calc_curvature(math.radians(CS.steeringRateDeg), CS.vEgo, 0.0)
    self.actual_lateral_jerk = actual_curvature_rate * CS.vEgo ** 2

    if self.model_valid:
      lookahead = np.interp(CS.vEgo, self.friction_look_ahead_bp, self.friction_look_ahead_v)
      friction_upper_idx = next((i for i, val in enumerate(ModelConstants.T_IDXS) if val > lookahead), 16)
      predicted_lateral_jerk = get_predicted_lateral_jerk(self.model_v2.acceleration.y, self.t_diffs)
      desired_lateral_jerk = (np.interp(self.desired_lat_jerk_time, ModelConstants.T_IDXS,
                              self.model_v2.acceleration.y) - desired_lateral_accel) / self.desired_lat_jerk_time
      self.lookahead_lateral_jerk = get_lookahead_value(predicted_lateral_jerk[LAT_PLAN_MIN_IDX:friction_upper_idx], desired_lateral_jerk)
      if self.lookahead_lateral_jerk == 0.0:
        self.actual_lateral_jerk = 0.0
        self.lat_accel_friction_factor = 1.0
      self.lateral_jerk_setpoint = self.lat_jerk_friction_factor * self.lookahead_lateral_jerk
      self.lateral_jerk_measurement = self.lat_jerk_friction_factor * self.actual_lateral_jerk


# -----------------------------------------------------------------------------
# Torque parameter override / speed-aware base policy (legacy override shape)
# -----------------------------------------------------------------------------
class TorqueParameterOverridePolicy:
  """Manual torque-parameter override and speed-aware adaptive base policy."""

  def __init__(self, CP: Any) -> None:
    self.CP = CP
    self.params = Params()
    self.enforce_torque_control_toggle = self.params.get_bool("EnforceTorqueControl")
    self.torque_override_enabled = self.params.get_bool("TorqueParamsOverrideEnabled")
    self.frame = -1
    self.base_latAccelFactor: float | None = None
    self.base_friction: float | None = None
    self.last_speed_applied: float | None = None
    self.last_manual_applied: float | None = None
    self.last_manual_friction_applied: float | None = None
    self._speed_runtime = SpeedAwareTorqueRuntime()
    self._speed_mode = 'off'
    self._speed_profile_raw: bytes | None = None
    self._speed_profile: Any = None
    self._live_torque_enabled = self.params.get_bool("LiveTorqueParamsToggle")
    self._custom_torque_params = self.params.get_bool("CustomTorqueParams")
    self._manual_latAccelFactor: float | None = None
    self._manual_friction: float | None = None
    self._manual_override_values_valid = False
    self._refresh_allowed = True
    self._refresh_deferred = False
    self._poll()
    self._torque_parameter_override_policy_initialized = True

  def set_torque_override_refresh_allowed(self, allowed: bool) -> None:
    self._refresh_allowed = bool(allowed)

  def _poll(self) -> None:
    self.enforce_torque_control_toggle = self.params.get_bool("EnforceTorqueControl")
    self.torque_override_enabled = self.params.get_bool("TorqueParamsOverrideEnabled")
    self._custom_torque_params = self.params.get_bool("CustomTorqueParams")
    self._live_torque_enabled = self.params.get_bool("LiveTorqueParamsToggle")
    if self.torque_override_enabled:
      self._manual_latAccelFactor = validate_torque_override_lat_accel_factor(self.params.get("TorqueParamsOverrideLatAccelFactor", return_default=True))
      self._manual_friction = validate_torque_override_friction(self.params.get("TorqueParamsOverrideFriction", return_default=True))
      self._manual_override_values_valid = self._manual_latAccelFactor is not None and self._manual_friction is not None
    else:
      self._manual_latAccelFactor = None
      self._manual_friction = None
      self._manual_override_values_valid = False
    mode = self.params.get("LiveTorqueSpeedAdaptiveMode", return_default=True)
    self._speed_mode = validate_live_torque_speed_adaptive_mode(mode)
    self._speed_profile_raw = self.params.get("LiveTorqueSpeedAdaptiveParams", return_default=True) if self._speed_mode == 'apply' else None
    self._speed_profile = None
    if self._speed_profile_raw and self._live_torque_enabled:
      try:
        self._speed_profile = parse_speed_aware_torque_profile(self.CP, json.loads(self._speed_profile_raw))
      except Exception:
        self._speed_profile = None

  def _capture_base(self, torque_params: Any) -> float:
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

  def _restore_manual_or_speed_base(self, torque_params: Any) -> bool:
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

  def _restore_base(self, torque_params: Any) -> bool:
    if self.base_latAccelFactor is not None and abs(float(torque_params.latAccelFactor) - self.base_latAccelFactor) > 1e-9:
      torque_params.latAccelFactor = self.base_latAccelFactor
      self.last_speed_applied = None
      return True
    return False

  def _maybe_poll(self, allow_refresh: bool | None) -> None:
    if allow_refresh is not None:
      self._refresh_allowed = bool(allow_refresh)
    self.frame += 1

    should_poll = self.frame % 300 == 0 or (self._refresh_allowed and self._refresh_deferred)
    if not should_poll:
      return

    if not self._refresh_allowed:
      self._refresh_deferred = True
      return

    self._poll()
    self._refresh_deferred = False

  def update_override_torque_params(self, torque_params: Any, v_ego: float | None = None, *, allow_refresh: bool | None = None) -> bool:
    self._maybe_poll(allow_refresh)

    if not self.enforce_torque_control_toggle:
      return False

    if self.torque_override_enabled and self._custom_torque_params:
      if not self._manual_override_values_valid:
        return self._restore_manual_or_speed_base(torque_params)
      manual_lat_accel_factor = self._manual_latAccelFactor
      manual_friction = self._manual_friction
      if manual_lat_accel_factor is None or manual_friction is None:
        return self._restore_manual_or_speed_base(torque_params)
      if self.base_latAccelFactor is None:
        self.base_latAccelFactor = float(torque_params.latAccelFactor)
        self.base_friction = float(torque_params.friction)
      elif self.last_speed_applied is not None and abs(float(torque_params.latAccelFactor) - self.last_speed_applied) < 1e-9:
        self._restore_base(torque_params)
      if not validate_manual_torque_override_against_base(
          manual_lat_accel_factor, manual_friction, self.base_latAccelFactor, self.base_friction):
        return self._restore_manual_or_speed_base(torque_params)
      torque_params.latAccelFactor = float(manual_lat_accel_factor)
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


# -----------------------------------------------------------------------------
# Underresponse sentinel / monitor (legacy UnderresponseSentinel shape)
# -----------------------------------------------------------------------------
MIN_V_EGO = 10.0
MIN_ABS_SETPOINT = 0.45
MIN_ABS_ERROR = 0.25
ERROR_FRACTION = 0.25
EWMA_TAU = 0.25
SOFT_DECAY_TAU = 0.30
DESIRED_PERSISTENCE_TIME = 0.25
TRIGGER_TIME = 0.20
MAX_DESIRED_JERK_FOR_STABLE = 4.0
MAX_ABS_ROLL = 0.08
MAX_ROLL_RATE = 0.08
MAX_OUTPUT_TORQUE_FRAC = 0.90
MIN_CLOSING_RATE = 0.50
SHADOW_CORR_GAIN = 0.50
MAX_SHADOW_LAT_ACCEL = 0.35
ACTUAL_OPPOSING_MIN_ABS = 0.20

BLOCK_INACTIVE = 1 << 0
BLOCK_LOW_SPEED = 1 << 1
BLOCK_STEERING_PRESSED = 1 << 2
BLOCK_STEER_LIMITED = 1 << 3
BLOCK_CURVATURE_LIMITED = 1 << 4
BLOCK_TORQUE_SATURATED = 1 << 5
BLOCK_ROLL_TOO_HIGH = 1 << 0o6
BLOCK_ROLL_UNSTABLE = 1 << 7
BLOCK_DESIRED_TOO_SMALL = 1 << 8
BLOCK_DESIRED_NOT_PERSISTENT = 1 << 9
BLOCK_ACTUAL_OPPOSING = 1 << 10
BLOCK_SIGN_MISMATCH = 1 << 11
BLOCK_ERROR_TOO_SMALL = 1 << 12
BLOCK_FAST_CLOSING = 1 << 13
BLOCK_INVALID_INPUT = 1 << 14

BLOCK_NAMES = {
  BLOCK_INACTIVE: "inactive",
  BLOCK_LOW_SPEED: "low_speed",
  BLOCK_STEERING_PRESSED: "steering_pressed",
  BLOCK_STEER_LIMITED: "steer_limited",
  BLOCK_CURVATURE_LIMITED: "curvature_limited",
  BLOCK_TORQUE_SATURATED: "torque_saturated",
  BLOCK_ROLL_TOO_HIGH: "roll_too_high",
  BLOCK_ROLL_UNSTABLE: "roll_unstable",
  BLOCK_DESIRED_TOO_SMALL: "desired_too_small",
  BLOCK_DESIRED_NOT_PERSISTENT: "desired_not_persistent",
  BLOCK_ACTUAL_OPPOSING: "actual_opposing",
  BLOCK_SIGN_MISMATCH: "sign_mismatch",
  BLOCK_ERROR_TOO_SMALL: "error_too_small",
  BLOCK_FAST_CLOSING: "fast_closing",
  BLOCK_INVALID_INPUT: "invalid_input",
}


@dataclass
class UnderresponseDebug:
  active: bool = False
  eligible: bool = False
  block_mask: int = 0
  error: float = 0.0
  error_filtered: float = 0.0
  duration: float = 0.0
  closing_rate: float = 0.0
  shadow_lat_accel: float = 0.0
  severity: float = 0.0


def _ur_sign(x: float, eps: float = 1e-6) -> int:
  if x > eps:
    return 1
  if x < -eps:
    return -1
  return 0


def _ur_clip(x: float, lo: float, hi: float) -> float:
  return min(max(x, lo), hi)


def _ur_finite(*vals: float) -> bool:
  return all(math.isfinite(float(v)) for v in vals)


class UnderresponseSentinel:
  def __init__(self, dt: float):
    self.dt = max(float(dt), 1e-6)
    self.history_len = max(2, int(DESIRED_PERSISTENCE_TIME / self.dt))
    self.desired_history: deque[float] = deque(maxlen=self.history_len)
    self.error_filtered = 0.0
    self.above_threshold_frames = 0
    self.prev_deficit: float | None = None
    self.prev_roll: float | None = None
    self.last_debug = UnderresponseDebug()

  def _reset_tracking(self) -> None:
    self.desired_history.clear()
    self.error_filtered = 0.0
    self.above_threshold_frames = 0
    self.prev_deficit = None

  def reset(self) -> UnderresponseDebug:
    self._reset_tracking()
    self.prev_roll = None
    self.last_debug = UnderresponseDebug(block_mask=BLOCK_INACTIVE)
    return self.last_debug

  def _soft_decay(self) -> None:
    alpha = math.exp(-self.dt / SOFT_DECAY_TAU)
    self.error_filtered *= alpha
    self.above_threshold_frames = 0

  def _desired_persistent(self, desired_sign: int) -> bool:
    if desired_sign == 0 or len(self.desired_history) < self.history_len:
      return False
    vals = list(self.desired_history)
    if any(_ur_sign(v) != desired_sign for v in vals):
      return False
    if min(abs(v) for v in vals) < MIN_ABS_SETPOINT:
      return False
    max_jerk = max(abs(vals[i] - vals[i - 1]) / self.dt for i in range(1, len(vals)))
    return max_jerk <= MAX_DESIRED_JERK_FOR_STABLE

  def update(self, *, active: bool, v_ego: float, steering_pressed: bool, steer_limited_by_safety: bool,
             curvature_limited: bool, setpoint: float, measurement: float, lateral_accel_deadzone: float,
             output_torque: float, steer_max: float, roll: float) -> UnderresponseDebug:
    if not _ur_finite(v_ego, setpoint, measurement, lateral_accel_deadzone, output_torque, steer_max, roll):
      self.reset()
      self.last_debug = UnderresponseDebug(block_mask=BLOCK_INVALID_INPUT)
      return self.last_debug

    error = float(setpoint - measurement)
    if not active:
      self.reset()
      self.last_debug = UnderresponseDebug(block_mask=BLOCK_INACTIVE, error=error)
      return self.last_debug

    block_mask = 0
    desired_sign = _ur_sign(setpoint)
    measurement_sign = _ur_sign(measurement)
    abs_deadzone = abs(float(lateral_accel_deadzone))
    threshold = max(MIN_ABS_ERROR, ERROR_FRACTION * abs(setpoint), 2.0 * abs_deadzone)
    deficit = max(0.0, desired_sign * error) if desired_sign != 0 else 0.0
    closing_rate = 0.0 if self.prev_deficit is None else (self.prev_deficit - deficit) / self.dt

    roll_rate = 0.0
    if self.prev_roll is not None:
      roll_rate = abs(roll - self.prev_roll) / self.dt
    self.prev_roll = float(roll)

    if v_ego < MIN_V_EGO:
      block_mask |= BLOCK_LOW_SPEED
    if steering_pressed:
      block_mask |= BLOCK_STEERING_PRESSED
    if steer_limited_by_safety:
      block_mask |= BLOCK_STEER_LIMITED
    if curvature_limited:
      block_mask |= BLOCK_CURVATURE_LIMITED
    if steer_max <= 0.0 or abs(output_torque) >= MAX_OUTPUT_TORQUE_FRAC * abs(steer_max):
      block_mask |= BLOCK_TORQUE_SATURATED
    if abs(roll) > MAX_ABS_ROLL:
      block_mask |= BLOCK_ROLL_TOO_HIGH
    if roll_rate > MAX_ROLL_RATE:
      block_mask |= BLOCK_ROLL_UNSTABLE

    if block_mask != 0:
      self._reset_tracking()
      self.last_debug = UnderresponseDebug(block_mask=block_mask, error=error, closing_rate=closing_rate)
      return self.last_debug

    self.desired_history.append(float(setpoint))

    if abs(setpoint) < MIN_ABS_SETPOINT or desired_sign == 0:
      block_mask |= BLOCK_DESIRED_TOO_SMALL
    if not self._desired_persistent(desired_sign):
      block_mask |= BLOCK_DESIRED_NOT_PERSISTENT
    if measurement_sign != 0 and measurement_sign != desired_sign and abs(measurement) > max(ACTUAL_OPPOSING_MIN_ABS, 2.0 * abs_deadzone):
      block_mask |= BLOCK_ACTUAL_OPPOSING
    if desired_sign != 0 and desired_sign * error <= 0.0:
      block_mask |= BLOCK_SIGN_MISMATCH
    if deficit <= threshold:
      block_mask |= BLOCK_ERROR_TOO_SMALL
    if closing_rate > MIN_CLOSING_RATE:
      block_mask |= BLOCK_FAST_CLOSING

    self.prev_deficit = deficit

    if block_mask != 0:
      self._soft_decay()
      self.last_debug = UnderresponseDebug(block_mask=block_mask, error=error, error_filtered=self.error_filtered,
                                           closing_rate=closing_rate)
      return self.last_debug

    alpha = math.exp(-self.dt / EWMA_TAU)
    self.error_filtered = alpha * self.error_filtered + (1.0 - alpha) * error
    filtered_deficit = max(0.0, desired_sign * self.error_filtered)

    if filtered_deficit > threshold:
      self.above_threshold_frames += 1
    else:
      self.above_threshold_frames = 0
      block_mask |= BLOCK_ERROR_TOO_SMALL

    duration = self.above_threshold_frames * self.dt
    active_trigger = block_mask == 0 and duration >= TRIGGER_TIME
    if active_trigger:
      shadow_lat_accel = desired_sign * min(MAX_SHADOW_LAT_ACCEL, SHADOW_CORR_GAIN * filtered_deficit)
      severity = _ur_clip((filtered_deficit - threshold) / max(threshold, 1e-6), 0.0, 1.0)
      severity *= _ur_clip((MIN_CLOSING_RATE - closing_rate) / max(MIN_CLOSING_RATE, 1e-6), 0.0, 1.0)
    else:
      shadow_lat_accel = 0.0
      severity = 0.0

    self.last_debug = UnderresponseDebug(active=active_trigger, eligible=(block_mask == 0), block_mask=block_mask,
                                         error=error, error_filtered=self.error_filtered, duration=duration,
                                         closing_rate=closing_rate, shadow_lat_accel=shadow_lat_accel,
                                         severity=severity)
    return self.last_debug


def write_underresponse_debug(pid_log: Any, debug: UnderresponseDebug) -> None:
  pid_log.underresponseActive = bool(debug.active)
  pid_log.underresponseEligible = bool(debug.eligible)
  pid_log.underresponseBlockMask = int(debug.block_mask)
  pid_log.underresponseError = float(debug.error)
  pid_log.underresponseErrorFiltered = float(debug.error_filtered)
  pid_log.underresponseDuration = float(debug.duration)
  pid_log.underresponseClosingRate = float(debug.closing_rate)
  pid_log.underresponseShadowLatAccel = float(debug.shadow_lat_accel)
  pid_log.underresponseSeverity = float(debug.severity)


# -----------------------------------------------------------------------------
# Unified parameter orchestrator facade for callers that want a single owner.
# During Phase 1 the per-domain classes above are exposed directly; the controller
# facades in the legacy modules compose or inherit from those classes.
# -----------------------------------------------------------------------------
class ParameterOrchestrator(TorqueModelEvidence, TorqueParameterOverridePolicy):
  """Single owner for lateral torque parameter adjustment and path evidence.

  The legacy ``LatControlTorqueExtBase`` and ``LatControlTorqueExtOverride`` facades
  inherit from this class so that existing callers see the same methods and state.
  """

  def __init__(
    self,
    *,
    CP: Any | None = None,
    lac_torque: Any | None = None,
    CP_SP: Any | None = None,
    CI: Any | None = None,
    init_evidence: bool = True,
    init_override: bool = True,
  ) -> None:
    # Match the legacy ordering: evidence init sets up PID/feedforward state, then
    # override init polls params and configures adaptive/speed-aware state.
    if (
      init_evidence and
      not hasattr(self, '_torque_model_evidence_initialized') and
      lac_torque is not None and CP is not None and CP_SP is not None and CI is not None
    ):
      TorqueModelEvidence.__init__(self, lac_torque, CP, CP_SP, CI)
    elif init_evidence and not hasattr(self, '_torque_model_evidence_initialized') and lac_torque is not None:
      # Partial evidence-only path; not used by current facades but kept symmetric.
      TorqueModelEvidence.__init__(self, lac_torque, CP or {}, CP_SP or {}, CI or {})
    if init_override and not hasattr(self, '_torque_parameter_override_policy_initialized') and CP is not None:
      TorqueParameterOverridePolicy.__init__(self, CP)
    self._param_orchestrator_initialized = True
