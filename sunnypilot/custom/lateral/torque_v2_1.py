"""Torque v2.1 controller — composes the response core and unified output governor.

Wires response_core -> NNLC/override extension -> output_governor behind the upstream
``LatControl`` interface, and is dispatched from ``controlsd_ext.initialize_lateral_control``
when ``TorqueControlTune == 2.1``. See
``docs/adr/2026-06-13-clean-room-torque-v2-1-architecture.md``.

First-cut approximations (land with engaged-route tuning / the steering-actuator-feedback
port): the governor's ``release_active`` is approximated by ``steering_pressed``; the governor's deferred comfort behaviors and
the additive assist learning are not yet present. End-to-end feel is validated against the
engaged-route corpus, not here.
"""
from __future__ import annotations

import math

from cereal import log
from opendbc.car.lateral import get_friction

from openpilot.selfdrive.controls.lib.latcontrol import LatControl
from openpilot.sunnypilot.selfdrive.controls.lib.latcontrol_torque_ext import LatControlTorqueExt
from openpilot.sunnypilot.selfdrive.controls.lib.underresponse_sentinel import UnderresponseSentinel, write_underresponse_debug
from openpilot.sunnypilot.custom.lateral.output_governor import OutputGovernor, OutputGovernorInputs
from openpilot.sunnypilot.custom.lateral.oscillation_observer import OscillationObserver
from openpilot.sunnypilot.custom.lateral.response_core import ResponseCore, ResponseCoreInputs

VERSION_V21 = 21
SAME_DIRECTION_TORQUE_EPS = 1e-3
SAME_DIRECTION_ERROR_EPS = 0.02


def _sign(value: float, eps: float) -> int:
  if value > eps:
    return 1
  if value < -eps:
    return -1
  return 0


def _safe_float(value, default: float = 0.0) -> float:
  try:
    v = float(value)
  except (TypeError, ValueError):
    return default
  return v if math.isfinite(v) else default


class LatControlTorqueV21(LatControl):
  def __init__(self, CP, CP_SP, CI, dt, extension=None):
    super().__init__(CP, CP_SP, CI, dt)
    self.torque_params = CP.lateralTuning.torque.as_builder()
    self._vm = None
    self.response_core = ResponseCore(
      dt, self.steer_max, self.torque_params,
      calc_curvature=lambda angle_rad, v_ego, roll: self._vm.calc_curvature(angle_rad, v_ego, roll),
      torque_from_lateral_accel=CI.torque_from_lateral_accel(),
      lateral_accel_from_torque=CI.lateral_accel_from_torque(),
      get_friction=get_friction,
    )
    self.governor = OutputGovernor(dt)
    self.underresponse_sentinel = UnderresponseSentinel(dt)
    self.oscillation_observer = OscillationObserver(dt)
    self.extension = extension if extension is not None else LatControlTorqueExt(self, CP, CP_SP, CI)
    self._under_response_path_evidence_valid = True
    self._limited_requested_torque = None
    self._limited_applied_torque = None

  def reset(self):
    super().reset()
    self.governor.reset()
    self.underresponse_sentinel.reset()
    self.oscillation_observer.reset()

  def set_torque_override_refresh_allowed(self, allowed: bool) -> None:
    if hasattr(self.extension, 'set_torque_override_refresh_allowed'):
      self.extension.set_torque_override_refresh_allowed(allowed)

  def set_under_response_path_evidence(self, valid: bool) -> None:
    self._under_response_path_evidence_valid = bool(valid)

  def set_steer_limited_output_context(self, requested_torque, applied_torque) -> None:
    try:
      requested = float(requested_torque)
      applied = float(applied_torque)
    except (TypeError, ValueError):
      requested = math.nan
      applied = math.nan
    self._limited_requested_torque = requested if math.isfinite(requested) else None
    self._limited_applied_torque = applied if math.isfinite(applied) else None

  def _same_direction_limit(self, steer_limited_by_safety: bool, nominal_torque: float,
                            desired_lateral_accel: float, actual_lateral_accel: float) -> bool:
    if not steer_limited_by_safety:
      return False
    try:
      nominal_sign = _sign(float(nominal_torque), SAME_DIRECTION_TORQUE_EPS)
      correction_sign = _sign(float(desired_lateral_accel) - float(actual_lateral_accel), SAME_DIRECTION_ERROR_EPS)
    except (TypeError, ValueError):
      return False
    if nominal_sign == 0 or nominal_sign != correction_sign:
      return False

    if self._limited_requested_torque is None or self._limited_applied_torque is None:
      return True
    requested_sign = _sign(self._limited_requested_torque, SAME_DIRECTION_TORQUE_EPS)
    applied_sign = _sign(self._limited_applied_torque, SAME_DIRECTION_TORQUE_EPS)
    return requested_sign == nominal_sign and applied_sign == requested_sign

  def set_under_response_path_evidence_from_lateral_demand(self, lateral_demand_result,
                                                           *, active: bool | None = None,
                                                           evidence_expected: bool | None = None) -> None:
    if active is False or evidence_expected is False:
      self.set_under_response_path_evidence(True)
      return
    if lateral_demand_result is None:
      self.set_under_response_path_evidence(False)
      return
    model_path = getattr(lateral_demand_result, "model_path_result", None)
    self.set_under_response_path_evidence(
      model_path is not None and not bool(getattr(model_path, "gated", True)) and str(getattr(model_path, "reason", "")) == "ok"
    )

  def update_live_torque_params(self, latAccelFactor, latAccelOffset, friction):
    self.torque_params.latAccelFactor = latAccelFactor
    self.torque_params.latAccelOffset = latAccelOffset
    self.torque_params.friction = friction
    self.response_core.update_limits()

  def _core_inputs(self, active, CS, params, steer_limited_by_safety, desired_curvature, lat_delay) -> ResponseCoreInputs:
    return ResponseCoreInputs(
      active=active,
      v_ego=CS.vEgo,
      steering_angle_deg=CS.steeringAngleDeg,
      steering_rate_deg=CS.steeringRateDeg,
      steering_pressed=CS.steeringPressed,
      angle_offset_deg=params.angleOffsetDeg,
      roll=params.roll,
      desired_curvature=desired_curvature,
      lat_delay=max(lat_delay, self.dt),
      steer_limited_by_safety=steer_limited_by_safety,
    )

  def update(self, active, CS, VM, params, steer_limited_by_safety, desired_curvature, calibrated_pose, curvature_limited, lat_delay):
    self._vm = VM
    if self.extension.update_override_torque_params(self.torque_params, CS.vEgo):
      self.response_core.update_limits()

    pid_log = log.ControlsState.LateralTorqueState.new_message()
    pid_log.version = VERSION_V21

    # The response core always advances its buffer/smoother (matching legacy v2) and resets
    # its PID when inactive; only its PID output is gated by `active`.
    rc = self.response_core.update(self._core_inputs(active, CS, params, steer_limited_by_safety, desired_curvature, lat_delay))

    if not active:
      self.governor.reset()
      write_underresponse_debug(pid_log, self.underresponse_sentinel.reset())
      self.oscillation_observer.reset()
      pid_log.active = False
      return 0.0, 0.0, pid_log

    pid_log.error = float(rc.error)
    output_torque = rc.output_torque

    # NNLC / override extension (explicit injection point; overrides pid_log and output_torque)
    pid_log, output_torque = self.extension.update(
      CS, VM, self.response_core.pid, params, rc.ff, pid_log, rc.setpoint, rc.measurement, calibrated_pose,
      rc.roll_compensation, rc.future_desired_lateral_accel, rc.measurement, rc.lateral_accel_deadzone,
      rc.gravity_adjusted_future_lateral_accel, desired_curvature, rc.measured_curvature, steer_limited_by_safety,
      output_torque,
    )

    nominal_output_torque = output_torque
    same_direction_limit = self._same_direction_limit(
      bool(steer_limited_by_safety), nominal_output_torque, rc.setpoint, rc.measurement
    )
    nominal_output_torque_log = _safe_float(nominal_output_torque)
    governed = self.governor.update(OutputGovernorInputs(
      active=True,
      v_ego=CS.vEgo,
      steering_rate_deg=-CS.steeringRateDeg,
      nominal_torque=nominal_output_torque,
      max_output=self.steer_max,
      desired_lateral_accel=rc.setpoint,
      actual_lateral_accel=rc.measurement,
      same_direction_limit=same_direction_limit,
      release_active=bool(CS.steeringPressed),
      path_evidence_valid=self._under_response_path_evidence_valid,
      controller_evidence_stable=not (rc.same_sign_unwind or rc.measurement_reset),
    ))
    output_torque = governed.output_torque
    ur_debug = self.underresponse_sentinel.update(
      active=True,
      v_ego=CS.vEgo,
      steering_pressed=CS.steeringPressed,
      steer_limited_by_safety=steer_limited_by_safety,
      curvature_limited=curvature_limited,
      setpoint=rc.setpoint,
      measurement=rc.measurement,
      lateral_accel_deadzone=rc.lateral_accel_deadzone,
      output_torque=output_torque,
      steer_max=self.steer_max,
      roll=params.roll,
    )
    write_underresponse_debug(pid_log, ur_debug)
    osc_debug = self.oscillation_observer.update(
      active=True,
      v_ego=CS.vEgo,
      steering_pressed=CS.steeringPressed,
      steer_limited_by_safety=steer_limited_by_safety,
      curvature_limited=curvature_limited,
      output_torque=output_torque,
      steer_max=self.steer_max,
      desired_lateral_accel=rc.setpoint,
      actual_lateral_accel=rc.measurement,
      steering_rate_deg=CS.steeringRateDeg,
    )
    adaptive = pid_log.adaptiveTorqueState
    adaptive.active = True
    adaptive.oscillationClassification = int(osc_debug.classification)
    adaptive.releaseActive = bool(CS.steeringPressed)
    adaptive.nominalOutput = float(-nominal_output_torque_log)
    adaptive.shapingActive = False
    adaptive.shapingReason = 0
    adaptive.unshapedOutput = float(-nominal_output_torque_log)
    adaptive.outputCap = float(governed.cap)
    adaptive.modelConfidence = float(getattr(rc, "model_confidence", 0.0) or 0.0)
    adaptive.steerLimitLimited = bool(steer_limited_by_safety)
    adaptive.steerLimitError = float(max(0.0, abs(nominal_output_torque_log) - (self.steer_max * governed.cap)))
    adaptive.steerLimitSameDirection = bool(same_direction_limit)
    adaptive.governorReason = int(governed.reason)
    adaptive.actualLateralJerk = float(rc.raw_actual_lateral_jerk)
    adaptive.governorFloor = float(governed.floor)
    adaptive.lowSpeedOutputMax = bool(CS.vEgo < self.sat_check_min_speed and abs(output_torque) >= self.steer_max * governed.cap - 1e-3)
    adaptive.rawActualLateralAccel = float(rc.raw_measurement)

    pid_log.active = True
    pid_log.p = float(self.response_core.pid.p)
    pid_log.i = float(self.response_core.pid.i)
    pid_log.d = float(self.response_core.pid.d)
    pid_log.f = float(self.response_core.pid.f)
    pid_log.output = float(-output_torque)
    pid_log.actualLateralAccel = float(rc.measurement)
    pid_log.desiredLateralAccel = float(rc.setpoint)
    pid_log.desiredLateralJerk = float(rc.desired_lateral_jerk)
    pid_log.saturated = bool(self._check_saturation(self.steer_max - abs(output_torque) < 1e-3, CS,
                                                    steer_limited_by_safety, curvature_limited))

    # TODO left is positive in this convention
    return -output_torque, 0.0, pid_log
