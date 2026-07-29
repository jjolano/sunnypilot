"""Phase 1 parity tests: legacy v1 modules vs new parameter_orchestrator facades.

The tests load the ``*_v1.py`` backups and the active modules side-by-side with the
same synthetic inputs and assert identical outputs, debug structures, and internal state.
No real Params access is required; Params is patched to a no-op fake.
"""
from __future__ import annotations

import importlib
import math
from types import SimpleNamespace

import numpy as np
import pytest

from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N
from openpilot.selfdrive.modeld.constants import ModelConstants


class _FakeParams:
  """Deterministic Params stand-in for override policy tests."""

  def get_bool(self, _key: str) -> bool:
    return False

  def get(self, _key: str, *, return_default: bool = True) -> bytes:
    return b""


def _load_modules():
  """Import v1 backups and active v2 modules under a patched Params."""
  mp = pytest.MonkeyPatch()
  with mp.context() as ctx:
    ctx.setattr("openpilot.common.params.Params", _FakeParams)
    return {
      "base_v1": importlib.import_module("openpilot.sunnypilot.selfdrive.controls.lib.latcontrol_torque_ext_base_v1"),
      "base_v2": importlib.import_module("openpilot.sunnypilot.selfdrive.controls.lib.latcontrol_torque_ext_base"),
      "override_v1": importlib.import_module("openpilot.sunnypilot.selfdrive.controls.lib.latcontrol_torque_ext_override_v1"),
      "override_v2": importlib.import_module("openpilot.sunnypilot.selfdrive.controls.lib.latcontrol_torque_ext_override"),
      "under_v1": importlib.import_module("openpilot.sunnypilot.selfdrive.controls.lib.underresponse_sentinel_v1"),
      "under_v2": importlib.import_module("openpilot.sunnypilot.selfdrive.controls.lib.underresponse_sentinel"),
    }


_MODULES = _load_modules()


def _make_lac_and_ci():
  torque_params = SimpleNamespace(
    latAccelFactor=2.0,
    latAccelOffset=0.0,
    friction=0.15,
    steeringAngleDeadzoneDeg=0.0,
  )
  lac_torque = SimpleNamespace(torque_params=torque_params)
  CI = SimpleNamespace(torque_from_lateral_accel_in_torque_space=lambda: lambda _lat, _tp, **_: 0.0)
  return lac_torque, CI


def _make_cp():
  return SimpleNamespace(steerActuatorDelay=0.2)


def _make_cp_sp():
  return SimpleNamespace()


def _model_v2(lat_accel: np.ndarray | None = None, orientation_len: int = CONTROL_N + 5):
  if lat_accel is None:
    lat_accel = np.zeros(len(ModelConstants.T_IDXS))
  return SimpleNamespace(
    orientation=SimpleNamespace(x=np.zeros(orientation_len)),
    acceleration=SimpleNamespace(y=np.asarray(lat_accel, dtype=float)),
  )


def _cs(v_ego: float = 20.0, steering_rate_deg: float = 10.0):
  return SimpleNamespace(vEgo=v_ego, steeringRateDeg=steering_rate_deg, aEgo=0.0, steeringPressed=False)


def _vm(curvature_rate_per_deg: float = 0.001):
  # calc_curvature(angle_rad, v_ego, roll) -> curvature; we want a constant rate per deg.
  return SimpleNamespace(
    calc_curvature=lambda angle_rad, _v, _roll: curvature_rate_per_deg * math.degrees(angle_rad),
  )


def _torque_params():
  return SimpleNamespace(latAccelFactor=2.0, friction=0.15)


def _debug_eq(d1, d2) -> bool:
  """Equality for UnderresponseDebug across separate dataclass definitions."""
  return d1.__dict__ == d2.__dict__


# -----------------------------------------------------------------------------
# Torque model evidence parity
# -----------------------------------------------------------------------------
class TestTorqueModelEvidenceParity:
  def _make_instances(self):
    lac_torque, CI = _make_lac_and_ci()
    CP = _make_cp()
    CP_SP = _make_cp_sp()
    return (
      _MODULES["base_v1"].LatControlTorqueExtBase(lac_torque, CP, CP_SP, CI),
      _MODULES["base_v2"].LatControlTorqueExtBase(lac_torque, CP, CP_SP, CI),
    )

  def test_initial_state_matches(self):
    v1, v2 = self._make_instances()
    assert v1.model_valid == v2.model_valid is False
    assert v1.actual_lateral_jerk == pytest.approx(v2.actual_lateral_jerk)
    assert v1.lateral_jerk_setpoint == pytest.approx(v2.lateral_jerk_setpoint)
    assert v1.lateral_jerk_measurement == pytest.approx(v2.lateral_jerk_measurement)
    assert v1.lookahead_lateral_jerk == pytest.approx(v2.lookahead_lateral_jerk)
    assert v1.torque_params is v2.torque_params

  def test_model_v2_validation_matches(self):
    v1, v2 = self._make_instances()
    v1.update_model_v2(_model_v2())
    v2.update_model_v2(_model_v2())
    assert v1.model_valid == v2.model_valid is True

    v1.update_model_v2(_model_v2(orientation_len=CONTROL_N - 1))
    v2.update_model_v2(_model_v2(orientation_len=CONTROL_N - 1))
    assert v1.model_valid == v2.model_valid is False

  def test_calculation_matches_for_clean_curve(self):
    v1, v2 = self._make_instances()
    t_idxs = ModelConstants.T_IDXS
    curvature = 0.002
    v_ego = 15.0
    lat_accel = curvature * v_ego * v_ego * np.ones(len(t_idxs))
    v1.update_model_v2(_model_v2(lat_accel))
    v2.update_model_v2(_model_v2(lat_accel))
    v1.update_calculations(_cs(v_ego=v_ego), _vm(), desired_lateral_accel=lat_accel[0])
    v2.update_calculations(_cs(v_ego=v_ego), _vm(), desired_lateral_accel=lat_accel[0])
    assert v1.actual_lateral_jerk == pytest.approx(v2.actual_lateral_jerk)
    assert v1.lateral_jerk_setpoint == pytest.approx(v2.lateral_jerk_setpoint)
    assert v1.lateral_jerk_measurement == pytest.approx(v2.lateral_jerk_measurement)
    assert v1.lookahead_lateral_jerk == pytest.approx(v2.lookahead_lateral_jerk)

  def test_calculation_zeroes_when_model_invalid(self):
    v1, v2 = self._make_instances()
    v1.update_calculations(_cs(), _vm(), desired_lateral_accel=0.5)
    v2.update_calculations(_cs(), _vm(), desired_lateral_accel=0.5)
    assert v1.lookahead_lateral_jerk == v2.lookahead_lateral_jerk == 0.0
    assert v1.lateral_jerk_setpoint == v2.lateral_jerk_setpoint == 0.0

  def test_lateral_lag_matches(self):
    v1, v2 = self._make_instances()
    v1.update_lateral_lag(0.35)
    v2.update_lateral_lag(0.35)
    assert v1.desired_lat_jerk_time == pytest.approx(v2.desired_lat_jerk_time)

  def test_friction_input_matches(self):
    v1, v2 = self._make_instances()
    v1.lookahead_lateral_jerk = 0.2
    v2.lookahead_lateral_jerk = 0.2
    assert v1.update_friction_input(0.8, 0.5) == pytest.approx(v2.update_friction_input(0.8, 0.5))


# -----------------------------------------------------------------------------
# Torque parameter override parity
# -----------------------------------------------------------------------------
class TestTorqueParameterOverrideParity:
  def _make_instances(self):
    CP = _make_cp()
    v1 = _MODULES["override_v1"].LatControlTorqueExtOverride(CP)
    v2 = _MODULES["override_v2"].LatControlTorqueExtOverride(CP)
    # Skip the frame-0 poll so that state set by the test is not overwritten.
    v1.frame = v2.frame = 1
    return v1, v2

  def _sync_state(self, v1, v2, **state):
    for key, value in state.items():
      setattr(v1, key, value)
      setattr(v2, key, value)

  def test_initial_state_matches(self):
    v1, v2 = self._make_instances()
    assert v1.enforce_torque_control_toggle == v2.enforce_torque_control_toggle is False
    assert v1.torque_override_enabled == v2.torque_override_enabled is False
    assert v1.base_latAccelFactor == v2.base_latAccelFactor is None
    assert v1.base_friction == v2.base_friction is None

  def test_disabled_enforce_returns_false(self):
    v1, v2 = self._make_instances()
    tp = _torque_params()
    assert v1.update_override_torque_params(tp, v_ego=20.0) is False
    assert v2.update_override_torque_params(tp, v_ego=20.0) is False
    assert tp.latAccelFactor == pytest.approx(v1.base_latAccelFactor or 2.0)
    assert tp.latAccelFactor == pytest.approx(v2.base_latAccelFactor or 2.0)

  def test_manual_override_valid_applies(self):
    v1, v2 = self._make_instances()
    self._sync_state(
      v1, v2,
      enforce_torque_control_toggle=True,
      torque_override_enabled=True,
      _custom_torque_params=True,
      _manual_latAccelFactor=2.5,
      _manual_friction=0.25,
      _manual_override_values_valid=True,
      base_latAccelFactor=2.0,
      base_friction=0.15,
    )
    tp1 = _torque_params()
    tp2 = _torque_params()
    r1 = v1.update_override_torque_params(tp1, v_ego=20.0)
    r2 = v2.update_override_torque_params(tp2, v_ego=20.0)
    assert r1 is True
    assert r2 is True
    assert tp1.latAccelFactor == tp2.latAccelFactor == 2.5
    assert tp1.friction == tp2.friction == 0.25
    assert v1.last_manual_applied == pytest.approx(v2.last_manual_applied)
    assert v1.last_manual_friction_applied == pytest.approx(v2.last_manual_friction_applied)

  def test_manual_override_invalid_restores_base(self):
    v1, v2 = self._make_instances()
    self._sync_state(
      v1, v2,
      enforce_torque_control_toggle=True,
      torque_override_enabled=True,
      _custom_torque_params=True,
      _manual_override_values_valid=False,
      base_latAccelFactor=2.0,
      base_friction=0.15,
      last_manual_applied=3.0,
      last_manual_friction_applied=0.3,
    )
    tp1 = SimpleNamespace(latAccelFactor=3.0, friction=0.3)
    tp2 = SimpleNamespace(latAccelFactor=3.0, friction=0.3)
    r1 = v1.update_override_torque_params(tp1, v_ego=20.0)
    r2 = v2.update_override_torque_params(tp2, v_ego=20.0)
    assert r1 is True
    assert r2 is True
    assert tp1.latAccelFactor == pytest.approx(tp2.latAccelFactor)
    assert tp1.friction == pytest.approx(tp2.friction)

  def test_speed_aware_apply_mutates_and_restores(self):
    v1, v2 = self._make_instances()
    profile = {
      "version": 1,
      "restoreKey": {
        "carFingerprint": "test",
        "lateralTuning": "torque",
        "latAccelFactor": 2.0,
        "friction": 0.15,
      },
      "anchors": [15.0, 25.0],
      "ratios": [1.1, 1.2],
      "confidence": [1.0, 1.0],
      "points": [1000, 1000],
      "globalLatAccelFactor": 2.0,
      "globalFriction": 0.15,
    }
    CP = _make_cp()
    CP.lateralTuning = SimpleNamespace(which=lambda: "torque", torque=SimpleNamespace(latAccelFactor=2.0, friction=0.15))
    self._sync_state(
      v1, v2,
      CP=CP,
      enforce_torque_control_toggle=True,
      torque_override_enabled=False,
      _live_torque_enabled=True,
      _speed_mode="apply",
      _speed_profile=profile,
      base_latAccelFactor=2.0,
      base_friction=0.15,
    )
    tp1 = _torque_params()
    tp2 = _torque_params()
    r1 = v1.update_override_torque_params(tp1, v_ego=20.0)
    r2 = v2.update_override_torque_params(tp2, v_ego=20.0)
    assert r1 is True
    assert r2 is True
    assert tp1.latAccelFactor == pytest.approx(tp2.latAccelFactor)
    assert v1.last_speed_applied == pytest.approx(v2.last_speed_applied)

  def test_repeated_manual_toggle_preserves_mutation_order(self):
    v1, v2 = self._make_instances()
    base = {"base_latAccelFactor": 2.0, "base_friction": 0.15}
    override = {
      "enforce_torque_control_toggle": True,
      "torque_override_enabled": True,
      "_custom_torque_params": True,
      "_manual_latAccelFactor": 2.5,
      "_manual_friction": 0.25,
      "_manual_override_values_valid": True,
    }
    self._sync_state(v1, v2, **base, **override)
    tp1 = _torque_params()
    tp2 = _torque_params()
    assert v1.update_override_torque_params(tp1, v_ego=20.0) == v2.update_override_torque_params(tp2, v_ego=20.0)
    self._sync_state(v1, v2, torque_override_enabled=False, _manual_override_values_valid=False)
    assert v1.update_override_torque_params(tp1, v_ego=20.0) == v2.update_override_torque_params(tp2, v_ego=20.0)
    self._sync_state(v1, v2, **override)
    assert v1.update_override_torque_params(tp1, v_ego=20.0) == v2.update_override_torque_params(tp2, v_ego=20.0)
    np.testing.assert_allclose([tp1.latAccelFactor, tp1.friction], [tp2.latAccelFactor, tp2.friction])


# -----------------------------------------------------------------------------
# Underresponse sentinel parity
# -----------------------------------------------------------------------------
class TestUnderresponseSentinelParity:
  def _make_instances(self, dt: float = 0.01):
    return (
      _MODULES["under_v1"].UnderresponseSentinel(dt),
      _MODULES["under_v2"].UnderresponseSentinel(dt),
    )

  def _update(self, v1, v2, **kwargs):
    return v1.update(**kwargs), v2.update(**kwargs)

  def test_reset_debug_matches(self):
    v1, v2 = self._make_instances()
    d1 = v1.reset()
    d2 = v2.reset()
    assert _debug_eq(d1, d2)

  def test_inactive_returns_inactive_block(self):
    v1, v2 = self._make_instances()
    d1, d2 = self._update(
      v1, v2,
      active=False,
      v_ego=20.0,
      steering_pressed=False,
      steer_limited_by_safety=False,
      curvature_limited=False,
      setpoint=1.0,
      measurement=0.5,
      lateral_accel_deadzone=0.0,
      output_torque=0.3,
      steer_max=1.0,
      roll=0.0,
    )
    assert d1.block_mask == d2.block_mask == _MODULES["under_v1"].BLOCK_INACTIVE
    assert d1.error == pytest.approx(d2.error)

  def test_low_speed_block_matches(self):
    v1, v2 = self._make_instances()
    d1, d2 = self._update(
      v1, v2,
      active=True,
      v_ego=5.0,
      steering_pressed=False,
      steer_limited_by_safety=False,
      curvature_limited=False,
      setpoint=1.0,
      measurement=0.2,
      lateral_accel_deadzone=0.0,
      output_torque=0.3,
      steer_max=1.0,
      roll=0.0,
    )
    assert d1.block_mask == d2.block_mask == _MODULES["under_v1"].BLOCK_LOW_SPEED

  def test_active_trigger_matches(self):
    v1, v2 = self._make_instances()
    kwargs = {
      "active": True,
      "v_ego": 20.0,
      "steering_pressed": False,
      "steer_limited_by_safety": False,
      "curvature_limited": False,
      "setpoint": 1.0,
      "measurement": 0.2,
      "lateral_accel_deadzone": 0.0,
      "output_torque": 0.3,
      "steer_max": 1.0,
      "roll": 0.0,
    }
    d1 = d2 = None
    for _ in range(300):
      d1, d2 = self._update(v1, v2, **kwargs)
      if d1.active:
        break
    assert d1 is not None and d2 is not None
    assert d1.active is True
    assert d2.active is True
    assert d1.block_mask == d2.block_mask == 0
    assert d1.shadow_lat_accel == pytest.approx(d2.shadow_lat_accel)
    assert d1.severity == pytest.approx(d2.severity)
    assert d1.error_filtered == pytest.approx(d2.error_filtered)

  def test_each_blocker_category_matches(self):
    base = {
      "active": True,
      "v_ego": 20.0,
      "steering_pressed": False,
      "steer_limited_by_safety": False,
      "curvature_limited": False,
      "setpoint": 1.0,
      "measurement": 0.2,
      "lateral_accel_deadzone": 0.0,
      "output_torque": 0.3,
      "steer_max": 1.0,
      "roll": 0.0,
    }
    blockers = [
      ("steering_pressed", {"steering_pressed": True}, "BLOCK_STEERING_PRESSED"),
      ("steer_limited", {"steer_limited_by_safety": True}, "BLOCK_STEER_LIMITED"),
      ("curvature_limited", {"curvature_limited": True}, "BLOCK_CURVATURE_LIMITED"),
      ("torque_saturated", {"output_torque": 0.95, "steer_max": 1.0}, "BLOCK_TORQUE_SATURATED"),
      ("roll_too_high", {"roll": 0.20}, "BLOCK_ROLL_TOO_HIGH"),
      ("desired_too_small", {"setpoint": 0.10}, "BLOCK_DESIRED_TOO_SMALL"),
      ("sign_mismatch", {"setpoint": 1.0, "measurement": 1.5}, "BLOCK_SIGN_MISMATCH"),
    ]
    for _name, overrides, const_name in blockers:
      v1, v2 = self._make_instances()
      kwargs = {**base, **overrides}
      d1, d2 = self._update(v1, v2, **kwargs)
      expected = getattr(_MODULES["under_v1"], const_name)
      assert d1.block_mask == d2.block_mask
      assert d1.block_mask & expected, (_name, d1.block_mask, expected)
      assert d2.block_mask & expected, (_name, d2.block_mask, expected)

  def test_ewma_state_decay_matches(self):
    v1, v2 = self._make_instances()
    kwargs = {
      "active": True,
      "v_ego": 20.0,
      "steering_pressed": False,
      "steer_limited_by_safety": False,
      "curvature_limited": False,
      "setpoint": 1.0,
      "measurement": 0.2,
      "lateral_accel_deadzone": 0.0,
      "output_torque": 0.3,
      "steer_max": 1.0,
      "roll": 0.0,
    }
    # Build EWMA then block to trigger soft decay.
    for _ in range(30):
      d1, d2 = self._update(v1, v2, **kwargs)
    assert d1.error_filtered == pytest.approx(d2.error_filtered)
    blocked_kwargs = {**kwargs, "steering_pressed": True}
    d1, d2 = self._update(v1, v2, **blocked_kwargs)
    assert v1.error_filtered == pytest.approx(v2.error_filtered)
    assert v1.above_threshold_frames == v2.above_threshold_frames == 0

  def test_debug_writers_match(self):
    v1, v2 = self._make_instances()
    d1 = v1.reset()
    d2 = v2.reset()
    pid1 = SimpleNamespace()
    pid2 = SimpleNamespace()
    _MODULES["under_v1"].write_underresponse_debug(pid1, d1)
    _MODULES["under_v2"].write_underresponse_debug(pid2, d2)
    assert pid1.__dict__ == pid2.__dict__

  def test_block_names_stable(self):
    assert _MODULES["under_v1"].BLOCK_NAMES == _MODULES["under_v2"].BLOCK_NAMES
    for name in ("BLOCK_INACTIVE", "BLOCK_LOW_SPEED", "BLOCK_TORQUE_SATURATED", "BLOCK_INVALID_INPUT"):
      assert getattr(_MODULES["under_v1"], name) == getattr(_MODULES["under_v2"], name)
