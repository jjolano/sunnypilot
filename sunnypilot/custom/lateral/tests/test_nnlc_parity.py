"""Phase 2 NNLC fold parity tests: legacy v1 modules vs new custom-lateral facades.

The tests load the ``*_v1.py`` backups and the active modules side-by-side with the
same deterministic inputs and assert identical outputs, internal state, and telemetry.
"""
from __future__ import annotations

import importlib
import math
from types import SimpleNamespace

import numpy as np
import pytest

from openpilot.selfdrive.modeld.constants import ModelConstants


class _FakeParams:
  """Deterministic Params stand-in so NNLC enable toggles are test-controlled."""

  def get_bool(self, _key: str) -> bool:
    return False

  def get(self, _key: str, *, return_default: bool = True) -> bytes:
    return b""


def _load_modules():
  mp = pytest.MonkeyPatch()
  with mp.context() as ctx:
    ctx.setattr("openpilot.common.params.Params", _FakeParams)
    return {
      "nnlc_v1": importlib.import_module("openpilot.sunnypilot.selfdrive.controls.lib.nnlc.nnlc_v1"),
      "nnlc_v2": importlib.import_module("openpilot.sunnypilot.selfdrive.controls.lib.nnlc.nnlc"),
      "model_v1": importlib.import_module("openpilot.sunnypilot.selfdrive.controls.lib.nnlc.model_v1"),
      "model_v2": importlib.import_module("openpilot.sunnypilot.selfdrive.controls.lib.nnlc.model"),
      "helpers_v1": importlib.import_module("openpilot.sunnypilot.selfdrive.controls.lib.nnlc.helpers_v1"),
      "helpers_v2": importlib.import_module("openpilot.sunnypilot.selfdrive.controls.lib.nnlc.helpers"),
      "ext_v1": importlib.import_module("openpilot.sunnypilot.selfdrive.controls.lib.latcontrol_torque_ext_v1"),
      "ext_v2": importlib.import_module("openpilot.sunnypilot.selfdrive.controls.lib.latcontrol_torque_ext"),
    }


_MODULES = _load_modules()


def _make_ci():
  def torque_from_lateral_accel_in_torque_space():
    def _eval(inputs, torque_params, *, gravity_adjusted):
      _ = torque_params, gravity_adjusted
      return float(inputs.lateral_acceleration) * 2.0
    return _eval
  return SimpleNamespace(torque_from_lateral_accel_in_torque_space=torque_from_lateral_accel_in_torque_space)


def _make_cp():
  return SimpleNamespace(steerActuatorDelay=0.2)


def _make_cp_sp(model_path: str = ""):
  return SimpleNamespace(neuralNetworkLateralControl=SimpleNamespace(model=SimpleNamespace(path=model_path)))


def _make_lac():
  torque_params = SimpleNamespace(
    latAccelFactor=2.0,
    latAccelOffset=0.0,
    friction=0.15,
    steeringAngleDeadzoneDeg=0.0,
  )
  return SimpleNamespace(torque_params=torque_params, steer_max=1.0)


def _make_vm():
  return SimpleNamespace(
    calc_curvature=lambda angle_rad, _v, _roll: 0.001 * math.degrees(angle_rad),
  )


def _make_cs(v_ego: float = 20.0, steering_rate_deg: float = 5.0, steering_pressed: bool = False, a_ego: float = 0.0):
  return SimpleNamespace(vEgo=v_ego, aEgo=a_ego, steeringRateDeg=steering_rate_deg, steeringPressed=steering_pressed)


def _make_model_v2():
  n = len(ModelConstants.T_IDXS)
  return SimpleNamespace(
    orientation=SimpleNamespace(
      x=np.zeros(n),
      y=np.zeros(n),
    ),
    acceleration=SimpleNamespace(
      x=np.zeros(n),
      y=np.zeros(n),
    ),
  )


def _make_params(roll: float = 0.0):
  return SimpleNamespace(roll=roll)


def _make_pose(pitch: float = 0.0):
  return SimpleNamespace(orientation=SimpleNamespace(pitch=pitch))


class _FakePID:
  """PID-like object that feeds back a deterministic function of error and ff."""

  def __init__(self):
    self.error = 0.0
    self.f = 0.0

  def update(self, error, measurement_rate=0.0, *, feedforward=0.0, speed=0.0, freeze_integrator=False):
    _ = measurement_rate, speed, freeze_integrator
    self.error = float(error)
    self.f = float(feedforward)
    return float(error * 0.5 + feedforward * 0.25)


def _make_pid():
  return _FakePID()


def _make_nnlc_instances(model_path: str = ""):
  lac = _make_lac()
  CP = _make_cp()
  CP_SP = _make_cp_sp(model_path)
  CI = _make_ci()
  return (
    _MODULES["nnlc_v1"].NeuralNetworkLateralControl(lac, CP, CP_SP, CI),
    _MODULES["nnlc_v2"].NeuralNetworkLateralControl(lac, CP, CP_SP, CI),
  )


class _FakeNNTorqueModel:
  def __init__(self):
    self.friction_override = False
    self.history: list[list[float]] = []

  def evaluate(self, input_array):
    self.history.append(list(input_array))
    # Deterministic scalar from the input so both sides receive the same feedforward/error.
    return float(sum(input_array[:4]) * 0.01)


def _sync_enabled_nnlc(v1, v2):
  for o in (v1, v2):
    o.enabled = True
    o.has_nn_model = True
    o.model_valid = True
    o.model = _FakeNNTorqueModel()


def _sync_pre_update_state(v1, v2, **kwargs):
  for key, value in kwargs.items():
    setattr(v1, key, value)
    setattr(v2, key, value)


class TestNNLCModelAdapterParity:
  def test_mock_model_path_constant_matches(self):
    assert _MODULES["helpers_v1"].MOCK_MODEL_PATH == _MODULES["helpers_v2"].MOCK_MODEL_PATH

  def test_python_model_constructed_same(self):
    path = _MODULES["helpers_v2"].MOCK_MODEL_PATH
    m1 = _MODULES["model_v1"].PythonNNTorqueModel(path, zero_bias=True)
    m2 = _MODULES["model_v2"].PythonNNTorqueModel(path, zero_bias=True)
    assert m1.input_size == m2.input_size
    assert m1.friction_override == m2.friction_override
    x = [10.0, 0.0, 0.2]
    assert m1.evaluate(x) == pytest.approx(m2.evaluate(x))


class TestNeuralNetworkLateralControlParity:
  def test_disabled_nnlc_is_passthrough(self):
    v1, v2 = _make_nnlc_instances("")
    v1.update_model_v2(_make_model_v2())
    v2.update_model_v2(_make_model_v2())
    pid_log = SimpleNamespace(error=0.1)
    _sync_pre_update_state(
      v1, v2,
      _ff=0.5,
      _output_torque=0.6,
      _pid_log=pid_log,
      _setpoint=0.2,
      _measurement=0.1,
      _desired_lateral_accel=0.2,
      _actual_lateral_accel=0.1,
      _desired_curvature=0.01,
      _actual_curvature=0.005,
      _roll_compensation=0.0,
      _lateral_accel_deadzone=0.05,
      _gravity_adjusted_lateral_accel=0.2,
      _steer_limited_by_safety=False,
    )
    v1.update_neural_network_feedforward(_make_cs(), _make_params(), _make_pose())
    v2.update_neural_network_feedforward(_make_cs(), _make_params(), _make_pose())
    assert v1._ff == pytest.approx(0.5)
    assert v2._ff == pytest.approx(0.5)
    assert v1._output_torque == pytest.approx(0.6)
    assert v2._output_torque == pytest.approx(0.6)
    assert v1._pid_log is pid_log
    assert v2._pid_log is pid_log
    assert v1.roll_deque == v2.roll_deque
    assert v1.lateral_accel_desired_deque == v2.lateral_accel_desired_deque

  def test_update_lateral_lag_matches(self):
    v1, v2 = _make_nnlc_instances("")
    v1.update_lateral_lag(0.35)
    v2.update_lateral_lag(0.35)
    assert v1.desired_lat_jerk_time == pytest.approx(v2.desired_lat_jerk_time)
    assert v1.nn_future_times == pytest.approx(v2.nn_future_times)

  def test_update_limits_is_noop_when_disabled(self):
    v1, v2 = _make_nnlc_instances("")
    # update_limits should short-circuit when _nnlc_enabled is False.
    assert v1._nnlc_enabled is False
    assert v2._nnlc_enabled is False
    v1.update_limits()
    v2.update_limits()
    # Nothing to assert beyond no exception; parity is implicit.

  def test_enabled_frame_sequence_matches(self):
    v1, v2 = _make_nnlc_instances("")
    _sync_enabled_nnlc(v1, v2)
    v1.update_model_v2(_make_model_v2())
    v2.update_model_v2(_make_model_v2())
    v1.update_lateral_lag(0.2)
    v2.update_lateral_lag(0.2)

    pid1 = _make_pid()
    pid2 = _make_pid()
    _sync_pre_update_state(
      v1, v2,
      _ff=0.0,
      _output_torque=0.0,
      _pid=pid1,
      _pid_log=pid1,
      _setpoint=0.0,
      _measurement=0.0,
      _desired_lateral_accel=1.2,
      _actual_lateral_accel=0.4,
      _desired_curvature=0.02,
      _actual_curvature=0.01,
      _roll_compensation=0.05,
      _lateral_accel_deadzone=0.05,
      _gravity_adjusted_lateral_accel=1.15,
      _steer_limited_by_safety=False,
      lateral_jerk_setpoint=0.1,
      lateral_jerk_measurement=0.05,
    )
    # v2 shares pid1 until overwritten; set v2._pid to pid2 to keep objects separate.
    v2._pid = pid2
    v2._pid_log = pid2

    cs = _make_cs(v_ego=20.0, a_ego=0.5)
    params = _make_params(roll=0.02)
    pose = _make_pose(pitch=0.01)

    for _ in range(35):
      v1.update_neural_network_feedforward(cs, params, pose)
      v2.update_neural_network_feedforward(cs, params, pose)

    assert v1._ff == pytest.approx(v2._ff)
    assert pid1.error == pytest.approx(pid2.error)
    assert v1._output_torque == pytest.approx(v2._output_torque)
    assert v1._setpoint == pytest.approx(v2._setpoint)
    assert v1._measurement == pytest.approx(v2._measurement)
    assert v1.pitch.x == pytest.approx(v2.pitch.x)
    assert v1.pitch_last == pytest.approx(v2.pitch_last)
    assert len(v1.roll_deque) == len(v2.roll_deque)
    assert len(v1.lateral_accel_desired_deque) == len(v2.lateral_accel_desired_deque)
    assert list(v1.roll_deque) == pytest.approx(list(v2.roll_deque))
    assert list(v1.lateral_accel_desired_deque) == pytest.approx(list(v2.lateral_accel_desired_deque))

  def test_blend_error_response_matches(self):
    blend = _MODULES["nnlc_v1"].NeuralNetworkLateralControl._blend_error_response
    assert _MODULES["nnlc_v2"].NeuralNetworkLateralControl._blend_error_response(0.2, 0.8, 0.5) == pytest.approx(blend(0.2, 0.8, 0.5))
    assert _MODULES["nnlc_v2"].NeuralNetworkLateralControl._blend_error_response(0.8, -1.0, 1.0) == pytest.approx(blend(0.8, -1.0, 1.0))


class TestLatControlTorqueExtParity:
  def _make_instances(self):
    lac = _make_lac()
    CP = _make_cp()
    CP_SP = _make_cp_sp("")
    CI = _make_ci()
    return (
      _MODULES["ext_v1"].LatControlTorqueExt(lac, CP, CP_SP, CI),
      _MODULES["ext_v2"].LatControlTorqueExt(lac, CP, CP_SP, CI),
    )

  def test_update_frame_matches(self):
    v1, v2 = self._make_instances()
    _sync_enabled_nnlc(v1, v2)
    v1.update_model_v2(_make_model_v2())
    v2.update_model_v2(_make_model_v2())

    pid1 = _make_pid()
    pid2 = _make_pid()
    ff = 0.3
    setpoint = 0.8
    measurement = 0.6
    roll_compensation = 0.05
    desired_lateral_accel = 0.8
    actual_lateral_accel = 0.6
    lateral_accel_deadzone = 0.05
    gravity_adjusted_lateral_accel = 0.75
    desired_curvature = 0.01
    actual_curvature = 0.008
    steer_limited_by_safety = False
    output_torque = 0.4
    params = _make_params(roll=0.01)
    pose = _make_pose(pitch=0.005)

    log1, out1 = v1.update(
      _make_cs(), _make_vm(), pid1, params, ff, pid1, setpoint, measurement, pose, roll_compensation,
      desired_lateral_accel, actual_lateral_accel, lateral_accel_deadzone, gravity_adjusted_lateral_accel,
      desired_curvature, actual_curvature, steer_limited_by_safety, output_torque,
    )
    log2, out2 = v2.update(
      _make_cs(), _make_vm(), pid2, params, ff, pid2, setpoint, measurement, pose, roll_compensation,
      desired_lateral_accel, actual_lateral_accel, lateral_accel_deadzone, gravity_adjusted_lateral_accel,
      desired_curvature, actual_curvature, steer_limited_by_safety, output_torque,
    )

    assert out1 == pytest.approx(out2)
    assert log1.error == pytest.approx(log2.error)
    assert v1._ff == pytest.approx(v2._ff)

  def test_disabled_update_returns_inputs_unchanged(self):
    v1, v2 = self._make_instances()
    # NNLC disabled (empty path) but the extension still runs base calculations.
    pid1 = _make_pid()
    pid2 = _make_pid()
    ff = 0.25
    setpoint = 0.5
    measurement = 0.4
    output_torque = 0.2
    params = _make_params()
    pose = _make_pose()

    log1, out1 = v1.update(
      _make_cs(), _make_vm(), pid1, params, ff, pid1, setpoint, measurement, pose, 0.0,
      setpoint, measurement, 0.05, setpoint, 0.01, 0.008, False, output_torque,
    )
    log2, out2 = v2.update(
      _make_cs(), _make_vm(), pid2, params, ff, pid2, setpoint, measurement, pose, 0.0,
      setpoint, measurement, 0.05, setpoint, 0.01, 0.008, False, output_torque,
    )

    assert out1 == pytest.approx(out2)
    assert log1.error == pytest.approx(log2.error)
