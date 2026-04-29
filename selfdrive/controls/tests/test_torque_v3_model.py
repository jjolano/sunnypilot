import math

from openpilot.sunnypilot.selfdrive.controls.lib.torque_v3_model import (
  TorqueModelAdapter,
  TorqueModelMode,
  TorqueModelParams,
)


class NativeParams:
  latAccelFactor = 3.0
  latAccelOffset = 0.05
  friction = 0.12


def test_synthetic_model_uses_conservative_linear_params():
  adapter = TorqueModelAdapter.synthetic()

  torque = adapter.torque_from_lateral_accel(0.6)
  lat_accel = adapter.lateral_accel_from_torque(torque)

  assert adapter.mode == TorqueModelMode.synthetic
  assert 0.0 < torque < 0.5
  assert math.isclose(lat_accel, 0.6, rel_tol=1e-6)


def test_native_model_uses_callbacks():
  adapter = TorqueModelAdapter.native(
    torque_params=NativeParams(),
    torque_from_lateral_accel=lambda lat_accel, params: (lat_accel - params.latAccelOffset) / params.latAccelFactor,
    lateral_accel_from_torque=lambda torque, params: torque * params.latAccelFactor + params.latAccelOffset,
  )

  torque = adapter.torque_from_lateral_accel(0.65)

  assert adapter.mode == TorqueModelMode.native
  assert math.isclose(torque, 0.2, rel_tol=1e-6)
  assert math.isclose(adapter.lateral_accel_from_torque(torque), 0.65, rel_tol=1e-6)


def test_lateral_accel_from_torque_clamps_normalized_torque_input():
  adapter = TorqueModelAdapter.synthetic()

  assert math.isclose(adapter.lateral_accel_from_torque(2.0), 2.5, rel_tol=1e-6)
  assert math.isclose(adapter.lateral_accel_from_torque(-2.0), -2.5, rel_tol=1e-6)


def test_learned_model_can_apply_bounded_residual():
  adapter = TorqueModelAdapter.synthetic()
  adapter.update_learned_params(TorqueModelParams(lat_accel_factor=2.0, lat_accel_offset=0.0, friction=0.1), confidence=0.8)
  adapter.set_residual(0.08)

  assert adapter.mode == TorqueModelMode.learned
  assert math.isclose(adapter.torque_from_lateral_accel(0.8), 0.48, rel_tol=1e-6)


def test_invalid_learned_params_keep_previous_safe_model():
  adapter = TorqueModelAdapter.synthetic()
  previous = adapter.torque_from_lateral_accel(0.5)

  accepted = adapter.update_learned_params(TorqueModelParams(lat_accel_factor=0.0, lat_accel_offset=0.0, friction=0.1), confidence=0.9)

  assert not accepted
  assert adapter.mode == TorqueModelMode.synthetic
  assert math.isclose(adapter.torque_from_lateral_accel(0.5), previous, rel_tol=1e-6)
