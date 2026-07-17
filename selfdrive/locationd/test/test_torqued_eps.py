import numpy as np
import pytest
from cereal import car, messaging
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.locationd.torqued import TorqueEstimator


def _build_live_pose(t: float, lateral_accel: float, v_ego: float = 25.0, roll_deg: float = 0.0):
  livePose = messaging.new_message('livePose').livePose
  livePose.orientationNED = {'x': float(np.deg2rad(roll_deg)), 'valid': True}
  livePose.angularVelocityDevice = {'z': float(lateral_accel / v_ego), 'valid': True}
  livePose.inputsOK, livePose.sensorsOK, livePose.posenetOK = True, True, True
  livePose.timestamp = int(t * 1e9)
  return livePose


def _feed(est: TorqueEstimator, t: float, steer: float, lateral_accel: float, *,
          v_ego: float = 25.0, steering_pressed: bool = False, steering_rate_deg: float = 0.0,
          steering_torque_eps: float | None = None):
  carControl = messaging.new_message('carControl').carControl
  carOutput = messaging.new_message('carOutput').carOutput
  carState = messaging.new_message('carState').carState
  carControl.latActive = True
  carState.vEgo = v_ego
  carState.steeringPressed = steering_pressed
  carState.steeringRateDeg = steering_rate_deg
  carOutput.actuatorsOutput.torque = float(-steer)
  if steering_torque_eps is not None:
    carState.steeringTorqueEps = float(steering_torque_eps)
  livePose = _build_live_pose(t, lateral_accel, v_ego=v_ego)
  for which, msg in (('carControl', carControl), ('carOutput', carOutput), ('carState', carState), ('livePose', livePose)):
    est.handle_log(t, which, msg)


def _warmup_samples() -> int:
  return int(6.0 / DT_MDL)


def _car_params_with_eps():
  cp = car.CarParams()
  cp.brand = 'toyota'
  return cp


def test_eps_fields_initialized_to_safe_defaults():
  est = TorqueEstimator(car.CarParams())
  msg = est.get_msg()
  ltp = msg.liveTorqueParameters
  assert ltp.epsObserved is False
  assert ltp.epsSampleCount == 0
  assert ltp.epsTorqueLatest == 0.0
  assert ltp.epsCommandTorqueLatest == 0.0
  assert ltp.epsDeltaMean == 0.0
  assert ltp.epsDeltaMax == 0.0


def test_eps_fields_update_with_valid_samples():
  est = TorqueEstimator(_car_params_with_eps())
  n = _warmup_samples()
  steer = 0.1
  # native Toyota units, carcontroller sign convention: -180/1500 -> 0.12 in
  # torqued's normalized steer convention (steer = -actuatorsOutput.torque)
  eps_torque_native = -180.0
  eps_torque_normalized = 0.12
  for i in range(n):
    _feed(est, i * DT_MDL, steer=steer, lateral_accel=0.2, steering_torque_eps=eps_torque_native)

  msg = est.get_msg()
  ltp = msg.liveTorqueParameters
  assert ltp.epsObserved is True
  assert ltp.epsSampleCount > 0
  assert ltp.epsTorqueLatest == pytest.approx(eps_torque_normalized, abs=1e-6)
  assert ltp.epsCommandTorqueLatest == pytest.approx(steer, abs=1e-6)
  expected_delta = abs(steer - eps_torque_normalized)
  assert ltp.epsDeltaMean == pytest.approx(expected_delta, abs=1e-6)
  assert ltp.epsDeltaMax == pytest.approx(expected_delta, abs=1e-6)


def test_eps_missing_values_do_not_contaminate_stats():
  est = TorqueEstimator(_car_params_with_eps())
  n = _warmup_samples()
  steer = 0.1
  for i in range(n):
    _feed(est, i * DT_MDL, steer=steer, lateral_accel=0.2, steering_torque_eps=float('nan'))

  msg = est.get_msg()
  ltp = msg.liveTorqueParameters
  assert ltp.epsObserved is False
  assert ltp.epsSampleCount == 0
  assert ltp.epsTorqueLatest == 0.0
  assert ltp.epsCommandTorqueLatest == pytest.approx(steer, abs=1e-6)
  assert ltp.epsDeltaMean == 0.0
  assert ltp.epsDeltaMax == 0.0


def test_eps_unsupported_brand_does_not_mark_observed():
  est = TorqueEstimator(car.CarParams())
  n = _warmup_samples()
  steer = 0.1
  for i in range(n):
    _feed(est, i * DT_MDL, steer=steer, lateral_accel=0.2, steering_torque_eps=0.12)

  ltp = est.get_msg().liveTorqueParameters
  assert ltp.epsObserved is False
  assert ltp.epsSampleCount == 0
  assert ltp.epsTorqueLatest == 0.0
  assert ltp.epsCommandTorqueLatest == pytest.approx(steer, abs=1e-6)
  assert ltp.epsDeltaMean == 0.0
  assert ltp.epsDeltaMax == 0.0


def test_eps_shadow_does_not_change_learning_points():
  with_eps = TorqueEstimator(_car_params_with_eps())
  without_eps = TorqueEstimator(_car_params_with_eps())
  n = _warmup_samples()
  steer = 0.1
  for i in range(n):
    t = i * DT_MDL
    _feed(with_eps, t, steer=steer, lateral_accel=0.2, steering_torque_eps=120.0)
    _feed(without_eps, t, steer=steer, lateral_accel=0.2, steering_torque_eps=float('nan'))

  assert with_eps.get_msg().liveTorqueParameters.epsObserved is True
  assert without_eps.get_msg().liveTorqueParameters.epsObserved is False
  assert len(with_eps.filtered_points) == len(without_eps.filtered_points)
  np.testing.assert_allclose(with_eps.filtered_points.get_points(), without_eps.filtered_points.get_points())


def test_eps_stats_reset():
  est = TorqueEstimator(_car_params_with_eps())
  n = _warmup_samples()
  for i in range(n):
    _feed(est, i * DT_MDL, steer=0.1, lateral_accel=0.2, steering_torque_eps=0.12)

  assert est.get_msg().liveTorqueParameters.epsObserved is True
  est.reset()
  msg = est.get_msg()
  ltp = msg.liveTorqueParameters
  assert ltp.epsObserved is False
  assert ltp.epsSampleCount == 0
  assert ltp.epsTorqueLatest == 0.0
  assert ltp.epsCommandTorqueLatest == 0.0
  assert ltp.epsDeltaMean == 0.0
  assert ltp.epsDeltaMax == 0.0
