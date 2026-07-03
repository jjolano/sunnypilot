import numpy as np
import pytest
from cereal import car, messaging
from openpilot.common.params import Params
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
          v_ego: float = 25.0, steering_pressed: bool = False, steering_rate_deg: float = 0.0):
  carControl = messaging.new_message('carControl').carControl
  carOutput = messaging.new_message('carOutput').carOutput
  carState = messaging.new_message('carState').carState
  carControl.latActive = True
  carState.vEgo = v_ego
  carState.steeringPressed = steering_pressed
  carState.steeringRateDeg = steering_rate_deg
  carOutput.actuatorsOutput.torque = float(-steer)
  livePose = _build_live_pose(t, lateral_accel, v_ego=v_ego)
  for which, msg in (('carControl', carControl), ('carOutput', carOutput), ('carState', carState), ('livePose', livePose)):
    est.handle_log(t, which, msg)


def test_torqued_shadow_counters_initialized():
  est = TorqueEstimator(car.CarParams())
  assert est.shadow_accepted == 0
  assert est.shadow_quarantined == 0
  assert est.shadow_rejected == 0


def _warmup_samples() -> int:
  # torqued requires HISTORY seconds of buffers before processing livePose.
  return int(6.0 / DT_MDL)


@pytest.fixture(autouse=True)
def cleanup_shadow_mode_params():
  params = Params()
  for key in (
    "LiveTorqueSpeedAdaptiveMode",
    "LiveTorqueLowSpeedShadow",
    "LiveTorqueSpeedAdaptiveParams",
    "RollCompGainMode",
    "RollCompGainParams",
  ):
    params.remove(key)
  yield
  for key in (
    "LiveTorqueSpeedAdaptiveMode",
    "LiveTorqueLowSpeedShadow",
    "LiveTorqueSpeedAdaptiveParams",
    "RollCompGainMode",
    "RollCompGainParams",
  ):
    params.remove(key)


def _make_estimator(speed_mode="shadow"):
  params = Params()
  params.put("LiveTorqueSpeedAdaptiveMode", speed_mode, block=True)
  est = TorqueEstimator(car.CarParams())
  est.update_use_params()
  return est


def test_torqued_shadow_counters_increment_on_clean_samples():
  est = _make_estimator("shadow")
  n = _warmup_samples()
  for i in range(n):
    _feed(est, i * DT_MDL, steer=0.1, lateral_accel=0.2)

  msg = est.get_msg()
  assert msg.liveTorqueParameters.shadowAccepted > 0
  assert msg.liveTorqueParameters.shadowQuarantined == 0
  assert msg.liveTorqueParameters.shadowRejected == 0


def test_torqued_shadow_counters_stay_zero_without_shadow_mode():
  est = TorqueEstimator(car.CarParams())
  n = _warmup_samples()
  for i in range(n):
    _feed(est, i * DT_MDL, steer=0.1, lateral_accel=0.2)

  msg = est.get_msg()
  assert msg.liveTorqueParameters.shadowAccepted == 0
  assert msg.liveTorqueParameters.shadowQuarantined == 0
  assert msg.liveTorqueParameters.shadowRejected == 0


def test_torqued_shadow_quarantine_via_steering_rate():
  est = _make_estimator("shadow")
  n = _warmup_samples()
  for i in range(n):
    _feed(est, i * DT_MDL, steer=0.1, lateral_accel=0.2, steering_rate_deg=150.0)

  msg = est.get_msg()
  assert msg.liveTorqueParameters.shadowQuarantined > 0
  assert msg.liveTorqueParameters.shadowRejected == 0


def test_torqued_shadow_quarantine_does_not_suppress_bucket_insertion():
  est = _make_estimator("shadow")
  n = _warmup_samples()
  # High steering rate triggers a shadow quarantine, but the sample still passes
  # the existing |lateral_accel| <= LAT_ACC_THRESHOLD gate and is inserted.
  for i in range(n):
    _feed(est, i * DT_MDL, steer=0.2, lateral_accel=0.5, steering_rate_deg=150.0)

  bucket_points_before = len(est.filtered_points)
  msg = est.get_msg()
  assert msg.liveTorqueParameters.shadowQuarantined > 0
  assert len(est.filtered_points) == bucket_points_before
  assert len(est.filtered_points) > 0


def test_torqued_shadow_reject_counter_updates_via_direct_call():
  est = TorqueEstimator(car.CarParams())
  from openpilot.sunnypilot.custom.lateral.disturbance_classifier import LateralSample
  sample = LateralSample(t=0.0, v_ego=25.0, lat_active=True, steering_pressed=True, actual_lateral_accel=0.2)
  est.shadow_classify_learning_point(sample)
  msg = est.get_msg()
  assert msg.liveTorqueParameters.shadowRejected > 0
