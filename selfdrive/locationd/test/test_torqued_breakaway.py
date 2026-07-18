import numpy as np
from cereal import car, messaging
from openpilot.selfdrive.locationd.torqued import TorqueEstimator

DT = 0.01


def _make_estimator(mode="shadow"):
  cp = car.CarParams()
  cp.brand = 'toyota'
  cp.carFingerprint = 'TOYOTA_RAV4_TSS2'
  cp.lateralTuning.init('torque')
  est = TorqueEstimator(cp)
  est.friction_breakaway_mode = mode
  return est


def _feed_car_state(est, t, rate_deg, eps_native, v_ego=20.0, lat_active=True, pressed=False, driver_torque=0.0):
  carControl = messaging.new_message('carControl').carControl
  carControl.latActive = lat_active
  est.handle_log(t, 'carControl', carControl)
  carState = messaging.new_message('carState').carState
  carState.vEgo = v_ego
  carState.steeringPressed = pressed
  carState.steeringRateDeg = float(rate_deg)
  carState.steeringTorqueEps = float(eps_native)
  carState.steeringTorque = float(driver_torque)
  est.handle_log(t, 'carState', carState)


def _dwell_then_jump(est, t0, eps_native, jump_rate=3.0, dwell_s=0.5, driver_torque=0.0):
  t = t0
  for _ in range(int(dwell_s / DT)):
    _feed_car_state(est, t, 0.0, eps_native, driver_torque=driver_torque)
    t += DT
  for _ in range(10):
    _feed_car_state(est, t, jump_rate, eps_native, driver_torque=driver_torque)
    t += DT
  return t


def test_breakout_recorded_per_direction():
  est = _make_estimator()
  t = _dwell_then_jump(est, 0.0, eps_native=300.0, jump_rate=3.0)
  t = _dwell_then_jump(est, t, eps_native=-225.0, jump_rate=-3.0)
  tele = est.breakaway_telemetry()
  assert tele['events'] == 2
  assert tele['left'] == np.float64(300.0 / 1500.0)
  assert tele['right'] == np.float64(225.0 / 1500.0)


def test_no_record_without_dwell():
  est = _make_estimator()
  t = 0.0
  for _ in range(200):  # continuous motion, no dwell
    _feed_car_state(est, t, 3.0, 300.0)
    t += DT
  assert est.breakaway_telemetry()['events'] == 0


def test_gentle_motion_resets_dwell_without_recording():
  est = _make_estimator()
  t = 0.0
  for _ in range(60):
    _feed_car_state(est, t, 0.0, 300.0)
    t += DT
  _feed_car_state(est, t, 1.0, 300.0)  # 0.5 <= rate < 1.5: reset, no record
  assert est.breakaway_telemetry()['events'] == 0


def test_subthreshold_driver_torque_suppresses_recording():
  # 60 units is below the steeringPressed threshold (~100) but contaminates EPS
  est = _make_estimator()
  _dwell_then_jump(est, 0.0, eps_native=300.0, driver_torque=60.0)
  assert est.breakaway_telemetry()['events'] == 0


def test_driver_noise_early_in_dwell_suppresses_recording():
  # noisy hands early in the dwell, quiet at the jump: still discarded
  est = _make_estimator()
  t = 0.0
  for _ in range(20):
    _feed_car_state(est, t, 0.0, 300.0, driver_torque=60.0)
    t += DT
  for _ in range(40):
    _feed_car_state(est, t, 0.0, 300.0, driver_torque=0.0)
    t += DT
  for _ in range(10):
    _feed_car_state(est, t, 3.0, 300.0, driver_torque=0.0)
    t += DT
  assert est.breakaway_telemetry()['events'] == 0


def test_quiet_driver_torque_still_records():
  est = _make_estimator()
  _dwell_then_jump(est, 0.0, eps_native=300.0, driver_torque=10.0)
  assert est.breakaway_telemetry()['events'] == 1


def test_off_mode_records_nothing():
  est = _make_estimator(mode="off")
  _dwell_then_jump(est, 0.0, eps_native=300.0)
  assert est.breakaway_telemetry()['events'] == 0


def test_inactive_or_pressed_records_nothing():
  est = _make_estimator()
  t = 0.0
  for _ in range(60):
    _feed_car_state(est, t, 0.0, 300.0, lat_active=False)
    t += DT
  for _ in range(10):
    _feed_car_state(est, t, 3.0, 300.0, lat_active=False)
    t += DT
  assert est.breakaway_telemetry()['events'] == 0


def test_telemetry_published_in_message():
  est = _make_estimator()
  _dwell_then_jump(est, 0.0, eps_native=300.0)
  ltp = est.get_msg().liveTorqueParameters
  assert ltp.breakawayEvents == 1
  assert abs(ltp.breakawayLeftMedian - 0.2) < 1e-6
  assert ltp.breakawayRightMedian == 0.0


def test_profile_persisted_after_enough_events_both_directions():
  import json
  from openpilot.common.params import Params
  from openpilot.sunnypilot.custom.lateral.torque_safety import parse_breakaway_profile
  params = Params()
  params.remove("LatFrictionBreakawayParams")

  est = _make_estimator()
  t = 0.0
  for _ in range(20):
    t = _dwell_then_jump(est, t, eps_native=300.0, jump_rate=3.0)
    t = _dwell_then_jump(est, t, eps_native=-225.0, jump_rate=-3.0)

  est.maybe_persist_speed_profile(cache_write=True)
  payload = params.get("LatFrictionBreakawayParams")
  assert payload
  profile = parse_breakaway_profile(est.CP, json.loads(payload))
  assert profile is not None
  assert abs(profile['left'] - 0.2) < 1e-6
  assert abs(profile['right'] - 0.15) < 1e-6
  assert profile['events'] == 40
  params.remove("LatFrictionBreakawayParams")


def test_profile_not_persisted_below_event_floor():
  from openpilot.common.params import Params
  params = Params()
  params.remove("LatFrictionBreakawayParams")

  est = _make_estimator()
  t = 0.0
  for _ in range(5):
    t = _dwell_then_jump(est, t, eps_native=300.0, jump_rate=3.0)
  est.maybe_persist_speed_profile(cache_write=True)
  assert not params.get("LatFrictionBreakawayParams")
