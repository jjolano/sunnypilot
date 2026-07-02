import json
import numpy as np
import pytest

from cereal import car, messaging
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.locationd.torqued import TorqueEstimator, MIN_BUCKET_POINTS, MIN_VEL, STEER_BUCKET_BOUNDS, STEER_MIN_THRESHOLD


@pytest.fixture(autouse=True)
def cleanup_roll_comp_params():
  params = Params()
  params.remove("RollCompGainMode")
  params.remove("RollCompGainParams")
  params.remove("LiveTorqueSpeedAdaptiveMode")
  params.remove("LiveTorqueLowSpeedShadow")
  params.remove("LiveTorqueSpeedAdaptiveParams")
  yield
  params.remove("RollCompGainMode")
  params.remove("RollCompGainParams")
  params.remove("LiveTorqueSpeedAdaptiveMode")
  params.remove("LiveTorqueLowSpeedShadow")
  params.remove("LiveTorqueSpeedAdaptiveParams")


def make_torque_cp():
  cp = car.CarParams.new_message()
  cp.brand = "toyota"
  cp.carFingerprint = "TOYOTA_CAMRY"
  cp.lateralTuning.init('torque')
  cp.lateralTuning.torque.latAccelFactor = 2.0
  cp.lateralTuning.torque.latAccelOffset = 0.0
  cp.lateralTuning.torque.friction = 0.2
  return cp


def _build_live_pose(t, lateral_accel, v_ego=25.0, roll_rad=0.0):
  livePose = messaging.new_message('livePose').livePose
  livePose.orientationNED = {'x': float(roll_rad), 'valid': True}
  livePose.angularVelocityDevice = {'z': float(lateral_accel / v_ego), 'valid': True}
  livePose.inputsOK, livePose.sensorsOK, livePose.posenetOK = True, True, True
  livePose.timestamp = int(t * 1e9)
  return livePose


def _feed(est, t, steer, lateral_accel, *, v_ego=25.0, steering_pressed=False, steering_rate_deg=0.0, roll_rad=0.0):
  carControl = messaging.new_message('carControl').carControl
  carOutput = messaging.new_message('carOutput').carOutput
  carState = messaging.new_message('carState').carState
  carControl.latActive = True
  carState.vEgo = v_ego
  carState.steeringPressed = steering_pressed
  carState.steeringRateDeg = steering_rate_deg
  carOutput.actuatorsOutput.torque = float(-steer)
  livePose = _build_live_pose(t, lateral_accel, v_ego=v_ego, roll_rad=roll_rad)
  for which, msg in (('carControl', carControl), ('carOutput', carOutput), ('carState', carState), ('livePose', livePose)):
    est.handle_log(t, which, msg)


def _warmup_samples():
  # torqued requires HISTORY seconds of buffers before processing livePose.
  return int(6.0 / DT_MDL)


def _bootstrap_filtered_points(est):
  """Make the base torque learner valid so roll-comp collection can proceed."""
  rng = np.random.default_rng(0)
  points_per_bucket = (1.5 * np.array(MIN_BUCKET_POINTS)).astype(int)
  for bound, n in zip(STEER_BUCKET_BOUNDS, points_per_bucket, strict=True):
    for _ in range(n):
      steer = rng.uniform(bound[0], bound[1])
      lat_accel = 2.0 * steer
      est.filtered_points.add_point(steer, lat_accel)
  # Run one get_msg so filtered_params settle and liveValid becomes true.
  est.get_msg()


def _make_estimator(mode="off"):
  params = Params()
  params.put("RollCompGainMode", mode, block=True)
  est = TorqueEstimator(make_torque_cp())
  est.update_use_params()
  return est


def test_roll_comp_off_collects_nothing():
  est = _make_estimator("off")
  _bootstrap_filtered_points(est)

  n = _warmup_samples()
  for i in range(n):
    roll_rad = np.deg2rad(1.0)
    lateral_accel = -np.sin(roll_rad) * 9.81
    steer = lateral_accel / 2.0
    _feed(est, i * DT_MDL, steer=steer, lateral_accel=lateral_accel, roll_rad=roll_rad)

  assert len(est.roll_comp_buckets.get_points()) == 0
  msg = est.get_msg()
  assert not msg.liveTorqueParameters.rollCompGainValid
  assert msg.liveTorqueParameters.rollCompGainPoints == 0


def test_roll_comp_shadow_collects_on_straight_frames():
  est = _make_estimator("shadow")
  _bootstrap_filtered_points(est)

  n = _warmup_samples()
  for i in range(n):
    roll_rad = np.deg2rad(0.8 if i % 2 == 0 else -0.8)
    lateral_accel = -np.sin(roll_rad) * 9.81
    steer = lateral_accel / 2.0
    _feed(est, i * DT_MDL, steer=steer, lateral_accel=lateral_accel, roll_rad=roll_rad)

  assert len(est.roll_comp_buckets.get_points()) > 0


def test_roll_comp_rejects_curved_frames():
  est = _make_estimator("shadow")
  _bootstrap_filtered_points(est)

  n = _warmup_samples()
  for i in range(n):
    roll_rad = np.deg2rad(0.5)
    # High yaw_rate makes v_ego * yaw_rate exceed the 0.15 threshold.
    lateral_accel = roll_rad * 9.81 + 1.0  # large yaw contribution
    steer = 0.0
    _feed(est, i * DT_MDL, steer=steer, lateral_accel=lateral_accel, roll_rad=roll_rad)

  assert len(est.roll_comp_buckets.get_points()) == 0


def test_roll_comp_rejects_steering_override():
  est = _make_estimator("shadow")
  _bootstrap_filtered_points(est)

  n = _warmup_samples()
  for i in range(n):
    roll_rad = np.deg2rad(0.8)
    lateral_accel = -np.sin(roll_rad) * 9.81
    steer = lateral_accel / 2.0
    _feed(est, i * DT_MDL, steer=steer, lateral_accel=lateral_accel, roll_rad=roll_rad, steering_pressed=True)

  assert len(est.roll_comp_buckets.get_points()) == 0


def test_roll_comp_rejects_low_speed_frames():
  est = _make_estimator("shadow")
  _bootstrap_filtered_points(est)

  n = _warmup_samples()
  for i in range(n):
    roll_rad = np.deg2rad(0.8)
    lateral_accel = -np.sin(roll_rad) * 9.81
    steer = lateral_accel / 2.0
    _feed(est, i * DT_MDL, steer=steer, lateral_accel=lateral_accel, roll_rad=roll_rad, v_ego=MIN_VEL - 1.0)

  assert len(est.roll_comp_buckets.get_points()) == 0


def test_roll_comp_rejects_high_steering_rate():
  est = _make_estimator("shadow")
  _bootstrap_filtered_points(est)

  n = _warmup_samples()
  for i in range(n):
    roll_rad = np.deg2rad(0.8)
    lateral_accel = -np.sin(roll_rad) * 9.81
    steer = lateral_accel / 2.0
    _feed(est, i * DT_MDL, steer=steer, lateral_accel=lateral_accel, roll_rad=roll_rad, steering_rate_deg=10.0)

  assert len(est.roll_comp_buckets.get_points()) == 0


def test_roll_comp_rejects_large_roll():
  est = _make_estimator("shadow")
  _bootstrap_filtered_points(est)

  n = _warmup_samples()
  for i in range(n):
    roll_rad = np.deg2rad(8.0)  # > 0.1 rad
    lateral_accel = -np.sin(roll_rad) * 9.81
    steer = lateral_accel / 2.0
    _feed(est, i * DT_MDL, steer=steer, lateral_accel=lateral_accel, roll_rad=roll_rad)

  assert len(est.roll_comp_buckets.get_points()) == 0


def test_roll_comp_rejects_when_base_learner_unconverged():
  est = _make_estimator("shadow")
  # Do NOT bootstrap filtered_points.

  n = _warmup_samples()
  for i in range(n):
    roll_rad = np.deg2rad(0.8)
    lateral_accel = -np.sin(roll_rad) * 9.81
    steer = lateral_accel / 2.0
    _feed(est, i * DT_MDL, steer=steer, lateral_accel=lateral_accel, roll_rad=roll_rad)

  assert not est.filtered_points.is_valid()
  assert len(est.roll_comp_buckets.get_points()) == 0


def test_roll_comp_persist_only_when_cache_write_enabled():
  est = _make_estimator("shadow")
  _bootstrap_filtered_points(est)

  # Populate enough points for a roll-comp fit (MIN_POINTS = 2000).
  rng = np.random.default_rng(1)
  for _ in range(2500):
    roll_rad = rng.uniform(-0.015, 0.015)
    lateral_accel = -np.sin(roll_rad) * 9.81
    steer = lateral_accel / 2.0
    est.roll_comp_buckets.add_point(roll_rad, 2.0 * steer, 25.0)

  est.maybe_persist_speed_profile(cache_write=False)
  assert Params().get("RollCompGainParams") is None

  est.maybe_persist_speed_profile(cache_write=True)
  assert Params().get("RollCompGainParams") is not None


def test_roll_comp_telemetry_populated_after_valid_fit():
  est = _make_estimator("shadow")
  _bootstrap_filtered_points(est)

  rng = np.random.default_rng(2)
  for _ in range(2500):
    roll_rad = rng.uniform(-0.015, 0.015)
    lateral_accel = -np.sin(roll_rad) * 9.81
    steer = lateral_accel / 2.0
    est.roll_comp_buckets.add_point(roll_rad, 2.0 * steer, 25.0)

  est.maybe_persist_speed_profile(cache_write=True)
  msg = est.get_msg()
  assert msg.liveTorqueParameters.rollCompGainValid
  assert msg.liveTorqueParameters.rollCompGainPoints >= 2000
  assert msg.liveTorqueParameters.rollCompGainSpan >= 0.25
  assert 0.3 <= msg.liveTorqueParameters.rollCompGainLearned <= 1.0


def test_torqued_strict_collection_gate_unchanged():
  """The restructured gate must still require v_ego > MIN_VEL and abs(steer) > STEER_MIN_THRESHOLD
  for the strict path that feeds filtered_points / shadow classification."""
  est = _make_estimator("shadow")

  n = _warmup_samples()
  for i in range(n):
    _feed(est, i * DT_MDL, steer=0.3, lateral_accel=0.5, v_ego=MIN_VEL + 1.0)

  bucket_points = len(est.filtered_points)
  shadow_accepted = est.shadow_accepted
  assert bucket_points > 0
  assert shadow_accepted > 0

  est2 = _make_estimator("shadow")
  for i in range(n):
    # Below speed threshold -> strict path must not run.
    _feed(est2, i * DT_MDL, steer=0.3, lateral_accel=0.5, v_ego=MIN_VEL - 1.0)
  assert len(est2.filtered_points) == 0
  assert est2.shadow_accepted == 0

  est3 = _make_estimator("shadow")
  for i in range(n):
    # Below steer threshold -> strict path must not run.
    _feed(est3, i * DT_MDL, steer=STEER_MIN_THRESHOLD / 2.0, lateral_accel=0.5, v_ego=MIN_VEL + 1.0)
  assert len(est3.filtered_points) == 0
  assert est3.shadow_accepted == 0


def _make_estimator_speed_aware(mode="shadow", low_speed_shadow=False):
  params = Params()
  params.put("LiveTorqueSpeedAdaptiveMode", mode, block=True)
  params.put_bool("LiveTorqueLowSpeedShadow", low_speed_shadow, block=True)
  est = TorqueEstimator(make_torque_cp())
  est.update_use_params()
  return est


def _low_speed_feed(est, t, steer, lateral_accel, v_ego):
  # Low-speed shadow collection happens inside collect_shadow_learning_points, which
  # is called under the same lat_active/no-override gate as the strict path.
  _feed(est, t, steer=steer, lateral_accel=lateral_accel, v_ego=v_ego)


def test_low_speed_shadow_off_collects_nothing():
  est = _make_estimator_speed_aware("shadow", low_speed_shadow=False)
  n = _warmup_samples()
  for i in range(n):
    _low_speed_feed(est, i * DT_MDL, steer=0.3, lateral_accel=0.5, v_ego=MIN_VEL - 1.0)
  assert all(len(b.get_points()) == 0 for _, b in est.low_speed_buckets.bucket_items())


def test_low_speed_shadow_requires_speed_aware_mode():
  est = _make_estimator_speed_aware("off", low_speed_shadow=True)
  n = _warmup_samples()
  for i in range(n):
    _low_speed_feed(est, i * DT_MDL, steer=0.3, lateral_accel=0.5, v_ego=MIN_VEL - 1.0)
  assert all(len(b.get_points()) == 0 for _, b in est.low_speed_buckets.bucket_items())


def test_low_speed_shadow_collects_5_to_15_mps():
  est = _make_estimator_speed_aware("shadow", low_speed_shadow=True)
  n = _warmup_samples()
  for i in range(n):
    v_ego = 7.0 + (i % 8)  # 7..14 m/s
    _low_speed_feed(est, i * DT_MDL, steer=0.3, lateral_accel=0.5, v_ego=v_ego)
  assert any(len(b.get_points()) > 0 for _, b in est.low_speed_buckets.bucket_items())


def test_low_speed_shadow_rejects_at_or_above_min_vel():
  est = _make_estimator_speed_aware("shadow", low_speed_shadow=True)
  n = _warmup_samples()
  for i in range(n):
    _low_speed_feed(est, i * DT_MDL, steer=0.3, lateral_accel=0.5, v_ego=MIN_VEL + float(i))
  assert all(len(b.get_points()) == 0 for _, b in est.low_speed_buckets.bucket_items())


def test_low_speed_shadow_rejects_low_steer():
  est = _make_estimator_speed_aware("shadow", low_speed_shadow=True)
  n = _warmup_samples()
  for i in range(n):
    _low_speed_feed(est, i * DT_MDL, steer=0.01, lateral_accel=0.5, v_ego=MIN_VEL - 1.0)
  assert all(len(b.get_points()) == 0 for _, b in est.low_speed_buckets.bucket_items())


def test_low_speed_shadow_rejects_high_lateral_accel():
  est = _make_estimator_speed_aware("shadow", low_speed_shadow=True)
  n = _warmup_samples()
  for i in range(n):
    _low_speed_feed(est, i * DT_MDL, steer=0.3, lateral_accel=2.6, v_ego=MIN_VEL - 1.0)
  assert all(len(b.get_points()) == 0 for _, b in est.low_speed_buckets.bucket_items())


def test_low_speed_shadow_keeps_high_curvature_frames():
  est = _make_estimator_speed_aware("shadow", low_speed_shadow=True)
  n = _warmup_samples()
  for i in range(n):
    # Moderate lateral accel (>1.0, <=2.5) should still be collected.
    _low_speed_feed(est, i * DT_MDL, steer=0.3, lateral_accel=1.5, v_ego=MIN_VEL - 1.0)
  assert any(len(b.get_points()) > 0 for _, b in est.low_speed_buckets.bucket_items())


def test_low_speed_persisted_in_speed_profile():
  est = _make_estimator_speed_aware("shadow", low_speed_shadow=True)
  # Need enough normal-speed points (>= MIN_GLOBAL_POINTS) for the speed-aware
  # profile to fit, plus alternating low-speed frames for the lowSpeed section.
  # Vary steer so SVD can fit a non-degenerate slope.
  n = int(65.0 / DT_MDL)
  for i in range(n):
    v_ego = (MIN_VEL - 1.0) if i % 2 == 0 else (MIN_VEL + 5.0)
    steer = -0.4 + 0.8 * ((i // 2) % 100) / 99.0
    lateral_accel = 2.0 * steer
    _low_speed_feed(est, i * DT_MDL, steer=steer, lateral_accel=lateral_accel, v_ego=v_ego)
  est.maybe_persist_speed_profile(cache_write=True)
  payload = Params().get("LiveTorqueSpeedAdaptiveParams")
  assert payload is not None
  data = json.loads(payload)
  assert "lowSpeed" in data
