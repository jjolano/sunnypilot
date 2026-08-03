import json
from types import SimpleNamespace

import numpy as np
import pytest

from openpilot.cereal import car, messaging
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.locationd.helpers import NPQueue
from openpilot.selfdrive.locationd.torqued import (
  MIN_BUCKET_POINTS,
  MIN_VEL,
  STEER_BUCKET_BOUNDS,
  STEER_MIN_THRESHOLD,
  TorqueEstimator,
)
from openpilot.sunnypilot.custom.lateral.direction_gain_learning import (
  DIRECTION_GAIN_SPEED_BANDS,
)
from openpilot.sunnypilot.custom.lateral.roll_comp_learning import (
  MIN_POINTS,
  ROLL_COMP_PRIMARY_BAND,
  RollCompBuckets,
)
from openpilot.sunnypilot.selfdrive.locationd.torqued_ext import ROLL_COMP_LEARN_MIN_V_EGO


PARAM_KEYS = (
  'RollCompGainMode', 'RollCompGainParams', 'LatDirectionGainMode', 'LatDirectionGainParams',
  'LiveTorqueSpeedAdaptiveMode', 'LiveTorqueLowSpeedShadow', 'LiveTorqueSpeedAdaptiveParams',
)


@pytest.fixture(autouse=True)
def cleanup_lateral_params():
  params = Params()
  for key in PARAM_KEYS:
    params.remove(key)
  yield
  for key in PARAM_KEYS:
    params.remove(key)


def make_torque_cp():
  cp = car.CarParams.new_message()
  cp.brand = 'toyota'
  cp.carFingerprint = 'TOYOTA_CAMRY'
  cp.lateralTuning.init('torque')
  cp.lateralTuning.torque.latAccelFactor = 2.0
  cp.lateralTuning.torque.latAccelOffset = 0.0
  cp.lateralTuning.torque.friction = 0.2
  return cp


def _live_pose(timestamp, *, valid=True):
  pose = messaging.new_message('livePose').livePose
  pose.orientationNED = {'x': 0.0, 'valid': valid}
  pose.angularVelocityDevice = {'z': 0.0, 'valid': valid}
  pose.inputsOK = pose.sensorsOK = pose.posenetOK = valid
  pose.timestamp = int(timestamp * 1e9)
  return pose


def _make_estimator(roll_mode='off', direction_mode='off'):
  params = Params()
  params.put('RollCompGainMode', roll_mode, block=True)
  params.put('LatDirectionGainMode', direction_mode, block=True)
  estimator = TorqueEstimator(make_torque_cp())
  estimator.frame = 0
  estimator.update_use_params()
  return estimator


def _feed(est, t, steer, lateral_accel, *, v_ego=25.0, steering_pressed=False,
          steering_rate_deg=0.0, roll_rad=0.0):
  car_control = messaging.new_message('carControl').carControl
  car_output = messaging.new_message('carOutput').carOutput
  car_state = messaging.new_message('carState').carState
  car_control.latActive = True
  car_state.vEgo = v_ego
  car_state.steeringPressed = steering_pressed
  car_state.steeringRateDeg = steering_rate_deg
  car_output.actuatorsOutput.torque = float(-steer)
  live_pose = messaging.new_message('livePose').livePose
  live_pose.orientationNED = {'x': float(roll_rad), 'valid': True}
  live_pose.angularVelocityDevice = {'z': float(lateral_accel / v_ego), 'valid': True}
  live_pose.inputsOK = live_pose.sensorsOK = live_pose.posenetOK = True
  live_pose.timestamp = int(t * 1e9)
  for which, msg in (('carControl', car_control), ('carOutput', car_output),
                     ('carState', car_state), ('livePose', live_pose)):
    est.handle_log(t, which, msg)


def _warmup_samples():
  return int(6.0 / DT_MDL)


def _bootstrap_filtered_points(est):
  rng = np.random.default_rng(0)
  points_per_bucket = (1.5 * np.array(MIN_BUCKET_POINTS)).astype(int)
  for bound, count in zip(STEER_BUCKET_BOUNDS, points_per_bucket, strict=True):
    for _ in range(count):
      steer = rng.uniform(bound[0], bound[1])
      est.filtered_points.add_point(steer, 2.0 * steer)
  est.get_msg()


def _make_speed_estimator(mode='shadow', low_speed_shadow=False):
  params = Params()
  params.put('LiveTorqueSpeedAdaptiveMode', mode, block=True)
  params.put_bool('LiveTorqueLowSpeedShadow', low_speed_shadow, block=True)
  estimator = TorqueEstimator(make_torque_cp())
  estimator.frame = 0
  estimator.update_use_params()
  return estimator


def _low_speed_feed(est, t, steer, lateral_accel, v_ego):
  _feed(est, t, steer, lateral_accel, v_ego=v_ego)


def _roll_for_x(x):
  return -float(np.arcsin(x / 9.81))


def _add_roll_block(buckets, block_id, count=200, slope=0.55):
  for x in np.linspace(-0.4, 0.4, count):
    buckets.add_point(_roll_for_x(float(x)), slope * x + block_id, 20.0, block_id)


def _add_direction_pair(buckets, band, direction, block_id, pair_index):
  speed = band[0] + min(band[1] - band[0], 2.0)
  magnitude = 0.08 + 0.005 * (pair_index % 5)
  delta = 0.06 + 0.01 * (pair_index % 10)
  x0 = direction * magnitude
  x1 = direction * (magnitude + delta)
  slope = 1.1 if direction < 0 else 1.0
  t = block_id * 1000.0 + pair_index * 2.0
  buckets.add_point(x0, slope * x0, speed, t, block_id)
  buckets.add_point(x1, slope * x1, speed, t + 1.0, block_id)
  buckets.clear_history()


def _fill_direction_buckets(buckets, blocks=12):
  for band in DIRECTION_GAIN_SPEED_BANDS:
    for direction in (-1, 1):
      for block_id in range(blocks):
        for pair_index in range(34):
          _add_direction_pair(buckets, band, direction, block_id, pair_index)
  buckets.set_completed_through(blocks - 1)


def _complete_twelve_blocks(estimator):
  estimator.advance_evidence_clock(0.0)
  estimator.advance_evidence_clock(60.0 + 11 * 65.0)  # guard entry completes block 11


def test_20hz_evidence_decimation_has_240_block_slots_and_no_guard_evidence(monkeypatch):
  estimator = _make_estimator(roll_mode='shadow', direction_mode='shadow')
  monkeypatch.setattr(estimator.filtered_points, 'is_valid', lambda: True)
  roll_calls, direction_calls = [], []
  roll_add = estimator.roll_comp_buckets.add_point
  direction_add = estimator.direction_gain_buckets.add_point

  def count_roll(*args):
    roll_calls.append(args)
    return roll_add(*args)

  def count_direction(*args):
    direction_calls.append(args)
    return direction_add(*args)

  monkeypatch.setattr(estimator.roll_comp_buckets, 'add_point', count_roll)
  monkeypatch.setattr(estimator.direction_gain_buckets, 'add_point', count_direction)

  block_slots = guard_slots = invalid_slots = 0
  for frame in range(1300):  # exactly 65 seconds at the 20 Hz livePose cadence
    timestamp = frame / 20.0
    block_id = estimator.advance_evidence_clock(timestamp)
    if block_id == 0 and estimator.custom_evidence_allowed:
      block_slots += 1
    if block_id is None and estimator.custom_evidence_allowed:
      guard_slots += 1
    invalid = frame == 4  # consume the first fifth-update opportunity
    if invalid:
      invalid_slots += int(block_id == 0 and estimator.custom_evidence_allowed)
      continue
    estimator.collect_shadow_learning_points(
      0.2, 0.2, 20.0, 0.0, 0.0, 0.0, timestamp, block_id,
      estimator.custom_evidence_allowed,
    )
    if frame == 1200:
      assert not estimator.direction_gain_buckets._history

  assert estimator._evidence_opportunity_count == 1300
  assert block_slots == 240
  assert invalid_slots == 1
  assert guard_slots == 20
  assert len(roll_calls) == len(direction_calls) == 239
  assert {call[-1] for call in roll_calls} == {0}
  assert {call[-1] for call in direction_calls} == {0}


def test_timestamp_discontinuity_is_fail_closed_and_clears_stale_evidence():
  estimator = _make_estimator(roll_mode='shadow', direction_mode='shadow')
  estimator.advance_evidence_clock(0.0)
  estimator.direction_gain_buckets.add_point(0.1, 0.1, 20.0, 0.0, 0)
  assert estimator.direction_gain_buckets._history

  for timestamp in (0.05, 0.1, 0.15):
    estimator.advance_evidence_clock(timestamp)
  state = (estimator.evidence_clock.block_id, estimator.evidence_clock.completed_through)
  assert estimator.advance_evidence_clock(0.1) is None
  assert estimator.evidence_clock.discontinuity
  assert estimator.evidence_clock.boundary
  assert (estimator.evidence_clock.block_id, estimator.evidence_clock.completed_through) == state
  assert not estimator.custom_evidence_allowed
  assert not estimator.direction_gain_buckets._history

  before_roll = len(estimator.roll_comp_buckets.get_points())
  estimator.collect_shadow_learning_points(
    0.2, 0.2, 20.0, 0.0, 0.0, 0.0, 0.1,
    estimator.evidence_clock.block_id, estimator.custom_evidence_allowed,
  )
  assert len(estimator.roll_comp_buckets.get_points()) == before_roll
  assert len(estimator.direction_gain_buckets.direction_pairs(DIRECTION_GAIN_SPEED_BANDS[1], 1)) == 0

  for timestamp in (0.2, 0.25, 0.3, 0.35):
    estimator.advance_evidence_clock(timestamp)
  assert estimator.advance_evidence_clock(float('nan')) is None
  assert estimator.evidence_clock.discontinuity
  assert not estimator.custom_evidence_allowed
  assert (estimator.evidence_clock.block_id, estimator.evidence_clock.completed_through) == state


def test_main_persists_at_one_true_60_second_1200_frame_interval(monkeypatch):
  import openpilot.selfdrive.locationd.torqued as torqued_module

  class StopLoop(Exception):
    pass

  class FakeParams:
    puts = []

    def get(self, key, block=False):
      return b'fake-car-params' if key == 'CarParams' else None

    def put(self, key, value, block=False):
      self.puts.append(key)

  class FakeEstimator:
    instances = []

    def __init__(self, _cp):
      self.speed_adaptive_mode = 'off'
      self.roll_comp_mode = 'shadow'
      self.direction_gain_mode = 'off'
      self.timestamps = []
      self.persist_frames = []
      self.__class__.instances.append(self)

    def advance_evidence_clock(self, timestamp):
      self.timestamps.append(timestamp)

    def update_use_params(self):
      pass

    def get_msg(self, **kwargs):
      return SimpleNamespace(to_bytes=lambda: b'live-torque')

    def maybe_persist_speed_profile(self, cache_write=False):
      if cache_write:
        self.persist_frames.append(FakeSubMaster.last_frame)

  class FakeSubMaster:
    last_frame = None

    def __init__(self, *_args, **_kwargs):
      self.frame = -1
      self.updated = {}
      self._live_pose = None

    def update(self):
      self.frame += 1
      FakeSubMaster.last_frame = self.frame
      if self.frame >= 1300:
        raise StopLoop
      self.updated = {'livePose': True}
      self._live_pose = SimpleNamespace(timestamp=int(self.frame / 20.0 * 1e9))

    def all_checks(self):
      return False

    def __getitem__(self, key):
      assert key == 'livePose'
      return self._live_pose

  class FakePubMaster:
    def __init__(self, *_args, **_kwargs):
      pass

    def send(self, *_args, **_kwargs):
      pass

  monkeypatch.setattr(torqued_module, 'Params', FakeParams)
  monkeypatch.setattr(torqued_module, 'TorqueEstimator', FakeEstimator)
  monkeypatch.setattr(torqued_module.messaging, 'SubMaster', FakeSubMaster)
  monkeypatch.setattr(torqued_module.messaging, 'PubMaster', FakePubMaster)
  monkeypatch.setattr(torqued_module.messaging, 'log_from_bytes', lambda *_args, **_kwargs: object())
  monkeypatch.setattr(torqued_module, 'config_realtime_process', lambda *_args, **_kwargs: None)
  monkeypatch.setattr(torqued_module.TorqueEstimatorExt, 'update_use_params', lambda _estimator: None)

  with pytest.raises(StopLoop):
    torqued_module.main()

  estimator = FakeEstimator.instances[0]
  assert len(estimator.timestamps) == 1300
  assert estimator.timestamps[-1] == pytest.approx(64.95)
  assert estimator.persist_frames == [1200]


def test_evidence_clock_advances_for_invalid_live_pose_cycles():
  estimator = _make_estimator(direction_mode='shadow')

  estimator.handle_log(0.0, 'livePose', _live_pose(0.0))
  estimator.handle_log(60.0, 'livePose', _live_pose(60.0, valid=False))
  assert estimator.evidence_clock.in_guard
  assert estimator.evidence_clock.completed_through == 0

  estimator.handle_log(65.0, 'livePose', _live_pose(65.0, valid=False))
  assert estimator.evidence_clock.block_id == 1
  assert not estimator.evidence_clock.in_guard


def test_direction_history_is_cleared_at_guard_and_block_boundaries():
  estimator = _make_estimator(direction_mode='shadow')
  buckets = estimator.direction_gain_buckets
  band = DIRECTION_GAIN_SPEED_BANDS[1]

  estimator.advance_evidence_clock(0.0)
  buckets.add_point(0.1, 0.1, 20.0, 0.0, 0)
  assert buckets._history
  estimator.advance_evidence_clock(60.0)  # guard boundary
  assert not buckets._history

  buckets.add_point(0.1, 0.1, 20.0, 61.0, 0)
  estimator.advance_evidence_clock(65.0)  # next evidence block boundary
  assert not buckets._history
  buckets.add_point(0.2, 0.2, 20.0, 66.0, 1)
  assert len(buckets.direction_pairs(band, 1)) == 0


def test_roll_persistence_replaces_snapshot_without_double_counting():
  estimator = _make_estimator(roll_mode='shadow')
  _complete_twelve_blocks(estimator)
  for block_id in range(12):
    _add_roll_block(estimator.roll_comp_buckets, block_id)
  estimator.roll_comp_buckets.set_completed_through(11)

  estimator.maybe_persist_speed_profile(cache_write=True)
  first_payload = Params().get('RollCompGainParams')
  first = json.loads(first_payload)
  estimator.maybe_persist_speed_profile(cache_write=True)
  assert Params().get('RollCompGainParams') == first_payload
  assert json.loads(Params().get('RollCompGainParams')) == first
  assert first['points'] == 2400
  assert first['blockCount'] == 12
  assert first['slopeRelSe'] == 0.0

  # New samples in an incomplete block do not alter the persisted snapshot.
  _add_roll_block(estimator.roll_comp_buckets, 12, count=200, slope=0.9)
  estimator.maybe_persist_speed_profile(cache_write=True)
  assert Params().get('RollCompGainParams') == first_payload


def test_direction_persistence_replaces_snapshot_without_double_counting():
  estimator = _make_estimator(direction_mode='shadow')
  _complete_twelve_blocks(estimator)
  _fill_direction_buckets(estimator.direction_gain_buckets)

  estimator.maybe_persist_speed_profile(cache_write=True)
  first_payload = Params().get('LatDirectionGainParams')
  first = json.loads(first_payload)
  estimator.maybe_persist_speed_profile(cache_write=True)

  assert Params().get('LatDirectionGainParams') == first_payload
  assert json.loads(Params().get('LatDirectionGainParams')) == first
  assert first['points'] == 1632
  assert first['blockCount'] == 12
  assert first['maxRelSe'] == 0.0


def test_no_roll_or_direction_profile_is_persisted_without_a_valid_snapshot():
  estimator = _make_estimator(roll_mode='shadow', direction_mode='shadow')
  estimator.maybe_persist_speed_profile(cache_write=True)
  assert Params().get('RollCompGainParams') is None
  assert Params().get('LatDirectionGainParams') is None


def test_roll_bucket_uses_the_configured_primary_band():
  buckets = RollCompBuckets()
  _add_roll_block(buckets, 0, count=1)
  assert len(buckets.band_points(ROLL_COMP_PRIMARY_BAND)) == 1


def test_npqueue_wraparound_keeps_insertion_order():
  queue = NPQueue(maxlen=3, rowsize=2)
  for i in range(5):
    queue.append([float(i), float(-i)])
  np.testing.assert_array_equal(queue.arr, np.array([[2.0, -2.0], [3.0, -3.0], [4.0, -4.0]]))


def test_roll_comp_off_collects_nothing():
  estimator = _make_estimator('off')
  _bootstrap_filtered_points(estimator)
  for i in range(_warmup_samples()):
    roll = np.deg2rad(1.0)
    lateral_accel = -np.sin(roll) * 9.81
    _feed(estimator, i * DT_MDL, lateral_accel / 2.0, lateral_accel, roll_rad=roll)
  assert len(estimator.roll_comp_buckets.get_points()) == 0
  msg = estimator.get_msg()
  assert not msg.liveTorqueParameters.rollCompGainValid
  assert msg.liveTorqueParameters.rollCompGainPoints == 0


def test_roll_comp_shadow_collects_on_straight_frames():
  estimator = _make_estimator('shadow')
  _bootstrap_filtered_points(estimator)
  for i in range(_warmup_samples()):
    roll = np.deg2rad(0.8 if i % 2 == 0 else -0.8)
    lateral_accel = -np.sin(roll) * 9.81
    _feed(estimator, i * DT_MDL, lateral_accel / 2.0, lateral_accel, roll_rad=roll)
  assert len(estimator.roll_comp_buckets.get_points()) > 0


def test_roll_comp_rejects_curved_frames_and_overrides():
  estimator = _make_estimator('shadow')
  _bootstrap_filtered_points(estimator)
  for i in range(_warmup_samples()):
    roll = np.deg2rad(0.5)
    _feed(estimator, i * DT_MDL, 0.0, roll * 9.81 + 1.0, roll_rad=roll)
  assert len(estimator.roll_comp_buckets.get_points()) == 0

  estimator = _make_estimator('shadow')
  _bootstrap_filtered_points(estimator)
  for i in range(_warmup_samples()):
    roll = np.deg2rad(0.8)
    lateral_accel = -np.sin(roll) * 9.81
    _feed(estimator, i * DT_MDL, lateral_accel / 2.0, lateral_accel, roll_rad=roll, steering_pressed=True)
  assert len(estimator.roll_comp_buckets.get_points()) == 0


@pytest.mark.parametrize('v_ego', [ROLL_COMP_LEARN_MIN_V_EGO - 1.0])
def test_roll_comp_low_speed_collection_floor(v_ego):
  estimator = _make_estimator('shadow')
  _bootstrap_filtered_points(estimator)
  for i in range(_warmup_samples()):
    roll = np.deg2rad(0.8)
    lateral_accel = -np.sin(roll) * 9.81
    _feed(estimator, i * DT_MDL, lateral_accel / 2.0, lateral_accel, v_ego=v_ego, roll_rad=roll)
  assert len(estimator.roll_comp_buckets.get_points()) == 0


def test_roll_comp_low_speed_frames_route_to_configured_band():
  estimator = _make_estimator('shadow')
  _bootstrap_filtered_points(estimator)
  for i in range(_warmup_samples()):
    roll = np.deg2rad(0.8)
    lateral_accel = -np.sin(roll) * 9.81
    _feed(estimator, i * DT_MDL, lateral_accel / 2.0, lateral_accel, v_ego=MIN_VEL - 1.0, roll_rad=roll)
  assert len(estimator.roll_comp_buckets.band_points((10.0, 15.0))) > 0
  assert len(estimator.roll_comp_buckets.band_points((5.0, 10.0))) == 0


def test_roll_comp_rejects_high_rate_large_roll_and_unconverged_base():
  for steering_rate, roll_rad in ((10.0, np.deg2rad(0.8)), (0.0, np.deg2rad(8.0))):
    estimator = _make_estimator('shadow')
    _bootstrap_filtered_points(estimator)
    for i in range(_warmup_samples()):
      lateral_accel = -np.sin(roll_rad) * 9.81
      _feed(estimator, i * DT_MDL, lateral_accel / 2.0, lateral_accel,
            steering_rate_deg=steering_rate, roll_rad=roll_rad)
    assert len(estimator.roll_comp_buckets.get_points()) == 0

  estimator = _make_estimator('shadow')
  for i in range(_warmup_samples()):
    roll = np.deg2rad(0.8)
    lateral_accel = -np.sin(roll) * 9.81
    _feed(estimator, i * DT_MDL, lateral_accel / 2.0, lateral_accel, roll_rad=roll)
  assert not estimator.filtered_points.is_valid()
  assert len(estimator.roll_comp_buckets.get_points()) == 0


def test_roll_comp_persist_only_when_cache_write_enabled():
  estimator = _make_estimator('shadow')
  _complete_twelve_blocks(estimator)
  for block_id in range(12):
    _add_roll_block(estimator.roll_comp_buckets, block_id)
  estimator.maybe_persist_speed_profile(cache_write=False)
  assert Params().get('RollCompGainParams') is None
  estimator.maybe_persist_speed_profile(cache_write=True)
  assert Params().get('RollCompGainParams') is not None


def test_roll_comp_telemetry_populates_after_valid_fit():
  estimator = _make_estimator('shadow')
  _complete_twelve_blocks(estimator)
  for block_id in range(12):
    _add_roll_block(estimator.roll_comp_buckets, block_id)
  estimator.maybe_persist_speed_profile(cache_write=True)
  msg = estimator.get_msg()
  assert msg.liveTorqueParameters.rollCompGainValid
  assert msg.liveTorqueParameters.rollCompGainPoints == 2400
  assert msg.liveTorqueParameters.rollCompGainSpan >= 0.25
  assert 0.3 <= msg.liveTorqueParameters.rollCompGainLearned <= 1.0
  assert list(msg.liveTorqueParameters.rollCompBandGains)[:2] == [0.0, 0.0]


def test_torqued_strict_collection_gates_unchanged():
  estimator = _make_estimator('shadow')
  for i in range(_warmup_samples()):
    _feed(estimator, i * DT_MDL, 0.3, 0.5, v_ego=MIN_VEL + 1.0)
  assert len(estimator.filtered_points) > 0

  estimator = _make_estimator('shadow')
  for i in range(_warmup_samples()):
    _feed(estimator, i * DT_MDL, 0.3, 0.5, v_ego=MIN_VEL - 1.0)
  assert len(estimator.filtered_points) == 0

  estimator = _make_estimator('shadow')
  for i in range(_warmup_samples()):
    _feed(estimator, i * DT_MDL, STEER_MIN_THRESHOLD / 2.0, 0.5, v_ego=MIN_VEL + 1.0)
  assert len(estimator.filtered_points) == 0


def test_low_speed_shadow_collection_modes_and_bands():
  estimator = _make_speed_estimator('shadow', low_speed_shadow=False)
  for i in range(_warmup_samples()):
    _low_speed_feed(estimator, i * DT_MDL, 0.3, 0.5, MIN_VEL - 1.0)
  assert all(len(bucket.get_points()) == 0 for _, bucket in estimator.low_speed_buckets.bucket_items())

  estimator = _make_speed_estimator('off', low_speed_shadow=True)
  for i in range(_warmup_samples()):
    _low_speed_feed(estimator, i * DT_MDL, 0.3, 0.5, MIN_VEL - 1.0)
  assert all(len(bucket.get_points()) == 0 for _, bucket in estimator.low_speed_buckets.bucket_items())

  estimator = _make_speed_estimator('shadow', low_speed_shadow=True)
  for i in range(_warmup_samples()):
    _low_speed_feed(estimator, i * DT_MDL, 0.3, 0.5, 7.0 + i % 8)
  assert any(len(bucket.get_points()) > 0 for _, bucket in estimator.low_speed_buckets.bucket_items())

  estimator = _make_speed_estimator('shadow', low_speed_shadow=True)
  for i in range(_warmup_samples()):
    _low_speed_feed(estimator, i * DT_MDL, 0.3, 0.5, MIN_VEL + float(i))
  assert all(len(bucket.get_points()) == 0 for _, bucket in estimator.low_speed_buckets.bucket_items())


def test_low_speed_shadow_rejects_bad_inputs_and_routes_outer_buckets():
  for steer, lateral_accel in ((0.01, 0.5), (0.3, 3.1)):
    estimator = _make_speed_estimator('shadow', low_speed_shadow=True)
    for i in range(_warmup_samples()):
      _low_speed_feed(estimator, i * DT_MDL, steer, lateral_accel, MIN_VEL - 1.0)
    assert all(len(bucket.get_points()) == 0 for _, bucket in estimator.low_speed_buckets.bucket_items())

  estimator = _make_speed_estimator('shadow', low_speed_shadow=True)
  for i in range(_warmup_samples()):
    _low_speed_feed(estimator, i * DT_MDL, 0.0, 0.0, MIN_VEL - 1.0)
  _low_speed_feed(estimator, _warmup_samples() * DT_MDL, 0.75, 2.9, MIN_VEL - 1.0)
  _low_speed_feed(estimator, (_warmup_samples() + 1) * DT_MDL, -0.75, 2.9, MIN_VEL - 1.0)
  bucket = estimator.low_speed_buckets.buckets[1]
  assert len(bucket.buckets[(0.5, 1.0)].arr) == 1
  assert len(bucket.buckets[(-1.0, -0.5)].arr) == 1


def test_low_speed_shadow_keeps_moderate_high_curvature_frames():
  estimator = _make_speed_estimator('shadow', low_speed_shadow=True)
  for i in range(_warmup_samples()):
    _low_speed_feed(estimator, i * DT_MDL, 0.3, 1.5, MIN_VEL - 1.0)
  assert any(len(bucket.get_points()) > 0 for _, bucket in estimator.low_speed_buckets.bucket_items())


def test_low_speed_profile_persists_with_existing_speed_profile_fixture():
  params = Params()
  cp = make_torque_cp()
  params.put('LiveTorqueSpeedAdaptiveParams', json.dumps({
    'version': 1,
    'restoreKey': {
      'carFingerprint': cp.carFingerprint,
      'lateralTuning': cp.lateralTuning.which(),
      'latAccelFactor': float(cp.lateralTuning.torque.latAccelFactor),
      'friction': float(cp.lateralTuning.torque.friction),
    },
    'anchors': [15.0, 20.0, 25.0, 30.0, 40.0],
    'ratios': [0.8] * 5,
    'confidence': [1.0] * 5,
    'points': [500] * 5,
    'globalLatAccelFactor': 2.0,
    'globalFriction': 0.2,
  }), block=True)
  estimator = _make_speed_estimator('shadow', low_speed_shadow=True)
  for i in range(int(65.0 / DT_MDL)):
    v_ego = MIN_VEL - 1.0 if i % 2 == 0 else MIN_VEL + 5.0
    steer = -0.4 + 0.8 * ((i // 2) % 100) / 99.0
    _low_speed_feed(estimator, i * DT_MDL, steer, 2.0 * steer, v_ego)
  estimator.maybe_persist_speed_profile(cache_write=True)
  payload = Params().get('LiveTorqueSpeedAdaptiveParams')
  assert payload is not None and 'lowSpeed' in json.loads(payload)
  estimator.maybe_persist_speed_profile(cache_write=True)
  assert Params().get('LiveTorqueSpeedAdaptiveParams') == payload
