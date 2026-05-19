#!/usr/bin/env python3
import numpy as np
import pytest

from cereal import car
from openpilot.selfdrive.locationd.helpers import PointBuckets
from openpilot.selfdrive.locationd.torqued import cache_speed_aware_params, update_speed_aware_param_cache
from openpilot.sunnypilot.selfdrive.locationd import torqued_ext
from openpilot.sunnypilot.selfdrive.locationd.torqued_ext import SPEED_AWARE_PARAMS_VERSION, SpeedAwareTorqueBuckets, format_speed_aware_params


def test_speed_aware_buckets_routing():
  buckets = SpeedAwareTorqueBuckets(
    x_bounds=[(-0.5, -0.3), (-0.3, 0.3), (0.3, 0.5)],
    speed_bp=[0, 10, 20],
    min_points=np.array([10, 50, 10]),
    min_points_total=20,
    points_per_bucket=100,
    rowsize=3
  )
  buckets.add_point(-0.4, 1.0, 5.0)
  buckets.add_point(0.4, 2.0, 15.0)
  buckets.add_point(0.0, 3.0, 25.0)

  assert len(buckets.buckets_for_speed(5.0).buckets) == 3
  assert len(buckets.buckets_for_speed(5.0).get_points()) == 1
  assert len(buckets.buckets_for_speed(15.0).get_points()) == 1
  assert len(buckets.buckets_for_speed(25.0).get_points()) == 1


def test_speed_aware_buckets_valid_percent():
  buckets = SpeedAwareTorqueBuckets(
    x_bounds=[(-0.5, -0.3), (-0.3, 0.3), (0.3, 0.5)],
    speed_bp=[0, 10, 20],
    min_points=np.array([1, 1, 1]),
    min_points_total=2,
    points_per_bucket=100,
    rowsize=3
  )
  for _ in range(10):
    for steer in (-0.4, 0.0, 0.4):
      buckets.add_point(steer, 1.0, 5.0)
      buckets.add_point(steer, 1.0, 15.0)

  assert buckets.is_calculable()
  assert buckets.is_valid()


def test_speed_aware_buckets_sparse_points_are_not_valid():
  buckets = SpeedAwareTorqueBuckets(
    x_bounds=[(-0.5, -0.3), (-0.3, 0.3), (0.3, 0.5)],
    speed_bp=[0, 10, 20],
    min_points=np.array([1, 1, 1]),
    min_points_total=2,
    points_per_bucket=100,
    rowsize=3
  )
  for _ in range(10):
    buckets.add_point(0.0, 1.0, 5.0)

  assert not buckets.is_calculable()
  assert not buckets.is_valid()


def test_speed_aware_get_points():
  buckets = SpeedAwareTorqueBuckets(
    x_bounds=[(-0.5, -0.3), (-0.3, 0.3), (0.3, 0.5)],
    speed_bp=[0, 10, 20],
    min_points=np.array([1, 1, 1]),
    min_points_total=2,
    points_per_bucket=100,
    rowsize=3
  )
  for i in range(5):
    buckets.add_point(float(i) * 0.1, float(i), 5.0)

  pts = buckets.get_points(10)
  assert len(pts) == 5
  assert pts.shape[1] == 3


def make_cp(fingerprint="mock-car", lateral_tuning="torque"):
  CP = car.CarParams.new_message()
  CP.carFingerprint = fingerprint
  CP.lateralTuning.init(lateral_tuning)
  return CP


def test_format_speed_aware_params_wraps_metadata():
  CP = make_cp()
  buckets = {"0_10": (1.2, 0.0, 0.1)}

  payload = format_speed_aware_params(CP, buckets)

  assert payload["version"] == 2
  assert payload["version"] == SPEED_AWARE_PARAMS_VERSION
  assert payload["carFingerprint"] == "mock-car"
  assert payload["lateralTuning"] == "torque"
  assert payload["torqueLatAccelFactor"] == pytest.approx(CP.lateralTuning.torque.latAccelFactor)
  assert payload["torqueFriction"] == pytest.approx(CP.lateralTuning.torque.friction)
  assert payload["buckets"] == buckets


class FakeParams:
  def __init__(self):
    self.writes = {}
    self.removed = []

  def put_nonblocking(self, key, value):
    self.writes[key] = value

  def remove(self, key):
    self.removed.append(key)


class FakeBoolParams:
  def __init__(self, values):
    self.values = values

  def get_bool(self, key):
    return self.values.get(key, False)

  def get(self, key, return_default=False):
    values = {
      "TorqueParamsOverrideLatAccelFactor": 200,
      "TorqueParamsOverrideFriction": 10,
    }
    return values[key]


def make_torque_ext(monkeypatch, *, custom_torque_params, torque_override_enabled):
  params = FakeBoolParams({
    "EnforceTorqueControl": True,
    "LiveTorqueParamsToggle": True,
    "LiveTorqueParamsRelaxedToggle": False,
    "CustomTorqueParams": custom_torque_params,
    "TorqueParamsOverrideEnabled": torque_override_enabled,
    "LiveTorqueSpeedAdaptiveToggle": False,
  })
  monkeypatch.setattr(torqued_ext, "Params", lambda: params)

  ext = torqued_ext.TorqueEstimatorExt(make_cp())
  ext.min_points_total = 1
  ext.initialize_custom_params()
  return ext


def test_torque_override_only_blocks_live_params_when_custom_torque_params_enabled(monkeypatch):
  ext = make_torque_ext(monkeypatch, custom_torque_params=False, torque_override_enabled=True)

  ext.update_use_params()

  assert ext.use_params


def test_torque_override_blocks_live_params_when_custom_torque_params_enabled(monkeypatch):
  ext = make_torque_ext(monkeypatch, custom_torque_params=True, torque_override_enabled=True)

  ext.update_use_params()

  assert not ext.use_params


class FakeEstimator:
  def __init__(self, speed_params, lateral_tuning="torque", speed_adaptive_enabled=True):
    self.CP = make_cp(lateral_tuning=lateral_tuning)
    self.speed_params = speed_params
    self.speed_adaptive_enabled = speed_adaptive_enabled
    self.estimate_calls = 0

  def estimate_speed_aware_params(self):
    self.estimate_calls += 1
    return self.speed_params


def test_cache_speed_aware_params_writes_metadata_payload():
  params = FakeParams()

  cache_speed_aware_params(params, FakeEstimator({"0_10": (1.2, 0.0, 0.1)}))

  assert "LiveTorqueSpeedAdaptiveParams" in params.writes
  assert "version" in params.writes["LiveTorqueSpeedAdaptiveParams"]
  assert not params.removed


def test_cache_speed_aware_params_clears_empty_estimate():
  params = FakeParams()

  cache_speed_aware_params(params, FakeEstimator({}))

  assert params.writes == {}
  assert params.removed == ["LiveTorqueSpeedAdaptiveParams"]


def test_cache_speed_aware_params_clears_non_torque_estimate():
  params = FakeParams()

  cache_speed_aware_params(params, FakeEstimator({"0_10": (1.2, 0.0, 0.1)}, lateral_tuning="pid"))

  assert params.writes == {}
  assert params.removed == ["LiveTorqueSpeedAdaptiveParams"]


def test_update_speed_aware_param_cache_clears_when_learning_disabled():
  params = FakeParams()
  estimator = FakeEstimator({"0_10": (1.2, 0.0, 0.1)}, speed_adaptive_enabled=False)

  update_speed_aware_param_cache(params, estimator)

  assert estimator.estimate_calls == 0
  assert params.writes == {}
  assert params.removed == ["LiveTorqueSpeedAdaptiveParams"]
