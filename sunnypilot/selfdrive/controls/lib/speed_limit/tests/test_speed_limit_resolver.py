"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import random
import time

import pytest
from pytest_mock import MockerFixture

from cereal import custom
from openpilot.sunnypilot.selfdrive.controls.lib.speed_limit import LIMIT_ADAPT_ACC, LIMIT_COAST_APPROACH_MARGIN_S, LIMIT_MAX_MAP_DATA_AGE

from openpilot.sunnypilot.selfdrive.controls.lib.speed_limit.speed_limit_resolver import SpeedLimitResolver, ALL_SOURCES
from openpilot.sunnypilot.selfdrive.controls.lib.speed_limit.common import OffsetType, Policy

SpeedLimitSource = custom.LongitudinalPlanSP.SpeedLimit.Source


class RuntimeParams:
  def __init__(self, values):
    self.values = values
    self.writes = {}

  def get(self, key, return_default=False):
    return self.values.get(key)

  def get_bool(self, key):
    return bool(self.values.get(key, False))

  def put(self, key, value):
    self.writes[key] = value


def create_mock(properties, mocker: MockerFixture):
  mock = mocker.MagicMock()
  for _property, value in properties.items():
    setattr(mock, _property, value)
  return mock


def setup_sm_mock(mocker: MockerFixture):
  cruise_speed_limit = random.uniform(0, 120)
  live_map_data_limit = random.uniform(0, 120)

  car_state = create_mock({
    'gasPressed': False,
    'brakePressed': False,
    'standstill': False,
  }, mocker)
  car_state_sp = create_mock({
    'speedLimit': cruise_speed_limit,
  }, mocker)
  live_map_data = create_mock({
    'speedLimit': live_map_data_limit,
    'speedLimitValid': True,
    'speedLimitAhead': 0.,
    'speedLimitAheadValid': 0.,
    'speedLimitAheadDistance': 0.,
  }, mocker)
  gps_data = create_mock({
    'unixTimestampMillis': time.time() * 1e3,
  }, mocker)
  sm_mock = mocker.MagicMock()
  sm_mock.__getitem__.side_effect = lambda key: {
    'carState': car_state,
    'liveMapDataSP': live_map_data,
    'carStateSP': car_state_sp,
    'gpsLocation': gps_data,
  }[key]
  return sm_mock


def setup_map_sm_mock(mocker: MockerFixture, speed_limit: float, next_speed_limit: float, next_distance: float):
  car_state_sp = create_mock({
    'speedLimit': 0.,
  }, mocker)
  live_map_data = create_mock({
    'speedLimit': speed_limit,
    'speedLimitValid': speed_limit > 0.,
    'speedLimitAhead': next_speed_limit,
    'speedLimitAheadValid': next_speed_limit > 0.,
    'speedLimitAheadDistance': next_distance,
  }, mocker)
  gps_data = create_mock({
    'unixTimestampMillis': time.time() * 1e3,
  }, mocker)
  sm_mock = mocker.MagicMock()
  sm_mock.__getitem__.side_effect = lambda key: {
    'carStateSP': car_state_sp,
    'liveMapDataSP': live_map_data,
    'gpsLocation': gps_data,
  }[key]
  return sm_mock


parametrized_policies = pytest.mark.parametrize(
  "policy, sm_key, function_key", [
    (Policy.car_state_only, 'carStateSP', SpeedLimitSource.car),
    (Policy.car_state_priority, 'carStateSP', SpeedLimitSource.car),
    (Policy.map_data_only, 'liveMapDataSP', SpeedLimitSource.map),
    (Policy.map_data_priority, 'liveMapDataSP', SpeedLimitSource.map),
  ],
  ids=lambda val: val.name if hasattr(val, 'name') else str(val)
)


@pytest.mark.parametrize("resolver_class", [SpeedLimitResolver])
class TestSpeedLimitResolverValidation:

  @pytest.mark.parametrize("policy", list(Policy), ids=lambda policy: policy.name)
  def test_initial_state(self, resolver_class, policy):
    resolver = resolver_class()
    resolver.policy = policy
    for source in ALL_SOURCES:
      if source in resolver.limit_solutions:
        assert resolver.limit_solutions[source] == 0.
        assert resolver.distance_solutions[source] == 0.

  def test_update_params_sanitizes_runtime_policy_and_offset_type(self, resolver_class):
    resolver = resolver_class()
    resolver.frame = 0
    resolver.params = RuntimeParams({
      "SpeedLimitPolicy": Policy.max().value + 10,
      "SpeedLimitOffsetType": OffsetType.max().value + 10,
      "SpeedLimitValueOffset": 3,
      "IsMetric": True,
    })

    resolver.update_params()

    assert resolver.policy == Policy.max().value
    assert resolver.offset_type == OffsetType.max().value
    assert resolver.params.writes["SpeedLimitPolicy"] == Policy.max().value
    assert resolver.params.writes["SpeedLimitOffsetType"] == OffsetType.max().value

  def test_update_params_defaults_non_numeric_runtime_offset_value(self, resolver_class):
    resolver = resolver_class()
    resolver.frame = 0
    resolver.params = RuntimeParams({
      "SpeedLimitPolicy": Policy.combined.value,
      "SpeedLimitOffsetType": OffsetType.fixed.value,
      "SpeedLimitValueOffset": "bad",
      "IsMetric": True,
    })

    resolver.update_params()

    assert resolver.offset_value == pytest.approx(0.0)

  @pytest.mark.parametrize("offset_value", ["nan", "inf", "-inf"])
  def test_update_params_defaults_non_finite_runtime_offset_value(self, resolver_class, offset_value):
    resolver = resolver_class()
    resolver.frame = 0
    resolver.params = RuntimeParams({
      "SpeedLimitPolicy": Policy.combined.value,
      "SpeedLimitOffsetType": OffsetType.fixed.value,
      "SpeedLimitValueOffset": offset_value,
      "IsMetric": True,
    })

    resolver.update_params()

    assert resolver.offset_value == pytest.approx(0.0)
    assert resolver._get_speed_limit_offset() == pytest.approx(0.0)

  @parametrized_policies
  def test_resolver(self, resolver_class, policy, sm_key, function_key, mocker: MockerFixture):
    resolver = resolver_class()
    resolver.policy = policy
    sm_mock = setup_sm_mock(mocker)
    source_speed_limit = sm_mock[sm_key].speedLimit

    # Assert the resolver
    resolver.update(source_speed_limit, sm_mock)
    assert resolver.speed_limit == source_speed_limit
    assert resolver.source == ALL_SOURCES[function_key]

  def test_resolver_combined(self, resolver_class, mocker: MockerFixture):
    resolver = resolver_class()
    resolver.policy = Policy.combined
    sm_mock = setup_sm_mock(mocker)
    socket_to_source = {'carStateSP': SpeedLimitSource.car, 'liveMapDataSP': SpeedLimitSource.map}
    minimum_key, minimum_speed_limit = min(
      ((key, sm_mock[key].speedLimit) for key in
       socket_to_source.keys()), key=lambda x: x[1])

    # Assert the resolver
    resolver.update(minimum_speed_limit, sm_mock)
    assert resolver.speed_limit == minimum_speed_limit
    assert resolver.source == socket_to_source[minimum_key]

  @parametrized_policies
  def test_parser(self, resolver_class, policy, sm_key, function_key, mocker: MockerFixture):
    resolver = resolver_class()
    resolver.policy = policy
    sm_mock = setup_sm_mock(mocker)
    source_speed_limit = sm_mock[sm_key].speedLimit

    # Assert the parsing
    resolver.update(source_speed_limit, sm_mock)
    assert resolver.limit_solutions[ALL_SOURCES[function_key]] == source_speed_limit
    assert resolver.distance_solutions[ALL_SOURCES[function_key]] == 0.

  @pytest.mark.parametrize("policy", list(Policy), ids=lambda policy: policy.name)
  def test_resolve_interaction_in_update(self, resolver_class, policy, mocker: MockerFixture):
    v_ego = 50
    resolver = resolver_class()
    resolver.policy = policy

    sm_mock = setup_sm_mock(mocker)
    resolver.update(v_ego, sm_mock)

    # After resolution
    assert resolver.speed_limit is not None
    assert resolver.distance is not None
    assert resolver.source is not None

  @pytest.mark.parametrize("policy", list(Policy), ids=lambda policy: policy.name)
  def test_old_map_data_ignored(self, resolver_class, policy, mocker: MockerFixture):
    resolver = resolver_class()
    resolver.policy = policy
    sm_mock = mocker.MagicMock()
    sm_mock['gpsLocation'].unixTimestampMillis = (time.time() - 2 * LIMIT_MAX_MAP_DATA_AGE) * 1e3
    resolver._get_from_map_data(sm_mock)
    assert resolver.limit_solutions[SpeedLimitSource.map] == 0.
    assert resolver.distance_solutions[SpeedLimitSource.map] == 0.

  def test_lower_next_map_limit_selected_inside_coast_distance(self, resolver_class, mocker: MockerFixture):
    v_ego = 30.0
    speed_limit = 30.0
    next_speed_limit = 20.0
    coast_accel = -0.3
    coast_distance = (next_speed_limit ** 2 - v_ego ** 2) / (2. * coast_accel)
    sm_mock = setup_map_sm_mock(mocker, speed_limit, next_speed_limit, coast_distance + v_ego * LIMIT_COAST_APPROACH_MARGIN_S - 1.0)

    resolver = resolver_class()
    resolver.policy = Policy.map_data_only
    resolver.update(v_ego, sm_mock, coast_accel=coast_accel)

    assert resolver.speed_limit == next_speed_limit
    assert resolver.distance == pytest.approx(coast_distance + v_ego * LIMIT_COAST_APPROACH_MARGIN_S - 1.0, abs=0.1)
    assert resolver.source == SpeedLimitSource.map

  def test_lower_next_map_limit_waits_until_coast_distance(self, resolver_class, mocker: MockerFixture):
    v_ego = 30.0
    speed_limit = 30.0
    next_speed_limit = 20.0
    coast_accel = -0.3
    coast_distance = (next_speed_limit ** 2 - v_ego ** 2) / (2. * coast_accel)
    sm_mock = setup_map_sm_mock(mocker, speed_limit, next_speed_limit, coast_distance + v_ego * LIMIT_COAST_APPROACH_MARGIN_S + 1.0)

    resolver = resolver_class()
    resolver.policy = Policy.map_data_only
    resolver.update(v_ego, sm_mock, coast_accel=coast_accel)

    assert resolver.speed_limit == speed_limit
    assert resolver.distance == 0.
    assert resolver.source == SpeedLimitSource.map

  def test_old_epoch_map_data_ignored(self, resolver_class, mocker: MockerFixture, monkeypatch):
    now = 1_700_000_000.0
    monkeypatch.setattr(time, "time", lambda: now)
    resolver = resolver_class()
    resolver.v_ego = 0.0
    sm_mock = setup_map_sm_mock(mocker, 30.0, 0.0, 0.0)
    sm_mock['gpsLocation'].unixTimestampMillis = (now - 2 * LIMIT_MAX_MAP_DATA_AGE) * 1e3

    resolver._get_from_map_data(sm_mock)

    assert resolver.limit_solutions[SpeedLimitSource.map] == 0.
    assert resolver.distance_solutions[SpeedLimitSource.map] == 0.

  def test_lower_next_map_limit_selected_inside_adapt_distance(self, resolver_class, mocker: MockerFixture):
    v_ego = 30.0
    speed_limit = 30.0
    next_speed_limit = 20.0
    adapt_distance = (next_speed_limit ** 2 - v_ego ** 2) / (2. * LIMIT_ADAPT_ACC)
    sm_mock = setup_map_sm_mock(mocker, speed_limit, next_speed_limit, adapt_distance - 10.)

    resolver = resolver_class()
    resolver.v_ego = v_ego
    resolver._get_from_map_data(sm_mock)

    assert resolver.limit_solutions[SpeedLimitSource.map] == next_speed_limit
    assert resolver.distance_solutions[SpeedLimitSource.map] == pytest.approx(adapt_distance - 10., abs=0.1)

  def test_lower_next_map_limit_uses_epoch_timestamp_distance(self, resolver_class, mocker: MockerFixture, monkeypatch):
    now = 1_700_000_000.0
    monkeypatch.setattr(time, "time", lambda: now)
    v_ego = 30.0
    speed_limit = 30.0
    next_speed_limit = 20.0
    adapt_distance = (next_speed_limit ** 2 - v_ego ** 2) / (2. * LIMIT_ADAPT_ACC)
    sm_mock = setup_map_sm_mock(mocker, speed_limit, next_speed_limit, adapt_distance - 10.)
    sm_mock['gpsLocation'].unixTimestampMillis = now * 1e3

    resolver = resolver_class()
    resolver.v_ego = v_ego
    resolver._get_from_map_data(sm_mock)

    assert resolver.limit_solutions[SpeedLimitSource.map] == next_speed_limit
    assert resolver.distance_solutions[SpeedLimitSource.map] == pytest.approx(adapt_distance - 10., abs=0.1)

  def test_lower_next_map_limit_waits_until_adapt_distance(self, resolver_class, mocker: MockerFixture):
    v_ego = 30.0
    speed_limit = 30.0
    next_speed_limit = 20.0
    adapt_distance = (next_speed_limit ** 2 - v_ego ** 2) / (2. * LIMIT_ADAPT_ACC)
    sm_mock = setup_map_sm_mock(mocker, speed_limit, next_speed_limit, adapt_distance + 10.)

    resolver = resolver_class()
    resolver.v_ego = v_ego
    resolver._get_from_map_data(sm_mock)

    assert resolver.limit_solutions[SpeedLimitSource.map] == speed_limit
    assert resolver.distance_solutions[SpeedLimitSource.map] == 0.

  def test_faster_next_map_limit_does_not_relax_current_limit(self, resolver_class, mocker: MockerFixture):
    speed_limit = 20.0
    next_speed_limit = 25.0
    sm_mock = setup_map_sm_mock(mocker, speed_limit, next_speed_limit, 0.)

    resolver = resolver_class()
    resolver.v_ego = 30.0
    resolver._get_from_map_data(sm_mock)

    assert resolver.limit_solutions[SpeedLimitSource.map] == speed_limit
    assert resolver.distance_solutions[SpeedLimitSource.map] == 0.
