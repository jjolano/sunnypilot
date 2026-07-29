"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import random
import math
import time

import pytest
from pytest_mock import MockerFixture

pytestmark = pytest.mark.xdist_group("speed_limit_params")

from openpilot.cereal import custom
from openpilot.sunnypilot import get_sanitize_int_param
from openpilot.sunnypilot.selfdrive.controls.lib.speed_limit import LIMIT_MAX_MAP_DATA_AGE

from openpilot.sunnypilot.selfdrive.controls.lib.speed_limit.speed_limit_resolver import SpeedLimitResolver, ALL_SOURCES
from openpilot.sunnypilot.selfdrive.controls.lib.speed_limit.common import Policy

SpeedLimitSource = custom.LongitudinalPlanSP.SpeedLimit.Source


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


class FakeParams:
  def __init__(self, value):
    self.value = value
    self.puts = []

  def get(self, key, return_default=False):
    return self.value

  def put(self, key, value, block=False):
    self.puts.append((key, value, block))


parametrized_policies = pytest.mark.parametrize(
  "policy, sm_key, function_key", [
    (Policy.car_state_only, 'carStateSP', SpeedLimitSource.car),
    (Policy.car_state_priority, 'carStateSP', SpeedLimitSource.car),
    (Policy.map_data_only, 'liveMapDataSP', SpeedLimitSource.map),
    (Policy.map_data_priority, 'liveMapDataSP', SpeedLimitSource.map),
  ],
  ids=lambda val: val.name if hasattr(val, 'name') else str(val)
)


@pytest.mark.parametrize("bad_value", ["nan", float("nan"), float("inf"), None])
def test_sanitize_int_param_defaults_malformed_values_to_minimum(bad_value):
  params = FakeParams(bad_value)

  value = get_sanitize_int_param("ExampleParam", 0, 4, params)

  assert value == 0
  assert params.puts == [("ExampleParam", 0, True)]


def test_sanitize_int_param_clips_out_of_range_values():
  params = FakeParams(10)

  value = get_sanitize_int_param("ExampleParam", 0, 4, params)

  assert value == 4
  assert params.puts == [("ExampleParam", 4, True)]


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

  @pytest.mark.parametrize("bad_limit", [float("nan"), float("inf"), -10., 0., None])
  def test_invalid_speed_limit_sources_fail_closed(self, resolver_class, bad_limit, mocker: MockerFixture):
    resolver = resolver_class()
    resolver.policy = Policy.combined
    sm_mock = setup_sm_mock(mocker)
    sm_mock['carStateSP'].speedLimit = bad_limit
    sm_mock['liveMapDataSP'].speedLimit = bad_limit

    resolver.update(25., sm_mock)

    assert resolver.speed_limit == 0.
    assert resolver.speed_limit_final == 0.
    assert resolver.source == SpeedLimitSource.none

  @pytest.mark.parametrize("bad_distance", [float("nan"), float("inf"), -20., None])
  def test_invalid_map_speed_limit_ahead_distance_fails_closed_to_current_limit(self, resolver_class, bad_distance, mocker: MockerFixture):
    resolver = resolver_class()
    resolver.policy = Policy.map_data_only
    sm_mock = setup_sm_mock(mocker)
    sm_mock['liveMapDataSP'].speedLimit = 30.
    sm_mock['liveMapDataSP'].speedLimitAhead = 10.
    sm_mock['liveMapDataSP'].speedLimitAheadValid = True
    sm_mock['liveMapDataSP'].speedLimitAheadDistance = bad_distance

    resolver.update(25., sm_mock)

    assert resolver.speed_limit == 30.
    assert resolver.distance == 0.
    assert resolver.source == SpeedLimitSource.map

  @pytest.mark.parametrize("bad_offset", [float("nan"), float("inf"), None])
  def test_invalid_speed_limit_offset_fails_closed_to_no_offset(self, resolver_class, bad_offset, mocker: MockerFixture):
    resolver = resolver_class()
    resolver.policy = Policy.car_state_only
    resolver.offset_type = 1
    resolver.offset_value = bad_offset
    sm_mock = setup_sm_mock(mocker)
    sm_mock['carStateSP'].speedLimit = 30.

    resolver.update(25., sm_mock)

    assert resolver.speed_limit == 30.
    assert resolver.speed_limit_offset == 0.
    assert resolver.speed_limit_final == 30.
    assert math.isfinite(resolver.speed_limit_final_last)

  def test_excessive_negative_offset_keeps_current_limit(self, resolver_class, mocker: MockerFixture):
    resolver = resolver_class()
    resolver.policy = Policy.car_state_only
    resolver.offset_type = 1
    resolver.offset_value = -999.
    sm_mock = setup_sm_mock(mocker)
    sm_mock['carStateSP'].speedLimit = 30.

    resolver.update(25., sm_mock)

    assert resolver.speed_limit == 30.
    assert resolver.speed_limit_offset == 0.
    assert resolver.speed_limit_final == 30.
    assert resolver.speed_limit_final_last == 30.
