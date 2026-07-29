"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import math
import time
import pytest
from pytest_mock import MockerFixture
from types import SimpleNamespace

from openpilot.cereal import custom
from openpilot.common.constants import CV
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
from openpilot.sunnypilot.selfdrive.selfdrived.events import EventsSP

# v2 canonical package modules
from openpilot.sunnypilot.custom.longitudinal.speed_limit import constants as v2_constants
from openpilot.sunnypilot.custom.longitudinal.speed_limit import types as v2_types
from openpilot.sunnypilot.custom.longitudinal.speed_limit import helpers as v2_helpers
from openpilot.sunnypilot.custom.longitudinal.speed_limit import resolver as v2_resolver
from openpilot.sunnypilot.custom.longitudinal.speed_limit import assist as v2_assist

# v1 legacy backup modules
from openpilot.sunnypilot.selfdrive.controls.lib.speed_limit import constants_v1 as v1_constants
from openpilot.sunnypilot.selfdrive.controls.lib.speed_limit import common_v1 as v1_common
from openpilot.sunnypilot.selfdrive.controls.lib.speed_limit import helpers_v1 as v1_helpers
from openpilot.sunnypilot.selfdrive.controls.lib.speed_limit import speed_limit_resolver_v1 as v1_resolver
from openpilot.sunnypilot.selfdrive.controls.lib.speed_limit import speed_limit_assist_v1 as v1_assist

# legacy facades sanity
from openpilot.sunnypilot.selfdrive.controls.lib.speed_limit import (
  LIMIT_ADAPT_ACC as facade_LIMIT_ADAPT_ACC,
  LIMIT_MAX_MAP_DATA_AGE as facade_LIMIT_MAX_MAP_DATA_AGE,
  PCM_LONG_REQUIRED_MAX_SET_SPEED as facade_PCM_LONG_REQUIRED_MAX_SET_SPEED,
  CONFIRM_SPEED_THRESHOLD as facade_CONFIRM_SPEED_THRESHOLD,
)
from openpilot.sunnypilot.selfdrive.controls.lib.speed_limit.speed_limit_resolver import (
  ALL_SOURCES as facade_ALL_SOURCES,
)
from openpilot.sunnypilot.selfdrive.controls.lib.speed_limit.speed_limit_assist import (
  SpeedLimitAssist as FacadeSpeedLimitAssist,
  PRE_ACTIVE_GUARD_PERIOD as facade_PRE_ACTIVE_GUARD_PERIOD,
  ACTIVE_STATES as facade_ACTIVE_STATES,
  ENABLED_STATES as facade_ENABLED_STATES,
)

SpeedLimitAssistState = custom.LongitudinalPlanSP.SpeedLimit.AssistState
SpeedLimitSource = custom.LongitudinalPlanSP.SpeedLimit.Source

SPEED_LIMITS = {
  'residential': 25 * CV.MPH_TO_MS,
  'city': 35 * CV.MPH_TO_MS,
  'highway': 65 * CV.MPH_TO_MS,
  'freeway': 80 * CV.MPH_TO_MS,
}


def _enum_identities(module_enum_cls):
  return [(e.name, e.value) for e in module_enum_cls]


def _resolver_snapshot(resolver):
  return {
    'speed_limit': resolver.speed_limit,
    'speed_limit_final': resolver.speed_limit_final,
    'speed_limit_final_last': resolver.speed_limit_final_last,
    'speed_limit_offset': resolver.speed_limit_offset,
    'source': resolver.source,
    'distance': resolver.distance,
    'limit_solutions': dict(resolver.limit_solutions),
    'distance_solutions': dict(resolver.distance_solutions),
    'frame': resolver.frame,
  }


def _assist_snapshot(sla):
  return {
    'state': sla.state,
    'is_enabled': sla.is_enabled,
    'is_active': sla.is_active,
    'output_v_target': sla.output_v_target,
    'output_a_target': sla.output_a_target,
    'v_offset': sla.v_offset,
    'target_set_speed_conv': sla.target_set_speed_conv,
    'v_cruise_cluster_conv': sla.v_cruise_cluster_conv,
    'speed_limit_final_last_conv': sla.speed_limit_final_last_conv,
    'pcm_op_long': sla.pcm_op_long,
    'enabled': sla.enabled,
  }


def _event_names(events_sp: EventsSP):
  return [events_sp.get_event_name(e) for e in events_sp.names]


def _make_cp(openpilot_long: bool = True, pcm_cruise: bool = True, brand: str = 'toyota'):
  return SimpleNamespace(
    brand=brand,
    openpilotLongitudinalControl=openpilot_long,
    pcmCruise=pcm_cruise,
  )


def _make_cp_sp(pcm_cruise_speed: bool = False):
  return SimpleNamespace(pcmCruiseSpeed=pcm_cruise_speed)


def _setup_resolver_sm(mocker: MockerFixture, *, car_limit: float = 0., map_limit: float = 0.,
                       map_ahead_limit: float = 0., map_ahead_distance: float = 0.,
                       map_ahead_valid: bool = False, gps_age_s: float = 0.):
  car_state_sp = mocker.MagicMock()
  car_state_sp.speedLimit = car_limit

  live_map_data = mocker.MagicMock()
  live_map_data.speedLimit = map_limit
  live_map_data.speedLimitValid = math.isfinite(map_limit) and map_limit > 0.
  live_map_data.speedLimitAhead = map_ahead_limit
  live_map_data.speedLimitAheadValid = map_ahead_valid
  live_map_data.speedLimitAheadDistance = map_ahead_distance

  gps_data = mocker.MagicMock()
  gps_data.unixTimestampMillis = (time.monotonic() - gps_age_s) * 1e3

  sm = mocker.MagicMock()
  sm.__getitem__.side_effect = lambda key: {
    'carStateSP': car_state_sp,
    'liveMapDataSP': live_map_data,
    'gpsLocation': gps_data,
  }[key]
  return sm


@pytest.fixture(autouse=True)
def reset_params():
  params = Params()
  params.put_bool("IsMetric", False, block=True)
  params.put_bool("UbloxAvailable", False, block=True)
  params.put("SpeedLimitPolicy", 0, block=True)
  params.put("SpeedLimitOffsetType", 0, block=True)
  params.put("SpeedLimitValueOffset", 0, block=True)
  params.put("SpeedLimitMode", int(v1_common.Mode.assist), block=True)
  params.put_bool("IsReleaseSpBranch", True, block=True)


@pytest.fixture(autouse=True)
def fixed_time(monkeypatch):
  # Pin both clocks to the same value: v1 (upstream) compares unixTimestampMillis against
  # time.monotonic() (the inherited clock-domain bug), v2 against time.time() (fixed).
  # With both pinned identically, the parity assertions stay valid for age-based logic.
  now = 1_000_000.
  monkeypatch.setattr(time, "monotonic", lambda: now)
  monkeypatch.setattr(time, "time", lambda: now)


class TestConstantsParity:
  def test_package_constants(self):
    assert v1_constants.LIMIT_ADAPT_ACC == v2_constants.LIMIT_ADAPT_ACC
    assert v1_constants.LIMIT_MAX_MAP_DATA_AGE == v2_constants.LIMIT_MAX_MAP_DATA_AGE
    assert v1_constants.PCM_LONG_REQUIRED_MAX_SET_SPEED == v2_constants.PCM_LONG_REQUIRED_MAX_SET_SPEED
    assert v1_constants.CONFIRM_SPEED_THRESHOLD == v2_constants.CONFIRM_SPEED_THRESHOLD

  def test_assist_module_constants(self):
    assert v1_assist.DISABLED_GUARD_PERIOD == v2_constants.DISABLED_GUARD_PERIOD
    assert v1_assist.PRE_ACTIVE_GUARD_PERIOD == v2_constants.PRE_ACTIVE_GUARD_PERIOD
    assert v1_assist.SPEED_LIMIT_CHANGED_HOLD_PERIOD == v2_constants.SPEED_LIMIT_CHANGED_HOLD_PERIOD
    assert v1_assist.LIMIT_MIN_ACC == v2_constants.LIMIT_MIN_ACC
    assert v1_assist.LIMIT_MAX_ACC == v2_constants.LIMIT_MAX_ACC
    assert v1_assist.LIMIT_MIN_SPEED == v2_constants.LIMIT_MIN_SPEED
    assert v1_assist.LIMIT_SPEED_OFFSET_TH == v2_constants.LIMIT_SPEED_OFFSET_TH
    assert v1_assist.V_CRUISE_UNSET == v2_constants.V_CRUISE_UNSET
    assert v1_assist.CRUISE_BUTTONS_PLUS == v2_constants.CRUISE_BUTTONS_PLUS
    assert v1_assist.CRUISE_BUTTONS_MINUS == v2_constants.CRUISE_BUTTONS_MINUS
    assert v1_assist.CRUISE_BUTTON_CONFIRM_HOLD == v2_constants.CRUISE_BUTTON_CONFIRM_HOLD

  def test_facade_constants(self):
    assert facade_LIMIT_ADAPT_ACC == v2_constants.LIMIT_ADAPT_ACC
    assert facade_LIMIT_MAX_MAP_DATA_AGE == v2_constants.LIMIT_MAX_MAP_DATA_AGE
    assert facade_PCM_LONG_REQUIRED_MAX_SET_SPEED == v2_constants.PCM_LONG_REQUIRED_MAX_SET_SPEED
    assert facade_CONFIRM_SPEED_THRESHOLD == v2_constants.CONFIRM_SPEED_THRESHOLD


class TestTypesParity:
  def test_policy_enum(self):
    assert _enum_identities(v1_common.Policy) == _enum_identities(v2_types.Policy)

  def test_offset_type_enum(self):
    assert _enum_identities(v1_common.OffsetType) == _enum_identities(v2_types.OffsetType)

  def test_mode_enum(self):
    assert _enum_identities(v1_common.Mode) == _enum_identities(v2_types.Mode)

  def test_resolver_all_sources(self):
    assert v1_resolver.ALL_SOURCES == v2_resolver.ALL_SOURCES
    assert facade_ALL_SOURCES == v2_resolver.ALL_SOURCES

  def test_assist_state_groupings(self):
    assert v1_assist.ACTIVE_STATES == v2_constants.ACTIVE_STATES
    assert v1_assist.ENABLED_STATES == v2_constants.ENABLED_STATES
    assert facade_ACTIVE_STATES == v2_constants.ACTIVE_STATES
    assert facade_ENABLED_STATES == v2_constants.ENABLED_STATES


class TestHelpersParity:
  @pytest.mark.parametrize("is_metric", [True, False])
  @pytest.mark.parametrize("v_cruise,target", [(29., 30.), (31., 30.), (30., 30.)])
  def test_compare_cluster_target(self, is_metric, v_cruise, target):
    assert v1_helpers.compare_cluster_target(v_cruise, target, is_metric) == \
           v2_helpers.compare_cluster_target(v_cruise, target, is_metric)

  def test_set_speed_limit_assist_availability_disallowed_in_release_tesla(self):
    params = Params()
    params.put_bool("IsReleaseSpBranch", True, block=True)
    params.put("SpeedLimitMode", int(v1_common.Mode.assist), block=True)
    CP = _make_cp(brand='tesla')
    CP_SP = _make_cp_sp()

    assert v1_helpers.set_speed_limit_assist_availability(CP, CP_SP, params) == \
           v2_helpers.set_speed_limit_assist_availability(CP, CP_SP, params)
    assert params.get("SpeedLimitMode", return_default=True) == int(v1_common.Mode.warning)

  def test_set_speed_limit_assist_availability_always_disallowed_rivian(self):
    params = Params()
    params.put("SpeedLimitMode", int(v1_common.Mode.assist), block=True)
    CP = _make_cp(brand='rivian')
    CP_SP = _make_cp_sp()

    assert v1_helpers.set_speed_limit_assist_availability(CP, CP_SP, params) == \
           v2_helpers.set_speed_limit_assist_availability(CP, CP_SP, params)
    assert params.get("SpeedLimitMode", return_default=True) == int(v1_common.Mode.warning)

  def test_set_speed_limit_assist_availability_allowed_toyota(self):
    params = Params()
    params.put("SpeedLimitMode", int(v1_common.Mode.assist), block=True)
    CP = _make_cp(brand='toyota')
    CP_SP = _make_cp_sp()

    assert v1_helpers.set_speed_limit_assist_availability(CP, CP_SP, params) == \
           v2_helpers.set_speed_limit_assist_availability(CP, CP_SP, params)
    assert params.get("SpeedLimitMode", return_default=True) == int(v1_common.Mode.assist)

  def test_set_speed_limit_assist_availability_pcm_cruise_speed_disallowed(self):
    params = Params()
    params.put("SpeedLimitMode", int(v1_common.Mode.assist), block=True)
    CP = _make_cp(openpilot_long=False, pcm_cruise=False, brand='toyota')
    CP_SP = _make_cp_sp(pcm_cruise_speed=True)

    assert v1_helpers.set_speed_limit_assist_availability(CP, CP_SP, params) == \
           v2_helpers.set_speed_limit_assist_availability(CP, CP_SP, params)
    assert params.get("SpeedLimitMode", return_default=True) == int(v1_common.Mode.warning)


class TestResolverParity:
  def _update_both(self, v1_r, v2_r, v_ego, sm):
    v1_r.update(v_ego, sm)
    v2_r.update(v_ego, sm)

  @pytest.mark.parametrize("policy", [0, 1, 2, 3, 4])
  def test_policy_resolution(self, policy, mocker: MockerFixture):
    Params().put("SpeedLimitPolicy", policy)
    sm = _setup_resolver_sm(mocker, car_limit=50., map_limit=40.)
    v1_r = v1_resolver.SpeedLimitResolver()
    v2_r = v2_resolver.SpeedLimitResolver()
    self._update_both(v1_r, v2_r, 30., sm)
    assert _resolver_snapshot(v1_r) == _resolver_snapshot(v2_r)

  def test_combined_policy_picks_minimum(self, mocker: MockerFixture):
    Params().put("SpeedLimitPolicy", 4)
    sm = _setup_resolver_sm(mocker, car_limit=40., map_limit=50.)
    v1_r = v1_resolver.SpeedLimitResolver()
    v2_r = v2_resolver.SpeedLimitResolver()
    self._update_both(v1_r, v2_r, 30., sm)
    assert v1_r.speed_limit == v2_r.speed_limit == 40.
    assert v1_r.source == v2_r.source == SpeedLimitSource.car
    assert _resolver_snapshot(v1_r) == _resolver_snapshot(v2_r)

  def test_fixed_offset(self, mocker: MockerFixture):
    params = Params()
    params.put("SpeedLimitOffsetType", 1)
    params.put("SpeedLimitValueOffset", 5)
    sm = _setup_resolver_sm(mocker, car_limit=30.)
    v1_r = v1_resolver.SpeedLimitResolver()
    v2_r = v2_resolver.SpeedLimitResolver()
    self._update_both(v1_r, v2_r, 25., sm)
    assert v1_r.speed_limit_offset == v2_r.speed_limit_offset
    assert v1_r.speed_limit_final == v2_r.speed_limit_final
    assert _resolver_snapshot(v1_r) == _resolver_snapshot(v2_r)

  def test_percentage_offset(self, mocker: MockerFixture):
    params = Params()
    params.put("SpeedLimitOffsetType", 2)
    params.put("SpeedLimitValueOffset", 10)
    sm = _setup_resolver_sm(mocker, car_limit=30.)
    v1_r = v1_resolver.SpeedLimitResolver()
    v2_r = v2_resolver.SpeedLimitResolver()
    self._update_both(v1_r, v2_r, 25., sm)
    assert v1_r.speed_limit_offset == v2_r.speed_limit_offset
    assert _resolver_snapshot(v1_r) == _resolver_snapshot(v2_r)

  @pytest.mark.parametrize("bad_offset", [float("nan"), float("inf"), "bad"])
  def test_invalid_offset_fails_closed(self, bad_offset, mocker: MockerFixture):
    Params().put("SpeedLimitOffsetType", 1, block=True)
    sm = _setup_resolver_sm(mocker, car_limit=30.)
    v1_r = v1_resolver.SpeedLimitResolver()
    v2_r = v2_resolver.SpeedLimitResolver()
    v1_r.offset_value = bad_offset
    v2_r.offset_value = bad_offset
    self._update_both(v1_r, v2_r, 25., sm)
    assert v1_r.speed_limit_offset == v2_r.speed_limit_offset == 0.
    assert v1_r.speed_limit_final == v2_r.speed_limit_final == 30.
    assert _resolver_snapshot(v1_r) == _resolver_snapshot(v2_r)

  @pytest.mark.parametrize("bad_limit", [float("nan"), float("inf"), -10., 0.])
  def test_invalid_speed_limits_fail_closed(self, bad_limit, mocker: MockerFixture):
    Params().put("SpeedLimitPolicy", 4)
    sm = _setup_resolver_sm(mocker, car_limit=bad_limit, map_limit=bad_limit)
    v1_r = v1_resolver.SpeedLimitResolver()
    v2_r = v2_resolver.SpeedLimitResolver()
    self._update_both(v1_r, v2_r, 25., sm)
    assert v1_r.speed_limit == v2_r.speed_limit == 0.
    assert v1_r.source == v2_r.source == SpeedLimitSource.none
    assert _resolver_snapshot(v1_r) == _resolver_snapshot(v2_r)

  def test_stale_map_data_ignored(self, mocker: MockerFixture):
    Params().put("SpeedLimitPolicy", 1)
    sm = _setup_resolver_sm(mocker, map_limit=40., gps_age_s=20.)
    v1_r = v1_resolver.SpeedLimitResolver()
    v2_r = v2_resolver.SpeedLimitResolver()
    self._update_both(v1_r, v2_r, 25., sm)
    assert v1_r.speed_limit == v2_r.speed_limit == 0.
    assert v1_r.source == v2_r.source == SpeedLimitSource.none
    assert _resolver_snapshot(v1_r) == _resolver_snapshot(v2_r)

  def test_map_ahead_adaptation(self, mocker: MockerFixture):
    """
    When the upcoming speed limit is lower than v_ego and close enough, the resolver
    uses the ahead limit/distance. (v2 fixed the upstream clock-domain bug that kept
    this from ever firing on-device; under pinned test clocks v1 behaves identically.)
    """
    Params().put("SpeedLimitPolicy", 1)
    sm = _setup_resolver_sm(
      mocker,
      map_limit=30.,
      map_ahead_limit=10.,
      map_ahead_distance=50.,
      map_ahead_valid=True,
    )
    v1_r = v1_resolver.SpeedLimitResolver()
    v2_r = v2_resolver.SpeedLimitResolver()
    self._update_both(v1_r, v2_r, 30., sm)
    assert v1_r.speed_limit == v2_r.speed_limit
    assert v1_r.distance == v2_r.distance
    assert v1_r.source == v2_r.source == SpeedLimitSource.map
    assert _resolver_snapshot(v1_r) == _resolver_snapshot(v2_r)


class TestAssistParity:
  @pytest.fixture(autouse=True)
  def _assist_params(self):
    params = Params()
    params.put_bool("IsMetric", False, block=True)
    params.put("SpeedLimitMode", int(v1_common.Mode.assist), block=True)
    params.put_bool("IsReleaseSpBranch", True, block=True)
    params.put("SpeedLimitOffsetType", 0, block=True)
    params.put("SpeedLimitValueOffset", 0, block=True)

  def test_initial_state(self):
    CP = _make_cp()
    CP_SP = _make_cp_sp()
    v1_sla = v1_assist.SpeedLimitAssist(CP, CP_SP)
    v2_sla = v2_assist.SpeedLimitAssist(CP, CP_SP)

    assert _assist_snapshot(v1_sla) == _assist_snapshot(v2_sla)
    assert v1_sla.get_a_target_from_control() == v2_sla.get_a_target_from_control()
    assert v1_sla.get_v_target_from_control() == v2_sla.get_v_target_from_control()

  def test_facade_class(self):
    CP = _make_cp()
    CP_SP = _make_cp_sp()
    facade_sla = FacadeSpeedLimitAssist(CP, CP_SP)
    v2_sla = v2_assist.SpeedLimitAssist(CP, CP_SP)
    assert _assist_snapshot(facade_sla) == _assist_snapshot(v2_sla)

  def test_disabled_to_preactive(self):
    CP = _make_cp()
    CP_SP = _make_cp_sp()
    v1_sla = v1_assist.SpeedLimitAssist(CP, CP_SP)
    v2_sla = v2_assist.SpeedLimitAssist(CP, CP_SP)
    v1_events = EventsSP()
    v2_events = EventsSP()

    for _ in range(int(3. / DT_MDL)):
      v1_sla.update(True, False, SPEED_LIMITS['city'], 0., SPEED_LIMITS['highway'],
                    SPEED_LIMITS['city'], SPEED_LIMITS['city'], True, 0., v1_events)
      v2_sla.update(True, False, SPEED_LIMITS['city'], 0., SPEED_LIMITS['highway'],
                    SPEED_LIMITS['city'], SPEED_LIMITS['city'], True, 0., v2_events)

    assert _assist_snapshot(v1_sla) == _assist_snapshot(v2_sla)
    assert _event_names(v1_events) == _event_names(v2_events)

  def test_disabled_to_pending_no_speed_limit(self):
    CP = _make_cp()
    CP_SP = _make_cp_sp()
    v1_sla = v1_assist.SpeedLimitAssist(CP, CP_SP)
    v2_sla = v2_assist.SpeedLimitAssist(CP, CP_SP)
    v1_events = EventsSP()
    v2_events = EventsSP()

    for _ in range(int(3. / DT_MDL)):
      v1_sla.update(True, False, SPEED_LIMITS['highway'], 0., SPEED_LIMITS['city'], 0., 0., False, 0., v1_events)
      v2_sla.update(True, False, SPEED_LIMITS['highway'], 0., SPEED_LIMITS['city'], 0., 0., False, 0., v2_events)

    assert _assist_snapshot(v1_sla) == _assist_snapshot(v2_sla)
    assert _event_names(v1_events) == _event_names(v2_events)

  def test_preactive_to_active_with_max_speed_confirmation(self):
    CP = _make_cp()
    CP_SP = _make_cp_sp()
    v1_sla = v1_assist.SpeedLimitAssist(CP, CP_SP)
    v2_sla = v2_assist.SpeedLimitAssist(CP, CP_SP)
    v1_events = EventsSP()
    v2_events = EventsSP()

    v1_sla.state = SpeedLimitAssistState.preActive
    v2_sla.state = SpeedLimitAssistState.preActive

    pcm_long_max = v1_constants.PCM_LONG_REQUIRED_MAX_SET_SPEED[False][1]
    v1_sla.update(True, False, SPEED_LIMITS['city'], 0., pcm_long_max,
                  SPEED_LIMITS['highway'], SPEED_LIMITS['highway'], True, 0., v1_events)
    v2_sla.update(True, False, SPEED_LIMITS['city'], 0., pcm_long_max,
                  SPEED_LIMITS['highway'], SPEED_LIMITS['highway'], True, 0., v2_events)

    assert _assist_snapshot(v1_sla) == _assist_snapshot(v2_sla)
    assert v1_sla.output_v_target == v2_sla.output_v_target == SPEED_LIMITS['highway']
    assert _event_names(v1_events) == _event_names(v2_events)

  def test_preactive_timeout_to_inactive(self):
    CP = _make_cp()
    CP_SP = _make_cp_sp()
    v1_sla = v1_assist.SpeedLimitAssist(CP, CP_SP)
    v2_sla = v2_assist.SpeedLimitAssist(CP, CP_SP)
    v1_events = EventsSP()
    v2_events = EventsSP()

    v1_sla.state = SpeedLimitAssistState.preActive
    v2_sla.state = SpeedLimitAssistState.preActive
    v1_sla.pre_active_timer = int(facade_PRE_ACTIVE_GUARD_PERIOD[True] / DT_MDL)
    v2_sla.pre_active_timer = int(v2_constants.PRE_ACTIVE_GUARD_PERIOD[True] / DT_MDL)

    # one update already decrements timer
    v1_sla.update(True, False, SPEED_LIMITS['city'], 0., SPEED_LIMITS['highway'],
                  SPEED_LIMITS['city'], SPEED_LIMITS['city'], True, 0., v1_events)
    v2_sla.update(True, False, SPEED_LIMITS['city'], 0., SPEED_LIMITS['highway'],
                  SPEED_LIMITS['city'], SPEED_LIMITS['city'], True, 0., v2_events)

    for _ in range(int(facade_PRE_ACTIVE_GUARD_PERIOD[True] / DT_MDL)):
      v1_sla.update(True, False, SPEED_LIMITS['city'], 0., SPEED_LIMITS['highway'],
                    SPEED_LIMITS['city'], SPEED_LIMITS['city'], True, 0., v1_events)
      v2_sla.update(True, False, SPEED_LIMITS['city'], 0., SPEED_LIMITS['highway'],
                    SPEED_LIMITS['city'], SPEED_LIMITS['city'], True, 0., v2_events)

    assert _assist_snapshot(v1_sla) == _assist_snapshot(v2_sla)
    assert v1_sla.state == v2_sla.state == SpeedLimitAssistState.inactive

  def test_pending_to_active_when_speed_limit_available(self):
    CP = _make_cp()
    CP_SP = _make_cp_sp()
    v1_sla = v1_assist.SpeedLimitAssist(CP, CP_SP)
    v2_sla = v2_assist.SpeedLimitAssist(CP, CP_SP)
    v1_events = EventsSP()
    v2_events = EventsSP()

    pcm_long_max = v1_constants.PCM_LONG_REQUIRED_MAX_SET_SPEED[False][1]
    v1_sla.state = SpeedLimitAssistState.pending
    v2_sla.state = SpeedLimitAssistState.pending
    v1_sla.v_cruise_cluster_prev = pcm_long_max
    v2_sla.v_cruise_cluster_prev = pcm_long_max
    v1_sla.prev_v_cruise_cluster_conv = round(pcm_long_max * CV.MS_TO_MPH)
    v2_sla.prev_v_cruise_cluster_conv = round(pcm_long_max * CV.MS_TO_MPH)

    v1_sla.update(True, False, SPEED_LIMITS['highway'], 0., pcm_long_max,
                  SPEED_LIMITS['highway'], SPEED_LIMITS['highway'], True, 0., v1_events)
    v2_sla.update(True, False, SPEED_LIMITS['highway'], 0., pcm_long_max,
                  SPEED_LIMITS['highway'], SPEED_LIMITS['highway'], True, 0., v2_events)

    assert _assist_snapshot(v1_sla) == _assist_snapshot(v2_sla)
    assert v1_sla.state == v2_sla.state == SpeedLimitAssistState.active

  def test_active_to_adapting_transition(self):
    CP = _make_cp()
    CP_SP = _make_cp_sp()
    v1_sla = v1_assist.SpeedLimitAssist(CP, CP_SP)
    v2_sla = v2_assist.SpeedLimitAssist(CP, CP_SP)
    v1_events = EventsSP()
    v2_events = EventsSP()

    pcm_long_max = v1_constants.PCM_LONG_REQUIRED_MAX_SET_SPEED[False][1]
    v1_sla.state = SpeedLimitAssistState.active
    v2_sla.state = SpeedLimitAssistState.active
    v1_sla.v_cruise_cluster = pcm_long_max
    v2_sla.v_cruise_cluster = pcm_long_max
    v1_sla.v_cruise_cluster_prev = pcm_long_max
    v2_sla.v_cruise_cluster_prev = pcm_long_max
    v1_sla.prev_v_cruise_cluster_conv = round(pcm_long_max * CV.MS_TO_MPH)
    v2_sla.prev_v_cruise_cluster_conv = round(pcm_long_max * CV.MS_TO_MPH)

    v1_sla.update(True, False, SPEED_LIMITS['highway'] + 2, 0., pcm_long_max,
                  SPEED_LIMITS['highway'], SPEED_LIMITS['highway'], True, 0., v1_events)
    v2_sla.update(True, False, SPEED_LIMITS['highway'] + 2, 0., pcm_long_max,
                  SPEED_LIMITS['highway'], SPEED_LIMITS['highway'], True, 0., v2_events)

    assert _assist_snapshot(v1_sla) == _assist_snapshot(v2_sla)
    assert v1_sla.state == v2_sla.state == SpeedLimitAssistState.adapting

  def test_long_disengaged_to_disabled(self):
    CP = _make_cp()
    CP_SP = _make_cp_sp()
    v1_sla = v1_assist.SpeedLimitAssist(CP, CP_SP)
    v2_sla = v2_assist.SpeedLimitAssist(CP, CP_SP)
    v1_events = EventsSP()
    v2_events = EventsSP()

    pcm_long_max = v1_constants.PCM_LONG_REQUIRED_MAX_SET_SPEED[False][1]
    v1_sla.state = SpeedLimitAssistState.active
    v2_sla.state = SpeedLimitAssistState.active
    v1_sla.v_cruise_cluster = pcm_long_max
    v2_sla.v_cruise_cluster = pcm_long_max
    v1_sla.v_cruise_cluster_prev = pcm_long_max
    v2_sla.v_cruise_cluster_prev = pcm_long_max
    v1_sla.prev_v_cruise_cluster_conv = round(pcm_long_max * CV.MS_TO_MPH)
    v2_sla.prev_v_cruise_cluster_conv = round(pcm_long_max * CV.MS_TO_MPH)

    v1_sla.update(False, False, SPEED_LIMITS['city'], 0., pcm_long_max,
                  SPEED_LIMITS['city'], SPEED_LIMITS['city'], True, 0., v1_events)
    v2_sla.update(False, False, SPEED_LIMITS['city'], 0., pcm_long_max,
                  SPEED_LIMITS['city'], SPEED_LIMITS['city'], True, 0., v2_events)

    assert _assist_snapshot(v1_sla) == _assist_snapshot(v2_sla)
    assert v1_sla.state == v2_sla.state == SpeedLimitAssistState.disabled
    assert v1_sla.output_v_target == v2_sla.output_v_target == v2_constants.V_CRUISE_UNSET

  def test_rapid_speed_limit_changes(self):
    CP = _make_cp()
    CP_SP = _make_cp_sp()
    v1_sla = v1_assist.SpeedLimitAssist(CP, CP_SP)
    v2_sla = v2_assist.SpeedLimitAssist(CP, CP_SP)
    v1_events = EventsSP()
    v2_events = EventsSP()

    pcm_long_max = v1_constants.PCM_LONG_REQUIRED_MAX_SET_SPEED[False][1]
    v1_sla.state = SpeedLimitAssistState.active
    v2_sla.state = SpeedLimitAssistState.active
    v1_sla.v_cruise_cluster = pcm_long_max
    v2_sla.v_cruise_cluster = pcm_long_max
    v1_sla.v_cruise_cluster_prev = pcm_long_max
    v2_sla.v_cruise_cluster_prev = pcm_long_max
    v1_sla.prev_v_cruise_cluster_conv = round(pcm_long_max * CV.MS_TO_MPH)
    v2_sla.prev_v_cruise_cluster_conv = round(pcm_long_max * CV.MS_TO_MPH)

    for limit in (SPEED_LIMITS['highway'], SPEED_LIMITS['freeway']):
      v1_sla.update(True, False, limit, 0., pcm_long_max, limit, limit, True, 0., v1_events)
      v2_sla.update(True, False, limit, 0., pcm_long_max, limit, limit, True, 0., v2_events)

    assert _assist_snapshot(v1_sla) == _assist_snapshot(v2_sla)
    assert v1_sla.state in v2_constants.ACTIVE_STATES
    assert v2_sla.state in v2_constants.ACTIVE_STATES

  def test_a_target_returns_a_ego(self):
    CP = _make_cp()
    CP_SP = _make_cp_sp()
    v1_sla = v1_assist.SpeedLimitAssist(CP, CP_SP)
    v2_sla = v2_assist.SpeedLimitAssist(CP, CP_SP)

    assert v1_sla.get_a_target_from_control() == v2_sla.get_a_target_from_control() == v1_sla.a_ego == v2_sla.a_ego

  def test_disallowed_brand_tesla(self):
    params = Params()
    params.put_bool("IsReleaseSpBranch", True, block=True)
    params.put("SpeedLimitMode", int(v1_common.Mode.assist), block=True)
    CP = _make_cp(brand='tesla')
    CP_SP = _make_cp_sp()
    v1_sla = v1_assist.SpeedLimitAssist(CP, CP_SP)
    v2_sla = v2_assist.SpeedLimitAssist(CP, CP_SP)

    assert not v1_sla.enabled
    assert v1_sla.enabled == v2_sla.enabled
    assert params.get("SpeedLimitMode", return_default=True) == int(v1_common.Mode.warning)
