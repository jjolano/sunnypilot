from types import SimpleNamespace

from cereal import custom
from openpilot.common.constants import CV
from openpilot.selfdrive.controls.lib.longitudinal_decision import CandidateRole, DecisionSource
import openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner as longitudinal_planner
from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlannerSP


LongitudinalPlanSource = custom.LongitudinalPlanSP.LongitudinalPlanSource


def test_target_selection_tie_prefers_cruise_when_speed_limit_auto_inactive():
  source, v_target, a_target = longitudinal_planner.select_lowest_longitudinal_target(
    speed_limit_active=False,
    cruise=(20.0, 0.1),
    scc_vision=(20.0, 0.2),
    scc_map=(20.0, 0.3),
    speed_limit_assist=(20.0, 0.4),
    osm_traffic_control=(20.0, 0.5),
  )

  assert source == LongitudinalPlanSource.cruise
  assert v_target == 20.0
  assert a_target == 0.1


def test_target_selection_tie_prefers_speed_limit_source_when_auto_active():
  source, v_target, a_target = longitudinal_planner.select_lowest_longitudinal_target(
    speed_limit_active=True,
    cruise=(20.0, 0.1),
    scc_vision=(20.0, 0.2),
    scc_map=(20.0, 0.3),
    speed_limit_assist=(20.0, 0.4),
    osm_traffic_control=(20.0, 0.5),
  )

  assert source == LongitudinalPlanSource.speedLimitAssist
  assert v_target == 20.0
  assert a_target == 0.1


def test_speed_limit_auto_tie_uses_manual_cruise_acceleration_seed():
  source, v_target, a_target = longitudinal_planner.select_lowest_longitudinal_target(
    speed_limit_active=True,
    cruise=(15.0, 0.1),
    scc_vision=(20.0, -0.5),
    scc_map=(255.0, 0.0),
    speed_limit_assist=(20.0, 0.1),
    osm_traffic_control=(255.0, 0.0),
  )

  assert source == LongitudinalPlanSource.speedLimitAssist
  assert v_target == 20.0
  assert a_target == 0.1


def test_target_selection_keeps_lowest_non_cruise_candidate():
  source, v_target, a_target = longitudinal_planner.select_lowest_longitudinal_target(
    speed_limit_active=True,
    cruise=(15.0, 0.1),
    scc_vision=(20.0, 0.2),
    scc_map=(18.0, 0.3),
    speed_limit_assist=(19.0, 0.4),
    osm_traffic_control=(17.0, 0.5),
  )

  assert source == LongitudinalPlanSource.osmTrafficControl
  assert v_target == 17.0
  assert a_target == 0.5


class FakeResolver:
  speed_limit = 30.0
  speed_limit_final_last = 30.0
  speed_limit_valid = True
  speed_limit_last_valid = False
  distance = 0.0
  coast_accel = None

  def update(self, v_ego, sm, coast_accel=None):
    self.coast_accel = coast_accel


class FakeSpeedLimitAssist:
  is_active = True
  output_v_target = 30.0
  output_a_target = -0.4

  def update(self, *args, **kwargs):
    pass


class FakeSmartCruiseControl:
  def __init__(self):
    self.vision = SimpleNamespace(output_v_target=255.0, output_a_target=0.0, is_active=False)
    self.map = SimpleNamespace(output_v_target=255.0, output_a_target=0.0, is_active=False)

  def update(self, sm, long_enabled, long_override, v_ego, a_ego, v_cruise):
    pass


class FakeOsmTrafficControlPrior:
  output_v_target = 255.0
  output_a_target = 0.0
  active = False

  def update(self, *args):
    pass


class FakeSubMaster(dict):
  def __getitem__(self, key):
    return super().__getitem__(key)


def test_speed_limit_auto_uses_assist_source_without_acceleration_seed():
  planner = LongitudinalPlannerSP.__new__(LongitudinalPlannerSP)
  planner.scc = FakeSmartCruiseControl()
  planner.resolver = FakeResolver()
  planner.sla = FakeSpeedLimitAssist()
  planner.osm_traffic_control_prior = FakeOsmTrafficControlPrior()
  planner.events_sp = SimpleNamespace()

  sm = FakeSubMaster({
    'carState': SimpleNamespace(vCruiseCluster=20.0),
    'carControl': SimpleNamespace(enabled=True, cruiseControl=SimpleNamespace(override=False)),
  })

  v_cruise = 20.0 * CV.KPH_TO_MS
  a_ego = 0.2
  v_target, a_target = LongitudinalPlannerSP.update_targets(planner, sm, v_ego=15.0, a_ego=a_ego, v_cruise=v_cruise)

  assert v_target == FakeSpeedLimitAssist.output_v_target
  assert a_target == a_ego
  assert planner.source == LongitudinalPlanSource.speedLimitAssist


def test_speed_limit_decision_candidate_uses_acceleration_seed():
  planner = LongitudinalPlannerSP.__new__(LongitudinalPlannerSP)
  planner.scc = FakeSmartCruiseControl()
  planner.resolver = FakeResolver()
  planner.sla = FakeSpeedLimitAssist()
  planner.osm_traffic_control_prior = FakeOsmTrafficControlPrior()
  planner.events_sp = SimpleNamespace()

  sm = FakeSubMaster({
    'carState': SimpleNamespace(vCruiseCluster=20.0),
    'carControl': SimpleNamespace(enabled=True, cruiseControl=SimpleNamespace(override=False)),
  })

  a_ego = 0.2
  v_cruise = 20.0 * CV.KPH_TO_MS
  LongitudinalPlannerSP.update_targets(planner, sm, v_ego=15.0, a_ego=a_ego, v_cruise=v_cruise)

  speed_limit_candidate = next(candidate for candidate in planner.decision_candidates_sp
                               if candidate.source == DecisionSource.SPEED_LIMIT)
  assert speed_limit_candidate.a_target == a_ego


def test_speed_limit_resolver_receives_coast_accel():
  planner = LongitudinalPlannerSP.__new__(LongitudinalPlannerSP)
  planner.scc = FakeSmartCruiseControl()
  planner.resolver = FakeResolver()
  planner.sla = FakeSpeedLimitAssist()
  planner.osm_traffic_control_prior = FakeOsmTrafficControlPrior()
  planner.events_sp = SimpleNamespace()

  sm = FakeSubMaster({
    'carState': SimpleNamespace(vCruiseCluster=20.0),
    'carControl': SimpleNamespace(enabled=True, cruiseControl=SimpleNamespace(override=False)),
  })

  coast_accel = -0.3
  v_cruise = 20.0 * CV.KPH_TO_MS
  LongitudinalPlannerSP.update_targets(planner, sm, v_ego=15.0, a_ego=0.0, v_cruise=v_cruise,
                                       coast_accel=coast_accel)

  assert planner.resolver.coast_accel == coast_accel


def test_sp_candidate_builder_includes_cruise_and_active_advisories():
  candidates = longitudinal_planner.build_sp_longitudinal_candidates(
    speed_limit_active=True,
    cruise=(25.0, 0.1),
    scc_vision=(22.0, -0.2),
    scc_vision_active=True,
    scc_map=(24.0, -0.1),
    scc_map_active=False,
    speed_limit_assist=(20.0, -0.3),
    osm_traffic_control=(18.0, -0.4),
    osm_traffic_control_active=True,
  )

  assert [candidate.source for candidate in candidates] == [
    DecisionSource.CRUISE,
    DecisionSource.SPEED_LIMIT,
    DecisionSource.SCC_VISION,
    DecisionSource.OSM_TRAFFIC_CONTROL,
  ]
  assert all(candidate.valid for candidate in candidates)
  assert candidates[0].role == CandidateRole.DRIVER_INTENT
  assert all(candidate.role == CandidateRole.ADVISORY_CAP for candidate in candidates[1:])


def test_sp_candidate_builder_skips_inactive_advisories():
  candidates = longitudinal_planner.build_sp_longitudinal_candidates(
    speed_limit_active=False,
    cruise=(25.0, 0.1),
    scc_vision=(22.0, -0.2),
    scc_vision_active=False,
    scc_map=(24.0, -0.1),
    scc_map_active=False,
    speed_limit_assist=(20.0, -0.3),
    osm_traffic_control=(18.0, -0.4),
    osm_traffic_control_active=False,
  )

  assert [candidate.source for candidate in candidates] == [DecisionSource.CRUISE]
