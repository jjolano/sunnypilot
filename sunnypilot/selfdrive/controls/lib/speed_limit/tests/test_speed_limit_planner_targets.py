from types import SimpleNamespace

import pytest

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


class SequenceSpeedLimitAssist:
  def __init__(self, active_sequence, target_sequence=None):
    self.active_sequence = list(active_sequence)
    self.target_sequence = list(target_sequence or [30.0] * len(self.active_sequence))
    self.output_v_target = self.target_sequence[0]
    self.output_a_target = -0.4
    self.is_active = self.active_sequence[0]
    self.auto_enabled = self.is_active
    self.update_count = 0

  def update(self, *args, **kwargs):
    index = min(self.update_count, len(self.active_sequence) - 1)
    self.is_active = self.active_sequence[index]
    self.auto_enabled = self.is_active
    self.output_v_target = self.target_sequence[min(index, len(self.target_sequence) - 1)]
    self.update_count += 1


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


def make_sm(v_cruise_cluster=20.0, lead_status=False, d_rel=100.0, v_rel=0.0, y_rel=0.0,
            gas_pressed=False, brake_pressed=False):
  return FakeSubMaster({
    'carState': SimpleNamespace(
      vCruiseCluster=v_cruise_cluster,
      gasPressed=gas_pressed,
      brakePressed=brake_pressed,
    ),
    'carControl': SimpleNamespace(enabled=True, cruiseControl=SimpleNamespace(override=False)),
    'radarState': SimpleNamespace(leadOne=SimpleNamespace(status=lead_status, dRel=d_rel, vRel=v_rel, yRel=y_rel)),
  })


def test_speed_limit_auto_uses_full_assist_target_like_manual_cruise():
  planner = LongitudinalPlannerSP.__new__(LongitudinalPlannerSP)
  planner.scc = FakeSmartCruiseControl()
  planner.resolver = FakeResolver()
  planner.sla = FakeSpeedLimitAssist()
  planner.osm_traffic_control_prior = FakeOsmTrafficControlPrior()
  planner.events_sp = SimpleNamespace()

  sm = make_sm(v_cruise_cluster=20.0)

  v_cruise = 20.0 * CV.KPH_TO_MS
  a_ego = 0.2
  v_ego = 15.0
  v_target, a_target = LongitudinalPlannerSP.update_targets(planner, sm, v_ego=v_ego, a_ego=a_ego, v_cruise=v_cruise)

  assert v_target == pytest.approx(planner.sla.output_v_target)
  assert a_target == a_ego
  assert planner.source == LongitudinalPlanSource.speedLimitAssist


def test_speed_limit_auto_speedup_blocked_by_close_closing_lead():
  planner = LongitudinalPlannerSP.__new__(LongitudinalPlannerSP)
  planner.scc = FakeSmartCruiseControl()
  planner.resolver = FakeResolver()
  planner.sla = FakeSpeedLimitAssist()
  planner.osm_traffic_control_prior = FakeOsmTrafficControlPrior()
  planner.events_sp = SimpleNamespace()
  planner.source = LongitudinalPlanSource.cruise
  planner._speed_limit_handoff_active = False
  planner._speed_limit_active_prev = False

  sm = make_sm(v_cruise_cluster=20.0, lead_status=True, d_rel=20.0, v_rel=-1.4)

  v_ego = 15.0
  v_cruise = 20.0 * CV.KPH_TO_MS
  v_target, a_target = LongitudinalPlannerSP.update_targets(planner, sm, v_ego=v_ego, a_ego=0.2, v_cruise=v_cruise)

  assert planner.source == LongitudinalPlanSource.speedLimitAssist
  assert v_target == pytest.approx(v_ego)
  assert a_target <= 0.0


def test_cruise_speedup_blocked_by_close_closing_lead():
  planner = LongitudinalPlannerSP.__new__(LongitudinalPlannerSP)
  planner.scc = FakeSmartCruiseControl()
  planner.resolver = FakeResolver()
  planner.sla = SequenceSpeedLimitAssist(active_sequence=[False], target_sequence=[255.0])
  planner.osm_traffic_control_prior = FakeOsmTrafficControlPrior()
  planner.events_sp = SimpleNamespace()
  planner.source = LongitudinalPlanSource.cruise
  planner._speed_limit_handoff_active = False
  planner._speed_limit_active_prev = False

  sm = make_sm(v_cruise_cluster=67.0, lead_status=True, d_rel=20.0, v_rel=-1.4)

  v_target, a_target = LongitudinalPlannerSP.update_targets(planner, sm, v_ego=15.0, a_ego=0.2, v_cruise=18.61)

  assert planner.source == LongitudinalPlanSource.cruise
  assert v_target == pytest.approx(15.0)
  assert a_target <= 0.0


def test_lead_speedup_guard_allows_far_lead():
  assert not longitudinal_planner.should_block_lead_speedup(
    v_ego=15.0,
    lead_status=True,
    d_rel=60.0,
    v_rel=-1.4,
    y_rel=0.0,
    gas_pressed=False,
    brake_pressed=False,
  )


def test_lead_speedup_guard_allows_opening_lead():
  assert not longitudinal_planner.should_block_lead_speedup(
    v_ego=15.0,
    lead_status=True,
    d_rel=20.0,
    v_rel=0.2,
    y_rel=0.0,
    gas_pressed=False,
    brake_pressed=False,
  )


def test_lead_speedup_guard_allows_driver_gas_override():
  assert not longitudinal_planner.should_block_lead_speedup(
    v_ego=15.0,
    lead_status=True,
    d_rel=20.0,
    v_rel=-1.4,
    y_rel=0.0,
    gas_pressed=True,
    brake_pressed=False,
  )


def test_active_speed_limit_decision_driver_uses_acceleration_seed():
  planner = LongitudinalPlannerSP.__new__(LongitudinalPlannerSP)
  planner.scc = FakeSmartCruiseControl()
  planner.resolver = FakeResolver()
  planner.sla = FakeSpeedLimitAssist()
  planner.osm_traffic_control_prior = FakeOsmTrafficControlPrior()
  planner.events_sp = SimpleNamespace()

  sm = make_sm(v_cruise_cluster=20.0)

  a_ego = 0.2
  v_cruise = 20.0 * CV.KPH_TO_MS
  LongitudinalPlannerSP.update_targets(planner, sm, v_ego=15.0, a_ego=a_ego, v_cruise=v_cruise)

  cruise_candidate = next(candidate for candidate in planner.decision_candidates_sp
                          if candidate.source == DecisionSource.CRUISE)
  assert cruise_candidate.a_target == a_ego


def test_active_speed_limit_decision_driver_uses_governed_speedup_target():
  planner = LongitudinalPlannerSP.__new__(LongitudinalPlannerSP)
  planner.scc = FakeSmartCruiseControl()
  planner.resolver = FakeResolver()
  planner.sla = FakeSpeedLimitAssist()
  planner.osm_traffic_control_prior = FakeOsmTrafficControlPrior()
  planner.events_sp = SimpleNamespace()

  sm = make_sm(v_cruise_cluster=20.0)

  v_ego = 15.0
  a_ego = 0.2
  v_cruise = 20.0 * CV.KPH_TO_MS
  LongitudinalPlannerSP.update_targets(planner, sm, v_ego=v_ego, a_ego=a_ego, v_cruise=v_cruise)

  expected = v_ego + longitudinal_planner.SPEED_LIMIT_SPEED_UP_ACCEL_CAP * longitudinal_planner.SPEED_LIMIT_SPEED_UP_LOOKAHEAD
  cruise_candidate = next(candidate for candidate in planner.decision_candidates_sp
                          if candidate.source == DecisionSource.CRUISE)
  assert cruise_candidate.v_target == pytest.approx(expected)
  assert cruise_candidate.v_target < FakeSpeedLimitAssist.output_v_target


def test_active_speed_limit_above_manual_cruise_is_decision_driver_intent():
  planner = LongitudinalPlannerSP.__new__(LongitudinalPlannerSP)
  planner.scc = FakeSmartCruiseControl()
  planner.resolver = FakeResolver()
  planner.sla = FakeSpeedLimitAssist()
  planner.osm_traffic_control_prior = FakeOsmTrafficControlPrior()
  planner.events_sp = SimpleNamespace()

  sm = make_sm(v_cruise_cluster=20.0)

  v_ego = 15.0
  manual_cruise = 20.0 * CV.KPH_TO_MS
  LongitudinalPlannerSP.update_targets(planner, sm, v_ego=v_ego, a_ego=0.2, v_cruise=manual_cruise)

  expected = v_ego + longitudinal_planner.SPEED_LIMIT_SPEED_UP_ACCEL_CAP * longitudinal_planner.SPEED_LIMIT_SPEED_UP_LOOKAHEAD
  cruise_candidate = next(candidate for candidate in planner.decision_candidates_sp
                          if candidate.source == DecisionSource.CRUISE)
  speed_limit_candidates = [candidate for candidate in planner.decision_candidates_sp
                            if candidate.source == DecisionSource.SPEED_LIMIT]
  assert cruise_candidate.v_target == pytest.approx(expected)
  assert cruise_candidate.role == CandidateRole.DRIVER_INTENT
  assert speed_limit_candidates == []


def test_lead_speedup_guard_allows_lateral_exit_lead():
  assert not longitudinal_planner.should_block_lead_speedup(
    v_ego=15.0,
    lead_status=True,
    d_rel=20.0,
    v_rel=-1.4,
    y_rel=longitudinal_planner.LEAD_SPEEDUP_GUARD_LATERAL_EXIT_Y_REL,
    gas_pressed=False,
    brake_pressed=False,
  )


def test_speed_limit_auto_disable_handoff_coasts_to_limit_before_manual_cruise_snap():
  planner = LongitudinalPlannerSP.__new__(LongitudinalPlannerSP)
  planner.scc = FakeSmartCruiseControl()
  planner.resolver = FakeResolver()
  planner.sla = SequenceSpeedLimitAssist(active_sequence=[True, False], target_sequence=[30.0, 255.0])
  planner.osm_traffic_control_prior = FakeOsmTrafficControlPrior()
  planner.events_sp = SimpleNamespace()
  planner.source = LongitudinalPlanSource.cruise
  planner._speed_limit_handoff_active = False
  planner._speed_limit_active_prev = False

  sm = make_sm(v_cruise_cluster=67.0)

  v_ego = 21.0
  a_ego = 0.2
  manual_cruise = 18.61

  LongitudinalPlannerSP.update_targets(planner, sm, v_ego=v_ego, a_ego=a_ego, v_cruise=manual_cruise)
  v_target, a_target = LongitudinalPlannerSP.update_targets(planner, sm, v_ego=v_ego, a_ego=a_ego, v_cruise=manual_cruise)

  assert planner.source == LongitudinalPlanSource.speedLimitAssist
  assert v_target == pytest.approx(v_ego)
  assert a_target <= 0.0
  assert v_target > manual_cruise


def test_speed_limit_handoff_decision_candidate_uses_handoff_target():
  planner = LongitudinalPlannerSP.__new__(LongitudinalPlannerSP)
  planner.scc = FakeSmartCruiseControl()
  planner.resolver = FakeResolver()
  planner.sla = SequenceSpeedLimitAssist(active_sequence=[True, False], target_sequence=[30.0, 255.0])
  planner.osm_traffic_control_prior = FakeOsmTrafficControlPrior()
  planner.events_sp = SimpleNamespace()
  planner.source = LongitudinalPlanSource.cruise
  planner._speed_limit_handoff_active = False
  planner._speed_limit_active_prev = False

  sm = make_sm(v_cruise_cluster=67.0)

  v_ego = 21.0
  manual_cruise = 18.61
  LongitudinalPlannerSP.update_targets(planner, sm, v_ego=v_ego, a_ego=0.2, v_cruise=manual_cruise)
  LongitudinalPlannerSP.update_targets(planner, sm, v_ego=v_ego, a_ego=0.2, v_cruise=manual_cruise)

  cruise_candidate = next(candidate for candidate in planner.decision_candidates_sp
                          if candidate.source == DecisionSource.CRUISE)
  speed_limit_candidates = [candidate for candidate in planner.decision_candidates_sp
                            if candidate.source == DecisionSource.SPEED_LIMIT]
  assert cruise_candidate.v_target == pytest.approx(v_ego)
  assert cruise_candidate.a_target <= 0.0
  assert speed_limit_candidates == []


def test_speed_limit_handoff_uses_limit_target_when_ego_is_above_limit():
  planner = LongitudinalPlannerSP.__new__(LongitudinalPlannerSP)
  planner.scc = FakeSmartCruiseControl()
  planner.resolver = FakeResolver()
  planner.sla = SequenceSpeedLimitAssist(active_sequence=[True, False], target_sequence=[30.0, 255.0])
  planner.osm_traffic_control_prior = FakeOsmTrafficControlPrior()
  planner.events_sp = SimpleNamespace()
  planner.source = LongitudinalPlanSource.cruise
  planner._speed_limit_handoff_active = False
  planner._speed_limit_active_prev = False

  sm = make_sm(v_cruise_cluster=67.0)

  manual_cruise = 18.61
  LongitudinalPlannerSP.update_targets(planner, sm, v_ego=31.0, a_ego=0.2, v_cruise=manual_cruise)
  v_target, a_target = LongitudinalPlannerSP.update_targets(planner, sm, v_ego=31.0, a_ego=0.2, v_cruise=manual_cruise)

  assert planner.source == LongitudinalPlanSource.speedLimitAssist
  assert v_target == pytest.approx(30.0)
  assert a_target <= 0.0


def test_speed_limit_handoff_exits_near_manual_cruise_target():
  planner = LongitudinalPlannerSP.__new__(LongitudinalPlannerSP)
  planner.scc = FakeSmartCruiseControl()
  planner.resolver = FakeResolver()
  planner.sla = SequenceSpeedLimitAssist(active_sequence=[True, False, False, False], target_sequence=[30.0, 255.0, 255.0, 255.0])
  planner.osm_traffic_control_prior = FakeOsmTrafficControlPrior()
  planner.events_sp = SimpleNamespace()
  planner.source = LongitudinalPlanSource.cruise
  planner._speed_limit_handoff_active = False
  planner._speed_limit_active_prev = False

  sm = make_sm(v_cruise_cluster=67.0)

  manual_cruise = 18.61
  LongitudinalPlannerSP.update_targets(planner, sm, v_ego=21.0, a_ego=0.2, v_cruise=manual_cruise)
  LongitudinalPlannerSP.update_targets(planner, sm, v_ego=21.0, a_ego=0.2, v_cruise=manual_cruise)
  v_target, a_target = LongitudinalPlannerSP.update_targets(planner, sm, v_ego=20.0, a_ego=0.1, v_cruise=manual_cruise)

  assert planner.source == LongitudinalPlanSource.speedLimitAssist
  assert v_target == pytest.approx(20.0)
  assert a_target <= 0.0

  v_target, _ = LongitudinalPlannerSP.update_targets(
    planner,
    sm,
    v_ego=manual_cruise + longitudinal_planner.SPEED_LIMIT_HANDOFF_EXIT_MARGIN,
    a_ego=0.0,
    v_cruise=manual_cruise,
  )

  assert planner.source == LongitudinalPlanSource.cruise
  assert v_target == pytest.approx(manual_cruise)


def test_speed_limit_handoff_exits_when_manual_cruise_reaches_limit_target():
  planner = LongitudinalPlannerSP.__new__(LongitudinalPlannerSP)
  planner.scc = FakeSmartCruiseControl()
  planner.resolver = FakeResolver()
  planner.sla = SequenceSpeedLimitAssist(active_sequence=[True, False, False], target_sequence=[30.0, 255.0, 255.0])
  planner.osm_traffic_control_prior = FakeOsmTrafficControlPrior()
  planner.events_sp = SimpleNamespace()
  planner.source = LongitudinalPlanSource.cruise
  planner._speed_limit_handoff_active = False
  planner._speed_limit_active_prev = False

  sm = make_sm(v_cruise_cluster=67.0)

  LongitudinalPlannerSP.update_targets(planner, sm, v_ego=31.0, a_ego=0.2, v_cruise=18.61)
  LongitudinalPlannerSP.update_targets(planner, sm, v_ego=31.0, a_ego=0.2, v_cruise=18.61)
  v_target, _ = LongitudinalPlannerSP.update_targets(planner, sm, v_ego=31.0, a_ego=0.0, v_cruise=30.0)

  assert planner.source == LongitudinalPlanSource.cruise
  assert v_target == pytest.approx(30.0)


def test_speed_limit_handoff_allows_lower_scc_candidate():
  planner = LongitudinalPlannerSP.__new__(LongitudinalPlannerSP)
  planner.scc = FakeSmartCruiseControl()
  planner.scc.vision.output_v_target = 20.0
  planner.scc.vision.output_a_target = -0.6
  planner.resolver = FakeResolver()
  planner.sla = SequenceSpeedLimitAssist(active_sequence=[True, False], target_sequence=[30.0, 255.0])
  planner.osm_traffic_control_prior = FakeOsmTrafficControlPrior()
  planner.events_sp = SimpleNamespace()
  planner.source = LongitudinalPlanSource.cruise
  planner._speed_limit_handoff_active = False
  planner._speed_limit_active_prev = False

  sm = make_sm(v_cruise_cluster=67.0)

  LongitudinalPlannerSP.update_targets(planner, sm, v_ego=31.0, a_ego=0.2, v_cruise=18.61)
  v_target, a_target = LongitudinalPlannerSP.update_targets(planner, sm, v_ego=31.0, a_ego=0.2, v_cruise=18.61)

  assert planner.source == LongitudinalPlanSource.sccVision
  assert v_target == pytest.approx(20.0)
  assert a_target == pytest.approx(-0.6)


def test_speed_limit_resolver_receives_coast_accel():
  planner = LongitudinalPlannerSP.__new__(LongitudinalPlannerSP)
  planner.scc = FakeSmartCruiseControl()
  planner.resolver = FakeResolver()
  planner.sla = FakeSpeedLimitAssist()
  planner.osm_traffic_control_prior = FakeOsmTrafficControlPrior()
  planner.events_sp = SimpleNamespace()

  sm = make_sm(v_cruise_cluster=20.0)

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
