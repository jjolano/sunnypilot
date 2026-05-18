import unittest
from types import SimpleNamespace
from typing import Callable, cast

from openpilot.selfdrive.controls.lib.longitudinal_decision import CandidateRole, DecisionSource, LongitudinalDecisionTelemetry
from openpilot.sunnypilot.selfdrive.controls.lib import longitudinal_planner as sp_longitudinal_planner
from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import (
  SignalProviderCandidate,
  LongitudinalPlanSource,
  build_sp_candidates_from_signal_providers,
  build_sp_longitudinal_candidates,
  select_lowest_longitudinal_target,
)

publish_decision_layer_telemetry = cast(
  Callable[[SimpleNamespace, LongitudinalDecisionTelemetry | None], None],
  getattr(sp_longitudinal_planner, "publish_decision_layer_telemetry"),
)

class TestLongitudinalPlannerHysteresis(unittest.TestCase):
  cruise: tuple[float, float] = (0.0, 0.0)
  scc_vision: tuple[float, float] = (0.0, 0.0)
  scc_map: tuple[float, float] = (0.0, 0.0)
  speed_limit_assist: tuple[float, float] = (0.0, 0.0)
  osm_traffic_control: tuple[float, float] = (0.0, 0.0)

  def setUp(self):
    self.cruise = (30.0, 0.0)
    self.scc_vision = (25.0, -0.5)
    self.scc_map = (28.0, -0.2)
    self.speed_limit_assist = (22.0, 0.0)
    self.osm_traffic_control = (24.0, -0.3)

  def test_source_selection_no_hysteresis(self):
    # Standard selection: lowest target wins
    source, v_target, a_target = select_lowest_longitudinal_target(
      False, self.cruise, self.scc_vision, self.scc_map, self.speed_limit_assist, self.osm_traffic_control
    )
    self.assertEqual(source, LongitudinalPlanSource.speedLimitAssist)
    self.assertEqual(v_target, 22.0)

  def test_hysteresis_safety_switch(self):
    # Always switch to a SLOWER target (safety)
    source_prev = LongitudinalPlanSource.cruise
    v_target_prev = 30.0
    
    # Vision is 25.0 (slower than 30.0). Should switch immediately.
    source, v_target, a_target = select_lowest_longitudinal_target(
      False, self.cruise, self.scc_vision, self.scc_map, self.speed_limit_assist, self.osm_traffic_control,
      source_prev=source_prev, v_target_prev=v_target_prev
    )
    # Actually SLAY is 22.0, which is slower than 30.0.
    self.assertEqual(source, LongitudinalPlanSource.speedLimitAssist)
    self.assertEqual(v_target, 22.0)

  def test_hysteresis_stability_prevent_faster_switch(self):
    # Do NOT switch to a slightly faster target (stability)
    source_prev = LongitudinalPlanSource.speedLimitAssist
    v_target_prev = 22.0
    
    # New speed limit is 22.1 (only 0.1 faster). Should retain prev source.
    new_sla = (22.1, 0.0)
    source, v_target, a_target = select_lowest_longitudinal_target(
      False, self.cruise, self.scc_vision, self.scc_map, new_sla, self.osm_traffic_control,
      source_prev=source_prev, v_target_prev=v_target_prev
    )
    self.assertEqual(source, LongitudinalPlanSource.speedLimitAssist)
    self.assertEqual(v_target, 22.0)

  def test_hysteresis_stability_allow_faster_switch(self):
    # Allow switch to a SIGNIFICANTLY faster target
    source_prev = LongitudinalPlanSource.speedLimitAssist
    v_target_prev = 22.0
    
    # New speed limit is 22.5 (exactly 0.5 faster, threshold is 0.25). Should switch.
    new_sla = (22.5, 0.0)
    source, v_target, a_target = select_lowest_longitudinal_target(
      False, self.cruise, self.scc_vision, self.scc_map, new_sla, self.osm_traffic_control,
      source_prev=source_prev, v_target_prev=v_target_prev
    )
    # Wait, in the chain, vision (25.0) and map (28.0) and osm (24.0) are also considered.
    # If SLA is 22.5, it's still the winner among all.
    self.assertEqual(source, LongitudinalPlanSource.speedLimitAssist)
    self.assertEqual(v_target, 22.5)


class TestDecisionLayerTelemetryPublish(unittest.TestCase):
  def make_plan_sp(self):
    return SimpleNamespace(decisionLayer=SimpleNamespace())

  def test_publish_decision_layer_telemetry_values(self):
    plan_sp = self.make_plan_sp()
    telemetry = LongitudinalDecisionTelemetry(
      raw_source=DecisionSource.SPEED_LIMIT,
      raw_v_target=18.0,
      raw_a_target=-0.4,
      raw_should_stop=False,
      raw_active_reason="speed_limit_active",
      legacy_a_target=-0.1,
      legacy_should_stop=False,
      applied_a_target=-0.4,
      applied_should_stop=False,
      applied_reason="advisory_min_legacy",
      accel_delta=0.0,
    )

    publish_decision_layer_telemetry(plan_sp, telemetry)

    self.assertTrue(plan_sp.decisionLayer.enabled)
    self.assertEqual(plan_sp.decisionLayer.rawSource, "speed_limit")
    self.assertEqual(plan_sp.decisionLayer.rawReason, "speed_limit_active")
    self.assertEqual(plan_sp.decisionLayer.appliedReason, "advisory_min_legacy")
    self.assertEqual(plan_sp.decisionLayer.rawVTarget, 18.0)
    self.assertEqual(plan_sp.decisionLayer.rawATarget, -0.4)
    self.assertEqual(plan_sp.decisionLayer.appliedATarget, -0.4)
    self.assertEqual(plan_sp.decisionLayer.legacyATarget, -0.1)
    self.assertEqual(plan_sp.decisionLayer.accelDelta, 0.0)
    self.assertFalse(plan_sp.decisionLayer.rawShouldStop)
    self.assertFalse(plan_sp.decisionLayer.appliedShouldStop)
    self.assertFalse(plan_sp.decisionLayer.legacyShouldStop)

  def test_publish_decision_layer_telemetry_defaults_when_absent(self):
    plan_sp = self.make_plan_sp()

    publish_decision_layer_telemetry(plan_sp, None)

    self.assertFalse(plan_sp.decisionLayer.enabled)
    self.assertEqual(plan_sp.decisionLayer.rawSource, "")
    self.assertEqual(plan_sp.decisionLayer.rawReason, "")
    self.assertEqual(plan_sp.decisionLayer.appliedReason, "")
    self.assertEqual(plan_sp.decisionLayer.rawVTarget, 0.0)
    self.assertEqual(plan_sp.decisionLayer.rawATarget, 0.0)
    self.assertEqual(plan_sp.decisionLayer.appliedATarget, 0.0)
    self.assertEqual(plan_sp.decisionLayer.legacyATarget, 0.0)
    self.assertEqual(plan_sp.decisionLayer.accelDelta, 0.0)
    self.assertFalse(plan_sp.decisionLayer.rawShouldStop)
    self.assertFalse(plan_sp.decisionLayer.appliedShouldStop)
    self.assertFalse(plan_sp.decisionLayer.legacyShouldStop)


class TestSignalProviderCandidates(unittest.TestCase):
  def test_provider_envelope_builds_only_active_decision_candidates(self):
    candidates = build_sp_candidates_from_signal_providers((
      SignalProviderCandidate(
        source=DecisionSource.CRUISE,
        role=CandidateRole.DRIVER_INTENT,
        target=(25.0, 0.1),
        active=True,
        confidence=1.0,
        urgency=0.1,
        active_reason="driver_cruise_target",
      ),
      SignalProviderCandidate(
        source=DecisionSource.SCC_VISION,
        role=CandidateRole.ADVISORY_CAP,
        target=(20.0, -0.3),
        active=False,
        confidence=0.8,
        urgency=0.4,
        active_reason="confident_vision_curve",
      ),
    ))

    self.assertEqual([candidate.source for candidate in candidates], [DecisionSource.CRUISE])
    self.assertEqual(candidates[0].role, CandidateRole.DRIVER_INTENT)

  def test_sp_candidate_builder_keeps_provider_source_semantics(self):
    candidates = build_sp_longitudinal_candidates(
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

    self.assertEqual([
      candidate.source for candidate in candidates
    ], [
      DecisionSource.CRUISE,
      DecisionSource.SPEED_LIMIT,
      DecisionSource.SCC_VISION,
      DecisionSource.OSM_TRAFFIC_CONTROL,
    ])
    self.assertTrue(all(candidate.valid for candidate in candidates))

if __name__ == "__main__":
  unittest.main()
