import unittest
from types import SimpleNamespace
from typing import Callable, cast

from cereal import messaging
from cereal import custom
from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N
from openpilot.selfdrive.controls.lib.longitudinal_decision import CandidateRole, DecisionSource, LongitudinalCandidate, LongitudinalDecisionTelemetry
from openpilot.selfdrive.controls.lib.longitudinal_modes import (
  DecCompatibilityState,
  LongitudinalActuationType,
  LongitudinalMode,
  LongitudinalModeResolution,
  ResolvedLongitudinalImplementation,
)
from openpilot.selfdrive.controls.lib.longitudinal_stacks.planner_seed import PlannerSeedCandidate
from openpilot.selfdrive.controls.lib.longitudinal_stacks.custom_v2 import CustomV2Scene, ONE_PEDAL_MODE_CREEP
from openpilot.selfdrive.controls.lib.longitudinal_stacks.interface import LongitudinalStackOutput
from openpilot.selfdrive.controls.lib.longitudinal_stacks.policy import CUSTOM_V2_DEBUG_DISABLE_JERK_LIMIT, CUSTOM_V2_DEBUG_INTENT
from openpilot.selfdrive.controls.lib.longitudinal_stacks.selector import CUSTOM_V2, SUNNYPILOT_CURRENT, StackResolution
from openpilot.selfdrive.controls.lib.longitudinal_planner import build_moving_lead_seed_candidates
from openpilot.sunnypilot.selfdrive.controls.lib import longitudinal_planner as sp_longitudinal_planner
from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import (
  SignalProviderCandidate,
  LongitudinalPlannerSP,
  LongitudinalPlanSource,
  LongitudinalModeStatus,
  StackId,
  build_sp_candidates_from_signal_providers,
  build_sp_longitudinal_candidates,
  publish_longitudinal_mode_telemetry,
  publish_stack_telemetry,
  select_lowest_longitudinal_target,
  should_block_lead_speedup_from_context,
)

publish_decision_layer_telemetry = cast(
  Callable[[SimpleNamespace, LongitudinalDecisionTelemetry | None], None],
  getattr(sp_longitudinal_planner, "publish_decision_layer_telemetry"),
)


class FakeEvents:
  def __init__(self):
    self.names = []

  def add(self, event):
    self.names.append(event)

  def clear(self):
    self.names.clear()


class FakeE2EAlertsHelper:
  def __init__(self):
    self.green_light_alert = True
    self.lead_depart_alert = True
    self.update_count = 0

  def update(self, sm, events_sp):
    self.update_count += 1


class FakeParams:
  def __init__(self, values=None):
    self.values = {} if values is None else dict(values)

  def get(self, key, *args, **kwargs):
    if kwargs.get("return_default", False) and key == "LongitudinalMode":
      return self.values.get(key, "0")
    return self.values.get(key)

  def get_bool(self, key):
    value = self.values.get(key)
    if isinstance(value, bool):
      return value
    return str(value).lower() in ("1", "true", "yes")


class PoisonModel:
  def __getattr__(self, name):
    raise AssertionError(f"ACC mode must not read modelV2.{name}")


class FakeStackSCC:
  def __init__(self, vision=(255.0, 0.0, False), map_target=(255.0, 0.0, False)):
    self.vision = SimpleNamespace(
      output_v_target=vision[0], output_a_target=vision[1], is_active=vision[2], state=0,
      current_lat_acc=0.0, max_pred_lat_acc=0.0, is_enabled=vision[2],
    )
    self.map = SimpleNamespace(
      output_v_target=map_target[0], output_a_target=map_target[1], is_active=map_target[2], state=0,
      is_enabled=map_target[2],
    )
    self.update_count = 0

  def update(self, *args):
    self.update_count += 1


class FakeStackResolver:
  def __init__(self, speed_limit=0.0, speed_limit_final_last=0.0, distance=0.0):
    self.speed_limit = speed_limit
    self.speed_limit_last = speed_limit
    self.speed_limit_final = speed_limit
    self.speed_limit_final_last = speed_limit_final_last
    self.speed_limit_valid = speed_limit > 0.0
    self.speed_limit_last_valid = speed_limit_final_last > 0.0
    self.speed_limit_offset = 0.0
    self.distance = distance
    self.source = 0
    self.update_count = 0

  def update(self, *args, **kwargs):
    self.update_count += 1


class FakeStackSLA:
  def __init__(self, target=(255.0, 0.0), active=False):
    self.output_v_target, self.output_a_target = target
    self.is_active = active
    self.is_enabled = active
    self.auto_enabled = active
    self.state = 0
    self.update_count = 0

  def update(self, *args, **kwargs):
    self.update_count += 1


class FakeStackOsmPrior:
  def __init__(self, target=(255.0, 0.0), active=False):
    self.output_v_target, self.output_a_target = target
    self.active = active
    self.update_count = 0

  def update(self, *args):
    self.update_count += 1


def make_target_sm():
  return {
    "carState": SimpleNamespace(vCruiseCluster=72.0, gasPressed=False, brakePressed=False),
    "carControl": SimpleNamespace(enabled=True, cruiseControl=SimpleNamespace(override=False)),
    "radarState": SimpleNamespace(leadOne=SimpleNamespace(status=False, dRel=100.0, vRel=0.0, yRel=0.0)),
  }


def make_target_planner(resolved_stack: str):
  planner = LongitudinalPlannerSP.__new__(LongitudinalPlannerSP)
  planner.longitudinal_stack_resolution = StackResolution(
    requested_stack=resolved_stack,
    resolved_stack=resolved_stack,
    available_stacks=(SUNNYPILOT_CURRENT, CUSTOM_V2),
  )
  planner.events_sp = FakeEvents()
  planner.source = LongitudinalPlanSource.cruise
  planner._speed_limit_handoff_active = False
  planner._speed_limit_active_prev = False
  return planner


class TestStackAwareTargetSelection(unittest.TestCase):
  def test_acc_mode_does_not_build_or_update_signal_provider_candidates(self):
    planner = make_target_planner(SUNNYPILOT_CURRENT)
    planner.longitudinal_mode_resolution = LongitudinalModeResolution(
      requested_mode=LongitudinalMode.ACC,
      resolved_implementation=ResolvedLongitudinalImplementation.HARDWARE_ACC,
      actuation_type=LongitudinalActuationType.DIRECT,
    )
    planner.scc = FakeStackSCC(vision=(8.0, -1.0, True), map_target=(9.0, -0.8, True))
    planner.resolver = FakeStackResolver(speed_limit=8.0, speed_limit_final_last=8.0)
    planner.sla = FakeStackSLA(target=(8.0, -1.0), active=True)
    planner.osm_traffic_control_prior = FakeStackOsmPrior(target=(7.0, -1.0), active=True)

    v_target, a_target = LongitudinalPlannerSP.update_targets(planner, make_target_sm(), 12.0, 0.2, 20.0)

    self.assertEqual(planner.source, LongitudinalPlanSource.cruise)
    self.assertEqual(v_target, 20.0)
    self.assertEqual(a_target, 0.2)
    self.assertEqual(planner.scc.update_count, 0)
    self.assertEqual(planner.resolver.update_count, 0)
    self.assertEqual(planner.sla.update_count, 0)
    self.assertEqual(planner.osm_traffic_control_prior.update_count, 0)
    self.assertEqual([candidate.source for candidate in planner.decision_candidates_sp], [DecisionSource.CRUISE])

  def test_acc_mode_update_does_not_touch_e2e_alert_model_inputs(self):
    planner = LongitudinalPlannerSP.__new__(LongitudinalPlannerSP)
    planner.params = FakeParams({"LongitudinalMode": str(int(LongitudinalMode.ACC))})
    planner.CP = SimpleNamespace(openpilotLongitudinalControl=True, radarUnavailable=False)
    planner.events_sp = FakeEvents()
    planner.e2e_alerts_helper = SimpleNamespace(green_light_alert=True, lead_depart_alert=True)

    LongitudinalPlannerSP.update(planner, {"modelV2": PoisonModel()})

    self.assertFalse(planner.e2e_alerts_helper.green_light_alert)
    self.assertFalse(planner.e2e_alerts_helper.lead_depart_alert)

  def test_scc_e2e_resolution_updates_e2e_alerts_after_evidence_promotion(self):
    planner = LongitudinalPlannerSP.__new__(LongitudinalPlannerSP)
    planner.longitudinal_mode_resolution = LongitudinalModeResolution(
      requested_mode=LongitudinalMode.SCC,
      resolved_implementation=ResolvedLongitudinalImplementation.SCC_E2E,
      actuation_type=LongitudinalActuationType.DIRECT,
    )
    planner.e2e_alerts_helper = FakeE2EAlertsHelper()
    planner.events_sp = FakeEvents()

    LongitudinalPlannerSP._update_e2e_alerts_for_mode(planner, {"modelV2": PoisonModel()})

    self.assertEqual(planner.e2e_alerts_helper.update_count, 1)

  def test_scc_acc_resolution_resets_e2e_alerts_without_model_inputs(self):
    planner = LongitudinalPlannerSP.__new__(LongitudinalPlannerSP)
    planner.longitudinal_mode_resolution = LongitudinalModeResolution(
      requested_mode=LongitudinalMode.SCC,
      resolved_implementation=ResolvedLongitudinalImplementation.SCC_ACC,
      actuation_type=LongitudinalActuationType.DIRECT,
    )
    planner.e2e_alerts_helper = FakeE2EAlertsHelper()
    planner.events_sp = FakeEvents()

    LongitudinalPlannerSP._update_e2e_alerts_for_mode(planner, {"modelV2": PoisonModel()})

    self.assertEqual(planner.e2e_alerts_helper.update_count, 0)
    self.assertFalse(planner.e2e_alerts_helper.green_light_alert)
    self.assertFalse(planner.e2e_alerts_helper.lead_depart_alert)

  def test_sunnypilot_current_uses_custom_target_providers(self):
    planner = make_target_planner(SUNNYPILOT_CURRENT)
    planner.scc = FakeStackSCC(vision=(8.0, -1.0, True))
    planner.resolver = FakeStackResolver(speed_limit=8.0, speed_limit_final_last=8.0)
    planner.sla = FakeStackSLA(target=(8.0, -1.0), active=True)
    planner.osm_traffic_control_prior = FakeStackOsmPrior(target=(7.0, -1.0), active=True)
    planner.sunnypilot_current_scc = FakeStackSCC(vision=(18.0, -0.1, True))
    planner.sunnypilot_current_resolver = FakeStackResolver(speed_limit=18.0, speed_limit_final_last=18.0)
    planner.sunnypilot_current_sla = FakeStackSLA(target=(18.0, -0.1), active=True)

    v_target, a_target = LongitudinalPlannerSP.update_targets(planner, make_target_sm(), 12.0, 0.2, 20.0)

    self.assertEqual(planner.source, LongitudinalPlanSource.osmTrafficControl)
    self.assertEqual(v_target, 7.0)
    self.assertEqual(a_target, -1.0)
    self.assertEqual(planner.scc.update_count, 1)
    self.assertEqual(planner.resolver.update_count, 1)
    self.assertEqual(planner.sla.update_count, 1)
    self.assertEqual(planner.osm_traffic_control_prior.update_count, 1)
    self.assertEqual(planner.sunnypilot_current_scc.update_count, 0)
    self.assertEqual(planner.sunnypilot_current_resolver.update_count, 0)
    self.assertEqual(planner.sunnypilot_current_sla.update_count, 0)

  def test_sunnypilot_current_ignores_baseline_duplicate_providers(self):
    planner = make_target_planner(SUNNYPILOT_CURRENT)
    planner.scc = FakeStackSCC()
    planner.resolver = FakeStackResolver()
    planner.sla = FakeStackSLA()
    planner.osm_traffic_control_prior = FakeStackOsmPrior()
    planner.sunnypilot_current_scc = FakeStackSCC(vision=(8.0, -1.0, True))
    planner.sunnypilot_current_resolver = FakeStackResolver(speed_limit=15.0, speed_limit_final_last=15.0)
    planner.sunnypilot_current_sla = FakeStackSLA(target=(15.0, -0.2), active=True)

    v_target, a_target = LongitudinalPlannerSP.update_targets(planner, make_target_sm(), 12.0, 0.2, 20.0)

    self.assertEqual(planner.source, LongitudinalPlanSource.cruise)
    self.assertEqual(v_target, 20.0)
    self.assertEqual(a_target, 0.2)
    self.assertEqual(planner.scc.update_count, 1)
    self.assertEqual(planner.sla.update_count, 1)
    self.assertEqual(planner.sunnypilot_current_scc.update_count, 0)
    self.assertEqual(planner.sunnypilot_current_sla.update_count, 0)

  def test_custom_stack_uses_custom_target_providers(self):
    planner = make_target_planner(CUSTOM_V2)
    planner.scc = FakeStackSCC(vision=(8.0, -1.0, True))
    planner.resolver = FakeStackResolver()
    planner.sla = FakeStackSLA()
    planner.osm_traffic_control_prior = FakeStackOsmPrior()
    planner.sunnypilot_current_scc = FakeStackSCC()
    planner.sunnypilot_current_resolver = FakeStackResolver()
    planner.sunnypilot_current_sla = FakeStackSLA()

    v_target, a_target = LongitudinalPlannerSP.update_targets(planner, make_target_sm(), 12.0, 0.2, 20.0)

    self.assertEqual(planner.source, LongitudinalPlanSource.sccVision)
    self.assertEqual(v_target, 8.0)
    self.assertEqual(a_target, -1.0)
    self.assertEqual(planner.scc.update_count, 1)
    self.assertEqual(planner.sunnypilot_current_scc.update_count, 0)

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


class TestLeadContextSpeedupGuard(unittest.TestCase):
  def test_context_guard_uses_close_closing_shadow_or_real_lead(self):
    context = SimpleNamespace(states=(
      SimpleNamespace(authority="suppress_only", d_rel=12.0, v_rel=-1.0, path_y_rel=0.0),
    ))

    self.assertTrue(should_block_lead_speedup_from_context(context, 8.0, False, False))

  def test_context_guard_ignores_released_false_positive(self):
    context = SimpleNamespace(states=(
      SimpleNamespace(authority="none", d_rel=12.0, v_rel=-1.0, path_y_rel=0.0),
    ))

    self.assertFalse(should_block_lead_speedup_from_context(context, 8.0, False, False))


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


class TestStackTelemetryPublish(unittest.TestCase):
  def make_plan_sp(self):
    return SimpleNamespace(stack=SimpleNamespace())

  def test_publish_stack_telemetry_defaults_to_sunnypilot_actuation(self):
    plan_sp = self.make_plan_sp()
    resolution = StackResolution(
      requested_stack="sunnypilot-current",
      resolved_stack="sunnypilot-current",
      available_stacks=("sunnypilot-current",),
    )

    publish_stack_telemetry(plan_sp, resolution, "sunnypilot-current", -0.25)

    self.assertEqual(plan_sp.stack.requestedStack, StackId.sunnypilotCurrent)
    self.assertEqual(plan_sp.stack.resolvedStack, StackId.sunnypilotCurrent)
    self.assertEqual(plan_sp.stack.actuatedStack, StackId.sunnypilotCurrent)
    self.assertEqual(plan_sp.stack.customVersion, "")
    self.assertFalse(plan_sp.stack.faultLatched)
    self.assertEqual(plan_sp.stack.faultReason, "")
    self.assertEqual(plan_sp.stack.actuatedATarget, -0.25)
    self.assertEqual(plan_sp.stack.selectedIntent, "")
    self.assertEqual(plan_sp.stack.selectedReason, "")
    self.assertEqual(plan_sp.stack.rejectedIntents, [])
    self.assertEqual(plan_sp.stack.rejectedReasons, [])
    self.assertEqual(plan_sp.stack.seedContext, "")
    self.assertEqual(plan_sp.stack.seedCandidate, "")

  def test_publish_stack_telemetry_reports_custom_v2_policy_debug(self):
    plan_sp = self.make_plan_sp()
    resolution = StackResolution(
      requested_stack="custom-2.0",
      resolved_stack="custom-2.0",
      available_stacks=("sunnypilot-current", "custom-2.0"),
      custom_version="2.0",
    )

    publish_stack_telemetry(
      plan_sp, resolution, "custom-2.0", 0.2,
      fault_latched=True, fault_reason="a_target_above_limits",
      selected_intent="launch", selected_reason="confirmed_lead_pullaway",
      rejected=(("stop_approach", "lead_pullaway_release"), ("speed_policy", "no_speed_reduction_needed")),
      seed_context="planner", seed_candidate="creep_pullaway_launch",
    )

    self.assertEqual(plan_sp.stack.resolvedStack, StackId.customV2)
    self.assertEqual(plan_sp.stack.actuatedStack, StackId.customV2)
    self.assertEqual(plan_sp.stack.customVersion, "2.0")
    self.assertTrue(plan_sp.stack.faultLatched)
    self.assertEqual(plan_sp.stack.faultReason, "a_target_above_limits")
    self.assertEqual(plan_sp.stack.selectedIntent, "launch")
    self.assertEqual(plan_sp.stack.selectedReason, "confirmed_lead_pullaway")
    self.assertEqual(plan_sp.stack.rejectedIntents, ["stop_approach", "speed_policy"])
    self.assertEqual(plan_sp.stack.rejectedReasons, ["lead_pullaway_release", "no_speed_reduction_needed"])
    self.assertEqual(plan_sp.stack.seedContext, "planner")
    self.assertEqual(plan_sp.stack.seedCandidate, "creep_pullaway_launch")

  def test_capnp_stack_schema_fields_exist(self):
    msg = messaging.new_message("longitudinalPlanSP")
    msg.longitudinalPlanSP.stack.requestedStack = StackId.sunnypilotCurrent
    msg.longitudinalPlanSP.stack.resolvedStack = StackId.sunnypilotCurrent
    msg.longitudinalPlanSP.stack.actuatedStack = StackId.sunnypilotCurrent
    msg.longitudinalPlanSP.stack.faultReason = ""
    msg.longitudinalPlanSP.stack.selectedIntent = "launch"
    msg.longitudinalPlanSP.stack.selectedReason = "no_lead_stop_clear"
    msg.longitudinalPlanSP.stack.rejectedIntents = ["stop_approach"]
    msg.longitudinalPlanSP.stack.rejectedReasons = ["lead_pullaway_release"]
    msg.longitudinalPlanSP.stack.seedContext = "planner"
    msg.longitudinalPlanSP.stack.seedCandidate = "creep_pullaway_launch"

    self.assertEqual(msg.longitudinalPlanSP.stack.requestedStack, StackId.sunnypilotCurrent)
    self.assertEqual(msg.longitudinalPlanSP.stack.faultReason, "")
    self.assertEqual(msg.longitudinalPlanSP.stack.selectedIntent, "launch")
    self.assertEqual(msg.longitudinalPlanSP.stack.rejectedIntents[0], "stop_approach")
    self.assertEqual(msg.longitudinalPlanSP.stack.seedContext, "planner")
    self.assertEqual(msg.longitudinalPlanSP.stack.seedCandidate, "creep_pullaway_launch")


class TestLongitudinalModeTelemetryPublish(unittest.TestCase):
  def make_plan_sp(self):
    return SimpleNamespace(longitudinalMode=SimpleNamespace())

  def test_publish_longitudinal_mode_telemetry_reports_resolution(self):
    plan_sp = self.make_plan_sp()
    resolution = LongitudinalModeResolution(
      requested_mode=LongitudinalMode.SCC,
      resolved_implementation=ResolvedLongitudinalImplementation.SCC_E2E,
      actuation_type=LongitudinalActuationType.DIRECT,
      restriction_status=("speed_limit_cap", "curve_cap"),
      compatibility_alias_state=DecCompatibilityState.BLENDED,
    )

    publish_longitudinal_mode_telemetry(plan_sp, resolution)

    self.assertEqual(plan_sp.longitudinalMode.requestedMode, LongitudinalModeStatus.Mode.scc)
    self.assertEqual(plan_sp.longitudinalMode.resolvedImplementation, LongitudinalModeStatus.Implementation.sccE2e)
    self.assertEqual(plan_sp.longitudinalMode.actuationType, LongitudinalModeStatus.ActuationType.direct)
    self.assertEqual(plan_sp.longitudinalMode.restrictionStatus, ["speed_limit_cap", "curve_cap"])
    self.assertEqual(plan_sp.longitudinalMode.compatibilityAliasState, LongitudinalModeStatus.CompatibilityAliasState.blended)

  def test_capnp_longitudinal_mode_schema_fields_exist(self):
    msg = messaging.new_message("longitudinalPlanSP")
    msg.longitudinalPlanSP.longitudinalMode.requestedMode = LongitudinalModeStatus.Mode.acc
    msg.longitudinalPlanSP.longitudinalMode.resolvedImplementation = LongitudinalModeStatus.Implementation.hardwareAcc
    msg.longitudinalPlanSP.longitudinalMode.actuationType = LongitudinalModeStatus.ActuationType.direct
    msg.longitudinalPlanSP.longitudinalMode.restrictionStatus = ["none"]
    msg.longitudinalPlanSP.longitudinalMode.unsupportedReason = ""
    msg.longitudinalPlanSP.longitudinalMode.compatibilityAliasState = LongitudinalModeStatus.CompatibilityAliasState.acc

    self.assertEqual(msg.longitudinalPlanSP.longitudinalMode.requestedMode, LongitudinalModeStatus.Mode.acc)
    self.assertEqual(msg.longitudinalPlanSP.longitudinalMode.restrictionStatus[0], "none")


class TestLongitudinalStackSelectionIntegration(unittest.TestCase):
  def make_planner(self, resolved_stack="custom-2.0"):
    planner = LongitudinalPlannerSP.__new__(LongitudinalPlannerSP)
    planner.output_a_target = -0.1
    planner.output_should_stop = False
    planner.allow_throttle = True
    planner.fcw = False
    planner.v_desired_trajectory = tuple(10.0 for _ in range(CONTROL_N))
    planner.a_desired_trajectory = tuple(-0.1 for _ in range(CONTROL_N))
    planner.j_desired_trajectory = tuple(0.0 for _ in range(CONTROL_N))
    planner.source = "cruise"
    planner.mpc = SimpleNamespace(source="cruise")
    planner.events_sp = FakeEvents()
    planner.longitudinal_stack_resolution = StackResolution(
      requested_stack=resolved_stack,
      resolved_stack=resolved_stack,
      available_stacks=("sunnypilot-current", resolved_stack),
      custom_version="2.0" if resolved_stack == "custom-2.0" else "",
    )
    planner.custom_longitudinal_stack = None
    planner.planner_seed_candidates = []
    planner.longitudinal_stack_actuated_stack = "sunnypilot-current"
    planner.longitudinal_stack_fault_latched = False
    planner.longitudinal_stack_fault_reason = ""
    planner.longitudinal_stack_selected_intent = ""
    planner.longitudinal_stack_selected_reason = ""
    planner.longitudinal_stack_rejected = ()
    planner.custom_v2_fault_latched = False
    planner.custom_v2_fault_reason = ""
    return planner

  def make_sm(self, enabled=True):
    return {"selfdriveState": SimpleNamespace(enabled=enabled)}

  def make_output(self, a_target, should_stop=False, has_lead=False, debug=None, speeds=None, accels=None, jerks=None):
    return LongitudinalStackOutput(
      a_target=a_target,
      should_stop=should_stop,
      has_lead=has_lead,
      source="custom",
      allow_throttle=True,
      allow_brake=True,
      speeds=tuple(10.0 for _ in range(CONTROL_N)) if speeds is None else speeds,
      accels=tuple(a_target for _ in range(CONTROL_N)) if accels is None else accels,
      jerks=tuple(0.0 for _ in range(CONTROL_N)) if jerks is None else jerks,
      debug={} if debug is None else debug,
    )

  def make_candidate(self, source, role, v_target, a_target, reason, should_stop=False):
    return LongitudinalCandidate(
      source=source,
      role=role,
      v_target=v_target,
      a_target=a_target,
      confidence=1.0,
      urgency=0.8,
      active_reason=reason,
      should_stop=should_stop,
    )

  def test_custom_v2_selection_actuates_fail_closed_stack_without_shadow(self):
    planner = self.make_planner()
    planner.custom_v2_scene = CustomV2Scene(v_ego=0.2, v_cruise=5.0, model_stop_distance=30.0, model_desired_accel=0.0)

    planner.apply_longitudinal_stack_selection(self.make_sm(), has_lead=False, accel_limits=(-2.0, 2.0))

    self.assertEqual(planner.output_a_target, 1.35)
    self.assertEqual(planner.longitudinal_stack_actuated_stack, "custom-2.0")
    self.assertFalse(planner.longitudinal_stack_fault_latched)
    self.assertEqual(planner.longitudinal_stack_selected_intent, "launch")
    self.assertEqual(planner.events_sp.names, [])

  def test_custom_v2_uses_planner_seed_candidate_without_public_legacy_stack(self):
    planner = self.make_planner()
    planner.planner_seed_candidates = [PlannerSeedCandidate("internal_stop", self.make_output(-0.7))]
    planner.custom_v2_scene = CustomV2Scene(v_ego=10.0, v_cruise=10.0)

    planner.apply_longitudinal_stack_selection(self.make_sm(), has_lead=True, accel_limits=(-2.0, 2.0))

    self.assertEqual(planner.output_a_target, -0.7)
    self.assertEqual(planner.longitudinal_stack_actuated_stack, "custom-2.0")
    self.assertFalse(planner.longitudinal_stack_fault_latched)

  def test_custom_v2_planner_seed_preserves_internal_lead_stop_behavior(self):
    planner = self.make_planner()
    speeds = tuple(float(idx) for idx in range(CONTROL_N))
    accels = tuple(-0.7 + 0.01 * idx for idx in range(CONTROL_N))
    jerks = tuple(-0.1 for _ in range(CONTROL_N))
    planner.planner_seed_candidates = [PlannerSeedCandidate(
      "internal_stop",
      self.make_output(
        -0.7, should_stop=True, has_lead=True, speeds=speeds, accels=accels, jerks=jerks,
        debug={"planner_seed_candidate_reason": "stopped_lead_stop_gap_guard"},
      ),
    )]
    planner.custom_v2_scene = CustomV2Scene(
      v_ego=0.3, v_cruise=6.0, has_lead=True, lead_v=0.2, lead_confirmed_pullaway=True,
      stop_threat=True, model_should_stop=True, model_stop_distance=5.0, model_desired_accel=-1.0,
    )

    planner.apply_longitudinal_stack_selection(self.make_sm(), has_lead=True, accel_limits=(-2.0, 2.0))

    self.assertEqual(planner.output_a_target, -0.7)
    self.assertTrue(planner.output_should_stop)
    self.assertEqual(tuple(planner.v_desired_trajectory), speeds)
    self.assertEqual(tuple(planner.a_desired_trajectory), accels)
    self.assertEqual(tuple(planner.j_desired_trajectory), jerks)
    self.assertEqual(planner.longitudinal_stack_actuated_stack, "custom-2.0")
    self.assertEqual(planner.longitudinal_stack_selected_intent, "lead_follow")
    self.assertEqual(planner.longitudinal_stack_selected_reason, "stopped_lead_stop_gap_guard")

  def test_equal_baseline_physical_candidate_suppresses_advisory_caps(self):
    planner = self.make_planner()
    planner.output_a_target = -0.4
    planner.a_desired_trajectory = tuple(-0.4 for _ in range(CONTROL_N))
    planner.decision_candidates_sp = [
      self.make_candidate(DecisionSource.CRUISE, CandidateRole.DRIVER_INTENT, 20.0, 0.0, "driver_cruise_target"),
    ]
    planner.longitudinal_decision_candidates = [
      self.make_candidate(DecisionSource.CRUISE, CandidateRole.DRIVER_INTENT, 20.0, 0.0, "driver_cruise_target"),
      self.make_candidate(DecisionSource.LEAD_MPC, CandidateRole.PHYSICAL_HAZARD, 20.0, -0.4, "confirmed_radar_lead"),
    ]
    planner.custom_v2_scene = CustomV2Scene(
      v_ego=15.0,
      v_cruise=20.0,
      has_lead=True,
      speed_limit_active=True,
      speed_limit_v_target=10.0,
      speed_limit_a_target=-1.0,
      curve_active=True,
      curve_a_target=-0.8,
      map_caution_active=True,
      map_caution_confirmed=True,
      map_caution_a_target=-0.9,
    )

    planner.apply_longitudinal_stack_selection(self.make_sm(), has_lead=True, accel_limits=(-2.0, 2.0))

    self.assertEqual(planner.output_a_target, -0.4)
    self.assertEqual(planner.longitudinal_stack_selected_intent, "lead_follow")
    self.assertEqual(planner.longitudinal_stack_selected_reason, "confirmed_radar_lead")
    self.assertIn(("speed_policy", "physical_hazard_active"), planner.longitudinal_stack_rejected)
    self.assertIn(("curve_policy", "physical_hazard_active"), planner.longitudinal_stack_rejected)
    self.assertIn(("map_caution", "physical_hazard_active"), planner.longitudinal_stack_rejected)

  def test_stop_launch_progress_relaxation_applies_through_custom_v2_logic(self):
    planner = self.make_planner()
    planner.output_a_target = 0.0
    planner.decision_candidates_sp = [
      self.make_candidate(DecisionSource.CRUISE, CandidateRole.DRIVER_INTENT, 8.0, 0.0, "driver_cruise_target"),
    ]
    planner.custom_v2_scene = CustomV2Scene(v_ego=0.2, v_cruise=8.0, model_stop_distance=30.0, model_desired_accel=0.0)

    planner.apply_longitudinal_stack_selection(self.make_sm(), has_lead=False, accel_limits=(-2.0, 2.0))

    self.assertEqual(planner.output_a_target, 1.35)
    self.assertEqual(planner.longitudinal_stack_selected_intent, "launch")
    self.assertEqual(planner.longitudinal_stack_selected_reason, "no_lead_stop_clear")

  def test_advisory_cap_wins_over_scene_derived_progress_relaxation(self):
    planner = self.make_planner()
    planner.output_a_target = 0.0
    planner.decision_candidates_sp = [
      self.make_candidate(DecisionSource.CRUISE, CandidateRole.DRIVER_INTENT, 8.0, 0.0, "driver_cruise_target"),
    ]
    planner.custom_v2_scene = CustomV2Scene(
      v_ego=2.0,
      v_cruise=8.0,
      accel_coast=-0.25,
      model_stop_distance=30.0,
      model_desired_accel=0.0,
      speed_limit_active=True,
      speed_limit_v_target=1.0,
      speed_limit_a_target=-0.4,
    )

    planner.apply_longitudinal_stack_selection(self.make_sm(), has_lead=False, accel_limits=(-2.0, 2.0))

    self.assertEqual(planner.output_a_target, -0.25)
    self.assertEqual(planner.longitudinal_stack_selected_intent, "speed_policy")
    self.assertEqual(planner.longitudinal_stack_selected_reason, "coast_biased_speed_reduction")

  def test_physical_hazard_blocks_stop_launch_progress_relaxation(self):
    planner = self.make_planner()
    planner.output_a_target = -0.6
    planner.decision_candidates_sp = [
      self.make_candidate(DecisionSource.CRUISE, CandidateRole.DRIVER_INTENT, 8.0, 0.0, "driver_cruise_target"),
    ]
    planner.longitudinal_decision_candidates = [
      self.make_candidate(DecisionSource.E2E_STOP, CandidateRole.PHYSICAL_HAZARD, 8.0, -0.6, "model_stop_or_slowdown", should_stop=True),
    ]
    planner.custom_v2_scene = CustomV2Scene(v_ego=0.2, v_cruise=8.0, model_stop_distance=30.0, model_desired_accel=0.0)

    planner.apply_longitudinal_stack_selection(self.make_sm(), has_lead=False, accel_limits=(-2.0, 2.0))

    self.assertEqual(planner.output_a_target, -0.6)
    self.assertTrue(planner.output_should_stop)
    self.assertEqual(planner.longitudinal_stack_selected_intent, "stop_approach")
    self.assertIn(("launch", "physical_hazard_active"), planner.longitudinal_stack_rejected)

  def test_stop_launch_safety_cap_is_not_generic_comfort_relaxation(self):
    planner = self.make_planner()
    planner.output_a_target = 0.4
    planner.decision_candidates_sp = [
      self.make_candidate(DecisionSource.CRUISE, CandidateRole.DRIVER_INTENT, 8.0, 0.4, "driver_cruise_target"),
    ]
    planner.custom_v2_scene = CustomV2Scene(v_ego=2.0, v_cruise=8.0, force_slow_decel=True)

    planner.apply_longitudinal_stack_selection(self.make_sm(), has_lead=False, accel_limits=(-2.0, 2.0))

    self.assertEqual(planner.output_a_target, -0.2)
    self.assertTrue(planner.output_should_stop)
    self.assertEqual(planner.longitudinal_stack_selected_intent, "safety_cap")
    self.assertEqual(planner.longitudinal_stack_selected_reason, "force_slow_decel")
    force_slow = next(candidate for candidate in planner.longitudinal_decision.candidates if candidate.active_reason == "force_slow_decel")
    self.assertEqual(force_slow.role, CandidateRole.PHYSICAL_HAZARD)
    self.assertEqual(force_slow.debug[CUSTOM_V2_DEBUG_INTENT], "safety_cap")
    self.assertTrue(force_slow.debug[CUSTOM_V2_DEBUG_DISABLE_JERK_LIMIT])

  def test_one_pedal_replaces_driver_intent_and_preserves_physical_braking(self):
    planner = self.make_planner()
    planner.output_a_target = -0.7
    planner.decision_candidates_sp = [
      self.make_candidate(DecisionSource.CRUISE, CandidateRole.DRIVER_INTENT, 20.0, 0.4, "driver_cruise_target"),
    ]
    planner.longitudinal_decision_candidates = [
      self.make_candidate(DecisionSource.LEAD_MPC, CandidateRole.PHYSICAL_HAZARD, 20.0, -0.7, "confirmed_radar_lead"),
    ]
    planner.custom_v2_scene = CustomV2Scene(v_ego=14.0, v_cruise=20.0, has_lead=True, one_pedal_mode=ONE_PEDAL_MODE_CREEP)

    planner.apply_longitudinal_stack_selection(self.make_sm(), has_lead=True, accel_limits=(-2.0, 2.0))

    decision = planner.longitudinal_decision
    driver = next(candidate for candidate in decision.candidates if candidate.role == CandidateRole.DRIVER_INTENT)
    self.assertEqual(sum(candidate.role == CandidateRole.DRIVER_INTENT for candidate in decision.candidates), 1)
    self.assertEqual(driver.debug[CUSTOM_V2_DEBUG_INTENT], "one_pedal")
    self.assertEqual(decision.winner, DecisionSource.LEAD_MPC)
    self.assertEqual(decision.a_target, -0.7)
    self.assertEqual(planner.output_a_target, -0.7)
    self.assertEqual(planner.longitudinal_stack_selected_intent, "lead_follow")
    self.assertIn(("one_pedal", "physical_hazard_active"), planner.longitudinal_stack_rejected)

  def test_source_stability_hold_applies_preserved_custom_v2_candidate_values(self):
    planner = self.make_planner()
    planner.output_a_target = 0.0
    planner.decision_candidates_sp = [
      self.make_candidate(DecisionSource.CRUISE, CandidateRole.DRIVER_INTENT, 5.0, 0.0, "driver_cruise_target"),
    ]
    planner.custom_v2_scene = CustomV2Scene(
      v_ego=2.0,
      v_cruise=5.0,
      accel_coast=-0.25,
      model_stop_distance=10.0,
      model_desired_accel=0.0,
      speed_limit_active=True,
      speed_limit_v_target=1.0,
      speed_limit_a_target=-0.4,
    )
    planner.apply_longitudinal_stack_selection(self.make_sm(), has_lead=False, accel_limits=(-2.0, 2.0))
    self.assertEqual(planner.output_a_target, -0.25)
    self.assertEqual(planner.longitudinal_stack_selected_intent, "speed_policy")

    planner.output_a_target = 0.2
    planner.a_desired_trajectory = tuple(0.2 for _ in range(CONTROL_N))
    planner.custom_v2_scene = CustomV2Scene(
      v_ego=2.0,
      v_cruise=5.0,
      model_stop_distance=10.0,
      model_desired_accel=0.0,
    )
    planner.apply_longitudinal_stack_selection(self.make_sm(), has_lead=False, accel_limits=(-2.0, 2.0))

    self.assertEqual(planner.output_a_target, -0.25)
    self.assertEqual(planner.longitudinal_stack_selected_intent, "speed_policy")
    self.assertEqual(planner.longitudinal_stack_selected_reason, "coast_biased_speed_reduction")
    self.assertIn(("driver_cruise", "source_stability_hold"), planner.longitudinal_stack_rejected)

  def test_one_pedal_no_lead_standstill_does_not_create_autonomous_creep(self):
    planner = self.make_planner()
    planner.output_a_target = 0.0
    planner.decision_candidates_sp = [
      self.make_candidate(DecisionSource.CRUISE, CandidateRole.DRIVER_INTENT, 8.0, 0.4, "driver_cruise_target"),
    ]
    planner.custom_v2_scene = CustomV2Scene(
      v_ego=0.0,
      v_cruise=8.0,
      model_stop_distance=10.0,
      model_desired_accel=0.0,
      one_pedal_mode=ONE_PEDAL_MODE_CREEP,
    )

    planner.apply_longitudinal_stack_selection(self.make_sm(), has_lead=False, accel_limits=(-2.0, 2.0))

    self.assertEqual(planner.output_a_target, 0.0)
    self.assertEqual(planner.longitudinal_stack_selected_intent, "one_pedal")
    self.assertEqual(planner.longitudinal_stack_selected_reason, "lift_off_coast")
    self.assertIn(("one_pedal", "terminal_creep_not_authorized"), planner.longitudinal_stack_rejected)

  def test_seed_context_and_candidate_publish_from_selected_seed(self):
    planner = self.make_planner()
    planner.planner_seed_candidates = [PlannerSeedCandidate(
      "lead_crawl_accel_cap",
      self.make_output(-0.2, has_lead=True, debug={"planner_seed_candidate_reason": "lead_crawl_accel_cap"}),
      reason="lead_crawl_accel_cap",
    )]
    planner.custom_v2_scene = CustomV2Scene(v_ego=0.5, v_cruise=6.0, has_lead=True)

    planner.apply_longitudinal_stack_selection(self.make_sm(), has_lead=True, accel_limits=(-2.0, 2.0))

    self.assertEqual(planner.longitudinal_stack_seed_context, "planner")
    self.assertEqual(planner.longitudinal_stack_seed_candidate, "lead_crawl_accel_cap")

  def test_moving_lead_stop_gap_guard_keeps_route_derived_slew_floor(self):
    planner = self.make_planner()
    planner.output_a_target = -0.462
    planner.a_desired_trajectory = tuple(-0.462 for _ in range(CONTROL_N))
    planner.planner_seed_candidates = [
      PlannerSeedCandidate(
        "lead_flicker_speedup_cap",
        self.make_output(0.0, has_lead=True, debug={"planner_seed_candidate_reason": "lead_flicker_speedup_cap"}),
        reason="lead_flicker_speedup_cap",
      ),
      *build_moving_lead_seed_candidates(
        planner, True, (-3.0, 2.0),
        moving_stop_guard_a_target=-1.95,
        lead_stop_approach_slewed_a_target=-0.852,
        lead_stop_approach_base_a_target=-1.95,
      ),
    ]
    planner.longitudinal_decision_candidates = [
      self.make_candidate(DecisionSource.LEAD_MPC, CandidateRole.PHYSICAL_HAZARD, 31.0, -0.462, "confirmed_radar_lead"),
    ]
    planner.custom_v2_scene = CustomV2Scene(v_ego=20.5, v_cruise=31.0, has_lead=True, lead_v=18.77, lead_v_rel=-1.83)

    planner.apply_longitudinal_stack_selection(self.make_sm(), has_lead=True, accel_limits=(-3.0, 2.0))

    self.assertEqual(planner.output_a_target, -0.852)
    self.assertEqual(planner.longitudinal_stack_selected_reason, "lead_stop_approach_slew")
    self.assertEqual(planner.longitudinal_stack_seed_candidate, "lead_stop_approach_slew")

  def test_non_custom_selection_uses_sunnypilot_current_without_wrapper(self):
    planner = self.make_planner(resolved_stack="sunnypilot-current")

    planner.apply_longitudinal_stack_selection(self.make_sm(), has_lead=False, accel_limits=(-2.0, 2.0))

    self.assertEqual(planner.output_a_target, -0.1)
    self.assertEqual(planner.longitudinal_stack_actuated_stack, "sunnypilot-current")
    self.assertFalse(planner.longitudinal_stack_fault_latched)
    self.assertEqual(planner.events_sp.names, [])

  def test_custom_v2_invalid_output_requests_immediate_disable_without_baseline_fallback(self):
    planner = self.make_planner()
    planner._custom_v2_stack_output = lambda _sunnypilot_output, _accel_limits: self.make_output(3.0)

    planner.apply_longitudinal_stack_selection(self.make_sm(), has_lead=False, accel_limits=(-2.0, 2.0))

    self.assertEqual(planner.output_a_target, -0.1)
    self.assertEqual(planner.longitudinal_stack_actuated_stack, "sunnypilot-current")
    self.assertTrue(planner.longitudinal_stack_fault_latched)
    self.assertEqual(planner.longitudinal_stack_fault_reason, "a_target_above_limits")
    self.assertEqual(planner.events_sp.names, [custom.OnroadEventSP.EventName.customLongitudinalStackFault])

  def test_custom_v2_invalid_scene_requests_immediate_disable_with_reason(self):
    planner = self.make_planner()
    planner.custom_v2_scene = CustomV2Scene(v_ego=float("nan"))

    planner.apply_longitudinal_stack_selection(self.make_sm(), has_lead=False, accel_limits=(-2.0, 2.0))

    self.assertEqual(planner.output_a_target, -0.1)
    self.assertEqual(planner.longitudinal_stack_actuated_stack, "sunnypilot-current")
    self.assertTrue(planner.longitudinal_stack_fault_latched)
    self.assertEqual(planner.longitudinal_stack_fault_reason, "invalid_scene_v_ego")
    self.assertEqual(planner.events_sp.names, [custom.OnroadEventSP.EventName.customLongitudinalStackFault])

  def test_custom_v2_fault_latch_resets_when_disabled(self):
    planner = self.make_planner()
    planner._custom_v2_stack_output = lambda _sunnypilot_output, _accel_limits: self.make_output(3.0)
    planner.apply_longitudinal_stack_selection(self.make_sm(enabled=True), has_lead=False, accel_limits=(-2.0, 2.0))

    planner.apply_longitudinal_stack_selection(self.make_sm(enabled=False), has_lead=False, accel_limits=(-2.0, 2.0))

    self.assertFalse(planner.custom_v2_fault_latched)
    self.assertFalse(planner.longitudinal_stack_fault_latched)


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
