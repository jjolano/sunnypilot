import unittest
from types import SimpleNamespace
from typing import Callable, cast

from cereal import messaging
from cereal import custom
from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N
from openpilot.selfdrive.controls.lib.longitudinal_decision import DecisionSource, LongitudinalDecisionTelemetry
from openpilot.selfdrive.controls.lib.longitudinal_stacks.fallback import CustomStackFallbackWrapper
from openpilot.selfdrive.controls.lib.longitudinal_stacks.interface import LongitudinalStackOutput
from openpilot.selfdrive.controls.lib.longitudinal_stacks.selector import StackResolution
from openpilot.sunnypilot.selfdrive.controls.lib import longitudinal_planner as sp_longitudinal_planner
from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import (
  LongitudinalPlannerSP,
  LongitudinalPlanSource,
  StackId,
  publish_stack_telemetry,
  select_lowest_longitudinal_target,
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
    self.assertEqual(plan_sp.stack.shadowStack, StackId.unknown)
    self.assertEqual(plan_sp.stack.customVersion, "")
    self.assertFalse(plan_sp.stack.fallbackLatched)
    self.assertEqual(plan_sp.stack.fallbackReason, "")
    self.assertEqual(plan_sp.stack.actuatedATarget, -0.25)
    self.assertEqual(plan_sp.stack.shadowATarget, 0.0)

  def test_publish_stack_telemetry_reports_custom_resolution_and_fallback_reason(self):
    plan_sp = self.make_plan_sp()
    resolution = StackResolution(
      requested_stack="custom-recommended",
      resolved_stack="custom-1.0",
      available_stacks=("sunnypilot-current", "custom-recommended", "custom-1.0"),
      recommended_stack="custom-1.0",
      custom_version="1.0",
      fallback_reason="runtime_contract_failure",
    )

    publish_stack_telemetry(
      plan_sp, resolution, "sunnypilot-current", -0.1,
      shadow_stack="custom-1.0", shadow_a_target=-0.3, fallback_latched=True,
    )

    self.assertEqual(plan_sp.stack.requestedStack, StackId.customRecommended)
    self.assertEqual(plan_sp.stack.resolvedStack, StackId.customV1)
    self.assertEqual(plan_sp.stack.actuatedStack, StackId.sunnypilotCurrent)
    self.assertEqual(plan_sp.stack.shadowStack, StackId.customV1)
    self.assertEqual(plan_sp.stack.customVersion, "1.0")
    self.assertTrue(plan_sp.stack.fallbackLatched)
    self.assertEqual(plan_sp.stack.fallbackReason, "runtime_contract_failure")
    self.assertEqual(plan_sp.stack.actuatedATarget, -0.1)
    self.assertEqual(plan_sp.stack.shadowATarget, -0.3)

  def test_capnp_stack_schema_fields_exist(self):
    msg = messaging.new_message("longitudinalPlanSP")
    msg.longitudinalPlanSP.stack.requestedStack = StackId.sunnypilotCurrent
    msg.longitudinalPlanSP.stack.resolvedStack = StackId.sunnypilotCurrent
    msg.longitudinalPlanSP.stack.actuatedStack = StackId.sunnypilotCurrent
    msg.longitudinalPlanSP.stack.fallbackReason = ""

    self.assertEqual(msg.longitudinalPlanSP.stack.requestedStack, StackId.sunnypilotCurrent)
    self.assertEqual(msg.longitudinalPlanSP.stack.fallbackReason, "")


class TestLongitudinalStackSelectionIntegration(unittest.TestCase):
  def make_planner(self, resolved_stack="custom-1.0", fallback_reason=""):
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
      custom_version="1.0" if resolved_stack == "custom-1.0" else "",
      fallback_reason=fallback_reason,
    )
    planner.longitudinal_stack_fallback = CustomStackFallbackWrapper(custom_stack=resolved_stack)
    planner.custom_longitudinal_stack = None
    planner.longitudinal_stack_actuated_stack = "sunnypilot-current"
    planner.longitudinal_stack_shadow_stack = ""
    planner.longitudinal_stack_shadow_a_target = 0.0
    planner.longitudinal_stack_fallback_latched = False
    planner.longitudinal_stack_fallback_reason = ""
    return planner

  def make_sm(self, enabled=True):
    return {"selfdriveState": SimpleNamespace(enabled=enabled)}

  def make_output(self, a_target):
    return LongitudinalStackOutput(
      a_target=a_target,
      should_stop=False,
      has_lead=False,
      source="custom",
      allow_throttle=True,
      allow_brake=True,
      speeds=tuple(10.0 for _ in range(CONTROL_N)),
      accels=tuple(a_target for _ in range(CONTROL_N)),
      jerks=tuple(0.0 for _ in range(CONTROL_N)),
    )

  def test_custom_selection_actuates_passthrough_custom_stack_without_behavior_change(self):
    planner = self.make_planner()

    planner.apply_longitudinal_stack_selection(self.make_sm(), has_lead=False, accel_limits=(-2.0, 2.0))

    self.assertEqual(planner.output_a_target, -0.1)
    self.assertEqual(planner.longitudinal_stack_actuated_stack, "custom-1.0")
    self.assertEqual(planner.longitudinal_stack_shadow_stack, "sunnypilot-current")
    self.assertEqual(planner.longitudinal_stack_shadow_a_target, -0.1)
    self.assertEqual(planner.custom_longitudinal_stack.stack_name, "custom-1.0")
    self.assertFalse(planner.longitudinal_stack_fallback_latched)
    self.assertEqual(planner.events_sp.names, [])

  def test_non_custom_selection_uses_sunnypilot_current_without_wrapper(self):
    planner = self.make_planner(resolved_stack="sunnypilot-current")

    planner.apply_longitudinal_stack_selection(self.make_sm(), has_lead=False, accel_limits=(-2.0, 2.0))

    self.assertEqual(planner.output_a_target, -0.1)
    self.assertEqual(planner.longitudinal_stack_actuated_stack, "sunnypilot-current")
    self.assertFalse(planner.longitudinal_stack_fallback_latched)
    self.assertEqual(planner.events_sp.names, [])

  def test_unimplemented_non_custom_stack_keeps_sunnypilot_actuation(self):
    planner = self.make_planner(resolved_stack="openpilot-current")

    planner.apply_longitudinal_stack_selection(self.make_sm(), has_lead=False, accel_limits=(-2.0, 2.0))

    self.assertEqual(planner.output_a_target, -0.1)
    self.assertEqual(planner.longitudinal_stack_actuated_stack, "sunnypilot-current")
    self.assertEqual(planner.longitudinal_stack_fallback_reason, "unimplemented_stack")
    self.assertFalse(planner.longitudinal_stack_fallback_latched)
    self.assertEqual(planner.events_sp.names, [])

  def test_selector_fallback_reason_is_preserved_for_non_custom_resolution(self):
    planner = self.make_planner(resolved_stack="sunnypilot-current", fallback_reason="unimplemented_stack")

    planner.apply_longitudinal_stack_selection(self.make_sm(), has_lead=False, accel_limits=(-2.0, 2.0))

    self.assertEqual(planner.output_a_target, -0.1)
    self.assertEqual(planner.longitudinal_stack_actuated_stack, "sunnypilot-current")
    self.assertEqual(planner.longitudinal_stack_fallback_reason, "unimplemented_stack")
    self.assertFalse(planner.longitudinal_stack_fallback_latched)

  def test_custom_selection_latches_fallback_and_emits_event_once(self):
    planner = self.make_planner()
    custom_calls = 0

    def invalid_custom(_sunnypilot_output):
      nonlocal custom_calls
      custom_calls += 1
      return self.make_output(3.0)

    planner._custom_v1_stack_output = invalid_custom
    planner.apply_longitudinal_stack_selection(self.make_sm(), has_lead=False, accel_limits=(-2.0, 2.0))
    first_events = list(planner.events_sp.names)
    planner.events_sp.names.clear()
    planner.apply_longitudinal_stack_selection(self.make_sm(), has_lead=False, accel_limits=(-2.0, 2.0))

    self.assertEqual(planner.output_a_target, -0.1)
    self.assertEqual(planner.longitudinal_stack_actuated_stack, "sunnypilot-current")
    self.assertTrue(planner.longitudinal_stack_fallback_latched)
    self.assertEqual(planner.longitudinal_stack_fallback_reason, "a_target_above_limits")
    self.assertEqual(first_events, [custom.OnroadEventSP.EventName.customLongitudinalFallback])
    self.assertEqual(planner.events_sp.names, [])
    self.assertEqual(custom_calls, 1)

  def test_custom_fallback_latch_resets_when_disabled(self):
    planner = self.make_planner()
    planner._custom_v1_stack_output = lambda _sunnypilot_output: self.make_output(3.0)
    planner.apply_longitudinal_stack_selection(self.make_sm(enabled=True), has_lead=False, accel_limits=(-2.0, 2.0))

    planner._custom_v1_stack_output = lambda sunnypilot_output: sunnypilot_output
    planner.apply_longitudinal_stack_selection(self.make_sm(enabled=False), has_lead=False, accel_limits=(-2.0, 2.0))

    self.assertFalse(planner.longitudinal_stack_fallback_latched)
    self.assertEqual(planner.longitudinal_stack_actuated_stack, "sunnypilot-current")

if __name__ == "__main__":
  unittest.main()
