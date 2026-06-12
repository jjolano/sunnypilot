from types import SimpleNamespace

from openpilot.selfdrive.controls.lib.longitudinal_modes import (
  LongitudinalActuationType,
  LongitudinalMode,
  LongitudinalModeResolution,
  ResolvedLongitudinalImplementation,
)
from openpilot.selfdrive.controls.lib.planner_stacks.scene_memory import SceneMemory, SceneMemorySnapshot
from openpilot.selfdrive.controls.lib.planner_stacks.selector import PLANNER_CURRENT, SCENE_MEMORY_V1, StackResolution


def make_resolution(requested=PLANNER_CURRENT, resolved=PLANNER_CURRENT, fallback_reason=""):
  return StackResolution(
    requested_stack=requested,
    resolved_stack=resolved,
    available_stacks=(PLANNER_CURRENT,),
    fallback_reason=fallback_reason,
  )


def make_mode(mode=LongitudinalMode.ACC, implementation=ResolvedLongitudinalImplementation.HARDWARE_ACC):
  return LongitudinalModeResolution(
    requested_mode=mode,
    resolved_implementation=implementation,
    actuation_type=LongitudinalActuationType.DIRECT,
  )


def test_scene_memory_defaults_are_inactive():
  snapshot = SceneMemorySnapshot()

  assert snapshot.enabled is False
  assert snapshot.active is False
  assert snapshot.shadow is False
  assert snapshot.provenance == ()
  assert snapshot.source_eligibility == ()


def test_scene_memory_consumes_existing_artifact_provenance_without_owning_output():
  planner = SimpleNamespace(
    primary_lead_context=SimpleNamespace(
      behavior=SimpleNamespace(confidence=0.75),
      physical=None,
    ),
    longitudinal_decision_telemetry=object(),
    active_resolver=SimpleNamespace(speed_limit_valid=True, speed_limit_last_valid=False),
    active_sla=SimpleNamespace(is_active=False),
    active_scc=SimpleNamespace(),
    osm_traffic_control_prior=SimpleNamespace(),
    output_a_target=-0.1,
  )

  snapshot = SceneMemory().update_from_planner(
    planner,
    make_mode(LongitudinalMode.SCC, ResolvedLongitudinalImplementation.SCC_ACC),
    make_resolution(requested=SCENE_MEMORY_V1, resolved=PLANNER_CURRENT, fallback_reason="validation_gate_unmet"),
    PLANNER_CURRENT,
  )

  assert snapshot.enabled is True
  assert snapshot.active is False
  assert snapshot.shadow is True
  assert snapshot.summary == "validation_gate_unmet"
  assert snapshot.lead_stability == 0.75
  assert snapshot.map_speed_stability == 1.0
  assert "lead:PrimaryLeadContext" in snapshot.provenance
  assert "decision:LongitudinalDecisionTelemetry" in snapshot.provenance
  assert "speed:SpeedLimitResolver" in snapshot.provenance
  assert "scc_curve" in snapshot.source_eligibility
  assert planner.output_a_target == -0.1


def test_scene_memory_acc_mode_marks_map_and_scc_sources_ineligible():
  snapshot = SceneMemory().update_from_planner(
    SimpleNamespace(),
    make_mode(LongitudinalMode.ACC, ResolvedLongitudinalImplementation.HARDWARE_ACC),
    make_resolution(),
    PLANNER_CURRENT,
  )

  assert snapshot.enabled is False
  assert snapshot.summary == "planner_current"
  assert snapshot.source_eligibility == ("cruise", "lead")
  assert "scc_curve" not in snapshot.source_eligibility
  assert "speed_limit" not in snapshot.source_eligibility
  assert "map_caution" not in snapshot.source_eligibility


def test_scene_memory_active_state_is_explicit_for_future_milestones():
  snapshot = SceneMemory().update_from_planner(
    SimpleNamespace(),
    make_mode(LongitudinalMode.E2E, ResolvedLongitudinalImplementation.E2E),
    StackResolution(
      requested_stack=SCENE_MEMORY_V1,
      resolved_stack=SCENE_MEMORY_V1,
      available_stacks=(PLANNER_CURRENT, SCENE_MEMORY_V1),
    ),
    SCENE_MEMORY_V1,
  )

  assert snapshot.enabled is True
  assert snapshot.active is True
  assert snapshot.shadow is False
  assert snapshot.summary == "scene_memory_active"
  assert "model_stop" in snapshot.source_eligibility


def test_validated_scene_memory_resolution_remains_shadow_when_actuated_stack_is_current():
  snapshot = SceneMemory().update_from_planner(
    SimpleNamespace(output_a_target=0.2),
    make_mode(LongitudinalMode.SCC, ResolvedLongitudinalImplementation.SCC_ACC),
    StackResolution(
      requested_stack=SCENE_MEMORY_V1,
      resolved_stack=SCENE_MEMORY_V1,
      available_stacks=(PLANNER_CURRENT, SCENE_MEMORY_V1),
    ),
    PLANNER_CURRENT,
  )

  assert snapshot.enabled is True
  assert snapshot.active is False
  assert snapshot.shadow is True
  assert snapshot.summary == "scene_memory_shadow"
