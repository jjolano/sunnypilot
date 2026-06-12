from types import SimpleNamespace

from cereal import custom
from openpilot.selfdrive.controls.lib.planner_stacks.selector import (
  DEFAULT_STACK,
  PLANNER_CURRENT,
  SCENE_MEMORY_V1,
  PlannerCapabilities,
  PlannerStackCatalog,
  load_stack_manifest,
  normalize_stack_value,
  planner_stack_id_for_name,
  resolve_planner_stack,
)


def make_cp(**kwargs):
  values = {
    "brand": "hyundai",
    "carFingerprint": "HYUNDAI_TEST",
    "openpilotLongitudinalControl": True,
    "alphaLongitudinalAvailable": True,
    "pcmCruise": False,
    "radarUnavailable": False,
  }
  values.update(kwargs)
  return SimpleNamespace(**values)


def test_normalize_stack_value_defaults_to_planner_current():
  assert normalize_stack_value(None) == DEFAULT_STACK
  assert normalize_stack_value("") == DEFAULT_STACK
  assert normalize_stack_value(b"scene-memory-v1") == SCENE_MEMORY_V1


def test_unset_stack_resolves_to_planner_current():
  resolution = resolve_planner_stack(None, make_cp())

  assert resolution.requested_stack == PLANNER_CURRENT
  assert resolution.resolved_stack == PLANNER_CURRENT
  assert resolution.fallback_reason == ""


def test_scene_memory_v1_is_validation_gated_by_default():
  resolution = resolve_planner_stack(SCENE_MEMORY_V1, make_cp(), validation_gate=False)

  assert resolution.requested_stack == SCENE_MEMORY_V1
  assert resolution.resolved_stack == PLANNER_CURRENT
  assert resolution.fallback_reason == "validation_gate_unmet"
  assert SCENE_MEMORY_V1 not in resolution.available_stacks


def test_scene_memory_v1_resolves_when_validation_gate_passes():
  resolution = resolve_planner_stack(SCENE_MEMORY_V1, make_cp(), validation_gate=True)

  assert resolution.resolved_stack == SCENE_MEMORY_V1
  assert resolution.custom_version == "1.0"
  assert resolution.fallback_reason == ""
  assert SCENE_MEMORY_V1 in resolution.available_stacks


def test_scene_memory_v1_requires_longitudinal_capability_even_when_validated():
  resolution = resolve_planner_stack(
    SCENE_MEMORY_V1,
    make_cp(openpilotLongitudinalControl=False, alphaLongitudinalAvailable=False),
    validation_gate=True,
  )

  assert resolution.resolved_stack == PLANNER_CURRENT
  assert resolution.fallback_reason == "unavailable_stack"
  assert SCENE_MEMORY_V1 not in resolution.available_stacks


def test_unknown_planner_stack_falls_back_to_default():
  resolution = resolve_planner_stack("world-model", make_cp(), validation_gate=True)

  assert resolution.requested_stack == "world-model"
  assert resolution.resolved_stack == PLANNER_CURRENT
  assert resolution.fallback_reason == "unknown_stack"


def test_planner_stack_catalog_owns_labels_availability_and_versions():
  catalog = PlannerStackCatalog(load_stack_manifest())

  assert catalog.default_stack == PLANNER_CURRENT
  assert catalog.stack_names == (PLANNER_CURRENT, SCENE_MEMORY_V1)
  assert catalog.stack_definition(SCENE_MEMORY_V1).label == "Scene Memory v1"
  assert catalog.stack_definition(SCENE_MEMORY_V1).family == "scene-memory"
  assert catalog.stack_definition(SCENE_MEMORY_V1).version == "1.0"
  assert catalog.available_stacks(PlannerCapabilities(openpilot_longitudinal_control=True, planner_validation_gate=False)) == (
    PLANNER_CURRENT,
  )
  assert catalog.available_stacks(PlannerCapabilities(openpilot_longitudinal_control=True, planner_validation_gate=True)) == (
    PLANNER_CURRENT,
    SCENE_MEMORY_V1,
  )


def test_planner_stack_id_mapping_is_centralized_with_catalog():
  StackId = custom.LongitudinalPlanSP.PlannerStack.PlannerStackId

  assert planner_stack_id_for_name(PLANNER_CURRENT) == StackId.plannerCurrent
  assert planner_stack_id_for_name(SCENE_MEMORY_V1) == StackId.sceneMemoryV1
  assert planner_stack_id_for_name("unknown") == StackId.unknown
