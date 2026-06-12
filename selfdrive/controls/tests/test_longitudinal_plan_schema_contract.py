from pathlib import Path


SCHEMA_CONTRACT_DECLARATIONS = (
  "dec @0 :DynamicExperimentalControl;",
  "longitudinalPlanSource @1 :LongitudinalPlanSource;",
  "smartCruiseControl @2 :SmartCruiseControl;",
  "speedLimit @3 :SpeedLimit;",
  "decisionLayer @8 :DecisionLayer;",
  "stack @9 :Stack;",
  "longitudinalMode @10 :LongitudinalModeStatus;",
  "plannerStack @11 :PlannerStack;",
  "sceneMemory @12 :SceneMemory;",
  "seedContext @11 :Text;",
  "seedCandidate @12 :Text;",
)

PLANNER_STACK_CONTRACT_DECLARATIONS = (
  "requestedStack @0 :PlannerStackId;",
  "resolvedStack @1 :PlannerStackId;",
  "actuatedStack @2 :PlannerStackId;",
  "validationGatePassed @3 :Bool;",
  "compatibilityFallbackReason @4 :Text;",
  "faultLatched @5 :Bool;",
  "faultReason @6 :Text;",
)

SCENE_MEMORY_CONTRACT_DECLARATIONS = (
  "enabled @0 :Bool;",
  "active @1 :Bool;",
  "shadow @2 :Bool;",
  "oldestEvidenceAge @3 :Float32;",
  "leadStability @4 :Float32;",
  "pathStability @5 :Float32;",
  "mapSpeedStability @6 :Float32;",
  "invalidEvidenceCount @7 :UInt32;",
  "staleEvidenceCount @8 :UInt32;",
  "provenance @9 :List(Text);",
  "sourceEligibility @10 :List(Text);",
  "summary @11 :Text;",
)

LONGITUDINAL_MODE_STATUS_CONTRACT_DECLARATIONS = (
  "requestedMode @0 :Mode;",
  "resolvedImplementation @1 :Implementation;",
  "actuationType @2 :ActuationType;",
  "restrictionStatus @3 :List(Text);",
  "unsupportedReason @4 :Text;",
  "compatibilityAliasState @5 :CompatibilityAliasState;",
  "evidenceTier @6 :Text;",
  "evidenceReason @7 :Text;",
  "evidenceConfidence @8 :Float32;",
  "evidenceUrgency @9 :Float32;",
  "independentOfLead @10 :Bool;",
  "evidenceAdvisories @11 :List(Text);",
  "confirmedLead @12 :Bool;",
)


def test_longitudinal_plan_sp_schema_field_ids_are_append_only():
  schema = Path("cereal/custom.capnp").read_text()

  assert "struct LongitudinalPlanSP @0xf35cc4560bbf6ec2" in schema
  assert "struct Stack {" in schema
  for declaration in SCHEMA_CONTRACT_DECLARATIONS:
    assert declaration in schema

  ordered_offsets = [schema.index(declaration) for declaration in SCHEMA_CONTRACT_DECLARATIONS]
  assert ordered_offsets == sorted(ordered_offsets)


def test_longitudinal_mode_status_schema_field_ids_are_stable():
  # Text-based schema guard: capnp introspection is heavier than needed for
  # append-only field IDs, and this catches accidental renumbering directly.
  schema = Path("cereal/custom.capnp").read_text()

  assert "struct LongitudinalModeStatus {" in schema
  for declaration in LONGITUDINAL_MODE_STATUS_CONTRACT_DECLARATIONS:
    assert declaration in schema

  ordered_offsets = [schema.index(declaration) for declaration in LONGITUDINAL_MODE_STATUS_CONTRACT_DECLARATIONS]
  assert ordered_offsets == sorted(ordered_offsets)


def test_planner_stack_schema_field_ids_are_stable():
  schema = Path("cereal/custom.capnp").read_text()
  schema = schema[schema.index("struct PlannerStack {"):]

  assert "struct PlannerStack {" in schema
  for declaration in PLANNER_STACK_CONTRACT_DECLARATIONS:
    assert declaration in schema

  ordered_offsets = [schema.index(declaration) for declaration in PLANNER_STACK_CONTRACT_DECLARATIONS]
  assert ordered_offsets == sorted(ordered_offsets)


def test_scene_memory_schema_field_ids_are_stable():
  schema = Path("cereal/custom.capnp").read_text()
  schema = schema[schema.index("struct SceneMemory {"):]

  assert "struct SceneMemory {" in schema
  for declaration in SCENE_MEMORY_CONTRACT_DECLARATIONS:
    assert declaration in schema

  ordered_offsets = [schema.index(declaration) for declaration in SCENE_MEMORY_CONTRACT_DECLARATIONS]
  assert ordered_offsets == sorted(ordered_offsets)
