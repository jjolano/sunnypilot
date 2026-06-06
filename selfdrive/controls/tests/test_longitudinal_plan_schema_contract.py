from pathlib import Path


SCHEMA_CONTRACT_DECLARATIONS = (
  "dec @0 :DynamicExperimentalControl;",
  "longitudinalPlanSource @1 :LongitudinalPlanSource;",
  "smartCruiseControl @2 :SmartCruiseControl;",
  "speedLimit @3 :SpeedLimit;",
  "decisionLayer @8 :DecisionLayer;",
  "stack @9 :Stack;",
  "longitudinalMode @10 :LongitudinalModeStatus;",
  "seedContext @11 :Text;",
  "seedCandidate @12 :Text;",
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
