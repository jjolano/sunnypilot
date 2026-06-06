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


def test_longitudinal_plan_sp_schema_field_ids_are_append_only():
  schema = Path("cereal/custom.capnp").read_text()

  assert "struct LongitudinalPlanSP @0xf35cc4560bbf6ec2" in schema
  assert "struct Stack {" in schema
  for declaration in SCHEMA_CONTRACT_DECLARATIONS:
    assert declaration in schema

  ordered_offsets = [schema.index(declaration) for declaration in SCHEMA_CONTRACT_DECLARATIONS]
  assert ordered_offsets == sorted(ordered_offsets)
