import openpilot.selfdrive.controls.lib.feature_registry_entries  # noqa: F401  (registers entries on import)
from openpilot.selfdrive.controls.lib.feature_registry import (
  FeatureRegistryEntry,
  ResolvedFeatureSet,
  StackFeatures,
  feature_registry_snapshot,
  get_features,
  known_stacks,
  merge_features,
  register_features,
  resolve_feature_set,
)


def test_known_stacks_lists_all_four():
  stacks = known_stacks()
  assert "sunnypilot-current" in stacks
  assert "custom-recommended" in stacks
  assert "custom-2.0" in stacks
  assert "custom-experimental" in stacks


def test_sunnypilot_current_features_are_empty():
  f = get_features("sunnypilot-current")
  assert f.one_pedal_longitudinal is False
  assert f.lead_confirmed_progress is False
  assert f.lane_change_path_shaping is False
  assert f.lane_centering_assist is False
  assert f.straight_road_damping is False
  assert f.lateral_oscillation_supervisor is False
  assert f.lateral_turn_exit_controller is False
  assert f.experimental_stage == "stable"


def test_custom_v2_features_are_full():
  f = get_features("custom-2.0")
  assert f.one_pedal_longitudinal is True
  assert f.lead_confirmed_progress is True
  assert f.lane_change_path_shaping is True
  assert f.lane_centering_assist is True
  assert f.straight_road_damping is True
  assert f.lateral_oscillation_supervisor is True
  assert f.lateral_turn_exit_controller is True
  assert f.experimental_stage == "stable"


def test_custom_experimental_features_match_v2_with_experimental_stage():
  v2 = get_features("custom-2.0")
  exp = get_features("custom-experimental")
  assert exp.one_pedal_longitudinal == v2.one_pedal_longitudinal
  assert exp.lead_confirmed_progress == v2.lead_confirmed_progress
  assert exp.lane_change_path_shaping == v2.lane_change_path_shaping
  assert exp.lane_centering_assist == v2.lane_centering_assist
  assert exp.straight_road_damping == v2.straight_road_damping
  assert exp.lateral_oscillation_supervisor == v2.lateral_oscillation_supervisor
  assert exp.lateral_turn_exit_controller == v2.lateral_turn_exit_controller
  assert exp.experimental_stage == "experimental"


def test_get_features_for_unknown_stack_returns_empty():
  f = get_features("does-not-exist")
  assert f.experimental_stage == "stable"
  assert f.one_pedal_longitudinal is False


def test_merge_features_unions_boolean_fields():
  merged = merge_features("sunnypilot-current", "custom-2.0")
  assert merged.one_pedal_longitudinal is True
  assert merged.lane_centering_assist is True
  assert merged.experimental_stage == "stable"


def test_merge_features_with_no_args_returns_stable():
  merged = merge_features()
  assert merged.experimental_stage == "stable"
  assert merged.one_pedal_longitudinal is False


def test_resolve_feature_set_reports_fallback():
  res = resolve_feature_set("custom-experimental", "custom-experimental")
  assert isinstance(res, ResolvedFeatureSet)
  assert res.is_custom is True
  assert res.fallback_active is False
  assert res.features.experimental_stage == "experimental"


def test_resolved_feature_set_with_fallback():
  res = resolve_feature_set(
    "custom-experimental",
    "sunnypilot-current",
    fallback_active=True,
    fallback_reason="unavailable_stack",
  )
  assert res.fallback_active is True
  assert res.fallback_reason == "unavailable_stack"
  assert res.is_custom is False
  assert res.features.one_pedal_longitudinal is False


def test_feature_registry_snapshot_contains_all_four():
  snap = feature_registry_snapshot()
  assert set(snap.keys()) == {"sunnypilot-current", "custom-recommended", "custom-2.0", "custom-experimental"}
  assert snap["custom-2.0"]["family"] == "custom"
  assert snap["custom-2.0"]["features"]["one_pedal_longitudinal"] is True


def test_register_features_adds_new_entry():
  register_features(FeatureRegistryEntry(
    stack="custom-test-3.0",
    family="custom",
    features=StackFeatures(one_pedal_longitudinal=True, experimental_stage="experimental"),
  ))
  try:
    assert "custom-test-3.0" in known_stacks()
    f = get_features("custom-test-3.0")
    assert f.one_pedal_longitudinal is True
    assert f.experimental_stage == "experimental"
  finally:
    pass
