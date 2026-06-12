from openpilot.selfdrive.controls.lib.feature_registry import (
  FeatureRegistryEntry,
  StackFeatures,
  feature_registry_snapshot,
  get_features,
  known_stacks,
  merge_features,
  register_features,
  resolve_feature_set,
)


register_features(FeatureRegistryEntry(
  stack="sunnypilot-current",
  family="baseline",
  features=StackFeatures(
    one_pedal_longitudinal=False,
    lead_confirmed_progress=False,
    lane_change_path_shaping=False,
    lane_centering_assist=False,
    straight_road_damping=False,
    lateral_oscillation_supervisor=False,
    lateral_turn_exit_controller=False,
    experimental_stage="stable",
  ),
))


register_features(FeatureRegistryEntry(
  stack="custom-recommended",
  family="custom-alias",
  features=StackFeatures(
    one_pedal_longitudinal=False,
    lead_confirmed_progress=False,
    lane_change_path_shaping=False,
    lane_centering_assist=False,
    straight_road_damping=False,
    lateral_oscillation_supervisor=False,
    lateral_turn_exit_controller=False,
    experimental_stage="stable",
  ),
))


register_features(FeatureRegistryEntry(
  stack="custom-2.0",
  family="custom",
  features=StackFeatures(
    one_pedal_longitudinal=True,
    lead_confirmed_progress=True,
    lane_change_path_shaping=True,
    lane_centering_assist=True,
    straight_road_damping=True,
    lateral_oscillation_supervisor=True,
    lateral_turn_exit_controller=True,
    experimental_stage="stable",
  ),
))


register_features(FeatureRegistryEntry(
  stack="custom-experimental",
  family="custom",
  features=StackFeatures(
    one_pedal_longitudinal=True,
    lead_confirmed_progress=True,
    lane_change_path_shaping=True,
    lane_centering_assist=True,
    straight_road_damping=True,
    lateral_oscillation_supervisor=True,
    lateral_turn_exit_controller=True,
    experimental_stage="experimental",
  ),
))
