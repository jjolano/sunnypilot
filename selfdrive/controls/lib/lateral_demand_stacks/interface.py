from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class LateralDemandStackInputs:
  lat_active: bool
  v_ego: float
  desired_curvature: float
  measured_curvature: float
  model_v2: object
  live_params: object
  curvature_limited: bool
  accurate_lateral_accel: bool
  manual_gas_lateral_accel_override: bool
  lateral_maneuver_curvature: float | None
  roll: float
  lateral_accel_limit_no_roll: float
  default_lateral_accel_limited: bool
  lane_change_state: int
  lane_change_direction: int
  turn_direction: int
  model_data_v2_sp_valid: bool
  lane_centering_assist_enabled: bool
  gas_pressed: bool
  brake_pressed: bool
  steering_pressed: bool
  left_blinker: bool
  right_blinker: bool
  left_lane_y0: float | None
  right_lane_y0: float | None
  frame_drop_perc: float
  smoothed_model_path_curvature: bool
  position_x: Sequence[float] = ()
  position_y: Sequence[float] = ()
  position_y_std: Sequence[float] = ()
  orientation_z: Sequence[float] = ()
  orientation_rate_z: Sequence[float] = ()
  lane_line_probs: Sequence[float] = ()
  lane_line_stds: Sequence[float] = ()
  sm_valid_model_v2: bool = True
  sm_valid_model_data_v2: bool = True
  sm_valid_live_parameters: bool = True
  sm_valid_lateral_maneuver_plan: bool = True


@dataclass(frozen=True)
class LateralDemandStackOutput:
  requested_stack: str
  resolved_stack: str
  fallback_reason: str
  version: str
  legacy: object
  profile: object | None = None
  debug: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StackOutputValidation:
  valid: bool
  reason: str = ""


def validate_lateral_demand_stack_output(output: object) -> StackOutputValidation:
  if not isinstance(output, LateralDemandStackOutput):
    return StackOutputValidation(False, "invalid_output_type")
  if not isinstance(output.legacy, object) or output.legacy is None:
    return StackOutputValidation(False, "missing_legacy")
  if not isinstance(output.requested_stack, str) or not output.requested_stack:
    return StackOutputValidation(False, "missing_requested_stack")
  if not isinstance(output.resolved_stack, str) or not output.resolved_stack:
    return StackOutputValidation(False, "missing_resolved_stack")
  if not isinstance(output.fallback_reason, str):
    return StackOutputValidation(False, "invalid_fallback_reason")
  if not isinstance(output.version, str):
    return StackOutputValidation(False, "invalid_version")
  if not isinstance(output.profile, (type(None), object)):
    return StackOutputValidation(False, "invalid_profile_type")
  if not isinstance(output.debug, dict):
    return StackOutputValidation(False, "invalid_debug")
  return StackOutputValidation(True)
