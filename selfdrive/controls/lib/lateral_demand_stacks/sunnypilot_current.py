from __future__ import annotations

from openpilot.selfdrive.controls.lib.drive_helpers import (
  MAX_LATERAL_ACCEL_NO_ROLL,
  clip_curvature,
  should_latch_lateral_accel_burst,
  update_lateral_accel_limit,
)
from openpilot.selfdrive.controls.lib.lateral_demand import (
  DEMAND_SOURCE_FALLBACK_MEASURED,
  DEMAND_SOURCE_LATERAL_MANEUVER,
  DEMAND_SOURCE_MODEL_PATH,
  ProcessedLateralDemand,
)
from openpilot.selfdrive.controls.lib.lateral_demand_stacks.interface import (
  LateralDemandStackInputs,
  LateralDemandStackOutput,
)
from openpilot.selfdrive.controls.lib.model_path_processor import ModelPathProcessorResult


class SunnypilotCurrentLateralDemandStack:
  NAME = "sunnypilot-current"
  VERSION = ""

  def __init__(self, dt: float) -> None:
    self.dt = float(dt)
    self._previous_desired_curvature = 0.0
    self._lateral_accel_limit_no_roll = MAX_LATERAL_ACCEL_NO_ROLL
    self._default_lateral_accel_limited = False
    self._last_legacy_demand: ProcessedLateralDemand | None = None
    self._model_path_result = ModelPathProcessorResult(0.0, 0.0, True, "inactive")
    self._model_path_raw_desired_curvature = 0.0

  @property
  def lateral_accel_limit_no_roll(self) -> float:
    return self._lateral_accel_limit_no_roll

  @property
  def last_legacy_demand(self) -> ProcessedLateralDemand | None:
    return self._last_legacy_demand

  @property
  def last_profile(self):
    return None

  @property
  def model_path_result(self) -> ModelPathProcessorResult:
    return self._model_path_result

  @property
  def model_path_raw_desired_curvature(self) -> float:
    return self._model_path_raw_desired_curvature

  def reset(self) -> None:
    self._previous_desired_curvature = 0.0
    self._lateral_accel_limit_no_roll = MAX_LATERAL_ACCEL_NO_ROLL
    self._default_lateral_accel_limited = False
    self._last_legacy_demand = None
    self._model_path_result = ModelPathProcessorResult(0.0, 0.0, True, "inactive")
    self._model_path_raw_desired_curvature = 0.0

  def update(self, inputs: LateralDemandStackInputs) -> LateralDemandStackOutput:
    raw_curvature = float(inputs.desired_curvature)
    self._model_path_raw_desired_curvature = raw_curvature
    self._lateral_accel_limit_no_roll = update_lateral_accel_limit(
      self._lateral_accel_limit_no_roll,
      inputs.manual_gas_lateral_accel_override,
      inputs.lat_active,
      inputs.brake_pressed,
      inputs.steering_pressed,
      default_lateral_accel_limited=self._default_lateral_accel_limited,
    )

    if inputs.lateral_maneuver_curvature is not None:
      new_desired_curvature = inputs.lateral_maneuver_curvature
      demand_source = DEMAND_SOURCE_LATERAL_MANEUVER
      path_quality = 1.0
      path_reason = "lateral_maneuver"
    elif inputs.lat_active:
      new_desired_curvature = raw_curvature
      demand_source = DEMAND_SOURCE_MODEL_PATH
      path_quality = 1.0
      path_reason = "ok"
    else:
      new_desired_curvature = inputs.measured_curvature
      demand_source = DEMAND_SOURCE_FALLBACK_MEASURED
      path_quality = 1.0
      path_reason = "ok"

    self._model_path_result = ModelPathProcessorResult(
      new_desired_curvature,
      path_quality,
      True,
      path_reason,
    )

    processed_curvature, curvature_limited, default_lateral_accel_limited = clip_curvature(
      inputs.v_ego,
      self._previous_desired_curvature,
      new_desired_curvature,
      inputs.roll,
      self._lateral_accel_limit_no_roll,
      accurate_lateral_accel=inputs.accurate_lateral_accel,
    )
    self._default_lateral_accel_limited = should_latch_lateral_accel_burst(
      default_lateral_accel_limited,
      inputs.lat_active,
      inputs.brake_pressed,
      inputs.steering_pressed,
      inputs.manual_gas_lateral_accel_override,
    )
    self._previous_desired_curvature = processed_curvature

    demand = ProcessedLateralDemand(
      raw_curvature=raw_curvature,
      processed_curvature=processed_curvature,
      measured_curvature=inputs.measured_curvature,
      curvature_limited=curvature_limited,
      path_quality=path_quality,
      path_reason=path_reason,
      lane_change_shaping_active=False,
      lane_change_blend=0.0,
      lateral_accel_limit=self._lateral_accel_limit_no_roll,
      demand_source=demand_source,
      lane_centering_assist_active=False,
      lane_centering_reason="disabled",
    )
    self._last_legacy_demand = demand

    return LateralDemandStackOutput(
      requested_stack=self.NAME,
      resolved_stack=self.NAME,
      fallback_reason="",
      version=self.VERSION,
      legacy=demand,
      profile=None,
      debug={
        "demand_source": demand_source,
        "path_reason": path_reason,
        "lane_change_shaping_active": False,
        "lane_centering_active": False,
      },
    )
