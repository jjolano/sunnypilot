from __future__ import annotations

import math
from typing import TYPE_CHECKING

from openpilot.selfdrive.controls.lib.drive_helpers import (
  MAX_LATERAL_ACCEL_NO_ROLL,
  clip_curvature,
  clip_curvature_with_result,
  should_latch_lateral_accel_burst,
  update_lateral_accel_limit,
)
from openpilot.selfdrive.controls.lib.lane_change_path_shaper import (
  LaneChangePathShaper,
  LaneChangePathShaperInputs,
)
from openpilot.selfdrive.controls.lib.lane_centering_assist import (
  LaneCenteringAssistInputs,
  LaneCenteringAssistTracker,
  inactive_lane_centering_assist_result,
)
from openpilot.selfdrive.controls.lib.lateral_demand import (
  DEMAND_SOURCE_FALLBACK_MEASURED,
  DEMAND_SOURCE_LATERAL_MANEUVER,
  DEMAND_SOURCE_MODEL_PATH,
  ProcessedLateralDemand,
)
from openpilot.selfdrive.controls.lib.lateral_demand_profile import (
  LateralDemandProfile,
  LateralDemandProfileBuilder,
)
from openpilot.selfdrive.controls.lib.lateral_demand_stacks.interface import (
  LateralDemandStackInputs,
  LateralDemandStackOutput,
)
from openpilot.selfdrive.controls.lib.model_path_processor import (
  ModelPathProcessor,
  ModelPathProcessorInputs,
  ModelPathProcessorResult,
)

if TYPE_CHECKING:
  pass


class CustomV2LateralDemandStack:
  NAME = "custom-2.0"
  VERSION = "2.0"

  def __init__(self, dt: float) -> None:
    self.dt = float(dt)
    self._lane_change_path_shaper = LaneChangePathShaper(dt)
    self._lane_centering_assist_tracker = LaneCenteringAssistTracker()
    self._model_path_processor = ModelPathProcessor()
    self._lateral_demand_profile_builder = LateralDemandProfileBuilder(dt=dt)

    self._model_path_result = ModelPathProcessorResult(0.0, 0.0, True, "inactive")
    self._model_path_raw_desired_curvature = 0.0
    self._smoothed_model_path_curvature = False
    self._previous_desired_curvature = 0.0
    self._lateral_accel_limit_no_roll = MAX_LATERAL_ACCEL_NO_ROLL
    self._default_lateral_accel_limited = False

    self._last_legacy_demand: ProcessedLateralDemand | None = None
    self._last_profile: LateralDemandProfile | None = None

  @property
  def model_path_result(self) -> ModelPathProcessorResult:
    return self._model_path_result

  @property
  def model_path_raw_desired_curvature(self) -> float:
    return self._model_path_raw_desired_curvature

  @property
  def smoothed_model_path_curvature(self) -> bool:
    return self._smoothed_model_path_curvature

  @property
  def lateral_accel_limit_no_roll(self) -> float:
    return self._lateral_accel_limit_no_roll

  @property
  def last_legacy_demand(self) -> ProcessedLateralDemand | None:
    return self._last_legacy_demand

  @property
  def last_profile(self) -> LateralDemandProfile | None:
    return self._last_profile

  def reset(self) -> None:
    self._lane_change_path_shaper.reset()
    self._lane_centering_assist_tracker.reset()
    self._model_path_processor.reset()
    self._model_path_result = ModelPathProcessorResult(0.0, 0.0, True, "inactive")
    self._model_path_raw_desired_curvature = 0.0
    self._smoothed_model_path_curvature = False
    self._previous_desired_curvature = 0.0
    self._lateral_accel_limit_no_roll = MAX_LATERAL_ACCEL_NO_ROLL
    self._default_lateral_accel_limited = False
    self._last_legacy_demand = None
    self._last_profile = None

  def update(self, inputs: LateralDemandStackInputs) -> LateralDemandStackOutput:
    raw_curvature = float(inputs.desired_curvature)
    self._lateral_accel_limit_no_roll = update_lateral_accel_limit(
      self._lateral_accel_limit_no_roll,
      inputs.manual_gas_lateral_accel_override,
      inputs.lat_active,
      inputs.brake_pressed,
      inputs.steering_pressed,
      default_lateral_accel_limited=self._default_lateral_accel_limited,
    )

    lane_change_shaping_active = False
    lane_change_blend = 0.0
    lane_centering_result = inactive_lane_centering_assist_result("disabled")
    demand_source = DEMAND_SOURCE_MODEL_PATH

    if inputs.lateral_maneuver_curvature is not None:
      self._lane_change_path_shaper.reset()
      self._model_path_processor.reset()
      self._lane_centering_assist_tracker.reset()
      new_desired_curvature = inputs.lateral_maneuver_curvature
      self._model_path_result = ModelPathProcessorResult(
        inputs.lateral_maneuver_curvature, 0.0, True, "lateral_maneuver",
      )
      self._model_path_raw_desired_curvature = raw_curvature
      demand_source = DEMAND_SOURCE_LATERAL_MANEUVER
    else:
      turn_curvature_sign = 0
      if inputs.lane_change_state == 0 and inputs.model_data_v2_sp_valid:  # LaneChangeState.off
        if inputs.turn_direction == 1:  # TurnDirection.turnRight
          turn_curvature_sign = 1
        elif inputs.turn_direction == 2:  # TurnDirection.turnLeft
          turn_curvature_sign = -1

      self._model_path_result = self._model_path_processor.update(
        ModelPathProcessorInputs(
          lat_active=inputs.lat_active,
          v_ego=inputs.v_ego,
          desired_curvature=inputs.desired_curvature,
          measured_curvature=inputs.measured_curvature,
          previous_desired_curvature=self._previous_desired_curvature,
          position_x=tuple(inputs.position_x),
          position_y=tuple(inputs.position_y),
          position_y_std=tuple(inputs.position_y_std),
          orientation_z=tuple(inputs.orientation_z),
          orientation_rate_z=tuple(inputs.orientation_rate_z),
          lane_line_probs=tuple(inputs.lane_line_probs),
          turn_curvature_sign=turn_curvature_sign,
          frame_drop_perc=inputs.frame_drop_perc,
          smooth_model_path_curvature=self._smoothed_model_path_curvature,
          lane_change_active=inputs.lane_change_state != 0,
        )
      )
      self._model_path_raw_desired_curvature = raw_curvature
      model_desired_curvature = (
        self._model_path_result.desired_curvature if inputs.lat_active
        else inputs.measured_curvature
      )
      if not inputs.lat_active:
        demand_source = DEMAND_SOURCE_FALLBACK_MEASURED

      lane_change_result = self._lane_change_path_shaper.update(
        LaneChangePathShaperInputs(
          lat_active=inputs.lat_active,
          v_ego=inputs.v_ego,
          left_blinker=inputs.left_blinker,
          right_blinker=inputs.right_blinker,
          steering_pressed=inputs.steering_pressed,
          lane_change_state=inputs.lane_change_state,
          lane_change_direction=inputs.lane_change_direction,
          model_curvature=model_desired_curvature,
          prev_desired_curvature=(
            self._previous_desired_curvature if inputs.lat_active
            else inputs.measured_curvature
          ),
          lane_line_probs=tuple(inputs.lane_line_probs),
          left_lane_y0=inputs.left_lane_y0,
          right_lane_y0=inputs.right_lane_y0,
        )
      )
      lane_change_shaping_active = bool(lane_change_result.active)
      lane_change_blend = float(lane_change_result.blend)
      new_desired_curvature = (
        lane_change_result.desired_curvature if inputs.lat_active
        else inputs.measured_curvature
      )

      if inputs.lane_centering_assist_enabled and demand_source == DEMAND_SOURCE_MODEL_PATH:
        base_clip_result = clip_curvature_with_result(
          inputs.v_ego,
          self._previous_desired_curvature,
          new_desired_curvature,
          inputs.roll,
          self._lateral_accel_limit_no_roll,
          accurate_lateral_accel=inputs.accurate_lateral_accel,
        )
        lane_centering_result = self._lane_centering_assist_tracker.update(
          LaneCenteringAssistInputs(
            lat_active=inputs.lat_active,
            v_ego=inputs.v_ego,
            measured_curvature=inputs.measured_curvature,
            model_curvature=new_desired_curvature,
            previous_processed_curvature=self._previous_desired_curvature,
            path_quality=self._model_path_result.quality,
            path_reason=self._model_path_result.reason,
            lane_change_shaping_active=lane_change_shaping_active,
            lane_change_blend=lane_change_blend,
            curvature_limited=base_clip_result.limited,
            steering_pressed=inputs.steering_pressed,
            left_blinker=inputs.left_blinker,
            right_blinker=inputs.right_blinker,
            position_x=tuple(inputs.position_x),
            position_y=tuple(inputs.position_y),
            orientation_z=tuple(inputs.orientation_z),
            lane_line_probs=tuple(inputs.lane_line_probs),
            demand_source=demand_source,
          ),
          self.dt,
        )
        new_desired_curvature += lane_centering_result.curvature_nudge
      elif not inputs.lane_centering_assist_enabled:
        self._lane_centering_assist_tracker.reset()

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
      path_quality=self._model_path_result.quality,
      path_reason=self._model_path_result.reason,
      lane_change_shaping_active=lane_change_shaping_active,
      lane_change_blend=lane_change_blend,
      lateral_accel_limit=self._lateral_accel_limit_no_roll,
      demand_source=demand_source,
      lane_centering_assist_active=lane_centering_result.active,
      lane_centering_reason=lane_centering_result.reason,
      lane_centering_lateral_error=lane_centering_result.lateral_error,
      lane_centering_heading_error=lane_centering_result.heading_error,
      lane_centering_predicted_error=lane_centering_result.predicted_lateral_error,
      lane_centering_curvature_nudge=lane_centering_result.curvature_nudge,
      lane_centering_confidence=lane_centering_result.confidence,
    )
    self._last_legacy_demand = demand
    profile = self._lateral_demand_profile_builder.update(
      demand, inputs.v_ego,
      curvature_limited=curvature_limited,
      saturated=False,
      steer_limited_by_safety=inputs.curvature_limited,
      steering_pressed=inputs.steering_pressed,
    )
    self._last_profile = profile

    return LateralDemandStackOutput(
      requested_stack=self.NAME,
      resolved_stack=self.NAME,
      fallback_reason="",
      version=self.VERSION,
      legacy=demand,
      profile=profile,
      debug={
        "lane_change_shaping_active": lane_change_shaping_active,
        "lane_change_blend": lane_change_blend,
        "lane_centering_active": bool(lane_centering_result.active),
        "model_path_reason": self._model_path_result.reason,
        "model_path_quality": float(self._model_path_result.quality),
        "demand_source": demand_source,
      },
    )
