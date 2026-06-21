"""LateralDemandPipeline — ordered, toggleable composition of the demand processors.

Replaces the legacy lateral-demand stack/registry/selector with one pipeline:

    raw model curvature
      -> model_path_processor   (quality gating + smoothing/damping)
      -> lane_change_path_shaper (minimum-jerk lane-change path)
      -> lane_centering_assist   (optional centering nudge)
      = processed desired curvature

Each stage is individually toggleable. The pipeline emits a ProcessedLateralDemand plus a
flat debug dict (raw vs post-each-stage curvature) so any future quirk attribution is a log
query, not a fork restart.

NOT in this first cut (deferred to the controlsd wiring): the lateral-accel-limit burst
logic and clip_curvature — those are control-loop concerns applied downstream. The pipeline
produces the pre-clip processed curvature; ``curvature_limited`` is passed through from the
caller's prior clip.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from openpilot.common.realtime import DT_CTRL
from openpilot.sunnypilot.custom.lateral.demand.types import (
  DEMAND_SOURCE_FALLBACK_MEASURED,
  DEMAND_SOURCE_LATERAL_MANEUVER,
  DEMAND_SOURCE_MODEL_PATH,
  ProcessedLateralDemand,
)
from openpilot.sunnypilot.custom.lateral.demand.lane_centering_assist import (
  LaneCenteringAssistInputs,
  LaneCenteringAssistTracker,
  inactive_lane_centering_assist_result,
)
from openpilot.sunnypilot.custom.lateral.demand.lane_change_path_shaper import (
  LaneChangePathShaper,
  LaneChangePathShaperInputs,
)
from openpilot.sunnypilot.custom.lateral.demand.curve_memory import (
  CurveMemory,
  CurveMemoryInputs,
)
from openpilot.sunnypilot.custom.lateral.demand.model_path_processor import (
  ModelPathProcessor,
  ModelPathProcessorInputs,
  ModelPathProcessorResult,
)

# turn_curvature_sign convention (matches legacy: TurnDirection.turnRight=1, turnLeft=2)
TURN_DIRECTION_RIGHT = 1
TURN_DIRECTION_LEFT = 2
LANE_CHANGE_STATE_OFF = 0


@dataclass(frozen=True)
class LateralDemandPipelineInputs:
  lat_active: bool
  v_ego: float
  roll: float
  desired_curvature: float          # raw model curvature
  measured_curvature: float
  # model path processor
  position_x: Sequence[float] = ()
  position_y: Sequence[float] = ()
  position_y_std: Sequence[float] = ()
  orientation_z: Sequence[float] = ()
  orientation_rate_z: Sequence[float] = ()
  lane_line_probs: Sequence[float] = ()
  frame_drop_perc: float = 0.0
  model_age_s: float = 0.0
  model_data_v2_sp_valid: bool = True
  turn_direction: int = 0
  # lane change
  lane_change_state: int = LANE_CHANGE_STATE_OFF
  lane_change_direction: int = 0
  lane_change_state_valid: bool = False
  left_blinker: bool = False
  right_blinker: bool = False
  steering_pressed: bool | None = None
  left_lane_y0: float | None = None
  right_lane_y0: float | None = None
  # lateral maneuver override (takes the whole pipeline)
  lateral_maneuver_curvature: float | None = None
  # toggles
  smooth_model_path_curvature: bool = False
  demand_jerk_smoothing_enabled: bool = False
  lane_centering_assist_enabled: bool = False
  curve_memory_enabled: bool = False
  # passed through (downstream clip result)
  curvature_limited: bool = False


@dataclass(frozen=True)
class LateralDemandPipelineResult:
  demand: ProcessedLateralDemand
  model_path_result: ModelPathProcessorResult
  debug: dict[str, float | str | bool] = field(default_factory=dict)


class LateralDemandPipeline:
  def __init__(self, dt: float = DT_CTRL) -> None:
    self.dt = float(dt)
    self._model_path_processor = ModelPathProcessor()
    self._curve_memory = CurveMemory()
    self._lane_change_path_shaper = LaneChangePathShaper(dt)
    self._lane_centering_assist = LaneCenteringAssistTracker()
    self._previous_desired_curvature = 0.0

  @property
  def previous_desired_curvature(self) -> float:
    return self._previous_desired_curvature

  def reset(self) -> None:
    self._model_path_processor.reset()
    self._curve_memory.reset()
    self._lane_change_path_shaper.reset()
    self._lane_centering_assist.reset()
    self._previous_desired_curvature = 0.0

  def update(self, inputs: LateralDemandPipelineInputs) -> LateralDemandPipelineResult:
    raw_curvature = float(inputs.desired_curvature)
    demand_source = DEMAND_SOURCE_MODEL_PATH
    lane_change_shaping_active = False
    lane_change_blend = 0.0
    lane_centering_result = inactive_lane_centering_assist_result("disabled")
    curve_memory_result = None

    if inputs.lateral_maneuver_curvature is not None:
      self._model_path_processor.reset()
      self._curve_memory.reset()
      self._lane_change_path_shaper.reset()
      self._lane_centering_assist.reset()
      new_desired_curvature = float(inputs.lateral_maneuver_curvature)
      model_path_result = ModelPathProcessorResult(new_desired_curvature, 0.0, True, "lateral_maneuver")
      demand_source = DEMAND_SOURCE_LATERAL_MANEUVER
    else:
      turn_curvature_sign = 0
      if inputs.lane_change_state == LANE_CHANGE_STATE_OFF and inputs.model_data_v2_sp_valid:
        if inputs.turn_direction == TURN_DIRECTION_RIGHT:
          turn_curvature_sign = 1
        elif inputs.turn_direction == TURN_DIRECTION_LEFT:
          turn_curvature_sign = -1

      model_path_result = self._model_path_processor.update(ModelPathProcessorInputs(
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
        model_age_s=inputs.model_age_s,
        smooth_model_path_curvature=inputs.smooth_model_path_curvature,
        demand_jerk_smoothing_enabled=inputs.demand_jerk_smoothing_enabled,
        demand_jerk_smoothing_allowed=(
          inputs.steering_pressed is False
          and inputs.lane_change_state_valid
          and inputs.lane_change_state == LANE_CHANGE_STATE_OFF
          and not inputs.left_blinker
          and not inputs.right_blinker
        ),
        lane_change_active=inputs.lane_change_state != LANE_CHANGE_STATE_OFF,
      ))
      # Pose-anchored CurveMemory stage: remember road curvature seen ahead with good vision and
      # recall it through the low-speed traverse where vision degrades (runs every frame to track
      # arc length + capture; only ever raises an under-curved vision, vetoed by confident vision).
      curve_memory_result = self._curve_memory.update(CurveMemoryInputs(
        enabled=inputs.curve_memory_enabled, lat_active=inputs.lat_active, v_ego=inputs.v_ego,
        desired_curvature=model_path_result.desired_curvature, path_quality=model_path_result.quality,
        path_gated=model_path_result.gated, path_reason=model_path_result.reason,
        steering_pressed=inputs.steering_pressed if inputs.steering_pressed is not None else None,
        lane_change_active=(inputs.lane_change_state != LANE_CHANGE_STATE_OFF) if inputs.lane_change_state_valid else True,
        position_x=tuple(inputs.position_x), position_y=tuple(inputs.position_y),
        orientation_z=tuple(inputs.orientation_z),
        valid_path=ModelPathProcessor._valid_core_path(inputs.position_x, inputs.position_y),
      ), self.dt)
      model_desired_curvature = curve_memory_result.desired_curvature if inputs.lat_active else inputs.measured_curvature
      if not inputs.lat_active:
        demand_source = DEMAND_SOURCE_FALLBACK_MEASURED

      lane_change_result = self._lane_change_path_shaper.update(LaneChangePathShaperInputs(
        lat_active=inputs.lat_active,
        v_ego=inputs.v_ego,
        left_blinker=inputs.left_blinker,
        right_blinker=inputs.right_blinker,
        steering_pressed=bool(inputs.steering_pressed),
        lane_change_state=inputs.lane_change_state,
        lane_change_direction=inputs.lane_change_direction,
        model_curvature=model_desired_curvature,
        prev_desired_curvature=(self._previous_desired_curvature if inputs.lat_active else inputs.measured_curvature),
        lane_line_probs=tuple(inputs.lane_line_probs),
        left_lane_y0=inputs.left_lane_y0,
        right_lane_y0=inputs.right_lane_y0,
      ))
      lane_change_shaping_active = bool(lane_change_result.active)
      lane_change_blend = float(lane_change_result.blend)
      new_desired_curvature = lane_change_result.desired_curvature if inputs.lat_active else inputs.measured_curvature

      if inputs.lane_centering_assist_enabled and demand_source == DEMAND_SOURCE_MODEL_PATH:
        lane_centering_result = self._lane_centering_assist.update(LaneCenteringAssistInputs(
          lat_active=inputs.lat_active,
          v_ego=inputs.v_ego,
          measured_curvature=inputs.measured_curvature,
          model_curvature=new_desired_curvature,
          previous_processed_curvature=self._previous_desired_curvature,
          path_quality=model_path_result.quality,
          path_reason=model_path_result.reason,
          lane_change_shaping_active=lane_change_shaping_active,
          lane_change_blend=lane_change_blend,
          curvature_limited=inputs.curvature_limited,
          steering_pressed=bool(inputs.steering_pressed),
          left_blinker=inputs.left_blinker,
          right_blinker=inputs.right_blinker,
          position_x=tuple(inputs.position_x),
          position_y=tuple(inputs.position_y),
          orientation_z=tuple(inputs.orientation_z),
          lane_line_probs=tuple(inputs.lane_line_probs),
          demand_source=demand_source,
        ), self.dt)
        new_desired_curvature += lane_centering_result.curvature_nudge
      elif not inputs.lane_centering_assist_enabled:
        self._lane_centering_assist.reset()

    processed_curvature = float(new_desired_curvature)
    # Stress guardrail: anomalous / non-finite curvature is contained before it reaches
    # the controller. Non-finite values fall back to straight-ahead (0.0) and reset the
    # previous-curvature memory so the next valid frame starts from a clean state.
    # Values beyond 0.05 1/m are logged but passed through unchanged.
    if not math.isfinite(processed_curvature):
      from openpilot.common.swaglog import cloudlog
      cloudlog.warning(f"lateral_demand nonfinite processed curvature: {processed_curvature}")
      processed_curvature = 0.0
    elif abs(processed_curvature) > 0.05:
      from openpilot.common.swaglog import cloudlog
      cloudlog.warning(f"lateral_demand extreme processed curvature: {processed_curvature:.4f}")
    self._previous_desired_curvature = processed_curvature

    demand = ProcessedLateralDemand(
      raw_curvature=raw_curvature,
      processed_curvature=processed_curvature,
      measured_curvature=inputs.measured_curvature,
      curvature_limited=inputs.curvature_limited,
      path_quality=model_path_result.quality,
      path_reason=model_path_result.reason,
      lane_change_shaping_active=lane_change_shaping_active,
      lane_change_blend=lane_change_blend,
      lateral_accel_limit=0.0,  # set by downstream clip; pipeline is pre-clip
      demand_source=demand_source,
      lane_centering_assist_active=lane_centering_result.active,
      lane_centering_reason=lane_centering_result.reason,
      lane_centering_lateral_error=lane_centering_result.lateral_error,
      lane_centering_heading_error=lane_centering_result.heading_error,
      lane_centering_predicted_error=lane_centering_result.predicted_lateral_error,
      lane_centering_curvature_nudge=lane_centering_result.curvature_nudge,
      lane_centering_confidence=lane_centering_result.confidence,
    )
    return LateralDemandPipelineResult(
      demand=demand,
      model_path_result=model_path_result,
      debug={
        "raw_curvature": raw_curvature,
        "model_path_curvature": float(model_path_result.desired_curvature),
        "model_path_reason": model_path_result.reason,
        "model_path_quality": float(model_path_result.quality),
        "model_age_s": float(inputs.model_age_s),
        "demand_jerk_smoothing_active": bool(model_path_result.demand_jerk_smoothing_active),
        "demand_jerk_smoothing_step": float(model_path_result.demand_jerk_smoothing_step),
        "demand_jerk_smoothing_lag": float(model_path_result.demand_jerk_smoothing_lag),
        "lane_change_blend": lane_change_blend,
        "lane_change_shaping_active": lane_change_shaping_active,
        "lane_centering_active": bool(lane_centering_result.active),
        "lane_centering_nudge": float(lane_centering_result.curvature_nudge),
        "curve_memory_active": bool(curve_memory_result.active) if curve_memory_result is not None else False,
        "curve_memory_remembered": float(curve_memory_result.remembered) if curve_memory_result is not None else float("nan"),
        "curve_memory_source": curve_memory_result.source if curve_memory_result is not None else "disabled",
        "curve_memory_samples": int(curve_memory_result.samples) if curve_memory_result is not None else 0,
        "processed_curvature": processed_curvature,
        "demand_source": demand_source,
        "dtle_estimate": _compute_dtle(inputs.left_lane_y0, inputs.right_lane_y0),
      },
    )


def _compute_dtle(left_lane_y0: float | None, right_lane_y0: float | None) -> float:
    """Curvature-independent DTLE estimate from lane-line near-point positions.

    Positive = vehicle right of lane center. Normalized to [-1, 1] where
    ±1 = at lane edge. Returns NaN if lane data unavailable.
    """
    if left_lane_y0 is None or right_lane_y0 is None:
        return float("nan")
    lane_center = (float(left_lane_y0) + float(right_lane_y0)) / 2.0
    half_width = abs(float(right_lane_y0) - float(left_lane_y0)) / 2.0
    if half_width < 0.1:
        return float("nan")
    return -lane_center / half_width
