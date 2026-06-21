"""controlsd wiring for the lateral demand pipeline (opt-in).

``LateralDemandAdapter`` is held by the controller loop; when ``CustomLateralDemandEnabled``
is set it processes the raw model curvature through the demand pipeline before clipping.
Default off => the stock model curvature is used, so this can never change default steering.

Evidence mapped from verified ``modelV2`` fields (position x/y/yStd, orientation z + rate,
laneLineProbs). Lane-change state/direction are wired when available; unknown values fail closed
for curve memory. See docs/touch-points.md.
"""
from __future__ import annotations

from dataclasses import replace
import math
from typing import Any

from openpilot.sunnypilot.custom.lateral.demand.pipeline import (
  LateralDemandPipeline,
  LateralDemandPipelineInputs,
)

PARAMS_REFRESH_PERIOD = 100  # control ticks (100Hz -> ~1s)
LANE_CHANGE_STATE_VALUES = {
  "off": 0,
  "preLaneChange": 1,
  "laneChangeStarting": 2,
  "laneChangeFinishing": 3,
}
LANE_CHANGE_DIRECTION_VALUES = {
  "none": 0,
  "left": 1,
  "right": 2,
}


def sanitized_model_age_s(model_age_s: float | None) -> float:
  try:
    age = float(model_age_s) if model_age_s is not None else float("inf")
  except (TypeError, ValueError):
    return float("inf")
  return age if math.isfinite(age) and age >= 0.0 else float("inf")


def _enum_to_int(value: Any, name_values: dict[str, int]) -> tuple[int, bool]:
  if value is None:
    return 0, False
  raw_value = getattr(value, "value", value)
  try:
    return int(raw_value), True
  except (TypeError, ValueError):
    pass

  name = str(raw_value).split(".")[-1]
  if name in name_values:
    return name_values[name], True
  return 0, False


def build_pipeline_inputs(*, lat_active: bool, v_ego: float, roll: float, raw_curvature: float,
                          measured_curvature: float, model_v2: Any,
                          lane_centering_assist_enabled: bool,
                          curve_memory_enabled: bool = False,
                          steering_pressed: bool | None = None,
                          model_age_s: float = 0.0,
                          yaw_rate: float | None = None,
                          steering_rate_deg: float | None = None,
                          steer_limited: bool = False,
                          demand_jerk_smoothing_enabled: bool = False) -> LateralDemandPipelineInputs:
  pos = getattr(model_v2, "position", None)
  ori = getattr(model_v2, "orientation", None)
  ori_rate = getattr(model_v2, "orientationRate", None)
  meta = getattr(model_v2, "meta", None)
  lane_change_state = getattr(meta, "laneChangeState", None)
  lane_change_direction = getattr(meta, "laneChangeDirection", None)
  lane_change_state_value, lane_change_state_valid = _enum_to_int(lane_change_state, LANE_CHANGE_STATE_VALUES)
  lane_change_direction_value, _ = _enum_to_int(lane_change_direction, LANE_CHANGE_DIRECTION_VALUES)
  return LateralDemandPipelineInputs(
    lat_active=lat_active, v_ego=v_ego, roll=roll,
    desired_curvature=raw_curvature, measured_curvature=measured_curvature,
    position_x=tuple(getattr(pos, "x", ()) or ()),
    position_y=tuple(getattr(pos, "y", ()) or ()),
    position_y_std=tuple(getattr(pos, "yStd", ()) or ()),
    orientation_z=tuple(getattr(ori, "z", ()) or ()),
    orientation_rate_z=tuple(getattr(ori_rate, "z", ()) or ()),
    lane_line_probs=tuple(getattr(model_v2, "laneLineProbs", ()) or ()),
    frame_drop_perc=float(getattr(model_v2, "frameDropPerc", 0.0) or 0.0),
    model_age_s=sanitized_model_age_s(model_age_s),
    yaw_rate=yaw_rate,
    steering_rate_deg=steering_rate_deg,
    steer_limited=bool(steer_limited),
    lane_change_state=lane_change_state_value,
    lane_change_direction=lane_change_direction_value,
    lane_change_state_valid=lane_change_state_valid,
    steering_pressed=steering_pressed,
    demand_jerk_smoothing_enabled=bool(demand_jerk_smoothing_enabled),
    lane_centering_assist_enabled=bool(lane_centering_assist_enabled),
    curve_memory_enabled=bool(curve_memory_enabled),
  )


class LateralDemandAdapter:
  def __init__(self, params: Any = None):
    self._params = params
    self._pipeline = LateralDemandPipeline()
    self._tick = 0
    self.enabled = False
    self.lane_centering_assist_enabled = False
    self.curve_memory_enabled = False
    self.last_result = None
    self.last_debug = {}
    if params is not None:
      self.refresh_params()

  def refresh_params(self) -> None:
    p = self._params
    if p is None:
      return
    try:
      self.enabled = bool(p.get_bool("CustomLateralDemandEnabled"))
      self.lane_centering_assist_enabled = bool(p.get_bool("LaneCenteringAssistEnabled"))
      self.curve_memory_enabled = bool(p.get_bool("CurveMemoryEnabled"))
    except Exception:
      self.enabled = False

  def clear(self) -> None:
    self.last_result = None
    self.last_debug = {}

  def process(self, lat_active: bool, v_ego: float, roll: float, raw_curvature: float,
              measured_curvature: float, model_v2: Any, steering_pressed: bool | None = None,
              model_age_s: float = 0.0, yaw_rate: float | None = None,
              steering_rate_deg: float | None = None, steer_limited: bool = False) -> float:
    """Return the processed desired curvature, or the unchanged raw curvature when disabled
    or on any fault (fail-closed)."""
    self._tick += 1
    if self._params is not None and self._tick % PARAMS_REFRESH_PERIOD == 0:
      self.refresh_params()
    if not self.enabled or model_v2 is None:
      self.clear()
      return raw_curvature
    try:
      inputs = build_pipeline_inputs(
        lat_active=lat_active, v_ego=v_ego, roll=roll, raw_curvature=raw_curvature,
        measured_curvature=measured_curvature, model_v2=model_v2,
        lane_centering_assist_enabled=self.lane_centering_assist_enabled,
        curve_memory_enabled=self.curve_memory_enabled,
        steering_pressed=steering_pressed,
        model_age_s=model_age_s,
        yaw_rate=yaw_rate,
        steering_rate_deg=steering_rate_deg,
        steer_limited=steer_limited,
      )
      inputs = replace(inputs, smooth_model_path_curvature=True)
      result = self._pipeline.update(inputs)
      self.last_result = result
      self.last_debug = dict(result.debug)
      return float(result.demand.processed_curvature)
    except Exception:
      self.clear()
      return raw_curvature
