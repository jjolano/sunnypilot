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
  sanitize_lane_fit_source_mode,
  sanitize_lane_rate_damping_mode,
)
from openpilot.sunnypilot.custom.lateral.demand.preview import sanitize_lateral_preview_assist_mode
from openpilot.sunnypilot.custom.lateral.demand.model_path_processor import sanitize_straight_path_stabilization_mode
from openpilot.sunnypilot.custom.lateral.demand.lane_centering_assist import sanitize_one_line_centering_mode
from openpilot.sunnypilot.custom.lateral.demand.sensor_confidence import (
  SensorConfidenceInputs,
  evaluate_sensor_confidence,
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


def _lane_y0(lane_lines: Any, index: int) -> float | None:
  """Extract near-point y0 for a lane line, returning None on missing/short/nonfinite data."""
  try:
    line = lane_lines[index]
    ys = getattr(line, "y", None)
    if ys is None or len(ys) == 0:
      return None
    y0 = float(ys[0])
  except (TypeError, IndexError, ValueError, AttributeError):
    return None
  return y0 if math.isfinite(y0) else None


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


_model_extract_cache: tuple | None = None  # (frame_id, extracted model-array fields)


def build_pipeline_inputs(*, lat_active: bool, v_ego: float, roll: float, raw_curvature: float,
                          measured_curvature: float, model_v2: Any,
                          lane_centering_assist_enabled: bool,
                          curve_memory_enabled: bool = False,
                          steering_pressed: bool | None = None,
                          model_age_s: float = 0.0,
                          yaw_rate: float | None = None,
                          steering_rate_deg: float | None = None,
                          steer_limited: bool = False,
                          left_blinker: bool = False,
                          right_blinker: bool = False,
                          curvature_limited: bool = False,
                          demand_jerk_smoothing_enabled: bool = False,
                          straight_path_stabilization_mode: str = "off",
                          lat_delay: float = 0.0,
                          lateral_preview_assist_mode: str = "off") -> LateralDemandPipelineInputs:
  meta = getattr(model_v2, "meta", None)
  lane_change_state = getattr(meta, "laneChangeState", None)
  lane_change_direction = getattr(meta, "laneChangeDirection", None)
  lane_change_state_value, lane_change_state_valid = _enum_to_int(lane_change_state, LANE_CHANGE_STATE_VALUES)
  lane_change_direction_value, _ = _enum_to_int(lane_change_direction, LANE_CHANGE_DIRECTION_VALUES)
  # ponytail: model arrays only change with a new modelV2 (20Hz); cache the tuple conversions across
  # the 100Hz ticks, keyed on frameId. frameId 0/absent (tests, mocks) bypasses the cache entirely.
  global _model_extract_cache
  frame_id = int(getattr(model_v2, "frameId", 0) or 0)
  if frame_id and _model_extract_cache is not None and _model_extract_cache[0] == frame_id:
    ext = _model_extract_cache[1]
  else:
    pos = getattr(model_v2, "position", None)
    ori = getattr(model_v2, "orientation", None)
    ori_rate = getattr(model_v2, "orientationRate", None)
    lane_lines = tuple(getattr(model_v2, "laneLines", ()) or ())
    ext = dict(
      position_x=tuple(getattr(pos, "x", ()) or ()),
      position_y=tuple(getattr(pos, "y", ()) or ()),
      position_y_std=tuple(getattr(pos, "yStd", ()) or ()),
      orientation_z=tuple(getattr(ori, "z", ()) or ()),
      orientation_rate_z=tuple(getattr(ori_rate, "z", ()) or ()),
      lane_line_probs=tuple(getattr(model_v2, "laneLineProbs", ()) or ()),
      lane_line_stds=tuple(getattr(model_v2, "laneLineStds", ()) or ()),
      lane_lines=lane_lines,
      frame_drop_perc=float(getattr(model_v2, "frameDropPerc", 0.0) or 0.0),
      left_lane_y0=_lane_y0(lane_lines, 1) if len(lane_lines) > 2 else None,
      right_lane_y0=_lane_y0(lane_lines, 2) if len(lane_lines) > 2 else None,
      model_frame_id=frame_id,
    )
    if frame_id:
      _model_extract_cache = (frame_id, ext)
  return LateralDemandPipelineInputs(
    lat_active=lat_active, v_ego=v_ego, roll=roll,
    desired_curvature=raw_curvature, measured_curvature=measured_curvature,
    **ext,
    model_age_s=sanitized_model_age_s(model_age_s),
    yaw_rate=yaw_rate,
    steering_rate_deg=steering_rate_deg,
    steer_limited=bool(steer_limited),
    left_blinker=bool(left_blinker),
    right_blinker=bool(right_blinker),
    lane_change_state=lane_change_state_value,
    lane_change_direction=lane_change_direction_value,
    lane_change_state_valid=lane_change_state_valid,
    steering_pressed=steering_pressed,
    demand_jerk_smoothing_enabled=bool(demand_jerk_smoothing_enabled),
    lane_centering_assist_enabled=bool(lane_centering_assist_enabled),
    curve_memory_enabled=bool(curve_memory_enabled),
    lat_delay=lat_delay,
    lateral_preview_assist_mode=sanitize_lateral_preview_assist_mode(lateral_preview_assist_mode),
    straight_path_stabilization_mode=sanitize_straight_path_stabilization_mode(straight_path_stabilization_mode),
    curvature_limited=bool(curvature_limited),
  )


def build_sensor_confidence_inputs(*, lat_active: bool, v_ego: float, raw_curvature: float,
                                   measured_curvature: float, model_v2: Any,
                                   steering_pressed: bool | None = None,
                                   model_age_s: float = 0.0,
                                   yaw_rate: float | None = None,
                                   steering_rate_deg: float | None = None,
                                   steer_limited: bool = False) -> SensorConfidenceInputs:
  meta = getattr(model_v2, "meta", None)
  lane_change_state = getattr(meta, "laneChangeState", None)
  lane_change_state_value, lane_change_state_valid = _enum_to_int(lane_change_state, LANE_CHANGE_STATE_VALUES)
  return SensorConfidenceInputs(
    lat_active=lat_active,
    v_ego=v_ego,
    model_curvature=raw_curvature,
    measured_curvature=measured_curvature,
    model_path_gated=False,
    model_path_reason="ok",
    model_age_s=sanitized_model_age_s(model_age_s),
    steering_pressed=steering_pressed,
    steering_rate_deg=steering_rate_deg,
    yaw_rate=yaw_rate,
    steer_limited=bool(steer_limited),
    lane_change_active=lane_change_state_valid and lane_change_state_value != LANE_CHANGE_STATE_VALUES["off"],
    lane_change_state_valid=lane_change_state_valid,
  )


def _param_string(params: Any, key: str) -> str | None:
  try:
    raw = params.get(key)
  except TypeError:
    raw = params.get(key, None)
  if raw is None:
    return None
  if isinstance(raw, bytes):
    raw = raw.decode(errors="ignore")
  return str(raw)


class LateralDemandAdapter:
  def __init__(self, params: Any = None):
    self._params = params
    self._pipeline = LateralDemandPipeline()
    self._tick = 0
    self.enabled = False
    self.lane_centering_assist_enabled = False
    self.curve_memory_enabled = False
    self.straight_path_stabilization_mode = "off"
    self.lane_rate_damping_mode = "off"
    self.lane_fit_source_mode = "off"
    self.lane_centering_one_line_mode = "off"
    self.lateral_preview_assist_mode = "off"
    self.last_result = None
    self.last_debug = {}
    if params is not None:
      self.refresh_params()
    self._was_enabled = self.enabled
    # ponytail: cache one disabled snapshot; recompute on the next enable/disable transition.
    self._disabled_debug_valid = False

  def refresh_params(self) -> None:
    p = self._params
    if p is None:
      return
    try:
      self.enabled = bool(p.get_bool("CustomLateralDemandEnabled"))
      self.lane_centering_assist_enabled = bool(p.get_bool("LaneCenteringAssistEnabled"))
      self.curve_memory_enabled = bool(p.get_bool("CurveMemoryEnabled"))
      mode = _param_string(p, "StraightPathStabilizationMode")
      self.straight_path_stabilization_mode = sanitize_straight_path_stabilization_mode(mode)
      self.lane_rate_damping_mode = sanitize_lane_rate_damping_mode(_param_string(p, "LaneRateDampingMode"))
      self.lane_fit_source_mode = sanitize_lane_fit_source_mode(_param_string(p, "LaneFitSourceMode"))
      self.lane_centering_one_line_mode = sanitize_one_line_centering_mode(_param_string(p, "LaneCenteringOneLineMode"))
      self.lateral_preview_assist_mode = sanitize_lateral_preview_assist_mode(_param_string(p, "LateralPreviewAssistMode"))
    except Exception:
      self.enabled = False
      self.straight_path_stabilization_mode = "off"
      self.lane_rate_damping_mode = "off"
      self.lane_fit_source_mode = "off"
      self.lane_centering_one_line_mode = "off"
      self.lateral_preview_assist_mode = "off"

  def clear(self) -> None:
    self.last_result = None
    self.last_debug = {}
    self._disabled_debug_valid = False

  def reset(self) -> None:
    try:
      self._pipeline.reset()
    except Exception:
      pass
    self.clear()
    self._was_enabled = False

  def _observe_sensor_confidence(self, lat_active: bool, v_ego: float, raw_curvature: float,
                                 measured_curvature: float, model_v2: Any,
                                 steering_pressed: bool | None, model_age_s: float,
                                 yaw_rate: float | None, steering_rate_deg: float | None,
                                 steer_limited: bool) -> dict[str, float | str | bool]:
    if model_v2 is None:
      return {}
    try:
      inputs = build_sensor_confidence_inputs(
        lat_active=lat_active,
        v_ego=v_ego,
        raw_curvature=raw_curvature,
        measured_curvature=measured_curvature,
        model_v2=model_v2,
        steering_pressed=steering_pressed,
        model_age_s=model_age_s,
        yaw_rate=yaw_rate,
        steering_rate_deg=steering_rate_deg,
        steer_limited=steer_limited,
      )
      return evaluate_sensor_confidence(inputs).debug_dict()
    except Exception:
      return {
        "sensor_confidence_available": False,
        "sensor_confidence_block_reason": "fault",
        "sensor_confidence_score": 0.0,
        "sensor_disagreement_level": "blocked",
        "sensor_suppress_candidate": False,
      }

  def process(self, lat_active: bool, v_ego: float, roll: float, raw_curvature: float,
              measured_curvature: float, model_v2: Any, steering_pressed: bool | None = None,
              model_age_s: float = 0.0, yaw_rate: float | None = None,
              steering_rate_deg: float | None = None, steer_limited: bool = False,
              left_blinker: bool = False, right_blinker: bool = False,
              curvature_limited: bool = False, lat_delay: float = 0.0) -> float:
    """Return the processed desired curvature, or the unchanged raw curvature when disabled
    or on any fault (fail-closed)."""
    self._tick += 1
    if self._params is not None and self._tick % PARAMS_REFRESH_PERIOD == 0:
      self.refresh_params()

    enabled = bool(self.enabled)
    if not enabled or model_v2 is None:
      if self._was_enabled and not enabled:
        self.reset()
      elif model_v2 is None and self.last_result is not None:
        self.reset()
      if model_v2 is not None and not self._disabled_debug_valid:
        self.last_debug = self._observe_sensor_confidence(
          lat_active, v_ego, raw_curvature, measured_curvature, model_v2,
          steering_pressed, model_age_s, yaw_rate, steering_rate_deg, steer_limited,
        )
        self._disabled_debug_valid = True
      elif model_v2 is None:
        self.last_debug = {}
        self._disabled_debug_valid = False
      self._was_enabled = enabled
      return raw_curvature
    self._was_enabled = enabled
    self._disabled_debug_valid = False
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
        left_blinker=left_blinker,
        right_blinker=right_blinker,
        curvature_limited=curvature_limited,
        straight_path_stabilization_mode=self.straight_path_stabilization_mode,
        lat_delay=lat_delay,
        lateral_preview_assist_mode=self.lateral_preview_assist_mode,
      )
      inputs = replace(inputs, lane_rate_damping_mode=self.lane_rate_damping_mode, lane_fit_source_mode=self.lane_fit_source_mode,
                       lane_centering_one_line_mode=self.lane_centering_one_line_mode,
                       smooth_model_path_curvature=True, demand_jerk_smoothing_enabled=True)
      result = self._pipeline.update(inputs)
      self.last_result = result
      self.last_debug = dict(result.debug)
      return float(result.demand.processed_curvature)
    except Exception:
      self.clear()
      return raw_curvature
