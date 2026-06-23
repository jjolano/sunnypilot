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

from openpilot.common.swaglog import cloudlog

from openpilot.sunnypilot.custom.lateral.demand.pipeline import (
  LateralDemandPipeline,
  LateralDemandPipelineInputs,
)
from openpilot.sunnypilot.custom.lateral.demand.sensor_confidence import (
  SensorConfidenceInputs,
  evaluate_sensor_confidence,
)

PARAMS_REFRESH_PERIOD = 100  # control ticks (100Hz -> ~1s)
SHADOW_VERSION = 1
SHADOW_TRACE_PERIOD = 100  # control ticks (100Hz -> <=1 Hz)
SHADOW_TRACE_XS = (10.0, 20.0, 30.0)
SHADOW_PROB_THRESHOLD = 0.3
SHADOW_MIN_LANE_WIDTH = 2.0
SHADOW_MAX_LANE_WIDTH = 6.0
SHADOW_V_EGO_FLOOR_MPS = 0.5
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


def _debug_trace_mode(value: Any) -> str:
  text = str(value or "").strip().lower()
  return text if text in ("off", "log") else "off"


def _finite_scalar(value: Any) -> float | None:
  try:
    v = float(value)
  except (TypeError, ValueError):
    return None
  return v if math.isfinite(v) else None


def _lane_change_state_from_model(model_v2: Any) -> tuple[int, int, bool]:
  meta = getattr(model_v2, "meta", None)
  lane_change_state = getattr(meta, "laneChangeState", None)
  lane_change_direction = getattr(meta, "laneChangeDirection", None)
  state_value, state_valid = _enum_to_int(lane_change_state, LANE_CHANGE_STATE_VALUES)
  direction_value, _ = _enum_to_int(lane_change_direction, LANE_CHANGE_DIRECTION_VALUES)
  return state_value, direction_value, state_valid


def _interp_scalar(x: float, xs: tuple[float, ...], ys: tuple[float, ...]) -> float | None:
  if len(xs) < 2 or len(xs) != len(ys):
    return None
  if x <= xs[0]:
    return _finite_scalar(ys[0])
  if x >= xs[-1]:
    return _finite_scalar(ys[-1])
  for i in range(len(xs) - 1):
    x0, x1 = xs[i], xs[i + 1]
    if x0 <= x <= x1:
      if x1 == x0:
        return _finite_scalar(ys[i])
      t = (x - x0) / (x1 - x0)
      y = ys[i] + t * (ys[i + 1] - ys[i])
      return _finite_scalar(y)
  return None


def _lane_line_y_at(lane_line: Any, x: float) -> float | None:
  xs = tuple(getattr(lane_line, "x", ()) or ())
  ys = tuple(getattr(lane_line, "y", ()) or ())
  return _interp_scalar(x, xs, ys)


def _shadow_trace_event(*, tick: int, last_trace_tick: int, debug_trace_mode: str,
                        lat_active: bool, v_ego: float, raw_curvature: float,
                        measured_curvature: float, model_v2: Any,
                        steering_pressed: bool | None, steer_limited: bool,
                        yaw_rate: float | None) -> tuple[dict[str, Any] | None, int]:
  if debug_trace_mode != "log":
    return None, last_trace_tick
  if tick - last_trace_tick < SHADOW_TRACE_PERIOD:
    return None, last_trace_tick

  lane_change_state, lane_change_direction, lane_change_state_valid = _lane_change_state_from_model(model_v2)
  event: dict[str, Any] = {
    "shadowVersion": SHADOW_VERSION,
    "shadowOnly": True,
    "applied": False,
    "gatePass": False,
    "latActive": bool(lat_active),
    "steeringPressed": bool(steering_pressed),
    "steerLimited": bool(steer_limited),
    "laneChangeState": lane_change_state,
    "laneChangeDirection": lane_change_direction,
  }
  v = _finite_scalar(v_ego)
  if v is not None:
    event["vEgo"] = v
  raw_k = _finite_scalar(raw_curvature)
  if raw_k is not None:
    event["desiredCurvature"] = raw_k
  measured_k = _finite_scalar(measured_curvature)
  if measured_k is not None:
    event["measuredCurvature"] = measured_k

  def finish(reason: str, geometry: dict[str, Any] | None = None) -> tuple[dict[str, Any], int]:
    event["blockReason"] = reason
    event["gatePass"] = reason == "ok"
    event["strictGatePass"] = event["gatePass"]
    event["strictBlockReason"] = reason
    if geometry:
      event.update(geometry)
    return event, tick

  if model_v2 is None:
    return finish("missing_lanes")
  if not lat_active:
    return finish("lat_inactive")
  if steering_pressed:
    return finish("driver_override")
  if v is None or v < SHADOW_V_EGO_FLOOR_MPS:
    return finish("bad_speed")
  if not lane_change_state_valid or lane_change_state != LANE_CHANGE_STATE_VALUES["off"]:
    return finish("lane_change")

  probs = tuple(getattr(model_v2, "laneLineProbs", ()) or ())
  if len(probs) < 4:
    return finish("missing_lanes")
  prob_l = _finite_scalar(probs[1])
  prob_r = _finite_scalar(probs[2])
  if prob_l is None or prob_r is None:
    return finish("missing_lanes")
  if prob_l < SHADOW_PROB_THRESHOLD or prob_r < SHADOW_PROB_THRESHOLD:
    return finish("low_prob")

  lane_lines = getattr(model_v2, "laneLines", None)
  if lane_lines is None or len(lane_lines) < 4:
    return finish("missing_lanes")
  position = getattr(model_v2, "position", None)
  if position is None:
    return finish("missing_lanes")
  pos_x = tuple(getattr(position, "x", ()) or ())
  pos_y = tuple(getattr(position, "y", ()) or ())
  if len(pos_x) < 2 or len(pos_x) != len(pos_y):
    return finish("missing_lanes")

  event["probL"] = prob_l
  event["probR"] = prob_r

  geometry: dict[str, Any] = {}
  offsets: list[float] = []
  weights: list[float] = []
  for x in SHADOW_TRACE_XS:
    path_y = _interp_scalar(x, pos_x, pos_y)
    left_y = _lane_line_y_at(lane_lines[1], x)
    right_y = _lane_line_y_at(lane_lines[2], x)
    if path_y is None or left_y is None or right_y is None:
      return finish("missing_lanes")
    width = abs(left_y - right_y)
    center_y = (left_y + right_y) * 0.5
    offset = path_y - center_y
    geometry[f"offset{int(x)}"] = offset
    geometry[f"width{int(x)}"] = width
    if not (SHADOW_MIN_LANE_WIDTH <= width <= SHADOW_MAX_LANE_WIDTH):
      return finish("bad_width", geometry)
    weights.append(min(prob_l, prob_r) * (1.0 / x))
    offsets.append(offset)

  num = 0.0
  den = 0.0
  for i, x in enumerate(SHADOW_TRACE_XS):
    num += weights[i] * (x * x) * offsets[i]
    den += weights[i] * (x ** 4)
  if den == 0.0 or not math.isfinite(den) or not math.isfinite(num):
    return finish("bad_geometry", geometry)

  raw_delta_k = -2.0 * num / den
  cap = min(0.08 / (max(v, SHADOW_V_EGO_FLOOR_MPS) ** 2), 0.001)
  if not math.isfinite(cap):
    return finish("bad_speed", geometry)
  clipped_delta_k = max(-cap, min(cap, raw_delta_k))

  geometry["rawDeltaK"] = raw_delta_k
  geometry["clippedDeltaK"] = clipped_delta_k
  geometry["cap"] = cap

  yaw_curvature = _finite_scalar(yaw_rate)
  if yaw_curvature is not None:
    yaw_curvature = yaw_curvature / max(v, SHADOW_V_EGO_FLOOR_MPS)
    if math.isfinite(yaw_curvature):
      geometry["yawCurvature"] = yaw_curvature

  return finish("ok", geometry)


def _emit_shadow_trace(event: dict[str, Any] | None) -> None:
  if event is None:
    return
  try:
    cloudlog.event("lateral_path_shadow", **event)
  except Exception:
    pass


def _maybe_emit_shadow_trace(*, tick: int, last_trace_tick: int, debug_trace_mode: str,
                             lat_active: bool, v_ego: float, raw_curvature: float,
                             measured_curvature: float, model_v2: Any,
                             steering_pressed: bool | None, steer_limited: bool,
                             yaw_rate: float | None) -> int:
  """Build and emit the shadow event, swallowing any telemetry fault so controlsd is unaffected."""
  if debug_trace_mode != "log":
    return last_trace_tick
  if tick - last_trace_tick < SHADOW_TRACE_PERIOD:
    return last_trace_tick
  try:
    event, new_last = _shadow_trace_event(
      tick=tick, last_trace_tick=last_trace_tick, debug_trace_mode=debug_trace_mode,
      lat_active=lat_active, v_ego=v_ego, raw_curvature=raw_curvature,
      measured_curvature=measured_curvature, model_v2=model_v2,
      steering_pressed=steering_pressed, steer_limited=steer_limited,
      yaw_rate=yaw_rate,
    )
    _emit_shadow_trace(event)
    return new_last
  except Exception:
    try:
      cloudlog.event(
        "lateral_path_shadow",
        strictGatePass=False,
        strictBlockReason="exception",
        shadowVersion=SHADOW_VERSION,
        shadowOnly=True,
        applied=False,
        latActive=bool(lat_active),
        steeringPressed=bool(steering_pressed),
        steerLimited=bool(steer_limited),
      )
    except Exception:
      pass
    return tick


class LateralDemandAdapter:
  def __init__(self, params: Any = None):
    self._params = params
    self._pipeline = LateralDemandPipeline()
    self._tick = 0
    self._last_trace_tick = -SHADOW_TRACE_PERIOD
    self.enabled = False
    self.lane_centering_assist_enabled = False
    self.curve_memory_enabled = False
    self.debug_trace_mode = "off"
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
      self.debug_trace_mode = _debug_trace_mode(_param_string(p, "LateralDebugTraceMode"))
    except Exception:
      self.enabled = False
      self.debug_trace_mode = "off"

  def clear(self) -> None:
    self.last_result = None
    self.last_debug = {}

  def reset(self) -> None:
    try:
      self._pipeline.reset()
    except Exception:
      pass
    self.clear()

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
              steering_rate_deg: float | None = None, steer_limited: bool = False) -> float:
    """Return the processed desired curvature, or the unchanged raw curvature when disabled
    or on any fault (fail-closed)."""
    self._tick += 1
    if self._params is not None and self._tick % PARAMS_REFRESH_PERIOD == 0:
      self.refresh_params()

    self._last_trace_tick = _maybe_emit_shadow_trace(
      tick=self._tick,
      last_trace_tick=self._last_trace_tick,
      debug_trace_mode=self.debug_trace_mode,
      lat_active=lat_active,
      v_ego=v_ego,
      raw_curvature=raw_curvature,
      measured_curvature=measured_curvature,
      model_v2=model_v2,
      steering_pressed=steering_pressed,
      steer_limited=steer_limited,
      yaw_rate=yaw_rate,
    )

    if not self.enabled or model_v2 is None:
      self.reset()
      self.last_debug = self._observe_sensor_confidence(
        lat_active, v_ego, raw_curvature, measured_curvature, model_v2,
        steering_pressed, model_age_s, yaw_rate, steering_rate_deg, steer_limited,
      )
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
      inputs = replace(inputs, smooth_model_path_curvature=True, demand_jerk_smoothing_enabled=True)
      result = self._pipeline.update(inputs)
      self.last_result = result
      self.last_debug = dict(result.debug)
      return float(result.demand.processed_curvature)
    except Exception:
      self.clear()
      return raw_curvature
