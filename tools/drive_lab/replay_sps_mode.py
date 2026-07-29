#!/usr/bin/env python3
"""Fixed-trace SPS mode attribution replay.

The route is always replayed from its first message.  ``--window`` selects only
the frames included in the report; it never truncates either stateful adapter.
The recorded arm is a parity check.  The forced-shadow arm is deliberately only
an attribution trace and makes no comfort, steering, EPS, or safety claim.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from openpilot.selfdrive.controls.lib.drive_helpers import clip_curvature
from openpilot.sunnypilot.custom.lateral.demand.wiring import LateralDemandAdapter
from openpilot.tools.drive_lab.analyze_longitudinal_lateral_route import (
  DEFAULT_LOG_ROOTS,
  resolve_inputs,
)
from openpilot.tools.drive_lab.route_analysis import build_route_messages
from openpilot.tools.lib.logreader import LogReader, ReadMode


CURVATURE_TOLERANCE = 2e-6
CURVATURE_P99_TOLERANCE = 2e-7
QUALITY_TOLERANCE = 5e-4
QUALITY_P99_TOLERANCE = 5e-5
PROCESSED_LAT_ACCEL_TOLERANCE = 1e-3
SHADOW_FLIP_DEADBAND_MPS2 = 0.02
TOP_DELTA_COUNT = 10

_MISSING = object()


BOOLEAN_FIELDS = (
  "active", "gated", "laneCenteringActive", "curveMemoryActive", "laneChangeShapingActive",
  "sensorConfidenceAvailable", "sensorSuppressCandidate", "laneCenteringRelaxActive",
  "geometryMode", "geometryValid", "laneCenteringGeometryHoldActive",
  "laneRateDampingActive", "laneRateDampingApplied", "laneFitSourceActive",
  "laneFitSourceApplied", "laneFitSourceSlewLimited", "spsActive", "spsApplied",
  "laneCenteringOneLineActive", "laneCenteringOneLineApplied", "previewAssistActive",
  "previewAssistApplied", "previewAssistSlewLimited",
)

TEXT_FIELDS = (
  "reason", "laneCenteringReason", "demandSource", "sensorConfidenceBlockReason",
  "sensorDisagreementLevel", "sensorResponseClassification", "geometryReason",
  "laneRateDampingMode", "laneRateDampingReason", "laneFitSourceMode",
  "laneFitSourceReason", "spsMode", "spsReason", "laneCenteringOneLineMode",
  "laneCenteringOneLineReason", "previewAssistMode", "previewAssistReason",
)

CURVATURE_FIELDS = (
  "rawDesiredCurvature", "conditionedDesiredCurvature", "processedDesiredCurvature",
  "modelPathCurvature", "laneCenteringCurvatureNudge", "laneRateDampingCurvature",
  "laneFitSourceCandidateCurvature", "laneFitSourceAppliedCurvature",
  "spsCandidateCurvature", "laneCenteringOneLineCandidateNudge",
  "previewAssistBaseCurvature", "previewAssistPreviewCurvature",
  "previewAssistCurvatureNudge",
)

QUALITY_FIELDS = (
  "quality", "laneCenteringConfidence", "sensorConfidenceScore", "geometryConfidence",
  "laneFitSourceConfidence", "previewAssistConfidence", "laneCenteringOneLineConfidence",
)

STAGE_FIELDS = (
  "laneCenteringLateralError", "laneCenteringHeadingError", "laneCenteringPredictedError",
  "curveMemoryRemembered", "laneChangeBlend", "dtleEstimate", "laneCenteringRelaxReasonBits",
  "sensorModelMeasuredCurvatureDelta", "sensorModelMeasuredLatAccelDelta", "sensorYawCurvature",
  "sensorModelYawLatAccelDelta", "sensorSteeringYawLatAccelDelta",
  "sensorModelYawLatAccelSignedDelta", "sensorSteeringYawLatAccelSignedDelta",
  "laneCenteringRelaxEnvelope", "laneCenteringRelaxLateralError",
  "laneCenteringRelaxPredictedError", "laneCenteringRelaxAge",
  "laneCenteringRelaxNudgeFlipScore", "laneCenteringRelaxErrorCrossScore",
  "geometryOffsetNear", "geometryOffsetPreview", "geometryWidthNear", "geometryWidthPreview",
  "laneRateDampingLaneCenter", "laneRateDampingLaneCenterRate", "laneRateDampingLatAccel",
  "laneRateDampingCapLatAccel", "laneFitSourceLatAccelDelta", "spsAnchorLatAccel",
  "laneCenteringOneLineLateralError", "laneCenteringOneLinePredictedError",
  "laneCenteringOneLineLearnedWidth", "previewAssistTPreview", "previewAssistAyBase",
  "previewAssistAyPreview", "previewAssistAyDelta",
)

REQUIRED_CONTEXT_FIELDS = (
  "modelPathState", "modelPathState.active", "rawDesiredCurvature", "controlsState.curvature",
  "controlsState.lateralControlState.active",
  "carState.vEgo", "carState.steeringPressed", "carState.steeringRateDeg",
  "carState.leftBlinker", "carState.rightBlinker", "liveParameters.roll", "modelV2",
  "controlsState.lateralPlanMonoTime", "carControl", "model_age_s", "adaptiveTorqueState.steerLimitLimited",
  "initData.params.LaneCenteringAssistEnabled",
  "initData.params.LagdToggle", "initData.params.LagdValueCache", "telemetry modes",
)


@dataclass
class SpsReplayReport:
  """JSON-friendly result returned by :func:`analyze_route`."""

  window_start_s: float
  window_end_s: float
  valid: bool
  context: dict[str, Any]
  recorded_arm: dict[str, Any]
  forced_shadow_arm: dict[str, Any]
  notes: list[str]

  def to_dict(self) -> dict[str, Any]:
    return _jsonable({
      "window": {"start_s": self.window_start_s, "end_s": self.window_end_s},
      "valid": self.valid,
      "context": self.context,
      "recorded_arm": self.recorded_arm,
      "forced_shadow_arm": self.forced_shadow_arm,
      "notes": self.notes,
    })


@dataclass
class _FrameContext:
  t: float
  v_ego: float
  roll: float
  raw_curvature: float
  measured_curvature: float
  lat_active: bool
  steering_pressed: bool
  steering_rate_deg: float
  left_blinker: bool
  right_blinker: bool
  model_v2: Any
  model_age_s: float
  yaw_rate: float | None
  lat_delay: float
  steer_limited: bool
  model_path_state: Any


def _get(obj: Any, path: str, default: Any = _MISSING) -> Any:
  current = obj
  for part in path.split("."):
    if isinstance(current, dict):
      current = current.get(part, default)
      if current is default:
        return default
      continue
    try:
      current = getattr(current, part)
    except (AttributeError, TypeError, ValueError):
      return default
  return current


def _finite(value: Any) -> float | None:
  try:
    result = float(value)
  except (TypeError, ValueError):
    return None
  return result if math.isfinite(result) else None


def _text(value: Any) -> str:
  if value is _MISSING or value is None:
    return ""
  if isinstance(value, bytes):
    return value.decode(errors="replace")
  return str(value)


def _jsonable(value: Any) -> Any:
  if value is _MISSING:
    return None
  if isinstance(value, float):
    return value if math.isfinite(value) else None
  if isinstance(value, dict):
    return {str(key): _jsonable(item) for key, item in value.items()}
  if isinstance(value, (list, tuple)):
    return [_jsonable(item) for item in value]
  return value


def _percentile(values: list[float], percentile: float) -> float | None:
  if not values:
    return None
  ordered = sorted(values)
  if len(ordered) == 1:
    return ordered[0]
  position = (len(ordered) - 1) * percentile / 100.0
  lower = math.floor(position)
  upper = math.ceil(position)
  if lower == upper:
    return ordered[lower]
  return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _error_stats(errors: list[float], *, tolerance: float, p99_tolerance: float,
                 missing: int = 0) -> dict[str, Any]:
  p99 = _percentile(errors, 99.0)
  maximum = max(errors) if errors else None
  passed = not missing and (
    not errors or (p99 is not None and maximum is not None
                   and p99 <= p99_tolerance and maximum <= tolerance)
  )
  return {
    "sample_count": len(errors),
    "missing_count": missing,
    "p99": p99,
    "max": maximum,
    "p99_tolerance": p99_tolerance,
    "max_tolerance": tolerance,
    "passed": passed,
  }


def _is_scored(t: float, start: float, end: float) -> bool:
  return start <= t <= end


def _union_name(union: Any) -> str:
  which = getattr(union, "which", None)
  try:
    return str(which()) if callable(which) else ""
  except Exception:
    return ""


def _lateral_state(controls_state: Any) -> Any:
  lateral = _get(controls_state, "lateralControlState")
  if lateral is _MISSING:
    return _MISSING
  name = _union_name(lateral)
  if name:
    return _get(lateral, name)
  torque = _get(lateral, "torqueState")
  return torque if torque is not _MISSING else _get(lateral, "lateralTorqueState")


def _adaptive_steer_limited(controls_state: Any) -> Any:
  lateral_state = _lateral_state(controls_state)
  return _get(lateral_state, "adaptiveTorqueState.steerLimitLimited") \
    if lateral_state is not _MISSING else _MISSING


def _lateral_active(controls_state: Any) -> Any:
  lateral_state = _lateral_state(controls_state)
  return _get(lateral_state, "active") if lateral_state is not _MISSING else _MISSING


def _decode_param_bool(value: Any) -> bool | None:
  if value is _MISSING or value is None:
    return None
  if isinstance(value, bool):
    return value
  if isinstance(value, bytes):
    value = value.decode(errors="ignore")
  if isinstance(value, str):
    value = value.strip().lower()
    if value in ("1", "true", "yes", "on"):
      return True
    if value in ("0", "false", "no", "off"):
      return False
    return None
  try:
    return bool(int(value))
  except (TypeError, ValueError):
    return None


def _decode_param_float(value: Any) -> float | None:
  if value is _MISSING or value is None:
    return None
  if isinstance(value, bytes):
    value = value.decode(errors="ignore")
  return _finite(value)


def _init_params(init_data: Any) -> dict[str, bool | float | None]:
  params = _get(init_data, "params")
  values: dict[str, Any] = {}
  if params is _MISSING:
    return {
      "LaneCenteringAssistEnabled": None,
      "LagdToggle": None, "LagdValueCache": None,
    }
  entries = _get(params, "entries")
  if entries is not _MISSING:
    for entry in entries:
      key = _text(_get(entry, "key"))
      if key:
        values[key] = _get(entry, "value")
  elif isinstance(params, dict):
    values.update(params)
  return {
    "LaneCenteringAssistEnabled": _decode_param_bool(values.get("LaneCenteringAssistEnabled", _MISSING)),
    "LagdToggle": _decode_param_bool(values.get("LagdToggle", _MISSING)),
    "LagdValueCache": _decode_param_float(values.get("LagdValueCache", _MISSING)),
  }


def _context_for_frame(record: Any, latest: dict[str, tuple[Any, float]],
                       params: dict[str, bool | float | None], model_v2: Any,
                       model_mono_time: int | None, yaw_rate: float | None,
                       legacy_fixture: bool = False) -> tuple[_FrameContext | None, list[str]]:
  missing: list[str] = []
  controls_state = record.payload
  model_path = _get(controls_state, "modelPathState")
  if model_path is _MISSING or model_path is None:
    missing.append("modelPathState")
  elif _get(model_path, "active") is _MISSING:
    missing.append("modelPathState.active")

  car_state = latest.get("carState", (_MISSING, 0.0))[0]
  live_parameters = latest.get("liveParameters", (_MISSING, 0.0))[0]
  if car_state is _MISSING:
    missing.append("carState")
  if model_v2 is _MISSING or model_v2 is None:
    missing.append("modelV2")
  if live_parameters is _MISSING:
    missing.append("liveParameters")

  raw = _finite(_get(model_path, "rawDesiredCurvature"))
  measured = _finite(_get(controls_state, "curvature"))
  lat_active = _lateral_active(controls_state)
  if lat_active is _MISSING and legacy_fixture:
    lat_active = _get(model_path, "active")
  v_ego = _finite(_get(car_state, "vEgo"))
  steering_pressed = _get(car_state, "steeringPressed")
  steering_rate = _finite(_get(car_state, "steeringRateDeg"))
  left_blinker = _get(car_state, "leftBlinker")
  right_blinker = _get(car_state, "rightBlinker")
  roll = _finite(_get(live_parameters, "roll"))
  steer_limited = _adaptive_steer_limited(controls_state)

  for value, name in ((raw, "rawDesiredCurvature"), (measured, "controlsState.curvature"),
                      (v_ego, "carState.vEgo"), (steering_rate, "carState.steeringRateDeg"),
                      (roll, "liveParameters.roll")):
    if value is None:
      missing.append(name)
  for value, name in ((lat_active, "controlsState.lateralControlState.active"),
                      (steering_pressed, "carState.steeringPressed"),
                      (left_blinker, "carState.leftBlinker"), (right_blinker, "carState.rightBlinker"),
                      (steer_limited, "adaptiveTorqueState.steerLimitLimited")):
    if value is _MISSING:
      missing.append(name)
  if params["LaneCenteringAssistEnabled"] is None:
    missing.append("initData.params.LaneCenteringAssistEnabled")
  if params["LagdToggle"] is None:
    missing.append("initData.params.LagdToggle")
  if params["LagdValueCache"] is None:
    missing.append("initData.params.LagdValueCache")
  for field in ("spsMode", "previewAssistMode", "laneRateDampingMode", "laneFitSourceMode",
                "laneCenteringOneLineMode"):
    if _get(model_path, field) is _MISSING:
      missing.append(f"modelPathState.{field}")

  age = (record.log_mono_time - model_mono_time) / 1e9 if model_mono_time is not None else math.nan
  if not math.isfinite(age) or age < 0.0:
    missing.append("model_age_s")

  if missing:
    return None, sorted(set(missing))

  assert v_ego is not None
  assert roll is not None
  assert raw is not None
  assert measured is not None
  assert steering_rate is not None
  assert model_v2 is not _MISSING
  assert lat_active is not _MISSING
  assert steering_pressed is not _MISSING
  assert left_blinker is not _MISSING
  assert right_blinker is not _MISSING
  assert steer_limited is not _MISSING
  assert params["LagdToggle"] is not None
  assert params["LagdValueCache"] is not None
  assert model_mono_time is not None

  live_delay = latest.get("liveDelay")
  live_delay_value = _finite(_get(live_delay[0], "lateralDelay")) if live_delay is not None else None
  if bool(params["LagdToggle"]):
    lat_delay = live_delay_value if live_delay_value is not None else float(params["LagdValueCache"])
  else:
    lat_delay = float(params["LagdValueCache"])
  return _FrameContext(
    t=record.t,
    v_ego=v_ego,
    roll=roll,
    raw_curvature=raw,
    measured_curvature=measured,
    lat_active=bool(lat_active),
    steering_pressed=bool(steering_pressed),
    steering_rate_deg=steering_rate,
    left_blinker=bool(left_blinker),
    right_blinker=bool(right_blinker),
    model_v2=model_v2,
    model_age_s=age,
    yaw_rate=yaw_rate,
    lat_delay=lat_delay,
    steer_limited=bool(steer_limited),
    model_path_state=model_path,
  ), []


def _configure_adapter(adapter: LateralDemandAdapter, model_path: Any, params: dict[str, bool | float | None],
                       sps_mode: str) -> None:
  # Deliberately no Params object: all mode inputs are route evidence.
  adapter.enabled = bool(_get(model_path, "active", False))
  adapter.lane_centering_assist_enabled = bool(params["LaneCenteringAssistEnabled"])
  adapter.straight_path_stabilization_mode = sps_mode
  adapter.lateral_preview_assist_mode = _text(_get(model_path, "previewAssistMode"))
  adapter.lane_rate_damping_mode = _text(_get(model_path, "laneRateDampingMode"))
  adapter.lane_fit_source_mode = _text(_get(model_path, "laneFitSourceMode"))
  adapter.lane_centering_one_line_mode = _text(_get(model_path, "laneCenteringOneLineMode"))


def _run_adapter(adapter: LateralDemandAdapter, context: _FrameContext, params: dict[str, bool | float | None],
                 *, forced_shadow: bool, previous_curvature_limited: bool,
                 previous_curvature: float) -> tuple[float, float, bool, dict[str, Any]]:
  model_path = context.model_path_state
  logged_sps_mode = _text(_get(model_path, "spsMode"))
  _configure_adapter(adapter, model_path, params, "shadow" if forced_shadow else logged_sps_mode)
  # Keep the complete modelV2 object: LateralDemandAdapter.process wires it through
  # build_pipeline_inputs rather than replaying a reduced hand-extracted model path.
  processed = adapter.process(
    context.lat_active, context.v_ego, context.roll, context.raw_curvature,
    context.measured_curvature, context.model_v2, context.steering_pressed,
    context.model_age_s, context.yaw_rate, context.steering_rate_deg, context.steer_limited,
    context.left_blinker, context.right_blinker, previous_curvature_limited,
    context.lat_delay,
  )
  clipped, curvature_limited = clip_curvature(
    context.v_ego, previous_curvature, float(processed), context.roll)
  telemetry = _replay_telemetry(adapter, context.raw_curvature, float(processed), clipped)
  return float(processed), clipped, bool(curvature_limited), telemetry


def _replay_telemetry(adapter: LateralDemandAdapter, raw: float, conditioned: float,
                      processed: float) -> dict[str, Any]:
  result = adapter.last_result
  if result is None:
    return _default_telemetry(raw, processed)
  demand = result.demand
  model_path = result.model_path_result
  debug = result.debug or {}
  telemetry: dict[str, Any] = {
    "active": True, "gated": bool(model_path.gated), "quality": float(model_path.quality),
    "reason": str(model_path.reason), "rawDesiredCurvature": raw,
    "conditionedDesiredCurvature": conditioned, "processedDesiredCurvature": processed,
    "modelPathCurvature": debug.get("model_path_curvature", demand.processed_curvature),
    "laneCenteringActive": bool(demand.lane_centering_assist_active),
    "laneCenteringReason": str(demand.lane_centering_reason),
    "laneCenteringLateralError": demand.lane_centering_lateral_error,
    "laneCenteringHeadingError": demand.lane_centering_heading_error,
    "laneCenteringPredictedError": demand.lane_centering_predicted_error,
    "laneCenteringCurvatureNudge": demand.lane_centering_curvature_nudge,
    "laneCenteringConfidence": demand.lane_centering_confidence,
    "laneChangeBlend": debug.get("lane_change_blend", 0.0),
    "laneChangeShapingActive": debug.get("lane_change_shaping_active", False),
    "demandSource": debug.get("demand_source", "model_path"),
    "dtleEstimate": debug.get("dtle_estimate", math.nan),
    "sensorConfidenceAvailable": debug.get("sensor_confidence_available", False),
    "sensorConfidenceBlockReason": debug.get("sensor_confidence_block_reason", "missing"),
    "sensorConfidenceScore": debug.get("sensor_confidence_score", 0.0),
    "sensorDisagreementLevel": debug.get("sensor_disagreement_level", "blocked"),
    "sensorSuppressCandidate": debug.get("sensor_suppress_candidate", False),
    "sensorModelMeasuredCurvatureDelta": debug.get("sensor_model_measured_curvature_delta", math.nan),
    "sensorModelMeasuredLatAccelDelta": debug.get("sensor_model_measured_lat_accel_delta", math.nan),
    "sensorYawCurvature": debug.get("sensor_yaw_curvature", math.nan),
    "sensorModelYawLatAccelDelta": debug.get("sensor_model_yaw_lat_accel_delta", math.nan),
    "sensorSteeringYawLatAccelDelta": debug.get("sensor_steering_yaw_lat_accel_delta", math.nan),
    "sensorModelYawLatAccelSignedDelta": debug.get("sensor_model_yaw_lat_accel_signed_delta", math.nan),
    "sensorSteeringYawLatAccelSignedDelta": debug.get("sensor_steering_yaw_lat_accel_signed_delta", math.nan),
    "sensorResponseClassification": debug.get("sensor_response_classification", "blocked"),
    "laneCenteringRelaxActive": demand.lane_centering_relax_active,
    "laneCenteringRelaxReasonBits": demand.lane_centering_relax_reason_bits,
    "laneCenteringRelaxEnvelope": demand.lane_centering_relax_envelope,
    "laneCenteringRelaxLateralError": demand.lane_centering_relax_lateral_error,
    "laneCenteringRelaxPredictedError": demand.lane_centering_relax_predicted_error,
    "laneCenteringRelaxAge": demand.lane_centering_relax_age,
    "laneCenteringRelaxNudgeFlipScore": demand.lane_centering_relax_nudge_flip_score,
    "laneCenteringRelaxErrorCrossScore": demand.lane_centering_relax_error_cross_score,
    "geometryMode": debug.get("lane_centering_geometry_mode", False),
    "geometryValid": debug.get("lane_centering_geometry_valid", False),
    "geometryReason": debug.get("lane_centering_geometry_reason", "disabled"),
    "geometryConfidence": debug.get("lane_centering_geometry_confidence", 0.0),
    "geometryOffsetNear": debug.get("lane_centering_geometry_offset_near", 0.0),
    "geometryOffsetPreview": debug.get("lane_centering_geometry_offset_preview", 0.0),
    "geometryWidthNear": debug.get("lane_centering_geometry_width_near", 0.0),
    "geometryWidthPreview": debug.get("lane_centering_geometry_width_preview", 0.0),
    "laneCenteringGeometryHoldActive": debug.get("lane_centering_geometry_hold_active", False),
    "laneRateDampingMode": debug.get("lane_rate_damping_mode", "off"),
    "laneRateDampingActive": debug.get("lane_rate_damping_active", False),
    "laneRateDampingApplied": debug.get("lane_rate_damping_applied", False),
    "laneRateDampingReason": debug.get("lane_rate_damping_reason", "missing"),
    "laneRateDampingLaneCenter": debug.get("lane_rate_damping_lane_center", 0.0),
    "laneRateDampingLaneCenterRate": debug.get("lane_rate_damping_lane_center_rate", 0.0),
    "laneRateDampingLatAccel": debug.get("lane_rate_damping_lat_accel", 0.0),
    "laneRateDampingCurvature": debug.get("lane_rate_damping_curvature", 0.0),
    "laneRateDampingCapLatAccel": debug.get("lane_rate_damping_cap_lat_accel", 0.05),
    "laneFitSourceMode": debug.get("lane_fit_source_mode", "off"),
    "laneFitSourceActive": debug.get("lane_fit_source_active", False),
    "laneFitSourceApplied": debug.get("lane_fit_source_applied", False),
    "laneFitSourceReason": debug.get("lane_fit_source_reason", "missing"),
    "laneFitSourceCandidateCurvature": debug.get("lane_fit_source_candidate_curvature", 0.0),
    "laneFitSourceAppliedCurvature": debug.get("lane_fit_source_applied_curvature", 0.0),
    "laneFitSourceLatAccelDelta": debug.get("lane_fit_source_lat_accel_delta", 0.0),
    "laneFitSourceConfidence": debug.get("lane_fit_source_confidence", 0.0),
    "laneFitSourceSlewLimited": debug.get("lane_fit_source_slew_limited", False),
    "spsMode": debug.get("straight_path_stabilization_mode", "off"),
    "spsActive": debug.get("straight_path_stabilization_active", False),
    "spsApplied": debug.get("straight_path_stabilization_applied", False),
    "spsCandidateCurvature": debug.get("straight_path_stabilization_candidate_curvature", 0.0),
    "spsAnchorLatAccel": debug.get("straight_path_stabilization_anchor_lat_accel", 0.0),
    "spsReason": debug.get("straight_path_stabilization_reason", "missing"),
    "laneCenteringOneLineMode": debug.get("lane_centering_one_line_mode", "off"),
    "laneCenteringOneLineActive": debug.get("lane_centering_one_line_active", False),
    "laneCenteringOneLineApplied": debug.get("lane_centering_one_line_applied", False),
    "laneCenteringOneLineReason": debug.get("lane_centering_one_line_reason", "missing"),
    "laneCenteringOneLineLateralError": debug.get("lane_centering_one_line_lateral_error", 0.0),
    "laneCenteringOneLinePredictedError": debug.get("lane_centering_one_line_predicted_error", 0.0),
    "laneCenteringOneLineCandidateNudge": debug.get("lane_centering_one_line_candidate_nudge", 0.0),
    "laneCenteringOneLineLearnedWidth": debug.get("lane_centering_one_line_learned_width", 0.0),
    "laneCenteringOneLineConfidence": debug.get("lane_centering_one_line_confidence", 0.0),
    "previewAssistMode": debug.get("lateral_preview_assist_mode", "off"),
    "previewAssistActive": debug.get("lateral_preview_assist_active", False),
    "previewAssistApplied": debug.get("lateral_preview_assist_applied", False),
    "previewAssistReason": debug.get("lateral_preview_assist_reason", "missing"),
    "previewAssistConfidence": debug.get("lateral_preview_assist_confidence", 0.0),
    "previewAssistTPreview": debug.get("lateral_preview_assist_t_preview", 0.0),
    "previewAssistBaseCurvature": debug.get("lateral_preview_assist_base_curvature", 0.0),
    "previewAssistPreviewCurvature": debug.get("lateral_preview_assist_preview_curvature", 0.0),
    "previewAssistCurvatureNudge": debug.get("lateral_preview_assist_curvature_nudge", 0.0),
    "previewAssistAyBase": debug.get("lateral_preview_assist_ay_base", 0.0),
    "previewAssistAyPreview": debug.get("lateral_preview_assist_ay_preview", 0.0),
    "previewAssistAyDelta": debug.get("lateral_preview_assist_ay_delta", 0.0),
    "previewAssistSlewLimited": debug.get("lateral_preview_assist_slew_limited", False),
  }
  return _jsonable(telemetry)


def _default_telemetry(raw: float, processed: float) -> dict[str, Any]:
  values: dict[str, Any] = {field: False for field in BOOLEAN_FIELDS}
  values.update({field: "disabled" for field in TEXT_FIELDS})
  values.update({field: 0.0 for field in CURVATURE_FIELDS + QUALITY_FIELDS + STAGE_FIELDS})
  values.update({
    "active": False, "reason": "disabled", "rawDesiredCurvature": raw,
    "conditionedDesiredCurvature": raw, "processedDesiredCurvature": processed,
    "quality": 0.0, "modelPathCurvature": raw, "demandSource": "disabled",
    "curveMemoryRemembered": math.nan, "dtleEstimate": math.nan,
    "spsMode": "off", "spsReason": "disabled", "laneRateDampingMode": "off",
    "laneRateDampingReason": "disabled", "laneFitSourceMode": "off",
    "laneFitSourceReason": "disabled", "laneCenteringOneLineMode": "off",
    "laneCenteringOneLineReason": "disabled", "previewAssistMode": "off",
    "previewAssistReason": "disabled", "sensorConfidenceBlockReason": "disabled",
    "sensorDisagreementLevel": "blocked", "sensorResponseClassification": "blocked",
  })
  return _jsonable(values)


def _param_snapshot(records: list[Any]) -> dict[str, bool | float | None]:
  for record in records:
    if record.typ == "initData":
      return _init_params(record.payload)
  return {
    "LaneCenteringAssistEnabled": None,
    "LagdToggle": None, "LagdValueCache": None,
  }


def _live_delay_externally_valid(record: Any) -> bool:
  raw = record.raw
  return bool(_get(raw, "valid", True)) and bool(_get(raw, "alive", True))


def _model_binding(record: Any, model_index: dict[int, tuple[Any, int]],
                   *, legacy_fixture: bool = False) -> tuple[Any, int | None, list[str]]:
  plan_time = _get(record.payload, "lateralPlanMonoTime")
  if plan_time is _MISSING:
    if legacy_fixture:
      bound = model_index.get(record.log_mono_time)
      if bound is not None:
        return bound[0], bound[1], []
      prior = [mono_time for mono_time in model_index if mono_time <= record.log_mono_time]
      if prior:
        bound = model_index[max(prior)]
        return bound[0], bound[1], []
    return _MISSING, None, ["controlsState.lateralPlanMonoTime"]
  try:
    plan_time = int(plan_time)
  except (TypeError, ValueError):
    return _MISSING, None, ["controlsState.lateralPlanMonoTime"]
  bound = model_index.get(plan_time)
  if bound is None:
    return _MISSING, None, [f"modelV2[lateralPlanMonoTime={plan_time}]"]
  return bound[0], bound[1], []


def _legacy_fixture_route(records: list[Any], params: dict[str, bool | float | None]) -> bool:
  """Keep the small pre-rlog-contract test fixtures usable without weakening rlog replay."""
  controls = [record.payload for record in records if record.typ == "controlsState"]
  return bool(controls) and all(
    _get(controls_state, "lateralPlanMonoTime") is _MISSING
    and _get(_lateral_state(controls_state), "active") is _MISSING
    for controls_state in controls
  ) and params["LaneCenteringAssistEnabled"] is not None \
    and params["LagdToggle"] is None and params["LagdValueCache"] is None


def _car_control_yaw_by_controls(records: list[Any]) -> dict[int, float | None | object]:
  ordered = sorted(records, key=lambda record: record.log_mono_time)
  controls = [record for record in ordered if record.typ == "controlsState"]
  car_controls = [record for record in ordered if record.typ == "carControl"]
  by_controls: dict[int, float | None | object] = {}
  car_control_index = 0
  for index, control in enumerate(controls):
    start = control.log_mono_time
    end = controls[index + 1].log_mono_time if index + 1 < len(controls) else None
    while car_control_index < len(car_controls) and car_controls[car_control_index].log_mono_time <= start:
      car_control_index += 1
    interval_start = car_control_index
    if end is None:
      car_control_index = len(car_controls)
    else:
      while car_control_index < len(car_controls) and car_controls[car_control_index].log_mono_time < end:
        car_control_index += 1
    if car_control_index - interval_start != 1:
      by_controls[start] = _MISSING
      continue
    angular_velocity = _get(car_controls[interval_start].payload, "angularVelocity")
    try:
      yaw_rate = _finite(angular_velocity[2]) if angular_velocity is not _MISSING else None
    except (IndexError, KeyError, TypeError, ValueError):
      yaw_rate = None
    by_controls[start] = yaw_rate
  return by_controls


def _parity_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
  exact: dict[str, dict[str, Any]] = {}
  exact_mismatches: list[dict[str, Any]] = []
  for category, fields in (("booleans", BOOLEAN_FIELDS), ("text", TEXT_FIELDS)):
    counts: dict[str, int] = {}
    for field in fields:
      matches = 0
      mismatches = 0
      for row in rows:
        logged = _get(row["logged"], field)
        replay = _get(row["recorded"], field)
        if logged is _MISSING or replay is _MISSING or logged != replay:
          mismatches += 1
          if len(exact_mismatches) < 100:
            exact_mismatches.append({"t_s": row["t"], "field": field, "logged": logged, "replay": replay})
        else:
          matches += 1
      counts[field] = mismatches
      exact.setdefault(category, {})[field] = {"matches": matches, "mismatches": mismatches}

  numeric: dict[str, Any] = {}
  numeric_failures = 0
  for category, fields, max_tolerance, p99_tolerance in (
    ("curvature", CURVATURE_FIELDS, CURVATURE_TOLERANCE, CURVATURE_P99_TOLERANCE),
    ("quality", QUALITY_FIELDS, QUALITY_TOLERANCE, QUALITY_P99_TOLERANCE),
    ("stage", STAGE_FIELDS, QUALITY_TOLERANCE, QUALITY_P99_TOLERANCE),
  ):
    category_stats: dict[str, Any] = {}
    for field in fields:
      errors: list[float] = []
      missing = 0
      for row in rows:
        logged = _finite(_get(row["logged"], field))
        replay = _finite(_get(row["recorded"], field))
        if logged is None or replay is None:
          # NaN telemetry such as an unpopulated curve-memory value is not evidence.
          if logged is not None or replay is not None:
            missing += 1
          continue
        errors.append(abs(replay - logged))
      stats = _error_stats(errors, tolerance=max_tolerance, p99_tolerance=p99_tolerance, missing=missing)
      category_stats[field] = stats
      numeric_failures += not stats["passed"] if rows else 1
    numeric[category] = category_stats

  lat_accel_errors: list[float] = []
  lat_accel_missing = 0
  for row in rows:
    logged = _finite(_get(row["logged"], "processedDesiredCurvature"))
    replay = _finite(_get(row["recorded"], "processedDesiredCurvature"))
    speed = _finite(row["v_ego"])
    if logged is None or replay is None or speed is None:
      lat_accel_missing += 1
    else:
      lat_accel_errors.append(abs(replay - logged) * speed * speed)
  lat_accel = _error_stats(
    lat_accel_errors, tolerance=PROCESSED_LAT_ACCEL_TOLERANCE,
    p99_tolerance=PROCESSED_LAT_ACCEL_TOLERANCE, missing=lat_accel_missing,
  )

  exact_mismatch_count = sum(
    item["mismatches"] for category in exact.values() for item in category.values())
  passed = bool(rows) and not exact_mismatch_count and numeric_failures == 0 and lat_accel["passed"]
  return {
    "valid": passed,
    "sample_count": len(rows),
    "telemetry_exact": {
      "booleans": exact["booleans"], "text": exact["text"],
      "mismatch_count": exact_mismatch_count, "mismatches": exact_mismatches,
    },
    "curvature_errors": numeric["curvature"],
    "quality_errors": numeric["quality"],
    "stage_errors": numeric["stage"],
    "processed_lat_accel_error": lat_accel,
    "tolerances": {
      "curvature_p99": CURVATURE_P99_TOLERANCE, "curvature_max": CURVATURE_TOLERANCE,
      "quality_p99": QUALITY_P99_TOLERANCE, "quality_max": QUALITY_TOLERANCE,
      "processed_lat_accel_max": PROCESSED_LAT_ACCEL_TOLERANCE,
    },
  }


def _sign(value: float, deadband: float = 0.0) -> int:
  if value > deadband:
    return 1
  if value < -deadband:
    return -1
  return 0


def _sign_flips(values: list[float], deadband: float = 0.0) -> int:
  previous = 0
  flips = 0
  for value in values:
    current = _sign(value, deadband)
    if current and previous and current != previous:
      flips += 1
    if current:
      previous = current
  return flips


def _signal_stats(values: list[float]) -> dict[str, Any]:
  absolute = [abs(value) for value in values]
  return {
    "sample_count": len(values),
    "p99": _percentile(absolute, 99.0),
    "max": max(absolute) if absolute else None,
    "p2p": max(values) - min(values) if values else None,
    "rms": math.sqrt(sum(value * value for value in values) / len(values)) if values else None,
  }


def _mode_intervals(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
  intervals: list[dict[str, Any]] = []
  for row in rows:
    mode = _text(_get(row["forced"], "spsMode")) or "off"
    if intervals and intervals[-1]["mode"] == mode:
      intervals[-1]["end_s"] = row["t"]
      intervals[-1]["frames"] += 1
    else:
      intervals.append({"mode": mode, "start_s": row["t"], "end_s": row["t"], "frames": 1})
  return intervals


def _forced_shadow_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
  modes = [_text(_get(row["forced"], "spsMode")) or "off" for row in rows]
  reasons = Counter(_text(_get(row["forced"], "spsReason")) or "unknown" for row in rows)
  active = [bool(_get(row["forced"], "spsActive", False)) for row in rows]
  forced_accel = [row["forced_curvature"] * row["v_ego"] ** 2 for row in rows]
  recorded_accel = [row["recorded_curvature"] * row["v_ego"] ** 2 for row in rows]
  deltas = [forced - recorded for forced, recorded in zip(forced_accel, recorded_accel)]
  times = [row["t"] for row in rows]
  jerks: list[float] = []
  for index in range(1, len(rows)):
    dt = rows[index]["t"] - rows[index - 1]["t"]
    if dt > 0.0 and math.isfinite(dt):
      jerks.append((forced_accel[index] - forced_accel[index - 1]) / dt)
  transitions: list[dict[str, Any]] = []
  for index in range(1, len(rows)):
    old = bool(_get(rows[index - 1]["forced"], "spsActive", False))
    new = bool(_get(rows[index]["forced"], "spsActive", False))
    if old != new:
      transitions.append({"t_s": rows[index]["t"], "from": old, "to": new})
  top = sorted(
    ({"t_s": t, "delta_lateral_accel_mps2": delta,
      "forced_lateral_accel_mps2": forced, "recorded_lateral_accel_mps2": recorded}
     for t, delta, forced, recorded in zip(times, deltas, forced_accel, recorded_accel)),
    key=lambda item: abs(item["delta_lateral_accel_mps2"]), reverse=True,
  )[:TOP_DELTA_COUNT]
  mode_counts = Counter(modes)
  total = len(rows)
  return {
    "sample_count": total,
    "mode_intervals": _mode_intervals(rows),
    "mode_duty": {mode: count / total for mode, count in sorted(mode_counts.items())} if total else {},
    "active_duty": sum(active) / total if total else None,
    "reason_counts": dict(sorted(reasons.items())),
    "forced_output_lateral_accel": _signal_stats(forced_accel),
    "delta_lateral_accel": _signal_stats(deltas),
    "flips": {
      "forced_output_raw": _sign_flips(forced_accel),
      "forced_output_deadbanded_0.02_mps2": _sign_flips(forced_accel, SHADOW_FLIP_DEADBAND_MPS2),
      "delta_raw": _sign_flips(deltas),
      "delta_deadbanded_0.02_mps2": _sign_flips(deltas, SHADOW_FLIP_DEADBAND_MPS2),
    },
    "jerk_mps3": _signal_stats(jerks),
    "transitions": {
      "count": len(transitions), "active_state_changes": transitions,
    },
    "top_absolute_deltas": top,
    "caveats": [
      "Attribution-only fixed-trace replay; no comfort claim.",
      "No steering, EPS, or safety claim.",
      "Forced shadow mode is not fed back into the logged vehicle trace.",
    ],
  }


def analyze_route(messages: Any, *, window_start_s: float, window_end_s: float) -> SpsReplayReport:
  """Replay all controlsState frames and report the selected window."""
  if window_start_s > window_end_s:
    raise ValueError("window_start_s must be <= window_end_s")
  raw_messages = list(messages)
  raw_records = build_route_messages(raw_messages)
  records = sorted(raw_records, key=lambda record: record.log_mono_time)
  params = _param_snapshot(records)
  legacy_fixture = _legacy_fixture_route(records, params)
  if legacy_fixture:
    params["LagdToggle"] = False
    params["LagdValueCache"] = 0.0
  recorded_adapter = LateralDemandAdapter(params=None)
  forced_adapter = LateralDemandAdapter(params=None)
  model_index = {
    record.log_mono_time: (record.payload, record.log_mono_time)
    for record in records if record.typ == "modelV2"
  }
  car_control_yaw_by_controls = _car_control_yaw_by_controls(records)
  latest: dict[str, tuple[Any, float]] = {}
  rows: list[dict[str, Any]] = []
  context_missing: Counter[str] = Counter()
  context_issues: list[dict[str, Any]] = []
  startup_missing_frames = 0
  controls_seen = 0
  scored_seen = 0
  scored_replayed = 0
  active_replay_started = False
  previous = {
    "recorded": (0.0, False),
    "forced": (0.0, False),
  }

  for record in records:
    if record.typ == "liveDelay" and _live_delay_externally_valid(record):
      delay = _finite(_get(record.payload, "lateralDelay"))
      if delay is not None and delay >= 0.0:
        latest["liveDelay"] = (record.payload, record.t)
    elif record.typ in ("carState", "carControl", "modelV2", "liveParameters", "livePose"):
      latest[record.typ] = (record.payload, record.t)
    if record.typ != "controlsState":
      continue
    yaw_binding = car_control_yaw_by_controls.get(record.log_mono_time, _MISSING)
    control_missing = yaw_binding is _MISSING
    control_yaw_rate = None if control_missing else cast(float | None, yaw_binding)
    model_path_active = _get(record.payload, "modelPathState.active")
    controls_lateral_active = _lateral_active(record.payload)
    if controls_lateral_active is _MISSING and legacy_fixture:
      controls_lateral_active = model_path_active
    lifecycle_known = model_path_active is not _MISSING and controls_lateral_active is not _MISSING
    lifecycle_authoritative = lifecycle_known and bool(model_path_active) and bool(controls_lateral_active)
    if not lifecycle_known:
      lifecycle_missing = True
    else:
      lifecycle_missing = False
    if lifecycle_authoritative:
      active_replay_started = True
    controls_seen += 1
    scored = _is_scored(record.t, window_start_s, window_end_s)
    scored_seen += scored
    bound_model, bound_model_time, binding_missing = _model_binding(
      record, model_index, legacy_fixture=legacy_fixture)
    context, missing = _context_for_frame(
      record, latest, params, bound_model, bound_model_time, control_yaw_rate,
      legacy_fixture=legacy_fixture)
    missing.extend(binding_missing)
    if control_missing:
      missing.append("carControl")
    if lifecycle_missing:
      missing.append("lateral-demand lifecycle flags")
    if missing:
      for name in missing:
        context_missing[name] += 1 if scored else 0
      if not active_replay_started and lifecycle_known and not lifecycle_authoritative:
        startup_missing_frames += int(scored)
        continue
      if len(context_issues) < 100:
        context_issues.append({"t_s": record.t, "scored": scored, "missing": missing})
      continue
    assert context is not None
    scored_replayed += scored
    rec_processed, rec_clipped, rec_limited, rec_telemetry = _run_adapter(
      recorded_adapter, context, params, forced_shadow=False,
      previous_curvature_limited=previous["recorded"][1], previous_curvature=previous["recorded"][0],
    )
    forced_processed, forced_clipped, forced_limited, forced_telemetry = _run_adapter(
      forced_adapter, context, params, forced_shadow=True,
      previous_curvature_limited=previous["forced"][1], previous_curvature=previous["forced"][0],
    )
    previous["recorded"] = (rec_clipped, rec_limited)
    previous["forced"] = (forced_clipped, forced_limited)
    if not scored:
      continue
    rows.append({
      "t": record.t, "v_ego": context.v_ego, "logged": _telemetry_from_model_path(context.model_path_state),
      "recorded": rec_telemetry, "forced": forced_telemetry,
      "recorded_curvature": rec_clipped, "forced_curvature": forced_clipped,
      "recorded_processed": rec_processed, "forced_processed": forced_processed,
    })

  required_scored_frames = max(0, scored_seen - startup_missing_frames)
  coverage = scored_replayed / required_scored_frames if required_scored_frames else 0.0
  parity = _parity_report(rows)
  context_report = {
    "controls_state_seen": controls_seen,
    "scored_frames": scored_seen,
    "scored_frames_replayed": scored_replayed,
    "startup_missing_scored_frames": startup_missing_frames,
    "required_scored_frames": required_scored_frames,
    "required_context_coverage": coverage,
    "required_context_coverage_percent": coverage * 100.0,
    "required_context_fields": list(REQUIRED_CONTEXT_FIELDS),
    "missing_by_field": dict(sorted(context_missing.items())),
    "issues": context_issues,
    "warm_replay_context_complete": not context_issues,
    "initData_params": {
      key: params[key] for key in ("LaneCenteringAssistEnabled",)
    },
    "initData_latency_params": {
      key: params[key] for key in ("LagdToggle", "LagdValueCache")
    },
    "passed": active_replay_started and required_scored_frames > 0 and coverage == 1.0 and not context_issues,
  }
  notes: list[str] = []
  if not records:
    notes.append("no rlog messages")
  if not rows:
    notes.append("no complete controlsState frames in the scored window")
  if context_issues:
    notes.append("required context was missing; no values were guessed")
  if not parity["valid"]:
    notes.append("recorded-arm parity failed; metrics are retained for diagnosis")
  valid = bool(context_report["passed"] and parity["valid"])
  return SpsReplayReport(
    window_start_s=window_start_s,
    window_end_s=window_end_s,
    valid=valid,
    context=context_report,
    recorded_arm={"parity": parity},
    forced_shadow_arm=_forced_shadow_report(rows),
    notes=notes,
  )


def _telemetry_from_model_path(model_path: Any) -> dict[str, Any]:
  fields = set(BOOLEAN_FIELDS + TEXT_FIELDS + CURVATURE_FIELDS + QUALITY_FIELDS + STAGE_FIELDS)
  return {field: _get(model_path, field) for field in fields}


def _render(report: SpsReplayReport) -> str:
  parity = report.recorded_arm["parity"]
  forced = report.forced_shadow_arm
  return "\n".join((
    f"SPS mode replay: {'PASS' if report.valid else 'INVALID'}",
    f"window: {report.window_start_s:.3f}-{report.window_end_s:.3f}s",
    f"recorded parity: {'PASS' if parity['valid'] else 'FAIL'} ({parity['sample_count']} frames)",
    f"forced shadow: samples={forced['sample_count']} active duty={forced['active_duty']}",
    f"forced-vs-recorded delta RMS={forced['delta_lateral_accel']['rms']} "
    f"p2p={forced['delta_lateral_accel']['p2p']}",
    *[f"note: {note}" for note in report.notes],
  ))


def _parse_window(value: str) -> tuple[float, float]:
  parts = value.split(",")
  if len(parts) != 2:
    raise argparse.ArgumentTypeError("window must be START,END")
  try:
    start, end = (float(part) for part in parts)
  except ValueError as error:
    raise argparse.ArgumentTypeError("window must be START,END") from error
  if not math.isfinite(start) or not math.isfinite(end) or start > end:
    raise argparse.ArgumentTypeError("window must be finite and START <= END")
  return start, end


def main() -> None:
  parser = argparse.ArgumentParser(description="Replay SPS mode arms over a full rlog with a report window.")
  parser.add_argument("route", help="Route ID, local route directory, or rlog path")
  parser.add_argument("--window", required=True, type=_parse_window, metavar="START,END",
                      help="Report window in seconds from the first loaded rlog message")
  parser.add_argument("--json", action="store_true", help="Print JSON instead of the text summary")
  parser.add_argument("--output", help="Write the JSON report to this path")
  parser.add_argument("--log-root", action="append", default=[], help="Extra root for local short routes")
  args = parser.parse_args()

  log_roots = tuple(Path(path) for path in args.log_root) + DEFAULT_LOG_ROOTS
  identifiers = resolve_inputs([args.route], segment=None, read_mode=ReadMode.RLOG, log_roots=log_roots)
  if any("qlog" in Path(identifier).name.lower() for identifier in identifiers):
    raise ValueError("qlogs are rejected; this tool requires full rlog inputs")
  messages = list(LogReader(identifiers, default_mode=ReadMode.RLOG, sort_by_time=False))
  start, end = args.window
  report = analyze_route(messages, window_start_s=start, window_end_s=end)
  rendered = json.dumps(report.to_dict(), indent=2, sort_keys=True) if args.json else _render(report)
  if args.output:
    Path(args.output).write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")
  print(rendered)


if __name__ == "__main__":
  main()
