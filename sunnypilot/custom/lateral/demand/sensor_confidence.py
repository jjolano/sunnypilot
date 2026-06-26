"""Shadow-only sensor/model lateral confidence metrics.

This module intentionally does not change requested curvature, path quality, gate reasons,
or torque/governor evidence. It only computes debug metrics that compare the model's current
curvature demand against fast physical signals so later route analysis can decide whether a
separate suppress-only control change is justified.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

LOW_SPEED_MIN_MPS = 5.0
MODEL_STALE_AGE_S = 0.20
STEERING_RATE_SPIKE_DEG_S = 120.0
SUPPRESS_CANDIDATE_LAT_ACCEL_DELTA = 1.0
HIGH_DISAGREEMENT_LAT_ACCEL_DELTA = 1.5
MEDIUM_DISAGREEMENT_LAT_ACCEL_DELTA = 0.7


@dataclass(frozen=True)
class SensorConfidenceInputs:
  lat_active: bool
  v_ego: float
  model_curvature: float
  measured_curvature: float
  model_path_gated: bool
  model_path_reason: str
  model_age_s: float
  steering_pressed: bool | None = None
  steering_rate_deg: float | None = None
  yaw_rate: float | None = None
  steer_limited: bool = False
  lane_change_active: bool = False
  lane_change_state_valid: bool = True
  left_blinker: bool = False
  right_blinker: bool = False


@dataclass(frozen=True)
class SensorConfidenceResult:
  available: bool
  block_reason: str
  model_measured_curvature_delta: float
  model_measured_lat_accel_delta: float
  yaw_curvature: float
  model_yaw_lat_accel_delta: float
  steering_yaw_lat_accel_delta: float
  model_yaw_lat_accel_signed_delta: float
  steering_yaw_lat_accel_signed_delta: float
  response_classification: str
  disagreement_level: str
  score: float
  suppress_candidate: bool

  def debug_dict(self) -> dict[str, float | str | bool]:
    return {
      "sensor_confidence_available": self.available,
      "sensor_confidence_block_reason": self.block_reason,
      "sensor_confidence_score": self.score,
      "sensor_disagreement_level": self.disagreement_level,
      "sensor_suppress_candidate": self.suppress_candidate,
      "sensor_response_classification": self.response_classification,
      "sensor_model_measured_curvature_delta": self.model_measured_curvature_delta,
      "sensor_model_measured_lat_accel_delta": self.model_measured_lat_accel_delta,
      "sensor_yaw_curvature": self.yaw_curvature,
      "sensor_model_yaw_lat_accel_delta": self.model_yaw_lat_accel_delta,
      "sensor_steering_yaw_lat_accel_delta": self.steering_yaw_lat_accel_delta,
      "sensor_model_yaw_lat_accel_signed_delta": self.model_yaw_lat_accel_signed_delta,
      "sensor_steering_yaw_lat_accel_signed_delta": self.steering_yaw_lat_accel_signed_delta,
    }


def evaluate_sensor_confidence(inputs: SensorConfidenceInputs) -> SensorConfidenceResult:
  v_ego = _finite(inputs.v_ego)
  model_curvature = _finite(inputs.model_curvature)
  measured_curvature = _finite(inputs.measured_curvature)
  yaw_rate = _finite(inputs.yaw_rate)
  steering_rate_deg = _finite(inputs.steering_rate_deg)

  model_measured_curvature_delta = _abs_delta(model_curvature, measured_curvature)
  model_measured_lat_accel_delta = _lat_accel_delta(model_measured_curvature_delta, v_ego)
  yaw_curvature = yaw_rate / v_ego if yaw_rate is not None and v_ego is not None and v_ego >= LOW_SPEED_MIN_MPS else None
  model_yaw_lat_accel_delta = _lat_accel_delta(_abs_delta(model_curvature, yaw_curvature), v_ego)
  steering_yaw_lat_accel_delta = _lat_accel_delta(_abs_delta(measured_curvature, yaw_curvature), v_ego)
  model_yaw_lat_accel_signed_delta = _signed_lat_accel_delta(model_curvature, yaw_curvature, v_ego)
  steering_yaw_lat_accel_signed_delta = _signed_lat_accel_delta(measured_curvature, yaw_curvature, v_ego)

  block_reason = _block_reason(inputs, v_ego, model_curvature, measured_curvature, yaw_rate, steering_rate_deg)
  available = block_reason == "ok"
  observed_delta = max(
    _or_zero(model_measured_lat_accel_delta),
    _or_zero(model_yaw_lat_accel_delta),
  )
  signed_model_yaw_delta = _or_zero(model_yaw_lat_accel_signed_delta)
  disagreement_level = _disagreement_level(observed_delta) if available else "blocked"
  score = max(0.0, min(1.0, 1.0 - observed_delta / HIGH_DISAGREEMENT_LAT_ACCEL_DELTA)) if available else 0.0
  suppress_candidate = bool(available and observed_delta >= SUPPRESS_CANDIDATE_LAT_ACCEL_DELTA)
  response_classification = _response_classification(available, observed_delta, signed_model_yaw_delta, model_curvature)

  return SensorConfidenceResult(
    available=available,
    block_reason=block_reason,
    model_measured_curvature_delta=_or_nan(model_measured_curvature_delta),
    model_measured_lat_accel_delta=_or_nan(model_measured_lat_accel_delta),
    yaw_curvature=_or_nan(yaw_curvature),
    model_yaw_lat_accel_delta=_or_nan(model_yaw_lat_accel_delta),
    steering_yaw_lat_accel_delta=_or_nan(steering_yaw_lat_accel_delta),
    model_yaw_lat_accel_signed_delta=_or_nan(model_yaw_lat_accel_signed_delta),
    steering_yaw_lat_accel_signed_delta=_or_nan(steering_yaw_lat_accel_signed_delta),
    response_classification=response_classification,
    disagreement_level=disagreement_level,
    score=float(score),
    suppress_candidate=suppress_candidate,
  )


def _block_reason(inputs: SensorConfidenceInputs, v_ego: float | None, model_curvature: float | None,
                  measured_curvature: float | None, yaw_rate: float | None,
                  steering_rate_deg: float | None) -> str:
  if not inputs.lat_active:
    return "inactive"
  if inputs.steering_pressed is not False:
    return "driver_override"
  if inputs.lane_change_active or inputs.left_blinker or inputs.right_blinker:
    return "lane_change"
  if not inputs.lane_change_state_valid:
    return "lane_change_unknown"
  if inputs.steer_limited:
    return "steer_limited"
  if inputs.model_path_gated:
    return "path_gated"
  if not math.isfinite(float(inputs.model_age_s)) or float(inputs.model_age_s) > MODEL_STALE_AGE_S:
    return "model_stale"
  if inputs.model_path_reason == "model_stale":
    return "model_stale"
  if v_ego is None or v_ego < LOW_SPEED_MIN_MPS:
    return "low_speed"
  if model_curvature is None or measured_curvature is None:
    return "nonfinite_curvature"
  if yaw_rate is None:
    return "yaw_unavailable"
  if steering_rate_deg is None:
    return "steering_rate_unavailable"
  if steering_rate_deg is not None and abs(steering_rate_deg) > STEERING_RATE_SPIKE_DEG_S:
    return "steering_rate_spike"
  return "ok"


def _finite(value: float | None) -> float | None:
  try:
    f = float(value) if value is not None else float("nan")
  except (TypeError, ValueError):
    return None
  return f if math.isfinite(f) else None


def _abs_delta(a: float | None, b: float | None) -> float | None:
  if a is None or b is None:
    return None
  return abs(a - b)


def _lat_accel_delta(curvature_delta: float | None, v_ego: float | None) -> float | None:
  if curvature_delta is None or v_ego is None:
    return None
  return curvature_delta * v_ego * v_ego


def _signed_lat_accel_delta(a: float | None, b: float | None, v_ego: float | None) -> float | None:
  if a is None or b is None or v_ego is None:
    return None
  return (a - b) * v_ego * v_ego


def _response_classification(available: bool, observed_delta: float, signed_model_yaw_delta: float, model_curvature: float | None) -> str:
  if not available:
    return "blocked"
  if observed_delta >= SUPPRESS_CANDIDATE_LAT_ACCEL_DELTA and abs(signed_model_yaw_delta) >= SUPPRESS_CANDIDATE_LAT_ACCEL_DELTA:
    requested_direction = 1.0 if model_curvature is None or model_curvature >= 0.0 else -1.0
    directional_delta = signed_model_yaw_delta * requested_direction
    if directional_delta > 0:
      return "underresponse_candidate"
    if directional_delta < 0:
      return "overresponse_candidate"
  if observed_delta >= HIGH_DISAGREEMENT_LAT_ACCEL_DELTA:
    return "high_disagreement"
  if observed_delta >= MEDIUM_DISAGREEMENT_LAT_ACCEL_DELTA:
    return "medium_disagreement"
  return "low_disagreement"


def _disagreement_level(lat_accel_delta: float) -> str:
  if lat_accel_delta >= HIGH_DISAGREEMENT_LAT_ACCEL_DELTA:
    return "high"
  if lat_accel_delta >= MEDIUM_DISAGREEMENT_LAT_ACCEL_DELTA:
    return "medium"
  return "low"


def _or_nan(value: float | None) -> float:
  return float(value) if value is not None else float("nan")


def _or_zero(value: float | None) -> float:
  return float(value) if value is not None else 0.0
