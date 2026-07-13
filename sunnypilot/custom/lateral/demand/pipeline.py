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

import collections
import math
from dataclasses import dataclass, field, replace
from typing import Any
from collections.abc import Sequence

import numpy as np

from openpilot.common.realtime import DT_CTRL
from openpilot.common.swaglog import cloudlog
from openpilot.sunnypilot.custom.lateral.demand.types import (
  DEMAND_SOURCE_FALLBACK_MEASURED,
  DEMAND_SOURCE_LANE_FIT,
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
from openpilot.sunnypilot.custom.lateral.demand.lane_geometry import (
  LANE_GEOMETRY_SAMPLE_XS,
  evaluate_lane_geometry,
)
from openpilot.sunnypilot.custom.lateral.demand.model_path_processor import (
  ModelPathProcessor,
  ModelPathProcessorInputs,
  ModelPathProcessorResult,
)
from openpilot.sunnypilot.custom.lateral.demand.preview import (
  PreviewAssistTracker,
  inactive_preview_assist_result,
)
from openpilot.sunnypilot.custom.lateral.demand.sensor_confidence import (
  SensorConfidenceInputs,
  evaluate_sensor_confidence,
)

# turn_curvature_sign convention (matches legacy: TurnDirection.turnRight=1, turnLeft=2)
TURN_DIRECTION_RIGHT = 1
TURN_DIRECTION_LEFT = 2
LANE_CHANGE_STATE_OFF = 0
LANE_RATE_DAMPING_MODES = frozenset(("off", "shadow", "apply"))
LANE_RATE_DAMPING_RATE_WINDOW_S = 0.5
LANE_RATE_DAMPING_SMOOTH_TAU_S = 0.4
LANE_RATE_DAMPING_DEADBAND_MPS = 0.005
LANE_RATE_DAMPING_GAIN = 1.0
LANE_RATE_DAMPING_CAP_LAT_ACCEL = 0.05
LANE_RATE_DAMPING_RELEASE_LAT_JERK = 0.30  # m/s^3; on/off edges decay instead of stepping
LANE_RATE_DAMPING_MIN_SPEED = 12.0
LANE_RATE_DAMPING_MIN_PATH_QUALITY = 0.85
LANE_RATE_DAMPING_MAX_MODEL_LAT_ACCEL = 0.6
LANE_RATE_DAMPING_V2_EPS = 1e-6
LANE_RATE_DAMPING_HARD_BLOCK_REASONS = frozenset((
  "disabled", "inactive", "invalid", "lane_change_unknown", "lane_change", "blinker", "driver_override",
))

LANE_FIT_SOURCE_MODES = frozenset(("off", "shadow", "apply"))
LANE_FIT_SOURCE_MIN_SPEED = 15.0
LANE_FIT_SOURCE_MIN_PATH_QUALITY = 0.85
LANE_FIT_SOURCE_MIN_GEOMETRY_CONFIDENCE = 0.6
LANE_FIT_SOURCE_MAX_LAT_ACCEL = 0.6
LANE_FIT_SOURCE_MAX_LAT_ACCEL_DELTA = 0.35
LANE_FIT_SOURCE_SIGN_CONFLICT_LAT_ACCEL = 0.05
LANE_FIT_SOURCE_MIN_PERSIST_S = 0.2
LANE_FIT_SOURCE_SLEW_LAT_JERK = 1.2
LANE_FIT_SOURCE_RELEASE_SLEW_SCALE = 0.5


def sanitize_lane_rate_damping_mode(mode: object) -> str:
  mode_s = str(mode).strip().lower()
  return mode_s if mode_s in LANE_RATE_DAMPING_MODES else "off"


def sanitize_lane_fit_source_mode(mode: object) -> str:
  mode_s = str(mode).strip().lower()
  return mode_s if mode_s in LANE_FIT_SOURCE_MODES else "off"


def _finite_float(value: Any) -> float | None:
  try:
    value = float(value)
  except (TypeError, ValueError):
    return None
  return value if math.isfinite(value) else None


def _lane_center_y0(left_lane_y0: float | None, right_lane_y0: float | None) -> tuple[float, bool]:
  left = _finite_float(left_lane_y0)
  right = _finite_float(right_lane_y0)
  if left is None or right is None:
    return 0.0, False
  return (left + right) / 2.0, True


def _deadband(value: float, threshold: float) -> float:
  if abs(value) <= threshold:
    return 0.0
  return value - math.copysign(threshold, value)


def _finite_array(values: Sequence[Any]) -> list[float] | None:
  result: list[float] = []
  for value in values:
    f = _finite_float(value)
    if f is None:
      return None
    result.append(f)
  return result


def _lane_line_y_at(lane_line: Any, x: float) -> float | None:
  raw_xs = getattr(lane_line, "x", None) or ()
  raw_ys = getattr(lane_line, "y", None) or ()
  xs = _finite_array(raw_xs)
  ys = _finite_array(raw_ys)
  if xs is None or ys is None or len(xs) < 2 or len(xs) != len(ys):
    return None
  if x < xs[0] or x > xs[-1]:
    return None
  return float(np.interp(x, xs, ys))


def _lane_fit_candidate_curvature(lane_lines: Sequence[Any]) -> float | None:
  if lane_lines is None or len(lane_lines) < 3:
    return None
  left_lane = lane_lines[1]
  right_lane = lane_lines[2]
  if left_lane is None or right_lane is None:
    return None

  sample_xs = np.asarray(LANE_GEOMETRY_SAMPLE_XS, dtype=float)
  center_ys: list[float] = []
  for x in sample_xs:
    left_y = _lane_line_y_at(left_lane, float(x))
    right_y = _lane_line_y_at(right_lane, float(x))
    if left_y is None or right_y is None:
      return None
    center_y = (left_y + right_y) * 0.5
    if not math.isfinite(center_y):
      return None
    center_ys.append(center_y)

  try:
    coeffs = np.polyfit(sample_xs, np.asarray(center_ys, dtype=float), deg=2)
  except (TypeError, ValueError, np.linalg.LinAlgError):
    return None

  curvature = float(2.0 * coeffs[0])
  return curvature if math.isfinite(curvature) else None


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
  lane_line_stds: Sequence[float] = ()
  lane_lines: Sequence[Any] = ()
  frame_drop_perc: float = 0.0
  model_age_s: float = 0.0
  model_frame_id: int = 0  # 0 = unknown; enables per-model-frame memoization downstream
  yaw_rate: float | None = None
  steering_rate_deg: float | None = None
  steer_limited: bool = False
  straight_path_stabilization_mode: str = "off"
  lane_rate_damping_mode: str = "off"
  lane_fit_source_mode: str = "off"
  lane_centering_one_line_mode: str = "off"
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
  lat_delay: float = 0.0
  lateral_preview_assist_mode: str = "off"
  # passed through (downstream clip result)
  curvature_limited: bool = False


@dataclass(frozen=True)
class LateralDemandPipelineResult:
  demand: ProcessedLateralDemand
  model_path_result: ModelPathProcessorResult
  debug: dict[str, float | str | bool] = field(default_factory=dict)


@dataclass(frozen=True)
class LaneRateDampingResult:
  mode: str
  active: bool
  applied: bool
  reason: str
  lane_center: float
  lane_center_rate: float
  lat_accel: float
  curvature: float
  cap_lat_accel: float


@dataclass(frozen=True)
class LaneFitSourceResult:
  mode: str
  active: bool
  applied: bool
  reason: str
  candidate_curvature: float
  applied_curvature: float
  lat_accel_delta: float
  confidence: float
  slew_limited: bool


class _AyReleaseSlew:
  """Bounded-rate approach of an additive lat-accel nudge, both directions.

  Transient releases decay at the entry rate instead of vanishing in one frame (route
  274: instant nudge removal was a top lateral-jerk source). Callers reset on explicit
  driver/mode hard stops. Pure state: feed the target and add the returned value."""

  def __init__(self, dt: float, rate_lat_jerk: float) -> None:
    self.dt = max(float(dt), 1e-3)
    self.rate = float(rate_lat_jerk)
    self.value = 0.0

  def reset(self) -> None:
    self.value = 0.0

  def update(self, target_ay: float) -> float:
    if not math.isfinite(target_ay):
      target_ay = 0.0
    step = self.rate * self.dt
    self.value += float(np.clip(target_ay - self.value, -step, step))
    return self.value


MODEL_STEP_BLEND_FRAMES = 5           # 20 Hz model cadence at the 100 Hz control rate
MODEL_STEP_BLEND_MAX_STEP_AY = 0.5    # m/s^2; larger steps pass through (urgent demand)


class _ModelStepBlender:
  """Linear cross-fade of the 20 Hz raw model-curvature steps at 100 Hz.

  controlsd holds ``action.desiredCurvature`` constant between model frames, so every new
  model frame lands as a step (route 274: 24% of high-jerk episodes were raw model jumps).
  Fading the step over the inter-frame gap (~50 ms) bounds the single-frame jerk with a
  mean added lag of ~25 ms on the step itself. Big steps pass through unfaded — comfort
  shaping must never delay an urgent demand; ``clip_curvature`` still governs downstream.
  """

  def __init__(self) -> None:
    self.reset()

  def reset(self) -> None:
    self._last_frame_id = 0
    self._output: float | None = None
    self._target: float = 0.0
    self._step: float = 0.0

  def update(self, raw_curvature: float, model_frame_id: int, v_ego: float, lat_active: bool) -> float:
    if not lat_active or model_frame_id <= 0 or not math.isfinite(raw_curvature) or not math.isfinite(v_ego):
      self.reset()
      return raw_curvature
    if self._output is None:
      self._last_frame_id = model_frame_id
      self._output = raw_curvature
      self._target = raw_curvature
      self._step = 0.0
      return raw_curvature
    if model_frame_id != self._last_frame_id:
      self._last_frame_id = model_frame_id
      step_ay = abs(raw_curvature - self._output) * max(v_ego * v_ego, 1.0)
      if step_ay > MODEL_STEP_BLEND_MAX_STEP_AY:
        self._output = raw_curvature   # urgent demand: no fade
      self._target = raw_curvature
      self._step = (raw_curvature - self._output) / MODEL_STEP_BLEND_FRAMES
    if self._output != self._target:
      self._output += self._step
      if (self._step >= 0.0 and self._output >= self._target) or (self._step < 0.0 and self._output <= self._target):
        self._output = self._target
    return float(self._output)


class _LaneRateDampingTracker:
  def __init__(self, dt: float) -> None:
    self.dt = max(float(dt), 1e-3)
    self._window_frames = max(2, int(round(LANE_RATE_DAMPING_RATE_WINDOW_S / self.dt)) + 1)
    self.reset()

  def reset(self) -> None:
    self._lane_centers: collections.deque[float] = collections.deque(maxlen=self._window_frames)
    self._smoothed_lane_center_rate = 0.0

  def update(self, inputs: LateralDemandPipelineInputs, model_path_result: ModelPathProcessorResult,
             demand_source: str) -> LaneRateDampingResult:
    mode = sanitize_lane_rate_damping_mode(inputs.lane_rate_damping_mode)
    cap_lat_accel = LANE_RATE_DAMPING_CAP_LAT_ACCEL
    if mode == "off":
      self.reset()
      return LaneRateDampingResult(mode, False, False, "disabled", 0.0, 0.0, 0.0, 0.0, cap_lat_accel)

    lane_center, lane_center_valid = _lane_center_y0(inputs.left_lane_y0, inputs.right_lane_y0)
    raw_desired_curvature = _finite_float(inputs.desired_curvature)
    model_desired_curvature = _finite_float(model_path_result.desired_curvature)
    if not lane_center_valid or _finite_float(inputs.v_ego) is None or raw_desired_curvature is None or model_desired_curvature is None:
      self.reset()
      return LaneRateDampingResult(mode, False, False, "invalid", 0.0, 0.0, 0.0, 0.0, cap_lat_accel)

    if not inputs.lat_active:
      self.reset()
      return LaneRateDampingResult(mode, False, False, "inactive", 0.0, 0.0, 0.0, 0.0, cap_lat_accel)
    if demand_source != DEMAND_SOURCE_MODEL_PATH:
      self.reset()
      return LaneRateDampingResult(mode, False, False, str(demand_source), 0.0, 0.0, 0.0, 0.0, cap_lat_accel)
    if not inputs.lane_change_state_valid:
      self.reset()
      return LaneRateDampingResult(mode, False, False, "lane_change_unknown", 0.0, 0.0, 0.0, 0.0, cap_lat_accel)
    if inputs.lane_change_state != LANE_CHANGE_STATE_OFF:
      self.reset()
      return LaneRateDampingResult(mode, False, False, "lane_change", 0.0, 0.0, 0.0, 0.0, cap_lat_accel)
    if inputs.left_blinker or inputs.right_blinker:
      self.reset()
      return LaneRateDampingResult(mode, False, False, "blinker", 0.0, 0.0, 0.0, 0.0, cap_lat_accel)
    if inputs.steering_pressed is not False:
      self.reset()
      return LaneRateDampingResult(mode, False, False, "driver_override", 0.0, 0.0, 0.0, 0.0, cap_lat_accel)
    if inputs.curvature_limited:
      self.reset()
      return LaneRateDampingResult(mode, False, False, "curvature_limited", 0.0, 0.0, 0.0, 0.0, cap_lat_accel)
    if inputs.steer_limited:
      self.reset()
      return LaneRateDampingResult(mode, False, False, "steer_limited", 0.0, 0.0, 0.0, 0.0, cap_lat_accel)
    if model_path_result.reason != "ok":
      self.reset()
      return LaneRateDampingResult(mode, False, False, str(model_path_result.reason), 0.0, 0.0, 0.0, 0.0, cap_lat_accel)

    path_quality = _finite_float(model_path_result.quality)
    if path_quality is None:
      self.reset()
      return LaneRateDampingResult(mode, False, False, "invalid", 0.0, 0.0, 0.0, 0.0, cap_lat_accel)
    if path_quality < LANE_RATE_DAMPING_MIN_PATH_QUALITY:
      self.reset()
      return LaneRateDampingResult(mode, False, False, "low_quality", 0.0, 0.0, 0.0, 0.0, cap_lat_accel)

    v_ego = float(inputs.v_ego)
    if not math.isfinite(v_ego):
      self.reset()
      return LaneRateDampingResult(mode, False, False, "invalid", 0.0, 0.0, 0.0, 0.0, cap_lat_accel)
    if v_ego < LANE_RATE_DAMPING_MIN_SPEED:
      self.reset()
      return LaneRateDampingResult(mode, False, False, "low_speed", 0.0, 0.0, 0.0, 0.0, cap_lat_accel)

    requested_lat_accel = max(abs(raw_desired_curvature), abs(model_desired_curvature)) * v_ego * v_ego
    if not math.isfinite(requested_lat_accel):
      self.reset()
      return LaneRateDampingResult(mode, False, False, "invalid", 0.0, 0.0, 0.0, 0.0, cap_lat_accel)
    if requested_lat_accel > LANE_RATE_DAMPING_MAX_MODEL_LAT_ACCEL:
      self.reset()
      return LaneRateDampingResult(mode, False, False, "high_demand", 0.0, 0.0, 0.0, 0.0, cap_lat_accel)

    self._lane_centers.append(lane_center)
    if len(self._lane_centers) < self._window_frames:
      return LaneRateDampingResult(mode, False, False, "warming_up", lane_center, 0.0, 0.0, 0.0, cap_lat_accel)

    lane_center_rate_raw = (self._lane_centers[-1] - self._lane_centers[0]) / ((len(self._lane_centers) - 1) * self.dt)
    if not math.isfinite(lane_center_rate_raw):
      self.reset()
      return LaneRateDampingResult(mode, False, False, "invalid", 0.0, 0.0, 0.0, 0.0, cap_lat_accel)

    alpha = self.dt / (LANE_RATE_DAMPING_SMOOTH_TAU_S + self.dt)
    self._smoothed_lane_center_rate += alpha * (lane_center_rate_raw - self._smoothed_lane_center_rate)
    lane_center_rate = self._smoothed_lane_center_rate
    if not math.isfinite(lane_center_rate):
      self.reset()
      return LaneRateDampingResult(mode, False, False, "invalid", 0.0, 0.0, 0.0, 0.0, cap_lat_accel)

    # y+right: positive lane-center rate means the car is drifting left, so positive accel damps it.
    lat_accel = LANE_RATE_DAMPING_GAIN * _deadband(lane_center_rate, LANE_RATE_DAMPING_DEADBAND_MPS)
    if not math.isfinite(lat_accel):
      self.reset()
      return LaneRateDampingResult(mode, False, False, "invalid", 0.0, 0.0, 0.0, 0.0, cap_lat_accel)
    lat_accel = max(-cap_lat_accel, min(cap_lat_accel, lat_accel))
    curvature = lat_accel / max(v_ego * v_ego, LANE_RATE_DAMPING_V2_EPS)
    if not math.isfinite(curvature):
      self.reset()
      return LaneRateDampingResult(mode, False, False, "invalid", 0.0, 0.0, 0.0, 0.0, cap_lat_accel)

    active = True
    applied = mode == "apply" and lat_accel != 0.0
    reason = "ok" if lat_accel != 0.0 else "deadband"
    return LaneRateDampingResult(mode, active, applied, reason, lane_center, lane_center_rate, lat_accel, curvature, cap_lat_accel)


class _LaneFitSourceTracker:
  def __init__(self, dt: float) -> None:
    self.dt = max(float(dt), 1e-3)
    self._min_frames = max(1, int(round(LANE_FIT_SOURCE_MIN_PERSIST_S / self.dt)))
    self.reset()

  def reset(self) -> None:
    self._eligible_frames = 0
    self._last_output_lat_accel: float | None = None

  def _slew_output(self, baseline_lat_accel: float, target_lat_accel: float) -> tuple[float, bool]:
    prev = self._last_output_lat_accel
    if prev is None or not math.isfinite(prev):
      prev = baseline_lat_accel
    max_step = LANE_FIT_SOURCE_SLEW_LAT_JERK * self.dt
    if prev is not None and target_lat_accel == baseline_lat_accel:
      # ponytail: release a touch slower than entry so an immediate apply->block still backs off.
      max_step *= LANE_FIT_SOURCE_RELEASE_SLEW_SCALE
    output_lat_accel = prev + float(np.clip(target_lat_accel - prev, -max_step, max_step))
    slew_limited = abs(output_lat_accel - target_lat_accel) > 1e-9
    if slew_limited or target_lat_accel != baseline_lat_accel:
      self._last_output_lat_accel = output_lat_accel
    else:
      self._last_output_lat_accel = None
    return output_lat_accel, slew_limited

  def update(self, inputs: LateralDemandPipelineInputs, model_path_result: ModelPathProcessorResult,
             demand_source: str, baseline_curvature: float) -> LaneFitSourceResult:
    mode = sanitize_lane_fit_source_mode(inputs.lane_fit_source_mode)
    baseline_curvature_f = _finite_float(baseline_curvature)
    v_ego = _finite_float(inputs.v_ego)
    if baseline_curvature_f is None or v_ego is None or v_ego < 1.0:
      self.reset()
      reason = "low_speed" if v_ego is not None and v_ego < 1.0 else "invalid"
      return LaneFitSourceResult(mode, False, False, reason, 0.0, baseline_curvature_f or 0.0, 0.0, 0.0, False)

    speed_sq = max(v_ego * v_ego, 1e-6)
    baseline_lat_accel = baseline_curvature_f * speed_sq
    candidate_curvature = 0.0
    confidence = 0.0
    active = False
    applied = False
    reason = "disabled"

    if mode == "off":
      self.reset()
      return LaneFitSourceResult(mode, False, False, reason, 0.0, baseline_curvature_f, 0.0, 0.0, False)

    shadow_mode = mode == "shadow"
    if shadow_mode:
      self.reset()

    if not inputs.lat_active:
      reason = "inactive"
    elif demand_source != DEMAND_SOURCE_MODEL_PATH:
      reason = str(demand_source)
    elif not inputs.lane_change_state_valid:
      reason = "lane_change_unknown"
    elif inputs.lane_change_state != LANE_CHANGE_STATE_OFF:
      reason = "lane_change"
    elif inputs.left_blinker or inputs.right_blinker:
      reason = "blinker"
    elif inputs.steering_pressed is not False:
      reason = "driver_override"
    elif inputs.curvature_limited:
      reason = "curvature_limited"
    elif inputs.steer_limited:
      reason = "steer_limited"
    elif model_path_result.reason != "ok":
      reason = str(model_path_result.reason)
    else:
      path_quality = _finite_float(model_path_result.quality)
      if path_quality is None or path_quality < LANE_FIT_SOURCE_MIN_PATH_QUALITY:
        reason = "low_quality"
      elif v_ego < LANE_FIT_SOURCE_MIN_SPEED:
        reason = "low_speed"
      else:
        geometry = evaluate_lane_geometry(
          lane_lines=inputs.lane_lines,
          lane_line_probs=inputs.lane_line_probs,
          lane_line_stds=inputs.lane_line_stds,
          position_x=inputs.position_x,
          position_y=inputs.position_y,
          near_x=float(LANE_GEOMETRY_SAMPLE_XS[0]),
          preview_x=float(LANE_GEOMETRY_SAMPLE_XS[-1]),
        )
        if not geometry.valid:
          reason = geometry.reason
        elif geometry.confidence < LANE_FIT_SOURCE_MIN_GEOMETRY_CONFIDENCE:
          reason = "geometry_low_confidence"
        else:
          candidate_curvature_value = _lane_fit_candidate_curvature(inputs.lane_lines)
          if candidate_curvature_value is None:
            reason = "invalid"
          else:
            candidate_curvature = candidate_curvature_value
            candidate_lat_accel = candidate_curvature * speed_sq
            lat_accel_delta = candidate_lat_accel - baseline_lat_accel
            if (not math.isfinite(candidate_lat_accel)
                or abs(baseline_lat_accel) > LANE_FIT_SOURCE_MAX_LAT_ACCEL
                or abs(candidate_lat_accel) > LANE_FIT_SOURCE_MAX_LAT_ACCEL):
              reason = "lat_accel_limit"
            else:
              if abs(lat_accel_delta) > LANE_FIT_SOURCE_MAX_LAT_ACCEL_DELTA:
                reason = "delta_lat_accel"
              elif (candidate_lat_accel * baseline_lat_accel < 0.0 and
                    min(abs(candidate_lat_accel), abs(baseline_lat_accel)) >= LANE_FIT_SOURCE_SIGN_CONFLICT_LAT_ACCEL):
                reason = "sign_conflict"
              else:
                active = True
                confidence = float(geometry.confidence)
                if shadow_mode:
                  self._eligible_frames = 0
                  return LaneFitSourceResult(
                    mode=mode,
                    active=active,
                    applied=False,
                    reason="ok",
                    candidate_curvature=candidate_curvature,
                    applied_curvature=baseline_curvature_f,
                    lat_accel_delta=lat_accel_delta,
                    confidence=confidence,
                    slew_limited=False,
                  )
                target_curvature = baseline_curvature_f
                if mode == "apply":
                  self._eligible_frames += 1
                  if self._eligible_frames >= self._min_frames:
                    applied = True
                    target_curvature = candidate_curvature
                    reason = "ok"
                  else:
                    reason = "warming_up"
                else:
                  self._eligible_frames = 0
                  reason = "ok"
                applied_curvature_lat_accel, slew_limited = self._slew_output(baseline_lat_accel, target_curvature * speed_sq)
                return LaneFitSourceResult(
                  mode=mode,
                  active=active,
                  applied=applied,
                  reason=reason,
                  candidate_curvature=candidate_curvature,
                  applied_curvature=applied_curvature_lat_accel / speed_sq,
                  lat_accel_delta=lat_accel_delta,
                  confidence=confidence,
                  slew_limited=slew_limited,
                )

    self._eligible_frames = 0
    if shadow_mode:
      return LaneFitSourceResult(
        mode=mode,
        active=active,
        applied=False,
        reason=reason,
        candidate_curvature=0.0,
        applied_curvature=baseline_curvature_f,
        lat_accel_delta=0.0,
        confidence=confidence,
        slew_limited=False,
      )
    applied_curvature, slew_limited = self._slew_output(baseline_lat_accel, baseline_lat_accel)
    releasing = abs(applied_curvature - baseline_lat_accel) > 1e-9
    return LaneFitSourceResult(
      mode=mode,
      active=active,
      applied=releasing,
      reason=f"releasing_{reason}" if releasing else reason,
      candidate_curvature=0.0,
      applied_curvature=applied_curvature / speed_sq,
      lat_accel_delta=0.0,
      confidence=confidence,
      slew_limited=slew_limited,
    )


class LateralDemandPipeline:
  def __init__(self, dt: float = DT_CTRL) -> None:
    self.dt = float(dt)
    self._model_path_processor = ModelPathProcessor()
    self._curve_memory = CurveMemory()
    self._lane_change_path_shaper = LaneChangePathShaper(dt)
    self._lane_centering_assist = LaneCenteringAssistTracker()
    self._lane_rate_damping = _LaneRateDampingTracker(dt)
    self._lane_rate_damping_slew = _AyReleaseSlew(dt, LANE_RATE_DAMPING_RELEASE_LAT_JERK)
    self._lane_fit_source = _LaneFitSourceTracker(dt)
    self._preview_assist = PreviewAssistTracker(dt)
    self._model_step_blender = _ModelStepBlender()
    self._previous_desired_curvature = 0.0
    self._last_extreme_processed_curvature = False

  @property
  def previous_desired_curvature(self) -> float:
    return self._previous_desired_curvature

  def reset(self) -> None:
    self._model_path_processor.reset()
    self._curve_memory.reset()
    self._lane_change_path_shaper.reset()
    self._lane_centering_assist.reset()
    self._lane_rate_damping.reset()
    self._lane_rate_damping_slew.reset()
    self._lane_fit_source.reset()
    self._preview_assist.reset()
    self._model_step_blender.reset()
    self._previous_desired_curvature = 0.0
    self._last_extreme_processed_curvature = False

  def update(self, inputs: LateralDemandPipelineInputs) -> LateralDemandPipelineResult:
    raw_curvature_finite = _finite_float(inputs.desired_curvature)
    if raw_curvature_finite is None:
      fallback_curvature = _finite_float(inputs.measured_curvature)
      if fallback_curvature is None:
        fallback_curvature = _finite_float(self._previous_desired_curvature) or 0.0
      cloudlog.warning(f"lateral_demand nonfinite raw curvature: {inputs.desired_curvature}; falling back to measured curvature {fallback_curvature}")
      # A non-finite request cannot safely retain actuation authority. Treat this frame as
      # inactive and hold the finite measured demand until the upstream source recovers.
      inputs = replace(
        inputs,
        desired_curvature=fallback_curvature,
        measured_curvature=fallback_curvature,
        lat_active=False,
      )
      raw_curvature = fallback_curvature
    else:
      raw_curvature = raw_curvature_finite
    demand_source = DEMAND_SOURCE_MODEL_PATH
    lane_change_shaping_active = False
    lane_change_blend = 0.0
    lane_centering_result = inactive_lane_centering_assist_result("disabled")
    lane_rate_damping_result = LaneRateDampingResult("off", False, False, "disabled", 0.0, 0.0, 0.0, 0.0, LANE_RATE_DAMPING_CAP_LAT_ACCEL)
    lane_fit_source_result = LaneFitSourceResult("off", False, False, "disabled", 0.0, 0.0, 0.0, 0.0, False)
    preview_result = inactive_preview_assist_result()
    curve_memory_result = None

    if inputs.lateral_maneuver_curvature is not None:
      self._model_path_processor.reset()
      self._curve_memory.reset()
      self._lane_change_path_shaper.reset()
      self._lane_centering_assist.reset()
      self._lane_fit_source.reset()
      self._lane_rate_damping_slew.reset()
      self._model_step_blender.reset()
      new_desired_curvature = float(inputs.lateral_maneuver_curvature)
      model_path_result = ModelPathProcessorResult(new_desired_curvature, 0.0, True, "lateral_maneuver")
      demand_source = DEMAND_SOURCE_LATERAL_MANEUVER
      lane_rate_damping_result = self._lane_rate_damping.update(inputs, model_path_result, demand_source)
      lane_fit_source_result = LaneFitSourceResult(
        mode=sanitize_lane_fit_source_mode(inputs.lane_fit_source_mode),
        active=False,
        applied=False,
        reason="maneuver_override",
        candidate_curvature=0.0,
        applied_curvature=new_desired_curvature,
        lat_accel_delta=0.0,
        confidence=0.0,
        slew_limited=False,
      )
      preview_result = self._preview_assist.update(inputs, model_path_result, demand_source, new_desired_curvature)
    else:
      turn_curvature_sign = 0
      if inputs.lane_change_state == LANE_CHANGE_STATE_OFF and inputs.model_data_v2_sp_valid:
        if inputs.turn_direction == TURN_DIRECTION_RIGHT:
          turn_curvature_sign = 1
        elif inputs.turn_direction == TURN_DIRECTION_LEFT:
          turn_curvature_sign = -1

      # Cross-fade the 20 Hz model steps at 100 Hz before any conditioning sees them.
      blended_desired_curvature = self._model_step_blender.update(
        inputs.desired_curvature, inputs.model_frame_id, inputs.v_ego, inputs.lat_active)
      model_path_result = self._model_path_processor.update(ModelPathProcessorInputs(
        lat_active=inputs.lat_active,
        v_ego=inputs.v_ego,
        desired_curvature=blended_desired_curvature,
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
        model_frame_id=inputs.model_frame_id,
        left_blinker=inputs.left_blinker,
        right_blinker=inputs.right_blinker,
        steering_pressed=inputs.steering_pressed,
        steer_limited=inputs.steer_limited,
        straight_path_stabilization_mode=inputs.straight_path_stabilization_mode,
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
      lane_rate_damping_result = self._lane_rate_damping.update(inputs, model_path_result, demand_source)
      if inputs.lat_active:
        hard_blocked = (
          lane_rate_damping_result.mode != "apply"
          or lane_rate_damping_result.reason in LANE_RATE_DAMPING_HARD_BLOCK_REASONS
          or model_path_result.gated
        )
        if hard_blocked:
          self._lane_rate_damping_slew.reset()
          lrd_ay = 0.0
        else:
          # Transient limiter/quality edges release smoothly. Report the slewed value that
          # actually reaches the demand so telemetry never hides residual authority.
          lrd_ay = self._lane_rate_damping_slew.update(
            lane_rate_damping_result.lat_accel if lane_rate_damping_result.applied else 0.0)
          if lrd_ay != 0.0:
            releasing = not lane_rate_damping_result.applied
            lane_rate_damping_result = replace(
              lane_rate_damping_result,
              applied=True,
              reason=(f"releasing_{lane_rate_damping_result.reason}" if releasing else lane_rate_damping_result.reason),
              lat_accel=lrd_ay,
              curvature=lrd_ay / max(inputs.v_ego * inputs.v_ego, 1.0),
            )
        if lrd_ay != 0.0:
          new_desired_curvature += lrd_ay / max(inputs.v_ego * inputs.v_ego, 1.0)
      else:
        self._lane_rate_damping_slew.reset()

      lane_fit_source_result = self._lane_fit_source.update(inputs, model_path_result, demand_source, new_desired_curvature)
      new_desired_curvature = lane_fit_source_result.applied_curvature

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
          lane_lines=tuple(inputs.lane_lines),
          lane_line_stds=tuple(inputs.lane_line_stds),
          demand_source=demand_source,
          one_line_mode=inputs.lane_centering_one_line_mode,
        ), self.dt)
        new_desired_curvature += lane_centering_result.curvature_nudge
      elif not inputs.lane_centering_assist_enabled:
        self._lane_centering_assist.reset()

      if lane_fit_source_result.applied:
        demand_source = DEMAND_SOURCE_LANE_FIT

      preview_result = self._preview_assist.update(inputs, model_path_result, demand_source, new_desired_curvature)
      if preview_result.applied:
        new_desired_curvature += preview_result.curvature_nudge

    sensor_confidence = evaluate_sensor_confidence(SensorConfidenceInputs(
      lat_active=inputs.lat_active,
      v_ego=inputs.v_ego,
      model_curvature=raw_curvature,
      measured_curvature=inputs.measured_curvature,
      model_path_gated=bool(model_path_result.gated),
      model_path_reason=str(model_path_result.reason),
      model_age_s=inputs.model_age_s,
      steering_pressed=inputs.steering_pressed,
      steering_rate_deg=inputs.steering_rate_deg,
      yaw_rate=inputs.yaw_rate,
      steer_limited=bool(inputs.steer_limited),
      lane_change_active=inputs.lane_change_state != LANE_CHANGE_STATE_OFF,
      lane_change_state_valid=bool(inputs.lane_change_state_valid),
      left_blinker=inputs.left_blinker,
      right_blinker=inputs.right_blinker,
    ))

    processed_curvature = float(new_desired_curvature)
    # Stress guardrail: anomalous / non-finite curvature is contained before it reaches
    # the controller. Non-finite values fall back to straight-ahead (0.0) and reset the
    # previous-curvature memory so the next valid frame starts from a clean state.
    # Values beyond 0.05 1/m are logged but passed through unchanged.
    if not math.isfinite(processed_curvature):
      cloudlog.warning(f"lateral_demand nonfinite processed curvature: {processed_curvature}")
      self._last_extreme_processed_curvature = False
      processed_curvature = 0.0
    elif abs(processed_curvature) > 0.05:
      if not self._last_extreme_processed_curvature:
        cloudlog.warning(f"lateral_demand extreme processed curvature: {processed_curvature:.4f}")
      self._last_extreme_processed_curvature = True
    else:
      self._last_extreme_processed_curvature = False
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
      lane_centering_relax_active=lane_centering_result.relax_active,
      lane_centering_relax_reason_bits=lane_centering_result.relax_reason_bits,
      lane_centering_relax_envelope=lane_centering_result.relax_envelope,
      lane_centering_relax_lateral_error=lane_centering_result.relaxed_lateral_error,
      lane_centering_relax_predicted_error=lane_centering_result.relaxed_predicted_error,
      lane_centering_relax_age=lane_centering_result.relax_age,
      lane_centering_relax_nudge_flip_score=lane_centering_result.relax_nudge_flip_score,
      lane_centering_relax_error_cross_score=lane_centering_result.relax_error_cross_score,
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
        **sensor_confidence.debug_dict(),
        "demand_jerk_smoothing_active": bool(model_path_result.demand_jerk_smoothing_active),
        "demand_jerk_smoothing_step": float(model_path_result.demand_jerk_smoothing_step),
        "demand_jerk_smoothing_lag": float(model_path_result.demand_jerk_smoothing_lag),
        "straight_path_stabilization_mode": str(model_path_result.straight_path_stabilization_mode),
        "straight_path_stabilization_active": bool(model_path_result.straight_path_stabilization_active),
        "straight_path_stabilization_applied": bool(model_path_result.straight_path_stabilization_applied),
        "straight_path_stabilization_candidate_curvature": float(model_path_result.straight_path_stabilization_candidate_curvature),
        "straight_path_stabilization_anchor_lat_accel": float(model_path_result.straight_path_stabilization_anchor_lat_accel),
        "straight_path_stabilization_reason": str(model_path_result.straight_path_stabilization_reason),
        "lane_rate_damping_mode": str(lane_rate_damping_result.mode),
        "lane_rate_damping_active": bool(lane_rate_damping_result.active),
        "lane_rate_damping_applied": bool(lane_rate_damping_result.applied),
        "lane_rate_damping_reason": str(lane_rate_damping_result.reason),
        "lane_rate_damping_lane_center": float(lane_rate_damping_result.lane_center),
        "lane_rate_damping_lane_center_rate": float(lane_rate_damping_result.lane_center_rate),
        "lane_rate_damping_lat_accel": float(lane_rate_damping_result.lat_accel),
        "lane_rate_damping_curvature": float(lane_rate_damping_result.curvature),
        "lane_rate_damping_cap_lat_accel": float(lane_rate_damping_result.cap_lat_accel),
        "lane_fit_source_mode": str(lane_fit_source_result.mode),
        "lane_fit_source_active": bool(lane_fit_source_result.active),
        "lane_fit_source_applied": bool(lane_fit_source_result.applied),
        "lane_fit_source_reason": str(lane_fit_source_result.reason),
        "lane_fit_source_candidate_curvature": float(lane_fit_source_result.candidate_curvature),
        "lane_fit_source_applied_curvature": float(lane_fit_source_result.applied_curvature),
        "lane_fit_source_lat_accel_delta": float(lane_fit_source_result.lat_accel_delta),
        "lane_fit_source_confidence": float(lane_fit_source_result.confidence),
        "lane_fit_source_slew_limited": bool(lane_fit_source_result.slew_limited),
        "lane_change_blend": lane_change_blend,
        "lane_change_shaping_active": lane_change_shaping_active,
        "lane_centering_active": bool(lane_centering_result.active),
        "lane_centering_nudge": float(lane_centering_result.curvature_nudge),
        **lane_centering_result.debug,
        "curve_memory_active": bool(curve_memory_result.active) if curve_memory_result is not None else False,
        "curve_memory_remembered": float(curve_memory_result.remembered) if curve_memory_result is not None else float("nan"),
        "curve_memory_source": curve_memory_result.source if curve_memory_result is not None else "disabled",
        "curve_memory_samples": int(curve_memory_result.samples) if curve_memory_result is not None else 0,
        **preview_result.debug_dict(),
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
