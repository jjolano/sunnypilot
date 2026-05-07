import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from openpilot.selfdrive.controls.lib.drive_helpers import get_curvature_from_plan
from openpilot.selfdrive.modeld.constants import ModelConstants


PATH_VALID_MIN_LEN = 17
PATH_VALID_MIN_X_SPAN = 1.0
MAX_CORE_PATH_Y_STEP = 1.0
PATH_CURVATURE_ACTION_T = 0.25
MIN_JUMP_CHECK_SPEED = 4.0
MAX_LAT_ACCEL_JUMP = 3.0
MAX_HARD_LAT_ACCEL_JUMP = 5.0
MAX_PATH_CURVATURE_DISAGREEMENT = 3.0
MAX_PATH_Y_STD = 0.8
TURN_INTENT_MAX_PATH_Y_STD = 1.8
TURN_INTENT_MIN_CURVATURE = 0.002
TURN_INTENT_MAX_PATH_CURVATURE_DISAGREEMENT = 0.75
LOW_LANE_LINE_PROB = 0.35
HIGH_FRAME_DROP_PERC = 20.0
LOW_QUALITY_BLEND_THRESHOLD = 0.75
LOW_QUALITY_BLEND_MIN_ALPHA = 0.4
HARD_INVALID_FALLBACK_MEASURED_ALPHA = 0.25
SOFT_GATE_HOLD_FRAMES = 2
SOFT_GATE_HOLD_QUALITY = 0.70
SOFT_GATE_REASONS = frozenset(("high_path_std", "frame_drop", "path_disagreement"))


@dataclass
class ModelPathProcessorInputs:
  lat_active: bool
  v_ego: float
  desired_curvature: float
  measured_curvature: float
  previous_desired_curvature: float
  position_x: Sequence[float]
  position_y: Sequence[float]
  position_y_std: Sequence[float]
  orientation_z: Sequence[float]
  orientation_rate_z: Sequence[float]
  lane_line_probs: Sequence[float]
  turn_curvature_sign: int = 0
  frame_drop_perc: float = 0.0


@dataclass
class ModelPathProcessorResult:
  desired_curvature: float
  quality: float
  gated: bool
  reason: str
  hold_frames_remaining: int = 0


class ModelPathProcessor:
  def __init__(self) -> None:
    self.reset()

  def reset(self) -> None:
    self._hold_frames_remaining = 0
    self._hold_reason = "ok"

  def update(self, inputs: ModelPathProcessorInputs) -> ModelPathProcessorResult:
    if not inputs.lat_active:
      self.reset()
      return ModelPathProcessorResult(float(inputs.measured_curvature), 0.0, True, "inactive")

    if not math.isfinite(inputs.desired_curvature):
      hard_invalid_fallback = self._hard_invalid_fallback_curvature(inputs.previous_desired_curvature, inputs.measured_curvature)
      return ModelPathProcessorResult(hard_invalid_fallback, 0.0, True, "nonfinite_curvature")

    if not self._valid_core_path(inputs.position_x, inputs.position_y):
      hard_invalid_fallback = self._hard_invalid_fallback_curvature(inputs.previous_desired_curvature, inputs.measured_curvature)
      return ModelPathProcessorResult(hard_invalid_fallback, 0.0, True, "invalid_path")

    desired_curvature = float(inputs.desired_curvature)
    quality = 1.0
    reason = "ok"

    if inputs.turn_curvature_sign != 0 and desired_curvature * inputs.turn_curvature_sign < 0.0:
      return ModelPathProcessorResult(0.0, 0.5, True, "turn_opposite_curvature")

    path_curvature = self._path_curvature(inputs.orientation_z, inputs.orientation_rate_z, inputs.v_ego)
    path_disagreement = None
    if path_curvature is not None:
      path_disagreement = abs(desired_curvature - path_curvature) * max(inputs.v_ego, 1.0) ** 2

    path_std_quality = self._path_std_quality(
      inputs.position_y_std,
      desired_curvature,
      path_disagreement,
      inputs.turn_curvature_sign,
    )
    if path_std_quality < quality:
      quality = path_std_quality
      reason = "high_path_std"

    lane_quality = self._lane_quality(inputs.lane_line_probs)
    if lane_quality < quality:
      quality = lane_quality
      reason = "low_lane_confidence"

    if math.isfinite(inputs.frame_drop_perc) and inputs.frame_drop_perc > HIGH_FRAME_DROP_PERC:
      quality = min(quality, SOFT_GATE_HOLD_QUALITY)
      reason = "frame_drop"

    if path_disagreement is not None:
      if path_disagreement > MAX_PATH_CURVATURE_DISAGREEMENT:
        quality = min(quality, 0.65)
        reason = "path_disagreement"

    fallback_curvature = self._fallback_curvature(inputs.previous_desired_curvature, inputs.measured_curvature)
    jump_result = self._limit_implausible_jump(inputs.v_ego, desired_curvature, fallback_curvature)
    if jump_result is not None:
      return jump_result

    quality, reason, hold_frames_remaining = self._apply_soft_gate_hold(quality, reason)

    if quality < LOW_QUALITY_BLEND_THRESHOLD:
      alpha = float(np.interp(quality, [0.0, LOW_QUALITY_BLEND_THRESHOLD], [LOW_QUALITY_BLEND_MIN_ALPHA, 1.0]))
      desired_curvature = self._blend(fallback_curvature, desired_curvature, alpha)
      return ModelPathProcessorResult(desired_curvature, quality, True, reason, hold_frames_remaining)

    return ModelPathProcessorResult(desired_curvature, quality, False, reason, hold_frames_remaining)

  def _apply_soft_gate_hold(self, quality: float, reason: str) -> tuple[float, str, int]:
    if reason in SOFT_GATE_REASONS and quality < LOW_QUALITY_BLEND_THRESHOLD:
      self._hold_frames_remaining = SOFT_GATE_HOLD_FRAMES
      self._hold_reason = reason
      return quality, reason, self._hold_frames_remaining

    if self._hold_frames_remaining > 0:
      self._hold_frames_remaining -= 1
      return min(quality, SOFT_GATE_HOLD_QUALITY), self._hold_reason, self._hold_frames_remaining

    self._hold_reason = "ok"
    return quality, reason, 0

  @staticmethod
  def _fallback_curvature(previous_desired_curvature: float, measured_curvature: float) -> float:
    if math.isfinite(previous_desired_curvature):
      return float(previous_desired_curvature)
    if math.isfinite(measured_curvature):
      return float(measured_curvature)
    return 0.0

  @classmethod
  def _hard_invalid_fallback_curvature(cls, previous_desired_curvature: float, measured_curvature: float) -> float:
    if math.isfinite(previous_desired_curvature) and math.isfinite(measured_curvature):
      return cls._blend(float(previous_desired_curvature), float(measured_curvature), HARD_INVALID_FALLBACK_MEASURED_ALPHA)
    if math.isfinite(measured_curvature):
      return float(measured_curvature)
    if math.isfinite(previous_desired_curvature):
      return float(previous_desired_curvature)
    return 0.0

  @staticmethod
  def _as_finite_array(values: Sequence[float]) -> np.ndarray | None:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or arr.size == 0 or not np.all(np.isfinite(arr)):
      return None
    return arr

  @classmethod
  def _valid_core_path(cls, position_x: Sequence[float], position_y: Sequence[float]) -> bool:
    x_vals = cls._as_finite_array(position_x)
    y_vals = cls._as_finite_array(position_y)
    if x_vals is None or y_vals is None or x_vals.size != y_vals.size or x_vals.size < PATH_VALID_MIN_LEN:
      return False
    core_x_vals = x_vals[:PATH_VALID_MIN_LEN]
    core_y_vals = y_vals[:PATH_VALID_MIN_LEN]
    return bool(
      np.all(np.diff(core_x_vals) >= 0.0) and
      core_x_vals[-1] - core_x_vals[0] >= PATH_VALID_MIN_X_SPAN and
      np.max(np.abs(np.diff(core_y_vals))) <= MAX_CORE_PATH_Y_STEP
    )

  @classmethod
  def _path_std_quality(
    cls,
    position_y_std: Sequence[float],
    desired_curvature: float,
    path_disagreement: float | None,
    turn_curvature_sign: int,
  ) -> float:
    y_std = cls._as_finite_array(position_y_std)
    if y_std is None:
      return 0.85
    max_y_std = float(np.max(y_std[:PATH_VALID_MIN_LEN]))
    if max_y_std <= MAX_PATH_Y_STD:
      return 1.0
    if cls._turn_intent_allows_path_std(max_y_std, desired_curvature, path_disagreement, turn_curvature_sign):
      return 1.0
    return float(np.interp(max_y_std, [MAX_PATH_Y_STD, MAX_PATH_Y_STD * 2.0], [0.7, 0.45]))

  @staticmethod
  def _turn_intent_allows_path_std(
    max_y_std: float,
    desired_curvature: float,
    path_disagreement: float | None,
    turn_curvature_sign: int,
  ) -> bool:
    if turn_curvature_sign == 0 or path_disagreement is None:
      return False
    if max_y_std > TURN_INTENT_MAX_PATH_Y_STD or abs(desired_curvature) < TURN_INTENT_MIN_CURVATURE:
      return False
    if desired_curvature * turn_curvature_sign <= 0.0:
      return False
    return path_disagreement <= TURN_INTENT_MAX_PATH_CURVATURE_DISAGREEMENT

  @staticmethod
  def _lane_quality(lane_line_probs: Sequence[float]) -> float:
    if len(lane_line_probs) <= 2:
      return 0.9
    central_prob = min(float(lane_line_probs[1]), float(lane_line_probs[2]))
    if not math.isfinite(central_prob):
      return 0.85
    if central_prob >= LOW_LANE_LINE_PROB:
      return 1.0
    return float(np.interp(central_prob, [0.0, LOW_LANE_LINE_PROB], [0.85, 1.0]))

  @classmethod
  def _path_curvature(cls, orientation_z: Sequence[float], orientation_rate_z: Sequence[float], v_ego: float) -> float | None:
    yaws = cls._as_finite_array(orientation_z)
    yaw_rates = cls._as_finite_array(orientation_rate_z)
    if yaws is None or yaw_rates is None or yaws.size != yaw_rates.size or yaws.size < ModelConstants.IDX_N:
      return None

    return float(get_curvature_from_plan(yaws, yaw_rates, ModelConstants.T_IDXS, v_ego, PATH_CURVATURE_ACTION_T))

  @classmethod
  def _limit_implausible_jump(cls, v_ego: float, desired_curvature: float, fallback_curvature: float) -> ModelPathProcessorResult | None:
    if v_ego < MIN_JUMP_CHECK_SPEED or not math.isfinite(fallback_curvature):
      return None

    lateral_accel_jump = abs(desired_curvature - fallback_curvature) * max(v_ego, 1.0) ** 2
    if lateral_accel_jump <= MAX_LAT_ACCEL_JUMP:
      return None
    if lateral_accel_jump > MAX_HARD_LAT_ACCEL_JUMP:
      return ModelPathProcessorResult(fallback_curvature, 0.2, True, "curvature_jump")

    return ModelPathProcessorResult(cls._blend(fallback_curvature, desired_curvature, 0.25), 0.35, True, "curvature_jump")

  @staticmethod
  def _blend(start: float, end: float, alpha: float) -> float:
    return float(start + alpha * (end - start))
