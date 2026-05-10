import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.drive_helpers import get_curvature_from_plan
from openpilot.selfdrive.modeld.constants import ModelConstants


PATH_VALID_MIN_LEN = 17
PATH_VALID_MIN_X_SPAN = 1.0
MIN_CORE_PATH_X_STEP = 1e-3
MAX_CORE_PATH_LATERAL_SLOPE = 1.0
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
HARD_INVALID_RECOVERY_LAT_JERK = 2.0
SMOOTHED_CURVATURE_MIN_SPEED = 5.0
SMOOTHED_CURVATURE_MIN_SAMPLES = 5
SMOOTHED_CURVATURE_SPEED_BP = [5.0, 15.0, 30.0]
SMOOTHED_CURVATURE_WINDOW_S = [0.28, 0.45, 0.65]
SMOOTHED_CURVATURE_BLEND_ALPHA = [0.20, 0.35, 0.50]
SMOOTHED_CURVATURE_MAX_LAT_ACCEL_DELTA = [0.20, 0.35, 0.50]
SMOOTHED_CURVATURE_MAX_RAW_LAT_ACCEL_DISAGREEMENT = 1.25


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
  smooth_model_path_curvature: bool = False
  lane_change_active: bool = False


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
    self._recovering_from_hard_invalid = False

  def update(self, inputs: ModelPathProcessorInputs) -> ModelPathProcessorResult:
    if not inputs.lat_active:
      self.reset()
      return ModelPathProcessorResult(float(inputs.measured_curvature), 0.0, True, "inactive")

    if not math.isfinite(inputs.desired_curvature):
      self._recovering_from_hard_invalid = False
      hard_invalid_fallback = self._hard_invalid_fallback_curvature(inputs.previous_desired_curvature, inputs.measured_curvature)
      return ModelPathProcessorResult(hard_invalid_fallback, 0.0, True, "nonfinite_curvature")

    if not self._valid_core_path(inputs.position_x, inputs.position_y):
      self._recovering_from_hard_invalid = True
      hard_invalid_fallback = self._hard_invalid_fallback_curvature(inputs.previous_desired_curvature, inputs.measured_curvature)
      return ModelPathProcessorResult(hard_invalid_fallback, 0.0, True, "invalid_path")

    desired_curvature = float(inputs.desired_curvature)
    fallback_curvature = self._fallback_curvature(inputs.previous_desired_curvature, inputs.measured_curvature)
    quality = 1.0
    reason = "ok"

    if inputs.turn_curvature_sign != 0 and desired_curvature * inputs.turn_curvature_sign < 0.0:
      self._recovering_from_hard_invalid = False
      turn_fallback_curvature = self._turn_compatible_fallback_curvature(
        inputs.previous_desired_curvature,
        inputs.measured_curvature,
        inputs.turn_curvature_sign,
      )
      return ModelPathProcessorResult(turn_fallback_curvature, 0.5, True, "turn_opposite_curvature")

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

    jump_result = self._limit_implausible_jump(inputs.v_ego, desired_curvature, fallback_curvature)
    if jump_result is not None:
      self._recovering_from_hard_invalid = False
      return jump_result

    quality, reason, hold_frames_remaining = self._apply_soft_gate_hold(quality, reason)

    if quality < LOW_QUALITY_BLEND_THRESHOLD:
      self._recovering_from_hard_invalid = False
      alpha = float(np.interp(quality, [0.0, LOW_QUALITY_BLEND_THRESHOLD], [LOW_QUALITY_BLEND_MIN_ALPHA, 1.0]))
      desired_curvature = self._blend(fallback_curvature, desired_curvature, alpha)
      return ModelPathProcessorResult(desired_curvature, quality, True, reason, hold_frames_remaining)

    smoothed_curvature = self._smoothed_path_curvature(inputs, desired_curvature, quality)
    if smoothed_curvature is not None:
      desired_curvature = smoothed_curvature
      jump_result = self._limit_implausible_jump(inputs.v_ego, desired_curvature, fallback_curvature)
      if jump_result is not None:
        self._recovering_from_hard_invalid = False
        return jump_result

    recovery_result = self._limit_hard_invalid_recovery(inputs, desired_curvature)
    if recovery_result is not None:
      return recovery_result

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

  def _limit_hard_invalid_recovery(self, inputs: ModelPathProcessorInputs, desired_curvature: float) -> ModelPathProcessorResult | None:
    if not self._recovering_from_hard_invalid:
      return None

    previous_desired_curvature = float(inputs.previous_desired_curvature)
    if not math.isfinite(previous_desired_curvature):
      self._recovering_from_hard_invalid = False
      return None

    if previous_desired_curvature * desired_curvature > 0.0 and abs(desired_curvature) >= abs(previous_desired_curvature):
      return None

    max_curvature_step = HARD_INVALID_RECOVERY_LAT_JERK * DT_CTRL / max(inputs.v_ego, 1.0) ** 2
    curvature_delta = desired_curvature - previous_desired_curvature
    if abs(curvature_delta) <= max_curvature_step:
      self._recovering_from_hard_invalid = False
      return None

    limited_curvature = previous_desired_curvature + math.copysign(max_curvature_step, curvature_delta)
    return ModelPathProcessorResult(float(limited_curvature), SOFT_GATE_HOLD_QUALITY, True, "invalid_path")

  @staticmethod
  def _fallback_curvature(previous_desired_curvature: float, measured_curvature: float) -> float:
    if math.isfinite(previous_desired_curvature):
      return float(previous_desired_curvature)
    if math.isfinite(measured_curvature):
      return float(measured_curvature)
    return 0.0

  @staticmethod
  def _turn_compatible_fallback_curvature(previous_desired_curvature: float, measured_curvature: float, turn_curvature_sign: int) -> float:
    if math.isfinite(previous_desired_curvature) and previous_desired_curvature * turn_curvature_sign >= 0.0:
      return float(previous_desired_curvature)
    if math.isfinite(measured_curvature) and measured_curvature * turn_curvature_sign >= 0.0:
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
    core_x_steps = np.diff(core_x_vals)
    core_y_steps = np.diff(core_y_vals)
    return bool(
      np.all(core_x_steps >= MIN_CORE_PATH_X_STEP) and
      core_x_vals[-1] - core_x_vals[0] >= PATH_VALID_MIN_X_SPAN and
      np.max(np.abs(core_y_steps) / core_x_steps) <= MAX_CORE_PATH_LATERAL_SLOPE
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
  def _smoothed_path_curvature(
    cls,
    inputs: ModelPathProcessorInputs,
    desired_curvature: float,
    quality: float,
  ) -> float | None:
    if not inputs.smooth_model_path_curvature or inputs.lane_change_active:
      return None

    v_ego = float(inputs.v_ego)
    if not math.isfinite(v_ego) or v_ego < SMOOTHED_CURVATURE_MIN_SPEED:
      return None

    yaws = cls._as_finite_array(inputs.orientation_z)
    if yaws is None or yaws.size < PATH_VALID_MIN_LEN:
      return None

    t_idxs = np.asarray(ModelConstants.T_IDXS[:PATH_VALID_MIN_LEN], dtype=float)
    yaws = yaws[:PATH_VALID_MIN_LEN]
    sample_count = min(t_idxs.size, yaws.size)
    if sample_count < SMOOTHED_CURVATURE_MIN_SAMPLES:
      return None

    t_idxs = t_idxs[:sample_count]
    yaws = yaws[:sample_count]
    window_s = float(np.interp(v_ego, SMOOTHED_CURVATURE_SPEED_BP, SMOOTHED_CURVATURE_WINDOW_S))
    distance_from_action_t = np.abs(t_idxs - PATH_CURVATURE_ACTION_T)
    sample_mask = distance_from_action_t <= window_s
    if int(np.count_nonzero(sample_mask)) < SMOOTHED_CURVATURE_MIN_SAMPLES:
      nearest_idxs = np.argsort(distance_from_action_t)[:SMOOTHED_CURVATURE_MIN_SAMPLES]
      sample_mask = np.zeros(sample_count, dtype=bool)
      sample_mask[nearest_idxs] = True

    fit_t = t_idxs[sample_mask]
    fit_yaws = yaws[sample_mask]
    if fit_t.size < SMOOTHED_CURVATURE_MIN_SAMPLES or np.unique(fit_t).size < 3:
      return None

    weight_width = max(window_s * 0.5, 1e-3)
    weights = np.exp(-0.5 * ((fit_t - PATH_CURVATURE_ACTION_T) / weight_width) ** 2)
    coefficients = np.polyfit(fit_t, fit_yaws, deg=2, w=weights)
    psi_target = float(np.polyval(coefficients, PATH_CURVATURE_ACTION_T))
    psi_rate = float(np.polyval(np.polyder(coefficients), 0.0))
    candidate_curvature = float(2.0 * (psi_target / (v_ego * PATH_CURVATURE_ACTION_T)) - psi_rate / v_ego)
    if not math.isfinite(candidate_curvature):
      return None

    if inputs.turn_curvature_sign != 0 and candidate_curvature * inputs.turn_curvature_sign < 0.0:
      return None

    speed_sq = max(v_ego, 1.0) ** 2
    candidate_delta_lat_accel = (candidate_curvature - desired_curvature) * speed_sq
    if abs(candidate_delta_lat_accel) > SMOOTHED_CURVATURE_MAX_RAW_LAT_ACCEL_DISAGREEMENT:
      return None

    quality_alpha = float(np.interp(quality, [LOW_QUALITY_BLEND_THRESHOLD, 1.0], [0.0, 1.0]))
    blend_alpha = float(np.interp(v_ego, SMOOTHED_CURVATURE_SPEED_BP, SMOOTHED_CURVATURE_BLEND_ALPHA)) * quality_alpha
    max_delta_lat_accel = float(np.interp(v_ego, SMOOTHED_CURVATURE_SPEED_BP, SMOOTHED_CURVATURE_MAX_LAT_ACCEL_DELTA)) * quality_alpha
    if blend_alpha <= 0.0 or max_delta_lat_accel <= 0.0:
      return None

    bounded_delta_lat_accel = float(np.clip(candidate_delta_lat_accel * blend_alpha, -max_delta_lat_accel, max_delta_lat_accel))
    return desired_curvature + bounded_delta_lat_accel / speed_sq

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
