import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Sequence

import numpy as np

from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.drive_helpers import MAX_LATERAL_ACCEL_NO_ROLL


MIN_ACTIVE_SPEED = 3.0
PATH_VALID_MIN_LEN = 17
PATH_VALID_MIN_X_SPAN = 1.0
MAX_PATH_Y_STD = 0.8
LOW_CONFIDENCE_PATH_Y_STD = 1.0
LOW_LANE_LINE_PROB = 0.45
ARM_SECONDS = 0.4
INVALID_GRACE_SECONDS = 0.3
FALLBACK_BLEND_SECONDS = 0.7
COOLDOWN_SECONDS = 0.6
PATH_CURVATURE_BLEND = 0.25
MAX_CANDIDATE_LAT_ACCEL_DELTA = 1.2
LANE_CENTER_MAX_OFFSET = 0.35
LANE_CENTER_MAX_LAT_ACCEL_BIAS = 0.25
ROAD_EDGE_CLEARANCE = 0.8
ROAD_EDGE_MAX_OFFSET = 0.45
ROAD_EDGE_MAX_LAT_ACCEL_BIAS = 0.35
MIN_BIAS_LOOKAHEAD = 12.0
MAX_BIAS_LOOKAHEAD = 28.0


class ExperimentalLateralPathPlannerState(IntEnum):
  baseline = 0
  arming = 1
  active = 2
  degraded_hold = 3
  blending_to_baseline = 4
  cooldown = 5


@dataclass
class ExperimentalLateralPathPlannerInputs:
  enabled: bool
  lat_active: bool
  v_ego: float
  baseline_curvature: float
  measured_curvature: float
  previous_desired_curvature: float
  position_x: Sequence[float]
  position_y: Sequence[float]
  position_y_std: Sequence[float]
  lane_line_probs: Sequence[float]
  left_lane_y0: float | None
  right_lane_y0: float | None
  left_road_edge_y0: float | None
  right_road_edge_y0: float | None
  lane_change_active: bool = False


@dataclass
class ExperimentalLateralPathPlannerResult:
  desired_curvature: float
  candidate_curvature: float
  baseline_curvature: float
  confidence: float
  active: bool
  state: ExperimentalLateralPathPlannerState
  reason: str


class ExperimentalLateralPathPlanner:
  def __init__(self, dt: float = DT_CTRL):
    self.dt = dt
    self.reset()

  def reset(self) -> None:
    self.state = ExperimentalLateralPathPlannerState.baseline
    self.valid_timer = 0.0
    self.invalid_timer = 0.0
    self.blend_timer = 0.0
    self.cooldown_timer = 0.0
    self.last_experimental_curvature = 0.0
    self.blend_start_curvature = 0.0
    self.last_invalid_reason = "ok"

  def update(self, inputs: ExperimentalLateralPathPlannerInputs) -> ExperimentalLateralPathPlannerResult:
    baseline_curvature = self._finite_or_zero(inputs.baseline_curvature)

    reset_reason = self._reset_reason(inputs)
    if reset_reason is not None:
      self.reset()
      return self._result(baseline_curvature, baseline_curvature, baseline_curvature, 0.0, False, self.state, reset_reason)

    candidate_curvature, confidence, candidate_reason = self._candidate(inputs, baseline_curvature)
    candidate_valid = candidate_reason == "ok"

    if self.state == ExperimentalLateralPathPlannerState.cooldown:
      self.cooldown_timer = max(self.cooldown_timer - self.dt, 0.0)
      if self.cooldown_timer <= 0.0:
        self.state = ExperimentalLateralPathPlannerState.baseline
        self.valid_timer = 0.0
      else:
        return self._result(baseline_curvature, candidate_curvature, baseline_curvature, confidence, False, self.state, "cooldown")

    if self.state in (ExperimentalLateralPathPlannerState.blending_to_baseline,):
      return self._continue_blend_to_baseline(baseline_curvature, candidate_curvature, confidence, candidate_reason)

    if not candidate_valid:
      return self._handle_invalid_candidate(baseline_curvature, candidate_curvature, confidence, candidate_reason)

    self.invalid_timer = 0.0
    if self.state in (ExperimentalLateralPathPlannerState.baseline, ExperimentalLateralPathPlannerState.arming):
      self.state = ExperimentalLateralPathPlannerState.arming
      self.valid_timer += self.dt
      if self.valid_timer < ARM_SECONDS:
        return self._result(baseline_curvature, candidate_curvature, baseline_curvature, confidence, False, self.state, "arming")

    self.state = ExperimentalLateralPathPlannerState.active
    self.last_experimental_curvature = candidate_curvature
    return self._result(candidate_curvature, candidate_curvature, baseline_curvature, confidence, True, self.state, "ok")

  def _handle_invalid_candidate(self, baseline_curvature: float, candidate_curvature: float, confidence: float,
                                candidate_reason: str) -> ExperimentalLateralPathPlannerResult:
    if self.state in (ExperimentalLateralPathPlannerState.active, ExperimentalLateralPathPlannerState.degraded_hold):
      self.invalid_timer += self.dt
      self.last_invalid_reason = candidate_reason
      if self.invalid_timer <= INVALID_GRACE_SECONDS:
        self.state = ExperimentalLateralPathPlannerState.degraded_hold
        return self._result(self.last_experimental_curvature, candidate_curvature, baseline_curvature, confidence, True, self.state,
                            f"{candidate_reason}_hold")

      self.state = ExperimentalLateralPathPlannerState.blending_to_baseline
      self.blend_timer = 0.0
      self.blend_start_curvature = self.last_experimental_curvature
      return self._continue_blend_to_baseline(baseline_curvature, candidate_curvature, confidence, candidate_reason)

    self.valid_timer = 0.0
    return self._result(baseline_curvature, candidate_curvature, baseline_curvature, confidence, False, ExperimentalLateralPathPlannerState.baseline,
                        candidate_reason)

  def _continue_blend_to_baseline(self, baseline_curvature: float, candidate_curvature: float, confidence: float,
                                  candidate_reason: str) -> ExperimentalLateralPathPlannerResult:
    self.blend_timer = min(self.blend_timer + self.dt, FALLBACK_BLEND_SECONDS)
    alpha = self.blend_timer / FALLBACK_BLEND_SECONDS
    output_curvature = self._blend(self.blend_start_curvature, baseline_curvature, alpha)
    if alpha >= 1.0:
      self.state = ExperimentalLateralPathPlannerState.cooldown
      self.cooldown_timer = COOLDOWN_SECONDS
      self.valid_timer = 0.0
      return self._result(baseline_curvature, candidate_curvature, baseline_curvature, confidence, False, self.state, "cooldown")

    return self._result(output_curvature, candidate_curvature, baseline_curvature, confidence, True, self.state, f"{candidate_reason}_blending")

  def _candidate(self, inputs: ExperimentalLateralPathPlannerInputs, baseline_curvature: float) -> tuple[float, float, str]:
    x_vals = self._as_finite_array(inputs.position_x)
    y_vals = self._as_finite_array(inputs.position_y)
    if not self._valid_core_path(x_vals, y_vals):
      return baseline_curvature, 0.0, "invalid_path"

    confidence = self._path_confidence(inputs.position_y_std)
    if confidence <= 0.0:
      return baseline_curvature, 0.0, "low_confidence"

    path_curvature = self._path_curvature(x_vals, y_vals)
    if path_curvature is None:
      return baseline_curvature, 0.0, "invalid_path"

    lane_bias = self._lane_center_bias(inputs, x_vals, y_vals)
    road_edge_bias = self._road_edge_bias(inputs, x_vals, y_vals)
    candidate = baseline_curvature + PATH_CURVATURE_BLEND * (path_curvature - baseline_curvature) + lane_bias + road_edge_bias
    candidate = self._limit_candidate_delta(candidate, baseline_curvature, inputs.v_ego)
    if not math.isfinite(candidate):
      return baseline_curvature, 0.0, "nonfinite_candidate"
    return candidate, confidence, "ok"

  def _reset_reason(self, inputs: ExperimentalLateralPathPlannerInputs) -> str | None:
    if not inputs.enabled:
      return "disabled"
    if not inputs.lat_active:
      return "inactive"
    if inputs.lane_change_active:
      return "lane_change"
    if not math.isfinite(inputs.v_ego) or inputs.v_ego < MIN_ACTIVE_SPEED:
      return "low_speed"
    if not math.isfinite(inputs.baseline_curvature):
      return "invalid_baseline"
    return None

  @staticmethod
  def _result(desired_curvature: float, candidate_curvature: float, baseline_curvature: float, confidence: float,
              active: bool, state: ExperimentalLateralPathPlannerState, reason: str) -> ExperimentalLateralPathPlannerResult:
    return ExperimentalLateralPathPlannerResult(
      desired_curvature=float(desired_curvature),
      candidate_curvature=float(candidate_curvature),
      baseline_curvature=float(baseline_curvature),
      confidence=float(confidence),
      active=bool(active),
      state=state,
      reason=reason,
    )

  @staticmethod
  def _as_finite_array(values: Sequence[float]) -> np.ndarray | None:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or arr.size == 0 or not np.all(np.isfinite(arr)):
      return None
    return arr

  @staticmethod
  def _valid_core_path(x_vals: np.ndarray | None, y_vals: np.ndarray | None) -> bool:
    if x_vals is None or y_vals is None or x_vals.size != y_vals.size or x_vals.size < PATH_VALID_MIN_LEN:
      return False
    core_x_vals = x_vals[:PATH_VALID_MIN_LEN]
    return bool(np.all(np.diff(core_x_vals) >= 0.0) and core_x_vals[-1] - core_x_vals[0] >= PATH_VALID_MIN_X_SPAN)

  @staticmethod
  def _path_confidence(position_y_std: Sequence[float]) -> float:
    y_std = ExperimentalLateralPathPlanner._as_finite_array(position_y_std)
    if y_std is None or y_std.size < PATH_VALID_MIN_LEN:
      return 0.0
    max_y_std = float(np.max(y_std[:PATH_VALID_MIN_LEN]))
    if max_y_std <= MAX_PATH_Y_STD:
      return 1.0
    if max_y_std >= LOW_CONFIDENCE_PATH_Y_STD:
      return 0.0
    return float(np.interp(max_y_std, [MAX_PATH_Y_STD, LOW_CONFIDENCE_PATH_Y_STD], [1.0, 0.0]))

  @staticmethod
  def _path_curvature(x_vals: np.ndarray, y_vals: np.ndarray) -> float | None:
    core_x = x_vals[:PATH_VALID_MIN_LEN]
    core_y = y_vals[:PATH_VALID_MIN_LEN]
    if core_x[-1] - core_x[0] < PATH_VALID_MIN_X_SPAN:
      return None
    centered_x = core_x - core_x[0]
    try:
      a, b, _ = np.polyfit(centered_x, core_y, 2)
    except (ValueError, np.linalg.LinAlgError):
      return None
    curvature = 2.0 * a / ((1.0 + b * b) ** 1.5)
    return float(curvature) if math.isfinite(curvature) else None

  @classmethod
  def _lane_center_bias(cls, inputs: ExperimentalLateralPathPlannerInputs, x_vals: np.ndarray, y_vals: np.ndarray) -> float:
    if len(inputs.lane_line_probs) <= 2:
      return 0.0
    central_lane_probs = (float(inputs.lane_line_probs[1]), float(inputs.lane_line_probs[2]))
    if not all(math.isfinite(prob) for prob in central_lane_probs) or min(central_lane_probs) < LOW_LANE_LINE_PROB:
      return 0.0
    if inputs.left_lane_y0 is None or inputs.right_lane_y0 is None:
      return 0.0
    if not math.isfinite(inputs.left_lane_y0) or not math.isfinite(inputs.right_lane_y0):
      return 0.0
    lane_width = abs(inputs.right_lane_y0 - inputs.left_lane_y0)
    if lane_width < 2.5 or lane_width > 5.0:
      return 0.0

    lane_center_y0 = (inputs.left_lane_y0 + inputs.right_lane_y0) * 0.5
    lateral_offset = float(y_vals[0] - lane_center_y0)
    desired_shift = float(np.clip(-lateral_offset, -LANE_CENTER_MAX_OFFSET, LANE_CENTER_MAX_OFFSET))
    return cls._limit_lat_accel_bias(cls._shift_to_curvature(desired_shift, inputs.v_ego, x_vals),
                                     inputs.v_ego, LANE_CENTER_MAX_LAT_ACCEL_BIAS)

  @classmethod
  def _road_edge_bias(cls, inputs: ExperimentalLateralPathPlannerInputs, x_vals: np.ndarray, y_vals: np.ndarray) -> float:
    y0 = float(y_vals[0])
    desired_shift = 0.0
    if inputs.left_road_edge_y0 is not None and math.isfinite(inputs.left_road_edge_y0):
      left_clearance = y0 - inputs.left_road_edge_y0
      if 0.0 < left_clearance < ROAD_EDGE_CLEARANCE:
        desired_shift += ROAD_EDGE_CLEARANCE - left_clearance
    if inputs.right_road_edge_y0 is not None and math.isfinite(inputs.right_road_edge_y0):
      right_clearance = inputs.right_road_edge_y0 - y0
      if 0.0 < right_clearance < ROAD_EDGE_CLEARANCE:
        desired_shift -= ROAD_EDGE_CLEARANCE - right_clearance
    desired_shift = float(np.clip(desired_shift, -ROAD_EDGE_MAX_OFFSET, ROAD_EDGE_MAX_OFFSET))
    return cls._limit_lat_accel_bias(cls._shift_to_curvature(desired_shift, inputs.v_ego, x_vals),
                                     inputs.v_ego, ROAD_EDGE_MAX_LAT_ACCEL_BIAS)

  @staticmethod
  def _shift_to_curvature(lateral_shift: float, v_ego: float, x_vals: np.ndarray) -> float:
    if abs(lateral_shift) < 1e-6:
      return 0.0
    lookahead = float(np.clip(max(v_ego, MIN_ACTIVE_SPEED), MIN_BIAS_LOOKAHEAD, MAX_BIAS_LOOKAHEAD))
    path_span = float(x_vals[min(PATH_VALID_MIN_LEN - 1, x_vals.size - 1)] - x_vals[0])
    lookahead = max(min(lookahead, path_span), MIN_BIAS_LOOKAHEAD)
    return 2.0 * lateral_shift / (lookahead * lookahead)

  @staticmethod
  def _limit_candidate_delta(candidate: float, baseline_curvature: float, v_ego: float) -> float:
    speed_sq = max(v_ego, MIN_ACTIVE_SPEED) ** 2
    max_delta = min(MAX_CANDIDATE_LAT_ACCEL_DELTA, MAX_LATERAL_ACCEL_NO_ROLL) / speed_sq
    return float(np.clip(candidate, baseline_curvature - max_delta, baseline_curvature + max_delta))

  @staticmethod
  def _limit_lat_accel_bias(curvature_bias: float, v_ego: float, max_lat_accel_bias: float) -> float:
    speed_sq = max(v_ego, MIN_ACTIVE_SPEED) ** 2
    max_curvature_bias = max_lat_accel_bias / speed_sq
    return float(np.clip(curvature_bias, -max_curvature_bias, max_curvature_bias))

  @staticmethod
  def _blend(start: float, end: float, alpha: float) -> float:
    return float(start + alpha * (end - start))

  @staticmethod
  def _finite_or_zero(value: float) -> float:
    return float(value) if math.isfinite(value) else 0.0
