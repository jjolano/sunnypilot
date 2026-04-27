import math
from dataclasses import dataclass
from typing import Sequence

from cereal import log
from openpilot.common.realtime import DT_CTRL


LaneChangeState = log.LaneChangeState
LaneChangeDirection = log.LaneChangeDirection

LANE_CHANGE_DURATION = 5.5
ENTRY_BLEND_DURATION = 1.0
EXIT_BLEND_DURATION = 0.5
MIN_LANE_LINE_PROB = 0.6
SOFT_FALLBACK_LANE_LINE_PROB = 0.45
MIN_LANE_WIDTH = 3.0
MAX_LANE_WIDTH = 4.0
LANE_WIDTH_TOLERANCE = 0.8
MAX_ENTRY_CURVATURE = 0.0015
MIN_VEGO = 1.0


@dataclass
class LaneChangeSCurveInputs:
  lat_active: bool
  v_ego: float
  left_blinker: bool
  right_blinker: bool
  steering_pressed: bool
  lane_change_state: int
  lane_change_direction: int
  model_curvature: float
  prev_desired_curvature: float
  lane_line_probs: Sequence[float]
  left_lane_y0: float | None
  right_lane_y0: float | None


@dataclass
class LaneChangeSCurveResult:
  desired_curvature: float
  blend: float
  active: bool
  soft_fallback: bool


class LaneChangeSCurveController:
  def __init__(self, dt: float = DT_CTRL):
    self.dt = dt
    self.reset()

  def reset(self) -> None:
    self.session_active = False
    self.planned = False
    self.soft_fallback = False
    self.direction = LaneChangeDirection.none
    self.elapsed = 0.0
    self.blend = 0.0
    self.baseline_curvature = 0.0
    self.lane_width = 0.0

  def update(self, inputs: LaneChangeSCurveInputs) -> LaneChangeSCurveResult:
    if not self._lane_change_active(inputs):
      self.reset()
      return LaneChangeSCurveResult(inputs.model_curvature, 0.0, False, False)

    if self._hard_abort(inputs):
      self.reset()
      return LaneChangeSCurveResult(inputs.model_curvature, 0.0, False, False)

    if not self.session_active or inputs.lane_change_direction != self.direction:
      self._start_session(inputs)

    if not self.planned:
      return LaneChangeSCurveResult(inputs.model_curvature, 0.0, False, False)

    if not self.soft_fallback and self._should_soft_fallback(inputs):
      self.soft_fallback = True

    finishing = inputs.lane_change_state == LaneChangeState.laneChangeFinishing
    target_blend = 1.0 if not finishing and not self.soft_fallback and self.elapsed < LANE_CHANGE_DURATION else 0.0
    self.blend = self._approach(self.blend, target_blend)

    scripted_curvature = self._scripted_curvature(inputs.v_ego)
    desired_curvature = inputs.model_curvature + self.blend * (scripted_curvature - inputs.model_curvature)

    if not finishing and not self.soft_fallback and self.elapsed < LANE_CHANGE_DURATION:
      self.elapsed = min(self.elapsed + self.dt, LANE_CHANGE_DURATION)

    return LaneChangeSCurveResult(desired_curvature, self.blend, self.blend > 1e-3, self.soft_fallback)

  def _start_session(self, inputs: LaneChangeSCurveInputs) -> None:
    self.session_active = True
    self.direction = inputs.lane_change_direction
    self.elapsed = 0.0
    self.blend = 0.0
    self.soft_fallback = False

    lane_width = self._lane_width(inputs, MIN_LANE_LINE_PROB)
    self.planned = bool(lane_width is not None and math.isfinite(inputs.prev_desired_curvature) and abs(inputs.prev_desired_curvature) <= MAX_ENTRY_CURVATURE)
    if self.planned:
      self.baseline_curvature = inputs.prev_desired_curvature
      self.lane_width = lane_width
    else:
      self.baseline_curvature = 0.0
      self.lane_width = 0.0

  def _should_soft_fallback(self, inputs: LaneChangeSCurveInputs) -> bool:
    lane_width = self._lane_width(inputs, SOFT_FALLBACK_LANE_LINE_PROB, clamp=False)
    if lane_width is None:
      return True
    return abs(lane_width - self.lane_width) > LANE_WIDTH_TOLERANCE

  def _scripted_curvature(self, v_ego: float) -> float:
    progress = min(max(self.elapsed / LANE_CHANGE_DURATION, 0.0), 1.0)
    # A minimum-jerk septic profile softens the onset by driving lateral jerk to zero at both ends.
    accel_scale = (
      420.0 * progress * progress
      - 1680.0 * progress * progress * progress
      + 2100.0 * progress * progress * progress * progress
      - 840.0 * progress * progress * progress * progress * progress
    )
    direction_sign = -1.0 if self.direction == LaneChangeDirection.left else 1.0
    lateral_accel = direction_sign * self.lane_width * accel_scale / (LANE_CHANGE_DURATION**2)
    curvature_offset = lateral_accel / max(v_ego, MIN_VEGO) ** 2
    return self.baseline_curvature + curvature_offset

  def _hard_abort(self, inputs: LaneChangeSCurveInputs) -> bool:
    one_blinker = inputs.left_blinker != inputs.right_blinker
    direction_matches = (inputs.left_blinker and inputs.lane_change_direction == LaneChangeDirection.left) or (
      inputs.right_blinker and inputs.lane_change_direction == LaneChangeDirection.right
    )
    return not inputs.lat_active or not one_blinker or not direction_matches

  @staticmethod
  def _lane_change_active(inputs: LaneChangeSCurveInputs) -> bool:
    return inputs.lane_change_state in (LaneChangeState.laneChangeStarting, LaneChangeState.laneChangeFinishing) and inputs.lane_change_direction in (
      LaneChangeDirection.left,
      LaneChangeDirection.right,
    )

  def _approach(self, current: float, target: float) -> float:
    duration = ENTRY_BLEND_DURATION if target > current else EXIT_BLEND_DURATION
    step = self.dt / duration
    if current < target:
      return min(current + step, target)
    return max(current - step, target)

  @staticmethod
  def _lane_width(inputs: LaneChangeSCurveInputs, min_prob: float, clamp: bool = True) -> float | None:
    if len(inputs.lane_line_probs) <= 2:
      return None
    if inputs.left_lane_y0 is None or inputs.right_lane_y0 is None:
      return None
    if not math.isfinite(inputs.left_lane_y0) or not math.isfinite(inputs.right_lane_y0):
      return None
    if inputs.lane_line_probs[1] < min_prob or inputs.lane_line_probs[2] < min_prob:
      return None

    lane_width = inputs.right_lane_y0 - inputs.left_lane_y0
    if not math.isfinite(lane_width) or lane_width <= 0.0:
      return None
    if not clamp:
      return lane_width
    return min(max(lane_width, MIN_LANE_WIDTH), MAX_LANE_WIDTH)
