import math
from dataclasses import dataclass
from collections.abc import Sequence

from openpilot.cereal import log
from openpilot.common.realtime import DT_CTRL


LaneChangeState = log.LaneChangeState
LaneChangeDirection = log.LaneChangeDirection

LANE_CHANGE_DURATION = 5.5
ENTRY_BLEND_DURATION = 1.6
EXIT_BLEND_DURATION = 0.8
EARLY_TURN_IN_AUTHORITY_DURATION = 1.8
MIN_LANE_LINE_PROB = 0.6
SOFT_FALLBACK_LANE_LINE_PROB = 0.45
MIN_LANE_WIDTH = 3.0
MAX_LANE_WIDTH = 4.0
NOMINAL_LANE_WIDTH = 3.6
LANE_WIDTH_GEOMETRY_WEIGHT = 0.6
LANE_WIDTH_TOLERANCE = 0.8
MAX_ENTRY_CURVATURE = 0.0015
MIN_VEGO = 1.0


@dataclass
class LaneChangePathShaperInputs:
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
class LaneChangePathShaperResult:
  desired_curvature: float
  blend: float
  active: bool
  soft_fallback: bool


class LaneChangePathShaper:
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
    self.blend_phase = 0.0
    self.baseline_curvature = 0.0
    self.lane_width = 0.0
    self.curvature_speed_floor = MIN_VEGO

  def update(self, inputs: LaneChangePathShaperInputs) -> LaneChangePathShaperResult:
    if not self._lane_change_active(inputs):
      self.reset()
      return LaneChangePathShaperResult(inputs.model_curvature, 0.0, False, False)

    if self._hard_abort(inputs):
      self.reset()
      return LaneChangePathShaperResult(inputs.model_curvature, 0.0, False, False)

    continuing_session = self.session_active and inputs.lane_change_direction == self.direction
    if not self.session_active or inputs.lane_change_direction != self.direction:
      self._start_session(inputs)

    if not self.planned:
      return LaneChangePathShaperResult(inputs.model_curvature, 0.0, False, False)

    driver_override = inputs.steering_pressed and continuing_session and self.elapsed > 0.0
    if not self.soft_fallback and (driver_override or self._should_soft_fallback(inputs)):
      self.soft_fallback = True

    finishing = inputs.lane_change_state == LaneChangeState.laneChangeFinishing
    target_phase = 1.0 if not finishing and not self.soft_fallback and self.elapsed < LANE_CHANGE_DURATION else 0.0
    # Ramp a linear phase, then ease the blend through a quintic smootherstep so the very start and
    # end of the maneuver have zero steering-rate (and zero steering-accel) — no wheel-jerk at the
    # endpoints, where a raw linear ramp would step the curvature rate. Timing is unchanged.
    self.blend_phase = self._approach(self.blend_phase, target_phase)
    self.blend = self._smootherstep(self.blend_phase)

    self.curvature_speed_floor = max(self.curvature_speed_floor, self._finite_v_ego(inputs.v_ego))
    reference_curvature = self._reference_curvature(inputs.v_ego)
    shaped_curvature = self._model_primary_curvature(inputs.model_curvature, reference_curvature)
    authority = self.blend * self._early_turn_in_authority()
    desired_curvature = inputs.model_curvature + authority * (shaped_curvature - inputs.model_curvature)

    if not finishing and not self.soft_fallback and self.elapsed < LANE_CHANGE_DURATION:
      self.elapsed = min(self.elapsed + self.dt, LANE_CHANGE_DURATION)

    return LaneChangePathShaperResult(desired_curvature, self.blend, self.blend > 1e-3, self.soft_fallback)

  def _start_session(self, inputs: LaneChangePathShaperInputs) -> None:
    self.session_active = True
    self.direction = inputs.lane_change_direction
    self.elapsed = 0.0
    self.blend = 0.0
    self.blend_phase = 0.0
    self.soft_fallback = False

    lane_width = self._lane_width(inputs, MIN_LANE_LINE_PROB)
    self.planned = False
    if lane_width is not None and math.isfinite(inputs.prev_desired_curvature) and abs(inputs.prev_desired_curvature) <= MAX_ENTRY_CURVATURE:
      self.planned = True
      self.baseline_curvature = inputs.prev_desired_curvature
      self.lane_width = self._smoothed_lane_width(lane_width)
      self.curvature_speed_floor = self._finite_v_ego(inputs.v_ego)
    else:
      self.baseline_curvature = 0.0
      self.lane_width = 0.0
      self.curvature_speed_floor = MIN_VEGO

  def _should_soft_fallback(self, inputs: LaneChangePathShaperInputs) -> bool:
    lane_width = self._lane_width(inputs, SOFT_FALLBACK_LANE_LINE_PROB, clamp=False)
    if lane_width is None:
      return True
    return abs(lane_width - self.lane_width) > LANE_WIDTH_TOLERANCE

  def _reference_curvature(self, v_ego: float) -> float:
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
    # Keep decel/coast from converting the same lateral profile into larger curvature demand.
    curvature_v_ego = max(self._finite_v_ego(v_ego), self.curvature_speed_floor)
    curvature_offset = lateral_accel / curvature_v_ego ** 2
    return self.baseline_curvature + curvature_offset

  def _model_primary_curvature(self, model_curvature: float, reference_curvature: float) -> float:
    model_offset = model_curvature - self.baseline_curvature
    reference_offset = reference_curvature - self.baseline_curvature
    if model_offset * reference_offset > 0.0 and abs(model_offset) >= abs(reference_offset):
      return model_curvature
    return reference_curvature

  def _early_turn_in_authority(self) -> float:
    if self.elapsed >= EARLY_TURN_IN_AUTHORITY_DURATION:
      return 1.0
    progress = min(max(self.elapsed / EARLY_TURN_IN_AUTHORITY_DURATION, 0.0), 1.0)
    return progress * progress * progress * (10.0 + progress * (-15.0 + 6.0 * progress))

  @staticmethod
  def _smootherstep(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * x * (10.0 + x * (-15.0 + 6.0 * x))

  @staticmethod
  def _smoothed_lane_width(lane_width: float) -> float:
    return NOMINAL_LANE_WIDTH + LANE_WIDTH_GEOMETRY_WEIGHT * (lane_width - NOMINAL_LANE_WIDTH)

  @staticmethod
  def _finite_v_ego(v_ego: float) -> float:
    try:
      value = float(v_ego)
    except (TypeError, ValueError):
      return MIN_VEGO
    return max(value, MIN_VEGO) if math.isfinite(value) else MIN_VEGO

  def _hard_abort(self, inputs: LaneChangePathShaperInputs) -> bool:
    one_blinker = inputs.left_blinker != inputs.right_blinker
    direction_matches = (inputs.left_blinker and inputs.lane_change_direction == LaneChangeDirection.left) or (
      inputs.right_blinker and inputs.lane_change_direction == LaneChangeDirection.right
    )
    return not inputs.lat_active or not one_blinker or not direction_matches

  @staticmethod
  def _lane_change_active(inputs: LaneChangePathShaperInputs) -> bool:
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
  def _lane_width(inputs: LaneChangePathShaperInputs, min_prob: float, clamp: bool = True) -> float | None:
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
