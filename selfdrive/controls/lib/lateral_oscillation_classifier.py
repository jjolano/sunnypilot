import math
from collections import deque
from dataclasses import dataclass, field

STRAIGHT_ROAD_MIN_SPEED = 20.0  # m/s
STRAIGHT_ROAD_MAX_CURVATURE = 3e-4  # 1/m
PLANNER_OSCILLATION_MIN_FLIPS = 3
CONTROLLER_OSCILLATION_TORQUE_FLIPS = 4
CONTROLLER_OSCILLATION_PROCESSED_FLIPS = 2
VEHICLE_BIAS_MIN_OFFSET = 1e-5  # 1/m
VEHICLE_BIAS_MIN_CONFIDENCE = 0.7
RECENTER_LAG_MIN_FRAMES = 5
SIGN_CHANGE_LAG_MIN_FRAMES = 3
STRAIGHT_ROAD_HUNTING_TORQUE_FLIPS = 6
STRAIGHT_ROAD_HUNTING_PROCESSED_FLIPS = 2


@dataclass(frozen=True)
class LateralOscillationClassification:
  classification: str  # one of: "planner_oscillation", "controller_oscillation", "vehicle_bias", "recenter_lag", "sign_change_lag", "straight_road_hunting", "none"
  confidence: float  # 0.0 to 1.0
  raw_curvature_sign_flips: int  # count of sign changes in raw curvature window
  processed_curvature_sign_flips: int  # count of sign changes in processed curvature window
  torque_sign_flips: int  # count of sign changes in torque output window
  curvature_offset: float  # mean offset of processed curvature (vehicle bias indicator)
  curvature_offset_confidence: float  # confidence in offset measurement
  recenter_lag_frames: int  # frames between target crossing zero and output crossing zero
  sign_change_lag_frames: int  # frames between target sign change and output sign change
  straight_road: bool  # whether conditions indicate straight-road driving
  path_quality: float  # path quality metric
  lane_change_active: bool  # whether lane change shaping is active
  debug: dict[str, object] = field(default_factory=dict)


LATERAL_OSCILLATION_TO_UINT8 = {
  "none": 0,
  "planner_oscillation": 1,
  "controller_oscillation": 2,
  "vehicle_bias": 3,
  "recenter_lag": 4,
  "sign_change_lag": 5,
  "straight_road_hunting": 6,
}

LATERAL_UINT8_TO_OSCILLATION = {value: key for key, value in LATERAL_OSCILLATION_TO_UINT8.items()}


def lateral_oscillation_to_uint8(classification: str) -> int:
  return LATERAL_OSCILLATION_TO_UINT8.get(classification, 0)


def uint8_to_lateral_oscillation(value: int) -> str:
  return LATERAL_UINT8_TO_OSCILLATION.get(int(value), "none")


WOBBLE_ACTIVE_CLASSIFICATIONS = frozenset({
  "planner_oscillation",
  "controller_oscillation",
  "straight_road_hunting",
})

WOBBLE_CONFIDENCE_THRESHOLD = 0.5


def is_wobble_active(classification: str, confidence: float) -> bool:
  return classification in WOBBLE_ACTIVE_CLASSIFICATIONS and confidence > WOBBLE_CONFIDENCE_THRESHOLD


@dataclass(frozen=True)
class WobbleResponse:
  classification: str
  confidence: float
  feedback_gain_multiplier: float
  damping_gain_multiplier: float
  source_active: bool
  source: str

  @property
  def is_neutral(self) -> bool:
    return self.feedback_gain_multiplier == 1.0 and self.damping_gain_multiplier == 1.0


WOBBLE_FEEDBACK_MULT_DEFAULT = 1.0
WOBBLE_DAMPING_MULT_DEFAULT = 1.0
WOBBLE_FEEDBACK_MULT_PLANNER_OSCILLATION = 0.7
WOBBLE_DAMPING_MULT_PLANNER_OSCILLATION = 1.2
WOBBLE_FEEDBACK_MULT_CONTROLLER_OSCILLATION = 0.6
WOBBLE_DAMPING_MULT_CONTROLLER_OSCILLATION = 1.5
WOBBLE_FEEDBACK_MULT_STRAIGHT_ROAD_HUNTING = 0.6
WOBBLE_DAMPING_MULT_STRAIGHT_ROAD_HUNTING = 1.5


def compute_wobble_response(classification: str, confidence: float) -> WobbleResponse:
  """Source-aware response to lateral oscillation patterns.

  Returns a WobbleResponse that the controller applies to its feedback and
  damping gains. The default is neutral (1.0, 1.0). The response targets
  the responsible layer only:

  - planner_oscillation: reduce feedback so the controller doesn't chase
    path noise. Slight damping boost.
  - controller_oscillation: aggressively reduce feedback and boost damping
    to stop the controller from amplifying itself.
  - straight_road_hunting: same as controller_oscillation.
  - other classifications: neutral.
  """
  if classification == "planner_oscillation" and confidence > WOBBLE_CONFIDENCE_THRESHOLD:
    return WobbleResponse(
      classification=classification,
      confidence=confidence,
      feedback_gain_multiplier=WOBBLE_FEEDBACK_MULT_PLANNER_OSCILLATION,
      damping_gain_multiplier=WOBBLE_DAMPING_MULT_PLANNER_OSCILLATION,
      source_active=True,
      source="planner",
    )
  if classification == "controller_oscillation" and confidence > WOBBLE_CONFIDENCE_THRESHOLD:
    return WobbleResponse(
      classification=classification,
      confidence=confidence,
      feedback_gain_multiplier=WOBBLE_FEEDBACK_MULT_CONTROLLER_OSCILLATION,
      damping_gain_multiplier=WOBBLE_DAMPING_MULT_CONTROLLER_OSCILLATION,
      source_active=True,
      source="controller",
    )
  if classification == "straight_road_hunting" and confidence > WOBBLE_CONFIDENCE_THRESHOLD:
    return WobbleResponse(
      classification=classification,
      confidence=confidence,
      feedback_gain_multiplier=WOBBLE_FEEDBACK_MULT_STRAIGHT_ROAD_HUNTING,
      damping_gain_multiplier=WOBBLE_DAMPING_MULT_STRAIGHT_ROAD_HUNTING,
      source_active=True,
      source="straight_road",
    )
  return WobbleResponse(
    classification=classification,
    confidence=confidence,
    feedback_gain_multiplier=WOBBLE_FEEDBACK_MULT_DEFAULT,
    damping_gain_multiplier=WOBBLE_DAMPING_MULT_DEFAULT,
    source_active=False,
    source="none",
  )


class LateralOscillationClassifier:
  """Diagnostic lateral oscillation classifier.

  Maintains rolling windows of lateral signals and classifies oscillation
  patterns. This module is diagnostic-only and does NOT change any control
  behavior.
  """

  def __init__(self, window_frames: int = 100):
    self.window_frames = window_frames

    # Rolling windows
    self._raw_curvature: deque[float] = deque(maxlen=window_frames)
    self._processed_curvature: deque[float] = deque(maxlen=window_frames)
    self._target_lateral_accel: deque[float] = deque(maxlen=window_frames)
    self._actual_lateral_accel: deque[float] = deque(maxlen=window_frames)
    self._torque_output: deque[float] = deque(maxlen=window_frames)
    self._straight_road_flags: deque[bool] = deque(maxlen=window_frames)

    # Lag tracking state
    self._recenter_waiting: bool = False
    self._recenter_counter: int = 0
    self._last_recenter_lag: int = 0

    self._sign_change_waiting: bool = False
    self._sign_change_counter: int = 0
    self._last_sign_change_lag: int = 0

    self._prev_target_lat_accel: float = 0.0
    self._prev_processed_curvature: float = 0.0
    self._prev_torque_output: float = 0.0

  def update(self, *, raw_curvature, processed_curvature, target_lateral_accel,
             actual_lateral_accel, torque_output, path_quality, lane_change_active,
             v_ego, curvature_limited, steering_pressed, **kwargs) -> LateralOscillationClassification:
    """Update rolling windows and return a classification for the current state."""

    # --- Push new values into rolling windows ---
    self._raw_curvature.append(float(raw_curvature))
    self._processed_curvature.append(float(processed_curvature))
    self._target_lateral_accel.append(float(target_lateral_accel))
    self._actual_lateral_accel.append(float(actual_lateral_accel))
    self._torque_output.append(float(torque_output))

    # Determine straight-road condition for this frame
    frame_straight = (
      float(v_ego) >= STRAIGHT_ROAD_MIN_SPEED
      and abs(float(processed_curvature)) <= STRAIGHT_ROAD_MAX_CURVATURE
      and not bool(lane_change_active)
    )
    self._straight_road_flags.append(frame_straight)

    # --- Latest frame values ---
    cur_target_lat = float(target_lateral_accel)
    cur_proc_curv = float(processed_curvature)
    cur_torque = float(torque_output)

    # --- Track recenter lag ---
    # Target crosses zero from non-zero -> start counting
    target_crossed = self._prev_target_lat_accel != 0.0 and cur_target_lat * self._prev_target_lat_accel < 0
    if target_crossed:
      self._recenter_waiting = True
      self._recenter_counter = 0

    if self._recenter_waiting:
      self._recenter_counter += 1
      # Actual crosses zero -> record lag and reset
      actual_zero = self._actual_lateral_accel[-1] == 0.0 or (
        len(self._actual_lateral_accel) >= 2
        and self._actual_lateral_accel[-2] != 0.0
        and self._actual_lateral_accel[-2] * self._actual_lateral_accel[-1] < 0
      )
      if actual_zero:
        self._last_recenter_lag = self._recenter_counter
        self._recenter_waiting = False

    # --- Track sign change lag ---
    # Processed curvature changes sign -> start counting
    proc_crossed = self._prev_processed_curvature != 0.0 and cur_proc_curv * self._prev_processed_curvature < 0
    if proc_crossed:
      self._sign_change_waiting = True
      self._sign_change_counter = 0

    if self._sign_change_waiting:
      self._sign_change_counter += 1
      # Torque output changes sign -> record lag and reset
      torque_crossed = cur_torque != 0.0 and self._prev_torque_output != 0.0 and cur_torque * self._prev_torque_output < 0
      if torque_crossed:
        self._last_sign_change_lag = self._sign_change_counter
        self._sign_change_waiting = False

    # --- Update previous values ---
    self._prev_target_lat_accel = cur_target_lat
    self._prev_processed_curvature = cur_proc_curv
    self._prev_torque_output = cur_torque

    # --- Compute metrics ---
    raw_flips = self._count_sign_flips(self._raw_curvature)
    proc_flips = self._count_sign_flips(self._processed_curvature)
    torque_flips = self._count_sign_flips(self._torque_output)

    straight_road, curvature_offset, offset_conf = self._compute_curvature_offset()

    recenter_lag = self._last_recenter_lag
    sign_change_lag = self._last_sign_change_lag

    # --- Classify ---
    classification, confidence = self._classify(
      raw_flips, proc_flips, torque_flips,
      curvature_offset, offset_conf,
      recenter_lag, sign_change_lag,
      straight_road, lane_change_active,
    )

    return LateralOscillationClassification(
      classification=classification,
      confidence=confidence,
      raw_curvature_sign_flips=raw_flips,
      processed_curvature_sign_flips=proc_flips,
      torque_sign_flips=torque_flips,
      curvature_offset=curvature_offset,
      curvature_offset_confidence=offset_conf,
      recenter_lag_frames=recenter_lag,
      sign_change_lag_frames=sign_change_lag,
      straight_road=straight_road,
      path_quality=float(path_quality),
      lane_change_active=bool(lane_change_active),
      debug={},
    )

  @staticmethod
  def _count_sign_flips(values: deque) -> int:
    """Count the number of times the sign changes between consecutive values."""
    flips = 0
    for i in range(1, len(values)):
      if values[i] * values[i - 1] < 0:
        flips += 1
    return flips

  def _compute_curvature_offset(self):
    """Compute straight-road flag, curvature offset, and offset confidence.

    Returns:
      (straight_road, curvature_offset, offset_confidence)
        straight_road: True if all frames in the window satisfy straight-road conditions.
        curvature_offset: Mean of processed_curvature on straight-road frames.
        offset_confidence: Fraction of window frames that were straight-road.
    """
    window_len = len(self._straight_road_flags)
    if window_len == 0:
      return True, 0.0, 0.0

    straight_count = sum(self._straight_road_flags)
    offset_conf = straight_count / window_len

    # straight_road is considered True when the majority of the window is straight
    all_straight = straight_count == window_len

    if offset_conf > 0.5:
      offset_values = [
        self._processed_curvature[i]
        for i in range(window_len)
        if self._straight_road_flags[i]
      ]
      offset = sum(offset_values) / len(offset_values) if offset_values else 0.0
    else:
      offset = 0.0

    return all_straight, offset, offset_conf

  def _classify(self, raw_flips, proc_flips, torque_flips,
                curvature_offset, offset_conf,
                recenter_lag, sign_change_lag,
                straight_road, lane_change_active):
    """Classify oscillation pattern based on computed metrics.

    Returns:
      (classification_string, confidence)
    """
    if lane_change_active:
      return "none", 0.0

    # a) planner_oscillation
    if raw_flips > PLANNER_OSCILLATION_MIN_FLIPS and proc_flips > PLANNER_OSCILLATION_MIN_FLIPS:
      confidence = min(1.0, (raw_flips + proc_flips) / (2 * PLANNER_OSCILLATION_MIN_FLIPS + 4))
      return "planner_oscillation", confidence

    # b) controller_oscillation
    if proc_flips <= CONTROLLER_OSCILLATION_PROCESSED_FLIPS and torque_flips > CONTROLLER_OSCILLATION_TORQUE_FLIPS:
      confidence = min(1.0, torque_flips / (CONTROLLER_OSCILLATION_TORQUE_FLIPS + 4))
      return "controller_oscillation", confidence

    # c) vehicle_bias
    if abs(curvature_offset) > VEHICLE_BIAS_MIN_OFFSET and offset_conf > VEHICLE_BIAS_MIN_CONFIDENCE:
      confidence = min(1.0, abs(curvature_offset) / (VEHICLE_BIAS_MIN_OFFSET * 5))
      return "vehicle_bias", confidence

    # d) recenter_lag
    if recenter_lag > RECENTER_LAG_MIN_FRAMES:
      confidence = min(1.0, recenter_lag / (RECENTER_LAG_MIN_FRAMES + 10))
      return "recenter_lag", confidence

    # e) sign_change_lag
    if sign_change_lag > SIGN_CHANGE_LAG_MIN_FRAMES:
      confidence = min(1.0, sign_change_lag / (SIGN_CHANGE_LAG_MIN_FRAMES + 7))
      return "sign_change_lag", confidence

    # f) straight_road_hunting
    if straight_road and torque_flips > STRAIGHT_ROAD_HUNTING_TORQUE_FLIPS and proc_flips > STRAIGHT_ROAD_HUNTING_PROCESSED_FLIPS:
      confidence = min(1.0, torque_flips / (STRAIGHT_ROAD_HUNTING_TORQUE_FLIPS + 4))
      return "straight_road_hunting", confidence

    # g) none
    return "none", 0.0
