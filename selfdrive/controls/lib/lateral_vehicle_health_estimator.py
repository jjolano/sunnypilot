import math
from dataclasses import dataclass, field
from typing import Optional

# --- Constants ---

# Learning conditions
HEALTH_EST_MIN_SPEED = 15.0  # m/s
HEALTH_EST_MAX_SPEED = 40.0  # m/s
HEALTH_EST_MIN_PATH_QUALITY = 0.7
HEALTH_EST_MAX_CURVATURE = 2e-4  # 1/m — near-straight only
HEALTH_EST_MIN_PERSISTENCE_FRAMES = 100  # ~2 seconds at 50 Hz
HEALTH_EST_SPEED_STABILITY_WINDOW = 20  # frames
HEALTH_EST_SPEED_STABILITY_MAX_DELTA = 1.0  # m/s

# Bounds
HEALTH_EST_BIAS_MAX = 0.06  # m/s² — hard bound on bias estimate
HEALTH_EST_BIAS_WARNING = 0.04  # m/s² — diagnostic warning threshold
HEALTH_EST_ASYMMETRY_MAX = 0.15  # — hard bound on response asymmetry
HEALTH_EST_RECENTER_LAG_MAX_FRAMES = 20  # — hard bound on recenter lag

# Smoothing
HEALTH_EST_BIAS_ALPHA = 0.005  # exponential moving average alpha
HEALTH_EST_ASYMMETRY_ALPHA = 0.003

# --- Helpers ---


def _finite(*values: float) -> bool:
  """Return True if all values are finite (not NaN, not inf)."""
  return all(math.isfinite(v) for v in values)


def _clip(value: float, low: float, high: float) -> float:
  """Clamp *value* to the closed interval [low, high]."""
  return max(low, min(high, value))


# --- Dataclass ---


@dataclass(frozen=True)
class LateralVehicleHealthEstimate:
  bias_estimate: float = 0.0  # m/s² — persistent lateral accel offset on straight road
  bias_confidence: float = 0.0  # 0-1 — how many valid samples contributed
  bias_warning: bool = False  # True if |bias| > HEALTH_EST_BIAS_WARNING
  left_response_estimate: float = 0.0  # normalized response gain for left turns
  right_response_estimate: float = 0.0  # normalized response gain for right turns
  response_asymmetry: float = 0.0  # |left - right| / max(left, right)
  recenter_lag_frames: int = 0  # frames between target crossing zero and output crossing zero
  persistence_frames: int = 0  # total valid frames accumulated
  learning_active: bool = False  # whether conditions allow learning this frame
  debug: dict = field(default_factory=dict)


# --- Estimator ---


class LateralVehicleHealthEstimator:
  """Session-only lateral vehicle health estimator.

  This is a **diagnostic-only** module.  It estimates vehicle lateral-health
  characteristics but does **not** change any control behaviour.  It only
  produces estimates and diagnostic warnings.
  """

  def __init__(self, dt: float = 0.01):
    self.dt = dt
    self.reset()

  # ------------------------------------------------------------------
  # Public API
  # ------------------------------------------------------------------

  def reset(self) -> None:
    """Reset all estimates.  Called at session start."""
    self._bias_ema = 0.0
    self._bias_sample_count = 0
    self._left_response_sum = 0.0
    self._left_response_count = 0
    self._right_response_sum = 0.0
    self._right_response_count = 0
    self._recenter_lag_frames = 0
    self._recenter_target_sign = 0
    self._recenter_target_cross_frame = -1
    self._frame_count = 0
    self._speed_window: list[float] = []

  def update(self, *, v_ego: float, target_lateral_accel: float,
             actual_lateral_accel: float, path_quality: float,
             demand_source: str, lane_change_active: bool,
             steering_pressed: bool, curvature_limited: bool,
             saturated: bool) -> LateralVehicleHealthEstimate:
    """Update the estimator with current frame data.

    Parameters
    ----------
    v_ego :
        Vehicle speed in m/s.
    target_lateral_accel :
        The lateral acceleration commanded by the planner / controller
        in m/s².
    actual_lateral_accel :
        The measured lateral acceleration in m/s².
    path_quality :
        Quality of the model path [0, 1].
    demand_source :
        Source of the lateral demand (e.g. ``"model_path"``).
    lane_change_active :
        Whether a lane change is in progress.
    steering_pressed :
        Whether the driver is applying steering torque.
    curvature_limited :
        Whether the curvature limiter is active.
    saturated :
        Whether the controller output is saturated.
    """
    self._frame_count += 1

    # Check learning conditions
    learning_active = self._check_learning_conditions(
      v_ego=v_ego, path_quality=path_quality, demand_source=demand_source,
      lane_change_active=lane_change_active, steering_pressed=steering_pressed,
      curvature_limited=curvature_limited, saturated=saturated,
    )

    if learning_active:
      self._update_bias(actual_lateral_accel)
      self._update_response_asymmetry(target_lateral_accel, actual_lateral_accel)
      self._update_recenter_lag(target_lateral_accel, actual_lateral_accel)

    # Build estimate
    bias_estimate = _clip(self._bias_ema, -HEALTH_EST_BIAS_MAX, HEALTH_EST_BIAS_MAX)
    bias_confidence = min(1.0, self._bias_sample_count / HEALTH_EST_MIN_PERSISTENCE_FRAMES)

    left_avg = self._left_response_sum / max(self._left_response_count, 1)
    right_avg = self._right_response_sum / max(self._right_response_count, 1)
    response_asymmetry = abs(left_avg - right_avg) / max(abs(left_avg), abs(right_avg), 1e-6)
    response_asymmetry = min(response_asymmetry, HEALTH_EST_ASYMMETRY_MAX)

    return LateralVehicleHealthEstimate(
      bias_estimate=bias_estimate,
      bias_confidence=bias_confidence,
      bias_warning=abs(bias_estimate) > HEALTH_EST_BIAS_WARNING,
      left_response_estimate=left_avg,
      right_response_estimate=right_avg,
      response_asymmetry=response_asymmetry,
      recenter_lag_frames=min(self._recenter_lag_frames, HEALTH_EST_RECENTER_LAG_MAX_FRAMES),
      persistence_frames=self._bias_sample_count,
      learning_active=learning_active,
    )

  # ------------------------------------------------------------------
  # Internal helpers
  # ------------------------------------------------------------------

  def _check_learning_conditions(self, *, v_ego, path_quality, demand_source,
                                   lane_change_active, steering_pressed,
                                   curvature_limited, saturated) -> bool:
    if not _finite(v_ego, path_quality):
      return False
    if v_ego < HEALTH_EST_MIN_SPEED or v_ego > HEALTH_EST_MAX_SPEED:
      return False
    if path_quality < HEALTH_EST_MIN_PATH_QUALITY:
      return False
    if demand_source != "model_path":
      return False
    if lane_change_active or steering_pressed or curvature_limited or saturated:
      return False
    # Check speed stability
    self._speed_window.append(v_ego)
    if len(self._speed_window) > HEALTH_EST_SPEED_STABILITY_WINDOW:
      self._speed_window = self._speed_window[-HEALTH_EST_SPEED_STABILITY_WINDOW:]
    if len(self._speed_window) >= HEALTH_EST_SPEED_STABILITY_WINDOW:
      if max(self._speed_window) - min(self._speed_window) > HEALTH_EST_SPEED_STABILITY_MAX_DELTA:
        return False
    return True

  def _update_bias(self, actual_lateral_accel: float) -> None:
    """Exponential-moving-average update of lateral-acceleration bias."""
    self._bias_sample_count += 1
    if self._bias_sample_count == 1:
      self._bias_ema = actual_lateral_accel
    else:
      self._bias_ema += HEALTH_EST_BIAS_ALPHA * (actual_lateral_accel - self._bias_ema)

  def _update_response_asymmetry(self, target_lateral_accel: float,
                                  actual_lateral_accel: float) -> None:
    """Accumulate response ratios separately for left vs right turns."""
    if abs(target_lateral_accel) < 0.01:
      return  # Skip near-zero targets
    ratio = actual_lateral_accel / target_lateral_accel if abs(target_lateral_accel) > 1e-6 else 0.0
    if not _finite(ratio):
      return
    if target_lateral_accel > 0:
      self._left_response_sum += ratio
      self._left_response_count += 1
    else:
      self._right_response_sum += ratio
      self._right_response_count += 1

  def _update_recenter_lag(self, target_lateral_accel: float,
                            actual_lateral_accel: float) -> None:
    """Measure how many frames it takes for actual accel to cross zero
    after target accel crosses zero."""
    current_target_sign = 1 if target_lateral_accel > 0.05 else (-1 if target_lateral_accel < -0.05 else 0)
    current_actual_sign = 1 if actual_lateral_accel > 0.05 else (-1 if actual_lateral_accel < -0.05 else 0)

    # Detect target crossing zero
    if current_target_sign == 0 and self._recenter_target_sign != 0:
      self._recenter_target_cross_frame = self._frame_count
      self._recenter_target_sign = 0
    elif current_target_sign != 0:
      self._recenter_target_sign = current_target_sign

    # Detect actual crossing zero after target crossing
    if self._recenter_target_cross_frame > 0 and current_actual_sign == 0:
      lag = self._frame_count - self._recenter_target_cross_frame
      if 0 < lag <= HEALTH_EST_RECENTER_LAG_MAX_FRAMES:
        self._recenter_lag_frames = lag
      self._recenter_target_cross_frame = -1
