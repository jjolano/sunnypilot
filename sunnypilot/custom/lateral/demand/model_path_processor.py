import collections
import math
from dataclasses import dataclass, replace
from collections.abc import Sequence

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
LOW_LANE_CONFIDENCE_SUSTAIN_FRAMES = 3
LOW_LANE_CONFIDENCE_SUSTAINED_QUALITY = 0.65

# SAE J3240-inspired degradation tiers (from published NCHRP/SAE lane-marking research).
# These provide named references for common environmental conditions:
#   - Worn markings: lane_line_probs 0.40–0.75
#   - Wet night:      lane_line_probs 0.30–0.70
#   - Night rain:     lane_line_probs 0.10–0.40
#   - Heavy fog:      lane_line_probs 0.00–0.10
# LOW_LANE_LINE_PROB (0.35) sits between wet-night and night-rain — reasonable for
# onset of confidence reduction. The tiers below document the envelope for future tuning.
SAE_WET_NIGHT_PROB = 0.30     # lane_line_probs expected in wet night driving
SAE_HEAVY_FOG_PROB = 0.10    # lane_line_probs expected in fog < 50 m visibility
SAE_FOG_SUSTAIN_FRAMES = 30  # frames before heavy-fog fallback (300ms at 100Hz)
HIGH_FRAME_DROP_PERC = 20.0
MODEL_STALE_AGE_S = 0.20
LOW_QUALITY_BLEND_THRESHOLD = 0.75
LOW_QUALITY_BLEND_MIN_ALPHA = 0.4
HARD_INVALID_FALLBACK_MEASURED_ALPHA = 0.25
SOFT_GATE_HOLD_FRAMES = 2
LOW_SPEED_SOFT_GATE_SPEED = 12.0
LOW_SPEED_SOFT_GATE_MAX_EXTRA_FRAMES = 3
SOFT_GATE_HOLD_QUALITY = 0.70
SOFT_GATE_REASONS = frozenset(("high_path_std", "frame_drop", "model_stale", "path_disagreement", "low_lane_confidence"))
SOFT_GATE_MAX_SAME_SIGN_RAW_LAT_ACCEL_DELTA = 0.04  # m/s^2
LOW_SPEED_UNTRUSTED_CURVATURE_STEP = 0.0025
LOW_SPEED_CURVE_RETENTION_FRAMES = 12
LOW_SPEED_CURVE_RETENTION_MIN_CURVATURE = 0.008
HARD_INVALID_RECOVERY_LAT_JERK = 2.0
LOW_SPEED_CONFIRMED_TURN_MIN_LAT_ACCEL = 0.05
LOW_SPEED_CONFIRMED_TURN_MAX_LAT_ACCEL_DELTA = 0.75
SMOOTHED_CURVATURE_MIN_SPEED = 5.0
SMOOTHED_CURVATURE_MIN_SAMPLES = 5
SMOOTHED_CURVATURE_SPEED_BP = [5.0, 15.0, 30.0]
SMOOTHED_CURVATURE_WINDOW_S = [0.28, 0.45, 0.65]
SMOOTHED_CURVATURE_BLEND_ALPHA = [0.20, 0.35, 0.50]
SMOOTHED_CURVATURE_MAX_LAT_ACCEL_DELTA = [0.20, 0.35, 0.50]
SMOOTHED_CURVATURE_MAX_RAW_LAT_ACCEL_DISAGREEMENT = 1.25

# Straight-path stabilization: conservative near-straight anchor.
SPS_MAX_FRAME_DROP_PERC = 5.0
SPS_MAX_MODEL_AGE_S = 0.20
SPS_MAX_PATH_Y_STD = 0.45
SPS_MAX_PATH_DISAGREEMENT = 0.35
SPS_MIN_SPEED = 12.0
SPS_MAX_SPEED = 26.0
SPS_MAX_RAW_LAT_ACCEL = 0.60
SPS_MAX_TARGET_LAT_ACCEL = 0.60
SPS_MAX_RAW_TARGET_LAT_ACCEL_DELTA = 0.30
SPS_ENTRY_MAX_RAW_LAT_JERK = 0.8
SPS_RELEASE_MAX_RAW_LAT_JERK = 1.0
SPS_MAX_SUPPRESSION_LAT_ACCEL = 0.35
SPS_ANCHOR_CLIP_LAT_ACCEL = 0.02
SPS_ANCHOR_WINDOW_S = 0.5
SPS_ANCHOR_WINDOW_FRAMES = int(round(SPS_ANCHOR_WINDOW_S / DT_CTRL))
SPS_MAX_PAUSE_FRAMES = int(round(1.5 / DT_CTRL))
SPS_ENTRY_MIN_FRAMES = int(round(0.3 / DT_CTRL))
SPS_SIGN_FLIP_EXEMPT_LAT_ACCEL = 0.05
SPS_RISING_FRAMES_THRESHOLD = 3
SPS_RISING_CUMULATIVE_LAT_ACCEL = 0.15
SPS_QUALITY_THRESHOLD = 0.85
SPS_MODES = frozenset(("off", "shadow", "apply"))


def sanitize_straight_path_stabilization_mode(mode: object) -> str:
  mode_s = str(mode).strip().lower()
  return mode_s if mode_s in SPS_MODES else "off"

# Temporal damping: tau (s) vs speed — larger tau at low speed (more smoothing).
DAMPING_TAU_SPEED_BP = [5.0, 15.0, 30.0]
DAMPING_TAU_S = [0.16, 0.10, 0.055]

# Experimental, default-off pre-governor demand smoothing.
DEMAND_JERK_SMOOTH_MIN_SPEED = 8.0
DEMAND_JERK_SMOOTH_MAX_SPEED = 22.0
DEMAND_JERK_SMOOTH_MIN_QUALITY = 0.95
DEMAND_JERK_SMOOTH_LOW_LANE_CONFIDENCE_MIN_QUALITY = 0.85
DEMAND_JERK_SMOOTH_MAX_FRAME_DROP_PERC = 5.0
DEMAND_JERK_SMOOTH_MIN_LANE_PROB = 0.65
DEMAND_JERK_SMOOTH_MAX_PATH_Y_STD = 0.45
DEMAND_JERK_SMOOTH_MAX_PATH_DISAGREEMENT = 0.35
DEMAND_JERK_SMOOTH_MAX_CURVATURE = 0.0012
DEMAND_JERK_SMOOTH_MAX_LAT_ACCEL = 0.35
DEMAND_JERK_SMOOTH_LAG_LAT_ACCEL = 0.08
DEMAND_JERK_SMOOTH_CURVE_EXIT_MAX_LAT_ACCEL = 1.0  # m/s^2
DEMAND_JERK_SMOOTH_CURVE_EXIT_FALL_MIN_LAT_ACCEL_DELTA = 0.03  # m/s^2
DEMAND_JERK_SMOOTH_CURVE_EXIT_FALL_MIN_FRAMES = 2
DEMAND_JERK_SMOOTH_CURVE_EXIT_NEAR_ZERO_LAT_ACCEL = 0.08  # m/s^2
DEMAND_JERK_SMOOTH_SPEED_BP = [8.0, 15.0, 22.0]
DEMAND_JERK_SMOOTH_MAX_LAT_JERK = [1.0, 1.4, 1.8]

# Straight-road damping: larger tau and deadband when driving near-straight.
# Gated by lateral acceleration so it does not over-damp at high speed.
STRAIGHT_ROAD_DAMPING_MIN_SPEED = 12.0
STRAIGHT_ROAD_DAMPING_FULL_SPEED = 20.0
STRAIGHT_ROAD_DAMPING_MAX_LAT_ACCEL = 0.20
STRAIGHT_ROAD_DAMPING_TAU_S = 0.35
STRAIGHT_ROAD_DAMPING_DEADBAND_LAT_ACCEL = 0.06
STRAIGHT_ROAD_DAMPING_BLEND_BP_LAT_ACCEL = [0.0, 0.20]
STRAIGHT_ROAD_DAMPING_BLEND_SCALE = [1.0, 0.0]

# Trust penalty after unstable frames.
TRUST_DECAY = 0.92
TRUST_BUMP = 0.38
TRUST_BUMP_REASONS = frozenset({"invalid_path", "frame_drop", "path_disagreement"})

# Soft hysteresis near zero curvature: scale spatial blend toward 1 as |kappa| increases.
NEAR_ZERO_CURVATURE_BP = [0.0, 0.00045]
NEAR_ZERO_BLEND_SCALE = [0.32, 1.0]

# Lane change: fade (smoothed - raw) correction toward raw.
LANE_CHANGE_OFFSET_FADE_S = 0.5


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
  model_age_s: float = 0.0
  smooth_model_path_curvature: bool = False
  demand_jerk_smoothing_enabled: bool = False
  demand_jerk_smoothing_allowed: bool = False
  lane_change_active: bool = False
  left_blinker: bool = False
  right_blinker: bool = False
  steering_pressed: bool | None = None
  steer_limited: bool = False
  straight_path_stabilization_mode: str = "off"


@dataclass
class ModelPathProcessorResult:
  desired_curvature: float
  quality: float
  gated: bool
  reason: str
  hold_frames_remaining: int = 0
  smoothing_tau_s: float = 0.0
  damping_alpha: float = 0.0
  trust_penalty: float = 0.0
  spatial_smoothed_curvature: float = 0.0
  lane_change_fade: float = 0.0
  straight_road_damping_active: bool = False
  demand_jerk_smoothing_active: bool = False
  demand_jerk_smoothing_step: float = 0.0
  demand_jerk_smoothing_lag: float = 0.0
  straight_path_stabilization_mode: str = "off"
  straight_path_stabilization_active: bool = False
  straight_path_stabilization_applied: bool = False
  straight_path_stabilization_candidate_curvature: float = 0.0
  straight_path_stabilization_anchor_lat_accel: float = 0.0
  straight_path_stabilization_reason: str = "disabled"


@dataclass
class _PathEvidence:
  """Quality/decision evidence produced from path observations."""
  quality: float
  reason: str
  path_disagreement: float | None


@dataclass
class _ReferenceSelection:
  """Fallback and retained-curve reference chosen for the current frame."""
  fallback: float
  retained_used: bool


@dataclass
class _ShapedCurvature:
  """Curvature after spatial and temporal shaping."""
  raw_base: float
  damped: float
  tau_s: float
  damping_alpha: float
  straight_road_active: bool
  spatial_smoothed: float


class ModelPathProcessor:
  def __init__(self) -> None:
    self.reset()

  def reset(self) -> None:
    self._hold_frames_remaining = 0
    self._hold_reason = "ok"
    self._hold_quality: float = SOFT_GATE_HOLD_QUALITY
    self._low_lane_confidence_frames: int = 0
    self._retained_curve_curvature: float | None = None
    self._retained_curve_frames: int = 0
    self._recovering_from_hard_invalid = False
    self._trust_penalty = 0.0
    self._temporal_smoothed_curvature: float | None = None
    self._lane_change_fade: float | None = None
    self._prev_lane_change_active = False
    self._last_smoothing_tau_s = 0.0
    self._last_damping_alpha = 0.0
    self._last_spatial_curvature = 0.0
    self._straight_road_damping_active = False
    self._demand_jerk_smoothed_curvature: float | None = None
    self._demand_jerk_smoothing_active = False
    self._last_demand_jerk_smoothing_step = 0.0
    self._last_demand_jerk_smoothing_lag = 0.0
    self._curve_exit_prev_raw_base: float | None = None
    self._curve_exit_fall_frames: int = 0
    self._prev_temporal_soft_boundary = False
    self._reset_straight_path_stabilization_state()

  def _reset_straight_path_stabilization_state(self) -> None:
    self._sp_anchor_buffer: collections.deque[float] = collections.deque(maxlen=SPS_ANCHOR_WINDOW_FRAMES)
    self._sp_prev_raw_lat_accel: float | None = None
    self._sp_rising_count: int = 0
    self._sp_rising_sum: float = 0.0
    self._sp_rising_sign: int = 0
    self._sp_pause_frames: int = 0

  # ---------------------------------------------------------------------------
  # Orchestration
  # ---------------------------------------------------------------------------
  def update(self, inputs: ModelPathProcessorInputs) -> ModelPathProcessorResult:
    self._last_smoothing_tau_s = 0.0
    self._last_damping_alpha = 0.0
    self._last_spatial_curvature = 0.0
    self._last_demand_jerk_smoothing_step = 0.0
    self._last_demand_jerk_smoothing_lag = 0.0

    # 1. inactive/nonfinite/stale/invalid terminal fallbacks
    early = self._terminal_fallbacks(inputs)
    if early is not None:
      self._reset_straight_path_stabilization_state()
      return early

    desired_curvature = float(inputs.desired_curvature)
    raw_base = desired_curvature
    fallback_curvature = self._fallback_curvature(inputs.previous_desired_curvature, inputs.measured_curvature)

    # turn opposite curvature
    if inputs.turn_curvature_sign != 0 and desired_curvature * inputs.turn_curvature_sign < 0.0:
      self._recovering_from_hard_invalid = False
      self._reset_curve_exit_history()
      self._reset_straight_path_stabilization_state()
      turn_fallback = self._turn_compatible_fallback_curvature(
        inputs.previous_desired_curvature,
        inputs.measured_curvature,
        inputs.turn_curvature_sign,
      )
      return ModelPathProcessorResult(
        turn_fallback, 0.5, True, "turn_opposite_curvature", 0,
        trust_penalty=self._trust_penalty,
        straight_road_damping_active=self._straight_road_damping_active,
      )

    # 2. quality scoring
    evidence = self._evaluate_path_evidence(inputs, desired_curvature)

    # 3. pre-smoothing implausible jump
    pre_jump = self._limit_implausible_jump(inputs.v_ego, desired_curvature, fallback_curvature)
    if pre_jump is not None:
      self._recovering_from_hard_invalid = False
      self._reset_curve_exit_history()
      self._reset_straight_path_stabilization_state()
      return replace(pre_jump, trust_penalty=self._trust_penalty, straight_road_damping_active=self._straight_road_damping_active)

    # 4. retained-curve refresh
    self._refresh_retained_curve(inputs, desired_curvature, evidence.quality, evidence.reason, evidence.path_disagreement)

    # 5. soft-gate hold/trust penalty
    quality, reason, hold_frames_remaining = self._apply_soft_gate_hold(evidence.quality, evidence.reason, inputs.v_ego)
    self._apply_temporal_soft_boundary(self._soft_temporal_boundary(quality, reason))
    if reason in TRUST_BUMP_REASONS:
      self._trust_penalty = min(1.0, self._trust_penalty + TRUST_BUMP)

    # 6. low-quality early return
    if quality < LOW_QUALITY_BLEND_THRESHOLD:
      return self._handle_low_quality(inputs, raw_base, fallback_curvature, quality, reason, hold_frames_remaining)

    # 7. spatial + temporal smoothing
    shaped = self._shape_curvature(inputs, raw_base, quality)

    # 8. demand-jerk smoothing
    demand_curvature, demand_active, demand_step, demand_lag = self._apply_demand_jerk_smoothing(
      inputs,
      shaped.raw_base,
      shaped.damped,
      quality,
      reason,
      evidence.path_disagreement,
    )
    curvature = demand_curvature

    # 9. lane-change fade
    curvature, lc_fade_report = self._apply_lane_change_fade(curvature, shaped.damped, shaped.raw_base, inputs)

    # 10. post-smoothing jump
    post_jump = self._limit_implausible_jump(inputs.v_ego, curvature, fallback_curvature)
    if post_jump is not None:
      self._recovering_from_hard_invalid = False
      self._reset_curve_exit_history()
      self._reset_straight_path_stabilization_state()
      return replace(
        post_jump,
        smoothing_tau_s=shaped.tau_s,
        damping_alpha=shaped.damping_alpha,
        trust_penalty=self._trust_penalty,
        spatial_smoothed_curvature=shaped.spatial_smoothed,
        lane_change_fade=lc_fade_report,
        straight_road_damping_active=shaped.straight_road_active,
        demand_jerk_smoothing_active=demand_active,
        demand_jerk_smoothing_step=demand_step,
        demand_jerk_smoothing_lag=demand_lag,
      )

    # 11. hard-invalid recovery
    recovery = self._limit_hard_invalid_recovery(inputs, curvature)
    if recovery is not None:
      self._reset_curve_exit_history()
      self._reset_straight_path_stabilization_state()
      return replace(
        recovery,
        smoothing_tau_s=shaped.tau_s,
        damping_alpha=shaped.damping_alpha,
        trust_penalty=self._trust_penalty,
        spatial_smoothed_curvature=shaped.spatial_smoothed,
        lane_change_fade=lc_fade_report,
        straight_road_damping_active=shaped.straight_road_active,
        demand_jerk_smoothing_active=demand_active,
        demand_jerk_smoothing_step=demand_step,
        demand_jerk_smoothing_lag=demand_lag,
      )

    sps_mode = sanitize_straight_path_stabilization_mode(inputs.straight_path_stabilization_mode)
    sps_output, sps_active, sps_applied, sps_candidate, sps_anchor, sps_reason = self._apply_straight_path_stabilization(
      replace(inputs, straight_path_stabilization_mode=sps_mode), raw_base, curvature, quality, reason, evidence.path_disagreement,
    )

    return ModelPathProcessorResult(
      sps_output,
      quality,
      False,
      reason,
      hold_frames_remaining,
      smoothing_tau_s=shaped.tau_s,
      damping_alpha=shaped.damping_alpha,
      trust_penalty=self._trust_penalty,
      spatial_smoothed_curvature=shaped.spatial_smoothed,
      lane_change_fade=lc_fade_report,
      straight_road_damping_active=shaped.straight_road_active,
      demand_jerk_smoothing_active=demand_active,
      demand_jerk_smoothing_step=demand_step,
      demand_jerk_smoothing_lag=demand_lag,
      straight_path_stabilization_mode=sps_mode,
      straight_path_stabilization_active=sps_active,
      straight_path_stabilization_applied=sps_applied,
      straight_path_stabilization_candidate_curvature=sps_candidate,
      straight_path_stabilization_anchor_lat_accel=sps_anchor,
      straight_path_stabilization_reason=sps_reason,
    )

  def _terminal_fallbacks(self, inputs: ModelPathProcessorInputs) -> ModelPathProcessorResult | None:
    """Handle lat_active/inactive and nonfinite/stale/invalid terminal states."""
    if not inputs.lat_active:
      self.reset()
      return ModelPathProcessorResult(
        float(inputs.measured_curvature), 0.0, True, "inactive",
        straight_road_damping_active=self._straight_road_damping_active,
      )

    self._trust_penalty *= TRUST_DECAY
    self._age_retained_curve()

    if not math.isfinite(inputs.desired_curvature):
      self._recovering_from_hard_invalid = False
      self._low_lane_confidence_frames = 0
      self._clear_retained_curve()
      self._reset_curve_exit_history()
      hard_invalid_fallback = self._hard_invalid_fallback_curvature(
        inputs.previous_desired_curvature,
        inputs.measured_curvature,
      )
      return ModelPathProcessorResult(
        hard_invalid_fallback, 0.0, True, "nonfinite_curvature", 0,
        trust_penalty=self._trust_penalty,
        straight_road_damping_active=self._straight_road_damping_active,
      )

    if not math.isfinite(inputs.model_age_s) or inputs.model_age_s > MODEL_STALE_AGE_S:
      self._recovering_from_hard_invalid = False
      self._low_lane_confidence_frames = 0
      self._clear_retained_curve()
      self._reset_curve_exit_history()
      self._trust_penalty = min(1.0, self._trust_penalty + TRUST_BUMP)
      self._apply_temporal_soft_boundary(True)
      fallback_curvature = self._hard_invalid_fallback_curvature(
        inputs.previous_desired_curvature,
        inputs.measured_curvature,
      )
      fallback_curvature = self._limit_low_speed_untrusted_curvature_step(
        inputs.v_ego,
        fallback_curvature,
        self._fallback_curvature(inputs.previous_desired_curvature, inputs.measured_curvature),
      )
      return ModelPathProcessorResult(
        fallback_curvature, 0.0, True, "model_stale", 0,
        trust_penalty=self._trust_penalty,
        straight_road_damping_active=self._straight_road_damping_active,
      )

    if not self._valid_core_path(inputs.position_x, inputs.position_y):
      self._recovering_from_hard_invalid = True
      self._low_lane_confidence_frames = 0
      self._reset_curve_exit_history()
      self._trust_penalty = min(1.0, self._trust_penalty + TRUST_BUMP)
      fallback_curvature = self._fallback_curvature(inputs.previous_desired_curvature, inputs.measured_curvature)
      retained_fallback = self._retained_curve_fallback(inputs, float(inputs.desired_curvature), fallback_curvature)
      hard_invalid_fallback = retained_fallback if retained_fallback is not None else self._hard_invalid_fallback_curvature(
        inputs.previous_desired_curvature,
        inputs.measured_curvature,
      )
      hard_invalid_fallback = self._limit_low_speed_untrusted_curvature_step(
        inputs.v_ego,
        hard_invalid_fallback,
        fallback_curvature,
      )
      return ModelPathProcessorResult(
        hard_invalid_fallback, 0.0, True, "invalid_path", 0,
        trust_penalty=self._trust_penalty,
        straight_road_damping_active=self._straight_road_damping_active,
      )

    return None

  # ---------------------------------------------------------------------------
  # Staged internals
  # ---------------------------------------------------------------------------
  def _evaluate_path_evidence(self, inputs: ModelPathProcessorInputs, desired_curvature: float) -> _PathEvidence:
    path_curvature = self._path_curvature(inputs.orientation_z, inputs.orientation_rate_z, inputs.v_ego)
    path_disagreement = None
    if path_curvature is not None:
      path_disagreement = abs(desired_curvature - path_curvature) * max(inputs.v_ego, 1.0) ** 2

    quality = 1.0
    reason = "ok"

    path_std_quality = self._path_std_quality(
      inputs.position_y_std,
      desired_curvature,
      path_disagreement,
      inputs.turn_curvature_sign,
    )
    if path_std_quality < quality:
      quality = path_std_quality
      reason = "high_path_std"

    lane_quality = self._lane_quality(inputs.lane_line_probs, inputs.v_ego)
    if lane_quality < quality:
      quality = lane_quality
      reason = "low_lane_confidence"

    if reason == "low_lane_confidence" and quality < LOW_QUALITY_BLEND_THRESHOLD and self._low_speed_measured_turn_confirms_curvature(
      inputs,
      desired_curvature,
      path_disagreement,
    ):
      quality = LOW_QUALITY_BLEND_THRESHOLD

    if math.isfinite(inputs.frame_drop_perc) and inputs.frame_drop_perc > HIGH_FRAME_DROP_PERC:
      quality = min(quality, SOFT_GATE_HOLD_QUALITY)
      reason = "frame_drop"

    if path_disagreement is not None and path_disagreement > MAX_PATH_CURVATURE_DISAGREEMENT:
      quality = min(quality, 0.65)
      reason = "path_disagreement"

    return _PathEvidence(quality=quality, reason=reason, path_disagreement=path_disagreement)

  def _select_reference_curvature(
    self,
    inputs: ModelPathProcessorInputs,
    desired_curvature: float,
    fallback_curvature: float,
  ) -> _ReferenceSelection:
    retained_fallback = self._retained_curve_fallback(inputs, desired_curvature, fallback_curvature)
    if retained_fallback is not None:
      return _ReferenceSelection(fallback=retained_fallback, retained_used=True)
    return _ReferenceSelection(fallback=fallback_curvature, retained_used=False)

  def _shape_curvature(self, inputs: ModelPathProcessorInputs, raw_base: float, quality: float) -> _ShapedCurvature:
    spatial_curvature: float | None = None
    if inputs.smooth_model_path_curvature:
      spatial_curvature = self._smoothed_path_curvature(inputs, raw_base, quality, self._trust_penalty)

    if spatial_curvature is not None:
      self._last_spatial_curvature = float(spatial_curvature)
      after_spatial = float(spatial_curvature)
      spatial_report = float(spatial_curvature)
    else:
      after_spatial = raw_base
      spatial_report = raw_base
      self._last_spatial_curvature = raw_base

    tau_s, damp_alpha, damped, straight_road_active = self._temporal_damp_curvature(
      inputs, after_spatial, bool(inputs.smooth_model_path_curvature),
    )
    self._last_smoothing_tau_s = tau_s
    self._last_damping_alpha = damp_alpha

    return _ShapedCurvature(
      raw_base=raw_base,
      damped=damped,
      tau_s=tau_s,
      damping_alpha=damp_alpha,
      straight_road_active=straight_road_active,
      spatial_smoothed=spatial_report,
    )

  def _handle_low_quality(
    self,
    inputs: ModelPathProcessorInputs,
    raw_base: float,
    fallback_curvature: float,
    quality: float,
    reason: str,
    hold_frames_remaining: int,
  ) -> ModelPathProcessorResult:
    self._recovering_from_hard_invalid = False
    selection = self._select_reference_curvature(inputs, raw_base, fallback_curvature)
    desired_curvature = self._blend(selection.fallback, raw_base, float(np.interp(
      quality, [0.0, LOW_QUALITY_BLEND_THRESHOLD], [LOW_QUALITY_BLEND_MIN_ALPHA, 1.0],
    )))
    if reason in SOFT_GATE_REASONS:
      desired_curvature = self._limit_low_speed_untrusted_curvature_step(
        inputs.v_ego,
        desired_curvature,
        selection.fallback,
      )
    if reason in SOFT_GATE_REASONS and not selection.retained_used:
      desired_curvature = self._limit_same_sign_amplification(
        raw_base,
        desired_curvature,
        inputs.v_ego,
        SOFT_GATE_MAX_SAME_SIGN_RAW_LAT_ACCEL_DELTA,
      )
    if reason in ("low_lane_confidence", "frame_drop"):
      self._pause_straight_path_stabilization(desired_curvature, f"pause_{reason}")
    else:
      self._reset_straight_path_stabilization_state()
    self._reset_curve_exit_history()
    return ModelPathProcessorResult(
      desired_curvature,
      quality,
      True,
      reason,
      hold_frames_remaining,
      trust_penalty=self._trust_penalty,
      straight_road_damping_active=self._straight_road_damping_active,
    )

  def _apply_lane_change_fade(
    self,
    curvature: float,
    damped: float,
    raw_base: float,
    inputs: ModelPathProcessorInputs,
  ) -> tuple[float, float]:
    fade_report = 0.0
    if inputs.smooth_model_path_curvature and inputs.lane_change_active:
      if not self._prev_lane_change_active or self._lane_change_fade is None:
        self._lane_change_fade = 1.0
      fade = float(self._lane_change_fade)
      fade_report = fade
      curvature = float(raw_base + (damped - raw_base) * fade)
      self._lane_change_fade = max(0.0, fade - DT_CTRL / LANE_CHANGE_OFFSET_FADE_S)
    else:
      self._lane_change_fade = None

    self._prev_lane_change_active = bool(inputs.lane_change_active)
    return curvature, fade_report

  def _apply_straight_path_stabilization(
    self,
    inputs: ModelPathProcessorInputs,
    raw_base: float,
    target: float,
    quality: float,
    reason: str,
    path_disagreement: float | None,
  ) -> tuple[float, bool, bool, float, float, str]:
    """Conservative near-straight path anchor. Returns
    (output_curvature, active, applied, candidate_curvature, anchor_lat_accel, debug_reason).
    In apply mode the candidate replaces the target when all gates hold; in shadow
    mode the candidate is computed and logged but the target is left unchanged.
    """
    mode = sanitize_straight_path_stabilization_mode(inputs.straight_path_stabilization_mode)
    if mode not in ("shadow", "apply"):
      self._reset_straight_path_stabilization_state()
      return target, False, False, 0.0, 0.0, "disabled"

    v_ego = float(inputs.v_ego)
    if not math.isfinite(v_ego) or v_ego < 1.0:
      return self._release_straight_path_stabilization(target, "gate_speed")

    speed_sq = v_ego * v_ego
    a_raw = raw_base * speed_sq
    a_target = target * speed_sq

    # Raw-jerk / same-sign rising-frame tracker (always update so release triggers).
    raw_jerk = 0.0
    if self._sp_prev_raw_lat_accel is not None and math.isfinite(self._sp_prev_raw_lat_accel):
      raw_jerk = (a_raw - self._sp_prev_raw_lat_accel) / DT_CTRL
      delta = a_raw - self._sp_prev_raw_lat_accel
      if abs(delta) > 1e-6:
        sign = 1 if delta > 0 else -1
        if sign == self._sp_rising_sign:
          self._sp_rising_count += 1
          self._sp_rising_sum += abs(delta)
        else:
          self._sp_rising_count = 1
          self._sp_rising_sum = abs(delta)
          self._sp_rising_sign = sign
      else:
        self._sp_rising_count = 0
        self._sp_rising_sum = 0.0
        self._sp_rising_sign = 0
    self._sp_prev_raw_lat_accel = float(a_raw)

    # Release gates (wider than entry gates).
    has_anchor = len(self._sp_anchor_buffer) >= SPS_ENTRY_MIN_FRAMES
    if abs(a_raw) > 0.75 or abs(a_target) > 0.75:
      return self._pause_straight_path_stabilization(target, "release_large_accel")
    if abs(a_raw - a_target) > 0.45:
      return self._pause_straight_path_stabilization(target, "release_raw_target_divergence")
    if abs(raw_jerk) > SPS_RELEASE_MAX_RAW_LAT_JERK and not has_anchor:
      return self._pause_straight_path_stabilization(target, "release_high_jerk")
    if (self._sp_rising_count >= SPS_RISING_FRAMES_THRESHOLD and
        self._sp_rising_sum > SPS_RISING_CUMULATIVE_LAT_ACCEL):
      return self._release_straight_path_stabilization(target, "release_rising_frames")

    # Entry gates.
    if v_ego < SPS_MIN_SPEED or v_ego > SPS_MAX_SPEED:
      return self._release_straight_path_stabilization(target, "gate_speed")
    if inputs.steering_pressed is not False:
      return self._release_straight_path_stabilization(target, "gate_steering_pressed")
    if inputs.lane_change_active:
      return self._release_straight_path_stabilization(target, "gate_lane_change")
    if inputs.left_blinker or inputs.right_blinker:
      return self._release_straight_path_stabilization(target, "gate_blinker")
    if inputs.turn_curvature_sign != 0:
      return self._release_straight_path_stabilization(target, "gate_turn_intent")
    if inputs.steer_limited:
      return self._pause_straight_path_stabilization(target, "gate_steer_limited")
    if reason not in ("ok", "low_lane_confidence"):
      return self._release_straight_path_stabilization(target, f"gate_reason_{reason}")
    if path_disagreement is None:
      return self._release_straight_path_stabilization(target, "gate_path_evidence")
    if quality < SPS_QUALITY_THRESHOLD:
      return self._release_straight_path_stabilization(target, "gate_quality")
    if math.isfinite(inputs.frame_drop_perc) and inputs.frame_drop_perc > SPS_MAX_FRAME_DROP_PERC:
      return self._release_straight_path_stabilization(target, "gate_frame_drop")
    if not math.isfinite(inputs.model_age_s) or inputs.model_age_s > SPS_MAX_MODEL_AGE_S:
      return self._release_straight_path_stabilization(target, "gate_model_stale")
    if not self._path_y_std_ok(inputs.position_y_std):
      return self._release_straight_path_stabilization(target, "gate_path_std")
    if path_disagreement is not None and path_disagreement > SPS_MAX_PATH_DISAGREEMENT:
      return self._release_straight_path_stabilization(target, "gate_path_disagreement")
    if abs(a_raw) > SPS_MAX_RAW_LAT_ACCEL:
      return self._pause_straight_path_stabilization(target, "gate_raw_accel")
    if abs(a_target) > SPS_MAX_TARGET_LAT_ACCEL:
      return self._pause_straight_path_stabilization(target, "gate_target_accel")
    if abs(a_raw - a_target) > SPS_MAX_RAW_TARGET_LAT_ACCEL_DELTA:
      return self._pause_straight_path_stabilization(target, "gate_raw_target_delta")
    if abs(raw_jerk) > SPS_ENTRY_MAX_RAW_LAT_JERK and not has_anchor:
      return self._pause_straight_path_stabilization(target, "gate_jerk")

    # Require sustained clean evidence before anchoring can affect output.
    self._sp_pause_frames = 0
    if len(self._sp_anchor_buffer) < SPS_ENTRY_MIN_FRAMES:
      self._sp_anchor_buffer.append(a_raw)
      return target, False, False, target, 0.0, "warming"

    # Rolling median anchor from historical raw lateral accel while eligible.
    anchor = float(np.median(self._sp_anchor_buffer)) if self._sp_anchor_buffer else a_raw
    if abs(a_raw - anchor) > SPS_MAX_SUPPRESSION_LAT_ACCEL:
      return self._release_straight_path_stabilization(target, "release_suppression")

    a_candidate = anchor + float(np.clip(
      a_raw - anchor, -SPS_ANCHOR_CLIP_LAT_ACCEL, SPS_ANCHOR_CLIP_LAT_ACCEL,
    ))
    candidate_curvature = a_candidate / speed_sq

    # Sign-flip guard vs raw (allowed only when raw lat accel is near zero).
    if (candidate_curvature * raw_base < 0.0 and
        min(abs(a_raw), abs(a_candidate)) >= SPS_SIGN_FLIP_EXEMPT_LAT_ACCEL):
      return self._release_straight_path_stabilization(target, "release_sign_flip")

    # Commit the bounded candidate, not raw wobble, so the anchor does not chase
    # the same near-zero sign flips it is suppressing.
    self._sp_anchor_buffer.append(a_candidate)

    applied = mode == "apply"
    output = candidate_curvature if applied else target
    return output, True, applied, candidate_curvature, anchor, "ok"

  def _release_straight_path_stabilization(
    self,
    target: float,
    reason: str,
  ) -> tuple[float, bool, bool, float, float, str]:
    self._reset_straight_path_stabilization_state()
    return target, False, False, 0.0, 0.0, reason

  def _pause_straight_path_stabilization(
    self,
    target: float,
    reason: str,
  ) -> tuple[float, bool, bool, float, float, str]:
    self._sp_pause_frames += 1
    if self._sp_pause_frames > SPS_MAX_PAUSE_FRAMES:
      return self._release_straight_path_stabilization(target, "pause_timeout")
    return target, False, False, 0.0, 0.0, reason

  # ---------------------------------------------------------------------------
  # Helpers used by tests / callers (stable signatures)
  # ---------------------------------------------------------------------------
  def _apply_demand_jerk_smoothing(
    self,
    inputs: ModelPathProcessorInputs,
    raw_base: float,
    target: float,
    quality: float,
    reason: str,
    path_disagreement: float | None,
  ) -> tuple[float, bool, float, float]:
    if not self._demand_jerk_smoothing_eligible(inputs, raw_base, target, quality, reason, path_disagreement):
      self._reset_demand_jerk_smoothing()
      return float(target), False, 0.0, 0.0

    v_ego = max(float(inputs.v_ego), 1.0)
    max_lat_jerk = float(np.interp(v_ego, DEMAND_JERK_SMOOTH_SPEED_BP, DEMAND_JERK_SMOOTH_MAX_LAT_JERK))
    max_step = max_lat_jerk * DT_CTRL / (v_ego * v_ego)
    lag_limit = DEMAND_JERK_SMOOTH_LAG_LAT_ACCEL / (v_ego * v_ego)

    max_abs = max(abs(raw_base), abs(target))
    near_straight = (max_abs <= DEMAND_JERK_SMOOTH_MAX_CURVATURE and
                     max_abs * v_ego * v_ego <= DEMAND_JERK_SMOOTH_MAX_LAT_ACCEL)
    curve_exit = (not near_straight and
                  self._curve_exit_smoothing_eligible(raw_base, target, v_ego))

    if curve_exit:
      effective_target = float(np.clip(target, raw_base - lag_limit, raw_base + lag_limit))
    else:
      effective_target = float(target)

    if self._demand_jerk_smoothed_curvature is None or not math.isfinite(self._demand_jerk_smoothed_curvature):
      self._demand_jerk_smoothed_curvature = effective_target
      active = abs(effective_target - float(target)) > 1e-9
      self._demand_jerk_smoothing_active = active
      return effective_target, active, max_step, abs(effective_target - float(target))

    prev = float(self._demand_jerk_smoothed_curvature)
    delta = effective_target - prev
    if abs(delta) <= max_step:
      candidate = effective_target
    else:
      candidate = prev + math.copysign(max_step, delta)

    candidate = self._clamp_demand_jerk_candidate(
      prev, candidate, effective_target, float(raw_base), lag_limit,
      raw_cap_active=curve_exit,
    )

    lag = candidate - float(target)
    self._demand_jerk_smoothed_curvature = candidate
    active = abs(lag) > 1e-9 or (curve_exit and abs(candidate - float(raw_base)) > 1e-9)
    self._demand_jerk_smoothing_active = active
    self._last_demand_jerk_smoothing_step = max_step
    self._last_demand_jerk_smoothing_lag = abs(lag)
    return candidate, active, max_step, abs(lag)

  def _demand_jerk_smoothing_eligible(
    self,
    inputs: ModelPathProcessorInputs,
    raw_base: float,
    target: float,
    quality: float,
    reason: str,
    path_disagreement: float | None,
  ) -> bool:
    gates_ok = self._demand_jerk_smoothing_gates_ok(
      inputs, raw_base, target, quality, reason, path_disagreement
    )
    if not gates_ok:
      self._reset_curve_exit_history()
      return False

    v_ego = float(inputs.v_ego)
    self._update_curve_exit_fall_history(raw_base, v_ego)

    max_abs = max(abs(raw_base), abs(target))
    near_straight = max_abs <= DEMAND_JERK_SMOOTH_MAX_CURVATURE and max_abs * v_ego * v_ego <= DEMAND_JERK_SMOOTH_MAX_LAT_ACCEL
    if near_straight:
      return True
    if reason == "low_lane_confidence":
      return False
    return self._curve_exit_smoothing_eligible(raw_base, target, v_ego)

  def _demand_jerk_smoothing_gates_ok(
    self,
    inputs: ModelPathProcessorInputs,
    raw_base: float,
    target: float,
    quality: float,
    reason: str,
    path_disagreement: float | None,
  ) -> bool:
    if not inputs.demand_jerk_smoothing_enabled or not inputs.demand_jerk_smoothing_allowed:
      return False
    if not inputs.smooth_model_path_curvature or inputs.lane_change_active:
      return False
    low_lane_confidence = reason == "low_lane_confidence"
    if reason == "ok":
      if quality < DEMAND_JERK_SMOOTH_MIN_QUALITY or not self._central_lane_confidence_ok(inputs.lane_line_probs):
        return False
    elif low_lane_confidence:
      if quality < DEMAND_JERK_SMOOTH_LOW_LANE_CONFIDENCE_MIN_QUALITY:
        return False
    else:
      return False
    v_ego = float(inputs.v_ego)
    if not math.isfinite(v_ego) or v_ego < DEMAND_JERK_SMOOTH_MIN_SPEED or v_ego > DEMAND_JERK_SMOOTH_MAX_SPEED:
      return False
    if not math.isfinite(raw_base) or not math.isfinite(target):
      return False
    if inputs.turn_curvature_sign != 0:
      return False
    if math.isfinite(inputs.frame_drop_perc) and inputs.frame_drop_perc > DEMAND_JERK_SMOOTH_MAX_FRAME_DROP_PERC:
      return False
    if path_disagreement is not None and path_disagreement > DEMAND_JERK_SMOOTH_MAX_PATH_DISAGREEMENT:
      return False
    if not self._path_y_std_ok(inputs.position_y_std):
      return False
    return True

  @staticmethod
  def _clamp_demand_jerk_candidate(
    prev: float,
    candidate: float,
    target: float,
    raw_base: float,
    lag_limit: float,
    raw_cap_active: bool,
  ) -> float:
    """Clamp jerk-limited candidate to stay within lag_limit of target, preserving
    monotonic movement toward target. When raw_cap_active is True (curve-exit guard),
    the candidate is also clamped to stay within lag_limit of raw_base; if that raw
    cap conflicts with the target/monotonic bounds, the raw cap wins so tracking
    accuracy is not sacrificed.
    """
    mono_lo = min(prev, target)
    mono_hi = max(prev, target)
    target_lo = target - lag_limit
    target_hi = target + lag_limit

    lo = max(mono_lo, target_lo)
    hi = min(mono_hi, target_hi)

    if raw_cap_active:
      raw_lo = raw_base - lag_limit
      raw_hi = raw_base + lag_limit
      lo2 = max(lo, raw_lo)
      hi2 = min(hi, raw_hi)
      if lo2 <= hi2:
        lo, hi = lo2, hi2
      else:
        # Raw cap and target/monotonic bounds conflict: raw reference cap wins.
        lo, hi = raw_lo, raw_hi

    if candidate < lo:
      return lo
    if candidate > hi:
      return hi
    return candidate

  def _curve_exit_smoothing_eligible(self, raw_base: float, target: float, v_ego: float) -> bool:
    """Narrow curve-exit eligibility: the current raw/reference curvature magnitude has
    already fallen below the stale damped target for a sustained number of frames.  The
    unwind must stay same-sign (or the current raw reference lateral accel is already
    near zero), and the peak stale demand must sit inside a conservative lateral-accel
    cap.
    """
    if not math.isfinite(raw_base) or not math.isfinite(target) or not math.isfinite(v_ego):
      return False
    if v_ego < 1.0:
      return False
    # Curve exit: raw/reference has fallen, damped target is the stale high value.
    if abs(raw_base) >= abs(target):
      return False
    curr_lat_accel = abs(raw_base) * v_ego * v_ego
    near_zero = curr_lat_accel <= DEMAND_JERK_SMOOTH_CURVE_EXIT_NEAR_ZERO_LAT_ACCEL
    if raw_base * target < 0.0 and not near_zero:
      # Current reference has crossed into a meaningful opposite-sign curve: do not smooth.
      return False
    if max(abs(raw_base), abs(target)) * v_ego * v_ego > DEMAND_JERK_SMOOTH_CURVE_EXIT_MAX_LAT_ACCEL:
      return False
    if self._curve_exit_fall_frames < DEMAND_JERK_SMOOTH_CURVE_EXIT_FALL_MIN_FRAMES:
      return False
    return True

  def _reset_demand_jerk_smoothing(self) -> None:
    self._demand_jerk_smoothed_curvature = None
    self._demand_jerk_smoothing_active = False
    self._last_demand_jerk_smoothing_step = 0.0
    self._last_demand_jerk_smoothing_lag = 0.0

  def _reset_curve_exit_history(self) -> None:
    self._curve_exit_prev_raw_base = None
    self._curve_exit_fall_frames = 0

  def _update_curve_exit_fall_history(self, raw_base: float, v_ego: float) -> None:
    """Track sustained raw/reference curvature magnitude falls.  A frame counts as a
    fall only if the current raw magnitude dropped by at least the configured lateral
    accel delta versus the previous frame while staying same-sign (or the current raw
    lateral accel is already near zero).
    """
    speed_sq = max(v_ego, 1.0) ** 2
    prev_raw = self._curve_exit_prev_raw_base
    if prev_raw is None or not math.isfinite(prev_raw):
      self._curve_exit_fall_frames = 0
    else:
      prev_lat_accel = abs(prev_raw) * speed_sq
      curr_lat_accel = abs(raw_base) * speed_sq
      same_sign = raw_base * prev_raw >= 0.0
      near_zero = curr_lat_accel <= DEMAND_JERK_SMOOTH_CURVE_EXIT_NEAR_ZERO_LAT_ACCEL
      if (same_sign or near_zero) and curr_lat_accel + DEMAND_JERK_SMOOTH_CURVE_EXIT_FALL_MIN_LAT_ACCEL_DELTA <= prev_lat_accel:
        self._curve_exit_fall_frames = min(self._curve_exit_fall_frames + 1, DEMAND_JERK_SMOOTH_CURVE_EXIT_FALL_MIN_FRAMES)
      else:
        self._curve_exit_fall_frames = 0
    self._curve_exit_prev_raw_base = float(raw_base)

  def _reset_temporal_curvature_smoothing(self) -> None:
    self._temporal_smoothed_curvature = None
    self._straight_road_damping_active = False

  @staticmethod
  def _soft_temporal_boundary(quality: float, reason: str) -> bool:
    return reason in SOFT_GATE_REASONS and quality <= LOW_QUALITY_BLEND_THRESHOLD

  def _apply_temporal_soft_boundary(self, active: bool) -> None:
    if active or active != self._prev_temporal_soft_boundary:
      self._reset_temporal_curvature_smoothing()
      self._reset_demand_jerk_smoothing()
    self._prev_temporal_soft_boundary = active

  @staticmethod
  def _central_lane_confidence_ok(lane_line_probs: Sequence[float]) -> bool:
    probs = list(lane_line_probs)
    if len(probs) >= 4:
      return min(float(probs[1]), float(probs[2])) >= DEMAND_JERK_SMOOTH_MIN_LANE_PROB
    if len(probs) >= 2:
      return min(float(v) for v in probs) >= DEMAND_JERK_SMOOTH_MIN_LANE_PROB
    return False

  @staticmethod
  def _path_y_std_ok(position_y_std: Sequence[float]) -> bool:
    values = list(position_y_std[:PATH_VALID_MIN_LEN])
    if len(values) < PATH_VALID_MIN_LEN:
      return False
    return max(float(v) for v in values) <= DEMAND_JERK_SMOOTH_MAX_PATH_Y_STD

  def _temporal_damp_curvature(
    self,
    inputs: ModelPathProcessorInputs,
    target: float,
    smoothing_enabled: bool,
  ) -> tuple[float, float, float, bool]:
    """Returns (tau_s, alpha, damped_curvature, straight_road_damping_active)."""
    if not smoothing_enabled:
      self._temporal_smoothed_curvature = None
      self._straight_road_damping_active = False
      return 0.0, 0.0, float(target), False

    v_ego = float(inputs.v_ego)
    if not math.isfinite(v_ego) or v_ego < SMOOTHED_CURVATURE_MIN_SPEED:
      self._temporal_smoothed_curvature = None
      self._straight_road_damping_active = False
      return 0.0, 0.0, float(target), False

    tau_s = float(np.interp(v_ego, DAMPING_TAU_SPEED_BP, DAMPING_TAU_S))
    tau_s = max(tau_s, 1e-4)

    # Straight-road damping: increase tau and apply deadband when near-straight.
    # Gated by lateral acceleration so it is not over-applied at high speed.
    straight_road_active = False
    if v_ego >= STRAIGHT_ROAD_DAMPING_MIN_SPEED and not inputs.lane_change_active:
      speed_factor = float(np.interp(
        v_ego,
        [STRAIGHT_ROAD_DAMPING_MIN_SPEED, STRAIGHT_ROAD_DAMPING_FULL_SPEED],
        [0.0, 1.0],
      ))
      if speed_factor > 0.0:
        lat_accel = abs(target) * v_ego * v_ego
        if lat_accel < STRAIGHT_ROAD_DAMPING_MAX_LAT_ACCEL:
          blend = float(np.interp(
            lat_accel, STRAIGHT_ROAD_DAMPING_BLEND_BP_LAT_ACCEL, STRAIGHT_ROAD_DAMPING_BLEND_SCALE,
          ))
          blend *= speed_factor
          if blend > 0.0:
            # Blend between base tau and straight-road tau
            tau_s = tau_s * (1.0 - blend) + STRAIGHT_ROAD_DAMPING_TAU_S * blend
            straight_road_active = True

            # Apply deadband: if curvature change in lateral accel is within deadband, hold previous value
            if self._temporal_smoothed_curvature is not None and math.isfinite(self._temporal_smoothed_curvature):
              delta_lat_accel = abs(target - self._temporal_smoothed_curvature) * v_ego * v_ego
              if delta_lat_accel < STRAIGHT_ROAD_DAMPING_DEADBAND_LAT_ACCEL * blend:
                # Within deadband — hold previous smoothed value
                self._straight_road_damping_active = True
                return tau_s, float(DT_CTRL / (DT_CTRL + tau_s)), float(self._temporal_smoothed_curvature), True

    alpha = float(DT_CTRL / (DT_CTRL + tau_s))

    if self._temporal_smoothed_curvature is None or not math.isfinite(self._temporal_smoothed_curvature):
      self._temporal_smoothed_curvature = float(target)
    else:
      self._temporal_smoothed_curvature = float(
        self._temporal_smoothed_curvature + alpha * (target - self._temporal_smoothed_curvature)
      )

    self._straight_road_damping_active = straight_road_active
    return tau_s, alpha, float(self._temporal_smoothed_curvature), straight_road_active

  def _apply_soft_gate_hold(self, quality: float, reason: str, v_ego: float) -> tuple[float, str, int]:
    if reason in SOFT_GATE_REASONS and quality < LOW_QUALITY_BLEND_THRESHOLD:
      self._hold_frames_remaining = self._soft_gate_hold_frames(v_ego, quality)
      self._hold_reason = reason
      self._hold_quality = self._soft_gate_hold_quality(v_ego, quality)
      return quality, reason, self._hold_frames_remaining

    if self._hold_frames_remaining > 0:
      self._hold_frames_remaining -= 1
      return min(quality, self._hold_quality), self._hold_reason, self._hold_frames_remaining

    self._hold_reason = "ok"
    self._hold_quality = SOFT_GATE_HOLD_QUALITY
    return quality, reason, 0

  @staticmethod
  def _soft_gate_hold_frames(v_ego: float, quality: float) -> int:
    if not math.isfinite(v_ego) or v_ego >= LOW_SPEED_SOFT_GATE_SPEED:
      return SOFT_GATE_HOLD_FRAMES

    extra_frames = int(round(float(np.interp(
      quality,
      [0.0, LOW_QUALITY_BLEND_THRESHOLD],
      [LOW_SPEED_SOFT_GATE_MAX_EXTRA_FRAMES, 1.0],
    ))))
    return SOFT_GATE_HOLD_FRAMES + max(1, extra_frames)

  @staticmethod
  def _soft_gate_hold_quality(v_ego: float, quality: float) -> float:
    if not math.isfinite(v_ego) or v_ego >= LOW_SPEED_SOFT_GATE_SPEED:
      return SOFT_GATE_HOLD_QUALITY
    return min(SOFT_GATE_HOLD_QUALITY, quality)

  @staticmethod
  def _limit_low_speed_untrusted_curvature_step(v_ego: float, desired_curvature: float, fallback_curvature: float) -> float:
    if not math.isfinite(v_ego) or v_ego >= LOW_SPEED_SOFT_GATE_SPEED:
      return desired_curvature
    if not math.isfinite(desired_curvature) or not math.isfinite(fallback_curvature):
      return desired_curvature

    curvature_delta = desired_curvature - fallback_curvature
    if abs(curvature_delta) <= LOW_SPEED_UNTRUSTED_CURVATURE_STEP:
      return desired_curvature
    return fallback_curvature + math.copysign(LOW_SPEED_UNTRUSTED_CURVATURE_STEP, curvature_delta)

  @staticmethod
  def _limit_same_sign_amplification(
    raw_curvature: float,
    processed_curvature: float,
    v_ego: float,
    max_lat_accel_delta: float,
  ) -> float:
    """Clamp same-sign processed curvature so it does not amplify the raw demand beyond a small lateral-accel margin."""
    if not math.isfinite(raw_curvature) or not math.isfinite(processed_curvature) or not math.isfinite(v_ego):
      return processed_curvature
    if raw_curvature * processed_curvature < 0.0:
      return processed_curvature
    max_delta_curvature = max_lat_accel_delta / max(abs(v_ego), 1.0) ** 2
    max_magnitude = abs(raw_curvature) + max_delta_curvature
    if abs(processed_curvature) <= max_magnitude:
      return processed_curvature
    return math.copysign(max_magnitude, processed_curvature)

  def _age_retained_curve(self) -> None:
    if self._retained_curve_frames <= 0:
      self._retained_curve_curvature = None
      self._retained_curve_frames = 0
      return
    self._retained_curve_frames -= 1
    if self._retained_curve_frames <= 0:
      self._retained_curve_curvature = None

  def _clear_retained_curve(self) -> None:
    self._retained_curve_curvature = None
    self._retained_curve_frames = 0

  def _refresh_retained_curve(
    self,
    inputs: ModelPathProcessorInputs,
    desired_curvature: float,
    quality: float,
    reason: str,
    path_disagreement: float | None,
  ) -> None:
    if not self._low_speed_curve_retention_active(inputs.v_ego):
      self._clear_retained_curve()
      return
    if reason not in ("ok", "low_lane_confidence") or quality < LOW_QUALITY_BLEND_THRESHOLD:
      return
    if path_disagreement is not None and path_disagreement > TURN_INTENT_MAX_PATH_CURVATURE_DISAGREEMENT:
      return
    if not self._curvature_is_plausible_for_retention(desired_curvature):
      return
    if not self._curvatures_compatible(desired_curvature, inputs.measured_curvature):
      return
    if not self._curvatures_close_for_retention(inputs.v_ego, desired_curvature, inputs.measured_curvature):
      return

    self._retained_curve_curvature = float(desired_curvature)
    self._retained_curve_frames = LOW_SPEED_CURVE_RETENTION_FRAMES

  def _retained_curve_fallback(
    self,
    inputs: ModelPathProcessorInputs,
    desired_curvature: float,
    fallback_curvature: float,
  ) -> float | None:
    retained_curvature = self._retained_curve_curvature
    if retained_curvature is None or self._retained_curve_frames <= 0:
      return None
    if not self._low_speed_curve_retention_active(inputs.v_ego):
      return None
    if not self._curvature_is_plausible_for_retention(retained_curvature):
      return None
    if not self._curvatures_compatible(retained_curvature, desired_curvature):
      return None
    if not self._curvatures_compatible(retained_curvature, inputs.measured_curvature):
      return None
    if not self._curvatures_compatible(retained_curvature, fallback_curvature):
      return None
    if not self._curvatures_close_for_retention(inputs.v_ego, retained_curvature, desired_curvature):
      return None
    if not self._curvatures_close_for_retention(inputs.v_ego, retained_curvature, inputs.measured_curvature):
      return None
    if not self._curvatures_close_for_retention(inputs.v_ego, retained_curvature, fallback_curvature):
      return None
    return float(retained_curvature)

  @staticmethod
  def _low_speed_curve_retention_active(v_ego: float) -> bool:
    return math.isfinite(v_ego) and v_ego < LOW_SPEED_SOFT_GATE_SPEED

  @classmethod
  def _low_speed_measured_turn_confirms_curvature(
    cls,
    inputs: ModelPathProcessorInputs,
    desired_curvature: float,
    path_disagreement: float | None,
  ) -> bool:
    if not cls._low_speed_curve_retention_active(inputs.v_ego):
      return False
    if not math.isfinite(desired_curvature) or not math.isfinite(inputs.measured_curvature):
      return False
    if desired_curvature * inputs.measured_curvature <= 0.0:
      return False
    if min(abs(desired_curvature), abs(inputs.measured_curvature)) < TURN_INTENT_MIN_CURVATURE:
      return False
    if inputs.turn_curvature_sign != 0 and desired_curvature * inputs.turn_curvature_sign <= 0.0:
      return False
    if path_disagreement is not None and path_disagreement > TURN_INTENT_MAX_PATH_CURVATURE_DISAGREEMENT:
      return False

    speed_sq = max(inputs.v_ego, 1.0) ** 2
    desired_lat_accel = abs(desired_curvature) * speed_sq
    measured_lat_accel = abs(inputs.measured_curvature) * speed_sq
    if min(desired_lat_accel, measured_lat_accel) < LOW_SPEED_CONFIRMED_TURN_MIN_LAT_ACCEL:
      return False
    return abs(desired_curvature - inputs.measured_curvature) * speed_sq <= LOW_SPEED_CONFIRMED_TURN_MAX_LAT_ACCEL_DELTA

  @staticmethod
  def _curvature_is_plausible_for_retention(curvature: float) -> bool:
    return math.isfinite(curvature) and abs(curvature) >= LOW_SPEED_CURVE_RETENTION_MIN_CURVATURE

  @staticmethod
  def _curvatures_compatible(reference_curvature: float, candidate_curvature: float) -> bool:
    if not math.isfinite(candidate_curvature):
      return False
    if abs(candidate_curvature) < LOW_SPEED_CURVE_RETENTION_MIN_CURVATURE:
      return True
    return reference_curvature * candidate_curvature > 0.0

  @staticmethod
  def _curvatures_close_for_retention(v_ego: float, reference_curvature: float, candidate_curvature: float) -> bool:
    if not math.isfinite(v_ego) or not math.isfinite(reference_curvature) or not math.isfinite(candidate_curvature):
      return False
    lateral_accel_delta = abs(reference_curvature - candidate_curvature) * max(v_ego, 1.0) ** 2
    return lateral_accel_delta <= MAX_LAT_ACCEL_JUMP

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

  def _lane_quality(self, lane_line_probs: Sequence[float], v_ego: float) -> float:
    if len(lane_line_probs) <= 2:
      self._low_lane_confidence_frames = 0
      return 0.9
    central_prob = min(float(lane_line_probs[1]), float(lane_line_probs[2]))
    if not math.isfinite(central_prob):
      self._low_lane_confidence_frames = 0
      return 0.85
    if central_prob >= LOW_LANE_LINE_PROB:
      self._low_lane_confidence_frames = 0
      return 1.0
    self._low_lane_confidence_frames += 1
    quality = float(np.interp(central_prob, [0.0, LOW_LANE_LINE_PROB], [0.85, 1.0]))
    low_speed = math.isfinite(v_ego) and v_ego < LOW_SPEED_SOFT_GATE_SPEED
    if low_speed and self._low_lane_confidence_frames >= LOW_LANE_CONFIDENCE_SUSTAIN_FRAMES:
      quality = min(quality, LOW_LANE_CONFIDENCE_SUSTAINED_QUALITY)
    return quality

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
    trust_penalty: float,
  ) -> float | None:
    if not inputs.smooth_model_path_curvature:
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
    psi_dot_at_action = float(np.polyval(np.polyder(coefficients), PATH_CURVATURE_ACTION_T))
    candidate_curvature = psi_dot_at_action / v_ego
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
    trust_scale = max(0.0, 1.0 - float(trust_penalty))
    blend_alpha *= trust_scale

    near_zero_scale = float(np.interp(abs(desired_curvature), NEAR_ZERO_CURVATURE_BP, NEAR_ZERO_BLEND_SCALE))
    blend_alpha *= near_zero_scale

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
