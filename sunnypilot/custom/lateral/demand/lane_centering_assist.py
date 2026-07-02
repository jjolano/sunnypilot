import math
from dataclasses import dataclass, field, replace
from typing import Any, Sequence

import numpy as np

from openpilot.sunnypilot.custom.lateral.demand.types import DEMAND_SOURCE_MODEL_PATH
from openpilot.sunnypilot.custom.lateral.demand.lane_geometry import (
  LaneGeometryResult,
  evaluate_lane_geometry,
)


LANE_CENTERING_ASSIST_MIN_SPEED = 5.0
LANE_CENTERING_ASSIST_MIN_PATH_QUALITY = 0.85
LANE_CENTERING_ASSIST_OK_REASON = "ok"
LANE_CENTERING_ASSIST_PATH_REASON_COOLDOWN_FRAMES = 50  # 0.5 s at 100 Hz
LANE_CENTERING_ASSIST_PATH_REASON_COOLDOWN_REASON = "path_reason_cooldown"
LANE_CENTERING_ASSIST_NEAR_LOOKAHEAD_T = 0.35
LANE_CENTERING_ASSIST_PREVIEW_T = 1.20
LANE_CENTERING_ASSIST_NEAR_LOOKAHEAD_MIN_M = 3.0
LANE_CENTERING_ASSIST_PREVIEW_MIN_M = 8.0
LANE_CENTERING_ASSIST_LATERAL_DEADBAND = 0.03
LANE_CENTERING_ASSIST_GROWTH_DEADBAND = 0.015
LANE_CENTERING_ASSIST_GEOMETRY_LATERAL_DEADBAND = 0.12
LANE_CENTERING_ASSIST_GEOMETRY_GROWTH_DEADBAND = 0.06
LANE_CENTERING_ASSIST_GEOMETRY_STRAIGHT_LATERAL_DEADBAND = 0.18
LANE_CENTERING_ASSIST_GEOMETRY_STRAIGHT_GROWTH_DEADBAND = 0.10
LANE_CENTERING_ASSIST_GEOMETRY_PERSISTENCE_FRAMES = 20  # 0.2 s at 100 Hz
LANE_CENTERING_ASSIST_SIGN_HYSTERESIS_NUDGE = 1e-6
LANE_CENTERING_ASSIST_MAX_LAT_ACCEL = 0.08
LANE_CENTERING_ASSIST_MAX_LAT_ACCEL_BP = [10.0, 20.0, 30.0]
# Route 00000246: at 30+ m/s the nudge saturated the old 0.025 cap in 51% of frames while the
# car held a ~0.2 m offset the driver kept correcting; keep enough authority to out-pull the
# measured ~0.02-0.03 m/s^2 steady execution deficit with margin.
LANE_CENTERING_ASSIST_MAX_LAT_ACCEL_V = [0.12, 0.08, 0.05]
LANE_CENTERING_ASSIST_SPEED_FLOOR = 5.0
LANE_CENTERING_ASSIST_LATERAL_GAIN = 0.00045
LANE_CENTERING_ASSIST_HEADING_GAIN = 0.0020
LANE_CENTERING_ASSIST_GROWTH_GAIN = 0.00070
LANE_CENTERING_ASSIST_BUILD_RATE = 0.00030
LANE_CENTERING_ASSIST_RELEASE_RATE = 0.00060
LANE_CENTERING_ASSIST_STRAIGHT_MIN_SPEED = 24.0
LANE_CENTERING_ASSIST_STRAIGHT_CURVATURE_MAX = 2.5e-4
LANE_CENTERING_ASSIST_STRAIGHT_LATERAL_DEADBAND = 0.06
LANE_CENTERING_ASSIST_STRAIGHT_GROWTH_DEADBAND = 0.035
LANE_CENTERING_ASSIST_STRAIGHT_BUILD_RATE = 0.00008

# Center-chase relaxation: temporary soft deadband around lane-centering error to avoid
# rapid exact-center chasing during fast lateral reversals. Triggered only under strict
# preconditions and never adds steering bias or freezes a nonzero nudge.
LANE_CENTERING_RELAX_MIN_SPEED = 8.0
LANE_CENTERING_RELAX_MIN_PATH_QUALITY = 0.90
LANE_CENTERING_RELAX_MIN_CONFIDENCE = 0.90
LANE_CENTERING_RELAX_ENVELOPE_MIN = 0.08
LANE_CENTERING_RELAX_ENVELOPE_MAX = 0.12
LANE_CENTERING_RELAX_NEAR_CENTER = 0.18
LANE_CENTERING_RELAX_TRIGGER_PREDICTED = 0.20
LANE_CENTERING_RELAX_ABORT_LATERAL = 0.30
LANE_CENTERING_RELAX_ABORT_PREDICTED = 0.35
LANE_CENTERING_RELAX_ABORT_HEADING = 0.025
LANE_CENTERING_RELAX_MAX_CURVE_LAT_ACCEL = 0.70
LANE_CENTERING_RELAX_HOLD_TIME = 0.40
LANE_CENTERING_RELAX_DECAY_TIME = 1.80
LANE_CENTERING_RELAX_MAX_ACTIVE_TIME = 3.00
LANE_CENTERING_RELAX_COOLDOWN_TIME = 0.70
LANE_CENTERING_RELAX_FLIP_WINDOW_TIME = 1.00
LANE_CENTERING_RELAX_MIN_FLIPS = 2

# Relaxation reason bits (UInt8) for telemetry.
LANE_CENTERING_RELAX_REASON_NONE = 0
LANE_CENTERING_RELAX_REASON_DRIVER = 1
LANE_CENTERING_RELAX_REASON_LANE_CHANGE = 2
LANE_CENTERING_RELAX_REASON_QUALITY = 4
LANE_CENTERING_RELAX_REASON_CURVE = 8
LANE_CENTERING_RELAX_REASON_LARGE_ERROR = 16
LANE_CENTERING_RELAX_REASON_HEADING = 32
LANE_CENTERING_RELAX_REASON_LOW_SPEED = 64
LANE_CENTERING_RELAX_REASON_OTHER = 128


@dataclass(frozen=True)
class LaneCenteringAssistInputs:
  lat_active: bool
  v_ego: float
  measured_curvature: float
  model_curvature: float
  previous_processed_curvature: float
  path_quality: float
  path_reason: str
  lane_change_shaping_active: bool
  lane_change_blend: float
  curvature_limited: bool
  steering_pressed: bool
  left_blinker: bool
  right_blinker: bool
  position_x: Sequence[float]
  position_y: Sequence[float]
  orientation_z: Sequence[float]
  lane_line_probs: Sequence[float]
  demand_source: str = DEMAND_SOURCE_MODEL_PATH
  lane_lines: Sequence[Any] = ()
  lane_line_stds: Sequence[float] = ()


@dataclass(frozen=True)
class LaneCenteringAssistResult:
  active: bool
  curvature_nudge: float
  lateral_error: float
  heading_error: float
  predicted_lateral_error: float
  confidence: float
  reason: str
  debug: dict[str, float | str | bool] = field(default_factory=dict)
  relaxed_lateral_error: float = 0.0
  relaxed_predicted_error: float = 0.0
  relax_active: bool = False
  relax_reason_bits: int = 0
  relax_envelope: float = 0.0
  relax_age: float = 0.0
  relax_nudge_flip_score: float = 0.0
  relax_error_cross_score: float = 0.0


def inactive_lane_centering_assist_result(reason: str = "inactive") -> LaneCenteringAssistResult:
  return LaneCenteringAssistResult(False, 0.0, 0.0, 0.0, 0.0, 0.0, reason, _debug(reason=reason))


def _evaluate_geometry(
  inputs: LaneCenteringAssistInputs,
  lateral_error: float,
  predicted_lateral_error: float,
) -> LaneGeometryResult:
  """Compute inner-lane geometry, falling back to model path when unavailable."""
  near_x = min(
    max(inputs.v_ego * LANE_CENTERING_ASSIST_NEAR_LOOKAHEAD_T, LANE_CENTERING_ASSIST_NEAR_LOOKAHEAD_MIN_M),
    max(float(x) for x in inputs.position_x) if inputs.position_x else LANE_CENTERING_ASSIST_NEAR_LOOKAHEAD_MIN_M,
  )
  preview_x = min(
    max(inputs.v_ego * LANE_CENTERING_ASSIST_PREVIEW_T, LANE_CENTERING_ASSIST_PREVIEW_MIN_M),
    max(float(x) for x in inputs.position_x) if inputs.position_x else LANE_CENTERING_ASSIST_PREVIEW_MIN_M,
  )
  return evaluate_lane_geometry(
    lane_lines=inputs.lane_lines,
    lane_line_probs=inputs.lane_line_probs,
    lane_line_stds=inputs.lane_line_stds,
    position_x=inputs.position_x,
    position_y=inputs.position_y,
    near_x=near_x,
    preview_x=preview_x,
  )


class LaneCenteringAssistTracker:
  def __init__(self) -> None:
    self.reset()

  def reset(self) -> None:
    self._filtered_nudge = 0.0
    self._active_sign = 0
    self._reason_cooldown_ticks = 0
    self._geometry_persistence_ticks = 0
    self._geometry_active = False
    self._relax = _CenterChaseRelaxation()

  def update(self, inputs: LaneCenteringAssistInputs, dt: float) -> LaneCenteringAssistResult:
    dt = max(_finite_float(dt), 0.0)
    metrics = _lane_centering_metrics(inputs)
    if metrics is None:
      return self._hard_block("invalid_path")

    lateral_error, heading_error, predicted_lateral_error = metrics
    geometry = _evaluate_geometry(inputs, lateral_error, predicted_lateral_error)

    # Apply geometry-corrected errors only after a short persistence period and only
    # while the harder geometry gates are satisfied. Existing LCA gates still apply.
    gate_reason = _gate_reason(inputs)
    if geometry.valid and gate_reason is None:
      if self._geometry_active:
        self._geometry_persistence_ticks = LANE_CENTERING_ASSIST_GEOMETRY_PERSISTENCE_FRAMES
      else:
        self._geometry_persistence_ticks = min(
          self._geometry_persistence_ticks + 1,
          LANE_CENTERING_ASSIST_GEOMETRY_PERSISTENCE_FRAMES,
        )
      if self._geometry_persistence_ticks >= LANE_CENTERING_ASSIST_GEOMETRY_PERSISTENCE_FRAMES:
        self._geometry_active = True
    else:
      self._geometry_persistence_ticks = 0
      self._geometry_active = False

    geometry_mode = self._geometry_active
    if geometry_mode:
      lateral_error = geometry.lateral_error
      predicted_lateral_error = geometry.predicted_lateral_error
      # Geometry errors are lane-center-relative (`lane_center_y - model_y`). Do not
      # mix in raw model-path heading here; it can oppose or flip the lane-center
      # correction when the model path is biased off-center.
      heading_error = 0.0

    if inputs.path_reason != LANE_CENTERING_ASSIST_OK_REASON:
      self._reason_cooldown_ticks = LANE_CENTERING_ASSIST_PATH_REASON_COOLDOWN_FRAMES

    # Compute confidence and an unrelaxed raw nudge before any gating so the relaxation
    # tracker can monitor nudge sign flips even when the assist is temporarily gated.
    confidence = _confidence(inputs, geometry_mode, geometry.confidence if geometry_mode else 0.0)
    straight_cruise = _straight_cruise(inputs)
    max_nudge = _max_nudge_curvature(inputs.v_ego, straight_cruise)
    unrelaxed_raw_nudge = confidence * (
      LANE_CENTERING_ASSIST_LATERAL_GAIN * lateral_error +
      LANE_CENTERING_ASSIST_HEADING_GAIN * heading_error +
      LANE_CENTERING_ASSIST_GROWTH_GAIN * (predicted_lateral_error - lateral_error)
    )
    unrelaxed_raw_nudge = float(np.clip(unrelaxed_raw_nudge, -max_nudge, max_nudge))

    self._relax.update(
      lateral_error=lateral_error,
      predicted_lateral_error=predicted_lateral_error,
      heading_error=heading_error,
      inputs=inputs,
      confidence=confidence,
      current_raw_nudge_sign=_sign(unrelaxed_raw_nudge, LANE_CENTERING_ASSIST_SIGN_HYSTERESIS_NUDGE),
      dt=dt,
    )
    lateral_error_eff, predicted_lateral_error_eff = self._relax.effective_errors(lateral_error, predicted_lateral_error)

    if gate_reason is not None:
      return self._hard_block(gate_reason, lateral_error, heading_error, predicted_lateral_error,
                              geometry=geometry, geometry_mode=geometry_mode)

    if self._reason_cooldown_ticks > 0:
      self._reason_cooldown_ticks -= 1
      return self._release(LANE_CENTERING_ASSIST_PATH_REASON_COOLDOWN_REASON, dt, lateral_error, heading_error,
                           predicted_lateral_error, geometry=geometry, geometry_mode=geometry_mode)

    lateral_deadband = _lateral_deadband(straight_cruise, geometry_mode)
    growth_deadband = _growth_deadband(straight_cruise, geometry_mode)
    error_sign = _sign(predicted_lateral_error_eff, lateral_deadband)
    now_sign = _sign(lateral_error_eff, lateral_deadband)
    if error_sign == 0:
      error_sign = now_sign
    same_direction = error_sign != 0 and (now_sign == 0 or now_sign == error_sign)
    error_growth = abs(predicted_lateral_error_eff) - abs(lateral_error_eff)
    growing = same_direction and error_growth > growth_deadband
    # Geometry mode may also act on a persisted same-sign steady offset (near≈preview)
    # as long as the offset is outside the wider geometry leeway. This preserves the
    # model-path mode's growth-only behavior unchanged.
    steady_offset = (
      geometry_mode and
      error_sign != 0 and
      now_sign == error_sign and
      abs(lateral_error_eff) > lateral_deadband
    )
    if not growing and not steady_offset:
      return self._release("error_not_growing", dt, lateral_error, heading_error, predicted_lateral_error,
                           geometry=geometry, geometry_mode=geometry_mode)

    raw_nudge = confidence * (
      LANE_CENTERING_ASSIST_LATERAL_GAIN * lateral_error_eff +
      LANE_CENTERING_ASSIST_HEADING_GAIN * heading_error +
      LANE_CENTERING_ASSIST_GROWTH_GAIN * (predicted_lateral_error_eff - lateral_error_eff)
    )
    target_nudge = float(np.clip(raw_nudge, -max_nudge, max_nudge))
    target_sign = _sign(target_nudge, LANE_CENTERING_ASSIST_SIGN_HYSTERESIS_NUDGE)
    geometry_sign = _sign(lateral_error_eff, lateral_deadband) if geometry_mode else 0
    if geometry_sign != 0 and target_sign != 0 and target_sign != geometry_sign:
      return self._release("geometry_sign_veto", dt, lateral_error, heading_error, predicted_lateral_error,
                           geometry=geometry, geometry_mode=geometry_mode)

    current_sign = _sign(self._filtered_nudge, LANE_CENTERING_ASSIST_SIGN_HYSTERESIS_NUDGE)
    if current_sign != 0 and target_sign != 0 and current_sign != target_sign:
      nudge = _approach(self._filtered_nudge, 0.0, LANE_CENTERING_ASSIST_RELEASE_RATE * dt)
      self._filtered_nudge = nudge
      if abs(nudge) <= LANE_CENTERING_ASSIST_SIGN_HYSTERESIS_NUDGE:
        self._filtered_nudge = 0.0
        self._active_sign = 0
      return self._with_relax(LaneCenteringAssistResult(
        abs(self._filtered_nudge) > 0.0, self._filtered_nudge, lateral_error, heading_error, predicted_lateral_error,
        confidence, "sign_hysteresis", _debug(inputs, lateral_error, heading_error, predicted_lateral_error,
                                               confidence, raw_nudge, target_nudge, self._filtered_nudge,
                                               "sign_hysteresis", max_nudge, straight_cruise,
                                               geometry=geometry, geometry_mode=geometry_mode),
      ))

    build_rate = _build_rate(straight_cruise)
    nudge = _approach(self._filtered_nudge, target_nudge, build_rate * dt)
    self._filtered_nudge = nudge
    self._active_sign = _sign(nudge, LANE_CENTERING_ASSIST_SIGN_HYSTERESIS_NUDGE)
    active = abs(nudge) > 0.0
    reason = "growing_lateral_error" if active else "below_deadband"
    return self._with_relax(LaneCenteringAssistResult(
      active, nudge, lateral_error, heading_error, predicted_lateral_error, confidence, reason,
      _debug(inputs, lateral_error, heading_error, predicted_lateral_error, confidence, raw_nudge, target_nudge, nudge, reason,
             max_nudge, straight_cruise, geometry=geometry, geometry_mode=geometry_mode),
    ))

  def _with_relax(self, result: LaneCenteringAssistResult) -> LaneCenteringAssistResult:
    debug = dict(result.debug)
    debug.update({
      "lane_centering_relax_active": self._relax.active,
      "lane_centering_relax_reason_bits": self._relax.reason_bits,
      "lane_centering_relax_envelope": self._relax.envelope,
      "lane_centering_relax_lateral_error": self._relax.relaxed_lateral_error,
      "lane_centering_relax_predicted_error": self._relax.relaxed_predicted_error,
      "lane_centering_relax_age": self._relax.age,
      "lane_centering_relax_nudge_flip_score": self._relax.nudge_flip_score,
      "lane_centering_relax_error_cross_score": self._relax.error_cross_score,
    })
    return replace(
      result,
      relaxed_lateral_error=self._relax.relaxed_lateral_error,
      relaxed_predicted_error=self._relax.relaxed_predicted_error,
      relax_active=self._relax.active,
      relax_reason_bits=self._relax.reason_bits,
      relax_envelope=self._relax.envelope,
      relax_age=self._relax.age,
      relax_nudge_flip_score=self._relax.nudge_flip_score,
      relax_error_cross_score=self._relax.error_cross_score,
      debug=debug,
    )

  def _release(self, reason: str, dt: float, lateral_error: float = 0.0, heading_error: float = 0.0,
               predicted_lateral_error: float = 0.0,
               geometry: LaneGeometryResult | None = None, geometry_mode: bool = False) -> LaneCenteringAssistResult:
    self._filtered_nudge = _approach(self._filtered_nudge, 0.0, LANE_CENTERING_ASSIST_RELEASE_RATE * dt)
    if abs(self._filtered_nudge) <= LANE_CENTERING_ASSIST_SIGN_HYSTERESIS_NUDGE:
      self._filtered_nudge = 0.0
      self._active_sign = 0
    return self._with_relax(LaneCenteringAssistResult(
      abs(self._filtered_nudge) > 0.0, self._filtered_nudge, lateral_error, heading_error, predicted_lateral_error, 0.0, reason,
      _debug(reason=reason, lateral_error=lateral_error, heading_error=heading_error,
             predicted_lateral_error=predicted_lateral_error, filtered_nudge=self._filtered_nudge,
             geometry=geometry, geometry_mode=geometry_mode),
    ))

  def _hard_block(self, reason: str, lateral_error: float = 0.0, heading_error: float = 0.0,
                  predicted_lateral_error: float = 0.0,
                  geometry: LaneGeometryResult | None = None, geometry_mode: bool = False) -> LaneCenteringAssistResult:
    # Safety gates must clear any stale relaxation state so it cannot re-arm immediately
    # once the gate condition clears (e.g. invalid path, driver override, lane change).
    self._relax.safety_abort(lateral_error, predicted_lateral_error, _relax_reason_bits_for_gate(reason))
    self._filtered_nudge = 0.0
    self._active_sign = 0
    self._geometry_persistence_ticks = 0
    self._geometry_active = False
    return self._with_relax(LaneCenteringAssistResult(
      False, 0.0, lateral_error, heading_error, predicted_lateral_error, 0.0, reason,
      _debug(reason=reason, lateral_error=lateral_error, heading_error=heading_error,
             predicted_lateral_error=predicted_lateral_error, filtered_nudge=0.0,
             geometry=geometry, geometry_mode=geometry_mode),
    ))


class _CenterChaseRelaxation:
  """Bounded center-chase relaxation for fast lateral reversals.

  Maintains a temporary soft deadband around lane-centering error when repeated
  near-center sign flips are detected. The deadband is held, then decays, and has a
  cooldown before it can re-arm. It never freezes a nonzero nudge or adds steering bias.
  """

  def __init__(self) -> None:
    self.reset()

  def reset(self) -> None:
    self._state = "idle"
    self._timer = 0.0
    self._clk = 0.0
    self._active_age = 0.0
    self._envelope = 0.0
    self._decay_from = 0.0
    self._cross_history: list[float] = []
    self._nudge_history: list[float] = []
    self._last_error_sign = 0
    self._last_nudge_sign = 0
    self._reason_bits = 0
    self._flip_score = 0
    self._cross_score = 0
    self._relaxed_lateral_error = 0.0
    self._relaxed_predicted_error = 0.0

  @property
  def active(self) -> bool:
    return self._state == "active"

  @property
  def envelope(self) -> float:
    return self._envelope

  @property
  def age(self) -> float:
    return self._active_age if self._state == "active" else 0.0

  @property
  def reason_bits(self) -> int:
    return self._reason_bits

  @property
  def nudge_flip_score(self) -> float:
    return float(self._flip_score)

  @property
  def error_cross_score(self) -> float:
    return float(self._cross_score)

  @property
  def relaxed_lateral_error(self) -> float:
    return self._relaxed_lateral_error

  @property
  def relaxed_predicted_error(self) -> float:
    return self._relaxed_predicted_error

  def update(self, *, lateral_error: float, predicted_lateral_error: float, heading_error: float,
             inputs: LaneCenteringAssistInputs, confidence: float, current_raw_nudge_sign: int, dt: float) -> None:
    dt = max(_finite_float(dt), 0.0)
    self._clk += dt

    error_sign = _sign(lateral_error, LANE_CENTERING_ASSIST_LATERAL_DEADBAND)
    if self._last_error_sign != 0 and error_sign != 0 and error_sign != self._last_error_sign:
      self._cross_history.append(self._clk)
    if error_sign != 0:
      self._last_error_sign = error_sign

    if self._last_nudge_sign != 0 and current_raw_nudge_sign != 0 and current_raw_nudge_sign != self._last_nudge_sign:
      self._nudge_history.append(self._clk)
    if current_raw_nudge_sign != 0:
      self._last_nudge_sign = current_raw_nudge_sign

    self._prune_histories()

    self._flip_score = len(self._nudge_history)
    self._cross_score = len(self._cross_history)

    self._reason_bits = self._compute_abort_bits(inputs, lateral_error, predicted_lateral_error, heading_error, confidence)
    near_center = (abs(lateral_error) <= LANE_CENTERING_RELAX_NEAR_CENTER and
                   abs(predicted_lateral_error) <= LANE_CENTERING_RELAX_TRIGGER_PREDICTED)
    curve_lat_accel = _curve_lat_accel(inputs)
    curve_ok = curve_lat_accel <= LANE_CENTERING_RELAX_MAX_CURVE_LAT_ACCEL
    quality_ok = (float(inputs.path_quality) >= LANE_CENTERING_RELAX_MIN_PATH_QUALITY and
                  confidence >= LANE_CENTERING_RELAX_MIN_CONFIDENCE and
                  inputs.path_reason == LANE_CENTERING_ASSIST_OK_REASON)
    speed_ok = inputs.v_ego >= LANE_CENTERING_RELAX_MIN_SPEED
    conditions_ok = self._reason_bits == 0 and near_center and curve_ok and quality_ok and speed_ok
    can_trigger = self._flip_score >= LANE_CENTERING_RELAX_MIN_FLIPS and self._cross_score >= LANE_CENTERING_RELAX_MIN_FLIPS

    target_envelope = self._envelope_size(inputs.path_quality, confidence) if conditions_ok else 0.0

    # Safety abort: any guarded abort condition must kill the deadband immediately,
    # clear stale flip history, and enter cooldown. Do not pass through decay.
    if self._reason_bits != 0:
      self.safety_abort(lateral_error, predicted_lateral_error, self._reason_bits)
    elif self._state == "idle":
      self._timer += dt
      if can_trigger and conditions_ok:
        self._state = "active"
        self._timer = 0.0
        self._active_age = 0.0
        self._envelope = target_envelope

    elif self._state == "active":
      self._timer += dt
      self._active_age += dt
      if self._active_age >= LANE_CENTERING_RELAX_MAX_ACTIVE_TIME:
        self._state = "decay"
        self._timer = 0.0
        self._decay_from = self._envelope
      elif self._active_age >= LANE_CENTERING_RELAX_HOLD_TIME:
        # After the guaranteed hold, stay active only while the reversal pattern persists.
        if can_trigger and conditions_ok:
          self._envelope = target_envelope
        else:
          self._state = "decay"
          self._timer = 0.0
          self._decay_from = self._envelope
      else:
        self._envelope = target_envelope

    elif self._state == "decay":
      self._timer += dt
      self._envelope = max(0.0, self._decay_from * (1.0 - self._timer / LANE_CENTERING_RELAX_DECAY_TIME))
      if self._envelope <= 0.0 or self._timer >= LANE_CENTERING_RELAX_DECAY_TIME:
        self._state = "cooldown"
        self._timer = 0.0
        self._envelope = 0.0

    elif self._state == "cooldown":
      self._timer += dt
      self._envelope = 0.0
      if self._timer >= LANE_CENTERING_RELAX_COOLDOWN_TIME:
        self._state = "idle"
        self._timer = 0.0

    self._relaxed_lateral_error = _soft_deadband(lateral_error, self._envelope)
    self._relaxed_predicted_error = _soft_deadband(predicted_lateral_error, self._envelope)

  def _prune_histories(self) -> None:
    cutoff = self._clk - LANE_CENTERING_RELAX_FLIP_WINDOW_TIME
    self._cross_history = [t for t in self._cross_history if t >= cutoff]
    self._nudge_history = [t for t in self._nudge_history if t >= cutoff]

  def safety_abort(self, lateral_error: float = 0.0, predicted_lateral_error: float = 0.0,
                   reason_bits: int = LANE_CENTERING_RELAX_REASON_OTHER) -> None:
    """Hard abort: zero the envelope, clear flip history, and enter cooldown.

    Safety aborts must not decay; the deadband must stop immediately so stale
    relaxation cannot re-arm from old history on recovery. The effective errors
    are reset to the raw values so the deadband is not applied this frame.
    """
    self._state = "cooldown"
    self._timer = 0.0
    self._envelope = 0.0
    self._decay_from = 0.0
    self._active_age = 0.0
    self._cross_history.clear()
    self._nudge_history.clear()
    self._last_error_sign = 0
    self._last_nudge_sign = 0
    self._reason_bits = reason_bits
    self._flip_score = 0
    self._cross_score = 0
    self._relaxed_lateral_error = lateral_error
    self._relaxed_predicted_error = predicted_lateral_error

  def _compute_abort_bits(self, inputs: LaneCenteringAssistInputs, lateral_error: float,
                          predicted_lateral_error: float, heading_error: float, confidence: float) -> int:
    bits = LANE_CENTERING_RELAX_REASON_NONE
    if inputs.steering_pressed:
      bits |= LANE_CENTERING_RELAX_REASON_DRIVER
    lane_change_blend = _finite_optional_float(inputs.lane_change_blend)
    if inputs.left_blinker or inputs.right_blinker or inputs.lane_change_shaping_active or (lane_change_blend is not None and abs(lane_change_blend) > 1e-3):
      bits |= LANE_CENTERING_RELAX_REASON_LANE_CHANGE
    if inputs.curvature_limited:
      bits |= LANE_CENTERING_RELAX_REASON_CURVE
    if float(inputs.path_quality) < LANE_CENTERING_RELAX_MIN_PATH_QUALITY:
      bits |= LANE_CENTERING_RELAX_REASON_QUALITY
    if inputs.path_reason != LANE_CENTERING_ASSIST_OK_REASON:
      bits |= LANE_CENTERING_RELAX_REASON_QUALITY
    if confidence < LANE_CENTERING_RELAX_MIN_CONFIDENCE:
      bits |= LANE_CENTERING_RELAX_REASON_QUALITY
    if abs(lateral_error) > LANE_CENTERING_RELAX_ABORT_LATERAL:
      bits |= LANE_CENTERING_RELAX_REASON_LARGE_ERROR
    if abs(predicted_lateral_error) > LANE_CENTERING_RELAX_ABORT_PREDICTED:
      bits |= LANE_CENTERING_RELAX_REASON_LARGE_ERROR
    if abs(heading_error) > LANE_CENTERING_RELAX_ABORT_HEADING:
      bits |= LANE_CENTERING_RELAX_REASON_HEADING
    if inputs.v_ego < LANE_CENTERING_RELAX_MIN_SPEED:
      bits |= LANE_CENTERING_RELAX_REASON_LOW_SPEED
    curve_lat_accel = _curve_lat_accel(inputs)
    if curve_lat_accel > LANE_CENTERING_RELAX_MAX_CURVE_LAT_ACCEL:
      bits |= LANE_CENTERING_RELAX_REASON_CURVE
    return bits

  def _envelope_size(self, path_quality: float, confidence: float) -> float:
    pq = _finite_float(path_quality)
    cf = _finite_float(confidence)
    quality_factor = float(np.clip((1.0 - pq) / 0.1, 0.0, 1.0))
    confidence_factor = float(np.clip(max(0.0, LANE_CENTERING_RELAX_MIN_CONFIDENCE - cf) / 0.1, 0.0, 1.0))
    return LANE_CENTERING_RELAX_ENVELOPE_MIN + (LANE_CENTERING_RELAX_ENVELOPE_MAX - LANE_CENTERING_RELAX_ENVELOPE_MIN) * max(quality_factor, confidence_factor)

  def effective_errors(self, lateral_error: float, predicted_lateral_error: float) -> tuple[float, float]:
    return self._relaxed_lateral_error, self._relaxed_predicted_error


def _gate_reason(inputs: LaneCenteringAssistInputs) -> str | None:
  if not inputs.lat_active:
    return "inactive"
  if not _finite(inputs.v_ego, inputs.model_curvature, inputs.measured_curvature, inputs.previous_processed_curvature):
    return "nonfinite"
  if inputs.v_ego < LANE_CENTERING_ASSIST_MIN_SPEED:
    return "low_speed"
  if inputs.demand_source != DEMAND_SOURCE_MODEL_PATH:
    return "non_model_demand"
  if inputs.steering_pressed:
    return "driver_steering"
  lane_change_blend = _finite_optional_float(inputs.lane_change_blend)
  if inputs.left_blinker or inputs.right_blinker or inputs.lane_change_shaping_active or lane_change_blend is None or abs(lane_change_blend) > 1e-3:
    return "lane_change"
  if inputs.curvature_limited:
    return "curvature_limited"
  if inputs.path_reason != LANE_CENTERING_ASSIST_OK_REASON:
    return "path_reason"
  if not _finite(inputs.path_quality) or float(inputs.path_quality) < LANE_CENTERING_ASSIST_MIN_PATH_QUALITY:
    return "low_path_quality"
  return None


def _relax_reason_bits_for_gate(reason: str) -> int:
  if reason == "driver_steering":
    return LANE_CENTERING_RELAX_REASON_DRIVER
  if reason == "lane_change":
    return LANE_CENTERING_RELAX_REASON_LANE_CHANGE
  if reason == "curvature_limited":
    return LANE_CENTERING_RELAX_REASON_CURVE
  if reason in ("low_path_quality", "path_reason", "invalid_path"):
    return LANE_CENTERING_RELAX_REASON_QUALITY
  if reason == "low_speed":
    return LANE_CENTERING_RELAX_REASON_LOW_SPEED
  return LANE_CENTERING_RELAX_REASON_OTHER


def _lane_centering_metrics(inputs: LaneCenteringAssistInputs) -> tuple[float, float, float] | None:
  xs = _finite_array(inputs.position_x)
  ys = _finite_array(inputs.position_y)
  headings = _finite_array(inputs.orientation_z)
  if len(xs) < 2 or len(ys) != len(xs) or len(headings) != len(xs):
    return None
  if xs[-1] <= xs[0]:
    return None
  near_x = min(max(inputs.v_ego * LANE_CENTERING_ASSIST_NEAR_LOOKAHEAD_T, LANE_CENTERING_ASSIST_NEAR_LOOKAHEAD_MIN_M), xs[-1])
  preview_x = min(max(inputs.v_ego * LANE_CENTERING_ASSIST_PREVIEW_T, LANE_CENTERING_ASSIST_PREVIEW_MIN_M), xs[-1])
  lateral_error = float(np.interp(near_x, xs, ys))
  predicted_lateral_error = float(np.interp(preview_x, xs, ys))
  heading_error = float(np.interp(near_x, xs, headings))
  return lateral_error, heading_error, predicted_lateral_error


def _confidence(inputs: LaneCenteringAssistInputs, geometry_mode: bool = False,
                geometry_confidence: float = 0.0) -> float:
  if geometry_mode:
    # Geometry confidence already blends prob/std/width; still cap by path quality.
    path_confidence = float(np.clip((float(inputs.path_quality) - LANE_CENTERING_ASSIST_MIN_PATH_QUALITY) / 0.15, 0.0, 1.0))
    return min(path_confidence, float(np.clip(geometry_confidence, 0.0, 1.0)))
  lane_probs = [_finite_float(prob) for prob in inputs.lane_line_probs]
  lane_probs = [prob for prob in lane_probs if math.isfinite(prob)]
  lane_confidence = min(lane_probs[1], lane_probs[2]) if len(lane_probs) >= 3 else 0.0
  path_confidence = float(np.clip((float(inputs.path_quality) - LANE_CENTERING_ASSIST_MIN_PATH_QUALITY) / 0.15, 0.0, 1.0))
  lane_confidence = float(np.clip((lane_confidence - 0.5) / 0.4, 0.0, 1.0))
  return min(path_confidence, lane_confidence)


def _max_nudge_curvature(v_ego: float, straight_cruise: bool = False) -> float:
  speed = max(abs(_finite_float(v_ego)), LANE_CENTERING_ASSIST_SPEED_FLOOR)
  max_lat_accel = LANE_CENTERING_ASSIST_MAX_LAT_ACCEL
  if straight_cruise:
    max_lat_accel = min(
      max_lat_accel,
      float(np.interp(speed, LANE_CENTERING_ASSIST_MAX_LAT_ACCEL_BP, LANE_CENTERING_ASSIST_MAX_LAT_ACCEL_V)),
    )
  return max_lat_accel / speed**2


def _straight_cruise(inputs: LaneCenteringAssistInputs) -> bool:
  return bool(
    inputs.v_ego >= LANE_CENTERING_ASSIST_STRAIGHT_MIN_SPEED and
    max(abs(inputs.model_curvature), abs(inputs.measured_curvature), abs(inputs.previous_processed_curvature)) <=
    LANE_CENTERING_ASSIST_STRAIGHT_CURVATURE_MAX
  )


def _lateral_deadband(straight_cruise: bool, geometry_mode: bool = False) -> float:
  if geometry_mode:
    return LANE_CENTERING_ASSIST_GEOMETRY_STRAIGHT_LATERAL_DEADBAND if straight_cruise else LANE_CENTERING_ASSIST_GEOMETRY_LATERAL_DEADBAND
  return LANE_CENTERING_ASSIST_STRAIGHT_LATERAL_DEADBAND if straight_cruise else LANE_CENTERING_ASSIST_LATERAL_DEADBAND


def _growth_deadband(straight_cruise: bool, geometry_mode: bool = False) -> float:
  if geometry_mode:
    return LANE_CENTERING_ASSIST_GEOMETRY_STRAIGHT_GROWTH_DEADBAND if straight_cruise else LANE_CENTERING_ASSIST_GEOMETRY_GROWTH_DEADBAND
  return LANE_CENTERING_ASSIST_STRAIGHT_GROWTH_DEADBAND if straight_cruise else LANE_CENTERING_ASSIST_GROWTH_DEADBAND


def _build_rate(straight_cruise: bool) -> float:
  return LANE_CENTERING_ASSIST_STRAIGHT_BUILD_RATE if straight_cruise else LANE_CENTERING_ASSIST_BUILD_RATE


def _curve_lat_accel(inputs: LaneCenteringAssistInputs) -> float:
  curvatures = [inputs.model_curvature, inputs.measured_curvature, inputs.previous_processed_curvature]
  max_curvature = max(abs(_finite_float(c)) for c in curvatures)
  speed = max(abs(_finite_float(inputs.v_ego)), LANE_CENTERING_ASSIST_SPEED_FLOOR)
  return max_curvature * speed**2


def _soft_deadband(value: float, threshold: float) -> float:
  if threshold <= 0.0:
    return value
  if abs(value) <= threshold:
    return 0.0
  return math.copysign(abs(value) - threshold, value)


def _finite_array(values: Sequence[float]) -> list[float]:
  result = []
  for value in values:
    try:
      finite = float(value)
    except (TypeError, ValueError):
      return []
    if not math.isfinite(finite):
      return []
    result.append(finite)
  return result


def _finite_float(value) -> float:
  try:
    result = float(value)
  except (TypeError, ValueError):
    return 0.0
  return result if math.isfinite(result) else 0.0


def _finite_optional_float(value) -> float | None:
  try:
    result = float(value)
  except (TypeError, ValueError):
    return None
  return result if math.isfinite(result) else None


def _finite(*values: float) -> bool:
  try:
    return all(math.isfinite(float(value)) for value in values)
  except (TypeError, ValueError):
    return False


def _sign(value: float, threshold: float = 0.0) -> int:
  return 1 if value > threshold else (-1 if value < -threshold else 0)


def _approach(value: float, target: float, step: float) -> float:
  step = max(step, 0.0)
  if target > value:
    return min(target, value + step)
  return max(target, value - step)


def _debug(inputs: LaneCenteringAssistInputs | None = None, lateral_error: float = 0.0, heading_error: float = 0.0,
           predicted_lateral_error: float = 0.0, confidence: float = 0.0, raw_nudge: float = 0.0,
           target_nudge: float = 0.0, filtered_nudge: float = 0.0, reason: str = "inactive",
           max_nudge: float = 0.0, straight_cruise: bool = False,
           geometry: LaneGeometryResult | None = None, geometry_mode: bool = False) -> dict[str, float | str | bool]:
  debug: dict[str, float | str | bool] = {
    "lane_centering_assist_active": abs(filtered_nudge) > 0.0,
    "lane_centering_reason": reason,
    "lane_centering_lateral_error": lateral_error,
    "lane_centering_heading_error": heading_error,
    "lane_centering_predicted_error": predicted_lateral_error,
    "lane_centering_confidence": confidence,
    "lane_centering_raw_nudge": raw_nudge,
    "lane_centering_target_nudge": target_nudge,
    "lane_centering_curvature_nudge": filtered_nudge,
    "lane_centering_max_nudge": max_nudge,
    "lane_centering_straight_cruise": straight_cruise,
    "lane_centering_v_ego": float(inputs.v_ego) if inputs is not None else 0.0,
    "lane_centering_geometry_mode": geometry_mode,
    "lane_centering_geometry_source": geometry.source if geometry is not None else "model_path",
    "lane_centering_geometry_valid": geometry.valid if geometry is not None else False,
    "lane_centering_geometry_reason": geometry.reason if geometry is not None else "none",
    "lane_centering_geometry_confidence": geometry.confidence if geometry is not None else 0.0,
    "lane_centering_geometry_offset_near": geometry.offset_near if geometry is not None else 0.0,
    "lane_centering_geometry_offset_preview": geometry.offset_preview if geometry is not None else 0.0,
    "lane_centering_geometry_width_near": geometry.width_near if geometry is not None else 0.0,
    "lane_centering_geometry_width_preview": geometry.width_preview if geometry is not None else 0.0,
  }
  return debug
