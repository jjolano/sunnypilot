"""
Pure reusable lateral disturbance classifier for live torque learning and offline route analysis.

Phase 0a/0b design:
- Decisions are advisory for learner observability; no output restriction.
- Uncertain or missing optional context lowers confidence and records a reason,
  but does NOT by itself quarantine or reject.
- Hard rejects are reserved for driver/lane-change/control-limit events.
- Quarantine is reserved for measurement spikes, bump-like transients, and
  model-demand/path jitter/low-quality events.
- Cooldown/hysteresis keeps high-confidence decisions sticky for a short window.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntFlag, IntEnum
from typing import Any


class DisturbanceReason(IntFlag):
  NONE = 0
  CLEAN = 1 << 0
  MISSING_CONTEXT = 1 << 1
  DRIVER_OVERRIDE = 1 << 2
  LANE_CHANGE = 1 << 3
  CONTROL_LIMIT = 1 << 4
  MEASUREMENT_SPIKE = 1 << 5
  BUMP = 1 << 6
  MODEL_PATH_LOW_QUALITY = 1 << 7
  MODEL_DEMAND_JITTER = 1 << 8
  COOLDOWN_ACTIVE = 1 << 9
  LAT_INACTIVE = 1 << 10


class LearningDecision(IntEnum):
  ACCEPT = 0
  QUARANTINE = 1
  REJECT_SHADOW = 2


@dataclass(frozen=True)
class LateralSample:
  """Canonical sample used by the classifier. Unknown optional values should be None."""
  t: float
  v_ego: float
  lat_active: bool
  steering_pressed: bool = False
  blinker_active: bool = False
  lane_change_active: bool = False
  steering_rate_deg: float | None = None
  output: float | None = None
  unshaped_output: float | None = None
  applied_torque: float | None = None
  desired_lateral_accel: float | None = None
  actual_lateral_accel: float | None = None
  output_sat: bool = False
  steer_limited: bool = False
  model_path_quality: float | None = None
  model_path_gated: bool = False
  shaping_reason: int = 0
  governor_reason: int = 0

  @classmethod
  def from_torqued_inputs(
    cls,
    *,
    t: float,
    v_ego: float,
    lat_active: bool,
    steering_pressed: bool,
    lateral_acc: float,
    steer: float,
    steering_rate_deg: float | None = None,
    output: float | None = None,
    unshaped_output: float | None = None,
    applied_torque: float | None = None,
    desired_lateral_accel: float | None = None,
    blinker_active: bool = False,
    lane_change_active: bool = False,
    output_sat: bool = False,
    steer_limited: bool = False,
    model_path_quality: float | None = None,
    model_path_gated: bool = False,
    shaping_reason: int = 0,
    governor_reason: int = 0,
  ) -> "LateralSample":
    return cls(
      t=t,
      v_ego=v_ego,
      lat_active=lat_active,
      steering_pressed=steering_pressed,
      blinker_active=blinker_active,
      lane_change_active=lane_change_active,
      steering_rate_deg=steering_rate_deg,
      output=output,
      unshaped_output=unshaped_output,
      applied_torque=applied_torque,
      desired_lateral_accel=desired_lateral_accel,
      actual_lateral_accel=lateral_acc,
      output_sat=output_sat,
      steer_limited=steer_limited,
      model_path_quality=model_path_quality,
      model_path_gated=model_path_gated,
      shaping_reason=shaping_reason,
      governor_reason=governor_reason,
    )


@dataclass(frozen=True)
class Classification:
  decision: LearningDecision
  confidence: float
  reasons: DisturbanceReason
  cooldown_remaining: float
  details: dict[str, Any] = field(default_factory=dict)


class DisturbanceClassifier:
  def __init__(
    self,
    *,
    measurement_spike_lataccel: float = 2.0,
    measurement_spike_rate_deg: float = 120.0,
    bump_rate_deg: float = 80.0,
    bump_output_delta: float = 0.12,
    model_demand_jerk: float = 1.2,
    model_path_quality_threshold: float = 0.7,
    cooldown_s: float = 1.0,
  ):
    self.measurement_spike_lataccel = measurement_spike_lataccel
    self.measurement_spike_rate_deg = measurement_spike_rate_deg
    self.bump_rate_deg = bump_rate_deg
    self.bump_output_delta = bump_output_delta
    self.model_demand_jerk = model_demand_jerk
    self.model_path_quality_threshold = model_path_quality_threshold
    self.cooldown_s = cooldown_s
    self._last_event_t: float | None = None
    self._last_event_confidence: float = 0.0
    self._last_event_decision: LearningDecision = LearningDecision.ACCEPT

  def classify(
    self,
    sample: LateralSample,
    *,
    prev_sample: LateralSample | None = None,
    dt: float | None = None,
  ) -> Classification:
    reasons = DisturbanceReason.NONE
    missing_context = False
    details: dict[str, Any] = {}

    # Derived deltas for jitter/spike detection.
    desired_jerk = 0.0
    output_delta = 0.0
    if prev_sample is not None and dt is not None and dt > 1e-6:
      if sample.desired_lateral_accel is not None and prev_sample.desired_lateral_accel is not None:
        desired_jerk = (sample.desired_lateral_accel - prev_sample.desired_lateral_accel) / dt
      if sample.output is not None and prev_sample.output is not None:
        output_delta = sample.output - prev_sample.output

    # Hard reject conditions (driver / lane-change / control-limit).
    if not sample.lat_active:
      reasons |= DisturbanceReason.LAT_INACTIVE
    if sample.steering_pressed:
      reasons |= DisturbanceReason.DRIVER_OVERRIDE
    if sample.blinker_active or sample.lane_change_active:
      reasons |= DisturbanceReason.LANE_CHANGE
    if sample.output_sat or sample.steer_limited:
      reasons |= DisturbanceReason.CONTROL_LIMIT

    hard_reject = bool(reasons & (
      DisturbanceReason.LAT_INACTIVE
      | DisturbanceReason.DRIVER_OVERRIDE
      | DisturbanceReason.LANE_CHANGE
      | DisturbanceReason.CONTROL_LIMIT
    ))

    if hard_reject:
      confidence = 1.0
      cooldown_remaining = self._update_cooldown(sample.t, LearningDecision.REJECT_SHADOW, confidence)
      return Classification(
        decision=LearningDecision.REJECT_SHADOW,
        confidence=confidence,
        reasons=reasons,
        cooldown_remaining=cooldown_remaining,
        details=details,
      )

    # Quarantine conditions.
    sr = sample.steering_rate_deg if sample.steering_rate_deg is not None else 0.0
    sr_abs = abs(sr)
    actual_abs = abs(sample.actual_lateral_accel) if sample.actual_lateral_accel is not None else 0.0

    if actual_abs >= self.measurement_spike_lataccel:
      reasons |= DisturbanceReason.MEASUREMENT_SPIKE
      details["actual_lataccel_abs"] = actual_abs

    if sr_abs >= self.measurement_spike_rate_deg:
      reasons |= DisturbanceReason.MEASUREMENT_SPIKE
      details["steering_rate_abs"] = sr_abs

    if sr_abs >= self.bump_rate_deg and abs(output_delta) >= self.bump_output_delta:
      reasons |= DisturbanceReason.BUMP
      details["steering_rate_abs"] = sr_abs
      details["output_delta"] = output_delta

    if abs(desired_jerk) >= self.model_demand_jerk:
      reasons |= DisturbanceReason.MODEL_DEMAND_JITTER
      details["desired_jerk"] = desired_jerk

    if sample.model_path_gated:
      reasons |= DisturbanceReason.MODEL_PATH_LOW_QUALITY
      details["model_path_gated"] = True
    elif sample.model_path_quality is not None and sample.model_path_quality < self.model_path_quality_threshold:
      reasons |= DisturbanceReason.MODEL_PATH_LOW_QUALITY
      details["model_path_quality"] = sample.model_path_quality

    # Missing optional context lowers confidence but must not blanket reject/quarantine.
    if any(v is None for v in (
      sample.steering_rate_deg,
      sample.output,
      sample.desired_lateral_accel,
      sample.model_path_quality,
    )):
      missing_context = True
      reasons |= DisturbanceReason.MISSING_CONTEXT

    quarantine = bool(reasons & (
      DisturbanceReason.MEASUREMENT_SPIKE
      | DisturbanceReason.BUMP
      | DisturbanceReason.MODEL_DEMAND_JITTER
      | DisturbanceReason.MODEL_PATH_LOW_QUALITY
    ))

    # Cooldown/hysteresis after a high-confidence event.
    cooldown_remaining = 0.0
    if self._last_event_t is not None:
      elapsed = sample.t - self._last_event_t
      if elapsed < self.cooldown_s and self._last_event_confidence >= 0.6:
        cooldown_remaining = max(0.0, self.cooldown_s - elapsed)
        reasons |= DisturbanceReason.COOLDOWN_ACTIVE
        if self._last_event_decision == LearningDecision.REJECT_SHADOW:
          # Re-boost only if the same trigger class still looks active; otherwise just lower confidence sticky.
          if hard_reject:
            return Classification(
              decision=LearningDecision.REJECT_SHADOW,
              confidence=0.7,
              reasons=reasons,
              cooldown_remaining=cooldown_remaining,
              details=details,
            )

    if quarantine:
      confidence = 0.7
      if missing_context:
        confidence = max(0.35, confidence - 0.2)
      if reasons & (DisturbanceReason.MEASUREMENT_SPIKE | DisturbanceReason.BUMP):
        confidence = min(0.9, confidence + 0.1)
      cooldown_remaining = self._update_cooldown(sample.t, LearningDecision.QUARANTINE, confidence)
      return Classification(
        decision=LearningDecision.QUARANTINE,
        confidence=confidence,
        reasons=reasons,
        cooldown_remaining=cooldown_remaining,
        details=details,
      )

    # Clean accept (cooldown_active samples are still clean if no reject/quarantine reasons).
    if not (reasons & ~(DisturbanceReason.MISSING_CONTEXT | DisturbanceReason.COOLDOWN_ACTIVE)):
      reasons |= DisturbanceReason.CLEAN

    confidence = 0.6 if missing_context else 1.0
    if reasons & DisturbanceReason.COOLDOWN_ACTIVE:
      confidence = max(0.35, confidence - 0.15)

    return Classification(
      decision=LearningDecision.ACCEPT,
      confidence=confidence,
      reasons=reasons,
      cooldown_remaining=cooldown_remaining,
      details=details,
    )

  def _update_cooldown(self, t: float, decision: LearningDecision, confidence: float) -> float:
    self._last_event_t = t
    self._last_event_confidence = confidence
    self._last_event_decision = decision
    return self.cooldown_s


def reason_names(reasons: DisturbanceReason) -> list[str]:
  names = []
  for bit in DisturbanceReason:
    if bit == DisturbanceReason.NONE:
      continue
    if reasons & bit:
      names.append(bit.name)
  return names


def decision_name(decision: LearningDecision) -> str:
  return decision.name.lower()
