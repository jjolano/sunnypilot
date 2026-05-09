import math
from dataclasses import dataclass
from enum import IntFlag

import numpy as np

from openpilot.sunnypilot.selfdrive.controls.lib.torque_v3_model import TorqueModelParams


MIN_LEARNING_VEGO = 5.0
MAX_MODEL_AGE = 0.25
MAX_PLAUSIBLE_JERK = 8.0
MIN_COMMAND_TORQUE = 0.05
SIGN_CONFLICT_LATERAL_ACCEL_THRESHOLD = 0.05
MAX_RESIDUAL_SPIKE = 0.8
CONFIDENCE_BUILD_RATE = 0.015
CONFIDENCE_DECAY_RATE = 0.25
PARAM_UPDATE_RATE = 0.04
DEFAULT_FACTOR = 2.5
DEFAULT_OFFSET = 0.0
DEFAULT_FRICTION = 0.1


class EstimatorRejectReason(IntFlag):
  NONE = 0
  INACTIVE = 1 << 0
  LOW_SPEED = 1 << 1
  STEERING_PRESSED = 1 << 2
  STEER_LIMITED = 1 << 3
  CURVATURE_LIMITED = 1 << 4
  SATURATED = 1 << 5
  LOW_COMMAND = 1 << 6
  NON_FINITE = 1 << 7
  HIGH_JERK = 1 << 8
  SIGN_CONFLICT = 1 << 9
  STALE_MODEL = 1 << 10
  RESIDUAL_SPIKE = 1 << 11


@dataclass
class TorqueObservation:
  active: bool
  v_ego: float
  steering_pressed: bool
  steer_limited_by_safety: bool
  curvature_limited: bool
  saturated: bool
  commanded_torque: float
  desired_lateral_accel: float
  actual_lateral_accel: float
  actual_lateral_jerk: float
  roll_compensation: float
  model_age: float


@dataclass
class EstimatorState:
  params: TorqueModelParams
  confidence: float
  positive_coverage: float
  negative_coverage: float
  residual_error: float
  response_delay: float


@dataclass
class EstimatorResult:
  params: TorqueModelParams
  confidence: float
  positive_coverage: float
  negative_coverage: float
  residual_error: float
  response_delay: float
  sample_accepted: bool
  reject_reason: EstimatorRejectReason


def _finite(*values: float) -> bool:
  return all(math.isfinite(float(value)) for value in values)


def _sign(value: float, threshold: float = 0.0) -> int:
  return 1 if value > threshold else (-1 if value < -threshold else 0)


class AdaptiveTorqueEstimator:
  def __init__(self, dt: float):
    self.dt = dt
    self.state = EstimatorState(
      params=TorqueModelParams(DEFAULT_FACTOR, DEFAULT_OFFSET, DEFAULT_FRICTION),
      confidence=0.0,
      positive_coverage=0.0,
      negative_coverage=0.0,
      residual_error=0.0,
      response_delay=max(dt, 0.01),
    )

  def update(self, observation: TorqueObservation) -> EstimatorResult:
    reject_reason = self._reject_reason(observation)
    if reject_reason != EstimatorRejectReason.NONE:
      self._decay_confidence(reject_reason)
      return self._result(False, reject_reason)

    observed_factor = abs(observation.actual_lateral_accel - self.state.params.lat_accel_offset) / max(abs(observation.commanded_torque), MIN_COMMAND_TORQUE)
    observed_factor = float(np.clip(observed_factor, 0.2, 8.0))
    self.state.params.lat_accel_factor += PARAM_UPDATE_RATE * (observed_factor - self.state.params.lat_accel_factor)
    self.state.params.lat_accel_offset += PARAM_UPDATE_RATE * np.clip(observation.roll_compensation - self.state.params.lat_accel_offset, -0.2, 0.2)

    expected = observation.commanded_torque * self.state.params.lat_accel_factor + self.state.params.lat_accel_offset
    residual = observation.actual_lateral_accel - expected
    self.state.residual_error = 0.9 * self.state.residual_error + 0.1 * residual

    if observation.commanded_torque > 0.0:
      self.state.positive_coverage = min(1.0, self.state.positive_coverage + CONFIDENCE_BUILD_RATE)
    elif observation.commanded_torque < 0.0:
      self.state.negative_coverage = min(1.0, self.state.negative_coverage + CONFIDENCE_BUILD_RATE)

    coverage = min(self.state.positive_coverage + self.state.negative_coverage, 1.0)
    self.state.confidence = min(1.0, self.state.confidence + CONFIDENCE_BUILD_RATE * (0.5 + coverage))
    return self._result(True, EstimatorRejectReason.NONE)

  def _reject_reason(self, observation: TorqueObservation) -> EstimatorRejectReason:
    reason = EstimatorRejectReason.NONE
    if not observation.active:
      reason |= EstimatorRejectReason.INACTIVE
    if observation.v_ego < MIN_LEARNING_VEGO:
      reason |= EstimatorRejectReason.LOW_SPEED
    if observation.steering_pressed:
      reason |= EstimatorRejectReason.STEERING_PRESSED
    if observation.steer_limited_by_safety:
      reason |= EstimatorRejectReason.STEER_LIMITED
    if observation.curvature_limited:
      reason |= EstimatorRejectReason.CURVATURE_LIMITED
    if observation.saturated:
      reason |= EstimatorRejectReason.SATURATED
    if abs(observation.commanded_torque) < MIN_COMMAND_TORQUE:
      reason |= EstimatorRejectReason.LOW_COMMAND
    if not _finite(observation.v_ego, observation.commanded_torque, observation.desired_lateral_accel, observation.actual_lateral_accel,
                   observation.actual_lateral_jerk, observation.roll_compensation, observation.model_age):
      reason |= EstimatorRejectReason.NON_FINITE
    if abs(observation.actual_lateral_jerk) > MAX_PLAUSIBLE_JERK:
      reason |= EstimatorRejectReason.HIGH_JERK
    if observation.model_age > MAX_MODEL_AGE:
      reason |= EstimatorRejectReason.STALE_MODEL
    command_sign = _sign(observation.commanded_torque, MIN_COMMAND_TORQUE)
    actual_sign = _sign(observation.actual_lateral_accel, SIGN_CONFLICT_LATERAL_ACCEL_THRESHOLD)
    desired_sign = _sign(observation.desired_lateral_accel, SIGN_CONFLICT_LATERAL_ACCEL_THRESHOLD)
    if command_sign != 0 and actual_sign != 0 and desired_sign != 0 and len({command_sign, actual_sign, desired_sign}) > 1:
      reason |= EstimatorRejectReason.SIGN_CONFLICT
    expected = observation.commanded_torque * self.state.params.lat_accel_factor + self.state.params.lat_accel_offset
    if abs(observation.actual_lateral_accel - expected) > MAX_RESIDUAL_SPIKE:
      reason |= EstimatorRejectReason.RESIDUAL_SPIKE
    return reason

  def _decay_confidence(self, reason: EstimatorRejectReason) -> None:
    clipping_limited = bool(reason & (EstimatorRejectReason.STEER_LIMITED | EstimatorRejectReason.SATURATED))
    sign_conflict_from_clipped_steer_limit = bool(
      reason & EstimatorRejectReason.SIGN_CONFLICT and reason & EstimatorRejectReason.STEER_LIMITED and reason & EstimatorRejectReason.SATURATED
    )
    fast_decay = bool(reason & EstimatorRejectReason.SIGN_CONFLICT and not sign_conflict_from_clipped_steer_limit) or bool(
      reason & EstimatorRejectReason.RESIDUAL_SPIKE and not clipping_limited
    )
    decay = CONFIDENCE_DECAY_RATE if fast_decay else CONFIDENCE_BUILD_RATE
    self.state.confidence = max(0.0, self.state.confidence - decay)

  def _result(self, accepted: bool, reject_reason: EstimatorRejectReason) -> EstimatorResult:
    params = TorqueModelParams(
      self.state.params.lat_accel_factor,
      self.state.params.lat_accel_offset,
      self.state.params.friction,
    )
    return EstimatorResult(
      params=params,
      confidence=self.state.confidence,
      positive_coverage=self.state.positive_coverage,
      negative_coverage=self.state.negative_coverage,
      residual_error=self.state.residual_error,
      response_delay=self.state.response_delay,
      sample_accepted=accepted,
      reject_reason=reject_reason,
    )
