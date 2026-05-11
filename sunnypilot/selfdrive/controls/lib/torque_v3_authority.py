from dataclasses import dataclass
from enum import IntEnum

from openpilot.sunnypilot.selfdrive.controls.lib.torque_v3_estimator import EstimatorRejectReason
from openpilot.sunnypilot.selfdrive.controls.lib.torque_v3_model import TorqueModelMode


class AuthorityBand(IntEnum):
  limited = 0
  partial = 1
  near_full = 2
  full = 3


@dataclass
class AuthorityState:
  band: AuthorityBand
  scale: float
  fallback_active: bool


CLIPPING_REASONS = EstimatorRejectReason.STEER_LIMITED | EstimatorRejectReason.SATURATED
HARD_FAULT_REASONS = (
  EstimatorRejectReason.NON_FINITE
  | EstimatorRejectReason.HIGH_JERK
  | EstimatorRejectReason.STALE_MODEL
)
AUTHORITY_RECOVERY_CLEAN_FRAMES = 20
AUTHORITY_SCALE_RECOVERY_STEP = 0.03


def authority_fault_active(reject_reason: EstimatorRejectReason) -> bool:
  clipped_steer_limit = bool(reject_reason & EstimatorRejectReason.STEER_LIMITED and reject_reason & EstimatorRejectReason.SATURATED)
  sign_fault = bool(reject_reason & EstimatorRejectReason.SIGN_CONFLICT and not clipped_steer_limit)
  residual_fault = bool(reject_reason & EstimatorRejectReason.RESIDUAL_SPIKE and not reject_reason & CLIPPING_REASONS)
  return bool(reject_reason & HARD_FAULT_REASONS or sign_fault or residual_fault)


class AuthorityManager:
  def __init__(self):
    self.state: AuthorityState = AuthorityState(AuthorityBand.limited, 0.45, False)
    self.clean_frames_after_fallback: int = 0
    self.recovering_from_fallback: bool = False

  def current_state(self) -> AuthorityState:
    return self.state

  @staticmethod
  def _target_state(mode: TorqueModelMode, confidence: float, positive_coverage: float,
                    negative_coverage: float) -> AuthorityState:
    bidirectional = positive_coverage >= 0.5 and negative_coverage >= 0.5
    if mode == TorqueModelMode.native and confidence >= 0.75:
      return AuthorityState(AuthorityBand.near_full, 0.85, False)
    if mode == TorqueModelMode.learned and confidence >= 0.95 and bidirectional:
      return AuthorityState(AuthorityBand.full, 1.0, False)
    if confidence >= 0.75 and bidirectional:
      return AuthorityState(AuthorityBand.near_full, 0.85, False)
    if confidence >= 0.4:
      return AuthorityState(AuthorityBand.partial, 0.65, False)
    return AuthorityState(AuthorityBand.limited, 0.45, False)

  @staticmethod
  def _band_for_scale(scale: float) -> AuthorityBand:
    if scale >= 1.0:
      return AuthorityBand.full
    if scale >= 0.85:
      return AuthorityBand.near_full
    if scale >= 0.65:
      return AuthorityBand.partial
    return AuthorityBand.limited

  def update(self, mode: TorqueModelMode, confidence: float, positive_coverage: float, negative_coverage: float,
             reject_reason: EstimatorRejectReason) -> AuthorityState:
    if authority_fault_active(reject_reason):
      self.state = AuthorityState(AuthorityBand.limited, 0.45, True)
      self.clean_frames_after_fallback = 0
      self.recovering_from_fallback = True
      return self.state

    target = self._target_state(mode, confidence, positive_coverage, negative_coverage)
    if target.scale <= self.state.scale:
      self.state = target
      self.clean_frames_after_fallback = 0
      self.recovering_from_fallback = False
      return self.state

    if not self.recovering_from_fallback:
      self.state = target
      self.clean_frames_after_fallback = 0
      return self.state

    if self.clean_frames_after_fallback < AUTHORITY_RECOVERY_CLEAN_FRAMES:
      self.clean_frames_after_fallback += 1
      return self.state

    next_scale = min(target.scale, self.state.scale + AUTHORITY_SCALE_RECOVERY_STEP)
    self.state = AuthorityState(self._band_for_scale(next_scale), next_scale, False)
    if next_scale >= target.scale:
      self.state = target
      self.clean_frames_after_fallback = 0
      self.recovering_from_fallback = False
    return self.state
