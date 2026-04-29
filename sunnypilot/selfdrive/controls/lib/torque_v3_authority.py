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


FAULT_REASONS = (
  EstimatorRejectReason.SIGN_CONFLICT
  | EstimatorRejectReason.STEER_LIMITED
  | EstimatorRejectReason.SATURATED
  | EstimatorRejectReason.NON_FINITE
  | EstimatorRejectReason.HIGH_JERK
)


class AuthorityManager:
  def __init__(self):
    self.state = AuthorityState(AuthorityBand.limited, 0.45, False)

  def update(self, mode: TorqueModelMode, confidence: float, positive_coverage: float, negative_coverage: float,
             reject_reason: EstimatorRejectReason) -> AuthorityState:
    if reject_reason & FAULT_REASONS:
      self.state = AuthorityState(AuthorityBand.limited, 0.45, True)
      return self.state

    bidirectional = positive_coverage >= 0.5 and negative_coverage >= 0.5
    if mode == TorqueModelMode.native and confidence >= 0.75:
      self.state = AuthorityState(AuthorityBand.near_full, 0.85, False)
    elif mode == TorqueModelMode.learned and confidence >= 0.95 and bidirectional:
      self.state = AuthorityState(AuthorityBand.full, 1.0, False)
    elif confidence >= 0.75 and bidirectional:
      self.state = AuthorityState(AuthorityBand.near_full, 0.85, False)
    elif confidence >= 0.4:
      self.state = AuthorityState(AuthorityBand.partial, 0.65, False)
    else:
      self.state = AuthorityState(AuthorityBand.limited, 0.45, False)
    return self.state
