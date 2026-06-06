from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

from openpilot.selfdrive.controls.lib.longitudinal_decision import (
  CandidateRole,
  DecisionSource,
  LongitudinalArbiter,
  LongitudinalCandidate,
)


CURVE_POLICY_SOURCES = frozenset((DecisionSource.SCC_VISION, DecisionSource.SCC_MAP))


def _finite(value: object) -> bool:
  return isinstance(value, Real) and math.isfinite(float(value))


def _clamp01(value: object) -> float:
  if not isinstance(value, Real):
    return 0.0
  value_float = float(value)
  return min(1.0, max(0.0, value_float)) if math.isfinite(value_float) else 0.0


@dataclass(frozen=True)
class LongitudinalPolicyHorizon:
  source: DecisionSource
  active: bool
  reason: str
  v_target: float
  a_target: float
  distance: float | None = None
  time: float | None = None
  confidence: float = 1.0
  urgency: float = 0.0

  @property
  def valid(self) -> bool:
    if not self.active:
      return False
    if not isinstance(self.source, DecisionSource):
      return False
    if not str(self.reason):
      return False
    if not (_finite(self.v_target) and float(self.v_target) >= 0.0 and _finite(self.a_target)):
      return False
    if self.distance is not None and (not _finite(self.distance) or float(self.distance) < 0.0):
      return False
    if self.time is not None and (not _finite(self.time) or float(self.time) < 0.0):
      return False
    return True

  @property
  def normalized_confidence(self) -> float:
    return _clamp01(self.confidence)

  @property
  def normalized_urgency(self) -> float:
    return _clamp01(self.urgency)


@dataclass(frozen=True)
class CurveSpeedPolicyResult:
  source: DecisionSource
  active: bool
  reason: str
  v_target: float
  a_target: float
  confidence: float
  urgency: float = 0.0
  lateral_accel_limit: float = 0.0
  horizon: LongitudinalPolicyHorizon | None = None

  @property
  def valid(self) -> bool:
    if not self.active:
      return False
    if self.source not in CURVE_POLICY_SOURCES:
      return False
    if not str(self.reason):
      return False
    if not (_finite(self.v_target) and float(self.v_target) >= 0.0 and _finite(self.a_target)):
      return False
    if self.horizon is not None and not self.horizon.valid:
      return False
    return True

  @property
  def normalized_confidence(self) -> float:
    if self.horizon is None:
      return _clamp01(self.confidence)
    return min(_clamp01(self.confidence), self.horizon.normalized_confidence)

  @property
  def normalized_urgency(self) -> float:
    if self.horizon is None:
      return _clamp01(self.urgency)
    return max(_clamp01(self.urgency), self.horizon.normalized_urgency)

  def to_candidate(self) -> LongitudinalCandidate | None:
    if not self.valid:
      return None
    debug = {
      "curve_lateral_accel_limit": float(self.lateral_accel_limit) if _finite(self.lateral_accel_limit) else 0.0,
      "curve_horizon_source": self.horizon.source.value if self.horizon is not None else self.source.value,
    }
    if self.horizon is not None:
      debug.update({
        "horizon_distance": float(self.horizon.distance) if self.horizon.distance is not None else None,
        "horizon_time": float(self.horizon.time) if self.horizon.time is not None else None,
        "required_a_target": float(self.horizon.a_target),
      })
    return LongitudinalCandidate(
      source=self.source,
      role=CandidateRole.ADVISORY_CAP,
      v_target=float(self.v_target),
      a_target=float(self.a_target),
      confidence=self.normalized_confidence,
      urgency=self.normalized_urgency,
      active_reason=str(self.reason),
      debug=debug,
    )


def select_curve_speed_policy_result(
  results: tuple[CurveSpeedPolicyResult, ...] | list[CurveSpeedPolicyResult],
  driver_v_target: float,
  driver_a_target: float = 0.0,
) -> CurveSpeedPolicyResult | None:
  driver = LongitudinalCandidate(
    source=DecisionSource.CRUISE,
    role=CandidateRole.DRIVER_INTENT,
    v_target=float(driver_v_target),
    a_target=float(driver_a_target),
    confidence=1.0,
    urgency=0.0,
    active_reason="driver_set_speed",
  )
  pairs = [(result, result.to_candidate()) for result in results]
  candidate_pairs = [(result, candidate) for result, candidate in pairs if candidate is not None]
  if not candidate_pairs:
    return None

  decision = LongitudinalArbiter().decide([driver, *(candidate for _result, candidate in candidate_pairs)])
  for result, candidate in candidate_pairs:
    if (
      candidate.source == decision.winner and
      candidate.v_target == decision.v_target and
      candidate.a_target == decision.a_target
    ):
      return result
  return None
