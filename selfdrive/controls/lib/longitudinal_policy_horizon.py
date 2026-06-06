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
RESTRICTIVE_AUTHORITY = "restrictive"
INFORMATIONAL_AUTHORITY = "informational"


def _finite(value: object) -> bool:
  return isinstance(value, Real) and math.isfinite(float(value))


def _clamp01(value: object) -> float:
  if not isinstance(value, Real):
    return 0.0
  value_float = float(value)
  return min(1.0, max(0.0, value_float)) if math.isfinite(value_float) else 0.0


@dataclass(frozen=True)
class AdvisoryConstraint:
  source: str
  target_speed: float | None
  target_distance: float | None
  target_accel: float | None
  confidence: float
  urgency: float
  authority: str = RESTRICTIVE_AUTHORITY

  @property
  def valid(self) -> bool:
    if not str(self.source):
      return False
    if self.authority not in (RESTRICTIVE_AUTHORITY, INFORMATIONAL_AUTHORITY):
      return False
    for value in (self.target_speed, self.target_distance, self.target_accel):
      if value is not None and not _finite(value):
        return False
    if self.target_speed is not None and float(self.target_speed) < 0.0:
      return False
    if self.target_distance is not None and float(self.target_distance) < 0.0:
      return False
    return True

  @property
  def restrictive(self) -> bool:
    return self.valid and self.authority == RESTRICTIVE_AUTHORITY

  @property
  def normalized_confidence(self) -> float:
    return _clamp01(self.confidence)


@dataclass(frozen=True)
class LongitudinalPolicyHorizonGrid:
  v_upper: tuple[float, ...]
  a_min: tuple[float, ...]
  a_max: tuple[float, ...]
  virtual_obstacle: tuple[float, ...]
  source_by_t: tuple[str, ...]
  confidence_by_t: tuple[float, ...]


def advisory_constraints_allowed_for_mode(mode: str) -> bool:
  return str(mode).upper() in {"SCC", "E2E"}


def advisory_constraint_for_speed_drop(source: str, current_speed: float, target_speed: float,
                                       distance: float, confidence: float = 1.0,
                                       urgency: float = 0.0) -> AdvisoryConstraint:
  current_speed = max(0.0, float(current_speed)) if _finite(current_speed) else 0.0
  target_speed = min(max(0.0, float(target_speed)) if _finite(target_speed) else 0.0, current_speed)
  distance = max(0.0, float(distance)) if _finite(distance) else 0.0
  target_accel = 0.0 if distance <= 0.0 else (target_speed * target_speed - current_speed * current_speed) / (2.0 * distance)
  return AdvisoryConstraint(
    source=source,
    target_speed=target_speed,
    target_distance=distance,
    target_accel=target_accel,
    confidence=confidence,
    urgency=urgency,
    authority=RESTRICTIVE_AUTHORITY,
  )


def build_longitudinal_policy_horizon(constraints: tuple[AdvisoryConstraint, ...] | list[AdvisoryConstraint], *,
                                      horizon_len: int = 5,
                                      default_v_upper: float = math.inf,
                                      default_a_min: float = -math.inf,
                                      default_a_max: float = math.inf,
                                      physical_hazard_active: bool = False) -> LongitudinalPolicyHorizonGrid:
  horizon_len = max(1, int(horizon_len))
  if physical_hazard_active:
    return LongitudinalPolicyHorizonGrid(
      v_upper=tuple(default_v_upper for _ in range(horizon_len)),
      a_min=tuple(default_a_min for _ in range(horizon_len)),
      a_max=tuple(default_a_max for _ in range(horizon_len)),
      virtual_obstacle=tuple(math.inf for _ in range(horizon_len)),
      source_by_t=tuple("physical_hazard" for _ in range(horizon_len)),
      confidence_by_t=tuple(1.0 for _ in range(horizon_len)),
    )

  valid = [constraint for constraint in constraints if constraint.restrictive]
  if not valid:
    return LongitudinalPolicyHorizonGrid(
      v_upper=tuple(default_v_upper for _ in range(horizon_len)),
      a_min=tuple(default_a_min for _ in range(horizon_len)),
      a_max=tuple(default_a_max for _ in range(horizon_len)),
      virtual_obstacle=tuple(math.inf for _ in range(horizon_len)),
      source_by_t=tuple("" for _ in range(horizon_len)),
      confidence_by_t=tuple(0.0 for _ in range(horizon_len)),
    )

  selected = min(valid, key=lambda constraint: (
    math.inf if constraint.target_accel is None else float(constraint.target_accel),
    math.inf if constraint.target_speed is None else float(constraint.target_speed),
    math.inf if constraint.target_distance is None else float(constraint.target_distance),
    -constraint.normalized_confidence,
  ))
  return LongitudinalPolicyHorizonGrid(
    v_upper=tuple(
      default_v_upper if selected.target_speed is None else min(default_v_upper, float(selected.target_speed))
      for _ in range(horizon_len)
    ),
    a_min=tuple(default_a_min for _ in range(horizon_len)),
    a_max=tuple(
      default_a_max if selected.target_accel is None else min(default_a_max, float(selected.target_accel))
      for _ in range(horizon_len)
    ),
    virtual_obstacle=tuple(
      math.inf if selected.target_distance is None else float(selected.target_distance)
      for _ in range(horizon_len)
    ),
    source_by_t=tuple(str(selected.source) for _ in range(horizon_len)),
    confidence_by_t=tuple(selected.normalized_confidence for _ in range(horizon_len)),
  )


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
