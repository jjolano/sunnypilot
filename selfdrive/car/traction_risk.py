from __future__ import annotations

from dataclasses import dataclass
from enum import IntFlag
import math


class TractionRiskReason(IntFlag):
  NONE = 0
  ESP_ACTIVE = 1 << 0
  WHEEL_SPEED_SPREAD = 1 << 1


@dataclass(frozen=True)
class TractionRiskState:
  risk: float
  raw_risk: float
  reason: int


MIN_WHEEL_SPREAD_SPEED = 3.0
MAX_WHEEL_SPREAD_STEERING_ANGLE_DEG = 8.0
WHEEL_SPREAD_START = 0.8
WHEEL_SPREAD_FULL = 3.0
WHEEL_SPREAD_MAX_RISK = 0.65
ESP_ACTIVE_RISK = 1.0
RISK_RISE_RATE = 4.0
RISK_DECAY_RATE = 0.20
RISK_REASON_CLEAR_THRESHOLD = 0.02


def _clip(value: float, lower: float, upper: float) -> float:
  return max(lower, min(upper, value))


def _as_finite(value: object, default: float = 0.0) -> float:
  try:
    value_float = float(value)
  except (TypeError, ValueError):
    return default
  return value_float if math.isfinite(value_float) else default


def _wheel_speed_spread_risk(car_state) -> float:
  v_ego = abs(_as_finite(getattr(car_state, "vEgo", 0.0)))
  steering_angle = abs(_as_finite(getattr(car_state, "steeringAngleDeg", 0.0)))
  if v_ego < MIN_WHEEL_SPREAD_SPEED or steering_angle > MAX_WHEEL_SPREAD_STEERING_ANGLE_DEG:
    return 0.0

  wheel_speeds = getattr(car_state, "wheelSpeeds", None)
  if wheel_speeds is None:
    return 0.0

  speeds = [
    _as_finite(getattr(wheel_speeds, "fl", 0.0)),
    _as_finite(getattr(wheel_speeds, "fr", 0.0)),
    _as_finite(getattr(wheel_speeds, "rl", 0.0)),
    _as_finite(getattr(wheel_speeds, "rr", 0.0)),
  ]
  spread = max(speeds) - min(speeds)
  if spread <= WHEEL_SPREAD_START:
    return 0.0

  blend = _clip((spread - WHEEL_SPREAD_START) / max(WHEEL_SPREAD_FULL - WHEEL_SPREAD_START, 1e-3), 0.0, 1.0)
  return WHEEL_SPREAD_MAX_RISK * blend


class TractionRiskEstimator:
  def __init__(self, dt: float):
    self.dt = max(float(dt), 1e-3)
    self.risk = 0.0
    self.reason = TractionRiskReason.NONE

  def reset(self) -> None:
    self.risk = 0.0
    self.reason = TractionRiskReason.NONE

  def update(self, car_state) -> TractionRiskState:
    raw_risk = 0.0
    raw_reason = TractionRiskReason.NONE

    if bool(getattr(car_state, "espActive", False)):
      raw_risk = max(raw_risk, ESP_ACTIVE_RISK)
      raw_reason |= TractionRiskReason.ESP_ACTIVE

    wheel_risk = _wheel_speed_spread_risk(car_state)
    if wheel_risk > 0.0:
      raw_risk = max(raw_risk, wheel_risk)
      raw_reason |= TractionRiskReason.WHEEL_SPEED_SPREAD

    if raw_risk > self.risk:
      self.risk = min(raw_risk, self.risk + RISK_RISE_RATE * self.dt)
    else:
      self.risk = max(raw_risk, self.risk - RISK_DECAY_RATE * self.dt)

    if raw_reason != TractionRiskReason.NONE:
      self.reason = raw_reason
    elif self.risk <= RISK_REASON_CLEAR_THRESHOLD:
      self.reason = TractionRiskReason.NONE

    return TractionRiskState(float(_clip(self.risk, 0.0, 1.0)), float(_clip(raw_risk, 0.0, 1.0)), int(self.reason))
