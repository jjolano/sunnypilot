"""Coast-horizon anticipation (Phase 5) — physics-based lift-off timing.

The principled upgrade to the leeway-band heuristic in ``policy.dynamic_cruise_coast_accel``:
instead of a fixed overspeed band, compute the coast-down trajectory from current speed, the
natural coast deceleration (drag + grade), and choose the lift-off point so speed bleeds to
the constraint target (a slower lead's speed, a curve cap, a speed-limit change) *exactly* at
the constraint — the human hypermiler's "lift early, arrive at the limit without braking".

The metric for validation is "matches the human lift-off point" from the manual-driving
baselines (engaged data not required to build/property-test this; the drag estimate is
learned online from observed coast decel). See the restart plan's Phase 5.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

ACCELERATION_DUE_TO_GRAVITY = 9.81

# Online drag/coast estimate bounds (m/s^2). Natural coast decel is negative.
MIN_COAST_DECEL = -2.0
MAX_COAST_DECEL = -0.02
DEFAULT_COAST_DECEL = -0.25  # flat-road rolling+aero proxy until learned
COAST_ARRIVAL_MARGIN_S = 0.6  # lift this many seconds of slack early (comfort)
MIN_USEFUL_DISTANCE = 1.0


class CoastAction(Enum):
  CRUISE = "cruise"    # constraint is far / not slower: hold speed
  COAST = "coast"      # lift off now: coasting arrives at the target ~at the constraint
  BRAKE = "brake"      # coasting cannot bleed enough: braking required


@dataclass(frozen=True)
class CoastHorizonInputs:
  v_ego: float
  v_target: float               # required speed at the constraint
  distance_to_constraint: float
  accel_coast: float            # natural coast decel (negative); from DragEstimator or grade model
  comfort_brake_decel: float = -1.0


@dataclass(frozen=True)
class CoastHorizonResult:
  action: CoastAction
  recommended_accel: float       # 0 while coasting; the required (negative) accel while braking
  coast_distance: float          # distance coasting alone needs to reach v_target
  lift_off_distance: float       # distance-before-constraint at which to lift off
  slack: float                   # distance_to_constraint - lift_off_distance (>0 => still cruising)


def coast_decel_from_grade(rolling_coast_decel: float, pitch_rad: float) -> float:
  """Natural coast decel including road grade. Uphill (positive pitch) adds deceleration."""
  grade_decel = -ACCELERATION_DUE_TO_GRAVITY * math.sin(_finite(pitch_rad))
  return _clip(_finite(rolling_coast_decel) + grade_decel, MIN_COAST_DECEL, -1e-3)


def coast_horizon(inp: CoastHorizonInputs) -> CoastHorizonResult:
  v0 = max(0.0, _finite(inp.v_ego))
  v_t = max(0.0, _finite(inp.v_target))
  dist = _finite(inp.distance_to_constraint)
  a_coast = _clip(_finite(inp.accel_coast), MIN_COAST_DECEL, MAX_COAST_DECEL)

  # No need to slow (target >= current): hold speed.
  if v_t >= v0:
    return CoastHorizonResult(CoastAction.CRUISE, 0.0, 0.0, 0.0, dist)

  # Need to slow but (almost) no runway left: coasting can't help, so brake at the decel the
  # remaining distance requires (floored to avoid a divide-by-~0 blowup). Treating this as
  # CRUISE — as the old combined guard did — zeroed out braking in the last metre of a stop or
  # curve approach (the runway governor / lead cushion would relax raw braking to 0).
  if dist <= MIN_USEFUL_DISTANCE:
    required = (v_t * v_t - v0 * v0) / (2.0 * MIN_USEFUL_DISTANCE)
    return CoastHorizonResult(CoastAction.BRAKE, required, 0.0, 0.0, dist)

  # Distance for coasting alone to bleed v0 -> v_t:  v_t^2 = v0^2 + 2*a*x  =>  x = (v_t^2 - v0^2)/(2a)
  coast_distance = (v_t * v_t - v0 * v0) / (2.0 * a_coast)
  # Lift a little early so we arrive with comfort slack, not braking at the last metre.
  lift_off_distance = coast_distance + COAST_ARRIVAL_MARGIN_S * v0
  slack = dist - lift_off_distance

  if dist > lift_off_distance:
    return CoastHorizonResult(CoastAction.CRUISE, 0.0, coast_distance, lift_off_distance, slack)
  if dist >= coast_distance:
    # within the lift window: lift off and let speed bleed at the natural coast decel
    # (commanding a_coast, not 0 — 0 would hold speed instead of coasting down)
    return CoastHorizonResult(CoastAction.COAST, a_coast, coast_distance, lift_off_distance, slack)
  # Coasting can no longer bleed enough over the remaining distance -> brake at exactly the
  # decel the kinematics require (gentle for a far constraint, hard for a near one) — never the
  # blanket comfort floor, which would over-brake a far stop.
  required = (v_t * v_t - v0 * v0) / (2.0 * max(dist, MIN_USEFUL_DISTANCE))
  return CoastHorizonResult(CoastAction.BRAKE, required, coast_distance, lift_off_distance, slack)


class DragEstimator:
  """Online estimate of the flat-road natural coast deceleration from observed decel while
  off-throttle and off-brake (drag + rolling resistance), grade-compensated."""

  def __init__(self, alpha: float = 0.02):
    self.alpha = float(alpha)
    self.coast_decel = DEFAULT_COAST_DECEL

  def update(self, v_ego: float, a_ego: float, pitch_rad: float, on_throttle: bool, on_brake: bool) -> float:
    if on_throttle or on_brake or _finite(v_ego) < 3.0:
      return self.coast_decel
    # Remove grade from the measured accel to isolate rolling+aero drag.
    grade_decel = -ACCELERATION_DUE_TO_GRAVITY * math.sin(_finite(pitch_rad))
    rolling = _finite(a_ego) - grade_decel
    if MIN_COAST_DECEL <= rolling <= MAX_COAST_DECEL:
      self.coast_decel += self.alpha * (rolling - self.coast_decel)
      self.coast_decel = _clip(self.coast_decel, MIN_COAST_DECEL, MAX_COAST_DECEL)
    return self.coast_decel


def _finite(value: float, default: float = 0.0) -> float:
  try:
    v = float(value)
  except (TypeError, ValueError):
    return default
  return v if math.isfinite(v) else default


def _clip(value: float, lo: float, hi: float) -> float:
  return max(lo, min(hi, value))
