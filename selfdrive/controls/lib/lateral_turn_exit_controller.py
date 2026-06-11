from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from openpilot.selfdrive.controls.lib.lateral_demand_profile import (
  LateralMode,
  LateralDemandProfile,
)


PREVIEW_HORIZON_S = (0.2, 0.5, 1.0)

TURN_EXIT_MAX_ABS_TARGET = 0.85
TURN_EXIT_COLLAPSE_PER_FRAME = 0.015
TURN_EXIT_STABLE_SIGN_FRAMES = 3
TURN_IN_MIN_LAT_ACCEL = 0.10
TURN_IN_MIN_ABS_TARGET_RATE = 0.15
TURN_IN_PREVIEW_BOOST_GAIN = 0.5
TURN_IN_PREVIEW_BOOST_CAP = 0.6
STEADY_CURVE_MIN_LAT_ACCEL = 0.10

RECENTER_PERSISTENCE_FRAMES = 3
RECENTER_LEAD_REDUCTION = 0.6
RECENTER_SLEW_BOOST = 1.5
RECENTER_LEAD_REDUCTION_FLOOR = 0.3
RECENTER_SAME_DIRECTION_SLEW_BOOST = 1.2
RECENTER_SAME_DIRECTION_LIMIT_RATE_BONUS = 0.0


class TurnExitMode(str, Enum):
  TURN_IN = "turn_in"
  STEADY_CURVE = "steady_curve"
  TURN_EXIT = "turn_exit"
  EARLY_RELEASE = "early_release"
  INACTIVE = "inactive"


@dataclass(frozen=True)
class TurnExitDecision:
  mode: str
  persistence_frames: int
  lead_gain_multiplier: float
  lead_delta_cap_multiplier: float
  slew_boost: float
  same_direction_slew_boost: float
  early_release_lead_zero: bool
  preview_boost: float
  confidence: float


@dataclass
class LateralTurnExitController:
  dt: float = 0.05

  def __post_init__(self) -> None:
    self._recenter_persistence_frames: int = 0
    self._previous_target_lateral_accel: float = 0.0
    self._initialised: bool = False

  def reset(self) -> None:
    self._recenter_persistence_frames = 0
    self._previous_target_lateral_accel = 0.0
    self._initialised = False

  def update(
    self,
    target: float,
    profile: LateralDemandProfile | None,
    *,
    active: bool = True,
    v_ego: float = 0.0,
    path_quality: float = 1.0,
    lane_change_active: bool = False,
    steering_pressed: bool = False,
    curvature_limited: bool = False,
    saturated: bool = False,
  ) -> TurnExitDecision:
    """Per-frame turn-in/turn-exit/early-release decision.

    Returns a decision object with the lead gain multiplier, the lead cap
    multiplier, the slew boost, an early-release flag that zeros the lead
    on the first collapse frame, and a preview boost derived from the
    demand profile's preview lateral accel.
    """
    if not active:
      self._initialised = False
      return TurnExitDecision(
        mode=TurnExitMode.INACTIVE.value,
        persistence_frames=0,
        lead_gain_multiplier=1.0,
        lead_delta_cap_multiplier=1.0,
        slew_boost=1.0,
        same_direction_slew_boost=1.0,
        early_release_lead_zero=False,
        preview_boost=0.0,
        confidence=0.0,
      )

    if not self._initialised:
      self._previous_target_lateral_accel = target
      self._recenter_persistence_frames = 0
      self._initialised = True

    previous_target = self._previous_target_lateral_accel
    target_rate = (target - previous_target) / max(float(self.dt), 1e-3)
    self._previous_target_lateral_accel = target

    can_use_recenter = (
      math.isfinite(v_ego) and v_ego >= 10.0
      and path_quality >= 0.5
      and not lane_change_active
      and not steering_pressed
      and not saturated
      and not curvature_limited
    )

    if not can_use_recenter:
      self._recenter_persistence_frames = 0
    else:
      target_decreasing_to_zero = abs(target) < abs(previous_target) and target != 0.0
      collapse_per_frame = abs(previous_target) - abs(target)
      target_near_center = abs(target) < TURN_EXIT_MAX_ABS_TARGET
      signs_stable = _signs_stable(target, previous_target)
      if target_decreasing_to_zero and signs_stable and target_near_center and collapse_per_frame > TURN_EXIT_COLLAPSE_PER_FRAME:
        self._recenter_persistence_frames = min(
          self._recenter_persistence_frames + 1,
          RECENTER_PERSISTENCE_FRAMES * 3,
        )
      else:
        self._recenter_persistence_frames = max(0, self._recenter_persistence_frames - 1)

    target_decreasing_to_zero = abs(target) < abs(previous_target) and target != 0.0
    signs_stable = _signs_stable(target, previous_target)
    early_release_lead_zero = bool(
      target_decreasing_to_zero
      and signs_stable
      and target != 0.0
      and math.isfinite(target_rate)
    )

    turn_in_active = abs(target) > TURN_IN_MIN_LAT_ACCEL and abs(target_rate) > TURN_IN_MIN_ABS_TARGET_RATE
    preview_boost = 0.0
    if turn_in_active and not early_release_lead_zero:
      preview_target = target + target_rate * PREVIEW_HORIZON_S[0]
      preview_boost = _compute_preview_boost(
        target=target,
        preview_0_2s=preview_target,
        target_rate=target_rate,
        v_ego=v_ego,
      )

    # Recenter is active only after the persistence floor has
    # cleared. Before that, lead gain / cap multipliers, slew
    # boost, and same-direction slew boost all stay at the
    # neutral 1.0 value. early_release_lead_zero may still be
    # true on a collapse frame; it just means the consumer of
    # this decision must zero lead directly, not that lead
    # multipliers should be reduced.
    recenter_active = self._recenter_persistence_frames >= RECENTER_PERSISTENCE_FRAMES
    if recenter_active:
      ramp = max(0.0, min(1.0, float(
        (self._recenter_persistence_frames - RECENTER_PERSISTENCE_FRAMES) / RECENTER_PERSISTENCE_FRAMES
      )))
      persistence_blend = RECENTER_LEAD_REDUCTION_FLOOR + (1.0 - RECENTER_LEAD_REDUCTION_FLOOR) * ramp
      lead_reduction = RECENTER_LEAD_REDUCTION * persistence_blend
      lead_gain_mult = 1.0 - lead_reduction
      lead_cap_mult = 1.0 - lead_reduction
      slew_boost = 1.0 + (RECENTER_SLEW_BOOST - 1.0) * persistence_blend
      same_direction_slew_boost = 1.0 + (RECENTER_SAME_DIRECTION_SLEW_BOOST - 1.0) * persistence_blend
    else:
      lead_gain_mult = 1.0
      lead_cap_mult = 1.0
      slew_boost = 1.0
      same_direction_slew_boost = 1.0

    if early_release_lead_zero and recenter_active:
      mode = TurnExitMode.TURN_EXIT.value
      confidence = 0.95
    elif early_release_lead_zero:
      mode = TurnExitMode.EARLY_RELEASE.value
      confidence = 0.95
    elif recenter_active:
      mode = TurnExitMode.TURN_EXIT.value
      confidence = 0.9
    elif turn_in_active:
      mode = TurnExitMode.TURN_IN.value
      confidence = 0.85
    elif abs(target) > STEADY_CURVE_MIN_LAT_ACCEL:
      mode = TurnExitMode.STEADY_CURVE.value
      confidence = 0.7
    else:
      mode = TurnExitMode.INACTIVE.value
      confidence = 0.5

    return TurnExitDecision(
      mode=mode,
      persistence_frames=self._recenter_persistence_frames,
      lead_gain_multiplier=lead_gain_mult,
      lead_delta_cap_multiplier=lead_cap_mult,
      slew_boost=slew_boost,
      same_direction_slew_boost=same_direction_slew_boost,
      early_release_lead_zero=early_release_lead_zero,
      preview_boost=preview_boost,
      confidence=confidence,
    )

    turn_in_active = abs(target) > TURN_IN_MIN_LAT_ACCEL and abs(target_rate) > TURN_IN_MIN_ABS_TARGET_RATE
    preview_boost = 0.0
    if turn_in_active and not early_release_lead_zero:
      preview_target = target + target_rate * PREVIEW_HORIZON_S[0]
      preview_boost = _compute_preview_boost(
        target=target,
        preview_0_2s=preview_target,
        target_rate=target_rate,
        v_ego=v_ego,
      )

    if early_release_lead_zero and self._recenter_persistence_frames >= RECENTER_PERSISTENCE_FRAMES:
      mode = TurnExitMode.TURN_EXIT.value
      confidence = 0.95
    elif early_release_lead_zero:
      mode = TurnExitMode.EARLY_RELEASE.value
      confidence = 0.95
    elif self._recenter_persistence_frames >= RECENTER_PERSISTENCE_FRAMES:
      mode = TurnExitMode.TURN_EXIT.value
      confidence = 0.9
    elif turn_in_active:
      mode = TurnExitMode.TURN_IN.value
      confidence = 0.85
    elif abs(target) > STEADY_CURVE_MIN_LAT_ACCEL:
      mode = TurnExitMode.STEADY_CURVE.value
      confidence = 0.7
    else:
      mode = TurnExitMode.INACTIVE.value
      confidence = 0.5

    return TurnExitDecision(
      mode=mode,
      persistence_frames=self._recenter_persistence_frames,
      lead_gain_multiplier=lead_gain_mult,
      lead_delta_cap_multiplier=lead_cap_mult,
      slew_boost=slew_boost,
      same_direction_slew_boost=same_direction_slew_boost,
      early_release_lead_zero=early_release_lead_zero,
      preview_boost=preview_boost,
      confidence=confidence,
    )


def _signs_stable(a: float, b: float) -> bool:
  if a == 0.0 and b == 0.0:
    return True
  if a == 0.0 or b == 0.0:
    return False
  return (a > 0.0) == (b > 0.0)


def _compute_preview_boost(*, target: float, preview_0_2s: float, target_rate: float, v_ego: float) -> float:
  if not math.isfinite(preview_0_2s) or not math.isfinite(target) or not math.isfinite(target_rate):
    return 0.0
  if v_ego < 5.0:
    return 0.0
  boost = (preview_0_2s - target) * TURN_IN_PREVIEW_BOOST_GAIN
  boost = max(-TURN_IN_PREVIEW_BOOST_CAP, min(TURN_IN_PREVIEW_BOOST_CAP, float(boost)))
  if (boost > 0.0) != (target_rate > 0.0):
    return 0.0
  return float(boost)
