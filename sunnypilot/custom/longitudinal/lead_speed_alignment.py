"""Bidirectional lead-speed alignment helper (Phase 1).

Pure/deterministic recommendation for:
- slower lead: react early with coast/gentle decel when far/medium and low risk;
- pulling-away lead: release/accelerate earlier when stable/opening and safe;
- standstill launch: clear stop hold quickly when a stable lead moves, with immediate
  cancellation guards.

MPC remains the hard safety authority. In Phase 1 this helper only produces advisory caps
and authorized progress candidates; it never removes the existing physical hazard.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from openpilot.sunnypilot.custom.longitudinal.lead_cushion import lead_speedup_guard
from openpilot.sunnypilot.custom.longitudinal.policy_tables import (
  LEAD_LAUNCH_TAU,
  Personality,
  launch_accel_max,
)


class AlignmentAction(Enum):
  IGNORE = "ignore"
  COAST = "coast"
  GENTLE_BRAKE = "gentle_brake"
  PULLAWAY = "pullaway"
  STANDSTILL_LAUNCH = "standstill_launch"


@dataclass(frozen=True)
class LeadSpeedAlignment:
  action: AlignmentAction
  a_target: float
  required_decel: float
  desired_gap: float
  excess_gap: float
  closing: float
  reason: str


# Slowdown thresholds
_ALIGN_MIN_V_EGO = 5.0                # m/s; avoid stop-and-go context
_ALIGN_MIN_LEAD_V = 3.0               # m/s; avoid stopped/crawling leads
_ALIGN_MIN_EXCESS_GAP = 3.0           # m
_ALIGN_TINY_REQUIRED_DECEL = 0.07     # m/s^2
_ALIGN_COMFORT_REQUIRED_DECEL = 0.50  # m/s^2; full gentle brake reached at this point
_ALIGN_MAX_REQUIRED_DECEL = 0.80      # m/s^2; above this, MPC stays authoritative
_ALIGN_GENTLE_BRAKE_MAX = -0.35       # m/s^2
_ALIGN_COAST_A_TARGET = 0.0           # lift to neutral

# TTC / THW gates (Phase 1b, derived from manual-driving baseline analysis)
_ALIGN_NO_ADVISORY_TTC = 24.0         # s; above this, don't block progress with advisory
_ALIGN_COAST_TTC_MAIN = 24.0          # s; below this, coast preference
_ALIGN_STRONG_PREP_TTC = 6.0          # s; firm brake prep threshold
_ALIGN_HAZARD_TTC = 4.0               # s; below this, defer to MPC physical hazard

# Pullaway / launch thresholds
_ALIGN_STANDSTILL_V_EGO = 0.3         # m/s
_ALIGN_NEAR_FOLLOW_MARGIN = 2.0       # m; never accelerate inside/near follow gap
_ALIGN_MIN_OPENING = 0.15             # m/s; lead considered opening
_ALIGN_MIN_V_LEAD_PULLAWAY = 0.3      # m/s; lead considered moving
_ALIGN_PROGRESS_CRUISE_MARGIN = 0.2   # m/s
_ALIGN_MIN_CONFIDENCE = 0.55          # enough confidence to react to a far lead


def _finite(value: object) -> bool:
  try:
    return math.isfinite(float(value))  # type: ignore[arg-type]
  except (TypeError, ValueError):
    return False


def _normalize_personality(personality: Personality) -> Personality:
  if isinstance(personality, Personality):
    return personality
  return Personality.from_value(personality)


def _result(action: AlignmentAction, a_target: float, required: float, desired_gap: float,
            excess: float, closing: float, reason: str) -> LeadSpeedAlignment:
  return LeadSpeedAlignment(
    action=action,
    a_target=float(a_target),
    required_decel=float(required),
    desired_gap=float(desired_gap),
    excess_gap=float(excess),
    closing=float(closing),
    reason=str(reason),
  )


def lead_speed_alignment(
  v_ego: float,
  a_ego: float,
  v_cruise: float,
  lead_d_rel: float,
  lead_v: float,
  lead_v_rel: float,
  lead_a_k: float,
  follow_gap: float,
  lead_confidence: float,
  lead_stable: bool,
  lead_progress_allowed: bool,
  lead_shadow_active: bool,
  alternate_threat_active: bool,
  model_should_stop: bool,
  force_slow_decel: bool,
  brake_pressed: bool,
  gas_pressed: bool,
  personality: Personality,
  lead_kinematics_valid: bool,
  has_lead: bool = False,
) -> LeadSpeedAlignment:
  """Return a soft recommendation for lead speed alignment.

  Fail-closed: any unsafe/untrusted condition returns IGNORE with a_target 0.0, leaving
  the existing physical hazard candidate authoritative.
  """
  personality = _normalize_personality(personality)

  # Safety gates: fail closed/ignore.
  if not has_lead:
    return _result(AlignmentAction.IGNORE, 0.0, 0.0, follow_gap, 0.0, 0.0, "no_lead")
  if not lead_kinematics_valid:
    return _result(AlignmentAction.IGNORE, 0.0, 0.0, follow_gap, 0.0, 0.0, "invalid_kinematics")
  required_fields = (
    v_ego, a_ego, v_cruise, lead_d_rel, lead_v, lead_v_rel, lead_a_k,
    follow_gap, lead_confidence,
  )
  if not all(_finite(v) for v in required_fields):
    return _result(AlignmentAction.IGNORE, 0.0, 0.0, follow_gap, 0.0, 0.0, "nonfinite")
  if brake_pressed or gas_pressed:
    return _result(AlignmentAction.IGNORE, 0.0, 0.0, follow_gap, 0.0, 0.0, "driver_override")
  if model_should_stop:
    return _result(AlignmentAction.IGNORE, 0.0, 0.0, follow_gap, 0.0, 0.0, "model_stop")
  if force_slow_decel:
    return _result(AlignmentAction.IGNORE, 0.0, 0.0, follow_gap, 0.0, 0.0, "force_slow")
  if lead_shadow_active or alternate_threat_active:
    return _result(AlignmentAction.IGNORE, 0.0, 0.0, follow_gap, 0.0, 0.0, "threat")

  desired_gap = max(follow_gap, 1.5 * max(0.0, v_ego))
  excess_gap = lead_d_rel - desired_gap
  closing = max(0.0, -lead_v_rel)
  if excess_gap > 0.1 and closing > 0.05:
    required_decel = (closing * closing) / (2.0 * excess_gap)
  else:
    required_decel = 0.0

  # Compute TTC and THW for behavioral gating.
  ttc = lead_d_rel / max(closing, 0.01) if closing > 0.05 else float('inf')
  thw = lead_d_rel / max(v_ego, 1.0)

  # True hazard zone (< 4s TTC): defer to MPC regardless of lead confidence.
  if ttc < _ALIGN_HAZARD_TTC:
    return _result(AlignmentAction.IGNORE, 0.0, required_decel, desired_gap,
                   excess_gap, closing, "ttc_hazard")

  # Slowdown side: we are closing on a slower lead.
  ego_faster = v_ego > lead_v + 0.1
  if ego_faster and v_ego >= _ALIGN_MIN_V_EGO and lead_v >= _ALIGN_MIN_LEAD_V:
    if lead_stable and lead_confidence >= _ALIGN_MIN_CONFIDENCE and excess_gap >= _ALIGN_MIN_EXCESS_GAP:
      # Very far / comfortable: don't block progress with advisory.
      if ttc > _ALIGN_NO_ADVISORY_TTC and thw >= 1.5:
        return _result(AlignmentAction.IGNORE, 0.0, required_decel, desired_gap,
                       excess_gap, closing, "ttc_far_comfort")
      # Decel-band routing with TTC gating.
      if required_decel <= _ALIGN_TINY_REQUIRED_DECEL:
        if ttc < _ALIGN_COAST_TTC_MAIN:
          return _result(AlignmentAction.COAST, _ALIGN_COAST_A_TARGET, required_decel,
                         desired_gap, excess_gap, closing, "tiny_decel_coast")
        return _result(AlignmentAction.IGNORE, 0.0, required_decel, desired_gap,
                       excess_gap, closing, "tiny_decel_ttc_far")
      if required_decel <= _ALIGN_COMFORT_REQUIRED_DECEL:
        t = (required_decel - _ALIGN_TINY_REQUIRED_DECEL) / (
          _ALIGN_COMFORT_REQUIRED_DECEL - _ALIGN_TINY_REQUIRED_DECEL
        )
        a_target = (1.0 - t) * _ALIGN_COAST_A_TARGET + t * _ALIGN_GENTLE_BRAKE_MAX
        return _result(AlignmentAction.GENTLE_BRAKE, a_target, required_decel,
                       desired_gap, excess_gap, closing, "comfort_gentle_brake")
      # reqDecel in (COMFORT, MAX): capped advisory instead of silence.
      # This removes the "nothing ... nothing ... full MPC wall" behavior.
      if required_decel < _ALIGN_MAX_REQUIRED_DECEL:
        return _result(AlignmentAction.GENTLE_BRAKE, _ALIGN_GENTLE_BRAKE_MAX, required_decel,
                       desired_gap, excess_gap, closing, "capped_advisory_brake")
      # reqDecel >= MAX (0.80): use TTC/speed/THW to distinguish true hazard from
      # high-speed light-brake cases. Corrected aEgo analysis: at >=18 m/s humans
      # brake 83% of the time but 71% is light braking only.
      # High-speed light-brake zone: moderate TTC with comfortable headway
      # → capped gentle advisory instead of MPC wall.
      if v_ego >= 18.0 and ttc >= _ALIGN_STRONG_PREP_TTC and thw >= 1.2:
        return _result(AlignmentAction.GENTLE_BRAKE, _ALIGN_GENTLE_BRAKE_MAX, required_decel,
                       desired_gap, excess_gap, closing, "high_decel_high_speed_mild")
      # TTC alone is not enough to authorize a capped advisory: mid-speed real-log
      # cases can have TTC >= 8s while consuming the usable follow-gap buffer in
      # ~1-2s. Leave those high-required-decel cases to MPC/lead physics.
      return _result(AlignmentAction.IGNORE, 0.0, required_decel, desired_gap,
                     excess_gap, closing, "high_required_decel")
    return _result(AlignmentAction.IGNORE, 0.0, required_decel, desired_gap,
                   excess_gap, closing, "unstable_low_confidence_or_excess")

  # Pullaway / launch side: lead is opening and moving.
  wants_progress = v_cruise > v_ego + _ALIGN_PROGRESS_CRUISE_MARGIN
  opening = lead_v_rel > _ALIGN_MIN_OPENING
  lead_moving = lead_v > _ALIGN_MIN_V_LEAD_PULLAWAY
  far_enough = lead_d_rel > follow_gap + _ALIGN_NEAR_FOLLOW_MARGIN

  can_pullaway = bool(
    wants_progress and lead_progress_allowed and lead_stable and
    lead_moving and opening and far_enough
  )
  can_standstill_launch = bool(
    v_ego < _ALIGN_STANDSTILL_V_EGO and wants_progress and lead_progress_allowed and
    lead_stable and lead_moving and opening
  )
  if not can_pullaway and not can_standstill_launch:
    return _result(AlignmentAction.IGNORE, 0.0, required_decel, desired_gap,
                   excess_gap, closing, "no_progress_evidence")

  proposed = min(launch_accel_max(personality), max(0.0, lead_v - v_ego) / LEAD_LAUNCH_TAU)
  a_target = lead_speedup_guard(v_ego, lead_v, lead_d_rel, follow_gap, proposed)
  if a_target <= 0.0:
    return _result(AlignmentAction.IGNORE, 0.0, required_decel, desired_gap,
                   excess_gap, closing, "guard_blocked_accel")

  if can_standstill_launch:
    return _result(AlignmentAction.STANDSTILL_LAUNCH, a_target, required_decel,
                   desired_gap, excess_gap, closing, "standstill_launch")

  return _result(AlignmentAction.PULLAWAY, a_target, required_decel, desired_gap,
                 excess_gap, closing, "guarded_pullaway")
