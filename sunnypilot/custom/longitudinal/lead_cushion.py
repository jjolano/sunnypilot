"""Lead-following cushion + lead-aware speedup guard (Phase 5 backlog).

Two refinements to lead-follow comfort, building on the coast horizon:

- lead_following_cushion: approaching a slower *moving* lead at normal speed should coast and
  taper before braking, not brake early. Uses the coast horizon with the lead's speed as the
  constraint target over the closeable runway (gap above the steady follow gap), so we lift
  off and bleed to the lead's speed instead of braking. (legacy spec 2026-05-03)
- lead_speedup_guard: cap lead-pullaway progress by the *actual* gap and the decel that would
  be required if the lead stopped pulling away — not by gap-excess prediction alone — so a
  speed-up never digs a hole we can only climb out of with hard braking. (legacy spec 2026-05-05)

Pure/deterministic; feel-validated against engaged data downstream.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from openpilot.sunnypilot.custom.longitudinal.coast_horizon import (
  CoastAction,
  CoastHorizonInputs,
  coast_horizon,
)

# steady follow gap ~ a time-gap; speedup guard keeps the post-speedup required decel comfortable
SPEEDUP_GUARD_MAX_REQUIRED_DECEL = -1.2
LEAD_MOVING_MIN_SPEED = 0.5
CUSHION_MIN_CLOSING = 0.2  # m/s; below this the lead isn't meaningfully slower


@dataclass(frozen=True)
class CushionResult:
  action: CoastAction
  a_target: float       # 0 while coasting; required (negative) accel while braking
  coast_first: bool     # True when coasting/tapering instead of braking now


def lead_following_cushion(v_ego: float, v_lead: float, d_rel: float, follow_gap: float,
                           coast_decel: float, comfort_brake_decel: float = -1.2) -> CushionResult:
  """Coast-first approach to a slower moving lead. Returns COAST (a_target 0) while runway
  allows bleeding to the lead's speed, else the required braking accel."""
  v_ego = max(0.0, float(v_ego))
  v_lead = max(0.0, float(v_lead))
  closing = v_ego - v_lead
  runway = float(d_rel) - float(follow_gap)
  # Not a slower moving lead, or already inside the follow gap -> no cushion (defer to MPC).
  if v_lead < LEAD_MOVING_MIN_SPEED or closing < CUSHION_MIN_CLOSING or runway <= 0.0:
    return CushionResult(CoastAction.BRAKE if closing > 0 else CoastAction.CRUISE, 0.0, False)

  r = coast_horizon(CoastHorizonInputs(
    v_ego=v_ego, v_target=v_lead, distance_to_constraint=runway,
    accel_coast=coast_decel, comfort_brake_decel=comfort_brake_decel,
  ))
  if r.action is CoastAction.BRAKE:
    return CushionResult(CoastAction.BRAKE, r.recommended_accel, False)
  # CRUISE (too early -> 0, keep cruising) or COAST (lift window -> gentle coast decel)
  return CushionResult(r.action, r.recommended_accel, True)


def lead_speedup_guard(v_ego: float, v_lead: float, d_rel: float, follow_gap: float,
                       proposed_accel: float, dt_lookahead: float = 1.0,
                       max_required_decel: float = SPEEDUP_GUARD_MAX_REQUIRED_DECEL) -> float:
  """Cap a lead-pullaway speed-up so that, if the lead immediately stopped pulling away, the
  decel required to re-settle to the follow gap stays within comfort. Returns the capped accel
  (<= proposed)."""
  proposed_accel = float(proposed_accel)
  if proposed_accel <= 0.0:
    return proposed_accel
  v_ego = max(0.0, float(v_ego))
  v_lead = max(0.0, float(v_lead))
  excess_gap = float(d_rel) - float(follow_gap)
  if excess_gap <= 0.0:
    return 0.0  # no room to speed up at all

  # Project ego speed if we apply proposed_accel for the lookahead; the lead holds v_lead.
  v_ego_next = v_ego + proposed_accel * float(dt_lookahead)
  closing_next = v_ego_next - v_lead
  if closing_next <= 0.0:
    return proposed_accel  # still not overtaking the lead's speed
  # Required decel to bleed that closing back to zero over the remaining excess gap.
  required = -(closing_next * closing_next) / (2.0 * max(excess_gap, 1e-3))
  if required >= max_required_decel:
    return proposed_accel  # comfortable -> allow
  # Bisection-free closed form: the max closing s.t. required == max_required_decel.
  max_closing = math.sqrt(max(0.0, -2.0 * max_required_decel * excess_gap))
  allowed_v_ego_next = v_lead + max_closing
  allowed_accel = max(0.0, (allowed_v_ego_next - v_ego) / float(dt_lookahead))
  return min(proposed_accel, allowed_accel)
