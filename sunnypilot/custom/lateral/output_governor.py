"""Torque v2.1 unified output governor — first cut (property-gated, NOT feel-certified).

Replaces the legacy four-module output stage (over-response attenuator -> guarded response
assist -> conservative output shaper -> refined output governor) with a single pass over
one observation struct, structured as three operations with one reason bitfield:

  AUGMENT     raise output toward the unclipped command to overcome under-response/lag
              (the low-speed "under-response floor", expressed as a relaxation factor).
  RESTRICT    cap output for safety/comfort (over-response, high steering rate,
              same-direction actuator limit, sign conflict, ISO lateral-accel, override
              release). cap is a fraction of max_output; the binding cap is the min.
  RATE-LIMIT  bound the rate of change (speed-scheduled slew, sign-change slew, high-rate
              slew scaling), with AUGMENT permitted to relax the slew toward the command.

Application order per tick: floor (augment) -> cap+clip (restrict) -> slew (rate-limit),
with the floor relaxing both the cap and the slew toward the unclipped command only for
clean same-sign lag cases.

SCOPE — this is a structural first cut. It carries v2.1's namesake refined-governor
behaviors plus the principal hard caps (over-response from the attenuator/shaper, sign
conflict, ISO accel, override release). It deliberately DEFERS, pending engaged-route
replay tuning (see docs/adr/2026-06-13-clean-room-torque-v2-1-architecture.md):
  - guarded response assist's additive assist/bias learning and curve-exit/preposition
    boosts (the under-response floor covers the principal catch-up here);
  - the conservative shaper's comfort sub-caps (steering-rate comfort, actuator-lag
    comfort, stale-actuator reversal, safety-limited ramp/sign-hold, low-speed steer-limited,
    bump) and its output-rate recovery-time trackers.
These are feel-tuning behaviors that can only be validated against engaged data, which does
not yet exist. Property tests below gate the governor's INVARIANTS, not its feel; default
promotion to TorqueControlTune=2.1 waits for engaged-route parity.

Constants are the legacy values (docs/legacy/tuned-constants.yaml); do not retune here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntFlag

import numpy as np

# --- RATE-LIMIT: slew schedules (refined governor) ---
OUTPUT_SLEW_RATE_BP = [0.0, 5.0, 10.0, 20.0, 30.0, 40.0]
OUTPUT_SLEW_RATE_V = [1.40, 2.00, 3.00, 4.20, 5.00, 5.60]
SIGN_CHANGE_SLEW_RATE_BP = [0.0, 5.0, 10.0, 20.0, 30.0, 40.0]
SIGN_CHANGE_SLEW_RATE_V = [0.90, 1.20, 1.80, 2.40, 3.00, 3.40]
SAME_DIRECTION_LIMIT_RATE_BP = [0.0, 10.0, 20.0, 30.0, 40.0]
SAME_DIRECTION_LIMIT_RATE_V = [1.30, 1.30, 2.10, 3.20, 3.60]

# --- RESTRICT: caps ---
SAME_DIRECTION_LIMIT_CAP = 0.85
STEERING_RATE_COMFORT_START_DEG = 25.0
STEERING_RATE_COMFORT_FULL_DEG = 80.0
STEERING_RATE_COMFORT_MIN_CAP = 0.88
STEERING_RATE_COMFORT_MIN_SLEW_SCALE = 0.75
HIGH_RATE_START_DEG = 80.0
HIGH_RATE_FULL_DEG = 100.0
HIGH_RATE_MIN_CAP = 0.62
HIGH_RATE_SLEW_SCALE = 0.70
SIGN_CONFLICT_CAP = 0.80
OVERRIDE_RELEASE_CAP = 0.80
OVER_RESPONSE_MARGIN = 0.12
OVER_RESPONSE_FULL_EXCESS = 0.60
OVER_RESPONSE_MIN_SCALE = 0.30
ISO_LATERAL_ACCEL = 3.0
ISO_ACCEL_MARGIN = 2.6
NEAR_ISO_ACCEL_CAP = 0.85
OVER_ISO_ACCEL_CAP = 0.80
UNDER_RESPONSE_MAX_TORQUE_FRACTION = 0.90
TRACKING_CORRECTION_MARGIN = 0.06

# nuPlan comfort bounds (84th %ile of 1,282 hours expert human driving, arXiv:2403.04133).
# These are reference thresholds — the ISO-derived caps above remain the active limits.
# Future: consider replacing ISO_LATERAL_ACCEL with NUPLAN_COMFORT_LAT_ACCEL after
# engaged-route validation.
NUPLAN_COMFORT_LAT_ACCEL = 4.89    # m/s² — lateral acceleration comfort bound
NUPLAN_COMFORT_JERK = 8.37         # m/s³ — jerk vector magnitude comfort bound

# --- AUGMENT: under-response floor ---
UNDER_RESPONSE_MARGIN = 0.12
UNDER_RESPONSE_FULL_SPEED = 9.0
UNDER_RESPONSE_FADE_SPEED = 12.0

SIGN_THRESHOLD = 0.05


def sign(value: float) -> float:
  return 1.0 if value > 0.0 else (-1.0 if value < 0.0 else 0.0)


def approach(value: float, target: float, step: float) -> float:
  if target > value:
    return min(target, value + step)
  return max(target, value - step)


class GovernorReason(IntFlag):
  NONE = 0
  CLIPPED = 1 << 0
  SLEW_LIMITED = 1 << 1
  SIGN_CHANGE_LIMITED = 1 << 2
  SAME_DIRECTION_LIMIT = 1 << 3
  HIGH_STEERING_RATE = 1 << 4
  OVER_RESPONSE = 1 << 5
  SIGN_CONFLICT = 1 << 6
  NEAR_ISO_ACCEL = 1 << 7
  OVERRIDE_RELEASE = 1 << 8
  UNDER_RESPONSE_FLOOR = 1 << 9
  INVALID = 1 << 10
  UNDER_RESPONSE_GUARDED = 1 << 11
  STEERING_RATE_COMFORT = 1 << 12


@dataclass(frozen=True)
class OutputGovernorInputs:
  active: bool
  v_ego: float
  steering_rate_deg: float
  nominal_torque: float
  max_output: float
  desired_lateral_accel: float
  actual_lateral_accel: float
  same_direction_limit: bool  # steer-limited in the command direction and not unwinding
  release_active: bool        # driver override / unwind release
  path_evidence_valid: bool = True
  controller_evidence_stable: bool = True


@dataclass(frozen=True)
class OutputGovernorResult:
  output_torque: float
  active: bool
  reason: int
  cap: float    # effective RESTRICT cap as a fraction of max_output (post floor relax)
  floor: float  # AUGMENT under-response floor in [0, 1]


class OutputGovernor:
  def __init__(self, dt: float):
    self.dt = max(float(dt), 1e-3)
    self.previous_output = 0.0

  def reset(self) -> None:
    self.previous_output = 0.0

  def update(self, inp: OutputGovernorInputs) -> OutputGovernorResult:
    reason = GovernorReason.NONE
    if not inp.active:
      self.reset()
      return OutputGovernorResult(0.0, False, int(reason), 1.0, 0.0)
    if not self._finite(inp.v_ego, inp.steering_rate_deg, inp.nominal_torque, inp.max_output,
                        inp.desired_lateral_accel, inp.actual_lateral_accel) or inp.max_output <= 0.0:
      self.reset()
      return OutputGovernorResult(0.0, True, int(GovernorReason.INVALID), 1.0, 0.0)

    iso_cap = self._iso_cap(inp)

    # --- AUGMENT ---
    floor = self._under_response_floor(inp)
    floor_guarded = floor > 0.0 and (
      not inp.path_evidence_valid or
      not inp.controller_evidence_stable or
      inp.release_active or
      inp.same_direction_limit or
      abs(inp.steering_rate_deg) >= HIGH_RATE_START_DEG or
      self._sign_conflict(inp) or
      self._over_response_scale(inp) < 1.0 or
      iso_cap < 1.0 or
      abs(inp.nominal_torque) >= UNDER_RESPONSE_MAX_TORQUE_FRACTION * inp.max_output
    )
    if floor > 0.0:
      if floor_guarded:
        reason |= GovernorReason.UNDER_RESPONSE_GUARDED
        floor = 0.0
      else:
        reason |= GovernorReason.UNDER_RESPONSE_FLOOR

    # --- RESTRICT: build cap as a fraction of max_output (binding cap = min) ---
    cap = 1.0
    comfort_blend = self._steering_rate_comfort_blend(inp)
    if comfort_blend > 0.0:
      cap = min(cap, 1.0 + comfort_blend * (STEERING_RATE_COMFORT_MIN_CAP - 1.0))
      reason |= GovernorReason.STEERING_RATE_COMFORT
    high_rate_blend = float(np.clip((abs(inp.steering_rate_deg) - HIGH_RATE_START_DEG) /
                                    max(HIGH_RATE_FULL_DEG - HIGH_RATE_START_DEG, 1e-3), 0.0, 1.0))
    if high_rate_blend > 0.0:
      cap = min(cap, 1.0 + high_rate_blend * (HIGH_RATE_MIN_CAP - 1.0))
      reason |= GovernorReason.HIGH_STEERING_RATE
    if inp.same_direction_limit:
      cap = min(cap, SAME_DIRECTION_LIMIT_CAP)
      reason |= GovernorReason.SAME_DIRECTION_LIMIT
    over_scale = self._over_response_scale(inp)
    if over_scale < 1.0:
      cap = min(cap, over_scale)
      reason |= GovernorReason.OVER_RESPONSE
    if self._sign_conflict(inp):
      cap = min(cap, SIGN_CONFLICT_CAP)
      reason |= GovernorReason.SIGN_CONFLICT
    if iso_cap < 1.0:
      cap = min(cap, iso_cap)
      reason |= GovernorReason.NEAR_ISO_ACCEL
    if inp.release_active:
      cap = min(cap, OVERRIDE_RELEASE_CAP)
      reason |= GovernorReason.OVERRIDE_RELEASE

    # floor relaxes the cap toward 1.0 (never tightens)
    cap_eff = cap + floor * (1.0 - cap)
    output_cap_abs = cap_eff * inp.max_output
    clipped = float(np.clip(inp.nominal_torque, -output_cap_abs, output_cap_abs))
    if abs(clipped - inp.nominal_torque) > 1e-6:
      reason |= GovernorReason.CLIPPED

    # --- RATE-LIMIT ---
    previous_sign = sign(self.previous_output)
    target_sign = sign(clipped)
    sign_change = previous_sign != 0.0 and target_sign != 0.0 and previous_sign != target_sign
    if sign_change:
      slew_rate = float(np.interp(inp.v_ego, SIGN_CHANGE_SLEW_RATE_BP, SIGN_CHANGE_SLEW_RATE_V))
      reason |= GovernorReason.SIGN_CHANGE_LIMITED
    else:
      slew_rate = float(np.interp(inp.v_ego, OUTPUT_SLEW_RATE_BP, OUTPUT_SLEW_RATE_V))
    if comfort_blend > 0.0:
      slew_rate *= 1.0 + comfort_blend * (STEERING_RATE_COMFORT_MIN_SLEW_SCALE - 1.0)
    if high_rate_blend > 0.0:
      slew_rate *= HIGH_RATE_SLEW_SCALE
    if inp.same_direction_limit:
      slew_rate = min(slew_rate, float(np.interp(inp.v_ego, SAME_DIRECTION_LIMIT_RATE_BP, SAME_DIRECTION_LIMIT_RATE_V)))

    # allow immediate reduction toward zero in the same direction (no upward rate gate)
    target_decreases_same_direction = (previous_sign != 0.0 and target_sign == previous_sign
                                       and abs(clipped) <= abs(self.previous_output))
    limited = clipped if target_decreases_same_direction else approach(self.previous_output, clipped, slew_rate * self.dt)
    output = limited + floor * (clipped - limited)  # floor relaxes slew toward the command
    if abs(output - clipped) > 1e-6:
      reason |= GovernorReason.SLEW_LIMITED

    self.previous_output = output
    active = abs(output - inp.nominal_torque) > 1e-6 or reason != GovernorReason.NONE
    return OutputGovernorResult(output, active, int(reason), cap_eff, floor)

  @staticmethod
  def _finite(*values: float) -> bool:
    try:
      return all(math.isfinite(float(v)) for v in values)
    except (TypeError, ValueError):
      return False

  @staticmethod
  def _over_response_scale(inp: OutputGovernorInputs) -> float:
    # Same-direction over-response: command reinforces an actual accel already exceeding the
    # target. Scale 1.0 -> OVER_RESPONSE_MIN_SCALE as excess grows MARGIN -> FULL_EXCESS.
    desired_sign = sign(inp.desired_lateral_accel)
    actual_sign = sign(inp.actual_lateral_accel)
    torque_sign = sign(inp.nominal_torque)
    if desired_sign == 0.0 or actual_sign != desired_sign or torque_sign != actual_sign:
      return 1.0
    over_response = desired_sign * (inp.actual_lateral_accel - inp.desired_lateral_accel)
    if over_response <= OVER_RESPONSE_MARGIN:
      return 1.0
    span = OVER_RESPONSE_FULL_EXCESS - OVER_RESPONSE_MARGIN
    ratio = float(np.clip((over_response - OVER_RESPONSE_MARGIN) / max(span, 1e-3), 0.0, 1.0))
    return 1.0 + ratio * (OVER_RESPONSE_MIN_SCALE - 1.0)

  @staticmethod
  def _sign_conflict(inp: OutputGovernorInputs) -> bool:
    desired_sign = sign(inp.desired_lateral_accel)
    actual_sign = sign(inp.actual_lateral_accel)
    return (desired_sign != 0.0 and actual_sign != 0.0 and desired_sign != actual_sign
            and abs(inp.actual_lateral_accel) > SIGN_THRESHOLD)

  @staticmethod
  def _iso_cap(inp: OutputGovernorInputs) -> float:
    # Cap when the command reinforces a high actual lateral accel (comfort/ISO limit).
    actual_abs = abs(inp.actual_lateral_accel)
    output_reinforces_actual = sign(inp.nominal_torque) != 0.0 and sign(inp.nominal_torque) == sign(inp.actual_lateral_accel)
    if not output_reinforces_actual or actual_abs <= ISO_ACCEL_MARGIN:
      return 1.0
    return OVER_ISO_ACCEL_CAP if actual_abs > ISO_LATERAL_ACCEL else NEAR_ISO_ACCEL_CAP

  @staticmethod
  def _under_response_floor(inp: OutputGovernorInputs) -> float:
    if inp.v_ego >= UNDER_RESPONSE_FADE_SPEED:
      return 0.0
    desired_sign = sign(inp.desired_lateral_accel)
    actual_sign = 0.0 if abs(inp.actual_lateral_accel) <= SIGN_THRESHOLD else sign(inp.actual_lateral_accel)
    output_sign = sign(inp.nominal_torque)
    under_response = desired_sign * (inp.desired_lateral_accel - inp.actual_lateral_accel)
    same_sign_lag = desired_sign != 0.0 and output_sign == desired_sign and actual_sign in (0.0, desired_sign)
    corrective_reversal = (desired_sign != 0.0 and actual_sign != 0.0 and actual_sign != desired_sign
                           and output_sign == desired_sign)
    if under_response <= UNDER_RESPONSE_MARGIN or not (same_sign_lag or corrective_reversal):
      return 0.0
    if inp.v_ego <= UNDER_RESPONSE_FULL_SPEED:
      return 1.0
    span = UNDER_RESPONSE_FADE_SPEED - UNDER_RESPONSE_FULL_SPEED
    return float(np.clip((UNDER_RESPONSE_FADE_SPEED - inp.v_ego) / max(span, 1e-3), 0.0, 1.0))

  @staticmethod
  def _tracking_correction_needed(inp: OutputGovernorInputs) -> bool:
    output_sign = sign(inp.nominal_torque)
    lateral_accel_error = inp.desired_lateral_accel - inp.actual_lateral_accel
    if abs(lateral_accel_error) > TRACKING_CORRECTION_MARGIN and output_sign == sign(lateral_accel_error):
      return True

    desired_sign = sign(inp.desired_lateral_accel)
    actual_sign = 0.0 if abs(inp.actual_lateral_accel) <= SIGN_THRESHOLD else sign(inp.actual_lateral_accel)
    under_response = desired_sign * (inp.desired_lateral_accel - inp.actual_lateral_accel)
    same_sign_lag = desired_sign != 0.0 and output_sign == desired_sign and actual_sign in (0.0, desired_sign)
    corrective_reversal = (desired_sign != 0.0 and actual_sign != 0.0 and actual_sign != desired_sign
                           and output_sign == desired_sign)
    return under_response > UNDER_RESPONSE_MARGIN and (same_sign_lag or corrective_reversal)

  @classmethod
  def _steering_rate_comfort_blend(cls, inp: OutputGovernorInputs) -> float:
    # Comfort shaping is deliberately one-sided: trim torque that reinforces already-fast
    # steering wheel motion, but do not interfere with tracking catch-up, driver-release
    # unwind, or torque that opposes/stabilizes the current wheel motion.
    torque_sign = sign(inp.nominal_torque)
    steering_rate_sign = sign(inp.steering_rate_deg)
    if torque_sign == 0.0 or torque_sign != steering_rate_sign:
      return 0.0
    if inp.release_active or cls._tracking_correction_needed(inp):
      return 0.0
    return float(np.clip((abs(inp.steering_rate_deg) - STEERING_RATE_COMFORT_START_DEG) /
                         max(STEERING_RATE_COMFORT_FULL_DEG - STEERING_RATE_COMFORT_START_DEG, 1e-3), 0.0, 1.0))
