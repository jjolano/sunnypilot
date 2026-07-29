"""Torque v2.1 unified output governor — first cut (property-gated, NOT feel-certified).

Replaces the legacy four-module output stage (over-response attenuator -> guarded response
assist -> conservative output shaper -> refined output governor) with a single pass over
one observation struct, structured as three operations with one reason bitfield:

  AUGMENT     raise output toward the unclipped command to overcome under-response/lag
              (the low-speed "under-response floor", expressed as a relaxation factor).
  RESTRICT    cap output for safety/comfort (over-response, high steering rate,
              same-direction actuator limit, sign conflict, ISO lateral-accel, override
              release). cap is a fraction of max_output; the binding cap is the min.
  RATE-LIMIT  bound the rate of change (actuator-aware build, sign-change unwind, high-rate
              slew scaling, same-sign release backstop). Driver release and safety cuts
              (sign conflict / over-response / ISO) bypass the release backstop and drop
              instantly.

Application order per tick: floor (augment) -> cap+clip (restrict) -> target-arrival blend
-> slew (rate-limit). The floor may relax caps but never the final actuator slew.

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

import bisect
import math
from dataclasses import dataclass, field
from enum import IntFlag

from openpilot.sunnypilot.custom.lateral._output_governor_constants import *


# --- pure-Python scalar helper kernels ---
def sign(value: float) -> float:
  return 1.0 if value > 0.0 else (-1.0 if value < 0.0 else 0.0)


def approach(value: float, target: float, step: float) -> float:
  if target > value:
    return min(target, value + step)
  return max(target, value - step)


def _finite(*values: float) -> bool:
  try:
    return all(math.isfinite(float(v)) for v in values)
  except (TypeError, ValueError):
    return False


def _interp(x: float, xp: list[float], fp: list[float]) -> float:
  if x <= xp[0]:
    return float(fp[0])
  if x >= xp[-1]:
    return float(fp[-1])

  idx = bisect.bisect_right(xp, x) - 1
  if idx < 0:
    idx = 0

  x0 = float(xp[idx])
  x1 = float(xp[idx + 1])
  denom = x1 - x0
  if denom == 0.0:
    return float(fp[idx])

  y0 = float(fp[idx])
  y1 = float(fp[idx + 1])
  return y0 + (float(x) - x0) * (y1 - y0) / denom


def _clip(value: float, lower: float, upper: float) -> float:
  return max(lower, min(value, upper))


def _over_response_scale(inp: OutputGovernorInputs) -> float:
  desired_sign = sign(inp.desired_lateral_accel)
  actual_sign = sign(inp.actual_lateral_accel)
  torque_sign = sign(inp.nominal_torque)
  if desired_sign == 0.0 or actual_sign != desired_sign or torque_sign != actual_sign:
    return 1.0
  over_response = desired_sign * (inp.actual_lateral_accel - inp.desired_lateral_accel)
  if over_response <= OVER_RESPONSE_MARGIN:
    return 1.0
  span = OVER_RESPONSE_FULL_EXCESS - OVER_RESPONSE_MARGIN
  ratio = _clip((over_response - OVER_RESPONSE_MARGIN) / max(span, 1e-3), 0.0, 1.0)
  return 1.0 + ratio * (OVER_RESPONSE_MIN_SCALE - 1.0)


def _sign_conflict(inp: OutputGovernorInputs) -> bool:
  desired_sign = sign(inp.desired_lateral_accel)
  actual_sign = sign(inp.actual_lateral_accel)
  return (desired_sign != 0.0 and actual_sign != 0.0 and desired_sign != actual_sign
          and abs(inp.actual_lateral_accel) > SIGN_THRESHOLD)


def _iso_cap(inp: OutputGovernorInputs) -> float:
  actual_abs = abs(inp.actual_lateral_accel)
  output_reinforces_actual = (sign(inp.nominal_torque) != 0.0 and
                              sign(inp.nominal_torque) == sign(inp.actual_lateral_accel))
  if not output_reinforces_actual or actual_abs <= ISO_ACCEL_MARGIN:
    return 1.0
  return OVER_ISO_ACCEL_CAP if actual_abs > ISO_LATERAL_ACCEL else NEAR_ISO_ACCEL_CAP


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
  return _clip((UNDER_RESPONSE_FADE_SPEED - inp.v_ego) / max(span, 1e-3), 0.0, 1.0)


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


def _steering_rate_comfort_blend(inp: OutputGovernorInputs) -> float:
  torque_sign = sign(inp.nominal_torque)
  steering_rate_sign = sign(inp.steering_rate_deg)
  if torque_sign == 0.0 or torque_sign != steering_rate_sign:
    return 0.0
  if inp.release_active or _tracking_correction_needed(inp):
    return 0.0
  return _clip((abs(inp.steering_rate_deg) - STEERING_RATE_COMFORT_START_DEG) /
               max(STEERING_RATE_COMFORT_FULL_DEG - STEERING_RATE_COMFORT_START_DEG, 1e-3), 0.0, 1.0)


class _PythonHelperSet:
  __slots__ = ()
  sign = staticmethod(sign)
  approach = staticmethod(approach)
  finite = staticmethod(_finite)
  interp = staticmethod(_interp)
  clip = staticmethod(_clip)
  over_response_scale = staticmethod(_over_response_scale)
  sign_conflict = staticmethod(_sign_conflict)
  iso_cap = staticmethod(_iso_cap)
  under_response_floor = staticmethod(_under_response_floor)
  tracking_correction_needed = staticmethod(_tracking_correction_needed)
  steering_rate_comfort_blend = staticmethod(_steering_rate_comfort_blend)


# --- optional Cython scalar kernels ---
_CYTHON_AVAILABLE = False
_CythonHelperSet = None

try:
  from openpilot.sunnypilot.custom.lateral.output_governor_pyx import (
    approach as _cy_approach,
    _clip as _cy_clip,
    _finite as _cy_finite,
    _interp as _cy_interp,
    _iso_cap as _cy_iso_cap,
    _over_response_scale as _cy_over_response_scale,
    sign as _cy_sign,
    _sign_conflict as _cy_sign_conflict,
    _steering_rate_comfort_blend as _cy_steering_rate_comfort_blend,
    _tracking_correction_needed as _cy_tracking_correction_needed,
    _under_response_floor as _cy_under_response_floor,
  )

  class _CythonHelperSet:
    __slots__ = ()
    sign = staticmethod(_cy_sign)
    approach = staticmethod(_cy_approach)
    finite = staticmethod(_cy_finite)
    interp = staticmethod(_cy_interp)
    clip = staticmethod(_cy_clip)
    over_response_scale = staticmethod(_cy_over_response_scale)
    sign_conflict = staticmethod(_cy_sign_conflict)
    iso_cap = staticmethod(_cy_iso_cap)
    under_response_floor = staticmethod(_cy_under_response_floor)
    tracking_correction_needed = staticmethod(_cy_tracking_correction_needed)
    steering_rate_comfort_blend = staticmethod(_cy_steering_rate_comfort_blend)

  _CYTHON_AVAILABLE = True
except ImportError:
  pass


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
  TARGET_ARRIVAL = 1 << 13
  # Telemetry-only marker: OR'd into the logged reason by torque_v2_1 while the
  # LateralSlewScaleMode apply scale is live. Never set by the governor itself, so
  # `active` and reason semantics stay identical across conditions.
  SLEW_SCALE_APPLIED = 1 << 14


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
  lateral_accel_error_rate: float = 0.0
  lat_delay: float = 0.0
  holding_torque: float | None = None


@dataclass(frozen=True)
class OutputGovernorDiagnostics:
  signConflictActive: bool = False        # sign-conflict condition is true
  signConflictBinding: bool = False       # sign-conflict cap would tighten before later caps/slew
  signConflictFloorGuarded: bool = False  # sign-conflict was one guard for the under-response floor

  # Shadow-only under-response floor guard diagnostics. No output changes.
  underResponseGuardPathEvidenceInvalid: bool = False
  underResponseGuardControllerUnstable: bool = False
  underResponseGuardRelease: bool = False
  underResponseGuardSameDirectionLimit: bool = False
  underResponseGuardHighSteeringRate: bool = False
  underResponseGuardSignConflict: bool = False
  underResponseGuardOverResponse: bool = False
  underResponseGuardIsoAccel: bool = False
  underResponseGuardTorqueFraction: bool = False


@dataclass(frozen=True)
class OutputGovernorResult:
  output_torque: float
  active: bool
  reason: int
  cap: float    # effective RESTRICT cap as a fraction of max_output (post floor relax)
  floor: float  # AUGMENT under-response floor in [0, 1]
  diagnostics: OutputGovernorDiagnostics = field(default_factory=OutputGovernorDiagnostics)


class OutputGovernor:
  def __init__(self, dt: float, slew_rate_scale: float = 1.0, _use_cython: bool = True):
    self.dt = max(float(dt), 1e-3)
    # Scales the BUILD slew only. Sign-change, release, caps, fast releases, and the
    # same-direction steer-limited rate cap are deliberately unscaled: scaling
    # sign/release sharpened catch-down steps on-road (2026-07-20, routes 2ba/2bc).
    self.slew_rate_scale = float(slew_rate_scale)
    self.previous_output = 0.0
    self._use_cython = _use_cython and _CYTHON_AVAILABLE
    self._helper_set = _CythonHelperSet if self._use_cython else _PythonHelperSet

  def reset(self) -> None:
    self.previous_output = 0.0

  def update(self, inp: OutputGovernorInputs) -> OutputGovernorResult:
    h = self._helper_set
    reason = GovernorReason.NONE

    if not inp.active:
      self.reset()
      return OutputGovernorResult(0.0, False, int(reason), 1.0, 0.0)

    if not h.finite(inp.v_ego, inp.steering_rate_deg, inp.nominal_torque, inp.max_output,
                    inp.desired_lateral_accel, inp.actual_lateral_accel) or inp.max_output <= 0.0:
      self.reset()
      return OutputGovernorResult(0.0, True, int(GovernorReason.INVALID), 1.0, 0.0)

    iso_cap = h.iso_cap(inp)

    # --- AUGMENT ---
    floor = h.under_response_floor(inp)
    initial_floor = floor
    sign_conflict_active = h.sign_conflict(inp)
    over_scale = h.over_response_scale(inp)

    path_evidence_invalid = not inp.path_evidence_valid
    controller_unstable = not inp.controller_evidence_stable
    release_guard = inp.release_active
    same_direction_guard = inp.same_direction_limit
    high_steering_rate_guard = abs(inp.steering_rate_deg) >= HIGH_RATE_START_DEG
    sign_conflict_guard = sign_conflict_active
    over_response_guard = over_scale < 1.0
    iso_accel_guard = iso_cap < 1.0
    torque_fraction_guard = abs(inp.nominal_torque) >= UNDER_RESPONSE_MAX_TORQUE_FRACTION * inp.max_output

    floor_guarded = floor > 0.0 and (
      path_evidence_invalid or
      controller_unstable or
      release_guard or
      same_direction_guard or
      high_steering_rate_guard or
      sign_conflict_guard or
      over_response_guard or
      iso_accel_guard or
      torque_fraction_guard
    )
    if floor > 0.0:
      if floor_guarded:
        reason |= GovernorReason.UNDER_RESPONSE_GUARDED
        floor = 0.0
      else:
        reason |= GovernorReason.UNDER_RESPONSE_FLOOR

    # --- RESTRICT: build cap as a fraction of max_output (binding cap = min) ---
    cap = 1.0
    comfort_blend = h.steering_rate_comfort_blend(inp)
    if comfort_blend > 0.0:
      cap = min(cap, 1.0 + comfort_blend * (STEERING_RATE_COMFORT_MIN_CAP - 1.0))
      reason |= GovernorReason.STEERING_RATE_COMFORT
    high_rate_blend = h.clip((abs(inp.steering_rate_deg) - HIGH_RATE_START_DEG) /
                             max(HIGH_RATE_FULL_DEG - HIGH_RATE_START_DEG, 1e-3), 0.0, 1.0)
    if high_rate_blend > 0.0:
      cap = min(cap, 1.0 + high_rate_blend * (HIGH_RATE_MIN_CAP - 1.0))
      reason |= GovernorReason.HIGH_STEERING_RATE
    if inp.same_direction_limit:
      cap = min(cap, SAME_DIRECTION_LIMIT_CAP)
      reason |= GovernorReason.SAME_DIRECTION_LIMIT
    if over_scale < 1.0:
      cap = min(cap, over_scale)
      reason |= GovernorReason.OVER_RESPONSE
    cap_without_sign_conflict = cap
    if sign_conflict_active:
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
    capped = h.clip(inp.nominal_torque, -output_cap_abs, output_cap_abs)

    attenuated = inp.nominal_torque
    if over_scale < 1.0:
      attenuated = inp.nominal_torque * over_scale
    clipped = capped if abs(capped) <= abs(attenuated) else attenuated

    if abs(clipped - inp.nominal_torque) > 1e-6:
      reason |= GovernorReason.CLIPPED

    # Manual steering treats every current wheel angle as a local origin: it starts
    # decisively, peaks near mid-stroke, then transfers into nonzero holding torque.
    # Predicted time-to-target gives the same relative behavior without an angle-state
    # machine, and the guards ensure this comfort blend can only reduce a clean command.
    error = inp.desired_lateral_accel - inp.actual_lateral_accel
    error_sign = h.sign(error)
    arrival_inputs_valid = inp.holding_torque is not None and h.finite(
      inp.lateral_accel_error_rate, inp.lat_delay, inp.holding_torque,
    )
    arrival_guarded = (floor > 0.0 or path_evidence_invalid or controller_unstable or release_guard or
                       same_direction_guard or high_steering_rate_guard or sign_conflict_guard or
                       over_response_guard or iso_accel_guard)
    closing_rate = -error_sign * float(inp.lateral_accel_error_rate) if arrival_inputs_valid else 0.0
    if arrival_inputs_valid and not arrival_guarded and error_sign != 0.0 and closing_rate > TARGET_ARRIVAL_MIN_CLOSING_RATE:
      predicted_remaining = max(0.0, error_sign * (error + inp.lateral_accel_error_rate * max(inp.lat_delay, 0.0)))
      time_to_target = predicted_remaining / closing_rate
      blend_x = h.clip((TARGET_ARRIVAL_TAPER_START - time_to_target) /
                       (TARGET_ARRIVAL_TAPER_START - TARGET_ARRIVAL_TAPER_FULL), 0.0, 1.0)
      blend = blend_x * blend_x * (3.0 - 2.0 * blend_x)
      holding = h.clip(float(inp.holding_torque), -abs(clipped), abs(clipped))
      if blend > 0.0 and h.sign(holding) in (0.0, h.sign(clipped)) and abs(holding) < abs(clipped):
        clipped += blend * (holding - clipped)
        reason |= GovernorReason.TARGET_ARRIVAL

    cap_eff_without_sign_conflict = cap_without_sign_conflict + floor * (1.0 - cap_without_sign_conflict)
    abs_nominal = abs(inp.nominal_torque)
    floor_context = initial_floor > 0.0
    diagnostics = OutputGovernorDiagnostics(
      signConflictActive=bool(sign_conflict_active),
      signConflictBinding=bool(
        sign_conflict_active and
        cap < cap_without_sign_conflict and
        (abs_nominal > cap_eff_without_sign_conflict * inp.max_output or
         abs_nominal > SIGN_CONFLICT_CAP * inp.max_output)
      ),
      signConflictFloorGuarded=bool(floor_context and sign_conflict_active and floor_guarded),
      underResponseGuardPathEvidenceInvalid=bool(floor_context and path_evidence_invalid),
      underResponseGuardControllerUnstable=bool(floor_context and controller_unstable),
      underResponseGuardRelease=bool(floor_context and release_guard),
      underResponseGuardSameDirectionLimit=bool(floor_context and same_direction_guard),
      underResponseGuardHighSteeringRate=bool(floor_context and high_steering_rate_guard),
      underResponseGuardSignConflict=bool(floor_context and sign_conflict_guard),
      underResponseGuardOverResponse=bool(floor_context and over_response_guard),
      underResponseGuardIsoAccel=bool(floor_context and iso_accel_guard),
      underResponseGuardTorqueFraction=bool(floor_context and torque_fraction_guard),
    )

    # --- RATE-LIMIT ---
    previous_sign = h.sign(self.previous_output)
    target_sign = h.sign(clipped)
    sign_change = previous_sign != 0.0 and target_sign != 0.0 and previous_sign != target_sign
    if sign_change:
      slew_rate = h.interp(inp.v_ego, SIGN_CHANGE_SLEW_RATE_BP, SIGN_CHANGE_SLEW_RATE_V)
      reason |= GovernorReason.SIGN_CHANGE_LIMITED
    else:
      slew_rate = h.interp(inp.v_ego, OUTPUT_SLEW_RATE_BP, OUTPUT_SLEW_RATE_V) * self.slew_rate_scale
    if comfort_blend > 0.0:
      slew_rate *= 1.0 + comfort_blend * (STEERING_RATE_COMFORT_MIN_SLEW_SCALE - 1.0)
    if high_rate_blend > 0.0:
      slew_rate *= HIGH_RATE_SLEW_SCALE
    if inp.same_direction_limit:
      slew_rate = min(slew_rate, h.interp(inp.v_ego, SAME_DIRECTION_LIMIT_RATE_BP, SAME_DIRECTION_LIMIT_RATE_V))

    slew_target = 0.0 if sign_change else clipped
    target_decreases_same_direction = (previous_sign != 0.0 and target_sign in (0.0, previous_sign)
                                       and abs(clipped) <= abs(self.previous_output))
    if target_decreases_same_direction:
      fast_release = inp.release_active or sign_conflict_active or over_scale < 1.0 or iso_cap < 1.0
      if fast_release:
        limited = clipped
      else:
        # release backstop: speed-scheduled only — comfort/high-rate slew scalings must
        # never slow a yield toward zero
        release_rate = h.interp(inp.v_ego, OUTPUT_SLEW_RATE_BP, OUTPUT_SLEW_RATE_V) * RELEASE_SLEW_SCALE
        limited = h.approach(self.previous_output, clipped, release_rate * self.dt)
    else:
      limited = h.approach(self.previous_output, slew_target, slew_rate * self.dt)
    output = limited
    if abs(output - clipped) > 1e-6:
      reason |= GovernorReason.SLEW_LIMITED

    self.previous_output = output
    active = abs(output - inp.nominal_torque) > 1e-6 or reason != GovernorReason.NONE
    return OutputGovernorResult(output, active, int(reason), cap_eff, floor, diagnostics)
