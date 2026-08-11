"""Invariant (property) tests for the unified output governor first cut.

These gate the governor's INVARIANTS — safety bound, reset, floor-relaxes-cap-only,
cap monotonicity, slew bounding, sign-change rate, passthrough. They do NOT certify feel;
that requires engaged-route replay against legacy v2.1 (see the ADR).
"""
from __future__ import annotations

import numpy as np
import pytest
from typing import cast

from openpilot.sunnypilot.custom.lateral.output_governor import (
  HIGH_RATE_SLEW_SCALE,
  OUTPUT_SLEW_RATE_BP,
  OUTPUT_SLEW_RATE_V,
  OVER_RESPONSE_FULL_EXCESS,
  OVER_RESPONSE_MARGIN,
  OVER_RESPONSE_MIN_SCALE,
  OVER_TURN_FADE_SPEED,
  OVER_TURN_MARGIN,
  OVER_TURN_MAX_OPPOSITE_FRAC,
  OVER_TURN_MAX_SPEED,
  OVER_TURN_RAMP_EXCESS,
  OVER_TURN_REVERSAL_FRAC,
  RELEASE_SLEW_SCALE,
  SIGN_CHANGE_SLEW_RATE_BP,
  SIGN_CHANGE_SLEW_RATE_V,
  SLEW_RATE_SCALE_STEP,
  STEERING_RATE_COMFORT_MIN_CAP,
  STEERING_RATE_COMFORT_MIN_SLEW_SCALE,
  GovernorReason,
  OutputGovernor,
  OutputGovernorInputs,
)

DT = 0.01
MAX = 1.0


def benign(nominal=0.0, v=20.0, rate=0.0, desired=0.0, actual=0.0,
           same_dir=False, release=False, active=True, path_valid=True, controller_stable=True,
           error_rate=0.0, lat_delay=0.0, holding=None):
  """An input with no cap/floor triggers unless overridden."""
  return OutputGovernorInputs(active=active, v_ego=v, steering_rate_deg=rate,
                              nominal_torque=nominal, max_output=MAX,
                              desired_lateral_accel=desired, actual_lateral_accel=actual,
                              same_direction_limit=same_dir, release_active=release,
                              path_evidence_valid=path_valid,
                              controller_evidence_stable=controller_stable,
                              lateral_accel_error_rate=error_rate, lat_delay=lat_delay,
                              holding_torque=holding)


def test_output_never_exceeds_max_output():
  gov = OutputGovernor(DT)
  rng = np.random.default_rng(20260613)
  for _ in range(5000):
    inp = OutputGovernorInputs(
      active=bool(rng.random() > 0.05),
      v_ego=float(rng.uniform(0.0, 40.0)),
      steering_rate_deg=float(rng.uniform(-150.0, 150.0)),
      nominal_torque=float(rng.uniform(-2.0, 2.0)),   # can exceed max_output; must be clipped
      max_output=MAX,
      desired_lateral_accel=float(rng.uniform(-4.0, 4.0)),
      actual_lateral_accel=float(rng.uniform(-4.0, 4.0)),
      same_direction_limit=bool(rng.random() > 0.7),
      release_active=bool(rng.random() > 0.8),
    )
    r = gov.update(inp)
    assert abs(r.output_torque) <= MAX + 1e-9
    assert 0.0 <= r.floor <= 1.0
    assert r.cap <= 1.0 + 1e-9


def test_inactive_resets_and_zero_output():
  gov = OutputGovernor(DT)
  for _ in range(20):
    gov.update(benign(nominal=0.8, v=20.0))
  assert gov.previous_output != 0.0
  r = gov.update(benign(nominal=0.8, active=False))
  assert r.output_torque == 0.0
  assert r.active is False
  assert gov.previous_output == 0.0


def test_invalid_inputs_flagged_and_safe():
  gov = OutputGovernor(DT)
  gov.update(benign(nominal=0.5))
  r = gov.update(benign(nominal=float("nan")))
  assert r.output_torque == 0.0
  assert r.reason & GovernorReason.INVALID
  assert gov.previous_output == 0.0


def test_uncastable_invalid_inputs_flagged_and_safe():
  gov = OutputGovernor(DT)
  gov.update(benign(nominal=0.5))
  bad = cast(OutputGovernorInputs, benign(nominal=None))

  r = gov.update(bad)

  assert r.output_torque == 0.0
  assert r.reason & GovernorReason.INVALID
  assert gov.previous_output == 0.0


def test_slew_bounds_rate_of_change():
  gov = OutputGovernor(DT)
  v = 20.0
  slew = float(np.interp(v, OUTPUT_SLEW_RATE_BP, OUTPUT_SLEW_RATE_V))
  prev = 0.0
  for _ in range(200):
    r = gov.update(benign(nominal=MAX, v=v))  # command pinned high, no caps/floor
    assert abs(r.output_torque - prev) <= slew * DT + 1e-9
    prev = r.output_torque
  assert prev == pytest.approx(MAX, abs=1e-6)  # eventually reaches the command


def test_actuator_slew_matches_toyota_raw_limits():
  build = float(np.interp(20.0, OUTPUT_SLEW_RATE_BP, OUTPUT_SLEW_RATE_V))
  release = build * RELEASE_SLEW_SCALE

  assert build * DT * 1500 == pytest.approx(12.0)
  assert release * DT * 1500 == pytest.approx(18.75)


def test_scaled_actuator_slew_stays_under_toyota_raw_limits():
  build = OUTPUT_SLEW_RATE_V[0] * SLEW_RATE_SCALE_STEP

  assert build * DT * 1500 == pytest.approx(13.5)
  assert build * DT * 1500 <= 15.0          # Toyota STEER_DELTA_UP


def test_slew_scale_default_is_identity():
  base = OutputGovernor(DT)
  explicit = OutputGovernor(DT, slew_rate_scale=1.0)
  rng = np.random.default_rng(20260719)
  for _ in range(500):
    inp = benign(nominal=float(rng.uniform(-1.5, 1.5)), v=float(rng.uniform(0.0, 40.0)),
                 desired=float(rng.uniform(-3.0, 3.0)), actual=float(rng.uniform(-3.0, 3.0)))
    assert base.update(inp).output_torque == explicit.update(inp).output_torque


def test_slew_scale_step_scales_build_only():
  build = OutputGovernor(DT, slew_rate_scale=SLEW_RATE_SCALE_STEP)
  r_build = build.update(benign(nominal=MAX))
  assert r_build.output_torque == pytest.approx(OUTPUT_SLEW_RATE_V[0] * SLEW_RATE_SCALE_STEP * DT)

  # sign-change and release stay at baseline rates: scaling them sharpened
  # catch-down steps on-road (2026-07-20)
  sign = OutputGovernor(DT, slew_rate_scale=SLEW_RATE_SCALE_STEP)
  sign.previous_output = 0.5
  r_sign = sign.update(benign(nominal=-0.5))
  assert r_sign.output_torque == pytest.approx(0.5 - SIGN_CHANGE_SLEW_RATE_V[0] * DT)

  release = OutputGovernor(DT, slew_rate_scale=SLEW_RATE_SCALE_STEP)
  release.previous_output = 0.5
  r_release = release.update(benign(nominal=0.1))
  assert r_release.output_torque == pytest.approx(0.5 - OUTPUT_SLEW_RATE_V[0] * RELEASE_SLEW_SCALE * DT)


def test_governor_never_sets_slew_scale_marker():
  gov = OutputGovernor(DT, slew_rate_scale=SLEW_RATE_SCALE_STEP)
  rng = np.random.default_rng(20260719)
  for _ in range(2000):
    inp = OutputGovernorInputs(
      active=bool(rng.random() > 0.05),
      v_ego=float(rng.uniform(0.0, 40.0)),
      steering_rate_deg=float(rng.uniform(-150.0, 150.0)),
      nominal_torque=float(rng.uniform(-2.0, 2.0)),
      max_output=MAX,
      desired_lateral_accel=float(rng.uniform(-4.0, 4.0)),
      actual_lateral_accel=float(rng.uniform(-4.0, 4.0)),
      same_direction_limit=bool(rng.random() > 0.7),
      release_active=bool(rng.random() > 0.8),
    )
    r = gov.update(inp)
    assert not (r.reason & GovernorReason.SLEW_SCALE_APPLIED)
    assert abs(r.output_torque) <= MAX + 1e-9


def test_under_response_floor_cannot_bypass_final_slew():
  r = OutputGovernor(DT).update(benign(nominal=0.89, v=8.0, desired=2.0, actual=0.5))
  build = float(np.interp(8.0, OUTPUT_SLEW_RATE_BP, OUTPUT_SLEW_RATE_V))

  assert r.floor == 1.0
  assert r.output_torque == pytest.approx(build * DT)


@pytest.mark.parametrize("direction", [1.0, -1.0])
def test_target_arrival_blends_toward_holding_torque(direction):
  arriving = OutputGovernor(DT)
  arriving.previous_output = direction * 0.4
  r_arriving = arriving.update(benign(nominal=direction * 0.6, desired=direction * 1.2, actual=direction,
                                      error_rate=direction * -1.0, lat_delay=0.1, holding=direction * 0.2))

  far = OutputGovernor(DT)
  far.previous_output = direction * 0.4
  r_far = far.update(benign(nominal=direction * 0.6, desired=direction * 1.5, actual=direction,
                            error_rate=direction * -0.5, lat_delay=0.1, holding=direction * 0.2))

  assert r_arriving.reason & GovernorReason.TARGET_ARRIVAL
  assert abs(r_arriving.output_torque) < 0.4 < abs(r_far.output_torque)
  assert not (r_far.reason & GovernorReason.TARGET_ARRIVAL)


def test_sign_change_unwinds_to_zero_before_opposite_build():
  gov = OutputGovernor(DT)
  gov.previous_output = 0.01

  zero = gov.update(benign(nominal=-0.5))
  opposite = gov.update(benign(nominal=-0.5))

  assert zero.output_torque == 0.0
  assert opposite.output_torque == pytest.approx(-OUTPUT_SLEW_RATE_V[0] * DT)


def test_floor_allows_clean_low_speed_catchup_and_fades_at_speed():
  gov_lo = OutputGovernor(DT)
  gov_hi = OutputGovernor(DT)
  # Under-response at low speed gets the floor; at higher speed it fades out.
  floored = benign(nominal=0.89, v=8.0, desired=2.0, actual=0.5)
  unfloored = benign(nominal=0.89, v=15.0, desired=2.0, actual=0.5)  # v>=12 -> floor 0
  r_lo = gov_lo.update(floored)
  r_hi = gov_hi.update(unfloored)
  assert r_lo.floor == 1.0
  assert r_hi.floor == 0.0
  assert r_lo.reason & GovernorReason.UNDER_RESPONSE_FLOOR
  assert not (r_hi.reason & GovernorReason.UNDER_RESPONSE_FLOOR)


@pytest.mark.parametrize("kwargs", [
  {"same_dir": True},
  {"release": True},
  {"rate": 80.0},
  {"desired": 1.0, "actual": -0.2},
  {"path_valid": False},
  {"controller_stable": False},
  {"nominal": 0.90},
])
def test_floor_guarded_for_unstable_evidence(kwargs):
  gov = OutputGovernor(DT)
  base = dict(nominal=0.89, v=8.0, desired=2.0, actual=0.5)
  base.update(kwargs)  # type: ignore[arg-type]
  r = gov.update(benign(**base))  # type: ignore[arg-type]
  assert r.reason & GovernorReason.UNDER_RESPONSE_GUARDED
  assert not (r.reason & GovernorReason.UNDER_RESPONSE_FLOOR)
  assert r.floor == 0.0


def test_iso_near_limit_guards_floor_and_caps_output():
  gov = OutputGovernor(DT)
  r = gov.update(benign(nominal=0.89, v=8.0, desired=3.0, actual=2.7))
  assert r.reason & GovernorReason.NEAR_ISO_ACCEL
  assert r.reason & GovernorReason.UNDER_RESPONSE_GUARDED
  assert not (r.reason & GovernorReason.UNDER_RESPONSE_FLOOR)
  assert r.floor == 0.0
  assert r.cap <= 0.85 + 1e-9


def test_clean_same_sign_lag_still_gets_floor():
  gov = OutputGovernor(DT)
  r = gov.update(benign(nominal=0.89, v=8.0, desired=2.0, actual=0.5))
  assert r.reason & GovernorReason.UNDER_RESPONSE_FLOOR
  assert not (r.reason & GovernorReason.UNDER_RESPONSE_GUARDED)
  assert r.floor > 0.0


def test_high_rate_boundary_guards_floor_at_threshold_only():
  below = OutputGovernor(DT).update(benign(nominal=0.89, v=8.0, rate=79.9, desired=2.0, actual=0.5))
  at = OutputGovernor(DT).update(benign(nominal=0.89, v=8.0, rate=80.0, desired=2.0, actual=0.5))
  assert below.reason & GovernorReason.UNDER_RESPONSE_FLOOR
  assert not (below.reason & GovernorReason.UNDER_RESPONSE_GUARDED)
  assert at.reason & GovernorReason.UNDER_RESPONSE_GUARDED
  assert not (at.reason & GovernorReason.UNDER_RESPONSE_FLOOR)


def test_over_response_does_not_trigger_under_response_floor():
  r = OutputGovernor(DT).update(benign(nominal=0.5, v=8.0, desired=1.0, actual=1.3))
  assert not (r.reason & GovernorReason.UNDER_RESPONSE_FLOOR)
  assert not (r.reason & GovernorReason.UNDER_RESPONSE_GUARDED)


def test_over_response_cap_monotonic():
  # Increasing same-direction over-response must not increase the cap (more excess -> tighter).
  caps = []
  for actual in [1.0, 1.12, 1.25, 1.4, 1.6, 1.8]:
    gov = OutputGovernor(DT)
    r = gov.update(benign(nominal=0.5, v=20.0, desired=1.0, actual=actual))  # v=20 -> no floor
    caps.append(r.cap)
  for a, b in zip(caps, caps[1:], strict=False):
    assert b <= a + 1e-9
  assert caps[-1] == pytest.approx(OVER_RESPONSE_MIN_SCALE, abs=1e-6)  # saturates at min scale


def test_moderate_over_response_attenuates_before_large_error():
  gov = OutputGovernor(DT)
  gov.previous_output = 0.5

  r = gov.update(benign(nominal=0.5, v=20.0, desired=1.0, actual=1.2))

  assert OVER_RESPONSE_MARGIN < 0.12
  assert OVER_RESPONSE_FULL_EXCESS < 0.60
  assert r.reason & GovernorReason.OVER_RESPONSE
  assert r.reason & GovernorReason.CLIPPED
  # The attenuation engages at moderate excess (this is what the test gates). It is
  # reached through the release backstop rather than in one frame -- see
  # test_over_response_yields_through_release_backstop.
  assert r.cap < 1.0
  for _ in range(200):
    r = gov.update(benign(nominal=0.5, v=20.0, desired=1.0, actual=1.2))
  assert r.output_torque < 0.4


def test_over_response_cap_triggers_in_both_directions():
  # Same-direction over-response caps the command regardless of sign.
  r_pos = OutputGovernor(DT).update(benign(nominal=0.5, v=25.0, desired=1.0, actual=1.8))
  assert r_pos.reason & GovernorReason.OVER_RESPONSE

  r_neg = OutputGovernor(DT).update(benign(nominal=-0.5, v=25.0, desired=-1.0, actual=-1.8))
  assert r_neg.reason & GovernorReason.OVER_RESPONSE


def test_over_response_attenuates_non_binding_nominal_torque():
  # Route logs showed OVER_RESPONSE caps often did not bind because nominal torque was
  # below cap*max_output. Same-direction overresponse should still reduce the command.
  gov = OutputGovernor(DT)
  gov.previous_output = 0.5

  r = gov.update(benign(nominal=0.5, v=20.0, desired=1.0, actual=1.3))

  assert r.reason & GovernorReason.OVER_RESPONSE
  assert r.reason & GovernorReason.CLIPPED
  assert r.cap > 0.5  # max-output cap alone would not have clipped nominal=0.5
  assert 0.0 < r.output_torque < 0.5


def test_over_response_attenuation_preserves_command_sign():
  gov = OutputGovernor(DT)
  gov.previous_output = -0.5

  r = gov.update(benign(nominal=-0.5, v=20.0, desired=-1.0, actual=-1.3))

  assert r.reason & GovernorReason.OVER_RESPONSE
  assert -0.5 < r.output_torque < 0.0


def test_over_response_skipped_when_torque_opposes_actual():
  # A same-direction excess is only dangerous when the command reinforces it.
  r = OutputGovernor(DT).update(benign(nominal=-0.5, v=25.0, desired=1.0, actual=1.8))
  assert not (r.reason & GovernorReason.OVER_RESPONSE)

  r = OutputGovernor(DT).update(benign(nominal=0.5, v=25.0, desired=-1.0, actual=-1.8))
  assert not (r.reason & GovernorReason.OVER_RESPONSE)


def test_over_turn_cap_binds_opposite_torque_low_speed():
  # City-corner over-turn (route 00000302 signature): demand lags the physical corner,
  # the car over-turns in the desired direction, and the command pushes OPPOSITE to the
  # turn. The opposite push is capped at OVER_TURN_MAX_OPPOSITE_FRAC of max_output.
  gov = OutputGovernor(DT)
  gov.previous_output = -0.5
  r = gov.update(benign(nominal=-0.5, v=8.0, desired=1.0, actual=1.3))

  assert r.reason & GovernorReason.OVER_TURN
  assert r.reason & GovernorReason.CLIPPED
  assert r.cap == pytest.approx(OVER_TURN_MAX_OPPOSITE_FRAC, abs=1e-9)

  # multi-frame: the output settles at the cap, never beyond it
  for _ in range(100):
    r = gov.update(benign(nominal=-0.5, v=8.0, desired=1.0, actual=1.3))
  assert r.output_torque == pytest.approx(-OVER_TURN_MAX_OPPOSITE_FRAC * MAX, abs=1e-6)


def test_over_turn_cap_binds_in_both_directions():
  r = OutputGovernor(DT).update(benign(nominal=0.5, v=8.0, desired=-1.0, actual=-1.3))
  assert r.reason & GovernorReason.OVER_TURN

  r = OutputGovernor(DT).update(benign(nominal=-0.5, v=8.0, desired=1.0, actual=1.3))
  assert r.reason & GovernorReason.OVER_TURN


def test_over_turn_cap_fades_out_by_speed():
  # Full cap below OVER_TURN_MAX_SPEED, linear fade to none by OVER_TURN_FADE_SPEED,
  # and no cap at speed (existing same-sign guards own the high-speed regime).
  r_full = OutputGovernor(DT).update(benign(nominal=-0.5, v=OVER_TURN_MAX_SPEED, desired=1.0, actual=1.3))
  assert r_full.cap == pytest.approx(OVER_TURN_MAX_OPPOSITE_FRAC, abs=1e-9)

  mid = (OVER_TURN_MAX_SPEED + OVER_TURN_FADE_SPEED) / 2.0
  r_mid = OutputGovernor(DT).update(benign(nominal=-0.5, v=mid, desired=1.0, actual=1.3))
  frac = 1.0 + ((OVER_TURN_FADE_SPEED - mid) / (OVER_TURN_FADE_SPEED - OVER_TURN_MAX_SPEED)) * (OVER_TURN_MAX_OPPOSITE_FRAC - 1.0)
  assert r_mid.cap == pytest.approx(frac, abs=1e-9)

  r_high = OutputGovernor(DT).update(benign(nominal=-0.5, v=OVER_TURN_FADE_SPEED, desired=1.0, actual=1.3))
  assert not (r_high.reason & GovernorReason.OVER_TURN)
  assert r_high.cap == 1.0


def test_over_turn_cap_applies_to_sign_conflict_reversal():
  # Direction reversal (route 00000302 S-transitions): the car still turns the old way
  # while the demand has flipped. This is a sign conflict, so the under-response floor
  # is guarded off and the over-turn cap bounds the corrective push -- the uncapped
  # reversal is exactly what whipped the wheel past center (t=901: +0.40 push, wheel
  # +31 deg -> -9 deg). The bounded push still unwinds the car, just without the whip.
  gov = OutputGovernor(DT)
  gov.previous_output = 0.5
  r = gov.update(benign(nominal=0.5, v=8.0, desired=0.02, actual=-0.3))

  assert r.reason & GovernorReason.SIGN_CONFLICT
  assert r.reason & GovernorReason.OVER_TURN
  assert r.floor == 0.0  # sign-conflict guard suppresses the floor
  assert r.cap == pytest.approx(OVER_TURN_MAX_OPPOSITE_FRAC, abs=1e-9)

  for _ in range(100):
    r = gov.update(benign(nominal=0.5, v=8.0, desired=0.02, actual=-0.3))
  assert r.output_torque == pytest.approx(OVER_TURN_MAX_OPPOSITE_FRAC * MAX, abs=1e-6)


def test_over_turn_cap_not_applied_same_sign_torque():
  # Same-sign over-response stays the province of the existing OVER_RESPONSE cap.
  r = OutputGovernor(DT).update(benign(nominal=0.5, v=8.0, desired=1.0, actual=1.3))
  assert r.reason & GovernorReason.OVER_RESPONSE
  assert not (r.reason & GovernorReason.OVER_TURN)


def test_over_turn_cap_not_applied_without_excess():
  # Under-turning in the actual's direction (or near-zero actual) is not an over-turn.
  r = OutputGovernor(DT).update(benign(nominal=-0.5, v=8.0, desired=1.0, actual=0.9))
  assert not (r.reason & GovernorReason.OVER_TURN)

  r = OutputGovernor(DT).update(benign(nominal=-0.5, v=8.0, desired=1.0, actual=0.02))
  assert not (r.reason & GovernorReason.OVER_TURN)


def test_over_turn_cap_ramps_to_ceiling_with_excess():
  # The cap ramps continuously from 1.0 at the margin to the strict ceiling -- no flat
  # step -- and is monotonic non-increasing with excess.
  caps = []
  for actual in [1.09, 1.12, 1.2, 1.4, 1.9]:
    caps.append(OutputGovernor(DT).update(benign(nominal=-0.5, v=8.0, desired=1.0, actual=actual)).cap)
  for a, b in zip(caps, caps[1:], strict=False):
    assert b <= a + 1e-9
  assert caps[-1] == pytest.approx(OVER_TURN_MAX_OPPOSITE_FRAC, abs=1e-6)  # saturates at strict ceiling
  assert caps[0] < 1.0  # ramping already just past the margin


def test_over_turn_cap_relaxes_to_reversal_ceiling_while_excess_opens():
  # Curve exit / S-curve: desired opposes actual AND the excess is opening (error rate
  # pulling actual away from desired) -> the unwind must not be starved; 0.30 ceiling.
  r = OutputGovernor(DT).update(benign(nominal=-0.5, v=8.0, desired=-0.5, actual=0.5, error_rate=-1.0))
  assert r.reason & GovernorReason.OVER_TURN
  assert r.cap == pytest.approx(OVER_TURN_REVERSAL_FRAC, abs=1e-9)

  r2 = OutputGovernor(DT).update(benign(nominal=0.5, v=8.0, desired=0.5, actual=-0.5, error_rate=1.0))
  assert r2.cap == pytest.approx(OVER_TURN_REVERSAL_FRAC, abs=1e-9)


def test_over_turn_cap_stays_strict_when_excess_closes():
  # The corrective response is already closing the excess -> strict ceiling even in a
  # reversal (the correction is working; no relaxation needed).
  r = OutputGovernor(DT).update(benign(nominal=-0.5, v=8.0, desired=-0.5, actual=0.5, error_rate=1.0))
  assert r.reason & GovernorReason.OVER_TURN
  assert r.cap == pytest.approx(OVER_TURN_MAX_OPPOSITE_FRAC, abs=1e-9)


def test_over_turn_cap_stays_strict_same_sign_desired():
  # In-curve over-turn (desired still in the actual's direction): strict ceiling even
  # while the excess opens -- the 302 whip regime stays capped at 0.10.
  r = OutputGovernor(DT).update(benign(nominal=-0.5, v=8.0, desired=1.0, actual=1.5, error_rate=-1.0))
  assert r.reason & GovernorReason.OVER_TURN
  assert r.cap == pytest.approx(OVER_TURN_MAX_OPPOSITE_FRAC, abs=1e-9)


def test_over_turn_cap_does_not_grow_with_nominal():
  # The opposite push is bounded absolutely; a bigger raw reversal gets the same cap.
  gov = OutputGovernor(DT)
  gov.previous_output = -1.0
  r = gov.update(benign(nominal=-1.0, v=8.0, desired=1.0, actual=1.9))
  assert r.reason & GovernorReason.OVER_TURN
  assert r.cap == pytest.approx(OVER_TURN_MAX_OPPOSITE_FRAC, abs=1e-9)


def test_sign_conflict_detected_both_directions():
  # Opposite signs of desired and actual lateral accel are a safety conflict.
  r = OutputGovernor(DT).update(benign(nominal=0.5, v=25.0, desired=1.0, actual=-0.5))
  assert r.reason & GovernorReason.SIGN_CONFLICT
  assert r.diagnostics.signConflictActive is True

  r = OutputGovernor(DT).update(benign(nominal=-0.5, v=25.0, desired=-1.0, actual=0.5))
  assert r.reason & GovernorReason.SIGN_CONFLICT
  assert r.diagnostics.signConflictActive is True


def test_sign_conflict_diagnostics_distinguish_active_from_binding():
  active_only = OutputGovernor(DT).update(benign(nominal=0.2, v=25.0, desired=1.0, actual=-0.5))
  binding = OutputGovernor(DT).update(benign(nominal=0.95, v=25.0, desired=1.0, actual=-0.5))

  assert active_only.diagnostics.signConflictActive is True
  assert active_only.diagnostics.signConflictBinding is False
  assert binding.diagnostics.signConflictActive is True
  assert binding.diagnostics.signConflictBinding is True


_UNDER_RESPONSE_GUARD_FIELDS = (
  "underResponseGuardPathEvidenceInvalid",
  "underResponseGuardControllerUnstable",
  "underResponseGuardRelease",
  "underResponseGuardSameDirectionLimit",
  "underResponseGuardHighSteeringRate",
  "underResponseGuardSignConflict",
  "underResponseGuardOverResponse",
  "underResponseGuardIsoAccel",
  "underResponseGuardTorqueFraction",
)


def _all_under_response_guards_false(diag):
  for field in _UNDER_RESPONSE_GUARD_FIELDS:
    assert getattr(diag, field) is False, field


def test_under_response_guard_source_diagnostics_all_false_for_clean_floor():
  r = OutputGovernor(DT).update(benign(nominal=0.89, v=8.0, desired=2.0, actual=0.5))
  assert r.floor > 0.0
  assert r.reason & GovernorReason.UNDER_RESPONSE_FLOOR
  _all_under_response_guards_false(r.diagnostics)


@pytest.mark.parametrize("kwargs, expected_field", [
  ({"path_valid": False}, "underResponseGuardPathEvidenceInvalid"),
  ({"controller_stable": False}, "underResponseGuardControllerUnstable"),
  ({"release": True}, "underResponseGuardRelease"),
  ({"same_dir": True}, "underResponseGuardSameDirectionLimit"),
  ({"rate": 80.0}, "underResponseGuardHighSteeringRate"),
  ({"nominal": 0.90}, "underResponseGuardTorqueFraction"),
])
def test_under_response_guard_source_diagnostics_individual_guard(kwargs, expected_field):
  base = dict(nominal=0.89, v=8.0, desired=2.0, actual=0.5)
  base.update(kwargs)
  r = OutputGovernor(DT).update(benign(**base))
  assert r.floor == 0.0
  assert r.reason & GovernorReason.UNDER_RESPONSE_GUARDED
  for field in _UNDER_RESPONSE_GUARD_FIELDS:
    if field == expected_field:
      assert getattr(r.diagnostics, field) is True
    else:
      assert getattr(r.diagnostics, field) is False, field


def test_under_response_guard_source_diagnostics_sign_conflict_and_high_rate():
  r = OutputGovernor(DT).update(benign(nominal=0.89, v=8.0, rate=80.0, desired=1.0, actual=-0.2))
  assert r.floor == 0.0
  assert r.reason & GovernorReason.UNDER_RESPONSE_GUARDED
  assert r.reason & GovernorReason.SIGN_CONFLICT
  assert r.diagnostics.underResponseGuardSignConflict is True
  assert r.diagnostics.underResponseGuardHighSteeringRate is True
  for field in _UNDER_RESPONSE_GUARD_FIELDS:
    if field not in ("underResponseGuardSignConflict", "underResponseGuardHighSteeringRate"):
      assert getattr(r.diagnostics, field) is False, field


def test_under_response_guard_source_diagnostics_iso_accel():
  r = OutputGovernor(DT).update(benign(nominal=0.89, v=8.0, desired=3.5, actual=2.7))
  assert r.floor == 0.0
  assert r.reason & GovernorReason.UNDER_RESPONSE_GUARDED
  assert r.reason & GovernorReason.NEAR_ISO_ACCEL
  assert r.diagnostics.underResponseGuardIsoAccel is True
  for field in _UNDER_RESPONSE_GUARD_FIELDS:
    if field != "underResponseGuardIsoAccel":
      assert getattr(r.diagnostics, field) is False, field


def test_sign_conflict_diagnostics_report_floor_guard():
  r = OutputGovernor(DT).update(benign(nominal=0.89, v=8.0, desired=1.0, actual=-0.2))

  assert r.reason & GovernorReason.SIGN_CONFLICT
  assert r.reason & GovernorReason.UNDER_RESPONSE_GUARDED
  assert r.diagnostics.signConflictFloorGuarded is True


@pytest.mark.parametrize("desired,actual", [
  (1.0, 0.5),   # same sign
  (-1.0, -0.5), # same sign
  (0.0, -0.5),  # desired is zero
  (1.0, 0.0),   # actual is zero
  (1.0, -0.03), # actual too small to count as opposite sign
  (-1.0, 0.03), # actual too small to count as opposite sign
])
def test_no_sign_conflict_for_same_sign_or_zero(desired, actual):
  r = OutputGovernor(DT).update(benign(nominal=0.5, v=25.0, desired=desired, actual=actual))
  assert not (r.reason & GovernorReason.SIGN_CONFLICT)
  assert r.diagnostics.signConflictActive is False


def test_sign_change_uses_slower_slew():
  gov = OutputGovernor(DT)
  v = 20.0
  for _ in range(300):
    gov.update(benign(nominal=MAX, v=v))   # ramp up to +MAX
  assert gov.previous_output == pytest.approx(MAX, abs=1e-6)
  prev = gov.previous_output
  r = gov.update(benign(nominal=-MAX, v=v))  # command flips negative
  sign_change_slew = float(np.interp(v, SIGN_CHANGE_SLEW_RATE_BP, SIGN_CHANGE_SLEW_RATE_V))
  assert r.reason & GovernorReason.SIGN_CHANGE_LIMITED
  assert abs(r.output_torque - prev) <= sign_change_slew * DT + 1e-9


def test_high_steering_rate_caps_and_scales_slew():
  gov = OutputGovernor(DT)
  v = 20.0
  r = gov.update(benign(nominal=MAX, v=v, rate=120.0))  # >= HIGH_RATE_FULL_DEG -> full blend
  assert r.reason & GovernorReason.HIGH_STEERING_RATE
  assert r.cap < 1.0
  # first step from 0 with scaled slew rate
  scaled_slew = float(np.interp(v, OUTPUT_SLEW_RATE_BP, OUTPUT_SLEW_RATE_V)) * HIGH_RATE_SLEW_SCALE
  assert abs(r.output_torque) <= scaled_slew * DT + 1e-9


def test_steering_rate_comfort_softly_caps_reinforcing_torque():
  gov = OutputGovernor(DT)
  v = 20.0
  r = gov.update(benign(nominal=MAX, v=v, rate=80.0, desired=0.2, actual=0.2))
  assert r.reason & GovernorReason.STEERING_RATE_COMFORT
  assert not (r.reason & GovernorReason.HIGH_STEERING_RATE)
  assert r.cap == pytest.approx(STEERING_RATE_COMFORT_MIN_CAP, abs=1e-9)
  comfort_slew = float(np.interp(v, OUTPUT_SLEW_RATE_BP, OUTPUT_SLEW_RATE_V)) * STEERING_RATE_COMFORT_MIN_SLEW_SCALE
  assert abs(r.output_torque) <= comfort_slew * DT + 1e-9


def test_steering_rate_comfort_ignores_torque_opposing_wheel_motion():
  r = OutputGovernor(DT).update(benign(nominal=-MAX, v=20.0, rate=80.0, desired=-0.2, actual=-0.2))
  assert not (r.reason & GovernorReason.STEERING_RATE_COMFORT)
  assert r.cap == pytest.approx(1.0, abs=1e-9)


def test_steering_rate_comfort_applies_symmetrically_negative():
  r = OutputGovernor(DT).update(benign(nominal=-MAX, v=20.0, rate=-80.0, desired=-0.2, actual=-0.2))
  assert r.reason & GovernorReason.STEERING_RATE_COMFORT
  assert not (r.reason & GovernorReason.HIGH_STEERING_RATE)
  assert r.cap == pytest.approx(STEERING_RATE_COMFORT_MIN_CAP, abs=1e-9)


def test_steering_rate_comfort_preserves_tracking_correction():
  # Actual lateral accel is lagging desired response in the commanded direction. Even with a
  # high steering rate, this is a tracking catch-up case and must not be comfort-throttled.
  r = OutputGovernor(DT).update(benign(nominal=MAX, v=20.0, rate=80.0, desired=1.0, actual=0.2))
  assert not (r.reason & GovernorReason.STEERING_RATE_COMFORT)
  assert r.cap == pytest.approx(1.0, abs=1e-9)


def test_steering_rate_comfort_preserves_over_response_correction():
  r = OutputGovernor(DT).update(benign(nominal=-MAX, v=20.0, rate=-80.0, desired=0.5, actual=0.8))
  assert not (r.reason & GovernorReason.STEERING_RATE_COMFORT)
  assert r.cap == pytest.approx(1.0, abs=1e-9)


def test_steering_rate_comfort_preserves_centering_correction():
  r = OutputGovernor(DT).update(benign(nominal=-MAX, v=20.0, rate=-80.0, desired=0.0, actual=0.3))
  assert not (r.reason & GovernorReason.STEERING_RATE_COMFORT)
  assert r.cap == pytest.approx(1.0, abs=1e-9)


def test_steering_rate_comfort_bypasses_release():
  r = OutputGovernor(DT).update(benign(nominal=MAX, v=20.0, rate=80.0, desired=0.2, actual=0.2, release=True))
  assert not (r.reason & GovernorReason.STEERING_RATE_COMFORT)


def test_steering_rate_comfort_stacks_with_hard_high_rate_guard():
  r = OutputGovernor(DT).update(benign(nominal=MAX, v=20.0, rate=100.0, desired=0.2, actual=0.2))
  assert r.reason & GovernorReason.STEERING_RATE_COMFORT
  assert r.reason & GovernorReason.HIGH_STEERING_RATE
  assert r.cap < STEERING_RATE_COMFORT_MIN_CAP


def test_steering_rate_comfort_does_not_slow_unwind():
  # Comfort/high-rate slew scalings must not apply to a same-sign decrease: the unwind
  # runs at the full release rate, not the comfort-scaled build rate.
  gov = OutputGovernor(DT)
  for _ in range(300):
    gov.update(benign(nominal=MAX, v=20.0))
  assert gov.previous_output == pytest.approx(MAX, abs=1e-6)
  r = gov.update(benign(nominal=0.2, v=20.0, rate=80.0, desired=0.2, actual=0.2))
  assert r.reason & GovernorReason.STEERING_RATE_COMFORT
  release = float(np.interp(20.0, OUTPUT_SLEW_RATE_BP, OUTPUT_SLEW_RATE_V)) * RELEASE_SLEW_SCALE
  assert r.output_torque == pytest.approx(MAX - release * DT, abs=1e-9)


def test_release_slew_bounds_same_sign_decrease():
  gov = OutputGovernor(DT)
  v = 20.0
  for _ in range(300):
    gov.update(benign(nominal=MAX, v=v))
  release = float(np.interp(v, OUTPUT_SLEW_RATE_BP, OUTPUT_SLEW_RATE_V)) * RELEASE_SLEW_SCALE
  prev = gov.previous_output
  for _ in range(200):
    r = gov.update(benign(nominal=0.0, v=v))
    assert abs(r.output_torque - prev) <= release * DT + 1e-9
    prev = r.output_torque
  assert prev == pytest.approx(0.0, abs=1e-6)  # eventually reaches the command


def test_driver_release_bypasses_release_slew():
  gov = OutputGovernor(DT)
  for _ in range(300):
    gov.update(benign(nominal=MAX, v=20.0))
  r = gov.update(benign(nominal=0.1, v=20.0, release=True))
  assert r.output_torque == pytest.approx(0.1, abs=1e-9)


def test_over_response_yields_through_release_backstop():
  # Over-response is a CONTINUOUS attenuation, not a discrete safety event, so it must
  # yield through the release backstop instead of snapping. It used to be in
  # `fast_release`: on route 00000302 it toggled 91-98x/min in curves and each toggle
  # discharged the whole accumulated command-vs-nominal gap in one frame, which is what
  # the wheel-jerk complaint was. The attenuation still applies -- only the step is gone.
  gov = OutputGovernor(DT)
  gov.previous_output = 0.5
  target = 0.3 * OVER_RESPONSE_MIN_SCALE
  release_step = OUTPUT_SLEW_RATE_V[0] * RELEASE_SLEW_SCALE * DT

  r = gov.update(benign(nominal=0.3, v=20.0, desired=1.0, actual=1.8))
  assert r.reason & GovernorReason.OVER_RESPONSE
  assert r.reason & GovernorReason.SLEW_LIMITED
  assert r.output_torque == pytest.approx(0.5 - release_step, abs=1e-9)

  # and it still gets all the way there, just rate-limited
  for _ in range(200):
    r = gov.update(benign(nominal=0.3, v=20.0, desired=1.0, actual=1.8))
  assert r.output_torque == pytest.approx(target, abs=1e-9)


@pytest.mark.parametrize("kwargs,reason", [
  ({"release": True}, GovernorReason.OVERRIDE_RELEASE),
  ({"desired": -1.0, "actual": 1.0}, GovernorReason.SIGN_CONFLICT),
])
def test_discrete_safety_events_still_bypass_release_slew(kwargs, reason):
  # The three genuinely discrete triggers must keep landing in one frame.
  gov = OutputGovernor(DT)
  gov.previous_output = 0.9
  r = gov.update(benign(nominal=0.05, v=20.0, **kwargs))
  assert r.reason & reason
  assert abs(r.output_torque) < 0.9 - OUTPUT_SLEW_RATE_V[0] * RELEASE_SLEW_SCALE * DT - 1e-9


def test_steady_state_passthrough():
  gov = OutputGovernor(DT)
  r = gov.update(benign(nominal=0.3, v=25.0))
  for _ in range(500):
    r = gov.update(benign(nominal=0.3, v=25.0))  # benign, no caps/floor
  assert r.output_torque == pytest.approx(0.3, abs=1e-6)
  assert not (r.reason & GovernorReason.CLIPPED)


def test_output_preserves_command_sign_or_passes_through_zero():
  # Output may lag/reduce but must not spontaneously flip to the side opposing both the
  # command and the previous output.
  gov = OutputGovernor(DT)
  rng = np.random.default_rng(7)
  prev = 0.0
  for _ in range(2000):
    nominal = float(rng.uniform(-MAX, MAX))
    r = gov.update(benign(nominal=nominal, v=float(rng.uniform(0, 40))))
    s = np.sign(r.output_torque)
    if s != 0:
      assert s == np.sign(nominal) or s == np.sign(prev)
    prev = r.output_torque


def test_pre_slew_target_isolates_the_slew_stage():
  """Observability: nominal -> pre_slew -> output must separate a cap/arrival step from
  stateful slew catch-up. With the command pinned high and no caps, pre_slew sits at the
  full command from frame 1 while the output crawls up under the slew limit."""
  gov = OutputGovernor(DT)
  slew = float(np.interp(20.0, OUTPUT_SLEW_RATE_BP, OUTPUT_SLEW_RATE_V))
  first = gov.update(benign(nominal=MAX, v=20.0))
  assert first.pre_slew_target == pytest.approx(MAX, abs=1e-9), "pre_slew must be pre-rate-limit"
  assert abs(first.output_torque) <= slew * DT + 1e-9, "output must still be slew limited"
  assert first.reason & GovernorReason.SLEW_LIMITED
  # and it converges to the output once the slew has caught up
  for _ in range(400):
    r = gov.update(benign(nominal=MAX, v=20.0))
  assert r.pre_slew_target == pytest.approx(r.output_torque, abs=1e-6)


def test_pre_slew_target_reflects_the_cap_not_the_command():
  """When a cap binds, pre_slew carries the CAPPED value — so nominal vs pre_slew shows the
  cap step, and pre_slew vs output shows the slew contribution, separately. Same over-turn
  scenario as test_over_turn_cap_binds_opposite_torque_low_speed."""
  gov = OutputGovernor(DT)
  gov.previous_output = -0.5
  r = gov.update(benign(nominal=-0.5, v=8.0, desired=1.0, actual=1.3))
  assert r.reason & GovernorReason.CLIPPED
  # pre_slew already carries the cap on the binding frame, before the slew stage runs
  assert r.pre_slew_target == pytest.approx(-OVER_TURN_MAX_OPPOSITE_FRAC * MAX, abs=1e-6)
  assert abs(r.pre_slew_target) < abs(-0.5), "cap step must be visible in pre_slew"
  for _ in range(200):  # let the slew settle so output converges onto pre_slew
    r = gov.update(benign(nominal=-0.5, v=8.0, desired=1.0, actual=1.3))
  assert r.pre_slew_target == pytest.approx(r.output_torque, abs=1e-6)


def test_pre_slew_target_defaults_zero_when_inactive():
  gov = OutputGovernor(DT)
  r = gov.update(benign(active=False))
  assert r.pre_slew_target == 0.0
