"""Invariant (property) tests for the unified output governor first cut.

These gate the governor's INVARIANTS — safety bound, reset, floor-relaxes-only,
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
  SAME_DIRECTION_LIMIT_CAP,
  SIGN_CHANGE_SLEW_RATE_BP,
  SIGN_CHANGE_SLEW_RATE_V,
  STEERING_RATE_COMFORT_MIN_CAP,
  STEERING_RATE_COMFORT_MIN_SLEW_SCALE,
  GovernorReason,
  OutputGovernor,
  OutputGovernorInputs,
)

DT = 0.01
MAX = 1.0


def benign(nominal=0.0, v=20.0, rate=0.0, desired=0.0, actual=0.0,
           same_dir=False, release=False, active=True, path_valid=True, controller_stable=True):
  """An input with no cap/floor triggers unless overridden."""
  return OutputGovernorInputs(active=active, v_ego=v, steering_rate_deg=rate,
                              nominal_torque=nominal, max_output=MAX,
                              desired_lateral_accel=desired, actual_lateral_accel=actual,
                              same_direction_limit=same_dir, release_active=release,
                              path_evidence_valid=path_valid,
                              controller_evidence_stable=controller_stable)


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
  gov = OutputGovernor(DT)
  for _ in range(300):
    gov.update(benign(nominal=MAX, v=20.0))
  assert gov.previous_output == pytest.approx(MAX, abs=1e-6)
  r = gov.update(benign(nominal=0.2, v=20.0, rate=80.0, desired=0.2, actual=0.2))
  assert r.reason & GovernorReason.STEERING_RATE_COMFORT
  assert r.output_torque == pytest.approx(0.2, abs=1e-9)
  assert not (r.reason & GovernorReason.SLEW_LIMITED)


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
