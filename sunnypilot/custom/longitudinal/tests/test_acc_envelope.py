from __future__ import annotations

import pytest

from openpilot.sunnypilot.custom.longitudinal.acc_envelope import AccEnvelopeInputs, evaluate_acc_envelope


def base(**overrides):
  data = dict(v_ego=20.0, candidate_a_target=0.4, previous_a_target=0.4, dt=0.05)
  data.update(overrides)
  return AccEnvelopeInputs(**data)


def reasons(result):
  return set(result.cap_reasons)


def test_no_lead_has_no_gap_cap():
  r = evaluate_acc_envelope(base())

  assert r.active is True
  assert r.would_cap is False
  assert r.allowed_a_target == pytest.approx(0.4)


def test_lead_inside_desired_gap_caps():
  r = evaluate_acc_envelope(base(has_lead=True, lead_d_rel=20.0, lead_v_rel=0.0, lead_v_lead=20.0))

  assert r.would_cap is True
  assert "inside_time_gap" in reasons(r)
  assert r.desired_gap == pytest.approx(30.0)
  assert r.allowed_a_target <= 0.0


def test_closing_lead_low_ttc_caps():
  r = evaluate_acc_envelope(base(has_lead=True, lead_d_rel=20.0, lead_v_rel=-8.0, lead_v_lead=12.0))

  assert r.would_cap is True
  assert "ttc_low" in reasons(r)
  assert r.ttc == pytest.approx(2.5)


def test_required_decel_above_limit_caps():
  r = evaluate_acc_envelope(base(has_lead=True, lead_d_rel=40.0, lead_v_rel=-5.0, lead_v_lead=15.0))

  assert r.would_cap is True
  assert "closing_decel_high" in reasons(r)
  assert r.required_stopping_decel > 1.0


def test_opening_lead_outside_gap_does_not_cap():
  r = evaluate_acc_envelope(base(has_lead=True, lead_d_rel=80.0, lead_v_rel=1.0, lead_v_lead=21.0))

  assert r.would_cap is False
  assert r.time_gap == pytest.approx(4.0)
  assert r.ttc == float("inf")


def test_stale_model_blocks_model_progress_authority():
  r = evaluate_acc_envelope(base(model_stale=True, model_progress_candidate=True))

  assert r.would_cap is True
  assert "model_stale_blocks_model_progress" in reasons(r)


def test_stock_acc_records_stock_control_reason():
  r = evaluate_acc_envelope(base(openpilot_longitudinal_control=False))

  assert r.would_cap is True
  assert "stock_longitudinal_control" in reasons(r)


def test_invalid_lead_fields_fail_closed_in_shadow():
  r = evaluate_acc_envelope(base(has_lead=True, lead_d_rel=float("nan")))

  assert r.active is False
  assert r.would_cap is True
  assert "invalid_data" in reasons(r)


def test_jerk_limiter_clips_accel_rise():
  r = evaluate_acc_envelope(base(candidate_a_target=1.0, previous_a_target=0.0, dt=0.05))

  assert r.would_cap is True
  assert "jerk_limited" in reasons(r)
  assert r.jerk_limited_a_target == pytest.approx(0.05)


def test_jerk_limiter_clips_braking_step():
  r = evaluate_acc_envelope(base(candidate_a_target=-1.0, previous_a_target=0.0, dt=0.05))

  assert r.would_cap is False  # decel limiting is shadow telemetry, not a safety cap upward
  assert r.jerk_limited_a_target == pytest.approx(-0.1)


def test_inside_time_gap_hardens_beyond_mild_candidate_without_compression_flag():
  # Inside the desired gap with mild closing: the raw kinematic demand (-required_decel)
  # is stronger than the candidate compression target, so it hardens without the flag.
  r = evaluate_acc_envelope(base(
    v_ego=20.0, candidate_a_target=-0.15, previous_a_target=-0.15, dt=0.05,
    has_lead=True, lead_d_rel=25.0, lead_v_rel=-0.2, lead_v_lead=19.8,
  ))

  assert r.would_cap is True
  assert "inside_time_gap" in reasons(r)
  assert "ttc_low" not in reasons(r)
  assert "closing_decel_high" not in reasons(r)
  assert r.allowed_a_target < -0.15


def test_lead_compression_candidate_skips_inside_time_gap_hardening():
  # Same kinematics as above, but identified as a controlled compression candidate:
  # inside_time_gap is still reported but the candidate target is allowed to bind.
  r = evaluate_acc_envelope(base(
    v_ego=20.0, candidate_a_target=-0.15, previous_a_target=-0.15, dt=0.05,
    has_lead=True, lead_d_rel=25.0, lead_v_rel=-0.2, lead_v_lead=19.8,
    lead_compression_candidate=True,
  ))

  assert r.would_cap is True
  assert "inside_time_gap" in reasons(r)
  assert r.allowed_a_target == pytest.approx(-0.15)


def test_lead_compression_candidate_still_bound_by_real_risk_reasons():
  # True low TTC must still override the compression allowance.
  r = evaluate_acc_envelope(base(
    v_ego=20.0, candidate_a_target=-0.15, previous_a_target=-0.15, dt=0.05,
    has_lead=True, lead_d_rel=10.0, lead_v_rel=-4.0, lead_v_lead=16.0,
    lead_compression_candidate=True,
  ))

  assert r.would_cap is True
  assert "inside_time_gap" in reasons(r)
  assert "ttc_low" in reasons(r)
  assert r.allowed_a_target < -0.15


def test_lead_compression_candidate_closing_decel_high_safe_ttc_does_not_harden():
  # closing_decel_high due to being inside the desired gap, but TTC is safe and kinematics
  # are valid. For a controlled compression candidate the target must not harden.
  r = evaluate_acc_envelope(base(
    v_ego=20.0, candidate_a_target=-0.60, previous_a_target=-0.60, dt=0.05,
    has_lead=True, lead_d_rel=25.0, lead_v_rel=-0.5, lead_v_lead=19.5,
    lead_compression_candidate=True,
  ))

  assert r.would_cap is True
  assert "inside_time_gap" in reasons(r)
  assert "closing_decel_high" in reasons(r)
  assert "ttc_low" not in reasons(r)
  assert r.allowed_a_target == pytest.approx(-0.60)


def test_non_compression_closing_decel_high_still_hardens():
  # Same closing_decel_high condition without the compression flag must harden.
  r = evaluate_acc_envelope(base(
    v_ego=20.0, candidate_a_target=-0.60, previous_a_target=-0.60, dt=0.05,
    has_lead=True, lead_d_rel=25.0, lead_v_rel=-0.5, lead_v_lead=19.5,
    lead_compression_candidate=False,
  ))

  assert r.would_cap is True
  assert "closing_decel_high" in reasons(r)
  assert r.allowed_a_target < -0.60
