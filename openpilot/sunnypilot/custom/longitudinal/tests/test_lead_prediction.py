"""Tests for the improved lead trajectory prediction (aLeadTau decay, ego accel, vRel)."""
from __future__ import annotations

import math

import pytest

from openpilot.sunnypilot.custom.longitudinal.lead_prediction import predict_lead_trajectory


def test_a_lead_decays_toward_zero():
  p = predict_lead_trajectory(d_rel=30.0, v_rel=0.0, v_lead=15.0, a_lead=2.0, a_lead_tau=1.0, v_ego=15.0)
  # decayed accel strictly decreases over the horizon and never exceeds a_lead
  assert all(p.a_lead[i + 1] < p.a_lead[i] for i in range(len(p.a_lead) - 1))
  assert p.a_lead[0] <= 2.0


def test_decayed_speed_below_constant_accel_projection():
  # with a_lead held constant, v(2s) would be 15 + 2*2 = 19; decayed must be lower
  p = predict_lead_trajectory(d_rel=30.0, v_rel=0.0, v_lead=15.0, a_lead=2.0, a_lead_tau=1.0, v_ego=15.0)
  v_2s_constant = 15.0 + 2.0 * 2.0
  assert p.v_lead[-1] < v_2s_constant
  rate = 1.5  # optimistic pullaway uses the same minimum decay rate as long_mpc
  expected_delta = 2.0 * math.sqrt(math.pi / (2.0 * rate)) * math.erf(math.sqrt(rate / 2.0) * 2.0)
  assert p.v_lead[-1] == pytest.approx(15.0 + expected_delta, abs=1e-9)


def test_positive_lead_accel_uses_mpc_pullaway_cap():
  capped = predict_lead_trajectory(40.0, 0.0, 15.0, 5.0, 1.5, 15.0)
  reference = predict_lead_trajectory(40.0, 0.0, 15.0, 2.0, 1.5, 15.0)

  assert capped == reference


def test_a_lead_tau_is_the_mpc_gaussian_decay_rate():
  sustained = predict_lead_trajectory(40.0, 0.0, 20.0, -2.0, 0.1, 20.0)
  fast_decay = predict_lead_trajectory(40.0, 0.0, 20.0, -2.0, 1.5, 20.0)

  assert sustained.a_lead[-1] < fast_decay.a_lead[-1]
  assert sustained.v_lead[-1] < fast_decay.v_lead[-1]
  assert sustained.gap[-1] < fast_decay.gap[-1]


def test_ego_accel_closes_gap_more():
  base = predict_lead_trajectory(30.0, v_rel=-2.0, v_lead=13.0, a_lead=0.0, a_lead_tau=1.0,
                                 v_ego=15.0, a_ego=0.0)
  ego_accel = predict_lead_trajectory(30.0, v_rel=-2.0, v_lead=13.0, a_lead=0.0, a_lead_tau=1.0,
                                      v_ego=15.0, a_ego=1.0)
  # accelerating ego closes the gap faster than the constant-ego projection
  assert ego_accel.gap[-1] < base.gap[-1]


def test_braking_ego_holds_at_standstill_instead_of_reversing():
  p = predict_lead_trajectory(20.0, v_rel=-2.0, v_lead=0.0, a_lead=0.0, a_lead_tau=1.0,
                              v_ego=2.0, a_ego=-2.0)

  assert p.gap[1] == pytest.approx(19.0)
  assert p.gap[2] == pytest.approx(p.gap[1])
  assert p.gap[3] == pytest.approx(p.gap[1])


def test_vrel_drives_linear_gap():
  # pure closing, no accel: gap(t) = d_rel + v_rel*t
  p = predict_lead_trajectory(40.0, v_rel=-4.0, v_lead=11.0, a_lead=0.0, a_lead_tau=1.0,
                              v_ego=15.0, a_ego=0.0)
  for t, g in zip(p.t, p.gap, strict=True):
    assert g == pytest.approx(max(0.0, 40.0 - 4.0 * t))


def test_low_accel_confidence_falls_back_to_constant_velocity():
  full = predict_lead_trajectory(30.0, v_rel=0.0, v_lead=15.0, a_lead=2.0, a_lead_tau=1.0,
                                 v_ego=15.0, accel_confidence=1.0)
  none = predict_lead_trajectory(30.0, v_rel=0.0, v_lead=15.0, a_lead=2.0, a_lead_tau=1.0,
                                 v_ego=15.0, accel_confidence=0.0)
  assert none.v_lead[-1] == pytest.approx(15.0)            # no accel trusted -> constant v
  assert full.v_lead[-1] > none.v_lead[-1]


def test_gaps_non_negative_and_finite():
  p = predict_lead_trajectory(5.0, v_rel=-10.0, v_lead=0.0, a_lead=-3.0, a_lead_tau=0.5,
                              v_ego=10.0, a_ego=2.0)
  assert all(g >= 0.0 and math.isfinite(g) for g in p.gap)


def test_braking_lead_holds_at_standstill_and_gap_freezes():
  # Stationary ego (v_rel = +v_lead, opening): once the decaying decel brings the lead to
  # rest, the gap must freeze instead of reversing the lead.
  p = predict_lead_trajectory(20.0, v_rel=2.0, v_lead=2.0, a_lead=-4.0, a_lead_tau=0.1,
                              v_ego=0.0, a_ego=0.0)
  assert p.v_lead[1:] == (0.0, 0.0, 0.0)
  assert p.a_lead[1:] == (0.0, 0.0, 0.0)
  assert p.gap[2] == pytest.approx(p.gap[1])
  assert p.gap[3] == pytest.approx(p.gap[1])
  assert p.gap[0] < p.gap[1]


def test_already_stopped_braking_lead_never_reverses():
  # A stopped lead reporting residual negative accel must behave as pure standstill:
  # gap(t) = d_rel + v_rel*t exactly, with no phantom reverse displacement.
  p = predict_lead_trajectory(10.0, v_rel=-5.0, v_lead=0.0, a_lead=-3.0, a_lead_tau=1.0,
                              v_ego=5.0, a_ego=0.0)
  for t, g in zip(p.t, p.gap, strict=True):
    assert g == pytest.approx(max(0.0, 10.0 - 5.0 * t))
