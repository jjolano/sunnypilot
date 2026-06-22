#!/usr/bin/env python3
import numpy as np
import pytest
from types import SimpleNamespace

from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import (
  LEAD_PULLAWAY_ACCEL_MAX,
  LongitudinalMpc,
  _LEAD_ACCEL_TAU,
)


def _fake_lead(d_rel, v_lead, a_lead, a_lead_tau=_LEAD_ACCEL_TAU):
  """Return a minimal lead-like object that process_lead can consume."""
  return SimpleNamespace(status=True, dRel=d_rel, vLead=v_lead, aLeadK=a_lead, aLeadTau=a_lead_tau)


class _FakeLongitudinalMpc:
  """Minimal stand-in for LongitudinalMpc when only testing process_lead/extrapolate_lead.

  Avoids constructing the Acados solver while still exercising the real method.
  """
  def __init__(self, v_ego):
    # x0 layout is [x_ego, v_ego, a_ego]
    self.x0 = np.array([0.0, v_ego, 0.0])

  extrapolate_lead = staticmethod(LongitudinalMpc.extrapolate_lead)
  limit_lead_accel_for_prediction = staticmethod(LongitudinalMpc.limit_lead_accel_for_prediction)
  process_lead = LongitudinalMpc.process_lead


class TestLongitudinalLeadExtrapolation:
  def test_positive_pullaway_is_conservatively_bounded(self):
    """A lead reported as accelerating hard should not be predicted to keep pulling away over the horizon.

    The a_lead_tau exponential decay keeps the extrapolated velocity far below the unbounded
    constant-acceleration prediction, which would make us chase the lead too aggressively.
    """
    x_lead, v_lead = 50.0, 15.0
    a_lead = 5.0
    mpc = _FakeLongitudinalMpc(v_ego=v_lead)
    lead_xv = mpc.process_lead(_fake_lead(x_lead, v_lead, a_lead))
    capped_xv = LongitudinalMpc.extrapolate_lead(x_lead, v_lead, LEAD_PULLAWAY_ACCEL_MAX, _LEAD_ACCEL_TAU)

    # t=0 must be exactly the observed lead state.
    assert lead_xv[0, 0] == pytest.approx(x_lead)
    assert lead_xv[0, 1] == pytest.approx(v_lead)

    # Velocity increases, but incrementally — never a discontinuous jump.
    assert np.all(np.diff(lead_xv[:, 1]) >= -1e-9)

    # process_lead must cap optimistic pullaway accel before extrapolating.
    np.testing.assert_allclose(lead_xv, capped_xv)

    # Continuous upper bound for integrating a*exp(-tau*t^2/2) from 0 to inf.
    max_delta_v = LEAD_PULLAWAY_ACCEL_MAX * np.sqrt(np.pi / (2.0 * _LEAD_ACCEL_TAU))
    assert lead_xv[-1, 1] <= v_lead + max_delta_v + 0.5

    # Sanity check that this is much less than constant acceleration over the full 10 s horizon.
    assert lead_xv[-1, 1] < v_lead + a_lead * 10.0 - 1.0

    # The acceleration itself is heavily discounted by the end of the horizon.
    a_lead_final = LEAD_PULLAWAY_ACCEL_MAX * np.exp(-_LEAD_ACCEL_TAU * (10.0 ** 2) / 2.)
    assert a_lead_final <= 0.1 * LEAD_PULLAWAY_ACCEL_MAX

  @pytest.mark.parametrize("model_prob", [0.0, 0.7, np.nan])
  def test_low_confidence_model_pullaway_is_not_extrapolated(self, model_prob):
    mpc = _FakeLongitudinalMpc(v_ego=15.0)
    model_only_lead = SimpleNamespace(status=True, dRel=40.0, vLead=15.0, aLeadK=4.0,
                                      aLeadTau=_LEAD_ACCEL_TAU, radar=False, modelProb=model_prob)

    lead_xv = mpc.process_lead(model_only_lead)

    assert np.allclose(lead_xv[:, 1], 15.0)

  @pytest.mark.parametrize("a_lead", [np.inf, -np.inf, np.nan])
  def test_non_finite_accel_fails_closed_to_constant_velocity(self, a_lead):
    mpc = _FakeLongitudinalMpc(v_ego=15.0)

    lead_xv = mpc.process_lead(_fake_lead(d_rel=40.0, v_lead=15.0, a_lead=a_lead))

    assert np.allclose(lead_xv[:, 1], 15.0)

  def test_positive_pullaway_uses_default_decay_even_with_low_tau(self):
    x_lead, v_lead = 50.0, 15.0

    low_tau = LongitudinalMpc.extrapolate_lead(x_lead, v_lead, LEAD_PULLAWAY_ACCEL_MAX, 0.0)
    default_tau = LongitudinalMpc.extrapolate_lead(x_lead, v_lead, LEAD_PULLAWAY_ACCEL_MAX, _LEAD_ACCEL_TAU)

    np.testing.assert_allclose(low_tau, default_tau)

  def test_braking_remains_responsive_with_low_tau(self):
    x_lead, v_lead = 50.0, 30.0

    braking = LongitudinalMpc.extrapolate_lead(x_lead, v_lead, -8.0, 0.0)

    assert braking[1, 1] < v_lead
    assert braking[-1, 1] <= v_lead - 5.0

  def test_braking_remains_more_responsive_than_pullaway(self):
    """Hard braking must propagate into the lead trajectory more strongly than an equally-reported pullaway.

    process_lead caps positive accel more tightly than negative accel, so a genuine brake should
    slow the predicted lead more than a symmetric positive spike speeds it up.
    """
    x_lead, v_lead = 50.0, 30.0
    mpc = _FakeLongitudinalMpc(v_ego=v_lead)

    pullaway = mpc.process_lead(_fake_lead(x_lead, v_lead, 8.0))
    braking = mpc.process_lead(_fake_lead(x_lead, v_lead, -8.0))

    # Sanity: both are forward motion over the discrete horizon.
    assert pullaway[-1, 0] > x_lead
    assert braking[-1, 0] > x_lead

    # Braking results in a closer predicted obstacle than pullaway.
    assert braking[-1, 0] < pullaway[-1, 0]

    # The braking lead is predicted to be much slower at the horizon than the pullaway lead is faster.
    assert (v_lead - braking[-1, 1]) > (pullaway[-1, 1] - v_lead) + 1.0

  def test_process_lead_uses_observed_vlead_at_t0(self):
    """process_lead must not replace the observed lead speed with ego speed or an extrapolated value.

    The first element of the returned trajectory represents the current lead state.
    """
    mpc = _FakeLongitudinalMpc(v_ego=25.0)
    lead = _fake_lead(d_rel=40.0, v_lead=12.5, a_lead=-2.0)

    lead_xv = mpc.process_lead(lead)

    assert lead_xv[0, 0] == pytest.approx(40.0)
    assert lead_xv[0, 1] == pytest.approx(12.5)
    # Ego speed must not leak into the lead's t=0 speed.
    assert lead_xv[0, 1] != pytest.approx(mpc.x0[1])
