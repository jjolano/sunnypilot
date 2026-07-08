"""Bounded moving-lead cruise-cap candidate: shadow/apply and fail-closed guards."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from openpilot.sunnypilot.custom.longitudinal.moving_lead_cruise_cap import MovingLeadCruiseCap

DT = 0.05
BASE = 24.0


class FakeParams:
  def __init__(self, **kw):
    self._d = kw

  def get_bool(self, k):
    return bool(self._d.get(k, False))

  def get(self, k, default=None, return_default=False):
    return self._d.get(k, default)


def lead(*, status=True, d_rel=45.0, y_rel=0.0, v_lead=20.0, v_rel=-1.0, a_lead=-0.5,
         radar_confirmed=True):
  return SimpleNamespace(status=status, dRel=d_rel, yRel=y_rel, vLead=v_lead, vRel=v_rel,
                         aLeadK=a_lead, radar=radar_confirmed)


def radar(l1=None):
  return SimpleNamespace(leadOne=l1, leadTwo=SimpleNamespace(status=False))


def _cap(mode="apply", **params):
  return MovingLeadCruiseCap(FakeParams(MovingLeadCruiseCapMode=mode, **params))


def _call(cap, rs, *, v_ego=20.0, v_cruise=BASE, research=True, **ctx):
  defaults = dict(long_active=True, brake_pressed=False, gas_pressed=False, force_decel=False,
                  custom_long_enabled=True, research_actuation_allowed=research)
  defaults.update(ctx)
  return cap.capped(rs, v_ego, v_cruise, DT, **defaults)


def test_off_returns_raw_and_clears_telemetry():
  cap = _cap("off")
  assert _call(cap, radar(lead())) == BASE
  assert cap.last_result is None


def test_missing_mode_defaults_to_shadow():
  cap = MovingLeadCruiseCap(FakeParams(CustomLongitudinalEnabled=True))
  assert cap.mode == "shadow"
  assert _call(cap, radar(lead())) == BASE
  assert cap.last_result is not None


def test_shadow_returns_raw_but_records_candidate():
  cap = _cap("shadow")
  out = _call(cap, radar(lead(v_lead=18.0, v_rel=-1.0, a_lead=-0.5)))
  assert out == BASE
  assert cap.last_result is not None
  assert cap.last_result["mode"] == "shadow"
  assert cap.last_result["apply"] is False
  assert cap.last_result["eligible"] is True
  assert cap.last_result["block_reason"] == ""
  assert cap.last_result["v_cruise"] == BASE
  assert cap.last_result["capped_v_cruise"] == pytest.approx(18.5)


def test_invalid_mode_fails_closed_to_off():
  cap = _cap("bad")
  assert cap.mode == "off"
  assert _call(cap, radar(lead())) == BASE
  assert cap.last_result is None


@pytest.mark.parametrize("ctx", [
  dict(custom_long_enabled=False),
  dict(research_actuation_allowed=False),
])
def test_apply_requires_custom_long_and_research_gate(ctx):
  cap = _cap("apply")
  out = _call(cap, radar(lead(v_lead=18.0, v_rel=-1.0, a_lead=-0.5)), **ctx)
  assert out == BASE
  assert cap.last_result is not None
  assert cap.last_result["eligible"] is True
  assert cap.last_result["apply"] is False


def test_apply_blocks_when_long_inactive():
  cap = _cap("apply")
  out = _call(cap, radar(lead()), long_active=False)
  assert out == BASE
  assert cap.last_result is not None
  assert cap.last_result["eligible"] is False
  assert cap.last_result["block_reason"] == "long_inactive"


def test_eligible_caps_to_lead_plus_allowance():
  cap = _cap("apply")
  out = _call(cap, radar(lead(v_lead=18.0, v_rel=-1.0, a_lead=-0.5)))
  assert out == pytest.approx(18.5)
  assert cap.last_result is not None
  assert cap.last_result["apply"] is True
  assert cap.last_result["capped_v_cruise"] == pytest.approx(18.5)


def test_radar_confirmed_coasting_lead_can_cap():
  cap = _cap("apply")
  out = _call(cap, radar(lead(v_lead=18.0, v_rel=-1.0, a_lead=0.0, radar_confirmed=True)))
  assert out == pytest.approx(18.5)
  assert cap.last_result is not None
  assert cap.last_result["eligible"] is True


def test_model_only_coasting_lead_stays_blocked():
  cap = _cap("apply")
  out = _call(cap, radar(lead(v_lead=18.0, v_rel=-1.0, a_lead=0.0, radar_confirmed=False)))
  assert out == BASE
  assert cap.last_result is not None
  assert cap.last_result["eligible"] is False
  assert cap.last_result["block_reason"] == "lead_not_braking"


def test_does_not_raise_when_raw_cruise_is_lower():
  cap = _cap("apply")
  out = _call(cap, radar(lead(v_lead=18.0, v_rel=-1.0, a_lead=-0.5)), v_cruise=17.0)
  assert out == 17.0
  assert cap.last_result is not None
  assert cap.last_result["apply"] is False
  assert cap.last_result["capped_v_cruise"] == 17.0


@pytest.mark.parametrize("lead_obj,ctx,reason", [
  (None, {}, "no_lead"),
  (lead(a_lead=0.2), {}, "lead_not_braking"),
  (lead(a_lead=-2.0), {}, "lead_hard_braking"),
  (lead(y_rel=2.0), {}, "lead_off_path"),
  (lead(v_lead=2.0), {}, "lead_slow"),
  (lead(v_rel=0.0, a_lead=-0.5), {}, "not_closing"),
  (lead(v_rel=-4.5, a_lead=-0.5), {}, "fast_closing"),
  (lead(d_rel=30.0, v_rel=-4.0, a_lead=-0.5), {}, "low_ttc"),
  (lead(d_rel=11.0, v_rel=-1.0, a_lead=-0.5), dict(v_ego=10.0), "too_close"),
  (lead(), dict(v_ego=7.0), "low_speed"),
  (lead(), dict(brake_pressed=True), "brake_pressed"),
  (lead(), dict(gas_pressed=True), "gas_pressed"),
  (lead(), dict(force_decel=True), "force_decel"),
  (lead(), dict(long_active=False), "long_inactive"),
])
def test_ineligible_reasons(lead_obj, ctx, reason):
  cap = _cap("apply")
  out = _call(cap, radar(lead_obj), **ctx)
  assert out == BASE
  assert cap.last_result is not None
  assert cap.last_result["eligible"] is False
  assert cap.last_result["block_reason"] == reason
