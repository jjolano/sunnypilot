"""Dynamic follow-gap scheduler: bounded, research-gated T_FOLLOW compression on approach.
Safety: result always in [T_FOLLOW_COMPRESSED, base]; slow compression / fast recovery;
off/shadow and every fault return the baseline unchanged."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from openpilot.sunnypilot.custom.longitudinal.follow_gap import (
  COMPRESS_RATE,
  RECOVER_RATE,
  T_FOLLOW_COMPRESSED,
  FollowGapScheduler,
)

DT = 0.05
BASE = 1.45


class FakeParams:
  def __init__(self, **kw):
    self._d = kw

  def get_bool(self, k):
    return bool(self._d.get(k, False))

  def get(self, k, default=None, return_default=False):
    return self._d.get(k, default)


def lead(d_rel=40.0, v_lead=15.0, v_rel=-1.0, a_lead=0.0, status=True):
  return SimpleNamespace(status=status, dRel=d_rel, vLead=v_lead, vRel=v_rel, aLeadK=a_lead)


def radar(l1, l2=None):
  return SimpleNamespace(leadOne=l1, leadTwo=l2 if l2 is not None else SimpleNamespace(status=False))


def _apply_scheduler():
  return FollowGapScheduler(FakeParams(DynamicFollowGapMode="apply", CustomLongitudinalEnabled=True))


def _call(fg, rs, v_ego=20.0, base=BASE, dt=DT, research=True, **ctx):
  defaults = dict(long_active=True, brake_pressed=False, gas_pressed=False, force_decel=False,
                  research_actuation_allowed=research)
  defaults.update(ctx)
  return fg.scheduled(rs, v_ego, base, dt, **defaults)


def _settled(fg, rs, seconds=20.0, **kw):
  out = BASE
  for _ in range(int(seconds / DT)):
    out = _call(fg, rs, **kw)
  return out


def test_off_mode_returns_base():
  fg = FollowGapScheduler(FakeParams(DynamicFollowGapMode="off", CustomLongitudinalEnabled=True))
  assert _call(fg, radar(lead())) == BASE
  assert fg.last_result is None


def test_shadow_mode_returns_base_but_records_would_be_value():
  fg = FollowGapScheduler(FakeParams(DynamicFollowGapMode="shadow", CustomLongitudinalEnabled=True))
  out = _settled(fg, radar(lead()))
  assert out == BASE
  assert fg.last_result is not None
  assert fg.last_result["eligible"] is True
  assert fg.last_result["t_follow"] == pytest.approx(T_FOLLOW_COMPRESSED)
  assert fg.last_result["apply"] is False


def test_apply_requires_research_gate_and_custom_long():
  fg = _apply_scheduler()
  assert _call(fg, radar(lead()), research=False) == BASE
  fg2 = FollowGapScheduler(FakeParams(DynamicFollowGapMode="apply", CustomLongitudinalEnabled=False))
  assert _call(fg2, radar(lead())) == BASE


def test_apply_compresses_slowly_and_bounded():
  fg = _apply_scheduler()
  first = _call(fg, radar(lead()))
  assert first == pytest.approx(BASE - COMPRESS_RATE * DT)
  settled = _settled(fg, radar(lead()))
  assert settled == pytest.approx(T_FOLLOW_COMPRESSED)
  # never below the floor, never above base
  assert T_FOLLOW_COMPRESSED <= settled <= BASE


def test_recovery_is_faster_than_compression():
  fg = _apply_scheduler()
  _settled(fg, radar(lead()))
  # lead starts braking hard -> ineligible -> recover at the fast rate
  out = _call(fg, radar(lead(a_lead=-1.5)))
  assert out == pytest.approx(min(BASE, T_FOLLOW_COMPRESSED + RECOVER_RATE * DT))
  assert RECOVER_RATE > COMPRESS_RATE


@pytest.mark.parametrize("bad_lead,reason", [
  (lead(status=False), "no_lead"),
  (lead(d_rel=10.0), "lead_0_too_close"),               # under 1.05 s * 20 m/s
  (lead(v_rel=-5.0), "lead_0_fast_closing"),
  (lead(d_rel=30.0, v_rel=-4.0), "lead_0_low_ttc"),     # ttc 7.5 < 8
  (lead(v_lead=1.0), "lead_slow_or_stopped"),
  (lead(a_lead=-1.5), "lead_braking"),
  (lead(v_rel=0.0), "not_closing"),
  (lead(d_rel=float("nan")), "lead_0_non_finite"),
])
def test_ineligible_contexts_return_base(bad_lead, reason):
  fg = _apply_scheduler()
  assert _call(fg, radar(bad_lead)) == BASE
  assert fg.last_result["eligible"] is False
  assert fg.last_result["block_reason"] == reason


@pytest.mark.parametrize("ctx,reason", [
  (dict(long_active=False), "long_inactive"),
  (dict(brake_pressed=True), "brake_pressed"),
  (dict(gas_pressed=True), "gas_pressed"),
  (dict(force_decel=True), "force_decel"),
  (dict(v_ego=4.0), "low_speed"),
])
def test_driver_and_state_overrides_return_base(ctx, reason):
  fg = _apply_scheduler()
  assert _call(fg, radar(lead()), **ctx) == BASE
  assert fg.last_result["block_reason"] == reason


def test_second_lead_floors_also_gate():
  fg = _apply_scheduler()
  rs = radar(lead(), lead(d_rel=12.0))  # leadTwo too close at 20 m/s
  assert _call(fg, rs) == BASE
  assert fg.last_result["block_reason"] == "lead_1_too_close"


def test_disengagement_snaps_back_to_base():
  fg = _apply_scheduler()
  _settled(fg, radar(lead()))
  _call(fg, radar(lead()), long_active=False)
  # re-engage: no lingering compression from before the disengagement
  out = _call(fg, radar(lead(v_rel=0.0)))  # not closing -> stays at base
  assert out == BASE


def test_aggressive_base_below_floor_side_stays_bounded():
  fg = _apply_scheduler()
  settled = _settled(fg, radar(lead()), base=1.25)
  assert T_FOLLOW_COMPRESSED <= settled <= 1.25


def test_fault_returns_base():
  fg = _apply_scheduler()
  assert _call(fg, object()) == BASE  # no leadOne attr -> no_lead, still base
  # non-finite base: no raise, passthrough, and no stale state left behind
  _call(fg, radar(lead()), base=float("nan"))
  assert fg.last_result is None
