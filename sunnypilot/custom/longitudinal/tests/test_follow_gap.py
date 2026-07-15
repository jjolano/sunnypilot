"""Dynamic follow-gap scheduler: bounded, research-gated T_FOLLOW compression on approach.
Safety: result always in [T_FOLLOW_COMPRESSED, base]; slow compression / fast recovery;
off/shadow and every fault return the baseline unchanged."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from openpilot.sunnypilot.custom.longitudinal.follow_gap import (
  COMPRESS_DEMAND_FULL,
  COMPRESS_RATE,
  COMPRESS_RATE_MAX,
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


def _compress_rate(d_rel, v_rel, v_ego=20.0, base=BASE):
  required = (min(v_rel, 0.0) ** 2) / (2 * (d_rel - base * v_ego))
  frac = min(required / COMPRESS_DEMAND_FULL, 1.0)
  return COMPRESS_RATE + (COMPRESS_RATE_MAX - COMPRESS_RATE) * frac


def test_apply_compresses_at_demand_scaled_rate_and_bounded():
  fg = _apply_scheduler()
  first = _call(fg, radar(lead()))
  assert first == pytest.approx(BASE - _compress_rate(40.0, -1.0) * DT)
  settled = _settled(fg, radar(lead()))
  assert settled == pytest.approx(T_FOLLOW_COMPRESSED)
  # never below the floor, never above base
  assert T_FOLLOW_COMPRESSED <= settled <= BASE


def test_hotter_approach_compresses_faster_capped():
  mild = _apply_scheduler()
  warm = _apply_scheduler()
  # closing 3 m/s at 40 m: TTC 13.3 s, still eligible, but real approach demand
  mild_first = _call(mild, radar(lead(v_rel=-1.0)))
  warm_first = _call(warm, radar(lead(v_rel=-3.0)))
  assert warm_first < mild_first
  assert warm_first == pytest.approx(BASE - _compress_rate(40.0, -3.0) * DT)
  # saturated demand: rate caps at COMPRESS_RATE_MAX (closing 3.9 at 40 m, TTC 10.3 s)
  hot = _apply_scheduler()
  hot_first = _call(hot, radar(lead(v_rel=-3.9)))
  assert hot_first == pytest.approx(BASE - COMPRESS_RATE_MAX * DT)


def test_recovery_is_faster_than_compression():
  fg = _apply_scheduler()
  _settled(fg, radar(lead()))
  # lead starts braking hard -> ineligible -> recover at the fast rate
  out = _call(fg, radar(lead(a_lead=-1.5)))
  assert out == pytest.approx(min(BASE, T_FOLLOW_COMPRESSED + RECOVER_RATE * DT))
  assert RECOVER_RATE > COMPRESS_RATE_MAX > COMPRESS_RATE


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


def test_benign_approach_end_recovers_at_gap_opening_pace():
  from openpilot.sunnypilot.custom.longitudinal.follow_gap import BENIGN_RECOVER_RATE_MIN
  fg = _apply_scheduler()
  _settled(fg, radar(lead()))
  # approach ends: lead pulls away at 1 m/s (v_ego 20) -> recovery paced at ~v_rel/v_ego
  out = _call(fg, radar(lead(v_rel=1.0)))
  assert out == pytest.approx(T_FOLLOW_COMPRESSED + (1.0 / 20.0) * DT)
  # stalled queue (not closing, not opening) still trickles back at the floor rate
  fg2 = _apply_scheduler()
  _settled(fg2, radar(lead()))
  out2 = _call(fg2, radar(lead(v_rel=0.0)))
  assert out2 == pytest.approx(T_FOLLOW_COMPRESSED + BENIGN_RECOVER_RATE_MIN * DT)
  # both far slower than the safety snap-back
  assert out < T_FOLLOW_COMPRESSED + RECOVER_RATE * DT


def test_safety_shaped_ineligibility_keeps_fast_recovery():
  # every safety reason recovers at RECOVER_RATE even though the approach also ended
  for bad in (lead(a_lead=-1.5),            # lead braking
              lead(v_rel=-5.0),             # fast closing
              lead(d_rel=30.0, v_rel=-4.0), # low ttc
              lead(d_rel=10.0)):            # too close at 20 m/s
    fg = _apply_scheduler()
    _settled(fg, radar(lead()))
    out = _call(fg, radar(bad))
    assert out == pytest.approx(min(BASE, T_FOLLOW_COMPRESSED + RECOVER_RATE * DT)), bad
  # pedals snap back fast too
  fg = _apply_scheduler()
  _settled(fg, radar(lead()))
  out = _call(fg, radar(lead(v_rel=1.0)), brake_pressed=True)
  assert out == pytest.approx(min(BASE, T_FOLLOW_COMPRESSED + RECOVER_RATE * DT))
