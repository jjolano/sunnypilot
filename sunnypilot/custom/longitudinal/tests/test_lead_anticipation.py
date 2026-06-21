"""§3 lead-motion anticipation: confidence-shape the radar lead's accel before the MPC. Safety: only
ever make a *braking* lead look *less* braking (shaped aLeadK in [raw, 0]); never touch a non-braking,
high-confidence, or sustained brake; opt-in; fail-closed."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from openpilot.sunnypilot.custom.longitudinal.lead_anticipation import (
  AL_CAP_MAX_SOFTENING,
  DISCOUNT_FLOOR,
  LeadAnticipation,
)

DT = 0.05


class FakeParams:
  def __init__(self, **kw):
    self._d = kw

  def get_bool(self, k):
    return bool(self._d.get(k, False))

  def get(self, k, default=None, return_default=False):
    return self._d.get(k, default)


def lead(a_lead, d_rel=30.0, v_lead=18.0, v_rel=0.0, status=True, tid=3):
  return SimpleNamespace(status=status, dRel=d_rel, vLead=v_lead, vLeadK=v_lead, vRel=v_rel,
                         aLeadK=a_lead, aLeadTau=1.5, yRel=0.0, radarTrackId=tid, radar=True,
                         modelProb=0.9)


def radar(l1, l2=None):
  return SimpleNamespace(leadOne=l1, leadTwo=l2)


def _on():
  return LeadAnticipation(FakeParams(LeadAnticipationMode="apply", CustomLongitudinalEnabled=True))


def _shape_apply(la, rs, dt=DT, **ctx):
  """Call shape with the apply context that satisfies the default safety gates."""
  defaults = dict(long_active=True, brake_pressed=False, gas_pressed=False,
                  force_decel=False, v_ego=15.0)
  defaults.update(ctx)
  return la.shape(rs, dt, **defaults)


def test_disabled_is_passthrough():
  la = LeadAnticipation(FakeParams(LeadAnticipationMode="off", CustomLongitudinalEnabled=True))
  rs = radar(lead(-3.0))
  assert la.shape(rs, DT) is rs                       # exact same object, no wrapping
  assert la.last_result is None


def test_shadow_passthrough_and_records_summary():
  la = LeadAnticipation(FakeParams(LeadAnticipationMode="shadow", CustomLongitudinalEnabled=False))
  rs = radar(lead(-3.0))
  assert la.shape(rs, DT) is rs
  assert la.last_result and la.last_result["mode"] == "shadow"
  assert la.last_result["apply"] is False
  assert la.last_result["block_reason"] == "mode_or_enabled"
  assert la.last_result["leadOneRaw"] == pytest.approx(-3.0)
  assert la.last_result["leadOneShaped"] == pytest.approx(min(-3.0 * DISCOUNT_FLOOR, -3.0 + AL_CAP_MAX_SOFTENING))


def test_non_braking_lead_unchanged():
  out = _shape_apply(_on(), radar(lead(0.5)))         # accelerating lead
  assert out.leadOne.aLeadK == 0.5


def test_new_low_confidence_brake_is_discounted_and_capped():
  la = _on()
  out = _shape_apply(la, radar(lead(-3.0)))           # first frame -> new lead -> low confidence
  a = out.leadOne.aLeadK
  # Floored discount is capped by the hard limit on softening delta.
  assert a == pytest.approx(min(-3.0 * DISCOUNT_FLOOR, -3.0 + AL_CAP_MAX_SOFTENING))
  assert -3.0 < a <= 0.0                              # less negative than raw, never positive
  assert -3.0 + AL_CAP_MAX_SOFTENING - 1e-9 <= a <= 0.0  # hard cap respected
  # the rest of the lead is proxied unchanged
  assert out.leadOne.dRel == 30.0 and out.leadOne.vLead == 18.0


def test_hard_cap_limits_softening_for_braking_leads():
  for raw_a in (-0.6, -2.0, -5.0, -10.0):
    la = _on()
    for _ in range(20):
      a = _shape_apply(la, radar(lead(raw_a))).leadOne.aLeadK
      # shaped value is always in [raw, 0] and softens by no more than AL_CAP_MAX_SOFTENING
      assert raw_a - 1e-9 <= a <= 0.0
      assert a <= raw_a + AL_CAP_MAX_SOFTENING + 1e-9
      assert a >= raw_a - 1e-9


def test_stable_confident_brake_passes_full():
  la = _on()
  for _ in range(15):                                 # warm the track to stable (> 0.45 s)
    _shape_apply(la, radar(lead(0.0)))
  out = _shape_apply(la, radar(lead(-3.0)))
  assert out.leadOne.aLeadK == pytest.approx(-3.0)    # confident -> full weight, no discount


def test_persistent_brake_settles_to_full_weight():
  la = _on()
  first = _shape_apply(la, radar(lead(-3.0))).leadOne.aLeadK
  last = None
  for _ in range(14):                                 # keep braking; track stabilises + brake sustains
    last = _shape_apply(la, radar(lead(-3.0))).leadOne.aLeadK
  assert first == pytest.approx(min(-3.0 * DISCOUNT_FLOOR, -3.0 + AL_CAP_MAX_SOFTENING))
  assert last == pytest.approx(-3.0)                  # trusted once stable/sustained


def test_shaped_accel_invariant_in_raw_to_zero():
  # across confidences/frames a braking lead's shaped accel is always within [raw, 0]
  for raw_a in (-0.6, -2.0, -5.0):
    la = _on()
    for _ in range(20):
      a = _shape_apply(la, radar(lead(raw_a))).leadOne.aLeadK
      assert raw_a - 1e-9 <= a <= 0.0


def test_no_lead_is_passthrough_value():
  out = _shape_apply(_on(), radar(lead(-3.0, status=False)))
  assert out.leadOne.aLeadK == -3.0                   # status False -> untouched


def test_bool_only_missing_mode_is_shadow_passthrough():
  la = LeadAnticipation(FakeParams(LeadAnticipationEnabled=True, CustomLongitudinalEnabled=True))
  rs = radar(lead(-3.0))
  assert la.mode == "shadow"
  assert not la.enabled
  assert la.shape(rs, DT) is rs


def test_param_default_shadow_is_exact_passthrough():
  la = LeadAnticipation(FakeParams(CustomLongitudinalEnabled=True))
  rs = radar(lead(-3.0))
  assert la.mode == "shadow"
  assert la.shape(rs, DT) is rs


def test_explicit_apply_still_shapes_when_gates_allow():
  la = LeadAnticipation(FakeParams(LeadAnticipationMode="apply", CustomLongitudinalEnabled=True))
  out = _shape_apply(la, radar(lead(-3.0)))
  assert la.last_result and la.last_result["apply"] is True
  assert la.last_result["block_reason"] == ""
  assert out.leadOne.aLeadK == pytest.approx(min(-3.0 * DISCOUNT_FLOOR, -3.0 + AL_CAP_MAX_SOFTENING))


def test_invalid_mode_falls_back_to_passthrough():
  la = LeadAnticipation(FakeParams(LeadAnticipationMode="bogus", LeadAnticipationEnabled=True, CustomLongitudinalEnabled=True))
  rs = radar(lead(-3.0))
  assert la.shape(rs, DT) is rs
  assert la.last_result is None


def test_custom_long_disabled_blocks_apply():
  la = LeadAnticipation(FakeParams(LeadAnticipationMode="apply", CustomLongitudinalEnabled=False))
  rs = radar(lead(-3.0))
  assert la.shape(rs, DT) is rs
  assert la.last_result and la.last_result["apply"] is False
  assert la.last_result["block_reason"] == "mode_or_enabled"


def test_missing_context_is_fail_closed():
  """No context keyword arguments -> defaults are fail-closed; apply is blocked but shadow summary
  is still recorded."""
  la = LeadAnticipation(FakeParams(LeadAnticipationMode="apply", CustomLongitudinalEnabled=True))
  rs = radar(lead(-3.0))
  out = la.shape(rs, DT)                              # no long_active, v_ego, etc.
  assert out is rs
  assert la.last_result and la.last_result["apply"] is False
  assert la.last_result["block_reason"] == "long_inactive"
  assert la.last_result["leadOneShaped"] is not None


def test_apply_blocked_when_long_inactive():
  la = _on()
  rs = radar(lead(-3.0))
  out = _shape_apply(la, rs, long_active=False)
  assert out is rs
  assert la.last_result and la.last_result["apply"] is False
  assert la.last_result["block_reason"] == "long_inactive"


def test_apply_blocked_when_brake_pressed():
  la = _on()
  rs = radar(lead(-3.0))
  out = _shape_apply(la, rs, brake_pressed=True)
  assert out is rs
  assert la.last_result and la.last_result["apply"] is False
  assert la.last_result["block_reason"] == "brake_pressed"


def test_apply_blocked_when_gas_pressed():
  la = _on()
  rs = radar(lead(-3.0))
  out = _shape_apply(la, rs, gas_pressed=True)
  assert out is rs
  assert la.last_result["apply"] is False
  assert la.last_result["block_reason"] == "gas_pressed"


def test_apply_blocked_when_force_decel():
  la = _on()
  rs = radar(lead(-3.0))
  out = _shape_apply(la, rs, force_decel=True)
  assert out is rs
  assert la.last_result["apply"] is False
  assert la.last_result["block_reason"] == "force_decel"


def test_apply_blocked_at_low_speed():
  la = _on()
  rs = radar(lead(-3.0))
  out = _shape_apply(la, rs, v_ego=1.5)
  assert out is rs
  assert la.last_result["apply"] is False
  assert la.last_result["block_reason"] == "low_speed"


def test_apply_blocked_when_too_close():
  la = _on()
  rs = radar(lead(-3.0, d_rel=7.0))
  out = _shape_apply(la, rs, v_ego=15.0)
  assert out is rs
  assert la.last_result["apply"] is False
  assert la.last_result["block_reason"] == "lead_0_too_close"


def test_apply_blocked_when_fast_closing():
  la = _on()
  rs = radar(lead(-3.0, v_rel=-5.0))
  out = _shape_apply(la, rs, v_ego=15.0)
  assert out is rs
  assert la.last_result["apply"] is False
  assert la.last_result["block_reason"] == "lead_0_fast_closing"


def test_apply_blocked_when_lead_two_too_close():
  la = _on()
  rs = radar(lead(-3.0, d_rel=30.0), lead(-2.0, d_rel=5.0, tid=4))
  out = _shape_apply(la, rs, v_ego=15.0)
  assert out is rs
  assert la.last_result["apply"] is False
  assert la.last_result["block_reason"] == "lead_1_too_close"


def test_apply_blocked_when_lead_two_fast_closing():
  la = _on()
  rs = radar(lead(-3.0, d_rel=30.0), lead(-2.0, v_rel=-6.0, tid=4))
  out = _shape_apply(la, rs, v_ego=15.0)
  assert out is rs
  assert la.last_result["apply"] is False
  assert la.last_result["block_reason"] == "lead_1_fast_closing"


def test_apply_blocked_when_low_ttc_even_if_not_fast_closing():
  # closing at 3 m/s with 8.5 m gap -> TTC ~2.8 s (< 3 s), but v_rel=-3 > -4 so not fast-closing.
  la = _on()
  rs = radar(lead(-3.0, d_rel=8.5, v_rel=-3.0))
  out = _shape_apply(la, rs, v_ego=10.0)
  assert out is rs
  assert la.last_result["apply"] is False
  assert la.last_result["block_reason"] == "lead_0_low_ttc"


def test_apply_blocked_when_v_ego_non_finite():
  la = _on()
  rs = radar(lead(-3.0))
  out = _shape_apply(la, rs, v_ego=float("nan"))
  assert out is rs
  assert la.last_result["apply"] is False
  assert la.last_result["block_reason"] == "low_speed"


def test_apply_blocked_when_d_rel_non_finite():
  la = _on()
  rs = radar(lead(-3.0, d_rel=float("inf")))
  out = _shape_apply(la, rs, v_ego=15.0)
  assert out is rs
  assert la.last_result["apply"] is False
  assert la.last_result["block_reason"] == "lead_0_non_finite"


def test_apply_blocked_when_v_rel_non_finite():
  la = _on()
  rs = radar(lead(-3.0, v_rel=float("nan")))
  out = _shape_apply(la, rs, v_ego=15.0)
  assert out is rs
  assert la.last_result["apply"] is False
  assert la.last_result["block_reason"] == "lead_0_non_finite"


def test_apply_passes_through_distance_time_gap():
  # At v_ego = 20 m/s, the distance gate is max(8, 0.8*20=16) = 16 m. d_rel=17 should allow apply.
  la = _on()
  rs = radar(lead(-3.0, d_rel=17.0))
  out = _shape_apply(la, rs, v_ego=20.0)
  assert out is not rs
  assert la.last_result and la.last_result["apply"] is True
