"""§3 lead-motion anticipation: confidence-shape the radar lead's accel before the MPC. Safety: only
ever make a *braking* lead look *less* braking (shaped aLeadK in [raw, 0]); never touch a non-braking,
high-confidence, or sustained brake; opt-in; fail-closed."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from openpilot.sunnypilot.custom.longitudinal.lead_anticipation import (
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


def lead(a_lead, d_rel=30.0, v_lead=18.0, status=True, tid=3):
  return SimpleNamespace(status=status, dRel=d_rel, vLead=v_lead, vLeadK=v_lead, aLeadK=a_lead,
                         aLeadTau=1.5, yRel=0.0, radarTrackId=tid, radar=True, modelProb=0.9)


def radar(l1, l2=None):
  return SimpleNamespace(leadOne=l1, leadTwo=l2)


def _on():
  return LeadAnticipation(FakeParams(LeadAnticipationMode="apply", CustomLongitudinalEnabled=True))


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
  assert la.last_result["leadOneRaw"] == pytest.approx(-3.0)
  assert la.last_result["leadOneShaped"] == pytest.approx(-3.0 * DISCOUNT_FLOOR)


def test_non_braking_lead_unchanged():
  out = _on().shape(radar(lead(0.5)), DT)             # accelerating lead
  assert out.leadOne.aLeadK == 0.5


def test_new_low_confidence_brake_is_discounted():
  out = _on().shape(radar(lead(-3.0)), DT)            # first frame -> new lead -> low confidence
  a = out.leadOne.aLeadK
  assert a == pytest.approx(-3.0 * DISCOUNT_FLOOR)    # floored discount toward 0
  assert -3.0 < a <= 0.0                              # less negative than raw, never positive
  # the rest of the lead is proxied unchanged
  assert out.leadOne.dRel == 30.0 and out.leadOne.vLead == 18.0


def test_stable_confident_brake_passes_full():
  la = _on()
  for _ in range(15):                                 # warm the track to stable (> 0.45 s)
    la.shape(radar(lead(0.0)), DT)
  out = la.shape(radar(lead(-3.0)), DT)
  assert out.leadOne.aLeadK == pytest.approx(-3.0)    # confident -> full weight, no discount


def test_persistent_brake_settles_to_full_weight():
  la = _on()
  first = la.shape(radar(lead(-3.0)), DT).leadOne.aLeadK
  last = None
  for _ in range(14):                                 # keep braking; track stabilises + brake sustains
    last = la.shape(radar(lead(-3.0)), DT).leadOne.aLeadK
  assert first == pytest.approx(-1.5)                 # discounted at first (unconfirmed)
  assert last == pytest.approx(-3.0)                  # trusted once stable/sustained


def test_shaped_accel_invariant_in_raw_to_zero():
  # across confidences/frames a braking lead's shaped accel is always within [raw, 0]
  for raw_a in (-0.6, -2.0, -5.0):
    la = _on()
    for _ in range(20):
      a = la.shape(radar(lead(raw_a)), DT).leadOne.aLeadK
      assert raw_a - 1e-9 <= a <= 0.0


def test_no_lead_is_passthrough_value():
  out = _on().shape(radar(lead(-3.0, status=False)), DT)
  assert out.leadOne.aLeadK == -3.0                   # status False -> untouched


def test_bool_compatibility_when_mode_missing():
  la = LeadAnticipation(FakeParams(LeadAnticipationEnabled=True, CustomLongitudinalEnabled=True))
  out = la.shape(radar(lead(-3.0)), DT)
  assert out.leadOne.aLeadK == pytest.approx(-3.0 * DISCOUNT_FLOOR)


def test_param_default_shadow_is_exact_passthrough():
  la = LeadAnticipation(FakeParams(CustomLongitudinalEnabled=True))
  rs = radar(lead(-3.0))
  assert la.mode == "shadow"
  assert la.shape(rs, DT) is rs


def test_bool_compatibility_wins_when_mode_missing_even_if_default_exists():
  # Real Params.get(key) returns None for an unset key unless return_default=True is requested; this
  # pins compatibility before the code-level default shadow fallback.
  la = LeadAnticipation(FakeParams(LeadAnticipationEnabled=True, CustomLongitudinalEnabled=True))
  assert la.mode == "apply"


def test_invalid_mode_falls_back_to_passthrough():
  la = LeadAnticipation(FakeParams(LeadAnticipationMode="bogus", LeadAnticipationEnabled=True, CustomLongitudinalEnabled=True))
  rs = radar(lead(-3.0))
  assert la.shape(rs, DT) is rs


def test_custom_long_disabled_blocks_apply():
  la = LeadAnticipation(FakeParams(LeadAnticipationMode="apply", CustomLongitudinalEnabled=False))
  rs = radar(lead(-3.0))
  assert la.shape(rs, DT) is rs
  assert la.last_result and la.last_result["apply"] is False
