"""Behavioral tests for the lead confidence / flicker tracker (faithful port)."""
from __future__ import annotations

from types import SimpleNamespace

from openpilot.sunnypilot.custom.longitudinal.lead_confidence import (
  LeadConfidenceTracker,
  adjust_new_lead_accel,
)


def lead(d_rel=30.0, v_lead=15.0, y_rel=0.0, track_id=7, radar=True, model_prob=0.9, status=True):
  return SimpleNamespace(status=status, radarTrackId=track_id, dRel=d_rel, vLeadK=v_lead,
                         yRel=y_rel, radar=radar, modelProb=model_prob)


def test_no_lead_is_inactive():
  t = LeadConfidenceTracker()
  s = t.update(None, 0.1)
  assert s.status is False
  assert s.new_lead is False


def test_new_lead_then_stabilizes():
  t = LeadConfidenceTracker()
  first = t.update(lead(), 0.1)
  assert first.status is True
  assert first.new_lead is True
  assert first.age == 0.0
  last = first
  for _ in range(12):
    last = t.update(lead(), 0.1)
  assert last.new_lead is False
  assert last.stable is True
  assert last.accel_blend == 1.0
  assert last.speed_trusted is True  # radar lead


def test_adjust_new_lead_accel_blends_only_positive():
  state_half = type(LeadConfidenceTracker().update(None, 0.1))(accel_blend=0.5)
  assert adjust_new_lead_accel(2.0, state_half) == 1.0   # positive scaled by blend
  assert adjust_new_lead_accel(-2.0, state_half) == -2.0  # braking never attenuated


def test_continuity_breaks_on_large_jump():
  t = LeadConfidenceTracker()
  for _ in range(5):
    t.update(lead(d_rel=30.0, track_id=7), 0.1)
  # large dRel jump with a different track -> not continuous -> new lead
  jumped = t.update(lead(d_rel=60.0, track_id=9), 0.1)
  assert jumped.new_lead is True
  assert jumped.age == 0.0


def test_flicker_guard_triggers_on_repeated_toggling():
  t = LeadConfidenceTracker()
  s = None
  for i in range(8):
    s = t.update(lead() if i % 2 == 0 else None, 0.1)
  # several status transitions inside the flicker window arm the flicker guard
  assert s.flicker_guard_timer > 0.0
