from types import SimpleNamespace

import pytest

from openpilot.selfdrive.controls.lib.lead_confidence import (
  LeadConfidenceTracker,
  NEW_LEAD_GUARD_TIME,
  NEW_LEAD_STABLE_TIME,
  adjust_new_lead_accel,
)


def make_lead(status=True, track_id=1, d_rel=20.0, v_lead=10.0, y_rel=0.0, a_lead=0.0, radar=True, model_prob=1.0):
  return SimpleNamespace(
    status=status,
    radarTrackId=track_id,
    dRel=d_rel,
    vLeadK=v_lead,
    vLead=v_lead,
    yRel=y_rel,
    aLeadK=a_lead,
    radar=radar,
    modelProb=model_prob,
  )


def test_new_lead_trusts_speed_but_suppresses_positive_accel():
  tracker = LeadConfidenceTracker()

  state = tracker.update(make_lead(a_lead=1.0), dt=0.1)

  assert state.status
  assert state.new_lead
  assert state.speed_trusted
  assert state.accel_blend == pytest.approx(0.0)
  assert state.guard_timer == pytest.approx(NEW_LEAD_GUARD_TIME)
  assert adjust_new_lead_accel(1.0, state) == pytest.approx(0.0)
  assert adjust_new_lead_accel(-1.0, state) == pytest.approx(-1.0)


def test_same_track_ages_into_stable_positive_accel():
  tracker = LeadConfidenceTracker()
  lead = make_lead(a_lead=0.8)

  state = tracker.update(lead, dt=0.1)
  while state.age < NEW_LEAD_STABLE_TIME:
    state = tracker.update(lead, dt=0.1)

  assert not state.new_lead
  assert state.stable
  assert state.accel_blend == pytest.approx(1.0)
  assert state.guard_timer == pytest.approx(0.0)
  assert adjust_new_lead_accel(0.8, state) == pytest.approx(0.8)


def test_discontinuous_track_change_resets_confidence():
  tracker = LeadConfidenceTracker()
  lead = make_lead(track_id=1, d_rel=20.0, v_lead=10.0)

  state = tracker.update(lead, dt=0.1)
  while state.age < NEW_LEAD_STABLE_TIME:
    state = tracker.update(lead, dt=0.1)

  state = tracker.update(make_lead(track_id=2, d_rel=35.0, v_lead=5.0), dt=0.1)

  assert state.new_lead
  assert not state.stable
  assert state.age == pytest.approx(0.0)
  assert state.guard_timer == pytest.approx(NEW_LEAD_GUARD_TIME)


def test_benign_track_churn_preserves_confidence():
  tracker = LeadConfidenceTracker()
  lead = make_lead(track_id=1, d_rel=20.0, v_lead=8.0, y_rel=1.7)

  state = tracker.update(lead, dt=0.1)
  while state.age < NEW_LEAD_STABLE_TIME:
    state = tracker.update(lead, dt=0.1)
  state = tracker.update(make_lead(track_id=2, d_rel=20.3, v_lead=8.2, y_rel=1.8), dt=0.1)

  assert not state.new_lead
  assert state.stable
  assert state.accel_blend == pytest.approx(1.0)


def test_centered_track_churn_resets_confidence():
  tracker = LeadConfidenceTracker()
  lead = make_lead(track_id=1, d_rel=20.0, v_lead=8.0, y_rel=0.0)

  state = tracker.update(lead, dt=0.1)
  while state.age < NEW_LEAD_STABLE_TIME:
    state = tracker.update(lead, dt=0.1)
  state = tracker.update(make_lead(track_id=2, d_rel=20.3, v_lead=8.2, y_rel=0.1), dt=0.1)

  assert state.new_lead
  assert not state.stable
  assert state.accel_blend == pytest.approx(0.0)
  assert state.guard_timer == pytest.approx(NEW_LEAD_GUARD_TIME)


def test_radarless_discontinuous_lead_resets_confidence():
  tracker = LeadConfidenceTracker()
  lead = make_lead(track_id=-1, d_rel=25.0, v_lead=12.0, radar=False, model_prob=0.9)

  state = tracker.update(lead, dt=0.1)
  while state.age < NEW_LEAD_STABLE_TIME:
    state = tracker.update(lead, dt=0.1)
  state = tracker.update(make_lead(track_id=-1, d_rel=12.0, v_lead=4.0, radar=False, model_prob=0.9), dt=0.1)

  assert state.new_lead
  assert state.accel_blend == pytest.approx(0.0)
  assert state.guard_timer == pytest.approx(NEW_LEAD_GUARD_TIME)
