"""Tests for the lead context risk/progress model (faithful port).

Covers the pure risk/progress functions (deterministic, no engaged data needed) plus a
tracker smoke test. The model's *policy use* is validated downstream against the corpus.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

from openpilot.sunnypilot.custom.longitudinal import lead_context as lc
from openpilot.sunnypilot.custom.longitudinal.lead_confidence import LeadConfidenceState


def test_lead_prediction_fields_finite():
  p = lc.lead_prediction(d_rel=30.0, v_lead=10.0, a_lead=0.5, v_ego=20.0)
  assert isinstance(p, lc.LeadTrajectoryPrediction)
  assert p.valid is True
  assert len(p.x) == len(p.v) == len(p.a) > 0
  for seq in (p.x, p.v, p.a):
    assert all(math.isfinite(float(val)) for val in seq)
  # closing lead (v_ego 20 > v_lead 10): predicted relative gap shrinks over the horizon
  assert p.x[-1] <= p.x[0]


def test_required_decel_zero_when_not_closing():
  assert lc._required_decel(d_rel=20.0, v_rel=2.0) == 0.0   # opening
  assert lc._required_decel(d_rel=20.0, v_rel=0.0) == 0.0


def test_required_decel_monotonic_in_closing_rate():
  slow = lc._required_decel(d_rel=20.0, v_rel=-3.0)
  fast = lc._required_decel(d_rel=20.0, v_rel=-8.0)
  assert fast > slow > 0.0


def test_required_decel_monotonic_in_gap():
  near = lc._required_decel(d_rel=10.0, v_rel=-5.0)
  far = lc._required_decel(d_rel=40.0, v_rel=-5.0)
  assert near > far > 0.0


def test_ttc_finite_when_closing_large_otherwise():
  assert lc._ttc(d_rel=20.0, v_rel=-4.0) == 5.0
  assert lc._ttc(d_rel=20.0, v_rel=2.0) >= 1e3  # opening -> effectively infinite


def test_time_gap_and_progress_gap():
  assert lc._time_gap(d_rel=30.0, v_ego=15.0) == 2.0
  assert lc._desired_progress_gap(v_ego=20.0) > 0.0


def test_gap_shortage_and_excess_complement():
  v_ego = 20.0
  desired = lc._desired_progress_gap(v_ego)
  # shortage when closer than desired, excess when farther
  assert lc._gap_shortage(d_rel=desired - 5.0, v_ego=v_ego) > 0.0
  assert lc._gap_excess(d_rel=desired + 5.0, v_ego=v_ego) > 0.0
  assert lc._gap_shortage(d_rel=desired + 5.0, v_ego=v_ego) == 0.0


def test_on_path_score_in_unit_interval_and_decreasing():
  scores = [lc._on_path_score(y) for y in (0.0, 0.5, 1.0, 2.0, 3.0, 5.0)]
  assert all(0.0 <= s <= 1.0 for s in scores)
  assert scores[0] == 1.0
  for a, b in zip(scores, scores[1:]):
    assert b <= a + 1e-9


def test_risk_score_bounded():
  for v_rel in (-10.0, -3.0, 0.0, 3.0):
    rd = lc._required_decel(20.0, v_rel)
    ttc = lc._ttc(20.0, v_rel)
    tg = lc._time_gap(20.0, 20.0)
    s = lc._risk_score(d_rel=20.0, v_rel=v_rel, v_lead=10.0, v_ego=20.0, required_decel=rd, ttc=ttc, time_gap=tg)
    assert 0.0 <= s <= 1.0


def lead(d_rel=30.0, v_lead=12.0, a_lead=0.0, y_rel=0.0, status=True, track_id=3):
  return SimpleNamespace(status=status, dRel=d_rel, vLead=v_lead, vLeadK=v_lead, aLeadK=a_lead,
                         yRel=y_rel, radarTrackId=track_id, radar=True, modelProb=0.9, aLeadTau=1.0)


def test_tracker_update_returns_primary_context():
  t = lc.LeadContextTracker()
  ctx = None
  for _ in range(10):
    ctx = t.update(
      leads=(lead(d_rel=30.0), None),
      confidence_states=(LeadConfidenceState(status=True, stable=True, accel_blend=1.0), LeadConfidenceState()),
      v_ego=20.0, dt=0.05,
    )
  assert isinstance(ctx, lc.PrimaryLeadContext)
