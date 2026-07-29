"""Regression tests for the lead-following profiler / overreaction metrics."""
from __future__ import annotations

from types import SimpleNamespace

from openpilot.tools.drive_lab.profile_lead_following import (
  LeadFollowParams,
  _FollowSample,
  analyze_route,
)


class FakeMsg(SimpleNamespace):
  def which(self):
    return self.kind


def msg(kind, t_s, **payload):
  return FakeMsg(kind=kind, logMonoTime=int(t_s * 1e9), **{kind: SimpleNamespace(**payload)})


def stream(rows):
  """rows: (vEgo, aEgo, dRel, vRel, vLead, aLeadK, longActive, leadStatus)."""
  msgs = []
  for i, (vego, aego, d_rel, v_rel, v_lead, a_lead, la, ls) in enumerate(rows):
    t = i * 0.05
    msgs.append(msg("carState", t, vEgo=vego, aEgo=aego))
    msgs.append(msg("carControl", t + 0.0005, longActive=la))
    msgs.append(msg("radarState", t + 0.001,
                    leadOne=SimpleNamespace(present=ls, dRel=d_rel, vRel=v_rel, vLead=v_lead,
                                            aLeadK=a_lead, modelProb=0.9)))
  return msgs


def _series(*samples: _FollowSample) -> list[SimpleNamespace]:
  """Build a minimal msg list from follow samples; assumes 20 Hz spacing."""
  msgs: list[SimpleNamespace] = []
  for i, s in enumerate(samples):
    t = i * 0.05
    msgs.append(msg("carState", t, vEgo=s.v, aEgo=s.a))
    msgs.append(msg("carControl", t + 0.0005, longActive=True))
    msgs.append(msg("radarState", t + 0.001,
                    leadOne=SimpleNamespace(present=True, dRel=s.d_rel, vRel=s.v_rel,
                                            vLead=s.v_lead, aLeadK=s.a_lead, modelProb=0.9)))
  return msgs


def test_steady_following_headway_and_gap():
  report = analyze_route(stream([(20.0, 0.0, 44.0, 0.0, 20.0, 0.0, True, True)] * 40), source="steady")
  assert report.follow_samples == 40
  assert report.steady_samples == 40
  assert report.thw_median is not None
  assert 2.1 < report.thw_median < 2.3        # 44 m / 20 m/s
  assert report.gap_median == 44.0
  assert report.thw_share_above_2s == 1.0


def test_approach_with_lead_resume_is_counted():
  rows = [(20.0, 0.0, 44.0, 0.0, 20.0, 0.0, True, True)] * 10            # steady
  rows += [(20.0, -1.0, 40.0, -3.0, 17.0, -1.0, True, True),            # closing, lead slow
           (20.0, -1.0, 38.0, -3.0, 17.0, 0.5, True, True),
           (20.0, -1.0, 37.0, -2.0, 18.0, 1.0, True, True),
           (20.0, -0.5, 36.0, -1.0, 19.0, 1.0, True, True)]             # lead climbing, still closing
  rows += [(20.0, 0.0, 36.0, 0.0, 20.0, 0.0, True, True)] * 3           # episode ends (vRel >= -0.5)
  report = analyze_route(stream(rows), source="approach")
  assert report.approach_events == 1
  assert report.lead_resumed == 1               # lead dipped to 17 then climbed to 19 mid-approach
  assert report.peak_decel_median is not None and report.peak_decel_median < 0.0


def test_not_engaged_is_ignored():
  report = analyze_route(stream([(20.0, 0.0, 44.0, 0.0, 20.0, 0.0, False, True)] * 20), source="manual")
  assert report.follow_samples == 0
  assert any("cruise-following" in note for note in report.notes)


def test_no_lead_is_ignored():
  report = analyze_route(stream([(20.0, 0.0, 44.0, 0.0, 20.0, 0.0, True, False)] * 20), source="leadless")
  assert report.follow_samples == 0


# -----------------------------------------------------------------------------
# Phase 3 / Phase 4 overreaction regression metrics
# -----------------------------------------------------------------------------

def test_stable_inside_gap_does_not_brake_for_lead_that_is_accelerating():
  """Regression for Phase 3: stable following with lead slightly inside gap and accelerating
  should not produce braking-while-lead-accelerating samples."""
  samples = []
  for _ in range(40):
    # Lead inside desired gap but opening gently; ego should coast, not brake.
    samples.append(_FollowSample(v=20.0, a=-0.1, d_rel=28.0, v_rel=0.2, v_lead=20.2, a_lead=0.3))
  report = analyze_route(_series(*samples), source="synthetic", params=LeadFollowParams())
  assert report.brake_samples == 0 or report.brake_lead_accel_frac == 0.0


def test_gentle_closing_inside_gap_is_tracked_as_braking():
  """Phase 3: gentle closing inside the gap is counted as braking, but the policy fix caps it."""
  samples = []
  for _ in range(20):
    samples.append(_FollowSample(v=20.0, a=-0.6, d_rel=28.0, v_rel=-0.2, v_lead=19.8, a_lead=0.0))
  report = analyze_route(_series(*samples), source="synthetic", params=LeadFollowParams())
  # v_rel=-0.2 is not an approach episode (needs <-1.0), but aEgo=-0.6 counts as braking.
  assert report.brake_samples == 20
  assert report.brake_lead_accel_frac == 0.0


def test_lead_resumed_counts_false_approach():
  """The profiler flags an approach where the lead accelerates while still closing."""
  samples: list[_FollowSample] = []
  # steady
  for _ in range(10):
    samples.append(_FollowSample(v=20.0, a=0.0, d_rel=40.0, v_rel=0.0, v_lead=20.0, a_lead=0.0))
  # closing, then lead starts speeding back up while v_rel is still < -0.5
  for _ in range(3):
    samples.append(_FollowSample(v=20.0, a=-0.5, d_rel=35.0, v_rel=-2.0, v_lead=18.0, a_lead=-0.5))
  for _ in range(5):
    samples.append(_FollowSample(v=20.0, a=-0.2, d_rel=34.0, v_rel=-0.8, v_lead=19.2, a_lead=0.6))
  report = analyze_route(_series(*samples), source="synthetic", params=LeadFollowParams())
  assert report.approach_events == 1
  assert report.lead_resumed == 1
  assert report.lead_resumed_frac == 1.0
