from types import SimpleNamespace

from openpilot.tools.drive_lab.profile_lead_following import analyze_route


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
                    leadOne=SimpleNamespace(status=ls, dRel=d_rel, vRel=v_rel, vLead=v_lead,
                                            aLeadK=a_lead, modelProb=0.9)))
  return msgs


def test_steady_following_headway_and_gap():
  report = analyze_route(stream([(20.0, 0.0, 44.0, 0.0, 20.0, 0.0, True, True)] * 40), source="steady")
  assert report.follow_samples == 40
  assert report.steady_samples == 40
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
