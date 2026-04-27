from types import SimpleNamespace

from openpilot.tools.drive_lab.timeline import render_summary, select_event_time, summarize_window


class FakeMsg(SimpleNamespace):
  def which(self):
    return self.kind


def msg(kind, t_s, **payload):
  return FakeMsg(kind=kind, logMonoTime=int(t_s * 1e9), **{kind: SimpleNamespace(**payload)})


def raw_msg(kind, t_s, payload):
  return FakeMsg(kind=kind, logMonoTime=int(t_s * 1e9), **{kind: payload})


def test_select_event_time_uses_latest_bookmark_without_requested_time():
  msgs = [msg("carState", 0.0), msg("userBookmark", 5.0), msg("userBookmark", 9.0)]

  assert select_event_time(msgs, nearest_bookmark=True) == 9.0


def test_summarize_window_tracks_planner_and_lead_changes():
  msgs = [
    msg("carState", 0.0, vEgo=10.0, vCruise=50.0, brakePressed=False, gasPressed=False, standstill=False),
    msg("radarState", 1.0, leadOne=SimpleNamespace(status=False, dRel=200.0, vRel=0.0)),
    msg("userBookmark", 2.0),
    msg("radarState", 2.5, leadOne=SimpleNamespace(status=True, dRel=25.0, vRel=-3.0)),
    msg("longitudinalPlan", 3.0, longitudinalPlanSource="lead0", shouldStop=False, fcw=False, aTarget=-0.8),
  ]

  summary = summarize_window(msgs, 2.0, 2.0, 2.0)
  rendered = render_summary(summary)

  assert "user bookmark" in rendered
  assert "leadOne status: True" in rendered
  assert "plan source: lead0" in rendered
  assert "aTarget" in rendered


def test_summarize_window_reads_list_shaped_onroad_events():
  msgs = [
    msg("carState", 0.0, vEgo=10.0),
    raw_msg("onroadEvents", 1.0, [SimpleNamespace(name="fcw")]),
  ]

  summary = summarize_window(msgs, 1.0, 1.0, 1.0)
  rendered = render_summary(summary)

  assert "events: fcw" in rendered
