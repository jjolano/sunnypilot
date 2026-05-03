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


def test_select_event_time_handles_unsorted_bookmarks():
  msgs = [msg("userBookmark", 9.0), msg("carState", 0.0), msg("userBookmark", 5.0)]

  assert select_event_time(msgs, nearest_bookmark=True) == 9.0
  assert select_event_time(msgs, requested_time_s=4.8, nearest_bookmark=True) == 5.0


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


def test_summarize_window_attributes_lead_braking():
  msgs = [
    msg("carState", 0.0, vEgo=12.0, brakePressed=False, gasPressed=False),
    msg("radarState", 1.0, leadOne=SimpleNamespace(status=True, dRel=24.8, vRel=-2.0)),
    msg("longitudinalPlan", 1.5, longitudinalPlanSource="lead0", shouldStop=False, fcw=False, aTarget=-0.8),
  ]

  rendered = render_summary(summarize_window(msgs, 1.0, 1.0, 1.0))

  assert "Attribution:" in rendered
  assert "likely cause: lead" in rendered
  assert "planner source lead0" in rendered
  assert "lead gap min 24.800 m" in rendered
  assert "aTarget min -0.800 m/s^2" in rendered


def test_summarize_window_attributes_active_radar_lead_braking():
  msgs = [
    msg("carState", 0.0, vEgo=12.0, brakePressed=False, gasPressed=False),
    msg("radarState", 0.8, leadOne=SimpleNamespace(status=True, dRel=18.4, vRel=-1.0)),
    msg("longitudinalPlan", 1.0, longitudinalPlanSource="cruise", shouldStop=False, fcw=False, aTarget=-0.4),
  ]

  rendered = render_summary(summarize_window(msgs, 1.0, 1.0, 1.0))

  assert "likely cause: lead" in rendered
  assert "planner source cruise" in rendered
  assert "lead gap min 18.400 m" in rendered
  assert "aTarget min -0.400 m/s^2" in rendered


def test_summarize_window_attributes_planner_source_with_lead_but_no_braking():
  msgs = [
    msg("carState", 0.0, vEgo=12.0, brakePressed=False, gasPressed=False),
    msg("radarState", 0.8, leadOne=SimpleNamespace(status=True, dRel=18.4, vRel=-1.0)),
    msg("longitudinalPlan", 1.0, longitudinalPlanSource="cruise", shouldStop=False, fcw=False, aTarget=0.0),
  ]

  rendered = render_summary(summarize_window(msgs, 1.0, 1.0, 1.0))

  assert "likely cause: planner_source" in rendered
  assert "planner source cruise" in rendered


def test_summarize_window_attributes_model_action_stop():
  msgs = [
    msg("carState", 0.0, vEgo=8.0, brakePressed=False, gasPressed=False),
    msg("modelV2", 1.0, action=SimpleNamespace(desiredAcceleration=-1.2, shouldStop=True)),
    msg("longitudinalPlan", 1.2, longitudinalPlanSource="model", shouldStop=False, fcw=False, aTarget=-1.0),
  ]

  rendered = render_summary(summarize_window(msgs, 1.0, 1.0, 1.0))

  assert "likely cause: model_stop" in rendered
  assert "model action shouldStop true" in rendered
  assert "planner source model" in rendered


def test_summarize_window_attributes_plan_should_stop():
  msgs = [
    msg("carState", 0.0, vEgo=8.0, brakePressed=False, gasPressed=False),
    msg("longitudinalPlan", 1.0, longitudinalPlanSource="e2e", shouldStop=True, fcw=False, aTarget=-1.0),
  ]

  rendered = render_summary(summarize_window(msgs, 1.0, 1.0, 1.0))

  assert "likely cause: model_stop" in rendered
  assert "plan shouldStop true" in rendered
  assert "planner source e2e" in rendered


def test_summarize_window_does_not_attribute_cruise_plan_should_stop_to_model_stop():
  msgs = [
    msg("carState", 0.0, vEgo=8.0, brakePressed=False, gasPressed=False),
    msg("longitudinalPlan", 1.0, longitudinalPlanSource="cruise", shouldStop=True, fcw=False, aTarget=-1.0),
  ]

  rendered = render_summary(summarize_window(msgs, 1.0, 1.0, 1.0))

  assert "likely cause: planner_source" in rendered
  assert "likely cause: model_stop" not in rendered
  assert "planner source cruise" in rendered


def test_summarize_window_attributes_same_time_radar_lead_braking_after_plan():
  msgs = [
    msg("carState", 0.0, vEgo=12.0, brakePressed=False, gasPressed=False),
    msg("longitudinalPlan", 1.0, longitudinalPlanSource="cruise", shouldStop=False, fcw=False, aTarget=-0.4),
    msg("radarState", 1.0, leadOne=SimpleNamespace(status=True, dRel=18.4, vRel=-1.0)),
  ]

  rendered = render_summary(summarize_window(msgs, 1.0, 1.0, 1.0))

  assert "likely cause: lead" in rendered
  assert "planner source cruise" in rendered
  assert "lead gap min 18.400 m" in rendered
  assert "aTarget min -0.400 m/s^2" in rendered


def test_summarize_window_does_not_attribute_stale_lead_to_later_braking():
  msgs = [
    msg("carState", 0.0, vEgo=12.0, brakePressed=False, gasPressed=False),
    msg("radarState", 0.2, leadOne=SimpleNamespace(status=True, dRel=18.4, vRel=-1.0)),
    msg("radarState", 0.8, leadOne=SimpleNamespace(status=False, dRel=200.0, vRel=0.0)),
    msg("longitudinalPlan", 1.0, longitudinalPlanSource="cruise", shouldStop=False, fcw=False, aTarget=-0.4),
  ]

  rendered = render_summary(summarize_window(msgs, 1.0, 1.0, 1.0))

  assert "likely cause: planner_source" in rendered
  assert "planner source cruise" in rendered
  assert "aTarget min -0.400 m/s^2" in rendered


def test_summarize_window_attributes_planner_source_fallback():
  msgs = [
    msg("carState", 0.0, vEgo=12.0, brakePressed=False, gasPressed=False),
    msg("longitudinalPlan", 1.0, longitudinalPlanSource="cruise", shouldStop=False, fcw=False, aTarget=0.1),
  ]

  rendered = render_summary(summarize_window(msgs, 1.0, 1.0, 1.0))

  assert "likely cause: planner_source" in rendered
  assert "planner source cruise" in rendered


def test_summarize_window_attributes_unknown_when_no_longitudinal_signals():
  msgs = [msg("carState", 0.0, vEgo=8.0, brakePressed=False, gasPressed=False)]

  rendered = render_summary(summarize_window(msgs, 0.0, 1.0, 1.0))

  assert "likely cause: unknown" in rendered
  assert "no longitudinal attribution signals found" in rendered


def test_summarize_window_unknown_keeps_available_lead_evidence():
  msgs = [
    msg("carState", 0.0, vEgo=8.0, brakePressed=False, gasPressed=False),
    msg("radarState", 1.0, leadOne=SimpleNamespace(status=True, dRel=31.2, vRel=-0.5)),
  ]

  rendered = render_summary(summarize_window(msgs, 1.0, 1.0, 1.0))

  assert "likely cause: unknown" in rendered
  assert "lead gap min 31.200 m" in rendered
  assert "no longitudinal attribution signals found" not in rendered


def test_summarize_window_reads_list_shaped_onroad_events():
  msgs = [
    msg("carState", 0.0, vEgo=10.0),
    raw_msg("onroadEvents", 1.0, [SimpleNamespace(name="fcw")]),
  ]

  summary = summarize_window(msgs, 1.0, 1.0, 1.0)
  rendered = render_summary(summary)

  assert "events: fcw" in rendered
