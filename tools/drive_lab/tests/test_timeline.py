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

  rendered = render_summary(summarize_window(msgs, 1.5, 1.0, 1.0))

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


def test_summarize_window_attributes_driver_override_before_planner_sources():
  msgs = [
    msg("carState", 1.0, vEgo=12.0, brakePressed=True, gasPressed=False),
    msg("radarState", 0.5, leadOne=SimpleNamespace(status=True, dRel=20.0, vRel=-3.0)),
    msg("longitudinalPlan", 1.0, longitudinalPlanSource="lead0", shouldStop=False, fcw=False, aTarget=-1.0),
  ]

  rendered = render_summary(summarize_window(msgs, 1.0, 1.0, 1.0))

  assert "likely cause: driver" in rendered
  assert "driver brake pressed" in rendered
  assert "planner source lead0" in rendered


def test_summarize_window_ignores_stale_driver_brake_before_event_local_car_state():
  msgs = [
    msg("carState", 0.1, vEgo=12.0, brakePressed=True, gasPressed=False),
    msg("radarState", 0.5, leadOne=SimpleNamespace(status=True, dRel=20.0, vRel=-3.0)),
    msg("carState", 1.0, vEgo=12.0, brakePressed=False, gasPressed=False),
    msg("longitudinalPlan", 1.0, longitudinalPlanSource="lead0", shouldStop=False, fcw=False, aTarget=-1.0),
  ]

  rendered = render_summary(summarize_window(msgs, 1.0, 1.0, 1.0))

  assert "likely cause: lead" in rendered
  assert "driver brake pressed" not in rendered
  assert "planner source lead0" in rendered


def test_summarize_window_ignores_post_event_driver_brake_when_prior_car_state_is_clear():
  msgs = [
    msg("carState", 0.0, vEgo=12.0, brakePressed=False, gasPressed=False),
    msg("carState", 0.8, vEgo=12.0, brakePressed=False, gasPressed=False),
    msg("longitudinalPlan", 1.0, longitudinalPlanSource="cruise", shouldStop=False, fcw=False, aTarget=0.1),
    msg("carState", 1.1, vEgo=12.0, brakePressed=True, gasPressed=False),
  ]

  rendered = render_summary(summarize_window(msgs, 1.0, 1.0, 1.0))

  assert "likely cause: planner_source" in rendered
  assert "driver brake pressed" not in rendered
  assert "planner source cruise" in rendered


def test_summarize_window_ignores_future_driver_brake_without_prior_car_state():
  msgs = [
    msg("userBookmark", 0.0),
    msg("radarState", 0.5, leadOne=SimpleNamespace(status=True, dRel=20.0, vRel=-3.0)),
    msg("longitudinalPlan", 1.0, longitudinalPlanSource="lead0", shouldStop=False, fcw=False, aTarget=-1.0),
    msg("carState", 1.1, vEgo=12.0, brakePressed=True, gasPressed=False),
  ]

  rendered = render_summary(summarize_window(msgs, 1.0, 1.0, 1.0))

  assert "likely cause: lead" in rendered
  assert "driver brake pressed" not in rendered
  assert "planner source lead0" in rendered


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


def test_summarize_window_attributes_speed_limit_assist_when_active():
  msgs = [
    msg("carState", 0.0, vEgo=15.0, brakePressed=False, gasPressed=False),
    msg(
      "longitudinalPlanSP",
      1.0,
      longitudinalPlanSource="speedLimitAssist",
      speedLimit=SimpleNamespace(assist=SimpleNamespace(active=True, autoCruiseEnabled=False)),
      smartCruiseControl=SimpleNamespace(map=SimpleNamespace(active=False), vision=SimpleNamespace(active=False)),
    ),
  ]

  rendered = render_summary(summarize_window(msgs, 1.0, 1.0, 1.0))

  assert "likely cause: speed_limit" in rendered
  assert "SP source speedLimitAssist" in rendered
  assert "speed-limit assist active" in rendered


def test_summarize_window_attributes_scc_map_before_scc_vision():
  msgs = [
    msg("carState", 0.0, vEgo=15.0, brakePressed=False, gasPressed=False),
    msg(
      "longitudinalPlanSP",
      1.0,
      longitudinalPlanSource="unknown",
      speedLimit=SimpleNamespace(assist=SimpleNamespace(active=False, autoCruiseEnabled=False)),
      smartCruiseControl=SimpleNamespace(map=SimpleNamespace(active=True), vision=SimpleNamespace(active=True)),
    ),
  ]

  rendered = render_summary(summarize_window(msgs, 1.0, 1.0, 1.0))

  assert "likely cause: scc_map" in rendered
  assert "SCC map active" in rendered
  assert "SCC vision active" in rendered


def test_summarize_window_prefers_scc_map_source_over_speed_limit_active():
  msgs = [
    msg("carState", 0.0, vEgo=15.0, brakePressed=False, gasPressed=False),
    msg(
      "longitudinalPlanSP",
      1.0,
      longitudinalPlanSource="sccMap",
      speedLimit=SimpleNamespace(assist=SimpleNamespace(active=True, autoCruiseEnabled=False)),
      smartCruiseControl=SimpleNamespace(map=SimpleNamespace(active=True), vision=SimpleNamespace(active=False)),
    ),
  ]

  rendered = render_summary(summarize_window(msgs, 1.0, 1.0, 1.0))

  assert "likely cause: scc_map" in rendered
  assert "SP source sccMap" in rendered
  assert "speed-limit assist active" in rendered
  assert "SCC map active" in rendered


def test_summarize_window_prefers_scc_vision_source_over_scc_map_active():
  msgs = [
    msg("carState", 0.0, vEgo=15.0, brakePressed=False, gasPressed=False),
    msg(
      "longitudinalPlanSP",
      1.0,
      longitudinalPlanSource="sccVision",
      speedLimit=SimpleNamespace(assist=SimpleNamespace(active=False, autoCruiseEnabled=False)),
      smartCruiseControl=SimpleNamespace(map=SimpleNamespace(active=True), vision=SimpleNamespace(active=True)),
    ),
  ]

  rendered = render_summary(summarize_window(msgs, 1.0, 1.0, 1.0))

  assert "likely cause: scc_vision" in rendered
  assert "SP source sccVision" in rendered
  assert "SCC map active" in rendered
  assert "SCC vision active" in rendered


def test_summarize_window_prefers_speed_limit_source_when_all_sp_flags_active():
  msgs = [
    msg("carState", 0.0, vEgo=15.0, brakePressed=False, gasPressed=False),
    msg(
      "longitudinalPlanSP",
      1.0,
      longitudinalPlanSource="speedLimitAssist",
      speedLimit=SimpleNamespace(assist=SimpleNamespace(active=True, autoCruiseEnabled=False)),
      smartCruiseControl=SimpleNamespace(map=SimpleNamespace(active=True), vision=SimpleNamespace(active=True)),
    ),
  ]

  rendered = render_summary(summarize_window(msgs, 1.0, 1.0, 1.0))

  assert "likely cause: speed_limit" in rendered
  assert "SP source speedLimitAssist" in rendered
  assert "speed-limit assist active" in rendered
  assert "SCC map active" in rendered
  assert "SCC vision active" in rendered


def test_summarize_window_reports_decision_layer_telemetry():
  msgs = [
    msg("carState", 0.0, vEgo=15.0, brakePressed=False, gasPressed=False),
    msg(
      "longitudinalPlanSP",
      1.0,
      longitudinalPlanSource="speedLimitAssist",
      speedLimit=SimpleNamespace(assist=SimpleNamespace(active=False, autoCruiseEnabled=False)),
      smartCruiseControl=SimpleNamespace(map=SimpleNamespace(active=False), vision=SimpleNamespace(active=False)),
      decisionLayer=SimpleNamespace(
        enabled=True,
        rawSource="speed_limit",
        rawReason="speed_limit_active",
        appliedReason="advisory_min_legacy",
        accelDelta=-0.3,
      ),
    ),
  ]

  rendered = render_summary(summarize_window(msgs, 1.0, 1.0, 1.0))

  assert "decision layer active" in rendered
  assert "decision raw source: speed_limit" in rendered
  assert "decision applied reason: advisory_min_legacy" in rendered
  assert "decision layer speed_limit -> advisory_min_legacy delta -0.300 m/s^2" in rendered


def test_summarize_window_uses_event_time_sp_source_after_transition():
  msgs = [
    msg("carState", 0.0, vEgo=15.0, brakePressed=False, gasPressed=False),
    msg(
      "longitudinalPlanSP",
      0.5,
      longitudinalPlanSource="speedLimitAssist",
      speedLimit=SimpleNamespace(assist=SimpleNamespace(active=True, autoCruiseEnabled=False)),
      smartCruiseControl=SimpleNamespace(map=SimpleNamespace(active=False), vision=SimpleNamespace(active=False)),
    ),
    msg(
      "longitudinalPlanSP",
      1.0,
      longitudinalPlanSource="sccMap",
      speedLimit=SimpleNamespace(assist=SimpleNamespace(active=False, autoCruiseEnabled=False)),
      smartCruiseControl=SimpleNamespace(map=SimpleNamespace(active=True), vision=SimpleNamespace(active=False)),
    ),
  ]

  rendered = render_summary(summarize_window(msgs, 1.0, 1.0, 1.0))

  assert "likely cause: scc_map" in rendered
  assert "SP source speedLimitAssist" in rendered
  assert "SP source sccMap" in rendered
  assert "speed-limit assist active" in rendered
  assert "SCC map active" in rendered


def test_summarize_window_uses_event_time_sp_active_fallback_after_transition():
  msgs = [
    msg("carState", 0.0, vEgo=15.0, brakePressed=False, gasPressed=False),
    msg(
      "longitudinalPlanSP",
      0.5,
      longitudinalPlanSource="unknown",
      speedLimit=SimpleNamespace(assist=SimpleNamespace(active=True, autoCruiseEnabled=False)),
      smartCruiseControl=SimpleNamespace(map=SimpleNamespace(active=False), vision=SimpleNamespace(active=False)),
    ),
    msg(
      "longitudinalPlanSP",
      1.0,
      longitudinalPlanSource="unknown",
      speedLimit=SimpleNamespace(assist=SimpleNamespace(active=False, autoCruiseEnabled=False)),
      smartCruiseControl=SimpleNamespace(map=SimpleNamespace(active=False), vision=SimpleNamespace(active=True)),
    ),
  ]

  rendered = render_summary(summarize_window(msgs, 1.0, 1.0, 1.0))

  assert "likely cause: scc_vision" in rendered
  assert "speed-limit assist active" in rendered
  assert "SCC vision active" in rendered


def test_summarize_window_prefers_pre_event_sp_source_over_closer_future_sample():
  msgs = [
    msg("carState", 0.0, vEgo=15.0, brakePressed=False, gasPressed=False),
    msg(
      "longitudinalPlanSP",
      0.8,
      longitudinalPlanSource="sccMap",
      speedLimit=SimpleNamespace(assist=SimpleNamespace(active=False, autoCruiseEnabled=False)),
      smartCruiseControl=SimpleNamespace(map=SimpleNamespace(active=True), vision=SimpleNamespace(active=False)),
    ),
    msg(
      "longitudinalPlanSP",
      1.1,
      longitudinalPlanSource="speedLimitAssist",
      speedLimit=SimpleNamespace(assist=SimpleNamespace(active=True, autoCruiseEnabled=False)),
      smartCruiseControl=SimpleNamespace(map=SimpleNamespace(active=False), vision=SimpleNamespace(active=False)),
    ),
  ]

  rendered = render_summary(summarize_window(msgs, 1.0, 1.0, 1.0))

  assert "likely cause: scc_map" in rendered
  assert "SP source sccMap" in rendered
  assert "SP source speedLimitAssist" in rendered


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


def test_summarize_window_uses_event_time_plan_source_over_stale_lead_source():
  msgs = [
    msg("carState", 0.0, vEgo=12.0, brakePressed=False, gasPressed=False),
    msg("longitudinalPlan", 0.4, longitudinalPlanSource="lead0", shouldStop=False, fcw=False, aTarget=-0.8),
    msg("radarState", 0.8, leadOne=SimpleNamespace(status=False, dRel=200.0, vRel=0.0)),
    msg("longitudinalPlan", 1.0, longitudinalPlanSource="cruise", shouldStop=False, fcw=False, aTarget=0.0),
  ]

  rendered = render_summary(summarize_window(msgs, 1.0, 1.0, 1.0))

  assert "likely cause: planner_source" in rendered
  assert "planner source lead0" in rendered
  assert "planner source cruise" in rendered


def test_summarize_window_uses_event_time_plan_source_over_future_lead_source():
  msgs = [
    msg("carState", 0.0, vEgo=12.0, brakePressed=False, gasPressed=False),
    msg("longitudinalPlan", 1.0, longitudinalPlanSource="cruise", shouldStop=False, fcw=False, aTarget=0.0),
    msg("longitudinalPlan", 1.2, longitudinalPlanSource="lead0", shouldStop=False, fcw=False, aTarget=-0.8),
  ]

  rendered = render_summary(summarize_window(msgs, 1.0, 1.0, 1.0))

  assert "likely cause: planner_source" in rendered
  assert "planner source cruise" in rendered
  assert "planner source lead0" in rendered


def test_summarize_window_uses_event_time_plan_braking_over_future_lead_braking():
  msgs = [
    msg("carState", 0.0, vEgo=12.0, brakePressed=False, gasPressed=False),
    msg("radarState", 0.8, leadOne=SimpleNamespace(status=True, dRel=18.4, vRel=-1.0)),
    msg("longitudinalPlan", 1.0, longitudinalPlanSource="cruise", shouldStop=False, fcw=False, aTarget=0.0),
    msg("longitudinalPlan", 1.2, longitudinalPlanSource="lead0", shouldStop=False, fcw=False, aTarget=-0.8),
  ]

  rendered = render_summary(summarize_window(msgs, 1.0, 1.0, 1.0))

  assert "likely cause: planner_source" in rendered
  assert "lead gap min 18.400 m" in rendered
  assert "aTarget min -0.800 m/s^2" in rendered


def test_summarize_window_uses_event_time_model_action_over_stale_model_stop():
  msgs = [
    msg("carState", 0.0, vEgo=8.0, brakePressed=False, gasPressed=False),
    msg("modelV2", 0.4, action=SimpleNamespace(desiredAcceleration=-1.2, shouldStop=True)),
    msg(
      "longitudinalPlanSP",
      0.9,
      longitudinalPlanSource="sccMap",
      smartCruiseControl=SimpleNamespace(map=SimpleNamespace(active=True), vision=SimpleNamespace(active=False)),
    ),
    msg("modelV2", 1.0, action=SimpleNamespace(desiredAcceleration=0.0, shouldStop=False)),
    msg("longitudinalPlan", 1.0, longitudinalPlanSource="cruise", shouldStop=False, fcw=False, aTarget=0.0),
  ]

  rendered = render_summary(summarize_window(msgs, 1.0, 1.0, 1.0))

  assert "likely cause: scc_map" in rendered
  assert "model action shouldStop true" in rendered


def test_summarize_window_uses_event_time_plan_stop_over_stale_model_plan_stop():
  msgs = [
    msg("carState", 0.0, vEgo=8.0, brakePressed=False, gasPressed=False),
    msg("longitudinalPlan", 0.4, longitudinalPlanSource="e2e", shouldStop=True, fcw=False, aTarget=-1.0),
    msg("longitudinalPlanSP", 0.9, longitudinalPlanSource="speedLimitAssist", speedLimit=SimpleNamespace(assist=SimpleNamespace(active=True))),
    msg("longitudinalPlan", 1.0, longitudinalPlanSource="cruise", shouldStop=False, fcw=False, aTarget=0.0),
  ]

  rendered = render_summary(summarize_window(msgs, 1.0, 1.0, 1.0))

  assert "likely cause: speed_limit" in rendered
  assert "plan shouldStop true" in rendered


def test_summarize_window_uses_event_time_model_action_over_future_model_stop():
  msgs = [
    msg("carState", 0.0, vEgo=8.0, brakePressed=False, gasPressed=False),
    msg(
      "longitudinalPlanSP",
      0.9,
      longitudinalPlanSource="sccVision",
      smartCruiseControl=SimpleNamespace(vision=SimpleNamespace(active=True), map=SimpleNamespace(active=False)),
    ),
    msg("modelV2", 1.0, action=SimpleNamespace(desiredAcceleration=0.0, shouldStop=False)),
    msg("longitudinalPlan", 1.0, longitudinalPlanSource="cruise", shouldStop=False, fcw=False, aTarget=0.0),
    msg("modelV2", 1.2, action=SimpleNamespace(desiredAcceleration=-1.2, shouldStop=True)),
  ]

  rendered = render_summary(summarize_window(msgs, 1.0, 1.0, 1.0))

  assert "likely cause: scc_vision" in rendered
  assert "model action shouldStop true" in rendered


def test_summarize_window_uses_event_time_plan_stop_over_future_model_plan_stop():
  msgs = [
    msg("carState", 0.0, vEgo=8.0, brakePressed=False, gasPressed=False),
    msg("longitudinalPlanSP", 0.9, longitudinalPlanSource="speedLimitAssist", speedLimit=SimpleNamespace(assist=SimpleNamespace(active=True))),
    msg("longitudinalPlan", 1.0, longitudinalPlanSource="cruise", shouldStop=False, fcw=False, aTarget=0.0),
    msg("longitudinalPlan", 1.2, longitudinalPlanSource="model", shouldStop=True, fcw=False, aTarget=-1.0),
  ]

  rendered = render_summary(summarize_window(msgs, 1.0, 1.0, 1.0))

  assert "likely cause: speed_limit" in rendered
  assert "plan shouldStop true" in rendered


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
