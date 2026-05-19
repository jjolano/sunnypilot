from types import SimpleNamespace

import pytest

from openpilot.tools.drive_lab import compare_manual_planner_targets as compare_cli
from openpilot.tools.drive_lab.compare_manual_planner_targets import (
  PlannerTargetSample,
  build_route_agreement_profile,
  build_suspicious_episodes,
  extract_planner_target_samples,
  is_opposite_intent,
  is_strong_opposite_intent,
  summarize_planner_target_agreement,
)
from openpilot.tools.lib.logreader import ReadMode


class FakeMsg(SimpleNamespace):
  def which(self):
    return self.kind


def msg(kind, t_s, **payload):
  return FakeMsg(kind=kind, logMonoTime=int(t_s * 1e9), **{kind: SimpleNamespace(**payload)})


def sample(t, plan_a, a_ego, route="route-a", route_id="route-a", segment=None, v=8.0, gas=False, brake=False,
           active=False, long_active=False, source="cruise", lead=False, d_rel=None, v_rel=None,
           should_stop=False, fcw=False, sp_source="cruise", sp_stack="sunnypilotCurrent"):
  return PlannerTargetSample(
    route=route,
    route_id=route_id,
    segment=segment,
    t=t,
    v_ego=v,
    a_ego=a_ego,
    gas_pressed=gas,
    brake_pressed=brake,
    standstill=False,
    selfdrive_enabled=active,
    selfdrive_active=active,
    long_active=long_active,
    long_control_state="pid" if long_active else "off",
    v_cruise_kph=80.0,
    plan_a_target=plan_a,
    plan_source=source,
    plan_should_stop=should_stop,
    plan_fcw=fcw,
    sp_a_target=plan_a,
    sp_source=sp_source,
    sp_stack=sp_stack,
    lead_status=lead,
    lead_d_rel=d_rel,
    lead_v_rel=v_rel,
    model_desired_accel=None,
    model_should_stop=False,
  )


def test_extract_planner_target_samples_persists_preview_context(monkeypatch):
  msgs = [
    msg("selfdriveState", 0.0, enabled=False, active=False),
    msg("carControl", 0.0, longActive=False),
    msg("controlsState", 0.0, longControlState="off"),
    msg("radarState", 0.1, leadOne=SimpleNamespace(status=True, dRel=7.0, vRel=-0.4)),
    msg("longitudinalPlan", 0.2, aTarget=-1.2, longitudinalPlanSource="lead0", shouldStop=False, fcw=False),
    msg("longitudinalPlanSP", 0.2, aTarget=0.0, longitudinalPlanSource="cruise",
        stack=SimpleNamespace(actuatedStack="sunnypilotCurrent")),
    msg("carState", 0.3, vEgo=7.5, aEgo=0.2, gasPressed=True, brakePressed=False, standstill=False, vCruise=255.0),
  ]
  monkeypatch.setattr(compare_cli, "LogReader", lambda route, default_mode, sort_by_time: msgs)

  samples = extract_planner_target_samples("route-a--7/qlog.zst", ReadMode.QLOG)

  assert len(samples) == 1
  assert samples[0].route_id == "route-a"
  assert samples[0].segment == 7
  assert samples[0].plan_source == "lead0"
  assert samples[0].sp_stack == "sunnypilotCurrent"
  assert samples[0].lead_d_rel == 7.0
  assert samples[0].gas_pressed
  assert not samples[0].long_active


def test_extract_planner_target_samples_ignores_stale_plan(monkeypatch):
  msgs = [
    msg("longitudinalPlan", 0.0, aTarget=0.8, longitudinalPlanSource="cruise", shouldStop=False, fcw=False),
    msg("carState", 2.0, vEgo=8.0, aEgo=0.1, gasPressed=False, brakePressed=False),
  ]
  monkeypatch.setattr(compare_cli, "LogReader", lambda route, default_mode, sort_by_time: msgs)

  assert extract_planner_target_samples("route-a", ReadMode.AUTO, max_plan_age_s=1.0) == []


def test_opposite_and_strong_intent_classification():
  hard_brake_driver_gas = sample(0.0, -1.4, 0.2, gas=True, source="lead0")
  accel_driver_brake = sample(1.0, 0.9, -0.7, brake=True)
  neutral_driver_brake = sample(2.0, 0.1, -0.7, brake=True)

  assert is_opposite_intent(hard_brake_driver_gas)
  assert is_strong_opposite_intent(hard_brake_driver_gas)
  assert is_opposite_intent(accel_driver_brake)
  assert is_strong_opposite_intent(accel_driver_brake)
  assert not is_opposite_intent(neutral_driver_brake)


def test_summary_filters_excluded_routes_and_counts_manual_disagreement():
  included = [
    sample(0.0, -1.4, 0.2, gas=True, source="lead0"),
    sample(0.1, 0.9, -0.7, brake=True, source="cruise"),
    sample(0.2, 0.2, 0.1, source="cruise"),
  ]
  excluded = [sample(0.0, -1.4, 0.2, route="active-route", route_id="active-route", gas=True, active=True, long_active=True)]
  samples_by_route = {"route-a": included, "active-route": excluded}
  profiles = [
    build_route_agreement_profile("route-a", included, min_manual_moving_samples=1, max_active_ratio=0.25),
    build_route_agreement_profile("active-route", excluded, min_manual_moving_samples=1, max_active_ratio=0.25),
  ]

  summary = summarize_planner_target_agreement(samples_by_route, profiles)

  assert summary.included_route_count == 1
  assert summary.sample_count == 3
  assert summary.opposite_count == 2
  assert summary.strong_opposite_count == 2
  assert summary.planner_source_counts == {"lead0": 1, "cruise": 2}


def test_suspicious_episodes_track_context_flips_and_jerk():
  samples = [
    sample(0.0, 0.2, 0.2, lead=False, source="cruise"),
    sample(0.1, 0.3, 0.3, lead=True, d_rel=8.0, v_rel=-0.2, source="cruise"),
    sample(0.2, -1.4, 0.1, gas=True, lead=True, d_rel=7.5, v_rel=-0.4, source="lead0"),
    sample(0.3, -1.3, 0.1, gas=True, lead=True, d_rel=7.2, v_rel=-0.5, source="lead0"),
    sample(0.9, 0.2, 0.2, lead=False, source="cruise"),
  ]

  episodes = build_suspicious_episodes(samples, large_error_threshold=1.2, episode_gap_s=0.6, context_s=1.0, high_jerk_threshold=8.0)

  assert len(episodes) == 1
  episode = episodes[0]
  assert episode.sample_count == 2
  assert episode.opposite_count == 2
  assert episode.lead_status_flips >= 2
  assert episode.plan_source_flips >= 2
  assert episode.plan_span == pytest.approx(1.7)
  assert episode.high_plan_jerk_count >= 1


def test_should_stop_conflict_is_reported_for_moving_preview():
  samples = [sample(0.0, -0.1, 0.1, v=2.0, gas=False, brake=False, should_stop=True)]
  profiles = [build_route_agreement_profile("route-a", samples, min_manual_moving_samples=1, max_active_ratio=0.25)]

  summary = summarize_planner_target_agreement({"route-a": samples}, profiles)

  assert summary.should_stop_moving_count == 1
  assert summary.should_stop_conflict_count == 1
