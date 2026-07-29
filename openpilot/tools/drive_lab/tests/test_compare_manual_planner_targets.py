from types import SimpleNamespace

import pytest

from openpilot.tools.drive_lab import compare_manual_planner_targets as compare_cli
from openpilot.tools.drive_lab.scenario_spec import ScenarioSpec
from openpilot.tools.drive_lab.compare_manual_planner_targets import (
  PlannerTargetSample,
  build_route_agreement_profile,
  build_suspicious_episodes,
  extract_planner_target_samples,
  is_low_confidence_manual_preview_sample,
  is_opposite_intent,
  is_strong_opposite_intent,
  low_confidence_manual_preview_reason,
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
           should_stop=False, fcw=False, sp_source="cruise", sp_stack="sunnypilotCurrent", v_cruise=80.0,
           long_state="pid"):
  closing_speed = (-v_rel) if v_rel is not None else None
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
    long_control_state=long_state,
    v_cruise_kph=v_cruise,
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
    ttc_s=(d_rel / closing_speed) if lead and d_rel is not None and closing_speed is not None and closing_speed > 0.1 and d_rel > 0 else None,
    required_decel_mps2=((closing_speed ** 2) / (2.0 * max(d_rel, 0.1))) if lead and d_rel is not None and closing_speed is not None and closing_speed > 0.1 and d_rel > 0 else None,
    time_headway_s=(d_rel / max(v, 0.1)) if lead and d_rel is not None and d_rel > 0 and v > 0.1 else None,
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
  assert samples[0].plan_time_s == pytest.approx(0.2)
  assert samples[0].ttc_s == pytest.approx(17.5)
  assert samples[0].required_decel_mps2 == pytest.approx(0.16 / 14.0)
  assert samples[0].time_headway_s == pytest.approx(7.0 / 7.5)
  assert is_low_confidence_manual_preview_sample(samples[0])
  assert low_confidence_manual_preview_reason(samples[0]) == "long_control_off"


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


def test_summary_excludes_low_confidence_preview_by_default():
  high_confidence = sample(0.0, -1.4, 0.2, gas=True, source="lead0")
  reset_preview = sample(0.1, -1.4, -2.4, brake=True, v_cruise=255.0, long_state="off")
  samples = [high_confidence, reset_preview]
  profiles = [build_route_agreement_profile("route-a", samples, min_manual_moving_samples=1, max_active_ratio=0.25)]

  summary = summarize_planner_target_agreement({"route-a": samples}, profiles)

  assert summary.sample_count == 1
  assert summary.low_confidence_preview_sample_count == 1
  assert summary.low_confidence_preview_reasons == {"long_control_off": 1}
  assert summary.opposite_count == 1

  exploratory_summary = summarize_planner_target_agreement({"route-a": samples}, profiles, include_low_confidence_preview=True)
  assert exploratory_summary.sample_count == 2


def test_low_confidence_preview_reasons_distinguish_unset_cruise():
  unset_cruise = sample(0.0, 0.1, 0.1, v_cruise=255.0, long_state="pid")
  missing_cruise = sample(0.1, 0.1, 0.1, v_cruise=None, long_state="pid")  # type: ignore[arg-type]

  assert low_confidence_manual_preview_reason(unset_cruise) == "unset_cruise"
  assert low_confidence_manual_preview_reason(missing_cruise) == "missing_cruise"


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


def test_lead_risk_metrics_compute_ttc_required_decel_and_headway():
  lead_sample = sample(0.0, -0.2, 0.1, lead=True, d_rel=10.0, v_rel=-5.0, v=20.0, source="lead0")

  assert lead_sample.ttc_s == pytest.approx(2.0)
  assert lead_sample.required_decel_mps2 == pytest.approx(1.25)
  assert lead_sample.time_headway_s == pytest.approx(0.5)


def test_summary_includes_lead_risk_counts_and_render_line():
  samples = [
    sample(0.0, -0.2, 0.1, lead=True, d_rel=2.0, v_rel=-5.0, v=10.0, source="lead0"),
    sample(0.1, -0.2, 0.1, lead=True, d_rel=15.0, v_rel=-1.0, v=10.0, source="cruise"),
    sample(0.2, 0.1, 0.1, lead=False, source="cruise"),
  ]
  profiles = [build_route_agreement_profile("route-a", samples, min_manual_moving_samples=1, max_active_ratio=0.25)]

  summary = summarize_planner_target_agreement({"route-a": samples}, profiles, include_low_confidence_preview=True)
  rendered = compare_cli.render_agreement_summary(summary)

  assert summary.high_required_decel_count == 1
  assert summary.low_ttc_count == 1
  assert summary.lead_risk_source_counts == {"lead0": 1}
  assert summary.min_ttc_s == pytest.approx(0.4)
  assert summary.max_required_decel_mps2 == pytest.approx(6.25)
  assert summary.mean_time_headway_s == pytest.approx((0.2 + 1.5) / 2)
  assert "lead risk:" in rendered
  assert "high_decel=1" in rendered


def test_no_lead_or_opening_lead_yields_no_risk_metrics():
  samples = [
    sample(0.0, 0.0, 0.0, lead=False),
    sample(0.1, 0.0, 0.0, lead=True, d_rel=10.0, v_rel=1.0, v=10.0),
  ]
  profiles = [build_route_agreement_profile("route-a", samples, min_manual_moving_samples=1, max_active_ratio=0.25)]

  summary = summarize_planner_target_agreement({"route-a": samples}, profiles)

  assert summary.high_required_decel_count == 0
  assert summary.low_ttc_count == 0
  assert summary.min_ttc_s is None
  assert summary.max_required_decel_mps2 is None
  assert summary.mean_time_headway_s == pytest.approx(1.0)


def test_episode_to_scenario_spec_includes_route_provenance_and_checks():
  episode = compare_cli.PlannerTargetEpisode(
    route="route-a--7",
    route_id="route-a",
    segment=7,
    start_time_s=10.0,
    end_time_s=14.0,
    duration_s=4.0,
    sample_count=3,
    opposite_count=1,
    strong_opposite_count=0,
    max_abs_error=1.5,
    planner_sources={"lead0": 2, "cruise": 1},
    driver_gas_count=1,
    driver_brake_count=0,
    lead_ratio=1.0,
    min_lead_d_rel=5.0,
    min_lead_v_rel=-2.0,
    lead_status_flips=1,
    plan_source_flips=1,
    plan_span=1.8,
    high_plan_jerk_count=2,
    min_ttc_s=1.4,
    max_required_decel_mps2=2.8,
  )

  spec = compare_cli.episode_to_scenario_spec(episode, index=2)

  assert isinstance(spec, ScenarioSpec)
  assert spec.kind == "lead_risk"
  assert spec.mode == "route-derived"
  assert spec.source == "manual-planner-target"
  assert spec.maneuver_kwargs == {"route_id": "route-a", "segment": 7, "start_time_s": 10.0, "end_time_s": 14.0}
  assert spec.actors["lead"]["min_ttc_s"] == 1.4
  assert spec.events == ("lead_risk", "high_plan_jerk")
  assert spec.oracle["checks"] == ("manual_agreement", "lead_risk", "jerk")
  assert "route-derived" in spec.tags
  assert spec.provenance["source_tool"] == "compare_manual_planner_targets"


def test_manual_planner_cli_writes_scenario_export(tmp_path, monkeypatch):
  summary = compare_cli.PlannerTargetAgreementSummary(
    route_count=1,
    included_route_count=1,
    sample_count=0,
    manual_moving_sample_count=0,
    low_confidence_preview_sample_count=0,
    low_confidence_preview_reasons={},
    actuation_applicable_sample_count=0,
    correlation=None,
    mean_abs_error=0.0,
    p90_abs_error=0.0,
    p95_abs_error=0.0,
    opposite_count=0,
    opposite_ratio=0.0,
    strong_opposite_count=0,
    strong_opposite_ratio=0.0,
    should_stop_moving_count=0,
    should_stop_conflict_count=0,
    fcw_count=0,
    high_plan_jerk_count=0,
    min_ttc_s=None,
    max_required_decel_mps2=None,
    mean_time_headway_s=None,
    high_required_decel_count=0,
    low_ttc_count=0,
    lead_risk_source_counts={},
    planner_source_counts={},
    sp_source_counts={},
    sp_stack_counts={},
    route_profiles=[],
    episodes=[compare_cli.PlannerTargetEpisode(
      route="route-a",
      route_id="route-a",
      segment=None,
      start_time_s=1.0,
      end_time_s=2.0,
      duration_s=1.0,
      sample_count=1,
      opposite_count=0,
      strong_opposite_count=0,
      max_abs_error=0.0,
      planner_sources={"cruise": 1},
      driver_gas_count=0,
      driver_brake_count=0,
      lead_ratio=0.0,
      min_lead_d_rel=None,
      min_lead_v_rel=None,
      lead_status_flips=0,
      plan_source_flips=0,
      plan_span=0.0,
      high_plan_jerk_count=0,
      min_ttc_s=None,
      max_required_decel_mps2=None,
    )],
  )
  monkeypatch.setattr(compare_cli, "summarize_planner_target_agreement", lambda *args, **kwargs: summary)
  monkeypatch.setattr(compare_cli, "extract_planner_target_samples", lambda *args, **kwargs: [])
  out = tmp_path / "scenarios.json"
  monkeypatch.setattr(compare_cli.sys, "argv", ["prog", "route-a", "--scenario-output", str(out), "--episodes", "1"])

  compare_cli.main()

  payload = ScenarioSpec.from_dict(__import__("json").loads(out.read_text())[0])
  assert payload.provenance["source_tool"] == "compare_manual_planner_targets"
