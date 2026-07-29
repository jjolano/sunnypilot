from types import SimpleNamespace

from openpilot.tools.drive_lab import compare_manual_planner_targets as compare_cli
from openpilot.tools.drive_lab.profile_lead_policy import (
  LeadPolicyBinSpec,
  assign_distance_bin,
  build_lead_policy_report,
  is_actuation_applicable,
)
from openpilot.tools.drive_lab.compare_manual_planner_targets import PlannerTargetSample, extract_planner_target_samples
from openpilot.tools.lib.logreader import ReadMode


class FakeMsg(SimpleNamespace):
  def which(self):
    return self.kind


def msg(kind, t_s, **payload):
  return FakeMsg(kind=kind, logMonoTime=int(t_s * 1e9), **{kind: SimpleNamespace(**payload)})


def sample(route="route-a", route_id="route-a", segment=None, t=0.0, d_rel=30.0, v_rel=-2.0, v=10.0,
           plan_a=-0.4, a_ego=-0.1, long_active=True, active=True, gas=False, brake=False,
           lead=True, source="lead0", sp_source="cruise", sp_stack="stack", fcw=False, should_stop=False):
  closing_speed = max(0.0, -v_rel)
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
    long_control_state="pid",
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
    ttc_s=(d_rel / closing_speed) if lead and closing_speed > 0 else None,
    required_decel_mps2=(closing_speed ** 2) / (2.0 * max(d_rel, 0.1)) if lead else None,
    time_headway_s=(d_rel / v) if lead and v > 0 else None,
  )


def test_bin_assignment_generalizes():
  assert assign_distance_bin(10.0, (45.0, 80.0)) == "near_<45m"
  assert assign_distance_bin(60.0, (45.0, 80.0)) == "mid_45_80m"
  assert assign_distance_bin(100.0, (45.0, 80.0)) == "far_>=80m"


def test_actuation_applicable_filters_nonapplicable_samples():
  assert is_actuation_applicable(sample())
  assert not is_actuation_applicable(sample(long_active=False))
  assert not is_actuation_applicable(sample(active=False))
  assert not is_actuation_applicable(sample(gas=True))
  assert not is_actuation_applicable(sample(lead=False))


def test_source_grouping_and_route_aggregation():
  samples_by_route = {
    "route-a": [sample(t=0.0, d_rel=20.0, source="lead0", sp_source="cruise"), sample(t=0.1, d_rel=70.0, source="cruise", sp_source="lead0")],
    "route-b": [sample(route="route-b", route_id="route-b", t=0.0, d_rel=90.0, source="lead0", sp_source="lead0")],
  }
  report = build_lead_policy_report(samples_by_route, LeadPolicyBinSpec())

  assert report.total_samples == 3
  assert report.buckets["near_<45m"].planner_sources == {"lead0": 1}
  assert report.buckets["mid_45_80m"].planner_sources == {"cruise": 1}
  assert report.buckets["far_>=80m"].sp_sources == {"lead0": 1}
  assert report.per_route_bins["route-a"]["near_<45m"].count == 1


def test_low_risk_non_closing_classification_counts():
  samples_by_route = {"route-a": [
    sample(d_rel=30.0, v_rel=1.0, plan_a=-0.4),
    sample(d_rel=30.0, v_rel=-1.0, plan_a=-0.4),
    sample(d_rel=10.0, v_rel=-5.0, plan_a=-0.4),
  ]}
  report = build_lead_policy_report(samples_by_route, LeadPolicyBinSpec(plan_brake_thresholds=(-0.3,), low_required_decel=0.3))

  bucket = report.buckets["near_<45m"]
  assert bucket.non_closing_plan_brake_count == 1
  assert bucket.low_required_decel_plan_brake_count == 1


def test_source_bucket_metrics_include_low_risk_counts():
  samples_by_route = {"route-a": [sample(source="lead0", d_rel=60.0, v_rel=-1.0, plan_a=-0.4), sample(source="cruise", d_rel=60.0, v_rel=-1.0, plan_a=0.0)]}
  report = build_lead_policy_report(samples_by_route, LeadPolicyBinSpec(plan_brake_thresholds=(-0.3,), low_required_decel=0.3))

  source_buckets = report.to_dict()["buckets"]["mid_45_80m"]["source_buckets"]
  assert source_buckets["lead0"]["low_required_decel_plan_brake_count"] == 1
  assert source_buckets["lead0"]["plan_brake_counts"]["plan_a_target<=-0.3"] == 1
  assert source_buckets["cruise"]["plan_brake_counts"]["plan_a_target<=-0.3"] == 0


def test_extract_lead_policy_samples_via_monkeypatched_logreader(monkeypatch):
  msgs = [
    msg("selfdriveState", 0.0, enabled=True, active=True),
    msg("carControl", 0.0, longActive=True),
    msg("radarState", 0.1, leadOne=SimpleNamespace(status=True, dRel=50.0, vRel=-4.0)),
    msg("longitudinalPlan", 0.2, aTarget=-0.6, longitudinalPlanSource="lead0", shouldStop=False, fcw=False),
    msg("longitudinalPlanSP", 0.2, aTarget=-0.5, longitudinalPlanSource="cruise", stack=SimpleNamespace(actuatedStack="stackA")),
    msg("carState", 0.3, vEgo=12.0, aEgo=-0.2, gasPressed=False, brakePressed=False, standstill=False, vCruise=80.0),
  ]
  monkeypatch.setattr(compare_cli, "LogReader", lambda route, default_mode, sort_by_time: msgs)

  samples = extract_planner_target_samples("route-a", ReadMode.AUTO)

  assert len(samples) == 1
  assert samples[0].lead_d_rel == 50.0
  assert samples[0].plan_source == "lead0"
  assert samples[0].sp_stack == "stackA"
