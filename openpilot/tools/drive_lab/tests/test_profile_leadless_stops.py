from types import SimpleNamespace

import pytest

from openpilot.tools.drive_lab import profile_leadless_stops as leadless_cli
from openpilot.tools.drive_lab.profile_leadless_stops import (
  LeadlessStopSample,
  build_leadless_stop_episodes,
  extract_leadless_stop_samples,
  model_stop_context,
  model_stop_matches_map_distance,
  route_identity,
  scc_early_model_stop_context,
  signal_active,
  summarize_leadless_stop_correlation,
)
from openpilot.tools.lib.logreader import ReadMode


class FakeMsg(SimpleNamespace):
  def which(self):
    return self.kind


def msg(kind, t_s, **payload):
  return FakeMsg(kind=kind, logMonoTime=int(t_s * 1e9), **{kind: SimpleNamespace(**payload)})


def sample(t, v, a, route="route-a--1/rlog.zst", route_id="route-a", segment=1, gas=False, brake=False,
            active=False, long_active=False, lead=False, lead_d=None, model_stop=False, model_accel=None,
            model_stop_distance=None, model_endpoint_x=None, model_endpoint_v=None, plan_stop=False,
            plan_source="cruise", scc_early_model_stop=False, tc_valid=False, tc_type="",
            tc_distance=None, tca_valid=False, tca_type="", tca_distance=None):
  return LeadlessStopSample(
    route=route,
    route_id=route_id,
    segment=segment,
    t=t,
    v_ego=v,
    a_ego=a,
    gas_pressed=gas,
    brake_pressed=brake,
    standstill=v <= 0.1,
    selfdrive_active=active,
    long_active=long_active,
    long_control_state="pid",
    lead_status=lead,
    lead_d_rel=lead_d,
    lead_v_rel=None,
    model_should_stop=model_stop,
    model_desired_accel=model_accel,
    model_stop_distance=model_stop_distance,
    model_endpoint_x=model_endpoint_x,
    model_endpoint_v=model_endpoint_v,
    scc_early_model_stop=scc_early_model_stop,
    plan_should_stop=plan_stop,
    plan_source=plan_source,
    plan_a_target=None,
    sp_source="cruise",
    sp_a_target=None,
    sp_stack="sunnypilotCurrent",
    traffic_control_valid=tc_valid,
    traffic_control_type=tc_type,
    traffic_control_distance=tc_distance,
    traffic_control_ahead_valid=tca_valid,
    traffic_control_ahead_type=tca_type,
    traffic_control_ahead_distance=tca_distance,
  )


def test_route_identity_handles_nested_segment_log_paths():
  assert route_identity("/tmp/00000187--ea39892416/4/rlog.zst") == ("00000187--ea39892416", 4)
  assert route_identity("/tmp/00000188--249e4349c3--2/qlog.zst") == ("00000188--249e4349c3", 2)
  assert route_identity("/tmp/00000188--249e4349c3--2.rlog.zst") == ("00000188--249e4349c3", 2)


def test_model_stop_context_uses_first_low_velocity_path_point():
  model = SimpleNamespace(
    action=SimpleNamespace(shouldStop=False),
    position=SimpleNamespace(x=[0.0, 12.0, 24.0]),
    velocity=SimpleNamespace(x=[8.0, 0.8, 0.0]),
  )

  assert model_stop_context(model) == (12.0, 24.0, 0.0)


def test_scc_early_model_stop_context_matches_planner_gate_shape():
  model = SimpleNamespace(
    action=SimpleNamespace(desiredAcceleration=-1.2),
    position=SimpleNamespace(x=[0.0, 10.0, 30.0]),
    velocity=SimpleNamespace(x=[8.0, 0.5, 0.0]),
  )
  far_endpoint = SimpleNamespace(
    action=SimpleNamespace(desiredAcceleration=-1.2),
    position=SimpleNamespace(x=[0.0, 40.0, 120.0]),
    velocity=SimpleNamespace(x=[8.0, 0.5, 0.0]),
  )

  assert scc_early_model_stop_context(model)
  assert not scc_early_model_stop_context(far_endpoint)


def test_extract_samples_persists_model_map_and_planner_context(monkeypatch):
  msgs = [
    msg("selfdriveState", 0.0, active=False),
    msg("carControl", 0.0, longActive=False),
    msg("radarState", 0.1, leadOne=SimpleNamespace(status=False), leadTwo=SimpleNamespace(status=False)),
    msg("modelV2", 0.2,
        action=SimpleNamespace(desiredAcceleration=-1.2, shouldStop=False),
        position=SimpleNamespace(x=[0.0, 10.0, 30.0]),
        velocity=SimpleNamespace(x=[8.0, 0.5, 0.0])),
    msg("longitudinalPlan", 0.3, shouldStop=True, longitudinalPlanSource="e2e", aTarget=-1.0),
    msg("longitudinalPlanSP", 0.3, longitudinalPlanSource="osmTrafficControl", aTarget=-0.5,
        stack=SimpleNamespace(actuatedStack="customV2")),
    msg("liveMapDataSP", 0.4, trafficControlValid=False, trafficControl="", trafficControlDistance=0.0,
        trafficControlAheadValid=True, trafficControlAhead="traffic_signal", trafficControlAheadDistance=28.0),
    msg("carState", 0.5, vEgo=8.0, aEgo=-0.4, gasPressed=False, brakePressed=True, standstill=False),
  ]
  monkeypatch.setattr(leadless_cli, "LogReader", lambda route, default_mode, sort_by_time: msgs)

  samples = extract_leadless_stop_samples("route-a--4/rlog.zst", ReadMode.RLOG)

  assert len(samples) == 1
  assert samples[0].model_stop_distance == 10.0
  assert samples[0].scc_early_model_stop
  assert samples[0].plan_should_stop
  assert samples[0].plan_source == "e2e"
  assert samples[0].sp_source == "osmTrafficControl"
  assert samples[0].traffic_control_ahead_type == "traffic_signal"
  assert samples[0].traffic_control_ahead_distance == 28.0


def test_strict_leadless_stop_counts_model_accel_as_timely_and_should_stop_as_late():
  samples = [
    sample(0.0, 8.0, 0.0),
    sample(1.0, 8.0, -0.1, model_accel=-1.0, scc_early_model_stop=True),
    sample(2.0, 7.5, -0.4),
    sample(3.0, 6.0, -0.8, brake=True),
    sample(4.0, 3.0, -1.0, brake=True),
    sample(5.0, 0.4, -0.2, brake=True, model_stop=True, plan_stop=True, plan_source="e2e"),
  ]

  episodes = build_leadless_stop_episodes(samples)

  assert len(episodes) == 1
  episode = episodes[0]
  assert episode.kind == "strict_leadless_stop"
  assert episode.signals["model_accel_le_-1.0"]["timely"]
  assert episode.signals["scc_early_model_stop_gate"]["timely"]
  assert episode.signals["model_should_stop"]["hit"]
  assert not episode.signals["model_should_stop"]["timely"]
  assert episode.signals["planner_should_stop"]["hit"]
  assert not episode.signals["planner_should_stop"]["timely"]


def test_low_lead_flicker_episode_is_separate_from_strict_leadless():
  samples = [
    sample(0.0, 7.0, 0.0),
    sample(1.0, 6.5, -0.6, brake=True),
    sample(2.0, 4.0, -0.8, brake=True, lead=True, lead_d=12.0),
    sample(3.0, 0.8, -0.2, brake=True),
  ]

  episodes = build_leadless_stop_episodes(samples)

  assert len(episodes) == 1
  assert episodes[0].kind == "low_lead_flicker_stop"
  assert 0.10 < episodes[0].lead_ratio <= 0.35


def test_map_model_distance_match_signal_requires_supported_control_and_distance_match():
  matched = sample(
    0.0,
    10.0,
    -0.2,
    model_stop_distance=31.0,
    tca_valid=True,
    tca_type="traffic_signal",
    tca_distance=28.0,
  )
  unsupported = sample(
    0.0,
    10.0,
    -0.2,
    model_stop_distance=31.0,
    tca_valid=True,
    tca_type="speed_bump",
    tca_distance=28.0,
  )
  mismatch = sample(
    0.0,
    10.0,
    -0.2,
    model_stop_distance=80.0,
    tca_valid=True,
    tca_type="stop_sign",
    tca_distance=20.0,
  )

  assert model_stop_matches_map_distance(31.0, 28.0)
  assert signal_active(matched, "map_model_distance_match")
  assert not signal_active(unsupported, "map_model_distance_match")
  assert not signal_active(mismatch, "map_model_distance_match")


def test_scc_early_model_stop_gate_is_not_raw_model_accel():
  raw_accel_only = sample(0.0, 10.0, 0.0, model_accel=-1.2)
  gated = sample(0.0, 10.0, 0.0, model_accel=-1.2, scc_early_model_stop=True)

  assert signal_active(raw_accel_only, "model_accel_le_-1.0")
  assert not signal_active(raw_accel_only, "scc_early_model_stop_gate")
  assert signal_active(gated, "scc_early_model_stop_gate")


def test_summary_reports_false_positive_clusters_outside_episode_windows():
  route_a = [
    sample(0.0, 8.0, 0.0, model_accel=-1.0),
    sample(1.0, 8.0, -0.2),
    sample(2.0, 7.0, -0.7, brake=True),
    sample(3.0, 0.8, -0.2, brake=True),
  ]
  route_b = [
    sample(0.0, 8.0, 0.0, route="route-b--0/rlog.zst", route_id="route-b", segment=0),
    sample(1.0, 8.0, -0.1, route="route-b--0/rlog.zst", route_id="route-b", segment=0, model_accel=-1.0),
    sample(2.0, 7.5, -0.2, route="route-b--0/rlog.zst", route_id="route-b", segment=0, plan_source="e2e"),
  ]

  summary = summarize_leadless_stop_correlation({"route-a": route_a, "route-b": route_b})

  metric = summary.signal_metrics["model_accel_le_-1.0"]["strict"]
  fp_clusters = [
    cluster for cluster in summary.false_positive_clusters
    if cluster.scope == "strict" and cluster.signal_name == "model_accel_le_-1.0"
  ]

  assert summary.episode_counts == {"strict_leadless_stop": 1}
  assert metric.timely_hit_count == 1
  assert metric.false_positive_clusters == 1
  assert metric.precision_proxy == pytest.approx(0.5)
  assert len(fp_clusters) == 1
  assert fp_clusters[0].route_id == "route-b"
  assert fp_clusters[0].segment == 0
  assert fp_clusters[0].start_time_s == pytest.approx(1.0)
  assert fp_clusters[0].end_time_s == pytest.approx(1.0)
  assert fp_clusters[0].v_start == pytest.approx(8.0)
  assert fp_clusters[0].v_end == pytest.approx(7.5)
  assert fp_clusters[0].delta_v == pytest.approx(-0.5)
  assert fp_clusters[0].model_desired_accel_min == pytest.approx(-1.0)
  assert fp_clusters[0].brake_ratio == 0.0
  assert fp_clusters[0].lead_ratio == 0.0
  assert fp_clusters[0].active_ratio == 0.0
  assert fp_clusters[0].planner_sources == {"cruise": 2, "e2e": 1}
