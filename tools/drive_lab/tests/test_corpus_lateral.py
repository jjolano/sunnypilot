import contextlib
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import openpilot.tools.drive_lab.corpus_lateral as corpus_module
from openpilot.tools.drive_lab.corpus_lateral import (
  CorpusFrameOutput,
  CorpusReport,
  RouteCorpusResult,
  RouteLateralMetrics,
  _compute_metrics,
  _empty_metrics,
  main,
  run_corpus,
)
from openpilot.tools.drive_lab.fuzz_lateral_route_replay import DT, LateralRouteFrame, N_PATH_POINTS
from openpilot.tools.drive_lab.log_profile import LateralProfile, ProfileRange


# ---------- helpers ----------


def _coherent_frame(t: float, v_ego: float = 20.0, curvature: float = 0.001) -> LateralRouteFrame:
  xs = tuple(float(x) for x in range(N_PATH_POINTS))
  ys = tuple(0.5 * curvature * x * x for x in range(N_PATH_POINTS))
  ystd = tuple(0.1 for _ in range(N_PATH_POINTS))
  yaw = tuple(curvature * x for x in range(N_PATH_POINTS))
  yaw_rate = tuple(curvature * v_ego for _ in range(N_PATH_POINTS))
  return LateralRouteFrame(
    t=t,
    v_ego=v_ego,
    lat_active=True,
    raw_curvature=curvature,
    measured_curvature=curvature,
    roll=0.0,
    steering_pressed=False,
    left_blinker=False,
    right_blinker=False,
    lane_change_state=0,
    lane_change_direction=0,
    position_x=xs,
    position_y=ys,
    position_y_std=ystd,
    orientation_z=yaw,
    orientation_rate_z=yaw_rate,
    lane_line_probs=(0.9, 0.9, 0.9, 0.9),
    frame_drop_perc=0.0,
  )


class _FakeMsg(SimpleNamespace):
  def which(self):
    return self.kind


def _fake_route_messages(n: int, desired_curvature: float = 0.001, start_t: float = 0.0) -> list[_FakeMsg]:
  msgs: list[_FakeMsg] = []
  for i in range(n):
    t = start_t + i * DT
    msgs.extend([
      _FakeMsg(kind="carState", logMonoTime=int(t * 1e9), carState=SimpleNamespace(
        vEgo=20.0, steeringPressed=False, leftBlinker=False, rightBlinker=False)),
      _FakeMsg(kind="carControl", logMonoTime=int(t * 1e9), carControl=SimpleNamespace(
        latActive=True, actuators=SimpleNamespace(curvature=desired_curvature))),
      _FakeMsg(kind="liveParameters", logMonoTime=int(t * 1e9), liveParameters=SimpleNamespace(roll=0.0)),
      _FakeMsg(kind="modelV2", logMonoTime=int(t * 1e9), modelV2=_model_v2_payload(desired_curvature)),
      _FakeMsg(kind="controlsState", logMonoTime=int(t * 1e9), controlsState=SimpleNamespace(
        curvature=desired_curvature, desiredCurvature=desired_curvature)),
    ])
  return msgs


def _model_v2_payload(desired_curvature: float, v_ego: float = 20.0) -> SimpleNamespace:
  n = N_PATH_POINTS
  xs = [float(x) for x in range(n)]
  ys = [0.5 * desired_curvature * x * x for x in range(n)]
  ystd = [0.1] * n
  yaw = [desired_curvature * x for x in range(n)]
  yaw_rate = [desired_curvature * v_ego] * n
  return SimpleNamespace(
    action=SimpleNamespace(desiredCurvature=desired_curvature),
    position=SimpleNamespace(x=xs, y=ys, yStd=ystd),
    orientation=SimpleNamespace(z=yaw),
    orientationRate=SimpleNamespace(z=yaw_rate),
    laneLineProbs=[0.9, 0.9, 0.9, 0.9],
    frameDropPerc=0.0,
    meta=SimpleNamespace(laneChangeState=0, laneChangeDirection=0),
  )


def _dummy_profile() -> LateralProfile:
  return LateralProfile(
    source="dummy",
    sample_count=0,
    ego_speed=ProfileRange(0.0, 0.0),
    curvature=ProfileRange(0.0, 0.0),
    lane_confidence=ProfileRange(0.0, 0.0),
    roll=ProfileRange(0.0, 0.0),
  )


# ---------- serialization ----------


def test_route_corpus_result_serializes_and_deserializes():
  metrics = RouteLateralMetrics(
    gating_rate=0.1,
    fallback_rate=0.2,
    path_quality_p5=0.3,
    path_quality_p50=0.5,
    path_quality_p95=0.9,
    curvature_rmse=0.01,
    source_distribution={"model_path": 0.8, "fallback_measured": 0.2},
    curvature_limited_rate=0.05,
  )
  profile = _dummy_profile()
  result = RouteCorpusResult(
    route="fake/route",
    frame_count=100,
    metrics=metrics,
    profile=profile,
  )

  data = result.to_dict()
  restored = RouteCorpusResult.from_dict(data)

  assert restored.route == result.route
  assert restored.frame_count == result.frame_count
  assert restored.metrics == result.metrics
  assert restored.profile == result.profile
  assert restored.error is None


def test_corpus_report_serializes_and_deserializes():
  result = RouteCorpusResult(
    route="fake/route",
    frame_count=10,
    metrics=_empty_metrics(),
    profile=_dummy_profile(),
  )
  report = CorpusReport(
    routes=(result,),
    aggregate_metrics={"gating_rate_mean": 0.0},
  )

  data = report.to_dict()
  restored = CorpusReport.from_dict(data)

  assert len(restored.routes) == 1
  assert restored.routes[0].route == "fake/route"
  assert restored.aggregate_metrics == report.aggregate_metrics


# ---------- metric computation ----------


def test_compute_metrics_on_synthetic_outputs():
  outputs = (
    CorpusFrameOutput(t=0.0, v_ego=20.0, raw_curvature=0.001, processed_curvature=0.0011,
                      measured_curvature=0.001, path_quality=0.95, path_reason="valid",
                      gated=False, demand_source="model_path", curvature_limited=False),
    CorpusFrameOutput(t=0.01, v_ego=20.0, raw_curvature=0.002, processed_curvature=0.0015,
                      measured_curvature=0.001, path_quality=0.85, path_reason="curvature_jump",
                      gated=True, demand_source="fallback_measured", curvature_limited=False),
    CorpusFrameOutput(t=0.02, v_ego=20.0, raw_curvature=0.001, processed_curvature=0.001,
                      measured_curvature=0.001, path_quality=0.50, path_reason="low_lane_confidence",
                      gated=True, demand_source="fallback_measured", curvature_limited=True),
    CorpusFrameOutput(t=0.03, v_ego=20.0, raw_curvature=0.000, processed_curvature=0.000,
                      measured_curvature=0.000, path_quality=0.30, path_reason="invalid_path",
                      gated=True, demand_source="fallback_measured", curvature_limited=False),
  )

  metrics = _compute_metrics(outputs)

  assert metrics.gating_rate == pytest.approx(0.75)
  assert metrics.fallback_rate == pytest.approx(0.75)
  assert metrics.curvature_limited_rate == pytest.approx(0.25)
  assert metrics.source_distribution["model_path"] == pytest.approx(0.25)
  assert metrics.source_distribution["fallback_measured"] == pytest.approx(0.75)
  assert metrics.curvature_rmse > 0.0
  assert 0.0 <= metrics.path_quality_p5 <= metrics.path_quality_p50 <= metrics.path_quality_p95 <= 1.0


def test_compute_metrics_returns_empty_for_no_outputs():
  metrics = _compute_metrics(())
  assert metrics == _empty_metrics()


# ---------- corpus runner ----------


def test_run_corpus_with_zero_routes_returns_empty_report():
  report = run_corpus([])
  assert report.routes == ()
  assert report.aggregate_metrics == {}


def test_run_corpus_runs_synthetic_route(monkeypatch):
  baseline = _fake_route_messages(5, desired_curvature=0.0005)
  original_load_route_msgs = corpus_module.load_route_msgs
  corpus_module.load_route_msgs = lambda route, qlog=False: baseline
  try:
    report = run_corpus(["fake/route"], qlog=True)
  finally:
    corpus_module.load_route_msgs = original_load_route_msgs

  assert len(report.routes) == 1
  result = report.routes[0]
  assert result.route == "fake/route"
  assert result.error is None
  assert result.frame_count == 5
  assert result.metrics.gating_rate >= 0.0
  assert result.metrics.fallback_rate >= 0.0
  assert result.profile.source == "fake/route"
  assert "gating_rate_mean" in report.aggregate_metrics
  assert report.aggregate_metrics["route_count"] == 1.0
  assert report.aggregate_metrics["total_frames"] == 5.0


# ---------- CLI ----------


def test_main_json_output_on_synthetic_route(monkeypatch):
  baseline = _fake_route_messages(5, desired_curvature=0.0005)
  original_load_route_msgs = corpus_module.load_route_msgs
  corpus_module.load_route_msgs = lambda route, qlog=False: baseline
  stdout = io.StringIO()
  try:
    with contextlib.redirect_stdout(stdout):
      main(["fake/route", "--json", "--qlog"])
  finally:
    corpus_module.load_route_msgs = original_load_route_msgs

  payload = json.loads(stdout.getvalue())
  assert len(payload["routes"]) == 1
  assert payload["routes"][0]["route"] == "fake/route"
  assert payload["routes"][0]["frame_count"] == 5
  assert "aggregate_metrics" in payload
  assert "gating_rate_mean" in payload["aggregate_metrics"]


def test_main_writes_output_file(tmp_path, monkeypatch):
  baseline = _fake_route_messages(3, desired_curvature=0.0005)
  original_load_route_msgs = corpus_module.load_route_msgs
  corpus_module.load_route_msgs = lambda route, qlog=False: baseline
  output_path = tmp_path / "report.json"
  try:
    main(["fake/route", "--output", str(output_path)])
  finally:
    corpus_module.load_route_msgs = original_load_route_msgs

  payload = json.loads(output_path.read_text())
  assert len(payload["routes"]) == 1
  assert payload["routes"][0]["route"] == "fake/route"


def test_main_text_output_on_synthetic_route(monkeypatch):
  baseline = _fake_route_messages(4, desired_curvature=0.0005)
  original_load_route_msgs = corpus_module.load_route_msgs
  corpus_module.load_route_msgs = lambda route, qlog=False: baseline
  stdout = io.StringIO()
  try:
    with contextlib.redirect_stdout(stdout):
      main(["fake/route"])
  finally:
    corpus_module.load_route_msgs = original_load_route_msgs

  output = stdout.getvalue()
  assert "Drive Lab lateral corpus report" in output
  assert "fake/route" in output
  assert "gating_rate" in output


def test_main_exits_nonzero_on_route_error(monkeypatch):
  original_load_route_msgs = corpus_module.load_route_msgs
  corpus_module.load_route_msgs = lambda route, qlog=False: (_ for _ in ()).throw(RuntimeError("boom"))
  try:
    code = main(["bad/route"])
  finally:
    corpus_module.load_route_msgs = original_load_route_msgs

  assert code == 1
