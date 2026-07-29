import json
import sys
from types import SimpleNamespace

import pytest

from openpilot.tools.drive_lab import analyze_longitudinal_lateral_route, manual_lateral_baseline
from openpilot.tools.drive_lab.manual_lateral_baseline import (
  _phase,
  build_manual_lateral_samples,
  render_manual_lateral_baseline,
  summarize_manual_lateral_baseline,
)


def test_curve_phase_uses_magnitude_for_left_and_right_turns():
  assert _phase(1.0, 0.2) == "entry"
  assert _phase(-1.0, -0.2) == "entry"
  assert _phase(1.0, -0.2) == "exit"
  assert _phase(-1.0, 0.2) == "exit"


class FakeMsg:
  def __init__(self, typ, payload, t_s):
    self._typ = typ
    setattr(self, typ, payload)
    self.logMonoTime = int(t_s * 1e9)

  def which(self):
    return self._typ


def cs(**kwargs):
  defaults = dict(
    vEgo=10.0, steeringAngleDeg=1.0, steeringPressed=False, leftBlinker=False,
    rightBlinker=False, standstill=False, steeringTorque=0.0, curvature=0.01,
  )
  defaults.update(kwargs)
  return SimpleNamespace(**defaults)


def cc(lat_active=True, **kwargs):
  defaults = dict(latActive=lat_active)
  defaults.update(kwargs)
  return SimpleNamespace(**defaults)


def ctrl_state(**kwargs):
  defaults = dict(
    desiredCurvature=0.02,
    latActive=True,
    lateralControlState=SimpleNamespace(torqueState=SimpleNamespace(active=True)),
    modelPathState=SimpleNamespace(processedDesiredCurvature=0.03, rawDesiredCurvature=0.04, quality=0.8, gated=False, reason="clean"),
  )
  defaults.update(kwargs)
  return SimpleNamespace(**defaults)


def model_v2(**kwargs):
  defaults = dict(modelPath=SimpleNamespace(quality=0.8, gated=False, state="active"))
  defaults.update(kwargs)
  return SimpleNamespace(**defaults)


def test_manual_vs_engaged_split_and_filters():
  msgs = [
    FakeMsg("carState", cs(vEgo=2.5), 0.0),
    FakeMsg("carControl", cc(True), 0.0),
    FakeMsg("controlsState", ctrl_state(), 0.0),
    FakeMsg("carState", cs(vEgo=10.0, curvature=0.01), 1.0),
    FakeMsg("carControl", cc(False), 1.0),
    FakeMsg("controlsState", ctrl_state(latActive=False, desiredCurvature=0.02), 1.0),
    FakeMsg("carState", cs(vEgo=12.0, steeringAngleDeg=2.0, curvature=0.015), 2.0),
    FakeMsg("carControl", cc(True), 2.0),
    FakeMsg("controlsState", ctrl_state(latActive=True, desiredCurvature=0.025), 2.0),
  ]

  samples = build_manual_lateral_samples("route-a", msgs)
  assert len(samples) == 3
  assert samples[0].exclusion_reason == "low_speed"
  assert samples[1].mode == "manual"
  assert samples[2].mode == "engaged"
  assert samples[2].lat_error == pytest.approx((0.025 - 0.015) * 12.0 * 12.0)

  summary = summarize_manual_lateral_baseline(samples)
  assert summary.sample_count == 3
  assert summary.excluded_sample_count == 1
  assert summary.low_speed_excluded_count == 1
  assert summary.manual_sample_count == 1
  assert summary.engaged_sample_count == 1


def test_summary_renders_conservative_caveat_and_bins():
  msgs = [
    FakeMsg("carState", cs(vEgo=10.0, curvature=0.01), 1.0),
    FakeMsg("carControl", cc(False), 1.0),
    FakeMsg("controlsState", ctrl_state(latActive=False, desiredCurvature=0.02), 1.0),
    FakeMsg("modelV2", model_v2(), 1.0),
    FakeMsg("carState", cs(vEgo=20.0, steeringAngleDeg=3.0, curvature=-0.02), 2.0),
    FakeMsg("carControl", cc(True), 2.0),
    FakeMsg("controlsState", ctrl_state(latActive=True, desiredCurvature=0.03), 2.0),
  ]

  samples = build_manual_lateral_samples("route-a", msgs)
  summary = summarize_manual_lateral_baseline(samples, source="route-a")
  rendered = render_manual_lateral_baseline(summary)

  assert summary.engaged_sample_count == 1
  assert summary.manual_sample_count == 1
  assert summary.speed_bins[8].sample_count == 1
  assert "manual lateral is a descriptive envelope" in rendered
  assert "no live tuning conclusions" in rendered


def test_curve_side_means_track_left_and_right():
  msgs = [
    FakeMsg("carState", cs(vEgo=10.0, curvature=0.01), 1.0),
    FakeMsg("carControl", cc(False), 1.0),
    FakeMsg("controlsState", ctrl_state(latActive=False, desiredCurvature=0.02), 1.0),
    FakeMsg("carState", cs(vEgo=10.0, curvature=-0.02, steeringAngleDeg=1.5), 2.0),
    FakeMsg("carControl", cc(True), 2.0),
    FakeMsg("controlsState", ctrl_state(latActive=True, desiredCurvature=0.03), 2.0),
  ]

  summary = summarize_manual_lateral_baseline(build_manual_lateral_samples("route-a", msgs), source="route-a")
  assert summary.curve_side_means["manual:left"]["current_lat_accel"] == pytest.approx(1.0)
  assert summary.curve_side_means["engaged:right"]["current_lat_accel"] == pytest.approx(-2.0)


def test_bucket_duration_counts_occupancy_not_first_to_last_span():
  msgs = []
  for t, speed in ((0.0, 10.0), (1.0, 10.0), (50.0, 20.0), (100.0, 10.0)):
    msgs.extend([
      FakeMsg("carState", cs(vEgo=speed), t),
      FakeMsg("carControl", cc(False), t),
      FakeMsg("controlsState", ctrl_state(latActive=False), t),
    ])

  summary = summarize_manual_lateral_baseline(build_manual_lateral_samples("route-a", msgs))
  bucket = next(item for item in summary.speed_bins if item.label == "manual 8-12 m/s")
  assert bucket.sample_count == 3
  assert bucket.duration_s == pytest.approx(1.0)


def test_excluded_interval_breaks_derivative_chain():
  msgs = []
  for t, speed, curvature in ((0.0, 10.0, 0.01), (1.0, 2.0, 0.02), (2.0, 10.0, 0.03)):
    msgs.extend([
      FakeMsg("carState", cs(vEgo=speed, curvature=curvature), t),
      FakeMsg("carControl", cc(False), t),
      FakeMsg("controlsState", ctrl_state(latActive=False), t),
    ])

  samples = build_manual_lateral_samples("route-a", msgs)

  assert samples[1].exclusion_reason == "low_speed"
  assert samples[2].lat_jerk is None
  assert samples[2].steering_rate_deg_s is None


def test_cli_keeps_independent_input_state_separate(monkeypatch, capsys):
  route_msgs = {
    "route-a": [FakeMsg("carState", cs(), 0.0)],
    "route-b": [FakeMsg("controlsState", ctrl_state(), 0.0)],
  }
  calls = []

  def fake_log_reader(route, **_kwargs):
    calls.append(route)
    if isinstance(route, list):
      return [msg for name in route for msg in route_msgs[name]]
    return route_msgs[route]

  monkeypatch.setattr(analyze_longitudinal_lateral_route, "resolve_inputs", lambda *_args, **_kwargs: ["route-a", "route-b"])
  monkeypatch.setattr("openpilot.tools.lib.logreader.LogReader", fake_log_reader)
  monkeypatch.setattr(sys, "argv", ["manual_lateral_baseline.py", "route-a", "route-b", "--json"])

  manual_lateral_baseline.main()

  report = json.loads(capsys.readouterr().out)
  assert calls == ["route-a", "route-b"]
  assert report["sample_count"] == 0
