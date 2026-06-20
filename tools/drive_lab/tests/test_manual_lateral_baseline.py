from types import SimpleNamespace

import pytest

from openpilot.tools.drive_lab.manual_lateral_baseline import (
  build_manual_lateral_samples,
  render_manual_lateral_baseline,
  summarize_manual_lateral_baseline,
)


class FakeMsg:
  def __init__(self, typ, payload, t_s):
    self._typ = typ
    setattr(self, typ, payload)
    self.logMonoTime = int(t_s * 1e9)

  def which(self):
    return self._typ


def cs(**kwargs):
  defaults = dict(vEgo=10.0, steeringAngleDeg=1.0, steeringPressed=False, leftBlinker=False, rightBlinker=False, standstill=False, steeringTorque=0.0, curvature=0.01)
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
  assert len(samples) == 2
  assert samples[0].mode == "manual"
  assert samples[1].mode == "engaged"
  assert samples[1].lat_error == pytest.approx((0.03 - 0.015) * 12.0 * 12.0)


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
