from types import SimpleNamespace

import pytest

from openpilot.tools.drive_lab import (
  analyze_longitudinal_lateral_route,
  lateral_oscillation_profile,
  lateral_performance_gate,
  lateral_torque_event_report,
)
from openpilot.tools.drive_lab.compare_manual_lateral_timing import build_lateral_timing_frames
from openpilot.tools.drive_lab.manual_lateral_baseline import build_manual_lateral_samples


class FakeMsg(SimpleNamespace):
  def which(self):
    return self.kind


class FakeUnion(SimpleNamespace):
  def which(self):
    return "torqueState"


def msg(kind: str, t_s: float, **payload):
  return FakeMsg(kind=kind, logMonoTime=int(t_s * 1e9), **{kind: SimpleNamespace(**payload)})


def lateral_msgs(commit: str) -> list[FakeMsg]:
  adaptive = SimpleNamespace(unshapedOutput=0.1, outputCap=1.0)
  torque_state = SimpleNamespace(
    active=True,
    version=21,
    output=0.1,
    desiredLateralAccel=0.0,
    actualLateralAccel=0.0,
    adaptiveTorqueState=adaptive,
  )
  model_path = SimpleNamespace(
    active=True,
    gated=False,
    quality=1.0,
    reason="clean",
    rawDesiredCurvature=0.012,
    conditionedDesiredCurvature=0.0,
    processedDesiredCurvature=0.009,
  )
  return [
    msg("initData", 0.0, gitCommit=commit),
    msg(
      "carState",
      1.0,
      vEgo=10.0,
      curvature=0.0,
      steeringAngleDeg=0.0,
      steeringRateDeg=0.0,
      steeringTorque=0.0,
      steeringTorqueEps=0.0,
      steeringPressed=False,
      leftBlinker=False,
      rightBlinker=False,
      standstill=False,
    ),
    msg("carControl", 1.0, latActive=True, actuators=SimpleNamespace(torque=0.1)),
    msg("carOutput", 1.0, actuatorsOutput=SimpleNamespace(torque=0.1)),
    msg(
      "modelV2",
      1.0,
      action=SimpleNamespace(desiredCurvature=0.012),
      meta=SimpleNamespace(laneChangeState="off"),
      position=SimpleNamespace(y=[0.0]),
      laneLines=[SimpleNamespace(y=[-3.6]), SimpleNamespace(y=[-1.8]), SimpleNamespace(y=[1.8]), SimpleNamespace(y=[3.6])],
    ),
    msg(
      "controlsState",
      1.0,
      active=True,
      curvature=0.0,
      desiredCurvature=0.009,
      lateralControlState=FakeUnion(torqueState=torque_state),
      modelPathState=model_path,
    ),
  ]


@pytest.mark.parametrize("commit", ("4cf9f40540", "d9ee43ce0f"))
def test_all_consumers_use_controller_facing_processed_demand(commit: str):
  msgs = lateral_msgs(commit)
  timing = build_lateral_timing_frames("route", msgs)
  gate = lateral_performance_gate._extract_gate_samples(msgs)
  oscillation = lateral_oscillation_profile._extract_lateral_samples(msgs)
  rows = analyze_longitudinal_lateral_route.extract_rows(msgs, analyze_longitudinal_lateral_route.AnalysisWindow(0.0, 2.0, "test"))
  baseline = build_manual_lateral_samples("route", msgs)
  torque = lateral_torque_event_report._extract_torque_samples(msgs)

  values = {
    "timing": timing[0].processed_lat_accel / 100.0,
    "gate": gate[0].processed_desired_curvature,
    "oscillation": oscillation[0].processed_desired_curvature,
    "route_csv": next(row for row in rows if row["msg_type"] == "controlsState")["model_path_processed_curvature"],
    "baseline": baseline[0].processed_desired_curvature,
    "torque": torque[0].processed_desired_curvature,
  }
  assert values == pytest.approx(dict.fromkeys(values, 0.009))
  assert timing[0].t == pytest.approx(1.0)
  assert baseline[0].t == pytest.approx(1.0)
