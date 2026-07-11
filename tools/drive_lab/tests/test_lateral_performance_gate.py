import math
from types import SimpleNamespace

import numpy as np
import pytest

from openpilot.tools.drive_lab.lateral_performance_gate import (
  ACTUATION_DRIVEN_WANDER,
  DEMAND_DRIVEN_WANDER,
  INSUFFICIENT_EVIDENCE,
  LOW_SPEED_LATERAL_DOMINANT,
  PATH_WANDER_DOMINANT,
  TORQUE_EVENT_DOMINANT,
  build_lateral_performance_gate,
  build_lateral_performance_gate_ab_report,
  _lane_center_offset_y,
  _recenter_candidate,
  load_lateral_performance_gate,
  render_lateral_performance_gate,
  render_lateral_performance_gate_ab_report,
  save_lateral_performance_gate,
)


class FakeMsg(SimpleNamespace):
  def which(self):
    return self.kind


class FakeUnion(SimpleNamespace):
  def which(self):
    return "torqueState"


class FakeEnum:
  def __init__(self, name: str):
    self.name = name


def msg(kind: str, t_s: float, **payload):
  return FakeMsg(kind=kind, logMonoTime=int(t_s * 1e9), **{kind: SimpleNamespace(**payload)})


def lane_lines(offset: float):
  return [
    SimpleNamespace(y=[-3.6 + offset]),
    SimpleNamespace(y=[-1.8 + offset]),
    SimpleNamespace(y=[1.8 + offset]),
    SimpleNamespace(y=[3.6 + offset]),
  ]


def sample_msgs(t_s: float, *, v_ego: float = 18.0, raw_curvature: float = 0.0,
                processed_curvature: float | None = None, actual_curvature: float | None = None,
                steering_angle: float = 0.0, steering_rate: float = 0.0, output: float = 0.0,
                unshaped_output: float | None = None, desired_accel: float | None = None,
                actual_accel: float | None = None, shaping_active: bool = False, shaping_reason: int = 0,
                steer_limited: bool = False, lane_change_state: str = "off", left_blinker: bool = False,
                right_blinker: bool = False, path_quality: float = 1.0, path_gated: bool = False,
                offset_y: float = 0.0, torque_version: int = 21):
  processed = raw_curvature if processed_curvature is None else processed_curvature
  actual = raw_curvature * 0.95 if actual_curvature is None else actual_curvature
  desired_lat_accel = processed * v_ego ** 2 if desired_accel is None else desired_accel
  actual_lat_accel = actual * v_ego ** 2 if actual_accel is None else actual_accel
  adaptive = SimpleNamespace(
    shapingActive=shaping_active,
    shapingReason=shaping_reason,
    governorReason=0,
    unshapedOutput=output if unshaped_output is None else unshaped_output,
    outputCap=0.8 if shaping_active else 1.0,
    releaseActive=False,
    steerLimitLimited=steer_limited,
    steerLimitError=0.2 if steer_limited else 0.0,
    modelConfidence=0.0,
  )
  torque_state = SimpleNamespace(
    active=True,
    version=torque_version,
    output=output,
    desiredLateralAccel=desired_lat_accel,
    actualLateralAccel=actual_lat_accel,
    adaptiveTorqueState=adaptive,
  )
  return [
    msg("carState", t_s, vEgo=v_ego, steeringPressed=False, leftBlinker=left_blinker, rightBlinker=right_blinker,
        steeringAngleDeg=steering_angle, steeringRateDeg=steering_rate),
    msg("carControl", t_s, latActive=True),
    msg("carOutput", t_s, actuatorsOutput=SimpleNamespace(torque=output * 0.8)),
    msg("modelV2", t_s,
        meta=SimpleNamespace(laneChangeState=FakeEnum(lane_change_state)),
        position=SimpleNamespace(y=[0.0, offset_y * 0.2, offset_y * 0.4, offset_y * 0.6, offset_y * 0.8, offset_y]),
        laneLines=lane_lines(offset_y)),
    msg("controlsState", t_s,
        curvature=actual,
        desiredCurvature=processed,
        lateralControlState=FakeUnion(torqueState=torque_state),
        modelPathState=SimpleNamespace(
          rawDesiredCurvature=raw_curvature,
          processedDesiredCurvature=processed,
          gated=path_gated,
          quality=path_quality,
        )),
  ]


def path_wander_msgs(*, lane_change_state: str = "off", offset: bool = False):
  msgs = []
  previous_angle = 0.0
  for i in range(420):
    t = i * 0.1
    raw = 0.0016 * math.sin(2.0 * math.pi * t / 22.0)
    steering = raw * 3800.0
    steering_rate = (steering - previous_angle) / 0.1 if i > 0 else 0.0
    previous_angle = steering
    offset_y = 0.25 * math.sin(2.0 * math.pi * t / 30.0) if offset else 0.0
    msgs.extend(sample_msgs(
      t,
      v_ego=20.0,
      raw_curvature=raw,
      actual_curvature=raw * 0.94,
      steering_angle=steering,
      steering_rate=steering_rate,
      output=raw * 120.0,
      lane_change_state=lane_change_state,
      offset_y=offset_y,
    ))
  return msgs


def test_lateral_performance_gate_classifies_demand_driven_wander_and_round_trips(tmp_path):
  report = build_lateral_performance_gate(path_wander_msgs(), source="wander", window_s=20.0, step_s=5.0)
  path = tmp_path / "gate.json"
  save_lateral_performance_gate(report, path)
  loaded = load_lateral_performance_gate(path)
  rendered = render_lateral_performance_gate(loaded)

  assert loaded.dominant_failure_class == PATH_WANDER_DOMINANT
  assert loaded.branch_recommendation == "feat/lateral-control"
  assert loaded.wander_candidate_windows
  assert loaded.wander_candidate_windows[0].cause == DEMAND_DRIVEN_WANDER
  assert "dominant failure class" in rendered


def test_lateral_performance_gate_uses_qlog_safe_unknown_lane_policy():
  safe = build_lateral_performance_gate(path_wander_msgs(lane_change_state="unknown"), source="qlog", window_s=20.0, step_s=5.0)
  strict = build_lateral_performance_gate(
    path_wander_msgs(lane_change_state="unknown"), source="strict", qlog_safe_lane_policy=False, window_s=20.0, step_s=5.0,
  )

  assert safe.wander_candidate_windows
  assert safe.lane_state_unknown_percent == pytest.approx(100.0)
  assert safe.wander_candidate_windows[0].confidence == "medium"
  assert strict.dominant_failure_class == INSUFFICIENT_EVIDENCE
  assert not strict.wander_candidate_windows


def test_lateral_performance_gate_classifies_torque_event_dominant():
  msgs = []
  previous_angle = 0.0
  for i in range(180):
    t = i * 0.1
    sign = 1.0 if (i // 2) % 2 == 0 else -1.0
    steering = sign * 0.8
    steering_rate = (steering - previous_angle) / 0.1 if i > 0 else 0.0
    previous_angle = steering
    msgs.extend(sample_msgs(
      t,
      v_ego=22.0,
      raw_curvature=0.00005,
      actual_curvature=0.00004 * sign,
      steering_angle=steering,
      steering_rate=steering_rate,
      output=0.18 * sign,
      unshaped_output=0.28 * sign,
      shaping_active=True,
      shaping_reason=1 << 9,
      steer_limited=True,
    ))

  report = build_lateral_performance_gate(msgs, source="torque", window_s=8.0, step_s=2.0)

  assert report.dominant_failure_class == TORQUE_EVENT_DOMINANT
  assert report.branch_recommendation == "feat/lateral-control"
  assert report.torque_event_score > report.path_wander_score


def test_lateral_performance_gate_classifies_low_speed_lateral_dominant():
  msgs = []
  for i in range(140):
    t = i * 0.1
    msgs.extend(sample_msgs(
      t,
      v_ego=5.0,
      raw_curvature=0.008,
      processed_curvature=0.008,
      actual_curvature=0.0,
      steering_angle=5.0,
      output=0.2,
      desired_accel=0.55,
      actual_accel=0.0,
    ))

  report = build_lateral_performance_gate(msgs, source="low-speed", window_s=8.0, step_s=2.0)

  assert report.dominant_failure_class == LOW_SPEED_LATERAL_DOMINANT
  assert report.branch_recommendation == "feat/lateral-control"


def test_lateral_performance_gate_reports_recenter_overshoot_candidates():
  report = build_lateral_performance_gate(path_wander_msgs(offset=True), source="recenter", window_s=30.0, step_s=5.0)

  assert report.recenter_overshoot_candidates
  candidate = report.recenter_overshoot_candidates[0]
  assert candidate.confidence == "high"
  assert candidate.offset_crossings >= 1
  assert candidate.correction_reversals >= 1


def test_recenter_offset_agreement_excludes_subthreshold_samples_from_denominator():
  alternating = np.tile([0.25, -0.25], 10)
  offsets = np.concatenate([np.zeros(20), alternating])
  cols = {
    "t": np.arange(40, dtype=float) * 0.1,
    "model_path_offset_y": offsets,
    "lane_center_offset_y": offsets,
    "processed_desired_curvature": np.tile([0.001, -0.001], 20),
    "lane_state_unknown": np.zeros(40),
  }

  candidate = _recenter_candidate(cols, np.ones(40, dtype=bool))

  assert candidate is not None
  assert candidate.offset_agreement_percent == pytest.approx(100.0)


def test_lane_center_offset_uses_same_horizon_as_model_path_offset():
  model = SimpleNamespace(laneLines=[
    SimpleNamespace(y=[-3.6] * 6),
    SimpleNamespace(y=[-1.8, -1.6, -1.4, -1.2, -1.0, -0.8]),
    SimpleNamespace(y=[1.8, 2.0, 2.2, 2.4, 2.6, 2.8]),
    SimpleNamespace(y=[3.6] * 6),
  ])

  assert _lane_center_offset_y(model) == pytest.approx(1.0)


def test_lateral_performance_gate_ab_report_renders_deltas():
  baseline = path_wander_msgs()
  candidate = []
  for i in range(220):
    t = i * 0.1
    candidate.extend(sample_msgs(t, v_ego=20.0, raw_curvature=0.0, actual_curvature=0.0))

  report = build_lateral_performance_gate_ab_report(baseline, candidate, already_sorted=True)
  rendered = render_lateral_performance_gate_ab_report(report)

  assert report.deltas["path_wander_score"] < 0.0
  assert "Deltas candidate-baseline" in rendered


def test_lateral_performance_gate_can_classify_actuation_driven_wander():
  msgs = []
  for i in range(360):
    t = i * 0.1
    actual = 0.0012 * math.sin(2.0 * math.pi * t / 20.0)
    msgs.extend(sample_msgs(
      t,
      v_ego=20.0,
      raw_curvature=0.0,
      processed_curvature=0.0,
      actual_curvature=actual,
      steering_angle=actual * 3600.0,
      output=0.0,
    ))

  report = build_lateral_performance_gate(msgs, source="actuation", window_s=20.0, step_s=5.0)

  assert report.wander_candidate_windows
  assert report.wander_candidate_windows[0].cause == ACTUATION_DRIVEN_WANDER
