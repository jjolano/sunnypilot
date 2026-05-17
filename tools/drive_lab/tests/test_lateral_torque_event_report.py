import math
from types import SimpleNamespace

import pytest

from openpilot.tools.drive_lab.lateral_torque_event_report import (
  build_lateral_torque_ab_report,
  build_lateral_torque_event_report,
  build_lateral_torque_lag_report,
  load_lateral_torque_event_report,
  render_lateral_torque_ab_report,
  render_lateral_torque_lag_report,
  render_lateral_torque_event_report,
  save_lateral_torque_event_report,
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


def sample_msgs(t_s: float, output: float, *, unshaped: float | None = None, steering_angle: float = 0.0,
                 desired_accel: float = 0.05, actual_accel: float = 0.02, shaping_reason: int = 0,
                 shaping_active: bool = False, steer_limited: bool = False, learner_confidence: float = 0.0):
  adaptive = SimpleNamespace(
    shapingActive=shaping_active,
    shapingReason=shaping_reason,
    unshapedOutput=output if unshaped is None else unshaped,
    outputCap=0.8 if shaping_active else 1.0,
    releaseActive=False,
    steerLimitLimited=steer_limited,
    steerLimitError=0.15 if steer_limited else 0.0,
    modelConfidence=learner_confidence,
  )
  torque_state = SimpleNamespace(
    active=True,
    output=output,
    desiredLateralAccel=desired_accel,
    actualLateralAccel=actual_accel,
    adaptiveTorqueState=adaptive,
  )
  return [
    msg("carState", t_s, vEgo=18.0, steeringPressed=False, leftBlinker=False, rightBlinker=False, steeringAngleDeg=steering_angle,
        steeringRateDeg=0.0),
    msg("carControl", t_s, latActive=True),
    msg("carOutput", t_s, actuatorsOutput=SimpleNamespace(torque=output * 0.8)),
    msg("modelV2", t_s, meta=SimpleNamespace(laneChangeState=FakeEnum("off"))),
    msg("controlsState", t_s, lateralControlState=FakeUnion(torqueState=torque_state)),
  ]


def test_lateral_torque_event_report_classifies_controller_unshaped_reversal(tmp_path):
  msgs = []
  for i in range(120):
    t = i * 0.1
    sign = 1.0 if (i // 2) % 2 == 0 else -1.0
    msgs.extend(sample_msgs(
      t,
      output=0.22 * sign,
      unshaped=0.24 * sign,
      steering_angle=1.2 * sign,
      desired_accel=0.05,
      actual_accel=0.04 * sign,
    ))

  report = build_lateral_torque_event_report(msgs, source="synthetic", max_events=4)
  path = tmp_path / "lateral-torque-events.json"
  save_lateral_torque_event_report(report, path)
  loaded = load_lateral_torque_event_report(path)
  rendered = render_lateral_torque_event_report(loaded)

  assert loaded.top_events
  assert loaded.top_events[0].likely_source == "controller_unshaped_reversal"
  assert loaded.top_events[0].unshaped_output_reversals >= loaded.top_events[0].output_reversals - 1
  assert "Lateral torque event report" in rendered


def test_lateral_torque_event_report_classifies_shaping_or_actuator_limit():
  msgs = []
  for i in range(80):
    t = i * 0.1
    sign = 1.0 if (i // 2) % 2 == 0 else -1.0
    msgs.extend(sample_msgs(
      t,
      output=0.16 * sign,
      unshaped=0.22 * sign,
      steering_angle=0.8 * sign,
      shaping_reason=512,
      shaping_active=True,
      steer_limited=True,
    ))

  report = build_lateral_torque_event_report(msgs, source="synthetic", max_events=4)

  assert report.top_events
  assert report.top_events[0].likely_source == "safety_shaping_or_actuator_limit"
  assert report.top_events[0].shaping_reason_counts["STEERING_RATE_COMFORT"] > 0


def test_lateral_torque_event_report_filters_inactive_samples():
  msgs = []
  for i in range(20):
    t = i * 0.1
    adaptive = SimpleNamespace(shapingActive=False, shapingReason=0, unshapedOutput=0.2, outputCap=1.0)
    torque_state = SimpleNamespace(active=True, output=0.2, desiredLateralAccel=0.1, actualLateralAccel=0.1, adaptiveTorqueState=adaptive)
    msgs.extend([
      msg("carState", t, vEgo=18.0, steeringPressed=False, leftBlinker=False, rightBlinker=False, steeringAngleDeg=10.0,
          steeringRateDeg=0.0),
      msg("carControl", t, latActive=False),
      msg("carOutput", t, actuatorsOutput=SimpleNamespace(torque=0.0)),
      msg("modelV2", t, meta=SimpleNamespace(laneChangeState=FakeEnum("off"))),
      msg("controlsState", t, lateralControlState=FakeUnion(torqueState=torque_state)),
    ])

  report = build_lateral_torque_event_report(msgs)

  assert report.active_percent == 0.0
  assert not report.top_events


def lag_msgs(delay_s: float, *, learner_confidence: float = 0.0):
  msgs = []
  dt = 0.1
  for i in range(240):
    t = i * dt
    desired = 0.35 * math.sin(t * 0.8)
    actual = 0.35 * math.sin((t - delay_s) * 0.8)
    msgs.extend(sample_msgs(
      t,
      output=0.15 * (1.0 if desired >= 0.0 else -1.0),
      desired_accel=desired,
      actual_accel=actual,
      learner_confidence=learner_confidence,
    ))
  return msgs


def test_lateral_torque_lag_report_estimates_tracking_lag():
  report = build_lateral_torque_lag_report(lag_msgs(0.2), source="delayed")
  rendered = render_lateral_torque_lag_report(report)
  curve = next(metric for metric in report.metrics if metric.segment == "curve")

  assert "Lateral torque lag report" in rendered
  assert curve.best_lag_s == pytest.approx(0.2, abs=0.11)
  assert curve.abs_error_p95 > 0.0


def test_lateral_torque_lag_report_splits_warm_learner_segments():
  report = build_lateral_torque_lag_report(lag_msgs(0.1, learner_confidence=0.8), source="warm")
  warm = next(metric for metric in report.metrics if metric.segment == "warm")
  cold = next(metric for metric in report.metrics if metric.segment == "cold")

  assert warm.sample_count > 0
  assert cold.sample_count == 0


def test_lateral_torque_ab_report_shows_candidate_lag_delta():
  report = build_lateral_torque_ab_report(lag_msgs(0.3), lag_msgs(0.1), already_sorted=True)
  rendered = render_lateral_torque_ab_report(report)

  assert "Lateral torque A/B report" in rendered
  assert report.deltas["curve.best_lag_s"] < 0.0
