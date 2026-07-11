import math
from types import SimpleNamespace

import pytest

from openpilot.tools.drive_lab.lateral_torque_event_report import (
  build_lateral_low_speed_report,
  build_lateral_torque_ab_report,
  build_lateral_torque_event_report,
  build_lateral_torque_lag_report,
  render_lateral_low_speed_report,
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
                 shaping_active: bool = False, steer_limited: bool = False, learner_confidence: float = 0.0,
                 v_ego: float = 18.0, steering_rate: float = 0.0, left_blinker: bool = False,
                 right_blinker: bool = False, lane_change_state: str = "off", path_gated: bool = False,
                 path_reason: str = "ok", path_quality: float = 1.0, raw_curvature: float = 0.0,
                 processed_curvature: float = 0.0, desired_curvature: float = 0.0,
                 include_model_path: bool = True, torque_version: int = 2, governor_reason: int = 0):
  adaptive = SimpleNamespace(
    shapingActive=shaping_active,
    shapingReason=shaping_reason,
    governorReason=governor_reason,
    unshapedOutput=output if unshaped is None else unshaped,
    outputCap=0.8 if shaping_active else 1.0,
    releaseActive=False,
    steerLimitLimited=steer_limited,
    steerLimitError=0.15 if steer_limited else 0.0,
    modelConfidence=learner_confidence,
  )
  torque_state = SimpleNamespace(
    active=True,
    version=torque_version,
    output=output,
    desiredLateralAccel=desired_accel,
    actualLateralAccel=actual_accel,
    adaptiveTorqueState=adaptive,
  )
  controls_state = {
    "curvature": processed_curvature,
    "desiredCurvature": desired_curvature,
    "lateralControlState": FakeUnion(torqueState=torque_state),
  }
  if include_model_path:
    controls_state["modelPathState"] = SimpleNamespace(
      gated=path_gated,
      quality=path_quality,
      reason=FakeEnum(path_reason),
      rawDesiredCurvature=raw_curvature,
      processedDesiredCurvature=processed_curvature,
    )
  return [
    msg("carState", t_s, vEgo=v_ego, steeringPressed=False, leftBlinker=left_blinker, rightBlinker=right_blinker,
        steeringAngleDeg=steering_angle, steeringRateDeg=steering_rate),
    msg("carControl", t_s, latActive=True),
    msg("carOutput", t_s, actuatorsOutput=SimpleNamespace(torque=output * 0.8)),
    msg("modelV2", t_s, meta=SimpleNamespace(laneChangeState=FakeEnum(lane_change_state))),
    msg("controlsState", t_s, **controls_state),
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


def test_lateral_torque_event_report_decodes_shaper_and_governor_reasons_separately():
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
      governor_reason=(1 << 1) | (1 << 9),
      shaping_active=True,
      steer_limited=True,
      torque_version=21,
    ))

  report = build_lateral_torque_event_report(msgs, source="synthetic", max_events=4)
  rendered = render_lateral_torque_event_report(report)

  assert report.top_events
  event = report.top_events[0]
  assert event.shaping_reason_counts["STEERING_RATE_COMFORT"] > 0
  assert event.governor_reason_counts["SLEW_LIMITED"] > 0
  assert event.governor_reason_counts["UNDER_RESPONSE_FLOOR"] > 0
  assert "shaper_reasons=" in rendered
  assert "governor_reasons=" in rendered


def test_lateral_torque_event_report_decodes_v21_governor_reason_names():
  msgs = []
  for i in range(80):
    t = i * 0.1
    sign = 1.0 if (i // 2) % 2 == 0 else -1.0
    msgs.extend(sample_msgs(
      t,
      output=0.12 * sign,
      unshaped=0.18 * sign,
      steering_angle=0.6 * sign,
      governor_reason=(1 << 5) | (1 << 9),
      torque_version=21,
    ))

  report = build_lateral_torque_event_report(msgs, source="synthetic", max_events=2)

  assert report.top_events
  event = report.top_events[0]
  assert event.governor_reason_counts["OVER_RESPONSE"] > 0
  assert event.governor_reason_counts["UNDER_RESPONSE_FLOOR"] > 0


def test_lateral_torque_event_report_decodes_v21_guarded_under_response_reason():
  msgs = []
  for i in range(80):
    t = i * 0.1
    sign = 1.0 if (i // 2) % 2 == 0 else -1.0
    msgs.extend(sample_msgs(
      t,
      output=0.12 * sign,
      unshaped=0.18 * sign,
      steering_angle=0.6 * sign,
      governor_reason=(1 << 11),
      torque_version=21,
    ))

  report = build_lateral_torque_event_report(msgs, source="synthetic", max_events=2)

  assert report.top_events
  assert report.top_events[0].governor_reason_counts["UNDER_RESPONSE_GUARDED"] > 0


def test_lateral_torque_event_report_decodes_v3_governor_reason_not_v2_shaper_reason():
  msgs = []
  for i in range(80):
    t = i * 0.1
    sign = 1.0 if (i // 2) % 2 == 0 else -1.0
    msgs.extend(sample_msgs(
      t,
      output=0.16 * sign,
      unshaped=0.22 * sign,
      steering_angle=0.8 * sign,
      shaping_reason=1 << 4,
      governor_reason=1 << 4,
      shaping_active=True,
      steer_limited=True,
      torque_version=3,
    ))

  report = build_lateral_torque_event_report(msgs, source="synthetic", max_events=4)

  assert report.top_events
  event = report.top_events[0]
  assert "NEAR_ISO_ACCEL" not in event.shaping_reason_counts
  assert event.governor_reason_counts["SAME_DIRECTION_LIMIT"] > 0


def test_lateral_torque_event_report_decodes_v4_governor_reason_names():
  msgs = []
  for i in range(80):
    t = i * 0.1
    sign = 1.0 if (i // 2) % 2 == 0 else -1.0
    msgs.extend(sample_msgs(
      t,
      output=0.16 * sign,
      unshaped=0.22 * sign,
      steering_angle=0.8 * sign,
      governor_reason=(1 << 7) | (1 << 8),
      shaping_active=True,
      steer_limited=True,
      torque_version=4,
    ))

  report = build_lateral_torque_event_report(msgs, source="synthetic", max_events=4)

  assert report.top_events
  event = report.top_events[0]
  assert event.governor_reason_counts["STALE_ACTUATOR_MISMATCH"] > 0
  assert event.governor_reason_counts["LOW_SPEED_UNDER_RESPONSE_RECOVERY"] > 0
  assert "UNDER_RESPONSE_FLOOR" not in event.governor_reason_counts


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


def test_lateral_torque_lag_right_turn_growth_is_entry_not_exit():
  msgs = []
  for i in range(20):
    desired = -0.02 * i
    msgs.extend(sample_msgs(i * 0.1, output=-0.1, v_ego=10.0,
                            desired_accel=desired, actual_accel=desired,
                            desired_curvature=desired / 100.0))

  report = build_lateral_torque_lag_report(msgs, already_sorted=True)
  entry = next(metric for metric in report.metrics if metric.segment == "entry")
  exit_ = next(metric for metric in report.metrics if metric.segment == "exit")
  assert entry.sample_count > 0
  assert exit_.sample_count == 0


def test_lateral_torque_lag_report_adds_high_curvature_low_quality_and_physics_metrics():
  msgs = []
  for i in range(20):
    t = i * 0.1
    v_ego = 10.0
    processed_curvature = 0.002 if i < 10 else 0.0
    current_curvature = 0.0015
    desired_accel = v_ego ** 2 * processed_curvature
    actual_accel = v_ego ** 2 * current_curvature
    msgs.extend(sample_msgs(
      t,
      output=0.12,
      v_ego=v_ego,
      desired_accel=desired_accel,
      actual_accel=actual_accel,
      path_gated=i % 2 == 0,
      path_reason="highPathStd" if i % 2 == 0 else "ok",
      path_quality=0.6 if i % 2 == 0 else 0.9,
      raw_curvature=processed_curvature,
      processed_curvature=processed_curvature,
      desired_curvature=processed_curvature,
    ))

  report = build_lateral_torque_lag_report(msgs, source="synthetic", already_sorted=True)
  high_curvature = next(metric for metric in report.metrics if metric.segment == "high_curvature")
  low_quality = next(metric for metric in report.metrics if metric.segment == "low_path_quality")

  assert high_curvature.sample_count > 0
  assert high_curvature.desired_lateral_accel_residual_abs_p95 == pytest.approx(0.0, abs=1e-9)
  assert high_curvature.actual_lateral_accel_residual_abs_p95 == pytest.approx(0.05, rel=1e-2)
  assert low_quality.sample_count == 10
  assert low_quality.model_path_low_quality_percent == pytest.approx(100.0)
  assert low_quality.model_path_reason_counts["highPathStd"] == 10


def test_lateral_torque_lag_report_renders_reason_counts():
  msgs = []
  for i in range(12):
    t = i * 0.1
    msgs.extend(sample_msgs(
      t,
      output=0.12,
      desired_accel=0.2,
      actual_accel=0.2,
      shaping_reason=512,
      governor_reason=(1 << 1),
      path_reason="ok",
      path_quality=0.8,
      steer_limited=True,
    ))

  report = build_lateral_torque_lag_report(msgs, source="synthetic", already_sorted=True)
  rendered = render_lateral_torque_lag_report(report)

  assert "jerk95=" in rendered
  assert "aresid95=" in rendered
  assert "path_low=" in rendered
  assert "shaper_reasons=" in rendered
  assert "governor_reasons=" in rendered


def test_lateral_torque_lag_report_splits_warm_learner_segments():
  report = build_lateral_torque_lag_report(lag_msgs(0.1, learner_confidence=0.8), source="warm")
  warm = next(metric for metric in report.metrics if metric.segment == "warm")
  cold = next(metric for metric in report.metrics if metric.segment == "cold")

  assert warm.sample_count > 0
  assert cold.sample_count == 0


def test_lateral_torque_lag_report_handles_missing_model_path_state_as_unknown():
  msgs = []
  for i in range(10):
    msgs.extend(sample_msgs(
      i * 0.1,
      output=0.1,
      desired_accel=0.1,
      actual_accel=0.1,
      include_model_path=False,
    ))

  report = build_lateral_torque_lag_report(msgs, source="synthetic", already_sorted=True)
  low_quality = next(metric for metric in report.metrics if metric.segment == "low_path_quality")

  assert low_quality.sample_count == 0
  assert low_quality.model_path_low_quality_percent == 0.0


def test_lateral_torque_ab_report_shows_candidate_lag_delta():
  report = build_lateral_torque_ab_report(lag_msgs(0.3), lag_msgs(0.1), already_sorted=True)
  rendered = render_lateral_torque_ab_report(report)

  assert "Lateral torque A/B report" in rendered
  assert report.deltas["curve.best_lag_s"] < 0.0


def test_low_speed_lateral_report_buckets_path_and_torque_metrics():
  msgs = []
  for i in range(24):
    t = i * 0.1
    sign = 1.0 if (i // 3) % 2 == 0 else -1.0
    msgs.extend(sample_msgs(
      t,
      output=0.10 * sign,
      unshaped=0.12 * sign,
      v_ego=4.0,
      steering_angle=6.0 * sign,
      steering_rate=12.0 * sign,
      desired_accel=0.12 * sign,
      actual_accel=0.08 * sign,
      path_gated=i % 2 == 0,
      path_reason="highPathStd" if i % 2 == 0 else "ok",
      path_quality=0.6 if i % 2 == 0 else 1.0,
      raw_curvature=0.004 * sign,
      processed_curvature=0.003 * sign,
      desired_curvature=0.0028 * sign,
    ))

  report = build_lateral_low_speed_report(msgs, source="synthetic", already_sorted=True)
  rendered = render_lateral_low_speed_report(report)
  tier = next(metric for metric in report.tiers if metric.segment == "3-5mps")

  assert "Low-speed lateral report" in rendered
  assert "Primary tiers:" in rendered
  assert "Signal-tagged tiers:" in rendered
  assert "signal-tagged categories:" in rendered
  assert tier.sample_count == 24
  assert tier.abs_error_p95 > 0.0
  assert tier.output_reversals > 0
  assert tier.model_path_gated_percent == pytest.approx(50.0)
  assert tier.raw_processed_curvature_delta_p95 == pytest.approx(0.0012)
  assert tier.model_path_reason_counts["highPathStd"] == 12
  assert tier.model_path_reason_counts["ok"] == 12


def test_low_speed_lateral_report_excludes_lane_changes_from_primary_tiers():
  msgs = []
  for i in range(12):
    msgs.extend(sample_msgs(
      i * 0.1,
      output=0.1,
      v_ego=4.0,
      left_blinker=True,
      lane_change_state="laneChangeStarting",
      desired_accel=0.12,
      actual_accel=0.09,
    ))

  report = build_lateral_low_speed_report(msgs, already_sorted=True)
  signal_tier = next(metric for metric in report.signal_tagged_tiers if metric.segment == "3-5mps")

  assert report.lane_change_excluded_count == 12
  assert all(metric.sample_count == 0 for metric in report.tiers)
  assert signal_tier.sample_count == 12
  assert signal_tier.abs_error_p95 > 0.0
  assert signal_tier.model_path_reason_counts["ok"] == 12
  assert report.signal_tagged_category_counts == {"blinker_only": 0, "lane_change_state_only": 0, "both": 12}
  assert report.signal_tagged_state_counts["laneChangeStarting"] == 12


def test_low_speed_lateral_report_handles_missing_path_state():
  msgs = []
  for i in range(10):
    sign = 1.0 if i % 2 == 0 else -1.0
    msgs.extend(sample_msgs(
      i * 0.1,
      output=0.08 * sign,
      v_ego=6.0,
      steering_angle=4.0 * sign,
      desired_accel=0.10 * sign,
      actual_accel=0.06 * sign,
      desired_curvature=0.002 * sign,
      include_model_path=False,
    ))

  report = build_lateral_low_speed_report(msgs, already_sorted=True)
  tier = next(metric for metric in report.tiers if metric.segment == "5-8mps")

  assert tier.sample_count == 10
  assert tier.model_path_reason_counts["unknown"] == 10
  assert tier.model_path_quality_median == pytest.approx(0.0)
