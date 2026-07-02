import json
from pathlib import Path

import pytest
from cereal import messaging
from openpilot.common.realtime import DT_MDL
from openpilot.sunnypilot.custom.lateral.speed_aware_torque import LOW_SPEED_BUCKET_BP, SPEED_BUCKET_BP, _restore_key, format_speed_aware_torque_profile
from openpilot.tools.drive_lab.speed_adaptive_verdict import (
  SpeedAdaptiveRouteProfile,
  analyze_route,
  build_speed_adaptive_verdict_report,
  render_speed_adaptive_verdict_report,
)


BASE_LAT_ACCEL_FACTOR = 2.0
LATERAL_ACCEL = 0.5
WARMUP_FRAMES = 110  # > HISTORY / DT_MDL so livePose processing begins.


def _make_torque_cp(lat_accel_factor: float = BASE_LAT_ACCEL_FACTOR):
  msg = messaging.new_message('carParams')
  cp = msg.carParams
  cp.brand = "toyota"
  cp.carFingerprint = "TOYOTA_CAMRY"
  cp.lateralTuning.init('torque')
  cp.lateralTuning.torque.latAccelFactor = float(lat_accel_factor)
  cp.lateralTuning.torque.friction = 0.2
  msg.logMonoTime = 0
  return msg


def _make_frame(t: float, v_ego: float, steer: float, lateral_accel: float = LATERAL_ACCEL):
  log_mono_time = int(t * 1e9)

  car_control = messaging.new_message('carControl')
  car_control.carControl.latActive = True
  car_control.logMonoTime = log_mono_time

  car_output = messaging.new_message('carOutput')
  car_output.carOutput.actuatorsOutput.torque = float(-steer)
  car_output.logMonoTime = log_mono_time

  car_state = messaging.new_message('carState')
  cs = car_state.carState
  cs.vEgo = float(v_ego)
  cs.steeringPressed = False
  cs.steeringRateDeg = 0.0
  cs.steeringTorqueEps = 0.0
  car_state.logMonoTime = log_mono_time

  live_pose = messaging.new_message('livePose')
  lp = live_pose.livePose
  lp.orientationNED = {'x': 0.0, 'valid': True}
  lp.angularVelocityDevice = {'z': float(lateral_accel / v_ego), 'valid': True}
  lp.inputsOK, lp.sensorsOK, lp.posenetOK = True, True, True
  lp.timestamp = log_mono_time
  live_pose.logMonoTime = log_mono_time

  return [car_control, car_output, car_state, live_pose]


def _make_route(source: str, ratios: list[float], samples_per_bin: int = 300):
  msgs = [_make_torque_cp()]
  total_samples = len(SPEED_BUCKET_BP) * samples_per_bin
  for i in range(WARMUP_FRAMES + total_samples):
    t = i * DT_MDL
    speed_idx = i % len(SPEED_BUCKET_BP)
    v_ego = SPEED_BUCKET_BP[speed_idx] + 1.0
    ratio = ratios[speed_idx]
    # Sweep steer across the torque estimator x-bounds so each speed bucket has
    # a spread of (steer, lateral_accel) points and SVD can fit a slope.
    steer = -0.4 + 0.8 * ((i // len(SPEED_BUCKET_BP)) % 100) / 99.0
    lateral_accel = ratio * BASE_LAT_ACCEL_FACTOR * steer
    msgs.extend(_make_frame(t, v_ego, steer, lateral_accel))
  return msgs


def _make_low_point_route(source: str):
  return _make_route(source, ratios=[1.0] * len(SPEED_BUCKET_BP), samples_per_bin=10)


def test_speed_adaptive_verdict_promote_with_three_routes():
  # Speed-dependent ratios: slope grows as speed drops.
  ratios = [1.10, 1.05, 1.00, 0.95, 0.90]
  routes = [_make_route(f"route_{i}", ratios) for i in range(3)]
  profiles = [analyze_route(msgs, source=f"route_{i}", already_sorted=True) for i, msgs in enumerate(routes)]
  report = build_speed_adaptive_verdict_report(profiles)
  rendered = render_speed_adaptive_verdict_report(report)

  assert report.verdict == "promote"
  assert report.confident_route_count == 3
  assert report.cross_route_ratio_spread is not None
  assert report.cross_route_ratio_spread < 0.05
  assert "promote" in rendered

  for profile in profiles:
    assert profile.fitted
    assert len(profile.anchors) == len(SPEED_BUCKET_BP)
    assert all(c >= 0.6 for c in profile.confidence)
    # Ratios should be ordered high-to-low across the speed range.
    assert profile.ratios[0] > profile.ratios[-1]
    assert profile.ratio_active_percent > 0.0
    assert profile.base_lat_accel_factor == pytest.approx(BASE_LAT_ACCEL_FACTOR)
    assert profile.global_slope is not None


def test_speed_adaptive_verdict_insufficient_evidence_for_low_point_route():
  msgs = _make_low_point_route("low_point_route")
  profile = analyze_route(msgs, source="low_point_route", already_sorted=True)
  report = build_speed_adaptive_verdict_report([profile])

  assert report.verdict == "insufficient_evidence"
  assert report.confident_route_count == 0
  assert all(p == 0 for p in profile.points)
  assert profile.global_slope is None


def test_speed_adaptive_profile_json_bypasses_fitting(tmp_path: Path):
  ratios = [1.08, 1.04, 1.00, 0.96, 0.92]
  msgs = _make_route("json_route", ratios, samples_per_bin=300)

  fitted_profile = analyze_route(msgs, source="json_route", already_sorted=True)
  assert fitted_profile.fitted

  profile_path = tmp_path / "speed_aware_profile.json"
  # Write a raw speed-aware profile matching the route's CP restore key.
  cp = _make_torque_cp(BASE_LAT_ACCEL_FACTOR).carParams
  raw_profile = {
    'version': 1,
    'restoreKey': _restore_key(cp),
    'anchors': list(fitted_profile.anchors),
    'ratios': list(fitted_profile.ratios),
    'confidence': [1.0] * len(fitted_profile.anchors),
    'points': [1000] * len(fitted_profile.anchors),
    'globalLatAccelFactor': BASE_LAT_ACCEL_FACTOR,
    'globalFriction': 0.2,
    'lowSpeed': {
      'anchors': [5.0, 10.0],
      'ratios': [1.4, 1.1],
      'slopes': [2.8, 2.2],
      'confidence': [1.0, 1.0],
      'points': [600, 600],
    },
  }
  profile_path.write_text(format_speed_aware_torque_profile(raw_profile))

  applied_profile = analyze_route(msgs, source="json_route", profile_json=str(profile_path), already_sorted=True)
  assert not applied_profile.fitted
  assert applied_profile.profile_source == str(profile_path)
  assert applied_profile.ratios == pytest.approx(fitted_profile.ratios, abs=0.02)
  assert applied_profile.ratio_active_percent > 0.0
  assert applied_profile.low_speed == raw_profile['lowSpeed']


def test_speed_adaptive_verdict_rejects_verdict_report_as_profile(tmp_path: Path):
  msgs = _make_route("any", [1.0] * len(SPEED_BUCKET_BP), samples_per_bin=300)
  profile = analyze_route(msgs, source="any", already_sorted=True)
  report = build_speed_adaptive_verdict_report([profile])

  bad_path = tmp_path / "report.json"
  bad_path.write_text(json.dumps(report.to_dict(), indent=2))

  with pytest.raises(ValueError):
    analyze_route(msgs, source="any", profile_json=str(bad_path), already_sorted=True)


def _route_profile(source: str, confidence: list[float], ratios: list[float] | None = None) -> SpeedAdaptiveRouteProfile:
  return SpeedAdaptiveRouteProfile(
    source=source,
    anchors=list(SPEED_BUCKET_BP),
    ratios=list(ratios or [1.0] * len(SPEED_BUCKET_BP)),
    confidence=list(confidence),
    points=[500] * len(SPEED_BUCKET_BP),
    global_slope=2.0,
    bin_slopes=[2.0] * len(SPEED_BUCKET_BP),
    engaged_frames=100,
    ratio_active_frames=10,
    ratio_active_percent=10.0,
    base_lat_accel_factor=BASE_LAT_ACCEL_FACTOR,
    lat_accel_deltas=[0.0] * len(SPEED_BUCKET_BP),
    fitted=True,
    profile_source="fit",
  )


def test_speed_adaptive_verdict_requires_shared_confident_anchors():
  profiles = [
    _route_profile("r1", [1.0, 0.0, 0.0, 0.0, 0.0]),
    _route_profile("r2", [0.0, 1.0, 0.0, 0.0, 0.0]),
    _route_profile("r3", [0.0, 0.0, 1.0, 0.0, 0.0]),
  ]

  report = build_speed_adaptive_verdict_report(profiles)

  assert report.verdict == "insufficient_evidence"
  assert report.cross_route_ratio_spread is None


def test_speed_adaptive_verdict_does_not_promote_partial_anchor_overlap():
  profiles = [
    _route_profile("r1", [1.0, 1.0, 0.0, 0.0, 0.0], ratios=[1.02, 1.01, 1.0, 1.0, 1.0]),
    _route_profile("r2", [1.0, 1.0, 0.0, 0.0, 0.0], ratios=[1.03, 1.02, 1.0, 1.0, 1.0]),
    _route_profile("r3", [1.0, 0.0, 1.0, 0.0, 0.0], ratios=[1.01, 1.0, 0.99, 1.0, 1.0]),
  ]

  report = build_speed_adaptive_verdict_report(profiles)

  assert report.verdict == "insufficient_evidence"


def test_speed_adaptive_verdict_includes_low_speed_when_present():
  profile = SpeedAdaptiveRouteProfile(
    source="low_speed_route",
    anchors=list(SPEED_BUCKET_BP),
    ratios=[1.0] * len(SPEED_BUCKET_BP),
    confidence=[1.0] * len(SPEED_BUCKET_BP),
    points=[500] * len(SPEED_BUCKET_BP),
    global_slope=2.0,
    bin_slopes=[2.0] * len(SPEED_BUCKET_BP),
    engaged_frames=100,
    ratio_active_frames=10,
    ratio_active_percent=10.0,
    base_lat_accel_factor=BASE_LAT_ACCEL_FACTOR,
    lat_accel_deltas=[0.0] * len(SPEED_BUCKET_BP),
    fitted=True,
    profile_source="fit",
    low_speed={
      "anchors": list(LOW_SPEED_BUCKET_BP),
      "ratios": [1.05, 1.02],
      "slopes": [2.1, 2.04],
      "confidence": [0.8, 0.9],
      "points": [300, 400],
    },
  )
  report = build_speed_adaptive_verdict_report([profile])
  rendered = render_speed_adaptive_verdict_report(report)
  assert "lowSpeed section" in rendered
  assert report.routes[0].low_speed is not None
  assert report.routes[0].low_speed["anchors"] == list(LOW_SPEED_BUCKET_BP)
