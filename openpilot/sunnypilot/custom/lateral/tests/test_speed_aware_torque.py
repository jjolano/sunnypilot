import json
import pytest

from types import SimpleNamespace

from openpilot.sunnypilot.custom.lateral.speed_aware_torque import (
  MIN_BIN_POINTS,
  blend_speed_aware_torque_profile,
  format_speed_aware_torque_profile,
  SpeedAwareTorqueBuckets, SpeedAwareTorqueRuntime, fit_speed_aware_torque_profile,
  parse_speed_aware_torque_profile, fit_low_speed_section, LOW_SPEED_BUCKET_BP,
  SPEED_BUCKET_BP,
)

X_BOUNDS = [(-0.5, -0.3), (-0.3, -0.2), (-0.2, -0.1), (-0.1, 0), (0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.5)]


def cp():
  torque = SimpleNamespace(latAccelFactor=2.0, friction=0.2)
  return SimpleNamespace(carFingerprint="test", lateralTuning=SimpleNamespace(which=lambda: 'torque', torque=torque))


def _profile(ratios, confidence, points, *, low_speed=None, global_lat_accel_factor=2.0, global_friction=0.2):
  car = cp()
  torque = car.lateralTuning.torque
  profile = {
    'version': 1,
    'restoreKey': {
      'carFingerprint': car.carFingerprint,
      'lateralTuning': car.lateralTuning.which(),
      'latAccelFactor': float(torque.latAccelFactor),
      'friction': float(torque.friction),
    },
    'anchors': list(SPEED_BUCKET_BP),
    'ratios': ratios,
    'confidence': confidence,
    'points': points,
    'globalLatAccelFactor': global_lat_accel_factor,
    'globalFriction': global_friction,
  }
  if low_speed is not None:
    profile['lowSpeed'] = low_speed
  return profile


def test_fit_ratio_and_clamp():
  buckets = SpeedAwareTorqueBuckets(X_BOUNDS, [15, 20, 30], [1] * len(X_BOUNDS), 1, 5000)
  for i in range(2500):
    steer = -0.4 + 0.0003 * i
    buckets.add_point(steer, 2.0 * steer, 15.1)
    buckets.add_point(steer, 2.2 * steer, 25.1)
  profile = fit_speed_aware_torque_profile(cp(), buckets)
  assert profile is not None
  assert profile['ratios'][0] == pytest.approx(2.0 / 2.1, abs=0.02)
  assert profile['ratios'][1] == pytest.approx(2.2 / 2.1, abs=0.02)


def test_blend_profile_caps_single_sided_confidence_and_round_trips():
  old = _profile(
    ratios=[0.80, 0.90, 0.95, 1.05, 1.00],
    confidence=[0.4, 0.3, 0.0, 0.7, 0.0],
    points=[700, 300, 0, 100, 0],
    low_speed={'source': 'old'},
  )
  new = _profile(
    ratios=[0.85, 1.10, 1.20, 0.95, 1.00],
    confidence=[0.0, 0.2, 0.9, 0.4, 0.0],
    points=[500, 100, 900, 100, 0],
    low_speed={'source': 'new'},
  )
  blended = blend_speed_aware_torque_profile(old, new)

  assert blended['lowSpeed'] == new['lowSpeed']
  assert blended['points'][0] == 4 * MIN_BIN_POINTS
  assert blended['ratios'][0] == pytest.approx(old['ratios'][0])
  assert blended['points'][1] == 400
  assert blended['ratios'][1] == pytest.approx((300 * old['ratios'][1] + 100 * new['ratios'][1]) / 400)
  assert blended['confidence'][1] == pytest.approx(1.0)
  assert blended['points'][2] == 4 * MIN_BIN_POINTS
  assert blended['ratios'][2] == pytest.approx(new['ratios'][2])
  assert blended['confidence'][2] == pytest.approx(1.0)
  assert blended['points'][3] == 200
  assert blended['ratios'][3] == pytest.approx((100 * old['ratios'][3] + 100 * new['ratios'][3]) / 200)
  assert blended['confidence'][3] == pytest.approx(0.5)

  parsed = parse_speed_aware_torque_profile(cp(), json.loads(format_speed_aware_torque_profile(blended)))
  assert parsed is not None
  assert parsed['anchors'] == list(SPEED_BUCKET_BP)
  assert parsed['points'] == blended['points']
  assert parsed['ratios'] == pytest.approx(blended['ratios'])
  assert parsed['confidence'] == pytest.approx(blended['confidence'])


def test_blend_profile_anchor_mismatch_overwrites():
  old = _profile(
    ratios=[0.80, 0.90, 0.95, 1.05, 1.00],
    confidence=[1.0, 1.0, 1.0, 1.0, 1.0],
    points=[500, 500, 500, 500, 500],
  )
  new = _profile(
    ratios=[0.85, 1.10, 1.20, 0.95, 1.00],
    confidence=[0.2, 0.2, 0.2, 0.2, 0.2],
    points=[100, 100, 100, 100, 100],
  )
  new['anchors'] = [bp + 0.5 for bp in SPEED_BUCKET_BP]

  assert blend_speed_aware_torque_profile(old, new) == new


def test_parse_rejects_wrong_identity():
  p = {'version': 1, 'restoreKey': {'carFingerprint': 'test', 'lateralTuning': 'torque', 'latAccelFactor': 2.0, 'friction': 0.2},
       'anchors': [15.0], 'ratios': [1.0], 'confidence': [1.0], 'points': [1], 'globalLatAccelFactor': 2.0, 'globalFriction': 0.2}
  p['restoreKey'] = {'carFingerprint': 'bad'}
  assert parse_speed_aware_torque_profile(cp(), p) is None


def test_ignore_low_speed_points():
  buckets = SpeedAwareTorqueBuckets(X_BOUNDS, [15, 20, 30], [1] * len(X_BOUNDS), 1, 1)
  buckets.add_point(0.1, 0.2, 10.0)
  assert all(len(b.get_points()) == 0 for _, b in buckets.bucket_items())


def test_runtime_no_low_speed_extrapolation():
  runtime = SpeedAwareTorqueRuntime({'anchors': [20.0, 30.0], 'ratios': [1.1, 1.2], 'confidence': [1.0, 1.0], 'points': [1, 1]})
  assert runtime.ratio(10.0) == 1.0
  assert runtime.ratio(25.0) == pytest.approx(1.15)


def test_runtime_uses_confident_last_bucket_above_last_anchor():
  runtime = SpeedAwareTorqueRuntime({'anchors': [20.0, 40.0], 'ratios': [1.1, 1.2], 'confidence': [1.0, 1.0], 'points': [1, 1]})
  assert runtime.ratio(45.0) == pytest.approx(1.2)


def test_degenerate_fit_rejected():
  buckets = SpeedAwareTorqueBuckets(X_BOUNDS, [15, 20, 30], [1] * len(X_BOUNDS), 1, 5000)
  for _ in range(600):
    buckets.add_point(0.1, 0.2, 20.1)
  assert fit_speed_aware_torque_profile(cp(), buckets) is None


def test_format_rejects_non_finite_json():
  with pytest.raises(ValueError):
    format_speed_aware_torque_profile({'bad': float('nan')})


def test_low_speed_section_is_ignored_by_parse_and_runtime():
  buckets = SpeedAwareTorqueBuckets(X_BOUNDS, [15, 20, 30], [1] * len(X_BOUNDS), 1, 5000)
  low_buckets = SpeedAwareTorqueBuckets(X_BOUNDS, LOW_SPEED_BUCKET_BP, [1] * len(X_BOUNDS), 1, 5000)
  for i in range(2500):
    steer = -0.4 + 0.0003 * i
    buckets.add_point(steer, 2.0 * steer, 15.1)
    low_buckets.add_point(steer, 2.0 * steer, 7.0)
  profile = fit_speed_aware_torque_profile(cp(), buckets, low_speed_buckets=low_buckets)
  assert profile is not None
  assert 'lowSpeed' in profile
  parsed = parse_speed_aware_torque_profile(cp(), profile)
  assert parsed is not None
  assert 'lowSpeed' not in parsed
  runtime = SpeedAwareTorqueRuntime(profile=parsed)
  assert runtime.ratio(7.0) == 1.0
  assert runtime.ratio(12.0) == 1.0


def test_low_speed_section_reports_evidence_fields():
  buckets = SpeedAwareTorqueBuckets(X_BOUNDS, [15, 20, 30], [1] * len(X_BOUNDS), 1, 5000)
  low_buckets = SpeedAwareTorqueBuckets(X_BOUNDS, LOW_SPEED_BUCKET_BP, [1] * len(X_BOUNDS), 1, 5000)
  for i in range(2500):
    steer = -0.4 + 0.0003 * i
    buckets.add_point(steer, 2.0 * steer, 15.1)
    low_buckets.add_point(steer, 3.0 * steer, 7.0)
  profile = fit_speed_aware_torque_profile(cp(), buckets, low_speed_buckets=low_buckets)
  assert profile is not None
  low = profile['lowSpeed']
  assert low['anchors'] == list(LOW_SPEED_BUCKET_BP)
  assert 'ratios' in low
  assert 'slopes' in low
  assert 'confidence' in low
  assert 'points' in low
  assert len(low['slopes']) == len(LOW_SPEED_BUCKET_BP)
  assert max(low['ratios']) > 1.25


def test_fit_low_speed_section_empty_returns_none():
  low_buckets = SpeedAwareTorqueBuckets(X_BOUNDS, LOW_SPEED_BUCKET_BP, [1] * len(X_BOUNDS), 1, 5000)
  assert fit_low_speed_section(cp(), low_buckets) is None
