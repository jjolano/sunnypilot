import numpy as np
import pytest

from cereal import car
from openpilot.selfdrive.locationd.torqued import STEER_BUCKET_BOUNDS, TorqueBuckets, TorqueEstimator


def make_torque_cp():
  CP = car.CarParams.new_message()
  CP.brand = "toyota"
  CP.carFingerprint = "mock-car"
  CP.lateralTuning.init('torque')
  CP.lateralTuning.torque.latAccelFactor = 2.5
  CP.lateralTuning.torque.friction = 0.1
  return CP


def make_test_torque_buckets():
  return TorqueBuckets(
    x_bounds=STEER_BUCKET_BOUNDS,
    min_points=np.ones(len(STEER_BUCKET_BOUNDS)),
    min_points_total=len(STEER_BUCKET_BOUNDS),
    points_per_bucket=100,
    rowsize=3,
  )


def add_synthetic_torque_line(buckets, *, slope: float, offset: float, repeats: int = 3):
  for low, high in STEER_BUCKET_BOUNDS:
    steer = (low + high) / 2.0
    for _ in range(repeats):
      buckets.add_point(steer, slope * steer + offset)


def test_cal_percent():
  est = TorqueEstimator(car.CarParams())
  msg = est.get_msg()
  assert msg.liveTorqueParameters.calPerc == 0

  for (low, high), min_pts in zip(est.filtered_points.buckets.keys(),
                                  est.filtered_points.buckets_min_points.values(), strict=True):
    for _ in range(int(min_pts)):
      est.filtered_points.add_point((low + high) / 2.0, 0.0)

  # enough bucket points, but not enough total points
  msg = est.get_msg()
  assert msg.liveTorqueParameters.calPerc == (len(est.filtered_points) / est.min_points_total * 100 + 100) / 2

  # add enough points to bucket with most capacity
  key = list(est.filtered_points.buckets)[0]
  for _ in range(est.min_points_total - len(est.filtered_points)):
    est.filtered_points.add_point((key[0] + key[1]) / 2.0, 0.0)

  msg = est.get_msg()
  assert msg.liveTorqueParameters.calPerc == 100


def test_torque_estimator_recovers_synthetic_tls_line():
  est = TorqueEstimator(make_torque_cp())
  est.filtered_points = make_test_torque_buckets()
  est.fit_points = 100
  add_synthetic_torque_line(est.filtered_points, slope=2.2, offset=-0.15)

  slope, offset, friction = est.estimate_params()

  assert slope == pytest.approx(2.2)
  assert offset == pytest.approx(-0.15)
  assert friction == pytest.approx(0.0, abs=1e-12)


def test_torque_buckets_ignore_non_finite_synthetic_inputs():
  buckets = make_test_torque_buckets()

  buckets.add_point(float("nan"), 1.0)
  buckets.add_point(0.0, float("inf"))

  assert len(buckets) == 0
