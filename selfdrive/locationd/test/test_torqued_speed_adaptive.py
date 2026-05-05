#!/usr/bin/env python3
import numpy as np
import pytest

from openpilot.selfdrive.locationd.helpers import PointBuckets
from openpilot.sunnypilot.selfdrive.locationd.torqued_ext import SpeedAwareTorqueBuckets


def test_speed_aware_buckets_routing():
  buckets = SpeedAwareTorqueBuckets(
    x_bounds=[(-0.5, -0.3), (-0.3, 0.3), (0.3, 0.5)],
    speed_bp=[0, 10, 20],
    min_points=np.array([10, 50, 10]),
    min_points_total=20,
    points_per_bucket=100,
    rowsize=3
  )
  buckets.add_point(-0.4, 1.0, 5.0)
  buckets.add_point(0.4, 2.0, 15.0)
  buckets.add_point(0.0, 3.0, 25.0)

  assert len(buckets.buckets_for_speed(5.0).buckets) == 3
  assert len(buckets.buckets_for_speed(5.0).get_points()) == 1
  assert len(buckets.buckets_for_speed(15.0).get_points()) == 1
  assert len(buckets.buckets_for_speed(25.0).get_points()) == 1


def test_speed_aware_buckets_valid_percent():
  buckets = SpeedAwareTorqueBuckets(
    x_bounds=[(-0.5, -0.3), (-0.3, 0.3), (0.3, 0.5)],
    speed_bp=[0, 10, 20],
    min_points=np.array([1, 1, 1]),
    min_points_total=2,
    points_per_bucket=100,
    rowsize=3
  )
  for _ in range(10):
    for steer in (-0.4, 0.0, 0.4):
      buckets.add_point(steer, 1.0, 5.0)
      buckets.add_point(steer, 1.0, 15.0)

  assert buckets.is_calculable()
  assert buckets.is_valid()


def test_speed_aware_buckets_sparse_points_are_not_valid():
  buckets = SpeedAwareTorqueBuckets(
    x_bounds=[(-0.5, -0.3), (-0.3, 0.3), (0.3, 0.5)],
    speed_bp=[0, 10, 20],
    min_points=np.array([1, 1, 1]),
    min_points_total=2,
    points_per_bucket=100,
    rowsize=3
  )
  for _ in range(10):
    buckets.add_point(0.0, 1.0, 5.0)

  assert not buckets.is_calculable()
  assert not buckets.is_valid()


def test_speed_aware_get_points():
  buckets = SpeedAwareTorqueBuckets(
    x_bounds=[(-0.5, -0.3), (-0.3, 0.3), (0.3, 0.5)],
    speed_bp=[0, 10, 20],
    min_points=np.array([1, 1, 1]),
    min_points_total=2,
    points_per_bucket=100,
    rowsize=3
  )
  for i in range(5):
    buckets.add_point(float(i) * 0.1, float(i), 5.0)

  pts = buckets.get_points(10)
  assert len(pts) == 5
  assert pts.shape[1] == 3
