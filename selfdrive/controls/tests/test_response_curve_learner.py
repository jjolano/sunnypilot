#!/usr/bin/env python3
import numpy as np
import pytest

from openpilot.sunnypilot.selfdrive.controls.lib.longcontrol_ext import ResponseCurveLearner


def test_bucket_routing():
  learner = ResponseCurveLearner()
  assert learner._bucket_idx(-3.0) == 0
  assert learner._bucket_idx(-0.3) == 3
  assert learner._bucket_idx(0.0) == 4
  assert learner._bucket_idx(0.3) == 4
  assert learner._bucket_idx(1.5) == 6
  assert learner._bucket_idx(5.0) == 8


def test_update_and_lookup():
  learner = ResponseCurveLearner()
  # Bucket 6 is [1.0, 2.0); add points with offset +0.3
  for _ in range(20):
    learner.update(1.5, 1.5 + 0.3)

  assert learner.is_bucket_valid(6)
  offset = learner.lookup_offset(1.5)
  assert abs(offset - 0.3) < 0.05


def test_interpolation():
  learner = ResponseCurveLearner()
  # Bucket 5: [0.5, 1.0) -> offset -0.2
  for _ in range(20):
    learner.update(0.7, 0.7 - 0.2)
  # Bucket 7: [2.0, 4.0) -> offset +0.4
  for _ in range(20):
    learner.update(3.0, 3.0 + 0.4)

  # Interpolate between 0.7 and 3.0 at 1.5
  offset = learner.lookup_offset(1.5)
  expected = np.interp(1.5, [0.7, 3.0], [-0.2, 0.4])
  assert abs(offset - expected) < 0.1


def test_sanity_clamp():
  learner = ResponseCurveLearner()
  for _ in range(20):
    learner.update(1.0, 10.0)  # impossible offset

  offset = learner.lookup_offset(1.0)
  assert offset <= 0.5
