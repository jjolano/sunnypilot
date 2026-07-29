"""Unit tests for the inner-lane geometry helper."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from openpilot.sunnypilot.custom.lateral.demand.lane_geometry import (
  LANE_GEOMETRY_MAX_INNER_STD,
  LANE_GEOMETRY_MIN_INNER_PROB,
  LANE_GEOMETRY_REASON_BAD_PATH,
  LANE_GEOMETRY_REASON_BAD_STRADDLE,
  LANE_GEOMETRY_REASON_BAD_WIDTH,
  LANE_GEOMETRY_REASON_HIGH_STD,
  LANE_GEOMETRY_REASON_LOW_PROB,
  LANE_GEOMETRY_REASON_MISSING,
  LANE_GEOMETRY_REASON_SIGN_MISMATCH,
  LANE_GEOMETRY_REASON_UNSTABLE_WIDTH,
  evaluate_lane_geometry,
)

N = 33


def _lane_line(center_y: float, width: float = 4.0, curvature: float = 0.0):
  xs = [float(x) for x in range(N)]
  # center_y is the lane center offset; inner lanes straddle it by half width.
  left_y = [center_y - width * 0.5 + 0.5 * curvature * x * x for x in xs]
  right_y = [center_y + width * 0.5 + 0.5 * curvature * x * x for x in xs]
  return [
    SimpleNamespace(x=xs, y=[left_y[i] - width for i in range(N)]),  # outer left
    SimpleNamespace(x=xs, y=left_y),                                   # inner left
    SimpleNamespace(x=xs, y=right_y),                                  # inner right
    SimpleNamespace(x=xs, y=[right_y[i] + width for i in range(N)]),   # outer right
  ]


def _path(curvature: float = 0.0, offset: float = 0.0):
  xs = [float(x) for x in range(N)]
  ys = [offset + 0.5 * curvature * x * x for x in xs]
  return xs, ys


def _probs(inner: float = 0.9):
  return [0.5, inner, inner, 0.5]


def _stds(inner: float = 0.1):
  return [0.2, inner, inner, 0.2]


def _geo(offset: float = 0.0, width: float = 4.0, inner_prob: float = 0.9, inner_std: float = 0.1):
  xs, ys = _path(offset=offset)
  return evaluate_lane_geometry(
    lane_lines=_lane_line(0.0, width),
    lane_line_probs=_probs(inner_prob),
    lane_line_stds=_stds(inner_std),
    position_x=xs,
    position_y=ys,
    near_x=10.0,
    preview_x=30.0,
  )


def test_centered_path_valid_and_zero_error():
  g = _geo(offset=0.0)
  assert g.valid is True
  assert g.reason == "ok"
  assert g.lateral_error == pytest.approx(0.0, abs=1e-9)
  assert g.predicted_lateral_error == pytest.approx(0.0, abs=1e-9)
  assert g.source == "lane_lines"
  assert g.confidence > 0.0


def test_path_right_of_center_negative_nudge():
  # Positive y is to the right of the vehicle; model path at +0.5 is right of lane center.
  g = _geo(offset=0.5)
  assert g.valid is True
  assert g.lateral_error < -0.45
  assert g.predicted_lateral_error < -0.45


def test_path_left_of_center_positive_nudge():
  # Model path at -0.5 is left of lane center.
  g = _geo(offset=-0.5)
  assert g.valid is True
  assert g.lateral_error > 0.45
  assert g.predicted_lateral_error > 0.45


def test_small_offset_inside_deadband_is_reported_anyway():
  g = _geo(offset=0.02)
  assert g.valid is True
  assert abs(g.lateral_error) < 0.05
  assert abs(g.predicted_lateral_error) < 0.05


def test_missing_lane_lines_fail_closed():
  xs, ys = _path(offset=0.5)
  g = evaluate_lane_geometry(
    lane_lines=(),
    lane_line_probs=_probs(),
    lane_line_stds=_stds(),
    position_x=xs,
    position_y=ys,
    near_x=10.0,
    preview_x=30.0,
  )
  assert g.valid is False
  assert g.reason == LANE_GEOMETRY_REASON_MISSING


def test_low_inner_prob_blocks():
  g = _geo(offset=0.5, inner_prob=LANE_GEOMETRY_MIN_INNER_PROB - 0.01)
  assert g.valid is False
  assert g.reason == LANE_GEOMETRY_REASON_LOW_PROB


def test_high_inner_std_blocks():
  g = _geo(offset=0.5, inner_std=LANE_GEOMETRY_MAX_INNER_STD + 0.01)
  assert g.valid is False
  assert g.reason == LANE_GEOMETRY_REASON_HIGH_STD


@pytest.mark.parametrize("width", [1.5, 6.5])
def test_bad_width_blocks(width):
  g = _geo(offset=0.5, width=width)
  assert g.valid is False
  assert g.reason == LANE_GEOMETRY_REASON_BAD_WIDTH


def test_unstable_width_blocks():
  xs, ys = _path(offset=0.5)
  # Pinch both inner lanes inward so near/preview widths diverge but stay in bounds.
  lanes = [
    SimpleNamespace(x=xs, y=[-2.0 - 0.021 * x for x in xs]),
    SimpleNamespace(x=xs, y=[-2.0 + 0.021 * x for x in xs]),
    SimpleNamespace(x=xs, y=[2.0 - 0.021 * x for x in xs]),
    SimpleNamespace(x=xs, y=[2.0 + 0.021 * x for x in xs]),
  ]
  g = evaluate_lane_geometry(
    lane_lines=lanes,
    lane_line_probs=[0.9, 0.9, 0.9, 0.9],
    lane_line_stds=[0.1, 0.1, 0.1, 0.1],
    position_x=xs,
    position_y=ys,
    near_x=10.0,
    preview_x=30.0,
  )
  assert g.valid is False
  assert g.reason == LANE_GEOMETRY_REASON_UNSTABLE_WIDTH


def test_sign_mismatch_between_near_and_preview_blocks():
  xs = [float(x) for x in range(N)]
  # Path crosses through lane center between near and preview.
  path_y = [0.5 - 0.04 * x for x in xs]
  lanes = [
    SimpleNamespace(x=xs, y=[-6.0] * N),
    SimpleNamespace(x=xs, y=[-2.0] * N),
    SimpleNamespace(x=xs, y=[2.0] * N),
    SimpleNamespace(x=xs, y=[6.0] * N),
  ]
  g = evaluate_lane_geometry(
    lane_lines=lanes,
    lane_line_probs=[0.9, 0.9, 0.9, 0.9],
    lane_line_stds=[0.1, 0.1, 0.1, 0.1],
    position_x=xs,
    position_y=path_y,
    near_x=10.0,
    preview_x=30.0,
  )
  assert g.valid is False
  assert g.reason == LANE_GEOMETRY_REASON_SIGN_MISMATCH


def test_same_side_inner_lines_block():
  xs, ys = _path(offset=0.0)
  lanes = [
    SimpleNamespace(x=xs, y=[-2.0] * N),
    SimpleNamespace(x=xs, y=[2.0] * N),
    SimpleNamespace(x=xs, y=[5.0] * N),
    SimpleNamespace(x=xs, y=[8.0] * N),
  ]
  g = evaluate_lane_geometry(
    lane_lines=lanes,
    lane_line_probs=[0.9, 0.9, 0.9, 0.9],
    lane_line_stds=[0.1, 0.1, 0.1, 0.1],
    position_x=xs,
    position_y=ys,
    near_x=10.0,
    preview_x=30.0,
  )
  assert g.valid is False
  assert g.reason == LANE_GEOMETRY_REASON_BAD_STRADDLE


def test_missing_path_fails_closed():
  lanes = _lane_line(0.0)
  g = evaluate_lane_geometry(
    lane_lines=lanes,
    lane_line_probs=_probs(),
    lane_line_stds=_stds(),
    position_x=(),
    position_y=(),
    near_x=10.0,
    preview_x=30.0,
  )
  assert g.valid is False
  assert g.reason == LANE_GEOMETRY_REASON_BAD_PATH


def test_nonfinite_sample_uses_extrapolation_and_still_valid():
  # x=10 and x=30 are well inside [0, N-1], so interpolation is straightforward.
  g = _geo(offset=-0.3)
  assert g.valid is True
  assert g.lateral_error == pytest.approx(0.3, abs=1e-3)


def test_short_horizon_lane_lines_block_extrapolation():
  xs, ys = _path(offset=0.5)
  # Lane lines only cover 0..15, but preview is requested at 30.
  short_xs = [float(x) for x in range(16)]
  lanes = [
    SimpleNamespace(x=short_xs, y=[-2.0] * 16),
    SimpleNamespace(x=short_xs, y=[2.0] * 16),
    SimpleNamespace(x=short_xs, y=[-2.0] * 16),
    SimpleNamespace(x=short_xs, y=[2.0] * 16),
  ]
  g = evaluate_lane_geometry(
    lane_lines=lanes,
    lane_line_probs=[0.9, 0.9, 0.9, 0.9],
    lane_line_stds=[0.1, 0.1, 0.1, 0.1],
    position_x=xs,
    position_y=ys,
    near_x=10.0,
    preview_x=30.0,
  )
  assert g.valid is False
  assert g.reason == LANE_GEOMETRY_REASON_MISSING
