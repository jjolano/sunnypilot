import math

import openpilot.sunnypilot.custom.lateral.block_jackknife as block_jackknife

import pytest

from openpilot.sunnypilot.custom.lateral.block_jackknife import (
  EvidenceBlockClock,
  fit_block_slope,
  fit_ratio_jackknife,
)


def test_evidence_clock_boundaries_completed_ids_and_invalid_time():
  clock = EvidenceBlockClock()

  assert clock.advance(100.0) == 0
  assert clock.boundary and not clock.in_guard
  assert clock.completed_through == -1
  assert list(clock.completed_block_ids) == []

  assert clock.advance(160.0) is None  # exact 60-second evidence boundary
  assert clock.boundary and clock.in_guard
  assert clock.completed_through == 0
  assert list(clock.completed_block_ids) == [0]

  assert clock.advance(160.0) is None
  assert not clock.boundary
  assert clock.advance(165.0) == 1  # 65 seconds starts the next block
  assert clock.boundary and not clock.in_guard
  assert clock.is_completed(0)
  assert not clock.is_completed(1)

  assert clock.advance(225.0) is None
  assert clock.in_guard
  assert list(clock.completed_block_ids) == [0, 1]

  assert clock.advance(100.0 + 3 * 65.0 + 1.0) == 3
  assert list(clock.completed_block_ids) == [0, 1, 2]
  before = (clock.block_id, clock.in_guard, clock.completed_through)
  for timestamp in (99.0, math.nan, math.inf, -math.inf, "not-a-time"):
    assert clock.advance(timestamp) is None
    assert (clock.block_id, clock.in_guard, clock.completed_through) == before
    assert clock.boundary and clock.discontinuity


def test_centered_block_slope_removes_each_block_intercept():
  points = [
    (1.0, 12.0, 0), (2.0, 14.0, 0), (3.0, 16.0, 0),
    (10.0, -80.0, 1), (12.0, -76.0, 1), (14.0, -72.0, 1),
  ]

  fit = fit_block_slope(points)

  assert fit is not None
  assert fit.slope == pytest.approx(2.0)
  assert fit.block_slopes == pytest.approx({0: 2.0, 1: 2.0})
  assert fit.loo_slopes == pytest.approx({0: 2.0, 1: 2.0})
  assert fit.jackknife_se == pytest.approx(0.0)
  assert fit.rel_se == pytest.approx(0.0)


def test_jackknife_se_is_nonzero_for_block_correlated_slopes():
  points = [
    (x, slope * x + block, block)
    for block, slope in enumerate((1.0, 2.0, 3.0))
    for x in (-1.0, 0.0, 1.0)
  ]

  fit = fit_block_slope(points)

  assert fit is not None
  assert fit.slope == pytest.approx(2.0)
  assert fit.loo_slopes == pytest.approx({0: 2.5, 1: 2.0, 2: 1.5})
  assert fit.jackknife_se == pytest.approx(math.sqrt(1 / 3))
  assert fit.rel_se == pytest.approx(math.sqrt(1 / 12))


def test_degenerate_blocks_are_excluded_and_all_degenerate_fails():
  fit = fit_block_slope([
    (1.0, 2.0, 0), (1.0, 3.0, 0),
    (1.0, 4.0, 1), (2.0, 6.0, 1), (3.0, 8.0, 1),
  ])
  assert fit is not None
  assert set(fit.block_slopes) == {1}
  assert math.isinf(fit.rel_se)  # one informative block cannot jackknife

  assert fit_block_slope([(1.0, 2.0, 0), (1.0, 3.0, 0)]) is None


def test_ratio_jackknife_uses_union_and_removes_shared_blocks_together():
  def rows(slopes, intercept=0.0):
    return [
      (x, slope * x + intercept + block, block)
      for block, slope in enumerate(slopes)
      for x in (1.0, 2.0, 3.0)
    ]

  left = rows((1.0, 2.0, 3.0), intercept=10.0)
  right = rows((0.8, 1.0, 1.2), intercept=-7.0)
  result = fit_ratio_jackknife(left, right)

  assert result is not None
  assert result['ratio'] == pytest.approx(2.0)
  assert set(result['ratio_loo']) == {0, 1, 2}
  assert result['ratio_loo'][1] == pytest.approx(2.0)  # shared block removed from both fits
  assert result['ratio_loo'][0] != pytest.approx(2.0)


def test_ratio_jackknife_handles_asymmetric_left_only_and_right_only_blocks():
  left = [(x, slope * x, block) for block, slope in ((0, 1.0), (1, 1.2)) for x in (1.0, 2.0, 3.0)]
  right = [(x, slope * x, block) for block, slope in ((1, 0.8), (2, 1.0)) for x in (1.0, 2.0, 3.0)]

  result = fit_ratio_jackknife(left, right)

  assert result is not None
  assert set(result['ratio_loo']) == {0, 1, 2}
  assert all(math.isfinite(value) for value in result['ratio_loo'].values())


def test_ratio_jackknife_builds_each_side_stats_once(monkeypatch):
  calls = 0
  original = block_jackknife._block_stats

  def counted(rows):
    nonlocal calls
    calls += 1
    return original(rows)

  monkeypatch.setattr(block_jackknife, '_block_stats', counted)
  points = [(x, 1.0 * x + block, block) for block in range(12) for x in (1.0, 2.0, 3.0)]

  result = fit_ratio_jackknife(points, points)

  assert result is not None
  assert calls == 2


def test_ratio_jackknife_cap_sized_smoke():
  left = [(x, 1.1 * x + block, block) for block in range(12) for x in range(1, 1 + 4000 // 12)]
  right = [(x, 1.0 * x - block, block) for block in range(12) for x in range(1, 1 + 4000 // 12)]

  result = fit_ratio_jackknife(left, right)

  assert result is not None
  assert math.isfinite(result['ratio_rel_se'])
