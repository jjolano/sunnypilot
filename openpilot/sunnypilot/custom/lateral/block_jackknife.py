"""Shared block-based evidence clock and slope uncertainty helpers."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Iterable, Sequence


EVIDENCE_BLOCK_S = 60.0
EVIDENCE_GUARD_S = 5.0
MIN_EVIDENCE_BLOCKS = 12
MAX_BLOCK_REL_SE = 1 / 3


class EvidenceBlockClock:
  """Session-local, deterministic non-overlapping evidence block clock.

  The first timestamp starts block zero.  Evidence occupies 60 seconds, followed
  by a five-second guard.  Timestamps in the guard return ``None``; block IDs are
  deliberately local to this clock instance rather than derived from wall time.
  """

  def __init__(self) -> None:
    self._origin: float | None = None
    self._last_timestamp: float | None = None
    self._block_id: int | None = None
    self._in_guard = False
    self._completed_through = -1
    self.boundary = False
    self.discontinuity = False

  @property
  def block_id(self) -> int | None:
    return self._block_id

  @property
  def in_guard(self) -> bool:
    return self._in_guard

  @property
  def completed_through(self) -> int:
    return self._completed_through

  @property
  def completed_block_ids(self) -> range:
    return range(self._completed_through + 1)

  def is_completed(self, block_id: int) -> bool:
    try:
      block_id = int(block_id)
    except (TypeError, ValueError):
      return False
    return 0 <= block_id <= self._completed_through

  def advance(self, timestamp: float) -> int | None:
    """Advance to ``timestamp`` and return its evidence block, or ``None`` in guard.

    ``boundary`` is set for the current call when the evidence/guard state
    changes.  Non-finite or backwards timestamps return ``None`` and signal a
    discontinuity without moving the clock.
    """
    self.discontinuity = False
    try:
      timestamp = float(timestamp)
    except (TypeError, ValueError):
      self.discontinuity = True
      self.boundary = True
      return None
    if not isfinite(timestamp):
      self.discontinuity = True
      self.boundary = True
      return None

    if self._origin is None:
      self._origin = timestamp
      self._last_timestamp = timestamp
      self._block_id = 0
      self._in_guard = False
      self.boundary = True
      return self._block_id

    if self._last_timestamp is not None and timestamp < self._last_timestamp:
      self.discontinuity = True
      self.boundary = True
      return None
    self._last_timestamp = timestamp
    elapsed = timestamp - self._origin
    cycle_s = EVIDENCE_BLOCK_S + EVIDENCE_GUARD_S
    cycle = int(elapsed // cycle_s)
    offset = elapsed - cycle * cycle_s
    in_guard = offset >= EVIDENCE_BLOCK_S
    block_id = None if in_guard else cycle
    completed_through = cycle if in_guard else cycle - 1

    self.boundary = block_id != self._block_id or in_guard != self._in_guard
    self._block_id = block_id
    self._in_guard = in_guard
    self._completed_through = max(self._completed_through, completed_through)
    return block_id


@dataclass(frozen=True)
class BlockSlopeFit:
  """Centered block-intercept slope and delete-one-block jackknife result."""

  slope: float
  jackknife_se: float
  rel_se: float
  block_slopes: dict[int, float]
  loo_slopes: dict[int, float]


def _coerce_points(points: Iterable[Sequence[float]]) -> list[tuple[float, float, int]]:
  rows: list[tuple[float, float, int]] = []
  for row in points:
    try:
      x, y, block_id = row[0], row[1], row[2]
      x = float(x)
      y = float(y)
      block_float = float(block_id)
      block_int = int(block_float)
    except (IndexError, TypeError, ValueError, OverflowError):
      continue
    if not (isfinite(x) and isfinite(y) and isfinite(block_float)) or block_float != block_int:
      continue
    rows.append((x, y, block_int))
  return rows


def _block_stats(rows: Iterable[tuple[float, float, int]]) -> dict[int, tuple[float, float]]:
  grouped: dict[int, list[tuple[float, float]]] = {}
  for x, y, block_id in rows:
    grouped.setdefault(block_id, []).append((x, y))

  stats: dict[int, tuple[float, float]] = {}
  for block_id, block_rows in grouped.items():
    x_bar = sum(x for x, _ in block_rows) / len(block_rows)
    y_bar = sum(y for _, y in block_rows) / len(block_rows)
    sxx = sum((x - x_bar) ** 2 for x, _ in block_rows)
    sxy = sum((x - x_bar) * (y - y_bar) for x, y in block_rows)
    if isfinite(sxx) and isfinite(sxy) and sxx > 0.0:
      stats[block_id] = (sxx, sxy)
  return stats


def _fit_from_stats(stats: dict[int, tuple[float, float]]) -> BlockSlopeFit | None:
  if not stats:
    return None

  total_sxx = sum(sxx for sxx, _ in stats.values())
  total_sxy = sum(sxy for _, sxy in stats.values())
  if not (isfinite(total_sxx) and isfinite(total_sxy)) or total_sxx <= 0.0:
    return None
  slope = total_sxy / total_sxx
  if not isfinite(slope):
    return None

  block_slopes = {block_id: sxy / sxx for block_id, (sxx, sxy) in stats.items()}
  loo_slopes: dict[int, float] = {}
  for block_id, (sxx, sxy) in stats.items():
    loo_sxx = total_sxx - sxx
    loo_sxy = total_sxy - sxy
    if loo_sxx <= 0.0:
      continue
    loo = loo_sxy / loo_sxx
    if isfinite(loo):
      loo_slopes[block_id] = loo

  if len(stats) < 2 or len(loo_slopes) != len(stats):
    jackknife_se = float("inf")
    rel_se = float("inf")
  else:
    loo_mean = sum(loo_slopes.values()) / len(loo_slopes)
    jackknife_se = sqrt((len(loo_slopes) - 1) / len(loo_slopes) *
                        sum((loo - loo_mean) ** 2 for loo in loo_slopes.values()))
    rel_se = jackknife_se / abs(slope) if slope != 0.0 else float("inf")

  return BlockSlopeFit(
    slope=float(slope),
    jackknife_se=float(jackknife_se),
    rel_se=float(rel_se),
    block_slopes={int(k): float(v) for k, v in block_slopes.items() if isfinite(v)},
    loo_slopes={int(k): float(v) for k, v in loo_slopes.items()},
  )


def fit_block_slope(points: Iterable[Sequence[float]]) -> BlockSlopeFit | None:
  """Fit a centered-intercept slope and delete-one-block jackknife.

  Each input row is ``(x, y, block_id)``.  Block-specific intercepts are removed
  before pooling ``Sxy`` and ``Sxx``.  The returned jackknife uses every block
  represented in the input; callers apply their own informative-block gates.
  """
  return _fit_from_stats(_block_stats(_coerce_points(points)))


def fit_ratio_jackknife(left_points: Iterable[Sequence[float]], right_points: Iterable[Sequence[float]]):
  """Fit left/right slopes and jackknife their ratio over the union of blocks."""
  left_rows = _coerce_points(left_points)
  right_rows = _coerce_points(right_points)
  left_stats = _block_stats(left_rows)
  right_stats = _block_stats(right_rows)
  left_fit = _fit_from_stats(left_stats)
  right_fit = _fit_from_stats(right_stats)
  if left_fit is None or right_fit is None or right_fit.slope == 0.0:
    return None

  ratio = left_fit.slope / right_fit.slope
  union = sorted(set(left_fit.block_slopes) | set(right_fit.block_slopes))
  left_sxx = sum(sxx for sxx, _ in left_stats.values())
  left_sxy = sum(sxy for _, sxy in left_stats.values())
  right_sxx = sum(sxx for sxx, _ in right_stats.values())
  right_sxy = sum(sxy for _, sxy in right_stats.values())
  ratio_loo: dict[int, float] = {}
  for block_id in union:
    left_block_sxx, left_block_sxy = left_stats.get(block_id, (0.0, 0.0))
    right_block_sxx, right_block_sxy = right_stats.get(block_id, (0.0, 0.0))
    left_loo_sxx = left_sxx - left_block_sxx
    right_loo_sxx = right_sxx - right_block_sxx
    if left_loo_sxx <= 0.0 or right_loo_sxx <= 0.0:
      continue
    left_loo_slope = (left_sxy - left_block_sxy) / left_loo_sxx
    right_loo_slope = (right_sxy - right_block_sxy) / right_loo_sxx
    if not isfinite(left_loo_slope) or not isfinite(right_loo_slope) or right_loo_slope == 0.0:
      continue
    ratio_loo_value = left_loo_slope / right_loo_slope
    if isfinite(ratio_loo_value):
      ratio_loo[block_id] = ratio_loo_value

  if len(ratio_loo) != len(union) or not isfinite(ratio):
    ratio_se = float("inf")
    ratio_rel_se = float("inf")
  else:
    ratio_mean = sum(ratio_loo.values()) / len(ratio_loo)
    ratio_se = sqrt((len(ratio_loo) - 1) / len(ratio_loo) *
                    sum((value - ratio_mean) ** 2 for value in ratio_loo.values()))
    ratio_rel_se = ratio_se / abs(ratio) if ratio != 0.0 else float("inf")

  return {
    'left': left_fit,
    'right': right_fit,
    'ratio': float(ratio),
    'ratio_se': float(ratio_se),
    'ratio_rel_se': float(ratio_rel_se),
    'ratio_loo': {int(k): float(v) for k, v in ratio_loo.items()},
  }
