"""Jerk-limited longitudinal trajectory synthesis (port, decoupled).

Ported from the legacy ``custom_v2_trajectory.py``, with one clean-up: the synthesis is
decoupled from the legacy ``LongitudinalStackOutput`` object — it takes raw speed/accel
sequences instead, so it has no dependency on the retired stack interface. The jerk-limit
math (asymmetric positive-progress vs negative-retreat jerk budgets) and the model-time
synthesis grid are unchanged.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N
from openpilot.selfdrive.modeld.constants import ModelConstants

SYNTH_TRAJECTORY_DT = 0.2
POSITIVE_PROGRESS_JERK = 4.0
NORMAL_NEGATIVE_RETREAT_JERK = -5.0
A_TARGET_EPS = 1e-4


def preserve_seed_trajectory(output_a_target: float, planner_seed_scalar: bool, a_target: float) -> bool:
  """Keep a planner seed's own trajectory when the stack did not override its a_target."""
  if bool(planner_seed_scalar):
    return False
  return math.isclose(float(output_a_target), float(a_target), abs_tol=A_TARGET_EPS)


def synth_trajectory(speeds_in: Sequence[float], accels_in: Sequence[float], v_ego: float,
                     a_target: float, limit_jerk: bool) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
  speeds_in = tuple(speeds_in)
  accels_in = tuple(accels_in)
  v0 = v_ego if math.isfinite(v_ego) and v_ego >= 0.0 else (float(speeds_in[0]) if speeds_in else 0.0)
  prev_accel = float(accels_in[0]) if accels_in else float(a_target)
  dts = synth_trajectory_dts()
  accels: list[float] = []
  jerks: list[float] = []
  current_accel = prev_accel
  for dt in dts:
    if limit_jerk:
      delta = _clip(
        float(a_target) - current_accel,
        NORMAL_NEGATIVE_RETREAT_JERK * dt,
        POSITIVE_PROGRESS_JERK * dt,
      )
      next_accel = current_accel + delta
    else:
      next_accel = float(a_target)
    jerks.append((next_accel - current_accel) / dt)
    accels.append(next_accel)
    current_accel = next_accel

  speeds: list[float] = []
  current_speed = max(0.0, v0)
  for accel, dt in zip(accels, dts, strict=True):
    speeds.append(current_speed)
    current_speed = max(0.0, current_speed + accel * dt)
  return tuple(speeds), tuple(accels), tuple(jerks)


def synth_trajectory_dts(t_idxs: Any = None) -> tuple[float, ...]:
  if t_idxs is None:
    t_idxs = ModelConstants.T_IDXS
  try:
    times = tuple(float(t) for t in t_idxs[:CONTROL_N])
  except (TypeError, ValueError):
    return (SYNTH_TRAJECTORY_DT,) * CONTROL_N
  if len(times) < CONTROL_N or not all(math.isfinite(t) for t in times):
    return (SYNTH_TRAJECTORY_DT,) * CONTROL_N

  intervals = [times[idx + 1] - times[idx] for idx in range(CONTROL_N - 1)]
  dts = [*intervals, intervals[-1]]
  if not all(math.isfinite(dt) and dt > 0.0 for dt in dts):
    return (SYNTH_TRAJECTORY_DT,) * CONTROL_N
  return tuple(dts)


def _clip(value: float, lower: float, upper: float) -> float:
  return max(lower, min(upper, value))
