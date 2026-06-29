"""Focused tests for ModelPathProcessor straight-road damping behavior."""
from __future__ import annotations

import numpy as np
import pytest

from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.sunnypilot.custom.lateral.demand.model_path_processor import (
  DAMPING_TAU_SPEED_BP,
  DAMPING_TAU_S,
  ModelPathProcessor,
  ModelPathProcessorInputs,
  STRAIGHT_ROAD_DAMPING_BLEND_BP_LAT_ACCEL,
  STRAIGHT_ROAD_DAMPING_FULL_SPEED,
  STRAIGHT_ROAD_DAMPING_MAX_LAT_ACCEL,
  STRAIGHT_ROAD_DAMPING_MIN_SPEED,
  STRAIGHT_ROAD_DAMPING_TAU_S,
)

N = ModelConstants.IDX_N


def _inputs(
  v_ego: float = 15.0,
  curvature: float = 1e-5,
  previous_desired_curvature: float = 0.0,
  measured_curvature: float = 0.0,
  lane_change_active: bool = False,
  smooth_model_path_curvature: bool = True,
  y_std: float = 0.05,
) -> ModelPathProcessorInputs:
  xs = [float(i) for i in range(N)]
  ys = [0.5 * curvature * x * x for x in xs]
  y_stds = [y_std] * N
  yaws = [curvature * x for x in xs]
  yaw_rates = [curvature * 20.0] * N
  lane_probs = [0.9, 0.9, 0.9, 0.9]

  return ModelPathProcessorInputs(
    lat_active=True,
    v_ego=v_ego,
    desired_curvature=curvature,
    measured_curvature=measured_curvature,
    previous_desired_curvature=previous_desired_curvature,
    position_x=xs,
    position_y=ys,
    position_y_std=y_stds,
    orientation_z=yaws,
    orientation_rate_z=yaw_rates,
    lane_line_probs=lane_probs,
    smooth_model_path_curvature=smooth_model_path_curvature,
    lane_change_active=lane_change_active,
  )


def _base_tau(v_ego: float) -> float:
  return float(np.interp(v_ego, DAMPING_TAU_SPEED_BP, DAMPING_TAU_S))


@pytest.mark.parametrize("v_ego", [13.0, 15.0, 17.0])
def test_straight_road_damping_active_mid_speed_near_straight(v_ego: float):
  proc = ModelPathProcessor()
  curvature = 1e-5
  inp = _inputs(v_ego=v_ego, curvature=curvature)

  result = proc.update(inp)

  assert result.straight_road_damping_active is True
  assert result.smoothing_tau_s > _base_tau(v_ego) + 1e-6
  assert result.smoothing_tau_s <= STRAIGHT_ROAD_DAMPING_TAU_S + 1e-6

  result2 = proc.update(inp)
  assert result2.straight_road_damping_active is True
  assert result2.desired_curvature == pytest.approx(result.desired_curvature, abs=1e-9)


def test_straight_road_damping_inactive_below_min_speed():
  proc = ModelPathProcessor()
  v_ego = STRAIGHT_ROAD_DAMPING_MIN_SPEED - 2.0
  result = proc.update(_inputs(v_ego=v_ego, curvature=1e-5))

  assert result.straight_road_damping_active is False
  assert result.smoothing_tau_s == pytest.approx(_base_tau(v_ego), abs=1e-6)


def test_straight_road_damping_inactive_above_lat_accel_cap():
  proc = ModelPathProcessor()
  v_ego = 15.0
  curvature = (STRAIGHT_ROAD_DAMPING_MAX_LAT_ACCEL + 0.05) / (v_ego * v_ego)
  result = proc.update(_inputs(v_ego=v_ego, curvature=curvature))

  assert result.straight_road_damping_active is False
  assert result.smoothing_tau_s == pytest.approx(_base_tau(v_ego), abs=1e-6)


def test_straight_road_damping_inactive_during_lane_change():
  proc = ModelPathProcessor()
  result = proc.update(_inputs(v_ego=15.0, curvature=1e-5, lane_change_active=True))

  assert result.straight_road_damping_active is False
  assert result.smoothing_tau_s == pytest.approx(_base_tau(15.0), abs=1e-6)


def test_straight_road_damping_inactive_when_smoothing_disabled():
  proc = ModelPathProcessor()
  result = proc.update(_inputs(v_ego=15.0, curvature=1e-5, smooth_model_path_curvature=False))

  assert result.straight_road_damping_active is False
  assert result.smoothing_tau_s == 0.0
  assert result.damping_alpha == 0.0


def test_straight_road_damping_unchanged_for_low_quality_path():
  proc = ModelPathProcessor()
  result = proc.update(_inputs(v_ego=15.0, curvature=1e-5, y_std=2.0))

  assert result.gated is True
  assert result.straight_road_damping_active is False


def test_straight_road_damping_partial_blend_respects_cap():
  proc = ModelPathProcessor()
  v_ego = STRAIGHT_ROAD_DAMPING_FULL_SPEED
  mid_lat_accel = (STRAIGHT_ROAD_DAMPING_BLEND_BP_LAT_ACCEL[0] +
                   STRAIGHT_ROAD_DAMPING_BLEND_BP_LAT_ACCEL[-1]) * 0.5
  curvature = mid_lat_accel / (v_ego * v_ego)
  result = proc.update(_inputs(v_ego=v_ego, curvature=curvature))

  assert result.straight_road_damping_active is True
  base = _base_tau(v_ego)
  assert base < result.smoothing_tau_s < STRAIGHT_ROAD_DAMPING_TAU_S


def test_straight_road_deadband_holds_small_change_allows_large_step():
  proc = ModelPathProcessor()
  v_ego = 15.0
  speed_sq = v_ego * v_ego
  start_curvature = 1e-5

  result1 = proc.update(_inputs(v_ego=v_ego, curvature=start_curvature))
  assert result1.straight_road_damping_active is True

  small_target = start_curvature + 1e-7
  result2 = proc.update(_inputs(v_ego=v_ego, curvature=small_target,
                                previous_desired_curvature=result1.desired_curvature))
  assert result2.straight_road_damping_active is True
  assert result2.desired_curvature == pytest.approx(result1.desired_curvature, abs=1e-9)

  large_target = start_curvature + 0.0003
  assert abs(large_target - start_curvature) * speed_sq > 0.06
  result3 = proc.update(_inputs(v_ego=v_ego, curvature=large_target,
                                previous_desired_curvature=result2.desired_curvature))
  assert result3.straight_road_damping_active is True
  assert result3.desired_curvature != pytest.approx(result2.desired_curvature, abs=1e-9)
