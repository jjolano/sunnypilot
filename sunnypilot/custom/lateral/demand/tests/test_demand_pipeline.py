"""Behavioral/property tests for the LateralDemandPipeline composition.

The ported processors keep their legacy behavior; these tests gate the PIPELINE wiring
(stage ordering, toggles, maneuver override, inactive fallback, bounded output). Per-stage
feel value is gated on the engaged corpus.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from openpilot.sunnypilot.custom.lateral.demand.pipeline import (
  LateralDemandPipeline,
  LateralDemandPipelineInputs,
)
from openpilot.sunnypilot.custom.lateral.demand.types import (
  DEMAND_SOURCE_FALLBACK_MEASURED,
  DEMAND_SOURCE_LATERAL_MANEUVER,
  DEMAND_SOURCE_MODEL_PATH,
)

DT = 0.01
N = 33  # >= ModelConstants.IDX_N and PATH_VALID_MIN_LEN


def valid_inputs(v_ego=20.0, curvature=0.001, lat_active=True, **kwargs):
  xs = [float(x) for x in range(N)]
  ys = [0.5 * curvature * x * x for x in range(N)]            # y ~ 0.5*k*x^2
  ystd = [0.1] * N                                           # < MAX_PATH_Y_STD
  yaw = [curvature * x for x in range(N)]                     # heading ~ k*x
  yaw_rate = [curvature * v_ego] * N
  base = dict(
    lat_active=lat_active, v_ego=v_ego, roll=0.0,
    desired_curvature=curvature, measured_curvature=curvature,
    position_x=xs, position_y=ys, position_y_std=ystd,
    orientation_z=yaw, orientation_rate_z=yaw_rate,
    lane_line_probs=[0.9, 0.9, 0.9, 0.9],
  )
  base.update(kwargs)
  return LateralDemandPipelineInputs(**base)


def test_constructs_and_runs_bounded():
  p = LateralDemandPipeline(DT)
  rng = np.random.default_rng(20260613)
  for _ in range(800):
    k = float(rng.uniform(-0.02, 0.02))
    r = p.update(valid_inputs(v_ego=float(rng.uniform(0, 35)), curvature=k))
    assert math.isfinite(r.demand.processed_curvature)
    # processed curvature must stay within a sane band of the request (gating only restricts)
    assert abs(r.demand.processed_curvature) <= 0.5
    assert r.demand.demand_source == DEMAND_SOURCE_MODEL_PATH


def test_inactive_falls_back_to_measured():
  p = LateralDemandPipeline(DT)
  r = p.update(valid_inputs(lat_active=False, curvature=0.01))
  assert r.demand.demand_source == DEMAND_SOURCE_FALLBACK_MEASURED
  assert r.demand.processed_curvature == pytest.approx(0.01)


def test_lateral_maneuver_override_takes_pipeline():
  p = LateralDemandPipeline(DT)
  for _ in range(10):
    p.update(valid_inputs(curvature=0.002))
  r = p.update(valid_inputs(curvature=0.002, lateral_maneuver_curvature=0.037))
  assert r.demand.demand_source == DEMAND_SOURCE_LATERAL_MANEUVER
  assert r.demand.processed_curvature == pytest.approx(0.037)


def test_lane_centering_toggle_off_means_no_nudge():
  p = LateralDemandPipeline(DT)
  for _ in range(50):
    r = p.update(valid_inputs(curvature=0.001, lane_centering_assist_enabled=False))
  assert r.demand.lane_centering_curvature_nudge == 0.0
  assert r.demand.lane_centering_assist_active is False


def test_valid_path_high_quality_passthrough():
  # A clean, low-curvature path with smoothing off should pass through near the request and
  # report high quality / not gated.
  p = LateralDemandPipeline(DT)
  r = None
  for _ in range(20):
    r = p.update(valid_inputs(v_ego=20.0, curvature=0.001, smooth_model_path_curvature=False))
  assert r.demand.path_quality >= 0.7
  assert r.demand.processed_curvature == pytest.approx(0.001, abs=2e-3)
  assert r.demand.demand_source == DEMAND_SOURCE_MODEL_PATH


def test_invalid_path_is_gated_and_falls_back():
  p = LateralDemandPipeline(DT)
  # prime a previous curvature, then feed an empty/invalid path
  for _ in range(5):
    p.update(valid_inputs(curvature=0.003))
  bad = LateralDemandPipelineInputs(
    lat_active=True, v_ego=20.0, roll=0.0, desired_curvature=0.003, measured_curvature=0.003,
    position_x=(), position_y=(), position_y_std=(), orientation_z=(), orientation_rate_z=(),
    lane_line_probs=(0.9, 0.9, 0.9),
  )
  r = p.update(bad)
  assert r.model_path_result.gated is True
  assert r.model_path_result.reason == "invalid_path"
  assert math.isfinite(r.demand.processed_curvature)


def test_debug_records_each_stage():
  p = LateralDemandPipeline(DT)
  r = p.update(valid_inputs(curvature=0.002))
  for key in ("raw_curvature", "model_path_curvature", "lane_change_blend",
              "lane_centering_nudge", "processed_curvature", "demand_source"):
    assert key in r.debug
  assert r.debug["raw_curvature"] == pytest.approx(0.002)
