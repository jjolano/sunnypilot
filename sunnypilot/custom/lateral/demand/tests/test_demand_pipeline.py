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
  return LateralDemandPipelineInputs(
    lat_active=kwargs.get("lat_active", lat_active),
    v_ego=kwargs.get("v_ego", v_ego),
    roll=kwargs.get("roll", 0.0),
    desired_curvature=kwargs.get("desired_curvature", curvature),
    measured_curvature=kwargs.get("measured_curvature", curvature),
    position_x=kwargs.get("position_x", xs),
    position_y=kwargs.get("position_y", ys),
    position_y_std=kwargs.get("position_y_std", ystd),
    orientation_z=kwargs.get("orientation_z", yaw),
    orientation_rate_z=kwargs.get("orientation_rate_z", yaw_rate),
    lane_line_probs=kwargs.get("lane_line_probs", [0.9, 0.9, 0.9, 0.9]),
    frame_drop_perc=kwargs.get("frame_drop_perc", 0.0),
    model_age_s=kwargs.get("model_age_s", 0.0),
    model_data_v2_sp_valid=kwargs.get("model_data_v2_sp_valid", True),
    turn_direction=kwargs.get("turn_direction", 0),
    lane_change_state=kwargs.get("lane_change_state", 0),
    lane_change_direction=kwargs.get("lane_change_direction", 0),
    lane_change_state_valid=kwargs.get("lane_change_state_valid", True),
    left_blinker=kwargs.get("left_blinker", False),
    right_blinker=kwargs.get("right_blinker", False),
    steering_pressed=kwargs.get("steering_pressed", False),
    left_lane_y0=kwargs.get("left_lane_y0", None),
    right_lane_y0=kwargs.get("right_lane_y0", None),
    lateral_maneuver_curvature=kwargs.get("lateral_maneuver_curvature", None),
    smooth_model_path_curvature=kwargs.get("smooth_model_path_curvature", False),
    demand_jerk_smoothing_enabled=kwargs.get("demand_jerk_smoothing_enabled", False),
    lane_centering_assist_enabled=kwargs.get("lane_centering_assist_enabled", False),
    curve_memory_enabled=kwargs.get("curve_memory_enabled", False),
    curvature_limited=kwargs.get("curvature_limited", False),
  )


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
  r = p.update(valid_inputs(curvature=0.001, lane_centering_assist_enabled=False))
  for _ in range(49):
    r = p.update(valid_inputs(curvature=0.001, lane_centering_assist_enabled=False))
  assert r.demand.lane_centering_curvature_nudge == 0.0
  assert r.demand.lane_centering_assist_active is False


def test_valid_path_high_quality_passthrough():
  # A clean, low-curvature path with smoothing off should pass through near the request and
  # report high quality / not gated.
  p = LateralDemandPipeline(DT)
  r = p.update(valid_inputs(v_ego=20.0, curvature=0.001, smooth_model_path_curvature=False))
  for _ in range(20):
    r = p.update(valid_inputs(v_ego=20.0, curvature=0.001, smooth_model_path_curvature=False))
  assert r.demand.path_quality >= 0.7
  assert r.demand.processed_curvature == pytest.approx(0.001, abs=2e-3)
  assert r.demand.demand_source == DEMAND_SOURCE_MODEL_PATH


def test_stale_model_bridges_to_previous_and_measured_curvature():
  p = LateralDemandPipeline(DT)
  for _ in range(5):
    p.update(valid_inputs(v_ego=20.0, curvature=0.001, measured_curvature=0.001))

  r = p.update(valid_inputs(v_ego=20.0, curvature=0.02, desired_curvature=0.02,
                            measured_curvature=0.001, model_age_s=0.30))

  assert r.model_path_result.gated is True
  assert r.model_path_result.reason == "model_stale"
  assert abs(r.demand.processed_curvature - 0.001) < abs(0.02 - 0.001)


@pytest.mark.parametrize("age, stale", [(0.19, False), (0.20, False), (0.21, True), (float("inf"), True)])
def test_model_age_stale_threshold_boundary(age, stale):
  p = LateralDemandPipeline(DT)
  r = p.update(valid_inputs(v_ego=20.0, curvature=0.001, model_age_s=age))
  assert (r.model_path_result.reason == "model_stale") is stale


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


def test_nonfinite_processed_curvature_falls_back_to_zero_and_resets_memory():
  p = LateralDemandPipeline(DT)
  for _ in range(5):
    p.update(valid_inputs(curvature=0.003))
  r = p.update(valid_inputs(curvature=0.003, lateral_maneuver_curvature=float('nan')))
  assert r.demand.processed_curvature == 0.0
  assert math.isfinite(r.demand.processed_curvature)
  assert p.previous_desired_curvature == 0.0
  r2 = p.update(valid_inputs(curvature=0.003))
  assert math.isfinite(r2.demand.processed_curvature)
  assert r2.demand.demand_source == DEMAND_SOURCE_MODEL_PATH


def test_curve_memory_resumes_corner_after_standstill():
  # corner -> stop mid-corner -> launch, through the pipeline (which manages prev curvature across
  # the stop). With curve memory the gated launch resumes the corner; without it, it starts cold.
  def run(curve_memory: bool) -> float:
    p = LateralDemandPipeline(DT)
    for _ in range(60):                                       # driving a trusted corner (k=0.015 at 8 m/s)
      p.update(valid_inputs(v_ego=8.0, curvature=0.015, curve_memory_enabled=curve_memory))
    p.update(valid_inputs(v_ego=0.0, curvature=0.0, lat_active=False, curve_memory_enabled=curve_memory))
    for _ in range(20):                                       # held at a stop
      p.update(valid_inputs(v_ego=0.0, curvature=0.0, lat_active=False, curve_memory_enabled=curve_memory))
    # gated launch: low speed, high path std, conservative ("forgotten") raw curvature
    r = p.update(valid_inputs(v_ego=3.0, curvature=0.005, position_y_std=[1.6] * N,
                              curve_memory_enabled=curve_memory))
    if curve_memory:
      assert r.debug["curve_memory_source"] in ("memory", "vision", "vetoed")
    return float(r.demand.processed_curvature)

  assert run(True) > 0.008                 # resumes the corner (vs 0.005 raw / 0.0025 cold start)
  assert run(True) > 3.0 * run(False)      # vs cold start without curve memory


def test_curve_memory_steering_pressed_blocks_recall():
  p = LateralDemandPipeline(DT)
  for _ in range(5):
    p.update(valid_inputs(v_ego=8.0, curvature=0.02, curve_memory_enabled=True, steering_pressed=False))
  r = p.update(valid_inputs(v_ego=3.0, curvature=0.005, curve_memory_enabled=True, steering_pressed=True))
  assert r.debug["curve_memory_source"] == "driver_override"
  assert r.demand.processed_curvature >= 0.005


def test_lane_change_state_propagates_and_resets_memory():
  p = LateralDemandPipeline(DT)
  for _ in range(5):
    p.update(valid_inputs(v_ego=8.0, curvature=0.02, curve_memory_enabled=True, steering_pressed=False))
  r = p.update(valid_inputs(v_ego=3.0, curvature=0.005, curve_memory_enabled=True,
                            lane_change_state=1, lane_change_state_valid=True, steering_pressed=False))
  assert r.debug["curve_memory_source"] == "lane_change"
  r2 = p.update(valid_inputs(v_ego=3.0, curvature=0.005, curve_memory_enabled=True,
                             lane_change_state=0, lane_change_state_valid=True, steering_pressed=False))
  assert r2.demand.processed_curvature >= 0.005


def test_unknown_lane_change_state_suppresses_curve_memory():
  p = LateralDemandPipeline(DT)
  for _ in range(60):
    p.update(valid_inputs(v_ego=8.0, curvature=0.015, curve_memory_enabled=True, steering_pressed=False))
  r = p.update(valid_inputs(v_ego=3.0, curvature=0.005, position_y_std=[1.6] * N,
                            curve_memory_enabled=True, steering_pressed=False, lane_change_state_valid=False))
  assert r.debug["curve_memory_source"] == "lane_change"
  assert r.debug["curve_memory_active"] is False
  assert r.debug["curve_memory_samples"] == 0


def test_demand_jerk_smoothing_default_off_matches_existing_smoothing():
  baseline = LateralDemandPipeline(DT)
  candidate = LateralDemandPipeline(DT)
  sequence = [0.0, 0.0008, -0.0007, 0.0009, -0.0006, 0.0005]

  for k in sequence:
    b = baseline.update(valid_inputs(v_ego=15.0, curvature=k, smooth_model_path_curvature=True,
                                     demand_jerk_smoothing_enabled=False, steering_pressed=False))
    c = candidate.update(valid_inputs(v_ego=15.0, curvature=k, smooth_model_path_curvature=True,
                                      demand_jerk_smoothing_enabled=False, steering_pressed=False))
    assert c.demand.processed_curvature == pytest.approx(b.demand.processed_curvature)
    assert c.debug["demand_jerk_smoothing_active"] is False


def test_demand_jerk_smoothing_requires_model_path_smoothing():
  p = LateralDemandPipeline(DT)

  r = p.update(valid_inputs(v_ego=15.0, curvature=0.001, smooth_model_path_curvature=False,
                            demand_jerk_smoothing_enabled=True, steering_pressed=False))

  assert r.debug["demand_jerk_smoothing_active"] is False
  assert r.demand.processed_curvature == pytest.approx(0.001)


def test_demand_jerk_smoothing_bounds_near_straight_changes():
  p = LateralDemandPipeline(DT)
  outputs = []
  active_count = 0
  for k in ([0.0] * 5 + [0.001] * 8 + [-0.001] * 8 + [0.001] * 8):
    r = p.update(valid_inputs(v_ego=15.0, curvature=k, smooth_model_path_curvature=True,
                              demand_jerk_smoothing_enabled=True, steering_pressed=False))
    outputs.append(float(r.demand.processed_curvature))
    active_count += int(bool(r.debug["demand_jerk_smoothing_active"]))
    assert float(r.debug["demand_jerk_smoothing_lag"]) <= 0.00036

  assert active_count > 0
  assert max(abs(b - a) for a, b in zip(outputs, outputs[1:])) < 0.00055


def test_demand_jerk_smoothing_bypasses_and_resets_on_lane_change():
  p = LateralDemandPipeline(DT)
  for k in ([0.0] * 5 + [0.001] * 8):
    p.update(valid_inputs(v_ego=15.0, curvature=k, smooth_model_path_curvature=True,
                          demand_jerk_smoothing_enabled=True, steering_pressed=False))

  lane_change = p.update(valid_inputs(v_ego=15.0, curvature=-0.001, smooth_model_path_curvature=True,
                                      demand_jerk_smoothing_enabled=True, steering_pressed=False,
                                      lane_change_state=1, lane_change_state_valid=True))
  after = p.update(valid_inputs(v_ego=15.0, curvature=-0.001, smooth_model_path_curvature=True,
                                demand_jerk_smoothing_enabled=True, steering_pressed=False))

  assert lane_change.debug["demand_jerk_smoothing_active"] is False
  assert after.debug["demand_jerk_smoothing_active"] is False


def test_demand_jerk_smoothing_bypasses_when_driver_state_unknown():
  p = LateralDemandPipeline(DT)
  r = p.update(valid_inputs(v_ego=15.0, curvature=0.001, smooth_model_path_curvature=True,
                            demand_jerk_smoothing_enabled=True, steering_pressed=None))

  assert r.debug["demand_jerk_smoothing_active"] is False
