"""Behavioral/property tests for the LateralDemandPipeline composition.

The ported processors keep their legacy behavior; these tests gate the PIPELINE wiring
(stage ordering, toggles, maneuver override, inactive fallback, bounded output). Per-stage
feel value is gated on the engaged corpus.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.sunnypilot.custom.lateral.demand.lane_centering_assist import (
  LANE_CENTERING_ASSIST_OK_REASON,
  LANE_CENTERING_ASSIST_PATH_REASON_COOLDOWN_REASON,
  LaneCenteringAssistInputs,
  LaneCenteringAssistTracker,
)
from openpilot.sunnypilot.custom.lateral.demand.model_path_processor import (
  LOW_QUALITY_BLEND_THRESHOLD,
  NEAR_ZERO_BLEND_SCALE,
  NEAR_ZERO_CURVATURE_BP,
  SMOOTHED_CURVATURE_MAX_LAT_ACCEL_DELTA,
  SMOOTHED_CURVATURE_SPEED_BP,
  SOFT_GATE_MAX_SAME_SIGN_RAW_LAT_ACCEL_DELTA,
  DEMAND_JERK_SMOOTH_CURVE_EXIT_FALL_MIN_FRAMES,
  DEMAND_JERK_SMOOTH_CURVE_EXIT_MAX_LAT_ACCEL,
  DEMAND_JERK_SMOOTH_CURVE_EXIT_NEAR_ZERO_LAT_ACCEL,
  DEMAND_JERK_SMOOTH_LAG_LAT_ACCEL,
  DEMAND_JERK_SMOOTH_MAX_CURVATURE,
  ModelPathProcessor,
  ModelPathProcessorInputs,
  ModelPathProcessorResult,
)
from openpilot.sunnypilot.custom.lateral.demand import model_path_processor_v1
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
    lane_line_stds=kwargs.get("lane_line_stds", [0.1, 0.1, 0.1, 0.1]),
    lane_lines=kwargs.get("lane_lines", ()),
    frame_drop_perc=kwargs.get("frame_drop_perc", 0.0),
    model_age_s=kwargs.get("model_age_s", 0.0),
    yaw_rate=kwargs.get("yaw_rate", None),
    steering_rate_deg=kwargs.get("steering_rate_deg", None),
    steer_limited=kwargs.get("steer_limited", False),
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


def spatial_smoothing_inputs(v_ego=20.0, desired_curvature=0.001, candidate_curvature=0.002):
  t_idxs = ModelConstants.T_IDXS
  return ModelPathProcessorInputs(
    lat_active=True,
    v_ego=v_ego,
    desired_curvature=desired_curvature,
    measured_curvature=desired_curvature,
    previous_desired_curvature=desired_curvature,
    position_x=[float(x) for x in range(len(t_idxs))],
    position_y=[0.0] * len(t_idxs),
    position_y_std=[0.1] * len(t_idxs),
    orientation_z=[float(candidate_curvature * v_ego * t) for t in t_idxs],
    orientation_rate_z=[float(candidate_curvature * v_ego)] * len(t_idxs),
    lane_line_probs=[0.9, 0.9, 0.9, 0.9],
    smooth_model_path_curvature=True,
  )


def temporal_smoothing_inputs(
  desired_curvature: float,
  measured_curvature: float,
  lane_line_probs: list[float],
  previous_desired_curvature: float | None = None,
  v_ego: float = 10.0,
) -> ModelPathProcessorInputs:
  t_idxs = ModelConstants.T_IDXS
  n = len(t_idxs)
  previous = desired_curvature if previous_desired_curvature is None else previous_desired_curvature
  return ModelPathProcessorInputs(
    lat_active=True,
    v_ego=v_ego,
    desired_curvature=desired_curvature,
    measured_curvature=measured_curvature,
    previous_desired_curvature=previous,
    position_x=[float(x) for x in range(n)],
    position_y=[0.0] * n,
    position_y_std=[0.1] * n,
    orientation_z=[float(desired_curvature * v_ego * t) for t in t_idxs],
    orientation_rate_z=[float(desired_curvature * v_ego)] * n,
    lane_line_probs=lane_line_probs,
    smooth_model_path_curvature=True,
  )


def _mpp_inputs(v_ego: float = 15.0, curvature: float = 0.001, **kwargs) -> ModelPathProcessorInputs:
  t_idxs = ModelConstants.T_IDXS
  n = len(t_idxs)
  v = kwargs.get("v_ego", v_ego)
  k = kwargs.get("desired_curvature", curvature)
  return ModelPathProcessorInputs(
    lat_active=True,
    v_ego=v,
    desired_curvature=k,
    measured_curvature=kwargs.get("measured_curvature", k),
    previous_desired_curvature=kwargs.get("previous_desired_curvature", k),
    position_x=kwargs.get("position_x", [float(x) for x in range(n)]),
    position_y=kwargs.get("position_y", [0.0] * n),
    position_y_std=kwargs.get("position_y_std", [0.1] * n),
    orientation_z=kwargs.get("orientation_z", [float(k * v * t) for t in t_idxs]),
    orientation_rate_z=kwargs.get("orientation_rate_z", [float(k * v)] * n),
    lane_line_probs=kwargs.get("lane_line_probs", [0.9, 0.9, 0.9, 0.9]),
    frame_drop_perc=kwargs.get("frame_drop_perc", 0.0),
    model_age_s=kwargs.get("model_age_s", 0.0),
    turn_curvature_sign=kwargs.get("turn_curvature_sign", 0),
    smooth_model_path_curvature=kwargs.get("smooth_model_path_curvature", True),
    demand_jerk_smoothing_enabled=kwargs.get("demand_jerk_smoothing_enabled", True),
    demand_jerk_smoothing_allowed=kwargs.get("demand_jerk_smoothing_allowed", True),
    lane_change_active=kwargs.get("lane_change_active", False),
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


def test_sensor_confidence_shadow_metrics_do_not_change_actuation_or_path_gate():
  baseline = LateralDemandPipeline(DT)
  candidate = LateralDemandPipeline(DT)
  base = baseline.update(valid_inputs(v_ego=20.0, curvature=0.001, measured_curvature=0.001))
  with_sensor = candidate.update(valid_inputs(v_ego=20.0, curvature=0.001, measured_curvature=0.001,
                                             yaw_rate=0.02, steering_rate_deg=3.0, steering_pressed=False))

  assert with_sensor.demand.processed_curvature == pytest.approx(base.demand.processed_curvature)
  assert with_sensor.model_path_result.reason == base.model_path_result.reason
  assert with_sensor.model_path_result.gated == base.model_path_result.gated
  assert with_sensor.debug["sensor_confidence_available"] is True
  assert with_sensor.debug["sensor_confidence_block_reason"] == "ok"
  assert "sensor_model_measured_lat_accel_delta" in with_sensor.debug


def test_sensor_suppress_candidate_is_debug_only():
  baseline = LateralDemandPipeline(DT)
  candidate = LateralDemandPipeline(DT)
  base = baseline.update(valid_inputs(v_ego=20.0, curvature=0.003, measured_curvature=0.0))
  with_sensor = candidate.update(valid_inputs(v_ego=20.0, curvature=0.003, measured_curvature=0.0,
                                             yaw_rate=0.0, steering_rate_deg=3.0, steering_pressed=False))

  assert with_sensor.debug["sensor_suppress_candidate"] is True
  assert with_sensor.demand.processed_curvature == pytest.approx(base.demand.processed_curvature)
  assert with_sensor.model_path_result.reason == base.model_path_result.reason
  assert with_sensor.model_path_result.gated == base.model_path_result.gated


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


def _prime_curve_exit_fall(proc: ModelPathProcessor, raw_base: float, delta: float = 0.0005) -> None:
  """Set history so that one more valid falling frame satisfies the sustained-fall requirement."""
  proc._curve_exit_prev_raw_base = raw_base + math.copysign(delta, raw_base)
  proc._curve_exit_fall_frames = DEMAND_JERK_SMOOTH_CURVE_EXIT_FALL_MIN_FRAMES - 1


@pytest.mark.parametrize("v_ego,raw_base,target", [
  (15.0, 0.0015, 0.0025),
  (15.0, -0.0015, -0.0025),
])
def test_curve_exit_smoothing_activates_falling_and_bounds_raw_lag(v_ego, raw_base, target):
  proc = ModelPathProcessor()
  inputs = _mpp_inputs(v_ego=v_ego)
  v_sq = v_ego * v_ego
  assert abs(target) > DEMAND_JERK_SMOOTH_MAX_CURVATURE
  assert abs(raw_base) < abs(target)
  assert max(abs(raw_base), abs(target)) * v_sq <= DEMAND_JERK_SMOOTH_CURVE_EXIT_MAX_LAT_ACCEL

  _prime_curve_exit_fall(proc, raw_base=raw_base)
  proc._demand_jerk_smoothed_curvature = target
  candidate, active, step, lag = proc._apply_demand_jerk_smoothing(
    inputs, raw_base, target, 1.0, "ok", 0.0
  )
  assert active is True
  assert abs(candidate - raw_base) * v_sq <= DEMAND_JERK_SMOOTH_LAG_LAT_ACCEL + 1e-9
  assert abs(candidate) <= max(abs(raw_base), abs(target))


def test_curve_entry_smoothing_inactive_above_near_straight():
  proc = ModelPathProcessor()
  inputs = _mpp_inputs(v_ego=15.0)
  # Curve-entry shape: raw has risen above the still-low damped target.
  raw_base = 0.0025
  target = 0.0015
  _prime_curve_exit_fall(proc, raw_base=raw_base)
  proc._demand_jerk_smoothed_curvature = target
  candidate, active, step, lag = proc._apply_demand_jerk_smoothing(
    inputs, raw_base, target, 1.0, "ok", 0.0
  )
  assert active is False
  assert candidate == pytest.approx(target)


@pytest.mark.parametrize("raw_base,target", [
  (0.0020, -0.0025),
  (-0.0020, 0.0025),
])
def test_curve_exit_opposite_sign_not_smoothed(raw_base, target):
  proc = ModelPathProcessor()
  inputs = _mpp_inputs(v_ego=15.0)
  # raw_base is below the curvature threshold but represents meaningful lateral accel.
  assert abs(raw_base) * (15.0 ** 2) > DEMAND_JERK_SMOOTH_CURVE_EXIT_NEAR_ZERO_LAT_ACCEL
  _prime_curve_exit_fall(proc, raw_base=raw_base)
  proc._demand_jerk_smoothed_curvature = target
  candidate, active, step, lag = proc._apply_demand_jerk_smoothing(
    inputs, raw_base, target, 1.0, "ok", 0.0
  )
  assert active is False
  assert candidate == pytest.approx(target)


@pytest.mark.parametrize("gate", [
  {"lane_change_active": True},
  {"demand_jerk_smoothing_allowed": False},
  {"turn_curvature_sign": 1},
])
def test_curve_exit_blocked_by_lane_change_or_steering_gate(gate):
  proc = ModelPathProcessor()
  inputs = _mpp_inputs(v_ego=15.0, **gate)
  raw_base = 0.0015
  target = 0.0025
  _prime_curve_exit_fall(proc, raw_base=raw_base)
  proc._demand_jerk_smoothed_curvature = target
  candidate, active, step, lag = proc._apply_demand_jerk_smoothing(
    inputs, raw_base, target, 1.0, "ok", 0.0
  )
  assert active is False
  assert candidate == pytest.approx(target)


def test_curve_exit_raw_reference_cap_wins_over_stale_smoothed():
  proc = ModelPathProcessor()
  inputs = _mpp_inputs(v_ego=15.0)
  raw_base = 0.0010
  target = 0.0025
  stale_high = 0.0030
  v_sq = 15.0 * 15.0
  assert max(abs(stale_high), abs(target)) * v_sq <= DEMAND_JERK_SMOOTH_CURVE_EXIT_MAX_LAT_ACCEL

  _prime_curve_exit_fall(proc, raw_base=raw_base)
  proc._demand_jerk_smoothed_curvature = stale_high
  candidate, active, step, lag = proc._apply_demand_jerk_smoothing(
    inputs, raw_base, target, 1.0, "ok", 0.0
  )
  assert active is True
  raw_lag_limit = DEMAND_JERK_SMOOTH_LAG_LAT_ACCEL / v_sq
  assert abs(candidate - raw_base) <= raw_lag_limit + 1e-9
  assert abs(candidate) <= abs(stale_high)


def test_curve_exit_steady_curve_no_guard():
  proc = ModelPathProcessor()
  inputs = _mpp_inputs(v_ego=15.0)
  raw_base = 0.0015
  target = 0.0025
  # No sustained fall history: raw is below target but not recently falling.
  proc._curve_exit_prev_raw_base = raw_base
  proc._curve_exit_fall_frames = 0
  proc._demand_jerk_smoothed_curvature = target
  candidate, active, step, lag = proc._apply_demand_jerk_smoothing(
    inputs, raw_base, target, 1.0, "ok", 0.0
  )
  assert active is False
  assert candidate == pytest.approx(target)


def test_curve_exit_one_frame_dip_no_guard():
  proc = ModelPathProcessor()
  inputs = _mpp_inputs(v_ego=15.0)
  raw_base = 0.0015
  target = 0.0025
  # Only a single-frame dip from the previous raw magnitude; history should count one
  # fall and therefore not yet satisfy the sustained-fall requirement.
  proc._curve_exit_prev_raw_base = raw_base + 0.0005
  proc._curve_exit_fall_frames = 0
  proc._demand_jerk_smoothed_curvature = target
  candidate, active, step, lag = proc._apply_demand_jerk_smoothing(
    inputs, raw_base, target, 1.0, "ok", 0.0
  )
  assert active is False


def test_curve_exit_high_speed_shallow_opposite_sign_no_guard():
  proc = ModelPathProcessor()
  v_ego = 22.0
  inputs = _mpp_inputs(v_ego=v_ego)
  raw_base = 0.0005  # below curvature threshold but meaningful lateral accel at high speed
  target = -0.0020
  v_sq = v_ego ** 2
  # Curvature is shallow but lateral accel is well above the near-zero threshold.
  assert abs(raw_base) < DEMAND_JERK_SMOOTH_MAX_CURVATURE
  assert abs(raw_base) * v_sq > DEMAND_JERK_SMOOTH_CURVE_EXIT_NEAR_ZERO_LAT_ACCEL
  _prime_curve_exit_fall(proc, raw_base=raw_base)
  proc._demand_jerk_smoothed_curvature = target
  candidate, active, step, lag = proc._apply_demand_jerk_smoothing(
    inputs, raw_base, target, 1.0, "ok", 0.0
  )
  assert active is False
  assert candidate == pytest.approx(target)


def test_curve_exit_guard_activates_through_pipeline():
  p = LateralDemandPipeline(DT)
  v_ego = 15.0
  v_sq = v_ego * v_ego
  sequence = [0.003] * 10 + [0.0025, 0.0020, 0.0016, 0.0012, 0.0008, 0.0004]
  active_above_ns = False
  lag_violations = 0

  for k in sequence:
    r = p.update(valid_inputs(v_ego=v_ego, curvature=k, smooth_model_path_curvature=True,
                              demand_jerk_smoothing_enabled=True, steering_pressed=False))
    raw_k = float(r.debug["raw_curvature"])
    if r.debug["demand_jerk_smoothing_active"] and abs(raw_k) > DEMAND_JERK_SMOOTH_MAX_CURVATURE:
      active_above_ns = True
      raw_lag_lat_accel = abs(r.demand.processed_curvature - raw_k) * v_sq
      if raw_lag_lat_accel > DEMAND_JERK_SMOOTH_LAG_LAT_ACCEL + 1e-9:
        lag_violations += 1

  assert active_above_ns is True
  assert lag_violations == 0


def test_spatial_smoothing_blend_is_bounded_by_quality_and_trust():
  v_ego = 20.0
  desired_curvature = 0.001

  for candidate_curvature in (0.002, 0.0):
    candidate_delta_lat_accel = (candidate_curvature - desired_curvature) * v_ego * v_ego
    for quality in (0.80, 0.90, 1.00):
      quality_alpha = float(np.interp(quality, [LOW_QUALITY_BLEND_THRESHOLD, 1.0], [0.0, 1.0]))
      max_delta_lat_accel = float(np.interp(v_ego, SMOOTHED_CURVATURE_SPEED_BP,
                                            SMOOTHED_CURVATURE_MAX_LAT_ACCEL_DELTA)) * quality_alpha
      for trust_penalty in (0.0, 0.4, 0.8):
        result = ModelPathProcessor._smoothed_path_curvature(
          spatial_smoothing_inputs(v_ego, desired_curvature, candidate_curvature),
          desired_curvature,
          quality,
          trust_penalty,
        )

        assert result is not None
        correction_lat_accel = (result - desired_curvature) * v_ego * v_ego
        assert math.copysign(1.0, correction_lat_accel) == math.copysign(1.0, candidate_delta_lat_accel)
        assert abs(correction_lat_accel) <= min(abs(candidate_delta_lat_accel), max_delta_lat_accel)

  assert ModelPathProcessor._smoothed_path_curvature(
    spatial_smoothing_inputs(v_ego, desired_curvature, candidate_curvature),
    desired_curvature,
    LOW_QUALITY_BLEND_THRESHOLD,
    0.0,
  ) is None
  assert ModelPathProcessor._smoothed_path_curvature(
    spatial_smoothing_inputs(v_ego, desired_curvature, candidate_curvature),
    desired_curvature,
    1.0,
    1.0,
  ) is None


def test_spatial_smoothing_correction_shrinks_as_trust_penalty_grows():
  v_ego = 20.0
  desired_curvature = 0.001
  candidate_curvature = 0.002
  corrections = []

  for trust_penalty in (0.0, 0.25, 0.50, 0.75):
    result = ModelPathProcessor._smoothed_path_curvature(
      spatial_smoothing_inputs(v_ego, desired_curvature, candidate_curvature),
      desired_curvature,
      1.0,
      trust_penalty,
    )
    assert result is not None
    corrections.append((result - desired_curvature) * v_ego * v_ego)

  assert all(earlier > later for earlier, later in zip(corrections, corrections[1:]))


def test_spatial_smoothing_near_zero_scale_is_monotonic():
  v_ego = 20.0
  curvature_values = [0.0, NEAR_ZERO_CURVATURE_BP[1] / 2.0, NEAR_ZERO_CURVATURE_BP[1], 0.001]
  corrections = []

  for desired_curvature in curvature_values:
    candidate_curvature = desired_curvature + 0.001
    result = ModelPathProcessor._smoothed_path_curvature(
      spatial_smoothing_inputs(v_ego, desired_curvature, candidate_curvature),
      desired_curvature,
      1.0,
      0.0,
    )
    assert result is not None
    corrections.append((result - desired_curvature) * v_ego * v_ego)

  assert all(earlier <= later for earlier, later in zip(corrections, corrections[1:]))
  assert corrections[0] < corrections[-1]
  assert corrections[0] == pytest.approx(corrections[-1] * NEAR_ZERO_BLEND_SCALE[0], rel=1e-6)


def test_lane_centering_assist_path_reason_cooldown_blocks_reactivation():
  tracker = LaneCenteringAssistTracker()
  xs = [float(x) for x in range(N)]
  ys = [0.01 * x for x in xs]
  yaws = [0.0] * N

  def make_inputs(path_reason: str) -> LaneCenteringAssistInputs:
    return LaneCenteringAssistInputs(
      lat_active=True,
      v_ego=20.0,
      measured_curvature=0.0,
      model_curvature=0.0,
      previous_processed_curvature=0.0,
      path_quality=1.0,
      path_reason=path_reason,
      lane_change_shaping_active=False,
      lane_change_blend=0.0,
      curvature_limited=False,
      steering_pressed=False,
      left_blinker=False,
      right_blinker=False,
      position_x=xs,
      position_y=ys,
      orientation_z=yaws,
      lane_line_probs=[0.9, 0.9, 0.9, 0.9],
    )

  ok = make_inputs(LANE_CENTERING_ASSIST_OK_REASON)
  r = tracker.update(ok, DT)
  assert r.active is True
  assert r.reason == "growing_lateral_error"
  assert r.curvature_nudge > 0.0

  bad = make_inputs("low_lane_confidence")
  r = tracker.update(bad, DT)
  assert r.active is False
  assert r.reason == "path_reason"
  assert r.curvature_nudge == pytest.approx(0.0)

  r = tracker.update(ok, DT)
  assert r.active is False
  assert r.reason == LANE_CENTERING_ASSIST_PATH_REASON_COOLDOWN_REASON

  for _ in range(49):
    r = tracker.update(ok, DT)
  assert r.active is False
  assert r.reason == LANE_CENTERING_ASSIST_PATH_REASON_COOLDOWN_REASON

  for _ in range(10):
    r = tracker.update(ok, DT)
  assert r.active is True
  assert r.curvature_nudge > 0.0


def test_low_lane_confidence_soft_gate_does_not_amplify_raw_curvature():
  proc = ModelPathProcessor()
  t_idxs = ModelConstants.T_IDXS
  n = len(t_idxs)

  def make_inputs(desired_curvature: float, previous_desired_curvature: float) -> ModelPathProcessorInputs:
    v_ego = 10.0
    return ModelPathProcessorInputs(
      lat_active=True,
      v_ego=v_ego,
      desired_curvature=desired_curvature,
      measured_curvature=0.0,
      previous_desired_curvature=previous_desired_curvature,
      position_x=[float(x) for x in range(n)],
      position_y=[0.0] * n,
      position_y_std=[0.1] * n,
      orientation_z=[float(desired_curvature * v_ego * t) for t in t_idxs],
      orientation_rate_z=[float(desired_curvature * v_ego)] * n,
      lane_line_probs=[0.9, 0.1, 0.1, 0.9],
      smooth_model_path_curvature=False,
    )

  for _ in range(2):
    proc.update(make_inputs(0.0, 0.0))

  raw = 0.005
  previous = 0.01
  result = proc.update(make_inputs(raw, previous))
  assert result.reason == "low_lane_confidence"
  assert result.gated is True
  margin = SOFT_GATE_MAX_SAME_SIGN_RAW_LAT_ACCEL_DELTA / (10.0 ** 2)
  assert abs(result.desired_curvature) <= abs(raw) + margin + 1e-9
  assert math.copysign(1.0, result.desired_curvature) == math.copysign(1.0, raw)


def test_low_lane_confidence_at_threshold_does_not_inherit_stale_temporal_curvature():
  proc = ModelPathProcessor()
  stale_curvature = -0.02
  good_confidence = [0.9, 0.9, 0.9, 0.9]
  low_confidence = [0.9, 0.1, 0.1, 0.9]

  for _ in range(10):
    proc.update(temporal_smoothing_inputs(stale_curvature, stale_curvature, good_confidence))

  raw = 0.005
  for _ in range(2):
    proc.update(temporal_smoothing_inputs(raw, raw, low_confidence))

  result = proc.update(temporal_smoothing_inputs(raw, raw, low_confidence))

  assert result.reason == "low_lane_confidence"
  assert result.quality == pytest.approx(LOW_QUALITY_BLEND_THRESHOLD)
  assert result.desired_curvature > 0.0
  assert abs(result.desired_curvature - raw) < abs(stale_curvature - raw) * 0.25


def test_clean_frame_after_soft_gate_reseeds_temporal_curvature():
  proc = ModelPathProcessor()
  stale_curvature = -0.02
  good_confidence = [0.9, 0.9, 0.9, 0.9]
  low_confidence = [0.9, 0.1, 0.1, 0.9]

  for _ in range(10):
    proc.update(temporal_smoothing_inputs(stale_curvature, stale_curvature, good_confidence))

  raw = 0.005
  degraded = None
  for _ in range(3):
    degraded = proc.update(temporal_smoothing_inputs(raw, 0.0, low_confidence))

  assert degraded is not None
  assert degraded.reason == "low_lane_confidence"
  assert degraded.gated is True
  assert degraded.quality < LOW_QUALITY_BLEND_THRESHOLD

  clean = None
  for _ in range(5):
    clean = proc.update(temporal_smoothing_inputs(raw, raw, good_confidence))
    if clean.reason == "ok":
      break

  assert clean is not None
  assert clean.reason == "ok"
  assert clean.gated is False
  assert clean.desired_curvature > 0.0
  assert abs(clean.desired_curvature - raw) < abs(stale_curvature - raw) * 0.25


def _lane_lines(center_offset: float = 0.0, width: float = 4.0, curvature: float = 0.0):
  xs = [float(x) for x in range(N)]
  left_y = [center_offset - width * 0.5 + 0.5 * curvature * x * x for x in xs]
  right_y = [center_offset + width * 0.5 + 0.5 * curvature * x * x for x in xs]
  return (
    SimpleNamespace(x=xs, y=[y - width for y in left_y]),
    SimpleNamespace(x=xs, y=left_y),
    SimpleNamespace(x=xs, y=right_y),
    SimpleNamespace(x=xs, y=[y + width for y in right_y]),
  )


def _run_with_geometry(path_curvature: float, lane_center_offset: float, frames: int = 80):
  p = LateralDemandPipeline(DT)
  lane_lines = _lane_lines(center_offset=lane_center_offset)
  for _ in range(frames):
    r = p.update(valid_inputs(
      v_ego=20.0,
      curvature=path_curvature,
      lane_centering_assist_enabled=True,
      lane_lines=lane_lines,
      lane_line_stds=[0.1, 0.1, 0.1, 0.1],
      lane_line_probs=[0.9, 0.9, 0.9, 0.9],
      smooth_model_path_curvature=False,
    ))
  return r


def test_lane_centering_geometry_path_right_negative_nudge():
  # Positive path curvature gives positive model y (right of lane center with y+ right convention).
  r = _run_with_geometry(path_curvature=0.005, lane_center_offset=0.0)
  assert r.debug["lane_centering_geometry_valid"] is True
  assert r.debug["lane_centering_geometry_mode"] is True
  assert r.debug["lane_centering_geometry_offset_near"] < -0.1
  assert r.demand.lane_centering_curvature_nudge < 0.0
  assert r.demand.lane_centering_assist_active is True


def test_lane_centering_geometry_path_left_positive_nudge():
  r = _run_with_geometry(path_curvature=-0.005, lane_center_offset=0.0)
  assert r.debug["lane_centering_geometry_valid"] is True
  assert r.debug["lane_centering_geometry_mode"] is True
  assert r.debug["lane_centering_geometry_offset_near"] > 0.1
  assert r.demand.lane_centering_curvature_nudge > 0.0
  assert r.demand.lane_centering_assist_active is True


def test_lane_centering_geometry_no_nudge_inside_leeway():
  # Tiny curvature keeps geometric offset inside the geometry deadband.
  r = _run_with_geometry(path_curvature=0.0002, lane_center_offset=0.0, frames=80)
  assert r.debug["lane_centering_geometry_valid"] is True
  assert r.debug["lane_centering_geometry_mode"] is True
  assert abs(r.debug["lane_centering_geometry_offset_near"]) < 0.06
  assert r.demand.lane_centering_curvature_nudge == pytest.approx(0.0, abs=1e-6)


def test_lane_centering_invalid_geometry_preserves_model_path_behavior():
  p = LateralDemandPipeline(DT)
  # No lane lines: geometry invalid, model-path LCA should still activate for a ramp.
  for _ in range(80):
    r = p.update(valid_inputs(
      v_ego=20.0,
      curvature=0.001,
      lane_centering_assist_enabled=True,
      lane_lines=(),
      lane_line_stds=[0.1, 0.1, 0.1, 0.1],
      lane_line_probs=[0.9, 0.9, 0.9, 0.9],
      smooth_model_path_curvature=False,
    ))
  assert r.debug["lane_centering_geometry_valid"] is False
  assert r.debug["lane_centering_geometry_mode"] is False
  assert r.demand.lane_centering_assist_active is True
  assert abs(r.demand.lane_centering_curvature_nudge) > 1e-5


def test_lane_centering_geometry_steady_off_center_nudge():
  # Parallel off-center path: near≈preview, no growth. Geometry mode should still
  # rate-limit a same-sign nudge under the shared cap.
  p = LateralDemandPipeline(DT)
  xs = [float(x) for x in range(N)]
  # Path is a constant +0.5 m offset (right of lane center, y+ right convention).
  # A constant offset gives the model path a slight heading, so also bend it back
  # gently to keep orientation_z small and avoid triggering high-curvature gating.
  path_y = [0.5 + 0.0001 * x for x in xs]
  yaw = [0.0001] * N
  lane_lines = _lane_lines(center_offset=0.0)
  for _ in range(80):
    r = p.update(valid_inputs(
      v_ego=20.0,
      curvature=0.0,
      position_x=xs,
      position_y=path_y,
      orientation_z=yaw,
      lane_centering_assist_enabled=True,
      lane_lines=lane_lines,
      lane_line_stds=[0.1, 0.1, 0.1, 0.1],
      lane_line_probs=[0.9, 0.9, 0.9, 0.9],
      smooth_model_path_curvature=False,
    ))
  assert r.debug["lane_centering_geometry_valid"] is True
  assert r.debug["lane_centering_geometry_mode"] is True
  assert r.debug["lane_centering_geometry_offset_near"] < -0.45
  assert r.demand.lane_centering_curvature_nudge < -1e-5
  assert r.demand.lane_centering_assist_active is True


def test_lane_centering_geometry_heading_cannot_flip_offset_correction():
  tracker = LaneCenteringAssistTracker()
  xs = [float(x) for x in range(N)]
  inputs = LaneCenteringAssistInputs(
    lat_active=True,
    v_ego=20.0,
    measured_curvature=0.0,
    model_curvature=0.0,
    previous_processed_curvature=0.0,
    path_quality=1.0,
    path_reason=LANE_CENTERING_ASSIST_OK_REASON,
    lane_change_shaping_active=False,
    lane_change_blend=0.0,
    curvature_limited=False,
    steering_pressed=False,
    left_blinker=False,
    right_blinker=False,
    position_x=xs,
    position_y=[0.5] * N,  # path right of lane center => geometry offset negative
    orientation_z=[0.2] * N,  # raw model heading points right and must not flip correction
    lane_line_probs=[0.9, 0.9, 0.9, 0.9],
    lane_lines=_lane_lines(center_offset=0.0),
    lane_line_stds=[0.1, 0.1, 0.1, 0.1],
  )
  for _ in range(80):
    result = tracker.update(inputs, DT)

  assert result.debug["lane_centering_geometry_mode"] is True
  assert result.debug["lane_centering_geometry_offset_near"] < -0.45
  assert result.heading_error == pytest.approx(0.0)
  assert result.curvature_nudge < 0.0


def test_lane_centering_geometry_growth_term_cannot_flip_offset_correction():
  tracker = LaneCenteringAssistTracker()
  xs = [float(x) for x in range(N)]
  # Geometry offset is positive at near and preview, but shrinks enough that the
  # growth term would make the raw nudge negative without the geometry sign veto.
  path_y = [-0.18 + 0.006 * x for x in xs]
  inputs = LaneCenteringAssistInputs(
    lat_active=True,
    v_ego=20.0,
    measured_curvature=0.0,
    model_curvature=0.0,
    previous_processed_curvature=0.0,
    path_quality=1.0,
    path_reason=LANE_CENTERING_ASSIST_OK_REASON,
    lane_change_shaping_active=False,
    lane_change_blend=0.0,
    curvature_limited=False,
    steering_pressed=False,
    left_blinker=False,
    right_blinker=False,
    position_x=xs,
    position_y=path_y,
    orientation_z=[0.0] * N,
    lane_line_probs=[0.9, 0.9, 0.9, 0.9],
    lane_lines=_lane_lines(center_offset=0.0),
    lane_line_stds=[0.1, 0.1, 0.1, 0.1],
  )
  for _ in range(80):
    result = tracker.update(inputs, DT)

  assert result.debug["lane_centering_geometry_mode"] is True
  assert result.debug["lane_centering_geometry_offset_near"] > 0.12
  assert result.curvature_nudge == pytest.approx(0.0, abs=1e-9)
  assert result.reason == "geometry_sign_veto"


def test_v1_backup_module_imports_and_matches_api():
  assert hasattr(model_path_processor_v1, "ModelPathProcessor")
  assert hasattr(model_path_processor_v1, "ModelPathProcessorInputs")
  assert hasattr(model_path_processor_v1, "ModelPathProcessorResult")
  v1 = model_path_processor_v1.ModelPathProcessor()
  result = v1.update(_mpp_inputs())
  assert math.isfinite(result.desired_curvature)
  assert model_path_processor_v1.MODEL_STALE_AGE_S == 0.20


def _assert_v1_v2_sequence_parity(sequence: list[ModelPathProcessorInputs]) -> None:
  p1 = model_path_processor_v1.ModelPathProcessor()
  p2 = ModelPathProcessor()
  fields = (
    "desired_curvature", "quality", "gated", "reason", "hold_frames_remaining",
    "smoothing_tau_s", "damping_alpha", "trust_penalty", "spatial_smoothed_curvature",
    "lane_change_fade", "straight_road_damping_active", "demand_jerk_smoothing_active",
    "demand_jerk_smoothing_step", "demand_jerk_smoothing_lag",
  )
  for frame, inputs in enumerate(sequence):
    kwargs = dict(vars(inputs))
    r1 = p1.update(model_path_processor_v1.ModelPathProcessorInputs(**kwargs))
    r2 = p2.update(ModelPathProcessorInputs(**kwargs))
    for field in fields:
      expected = getattr(r1, field)
      actual = getattr(r2, field)
      if isinstance(expected, float):
        assert actual == pytest.approx(expected, abs=1e-12), (frame, field)
      else:
        assert actual == expected, (frame, field)


def test_v2_matches_v1_characterization_sequences():
  low_conf = [0.9, 0.1, 0.1, 0.9]
  _assert_v1_v2_sequence_parity([
    *[_mpp_inputs(v_ego=20.0, desired_curvature=k, measured_curvature=k,
                  smooth_model_path_curvature=True)
      for k in (0.0, 0.0004, 0.0010, 0.0014)],
    _mpp_inputs(v_ego=20.0, desired_curvature=0.0014, measured_curvature=0.0014,
                smooth_model_path_curvature=True, lane_change_active=True),
  ])
  _assert_v1_v2_sequence_parity([
    *[_mpp_inputs(v_ego=10.0, desired_curvature=0.005, measured_curvature=0.0,
                  lane_line_probs=low_conf, smooth_model_path_curvature=True)
      for _ in range(5)],
    _mpp_inputs(v_ego=10.0, desired_curvature=0.005, measured_curvature=0.0,
                model_age_s=0.21),
    _mpp_inputs(v_ego=10.0, desired_curvature=0.005, measured_curvature=0.0,
                position_x=(), position_y=(), position_y_std=(), orientation_z=(), orientation_rate_z=()),
  ])
  _assert_v1_v2_sequence_parity([
    *[_mpp_inputs(v_ego=15.0, desired_curvature=k, measured_curvature=k,
                  smooth_model_path_curvature=True, demand_jerk_smoothing_enabled=True,
                  demand_jerk_smoothing_allowed=True)
      for k in ([0.003] * 8 + [0.0025, 0.0020, 0.0016, 0.0012, 0.0008, 0.0004])],
  ])


def test_model_path_processor_api_returns_finite():
  proc = ModelPathProcessor()
  result = proc.update(_mpp_inputs())
  assert isinstance(result, ModelPathProcessorResult)
  assert math.isfinite(result.desired_curvature)
  assert 0.0 <= result.quality <= 1.0
  assert result.reason in ("ok", "high_path_std", "low_lane_confidence", "frame_drop", "path_disagreement")


def test_model_path_processor_random_inputs_stay_finite():
  rng = np.random.default_rng(20260628)
  proc = ModelPathProcessor()
  for _ in range(200):
    k = float(rng.uniform(-0.02, 0.02))
    v = float(rng.uniform(0.0, 35.0))
    ystd = [float(rng.uniform(0.0, 2.0)) for _ in range(len(ModelConstants.T_IDXS))]
    lane_probs = [float(rng.uniform(0.0, 1.0)) for _ in range(4)]
    result = proc.update(_mpp_inputs(
      v_ego=v,
      desired_curvature=k,
      measured_curvature=k,
      previous_desired_curvature=k,
      position_y_std=ystd,
      lane_line_probs=lane_probs,
      smooth_model_path_curvature=bool(rng.integers(0, 2)),
      demand_jerk_smoothing_enabled=bool(rng.integers(0, 2)),
      lane_change_active=bool(rng.integers(0, 2)),
    ))
    assert math.isfinite(result.desired_curvature)
    assert abs(result.desired_curvature) <= 0.5
