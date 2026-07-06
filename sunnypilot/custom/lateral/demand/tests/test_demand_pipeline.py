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
  LANE_CENTERING_ASSIST_ESCALATION_ERR_BP,
  LANE_CENTERING_ASSIST_ESCALATION_MAX_LAT_ACCEL,
  LANE_CENTERING_ASSIST_MAX_LAT_ACCEL,
  LANE_CENTERING_ASSIST_OK_REASON,
  LANE_CENTERING_ASSIST_PATH_REASON_COOLDOWN_REASON,
  LaneCenteringAssistInputs,
  LaneCenteringAssistTracker,
  _max_nudge_curvature,
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
from openpilot.sunnypilot.custom.lateral.demand.pipeline import (
  LateralDemandPipeline,
  LateralDemandPipelineInputs,
)
from openpilot.sunnypilot.custom.lateral.demand.types import (
  DEMAND_SOURCE_FALLBACK_MEASURED,
  DEMAND_SOURCE_LANE_FIT,
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
    lat_delay=kwargs.get("lat_delay", 0.0),
    lateral_preview_assist_mode=kwargs.get("lateral_preview_assist_mode", "off"),
    straight_path_stabilization_mode=kwargs.get("straight_path_stabilization_mode", "off"),
    lane_rate_damping_mode=kwargs.get("lane_rate_damping_mode", "off"),
    lane_fit_source_mode=kwargs.get("lane_fit_source_mode", "off"),
    curvature_limited=kwargs.get("curvature_limited", False),
  )


def lane_rate_damping_inputs(lane_center_y0: float, *, lane_width: float = 3.6, lane_rate_damping_mode: str = "off", **kwargs):
  half_width = lane_width / 2.0
  return valid_inputs(
    left_lane_y0=lane_center_y0 - half_width,
    right_lane_y0=lane_center_y0 + half_width,
    lane_rate_damping_mode=lane_rate_damping_mode,
    **kwargs,
  )


def run_lane_rate_damping_sequence(mode: str, lane_centers: list[float], **kwargs) -> tuple[LateralDemandPipeline, object]:
  p = LateralDemandPipeline(DT)
  result = None
  for lane_center_y0 in lane_centers:
    result = p.update(lane_rate_damping_inputs(lane_center_y0, lane_rate_damping_mode=mode, **kwargs))
  assert result is not None
  return p, result


def _lane_lines(center_offset: float = 0.0, width: float = 4.0, curvature: float = 0.0):
  xs = [float(x) for x in range(N)]
  left_y = [center_offset - width * 0.5 + 0.5 * curvature * x * x for x in xs]
  right_y = [center_offset + width * 0.5 + 0.5 * curvature * x * x for x in xs]
  return [
    SimpleNamespace(x=xs, y=[y - width for y in left_y]),
    SimpleNamespace(x=xs, y=left_y),
    SimpleNamespace(x=xs, y=right_y),
    SimpleNamespace(x=xs, y=[y + width for y in right_y]),
  ]


def lane_fit_source_inputs(
  *,
  baseline_curvature: float = 0.0,
  lane_curvature: float = 0.0008,
  lane_fit_source_mode: str = "off",
  lane_center_offset: float = 0.0,
  lane_width: float = 4.0,
  lane_lines=None,
  lane_line_probs=None,
  lane_line_stds=None,
  **kwargs,
):
  params = dict(kwargs)
  v_ego = params.pop("v_ego", 20.0)
  lane_fit_source_mode = params.pop("lane_fit_source_mode", lane_fit_source_mode)
  lane_lines = params.pop("lane_lines", lane_lines)
  lane_line_probs = params.pop("lane_line_probs", lane_line_probs)
  lane_line_stds = params.pop("lane_line_stds", lane_line_stds)
  xs = [float(x) for x in range(N)]
  ys = [0.5 * baseline_curvature * x * x for x in xs]
  lane_lines = lane_lines if lane_lines is not None else _lane_lines(lane_center_offset, lane_width, lane_curvature)
  lane_line_probs = lane_line_probs if lane_line_probs is not None else [0.95, 0.95, 0.95, 0.95]
  lane_line_stds = lane_line_stds if lane_line_stds is not None else [0.05, 0.05, 0.05, 0.05]
  return valid_inputs(
    v_ego=v_ego,
    curvature=baseline_curvature,
    desired_curvature=baseline_curvature,
    measured_curvature=baseline_curvature,
    position_x=xs,
    position_y=ys,
    lane_lines=lane_lines,
    lane_line_probs=lane_line_probs,
    lane_line_stds=lane_line_stds,
    lane_fit_source_mode=lane_fit_source_mode,
    **params,
  )


def preview_assist_inputs(
  *,
  baseline_curvature: float = 0.0,
  preview_curvature: float = 0.001,
  v_ego: float = 20.0,
  lat_delay: float = 0.4,
  lateral_preview_assist_mode: str = "off",
  **kwargs,
):
  t_idxs = ModelConstants.T_IDXS
  n = len(t_idxs)
  return valid_inputs(
    v_ego=v_ego,
    curvature=baseline_curvature,
    desired_curvature=baseline_curvature,
    measured_curvature=baseline_curvature,
    orientation_z=kwargs.pop("orientation_z", [float(preview_curvature * v_ego * t) for t in t_idxs]),
    orientation_rate_z=kwargs.pop("orientation_rate_z", [float(preview_curvature * v_ego)] * n),
    lat_delay=lat_delay,
    lateral_preview_assist_mode=lateral_preview_assist_mode,
    **kwargs,
  )


def run_lane_fit_source_sequence(mode: str, *, frames: int = 25, **kwargs):
  p = LateralDemandPipeline(DT)
  result = None
  for _ in range(frames):
    result = p.update(lane_fit_source_inputs(lane_fit_source_mode=mode, **kwargs))
  assert result is not None
  return p, result


def run_lane_fit_source_until_active(mode: str, *, max_frames: int = 120, **kwargs):
  p = LateralDemandPipeline(DT)
  result = None
  for _ in range(max_frames):
    result = p.update(lane_fit_source_inputs(lane_fit_source_mode=mode, **kwargs))
    if bool(result.debug["lane_fit_source_active"]):
      return p, result
  assert result is not None
  return p, result


def run_lane_fit_source_until_applied(mode: str, *, max_frames: int = 120, **kwargs):
  p = LateralDemandPipeline(DT)
  result = None
  applied_frames = 0
  for _ in range(max_frames):
    result = p.update(lane_fit_source_inputs(lane_fit_source_mode=mode, **kwargs))
    if bool(result.debug["lane_fit_source_applied"]):
      applied_frames += 1
      if applied_frames >= 2:
        return p, result
  assert result is not None
  return p, result


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


def test_curvature_limited_passes_through_to_demand():
  p = LateralDemandPipeline(DT)
  r = p.update(valid_inputs(curvature=0.001, curvature_limited=True))
  assert r.demand.curvature_limited is True


def test_preview_assist_off_is_disabled():
  result = LateralDemandPipeline(DT).update(preview_assist_inputs(lateral_preview_assist_mode="off"))
  assert result.debug["lateral_preview_assist_mode"] == "off"
  assert result.debug["lateral_preview_assist_active"] is False
  assert result.debug["lateral_preview_assist_applied"] is False
  assert result.debug["lateral_preview_assist_reason"] == "disabled"


def test_preview_assist_shadow_logs_candidate_without_changing_curvature():
  off = LateralDemandPipeline(DT).update(preview_assist_inputs(lateral_preview_assist_mode="off"))
  result = LateralDemandPipeline(DT).update(preview_assist_inputs(lateral_preview_assist_mode="shadow"))

  assert result.debug["lateral_preview_assist_mode"] == "shadow"
  assert result.debug["lateral_preview_assist_active"] is True
  assert result.debug["lateral_preview_assist_applied"] is False
  assert result.debug["lateral_preview_assist_reason"] == "ok"
  assert abs(float(result.debug["lateral_preview_assist_curvature_nudge"])) > 0.0
  assert result.demand.processed_curvature == pytest.approx(off.demand.processed_curvature)


def test_preview_assist_apply_changes_curvature_and_is_slew_limited():
  off = LateralDemandPipeline(DT).update(preview_assist_inputs(lateral_preview_assist_mode="off"))
  result = LateralDemandPipeline(DT).update(preview_assist_inputs(lateral_preview_assist_mode="apply"))

  diff = result.demand.processed_curvature - off.demand.processed_curvature
  assert result.debug["lateral_preview_assist_mode"] == "apply"
  assert result.debug["lateral_preview_assist_active"] is True
  assert result.debug["lateral_preview_assist_applied"] is True
  assert result.debug["lateral_preview_assist_reason"] == "ok"
  assert result.debug["lateral_preview_assist_slew_limited"] is True
  assert diff == pytest.approx(float(result.debug["lateral_preview_assist_curvature_nudge"]), abs=1e-9)
  assert 0.0 < diff <= 0.05 / (20.0 ** 2) + 1e-9


@pytest.mark.parametrize("blocked_kwargs, expected_reason", [
  ({"lane_change_state_valid": False}, "lane_change_unknown"),
  ({"lane_change_state": 1}, "lane_change"),
  ({"left_blinker": True}, "blinker"),
  ({"steer_limited": True}, "steer_limited"),
  ({"curvature_limited": True}, "curvature_limited"),
  ({"model_age_s": 0.3}, "model_stale"),
])
def test_preview_assist_blocks_common_gates(blocked_kwargs, expected_reason):
  result = LateralDemandPipeline(DT).update(preview_assist_inputs(lateral_preview_assist_mode="apply", **blocked_kwargs))
  assert result.debug["lateral_preview_assist_active"] is False
  assert result.debug["lateral_preview_assist_applied"] is False
  assert result.debug["lateral_preview_assist_reason"] == expected_reason
  assert result.debug["lateral_preview_assist_curvature_nudge"] == 0.0


def test_preview_assist_blocks_sign_conflict():
  result = LateralDemandPipeline(DT).update(preview_assist_inputs(
    baseline_curvature=0.001,
    preview_curvature=-0.001,
    lateral_preview_assist_mode="apply",
  ))

  assert result.debug["lateral_preview_assist_active"] is False
  assert result.debug["lateral_preview_assist_applied"] is False
  assert result.debug["lateral_preview_assist_reason"] == "sign_conflict"
  assert result.debug["lateral_preview_assist_curvature_nudge"] == 0.0


def test_preview_assist_blocks_lane_fit_source_and_maneuver_override():
  _, lane_fit_result = run_lane_fit_source_until_applied("apply", lateral_preview_assist_mode="apply")
  maneuver_result = LateralDemandPipeline(DT).update(preview_assist_inputs(
    lateral_preview_assist_mode="apply",
    lateral_maneuver_curvature=0.02,
  ))

  assert lane_fit_result.debug["lateral_preview_assist_active"] is False
  assert lane_fit_result.debug["lateral_preview_assist_applied"] is False
  assert lane_fit_result.debug["lateral_preview_assist_reason"] == "lane_fit_source"
  assert maneuver_result.debug["lateral_preview_assist_active"] is False
  assert maneuver_result.debug["lateral_preview_assist_applied"] is False
  assert maneuver_result.debug["lateral_preview_assist_reason"] == "maneuver_override"


def test_preview_assist_blocks_straight_path_stabilization():
  pipeline = LateralDemandPipeline(DT)
  result = None
  for _ in range(40):
    result = pipeline.update(preview_assist_inputs(
      baseline_curvature=0.00005,
      preview_curvature=0.0002,
      lateral_preview_assist_mode="apply",
      straight_path_stabilization_mode="apply",
    ))

  assert result is not None
  assert result.debug["straight_path_stabilization_active"] is True
  assert result.debug["lateral_preview_assist_active"] is False
  assert result.debug["lateral_preview_assist_applied"] is False
  assert result.debug["lateral_preview_assist_reason"] == "straight_path_stabilization"


def test_preview_assist_resets_after_block():
  warm = LateralDemandPipeline(DT)
  for _ in range(5):
    warm.update(preview_assist_inputs(lateral_preview_assist_mode="apply"))

  blocked = warm.update(preview_assist_inputs(lateral_preview_assist_mode="apply", steering_pressed=True))
  assert blocked.debug["lateral_preview_assist_active"] is False
  assert blocked.debug["lateral_preview_assist_reason"] == "driver_override"

  fresh = LateralDemandPipeline(DT).update(preview_assist_inputs(lateral_preview_assist_mode="apply"))
  resumed = warm.update(preview_assist_inputs(lateral_preview_assist_mode="apply"))
  assert resumed.debug["lateral_preview_assist_curvature_nudge"] == pytest.approx(
    fresh.debug["lateral_preview_assist_curvature_nudge"], abs=1e-12
  )


def test_preview_assist_resets_when_switching_to_apply():
  warm = LateralDemandPipeline(DT)
  for _ in range(5):
    warm.update(preview_assist_inputs(lateral_preview_assist_mode="shadow"))

  fresh = LateralDemandPipeline(DT).update(preview_assist_inputs(lateral_preview_assist_mode="apply"))
  switched = warm.update(preview_assist_inputs(lateral_preview_assist_mode="apply"))

  assert switched.debug["lateral_preview_assist_curvature_nudge"] == pytest.approx(
    fresh.debug["lateral_preview_assist_curvature_nudge"], abs=1e-12
  )


def test_lane_rate_damping_shadow_logs_candidate_without_changing_curvature():
  centers = [i * 0.01 for i in range(60)]
  shadow = LateralDemandPipeline(DT)
  off = LateralDemandPipeline(DT)
  shadow_result = None
  off_result = None
  for lane_center_y0 in centers:
    shadow_result = shadow.update(lane_rate_damping_inputs(lane_center_y0, lane_rate_damping_mode="shadow"))
    off_result = off.update(lane_rate_damping_inputs(lane_center_y0, lane_rate_damping_mode="off"))

  assert shadow_result is not None
  assert off_result is not None
  assert shadow_result.debug["lane_rate_damping_mode"] == "shadow"
  assert shadow_result.debug["lane_rate_damping_active"] is True
  assert shadow_result.debug["lane_rate_damping_applied"] is False
  assert float(shadow_result.debug["lane_rate_damping_lat_accel"]) > 0.0
  assert shadow_result.demand.processed_curvature == pytest.approx(off_result.demand.processed_curvature)


def test_lane_rate_damping_off_is_disabled_without_lane_lines():
  result = LateralDemandPipeline(DT).update(valid_inputs(lane_rate_damping_mode="off", left_lane_y0=None, right_lane_y0=None))
  assert result.debug["lane_rate_damping_mode"] == "off"
  assert result.debug["lane_rate_damping_reason"] == "disabled"
  assert result.debug["lane_rate_damping_active"] is False
  assert result.debug["lane_rate_damping_curvature"] == 0.0


def test_lane_rate_damping_apply_changes_curvature_and_caps_lat_accel():
  centers = [i * 0.01 for i in range(60)]
  shadow = LateralDemandPipeline(DT)
  apply = LateralDemandPipeline(DT)
  shadow_result = None
  apply_result = None
  for lane_center_y0 in centers:
    shadow_result = shadow.update(lane_rate_damping_inputs(lane_center_y0, lane_rate_damping_mode="shadow"))
    apply_result = apply.update(lane_rate_damping_inputs(lane_center_y0, lane_rate_damping_mode="apply"))

  assert shadow_result is not None
  assert apply_result is not None
  assert apply_result.debug["lane_rate_damping_mode"] == "apply"
  assert apply_result.debug["lane_rate_damping_active"] is True
  assert apply_result.debug["lane_rate_damping_applied"] is True
  assert apply_result.debug["lane_rate_damping_reason"] == "ok"
  assert float(apply_result.debug["lane_rate_damping_lat_accel"]) == pytest.approx(0.05, abs=1e-6)
  assert float(apply_result.debug["lane_rate_damping_curvature"]) == pytest.approx(0.05 / (20.0 ** 2), abs=1e-6)
  assert apply_result.demand.processed_curvature == pytest.approx(
    shadow_result.demand.processed_curvature + float(apply_result.debug["lane_rate_damping_curvature"]),
    abs=1e-9,
  )
  assert apply_result.demand.processed_curvature > shadow_result.demand.processed_curvature


@pytest.mark.parametrize("blocked_kwargs, expected_reason", [
  ({"steering_pressed": True}, "driver_override"),
  ({"curvature_limited": True}, "curvature_limited"),
  ({"steer_limited": True}, "steer_limited"),
  ({"lane_change_state": 1}, "lane_change"),
  ({"model_age_s": 0.3}, "model_stale"),
])
def test_lane_rate_damping_blocks_and_resets_history(blocked_kwargs, expected_reason):
  p = LateralDemandPipeline(DT)
  for lane_center_y0 in [i * 0.01 for i in range(60)]:
    p.update(lane_rate_damping_inputs(lane_center_y0, lane_rate_damping_mode="shadow"))

  blocked = p.update(lane_rate_damping_inputs(0.60, lane_rate_damping_mode="shadow", **blocked_kwargs))
  assert blocked.debug["lane_rate_damping_active"] is False
  assert blocked.debug["lane_rate_damping_applied"] is False
  assert blocked.debug["lane_rate_damping_curvature"] == 0.0
  assert blocked.debug["lane_rate_damping_reason"] == expected_reason

  resumed = p.update(lane_rate_damping_inputs(0.61, lane_rate_damping_mode="shadow"))
  assert resumed.debug["lane_rate_damping_reason"] == "warming_up"
  assert resumed.debug["lane_rate_damping_active"] is False
  assert resumed.debug["lane_rate_damping_curvature"] == 0.0


def test_lane_fit_source_shadow_logs_candidate_without_changing_curvature():
  _, off_result = run_lane_fit_source_sequence("off", frames=120)
  _, shadow_result = run_lane_fit_source_sequence("shadow", frames=120)

  assert shadow_result.debug["lane_fit_source_mode"] == "shadow"
  assert shadow_result.debug["lane_fit_source_active"] is True
  assert shadow_result.debug["lane_fit_source_applied"] is False
  assert shadow_result.debug["lane_fit_source_reason"] == "ok"
  assert shadow_result.debug["lane_fit_source_candidate_curvature"] == pytest.approx(0.0008, abs=1e-6)
  assert shadow_result.demand.processed_curvature == pytest.approx(off_result.demand.processed_curvature, abs=1e-9)


@pytest.mark.parametrize("mode", ["shadow", "off"])
def test_lane_fit_source_apply_then_shadow_or_off_returns_exact_baseline(mode):
  p, _ = run_lane_fit_source_until_applied("apply")
  result = p.update(lane_fit_source_inputs(lane_fit_source_mode=mode))

  assert result.demand.processed_curvature == pytest.approx(0.0, abs=1e-12)
  assert result.debug["lane_fit_source_applied_curvature"] == pytest.approx(0.0, abs=1e-12)
  assert result.debug["lane_fit_source_slew_limited"] is False
  if mode == "shadow":
    assert result.debug["lane_fit_source_active"] is True
    assert result.debug["lane_fit_source_applied"] is False
    assert result.debug["lane_fit_source_reason"] == "ok"
  else:
    assert result.debug["lane_fit_source_active"] is False
    assert result.debug["lane_fit_source_applied"] is False
    assert result.debug["lane_fit_source_reason"] == "disabled"


def test_lane_fit_source_apply_moves_curvature_after_persistence():
  _, result = run_lane_fit_source_until_applied("apply")

  assert result.debug["lane_fit_source_mode"] == "apply"
  assert result.debug["lane_fit_source_active"] is True
  assert result.debug["lane_fit_source_applied"] is True
  assert result.debug["lane_fit_source_reason"] == "ok"
  assert result.debug["lane_fit_source_slew_limited"] is True
  assert result.debug["lane_fit_source_candidate_curvature"] == pytest.approx(0.0008, abs=1e-6)
  assert 0.0 < float(result.debug["lane_fit_source_applied_curvature"]) < float(result.debug["lane_fit_source_candidate_curvature"])


@pytest.mark.parametrize("blocked_kwargs, expected_reason", [
  ({"lane_lines": ()}, "missing_lanes"),
  ({"lane_line_probs": [0.95, 0.4, 0.4, 0.95]}, "low_prob"),
  ({"lane_change_state": 1, "lane_change_state_valid": True}, "lane_change"),
  ({"steer_limited": True}, "steer_limited"),
])
def test_lane_fit_source_blocks_and_slews_release(blocked_kwargs, expected_reason):
  p, applied = run_lane_fit_source_until_applied("apply")
  prev = float(applied.demand.processed_curvature)

  blocked = p.update(lane_fit_source_inputs(lane_fit_source_mode="apply", **blocked_kwargs))

  assert blocked.debug["lane_fit_source_active"] is False
  assert blocked.debug["lane_fit_source_applied"] is False
  assert blocked.debug["lane_fit_source_reason"] == expected_reason
  assert blocked.debug["lane_fit_source_slew_limited"] is True
  assert 0.0 < float(blocked.demand.processed_curvature) < prev


def test_lane_fit_source_sign_conflict_blocks():
  p, applied = run_lane_fit_source_until_applied(
    "apply",
    baseline_curvature=0.00025,
    lane_curvature=0.0002,
  )
  blocked = p.update(lane_fit_source_inputs(
    lane_fit_source_mode="apply",
    baseline_curvature=0.00025,
    lane_curvature=-0.00025,
  ))

  assert blocked.debug["lane_fit_source_active"] is False
  assert blocked.debug["lane_fit_source_applied"] is False
  assert blocked.debug["lane_fit_source_reason"] == "sign_conflict"
  assert blocked.debug["lane_fit_source_slew_limited"] is True
  assert float(blocked.demand.processed_curvature) > float(applied.demand.processed_curvature)
  assert float(blocked.demand.processed_curvature) < 0.00025


def test_lane_fit_source_apply_keeps_lca_running_and_marks_final_source():
  _, result = run_lane_fit_source_sequence("apply", frames=80, lane_centering_assist_enabled=True)

  assert result.debug["lane_fit_source_applied"] is True
  assert result.demand.demand_source == DEMAND_SOURCE_LANE_FIT
  assert result.demand.lane_centering_assist_active is True


def test_lane_fit_source_apply_does_not_override_maneuver_curvature():
  p, _ = run_lane_fit_source_until_applied("apply")
  maneuver = p.update(valid_inputs(curvature=0.001, lateral_maneuver_curvature=0.037))

  assert maneuver.demand.demand_source == DEMAND_SOURCE_LATERAL_MANEUVER
  assert maneuver.demand.processed_curvature == pytest.approx(0.037, abs=1e-12)
  assert maneuver.debug["lane_fit_source_slew_limited"] is False


def test_extreme_curvature_warning_logs_once_per_transition(monkeypatch):
  warnings: list[str] = []
  monkeypatch.setattr(
    "openpilot.sunnypilot.custom.lateral.demand.pipeline.cloudlog.warning",
    lambda msg: warnings.append(str(msg)),
  )

  p = LateralDemandPipeline(DT)
  for _ in range(2):
    p.update(valid_inputs(curvature=0.001, lateral_maneuver_curvature=0.06))

  assert sum("extreme processed curvature" in msg for msg in warnings) == 1


def test_lane_y0_reaches_debug_dtle():
  p = LateralDemandPipeline(DT)
  r = p.update(valid_inputs(curvature=0.001, left_lane_y0=1.8, right_lane_y0=-1.8))
  dtle = float(r.debug["dtle_estimate"])
  assert math.isfinite(dtle)
  assert abs(dtle) < 0.01


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
  assert max(abs(b - a) for a, b in zip(outputs, outputs[1:], strict=False)) < 0.00055


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


def test_low_lane_confidence_near_straight_demand_smoothing_reduces_wobble():
  p = LateralDemandPipeline(DT)
  v_ego = 15.0
  raw_values = [0.0] * 3 + [0.0012, -0.0012] * 12
  processed_values = []
  smoothing_active = False
  reasons = set()

  for raw in raw_values:
    r = p.update(valid_inputs(
      v_ego=v_ego,
      curvature=raw,
      measured_curvature=0.0,
      lane_line_probs=[0.9, 0.2, 0.2, 0.9],
      smooth_model_path_curvature=True,
      demand_jerk_smoothing_enabled=True,
      steering_pressed=False,
    ))
    processed_values.append(float(r.demand.processed_curvature))
    smoothing_active = smoothing_active or bool(r.debug["demand_jerk_smoothing_active"])
    reasons.add(str(r.debug["model_path_reason"]))

  raw_pp = max(raw_values) - min(raw_values)
  processed_pp = max(processed_values[3:]) - min(processed_values[3:])

  assert "low_lane_confidence" in reasons
  assert smoothing_active is True
  assert processed_pp < raw_pp


def test_low_lane_confidence_demand_smoothing_does_not_smooth_larger_curve():
  proc = ModelPathProcessor()
  inputs = _mpp_inputs(v_ego=15.0, lane_line_probs=[0.9, 0.2, 0.2, 0.9])

  candidate, active, _, _ = proc._apply_demand_jerk_smoothing(
    inputs, raw_base=0.0030, target=0.0030, quality=0.90, reason="low_lane_confidence", path_disagreement=0.0,
  )

  assert active is False
  assert candidate == pytest.approx(0.0030)


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
        result = ModelPathProcessor()._smoothed_path_curvature(
          spatial_smoothing_inputs(v_ego, desired_curvature, candidate_curvature),
          desired_curvature,
          quality,
          trust_penalty,
        )

        assert result is not None
        correction_lat_accel = (result - desired_curvature) * v_ego * v_ego
        assert math.copysign(1.0, correction_lat_accel) == math.copysign(1.0, candidate_delta_lat_accel)
        assert abs(correction_lat_accel) <= min(abs(candidate_delta_lat_accel), max_delta_lat_accel)

  assert ModelPathProcessor()._smoothed_path_curvature(
    spatial_smoothing_inputs(v_ego, desired_curvature, candidate_curvature),
    desired_curvature,
    LOW_QUALITY_BLEND_THRESHOLD,
    0.0,
  ) is None
  assert ModelPathProcessor()._smoothed_path_curvature(
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
    result = ModelPathProcessor()._smoothed_path_curvature(
      spatial_smoothing_inputs(v_ego, desired_curvature, candidate_curvature),
      desired_curvature,
      1.0,
      trust_penalty,
    )
    assert result is not None
    corrections.append((result - desired_curvature) * v_ego * v_ego)

  assert all(earlier > later for earlier, later in zip(corrections, corrections[1:], strict=False))


def test_spatial_smoothing_near_zero_scale_is_monotonic():
  v_ego = 20.0
  curvature_values = [0.0, NEAR_ZERO_CURVATURE_BP[1] / 2.0, NEAR_ZERO_CURVATURE_BP[1], 0.001]
  corrections = []

  for desired_curvature in curvature_values:
    candidate_curvature = desired_curvature + 0.001
    result = ModelPathProcessor()._smoothed_path_curvature(
      spatial_smoothing_inputs(v_ego, desired_curvature, candidate_curvature),
      desired_curvature,
      1.0,
      0.0,
    )
    assert result is not None
    corrections.append((result - desired_curvature) * v_ego * v_ego)

  assert all(earlier <= later for earlier, later in zip(corrections, corrections[1:], strict=False))
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

  bad = make_inputs("high_path_std")
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


def test_lane_centering_assist_runs_under_low_lane_confidence():
  # Route 0000025e: `low_lane_confidence` held 81.6% of active time and hard-blocked
  # LCA in exactly the one-weak-line regime one-line centering exists for. It must be
  # treated like "ok"; per-line prob gating lives in _confidence()/geometry validity.
  tracker = LaneCenteringAssistTracker()
  xs = [float(x) for x in range(N)]
  ys = [0.01 * x for x in xs]
  yaws = [0.0] * N

  inputs = LaneCenteringAssistInputs(
    lat_active=True,
    v_ego=20.0,
    measured_curvature=0.0,
    model_curvature=0.0,
    previous_processed_curvature=0.0,
    path_quality=1.0,
    path_reason="low_lane_confidence",
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
  r = tracker.update(inputs, DT)
  assert r.active is True
  assert r.reason == "growing_lateral_error"
  assert r.curvature_nudge > 0.0


def test_lane_centering_nudge_cap_escalates_with_predicted_drift():
  # Route 0000025e t=364-376s: the flat 0.08 m/s^2 cap saturated while predicted drift
  # grew to 0.7 m and the driver overrode 1.0 m from the line.
  v_ego = 11.0  # city speed of the observed drift
  speed_sq = v_ego * v_ego
  base = _max_nudge_curvature(v_ego) * speed_sq
  assert base == pytest.approx(LANE_CENTERING_ASSIST_MAX_LAT_ACCEL)

  # Small drift: unchanged.
  small = _max_nudge_curvature(v_ego, predicted_lateral_error=LANE_CENTERING_ASSIST_ESCALATION_ERR_BP[0]) * speed_sq
  assert small == pytest.approx(base)

  # Full escalation at the upper breakpoint, symmetric in sign.
  for err in (LANE_CENTERING_ASSIST_ESCALATION_ERR_BP[1], -LANE_CENTERING_ASSIST_ESCALATION_ERR_BP[1]):
    full = _max_nudge_curvature(v_ego, predicted_lateral_error=err) * speed_sq
    assert full == pytest.approx(LANE_CENTERING_ASSIST_ESCALATION_MAX_LAT_ACCEL)

  # Straight-cruise damping still applies at small errors and escalates the same way.
  v_hwy = 30.0
  damped = _max_nudge_curvature(v_hwy, straight_cruise=True) * v_hwy * v_hwy
  assert damped < LANE_CENTERING_ASSIST_MAX_LAT_ACCEL
  escalated = _max_nudge_curvature(v_hwy, straight_cruise=True,
                                   predicted_lateral_error=LANE_CENTERING_ASSIST_ESCALATION_ERR_BP[1]) * v_hwy * v_hwy
  assert escalated == pytest.approx(LANE_CENTERING_ASSIST_ESCALATION_MAX_LAT_ACCEL)


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


def test_psi_dot_fit_is_cached_per_model_frame():
  import dataclasses
  v_ego = 20.0
  proc = ModelPathProcessor()

  inputs_a = dataclasses.replace(spatial_smoothing_inputs(v_ego, 0.001, 0.002), model_frame_id=991)
  psi_a = proc._psi_dot_at_action(inputs_a, v_ego)
  assert psi_a is not None

  # same frame id: the cached fit is reused even if the arrays were to change
  changed = dataclasses.replace(spatial_smoothing_inputs(v_ego, 0.001, 0.004), model_frame_id=991)
  assert proc._psi_dot_at_action(changed, v_ego) == psi_a

  # new frame id: recomputed from the new arrays
  psi_b = proc._psi_dot_at_action(dataclasses.replace(changed, model_frame_id=992), v_ego)
  assert psi_b is not None and psi_b != psi_a

  # frame id 0 (tests/replays) bypasses the cache
  fresh = ModelPathProcessor()
  psi_bypass = fresh._psi_dot_at_action(dataclasses.replace(changed, model_frame_id=0), v_ego)
  assert psi_bypass == pytest.approx(psi_b)
