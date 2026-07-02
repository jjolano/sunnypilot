"""Focused tests for ModelPathProcessor straight-road damping behavior."""
from __future__ import annotations

import math

import numpy as np
import pytest

from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.sunnypilot.custom.lateral.demand.model_path_processor import (
  DAMPING_TAU_SPEED_BP,
  DAMPING_TAU_S,
  DT_CTRL,
  ModelPathProcessor,
  ModelPathProcessorInputs,
  SPS_ANCHOR_CLIP_LAT_ACCEL,
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


# ---------- straight-path stabilization ----------


def _sps_inputs(
  v_ego: float = 20.0,
  curvature: float = 0.0,
  previous_desired_curvature: float = 0.0,
  measured_curvature: float = 0.0,
  mode: str = "off",
  steering_pressed: bool | None = False,
  lane_change_active: bool = False,
  left_blinker: bool = False,
  right_blinker: bool = False,
  steer_limited: bool = False,
  y_std: float = 0.05,
  lane_probs: list[float] | None = None,
  frame_drop_perc: float = 0.0,
  model_age_s: float = 0.0,
  smooth_model_path_curvature: bool = False,
) -> ModelPathProcessorInputs:
  xs = [float(i) for i in range(N)]
  ys = [0.5 * curvature * x * x for x in xs]
  y_stds = [y_std] * N
  yaws = [curvature * x for x in xs]
  yaw_rates = [curvature * v_ego] * N
  lane_probs = lane_probs if lane_probs is not None else [0.9, 0.9, 0.9, 0.9]

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
    left_blinker=left_blinker,
    right_blinker=right_blinker,
    steering_pressed=steering_pressed,
    steer_limited=steer_limited,
    straight_path_stabilization_mode=mode,
    frame_drop_perc=frame_drop_perc,
    model_age_s=model_age_s,
  )


def _run_frames(proc: ModelPathProcessor, frames: list[ModelPathProcessorInputs]) -> list:
  results = []
  prev = 0.0
  for inp in frames:
    inp = inp if inp.previous_desired_curvature != 0.0 else inp.__class__(**{**inp.__dict__, "previous_desired_curvature": prev})
    result = proc.update(inp)
    results.append(result)
    prev = result.desired_curvature
  return results


def test_sps_unknown_mode_is_disabled():
  proc = ModelPathProcessor()
  inp = _sps_inputs(curvature=0.0002, mode="banana")
  result = proc.update(inp)

  assert result.straight_path_stabilization_mode == "off"
  assert result.straight_path_stabilization_active is False
  assert result.straight_path_stabilization_applied is False
  assert result.straight_path_stabilization_reason == "disabled"
  assert result.desired_curvature == pytest.approx(0.0002, abs=1e-9)


def test_sps_shadow_computes_candidate_but_unchanged():
  proc = ModelPathProcessor()
  v_ego = 20.0
  speed_sq = v_ego * v_ego
  k = 0.0003  # lat accel 0.12 m/s^2

  # Fill anchor buffer with stable near-straight driving.
  for _ in range(55):
    proc.update(_sps_inputs(v_ego=v_ego, curvature=k, mode="shadow"))

  result = proc.update(_sps_inputs(v_ego=v_ego, curvature=k, mode="shadow"))

  assert result.straight_path_stabilization_active is True
  assert result.straight_path_stabilization_applied is False
  assert result.straight_path_stabilization_reason == "ok"
  assert result.straight_path_stabilization_candidate_curvature != pytest.approx(0.0, abs=1e-9)
  # Shadow mode leaves the processed curvature unchanged.
  assert result.desired_curvature == pytest.approx(k, abs=1e-9)
  assert abs(
    result.straight_path_stabilization_candidate_curvature * speed_sq - result.straight_path_stabilization_anchor_lat_accel
  ) <= SPS_ANCHOR_CLIP_LAT_ACCEL + 1e-6


def test_sps_apply_uses_candidate_when_active():
  proc = ModelPathProcessor()
  v_ego = 20.0
  speed_sq = v_ego * v_ego
  k = 0.0003  # lat accel 0.12 m/s^2

  # Fill anchor buffer with stable near-straight driving.
  for _ in range(55):
    proc.update(_sps_inputs(v_ego=v_ego, curvature=k, mode="apply"))

  result = proc.update(_sps_inputs(v_ego=v_ego, curvature=k, mode="apply"))

  assert result.straight_path_stabilization_active is True
  assert result.straight_path_stabilization_applied is True
  assert result.straight_path_stabilization_reason == "ok"
  assert result.desired_curvature == pytest.approx(k, abs=1e-9)
  assert abs(
    result.straight_path_stabilization_candidate_curvature * speed_sq - result.straight_path_stabilization_anchor_lat_accel
  ) <= SPS_ANCHOR_CLIP_LAT_ACCEL + 1e-6


def test_sps_apply_waits_for_sustained_clean_path():
  proc = ModelPathProcessor()
  v_ego = 20.0
  k = 0.0003

  result = proc.update(_sps_inputs(v_ego=v_ego, curvature=k, mode="apply"))

  assert result.straight_path_stabilization_active is False
  assert result.straight_path_stabilization_applied is False
  assert result.straight_path_stabilization_reason == "warming"
  assert result.desired_curvature == pytest.approx(k, abs=1e-9)


def test_sps_apply_reduces_slow_near_straight_wiggle():
  """A slow near-straight sinusoid stays inside jerk gates and the stabilizer
  keeps the output bounded around the rolling anchor."""
  proc = ModelPathProcessor()
  v_ego = 20.0
  speed_sq = v_ego * v_ego
  amplitude = 0.10  # m/s^2
  period_s = 3.0
  dt = DT_CTRL

  n_frames = int(period_s / dt)
  active_count = 0
  max_out_lat_accel = 0.0
  for i in range(n_frames):
    a_raw = amplitude * math.sin(2.0 * math.pi * i * dt / period_s)
    k = a_raw / speed_sq
    result = proc.update(_sps_inputs(v_ego=v_ego, curvature=k, mode="apply"))
    if result.straight_path_stabilization_active:
      active_count += 1
    max_out_lat_accel = max(max_out_lat_accel, abs(result.desired_curvature * speed_sq))

  assert active_count > n_frames // 2
  # Clipped deviation from anchor is bounded; allow small overshoot for the
  # first frame before the anchor is populated.
  assert max_out_lat_accel <= amplitude + SPS_ANCHOR_CLIP_LAT_ACCEL + 0.02


def test_sps_preserves_anchor_through_transient_steer_limit():
  proc = ModelPathProcessor()
  v_ego = 20.0
  k = 0.0002

  for _ in range(55):
    proc.update(_sps_inputs(v_ego=v_ego, curvature=k, mode="apply"))
  assert proc.update(_sps_inputs(v_ego=v_ego, curvature=k, mode="apply")).straight_path_stabilization_active is True

  limited = proc.update(_sps_inputs(v_ego=v_ego, curvature=k, mode="apply", steer_limited=True))
  assert limited.straight_path_stabilization_active is False
  assert limited.straight_path_stabilization_reason == "gate_steer_limited"

  clean = proc.update(_sps_inputs(v_ego=v_ego, curvature=k, mode="apply"))
  assert clean.straight_path_stabilization_active is True
  assert clean.straight_path_stabilization_reason == "ok"


def test_sps_anchors_first_near_straight_redetect_after_lane_dropout():
  proc = ModelPathProcessor()
  v_ego = 20.0
  speed_sq = v_ego * v_ego
  straight_k = 0.0001  # 0.04 m/s^2
  redetect_k = -0.0006  # -0.24 m/s^2: near-straight but a visible jump

  for _ in range(55):
    proc.update(_sps_inputs(v_ego=v_ego, curvature=straight_k, mode="apply"))
  active = proc.update(_sps_inputs(v_ego=v_ego, curvature=straight_k, mode="apply"))
  assert active.straight_path_stabilization_active is True

  for _ in range(10):
    dropout = proc.update(_sps_inputs(v_ego=v_ego, curvature=straight_k, mode="apply", lane_probs=[0.9, 0.0, 0.0, 0.9]))
    assert dropout.reason == "low_lane_confidence"

  redetect = proc.update(_sps_inputs(v_ego=v_ego, curvature=redetect_k, mode="apply"))
  assert redetect.straight_path_stabilization_active is True
  assert redetect.straight_path_stabilization_reason == "ok"
  assert abs(redetect.desired_curvature * speed_sq) < abs(redetect_k * speed_sq)


def test_sps_releases_on_curve_entry():
  proc = ModelPathProcessor()
  v_ego = 20.0
  speed_sq = v_ego * v_ego
  straight_k = 0.0001  # lat accel 0.04
  curve_k = 0.0015  # lat accel 0.60 -> exceeds 0.50 release

  # Stabilize near straight.
  for _ in range(55):
    proc.update(_sps_inputs(v_ego=v_ego, curvature=straight_k, mode="apply"))

  active_result = proc.update(_sps_inputs(v_ego=v_ego, curvature=straight_k, mode="apply"))
  assert active_result.straight_path_stabilization_active is True

  released = proc.update(_sps_inputs(v_ego=v_ego, curvature=curve_k, mode="apply"))
  assert released.straight_path_stabilization_active is False
  assert "release" in released.straight_path_stabilization_reason
  assert abs(curve_k * speed_sq) > 0.50


def test_sps_releases_on_high_jerk():
  proc = ModelPathProcessor()
  v_ego = 20.0

  for _ in range(10):
    proc.update(_sps_inputs(v_ego=v_ego, curvature=0.0, mode="apply"))

  small_k = 0.00005  # lat accel 0.02
  released = proc.update(_sps_inputs(v_ego=v_ego, curvature=small_k, mode="apply"))
  # 0.02 m/s^2 in 0.01s -> 2.0 m/s^3 > 1.0 release threshold
  assert released.straight_path_stabilization_active is False
  assert released.straight_path_stabilization_reason == "release_high_jerk"


def test_sps_releases_on_steering_pressed():
  proc = ModelPathProcessor()
  v_ego = 20.0
  k = 0.0001

  for _ in range(10):
    proc.update(_sps_inputs(v_ego=v_ego, curvature=k, mode="apply"))

  released = proc.update(_sps_inputs(v_ego=v_ego, curvature=k, mode="apply", steering_pressed=True))
  assert released.straight_path_stabilization_active is False
  assert released.straight_path_stabilization_reason == "gate_steering_pressed"


def test_sps_releases_on_lane_change():
  proc = ModelPathProcessor()
  v_ego = 20.0
  k = 0.0001

  for _ in range(10):
    proc.update(_sps_inputs(v_ego=v_ego, curvature=k, mode="apply"))

  released = proc.update(_sps_inputs(v_ego=v_ego, curvature=k, mode="apply", lane_change_active=True))
  assert released.straight_path_stabilization_active is False
  assert released.straight_path_stabilization_reason == "gate_lane_change"


def test_sps_fallback_resets_anchor_before_next_apply():
  proc = ModelPathProcessor()
  v_ego = 20.0
  k = 0.0001

  for _ in range(55):
    proc.update(_sps_inputs(v_ego=v_ego, curvature=k, mode="apply"))
  assert proc.update(_sps_inputs(v_ego=v_ego, curvature=k, mode="apply")).straight_path_stabilization_active is True

  stale = proc.update(_sps_inputs(v_ego=v_ego, curvature=k, mode="apply", model_age_s=1.0))
  assert stale.gated is True
  assert stale.reason == "model_stale"

  next_clean = proc.update(_sps_inputs(v_ego=v_ego, curvature=k, mode="apply"))
  assert next_clean.straight_path_stabilization_active is False
  assert next_clean.straight_path_stabilization_applied is False
  assert next_clean.straight_path_stabilization_reason == "warming"


def test_sps_requires_path_curvature_evidence():
  proc = ModelPathProcessor()
  v_ego = 20.0
  k = 0.0001
  missing_orientation = _sps_inputs(v_ego=v_ego, curvature=k, mode="apply")
  missing_orientation = missing_orientation.__class__(**{
    **missing_orientation.__dict__,
    "orientation_z": (),
    "orientation_rate_z": (),
  })

  result = proc.update(missing_orientation)
  for _ in range(54):
    result = proc.update(missing_orientation)

  assert result.straight_path_stabilization_active is False
  assert result.straight_path_stabilization_applied is False
  assert result.straight_path_stabilization_reason == "gate_path_evidence"


def test_sps_no_sign_flip_except_tiny_raw():
  proc = ModelPathProcessor()
  v_ego = 20.0

  # Tiny raw (< 0.05 m/s^2 lat accel) is allowed to cross zero.
  tiny_k = 0.0001  # lat accel 0.04
  for _ in range(10):
    proc.update(_sps_inputs(v_ego=v_ego, curvature=tiny_k, mode="apply"))
  tiny_neg = -0.0001
  result = proc.update(_sps_inputs(v_ego=v_ego, curvature=tiny_neg, mode="apply"))
  assert result.desired_curvature * tiny_neg >= 0.0

  # Larger raw must keep the same sign as the raw demand.
  proc2 = ModelPathProcessor()
  large_k = 0.00025  # lat accel 0.10, well inside gates when stable
  for _ in range(55):
    proc2.update(_sps_inputs(v_ego=v_ego, curvature=large_k, mode="apply"))
  for sign in (1.0, -1.0):
    result = proc2.update(_sps_inputs(v_ego=v_ego, curvature=sign * large_k, mode="apply"))
    assert result.desired_curvature * (sign * large_k) >= 0.0


# --- tight-corner gate escapes (city intersection turns) ---

def _corner_inputs(v_ego: float, curvature: float, measured_curvature: float, y_std: float = 0.05):
  # _inputs() indexes the path by point number; corners need geometry consistent with
  # T_IDXS at the given speed so get_curvature_from_plan recovers the same curvature.
  from dataclasses import replace
  ts = list(ModelConstants.T_IDXS)
  xs = [v_ego * t for t in ts]
  return replace(
    _inputs(v_ego=v_ego, curvature=curvature, measured_curvature=measured_curvature, y_std=y_std),
    position_x=xs,
    position_y=[0.5 * curvature * x * x for x in xs],
    orientation_z=[curvature * x for x in xs],
    orientation_rate_z=[curvature * v_ego] * N,
  )


def test_high_path_std_forgiven_when_measured_turn_confirms():
  proc = ModelPathProcessor()
  # Tight city corner: high path y-std, but the car is measurably turning the same way.
  result = proc.update(_corner_inputs(v_ego=5.0, curvature=0.05, measured_curvature=0.05, y_std=1.5))
  assert not result.gated
  assert result.quality == pytest.approx(0.75)


def test_high_path_std_still_gates_without_measured_confirmation():
  proc = ModelPathProcessor()
  result = proc.update(_corner_inputs(v_ego=5.0, curvature=0.05, measured_curvature=0.0, y_std=1.5))
  assert result.gated
  assert result.reason == "high_path_std"


def test_steep_turn_path_valid_when_measured_turn_confirms():
  proc = ModelPathProcessor()
  # Parabolic 0.1 1/m path exceeds dy/dx of 1.0 within the core points at 5 m/s.
  result = proc.update(_corner_inputs(v_ego=5.0, curvature=0.1, measured_curvature=0.1))
  assert result.reason != "invalid_path"


def test_steep_path_still_invalid_without_measured_turn():
  proc = ModelPathProcessor()
  result = proc.update(_corner_inputs(v_ego=5.0, curvature=0.1, measured_curvature=0.0))
  assert result.gated
  assert result.reason == "invalid_path"


def test_core_path_slope_limit_stays_tight_at_speed():
  # Widened slope allowance is low-speed + measured-turn only.
  fast = _corner_inputs(v_ego=20.0, curvature=0.08, measured_curvature=0.08)
  slow_straight = _corner_inputs(v_ego=5.0, curvature=0.08, measured_curvature=0.0)
  slow_turning = _corner_inputs(v_ego=5.0, curvature=0.08, measured_curvature=0.06)
  assert ModelPathProcessor._core_path_slope_limit(fast) == pytest.approx(1.0)
  assert ModelPathProcessor._core_path_slope_limit(slow_straight) == pytest.approx(1.0)
  assert ModelPathProcessor._core_path_slope_limit(slow_turning) == pytest.approx(2.0)
