from __future__ import annotations

import math

import pytest

from openpilot.sunnypilot.custom.lateral.demand.sensor_confidence import (
  SensorConfidenceInputs,
  evaluate_sensor_confidence,
)


def base_inputs(**overrides):
  data = dict(
    lat_active=True,
    v_ego=20.0,
    model_curvature=0.003,
    measured_curvature=0.001,
    model_path_gated=False,
    model_path_reason="ok",
    model_age_s=0.05,
    steering_pressed=False,
    steering_rate_deg=5.0,
    yaw_rate=0.02,
    steer_limited=False,
    lane_change_active=False,
    lane_change_state_valid=True,
    left_blinker=False,
    right_blinker=False,
  )
  data.update(overrides)
  return SensorConfidenceInputs(**data)


def test_clean_case_computes_shadow_deltas():
  result = evaluate_sensor_confidence(base_inputs())

  assert result.available is True
  assert result.block_reason == "ok"
  assert result.model_measured_curvature_delta == pytest.approx(0.002)
  assert result.model_measured_lat_accel_delta == pytest.approx(0.8)
  assert result.yaw_curvature == pytest.approx(0.001)
  assert result.model_yaw_lat_accel_delta == pytest.approx(0.8)
  assert result.steering_yaw_lat_accel_delta == pytest.approx(0.0)
  assert result.model_yaw_lat_accel_signed_delta == pytest.approx(0.8)
  assert result.steering_yaw_lat_accel_signed_delta == pytest.approx(0.0)
  assert result.response_classification == "medium_disagreement"
  assert result.disagreement_level == "medium"


@pytest.mark.parametrize("kwargs, reason", [
  ({"v_ego": 3.0}, "low_speed"),
  ({"model_path_gated": True}, "path_gated"),
  ({"model_path_reason": "model_stale"}, "model_stale"),
  ({"model_age_s": 0.30}, "model_stale"),
  ({"steering_pressed": True}, "driver_override"),
  ({"lane_change_active": True}, "lane_change"),
  ({"lane_change_state_valid": False}, "lane_change_unknown"),
  ({"steer_limited": True}, "steer_limited"),
  ({"steering_rate_deg": 140.0}, "steering_rate_spike"),
  ({"yaw_rate": None}, "yaw_unavailable"),
  ({"steering_rate_deg": None}, "steering_rate_unavailable"),
  ({"model_curvature": float("nan")}, "nonfinite_curvature"),
])
def test_blockers_fail_closed(kwargs, reason):
  result = evaluate_sensor_confidence(base_inputs(**kwargs))

  assert result.available is False
  assert result.block_reason == reason
  assert result.disagreement_level == "blocked"
  assert result.suppress_candidate is False


def test_high_disagreement_is_debug_only_suppress_candidate():
  result = evaluate_sensor_confidence(base_inputs(model_curvature=0.008, measured_curvature=0.0, yaw_rate=0.0))

  assert result.available is True
  assert result.disagreement_level == "high"
  assert result.suppress_candidate is True
  assert result.response_classification == "underresponse_candidate"
  assert math.isfinite(result.score)


def test_negative_turn_underresponse_candidate_uses_requested_direction():
  result = evaluate_sensor_confidence(base_inputs(model_curvature=-0.008, measured_curvature=-0.001, yaw_rate=-0.02))

  assert result.available is True
  assert result.model_yaw_lat_accel_signed_delta < -1.0
  assert result.response_classification == "underresponse_candidate"


def test_negative_turn_overresponse_candidate_uses_requested_direction():
  result = evaluate_sensor_confidence(base_inputs(model_curvature=-0.001, measured_curvature=-0.008, yaw_rate=-0.16))

  assert result.available is True
  assert result.model_yaw_lat_accel_signed_delta > 1.0
  assert result.response_classification == "overresponse_candidate"


def test_low_disagreement_no_response_candidate():
  result = evaluate_sensor_confidence(base_inputs(model_curvature=0.001, measured_curvature=0.001, yaw_rate=0.02))

  assert result.available is True
  assert result.response_classification == "low_disagreement"
  assert result.suppress_candidate is False
