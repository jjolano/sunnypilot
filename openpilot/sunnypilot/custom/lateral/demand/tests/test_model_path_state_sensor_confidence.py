"""Tests for the ModelPathState sensor-confidence telemetry helper."""
from __future__ import annotations

import math

import pytest
import openpilot.cereal.messaging as messaging
from openpilot.sunnypilot.custom.lateral.demand.telemetry import (
  CONTROL_N_T_IDXS,
  set_model_path_state_preview,
  set_model_path_state_sensor_confidence,
  set_model_path_state_speed_shadow,
)


def _new_model_path_state():
  msg = messaging.new_message('controlsState')
  return msg.controlsState.modelPathState


def test_default_debug_populates_disabled_state():
  mps = _new_model_path_state()
  set_model_path_state_sensor_confidence(mps)

  assert mps.sensorConfidenceAvailable is False
  assert mps.sensorConfidenceBlockReason == "disabled"
  assert mps.sensorConfidenceScore == 0.0
  assert mps.sensorDisagreementLevel == "blocked"
  assert mps.sensorSuppressCandidate is False
  assert math.isnan(mps.sensorModelMeasuredCurvatureDelta)
  assert math.isnan(mps.sensorModelMeasuredLatAccelDelta)
  assert math.isnan(mps.sensorYawCurvature)
  assert math.isnan(mps.sensorModelYawLatAccelDelta)
  assert math.isnan(mps.sensorSteeringYawLatAccelDelta)


def test_custom_default_reason_when_debug_missing():
  mps = _new_model_path_state()
  set_model_path_state_sensor_confidence(mps, default_reason="missing")
  assert mps.sensorConfidenceBlockReason == "missing"


def test_default_signed_and_response_fields_are_nan_and_blocked():
  mps = _new_model_path_state()
  set_model_path_state_sensor_confidence(mps)

  assert math.isnan(mps.sensorModelYawLatAccelSignedDelta)
  assert math.isnan(mps.sensorSteeringYawLatAccelSignedDelta)
  assert mps.sensorResponseClassification == "blocked"


def test_populated_debug_maps_all_fields():
  mps = _new_model_path_state()
  debug = {
    "sensor_confidence_available": True,
    "sensor_confidence_block_reason": "ok",
    "sensor_confidence_score": 0.72,
    "sensor_disagreement_level": "medium",
    "sensor_suppress_candidate": True,
    "sensor_response_classification": "underresponse_candidate",
    "sensor_model_measured_curvature_delta": 0.002,
    "sensor_model_measured_lat_accel_delta": 0.8,
    "sensor_yaw_curvature": 0.001,
    "sensor_model_yaw_lat_accel_delta": 0.6,
    "sensor_steering_yaw_lat_accel_delta": 0.1,
    "sensor_model_yaw_lat_accel_signed_delta": 0.6,
    "sensor_steering_yaw_lat_accel_signed_delta": 0.1,
  }
  set_model_path_state_sensor_confidence(mps, debug)

  assert mps.sensorConfidenceAvailable is True
  assert mps.sensorConfidenceBlockReason == "ok"
  assert mps.sensorConfidenceScore == pytest.approx(0.72)
  assert mps.sensorDisagreementLevel == "medium"
  assert mps.sensorSuppressCandidate is True
  assert mps.sensorResponseClassification == "underresponse_candidate"
  assert mps.sensorModelMeasuredCurvatureDelta == pytest.approx(0.002)
  assert mps.sensorModelMeasuredLatAccelDelta == pytest.approx(0.8)
  assert mps.sensorYawCurvature == pytest.approx(0.001)
  assert mps.sensorModelYawLatAccelDelta == pytest.approx(0.6)
  assert mps.sensorSteeringYawLatAccelDelta == pytest.approx(0.1)
  assert mps.sensorModelYawLatAccelSignedDelta == pytest.approx(0.6)
  assert mps.sensorSteeringYawLatAccelSignedDelta == pytest.approx(0.1)


def test_partial_debug_fills_defaults_for_missing_keys():
  mps = _new_model_path_state()
  set_model_path_state_sensor_confidence(mps, {"sensor_confidence_available": True})

  assert mps.sensorConfidenceAvailable is True
  assert mps.sensorConfidenceBlockReason == "disabled"
  assert mps.sensorConfidenceScore == 0.0
  assert mps.sensorDisagreementLevel == "blocked"
  assert mps.sensorSuppressCandidate is False
  assert mps.sensorResponseClassification == "blocked"
  assert math.isnan(mps.sensorModelYawLatAccelSignedDelta)
  assert math.isnan(mps.sensorSteeringYawLatAccelSignedDelta)


def test_preview_assist_default_debug_populates_disabled_state():
  mps = _new_model_path_state()
  set_model_path_state_preview(mps)

  assert mps.previewAssistMode == "off"
  assert mps.previewAssistActive is False
  assert mps.previewAssistApplied is False
  assert mps.previewAssistReason == "disabled"
  assert mps.previewAssistConfidence == 0.0
  assert mps.previewAssistCurvatureNudge == 0.0


def test_preview_assist_debug_maps_all_fields():
  mps = _new_model_path_state()
  set_model_path_state_preview(mps, {
    "lateral_preview_assist_mode": "apply",
    "lateral_preview_assist_active": True,
    "lateral_preview_assist_applied": True,
    "lateral_preview_assist_reason": "ok",
    "lateral_preview_assist_confidence": 0.91,
    "lateral_preview_assist_t_preview": 0.75,
    "lateral_preview_assist_base_curvature": 0.001,
    "lateral_preview_assist_preview_curvature": 0.0012,
    "lateral_preview_assist_curvature_nudge": 0.00001,
    "lateral_preview_assist_ay_base": 0.4,
    "lateral_preview_assist_ay_preview": 0.48,
    "lateral_preview_assist_ay_delta": 0.05,
    "lateral_preview_assist_slew_limited": True,
  })

  assert mps.previewAssistMode == "apply"
  assert mps.previewAssistActive is True
  assert mps.previewAssistApplied is True
  assert mps.previewAssistReason == "ok"
  assert mps.previewAssistConfidence == pytest.approx(0.91)
  assert mps.previewAssistTPreview == pytest.approx(0.75)
  assert mps.previewAssistBaseCurvature == pytest.approx(0.001)
  assert mps.previewAssistPreviewCurvature == pytest.approx(0.0012)
  assert mps.previewAssistCurvatureNudge == pytest.approx(0.00001)
  assert mps.previewAssistAyBase == pytest.approx(0.4)
  assert mps.previewAssistAyPreview == pytest.approx(0.48)
  assert mps.previewAssistAyDelta == pytest.approx(0.05)
  assert mps.previewAssistSlewLimited is True


def test_speed_shadow_uses_current_and_predicted_speed():
  mps = _new_model_path_state()
  speeds = [10.0 + t for t in CONTROL_N_T_IDXS]
  accels = [0.5 for _ in CONTROL_N_T_IDXS]

  set_model_path_state_speed_shadow(mps, 0.01, 10.0, 0.2, speeds, accels, 0.5)

  assert mps.shadowCurrentLatAccel == pytest.approx(1.0)
  assert mps.shadowCurrentJerkSpeedTerm == pytest.approx(0.04)
  assert mps.shadowLatDelayLatAccel == pytest.approx(0.01 * 10.5 ** 2)
  assert mps.shadow05sLatAccel == pytest.approx(0.01 * 10.5 ** 2)
  assert mps.shadow10sLatAccel == pytest.approx(0.01 * 11.0 ** 2)
  assert mps.shadowLatDelayJerkSpeedTerm == pytest.approx(2.0 * 0.01 * 10.5 * 0.5)


def test_speed_shadow_missing_plan_writes_nan_for_predicted_fields():
  mps = _new_model_path_state()

  set_model_path_state_speed_shadow(mps, 0.01, 10.0, 0.2, [], [], 0.5)

  assert mps.shadowCurrentLatAccel == pytest.approx(1.0)
  assert mps.shadowCurrentJerkSpeedTerm == pytest.approx(0.04)
  assert math.isnan(mps.shadowLatDelayLatAccel)
  assert math.isnan(mps.shadow05sLatAccel)
  assert math.isnan(mps.shadow10sLatAccel)
  assert math.isnan(mps.shadowLatDelayJerkSpeedTerm)


def test_speed_shadow_invalid_plan_writes_nan_for_predicted_fields():
  mps = _new_model_path_state()
  speeds = [10.0 for _ in CONTROL_N_T_IDXS]
  accels = [0.5 for _ in CONTROL_N_T_IDXS]

  set_model_path_state_speed_shadow(mps, 0.01, 10.0, 0.2, speeds, accels, 0.5, plan_valid=False)

  assert mps.shadowCurrentLatAccel == pytest.approx(1.0)
  assert math.isnan(mps.shadowLatDelayLatAccel)
  assert math.isnan(mps.shadow05sLatAccel)
  assert math.isnan(mps.shadow10sLatAccel)
  assert math.isnan(mps.shadowLatDelayJerkSpeedTerm)


def test_speed_shadow_nonfinite_inputs_do_not_look_valid():
  mps = _new_model_path_state()
  speeds = [10.0 for _ in CONTROL_N_T_IDXS]
  accels = [0.5 for _ in CONTROL_N_T_IDXS]
  speeds[3] = float('inf')

  set_model_path_state_speed_shadow(mps, 0.01, 10.0, 0.2, speeds, accels, float('nan'))

  assert mps.shadowCurrentLatAccel == pytest.approx(1.0)
  assert math.isnan(mps.shadowLatDelayLatAccel)
  assert math.isnan(mps.shadow05sLatAccel)
  assert math.isnan(mps.shadow10sLatAccel)
  assert math.isnan(mps.shadowLatDelayJerkSpeedTerm)


def test_speed_shadow_mismatched_accels_only_nan_jerk():
  mps = _new_model_path_state()
  speeds = [10.0 for _ in CONTROL_N_T_IDXS]

  set_model_path_state_speed_shadow(mps, 0.01, 10.0, 0.2, speeds, [], 0.5)

  assert mps.shadowLatDelayLatAccel == pytest.approx(1.0)
  assert math.isnan(mps.shadowLatDelayJerkSpeedTerm)
