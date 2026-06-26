"""Tests for the ModelPathState sensor-confidence telemetry helper."""
from __future__ import annotations

import math

import pytest
import cereal.messaging as messaging
from openpilot.selfdrive.controls.controlsd import set_model_path_state_sensor_confidence


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
