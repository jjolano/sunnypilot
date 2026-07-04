"""Tests for ModelPathState lane-rate damping telemetry defaults and population."""
from __future__ import annotations

import pytest
import cereal.messaging as messaging

from openpilot.selfdrive.controls.controlsd import set_model_path_state_lane_rate_damping


def _new_model_path_state():
  msg = messaging.new_message('controlsState')
  return msg.controlsState.modelPathState


def test_default_debug_populates_disabled_state():
  mps = _new_model_path_state()
  set_model_path_state_lane_rate_damping(mps)

  assert mps.laneRateDampingMode == "off"
  assert mps.laneRateDampingActive is False
  assert mps.laneRateDampingApplied is False
  assert mps.laneRateDampingReason == "disabled"
  assert mps.laneRateDampingLaneCenter == pytest.approx(0.0)
  assert mps.laneRateDampingLaneCenterRate == pytest.approx(0.0)
  assert mps.laneRateDampingLatAccel == pytest.approx(0.0)
  assert mps.laneRateDampingCurvature == pytest.approx(0.0)
  assert mps.laneRateDampingCapLatAccel == pytest.approx(0.05)


def test_missing_debug_uses_missing_reason():
  mps = _new_model_path_state()
  set_model_path_state_lane_rate_damping(mps, {}, default_reason="missing")

  assert mps.laneRateDampingMode == "off"
  assert mps.laneRateDampingReason == "missing"
  assert mps.laneRateDampingActive is False
  assert mps.laneRateDampingApplied is False


def test_populated_debug_maps_all_fields():
  mps = _new_model_path_state()
  debug = {
    "lane_rate_damping_mode": "apply",
    "lane_rate_damping_active": True,
    "lane_rate_damping_applied": True,
    "lane_rate_damping_reason": "ok",
    "lane_rate_damping_lane_center": 0.12,
    "lane_rate_damping_lane_center_rate": 0.34,
    "lane_rate_damping_lat_accel": -0.05,
    "lane_rate_damping_curvature": -0.000125,
    "lane_rate_damping_cap_lat_accel": 0.05,
  }
  set_model_path_state_lane_rate_damping(mps, debug)

  assert mps.laneRateDampingMode == "apply"
  assert mps.laneRateDampingActive is True
  assert mps.laneRateDampingApplied is True
  assert mps.laneRateDampingReason == "ok"
  assert mps.laneRateDampingLaneCenter == pytest.approx(0.12)
  assert mps.laneRateDampingLaneCenterRate == pytest.approx(0.34)
  assert mps.laneRateDampingLatAccel == pytest.approx(-0.05)
  assert mps.laneRateDampingCurvature == pytest.approx(-0.000125)
  assert mps.laneRateDampingCapLatAccel == pytest.approx(0.05)
