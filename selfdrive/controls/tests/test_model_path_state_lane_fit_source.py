"""Tests for ModelPathState lane-fit source telemetry defaults and population."""
from __future__ import annotations

import pytest

from cereal import log

from openpilot.selfdrive.controls.controlsd import set_model_path_state_lane_fit_source


def _make_model_path_state():
  msg = log.ControlsState.new_message()
  return msg.modelPathState


def test_lane_fit_source_defaults_when_debug_empty():
  mps = _make_model_path_state()
  set_model_path_state_lane_fit_source(mps)

  assert mps.laneFitSourceMode == "off"
  assert mps.laneFitSourceActive is False
  assert mps.laneFitSourceApplied is False
  assert mps.laneFitSourceReason == "disabled"
  assert mps.laneFitSourceCandidateCurvature == pytest.approx(0.0)
  assert mps.laneFitSourceAppliedCurvature == pytest.approx(0.0)
  assert mps.laneFitSourceLatAccelDelta == pytest.approx(0.0)
  assert mps.laneFitSourceConfidence == pytest.approx(0.0)
  assert mps.laneFitSourceSlewLimited is False


def test_lane_fit_source_missing_debug_uses_missing_reason():
  mps = _make_model_path_state()
  set_model_path_state_lane_fit_source(mps, {}, default_reason="missing")

  assert mps.laneFitSourceMode == "off"
  assert mps.laneFitSourceReason == "missing"


def test_lane_fit_source_populated_from_debug():
  mps = _make_model_path_state()
  debug = {
    "lane_fit_source_mode": "apply",
    "lane_fit_source_active": True,
    "lane_fit_source_applied": True,
    "lane_fit_source_reason": "ok",
    "lane_fit_source_candidate_curvature": 0.0008,
    "lane_fit_source_applied_curvature": 0.0002,
    "lane_fit_source_lat_accel_delta": 0.32,
    "lane_fit_source_confidence": 0.8,
    "lane_fit_source_slew_limited": True,
  }
  set_model_path_state_lane_fit_source(mps, debug)

  assert mps.laneFitSourceMode == "apply"
  assert mps.laneFitSourceActive is True
  assert mps.laneFitSourceApplied is True
  assert mps.laneFitSourceReason == "ok"
  assert mps.laneFitSourceCandidateCurvature == pytest.approx(0.0008)
  assert mps.laneFitSourceAppliedCurvature == pytest.approx(0.0002)
  assert mps.laneFitSourceLatAccelDelta == pytest.approx(0.32)
  assert mps.laneFitSourceConfidence == pytest.approx(0.8)
  assert mps.laneFitSourceSlewLimited is True
