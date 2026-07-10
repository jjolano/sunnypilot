"""Tests for ModelPathState lane-geometry telemetry defaults and population."""
from __future__ import annotations

import pytest

from cereal import log
from openpilot.sunnypilot.custom.lateral.demand.telemetry import set_model_path_state_geometry


def _make_model_path_state():
  msg = log.ControlsState.new_message()
  return msg.modelPathState


def test_geometry_defaults_when_debug_empty():
  mps = _make_model_path_state()
  set_model_path_state_geometry(mps)
  assert mps.geometryMode is False
  assert mps.geometryValid is False
  assert mps.geometryReason == "disabled"
  assert mps.geometryConfidence == pytest.approx(0.0)
  assert mps.geometryOffsetNear == pytest.approx(0.0)
  assert mps.geometryOffsetPreview == pytest.approx(0.0)
  assert mps.geometryWidthNear == pytest.approx(0.0)
  assert mps.geometryWidthPreview == pytest.approx(0.0)


def test_geometry_populated_from_debug():
  mps = _make_model_path_state()
  debug = {
    "lane_centering_geometry_mode": True,
    "lane_centering_geometry_valid": True,
    "lane_centering_geometry_reason": "ok",
    "lane_centering_geometry_confidence": 0.85,
    "lane_centering_geometry_offset_near": -0.12,
    "lane_centering_geometry_offset_preview": -0.18,
    "lane_centering_geometry_width_near": 3.55,
    "lane_centering_geometry_width_preview": 3.60,
  }
  set_model_path_state_geometry(mps, debug)
  assert mps.geometryMode is True
  assert mps.geometryValid is True
  assert mps.geometryReason == "ok"
  assert mps.geometryConfidence == pytest.approx(0.85)
  assert mps.geometryOffsetNear == pytest.approx(-0.12)
  assert mps.geometryOffsetPreview == pytest.approx(-0.18)
  assert mps.geometryWidthNear == pytest.approx(3.55)
  assert mps.geometryWidthPreview == pytest.approx(3.60)
