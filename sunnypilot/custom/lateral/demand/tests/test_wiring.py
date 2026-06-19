"""Tests for the lateral demand pipeline controlsd wiring adapter (opt-in)."""
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from openpilot.sunnypilot.custom.lateral.demand.wiring import LateralDemandAdapter, build_pipeline_inputs

N = 33


def fake_model(curvature=0.001):
  xs = [float(x) for x in range(N)]
  ys = [0.5 * curvature * x * x for x in range(N)]
  return SimpleNamespace(
    position=SimpleNamespace(x=xs, y=ys, yStd=[0.1] * N),
    orientation=SimpleNamespace(z=[curvature * x for x in range(N)]),
    orientationRate=SimpleNamespace(z=[curvature * 20.0] * N),
    laneLineProbs=[0.9, 0.9, 0.9, 0.9],
    frameDropPerc=0.0,
  )


class FakeParams:
  def __init__(self, **vals):
    self._v = vals
  def get_bool(self, k):
    return bool(self._v.get(k, False))


class SpyPipeline:
  def __init__(self):
    self.inputs = None

  def update(self, inputs):
    self.inputs = inputs
    return SimpleNamespace(
      demand=SimpleNamespace(processed_curvature=inputs.desired_curvature),
      debug={"raw_curvature": inputs.desired_curvature},
    )


def test_build_pipeline_inputs_extracts_model_arrays():
  inp = build_pipeline_inputs(lat_active=True, v_ego=20.0, roll=0.0, raw_curvature=0.002,
                              measured_curvature=0.0015, model_v2=fake_model(0.002),
                              lane_centering_assist_enabled=False)
  assert len(inp.position_x) == N
  assert inp.desired_curvature == 0.002
  assert inp.lane_change_state == 0  # conservative default (harness-gated)


def test_adapter_disabled_passthrough():
  a = LateralDemandAdapter(FakeParams(CustomLateralDemandEnabled=False))
  out = a.process(True, 20.0, 0.0, 0.0123, 0.0123, fake_model())
  assert out == 0.0123


def test_adapter_enabled_processes_curvature():
  a = LateralDemandAdapter(FakeParams(CustomLateralDemandEnabled=True))
  out = a.process(True, 20.0, 0.0, 0.001, 0.001, fake_model(0.001))
  assert math.isfinite(out)
  assert out == pytest.approx(0.001, abs=3e-3)  # high-quality straight path passes near through


def test_adapter_enabled_turns_on_model_path_smoothing():
  a = LateralDemandAdapter(FakeParams(CustomLateralDemandEnabled=True))
  spy = SpyPipeline()
  setattr(a, "_pipeline", spy)
  out = a.process(True, 20.0, 0.0, 0.001, 0.001, fake_model(0.001))
  assert out == pytest.approx(0.001)
  assert spy.inputs is not None
  assert spy.inputs.smooth_model_path_curvature is True
  assert a.last_result is not None
  assert a.last_debug.get("raw_curvature") == pytest.approx(0.001)


def test_adapter_fail_closed_on_bad_model():
  a = LateralDemandAdapter(FakeParams(CustomLateralDemandEnabled=True))
  out = a.process(True, 20.0, 0.0, 0.005, 0.005, None)  # None model -> must not raise
  assert out == 0.005
  assert a.last_result is None


def test_adapter_disabled_clears_previous_result():
  a = LateralDemandAdapter(FakeParams(CustomLateralDemandEnabled=True))
  assert math.isfinite(a.process(True, 20.0, 0.0, 0.001, 0.001, fake_model(0.001)))
  assert a.last_result is not None

  a.enabled = False
  out = a.process(True, 20.0, 0.0, 0.002, 0.001, fake_model(0.002))

  assert out == 0.002
  assert a.last_result is None
  assert a.last_debug == {}
