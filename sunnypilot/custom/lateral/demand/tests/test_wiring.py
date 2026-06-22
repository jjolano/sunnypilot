"""Tests for the lateral demand pipeline controlsd wiring adapter (opt-in)."""
from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import pytest

from openpilot.sunnypilot.custom.lateral.demand.wiring import LateralDemandAdapter, build_pipeline_inputs, sanitized_model_age_s

N = 33


def fake_model(curvature=0.001, include_meta=True):
  xs = [float(x) for x in range(N)]
  ys = [0.5 * curvature * x * x for x in range(N)]
  model = SimpleNamespace(
    position=SimpleNamespace(x=xs, y=ys, yStd=[0.1] * N),
    orientation=SimpleNamespace(z=[curvature * x for x in range(N)]),
    orientationRate=SimpleNamespace(z=[curvature * 20.0] * N),
    laneLineProbs=[0.9, 0.9, 0.9, 0.9],
    frameDropPerc=0.0,
  )
  if include_meta:
    model.meta = SimpleNamespace(laneChangeState=SimpleNamespace(value=0), laneChangeDirection=SimpleNamespace(value=0))
  return model


class FakeParams:
  def __init__(self, **vals):
    self._v = vals
  def get_bool(self, k):
    return bool(self._v.get(k, False))


class FakeCapnpEnum:
  def __init__(self, name: str):
    self._name = name

  def __str__(self) -> str:
    return self._name


class SpyPipeline:
  def __init__(self):
    self.inputs: Any | None = None

  def update(self, inputs):
    self.inputs = inputs
    return SimpleNamespace(
      demand=SimpleNamespace(processed_curvature=inputs.desired_curvature),
      debug={"raw_curvature": inputs.desired_curvature},
    )


def test_build_pipeline_inputs_extracts_model_arrays():
  inp = build_pipeline_inputs(lat_active=True, v_ego=20.0, roll=0.0, raw_curvature=0.002,
                               measured_curvature=0.0015, model_v2=fake_model(0.002),
                              lane_centering_assist_enabled=False, model_age_s=0.25)
  assert len(inp.position_x) == N
  assert inp.desired_curvature == 0.002
  assert inp.model_age_s == pytest.approx(0.25)
  assert inp.lane_change_state == 0  # conservative default (harness-gated)
  assert inp.lane_change_state_valid is True


def test_sanitized_model_age_fails_nonfinite_missing_negative_to_stale():
  assert math.isinf(sanitized_model_age_s(None))
  assert math.isinf(sanitized_model_age_s(float("nan")))
  assert math.isinf(sanitized_model_age_s(-0.1))
  assert sanitized_model_age_s(0.19) == pytest.approx(0.19)


def test_build_pipeline_inputs_marks_missing_lane_change_meta_unknown():
  inp = build_pipeline_inputs(lat_active=True, v_ego=20.0, roll=0.0, raw_curvature=0.002,
                              measured_curvature=0.0015, model_v2=fake_model(0.002, include_meta=False),
                              lane_centering_assist_enabled=False, steering_pressed=False)
  assert inp.lane_change_state_valid is False


def test_build_pipeline_inputs_decodes_capnp_lane_change_enums():
  model = fake_model(0.002)
  model.meta = SimpleNamespace(laneChangeState=FakeCapnpEnum("laneChangeStarting"),
                               laneChangeDirection=FakeCapnpEnum("left"))

  inp = build_pipeline_inputs(lat_active=True, v_ego=20.0, roll=0.0, raw_curvature=0.002,
                              measured_curvature=0.0015, model_v2=model,
                              lane_centering_assist_enabled=False, steering_pressed=False)

  assert inp.lane_change_state == 2
  assert inp.lane_change_direction == 1
  assert inp.lane_change_state_valid is True


def test_build_pipeline_inputs_marks_unknown_lane_change_enum_invalid():
  model = fake_model(0.002)
  model.meta = SimpleNamespace(laneChangeState=FakeCapnpEnum("futureLaneChangeState"),
                               laneChangeDirection=FakeCapnpEnum("none"))

  inp = build_pipeline_inputs(lat_active=True, v_ego=20.0, roll=0.0, raw_curvature=0.002,
                              measured_curvature=0.0015, model_v2=model,
                              lane_centering_assist_enabled=False, steering_pressed=False)

  assert inp.lane_change_state == 0
  assert inp.lane_change_state_valid is False


def test_adapter_disabled_passthrough():
  a = LateralDemandAdapter(FakeParams(CustomLateralDemandEnabled=False))
  out = a.process(True, 20.0, 0.0, 0.0123, 0.0123, fake_model(), steering_pressed=False,
                  model_age_s=0.01, yaw_rate=0.0, steering_rate_deg=0.0)
  assert out == 0.0123
  assert a.last_result is None
  assert a.last_debug["sensor_confidence_available"] is True
  assert a.last_debug["sensor_disagreement_level"] == "high"
  assert a.last_debug["sensor_suppress_candidate"] is True


def test_default_params_disable_adapter_and_curve_memory():
  a = LateralDemandAdapter(FakeParams())
  assert a.enabled is False
  assert a.curve_memory_enabled is False
  out = a.process(True, 20.0, 0.0, 0.0123, 0.0123, fake_model(), steering_pressed=False,
                  model_age_s=0.01, yaw_rate=0.246, steering_rate_deg=0.0)
  assert out == 0.0123


def test_adapter_enabled_processes_curvature():
  a = LateralDemandAdapter(FakeParams(CustomLateralDemandEnabled=True))
  out = a.process(True, 20.0, 0.0, 0.001, 0.001, fake_model(0.001))
  assert math.isfinite(out)
  assert out == pytest.approx(0.001, abs=3e-3)  # high-quality straight path passes near through


@pytest.mark.parametrize("curvature", [0.002, -0.002])
def test_adapter_preserves_curvature_sign(curvature):
  """A nonzero demand curvature must not have its sign flipped by the wiring adapter."""
  a = LateralDemandAdapter(FakeParams(CustomLateralDemandEnabled=True))
  spy = SpyPipeline()
  setattr(a, "_pipeline", spy)
  out = a.process(True, 20.0, 0.0, curvature, curvature, fake_model(curvature))
  assert out != 0.0
  assert math.copysign(1.0, out) == math.copysign(1.0, curvature)
  assert spy.inputs is not None
  assert spy.inputs.desired_curvature == pytest.approx(curvature, abs=1e-9)


def test_adapter_forwards_steering_pressed_and_lane_change_state():
  a = LateralDemandAdapter(FakeParams(CustomLateralDemandEnabled=True))
  spy = SpyPipeline()
  setattr(a, "_pipeline", spy)
  a.process(True, 20.0, 0.0, 0.001, 0.001, fake_model(0.001), steering_pressed=True,
            yaw_rate=0.02, steering_rate_deg=4.0, steer_limited=True)
  assert getattr(spy.inputs, "steering_pressed") is True
  assert getattr(spy.inputs, "lane_change_state_valid") is True
  assert getattr(spy.inputs, "yaw_rate") == pytest.approx(0.02)
  assert getattr(spy.inputs, "steering_rate_deg") == pytest.approx(4.0)
  assert getattr(spy.inputs, "steer_limited") is True


def test_adapter_enabled_turns_on_model_path_smoothing():
  a = LateralDemandAdapter(FakeParams(CustomLateralDemandEnabled=True))
  spy = SpyPipeline()
  setattr(a, "_pipeline", spy)
  out = a.process(True, 20.0, 0.0, 0.001, 0.001, fake_model(0.001))
  assert out == pytest.approx(0.001)
  assert spy.inputs is not None
  assert spy.inputs.smooth_model_path_curvature is True
  assert spy.inputs.demand_jerk_smoothing_enabled is False
  assert a.last_result is not None
  assert a.last_debug.get("raw_curvature") == pytest.approx(0.001)


def test_adapter_disabled_does_not_call_pipeline():
  class RaisingPipeline:
    def update(self, _inputs):
      raise AssertionError("disabled adapter must not run the demand pipeline")

  a = LateralDemandAdapter(FakeParams(CustomLateralDemandEnabled=False))
  setattr(a, "_pipeline", RaisingPipeline())

  out = a.process(True, 10.0, 0.0, 0.02, 0.02, fake_model(), steering_pressed=False,
                  model_age_s=0.01, yaw_rate=0.0, steering_rate_deg=0.0)

  assert out == 0.02
  assert a.last_result is None
  assert a.last_debug["sensor_confidence_available"] is True
  assert a.last_debug["sensor_disagreement_level"] == "high"
  assert a.last_debug["sensor_suppress_candidate"] is True


def test_adapter_disabled_shadow_blocks_driver_override():
  a = LateralDemandAdapter(FakeParams(CustomLateralDemandEnabled=False))

  out = a.process(True, 10.0, 0.0, 0.02, 0.02, fake_model(), steering_pressed=True,
                  model_age_s=0.01, yaw_rate=0.0, steering_rate_deg=0.0)

  assert out == 0.02
  assert a.last_result is None
  assert a.last_debug["sensor_confidence_available"] is False
  assert a.last_debug["sensor_confidence_block_reason"] == "driver_override"
  assert a.last_debug["sensor_suppress_candidate"] is False


def test_build_pipeline_inputs_allows_harness_demand_jerk_smoothing():
  inp = build_pipeline_inputs(lat_active=True, v_ego=20.0, roll=0.0, raw_curvature=0.001,
                              measured_curvature=0.001, model_v2=fake_model(0.001),
                              lane_centering_assist_enabled=False, steering_pressed=False,
                              demand_jerk_smoothing_enabled=True)

  assert inp.demand_jerk_smoothing_enabled is True


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
  assert a.last_debug["sensor_confidence_available"] is False
  assert a.last_debug["sensor_confidence_block_reason"] == "driver_override"
