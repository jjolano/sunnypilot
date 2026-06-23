"""Tests for the lateral demand pipeline controlsd wiring adapter (opt-in)."""
from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import pytest

import openpilot.sunnypilot.custom.lateral.demand.wiring as _wiring
from openpilot.sunnypilot.custom.lateral.demand.wiring import (
  LateralDemandAdapter,
  build_pipeline_inputs,
  sanitized_model_age_s,
  _debug_trace_mode,
  _shadow_trace_event,
)

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


def fake_model_shadow(offset=0.0, prob=0.9, width=4.0, curvature=0.0):
  xs = [float(x) for x in range(N)]
  path_ys = [offset + 0.5 * curvature * x * x for x in xs]
  left_ys = [width * 0.5 + 0.5 * curvature * x * x for x in xs]
  right_ys = [-width * 0.5 + 0.5 * curvature * x * x for x in xs]
  return SimpleNamespace(
    position=SimpleNamespace(x=xs, y=path_ys, yStd=[0.1] * N),
    orientation=SimpleNamespace(z=[curvature * x for x in xs]),
    orientationRate=SimpleNamespace(z=[curvature * 20.0] * N),
    laneLineProbs=[0.5, prob, prob, 0.5],
    laneLines=[
      SimpleNamespace(x=xs, y=[(width * 1.5) + 0.5 * curvature * x * x for x in xs]),
      SimpleNamespace(x=xs, y=left_ys),
      SimpleNamespace(x=xs, y=right_ys),
      SimpleNamespace(x=xs, y=[(-width * 1.5) + 0.5 * curvature * x * x for x in xs]),
    ],
    frameDropPerc=0.0,
    meta=SimpleNamespace(
      laneChangeState=SimpleNamespace(value=0),
      laneChangeDirection=SimpleNamespace(value=0),
    ),
  )


class FakeParams:
  def __init__(self, **vals):
    self._v = vals

  def get_bool(self, k):
    return bool(self._v.get(k, False))

  def get(self, k, default=None):
    return self._v.get(k, default)


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


@pytest.fixture
def cloudlog_spy(monkeypatch):
  calls: list[tuple[str, dict[str, Any]]] = []

  def _event(name: str, **kwargs: Any):
    calls.append((name, dict(kwargs)))

  monkeypatch.setattr(_wiring.cloudlog, "event", _event)
  return calls


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
  assert spy.inputs.demand_jerk_smoothing_enabled is True
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


# --- LateralDebugTraceMode shadow telemetry tests ---


def test_debug_trace_mode_defaults_and_invalid_resolves_to_off():
  assert _debug_trace_mode(None) == "off"
  assert _debug_trace_mode("") == "off"
  assert _debug_trace_mode("LOG") == "log"
  assert _debug_trace_mode("bad") == "off"
  assert _debug_trace_mode("log") == "log"


def test_refresh_params_reads_lateral_debug_trace_mode():
  a = LateralDemandAdapter(FakeParams(LateralDebugTraceMode="log"))
  assert a.debug_trace_mode == "log"

  a = LateralDemandAdapter(FakeParams(LateralDebugTraceMode="bad"))
  assert a.debug_trace_mode == "off"

  a = LateralDemandAdapter(FakeParams())
  assert a.debug_trace_mode == "off"


def test_shadow_trace_not_emitted_when_mode_off(cloudlog_spy):
  a = LateralDemandAdapter(FakeParams(CustomLateralDemandEnabled=False, LateralDebugTraceMode="off"))
  a.process(True, 20.0, 0.0, 0.001, 0.001, fake_model_shadow(), steering_pressed=False)
  assert cloudlog_spy == []


def test_shadow_trace_not_emitted_when_adapter_enabled_but_mode_off(cloudlog_spy):
  a = LateralDemandAdapter(FakeParams(CustomLateralDemandEnabled=True, LateralDebugTraceMode="off"))
  a.process(True, 20.0, 0.0, 0.001, 0.001, fake_model_shadow(), steering_pressed=False)
  assert cloudlog_spy == []


def test_shadow_trace_rate_limited(cloudlog_spy):
  a = LateralDemandAdapter(FakeParams(CustomLateralDemandEnabled=False, LateralDebugTraceMode="log"))
  for _ in range(250):
    a.process(True, 20.0, 0.0, 0.001, 0.001, fake_model_shadow(), steering_pressed=False)
  names = [name for name, _ in cloudlog_spy]
  assert names.count("lateral_path_shadow") == 3  # ticks 1, 101, 201


def test_shadow_trace_centered_path_zero_delta(cloudlog_spy):
  event = _shadow_trace_event(
    tick=100, last_trace_tick=0, debug_trace_mode="log",
    lat_active=True, v_ego=20.0, raw_curvature=0.001, measured_curvature=0.001,
    model_v2=fake_model_shadow(offset=0.0), steering_pressed=False, steer_limited=False,
    yaw_rate=0.0,
  )[0]
  assert event is not None
  assert event["gatePass"] is True
  assert event["blockReason"] == "ok"
  assert event["rawDeltaK"] == pytest.approx(0.0, abs=1e-9)
  assert event["clippedDeltaK"] == pytest.approx(0.0, abs=1e-9)


def test_shadow_trace_positive_offset_negative_delta(cloudlog_spy):
  event = _shadow_trace_event(
    tick=100, last_trace_tick=0, debug_trace_mode="log",
    lat_active=True, v_ego=20.0, raw_curvature=0.001, measured_curvature=0.001,
    model_v2=fake_model_shadow(offset=0.5), steering_pressed=False, steer_limited=False,
    yaw_rate=0.0,
  )[0]
  assert event is not None
  assert event["gatePass"] is True
  assert event["rawDeltaK"] < 0.0


def test_shadow_trace_negative_offset_positive_delta(cloudlog_spy):
  event = _shadow_trace_event(
    tick=100, last_trace_tick=0, debug_trace_mode="log",
    lat_active=True, v_ego=20.0, raw_curvature=0.001, measured_curvature=0.001,
    model_v2=fake_model_shadow(offset=-0.5), steering_pressed=False, steer_limited=False,
    yaw_rate=0.0,
  )[0]
  assert event is not None
  assert event["gatePass"] is True
  assert event["rawDeltaK"] > 0.0


@pytest.mark.parametrize("v_ego", [5.0, 15.0, 30.0])
def test_shadow_trace_cap_scales_with_speed(v_ego, cloudlog_spy):
  event = _shadow_trace_event(
    tick=100, last_trace_tick=0, debug_trace_mode="log",
    lat_active=True, v_ego=v_ego, raw_curvature=0.001, measured_curvature=0.001,
    model_v2=fake_model_shadow(offset=2.0), steering_pressed=False, steer_limited=False,
    yaw_rate=0.0,
  )[0]
  assert event is not None
  expected_cap = min(0.08 / (max(v_ego, 0.5) ** 2), 0.001)
  assert event["cap"] == pytest.approx(expected_cap, abs=1e-12)
  assert abs(event["clippedDeltaK"]) <= expected_cap
  assert abs(event["clippedDeltaK"]) == pytest.approx(expected_cap, abs=1e-12)
  assert abs(event["rawDeltaK"]) > expected_cap


def test_shadow_trace_blocks_low_probs(cloudlog_spy):
  event = _shadow_trace_event(
    tick=100, last_trace_tick=0, debug_trace_mode="log",
    lat_active=True, v_ego=20.0, raw_curvature=0.001, measured_curvature=0.001,
    model_v2=fake_model_shadow(prob=0.1), steering_pressed=False, steer_limited=False,
    yaw_rate=0.0,
  )[0]
  assert event is not None
  assert event["gatePass"] is False
  assert event["blockReason"] == "low_prob"


def test_shadow_trace_blocks_missing_lanes(cloudlog_spy):
  model = fake_model_shadow()
  del model.laneLines
  event = _shadow_trace_event(
    tick=100, last_trace_tick=0, debug_trace_mode="log",
    lat_active=True, v_ego=20.0, raw_curvature=0.001, measured_curvature=0.001,
    model_v2=model, steering_pressed=False, steer_limited=False,
    yaw_rate=0.0,
  )[0]
  assert event is not None
  assert event["gatePass"] is False
  assert event["blockReason"] == "missing_lanes"


@pytest.mark.parametrize("width", [0.5, 10.0])
def test_shadow_trace_blocks_bad_width(width, cloudlog_spy):
  event = _shadow_trace_event(
    tick=100, last_trace_tick=0, debug_trace_mode="log",
    lat_active=True, v_ego=20.0, raw_curvature=0.001, measured_curvature=0.001,
    model_v2=fake_model_shadow(width=width), steering_pressed=False, steer_limited=False,
    yaw_rate=0.0,
  )[0]
  assert event is not None
  assert event["gatePass"] is False
  assert event["blockReason"] == "bad_width"


def test_shadow_trace_blocks_lane_change(cloudlog_spy):
  model = fake_model_shadow()
  model.meta = SimpleNamespace(
    laneChangeState=SimpleNamespace(value=2),
    laneChangeDirection=SimpleNamespace(value=1),
  )
  event = _shadow_trace_event(
    tick=100, last_trace_tick=0, debug_trace_mode="log",
    lat_active=True, v_ego=20.0, raw_curvature=0.001, measured_curvature=0.001,
    model_v2=model, steering_pressed=False, steer_limited=False,
    yaw_rate=0.0,
  )[0]
  assert event is not None
  assert event["gatePass"] is False
  assert event["blockReason"] == "lane_change"


def test_shadow_trace_blocks_steering_pressed(cloudlog_spy):
  event = _shadow_trace_event(
    tick=100, last_trace_tick=0, debug_trace_mode="log",
    lat_active=True, v_ego=20.0, raw_curvature=0.001, measured_curvature=0.001,
    model_v2=fake_model_shadow(), steering_pressed=True, steer_limited=False,
    yaw_rate=0.0,
  )[0]
  assert event is not None
  assert event["gatePass"] is False
  assert event["blockReason"] == "driver_override"


def test_shadow_trace_blocks_lat_inactive(cloudlog_spy):
  event = _shadow_trace_event(
    tick=100, last_trace_tick=0, debug_trace_mode="log",
    lat_active=False, v_ego=20.0, raw_curvature=0.001, measured_curvature=0.001,
    model_v2=fake_model_shadow(), steering_pressed=False, steer_limited=False,
    yaw_rate=0.0,
  )[0]
  assert event is not None
  assert event["gatePass"] is False
  assert event["blockReason"] == "lat_inactive"


def test_shadow_trace_blocks_bad_speed(cloudlog_spy):
  for v_ego in [0.0, float("nan"), -1.0]:
    event = _shadow_trace_event(
      tick=100, last_trace_tick=0, debug_trace_mode="log",
      lat_active=True, v_ego=v_ego, raw_curvature=0.001, measured_curvature=0.001,
      model_v2=fake_model_shadow(), steering_pressed=False, steer_limited=False,
      yaw_rate=0.0,
    )[0]
    assert event is not None
    assert event["gatePass"] is False
    assert event["blockReason"] == "bad_speed"


def test_adapter_output_unchanged_with_logging_enabled_disabled(cloudlog_spy):
  """Logging must never affect the returned curvature, whether the pipeline is on or off."""
  model = fake_model(0.001)
  raw_curvature = 0.001

  disabled = LateralDemandAdapter(FakeParams(CustomLateralDemandEnabled=False, LateralDebugTraceMode="log"))
  disabled_out = disabled.process(True, 20.0, 0.0, raw_curvature, raw_curvature, model)
  assert disabled_out == raw_curvature
  assert disabled.last_result is None

  enabled = LateralDemandAdapter(FakeParams(CustomLateralDemandEnabled=True, LateralDebugTraceMode="log"))
  spy = SpyPipeline()
  setattr(enabled, "_pipeline", spy)
  enabled_out = enabled.process(True, 20.0, 0.0, raw_curvature, raw_curvature, model)
  assert enabled_out == raw_curvature
  assert spy.inputs is not None
  assert spy.inputs.desired_curvature == pytest.approx(raw_curvature, abs=1e-9)

  assert cloudlog_spy


def test_disabled_adapter_with_logging_does_not_touch_pipeline(cloudlog_spy):
  class RaisingPipeline:
    def update(self, _inputs):
      raise AssertionError("disabled adapter must not run the demand pipeline for logging")

  a = LateralDemandAdapter(FakeParams(CustomLateralDemandEnabled=False, LateralDebugTraceMode="log"))
  setattr(a, "_pipeline", RaisingPipeline())
  out = a.process(True, 20.0, 0.0, 0.02, 0.02, fake_model_shadow(), steering_pressed=False)
  assert out == 0.02
  assert a.last_result is None
  assert any(name == "lateral_path_shadow" for name, _ in cloudlog_spy)


def test_logging_exception_is_swallowed(cloudlog_spy, monkeypatch):
  def _explode(_name: str, **_kwargs: Any):
    raise RuntimeError("logging must not propagate")

  monkeypatch.setattr(_wiring.cloudlog, "event", _explode)

  a = LateralDemandAdapter(FakeParams(CustomLateralDemandEnabled=True, LateralDebugTraceMode="log"))
  spy = SpyPipeline()
  setattr(a, "_pipeline", spy)
  raw_curvature = 0.001
  out = a.process(True, 20.0, 0.0, raw_curvature, raw_curvature, fake_model_shadow())
  assert out == raw_curvature
  assert spy.inputs is not None


def test_shadow_trace_includes_yaw_curvature_when_available(cloudlog_spy):
  event = _shadow_trace_event(
    tick=100, last_trace_tick=0, debug_trace_mode="log",
    lat_active=True, v_ego=20.0, raw_curvature=0.001, measured_curvature=0.001,
    model_v2=fake_model_shadow(), steering_pressed=False, steer_limited=False,
    yaw_rate=0.02,
  )[0]
  assert event is not None
  assert event["gatePass"] is True
  assert event["yawCurvature"] == pytest.approx(0.02 / 20.0)


def test_shadow_trace_omits_yaw_curvature_when_unavailable(cloudlog_spy):
  event = _shadow_trace_event(
    tick=100, last_trace_tick=0, debug_trace_mode="log",
    lat_active=True, v_ego=20.0, raw_curvature=0.001, measured_curvature=0.001,
    model_v2=fake_model_shadow(), steering_pressed=False, steer_limited=False,
    yaw_rate=None,
  )[0]
  assert event is not None
  assert event["gatePass"] is True
  assert "yawCurvature" not in event


def test_shadow_trace_has_strict_gate_fields(cloudlog_spy):
  event = _shadow_trace_event(
    tick=100, last_trace_tick=0, debug_trace_mode="log",
    lat_active=True, v_ego=20.0, raw_curvature=0.001, measured_curvature=0.001,
    model_v2=fake_model_shadow(), steering_pressed=False, steer_limited=False,
    yaw_rate=0.0,
  )[0]
  assert event is not None
  assert event["strictGatePass"] == event["gatePass"]
  assert event["strictBlockReason"] == event["blockReason"]


class RaisingAttributeLaneLine:
  @property
  def x(self):
    raise RuntimeError("bad lane line x")

  @property
  def y(self):
    raise RuntimeError("bad lane line y")


def test_shadow_trace_malformed_geometry_does_not_raise(cloudlog_spy):
  model = fake_model_shadow()
  model.laneLines = [RaisingAttributeLaneLine() for _ in range(4)]
  a = LateralDemandAdapter(FakeParams(CustomLateralDemandEnabled=False, LateralDebugTraceMode="log"))
  out = a.process(True, 20.0, 0.0, 0.0123, 0.0123, model, steering_pressed=False)
  assert out == 0.0123
  assert a.last_result is None


def test_shadow_trace_construction_exception_swallowed(cloudlog_spy, monkeypatch):
  def _explode(**_kwargs):
    raise RuntimeError("shadow event construction fault")

  monkeypatch.setattr(_wiring, "_shadow_trace_event", _explode)

  a = LateralDemandAdapter(FakeParams(CustomLateralDemandEnabled=True, LateralDebugTraceMode="log"))
  spy = SpyPipeline()
  setattr(a, "_pipeline", spy)
  raw_curvature = 0.002
  out = a.process(True, 20.0, 0.0, raw_curvature, raw_curvature, fake_model_shadow())
  assert out == raw_curvature
  assert spy.inputs is not None
  # fallback exception-marker event should still be emitted
  assert any(
    name == "lateral_path_shadow" and kwargs.get("strictBlockReason") == "exception"
    for name, kwargs in cloudlog_spy
  )
  assert a._last_trace_tick == 1

  before = len(cloudlog_spy)
  out = a.process(True, 20.0, 0.0, raw_curvature, raw_curvature, fake_model_shadow())
  assert out == raw_curvature
  assert len(cloudlog_spy) == before


def test_shadow_trace_exception_and_cloudlog_exception_swallowed(monkeypatch):
  def _explode(*_args, **_kwargs):
    raise RuntimeError("boom")

  monkeypatch.setattr(_wiring, "_shadow_trace_event", _explode)
  monkeypatch.setattr(_wiring.cloudlog, "event", _explode)

  a = LateralDemandAdapter(FakeParams(CustomLateralDemandEnabled=True, LateralDebugTraceMode="log"))
  spy = SpyPipeline()
  setattr(a, "_pipeline", spy)
  raw_curvature = 0.002
  out = a.process(True, 20.0, 0.0, raw_curvature, raw_curvature, fake_model_shadow())
  assert out == raw_curvature
  assert spy.inputs is not None
