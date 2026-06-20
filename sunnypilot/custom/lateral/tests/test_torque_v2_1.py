"""Integration test for the LatControlTorqueV21 wiring.

Exercises the adapter (response_core -> extension -> governor, pid_log population, the
active/inactive paths) with fake car objects and a no-op extension. This validates the GLUE
— it does not certify feel, which requires engaged-route replay (see the ADR). The
components themselves are covered by test_response_core_parity and test_output_governor.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from cereal import log
from openpilot.sunnypilot.custom.lateral.torque_v2_1 import LatControlTorqueV21, VERSION_V21

DT = 0.01


class NoOpExtension:
  """Stands in for LatControlTorqueExt (NNLC/override) — passes torque through unchanged."""
  def update_override_torque_params(self, torque_params, v_ego=None) -> bool:
    return False

  def update(self, CS, VM, pid, params, ff, pid_log, *rest):
    return pid_log, rest[-1]  # rest[-1] is output_torque (last positional arg)


def make_torque_params():
  return SimpleNamespace(latAccelFactor=2.5, latAccelOffset=0.05, friction=0.1,
                         steeringAngleDeadzoneDeg=0.5)


def make_cp():
  torque = SimpleNamespace(as_builder=make_torque_params)
  return SimpleNamespace(steerLimitTimer=3.0, lateralTuning=SimpleNamespace(torque=torque))


def make_ci():
  return SimpleNamespace(
    torque_from_lateral_accel=lambda: (lambda la, tp: la / tp.latAccelFactor),
    lateral_accel_from_torque=lambda: (lambda t, tp: t * tp.latAccelFactor),
  )


class FakeVM:
  @staticmethod
  def calc_curvature(angle_rad, v_ego, roll):
    return angle_rad / (10.0 + 0.05 * v_ego * v_ego) - 0.02 * roll


def make_cs(v_ego=20.0, angle=5.0, rate=0.0, pressed=False):
  return SimpleNamespace(vEgo=v_ego, steeringAngleDeg=angle, steeringRateDeg=rate, steeringPressed=pressed)


def make_params(roll=0.0, angle_offset=0.0):
  return SimpleNamespace(roll=roll, angleOffsetDeg=angle_offset)


def make_controller():
  return LatControlTorqueV21(make_cp(), SimpleNamespace(), make_ci(), DT, extension=NoOpExtension())


def test_constructs_and_runs_bounded():
  c = make_controller()
  vm = FakeVM()
  rng = np.random.default_rng(20260613)
  for _ in range(1500):
    cs = make_cs(v_ego=float(rng.uniform(0, 35)), angle=float(rng.uniform(-90, 90)),
                 rate=float(rng.uniform(-40, 40)), pressed=bool(rng.random() > 0.85))
    params = make_params(roll=float(rng.uniform(-0.08, 0.08)))
    out, _, pid_log = c.update(True, cs, vm, params, bool(rng.random() > 0.8),
                               float(rng.uniform(-0.05, 0.05)), None, False, 0.2)
    assert math.isfinite(out)
    assert abs(out) <= c.steer_max + 1e-9
    assert pid_log.version == VERSION_V21
    assert pid_log.active is True


def test_inactive_returns_zero_and_resets_governor():
  c = make_controller()
  vm = FakeVM()
  for _ in range(30):
    c.update(True, make_cs(), vm, make_params(), False, 0.02, None, False, 0.2)
  assert c.governor.previous_output != 0.0
  out, zero, pid_log = c.update(False, make_cs(), vm, make_params(), False, 0.02, None, False, 0.2)
  assert out == 0.0
  assert zero == 0.0
  assert pid_log.active is False
  assert c.governor.previous_output == 0.0


def test_return_torque_is_negated_governor_output():
  # The controller returns -output_torque (upstream convention); with the no-op extension the
  # magnitude must equal the governor's output magnitude and stay within steer_max.
  c = make_controller()
  vm = FakeVM()
  out = 0.0
  for _ in range(100):
    out, _, _ = c.update(True, make_cs(v_ego=15.0, angle=20.0), vm, make_params(), False, 0.03, None, False, 0.2)
  assert abs(out) == pytest.approx(abs(c.governor.previous_output), abs=1e-9)


def test_populates_adaptive_torque_telemetry():
  c = make_controller()
  vm = FakeVM()
  out, _, pid_log = c.update(True, make_cs(v_ego=12.0, angle=15.0, rate=90.0, pressed=True), vm,
                             make_params(), True, 0.05, None, False, 0.2)

  adaptive = pid_log.adaptiveTorqueState
  assert pid_log.output == pytest.approx(out)
  assert adaptive.active is True
  assert adaptive.releaseActive is True
  assert abs(adaptive.nominalOutput) >= abs(pid_log.output)
  assert adaptive.unshapedOutput == pytest.approx(adaptive.nominalOutput)
  assert adaptive.outputCap <= 1.0
  assert adaptive.steerLimitLimited is True
  assert adaptive.steerLimitSameDirection is True
  assert adaptive.governorReason != 0
  assert adaptive.rawActualLateralAccel == pytest.approx(pid_log.actualLateralAccel)


def test_controller_passes_controller_evidence_to_governor():
  c = make_controller()
  vm = FakeVM()
  captured = {}
  original_update = c.governor.update

  def capture_update(inp):
    captured["inp"] = inp
    return original_update(inp)

  c.governor.update = capture_update
  c.update(True, make_cs(v_ego=12.0, angle=15.0), vm, make_params(), False, 0.05, None, False, 0.2)

  assert captured["inp"].controller_evidence_stable is True


def test_controller_passes_path_evidence_to_governor():
  c = make_controller()
  vm = FakeVM()
  captured = {}
  original_update = c.governor.update

  def capture_update(inp):
    captured["inp"] = inp
    return original_update(inp)

  c.governor.update = capture_update
  c.set_under_response_path_evidence(False)
  c.update(True, make_cs(v_ego=12.0, angle=15.0), vm, make_params(), False, 0.05, None, False, 0.2)

  assert captured["inp"].path_evidence_valid is False


@pytest.mark.parametrize("model_path, expected", [
  (None, True),
  (SimpleNamespace(gated=False, reason="ok"), True),
  (SimpleNamespace(gated=True, reason="ok"), False),
  (SimpleNamespace(gated=False, reason="low_lane_confidence"), False),
  (SimpleNamespace(gated=False, reason="high_path_std"), False),
  (SimpleNamespace(gated=False, reason="invalid_path"), False),
])
def test_path_evidence_from_lateral_demand_mapping(model_path, expected):
  c = make_controller()
  result = None if model_path is None else SimpleNamespace(model_path_result=model_path)
  c.set_under_response_path_evidence_from_lateral_demand(result)
  assert c._under_response_path_evidence_valid is expected


def test_lateral_observability_schema_fields_are_writable():
  msg = log.ControlsState.new_message()
  msg.modelPathState.active = True
  msg.modelPathState.reason = "ok"
  msg.modelPathState.rawDesiredCurvature = 0.001
  torque_state = msg.lateralControlState.init('torqueState')
  torque_state.version = VERSION_V21
  torque_state.adaptiveTorqueState.governorReason = 1 << 9
  torque_state.adaptiveTorqueState.lowSpeedOutputMax = True

  assert msg.modelPathState.reason == "ok"
  assert torque_state.adaptiveTorqueState.governorReason == 1 << 9
  assert torque_state.adaptiveTorqueState.lowSpeedOutputMax is True


def test_live_torque_params_update_limits():
  c = make_controller()
  c.update_live_torque_params(3.0, 0.1, 0.2)
  assert c.torque_params.latAccelFactor == 3.0
  assert c.torque_params.friction == 0.2
  # PID limits track lateral_accel_from_torque(steer_max) = steer_max * latAccelFactor
  assert c.response_core.pid.pos_limit == pytest.approx(3.0)
  assert c.response_core.pid.neg_limit == pytest.approx(-3.0)
