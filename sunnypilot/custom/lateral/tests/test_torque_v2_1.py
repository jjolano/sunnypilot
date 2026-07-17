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
from openpilot.sunnypilot.selfdrive.controls.lib.underresponse_sentinel import BLOCK_INACTIVE, BLOCK_STEERING_PRESSED
from openpilot.sunnypilot.custom.lateral.output_governor import GovernorReason, OutputGovernorDiagnostics, OutputGovernorResult
from openpilot.sunnypilot.custom.lateral.response_core import ROLL_COMPENSATION_GAIN
from openpilot.sunnypilot.custom.lateral.torque_v2_1 import LatControlTorqueV21, VERSION_V21

DT = 0.01


class NoOpExtension:
  """Stands in for LatControlTorqueExt (NNLC/override) — passes torque through unchanged."""
  def update_override_torque_params(self, torque_params, v_ego=None) -> bool:
    return False

  def update(self, CS, VM, pid, params, ff, pid_log, *rest):
    return pid_log, rest[-1]  # rest[-1] is output_torque (last positional arg)


class BadTorqueExtension(NoOpExtension):
  def update(self, CS, VM, pid, params, ff, pid_log, *rest):
    return pid_log, None


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


def make_pose():
  return SimpleNamespace()


def make_controller():
  return LatControlTorqueV21(make_cp(), SimpleNamespace(), make_ci(), DT, extension=NoOpExtension())


def make_bad_extension_controller():
  return LatControlTorqueV21(make_cp(), SimpleNamespace(), make_ci(), DT, extension=BadTorqueExtension())


def test_constructs_and_runs_bounded():
  c = make_controller()
  vm = FakeVM()
  rng = np.random.default_rng(20260613)
  for _ in range(1500):
    cs = make_cs(v_ego=float(rng.uniform(0, 35)), angle=float(rng.uniform(-90, 90)),
                 rate=float(rng.uniform(-40, 40)), pressed=bool(rng.random() > 0.85))
    params = make_params(roll=float(rng.uniform(-0.08, 0.08)))
    out, _, pid_log = c.update(True, cs, vm, params, bool(rng.random() > 0.8),
                               float(rng.uniform(-0.05, 0.05)), make_pose(), False, 0.2)  # type: ignore[arg-type]
    assert math.isfinite(out)
    assert abs(out) <= c.steer_max + 1e-9
    assert pid_log.version == VERSION_V21
    assert pid_log.active is True


def test_inactive_returns_zero_and_resets_governor():
  c = make_controller()
  vm = FakeVM()
  for _ in range(30):
    c.update(True, make_cs(), vm, make_params(), False, 0.02, make_pose(), False, 0.2)  # type: ignore[arg-type]
  assert c.governor.previous_output != 0.0
  out, zero, pid_log = c.update(False, make_cs(), vm, make_params(), False, 0.02, make_pose(), False, 0.2)  # type: ignore[arg-type]
  assert out == 0.0
  assert zero == 0.0
  assert pid_log.active is False
  assert c.governor.previous_output == 0.0
  assert pid_log.underresponseActive is False
  assert pid_log.underresponseBlockMask & BLOCK_INACTIVE


def test_return_torque_is_negated_governor_output():
  # The controller returns -output_torque (upstream convention); with the no-op extension the
  # magnitude must equal the governor's output magnitude and stay within steer_max.
  c = make_controller()
  vm = FakeVM()
  out = 0.0
  for _ in range(100):
    out, _, _ = c.update(True, make_cs(v_ego=15.0, angle=20.0), vm, make_params(), False, 0.03, make_pose(), False, 0.2)  # type: ignore[arg-type]
  assert abs(out) == pytest.approx(abs(c.governor.previous_output), abs=1e-9)


def test_populates_adaptive_torque_telemetry():
  c = make_controller()
  vm = FakeVM()
  out, _, pid_log = c.update(True, make_cs(v_ego=12.0, angle=15.0, rate=90.0, pressed=True), vm,
                             make_params(), True, 0.05, make_pose(), False, 0.2)  # type: ignore[arg-type]

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
  assert math.isfinite(adaptive.responseCoreError)
  assert adaptive.responseCoreMeasurementReset is True
  assert adaptive.responseCoreSameSignUnwind is False
  assert adaptive.responseCoreFreezeIntegrator is True
  assert math.isfinite(adaptive.responseCoreFf)


def test_same_direction_limit_requires_tracking_correction_direction():
  c = make_controller()
  assert c._same_direction_limit(True, 0.5, 1.0, 0.5) is True
  assert c._same_direction_limit(True, -0.5, 1.0, 0.5) is False
  assert c._same_direction_limit(True, 0.5, 0.5, 1.0) is False
  assert c._same_direction_limit(False, 0.5, 1.0, 0.5) is False
  assert c._same_direction_limit(True, None, 1.0, 0.5) is False


def test_same_direction_limit_uses_requested_and_applied_torque_signs_when_available():
  c = make_controller()
  # set_steer_limited_output_context receives actuator-sign torque from controlsd; the
  # controller's nominal_torque argument below is the opposite internal governor sign.
  c.set_steer_limited_output_context(-0.8, -0.6)
  assert c._same_direction_limit(True, 0.5, 1.0, 0.5) is True

  c.set_steer_limited_output_context(-0.8, 0.2)
  assert c._same_direction_limit(True, 0.5, 1.0, 0.5) is False

  c.set_steer_limited_output_context(0.8, 0.6)
  assert c._same_direction_limit(True, 0.5, 1.0, 0.5) is False


def test_same_direction_limit_accepts_returned_actuator_torque_context():
  c = make_controller()
  vm = FakeVM()

  returned_torque, _, _ = c.update(True, make_cs(v_ego=12.0, angle=-20.0), vm,
                                  make_params(), False, 0.08, make_pose(), False, 0.2)  # type: ignore[arg-type]
  internal_nominal = c.governor.previous_output

  c.set_steer_limited_output_context(returned_torque, returned_torque * 0.8)

  assert returned_torque == pytest.approx(-internal_nominal)
  assert c._same_direction_limit(True, internal_nominal, 1.0, 0.5) is True


def test_populates_underresponse_shadow_telemetry_without_changing_output():
  c = make_controller()
  vm = FakeVM()

  out, _, pid_log = c.update(True, make_cs(v_ego=20.0, angle=15.0, pressed=True), vm,
                             make_params(), False, 0.04, make_pose(), False, 0.2)  # type: ignore[arg-type]

  assert pid_log.output == pytest.approx(out)
  assert pid_log.underresponseActive is False
  assert pid_log.underresponseEligible is False
  assert pid_log.underresponseBlockMask & BLOCK_STEERING_PRESSED
  assert math.isfinite(pid_log.underresponseError)
  assert math.isfinite(pid_log.underresponseShadowLatAccel)


def test_populates_oscillation_classification_in_active_path():
  c = make_controller()
  vm = FakeVM()
  _, _, pid_log = c.update(True, make_cs(v_ego=20.0, angle=15.0, pressed=True), vm,
                           make_params(), False, 0.04, make_pose(), False, 0.2)  # type: ignore[arg-type]
  assert hasattr(pid_log.adaptiveTorqueState, 'oscillationClassification')
  assert isinstance(pid_log.adaptiveTorqueState.oscillationClassification, int)


def test_oscillation_observer_resets_on_inactive():
  c = make_controller()
  vm = FakeVM()
  for _ in range(20):
    c.update(True, make_cs(v_ego=20.0, angle=0.0, rate=0.0), vm,
             make_params(), False, 0.0, make_pose(), False, 0.2)  # type: ignore[arg-type]
  c.update(False, make_cs(v_ego=20.0), vm, make_params(), False, 0.0, make_pose(), False, 0.2)  # type: ignore[arg-type]
  assert len(c.oscillation_observer.torque_history) == 0
  assert c.oscillation_observer.last_debug.classification == 0


def test_near_zero_observer_resets_on_inactive():
  c = make_controller()
  vm = FakeVM()
  c.near_zero_recenter_observer.update(
    active=True, v_ego=20.0, steering_pressed=False, steer_limited_by_safety=False,
    curvature_limited=False, desired_lateral_accel=-0.04, actual_lateral_accel=0.10,
    steering_rate_deg=0.0, output_torque=-0.2, steer_max=1.0,
  )
  assert c.near_zero_recenter_observer._duration > 0.0

  c.update(False, make_cs(v_ego=20.0), vm, make_params(), False, 0.0, make_pose(), False, 0.2)  # type: ignore[arg-type]

  assert c.near_zero_recenter_observer._duration == 0.0


def test_controller_passes_controller_evidence_to_governor():
  c = make_controller()
  vm = FakeVM()
  captured = {}
  original_update = c.governor.update

  def capture_update(inp):
    captured["inp"] = inp
    return original_update(inp)

  c.governor.update = capture_update
  c.update(True, make_cs(v_ego=12.0, angle=15.0), vm, make_params(), False, 0.05, make_pose(), False, 0.2)  # type: ignore[arg-type]

  assert captured["inp"].controller_evidence_stable is True


def test_controller_passes_relative_arrival_inputs_to_governor():
  c = make_controller()
  vm = FakeVM()
  captured = {}
  original_update = c.governor.update

  def capture_update(inp):
    captured["inp"] = inp
    return original_update(inp)

  c.governor.update = capture_update
  c.update(True, make_cs(v_ego=12.0, angle=15.0), vm, make_params(), False, 0.05, make_pose(), False, 0.2)  # type: ignore[arg-type]

  assert math.isfinite(captured["inp"].lateral_accel_error_rate)
  assert captured["inp"].lat_delay == pytest.approx(0.2)
  assert math.isfinite(captured["inp"].holding_torque)


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
  c.update(True, make_cs(v_ego=12.0, angle=15.0), vm, make_params(), False, 0.05, make_pose(), False, 0.2)  # type: ignore[arg-type]

  assert captured["inp"].path_evidence_valid is False


def test_path_evidence_from_lateral_demand_mapping():
  c = make_controller()
  c.set_under_response_path_evidence_from_lateral_demand(
    SimpleNamespace(model_path_result=SimpleNamespace(gated=False, reason="ok"))
  )
  assert c._under_response_path_evidence_valid is True


def test_path_evidence_from_lateral_demand_fail_closed_when_expected_but_missing():
  c = make_controller()
  c.set_under_response_path_evidence_from_lateral_demand(None)
  assert c._under_response_path_evidence_valid is False


def test_path_evidence_from_lateral_demand_valid_when_inactive_or_not_expected():
  c = make_controller()
  c.set_under_response_path_evidence_from_lateral_demand(None, active=False)
  assert c._under_response_path_evidence_valid is True
  c.set_under_response_path_evidence_from_lateral_demand(None, active=True, evidence_expected=False)
  assert c._under_response_path_evidence_valid is True


def test_missing_evidence_is_valid_when_lateral_demand_pipeline_disabled():
  """Regression: controlsd only expects path evidence when CustomLateralDemandEnabled is on.

  When the custom lateral demand pipeline is disabled (evidence_expected=False), a missing
  evidence object must not fail-closed. It must still fail-closed when evidence is expected.
  """
  c = make_controller()
  c.set_under_response_path_evidence_from_lateral_demand(None, active=True, evidence_expected=False)
  assert c._under_response_path_evidence_valid is True
  c.set_under_response_path_evidence_from_lateral_demand(None, active=True, evidence_expected=True)
  assert c._under_response_path_evidence_valid is False


@pytest.mark.parametrize("model_path", [
  SimpleNamespace(gated=True, reason="ok"),
  SimpleNamespace(gated=False, reason="low_lane_confidence"),
  SimpleNamespace(gated=False, reason="high_path_std"),
  SimpleNamespace(gated=False, reason="invalid_path"),
])
def test_path_evidence_from_lateral_demand_invalid_mapping(model_path):
  c = make_controller()
  c.set_under_response_path_evidence_from_lateral_demand(SimpleNamespace(model_path_result=model_path))
  assert c._under_response_path_evidence_valid is False


def test_lateral_observability_schema_fields_are_writable():
  msg = log.ControlsState.new_message()
  msg.modelPathState.active = True
  msg.modelPathState.reason = "ok"
  msg.modelPathState.rawDesiredCurvature = 0.001
  torque_state = msg.lateralControlState.init('torqueState')
  torque_state.version = VERSION_V21
  torque_state.adaptiveTorqueState.governorReason = 1 << 9
  torque_state.adaptiveTorqueState.lowSpeedOutputMax = True
  torque_state.adaptiveTorqueState.signConflictActive = True
  torque_state.adaptiveTorqueState.signConflictBinding = True
  torque_state.adaptiveTorqueState.signConflictFloorGuarded = True
  torque_state.adaptiveTorqueState.nearZeroRecenterConflict = True
  torque_state.adaptiveTorqueState.nearZeroRecenterError = -0.12
  torque_state.adaptiveTorqueState.nearZeroRecenterClosingRate = 0.3
  torque_state.adaptiveTorqueState.nearZeroRecenterDuration = 0.2
  torque_state.adaptiveTorqueState.underResponseGuardPathEvidenceInvalid = True
  torque_state.adaptiveTorqueState.underResponseGuardControllerUnstable = True
  torque_state.adaptiveTorqueState.underResponseGuardRelease = True
  torque_state.adaptiveTorqueState.underResponseGuardSameDirectionLimit = True
  torque_state.adaptiveTorqueState.underResponseGuardHighSteeringRate = True
  torque_state.adaptiveTorqueState.underResponseGuardSignConflict = True
  torque_state.adaptiveTorqueState.underResponseGuardOverResponse = True
  torque_state.adaptiveTorqueState.underResponseGuardIsoAccel = True
  torque_state.adaptiveTorqueState.underResponseGuardTorqueFraction = True
  torque_state.adaptiveTorqueState.responseCoreError = 0.34
  torque_state.adaptiveTorqueState.responseCoreMeasurementReset = True
  torque_state.adaptiveTorqueState.responseCoreSameSignUnwind = True
  torque_state.adaptiveTorqueState.responseCoreFreezeIntegrator = True
  torque_state.adaptiveTorqueState.responseCoreFf = 0.12

  assert msg.modelPathState.reason == "ok"
  assert torque_state.adaptiveTorqueState.governorReason == 1 << 9
  assert torque_state.adaptiveTorqueState.lowSpeedOutputMax is True
  assert torque_state.adaptiveTorqueState.signConflictActive is True
  assert torque_state.adaptiveTorqueState.nearZeroRecenterConflict is True
  assert torque_state.adaptiveTorqueState.underResponseGuardPathEvidenceInvalid is True
  assert torque_state.adaptiveTorqueState.underResponseGuardTorqueFraction is True
  assert torque_state.adaptiveTorqueState.responseCoreError == pytest.approx(0.34)
  assert torque_state.adaptiveTorqueState.responseCoreMeasurementReset is True
  assert torque_state.adaptiveTorqueState.responseCoreSameSignUnwind is True
  assert torque_state.adaptiveTorqueState.responseCoreFreezeIntegrator is True
  assert torque_state.adaptiveTorqueState.responseCoreFf == pytest.approx(0.12)


def test_near_zero_observer_receives_pre_governor_torque():
  c = make_controller()
  vm = FakeVM()
  captured = {}
  original_governor_update = c.governor.update

  def capture_governor(inp):
    captured["nominal_torque"] = inp.nominal_torque
    return original_governor_update(inp)

  def capture_near_zero(**kwargs):
    captured["near_zero_torque"] = kwargs["output_torque"]
    return SimpleNamespace(conflict=False, error=0.0, closingRate=0.0, duration=0.0)

  c.governor.update = capture_governor
  c.near_zero_recenter_observer.update = capture_near_zero
  c.update(True, make_cs(v_ego=20.0, angle=15.0), vm, make_params(), False, 0.04, make_pose(), False, 0.2)  # type: ignore[arg-type]

  assert captured["near_zero_torque"] == pytest.approx(captured["nominal_torque"], abs=1e-9)


def test_copies_shadow_diagnostics_to_adaptive_torque_state():
  c = make_controller()
  vm = FakeVM()

  def fake_governor_update(_inp):
    return OutputGovernorResult(
      output_torque=0.0,
      active=True,
      reason=int(GovernorReason.SIGN_CONFLICT),
      cap=0.8,
      floor=0.0,
      diagnostics=OutputGovernorDiagnostics(
        True, True, True,
        underResponseGuardHighSteeringRate=True,
        underResponseGuardSignConflict=True,
      ),
    )

  def fake_near_zero(**_kwargs):
    return SimpleNamespace(conflict=True, error=-0.12, closingRate=0.3, duration=0.2)

  c.governor.update = fake_governor_update
  c.near_zero_recenter_observer.update = fake_near_zero
  _, _, pid_log = c.update(True, make_cs(v_ego=20.0, angle=15.0), vm, make_params(), False, 0.04, make_pose(), False, 0.2)  # type: ignore[arg-type]

  adaptive = pid_log.adaptiveTorqueState
  assert adaptive.signConflictActive is True
  assert adaptive.signConflictBinding is True
  assert adaptive.signConflictFloorGuarded is True
  assert adaptive.underResponseGuardHighSteeringRate is True
  assert adaptive.underResponseGuardSignConflict is True
  assert adaptive.underResponseGuardPathEvidenceInvalid is False
  assert adaptive.underResponseGuardTorqueFraction is False
  assert adaptive.nearZeroRecenterConflict is True
  assert adaptive.nearZeroRecenterError == pytest.approx(-0.12)
  assert adaptive.nearZeroRecenterClosingRate == pytest.approx(0.3)
  assert adaptive.nearZeroRecenterDuration == pytest.approx(0.2)


def test_live_torque_params_update_limits():
  c = make_controller()
  c.update_live_torque_params(3.0, 0.1, 0.2)
  assert c.torque_params.latAccelFactor == 3.0
  assert c.torque_params.friction == 0.2
  # PID limits track lateral_accel_from_torque(steer_max) = steer_max * latAccelFactor
  assert c.response_core.pid.pos_limit == pytest.approx(3.0)
  assert c.response_core.pid.neg_limit == pytest.approx(-3.0)


def test_bad_extension_torque_fails_closed_without_crashing():
  c = make_bad_extension_controller()
  vm = FakeVM()

  out, _, pid_log = c.update(
    True, make_cs(v_ego=20.0, angle=10.0), vm,
    make_params(), False, 0.02, make_pose(), False, 0.2,
  )  # type: ignore[arg-type]

  assert out == 0.0
  assert pid_log.output == 0.0
  assert pid_log.active is True
  assert pid_log.adaptiveTorqueState.governorReason & GovernorReason.INVALID
  assert c.governor.previous_output == 0.0


@pytest.mark.parametrize("desired_curvature", [0.005, -0.005])
def test_controller_sign_contract_for_desired_curvature(desired_curvature):
  """Positive desired curvature must produce positive internal/governor torque and a negative
  returned actuator torque; negative curvature must produce the opposite sign. pid_log.output must equal the
  returned torque."""
  c = make_controller()
  vm = FakeVM()
  last_out = 0.0
  last_pid_log = None
  # Run long enough for slew and response-core state to settle; high speed avoids low-speed
  # under-response floor and low-speed unwind confounds.
  for _ in range(300):
    last_out, _, last_pid_log = c.update(
      True, make_cs(v_ego=25.0, angle=0.0, rate=0.0, pressed=False), vm,
      make_params(roll=0.0, angle_offset=0.0), False, desired_curvature, make_pose(), False, 0.2,
    )  # type: ignore[arg-type]

  expected_internal_sign = 1.0 if desired_curvature > 0 else -1.0
  # Internal governor torque retains the command-side sign.
  assert c.governor.previous_output != 0.0
  assert math.copysign(1.0, c.governor.previous_output) == expected_internal_sign
  # Returned actuator torque is negated (upstream convention).
  assert last_out != 0.0
  assert math.copysign(1.0, last_out) == -expected_internal_sign
  # Logged output matches the returned actuator torque.
  assert last_pid_log is not None
  assert last_pid_log.output == pytest.approx(last_out, abs=1e-9)


@pytest.mark.parametrize("steering_rate", [5.0, -10.0, 25.0])
def test_steering_rate_is_negated_before_governor(steering_rate):
  """The governor receives the steering rate with the sign flipped from carState."""
  c = make_controller()
  vm = FakeVM()
  captured = {}
  original_update = c.governor.update

  def capture_update(inp):
    captured["inp"] = inp
    return original_update(inp)

  c.governor.update = capture_update
  c.update(
    True, make_cs(v_ego=20.0, angle=0.0, rate=steering_rate, pressed=False), vm,
    make_params(), False, 0.0, make_pose(), False, 0.2,
  )  # type: ignore[arg-type]

  assert captured["inp"].steering_rate_deg == pytest.approx(-steering_rate, abs=1e-9)


def test_default_extension_uses_constant_roll_compensation_gain():
  c = make_controller()
  assert c.response_core.roll_compensation_gain == ROLL_COMPENSATION_GAIN
  c.update(True, make_cs(v_ego=20.0, angle=5.0), FakeVM(), make_params(roll=-0.04), False, 0.0, make_pose(), False, 0.2)  # type: ignore[arg-type]
  assert c.response_core.roll_compensation_gain == ROLL_COMPENSATION_GAIN


class RollGainExtension(NoOpExtension):
  def __init__(self, learned_roll_gain):
    self.learned_roll_gain = learned_roll_gain


def test_valid_learned_roll_gain_copied_to_response_core():
  c = LatControlTorqueV21(make_cp(), SimpleNamespace(), make_ci(), DT, extension=RollGainExtension(0.62))
  assert c.response_core.roll_compensation_gain == ROLL_COMPENSATION_GAIN
  c.update(True, make_cs(v_ego=20.0, angle=5.0), FakeVM(), make_params(roll=-0.04), False, 0.0, make_pose(), False, 0.2)  # type: ignore[arg-type]
  assert c.response_core.roll_compensation_gain == 0.62


class SpeedResolvedRollGainExtension(NoOpExtension):
  def __init__(self, gain):
    self.gain = gain
    self.learned_roll_gain = 0.99  # must be ignored when the speed-resolved hook exists
    self.calls = []

  def learned_roll_gain_at(self, v_ego, base_gain):
    self.calls.append((v_ego, base_gain))
    return self.gain


def test_speed_resolved_roll_gain_preferred_over_scalar():
  ext = SpeedResolvedRollGainExtension(0.44)
  c = LatControlTorqueV21(make_cp(), SimpleNamespace(), make_ci(), DT, extension=ext)
  c.update(True, make_cs(v_ego=12.0, angle=5.0), FakeVM(), make_params(roll=-0.04), False, 0.0, make_pose(), False, 0.2)  # type: ignore[arg-type]
  assert c.response_core.roll_compensation_gain == 0.44
  assert ext.calls[-1] == (12.0, ROLL_COMPENSATION_GAIN)


def test_speed_resolved_roll_gain_none_falls_back_to_constant():
  ext = SpeedResolvedRollGainExtension(None)
  c = LatControlTorqueV21(make_cp(), SimpleNamespace(), make_ci(), DT, extension=ext)
  c.update(True, make_cs(v_ego=12.0, angle=5.0), FakeVM(), make_params(roll=-0.04), False, 0.0, make_pose(), False, 0.2)  # type: ignore[arg-type]
  assert c.response_core.roll_compensation_gain == ROLL_COMPENSATION_GAIN
