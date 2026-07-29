"""Integration tests for the lateral controller fuzzer."""
from __future__ import annotations

import numpy as np

from openpilot.tools.drive_lab.fuzz_lateral_controller import (
    ControllerFuzzerConfig,
    ControllerFuzzThresholds,
    SCENARIO_GENERATORS,
    _FakeVM,
    _curvature_to_steering_deg,
    _make_controller,
    evaluate_scenario,
    generate_scenarios,
)


def test_all_scenario_kinds_have_generators():
    from openpilot.tools.drive_lab.fuzz_lateral_controller import CONTROLLER_SCENARIO_KINDS
    for kind in CONTROLLER_SCENARIO_KINDS:
        assert kind in SCENARIO_GENERATORS, f"missing generator for {kind}"


def test_generate_scenarios_produces_valid_scenarios():
    config = ControllerFuzzerConfig(seed=1, cases=6, duration_s=1.0)
    scenarios = generate_scenarios(config)
    assert len(scenarios) == 6
    for s in scenarios:
        assert s.duration_s == 1.0
        assert len(s.frames) > 0
        assert s.kind in SCENARIO_GENERATORS


def test_single_kind_generates_only_that_kind():
    config = ControllerFuzzerConfig(seed=1, cases=10, kind="steady_curve", duration_s=1.0)
    scenarios = generate_scenarios(config)
    assert len(scenarios) == 10
    assert all(s.kind == "steady_curve" for s in scenarios)


def test_curvature_to_steering_deg_inverts_fake_vm():
    vm = _FakeVM()
    for k in (-0.005, -0.001, 0.0, 0.001, 0.005):
        for v in (10.0, 20.0, 30.0):
            angle = _curvature_to_steering_deg(k, v)
            angle_rad = np.deg2rad(angle)
            k_recovered = vm.calc_curvature(angle_rad, v, 0.0)
            assert abs(k - k_recovered) < 1e-6, f"k={k:.4f} v={v:.1f} angle={angle:.2f} -> recovered k={k_recovered:.6f}"


def test_controller_never_crashes_on_random_inputs():
    """Controller must produce finite torque and never crash for random inputs."""
    controller = _make_controller()
    vm = _FakeVM()
    rng = np.random.default_rng(20260616)
    for _ in range(100):
        from types import SimpleNamespace
        v = float(rng.uniform(0, 35))
        k = float(rng.uniform(-0.01, 0.01))
        angle = _curvature_to_steering_deg(k, v)
        roll = float(rng.uniform(-0.1, 0.1))
        active = bool(rng.random() > 0.1)
        try:
            cs = SimpleNamespace(vEgo=v, steeringAngleDeg=angle, steeringRateDeg=0.0,
                                  steeringPressed=bool(rng.random() > 0.9))
            params = SimpleNamespace(roll=roll, angleOffsetDeg=0.0)
            out, _, pid = controller.update(active, cs, vm, params, bool(rng.random() > 0.8),
                                            k, None, bool(rng.random() > 0.9), 0.2)
            assert np.isfinite(out), f"non-finite output torque: {out}"
            assert abs(out) <= 1.0 + 1e-9, f"output exceeds steer_max: {out}"
        except Exception as e:
            raise AssertionError(f"controller crashed on v={v:.1f} k={k:.4f}: {e}")


def test_evaluate_scenario_produces_result():
    scenarios = generate_scenarios(ControllerFuzzerConfig(seed=1, cases=3, duration_s=0.5))
    for s in scenarios:
        result = evaluate_scenario(s)
        assert result.scenario == s
        assert len(result.outputs) > 0
        # Structural: output torque must always be finite and bounded.
        for o in result.outputs:
            assert np.isfinite(o.output_torque), f"non-finite torque at t={o.t:.2f}"
            assert abs(o.output_torque) <= 1.0 + 1e-9, f"torque {o.output_torque} exceeds limit"


def test_inactive_scenario_returns_zero_torque():
    """When active=False, the controller must return zero torque."""
    from types import SimpleNamespace
    controller = _make_controller()
    vm = _FakeVM()
    for _ in range(5):
        # Run active first to build state.
        cs = SimpleNamespace(vEgo=20.0, steeringAngleDeg=5.0, steeringRateDeg=0.0, steeringPressed=False)
        controller.update(True, cs, vm, SimpleNamespace(roll=0.0, angleOffsetDeg=0.0),
                          False, 0.002, None, False, 0.2)
    # Now deactivate.
    out, _, pid = controller.update(False, SimpleNamespace(vEgo=20.0, steeringAngleDeg=5.0,
                                     steeringRateDeg=0.0, steeringPressed=False),
                                     vm, SimpleNamespace(roll=0.0, angleOffsetDeg=0.0),
                                     False, 0.002, None, False, 0.2)
    assert out == 0.0
    assert pid.active is False


def test_thresholds_roundtrip():
    t = ControllerFuzzThresholds(max_abs_output_torque=1.5, max_oscillation_reversals=100)
    d = t.to_dict()
    t2 = ControllerFuzzThresholds.from_dict(d)
    assert t2.max_abs_output_torque == 1.5
    assert t2.max_oscillation_reversals == 100
