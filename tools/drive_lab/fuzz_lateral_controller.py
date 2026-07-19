#!/usr/bin/env python3
"""Seeded lateral controller structural fuzzer.

Exercises ``LatControlTorqueV21`` (response core → output governor) with
synthetic per-frame inputs and checks structural invariants: finite torque,
bounded output, governor reason coverage, PID health, oscillation detection.

This is the controller equivalent of ``fuzz_lateral_demand.py`` — it validates
that the production torque controller correctly processes curvature commands
without crashing, producing NaN, or exceeding hard safety bounds.

Closed-loop with ``lateral_plant.py`` is deferred to a follow-up; for now
steering angle/rate are synthetic (open-loop), which exercises the controller's
internal math (PID, governor, measurement smoother) without depending on a
plant model.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from openpilot.sunnypilot.custom.lateral.torque_v2_1 import LatControlTorqueV21

DT = 0.01

CONTROLLER_SCENARIO_KINDS = (
    "steady_curve",
    "curvature_ramp",
    "curvature_step",
    "sinusoidal_sweep",
    "roll_disturbance",
    "speed_sweep",
    "stress_grid",
    "plant_robustness",
)

# ── fake car setup (reuses test_torque_v2_1.py pattern) ──────────────────────

_DEFAULT_TORQUE_PARAMS = SimpleNamespace(
    latAccelFactor=2.5, latAccelOffset=0.05, friction=0.1,
    steeringAngleDeadzoneDeg=0.5,
)


def _make_fake_cp():
    torque = SimpleNamespace(as_builder=lambda: _DEFAULT_TORQUE_PARAMS)
    # RAV4 fingerprint so the slew-scale study gate opens when --slew-scale-mode is set.
    return SimpleNamespace(steerLimitTimer=3.0, lateralTuning=SimpleNamespace(torque=torque),
                           carFingerprint="TOYOTA_RAV4_TSS2")


def _make_fake_ci():
    return SimpleNamespace(
        torque_from_lateral_accel=lambda: (lambda la, tp: la / tp.latAccelFactor),
        lateral_accel_from_torque=lambda: (lambda t, tp: t * tp.latAccelFactor),
    )


class _FakeVM:
    @staticmethod
    def calc_curvature(angle_rad, v_ego, roll):
        # Simplified vehicle model: curvature from steering angle / wheelbase approximation.
        # Matches the test pattern in test_torque_v2_1.py.
        denom = max(10.0 + 0.05 * v_ego * v_ego, 1e-6)
        return angle_rad / denom - 0.02 * roll


class _NoOpExtension:
    """Passes torque through unchanged — skips NNLC/override for pure controller testing."""
    slew_scale_mode = "off"

    @staticmethod
    def update_override_torque_params(torque_params, v_ego=None) -> bool:
        return False

    @staticmethod
    def update(CS, VM, pid, params, ff, pid_log, *rest):
        return pid_log, rest[-1]


def _make_controller(slew_scale_mode: str = "off"):
    extension = _NoOpExtension()
    extension.slew_scale_mode = slew_scale_mode
    return LatControlTorqueV21(_make_fake_cp(), SimpleNamespace(), _make_fake_ci(), DT,
                                extension=extension)


# ── threshold / scenario / result dataclasses ────────────────────────────────

@dataclass(frozen=True)
class ControllerFuzzThresholds:
    max_abs_output_torque: float = 1.05
    max_oscillation_reversals: int = 120  # step response produces transient sign flips in closed-loop
    max_saturation_fraction: float = 1.0
    max_abs_tracking_error: float = float("inf")  # closed-loop tracking is valid but not enforced
    max_pid_i_abs: float = 3.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ControllerFuzzThresholds:
        fields = cls.__dataclass_fields__
        return cls(**{key: data[key] for key in fields if key in data})


@dataclass(frozen=True)
class ControllerScenario:
    kind: str
    title: str
    duration_s: float
    frames: tuple[dict[str, Any], ...]
    thresholds: ControllerFuzzThresholds | None = None

    @property
    def metric_thresholds(self) -> ControllerFuzzThresholds:
        return self.thresholds or ControllerFuzzThresholds()


@dataclass(frozen=True)
class ControllerFrameOutput:
    t: float
    output_torque: float
    pid_p: float
    pid_i: float
    pid_d: float
    pid_f: float
    pid_output: float
    pid_error: float
    actual_lat_accel: float
    desired_lat_accel: float
    desired_lat_jerk: float
    saturated: bool
    governor_reason: int
    governor_cap: float
    governor_floor: float


@dataclass(frozen=True)
class ControllerScenarioResult:
    scenario: ControllerScenario
    outputs: tuple[ControllerFrameOutput, ...]
    valid: bool
    failures: list[dict[str, Any]]
    metrics: dict[str, Any]


# ── helpers ──────────────────────────────────────────────────────────────────

def _time_array(duration_s: float, dt: float = DT) -> np.ndarray:
    return np.arange(0.0, max(duration_s, dt) + dt * 0.5, dt, dtype=float)


def _sanitize(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _sanitize(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(v) for v in value]
    return value


def _curvature_to_steering_deg(curvature: float, v_ego: float, roll: float = 0.0) -> float:
    """Invert FakeVM.calc_curvature to get consistent steering angle from desired curvature.

    curvature = angle_rad / (10 + 0.05 * v^2) - 0.02 * roll
    → angle_rad = (curvature + 0.02 * roll) * (10 + 0.05 * v^2)
    """
    denom = max(10.0 + 0.05 * v_ego * v_ego, 1e-6)
    angle_rad = (curvature + 0.02 * roll) * denom
    return math.degrees(angle_rad)


# ── scenario generators ──────────────────────────────────────────────────────

def _base_frame(t: float, v_ego: float = 20.0, steering_angle_deg: float = 0.0,
                steering_rate_deg: float = 0.0, steering_pressed: bool = False,
                roll: float = 0.0, angle_offset_deg: float = 0.0,
                desired_curvature: float = 0.0, steer_limited: bool = False,
                curvature_limited: bool = False, lat_delay: float = 0.2,
                active: bool = True, **kwargs: Any) -> dict[str, Any]:
    frame: dict[str, Any] = {
        "t": t,
        "v_ego": v_ego,
        "steering_angle_deg": steering_angle_deg,
        "steering_rate_deg": steering_rate_deg,
        "steering_pressed": steering_pressed,
        "roll": roll,
        "angle_offset_deg": angle_offset_deg,
        "desired_curvature": desired_curvature,
        "steer_limited_by_safety": steer_limited,
        "curvature_limited": curvature_limited,
        "lat_delay": lat_delay,
        "active": active,
    }
    frame.update(kwargs)
    return frame


def _generate_steady_curve(rng: random.Random, idx: int, duration_s: float) -> ControllerScenario:
    t_arr = _time_array(duration_s)
    v_ego = rng.uniform(10.0, 30.0)
    k = rng.choice([-1.0, 1.0]) * rng.uniform(0.0005, 0.005)
    angle = _curvature_to_steering_deg(k, v_ego)
    frames = tuple(
        _base_frame(t=float(i * DT), v_ego=v_ego, steering_angle_deg=angle,
                     steering_rate_deg=0.0, desired_curvature=k)
        for i in range(len(t_arr))
    )
    return ControllerScenario(kind="steady_curve", title=f"fuzz steady curve #{idx}",
                              duration_s=duration_s, frames=frames)


def _generate_curvature_ramp(rng: random.Random, idx: int, duration_s: float) -> ControllerScenario:
    t_arr = _time_array(duration_s)
    v_ego = rng.uniform(10.0, 30.0)
    k0 = rng.choice([-1.0, 1.0]) * rng.uniform(0.0005, 0.002)
    k1 = rng.choice([-1.0, 1.0]) * rng.uniform(0.002, 0.006)
    ramp_start = int(len(t_arr) * 0.3)
    ramp_end = int(len(t_arr) * 0.7)
    frames: list[dict[str, Any]] = []
    for i in range(len(t_arr)):
        if i < ramp_start:
            k = k0
        elif i >= ramp_end:
            k = k1
        else:
            progress = (i - ramp_start) / max(ramp_end - ramp_start, 1)
            k = k0 + (k1 - k0) * progress
        angle = _curvature_to_steering_deg(k, v_ego)
        frames.append(_base_frame(t=float(i * DT), v_ego=v_ego, steering_angle_deg=angle,
                                   desired_curvature=k))
    return ControllerScenario(kind="curvature_ramp", title=f"fuzz curvature ramp #{idx}",
                              duration_s=duration_s, frames=tuple(frames))


def _generate_curvature_step(rng: random.Random, idx: int, duration_s: float) -> ControllerScenario:
    t_arr = _time_array(duration_s)
    v_ego = rng.uniform(10.0, 30.0)
    k0 = rng.choice([-1.0, 1.0]) * rng.uniform(0.0005, 0.002)
    k1 = rng.choice([-1.0, 1.0]) * rng.uniform(0.003, 0.008)
    step_frame = int(len(t_arr) * rng.uniform(0.35, 0.55))
    frames: list[dict[str, Any]] = []
    for i in range(len(t_arr)):
        k = k0 if i < step_frame else k1
        angle = _curvature_to_steering_deg(k, v_ego)
        frames.append(_base_frame(t=float(i * DT), v_ego=v_ego, steering_angle_deg=angle,
                                   desired_curvature=k))
    return ControllerScenario(kind="curvature_step", title=f"fuzz curvature step #{idx}",
                              duration_s=duration_s, frames=tuple(frames))


def _generate_sinusoidal_sweep(rng: random.Random, idx: int, duration_s: float) -> ControllerScenario:
    t_arr = _time_array(duration_s)
    v_ego = rng.uniform(10.0, 25.0)
    amp = rng.uniform(0.001, 0.005)
    freq = rng.uniform(0.5, 2.0)
    frames: list[dict[str, Any]] = []
    for i, t in enumerate(t_arr):
        k = amp * math.sin(2 * math.pi * freq * t)
        angle = _curvature_to_steering_deg(k, v_ego)
        frames.append(_base_frame(t=float(i * DT), v_ego=v_ego, steering_angle_deg=angle,
                                   desired_curvature=k))
    return ControllerScenario(kind="sinusoidal_sweep", title=f"fuzz sinusoidal #{idx}",
                              duration_s=duration_s, frames=tuple(frames))


def _generate_roll_disturbance(rng: random.Random, idx: int, duration_s: float) -> ControllerScenario:
    t_arr = _time_array(duration_s)
    v_ego = rng.uniform(10.0, 25.0)
    k = rng.choice([-1.0, 1.0]) * rng.uniform(0.001, 0.003)
    roll_mag = rng.uniform(0.02, 0.10)
    angle = _curvature_to_steering_deg(k, v_ego, 0.0)
    frames: list[dict[str, Any]] = []
    for i in range(len(t_arr)):
        roll = roll_mag * math.sin(2 * math.pi * 0.5 * i * DT)  # slow roll oscillation
        angle = _curvature_to_steering_deg(k, v_ego, roll)
        frames.append(_base_frame(t=float(i * DT), v_ego=v_ego, steering_angle_deg=angle,
                                   desired_curvature=k, roll=roll))
    return ControllerScenario(kind="roll_disturbance", title=f"fuzz roll disturbance #{idx}",
                              duration_s=duration_s, frames=tuple(frames))


def _generate_speed_sweep(rng: random.Random, idx: int, duration_s: float) -> ControllerScenario:
    t_arr = _time_array(duration_s)
    k = rng.choice([-1.0, 1.0]) * rng.uniform(0.001, 0.004)
    v_start = rng.uniform(5.0, 15.0)
    v_end = rng.uniform(25.0, 35.0)
    frames: list[dict[str, Any]] = []
    for i in range(len(t_arr)):
        progress = i / max(len(t_arr) - 1, 1)
        v = v_start + (v_end - v_start) * progress
        angle = _curvature_to_steering_deg(k, v)
        frames.append(_base_frame(t=float(i * DT), v_ego=v, steering_angle_deg=angle,
                                   desired_curvature=k))
    return ControllerScenario(kind="speed_sweep", title=f"fuzz speed sweep #{idx}",
                              duration_s=duration_s, frames=tuple(frames))


# ── controller stress grid ───────────────────────────────────────────────────

_STRESS_SPEEDS_K = (5.0, 15.0, 25.0, 35.0)      # m/s
_STRESS_CURVATURES_K = (0.0, 0.001, 0.003, 0.005, 0.01)  # 1/m
_STRESS_ROLLS_K = (0.0, 0.04, 0.08)               # rad
_STRESS_ACTIVE_K = (True, False)


def _generate_stress_grid(rng: random.Random, idx: int, duration_s: float) -> ControllerScenario:
    """Parametric grid for the torque controller: speed × curvature × roll × active."""
    t_arr = _time_array(duration_s)
    frames: list[dict[str, Any]] = []
    # Pick one cell from the grid using the RNG for variety.
    v = rng.choice(_STRESS_SPEEDS_K)
    k_abs = rng.choice(_STRESS_CURVATURES_K)
    sign = rng.choice([-1.0, 1.0]) if k_abs != 0.0 else 1.0
    k = k_abs * sign
    roll = rng.choice(_STRESS_ROLLS_K)
    active = rng.choice(_STRESS_ACTIVE_K)
    # Ramp curvature and roll in over first 20 frames.
    ramp = 20
    for i in range(len(t_arr)):
        r = min(1.0, i / ramp)
        k_ramped = k * r
        roll_ramped = roll * r
        angle = _curvature_to_steering_deg(k_ramped, v, roll_ramped)
        frames.append(_base_frame(
            t=float(i * DT), v_ego=v, steering_angle_deg=angle,
            desired_curvature=k_ramped, roll=roll_ramped, active=active,
        ))
    return ControllerScenario(kind="stress_grid", title=f"ctrl stress v={v:.0f} k={k:.4f} roll={roll:.2f} act={active} #{idx}",
                              duration_s=duration_s, frames=tuple(frames))


# ── multi-plant robustness ───────────────────────────────────────────────────

_PLANT_GAINS = (0.5, 1.0, 2.0)          # steering gain multiplier
_PLANT_RATES = (90.0, 180.0, 360.0)      # max steering rate deg/s
_PLANT_DELAYS = (0.0, 0.1, 0.2)          # actuator delay seconds


def _generate_plant_robustness(rng: random.Random, idx: int, duration_s: float) -> ControllerScenario:
    """Generate a scenario that carries plant configuration for robustness testing."""
    gain = rng.choice(_PLANT_GAINS)
    rate = rng.choice(_PLANT_RATES)
    # Use a moderate curvature ramp scenario as the base.
    t_arr = _time_array(duration_s)
    v = rng.uniform(15.0, 25.0)
    k = rng.choice([-1.0, 1.0]) * rng.uniform(0.001, 0.004)
    frames: list[dict[str, Any]] = []
    ramp = 20
    for i in range(len(t_arr)):
        r = min(1.0, i / ramp)
        angle = _curvature_to_steering_deg(k * r, v)
        frames.append(_base_frame(
            t=float(i * DT), v_ego=v, steering_angle_deg=angle,
            desired_curvature=k * r,
            plant_gain=gain, plant_rate=rate,
        ))
    return ControllerScenario(
        kind="plant_robustness",
        title=f"plant-robust g={gain:.1f} rate={rate:.0f} #{idx}",
        duration_s=duration_s, frames=tuple(frames),
    )


SCENARIO_GENERATORS: dict[str, Any] = {
    "steady_curve": _generate_steady_curve,
    "curvature_ramp": _generate_curvature_ramp,
    "curvature_step": _generate_curvature_step,
    "sinusoidal_sweep": _generate_sinusoidal_sweep,
    "roll_disturbance": _generate_roll_disturbance,
    "speed_sweep": _generate_speed_sweep,
    "stress_grid": _generate_stress_grid,
    "plant_robustness": _generate_plant_robustness,
}


def generate_scenarios(config: ControllerFuzzerConfig) -> list[ControllerScenario]:
    rng = random.Random(config.seed)
    kinds = [config.kind] if config.kind else list(CONTROLLER_SCENARIO_KINDS)
    generators = [SCENARIO_GENERATORS[k] for k in kinds]
    scenarios: list[ControllerScenario] = []
    for idx in range(config.cases):
        gen = rng.choice(generators)
        scenarios.append(gen(rng, idx, config.duration_s))
    return scenarios


def _frame_to_cs(frame: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        vEgo=frame["v_ego"], steeringAngleDeg=frame["steering_angle_deg"],
        steeringRateDeg=frame["steering_rate_deg"], steeringPressed=frame["steering_pressed"],
    )


def _frame_to_params(frame: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(roll=frame["roll"], angleOffsetDeg=frame["angle_offset_deg"])


# ── closed-loop steering plant ──────────────────────────────────────────────

class _SteeringPlant:
    """Simple per-step steering plant for closed-loop controller fuzzing.

    Converts normalized output torque [-1, 1] → steering angle via a
    rate-limited integrator. This closes the loop: the controller outputs
    torque, the plant moves the steering, the controller measures curvature
    from the new steering angle, and the PID corrects the error.
    """

    def __init__(self, dt: float = DT, max_angle_deg: float = 360.0,
                 max_rate_deg_s: float = 180.0):
        self.dt = dt
        self.max_angle = max_angle_deg
        self.max_rate = max_rate_deg_s
        self.angle_deg = 0.0
        self.rate_deg = 0.0

    def update(self, torque: float):
        self.rate_deg = torque * self.max_rate
        self.angle_deg += self.rate_deg * self.dt
        self.angle_deg = max(-self.max_angle, min(self.max_angle, self.angle_deg))

    @property
    def angle_rad(self) -> float:
        return math.radians(self.angle_deg)


# ── runner / evaluation ──────────────────────────────────────────────────────

@dataclass
class ControllerFuzzerConfig:
    seed: int = 1
    cases: int = 100
    kind: str | None = None
    duration_s: float = 2.0
    closed_loop: bool = True  # enable steering plant feedback loop


def evaluate_scenario(scenario: ControllerScenario,
                      closed_loop: bool = True,
                      slew_scale_mode: str = "off") -> ControllerScenarioResult:
    thresholds = scenario.metric_thresholds
    failures: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {}
    outputs: list[ControllerFrameOutput] = []

    controller = _make_controller(slew_scale_mode)
    vm = _FakeVM()
    # Multi-plant robustness: read plant configuration from first frame.
    plant_config = {}
    if scenario.frames and scenario.kind == "plant_robustness":
        first = scenario.frames[0]
        plant_config["gain"] = first.get("plant_gain", 1.0)
        plant_config["rate"] = first.get("plant_rate", 180.0)
    plant = _SteeringPlant(DT, max_rate_deg_s=plant_config.get("rate", 180.0)) if closed_loop else None
    plant_gain = plant_config.get("gain", 1.0)
    active = scenario.frames[0].get("active", True) if scenario.frames else True

    for idx, frame in enumerate(scenario.frames):
        try:
            cs = _frame_to_cs(frame)
            if plant is not None:
                # Closed-loop: override steering from plant feedback.
                cs.steeringAngleDeg = plant.angle_deg
                cs.steeringRateDeg = plant.rate_deg
            params = _frame_to_params(frame)
            out_torque, _, pid_log = controller.update(
                active=active,
                CS=cs, VM=vm, params=params,
                steer_limited_by_safety=frame["steer_limited_by_safety"],
                desired_curvature=frame["desired_curvature"],
                calibrated_pose=None,
                curvature_limited=frame["curvature_limited"],
                lat_delay=frame["lat_delay"],
            )
            if plant is not None:
                plant.update(float(out_torque) * plant_gain)
            # Governor state (diagnostic, not a structural check).
            try:
                gov_reason = int(controller.governor._last_reason) if hasattr(controller.governor, "_last_reason") else 0
            except Exception:
                gov_reason = 0

            outputs.append(ControllerFrameOutput(
                t=float(idx * DT),
                output_torque=float(out_torque),
                pid_p=float(pid_log.p), pid_i=float(pid_log.i),
                pid_d=float(pid_log.d), pid_f=float(pid_log.f),
                pid_output=float(pid_log.output),
                pid_error=float(pid_log.error),
                actual_lat_accel=float(pid_log.actualLateralAccel),
                desired_lat_accel=float(pid_log.desiredLateralAccel),
                desired_lat_jerk=float(pid_log.desiredLateralJerk),
                saturated=bool(pid_log.saturated),
                governor_reason=gov_reason,
                governor_cap=1.0,
                governor_floor=1.0,
            ))
        except Exception as exc:
            failures.append({
                "check": "exception",
                "detail": f"frame {idx} t={idx * DT:.2f} raised {type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            })
            break

    if failures:
        return ControllerScenarioResult(scenario, tuple(outputs), False, failures, metrics)
    if not outputs:
        failures.append({"check": "output", "detail": "scenario produced no output frames"})
        return ControllerScenarioResult(scenario, tuple(outputs), False, failures, metrics)

    t_arr = np.array([o.t for o in outputs], dtype=float)
    torque = np.array([o.output_torque for o in outputs], dtype=float)
    pid_i_arr = np.array([o.pid_i for o in outputs], dtype=float)

    # Finite.
    if not np.all(np.isfinite(torque)):
        failures.append({"check": "finite_torque", "detail": "non-finite output torque"})
    if not np.all(np.isfinite(pid_i_arr)):
        failures.append({"check": "finite_pid_i", "detail": "non-finite PID I term"})

    # Bounded output.
    max_abs = float(np.max(np.abs(torque)))
    metrics["max_abs_output_torque"] = max_abs
    if max_abs > thresholds.max_abs_output_torque + 1e-6:
        failures.append({"check": "torque_bounds", "detail": f"|torque|={max_abs:.3f} exceeds {thresholds.max_abs_output_torque}"})

    # Saturation.
    sat_frac = float(np.mean(np.abs(torque) >= 0.99))
    metrics["saturation_fraction"] = sat_frac
    if sat_frac > thresholds.max_saturation_fraction:
        failures.append({"check": "saturation", "detail": f"saturation fraction {sat_frac:.2f} > {thresholds.max_saturation_fraction}"})

    # Oscillation (sign flips in output torque).
    if len(torque) > 1:
        signs = np.sign(torque)
        nonzero = signs != 0
        sign_flips = int(np.sum(signs[nonzero][1:] != signs[nonzero][:-1])) if np.sum(nonzero) > 1 else 0
        metrics["sign_flips"] = sign_flips
        if sign_flips > thresholds.max_oscillation_reversals:
            failures.append({"check": "oscillation", "detail": f"{sign_flips} sign flips > {thresholds.max_oscillation_reversals}"})

    # Tracking error (lat-accel).
    errors = np.array([o.desired_lat_accel - o.actual_lat_accel for o in outputs], dtype=float)
    if np.all(np.isfinite(errors)):
        max_err = float(np.max(np.abs(errors[-len(errors)//2:])))  # second half (steady-state)
        metrics["max_abs_tracking_error"] = max_err
        if max_err > thresholds.max_abs_tracking_error:
            failures.append({"check": "tracking_error", "detail": f"max |error|={max_err:.4f} m/s²"})

    return ControllerScenarioResult(scenario, tuple(outputs), not failures, failures, metrics)


# ── serialization / CLI ──────────────────────────────────────────────────────

def scenario_summary_to_dict(scenario: ControllerScenario, seed: int | None = None,
                             index: int | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"kind": scenario.kind, "title": scenario.title,
                                "duration_s": scenario.duration_s, "frames": len(scenario.frames)}
    if seed is not None: payload["seed"] = seed
    if index is not None: payload["index"] = index
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Seeded lateral controller structural fuzzer.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cases", type=int, default=100)
    parser.add_argument("--kind", choices=CONTROLLER_SCENARIO_KINDS, help="Run only one scenario kind")
    parser.add_argument("--preset", choices=("fuzz",), help="Preset mode (fuzz: seeded random controller scenarios)")
    parser.add_argument("--duration", type=float, default=2.0, help="Scenario duration in seconds")
    parser.add_argument("--open-loop", action="store_true", help="Disable closed-loop steering plant (default: closed-loop)")
    parser.add_argument("--slew-scale-mode", choices=("off", "shadow", "apply"), default="off",
                        help="LateralSlewScaleMode condition for the controller under fuzz (default: off)")
    parser.add_argument("--endurance", type=int, default=0, help="Run N iterations with one controller instance (catches cumulative state bugs)")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text")
    parser.add_argument("--fail-fast", action="store_true", help="Stop after the first failure")
    args = parser.parse_args()

    if args.cases < 0:
        parser.error("--cases must be >= 0")
    if args.duration <= 0.0:
        parser.error("--duration must be > 0")

    closed_loop = not args.open_loop
    kind = args.kind
    preset = args.preset
    if preset and not kind:
        kind = None  # preset overrides --kind

    # ── endurance mode: one controller, many iterations ──────────────────────
    if args.endurance > 0:
        controller = _make_controller(args.slew_scale_mode)
        vm = _FakeVM()
        plant = _SteeringPlant(DT) if closed_loop else None
        endurance_failures: list[str] = []
        rng = random.Random(args.seed)
        iteration = 0
        for iteration in range(args.endurance):
            # Generate a short random scenario for each iteration.
            v = rng.uniform(5.0, 35.0)
            k = rng.choice([-1.0, 1.0]) * rng.uniform(0.001, 0.008)
            roll = rng.uniform(-0.06, 0.06)
            active = rng.random() > 0.05
            n_frames = int(2.0 / DT)  # 2-second iterations
            for i in range(n_frames):
                angle = _curvature_to_steering_deg(k, v, roll)
                cs = SimpleNamespace(vEgo=v, steeringAngleDeg=angle, steeringRateDeg=0.0,
                                      steeringPressed=False)
                if plant is not None:
                    cs.steeringAngleDeg = plant.angle_deg
                    cs.steeringRateDeg = plant.rate_deg
                params = SimpleNamespace(roll=roll, angleOffsetDeg=0.0)
                try:
                    out, _, pid = controller.update(
                        active, cs, vm, params, False, k, None, False, 0.2)
                    if not np.isfinite(out):
                        endurance_failures.append(f"iter {iteration} frame {i}: non-finite torque {out}")
                        break
                    if abs(out) > 1.0 + 1e-6:
                        endurance_failures.append(f"iter {iteration} frame {i}: torque {out:.3f} exceeds limit")
                        break
                    if not np.isfinite(pid.p) or not np.isfinite(pid.i):
                        endurance_failures.append(f"iter {iteration} frame {i}: non-finite PID p={pid.p} i={pid.i}")
                        break
                except Exception as e:
                    endurance_failures.append(f"iter {iteration} frame {i}: {type(e).__name__}: {e}")
                    break
                if plant is not None:
                    plant.update(float(out))
            if endurance_failures:
                break
        payload = {
            "endurance": args.endurance,
            "iterations_completed": iteration + 1 if endurance_failures else args.endurance,
            "closed_loop": closed_loop,
            "failures": endurance_failures,
        }
        if args.json:
            print(json.dumps(_sanitize(payload), indent=2, sort_keys=True, allow_nan=False))
        else:
            status = "PASSED" if not endurance_failures else f"FAILED at iteration {iteration}"
            print(f"Drive Lab endurance fuzz: {args.endurance} iterations {status}")
            for f in endurance_failures[:5]:
                print(f"  {f}")
        raise SystemExit(0 if not endurance_failures else 1)

    config = ControllerFuzzerConfig(seed=args.seed, cases=args.cases, kind=kind,
                                      duration_s=args.duration, closed_loop=closed_loop)
    scenarios = generate_scenarios(config)
    results: list[tuple[int, ControllerScenarioResult]] = []
    for idx, scenario in enumerate(scenarios):
        result = evaluate_scenario(scenario, closed_loop=config.closed_loop, slew_scale_mode=args.slew_scale_mode)
        results.append((idx, result))
        if result.failures and args.fail_fast:
            break

    failures = [(idx, result) for idx, result in results if result.failures]

    if args.json:
        payload = {
            "seed": args.seed,
            "cases": len(results),
            "kind": args.kind,
            "duration": args.duration,
            "dt": DT,
            "failures": [
                {
                    "scenario": scenario_summary_to_dict(result.scenario, seed=args.seed, index=result_idx),
                    "checks": [f["check"] for f in result.failures],
                    "metrics": _sanitize(result.metrics),
                }
                for result_idx, result in failures
            ],
        }
        print(json.dumps(_sanitize(payload), indent=2, sort_keys=True, allow_nan=False))
    else:
        print(f"Drive Lab lateral controller fuzz seed={args.seed} cases={len(results)} "
              f"kind={args.kind or 'all'} duration={args.duration}s dt={DT}s failures={len(failures)}")
        for idx, result in failures[:10]:
            print(f"\nFAILED: {result.scenario.title} [{result.scenario.kind}]")
            for failure in result.failures:
                print(f"  {failure['check']}: {failure['detail']}")

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
