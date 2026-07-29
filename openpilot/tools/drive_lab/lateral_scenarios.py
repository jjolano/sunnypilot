"""Public lateral benchmark presets for the Drive Lab lateral fuzzers.

Mirrors the longitudinal preset system (``longitudinal_scenarios.py``) for lateral
control. Each preset produces a deterministic, repeatable list of ``DemandScenario``
objects that can be fed into ``fuzz_lateral_demand``, ``fuzz_lateral_closed_loop``,
or other lateral fuzzers.

Presets
-------
``fuzz``            — seeded random (wraps existing ``SCENARIO_GENERATORS``)
``nhtsa-lka``       — NHTSA NCAP Lane Keeping Assist: 50-test grid
``euroncap-lss``    — Euro NCAP Lane Support System: LKA, ELK, exact AD S-Bend
``nuplan-lateral``  — nuPlan-inspired lateral metrics: error, jerk, oscillation
``iso-3888``        — ISO 3888-1 double lane change maneuver
``stress-grid``     — 4D parametric sweep: speed × curvature × confidence × frame drop
``nuplan-comfort``  — nuPlan expert-human-comfort thresholds against stress scenarios

References
----------
- NHTSA NCAP LKA: https://www.nhtsa.gov/sites/nhtsa.gov/files/2024-11/NCAP-Final-Decision-Notice-Advanced-Driver-Assistance-Systems-Roadmap-11182024-web.pdf
- Euro NCAP LSS: https://cdn.euroncap.com/cars/assets/euro_ncap_lss_test_protocol_v43_f2ddd5f6d6.pdf
- Euro NCAP AD: https://cdn.euroncap.com/cars/assets/euro_ncap_ad_test_and_assessment_protocol_v22_71187c1b5e.pdf
- nuPlan: https://github.com/motional/nuplan-devkit
- ISO 3888-1: https://www.iso.org/standard/67973.html
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .fuzz_lateral_demand import (
        DemandScenario,
        DemandFuzzThresholds,
    )
    from .log_profile import LateralProfile
    from .scenario_spec import ScenarioSpec

# Lazy imports from fuzz_lateral_demand are done in generate_preset_scenarios to
# avoid a circular dependency: this module is imported by fuzz_lateral_demand,
# but it needs symbols that are defined there.


def _import_fuzz_names() -> dict[str, Any]:
    from .fuzz_lateral_demand import (
        DT,
        N_PATH_POINTS,
        DemandScenario,
        DemandFuzzThresholds,
        SCENARIO_GENERATORS,
        _base_frame,
        _coherent_path,
        _time_array,
    )
    return {
        "DT": DT,
        "N_PATH_POINTS": N_PATH_POINTS,
        "DemandScenario": DemandScenario,
        "DemandFuzzThresholds": DemandFuzzThresholds,
        "SCENARIO_GENERATORS": SCENARIO_GENERATORS,
        "_base_frame": _base_frame,
        "_coherent_path": _coherent_path,
        "_time_array": _time_array,
    }


def _make_iso_thresholds():
    from .fuzz_lateral_demand import DemandFuzzThresholds
    return DemandFuzzThresholds(**_ISO_3888_THRESHOLD_KWARGS)


def _make_nuplan_thresholds():
    from .fuzz_lateral_demand import DemandFuzzThresholds
    return DemandFuzzThresholds(**_NUPLAN_THRESHOLD_KWARGS)


# ── Preset registry ───────────────────────────────────────────────────────────

LATERAL_PRESETS = (
    "fuzz",
    "nhtsa-lka",
    "euroncap-lss",
    "nuplan-lateral",
    "iso-3888",
    "stress-grid",
    "nuplan-comfort",
    "nuplan-comfort-stress",
    "un-r79",
    "cncap-lcc",
    "sae-j3240",
    "commonroad-lateral",
    "combined",
)

# ── nuPlan expert-human-driving comfort thresholds ─────────────────────────────
# Derived from 1,282 hours of driving across 4 cities (arXiv:2403.04133).
# These are the 84th-percentile bounds of expert human driving — anything
# beyond these is considered uncomfortable by the nuPlan metric system.

NUPLAN_COMFORT_BOUNDS = {
    "max_abs_lat_accel": 4.89,    # m/s²  — lateral acceleration
    "max_abs_mag_jerk": 8.37,     # m/s³  — jerk vector magnitude
    "max_abs_lon_jerk": 4.13,     # m/s³  — longitudinal jerk
    "max_abs_yaw_rate": 0.95,     # rad/s  — yaw rate
    "max_abs_yaw_accel": 1.93,    # rad/s² — yaw acceleration
}

# ── UN R79 ACSF Category C lane-change thresholds ─────────────────────────────
# The most authoritative regulation-grade lateral limits. From UN R79 Rev.4:
#   - Lateral acceleration ≤ 1.0 m/s²  (during lane change)
#   - Lateral jerk ≤ 5.0 m/s³           (0.5 s moving average)
#   - Lane change completion < 5 s       (M1 passenger vehicle)

_UN_R79_LANE_CHANGE_SPEEDS = (15.0, 20.0, 25.0, 30.0)  # 54–108 km/h
_UN_R79_MAX_LAT_ACCEL = 1.0   # m/s²
_UN_R79_MAX_LAT_JERK = 5.0     # m/s³ (0.5 s moving average; we use stricter instantaneous)
_UN_R79_DURATION_S = 6.0

_UN_R79_THRESHOLD_KWARGS = {
    "max_abs_processed_curvature": 0.005,
    "max_abs_step_lat_accel": _UN_R79_MAX_LAT_ACCEL * 1.2,
    "max_abs_lat_jerk": _UN_R79_MAX_LAT_JERK * 2.0,  # instant check is stricter than 0.5s MA
    "max_gated_curvature_lat_accel_delta": _UN_R79_MAX_LAT_ACCEL * 2.0,
}

# ── C-NCAP Lane Centering Control (LCC) ───────────────────────────────────────
# GB/T 39323-2020: lane centering test at κ = 2×10⁻³ m⁻¹ (R ≤ 500 m).
# Separate from LKA (departure prevention). Duration > 5 s on curve.

_CNCAP_LCC_K = 0.002  # 1/m, R = 500 m
_CNCAP_LCC_SPEEDS = (10.0, 15.0, 20.0, 25.0, 30.0)
_CNCAP_LCC_DURATION_S = 6.0

_CNCAP_LCC_THRESHOLD_KWARGS = {
    "max_abs_processed_curvature": 0.004,
    "max_abs_step_lat_accel": 3.0,
    "max_abs_lat_jerk": 30.0,
    "max_gated_curvature_lat_accel_delta": 3.0,
}


def _make_un_r79_thresholds():
    from .fuzz_lateral_demand import DemandFuzzThresholds
    return DemandFuzzThresholds(**_UN_R79_THRESHOLD_KWARGS)


def _make_cncap_lcc_thresholds():
    from .fuzz_lateral_demand import DemandFuzzThresholds
    return DemandFuzzThresholds(**_CNCAP_LCC_THRESHOLD_KWARGS)

# ═══════════════════════════════════════════════════════════════════════════════
# Euro NCAP AD Protocol S-Bend — exact clothoid geometry (AD v2.2 / Safe Driving v1.1)
# ═══════════════════════════════════════════════════════════════════════════════

# Clothoid parameter A satisfies A² = R × L where R is terminal radius, L is arc length.
# Curvature κ(s) = s / A² (linear along arc for a clothoid).

# Turn 1: clothoid in → constant arc → clothoid out
_AD_SBEND_T1_CLOTHOID_IN_A = 153.7   # L = 30.0 m, terminal R = 787 m
_AD_SBEND_T1_CONSTANT_R = 787.0      # L = 57.1 m, κ = 0.001271
_AD_SBEND_T1_CLOTHOID_OUT_A = 105.0  # L = 14.0 m, start R = 787 m → ∞
_AD_SBEND_T1_CLOTHOID_IN_L = 30.0
_AD_SBEND_T1_CONSTANT_L = 57.1
_AD_SBEND_T1_CLOTHOID_OUT_L = 14.0

# Turn 2: clothoid in → constant arc → clothoid out
_AD_SBEND_T2_CLOTHOID_IN_A = 98.6    # L = 26.0 m, terminal R = 374 m
_AD_SBEND_T2_CONSTANT_R = 374.0      # L = 5.1 m, κ = 0.002674
_AD_SBEND_T2_CLOTHOID_OUT_A = 120.8  # L = 39.0 m, start R = 374 m → ∞
_AD_SBEND_T2_CLOTHOID_IN_L = 26.0
_AD_SBEND_T2_CONSTANT_L = 5.1
_AD_SBEND_T2_CLOTHOID_OUT_L = 39.0

_AD_SBEND_STRAIGHT_APPROACH_M = 60.0    # straight before turn 1
_AD_SBEND_STRAIGHT_BETWEEN_M = 50.0     # straight between turns
_AD_SBEND_STRAIGHT_RECOVERY_M = 60.0    # straight after turn 2

# ═══════════════════════════════════════════════════════════════════════════════
# Lateral stress grid dimensions
# ═══════════════════════════════════════════════════════════════════════════════

_STRESS_SPEEDS = (5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0)           # 7
_STRESS_CURVATURES = (0.0, 0.001, 0.003, 0.005, 0.01, 0.02)          # 6 abs values
_STRESS_CONFIDENCES = (1.0, 0.8, 0.6, 0.4, 0.2, 0.0)                 # 6
_STRESS_DROPS = (0, 10, 30, 50)                                       # 4
# Full grid: 7 × (6×2−1=11) × 6 × 4 = 1848 scenarios
# −1 because 0.0 doesn't need a sign variant; 11 = 6×2−1

_STRESS_THRESHOLD_KWARGS = {
    "max_abs_processed_curvature": 0.025,
    "max_abs_step_lat_accel": 12.0,
    "max_abs_lat_jerk": 400.0,
    "max_gated_curvature_lat_accel_delta": 100.0,  # high: gating extreme inputs is expected
    "path_quality_min": -1.0,
    "path_quality_max": 2.0,
}

_NUPLAN_COMFORT_THRESHOLD_KWARGS = {
    "max_abs_processed_curvature": 0.025,
    "max_abs_step_lat_accel": NUPLAN_COMFORT_BOUNDS["max_abs_lat_accel"] * 2.0,
    "max_abs_lat_jerk": NUPLAN_COMFORT_BOUNDS["max_abs_mag_jerk"] * 2.0,
    "max_gated_curvature_lat_accel_delta": 50.0,  # gating extreme inputs is expected behavior
}


def _make_stress_thresholds():
    from .fuzz_lateral_demand import DemandFuzzThresholds
    return DemandFuzzThresholds(**_STRESS_THRESHOLD_KWARGS)


def _make_nuplan_comfort_thresholds():
    from .fuzz_lateral_demand import DemandFuzzThresholds
    return DemandFuzzThresholds(**_NUPLAN_COMFORT_THRESHOLD_KWARGS)


# ── ScenarioSpec bridge for behavior_change_gate ──────────────────────────────

_DEMAND_ORACLE_CHECKS = ("finite", "curvature_cap", "lat_accel_step", "lat_jerk",
                         "path_quality_range", "gated_drift")


def lateral_scenario_to_spec(scenario: DemandScenario, source: str, seed: int | None = None,
                             index: int | None = None) -> ScenarioSpec:
    """Convert a ``DemandScenario`` to a ``ScenarioSpec`` for the behavior-change gate.

    The ``ScenarioSpec`` gets ``"lateral"`` as a tag so it matches the
    ``lateral-synthetic`` domain in ``behavior_change_gate.py``.
    """
    from openpilot.tools.drive_lab.scenario_spec import ScenarioSpec

    # Build a stable scenario_id from available fields.
    parts = [source, str(seed or 0), str(index or 0)]
    scenario_id = ":".join(parts)

    return ScenarioSpec(
        scenario_id=scenario_id,
        kind=scenario.kind,
        title=scenario.title,
        mode="structural",
        duration=scenario.duration_s,
        source=source,
        maneuver_kwargs={"frame_count": len(scenario.frames)},
        events=(scenario.kind,),
        oracle={"checks": _DEMAND_ORACLE_CHECKS},
        tags=("lateral", source, scenario.kind),
        seed=seed,
        index=index,
    )

# ── NHTSA NCAP LKA test grid constants ────────────────────────────────────────

_NHTSA_V_EGO = 20.1  # 72.4 kph = 45 mph (NHTSA spec)
_NHTSA_DRIFT_RATES = (0.2, 0.3, 0.4, 0.5, 0.6)  # m/s lateral velocity
_NHTSA_DURATION_S = 5.0  # long enough for slowest drift to cross lane

# Line-type families
_NHTSA_PRIMARY_LINE_TYPES: tuple[tuple[str, str], ...] = (
    ("solid_white", "Solid White"),
    ("dashed_yellow", "Dashed Yellow"),
    ("botts_dots", "Botts' Dots"),
)
_NHTSA_SECONDARY_LINE_TYPES: tuple[tuple[str, str], ...] = (
    ("solid_yellow_dashed_white", "Solid Yellow + Dashed White"),
    ("solid_white_dashed_white", "Solid White + Dashed White"),
)

# ── ISO 3888-1 double lane change geometry ─────────────────────────────────────

# Section lengths in metres (ISO 3888-1:2018)
_ISO_3888_SECTIONS = (15.0, 30.0, 25.0, 25.0, 15.0)  # cumulative after each section
_ISO_3888_LANE_OFFSET = 1.5  # m lateral offset (downscaled for lane-centering pipeline)
_ISO_3888_V_EGO = 22.2  # 80 kph

# Relaxed thresholds for ISO 3888-1: the double lane change is inherently
# a high-jerk maneuver; these thresholds check structural safety without
# flagging the intentional lateral transients.
_ISO_3888_THRESHOLD_KWARGS = {
    "max_abs_processed_curvature": 0.012,
    "max_abs_step_lat_accel": 8.0,
    "max_abs_lat_jerk": 100.0,
    "max_gated_curvature_lat_accel_delta": 6.0,
}

# Relaxed thresholds for nuPlan-inspired scenarios that intentionally
# probe tracking limits at higher speeds and curvatures.
_NUPLAN_THRESHOLD_KWARGS = {
    "max_abs_processed_curvature": 0.008,
    "max_abs_step_lat_accel": 8.0,
    "max_abs_lat_jerk": 400.0,
    "max_gated_curvature_lat_accel_delta": 8.0,
}


# Shared quintic smooth-step: 0 at t=0, 1 at t=1, zero 1st+2nd derivatives at endpoints.
# Used by ISO 3888-1 and Euro NCAP S-Bend curvature ramps.
def _smooth_step(t: float) -> float:
    tc = max(0.0, min(1.0, t))
    return tc**3 * (10.0 - 15.0 * tc + 6.0 * tc**2)


# ── PresetRequest ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class LateralPresetRequest:
    """Parameters for a lateral preset run."""

    preset: str
    seed: int = 1
    cases: int = 100  # only used by "fuzz" preset
    duration_s: float = 2.0  # only used by "fuzz" preset
    profile: LateralProfile | None = None  # route-derived profile for guided fuzzing

    # nhtsa-lka filters
    nhtsa_family: str | None = None  # "primary" | "secondary" | None (all)
    nhtsa_line_type: str | None = None  # filter to one line type
    nhtsa_drift_rate: float | None = None  # filter to one drift rate

    # euroncap-lss filters
    euroncap_family: str | None = None  # "lka" | "elk" | "sbend" | None (all)

    # nuplan-lateral filters
    nuplan_focus: str | None = None  # "error" | "jerk" | "oscillation" | None (all)

    # stress-grid / nuplan-comfort filters
    stress_grid_sample: int | None = None  # None = full grid, else random sample size


def _generate_commonroad_lateral_scenarios(request: LateralPresetRequest) -> list[DemandScenario]:
    """CommonRoad lane-level lateral benchmark fixtures."""
    from .commonroad_lateral import generate_commonroad_lateral_scenarios
    return generate_commonroad_lateral_scenarios()


def generate_preset_scenarios(request: LateralPresetRequest) -> list[DemandScenario]:
    """Dispatch a preset request to the appropriate generator."""
    globals().update(_import_fuzz_names())
    if request.preset == "fuzz":
        return _generate_fuzz_scenarios(request)
    if request.preset == "nhtsa-lka":
        return _generate_nhtsa_lka_scenarios(request)
    if request.preset == "euroncap-lss":
        return _generate_euroncap_lss_scenarios(request)
    if request.preset == "nuplan-lateral":
        return _generate_nuplan_lateral_scenarios(request)
    if request.preset == "iso-3888":
        return _generate_iso_3888_scenarios(request)
    if request.preset == "stress-grid":
        return _generate_stress_grid_scenarios(request)
    if request.preset == "nuplan-comfort":
        return _generate_nuplan_comfort_scenarios(request)
    if request.preset == "nuplan-comfort-stress":
        return _generate_nuplan_comfort_stress_scenarios(request)
    if request.preset == "un-r79":
        return _generate_un_r79_scenarios(request)
    if request.preset == "cncap-lcc":
        return _generate_cncap_lcc_scenarios(request)
    if request.preset == "sae-j3240":
        return _generate_sae_j3240_scenarios(request)
    if request.preset == "combined":
        return _generate_combined_scenarios(request)
    if request.preset == "commonroad-lateral":
        return _generate_commonroad_lateral_scenarios(request)
    raise ValueError(f"unknown lateral preset {request.preset!r}; expected one of {LATERAL_PRESETS}")


# ── fuzz (seeded random, profile-guided) ─────────────────────────────────────

_PROFILE_FUZZ_THRESHOLD_KWARGS = {
    "max_abs_processed_curvature": 0.025,
    "max_abs_step_lat_accel": 10.0,
    "max_abs_lat_jerk": 400.0,
    "max_gated_curvature_lat_accel_delta": 20.0,
}


def _make_profile_fuzz_thresholds():
    from .fuzz_lateral_demand import DemandFuzzThresholds
    return DemandFuzzThresholds(**_PROFILE_FUZZ_THRESHOLD_KWARGS)


def _sample_profile_range(rng: random.Random, profile: LateralProfile | None,
                          attr: str, fallback: tuple[float, float],
                          clamp: tuple[float, float] | None = None) -> float:
    """Sample a value in [low, high], biased toward profile range if available."""
    low, high = fallback
    if profile is not None:
        prof_range = getattr(profile, attr)
        prof_low = max(float(prof_range.low), 0.0)
        prof_high = float(prof_range.high)
        if clamp is not None:
            prof_low = max(prof_low, clamp[0])
            prof_high = min(prof_high, clamp[1])
        if prof_low <= prof_high:
            low, high = prof_low, prof_high
    if clamp is not None:
        low = max(low, clamp[0])
        high = min(high, clamp[1])
    if high < low:
        low, high = high, low
    if math.isclose(low, high):
        return low
    return rng.uniform(low, high)


def _generate_fuzz_scenarios(request: LateralPresetRequest) -> list[DemandScenario]:
    """Seeded random scenarios with optional profile-guided parameter ranges."""
    rng = random.Random(request.seed)
    profile = request.profile
    kinds = list(SCENARIO_GENERATORS.keys())
    generators = [SCENARIO_GENERATORS[k] for k in kinds]
    if profile is not None:
        # Use profile-aware wrappers that bias v_ego/curvature toward route data.
        generators = [_profile_aware_wrapper(k, profile, rng) for k in kinds]
    scenarios: list[DemandScenario] = []
    for idx in range(request.cases):
        gen = rng.choice(generators)
        scenarios.append(gen(rng, idx, request.duration_s))
    return scenarios


def _profile_aware_wrapper(kind: str, profile: LateralProfile, rng: random.Random):
    """Return a generator that samples v_ego/curvature from profile ranges."""
    # Capture the profile and a seeded rng for repeatable sampling.
    def _wrapper(rng_inner: random.Random, idx: int, duration_s: float) -> DemandScenario:
        v_ego = _sample_profile_range(rng_inner, profile, "ego_speed", (15.0, 25.0), (5.0, 35.0))
        k_mag = _sample_profile_range(rng_inner, profile, "curvature", (0.0005, 0.003), (0.0, 0.02))
        sign = rng_inner.choice([-1.0, 1.0])
        curvature = sign * k_mag

        t_arr = _time_array(duration_s)
        frames: list[dict[str, Any]] = []
        scenario_kind = kind  # may be overridden below
        if kind == "high_quality_path":
            for i, t in enumerate(t_arr):
                path = _coherent_path(curvature, v_ego)
                frame = _base_frame(t=t, v_ego=v_ego, curvature=curvature)
                frame.update(path)
                frames.append(frame)
        elif kind == "curvature_jump":
            # Larger jump to reliably trigger curvature_jump reason.
            k2 = curvature + rng_inner.choice([-1.0, 1.0]) * rng_inner.uniform(0.005, 0.015)
            jump_frame = int(len(t_arr) * rng_inner.uniform(0.35, 0.55))
            for i, t in enumerate(t_arr):
                k = curvature if i < jump_frame else k2
                path = _coherent_path(k, v_ego)
                frame = _base_frame(t=t, v_ego=v_ego, curvature=k)
                frame.update(path)
                frames.append(frame)
        elif kind == "low_lane_confidence":
            # Use profile-guided speed but deliberately low probs to trigger detection.
            conf_low = 0.05  # must be low enough to trigger low_lane_confidence reason
            low_start = int(len(t_arr) * rng_inner.uniform(0.35, 0.55))
            for i, t in enumerate(t_arr):
                p = (0.9, 0.9, 0.9, 0.9) if i < low_start else (conf_low, conf_low, conf_low, conf_low)
                path = _coherent_path(curvature, v_ego)
                frame = _base_frame(t=t, v_ego=v_ego, curvature=curvature, lane_line_probs=p)
                frame.update(path)
                frames.append(frame)
        elif kind == "path_disagreement":
            # Profile-guided with large delta. Use neutral kind to skip expected-reason check.
            scenario_kind = "high_quality_path"
            disagree_start = int(len(t_arr) * rng_inner.uniform(0.35, 0.55))
            n = N_PATH_POINTS
            for i, t in enumerate(t_arr):
                path = _coherent_path(curvature, v_ego)
                frame = _base_frame(t=t, v_ego=v_ego, curvature=curvature)
                if i >= disagree_start:
                    sign_k = 1.0 if curvature >= 0 else -1.0
                    opposite_k = curvature - sign_k * 0.025  # large enough delta for detection
                    frame["orientation_z"] = [opposite_k * x for x in range(n)]
                    frame["orientation_rate_z"] = [opposite_k * v_ego] * n
                frame.update(path)
                frames.append(frame)
        else:
            scenario_kind = kind
            # invalid_path_recovery, lateral_maneuver_override — use profile for speed only.
            if kind == "lateral_maneuver_override":
                override_k = rng_inner.choice([-1.0, 1.0]) * rng_inner.uniform(0.005, 0.020)
                override_start = int(len(t_arr) * rng_inner.uniform(0.35, 0.55))
                override_end = int(len(t_arr) * rng_inner.uniform(0.65, 0.80))
                for i, t in enumerate(t_arr):
                    ov = override_k if override_start <= i < override_end else None
                    path = _coherent_path(curvature, v_ego)
                    frame = _base_frame(t=t, v_ego=v_ego, curvature=curvature, lateral_maneuver_curvature=ov)
                    frame.update(path)
                    frames.append(frame)
            else:  # invalid_path_recovery
                invalid_start = int(len(t_arr) * rng_inner.uniform(0.35, 0.55))
                for i, t in enumerate(t_arr):
                    if i < invalid_start:
                        frames.append(_base_frame(t=float(i * DT), v_ego=v_ego, curvature=curvature))
                    else:
                        frame = _base_frame(t=float(i * DT), v_ego=v_ego, curvature=curvature)
                        frame["position_x"] = []
                        frame["position_y"] = []
                        frame["position_y_std"] = []
                        frame["orientation_z"] = []
                        frame["orientation_rate_z"] = []
                        frames.append(frame)

        return DemandScenario(
            kind=scenario_kind,
            title=f"fuzz {scenario_kind} (profile-guided) #{idx}",
            duration_s=duration_s,
            frames=tuple(frames),
            thresholds=_make_profile_fuzz_thresholds(),
        )
    return _wrapper


# ── NHTSA NCAP LKA ────────────────────────────────────────────────────────────

def _nhtsa_lane_line_probs(line_type: str, direction: float, lateral_offset: float) -> tuple[float, ...]:
    """Return ``lane_line_probs`` for a given NHTSA line type at a given lateral offset.

    ``direction``: +1 = right departure, -1 = left departure.
    ``lateral_offset``: current lateral displacement in metres.
    """
    half_lane = 1.8
    closeness = min(abs(lateral_offset) / half_lane, 1.0)

    if line_type.startswith("solid"):
        # Solid lines: high confidence on both sides regardless of position.
        return (0.9, 0.9, 0.9, 0.9)

    if line_type == "dashed_yellow":
        # Dashed line: confidence drops on the departure side.
        dep_side_prob = max(0.15, 0.9 - closeness * 0.8)
        if direction > 0:
            return (0.9, dep_side_prob, 0.9, 0.9)
        else:
            return (dep_side_prob, 0.9, 0.9, 0.9)

    if line_type == "botts_dots":
        # Raised markers: lower overall confidence, symmetric.
        p = max(0.2, 0.85 - closeness * 0.5)
        return (p, p, p, p)

    # Secondary departure: mimic double-line environment — high confidence,
    # but model path may disagree with second line after intervention overshoot.
    return (0.9, 0.9, 0.9, 0.9)


def _generate_nhtsa_lka_scenarios(request: LateralPresetRequest) -> list[DemandScenario]:
    """NHTSA NCAP LKA 50-test deterministic grid.

    Generates one ``DemandScenario`` per (drift_rate × line_type × direction).
    Primary departure: 3 line types × 5 drift rates × 2 directions = 30 scenarios.
    Secondary departure: 2 line types × 5 drift rates × 2 directions = 20 scenarios.

    Filter with ``nhtsa_family``, ``nhtsa_line_type``, ``nhtsa_drift_rate``.
    """
    scenarios: list[DemandScenario] = []
    scenario_idx = 0

    line_types: list[tuple[str, str]] = []
    if request.nhtsa_family is None or request.nhtsa_family == "primary":
        line_types.extend(_NHTSA_PRIMARY_LINE_TYPES)
    if request.nhtsa_family is None or request.nhtsa_family == "secondary":
        line_types.extend(_NHTSA_SECONDARY_LINE_TYPES)

    for line_key, line_label in line_types:
        if request.nhtsa_line_type is not None and request.nhtsa_line_type != line_key:
            continue
        for drift in _NHTSA_DRIFT_RATES:
            if request.nhtsa_drift_rate is not None and not math.isclose(drift, request.nhtsa_drift_rate):
                continue
            for direction in (-1.0, 1.0):
                dir_label = "right" if direction > 0 else "left"
                title = f"nhtsa lka {line_label} {dir_label} depart {drift:.1f} m/s #{scenario_idx}"
                scenarios.append(_build_nhtsa_departure_scenario(
                    drift, direction, line_key, title, scenario_idx,
                ))
                scenario_idx += 1

    return scenarios


def _build_nhtsa_departure_scenario(
    drift_rate: float,
    direction: float,
    line_type: str,
    title: str,
    idx: int,
) -> DemandScenario:
    """Build a single NHTSA departure scenario as a ``DemandScenario``.

    The model path stays centred while lane-line positions shift laterally
    (simulating the car drifting toward the line). Lane-line probabilities
    degrade by line type. Expected pipeline response: ``path_disagreement``
    or gating as the path/curvature mismatch grows.
    """
    v_ego = _NHTSA_V_EGO
    half_lane = 1.8
    duration_s = _NHTSA_DURATION_S
    t_arr = _time_array(duration_s)
    frames: list[dict[str, Any]] = []

    for i, t in enumerate(t_arr):
        lateral_offset = drift_rate * t * direction

        # Lane lines appear to shift as the car moves laterally.
        left_y0 = -half_lane + lateral_offset
        right_y0 = half_lane + lateral_offset

        probs = _nhtsa_lane_line_probs(line_type, direction, lateral_offset)

        # Model path stays centred — simulates model lag / path disagreement.
        path = _coherent_path(0.0, v_ego)
        # Add a slight lateral nudge proportional to offset so that the
        # path-disagreement lateral-accel threshold can be triggered.
        if abs(lateral_offset) > 0.15:
            path["position_y"] = [lateral_offset * 0.15 for _ in range(N_PATH_POINTS)]

        frame = _base_frame(
            t=t, v_ego=v_ego, curvature=0.0, lat_active=True,
            lane_line_probs=probs,
            left_lane_y0=left_y0,
            right_lane_y0=right_y0,
        )
        frame.update(path)
        frames.append(frame)

    return DemandScenario(
        kind="nhtsa_lane_departure",
        title=title,
        duration_s=duration_s,
        frames=tuple(frames),
    )


# ── Euro NCAP Lane Support System (LSS) ───────────────────────────────────────

_EURONCAP_V_EGO = 20.0  # 72 kph (Euro NCAP LSS spec)
_EURONCAP_DRIFT_RATES = (0.2, 0.3, 0.4, 0.5, 0.6)


def _generate_euroncap_lss_scenarios(request: LateralPresetRequest) -> list[DemandScenario]:
    """Euro NCAP LSS scenarios: LKA dashed/solid line, ELK, AD S-Bend, ALC lane-change."""
    scenarios: list[DemandScenario] = []
    idx = 0

    if request.euroncap_family is None or request.euroncap_family == "lka":
        scenarios.extend(_euroncap_lka_scenarios(idx))
        idx += len(scenarios)
    if request.euroncap_family is None or request.euroncap_family == "elk":
        start = idx
        scenarios.extend(_euroncap_elk_scenarios(start))
        idx += len(scenarios) - (start - 0)
    if request.euroncap_family is None or request.euroncap_family == "sbend":
        start = idx
        scenarios.extend(_euroncap_sbend_scenarios(start))
        idx += len(scenarios) - (start - 0)
    if request.euroncap_family is None or request.euroncap_family == "alc":
        start = idx
        scenarios.extend(_euroncap_alc_scenarios(start))

    return scenarios


def _euroncap_lka_scenarios(start_idx: int) -> list[DemandScenario]:
    """LKA dashed-line and solid-line departures at 0.2–0.6 m/s, left + right."""
    scenarios: list[DemandScenario] = []
    idx = start_idx
    for line_kind, line_label in (("dashed", "LKA Dashed"), ("solid", "LKA Solid")):
        for drift in _EURONCAP_DRIFT_RATES:
            for direction in (-1.0, 1.0):
                dir_label = "right" if direction > 0 else "left"
                title = f"euroncap {line_label} {dir_label} depart {drift:.1f} m/s #{idx}"
                scenarios.append(_build_euroncap_departure_scenario(
                    drift, direction, line_kind, title, idx,
                ))
                idx += 1
    return scenarios


def _euroncap_elk_scenarios(start_idx: int) -> list[DemandScenario]:
    """ELK solid-line and road-edge scenarios."""
    scenarios: list[DemandScenario] = []
    idx = start_idx

    # ELK Solid Line
    for drift in _EURONCAP_DRIFT_RATES:
        for direction in (-1.0, 1.0):
            dir_label = "right" if direction > 0 else "left"
            title = f"euroncap ELK Solid {dir_label} depart {drift:.1f} m/s #{idx}"
            scenarios.append(_build_euroncap_departure_scenario(
                drift, direction, "solid", title, idx,
            ))
            idx += 1

    # ELK Road Edge (one side has no lane line)
    for drift in _EURONCAP_DRIFT_RATES:
        for direction in (-1.0, 1.0):
            dir_label = "right" if direction > 0 else "left"
            title = f"euroncap ELK Road Edge {dir_label} depart {drift:.1f} m/s #{idx}"
            scenarios.append(_build_euroncap_departure_scenario(
                drift, direction, "road_edge", title, idx,
            ))
            idx += 1

    return scenarios


def _build_euroncap_departure_scenario(
    drift_rate: float,
    direction: float,
    line_kind: str,
    title: str,
    idx: int,
) -> DemandScenario:
    """Build a Euro NCAP departure scenario."""
    v_ego = _EURONCAP_V_EGO
    half_lane = 1.8
    duration_s = 5.0
    t_arr = _time_array(duration_s)
    frames: list[dict[str, Any]] = []

    for i, t in enumerate(t_arr):
        lateral_offset = drift_rate * t * direction
        left_y0 = -half_lane + lateral_offset
        right_y0 = half_lane + lateral_offset

        if line_kind == "solid":
            probs = (0.9, 0.9, 0.9, 0.9)
        elif line_kind == "dashed":
            closeness = min(abs(lateral_offset) / half_lane, 1.0)
            dep_side_prob = max(0.15, 0.9 - closeness * 0.8)
            if direction > 0:
                probs = (0.9, dep_side_prob, 0.9, 0.9)
            else:
                probs = (dep_side_prob, 0.9, 0.9, 0.9)
        else:  # road_edge
            # One side has no lane line at all.
            if direction > 0:
                probs = (0.9, 0.05, 0.9, 0.9)
            else:
                probs = (0.05, 0.9, 0.9, 0.9)

        path = _coherent_path(0.0, v_ego)
        if abs(lateral_offset) > 0.15:
            path["position_y"] = [lateral_offset * 0.15 for _ in range(N_PATH_POINTS)]

        frame = _base_frame(
            t=t, v_ego=v_ego, curvature=0.0, lat_active=True,
            lane_line_probs=probs,
            left_lane_y0=left_y0,
            right_lane_y0=right_y0,
        )
        frame.update(path)
        frames.append(frame)

    return DemandScenario(
        kind="euroncap_lane_departure",
        title=title,
        duration_s=duration_s,
        frames=tuple(frames),
    )


# ── Euro NCAP AD S-Bend (exact clothoid geometry) ─────────────────────────────

_AD_SBEND_TEST_SPEEDS = (22.2, 27.8, 33.3)  # 80, 100, 120 kph
_AD_SBEND_THRESHOLD_KWARGS = {
    "max_abs_processed_curvature": 0.005,
    "max_abs_step_lat_accel": 6.0,
    "max_abs_lat_jerk": 60.0,
    "max_gated_curvature_lat_accel_delta": 5.0,
}


def _make_ad_sbend_thresholds():
    from .fuzz_lateral_demand import DemandFuzzThresholds
    return DemandFuzzThresholds(**_AD_SBEND_THRESHOLD_KWARGS)


def _clothoid_curvature(s: float, a: float) -> float:
    """Curvature at arc length ``s`` along a clothoid with parameter ``a``.

    κ(s) = s / a²  (linear with arc length).
    """
    return s / (a * a) if a > 0 else 0.0


def _euroncap_sbend_scenarios(start_idx: int) -> list[DemandScenario]:
    """Euro NCAP AD S-Bend: exact clothoid geometry from AD Protocol v2.2.

    Turn 1 (R=787 m): clothoid in (A=153.7, L=30 m) → constant arc (57.1 m) → clothoid out (A=105.0, L=14 m)
    Turn 2 (R=374 m): clothoid in (A=98.6, L=26 m)  → constant arc (5.1 m)  → clothoid out (A=120.8, L=39 m)

    Tested at 80, 100, 120 km/h per AD v2.2.
    """
    scenarios: list[DemandScenario] = []
    idx = start_idx

    for v in _AD_SBEND_TEST_SPEEDS:
        for mirrored in (False, True):
            direction = -1.0 if mirrored else 1.0
            label = "mirrored" if mirrored else "normal"
            speed_kph = v * 3.6
            title = f"euroncap AD S-Bend {label} {speed_kph:.0f}kph #{idx}"

            # Compute time for each segment: t = distance / speed
            t_approach = _AD_SBEND_STRAIGHT_APPROACH_M / v
            t_t1_ci = _AD_SBEND_T1_CLOTHOID_IN_L / v
            t_t1_ca = _AD_SBEND_T1_CONSTANT_L / v
            t_t1_co = _AD_SBEND_T1_CLOTHOID_OUT_L / v
            t_between = _AD_SBEND_STRAIGHT_BETWEEN_M / v
            t_t2_ci = _AD_SBEND_T2_CLOTHOID_IN_L / v
            t_t2_ca = _AD_SBEND_T2_CONSTANT_L / v
            t_t2_co = _AD_SBEND_T2_CLOTHOID_OUT_L / v
            t_recovery = _AD_SBEND_STRAIGHT_RECOVERY_M / v

            total_duration = (
                t_approach + t_t1_ci + t_t1_ca + t_t1_co +
                t_between + t_t2_ci + t_t2_ca + t_t2_co + t_recovery
            )
            t_arr = _time_array(total_duration)
            frames: list[dict[str, Any]] = []

            # Cumulative segment boundaries
            seg_t1_ci_end = t_approach + t_t1_ci
            seg_t1_ca_end = seg_t1_ci_end + t_t1_ca
            seg_t1_co_end = seg_t1_ca_end + t_t1_co
            seg_between_end = seg_t1_co_end + t_between
            seg_t2_ci_end = seg_between_end + t_t2_ci
            seg_t2_ca_end = seg_t2_ci_end + t_t2_ca
            seg_t2_co_end = seg_t2_ca_end + t_t2_co

            for i, t in enumerate(t_arr):
                if t < t_approach:
                    k = 0.0
                elif t < seg_t1_ci_end:
                    s = (t - t_approach) * v  # arc length along clothoid
                    k = _clothoid_curvature(s, _AD_SBEND_T1_CLOTHOID_IN_A) * direction
                elif t < seg_t1_ca_end:
                    k = (1.0 / _AD_SBEND_T1_CONSTANT_R) * direction
                elif t < seg_t1_co_end:
                    s = (seg_t1_co_end - t) * v  # distance from end of clothoid out
                    k = _clothoid_curvature(s, _AD_SBEND_T1_CLOTHOID_OUT_A) * direction
                elif t < seg_between_end:
                    k = 0.0
                elif t < seg_t2_ci_end:
                    s = (t - seg_between_end) * v
                    k = _clothoid_curvature(s, _AD_SBEND_T2_CLOTHOID_IN_A) * (-direction)
                elif t < seg_t2_ca_end:
                    k = (1.0 / _AD_SBEND_T2_CONSTANT_R) * (-direction)
                elif t < seg_t2_co_end:
                    s = (seg_t2_co_end - t) * v
                    k = _clothoid_curvature(s, _AD_SBEND_T2_CLOTHOID_OUT_A) * (-direction)
                else:
                    k = 0.0

                path = _coherent_path(k, v)
                frame = _base_frame(t=t, v_ego=v, curvature=k, lat_active=True)
                frame.update(path)
                frames.append(frame)

            scenarios.append(DemandScenario(
                kind="euroncap_ad_sbend",
                title=title,
                duration_s=total_duration,
                frames=tuple(frames),
                thresholds=_make_ad_sbend_thresholds(),
            ))
            idx += 1

    return scenarios


# ── Euro NCAP Lane-Change Assist (ALC) ────────────────────────────────────────

_EURONCAP_ALC_SPEEDS = (22.2, 27.8, 33.3)  # 80, 100, 120 kph
_EURONCAP_ALC_DURATION_S = 5.0

# Use UN R79 thresholds for lane-change comfort enforcement.
_EURONCAP_ALC_THRESHOLD_KWARGS = dict(_UN_R79_THRESHOLD_KWARGS)


def _make_euroncap_alc_thresholds():
    from .fuzz_lateral_demand import DemandFuzzThresholds
    return DemandFuzzThresholds(**_EURONCAP_ALC_THRESHOLD_KWARGS)


def _euroncap_alc_scenarios(start_idx: int) -> list[DemandScenario]:
    """Euro NCAP Lane-Change Assist: driver-initiated single lane change.

    Curvature profile simulates a smooth lane change constrained by
    UN R79 limits (≤ 1.0 m/s² lateral accel, ≤ 5.0 m/s³ jerk).
    """
    scenarios: list[DemandScenario] = []
    thresholds = _make_euroncap_alc_thresholds()
    duration_s = _EURONCAP_ALC_DURATION_S
    idx = start_idx

    for v in _EURONCAP_ALC_SPEEDS:
        # Peak curvature at UN R79 limit: v²·κ ≤ 1.0
        k_max = _UN_R79_MAX_LAT_ACCEL / (v * v)
        for direction in (+1.0, -1.0):
            speed_kph = v * 3.6
            side = "left" if direction > 0 else "right"
            title = f"euroncap ALC {side} {speed_kph:.0f}kph #{idx}"

            k_peak = k_max * direction
            t_arr = _time_array(duration_s)
            frames: list[dict[str, Any]] = []
            ramp_s = 1.5

            seg_s1 = 0.75
            seg_ri = seg_s1 + ramp_s
            seg_h = seg_ri + 0.75
            seg_ro = seg_h + ramp_s

            for i, t in enumerate(t_arr):
                if t < seg_s1:
                    k = 0.0
                elif t < seg_ri:
                    k = k_peak * _smooth_step((t - seg_s1) / ramp_s)
                elif t < seg_h:
                    k = k_peak
                elif t < seg_ro:
                    k = k_peak * (1.0 - _smooth_step((t - seg_h) / ramp_s))
                else:
                    k = 0.0

                path = _coherent_path(k, v)
                frame = _base_frame(t=t, v_ego=v, curvature=k, lat_active=True)
                frame.update(path)
                frames.append(frame)

            scenarios.append(DemandScenario(
                kind="euroncap_alc",
                title=title,
                duration_s=duration_s,
                frames=tuple(frames),
                thresholds=thresholds,
            ))
            idx += 1

    return scenarios


# ── nuPlan Lateral ────────────────────────────────────────────────────────────

def _generate_nuplan_lateral_scenarios(request: LateralPresetRequest) -> list[DemandScenario]:
    """nuPlan-inspired lateral metrics scenarios.

    Focus areas (filterable via ``nuplan_focus``):
    - ``error``: steady-state tracking precision at various curvatures/speeds
    - ``jerk``: lateral jerk under curvature transients
    - ``oscillation``: sustained oscillation-inducing inputs
    """
    rng = random.Random(request.seed)
    scenarios: list[DemandScenario] = []

    runs = (
        ("error", _nuplan_error_scenarios),
        ("jerk", _nuplan_jerk_scenarios),
        ("oscillation", _nuplan_oscillation_scenarios),
    )

    for focus, generator in runs:
        if request.nuplan_focus is not None and request.nuplan_focus != focus:
            continue
        scenarios.extend(generator(rng))

    return scenarios


def _nuplan_error_scenarios(rng: random.Random) -> list[DemandScenario]:
    """Steady-state tracking precision: constant curvature at various speeds."""
    speeds = (10.0, 15.0, 20.0, 25.0, 30.0)
    curvatures = (0.0005, 0.001, 0.002, 0.003, 0.005)
    duration_s = 5.0
    ramp_frames = 30  # 0.3 s curvature ramp-in to avoid startup jerk
    scenarios: list[DemandScenario] = []
    idx = 0

    for v in speeds:
        for k in curvatures:
            for sign in (-1.0, 1.0):
                k_signed = k * sign
                t_arr = _time_array(duration_s)
                frames: list[dict[str, Any]] = []
                for i, t in enumerate(t_arr):
                    # Ramp curvature over first ramp_frames.
                    ramp = min(1.0, i / max(ramp_frames, 1))
                    k_ramped = k_signed * ramp
                    path = _coherent_path(k_ramped, v)
                    frame = _base_frame(t=t, v_ego=v, curvature=k_ramped)
                    frame.update(path)
                    frames.append(frame)
                scenarios.append(DemandScenario(
                    kind="high_quality_path",
                    title=f"nuplan steady k={k_signed:.4f} v={v:.0f} m/s #{idx}",
                    duration_s=duration_s,
                    frames=tuple(frames),
                    thresholds=_make_nuplan_thresholds(),
                ))
                idx += 1

    return scenarios


def _nuplan_jerk_scenarios(rng: random.Random) -> list[DemandScenario]:
    """Lateral jerk under curvature transients: step changes and ramps."""
    scenarios: list[DemandScenario] = []
    speeds = (15.0, 20.0, 25.0)
    k_pairs: list[tuple[float, float]] = [
        (0.0005, 0.003), (0.001, 0.005), (-0.001, 0.002),
        (-0.002, 0.004), (0.003, -0.001), (-0.005, 0.001),
    ]
    duration_s = 4.0
    idx = 0

    for v in speeds:
        for k0, k1 in k_pairs:
            t_arr = _time_array(duration_s)
            jump_frame = len(t_arr) // 2
            frames: list[dict[str, Any]] = []
            for i, t in enumerate(t_arr):
                k = k0 if i < jump_frame else k1
                path = _coherent_path(k, v)
                frame = _base_frame(t=t, v_ego=v, curvature=k)
                frame.update(path)
                frames.append(frame)
            scenarios.append(DemandScenario(
                kind="high_quality_path",
                title=f"nuplan jerk jump {k0:.4f}→{k1:.4f} v={v:.0f} #{idx}",
                duration_s=duration_s,
                frames=tuple(frames),
                thresholds=_make_nuplan_thresholds(),
            ))
            idx += 1

    return scenarios


def _nuplan_oscillation_scenarios(rng: random.Random) -> list[DemandScenario]:
    """Sustained oscillation: sinusoidal curvature input."""
    scenarios: list[DemandScenario] = []
    speeds = (15.0, 20.0, 25.0)
    amplitudes = (0.001, 0.002, 0.004)
    frequencies = (0.5, 1.0, 2.0)  # Hz
    duration_s = 8.0
    idx = 0

    for v in speeds:
        for amp in amplitudes:
            for freq in frequencies:
                t_arr = _time_array(duration_s)
                ramp_frames = 30
                frames: list[dict[str, Any]] = []
                for i, t in enumerate(t_arr):
                    ramp = min(1.0, i / max(ramp_frames, 1))
                    k = amp * math.sin(2 * math.pi * freq * t) * ramp
                    path = _coherent_path(k, v)
                    frame = _base_frame(t=t, v_ego=v, curvature=k)
                    frame.update(path)
                    frames.append(frame)
                kind = "high_quality_path"
                scenarios.append(DemandScenario(
                    kind=kind,
                    title=f"nuplan osc amp={amp:.4f} freq={freq:.1f}Hz v={v:.0f} #{idx}",
                    duration_s=duration_s,
                    frames=tuple(frames),
                    thresholds=_make_nuplan_thresholds(),
                ))
                idx += 1

    return scenarios


# ── ISO 3888-1 Double Lane Change ─────────────────────────────────────────────

def _generate_iso_3888_scenarios(request: LateralPresetRequest) -> list[DemandScenario]:
    """ISO 3888-1 double lane change maneuver.

    Produces scenarios at the standard 80 kph entry speed (22.2 m/s) plus
    a few speed variants.
    """
    speeds = (18.0, _ISO_3888_V_EGO, 26.0)  # ~65, 80, 94 kph
    scenarios: list[DemandScenario] = []
    idx = 0

    for v in speeds:
        title = f"iso 3888-1 double lane change v={v:.1f} m/s #{idx}"
        scenarios.append(_build_iso_3888_scenario(v, title, idx))
        idx += 1

    return scenarios


def _iso_3888_lateral_offset_and_curvature(t: float, v_ego: float) -> tuple[float, float]:
    """Compute lateral offset and analytic curvature for the ISO 3888-1 path.

    Returns ``(lateral_m, curvature_1pm)``.

    Curvature is computed analytically from the quintic smooth-step profile:
    ``s''(t) = 60t(1-t)(1-2t)``, ``κ ≈ y'' = offset/L² · s''(t)``.
    """
    offset = _ISO_3888_LANE_OFFSET
    x = t * v_ego

    def _smooth_step_d2(t_i: float) -> float:
        """Second derivative of quintic smooth step. Zero at endpoints."""
        if t_i <= 0.0 or t_i >= 1.0:
            return 0.0
        return 60.0 * t_i * (1.0 - t_i) * (1.0 - 2.0 * t_i)

    if x < 15.0:
        return 0.0, 0.0
    if x < 45.0:
        progress = (x - 15.0) / 30.0
        lat = offset * _smooth_step(progress)
        curv = (offset / (30.0 * 30.0)) * _smooth_step_d2(progress)
        return lat, curv
    if x < 70.0:
        return offset, 0.0
    if x < 95.0:
        progress = (x - 70.0) / 25.0
        lat = offset * (1.0 - _smooth_step(progress))
        curv = -(offset / (25.0 * 25.0)) * _smooth_step_d2(progress)
        return lat, curv
    return 0.0, 0.0


def _build_iso_3888_scenario(v_ego: float, title: str, idx: int) -> DemandScenario:
    """Build a single ISO 3888-1 scenario."""
    duration_s = 125.0 / v_ego
    t_arr = _time_array(duration_s)
    frames: list[dict[str, Any]] = []

    for i, t in enumerate(t_arr):
        lateral, curvature = _iso_3888_lateral_offset_and_curvature(t, v_ego)
        # Clamp curvature to road-reasonable bounds.
        curvature = max(-0.008, min(0.008, curvature))

        # Build path: parabolic centred at lateral offset.
        n = N_PATH_POINTS
        xs = [float(x) for x in range(n)]
        ys = [lateral + 0.5 * curvature * x * x for x in range(n)]
        path = {
            "position_x": xs,
            "position_y": ys,
            "position_y_std": [0.1] * n,
            "orientation_z": [curvature * x for x in range(n)],
            "orientation_rate_z": [curvature * v_ego] * n,
        }

        frame = _base_frame(t=t, v_ego=v_ego, curvature=curvature, lat_active=True)
        frame.update(path)
        frames.append(frame)

    return DemandScenario(
        kind="iso_3888_lane_change",
        title=title,
        duration_s=duration_s,
        frames=tuple(frames),
        thresholds=_make_iso_thresholds(),
    )


# ── Lateral Stress Grid ────────────────────────────────────────────────────────

def _curvature_grid_values() -> tuple[float, ...]:
    """Expand absolute curvature values to signed variants: 0 stays 0, others get ±."""
    values: list[float] = []
    for k_abs in _STRESS_CURVATURES:
        if k_abs == 0.0:
            values.append(0.0)
        else:
            values.append(k_abs)
            values.append(-k_abs)
    return tuple(values)


def _generate_stress_grid_scenarios(request: LateralPresetRequest) -> list[DemandScenario]:
    """4D parametric stress grid: speed × curvature × lane confidence × frame drop.

    Each cell is a constant-parameter DemandScenario. Designed to catch structural
    failures (NaN, divergence, excessive gating) across the full operating envelope.

    Use ``stress_grid_sample`` to run a random subset for CI.
    """
    scenarios: list[DemandScenario] = []
    rng = random.Random(request.seed)
    duration_s = 2.0

    curvatures = _curvature_grid_values()

    # Build all grid cells.
    cells: list[tuple[float, float, float, int]] = []
    for v in _STRESS_SPEEDS:
        for k in curvatures:
            for conf in _STRESS_CONFIDENCES:
                for drop in _STRESS_DROPS:
                    cells.append((v, k, conf, drop))

    if request.stress_grid_sample is not None and request.stress_grid_sample < len(cells):
        cells = rng.sample(cells, request.stress_grid_sample)

    thresholds = _make_stress_thresholds()
    idx = 0
    for v, k, conf, drop in cells:
        probs = (conf, conf, conf, conf)
        title = f"stress-grid v={v:.0f} k={k:.4f} conf={conf:.1f} drop={drop}% #{idx}"
        t_arr = _time_array(duration_s)
        frames: list[dict[str, Any]] = []
        for i, t in enumerate(t_arr):
            path = _coherent_path(k, v)
            frame = _base_frame(
                t=t, v_ego=v, curvature=k, lat_active=True,
                lane_line_probs=probs,
                frame_drop_perc=float(drop),
            )
            frame.update(path)
            frames.append(frame)
        scenarios.append(DemandScenario(
            kind="stress_grid",
            title=title,
            duration_s=duration_s,
            frames=tuple(frames),
            thresholds=thresholds,
        ))
        idx += 1

    return scenarios


# ── nuPlan Comfort (highway-realistic) ────────────────────────────────────────

# Highway curvatures: R ≥ 333 m (κ ≤ 0.003) — Euro NCAP S-Bend turn 1 is 787 m.
_COMFORT_MAX_K = 0.003

# nuPlan comfort-stress uses the full stress-grid curvature set.
_COMFORT_STRESS_CURVATURES = _STRESS_CURVATURES


def _build_nuplan_comfort_cells(
    request: LateralPresetRequest,
    max_k: float | None = None,  # None = no filter (stress variant)
) -> list[tuple[float, float, float, int]]:
    """Build the cell list for nuPlan comfort scenarios, optionally filtered by max κ."""
    curvatures = _curvature_grid_values()
    cells: list[tuple[float, float, float, int]] = []
    for v in _STRESS_SPEEDS:
        for k in curvatures:
            if max_k is not None and abs(k) > max_k:
                continue
            for conf in _STRESS_CONFIDENCES:
                for drop in _STRESS_DROPS:
                    cells.append((v, k, conf, drop))
    if request.stress_grid_sample is not None and request.stress_grid_sample < len(cells):
        rng = random.Random(request.seed)
        cells = rng.sample(cells, request.stress_grid_sample)
    return cells


def _generate_nuplan_comfort_frames(cells, thresholds, kind: str, label: str, ramp_frames: int = 30) -> list[DemandScenario]:
    """Shared frame generation for nuPlan comfort presets."""
    duration_s = 2.0
    result: list[DemandScenario] = []
    idx = 0
    for v, k, conf, drop in cells:
        probs = (conf, conf, conf, conf)
        title = f"{label} v={v:.0f} k={k:.4f} conf={conf:.1f} drop={drop}% #{idx}"
        t_arr = _time_array(duration_s)
        frames: list[dict[str, Any]] = []
        for i, t in enumerate(t_arr):
            ramp = min(1.0, i / max(ramp_frames, 1))
            k_ramped = k * ramp
            path = _coherent_path(k_ramped, v)
            frame = _base_frame(
                t=t, v_ego=v, curvature=k_ramped, lat_active=True,
                lane_line_probs=probs,
                frame_drop_perc=float(drop),
            )
            frame.update(path)
            frames.append(frame)
        result.append(DemandScenario(
            kind=kind,
            title=title,
            duration_s=duration_s,
            frames=tuple(frames),
            thresholds=thresholds,
        ))
        idx += 1
    return result


def _generate_nuplan_comfort_scenarios(request: LateralPresetRequest) -> list[DemandScenario]:
    """nuPlan comfort: highway-realistic curvatures (κ ≤ 0.003, R ≥ 333 m).

    Filters out sharp turns that would never appear on a highway, so failures
    flag genuinely uncomfortable highway behavior rather than racing maneuvers.
    """
    cells = _build_nuplan_comfort_cells(request, max_k=_COMFORT_MAX_K)
    return _generate_nuplan_comfort_frames(cells, _make_nuplan_comfort_thresholds(),
                                            kind="nuplan_comfort", label="nuplan-comfort")


def _generate_nuplan_comfort_stress_scenarios(request: LateralPresetRequest) -> list[DemandScenario]:
    """nuPlan comfort stress: full curvature grid with relaxed thresholds.

    Includes extreme curvatures (κ up to 0.02, R = 50 m). Uses structural
    thresholds (like stress-grid) because extreme maneuvers naturally exceed
    highway-driving comfort bounds. The nuPlan comfort bounds are recorded
    as metrics, not enforcement — see ``nuplan-comfort`` for enforcement.
    """
    cells = _build_nuplan_comfort_cells(request, max_k=None)
    return _generate_nuplan_comfort_frames(cells, _make_stress_thresholds(),
                                            kind="nuplan_comfort_stress", label="nuplan-comfort-stress")


# ── UN R79 ACSF Category C Lane-Change ────────────────────────────────────────

def _generate_un_r79_scenarios(request: LateralPresetRequest) -> list[DemandScenario]:
    """UN R79 Category C lane-change comfort scenarios.

    Generates lane-change-like curvature profiles constrained by the
    regulation-grade limits: lateral acceleration ≤ 1.0 m/s²,
    lateral jerk ≤ 5.0 m/s³ (0.5 s moving average), completion < 5 s.

    Each scenario follows a smooth lane-change curvature profile:
    straight → curve left → straight → curve right → straight.
    Peak curvature is speed-limited: κ_max = 1.0 / v².
    """
    scenarios: list[DemandScenario] = []
    thresholds = _make_un_r79_thresholds()
    duration_s = _UN_R79_DURATION_S
    idx = 0

    for v in _UN_R79_LANE_CHANGE_SPEEDS:
        # κ_max keeps lat_accel = v²·κ ≤ 1.0 m/s²
        k_max = _UN_R79_MAX_LAT_ACCEL / (v * v)
        for direction in (+1.0, -1.0):
            k_left = k_max * direction
            k_right = -k_max * direction
            speed_kph = v * 3.6
            label = "left→right" if direction > 0 else "right→left"
            title = f"un-r79 lane-change {label} {speed_kph:.0f}kph #{idx}"

            t_arr = _time_array(duration_s)
            frames: list[dict[str, Any]] = []

            # Segment timing: straight | ramp-in | hold | ramp-out | straight | ramp-in | hold | ramp-out | straight
            ramp_s = 1.5
            seg_s1 = 1.0
            seg_ri1 = seg_s1 + ramp_s
            seg_h1 = seg_ri1 + 1.0
            seg_ro1 = seg_h1 + ramp_s
            seg_s2 = seg_ro1 + 0.5
            seg_ri2 = seg_s2 + ramp_s
            seg_h2 = seg_ri2 + 1.0
            seg_ro2 = seg_h2 + ramp_s

            for i, t in enumerate(t_arr):
                if t < seg_s1:
                    k = 0.0
                elif t < seg_ri1:
                    k = k_left * _smooth_step((t - seg_s1) / ramp_s)
                elif t < seg_h1:
                    k = k_left
                elif t < seg_ro1:
                    k = k_left * (1.0 - _smooth_step((t - seg_h1) / ramp_s))
                elif t < seg_s2:
                    k = 0.0
                elif t < seg_ri2:
                    k = k_right * _smooth_step((t - seg_s2) / ramp_s)
                elif t < seg_h2:
                    k = k_right
                elif t < seg_ro2:
                    k = k_right * (1.0 - _smooth_step((t - seg_h2) / ramp_s))
                else:
                    k = 0.0

                path = _coherent_path(k, v)
                frame = _base_frame(t=t, v_ego=v, curvature=k, lat_active=True)
                frame.update(path)
                frames.append(frame)

            scenarios.append(DemandScenario(
                kind="un_r79_lane_change",
                title=title,
                duration_s=duration_s,
                frames=tuple(frames),
                thresholds=thresholds,
            ))
            idx += 1

    return scenarios


# ── C-NCAP Lane Centering Control (LCC) ───────────────────────────────────────

def _generate_cncap_lcc_scenarios(request: LateralPresetRequest) -> list[DemandScenario]:
    """C-NCAP / GB/T 39323-2020 lane centering control benchmark.

    Constant curvature κ = 0.002 m⁻¹ (R = 500 m) across a speed range,
    testing steady-state tracking precision. The curvature is ramped in
    over 0.5 s to avoid synthetic startup jerk.
    """
    scenarios: list[DemandScenario] = []
    thresholds = _make_cncap_lcc_thresholds()
    duration_s = _CNCAP_LCC_DURATION_S
    ramp_frames = 50  # 0.5 s smooth entry
    idx = 0

    for v in _CNCAP_LCC_SPEEDS:
        for sign in (+1.0, -1.0):
            k = _CNCAP_LCC_K * sign
            speed_kph = v * 3.6
            title = f"cncap LCC κ={k:.4f} {speed_kph:.0f}kph #{idx}"

            t_arr = _time_array(duration_s)
            frames: list[dict[str, Any]] = []
            for i, t in enumerate(t_arr):
                ramp = min(1.0, i / max(ramp_frames, 1))
                k_ramped = k * ramp
                path = _coherent_path(k_ramped, v)
                frame = _base_frame(t=t, v_ego=v, curvature=k_ramped, lat_active=True)
                frame.update(path)
                frames.append(frame)

            scenarios.append(DemandScenario(
                kind="cncap_lcc",
                title=title,
                duration_s=duration_s,
                frames=tuple(frames),
                thresholds=thresholds,
            ))
            idx += 1

    return scenarios


# ── SAE J3240 Tier Two — Perception Degradation ──────────────────────────────

# Each SAE J3240-inspired family represents a known environmental condition.
# Parameters are inferred from published NCHRP/SAE literature on lane-marking
# detection performance. These are "SAE J3240-inspired" — the exact SAE Tier Two
# thresholds are proprietary, but the parametric ranges are grounded in research.

_SAE_J3240_FAMILIES: tuple[tuple[str, str, tuple[float, ...], float, bool], ...] = (
    # (kind, label, lane_line_probs, frame_drop_perc, model_data_v2_sp_valid)
    ("dry_day", "Dry day, good markings", (0.95, 0.95, 0.95, 0.95), 0.0, True),
    ("worn_markings", "Worn markings", (0.55, 0.55, 0.55, 0.55), 5.0, True),
    ("wet_day", "Wet surface, day", (0.50, 0.50, 0.50, 0.50), 5.0, True),
    ("night_dry", "Night, dry", (0.80, 0.80, 0.80, 0.80), 0.0, True),
    ("night_wet", "Night, wet", (0.65, 0.65, 0.65, 0.65), 15.0, True),
    ("night_rain", "Night, continuous rain", (0.25, 0.25, 0.25, 0.25), 30.0, True),
    ("night_glare", "Night rain + glare", (0.15, 0.15, 0.15, 0.15), 40.0, False),
    ("heavy_fog", "Heavy fog < 50 m", (0.05, 0.05, 0.05, 0.05), 80.0, False),
)

_SAE_J3240_SPEEDS = (10.0, 20.0, 30.0)  # m/s — urban, highway, fast highway
_SAE_J3240_CURVATURES = (0.0, 0.001, 0.003)  # 1/m — straight, mild, moderate
_SAE_J3240_DURATION_S = 4.0

_SAE_J3240_THRESHOLD_KWARGS = {
    "max_abs_processed_curvature": 0.005,
    "max_abs_step_lat_accel": 5.0,
    "max_abs_lat_jerk": 40.0,
    "max_gated_curvature_lat_accel_delta": 5.0,
}


def _make_sae_j3240_thresholds():
    from .fuzz_lateral_demand import DemandFuzzThresholds
    return DemandFuzzThresholds(**_SAE_J3240_THRESHOLD_KWARGS)


def _generate_sae_j3240_scenarios(request: LateralPresetRequest) -> list[DemandScenario]:
    """SAE J3240-inspired perception degradation benchmark.

    Sweeps environmental conditions (dry/wet/night/fog) × speeds × curvatures.
    Each condition exercises the demand pipeline under reduced lane-line
    confidence and frame drop, simulating real-world sensor degradation.
    """
    scenarios: list[DemandScenario] = []
    thresholds = _make_sae_j3240_thresholds()
    duration_s = _SAE_J3240_DURATION_S
    idx = 0

    for family, label, probs, drop, sp_valid in _SAE_J3240_FAMILIES:
        for v in _SAE_J3240_SPEEDS:
            for k_abs in _SAE_J3240_CURVATURES:
                for sign in (+1.0, -1.0):
                    k = k_abs * sign if k_abs != 0.0 else 0.0
                    # Skip sign duplicate for k=0.
                    if k_abs == 0.0 and sign < 0:
                        continue
                    title = f"sae-j3240 {label} v={v:.0f} k={k:.4f} #{idx}"
                    t_arr = _time_array(duration_s)
                    frames: list[dict[str, Any]] = []
                    for i, t in enumerate(t_arr):
                        path = _coherent_path(k, v)
                        frame = _base_frame(
                            t=t, v_ego=v, curvature=k, lat_active=True,
                            lane_line_probs=probs,
                            frame_drop_perc=drop,
                            model_data_v2_sp_valid=sp_valid,
                        )
                        frame.update(path)
                        frames.append(frame)
                    scenarios.append(DemandScenario(
                        kind="sae_j3240_degraded",
                        title=title,
                        duration_s=duration_s,
                        frames=tuple(frames),
                        thresholds=thresholds,
                    ))
                    idx += 1

    return scenarios


# ── Combined longitudinal+lateral ─────────────────────────────────────────────

_COMBINED_DURATION_S = 4.0
_COMBINED_SPEEDS = (10.0, 20.0, 30.0)
_COMBINED_CURVATURES = (0.0, 0.001, 0.003)
_COMBINED_CONFIDENCES = (0.9, 0.5, 0.2)

_COMBINED_THRESHOLD_KWARGS = {
    "max_abs_processed_curvature": 0.005,
    "max_abs_step_lat_accel": 8.0,
    "max_abs_lat_jerk": 80.0,
    "max_gated_curvature_lat_accel_delta": 5.0,
}


def _make_combined_thresholds():
    from .fuzz_lateral_demand import DemandFuzzThresholds
    return DemandFuzzThresholds(**_COMBINED_THRESHOLD_KWARGS)


def _generate_combined_scenarios(request: LateralPresetRequest) -> list[DemandScenario]:
    """Combined longitudinal+lateral: dynamic speed + curvature + confidence.

    Unlike other presets where speed is constant, this varies speed over time
    within each scenario (accelerating, decelerating, or sinusoidal speed profile)
    while also varying curvature and lane-line confidence. Tests the demand
    pipeline's response to simultaneous longitudinal+lateral changes.
    """
    scenarios: list[DemandScenario] = []
    thresholds = _make_combined_thresholds()
    duration_s = _COMBINED_DURATION_S
    idx = 0

    speed_profiles = ("accelerating", "decelerating", "sinusoidal")
    for profile in speed_profiles:
        for v in _COMBINED_SPEEDS:
            for k_abs in _COMBINED_CURVATURES:
                for sign in (+1.0, -1.0):
                    k = k_abs * sign if k_abs != 0.0 else 0.0
                    if k_abs == 0.0 and sign < 0:
                        continue
                    for conf in _COMBINED_CONFIDENCES:
                        title = f"combined {profile} v={v:.0f} k={k:.4f} conf={conf:.1f} #{idx}"
                        t_arr = _time_array(duration_s)
                        frames: list[dict[str, Any]] = []
                        for i, t in enumerate(t_arr):
                            # Dynamic speed profile.
                            if profile == "accelerating":
                                v_ego = v * (0.5 + 0.5 * t / duration_s)
                            elif profile == "decelerating":
                                v_ego = v * (1.0 - 0.4 * t / duration_s)
                            else:  # sinusoidal
                                v_ego = v * (0.7 + 0.3 * math.sin(2 * math.pi * 0.5 * t))

                            # Curvature ramped in smoothly.
                            ramp = min(1.0, i / 30.0)
                            k_ramped = k * ramp

                            probs = (conf, conf, conf, conf)
                            path = _coherent_path(k_ramped, v_ego)
                            frame = _base_frame(
                                t=t, v_ego=v_ego, curvature=k_ramped, lat_active=True,
                                lane_line_probs=probs,
                            )
                            frame.update(path)
                            frames.append(frame)
                        scenarios.append(DemandScenario(
                            kind="combined",
                            title=title,
                            duration_s=duration_s,
                            frames=tuple(frames),
                            thresholds=thresholds,
                        ))
                        idx += 1

    return scenarios
