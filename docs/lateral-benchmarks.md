# Lateral Benchmarking Reference

Drive Lab lateral benchmark ecosystem — presets, fuzzers, profiling, and gating.

## Quick Start

```bash
# Run any preset through the demand pipeline fuzzer
uv run --extra testing --extra tools python tools/drive_lab/fuzz_lateral_demand.py --preset nhtsa-lka
uv run --extra testing --extra tools python tools/drive_lab/fuzz_lateral_demand.py --preset stress-grid --stress-grid-sample 50

# Profile-guided fuzzing from real route logs
uv run --extra testing --extra tools python tools/drive_lab/profile_lateral.py ROUTE --output profile.json
uv run --extra testing --extra tools python tools/drive_lab/fuzz_lateral_demand.py --preset fuzz --profile profile.json

# Direct controller fuzzer (closed-loop)
uv run --extra testing --extra tools python tools/drive_lab/fuzz_lateral_controller.py --cases 100

# CI gating
uv run --extra testing --extra tools python tools/drive_lab/fuzz_lateral_demand.py --preset nhtsa-lka --export-specs /tmp/specs.json
uv run --extra testing --extra tools python tools/drive_lab/behavior_change_gate.py /tmp/specs.json --domain lateral-synthetic --json

# Run all tests
uv run --extra testing --extra tools python -m pytest tools/drive_lab/tests/ -q
```

## Presets

| # | Preset | Count | Source | Description |
|---|--------|-------|--------|-------------|
| 1 | `fuzz` | seeded | Drive Lab | Random generators (profile-guideable) |
| 2 | `nhtsa-lka` | 50 | NHTSA NCAP | Lane departure grid: 5 velocities × 5 line types × 2 dirs |
| 3 | `euroncap-lss` | 52 | Euro NCAP | LKA/ELK (40) + AD S-Bend (6) + ALC (6) |
| 4 | `nuplan-lateral` | 95 | nuPlan | Error (50) + jerk transients (18) + oscillation (27) |
| 5 | `iso-3888` | 3 | ISO 3888-1 | Double lane change (analytic curvature) |
| 6 | `stress-grid` | 1848 | Drive Lab | 4D parametric: speed × curvature × confidence × drop |
| 7 | `nuplan-comfort` | 504 | nuPlan | Highway comfort (κ ≤ 0.003, R ≥ 333 m) |
| 8 | `nuplan-comfort-stress` | 1848 | nuPlan | Full envelope structural |
| 9 | `un-r79` | 8 | UN R79 Cat. C | Lane-change: ≤ 1.0 m/s² accel, ≤ 5.0 m/s³ jerk |
| 10 | `cncap-lcc` | 10 | C-NCAP | Constant curvature κ = 0.002 (R = 500 m) |
| 11 | `sae-j3240` | 120 | SAE J3240 | Perception degradation: 8 conditions × 3 speeds × 5 κ |
| 12 | `combined` | 243 | Drive Lab | Dynamic speed + curvature + confidence |
| 13 | `commonroad-lateral` | 4 | CommonRoad | Lane-level fixture scenarios |

### Preset Filters

```bash
# NHTSA: filter by test family, line type, or drift rate
--nhtsa-family primary|secondary
--nhtsa-line-type solid_white
--nhtsa-drift-rate 0.3

# Euro NCAP: filter by test family
--euroncap-family lka|elk|sbend|alc

# nuPlan: filter by metric focus
--nuplan-focus error|jerk|oscillation

# Stress grid / comfort: sample subset
--stress-grid-sample 100
```

## Fuzzers

### Demand Pipeline Fuzzer (`fuzz_lateral_demand.py`)

Exercises `LateralDemandPipeline` with synthetic per-frame inputs. Checks finite outputs, bounded curvature, sane rate/jerk, path quality, gating behavior.

```bash
uv run python tools/drive_lab/fuzz_lateral_demand.py --preset nhtsa-lka --json
uv run python tools/drive_lab/fuzz_lateral_demand.py --kind high_quality_path --cases 50
```

### Closed-Loop Demand Fuzzer (`fuzz_lateral_closed_loop.py`)

Demand pipeline output → synthetic vehicle plant. Two-layer validation: demand invariants + plant structural checks.

```bash
uv run python tools/drive_lab/fuzz_lateral_closed_loop.py --preset nhtsa-lka
```

### Controller Fuzzer (`fuzz_lateral_controller.py`)

Exercises `LatControlTorqueV21` (response core → governor) directly. Closed-loop mode (default) feeds torque through a steering plant; open-loop mode checks structural invariants only.

```bash
uv run python tools/drive_lab/fuzz_lateral_controller.py --cases 100
uv run python tools/drive_lab/fuzz_lateral_controller.py --open-loop --cases 100
```

### Transition Fuzzer (`fuzz_lateral_transitions.py`)

Lateral state-transition fuzzer for `LateralDemandPipeline`.

```bash
uv run python tools/drive_lab/fuzz_lateral_transitions.py --preset fuzz --cases 50
```

### Route Replay Fuzzer (`fuzz_lateral_route_replay.py`)

Extracts frames from real route logs, perturbs and replays through demand pipeline.

```bash
uv run python tools/drive_lab/fuzz_lateral_route_replay.py --route ROUTE --perturbation noise
```

## Profiling

### Lateral Route Profiling (`profile_lateral.py`)

Extracts speed, curvature, lane-line confidence, and road roll from route logs into a `LateralProfile` for profile-guided fuzzing.

```bash
uv run python tools/drive_lab/profile_lateral.py ROUTE --output profile.json
uv run python tools/drive_lab/profile_lateral.py ROUTE --json
```

### Profile-Guided Fuzzing

Use route-derived profiles to bias synthetic scenario generation toward real-world parameter ranges.

```bash
uv run python tools/drive_lab/fuzz_lateral_demand.py --preset fuzz --profile profile.json --cases 100
```

## CI Gating

Export scenarios to `ScenarioSpec` JSON and validate with `behavior_change_gate.py`:

```bash
# Export
uv run python tools/drive_lab/fuzz_lateral_demand.py --preset nhtsa-lka --export-specs /tmp/specs.json

# Gate (synthetic domain)
uv run python tools/drive_lab/behavior_change_gate.py /tmp/specs.json --domain lateral-synthetic --json
# → {"ready": true, "matching_count": 50, ...}
```

## Architecture

```
Route logs
    │
    ├──► profile_lateral.py ──► LateralProfile ──► --profile (guided fuzz)
    │
    └──► fuzz_lateral_route_replay.py ──► LateralRouteFrame ──► pipeline

Synthetic scenarios (lateral_scenarios.py)
    │
    ├──► fuzz_lateral_demand.py ──► LateralDemandPipeline ──► structural checks
    ├──► fuzz_lateral_closed_loop.py ──► pipeline + plant ──► 2-layer checks
    ├──► fuzz_lateral_transitions.py ──► pipeline transition checks
    └──► fuzz_lateral_controller.py ──► LatControlTorqueV21 ──► torque checks

Export ──► ScenarioSpec JSON ──► behavior_change_gate.py ──► ready/not-ready
```

## Test Coverage

- `tools/drive_lab/tests/` — 428 tests
- `sunnypilot/custom/lateral/tests/` — Controller integration tests

## Key Thresholds Reference

| Source | Metric | Value |
|--------|--------|-------|
| UN R79 Cat. C | Lateral acceleration (lane change) | ≤ 1.0 m/s² |
| UN R79 Cat. C | Lateral jerk (0.5 s MA) | ≤ 5.0 m/s³ |
| nuPlan | Lateral acceleration (comfort) | 4.89 m/s² |
| nuPlan | Jerk magnitude (comfort) | 8.37 m/s³ |
| Euro NCAP 2026+ | Steering override torque | ≤ 3.0 Nm |
| Euro NCAP 2026+ | Continuous intervention | ≤ 1.0 s |
| C-NCAP | Lane centering curvature | κ = 0.002 (R ≤ 500 m) |
| IIHS 2018 | Curve radii tested | 396–610 m |
