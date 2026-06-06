# Longitudinal validation packs

Use these focused packs when touching planner authority, mode resolution, stack policy, speed/map advisories, or Torque v4 lateral contracts.

## Longitudinal planner and mode contracts

```bash
uv run --extra testing --extra tools python -m pytest \
  selfdrive/controls/tests/test_longitudinal_candidate_authority.py \
  selfdrive/controls/tests/test_longitudinal_decision.py \
  selfdrive/controls/tests/test_lead_context.py \
  selfdrive/controls/tests/test_scc_evidence.py \
  selfdrive/controls/tests/test_longitudinal_modes.py \
  selfdrive/controls/tests/test_longitudinal_mode_contracts.py \
  selfdrive/controls/tests/test_longitudinal_profile.py \
  selfdrive/controls/tests/test_longitudinal_scenario_simulator.py \
  selfdrive/controls/tests/test_longitudinal_plan_schema_contract.py
```

This pack protects candidate authority, lead risk/progress classification, SCC evidence tiers, braking-profile helpers, deterministic scenario scaffolds, and append-only `LongitudinalPlanSP` telemetry IDs.

## Planner seeds and custom-stack policy

```bash
uv run --extra testing --extra tools python -m pytest \
  selfdrive/controls/tests/test_longitudinal_seed_scenarios.py \
  selfdrive/controls/tests/test_custom_longitudinal_stack.py \
  selfdrive/controls/tests/test_longitudinal_stack_selector.py \
  selfdrive/controls/tests/test_longitudinal_stack_policy.py \
  sunnypilot/selfdrive/controls/lib/tests/test_longitudinal_planner.py
```

This pack protects planner seed metadata, custom-v2 fail-closed behavior, stack selection, Sunnylink-visible telemetry, and retained planner publish helpers.

## Speed, map, and curve advisories

```bash
uv run --extra testing --extra tools python -m pytest \
  selfdrive/controls/tests/test_longitudinal_policy_horizon.py \
  selfdrive/controls/tests/test_curve_speed_policy.py \
  sunnypilot/selfdrive/controls/lib/speed_limit/tests/test_speed_limit_assist.py \
  sunnypilot/selfdrive/controls/lib/speed_limit/tests/test_speed_limit_planner_targets.py \
  sunnypilot/selfdrive/controls/lib/smart_cruise_control/tests/test_map_controller.py
```

This pack protects restrictive-only speed-limit, map, curve, OSM, and SCC advisory behavior. ACC-mode changes must not consume these actuation inputs.

## Torque v4/v4.1 and lateral-demand contracts

```bash
uv run --extra testing --extra tools python -m pytest \
  selfdrive/controls/tests/test_torque_v4_contracts.py \
  selfdrive/controls/tests/test_torque_v4_route_metrics.py \
  selfdrive/controls/tests/test_latcontrol_torque_v4.py
```

This pack protects ProcessedLateralDemand gating, Torque v4 learner rejection reasons, Torque v4.1 as a v4 governor-profile variant, and route-level v4 tracking metrics.

## Sunnylink migration sanity

```bash
uv run --extra testing --extra tools python -m pytest \
  sunnypilot/sunnylink/tests/test_params_sync.py
```

Use this when adding Params or telemetry that must survive Sunnylink metadata and migration checks.
