---
name: route-to-regression
description: Converts route-log findings into deterministic sunnypilot regression coverage using Drive Lab profiles, replay context, maneuver tests, fuzz seeds, or lateral disturbance scenarios. Use when the user asks to turn a drive log issue into a test, reproduce a route event, create a regression from longitudinal or lateral analysis, preserve a bug as a scenario, or validate a planned behavior fix against logged evidence.
---

# Route To Regression

## Quick Start

Use this skill after a longitudinal or lateral log-analysis finding has a route, segment, timestamp, or event cluster.

```bash
uv run --extra tools python -m openpilot.tools.drive_lab.explain_route_event <route-or-log> --time <seconds> --before 10 --after 10 --qlog
uv run --extra tools python -m openpilot.tools.drive_lab.profile_route <route-or-log> --output /tmp/drive-lab-profile.json
uv run --extra tools python -m openpilot.tools.drive_lab.fuzz_longitudinal --seed 1 --mode comfort --profile /tmp/drive-lab-profile.json --cases 100
uv run --extra tools python -m openpilot.tools.drive_lab.simulate_lateral_disturbance --route <route-or-log> --qlog
```

## Guardrails

Do not tune planner or controller behavior directly from a route cluster before reducing it to a reproducible test, replay note, or synthetic scenario.
Keep raw route evidence immutable; copy logs read-only into `/tmp/opencode/route-to-regression/` when local analysis is needed.
Prefer qlogs for initial reduction and rlogs for final assertions that depend on high-rate control, torque, steering, lead, or lane-change state.
Preserve safety behavior in regressions: close leads, high required decel, FCW risk, manual steering override, lane-change intent, torque saturation, and roll/lateral-accel validity.
Put Drive Lab tooling changes on `feat/offline-drive-analysis`; put product behavior tests on the retained domain branch that owns the behavior under test.

## Workflow

1. Start from a concrete route ID, log path, segment range, timestamp, bookmark, or ranked Drive Lab cluster.
2. Re-run the relevant log-analysis tool and `explain_route_event` around the event window to capture evidence before writing code.
3. Decide the smallest regression shape: unit helper test, longitudinal maneuver test, Drive Lab fuzz seed/profile, lateral disturbance simulation, replay instruction, or integration test.
4. Extract only the necessary values: speed, acceleration, lead distance, relative speed, desired and actual curvature, steering state, torque, planner source, stack state, lane-change state, and override state.
5. Write the regression on the owning retained branch, not on `custom`.
6. Make the test fail or demonstrate the current risk before changing product code when feasible.
7. Implement the smallest behavior or tooling change needed to pass the regression.
8. Re-run the new regression plus nearby owning tests with `uv run --extra testing --extra tools python -m pytest <paths>`.
9. Document route provenance in the test name or a short comment without embedding private paths, secrets, or large log data.
10. Summarize what the regression proves and what still requires real-drive validation.

## Regression Shapes

Use a unit helper test when the route exposes a pure threshold, interpolation, source-selection, or candidate-arbitration bug.
Use a longitudinal maneuver test when timing over several seconds matters for following, stopping, launch, cruise coast, lead transition, or engage bootstrap.
Use Drive Lab fuzz/profile changes when the route reveals a missing scenario distribution or adversarial edge case.
Use lateral disturbance simulation when the route exposes steering oscillation, torque lag, authority loss, reversal sensitivity, or curve-entry behavior.
Use replay notes when camera/model context, process interaction, or UI behavior cannot be reduced safely yet.

## Evidence To Report

Report route ID, segment, timestamp window, original symptom, active or override state, extracted values, selected regression shape, owning branch, new or modified test path, command run, and whether the regression is qlog-derived, rlog-confirmed, synthetic, or replay-only.

## Safe Improvement Criteria

A good route-derived regression is deterministic, minimal, branch-owned, and traceable to the observed event without depending on unavailable logs. It should fail for the original bug or protect the behavior boundary that made the route suspicious.
