---
name: lateral-log-analysis
description: Analyzes recent sunnypilot device route logs for lateral control, steering, curvature tracking, torque behavior, oscillation, and lane-change context using Drive Lab tools. Use when the user asks to inspect drive logs for lateral behavior, steering wiggle, torque lag, oscillation, lane-change path issues, lateral tracking errors, roll/lateral-accel concerns, or compare lateral behavior across routes/builds.
---

# Lateral Log Analysis

## Quick Start

Use this skill from the `custom` admin worktree unless intentionally working in a Drive Lab retained-branch worktree.

```bash
source .sync-config
ssh -o ConnectTimeout=10 "$DEPLOY_HOST" "ls /data/media/0/realdata | tail"
uv run --extra tools python -m openpilot.tools.drive_lab.profile_lateral_events <routes-or-qlogs> --qlog --max-events 15
uv run --extra tools python -m openpilot.tools.drive_lab.profile_lateral_torque_events <routes-or-qlogs> --qlog --max-events 12
uv run --extra tools python -m openpilot.tools.drive_lab.profile_lateral_oscillation <routes-or-qlogs> --qlog
uv run --extra tools python -m openpilot.tools.drive_lab.profile_lateral_performance <routes-or-qlogs> --qlog
```

Download logs read-only into `/tmp/opencode/lateral-log-analysis/` when local analysis is needed. Prefer qlogs for broad scans; pull rlogs only for suspicious clusters that need higher-rate steering, torque, lane-change, or controller-state confirmation.

## Guardrails

Do not change lateral driving behavior from disengaged, manually steered, or override-heavy logs alone.
Separate active lateral-control samples from preview, inactive, lane-change, and driver-intervention context before drawing conclusions.
Treat `selfdriveState.active=false`, `carControl.latActive=false`, `carState.steeringPressed=true`, high driver steering torque, or lateral-control inactive state as preview/override context.
Treat qlog-only lane-change state, torque saturation, and fast-reversal conclusions as candidates for rlog confirmation when they drive product decisions.
Use route evidence to design regression scenarios before tuning controller, model-path, lane-change, or learned-stat behavior.
Keep Drive Lab tooling changes on `feat/offline-drive-analysis`; put controller-side lateral behavior changes on `feat/lateral-control` and learning/stat fixes on `feat/control-learning-stats` unless ownership clearly says otherwise.

## Workflow

1. Read `.sync-config` for `DEPLOY_HOST` and `DEPLOY_PATH`; do not assume environment variables are exported.
2. List recent device route prefixes under `/data/media/0/realdata` and identify complete segments with `qlog.zst` or `rlog.zst`.
3. Copy selected logs to `/tmp/opencode/lateral-log-analysis/ROUTE/` using read-only transfer.
4. Run `profile_lateral_events` to find slow-wander, rebound, and fast-reversal clusters.
5. Run `profile_lateral_torque_events` to inspect torque telemetry, lag, saturation, and low-speed lateral tiers.
6. Run `profile_lateral_oscillation` for straight-road steering and curvature oscillation windows.
7. Run `profile_lateral_performance` for the combined lateral gate and optional baseline-vs-candidate comparison.
8. Use `compare_lateral_torque` when comparing two builds or routes for torque tracking lag.
9. Re-run suspicious clusters on rlogs if qlog cadence or missing fields could hide steering override, torque saturation, lane-change state, or controller transitions.
10. Summarize what is safe to conclude, what is override/preview-only, and what needs video, rlog, replay, or synthetic regression before behavior changes.

## Metrics To Report

Report route IDs, segment numbers, timestamps, sample counts, active lateral ratio, steering override ratio, speed range, lane-change/blinker context, steering angle peak-to-peak, desired-vs-actual curvature error, model raw/processed/desired curvature differences, lateral acceleration, roll/lateral-accel validity where available, controller output, applied torque, torque saturation, torque lag, fast reversals, oscillation windows, and top suspicious clusters.

For each cluster, report `vEgo`, `steeringAngleDeg`, `steeringPressed`, lateral active state, lane-change state, blinker state, desired curvature, actual curvature, raw/processed model curvature, lateral-control output, applied torque, driver torque where available, and whether the event appears model-path, controller, learned-stat, lane-change, or driver-intervention related.

## Safe Improvement Criteria

Only propose product changes when a cluster is active-lateral-control applicable, reproducible, or confirmed by rlog/video. A safe candidate must preserve manual override handling, lane-change intent, high-curvature authority needs, controller saturation safety, and roll/lateral-accel validity checks. Prefer instrumentation, scenario extraction, or Drive Lab regression before controller tuning or smoothing changes.
