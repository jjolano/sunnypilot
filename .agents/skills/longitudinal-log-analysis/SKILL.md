---
name: longitudinal-log-analysis
description: Analyzes recent sunnypilot device route logs for longitudinal planner/manual-driving agreement using Drive Lab tools. Use when the user asks to inspect drive logs, compare planner targets to manual driving, analyze lead/source flicker, validate longitudinal behavior from route data, or decide whether logged potential actions match human driving.
---

# Longitudinal Log Analysis

## Quick Start

Use this skill from the `custom` admin worktree unless intentionally working in a Drive Lab retained-branch worktree.

```bash
source .sync-config
ssh -o ConnectTimeout=10 "$DEPLOY_HOST" "ls /data/media/0/realdata | tail"
uv run --extra tools python -m openpilot.tools.drive_lab.profile_manual_longitudinal <routes-or-qlogs> --qlog --min-manual-moving 100 --max-active-ratio 0.01
uv run --extra tools python -m openpilot.tools.drive_lab.explain_route_event <route-or-log> --time <seconds> --before 6 --after 6 --qlog
```

Download logs read-only into `/tmp/opencode/longitudinal-log-analysis/` when local analysis is needed. Prefer qlogs for broad scans; pull rlogs only for suspicious clusters that need higher-rate confirmation.

## Guardrails

Do not change driving behavior from manual/disengaged logs alone.
Separate preview-only samples from actuation-applicable samples before drawing conclusions.
Treat `selfdriveState.active=false`, `carControl.longActive=false`, gas, brake, or `controlsState.longControlState=off` as preview/override context.
Treat disengaged cruise-source samples with unset cruise speed as low-confidence for behavior tuning.
Use route evidence to design regression scenarios before tuning planner/controller code.
Keep Drive Lab tooling changes on `feat/offline-drive-analysis`; put core longitudinal behavior changes on `feat/longitudinal-control`, speed/map/SCC changes on `feat/speed-map-control`, and learning/stat fixes on `feat/control-learning-stats` unless ownership clearly says otherwise.

## Workflow

1. Read `.sync-config` for `DEPLOY_HOST` and `DEPLOY_PATH`; do not assume environment variables are exported.
2. List recent device route prefixes under `/data/media/0/realdata` and identify complete segments with `qlog.zst` or `rlog.zst`.
3. Copy selected logs to `/tmp/opencode/longitudinal-log-analysis/ROUTE/` using SSH/tar or another read-only transfer.
4. Run `profile_manual_longitudinal` to classify manual style, active ratio, moving samples, launches, stops, following bins, and crawl episodes.
5. Compare `longitudinalPlan.aTarget` to `carState.aEgo`, gas/brake state, planner source, lead status, lead distance, lead relative speed, and `longitudinalPlanSP` stack/source.
6. Group findings into contiguous episodes, not isolated samples. Include source flips, lead-status flips, planner-target jerk, and whether the driver was overriding.
7. Use `explain_route_event` around suspicious timestamps. Re-run on rlog if qlog cadence is insufficient.
8. Summarize what is safe to conclude, what is preview-only, and what needs video, rlog, replay, or synthetic regression before behavior changes.

## Metrics To Report

Report route IDs, segment numbers, timestamps, sample counts, active/manual ratio, planner-target vs measured-accel correlation, mean and P90 absolute error, opposite-intent sample ratio, strong opposite-intent sample ratio, `shouldStop` conflicts while moving, FCW count, and top suspicious clusters.

For each cluster, report `vEgo`, `aEgo`, `aTarget`, planner source, SP source/stack, gas/brake state, lead status, `dRel`, `vRel`, `shouldStop`, lead/source flips, and attribution from Drive Lab.

## Safe Improvement Criteria

Only propose product changes when a cluster is actuation-applicable or reproducible in a scenario. A safe candidate must preserve hard braking for close leads, high required decel, FCW risk, confirmed cut-ins, and model-confirmed hazards. Prefer instrumentation or scenario extraction before smoothing or threshold changes.
