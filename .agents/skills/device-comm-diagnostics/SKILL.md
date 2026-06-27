---
name: device-comm-diagnostics
description: Diagnose on-device openpilot process communication alerts using deploy config, SSH health checks, manager/process state, and swaglog commIssue records. Use when the user reports "Communication Issue Between Processes", "Low Communication Rate Between Processes", "Process Not Running", or asks to debug IPC/msgq/process health on the comma device.
---

# Device Communication Diagnostics

## Quick start

From the repo root, run the read-only collector:

```bash
bash .agents/skills/device-comm-diagnostics/scripts/collect_device_comm_diagnostics.sh
```

It uses `.deploy-config` for `DEPLOY_HOST` and `DEPLOY_PATH`, then prints device uptime,
deployed commit, dirty/untracked checkout state, relevant processes, IPC/log paths, and recent
`commIssue` / `process_not_running` swaglog records.

## Workflow

1. Confirm device target from `.deploy-config` and `scripts/deploy.sh`.
   - Default host is `DEPLOY_HOST`; pass a different config path to the collector if needed.
   - Do not deploy, reboot, restart manager, or remove files unless the user explicitly asks.
2. Establish the current state.
   - `uptime` proves SSH access.
   - `git log -1 --oneline` must match the intended deployed commit.
   - `git status --short` highlights device-local artifacts that may affect runtime.
3. Interpret process state carefully.
   - Offroad: `selfdrived`, `controlsd`, `modeld`, `radard`, `locationd`, `paramsd` may be absent.
   - Onroad: manager should start all `only_onroad` processes from `system/manager/process_config.py`.
4. Use swaglog as the primary signal for UI communication alerts.
   - Logs live in `/data/log/swaglog.*` via `system/logmessaged.py` and `common/swaglog.py`.
   - `selfdrive/selfdrived/selfdrived.py` raises:
     - `selfdrived.initialized` after all checks pass or after the startup timeout.
     - `process_not_running` when `managerState.processes` has `shouldBeRunning && !running`.
     - `commIssue` when `SubMaster.all_checks()` fails because a service is invalid/not alive.
     - `commIssueAvgFreq` when services are alive but below expected frequency.
5. Map bad services back to publishers.
   - Treat `alertDebug`, `lateralManeuverPlan`, `modelDataV2SP`, sensors, and GPS entries as often
     ignored/noisy in `selfdrived`; focus first on non-ignored entries.
   - `longitudinalPlan`, `driverAssistance`, `longitudinalPlanSP`: `selfdrive/controls/plannerd.py`.
   - `longitudinalPlan.valid` depends on `carState`, `controlsState`, `selfdriveState`, `radarState`.
   - `driverAssistance.valid` depends on `carState`, `carControl`, `modelV2`, `liveParameters`.
   - Service rates/validity expectations are in `cereal/services.py`.
6. Form hypotheses only after extracting the exact `invalid`, `not_alive`, and `not_freq_ok` lists.

## Interpretation rules

- `selfdrived.initialized timeout=True` with many invalid/not-alive/not-freq services usually indicates
  an onroad startup race. Treat it separately from later steady-state alerts.
- If only `longitudinalPlan`, `driverAssistance`, and/or `longitudinalPlanSP` are invalid while they are
  alive and frequency-ok, `plannerd` is publishing `valid=False`; inspect its input checks next.
- If all three planner outputs are invalid, the shared hidden dependency is usually `carState` from
  `plannerd`'s perspective. `selfdrived` reads `carState` separately, so this may not appear directly in
  the `selfdrived` SubMaster lists.
- Route logs can show producer messages as top-level valid/frequent while `plannerd` still sees an
  internal receive-side `alive`/`freq_ok` failure at publish time. Prefer `plannerd_validity` swaglog
  events when available because they capture `plannerd`'s actual `SubMaster` state.
- Do not suppress planner-output invalidity in `selfdrived`; that can mask a real control dependency issue.

## Next probe for planner-output-only alerts

Find the matching route window and inspect validity/frequency for:

```text
carState, controlsState, carControl, modelV2, liveParameters, radarState, selfdriveState,
longitudinalPlan, driverAssistance, longitudinalPlanSP
```

Falsify the `carState` hypothesis if `carState` is valid/alive/frequency-ok throughout the alert while
another `plannerd` dependency is bad. If route logs cannot answer it, inspect `plannerd_validity` logs
or add temporary change-only `plannerd` instrumentation for failed `sm.all_checks(...)` inputs, then
remove or demote it after diagnosis.

Log choice:

- Use `qlog.zst` only for coarse correlation: route/segment, alert timing, engagement windows, and whether expected messages appear at all.
- Use `rlog.zst` for the actual comm diagnosis: exact `valid`/`alive`/`freq_ok` behavior, dropped/update timing, rate checks, and any conclusion about planner dependencies.
- Prefer `plannerd_validity` swaglog records over route logs when available because they capture `plannerd`'s receive-side `SubMaster` state at publish time.

`plannerd_validity` should include output booleans, failed lists by category, and per-service
`valid`/`alive`/`freqOk`/`updated`/`frameAge`/frequency tracker stats for:

```text
carState, controlsState, carControl, modelV2, liveParameters, radarState, selfdriveState
```

## Cautions

- Broad `journalctl` searches are noisy on-device; prefer `/data/log/swaglog.*` for openpilot events.
- A missing onroad process while the device is offroad is not a failure.
- Repeated sunnylink/Tailscale/network warnings can obscure the real `selfdrived` commIssue records.
- If SSH drops, report the last confirmed state and ask the user to wake/connect the device before continuing.
