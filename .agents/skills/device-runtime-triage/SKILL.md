---
name: device-runtime-triage
description: Diagnoses a deployed sunnypilot device runtime without rebuilding or deploying, using SSH health checks, process state, journal output, params, and checkout state. Use when the user says the device is broken, offline, crash-looping, not starting, missing processes, showing import errors, having Tailscale or SSH issues, or asks for a deploy health check, runtime triage, or baseline reset investigation.
---

# Device Runtime Triage

## Quick Start

Use this skill from the `custom` admin worktree. Read `.sync-config` for `DEPLOY_HOST` and `DEPLOY_PATH`; do not assume they are already exported.

```bash
source .sync-config
ssh -o ConnectTimeout=10 "$DEPLOY_HOST" "uptime"
ssh "$DEPLOY_HOST" "cd '$DEPLOY_PATH' && git log -1 --oneline && git status --short"
ssh "$DEPLOY_HOST" "pgrep -af manager"
ssh "$DEPLOY_HOST" "pgrep -af 'selfdrive.ui.ui|pandad|loggerd|modeld|controlsd|selfdrived|locationd|paramsd|radard|manage_tailscaled'"
ssh "$DEPLOY_HOST" "journalctl --since '10 min ago' 2>/dev/null | rg -i 'traceback|ImportError|ModuleNotFoundError|exception|crash|segfault'"
```

## Guardrails

Start read-only. Do not reboot, reset params, delete files, restart services, deploy, or hard-reset the device unless the user explicitly approves that action.
Do not treat onroad-only processes as missing when the device is offroad; `controlsd`, `modeld`, `radard`, and related controls processes may be absent offroad.
Do not change product code during runtime triage until the failing import, process, param, or deployed commit is identified.
Do not run the optional baseline reset playbook unless the user asks for a reset or approves it after seeing the exact params that will be removed.
Capture the exact failed command and output before proposing fixes.

## Workflow

1. Read `.sync-config` and identify `DEPLOY_HOST`, `DEPLOY_PATH`, and expected branch or commit.
2. Check SSH reachability with a short timeout and report if the device is unreachable, slow, or refusing connections.
3. Verify the deployed checkout with `git log -1 --oneline`, `git status --short`, and optionally `git remote -v`.
4. Check manager and core process state with `pgrep -af`; classify missing processes as expected offroad, suspicious, or confirmed crash-loop.
5. Inspect recent journal output for tracebacks, import errors, module errors, crashes, safety faults, manager restarts, and repeated process exits.
6. For import errors, run a focused Python import check from `DEPLOY_PATH` with `PYTHONPATH='$DEPLOY_PATH'` and the device venv.
7. For Tailscale issues, check `manage_tailscaled`, Tailscale params, daemon logs, and network reachability before reinstalling or changing params.
8. For behavior anomalies after deploy, compare local `custom` HEAD to the deployed commit before changing code.
9. For learning or control-state suspicion, inspect relevant params read-only before recommending a baseline reset.
10. Summarize root cause, confidence, exact evidence, and the smallest safe next action.

## Checks To Report

Report SSH reachability, uptime, deployed commit, checkout cleanliness, manager PID, core process PIDs, recent crash/import log lines, whether device appears onroad or offroad, free disk concerns if checked, relevant params inspected, and any mismatch between local `custom` and deployed HEAD.

For import triage, report the module name, stack trace line, local file existence, deployed file existence, and whether the error is from stale deploy, missing generated module, missing dependency, or product-code import path.

## Safe Recovery Criteria

Prefer the least destructive recovery. A safe recovery either fixes the deployed checkout to an already-built `custom` commit, restores a missing dependency or generated file through the documented deploy path, or asks for approval before rebooting, resetting params, or redeploying.
