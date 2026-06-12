# AGENTS.md

Instructions and context for AI agents working in this repository.

## Project Overview

This repository is a custom fork of [sunnypilot](https://github.com/sunnypilot/sunnypilot), which is itself a fork of [commaai/openpilot](https://github.com/commaai/openpilot).

## Repo-Local Skills

Repo-local skills live under `.agents/skills/`, the OpenCode-compatible project skill directory. When a user request matches one of these descriptions, load or read that skill's `SKILL.md` before acting.

- `rebuild-deploy-workflow`: Runs the sunnypilot retained-branch propagation, `custom` rebuild, deploy, and post-deploy health-check workflow safely. Use when the user asks to rebuild custom, deploy custom, propagate retained branches before deployment, or validate a deployed branch.
- `sync-upstream-workflow`: Runs the sunnypilot upstream synchronization workflow safely, including default `master` fast-forward sync and optional retained-branch sync/rebuild flows. Use when the user asks to sync upstream, sync retained branches, update from `upstream/master`, or prepare retained branches after upstream changes.
- `longitudinal-log-analysis`: Analyzes recent device route logs for longitudinal planner/manual-driving agreement using Drive Lab tools. Use when the user asks to inspect drive logs, compare planner targets to manual driving, analyze lead/source flicker, validate longitudinal behavior from route data, or decide whether logged potential actions match human driving.
- `lateral-log-analysis`: Analyzes recent device route logs for lateral control, steering, curvature tracking, torque behavior, oscillation, and lane-change context using Drive Lab tools. Use when the user asks to inspect drive logs for lateral behavior, steering wiggle, torque lag, oscillation, lane-change path issues, lateral tracking errors, roll/lateral-accel concerns, or compare lateral behavior across routes/builds.
- `device-runtime-triage`: Diagnoses a deployed device runtime without rebuilding or deploying, using SSH health checks, process state, journal output, params, and checkout state. Use when the user says the device is broken, offline, crash-looping, not starting, missing processes, showing import errors, having Tailscale or SSH issues, or asks for a deploy health check, runtime triage, or baseline reset investigation.
- `route-to-regression`: Converts route-log findings into deterministic regression coverage using Drive Lab profiles, replay context, maneuver tests, fuzz seeds, or lateral disturbance scenarios. Use when the user asks to turn a drive log issue into a test, reproduce a route event, create a regression from log analysis, preserve a bug as a scenario, or validate a planned behavior fix against logged evidence.
- `retained-branch-change-workflow`: Guides code changes onto the correct retained branch and worktree, including ownership selection, setup, testing, commit readiness, and propagation reminders. Use when the user asks to implement or fix product behavior, add retained feature code, choose a branch, create a worktree, prepare a retained branch for rebuild/deploy, or avoid putting long-term changes on `custom`.

**Upstream chain:**
```text
commaai/openpilot  ->  sunnypilot/sunnypilot  ->  jjolano/sunnypilot (this repo)
     (origin)             (upstream remote)          (origin remote)
```

## Current Branch Model

Only these branches are part of the live workflow:

| Branch | Purpose | Rules |
|---|---|---|
| `master` | Pristine mirror of `upstream/master` | Fast-forward only. Never commit directly. |
| `feat/retained-baseline` | Squashed retained product snapshot and shared upstream-sync base | Receives upstream `master` merges. No feature tuning unless it is a baseline compatibility fix. |
| `feat/device-admin` | Device/admin runtime support | Tailscale, startup comm health, device runtime/process/health support. |
| `feat/longitudinal-control` | Core longitudinal behavior | Follow gap, lead transition, engage bootstrap, stop approach, cruise coast, FCW, launch, decision layer, and stack selector. |
| `feat/speed-map-control` | Speed, map, SCC, and OSM behavior | SCC vision/map, OSM/mapd enrichment, speed-limit resolver, and speed-limit auto-cruise. |
| `feat/lateral-control` | Controller-side lateral behavior | Torque controller, lane-change path shaping, lateral model path processing, steering actuator feedback, and lateral controller tuning. |
| `feat/control-learning-stats` | Learned control and calculated-stat correctness | Live-learning expansion, mass/drag and response learning, calculated roll/lateral-accel/curvature correctness, accurate lateral accel. |
| `feat/offline-drive-analysis` | Offline analysis and regression tooling | Drive Lab route/log analysis, replay/test generation, fuzzing, and scenario extraction tools. |
| `custom` | Disposable integration/deploy branch | Rebuilt from `master + MERGE_ORDER + custom-only files`, then force-pushed/deployed. |

Core rules:

- `custom` is not source of truth.
- Long-term product code lives on one of the six domain branches or, when intentionally shared as a base, on `feat/retained-baseline`.
- Domain branches are independent siblings based on `feat/retained-baseline`; do not merge one domain branch into another just to make `custom` rebuild.
- `feat/retained-baseline` propagates into domain branches. Domain branches do not propagate downstream into each other.
- If a new live branch is introduced later, update `.sync-config`, `AGENTS.md`, and the workflow scripts together.
- Never modify submodules in this fork.

### Legacy Branches

These old fine-grained retained branches are preserved for history but are no longer live workflow branches: `feat/tailscale`, `feat/longitudinal-follow-gap`, `feat/longitudinal-lead-transition`, `feat/longitudinal-engage-bootstrap`, `feat/longitudinal-e2e-stop-approach`, `feat/longitudinal-cruise-coast`, `feat/longitudinal-fcw`, `feat/startup-comm-health`, `feat/longitudinal-launch`, `feat/longitudinal-scc-vision`, `feat/longitudinal-osm-planner`, `feat/speed-limit-auto-cruise`, `feat/torque-v2`, `feat/lane-change-path-shaping`, `feat/lateral-model-path-processing`, `feat/drive-lab`, `feat/longitudinal-decision-layer`, `feat/live-learning-expansion`, `feat/calculated-stats-accuracy`, and `feat/longitudinal-stack-selector`.

Do not start new work on legacy branches unless the user explicitly asks to inspect or recover history from them.

## Worktree Policy

- Keep a persistent worktree for `custom`. It is the workflow/admin worktree.
- Use `.worktrees/` at the repository root for agent-created implementation worktrees. The directory is intentionally project-local, hidden, and git-ignored.
- Name agent-created retained-branch worktrees after the branch suffix, for example `.worktrees/longitudinal-control` for `feat/longitudinal-control`.
- Run `scripts/sync-upstream.sh`, `scripts/propagate-retained.sh`, `scripts/rebuild-custom.sh`, and `scripts/deploy.sh` from the `custom` worktree.
- Do not keep `master` or retained branches open in persistent worktrees unless the user explicitly asks for that setup.
- `scripts/sync-upstream.sh` and `scripts/propagate-retained.sh` may create temporary worktrees while syncing or propagating.
- Script-created temporary worktrees must be removed before the script exits, including on failure paths.
- If a branch needed as a temporary target is already checked out in another worktree, stop and report that path instead of reusing or force-closing it.
- Only script-created temporary worktrees are auto-closed. Never auto-remove a user-opened worktree.
- When deleting a merged short-lived branch, remove any linked worktree for it and run `git worktree prune`.

### Fresh Worktree Test Setup

Fresh git worktrees do not share initialized submodules or generated Python extension modules with the main workspace. Before running Python tests in a new or recently rebased worktree, initialize/build local test prerequisites first:

```bash
git submodule update --init --recursive
uv run --extra testing --extra tools scons -j$(nproc) common/params_pyx.so msgq_repo/msgq/ipc_pyx.so cereal selfdrive/controls/lib/longitudinal_mpc_lib/c_generated_code/acados_ocp_solver_pyx.so
```

Then run tests through the same extras environment:

```bash
uv run --extra testing --extra tools python -m pytest <test paths>
```

If a fresh-worktree test fails with missing `openpilot.common.params_pyx`, `msgq.ipc_pyx`, `openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.c_generated_code`, `rednose_filter`, or `imgui`, treat it as worktree setup first rather than product-code failure.

## Domain Ownership

### Retained Baseline

`feat/retained-baseline` owns the squashed retained product snapshot and upstream-base compatibility fixes that all domain branches should inherit.

Use it for upstream conflict resolution that is genuinely shared by every domain branch. Do not put new feature behavior here unless the user explicitly asks to make it shared baseline behavior.

### Device Admin

`feat/device-admin` owns device runtime/admin support, including Tailscale backend/UI, process wiring, startup communication-health behavior, deploy/runtime health helpers, and device-only integration support.

Examples: `sunnypilot/system/tailscale/`, Tailscale params/UI, `manage_tailscaled`, startup comm issue grace, runtime triage support.

### Longitudinal Control

`feat/longitudinal-control` owns core longitudinal control behavior and longitudinal stack rollout boundaries.

Examples: follow-distance spacing, stopped-gap creep/release, lead lane-exit handoff, engage-time stop-threat bootstrap, no-lead e2e stop approach, plain-cruise coasting, FCW gating, generic launch shaping, decision-layer arbitration, longitudinal stack selector, stack telemetry, and custom longitudinal stack versions.

### Speed Map Control

`feat/speed-map-control` owns speed-limit, map, SCC, and OSM-derived behavior.

Examples: `sunnypilot/mapd/`, OSM/live map enrichment, speed-limit resolver and assist, speed-limit auto-cruise, SCC vision turn-speed, SCC map/advisory-speed behavior, map traffic-control priors.

### Lateral Control

`feat/lateral-control` owns controller-side lateral behavior.

Examples: torque controller versions, torque output shaping, guarded response assist, lane-change path shaping, lateral model path quality gating/sanitization, steering actuator feedback, lateral tracking and oscillation fixes.

### Control Learning Stats

`feat/control-learning-stats` owns live-learning expansion and calculated-stat correctness.

Examples: speed-aware torque learning, longitudinal mass/drag learning, brake/gas response learning, learned-cache invalidation, calibrated-roll observation, lateral-accel/curvature correctness, accurate lateral accel control paths, related params/UI/metadata/tests.

### Offline Drive Analysis

`feat/offline-drive-analysis` owns offline route analysis and regression tooling.

Examples: `tools/drive_lab/`, route-event explanation, manual-driving profiles, lateral event profiles, replay/test generation, longitudinal fuzzing, lateral disturbance simulation, and scenario extraction.

## Drive Lab Tooling

Use Drive Lab tools when investigating route bookmarks, longitudinal behavior, lateral behavior, fuzz/regression questions, or log-derived scenarios. Run these from a worktree that includes `feat/offline-drive-analysis` and use `uv run`.

Explain a bookmarked or timestamped route event:

```bash
uv run python -m openpilot.tools.drive_lab.explain_route_event ROUTE --nearest-bookmark
uv run python -m openpilot.tools.drive_lab.explain_route_event ROUTE --time 123.4 --before 30 --after 30
```

Build a route-derived profile to bias synthetic fuzzing toward observed driving ranges:

```bash
uv run python -m openpilot.tools.drive_lab.profile_route ROUTE --output /tmp/drive-lab-profile.json
```

Run seeded longitudinal fuzzing:

```bash
uv run python -m openpilot.tools.drive_lab.fuzz_longitudinal --seed 1 --mode comfort --cases 100
uv run python -m openpilot.tools.drive_lab.fuzz_longitudinal --seed 1 --mode comfort --profile /tmp/drive-lab-profile.json --cases 100
uv run python -m openpilot.tools.drive_lab.fuzz_longitudinal --seed 1 --mode adversarial --cases 100
```

Interpretation rules:

- `comfort` failures are candidates for real behavior fixes, especially if route-profile-guided.
- `emergency` failures need triage; they may represent valid hard-braking or cut-in edge cases.
- `adversarial` failures should usually improve the fuzzer or become stress tests before changing planner/controller behavior.
- Jerk-only launch/pullaway failures belong on `feat/longitudinal-control` if they remain plausible after route-profile-guided fuzzing.
- Route explanation/profile/fuzzer tool changes belong on `feat/offline-drive-analysis`; planner/controller behavior fixes belong on the owning domain branch.

## Custom-Only Files

These files intentionally live only on `custom` and are restored by `scripts/rebuild-custom.sh`:

- `.gitignore`
- `.sync-config`
- `AGENTS.md`
- `.agents/skills/`
- `scripts/lib/workflow.sh`
- `scripts/rebuild-custom.sh`
- `scripts/sync-upstream.sh`
- `scripts/propagate-retained.sh`
- `scripts/deploy.sh`

## Workflow Keywords

Agents should treat these phrases as direct workflow tasks:

- `sync upstream`: run `scripts/sync-upstream.sh`
- `sync retained`: run `scripts/sync-upstream.sh --sync-retained`
- `propagate retained`: run `scripts/propagate-retained.sh --from feat/retained-baseline`
- `rebuild custom`: run `scripts/rebuild-custom.sh`
- `deploy custom`: run `scripts/deploy.sh`
- `baseline reset`: use the optional device reset playbook below
- `deploy health check`: run the post-deploy checklist below

## Standard Workflow

### 1. Sync Upstream

Run this from the persistent `custom` worktree.

Default command:

```bash
scripts/sync-upstream.sh
```

Default behavior:

- fetch `upstream`
- fast-forward local `master` from `upstream/master` in a temporary worktree
- push `master` to `origin`

This command should not update retained branches or rebuild `custom` unless explicitly requested.

Optional retained-branch sync:

```bash
scripts/sync-upstream.sh --sync-retained
```

That merges the updated `master` into `feat/retained-baseline`, then merges `feat/retained-baseline` independently into each remaining domain branch listed in `MERGE_ORDER`. The script pushes `master` and retained branches to `origin` by default; use `--no-push` only when intentionally keeping updated branch tips local.

### 2. Propagate Retained Baseline

Run this from the persistent `custom` worktree after committing and pushing changes to `feat/retained-baseline` that every domain branch should inherit.

Default command:

```bash
scripts/propagate-retained.sh --from feat/retained-baseline
```

Behavior:

- find `feat/retained-baseline` in `MERGE_ORDER`
- merge `feat/retained-baseline` independently into each following domain branch
- push updated domain branches to `origin` by default

Use `--to BRANCH` to stop at a specific domain branch. Use `--no-push` only when intentionally keeping propagated branch tips local.

Do not propagate from a domain branch. Domain branches are siblings. If a domain branch conflicts with baseline, resolve the conflict on that domain branch, commit it there, push it, then rerun propagation from `feat/retained-baseline` if more domains remain.

### 3. Rebuild Custom

Run this from the persistent `custom` worktree.

Rebuild only after required retained-branch commits are local and pushed, and after baseline propagation has completed when `feat/retained-baseline` changed.

Preflight:

```bash
git branch --show-current
git status --short
git log -1 --oneline
```

Expected preflight:

- current branch is `custom`
- no tracked or untracked working-tree changes are present
- retained source branches contain the commits intended for deployment
- domain branches either already contain the current `feat/retained-baseline` or propagation was intentionally skipped for a local-only test rebuild

If `AGENTS.md`, `.sync-config`, or scripts need workflow updates, edit and commit them on `custom` before rebuilding. `scripts/rebuild-custom.sh` refuses dirty or untracked `custom` worktrees; do not stash product changes onto `custom` to satisfy this check.

Explicit command:

```bash
scripts/rebuild-custom.sh
```

Behavior:

- save `CUSTOM_FILES` from `custom`
- reset `custom` to `master`
- merge every retained branch in `MERGE_ORDER`
- restore `CUSTOM_FILES`
- create the metadata/custom-only commit if needed
- rebuild `selfdrive/controls/lib/longitudinal_mpc_lib/c_generated_code/acados_ocp_solver_pyx.so`

If the rebuild fails during a retained-branch merge, do not treat `custom` as the long-term fix location. Resolve the conflict on the owning domain branch or, if every domain needs the same fix, on `feat/retained-baseline` followed by propagation.

Post-rebuild verification:

```bash
git status --short
git log -5 --oneline
uv run --extra testing --extra tools python -m pytest <affected test paths>
```

Expected post-rebuild state:

- `custom` is clean
- the latest commits include retained-branch merges and any metadata/custom-only commit
- affected tests pass in the rebuilt integration tree

### 4. Deploy Custom

Run this from the persistent `custom` worktree.

Deploy only after `custom` has been rebuilt and verified. Do not deploy a retained feature branch or a dirty `custom` worktree. `scripts/deploy.sh` will warn if the worktree is dirty, but deployment uses only committed `custom` HEAD.

Preflight:

```bash
git branch --show-current
git status --short
git log -1 --oneline
scripts/deploy.sh --dry-run
```

Expected preflight:

- current branch is `custom`
- working tree is clean
- dry run shows `git push --force-with-lease origin custom`, device fetch/reset, submodule update, and reboot unless `--no-reboot` is specified

Explicit command:

```bash
scripts/deploy.sh
```

Behavior:

- force-push `custom` to `origin`
- SSH to the device
- fetch the repo remote on-device
- hard-reset the device checkout to `custom`
- update submodules
- reboot by default

Use deployment options only for intentional deviations:

```bash
scripts/deploy.sh --no-reboot
scripts/deploy.sh --host HOST --path PATH --remote REMOTE
```

After deploy, run the deploy health check below once SSH returns. If deploy health check fails, report the failed command and output before making further changes.

## Merge Conflict Policy

When conflicts happen during sync, propagation, or rebuild, resolve them on the branch that owns the long-term code, not on `custom`.

If `scripts/sync-upstream.sh --sync-retained` conflicts while merging `master` into `feat/retained-baseline`, resolve the conflict on `feat/retained-baseline`, then rerun retained sync or propagation.

If retained sync or propagation conflicts while merging `feat/retained-baseline` into a domain branch, resolve the conflict on that domain branch, commit it there, push it, and rerun the workflow.

If `scripts/rebuild-custom.sh` conflicts while merging a domain branch into `custom`, resolve the underlying compatibility issue on the owning domain branch or on `feat/retained-baseline` when the fix is shared by every domain.

`custom` is disposable and is rebuilt from source branches, so conflict-only fixes should not live there unless they are truly custom-only workflow metadata.

If a conflict involves files listed in `CUSTOM_FILES`, resolve those only as custom-only workflow or configuration changes. Do not move product code onto `custom` just to make the rebuild pass.

## Deploy Health Check

Run this after deploys when validating the branch.

Verify SSH is back:

```bash
ssh -o ConnectTimeout=10 "$DEPLOY_HOST" "uptime"
```

Verify the deployed commit:

```bash
ssh "$DEPLOY_HOST" "cd '$DEPLOY_PATH' && git log -1 --oneline"
```

Check manager and core processes:

```bash
ssh "$DEPLOY_HOST" "pgrep -af manager"
ssh "$DEPLOY_HOST" "pgrep -af 'selfdrive.ui.ui|pandad|loggerd|modeld|controlsd|selfdrived|locationd|paramsd|radard|manage_tailscaled'"
```

Check recent crashes or import errors:

```bash
ssh "$DEPLOY_HOST" "journalctl --since '5 min ago' 2>/dev/null | rg -i 'traceback|ImportError|ModuleNotFoundError|exception|crash'"
```

Run the retained-feature import sanity check:

```bash
ssh "$DEPLOY_HOST" "cd '$DEPLOY_PATH' && PYTHONPATH='$DEPLOY_PATH' /usr/local/venv/bin/python3 -c '
from openpilot.sunnypilot.system.tailscale.manage_tailscaled import TailscaleDaemon
from openpilot.system.ui.sunnypilot.widgets.tailscale_pairing_dialog import TailscalePairingDialog
print("tailscale-import-ok")
'"
```

Healthy result:

- deployed commit matches local `custom`
- manager and core processes are running
- `manage_tailscaled` is present
- no obvious crash/import loops in the recent journal
- Tailscale imports succeed

## Optional Device Reset Playbook

Use this only when intentionally returning the device to a clean baseline for troubleshooting.

Typical lateral-learning reset:

```bash
ssh "$DEPLOY_HOST" "rm -f \
  /data/params/d/LiveTorqueParameters \
  /data/params/d/LiveTorqueSpeedTable \
  /data/params/d/LiveParameters \
  /data/params/d/LiveParametersV2 \
  /data/params/d/EnforceTorqueControl \
  /data/params/d/TorqueControlTune \
  /data/params/d/LiveTorqueParamsToggle \
  /data/params/d/LiveTorqueParamsRelaxedToggle \
  /data/params/d/TorqueParamsOverrideEnabled \
  /data/params/d/TorqueParamsOverrideFriction \
  /data/params/d/TorqueParamsOverrideLatAccelFactor"
```

Notes:

- This is a recovery/troubleshooting tool, not part of the normal deploy path.
- Leave Tailscale params alone unless the problem is specifically with Tailscale state.

## Agent Guidelines

When making code changes:

1. Do not commit directly to `master`.
2. Do not treat `custom` as the source branch for long-term behavior changes.
3. Put shared baseline compatibility fixes on `feat/retained-baseline` only when every domain branch should inherit them.
4. Put Tailscale, startup comm health, and device runtime/admin support on `feat/device-admin`.
5. Put core longitudinal behavior, longitudinal decision-layer behavior, and longitudinal stack selector changes on `feat/longitudinal-control`.
6. Put speed-limit, SCC vision/map, OSM/mapd, and speed-map behavior on `feat/speed-map-control`.
7. Put torque-controller, lane-change path shaping, lateral model path processing, and lateral controller behavior on `feat/lateral-control`.
8. Put live-learning expansion, calculated-stat correctness, learned-cache invalidation, and accurate lateral-accel behavior on `feat/control-learning-stats`.
9. Put Drive Lab, route-event explanation, replay/test generation, and scenario fuzzing changes on `feat/offline-drive-analysis`.
10. When a route/log/timestamp/bookmark is available for a longitudinal or lateral issue, use Drive Lab explanation/profile tools before guessing from code alone.
11. When fuzzing finds a failure, classify it by Drive Lab mode before deciding whether to fix the fuzzer or planner/controller behavior.
12. Use `uv run` for Python commands, tests, and scripts in this repository.
13. Keep `.sync-config`, scripts, skills, and `AGENTS.md` aligned whenever the workflow changes.

When syncing or deploying:

1. Default upstream sync should stop at `master` unless the user explicitly asks for retained-branch sync or a `custom` rebuild.
2. Push retained branches explicitly to `origin` as part of the standard retained-branch workflow. Do not rely on local-only retained branch commits when rebuilding or deploying `custom` unless the user explicitly wants a local-only test build.
3. Before rebuilding `custom`, propagate `feat/retained-baseline` changes if the baseline changed and domain branches do not already contain them.
4. Rebuild `custom` explicitly before deploys.
5. Treat `custom` as disposable and safe to force-push with `--force-with-lease`.
6. Prefer the documented scripts over ad hoc manual Git flows unless debugging the scripts themselves.
7. Treat the `custom` worktree as the admin worktree and let the scripts create and remove temporary worktrees.
8. If a temporary target branch is already open in another worktree, stop and ask before reusing or closing it.

Skills provide specialized instructions and workflows for specific tasks.
