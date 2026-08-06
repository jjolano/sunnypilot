# AGENTS.md

Instructions and context for AI agents working in this repository. `CLAUDE.md` is a
symlink to this file.

## Project Overview

A custom fork of [sunnypilot](https://github.com/sunnypilot/sunnypilot) (itself a fork of
[commaai/openpilot](https://github.com/commaai/openpilot)), restarted from upstream on
2026-06-12. The restart reimplements the proven behavior of the previous fork — the
torque v2.1 lateral controller and the custom-2.0 longitudinal policy — as cleaner
rewrites, guided by
[docs/plans/2026-06-12-fork-restart-reimplementation.md](docs/plans/2026-06-12-fork-restart-reimplementation.md).

```text
commaai/openpilot  ->  sunnypilot/sunnypilot  ->  jjolano/sunnypilot (this repo)
                          (upstream remote)          (origin remote)
```

## Repo Model

Deliberately simple — the previous fork's branch machinery is retired.

- `master` is the fork: linear commits land directly on `master`, and deploys run from it
  (`.deploy-config` `DEPLOY_BRANCH=master`). The `restart` branch (the reimplementation
  runway) is retired now that it has fast-forwarded onto `master`.
- Upstream updates are deliberate merges of `upstream/master`, not scheduled syncs.
- The old `custom` branch (frozen at `8a97000e19`) is the executable reference
  implementation of the previous fork. Read from it (`git show custom:<path>`); never
  develop on it. Old feature branches (`feat/*`) are historical only.
- One implementation per axis: features are plain params, never selectable "stacks".
- All custom behavior goes in new files; diffs to upstream files stay hook-sized and are
  listed in `docs/touch-points.md`. Never modify submodules.

## Documentation Map

- `docs/plans/` — the restart plan. Read it before starting any porting or rewrite work.
- `docs/legacy/` — reference material imported from the old fork: domain language
  (`CONTEXT-*.md`), kept ADRs, behavior concepts, Phase 5 backlog specs, and
  `tuned-constants.yaml` (369 tuned constants with old-fork source locations — the source
  of truth for reimplementation values).
- `docs/adr/` — new decisions get new ADRs here; legacy ADRs are not edited.

## Repo-Local Skills

Repo-local skills live under `.agents/skills/`. When a user request matches a
description, read that skill's `SKILL.md` before acting.

- `deploy-workflow`: commit/test/deploy/health-check workflow. Use when the user asks to
  deploy, rebuild, "commit rebuild deploy", push to the device, or validate a deployed
  branch.
- `device-route-log-analysis`: access route logs from the configured deployment target and
  analyze recent drives with drive_lab. Use when the user asks to inspect device route logs
  or analyze lateral/longitudinal control from logs.
- `route-drive-diagnosis`: deep rlog diagnosis of drive complaints — extract full-route
  signals to numpy, detect episodes (stop gaps, launches, interventions, cut-outs, wander,
  lane position), and correlate with the longitudinalDebug trace to name the owning
  constant/module. Use when the user reports drive behavior complaints to root-cause.
- `device-comm-diagnostics`: read-only diagnosis for on-device "Communication Issue Between
  Processes", "Low Communication Rate Between Processes", process-not-running, IPC/msgq, and
  manager/process health alerts.
- `param-settings-ui`: checklist for adding Params keys with matching Settings UI controls,
  generated schema JSON, encoding tests, and runtime fail-closed behavior. Use whenever adding
  or changing a user/tester-facing param, feature flag, or shadow/apply mode.

## Workflow Keywords

- `commit`: commit working-tree changes on the deploy branch (see `.deploy-config`).
- `rebuild`: legacy verb from the retired integration-branch model — now a no-op;
  just confirm the deploy branch has the intended commits.
- `deploy`: run `scripts/deploy.sh --dry-run`, inspect, then `scripts/deploy.sh`.
- `deploy health check`: run the checklist below.

## Testing

Build prerequisites once per fresh checkout/worktree:

```bash
git submodule update --init --recursive
uv run --extra testing --extra tools scons -j$(nproc) openpilot/common/params_pyx.so msgq_repo/msgq/ipc_pyx.so openpilot/cereal openpilot/selfdrive/controls/lib/longitudinal_mpc_lib/c_generated_code/acados_ocp_solver_pyx.so
```

Run tests through the same extras environment:

```bash
uv run --extra testing --extra tools python -m pytest <test paths>
```

Missing `params_pyx`, `ipc_pyx`, `acados_ocp_solver_pyx`, `rednose_filter`, or `imgui`
in a fresh worktree is a setup problem, not a product-code failure.

## Drive Lab

`tools/drive_lab/` is the offline route-analysis toolkit and the validation gate for
every port/rewrite phase (see the restart plan's validation strategy). Typical entry
points, run with `uv run`:

```bash
uv run python -m openpilot.tools.drive_lab.explain_route_event ROUTE --nearest-bookmark
uv run python -m openpilot.tools.drive_lab.profile_route ROUTE --output /tmp/profile.json
uv run python -m openpilot.tools.drive_lab.fuzz_longitudinal --seed 1 --mode comfort --cases 100
```

Analyses that decode the old fork's custom log structs must run from a `custom`-branch
checkout, which has the matching cereal schema.

Longitudinal fuzz presets (`fuzz_longitudinal --preset`): `fuzz` (seeded random),
`udacity-acc` (15 fixed ACC cases), `openpilot-acc` (upstream pytest maneuvers),
`ncap-acc` (Euro NCAP ACC grid), `commonroad-acc` (bundled ZAM fixtures).
Route replay: `fuzz_longitudinal_route_replay --route ROUTE`. OpenACC CSV profiling:
`openacc_segments.py` → use with `--profile` on the fuzzer.

## Deploy Health Check

Run after deploys when validating the branch. Config values come from `.deploy-config`.

```bash
ssh -o ConnectTimeout=10 "$DEPLOY_HOST" "uptime"
ssh "$DEPLOY_HOST" "cd '$DEPLOY_PATH' && git log -1 --oneline"   # matches local HEAD
ssh "$DEPLOY_HOST" "pgrep -af manager"
ssh "$DEPLOY_HOST" "pgrep -af 'selfdrive.ui.ui|pandad|loggerd|modeld|controlsd|selfdrived|locationd|paramsd|radard'"
ssh "$DEPLOY_HOST" "journalctl --since '5 min ago' 2>/dev/null | rg -i 'traceback|ImportError|ModuleNotFoundError|exception|crash'"
```

Healthy result: deployed commit matches local HEAD, manager and core processes running,
no crash/import loops in the recent journal. If the device is offroad, onroad-only
processes (`controlsd`, `modeld`, `radard`) are expected to be absent.
