---
name: rebuild-deploy-workflow
description: Runs the sunnypilot retained-branch propagation, custom rebuild, deploy, and post-deploy health-check workflow safely. Use when the user asks to rebuild custom, deploy custom, propagate retained branches before deployment, or validate a deployed branch.
---

# Rebuild Deploy Workflow

## Quick Start

Run this skill from the persistent `custom` worktree.

```bash
bash .agents/skills/rebuild-deploy-workflow/scripts/rebuild_deploy_preflight.sh
scripts/rebuild-custom.sh
uv run --extra testing --extra tools python -m pytest <affected test paths>
scripts/deploy.sh --dry-run
scripts/deploy.sh
```

After deploy, run the health checks from `AGENTS.md` under `Deploy Health Check`.

## Guardrails

Do not commit directly to `master`.
Do not put long-term product changes on `custom`.
Do not rebuild `custom` to discover retained-branch conflicts.
Do not deploy a dirty `custom` worktree.
Do not force-remove user-opened worktrees.
Do not propagate from a domain branch; only `feat/retained-baseline` propagates into domain branches.
Do not skip baseline propagation when `feat/retained-baseline` changed unless the user explicitly requested a local-only test rebuild.

## Rebuild Workflow

1. Confirm the current worktree is `custom` and clean.
2. Identify whether `feat/retained-baseline` or a domain branch has commits intended for deployment.
3. If `feat/retained-baseline` changed, run `scripts/propagate-retained.sh --from feat/retained-baseline` before rebuilding.
4. If propagation reports a checked-out domain worktree, remove only clean agent-created blockers and rerun the same command.
5. If propagation hits a conflict, resolve and commit it on the domain branch that owns the long-term compatibility fix.
6. Run `scripts/rebuild-custom.sh` only after retained source branches are committed, pushed, and any baseline propagation is complete.
7. Run affected tests in the rebuilt integration tree with `uv run --extra testing --extra tools python -m pytest <paths>`.

## Deploy Workflow

1. Confirm `git branch --show-current` returns `custom`.
2. Confirm `git status --short` has no output.
3. Confirm `git log -1 --oneline` is the rebuilt commit intended for deploy.
4. Run `scripts/deploy.sh --dry-run` and inspect the push, device reset, submodule update, and reboot commands.
5. Run `scripts/deploy.sh` only after the dry run is expected.
6. Wait for SSH to return with `ssh -o ConnectTimeout=10 "$DEPLOY_HOST" "uptime"`.
7. Verify deployed commit, manager/core processes, recent crash/import logs, and retained-feature imports.
8. If the device is offroad, expect onroad-only processes such as `controlsd`, `modeld`, and `radard` to be absent.

## Failure Handling

If `scripts/rebuild-custom.sh` fails during a retained-branch merge, stop and fix the underlying conflict on the owning domain branch or on `feat/retained-baseline` if every domain needs the fix, then propagate baseline if needed and rebuild again.
If deploy fails before the push, keep local state unchanged and report the failed command.
If deploy fails after the push but before device reset completes, check the device checkout and rerun `scripts/deploy.sh` only after understanding the partial state.
If health checks show tracebacks, import errors, or crash loops, report the exact command and output before making new changes.

## Helper Script

Use `.agents/skills/rebuild-deploy-workflow/scripts/rebuild_deploy_preflight.sh` for deterministic local checks before rebuild or deploy. The helper checks branch, cleanliness, required workflow scripts, and shows the current `custom` commit without modifying the worktree.

## Advanced Features

See [REFERENCE.md](REFERENCE.md) for propagation decision rules, conflict handling, health-check interpretation, and partial deploy recovery.
