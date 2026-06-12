---
name: sync-upstream-workflow
description: Runs the sunnypilot upstream synchronization workflow safely, including default master fast-forward sync and optional retained-branch sync/rebuild flows. Use when the user asks to sync upstream, sync retained branches, update from upstream/master, or prepare retained branches after upstream changes.
---

# Sync Upstream Workflow

## Quick Start

Run this skill from the persistent `custom` worktree.

Default upstream sync only:

```bash
bash .agents/skills/sync-upstream-workflow/scripts/sync_upstream_preflight.sh
scripts/sync-upstream.sh
```

Retained-branch sync is explicit:

```bash
bash .agents/skills/sync-upstream-workflow/scripts/sync_upstream_preflight.sh --sync-retained
scripts/sync-upstream.sh --sync-retained
```

## Guardrails

Default `sync upstream` stops at `master`; do not sync retained branches, rebuild `custom`, or deploy unless the user explicitly asks.
Run from `custom`, not `master` or a retained branch worktree.
Require a clean `custom` worktree before syncing.
Do not reuse or force-close user-opened worktrees.
If a target branch is already checked out in another worktree, stop and report that path.
If retained sync conflicts, resolve and commit the merge on `feat/retained-baseline` or on the target domain branch where the conflict happened.

## Workflows

1. Read `.sync-config` to confirm `LOCAL_BRANCH`, `INTEGRATION_BRANCH`, `RETAINED_BASE_BRANCH`, remotes, and `MERGE_ORDER`.
2. Run the preflight helper with the same mode you intend to run.
3. For plain upstream sync, run `scripts/sync-upstream.sh`.
4. For retained sync, run `scripts/sync-upstream.sh --sync-retained`.
5. Use `--no-push` only when intentionally keeping updated branch tips local.
6. Use `--rebuild-custom` only when the user explicitly asks to rebuild after retained sync.
7. Report updated branches and any preserved conflict worktree path.

## Failure Handling

If fast-forwarding `master` fails, stop; `master` must remain a pristine fast-forward mirror of `upstream/master`.
If `feat/retained-baseline` conflicts while merging updated `master`, use the conflict worktree printed by the script, resolve there, commit on `feat/retained-baseline`, push it, then rerun `scripts/sync-upstream.sh --sync-retained`.
If a domain branch conflicts while merging `feat/retained-baseline`, resolve and commit on that domain branch, push it, then rerun `scripts/sync-upstream.sh --sync-retained`.
If a branch is already checked out elsewhere, do not remove it unless it is clearly clean and agent-created; otherwise ask the user.
If `--rebuild-custom` fails, switch to the rebuild/deploy workflow skill and resolve the underlying retained-branch issue before retrying.

## Helper Script

Use `.agents/skills/sync-upstream-workflow/scripts/sync_upstream_preflight.sh` for deterministic local checks before sync. The helper validates branch, cleanliness, config, remotes, required local branches, and target worktree blockers without modifying git state.

## Advanced Features

See [REFERENCE.md](REFERENCE.md) for command modes, conflict recovery, worktree blocker handling, push failure handling, and post-sync checks. See [EXAMPLES.md](EXAMPLES.md) for common request-to-command mappings.
