# Sync Upstream Reference

## Command Modes

| User request | Command |
|---|---|
| `sync upstream` | `scripts/sync-upstream.sh` |
| `sync retained` | `scripts/sync-upstream.sh --sync-retained` |
| sync upstream without pushing | `scripts/sync-upstream.sh --no-push` |
| sync retained without pushing | `scripts/sync-upstream.sh --sync-retained --no-push` |
| sync retained and rebuild custom | `scripts/sync-upstream.sh --rebuild-custom` |
| preview actions | add `--dry-run` |

Default behavior updates only local `master` from `upstream/master` in a temporary worktree and pushes `master` to `origin`.

`--sync-retained` merges updated `master` into `RETAINED_BASE_BRANCH`, then merges `RETAINED_BASE_BRANCH` independently into each remaining domain branch in `MERGE_ORDER`. It pushes retained branch tips to `origin` unless `--no-push` is set.

`--rebuild-custom` implies `--sync-retained` and should only be used when the user asked for a custom rebuild after retained sync.

Do not infer `--sync-retained` from a plain upstream-sync request. The retained-branch sync touches every retained source branch, can create conflicts, and pushes branch tips by default.

Do not infer `--rebuild-custom` from `--sync-retained`. Rebuilding `custom` is a deployment-prep action and should remain explicit.

## Preflight Checklist

Run from the `custom` worktree:

```bash
git branch --show-current
git status --short
bash .agents/skills/sync-upstream-workflow/scripts/sync_upstream_preflight.sh
```

Expected state:

- Current branch is `custom`.
- Working tree has no tracked, staged, or untracked changes.
- `upstream` and `origin` remotes exist.
- `master` exists locally.
- Retained branches exist locally when using `--sync-retained` or `--rebuild-custom`.
- Target branches are not checked out in another worktree.

## Worktree Blockers

The sync script creates temporary worktrees for `master` and retained target branches. A target branch already checked out elsewhere blocks sync.

When blocked:

```bash
git worktree list --porcelain
```

If the blocker is an agent-created worktree under `.worktrees/` and `git status --short` inside it is empty, remove it and prune:

```bash
git worktree remove .worktrees/<branch-suffix>
git worktree prune
```

If the blocker is dirty, user-opened, outside `.worktrees/`, or ambiguous, stop and ask before removing it.

The `custom` worktree itself is allowed and required because it is the admin worktree. Any `master` or retained-branch worktree is a target blocker because the sync script creates temporary target worktrees.

If a temporary worktree path from a previous failed sync still exists, inspect it before cleanup. If it contains unresolved conflicts, continue the conflict recovery path instead of deleting it.

## Retained Sync Conflicts

When `--sync-retained` hits a conflict, the sync script preserves the conflicted temporary worktree and prints its path.

Recovery:

```bash
git -C <conflict-worktree> status --short
# resolve conflicts in the retained branch worktree
uv run --extra testing --extra tools python -m pytest <owning test paths>
git -C <conflict-worktree> add <resolved files>
git -C <conflict-worktree> commit
git -C <conflict-worktree> push origin <retained-branch>
scripts/sync-upstream.sh --sync-retained
```

Resolve conflicts according to retained-branch ownership. Do not commit retained sync conflict fixes to `custom`.

After resolving a retained sync conflict, rerun the full retained sync command. The script will skip branches that already contain the updated source and continue through remaining retained branches.

If the same conflict returns after a resolved commit, check whether the resolution was committed on the retained branch printed by the script and pushed to `origin`.

## Rebuild Custom Mode

`scripts/sync-upstream.sh --rebuild-custom` performs three operations:

1. Fast-forwards `master` from `upstream/master`.
2. Syncs `feat/retained-baseline` with updated `master`, then syncs each domain branch with `feat/retained-baseline`.
3. Rebuilds `custom` from the synced retained branches.

Use this mode only when the user explicitly asks for retained sync plus custom rebuild. If the user also asks to deploy, finish this mode, then switch to `rebuild-deploy-workflow` for deploy dry-run, deploy, and health checks.

If rebuild fails after retained sync succeeds, do not fix conflicts directly on `custom`. Treat it as retained-branch compatibility work and follow the rebuild/deploy workflow reference.

## Push Failure Handling

The sync script pushes `master` by default and pushes retained branches when `--sync-retained` is used. A push failure can leave local branch tips updated while `origin` is stale.

When a push fails:

```bash
git status --short
git log -1 --oneline master
git ls-remote origin master
```

For retained sync, inspect the affected retained branch tips named in script output. Retry the same sync command only after confirming the failure was transient. If remote rejected because it advanced independently, fetch and inspect before retrying.

Do not force-push `master` or retained branches as part of sync. Only `custom` deploy/rebuild flows use force-with-lease.

## No-Push Mode

Use `--no-push` only when the user explicitly wants a local-only sync or dry integration test. Report that local branch tips differ from `origin` and that rebuild/deploy should not rely on them unless the user accepts a local-only workflow.

After a no-push retained sync, a later normal sync may say branches already contain updated `master` locally and then push them. Make that behavior clear to the user.

## Post-Sync Checks

For default upstream sync:

```bash
git log -1 --oneline master
git status --short
```

For retained sync, additionally inspect branch tips or script output to confirm every retained branch in `MERGE_ORDER` was processed.

Useful retained sync spot check:

```bash
git branch --contains master --format='%(refname:short)'
```

This is a broad check only. The authoritative result is the sync script output and absence of preserved conflict worktrees.

If the user intends to deploy after retained sync, switch to `rebuild-deploy-workflow`: propagate any retained-branch-specific changes if needed, rebuild `custom`, run affected tests, deploy dry-run, deploy, and health check.

## Failure Interpretation

Fast-forward failure on `master` means local `master` diverged from `upstream/master`; stop and report it.

Push failure after local updates means branch tips may be updated locally but not on `origin`; inspect script output and retry only after understanding remote state.

A conflict during retained sync means upstream changed code overlapping `feat/retained-baseline`, or baseline changed code overlapping a domain branch. Resolve on the branch printed by the script and continue sync from the normal command.

A rebuild failure during `--rebuild-custom` belongs to the rebuild/deploy workflow, not ad hoc fixes on `custom`.

Untracked files in `custom` are a hard stop because sync scripts must not hide or overwrite workflow metadata or user work. Commit intentional custom-only files first, or ask the user how to handle unrelated files.
