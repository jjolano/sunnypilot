# Sync Upstream Examples

## Plain Upstream Sync

User says: `sync upstream`

Run:

```bash
bash .agents/skills/sync-upstream-workflow/scripts/sync_upstream_preflight.sh
scripts/sync-upstream.sh
git log -1 --oneline master
git status --short
```

Expected behavior:

- Fetches `upstream`.
- Fast-forwards local `master` from `upstream/master` in a temporary worktree.
- Pushes `master` to `origin`.
- Does not touch retained branches.
- Does not rebuild `custom`.

## Retained Branch Sync

User says: `sync retained branches with upstream`

Run:

```bash
bash .agents/skills/sync-upstream-workflow/scripts/sync_upstream_preflight.sh --sync-retained
scripts/sync-upstream.sh --sync-retained
git status --short
```

Expected behavior:

- Updates `master` first.
- Merges updated `master` into `feat/retained-baseline`.
- Merges `feat/retained-baseline` independently into each remaining domain branch in `MERGE_ORDER`.
- Pushes `master` and retained branches to `origin`.
- Leaves successful temporary worktrees removed.
- Preserves any conflicted branch worktree and prints its path.

## Sync Retained And Rebuild

User says: `sync retained and rebuild custom`

Run:

```bash
bash .agents/skills/sync-upstream-workflow/scripts/sync_upstream_preflight.sh --rebuild-custom
scripts/sync-upstream.sh --rebuild-custom
git status --short
git log -5 --oneline
```

Expected behavior:

- Updates `master`.
- Syncs retained branches.
- Rebuilds `custom` after retained sync.
- Pushes `master`, retained branches, and rebuilt `custom`.
- Does not deploy the device.

## Dry Run Before Risky Sync

User says: `show me what sync retained would do`

Run:

```bash
bash .agents/skills/sync-upstream-workflow/scripts/sync_upstream_preflight.sh --sync-retained
scripts/sync-upstream.sh --sync-retained --dry-run
```

Expected behavior:

- Prints planned fetch, temporary worktree, merge, cleanup, and push commands.
- Does not modify branch tips.

## Conflict During Retained Sync

Script prints a conflict worktree path for `feat/example-domain`.

Run:

```bash
git -C <conflict-worktree> status --short
# resolve files according to feat/example-domain ownership
uv run --extra testing --extra tools python -m pytest <owning tests>
git -C <conflict-worktree> add <resolved files>
git -C <conflict-worktree> commit
git -C <conflict-worktree> push origin feat/example-domain
scripts/sync-upstream.sh --sync-retained
```

Expected behavior:

- The compatibility fix lives on the branch printed by the script.
- The rerun continues retained sync.
- No conflict-only product fix is committed to `custom`.
