# Retained Branch Propagation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a documented, explicit helper workflow for cascading retained-branch changes through downstream branches in `.sync-config` `MERGE_ORDER`.

**Architecture:** Add one custom-only shell helper, `scripts/propagate-retained.sh`, that mirrors the safety style of `scripts/sync-upstream.sh`: clean integration worktree, config-driven branch order, temporary worktrees, conflict preservation, and default pushes. Update `.sync-config` and `AGENTS.md` so the helper survives custom rebuilds and agents know when to use it.

**Tech Stack:** Bash, Git worktrees, existing `.sync-config`, existing custom-only workflow scripts, Markdown docs.

---

## File Structure

- Create `scripts/propagate-retained.sh`: CLI helper for downstream retained branch propagation.
- Modify `.sync-config`: add `scripts/propagate-retained.sh` to `CUSTOM_FILES`.
- Modify `AGENTS.md`: add the helper to workflow keywords, standard workflow, custom-only files, and conflict policy.
- Use `docs/superpowers/specs/2026-05-02-retained-branch-propagation-design.md` as the design source.

## Task 1: Add Helper Script Failing Checks

**Files:**
- Verify missing behavior before creating: `scripts/propagate-retained.sh`

- [ ] **Step 1: Run missing helper help check**

Run:

```bash
scripts/propagate-retained.sh --help
```

Expected: command fails because `scripts/propagate-retained.sh` does not exist.

- [ ] **Step 2: Run missing helper validation check**

Run:

```bash
scripts/propagate-retained.sh --from feat/longitudinal-osm-planner --to feat/speed-limit-auto-cruise --dry-run
```

Expected: command fails because `scripts/propagate-retained.sh` does not exist.

## Task 2: Create `scripts/propagate-retained.sh`

**Files:**
- Create: `scripts/propagate-retained.sh`

- [ ] **Step 1: Add executable helper script**

Create `scripts/propagate-retained.sh` with this content:

```bash
#!/usr/bin/env bash

set -euo pipefail

CONFIG_FILE=".sync-config"
DRY_RUN=false
NO_PUSH=false
KEEP_TEMP=false
FROM_BRANCH=""
TO_BRANCH=""
TEMP_WORKTREES=()
REMOVED_TEMP_WORKTREE=false

usage() {
  cat <<'EOF'
Usage: scripts/propagate-retained.sh --from BRANCH [OPTIONS]

Merge a retained branch through each downstream retained branch in MERGE_ORDER.
Run this from the persistent integration worktree. Successful temporary worktrees
are removed before exit unless --keep-temp is used. Conflicted temporary
worktrees are preserved for manual conflict resolution.

Options:
  --from <branch>  Source retained branch to propagate downstream (required)
  --to <branch>    Final downstream branch to update (default: last MERGE_ORDER branch)
  --config <path>  Config file path relative to repo root (default: .sync-config)
  --dry-run        Print the actions without modifying git state
  --no-push        Skip pushing updated downstream branches to origin
  --keep-temp      Preserve successful temporary worktrees for inspection
  --help           Show this help message
EOF
}

info() {
  printf '[INFO] %s\n' "$*"
}

warn() {
  printf '[WARN] %s\n' "$*" >&2
}

fail() {
  printf '[FAIL] %s\n' "$*" >&2
  exit 1
}

print_cmd() {
  local arg
  printf '+'
  for arg in "$@"; do
    printf ' %q' "$arg"
  done
  printf '\n'
}
```

- [ ] **Step 2: Append helper implementation after `print_cmd`**

Append this content to `scripts/propagate-retained.sh`:

```bash
run() {
  if $DRY_RUN; then
    print_cmd "$@"
  else
    "$@"
  fi
}

register_temp_worktree() {
  TEMP_WORKTREES+=("$1")
}

forget_temp_worktree() {
  local path="$1"
  local remaining=()
  local existing

  for existing in "${TEMP_WORKTREES[@]}"; do
    if [[ "$existing" != "$path" ]]; then
      remaining+=("$existing")
    fi
  done

  TEMP_WORKTREES=("${remaining[@]}")
}

find_branch_worktree() {
  local branch="$1"
  local line worktree_path="" worktree_branch=""

  while IFS= read -r line; do
    if [[ -z "$line" ]]; then
      if [[ "$worktree_branch" == "refs/heads/$branch" ]]; then
        printf '%s\n' "$worktree_path"
        return 0
      fi
      worktree_path=""
      worktree_branch=""
      continue
    fi

    case "$line" in
      worktree\ *)
        worktree_path="${line#worktree }"
        ;;
      branch\ *)
        worktree_branch="${line#branch }"
        ;;
    esac
  done < <(git worktree list --porcelain)

  if [[ "$worktree_branch" == "refs/heads/$branch" ]]; then
    printf '%s\n' "$worktree_path"
    return 0
  fi

  return 1
}

ensure_branch_available_for_temp_worktree() {
  local branch="$1"
  local existing_worktree

  existing_worktree="$(find_branch_worktree "$branch" || true)"
  if [[ -n "$existing_worktree" ]]; then
    fail "Branch '$branch' is already checked out in worktree: $existing_worktree"
  fi
}

create_temp_worktree() {
  local branch="$1"
  local out_var="$2"
  local safe_branch path

  ensure_branch_available_for_temp_worktree "$branch"

  safe_branch="${branch//\//-}"
  if $DRY_RUN; then
    path="${TMPDIR:-/tmp}/sunnypilot-propagate-${safe_branch}.DRYRUN"
    print_cmd git worktree add "$path" "$branch"
  else
    path="$(mktemp -d "${TMPDIR:-/tmp}/sunnypilot-propagate-${safe_branch}.XXXXXX")"
    if ! git worktree add "$path" "$branch" >/dev/null 2>&1; then
      rmdir "$path" 2>/dev/null || true
      fail "Failed to create temporary worktree for $branch"
    fi
  fi

  register_temp_worktree "$path"
  printf -v "$out_var" '%s' "$path"
}

remove_temp_worktree() {
  local path="$1"

  [[ -n "$path" ]] || return

  if $DRY_RUN; then
    print_cmd git worktree remove --force "$path"
  else
    git worktree remove --force "$path" >/dev/null 2>&1 || rm -rf "$path"
  fi

  REMOVED_TEMP_WORKTREE=true
  forget_temp_worktree "$path"
}

cleanup() {
  local path

  if $KEEP_TEMP; then
    return
  fi

  for path in "${TEMP_WORKTREES[@]}"; do
    if $DRY_RUN; then
      print_cmd git worktree remove --force "$path"
    else
      git worktree remove --force "$path" >/dev/null 2>&1 || rm -rf "$path"
    fi
    REMOVED_TEMP_WORKTREE=true
  done

  if $REMOVED_TEMP_WORKTREE; then
    if $DRY_RUN; then
      print_cmd git worktree prune
    else
      git worktree prune >/dev/null 2>&1 || true
    fi
  fi
}

trap cleanup EXIT

find_merge_order_index() {
  local branch="$1"
  local idx

  for idx in "${!MERGE_ORDER[@]}"; do
    if [[ "${MERGE_ORDER[$idx]}" == "$branch" ]]; then
      printf '%s\n' "$idx"
      return 0
    fi
  done

  return 1
}

push_updated_branches() {
  if $DRY_RUN; then
    print_cmd env GIT_LFS_SKIP_PUSH=1 git push origin "${UPDATED_BRANCHES[@]}"
  else
    GIT_LFS_SKIP_PUSH=1 git push origin "${UPDATED_BRANCHES[@]}"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --from)
      [[ $# -ge 2 ]] || fail "Missing value for --from"
      FROM_BRANCH="$2"
      shift 2
      ;;
    --to)
      [[ $# -ge 2 ]] || fail "Missing value for --to"
      TO_BRANCH="$2"
      shift 2
      ;;
    --config)
      [[ $# -ge 2 ]] || fail "Missing value for --config"
      CONFIG_FILE="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --no-push)
      NO_PUSH=true
      shift
      ;;
    --keep-temp)
      KEEP_TEMP=true
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      fail "Unknown option: $1"
      ;;
  esac
done

[[ -n "$FROM_BRANCH" ]] || fail "Missing required --from BRANCH"

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
[[ -n "$REPO_ROOT" ]] || fail "Not inside a git repository"
cd "$REPO_ROOT"

if [[ "$CONFIG_FILE" == /* ]]; then
  CONFIG_PATH="$CONFIG_FILE"
else
  CONFIG_PATH="$REPO_ROOT/$CONFIG_FILE"
fi
[[ -f "$CONFIG_PATH" ]] || fail "Config file not found: $CONFIG_FILE"
# shellcheck source=/dev/null
source "$CONFIG_PATH"

CURRENT_BRANCH="$(git branch --show-current)"
[[ "$CURRENT_BRANCH" == "$INTEGRATION_BRANCH" ]] || fail "Run this script from the $INTEGRATION_BRANCH worktree (current branch: ${CURRENT_BRANCH:-detached HEAD})"

[[ ${#MERGE_ORDER[@]} -gt 0 ]] || fail "MERGE_ORDER is empty; no retained branches to propagate"

FROM_INDEX="$(find_merge_order_index "$FROM_BRANCH" || true)"
[[ -n "$FROM_INDEX" ]] || fail "--from branch is not in MERGE_ORDER: $FROM_BRANCH"

if [[ -z "$TO_BRANCH" ]]; then
  TO_INDEX=$((${#MERGE_ORDER[@]} - 1))
  TO_BRANCH="${MERGE_ORDER[$TO_INDEX]}"
else
  TO_INDEX="$(find_merge_order_index "$TO_BRANCH" || true)"
  [[ -n "$TO_INDEX" ]] || fail "--to branch is not in MERGE_ORDER: $TO_BRANCH"
fi

if (( TO_INDEX < FROM_INDEX )); then
  fail "--to branch must be the same as or downstream from --from in MERGE_ORDER"
fi

for ((idx = FROM_INDEX; idx <= TO_INDEX; idx++)); do
  branch="${MERGE_ORDER[$idx]}"
  git show-ref --verify --quiet "refs/heads/$branch" || fail "Missing local branch: $branch"
done

if ! git diff --quiet || ! git diff --cached --quiet; then
  fail "Working tree is not clean. Commit or stash changes before propagating retained branches"
fi

if [[ -n "$(git ls-files --others --exclude-standard)" ]]; then
  fail "Untracked files are present. Commit, stash, or clean them before propagating retained branches"
fi

git remote | grep -qx "origin" || fail "Remote 'origin' not found"

for ((idx = FROM_INDEX; idx <= TO_INDEX; idx++)); do
  ensure_branch_available_for_temp_worktree "${MERGE_ORDER[$idx]}"
done

if (( FROM_INDEX == TO_INDEX )); then
  info "No downstream branches to update from $FROM_BRANCH to $TO_BRANCH"
  exit 0
fi

UPDATED_BRANCHES=()
source_branch="$FROM_BRANCH"
dry_run_stack_changed=false

for ((idx = FROM_INDEX + 1; idx <= TO_INDEX; idx++)); do
  target_branch="${MERGE_ORDER[$idx]}"
  target_worktree=""

  info "Propagating $source_branch into $target_branch"
  if ! $dry_run_stack_changed && git merge-base --is-ancestor "$source_branch" "$target_branch"; then
    info "$target_branch already contains $source_branch"
    source_branch="$target_branch"
    continue
  fi

  create_temp_worktree "$target_branch" target_worktree
  if $DRY_RUN; then
    print_cmd git -C "$target_worktree" merge --no-edit "$source_branch"
    UPDATED_BRANCHES+=("$target_branch")
    dry_run_stack_changed=true
  else
    if git -C "$target_worktree" merge --no-edit "$source_branch"; then
      info "Merged $source_branch into $target_branch"
      UPDATED_BRANCHES+=("$target_branch")
    else
      warn "Merge failed for $target_branch. Conflicted worktree kept at: $target_worktree"
      warn "Resolve conflicts, commit the merge on $target_branch, push it, then rerun from $target_branch if downstream branches remain"
      if [[ ${#UPDATED_BRANCHES[@]} -gt 0 ]]; then
        if $NO_PUSH; then
          warn "Previously updated downstream branches were not pushed because --no-push was used: ${UPDATED_BRANCHES[*]}"
        else
          warn "Pushing previously updated downstream branches before stopping: ${UPDATED_BRANCHES[*]}"
          if ! push_updated_branches; then
            warn "Failed to push previously updated downstream branches; push them manually: ${UPDATED_BRANCHES[*]}"
          fi
        fi
      fi
      forget_temp_worktree "$target_worktree"
      exit 1
    fi
  fi

  if $KEEP_TEMP; then
    info "Preserving temp worktree: $target_worktree"
    forget_temp_worktree "$target_worktree"
  else
    remove_temp_worktree "$target_worktree"
  fi

  source_branch="$target_branch"
done

if [[ ${#UPDATED_BRANCHES[@]} -eq 0 ]]; then
  info "No downstream branches needed updates"
elif $NO_PUSH; then
  info "Skipping push (--no-push)"
else
  push_updated_branches
fi

info "Retained branch propagation complete"
```

- [ ] **Step 3: Run shell syntax check**

Run:

```bash
uv run bash -n scripts/propagate-retained.sh
```

Expected: exit code `0` and no output.

- [ ] **Step 4: Mark helper executable**

Run:

```bash
chmod +x scripts/propagate-retained.sh
```

Expected: no output.

## Task 3: Add Custom-Only Preservation and Workflow Docs

**Files:**
- Modify: `.sync-config`
- Modify: `AGENTS.md`

- [ ] **Step 1: Add helper to `.sync-config` custom files**

In `.sync-config`, update `CUSTOM_FILES` so this block includes the new helper:

```bash
CUSTOM_FILES=(
  .sync-config
  AGENTS.md
  scripts/rebuild-custom.sh
  scripts/sync-upstream.sh
  scripts/propagate-retained.sh
  scripts/deploy.sh
)
```

- [ ] **Step 2: Add workflow keyword to `AGENTS.md`**

In `AGENTS.md` under `## Workflow Keywords`, add this item after `sync retained`:

```markdown
- `propagate retained`
  Run `scripts/propagate-retained.sh --from BRANCH`
```

- [ ] **Step 3: Add helper to custom-only files list**

In `AGENTS.md` under `## Custom-Only Files`, make the list include:

```markdown
- `.sync-config`
- `AGENTS.md`
- `scripts/rebuild-custom.sh`
- `scripts/sync-upstream.sh`
- `scripts/propagate-retained.sh`
- `scripts/deploy.sh`
```

- [ ] **Step 4: Add retained stack propagation standard workflow section**

In `AGENTS.md`, insert this section between `### 1. Sync upstream` and `### 2. Rebuild custom`, then renumber the later headings so rebuild becomes `### 3. Rebuild custom` and deploy becomes `### 4. Deploy custom`:

```markdown
### 2. Propagate retained branch stack

Run this from the persistent `custom` worktree after committing and pushing a retained branch that is not the last branch in `MERGE_ORDER`.

Default command:

```bash
scripts/propagate-retained.sh --from BRANCH
```

Behavior:

- find `BRANCH` in `MERGE_ORDER`
- merge that branch into each downstream retained branch in order
- after each merge, use the updated downstream branch as the next source branch
- push updated downstream retained branches to `origin` by default

Use `--to BRANCH` to stop at a specific downstream branch. Use `--no-push` only when intentionally keeping propagated branch tips local.

If propagation hits a conflict, resolve it on the downstream retained branch where the conflict occurred, commit that merge there, push it, and rerun propagation from that branch if more downstream branches remain.

Retained branches are self-contained by ownership, but ordered by ancestry for integration compatibility. Do not resolve retained-branch compatibility conflicts only on `custom`.
```

- [ ] **Step 5: Update conflict policy wording**

In `AGENTS.md` under `### Conflicts During Custom Rebuild`, add this sentence after the existing instruction to fix conflicts on relevant retained branches:

```markdown
If the conflict is caused by an earlier retained branch changing after downstream branches already merged older ancestry, run `scripts/propagate-retained.sh --from <earlier-branch>` after committing the retained-branch fix, then rerun `scripts/rebuild-custom.sh`.
```

## Task 4: Verify Helper Behavior

**Files:**
- Verify: `scripts/propagate-retained.sh`
- Verify: `.sync-config`
- Verify: `AGENTS.md`

- [ ] **Step 1: Run shell syntax check**

Run:

```bash
uv run bash -n scripts/propagate-retained.sh
```

Expected: exit code `0` and no output.

- [ ] **Step 2: Verify help output**

Run:

```bash
scripts/propagate-retained.sh --help
```

Expected: exit code `0` and output starts with:

```text
Usage: scripts/propagate-retained.sh --from BRANCH [OPTIONS]
```

- [ ] **Step 3: Verify missing `--from` fails clearly**

Run:

```bash
scripts/propagate-retained.sh --dry-run
```

Expected: non-zero exit and output contains:

```text
[FAIL] Missing required --from BRANCH
```

- [ ] **Step 4: Verify unknown branch fails clearly**

Run:

```bash
scripts/propagate-retained.sh --from feat/not-a-real-branch --dry-run
```

Expected: non-zero exit and output contains:

```text
[FAIL] --from branch is not in MERGE_ORDER: feat/not-a-real-branch
```

- [ ] **Step 5: Verify git state contains only intended implementation files**

Run:

```bash
git status --short --branch
```

Expected output:

```text
## custom...origin/custom
 M .sync-config
 M AGENTS.md
?? scripts/propagate-retained.sh
```

The exact branch status line may show `[ahead N]` if commits have not been pushed yet, but only `.sync-config`, `AGENTS.md`, and `scripts/propagate-retained.sh` should be changed.

## Task 5: Commit, Dry-Run, and Push Workflow Helper

**Files:**
- Commit: `scripts/propagate-retained.sh`
- Commit: `.sync-config`
- Commit: `AGENTS.md`

- [ ] **Step 1: Review final diff**

Run:

```bash
git diff -- .sync-config AGENTS.md scripts/propagate-retained.sh
```

Expected: diff only contains the helper script, `.sync-config` custom file addition, and `AGENTS.md` workflow documentation.

- [ ] **Step 2: Commit helper and docs**

Run:

```bash
git add .sync-config AGENTS.md scripts/propagate-retained.sh
git commit -m "workflow: add retained branch propagation helper"
```

Expected: commit succeeds.

- [ ] **Step 3: Verify short-range dry run from clean committed state**

Run:

```bash
scripts/propagate-retained.sh --from feat/longitudinal-osm-planner --to feat/speed-limit-auto-cruise --dry-run --no-push
```

Expected: exit code `0`. If `feat/speed-limit-auto-cruise` already contains `feat/longitudinal-osm-planner`, output contains:

```text
[INFO] feat/speed-limit-auto-cruise already contains feat/longitudinal-osm-planner
[INFO] No downstream branches needed updates
[INFO] Retained branch propagation complete
```

If it does not already contain the source, output contains a dry-run `git worktree add` command and a dry-run `git -C ... merge --no-edit feat/longitudinal-osm-planner` command.

- [ ] **Step 4: Push custom**

Run:

```bash
git push origin custom
```

Expected: push succeeds.

- [ ] **Step 5: Verify final status**

Run:

```bash
git status --short --branch
```

Expected:

```text
## custom...origin/custom
```
