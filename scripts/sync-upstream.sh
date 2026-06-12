#!/usr/bin/env bash

set -euo pipefail

DRY_RUN=false
NO_PUSH=false
SYNC_RETAINED=false
REBUILD_CUSTOM=false
CONFIG_FILE=".sync-config"
TEMP_DIR=""
REBUILD_SCRIPT_PATH=""
REBUILD_CONFIG_PATH=""
TEMP_WORKTREES=()
REMOVED_TEMP_WORKTREE=false
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

source "$SCRIPT_DIR/lib/workflow.sh"

usage() {
  cat <<'EOF'
Usage: scripts/sync-upstream.sh [OPTIONS]

Fast-forward the configured base branch from upstream and push it to origin.
With --sync-retained, merge the updated base branch into RETAINED_BASE_BRANCH,
then merge RETAINED_BASE_BRANCH independently into each remaining domain branch
in MERGE_ORDER. Updated branches are pushed to origin unless --no-push is used.
Run this from the persistent integration worktree. Retained-branch sync and
custom rebuilds are explicit opt-in steps, and branch updates run in temporary
worktrees. Conflict worktrees are preserved for resolution; successful temporary
worktrees are removed before the script exits.

Options:
  --config <path>   Config file path relative to repo root (default: .sync-config)
  --dry-run         Print the actions without modifying git state
  --no-push         Skip pushing updated branches to origin
  --sync-retained   Merge the updated base branch into every branch in MERGE_ORDER
  --rebuild-custom  Rebuild the integration branch after syncing retained branches
  --help            Show this help message
EOF
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

create_temp_worktree() {
  local branch="$1"
  local out_var="$2"
  local safe_branch path

  ensure_branch_available_for_temp_worktree "$branch"

  safe_branch="${branch//\//-}"
  if $DRY_RUN; then
    path="${TMPDIR:-/tmp}/sunnypilot-${safe_branch}.DRYRUN"
    print_cmd git worktree add "$path" "$branch"
  else
    path="$(mktemp -d "${TMPDIR:-/tmp}/sunnypilot-${safe_branch}.XXXXXX")"
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

  if [[ -n "$TEMP_DIR" && -d "$TEMP_DIR" ]]; then
    rm -rf "$TEMP_DIR"
  fi
}

trap cleanup EXIT

while [[ $# -gt 0 ]]; do
  case "$1" in
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
    --sync-retained)
      SYNC_RETAINED=true
      shift
      ;;
    --rebuild-custom)
      REBUILD_CUSTOM=true
      SYNC_RETAINED=true
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

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
[[ -n "$REPO_ROOT" ]] || fail "Not inside a git repository"
cd "$REPO_ROOT"

load_sync_config "$REPO_ROOT" "$CONFIG_FILE"

RETAINED_BASE_BRANCH="${RETAINED_BASE_BRANCH:-}"
if [[ -z "$RETAINED_BASE_BRANCH" && ${#MERGE_ORDER[@]} -gt 0 ]]; then
  RETAINED_BASE_BRANCH="${MERGE_ORDER[0]}"
fi

REBUILD_SCRIPT_PATH="$REPO_ROOT/scripts/rebuild-custom.sh"
REBUILD_CONFIG_PATH="$CONFIG_PATH"

CURRENT_BRANCH="$(git branch --show-current)"
[[ "$CURRENT_BRANCH" == "$INTEGRATION_BRANCH" ]] || fail "Run this script from the $INTEGRATION_BRANCH worktree (current branch: ${CURRENT_BRANCH:-detached HEAD})"

if ! $DRY_RUN && $REBUILD_CUSTOM; then
  TEMP_DIR="$(mktemp -d)"
  mkdir -p "$TEMP_DIR/lib"
  cp "$REPO_ROOT/scripts/rebuild-custom.sh" "$TEMP_DIR/rebuild-custom.sh"
  cp "$REPO_ROOT/scripts/lib/workflow.sh" "$TEMP_DIR/lib/workflow.sh"
  chmod +x "$TEMP_DIR/rebuild-custom.sh"
  cp "$CONFIG_PATH" "$TEMP_DIR/.sync-config"
  REBUILD_SCRIPT_PATH="$TEMP_DIR/rebuild-custom.sh"
  REBUILD_CONFIG_PATH="$TEMP_DIR/.sync-config"
fi

SYNC_SOURCE_BRANCH="$LOCAL_BRANCH"

if ! git diff --quiet || ! git diff --cached --quiet; then
  fail "Working tree is not clean. Commit or stash changes before syncing"
fi

if [[ -n "$(git ls-files --others --exclude-standard)" ]]; then
  fail "Untracked files are present. Commit, stash, or clean them before syncing"
fi

git remote | grep -qx "$UPSTREAM_REMOTE" || fail "Remote '$UPSTREAM_REMOTE' not found"
git remote | grep -qx "origin" || fail "Remote 'origin' not found"
git show-ref --verify --quiet "refs/heads/$LOCAL_BRANCH" || fail "Missing local branch: $LOCAL_BRANCH"
if $SYNC_RETAINED; then
  for branch in "${MERGE_ORDER[@]}"; do
    git show-ref --verify --quiet "refs/heads/$branch" || fail "Missing local branch: $branch"
  done
fi

if $REBUILD_CUSTOM; then
  git show-ref --verify --quiet "refs/heads/$INTEGRATION_BRANCH" || fail "Missing local branch: $INTEGRATION_BRANCH"
fi

info "Fetching $UPSTREAM_REMOTE"
run git fetch "$UPSTREAM_REMOTE"

if $DRY_RUN; then
  SYNC_SOURCE_BRANCH="$UPSTREAM_REMOTE/$UPSTREAM_BRANCH"
fi

info "Fast-forwarding $LOCAL_BRANCH"
LOCAL_BRANCH_WORKTREE=""
create_temp_worktree "$LOCAL_BRANCH" LOCAL_BRANCH_WORKTREE
if $DRY_RUN; then
  print_cmd git -C "$LOCAL_BRANCH_WORKTREE" merge --ff-only "$UPSTREAM_REMOTE/$UPSTREAM_BRANCH"
else
  git -C "$LOCAL_BRANCH_WORKTREE" merge --ff-only "$UPSTREAM_REMOTE/$UPSTREAM_BRANCH"
fi
remove_temp_worktree "$LOCAL_BRANCH_WORKTREE"

sync_branch() {
  local branch="$1"
  local source_branch="$2"
  local branch_worktree=""

  info "Syncing $branch from $source_branch"

  if git merge-base --is-ancestor "$source_branch" "$branch" 2>/dev/null; then
    info "$branch already contains $source_branch"
  else
    create_temp_worktree "$branch" branch_worktree

    if $DRY_RUN; then
      info "Would merge $source_branch into $branch"
      print_cmd git -C "$branch_worktree" merge --no-edit "$source_branch"
    else
      if git -C "$branch_worktree" merge --no-edit "$source_branch"; then
        info "Merged $source_branch into $branch"
      else
        warn "Merge failed for $branch. Conflicted worktree kept at: $branch_worktree"
        warn "Resolve conflicts in that worktree, commit the merge on $branch, then resume sync with:"
        warn "  scripts/sync-upstream.sh --sync-retained"
        forget_temp_worktree "$branch_worktree"
        exit 1
      fi
    fi

    remove_temp_worktree "$branch_worktree"
  fi
}

if $SYNC_RETAINED; then
  if [[ ${#MERGE_ORDER[@]} -eq 0 ]]; then
    info "MERGE_ORDER is empty; no retained branches to sync"
  fi

  if [[ -n "$RETAINED_BASE_BRANCH" ]]; then
    sync_branch "$RETAINED_BASE_BRANCH" "$SYNC_SOURCE_BRANCH"
  fi

  for branch in "${MERGE_ORDER[@]}"; do
    if [[ "$branch" == "$RETAINED_BASE_BRANCH" ]]; then
      continue
    fi
    sync_branch "$branch" "$RETAINED_BASE_BRANCH"
  done
else
  info "Skipping retained branch sync"
fi

if $REBUILD_CUSTOM; then
  info "Rebuilding $INTEGRATION_BRANCH"
  if $DRY_RUN; then
    print_cmd "$REPO_ROOT/scripts/rebuild-custom.sh" --config "$CONFIG_FILE" --dry-run
  else
      "$REBUILD_SCRIPT_PATH" --config "$REBUILD_CONFIG_PATH"
  fi
else
  info "Skipping rebuild (use --rebuild-custom to refresh $INTEGRATION_BRANCH)"
fi

if $NO_PUSH; then
  info "Skipping push (--no-push)"
else
  push_branches=("$LOCAL_BRANCH")
  if $SYNC_RETAINED; then
    push_branches+=("${MERGE_ORDER[@]}")
  fi

  if $DRY_RUN; then
    print_cmd env GIT_LFS_SKIP_PUSH=1 git push origin "${push_branches[@]}"
    if $REBUILD_CUSTOM; then
      print_cmd env GIT_LFS_SKIP_PUSH=1 git push --force-with-lease origin "$INTEGRATION_BRANCH"
    fi
  else
    retry env GIT_LFS_SKIP_PUSH=1 git push origin "${push_branches[@]}"
    if $REBUILD_CUSTOM; then
      retry env GIT_LFS_SKIP_PUSH=1 git push --force-with-lease origin "$INTEGRATION_BRANCH"
    fi
  fi
fi

info "Sync complete"
