#!/usr/bin/env bash
set -euo pipefail

SYNC_RETAINED=false
REBUILD_CUSTOM=false
CONFIG_FILE=".sync-config"

fail() {
  printf 'ERROR: %s\n' "$1" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage: sync_upstream_preflight.sh [OPTIONS]

Options:
  --config <path>   Config file path relative to repo root (default: .sync-config)
  --sync-retained   Also check retained branches from MERGE_ORDER
  --rebuild-custom  Also check retained branches and integration branch
  --help            Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      [[ $# -ge 2 ]] || fail "missing value for --config"
      CONFIG_FILE="$2"
      shift 2
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
      fail "unknown option: $1"
      ;;
  esac
done

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
[[ -n "$ROOT" ]] || fail "not inside a git repository"
cd "$ROOT"

if [[ "$CONFIG_FILE" == /* ]]; then
  CONFIG_PATH="$CONFIG_FILE"
else
  CONFIG_PATH="$ROOT/$CONFIG_FILE"
fi
[[ -f "$CONFIG_PATH" ]] || fail "config file not found: $CONFIG_FILE"
# shellcheck source=/dev/null
source "$CONFIG_PATH"

current_branch="$(git branch --show-current)"
[[ "$current_branch" == "$INTEGRATION_BRANCH" ]] || fail "expected $INTEGRATION_BRANCH branch, got ${current_branch:-detached}"

[[ -z "$(git status --short)" ]] || fail "$INTEGRATION_BRANCH worktree is dirty or has untracked files"

git remote | grep -qx "$UPSTREAM_REMOTE" || fail "remote '$UPSTREAM_REMOTE' not found"
git remote | grep -qx "origin" || fail "remote 'origin' not found"
git show-ref --verify --quiet "refs/heads/$LOCAL_BRANCH" || fail "missing local branch: $LOCAL_BRANCH"

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
      worktree\ *) worktree_path="${line#worktree }" ;;
      branch\ *) worktree_branch="${line#branch }" ;;
    esac
  done < <(git worktree list --porcelain)

  if [[ "$worktree_branch" == "refs/heads/$branch" ]]; then
    printf '%s\n' "$worktree_path"
    return 0
  fi

  return 1
}

check_branch_available() {
  local branch="$1"
  local existing_worktree

  existing_worktree="$(find_branch_worktree "$branch" || true)"
  [[ -z "$existing_worktree" ]] || fail "branch '$branch' is already checked out in worktree: $existing_worktree"
}

check_branch_available "$LOCAL_BRANCH"

if $SYNC_RETAINED; then
  for branch in "${MERGE_ORDER[@]}"; do
    git show-ref --verify --quiet "refs/heads/$branch" || fail "missing local branch: $branch"
    check_branch_available "$branch"
  done
fi

if $REBUILD_CUSTOM; then
  git show-ref --verify --quiet "refs/heads/$INTEGRATION_BRANCH" || fail "missing local branch: $INTEGRATION_BRANCH"
fi

printf 'branch: %s\n' "$current_branch"
printf 'config: %s\n' "$CONFIG_FILE"
printf 'base branch: %s\n' "$LOCAL_BRANCH"
printf 'upstream: %s/%s\n' "$UPSTREAM_REMOTE" "$UPSTREAM_BRANCH"
printf 'retained sync: %s\n' "$SYNC_RETAINED"
printf 'rebuild custom: %s\n' "$REBUILD_CUSTOM"
printf 'status: clean\n'
printf 'target worktrees: available\n'
