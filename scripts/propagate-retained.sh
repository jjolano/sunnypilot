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
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

source "$SCRIPT_DIR/lib/workflow.sh"

usage() {
  cat <<'EOF'
Usage: scripts/propagate-retained.sh --from BRANCH [OPTIONS]

Merge RETAINED_BASE_BRANCH into the independent domain branches listed after it
in MERGE_ORDER. Run this from the persistent integration worktree. Successful
temporary worktrees are removed before exit unless --keep-temp is used.
Conflicted temporary worktrees are preserved for manual conflict resolution.

Options:
  --from <branch>  Source retained branch to propagate (must be RETAINED_BASE_BRANCH)
  --to <branch>    Final domain branch to update (default: last MERGE_ORDER branch)
  --config <path>  Config file path relative to repo root (default: .sync-config)
  --dry-run        Print the actions without modifying git state
  --no-push        Skip pushing updated domain branches to origin
  --keep-temp      Preserve successful temporary worktrees for inspection
  --help           Show this help message
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
    retry env GIT_LFS_SKIP_PUSH=1 git push origin "${UPDATED_BRANCHES[@]}"
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

load_sync_config "$REPO_ROOT" "$CONFIG_FILE"

RETAINED_BASE_BRANCH="${RETAINED_BASE_BRANCH:-}"
if [[ -z "$RETAINED_BASE_BRANCH" && ${#MERGE_ORDER[@]} -gt 0 ]]; then
  RETAINED_BASE_BRANCH="${MERGE_ORDER[0]}"
fi

CURRENT_BRANCH="$(git branch --show-current)"
[[ "$CURRENT_BRANCH" == "$INTEGRATION_BRANCH" ]] || fail "Run this script from the $INTEGRATION_BRANCH worktree (current branch: ${CURRENT_BRANCH:-detached HEAD})"

[[ ${#MERGE_ORDER[@]} -gt 0 ]] || fail "MERGE_ORDER is empty; no retained branches to propagate"

FROM_INDEX="$(find_merge_order_index "$FROM_BRANCH" || true)"
[[ -n "$FROM_INDEX" ]] || fail "--from branch is not in MERGE_ORDER: $FROM_BRANCH"
[[ "$FROM_BRANCH" == "$RETAINED_BASE_BRANCH" ]] || fail "Domain branches are independent. Propagate only from $RETAINED_BASE_BRANCH, not $FROM_BRANCH"

if [[ -z "$TO_BRANCH" ]]; then
  TO_INDEX=$((${#MERGE_ORDER[@]} - 1))
  TO_BRANCH="${MERGE_ORDER[$TO_INDEX]}"
else
  TO_INDEX="$(find_merge_order_index "$TO_BRANCH" || true)"
  [[ -n "$TO_INDEX" ]] || fail "--to branch is not in MERGE_ORDER: $TO_BRANCH"
fi

if (( TO_INDEX < FROM_INDEX )); then
  fail "--to branch must be the same as or after --from in MERGE_ORDER"
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

for ((idx = FROM_INDEX + 1; idx <= TO_INDEX; idx++)); do
  ensure_branch_available_for_temp_worktree "${MERGE_ORDER[$idx]}"
done

if (( FROM_INDEX == TO_INDEX )); then
  info "No domain branches to update from $FROM_BRANCH to $TO_BRANCH"
  exit 0
fi

UPDATED_BRANCHES=()

for ((idx = FROM_INDEX + 1; idx <= TO_INDEX; idx++)); do
  target_branch="${MERGE_ORDER[$idx]}"
  target_worktree=""

  info "Propagating $RETAINED_BASE_BRANCH into $target_branch"
  if git merge-base --is-ancestor "$RETAINED_BASE_BRANCH" "$target_branch"; then
    info "$target_branch already contains $RETAINED_BASE_BRANCH"
    continue
  fi

  create_temp_worktree "$target_branch" target_worktree
  if $DRY_RUN; then
    print_cmd git -C "$target_worktree" merge --no-edit "$RETAINED_BASE_BRANCH"
    UPDATED_BRANCHES+=("$target_branch")
  else
    if git -C "$target_worktree" merge --no-edit "$RETAINED_BASE_BRANCH"; then
      info "Merged $RETAINED_BASE_BRANCH into $target_branch"
      UPDATED_BRANCHES+=("$target_branch")
    else
      warn "Merge failed for $target_branch. Conflicted worktree kept at: $target_worktree"
      warn "Resolve conflicts in that worktree, commit the merge on $target_branch, push it, then rerun propagation with:"
      warn "  scripts/propagate-retained.sh --from $RETAINED_BASE_BRANCH"
      if [[ ${#UPDATED_BRANCHES[@]} -gt 0 ]]; then
        if $NO_PUSH; then
          warn "Previously updated domain branches were not pushed because --no-push was used: ${UPDATED_BRANCHES[*]}"
        else
          warn "Pushing previously updated domain branches before stopping: ${UPDATED_BRANCHES[*]}"
          if ! push_updated_branches; then
            warn "Failed to push previously updated domain branches; push them manually: ${UPDATED_BRANCHES[*]}"
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

done

if [[ ${#UPDATED_BRANCHES[@]} -gt 0 ]]; then
  if $NO_PUSH; then
    info "Retained branch propagation complete. Updated locally: ${UPDATED_BRANCHES[*]}"
  else
    push_updated_branches
    info "Retained branch propagation complete. Updated and pushed: ${UPDATED_BRANCHES[*]}"
  fi
else
  info "Retained branch propagation complete (no updates needed)"
fi
