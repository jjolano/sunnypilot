#!/usr/bin/env bash

set -euo pipefail

DRY_RUN=false
KEEP_TEMP=false
NO_ABORT=false
REBUILD_SUCCESS=false
CONFIG_FILE=".sync-config"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

source "$SCRIPT_DIR/lib/workflow.sh"

usage() {
  cat <<'EOF'
Usage: scripts/rebuild-custom.sh [OPTIONS]

Rebuild the integration branch from the configured base branch plus any
retained feature branches listed in MERGE_ORDER. Run this from the persistent
integration worktree.

Options:
  --config <path>   Config file path relative to repo root (default: .sync-config)
  --dry-run         Print the actions without modifying git state
  --keep-temp       Preserve the temporary save directory on success/failure
  --no-abort        Keep conflicted merges for investigation
  --help            Show this help message
EOF
}

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
    --keep-temp)
      KEEP_TEMP=true
      shift
      ;;
    --no-abort)
      NO_ABORT=true
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

CURRENT_BRANCH="$(git branch --show-current)"
[[ "$CURRENT_BRANCH" == "$INTEGRATION_BRANCH" ]] || fail "Run this script from the $INTEGRATION_BRANCH worktree (current branch: ${CURRENT_BRANCH:-detached HEAD})"

if ! git diff --quiet || ! git diff --cached --quiet; then
  fail "Working tree is not clean. Commit or stash changes before rebuilding"
fi

if [[ -n "$(git ls-files --others --exclude-standard)" ]]; then
  fail "Untracked files are present. Commit, stash, or clean them before rebuilding"
fi

for branch in "$LOCAL_BRANCH" "$INTEGRATION_BRANCH" "${MERGE_ORDER[@]}"; do
  git show-ref --verify --quiet "refs/heads/$branch" || fail "Missing local branch: $branch"
done

info "Verifying build environment"
command -v uv >/dev/null 2>&1 || fail "uv not found in PATH"
uv run --python 3.12 python --version >/dev/null 2>&1 || fail "uv could not find/run python 3.12"

TMPDIR=""
MODES_FILE=""

cleanup() {
  if [[ -n "$TMPDIR" && -d "$TMPDIR" ]]; then
    if $KEEP_TEMP || ! $REBUILD_SUCCESS; then
      info "Preserving temp directory: $TMPDIR"
      if ! $REBUILD_SUCCESS; then
        warn "Rebuild failed. Custom files are preserved in $TMPDIR"
      fi
    else
      rm -rf "$TMPDIR"
    fi
  fi
}

trap cleanup EXIT

save_custom_files() {
  local path rel_dir object_type tree_entry mode

  if $DRY_RUN; then
    for path in "${CUSTOM_FILES[@]}"; do
      info "Would save $path from $INTEGRATION_BRANCH"
    done
    return
  fi

  TMPDIR="$(mktemp -d)"
  MODES_FILE="$TMPDIR/.modes"
  for path in "${CUSTOM_FILES[@]}"; do
    rel_dir="$(dirname "$TMPDIR/$path")"
    mkdir -p "$rel_dir"
    if git cat-file -e "$INTEGRATION_BRANCH:$path" 2>/dev/null; then
      object_type="$(git cat-file -t "$INTEGRATION_BRANCH:$path")"
      if [[ "$object_type" == "tree" ]]; then
        git archive "$INTEGRATION_BRANCH" "$path" | tar -x -C "$TMPDIR"
        info "Saved $path"
      else
        git show "$INTEGRATION_BRANCH:$path" >"$TMPDIR/$path"
        tree_entry="$(git ls-tree "$INTEGRATION_BRANCH" -- "$path")"
        mode="${tree_entry%% *}"
        printf '%s\t%s\n' "$path" "$mode" >>"$MODES_FILE"
        info "Saved $path"
      fi
    else
      info "Skipping missing custom-only file: $path"
    fi
  done
}

restore_custom_files() {
  local path target_dir restored_any=false mode_entry mode

  if $DRY_RUN; then
    for path in "${CUSTOM_FILES[@]}"; do
      info "Would restore $path onto $INTEGRATION_BRANCH"
    done
    return
  fi

  for path in "${CUSTOM_FILES[@]}"; do
    if [[ -d "$TMPDIR/$path" ]]; then
      mkdir -p "$path"
      cp -a "$TMPDIR/$path/." "$path/"
      git add -f -- "$path"
      restored_any=true
      info "Restored $path"
    elif [[ -f "$TMPDIR/$path" ]]; then
      target_dir="$(dirname "$path")"
      mkdir -p "$target_dir"
      cp "$TMPDIR/$path" "$path"
      if [[ -f "$MODES_FILE" ]]; then
        mode_entry="$(grep -F "$path"$'\t' "$MODES_FILE" || true)"
        mode="${mode_entry##*$'\t'}"
        if [[ "$mode" == "100755" ]]; then
          chmod +x "$path"
        else
          chmod 0644 "$path"
        fi
      fi
      git add -f -- "$path"
      restored_any=true
      info "Restored $path"
    fi
  done

  if $restored_any && ! git diff --cached --quiet; then
    git commit -m "$METADATA_COMMIT_MSG"
    info "Created metadata commit: $METADATA_COMMIT_MSG"
  else
    info "No custom-only file changes to commit"
  fi
}

merge_branch() {
  local branch="$1"

  if $DRY_RUN; then
    info "Would merge $branch into $INTEGRATION_BRANCH"
    return
  fi

  if git merge-base --is-ancestor "$branch" HEAD; then
    info "$branch already contained in $INTEGRATION_BRANCH"
    return
  fi

  if git merge --no-edit "$branch"; then
    info "Merged $branch"
  else
    warn "Merge failed for branch: $branch"
    warn "Resolve the long-term fix on the owning retained branch, then rerun the rebuild"
    if $NO_ABORT; then
      warn "--no-abort specified; keeping conflicted state for investigation"
    else
      git merge --abort 2>/dev/null || true
    fi
    exit 1
  fi
}

rebuild_longitudinal_mpc() {
  local target="selfdrive/controls/lib/longitudinal_mpc_lib/c_generated_code/acados_ocp_solver_pyx.so"

  info "Rebuilding longitudinal MPC artifacts"
  run uv run --extra testing --extra tools --python 3.12 scons -u "$target"
}

info "Saving custom-only files"
save_custom_files

info "Resetting $INTEGRATION_BRANCH to $LOCAL_BRANCH in the current worktree"
run git reset --hard "$LOCAL_BRANCH"

if [[ ${#MERGE_ORDER[@]} -eq 0 ]]; then
  info "MERGE_ORDER is empty; rebuilding a baseline $INTEGRATION_BRANCH"
fi

for branch in "${MERGE_ORDER[@]}"; do
  merge_branch "$branch"
done

info "Restoring custom-only files"
restore_custom_files

rebuild_longitudinal_mpc

if $DRY_RUN; then
  info "Dry run complete"
else
  REBUILD_SUCCESS=true
  info "Rebuild complete on $INTEGRATION_BRANCH"
fi
