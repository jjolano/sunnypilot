#!/usr/bin/env bash

set -euo pipefail

DRY_RUN=false
REBOOT=true
CONFIG_FILE=".deploy-config"
CLI_DEPLOY_HOST=""
CLI_DEPLOY_REMOTE=""
CLI_DEPLOY_PATH=""
PUSH_REMOTE="origin"
RETRY_MAX=3
RETRY_DELAY_SECONDS="${DEPLOY_RETRY_DELAY_SECONDS:-1}"

usage() {
  cat <<'EOF'
Usage: scripts/deploy.sh [OPTIONS]

Push the deploy branch and reset the device repo to its pinned commit. Run from
the deploy branch with a clean tree of committed work.

Options:
  --config <path>   Config file path relative to repo root (default: .deploy-config)
  --dry-run         Print the actions without modifying git state or device state
  --host <host>     Override DEPLOY_HOST from config
  --path <path>     Override DEPLOY_PATH from config
  --reboot          Reboot the device after deployment (default)
  --no-reboot       Skip reboot after deployment
  --remote <name>   Override DEPLOY_REMOTE from config
  --help            Show this help message
EOF
}

info() { printf 'INFO: %s\n' "$1"; }
warn() { printf 'WARN: %s\n' "$1" >&2; }
fail() { printf 'ERROR: %s\n' "$1" >&2; exit 1; }

print_cmd() {
  local arg
  printf '+'
  for arg in "$@"; do
    printf ' %q' "$arg"
  done
  printf '\n'
}

remote_quote() {
  local value="$1"
  local quoted="'"
  local i char
  for ((i = 0; i < ${#value}; i++)); do
    char="${value:i:1}"
    if [[ "$char" == "'" ]]; then
      quoted+="'\\''"
    else
      quoted+="$char"
    fi
  done
  quoted+="'"
  printf '%s' "$quoted"
}

retry_local() {
  local description="$1"
  shift
  local attempt rc
  for ((attempt = 1; attempt <= RETRY_MAX; attempt++)); do
    if "$@"; then
      return 0
    else
      rc=$?
    fi
    if ((attempt < RETRY_MAX)); then
      warn "$description failed (attempt $attempt/$RETRY_MAX); retrying"
      sleep "$RETRY_DELAY_SECONDS"
    else
      fail "$description failed after $RETRY_MAX attempts (exit $rc)"
    fi
  done
}

run_remote() {
  ssh -o ConnectTimeout=10 "$DEPLOY_HOST" "$1"
}

retry_remote_prepare() {
  local description="$1"
  local command="$2"
  local attempt rc
  for ((attempt = 1; attempt <= RETRY_MAX; attempt++)); do
    if run_remote "$command"; then
      return 0
    else
      rc=$?
    fi
    if ((attempt < RETRY_MAX)); then
      warn "$description failed (attempt $attempt/$RETRY_MAX); retrying"
      sleep "$RETRY_DELAY_SECONDS"
    else
      fail "$description failed after $RETRY_MAX attempts (exit $rc)"
    fi
  done
}

check_device_offroad() {
  local gate="$1"
  local rc
  info "Checking live deviceState.started before $gate"
  set +e
  timeout 20s ssh -o ConnectTimeout=10 "$DEPLOY_HOST" "$REMOTE_OFFROAD_CHECK_CMD" < "$CHECKER_PATH"
  rc=$?
  set -e
  case "$rc" in
    0) return 0 ;;
    42) fail "Device is onroad (deviceState.started=true); deployment stopped" ;;
    *) fail "Device offroad check failed (exit $rc); deployment stopped fail-closed" ;;
  esac
}

verify_pushed_sha() {
  local remote_line remote_sha remote_ref
  if ! remote_line="$(git ls-remote "$PUSH_REMOTE" "refs/heads/$DEPLOY_BRANCH")"; then
    fail "Unable to verify the deploy branch on the push remote; inspect remote state"
  fi
  read -r remote_sha remote_ref <<< "$remote_line" || fail "Push remote returned no deploy branch"
  [[ "$remote_ref" == "refs/heads/$DEPLOY_BRANCH" ]] || fail "Push remote returned an unexpected deploy ref"
  [[ "$remote_sha" =~ ^[0-9a-f]{40}$ ]] || fail "Push remote returned an invalid deploy SHA"
  [[ "$remote_sha" == "$TARGET_SHA" ]] || fail "Push remote deploy branch is $remote_sha, expected $TARGET_SHA"
}

print_remote_command() {
  printf '+ ssh -o ConnectTimeout=10 <device> %s\n' "$1"
}

print_offroad_command() {
  printf '+ timeout 20s ssh -o ConnectTimeout=10 <device> %s < %q\n' "$REMOTE_OFFROAD_CHECK_CMD" "$CHECKER_PATH"
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
    --host)
      [[ $# -ge 2 ]] || fail "Missing value for --host"
      CLI_DEPLOY_HOST="$2"
      shift 2
      ;;
    --path)
      [[ $# -ge 2 ]] || fail "Missing value for --path"
      CLI_DEPLOY_PATH="$2"
      shift 2
      ;;
    --reboot)
      REBOOT=true
      shift
      ;;
    --no-reboot)
      REBOOT=false
      shift
      ;;
    --remote)
      [[ $# -ge 2 ]] || fail "Missing value for --remote"
      CLI_DEPLOY_REMOTE="$2"
      shift 2
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

[[ -f "$REPO_ROOT/$CONFIG_FILE" ]] || fail "Config file not found: $CONFIG_FILE"
# shellcheck source=/dev/null
source "$REPO_ROOT/$CONFIG_FILE"
for var in DEPLOY_BRANCH DEPLOY_HOST DEPLOY_REMOTE DEPLOY_PATH; do
  [[ -n "${!var:-}" ]] || fail "Missing $var in $CONFIG_FILE"
done

[[ -n "$CLI_DEPLOY_HOST" ]] && DEPLOY_HOST="$CLI_DEPLOY_HOST"
[[ -n "$CLI_DEPLOY_REMOTE" ]] && DEPLOY_REMOTE="$CLI_DEPLOY_REMOTE"
[[ -n "$CLI_DEPLOY_PATH" ]] && DEPLOY_PATH="$CLI_DEPLOY_PATH"

CURRENT_BRANCH="$(git branch --show-current)"
[[ "$CURRENT_BRANCH" == "$DEPLOY_BRANCH" ]] || fail "Run this script from the $DEPLOY_BRANCH branch (current branch: ${CURRENT_BRANCH:-detached HEAD})"
git remote | grep -qx "$PUSH_REMOTE" || fail "Remote '$PUSH_REMOTE' not found"

if [[ -n "$(git ls-files -u)" ]]; then
  if $DRY_RUN; then
    warn "Working tree has unmerged files; dry-run will not execute deployment actions"
  else
    fail "Working tree has unmerged files (conflicts). Resolve them before deploying"
  fi
fi
if [[ -f "$(git rev-parse --git-dir)/MERGE_HEAD" ]]; then
  if $DRY_RUN; then
    warn "A merge is in progress; dry-run will not execute deployment actions"
  else
    fail "A merge is currently in progress. Finish or abort it before deploying"
  fi
fi
WORKTREE_STATUS="$(git status --porcelain)"
if [[ -n "$WORKTREE_STATUS" ]]; then
  if $DRY_RUN; then
    warn "Working tree has local changes; dry-run will not execute deployment actions"
  else
    fail "Working tree has local tracked or untracked changes; commit or stash them before deploying"
  fi
fi

TARGET_SHA="$(git rev-parse HEAD)"
[[ "$TARGET_SHA" =~ ^[0-9a-f]{40}$ ]] || fail "HEAD is not a full lowercase 40-hex commit SHA"

CHECKER_PATH="$REPO_ROOT/scripts/device_offroad_check.py"
[[ -f "$CHECKER_PATH" ]] || fail "Missing device offroad checker: scripts/device_offroad_check.py"

REMOTE_DEPLOY_PATH="$(remote_quote "$DEPLOY_PATH")"
REMOTE_DEPLOY_REMOTE="$(remote_quote "$DEPLOY_REMOTE")"
REMOTE_DEPLOY_BRANCH="$(remote_quote "$DEPLOY_BRANCH")"
REMOTE_DEPLOY_REF="$(remote_quote "$DEPLOY_REMOTE/$DEPLOY_BRANCH")"
REMOTE_TARGET_SHA="$(remote_quote "$TARGET_SHA")"
REMOTE_OFFROAD_CHECK_CMD="cd $REMOTE_DEPLOY_PATH && timeout 8s /usr/local/venv/bin/python -"
REMOTE_FETCH_CMD="cd $REMOTE_DEPLOY_PATH && git fetch $REMOTE_DEPLOY_REMOTE $REMOTE_DEPLOY_BRANCH"
REMOTE_FETCH_VERIFY_CMD="cd $REMOTE_DEPLOY_PATH && test \"\$(git rev-parse --verify FETCH_HEAD)\" = $REMOTE_TARGET_SHA && test \"\$(git rev-parse --verify $REMOTE_DEPLOY_REF)\" = $REMOTE_TARGET_SHA"
REMOTE_LFS_FETCH_CMD="cd $REMOTE_DEPLOY_PATH && git -c lfs.concurrenttransfers=1 lfs fetch $REMOTE_DEPLOY_REMOTE $REMOTE_TARGET_SHA"

# shellcheck disable=SC2016 # This is evaluated by the remote shell, not locally.
REMOTE_LFS_CHECK_SNIPPET='lfs_listing="$(git lfs ls-files --long)" || exit 1; printf "%s\\n" "$lfs_listing" | while IFS=" " read -r oid marker path; do test -z "$oid" && continue; test "$oid" != "-" || exit 1; test -n "$path" || exit 1; test "$marker" = "*" || exit 1; test -f "$path" || exit 1; first_line=; IFS= read -r first_line < "$path" || true; case "$first_line" in "version https://git-lfs.github.com/spec/v1"*|"-") exit 1;; esac; done'
REMOTE_LFS_CHECK_FUNCTION="check_lfs_materialized() { $REMOTE_LFS_CHECK_SNIPPET; }"
# shellcheck disable=SC2016 # This is evaluated by the remote shell, not locally.
REMOTE_SUBMODULE_LFS_BODY='set -eu; lfs_listing="$(git lfs ls-files --long)" || exit 1; if test -n "$lfs_listing"; then sub_sha="$(git rev-parse HEAD)"; git -c lfs.concurrenttransfers=1 lfs fetch origin "$sub_sha"; git lfs checkout; fi'
REMOTE_SUBMODULE_LFS_CMD="git submodule foreach --recursive $(remote_quote "$REMOTE_SUBMODULE_LFS_BODY")"
REMOTE_SUBMODULE_VERIFY_BODY="set -eu; git diff --quiet; git diff --cached --quiet; git lfs fsck; $REMOTE_LFS_CHECK_SNIPPET"
REMOTE_SUBMODULE_VERIFY_CMD="git submodule foreach --recursive $(remote_quote "$REMOTE_SUBMODULE_VERIFY_BODY")"
REMOTE_APPLY_CMD="$REMOTE_LFS_CHECK_FUNCTION; set -eu; cd $REMOTE_DEPLOY_PATH; GIT_LFS_SKIP_SMUDGE=1 git reset --hard $REMOTE_TARGET_SHA; git submodule sync --recursive; GIT_LFS_SKIP_SMUDGE=1 git submodule update --init --recursive --jobs 1; git lfs checkout; $REMOTE_SUBMODULE_LFS_CMD; git lfs fsck; check_lfs_materialized; submodule_status=\"\$(git submodule status --recursive)\"; if printf \"%s\\n\" \"\$submodule_status\" | grep -Eq \"^[-+U]\"; then exit 1; fi; $REMOTE_SUBMODULE_VERIFY_CMD; git diff --quiet; git diff --cached --quiet; test \"\$(git rev-parse HEAD)\" = $REMOTE_TARGET_SHA"
REMOTE_REBOOT_CMD="sudo reboot"

if $DRY_RUN; then
  info "Dry-run target SHA: $TARGET_SHA"
  info "Dry-run remote endpoint: <device>"
  info "Phase 1: live offroad gate, then exact-SHA push and remote-branch verification"
  print_offroad_command
  print_cmd env GIT_LFS_SKIP_PUSH=1 git push "--force-with-lease=refs/heads/$DEPLOY_BRANCH" "$PUSH_REMOTE" "$TARGET_SHA:refs/heads/$DEPLOY_BRANCH"
  print_cmd git ls-remote "$PUSH_REMOTE" "refs/heads/$DEPLOY_BRANCH"
  info "Phase 2: retryable device prepare without worktree mutation"
  print_remote_command "$REMOTE_FETCH_CMD"
  print_remote_command "$REMOTE_FETCH_VERIFY_CMD"
  print_remote_command "$REMOTE_LFS_FETCH_CMD"
  info "Phase 3: second live offroad gate, then apply exactly once"
  print_offroad_command
  print_remote_command "$REMOTE_APPLY_CMD"
  info "Phase 4: third live offroad gate, then final reboot or no-reboot staging"
  print_offroad_command
  if $REBOOT; then
    print_remote_command "$REMOTE_REBOOT_CMD"
  else
    info "No reboot requested; report success only after the third gate"
  fi
  exit 0
fi

check_device_offroad "push and device prepare"
info "Pushing exact target SHA to $DEPLOY_BRANCH"
retry_local "Exact-SHA push" env GIT_LFS_SKIP_PUSH=1 git push "--force-with-lease=refs/heads/$DEPLOY_BRANCH" "$PUSH_REMOTE" "$TARGET_SHA:refs/heads/$DEPLOY_BRANCH"
verify_pushed_sha

info "Preparing device without changing its worktree"
retry_remote_prepare "Remote Git fetch" "$REMOTE_FETCH_CMD"
if run_remote "$REMOTE_FETCH_VERIFY_CMD"; then
  :
else
  rc=$?
  if [[ "$rc" == 124 || "$rc" == 255 ]]; then
    fail "Device fetch verification ended ambiguously (exit $rc); inspection required before retrying"
  fi
  fail "Device FETCH_HEAD or remote branch does not resolve to $TARGET_SHA"
fi
retry_remote_prepare "Main exact-SHA LFS fetch" "$REMOTE_LFS_FETCH_CMD"

info "Applying pinned target exactly once"
check_device_offroad "reset"
if run_remote "$REMOTE_APPLY_CMD"; then
  :
else
  rc=$?
  fail "Device apply failed (exit $rc); inspection required before any retry; reboot was not attempted"
fi

check_device_offroad "reboot or successful no-reboot staging"
if $REBOOT; then
  info "Rebooting device"
  if run_remote "$REMOTE_REBOOT_CMD"; then
    :
  else
    rc=$?
    fail "Reboot command failed (exit $rc); inspection required before any retry"
  fi
  info "Deploy complete; reboot requested"
else
  info "Deploy complete; exact pinned staging verified and no reboot requested"
fi
