#!/usr/bin/env bash

set -euo pipefail

DRY_RUN=false
REBOOT=true
REBUILD=false
CONFIG_FILE=".sync-config"
CLI_DEPLOY_HOST=""
CLI_DEPLOY_REMOTE=""
CLI_DEPLOY_PATH=""
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

source "$SCRIPT_DIR/lib/workflow.sh"

usage() {
  cat <<'EOF'
Usage: scripts/deploy.sh [OPTIONS]

Push the integration branch and reset the device repo to that branch. Run this
from the persistent integration worktree.

Options:
  --config <path>   Config file path relative to repo root (default: .sync-config)
  --dry-run         Print the actions without modifying git state or device state
  --host <host>     Override DEPLOY_HOST from config
  --path <path>     Override DEPLOY_PATH from config
  --reboot          Reboot the device after deployment (default)
  --no-reboot       Skip reboot after deployment
  --remote <name>   Override DEPLOY_REMOTE from config
  --rebuild         Rebuild the integration branch before pushing
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
    --rebuild)
      REBUILD=true
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

if [[ -n "$CLI_DEPLOY_HOST" ]]; then
  DEPLOY_HOST="$CLI_DEPLOY_HOST"
fi

if [[ -n "$CLI_DEPLOY_REMOTE" ]]; then
  DEPLOY_REMOTE="$CLI_DEPLOY_REMOTE"
fi

if [[ -n "$CLI_DEPLOY_PATH" ]]; then
  DEPLOY_PATH="$CLI_DEPLOY_PATH"
fi

git remote | grep -qx "origin" || fail "Remote 'origin' not found"
git show-ref --verify --quiet "refs/heads/$INTEGRATION_BRANCH" || fail "Missing local branch: $INTEGRATION_BRANCH"

if [[ -n "$(git ls-files -u)" ]]; then
  fail "Working tree has unmerged files (conflicts). Resolve them before deploying"
fi

if [[ -f "$(git rev-parse --git-dir)/MERGE_HEAD" ]]; then
  fail "A merge is currently in progress. Finish or abort it before deploying"
fi

if [[ -n "$(git status --porcelain)" ]]; then
  if ! git diff --quiet || ! git diff --cached --quiet || [[ -n "$(git ls-files --others --exclude-standard)" ]]; then
    warn "Working tree has local changes. Deployment uses committed $INTEGRATION_BRANCH HEAD only"
  fi
fi

if $REBUILD; then
  info "Rebuilding $INTEGRATION_BRANCH before deployment"
  if $DRY_RUN; then
    print_cmd "$REPO_ROOT/scripts/rebuild-custom.sh" --config "$CONFIG_FILE"
  else
    "$REPO_ROOT/scripts/rebuild-custom.sh" --config "$CONFIG_FILE"
  fi
fi

info "Verifying upstream state"
git fetch origin "$INTEGRATION_BRANCH" --quiet
if [[ "$(git rev-list HEAD..origin/"$INTEGRATION_BRANCH" --count)" -gt 0 ]]; then
  warn "Local branch $INTEGRATION_BRANCH is behind origin/$INTEGRATION_BRANCH. Push may require --force if not using --force-with-lease carefully"
fi

REMOTE_DEPLOY_PATH="$(remote_quote "$DEPLOY_PATH")"
REMOTE_DEPLOY_REMOTE="$(remote_quote "$DEPLOY_REMOTE")"
REMOTE_DEPLOY_REF="$(remote_quote "$DEPLOY_REMOTE/$INTEGRATION_BRANCH")"
REMOTE_UPDATE_CMD="cd $REMOTE_DEPLOY_PATH && git fetch $REMOTE_DEPLOY_REMOTE && git reset --hard $REMOTE_DEPLOY_REF && git submodule sync --recursive && git submodule update --init --recursive"
REMOTE_REBOOT_CMD="sudo reboot"

if $DRY_RUN; then
  print_cmd env GIT_LFS_SKIP_PUSH=1 git push --force-with-lease origin "$INTEGRATION_BRANCH"
  print_cmd ssh "$DEPLOY_HOST" "$REMOTE_UPDATE_CMD"
  if $REBOOT; then
    print_cmd ssh "$DEPLOY_HOST" "$REMOTE_REBOOT_CMD"
  fi
else
  info "Pushing $INTEGRATION_BRANCH to origin"
  retry env GIT_LFS_SKIP_PUSH=1 git push --force-with-lease origin "$INTEGRATION_BRANCH"

  info "Updating device at $DEPLOY_HOST"
  retry ssh "$DEPLOY_HOST" "$REMOTE_UPDATE_CMD"

  if $REBOOT; then
    info "Rebooting device"
    ssh "$DEPLOY_HOST" "$REMOTE_REBOOT_CMD" || warn "Reboot command failed (device might be offline or sudo required)"
  fi
fi

if $REBOOT; then
  info "Deploy complete; reboot requested"
else
  info "Deploy complete"
fi
