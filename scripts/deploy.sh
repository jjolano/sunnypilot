#!/usr/bin/env bash

set -euo pipefail

DRY_RUN=false
REBOOT=true
ALLOW_ONROAD=false
CONFIG_FILE=".deploy-config"
CLI_DEPLOY_HOST=""
CLI_DEPLOY_REMOTE=""
CLI_DEPLOY_PATH=""

usage() {
  cat <<'EOF'
Usage: scripts/deploy.sh [OPTIONS]

Push the deploy branch and reset the device repo to it. Run from the deploy
branch with a clean tree of committed work.

Options:
  --config <path>   Config file path relative to repo root (default: .deploy-config)
  --dry-run         Print the actions without modifying git state or device state
  --host <host>     Override DEPLOY_HOST from config
  --path <path>     Override DEPLOY_PATH from config
  --reboot          Reboot the device after deployment (default)
  --no-reboot       Skip reboot after deployment
  --allow-onroad    Allow deploy while device reports IsOnroad=true (dangerous)
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

retry() {
  local n=1
  local max=3
  local delay=5
  while true; do
    if "$@"; then
      break
    elif [[ $n -lt $max ]]; then
      warn "Command failed, attempt $n/$max. Retrying in ${delay}s..."
      ((n++))
      sleep $delay
    else
      fail "Command failed after $max attempts: $*"
    fi
  done
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
    --allow-onroad)
      ALLOW_ONROAD=true
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

git remote | grep -qx "origin" || fail "Remote 'origin' not found"

if [[ -n "$(git ls-files -u)" ]]; then
  fail "Working tree has unmerged files (conflicts). Resolve them before deploying"
fi

if [[ -f "$(git rev-parse --git-dir)/MERGE_HEAD" ]]; then
  fail "A merge is currently in progress. Finish or abort it before deploying"
fi

if [[ -n "$(git status --porcelain)" ]]; then
  warn "Working tree has local changes. Deployment uses committed $DEPLOY_BRANCH HEAD only"
fi

info "Verifying origin state"
if git fetch origin "$DEPLOY_BRANCH" --quiet 2>/dev/null; then
  if [[ "$(git rev-list HEAD..origin/"$DEPLOY_BRANCH" --count)" -gt 0 ]]; then
    warn "Local branch $DEPLOY_BRANCH is behind origin/$DEPLOY_BRANCH; --force-with-lease will refuse unless that is intended"
  fi
else
  info "Branch $DEPLOY_BRANCH not on origin yet; first push will create it"
fi

REMOTE_DEPLOY_PATH="$(remote_quote "$DEPLOY_PATH")"
REMOTE_DEPLOY_REMOTE="$(remote_quote "$DEPLOY_REMOTE")"
REMOTE_DEPLOY_REF="$(remote_quote "$DEPLOY_REMOTE/$DEPLOY_BRANCH")"
REMOTE_UPDATE_CMD="cd $REMOTE_DEPLOY_PATH && git fetch $REMOTE_DEPLOY_REMOTE && git reset --hard $REMOTE_DEPLOY_REF && git submodule sync --recursive && git submodule update --init --recursive"
REMOTE_REBOOT_CMD="sudo reboot"
REMOTE_ONROAD_CHECK_CMD="cd $REMOTE_DEPLOY_PATH && /usr/local/venv/bin/python - <<'PY'
from openpilot.common.params import Params

is_onroad = Params().get_bool('IsOnroad')
print(f'IsOnroad={is_onroad}')
raise SystemExit(42 if is_onroad else 0)
PY"

if $DRY_RUN; then
  print_cmd env GIT_LFS_SKIP_PUSH=1 git push --force-with-lease origin "$DEPLOY_BRANCH"
  if ! $ALLOW_ONROAD; then
    print_cmd ssh "$DEPLOY_HOST" "$REMOTE_ONROAD_CHECK_CMD"
  fi
  print_cmd ssh "$DEPLOY_HOST" "$REMOTE_UPDATE_CMD"
  if $REBOOT; then
    print_cmd ssh "$DEPLOY_HOST" "$REMOTE_REBOOT_CMD"
  fi
else
  if ! $ALLOW_ONROAD; then
    info "Checking device is offroad before update/reboot"
    set +e
    ssh "$DEPLOY_HOST" "$REMOTE_ONROAD_CHECK_CMD"
    onroad_rc=$?
    set -e
    if [[ $onroad_rc -eq 42 ]]; then
      fail "Device reports IsOnroad=true; refusing to deploy. Stop/offroad first, or pass --allow-onroad if you intentionally accept the risk."
    elif [[ $onroad_rc -ne 0 ]]; then
      fail "Unable to verify device offroad state; refusing to deploy. Use --allow-onroad only if you have confirmed the car is safely offroad."
    fi
  fi

  info "Pushing $DEPLOY_BRANCH to origin"
  retry env GIT_LFS_SKIP_PUSH=1 git push --force-with-lease origin "$DEPLOY_BRANCH"

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
