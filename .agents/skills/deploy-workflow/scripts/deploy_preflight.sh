#!/usr/bin/env bash
set -euo pipefail

fail() {
  printf 'ERROR: %s\n' "$1" >&2
  exit 1
}

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
[[ -n "$ROOT" ]] || fail "not inside a git repository"
cd "$ROOT"

[[ -f .deploy-config ]] || fail "missing .deploy-config"
# shellcheck source=/dev/null
source .deploy-config
[[ -n "${DEPLOY_BRANCH:-}" ]] || fail "missing DEPLOY_BRANCH in .deploy-config"

branch="$(git branch --show-current)"
[[ "$branch" == "$DEPLOY_BRANCH" ]] || fail "expected $DEPLOY_BRANCH branch, got ${branch:-detached}"

status="$(git status --short)"
[[ -z "$status" ]] || fail "working tree is dirty or has untracked files"

for path in scripts/deploy.sh AGENTS.md; do
  [[ -e "$path" ]] || fail "missing required workflow file: $path"
done

printf 'branch: %s\n' "$branch"
printf 'commit: %s\n' "$(git log -1 --oneline)"
printf 'status: clean\n'
printf 'workflow files: present\n'
