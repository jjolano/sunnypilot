#!/usr/bin/env bash
set -euo pipefail

fail() {
  printf 'ERROR: %s\n' "$1" >&2
  exit 1
}

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
[[ -n "$ROOT" ]] || fail "not inside a git repository"
cd "$ROOT"

branch="$(git branch --show-current)"
[[ "$branch" == "custom" ]] || fail "expected custom branch, got ${branch:-detached}"

status="$(git status --short)"
[[ -z "$status" ]] || fail "custom worktree is dirty or has untracked files"

for path in scripts/rebuild-custom.sh scripts/deploy.sh scripts/propagate-retained.sh .sync-config AGENTS.md; do
  [[ -e "$path" ]] || fail "missing required workflow file: $path"
done

printf 'branch: %s\n' "$branch"
printf 'commit: %s\n' "$(git log -1 --oneline)"
printf 'status: clean\n'
printf 'workflow files: present\n'
