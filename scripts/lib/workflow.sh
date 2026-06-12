#!/usr/bin/env bash

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

retry() {
  local n=1
  local max=3
  local delay=5
  while true; do
    if "$@"; then
      break
    else
      if [[ $n -lt $max ]]; then
        warn "Command failed, attempt $n/$max. Retrying in ${delay}s..."
        ((n++))
        sleep $delay
      else
        fail "Command failed after $max attempts: $*"
      fi
    fi
  done
}

run() {
  if ${DRY_RUN:-false}; then
    print_cmd "$@"
  else
    "$@"
  fi
}

load_sync_config() {
  local repo_root="$1"
  local config_file="$2"

  if [[ "$config_file" == /* ]]; then
    CONFIG_PATH="$config_file"
  else
    CONFIG_PATH="$repo_root/$config_file"
  fi

  [[ -f "$CONFIG_PATH" ]] || fail "Config file not found: $config_file"
  # shellcheck source=/dev/null
  source "$CONFIG_PATH"
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

ensure_branch_available_for_temp_worktree() {
  local branch="$1"
  local existing_worktree

  existing_worktree="$(find_branch_worktree "$branch" || true)"
  if [[ -n "$existing_worktree" ]]; then
    fail "Branch '$branch' is already checked out in worktree: $existing_worktree"
  fi
}
