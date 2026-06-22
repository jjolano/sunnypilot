#!/usr/bin/env python3
"""Validate that modified upstream files are listed in docs/touch-points.md."""

from __future__ import annotations

import argparse
import re
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path


def repo_root() -> Path:
    return Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())


ROOT = repo_root()


def run_git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def upstream_exists(upstream_ref: str, path: str) -> bool:
    try:
        subprocess.check_output(
            ["git", "cat-file", "-e", f"{upstream_ref}:{path}"],
            cwd=ROOT,
            stderr=subprocess.DEVNULL,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def parse_touch_points(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    return {m.group(1) for m in re.finditer(r"^\|\s+`([^`]+)`\s+\|", text, re.MULTILINE)}


def changed_files(base: str, head: str, merge_base: bool) -> list[tuple[str, str, str | None]]:
    if merge_base:
        base = run_git("merge-base", base, head)
    out = run_git("diff", "--name-status", "--find-renames", base, head, "--")
    result: list[tuple[str, str, str | None]] = []
    for line in out.splitlines():
        if not line:
            continue
        status, *rest = line.split("\t")
        if status.startswith(("R", "C")) and len(rest) >= 2:
            result.append((status, rest[0], rest[1]))
        elif rest:
            result.append((status, rest[-1], None))
    return result


def missing_touch_points(
    changed_files: Sequence[tuple[str, str, str | None]],
    touched: set[str],
    upstream_exists_predicate: Callable[[str], bool],
    touch_points_path: str,
) -> list[str]:
    """Return sorted, deduplicated upstream paths missing from the touch-points list."""
    missing: list[str] = []
    for _, path, source in changed_files:
        if path == touch_points_path:
            continue
        for candidate in [path, source] if source else [path]:
            if candidate and upstream_exists_predicate(candidate) and candidate not in touched:
                missing.append(candidate)
    return sorted(set(missing))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="Base git ref")
    parser.add_argument("--head", required=True, help="Head git ref")
    parser.add_argument("--merge-base", action="store_true", help="Compare against the merge-base of base and head")
    parser.add_argument("--upstream-ref", default="upstream/master", help="Upstream git ref to compare against")
    parser.add_argument("--touch-points", default="docs/touch-points.md", help="Touch points markdown file")
    args = parser.parse_args()

    touched = parse_touch_points(ROOT / args.touch_points)
    files = changed_files(args.base, args.head, args.merge_base)
    missing = missing_touch_points(
        files,
        touched,
        lambda p: upstream_exists(args.upstream_ref, p),
        args.touch_points,
    )

    if missing:
        print("Missing upstream touch-points entries for:")
        for path in missing:
            print(f"- `{path}`")
        print(f"\nAdd the paths above to `{args.touch_points}` with a short why.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
