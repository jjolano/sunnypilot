#!/usr/bin/env python3
"""Move the fork onto upstream's `openpilot/` package layout.

WHY
    upstream/sunnypilot relocated the whole tree into an `openpilot/` package directory:
    cereal/, common/, selfdrive/, sunnypilot/, system/, third_party/ and most of tools/ now
    live under `openpilot/`. This fork is still flat, and carries an `openpilot/` SHIM --
    `__init__.py` plus symlinks (common -> ../common, selfdrive -> ../selfdrive, ...) -- which
    is what makes `import openpilot.selfdrive...` resolve today.

    Because the fork already imports everything as `openpilot.*`, this migration is a FILE
    MOVE, not an import rewrite. That is what makes it tractable.

    Do it before merging upstream, not after. With both trees in the same shape the 50-commit
    merge becomes an ordinary content merge (~142 files where fork and upstream both changed
    the same file); attempted the other way round, git has to reconcile a whole-tree rename
    against 478 fork-only files that upstream has never seen, and the fork's own files would
    be stranded at old paths while everything around them moved.

MAPPING
    Derived from upstream/master at run time rather than hardcoded, so it cannot drift:

      * upstream has `openpilot/<path>`        -> move there   (upstream is authoritative)
      * upstream has `<path>` at root          -> leave alone
      * fork-only, under a wholesale-moved root
        (cereal/common/selfdrive/sunnypilot/
         system/third_party)                   -> move
      * fork-only under tools/                 -> follow the sibling subdir upstream kept
                                                  (tools/lib moved, tools/car_porting did not);
                                                  a fork-only tool such as drive_lab moves,
                                                  because 128 files import
                                                  `openpilot.tools.drive_lab`
      * anything else (docs, scripts, release,
        .github, dotfiles)                     -> leave alone

USAGE
    python scripts/migrate_to_upstream_layout.py --dry-run     # print the plan, change nothing
    python scripts/migrate_to_upstream_layout.py --execute     # do it (clean tree required)

AFTER
    1. `scons -j$(nproc)` and the test suite -- paths are baked into SConscripts and a few
       tools resolve files relative to __file__.
    2. Commit as a pure move, so the merge sees renames rather than delete+add.
    3. `git merge upstream/master`.

STILL TO DO BEFORE THE MERGE (measured 2026-07-29, after step 2 landed)
    `git merge-tree HEAD upstream/master` reports 358 conflicts:

        93  rename/rename     |  156 of these are upstream INTRA-TREE moves this
        63  add/add           |  script does not replicate -- it only knows how to
        97  rename/delete     |  prefix a path with openpilot/, so where upstream
        96  content           |  also relocated a directory it lands somewhere else
         7  modify/delete     |  and both sides look like independent renames.
         2  submodule / type  |

    Known cases:
        openpilot/system/hardware   -> upstream openpilot/common/hardware
        openpilot/tools/profiling   -> upstream tools/scripts/profiling
        openpilot/selfdrive/debug   -> deleted upstream

    ATTEMPTED AND REVERTED 2026-07-29 -- do not retry this as a pre-pass. Deriving the
    moves from git's own rename detection (`git diff -M --name-status` between the merge
    base and upstream/master) yields 171 destination mismatches, of which only 79 are
    applicable; 84 have their destination already occupied in the fork. Following them
    breaks the tree, because upstream did not simply *relocate* these directories, it
    SPLIT them semantically:

        upstream openpilot/common/hardware/   24 files
        upstream openpilot/system/hardware/    7 files  (fan_controller, hardwared,
                                                         power_monitoring, tests/, agnos.json)

    Following the renames mechanically leaves `common/hardware/tici` without an
    `__init__.py` and orphans `iwlist` in the old package -- the import graph no longer
    closes. Deciding which half each file belongs to IS merge work, not move work, so do
    it inside the merge with the conflicts visible rather than guessing beforehand.

    Budget the merge as its own session regardless: two upstream commits land inside the
    custom longitudinal stack ("longitudinal: remove per-car stopping tune",
    "longcontrol: remove starting state"), so some conflicts are semantic, not textual.

NOTE
    tools/bodyteleop and tools/profiling are moved for consistency but upstream has deleted
    both and nothing in this fork imports them -- they are deletion candidates, decide
    separately rather than carrying them into the new layout by accident.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from collections import Counter

MOVED_ROOTS = {"cereal", "common", "selfdrive", "sunnypilot", "system", "third_party"}
SHIM_LINKS = ("openpilot/common", "openpilot/selfdrive", "openpilot/sunnypilot",
              "openpilot/system", "openpilot/tools")
UPSTREAM = "upstream/master"


def git(*args: str) -> str:
  return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout


def tracked(ref: str) -> list[str]:
  return [l for l in git("ls-tree", "-r", "--name-only", ref).splitlines() if l.strip()]


def build_plan() -> tuple[list[tuple[str, str]], list[str]]:
  head = tracked("HEAD")
  up = set(tracked(UPSTREAM))
  up_dirs: set[str] = set()
  for p in up:
    parts = p.split("/")
    for i in range(1, len(parts)):
      up_dirs.add("/".join(parts[:i]))

  def target(p: str) -> str | None:
    if f"openpilot/{p}" in up:
      return f"openpilot/{p}"
    if p in up:
      return p
    top = p.split("/")[0]
    if top in MOVED_ROOTS:
      return f"openpilot/{p}"
    if top == "openpilot":
      return None                                   # the shim; drop it
    if top == "tools":
      parts = p.split("/")
      if len(parts) < 2:
        return p
      sub = parts[1]
      if f"openpilot/tools/{sub}" in up_dirs:
        return f"openpilot/{p}"
      if f"tools/{sub}" in up_dirs or f"tools/{sub}" in up:
        return p                                    # upstream kept this one at root
      return f"openpilot/{p}"                       # fork-only tool
    return p

  moves, drops = [], []
  for p in head:
    t = target(p)
    if t is None:
      drops.append(p)
    elif t != p:
      moves.append((p, t))
  return moves, drops


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  mode = ap.add_mutually_exclusive_group(required=True)
  mode.add_argument("--dry-run", action="store_true")
  mode.add_argument("--execute", action="store_true")
  args = ap.parse_args()

  if git("status", "--porcelain").strip():
    print("working tree is dirty; commit or stash first", file=sys.stderr)
    return 1

  moves, drops = build_plan()
  print(f"shim symlinks to remove : {len(drops)}")
  for d in drops:
    print(f"    {d}")
  print(f"files to move           : {len(moves)}")
  by_root = Counter(src.split("/")[0] for src, _ in moves)
  for root, n in by_root.most_common():
    print(f"    {n:>5}  {root}/")
  untouched = len(tracked("HEAD")) - len(moves) - len(drops)
  print(f"files left in place     : {untouched}")

  if args.dry_run:
    print("\n(dry run -- nothing changed)")
    return 0

  # Symlinks first: moving a directory into openpilot/ while openpilot/<name> is still a
  # symlink pointing back at it would follow the link and recurse.
  for d in drops:
    git("rm", "-q", d)

  # Whole directories where every file moves -- one rename each, cheap and history-clean.
  moved_srcs: set[str] = set()
  for root in sorted(MOVED_ROOTS):
    in_root = [m for m in moves if m[0].split("/")[0] == root]
    if in_root and all(dst == f"openpilot/{src}" for src, dst in in_root):
      git("mv", root, f"openpilot/{root}")
      moved_srcs.update(src for src, _ in in_root)
      print(f"moved {root}/ -> openpilot/{root}/")

  # Everything else (the tools split) file by file.
  remaining = [(s, d) for s, d in moves if s not in moved_srcs]
  dirs_done: set[str] = set()
  for src, dst in remaining:
    parent = dst.rsplit("/", 1)[0]
    if parent not in dirs_done:
      subprocess.run(["mkdir", "-p", parent], check=True)
      dirs_done.add(parent)
    git("mv", src, dst)
  print(f"moved {len(remaining)} remaining file(s)")
  print("\ndone -- now run scons + the test suite, then commit as a pure move.")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
