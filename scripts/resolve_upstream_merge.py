#!/usr/bin/env python3
"""Replay the mechanical half of the upstream merge, leaving only real content conflicts.

`git merge upstream/master` from the migrated layout produces 473 conflicts. 312 of those
are mechanical -- consequences of upstream relocating files, not disagreements about content.
This script resolves exactly those, by rule, so a session can start at the ~161 conflicts
that actually need a human decision.

    git merge upstream/master --no-commit      # will fail with conflicts; that is expected
    python scripts/resolve_upstream_merge.py   # resolves the mechanical ones
    git status                                 # ~161 AA/UU left, the real work

THE RULES (and why each is safe)

  DD  both sides deleted            -> git rm. No disagreement.

  AU  ours at a path upstream moved -> drop ours IF upstream's rename map (git diff -M
      away from, or deleted             between the merge base and upstream/master) shows
                                        the merge-base path was renamed or deleted. The
                                        content survives at upstream's new path, which
                                        arrives as the matching UA.
                                        NEVER drop by basename matching: 15 files here are
                                        fork-only (`*_v1.py`, the fork's versioned modules)
                                        with no upstream counterpart at all.

  UA  upstream added at a new path  -> git add. This is the other half of the AU rename.

  UD  upstream deleted, we touched  -> accept the deletion UNLESS something still in the
                                        tree references the file by name. Two sets survive
                                        that test and must be kept:
                                          openpilot/sunnypilot/system/hardware/c3/*
                                            launch_openpilot.sh execs this chain; upstream
                                            dropped C3 support, the fork still wants it.
                                          openpilot/selfdrive/modeld/models/*.onnx
                                            sunnypilot/models/default_model.py loads these
                                            by filename.

WHAT IS LEFT AFTERWARDS (~161, genuinely needs judgment)

  AA  both added the same path with different content -- concentrated in
      sunnypilot/selfdrive/controls/lib/{speed_limit,smart_cruise_control,nnlc} and
      sunnylink/settings_ui_src/pages.
  UU  both modified -- selfdrive/{ui,locationd,car,selfdrived}, sunnypilot/sunnylink.

  Each is a fork customization meeting an upstream change; deciding which wins is fork-owner
  judgment. Two upstream commits in this range land inside the custom longitudinal stack
  ("longitudinal: remove per-car stopping tune", "longcontrol: remove starting state"), so
  some conflicts are semantic rather than textual -- the file can merge cleanly and still be
  wrong. Budget build + full test validation afterwards, not just a green merge.

  Reassuring datum: sunnypilot/custom/** and tools/drive_lab/** -- the fork's actual value --
  have ZERO conflicts in this merge.
"""
from __future__ import annotations

import subprocess
import sys


def git(*args: str, check: bool = True) -> str:
  r = subprocess.run(["git", *args], capture_output=True, text=True)
  if check and r.returncode and "--ignore-unmatch" not in args:
    print(f"git {' '.join(args[:3])}...: {r.stderr.strip()[:120]}", file=sys.stderr)
  return r.stdout


def staged(code: str) -> list[str]:
  out = []
  for line in git("status", "--porcelain").splitlines():
    if line[:2] == code:
      out.append(line[3:].strip())
  return out


def upstream_rename_map() -> tuple[dict[str, str], set[str]]:
  base = git("merge-base", "HEAD", "upstream/master").strip()
  renames, deletes = {}, set()
  for line in git("diff", "-M", "--find-renames=40%", "--name-status", base, "upstream/master").splitlines():
    parts = line.split("\t")
    if parts[0].startswith("R") and len(parts) == 3:
      renames[parts[1]] = parts[2]
    elif parts[0] == "D":
      deletes.add(parts[1])
  return renames, deletes


# Referenced by surviving code, so upstream's deletion must not be accepted.
UD_KEEP_PREFIXES = (
  "openpilot/sunnypilot/system/hardware/c3/",
  "openpilot/selfdrive/modeld/models/",
)


def main() -> int:
  if not git("status", "--porcelain").strip():
    print("no merge in progress -- run `git merge upstream/master --no-commit` first")
    return 1

  renames, deletes = upstream_rename_map()

  dd = staged("DD")
  if dd:
    git("rm", "-q", "--ignore-unmatch", *dd)
  print(f"DD both-deleted                 : {len(dd)} resolved")

  drop, keep = [], []
  for p in staged("AU"):
    base = p[len("openpilot/"):] if p.startswith("openpilot/") else p
    (drop if (base in renames or base in deletes) else keep).append(p)
  if drop:
    git("rm", "-q", "--ignore-unmatch", "-f", *drop)
  if keep:
    git("add", *keep)
  print(f"AU superseded by upstream rename: {len(drop)} dropped, {len(keep)} fork-only kept")

  ua = staged("UA")
  if ua:
    git("add", *ua)
  print(f"UA upstream's new path          : {len(ua)} accepted")

  ud_keep = [p for p in staged("UD") if p.startswith(UD_KEEP_PREFIXES)]
  ud_drop = [p for p in staged("UD") if not p.startswith(UD_KEEP_PREFIXES)]
  if ud_keep:
    git("add", *ud_keep)
  if ud_drop:
    git("rm", "-q", "--ignore-unmatch", "-f", *ud_drop)
  print(f"UD upstream-deleted             : {len(ud_drop)} accepted, {len(ud_keep)} kept (referenced)")

  left = len(staged("AA")) + len(staged("UU"))
  print(f"\nremaining content conflicts     : {left}  <- the real work")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
