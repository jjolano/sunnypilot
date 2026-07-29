#!/usr/bin/env python3
"""Resolve the mechanically-decidable part of the upstream merge.

`git merge upstream/master` from the migrated layout produces 473 conflicts. 434 of those
are decidable by rule; this resolves them, leaving 39 files (137 hunks) where fork feature
work meets an upstream change and a human has to choose.

    git merge upstream/master --no-commit      # fails with 473 conflicts; expected
    ./.venv/bin/python scripts/resolve_upstream_merge.py
    git status                                 # 39 files left, the real work

Use the venv python directly, not `uv run` -- pyproject.toml is itself conflicted at that
point and uv refuses to parse it.

THE RULES, in the order applied

  1. DD  both sides deleted           -> git rm. No disagreement.

  2. AU  ours at a path upstream      -> drop ours IF upstream's rename map (git diff -M
         moved away from / deleted       between merge base and upstream/master) says that
                                         path was renamed or deleted. The content arrives
                                         as the matching UA.
                                         NEVER match on basename: 15 files here are
                                         fork-only (`*_v1.py`) with no upstream counterpart
                                         and basename matching would delete them.

  3. UA  upstream's new path          -> git add. Other half of the AU rename.

  4. UD  upstream deleted, we touched -> accept UNLESS surviving code references the file.
                                         Two sets survive: the C3 launch chain that
                                         launch_openpilot.sh execs (upstream dropped C3
                                         support, this fork wants it) and the .onnx models
                                         sunnypilot/models/default_model.py loads by name.

  5. AA/UU where the fork never       -> take theirs.
     touched the file
                                         "Never touched" means NO commit between the merge
                                         base and the pre-merge head touches it, ignoring
                                         the layout migration itself.

  *** DO NOT use docs/touch-points.md for this. It is INCOMPLETE. ***
  It lists 10 of the conflicted files; git history shows 30 MORE carrying real fork feature
  commits (move speed-limit stack, curve evidence, CurveMemory deletion, slew-scale study,
  direction-gain estimator...). Trusting the registry here would have silently reverted a
  large slice of the fork's work. Ask git, not the doc.

WHAT IS LEFT (39 files / 137 semantic hunks) -- resolve by hand

  Fork-modified sunnypilot subsystems where upstream also moved:
    selfdrive/controls/lib/{speed_limit,smart_cruise_control,nnlc}/**
    selfdrive/controls/lib/latcontrol_torque_ext*, dec/dec.py
    selfdrive/locationd/torqued_ext.py, sunnylink/settings_ui_src/pages/*.yaml
  Upstream files carrying fork hooks (docs/touch-points.md is right about these):
    selfdrive/controls/{controlsd,plannerd}.py, lib/longitudinal_planner.py,
    lib/longitudinal_mpc_lib/long_mpc.py, selfdrive/locationd/torqued.py,
    system/manager/process_config.py, sunnypilot/selfdrive/controls/controlsd_ext.py

  Two upstream commits in range change the longitudinal stack semantically
  ("longitudinal: remove per-car stopping tune", "longcontrol: remove starting state"), so a
  file can merge cleanly and still be wrong. Validate with a build and the full suite, not a
  green merge.

  Reassuring: sunnypilot/custom/** and tools/drive_lab/** have ZERO conflicts.

DECISIONS ALREADY MADE (root config, reapply if you restart the merge)
  SConstruct        -> theirs (explicit #msgq_repo/#opendbc_repo/#rednose_repo include and
                       lib paths; the root symlinks are going away; drops lateral_mpc_lib,
                       whose deletion rule 4 already accepted)
  pyproject.toml    -> theirs (testpaths already scopes collection to openpilot/, so our
                       --ignore= flags were redundant)
  conftest.py       -> union: upstream's tools/sim ignore + pytest_sessionstart, plus our
                       RLIMIT_CORE limiter; drop the selfdrive/debug glob (dir moved)
  launch_openpilot  -> OURS (the fork's C3 branch; upstream dropped C3 support)
  launch_chffrplus  -> theirs (hardware moved to common/hardware)
  opendbc_repo      -> theirs (d6b9c1a). This is what retires the cereal shim: that revision
                       loads car.capnp itself with no `from cereal import car` fallback.
  cereal/__init__   -> upstream's body, plus `from opendbc.car.structs import car` re-export,
                       because ~80 fork files still do `from openpilot.cereal import car`.
  version.h         -> theirs (upstream's release bump)
  default_model.py,
  model_hash        -> OURS (fork runs split vision+policy; upstream moved to supercombo)
  smart_cruise_control/__init__.py -> OURS (fork owns MIN_V in its own curve_evidence
                       constants, same value, and its vision_controller imports from there)
"""
from __future__ import annotations

import pathlib
import re
import subprocess
import sys

CONFLICT = re.compile(r'<<<<<<< [^\n]*\n(.*?)=======\n(.*?)>>>>>>> [^\n]*\n', re.S)
UD_KEEP_PREFIXES = (
  "openpilot/sunnypilot/system/hardware/c3/",
  "openpilot/selfdrive/modeld/models/",
)


def git(*args: str) -> str:
  return subprocess.run(["git", *args], capture_output=True, text=True).stdout


def staged(code: str) -> list[str]:
  return [l[3:].strip() for l in git("status", "--porcelain").splitlines() if l[:2] == code]


def take_theirs(paths: list[str]) -> int:
  n = 0
  for f in paths:
    p = pathlib.Path(f)
    try:
      t = p.read_text()
    except (UnicodeDecodeError, OSError):
      continue
    t2, k = CONFLICT.subn(lambda m: m.group(2), t)
    if k:
      p.write_text(t2)
      n += k
  return n


def main() -> int:
  if not git("status", "--porcelain").strip():
    print("no merge in progress -- run `git merge upstream/master --no-commit` first")
    return 1

  base = git("merge-base", "HEAD", "upstream/master").strip()
  renames, deletes = {}, set()
  for line in git("diff", "-M", "--find-renames=40%", "--name-status", base, "upstream/master").splitlines():
    parts = line.split("\t")
    if parts[0].startswith("R") and len(parts) == 3:
      renames[parts[1]] = parts[2]
    elif parts[0] == "D":
      deletes.add(parts[1])

  dd = staged("DD")
  if dd:
    git("rm", "-q", "--ignore-unmatch", *dd)
  print(f"1. DD both-deleted              : {len(dd)}")

  drop, keep = [], []
  for p in staged("AU"):
    old = p[len("openpilot/"):] if p.startswith("openpilot/") else p
    (drop if (old in renames or old in deletes) else keep).append(p)
  if drop:
    git("rm", "-q", "--ignore-unmatch", "-f", *drop)
  if keep:
    git("add", *keep)
  print(f"2. AU renamed away by upstream  : {len(drop)} dropped, {len(keep)} fork-only kept")

  ua = staged("UA")
  if ua:
    git("add", *ua)
  print(f"3. UA upstream's new path       : {len(ua)}")

  ud = staged("UD")
  ud_keep = [p for p in ud if p.startswith(UD_KEEP_PREFIXES)]
  ud_drop = [p for p in ud if not p.startswith(UD_KEEP_PREFIXES)]
  if ud_keep:
    git("add", *ud_keep)
  if ud_drop:
    git("rm", "-q", "--ignore-unmatch", "-f", *ud_drop)
  print(f"4. UD upstream-deleted          : {len(ud_drop)} accepted, {len(ud_keep)} kept (referenced)")

  # 5. content conflicts in files the fork never touched. Ask git, not touch-points.md.
  untouched = []
  for p in staged("AA") + staged("UU"):
    old = p[len("openpilot/"):] if p.startswith("openpilot/") else p
    log = git("log", "--oneline", f"{base}..HEAD^", "--", p, old).splitlines()
    real = [c for c in log if "migrate the fork" not in c and "layout" not in c.lower()]
    if not real:
      untouched.append(p)
  hunks = take_theirs(untouched)
  if untouched:
    git("add", *untouched)
  print(f"5. AA/UU fork never touched     : {len(untouched)} files, {hunks} hunks -> theirs")

  left = staged("AA") + staged("UU")
  print(f"\nleft for a human                : {len(left)} files")
  for f in left:
    print(f"    {f}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
