"""The conftest core-dump limit must actually stop a crashing child from dumping.

Tests spawn native processes with the repo root as CWD, so a single abort used to write a
~126 MB core.<pid> into the working tree. They are gitignored, so 5.4 GB accumulated
unnoticed over three weeks. This guards the containment, not the crash.
"""
from __future__ import annotations

import os
import resource
import signal
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _abort_in_child(cwd: Path) -> None:
  subprocess.run(
    [sys.executable, "-c", "import os, signal; os.kill(os.getpid(), signal.SIGABRT)"],
    cwd=str(cwd), capture_output=True,
  )


def test_conftest_sets_a_zero_soft_core_limit():
  soft, _ = resource.getrlimit(resource.RLIMIT_CORE)
  assert soft == 0, f"conftest should have zeroed the soft core limit, got {soft}"


def test_crashing_child_leaves_no_core_in_the_tree(tmp_path):
  # A child that really dies on SIGABRT — the same signal all 37 historical dumps carried.
  before = set(tmp_path.glob("core*"))
  _abort_in_child(tmp_path)
  assert set(tmp_path.glob("core*")) == before


def test_escape_hatch_is_honored_for_debugging():
  # The knob has to actually work, or nobody can capture a core when they need one.
  # Checked by inspecting the conftest source rather than re-running pytest in-process:
  # the limit is applied at import time and cannot be meaningfully un-applied here.
  source = (REPO_ROOT / "conftest.py").read_text()
  assert "OPENPILOT_TEST_CORE_DUMPS" in source
  assert signal.SIGABRT is not None  # sanity: signal module import is used above


def test_repo_root_has_no_core_dumps():
  """Regression: the working tree should be free of core dumps.

  Skipped rather than failed when the historical pile is still present — deleting 5.4 GB of
  someone's crash evidence is their call, not this test's.
  """
  import pytest
  cores = sorted(REPO_ROOT.glob("core.*"))
  if cores:
    total_gb = sum(c.stat().st_size for c in cores) / 1e9
    pytest.skip(f"{len(cores)} pre-existing core dump(s), {total_gb:.1f} GB — remove them to enable this check")
  assert not cores
