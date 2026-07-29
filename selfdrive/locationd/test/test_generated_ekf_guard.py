"""A missing generated EKF library must raise, not abort the process.

rednose's ekf_load.cc does `assert(handle)` on the dlopen result, so an unbuilt model
library SIGABRTs the whole process. 30 of the 37 core dumps found in this tree on
2026-07-29 carried exactly that assert. `generated/` is gitignored build output, so a
fresh checkout hits it every time until scons has run.
"""
from __future__ import annotations

import subprocess
import sys

import pytest

from openpilot.selfdrive.locationd.models.constants import GENERATED_DIR, require_generated_ekf


def test_missing_library_raises_with_the_build_command(tmp_path):
  with pytest.raises(FileNotFoundError) as excinfo:
    require_generated_ekf(str(tmp_path), "pose")
  message = str(excinfo.value)
  assert "libpose.so" in message
  assert "scons selfdrive/locationd" in message, "the error must say how to fix it"


def test_present_library_passes():
  # The real generated dir in a built tree; skip rather than fail an unbuilt checkout.
  import os
  if not os.path.exists(os.path.join(GENERATED_DIR, "libpose.so")):
    pytest.skip("generated EKF not built in this tree")
  require_generated_ekf(GENERATED_DIR, "pose")


def test_construction_raises_instead_of_aborting(tmp_path):
  """The regression that matters: constructing the filter against an empty generated dir
  must exit with a Python traceback, NOT die on SIGABRT (-6)."""
  code = (
    "from openpilot.selfdrive.locationd.models.pose_kf import PoseKalman\n"
    f"PoseKalman({str(tmp_path)!r}, 0.8)\n"
  )
  proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
  assert proc.returncode != -6, "process aborted (SIGABRT) instead of raising"
  assert proc.returncode != 0, "should have failed"
  assert "FileNotFoundError" in proc.stderr
  assert "scons selfdrive/locationd" in proc.stderr
