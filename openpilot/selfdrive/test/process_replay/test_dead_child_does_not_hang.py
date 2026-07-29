"""A replayed process that dies must fail the test, not wedge the worker forever.

Every ReplayContext wait is on an event the child sets. If the child dies, nothing sets it,
and an untimed wait_for_one_event() sleeps in poll() indefinitely — a pytest-xdist worker
was once found stuck 12.6 h this way, and the crash that caused it (a missing generated EKF
aborting locationd) also wrote a 126 MB core dump each time.

These are bounded by construction: if the fix regresses, they hang, so they carry their own
wall-clock assertion rather than relying on an outer timeout.
"""
from __future__ import annotations

import time

import pytest

from openpilot.selfdrive.test.process_replay.process_replay import ReplayContext


class _Cfg:
  proc_name = "test_proc"
  pubs = ["carState"]
  main_pub = None
  main_pub_drained = True
  timeout = 2  # seconds; keep the test fast


def test_replay_context_carries_a_finite_timeout():
  rc = ReplayContext(_Cfg())
  assert rc.timeout == _Cfg.timeout
  assert rc.timeout > 0, "an untimed wait is what hangs the worker"


def test_wait_for_recv_called_gives_up_instead_of_hanging():
  # Nobody ever sets the event: this is exactly the dead-child situation.
  cfg = _Cfg()
  with ReplayContext(cfg) as rc:
    start = time.monotonic()
    with pytest.raises(RuntimeError):
      rc.wait_for_recv_called()
    elapsed = time.monotonic() - start

  assert elapsed < cfg.timeout + 5, f"waited {elapsed:.1f}s — should give up near {cfg.timeout}s"
  assert elapsed >= cfg.timeout - 1, f"returned after {elapsed:.1f}s — timeout not actually applied"
