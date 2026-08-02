#!/usr/bin/env python3
"""Fail-closed live deviceState.started check for deploy.sh."""

from __future__ import annotations

import importlib
import time


def _messaging_module():
  try:
    return importlib.import_module("openpilot.cereal.messaging")
  except ModuleNotFoundError as exc:
    if exc.name != "openpilot.cereal":
      raise
    return importlib.import_module("cereal.messaging")


def check() -> int:
  try:
    messaging = _messaging_module()
    sock = messaging.sub_sock("deviceState", conflate=True, timeout=5000)
  except Exception:
    return 44

  try:
    raw = sock.receive()
  except TimeoutError:
    return 43
  except Exception:
    return 44

  if raw is None:
    return 43

  try:
    msg = messaging.log_from_bytes(raw)
    if msg.which() != "deviceState":
      return 43
    valid = msg.valid
    log_mono_time = msg.logMonoTime
    started = msg.deviceState.started
    if not isinstance(valid, bool) or isinstance(log_mono_time, bool) or not isinstance(log_mono_time, int) or not isinstance(started, bool):
      return 44
    if not valid:
      return 43
    age_ns = time.monotonic_ns() - log_mono_time
    if age_ns < 0 or age_ns > 3_000_000_000:
      return 43
    return 42 if started else 0
  except Exception:
    return 44


if __name__ == "__main__":
  raise SystemExit(check())
