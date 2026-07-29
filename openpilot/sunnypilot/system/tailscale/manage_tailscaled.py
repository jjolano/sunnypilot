#!/usr/bin/env python3
"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

import json
import os
import select
import shlex
import subprocess
import time

from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog

from openpilot.sunnypilot.system.tailscale import (
  TAILSCALE_SOCKET,
  TAILSCALE_STATE_FILE,
  is_installed,
  tailscale_bin,
  tailscaled_bin,
)
from openpilot.sunnypilot.system.tailscale.installer import (
  download_and_install,
  fetch_latest_version,
)

POLL_INTERVAL = 5  # seconds between status polls
INSTALL_CHECK_INTERVAL = 30  # seconds between install-request checks when not installed
VERSION_CHECK_INTERVAL = 3600  # seconds between latest-version checks


def _decode_json_objects(buffer: str) -> tuple[list[dict], str]:
  decoder = json.JSONDecoder()
  objects = []
  idx = 0
  while idx < len(buffer):
    while idx < len(buffer) and buffer[idx].isspace():
      idx += 1
    if idx >= len(buffer):
      return objects, ""
    try:
      data, end = decoder.raw_decode(buffer, idx)
    except json.JSONDecodeError:
      return objects, buffer[idx:]
    if isinstance(data, dict):
      objects.append(data)
    idx = end
  return objects, ""


def _stale_tailscaled_pids(pgrep_output: str) -> list[str]:
  pids = []
  for line in pgrep_output.splitlines():
    parts = line.strip().split(maxsplit=1)
    if len(parts) != 2:
      continue
    pid, cmdline = parts
    try:
      argv = shlex.split(cmdline)
    except ValueError:
      continue
    if argv and argv[0] == tailscaled_bin() and f"--socket={TAILSCALE_SOCKET}" in argv[1:]:
      pids.append(pid)
  return pids


class TailscaleDaemon:
  """Manages the tailscaled subprocess and handles param-driven commands."""

  def __init__(self):
    self.params = Params()
    self._tailscaled_proc: subprocess.Popen | None = None
    self._login_proc: subprocess.Popen | None = None
    self._last_version_check: float = 0

  # --- tailscaled process management ---

  def _kill_stale_tailscaled(self) -> None:
    """Kill any orphaned tailscaled processes (e.g. from a previous daemon crash)."""
    try:
      result = subprocess.run(
        ["pgrep", "-af", "tailscaled.*--socket="],
        capture_output=True,
        text=True,
        timeout=5,
      )
      if result.returncode == 0 and result.stdout.strip():
        pids = _stale_tailscaled_pids(result.stdout)
        if not pids:
          return
        cloudlog.warning(f"tailscale: killing stale tailscaled processes: {pids}")
        subprocess.run(["sudo", "-n", "kill"] + pids, timeout=5)
        time.sleep(1)
        # Force kill any survivors
        result2 = subprocess.run(
          ["pgrep", "-af", "tailscaled.*--socket="],
          capture_output=True,
          text=True,
          timeout=5,
        )
        if result2.returncode == 0 and result2.stdout.strip():
          survivor_pids = _stale_tailscaled_pids(result2.stdout)
          if survivor_pids:
            subprocess.run(["sudo", "-n", "kill", "-9"] + survivor_pids, timeout=5)
            time.sleep(1)
    except Exception:
      cloudlog.exception("tailscale: failed to kill stale tailscaled")

    # Clean up stale socket
    if os.path.exists(TAILSCALE_SOCKET):
      try:
        os.remove(TAILSCALE_SOCKET)
      except OSError:
        subprocess.run(["sudo", "-n", "rm", "-f", TAILSCALE_SOCKET], timeout=5)

  def _start_tailscaled(self) -> bool:
    """Start the tailscaled daemon via sudo. Returns True if started."""
    if self._tailscaled_proc is not None and self._tailscaled_proc.poll() is None:
      return True  # already running

    # Clean up any orphaned processes from a previous daemon instance
    self._kill_stale_tailscaled()

    try:
      cmd = [
        "sudo",
        "-n",
        tailscaled_bin(),
        f"--state={TAILSCALE_STATE_FILE}",
        f"--socket={TAILSCALE_SOCKET}",
        "--tun=userspace-networking",
      ]
      cloudlog.info(f"tailscale: starting tailscaled: {' '.join(cmd)}")
      self._tailscaled_proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setpgrp,
      )
      # Give it a moment to bind the socket
      time.sleep(2)
      return self._tailscaled_proc.poll() is None
    except Exception:
      cloudlog.exception("tailscale: failed to start tailscaled")
      self._tailscaled_proc = None
      return False

  def _stop_tailscaled(self) -> None:
    """Stop the tailscaled subprocess if running. Uses sudo kill since tailscaled runs as root."""
    if self._tailscaled_proc is None:
      self._kill_stale_tailscaled()
      return

    pid = self._tailscaled_proc.pid
    if self._tailscaled_proc.poll() is None:
      try:
        cloudlog.info(f"tailscale: stopping tailscaled (pid={pid})")
        # Use sudo to kill the root-owned process tree
        subprocess.run(["sudo", "-n", "kill", str(pid)], timeout=5)
        self._tailscaled_proc.wait(timeout=10)
      except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
          subprocess.run(["sudo", "-n", "kill", "-9", str(pid)], timeout=5)
          self._tailscaled_proc.wait(timeout=5)
        except Exception:
          cloudlog.exception("tailscale: failed to kill tailscaled")
      except Exception:
        cloudlog.exception("tailscale: error stopping tailscaled")

    self._tailscaled_proc = None

  def _is_tailscaled_running(self) -> bool:
    return self._tailscaled_proc is not None and self._tailscaled_proc.poll() is None

  # --- tailscale CLI wrappers ---

  def _run_tailscale(self, *args, timeout: int = 30) -> subprocess.CompletedProcess:
    """Run a tailscale CLI command and return the result."""
    cmd = ["sudo", "-n", tailscale_bin(), f"--socket={TAILSCALE_SOCKET}"] + list(args)
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

  def _get_status(self) -> dict | None:
    """Poll tailscale status --json and return parsed dict, or None on failure."""
    try:
      result = self._run_tailscale("status", "--json")
      if result.returncode == 0:
        return json.loads(result.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception):
      cloudlog.exception("tailscale: status poll failed")
    return None

  def _publish_status(self, status: dict) -> None:
    """Write relevant status fields to params."""
    backend_state = status.get("BackendState", "Unknown")

    state_value = backend_state
    self_status = status.get("Self", {})
    tailscale_ips = self_status.get("TailscaleIPs", [])
    if backend_state == "Running" and tailscale_ips:
      state_value = f"Running:{', '.join(tailscale_ips)}"

    if (self.params.get("TailscaleState") or "") != state_value:
      self.params.put("TailscaleState", state_value)

    # Clear auth URL when no longer needed
    if backend_state == "Running":
      auth_url = self.params.get("TailscaleAuthURL") or ""
      if auth_url:
        self.params.remove("TailscaleAuthURL")

  # --- Command handlers ---

  def _handle_login(self) -> None:
    """Handle a login request by running 'tailscale up --json'.

    tailscale up --json outputs pretty-printed JSON objects to stdout. The
    command blocks until authentication completes, so we use Popen and read
    incrementally to capture the auth URL as soon as the first JSON object
    is complete, while letting the process continue for the auth flow.
    """
    self.params.put_bool("TailscaleLoginRequested", False)
    cloudlog.info("tailscale: login requested")

    cmd = [
      "sudo",
      "-n",
      tailscale_bin(),
      f"--socket={TAILSCALE_SOCKET}",
      "up",
      "--json",
      "--accept-dns=false",
      "--accept-routes=false",
    ]

    try:
      proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
      self._login_proc = proc

      auth_url_found = False
      deadline = time.monotonic() + 120
      buf = ""

      while proc.poll() is None and time.monotonic() < deadline:
        ready, _, _ = select.select([proc.stdout], [], [], 2.0)
        if not ready:
          if auth_url_found:
            status = self._get_status()
            if status and status.get("BackendState") == "Running":
              cloudlog.info("tailscale: authenticated while waiting for 'tailscale up' to exit")
              break
          continue

        chunk = os.read(proc.stdout.fileno(), 4096)
        if not chunk:
          break
        text = chunk.decode("utf-8", errors="replace")
        buf += text

        objects, buf = _decode_json_objects(buf)
        login_complete = False
        for data in objects:
          auth_url = data.get("AuthURL", "")
          if auth_url and not auth_url_found:
            self.params.put("TailscaleAuthURL", auth_url)
            auth_url_found = True
            cloudlog.info(f"tailscale: auth URL received, length={len(auth_url)}")

          backend_state = data.get("BackendState", "")
          if backend_state:
            self.params.put("TailscaleState", backend_state)
            if backend_state == "Running":
              cloudlog.info("tailscale: login completed, state=Running")
              login_complete = True
              break
        if login_complete:
          break

      # Clean up the login subprocess
      if proc.poll() is None:
        proc.terminate()
        try:
          proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
          proc.kill()
          proc.wait(timeout=5)

      if not auth_url_found and not self.params.get("TailscaleAuthURL"):
        cloudlog.warning("tailscale: login flow completed without receiving auth URL")

    except Exception:
      cloudlog.exception("tailscale: login failed")
    finally:
      self._login_proc = None

  def _handle_logout(self) -> None:
    """Handle a logout request by running 'tailscale logout'."""
    self.params.put_bool("TailscaleLogoutRequested", False)
    cloudlog.info("tailscale: logout requested")

    try:
      result = self._run_tailscale("logout", timeout=30)
      if result.returncode == 0:
        self.params.put("TailscaleState", "NeedsLogin")
        self.params.remove("TailscaleAuthURL")
        cloudlog.info("tailscale: logged out successfully")
      else:
        err = result.stderr.strip()[:500] if result.stderr else "unknown error"
        self.params.put("TailscaleLastError", err)
        cloudlog.warning(f"tailscale: logout failed: {err}")
    except Exception:
      cloudlog.exception("tailscale: logout failed")

  def _handle_install(self) -> None:
    """Handle an install/update request."""
    self.params.put_bool("TailscaleInstallRequested", False)
    cloudlog.info("tailscale: install requested")

    version = self.params.get("TailscaleLatestVersion") or ""
    if not version:
      version = fetch_latest_version()
      if version:
        self.params.put("TailscaleLatestVersion", version)

    if not version:
      self.params.put("TailscaleInstallState", "error:could not determine version")
      return

    # Stop tailscaled before updating binaries
    was_running = self._is_tailscaled_running()
    if was_running:
      self._stop_tailscaled()

    success = download_and_install(version, self.params)

    # Restart tailscaled if it was running and the install succeeded
    if success and was_running and self.params.get_bool("EnableTailscale"):
      self._start_tailscaled()

  def _check_latest_version(self) -> None:
    """Periodically check for the latest Tailscale version."""
    if not self.params.get_bool("EnableTailscale") and not is_installed():
      return

    now = time.monotonic()
    if now - self._last_version_check < VERSION_CHECK_INTERVAL:
      return

    self._last_version_check = now
    version = fetch_latest_version()
    if version:
      self.params.put("TailscaleLatestVersion", version)

  # --- Main loop ---

  def run(self) -> None:
    """Main daemon loop. Runs forever."""
    cloudlog.info("tailscale: manage_tailscaled starting")

    try:
      while True:
        # Always handle install requests regardless of enable state
        if self.params.get_bool("TailscaleInstallRequested"):
          self._handle_install()
          continue

        # Periodically check for new versions
        self._check_latest_version()

        # If not installed, just wait
        if not is_installed():
          time.sleep(INSTALL_CHECK_INTERVAL)
          continue

        # If disabled, ensure tailscaled is stopped
        if not self.params.get_bool("EnableTailscale"):
          self._stop_tailscaled()
          time.sleep(POLL_INTERVAL)
          continue

        # Enabled and installed: ensure tailscaled is running
        if not self._is_tailscaled_running():
          if not self._start_tailscaled():
            self.params.put("TailscaleLastError", "tailscaled failed to start")
            time.sleep(POLL_INTERVAL)
            continue

        # Handle login/logout requests
        if self.params.get_bool("TailscaleLoginRequested"):
          self._handle_login()
        elif self.params.get_bool("TailscaleLogoutRequested"):
          self._handle_logout()

        # Poll and publish status
        status = self._get_status()
        if status:
          self._publish_status(status)

        time.sleep(POLL_INTERVAL)

    except Exception:
      cloudlog.exception("tailscale: manage_tailscaled crashed")
    finally:
      self._stop_tailscaled()
      self.params.remove("TailscaledPid")


def main():
  daemon = TailscaleDaemon()
  daemon.run()


if __name__ == "__main__":
  main()
