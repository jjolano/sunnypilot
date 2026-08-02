from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
DEPLOY = ROOT / "scripts/deploy.sh"
CHECKER = ROOT / "scripts/device_offroad_check.py"
SHA = "a" * 40
HOST = "comma@configured-host-must-not-leak"


FAKE_GIT = r'''#!/usr/bin/env bash
set -u
printf 'git' >> "$FAKE_LOG"
printf ' %s' "$@" >> "$FAKE_LOG"
printf '\n' >> "$FAKE_LOG"

if [[ "$1" == rev-parse && "$2" == --show-toplevel ]]; then
  printf '%s\n' "$FAKE_REPO"
elif [[ "$1" == branch && "$2" == --show-current ]]; then
  printf 'master\n'
elif [[ "$1" == remote ]]; then
  printf 'origin\n'
elif [[ "$1" == ls-files ]]; then
  :
elif [[ "$1" == rev-parse && "$2" == --git-dir ]]; then
  printf '%s/.git\n' "$FAKE_REPO"
elif [[ "$1" == status ]]; then
  if [[ "${FAKE_DIRTY:-0}" == 1 ]]; then
    printf ' M tracked.py\n?? untracked.py\n'
  fi
elif [[ "$1" == rev-parse && "$2" == HEAD ]]; then
  printf '%s\n' "$FAKE_SHA"
elif [[ "$1" == push ]]; then
  count=0
  [[ -f "$FAKE_PUSH_COUNT" ]] && read -r count < "$FAKE_PUSH_COUNT"
  count=$((count + 1))
  printf '%s\n' "$count" > "$FAKE_PUSH_COUNT"
  ((count > FAKE_PUSH_FAILURES)) || exit 1
elif [[ "$1" == ls-remote ]]; then
  printf '%s refs/heads/master\n' "$FAKE_REMOTE_SHA"
fi
'''

FAKE_SSH = r'''#!/usr/bin/env bash
set -u
printf 'ssh' >> "$FAKE_LOG"
printf ' %s' "$@" >> "$FAKE_LOG"
printf '\n' >> "$FAKE_LOG"
count=0
[[ -f "$FAKE_SSH_COUNT" ]] && read -r count < "$FAKE_SSH_COUNT"
count=$((count + 1))
printf '%s\n' "$count" > "$FAKE_SSH_COUNT"
read -r -a results <<< "${FAKE_SSH_RESULTS:-}"
if [[ "${FAKE_FETCH_MISMATCH:-0}" == 1 && "$*" == *FETCH_HEAD* ]]; then
  exit 1
fi
exit "${results[$((count - 1))]:-0}"
'''

FAKE_TIMEOUT = r'''#!/usr/bin/env bash
set -u
printf 'timeout' >> "$FAKE_LOG"
printf ' %s' "$@" >> "$FAKE_LOG"
printf '\n' >> "$FAKE_LOG"
[[ "$1" == 20s ]] && shift
exec "$@"
'''


class DeployFixture:
  def __init__(self, tmp_path: Path) -> None:
    self.repo = tmp_path / "repo"
    self.repo.mkdir()
    (self.repo / ".git").mkdir()
    (self.repo / "scripts").mkdir()
    (self.repo / "scripts" / "device_offroad_check.py").write_text(CHECKER.read_text())
    (self.repo / ".deploy-config").write_text(
      f'DEPLOY_BRANCH="master"\nDEPLOY_HOST="{HOST}"\nDEPLOY_REMOTE="jjolano"\nDEPLOY_PATH="/data/openpilot"\n'
    )

    self.bin = tmp_path / "bin"
    self.bin.mkdir()
    for name, body in (("git", FAKE_GIT), ("ssh", FAKE_SSH), ("timeout", FAKE_TIMEOUT)):
      path = self.bin / name
      path.write_text(body)
      path.chmod(0o755)
    self.log = tmp_path / "commands.log"
    self.push_count = tmp_path / "push.count"
    self.ssh_count = tmp_path / "ssh.count"

  def run(
    self,
    *args: str,
    ssh_results: list[int] | None = None,
    push_failures: int = 0,
    remote_sha: str = SHA,
    fetch_mismatch: bool = False,
    dirty: bool = False,
  ) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update({
      "PATH": f"{self.bin}:{env['PATH']}",
      "FAKE_LOG": str(self.log),
      "FAKE_REPO": str(self.repo),
      "FAKE_SHA": SHA,
      "FAKE_REMOTE_SHA": remote_sha,
      "FAKE_PUSH_FAILURES": str(push_failures),
      "FAKE_PUSH_COUNT": str(self.push_count),
      "FAKE_SSH_COUNT": str(self.ssh_count),
      "FAKE_SSH_RESULTS": " ".join(str(rc) for rc in (ssh_results or [])),
      "FAKE_FETCH_MISMATCH": "1" if fetch_mismatch else "0",
      "FAKE_DIRTY": "1" if dirty else "0",
      "DEPLOY_RETRY_DELAY_SECONDS": "0",
    })
    return subprocess.run(
      ["bash", str(DEPLOY), *args],
      cwd=self.repo,
      env=env,
      text=True,
      capture_output=True,
      check=False,
    )

  def commands(self) -> list[str]:
    if not self.log.exists():
      return []
    return self.log.read_text().splitlines()


def test_allow_onroad_is_unknown_option(tmp_path: Path) -> None:
  fixture = DeployFixture(tmp_path)
  result = fixture.run("--allow-onroad")
  assert result.returncode != 0
  assert "Unknown option: --allow-onroad" in result.stderr
  assert fixture.commands() == []


def test_dirty_real_deploy_stops_before_checker_and_mutation(tmp_path: Path) -> None:
  fixture = DeployFixture(tmp_path)
  result = fixture.run("--no-reboot", dirty=True, ssh_results=[0] * 8)
  assert result.returncode != 0
  assert "tracked or untracked" in result.stderr
  commands = fixture.commands()
  assert not any(any(token in command for token in ("ssh", "timeout", "git push", "git fetch", "ls-remote")) for command in commands)


def test_dirty_dry_run_warns_but_remains_side_effect_free(tmp_path: Path) -> None:
  fixture = DeployFixture(tmp_path)
  result = fixture.run("--dry-run", "--no-reboot", dirty=True)
  assert result.returncode == 0, result.stderr
  assert "local changes" in result.stderr
  commands = fixture.commands()
  assert not any(any(token in command for token in ("ssh", "timeout", "git push", "git fetch", "ls-remote")) for command in commands)


@pytest.mark.parametrize("status", [42, 43, 44, 124, 255, 7])
def test_nonzero_or_unexpected_offroad_check_prevents_mutation(tmp_path: Path, status: int) -> None:
  fixture = DeployFixture(tmp_path)
  result = fixture.run("--no-reboot", ssh_results=[status])
  assert result.returncode != 0
  assert not any("git push" in command for command in fixture.commands())
  assert not any("git reset --hard" in command for command in fixture.commands())


def test_offroad_allows_complete_pinned_deploy(tmp_path: Path) -> None:
  fixture = DeployFixture(tmp_path)
  result = fixture.run("--no-reboot", ssh_results=[0] * 7)
  assert result.returncode == 0, result.stderr
  commands = fixture.commands()
  assert any("git push" in command and SHA in command for command in commands)
  assert any("git reset --hard" in command and SHA in command for command in commands)
  assert not any("sudo reboot" in command for command in commands)


def test_fetch_retries_twice_then_succeeds(tmp_path: Path) -> None:
  fixture = DeployFixture(tmp_path)
  result = fixture.run("--no-reboot", ssh_results=[0, 1, 1, 0, 0, 0, 0, 0, 0])
  assert result.returncode == 0, result.stderr
  fetches = [command for command in fixture.commands() if "git fetch" in command]
  assert len(fetches) == 3
  assert sum("lfs fetch" in command and "'jjolano'" in command for command in fixture.commands()) == 1


def test_remote_fetch_retries_ssh_255_twice_then_succeeds(tmp_path: Path) -> None:
  fixture = DeployFixture(tmp_path)
  result = fixture.run("--no-reboot", ssh_results=[0, 255, 255, 0, 0, 0, 0, 0, 0])
  assert result.returncode == 0, result.stderr
  assert sum("git fetch" in command for command in fixture.commands()) == 3


def test_fetch_sha_mismatch_blocks_reset(tmp_path: Path) -> None:
  fixture = DeployFixture(tmp_path)
  result = fixture.run("--no-reboot", ssh_results=[0, 0], fetch_mismatch=True)
  assert result.returncode != 0
  commands = fixture.commands()
  assert any("FETCH_HEAD" in command for command in commands)
  assert not any("git reset --hard" in command for command in commands)


def test_one_shot_apply_failure_blocks_following_phases(tmp_path: Path) -> None:
  fixture = DeployFixture(tmp_path)
  result = fixture.run("--no-reboot", ssh_results=[0, 0, 0, 0, 0, 1])
  assert result.returncode != 0
  commands = fixture.commands()
  assert sum("git reset --hard" in command for command in commands) == 1
  assert not any("sudo reboot" in command for command in commands)
  assert "inspection" in result.stderr


def test_one_shot_apply_failure_blocks_reboot_and_contains_validation(tmp_path: Path) -> None:
  fixture = DeployFixture(tmp_path)
  result = fixture.run(ssh_results=[0, 0, 0, 0, 0, 1])
  assert result.returncode != 0
  commands = fixture.commands()
  apply = next(command for command in commands if "git reset --hard" in command)
  assert "git submodule update --init --recursive --jobs 1" in apply
  assert "git lfs fsck" in apply
  assert "git lfs ls-files --long" in apply
  assert "version https://git-lfs.github.com/spec/v1" in apply
  assert "^[-+U]" in apply
  assert not any("sudo reboot" in command for command in commands)


def test_dry_run_is_pinned_redacted_and_network_free(tmp_path: Path) -> None:
  fixture = DeployFixture(tmp_path)
  result = fixture.run("--dry-run", "--no-reboot")
  output = result.stdout + result.stderr
  assert result.returncode == 0, result.stderr
  assert SHA in output
  assert "<device>" in output
  assert "GIT_LFS_SKIP_SMUDGE=1" in output
  assert "--jobs 1" in output
  assert "lfs.concurrenttransfers=1" in output
  assert "git reset --hard" in output and SHA in output
  assert "git reset --hard origin/master" not in output
  assert "--untracked-files" not in output
  assert HOST not in output
  assert "IsOnroad" not in output
  assert "--allow-onroad" not in output
  assert "comma.service" not in output
  commands = fixture.commands()
  assert not any(any(token in command for token in ("git push", "git fetch", "ls-remote", "ssh", "timeout")) for command in commands)


def _checker_module(started: bool, outcome: str = "normal") -> str:
  return f'''import time

OUTCOME = {outcome!r}

class _Socket:
  def receive(self):
    if OUTCOME == "none":
      return None
    if OUTCOME == "receive-timeout":
      raise TimeoutError
    return b"device-state"

class _DeviceState:
  started = {started!r}

class _Message:
  valid = True
  logMonoTime = time.monotonic_ns()
  deviceState = _DeviceState()

  def which(self):
    return "other" if OUTCOME == "wrong-type" else "deviceState"

if OUTCOME == "invalid":
  _Message.valid = False
elif OUTCOME == "schema":
  _Message.valid = "not-a-bool"
elif OUTCOME == "stale":
  _Message.logMonoTime -= 4_000_000_000
elif OUTCOME == "future":
  _Message.logMonoTime += 1_000_000_000
elif OUTCOME == "missing":
  del _Message.deviceState

def sub_sock(service, *, conflate, timeout):
  assert service == "deviceState"
  assert conflate is True
  assert timeout == 5000
  return _Socket()

def log_from_bytes(raw):
  if OUTCOME == "decode":
    raise ValueError("malformed message")
  assert raw == b"device-state"
  return _Message()
'''


@pytest.mark.parametrize(("layout", "started", "expected"), [("current", False, 0), ("old", True, 42)])
def test_checker_supports_current_and_old_messaging_layouts(tmp_path: Path, layout: str, started: bool, expected: int) -> None:
  if layout == "current":
    package = tmp_path / "openpilot" / "cereal"
    package.mkdir(parents=True)
    (tmp_path / "openpilot" / "__init__.py").write_text("")
    (package / "__init__.py").write_text("")
    (package / "messaging.py").write_text(_checker_module(started))
  else:
    (tmp_path / "openpilot").mkdir()
    (tmp_path / "openpilot" / "__init__.py").write_text("")
    package = tmp_path / "cereal"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "messaging.py").write_text(_checker_module(started))

  env = os.environ.copy()
  env["PYTHONPATH"] = str(tmp_path)
  result = subprocess.run(
    [sys.executable, str(CHECKER)],
    cwd=tmp_path,
    env=env,
    text=True,
    capture_output=True,
    check=False,
  )
  assert result.returncode == expected, result.stderr


@pytest.mark.parametrize(
  ("outcome", "expected"),
  [
    ("none", 43),
    ("receive-timeout", 43),
    ("invalid", 43),
    ("stale", 43),
    ("future", 43),
    ("wrong-type", 43),
    ("schema", 44),
    ("missing", 44),
    ("decode", 44),
  ],
)
def test_checker_rejects_invalid_or_stale_messages(tmp_path: Path, outcome: str, expected: int) -> None:
  package = tmp_path / "openpilot" / "cereal"
  package.mkdir(parents=True)
  (tmp_path / "openpilot" / "__init__.py").write_text("")
  (package / "__init__.py").write_text("")
  (package / "messaging.py").write_text(_checker_module(False, outcome))
  env = os.environ.copy()
  env["PYTHONPATH"] = str(tmp_path)
  result = subprocess.run(
    [sys.executable, str(CHECKER)],
    cwd=tmp_path,
    env=env,
    text=True,
    capture_output=True,
    check=False,
  )
  assert result.returncode == expected, result.stderr


def test_checker_unrelated_import_failure_is_44(tmp_path: Path) -> None:
  package = tmp_path / "openpilot" / "cereal"
  package.mkdir(parents=True)
  (tmp_path / "openpilot" / "__init__.py").write_text("")
  (package / "__init__.py").write_text("")
  (package / "messaging.py").write_text('raise ImportError("unrelated dependency failure")\n')
  env = os.environ.copy()
  env["PYTHONPATH"] = str(tmp_path)
  result = subprocess.run(
    [sys.executable, str(CHECKER)],
    cwd=tmp_path,
    env=env,
    text=True,
    capture_output=True,
    check=False,
  )
  assert result.returncode == 44
