"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

import hashlib
import io
import os
import subprocess
import tarfile
import tempfile

import pytest
from pytest_mock import MockerFixture

from openpilot.common.params import ParamKeyFlag, Params
from openpilot.sunnypilot.system.tailscale import (
  TAILSCALE_CURRENT_LINK,
  TAILSCALE_ROOT,
  TAILSCALE_SOCKET,
  TAILSCALE_STATE_FILE,
  checksum_url,
  is_installed,
  tailscale_bin,
  tailscaled_bin,
  tarball_url,
  version_dir,
)
from openpilot.sunnypilot.system.tailscale.installer import (
  _extract_tailscale_binaries,
  _verify_checksum,
  fetch_latest_version,
)
from openpilot.sunnypilot.system.tailscale.manage_tailscaled import TailscaleDaemon, VERSION_CHECK_INTERVAL, _decode_json_objects, _stale_tailscaled_pids


class FakeParams:
  def __init__(self):
    self.values = {}
    self.put_counts = {}

  def put(self, key, value):
    self.put_counts[key] = self.put_counts.get(key, 0) + 1
    self.values[key] = value

  def put_bool(self, key, value):
    self.values[key] = bool(value)

  def get(self, key):
    return self.values.get(key, "")

  def get_bool(self, key):
    return bool(self.values.get(key, False))

  def remove(self, key):
    self.values.pop(key, None)


def add_tar_file(tar: tarfile.TarFile, name: str, data: bytes) -> None:
  info = tarfile.TarInfo(name)
  info.size = len(data)
  tar.addfile(info, io.BytesIO(data))


class TestTailscaleConstants:
  """Verify path helpers and URL builders produce correct strings."""

  def test_tailscaled_pid_survives_manager_start_clear(self):
    params = Params()
    try:
      params.put("TailscaledPid", 123, block=True)
      params.put("TailscaleState", "Running", block=True)

      params.clear_all(ParamKeyFlag.CLEAR_ON_MANAGER_START)

      assert params.get("TailscaledPid") == 123
      assert params.get("TailscaleState") is None
    finally:
      params.remove("TailscaledPid")
      params.remove("TailscaleState")

  def test_paths_under_data_tailscale(self):
    assert TAILSCALE_ROOT == "/data/tailscale"
    assert TAILSCALE_STATE_FILE.startswith("/data/tailscale/")
    assert TAILSCALE_SOCKET.startswith("/data/tailscale/")

  def test_tailscale_bin_paths(self):
    assert tailscale_bin().endswith("/tailscale")
    assert tailscaled_bin().endswith("/tailscaled")
    assert TAILSCALE_CURRENT_LINK in tailscale_bin()

  def test_version_dir(self):
    d = version_dir("1.96.4")
    assert d.endswith("/1.96.4")
    assert d.startswith(TAILSCALE_ROOT)

  @pytest.mark.parametrize("version", ["", "../1.96.4", "1.96.4/evil", ".current", "1.96.4\n"])
  def test_version_dir_rejects_unsafe_versions(self, version):
    with pytest.raises(ValueError):
      version_dir(version)

  def test_tarball_url(self):
    url = tarball_url("1.96.4")
    assert "tailscale_1.96.4_arm64.tgz" in url
    assert url.startswith("https://pkgs.tailscale.com/stable/")

  def test_checksum_url(self):
    url = checksum_url("1.96.4")
    assert url.endswith(".sha256")
    assert "1.96.4" in url

  def test_is_installed_false_when_no_symlink(self):
    assert not is_installed()


class TestTailscaleInstaller:
  """Test installer helper functions."""

  def test_extract_tailscale_binaries_writes_only_controlled_paths(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      tarball_path = os.path.join(tmp_dir, "tailscale.tgz")
      staging_dir = os.path.join(tmp_dir, "staged")
      os.makedirs(staging_dir)

      with tarfile.open(tarball_path, "w:gz") as tar:
        add_tar_file(tar, "tailscale_1.96.4_arm64/tailscale", b"tailscale-bin")
        add_tar_file(tar, "tailscale_1.96.4_arm64/tailscaled", b"tailscaled-bin")

      assert _extract_tailscale_binaries(tarball_path, staging_dir)
      with open(os.path.join(staging_dir, "tailscale"), "rb") as f:
        assert f.read() == b"tailscale-bin"
      with open(os.path.join(staging_dir, "tailscaled"), "rb") as f:
        assert f.read() == b"tailscaled-bin"

  def test_extract_tailscale_binaries_rejects_traversal_member(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      tarball_path = os.path.join(tmp_dir, "tailscale.tgz")
      staging_dir = os.path.join(tmp_dir, "staged")
      os.makedirs(staging_dir)

      with tarfile.open(tarball_path, "w:gz") as tar:
        add_tar_file(tar, "../tailscale", b"bad")
        add_tar_file(tar, "tailscale_1.96.4_arm64/tailscaled", b"tailscaled-bin")

      assert not _extract_tailscale_binaries(tarball_path, staging_dir)
      assert not os.path.exists(os.path.join(staging_dir, "tailscale"))

  def test_verify_checksum_valid(self):
    content = b"hello tailscale"
    expected = hashlib.sha256(content).hexdigest()

    with tempfile.NamedTemporaryFile(delete=False) as f:
      f.write(content)
      f.flush()
      try:
        assert _verify_checksum(f.name, expected)
      finally:
        os.unlink(f.name)

  def test_verify_checksum_invalid(self):
    with tempfile.NamedTemporaryFile(delete=False) as f:
      f.write(b"hello tailscale")
      f.flush()
      try:
        assert not _verify_checksum(f.name, "0" * 64)
      finally:
        os.unlink(f.name)

  def test_verify_checksum_with_filename_suffix(self):
    content = b"test data"
    expected = hashlib.sha256(content).hexdigest()
    checksum_line = f"{expected}  tailscale_1.96.4_arm64.tgz"

    with tempfile.NamedTemporaryFile(delete=False) as f:
      f.write(content)
      f.flush()
      try:
        assert _verify_checksum(f.name, checksum_line)
      finally:
        os.unlink(f.name)

  def test_fetch_latest_version_success(self, mocker: MockerFixture):
    mock_resp = mocker.MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"TarballsVersion": "1.96.4"}
    mock_resp.raise_for_status = mocker.MagicMock()
    mocker.patch("openpilot.sunnypilot.system.tailscale.installer.requests.get", return_value=mock_resp)

    version = fetch_latest_version()
    assert version == "1.96.4"

  def test_fetch_latest_version_strips_v_prefix(self, mocker: MockerFixture):
    mock_resp = mocker.MagicMock()
    mock_resp.json.return_value = {"TarballsVersion": "v1.96.4"}
    mock_resp.raise_for_status = mocker.MagicMock()
    mocker.patch("openpilot.sunnypilot.system.tailscale.installer.requests.get", return_value=mock_resp)

    version = fetch_latest_version()
    assert version == "1.96.4"

  def test_fetch_latest_version_failure(self, mocker: MockerFixture):
    mocker.patch("openpilot.sunnypilot.system.tailscale.installer.requests.get", side_effect=Exception("network error"))
    version = fetch_latest_version()
    assert version is None

  def test_download_and_install_rejects_unsafe_version_before_network(self, mocker: MockerFixture):
    params = FakeParams()
    download = mocker.patch("openpilot.sunnypilot.system.tailscale.installer._download_with_retry")

    from openpilot.sunnypilot.system.tailscale.installer import download_and_install

    assert not download_and_install("../1.96.4", params)
    assert params.values["TailscaleInstallState"] == "error:invalid version"
    download.assert_not_called()

  def test_login_subprocess_does_not_pipe_stderr(self, mocker: MockerFixture):
    fake_proc = mocker.MagicMock()
    fake_proc.stdout.fileno.return_value = 0
    fake_proc.stderr = None
    fake_proc.poll.return_value = 1
    fake_proc.returncode = 1
    popen = mocker.patch("openpilot.sunnypilot.system.tailscale.manage_tailscaled.subprocess.Popen", return_value=fake_proc)

    daemon = TailscaleDaemon.__new__(TailscaleDaemon)
    daemon.params = FakeParams()
    daemon._login_proc = None
    daemon._handle_login()

    assert popen.call_args.kwargs["stderr"] is subprocess.DEVNULL

  def test_decode_json_objects_handles_multiple_objects_and_partial_tail(self):
    objects, remaining = _decode_json_objects('{"AuthURL":"https://login"}\n{"BackendState":"Running"}{"partial"')

    assert objects == [{"AuthURL": "https://login"}, {"BackendState": "Running"}]
    assert remaining == '{"partial"'

  def test_stale_tailscaled_pids_match_managed_socket_and_binary_only(self):
    output = "\n".join([
      f"123 {tailscaled_bin()} --socket={TAILSCALE_SOCKET} --tun=userspace-networking",
      f"456 /usr/bin/tailscaled --socket={TAILSCALE_SOCKET}",
      "789 /usr/bin/tailscaled --socket=/tmp/other.sock",
      f"999 {tailscaled_bin()} --socket={TAILSCALE_SOCKET}.bak",
      f"321 /bin/echo {tailscaled_bin()} --socket={TAILSCALE_SOCKET}",
    ])

    assert _stale_tailscaled_pids(output) == ["123"]

  def test_stop_tailscaled_cleans_stale_processes_without_tracked_process(self, mocker: MockerFixture):
    daemon = TailscaleDaemon.__new__(TailscaleDaemon)
    daemon._tailscaled_proc = None
    kill_stale = mocker.patch.object(daemon, "_kill_stale_tailscaled")

    daemon._stop_tailscaled()

    kill_stale.assert_called_once()

  def test_check_latest_version_skips_network_when_disabled_and_not_installed(self, mocker: MockerFixture):
    daemon = TailscaleDaemon.__new__(TailscaleDaemon)
    daemon.params = FakeParams()
    daemon._last_version_check = -VERSION_CHECK_INTERVAL
    fetch = mocker.patch("openpilot.sunnypilot.system.tailscale.manage_tailscaled.fetch_latest_version")
    mocker.patch("openpilot.sunnypilot.system.tailscale.manage_tailscaled.is_installed", return_value=False)

    daemon._check_latest_version()

    fetch.assert_not_called()

  def test_publish_status_writes_final_state_once_and_skips_unchanged(self):
    daemon = TailscaleDaemon.__new__(TailscaleDaemon)
    daemon.params = FakeParams()
    status = {
      "BackendState": "Running",
      "Self": {"TailscaleIPs": ["100.64.0.1"]},
    }

    daemon._publish_status(status)

    assert daemon.params.values["TailscaleState"] == "Running:100.64.0.1"
    assert daemon.params.put_counts["TailscaleState"] == 1

    daemon._publish_status(status)

    assert daemon.params.put_counts["TailscaleState"] == 1


class TestTailscalePairingDialog:
  def test_clear_auth_url_helper_avoids_ui_import(self):
    from openpilot.sunnypilot.system.tailscale.auth import clear_tailscale_auth_url

    params = FakeParams()
    params.put("TailscaleAuthURL", "https://login.tailscale.com/a/abc123")

    clear_tailscale_auth_url(params)

    assert params.get("TailscaleAuthURL") == ""


class TestTailscaleStatusParsing:
  """Test parsing of tailscale status --json output."""

  def test_parse_running_status(self):
    status = {
      "BackendState": "Running",
      "Self": {
        "TailscaleIPs": ["100.64.0.1", "fd7a:115c:a1e0::1"],
        "HostName": "comma-device",
      },
      "CurrentTailnet": {"Name": "example@github"},
      "Version": "1.96.4",
    }

    backend_state = status.get("BackendState", "Unknown")
    assert backend_state == "Running"

    self_status = status.get("Self", {})
    ips = self_status.get("TailscaleIPs", [])
    assert len(ips) == 2
    assert "100.64.0.1" in ips

  def test_parse_needs_login_status(self):
    status = {
      "BackendState": "NeedsLogin",
      "AuthURL": "https://login.tailscale.com/a/abc123",
      "Self": {"TailscaleIPs": []},
    }

    assert status["BackendState"] == "NeedsLogin"
    assert status["AuthURL"].startswith("https://")

  def test_parse_empty_status(self):
    status = {}
    backend_state = status.get("BackendState", "Unknown")
    assert backend_state == "Unknown"

    self_status = status.get("Self", {})
    ips = self_status.get("TailscaleIPs", [])
    assert ips == []
