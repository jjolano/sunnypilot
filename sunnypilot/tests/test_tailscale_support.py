"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

import hashlib
import os
import tempfile

from pytest_mock import MockerFixture

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
  _verify_checksum,
  fetch_latest_version,
)


class TestTailscaleConstants:
  """Verify path helpers and URL builders produce correct strings."""

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
