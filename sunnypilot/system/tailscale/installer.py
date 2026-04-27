"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

import hashlib
import logging
import os
import shutil
import stat
import tarfile
import tempfile
import time

import requests

from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog

from openpilot.sunnypilot.system.tailscale import (
  TAILSCALE_BIN_DIR,
  TAILSCALE_CURRENT_LINK,
  TAILSCALE_MANIFEST_URL,
  TAILSCALE_ROOT,
  TAILSCALE_STATE_DIR,
  checksum_url,
  tarball_url,
  version_dir,
)

logger = logging.getLogger(__name__)

DOWNLOAD_TIMEOUT = 120  # seconds
MAX_RETRIES = 3


def _set_install_state(params: Params, state: str, progress: str = "") -> None:
  params.put("TailscaleInstallState", state)
  params.put("TailscaleInstallProgress", progress)


def fetch_latest_version(timeout: int = 15) -> str | None:
  """Fetch the latest stable Tailscale version string from the manifest."""
  try:
    resp = requests.get(TAILSCALE_MANIFEST_URL, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    version = data.get("TarballsVersion", "").lstrip("v")
    return version if version else None
  except Exception:
    cloudlog.exception("tailscale: failed to fetch latest version")
    return None


def ensure_directories() -> None:
  """Create the Tailscale storage directories if they don't exist."""
  for d in (TAILSCALE_ROOT, TAILSCALE_BIN_DIR, TAILSCALE_STATE_DIR):
    os.makedirs(d, exist_ok=True)


def _download_with_retry(url: str, dest: str, params: Params, num_retries: int = MAX_RETRIES) -> bool:
  """Download a URL to a local file with retry logic and progress reporting."""
  for attempt in range(num_retries):
    try:
      resp = requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT)
      resp.raise_for_status()

      total = int(resp.headers.get("content-length", 0))
      downloaded = 0

      with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=256 * 1024):
          f.write(chunk)
          downloaded += len(chunk)
          if total > 0:
            pct = min(100, int(downloaded * 100 / total))
            _set_install_state(params, "downloading", str(pct))

      return True
    except requests.exceptions.RequestException as e:
      cloudlog.warning(f"tailscale: download attempt {attempt + 1}/{num_retries} failed: {e}")
      time.sleep(1 + attempt)

  return False


def _verify_checksum(file_path: str, expected_hex: str) -> bool:
  """Verify SHA-256 checksum of a file against an expected hex digest."""
  sha = hashlib.sha256()
  with open(file_path, "rb") as f:
    while chunk := f.read(256 * 1024):
      sha.update(chunk)
  return sha.hexdigest() == expected_hex.strip().split()[0]


def _make_executable(path: str) -> None:
  """Add the executable bit to a file."""
  current = stat.S_IMODE(os.lstat(path).st_mode)
  os.chmod(path, current | stat.S_IEXEC)


def _is_safe_tar_member_path(path: str) -> bool:
  parts = path.split("/")
  return not os.path.isabs(path) and ".." not in parts


def _extract_tailscale_binaries(tarball_path: str, staging_dir: str) -> bool:
  """Extract only the Tailscale binaries into controlled staging paths."""
  expected = {"tailscale", "tailscaled"}
  extracted = set()

  with tarfile.open(tarball_path, "r:gz") as tar:
    for member in tar.getmembers():
      basename = os.path.basename(member.name)
      if basename not in expected:
        continue
      if not member.isfile() or not _is_safe_tar_member_path(member.name) or basename in extracted:
        return False

      src = tar.extractfile(member)
      if src is None:
        return False
      dst = os.path.join(staging_dir, basename)
      with src, open(dst, "wb") as f:
        shutil.copyfileobj(src, f)
      _make_executable(dst)
      extracted.add(basename)

  return extracted == expected


def download_and_install(version: str, params: Params | None = None) -> bool:
  """Download, verify, and install a specific Tailscale version.

  Returns True on success, False on failure. Updates TailscaleInstallState
  and TailscaleInstallProgress params throughout the process.
  """
  if params is None:
    params = Params()

  try:
    dest_dir = version_dir(version)
  except ValueError:
    _set_install_state(params, "error:invalid version")
    return False

  ensure_directories()
  _set_install_state(params, "downloading", "0")

  # Use a temporary directory for atomic install
  tmp_dir = tempfile.mkdtemp(dir=TAILSCALE_BIN_DIR, prefix=f".install-{version}-")
  tarball_path = os.path.join(tmp_dir, "tailscale.tgz")

  try:
    # 1. Download tarball
    if not _download_with_retry(tarball_url(version), tarball_path, params):
      _set_install_state(params, "error:download failed")
      return False

    # 2. Download and verify checksum
    _set_install_state(params, "verifying")
    try:
      resp = requests.get(checksum_url(version), timeout=15)
      resp.raise_for_status()
      expected_checksum = resp.text.strip()
    except requests.exceptions.RequestException:
      cloudlog.exception("tailscale: failed to download checksum")
      _set_install_state(params, "error:checksum download failed")
      return False

    if not _verify_checksum(tarball_path, expected_checksum):
      _set_install_state(params, "error:checksum mismatch")
      return False

    # 3. Extract binaries
    _set_install_state(params, "extracting")
    staging_dir = os.path.join(tmp_dir, "staged")
    os.makedirs(staging_dir)

    if not _extract_tailscale_binaries(tarball_path, staging_dir):
      _set_install_state(params, "error:binaries not found in tarball")
      return False

    # Atomic rename into place
    if os.path.exists(dest_dir):
      shutil.rmtree(dest_dir)
    os.rename(staging_dir, dest_dir)

    # 5. Update the 'current' symlink atomically
    tmp_link = os.path.join(TAILSCALE_BIN_DIR, f".current-{version}.tmp")
    if os.path.islink(tmp_link) or os.path.exists(tmp_link):
      os.remove(tmp_link)
    os.symlink(dest_dir, tmp_link)
    os.rename(tmp_link, TAILSCALE_CURRENT_LINK)

    # 6. Record installed version
    params.put("TailscaleInstalledVersion", version)
    _set_install_state(params, "done")
    cloudlog.info(f"tailscale: installed version {version}")
    return True

  except Exception:
    cloudlog.exception(f"tailscale: install failed for version {version}")
    _set_install_state(params, "error:unexpected failure")
    return False

  finally:
    # Clean up temp directory
    if os.path.exists(tmp_dir):
      shutil.rmtree(tmp_dir, ignore_errors=True)
