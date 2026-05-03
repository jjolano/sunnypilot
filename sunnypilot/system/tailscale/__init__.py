"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

import os
import re

# --- Storage layout (all under /data/tailscale/) ---
TAILSCALE_ROOT = "/data/tailscale"
TAILSCALE_BIN_DIR = os.path.join(TAILSCALE_ROOT, "bin")
TAILSCALE_STATE_DIR = os.path.join(TAILSCALE_ROOT, "state")
TAILSCALE_STATE_FILE = os.path.join(TAILSCALE_STATE_DIR, "tailscaled.state")
TAILSCALE_SOCKET = os.path.join(TAILSCALE_ROOT, "tailscaled.sock")

# Symlink that always points to the active version directory
TAILSCALE_CURRENT_LINK = os.path.join(TAILSCALE_BIN_DIR, "current")

# --- Download URLs ---
TAILSCALE_STABLE_BASE = "https://pkgs.tailscale.com/stable"
TAILSCALE_MANIFEST_URL = f"{TAILSCALE_STABLE_BASE}/?mode=json"
TAILSCALE_VERSION_RE = re.compile(r"[0-9]+(?:\.[0-9]+){1,3}")


def _validate_version(version: str) -> str:
  if not TAILSCALE_VERSION_RE.fullmatch(version):
    raise ValueError(f"unsafe Tailscale version: {version!r}")
  return version


def tailscale_bin() -> str:
  """Path to the active 'tailscale' CLI binary."""
  return os.path.join(TAILSCALE_CURRENT_LINK, "tailscale")


def tailscaled_bin() -> str:
  """Path to the active 'tailscaled' daemon binary."""
  return os.path.join(TAILSCALE_CURRENT_LINK, "tailscaled")


def tarball_url(version: str) -> str:
  """Download URL for a specific Tailscale stable tarball (arm64)."""
  version = _validate_version(version)
  return f"{TAILSCALE_STABLE_BASE}/tailscale_{version}_arm64.tgz"


def checksum_url(version: str) -> str:
  """Download URL for the SHA-256 checksum of a stable tarball."""
  return f"{tarball_url(version)}.sha256"


def version_dir(version: str) -> str:
  """Versioned install directory for a given release."""
  version = _validate_version(version)
  path = os.path.realpath(os.path.join(TAILSCALE_BIN_DIR, version))
  if os.path.commonpath([os.path.realpath(TAILSCALE_BIN_DIR), path]) != os.path.realpath(TAILSCALE_BIN_DIR):
    raise ValueError(f"unsafe Tailscale version path: {version!r}")
  return path


def is_installed() -> bool:
  """Return True if the tailscale binaries are reachable via the 'current' symlink."""
  return os.path.isfile(tailscale_bin()) and os.path.isfile(tailscaled_bin())
