from pathlib import Path

from openpilot.sunnypilot.mapd.mapd_installer import VERSION


MAPD_DIR = Path(__file__).resolve().parents[1]


def test_mapd_version_uses_v2_release():
  assert VERSION.startswith("v2.")


def test_mapd_v2_branch_does_not_ship_legacy_source_patches():
  patch_dir = MAPD_DIR / "patches"

  assert not list(patch_dir.glob("mapd-v1.*.patch"))
