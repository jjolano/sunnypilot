from pathlib import Path


LAUNCHER = Path(__file__).resolve().parents[1] / "launch_chffrplus.sh"
REPO_ROOT = Path(__file__).resolve().parents[6]


def test_launcher_uses_migrated_repo_layout():
  script = LAUNCHER.read_text()

  assert 'SP_C3_DIR/../../../../..' in script
  for path in (
    "openpilot/common/hardware/tici/agnos.py",
    "openpilot/common/hardware/tici/updater",
    "openpilot/system/manager",
  ):
    assert (REPO_ROOT / path).exists()
    assert f'$DIR/{path}' in script

  for package in ("msgq", "opendbc", "rednose", "teleoprtc", "tinygrad"):
    assert f"ln -sfn {package}_repo/{package} {package}" in script
