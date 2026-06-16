import json
from pathlib import Path

from openpilot.tools.drive_lab.commonroad_acc import FIXTURE_DIR, generate_commonroad_acc_scenarios


def test_commonroad_fixture_files_exist():
  for name in ("ZAM_ACC-1_1_T-1", "ZAM_ACC-1_2_T-1", "ZAM_ACC-1_3_T-1", "ZAM_CC-1_1_T-1"):
    assert (FIXTURE_DIR / f"{name}.json").is_file()


def test_commonroad_fixture_json_valid():
  for path in FIXTURE_DIR.glob("*.json"):
    data = json.loads(path.read_text(encoding="utf-8"))
    assert "kind" in data
    assert "kwargs" in data
    assert "breakpoints" in data["kwargs"]
