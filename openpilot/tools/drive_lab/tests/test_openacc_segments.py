import json
from pathlib import Path

from openpilot.tools.drive_lab.log_profile import load_profile
from openpilot.tools.drive_lab.openacc_segments import build_openacc_profile


def test_build_openacc_profile_from_csv(tmp_path: Path):
  csv_path = tmp_path / "segment.csv"
  csv_path.write_text(
    "time,v_ego,v_lead,gap\n"
    "0.0,10.0,9.0,30.0\n"
    "0.1,10.1,9.0,29.5\n"
    "0.2,10.2,8.9,29.0\n"
    "0.3,10.3,8.8,28.5\n"
    "0.4,10.4,8.7,28.0\n"
    "0.5,10.5,8.6,27.5\n",
    encoding="utf-8",
  )
  profile = build_openacc_profile(csv_path)
  assert profile.source.startswith("openacc:")
  assert profile.ego_speed.low <= profile.ego_speed.high
  out = tmp_path / "profile.json"
  out.write_text(json.dumps(profile.to_dict()), encoding="utf-8")
  loaded = load_profile(out)
  assert loaded.source == profile.source
