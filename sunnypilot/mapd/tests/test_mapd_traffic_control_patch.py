from pathlib import Path


PATCH_PATH = Path(__file__).resolve().parents[1] / "patches" / "mapd-v1.12-traffic-controls.patch"


def test_traffic_control_patch_stores_direction_specific_controls():
  patch = PATCH_PATH.read_text()

  assert "TrafficControlForward" in patch
  assert "TrafficControlBackward" in patch
  assert "SetTrafficControlForward" in patch
  assert "SetTrafficControlBackward" in patch
  assert "trafficControlForDirection" in patch
