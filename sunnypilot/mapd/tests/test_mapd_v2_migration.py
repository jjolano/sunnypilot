from pathlib import Path
import re

from openpilot.sunnypilot.mapd.mapd_installer import VERSION


MAPD_DIR = Path(__file__).resolve().parents[1]
CUSTOM_CAPNP_PATH = Path(__file__).resolve().parents[3] / "cereal" / "custom.capnp"
EXPECTED_MAPD_V2_VERSION = "v2.0.6"
EXPECTED_MAPD_OUT_FIELDS = (
  ("wayName", "0", "Text"),
  ("wayRef", "1", "Text"),
  ("roadName", "2", "Text"),
  ("speedLimit", "3", "Float32"),
  ("nextSpeedLimit", "4", "Float32"),
  ("nextSpeedLimitDistance", "5", "Float32"),
  ("hazard", "6", "Text"),
  ("nextHazard", "7", "Text"),
  ("nextHazardDistance", "8", "Float32"),
  ("advisorySpeed", "9", "Float32"),
  ("nextAdvisorySpeed", "10", "Float32"),
  ("nextAdvisorySpeedDistance", "11", "Float32"),
  ("oneWay", "12", "Bool"),
  ("lanes", "13", "UInt8"),
  ("tileLoaded", "14", "Bool"),
  ("speedLimitSuggestedSpeed", "15", "Float32"),
  ("suggestedSpeed", "16", "Float32"),
  ("estimatedRoadWidth", "17", "Float32"),
  ("roadContext", "18", "RoadContext"),
  ("distanceFromWayCenter", "19", "Float32"),
  ("visionCurveSpeed", "20", "Float32"),
  ("mapCurveSpeed", "21", "Float32"),
  ("waySelectionType", "22", "WaySelectionType"),
  ("speedLimitAccepted", "23", "Bool"),
)


def test_mapd_version_uses_v2_release():
  assert VERSION.startswith("v2.")
  assert VERSION == EXPECTED_MAPD_V2_VERSION


def test_mapd_v2_branch_does_not_ship_legacy_source_patches():
  patch_dir = MAPD_DIR / "patches"

  assert not list(patch_dir.glob("mapd-v1.*.patch"))


def test_local_mapd_out_schema_matches_pinned_v2_release():
  schema = CUSTOM_CAPNP_PATH.read_text()
  struct_match = re.search(r"struct MapdOut @0x[0-9a-f]+ \{(?P<body>.*?)\n\}", schema, re.S)
  assert struct_match is not None

  fields = tuple(re.findall(r"^\s+(\w+) @(\d+) :(\w+);", struct_match.group("body"), re.M))

  assert fields == EXPECTED_MAPD_OUT_FIELDS
