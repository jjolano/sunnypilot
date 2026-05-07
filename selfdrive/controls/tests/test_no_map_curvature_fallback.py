from pathlib import Path

from openpilot.common.basedir import BASEDIR


FORBIDDEN_SYMBOLS_BY_FILE = {
  "cereal/custom.capnp": ("roadCurvatureValid", "roadCurvatureDistances", "roadCurvatures"),
  "cereal/log.capnp": ("mapCurvatureFallback", "mapCurvatureUsed"),
  "common/params_keys.h": ("LateralMapCurvatureFallback",),
  "selfdrive/controls/controlsd.py": ("map_curvature", "roadCurvature", "liveMapDataSP"),
  "selfdrive/controls/lib/model_path_processor.py": ("map_curvature", "MAP_FALLBACK"),
  "selfdrive/ui/sunnypilot/layouts/settings/steering.py": ("LateralMapCurvatureFallback", "lateral_map_curvature"),
  "sunnypilot/mapd/live_map_data/base_map_data.py": ("get_road_curvatures", "roadCurvature"),
  "sunnypilot/mapd/live_map_data/mapd_v2_map_data.py": ("get_road_curvatures", "MAP_CURVATURE"),
  "sunnypilot/mapd/live_map_data/osm_map_data.py": ("get_road_curvatures",),
  "sunnypilot/selfdrive/controls/controlsd_ext.py": ("LateralMapCurvatureFallback", "lateral_map_curvature"),
  "sunnypilot/sunnylink/params_metadata.json": ("LateralMapCurvatureFallback",),
}


def test_map_curvature_fallback_feature_is_removed():
  for relative_path, forbidden_symbols in FORBIDDEN_SYMBOLS_BY_FILE.items():
    contents = Path(BASEDIR, relative_path).read_text()

    for symbol in forbidden_symbols:
      assert symbol not in contents, f"{symbol} still exists in {relative_path}"
