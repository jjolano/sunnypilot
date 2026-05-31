from dataclasses import dataclass


DEMAND_SOURCE_MODEL_PATH = "model_path"
DEMAND_SOURCE_LATERAL_MANEUVER = "lateral_maneuver"
DEMAND_SOURCE_FALLBACK_MEASURED = "fallback_measured"


@dataclass(frozen=True)
class ProcessedLateralDemand:
  raw_curvature: float
  processed_curvature: float
  measured_curvature: float
  curvature_limited: bool
  path_quality: float
  path_reason: str
  lane_change_shaping_active: bool
  lane_change_blend: float
  lateral_accel_limit: float
  demand_source: str = DEMAND_SOURCE_MODEL_PATH
