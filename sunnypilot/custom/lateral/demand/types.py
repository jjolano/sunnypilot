from dataclasses import dataclass


DEMAND_SOURCE_MODEL_PATH = "model_path"
DEMAND_SOURCE_LATERAL_MANEUVER = "lateral_maneuver"
DEMAND_SOURCE_FALLBACK_MEASURED = "fallback_measured"
DEMAND_SOURCE_LANE_FIT = "lane_fit"


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
  lane_centering_assist_active: bool = False
  lane_centering_reason: str = ""
  lane_centering_lateral_error: float = 0.0
  lane_centering_heading_error: float = 0.0
  lane_centering_predicted_error: float = 0.0
  lane_centering_curvature_nudge: float = 0.0
  lane_centering_confidence: float = 0.0
  lane_centering_relax_active: bool = False
  lane_centering_relax_reason_bits: int = 0
  lane_centering_relax_envelope: float = 0.0
  lane_centering_relax_lateral_error: float = 0.0
  lane_centering_relax_predicted_error: float = 0.0
  lane_centering_relax_age: float = 0.0
  lane_centering_relax_nudge_flip_score: float = 0.0
  lane_centering_relax_error_cross_score: float = 0.0
