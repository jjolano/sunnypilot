from dataclasses import dataclass


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
