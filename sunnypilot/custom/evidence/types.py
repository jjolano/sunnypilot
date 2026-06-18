from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SourceHealth(Enum):
  HEALTHY = "healthy"
  DEGRADED = "degraded"
  UNAVAILABLE = "unavailable"
  UNKNOWN = "unknown"


@dataclass(frozen=True)
class SourceStatus:
  health: SourceHealth = SourceHealth.UNKNOWN
  reasons: tuple[str, ...] = ()
  age_s: float | None = None


@dataclass(frozen=True)
class EgoEvidence:
  source: SourceStatus
  v_ego_mps: float | None = None
  a_ego_mps2: float | None = None
  standstill: bool | None = None
  brake_pressed: bool | None = None
  gas_pressed: bool | None = None
  steering_pressed: bool | None = None


@dataclass(frozen=True)
class ModelPathEvidence:
  source: SourceStatus
  position_x_m: tuple[float | None, ...] = ()
  position_y_m: tuple[float | None, ...] = ()
  position_y_std_m: tuple[float | None, ...] = ()
  orientation_z_rad: tuple[float | None, ...] = ()
  orientation_rate_z_rad_s: tuple[float | None, ...] = ()
  lane_line_probs: tuple[float | None, ...] = ()
  frame_drop_perc: float | None = None
  desired_curvature_1_m: float | None = None


@dataclass(frozen=True)
class ModelActionEvidence:
  source: SourceStatus
  should_stop: bool | None = None
  desired_acceleration_mps2: float | None = None
  model_stop_distance_m: float | None = None


@dataclass(frozen=True)
class LeadObservation:
  source: SourceStatus
  status: bool | None = None
  d_rel_m: float | None = None
  y_rel_m: float | None = None
  v_rel_mps: float | None = None
  v_lead_mps: float | None = None
  a_lead_mps2: float | None = None
  radar_track_id: int | None = None
  model_prob: float | None = None
  closing_speed_mps: float | None = None
  ttc_s: float | None = None
  time_headway_s: float | None = None


@dataclass(frozen=True)
class LeadEvidence:
  source: SourceStatus
  lead_one: LeadObservation
  lead_two: LeadObservation


@dataclass(frozen=True)
class EvidenceSnapshot:
  ego: EgoEvidence
  model_path: ModelPathEvidence
  model_action: ModelActionEvidence
  lead: LeadEvidence
  source_statuses: tuple[tuple[str, SourceStatus], ...]
