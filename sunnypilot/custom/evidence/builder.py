from __future__ import annotations

import math
from typing import Any

from openpilot.sunnypilot.custom.evidence.types import (
  EvidenceSnapshot,
  EgoEvidence,
  LeadEvidence,
  LeadObservation,
  ModelActionEvidence,
  ModelPathEvidence,
  SourceHealth,
  SourceStatus,
)

MODEL_STOP_SPEED = 0.3
MODEL_PATH_MIN_SAMPLES = 17
CANONICAL_SOURCE_NAMES = ("carState", "modelV2", "radarState")


def _finite_or_none(value: Any) -> float | None:
  try:
    v = float(value)
  except (TypeError, ValueError):
    return None
  return v if math.isfinite(v) else None


def _bool_or_none(value: Any) -> bool | None:
  return value if isinstance(value, bool) else None


def _tuple_finite_or_none(values: Any) -> tuple[float | None, ...]:
  if isinstance(values, (str, bytes)) or not hasattr(values, "__iter__"):
    return ()
  return tuple(_finite_or_none(v) for v in values)


def _source_status(status: Any, *, present: bool) -> SourceStatus:
  if isinstance(status, SourceStatus):
    return status
  if status is None:
    return SourceStatus(SourceHealth.UNKNOWN if present else SourceHealth.UNAVAILABLE,
                        ("unknown_freshness",) if present else ("missing_source",))
  if isinstance(status, dict):
    health = status.get("health", SourceHealth.UNKNOWN)
    if not isinstance(health, SourceHealth):
      health = SourceHealth.UNKNOWN
    reasons = tuple(str(r) for r in status.get("reasons", ()) if r is not None)
    return SourceStatus(health=health, reasons=reasons, age_s=_finite_or_none(status.get("age_s")))
  return SourceStatus(SourceHealth.UNKNOWN, ("unknown_freshness",) if present else ("missing_source",))


def _degraded(source: SourceStatus, *reasons: str) -> SourceStatus:
  health = SourceHealth.UNAVAILABLE if source.health is SourceHealth.UNAVAILABLE else SourceHealth.DEGRADED
  return SourceStatus(health, tuple(dict.fromkeys((*source.reasons, *reasons))), source.age_s)


def _normalize_source_statuses(source_status: Any) -> dict[str, SourceStatus]:
  if not isinstance(source_status, dict):
    return {}
  out: dict[str, SourceStatus] = {}
  for name in CANONICAL_SOURCE_NAMES:
    if name in source_status:
      out[name] = _source_status(source_status.get(name), present=True)
  return out


def _model_stop_distance(model: Any) -> float | None:
  position = getattr(model, "position", None)
  velocity = getattr(model, "velocity", None)
  xs = _tuple_finite_or_none(getattr(position, "x", None) if position is not None else None)
  vs = _tuple_finite_or_none(getattr(velocity, "x", None) if velocity is not None else None)
  if not xs or not vs or len(xs) != len(vs):
    return None
  for x, v in zip(xs, vs, strict=True):
    if v is not None and v <= MODEL_STOP_SPEED:
      return x if x is not None and x > 0.0 else None
  return None


def _path_source(position_x: tuple[float | None, ...], position_y: tuple[float | None, ...], explicit: SourceStatus,
                 *, extra_nonfinite: bool = False) -> SourceStatus:
  reasons: list[str] = []
  if not position_x or not position_y:
    reasons.append("missing_field")
    return _degraded(explicit, *reasons)
  if len(position_x) != len(position_y):
    reasons.append("length_mismatch")
  if len(position_x) < MODEL_PATH_MIN_SAMPLES or len(position_y) < MODEL_PATH_MIN_SAMPLES:
    reasons.append("short_path")
  if any(v is None for v in position_x + position_y):
    reasons.append("nonfinite_field")
  if extra_nonfinite:
    reasons.append("nonfinite_field")
  health = explicit.health
  if explicit.health is SourceHealth.HEALTHY and reasons:
    health = SourceHealth.DEGRADED
  if explicit.health is SourceHealth.UNKNOWN and reasons:
    health = SourceHealth.DEGRADED
  if reasons and health is SourceHealth.HEALTHY:
    health = SourceHealth.DEGRADED
  return SourceStatus(health=health, reasons=tuple(dict.fromkeys((*explicit.reasons, *reasons))), age_s=explicit.age_s)


def _lead_source(explicit: SourceStatus, lead_present: bool | None) -> SourceStatus:
  if lead_present is False:
    return SourceStatus(SourceHealth.UNAVAILABLE, ("missing_field",))
  if lead_present is None:
    return SourceStatus(SourceHealth.UNAVAILABLE, ("missing_source",))
  return explicit


def _final_source(present: bool, explicit: SourceStatus | None) -> SourceStatus:
  if explicit is not None:
    return explicit
  return _source_status(None, present=present)


def _lead_observation(lead: Any, v_ego_mps: float | None, source: SourceStatus) -> LeadObservation:
  lead_present = _bool_or_none(getattr(lead, "status", None)) if lead is not None else None
  if lead is None:
    source = SourceStatus(SourceHealth.UNAVAILABLE, ("missing_source",))
  elif lead_present is False:
    source = SourceStatus(SourceHealth.UNAVAILABLE, ("missing_field",))
  d_rel_raw = getattr(lead, "dRel", None) if lead is not None else None
  d_rel = _finite_or_none(d_rel_raw)
  y_rel = _finite_or_none(getattr(lead, "yRel", None))
  v_rel = _finite_or_none(getattr(lead, "vRel", None))
  v_lead = _finite_or_none(getattr(lead, "vLeadK", None))
  if v_lead is None:
    v_lead = _finite_or_none(getattr(lead, "vLead", None))
  a_lead = _finite_or_none(getattr(lead, "aLeadK", None))
  if a_lead is None:
    a_lead = _finite_or_none(getattr(lead, "aLead", None))
  radar_track_id = getattr(lead, "radarTrackId", None)
  radar_track_id = radar_track_id if isinstance(radar_track_id, int) else None
  model_prob = _finite_or_none(getattr(lead, "modelProb", None))
  closing_speed = -v_rel if v_rel is not None and v_rel < 0.0 else 0.0 if v_rel is not None else None
  if lead_present is True and (d_rel is None or v_rel is None):
    source = _degraded(source, "missing_field")
  if lead_present is True and ((d_rel_raw is not None and d_rel is None) or (getattr(lead, "vRel", None) is not None and v_rel is None)):
    source = _degraded(source, "nonfinite_field")
  ttc = (d_rel / closing_speed) if (source.health is not SourceHealth.DEGRADED and d_rel is not None and closing_speed not in (None, 0.0) and closing_speed > 0.0) else None
  time_headway = d_rel / v_ego_mps if d_rel is not None and v_ego_mps is not None and v_ego_mps > 0.0 else None
  if lead_present is not True:
    return LeadObservation(source=source, status=lead_present)
  return LeadObservation(source=source, status=lead_present, d_rel_m=d_rel, y_rel_m=y_rel, v_rel_mps=v_rel, v_lead_mps=v_lead,
                         a_lead_mps2=a_lead, radar_track_id=radar_track_id, model_prob=model_prob,
                         closing_speed_mps=closing_speed, ttc_s=ttc, time_headway_s=time_headway)


def build_snapshot(*, car_state=None, model_v2=None, radar_state=None, source_status=None) -> EvidenceSnapshot:
  source_statuses = _normalize_source_statuses(source_status)
  car_source = source_statuses.get("carState", _source_status(None, present=car_state is not None))
  model_source = source_statuses.get("modelV2", _source_status(None, present=model_v2 is not None))
  radar_source = source_statuses.get("radarState", _source_status(None, present=radar_state is not None))
  ego = EgoEvidence(
    source=car_source,
    v_ego_mps=_finite_or_none(getattr(car_state, "vEgo", None)) if car_state is not None else None,
    a_ego_mps2=_finite_or_none(getattr(car_state, "aEgo", None)) if car_state is not None else None,
    standstill=_bool_or_none(getattr(car_state, "standstill", None)) if car_state is not None else None,
    brake_pressed=_bool_or_none(getattr(car_state, "brakePressed", None)) if car_state is not None else None,
    gas_pressed=_bool_or_none(getattr(car_state, "gasPressed", None)) if car_state is not None else None,
    steering_pressed=_bool_or_none(getattr(car_state, "steeringPressed", None)) if car_state is not None else None,
  )
  if car_state is not None and (getattr(car_state, "vEgo", None) is not None and ego.v_ego_mps is None or getattr(car_state, "aEgo", None) is not None and ego.a_ego_mps2 is None):
    ego = EgoEvidence(source=_degraded(car_source, "nonfinite_field"), v_ego_mps=ego.v_ego_mps, a_ego_mps2=ego.a_ego_mps2, standstill=ego.standstill, brake_pressed=ego.brake_pressed, gas_pressed=ego.gas_pressed, steering_pressed=ego.steering_pressed)
  position_x = _tuple_finite_or_none(getattr(getattr(model_v2, "position", None), "x", None)) if model_v2 is not None else ()
  position_y = _tuple_finite_or_none(getattr(getattr(model_v2, "position", None), "y", None)) if model_v2 is not None else ()
  action = getattr(model_v2, "action", None) if model_v2 is not None else None
  desired_curvature_raw = getattr(action, "desiredCurvature", None) if action is not None else None
  desired_curvature = _finite_or_none(desired_curvature_raw)
  frame_drop_raw = getattr(model_v2, "frameDropPerc", None) if model_v2 is not None else None
  frame_drop = _finite_or_none(frame_drop_raw)
  model_path = ModelPathEvidence(
    source=_path_source(
      position_x, position_y, model_source,
      extra_nonfinite=(desired_curvature_raw is not None and desired_curvature is None) or (frame_drop_raw is not None and frame_drop is None),
    ),
    position_x_m=position_x,
    position_y_m=position_y,
    position_y_std_m=_tuple_finite_or_none(getattr(getattr(model_v2, "position", None), "yStd", None)) if model_v2 is not None else (),
    orientation_z_rad=_tuple_finite_or_none(getattr(getattr(model_v2, "orientation", None), "z", None)) if model_v2 is not None else (),
    orientation_rate_z_rad_s=_tuple_finite_or_none(getattr(getattr(model_v2, "orientationRate", None), "z", None)) if model_v2 is not None else (),
    lane_line_probs=_tuple_finite_or_none(getattr(model_v2, "laneLineProbs", None)) if model_v2 is not None else (),
    frame_drop_perc=frame_drop,
    desired_curvature_1_m=desired_curvature,
  )
  model_action = ModelActionEvidence(
    source=model_source,
    should_stop=_bool_or_none(getattr(action, "shouldStop", None)) if action is not None else None,
    desired_acceleration_mps2=_finite_or_none(getattr(action, "desiredAcceleration", None)) if action is not None else None,
    model_stop_distance_m=_model_stop_distance(model_v2) if model_v2 is not None else None,
  )
  if model_v2 is not None and action is None:
    model_action = ModelActionEvidence(source=_degraded(model_source, "missing_field"))
  elif action is not None and getattr(action, "desiredAcceleration", None) is not None and model_action.desired_acceleration_mps2 is None:
    model_action = ModelActionEvidence(source=_degraded(model_source, "nonfinite_field"), should_stop=model_action.should_stop, desired_acceleration_mps2=model_action.desired_acceleration_mps2, model_stop_distance_m=model_action.model_stop_distance_m)
  lead_one_raw = getattr(radar_state, "leadOne", None) if radar_state is not None else None
  lead_two_raw = getattr(radar_state, "leadTwo", None) if radar_state is not None else None
  lead_one = _lead_observation(lead_one_raw, ego.v_ego_mps, _lead_source(radar_source, getattr(lead_one_raw, "status", None) if lead_one_raw is not None else None))
  lead_two = _lead_observation(lead_two_raw, ego.v_ego_mps, _lead_source(radar_source, getattr(lead_two_raw, "status", None) if lead_two_raw is not None else None))
  final_statuses = (
    ("carState", _final_source(car_state is not None, source_statuses.get("carState"))),
    ("modelV2", _final_source(model_v2 is not None, source_statuses.get("modelV2"))),
    ("radarState", _final_source(radar_state is not None, source_statuses.get("radarState"))),
  )
  return EvidenceSnapshot(ego=ego, model_path=model_path, model_action=model_action,
                          lead=LeadEvidence(source=radar_source, lead_one=lead_one, lead_two=lead_two),
                          source_statuses=final_statuses)
