#!/usr/bin/env python3
from dataclasses import dataclass
import math


LEAD_CONFIDENCE_TRACK_UNKNOWN = -2
NEW_LEAD_GUARD_TIME = 0.35
NEW_LEAD_POS_ACCEL_HOLD_TIME = 0.25
NEW_LEAD_STABLE_TIME = 0.45
NEW_LEAD_CONTINUITY_MAX_D_REL_DELTA = 5.0
NEW_LEAD_CONTINUITY_MAX_V_LEAD_DELTA = 5.0
NEW_LEAD_CONTINUITY_MAX_Y_REL_DELTA = 1.0
NEW_LEAD_LATERAL_CHURN_Y_REL_MIN = 1.0


@dataclass(frozen=True)
class LeadConfidenceState:
  status: bool = False
  new_lead: bool = False
  stable: bool = False
  speed_trusted: bool = False
  radar: bool = False
  age: float = 0.0
  accel_blend: float = 0.0
  guard_timer: float = 0.0
  track_id: int = LEAD_CONFIDENCE_TRACK_UNKNOWN
  d_rel: float = 0.0
  v_lead: float = 0.0
  y_rel: float = 0.0


def _finite_float(value, default=0.0):
  try:
    value = float(value)
  except (TypeError, ValueError):
    return default
  return value if math.isfinite(value) else default


def _lead_track_id(lead):
  try:
    return int(getattr(lead, "radarTrackId", LEAD_CONFIDENCE_TRACK_UNKNOWN))
  except (TypeError, ValueError):
    return LEAD_CONFIDENCE_TRACK_UNKNOWN


def _lead_values(lead):
  return (
    _lead_track_id(lead),
    _finite_float(getattr(lead, "dRel", 0.0)),
    _finite_float(getattr(lead, "vLeadK", getattr(lead, "vLead", 0.0))),
    _finite_float(getattr(lead, "yRel", 0.0)),
    bool(getattr(lead, "radar", False)),
    _finite_float(getattr(lead, "modelProb", 0.0)),
  )


def _lead_continuity(prev_d_rel, d_rel, prev_v_lead, v_lead, prev_y_rel, y_rel):
  return (
    abs(d_rel - prev_d_rel) <= NEW_LEAD_CONTINUITY_MAX_D_REL_DELTA and
    abs(v_lead - prev_v_lead) <= NEW_LEAD_CONTINUITY_MAX_V_LEAD_DELTA and
    abs(y_rel - prev_y_rel) <= NEW_LEAD_CONTINUITY_MAX_Y_REL_DELTA
  )


def _positive_accel_blend(age):
  if age <= NEW_LEAD_POS_ACCEL_HOLD_TIME:
    return 0.0
  if age >= NEW_LEAD_STABLE_TIME:
    return 1.0
  return (age - NEW_LEAD_POS_ACCEL_HOLD_TIME) / (NEW_LEAD_STABLE_TIME - NEW_LEAD_POS_ACCEL_HOLD_TIME)


def adjust_new_lead_accel(a_lead, state: LeadConfidenceState):
  a_lead = _finite_float(a_lead)
  if a_lead <= 0.0:
    return a_lead
  return a_lead * state.accel_blend


class LeadConfidenceTracker:
  def __init__(self):
    self.track_id = LEAD_CONFIDENCE_TRACK_UNKNOWN
    self.d_rel = 0.0
    self.v_lead = 0.0
    self.y_rel = 0.0
    self.age = 0.0
    self.guard_timer = 0.0
    self.was_status = False

  def _is_continuous(self, track_id, d_rel, v_lead, y_rel):
    same_radar_track = track_id >= 0 and self.track_id >= 0 and track_id == self.track_id
    radarless_or_unknown = track_id < 0 or self.track_id < 0
    lateral_exit_churn = (
      track_id >= 0 and self.track_id >= 0 and track_id != self.track_id and
      abs(self.y_rel) >= NEW_LEAD_LATERAL_CHURN_Y_REL_MIN and
      abs(y_rel) >= NEW_LEAD_LATERAL_CHURN_Y_REL_MIN and
      math.copysign(1.0, self.y_rel) == math.copysign(1.0, y_rel)
    )
    motion_continuous = _lead_continuity(self.d_rel, d_rel, self.v_lead, v_lead, self.y_rel, y_rel)
    return motion_continuous and (same_radar_track or radarless_or_unknown or lateral_exit_churn)

  def update(self, lead, dt):
    dt = max(_finite_float(dt), 0.0)
    self.guard_timer = max(0.0, self.guard_timer - dt)

    if lead is None or not bool(getattr(lead, "status", False)):
      self.was_status = False
      self.track_id = LEAD_CONFIDENCE_TRACK_UNKNOWN
      self.age = 0.0
      return LeadConfidenceState(guard_timer=self.guard_timer)

    track_id, d_rel, v_lead, y_rel, radar, model_prob = _lead_values(lead)
    continuous = self.was_status and self._is_continuous(track_id, d_rel, v_lead, y_rel)
    new_lead = not continuous

    if new_lead:
      self.age = 0.0
      self.guard_timer = NEW_LEAD_GUARD_TIME
    else:
      self.age += dt

    self.was_status = True
    self.track_id = track_id
    self.d_rel = d_rel
    self.v_lead = v_lead
    self.y_rel = y_rel

    accel_blend = _positive_accel_blend(self.age)
    return LeadConfidenceState(
      status=True,
      new_lead=new_lead,
      stable=accel_blend >= 1.0,
      speed_trusted=radar or model_prob >= 0.5,
      radar=radar,
      age=self.age,
      accel_blend=accel_blend,
      guard_timer=self.guard_timer,
      track_id=track_id,
      d_rel=d_rel,
      v_lead=v_lead,
      y_rel=y_rel,
    )
