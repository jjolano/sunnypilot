#!/usr/bin/env python3
"""Profile lead-reaction behaviour: compare OP vs manual response to lead speed changes,
lead exits, and cut-ins.

Refined methodology vs prior baseline tools:

1. **Creep filtering**: ``brakePressed`` with ``vEgo > CREEP_SPEED`` is classified as
   *creeping*, not *stopped*.  This prevents creep-through-stop-and-go from contaminating
   launch / stop-hold metrics.

2. **Lead speed change events**: direction changes in ``aLeadK`` (lead acceleration), not
   raw ``vLead``.  Minimum magnitude and sustained-duration gates filter radar noise.

3. **Reaction time**: time from lead speed change to ego response.
   - Manual: ``aEgo`` direction change (actual vehicle deceleration, not pedal switches).
   - OP: ``longitudinalPlanSP.aTarget`` direction change (planner output, before actuator lag).

4. **Lead exits**: tracked lead (``radarTrackId``) disappears for > ``dropout_s``.
   Measures time-to-accelerate after the lead is gone.

5. **Cut-ins**: new ``radarTrackId`` appears with ``present=True`` inside ``cut_in_d_rel_max``.
   Measures time-to-brake and peak decel.

Run:
  uv run python -m openpilot.tools.drive_lab.profile_lead_reaction ROUTE --qlog
  uv run python -m openpilot.tools.drive_lab.profile_lead_reaction ROUTE --qlog --json --output /tmp/x.json
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from openpilot.tools.drive_lab.route_analysis import build_route_messages
from openpilot.tools.drive_lab.route_io import load_route_msgs, output_report
from openpilot.tools.drive_lab.timeline import safe_get


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LeadReactionParams:
  # Lead speed change detection
  a_lead_k_threshold: float = 0.3  # m/s^2; minimum |aLeadK| to register a direction change
  sustained_duration_s: float = 0.3  # minimum sustained duration for a speed change
  min_lead_v: float = 0.5  # m/s; ignore lead speed changes when lead is nearly stopped

  # Reaction measurement
  reaction_window_s: float = 5.0  # max window to look for ego response after lead change
  ego_response_threshold: float = 0.2  # m/s^2; minimum |aEgo| or |aTarget| change to count as response
  min_ego_speed: float = 1.0  # m/s; ignore reactions at very low speed (parking, etc.)
  already_responding_decel_threshold: float = -0.2  # m/s^2; aTarget/aEgo already this low on a decel lead-change
  already_responding_accel_threshold: float = 0.2  # m/s^2; aTarget/aEgo already this high on an accel lead-change
  already_responding_delta_threshold: float = 0.2  # m/s^2; baseline-to-current drop/rise before event counts as responding

  # Creep filtering
  creep_speed: float = 0.1  # m/s; brakePressed + vEgo > this = creeping, not stopped

  # Lead exit detection
  dropout_s: float = 0.5  # lead missing for this long = exit
  exit_accel_threshold: float = 0.3  # m/s^2; ego accel above this = "accelerated after exit"
  exit_window_s: float = 5.0  # window to look for accel response

  # Cut-in detection
  cut_in_d_rel_max: float = 30.0  # m; only count cut-ins within this distance
  cut_in_brake_threshold: float = -0.3  # m/s^2; ego decel below this = "braked for cut-in"
  cut_in_window_s: float = 5.0  # window to look for brake response
  cut_in_cluster_window_s: float = 1.0  # s; ignore repeated cut-in detections within window (lead-id churn)
  cut_in_event_gap_s: float = 5.0  # s; adjacent detected cut-ins within this gap are the same cluster in summaries
  cut_in_already_braking_threshold: float = -0.2  # m/s^2; ego already braking when cut-in appears
  cut_in_min_v_rel: float = -0.5  # m/s; require lead closing at least this fast (more negative = faster)
  cut_in_max_ttc_s: float = 8.0  # s; plausible time-to-collision when closing
  cut_in_min_stable_s: float = 0.15  # s; lead ID/presence must not churn immediately after detection
  cut_in_min_model_prob: float = 0.5  # minimum radar modelProb; rejects detector noise

  # Lead tracking
  min_model_prob: float = 0.0  # minimum modelProb (0 = accept all present=True leads)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class LeadSpeedChange:
  t: float
  direction: str  # "decel_to_accel" or "accel_to_decel"
  lead_v_before: float
  lead_v_after: float
  lead_a_peak: float
  d_rel: float
  v_ego: float
  op_engaged: bool
  lead_id: int | None


@dataclass
class ReactionEvent:
  lead_change: LeadSpeedChange
  reaction_time: float | None  # seconds from lead change to ego response
  ego_a_before: float
  ego_a_after: float
  response_type: str  # "accel", "brake", "none"
  already_responding: bool = False
  valid_reaction: bool = True


@dataclass
class LeadExitEvent:
  t: float
  lead_id: int | None
  d_rel: float
  v_ego: float
  op_engaged: bool
  accel_reaction_time: float | None
  peak_accel: float | None


@dataclass
class CutInEvent:
  t: float
  lead_id: int | None
  d_rel: float
  v_rel: float
  v_ego: float
  op_engaged: bool
  brake_reaction_time: float | None
  peak_decel: float | None
  already_braking: bool = False
  valid_reaction: bool = True


@dataclass
class LeadReactionReport:
  source: str
  duration_s: float
  op_engaged_s: float
  manual_moving_s: float
  creep_filtered_samples: int
  lead_speed_changes: list[LeadSpeedChange] = field(default_factory=list)
  op_reactions: list[ReactionEvent] = field(default_factory=list)
  manual_reactions: list[ReactionEvent] = field(default_factory=list)
  lead_exits: list[LeadExitEvent] = field(default_factory=list)
  cut_ins: list[CutInEvent] = field(default_factory=list)
  notes: list[str] = field(default_factory=list)

  def to_dict(self) -> dict[str, Any]:
    op_cut_ins = [e for e in self.cut_ins if e.op_engaged]
    manual_cut_ins = [e for e in self.cut_ins if not e.op_engaged]
    op_valid_cut_ins = [e for e in op_cut_ins if e.valid_reaction]
    manual_valid_cut_ins = [e for e in manual_cut_ins if e.valid_reaction]

    gap_s = LeadReactionParams().cut_in_event_gap_s
    op_clusters = _cluster_cut_ins(op_cut_ins, gap_s)
    manual_clusters = _cluster_cut_ins(manual_cut_ins, gap_s)
    op_canonical = [_canonical_cut_in(c) for c in op_clusters]
    manual_canonical = [_canonical_cut_in(c) for c in manual_clusters]

    # Clusters that are physically valid and actually produced a measured brake response.
    op_braking = [c for c in op_canonical if c.valid_reaction and c.brake_reaction_time is not None]
    manual_braking = [c for c in manual_canonical if c.valid_reaction and c.brake_reaction_time is not None]

    return {
      "source": self.source,
      "duration_s": _r(self.duration_s, 2),
      "op_engaged_s": _r(self.op_engaged_s, 2),
      "manual_moving_s": _r(self.manual_moving_s, 2),
      "creep_filtered_samples": self.creep_filtered_samples,
      "summary": {
        "op_reaction_median_s": _r(
          _opt_median([e.reaction_time for e in self.op_reactions if e.valid_reaction and e.reaction_time is not None]), 3
        ),
        "manual_reaction_median_s": _r(
          _opt_median([e.reaction_time for e in self.manual_reactions if e.valid_reaction and e.reaction_time is not None]), 3
        ),
        "op_reaction_count": len([e for e in self.op_reactions if e.valid_reaction and e.reaction_time is not None]),
        "manual_reaction_count": len([e for e in self.manual_reactions if e.valid_reaction and e.reaction_time is not None]),
        "op_already_responding_count": len([e for e in self.op_reactions if e.already_responding]),
        "manual_already_responding_count": len([e for e in self.manual_reactions if e.already_responding]),
        "op_lead_exit_accel_median_s": _r(
          _opt_median([e.accel_reaction_time for e in self.lead_exits if e.op_engaged and e.accel_reaction_time is not None]),
          3,
        ),
        "manual_lead_exit_accel_median_s": _r(
          _opt_median([e.accel_reaction_time for e in self.lead_exits if not e.op_engaged and e.accel_reaction_time is not None]),
          3,
        ),
        "op_cut_in_brake_median_s": _r(
          _opt_median([e.brake_reaction_time for e in op_braking if e.brake_reaction_time is not None]), 3
        ),
        "manual_cut_in_brake_median_s": _r(
          _opt_median([e.brake_reaction_time for e in manual_braking if e.brake_reaction_time is not None]), 3
        ),
        "op_cut_in_peak_decel_median": _r(
          _opt_median([e.peak_decel for e in op_braking if e.peak_decel is not None and e.peak_decel < 0]), 3
        ),
        "manual_cut_in_peak_decel_median": _r(
          _opt_median([e.peak_decel for e in manual_braking if e.peak_decel is not None and e.peak_decel < 0]), 3
        ),
        # Event-level counts (preserved)
        "op_cut_in_count": len(op_cut_ins),
        "manual_cut_in_count": len(manual_cut_ins),
        "op_cut_in_valid_count": len(op_valid_cut_ins),
        "manual_cut_in_valid_count": len(manual_valid_cut_ins),
        "op_cut_in_already_braking_count": len([e for e in op_cut_ins if e.already_braking]),
        "manual_cut_in_already_braking_count": len([e for e in manual_cut_ins if e.already_braking]),
        # Cluster-level counts (adjacent events within cut_in_event_gap_s)
        "op_cut_in_cluster_count": len(op_clusters),
        "manual_cut_in_cluster_count": len(manual_clusters),
        "op_cut_in_valid_cluster_count": len([c for c in op_canonical if c.valid_reaction]),
        "manual_cut_in_valid_cluster_count": len([c for c in manual_canonical if c.valid_reaction]),
        "op_cut_in_already_braking_cluster_count": len([c for c in op_canonical if c.already_braking]),
        "manual_cut_in_already_braking_cluster_count": len([c for c in manual_canonical if c.already_braking]),
      },
      "lead_speed_changes": [_change_to_dict(c) for c in self.lead_speed_changes],
      "op_reactions": [_reaction_to_dict(e) for e in self.op_reactions],
      "manual_reactions": [_reaction_to_dict(e) for e in self.manual_reactions],
      "lead_exits": [_exit_to_dict(e) for e in self.lead_exits],
      "cut_ins": [_cutin_to_dict(e) for e in self.cut_ins],
      "notes": list(self.notes),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _r(value: Any, ndigits: int = 6) -> float | None:
  if value is None or not isinstance(value, int | float) or not math.isfinite(float(value)):
    return None
  return round(float(value), ndigits)


def _f(value: Any) -> float:
  try:
    out = float(value)
  except (TypeError, ValueError):
    return math.nan
  return out if math.isfinite(out) else math.nan


def _opt_median(values: list[float]) -> float | None:
  return float(np.median(values)) if values else None


def _change_to_dict(c: LeadSpeedChange) -> dict[str, Any]:
  return {
    "t": _r(c.t, 3),
    "direction": c.direction,
    "lead_v_before": _r(c.lead_v_before, 3),
    "lead_v_after": _r(c.lead_v_after, 3),
    "lead_a_peak": _r(c.lead_a_peak, 3),
    "d_rel": _r(c.d_rel, 2),
    "v_ego": _r(c.v_ego, 3),
    "op_engaged": c.op_engaged,
    "lead_id": c.lead_id,
  }


def _reaction_to_dict(e: ReactionEvent) -> dict[str, Any]:
  return {
    "t": _r(e.lead_change.t, 3),
    "direction": e.lead_change.direction,
    "reaction_time": _r(e.reaction_time, 3),
    "ego_a_before": _r(e.ego_a_before, 3),
    "ego_a_after": _r(e.ego_a_after, 3),
    "response_type": e.response_type,
    "op_engaged": e.lead_change.op_engaged,
    "already_responding": e.already_responding,
    "valid_reaction": e.valid_reaction,
  }


def _exit_to_dict(e: LeadExitEvent) -> dict[str, Any]:
  return {
    "t": _r(e.t, 3),
    "lead_id": e.lead_id,
    "d_rel": _r(e.d_rel, 2),
    "v_ego": _r(e.v_ego, 3),
    "op_engaged": e.op_engaged,
    "accel_reaction_time": _r(e.accel_reaction_time, 3),
    "peak_accel": _r(e.peak_accel, 3),
  }


def _cutin_to_dict(e: CutInEvent) -> dict[str, Any]:
  return {
    "t": _r(e.t, 3),
    "lead_id": e.lead_id,
    "d_rel": _r(e.d_rel, 2),
    "v_rel": _r(e.v_rel, 3),
    "v_ego": _r(e.v_ego, 3),
    "op_engaged": e.op_engaged,
    "already_braking": e.already_braking,
    "valid_reaction": e.valid_reaction,
    "brake_reaction_time": _r(e.brake_reaction_time, 3),
    "peak_decel": _r(e.peak_decel, 3),
  }


# ---------------------------------------------------------------------------
# Core collection
# ---------------------------------------------------------------------------


@dataclass
class _RadarSample:
  t: float
  d_rel: float
  v_lead: float  # vLeadK (filtered)
  v_rel: float
  a_lead: float  # aLeadK
  status: bool
  lead_id: int | None
  model_prob: float


@dataclass
class _EgoSample:
  t: float
  v: float
  a: float  # aEgo
  a_target: float | None  # longitudinalPlanSP.aTarget (OP only)
  long_active: bool
  brake_pressed: bool
  gas_pressed: bool


def _collect(msgs: list[Any], p: LeadReactionParams) -> tuple[list[_RadarSample], list[_EgoSample], float, float, int]:
  radar: list[_RadarSample] = []
  ego: list[_EgoSample] = []
  op_engaged_s = 0.0
  manual_moving_s = 0.0
  creep_filtered = 0
  last_t: float | None = None
  latest: dict[str, Any] = {}

  for rec in build_route_messages(msgs):
    if rec.typ in ("carState", "carControl", "radarState", "longitudinalPlanSP"):
      latest[rec.typ] = rec.payload

    if rec.typ == "carState":
      cs = rec.payload
      cc = latest.get("carControl")
      sp = latest.get("longitudinalPlanSP")
      v = _f(safe_get(cs, "vEgo"))
      a = _f(safe_get(cs, "aEgo"))
      brake = bool(safe_get(cs, "brakePressed", False))
      gas = bool(safe_get(cs, "gasPressed", False))
      long_active = bool(safe_get(cc, "longActive", False))
      a_target = _f(safe_get(sp, "aTarget")) if sp is not None else None

      if last_t is not None and math.isfinite(v):
        dt = max(0.0, min(2.0, rec.t - last_t))
        if long_active:
          op_engaged_s += dt
        elif v > p.creep_speed:
          manual_moving_s += dt
        # Creep filtering: brakePressed + vEgo > creep_speed = creeping
        if brake and v > p.creep_speed:
          creep_filtered += 1
      last_t = rec.t

      if math.isfinite(v) and math.isfinite(a):
        ego.append(
          _EgoSample(
            t=rec.t,
            v=v,
            a=a,
            a_target=a_target if (a_target is not None and math.isfinite(a_target)) else None,
            long_active=long_active,
            brake_pressed=brake,
            gas_pressed=gas,
          )
        )

    elif rec.typ == "radarState":
      lead = safe_get(rec.payload, "leadOne")
      if lead is None:
        continue
      status = bool(safe_get(lead, "present", False))
      d_rel = _f(safe_get(lead, "dRel"))
      v_lead = _f(safe_get(lead, "vLeadK", safe_get(lead, "vLead")))
      v_rel = _f(safe_get(lead, "vRel"))
      a_lead = _f(safe_get(lead, "aLeadK"))
      lead_id = safe_get(lead, "radarTrackId")
      model_prob = _f(safe_get(lead, "modelProb"))
      if not (math.isfinite(d_rel) and math.isfinite(v_lead) and math.isfinite(a_lead)):
        continue
      radar.append(
        _RadarSample(
          t=rec.t,
          d_rel=d_rel,
          v_lead=v_lead,
          v_rel=v_rel,
          a_lead=a_lead,
          status=status,
          lead_id=lead_id,
          model_prob=model_prob if math.isfinite(model_prob) else 0.0,
        )
      )

  return radar, ego, op_engaged_s, manual_moving_s, creep_filtered


# ---------------------------------------------------------------------------
# Lead speed change detection
# ---------------------------------------------------------------------------


def _detect_lead_speed_changes(radar: list[_RadarSample], ego: list[_EgoSample], p: LeadReactionParams) -> list[LeadSpeedChange]:
  changes: list[LeadSpeedChange] = []
  if len(radar) < 5:
    return changes

  # Build aLeadK sign sequence with time
  prev_sign = 0
  sustained_since: float | None = None
  sustained_a: list[float] = []
  sustained_t: list[float] = []

  for i in range(1, len(radar)):
    s = radar[i]
    if not s.status or s.v_lead < p.min_lead_v:
      prev_sign = 0
      sustained_since = None
      sustained_a = []
      sustained_t = []
      continue

    a = s.a_lead
    if not math.isfinite(a):
      continue
    cur_sign = 1 if a > p.a_lead_k_threshold else (-1 if a < -p.a_lead_k_threshold else 0)

    if cur_sign != 0 and cur_sign != prev_sign and prev_sign != 0:
      # Direction change detected
      if sustained_since is not None and (s.t - sustained_since) >= p.sustained_duration_s:
        # Find ego state at this time
        ego_idx = _nearest_index(ego, s.t)
        op_engaged = ego[ego_idx].long_active if ego_idx is not None else False
        v_ego = ego[ego_idx].v if ego_idx is not None else float("nan")
        prev_sample = radar[i - 1] if i > 0 else radar[0]
        direction = "decel_to_accel" if cur_sign > 0 else "accel_to_decel"
        peak_a = max(sustained_a, key=abs) if sustained_a else a
        changes.append(
          LeadSpeedChange(
            t=s.t,
            direction=direction,
            lead_v_before=prev_sample.v_lead,
            lead_v_after=s.v_lead,
            lead_a_peak=peak_a,
            d_rel=s.d_rel,
            v_ego=v_ego,
            op_engaged=op_engaged,
            lead_id=s.lead_id,
          )
        )
      sustained_since = s.t
      sustained_a = [a]
      sustained_t = [s.t]
    elif cur_sign != 0:
      if sustained_since is None:
        sustained_since = s.t
        sustained_a = [a]
        sustained_t = [s.t]
      else:
        sustained_a.append(a)
        sustained_t.append(s.t)

    if cur_sign != 0:
      prev_sign = cur_sign

  return changes


def _nearest_index(samples: list, t: float) -> int | None:
  if not samples:
    return None
  lo, hi = 0, len(samples) - 1
  if t <= samples[0].t:
    return 0
  if t >= samples[-1].t:
    return len(samples) - 1
  while lo < hi:
    mid = (lo + hi) // 2
    if samples[mid].t < t:
      lo = mid + 1
    else:
      hi = mid
  return lo


def _latest_index_at_or_before(samples: list, t: float) -> int | None:
  idx = _nearest_index(samples, t)
  if idx is None:
    return None
  if samples[idx].t > t and idx > 0:
    return idx - 1
  return idx


# ---------------------------------------------------------------------------
# Reaction measurement
# ---------------------------------------------------------------------------


def _measure_reaction(ego: list[_EgoSample], change: LeadSpeedChange, p: LeadReactionParams) -> ReactionEvent:
  """Measure ego reaction time after a lead speed change; exclude cases already responding at event time."""
  idx = _nearest_index(ego, change.t)
  if idx is None:
    return ReactionEvent(change, None, 0.0, 0.0, "none", already_responding=False, valid_reaction=True)

  # Use the latest ego sample at or before the event so we don't peek into a response.
  event_idx = _latest_index_at_or_before(ego, change.t) or idx
  event_ego = ego[event_idx]

  # For OP, use aTarget; for manual, use aEgo
  use_target = change.op_engaged

  # Signal at event time
  current_signal = event_ego.a_target if (use_target and event_ego.a_target is not None) else event_ego.a
  if current_signal is None or not math.isfinite(current_signal):
    current_signal = 0.0

  # Baseline from samples strictly before the event
  baseline_a = ego[max(0, event_idx - 3) : event_idx]
  baseline_a = [e.a for e in baseline_a if math.isfinite(e.a)]
  a_before = float(np.median(baseline_a)) if baseline_a else 0.0

  # Detect whether ego/planner was already responding before the marker.
  if change.direction == "decel_to_accel":
    already_responding = (
      current_signal >= p.already_responding_accel_threshold
      or (current_signal - a_before) >= p.already_responding_delta_threshold
    )
  else:
    already_responding = (
      current_signal <= p.already_responding_decel_threshold
      or (a_before - current_signal) >= p.already_responding_delta_threshold
    )

  # Look for direction response in the window
  end_t = change.t + p.reaction_window_s
  response_time = None
  a_after = current_signal
  response_type = "none"

  if not already_responding:
    for i in range(idx, len(ego)):
      if ego[i].t > end_t:
        break
      raw_signal = ego[i].a_target if (use_target and ego[i].a_target is not None) else ego[i].a
      if raw_signal is None or not math.isfinite(raw_signal):
        continue
      signal = float(raw_signal)
      delta = signal - a_before
      if change.direction == "decel_to_accel":
        # Lead is accelerating; ego should accelerate
        if delta > p.ego_response_threshold:
          response_time = float(ego[i].t - change.t)
          a_after = signal
          response_type = "accel"
          break
      else:
        # Lead is decelerating; ego should brake
        if delta < -p.ego_response_threshold:
          response_time = float(ego[i].t - change.t)
          a_after = signal
          response_type = "brake"
          break

  return ReactionEvent(
    change,
    response_time,
    a_before,
    float(a_after),
    response_type,
    already_responding=already_responding,
    valid_reaction=not already_responding,
  )


# ---------------------------------------------------------------------------
# Lead exit detection
# ---------------------------------------------------------------------------


def _detect_lead_exits(radar: list[_RadarSample], ego: list[_EgoSample], p: LeadReactionParams) -> list[LeadExitEvent]:
  exits: list[LeadExitEvent] = []
  if len(radar) < 3:
    return exits

  prev_id: int | None = None
  prev_status = False
  prev_d_rel = 0.0
  dropout_start: float | None = None
  exit_id: int | None = None
  exit_d_rel = 0.0

  for s in radar:
    if prev_status and not s.status:
      # Lead just disappeared
      dropout_start = s.t
      exit_id = prev_id
      exit_d_rel = prev_d_rel
    elif not prev_status and s.status and dropout_start is not None:
      # Lead reappeared — check if this was a real exit (> dropout_s)
      if s.t - dropout_start >= p.dropout_s:
        # Also check if it's a different ID (real exit, not brief dropout)
        if exit_id is not None and s.lead_id is not None and s.lead_id != exit_id:
          _record_exit(ego, exits, dropout_start, exit_id, exit_d_rel, p)
        elif exit_id is not None and s.lead_id is not None and s.lead_id == exit_id:
          # Same lead reappeared after long dropout — still count as exit
          _record_exit(ego, exits, dropout_start, exit_id, exit_d_rel, p)
      dropout_start = None
    elif prev_status and s.status and prev_id is not None and s.lead_id is not None and s.lead_id != prev_id:
      # Lead ID changed while status stayed True — previous lead exited
      _record_exit(ego, exits, s.t, prev_id, prev_d_rel, p)

    prev_id = s.lead_id
    prev_status = s.status
    prev_d_rel = s.d_rel

  # Handle end-of-route dropout: if lead was missing at the end, record the exit
  if dropout_start is not None and exit_id is not None:
    _record_exit(ego, exits, dropout_start, exit_id, exit_d_rel, p)

  return exits


def _record_exit(ego: list[_EgoSample], exits: list[LeadExitEvent], t: float, lead_id: int | None, d_rel: float, p: LeadReactionParams) -> None:
  idx = _nearest_index(ego, t)
  if idx is None:
    return
  v_ego = ego[idx].v
  op_engaged = ego[idx].long_active
  if not math.isfinite(v_ego) or v_ego < p.min_ego_speed:
    return

  # Measure time to accelerate after exit
  end_t = t + p.exit_window_s
  accel_time = None
  peak_accel = None
  baseline_a = ego[max(0, idx - 3) : idx + 1]
  baseline_a = [e.a for e in baseline_a if math.isfinite(e.a)]
  a_before = float(np.median(baseline_a)) if baseline_a else 0.0

  for i in range(idx, len(ego)):
    if ego[i].t > end_t:
      break
    raw_signal = ego[i].a_target if (op_engaged and ego[i].a_target is not None) else ego[i].a
    if raw_signal is None or not math.isfinite(raw_signal):
      continue
    signal = float(raw_signal)
    if peak_accel is None or signal > peak_accel:
      peak_accel = signal
    if signal - a_before > p.exit_accel_threshold and accel_time is None:
      accel_time = float(ego[i].t - t)

  exits.append(
    LeadExitEvent(
      t=t,
      lead_id=lead_id,
      d_rel=d_rel,
      v_ego=v_ego,
      op_engaged=op_engaged,
      accel_reaction_time=accel_time,
      peak_accel=peak_accel,
    )
  )


# ---------------------------------------------------------------------------
# Cut-in detection
# ---------------------------------------------------------------------------


def _cut_in_stable(radar: list[_RadarSample], start_idx: int, event_t: float, event_id: int | None, p: LeadReactionParams) -> bool:
  """Return True if the same lead ID and status remain True for at least cut_in_min_stable_s."""
  if event_id is None:
    return False
  for j in range(start_idx + 1, len(radar)):
    if radar[j].t > event_t + p.cut_in_min_stable_s:
      # Reached end of the stability window without finding a churn, so it's stable.
      return True
    if not radar[j].status or radar[j].lead_id != event_id:
      return False
  # No churn and no more samples; be conservative and call it unstable.
  return False


def _cluster_cut_ins(cut_ins: list[CutInEvent], gap_s: float) -> list[list[CutInEvent]]:
  """Group detected cut-ins into clusters where adjacent events are within gap_s."""
  sorted_events = sorted(cut_ins, key=lambda e: e.t)
  clusters: list[list[CutInEvent]] = []
  for e in sorted_events:
    if not clusters or (e.t - clusters[-1][-1].t) > gap_s:
      clusters.append([e])
    else:
      clusters[-1].append(e)
  return clusters


def _canonical_cut_in(cluster: list[CutInEvent]) -> CutInEvent:
  """Pick one representative event per cluster for summary medians/counts.

  Prefer the first event that is physically valid and has a measured brake reaction.
  Fall back to the first valid event, then the first event in the cluster.
  """
  for e in cluster:
    if e.valid_reaction and e.brake_reaction_time is not None:
      return e
  for e in cluster:
    if e.valid_reaction:
      return e
  return cluster[0]


def _detect_cut_ins(radar: list[_RadarSample], ego: list[_EgoSample], p: LeadReactionParams) -> list[CutInEvent]:
  cut_ins: list[CutInEvent] = []
  if len(radar) < 3:
    return cut_ins

  prev_id: int | None = None
  prev_status = False
  last_cut_in_t: float | None = None

  for i, s in enumerate(radar):
    if not s.status or s.d_rel > p.cut_in_d_rel_max or s.d_rel <= 0:
      prev_id = s.lead_id
      prev_status = s.status
      continue

    is_new = False
    if not prev_status and s.status:
      is_new = True
    elif prev_status and s.status and prev_id is not None and s.lead_id is not None and s.lead_id != prev_id:
      is_new = True

    # Cluster repeated cut-in detections (e.g., alternating radar lead IDs)
    # within the same physical event.
    if is_new and last_cut_in_t is not None and (s.t - last_cut_in_t) < p.cut_in_cluster_window_s:
      is_new = False

    if is_new:
      idx = _nearest_index(ego, s.t)
      if idx is not None:
        event_idx = _latest_index_at_or_before(ego, s.t) or idx
        event_ego = ego[event_idx]
        v_ego = event_ego.v
        op_engaged = event_ego.long_active
        if math.isfinite(v_ego) and v_ego >= p.min_ego_speed:
          # Stricter event validity: real closing, plausible TTC, stable lead ID, model confidence.
          closing = math.isfinite(s.v_rel) and s.v_rel <= p.cut_in_min_v_rel
          ttc_ok = False
          if closing and s.v_rel < 0:
            ttc = -s.d_rel / s.v_rel
            ttc_ok = 0.0 < ttc <= p.cut_in_max_ttc_s
          stable = _cut_in_stable(radar, i, s.t, s.lead_id, p)
          model_ok = math.isfinite(s.model_prob) and s.model_prob >= p.cut_in_min_model_prob

          # Signal at/before event time: planner target when OP is engaged, otherwise actual ego accel.
          # Avoid peeking one carState sample into the future, which would classify an immediate
          # response as "already braking" instead of a fast reaction.
          current_signal = event_ego.a_target if (op_engaged and event_ego.a_target is not None) else event_ego.a
          if current_signal is None or not math.isfinite(current_signal):
            current_signal = 0.0

          # Baseline ego accel before the cut-in
          baseline_a = ego[max(0, event_idx - 3) : event_idx + 1]
          baseline_a = [e.a for e in baseline_a if math.isfinite(e.a)]
          a_before = float(np.median(baseline_a)) if baseline_a else 0.0

          # Determine whether the ego was already braking before this cut-in.
          already_braking = current_signal <= p.cut_in_already_braking_threshold or (a_before - current_signal) >= abs(p.cut_in_already_braking_threshold)

          valid_reaction = (not already_braking) and closing and ttc_ok and stable and model_ok

          # Measure time to brake and peak decel over the response window
          end_t = s.t + p.cut_in_window_s
          brake_time = None
          peak_decel = None

          for j in range(idx, len(ego)):
            if ego[j].t > end_t:
              break
            raw_signal = ego[j].a_target if (op_engaged and ego[j].a_target is not None) else ego[j].a
            if raw_signal is None or not math.isfinite(raw_signal):
              continue
            signal = float(raw_signal)
            if peak_decel is None or signal < peak_decel:
              peak_decel = signal
            if valid_reaction and (signal - a_before < p.cut_in_brake_threshold) and brake_time is None:
              brake_time = float(ego[j].t - s.t)

          if already_braking:
            brake_time = None

          cut_ins.append(
            CutInEvent(
              t=s.t,
              lead_id=s.lead_id,
              d_rel=s.d_rel,
              v_rel=s.v_rel,
              v_ego=v_ego,
              op_engaged=op_engaged,
              brake_reaction_time=brake_time,
              peak_decel=peak_decel,
              already_braking=already_braking,
              valid_reaction=valid_reaction,
            )
          )
          last_cut_in_t = s.t

    prev_id = s.lead_id
    prev_status = s.status

  return cut_ins


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------


def analyze_route(msgs: list[Any], source: str = "unknown", params: LeadReactionParams | None = None) -> LeadReactionReport:
  p = params or LeadReactionParams()
  radar, ego, op_engaged_s, manual_moving_s, creep_filtered = _collect(msgs, p)
  notes: list[str] = []

  if not radar:
    notes.append("no radarState samples with valid lead data found")
  if not ego:
    notes.append("no carState samples found")

  duration = 0.0
  if ego:
    duration = ego[-1].t - ego[0].t

  changes = _detect_lead_speed_changes(radar, ego, p)
  reactions = [_measure_reaction(ego, c, p) for c in changes]
  op_reactions = [r for r in reactions if r.lead_change.op_engaged]
  manual_reactions = [r for r in reactions if not r.lead_change.op_engaged]

  exits = _detect_lead_exits(radar, ego, p)
  cut_ins = _detect_cut_ins(radar, ego, p)

  return LeadReactionReport(
    source=source,
    duration_s=duration,
    op_engaged_s=op_engaged_s,
    manual_moving_s=manual_moving_s,
    creep_filtered_samples=creep_filtered,
    lead_speed_changes=changes,
    op_reactions=op_reactions,
    manual_reactions=manual_reactions,
    lead_exits=exits,
    cut_ins=cut_ins,
    notes=notes,
  )


def render_report(report: LeadReactionReport) -> str:
  lines = [
    f"Lead-reaction profile: {report.source}",
    f"  duration {report.duration_s:.1f}s, OP engaged {report.op_engaged_s:.1f}s, " + f"manual moving {report.manual_moving_s:.1f}s",
    f"  creep-filtered samples (brakePressed + vEgo > 0.1): {report.creep_filtered_samples}",
  ]

  s = report.to_dict()["summary"]
  lines += [
    "",
    "  lead speed change reaction (median):",
    f"    OP:      {s['op_reaction_median_s']} s  ({s['op_reaction_count']} valid events, {s['op_already_responding_count']} already responding)",
    f"    manual:  {s['manual_reaction_median_s']} s  ({s['manual_reaction_count']} valid events, {s['manual_already_responding_count']} already responding)",
    "",
    "  lead exit accel reaction (median):",
    f"    OP:      {s['op_lead_exit_accel_median_s']} s",
    f"    manual:  {s['manual_lead_exit_accel_median_s']} s",
    "",
    "  cut-in brake reaction (median):",
    f"    OP:      {s['op_cut_in_brake_median_s']} s  peak decel {s['op_cut_in_peak_decel_median']} m/s^2",
    f"             ({s['op_cut_in_valid_cluster_count']} valid / "
    + f"{s['op_cut_in_cluster_count']} clusters, {s['op_cut_in_already_braking_cluster_count']} already braking)",
    f"    manual:  {s['manual_cut_in_brake_median_s']} s  peak decel {s['manual_cut_in_peak_decel_median']} m/s^2",
    f"             ({s['manual_cut_in_valid_cluster_count']} valid / "
    + f"{s['manual_cut_in_cluster_count']} clusters, {s['manual_cut_in_already_braking_cluster_count']} already braking)",
  ]

  # Per-event detail for small counts
  if len(report.op_reactions) <= 10:
    for e in report.op_reactions:
      lines.append(f"    OP  t={e.lead_change.t:.1f} {e.lead_change.direction} reaction={_fmt(e.reaction_time, 2)}s type={e.response_type}")
  if len(report.manual_reactions) <= 10:
    for e in report.manual_reactions:
      lines.append(f"    MAN t={e.lead_change.t:.1f} {e.lead_change.direction} reaction={_fmt(e.reaction_time, 2)}s type={e.response_type}")
  if len(report.lead_exits) <= 10:
    for e in report.lead_exits:
      tag = "OP" if e.op_engaged else "MAN"
      lines.append(f"    {tag} exit t={e.t:.1f} id={e.lead_id} d={e.d_rel:.1f}m accel={_fmt(e.accel_reaction_time, 2)}s peak={_fmt(e.peak_accel, 2)}")
  if len(report.cut_ins) <= 10:
    for e in report.cut_ins:
      tag = "OP" if e.op_engaged else "MAN"
      lines.append(
        f"    {tag} cut-in t={e.t:.1f} id={e.lead_id} d={e.d_rel:.1f}m vR={e.v_rel:.1f} "
        + f"brake={_fmt(e.brake_reaction_time, 2)}s peak={_fmt(e.peak_decel, 2)}"
      )

  for note in report.notes:
    lines.append(f"  note: {note}")
  return "\n".join(lines)


def _fmt(value: float | None, ndigits: int = 3) -> str:
  return "n/a" if value is None or not math.isfinite(float(value)) else f"{float(value):.{ndigits}f}"


def main() -> None:
  parser = argparse.ArgumentParser(description="Profile lead-reaction behaviour: OP vs manual.")
  parser.add_argument("route", help="Route, segment range, log file, or URL accepted by LogReader")
  parser.add_argument("--output", help="Write the report JSON to this path")
  parser.add_argument("--json", action="store_true", help="Print JSON instead of the text summary")
  parser.add_argument("--qlog", action="store_true", help="Prefer qlogs instead of rlogs (lower rate)")
  args = parser.parse_args()

  msgs = load_route_msgs(args.route, qlog=args.qlog)
  report = analyze_route(msgs, source=args.route)
  print(output_report(report, json_output=args.json, renderer=render_report, output_path=args.output))


if __name__ == "__main__":
  main()
