#!/usr/bin/env python3
"""General per-decision signal scorer for longitudinal disagreements.

The longitudinal disagreement pipeline needs to know *which* of the planner's
many signals (model.shouldStop, model.desiredAcceleration thresholds,
planner.shouldStop, planner.source, lead dRel/vRel, traffic-control, etc.) is
the right one to key a decision on. This module generalizes the recall/timely
scorecard already used by `profile_leadless_stops.signal_active()` so it can
score signals for any decision category: stop/go, opposite-intent, lead
transition.

Each signal evaluator takes a per-timestep ``SignalSample`` and returns whether
the signal is active. Each category labeler takes a sample stream and emits
ground-truth ``DecisionEpisode`` records anchored on the human action. The
scorer compares them on recall (did the signal fire on a real human action?),
precision/false-alarm rate (did it stay silent otherwise?), lead-time (how
early vs the human?), and stability (flicker/lag).

The output answers the question: "if we re-tuned the decision to use signal X
with threshold Y, how often would we agree with the human, and how often
would we over-react?"

Reads-only. No driving behavior changes.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from math import isfinite
from typing import Any, Callable, Iterable

from openpilot.tools.drive_lab import profile_leadless_stops
from openpilot.tools.drive_lab.planner_target_analysis import (
  MOVING_SPEED,
  PlannerTargetSample,
  is_opposite_intent,
)
from openpilot.tools.drive_lab.profile_leadless_stops import LeadlessStopSample
from openpilot.tools.drive_lab.shadow_manual_longitudinal import (
  REQUIRED_SHADOW_SERVICES,
  ShadowReplayOptions,
  extract_shadow_samples,
)
from openpilot.tools.drive_lab.route_analysis import iter_route_messages, route_identity
from openpilot.tools.drive_lab.timeline import safe_get
from openpilot.tools.lib.logreader import LogReader, ReadMode


SignalEval = Callable[["SignalSample"], bool]
Labeler = Callable[[list["SignalSample"]], list["DecisionEpisode"]]
MOVING_SPEED_FLOOR = MOVING_SPEED
PLAN_DIRECTION_THRESHOLD = 0.35
DRIVER_DIRECTION_THRESHOLD = 0.35
DEFAULT_TIMELY_GRACE_S = 1.5
DEFAULT_SIGNAL_LOOKBACK_S = 8.0
LEADLESS_DEFAULT_DECEL = -0.3
LEADLESS_MIN_DURATION_S = 0.6


@dataclass(frozen=True)
class SignalSample:
  route: str
  route_id: str
  segment: int | None
  t: float
  v_ego: float
  a_ego: float
  gas_pressed: bool
  brake_pressed: bool
  standstill: bool
  selfdrive_active: bool
  long_active: bool
  plan_a_target: float | None
  plan_should_stop: bool
  plan_source: str
  model_should_stop: bool
  model_desired_accel: float | None
  model_stop_distance: float | None
  model_endpoint_x: float | None
  model_endpoint_v: float | None
  lead_status: bool
  lead_d_rel: float | None
  lead_v_rel: float | None
  sp_a_target: float | None
  sp_source: str
  sp_stack: str
  fcw: bool

  def direction_plan(self) -> str:
    pt = self.plan_a_target
    if pt is None:
      return "neutral"
    if pt <= -PLAN_DIRECTION_THRESHOLD:
      return "brake"
    if pt >= PLAN_DIRECTION_THRESHOLD:
      return "accel"
    return "neutral"

  def direction_driver(self) -> str:
    if self.brake_pressed or self.a_ego <= -DRIVER_DIRECTION_THRESHOLD:
      return "brake"
    if self.gas_pressed or self.a_ego >= DRIVER_DIRECTION_THRESHOLD:
      return "accel"
    return "neutral"

  def is_opposite_intent(self) -> bool:
    p, d = self.direction_plan(), self.direction_driver()
    return (p == "brake" and d == "accel") or (p == "accel" and d == "brake")


@dataclass(frozen=True)
class DecisionEpisode:
  category: str
  route: str
  route_id: str
  segment: int | None
  decision_time_s: float
  start_time_s: float
  end_time_s: float
  duration_s: float
  kind: str
  v_ego: float
  a_ego: float
  active_ratio: float
  long_active_ratio: float
  extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SignalMetric:
  category: str
  signal: str
  episodes: int
  hit_count: int
  timely_hit_count: int
  missed_count: int
  fire_count: int
  recall: float
  recall_timely: float
  precision: float
  true_positives: int
  false_positives: int
  false_alarm_count: int
  false_alarm_windows: int
  median_lead_time_s: float | None
  p10_lead_time_s: float | None
  flicker_count: int
  onset_lag_s: float | None
  notes: str = ""


@dataclass(frozen=True)
class CategoryScorecard:
  category: str
  episode_count: int
  signals: dict[str, SignalMetric]
  notes: str = ""


@dataclass(frozen=True)
class ScorerSummary:
  route_count: int
  sample_count: int
  episode_count: int
  category_count: int
  signal_count: int
  categories: list[CategoryScorecard]


def planner_target_sample_to_signal_sample(sample: PlannerTargetSample, *, fallback_shadow_sample: LeadlessStopSample | None = None) -> SignalSample:
  return SignalSample(
    route=sample.route,
    route_id=sample.route_id,
    segment=sample.segment,
    t=sample.t,
    v_ego=sample.v_ego,
    a_ego=sample.a_ego,
    gas_pressed=sample.gas_pressed,
    brake_pressed=sample.brake_pressed,
    standstill=sample.standstill,
    selfdrive_active=sample.selfdrive_active,
    long_active=sample.long_active,
    plan_a_target=sample.plan_a_target,
    plan_should_stop=sample.plan_should_stop,
    plan_source=str(sample.plan_source),
    model_should_stop=sample.model_should_stop,
    model_desired_accel=sample.model_desired_accel,
    model_stop_distance=getattr(fallback_shadow_sample, "model_stop_distance", None) if fallback_shadow_sample else None,
    model_endpoint_x=getattr(fallback_shadow_sample, "model_endpoint_x", None) if fallback_shadow_sample else None,
    model_endpoint_v=getattr(fallback_shadow_sample, "model_endpoint_v", None) if fallback_shadow_sample else None,
    lead_status=sample.lead_status,
    lead_d_rel=sample.lead_d_rel,
    lead_v_rel=sample.lead_v_rel,
    sp_a_target=sample.sp_a_target,
    sp_source=str(sample.sp_source),
    sp_stack=str(sample.sp_stack),
    fcw=sample.plan_fcw,
  )


def _mirror_leadless_sample(sample: PlannerTargetSample) -> LeadlessStopSample:
  return LeadlessStopSample(
    route=sample.route,
    route_id=sample.route_id,
    segment=sample.segment,
    t=sample.t,
    v_ego=sample.v_ego,
    a_ego=sample.a_ego,
    gas_pressed=sample.gas_pressed,
    brake_pressed=sample.brake_pressed,
    standstill=sample.standstill,
    selfdrive_active=sample.selfdrive_active,
    long_active=sample.long_active,
    long_control_state="pid",
    lead_status=sample.lead_status,
    lead_d_rel=sample.lead_d_rel,
    lead_v_rel=sample.lead_v_rel,
    model_should_stop=sample.model_should_stop,
    model_desired_accel=sample.model_desired_accel,
    model_stop_distance=None,
    model_endpoint_x=None,
    model_endpoint_v=None,
    scc_early_model_stop=False,
    plan_should_stop=sample.plan_should_stop,
    plan_source=str(sample.plan_source),
    plan_a_target=sample.plan_a_target,
    sp_source=str(sample.sp_source),
    sp_a_target=sample.sp_a_target,
    sp_stack=str(sample.sp_stack),
    traffic_control_valid=False,
    traffic_control_type="",
    traffic_control_distance=None,
    traffic_control_ahead_valid=False,
    traffic_control_ahead_type="",
    traffic_control_ahead_distance=None,
  )


def shadow_samples_to_signal_samples(samples: list[PlannerTargetSample]) -> list[SignalSample]:
  out: list[SignalSample] = []
  for sample in samples:
    out.append(planner_target_sample_to_signal_sample(sample, fallback_shadow_sample=_mirror_leadless_sample(sample)))
  return out


def qlog_samples_to_signal_samples(route: str, read_mode: ReadMode) -> list[SignalSample]:
  out: list[SignalSample] = []
  state: dict = {
    "selfdrive_active": False,
    "long_active": False,
    "plan_a_target": None,
    "plan_should_stop": False,
    "plan_source": "",
    "model_should_stop": False,
    "model_desired_accel": None,
    "model_stop_distance": None,
    "model_endpoint_x": None,
    "model_endpoint_v": None,
    "lead_status": False,
    "lead_d_rel": None,
    "lead_v_rel": None,
    "sp_a_target": None,
    "sp_source": "",
    "sp_stack": "",
    "fcw": False,
    "v_cruise_kph": None,
  }
  route_id_str, seg_int = route_identity(route)
  for route_msg in iter_route_messages(route, read_mode, log_reader_factory=LogReader):
    t = route_msg.t
    typ = route_msg.typ
    payload = route_msg.payload
    if typ == "selfdriveState":
      state["selfdrive_active"] = bool(safe_get(payload, "active", False))
    elif typ == "carControl":
      state["long_active"] = bool(safe_get(payload, "longActive", False))
    elif typ == "longitudinalPlan":
      state["plan_a_target"] = safe_get(payload, "aTarget")
      state["plan_should_stop"] = bool(safe_get(payload, "shouldStop", False))
      state["plan_source"] = str(safe_get(payload, "longitudinalPlanSource", ""))
      state["fcw"] = bool(safe_get(payload, "fcw", False))
    elif typ == "longitudinalPlanSP":
      state["sp_a_target"] = safe_get(payload, "aTarget")
      state["sp_source"] = str(safe_get(payload, "longitudinalPlanSource", ""))
      state["sp_stack"] = str(safe_get(payload, "stack.actuatedStack", ""))
    elif typ == "modelV2":
      state["model_should_stop"] = bool(safe_get(payload, "action.shouldStop", False))
      state["model_desired_accel"] = safe_get(payload, "action.desiredAcceleration")
      state["model_stop_distance"] = safe_get(payload, "action.stopDistance")
      if state["model_stop_distance"] is None:
        try:
          state["model_stop_distance"] = safe_get(payload, "action.stoppingDistance")
        except Exception:
          state["model_stop_distance"] = None
    elif typ == "radarState":
      lead = safe_get(payload, "leadOne", {})
      state["lead_status"] = bool(safe_get(lead, "present", False))
      state["lead_d_rel"] = safe_get(lead, "dRel")
      state["lead_v_rel"] = safe_get(lead, "vRel")
    elif typ == "carState":
      v_ego = safe_get(payload, "vEgo")
      a_ego = safe_get(payload, "aEgo")
      if not isinstance(v_ego, (int, float)) or not isinstance(a_ego, (int, float)):
        continue
      state["v_cruise_kph"] = safe_get(payload, "vCruise")
      ss = LeadlessStopSample(
        route=route,
        route_id=route_id_str,
        segment=seg_int,
        t=t,
        v_ego=float(v_ego),
        a_ego=float(a_ego),
        gas_pressed=bool(safe_get(payload, "gasPressed", False)),
        brake_pressed=bool(safe_get(payload, "brakePressed", False)),
        standstill=bool(safe_get(payload, "standstill", False)),
        selfdrive_active=state["selfdrive_active"],
        long_active=state["long_active"],
        long_control_state="pid",
        lead_status=state["lead_status"],
        lead_d_rel=state["lead_d_rel"],
        lead_v_rel=state["lead_v_rel"],
        model_should_stop=state["model_should_stop"],
        model_desired_accel=state["model_desired_accel"],
        model_stop_distance=state["model_stop_distance"],
        model_endpoint_x=state["model_endpoint_x"],
        model_endpoint_v=state["model_endpoint_v"],
        scc_early_model_stop=False,
        plan_should_stop=state["plan_should_stop"],
        plan_source=state["plan_source"],
        plan_a_target=state["plan_a_target"],
        traffic_control_valid=False,
        traffic_control_type="",
        traffic_control_distance=None,
        traffic_control_ahead_valid=False,
        traffic_control_ahead_type="",
        traffic_control_ahead_distance=None,
        sp_source=state["sp_source"],
        sp_a_target=state["sp_a_target"],
        sp_stack=state["sp_stack"],
      )
      yield_p = planner_target_sample_to_signal_sample(_planner_from_leadless(ss), fallback_shadow_sample=ss)
      out.append(yield_p)
  return out


def _planner_from_leadless(ss: LeadlessStopSample) -> PlannerTargetSample:
  return PlannerTargetSample(
    route=ss.route,
    route_id=ss.route_id,
    segment=ss.segment,
    t=ss.t,
    v_ego=ss.v_ego,
    a_ego=ss.a_ego,
    gas_pressed=ss.gas_pressed,
    brake_pressed=ss.brake_pressed,
    standstill=ss.standstill,
    selfdrive_enabled=True,
    selfdrive_active=ss.selfdrive_active,
    long_active=ss.long_active,
    long_control_state=ss.long_control_state,
    v_cruise_kph=None,
    plan_a_target=ss.plan_a_target if ss.plan_a_target is not None else 0.0,
    plan_source=ss.plan_source,
    plan_should_stop=ss.plan_should_stop,
    plan_fcw=False,
    sp_a_target=ss.sp_a_target,
    sp_source=ss.sp_source,
    sp_stack=ss.sp_stack,
    lead_status=ss.lead_status,
    lead_d_rel=ss.lead_d_rel,
    lead_v_rel=ss.lead_v_rel,
    model_desired_accel=ss.model_desired_accel,
    model_should_stop=ss.model_should_stop,
  )


def is_stop_go_candidate(sample: SignalSample) -> bool:
  return (
    sample.v_ego > MOVING_SPEED_FLOOR
    and not sample.lead_status
    and sample.a_ego <= LEADLESS_DEFAULT_DECEL
  )


def label_stop_go_episodes(samples: list[SignalSample], min_duration_s: float = LEADLESS_MIN_DURATION_S) -> list[DecisionEpisode]:
  out: list[DecisionEpisode] = []
  if not samples:
    return out
  run: list[SignalSample] = []
  for s in samples:
    if is_stop_go_candidate(s):
      run.append(s)
    else:
      if run and (run[-1].t - run[0].t) >= min_duration_s:
        v0 = run[0].v_ego
        v1 = run[-1].v_ego
        if v1 <= v0 * 0.7 + 0.5:
          out.append(DecisionEpisode(
            category="stop_go",
            route=run[0].route,
            route_id=run[0].route_id,
            segment=run[0].segment,
            decision_time_s=_first_peak_decel_time(run),
            start_time_s=run[0].t,
            end_time_s=run[-1].t,
            duration_s=run[-1].t - run[0].t,
            kind="leadless_slowdown",
            v_ego=v0,
            a_ego=min(r.a_ego for r in run),
            active_ratio=sum(1 for r in run if r.selfdrive_active) / len(run),
            long_active_ratio=sum(1 for r in run if r.long_active) / len(run),
            extra={"v_end": v1, "peak_decel": min(r.a_ego for r in run)},
          ))
      run = []
  if run and (run[-1].t - run[0].t) >= min_duration_s:
    v0 = run[0].v_ego
    v1 = run[-1].v_ego
    if v1 <= v0 * 0.7 + 0.5:
      out.append(DecisionEpisode(
        category="stop_go",
        route=run[0].route,
        route_id=run[0].route_id,
        segment=run[0].segment,
        decision_time_s=_first_peak_decel_time(run),
        start_time_s=run[0].t,
        end_time_s=run[-1].t,
        duration_s=run[-1].t - run[0].t,
        kind="leadless_slowdown",
        v_ego=v0,
        a_ego=min(r.a_ego for r in run),
        active_ratio=sum(1 for r in run if r.selfdrive_active) / len(run),
        long_active_ratio=sum(1 for r in run if r.long_active) / len(run),
        extra={"v_end": v1, "peak_decel": min(r.a_ego for r in run)},
      ))
  return out


def _first_peak_decel_time(run: list[SignalSample]) -> float:
  if not run:
    return 0.0
  return min(run, key=lambda r: r.a_ego).t


def label_opposite_intent_episodes(samples: list[SignalSample], min_duration_s: float = 0.4) -> list[DecisionEpisode]:
  out: list[DecisionEpisode] = []
  if not samples:
    return out
  run: list[SignalSample] = []
  for s in samples:
    if s.v_ego > MOVING_SPEED_FLOOR and s.is_opposite_intent():
      run.append(s)
    else:
      if run and (run[-1].t - run[0].t) >= min_duration_s:
        out.append(DecisionEpisode(
          category="opposite_intent",
          route=run[0].route,
          route_id=run[0].route_id,
          segment=run[0].segment,
          decision_time_s=run[0].t,
          start_time_s=run[0].t,
          end_time_s=run[-1].t,
          duration_s=run[-1].t - run[0].t,
          kind=f"plan_{run[-1].direction_plan()}_driver_{run[-1].direction_driver()}",
          v_ego=run[0].v_ego,
          a_ego=run[0].a_ego,
          active_ratio=sum(1 for r in run if r.selfdrive_active) / len(run),
          long_active_ratio=sum(1 for r in run if r.long_active) / len(run),
        ))
      run = []
  if run and (run[-1].t - run[0].t) >= min_duration_s:
    out.append(DecisionEpisode(
      category="opposite_intent",
      route=run[0].route,
      route_id=run[0].route_id,
      segment=run[0].segment,
      decision_time_s=run[0].t,
      start_time_s=run[0].t,
      end_time_s=run[-1].t,
      duration_s=run[-1].t - run[0].t,
      kind=f"plan_{run[-1].direction_plan()}_driver_{run[-1].direction_driver()}",
      v_ego=run[0].v_ego,
      a_ego=run[0].a_ego,
      active_ratio=sum(1 for r in run if r.selfdrive_active) / len(run),
      long_active_ratio=sum(1 for r in run if r.long_active) / len(run),
    ))
  return out


def label_lead_transition_episodes(samples: list[SignalSample], min_window_s: float = 0.5) -> list[DecisionEpisode]:
  out: list[DecisionEpisode] = []
  if not samples:
    return out
  prev_status = bool(samples[0].lead_status)
  pending_flip_t: float | None = None
  pending_flip_to: bool | None = None
  tolerance = 1e-3
  for s in samples:
    if s.v_ego <= MOVING_SPEED_FLOOR:
      continue
    if bool(s.lead_status) != prev_status:
      if pending_flip_t is None:
        pending_flip_t = s.t
        pending_flip_to = bool(s.lead_status)
      elif bool(s.lead_status) != pending_flip_to:
        pending_flip_t = s.t
        pending_flip_to = bool(s.lead_status)
      elif (s.t - pending_flip_t) >= min_window_s - tolerance:
        out.append(DecisionEpisode(
          category="lead_transition",
          route=s.route,
          route_id=s.route_id,
          segment=s.segment,
          decision_time_s=pending_flip_t,
          start_time_s=pending_flip_t,
          end_time_s=s.t,
          duration_s=s.t - pending_flip_t,
          kind="lead_acquired" if pending_flip_to else "lead_lost",
          v_ego=s.v_ego,
          a_ego=s.a_ego,
          active_ratio=1.0 if s.selfdrive_active else 0.0,
          long_active_ratio=1.0 if s.long_active else 0.0,
          extra={"d_rel": s.lead_d_rel, "v_rel": s.lead_v_rel},
        ))
        prev_status = bool(s.lead_status)
        pending_flip_t = None
        pending_flip_to = None
    else:
      pending_flip_t = None
      pending_flip_to = None
      prev_status = bool(s.lead_status)
  return out


SIGNAL_REGISTRY: dict[str, SignalEval] = {}


def register_signal(name: str):
  def decorator(fn: SignalEval) -> SignalEval:
    SIGNAL_REGISTRY[name] = fn
    return fn
  return decorator


@register_signal("model_should_stop")
def _sig_model_should_stop(s: SignalSample) -> bool:
  return s.model_should_stop


@register_signal("model_accel_le_-0.5")
def _sig_model_accel_mild(s: SignalSample) -> bool:
  return s.model_desired_accel is not None and s.model_desired_accel <= -0.5


@register_signal("model_accel_le_-1.0")
def _sig_model_accel_strong(s: SignalSample) -> bool:
  return s.model_desired_accel is not None and s.model_desired_accel <= -1.0


@register_signal("model_path_or_should_stop")
def _sig_model_path_or_should_stop(s: SignalSample) -> bool:
  if s.model_should_stop:
    return True
  if s.model_stop_distance is not None and s.model_stop_distance <= profile_leadless_stops.MODEL_STOP_PATH_MAX_DISTANCE:
    return True
  return False


@register_signal("planner_should_stop")
def _sig_planner_should_stop(s: SignalSample) -> bool:
  return s.plan_should_stop


@register_signal("planner_e2e_or_should_stop")
def _sig_planner_e2e_or_should_stop(s: SignalSample) -> bool:
  if s.plan_should_stop:
    return True
  src = s.plan_source.lower()
  return src in {"e2e", "model"}


@register_signal("plan_brake_direction")
def _sig_plan_brake(s: SignalSample) -> bool:
  return s.direction_plan() == "brake"


@register_signal("driver_brake")
def _sig_driver_brake(s: SignalSample) -> bool:
  return s.direction_driver() == "brake"


@register_signal("driver_gas")
def _sig_driver_gas(s: SignalSample) -> bool:
  return s.direction_driver() == "accel"


@register_signal("plan_a_target_negative")
def _sig_plan_accel_negative(s: SignalSample) -> bool:
  return s.plan_a_target is not None and s.plan_a_target < 0


@register_signal("plan_a_target_positive")
def _sig_plan_accel_positive(s: SignalSample) -> bool:
  return s.plan_a_target is not None and s.plan_a_target > 0


@register_signal("sp_e2e_like")
def _sig_sp_e2e_like(s: SignalSample) -> bool:
  return s.sp_source.lower() in {"e2e", "model"}


@register_signal("plan_e2e_source")
def _sig_plan_e2e_source(s: SignalSample) -> bool:
  return s.plan_source.lower() in {"e2e", "model"}


@register_signal("plan_e2e_or_should_stop")
def _sig_plan_e2e_or_should_stop(s: SignalSample) -> bool:
  if s.plan_should_stop:
    return True
  return s.plan_source.lower() in {"e2e", "model"}


@register_signal("sp_customv2_active")
def _sig_sp_customv2_active(s: SignalSample) -> bool:
  return s.sp_stack.lower() in {"customv2", "custom-2.0"}


@register_signal("plan_braking")
def _sig_plan_braking(s: SignalSample) -> bool:
  return s.plan_a_target is not None and s.plan_a_target < -0.3


@register_signal("plan_strong_brake")
def _sig_plan_strong_brake(s: SignalSample) -> bool:
  return s.plan_a_target is not None and s.plan_a_target < -1.0


@register_signal("lead_present")
def _sig_lead_present(s: SignalSample) -> bool:
  return s.lead_status


@register_signal("lead_closing_fast")
def _sig_lead_closing_fast(s: SignalSample) -> bool:
  if not s.lead_status or s.lead_v_rel is None:
    return False
  return s.lead_v_rel < -1.0


def _samples_in_window(samples: list[SignalSample], start_t: float, end_t: float) -> list[SignalSample]:
  return [s for s in samples if start_t - DEFAULT_SIGNAL_LOOKBACK_S <= s.t <= end_t + DEFAULT_TIMELY_GRACE_S]


def _signal_hits_in_window(samples: list[SignalSample], decision_time: float,
                           signal: SignalEval, *,
                           lookback_s: float, timely_grace_s: float) -> tuple[list[SignalSample], list[SignalSample]]:
  hits = [s for s in samples if (decision_time - lookback_s) <= s.t <= (decision_time + timely_grace_s) and signal(s)]
  timely = [s for s in hits if s.t <= decision_time + timely_grace_s]
  return hits, timely


def _non_episode_windows(sorted_samples: list[SignalSample], episodes: list[DecisionEpisode],
                        window_s: float, max_windows: int = 50) -> list[tuple[float, float]]:
  """Sample non-episode windows of length window_s for false-alarm measurement."""
  if not episodes or not sorted_samples:
    return []
  t_min = sorted_samples[0].t
  t_max = sorted_samples[-1].t
  if t_max - t_min <= window_s:
    return []

  episode_ranges = [(ep.start_time_s, ep.end_time_s) for ep in episodes]
  episode_ranges.sort()

  def overlaps(a_start, a_end, b_start, b_end):
    return a_start < b_end and b_start < a_end

  stride = max(window_s, (t_max - t_min - window_s) / max_windows)
  out: list[tuple[float, float]] = []
  t = t_min
  while t + window_s <= t_max and len(out) < max_windows:
    if not any(overlaps(t, t + window_s, er[0], er[1]) for er in episode_ranges):
      out.append((t, t + window_s))
    t += stride
  return out


def score_signal_for_category(samples: list[SignalSample], episodes: list[DecisionEpisode], signal_name: str,
                              lookback_s: float = DEFAULT_SIGNAL_LOOKBACK_S,
                              timely_grace_s: float = DEFAULT_TIMELY_GRACE_S) -> SignalMetric:
  if signal_name not in SIGNAL_REGISTRY:
    raise KeyError(f"unknown signal {signal_name!r}")
  signal = SIGNAL_REGISTRY[signal_name]
  hit_count = 0
  timely_count = 0
  miss_count = 0
  lead_times: list[float] = []
  fire_count = 0
  flicker_count = 0
  onset_lags: list[float] = []
  true_positive_fires = 0

  sorted_samples = sorted(samples, key=lambda s: s.t)
  for ep in episodes:
    window = [s for s in sorted_samples if (ep.decision_time_s - lookback_s) <= s.t <= (ep.decision_time_s + timely_grace_s)]
    hits = [s for s in window if signal(s)]
    fire_count += len(hits)
    true_positive_fires += len(hits)
    if hits:
      hit_count += 1
      if any(s.t <= ep.decision_time_s + timely_grace_s for s in hits):
        timely_count += 1
      lead_times.extend(ep.decision_time_s - s.t for s in hits)
      flips = sum(1 for prev, cur in zip(window, window[1:], strict=False) if signal(prev) != signal(cur))
      flicker_count += flips
      first_hit = min(hits, key=lambda s: s.t)
      onset_lags.append(first_hit.t - (ep.decision_time_s - lookback_s))
    else:
      miss_count += 1

  false_alarm_count = 0
  false_alarm_windows = 0
  false_positive_fires = 0
  gap_window_s = lookback_s + timely_grace_s
  non_ep_windows = _non_episode_windows(sorted_samples, episodes, gap_window_s)
  for nw_start, nw_end in non_ep_windows:
    nw_samples = [s for s in sorted_samples if nw_start <= s.t <= nw_end]
    fires_in_window = sum(1 for s in nw_samples if signal(s))
    false_positive_fires += fires_in_window
    if fires_in_window > 0:
      false_alarm_count += 1
    false_alarm_windows += 1

  precision = _ratio(true_positive_fires, true_positive_fires + false_positive_fires)

  return SignalMetric(
    category="",
    signal=signal_name,
    episodes=len(episodes),
    hit_count=hit_count,
    timely_hit_count=timely_count,
    missed_count=miss_count,
    fire_count=fire_count,
    recall=_ratio(hit_count, len(episodes)),
    recall_timely=_ratio(timely_count, len(episodes)),
    precision=precision,
    true_positives=true_positive_fires,
    false_positives=false_positive_fires,
    false_alarm_count=false_alarm_count,
    false_alarm_windows=false_alarm_windows,
    median_lead_time_s=_median_or_none(lead_times),
    p10_lead_time_s=_percentile_or_none(lead_times, 10.0),
    flicker_count=flicker_count,
    onset_lag_s=_median_or_none(onset_lags),
  )


def _ratio(num: int, denom: int) -> float:
  return num / denom if denom else 0.0


def _median_or_none(values: list[float]) -> float | None:
  if not values:
    return None
  return float(statistics.median(values))


def _percentile_or_none(values: list[float], percentile: float) -> float | None:
  if not values:
    return None
  idx = max(0, min(len(values) - 1, int(round(percentile / 100.0 * (len(values) - 1)))))
  return float(sorted(values)[idx])


def score_category(samples: list[SignalSample], category: str, signal_names: list[str],
                   labeler: Labeler, **labeler_kwargs) -> CategoryScorecard:
  episodes = labeler(samples, **labeler_kwargs) if labeler_kwargs else labeler(samples)
  metrics: dict[str, SignalMetric] = {}
  for name in signal_names:
    if name not in SIGNAL_REGISTRY:
      continue
    metric = score_signal_for_category(samples, episodes, name)
    metrics[name] = SignalMetric(
      category=category,
      signal=name,
      episodes=metric.episodes,
      hit_count=metric.hit_count,
      timely_hit_count=metric.timely_hit_count,
      missed_count=metric.missed_count,
      fire_count=metric.fire_count,
      recall=metric.recall,
      recall_timely=metric.recall_timely,
      precision=metric.precision,
      true_positives=metric.true_positives,
      false_positives=metric.false_positives,
      false_alarm_count=metric.false_alarm_count,
      false_alarm_windows=metric.false_alarm_windows,
      median_lead_time_s=metric.median_lead_time_s,
      p10_lead_time_s=metric.p10_lead_time_s,
      flicker_count=metric.flicker_count,
      onset_lag_s=metric.onset_lag_s,
    )
  return CategoryScorecard(
    category=category,
    episode_count=len(episodes),
    signals=metrics,
  )


def extract_signal_samples_for_routes(routes: list[str], options: ShadowReplayOptions) -> dict[str, list[SignalSample]]:
  out: dict[str, list[SignalSample]] = {}
  for route in routes:
    samples = extract_shadow_samples(route, ReadMode.AUTO, options)
    out[route] = shadow_samples_to_signal_samples(samples)
  return out


CATEGORY_DEFINITIONS: dict[str, dict[str, Any]] = {
  "stop_go": {
    "labeler": label_stop_go_episodes,
    "signals": [
      "model_should_stop",
      "model_path_or_should_stop",
      "model_accel_le_-0.5",
      "model_accel_le_-1.0",
      "planner_should_stop",
      "planner_e2e_or_should_stop",
      "sp_e2e_like",
    ],
  },
  "opposite_intent": {
    "labeler": label_opposite_intent_episodes,
    "signals": [
      "model_should_stop",
      "planner_should_stop",
      "plan_brake_direction",
      "plan_a_target_negative",
      "plan_a_target_positive",
      "sp_e2e_like",
      "plan_e2e_source",
      "plan_e2e_or_should_stop",
      "sp_customv2_active",
    ],
  },
  "lead_transition": {
    "labeler": label_lead_transition_episodes,
    "signals": [
      "lead_present",
      "lead_closing_fast",
      "plan_a_target_negative",
      "plan_brake_direction",
      "sp_e2e_like",
      "plan_e2e_source",
      "plan_e2e_or_should_stop",
      "sp_customv2_active",
      "plan_braking",
      "plan_strong_brake",
    ],
  },
}


def run_scorer(samples_by_route: dict[str, list[SignalSample]], categories: list[str] | None = None) -> ScorerSummary:
  cats = categories or list(CATEGORY_DEFINITIONS.keys())
  all_samples: list[SignalSample] = [s for samples in samples_by_route.values() for s in samples]
  out: list[CategoryScorecard] = []
  for cat in cats:
    if cat not in CATEGORY_DEFINITIONS:
      continue
    labeler = CATEGORY_DEFINITIONS[cat]["labeler"]
    sigs = CATEGORY_DEFINITIONS[cat]["signals"]
    out.append(score_category(all_samples, cat, sigs, labeler))
  return ScorerSummary(
    route_count=len(samples_by_route),
    sample_count=len(all_samples),
    episode_count=sum(c.episode_count for c in out),
    category_count=len(out),
    signal_count=sum(len(c.signals) for c in out),
    categories=out,
  )


def summary_to_dict(summary: ScorerSummary) -> dict:
  return {
    "route_count": summary.route_count,
    "sample_count": summary.sample_count,
    "episode_count": summary.episode_count,
    "category_count": summary.category_count,
    "signal_count": summary.signal_count,
    "categories": [
      {
        "category": c.category,
        "episode_count": c.episode_count,
        "signals": {name: asdict(metric) for name, metric in c.signals.items()},
      }
      for c in summary.categories
    ],
  }


def render_summary(summary: ScorerSummary) -> str:
  lines = [
    "Drive Lab longitudinal signal scorecard",
    f"routes={summary.route_count} samples={summary.sample_count} "
    f"episodes={summary.episode_count} categories={summary.category_count} "
    f"signals={summary.signal_count}",
  ]
  for cat in summary.categories:
    lines.append(f"== category={cat.category} episodes={cat.episode_count}")
    if not cat.signals:
      lines.append("  (no signals scored)")
      continue
    lines.append("  (signals with recall > 0, ranked by precision then recall then -fire_count)")
    ranked = sorted(
      cat.signals.items(),
      key=lambda item: (-item[1].precision, -item[1].recall_timely, -item[1].recall, item[1].fire_count),
    )
    for name, m in ranked:
      if m.recall <= 0:
        continue
      lead = f"{m.median_lead_time_s:+.2f}s" if m.median_lead_time_s is not None else "  n/a"
      p10 = f"{m.p10_lead_time_s:+.2f}s" if m.p10_lead_time_s is not None else "  n/a"
      lines.append(
        f"  {name:30s} recall={m.recall:.2f} timely={m.recall_timely:.2f} "
        f"precision={m.precision:.2f} flickers={m.flicker_count:5d} "
        f"lead={lead} p10={p10} fires={m.fire_count:5d} "
        f"tp={m.true_positives:4d} fp={m.false_positives:4d} "
        f"fa_windows={m.false_alarm_count}/{m.false_alarm_windows}"
      )
    skipped = [(n, m) for n, m in cat.signals.items() if m.recall <= 0]
    if skipped:
      lines.append("  (zero-recall signals: " + ", ".join(n for n, _ in skipped) + ")")
  return "\n".join(lines)


def main() -> None:
  parser = argparse.ArgumentParser(description="Score planner/decision signals against human-driven decisions.")
  parser.add_argument("routes", nargs="+", help="Routes, segment ranges, or rlog files (shadow replay needs modelV2).")
  parser.add_argument("--stack", default="customV2", choices=sorted({"customV2", "sunnypilotCurrent", "customRecommended"}))
  parser.add_argument("--qlog", action="store_true", help="Use qlogs. Shadow replay requires modelV2, so this is mostly a no-op.")
  parser.add_argument("--json", action="store_true", help="Emit JSON instead of text")
  parser.add_argument("--output", help="Write JSON summary to this path")
  parser.add_argument("--categories", nargs="*", help="Limit to specific categories")
  args = parser.parse_args()

  options = ShadowReplayOptions(stack=args.stack)
  samples_by_route = extract_signal_samples_for_routes(args.routes, options)
  summary = run_scorer(samples_by_route, args.categories)

  if args.output:
    with open(args.output, "w") as f:
      json.dump(summary_to_dict(summary), f, indent=2)
  if args.json:
    print(json.dumps(summary_to_dict(summary), indent=2))
  else:
    print(render_summary(summary))


if __name__ == "__main__":
  main()
