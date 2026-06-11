#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path
from typing import Any

import numpy as np

from openpilot.tools.drive_lab.scenario_spec import ScenarioSpec, route_window_provenance
from openpilot.tools.drive_lab.route_io import load_route_msgs, output_report
from openpilot.tools.drive_lab.timeline import format_enum, msg_payload, msg_time_s, msg_type, safe_get
from openpilot.tools.lib.logreader import LogReader, ReadMode


PLAN_DIRECTION_THRESHOLD = 0.35
DRIVER_DIRECTION_THRESHOLD = 0.35
STRONG_BRAKE_TARGET = -1.0
STRONG_ACCEL_TARGET = 0.8
STRONG_DRIVER_ACCEL = 0.6
STRONG_DRIVER_BRAKE = -0.6
MOVING_SPEED = 1.0
DEFAULT_MAX_PLAN_AGE_S = 1.0
DEFAULT_EPISODE_GAP_S = 0.6
DEFAULT_EPISODE_CONTEXT_S = 3.0
DEFAULT_LARGE_ERROR_THRESHOLD = 1.2
DEFAULT_HIGH_JERK_THRESHOLD = 8.0
LOW_TTC_THRESHOLD_S = 2.5
HIGH_REQUIRED_DECEL_THRESHOLD_MPS2 = 2.5
MIN_CLOSING_SPEED_MPS = 0.1
MIN_HEADWAY_SPEED_MPS = 0.1
UNSET_CRUISE_KPH = 250.0


@dataclass(frozen=True)
class PlannerTargetSample:
  route: str
  route_id: str
  segment: int | None
  t: float
  v_ego: float
  a_ego: float
  gas_pressed: bool
  brake_pressed: bool
  standstill: bool
  selfdrive_enabled: bool
  selfdrive_active: bool
  long_active: bool
  long_control_state: str
  v_cruise_kph: float | None
  plan_a_target: float
  plan_source: str
  plan_should_stop: bool
  plan_fcw: bool
  sp_a_target: float | None
  sp_source: str
  sp_stack: str
  lead_status: bool
  lead_d_rel: float | None
  lead_v_rel: float | None
  model_desired_accel: float | None
  model_should_stop: bool
  ttc_s: float | None = None
  required_decel_mps2: float | None = None
  time_headway_s: float | None = None


@dataclass(frozen=True)
class RouteAgreementProfile:
  route: str
  route_id: str
  segment: int | None
  samples: int
  duration_s: float
  moving_samples: int
  manual_moving_samples: int
  low_confidence_preview_samples: int
  actuation_applicable_samples: int
  active_ratio: float
  include: bool


@dataclass(frozen=True)
class PlannerTargetEpisode:
  route: str
  route_id: str
  segment: int | None
  start_time_s: float
  end_time_s: float
  duration_s: float
  sample_count: int
  opposite_count: int
  strong_opposite_count: int
  max_abs_error: float
  planner_sources: dict[str, int]
  driver_gas_count: int
  driver_brake_count: int
  lead_ratio: float
  min_lead_d_rel: float | None
  min_lead_v_rel: float | None
  lead_status_flips: int
  plan_source_flips: int
  plan_span: float
  high_plan_jerk_count: int
  min_ttc_s: float | None
  max_required_decel_mps2: float | None


@dataclass(frozen=True)
class PlannerTargetAgreementSummary:
  route_count: int
  included_route_count: int
  sample_count: int
  manual_moving_sample_count: int
  low_confidence_preview_sample_count: int
  low_confidence_preview_reasons: dict[str, int]
  actuation_applicable_sample_count: int
  correlation: float | None
  mean_abs_error: float
  p90_abs_error: float
  p95_abs_error: float
  opposite_count: int
  opposite_ratio: float
  strong_opposite_count: int
  strong_opposite_ratio: float
  should_stop_moving_count: int
  should_stop_conflict_count: int
  fcw_count: int
  high_plan_jerk_count: int
  min_ttc_s: float | None
  max_required_decel_mps2: float | None
  mean_time_headway_s: float | None
  high_required_decel_count: int
  low_ttc_count: int
  lead_risk_source_counts: dict[str, int]
  planner_source_counts: dict[str, int]
  sp_source_counts: dict[str, int]
  sp_stack_counts: dict[str, int]
  route_profiles: list[RouteAgreementProfile]
  episodes: list[PlannerTargetEpisode]


def main() -> None:
  parser = argparse.ArgumentParser(description="Compare logged longitudinal planner targets with manual driving response.")
  parser.add_argument("routes", nargs="+", help="Routes, segment ranges, log files, or URLs accepted by LogReader")
  parser.add_argument("--qlog", action="store_true", help="Prefer qlogs instead of rlogs")
  parser.add_argument("--json", action="store_true", help="Print JSON instead of text")
  parser.add_argument("--output", help="Write JSON summary to this path")
  parser.add_argument("--min-manual-moving", type=int, default=100)
  parser.add_argument("--max-active-ratio", type=float, default=0.25)
  parser.add_argument("--max-plan-age", type=float, default=DEFAULT_MAX_PLAN_AGE_S)
  parser.add_argument("--large-error", type=float, default=DEFAULT_LARGE_ERROR_THRESHOLD)
  parser.add_argument("--episode-gap", type=float, default=DEFAULT_EPISODE_GAP_S)
  parser.add_argument("--episode-context", type=float, default=DEFAULT_EPISODE_CONTEXT_S)
  parser.add_argument("--high-jerk", type=float, default=DEFAULT_HIGH_JERK_THRESHOLD)
  parser.add_argument("--episodes", type=int, default=12, help="Number of suspicious episodes to show in text output")
  parser.add_argument("--scenario-output", help="Write route-derived scenario specs JSON to this path")
  parser.add_argument("--include-low-confidence-preview", action="store_true",
                      help="Include reset/manual-preview samples in disagreement metrics for exploratory use")
  args = parser.parse_args()

  read_mode = ReadMode.QLOG if args.qlog else ReadMode.AUTO
  samples_by_route: dict[str, list[PlannerTargetSample]] = {}
  profiles: list[RouteAgreementProfile] = []
  for route in args.routes:
    samples = extract_planner_target_samples(route, read_mode, max_plan_age_s=args.max_plan_age)
    samples_by_route[route] = samples
    profiles.append(build_route_agreement_profile(route, samples, args.min_manual_moving, args.max_active_ratio))

  summary = summarize_planner_target_agreement(
    samples_by_route,
    profiles,
    large_error_threshold=args.large_error,
    episode_gap_s=args.episode_gap,
    episode_context_s=args.episode_context,
    high_jerk_threshold=args.high_jerk,
    include_low_confidence_preview=args.include_low_confidence_preview,
  )
  payload = summary_to_dict(summary)
  if args.scenario_output:
    scenarios = [episode_to_scenario_spec(episode, index=i).to_dict() for i, episode in enumerate(summary.episodes[:args.episodes])]
    with open(args.scenario_output, "w") as f:
      json.dump(scenarios, f, indent=2)
      f.write("\n")
  class _PlannerPayload:
    def to_dict(self):
      return payload

  print(output_report(_PlannerPayload(), json_output=args.json, renderer=lambda _: render_agreement_summary(summary, max_episodes=args.episodes), output_path=args.output))


def extract_planner_target_samples(route: str, read_mode: ReadMode, max_plan_age_s: float = DEFAULT_MAX_PLAN_AGE_S) -> list[PlannerTargetSample]:
  route_id, segment = route_identity(route)
  samples: list[PlannerTargetSample] = []
  state: dict[str, Any] = {
    "selfdrive_enabled": False,
    "selfdrive_active": False,
    "long_active": False,
    "long_control_state": "unknown",
    "plan_a_target": None,
    "plan_source": "unknown",
    "plan_should_stop": False,
    "plan_fcw": False,
    "plan_time_s": None,
    "sp_a_target": None,
    "sp_source": "unknown",
    "sp_stack": "unknown",
    "lead_status": False,
    "lead_d_rel": None,
    "lead_v_rel": None,
    "model_desired_accel": None,
    "model_should_stop": False,
  }
  base_mono_time: int | None = None

  for msg in LogReader(route, default_mode=read_mode, sort_by_time=True):
    if base_mono_time is None:
      base_mono_time = int(getattr(msg, "logMonoTime", 0))
    t = msg_time_s(msg, base_mono_time)
    typ = msg_type(msg)
    payload = msg_payload(msg)

    if typ == "selfdriveState":
      state["selfdrive_enabled"] = bool(safe_get(payload, "enabled", False))
      state["selfdrive_active"] = bool(safe_get(payload, "active", False))
    elif typ == "carControl":
      state["long_active"] = bool(safe_get(payload, "longActive", False))
    elif typ == "controlsState":
      state["long_control_state"] = format_enum(safe_get(payload, "longControlState", "unknown"))
    elif typ == "radarState":
      lead = safe_get(payload, "leadOne")
      state["lead_status"] = bool(safe_get(lead, "status", False))
      state["lead_d_rel"] = _finite_or_none(safe_get(lead, "dRel"))
      state["lead_v_rel"] = _finite_or_none(safe_get(lead, "vRel"))
    elif typ == "modelV2":
      state["model_desired_accel"] = _finite_or_none(safe_get(payload, "action.desiredAcceleration"))
      state["model_should_stop"] = bool(safe_get(payload, "action.shouldStop", False))
    elif typ == "longitudinalPlan":
      state["plan_a_target"] = _finite_or_none(safe_get(payload, "aTarget"))
      state["plan_source"] = format_enum(safe_get(payload, "longitudinalPlanSource", "unknown"))
      state["plan_should_stop"] = bool(safe_get(payload, "shouldStop", False))
      state["plan_fcw"] = bool(safe_get(payload, "fcw", False))
      state["plan_time_s"] = t
    elif typ == "longitudinalPlanSP":
      state["sp_a_target"] = _finite_or_none(safe_get(payload, "aTarget"))
      state["sp_source"] = format_enum(safe_get(payload, "longitudinalPlanSource", "unknown"))
      state["sp_stack"] = format_enum(safe_get(payload, "stack.actuatedStack", "unknown"))
    elif typ == "carState":
      plan_time_s = state["plan_time_s"]
      plan_a_target = state["plan_a_target"]
      if plan_time_s is None or plan_a_target is None or t - float(plan_time_s) > max_plan_age_s:
        continue
      v_ego = _finite_or_none(safe_get(payload, "vEgo"))
      a_ego = _finite_or_none(safe_get(payload, "aEgo"))
      if v_ego is None or a_ego is None:
        continue
      lead_status = bool(state["lead_status"])
      lead_d_rel = state["lead_d_rel"]
      lead_v_rel = state["lead_v_rel"]
      closing_speed = (-float(lead_v_rel)) if lead_v_rel is not None else None
      ttc_s = _ttc_s(lead_status, lead_d_rel, closing_speed)
      required_decel_mps2 = _required_decel_mps2(lead_status, lead_d_rel, closing_speed)
      time_headway_s = _time_headway_s(lead_status, lead_d_rel, v_ego)
      samples.append(PlannerTargetSample(
        route=route,
        route_id=route_id,
        segment=segment,
        t=t,
        v_ego=v_ego,
        a_ego=a_ego,
        gas_pressed=bool(safe_get(payload, "gasPressed", False)),
        brake_pressed=bool(safe_get(payload, "brakePressed", False)),
        standstill=bool(safe_get(payload, "standstill", False)),
        selfdrive_enabled=bool(state["selfdrive_enabled"]),
        selfdrive_active=bool(state["selfdrive_active"]),
        long_active=bool(state["long_active"]),
        long_control_state=str(state["long_control_state"]),
        v_cruise_kph=_finite_or_none(safe_get(payload, "vCruise")),
        plan_a_target=float(plan_a_target),
        plan_source=str(state["plan_source"]),
        plan_should_stop=bool(state["plan_should_stop"]),
        plan_fcw=bool(state["plan_fcw"]),
        sp_a_target=state["sp_a_target"],
        sp_source=str(state["sp_source"]),
        sp_stack=str(state["sp_stack"]),
        lead_status=lead_status,
        lead_d_rel=lead_d_rel,
        lead_v_rel=lead_v_rel,
        ttc_s=ttc_s,
        required_decel_mps2=required_decel_mps2,
        time_headway_s=time_headway_s,
        model_desired_accel=state["model_desired_accel"],
        model_should_stop=bool(state["model_should_stop"]),
      ))
  return samples


def route_identity(route: str) -> tuple[str, int | None]:
  path = Path(str(route))
  name = path.parent.name if path.name in {"qlog.zst", "rlog.zst", "qlog.bz2", "rlog.bz2"} else path.name
  if "--" not in name:
    return str(route), None
  prefix, segment = name.rsplit("--", 1)
  try:
    return prefix, int(segment)
  except ValueError:
    return name, None


def build_route_agreement_profile(route: str, samples: list[PlannerTargetSample], min_manual_moving_samples: int = 100,
                                  max_active_ratio: float = 0.25) -> RouteAgreementProfile:
  route_id, segment = route_identity(route)
  moving_samples = [sample for sample in samples if sample.v_ego > MOVING_SPEED]
  manual_moving = [sample for sample in moving_samples if is_manual_preview_sample(sample)]
  low_confidence_preview = [sample for sample in manual_moving if is_low_confidence_manual_preview_sample(sample)]
  actuation_applicable = [sample for sample in samples if is_actuation_applicable_sample(sample)]
  active_count = sum(1 for sample in samples if sample.selfdrive_active or sample.long_active)
  active_ratio = active_count / len(samples) if samples else 1.0
  include = len(manual_moving) >= min_manual_moving_samples and active_ratio <= max_active_ratio
  duration_s = max((sample.t for sample in samples), default=0.0) - min((sample.t for sample in samples), default=0.0)
  return RouteAgreementProfile(
    route=route,
    route_id=route_id,
    segment=segment,
    samples=len(samples),
    duration_s=duration_s,
    moving_samples=len(moving_samples),
    manual_moving_samples=len(manual_moving),
    low_confidence_preview_samples=len(low_confidence_preview),
    actuation_applicable_samples=len(actuation_applicable),
    active_ratio=active_ratio,
    include=include,
  )


def summarize_planner_target_agreement(samples_by_route: dict[str, list[PlannerTargetSample]], profiles: list[RouteAgreementProfile],
                                       large_error_threshold: float = DEFAULT_LARGE_ERROR_THRESHOLD,
                                       episode_gap_s: float = DEFAULT_EPISODE_GAP_S,
                                       episode_context_s: float = DEFAULT_EPISODE_CONTEXT_S,
                                       high_jerk_threshold: float = DEFAULT_HIGH_JERK_THRESHOLD,
                                       include_low_confidence_preview: bool = False) -> PlannerTargetAgreementSummary:
  included_routes = {profile.route for profile in profiles if profile.include}
  manual_preview_samples = [
    sample
    for route, samples in samples_by_route.items()
    if route in included_routes
    for sample in samples
    if sample.v_ego > MOVING_SPEED and is_manual_preview_sample(sample)
  ]
  low_confidence_preview_reasons = Counter(
    reason
    for sample in manual_preview_samples
    if (reason := low_confidence_manual_preview_reason(sample))
  )
  comparison_samples = manual_preview_samples if include_low_confidence_preview else [
    sample for sample in manual_preview_samples if not is_low_confidence_manual_preview_sample(sample)
  ]
  actuation_applicable_count = sum(
    1
    for route, samples in samples_by_route.items()
    if route in included_routes
    for sample in samples
    if is_actuation_applicable_sample(sample)
  )
  abs_errors = [abs(sample.plan_a_target - sample.a_ego) for sample in comparison_samples]
  opposite_samples = [sample for sample in comparison_samples if is_opposite_intent(sample)]
  strong_samples = [sample for sample in comparison_samples if is_strong_opposite_intent(sample)]
  should_stop_moving = [sample for sample in comparison_samples if sample.plan_should_stop and sample.v_ego > MOVING_SPEED]
  should_stop_conflicts = [sample for sample in should_stop_moving if not sample.brake_pressed and sample.a_ego > -0.2]
  high_jerks = high_plan_jerk_pairs(comparison_samples, high_jerk_threshold)
  risk_samples = [sample for sample in comparison_samples if _is_lead_risk_sample(sample)]
  episodes = build_suspicious_episodes(
    comparison_samples,
    large_error_threshold=large_error_threshold,
    episode_gap_s=episode_gap_s,
    context_s=episode_context_s,
    high_jerk_threshold=high_jerk_threshold,
  )
  return PlannerTargetAgreementSummary(
    route_count=len(profiles),
    included_route_count=len(included_routes),
    sample_count=len(comparison_samples),
    manual_moving_sample_count=len(comparison_samples),
    low_confidence_preview_sample_count=sum(low_confidence_preview_reasons.values()),
    low_confidence_preview_reasons=dict(low_confidence_preview_reasons),
    actuation_applicable_sample_count=actuation_applicable_count,
    correlation=_correlation([sample.plan_a_target for sample in comparison_samples], [sample.a_ego for sample in comparison_samples]),
    mean_abs_error=_mean(abs_errors),
    p90_abs_error=_percentile(abs_errors, 90.0),
    p95_abs_error=_percentile(abs_errors, 95.0),
    opposite_count=len(opposite_samples),
    opposite_ratio=_ratio(len(opposite_samples), len(comparison_samples)),
    strong_opposite_count=len(strong_samples),
    strong_opposite_ratio=_ratio(len(strong_samples), len(comparison_samples)),
    should_stop_moving_count=len(should_stop_moving),
    should_stop_conflict_count=len(should_stop_conflicts),
    fcw_count=sum(1 for sample in comparison_samples if sample.plan_fcw),
    high_plan_jerk_count=len(high_jerks),
    min_ttc_s=min((sample.ttc_s for sample in comparison_samples if sample.ttc_s is not None), default=None),
    max_required_decel_mps2=max((sample.required_decel_mps2 for sample in comparison_samples if sample.required_decel_mps2 is not None), default=None),
    mean_time_headway_s=_optional_mean([sample.time_headway_s for sample in comparison_samples if sample.time_headway_s is not None]),
    high_required_decel_count=sum(
      1 for sample in comparison_samples
      if sample.required_decel_mps2 is not None and sample.required_decel_mps2 >= HIGH_REQUIRED_DECEL_THRESHOLD_MPS2
    ),
    low_ttc_count=sum(1 for sample in comparison_samples if sample.ttc_s is not None and sample.ttc_s <= LOW_TTC_THRESHOLD_S),
    lead_risk_source_counts=dict(Counter(sample.plan_source for sample in risk_samples)),
    planner_source_counts=dict(Counter(sample.plan_source for sample in comparison_samples)),
    sp_source_counts=dict(Counter(sample.sp_source for sample in comparison_samples)),
    sp_stack_counts=dict(Counter(sample.sp_stack for sample in comparison_samples)),
    route_profiles=profiles,
    episodes=episodes,
  )


def episode_to_scenario_spec(episode: PlannerTargetEpisode, source: str = "manual-planner-target", index: int | None = None) -> ScenarioSpec:
  kind = "lead_risk" if episode.min_ttc_s is not None or episode.max_required_decel_mps2 is not None else "planner_target_disagreement"
  title = f"{episode.route_id}{f'--{episode.segment}' if episode.segment is not None else ''} {episode.start_time_s:.1f}-{episode.end_time_s:.1f}s"
  maneuver_kwargs = {
    "route_id": episode.route_id,
    "segment": episode.segment,
    "start_time_s": episode.start_time_s,
    "end_time_s": episode.end_time_s,
  }
  actors: dict[str, Any] = {}
  if episode.min_lead_d_rel is not None or episode.min_lead_v_rel is not None or episode.min_ttc_s is not None or episode.max_required_decel_mps2 is not None:
    lead: dict[str, Any] = {}
    if episode.min_lead_d_rel is not None:
      lead["min_d_rel"] = episode.min_lead_d_rel
    if episode.min_lead_v_rel is not None:
      lead["min_v_rel"] = episode.min_lead_v_rel
    if episode.min_ttc_s is not None:
      lead["min_ttc_s"] = episode.min_ttc_s
    if episode.max_required_decel_mps2 is not None:
      lead["max_required_decel_mps2"] = episode.max_required_decel_mps2
    actors["lead"] = lead
  events = [kind]
  if episode.high_plan_jerk_count > 0:
    events.append("high_plan_jerk")
  return ScenarioSpec(
    scenario_id=f"{source}:{kind}:{episode.route_id}:{episode.start_time_s:.1f}:{episode.end_time_s:.1f}" if index is None else f"{source}:{kind}:{episode.route_id}:{index}",
    kind=kind,
    title=title,
    mode="route-derived",
    duration=episode.duration_s,
    source=source,
    maneuver_kwargs=maneuver_kwargs,
    actors=actors,
    events=tuple(events),
    oracle={"checks": ("manual_agreement", "lead_risk", "jerk")},
    tags=(source, "route-derived", "longitudinal", kind),
    seed=None,
    index=index,
    provenance=route_window_provenance(episode.route_id, episode.segment, episode.start_time_s, episode.end_time_s, "compare_manual_planner_targets"),
  )


def is_manual_preview_sample(sample: PlannerTargetSample) -> bool:
  return not sample.selfdrive_active and not sample.long_active


def low_confidence_manual_preview_reason(sample: PlannerTargetSample) -> str:
  if not is_manual_preview_sample(sample):
    return ""
  if _state_name(sample.long_control_state) == "off":
    return "long_control_off"
  if sample.v_cruise_kph is None:
    return "missing_cruise"
  if sample.v_cruise_kph >= UNSET_CRUISE_KPH:
    return "unset_cruise"
  return ""


def is_low_confidence_manual_preview_sample(sample: PlannerTargetSample) -> bool:
  return bool(low_confidence_manual_preview_reason(sample))


def is_actuation_applicable_sample(sample: PlannerTargetSample) -> bool:
  return sample.selfdrive_active and sample.long_active and not sample.gas_pressed and not sample.brake_pressed and \
    sample.long_control_state.lower() != "off"


def planner_direction(sample: PlannerTargetSample) -> str:
  if sample.plan_a_target <= -PLAN_DIRECTION_THRESHOLD:
    return "brake"
  if sample.plan_a_target >= PLAN_DIRECTION_THRESHOLD:
    return "accel"
  return "neutral"


def driver_direction(sample: PlannerTargetSample) -> str:
  if sample.brake_pressed or sample.a_ego <= -DRIVER_DIRECTION_THRESHOLD:
    return "brake"
  if sample.gas_pressed or sample.a_ego >= DRIVER_DIRECTION_THRESHOLD:
    return "accel"
  return "neutral"


def is_opposite_intent(sample: PlannerTargetSample) -> bool:
  plan = planner_direction(sample)
  driver = driver_direction(sample)
  return (plan == "brake" and driver == "accel") or (plan == "accel" and driver == "brake")


def is_strong_opposite_intent(sample: PlannerTargetSample) -> bool:
  return (
    sample.plan_a_target <= STRONG_BRAKE_TARGET and (sample.gas_pressed or sample.a_ego >= STRONG_DRIVER_ACCEL)
  ) or (
    sample.plan_a_target >= STRONG_ACCEL_TARGET and (sample.brake_pressed or sample.a_ego <= STRONG_DRIVER_BRAKE)
  )


def build_suspicious_episodes(samples: list[PlannerTargetSample], large_error_threshold: float = DEFAULT_LARGE_ERROR_THRESHOLD,
                              episode_gap_s: float = DEFAULT_EPISODE_GAP_S, context_s: float = DEFAULT_EPISODE_CONTEXT_S,
                              high_jerk_threshold: float = DEFAULT_HIGH_JERK_THRESHOLD) -> list[PlannerTargetEpisode]:
  samples_by_key: dict[tuple[str, int | None], list[PlannerTargetSample]] = defaultdict(list)
  for sample in samples:
    samples_by_key[(sample.route, sample.segment)].append(sample)

  raw_episodes: list[list[PlannerTargetSample]] = []
  for key_samples in samples_by_key.values():
    suspicious = [sample for sample in sorted(key_samples, key=lambda s: s.t) if _is_suspicious_sample(sample, large_error_threshold)]
    current: list[PlannerTargetSample] = []
    for sample in suspicious:
      if not current or sample.t - current[-1].t <= episode_gap_s:
        current.append(sample)
      else:
        raw_episodes.append(current)
        current = [sample]
    if current:
      raw_episodes.append(current)

  all_by_key = {key: sorted(value, key=lambda s: s.t) for key, value in samples_by_key.items()}
  episodes = [_summarize_episode(episode, all_by_key[(episode[0].route, episode[0].segment)], context_s, high_jerk_threshold)
              for episode in raw_episodes]
  return sorted(episodes, key=lambda e: (e.strong_opposite_count, e.max_abs_error, e.sample_count), reverse=True)


def high_plan_jerk_pairs(samples: list[PlannerTargetSample], threshold: float = DEFAULT_HIGH_JERK_THRESHOLD) -> list[tuple[PlannerTargetSample, PlannerTargetSample, float]]:
  pairs: list[tuple[PlannerTargetSample, PlannerTargetSample, float]] = []
  samples_by_key: dict[tuple[str, int | None], list[PlannerTargetSample]] = defaultdict(list)
  for sample in samples:
    samples_by_key[(sample.route, sample.segment)].append(sample)
  for route_samples in samples_by_key.values():
    ordered = sorted(route_samples, key=lambda s: s.t)
    for prev, cur in zip(ordered, ordered[1:], strict=False):
      dt = cur.t - prev.t
      if 0.05 <= dt <= 0.3:
        jerk = (cur.plan_a_target - prev.plan_a_target) / dt
        if abs(jerk) >= threshold:
          pairs.append((prev, cur, jerk))
  return pairs


def render_agreement_summary(summary: PlannerTargetAgreementSummary, max_episodes: int = 12) -> str:
  lines = [
    "Drive Lab manual planner-target agreement",
    f"routes={summary.route_count} included={summary.included_route_count} samples={summary.sample_count}",
    f"manual_moving_high_confidence={summary.manual_moving_sample_count} "
    + f"low_confidence_preview={summary.low_confidence_preview_sample_count} "
    + f"actuation_applicable={summary.actuation_applicable_sample_count}",
    "Low-confidence preview reasons: " + _format_counts(summary.low_confidence_preview_reasons),
    f"aTarget vs aEgo: corr={_format_optional(summary.correlation)} mean_abs={summary.mean_abs_error:.3f} "
    + f"p90_abs={summary.p90_abs_error:.3f} p95_abs={summary.p95_abs_error:.3f} m/s^2",
    f"opposite_intent={summary.opposite_count} ({summary.opposite_ratio:.3%}) "
    + f"strong={summary.strong_opposite_count} ({summary.strong_opposite_ratio:.3%})",
    f"shouldStop_moving={summary.should_stop_moving_count} shouldStop_conflicts={summary.should_stop_conflict_count} "
    + f"fcw={summary.fcw_count} high_plan_jerk={summary.high_plan_jerk_count}",
    f"lead risk: high_decel={summary.high_required_decel_count} low_ttc={summary.low_ttc_count} "
    + f"min_ttc={_format_optional(summary.min_ttc_s)} max_req_decel={_format_optional(summary.max_required_decel_mps2)} "
    + f"mean_headway={_format_optional(summary.mean_time_headway_s)} sources={_format_counts(summary.lead_risk_source_counts)}",
    "Planner sources: " + _format_counts(summary.planner_source_counts),
    "SP sources: " + _format_counts(summary.sp_source_counts),
    "SP stacks: " + _format_counts(summary.sp_stack_counts),
    "Routes:",
  ]
  for profile in summary.route_profiles:
    status = "include" if profile.include else "exclude"
    segment = f"--{profile.segment}" if profile.segment is not None else ""
    lines.append(
      f"  {profile.route_id}{segment}: {status}, samples {profile.samples}, moving {profile.moving_samples}, "
      + f"manual moving {profile.manual_moving_samples}, low confidence {profile.low_confidence_preview_samples}, "
      + f"active {profile.active_ratio:.3f}, duration {profile.duration_s:.1f}s"
    )
  lines.append("Suspicious episodes:")
  if not summary.episodes:
    lines.append("  none")
  for episode in summary.episodes[:max_episodes]:
    segment = f"--{episode.segment}" if episode.segment is not None else ""
    lines.append(
      f"  {episode.route_id}{segment} {episode.start_time_s:.1f}-{episode.end_time_s:.1f}s "
      + f"n={episode.sample_count} opp={episode.opposite_count} strong={episode.strong_opposite_count} "
      + f"max_err={episode.max_abs_error:.2f} sources={_format_counts(episode.planner_sources)} "
      + f"gas={episode.driver_gas_count} brake={episode.driver_brake_count} lead={episode.lead_ratio:.2f} "
      + f"min_d={_format_optional(episode.min_lead_d_rel)} min_vrel={_format_optional(episode.min_lead_v_rel)} "
      + f"lead_flips={episode.lead_status_flips} source_flips={episode.plan_source_flips} "
      + f"plan_span={episode.plan_span:.2f} high_jerk={episode.high_plan_jerk_count} "
      + f"min_ttc={_format_optional(episode.min_ttc_s)} max_req_decel={_format_optional(episode.max_required_decel_mps2)}"
    )
  return "\n".join(lines)


def summary_to_dict(summary: PlannerTargetAgreementSummary) -> dict[str, Any]:
  return asdict(summary)


def _summarize_episode(episode: list[PlannerTargetSample], all_samples: list[PlannerTargetSample], context_s: float,
                       high_jerk_threshold: float) -> PlannerTargetEpisode:
  start = episode[0].t
  end = episode[-1].t
  window = [sample for sample in all_samples if start - context_s <= sample.t <= end + context_s]
  plan_values = [sample.plan_a_target for sample in window]
  lead_d_values = [sample.lead_d_rel for sample in episode if sample.lead_d_rel is not None]
  lead_v_rel_values = [sample.lead_v_rel for sample in episode if sample.lead_v_rel is not None]
  return PlannerTargetEpisode(
    route=episode[0].route,
    route_id=episode[0].route_id,
    segment=episode[0].segment,
    start_time_s=start,
    end_time_s=end,
    duration_s=end - start + 0.1,
    sample_count=len(episode),
    opposite_count=sum(1 for sample in episode if is_opposite_intent(sample)),
    strong_opposite_count=sum(1 for sample in episode if is_strong_opposite_intent(sample)),
    max_abs_error=max(abs(sample.plan_a_target - sample.a_ego) for sample in episode),
    planner_sources=dict(Counter(sample.plan_source for sample in episode)),
    driver_gas_count=sum(1 for sample in episode if sample.gas_pressed),
    driver_brake_count=sum(1 for sample in episode if sample.brake_pressed),
    lead_ratio=_ratio(sum(1 for sample in episode if sample.lead_status), len(episode)),
    min_lead_d_rel=min(lead_d_values) if lead_d_values else None,
    min_lead_v_rel=min(lead_v_rel_values) if lead_v_rel_values else None,
    lead_status_flips=sum(1 for prev, cur in zip(window, window[1:], strict=False) if prev.lead_status != cur.lead_status),
    plan_source_flips=sum(1 for prev, cur in zip(window, window[1:], strict=False) if prev.plan_source != cur.plan_source),
    plan_span=(max(plan_values) - min(plan_values)) if plan_values else 0.0,
    high_plan_jerk_count=len(high_plan_jerk_pairs(window, high_jerk_threshold)),
    min_ttc_s=min((sample.ttc_s for sample in episode if sample.ttc_s is not None), default=None),
    max_required_decel_mps2=max((sample.required_decel_mps2 for sample in episode if sample.required_decel_mps2 is not None), default=None),
  )


def _is_suspicious_sample(sample: PlannerTargetSample, large_error_threshold: float) -> bool:
  return is_opposite_intent(sample) or is_strong_opposite_intent(sample) or \
    abs(sample.plan_a_target - sample.a_ego) >= large_error_threshold or \
    (sample.plan_should_stop and sample.v_ego > MOVING_SPEED and not sample.brake_pressed and sample.a_ego > -0.2)


def _is_lead_risk_sample(sample: PlannerTargetSample) -> bool:
  return (
    sample.required_decel_mps2 is not None and sample.required_decel_mps2 >= HIGH_REQUIRED_DECEL_THRESHOLD_MPS2
  ) or (
    sample.ttc_s is not None and sample.ttc_s <= LOW_TTC_THRESHOLD_S
  )


def _ttc_s(lead_status: bool, d_rel: float | None, closing_speed: float | None) -> float | None:
  if not lead_status or d_rel is None or closing_speed is None or closing_speed <= MIN_CLOSING_SPEED_MPS or d_rel <= 0.0:
    return None
  return float(d_rel / closing_speed)


def _required_decel_mps2(lead_status: bool, d_rel: float | None, closing_speed: float | None, epsilon: float = 0.1) -> float | None:
  if not lead_status or d_rel is None or closing_speed is None or closing_speed <= MIN_CLOSING_SPEED_MPS or d_rel <= 0.0:
    return None
  return float((closing_speed ** 2) / (2.0 * max(d_rel, epsilon)))


def _time_headway_s(lead_status: bool, d_rel: float | None, v_ego: float, epsilon: float = 0.1) -> float | None:
  if not lead_status or d_rel is None or d_rel <= 0.0 or v_ego <= MIN_HEADWAY_SPEED_MPS:
    return None
  return float(d_rel / max(v_ego, epsilon))


def _finite_or_none(value: Any) -> float | None:
  if isinstance(value, int | float) and isfinite(float(value)):
    return float(value)
  return None


def _correlation(xs: list[float], ys: list[float]) -> float | None:
  if len(xs) < 2 or len(ys) < 2:
    return None
  x = np.asarray(xs, dtype=float)
  y = np.asarray(ys, dtype=float)
  if float(np.std(x)) <= 1e-9 or float(np.std(y)) <= 1e-9:
    return None
  return float(np.corrcoef(x, y)[0, 1])


def _mean(values: list[float]) -> float:
  return float(np.mean(values)) if values else 0.0


def _optional_mean(values: list[float]) -> float | None:
  return float(np.mean(values)) if values else None


def _percentile(values: list[float], percentile: float) -> float:
  return float(np.percentile(values, percentile)) if values else 0.0


def _ratio(count: int, total: int) -> float:
  return count / total if total else 0.0


def _format_counts(counts: dict[str, int]) -> str:
  if not counts:
    return "none"
  return ", ".join(f"{key}={value}" for key, value in sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _format_optional(value: float | None) -> str:
  return "n/a" if value is None else f"{value:.3f}"


def _state_name(value: str) -> str:
  return str(value).split(".")[-1].lower()


if __name__ == "__main__":
  main()
