#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np

from openpilot.tools.drive_lab.timeline import format_enum, safe_get
from openpilot.tools.drive_lab.route_io import output_report
from openpilot.tools.drive_lab.route_analysis import (
  finite_list as _finite_list,
  finite_or_none as _finite_or_none,
  format_counts as _format_counts,
  iter_route_messages,
  min_optional as _min_optional,
  route_duration,
  route_identity,
)
from openpilot.tools.lib.logreader import LogReader, ReadMode


MOVING_SPEED = 1.0
DEFAULT_MIN_START_SPEED = 4.0
DEFAULT_STOP_SPEED = 1.2
DEFAULT_SLOWDOWN_DELTA_V = 2.5
DEFAULT_DRIVER_DECEL_THRESHOLD = -0.45
DEFAULT_DECEL_RUN_GAP_S = 0.8
DEFAULT_DECEL_RUN_MERGE_GAP_S = 2.0
DEFAULT_CONTEXT_BEFORE_S = 2.0
DEFAULT_CONTEXT_AFTER_S = 4.0
DEFAULT_SIGNAL_LOOKBACK_S = 8.0
DEFAULT_TIMELY_SIGNAL_GRACE_S = 0.5
STRICT_LEAD_RATIO_MAX = 0.10
LOW_LEAD_RATIO_MAX = 0.35

E2E_STOP_APPROACH_EXPECTED_DIST_BP = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 55.0, 60.0]
E2E_STOP_APPROACH_EXPECTED_DIST_V = [8.0, 18.0, 30.0, 43.0, 58.0, 74.0, 85.0, 96.0]
MODEL_STOP_PATH_MAX_DISTANCE = 90.0
MODEL_NEAR_ENDPOINT_MAX_DISTANCE = 80.0
MODEL_NEAR_ENDPOINT_MAX_SPEED = 2.0
MODEL_NEAR_ENDPOINT_ACCEL = -0.8
SCC_EARLY_MODEL_STOP_ACCEL = -1.0
SCC_EARLY_MODEL_STOP_MIN_INITIAL_V = 8.0
SCC_EARLY_MODEL_STOP_MAX_MID_V = 4.0
SCC_EARLY_MODEL_STOP_MIN_SPEED_DROP = 6.0
SCC_EARLY_MODEL_STOP_ENDPOINT_MARGIN = 5.0
SCC_EARLY_MODEL_STOP_MIN_REQUIRED_DECEL = 1.0
SCC_EARLY_MODEL_STOP_EXPECTED_DISTANCE_SCALE = 1.0
TRAFFIC_CONTROL_MAX_DISTANCE = 80.0
TRAFFIC_CONTROL_MIN_DISTANCE = 1.0
SUPPORTED_TRAFFIC_CONTROLS = frozenset((
  "stop",
  "stop_sign",
  "traffic_light",
  "traffic_lights",
  "traffic_signal",
  "traffic_signals",
))

SIGNAL_NAMES = (
  "model_should_stop",
  "model_stop_path",
  "model_path_or_should_stop",
  "model_near_endpoint",
  "scc_early_model_stop_gate",
  "model_accel_le_-0.5",
  "model_accel_le_-1.0",
  "model_accel_le_-1.5",
  "planner_should_stop",
  "planner_e2e_source",
  "planner_e2e_or_should_stop",
  "map_traffic_control",
  "map_model_distance_match",
  "map_model_or_should_stop_match",
  "planner_or_sp_tc_source",
)


@dataclass(frozen=True)
class LeadlessStopSample:
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
  long_control_state: str
  lead_status: bool
  lead_d_rel: float | None
  lead_v_rel: float | None
  model_should_stop: bool
  model_desired_accel: float | None
  model_stop_distance: float | None
  model_endpoint_x: float | None
  model_endpoint_v: float | None
  scc_early_model_stop: bool
  plan_should_stop: bool
  plan_source: str
  plan_a_target: float | None
  sp_source: str
  sp_a_target: float | None
  sp_stack: str
  traffic_control_valid: bool
  traffic_control_type: str
  traffic_control_distance: float | None
  traffic_control_ahead_valid: bool
  traffic_control_ahead_type: str
  traffic_control_ahead_distance: float | None


@dataclass(frozen=True)
class RouteLeadlessStopProfile:
  route: str
  route_id: str
  segment: int | None
  samples: int
  duration_s: float
  manual_moving_samples: int
  manual_leadless_moving_samples: int
  active_ratio: float
  episode_counts: dict[str, int]


@dataclass(frozen=True)
class LeadlessStopEpisode:
  route: str
  route_id: str
  segment: int | None
  start_time_s: float
  end_time_s: float
  duration_s: float
  kind: str
  sample_count: int
  v_start: float
  v_end: float
  delta_v: float
  mean_accel: float
  min_accel: float
  brake_ratio: float
  gas_ratio: float
  lead_ratio: float
  min_lead_d_rel: float | None
  first_brake_time_s: float | None
  first_decel_time_s: float | None
  stop_time_s: float | None
  planner_sources: dict[str, int]
  sp_sources: dict[str, int]
  traffic_control_types: dict[str, int]
  signals: dict[str, dict[str, float | bool | None]]


@dataclass(frozen=True)
class StopSignalMetric:
  episodes: int
  hit_count: int
  timely_hit_count: int
  recall_any: float
  recall_timely: float
  false_positive_clusters: int
  precision_proxy: float
  median_earliest_rel_onset_s: float | None
  p10_earliest_rel_onset_s: float | None


@dataclass(frozen=True)
class FalsePositiveCluster:
  signal_name: str
  scope: str
  route: str
  route_id: str
  segment: int | None
  start_time_s: float
  end_time_s: float
  duration_s: float
  sample_count: int
  context_start_time_s: float
  context_end_time_s: float
  v_start: float
  v_end: float
  delta_v: float
  min_v_ego: float
  max_v_ego: float
  mean_accel: float
  min_accel: float
  brake_ratio: float
  gas_ratio: float
  lead_ratio: float
  min_lead_d_rel: float | None
  active_ratio: float
  long_active_ratio: float
  first_brake_time_s: float | None
  first_decel_time_s: float | None
  model_desired_accel_min: float | None
  model_stop_distance_min: float | None
  model_endpoint_x_min: float | None
  model_endpoint_v_min: float | None
  planner_sources: dict[str, int]
  sp_sources: dict[str, int]
  traffic_control_types: dict[str, int]


@dataclass(frozen=True)
class LeadlessStopCorrelationSummary:
  route_count: int
  sample_count: int
  manual_leadless_moving_sample_count: int
  episode_counts: dict[str, int]
  signal_metrics: dict[str, dict[str, StopSignalMetric]]
  false_positive_clusters: list[FalsePositiveCluster]
  route_profiles: list[RouteLeadlessStopProfile]
  episodes: list[LeadlessStopEpisode]


def main() -> None:
  parser = argparse.ArgumentParser(description="Correlate leadless human slowdown/stop episodes with model and map stop signals.")
  parser.add_argument("routes", nargs="+", help="Routes, segment ranges, log files, or URLs accepted by LogReader")
  parser.add_argument("--qlog", action="store_true", help="Prefer qlogs instead of rlogs")
  parser.add_argument("--json", action="store_true", help="Print JSON instead of text")
  parser.add_argument("--output", help="Write JSON summary to this path")
  parser.add_argument("--episodes", type=int, default=20, help="Number of episodes to show in text output")
  parser.add_argument("--false-positives", type=int, default=20, help="Number of false-positive clusters to show in text output")
  parser.add_argument("--min-start-speed", type=float, default=DEFAULT_MIN_START_SPEED)
  parser.add_argument("--stop-speed", type=float, default=DEFAULT_STOP_SPEED)
  parser.add_argument("--slowdown-delta-v", type=float, default=DEFAULT_SLOWDOWN_DELTA_V)
  parser.add_argument("--driver-decel", type=float, default=DEFAULT_DRIVER_DECEL_THRESHOLD)
  parser.add_argument("--signal-lookback", type=float, default=DEFAULT_SIGNAL_LOOKBACK_S)
  args = parser.parse_args()

  read_mode = ReadMode.QLOG if args.qlog else ReadMode.AUTO
  samples_by_route = {
    route: extract_leadless_stop_samples(route, read_mode)
    for route in args.routes
  }
  summary = summarize_leadless_stop_correlation(
    samples_by_route,
    min_start_speed=args.min_start_speed,
    stop_speed=args.stop_speed,
    slowdown_delta_v=args.slowdown_delta_v,
    driver_decel_threshold=args.driver_decel,
    signal_lookback_s=args.signal_lookback,
  )
  payload = asdict(summary)
  class _LeadlessPayload:
    def to_dict(self):
      return payload

  print(output_report(_LeadlessPayload(), json_output=args.json, renderer=lambda _: render_leadless_stop_summary(
    summary,
    max_episodes=args.episodes,
    max_false_positives=args.false_positives,
  ), output_path=args.output))


def extract_leadless_stop_samples(route: str, read_mode: ReadMode) -> list[LeadlessStopSample]:
  route_id, segment = route_identity(route)
  state: dict[str, Any] = {
    "selfdrive_active": False,
    "long_active": False,
    "long_control_state": "unknown",
    "lead_status": False,
    "lead_d_rel": None,
    "lead_v_rel": None,
    "model_should_stop": False,
    "model_desired_accel": None,
    "model_stop_distance": None,
    "model_endpoint_x": None,
    "model_endpoint_v": None,
    "scc_early_model_stop": False,
    "plan_should_stop": False,
    "plan_source": "unknown",
    "plan_a_target": None,
    "sp_source": "unknown",
    "sp_a_target": None,
    "sp_stack": "unknown",
    "traffic_control_valid": False,
    "traffic_control_type": "",
    "traffic_control_distance": None,
    "traffic_control_ahead_valid": False,
    "traffic_control_ahead_type": "",
    "traffic_control_ahead_distance": None,
  }
  samples: list[LeadlessStopSample] = []

  for route_msg in iter_route_messages(route, read_mode, log_reader_factory=LogReader):
    typ = route_msg.typ
    payload = route_msg.payload

    if typ == "selfdriveState":
      state["selfdrive_active"] = bool(safe_get(payload, "active", False))
    elif typ == "carControl":
      state["long_active"] = bool(safe_get(payload, "longActive", False))
    elif typ == "controlsState":
      state["long_control_state"] = format_enum(safe_get(payload, "longControlState", "unknown"))
    elif typ == "radarState":
      lead_status, lead_d_rel, lead_v_rel = _nearest_lead(safe_get(payload, "leadOne"), safe_get(payload, "leadTwo"))
      state["lead_status"] = lead_status
      state["lead_d_rel"] = lead_d_rel
      state["lead_v_rel"] = lead_v_rel
    elif typ == "modelV2":
      state["model_should_stop"] = bool(safe_get(payload, "action.shouldStop", False))
      state["model_desired_accel"] = _finite_or_none(safe_get(payload, "action.desiredAcceleration"))
      stop_distance, endpoint_x, endpoint_v = model_stop_context(payload)
      state["model_stop_distance"] = stop_distance
      state["model_endpoint_x"] = endpoint_x
      state["model_endpoint_v"] = endpoint_v
      state["scc_early_model_stop"] = scc_early_model_stop_context(payload)
    elif typ == "longitudinalPlan":
      state["plan_should_stop"] = bool(safe_get(payload, "shouldStop", False))
      state["plan_source"] = format_enum(safe_get(payload, "longitudinalPlanSource", "unknown"))
      state["plan_a_target"] = _finite_or_none(safe_get(payload, "aTarget"))
    elif typ == "longitudinalPlanSP":
      state["sp_source"] = format_enum(safe_get(payload, "longitudinalPlanSource", "unknown"))
      state["sp_a_target"] = _finite_or_none(safe_get(payload, "aTarget"))
      state["sp_stack"] = format_enum(safe_get(payload, "stack.actuatedStack", "unknown"))
    elif typ == "liveMapDataSP":
      state["traffic_control_valid"] = bool(safe_get(payload, "trafficControlValid", False))
      state["traffic_control_type"] = normalize_traffic_control(safe_get(payload, "trafficControl", ""))
      state["traffic_control_distance"] = _finite_or_none(safe_get(payload, "trafficControlDistance"))
      state["traffic_control_ahead_valid"] = bool(safe_get(payload, "trafficControlAheadValid", False))
      state["traffic_control_ahead_type"] = normalize_traffic_control(safe_get(payload, "trafficControlAhead", ""))
      state["traffic_control_ahead_distance"] = _finite_or_none(safe_get(payload, "trafficControlAheadDistance"))
    elif typ == "carState":
      v_ego = _finite_or_none(safe_get(payload, "vEgo"))
      a_ego = _finite_or_none(safe_get(payload, "aEgo"))
      if v_ego is None or a_ego is None:
        continue
      samples.append(LeadlessStopSample(
        route=route,
        route_id=route_id,
        segment=segment,
        t=route_msg.t,
        v_ego=v_ego,
        a_ego=a_ego,
        gas_pressed=bool(safe_get(payload, "gasPressed", False)),
        brake_pressed=bool(safe_get(payload, "brakePressed", False)),
        standstill=bool(safe_get(payload, "standstill", False)),
        selfdrive_active=bool(state["selfdrive_active"]),
        long_active=bool(state["long_active"]),
        long_control_state=str(state["long_control_state"]),
        lead_status=bool(state["lead_status"]),
        lead_d_rel=state["lead_d_rel"],
        lead_v_rel=state["lead_v_rel"],
        model_should_stop=bool(state["model_should_stop"]),
        model_desired_accel=state["model_desired_accel"],
        model_stop_distance=state["model_stop_distance"],
        model_endpoint_x=state["model_endpoint_x"],
        model_endpoint_v=state["model_endpoint_v"],
        scc_early_model_stop=bool(state["scc_early_model_stop"]),
        plan_should_stop=bool(state["plan_should_stop"]),
        plan_source=str(state["plan_source"]),
        plan_a_target=state["plan_a_target"],
        sp_source=str(state["sp_source"]),
        sp_a_target=state["sp_a_target"],
        sp_stack=str(state["sp_stack"]),
        traffic_control_valid=bool(state["traffic_control_valid"]),
        traffic_control_type=str(state["traffic_control_type"]),
        traffic_control_distance=state["traffic_control_distance"],
        traffic_control_ahead_valid=bool(state["traffic_control_ahead_valid"]),
        traffic_control_ahead_type=str(state["traffic_control_ahead_type"]),
        traffic_control_ahead_distance=state["traffic_control_ahead_distance"],
      ))
  return samples


def summarize_leadless_stop_correlation(samples_by_route: dict[str, list[LeadlessStopSample]],
                                        min_start_speed: float = DEFAULT_MIN_START_SPEED,
                                        stop_speed: float = DEFAULT_STOP_SPEED,
                                        slowdown_delta_v: float = DEFAULT_SLOWDOWN_DELTA_V,
                                        driver_decel_threshold: float = DEFAULT_DRIVER_DECEL_THRESHOLD,
                                        signal_lookback_s: float = DEFAULT_SIGNAL_LOOKBACK_S) -> LeadlessStopCorrelationSummary:
  profiles: list[RouteLeadlessStopProfile] = []
  episodes: list[LeadlessStopEpisode] = []
  for route, samples in samples_by_route.items():
    route_episodes = build_leadless_stop_episodes(
      samples,
      min_start_speed=min_start_speed,
      stop_speed=stop_speed,
      slowdown_delta_v=slowdown_delta_v,
      driver_decel_threshold=driver_decel_threshold,
      signal_lookback_s=signal_lookback_s,
    )
    episodes.extend(route_episodes)
    profiles.append(build_route_profile(route, samples, route_episodes))

  all_samples = [sample for samples in samples_by_route.values() for sample in samples]
  episodes = sorted(episodes, key=lambda e: (e.route_id, e.segment if e.segment is not None else -1, e.start_time_s))
  return LeadlessStopCorrelationSummary(
    route_count=len(samples_by_route),
    sample_count=len(all_samples),
    manual_leadless_moving_sample_count=sum(
      1 for sample in all_samples if is_manual_preview_sample(sample) and not sample.lead_status and sample.v_ego > MOVING_SPEED
    ),
    episode_counts=dict(Counter(episode.kind for episode in episodes)),
    signal_metrics=build_signal_metrics(all_samples, episodes, signal_lookback_s=signal_lookback_s),
    false_positive_clusters=build_false_positive_clusters(
      all_samples,
      episodes,
      signal_lookback_s=signal_lookback_s,
      driver_decel_threshold=driver_decel_threshold,
    ),
    route_profiles=profiles,
    episodes=episodes,
  )


def build_route_profile(route: str, samples: list[LeadlessStopSample], episodes: list[LeadlessStopEpisode]) -> RouteLeadlessStopProfile:
  route_id, segment = route_identity(route)
  active_count = sum(1 for sample in samples if sample.selfdrive_active or sample.long_active)
  return RouteLeadlessStopProfile(
    route=route,
    route_id=route_id,
    segment=segment,
    samples=len(samples),
    duration_s=route_duration(samples),
    manual_moving_samples=sum(1 for sample in samples if is_manual_preview_sample(sample) and sample.v_ego > MOVING_SPEED),
    manual_leadless_moving_samples=sum(
      1 for sample in samples if is_manual_preview_sample(sample) and not sample.lead_status and sample.v_ego > MOVING_SPEED
    ),
    active_ratio=active_count / len(samples) if samples else 1.0,
    episode_counts=dict(Counter(episode.kind for episode in episodes)),
  )


def build_leadless_stop_episodes(samples: list[LeadlessStopSample],
                                 min_start_speed: float = DEFAULT_MIN_START_SPEED,
                                 stop_speed: float = DEFAULT_STOP_SPEED,
                                 slowdown_delta_v: float = DEFAULT_SLOWDOWN_DELTA_V,
                                 driver_decel_threshold: float = DEFAULT_DRIVER_DECEL_THRESHOLD,
                                 decel_run_gap_s: float = DEFAULT_DECEL_RUN_GAP_S,
                                 decel_run_merge_gap_s: float = DEFAULT_DECEL_RUN_MERGE_GAP_S,
                                 context_before_s: float = DEFAULT_CONTEXT_BEFORE_S,
                                 context_after_s: float = DEFAULT_CONTEXT_AFTER_S,
                                 signal_lookback_s: float = DEFAULT_SIGNAL_LOOKBACK_S,
                                 timely_signal_grace_s: float = DEFAULT_TIMELY_SIGNAL_GRACE_S) -> list[LeadlessStopEpisode]:
  ordered = sorted(samples, key=lambda sample: sample.t)
  episodes = []
  for run in _decel_runs(ordered, driver_decel_threshold, decel_run_gap_s, decel_run_merge_gap_s):
    episode = _summarize_decel_episode(
      ordered,
      run,
      min_start_speed=min_start_speed,
      stop_speed=stop_speed,
      slowdown_delta_v=slowdown_delta_v,
      driver_decel_threshold=driver_decel_threshold,
      context_before_s=context_before_s,
      context_after_s=context_after_s,
      signal_lookback_s=signal_lookback_s,
      timely_signal_grace_s=timely_signal_grace_s,
    )
    if episode is not None:
      episodes.append(episode)
  return episodes


def build_signal_metrics(samples: list[LeadlessStopSample], episodes: list[LeadlessStopEpisode],
                         signal_lookback_s: float = DEFAULT_SIGNAL_LOOKBACK_S) -> dict[str, dict[str, StopSignalMetric]]:
  strict = [episode for episode in episodes if episode.kind.startswith("strict_leadless")]
  review = [episode for episode in episodes if episode.kind.startswith("strict_leadless") or episode.kind.startswith("low_lead_flicker")]
  metrics: dict[str, dict[str, StopSignalMetric]] = {}
  for signal_name in SIGNAL_NAMES:
    metrics[signal_name] = {
      "strict": _signal_metric(samples, strict, signal_name, signal_lookback_s),
      "review": _signal_metric(samples, review, signal_name, signal_lookback_s),
    }
  return metrics


def build_false_positive_clusters(samples: list[LeadlessStopSample], episodes: list[LeadlessStopEpisode],
                                  signal_lookback_s: float = DEFAULT_SIGNAL_LOOKBACK_S,
                                  driver_decel_threshold: float = DEFAULT_DRIVER_DECEL_THRESHOLD) -> list[FalsePositiveCluster]:
  strict = [episode for episode in episodes if episode.kind.startswith("strict_leadless")]
  review = [episode for episode in episodes if episode.kind.startswith("strict_leadless") or episode.kind.startswith("low_lead_flicker")]
  clusters: list[FalsePositiveCluster] = []
  for scope, scope_episodes in (("strict", strict), ("review", review)):
    for signal_name in SIGNAL_NAMES:
      signal_samples = _false_positive_signal_samples(samples, scope_episodes, signal_name, signal_lookback_s)
      clusters.extend(
        _summarize_false_positive_cluster(samples, cluster, signal_name, scope, driver_decel_threshold)
        for cluster in _cluster_signal_samples(signal_samples)
      )
  return sorted(clusters, key=lambda cluster: (cluster.scope, cluster.signal_name, cluster.route_id, cluster.segment or -1, cluster.start_time_s))


def render_leadless_stop_summary(summary: LeadlessStopCorrelationSummary, max_episodes: int = 20,
                                 max_false_positives: int = 20) -> str:
  lines = [
    "Drive Lab leadless stop correlation",
    f"routes={summary.route_count} samples={summary.sample_count} "
    + f"manual_leadless_moving={summary.manual_leadless_moving_sample_count}",
    "Episodes: " + _format_counts(summary.episode_counts),
    "Signal metrics, strict leadless episodes:",
  ]
  for signal_name, metric_by_scope in _rank_signal_metrics(summary.signal_metrics, "strict"):
    metric = metric_by_scope["strict"]
    lines.append(
      f"  {signal_name:30s} timely={metric.recall_timely:.3f} any={metric.recall_any:.3f} "
      + f"fp={metric.false_positive_clusters} precision_proxy={metric.precision_proxy:.3f} "
      + f"median_rel={_format_optional_seconds(metric.median_earliest_rel_onset_s)}"
    )

  lines.append("Signal metrics, strict + low-lead review episodes:")
  for signal_name, metric_by_scope in _rank_signal_metrics(summary.signal_metrics, "review"):
    metric = metric_by_scope["review"]
    lines.append(
      f"  {signal_name:30s} timely={metric.recall_timely:.3f} any={metric.recall_any:.3f} "
      + f"fp={metric.false_positive_clusters} precision_proxy={metric.precision_proxy:.3f} "
      + f"median_rel={_format_optional_seconds(metric.median_earliest_rel_onset_s)}"
    )

  lines.append("Routes:")
  for profile in summary.route_profiles:
    segment = f"--{profile.segment}" if profile.segment is not None else ""
    lines.append(
      f"  {profile.route_id}{segment}: samples {profile.samples}, manual moving {profile.manual_moving_samples}, "
      + f"leadless moving {profile.manual_leadless_moving_samples}, active {profile.active_ratio:.3f}, "
      + "episodes " + _format_counts(profile.episode_counts)
    )

  lines.append("Episodes:")
  if not summary.episodes:
    lines.append("  none")
  for episode in summary.episodes[:max_episodes]:
    segment = f"--{episode.segment}" if episode.segment is not None else ""
    hits = [name for name in SIGNAL_NAMES if episode.signals.get(name, {}).get("hit")]
    lines.append(
      f"  {episode.route_id}{segment} {episode.start_time_s:.1f}-{episode.end_time_s:.1f}s {episode.kind} "
      + f"v={episode.v_start:.1f}->{episode.v_end:.1f} dv={episode.delta_v:.1f} "
      + f"brake={episode.brake_ratio:.2f} lead={episode.lead_ratio:.2f} "
      + f"sources={_format_counts(episode.planner_sources)} tc={_format_counts(episode.traffic_control_types)} "
      + "signals=" + (", ".join(hits) if hits else "none")
    )

  lines.append("False-positive clusters:")
  if not summary.false_positive_clusters:
    lines.append("  none")
  for cluster in summary.false_positive_clusters[:max_false_positives]:
    segment = f"--{cluster.segment}" if cluster.segment is not None else ""
    lines.append(
      f"  {cluster.scope}:{cluster.signal_name} {cluster.route_id}{segment} "
      + f"{cluster.start_time_s:.1f}-{cluster.end_time_s:.1f}s dur={cluster.duration_s:.1f}s "
      + f"v={cluster.v_start:.1f}->{cluster.v_end:.1f} dv={cluster.delta_v:.1f} "
      + f"a_min={cluster.min_accel:.2f} brake={cluster.brake_ratio:.2f} gas={cluster.gas_ratio:.2f} "
      + f"lead={cluster.lead_ratio:.2f} active={cluster.active_ratio:.2f} "
      + f"model_accel_min={_format_optional_float(cluster.model_desired_accel_min)} "
      + f"endpoint_min={_format_optional_float(cluster.model_endpoint_x_min)} "
      + f"sources={_format_counts(cluster.planner_sources)}"
    )
  return "\n".join(lines)


def model_stop_context(model_data: Any) -> tuple[float | None, float | None, float | None]:
  positions = _finite_list(safe_get(model_data, "position.x"))
  velocities = _finite_list(safe_get(model_data, "velocity.x"))
  endpoint_x = positions[-1] if positions else None
  endpoint_v = velocities[-1] if velocities else None
  stop_distance = None
  for x, v in zip(positions, velocities, strict=False):
    if x > 0.0 and v <= 1.0:
      stop_distance = float(x)
      break
  if stop_distance is None and bool(safe_get(model_data, "action.shouldStop", False)) and endpoint_x is not None and endpoint_v is not None:
    if endpoint_x > 0.0 and endpoint_v <= 2.0:
      stop_distance = float(endpoint_x)
  return stop_distance, endpoint_x, endpoint_v


def scc_early_model_stop_context(model_data: Any) -> bool:
  desired_accel = _finite_or_none(safe_get(model_data, "action.desiredAcceleration"))
  if desired_accel is None or desired_accel > SCC_EARLY_MODEL_STOP_ACCEL:
    return False

  positions = np.asarray(_finite_list(safe_get(model_data, "position.x")), dtype=float)
  velocities = np.asarray(_finite_list(safe_get(model_data, "velocity.x")), dtype=float)
  if len(positions) < 3 or len(positions) != len(velocities):
    return False

  initial_v = float(velocities[0])
  endpoint_x = float(positions[-1])
  endpoint_v = max(float(velocities[-1]), 0.0)
  if initial_v < SCC_EARLY_MODEL_STOP_MIN_INITIAL_V or endpoint_x <= 0.0:
    return False
  if initial_v - float(np.min(velocities)) < SCC_EARLY_MODEL_STOP_MIN_SPEED_DROP:
    return False

  expected_distance = float(np.interp(
    initial_v * 3.6,
    E2E_STOP_APPROACH_EXPECTED_DIST_BP,
    E2E_STOP_APPROACH_EXPECTED_DIST_V,
  ))
  if endpoint_x > expected_distance * SCC_EARLY_MODEL_STOP_EXPECTED_DISTANCE_SCALE:
    return False

  required_decel = (initial_v**2 - endpoint_v**2) / (2.0 * endpoint_x)
  if required_decel < SCC_EARLY_MODEL_STOP_MIN_REQUIRED_DECEL:
    return False

  middle_positions = positions[1:-1]
  middle_velocities = velocities[1:-1]
  return bool(np.any(
    (middle_positions >= 0.0) &
    (endpoint_x - middle_positions >= SCC_EARLY_MODEL_STOP_ENDPOINT_MARGIN) &
    (middle_velocities <= SCC_EARLY_MODEL_STOP_MAX_MID_V)
  ))


def normalize_traffic_control(control_type: Any) -> str:
  return str(control_type or "").strip().lower().replace("-", "_").replace(" ", "_")


def model_stop_matches_map_distance(model_distance: float | None, map_distance: float | None) -> bool:
  if model_distance is None or map_distance is None:
    return False
  if map_distance <= TRAFFIC_CONTROL_MIN_DISTANCE:
    return model_distance <= TRAFFIC_CONTROL_MAX_DISTANCE
  return abs(model_distance - map_distance) <= max(12.0, map_distance * 0.35)


def traffic_control_candidate(sample: LeadlessStopSample) -> tuple[str, float | None]:
  candidates = []
  if sample.traffic_control_ahead_valid and sample.traffic_control_ahead_type in SUPPORTED_TRAFFIC_CONTROLS:
    if sample.traffic_control_ahead_distance is not None:
      candidates.append((sample.traffic_control_ahead_type, max(0.0, sample.traffic_control_ahead_distance)))
  if sample.traffic_control_valid and sample.traffic_control_type in SUPPORTED_TRAFFIC_CONTROLS:
    if sample.traffic_control_distance is not None:
      candidates.append((sample.traffic_control_type, max(0.0, sample.traffic_control_distance)))
  return min(candidates, key=lambda item: item[1]) if candidates else ("", None)


def is_manual_preview_sample(sample: LeadlessStopSample) -> bool:
  return not sample.selfdrive_active and not sample.long_active


def signal_active(sample: LeadlessStopSample, signal_name: str) -> bool:
  traffic_control, traffic_control_distance = traffic_control_candidate(sample)
  map_traffic_control = bool(traffic_control and traffic_control_distance is not None and traffic_control_distance <= TRAFFIC_CONTROL_MAX_DISTANCE)
  model_stop_path = sample.model_stop_distance is not None and sample.model_stop_distance <= MODEL_STOP_PATH_MAX_DISTANCE
  model_near_endpoint = bool(
    sample.model_desired_accel is not None and sample.model_desired_accel <= MODEL_NEAR_ENDPOINT_ACCEL and
    sample.model_endpoint_x is not None and sample.model_endpoint_x <= MODEL_NEAR_ENDPOINT_MAX_DISTANCE and
    (sample.model_endpoint_v is None or sample.model_endpoint_v <= MODEL_NEAR_ENDPOINT_MAX_SPEED)
  )
  source = sample.plan_source.lower()
  sp_source = sample.sp_source.lower()

  if signal_name == "model_should_stop":
    return sample.model_should_stop
  if signal_name == "model_stop_path":
    return model_stop_path
  if signal_name == "model_path_or_should_stop":
    return sample.model_should_stop or model_stop_path
  if signal_name == "model_near_endpoint":
    return model_stop_path or model_near_endpoint
  if signal_name == "scc_early_model_stop_gate":
    return sample.scc_early_model_stop
  if signal_name == "model_accel_le_-0.5":
    return sample.model_desired_accel is not None and sample.model_desired_accel <= -0.5
  if signal_name == "model_accel_le_-1.0":
    return sample.model_desired_accel is not None and sample.model_desired_accel <= -1.0
  if signal_name == "model_accel_le_-1.5":
    return sample.model_desired_accel is not None and sample.model_desired_accel <= -1.5
  if signal_name == "planner_should_stop":
    return sample.plan_should_stop
  if signal_name == "planner_e2e_source":
    return source in {"e2e", "model"}
  if signal_name == "planner_e2e_or_should_stop":
    return sample.plan_should_stop or source in {"e2e", "model"}
  if signal_name == "map_traffic_control":
    return map_traffic_control
  if signal_name == "map_model_distance_match":
    return map_traffic_control and model_stop_matches_map_distance(sample.model_stop_distance, traffic_control_distance)
  if signal_name == "map_model_or_should_stop_match":
    return map_traffic_control and (model_stop_matches_map_distance(sample.model_stop_distance, traffic_control_distance) or sample.model_should_stop)
  if signal_name == "planner_or_sp_tc_source":
    return "osm" in source or "traffic" in source or "osm" in sp_source or "traffic" in sp_source
  raise ValueError(f"unknown stop signal {signal_name!r}")


def _decel_runs(samples: list[LeadlessStopSample], driver_decel_threshold: float,
                decel_run_gap_s: float, decel_run_merge_gap_s: float) -> list[list[LeadlessStopSample]]:
  raw_runs: list[list[LeadlessStopSample]] = []
  current: list[LeadlessStopSample] = []
  for sample in samples:
    decel_sample = _is_driver_decel_sample(sample, driver_decel_threshold)
    if decel_sample and (not current or sample.t - current[-1].t <= decel_run_gap_s):
      current.append(sample)
    elif decel_sample:
      if current:
        raw_runs.append(current)
      current = [sample]
    elif current and sample.t - current[-1].t > decel_run_gap_s:
      raw_runs.append(current)
      current = []
  if current:
    raw_runs.append(current)

  merged: list[list[LeadlessStopSample]] = []
  for run in raw_runs:
    if merged and run[0].t - merged[-1][-1].t <= decel_run_merge_gap_s:
      merged[-1].extend(run)
    else:
      merged.append(list(run))
  return merged


def _summarize_decel_episode(samples: list[LeadlessStopSample], run: list[LeadlessStopSample],
                             min_start_speed: float, stop_speed: float, slowdown_delta_v: float,
                             driver_decel_threshold: float, context_before_s: float, context_after_s: float,
                             signal_lookback_s: float, timely_signal_grace_s: float) -> LeadlessStopEpisode | None:
  start_t = run[0].t
  end_t = run[-1].t
  context = [sample for sample in samples if start_t - context_before_s <= sample.t <= end_t + context_after_s]
  if not context:
    return None
  pre_context = [sample for sample in samples if start_t - context_before_s <= sample.t <= start_t + 0.5] or run
  post_context = [sample for sample in samples if start_t <= sample.t <= end_t + context_after_s] or run
  v_start = max(sample.v_ego for sample in pre_context)
  v_end = min(sample.v_ego for sample in post_context)
  delta_v = v_end - v_start
  if v_start < min_start_speed or not (v_end <= stop_speed or delta_v <= -slowdown_delta_v):
    return None

  lead_ratio = sum(1 for sample in context if sample.lead_status) / len(context)
  lead_distances = [sample.lead_d_rel for sample in context if sample.lead_d_rel is not None]
  brake_times = [sample.t for sample in context if sample.brake_pressed]
  decel_times = [sample.t for sample in context if sample.a_ego <= driver_decel_threshold]
  stop_times = [sample.t for sample in context if sample.v_ego <= stop_speed]

  return LeadlessStopEpisode(
    route=run[0].route,
    route_id=run[0].route_id,
    segment=run[0].segment,
    start_time_s=start_t,
    end_time_s=end_t,
    duration_s=max(0.0, end_t - start_t),
    kind=_episode_kind(v_end, stop_speed, lead_ratio),
    sample_count=len(context),
    v_start=v_start,
    v_end=v_end,
    delta_v=delta_v,
    mean_accel=float(np.mean([sample.a_ego for sample in context])),
    min_accel=min(sample.a_ego for sample in context),
    brake_ratio=sum(1 for sample in context if sample.brake_pressed) / len(context),
    gas_ratio=sum(1 for sample in context if sample.gas_pressed) / len(context),
    lead_ratio=lead_ratio,
    min_lead_d_rel=min(lead_distances) if lead_distances else None,
    first_brake_time_s=min(brake_times) if brake_times else None,
    first_decel_time_s=min(decel_times) if decel_times else None,
    stop_time_s=min(stop_times) if stop_times else None,
    planner_sources=dict(Counter(sample.plan_source for sample in context)),
    sp_sources=dict(Counter(sample.sp_source for sample in context)),
    traffic_control_types=_traffic_control_counts(context),
    signals=_signal_windows(samples, start_t, end_t, signal_lookback_s, timely_signal_grace_s),
  )


def _signal_windows(samples: list[LeadlessStopSample], start_t: float, end_t: float,
                    signal_lookback_s: float, timely_signal_grace_s: float) -> dict[str, dict[str, float | bool | None]]:
  windows = {}
  for signal_name in SIGNAL_NAMES:
    hits = [
      sample for sample in samples
      if start_t - signal_lookback_s <= sample.t <= end_t + 1.0 and signal_active(sample, signal_name)
    ]
    timely_hits = [sample for sample in hits if sample.t <= start_t + timely_signal_grace_s]
    windows[signal_name] = {
      "hit": bool(hits),
      "timely": bool(timely_hits),
      "earliest_rel_onset_s": min(sample.t for sample in hits) - start_t if hits else None,
      "earliest_timely_rel_onset_s": min(sample.t for sample in timely_hits) - start_t if timely_hits else None,
    }
  return windows


def _signal_metric(samples: list[LeadlessStopSample], episodes: list[LeadlessStopEpisode],
                   signal_name: str, signal_lookback_s: float) -> StopSignalMetric:
  hit_episodes = [episode for episode in episodes if bool(episode.signals[signal_name]["hit"])]
  timely_hit_episodes = [episode for episode in episodes if bool(episode.signals[signal_name]["timely"])]
  rel_onsets = [
    float(episode.signals[signal_name]["earliest_rel_onset_s"])
    for episode in hit_episodes
    if episode.signals[signal_name]["earliest_rel_onset_s"] is not None
  ]
  false_positive_samples = _false_positive_signal_samples(samples, episodes, signal_name, signal_lookback_s)
  false_positive_clusters = len(_cluster_signal_samples(false_positive_samples))
  timely_hit_count = len(timely_hit_episodes)
  return StopSignalMetric(
    episodes=len(episodes),
    hit_count=len(hit_episodes),
    timely_hit_count=timely_hit_count,
    recall_any=len(hit_episodes) / len(episodes) if episodes else 0.0,
    recall_timely=timely_hit_count / len(episodes) if episodes else 0.0,
    false_positive_clusters=false_positive_clusters,
    precision_proxy=timely_hit_count / (timely_hit_count + false_positive_clusters) if timely_hit_count + false_positive_clusters else 0.0,
    median_earliest_rel_onset_s=float(np.median(rel_onsets)) if rel_onsets else None,
    p10_earliest_rel_onset_s=float(np.percentile(rel_onsets, 10.0)) if rel_onsets else None,
  )


def _false_positive_signal_samples(samples: list[LeadlessStopSample], episodes: list[LeadlessStopEpisode],
                                   signal_name: str, signal_lookback_s: float) -> list[LeadlessStopSample]:
  windows = [(episode.route, episode.start_time_s - signal_lookback_s, episode.end_time_s + 2.0) for episode in episodes]
  return [
    sample for sample in samples
    if is_manual_preview_sample(sample) and not sample.lead_status and sample.v_ego > MOVING_SPEED and
       signal_active(sample, signal_name) and not _sample_in_windows(sample, windows)
  ]


def _summarize_false_positive_cluster(samples: list[LeadlessStopSample], cluster: list[LeadlessStopSample],
                                      signal_name: str, scope: str,
                                      driver_decel_threshold: float) -> FalsePositiveCluster:
  start_t = cluster[0].t
  end_t = cluster[-1].t
  context = [
    sample for sample in samples
    if sample.route == cluster[0].route and start_t - DEFAULT_CONTEXT_BEFORE_S <= sample.t <= end_t + DEFAULT_CONTEXT_AFTER_S
  ] or cluster
  lead_distances = [sample.lead_d_rel for sample in context if sample.lead_d_rel is not None]
  brake_times = [sample.t for sample in context if sample.brake_pressed]
  decel_times = [sample.t for sample in context if sample.a_ego <= driver_decel_threshold]
  return FalsePositiveCluster(
    signal_name=signal_name,
    scope=scope,
    route=cluster[0].route,
    route_id=cluster[0].route_id,
    segment=cluster[0].segment,
    start_time_s=start_t,
    end_time_s=end_t,
    duration_s=max(0.0, end_t - start_t),
    sample_count=len(cluster),
    context_start_time_s=context[0].t,
    context_end_time_s=context[-1].t,
    v_start=context[0].v_ego,
    v_end=context[-1].v_ego,
    delta_v=context[-1].v_ego - context[0].v_ego,
    min_v_ego=min(sample.v_ego for sample in context),
    max_v_ego=max(sample.v_ego for sample in context),
    mean_accel=float(np.mean([sample.a_ego for sample in context])),
    min_accel=min(sample.a_ego for sample in context),
    brake_ratio=sum(1 for sample in context if sample.brake_pressed) / len(context),
    gas_ratio=sum(1 for sample in context if sample.gas_pressed) / len(context),
    lead_ratio=sum(1 for sample in context if sample.lead_status) / len(context),
    min_lead_d_rel=min(lead_distances) if lead_distances else None,
    active_ratio=sum(1 for sample in context if sample.selfdrive_active or sample.long_active) / len(context),
    long_active_ratio=sum(1 for sample in context if sample.long_active) / len(context),
    first_brake_time_s=min(brake_times) if brake_times else None,
    first_decel_time_s=min(decel_times) if decel_times else None,
    model_desired_accel_min=_min_optional(sample.model_desired_accel for sample in cluster),
    model_stop_distance_min=_min_optional(sample.model_stop_distance for sample in cluster),
    model_endpoint_x_min=_min_optional(sample.model_endpoint_x for sample in cluster),
    model_endpoint_v_min=_min_optional(sample.model_endpoint_v for sample in cluster),
    planner_sources=dict(Counter(sample.plan_source for sample in context)),
    sp_sources=dict(Counter(sample.sp_source for sample in context)),
    traffic_control_types=_traffic_control_counts(context),
  )


def _is_driver_decel_sample(sample: LeadlessStopSample, driver_decel_threshold: float) -> bool:
  return is_manual_preview_sample(sample) and sample.v_ego > 0.5 and (sample.brake_pressed or sample.a_ego <= driver_decel_threshold)


def _episode_kind(v_end: float, stop_speed: float, lead_ratio: float) -> str:
  stopped = v_end <= stop_speed
  if lead_ratio <= STRICT_LEAD_RATIO_MAX:
    return "strict_leadless_stop" if stopped else "strict_leadless_slowdown"
  if lead_ratio <= LOW_LEAD_RATIO_MAX:
    return "low_lead_flicker_stop" if stopped else "low_lead_flicker_slowdown"
  return "lead_present_stop" if stopped else "lead_present_slowdown"


def _nearest_lead(*leads: Any) -> tuple[bool, float | None, float | None]:
  valid_leads = []
  for lead in leads:
    if not bool(safe_get(lead, "present", False)):
      continue
    d_rel = _finite_or_none(safe_get(lead, "dRel"))
    v_rel = _finite_or_none(safe_get(lead, "vRel"))
    valid_leads.append((float("inf") if d_rel is None else d_rel, v_rel))
  if not valid_leads:
    return False, None, None
  d_rel, v_rel = min(valid_leads, key=lambda item: item[0])
  return True, None if d_rel == float("inf") else d_rel, v_rel


def _traffic_control_counts(samples: Iterable[LeadlessStopSample]) -> dict[str, int]:
  counts: Counter[str] = Counter()
  for sample in samples:
    traffic_control, _ = traffic_control_candidate(sample)
    if traffic_control:
      counts[traffic_control] += 1
  return dict(counts)


def _cluster_signal_samples(samples: list[LeadlessStopSample], gap_s: float = 1.0) -> list[list[LeadlessStopSample]]:
  clusters: list[list[LeadlessStopSample]] = []
  current: list[LeadlessStopSample] = []
  for sample in sorted(samples, key=lambda s: (s.route, s.t)):
    if current and (sample.route != current[-1].route or sample.t - current[-1].t > gap_s):
      clusters.append(current)
      current = []
    current.append(sample)
  if current:
    clusters.append(current)
  return clusters


def _sample_in_windows(sample: LeadlessStopSample, windows: list[tuple[str, float, float]]) -> bool:
  return any(sample.route == route and start_t <= sample.t <= end_t for route, start_t, end_t in windows)


def _format_optional_seconds(value: float | None) -> str:
  return "n/a" if value is None else f"{value:.3f}s"


def _format_optional_float(value: float | None) -> str:
  return "n/a" if value is None else f"{value:.2f}"


def _rank_signal_metrics(metrics: dict[str, dict[str, StopSignalMetric]], scope: str) -> list[tuple[str, dict[str, StopSignalMetric]]]:
  return sorted(
    metrics.items(),
    key=lambda item: (
      item[1][scope].recall_timely,
      item[1][scope].precision_proxy,
      -item[1][scope].false_positive_clusters,
      item[1][scope].recall_any,
    ),
    reverse=True,
  )


if __name__ == "__main__":
  main()
