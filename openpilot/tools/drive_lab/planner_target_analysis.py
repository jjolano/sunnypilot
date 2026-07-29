from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

from openpilot.tools.drive_lab.route_analysis import ratio as _ratio


PLAN_DIRECTION_THRESHOLD = 0.35
DRIVER_DIRECTION_THRESHOLD = 0.35
STRONG_BRAKE_TARGET = -1.0
STRONG_ACCEL_TARGET = 0.8
STRONG_DRIVER_ACCEL = 0.6
STRONG_DRIVER_BRAKE = -0.6
MOVING_SPEED = 1.0
DEFAULT_EPISODE_GAP_S = 0.6
DEFAULT_EPISODE_CONTEXT_S = 3.0
DEFAULT_LARGE_ERROR_THRESHOLD = 1.2
DEFAULT_HIGH_JERK_THRESHOLD = 8.0
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
  plan_time_s: float | None = None


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


def high_plan_jerk_pairs(
  samples: list[PlannerTargetSample], threshold: float = DEFAULT_HIGH_JERK_THRESHOLD,
) -> list[tuple[PlannerTargetSample, PlannerTargetSample, float]]:
  pairs: list[tuple[PlannerTargetSample, PlannerTargetSample, float]] = []
  samples_by_key: dict[tuple[str, int | None], list[PlannerTargetSample]] = defaultdict(list)
  for sample in samples:
    samples_by_key[(sample.route, sample.segment)].append(sample)
  for route_samples in samples_by_key.values():
    updates: dict[float, PlannerTargetSample] = {}
    for sample in route_samples:
      updates.setdefault(sample.plan_time_s if sample.plan_time_s is not None else sample.t, sample)
    ordered = sorted(updates.items())
    for (prev_t, prev), (cur_t, cur) in zip(ordered, ordered[1:], strict=False):
      dt = cur_t - prev_t
      if 0.0 < dt <= 0.3:
        jerk = (cur.plan_a_target - prev.plan_a_target) / dt
        if abs(jerk) >= threshold:
          pairs.append((prev, cur, jerk))
  return pairs


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
    duration_s=max(0.0, end - start),
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
