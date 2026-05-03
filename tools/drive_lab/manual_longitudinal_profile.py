from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import isfinite

import numpy as np


@dataclass(frozen=True)
class ProfileRange:
  low: float
  high: float

  def contains(self, value: float) -> bool:
    return self.low <= value <= self.high


@dataclass(frozen=True)
class SmoothAssertiveEnvelope:
  accel_p10: ProfileRange = ProfileRange(-1.20, -0.45)
  accel_p90: ProfileRange = ProfileRange(0.55, 1.25)
  launch_mean_p50: ProfileRange = ProfileRange(0.45, 0.90)
  launch_mean_p90: ProfileRange = ProfileRange(0.75, 1.25)
  stop_mean_p10: ProfileRange = ProfileRange(-1.25, -0.55)
  stop_mean_p50: ProfileRange = ProfileRange(-0.60, -0.20)
  coast_p50: ProfileRange = ProfileRange(-0.45, -0.18)
  coast_p90: ProfileRange = ProfileRange(-0.35, 0.05)


@dataclass(frozen=True)
class ManualSample:
  route: str
  t: float
  v_ego: float
  a_ego: float
  active: bool
  gas_pressed: bool
  brake_pressed: bool
  lead_status: bool
  lead_d_rel: float | None = None
  lead_v_rel: float | None = None
  lead_v_lead: float | None = None
  lead_a_lead: float | None = None
  lead_model_prob: float | None = None


@dataclass(frozen=True)
class RouteProfile:
  route: str
  samples: int
  duration_s: float
  manual_moving_samples: int
  active_ratio: float
  include: bool


@dataclass(frozen=True)
class SpeedBinSummary:
  label: str
  sample_count: int
  gas_ratio: float
  brake_ratio: float
  coast_ratio: float
  accel: ProfileRange
  coast_accel: ProfileRange


@dataclass(frozen=True)
class FollowingBinSummary:
  label: str
  sample_count: int
  closing_ratio: float
  distance: ProfileRange
  time_gap: ProfileRange
  closing_time: ProfileRange


@dataclass(frozen=True)
class LeadCrawlBucketSummary:
  label: str
  sample_count: int
  gas_ratio: float
  brake_ratio: float
  coast_ratio: float
  gap_excess: ProfileRange
  ego_speed: ProfileRange
  lead_speed: ProfileRange
  relative_speed: ProfileRange
  accel: ProfileRange
  closing_ratio: float
  closing_speed: ProfileRange


@dataclass(frozen=True)
class LeadCrawlEpisodeSummary:
  label: str
  count: int
  duration: ProfileRange
  start_gap_excess: ProfileRange
  end_gap_excess: ProfileRange
  min_gap_excess: ProfileRange
  mean_accel: ProfileRange


@dataclass(frozen=True)
class ManualStyleSummary:
  sample_count: int
  accel: ProfileRange
  lead_launch_count: int
  clear_launch_count: int
  lead_launch_mean_accel: ProfileRange
  clear_launch_mean_accel: ProfileRange
  lead_launch_peak_accel: ProfileRange
  clear_launch_peak_accel: ProfileRange
  lead_stop_count: int
  clear_stop_count: int
  stop_mean_accel: ProfileRange
  stop_peak_decel: ProfileRange
  coast_accel: ProfileRange
  speed_bins: list[SpeedBinSummary]
  following_bins: list[FollowingBinSummary]
  lead_crawl_bins: list[LeadCrawlBucketSummary]
  lead_crawl_episodes: list[LeadCrawlEpisodeSummary]
  style: str


_SPEED_BINS = (
  ("1-3 m/s", 1.0, 3.0),
  ("3-7 m/s", 3.0, 7.0),
  ("7-13 m/s", 7.0, 13.0),
  ("13-20 m/s", 13.0, 20.0),
  ("20+ m/s", 20.0, float("inf")),
)
LEAD_CRAWL_MAX_SPEED = 2.5
LEAD_CRAWL_CLOSING_THRESHOLD = -0.1
_LEAD_CRAWL_BUCKETS = (
  ("open_to_crawl", 2.0, float("inf")),
  ("crawl_to_follow", 1.0, 2.0),
  ("soft_stop", 0.0, 1.0),
  ("inside_stop_target", -float("inf"), 0.0),
)

STOP_DISTANCE = 6.0
LEAD_STOP_PRESENTATION_DISTANCE = 5.0
LEAD_STOP_PRESENTATION_CONFIDENCE_MIN = 0.75
LEAD_STOP_PRESENTATION_V_EGO_BP = [0.0, 3.0]
LEAD_STOP_PRESENTATION_V_LEAD_BP = [0.2, 1.0]
LEAD_STOP_PRESENTATION_DECEL_BP = [0.0, 0.6]


def manual_moving_samples(samples: Iterable[ManualSample]) -> list[ManualSample]:
  return [sample for sample in samples if not sample.active and sample.v_ego > 1.0]


def lead_stop_presentation_distance(v_ego, v_lead, a_lead=0.0, model_prob=1.0):
  confidence_blend = np.interp(model_prob, [LEAD_STOP_PRESENTATION_CONFIDENCE_MIN, 1.0], [0.0, 1.0])
  ego_blend = 1.0 - np.interp(v_ego, LEAD_STOP_PRESENTATION_V_EGO_BP, [0.0, 1.0])
  stopped_blend = 1.0 - np.interp(v_lead, LEAD_STOP_PRESENTATION_V_LEAD_BP, [0.0, 1.0])
  decel_blend = 1.0 - np.interp(np.clip(-a_lead, 0.0, LEAD_STOP_PRESENTATION_DECEL_BP[-1]),
                                LEAD_STOP_PRESENTATION_DECEL_BP, [0.0, 1.0])
  presentation_blend = confidence_blend * ego_blend * stopped_blend * decel_blend
  return STOP_DISTANCE - presentation_blend * (STOP_DISTANCE - LEAD_STOP_PRESENTATION_DISTANCE)


def _lead_speed(sample: ManualSample) -> float | None:
  if sample.lead_v_lead is not None:
    return sample.lead_v_lead
  if sample.lead_v_rel is not None:
    return sample.v_ego + sample.lead_v_rel
  return None


def _lead_accel(sample: ManualSample) -> float:
  return sample.lead_a_lead or 0.0


def _lead_model_prob(sample: ManualSample) -> float:
  return sample.lead_model_prob if sample.lead_model_prob is not None else 1.0


def lead_crawl_gap_excess(sample: ManualSample) -> float | None:
  if not sample.lead_status or sample.lead_d_rel is None:
    return None
  v_lead = _lead_speed(sample)
  if v_lead is None:
    return None
  stop_target = lead_stop_presentation_distance(sample.v_ego, v_lead, _lead_accel(sample), _lead_model_prob(sample))
  return float(sample.lead_d_rel - stop_target)


def build_route_profile(route: str, samples: Iterable[ManualSample], min_manual_moving_samples: int = 1200,
                        max_active_ratio: float = 0.25) -> RouteProfile:
  sample_list = list(samples)
  active_count = sum(1 for sample in sample_list if sample.active)
  active_ratio = active_count / len(sample_list) if sample_list else 1.0
  moving_count = len(manual_moving_samples(sample_list))
  include = moving_count >= min_manual_moving_samples and active_ratio <= max_active_ratio
  duration_s = max((sample.t for sample in sample_list), default=0.0) - min((sample.t for sample in sample_list), default=0.0)
  return RouteProfile(route, len(sample_list), duration_s, moving_count, active_ratio, include)


def summarize_manual_style(samples: Iterable[ManualSample]) -> ManualStyleSummary:
  ordered = sorted((sample for sample in samples if not sample.active), key=lambda sample: (sample.route, sample.t))
  moving = manual_moving_samples(ordered)
  launches = _pedal_episodes(ordered, pedal="gas_pressed")
  stops = _pedal_episodes(ordered, pedal="brake_pressed")
  launch_candidates = [episode for episode in launches if episode["v0"] < 1.5 and episode["v1"] > 5.0 and episode["duration"] > 1.0]
  stop_candidates = [episode for episode in stops if episode["v0"] > 5.0 and episode["v1"] < 1.0 and episode["duration"] > 1.0]
  lead_launches = [episode for episode in launch_candidates if episode["lead"]]
  clear_launches = [episode for episode in launch_candidates if not episode["lead"]]
  lead_stops = [episode for episode in stop_candidates if episode["lead"]]
  clear_stops = [episode for episode in stop_candidates if not episode["lead"]]
  coast_samples = [sample for sample in moving if not sample.gas_pressed and not sample.brake_pressed and sample.v_ego >= 7.0]

  accel = percentile_range([sample.a_ego for sample in moving], 10.0, 90.0)
  lead_launch_mean = percentile_range([episode["mean_accel"] for episode in lead_launches], 50.0, 90.0)
  clear_launch_mean = percentile_range([episode["mean_accel"] for episode in clear_launches], 50.0, 90.0)
  stop_mean = percentile_range([episode["mean_accel"] for episode in stop_candidates], 10.0, 50.0)
  coast = percentile_range([sample.a_ego for sample in coast_samples], 50.0, 90.0)
  launch_mean = lead_launch_mean if lead_launches else clear_launch_mean
  style = classify_style(accel, launch_mean, stop_mean, coast)

  return ManualStyleSummary(
    sample_count=len(moving),
    accel=accel,
    lead_launch_count=len(lead_launches),
    clear_launch_count=len(clear_launches),
    lead_launch_mean_accel=lead_launch_mean,
    clear_launch_mean_accel=clear_launch_mean,
    lead_launch_peak_accel=percentile_range([episode["peak_accel"] for episode in lead_launches], 50.0, 90.0),
    clear_launch_peak_accel=percentile_range([episode["peak_accel"] for episode in clear_launches], 50.0, 90.0),
    lead_stop_count=len(lead_stops),
    clear_stop_count=len(clear_stops),
    stop_mean_accel=stop_mean,
    stop_peak_decel=percentile_range([episode["peak_decel"] for episode in stop_candidates], 10.0, 50.0),
    coast_accel=coast,
    speed_bins=_summarize_speed_bins(moving),
    following_bins=_summarize_following_bins(moving),
    lead_crawl_bins=_summarize_lead_crawl_bins(ordered),
    lead_crawl_episodes=_summarize_lead_crawl_episodes(ordered),
    style=style,
  )


def render_manual_style_summary(summary: ManualStyleSummary, route_profiles: Iterable[RouteProfile] | None = None) -> str:
  lead_launch_line = (
    f"lead launches: {summary.lead_launch_count} mean {summary.lead_launch_mean_accel.low:.3f} to "
    + f"{summary.lead_launch_mean_accel.high:.3f} m/s^2 peak {summary.lead_launch_peak_accel.low:.3f} to "
    + f"{summary.lead_launch_peak_accel.high:.3f} m/s^2"
  )
  clear_launch_line = (
    f"clear launches: {summary.clear_launch_count} mean {summary.clear_launch_mean_accel.low:.3f} to "
    + f"{summary.clear_launch_mean_accel.high:.3f} m/s^2 peak {summary.clear_launch_peak_accel.low:.3f} to "
    + f"{summary.clear_launch_peak_accel.high:.3f} m/s^2"
  )
  stop_approach_line = (
    f"stop approaches: {summary.lead_stop_count + summary.clear_stop_count} "
    + f"(lead {summary.lead_stop_count} clear {summary.clear_stop_count}) mean {summary.stop_mean_accel.low:.3f} to "
    + f"{summary.stop_mean_accel.high:.3f} m/s^2 peak {summary.stop_peak_decel.low:.3f} to "
    + f"{summary.stop_peak_decel.high:.3f} m/s^2"
  )
  lines = [
    "Manual longitudinal style",
    f"style: {summary.style}",
    f"samples: {summary.sample_count}",
    f"accel p10-p90: {summary.accel.low:.3f} to {summary.accel.high:.3f} m/s^2",
    lead_launch_line,
    clear_launch_line,
    stop_approach_line,
    f"coast accel: {summary.coast_accel.low:.3f} to {summary.coast_accel.high:.3f} m/s^2",
  ]
  route_profiles = list(route_profiles or [])
  if route_profiles:
    lines.append("Routes:")
    lines.extend(_render_route_profile(profile) for profile in route_profiles)
  lines.append("Speed bins:")
  lines.extend(_render_speed_bin(speed_bin) for speed_bin in summary.speed_bins)
  if not summary.speed_bins:
    lines.append("  none")
  lines.append("Following bins:")
  lines.extend(_render_following_bin(following_bin) for following_bin in summary.following_bins)
  if not summary.following_bins:
    lines.append("  none")
  return "\n".join(lines)


def _summarize_speed_bins(samples: list[ManualSample]) -> list[SpeedBinSummary]:
  summaries: list[SpeedBinSummary] = []
  for label, low, high in _SPEED_BINS:
    bucket = [sample for sample in samples if low <= sample.v_ego < high]
    if not bucket:
      continue
    coast_samples = [sample for sample in bucket if not sample.gas_pressed and not sample.brake_pressed]
    summaries.append(SpeedBinSummary(
      label=label,
      sample_count=len(bucket),
      gas_ratio=_ratio(sum(1 for sample in bucket if sample.gas_pressed), len(bucket)),
      brake_ratio=_ratio(sum(1 for sample in bucket if sample.brake_pressed), len(bucket)),
      coast_ratio=_ratio(len(coast_samples), len(bucket)),
      accel=percentile_range([sample.a_ego for sample in bucket], 10.0, 90.0),
      coast_accel=percentile_range([sample.a_ego for sample in coast_samples], 50.0, 90.0),
    ))
  return summaries


def _summarize_following_bins(samples: list[ManualSample]) -> list[FollowingBinSummary]:
  summaries: list[FollowingBinSummary] = []
  for label, low, high in _SPEED_BINS:
    bucket = [sample for sample in samples if low <= sample.v_ego < high and sample.lead_status and sample.lead_d_rel is not None and sample.lead_d_rel > 0.0]
    if not bucket:
      continue
    closing_samples = [sample for sample in bucket if sample.lead_v_rel is not None and sample.lead_v_rel < -0.1]
    summaries.append(FollowingBinSummary(
      label=label,
      sample_count=len(bucket),
      closing_ratio=_ratio(len(closing_samples), len(bucket)),
      distance=percentile_range([sample.lead_d_rel for sample in bucket if sample.lead_d_rel is not None], 10.0, 90.0),
      time_gap=percentile_range([sample.lead_d_rel / max(sample.v_ego, 0.1) for sample in bucket if sample.lead_d_rel is not None], 10.0, 90.0),
      closing_time=percentile_range([
        sample.lead_d_rel / max(-(sample.lead_v_rel or 0.0), 0.1)
        for sample in closing_samples
        if sample.lead_d_rel is not None
      ], 10.0, 90.0),
    ))
  return summaries


def _lead_crawl_sample_details(samples: list[ManualSample]) -> list[tuple[ManualSample, float, float]]:
  details: list[tuple[ManualSample, float, float]] = []
  for sample in samples:
    detail = _lead_crawl_sample_detail(sample)
    if detail is None:
      continue
    details.append(detail)
  return details


def _lead_crawl_sample_detail(sample: ManualSample) -> tuple[ManualSample, float, float] | None:
  gap_excess = lead_crawl_gap_excess(sample)
  v_lead = _lead_speed(sample)
  if gap_excess is None or v_lead is None:
    return None
  if sample.v_ego > LEAD_CRAWL_MAX_SPEED or v_lead > LEAD_CRAWL_MAX_SPEED:
    return None
  return sample, gap_excess, v_lead


def _summarize_lead_crawl_bins(samples: list[ManualSample]) -> list[LeadCrawlBucketSummary]:
  summaries: list[LeadCrawlBucketSummary] = []
  details = _lead_crawl_sample_details(samples)
  for label, low, high in _LEAD_CRAWL_BUCKETS:
    bucket = [(sample, gap_excess, v_lead) for sample, gap_excess, v_lead in details if low <= gap_excess < high]
    if not bucket:
      continue
    closing = [(sample, gap_excess, v_lead) for sample, gap_excess, v_lead in bucket if v_lead - sample.v_ego < LEAD_CRAWL_CLOSING_THRESHOLD]
    summaries.append(LeadCrawlBucketSummary(
      label=label,
      sample_count=len(bucket),
      gas_ratio=_ratio(sum(1 for sample, _, _ in bucket if sample.gas_pressed), len(bucket)),
      brake_ratio=_ratio(sum(1 for sample, _, _ in bucket if sample.brake_pressed), len(bucket)),
      coast_ratio=_ratio(sum(1 for sample, _, _ in bucket if not sample.gas_pressed and not sample.brake_pressed), len(bucket)),
      gap_excess=percentile_range([gap_excess for _, gap_excess, _ in bucket], 10.0, 90.0),
      ego_speed=percentile_range([sample.v_ego for sample, _, _ in bucket], 10.0, 90.0),
      lead_speed=percentile_range([v_lead for _, _, v_lead in bucket], 10.0, 90.0),
      relative_speed=percentile_range([v_lead - sample.v_ego for sample, _, v_lead in bucket], 10.0, 90.0),
      accel=percentile_range([sample.a_ego for sample, _, _ in bucket], 10.0, 90.0),
      closing_ratio=_ratio(len(closing), len(bucket)),
      closing_speed=percentile_range([sample.v_ego - v_lead for sample, _, v_lead in closing], 10.0, 90.0),
    ))
  return summaries


def _summarize_lead_crawl_episodes(samples: list[ManualSample]) -> list[LeadCrawlEpisodeSummary]:
  ordered = sorted(samples, key=lambda sample: (sample.route, sample.t))
  crawl_episodes = _extract_gap_closure_episodes(ordered, "crawl_to_follow", start_min=2.0, end_max=1.0)
  soft_stop_episodes = _extract_gap_closure_episodes(ordered, "soft_stop", start_min=1.0, end_max=0.05, start_max=1.0)
  summaries = []
  for label, episodes in (("crawl_to_follow", crawl_episodes), ("soft_stop", soft_stop_episodes)):
    if not episodes:
      continue
    summaries.append(LeadCrawlEpisodeSummary(
      label=label,
      count=len(episodes),
      duration=percentile_range([episode["duration"] for episode in episodes], 50.0, 90.0),
      start_gap_excess=percentile_range([episode["start_gap_excess"] for episode in episodes], 50.0, 90.0),
      end_gap_excess=percentile_range([episode["end_gap_excess"] for episode in episodes], 50.0, 90.0),
      min_gap_excess=percentile_range([episode["min_gap_excess"] for episode in episodes], 10.0, 50.0),
      mean_accel=percentile_range([episode["mean_accel"] for episode in episodes], 50.0, 90.0),
    ))
  return summaries


def _extract_gap_closure_episodes(samples: list[ManualSample], label: str, start_min: float,
                                  end_max: float, start_max: float | None = None) -> list[dict[str, float]]:
  episodes: list[dict[str, float]] = []
  current: list[tuple[ManualSample, float, float]] = []
  current_route: str | None = None
  for sample in samples:
    route_changed = current and sample.route != current_route
    detail = _lead_crawl_sample_detail(sample)
    if detail is None or route_changed:
      current = []
      current_route = None
      if detail is None:
        continue
    sample, gap_excess, _ = detail
    can_start = gap_excess >= start_min if start_max is None else start_min >= gap_excess >= end_max
    if not current and can_start:
      current = [detail]
      current_route = sample.route
      continue
    if not current:
      continue
    current.append(detail)
    if gap_excess <= end_max:
      episodes.append(_gap_closure_episode_summary(current))
      current = []
      current_route = None
  return episodes


def _gap_closure_episode_summary(details: list[tuple[ManualSample, float, float]]) -> dict[str, float]:
  samples = [sample for sample, _, _ in details]
  gaps = [gap_excess for _, gap_excess, _ in details]
  return {
    "duration": max(0.0, samples[-1].t - samples[0].t),
    "start_gap_excess": gaps[0],
    "end_gap_excess": gaps[-1],
    "min_gap_excess": min(gaps),
    "mean_accel": sum(sample.a_ego for sample in samples) / len(samples),
  }


def _ratio(count: int, total: int) -> float:
  return count / total if total else 0.0


def _render_route_profile(profile: RouteProfile) -> str:
  status = "include" if profile.include else "exclude"
  return (
    f"  {profile.route}: {status}, samples {profile.samples}, moving {profile.manual_moving_samples}, "
    + f"active {profile.active_ratio:.3f}, duration {profile.duration_s:.1f}s"
  )


def _render_speed_bin(speed_bin: SpeedBinSummary) -> str:
  return (
    f"  {speed_bin.label}: samples {speed_bin.sample_count}, gas {speed_bin.gas_ratio:.1%}, "
    + f"brake {speed_bin.brake_ratio:.1%}, coast {speed_bin.coast_ratio:.1%}, "
    + f"accel {speed_bin.accel.low:.3f} to {speed_bin.accel.high:.3f} m/s^2, "
    + f"coast accel {speed_bin.coast_accel.low:.3f} to {speed_bin.coast_accel.high:.3f} m/s^2"
  )


def _render_following_bin(following_bin: FollowingBinSummary) -> str:
  return (
    f"  {following_bin.label}: samples {following_bin.sample_count}, closing {following_bin.closing_ratio:.1%}, "
    + f"distance {following_bin.distance.low:.1f} to {following_bin.distance.high:.1f} m, "
    + f"time gap {following_bin.time_gap.low:.1f} to {following_bin.time_gap.high:.1f}s, "
    + f"closing time {following_bin.closing_time.low:.1f} to {following_bin.closing_time.high:.1f}s"
  )


def _pedal_episodes(samples: list[ManualSample], pedal: str) -> list[dict[str, float | bool]]:
  episodes: list[dict[str, float | bool]] = []
  current: list[ManualSample] = []
  current_route: str | None = None
  trim_stopped_accel = pedal == "brake_pressed"
  for sample in samples:
    if current and sample.route != current_route:
      episodes.append(_episode_summary(current, trim_stopped_accel=trim_stopped_accel))
      current = []
    pressed = bool(getattr(sample, pedal))
    if pressed:
      current.append(sample)
      current_route = sample.route
      continue
    if current:
      episodes.append(_episode_summary(current, end_sample=sample, trim_stopped_accel=trim_stopped_accel))
      current = []
  if current:
    episodes.append(_episode_summary(current, trim_stopped_accel=trim_stopped_accel))
  return episodes


def _episode_summary(samples: list[ManualSample], end_sample: ManualSample | None = None,
                     trim_stopped_accel: bool = False) -> dict[str, float | bool]:
  accel_samples = [sample for sample in samples if not trim_stopped_accel or sample.v_ego > 1.0] or samples
  accels = [sample.a_ego for sample in accel_samples]
  end_sample = end_sample or samples[-1]
  return {
    "v0": samples[0].v_ego,
    "v1": end_sample.v_ego,
    "duration": max(0.0, end_sample.t - samples[0].t),
    "lead": bool(samples[0].lead_status),
    "mean_accel": sum(accels) / len(accels),
    "peak_accel": max(accels),
    "peak_decel": min(accels),
  }


def clean_finite(values: Iterable[float]) -> list[float]:
  clean = []
  for value in values:
    try:
      numeric_value = float(value)
    except (TypeError, ValueError):
      continue
    if isfinite(numeric_value):
      clean.append(numeric_value)
  return clean


def percentile_range(values: Iterable[float], low_pct: float, high_pct: float) -> ProfileRange:
  clean = clean_finite(values)
  if not clean:
    return ProfileRange(0.0, 0.0)
  return ProfileRange(float(np.percentile(clean, low_pct)), float(np.percentile(clean, high_pct)))


def classify_style(accel: ProfileRange, launch_mean: ProfileRange, stop_mean: ProfileRange,
                   coast_accel: ProfileRange, envelope: SmoothAssertiveEnvelope | None = None) -> str:
  envelope = envelope or SmoothAssertiveEnvelope()
  if not envelope.accel_p10.contains(accel.low):
    return "unknown"
  if not envelope.accel_p90.contains(accel.high):
    return "unknown"
  if not envelope.launch_mean_p50.contains(launch_mean.low):
    return "unknown"
  if not envelope.launch_mean_p90.contains(launch_mean.high):
    return "unknown"
  if not envelope.stop_mean_p10.contains(stop_mean.low):
    return "unknown"
  if not envelope.stop_mean_p50.contains(stop_mean.high):
    return "unknown"
  if not envelope.coast_p50.contains(coast_accel.low):
    return "unknown"
  if not envelope.coast_p90.contains(coast_accel.high):
    return "unknown"
  return "smooth_assertive"
