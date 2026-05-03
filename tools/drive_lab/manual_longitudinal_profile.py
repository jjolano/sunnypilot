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
  style: str


_SPEED_BINS = (
  ("1-3 m/s", 1.0, 3.0),
  ("3-7 m/s", 3.0, 7.0),
  ("7-13 m/s", 7.0, 13.0),
  ("13-20 m/s", 13.0, 20.0),
  ("20+ m/s", 20.0, float("inf")),
)


def manual_moving_samples(samples: Iterable[ManualSample]) -> list[ManualSample]:
  return [sample for sample in samples if not sample.active and sample.v_ego > 1.0]


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
  for sample in samples:
    if current and sample.route != current_route:
      episodes.append(_episode_summary(current))
      current = []
    pressed = bool(getattr(sample, pedal))
    if pressed:
      current.append(sample)
      current_route = sample.route
      continue
    if current:
      episodes.append(_episode_summary(current, end_sample=sample))
      current = []
  if current:
    episodes.append(_episode_summary(current))
  return episodes


def _episode_summary(samples: list[ManualSample], end_sample: ManualSample | None = None) -> dict[str, float | bool]:
  accels = [sample.a_ego for sample in samples]
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
