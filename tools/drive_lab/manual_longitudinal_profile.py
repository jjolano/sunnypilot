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
  manual_moving_samples: int
  active_ratio: float
  include: bool


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
  style: str


def manual_moving_samples(samples: Iterable[ManualSample]) -> list[ManualSample]:
  return [sample for sample in samples if not sample.active and sample.v_ego > 1.0]


def build_route_profile(route: str, samples: Iterable[ManualSample], min_manual_moving_samples: int = 1200,
                        max_active_ratio: float = 0.25) -> RouteProfile:
  sample_list = list(samples)
  active_count = sum(1 for sample in sample_list if sample.active)
  active_ratio = active_count / len(sample_list) if sample_list else 1.0
  moving_count = len(manual_moving_samples(sample_list))
  include = moving_count >= min_manual_moving_samples and active_ratio <= max_active_ratio
  return RouteProfile(route, len(sample_list), moving_count, active_ratio, include)


def summarize_manual_style(samples: Iterable[ManualSample]) -> ManualStyleSummary:
  ordered = sorted((sample for sample in samples if not sample.active), key=lambda sample: (sample.route, sample.t))
  launches = _pedal_episodes(ordered, pedal="gas_pressed")
  stops = _pedal_episodes(ordered, pedal="brake_pressed")
  launch_candidates = [episode for episode in launches if episode["v0"] < 1.5 and episode["v1"] > 5.0 and episode["duration"] > 1.0]
  stop_candidates = [episode for episode in stops if episode["v0"] > 5.0 and episode["v1"] < 1.0 and episode["duration"] > 1.0]
  lead_launches = [episode for episode in launch_candidates if episode["lead"]]
  clear_launches = [episode for episode in launch_candidates if not episode["lead"]]
  lead_stops = [episode for episode in stop_candidates if episode["lead"]]
  clear_stops = [episode for episode in stop_candidates if not episode["lead"]]
  coast_samples = [sample for sample in ordered if not sample.gas_pressed and not sample.brake_pressed and sample.v_ego >= 7.0]

  accel = percentile_range([sample.a_ego for sample in ordered], 10.0, 90.0)
  lead_launch_mean = percentile_range([episode["mean_accel"] for episode in lead_launches], 50.0, 90.0)
  clear_launch_mean = percentile_range([episode["mean_accel"] for episode in clear_launches], 50.0, 90.0)
  stop_mean = percentile_range([episode["mean_accel"] for episode in stop_candidates], 10.0, 50.0)
  coast = percentile_range([sample.a_ego for sample in coast_samples], 50.0, 90.0)
  style = classify_style(accel, lead_launch_mean, stop_mean, coast)

  return ManualStyleSummary(
    sample_count=len(ordered),
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
    style=style,
  )


def render_manual_style_summary(summary: ManualStyleSummary) -> str:
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
    f"stop approaches: {summary.lead_stop_count + summary.clear_stop_count} mean {summary.stop_mean_accel.low:.3f} to "
    + f"{summary.stop_mean_accel.high:.3f} m/s^2 peak {summary.stop_peak_decel.low:.3f} to "
    + f"{summary.stop_peak_decel.high:.3f} m/s^2"
  )
  return "\n".join([
    "Manual longitudinal style",
    f"style: {summary.style}",
    f"samples: {summary.sample_count}",
    f"accel p10-p90: {summary.accel.low:.3f} to {summary.accel.high:.3f} m/s^2",
    lead_launch_line,
    clear_launch_line,
    stop_approach_line,
    f"coast accel: {summary.coast_accel.low:.3f} to {summary.coast_accel.high:.3f} m/s^2",
  ])


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
