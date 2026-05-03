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


def clean_finite(values: Iterable[float]) -> list[float]:
  return [float(value) for value in values if isinstance(value, int | float) and isfinite(float(value))]


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
