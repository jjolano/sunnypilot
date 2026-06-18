from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, field
import math
from typing import Any, Sequence


@dataclass(frozen=True)
class LocalizationQualityThresholds:
  large_gap_s: float = 0.5
  high_trans_std_norm: float = 7.0
  camera_yaw_window_s: float = 0.2
  gps_heading_window_s: float = 0.5
  gps_speed_min: float = 0.5
  gps_bearing_accuracy_max_deg: float = 20.0


def finite_float(value: Any) -> float | None:
  try:
    out = float(value)
  except (TypeError, ValueError):
    return None
  return out if math.isfinite(out) else None


def vector_norm3(values: Any) -> float | None:
  if values is None:
    return None
  try:
    xs = list(values)
  except TypeError:
    return None
  if len(xs) < 3:
    return None
  nums = [finite_float(v) for v in xs[:3]]
  if any(v is None for v in nums):
    return None
  return math.sqrt(sum(v * v for v in nums if v is not None))


def percentile(values: Sequence[float], pct: float) -> float | None:
  finite = sorted(v for v in values if isinstance(v, (int, float)) and math.isfinite(float(v)))
  if not finite:
    return None
  idx = max(0, min(len(finite) - 1, int(math.ceil((pct / 100.0) * len(finite)) - 1)))
  return float(finite[idx])


@dataclass(frozen=True)
class FrequencySummary:
  samples: int
  observed_hz: float | None
  max_gap_s: float | None
  p95_gap_s: float | None
  large_gap_count: int

  def to_dict(self) -> dict[str, Any]:
    return {
      "samples": self.samples,
      "observed_hz": self.observed_hz,
      "max_gap_s": self.max_gap_s,
      "p95_gap_s": self.p95_gap_s,
      "large_gap_count": self.large_gap_count,
    }


def freshness_summary(times: list[float], *, thresholds: LocalizationQualityThresholds | None = None) -> FrequencySummary:
  p = thresholds or LocalizationQualityThresholds()
  if len(times) < 2:
    return FrequencySummary(len(times), None, None, None, 0)
  gaps = [b - a for a, b in zip(times, times[1:]) if math.isfinite(a) and math.isfinite(b) and b >= a]
  duration = times[-1] - times[0]
  hz = ((len(times) - 1) / duration) if duration > 0 else None
  return FrequencySummary(
    samples=len(times),
    observed_hz=hz,
    max_gap_s=max(gaps) if gaps else None,
    p95_gap_s=percentile(gaps, 95),
    large_gap_count=sum(1 for gap in gaps if gap > p.large_gap_s),
  )


@dataclass(frozen=True)
class StdSummary:
  p95: float | None
  max: float | None

  def to_dict(self) -> dict[str, Any]:
    return {"p95": self.p95, "max": self.max}


def std_summary(values: list[float]) -> StdSummary:
  finite = [v for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]
  return StdSummary(p95=percentile(finite, 95), max=max(finite) if finite else None)


def heading_error_deg(reference_deg: float, measured_deg: float) -> float:
  return ((measured_deg - reference_deg + 180.0) % 360.0) - 180.0


def nearest_value(series: list[tuple[float, Any]], t: float, window_s: float, *, times: list[float] | None = None) -> Any:
  if not series:
    return None
  times = times or [st for st, _ in series]
  idx = bisect_left(times, t)
  candidates = []
  if idx < len(series):
    candidates.append(series[idx])
  if idx > 0:
    candidates.append(series[idx - 1])
  best = None
  best_dt = window_s
  for st, value in candidates:
    dt = abs(st - t)
    if dt <= best_dt:
      best_dt = dt
      best = value
  return best


@dataclass(frozen=True)
class LocalizationQualityHealth:
  ok: bool
  degraded_reasons: tuple[str, ...] = field(default_factory=tuple)

  @classmethod
  def from_signals(
    cls,
    *,
    camera_fresh: FrequencySummary | None = None,
    live_fresh: FrequencySummary | None = None,
    high_trans_std_count: int | None = None,
    yaw_pair_count: int | None = None,
    gps_pair_count: int | None = None,
    gps_p95_abs_error_deg: float | None = None,
    thresholds: LocalizationQualityThresholds | None = None,
  ) -> "LocalizationQualityHealth":
    p = thresholds or LocalizationQualityThresholds()
    reasons: list[str] = []
    if camera_fresh is None or live_fresh is None:
      reasons.append("missing freshness evidence")
    if camera_fresh is not None and (camera_fresh.large_gap_count > 0 or (camera_fresh.max_gap_s or 0.0) > p.large_gap_s):
      reasons.append("cameraOdometry freshness degraded")
    if live_fresh is not None and (live_fresh.large_gap_count > 0 or (live_fresh.max_gap_s or 0.0) > p.large_gap_s):
      reasons.append("livePose freshness degraded")
    if high_trans_std_count is not None and high_trans_std_count > 0:
      reasons.append("cameraOdometry high translation std")
    if yaw_pair_count is not None and yaw_pair_count == 0:
      reasons.append("no yaw-rate consistency pairs")
    if gps_pair_count is not None and gps_pair_count == 0:
      reasons.append("no GPS heading pairs")
    if gps_p95_abs_error_deg is not None and gps_p95_abs_error_deg > 15.0:
      reasons.append("GPS heading mismatch")
    return cls(ok=not reasons, degraded_reasons=tuple(reasons))
