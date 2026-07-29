#!/usr/bin/env python3
"""Measurement-only manual-versus-engaged steady-cruise smoothness profiler.

The profiler keeps manual and engaged samples separate and reports acceleration and
speed ripple without applying a pass/fail decision.  Its steady-cruise assumptions are
deliberately exposed in :class:`CruiseSmoothnessParams` and in the report notes.

Signals used when present:

* ``carState.{vEgo,aEgo,vCruise,gasPressed,brakePressed,standstill}``
* ``carControl.{longActive,actuators.accel}``
* ``carOutput.actuatorsOutput.accel``
* ``longitudinalPlan.aTarget`` and ``longitudinalPlanSP.aTarget``
* ``radarState.leadOne.{present,dRel}``

Run with::

  uv run python -m openpilot.tools.drive_lab.profile_cruise_smoothness ROUTE
  uv run python -m openpilot.tools.drive_lab.profile_cruise_smoothness ROUTE --json --output /tmp/x.json
"""

from __future__ import annotations

import argparse
import json
import math
from bisect import bisect_left
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from openpilot.tools.drive_lab.route_analysis import build_route_messages
from openpilot.tools.drive_lab.route_io import load_route_msgs, output_report
from openpilot.tools.drive_lab.timeline import safe_get


ACCEL_CHANNELS = (
  "a_ego",
  "longitudinal_plan_a_target",
  "longitudinal_plan_sp_a_target",
  "car_control_accel",
  "car_output_accel",
)
ENGAGED_ACCEL_CHANNELS = ACCEL_CHANNELS[1:]
ACCELERATION_SOURCE_PATHS = {
  "longitudinal_plan_a_target": ("longitudinalPlan", "aTarget"),
  "longitudinal_plan_sp_a_target": ("longitudinalPlanSP", "aTarget"),
  "car_control_accel": ("carControl", "actuators.accel"),
  "car_output_accel": ("carOutput", "actuatorsOutput.accel"),
}


@dataclass(frozen=True)
class CruiseSmoothnessParams:
  min_speed_mps: float = 8.0
  window_s: float = 10.0
  step_s: float = 5.0
  min_window_s: float = 5.0
  max_sample_gap_s: float = 0.5
  max_mode_age_s: float = 0.5
  max_radar_age_s: float = 0.5
  max_channel_age_s: float = 0.25
  transition_exclusion_s: float = 1.0
  close_lead_d_rel_m: float = 25.0
  set_speed_change_kph: float = 1.0
  max_speed_stddev_mps: float = 0.5
  max_speed_peak_to_peak_mps: float = 2.0
  accel_deadband_mps2: float = 0.05


@dataclass(frozen=True)
class AccelerationMetrics:
  sample_count: int
  duration_s: float
  acceleration_stddev_mps2: float
  acceleration_peak_to_peak_mps2: float
  jerk_p50_mps3: float | None
  jerk_p90_mps3: float | None
  jerk_p99_mps3: float | None
  acceleration_sign_reversals_per_min: float | None
  acceleration_deadband_share: float


@dataclass(frozen=True)
class ChannelCoverage:
  fresh_sample_count: int
  distinct_source_sample_count: int
  coverage_percent: float
  source_duration_s: float


@dataclass(frozen=True)
class CruiseSmoothnessWindow:
  start_s: float
  end_s: float
  duration_s: float
  sample_count: int
  speed_stddev_mps: float
  speed_peak_to_peak_mps: float
  acceleration: dict[str, AccelerationMetrics]
  channel_coverage: dict[str, ChannelCoverage]


@dataclass(frozen=True)
class CruiseSmoothnessView:
  steady_sample_count: int
  steady_duration_s: float
  window_count: int
  windows: list[CruiseSmoothnessWindow]


@dataclass(frozen=True)
class CruiseSmoothnessReport:
  source: str
  sample_count: int
  duration_s: float
  manual: CruiseSmoothnessView
  engaged: CruiseSmoothnessView
  available_acceleration_channels: list[str]
  exclusion_counts: dict[str, int]
  params: dict[str, Any]
  notes: list[str]

  def to_dict(self) -> dict[str, Any]:
    return asdict(self)

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> CruiseSmoothnessReport:
    def metrics(item: dict[str, Any]) -> AccelerationMetrics:
      return AccelerationMetrics(
        sample_count=int(item.get("sample_count", item.get("sampleCount", 0))),
        duration_s=float(item.get("duration_s", item.get("durationS", 0.0))),
        acceleration_stddev_mps2=float(item.get("acceleration_stddev_mps2", item.get("accelerationStddevMps2", 0.0))),
        acceleration_peak_to_peak_mps2=float(item.get("acceleration_peak_to_peak_mps2", item.get("accelerationPeakToPeakMps2", 0.0))),
        jerk_p50_mps3=_optional_float(item.get("jerk_p50_mps3", item.get("jerkP50Mps3"))),
        jerk_p90_mps3=_optional_float(item.get("jerk_p90_mps3", item.get("jerkP90Mps3"))),
        jerk_p99_mps3=_optional_float(item.get("jerk_p99_mps3", item.get("jerkP99Mps3"))),
        acceleration_sign_reversals_per_min=_optional_float(item.get("acceleration_sign_reversals_per_min", item.get("accelerationSignReversalsPerMin"))),
        acceleration_deadband_share=float(item.get("acceleration_deadband_share", item.get("accelerationDeadbandShare", 0.0))),
      )

    def window(item: dict[str, Any]) -> CruiseSmoothnessWindow:
      coverage = {
        str(k): ChannelCoverage(
          fresh_sample_count=int(v.get("fresh_sample_count", v.get("freshSampleCount", 0))),
          distinct_source_sample_count=int(v.get("distinct_source_sample_count", v.get("distinctSourceSampleCount", 0))),
          coverage_percent=float(v.get("coverage_percent", v.get("coveragePercent", 0.0))),
          source_duration_s=float(v.get("source_duration_s", v.get("sourceDurationS", 0.0))),
        )
        for k, v in item.get("channel_coverage", item.get("channelCoverage", {})).items()
      }
      return CruiseSmoothnessWindow(
        start_s=float(item.get("start_s", item.get("startS", 0.0))),
        end_s=float(item.get("end_s", item.get("endS", 0.0))),
        duration_s=float(item.get("duration_s", item.get("durationS", 0.0))),
        sample_count=int(item.get("sample_count", item.get("sampleCount", 0))),
        speed_stddev_mps=float(item.get("speed_stddev_mps", item.get("speedStddevMps", 0.0))),
        speed_peak_to_peak_mps=float(item.get("speed_peak_to_peak_mps", item.get("speedPeakToPeakMps", 0.0))),
        acceleration={str(k): metrics(v) for k, v in item.get("acceleration", {}).items()},
        channel_coverage=coverage,
      )

    def view(item: dict[str, Any]) -> CruiseSmoothnessView:
      windows = [window(entry) for entry in item.get("windows", [])]
      return CruiseSmoothnessView(
        steady_sample_count=int(item.get("steady_sample_count", item.get("steadySampleCount", 0))),
        steady_duration_s=float(item.get("steady_duration_s", item.get("steadyDurationS", 0.0))),
        window_count=int(item.get("window_count", item.get("windowCount", len(windows)))),
        windows=windows,
      )

    return cls(
      source=str(data.get("source", "unknown")),
      sample_count=int(data.get("sample_count", data.get("sampleCount", 0))),
      duration_s=float(data.get("duration_s", data.get("durationS", 0.0))),
      manual=view(data.get("manual", {})),
      engaged=view(data.get("engaged", {})),
      available_acceleration_channels=[str(value) for value in data.get("available_acceleration_channels", data.get("availableAccelerationChannels", []))],
      exclusion_counts={str(k): int(v) for k, v in data.get("exclusion_counts", data.get("exclusionCounts", {})).items()},
      params=dict(data.get("params", {})),
      notes=[str(note) for note in data.get("notes", [])],
    )


@dataclass(frozen=True)
class _CruiseSample:
  index: int
  t: float
  v_ego: float | None
  a_ego: float | None
  v_cruise_kph: float | None
  gas_pressed: bool
  brake_pressed: bool
  standstill: bool
  close_lead: bool | None
  engaged: bool | None
  acceleration: dict[str, float | None]
  acceleration_source_times: dict[str, float | None]
  stale_channels: tuple[str, ...]


def build_cruise_smoothness_report(
  msgs: list[Any],
  source: str = "unknown",
  *,
  params: CruiseSmoothnessParams | None = None,
  already_sorted: bool = False,
) -> CruiseSmoothnessReport:
  p = params or CruiseSmoothnessParams()
  _validate_params(p)
  ordered_msgs = list(msgs) if already_sorted else sorted(msgs, key=lambda m: int(getattr(m, "logMonoTime", 0)))
  samples = _extract_samples(ordered_msgs, p)
  duration_s = float(samples[-1].t - samples[0].t) if len(samples) > 1 else 0.0
  notes = _notes(p, samples)
  if not samples:
    notes.append("no carState samples found")
    empty = CruiseSmoothnessView(0, 0.0, 0, [])
    return CruiseSmoothnessReport(source, len(ordered_msgs), 0.0, empty, empty, [], {}, asdict(p), notes)

  transition_events = _transition_events(samples, p)
  reasons = [_sample_reasons(sample, index, samples, transition_events, p) for index, sample in enumerate(samples)]
  for index in range(1, len(samples)):
    if samples[index].t - samples[index - 1].t > p.max_sample_gap_s:
      reasons[index].add("time_gap")
      reasons[index - 1].add("time_gap")
  exclusion_counts: dict[str, int] = {}
  for sample_reasons in reasons:
    for reason in sample_reasons:
      exclusion_counts[reason] = exclusion_counts.get(reason, 0) + 1
  eligible = [not sample_reasons for sample_reasons in reasons]

  manual, manual_unsteady = _build_view(samples, eligible, False, p)
  engaged, engaged_unsteady = _build_view(samples, eligible, True, p)
  exclusion_counts["unstable_speed_window"] = manual_unsteady + engaged_unsteady
  if exclusion_counts["unstable_speed_window"] == 0:
    exclusion_counts.pop("unstable_speed_window")

  available_channels = [
    channel for channel in ACCEL_CHANNELS
    if any(sample.acceleration.get(channel) is not None for sample in samples)
  ]
  stale_channel_counts: dict[str, int] = {}
  for sample in samples:
    for channel in sample.stale_channels:
      stale_channel_counts[channel] = stale_channel_counts.get(channel, 0) + 1
  notes.extend(_channel_notes(available_channels, stale_channel_counts, p.max_channel_age_s))
  return CruiseSmoothnessReport(
    source=source,
    sample_count=len(ordered_msgs),
    duration_s=duration_s,
    manual=manual,
    engaged=engaged,
    available_acceleration_channels=available_channels,
    exclusion_counts=exclusion_counts,
    params=asdict(p),
    notes=notes,
  )


def analyze_route(
  msgs: list[Any],
  source: str = "unknown",
  *,
  params: CruiseSmoothnessParams | None = None,
  already_sorted: bool = False,
) -> CruiseSmoothnessReport:
  """Analyze already-loaded route messages for route harnesses and tests."""
  return build_cruise_smoothness_report(msgs, source, params=params, already_sorted=already_sorted)


def render_report(report: CruiseSmoothnessReport) -> str:
  lines = [
    f"Cruise smoothness measurement: {report.source}",
    f"samples: {report.sample_count}",
    f"duration: {report.duration_s:.1f} s",
    "measurement-only: no pass/fail threshold applied",
    f"available acceleration channels: {', '.join(report.available_acceleration_channels) or 'none'}",
  ]
  for name, view in (("manual", report.manual), ("engaged", report.engaged)):
    lines.append(f"{name}: steady_samples={view.steady_sample_count} steady_duration={view.steady_duration_s:.1f}s windows={view.window_count}")
    for window in view.windows:
      lines.append(
        f"  {window.start_s:.1f}-{window.end_s:.1f}s samples={window.sample_count} "
        + f"speed_std={window.speed_stddev_mps:.4f} speed_pp={window.speed_peak_to_peak_mps:.4f}"
      )
      for channel, metric in sorted(window.acceleration.items()):
        lines.append(
          f"    {channel}: accel_std={metric.acceleration_stddev_mps2:.4f} "
          + f"accel_pp={metric.acceleration_peak_to_peak_mps2:.4f} "
          + f"jerk_p50/p90/p99={_fmt(metric.jerk_p50_mps3)}/{_fmt(metric.jerk_p90_mps3)}/{_fmt(metric.jerk_p99_mps3)} "
          + f"reversals_per_min={_fmt(metric.acceleration_sign_reversals_per_min)} "
          + f"deadband_share={metric.acceleration_deadband_share:.3f}"
        )
      for channel, coverage in sorted(window.channel_coverage.items()):
        lines.append(
          f"    coverage {channel}: fresh={coverage.fresh_sample_count} "
          + f"distinct_source={coverage.distinct_source_sample_count} "
          + f"share={coverage.coverage_percent:.1f}% source_duration={coverage.source_duration_s:.3f}s"
        )
  if report.exclusion_counts:
    lines.append(f"excluded samples/windows: {report.exclusion_counts}")
  if report.notes:
    lines.append("notes:")
    lines.extend(f"  {note}" for note in report.notes)
  return "\n".join(lines)


def save_report(report: CruiseSmoothnessReport, path: str | Path) -> None:
  Path(path).write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")


def load_report(path: str | Path) -> CruiseSmoothnessReport:
  return CruiseSmoothnessReport.from_dict(json.loads(Path(path).read_text()))


def _extract_samples(msgs: list[Any], p: CruiseSmoothnessParams) -> list[_CruiseSample]:
  records = build_route_messages(msgs)
  latest: dict[str, Any] = {}
  latest_times: dict[str, float] = {}
  samples: list[_CruiseSample] = []
  for record in records:
    latest[record.typ] = record.payload
    latest_times[record.typ] = record.t
    if record.typ != "carState":
      continue
    car_state = record.payload
    engaged = _engagement_state(latest, latest_times, record.t, p.max_mode_age_s)
    acceleration: dict[str, float | None] = {"a_ego": _finite_or_none(safe_get(car_state, "aEgo"))}
    acceleration_source_times: dict[str, float | None] = {"a_ego": record.t}
    stale_channels: list[str] = []
    if engaged is True:
      for channel, (source_type, path) in ACCELERATION_SOURCE_PATHS.items():
        source_time = latest_times.get(source_type)
        acceleration_source_times[channel] = source_time
        if source_time is None:
          acceleration[channel] = None
          continue
        if record.t - source_time > p.max_channel_age_s:
          acceleration[channel] = None
          stale_channels.append(channel)
          continue
        acceleration[channel] = _finite_or_none(safe_get(latest.get(source_type), path))
    samples.append(_CruiseSample(
      index=len(samples),
      t=record.t,
      v_ego=_finite_or_none(safe_get(car_state, "vEgo")),
      a_ego=acceleration["a_ego"],
      v_cruise_kph=_finite_or_none(safe_get(car_state, "vCruise")),
      gas_pressed=bool(safe_get(car_state, "gasPressed", False)),
      brake_pressed=bool(safe_get(car_state, "brakePressed", False)),
      standstill=bool(safe_get(car_state, "standstill", False)),
      close_lead=_close_lead(
        latest.get("radarState"), latest_times.get("radarState"), record.t,
        p.close_lead_d_rel_m, p.max_radar_age_s,
      ),
      engaged=engaged,
      acceleration=acceleration,
      acceleration_source_times=acceleration_source_times,
      stale_channels=tuple(stale_channels),
    ))
  return samples


def _engagement_state(latest: dict[str, Any], latest_times: dict[str, float], t_s: float, max_age_s: float) -> bool | None:
  car_control = latest.get("carControl")
  long_active = _fresh_bool(car_control, latest_times.get("carControl"), "longActive", t_s, max_age_s)
  selfdrive = latest.get("selfdriveState")
  system_enabled = _fresh_bool(selfdrive, latest_times.get("selfdriveState"), "enabled", t_s, max_age_s)
  if system_enabled is None or long_active is None:
    return None
  if not system_enabled and not long_active:
    return False
  if system_enabled and long_active:
    return True
  return None


def _fresh_bool(payload: Any, source_time: float | None, path: str, t_s: float, max_age_s: float) -> bool | None:
  if payload is None or source_time is None or t_s - source_time > max_age_s:
    return None
  value = safe_get(payload, path)
  return None if value is None else bool(value)


def _close_lead(
  radar_state: Any,
  source_time: float | None,
  t_s: float,
  max_d_rel_m: float,
  max_age_s: float,
) -> bool | None:
  if radar_state is None or source_time is None or t_s - source_time > max_age_s:
    return None
  lead = safe_get(radar_state, "leadOne")
  if lead is None:
    return None
  if not bool(safe_get(lead, "present", False)):
    return False
  d_rel = _finite_or_none(safe_get(lead, "dRel"))
  return None if d_rel is None else d_rel <= max_d_rel_m


def _transition_events(samples: list[_CruiseSample], p: CruiseSmoothnessParams) -> dict[str, list[float]]:
  events: dict[str, list[float]] = {}
  last_known_engaged: bool | None = samples[0].engaged if samples else None
  for previous, current in zip(samples, samples[1:], strict=False):
    if current.t - previous.t > p.max_sample_gap_s:
      last_known_engaged = current.engaged
      continue
    if previous.gas_pressed != current.gas_pressed or previous.brake_pressed != current.brake_pressed:
      _add_event(events, "pedal_transition", current.t)
    if current.engaged is not None and last_known_engaged is not None and last_known_engaged != current.engaged:
      _add_event(events, "engagement_transition", current.t)
    if current.engaged is not None:
      last_known_engaged = current.engaged
    if previous.v_ego is not None and current.v_ego is not None and (
      (previous.v_ego < p.min_speed_mps) != (current.v_ego < p.min_speed_mps)
    ):
      _add_event(events, "speed_transition", current.t)
    if previous.v_cruise_kph is not None and current.v_cruise_kph is not None and (
      abs(current.v_cruise_kph - previous.v_cruise_kph) >= p.set_speed_change_kph
    ):
      _add_event(events, "set_speed_change", current.t)
  return events


def _add_event(events: dict[str, list[float]], reason: str, time_s: float) -> None:
  events.setdefault(reason, []).append(time_s)


def _sample_reasons(
  sample: _CruiseSample,
  index: int,
  samples: list[_CruiseSample],
  events: dict[str, list[float]],
  p: CruiseSmoothnessParams,
) -> set[str]:
  reasons: set[str] = set()
  if sample.v_ego is None:
    reasons.add("missing_v_ego")
  elif sample.v_ego < p.min_speed_mps or sample.standstill:
    reasons.add("stopping_or_launching")
  if sample.a_ego is None:
    reasons.add("missing_a_ego")
  if sample.brake_pressed or (sample.gas_pressed and sample.engaged is not False):
    reasons.add("pedal_active")
  if sample.close_lead is None:
    reasons.add("radar_unknown")
  elif sample.close_lead:
    reasons.add("close_lead_following")
  if sample.engaged is None:
    reasons.add("mode_unknown")
  for reason, times in events.items():
    if _near_event(times, sample.t, p.transition_exclusion_s):
      reasons.add(reason)
  if index > 0 and sample.t - samples[index - 1].t > p.max_sample_gap_s:
    reasons.add("time_gap")
  return reasons


def _near_event(times: list[float], time_s: float, exclusion_s: float) -> bool:
  if not times:
    return False
  index = bisect_left(times, time_s)
  return (
    index < len(times) and abs(times[index] - time_s) <= exclusion_s
  ) or (
    index > 0 and abs(times[index - 1] - time_s) <= exclusion_s
  )


def _build_view(
  samples: list[_CruiseSample],
  eligible: list[bool],
  engaged: bool,
  p: CruiseSmoothnessParams,
) -> tuple[CruiseSmoothnessView, int]:
  view_samples = [sample for sample, ok in zip(samples, eligible, strict=False) if ok and sample.engaged is engaged]
  segments: list[list[_CruiseSample]] = []
  current: list[_CruiseSample] = []
  for sample in view_samples:
    if current and (sample.index != current[-1].index + 1 or sample.t - current[-1].t > p.max_sample_gap_s):
      segments.append(current)
      current = []
    current.append(sample)
  if current:
    segments.append(current)

  accepted: list[tuple[CruiseSmoothnessWindow, list[_CruiseSample]]] = []
  unsteady_count = 0
  for segment in segments:
    for chunk in _window_chunks(segment, p):
      if not _steady_speed(chunk, p):
        unsteady_count += 1
        continue
      accepted.append((_window_metrics(chunk, engaged, p), chunk))
  accepted.sort(key=lambda item: (item[0].start_s, item[0].end_s))
  windows = [window for window, _ in accepted]
  accepted_sample_indices = {sample.index for _, chunk in accepted for sample in chunk}
  steady_duration_s = _interval_union_duration([(window.start_s, window.end_s) for window in windows])
  return CruiseSmoothnessView(len(accepted_sample_indices), steady_duration_s, len(windows), windows), unsteady_count


def _window_chunks(segment: list[_CruiseSample], p: CruiseSmoothnessParams) -> list[list[_CruiseSample]]:
  if len(segment) < 2 or segment[-1].t - segment[0].t < p.min_window_s:
    return []
  chunks: list[list[_CruiseSample]] = []
  start = segment[0].t
  final = segment[-1].t
  while start + p.min_window_s <= final + 1e-9:
    end = min(start + p.window_s, final)
    chunk = [sample for sample in segment if start - 1e-9 <= sample.t <= end + 1e-9]
    if len(chunk) >= 2 and chunk[-1].t - chunk[0].t >= p.min_window_s - 1e-9:
      chunks.append(chunk)
    start += p.step_s
  return chunks


def _interval_union_duration(intervals: list[tuple[float, float]]) -> float:
  if not intervals:
    return 0.0
  ordered = sorted(intervals)
  current_start, current_end = ordered[0]
  duration = 0.0
  for start, end in ordered[1:]:
    if start <= current_end:
      current_end = max(current_end, end)
    else:
      duration += current_end - current_start
      current_start, current_end = start, end
  return float(duration + current_end - current_start)


def _steady_speed(chunk: list[_CruiseSample], p: CruiseSmoothnessParams) -> bool:
  speed = np.asarray([sample.v_ego for sample in chunk if sample.v_ego is not None], dtype=float)
  return bool(
    speed.size >= 2
    and np.std(speed) <= p.max_speed_stddev_mps
    and np.ptp(speed) <= p.max_speed_peak_to_peak_mps
  )


def _window_metrics(chunk: list[_CruiseSample], engaged: bool, p: CruiseSmoothnessParams) -> CruiseSmoothnessWindow:
  speed = np.asarray([sample.v_ego for sample in chunk if sample.v_ego is not None], dtype=float)
  channels = ("a_ego",) + ENGAGED_ACCEL_CHANNELS if engaged else ("a_ego",)
  acceleration: dict[str, AccelerationMetrics] = {}
  channel_coverage: dict[str, ChannelCoverage] = {}
  for channel in channels:
    entries, fresh_sample_count = _channel_entries(chunk, channel)
    channel_coverage[channel] = _channel_coverage(entries, fresh_sample_count, len(chunk))
    if (metrics := _acceleration_metrics(entries, p)) is not None:
      acceleration[channel] = metrics
  return CruiseSmoothnessWindow(
    start_s=float(chunk[0].t),
    end_s=float(chunk[-1].t),
    duration_s=float(chunk[-1].t - chunk[0].t),
    sample_count=len(chunk),
    speed_stddev_mps=float(np.std(speed)),
    speed_peak_to_peak_mps=float(np.ptp(speed)),
    acceleration=acceleration,
    channel_coverage=channel_coverage,
  )


def _channel_entries(
  chunk: list[_CruiseSample], channel: str,
) -> tuple[list[tuple[float | None, float | None]], int]:
  entries: list[tuple[float | None, float | None]] = []
  fresh_sample_count = 0
  last_source_time: float | None = None
  for sample in chunk:
    source_time = sample.acceleration_source_times.get(channel)
    value = sample.acceleration.get(channel)
    if source_time is None or value is None:
      entries.append((None, None))
      last_source_time = None
      continue
    fresh_sample_count += 1
    if source_time == last_source_time:
      continue
    entries.append((source_time, value))
    last_source_time = source_time
  return entries, fresh_sample_count


def _entry_segments(entries: list[tuple[float | None, float | None]]) -> list[list[tuple[float, float]]]:
  segments: list[list[tuple[float, float]]] = []
  current: list[tuple[float, float]] = []
  for time_s, value in entries:
    if time_s is None or value is None:
      if current:
        segments.append(current)
        current = []
      continue
    current.append((time_s, value))
  if current:
    segments.append(current)
  return segments


def _channel_coverage(
  entries: list[tuple[float | None, float | None]], fresh_sample_count: int, window_sample_count: int,
) -> ChannelCoverage:
  segments = _entry_segments(entries)
  source_duration_s = sum(segment[-1][0] - segment[0][0] for segment in segments if len(segment) > 1)
  distinct_source_sample_count = sum(len(segment) for segment in segments)
  coverage_percent = 100.0 * fresh_sample_count / window_sample_count if window_sample_count else 0.0
  return ChannelCoverage(fresh_sample_count, distinct_source_sample_count, float(coverage_percent), float(source_duration_s))


def _acceleration_metrics(
  entries: list[tuple[float | None, float | None]],
  p: CruiseSmoothnessParams,
) -> AccelerationMetrics | None:
  segments = _entry_segments(entries)
  distinct_sample_count = sum(len(segment) for segment in segments)
  if distinct_sample_count < 2:
    return None
  values = np.asarray([value for segment in segments for _, value in segment], dtype=float)
  jerk_parts: list[np.ndarray] = []
  reversals = 0
  duration_s = 0.0
  for segment in segments:
    times = np.asarray([time_s for time_s, _ in segment], dtype=float)
    segment_values = np.asarray([value for _, value in segment], dtype=float)
    duration_s += float(times[-1] - times[0]) if len(times) > 1 else 0.0
    dt = np.diff(times)
    dv = np.diff(segment_values)
    valid_jerk = (dt > 1e-6) & (dt <= p.max_sample_gap_s) & np.isfinite(dv)
    jerk_parts.append(np.abs(dv[valid_jerk] / dt[valid_jerk]))
    non_deadband = np.abs(segment_values) > p.accel_deadband_mps2
    signs = np.sign(segment_values[non_deadband])
    if signs.size > 1:
      reversals += int(np.sum(signs[1:] != signs[:-1]))
  jerk = np.concatenate(jerk_parts) if jerk_parts else np.array([], dtype=float)
  non_deadband = np.abs(values) > p.accel_deadband_mps2
  reversals_per_min = 60.0 * reversals / duration_s if duration_s > 1e-6 else None
  return AccelerationMetrics(
    sample_count=distinct_sample_count,
    duration_s=duration_s,
    acceleration_stddev_mps2=float(np.std(values)),
    acceleration_peak_to_peak_mps2=float(np.ptp(values)),
    jerk_p50_mps3=_percentile_or_none(jerk, 50.0),
    jerk_p90_mps3=_percentile_or_none(jerk, 90.0),
    jerk_p99_mps3=_percentile_or_none(jerk, 99.0),
    acceleration_sign_reversals_per_min=reversals_per_min,
    acceleration_deadband_share=float(np.mean(~non_deadband)),
  )


def _validate_params(p: CruiseSmoothnessParams) -> None:
  if p.min_speed_mps < 0.0 or p.min_window_s <= 0.0 or p.window_s < p.min_window_s:
    raise ValueError("cruise smoothness speed/window parameters are invalid")
  if (
    p.step_s <= 0.0 or p.max_sample_gap_s <= 0.0 or p.max_mode_age_s < 0.0
    or p.max_radar_age_s < 0.0 or p.max_channel_age_s < 0.0 or p.transition_exclusion_s < 0.0
  ):
    raise ValueError("cruise smoothness timing parameters are invalid")
  if p.close_lead_d_rel_m < 0.0 or p.set_speed_change_kph < 0.0:
    raise ValueError("cruise smoothness transition parameters are invalid")
  if p.max_speed_stddev_mps < 0.0 or p.max_speed_peak_to_peak_mps < 0.0 or p.accel_deadband_mps2 < 0.0:
    raise ValueError("cruise smoothness metric parameters are invalid")
def _notes(p: CruiseSmoothnessParams, samples: list[_CruiseSample]) -> list[str]:
  return [
    "measurement-only: no pass/fail threshold is applied; use the metrics to build a corpus",
    "steady_sample_count and steady_duration_s count unique samples and interval-union duration from accepted windows; "
    + "speed ripple limits are applied at acceptance",
    f"steady windows require vEgo >= {p.min_speed_mps:.1f} m/s, speed stddev <= {p.max_speed_stddev_mps:.2f} m/s, and "
    + f"speed peak-to-peak <= {p.max_speed_peak_to_peak_mps:.2f} m/s; these are corpus assumptions, not universal limits",
    f"pedal, set-speed, engagement, and speed transitions are excluded for {p.transition_exclusion_s:.1f} s; "
    + f"close lead-following means leadOne.present with dRel <= {p.close_lead_d_rel_m:.1f} m",
    "manual gas-held samples remain eligible after transition exclusion; braking and engaged gas override samples remain excluded",
    f"discontinuous samples are segmented at gaps > {p.max_sample_gap_s:.2f} s; jerk uses finite adjacent samples within that gap",
    f"engaged acceleration channels are zero-order-held only while their source timestamp age is <= {p.max_channel_age_s:.2f} s",
    "jerk p50/p90/p99 are percentiles of absolute acceleration slope between adjacent samples",
    f"acceleration sign reversals ignore |accel| <= {p.accel_deadband_mps2:.3f} m/s^2",
    f"mode requires fresh selfdriveState.enabled and carControl.longActive within {p.max_mode_age_s:.2f} s; enabled with longActive false is ineligible",
    f"radarState.leadOne must be fresh within {p.max_radar_age_s:.2f} s; missing/stale radar is ineligible rather than no-lead manual evidence",
  ]


def _channel_notes(available: list[str], stale: dict[str, int], max_age_s: float) -> list[str]:
  notes: list[str] = []
  missing = [channel for channel in ACCEL_CHANNELS if channel not in available and channel not in stale]
  if missing:
    notes.append(f"missing acceleration channels are omitted from windows: {', '.join(missing)}")
  if stale:
    details = ", ".join(f"{channel} ({count} stale samples)" for channel, count in sorted(stale.items()))
    notes.append(f"stale acceleration channels are omitted after {max_age_s:.2f} s: {details}")
  return notes


def _finite_or_none(value: Any) -> float | None:
  try:
    numeric = float(value)
  except (TypeError, ValueError):
    return None
  return numeric if math.isfinite(numeric) else None


def _optional_float(value: Any) -> float | None:
  return _finite_or_none(value)


def _percentile_or_none(values: np.ndarray, percentile: float) -> float | None:
  return float(np.percentile(values, percentile)) if values.size else None


def _fmt(value: float | None) -> str:
  return "n/a" if value is None else f"{value:.4f}"


def main() -> None:
  parser = argparse.ArgumentParser(description="Measure manual and engaged steady-cruise acceleration flicker from route logs.")
  parser.add_argument("route", help="Route, segment range, log file, or URL accepted by LogReader")
  parser.add_argument("--output", help="Write report JSON to this path")
  parser.add_argument("--json", action="store_true", help="Print JSON instead of a text summary")
  parser.add_argument("--qlog", action="store_true", help="Prefer qlogs instead of rlogs")
  args = parser.parse_args()

  msgs = load_route_msgs(args.route, qlog=args.qlog)
  report = build_cruise_smoothness_report(msgs, source=args.route, already_sorted=True)
  print(output_report(report, json_output=args.json, renderer=render_report, output_path=args.output, save=save_report))


if __name__ == "__main__":
  main()
