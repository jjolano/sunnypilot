#!/usr/bin/env python3
"""Profile how the car recovers (or fails to recover) path curvature after stopping in a corner.

Motivation
----------
The lateral stack is memoryless per-frame vision: each frame the driving model emits
``modelV2.action.desiredCurvature`` from the current camera, and at standstill the
lateral-demand pipeline's retained-curve state is wiped (``ModelPathProcessor.reset`` fires
because ``CC.latActive`` goes False at standstill in controlsd). So when the car stops
mid-corner it "forgets" the curve and launches under-steering until vision re-acquires it.

This analysis quantifies that amnesia with a *self-supervised, speed-independent* signature that
needs no map and no pose integration — important because the symptom mostly lives in the
low-speed regime (tight intersection turns), where the whole stop→launch happens below the speed
at which vision is reliable. For each stop we use three logged curvatures:

  Z       = the curve held just before the stop (commanded, while moving + engaged) — "the curve",
  X       = the curvature commanded right after launch — "did it forget?",
  X_vis   = the *raw vision* curvature at launch (modelV2.action) — isolates perception amnesia,
  cont    = the peak *actual* curvature the car follows after launch, in Z's direction — proof the
            curve actually continued (vs. an intersection turn that completes / road that straightens).

A stop is a *corner* when ``|Z| >= kappa_corner``. If the curve continued (``cont >= continuation_frac*|Z|``)
and the launch command under-shoots Z (``(|Z|-|X|)/|Z| >= amnesia_deficit_frac``), and it is not a
driver override or lane change, it is an *amnesia* event. The deficit (``|Z|-|X|``, and the purer
``|Z|-|X_vis|``) and the recovery lag (time / distance until the command climbs back to
``recovery_fraction*|Z|`` in Z's direction) size the problem and form the regression baseline the
future CurveMemory change must beat. If the curve did *not* continue, it is ``curve_ended`` — the car
correctly straightened, not amnesia.

Known limitation: a turn that fully fails and is never corrected leaves no continuation evidence, so
it reads as ``curve_ended`` rather than amnesia. In practice such turns get corrected (driver override,
flagged) or recovered late (captured as lag); fully-failed-and-uncorrected turns are rare when engaged.

Signals (all logged, see cereal/services.py): controlsState.desiredCurvature (commanded),
controlsState.curvature (actual VM path curvature), modelV2.action.desiredCurvature (raw vision),
carState.vEgo / standstill / steeringPressed, carControl.latActive (only score engaged driving),
modelV2.meta.laneChangeState.

Run:
  uv run python -m openpilot.tools.drive_lab.profile_corner_recovery ROUTE
  uv run python -m openpilot.tools.drive_lab.profile_corner_recovery ROUTE --json --output /tmp/x.json
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from openpilot.tools.drive_lab.route_analysis import build_route_messages
from openpilot.tools.drive_lab.route_io import load_route_msgs, output_report
from openpilot.tools.drive_lab.timeline import format_enum, safe_get


@dataclass(frozen=True)
class DetectorParams:
  v_stop: float = 0.3                # m/s; at/below this (or carState.standstill) the car is "stopped"
  v_moving: float = 1.0              # m/s; characterise the pre-stop curve while moving (low, so tight
  #                                    low-speed corners — intersection turns — aren't dropped)
  v_launch: float = 1.0              # m/s; launch is when speed climbs back above this after a stop
  kappa_corner: float = 0.008        # 1/m; min pre-stop curvature to count as a corner (~125 m radius)
  pre_stop_window_s: float = 2.0     # look-back window to estimate Z
  launch_window_s: float = 0.5       # window after launch to estimate X
  recovery_max_s: float = 12.0       # stop looking for recovery after this long (low speed evolves slowly)
  recovery_max_m: float = 30.0       # ...or after this far travelled, whichever comes first
  recovery_fraction: float = 0.7     # "recovered" when command reaches this fraction of |Z| in Z's dir
  continuation_frac: float = 0.5     # actual curvature must reach this fraction of |Z| to prove the curve continued
  continuation_bin_s: float = 0.5    # bin size for a robust (de-spiked) continuation peak
  min_stop_s: float = 0.5            # ignore micro-stops shorter than this
  amnesia_deficit_frac: float = 0.4  # launch undershoot >= this fraction of |Z| ⇒ amnesia
  pre_manual_frac: float = 0.5       # if the driver steered >= this fraction of the approach, the
  #                                    corner was human-driven — not attributable to openpilot


@dataclass(frozen=True)
class Sample:
  t: float
  v_ego: float
  standstill: bool
  steering_pressed: bool
  lane_change_active: bool
  lat_active: bool   # carControl.latActive — openpilot was actually steering
  cmd_curv: float    # controlsState.desiredCurvature — commanded (behaviour)
  act_curv: float    # controlsState.curvature — actual VM path curvature
  model_curv: float  # modelV2.action.desiredCurvature — raw vision


@dataclass
class CornerStopEvent:
  stop_start_s: float
  stop_end_s: float
  stop_duration_s: float
  classification: str  # amnesia | clean_recovery | straight_stop | curve_ended | driver_override | lane_change | no_launch
  is_amnesia: bool = False
  launch_s: float | None = None
  pre_stop_curv: float | None = None        # Z (signed)
  launch_curv: float | None = None          # X (signed, commanded)
  launch_curv_vision: float | None = None   # X_vis (signed, raw vision)
  launch_curv_actual: float | None = None   # actual VM curvature at launch
  continuation_curv: float | None = None    # peak actual curvature post-launch, in Z's direction (signed)
  deficit: float | None = None              # |Z| - |X|, 1/m
  deficit_frac: float | None = None         # deficit / |Z|
  deficit_vision: float | None = None       # |Z| - |X_vis|, 1/m (perception amnesia)
  recovery_lag_s: float | None = None
  recovery_lag_m: float | None = None
  recovery_censored: bool = False           # lag hit the window cap without the command reaching target
  flags: list[str] = field(default_factory=list)

  def to_dict(self) -> dict[str, Any]:
    return {
      "stop_start_s": _r(self.stop_start_s, 3),
      "stop_end_s": _r(self.stop_end_s, 3),
      "stop_duration_s": _r(self.stop_duration_s, 3),
      "launch_s": _r(self.launch_s, 3),
      "classification": self.classification,
      "is_amnesia": self.is_amnesia,
      "pre_stop_curv": _r(self.pre_stop_curv),
      "launch_curv": _r(self.launch_curv),
      "launch_curv_vision": _r(self.launch_curv_vision),
      "launch_curv_actual": _r(self.launch_curv_actual),
      "continuation_curv": _r(self.continuation_curv),
      "deficit": _r(self.deficit),
      "deficit_frac": _r(self.deficit_frac, 3),
      "deficit_vision": _r(self.deficit_vision),
      "recovery_lag_s": _r(self.recovery_lag_s, 3),
      "recovery_lag_m": _r(self.recovery_lag_m, 2),
      "recovery_censored": self.recovery_censored,
      "flags": list(self.flags),
    }


@dataclass
class CornerRecoveryReport:
  source: str
  duration_s: float
  sample_count: int
  total_stops: int
  corner_stops: int
  amnesia_events: int
  amnesia_driver_corrected: int
  clean_recoveries: int
  curve_ended: int
  manual_corner: int
  lane_change: int
  no_launch: int
  median_deficit: float | None
  p90_deficit: float | None
  median_deficit_frac: float | None
  median_deficit_vision: float | None
  median_recovery_lag_s: float | None
  p90_recovery_lag_s: float | None
  median_recovery_lag_m: float | None
  p90_recovery_lag_m: float | None
  worst_events: list[CornerStopEvent]
  notes: list[str]

  def to_dict(self) -> dict[str, Any]:
    return {
      "source": self.source,
      "duration_s": _r(self.duration_s, 2),
      "sample_count": self.sample_count,
      "total_stops": self.total_stops,
      "corner_stops": self.corner_stops,
      "amnesia_events": self.amnesia_events,
      "amnesia_driver_corrected": self.amnesia_driver_corrected,
      "clean_recoveries": self.clean_recoveries,
      "curve_ended": self.curve_ended,
      "manual_corner": self.manual_corner,
      "lane_change": self.lane_change,
      "no_launch": self.no_launch,
      "median_deficit": _r(self.median_deficit),
      "p90_deficit": _r(self.p90_deficit),
      "median_deficit_frac": _r(self.median_deficit_frac, 3),
      "median_deficit_vision": _r(self.median_deficit_vision),
      "median_recovery_lag_s": _r(self.median_recovery_lag_s, 3),
      "p90_recovery_lag_s": _r(self.p90_recovery_lag_s, 3),
      "median_recovery_lag_m": _r(self.median_recovery_lag_m, 2),
      "p90_recovery_lag_m": _r(self.p90_recovery_lag_m, 2),
      "worst_events": [event.to_dict() for event in self.worst_events],
      "notes": list(self.notes),
    }


def _r(value: Any, ndigits: int = 6) -> float | None:
  """Round for JSON, mapping non-finite/None to None so output stays valid JSON."""
  if value is None or not isinstance(value, int | float) or not math.isfinite(float(value)):
    return None
  return round(float(value), ndigits)


def _f(value: Any) -> float:
  try:
    out = float(value)
  except (TypeError, ValueError):
    return math.nan
  return out if math.isfinite(out) else math.nan


def _median(values: list[float]) -> float:
  finite = [v for v in values if math.isfinite(v)]
  return float(np.median(finite)) if finite else math.nan


def _lane_change_active(model_v2: Any) -> bool:
  if model_v2 is None:
    return False
  return format_enum(safe_get(model_v2, "meta.laneChangeState")) not in ("off", "unknown")


def extract_samples(msgs: list[Any]) -> tuple[list[Sample], bool]:
  """Resample the route onto the 100 Hz controlsState clock, zero-order-holding the other services.

  Returns (samples, saw_carcontrol). When carControl is absent we cannot tell engaged from manual
  driving, so lat_active is left True and the caller notes that engagement gating was disabled.
  """
  latest: dict[str, Any] = {}
  samples: list[Sample] = []
  saw_carcontrol = False
  for record in build_route_messages(msgs):
    typ, payload = record.typ, record.payload
    if typ in ("carState", "modelV2", "carControl"):
      latest[typ] = payload
      saw_carcontrol = saw_carcontrol or typ == "carControl"
      continue
    if typ != "controlsState":
      continue
    car_state = latest.get("carState")
    if car_state is None:
      continue
    model_v2 = latest.get("modelV2")
    car_control = latest.get("carControl")
    samples.append(Sample(
      t=record.t,
      v_ego=_f(safe_get(car_state, "vEgo")),
      standstill=bool(safe_get(car_state, "standstill", False)),
      steering_pressed=bool(safe_get(car_state, "steeringPressed", False)),
      lane_change_active=_lane_change_active(model_v2),
      lat_active=bool(safe_get(car_control, "latActive", False)) if car_control is not None else True,
      cmd_curv=_f(safe_get(payload, "desiredCurvature")),
      act_curv=_f(safe_get(payload, "curvature")),
      model_curv=_f(safe_get(model_v2, "action.desiredCurvature")),
    ))
  return samples, saw_carcontrol


def _stop_intervals(samples: list[Sample], p: DetectorParams) -> list[tuple[int, int]]:
  intervals: list[tuple[int, int]] = []
  start: int | None = None
  for i, s in enumerate(samples):
    stopped = s.standstill or (math.isfinite(s.v_ego) and s.v_ego <= p.v_stop)
    if stopped and start is None:
      start = i
    elif not stopped and start is not None:
      intervals.append((start, i - 1))
      start = None
  if start is not None:
    intervals.append((start, len(samples) - 1))
  return intervals


def _window(samples: list[Sample], t0: float, t1: float) -> list[Sample]:
  return [s for s in samples if t0 <= s.t < t1]


def _post_window(samples: list[Sample], launch_i: int, p: DetectorParams) -> list[tuple[Sample, float]]:
  """Samples from launch onward, each paired with distance travelled since launch, bounded by time/dist."""
  out: list[tuple[Sample, float]] = []
  dist = 0.0
  prev = samples[launch_i]
  launch_t = samples[launch_i].t
  for i in range(launch_i, len(samples)):
    s = samples[i]
    if i > launch_i and math.isfinite(s.v_ego) and math.isfinite(prev.v_ego):
      dist += 0.5 * (s.v_ego + prev.v_ego) * (s.t - prev.t)
    if s.t - launch_t > p.recovery_max_s or dist > p.recovery_max_m:
      break
    out.append((s, dist))
    prev = s
  return out


def _analyze_stop(samples: list[Sample], start_i: int, end_i: int, p: DetectorParams) -> CornerStopEvent | None:
  t_start = samples[start_i].t
  t_end = samples[end_i].t
  duration = t_end - t_start
  if duration < p.min_stop_s:
    return None  # micro-stop / roll, not the failure we care about

  # Z — what curve were we holding on the approach, while moving AND engaged?
  pre = [s for s in _window(samples, t_start - p.pre_stop_window_s, t_start)
         if math.isfinite(s.v_ego) and s.v_ego >= p.v_moving and s.lat_active]
  z_signed = _median([s.cmd_curv for s in pre])
  if not math.isfinite(z_signed):
    return None  # no engaged, moving approach to characterise (manual driving, or started stopped)
  z_mag = abs(z_signed)
  base = dict(stop_start_s=t_start, stop_end_s=t_end, stop_duration_s=duration, pre_stop_curv=z_signed)

  if z_mag < p.kappa_corner:
    return CornerStopEvent(classification="straight_stop", **base)

  # Launch — first time speed climbs back up after the stop while openpilot is steering again.
  launch_i: int | None = None
  for i in range(end_i + 1, len(samples)):
    if samples[i].t - t_end > p.recovery_max_s:
      break
    if math.isfinite(samples[i].v_ego) and samples[i].v_ego >= p.v_launch and samples[i].lat_active:
      launch_i = i
      break
  if launch_i is None:
    return CornerStopEvent(classification="no_launch", **base)  # never re-engaged / moved in window
  t_launch = samples[launch_i].t
  zsign = 1.0 if z_signed > 0 else -1.0

  post = _post_window(samples, launch_i, p)
  driver_corrected = any(s.steering_pressed for s, _ in post)        # driver on the wheel at launch
  lane_change = any(s.lane_change_active for s, _ in post)
  pre_manual = (sum(1 for s in pre if s.steering_pressed) / len(pre)) >= p.pre_manual_frac if pre else False
  flags: list[str] = []
  if driver_corrected:
    flags.append("driver_corrected")
  if lane_change:
    flags.append("lane_change")

  # X — curvature commanded right after launch (raw-vision / actual counterparts).
  launch_win = _window(samples, t_launch, t_launch + p.launch_window_s)
  x_signed = _median([s.cmd_curv for s in launch_win])
  if not math.isfinite(x_signed):
    x_signed = samples[launch_i].cmd_curv
  x_vis = _median([s.model_curv for s in launch_win])
  x_act = _median([s.act_curv for s in launch_win])
  base.update(launch_s=t_launch, launch_curv=x_signed, launch_curv_vision=x_vis, launch_curv_actual=x_act)

  # Continuation — did the curve actually continue? Peak of per-bin median actual curvature in Z's
  # direction, over moving samples, de-spiked by binning.
  bins: dict[int, list[float]] = {}
  for s, _ in post:
    if not (math.isfinite(s.v_ego) and s.v_ego >= p.v_launch and math.isfinite(s.act_curv)):
      continue
    bins.setdefault(int((s.t - t_launch) / p.continuation_bin_s), []).append(s.act_curv * zsign)
  continuation = max((float(np.median(v)) for v in bins.values()), default=float("-inf"))
  continued = math.isfinite(continuation) and continuation >= p.continuation_frac * z_mag
  base.update(continuation_curv=(continuation * zsign) if math.isfinite(continuation) else None)

  deficit = max(0.0, z_mag - abs(x_signed))
  deficit_frac = deficit / z_mag if z_mag > 0 else 0.0
  deficit_vision = max(0.0, z_mag - abs(x_vis)) if math.isfinite(x_vis) else None

  # Recovery lag — time/distance from launch until the command climbs back to recovery_fraction*|Z|
  # in Z's direction.
  target = p.recovery_fraction * z_mag
  lag_s: float = p.recovery_max_s
  lag_m: float = post[-1][1] if post else 0.0
  censored = True
  for s, d in post:
    if math.isfinite(s.cmd_curv) and s.cmd_curv * zsign >= target:
      lag_s, lag_m, censored = s.t - t_launch, d, False
      break

  # A driver grabbing the wheel during launch is amnesia *evidence* (a correction), not a disqualifier;
  # only a human-driven approach (pre_manual) makes the corner unattributable to openpilot.
  if pre_manual:
    classification = "manual_corner"
  elif lane_change:
    classification = "lane_change"
  elif not continued:
    classification = "curve_ended"
  elif deficit_frac >= p.amnesia_deficit_frac:
    classification = "amnesia"
  else:
    classification = "clean_recovery"

  return CornerStopEvent(
    classification=classification,
    is_amnesia=classification == "amnesia",
    deficit=deficit,
    deficit_frac=deficit_frac,
    deficit_vision=deficit_vision,
    recovery_lag_s=lag_s,
    recovery_lag_m=lag_m,
    recovery_censored=censored,
    flags=flags,
    **base,
  )


def analyze_route(msgs: list[Any], source: str = "unknown", params: DetectorParams | None = None,
                  top_n: int = 10) -> CornerRecoveryReport:
  p = params or DetectorParams()
  samples, saw_carcontrol = extract_samples(msgs)
  notes: list[str] = []
  if not samples:
    notes.append("no controlsState samples found (need rlogs with controlsState + carState)")
    return CornerRecoveryReport(
      source=source, duration_s=0.0, sample_count=0, total_stops=0, corner_stops=0,
      amnesia_events=0, amnesia_driver_corrected=0, clean_recoveries=0, curve_ended=0,
      manual_corner=0, lane_change=0, no_launch=0,
      median_deficit=None, p90_deficit=None, median_deficit_frac=None, median_deficit_vision=None,
      median_recovery_lag_s=None, p90_recovery_lag_s=None, median_recovery_lag_m=None,
      p90_recovery_lag_m=None, worst_events=[], notes=notes,
    )
  if not saw_carcontrol:
    notes.append("no carControl in logs — engagement gating disabled; results may include manual driving")

  duration = samples[-1].t - samples[0].t
  events = [e for (a, b) in _stop_intervals(samples, p) if (e := _analyze_stop(samples, a, b, p)) is not None]

  def count(name: str) -> int:
    return sum(1 for e in events if e.classification == name)

  amnesia = [e for e in events if e.classification == "amnesia"]
  corner_stops = sum(1 for e in events if e.classification != "straight_stop")
  deficits = [e.deficit for e in amnesia if e.deficit is not None]
  deficit_fracs = [e.deficit_frac for e in amnesia if e.deficit_frac is not None]
  deficit_visions = [e.deficit_vision for e in amnesia if e.deficit_vision is not None]
  lags_s = [e.recovery_lag_s for e in amnesia if e.recovery_lag_s is not None]
  lags_m = [e.recovery_lag_m for e in amnesia if e.recovery_lag_m is not None]
  worst = sorted(amnesia, key=lambda e: (e.deficit_frac or 0.0, e.deficit or 0.0), reverse=True)[:top_n]

  if not events:
    notes.append("no stops detected on this route")
  elif not corner_stops:
    notes.append("stops detected, but none while on a corner (|Z| >= kappa_corner)")

  return CornerRecoveryReport(
    source=source,
    duration_s=duration,
    sample_count=len(samples),
    total_stops=len(events),
    corner_stops=corner_stops,
    amnesia_events=len(amnesia),
    amnesia_driver_corrected=sum(1 for e in amnesia if "driver_corrected" in e.flags),
    clean_recoveries=count("clean_recovery"),
    curve_ended=count("curve_ended"),
    manual_corner=count("manual_corner"),
    lane_change=count("lane_change"),
    no_launch=count("no_launch"),
    median_deficit=_opt_median(deficits),
    p90_deficit=_opt_pct(deficits, 90),
    median_deficit_frac=_opt_median(deficit_fracs),
    median_deficit_vision=_opt_median(deficit_visions),
    median_recovery_lag_s=_opt_median(lags_s),
    p90_recovery_lag_s=_opt_pct(lags_s, 90),
    median_recovery_lag_m=_opt_median(lags_m),
    p90_recovery_lag_m=_opt_pct(lags_m, 90),
    worst_events=worst,
    notes=notes,
  )


def _opt_median(values: list[float]) -> float | None:
  return float(np.median(values)) if values else None


def _opt_pct(values: list[float], pct: float) -> float | None:
  return float(np.percentile(values, pct)) if values else None


def render_report(report: CornerRecoveryReport) -> str:
  lines = [
    f"Corner-recovery profile: {report.source}",
    (f"  duration {report.duration_s:.1f}s, {report.sample_count} samples, "
     + f"{report.total_stops} stops ({report.corner_stops} on a corner)"),
    "",
    f"  amnesia events:      {report.amnesia_events}  ({report.amnesia_driver_corrected} driver-corrected)",
    f"  clean recoveries:    {report.clean_recoveries}",
    f"  curve ended:         {report.curve_ended}   (road straightened / turn completed — not amnesia)",
    f"  manual corner:       {report.manual_corner}   (human steered the approach — not attributable)",
    f"  lane change:         {report.lane_change}",
    f"  never relaunched:    {report.no_launch}",
  ]
  if report.amnesia_events:
    deficit_line = (f"    curvature deficit |Z|-|X|:  median {_fmt(report.median_deficit)} 1/m, "
                    + f"p90 {_fmt(report.p90_deficit)} 1/m")
    vision_line = f"    perception deficit |Z|-|Xvis|: median {_fmt(report.median_deficit_vision)} 1/m"
    lag_line = (f"    recovery lag:               median {_fmt(report.median_recovery_lag_s, 2)} s / "
                + f"{_fmt(report.median_recovery_lag_m, 1)} m, "
                + f"p90 {_fmt(report.p90_recovery_lag_s, 2)} s / {_fmt(report.p90_recovery_lag_m, 1)} m")
    lines += [
      "",
      "  amnesia severity:",
      deficit_line,
      f"    deficit fraction:           median {_fmt(report.median_deficit_frac, 2)}",
      vision_line,
      lag_line,
      "",
      "  worst events (stop_start_s → launch_s : Z→X, deficit, lag):",
    ]
    for e in report.worst_events:
      lines.append(
        f"    t={_fmt(e.stop_start_s, 1)}s→{_fmt(e.launch_s, 1)}s  "
        + f"Z={_fmt(e.pre_stop_curv)} X={_fmt(e.launch_curv)} Xvis={_fmt(e.launch_curv_vision)} "
        + f"cont={_fmt(e.continuation_curv)}  deficit={_fmt(e.deficit)} ({_fmt(e.deficit_frac, 2)})  "
        + f"lag={_fmt(e.recovery_lag_s, 2)}s/{_fmt(e.recovery_lag_m, 1)}m"
      )
  for note in report.notes:
    lines.append(f"  note: {note}")
  return "\n".join(lines)


def _fmt(value: float | None, ndigits: int = 4) -> str:
  return "n/a" if value is None or not math.isfinite(float(value)) else f"{float(value):.{ndigits}f}"


def main() -> None:
  parser = argparse.ArgumentParser(description="Quantify steering 'amnesia' after stopping mid-corner.")
  parser.add_argument("route", help="Route, segment range, log file, or URL accepted by LogReader")
  parser.add_argument("--output", help="Write the report JSON to this path")
  parser.add_argument("--json", action="store_true", help="Print JSON instead of the text summary")
  parser.add_argument("--qlog", action="store_true", help="Prefer qlogs instead of rlogs (lower rate)")
  parser.add_argument("--kappa-corner", type=float, default=None, help="Min pre-stop curvature to count as a corner (1/m)")
  parser.add_argument("--amnesia-frac", type=float, default=None, help="Launch undershoot fraction that marks amnesia")
  parser.add_argument("--top", type=int, default=10, help="How many worst events to list")
  args = parser.parse_args()

  overrides: dict[str, float] = {}
  if args.kappa_corner is not None:
    overrides["kappa_corner"] = args.kappa_corner
  if args.amnesia_frac is not None:
    overrides["amnesia_deficit_frac"] = args.amnesia_frac
  params = DetectorParams(**overrides) if overrides else DetectorParams()

  msgs = load_route_msgs(args.route, qlog=args.qlog)
  report = analyze_route(msgs, source=args.route, params=params, top_n=args.top)
  print(output_report(report, json_output=args.json, renderer=render_report, output_path=args.output))


if __name__ == "__main__":
  main()
