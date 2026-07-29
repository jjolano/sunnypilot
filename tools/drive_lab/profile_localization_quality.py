#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Any

from openpilot.tools.drive_lab.route_analysis import build_route_messages
from openpilot.tools.drive_lab.route_io import load_route_msgs, output_report
from openpilot.tools.drive_lab.timeline import safe_get
from sunnypilot.custom.localization.quality import (
  LocalizationQualityHealth,
  LocalizationQualityThresholds,
  FrequencySummary,
  StdSummary,
  freshness_summary,
  heading_error_deg,
  nearest_value,
  std_summary,
  vector_norm3,
)


def _f(value: Any) -> float:
  try:
    out = float(value)
  except (TypeError, ValueError):
    return math.nan
  return out if math.isfinite(out) else math.nan


def _opt_round(value: float | None, ndigits: int = 6) -> float | None:
  if value is None or not math.isfinite(float(value)):
    return None
  return round(float(value), ndigits)


def _percent(value: int, total: int) -> float:
  return (100.0 * value / total) if total else 0.0


THRESHOLDS = LocalizationQualityThresholds()


@dataclass(frozen=True)
class FlagRateSummary:
  inputsOK_count: int
  posenetOK_count: int
  sensorsOK_count: int
  all_ok_count: int
  inputsOK_pct: float
  posenetOK_pct: float
  sensorsOK_pct: float
  all_ok_pct: float

  def to_dict(self) -> dict[str, Any]:
    return {k: _opt_round(v, 3) if isinstance(v, float) else v for k, v in self.__dict__.items()}


@dataclass(frozen=True)
class ConsistencySummary:
  pair_count: int
  p95_abs_error: float | None
  max_abs_error: float | None

  def to_dict(self) -> dict[str, Any]:
    return {"pair_count": self.pair_count, "p95_abs_error": _opt_round(self.p95_abs_error, 6), "max_abs_error": _opt_round(self.max_abs_error, 6)}


@dataclass(frozen=True)
class LocalizationQualityReport:
  source: str
  sample_count: int
  duration_s: float
  cameraOdometry_frequency: FrequencySummary
  livePose_frequency: FrequencySummary
  livePose_flags: FlagRateSummary
  cameraOdometry_std: dict[str, StdSummary]
  cameraOdometry_high_trans_std_count: int
  cameraOdometry_invalid_missing_vector_count: int
  livePose_measurement_std: dict[str, StdSummary]
  consistency: dict[str, ConsistencySummary]
  health: LocalizationHealthSummary
  notes: list[str]

  def to_dict(self) -> dict[str, Any]:
    return {
      "source": self.source,
      "sample_count": self.sample_count,
      "duration_s": _opt_round(self.duration_s, 3),
      "cameraOdometry_frequency": self.cameraOdometry_frequency.to_dict(),
      "livePose_frequency": self.livePose_frequency.to_dict(),
      "livePose_flags": self.livePose_flags.to_dict(),
      "cameraOdometry_std": {k: v.to_dict() for k, v in self.cameraOdometry_std.items()},
      "cameraOdometry_high_trans_std_count": self.cameraOdometry_high_trans_std_count,
      "cameraOdometry_invalid_missing_vector_count": self.cameraOdometry_invalid_missing_vector_count,
      "livePose_measurement_std": {k: v.to_dict() for k, v in self.livePose_measurement_std.items()},
      "consistency": {k: v.to_dict() for k, v in self.consistency.items()},
      "health": self.health.to_dict(),
      "notes": list(self.notes),
    }


@dataclass(frozen=True)
class LocalizationHealthSummary:
  ok: bool
  degraded_reasons: list[str]

  def to_dict(self) -> dict[str, Any]:
    return {"ok": self.ok, "degraded_reasons": list(self.degraded_reasons)}


def _measurement_std_values(meas: Any) -> list[float]:
  values = []
  for key in ("xStd", "yStd", "zStd"):
    values.append(_f(safe_get(meas, key)))
  if any(math.isfinite(v) for v in values):
    return values
  legacy = safe_get(meas, "std")
  if isinstance(legacy, (list, tuple)):
    return [_f(v) for v in legacy]
  return []


def _gps_message_fields(payload: Any) -> tuple[bool, float | None, float | None, float | None]:
  fix = bool(safe_get(payload, "hasFix", safe_get(payload, "valid", False)))
  speed = _f(safe_get(payload, "speed"))
  accuracy = _f(safe_get(payload, "bearingAccuracyDeg", safe_get(payload, "accuracy")))
  bearing = _f(safe_get(payload, "bearingDeg"))
  return fix, speed, accuracy, bearing


def _extract_report(msgs: list[Any], source: str) -> LocalizationQualityReport:
  ordered_msgs = sorted(msgs, key=lambda m: int(getattr(m, "logMonoTime", 0)))
  records = build_route_messages(ordered_msgs)
  sample_count = len(records)
  notes: list[str] = []
  cam_t: list[float] = []
  live_t: list[float] = []
  live_flags = dict(inputs=0, posenet=0, sensors=0, all_ok=0)
  cam_trans_norm: list[float] = []
  cam_rot_norm: list[float] = []
  cam_invalid = 0
  live_measurement_std: dict[str, list[float]] = {k: [] for k in ("orientationNED", "velocityDevice", "angularVelocityDevice", "accelerationDevice")}
  cam_yr: list[tuple[float, float]] = []
  live_yaw: list[tuple[float, float]] = []
  gps_headings: list[tuple[float, float]] = []
  gps_messages_present = False

  for rec in records:
    typ, payload, t = rec.typ, rec.payload, rec.t
    if typ == "cameraOdometry":
      cam_t.append(t)
      trans_std = safe_get(payload, "transStd") or []
      rot_std = safe_get(payload, "rotStd") or []
      if len(trans_std) < 3 or len(rot_std) < 3:
        cam_invalid += 1
      trans_norm = vector_norm3(trans_std)
      rot_norm = vector_norm3(rot_std)
      cam_trans_norm.append(trans_norm if trans_norm is not None else math.nan)
      cam_rot_norm.append(rot_norm if rot_norm is not None else math.nan)
      rot = safe_get(payload, "rot") or []
      cam_yr.append((t, _f(rot[2]) if len(rot) > 2 else math.nan))
    elif typ == "livePose":
      live_t.append(t)
      inputs_ok = bool(safe_get(payload, "inputsOK", False))
      posenet_ok = bool(safe_get(payload, "posenetOK", False))
      sensors_ok = bool(safe_get(payload, "sensorsOK", False))
      live_flags["inputs"] += int(inputs_ok)
      live_flags["posenet"] += int(posenet_ok)
      live_flags["sensors"] += int(sensors_ok)
      live_flags["all_ok"] += int(inputs_ok and posenet_ok and sensors_ok)
      for key in live_measurement_std:
        meas = safe_get(payload, key)
        live_measurement_std[key].extend(_measurement_std_values(meas))
      live_yaw.append((t, _f(safe_get(payload, "angularVelocityDevice.z"))))
    elif typ in ("gpsLocation", "gpsLocationExternal"):
      gps_messages_present = True
      fix, speed, accuracy, bearing = _gps_message_fields(payload)
      if (fix and speed is not None and accuracy is not None and bearing is not None
          and math.isfinite(speed) and speed > THRESHOLDS.gps_speed_min
          and math.isfinite(accuracy) and accuracy <= THRESHOLDS.gps_bearing_accuracy_max_deg
          and math.isfinite(bearing)):
        gps_headings.append((t, math.radians(bearing)))

  cam_near = [(t, v) for t, v in cam_yr if math.isfinite(v)]
  live_near = [(t, v) for t, v in live_yaw if math.isfinite(v)]
  live_near_times = [t for t, _ in live_near]
  yaw_errors = []
  for t, yr in cam_near:
    lp = nearest_value(live_near, t, THRESHOLDS.camera_yaw_window_s, times=live_near_times)
    if lp is not None:
      yaw_errors.append(abs(yr - lp))
  gps_errors = []
  live_heading = [(rec.t, _f(safe_get(rec.payload, "orientationNED.z"))) for rec in records if rec.typ == "livePose"]
  live_heading = [(t, v) for t, v in live_heading if math.isfinite(v)]
  live_heading_times = [t for t, _ in live_heading]
  for t, heading in gps_headings:
    lp = nearest_value(live_heading, t, THRESHOLDS.gps_heading_window_s, times=live_heading_times)
    if lp is not None:
      gps_errors.append(abs(heading_error_deg(math.degrees(lp), math.degrees(heading))))

  health_model = LocalizationQualityHealth.from_signals(
    camera_fresh=freshness_summary(cam_t, thresholds=THRESHOLDS),
    live_fresh=freshness_summary(live_t, thresholds=THRESHOLDS),
    high_trans_std_count=sum(1 for v in cam_trans_norm if math.isfinite(v) and v > THRESHOLDS.high_trans_std_norm),
    yaw_pair_count=len(yaw_errors),
    gps_pair_count=len(gps_errors) if gps_messages_present else None,
    gps_p95_abs_error_deg=std_summary(gps_errors).p95 if len(gps_errors) >= 3 else None,
    thresholds=THRESHOLDS,
  )
  health = LocalizationHealthSummary(ok=health_model.ok, degraded_reasons=list(health_model.degraded_reasons))
  if gps_messages_present and not gps_errors:
    notes.append("GPS messages present but no valid heading pairs")

  return LocalizationQualityReport(
    source=source, sample_count=sample_count, duration_s=(records[-1].t - records[0].t) if records else 0.0,
    cameraOdometry_frequency=freshness_summary(cam_t, thresholds=THRESHOLDS), livePose_frequency=freshness_summary(live_t, thresholds=THRESHOLDS),
    livePose_flags=FlagRateSummary(live_flags["inputs"], live_flags["posenet"], live_flags["sensors"], live_flags["all_ok"],
                                   _percent(live_flags["inputs"], len(live_t)), _percent(live_flags["posenet"], len(live_t)),
                                   _percent(live_flags["sensors"], len(live_t)), _percent(live_flags["all_ok"], len(live_t))),
    cameraOdometry_std={"transStd": std_summary(cam_trans_norm), "rotStd": std_summary(cam_rot_norm)},
    cameraOdometry_high_trans_std_count=sum(1 for v in cam_trans_norm if math.isfinite(v) and v > THRESHOLDS.high_trans_std_norm),
    cameraOdometry_invalid_missing_vector_count=cam_invalid, livePose_measurement_std={k: std_summary(v) for k, v in live_measurement_std.items()},
    consistency={
      "cameraOdometry_yaw_rate_z_vs_livePose_angularVelocityDevice.z": ConsistencySummary(len(yaw_errors), std_summary(yaw_errors).p95, max(yaw_errors) if yaw_errors else None),
      "gps_bearing_vs_livePose_orientationNED.z": ConsistencySummary(len(gps_errors), std_summary(gps_errors).p95 if len(gps_errors) >= 3 else None, max(gps_errors) if gps_errors else None),
    },
    health=health,
    notes=notes,
  )


def render_report(report: LocalizationQualityReport) -> str:
  lines = [
    f"Localization quality: {report.source}",
    f"  samples={report.sample_count} duration={report.duration_s:.1f}s",
    f"  cameraOdometry: n={report.cameraOdometry_frequency.samples} hz={_fmt(report.cameraOdometry_frequency.observed_hz)} max_gap={_fmt(report.cameraOdometry_frequency.max_gap_s)} p95_gap={_fmt(report.cameraOdometry_frequency.p95_gap_s)} large_gaps={report.cameraOdometry_frequency.large_gap_count}",
    f"  livePose: n={report.livePose_frequency.samples} hz={_fmt(report.livePose_frequency.observed_hz)} max_gap={_fmt(report.livePose_frequency.max_gap_s)} p95_gap={_fmt(report.livePose_frequency.p95_gap_s)} large_gaps={report.livePose_frequency.large_gap_count}",
    f"  health: ok={report.health.ok} degraded={'; '.join(report.health.degraded_reasons) if report.health.degraded_reasons else 'none'}",
    f"  livePose flags: inputsOK {report.livePose_flags.inputsOK_count} ({report.livePose_flags.inputsOK_pct:.1f}%) posenetOK {report.livePose_flags.posenetOK_count} ({report.livePose_flags.posenetOK_pct:.1f}%) sensorsOK {report.livePose_flags.sensorsOK_count} ({report.livePose_flags.sensorsOK_pct:.1f}%) all_ok {report.livePose_flags.all_ok_count} ({report.livePose_flags.all_ok_pct:.1f}%)",
    f"  cameraOdometry std: transStd(norm) p95={_fmt(report.cameraOdometry_std['transStd'].p95)} max={_fmt(report.cameraOdometry_std['transStd'].max)} rotStd(norm) p95={_fmt(report.cameraOdometry_std['rotStd'].p95)} max={_fmt(report.cameraOdometry_std['rotStd'].max)} high_trans_std_count={report.cameraOdometry_high_trans_std_count} invalid/missing={report.cameraOdometry_invalid_missing_vector_count}",
  ]
  for name, summary in report.livePose_measurement_std.items():
    lines.append(f"  livePose {name} std: p95={_fmt(summary.p95)} max={_fmt(summary.max)}")
  for name, summary in report.consistency.items():
    lines.append(f"  {name}: pairs={summary.pair_count} p95_abs={_fmt(summary.p95_abs_error)} max_abs={_fmt(summary.max_abs_error)}")
  for note in report.notes:
    lines.append(f"  note: {note}")
  return "\n".join(lines)


def _fmt(value: float | None, ndigits: int = 3) -> str:
  return "n/a" if value is None or not math.isfinite(float(value)) else f"{float(value):.{ndigits}f}"


def main() -> None:
  parser = argparse.ArgumentParser(description="Profile localization quality from route logs.")
  parser.add_argument("routes", nargs="+", help="Routes, rlog files, or URLs accepted by LogReader")
  parser.add_argument("--qlog", action="store_true", help="Prefer qlogs instead of rlogs")
  parser.add_argument("--json", action="store_true", help="Print JSON instead of the text summary")
  parser.add_argument("--output", help="Write the report JSON to this path")
  args = parser.parse_args()
  for route in args.routes:
    msgs = load_route_msgs(route, qlog=args.qlog)
    report = _extract_report(msgs, source=route)
    print(output_report(report, json_output=args.json, renderer=render_report, output_path=args.output))


if __name__ == "__main__":
  main()
