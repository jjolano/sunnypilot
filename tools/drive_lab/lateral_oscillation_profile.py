from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path
from typing import Any

import numpy as np

from openpilot.tools.drive_lab.route_analysis import build_route_messages, conditioned_desired_curvature, lateral_demand_schema
from openpilot.tools.drive_lab.timeline import format_enum, safe_get


@dataclass(frozen=True)
class LateralWindow:
  start_s: float
  end_s: float
  sample_count: int
  straight_coverage: float
  speed_mps_median: float
  steering_angle_pp: float
  raw_curvature_pp: float
  processed_curvature_pp: float
  desired_curvature_pp: float
  actual_curvature_pp: float
  output_pp: float
  requested_torque_pp: float
  applied_torque_pp: float
  eps_torque_pp: float
  driver_torque_p95: float
  command_eps_corr: float | None
  command_applied_corr: float | None
  raw_actual_corr: float | None
  raw_steering_corr: float | None
  gated_percent: float
  quality_median: float


@dataclass(frozen=True)
class LateralOscillationProfile:
  source: str
  sample_count: int
  duration_s: float
  active_percent: float
  straight_candidate_percent: float
  straight_sample_count: int
  straight_steering_angle_pp: float
  straight_raw_curvature_pp: float
  straight_processed_curvature_pp: float
  straight_desired_curvature_pp: float
  straight_actual_curvature_pp: float
  straight_requested_torque_pp: float
  straight_applied_torque_pp: float
  straight_eps_torque_pp: float
  straight_driver_torque_p95: float
  straight_command_eps_corr: float | None
  raw_actual_corr: float | None
  raw_steering_corr: float | None
  desired_actual_corr: float | None
  top_windows: list[LateralWindow]

  def to_dict(self) -> dict[str, Any]:
    return asdict(self)

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> LateralOscillationProfile:
    return cls(
      source=str(data.get("source", "unknown")),
      sample_count=int(data.get("sample_count", data.get("sampleCount", 0))),
      duration_s=float(data.get("duration_s", data.get("durationS", 0.0))),
      active_percent=float(data.get("active_percent", data.get("activePercent", 0.0))),
      straight_candidate_percent=float(data.get("straight_candidate_percent", data.get("straightCandidatePercent", 0.0))),
      straight_sample_count=int(data.get("straight_sample_count", data.get("straightSampleCount", 0))),
      straight_steering_angle_pp=float(data.get("straight_steering_angle_pp", data.get("straightSteeringAnglePp", 0.0))),
      straight_raw_curvature_pp=float(data.get("straight_raw_curvature_pp", data.get("straightRawCurvaturePp", 0.0))),
      straight_processed_curvature_pp=float(data.get("straight_processed_curvature_pp", data.get("straightProcessedCurvaturePp", 0.0))),
      straight_desired_curvature_pp=float(data.get("straight_desired_curvature_pp", data.get("straightDesiredCurvaturePp", 0.0))),
      straight_actual_curvature_pp=float(data.get("straight_actual_curvature_pp", data.get("straightActualCurvaturePp", 0.0))),
      straight_requested_torque_pp=float(data.get("straight_requested_torque_pp", data.get("straightRequestedTorquePp", 0.0))),
      straight_applied_torque_pp=float(data.get("straight_applied_torque_pp", data.get("straightAppliedTorquePp", 0.0))),
      straight_eps_torque_pp=float(data.get("straight_eps_torque_pp", data.get("straightEpsTorquePp", 0.0))),
      straight_driver_torque_p95=float(data.get("straight_driver_torque_p95", data.get("straightDriverTorqueP95", 0.0))),
      straight_command_eps_corr=_optional_float(data.get("straight_command_eps_corr", data.get("straightCommandEpsCorr"))),
      raw_actual_corr=_optional_float(data.get("raw_actual_corr", data.get("rawActualCorr"))),
      raw_steering_corr=_optional_float(data.get("raw_steering_corr", data.get("rawSteeringCorr"))),
      desired_actual_corr=_optional_float(data.get("desired_actual_corr", data.get("desiredActualCorr"))),
      top_windows=[_window_from_dict(window) for window in data.get("top_windows", data.get("topWindows", []))],
    )


@dataclass(frozen=True)
class _LateralSample:
  t: float
  v_ego: float
  lat_active: bool
  steering_pressed: bool
  blinker_active: bool
  lane_change_state: str
  steering_angle_deg: float
  curvature: float
  raw_desired_curvature: float
  processed_desired_curvature: float
  desired_curvature: float
  lat_output: float
  requested_torque: float
  applied_torque: float
  eps_torque: float
  driver_torque: float
  model_path_gated: bool
  model_path_quality: float


def build_lateral_oscillation_profile(
  msgs: list[Any],
  source: str = "unknown",
  already_sorted: bool = False,
  min_speed: float = 8.0,
  max_raw_curvature: float = 0.002,
  window_s: float = 30.0,
  step_s: float = 5.0,
  max_windows: int = 8,
) -> LateralOscillationProfile:
  ordered_msgs = list(msgs) if already_sorted else sorted(msgs, key=lambda m: int(getattr(m, "logMonoTime", 0)))
  samples = _extract_lateral_samples(ordered_msgs)
  if not samples:
    return LateralOscillationProfile(source, 0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                     0.0, 0.0, 0.0, 0.0, None, None, None, None, [])

  cols = _columns(samples)
  straight = _straight_mask(cols, min_speed, max_raw_curvature)
  active = cols["lat_active"] > 0.5
  windows = _rank_windows(cols, straight, window_s, step_s, max_windows)
  straight_idx = np.where(straight)[0]
  duration_s = float(cols["t"][-1] - cols["t"][0]) if len(samples) > 1 else 0.0
  return LateralOscillationProfile(
    source=source,
    sample_count=len(samples),
    duration_s=duration_s,
    active_percent=_percent(active),
    straight_candidate_percent=_percent(straight),
    straight_sample_count=int(np.sum(straight)),
    straight_steering_angle_pp=_percentile_span(cols["steering_angle_deg"][straight_idx]),
    straight_raw_curvature_pp=_percentile_span(cols["raw_desired_curvature"][straight_idx]),
    straight_processed_curvature_pp=_percentile_span(cols["processed_desired_curvature"][straight_idx]),
    straight_desired_curvature_pp=_percentile_span(cols["desired_curvature"][straight_idx]),
    straight_actual_curvature_pp=_percentile_span(cols["curvature"][straight_idx]),
    straight_requested_torque_pp=_percentile_span(cols["requested_torque"][straight_idx]),
    straight_applied_torque_pp=_percentile_span(cols["applied_torque"][straight_idx]),
    straight_eps_torque_pp=_percentile_span(cols["eps_torque"][straight_idx]),
    straight_driver_torque_p95=_p95_abs(cols["driver_torque"][straight_idx]),
    straight_command_eps_corr=_correlation(cols["requested_torque"][straight_idx], cols["eps_torque"][straight_idx]),
    raw_actual_corr=_correlation(cols["raw_desired_curvature"][straight_idx], cols["curvature"][straight_idx]),
    raw_steering_corr=_correlation(cols["raw_desired_curvature"][straight_idx], cols["steering_angle_deg"][straight_idx]),
    desired_actual_corr=_correlation(cols["desired_curvature"][straight_idx], cols["curvature"][straight_idx]),
    top_windows=windows,
  )


def render_lateral_profile(profile: LateralOscillationProfile) -> str:
  lines = [
    f"Lateral oscillation profile: {profile.source}",
    f"samples: {profile.sample_count}",
    f"duration: {profile.duration_s:.1f} s",
    f"active: {profile.active_percent:.1f}%",
    f"straight candidates: {profile.straight_candidate_percent:.1f}% ({profile.straight_sample_count} samples)",
    "Straight-candidate summary:",
    f"  steering angle pp: {profile.straight_steering_angle_pp:.3f} deg",
    f"  raw curvature pp: {profile.straight_raw_curvature_pp:.6f}",
    f"  processed curvature pp: {profile.straight_processed_curvature_pp:.6f}",
    f"  desired curvature pp: {profile.straight_desired_curvature_pp:.6f}",
    f"  actual curvature pp: {profile.straight_actual_curvature_pp:.6f}",
    f"  requested/applied torque pp: {profile.straight_requested_torque_pp:.3f}/{profile.straight_applied_torque_pp:.3f}",
    f"  eps torque pp: {profile.straight_eps_torque_pp:.3f}",
    f"  driver torque p95: {profile.straight_driver_torque_p95:.3f}",
    f"  corr command->eps: {_format_optional(profile.straight_command_eps_corr)}",
    f"  corr raw->actual: {_format_optional(profile.raw_actual_corr)}",
    f"  corr raw->steering: {_format_optional(profile.raw_steering_corr)}",
    f"  corr desired->actual: {_format_optional(profile.desired_actual_corr)}",
  ]
  if profile.top_windows:
    lines.append("Top windows:")
    for window in profile.top_windows:
      lines.append(
        f"  {window.start_s:.1f}-{window.end_s:.1f}s "
        f"steer_pp={window.steering_angle_pp:.3f}deg "
        f"raw_pp={window.raw_curvature_pp:.6f} "
        f"desired_pp={window.desired_curvature_pp:.6f} "
        f"actual_pp={window.actual_curvature_pp:.6f} "
        f"cmd_pp={window.requested_torque_pp:.3f} "
        f"eps_pp={window.eps_torque_pp:.3f} "
        f"driver95={window.driver_torque_p95:.3f} "
        f"cmd_eps={_format_optional(window.command_eps_corr)} "
        f"corr_raw_actual={_format_optional(window.raw_actual_corr)} "
        f"gated={window.gated_percent:.1f}%"
      )
  return "\n".join(lines)


def save_lateral_profile(profile: LateralOscillationProfile, path: str | Path) -> None:
  Path(path).write_text(json.dumps(profile.to_dict(), indent=2) + "\n")


def load_lateral_profile(path: str | Path) -> LateralOscillationProfile:
  return LateralOscillationProfile.from_dict(json.loads(Path(path).read_text()))


def _extract_lateral_samples(msgs: list[Any]) -> list[_LateralSample]:
  if not msgs:
    return []
  demand_schema = lateral_demand_schema(msgs)
  latest: dict[str, Any] = {}
  samples: list[_LateralSample] = []
  for route_msg in build_route_messages(msgs):
    typ = route_msg.typ
    payload = route_msg.payload
    if typ in ("carState", "carControl", "carOutput", "modelV2"):
      latest[typ] = payload
    if typ != "controlsState":
      continue

    car_state = latest.get("carState")
    car_control = latest.get("carControl")
    car_output = latest.get("carOutput")
    model_v2 = latest.get("modelV2")
    lateral_state = safe_get(payload, "lateralControlState")
    lateral_payload = safe_get(lateral_state, format_enum(lateral_state.which()) if lateral_state is not None and hasattr(lateral_state, "which") else "torqueState", lateral_state)
    model_path = safe_get(payload, "modelPathState")
    samples.append(_LateralSample(
      t=route_msg.t,
      v_ego=_finite_float(safe_get(car_state, "vEgo")),
      lat_active=bool(safe_get(lateral_payload, "active", False)) and bool(safe_get(car_control, "latActive", False)),
      steering_pressed=bool(safe_get(car_state, "steeringPressed", False)),
      blinker_active=bool(safe_get(car_state, "leftBlinker", False)) or bool(safe_get(car_state, "rightBlinker", False)),
      lane_change_state=format_enum(safe_get(model_v2, "meta.laneChangeState")),
      steering_angle_deg=_finite_float(safe_get(car_state, "steeringAngleDeg")),
      curvature=_finite_float(safe_get(payload, "curvature")),
      raw_desired_curvature=_finite_float(safe_get(model_path, "rawDesiredCurvature")),
      processed_desired_curvature=_finite_float(conditioned_desired_curvature(model_path, demand_schema)),
      desired_curvature=_finite_float(safe_get(payload, "desiredCurvature")),
      lat_output=_finite_float(safe_get(lateral_payload, "output")),
      requested_torque=_finite_float(safe_get(car_control, "actuators.torque")),
      applied_torque=_finite_float(safe_get(car_output, "actuatorsOutput.torque")),
      eps_torque=_finite_float(safe_get(car_state, "steeringTorqueEps")),
      driver_torque=_finite_float(safe_get(car_state, "steeringTorque")),
      model_path_gated=bool(safe_get(model_path, "gated", False)),
      model_path_quality=_finite_float(safe_get(model_path, "quality")),
    ))
  return samples


def _columns(samples: list[_LateralSample]) -> dict[str, np.ndarray]:
  return {
    "t": np.array([sample.t for sample in samples], dtype=float),
    "v_ego": np.array([sample.v_ego for sample in samples], dtype=float),
    "lat_active": np.array([float(sample.lat_active) for sample in samples], dtype=float),
    "steering_pressed": np.array([float(sample.steering_pressed) for sample in samples], dtype=float),
    "blinker_active": np.array([float(sample.blinker_active) for sample in samples], dtype=float),
    "lane_change_off": np.array([float(sample.lane_change_state == "off") for sample in samples], dtype=float),
    "lane_change_unknown": np.array([float(sample.lane_change_state == "unknown") for sample in samples], dtype=float),
    "steering_angle_deg": np.array([sample.steering_angle_deg for sample in samples], dtype=float),
    "curvature": np.array([sample.curvature for sample in samples], dtype=float),
    "raw_desired_curvature": np.array([sample.raw_desired_curvature for sample in samples], dtype=float),
    "processed_desired_curvature": np.array([sample.processed_desired_curvature for sample in samples], dtype=float),
    "desired_curvature": np.array([sample.desired_curvature for sample in samples], dtype=float),
    "lat_output": np.array([sample.lat_output for sample in samples], dtype=float),
    "requested_torque": np.array([sample.requested_torque for sample in samples], dtype=float),
    "applied_torque": np.array([sample.applied_torque for sample in samples], dtype=float),
    "eps_torque": np.array([sample.eps_torque for sample in samples], dtype=float),
    "driver_torque": np.array([sample.driver_torque for sample in samples], dtype=float),
    "model_path_gated": np.array([float(sample.model_path_gated) for sample in samples], dtype=float),
    "model_path_quality": np.array([sample.model_path_quality for sample in samples], dtype=float),
  }


def _straight_mask(cols: dict[str, np.ndarray], min_speed: float, max_raw_curvature: float) -> np.ndarray:
  return (
    (cols["lat_active"] > 0.5)
    & (cols["v_ego"] > min_speed)
    & (cols["steering_pressed"] < 0.5)
    & (cols["blinker_active"] < 0.5)
    & ((cols["lane_change_off"] > 0.5) | (cols["lane_change_unknown"] > 0.5))
    & (np.abs(cols["raw_desired_curvature"]) < max_raw_curvature)
  )


def _rank_windows(cols: dict[str, np.ndarray], straight: np.ndarray, window_s: float, step_s: float, max_windows: int) -> list[LateralWindow]:
  t = cols["t"]
  if len(t) == 0:
    return []
  windows: list[LateralWindow] = []
  start = float(t[0])
  end = float(t[-1])
  cur = start
  while cur + window_s <= end:
    mask = (t >= cur) & (t < cur + window_s)
    if np.any(mask) and np.mean(straight[mask]) >= 0.75:
      idx = np.where(mask & straight)[0]
      windows.append(LateralWindow(
        start_s=cur,
        end_s=cur + window_s,
        sample_count=int(len(idx)),
        straight_coverage=float(np.mean(straight[mask])),
        speed_mps_median=_median(cols["v_ego"][idx]),
        steering_angle_pp=_percentile_span(cols["steering_angle_deg"][idx]),
        raw_curvature_pp=_percentile_span(cols["raw_desired_curvature"][idx]),
        processed_curvature_pp=_percentile_span(cols["processed_desired_curvature"][idx]),
        desired_curvature_pp=_percentile_span(cols["desired_curvature"][idx]),
        actual_curvature_pp=_percentile_span(cols["curvature"][idx]),
        output_pp=_percentile_span(cols["lat_output"][idx]),
        requested_torque_pp=_percentile_span(cols["requested_torque"][idx]),
        applied_torque_pp=_percentile_span(cols["applied_torque"][idx]),
        eps_torque_pp=_percentile_span(cols["eps_torque"][idx]),
        driver_torque_p95=_p95_abs(cols["driver_torque"][idx]),
        command_eps_corr=_correlation(cols["requested_torque"][idx], cols["eps_torque"][idx]),
        command_applied_corr=_correlation(cols["requested_torque"][idx], cols["applied_torque"][idx]),
        raw_actual_corr=_correlation(cols["raw_desired_curvature"][idx], cols["curvature"][idx]),
        raw_steering_corr=_correlation(cols["raw_desired_curvature"][idx], cols["steering_angle_deg"][idx]),
        gated_percent=_percent(cols["model_path_gated"][idx] > 0.5),
        quality_median=_median(cols["model_path_quality"][idx]),
      ))
    cur += step_s
  windows.sort(key=lambda window: window.steering_angle_pp, reverse=True)
  return windows[:max_windows]


def _window_from_dict(data: dict[str, Any]) -> LateralWindow:
  return LateralWindow(
    start_s=float(data.get("start_s", data.get("startS", 0.0))),
    end_s=float(data.get("end_s", data.get("endS", 0.0))),
    sample_count=int(data.get("sample_count", data.get("sampleCount", 0))),
    straight_coverage=float(data.get("straight_coverage", data.get("straightCoverage", 0.0))),
    speed_mps_median=float(data.get("speed_mps_median", data.get("speedMpsMedian", 0.0))),
    steering_angle_pp=float(data.get("steering_angle_pp", data.get("steeringAnglePp", 0.0))),
    raw_curvature_pp=float(data.get("raw_curvature_pp", data.get("rawCurvaturePp", 0.0))),
    processed_curvature_pp=float(data.get("processed_curvature_pp", data.get("processedCurvaturePp", 0.0))),
    desired_curvature_pp=float(data.get("desired_curvature_pp", data.get("desiredCurvaturePp", 0.0))),
    actual_curvature_pp=float(data.get("actual_curvature_pp", data.get("actualCurvaturePp", 0.0))),
    output_pp=float(data.get("output_pp", data.get("outputPp", 0.0))),
    requested_torque_pp=float(data.get("requested_torque_pp", data.get("requestedTorquePp", 0.0))),
    applied_torque_pp=float(data.get("applied_torque_pp", data.get("appliedTorquePp", 0.0))),
    eps_torque_pp=float(data.get("eps_torque_pp", data.get("epsTorquePp", 0.0))),
    driver_torque_p95=float(data.get("driver_torque_p95", data.get("driverTorqueP95", 0.0))),
    command_eps_corr=_optional_float(data.get("command_eps_corr", data.get("commandEpsCorr"))),
    command_applied_corr=_optional_float(data.get("command_applied_corr", data.get("commandAppliedCorr"))),
    raw_actual_corr=_optional_float(data.get("raw_actual_corr", data.get("rawActualCorr"))),
    raw_steering_corr=_optional_float(data.get("raw_steering_corr", data.get("rawSteeringCorr"))),
    gated_percent=float(data.get("gated_percent", data.get("gatedPercent", 0.0))),
    quality_median=float(data.get("quality_median", data.get("qualityMedian", 0.0))),
  )


def _finite_float(value: Any, default: float = np.nan) -> float:
  try:
    candidate = float(value)
  except (TypeError, ValueError):
    return default
  return candidate if isfinite(candidate) else default


def _optional_float(value: Any) -> float | None:
  if value is None:
    return None
  candidate = _finite_float(value)
  return candidate if isfinite(candidate) else None


def _finite_values(values: np.ndarray) -> np.ndarray:
  return values[np.isfinite(values)]


def _percent(mask: np.ndarray) -> float:
  return float(100.0 * np.mean(mask)) if len(mask) else 0.0


def _median(values: np.ndarray) -> float:
  finite = _finite_values(values)
  return float(np.median(finite)) if len(finite) else 0.0


def _percentile_span(values: np.ndarray) -> float:
  finite = _finite_values(values)
  if len(finite) == 0:
    return 0.0
  return float(np.percentile(finite, 95.0) - np.percentile(finite, 5.0))


def _p95_abs(values: np.ndarray) -> float:
  finite = np.abs(_finite_values(values))
  return float(np.percentile(finite, 95.0)) if len(finite) else 0.0


def _correlation(left: np.ndarray, right: np.ndarray) -> float | None:
  good = np.isfinite(left) & np.isfinite(right)
  if int(np.sum(good)) < 3:
    return None
  left_good = left[good]
  right_good = right[good]
  if float(np.std(left_good)) <= 1e-12 or float(np.std(right_good)) <= 1e-12:
    return None
  return float(np.corrcoef(left_good, right_good)[0, 1])


def _format_optional(value: float | None) -> str:
  return "n/a" if value is None else f"{value:.3f}"
