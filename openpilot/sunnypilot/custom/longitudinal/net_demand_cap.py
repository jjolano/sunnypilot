"""Calibration-gated uphill net-demand acceleration cap.

Shadow mode always returns the existing target. Apply mode additionally requires a
Drive Lab-produced steep-climb profile and the global research-actuation gate.
"""
from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

GRAVITY = 9.81
PROFILE_VERSION = 1

# Shadow collection candidates only. Apply takes every estimator/filter threshold
# from the calibrated profile instead of promoting these guesses to actuation.
_SHADOW_MEDIAN_WINDOW_S = 0.25
_SHADOW_FILTER_TAU_S = 0.35
_SHADOW_MAX_POSE_AGE_S = 0.25
_SHADOW_MAX_PITCH_STD_RAD = 0.10
_SHADOW_MAX_DYNAMIC_A_EGO = 0.35
_SHADOW_MAX_DYNAMIC_JERK = 1.0
_SHADOW_MAX_GRADE_HOLD_S = 0.75
_COAST_SAMPLE_PERIOD_S = 0.5
_COAST_FIT_PERIOD_S = 10.0
_COAST_SAMPLE_LIMIT = 1200  # ten minutes of eligible 2 Hz manual-coast evidence


def _finite(value: Any) -> float | None:
  try:
    result = float(value)
  except (TypeError, ValueError):
    return None
  return result if math.isfinite(result) else None


def _param_text(params: Any, key: str) -> str | None:
  try:
    raw = params.get(key)
  except TypeError:
    raw = params.get(key, None)
  if raw is None:
    return None
  if isinstance(raw, bytes):
    raw = raw.decode(errors="ignore")
  return str(raw)


def sanitize_mode(value: Any) -> str:
  mode = str(value or "").strip().lower()
  return mode if mode in ("off", "shadow", "apply") else "off"


def parse_ceiling(value: Any) -> float | None:
  ceiling = _finite(value)
  return ceiling if ceiling is not None and 0.5 <= ceiling <= 2.0 else None


@dataclass(frozen=True)
class GradeProfile:
  pitch_zero_rad: float
  grade_enter_percent: float
  grade_hysteresis_percent: float
  entry_dwell_s: float
  exit_dwell_s: float
  min_speed_mps: float
  max_speed_mps: float
  median_window_s: float
  filter_tau_s: float
  max_pose_age_s: float
  max_pitch_std_rad: float
  max_dynamic_a_ego: float
  max_dynamic_jerk: float
  max_grade_hold_s: float
  calibration_rpy: tuple[float, float, float]
  max_calibration_delta_rad: float
  fit_slope: float
  fit_score: float
  fit_pitch_span: float
  fit_residual_mad: float
  fit_sample_count: int
  fit_speed_band_spread: float

  @property
  def grade_exit_percent(self) -> float:
    return self.grade_enter_percent - self.grade_hysteresis_percent


def parse_profile(value: Any) -> GradeProfile | None:
  if value is None:
    return None
  if isinstance(value, bytes):
    value = value.decode(errors="ignore")
  try:
    raw = json.loads(str(value))
  except (TypeError, ValueError, json.JSONDecodeError):
    return None
  if not isinstance(raw, dict) or raw.get("version") != PROFILE_VERSION or raw.get("calibrated") is not True:
    return None

  names = (
    "pitch_zero_rad", "grade_enter_percent", "grade_hysteresis_percent", "entry_dwell_s", "exit_dwell_s",
    "min_speed_mps", "max_speed_mps", "median_window_s", "filter_tau_s", "max_pose_age_s",
    "max_pitch_std_rad", "max_dynamic_a_ego", "max_dynamic_jerk", "max_grade_hold_s",
    "max_calibration_delta_rad", "fit_slope", "fit_score", "fit_pitch_span", "fit_residual_mad",
    "fit_sample_count", "fit_speed_band_spread",
  )
  values = {name: _finite(raw.get(name)) for name in names}
  rpy_raw = raw.get("calibration_rpy")
  if any(value is None for value in values.values()) or not isinstance(rpy_raw, list) or len(rpy_raw) != 3:
    return None
  rpy_values = tuple(_finite(value) for value in rpy_raw)
  if any(value is None for value in rpy_values):
    return None

  p = values
  if not (
    abs(p["pitch_zero_rad"]) <= 0.25
    and 0.0 < p["grade_enter_percent"] <= 30.0
    and 0.0 < p["grade_hysteresis_percent"] < p["grade_enter_percent"]
    and 0.0 <= p["entry_dwell_s"] <= 10.0
    and 0.0 <= p["exit_dwell_s"] <= 10.0
    and 0.0 <= p["min_speed_mps"] < p["max_speed_mps"] <= 60.0
    and 0.05 <= p["median_window_s"] <= 2.0
    and 0.05 <= p["filter_tau_s"] <= 3.0
    and 0.05 <= p["max_pose_age_s"] <= 2.0
    and 0.001 <= p["max_pitch_std_rad"] <= 0.5
    and 0.05 <= p["max_dynamic_a_ego"] <= 3.0
    and 0.1 <= p["max_dynamic_jerk"] <= 10.0
    and 0.05 <= p["max_grade_hold_s"] <= 10.0
    and 0.0001 <= p["max_calibration_delta_rad"] <= 0.2
    and p["fit_slope"] < 0.0
    and p["fit_pitch_span"] > 0.0
    and p["fit_residual_mad"] >= 0.0
    and p["fit_sample_count"] >= 1.0
    and p["fit_sample_count"].is_integer()
    and p["fit_speed_band_spread"] >= 0.0
  ):
    return None

  return GradeProfile(
    **{name: value for name, value in p.items() if name != "fit_sample_count"},
    fit_sample_count=int(p["fit_sample_count"]),
    calibration_rpy=(float(rpy_values[0]), float(rpy_values[1]), float(rpy_values[2])),
  )


@dataclass(frozen=True)
class CoastFit:
  ready: bool = False
  pitch_zero_rad: float = 0.0
  slope: float = 0.0
  score: float = 0.0
  pitch_span: float = 0.0
  residual_mad: float = 0.0
  sample_count: int = 0
  speed_band_spread: float = 0.0


def _ols(samples: list[tuple[float, float, float]]) -> tuple[float, float, float, list[float]] | None:
  if len(samples) < 3:
    return None
  pitches = [sample[0] for sample in samples]
  accels = [sample[1] for sample in samples]
  mean_pitch = sum(pitches) / len(pitches)
  mean_accel = sum(accels) / len(accels)
  variance = sum((pitch - mean_pitch) ** 2 for pitch in pitches)
  if variance <= 1e-10:
    return None
  slope = sum((pitch - mean_pitch) * (accel - mean_accel) for pitch, accel in zip(pitches, accels, strict=True)) / variance
  intercept = mean_accel - slope * mean_pitch
  residuals = [accel - (slope * pitch + intercept) for pitch, accel in zip(pitches, accels, strict=True)]
  total = sum((accel - mean_accel) ** 2 for accel in accels)
  score = 1.0 - sum(residual * residual for residual in residuals) / total if total > 1e-10 else 0.0
  return slope, intercept, score, residuals


def fit_coast_samples(samples: Iterable[tuple[float, float, float]]) -> CoastFit:
  rows = [row for row in samples if all(math.isfinite(value) for value in row)]
  initial = _ols(rows)
  if initial is None:
    return CoastFit(sample_count=len(rows))

  _, _, _, residuals = initial
  center = statistics.median(residuals)
  mad = statistics.median(abs(residual - center) for residual in residuals)
  if mad > 0.0:
    limit = 3.0 * 1.4826 * mad
    trimmed = [row for row, residual in zip(rows, residuals, strict=True) if abs(residual - center) <= limit]
  else:
    trimmed = rows
  fitted = _ols(trimmed)
  if fitted is None:
    return CoastFit(sample_count=len(trimmed))
  slope, intercept, score, residuals = fitted
  pitch_span = max(row[0] for row in trimmed) - min(row[0] for row in trimmed)
  residual_mad = statistics.median(abs(value - statistics.median(residuals)) for value in residuals)

  band_zeros = []
  bands: dict[int, list[tuple[float, float, float]]] = defaultdict(list)
  for row in trimmed:
    bands[int(row[2] // 5.0)].append(row)
  for band in bands.values():
    band_fit = _ols(band) if len(band) >= 10 else None
    if band_fit is not None and band_fit[0] < 0.0:
      band_zeros.append(-band_fit[1] / band_fit[0])

  # The estimator contract is the route-wide two-term regression zero. Speed-band
  # zeros are diagnostics: disagreement blocks offline profile promotion rather than
  # silently changing the definition of pitch zero.
  pitch_zero = -intercept / slope if slope != 0.0 else 0.0
  speed_band_spread = max(band_zeros) - min(band_zeros) if len(band_zeros) > 1 else 0.0
  ready = len(trimmed) >= 20 and slope < 0.0 and pitch_span > 1e-4 and math.isfinite(pitch_zero) and abs(pitch_zero) <= 0.25
  return CoastFit(
    ready=ready,
    pitch_zero_rad=pitch_zero if math.isfinite(pitch_zero) else 0.0,
    slope=slope,
    score=score,
    pitch_span=pitch_span,
    residual_mad=residual_mad,
    sample_count=len(trimmed),
    speed_band_spread=speed_band_spread,
  )


class _OnlineCoastFit:
  def __init__(self) -> None:
    self.samples: deque[tuple[float, float, float]] = deque(maxlen=_COAST_SAMPLE_LIMIT)
    self.sample_elapsed_s = 0.0
    self.fit_elapsed_s = 0.0
    self.result = CoastFit()

  def update(self, *, eligible: bool, pitch: float, a_ego: float, v_ego: float, dt: float) -> CoastFit:
    self.fit_elapsed_s += dt
    if eligible and v_ego >= 3.0:
      self.sample_elapsed_s += dt
      if self.sample_elapsed_s >= _COAST_SAMPLE_PERIOD_S:
        self.samples.append((pitch, a_ego, v_ego))
        self.sample_elapsed_s = 0.0
    else:
      self.sample_elapsed_s = 0.0
    if self.fit_elapsed_s >= _COAST_FIT_PERIOD_S:
      self.result = fit_coast_samples(self.samples)
      self.fit_elapsed_s = 0.0
    return self.result


@dataclass(frozen=True)
class NetDemandEvidence:
  mode: str = "off"
  ceiling: float | None = None
  profile: GradeProfile | None = None
  profile_ready: bool = False
  source_healthy: bool = False
  block_reason: str = "off"
  source_age_s: float = math.inf
  car_pitch: float = 0.0
  live_pose_pitch: float = 0.0
  pitch_zero: float = 0.0
  relative_pitch: float = 0.0
  filtered_grade_percent: float = 0.0
  grade_accel: float | None = None
  grade_held: bool = False
  fit: CoastFit = CoastFit()
  long_active: bool = False
  gas_pressed: bool = False
  brake_pressed: bool = False
  force_decel: bool = False
  has_lead: bool = False
  v_ego: float = 0.0
  research_actuation_allowed: bool = False


class UphillGradeEstimator:
  def __init__(self) -> None:
    self.mode = "off"
    self.ceiling: float | None = None
    self.profile: GradeProfile | None = None
    self._coast_fit = _OnlineCoastFit()
    self._median: deque[float] = deque(maxlen=1)
    self._filtered_pitch: float | None = None
    self._grade_hold_age_s = math.inf
    self._previous_a_ego: float | None = None

  def refresh_params(self, params: Any, *, allow_profile_change: bool = True) -> None:
    mode = sanitize_mode(_param_text(params, "UphillNetDemandCapMode"))
    ceiling = parse_ceiling(_param_text(params, "UphillNetDemandCeiling"))
    profile = parse_profile(_param_text(params, "UphillNetDemandGradeProfile"))
    if not allow_profile_change and profile != self.profile:
      profile = self.profile
    if profile != self.profile:
      self._median = deque(maxlen=1)
      self._filtered_pitch = None
      self._grade_hold_age_s = math.inf
    self.mode, self.ceiling, self.profile = mode, ceiling, profile

  def update(
    self, *, car_pitch: Any, live_pose_pitch: Any, pitch_std: Any, source_age_s: Any,
    source_valid: bool, calibration_valid: bool, calibration_rpy: Any,
    v_ego: Any, a_ego: Any, long_active: bool, gas_pressed: bool, brake_pressed: bool,
    force_decel: bool, has_lead: bool, research_actuation_allowed: bool, dt: Any,
  ) -> NetDemandEvidence:
    dt_f = _finite(dt)
    dt_f = dt_f if dt_f is not None and dt_f > 0.0 else 0.05
    car_pitch_f = _finite(car_pitch)
    live_pose_pitch_f = _finite(live_pose_pitch)
    pitch_std_f = _finite(pitch_std)
    source_age_f = _finite(source_age_s)
    v_ego_f = _finite(v_ego)
    a_ego_f = _finite(a_ego)
    try:
      # cereal exposes List(Float32) as a Cap'n Proto dynamic list, not a Python
      # list/tuple. Accept any finite three-value iterable.
      rpy = tuple(_finite(value) for value in calibration_rpy)
    except TypeError:
      rpy = ()
    if len(rpy) != 3:
      rpy = ()

    source_basic = bool(
      source_valid and calibration_valid and car_pitch_f is not None and live_pose_pitch_f is not None
      and pitch_std_f is not None and source_age_f is not None and v_ego_f is not None and a_ego_f is not None
      and len(rpy) == 3 and all(value is not None for value in rpy)
    )
    jerk = 0.0
    if a_ego_f is not None and self._previous_a_ego is not None:
      jerk = (a_ego_f - self._previous_a_ego) / dt_f
    self._previous_a_ego = a_ego_f
    fit_max_jerk = self.profile.max_dynamic_jerk if self.profile is not None else _SHADOW_MAX_DYNAMIC_JERK
    fit = self._coast_fit.update(
      eligible=(self.mode != "off" and source_basic and not long_active and not gas_pressed and not brake_pressed
                and abs(jerk) <= fit_max_jerk),
      pitch=car_pitch_f or 0.0,
      a_ego=a_ego_f or 0.0,
      v_ego=v_ego_f or 0.0,
      dt=dt_f,
    )

    profile_match = False
    if source_basic and self.profile is not None:
      calibration_delta = max(
        abs(float(value) - expected)
        for value, expected in zip(rpy, self.profile.calibration_rpy, strict=True)
      )
      profile_match = calibration_delta <= self.profile.max_calibration_delta_rad
    profile_ready = self.profile is not None and profile_match
    if profile_ready:
      fit = CoastFit(
        ready=True,
        pitch_zero_rad=self.profile.pitch_zero_rad,
        slope=self.profile.fit_slope,
        score=self.profile.fit_score,
        pitch_span=self.profile.fit_pitch_span,
        residual_mad=self.profile.fit_residual_mad,
        sample_count=self.profile.fit_sample_count,
        speed_band_spread=self.profile.fit_speed_band_spread,
      )

    max_age = self.profile.max_pose_age_s if profile_ready else _SHADOW_MAX_POSE_AGE_S
    max_std = self.profile.max_pitch_std_rad if profile_ready else _SHADOW_MAX_PITCH_STD_RAD
    source_healthy = source_basic and source_age_f <= max_age and pitch_std_f <= max_std
    pitch_zero = self.profile.pitch_zero_rad if profile_ready else (fit.pitch_zero_rad if fit.ready else None)

    max_a = self.profile.max_dynamic_a_ego if profile_ready else _SHADOW_MAX_DYNAMIC_A_EGO
    max_jerk = self.profile.max_dynamic_jerk if profile_ready else _SHADOW_MAX_DYNAMIC_JERK
    max_hold = self.profile.max_grade_hold_s if profile_ready else _SHADOW_MAX_GRADE_HOLD_S
    window_s = self.profile.median_window_s if profile_ready else _SHADOW_MEDIAN_WINDOW_S
    tau_s = self.profile.filter_tau_s if profile_ready else _SHADOW_FILTER_TAU_S
    grade_held = False

    if not source_healthy or pitch_zero is None:
      self._filtered_pitch = None
      self._median.clear()
      self._grade_hold_age_s = math.inf
      grade_accel = None
    else:
      dynamic = abs(a_ego_f or 0.0) > max_a or abs(jerk) > max_jerk
      filter_warmed = False
      if not dynamic:
        sample_count = max(1, round(window_s / dt_f))
        if self._median.maxlen != sample_count:
          self._median = deque(self._median, maxlen=sample_count)
        self._median.append(car_pitch_f - pitch_zero)
        median_pitch = statistics.median(self._median)
        alpha = dt_f / (tau_s + dt_f)
        self._filtered_pitch = median_pitch if self._filtered_pitch is None else self._filtered_pitch + alpha * (median_pitch - self._filtered_pitch)
        self._grade_hold_age_s = 0.0
        filter_warmed = len(self._median) >= sample_count
      else:
        self._grade_hold_age_s += dt_f
        grade_held = self._filtered_pitch is not None and self._grade_hold_age_s <= max_hold
        if not grade_held:
          self._filtered_pitch = None
        filter_warmed = grade_held
      grade_accel = (
        GRAVITY * math.sin(self._filtered_pitch)
        if self._filtered_pitch is not None and (not profile_ready or filter_warmed) else None
      )

    if self.mode == "off":
      block_reason = "off"
    elif self.ceiling is None:
      block_reason = "invalid_ceiling"
    elif not source_basic:
      block_reason = "source_invalid"
    elif self.mode == "apply" and self.profile is None:
      block_reason = "profile_not_calibrated"
    elif self.mode == "apply" and not profile_match:
      block_reason = "calibration_mismatch"
    elif not source_healthy:
      block_reason = "source_stale_or_noisy"
    elif pitch_zero is None:
      block_reason = "pitch_zero_unavailable"
    elif grade_accel is None:
      block_reason = "grade_stale"
    else:
      block_reason = ""

    relative_pitch = self._filtered_pitch or 0.0
    return NetDemandEvidence(
      mode=self.mode,
      ceiling=self.ceiling,
      profile=self.profile,
      profile_ready=profile_ready,
      source_healthy=source_healthy,
      block_reason=block_reason,
      source_age_s=source_age_f if source_age_f is not None else math.inf,
      car_pitch=car_pitch_f or 0.0,
      live_pose_pitch=live_pose_pitch_f or 0.0,
      pitch_zero=pitch_zero or 0.0,
      relative_pitch=relative_pitch,
      filtered_grade_percent=100.0 * math.tan(relative_pitch),
      grade_accel=grade_accel,
      grade_held=grade_held,
      fit=fit,
      long_active=bool(long_active),
      gas_pressed=bool(gas_pressed),
      brake_pressed=bool(brake_pressed),
      force_decel=bool(force_decel),
      has_lead=bool(has_lead),
      v_ego=v_ego_f or 0.0,
      research_actuation_allowed=bool(research_actuation_allowed),
    )


@dataclass(frozen=True)
class NetDemandCapTrace:
  mode: str = "off"
  effective_mode: str = "off"
  eligible: bool = False
  would_cap: bool = False
  applied: bool = False
  block_reason: str = "off"
  regime: str = "hold"
  source: str = "carControl.orientationNED"
  source_age_s: float = 0.0
  car_pitch: float = 0.0
  live_pose_pitch: float = 0.0
  pitch_zero: float = 0.0
  relative_pitch: float = 0.0
  filtered_grade_percent: float = 0.0
  profile_ready: bool = False
  fit_slope: float = 0.0
  fit_score: float = 0.0
  fit_pitch_span: float = 0.0
  fit_residual_mad: float = 0.0
  fit_sample_count: int = 0
  fit_speed_band_spread: float = 0.0
  ceiling: float = 0.0
  grade_enter_percent: float = 0.0
  grade_exit_percent: float = 0.0
  grade_accel: float = 0.0
  a_target_before: float = 0.0
  a_target_cap: float = 0.0
  a_target_after: float = 0.0
  requested_net_demand: float = 0.0
  delta_a: float = 0.0
  grade_load_exceeds_ceiling: bool = False
  grade_held: bool = False
  research_actuation_allowed: bool = False
  has_lead: bool = False


class NetDemandCapFinalStage:
  def __init__(self) -> None:
    self.regime = "hold"
    self._entry_s = 0.0
    self._exit_s = 0.0
    self.applied_last_tick = False

  def _reset(self) -> None:
    self.regime = "hold"
    self._entry_s = 0.0
    self._exit_s = 0.0

  def apply(self, a_target: float, evidence: NetDemandEvidence, *, should_stop: bool, dt: float) -> tuple[float, NetDemandCapTrace]:
    before = float(a_target)
    ceiling = _finite(evidence.ceiling)
    grade_accel = _finite(evidence.grade_accel)
    calculable = ceiling is not None and grade_accel is not None and math.isfinite(before)
    cap = max(0.0, ceiling - grade_accel) if calculable else before
    would_cap = bool(calculable and grade_accel > 0.0 and before > cap)
    requested_net = before + grade_accel if grade_accel is not None else before
    profile = evidence.profile
    effective_mode = evidence.mode
    eligible = False
    block_reason = evidence.block_reason

    scene_block = ""
    if not evidence.long_active:
      scene_block = "longitudinal_inactive"
    elif evidence.gas_pressed or evidence.brake_pressed:
      scene_block = "pedal_pressed"
    elif evidence.force_decel:
      scene_block = "force_decel"
    elif should_stop:
      scene_block = "stop_context"
    elif before <= 0.0:
      scene_block = "nonpositive_target"
    elif evidence.has_lead:
      scene_block = "lead_present"

    after = before
    if evidence.mode == "off":
      self._reset()
      block_reason = "off"
    elif evidence.mode == "shadow":
      self._reset()
      effective_mode = "shadow"
      eligible = calculable and not bool(scene_block)
      block_reason = scene_block or evidence.block_reason or "shadow_only"
    else:
      if not evidence.research_actuation_allowed:
        effective_mode = "shadow"
        block_reason = "research_gate_off"
      elif not evidence.profile_ready or profile is None:
        effective_mode = "shadow"
        block_reason = evidence.block_reason or "profile_not_calibrated"
      elif scene_block:
        block_reason = scene_block
      elif not calculable:
        block_reason = evidence.block_reason or "grade_unavailable"
      elif not profile.min_speed_mps <= evidence.v_ego <= profile.max_speed_mps:
        block_reason = "outside_calibrated_speed"
      else:
        eligible = True
        dt_f = dt if math.isfinite(dt) and dt > 0.0 else 0.0
        grade = evidence.filtered_grade_percent
        if self.regime == "hold":
          self._exit_s = 0.0
          above_enter = grade >= profile.grade_enter_percent
          self._entry_s = self._entry_s + dt_f if above_enter else 0.0
          if above_enter and self._entry_s >= profile.entry_dwell_s:
            self.regime = "cap"
            self._entry_s = 0.0
        else:
          self._entry_s = 0.0
          below_exit = grade <= profile.grade_exit_percent
          self._exit_s = self._exit_s + dt_f if below_exit else 0.0
          if below_exit and self._exit_s >= profile.exit_dwell_s:
            self._reset()
        if self.regime == "cap":
          after = min(before, cap)
          block_reason = "" if after < before else "below_net_demand_ceiling"
        else:
          block_reason = "below_grade_threshold"

    if evidence.mode != "apply" or effective_mode != "apply" or not eligible:
      if evidence.mode == "apply" and not eligible:
        self._reset()
      after = before

    trace = NetDemandCapTrace(
      mode=evidence.mode,
      effective_mode=effective_mode,
      eligible=eligible,
      would_cap=would_cap,
      applied=after < before,
      block_reason=block_reason,
      regime=self.regime,
      source_age_s=evidence.source_age_s if math.isfinite(evidence.source_age_s) else 0.0,
      car_pitch=evidence.car_pitch,
      live_pose_pitch=evidence.live_pose_pitch,
      pitch_zero=evidence.pitch_zero,
      relative_pitch=evidence.relative_pitch,
      filtered_grade_percent=evidence.filtered_grade_percent,
      profile_ready=evidence.profile_ready,
      fit_slope=evidence.fit.slope,
      fit_score=evidence.fit.score,
      fit_pitch_span=evidence.fit.pitch_span,
      fit_residual_mad=evidence.fit.residual_mad,
      fit_sample_count=evidence.fit.sample_count,
      fit_speed_band_spread=evidence.fit.speed_band_spread,
      ceiling=ceiling or 0.0,
      grade_enter_percent=profile.grade_enter_percent if profile is not None else 0.0,
      grade_exit_percent=profile.grade_exit_percent if profile is not None else 0.0,
      grade_accel=grade_accel or 0.0,
      a_target_before=before,
      a_target_cap=cap,
      a_target_after=after,
      requested_net_demand=requested_net,
      delta_a=after - before,
      grade_load_exceeds_ceiling=bool(calculable and grade_accel > ceiling),
      grade_held=evidence.grade_held,
      research_actuation_allowed=evidence.research_actuation_allowed,
      has_lead=evidence.has_lead,
    )
    self.applied_last_tick = trace.applied
    return after, trace
