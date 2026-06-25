"""Shadow-only curve-aware advisory + traffic context for the custom-2.0 longitudinal policy.

Phase 1 is intentionally non-actuating. It estimates path curvature from modelV2.position,
proposes a speed/accel cap for telemetry, and classifies the traffic context around the ego
vehicle. All outputs are debug-only; no value here is allowed to shape a_target, v_cruise,
should_stop, release behavior, or policy candidates.

The helper is pure: it reads mode, kinematics, model path, and lead state passed in by the
stack/wiring and performs no Params I/O.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any

MODE_OFF = "off"
MODE_SHADOW = "shadow"
MODE_APPLY_CONSERVATIVE = "apply_conservative"
VALID_MODES = {MODE_OFF, MODE_SHADOW, MODE_APPLY_CONSERVATIVE}

KAPPA_EPS = 1e-4
KAPPA_NOISE_FLOOR = 0.0015
A_LAT_TARGET_STANDARD = 1.8
PRE_ENTRY_A_LAT = 1.0
ENTRY_A_LAT = 1.3
APEX_CURRENT_A_LAT = 1.6
LEAVING_A_LAT = 1.3
FINISH_A_LAT = 1.1
A_CURVE_DECEL_FLOOR = -0.85
LOW_SPEED_MIN_M_S = 3.0

MIN_PATH_SAMPLES = 8
MIN_MONOTONIC_FRACTION = 0.80
MAX_CURVATURE = 0.25

PEAK_NEAR_WINDOW_M = 6.0
EXIT_LOOKAHEAD_WINDOW_M = 30.0


@dataclass(frozen=True)
class CurveTrafficAdvisorInputs:
  v_ego: float = 0.0
  a_ego: float = 0.0
  model_msg: Any | None = None
  leads: tuple[Any, Any] = (None, None)
  lead_shadow_active: bool = False
  alternate_threat_active: bool = False
  long_active: bool = False
  model_stale: bool = False
  brake_pressed: bool = False
  gas_pressed: bool = False
  force_slow_decel: bool = False


@dataclass(frozen=True)
class CurveTrafficAdvisorResult:
  mode: str = MODE_OFF
  effective_mode: str = MODE_OFF
  apply_supported: bool = False
  eligible: bool = False
  active: bool = False
  confidence: float = 0.0
  phase: str = "inactive"
  curvature_now: float = 0.0
  curvature_peak: float = 0.0
  curvature_sign: float = 0.0
  distance_to_curve: float = 0.0
  distance_to_apex: float = 0.0
  v_curve_cap_proposed: float = 0.0
  a_curve_cap_proposed: float = 0.0
  suppress_accel: bool = False
  traffic_block_reason: str = ""
  s_curve: bool = False
  compound_curve: bool = False
  block_reason: str = "mode_off"
  fault: bool = False

  def debug_dict(self) -> dict[str, Any]:
    prefix = "curve_traffic"
    return {
      f"{prefix}_mode": self.mode,
      f"{prefix}_effective_mode": self.effective_mode,
      f"{prefix}_apply_supported": self.apply_supported,
      f"{prefix}_eligible": self.eligible,
      f"{prefix}_active": self.active,
      f"{prefix}_confidence": self.confidence,
      f"{prefix}_phase": self.phase,
      f"{prefix}_curvature_now": self.curvature_now,
      f"{prefix}_curvature_peak": self.curvature_peak,
      f"{prefix}_curvature_sign": self.curvature_sign,
      f"{prefix}_distance_to_curve": self.distance_to_curve,
      f"{prefix}_distance_to_apex": self.distance_to_apex,
      f"{prefix}_v_curve_cap_proposed": self.v_curve_cap_proposed,
      f"{prefix}_a_curve_cap_proposed": self.a_curve_cap_proposed,
      f"{prefix}_suppress_accel": self.suppress_accel,
      f"{prefix}_traffic_block_reason": self.traffic_block_reason,
      f"{prefix}_s_curve": self.s_curve,
      f"{prefix}_compound_curve": self.compound_curve,
      f"{prefix}_block_reason": self.block_reason,
      f"{prefix}_fault": self.fault,
    }


def _f(value: Any, default: float = 0.0) -> float:
  try:
    v = float(value)
  except (TypeError, ValueError):
    return default
  return v if math.isfinite(v) else default


def _combine_block_reasons(geometry_reason: str, traffic_reason: str) -> str:
  parts = [geometry_reason, traffic_reason] if geometry_reason else [traffic_reason]
  joined = ",".join(p for p in parts if p)
  return joined


def _finite_sequence(value: Any) -> list[float]:
  out: list[float] = []
  try:
    it = iter(value)
  except TypeError:
    v = _f(value, math.nan)
    return [v] if math.isfinite(v) else []
  for item in it:
    v = _f(item, math.nan)
    if not math.isfinite(v):
      return []
    out.append(v)
  return out


def _extract_path(model_msg: Any | None) -> tuple[list[float], list[float]] | None:
  if model_msg is None:
    return None
  position = getattr(model_msg, "position", None)
  if position is None:
    return None
  xs = _finite_sequence(getattr(position, "x", None))
  ys = _finite_sequence(getattr(position, "y", None))
  if len(xs) < MIN_PATH_SAMPLES or len(xs) != len(ys):
    return None
  return xs, ys


def _validate_path(xs: list[float], ys: list[float]) -> bool:
  if len(xs) < MIN_PATH_SAMPLES or len(xs) != len(ys):
    return False
  n = len(xs) - 1
  increasing = sum(1 for i in range(n) if xs[i + 1] > xs[i] - 1e-6)
  return increasing >= MIN_MONOTONIC_FRACTION * n and xs[-1] > xs[0]


def _first_derivative(vs: list[float]) -> list[float]:
  n = len(vs)
  if n < 2:
    return [0.0] * n
  out: list[float] = [vs[1] - vs[0]]
  for i in range(1, n - 1):
    out.append((vs[i + 1] - vs[i - 1]) * 0.5)
  out.append(vs[-1] - vs[-2])
  return out


def _second_derivative(vs: list[float]) -> list[float]:
  n = len(vs)
  if n < 3:
    return [0.0] * n
  out: list[float] = []
  d0 = vs[2] - 2.0 * vs[1] + vs[0]
  out.append(d0)
  for i in range(1, n - 1):
    out.append(vs[i + 1] - 2.0 * vs[i] + vs[i - 1])
  out.append(vs[-1] - 2.0 * vs[-2] + vs[-3])
  return out


def _smooth_3tap(values: list[float]) -> list[float]:
  n = len(values)
  if n < 3:
    return list(values)
  out: list[float] = [values[0]]
  for i in range(1, n - 1):
    out.append((values[i - 1] + values[i] + values[i + 1]) / 3.0)
  out.append(values[-1])
  return out


def _estimate_curvature(xs: list[float], ys: list[float]) -> list[float]:
  dx = _first_derivative(xs)
  dy = _first_derivative(ys)
  d2x = _second_derivative(xs)
  d2y = _second_derivative(ys)
  kappa: list[float] = []
  for x1, y1, x2, y2 in zip(dx, dy, d2x, d2y, strict=True):
    denom = (x1 * x1 + y1 * y1) ** 1.5
    if denom < KAPPA_EPS:
      kappa.append(0.0)
      continue
    k = (x1 * y2 - y1 * x2) / denom
    k = max(-MAX_CURVATURE, min(MAX_CURVATURE, k))
    kappa.append(k)
  return _smooth_3tap(kappa)


def _local_maxima_indices(values: list[float], xs: list[float], threshold: float) -> list[int]:
  """Return indices of real curvature humps, filtering noisy plateau edges.

  A local maximum only counts if it stands above its immediate neighbors by a
  meaningful prominence. Tiny numerical ripples on an otherwise constant arc are
  rejected, while two separate bends separated by a straight section still count.
  """
  peaks: list[int] = []
  n = len(values)
  if n < 3:
    return peaks
  for i in range(1, n - 1):
    if abs(values[i]) < threshold:
      continue
    if not (abs(values[i]) > abs(values[i - 1]) and abs(values[i]) > abs(values[i + 1])):
      continue
    prominence = min(0.25 * abs(values[i]), 4.0 * KAPPA_NOISE_FLOOR)
    if abs(abs(values[i]) - abs(values[i - 1])) < prominence and \
       abs(abs(values[i]) - abs(values[i + 1])) < prominence:
      continue
    if not peaks or (xs[i] - xs[peaks[-1]]) > 12.0:
      peaks.append(i)
    elif abs(values[i]) > abs(values[peaks[-1]]):
      peaks[-1] = i
  return peaks


def _detect_compound_curve(kappa: list[float]) -> bool:
  """True when there are two distinct curve regions separated by a low-curvature gap.

  A single sustained bend above a high threshold is one region; two separate bends
  (e.g., radius change with a tangent between) produce two regions and are labeled
  compound. The threshold is relative to the peak so modest noise does not split a
  single arc into pieces.
  """
  significant = [abs(k) for k in kappa if abs(k) > KAPPA_NOISE_FLOOR]
  if len(significant) < 4:
    return False
  peak = max(significant)
  threshold = max(0.5 * peak, 2.0 * KAPPA_NOISE_FLOOR)
  groups = 0
  in_group = False
  for k in kappa:
    above = abs(k) > threshold
    if above and not in_group:
      groups += 1
      in_group = True
    elif not above:
      in_group = False
  return groups >= 2


def _phase_label(idx: int, peak_idx: int, xs: list[float], kappa: list[float],
                 s_curve: bool, compound_curve: bool, curvature_now: float) -> str:
  if s_curve and compound_curve:
    return "compound"
  if s_curve:
    return "s_curve"
  if compound_curve:
    return "compound"

  if peak_idx < 0 or idx >= len(xs) or peak_idx >= len(xs):
    return "inactive"

  peak_x = xs[peak_idx]
  current_x = xs[idx] if idx >= 0 else 0.0
  distance_to_peak = peak_x - current_x

  if abs(curvature_now) < KAPPA_NOISE_FLOOR and distance_to_peak > PEAK_NEAR_WINDOW_M:
    return "pre_entry"

  if abs(curvature_now) < KAPPA_NOISE_FLOOR:
    # Either just before/after a short curve or in a quiet stretch.
    return "exit" if current_x > peak_x else "pre_entry"

  if abs(distance_to_peak) <= PEAK_NEAR_WINDOW_M:
    return "apex"

  if current_x < peak_x:
    return "entry"

  after_peak = current_x - peak_x
  if after_peak > 0.0 and after_peak < EXIT_LOOKAHEAD_WINDOW_M:
    return "exit"
  return "exit"


def _a_lat_for_phase(phase: str) -> float:
  return {
    "pre_entry": PRE_ENTRY_A_LAT,
    "entry": ENTRY_A_LAT,
    "apex": APEX_CURRENT_A_LAT,
    "exit": LEAVING_A_LAT,
    "compound": A_LAT_TARGET_STANDARD,
    "s_curve": A_LAT_TARGET_STANDARD,
  }.get(phase, A_LAT_TARGET_STANDARD)


def _v_from_curvature(kappa: float, a_lat: float) -> float:
  k = max(abs(kappa), KAPPA_EPS)
  return math.sqrt(max(0.0, a_lat / k))


def _evaluate_traffic(data: CurveTrafficAdvisorInputs) -> tuple[bool, str]:
  v_ego = max(0.0, _f(data.v_ego))
  reasons: list[str] = []

  if data.lead_shadow_active:
    reasons.append("shadow_lead")
  if data.alternate_threat_active:
    reasons.append("alternate_threat")

  for lead in data.leads:
    if lead is None or not bool(getattr(lead, "status", False)):
      continue
    d_rel = _f(getattr(lead, "dRel", math.inf), math.inf)
    v_rel = _f(getattr(lead, "vRel", 0.0))
    a_lead_k = _f(getattr(lead, "aLeadK", 0.0))
    gap = max(6.0, 1.5 * v_ego)

    if d_rel < 15.0 and v_rel < -1.5:
      reasons.append("close_closing_lead")
    elif d_rel < gap and v_rel < -0.5:
      reasons.append("closing_lead")
    if a_lead_k < -1.5 and d_rel < 60.0:
      reasons.append("braking_lead")

  block_reason = ",".join(reasons)
  return bool(reasons), block_reason


def predict_curve_traffic_advisor(mode: Any, data: CurveTrafficAdvisorInputs) -> CurveTrafficAdvisorResult:
  mode_s = str(mode or "").strip().lower()
  if mode_s not in VALID_MODES:
    mode_s = MODE_OFF

  if mode_s == MODE_OFF:
    return CurveTrafficAdvisorResult(mode=MODE_OFF, effective_mode=MODE_OFF)

  # apply_conservative is accepted for storage compatibility but remains non-actuating.
  effective_mode = MODE_SHADOW if mode_s == MODE_APPLY_CONSERVATIVE else mode_s
  base = CurveTrafficAdvisorResult(
    mode=mode_s,
    effective_mode=effective_mode,
    apply_supported=False,
  )

  if not data.long_active:
    return replace(base, block_reason="long_inactive")

  # Traffic awareness is evaluated independently of road geometry so failures in the
  # path-side computation never suppress this debug-only signal and vice-versa.
  suppress_accel, traffic_block_reason = _evaluate_traffic(data)

  if data.model_stale:
    return replace(
      base,
      suppress_accel=suppress_accel,
      traffic_block_reason=traffic_block_reason,
      block_reason=_combine_block_reasons("model_stale", traffic_block_reason),
    )
  if data.brake_pressed or data.gas_pressed:
    return replace(
      base,
      suppress_accel=suppress_accel,
      traffic_block_reason=traffic_block_reason,
      block_reason=_combine_block_reasons("driver_override", traffic_block_reason),
    )
  if data.force_slow_decel:
    return replace(
      base,
      suppress_accel=suppress_accel,
      traffic_block_reason=traffic_block_reason,
      block_reason=_combine_block_reasons("force_slow", traffic_block_reason),
    )

  v_ego = max(0.0, _f(data.v_ego))
  path = _extract_path(data.model_msg)
  if path is None:
    return replace(
      base,
      suppress_accel=suppress_accel,
      traffic_block_reason=traffic_block_reason,
      block_reason=_combine_block_reasons("invalid_path", traffic_block_reason),
    )
  xs, ys = path
  if not _validate_path(xs, ys):
    return replace(
      base,
      suppress_accel=suppress_accel,
      traffic_block_reason=traffic_block_reason,
      block_reason=_combine_block_reasons("invalid_path", traffic_block_reason),
    )

  kappa = _estimate_curvature(xs, ys)
  if len(kappa) != len(xs):
    return replace(
      base,
      suppress_accel=suppress_accel,
      traffic_block_reason=traffic_block_reason,
      block_reason=_combine_block_reasons("invalid_path", traffic_block_reason),
    )

  current_kappa = kappa[0]
  curvature_now = float(current_kappa)
  curvature_sign = math.copysign(1.0, current_kappa) if abs(current_kappa) > KAPPA_EPS else 0.0

  # Sign changes indicate an s-curve; exclude tiny noise near zero.
  signs = [math.copysign(1.0, k) if abs(k) > KAPPA_NOISE_FLOOR else 0.0 for k in kappa]
  non_zero_signs = [s for s in signs if s != 0.0]
  s_curve = len(non_zero_signs) >= 2 and any(
    non_zero_signs[i] != non_zero_signs[i + 1]
    for i in range(len(non_zero_signs) - 1)
  )

  peaks = _local_maxima_indices(kappa, xs, KAPPA_NOISE_FLOOR)
  compound_curve = _detect_compound_curve(kappa)

  if not peaks:
    # If no formal peak but curvature is sustained, treat the first significant point as the peak.
    significant = [i for i, k in enumerate(kappa) if abs(k) > KAPPA_NOISE_FLOOR]
    if not significant:
      return replace(
        base,
        block_reason=_combine_block_reasons("below_noise_floor", traffic_block_reason),
        curvature_now=curvature_now,
        curvature_sign=curvature_sign,
        suppress_accel=suppress_accel,
        traffic_block_reason=traffic_block_reason,
      )
    peak_idx = max(significant, key=lambda i: abs(kappa[i]))
  else:
    peak_idx = peaks[0]

  curvature_peak = float(kappa[peak_idx])

  # Distances are relative to the first model path point (ego position).
  significant_indices = [i for i, k in enumerate(kappa) if abs(k) > KAPPA_NOISE_FLOOR]
  distance_to_curve = xs[significant_indices[0]] - xs[0] if significant_indices else xs[peak_idx] - xs[0]
  distance_to_apex = xs[peak_idx] - xs[0]

  phase = _phase_label(0, peak_idx, xs, kappa, s_curve, compound_curve, curvature_now)

  # Proposed speed cap: conservative minimum over the curve region.
  curve_region = [i for i, k in enumerate(kappa) if abs(k) > KAPPA_NOISE_FLOOR]
  if not curve_region:
    curve_region = [peak_idx]
  v_cap = float("inf")
  for i in curve_region:
    phase_i = _phase_label(i, peak_idx, xs, kappa, s_curve, compound_curve, kappa[i])
    v_i = _v_from_curvature(kappa[i], _a_lat_for_phase(phase_i))
    v_cap = min(v_cap, v_i)
  v_cap = max(LOW_SPEED_MIN_M_S, v_cap)

  # Proposed accel cap: kinematic deceleration to v_cap by the apex; negative-only.
  d_apex = distance_to_apex
  if v_ego > v_cap and d_apex > 0.5:
    a_proposed = (v_cap * v_cap - v_ego * v_ego) / (2.0 * d_apex)
    a_proposed = max(A_CURVE_DECEL_FLOOR, min(0.0, a_proposed))
  else:
    a_proposed = 0.0

  # Eligibility/confidence gating.
  confidence = 0.50
  if abs(curvature_peak) > 2.0 * KAPPA_NOISE_FLOOR:
    confidence += 0.25
  if len(xs) >= 16:
    confidence += 0.15
  if s_curve or compound_curve:
    confidence += 0.10
  confidence = min(1.0, confidence)

  low_speed_block = v_ego < LOW_SPEED_MIN_M_S
  active = abs(curvature_peak) > KAPPA_NOISE_FLOOR and not low_speed_block
  eligible = active and confidence >= 0.45

  if low_speed_block:
    geometry_block = "low_speed"
  elif not active:
    geometry_block = "below_noise_floor"
  elif not eligible:
    geometry_block = "low_confidence"
  else:
    geometry_block = ""
  block_reason = _combine_block_reasons(geometry_block, traffic_block_reason)

  return replace(
    base,
    eligible=eligible,
    active=active,
    confidence=round(confidence, 3),
    phase=phase,
    curvature_now=round(curvature_now, 6),
    curvature_peak=round(curvature_peak, 6),
    curvature_sign=curvature_sign,
    distance_to_curve=round(distance_to_curve, 2),
    distance_to_apex=round(distance_to_apex, 2),
    v_curve_cap_proposed=round(v_cap, 2),
    a_curve_cap_proposed=round(a_proposed, 3),
    suppress_accel=suppress_accel,
    traffic_block_reason=traffic_block_reason,
    s_curve=s_curve,
    compound_curve=compound_curve,
    block_reason=block_reason,
  )
