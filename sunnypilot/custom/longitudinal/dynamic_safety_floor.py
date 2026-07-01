"""Dynamic safety-floor shadow telemetry for custom longitudinal.

Computes a conservative safety-distance risk signal for route-log telemetry only.
No actuation, no MPC change. Distances are intentionally conservative: bad or missing
inputs fall back to the current fork distance model rather than producing a shorter
"proposed" distance.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

# Current fork MPC distance constants (from long_mpc.py). These define the product
# baseline distance model; Phase 2 only compares against them, never changes them.
_COMFORT_BRAKE_NOMINAL = 2.5
_STOP_DISTANCE = 6.0
_MOVING_GAP = 1.25
_GAP_FADE_V = 4.0

# Phase 2 dynamic-floor constants.
_LATENCY_S = 0.35
_BUFFER_M = 1.5
_STANDSTILL_MIN_M = 4.0
_COMFORT_BRAKE_MIN = 0.5

# Avoid sqrt of tiny negative due to rounding.
_MIN_FRICTION_SQUARE = 0.0


def _f(value: Any, default: float = 0.0) -> float:
  try:
    v = float(value)
  except (TypeError, ValueError):
    return default
  return v if math.isfinite(v) else default


def _finite_or_none(value: Any) -> float | None:
  try:
    v = float(value)
  except (TypeError, ValueError):
    return None
  return v if math.isfinite(v) else None


def follow_offset(v_ego: float) -> float:
  """Current fork moving-gap fade model."""
  v = max(0.0, v_ego)
  return _MOVING_GAP + (_STOP_DISTANCE - _MOVING_GAP) / (1.0 + (v / _GAP_FADE_V) ** 2)


def get_safe_obstacle_distance(v_ego: float, t_follow: float) -> float:
  """Current fork safe obstacle distance model."""
  v = max(0.0, v_ego)
  return (v * v) / (2.0 * _COMFORT_BRAKE_NOMINAL) + t_follow * v + follow_offset(v)


def _compute_dynamic_floor(v_ego: float) -> float:
  """Dynamic floor from latency budget and standstill minimum."""
  v = max(0.0, v_ego)
  kinematic_floor = v * _LATENCY_S + _BUFFER_M
  return max(_STANDSTILL_MIN_M, kinematic_floor)


def _effective_comfort_brake(a_lat: float | None, pitch: float | None) -> tuple[float, dict[str, Any]]:
  """Conservative comfort brake with lateral accel and downhill penalty only.

  Returns the effective comfort brake value and a small debug bundle.
  Missing inputs are safe defaults (no distance shortening); block_reason explains why.
  """
  block_reasons: list[str] = []

  if a_lat is None:
    a_lat_used = 0.0
    block_reasons.append("lat_accel_unavailable")
  else:
    a_lat_f = _finite_or_none(a_lat)
    if a_lat_f is None:
      a_lat_used = 0.0
      block_reasons.append("lat_accel_invalid")
    else:
      a_lat_used = a_lat_f

  if pitch is None:
    pitch_used = 0.0
    block_reasons.append("pitch_unavailable")
  else:
    pitch_f = _finite_or_none(pitch)
    if pitch_f is None:
      pitch_used = 0.0
      block_reasons.append("pitch_invalid")
    else:
      pitch_used = pitch_f

  # Lateral accel reduces available longitudinal friction.
  friction_avail = math.sqrt(max(_MIN_FRICTION_SQUARE, _COMFORT_BRAKE_NOMINAL ** 2 - a_lat_used ** 2))

  # Grade component: downhill (negative pitch) reduces braking, uphill does not help in Phase 2.
  grade_comp = 9.81 * math.sin(pitch_used)
  comfort_eff = friction_avail + min(0.0, grade_comp)
  comfort_eff = min(max(comfort_eff, _COMFORT_BRAKE_MIN), _COMFORT_BRAKE_NOMINAL)

  return comfort_eff, {
    "lat_accel": a_lat_used,
    "pitch": pitch_used,
    "block_reason": "; ".join(block_reasons),
  }


@dataclass(frozen=True)
class DynamicSafetyFloorResult:
  active: bool
  block_reason: str
  current_safe_distance: float
  proposed_safe_distance: float
  delta_safe_distance: float
  dynamic_floor_value: float
  kinematic_floor_violation: bool
  comfort_brake_effective: float
  latency_s: float
  lat_accel: float
  pitch: float


def compute_dynamic_safety_floor(
  v_ego: float,
  t_follow: float,
  *,
  lead_d_rel: float | None = None,
  a_lat: float | None = None,
  pitch: float | None = None,
) -> DynamicSafetyFloorResult:
  """Compute shadow telemetry for dynamic safety floor.

  The returned ``proposed_safe_distance`` is never shorter than the current fork
  distance model, even when inputs are invalid, so telemetry cannot imply an
  optimistic active distance.
  """
  v = max(0.0, _f(v_ego, default=0.0))
  t_follow_f = max(0.0, _f(t_follow, default=1.5))

  current_distance = get_safe_obstacle_distance(v, t_follow_f)
  dynamic_floor = _compute_dynamic_floor(v)
  comfort_eff, brake_debug = _effective_comfort_brake(a_lat, pitch)

  proposed_distance = (v * v) / (2.0 * comfort_eff) + t_follow_f * v + dynamic_floor
  # Fail-closed: never report a proposed distance shorter than the current fork model.
  proposed_distance = max(proposed_distance, current_distance)

  delta = proposed_distance - current_distance

  violation = False
  if lead_d_rel is not None:
    lead_d = _f(lead_d_rel, default=float("inf"))
    if math.isfinite(lead_d) and lead_d < dynamic_floor:
      violation = True

  block_reason = brake_debug["block_reason"]
  if not block_reason and proposed_distance <= current_distance + 1e-9:
    block_reason = "proposed_equals_current"

  return DynamicSafetyFloorResult(
    active=True,
    block_reason=block_reason,
    current_safe_distance=current_distance,
    proposed_safe_distance=proposed_distance,
    delta_safe_distance=delta,
    dynamic_floor_value=dynamic_floor,
    kinematic_floor_violation=violation,
    comfort_brake_effective=comfort_eff,
    latency_s=_LATENCY_S,
    lat_accel=brake_debug["lat_accel"],
    pitch=brake_debug["pitch"],
  )


def debug_dict(result: DynamicSafetyFloorResult) -> dict[str, Any]:
  """Return a debug dict with the required dynamic_safety_floor_ prefix."""
  return {
    "dynamic_safety_floor_active": result.active,
    "dynamic_safety_floor_block_reason": result.block_reason,
    "dynamic_safety_floor_current_safe_distance": result.current_safe_distance,
    "dynamic_safety_floor_proposed_safe_distance": result.proposed_safe_distance,
    "dynamic_safety_floor_delta_safe_distance": result.delta_safe_distance,
    "dynamic_safety_floor_dynamic_floor_value": result.dynamic_floor_value,
    "dynamic_safety_floor_kinematic_floor_violation": result.kinematic_floor_violation,
    "dynamic_safety_floor_comfort_brake_effective": result.comfort_brake_effective,
    "dynamic_safety_floor_latency_s": result.latency_s,
    "dynamic_safety_floor_lat_accel": result.lat_accel,
    "dynamic_safety_floor_pitch": result.pitch,
  }
