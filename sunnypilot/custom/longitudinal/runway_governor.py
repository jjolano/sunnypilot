"""Runway comfort governor (Phase 5) — coast-first for long-runway slowdowns.

Serves two backlog items with one mechanism:
- E2E runway comfort governor (legacy spec 2026-05-03): for a model slowdown with a long
  runway, prefer coast / light braking over the model's raw (often abrupt) decel; let the
  raw model through only when the runway is short enough that it knows best.
- Map-curve soft advance (legacy spec 2026-05-05): smoother curve entry by applying the
  same coast-first shaping to a curve-speed target ahead.

Both are "reach v_target by distance d" constraints, so they share the coast horizon:
long runway -> coast at the natural decel (gentler than the model); short runway -> the raw
model accel binds.
"""
from __future__ import annotations

from openpilot.sunnypilot.custom.longitudinal.coast_horizon import (
  CoastAction,
  CoastHorizonInputs,
  coast_horizon,
)


def runway_comfort_governor(v_ego: float, v_target: float, distance: float, raw_model_accel: float,
                            coast_decel: float, comfort_brake_decel: float = -1.5) -> float:
  """Return the comfort-shaped accel. Long runway -> the gentler of (coast, raw model);
  short runway (coast can't bleed enough) -> the raw model accel."""
  r = coast_horizon(CoastHorizonInputs(
    v_ego=v_ego, v_target=v_target, distance_to_constraint=distance,
    accel_coast=coast_decel, comfort_brake_decel=comfort_brake_decel,
  ))
  raw = float(raw_model_accel)
  if r.action is CoastAction.BRAKE:
    # short runway: the model has authority (don't be gentler than it wants)
    return min(raw, r.recommended_accel) if raw < 0 else r.recommended_accel
  # long runway / lift window: prefer the gentler command (coast), never brake harder than raw
  # demands. r.recommended_accel is 0 (cruise, too early) or the natural coast decel.
  return max(raw, r.recommended_accel)


def map_curve_soft_advance(v_ego: float, curve_speed: float, distance_to_curve: float,
                           raw_curve_accel: float, coast_decel: float,
                           comfort_brake_decel: float = -1.5) -> float:
  """Map-curve soft advance = the runway comfort governor applied to a curve-speed target."""
  return runway_comfort_governor(v_ego, curve_speed, distance_to_curve, raw_curve_accel,
                                 coast_decel, comfort_brake_decel)
