"""Shadow-only scenario attribution for the custom-2.0 longitudinal policy.

This module is intentionally pure and produces only telemetry. It must never change
actuation, a_target, stop commitment, or candidate selection. The goal is to label the
current driving situation so future phases can reason about which longitudinal behaviors
are appropriate, while today we only observe safely from the side.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

MODE_OFF = "off"
MODE_SHADOW = "shadow"


@dataclass(frozen=True)
class ScenarioContextResult:
  mode: str = MODE_OFF
  effective_mode: str = MODE_OFF
  apply_supported: bool = False
  scenario: str = "mode_off"
  active: bool = False
  confidence: float = 0.0
  allowed_effect: str = "none"
  current_effect: str = "none"
  road_grade: str = "flat"
  reason: str = "mode_off"

  def debug_dict(self) -> dict[str, Any]:
    prefix = "scenario_context"
    return {
      f"{prefix}_mode": self.mode,
      f"{prefix}_effective_mode": self.effective_mode,
      f"{prefix}_apply_supported": self.apply_supported,
      f"{prefix}_scenario": self.scenario,
      f"{prefix}_active": self.active,
      f"{prefix}_confidence": self.confidence,
      f"{prefix}_allowed_effect": self.allowed_effect,
      f"{prefix}_current_effect": self.current_effect,
      f"{prefix}_road_grade": self.road_grade,
      f"{prefix}_reason": self.reason,
    }


def _f(value: Any, default: float = 0.0) -> float:
  try:
    v = float(value)  # type: ignore[arg-type]
  except (TypeError, ValueError):
    return default
  return v if math.isfinite(v) else default


def _active_status(leads: Any, idx: int) -> bool:
  if isinstance(leads, (tuple, list)):
    if idx < len(leads):
      lead = leads[idx]
      return lead is not None and bool(getattr(lead, "status", False))
    return False
  if idx == 0 and leads is not None:
    return bool(getattr(leads, "status", False))
  return False


def _lead_attr(leads: Any, idx: int, attr: str, default: Any = 0.0) -> Any:
  if isinstance(leads, (tuple, list)):
    lead = leads[idx] if idx < len(leads) else None
  else:
    lead = leads if idx == 0 else None
  if lead is None:
    return default
  return getattr(lead, attr, default)


def _road_grade(accel_coast: float) -> str:
  # accel_coast ~= sin(pitch) * -5.65 - 0.3 (see wiring._coast_accel).
  # A flat road sits near -0.3 m/s^2; downhill assists become positive, uphill fights.
  if accel_coast > 0.1:
    return "downhill"
  if accel_coast < -0.6:
    return "uphill"
  return "flat"


def predict_scenario_context(
    mode: Any, *,
    v_ego: Any,
    a_ego: Any,
    accel_coast: Any,
    standstill: Any,
    steering_angle_deg: Any,
    steering_torque: Any,
    leads: Any,
    model_should_stop: Any,
    model_stop_distance: Any,
    speed_limit_active: Any,
    curve_active: Any,
    gas_pressed: Any,
    brake_pressed: Any,
) -> ScenarioContextResult:
  """Classify the current longitudinal situation using only available signals.

  The returned ``allowed_effect`` is a future-facing hint; in shadow mode the current
  effect is always ``none``. The classifier is deliberately conservative and fallible:
  it prefers ``unknown`` over inventing a high-confidence label from marginal evidence.
  """
  mode_s = str(mode or "").strip().lower()
  if mode_s not in (MODE_OFF, MODE_SHADOW):
    mode_s = MODE_OFF
  if mode_s == MODE_OFF:
    return ScenarioContextResult(mode=MODE_OFF, effective_mode=MODE_OFF)

  apply_supported = False  # this module is intentionally shadow-only

  v = _f(v_ego)
  a = _f(a_ego)
  coast = _f(accel_coast)
  stopped = bool(standstill)
  steering_deg = abs(_f(steering_angle_deg))
  steering_torque_f = abs(_f(steering_torque))
  has_lead = _active_status(leads, 0)
  lead_d = _f(_lead_attr(leads, 0, "dRel")) if has_lead else float("inf")
  lead_v = _f(_lead_attr(leads, 0, "vLead", _lead_attr(leads, 0, "vLeadK", 0.0))) if has_lead else 0.0
  lead_v_rel = _f(_lead_attr(leads, 0, "vRel", lead_v - v)) if has_lead else 0.0
  lead_a = _f(_lead_attr(leads, 0, "aLeadK", 0.0)) if has_lead else 0.0
  model_stop = bool(model_should_stop)
  stop_dist = _f(model_stop_distance, default=float("inf"))
  speed_limit = bool(speed_limit_active)
  curve = bool(curve_active)
  gas = bool(gas_pressed)
  brake = bool(brake_pressed)
  grade = _road_grade(coast)

  def result(label: str, confidence: float, effect: str, reason: str) -> ScenarioContextResult:
    return ScenarioContextResult(
      mode=mode_s,
      effective_mode=mode_s,
      apply_supported=apply_supported,
      scenario=label,
      active=True,
      confidence=max(0.0, min(1.0, confidence)),
      allowed_effect=effect,
      current_effect="none",  # shadow-only today: never an actuation effect
      road_grade=grade,
      reason=reason,
    )

  # 1. Driver/safety layers always take precedence and are marked effect=none so they cannot
  #    be interpreted as authorizing anything by downstream consumers.
  if gas or brake:
    return result("driver_override", 0.9, "none", "driver_pedal_pressed")

  if stopped:
    return result("standstill", 0.9, "none", "v_ego_and_standstill_low")

  # 2. Commit-class stop evidence (today shadow-only; no actuation).
  if model_stop and math.isfinite(stop_dist) and stop_dist < 60.0 and v > 0.5:
    conf = 0.75
    if stop_dist < 25.0:
      conf = 0.90
    return result("approach_stop", conf, "stop_commit", "model_stop_within_horizon")

  # 3. Lead-relative situations.
  if has_lead:
    follow_gap = max(6.0, 1.5 * max(0.0, v))
    if v < 2.0 and lead_d < 8.0 and lead_v < 1.0:
      return result("stop_and_go", 0.80, "stop_commit", "lead_close_and_stopped_at_low_speed")
    if lead_v_rel < -1.0 or (lead_v_rel < -0.5 and lead_d < follow_gap):
      conf = 0.70
      if lead_v_rel < -2.5 or lead_a < -1.5:
        conf = 0.85
      return result("closing_lead", conf, "restrict_only", "lead_closing_relative_motion")
    if lead_v_rel > 1.0 and v < lead_v and lead_d > 10.0:
      return result("lead_pullaway", 0.75, "progress_with_guard", "lead_accelerating_ahead")
    return result("lead_follow", 0.70, "shadow_only", "lead_present_not_closing")

  # 4. Road geometry / speed advisories.
  if curve and v > 5.0:
    return result("curve_approach", 0.65, "restrict_only", "curve_advisory_active")

  if speed_limit and v > 5.0:
    return result("speed_limit_drop", 0.65, "restrict_only", "speed_limit_advisory_active")

  # 5. Turning without a lead outranks grade progress labels; it is treated as a restrict class
  #    for future lane-keeping / progress guarding.
  if steering_deg > 15.0 or steering_torque_f > 60.0:
    return result("turning", 0.60, "restrict_only", "significant_steering_input")

  # 6. Grade-based coasting / recovery hints.
  if grade == "downhill" and v > 5.0:
    return result("downhill_coast", 0.55, "progress_with_guard", "negative_pitch_assisting")

  if grade == "uphill" and v > 5.0:
    return result("uphill_recovery", 0.55, "progress_with_guard", "positive_pitch_resisting")

  # 7. Open road when nothing else is active and there is speed.
  if v > 8.0:
    return result("open_road", 0.60, "progress_with_guard", "no_hazards_or_advisories")

  # Marginal evidence: be honest.
  return result("unknown", 0.30, "none", "insufficient_discriminating_evidence")
