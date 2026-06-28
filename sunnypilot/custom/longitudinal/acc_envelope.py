"""ACC envelope: advisory/telemetry guard for lead-follow comfort.

This module estimates whether the current acceleration target keeps the ego vehicle inside
a comfortable gap/time-to-collision envelope relative to a lead. It is intentionally
advisory: the computed `allowed_a_target` is exposed as telemetry/debug fields and may be
used downstream for smoothing/telemetry, but it is NOT a hard safety cap. Binding emergency
braking and collision-avoidance remain the responsibility of the upstream lead-follow
controller and the vehicle's safety limits.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


MIN_GAP_M = 6.0
TIME_GAP_S = 1.5
TTC_MIN_S = 3.0
REQUIRED_DECEL_COMFORT_LIMIT = 1.0  # positive braking magnitude, m/s^2
MAX_ACCEL_JERK = 1.0                # m/s^3
MAX_DECEL_JERK = -2.0               # m/s^3
EPS_GAP_M = 0.1


@dataclass(frozen=True)
class AccEnvelopeInputs:
  v_ego: float
  candidate_a_target: float
  previous_a_target: float
  dt: float
  openpilot_longitudinal_control: bool = True
  has_lead: bool = False
  lead_d_rel: float = 0.0
  lead_v_rel: float = 0.0
  lead_v_lead: float = 0.0
  lead_a_lead_k: float = 0.0
  lead_kinematics_valid: bool = True
  model_stale: bool = False
  model_progress_candidate: bool = False
  lead_compression_candidate: bool = False
  radar_stale: bool = False
  lead_required: bool = False
  min_gap_m: float = MIN_GAP_M
  time_gap_s: float = TIME_GAP_S
  ttc_min_s: float = TTC_MIN_S
  required_decel_limit: float = REQUIRED_DECEL_COMFORT_LIMIT
  max_accel_jerk: float = MAX_ACCEL_JERK
  max_decel_jerk: float = MAX_DECEL_JERK


@dataclass(frozen=True)
class AccEnvelopeResult:
  # All fields are advisory/telemetry. The envelope never acts as a hard safety cap;
  # downstream code may smooth toward allowed_a_target but must not treat it as binding.
  active: bool
  would_cap: bool
  cap_reasons: tuple[str, ...] = field(default_factory=tuple)
  allowed_a_target: float = 0.0
  delta_a: float = 0.0
  desired_gap: float = 0.0
  time_gap: float = math.inf
  ttc: float = math.inf
  usable_stopping_gap: float = 0.0
  required_stopping_decel: float = 0.0
  closing_speed_decel: float = 0.0
  jerk_limited_a_target: float = 0.0

  def debug_dict(self) -> dict[str, float | bool | str]:
    return {
      "acc_envelope_active": self.active,
      "acc_envelope_would_cap": self.would_cap,
      "acc_envelope_cap_reason": ",".join(self.cap_reasons),
      "acc_envelope_allowed_a_target": self.allowed_a_target,
      "acc_envelope_delta_a": self.delta_a,
      "acc_envelope_desired_gap": self.desired_gap,
      "acc_envelope_time_gap": self.time_gap,
      "acc_envelope_ttc": self.ttc,
      "acc_envelope_usable_stopping_gap": self.usable_stopping_gap,
      "acc_envelope_required_stopping_decel": self.required_stopping_decel,
      "acc_envelope_closing_speed_decel": self.closing_speed_decel,
      "acc_envelope_jerk_limited_a_target": self.jerk_limited_a_target,
    }


def evaluate_acc_envelope(inp: AccEnvelopeInputs) -> AccEnvelopeResult:
  finite_values = (
    inp.v_ego, inp.candidate_a_target, inp.previous_a_target, inp.dt,
    inp.lead_d_rel, inp.lead_v_rel, inp.lead_v_lead, inp.lead_a_lead_k,
  )
  if not all(_finite(v) for v in finite_values) or inp.dt <= 0.0:
    return AccEnvelopeResult(active=False, would_cap=True, cap_reasons=("invalid_data",), allowed_a_target=0.0)

  candidate_a = float(inp.candidate_a_target)
  previous_a = float(inp.previous_a_target)
  jerk_limited = _jerk_limit(candidate_a, previous_a, float(inp.dt), float(inp.max_decel_jerk), float(inp.max_accel_jerk))

  reasons: list[str] = []
  if not inp.openpilot_longitudinal_control:
    reasons.append("stock_longitudinal_control")

  desired_gap = max(float(inp.min_gap_m), float(inp.time_gap_s) * max(0.0, float(inp.v_ego)))
  time_gap = math.inf
  ttc = math.inf
  usable_gap = 0.0
  required_decel = 0.0
  closing_decel = 0.0

  if inp.has_lead:
    if not inp.lead_kinematics_valid:
      reasons.append("invalid_lead_kinematics")
    d_rel = float(inp.lead_d_rel)
    v_rel = float(inp.lead_v_rel)
    if d_rel <= 0.0:
      reasons.append("invalid_lead_distance")
    else:
      time_gap = d_rel / max(float(inp.v_ego), 0.1)
      usable_gap = d_rel - desired_gap
      closing = max(0.0, -v_rel)
      ttc = d_rel / closing if closing > 1e-3 else math.inf
      required_decel = closing * closing / (2.0 * max(usable_gap, EPS_GAP_M)) if closing > 0.0 else 0.0
      closing_decel = required_decel
      if d_rel < desired_gap:
        reasons.append("inside_time_gap")
      if ttc < float(inp.ttc_min_s):
        reasons.append("ttc_low")
      if required_decel > float(inp.required_decel_limit):
        reasons.append("closing_decel_high")

  if inp.model_stale and inp.model_progress_candidate:
    reasons.append("model_stale_blocks_model_progress")
  if inp.radar_stale and inp.lead_required:
    reasons.append("radar_stale_blocks_lead_progress")
  if jerk_limited < candidate_a - 1e-6:
    reasons.append("jerk_limited")

  allowed_a = min(candidate_a, jerk_limited)
  # True collision-risk / invalid-lead reasons are always binding.
  binding_risks = {"ttc_low", "invalid_lead_kinematics", "invalid_lead_distance"}
  if inp.has_lead and bool(binding_risks & set(reasons)):
    allowed_a = min(allowed_a, 0.0, -required_decel)
  elif inp.has_lead and "closing_decel_high" in reasons:
    # For controlled compression candidates, the policy already gates both desired-gap
    # and collision-buffer decel; closing_decel_high alone is not allowed to harden here.
    if not inp.lead_compression_candidate:
      allowed_a = min(allowed_a, 0.0, -required_decel)
  elif inp.has_lead and "inside_time_gap" in reasons:
    if not inp.lead_compression_candidate:
      allowed_a = min(allowed_a, 0.0, -required_decel)
    # For lead_gap_compression candidates, the inside_time_gap reason is still recorded for
    # telemetry, but the mild compression target is allowed to bind instead of hardening to
    # the raw kinematic -required_decel.

  would_cap = bool(allowed_a < candidate_a - 1e-6 or reasons)
  return AccEnvelopeResult(
    active=True,
    would_cap=would_cap,
    cap_reasons=tuple(reasons),
    allowed_a_target=float(allowed_a),
    delta_a=float(allowed_a - candidate_a),
    desired_gap=float(desired_gap),
    time_gap=float(time_gap),
    ttc=float(ttc),
    usable_stopping_gap=float(usable_gap),
    required_stopping_decel=float(required_decel),
    closing_speed_decel=float(closing_decel),
    jerk_limited_a_target=float(jerk_limited),
  )


def _finite(value: float) -> bool:
  try:
    return math.isfinite(float(value))
  except (TypeError, ValueError):
    return False


def _jerk_limit(candidate_a: float, previous_a: float, dt: float, decel_jerk: float, accel_jerk: float) -> float:
  lo = previous_a + decel_jerk * dt
  hi = previous_a + accel_jerk * dt
  return min(max(candidate_a, lo), hi)
