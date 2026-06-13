"""Longitudinal model-evidence trust gate (Phase 5).

The longitudinal analog of the lateral ``model_path_processor`` quality gating: the model's
stop / slowdown is treated as ONE evidence source, not ground truth. Trust is earned
per-signal and applied asymmetrically — caution is honored cheaply, relaxation expensively —
so a low-confidence or contradicted model prediction degrades gracefully toward a
conservative precautionary decel instead of being obeyed or ignored outright. This directly
targets the "over-trust the model in low-quality regimes (occlusion / flicker / glare)"
failure that likely accounts for some unattributed quirks.

Principles (from the trajectory/trust review):
- Cross-validate against non-learned signals (radar d_rel/v_rel, kinematics).
- Disagreement => lower trust, fall back toward the conservative envelope.
- Trust caution cheaply; require corroboration to honor a full hard stop.
- Blend by confidence (no hard model-on/off switching -> no discontinuity/oscillation).
- Relaxation can never override the reactive/MPC lead-follow floor (that binds in the
  decision core independently), so this gate only shapes the *caution* side.
"""
from __future__ import annotations

from dataclasses import dataclass

GENTLE_CAUTION_DECEL = -0.4     # precautionary decel for a low-confidence model slowdown
TRUST_FULL_STOP = 0.7           # stop_prob/trust above which a hard should_stop is honored
RADAR_CORROBORATION_TRUST = 0.85
LEAD_CLOSING_MIN = 0.5          # m/s relative closing to count as radar corroboration


@dataclass(frozen=True)
class ModelStopTrustResult:
  should_stop: bool
  desired_accel: float   # trust-scaled: gentle at low trust, full model decel at high trust
  trust: float
  reason: str


def _clip(v: float, lo: float, hi: float) -> float:
  return max(lo, min(hi, v))


def gate_model_stop(model_should_stop: bool, model_desired_accel: float, stop_prob: float,
                    has_radar_lead: bool = False, lead_v_rel: float = 0.0) -> ModelStopTrustResult:
  """Trust-gate the model's stop/slowdown.

  ``stop_prob`` is the model's confidence in the stop (modelV2). ``lead_v_rel`` < 0 means a
  radar lead closing (physical corroboration of a slowdown)."""
  stop_prob = _clip(float(stop_prob), 0.0, 1.0)
  model_decel = min(0.0, float(model_desired_accel))

  # Model says go (no slowdown): nothing to gate here. Any real hazard (radar lead, map,
  # curve) binds independently in the decision core, so we never relax safety here.
  if not model_should_stop and model_decel >= 0.0:
    return ModelStopTrustResult(False, float(model_desired_accel), 1.0, "model_clear")

  # Caution: the model wants to slow/stop. Earn trust from model confidence, raised by radar
  # corroboration (a closing radar lead physically agrees with slowing down).
  trust = stop_prob
  reason = "model_only"
  if has_radar_lead and float(lead_v_rel) < -LEAD_CLOSING_MIN:
    trust = max(trust, RADAR_CORROBORATION_TRUST)
    reason = "radar_corroborated"

  # Honor a full hard stop only when trusted; otherwise hold the stop flag back (still decel,
  # just not a committed stop) to avoid slamming on a flickery model stop.
  should_stop = bool(model_should_stop and trust >= TRUST_FULL_STOP)

  # Blend the commanded decel from a gentle precautionary value (low trust) to the full model
  # decel (high trust). Never command less caution than the model asks if it is fully trusted.
  desired_accel = GENTLE_CAUTION_DECEL + trust * (model_decel - GENTLE_CAUTION_DECEL)
  return ModelStopTrustResult(should_stop, float(desired_accel), float(trust), reason)
