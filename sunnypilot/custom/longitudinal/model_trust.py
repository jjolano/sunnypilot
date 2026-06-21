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

# StopTrustLearner: learn how much to trust the upstream model stop from driver disagreement.
STOP_TRUST_INITIAL = 0.8
STOP_TRUST_MIN = 0.2
STOP_TRUST_MAX = 0.95
STOP_TRUST_DISAGREE_RATE = 0.5   # confidence/s drop when the driver countermands a model stop
STOP_TRUST_AGREE_RATE = 0.05     # confidence/s recovery when the driver accepts it


@dataclass(frozen=True)
class ModelStopTrustResult:
  should_stop: bool
  desired_accel: float   # trust-scaled: gentle at low trust, full model decel at high trust
  trust: float
  reason: str


def _clip(v: float, lo: float, hi: float) -> float:
  return max(lo, min(hi, v))


def gate_model_stop(model_should_stop: bool, model_desired_accel: float, stop_prob: float,
                    has_radar_lead: bool = False, lead_v_rel: float = 0.0,
                    model_stale: bool = False) -> ModelStopTrustResult:
  """Trust-gate the model's stop/slowdown.

  ``stop_prob`` is the model's confidence in the stop (modelV2). ``lead_v_rel`` < 0 means a
  radar lead closing (physical corroboration of a slowdown)."""
  stop_prob = _clip(float(stop_prob), 0.0, 1.0)
  model_decel = min(0.0, float(model_desired_accel))

  # Model says go (no slowdown): nothing to gate here. Any real hazard (radar lead, map,
  # curve) binds independently in the decision core, so we never relax safety here.
  if not model_should_stop and model_decel >= 0.0:
    return ModelStopTrustResult(False, float(model_desired_accel), 1.0, "model_clear")

  # A stale model stop is semantic evidence that arrived too late. Do not commit a stop or
  # honor hard model decel from it; radar/MPC lead physics remains a separate safety floor.
  if model_stale:
    return ModelStopTrustResult(False, GENTLE_CAUTION_DECEL, 0.0, "model_stale")

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


class StopTrustLearner:
  """Learn the confidence to feed gate_model_stop from real driver disagreement.

  Rather than guessing a model stop probability, we take upstream's verified model stop
  (modelV2.action.shouldStop) at face value and adjust how much to trust it from how the
  driver reacts: a driver who countermands a model stop (gas, or disengages) during the stop
  is telling us the stop was wrong -> drop confidence fast; a driver who lets it happen agrees
  -> recover slowly. Over a drive this softens repeatedly-false model stops while keeping the
  ones the driver accepts. (Session-scoped; param persistence is a later enhancement.)"""

  def __init__(self, initial: float = STOP_TRUST_INITIAL):
    self.confidence = _clip(float(initial), STOP_TRUST_MIN, STOP_TRUST_MAX)

  def update(self, model_should_stop: bool, driver_disagrees: bool, dt: float) -> float:
    if model_should_stop:
      rate = -STOP_TRUST_DISAGREE_RATE if driver_disagrees else STOP_TRUST_AGREE_RATE
      self.confidence = _clip(self.confidence + rate * max(0.0, float(dt)), STOP_TRUST_MIN, STOP_TRUST_MAX)
    return self.confidence
