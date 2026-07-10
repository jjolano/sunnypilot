"""Improved lead trajectory prediction (Phase 5, anti-quirk).

Addresses the three accuracy limitations flagged in the lead-math review of the ported
``lead_context.lead_prediction`` — all plausible sources of lead-related longitudinal quirks
when leaned on for anticipation:

1. Decay ``a_lead`` toward zero with ``aLeadTau`` (as the MPC does) instead of propagating the
   noisy double-derivative constant — phantom lead accel is the leading quirk suspect.
2. Add the ego-accel term to the gap projection (the old one held ego speed constant, so the
   projected gap was wrong during your own braking/accelerating transients).
3. Unify on the measured relative speed ``vRel`` for the linear gap term instead of
   recomputing ``v_lead - v_ego``.

Plus per-quantity confidence: ``accel_confidence`` (∈[0,1]) scales the a_lead contribution,
since aLeadK is far less trustworthy than d_rel/vRel — low confidence => the projection falls
back toward the conservative constant-velocity gap.

A braking lead is held at standstill once its predicted speed reaches zero: the displacement
integral freezes at the stop time instead of letting the closed form "reverse" the lead and
under-predict the gap (lead-math review finding, 2026-07-10).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

DEFAULT_HORIZON_T = (0.5, 1.0, 1.5, 2.0)
MIN_A_LEAD_TAU = 0.1


@dataclass(frozen=True)
class LeadPrediction:
  t: tuple[float, ...]
  gap: tuple[float, ...]        # predicted d_rel at each horizon time
  v_lead: tuple[float, ...]     # predicted lead absolute speed
  a_lead: tuple[float, ...]     # decayed lead accel
  valid: bool = True


def _lead_stop_time(v_lead: float, a0: float, tau: float) -> float | None:
  """Time at which the τ-decaying decel brings the lead to rest, or None if it never does.

  v(t) = v0 + a0·τ·(1 − e^(−t/τ)) reaches 0 only when the total available Δv (= a0·τ, a0 < 0)
  exceeds v0; then e^(−t_s/τ) = 1 + v0/(a0·τ).
  """
  if a0 >= 0.0:
    return None
  if v_lead <= 0.0:
    return 0.0
  ratio = 1.0 + v_lead / (a0 * tau)
  if ratio <= 0.0:
    return None  # decaying decel runs out before the lead stops
  return -tau * math.log(ratio)


def predict_lead_trajectory(d_rel: float, v_rel: float, v_lead: float, a_lead: float,
                            a_lead_tau: float, v_ego: float, a_ego: float = 0.0,
                            accel_confidence: float = 1.0, valid: bool = True,
                            horizon_t: tuple[float, ...] = DEFAULT_HORIZON_T) -> LeadPrediction:
  tau = max(float(a_lead_tau), MIN_A_LEAD_TAU)
  conf = max(0.0, min(1.0, float(accel_confidence)))
  a0 = float(a_lead) * conf  # discount the noisy lead accel by its confidence
  v_lead = float(v_lead)
  v_rel = float(v_rel)
  v_ego = float(v_ego)
  a_ego = float(a_ego)
  t_stop = _lead_stop_time(v_lead, a0, tau)

  gaps: list[float] = []
  v_leads: list[float] = []
  a_leads: list[float] = []
  for t in horizon_t:
    if t_stop is not None and t >= t_stop:
      # Lead at rest: freeze its displacement at t_stop. Relative to the constant-v_lead
      # baseline the linear vRel term assumes, the correction is the accel displacement up
      # to t_stop minus the baseline motion the stopped lead no longer makes.
      decay_s = math.exp(-t_stop / tau)
      lead_disp_accel = a0 * tau * (t_stop - tau * (1.0 - decay_s)) - v_lead * (t - t_stop)
      v_lead_t = 0.0
      a_t = 0.0
    else:
      decay = math.exp(-t / tau)
      a_t = a0 * decay                                        # lead accel decays toward 0 (aLeadTau)
      v_lead_t = max(0.0, v_lead + a0 * tau * (1.0 - decay))  # integral of a0*exp(-t/tau)
      lead_disp_accel = a0 * tau * (t - tau * (1.0 - decay))  # ∫∫ of the decaying accel
    ego_disp_accel = 0.5 * a_ego * t * t                      # ego-accel term (was missing)
    # measured vRel for the linear closing; explicit accel displacements on top
    gap = max(0.0, d_rel + v_rel * t + lead_disp_accel - ego_disp_accel)
    gaps.append(float(gap))
    v_leads.append(float(v_lead_t))
    a_leads.append(float(a_t))
  return LeadPrediction(tuple(horizon_t), tuple(gaps), tuple(v_leads), tuple(a_leads), valid)
