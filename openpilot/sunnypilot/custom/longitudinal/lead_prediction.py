"""Improved lead trajectory prediction (Phase 5, anti-quirk).

Addresses the three accuracy limitations flagged in the lead-math review of the ported
``lead_context.lead_prediction`` — all plausible sources of lead-related longitudinal quirks
when leaned on for anticipation:

1. Decay ``a_lead`` with the MPC's Gaussian ``aLeadTau`` rate instead of propagating the noisy
   double-derivative constant — phantom lead accel is the leading quirk suspect.
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
from statistics import NormalDist

DEFAULT_HORIZON_T = (0.5, 1.0, 1.5, 2.0)
DEFAULT_A_LEAD_TAU = 1.5
MIN_A_LEAD_TAU = 0.1
MIN_PULLAWAY_A_LEAD_TAU = DEFAULT_A_LEAD_TAU
MIN_A_LEAD = -10.0
MAX_A_LEAD = 2.0  # long_mpc's optimistic pullaway ceiling


@dataclass(frozen=True)
class LeadPrediction:
  t: tuple[float, ...]
  gap: tuple[float, ...]        # predicted d_rel at each horizon time
  v_lead: tuple[float, ...]     # predicted lead absolute speed
  a_lead: tuple[float, ...]     # decayed lead accel
  valid: bool = True


def _velocity_delta(a0: float, decay_rate: float, t: float) -> float:
  scale = math.sqrt(math.pi / (2.0 * decay_rate))
  return a0 * scale * math.erf(math.sqrt(decay_rate / 2.0) * t)


def _displacement_delta(a0: float, decay_rate: float, t: float) -> float:
  scale = math.sqrt(math.pi / (2.0 * decay_rate))
  decay = math.exp(-decay_rate * t * t / 2.0)
  return a0 * (t * scale * math.erf(math.sqrt(decay_rate / 2.0) * t) + (decay - 1.0) / decay_rate)


def _lead_stop_time(v_lead: float, a0: float, decay_rate: float) -> float | None:
  """Time when Gaussian-decaying lead decel reaches rest, or None if it cannot."""
  if a0 >= 0.0:
    return None
  if v_lead <= 0.0:
    return 0.0
  total_delta = a0 * math.sqrt(math.pi / (2.0 * decay_rate))
  ratio = -v_lead / total_delta
  if ratio >= 1.0:
    return None
  return NormalDist().inv_cdf((ratio + 1.0) / 2.0) / math.sqrt(decay_rate)


def predict_lead_trajectory(d_rel: float, v_rel: float, v_lead: float, a_lead: float,
                            a_lead_tau: float, v_ego: float, a_ego: float = 0.0,
                            accel_confidence: float = 1.0, valid: bool = True,
                            horizon_t: tuple[float, ...] = DEFAULT_HORIZON_T) -> LeadPrediction:
  conf = max(0.0, min(1.0, float(accel_confidence)))
  a0 = max(MIN_A_LEAD, min(MAX_A_LEAD, float(a_lead))) * conf
  decay_floor = MIN_PULLAWAY_A_LEAD_TAU if a0 > 0.0 else MIN_A_LEAD_TAU
  decay_rate = max(float(a_lead_tau), decay_floor)
  v_lead = max(0.0, float(v_lead))
  v_rel = float(v_rel)
  v_ego = float(v_ego)
  a_ego = float(a_ego)
  t_stop = _lead_stop_time(v_lead, a0, decay_rate)
  t_ego_stop = max(0.0, v_ego) / -a_ego if a_ego < 0.0 else None

  gaps: list[float] = []
  v_leads: list[float] = []
  a_leads: list[float] = []
  for t in horizon_t:
    if t_stop is not None and t >= t_stop:
      # Lead at rest: freeze its displacement at t_stop. Relative to the constant-v_lead
      # baseline the linear vRel term assumes, the correction is the accel displacement up
      # to t_stop minus the baseline motion the stopped lead no longer makes.
      lead_disp_accel = _displacement_delta(a0, decay_rate, t_stop) - v_lead * (t - t_stop)
      v_lead_t = 0.0
      a_t = 0.0
    else:
      a_t = a0 * math.exp(-decay_rate * t * t / 2.0)
      v_lead_t = max(0.0, v_lead + _velocity_delta(a0, decay_rate, t))
      lead_disp_accel = _displacement_delta(a0, decay_rate, t)
    if t_ego_stop is not None and t >= t_ego_stop:
      ego_disp_accel = v_ego * t_ego_stop + 0.5 * a_ego * t_ego_stop * t_ego_stop - v_ego * t
    else:
      ego_disp_accel = 0.5 * a_ego * t * t
    # measured vRel for the linear closing; explicit accel displacements on top
    gap = max(0.0, d_rel + v_rel * t + lead_disp_accel - ego_disp_accel)
    gaps.append(float(gap))
    v_leads.append(float(v_lead_t))
    a_leads.append(float(a_t))
  return LeadPrediction(tuple(horizon_t), tuple(gaps), tuple(v_leads), tuple(a_leads), valid)
