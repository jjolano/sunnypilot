"""Dynamic follow-gap scheduler — bounded T_FOLLOW compression during a low-risk approach.

The engaged-corpus analysis closed the pre-MPC shaping line: steady-follow reactive braking is
the MPC genuinely restoring the fixed follow gap behind a slower lead (not an aLeadK artifact),
so the only honest lever on approach decel is the desired gap itself. This scheduler makes that
tradeoff explicit and bounded: during a confident, low-TTC-risk approach to a slower *moving*
lead it compresses T_FOLLOW toward ``T_FOLLOW_COMPRESSED``, letting the gap close smoothly
inside the MPC (jerk penalized in the solve) instead of braking early to hold the baseline gap;
it recovers to the personality baseline — faster than it compresses — the moment the context
degrades.

SAFETY — this reaches the MPC desired-gap parameter and can reduce headway. So it:
- never goes below ``T_FOLLOW_COMPRESSED`` (1.2 s) nor above the personality baseline;
- only applies with explicit planner safety context: long active, no driver override, no
  force-decel, enough ego speed, finite lead kinematics, a moving lead that is slower but not
  braking hard, no fast closing, no low TTC, and every status lead above hard distance floors;
- slews into compression at a bounded, demand-scaled rate and quickly back out, and snaps
  back to baseline whenever long control is inactive;
- is opt-in via ``DynamicFollowGapMode=apply``, runtime-gated by ``CustomLongitudinalEnabled``
  and the default-off ``AllowLongitudinalResearchActuation`` switch; shadow mode computes the
  would-be value for telemetry only, and any fault returns the baseline unchanged.

Validate via ``profile_lead_following`` replay (approach decel peak down, zero new close
approaches, headway recovers after the approach) before any default-on.
"""
from __future__ import annotations

import math
from typing import Any

T_FOLLOW_COMPRESSED = 1.2   # s; hard floor of the compressed follow gap
# Compression pace scales with the decel the approach would otherwise demand
# (closing^2 / 2*excess). A fixed slow rate meant warm-but-eligible approaches ended before
# compression arrived — exactly the ones where it would soften the peak — while long lazy
# approaches got full compression they didn't need.
COMPRESS_RATE = 0.05        # s of t_follow per s; slide rate at negligible approach demand
COMPRESS_RATE_MAX = 0.25    # s of t_follow per s; slide rate at/above COMPRESS_DEMAND_FULL
COMPRESS_DEMAND_FULL = 0.5  # m/s^2 required decel; comfort-band demand that earns the fast slide
RECOVER_RATE = 0.5          # s of t_follow per s; fast, safety-shaped recovery to baseline
# Route 290 stop-and-go ask: when the approach merely ends (lead stops closing / slows /
# ego drops below apply speed) snapping t_follow back at RECOVER_RATE makes the MPC brake
# to re-open the gap. Recover instead at the rate the lead physically opens it
# (d(time_gap)/dt ~ v_rel/v_ego), floored so a stalled queue still trickles back. Any
# safety-shaped ineligibility (braking/fast-closing/low-TTC/too-close lead, pedals,
# force-decel, disengage, non-finite) keeps the fast rate.
BENIGN_RECOVER_RATE_MIN = 0.02  # s of t_follow per s
MIN_V_EGO_FOR_APPLY = 5.0   # m/s; route 00000274: 8.0 disabled the scheduler for most city
                            # stop-and-go (mean 37 km/h), so moving-lead compression never engaged.
                            # 5.0 (~18 km/h) lets it apply in city while the hard d_rel/time-gap
                            # floors below still bound safety. Full standstill is excluded by MIN_LEAD_V.
MIN_LEAD_V = 3.0            # m/s; only compress behind a genuinely moving lead
MIN_CLOSING = 0.3           # m/s; below this there is no approach to compress into
MAX_CLOSING = 4.0           # m/s; fast closing is a hazard, not an approach
MIN_TTC_S = 8.0             # s; conservative time-to-collision gate
MIN_LEAD_A_K = -1.0         # m/s^2; a harder-braking lead ends compression
MIN_D_REL_M = 8.0           # m; absolute distance floor for every status lead
MIN_TIME_GAP_S = 1.05       # s; hard floor on the *current* time gap of every status lead
PARAMS_REFRESH_PERIOD = 50  # planner ticks

MODE_OFF = "off"
MODE_SHADOW = "shadow"
MODE_APPLY = "apply"
VALID_MODES = {MODE_OFF, MODE_SHADOW, MODE_APPLY}


def _param_string(params: Any, key: str) -> str | None:
  try:
    raw = params.get(key)
  except TypeError:
    raw = params.get(key, None)
  if raw is None:
    return None
  if isinstance(raw, bytes):
    try:
      raw = raw.decode()
    except Exception:
      return None
  return str(raw)


def _is_finite(value: Any) -> bool:
  if value is None:
    return False
  try:
    return math.isfinite(float(value))
  except (TypeError, ValueError):
    return False


class FollowGapScheduler:
  def __init__(self, params: Any = None):
    self._params = params
    self._tick = 0
    self._t_follow: float | None = None
    self.mode = MODE_OFF
    self.enabled = False
    self.last_result: dict[str, Any] | None = None
    if params is not None:
      self.refresh_params()

  def refresh_params(self) -> None:
    if self._params is None:
      return
    try:
      mode_raw = (_param_string(self._params, "DynamicFollowGapMode") or "").strip().lower()
      if mode_raw == "":
        self.mode = MODE_SHADOW
        self.enabled = False
      elif mode_raw in VALID_MODES:
        self.mode = mode_raw
        self.enabled = self.mode == MODE_APPLY
      else:
        self.mode = MODE_OFF
        self.enabled = False
    except Exception:
      self.mode = MODE_OFF
      self.enabled = False

  def scheduled(self, radarstate: Any, v_ego: float, base_t_follow: float, dt: float, *,
                long_active: bool = False, brake_pressed: bool = False, gas_pressed: bool = False,
                force_decel: bool = False, custom_long_enabled: bool | None = None,
                research_actuation_allowed: bool = False) -> float:
    """Return the T_FOLLOW to feed the MPC; the unchanged baseline unless apply mode is enabled,
    custom longitudinal is on, research actuation is allowed, and the approach context is safe."""
    self._tick += 1
    if self._params is not None and self._tick % PARAMS_REFRESH_PERIOD == 0:
      self.refresh_params()
    base = float(base_t_follow) if _is_finite(base_t_follow) and float(base_t_follow) > 0.0 else 0.0
    if base <= 0.0:
      self._t_follow = None
      self.last_result = None
      return float(base_t_follow)

    if custom_long_enabled is None:
      custom_long_enabled = False
      if self._params is not None:
        try:
          custom_long_enabled = bool(self._params.get_bool("CustomLongitudinalEnabled"))
        except Exception:
          custom_long_enabled = False
    else:
      custom_long_enabled = bool(custom_long_enabled)
    should_apply = self.mode == MODE_APPLY and self.enabled and custom_long_enabled and research_actuation_allowed
    should_shadow = self.mode in (MODE_SHADOW, MODE_APPLY)
    if not should_shadow:
      self._t_follow = None
      self.last_result = None
      return base

    try:
      if not long_active:
        # Never carry a compressed gap across a disengagement.
        self._t_follow = base
      eligible, block_reason = self._eligibility(radarstate, v_ego, long_active,
                                                 brake_pressed, gas_pressed, force_decel)
      lo = min(T_FOLLOW_COMPRESSED, base)
      target = lo if eligible else base
      prev = self._t_follow if self._t_follow is not None else base
      step = max(0.0, float(dt))
      if target < prev:
        nxt = max(prev - self._compress_rate(radarstate, v_ego, base) * step, target)
      else:
        nxt = min(prev + self._recovery_rate(radarstate, v_ego, block_reason) * step, target)
      nxt = min(max(nxt, lo), base)
      self._t_follow = nxt
      self.last_result = {
        "mode": self.mode,
        "apply": bool(should_apply and nxt < base),
        "eligible": eligible,
        "block_reason": block_reason,
        "base_t_follow": base,
        "t_follow": nxt,
      }
      return nxt if should_apply else base
    except Exception:  # fail closed: never let the scheduler break the planner
      self._t_follow = None
      self.last_result = None
      return base

  def _compress_rate(self, radarstate: Any, v_ego: float, base: float) -> float:
    """Demand-scaled compression pace. Only paces a slew already bounded to
    [T_FOLLOW_COMPRESSED, base]; the eligibility gates rule first, so this never widens
    where compression applies — only how soon an eligible approach reaches it."""
    lead_one = getattr(radarstate, "leadOne", None)
    d_rel_raw = getattr(lead_one, "dRel", None)
    v_rel_raw = getattr(lead_one, "vRel", None)
    if not (_is_finite(d_rel_raw) and _is_finite(v_rel_raw) and _is_finite(v_ego)):
      return COMPRESS_RATE
    closing = max(0.0, -float(v_rel_raw))
    # ponytail: base*v_ego approximates the desired gap (skips the MPC offset terms);
    # plenty for pacing a bounded slew.
    excess = float(d_rel_raw) - base * max(0.0, float(v_ego))
    required = COMPRESS_DEMAND_FULL if excess <= 0.1 else (closing * closing) / (2.0 * excess)
    frac = min(max(required, 0.0) / COMPRESS_DEMAND_FULL, 1.0)
    return COMPRESS_RATE + (COMPRESS_RATE_MAX - COMPRESS_RATE) * frac

  def _recovery_rate(self, radarstate: Any, v_ego: float, block_reason: str) -> float:
    """Fast recovery for safety-shaped ineligibility; gap-opening-paced recovery when the
    approach merely ended, so the MPC never brakes just to re-open the desired gap."""
    if block_reason in ("long_inactive", "brake_pressed", "gas_pressed", "force_decel"):
      return RECOVER_RATE
    lead_one = getattr(radarstate, "leadOne", None)
    if lead_one is None or not bool(getattr(lead_one, "status", False)):
      return RECOVER_RATE  # no lead: t_follow is inert to the MPC, snap-back is free
    v = float(v_ego) if _is_finite(v_ego) else 0.0
    for lead in (lead_one, getattr(radarstate, "leadTwo", None)):
      if lead is None or not bool(getattr(lead, "status", False)):
        continue
      d_rel_raw = getattr(lead, "dRel", None)
      v_rel_raw = getattr(lead, "vRel", None)
      a_lead_raw = getattr(lead, "aLeadK", None)
      if not (_is_finite(d_rel_raw) and _is_finite(v_rel_raw)):
        return RECOVER_RATE
      d_rel = float(d_rel_raw)
      closing = max(0.0, -float(v_rel_raw))
      if d_rel < max(MIN_D_REL_M, MIN_TIME_GAP_S * max(v, 0.0)) or closing > MAX_CLOSING:
        return RECOVER_RATE
      if closing > 0.0 and d_rel / closing < MIN_TTC_S:
        return RECOVER_RATE
      if _is_finite(a_lead_raw) and float(a_lead_raw) < MIN_LEAD_A_K:
        return RECOVER_RATE
    opening = max(0.0, float(getattr(lead_one, "vRel", 0.0)))
    return max(BENIGN_RECOVER_RATE_MIN, opening / max(v, 1.0))

  def _eligibility(self, radarstate: Any, v_ego: float, long_active: bool,
                   brake_pressed: bool, gas_pressed: bool, force_decel: bool) -> tuple[bool, str]:
    if not long_active:
      return False, "long_inactive"
    if brake_pressed:
      return False, "brake_pressed"
    if gas_pressed:
      return False, "gas_pressed"
    if force_decel:
      return False, "force_decel"
    if not _is_finite(v_ego) or float(v_ego) < MIN_V_EGO_FOR_APPLY:
      return False, "low_speed"
    v_ego = float(v_ego)

    lead_one = getattr(radarstate, "leadOne", None)
    if lead_one is None or not bool(getattr(lead_one, "status", False)):
      return False, "no_lead"

    # Hard floors for every status lead: t_follow is one MPC parameter shared by both lead
    # obstacles, so no status lead may sit close/fast enough to make compression a hazard.
    for idx, lead in enumerate((lead_one, getattr(radarstate, "leadTwo", None))):
      if lead is None or not bool(getattr(lead, "status", False)):
        continue
      d_rel_raw = getattr(lead, "dRel", None)
      v_rel_raw = getattr(lead, "vRel", None)
      if not (_is_finite(d_rel_raw) and _is_finite(v_rel_raw)):
        return False, f"lead_{idx}_non_finite"
      d_rel = float(d_rel_raw)
      v_rel = float(v_rel_raw)
      if d_rel < max(MIN_D_REL_M, MIN_TIME_GAP_S * v_ego):
        return False, f"lead_{idx}_too_close"
      closing = max(0.0, -v_rel)
      if closing > MAX_CLOSING:
        return False, f"lead_{idx}_fast_closing"
      if closing > 0.0 and d_rel / closing < MIN_TTC_S:
        return False, f"lead_{idx}_low_ttc"

    # Approach context on the primary lead: slower, moving, not braking hard.
    v_lead_raw = getattr(lead_one, "vLead", None)
    a_lead_raw = getattr(lead_one, "aLeadK", None)
    if not (_is_finite(v_lead_raw) and _is_finite(a_lead_raw)):
      return False, "lead_0_non_finite"
    v_lead = float(v_lead_raw)
    a_lead = float(a_lead_raw)
    if v_lead < MIN_LEAD_V:
      return False, "lead_slow_or_stopped"
    if a_lead < MIN_LEAD_A_K:
      return False, "lead_braking"
    closing = max(0.0, -float(getattr(lead_one, "vRel", 0.0)))
    if closing < MIN_CLOSING:
      return False, "not_closing"
    return True, ""
