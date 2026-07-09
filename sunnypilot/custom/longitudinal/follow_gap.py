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
- slews slowly into compression and quickly back out (asymmetric rate limit), and snaps back
  to baseline whenever long control is inactive;
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
COMPRESS_RATE = 0.05        # s of t_follow per s; slow slide into compression
RECOVER_RATE = 0.5          # s of t_follow per s; fast recovery to baseline
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
        nxt = max(prev - COMPRESS_RATE * step, target)
      else:
        nxt = min(prev + RECOVER_RATE * step, target)
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
