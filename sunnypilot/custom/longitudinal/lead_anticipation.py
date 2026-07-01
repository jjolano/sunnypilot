"""Lead-motion anticipation (§3) — confidence-shape the radar lead's accel before the MPC.

A transient, low-confidence negative ``aLeadK`` spike, extrapolated by the MPC's ``process_lead``,
drives reactive braking (see docs/adr/2026-06-14-longitudinal-lead-anticipation.md). This adapter
discounts ``aLeadK`` by a confidence derived from lead-track stability (the same
``LeadConfidenceTracker`` the custom stack uses), so an unconfirmed/transient lead-decel doesn't brake
the car — while a confident or *sustained* decel propagates at full weight.

SAFETY — unlike the a_target shaper, this reaches the MPC *input* and can reduce braking. So it:
- only ever makes a *braking* lead look *less* braking — the shaped ``aLeadK`` stays in ``[raw, 0]``
  (never positive, never beyond the raw measurement, so the lead is never predicted faster/closer);
- caps any raw ``aLeadK`` softening to ``AL_CAP_MAX_SOFTENING``;
- only applies with explicit planner safety context: long active, no driver/force-decel override,
  enough speed, finite lead kinematics, no close gap, no fast closing, and no low TTC on any status lead;
- never discounts a non-braking lead, a high-confidence lead, or a sustained brake;
- floors the discount at ``DISCOUNT_FLOOR``;
- is opt-in via ``LeadAnticipationMode=apply``; legacy ``LeadAnticipationEnabled`` is storage-only,
  and all missing/invalid modes fail closed to raw radarState passthrough.

Validate via ``profile_lead_following`` replay (reactive-brake + decel-peak down, headway not up,
zero new close-approaches) before any default-on.
"""
from __future__ import annotations

import math
from typing import Any

from openpilot.sunnypilot.custom.longitudinal.lead_confidence import NEW_LEAD_STABLE_TIME, LeadConfidenceTracker

HIGH_CONFIDENCE = 0.8        # at/above this confidence, trust the decel fully (no discount)
DISCOUNT_FLOOR = 0.5         # never discount a measured decel below this fraction of the raw aLeadK
SUSTAINED_BRAKE_S = 0.6      # consecutive braking this long => a real brake, no discount
SUSTAINED_BRAKE_A = -0.5     # m/s^2; aLeadK at/below this counts as a meaningful brake
PARAMS_REFRESH_PERIOD = 50   # planner ticks
AL_CAP_MAX_SOFTENING = 0.25  # m/s^2; shaped aLeadK may be at most raw + this and stays in [raw, 0]
MIN_V_EGO_FOR_APPLY = 2.0    # m/s; below this, shaping is too risky in stop-and-go / standstill context
MIN_D_REL_M = 8.0            # m; absolute minimum lead distance for apply
MIN_D_REL_TIME_GAP = 0.8     # s; lead distance must also exceed this * v_ego
MAX_CLOSING_V_REL = -4.0     # m/s; block shaping for fast-closing leads
MIN_TTC_S = 3.0              # s; conservative time-to-collision gate for closing leads
MODE_OFF = "off"
MODE_SHADOW = "shadow"
MODE_APPLY = "apply"
VALID_MODES = {MODE_OFF, MODE_SHADOW, MODE_APPLY}


def _param_string(params: Any, key: str) -> str | None:
  """Read the explicitly stored string param; defaults are applied by LeadAnticipation."""
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


def _f(value: Any, default: float = 0.0) -> float:
  try:
    v = float(value)
  except (TypeError, ValueError):
    return default
  return v if math.isfinite(v) else default


def _is_finite(value: Any) -> bool:
  """True only when the raw value is present and finite (do not sanitize to a permissive default)."""
  if value is None:
    return False
  try:
    return math.isfinite(float(value))
  except (TypeError, ValueError):
    return False


def _confidence(state: Any) -> float:
  """0..1 from lead-track stability: a stable, non-flickering, established track is fully trusted; a
  new / flickering / just-acquired track is not (its aLeadK is the noisy term we guard against)."""
  if state is None or not bool(getattr(state, "status", False)):
    return 0.0
  if state.new_lead or state.guard_timer > 0.0 or state.flicker_guard_timer > 0.0:
    return 0.0
  if state.stable:
    return 1.0
  return max(0.0, min(0.9, _f(state.age) / NEW_LEAD_STABLE_TIME))


def _clamped_brake_softening(raw_a: float, proposed_a: float, cap: float) -> float:
  return max(raw_a, min(proposed_a, raw_a + cap, 0.0))


class _ShapedLead:
  """Proxies a radar lead, overriding only aLeadK (and aLeadTau)."""
  def __init__(self, lead: Any, a_lead: float, a_lead_tau: float):
    object.__setattr__(self, "_lead", lead)
    object.__setattr__(self, "aLeadK", float(a_lead))
    object.__setattr__(self, "aLeadTau", float(a_lead_tau))

  def __getattr__(self, name: str) -> Any:
    return getattr(self._lead, name)


class _ShapedRadarState:
  """Proxies radarState, overriding only leadOne/leadTwo."""
  def __init__(self, rs: Any, lead_one: Any, lead_two: Any):
    object.__setattr__(self, "_rs", rs)
    object.__setattr__(self, "leadOne", lead_one)
    object.__setattr__(self, "leadTwo", lead_two)

  def __getattr__(self, name: str) -> Any:
    return getattr(self._rs, name)


class LeadAnticipation:
  def __init__(self, params: Any = None):
    self._params = params
    self._conf = (LeadConfidenceTracker(), LeadConfidenceTracker())
    self._brake_s = [0.0, 0.0]
    self._tick = 0
    self.mode = MODE_OFF
    self.enabled = False
    self.last_result = None
    if params is not None:
      self.refresh_params()

  def refresh_params(self) -> None:
    if self._params is None:
      return
    try:
      mode_raw = (_param_string(self._params, "LeadAnticipationMode") or "").strip().lower()
      if mode_raw == "":
        # Legacy bool is inert for behavior; missing mode resolves to shadow.
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

  def shape(self, radarstate: Any, dt: float, *, long_active: bool = False,
            brake_pressed: bool = False, gas_pressed: bool = False,
            force_decel: bool = False, v_ego: float = 0.0,
            research_actuation_allowed: bool = False) -> Any:
    """Return radarstate unchanged unless apply mode is enabled, custom longitudinal is on, and
    research actuation is allowed.

    The optional apply-context keyword arguments default fail-closed (no actuation). Shadow mode
    still computes and records shaped candidates for telemetry regardless of context.
    """
    self._tick += 1
    if self._params is not None and self._tick % PARAMS_REFRESH_PERIOD == 0:
      self.refresh_params()
    custom_long_enabled = False
    if self._params is not None:
      try:
        custom_long_enabled = bool(self._params.get_bool("CustomLongitudinalEnabled"))
      except Exception:
        custom_long_enabled = False
    should_apply = self.mode == MODE_APPLY and self.enabled and custom_long_enabled and research_actuation_allowed
    should_shadow = self.mode in (MODE_SHADOW, MODE_APPLY)
    if not should_shadow and not self.enabled:
      self.last_result = None
      return radarstate
    try:
      one = self._shape_lead(getattr(radarstate, "leadOne", None), 0, dt)
      two = self._shape_lead(getattr(radarstate, "leadTwo", None), 1, dt)
      raw_one = getattr(radarstate, "leadOne", None)
      raw_two = getattr(radarstate, "leadTwo", None)
      apply_allowed, block_reason = self._apply_gate(should_apply, (raw_one, raw_two),
                                                     long_active, brake_pressed, gas_pressed,
                                                     force_decel, v_ego)
      self.last_result = {
        "mode": self.mode,
        "apply": apply_allowed,
        "block_reason": block_reason,
        "leadOneRaw": _f(getattr(raw_one, "aLeadK", 0.0)) if raw_one is not None else None,
        "leadOneShaped": _f(getattr(one, "aLeadK", 0.0)) if one is not None else None,
        "leadTwoRaw": _f(getattr(raw_two, "aLeadK", 0.0)) if raw_two is not None else None,
        "leadTwoShaped": _f(getattr(two, "aLeadK", 0.0)) if two is not None else None,
      }
      if not apply_allowed:
        return radarstate
      return _ShapedRadarState(radarstate, one, two)
    except Exception:   # fail closed: never let anticipation break the planner
      self.last_result = None
      return radarstate

  def _apply_gate(self, should_apply: bool, leads: tuple[Any, Any], long_active: bool,
                  brake_pressed: bool, gas_pressed: bool, force_decel: bool,
                  v_ego: float) -> tuple[bool, str]:
    """Conservative context gate: only allow shaping when custom long is on, long is active, the
    driver is not overriding, forceDecel is not in effect, speed is high enough, and *every*
    status lead that would be shaped passes close / low-TTC / fast-closing / finite checks."""
    if not should_apply:
      return False, "mode_or_enabled"
    if not long_active:
      return False, "long_inactive"
    if brake_pressed:
      return False, "brake_pressed"
    if gas_pressed:
      return False, "gas_pressed"
    if force_decel:
      return False, "force_decel"
    if not _is_finite(v_ego) or v_ego < MIN_V_EGO_FOR_APPLY:
      return False, "low_speed"
    for idx, lead in enumerate(leads):
      if lead is None or not bool(getattr(lead, "status", False)):
        continue
      d_rel_raw = getattr(lead, "dRel", None)
      v_rel_raw = getattr(lead, "vRel", None)
      if not (_is_finite(d_rel_raw) and _is_finite(v_rel_raw)):
        return False, f"lead_{idx}_non_finite"
      # Values are already verified finite above; _f is only used for type-compatible extraction.
      d_rel = _f(d_rel_raw)
      v_rel = _f(v_rel_raw)
      if d_rel < max(MIN_D_REL_M, MIN_D_REL_TIME_GAP * v_ego):
        return False, f"lead_{idx}_too_close"
      if v_rel < MAX_CLOSING_V_REL:
        return False, f"lead_{idx}_fast_closing"
      closing_speed = max(0.0, -v_rel)
      if closing_speed > 0.0 and d_rel / closing_speed < MIN_TTC_S:
        return False, f"lead_{idx}_low_ttc"
    return True, ""

  def _shape_lead(self, lead: Any, idx: int, dt: float) -> Any:
    state = self._conf[idx].update(lead, dt)
    if lead is None or not bool(getattr(lead, "status", False)):
      self._brake_s[idx] = 0.0
      return lead
    a_lead = _f(getattr(lead, "aLeadK", 0.0))
    # sustained-brake tracker: count consecutive time the lead has been meaningfully braking.
    self._brake_s[idx] = self._brake_s[idx] + max(0.0, _f(dt)) if a_lead <= SUSTAINED_BRAKE_A else 0.0
    if a_lead >= 0.0:
      return lead                            # never touch a non-braking lead
    if _confidence(state) >= HIGH_CONFIDENCE or self._brake_s[idx] >= SUSTAINED_BRAKE_S:
      return lead                            # trust a confident or sustained decel fully
    discount = max(DISCOUNT_FLOOR, _confidence(state))
    a_shaped = a_lead * discount                   # a_lead < 0, discount in (0,1] -> in [a_lead, 0]
    # Hard safety cap: the softening delta is bounded so the lead can never look meaningfully
    # faster/closer. Shaped aLeadK stays in [raw, 0] and is at most raw + AL_CAP_MAX_SOFTENING.
    a_shaped = _clamped_brake_softening(a_lead, a_shaped, AL_CAP_MAX_SOFTENING)
    return _ShapedLead(lead, a_shaped, _f(getattr(lead, "aLeadTau", 1.5)))
