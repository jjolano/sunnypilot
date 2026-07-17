"""Longitudinal policy — builds candidates for the decision core (custom-2.0 mechanisms).

Faithful port of the custom-2.0 policy logic (custom_v2.py) into the clean candidate model:
each mechanism produces a ``LongitudinalCandidate`` with the role + evidence source that lets
the decision core arbitrate it and the mode gate admit/exclude it. The hypermile behaviors —
downhill free-coast leeway, early/gentle stop-approach, no-lead launch pulse, comfort relax,
coast-biased advisory caps — are preserved with their legacy constants (``policy_tables``).

Routing through the mode gate (so ACC stays OEM-like): the cruise-domain behaviors (coast,
launch, comfort relax) carry the CRUISE source (admitted in every mode), while model-stop,
map, speed-limit, and curve carry their own sources and are admitted only in E2E/SCC. The
decision core applies the admission, so no per-mode flags are needed here.

Feel-VALIDATION (does the tuning match the old fork on the road) is gated on the engaged
corpus; the implementation and its constants are complete.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

from openpilot.sunnypilot.custom.longitudinal.decision import CandidateRole, LongitudinalCandidate
from openpilot.sunnypilot.custom.longitudinal.lead_cushion import lead_following_cushion, lead_speedup_guard
from openpilot.sunnypilot.custom.longitudinal.lead_speed_alignment import (
  AlignmentAction,
  lead_speed_alignment,
)
from openpilot.sunnypilot.custom.longitudinal.model_trust import GENTLE_CAUTION_DECEL, gate_model_stop
from openpilot.sunnypilot.custom.longitudinal.modes import EvidenceClass
from openpilot.sunnypilot.custom.longitudinal.runway_governor import runway_comfort_governor
from openpilot.sunnypilot.custom.longitudinal.policy_tables import (
  COMFORT_RELAX_ACCEL_MIN,
  CRUISE_LEEWAY_DOWNHILL_ACCEL,
  CRUISE_LEEWAY_HIGHWAY_MAX,
  CRUISE_LEEWAY_HIGHWAY_MIN_V_EGO,
  CRUISE_LEEWAY_MAX,
  CRUISE_LEEWAY_MIN,
  CRUISE_LEEWAY_RECOVERY,
  FLAT_COAST_BASELINE,
  GRADE_FLAT_BAND_HALF_WIDTH,
  LEAD_CRAWL_ACCEL_MAX,
  LEAD_CRAWL_BREAKOUT_MIN_OPENING,
  LEAD_CRAWL_LAUNCH_TAU,
  LEAD_CRAWL_MAX_D_REL,
  LEAD_CRAWL_MAX_V_EGO,
  LEAD_CRAWL_MAX_V_LEAD,
  LEAD_LAUNCH_MAX_V_EGO,
  LEAD_LAUNCH_TAU,
  LEAD_PULLAWAY_MIN_OPENING,
  LEAD_PULLAWAY_MIN_V_LEAD,
  NO_LEAD_LAUNCH_MAX_V_EGO,
  NO_LEAD_STOP_CLEAR_ACCEL_MIN,
  NO_LEAD_STOP_CLEAR_DISTANCE,
  PROGRESS_CRUISE_SPEED_MARGIN,
  STOP_APPROACH_DECEL_MIN,
  STOP_LANDING_DECEL_MIN,
  STOP_LANDING_SOFTEN_MAX_V_EGO,
  Personality,
  launch_accel_max,
  stop_approach_comfort_decel,
)
from openpilot.sunnypilot.custom.longitudinal.coast_horizon import (
  DEFAULT_COAST_DECEL,
  MAX_COAST_DECEL,
  CoastAction,
  CoastHorizonInputs,
  coast_horizon,
)

SAFETY_FORCE_SLOW_DECEL = -0.2

# Early, non-committing model-slowdown caution: fresh model desired decel below this
# threshold, before shouldStop / stop distance are available, produces a bounded
# stop_approach candidate capped at the existing precautionary decel.
EARLY_MODEL_SLOWDOWN_DECEL_THRESHOLD = -0.2

# Lead crawl pull-away: keep the damped close-crawl behavior at the initial close gap, but
# ramp toward the normal personality launch accel once the lead opens usable follow-gap space.
_LEAD_CRAWL_RAMP_START_EXCESS_M = 1.0
_LEAD_CRAWL_RAMP_FULL_EXCESS_M = 4.0

# Overspeed coast leeway guards.
_OVERSPEED_RATE_GUARD_ACCEL = -0.25          # mild braking floor when climbing above leeway
_OVERSPEED_RATE_GUARD_A_EGO_THRESHOLD = 0.05
_OVERSPEED_FAR_LEAD_MARGIN_M = 10.0
_OVERSPEED_FAR_LEAD_TIME_GAP_S = 3.0
_OVERSPEED_LEAD_CONFIDENCE_MIN = 0.55
_OVERSPEED_LEAD_CRAWL_V_MAX = 1.0
_OVERSPEED_LEAD_TARGET_EPS = 0.05
_OVERSPEED_LEAD_HARD_BRAKE_A_K = -0.5

# Far, non-closing radar lead => the model's *uncorroborated* slowdown has no physical reason to
# brake yet. Route 00000274: the early model stop_approach caution floor deepened to
# ~-0.7 on same-speed leads 40-65 m out, then released within ~1 s (a spurious "overreaction to a
# far lead"). Routes 00000275-0000027b: 72.9% of matching manual frames coasted and none used
# medium braking. Coast only when desired acceleration is the sole model slowdown evidence; a
# finite stop point (for example, a traffic-light stop) always keeps its braking authority. The
# MPC lead hazard also binds independently downstream.
_FAR_LEAD_CAUTION_MIN_D_REL = 35.0     # m; lead far enough that its slowdown isn't imminent
_FAR_LEAD_CAUTION_MAX_CLOSING = 1.0    # m/s; closing slower than this is effectively same-speed
_FAR_LEAD_CAUTION_DECEL_CAP = 0.0      # m/s^2; coast for uncorroborated far-lead caution


def _uncorroborated_far_lead(scene: LongitudinalScene) -> bool:
  """A tracked lead that is far and not meaningfully closing — the model's non-committed slowdown
  here is uncorroborated by lead kinematics, so its caution decel is coast-capped."""
  return bool(scene.has_lead and math.isfinite(scene.lead_d_rel) and math.isfinite(scene.lead_v_rel)
              and scene.lead_d_rel > _FAR_LEAD_CAUTION_MIN_D_REL
              and scene.lead_v_rel > -_FAR_LEAD_CAUTION_MAX_CLOSING)

# Conservative uphill grade recovery: small extra cruise accel only when the grade signal is
# clearly uphill and there are no lead/stop/curve/speed-limit/override contexts. The gain is
# additive to the cruise seed; a measured-accel guard backs it off after a surge/downshift.
_UPHILL_RECOVERY_MAX_MS2 = 0.15
_UPHILL_RECOVERY_TARGET_MAX_MS2 = 0.80
_UPHILL_RECOVERY_MIN_V_EGO = 2.0
_UPHILL_A_EGO_SURGE_MARGIN = 0.25

# Lead-softening thresholds (Phase 4): narrow pre-MPC shaping for far/medium low-risk
# lead-follow decel. Conservative first cut; before/deploy validation must replay engaged routes.
_LEAD_SOFTEN_MAX_LEAD_DECEL = -0.7          # never soften a lead that already wants real braking
_LEAD_SOFTEN_MIN_V_EGO = 8.0                # m/s; avoid stop-and-go / low-speed context
_LEAD_SOFTEN_MIN_LEAD_V = 5.0               # m/s; avoid stopped/crawling leads
_LEAD_SOFTEN_FOLLOW_TIME_GAP_S = 2.2        # seconds; stricter than steady follow gap
_LEAD_SOFTEN_MIN_DISTANCE_BASE_M = 45.0     # m; fixed medium-bin boundary
_LEAD_SOFTEN_USABLE_EXCESS_MARGIN_M = 3.0   # m; small safety margin on excess gap
_LEAD_SOFTEN_MAX_REQUIRED_DECEL = 0.25      # m/s^2; low-risk closing threshold
_LEAD_SOFTEN_RAISE_DELTA = 0.4              # m/s^2; bounded raise over the lead target
_LEAD_SOFTEN_CEILING = -0.05                # m/s^2; near-coast ceiling, never positive

# Far-lead decel authority is bounded by a multiple of the decel the closing kinematics
# actually require. Radar track churn beyond ~60 m flaps dRel/aLeadK (routes 0000027d,
# 00000282: -0.5..-0.8 aLeadK spikes on same-speed/pulling-away leads leaked -0.15..-0.35
# dips into finalA), and a noise spike carries no kinematic requirement. Continuous in
# distance and closing speed — no on/off threshold for a churny track to flicker across.
# The asymmetric "trust braking promptly" doctrine is untouched: this bounds how much
# authority geometry grants, not whether the braking signal is believed.
_LEAD_RELEVANCE_AUTHORITY_K = 2.5      # decel allowance as a multiple of required decel
_LEAD_RELEVANCE_COAST_MARGIN = 0.1     # m/s^2; a gentle coast is always allowed

# Hard-braker rest-point coast: a persistently hard-braking lead defines a future
# constraint point — its constant-decel rest distance — so lift off per the coast horizon
# now instead of waiting for closing speed to build. Corpus 2026-07-15 (11 far episodes,
# 0 interventions): peaks ran ~0.4 m/s^2 beyond the constant-decel need and episodes
# resolved with 2.6 s median time gap unused (MPC's Gaussian a_lead decay assumes the
# braking fades; the relevance cap above is reactive by design). COAST authority only —
# never below natural coast — so a phantom (10% of corpus armings) costs a brief lift;
# braking depth stays owned by the MPC/lead physics floor, which advisory caps cannot relax.
LEAD_REST_POINT_HARD_BRAKE_A = -1.0  # m/s^2; sustained lead decel that defines a rest point
LEAD_REST_POINT_PERSIST_S = 1.0      # s; anti-churn persistence before the advisory arms
LEAD_REST_POINT_MARGIN_M = 6.0       # m; arrive short of the projected rest point

# Inside-gap compression/recovery thresholds (Phase 3): controlled compression for
# stable/confident same-lead braking when projected collision risk is low/moderate.
# Strong/flickery/new-stop threats are left to the normal lead-follow physical hazard.
_LEAD_INSIDE_GAP_MIN_TIME_GAP_S = 1.1
_LEAD_INSIDE_GAP_MIN_TTC_S = 5.0
_LEAD_INSIDE_GAP_MAX_REQUIRED_DECEL = 0.45  # m/s^2; kinematic closing demand ceiling
_LEAD_INSIDE_GAP_MAX_CLOSING_MPS = 2.0      # m/s; moderate closing only
_LEAD_INSIDE_GAP_MIN_LEAD_A_K = -1.2        # m/s^2; reject compression if lead is braking harder
_LEAD_INSIDE_GAP_COMPRESSION_ENTER_MPS = 0.10
_LEAD_INSIDE_GAP_COMPRESSION_EXIT_MPS = 0.0

# Routine-braking compression tier (Phase 3b): allows a stronger but still bounded
# compression target when the desired-gap kinematic demand is higher, as long as the
# true collision-buffer demand and TTC remain safe.
_LEAD_ROUTINE_GAP_MIN_TIME_GAP_S = 1.2
_LEAD_ROUTINE_GAP_MIN_TTC_S = 6.0
_LEAD_ROUTINE_GAP_MAX_CLOSING_MPS = 2.5
_LEAD_ROUTINE_GAP_MIN_LEAD_A_K = -2.0
_LEAD_ROUTINE_GAP_MAX_REQUIRED_DECEL = 0.90
_LEAD_ROUTINE_GAP_MAX_COLLISION_DECEL = 1.0
_LEAD_ROUTINE_GAP_TARGET_MIN = 0.45
_LEAD_ROUTINE_GAP_TARGET_MAX = 0.85


@dataclass(frozen=True)
class LongitudinalScene:
  v_ego: float
  a_ego: float = 0.0            # current ego accel (wired for future smoothing; Phase 1 unused)
  v_cruise: float = 0.0
  seed_a_target: float = 0.0    # planner baseline (MPC cruise/seed) accel
  accel_coast: float = 0.0      # natural coast estimate (negative downhill; can be non-negative when no useful coast)
  pitch: float | None = None    # optional road pitch; None means unavailable
  personality: Personality = Personality.STANDARD
  # lead (pre-MPC lead-present seed shaping; final MPC lead physics remains downstream)
  has_lead: bool = False
  lead_a_target: float = 0.0
  lead_should_stop: bool = False
  lead_gap_excess: float = 0.0
  lead_progress_allowed: bool = False
  # lead kinematics (for cushion / speedup guard / radar corroboration)
  lead_v: float = 0.0
  lead_d_rel: float = 0.0
  lead_v_rel: float = 0.0
  lead_a_k: float = 0.0
  lead_a_tau: float = 1.5
  lead_hard_brake_s: float = 0.0  # s of sustained leadOne aLeadK < LEAD_REST_POINT_HARD_BRAKE_A
  follow_gap: float = 0.0
  lead_kinematics_valid: bool = True
  # lead context (alignment gating)
  lead_confidence: float = 0.0
  lead_stable: bool = False
  lead_shadow_active: bool = False
  alternate_threat_active: bool = False
  # model stop (E2E)
  model_should_stop: bool = False
  model_stop_distance: float | None = None
  model_desired_accel: float = 0.0
  model_stop_prob: float = 1.0   # model confidence in the stop (trust gate); 1.0 = fully trusted
  model_stale: bool = False
  stop_threat: bool = False
  # rate-limited caution floor (CautionRamp in wiring); non-committed model-stop decel
  # may not exceed this, so sustained demand earns depth and flickers stay gentle
  model_caution_floor: float = GENTLE_CAUTION_DECEL
  # advisory sources
  speed_limit_active: bool = False
  speed_limit_v_target: float = 0.0
  speed_limit_a_target: float = 0.0
  speed_limit_distance: float | None = None  # m to the limit change (resolver), when known
  curve_active: bool = False
  curve_a_target: float = 0.0
  curve_v_target: float = 0.0                # required speed at the binding curve point
  curve_distance: float | None = None        # m to the binding curve point, when known
  curve_source: EvidenceClass = EvidenceClass.CURVE_VISION   # which SCC curve source bound the cap
  # map-coast tier (SCC-Map bounded apply: lift-off only, never a braking request)
  map_coast_active: bool = False              # apply-eligible (mode/research/toggle gates upstream)
  map_coast_v_target: float = 0.0             # required speed at the map slowdown
  map_coast_distance: float | None = None     # m to the map slowdown, when known
  # driver / safety
  force_slow_decel: bool = False
  brake_pressed: bool = False
  gas_pressed: bool = False


def _clip(value: float, lo: float, hi: float) -> float:
  return max(lo, min(hi, value))


def stopping_decel(v_ego: float, distance: float, min_distance: float = 1.0) -> float:
  """Signed constant accel required to stop over distance (kinematic)."""
  return -(float(v_ego) ** 2) / (2.0 * max(float(distance), min_distance))


def dynamic_cruise_overspeed_leeway(accel_coast: float, v_ego: float = 0.0) -> float:
  confidence = _clip(float(accel_coast) / CRUISE_LEEWAY_DOWNHILL_ACCEL, 0.0, 1.0)
  leeway_max = CRUISE_LEEWAY_HIGHWAY_MAX if v_ego >= CRUISE_LEEWAY_HIGHWAY_MIN_V_EGO else CRUISE_LEEWAY_MAX
  return CRUISE_LEEWAY_MIN + confidence * (leeway_max - CRUISE_LEEWAY_MIN)


def _lead_allows_overspeed_coast(scene: LongitudinalScene) -> bool:
  """Treat only far, stable, opening/faster leads as effectively no-lead for overspeed coast."""
  if not scene.has_lead:
    return True
  if scene.lead_should_stop or not scene.lead_kinematics_valid or not scene.lead_stable:
    return False
  if scene.lead_shadow_active or scene.alternate_threat_active:
    return False
  if not all(math.isfinite(v) for v in (scene.v_ego, scene.v_cruise, scene.lead_v,
                                        scene.lead_v_rel, scene.lead_d_rel, scene.follow_gap,
                                        scene.lead_confidence, scene.lead_a_k, scene.lead_a_target,
                                        scene.seed_a_target)):
    return False
  if scene.lead_a_target < scene.seed_a_target - _OVERSPEED_LEAD_TARGET_EPS:
    return False
  if scene.lead_a_k < _OVERSPEED_LEAD_HARD_BRAKE_A_K:
    return False
  if scene.lead_confidence < _OVERSPEED_LEAD_CONFIDENCE_MIN:
    return False
  if scene.lead_v <= _OVERSPEED_LEAD_CRAWL_V_MAX:
    return False
  if scene.lead_v_rel < 0.0 or scene.lead_v < max(scene.v_ego, scene.v_cruise):
    return False
  far_gate = max(scene.follow_gap + _OVERSPEED_FAR_LEAD_MARGIN_M,
                 _OVERSPEED_FAR_LEAD_TIME_GAP_S * scene.v_ego)
  return bool(scene.lead_d_rel > far_gate)


def dynamic_cruise_coast_accel(scene: LongitudinalScene, a_target: float) -> float:
  """Let speed bleed off naturally (coast) instead of braking, within a downhill-scaled
  overspeed leeway — the hypermile free-coast behavior."""
  if scene.stop_threat or scene.force_slow_decel or scene.v_cruise <= 0.0:
    return a_target
  if not _lead_allows_overspeed_coast(scene):
    return a_target
  overspeed = scene.v_ego - scene.v_cruise
  if overspeed <= 0.0:
    return a_target
  leeway = dynamic_cruise_overspeed_leeway(scene.accel_coast, scene.v_ego)
  if overspeed <= leeway:
    return min(0.0, max(a_target, scene.accel_coast))
  recovery = _clip((overspeed - leeway) / CRUISE_LEEWAY_RECOVERY, 0.0, 1.0)
  coast_target = (1.0 - recovery) * min(0.0, scene.accel_coast) + recovery * a_target
  if scene.a_ego > _OVERSPEED_RATE_GUARD_A_EGO_THRESHOLD:
    coast_target = min(coast_target, _OVERSPEED_RATE_GUARD_ACCEL)
  return min(0.0, max(a_target, coast_target))


def uphill_grade_recovery_accel(scene: LongitudinalScene) -> float | None:
  """Return a small authorized PROGRESS accel target for clear uphill cruise recovery.

  Returns None when the context is not a clean no-lead cruise recovery or when the car has
  already surged enough that extra gain is unnecessary (transient / downshift guard).
  """
  if (scene.has_lead or scene.stop_threat or scene.model_should_stop or scene.force_slow_decel
      or scene.brake_pressed or scene.gas_pressed):
    return None
  if scene.speed_limit_active or scene.curve_active:
    return None
  if not (scene.v_cruise > scene.v_ego + PROGRESS_CRUISE_SPEED_MARGIN):
    return None
  if scene.v_ego < _UPHILL_RECOVERY_MIN_V_EGO:
    return None
  if not math.isfinite(scene.accel_coast) or not math.isfinite(scene.a_ego):
    return None
  estimated_bias = scene.accel_coast - FLAT_COAST_BASELINE
  if estimated_bias >= -GRADE_FLAT_BAND_HALF_WIDTH:
    return None
  seed = float(scene.seed_a_target)
  if seed <= 0.0 or seed >= _UPHILL_RECOVERY_TARGET_MAX_MS2:
    return None
  gain = _UPHILL_RECOVERY_MAX_MS2
  candidate = min(seed + gain, _UPHILL_RECOVERY_TARGET_MAX_MS2, launch_accel_max(scene.personality))
  # Downshift / transient guard: if measured accel already exceeds what we would ask for, or
  # is already well above the cruise seed, skip the extra gain.
  if scene.a_ego > seed + _UPHILL_A_EGO_SURGE_MARGIN:
    return None
  if scene.a_ego > candidate:
    return None
  if candidate <= seed or candidate <= 0.0:
    return None
  return float(candidate)


def no_lead_stop_clear(scene: LongitudinalScene) -> bool:
  d = scene.model_stop_distance
  return bool(not scene.model_should_stop and (d is None or d > NO_LEAD_STOP_CLEAR_DISTANCE)
              and scene.model_desired_accel >= NO_LEAD_STOP_CLEAR_ACCEL_MIN)


def _lead_crawl_launch_context(scene: LongitudinalScene) -> bool:
  return bool(
    scene.v_ego < LEAD_CRAWL_MAX_V_EGO and
    scene.lead_v < LEAD_CRAWL_MAX_V_LEAD and
    scene.lead_d_rel < LEAD_CRAWL_MAX_D_REL and
    scene.lead_v_rel < LEAD_CRAWL_BREAKOUT_MIN_OPENING
  )


def _crawl_launch_accel(delta_v: float, gap_excess: float, launch_cap: float) -> float:
  """Gentle for tiny twitches; ramp up when the lead keeps opening usable gap."""
  gentle = delta_v / LEAD_CRAWL_LAUNCH_TAU
  if delta_v <= 0.4:
    base = gentle
  else:
    # ramp up from (0.4 m/s -> 0.16 m/s^2) with a shorter tau once the lead is genuinely moving
    stronger = 0.16 + (delta_v - 0.4) / 1.5
    base = min(LEAD_CRAWL_ACCEL_MAX, max(gentle, stronger))
  base = min(base, launch_cap)
  if base >= launch_cap or not math.isfinite(gap_excess) or gap_excess <= _LEAD_CRAWL_RAMP_START_EXCESS_M:
    return base

  ramp = _clip((gap_excess - _LEAD_CRAWL_RAMP_START_EXCESS_M) /
               (_LEAD_CRAWL_RAMP_FULL_EXCESS_M - _LEAD_CRAWL_RAMP_START_EXCESS_M), 0.0, 1.0)
  return base + ramp * (launch_cap - base)


def lead_pullaway_accel(scene: LongitudinalScene, personality: Personality) -> float:
  delta_v = max(0.0, scene.lead_v - scene.v_ego)
  launch_cap = launch_accel_max(personality)
  if _lead_crawl_launch_context(scene):
    if math.isfinite(scene.lead_d_rel) and math.isfinite(scene.follow_gap) and scene.follow_gap > 0.0:
      gap_excess = scene.lead_d_rel - scene.follow_gap
    else:
      gap_excess = 0.0
    return _crawl_launch_accel(delta_v, max(0.0, gap_excess), launch_cap)
  return min(launch_cap, delta_v / LEAD_LAUNCH_TAU)


def stop_approach_accel(scene: LongitudinalScene) -> tuple[float, bool]:
  """Early, gentle stop-approach: start at the personality comfort decel, allow a stronger
  transient only when the runway physics genuinely require it. Returns (a_target, hard_stop)."""
  comfort_decel = stop_approach_comfort_decel(scene.personality)
  a = comfort_decel
  hard = False
  if scene.model_stop_distance is not None and scene.model_stop_distance > 0.0:
    required = stopping_decel(scene.v_ego, scene.model_stop_distance, min_distance=1.0)
    if required < comfort_decel:
      a = min(a, required)
    if scene.model_should_stop:
      a = min(a, scene.model_desired_accel, required)
    if scene.model_should_stop and required < STOP_APPROACH_DECEL_MIN:
      return a, True
  elif scene.model_should_stop:
    a = min(a, scene.model_desired_accel)
  if hard:
    return a, True
  floor = _stop_landing_decel_floor(scene.v_ego)
  # Corroboration-earned depth for uncommitted stops: route 28c t=741, the floor pinned
  # -1.50 for 1.2 s while the radar-corroborated model demand deepened to -2.5 (required
  # ~-2.0 from the model's own stop point) and the driver overrode at -4. The CautionRamp
  # already rate-limits how fast sustained demand may earn depth (0.45 m/s^2 per s, bounded
  # -2.5), so one-frame quirks still land on the -1.5 floor while a persistent, corroborated
  # deepening is allowed to follow the kinematic requirement. Wiring (CorroborationHold)
  # clamps the incoming floor back to STOP_APPROACH_DECEL_MIN unless a closing radar echo
  # was seen recently, so vision-only demand cannot earn depth here. The final landing band
  # (< 2.5 m/s) keeps its softened floor untouched.
  if scene.v_ego >= STOP_LANDING_SOFTEN_MAX_V_EGO and math.isfinite(scene.model_caution_floor):
    floor = min(floor, scene.model_caution_floor)
  return max(floor, a), False


def _stop_landing_decel_floor(v_ego: float) -> float:
  """Soften only the final routine stop landing.

  The full stop-approach decel floor remains unchanged above the landing band.
  Below ``STOP_LANDING_SOFTEN_MAX_V_EGO`` ramp the floor toward a gentler hold so
  the last few mph do not carry a needless -1.5 m/s^2 command into standstill.
  """
  if not math.isfinite(v_ego) or v_ego >= STOP_LANDING_SOFTEN_MAX_V_EGO:
    return STOP_APPROACH_DECEL_MIN
  ratio = _clip(max(0.0, v_ego) / STOP_LANDING_SOFTEN_MAX_V_EGO, 0.0, 1.0)
  return STOP_LANDING_DECEL_MIN + ratio * (STOP_APPROACH_DECEL_MIN - STOP_LANDING_DECEL_MIN)


def comfort_relax_allowed(scene: LongitudinalScene) -> bool:
  return bool(not scene.has_lead and not scene.stop_threat and not scene.force_slow_decel
              and not scene.brake_pressed and not scene.gas_pressed)


def _scene_fields_finite(scene: LongitudinalScene) -> bool:
  """All lead kinematic fields used by the softening helper are finite."""
  return all(math.isfinite(v) for v in (
    scene.v_ego, scene.lead_d_rel, scene.lead_v_rel, scene.lead_v,
    scene.lead_a_target, scene.follow_gap,
  ))


def _lead_softening_target(scene: LongitudinalScene) -> float | None:
  """Return a bounded soft pre-MPC lead-follow target for low-risk far/medium leads.

  None means: keep the normal lead-follow PHYSICAL_HAZARD. The helper is deliberately
  conservative: it rejects anything close, anything requiring meaningful braking, any
  stop/override condition, and any non-finite or invalid raw lead kinematics.
  """
  if not scene.has_lead:
    return None
  if not (scene.lead_a_target < 0.0 and scene.lead_a_target >= _LEAD_SOFTEN_MAX_LEAD_DECEL):
    return None
  if scene.v_ego < _LEAD_SOFTEN_MIN_V_EGO:
    return None
  if scene.lead_v <= _LEAD_SOFTEN_MIN_LEAD_V:
    return None
  if not _scene_fields_finite(scene):
    return None
  if not scene.lead_kinematics_valid:
    return None
  if scene.lead_should_stop or scene.model_should_stop or scene.stop_threat:
    return None
  if scene.force_slow_decel or scene.brake_pressed or scene.gas_pressed:
    return None

  # near model stop: if a stop distance is present/non-finite or within the dynamic clearance
  if scene.model_stop_distance is not None:
    if not math.isfinite(scene.model_stop_distance):
      return None
    min_distance = max(_LEAD_SOFTEN_MIN_DISTANCE_BASE_M, _LEAD_SOFTEN_FOLLOW_TIME_GAP_S * scene.v_ego)
    if scene.model_stop_distance < min_distance:
      return None

  desired_gap = max(scene.follow_gap, _LEAD_SOFTEN_FOLLOW_TIME_GAP_S * scene.v_ego)
  min_distance = max(_LEAD_SOFTEN_MIN_DISTANCE_BASE_M, _LEAD_SOFTEN_FOLLOW_TIME_GAP_S * scene.v_ego)
  if scene.lead_d_rel < min_distance:
    return None

  usable_excess = scene.lead_d_rel - desired_gap
  if usable_excess <= _LEAD_SOFTEN_USABLE_EXCESS_MARGIN_M:
    return None

  closing = max(0.0, -scene.lead_v_rel)
  required = (closing * closing) / (2.0 * usable_excess)
  if required > _LEAD_SOFTEN_MAX_REQUIRED_DECEL:
    return None

  soft = min(_LEAD_SOFTEN_CEILING, scene.lead_a_target + _LEAD_SOFTEN_RAISE_DELTA)
  if soft <= scene.lead_a_target:
    return None
  # Final safety clamps: never positive, never harden the original target.
  return float(max(scene.lead_a_target, min(soft, 0.0)))


def _lead_relevance_cap(scene: LongitudinalScene) -> float | None:
  """Decel authority bound for far leads, from kinematically required decel.

  None means fully trusted: close leads (inside the softening trust floor), committed
  stops, and invalid kinematics all keep normal hazard authority. Applied with max(),
  so it can only soften a lead candidate, never harden it.
  """
  if not (scene.has_lead and scene.lead_kinematics_valid):
    return None
  if not _scene_fields_finite(scene):
    return None
  if scene.lead_should_stop or scene.model_should_stop or scene.stop_threat:
    return None
  min_distance = max(_LEAD_SOFTEN_MIN_DISTANCE_BASE_M, _LEAD_SOFTEN_FOLLOW_TIME_GAP_S * scene.v_ego)
  if scene.lead_d_rel <= min_distance:
    return None
  desired_gap = max(scene.follow_gap, _LEAD_SOFTEN_FOLLOW_TIME_GAP_S * scene.v_ego)
  usable_excess = scene.lead_d_rel - desired_gap
  if usable_excess <= _LEAD_SOFTEN_USABLE_EXCESS_MARGIN_M:
    return None
  closing = max(0.0, -scene.lead_v_rel)
  required = (closing * closing) / (2.0 * usable_excess)
  return float(-(_LEAD_RELEVANCE_AUTHORITY_K * required + _LEAD_RELEVANCE_COAST_MARGIN))


def _lead_inside_gap_recovery(scene: LongitudinalScene,
                              lead_gap_compression_active: bool = False) -> tuple[float, bool] | None:
  """For low/moderate-risk inside-gap compression/recovery, return (target, is_hazard).

  Returns:
    (0.0, False)      -> coast; caller should emit advisory cap + progress, no physical hazard.
    (negative, True)  -> gentle closing; caller should emit physical hazard at this target.
    None              -> not applicable; fall back to normal lead-follow hazard / softening.
  """
  if not (scene.has_lead and scene.lead_kinematics_valid):
    return None
  if not _scene_fields_finite(scene):
    return None
  if scene.lead_a_target >= 0.0:
    return None
  if scene.lead_should_stop or scene.model_should_stop or scene.stop_threat:
    return None
  if scene.force_slow_decel or scene.brake_pressed or scene.gas_pressed:
    return None
  if scene.lead_shadow_active or scene.alternate_threat_active:
    return None
  if not math.isfinite(scene.lead_confidence) or not scene.lead_stable or scene.lead_confidence < 0.55:
    return None
  if scene.v_ego < 5.0:
    return None
  if scene.lead_v < 3.0:
    return None
  if scene.lead_d_rel > scene.follow_gap + 2.0:
    return None

  closing = max(0.0, -scene.lead_v_rel)
  v_ego_safe = max(scene.v_ego, 0.1)
  time_gap = scene.lead_d_rel / v_ego_safe
  if closing > 0.05:
    ttc = scene.lead_d_rel / max(closing, 0.01)
  else:
    ttc = float("inf")
  usable_gap = max(scene.lead_d_rel - max(5.0, 0.5 * scene.follow_gap), 0.1)
  required_decel = (closing * closing) / (2.0 * usable_gap)
  collision_gap = max(scene.lead_d_rel - 5.0, 0.1)
  collision_required_decel = (closing * closing) / (2.0 * collision_gap)
  compression_threshold = (_LEAD_INSIDE_GAP_COMPRESSION_EXIT_MPS
                           if lead_gap_compression_active else _LEAD_INSIDE_GAP_COMPRESSION_ENTER_MPS)
  compression_allowed = closing > compression_threshold

  # Comfort tier: very low kinematic demand -> very mild compression target.
  comfort_ok = bool(
    time_gap >= _LEAD_INSIDE_GAP_MIN_TIME_GAP_S and
    ttc >= _LEAD_INSIDE_GAP_MIN_TTC_S and
    closing <= _LEAD_INSIDE_GAP_MAX_CLOSING_MPS and
    required_decel <= _LEAD_INSIDE_GAP_MAX_REQUIRED_DECEL and
    math.isfinite(scene.lead_a_k) and scene.lead_a_k >= _LEAD_INSIDE_GAP_MIN_LEAD_A_K
  )
  if comfort_ok:
    # Stable/opening inside the gap: suppress the physical hazard and coast.
    if not compression_allowed:
      return (0.0, False)
    raw_magnitude = max(
      0.15,
      min(0.45, required_decel + 0.10, 0.15 + 0.15 * closing),
    )
    return (float(max(scene.lead_a_target, -raw_magnitude)), True)

  # Routine tier: moderate desired-gap demand but collision risk still controlled.
  routine_ok = bool(
    time_gap >= _LEAD_ROUTINE_GAP_MIN_TIME_GAP_S and
    ttc >= _LEAD_ROUTINE_GAP_MIN_TTC_S and
    closing <= _LEAD_ROUTINE_GAP_MAX_CLOSING_MPS and
    required_decel <= _LEAD_ROUTINE_GAP_MAX_REQUIRED_DECEL and
    collision_required_decel <= _LEAD_ROUTINE_GAP_MAX_COLLISION_DECEL and
    math.isfinite(scene.lead_a_k) and scene.lead_a_k >= _LEAD_ROUTINE_GAP_MIN_LEAD_A_K
  )
  if routine_ok:
    if not compression_allowed:
      return (0.0, False)
    raw_magnitude = max(
      _LEAD_ROUTINE_GAP_TARGET_MIN,
      min(_LEAD_ROUTINE_GAP_TARGET_MAX, required_decel + 0.15, 0.20 + 0.25 * closing),
    )
    return (float(max(scene.lead_a_target, -raw_magnitude)), True)

  return None


def build_candidates(scene: LongitudinalScene,
                     lead_gap_compression_active: bool = False) -> list[LongitudinalCandidate]:
  """Produce the custom-2.0 candidate set; the decision core arbitrates and the mode gate
  admits them. Personality is normalized to a known value."""
  personality = scene.personality if isinstance(scene.personality, Personality) else Personality.from_value(scene.personality)
  scene = scene if scene.personality is personality else _with_personality(scene, personality)
  cands: list[LongitudinalCandidate] = []
  blocked = scene.force_slow_decel or scene.brake_pressed or scene.gas_pressed

  # baseline cruise desire (planner seed)
  cands.append(LongitudinalCandidate(float(scene.seed_a_target), CandidateRole.CRUISE, EvidenceClass.CRUISE, "driver_cruise"))

  # downhill free-coast leeway (raises a_target toward coast)
  coast_a = dynamic_cruise_coast_accel(scene, float(scene.seed_a_target))
  coast_candidate_active = coast_a > float(scene.seed_a_target)
  if coast_candidate_active:
    cands.append(LongitudinalCandidate(coast_a, CandidateRole.PROGRESS, EvidenceClass.CRUISE,
                                       "dynamic_overspeed_coast_leeway", authorized=True))

  # conservative uphill grade recovery: small authorized cruise-progress bump on clear grades
  recovery_a = uphill_grade_recovery_accel(scene)
  if recovery_a is not None and recovery_a > float(scene.seed_a_target):
    cands.append(LongitudinalCandidate(recovery_a, CandidateRole.PROGRESS, EvidenceClass.CRUISE,
                                       "uphill_grade_recovery", authorized=True))

  wants_progress = scene.v_cruise > scene.v_ego + PROGRESS_CRUISE_SPEED_MARGIN

  # no-lead launch pulse (personality-scaled, clear-path gated)
  if (not blocked and wants_progress and not scene.has_lead and not scene.stop_threat
      and scene.v_ego < NO_LEAD_LAUNCH_MAX_V_EGO and no_lead_stop_clear(scene)):
    cands.append(LongitudinalCandidate(launch_accel_max(personality), CandidateRole.PROGRESS,
                                       EvidenceClass.CRUISE, "no_lead_launch", authorized=True))

  # Lead pull-away progress (only when the lead context authorizes it), always capped by the
  # lead-aware speedup guard so a speed-up can never dig a hole that needs hard braking to climb
  # out of. Two regimes:
  #  - off-the-line behind an *opening* lead (stop-and-go / launch): key the pull-away on a
  #    lead-tracking launch accel so we follow the lead off the line instead of waiting for a 25 m
  #    gap to open. The gap_excess gate never fires at a close launch (desired gap is >=25 m, the
  #    launch gap is ~6 m), which is exactly the launch-hesitancy the hypermile tuning targets.
  #  - steady, far pull-away (a real gap excess): the gentle seed-based pull, unchanged.
  lead_opening = bool(scene.v_ego < LEAD_LAUNCH_MAX_V_EGO and scene.lead_v > LEAD_PULLAWAY_MIN_V_LEAD
                      and scene.lead_v_rel > LEAD_PULLAWAY_MIN_OPENING)
  opening_pullaway = bool(not blocked and wants_progress and scene.has_lead
                          and scene.lead_progress_allowed and lead_opening)
  far_pullaway = bool(not blocked and wants_progress and scene.has_lead
                      and scene.lead_progress_allowed and scene.lead_gap_excess > 0.0)
  if opening_pullaway or far_pullaway:
    if opening_pullaway:
      # Match the lead's speed over the launch time constant: gentle for a crawling lead, brisk
      # when it genuinely goes, never a fixed lurch. Capped by the personality launch accel.
      proposed = lead_pullaway_accel(scene, personality)
    else:
      proposed = max(0.0, float(scene.seed_a_target))
    pullaway = lead_speedup_guard(scene.v_ego, scene.lead_v, scene.lead_d_rel, scene.follow_gap, proposed)
    cands.append(LongitudinalCandidate(pullaway, CandidateRole.PROGRESS,
                                       EvidenceClass.LEAD, "lead_pullaway", authorized=True))

  # Phase 1 bidirectional lead-speed alignment: additional guarded advisory/progress candidates.
  # The existing physical hazard and lead_pullaway above are preserved; these only add behavior.
  alignment = lead_speed_alignment(
    v_ego=scene.v_ego, a_ego=scene.a_ego, v_cruise=scene.v_cruise,
    lead_d_rel=scene.lead_d_rel, lead_v=scene.lead_v, lead_v_rel=scene.lead_v_rel,
    lead_a_k=scene.lead_a_k, follow_gap=scene.follow_gap,
    lead_confidence=scene.lead_confidence, lead_stable=scene.lead_stable,
    lead_progress_allowed=scene.lead_progress_allowed,
    lead_shadow_active=scene.lead_shadow_active,
    alternate_threat_active=scene.alternate_threat_active,
    model_should_stop=scene.model_should_stop,
    force_slow_decel=scene.force_slow_decel,
    brake_pressed=scene.brake_pressed, gas_pressed=scene.gas_pressed,
    personality=personality, lead_kinematics_valid=scene.lead_kinematics_valid,
    has_lead=scene.has_lead, a_lead_tau=scene.lead_a_tau,
  )
  if alignment.action is AlignmentAction.COAST:
    cands.append(LongitudinalCandidate(alignment.a_target, CandidateRole.ADVISORY_CAP,
                                       EvidenceClass.LEAD, "lead_alignment_coast"))
  elif alignment.action is AlignmentAction.GENTLE_BRAKE:
    cands.append(LongitudinalCandidate(alignment.a_target, CandidateRole.ADVISORY_CAP,
                                       EvidenceClass.LEAD, "lead_alignment_gentle_brake"))
  elif alignment.action is AlignmentAction.PULLAWAY:
    cands.append(LongitudinalCandidate(alignment.a_target, CandidateRole.PROGRESS,
                                       EvidenceClass.LEAD, "lead_pullaway_alignment", authorized=True))
  elif alignment.action is AlignmentAction.STANDSTILL_LAUNCH:
    cands.append(LongitudinalCandidate(alignment.a_target, CandidateRole.PROGRESS,
                                       EvidenceClass.LEAD, "lead_standstill_launch", authorized=True))
  alignment_pullaway = alignment.action in (AlignmentAction.PULLAWAY, AlignmentAction.STANDSTILL_LAUNCH)

  # comfort relax: soften advisory braking when clear
  if comfort_relax_allowed(scene):
    cands.append(LongitudinalCandidate(COMFORT_RELAX_ACCEL_MIN, CandidateRole.COMFORT_RELAX,
                                       EvidenceClass.CRUISE, "comfort_relax"))

  # Lead-present pre-MPC seed: before the MPC solves final lead physics, a braking lead-present
  # seed normally binds as a physical hazard. Phase 4 allows only low-risk far/medium cases to be
  # softened into a bounded advisory/desire pair; final MPC lead output remains authoritative
  # downstream. In the authorized launch case, when the seed is non-braking, the non-braking seed
  # imposes no floor so the speedup-guarded launch pulse can follow an opening lead off the line.
  # (A positive "hazard" is not a hazard; it would otherwise clamp the PROGRESS layer that is
  # designed to raise accel when authorized. The instant the seed goes negative and is not a
  # low-risk soft case, it re-binds.)
  if (scene.has_lead and not coast_candidate_active
      and (scene.lead_a_target < 0.0 or not (opening_pullaway or alignment_pullaway))):
    recovery = _lead_inside_gap_recovery(scene, lead_gap_compression_active)
    if recovery is not None:
      target, is_hazard = recovery
      if is_hazard:
        # Gentle inside-gap closing: keep a binding hazard but cap it, never harden the seed.
        cands.append(LongitudinalCandidate(float(target), CandidateRole.PHYSICAL_HAZARD,
                                           EvidenceClass.LEAD, "lead_gap_compression",
                                           is_stop=bool(scene.lead_should_stop)))
        # Express the coast/recovery desire so the capped hazard binds instead of a more
        # negative cruise seed.
        cands.append(LongitudinalCandidate(0.0, CandidateRole.PROGRESS,
                                           EvidenceClass.LEAD, "lead_gap_compression_desire", authorized=True))
      else:
        # Stable/opening inside the gap: suppress the physical hazard and coast.
        cands.append(LongitudinalCandidate(float(target), CandidateRole.ADVISORY_CAP,
                                           EvidenceClass.LEAD, "lead_gap_recovery_coast"))
        cands.append(LongitudinalCandidate(float(target), CandidateRole.PROGRESS,
                                           EvidenceClass.LEAD, "lead_gap_recovery_coast_desire", authorized=True))
    else:
      soft_target = _lead_softening_target(scene)
      if soft_target is not None:
        # Phase 4: low-risk far/medium lead-follow decel is softened from a binding hazard to an
        # advisory cap plus an authorized LEAD progress/desire candidate at the same target. This
        # raises the pre-MPC seed when it equals the lead-influenced target, while other caps
        # (curve/SLA/model stop) and the lead-following cushion remain free to bind lower.
        cands.append(LongitudinalCandidate(soft_target, CandidateRole.ADVISORY_CAP,
                                           EvidenceClass.LEAD, "lead_follow_soft"))
        cands.append(LongitudinalCandidate(soft_target, CandidateRole.PROGRESS,
                                           EvidenceClass.LEAD, "lead_follow_soft_desire", authorized=True))
      else:
        cands.append(LongitudinalCandidate(float(scene.lead_a_target), CandidateRole.PHYSICAL_HAZARD,
                                           EvidenceClass.LEAD, "lead_follow", is_stop=bool(scene.lead_should_stop)))
    # lead-following cushion: anticipatory gentle coast to a slower moving lead while runway
    # allows, as an advisory cap. The MPC hazard above still binds when it must brake harder
    # (tighten = safe; this only relaxes the approach, never overrides the physics floor).
    if scene.lead_v > 0.0 and scene.lead_d_rel > 0.0 and scene.follow_gap > 0.0:
      cushion = lead_following_cushion(scene.v_ego, scene.lead_v, scene.lead_d_rel, scene.follow_gap,
                                       coast_decel=_usable_coast_decel(scene))
      if cushion.coast_first and cushion.a_target < 0.0:
        cands.append(LongitudinalCandidate(cushion.a_target, CandidateRole.ADVISORY_CAP,
                                           EvidenceClass.LEAD, "lead_cushion"))
    # hard-braker rest-point coast (see LEAD_REST_POINT_* constants): treat a persistently
    # hard-braking lead as a future constraint at its rest distance and lift off per the
    # coast horizon. Emitted only inside the lift window (CRUISE = too far, stays silent);
    # clamped to [natural coast, 0] so it can only request lift-off, never braking.
    if (scene.lead_hard_brake_s >= LEAD_REST_POINT_PERSIST_S and scene.lead_v > 0.0
        and scene.lead_d_rel > 0.0 and scene.lead_a_k < LEAD_REST_POINT_HARD_BRAKE_A):
      rest_distance = scene.lead_d_rel + scene.lead_v ** 2 / (2.0 * abs(scene.lead_a_k)) - LEAD_REST_POINT_MARGIN_M
      if rest_distance > 0.0:
        coast = _usable_coast_decel(scene)
        horizon = coast_horizon(CoastHorizonInputs(
          v_ego=scene.v_ego, v_target=0.0, distance_to_constraint=rest_distance, accel_coast=coast,
        ))
        if horizon.action is not CoastAction.CRUISE:
          cands.append(LongitudinalCandidate(min(0.0, max(horizon.recommended_accel, coast)),
                                             CandidateRole.ADVISORY_CAP, EvidenceClass.LEAD,
                                             "lead_rest_point_coast"))

  # model stop-approach hazard (E2E/SCC only via the mode gate), trust-gated: a low-confidence
  # or uncorroborated model stop is softened toward a gentle precautionary decel and is not
  # committed as a stop until trusted (anti-quirk: don't slam on a flickery model stop).
  if scene.model_should_stop or (scene.model_stop_distance is not None and scene.model_stop_distance > 0.0):
    if scene.model_stale:
      cands.append(LongitudinalCandidate(GENTLE_CAUTION_DECEL, CandidateRole.PHYSICAL_HAZARD,
                                         EvidenceClass.MODEL_STOP, "stop_approach", is_stop=False))
    else:
      trust = gate_model_stop(scene.model_should_stop, scene.model_desired_accel, scene.model_stop_prob,
                              has_radar_lead=scene.has_lead, lead_v_rel=scene.lead_v_rel,
                              model_stale=scene.model_stale)
      trusted_scene = replace(scene, model_should_stop=trust.should_stop, model_desired_accel=trust.desired_accel)
      stop_a, hard = stop_approach_accel(trusted_scene)
      # Coast-first for a long stop runway: only for a COMMITTED (trusted) stop, so the trust
      # gate's softening of a low-confidence stop is never re-hardened by stop kinematics.
      if trust.should_stop and not hard and scene.model_stop_distance is not None and scene.model_stop_distance > 0.0:
        stop_a = runway_comfort_governor(scene.v_ego, 0.0, scene.model_stop_distance, stop_a,
                                         _usable_coast_decel(scene))
      # Non-committed stop decel may not outrun the earned caution floor: a stop distance
      # that flickers in for one frame no longer bangs from gentle straight to the stop
      # floor. Trusted committed stops keep full stop physics.
      if not trust.should_stop:
        stop_a = max(stop_a, scene.model_caution_floor)
      cands.append(LongitudinalCandidate(stop_a, CandidateRole.PHYSICAL_HAZARD, EvidenceClass.MODEL_STOP,
                                         "stop_approach", is_stop=bool(trust.should_stop and hard)))
  # Early, non-committing model-slowdown caution from fresh model decel before shouldStop or
  # stop distance are available. Bounded to be no stronger than the existing precautionary
  # decel and never committed as a stop.
  elif (not scene.model_stale and scene.model_stop_distance is None
        and math.isfinite(scene.model_desired_accel)
        and scene.model_desired_accel < EARLY_MODEL_SLOWDOWN_DECEL_THRESHOLD):
    early_a = max(scene.model_desired_accel, scene.model_caution_floor)
    if _uncorroborated_far_lead(scene):
      early_a = max(early_a, _FAR_LEAD_CAUTION_DECEL_CAP)
    cands.append(LongitudinalCandidate(early_a, CandidateRole.PHYSICAL_HAZARD,
                                       EvidenceClass.MODEL_STOP, "stop_approach", is_stop=False))

  # advisory caps
  if scene.speed_limit_active and scene.speed_limit_v_target > 0.0 and scene.speed_limit_v_target < scene.v_ego:
    raw_cap = min(0.0, max(scene.speed_limit_a_target, scene.accel_coast))
    shaped_cap = _advisory_speed_limit_cap(scene)
    cands.append(LongitudinalCandidate(shaped_cap, CandidateRole.ADVISORY_CAP,
                                       EvidenceClass.SPEED_LIMIT, "speed_policy"))
    # Source-local desire: if the runway-shaped cap is gentler than the raw source cap,
    # raise the baseline so decide() reflects the shaped target even when the seed already
    # came from this source. Hazards still bind lower.
    if shaped_cap > raw_cap:
      cands.append(LongitudinalCandidate(shaped_cap, CandidateRole.PROGRESS,
                                         EvidenceClass.SPEED_LIMIT, "speed_policy_desire", authorized=True))
  if scene.curve_active:
    raw_cap = float(scene.curve_a_target)
    shaped_cap = _advisory_curve_cap(scene)
    cands.append(LongitudinalCandidate(shaped_cap, CandidateRole.ADVISORY_CAP,
                                       scene.curve_source, "curve_policy"))
    if shaped_cap > raw_cap:
      cands.append(LongitudinalCandidate(shaped_cap, CandidateRole.PROGRESS,
                                         scene.curve_source, "curve_policy_desire", authorized=True))
  # map-coast tier: lift-off-only cap toward a map slowdown beyond vision range. Admissibility
  # (SCC mode + SmartCruiseControlMap toggle) is enforced by decide() on CURVE_MAP evidence.
  if scene.map_coast_active:
    coast_cap = map_coast_cap(scene)
    if coast_cap is not None:
      cands.append(LongitudinalCandidate(coast_cap, CandidateRole.ADVISORY_CAP,
                                         EvidenceClass.CURVE_MAP, "map_coast"))

  # force-slow safety hazard (driver/system force)
  if scene.force_slow_decel:
    cands.append(LongitudinalCandidate(min(float(scene.seed_a_target), SAFETY_FORCE_SLOW_DECEL),
                                       CandidateRole.PHYSICAL_HAZARD, EvidenceClass.CRUISE, "force_slow", is_stop=True))

  # Far-lead relevance cap: every lead-evidence decel candidate (raw hazard, softened pair,
  # cushion) is bounded by a multiple of the decel the closing kinematics require. One shared
  # sweep so no lead path can out-brake the geometry on an uncorroborated far lead.
  relevance_cap = _lead_relevance_cap(scene)
  if relevance_cap is not None:
    cands = [replace(c, a_target=relevance_cap) if c.source is EvidenceClass.LEAD and c.a_target < relevance_cap
             else c for c in cands]

  return cands


def _with_personality(scene: LongitudinalScene, personality: Personality) -> LongitudinalScene:
  return replace(scene, personality=personality)


def _scene_coast_decel(scene: LongitudinalScene) -> float:
  """Honest natural-coast estimate for the advisory runway governors (speed-limit/curve): when no
  useful coast is measured, report MAX_COAST_DECEL so a fabricated proxy never relaxes those caps."""
  return scene.accel_coast if scene.accel_coast < MAX_COAST_DECEL else MAX_COAST_DECEL


def _usable_coast_decel(scene: LongitudinalScene) -> float:
  """Coast decel for lead-cushion and committed-stop shaping.

  Use the flat-road proxy only when pitch is unavailable; otherwise keep the honest clamped
  coast estimate so downhill pitch does not get treated like flat-road drag.
  """
  if scene.pitch is None or not math.isfinite(scene.pitch):
    return scene.accel_coast if scene.accel_coast < MAX_COAST_DECEL else DEFAULT_COAST_DECEL
  return _scene_coast_decel(scene)


def _advisory_speed_limit_cap(scene: LongitudinalScene) -> float:
  """Speed-limit advisory cap shaped by the runway comfort governor when distance is known."""
  if not (scene.speed_limit_active and scene.speed_limit_v_target > 0.0 and scene.speed_limit_v_target < scene.v_ego):
    return min(0.0, max(scene.speed_limit_a_target, scene.accel_coast))
  if scene.speed_limit_distance is not None and scene.speed_limit_distance > 0.0:
    shaped = runway_comfort_governor(
      scene.v_ego, scene.speed_limit_v_target, scene.speed_limit_distance,
      scene.speed_limit_a_target, _scene_coast_decel(scene),
    )
    return min(0.0, shaped)
  return min(0.0, max(scene.speed_limit_a_target, scene.accel_coast))


def map_coast_cap(scene: LongitudinalScene) -> float | None:
  """Coast-only advisory cap toward an SCC-Map slowdown beyond vision range.

  Lifts off early enough that natural coast bleeds speed to the map target, and is floored at
  the natural coast decel — map evidence alone never brakes (worst case with wrong map data is
  a mild unneeded coast). None while cruising is still fine or the target is invalid."""
  if not (scene.map_coast_v_target > 0.0 and scene.map_coast_distance is not None
          and scene.map_coast_distance > 0.0 and scene.map_coast_v_target < scene.v_ego):
    return None
  coast = _usable_coast_decel(scene)
  r = coast_horizon(CoastHorizonInputs(
    v_ego=scene.v_ego, v_target=scene.map_coast_v_target,
    distance_to_constraint=scene.map_coast_distance, accel_coast=coast,
  ))
  if r.action is CoastAction.CRUISE:
    return None
  return min(0.0, max(r.recommended_accel, coast))


def _advisory_curve_cap(scene: LongitudinalScene) -> float:
  """Curve advisory cap shaped by the runway comfort governor when target speed and runway are known."""
  if not scene.curve_active:
    return float(scene.curve_a_target)
  if (scene.curve_distance is not None and scene.curve_distance > 0.0
      and scene.curve_v_target > 0.0 and scene.curve_v_target < scene.v_ego):
    shaped = runway_comfort_governor(
      scene.v_ego, scene.curve_v_target, scene.curve_distance,
      scene.curve_a_target, _scene_coast_decel(scene),
    )
    return min(0.0, shaped)
  return float(scene.curve_a_target)
