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
from openpilot.sunnypilot.custom.longitudinal.model_trust import gate_model_stop
from openpilot.sunnypilot.custom.longitudinal.modes import EvidenceClass
from openpilot.sunnypilot.custom.longitudinal.runway_governor import runway_comfort_governor
from openpilot.sunnypilot.custom.longitudinal.policy_tables import (
  COMFORT_RELAX_ACCEL_MIN,
  CRUISE_LEEWAY_DOWNHILL_ACCEL,
  CRUISE_LEEWAY_MAX,
  CRUISE_LEEWAY_MIN,
  CRUISE_LEEWAY_RECOVERY,
  NO_LEAD_LAUNCH_MAX_V_EGO,
  NO_LEAD_STOP_CLEAR_ACCEL_MIN,
  NO_LEAD_STOP_CLEAR_DISTANCE,
  PROGRESS_CRUISE_SPEED_MARGIN,
  STOP_APPROACH_DECEL_MIN,
  Personality,
  launch_accel_max,
  stop_approach_comfort_decel,
)

SAFETY_FORCE_SLOW_DECEL = -0.2


@dataclass(frozen=True)
class LongitudinalScene:
  v_ego: float
  v_cruise: float
  seed_a_target: float          # planner baseline (MPC cruise/seed) accel
  accel_coast: float = 0.0      # current coast-down accel (negative downhill)
  personality: Personality = Personality.STANDARD
  # lead (MPC physical hazard)
  has_lead: bool = False
  lead_a_target: float = 0.0
  lead_should_stop: bool = False
  lead_gap_excess: float = 0.0
  lead_progress_allowed: bool = False
  # lead kinematics (for cushion / speedup guard / radar corroboration)
  lead_v: float = 0.0
  lead_d_rel: float = 0.0
  lead_v_rel: float = 0.0
  follow_gap: float = 0.0
  # model stop (E2E)
  model_should_stop: bool = False
  model_stop_distance: float | None = None
  model_desired_accel: float = 0.0
  model_stop_prob: float = 1.0   # model confidence in the stop (trust gate); 1.0 = fully trusted
  stop_threat: bool = False
  # advisory sources
  speed_limit_active: bool = False
  speed_limit_v_target: float = 0.0
  speed_limit_a_target: float = 0.0
  map_caution_active: bool = False
  map_caution_confirmed: bool = False
  map_caution_a_target: float = 0.0
  curve_active: bool = False
  curve_a_target: float = 0.0
  # driver / safety
  force_slow_decel: bool = False
  brake_pressed: bool = False
  gas_pressed: bool = False


def _clip(value: float, lo: float, hi: float) -> float:
  return max(lo, min(hi, value))


def stopping_decel(v_ego: float, distance: float, min_distance: float = 1.0) -> float:
  """Signed constant accel required to stop over distance (kinematic)."""
  return -(float(v_ego) ** 2) / (2.0 * max(float(distance), min_distance))


def dynamic_cruise_overspeed_leeway(accel_coast: float) -> float:
  confidence = _clip(float(accel_coast) / CRUISE_LEEWAY_DOWNHILL_ACCEL, 0.0, 1.0)
  return CRUISE_LEEWAY_MIN + confidence * (CRUISE_LEEWAY_MAX - CRUISE_LEEWAY_MIN)


def dynamic_cruise_coast_accel(scene: LongitudinalScene, a_target: float) -> float:
  """Let speed bleed off naturally (coast) instead of braking, within a downhill-scaled
  overspeed leeway — the hypermile free-coast behavior."""
  if scene.has_lead or scene.stop_threat or scene.force_slow_decel or scene.v_cruise <= 0.0:
    return a_target
  overspeed = scene.v_ego - scene.v_cruise
  if overspeed <= 0.0:
    return a_target
  leeway = dynamic_cruise_overspeed_leeway(scene.accel_coast)
  if overspeed <= leeway:
    return min(0.0, max(a_target, scene.accel_coast))
  recovery = _clip((overspeed - leeway) / CRUISE_LEEWAY_RECOVERY, 0.0, 1.0)
  coast_target = (1.0 - recovery) * min(0.0, scene.accel_coast) + recovery * a_target
  return min(0.0, max(a_target, coast_target))


def no_lead_stop_clear(scene: LongitudinalScene) -> bool:
  d = scene.model_stop_distance
  return bool(not scene.model_should_stop and (d is None or d > NO_LEAD_STOP_CLEAR_DISTANCE)
              and scene.model_desired_accel >= NO_LEAD_STOP_CLEAR_ACCEL_MIN)


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
  return max(STOP_APPROACH_DECEL_MIN, a), False


def comfort_relax_allowed(scene: LongitudinalScene) -> bool:
  return bool(not scene.has_lead and not scene.stop_threat and not scene.force_slow_decel
              and not scene.brake_pressed and not scene.gas_pressed
              and not (scene.map_caution_active and scene.map_caution_confirmed))


def build_candidates(scene: LongitudinalScene) -> list[LongitudinalCandidate]:
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
  if coast_a > float(scene.seed_a_target):
    cands.append(LongitudinalCandidate(coast_a, CandidateRole.PROGRESS, EvidenceClass.CRUISE,
                                       "dynamic_overspeed_coast_leeway", authorized=True))

  wants_progress = scene.v_cruise > scene.v_ego + PROGRESS_CRUISE_SPEED_MARGIN

  # no-lead launch pulse (personality-scaled, clear-path gated)
  if (not blocked and wants_progress and not scene.has_lead and not scene.stop_threat
      and scene.v_ego < NO_LEAD_LAUNCH_MAX_V_EGO and no_lead_stop_clear(scene)):
    cands.append(LongitudinalCandidate(launch_accel_max(personality), CandidateRole.PROGRESS,
                                       EvidenceClass.CRUISE, "no_lead_launch", authorized=True))

  # lead pullaway progress (only when authorized), capped by the lead-aware speedup guard so a
  # speed-up can never dig a hole that needs hard braking to climb out of.
  if not blocked and wants_progress and scene.has_lead and scene.lead_progress_allowed and scene.lead_gap_excess > 0.0:
    pullaway = lead_speedup_guard(scene.v_ego, scene.lead_v, scene.lead_d_rel, scene.follow_gap,
                                  max(0.0, float(scene.seed_a_target)))
    cands.append(LongitudinalCandidate(pullaway, CandidateRole.PROGRESS,
                                       EvidenceClass.LEAD, "lead_pullaway", authorized=True))

  # comfort relax: soften advisory braking when clear
  if comfort_relax_allowed(scene):
    cands.append(LongitudinalCandidate(COMFORT_RELAX_ACCEL_MIN, CandidateRole.COMFORT_RELAX,
                                       EvidenceClass.CRUISE, "comfort_relax"))

  # lead-follow hazard (MPC owns the physics; we carry its a_target as the binding decel)
  if scene.has_lead:
    cands.append(LongitudinalCandidate(float(scene.lead_a_target), CandidateRole.PHYSICAL_HAZARD,
                                       EvidenceClass.LEAD, "lead_follow", is_stop=bool(scene.lead_should_stop)))
    # lead-following cushion: anticipatory gentle coast to a slower moving lead while runway
    # allows, as an advisory cap. The MPC hazard above still binds when it must brake harder
    # (tighten = safe; this only relaxes the approach, never overrides the physics floor).
    if scene.lead_v > 0.0 and scene.lead_d_rel > 0.0 and scene.follow_gap > 0.0:
      cushion = lead_following_cushion(scene.v_ego, scene.lead_v, scene.lead_d_rel, scene.follow_gap,
                                       coast_decel=_scene_coast_decel(scene))
      if cushion.coast_first and cushion.a_target < 0.0:
        cands.append(LongitudinalCandidate(cushion.a_target, CandidateRole.ADVISORY_CAP,
                                           EvidenceClass.LEAD, "lead_cushion"))

  # model stop-approach hazard (E2E/SCC only via the mode gate), trust-gated: a low-confidence
  # or uncorroborated model stop is softened toward a gentle precautionary decel and is not
  # committed as a stop until trusted (anti-quirk: don't slam on a flickery model stop).
  if scene.model_should_stop or (scene.model_stop_distance is not None and scene.model_stop_distance > 0.0):
    trust = gate_model_stop(scene.model_should_stop, scene.model_desired_accel, scene.model_stop_prob,
                            has_radar_lead=scene.has_lead, lead_v_rel=scene.lead_v_rel)
    trusted_scene = replace(scene, model_should_stop=trust.should_stop, model_desired_accel=trust.desired_accel)
    stop_a, hard = stop_approach_accel(trusted_scene)
    # Coast-first for a long stop runway: only for a COMMITTED (trusted) stop, so the trust
    # gate's softening of a low-confidence stop is never re-hardened by stop kinematics.
    if trust.should_stop and not hard and scene.model_stop_distance is not None and scene.model_stop_distance > 0.0:
      stop_a = runway_comfort_governor(scene.v_ego, 0.0, scene.model_stop_distance, stop_a,
                                       _scene_coast_decel(scene))
    cands.append(LongitudinalCandidate(stop_a, CandidateRole.PHYSICAL_HAZARD, EvidenceClass.MODEL_STOP,
                                       "stop_approach", is_stop=bool(trust.should_stop and hard)))

  # advisory caps
  if scene.speed_limit_active and scene.speed_limit_v_target > 0.0 and scene.speed_limit_v_target < scene.v_ego:
    cap = min(0.0, max(scene.speed_limit_a_target, scene.accel_coast))  # coast-biased
    cands.append(LongitudinalCandidate(cap, CandidateRole.ADVISORY_CAP, EvidenceClass.SPEED_LIMIT, "speed_policy"))
  if scene.map_caution_active and scene.map_caution_confirmed:
    cands.append(LongitudinalCandidate(min(0.0, scene.map_caution_a_target), CandidateRole.ADVISORY_CAP,
                                       EvidenceClass.MAP_CAUTION, "map_caution"))
  if scene.curve_active:
    cands.append(LongitudinalCandidate(float(scene.curve_a_target), CandidateRole.ADVISORY_CAP,
                                       EvidenceClass.CURVE_VISION, "curve_policy"))

  # force-slow safety hazard (driver/system force)
  if scene.force_slow_decel:
    cands.append(LongitudinalCandidate(min(float(scene.seed_a_target), SAFETY_FORCE_SLOW_DECEL),
                                       CandidateRole.PHYSICAL_HAZARD, EvidenceClass.CRUISE, "force_slow", is_stop=True))

  return cands


def _with_personality(scene: LongitudinalScene, personality: Personality) -> LongitudinalScene:
  return replace(scene, personality=personality)


def _scene_coast_decel(scene: LongitudinalScene) -> float:
  """A usable (negative) natural coast decel for the cushion; falls back to a flat-road proxy."""
  return scene.accel_coast if scene.accel_coast < -0.02 else -0.25
