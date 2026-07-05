"""Longitudinal decision core — candidate/authority arbitration (first cut).

The clean replacement for the legacy decision-core/policy duality (longitudinal_decision.py
+ custom_v2.py). It arbitrates longitudinal *candidates* by authority within the active
mode's admitted evidence, and never lets policy override safety:

    PHYSICAL_HAZARD  binds decel (Lead MPC / E2E stop) — always respected
    ADVISORY_CAP     restricts accel from above (curve / map caution)
    COMFORT_RELAX    softens an advisory cap toward a floor (never below a hazard)
    PROGRESS         raises accel only when authorized (launch / pullaway)
    CRUISE           the set-speed baseline desire

This is the structure + the safety invariants (a candidate may only restrict, a physical
hazard always binds, mode-excluded evidence can never act, fail-closed on bad input). The
detailed per-intent tuning that carries the hypermile *feel* (the ~72 legacy constants,
comfort curves, coast-leeway shaping) is DEFERRED to engaged-route replay tuning, exactly as
the torque governor's comfort behaviors are — see
``docs/adr/2026-06-13-clean-room-longitudinal-architecture.md``. Promotion to default-on
waits for engaged parity.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

from openpilot.sunnypilot.custom.longitudinal.modes import (
  EvidenceClass,
  LongitudinalMode,
  SourceToggles,
  admitted_evidence,
)


class CandidateRole(Enum):
  PHYSICAL_HAZARD = "physical_hazard"  # binds decel; only restricts
  ADVISORY_CAP = "advisory_cap"        # restricts accel from above
  COMFORT_RELAX = "comfort_relax"      # softens an advisory cap toward a floor
  PROGRESS = "progress"                # raises accel only when authorized
  CRUISE = "cruise"                    # set-speed baseline desire


@dataclass(frozen=True)
class LongitudinalCandidate:
  a_target: float
  role: CandidateRole
  source: EvidenceClass
  intent: str = ""
  authorized: bool = True
  is_stop: bool = False  # a confirmed stop hazard (sets should_stop)


@dataclass(frozen=True)
class Decision:
  a_target: float
  should_stop: bool
  selected_intent: str
  reason: str
  admitted_sources: frozenset = field(default_factory=frozenset)
  rejected: tuple = ()


def _finite(value: object) -> bool:
  try:
    return math.isfinite(float(value))  # type: ignore[arg-type]
  except (TypeError, ValueError):
    return False


def decide(candidates: list[LongitudinalCandidate], mode: LongitudinalMode, accel_limits: tuple[float, float],
           sources: SourceToggles = SourceToggles()) -> Decision:
  """Arbitrate candidates into a single a_target. Fail-closed: bad inputs yield the
  conservative accel-limit floor, never an unsafe value."""
  a_min, a_max = (float(accel_limits[0]), float(accel_limits[1]))
  if not (_finite(a_min) and _finite(a_max)) or a_min > a_max:
    return Decision(0.0, False, "fault", "invalid_accel_limits")

  admitted = admitted_evidence(mode, sources)
  usable: list[LongitudinalCandidate] = []
  rejected: list[str] = []
  for c in candidates:
    if not _finite(c.a_target):
      rejected.append(f"{c.intent or c.role.value}:nonfinite")
      continue
    if c.source not in admitted:
      rejected.append(f"{c.intent or c.role.value}:mode_excluded")
      continue
    usable.append(c)

  hazards = [c for c in usable if c.role is CandidateRole.PHYSICAL_HAZARD]
  caps = [c for c in usable if c.role is CandidateRole.ADVISORY_CAP]
  relaxes = [c for c in usable if c.role is CandidateRole.COMFORT_RELAX]
  progress = [c for c in usable if c.role is CandidateRole.PROGRESS and c.authorized]
  cruise = [c for c in usable if c.role is CandidateRole.CRUISE]

  # Baseline desire: the highest accel any cruise/authorized-progress candidate asks for.
  desire = max((c.a_target for c in cruise + progress), default=min(0.0, a_max))

  # Advisory caps restrict accel from above; comfort relax softens the binding cap toward its
  # (higher) floor but never removes it.
  advisory_cap = min((c.a_target for c in caps), default=a_max)
  if relaxes and caps:
    relax_floor = max(c.a_target for c in relaxes)
    advisory_cap = max(advisory_cap, min(relax_floor, a_max))

  a_target = min(desire, advisory_cap)

  # Physical hazards always bind: the strongest decel wins and policy cannot raise above it.
  selected_intent = "cruise" if not progress else (progress[0].intent or "progress")
  reason = "cruise" if not caps else "advisory_capped"
  if hazards:
    hazard_a = min(c.a_target for c in hazards)
    if hazard_a < a_target:
      binding = min(hazards, key=lambda c: c.a_target)
      a_target = hazard_a
      selected_intent = binding.intent or "physical_hazard"
      reason = "physical_hazard"

  should_stop = any(c.is_stop for c in hazards)
  a_target = min(max(a_target, a_min), a_max)
  return Decision(
    a_target=float(a_target),
    should_stop=bool(should_stop),
    selected_intent=selected_intent,
    reason=reason,
    admitted_sources=admitted,
    rejected=tuple(rejected),
  )
