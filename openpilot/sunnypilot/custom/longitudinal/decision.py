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


# Intent-hold margin (m/s^2). `selected_intent` is a *label* for whichever candidate happened
# to win this frame's min/max. When two candidates sit within noise of each other the label
# flips every frame even though the commanded target barely moves — route 000002dc measured
# 94 intent changes/min (top pairs lead_follow<->cruise, stop_approach<->cruise), and several
# downstream stages branch on the label, so the flapping becomes real shaping churn.
#
# The hold suppresses only the label flip: it keeps the incumbent intent while that candidate
# is STILL LIVE and STILL asking for within-margin what we are already commanding. a_target,
# should_stop and reason are always computed truthfully, so a candidate that disappears or
# diverges takes over immediately — real transitions are never delayed, only noise-crossings.
#
# Margin sweep over udacity-acc + openpilot-acc (30 scenarios, 16.8 min, baseline 9.9 churn/min),
# measuring churn against whether the commanded accel diverges at all:
#     margin   churn/min   delta    scenarios w/ cmd change   max |dA|
#       0.02        9.3    -6.0%                          0     0.0000
#       0.05        9.0    -9.0%                          0     0.0000
#       0.10        8.8   -11.4%                          1     0.0008
#       0.20        8.5   -13.9%                          5     0.2327
#       0.40        7.5   -24.7%                          5     0.2327
# 0.05 is the last strictly command-neutral point, so it is what ships: the label stops
# flapping and not one commanded sample moves. Raising it past ~0.10 buys more churn
# reduction by actually changing behavior through the downstream label branches — that
# needs engaged-route evidence first, not a synthetic gate.
INTENT_HOLD_MARGIN = 0.05


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
           sources: SourceToggles = SourceToggles(), previous_intent: str = "") -> Decision:
  """Arbitrate candidates into a single a_target. Fail-closed: bad inputs yield the
  conservative accel-limit floor, never an unsafe value.

  ``previous_intent`` is last frame's ``selected_intent``; passing it enables the
  INTENT_HOLD_MARGIN label hold. It never changes the arbitrated a_target."""
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
  # Prefer progress on a tie because its evidence is what authorizes a standstill release.
  desire_winner = max(cruise + progress,
                      key=lambda c: (c.a_target, c.role is CandidateRole.PROGRESS), default=None)
  desire = desire_winner.a_target if desire_winner is not None else min(0.0, a_max)

  # Advisory caps restrict accel from above; comfort relax softens the binding cap toward its
  # (higher) floor but never removes it.
  advisory_cap = min((c.a_target for c in caps), default=a_max)
  if relaxes and caps:
    relax_floor = max(c.a_target for c in relaxes)
    advisory_cap = max(advisory_cap, min(relax_floor, a_max))

  a_target = min(desire, advisory_cap)

  # Physical hazards always bind: the strongest decel wins and policy cannot raise above it.
  selected_intent = (desire_winner.intent or "progress") if (
    desire_winner is not None and desire_winner.role is CandidateRole.PROGRESS
  ) else "cruise"
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

  # Label hold: keep the incumbent intent while its candidate is still live and still wants
  # within INTENT_HOLD_MARGIN of what we are commanding anyway. Only the label is held —
  # a_target/should_stop/reason above are already final and stay truthful.
  if previous_intent and previous_intent != selected_intent:
    incumbent = next((c for c in usable if c.intent == previous_intent), None)
    if incumbent is not None and abs(float(incumbent.a_target) - a_target) <= INTENT_HOLD_MARGIN:
      selected_intent = previous_intent

  return Decision(
    a_target=float(a_target),
    should_stop=bool(should_stop),
    selected_intent=selected_intent,
    reason=reason,
    admitted_sources=admitted,
    rejected=tuple(rejected),
  )
