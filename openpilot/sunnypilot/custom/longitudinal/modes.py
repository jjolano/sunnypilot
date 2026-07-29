"""Longitudinal modes — the ACC / E2E / SCC evidence-admission gate.

The single authority that decides which *classes of evidence* may reach actuation, before
any policy candidate is built. Latched per onroad cycle, it sits above the policy overlay;
nothing downstream re-admits what the mode excluded. See
``docs/adr/2026-06-13-clean-room-longitudinal-architecture.md`` and
``docs/legacy/CONTEXT-longitudinal.md`` (Longitudinal Mode / SCC Mode / SCC Curve Control).

Owner intent (2026-06-13):
  ACC  — OEM-like cruise: CRUISE + LEAD only.
  E2E  — the model drives the car: adds MODEL_STOP (traffic lights / stop signs).
  SCC  — intelligent ACC/E2E blend: ACC-like base + MODEL_STOP + speed-limit, with the curve
         sources gated by SccCurveVisionEnabled / SccCurveMapEnabled.

This module is pure and exhaustively property-tested; correctness needs no engaged data.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class LongitudinalMode(Enum):
  ACC = "acc"
  E2E = "e2e"
  SCC = "scc"

  @classmethod
  def from_value(cls, value: object, default: "LongitudinalMode" = None) -> "LongitudinalMode":
    default = default if default is not None else cls.ACC
    if isinstance(value, cls):
      return value
    if isinstance(value, bytes):
      value = value.decode(errors="ignore")
    text = str(value or "").strip().lower()
    for mode in cls:
      if text == mode.value:
        return mode
    # tolerate enum-name spellings ("ACC") and the legacy int encoding (0/1/2)
    for mode in cls:
      if text == mode.name.lower():
        return mode
    int_map = {"0": cls.ACC, "1": cls.E2E, "2": cls.SCC}
    return int_map.get(text, default)


class EvidenceClass(Enum):
  CRUISE = auto()        # speed-hold / set-speed cruise (always admissible)
  LEAD = auto()          # confirmed lead following (MPC physical hazard)
  MODEL_STOP = auto()    # E2E model stop: traffic lights, stop signs, model-detected stops
  SPEED_LIMIT = auto()   # speed-limit assist (SLA)
  CURVE_VISION = auto()  # SCC vision-predicted curve cap
  CURVE_MAP = auto()     # SCC map-derived curve cap


@dataclass(frozen=True)
class SourceToggles:
  """Mode-owned source controls. Only consulted inside the mode that owns them (SCC)."""
  scc_curve_vision_enabled: bool = False
  scc_curve_map_enabled: bool = False


_ACC_EVIDENCE = frozenset({EvidenceClass.CRUISE, EvidenceClass.LEAD})
_E2E_EVIDENCE = _ACC_EVIDENCE | {EvidenceClass.MODEL_STOP}
_SCC_BASE_EVIDENCE = _E2E_EVIDENCE | {EvidenceClass.SPEED_LIMIT}


def admitted_evidence(mode: LongitudinalMode, sources: SourceToggles = SourceToggles()) -> frozenset[EvidenceClass]:
  """The set of evidence classes the active mode permits to affect actuation.

  ACC and E2E ignore ``sources`` entirely — the curve toggles are SCC-owned, so they can
  never make ACC or E2E consume curve/map/speed-limit evidence.
  """
  if mode is LongitudinalMode.ACC:
    return _ACC_EVIDENCE
  if mode is LongitudinalMode.E2E:
    return _E2E_EVIDENCE
  # SCC: ACC/E2E blend + map/speed-limit, with curve sources gated by their toggles.
  admitted = set(_SCC_BASE_EVIDENCE)
  if sources.scc_curve_vision_enabled:
    admitted.add(EvidenceClass.CURVE_VISION)
  if sources.scc_curve_map_enabled:
    admitted.add(EvidenceClass.CURVE_MAP)
  return frozenset(admitted)


def is_admitted(mode: LongitudinalMode, evidence: EvidenceClass, sources: SourceToggles = SourceToggles()) -> bool:
  return evidence in admitted_evidence(mode, sources)
