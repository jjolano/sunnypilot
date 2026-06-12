"""Diagnostic normalization contract only.

This module provides pure normalization helpers for diagnostic telemetry and must not
alter control authority without branch-specific tests and route-derived evidence.
"""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class NormalizedDiagnostic:
  source: str
  kind: str
  value: float
  valid: bool
  reason: str = ""


def clamp01(value: float) -> float:
  try:
    value = float(value)
  except (TypeError, ValueError):
    return 0.0
  if not math.isfinite(value):
    return 0.0
  return min(1.0, max(0.0, value))


def _invalid(source: str, kind: str, reason: str = "") -> NormalizedDiagnostic:
  return NormalizedDiagnostic(source=source, kind=kind, value=0.0, valid=False, reason=reason)


def normalize_unit_interval(value, source: str, kind: str, reason: str = "") -> NormalizedDiagnostic:
  try:
    normalized = clamp01(value)
  except (TypeError, ValueError):
    return _invalid(source, kind, reason)
  try:
    numeric = float(value)
  except (TypeError, ValueError):
    return _invalid(source, kind, reason)
  if not math.isfinite(numeric):
    return _invalid(source, kind, reason)
  return NormalizedDiagnostic(source=source, kind=kind, value=normalized, valid=True, reason=reason)


def normalize_inverse_unit_interval(value, source: str, kind: str, reason: str = "") -> NormalizedDiagnostic:
  diag = normalize_unit_interval(value, source, kind, reason)
  if not diag.valid:
    return diag
  return NormalizedDiagnostic(source=diag.source, kind=diag.kind, value=1.0 - diag.value, valid=True, reason=diag.reason)


def normalize_range(value, low: float, high: float, source: str, kind: str, reason: str = "", invert: bool = False) -> NormalizedDiagnostic:
  try:
    value = float(value)
    low = float(low)
    high = float(high)
  except (TypeError, ValueError):
    return _invalid(source, kind, reason)
  if not all(math.isfinite(x) for x in (value, low, high)) or high == low:
    return _invalid(source, kind, reason)

  normalized = (value - low) / (high - low)
  normalized = clamp01(normalized)
  if invert:
    normalized = 1.0 - normalized
  return NormalizedDiagnostic(source=source, kind=kind, value=normalized, valid=True, reason=reason)
