from __future__ import annotations

from dataclasses import replace
import math
from typing import Any

from openpilot.selfdrive.controls.lib.longitudinal_decision import (
  DecisionSource,
  LongitudinalCandidate,
  LongitudinalDecision,
)
from openpilot.selfdrive.controls.lib.longitudinal_stacks.interface import LongitudinalStackOutput


CUSTOM_V2_DEBUG_INTENT = "custom_v2_intent"
CUSTOM_V2_DEBUG_REASON = "custom_v2_reason"
CUSTOM_V2_DEBUG_SEED_CONTEXT = "custom_v2_seed_context"
CUSTOM_V2_DEBUG_SEED_CANDIDATE = "custom_v2_seed_candidate"
CUSTOM_V2_DEBUG_STACK_OUTPUT = "custom_v2_stack_output"
CUSTOM_V2_DEBUG_DISABLE_JERK_LIMIT = "custom_v2_disable_jerk_limit"


def custom_v2_candidate_with_debug(candidate: LongitudinalCandidate, intent: str, reason: str,
                                   output: LongitudinalStackOutput | None = None,
                                   seed_context: str = "", seed_candidate: str = "",
                                   disable_jerk_limit: bool = False,
                                   extra_rejected: tuple[tuple[str, str], ...] = ()) -> LongitudinalCandidate:
  debug: dict[str, Any] = dict(candidate.debug)
  debug.update({
    CUSTOM_V2_DEBUG_INTENT: str(intent),
    CUSTOM_V2_DEBUG_REASON: str(reason),
    CUSTOM_V2_DEBUG_SEED_CONTEXT: str(seed_context),
    CUSTOM_V2_DEBUG_SEED_CANDIDATE: str(seed_candidate),
  })
  if output is not None:
    debug[CUSTOM_V2_DEBUG_STACK_OUTPUT] = output
  if disable_jerk_limit:
    debug[CUSTOM_V2_DEBUG_DISABLE_JERK_LIMIT] = True
  if extra_rejected:
    debug["custom_v2_extra_rejected"] = extra_rejected
  return replace(candidate, debug=debug)


def selected_candidate_for_decision(decision: LongitudinalDecision) -> LongitudinalCandidate | None:
  for candidate in decision.candidates:
    if (
      candidate.source == decision.winner and
      candidate.active_reason == decision.active_reason and
      math.isclose(candidate.a_target, decision.a_target, abs_tol=1e-6)
    ):
      return candidate
  for candidate in decision.candidates:
    if candidate.source == decision.winner and candidate.active_reason == decision.active_reason:
      return candidate
  for candidate in decision.candidates:
    if candidate.source == decision.winner:
      return candidate
  return None


def custom_v2_rejections_from_decision(decision: LongitudinalDecision) -> tuple[tuple[str, str], ...]:
  rich_suppressed = getattr(decision, "suppressed_candidates", ())
  rich_keys = {
    (getattr(candidate, "source", None), _suppressed_candidate_reason(candidate))
    for candidate in rich_suppressed
  }
  rejected: list[tuple[str, str]] = [
    (_suppressed_candidate_custom_v2_intent(candidate), _suppressed_candidate_reason(candidate))
    for candidate in rich_suppressed
  ]

  candidates_by_source = {candidate.source: candidate for candidate in decision.candidates}
  for source, reason in decision.suppressed:
    if (source, str(reason)) in rich_keys:
      continue
    candidate = candidates_by_source.get(source)
    intent = _candidate_custom_v2_intent(candidate) if candidate is not None else custom_v2_intent_for_source(source)
    rejected.append((intent, str(reason)))
  return tuple(dict.fromkeys(rejected))


def custom_v2_intent_for_source(source: DecisionSource) -> str:
  if source == DecisionSource.SPEED_LIMIT:
    return "speed_policy"
  if source in (DecisionSource.SCC_VISION, DecisionSource.SCC_MAP):
    return "curve_policy"
  if source == DecisionSource.OSM_TRAFFIC_CONTROL:
    return "map_caution"
  if source == DecisionSource.LEAD_MPC:
    return "lead_follow"
  if source == DecisionSource.E2E_STOP:
    return "stop_approach"
  if source == DecisionSource.STOP_LAUNCH:
    return "launch"
  if source == DecisionSource.CRUISE_COAST:
    return "comfort_relax"
  return "driver_cruise"


def _candidate_custom_v2_intent(candidate: LongitudinalCandidate | None) -> str:
  if candidate is None:
    return "driver_cruise"
  return str(candidate.debug.get(CUSTOM_V2_DEBUG_INTENT) or custom_v2_intent_for_source(candidate.source))


def _suppressed_candidate_custom_v2_intent(candidate: Any) -> str:
  debug = getattr(candidate, "debug", {})
  intent = debug.get(CUSTOM_V2_DEBUG_INTENT) if isinstance(debug, dict) else ""
  if intent:
    return str(intent)
  return custom_v2_intent_for_source(getattr(candidate, "source", DecisionSource.CRUISE))


def _suppressed_candidate_reason(candidate: Any) -> str:
  reason = getattr(candidate, "suppression_reason", "")
  if reason:
    return str(reason)
  debug = getattr(candidate, "debug", {})
  if isinstance(debug, dict):
    debug_reason = debug.get(CUSTOM_V2_DEBUG_REASON)
    if debug_reason:
      return str(debug_reason)
  return str(getattr(candidate, "active_reason", ""))


def _physical_candidate_identity(candidate: LongitudinalCandidate) -> tuple[DecisionSource, str, str]:
  return (candidate.source, _candidate_custom_v2_intent(candidate), candidate.active_reason)
