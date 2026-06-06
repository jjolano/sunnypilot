"""Compatibility exports for longitudinal stack policy helpers.

The implementations live in focused Modules so signal-provider candidate
building, Planner Seed Candidate conversion, and custom-v2 debug metadata have
separate Locality.  Keep this shim while call sites migrate gradually.
"""

from openpilot.selfdrive.controls.lib.longitudinal_stacks.candidate_builders import (
  SignalProviderCandidate,
  build_sp_candidates_from_signal_providers,
  build_sp_longitudinal_candidates,
  ensure_driver_intent,
  replace_driver_intent,
)
from openpilot.selfdrive.controls.lib.longitudinal_stacks.custom_v2_debug import (
  CUSTOM_V2_DEBUG_DISABLE_JERK_LIMIT,
  CUSTOM_V2_DEBUG_INTENT,
  CUSTOM_V2_DEBUG_REASON,
  CUSTOM_V2_DEBUG_SEED_CANDIDATE,
  CUSTOM_V2_DEBUG_SEED_CONTEXT,
  CUSTOM_V2_DEBUG_STACK_OUTPUT,
  custom_v2_candidate_with_debug,
  custom_v2_intent_for_source,
  custom_v2_rejections_from_decision,
  selected_candidate_for_decision,
)
from openpilot.selfdrive.controls.lib.longitudinal_stacks.planner_seed_policy import (
  E2E_SOURCE_VALUES,
  LEAD_MPC_SOURCE_VALUES,
  LONGITUDINAL_PLAN_SOURCE,
  fallback_physical_candidates,
  planner_seed_candidate_to_longitudinal_candidate,
  planner_seed_candidates_to_longitudinal_candidates,
)


__all__ = (
  "CUSTOM_V2_DEBUG_DISABLE_JERK_LIMIT",
  "CUSTOM_V2_DEBUG_INTENT",
  "CUSTOM_V2_DEBUG_REASON",
  "CUSTOM_V2_DEBUG_SEED_CANDIDATE",
  "CUSTOM_V2_DEBUG_SEED_CONTEXT",
  "CUSTOM_V2_DEBUG_STACK_OUTPUT",
  "E2E_SOURCE_VALUES",
  "LEAD_MPC_SOURCE_VALUES",
  "LONGITUDINAL_PLAN_SOURCE",
  "SignalProviderCandidate",
  "build_sp_candidates_from_signal_providers",
  "build_sp_longitudinal_candidates",
  "custom_v2_candidate_with_debug",
  "custom_v2_intent_for_source",
  "custom_v2_rejections_from_decision",
  "ensure_driver_intent",
  "fallback_physical_candidates",
  "planner_seed_candidate_to_longitudinal_candidate",
  "planner_seed_candidates_to_longitudinal_candidates",
  "replace_driver_intent",
  "selected_candidate_for_decision",
)
