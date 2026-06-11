import pytest

from openpilot.selfdrive.controls.lib.longitudinal_decision import (
  ADVISORY_CAP_ACTIVE_REASON,
  ADVISORY_INCREASES_ACCEL_REASON,
  CandidateRole,
  DecisionSource,
  LongitudinalArbiter,
  LongitudinalCandidate,
)
from openpilot.selfdrive.controls.lib.longitudinal_stacks.policy import (
  CUSTOM_V2_DEBUG_INTENT,
  custom_v2_candidate_with_debug,
  replace_driver_intent,
)


def make_candidate(source, role, a_target, reason, *, v_target=20.0, confidence=1.0, urgency=0.5,
                   should_stop=False, debug=None, **kwargs):
  return LongitudinalCandidate(
    source=source,
    role=role,
    v_target=v_target,
    a_target=a_target,
    confidence=confidence,
    urgency=urgency,
    active_reason=reason,
    should_stop=should_stop,
    debug=debug or {},
    **kwargs,
  )


def suppressed_reason(decision, source, active_reason):
  for candidate in decision.suppressed_candidates:
    if candidate.source == source and candidate.active_reason == active_reason:
      return candidate.suppression_reason
  return ""


def test_physical_hazard_suppresses_advisory_and_relaxation():
  cruise = make_candidate(DecisionSource.CRUISE, CandidateRole.DRIVER_INTENT, 0.1, "driver_cruise")
  advisory = make_candidate(DecisionSource.SPEED_LIMIT, CandidateRole.ADVISORY_CAP, -0.2, "speed_cap", v_target=12.0, confidence=0.9)
  relax = make_candidate(DecisionSource.STOP_LAUNCH, CandidateRole.RELAXATION, 0.8, "progress_floor")
  lead = make_candidate(DecisionSource.LEAD_MPC, CandidateRole.PHYSICAL_HAZARD, -0.7, "confirmed_lead", confidence=0.9)

  decision = LongitudinalArbiter().decide([cruise, advisory, relax, lead])

  assert decision.winner == DecisionSource.LEAD_MPC
  assert decision.a_target == pytest.approx(-0.7)
  assert suppressed_reason(decision, DecisionSource.SPEED_LIMIT, "speed_cap") == "physical_hazard_active"
  assert suppressed_reason(decision, DecisionSource.STOP_LAUNCH, "progress_floor") == "physical_hazard_active"


def test_advisory_cannot_increase_accel():
  cruise = make_candidate(DecisionSource.CRUISE, CandidateRole.DRIVER_INTENT, 0.0, "driver_cruise", v_target=25.0)
  bad_advisory = make_candidate(
    DecisionSource.SPEED_LIMIT, CandidateRole.ADVISORY_CAP, 0.3, "bad_speed_cap", v_target=12.0, confidence=0.9,
  )

  decision = LongitudinalArbiter().decide([cruise, bad_advisory])

  assert decision.winner == DecisionSource.CRUISE
  assert suppressed_reason(decision, DecisionSource.SPEED_LIMIT, "bad_speed_cap") == ADVISORY_INCREASES_ACCEL_REASON


def test_advisory_cannot_authorize_launch_or_progress():
  cruise = make_candidate(DecisionSource.CRUISE, CandidateRole.DRIVER_INTENT, -0.1, "driver_cruise", v_target=25.0)
  advisory = make_candidate(DecisionSource.SPEED_LIMIT, CandidateRole.ADVISORY_CAP, -0.3, "speed_cap", v_target=12.0, confidence=0.9)
  launch = make_candidate(DecisionSource.STOP_LAUNCH, CandidateRole.RELAXATION, 0.8, "no_lead_stop_clear", v_target=25.0)

  decision = LongitudinalArbiter().decide([cruise, advisory, launch])

  assert decision.winner == DecisionSource.SPEED_LIMIT
  assert decision.a_target == pytest.approx(-0.3)
  assert suppressed_reason(decision, DecisionSource.STOP_LAUNCH, "no_lead_stop_clear") == ADVISORY_CAP_ACTIVE_REASON


def test_relaxation_loses_to_physical_and_advisory_authority():
  cruise = make_candidate(DecisionSource.CRUISE, CandidateRole.DRIVER_INTENT, -0.4, "driver_cruise", v_target=25.0)
  relax = make_candidate(DecisionSource.CRUISE_COAST, CandidateRole.RELAXATION, -0.1, "comfort_relax", v_target=25.0)
  advisory = make_candidate(DecisionSource.SCC_VISION, CandidateRole.ADVISORY_CAP, -0.5, "curve_cap", v_target=18.0, confidence=0.9)
  lead = make_candidate(DecisionSource.LEAD_MPC, CandidateRole.PHYSICAL_HAZARD, -0.8, "confirmed_lead", v_target=18.0, confidence=0.9)

  advisory_decision = LongitudinalArbiter().decide([cruise, relax, advisory])
  physical_decision = LongitudinalArbiter().decide([cruise, relax, advisory, lead])

  assert advisory_decision.winner == DecisionSource.SCC_VISION
  assert suppressed_reason(advisory_decision, DecisionSource.CRUISE_COAST, "comfort_relax") == ADVISORY_CAP_ACTIVE_REASON
  assert physical_decision.winner == DecisionSource.LEAD_MPC
  assert suppressed_reason(physical_decision, DecisionSource.CRUISE_COAST, "comfort_relax") == "physical_hazard_active"


def test_exactly_one_driver_intent_is_effective():
  cruise = make_candidate(DecisionSource.CRUISE, CandidateRole.DRIVER_INTENT, 0.1, "driver_cruise")
  duplicate = make_candidate(DecisionSource.STOP_LAUNCH, CandidateRole.DRIVER_INTENT, 0.0, "duplicate_driver")

  decision = LongitudinalArbiter().decide([cruise, duplicate])

  assert decision.winner == DecisionSource.LEGACY_FALLBACK
  assert decision.fallback_reason == "duplicate_driver_intent"
  assert suppressed_reason(decision, DecisionSource.CRUISE, "driver_cruise") == "duplicate_driver_intent"
  assert suppressed_reason(decision, DecisionSource.STOP_LAUNCH, "duplicate_driver") == "duplicate_driver_intent"


def test_one_pedal_replaces_driver_intent_and_preserves_physical_braking():
  driver = custom_v2_candidate_with_debug(
    make_candidate(DecisionSource.CRUISE, CandidateRole.DRIVER_INTENT, 0.3, "driver_cruise"),
    intent="driver_cruise",
    reason="driver_cruise",
  )
  one_pedal = custom_v2_candidate_with_debug(
    make_candidate(DecisionSource.CRUISE, CandidateRole.DRIVER_INTENT, 0.0, "lift_off_coast"),
    intent="one_pedal",
    reason="lift_off_coast",
  )
  lead = custom_v2_candidate_with_debug(
    make_candidate(DecisionSource.LEAD_MPC, CandidateRole.PHYSICAL_HAZARD, -0.6, "confirmed_lead", confidence=0.9),
    intent="lead_follow",
    reason="confirmed_lead",
  )

  candidates = (*replace_driver_intent((driver,), one_pedal), lead)
  decision = LongitudinalArbiter().decide(candidates)

  assert [candidate.role for candidate in candidates].count(CandidateRole.DRIVER_INTENT) == 1
  assert candidates[0].debug[CUSTOM_V2_DEBUG_INTENT] == "one_pedal"
  assert decision.winner == DecisionSource.LEAD_MPC
  assert decision.a_target == pytest.approx(-0.6)
  assert suppressed_reason(decision, DecisionSource.CRUISE, "lift_off_coast") == "physical_hazard_active"


def test_equal_to_baseline_physical_authority_remains_represented():
  cruise = make_candidate(DecisionSource.CRUISE, CandidateRole.DRIVER_INTENT, -0.4, "driver_cruise", v_target=20.0)
  lead_equal = make_candidate(
    DecisionSource.LEAD_MPC, CandidateRole.PHYSICAL_HAZARD, -0.4, "equal_lead_cap", v_target=20.0, confidence=0.9,
  )

  decision = LongitudinalArbiter().decide([cruise, lead_equal])

  assert decision.winner == DecisionSource.CRUISE
  assert lead_equal in decision.candidates
  assert suppressed_reason(decision, DecisionSource.LEAD_MPC, "equal_lead_cap") == ""


def test_rejected_telemetry_identifies_actual_suppressed_candidate():
  cruise = make_candidate(DecisionSource.CRUISE, CandidateRole.DRIVER_INTENT, 0.0, "driver_cruise", v_target=25.0)
  actual = make_candidate(
    DecisionSource.SPEED_LIMIT, CandidateRole.ADVISORY_CAP, -0.2, "actual_suppressed_limit",
    v_target=20.0, confidence=0.9, debug={"provider": "speed_limit"},
  )
  winner = make_candidate(
    DecisionSource.SCC_VISION, CandidateRole.ADVISORY_CAP, -0.4, "near_curve",
    v_target=18.0, confidence=0.9, required_a_target=-0.7,
  )

  decision = LongitudinalArbiter().decide([cruise, actual, winner])
  suppressed = next(candidate for candidate in decision.suppressed_candidates if candidate.active_reason == "actual_suppressed_limit")

  assert decision.winner == DecisionSource.SCC_VISION
  assert suppressed.source == DecisionSource.SPEED_LIMIT
  assert suppressed.role == CandidateRole.ADVISORY_CAP
  assert suppressed.suppression_reason == "higher_advisory_target"
  assert suppressed.debug == {"provider": "speed_limit"}
