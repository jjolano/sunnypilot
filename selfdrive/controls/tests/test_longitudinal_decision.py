import math

from openpilot.selfdrive.controls.lib.longitudinal_decision import (
  CandidateRole,
  DecisionSource,
  LongitudinalArbiter,
  LongitudinalCandidate,
  resolve_longitudinal_decision,
)


def test_candidate_clamps_confidence_and_urgency():
  candidate = LongitudinalCandidate(
    source=DecisionSource.CRUISE,
    role=CandidateRole.DRIVER_INTENT,
    v_target=20.0,
    a_target=0.1,
    confidence=2.0,
    urgency=-1.0,
    active_reason="driver_set_speed",
  )

  assert candidate.confidence == 1.0
  assert candidate.urgency == 0.0


def test_candidate_rejects_non_finite_targets():
  candidate = LongitudinalCandidate(
    source=DecisionSource.CRUISE,
    role=CandidateRole.DRIVER_INTENT,
    v_target=math.inf,
    a_target=0.0,
    confidence=1.0,
    urgency=0.0,
    active_reason="bad_speed",
  )

  assert not candidate.valid
  assert "non_finite" in candidate.invalid_reason


def test_candidate_rejects_empty_reason():
  candidate = LongitudinalCandidate(
    source=DecisionSource.SPEED_LIMIT,
    role=CandidateRole.ADVISORY_CAP,
    v_target=15.0,
    a_target=-0.2,
    confidence=0.8,
    urgency=0.4,
    active_reason="",
  )

  assert not candidate.valid
  assert candidate.invalid_reason == "missing_active_reason"


def test_candidate_records_debug_without_affecting_validity():
  candidate = LongitudinalCandidate(
    source=DecisionSource.SCC_MAP,
    role=CandidateRole.ADVISORY_CAP,
    v_target=12.0,
    a_target=-0.3,
    confidence=0.9,
    urgency=0.5,
    active_reason="confident_map_curve",
    debug={"distance": 80.0},
  )

  assert candidate.valid
  assert candidate.debug == {"distance": 80.0}


def make_candidate(source, role, v_target, a_target, confidence, urgency, reason, should_stop=False):
  return LongitudinalCandidate(
    source=source,
    role=role,
    v_target=v_target,
    a_target=a_target,
    confidence=confidence,
    urgency=urgency,
    active_reason=reason,
    should_stop=should_stop,
  )


def test_arbiter_defaults_to_driver_intent():
  arbiter = LongitudinalArbiter()
  cruise = make_candidate(DecisionSource.CRUISE, CandidateRole.DRIVER_INTENT, 25.0, 0.2, 1.0, 0.1, "driver_set_speed")

  decision = arbiter.decide([cruise])

  assert decision.enabled
  assert decision.winner == DecisionSource.CRUISE
  assert decision.a_target == 0.2
  assert decision.suppressed == ()


def test_arbiter_missing_driver_intent_uses_internal_fallback():
  decision = LongitudinalArbiter().decide([])

  assert decision.enabled
  assert decision.winner == DecisionSource.LEGACY_FALLBACK
  assert decision.suppressed == ()


def test_confirmed_physical_hazard_overrides_speed_limit_advisory():
  arbiter = LongitudinalArbiter()
  cruise = make_candidate(DecisionSource.CRUISE, CandidateRole.DRIVER_INTENT, 27.0, 0.1, 1.0, 0.1, "driver_set_speed")
  speed_limit = make_candidate(DecisionSource.SPEED_LIMIT, CandidateRole.ADVISORY_CAP, 18.0, -0.2, 0.9, 0.3, "advisory_limit")
  lead = make_candidate(DecisionSource.LEAD_MPC, CandidateRole.PHYSICAL_HAZARD, 20.0, -0.8, 0.8, 0.7, "confirmed_lead")

  decision = arbiter.decide([cruise, speed_limit, lead])

  assert decision.winner == DecisionSource.LEAD_MPC
  assert decision.a_target == -0.8
  assert (DecisionSource.SPEED_LIMIT, "physical_hazard_active") in decision.suppressed


def test_high_confidence_advisory_cap_shapes_driver_intent_without_hazard():
  arbiter = LongitudinalArbiter()
  cruise = make_candidate(DecisionSource.CRUISE, CandidateRole.DRIVER_INTENT, 27.0, 0.2, 1.0, 0.1, "driver_set_speed")
  speed_limit = make_candidate(DecisionSource.SPEED_LIMIT, CandidateRole.ADVISORY_CAP, 20.0, -0.3, 0.9, 0.4, "advisory_limit")

  decision = arbiter.decide([cruise, speed_limit])

  assert decision.winner == DecisionSource.SPEED_LIMIT
  assert decision.a_target == -0.3


def test_low_confidence_advisory_is_suppressed_for_driver_intent():
  arbiter = LongitudinalArbiter()
  cruise = make_candidate(DecisionSource.CRUISE, CandidateRole.DRIVER_INTENT, 27.0, 0.2, 1.0, 0.1, "driver_set_speed")
  weak_curve = make_candidate(DecisionSource.SCC_VISION, CandidateRole.ADVISORY_CAP, 18.0, -0.4, 0.3, 0.6, "weak_curve")

  decision = arbiter.decide([cruise, weak_curve])

  assert decision.winner == DecisionSource.CRUISE
  assert (DecisionSource.SCC_VISION, "low_confidence") in decision.suppressed


def test_comfort_candidate_can_relax_accel_when_no_hazard_or_advisory():
  arbiter = LongitudinalArbiter()
  cruise = make_candidate(DecisionSource.CRUISE, CandidateRole.DRIVER_INTENT, 25.0, -0.8, 1.0, 0.1, "driver_set_speed")
  coast = make_candidate(DecisionSource.CRUISE_COAST, CandidateRole.COMFORT_SHAPING, 25.0, -0.2, 0.8, 0.2, "harmless_overspeed")

  decision = arbiter.decide([cruise, coast])

  assert decision.winner == DecisionSource.CRUISE_COAST
  assert decision.a_target == -0.2


def test_resolver_returns_legacy_fallback_when_toggle_disabled():
  cruise = make_candidate(DecisionSource.CRUISE, CandidateRole.DRIVER_INTENT, 25.0, 0.2, 1.0, 0.1, "driver_set_speed")

  decision = resolve_longitudinal_decision(
    enabled=False,
    candidates=[cruise],
    fallback_v_target=25.0,
    fallback_a_target=-0.4,
    fallback_should_stop=True,
    accel_limits=(-1.2, 1.0),
    arbiter=LongitudinalArbiter(),
  )

  assert not decision.enabled
  assert decision.winner == DecisionSource.LEGACY_FALLBACK
  assert decision.a_target == -0.4
  assert decision.should_stop
  assert decision.fallback_reason == "feature_flag_disabled"


def test_resolver_falls_back_when_decision_exceeds_accel_limits():
  bad = make_candidate(DecisionSource.LEAD_MPC, CandidateRole.PHYSICAL_HAZARD, 15.0, -5.0, 1.0, 1.0, "bad_decel")

  decision = resolve_longitudinal_decision(
    enabled=True,
    candidates=[bad],
    fallback_v_target=25.0,
    fallback_a_target=-0.4,
    fallback_should_stop=False,
    accel_limits=(-1.2, 1.0),
    arbiter=LongitudinalArbiter(),
  )

  assert not decision.enabled
  assert decision.winner == DecisionSource.LEGACY_FALLBACK
  assert decision.fallback_reason == "decision_outside_accel_limits"


def test_resolver_falls_back_when_driver_intent_is_missing():
  decision = resolve_longitudinal_decision(
    enabled=True,
    candidates=[],
    fallback_v_target=25.0,
    fallback_a_target=-0.4,
    fallback_should_stop=False,
    accel_limits=(-1.2, 1.0),
    arbiter=LongitudinalArbiter(),
  )

  assert not decision.enabled
  assert decision.winner == DecisionSource.LEGACY_FALLBACK
  assert decision.v_target == 25.0
  assert decision.a_target == -0.4
  assert decision.fallback_reason == "missing_driver_intent"
