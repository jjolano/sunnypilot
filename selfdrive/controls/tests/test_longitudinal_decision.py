import math
from types import SimpleNamespace

from cereal import log
import pytest

from openpilot.selfdrive.controls.lib.longitudinal_decision import (
  CandidateRole,
  DecisionSource,
  LongitudinalArbiter,
  LongitudinalCandidate,
  LongitudinalDecision,
  SOURCE_STABILITY_HOLD_REASON,
  SOURCE_STABILITY_RELEASE_FRAMES,
  apply_personality_accel_comfort,
  apply_longitudinal_decision_output,
  apply_longitudinal_decision_output_with_telemetry,
  build_core_longitudinal_candidates,
  get_active_lead_confidence,
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


def test_candidate_rejects_non_finite_bounds():
  candidate = LongitudinalCandidate(
    source=DecisionSource.SPEED_LIMIT,
    role=CandidateRole.ADVISORY_CAP,
    v_target=15.0,
    a_target=-0.2,
    confidence=0.8,
    urgency=0.4,
    active_reason="advisory_limit",
    comfort_bounds=(-1.0, math.inf),
  )

  assert not candidate.valid
  assert candidate.invalid_reason == "non_finite_comfort_bounds"


def test_candidate_rejects_inverted_bounds():
  candidate = LongitudinalCandidate(
    source=DecisionSource.LEAD_MPC,
    role=CandidateRole.PHYSICAL_HAZARD,
    v_target=10.0,
    a_target=-0.8,
    confidence=0.9,
    urgency=0.8,
    active_reason="confirmed_lead",
    safety_bounds=(0.0, -1.0),
  )

  assert not candidate.valid
  assert candidate.invalid_reason == "inverted_safety_bounds"


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


def make_decision(winner, a_target, should_stop=False, enabled=True, active_reason=""):
  return LongitudinalDecision(
    enabled=enabled,
    winner=winner,
    v_target=25.0,
    a_target=a_target,
    should_stop=should_stop,
    active_reason=active_reason,
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


def test_arbiter_duplicate_driver_intent_uses_internal_fallback():
  cruise = make_candidate(DecisionSource.CRUISE, CandidateRole.DRIVER_INTENT, 25.0, 0.2, 1.0, 0.1, "driver_set_speed")
  launch = make_candidate(DecisionSource.STOP_LAUNCH, CandidateRole.DRIVER_INTENT, 25.0, 0.1, 1.0, 0.2, "launch_target")

  decision = LongitudinalArbiter().decide([cruise, launch])

  assert decision.enabled
  assert decision.winner == DecisionSource.LEGACY_FALLBACK
  assert decision.fallback_reason == "duplicate_driver_intent"
  assert (DecisionSource.CRUISE, "duplicate_driver_intent") in decision.suppressed
  assert (DecisionSource.STOP_LAUNCH, "duplicate_driver_intent") in decision.suppressed


def test_arbiter_suppresses_malformed_advisory_metadata():
  cruise = make_candidate(DecisionSource.CRUISE, CandidateRole.DRIVER_INTENT, 25.0, 0.2, 1.0, 0.1, "driver_set_speed")
  malformed = LongitudinalCandidate(
    source=DecisionSource.SPEED_LIMIT,
    role=CandidateRole.ADVISORY_CAP,
    v_target=20.0,
    a_target=-0.3,
    confidence=0.9,
    urgency=0.4,
    active_reason="advisory_limit",
    comfort_bounds=(math.nan, None),
  )

  decision = LongitudinalArbiter().decide([cruise, malformed])

  assert decision.winner == DecisionSource.CRUISE
  assert (DecisionSource.SPEED_LIMIT, "non_finite_comfort_bounds") in decision.suppressed


def test_arbiter_physical_hazard_without_driver_intent_uses_internal_fallback():
  lead = make_candidate(DecisionSource.LEAD_MPC, CandidateRole.PHYSICAL_HAZARD, 15.0, -0.8, 1.0, 1.0, "confirmed_lead")

  decision = LongitudinalArbiter().decide([lead])

  assert decision.enabled
  assert decision.winner == DecisionSource.LEGACY_FALLBACK
  assert decision.v_target == 0.0
  assert decision.a_target == 0.0


def test_confirmed_physical_hazard_overrides_speed_limit_advisory():
  arbiter = LongitudinalArbiter()
  cruise = make_candidate(DecisionSource.CRUISE, CandidateRole.DRIVER_INTENT, 27.0, 0.1, 1.0, 0.1, "driver_set_speed")
  speed_limit = make_candidate(DecisionSource.SPEED_LIMIT, CandidateRole.ADVISORY_CAP, 18.0, -0.2, 0.9, 0.3, "advisory_limit")
  lead = make_candidate(DecisionSource.LEAD_MPC, CandidateRole.PHYSICAL_HAZARD, 20.0, -0.8, 0.8, 0.7, "confirmed_lead")

  decision = arbiter.decide([cruise, speed_limit, lead])

  assert decision.winner == DecisionSource.LEAD_MPC
  assert decision.a_target == -0.8
  assert (DecisionSource.SPEED_LIMIT, "physical_hazard_active") in decision.suppressed


def test_physical_hazard_tie_breaking_is_independent_of_candidate_order():
  cruise = make_candidate(DecisionSource.CRUISE, CandidateRole.DRIVER_INTENT, 27.0, 0.1, 1.0, 0.1, "driver_set_speed")
  lead = make_candidate(DecisionSource.LEAD_MPC, CandidateRole.PHYSICAL_HAZARD, 12.0, -0.8, 0.9, 0.7, "confirmed_lead")
  e2e_stop = make_candidate(DecisionSource.E2E_STOP, CandidateRole.PHYSICAL_HAZARD, 12.0, -0.8, 0.9, 0.7, "model_stop")

  first_decision = LongitudinalArbiter().decide([cruise, lead, e2e_stop])
  reversed_decision = LongitudinalArbiter().decide([cruise, e2e_stop, lead])

  assert first_decision.winner == DecisionSource.LEAD_MPC
  assert reversed_decision.winner == DecisionSource.LEAD_MPC


def test_high_confidence_advisory_cap_shapes_driver_intent_without_hazard():
  arbiter = LongitudinalArbiter()
  cruise = make_candidate(DecisionSource.CRUISE, CandidateRole.DRIVER_INTENT, 27.0, 0.2, 1.0, 0.1, "driver_set_speed")
  speed_limit = make_candidate(DecisionSource.SPEED_LIMIT, CandidateRole.ADVISORY_CAP, 20.0, -0.3, 0.9, 0.4, "advisory_limit")

  decision = arbiter.decide([cruise, speed_limit])

  assert decision.winner == DecisionSource.SPEED_LIMIT
  assert decision.a_target == -0.3


def test_advisory_tie_breaking_is_independent_of_candidate_order():
  cruise = make_candidate(DecisionSource.CRUISE, CandidateRole.DRIVER_INTENT, 27.0, 0.2, 1.0, 0.1, "driver_set_speed")
  speed_limit = make_candidate(DecisionSource.SPEED_LIMIT, CandidateRole.ADVISORY_CAP, 18.0, -0.3, 0.9, 0.4, "speed_limit")
  scc_vision = make_candidate(DecisionSource.SCC_VISION, CandidateRole.ADVISORY_CAP, 18.0, -0.3, 0.9, 0.4, "vision_curve")

  first_decision = LongitudinalArbiter().decide([cruise, speed_limit, scc_vision])
  reversed_decision = LongitudinalArbiter().decide([cruise, scc_vision, speed_limit])

  assert first_decision.winner == DecisionSource.SPEED_LIMIT
  assert reversed_decision.winner == DecisionSource.SPEED_LIMIT


def test_low_confidence_advisory_is_suppressed_for_driver_intent():
  arbiter = LongitudinalArbiter()
  cruise = make_candidate(DecisionSource.CRUISE, CandidateRole.DRIVER_INTENT, 27.0, 0.2, 1.0, 0.1, "driver_set_speed")
  weak_curve = make_candidate(DecisionSource.SCC_VISION, CandidateRole.ADVISORY_CAP, 18.0, -0.4, 0.3, 0.6, "weak_curve")

  decision = arbiter.decide([cruise, weak_curve])

  assert decision.winner == DecisionSource.CRUISE
  assert (DecisionSource.SCC_VISION, "low_confidence") in decision.suppressed


def test_relaxation_candidate_can_relax_accel_when_no_hazard_or_advisory():
  arbiter = LongitudinalArbiter()
  cruise = make_candidate(DecisionSource.CRUISE, CandidateRole.DRIVER_INTENT, 25.0, -0.8, 1.0, 0.1, "driver_set_speed")
  coast = make_candidate(DecisionSource.CRUISE_COAST, CandidateRole.RELAXATION, 25.0, -0.2, 0.8, 0.2, "harmless_overspeed")

  decision = arbiter.decide([cruise, coast])

  assert decision.winner == DecisionSource.CRUISE_COAST
  assert decision.a_target == -0.2


def test_relaxation_candidate_cannot_raise_driver_target_speed():
  arbiter = LongitudinalArbiter()
  cruise = make_candidate(DecisionSource.CRUISE, CandidateRole.DRIVER_INTENT, 25.0, -0.8, 1.0, 0.1, "driver_set_speed")
  coast = make_candidate(DecisionSource.CRUISE_COAST, CandidateRole.RELAXATION, 30.0, -0.2, 0.8, 0.2, "harmless_overspeed")

  decision = arbiter.decide([cruise, coast])

  assert decision.winner == DecisionSource.CRUISE
  assert decision.v_target == 25.0
  assert decision.a_target == -0.8


def test_relaxation_candidate_guarded_against_runaway_accel():
  arbiter = LongitudinalArbiter()
  cruise = make_candidate(DecisionSource.CRUISE, CandidateRole.DRIVER_INTENT, 25.0, -0.8, 1.0, 0.1, "driver_set_speed")
  runaway = make_candidate(DecisionSource.CRUISE_COAST, CandidateRole.RELAXATION, 25.0, 0.6, 0.8, 0.2, "buggy_overspeed")

  decision = arbiter.decide([cruise, runaway])

  assert decision.winner == DecisionSource.CRUISE
  assert decision.a_target == -0.8
  assert (DecisionSource.CRUISE_COAST, "comfort_accel_exceeds_margin") in set(decision.suppressed)


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
  cruise = make_candidate(DecisionSource.CRUISE, CandidateRole.DRIVER_INTENT, 25.0, 0.2, 1.0, 0.1, "driver_set_speed")
  bad = make_candidate(DecisionSource.LEAD_MPC, CandidateRole.PHYSICAL_HAZARD, 15.0, -5.0, 1.0, 1.0, "bad_decel")

  decision = resolve_longitudinal_decision(
    enabled=True,
    candidates=[cruise, bad],
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


def test_resolver_falls_back_when_driver_intent_is_duplicated():
  cruise = make_candidate(DecisionSource.CRUISE, CandidateRole.DRIVER_INTENT, 25.0, 0.2, 1.0, 0.1, "driver_set_speed")
  launch = make_candidate(DecisionSource.STOP_LAUNCH, CandidateRole.DRIVER_INTENT, 25.0, 0.1, 1.0, 0.2, "launch_target")

  decision = resolve_longitudinal_decision(
    enabled=True,
    candidates=[cruise, launch],
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
  assert decision.fallback_reason == "duplicate_driver_intent"


def test_resolver_falls_back_when_physical_hazard_has_no_driver_intent():
  lead = make_candidate(DecisionSource.LEAD_MPC, CandidateRole.PHYSICAL_HAZARD, 15.0, -0.8, 1.0, 1.0, "confirmed_lead")

  decision = resolve_longitudinal_decision(
    enabled=True,
    candidates=[lead],
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


def test_enabled_resolver_keeps_candidate_telemetry():
  cruise = make_candidate(DecisionSource.CRUISE, CandidateRole.DRIVER_INTENT, 25.0, 0.2, 1.0, 0.1, "driver_set_speed")
  speed_limit = make_candidate(DecisionSource.SPEED_LIMIT, CandidateRole.ADVISORY_CAP, 20.0, -0.3, 0.9, 0.4, "advisory_limit")

  decision = resolve_longitudinal_decision(
    enabled=True,
    candidates=[cruise, speed_limit],
    fallback_v_target=25.0,
    fallback_a_target=0.2,
    fallback_should_stop=False,
    accel_limits=(-1.2, 1.0),
    arbiter=LongitudinalArbiter(),
  )

  assert decision.enabled
  assert decision.winner == DecisionSource.SPEED_LIMIT
  assert [candidate.source for candidate in decision.candidates] == [DecisionSource.CRUISE, DecisionSource.SPEED_LIMIT]
  assert decision.fallback_reason == ""


def test_low_speed_source_stability_holds_recent_advisory_release():
  arbiter = LongitudinalArbiter()
  cruise = make_candidate(DecisionSource.CRUISE, CandidateRole.DRIVER_INTENT, 5.0, 0.2, 1.0, 0.1, "driver_set_speed")
  scc_vision = make_candidate(DecisionSource.SCC_VISION, CandidateRole.ADVISORY_CAP, 3.0, -0.3, 0.8, 0.5, "vision_curve")

  decision = resolve_longitudinal_decision(
    enabled=True,
    candidates=[cruise, scc_vision],
    fallback_v_target=5.0,
    fallback_a_target=0.2,
    fallback_should_stop=False,
    accel_limits=(-1.2, 1.0),
    arbiter=arbiter,
    v_ego=0.4,
  )

  assert decision.winner == DecisionSource.SCC_VISION

  for _ in range(SOURCE_STABILITY_RELEASE_FRAMES):
    decision = resolve_longitudinal_decision(
      enabled=True,
      candidates=[cruise],
      fallback_v_target=5.0,
      fallback_a_target=0.2,
      fallback_should_stop=False,
      accel_limits=(-1.2, 1.0),
      arbiter=arbiter,
      v_ego=0.4,
    )

    assert decision.winner == DecisionSource.SCC_VISION
    assert decision.a_target == pytest.approx(-0.3)
    assert (DecisionSource.CRUISE, "source_stability_hold") in decision.suppressed

  decision = resolve_longitudinal_decision(
    enabled=True,
    candidates=[cruise],
    fallback_v_target=5.0,
    fallback_a_target=0.2,
    fallback_should_stop=False,
    accel_limits=(-1.2, 1.0),
    arbiter=arbiter,
    v_ego=0.4,
  )

  assert decision.winner == DecisionSource.CRUISE
  assert decision.a_target == pytest.approx(0.2)


def test_source_stability_none_v_ego_clears_seeded_low_speed_hold():
  arbiter = LongitudinalArbiter()
  cruise = make_candidate(DecisionSource.CRUISE, CandidateRole.DRIVER_INTENT, 5.0, 0.2, 1.0, 0.1, "driver_set_speed")
  scc_vision = make_candidate(DecisionSource.SCC_VISION, CandidateRole.ADVISORY_CAP, 3.0, -0.3, 0.8, 0.5, "vision_curve")

  first_decision = resolve_longitudinal_decision(
    enabled=True,
    candidates=[cruise, scc_vision],
    fallback_v_target=5.0,
    fallback_a_target=0.2,
    fallback_should_stop=False,
    accel_limits=(-1.2, 1.0),
    arbiter=arbiter,
    v_ego=0.4,
  )
  held_decision = resolve_longitudinal_decision(
    enabled=True,
    candidates=[cruise],
    fallback_v_target=5.0,
    fallback_a_target=0.2,
    fallback_should_stop=False,
    accel_limits=(-1.2, 1.0),
    arbiter=arbiter,
    v_ego=0.4,
  )
  reset_decision = resolve_longitudinal_decision(
    enabled=True,
    candidates=[cruise],
    fallback_v_target=5.0,
    fallback_a_target=0.2,
    fallback_should_stop=False,
    accel_limits=(-1.2, 1.0),
    arbiter=arbiter,
    v_ego=None,
  )
  next_low_speed_decision = resolve_longitudinal_decision(
    enabled=True,
    candidates=[cruise],
    fallback_v_target=5.0,
    fallback_a_target=0.2,
    fallback_should_stop=False,
    accel_limits=(-1.2, 1.0),
    arbiter=arbiter,
    v_ego=0.4,
  )

  assert first_decision.winner == DecisionSource.SCC_VISION
  assert held_decision.winner == DecisionSource.SCC_VISION
  assert (DecisionSource.CRUISE, "source_stability_hold") in held_decision.suppressed
  assert reset_decision.winner == DecisionSource.CRUISE
  assert next_low_speed_decision.winner == DecisionSource.CRUISE
  assert not any(reason == "source_stability_hold" for _, reason in next_low_speed_decision.suppressed)


def test_low_speed_source_stability_allows_immediate_safer_physical_hazard():
  arbiter = LongitudinalArbiter()
  cruise = make_candidate(DecisionSource.CRUISE, CandidateRole.DRIVER_INTENT, 5.0, 0.2, 1.0, 0.1, "driver_set_speed")
  scc_vision = make_candidate(DecisionSource.SCC_VISION, CandidateRole.ADVISORY_CAP, 3.0, -0.3, 0.8, 0.5, "vision_curve")
  lead = make_candidate(DecisionSource.LEAD_MPC, CandidateRole.PHYSICAL_HAZARD, 5.0, -0.8, 0.9, 0.8, "confirmed_lead")

  first_decision = resolve_longitudinal_decision(
    enabled=True,
    candidates=[cruise, scc_vision],
    fallback_v_target=5.0,
    fallback_a_target=0.2,
    fallback_should_stop=False,
    accel_limits=(-1.2, 1.0),
    arbiter=arbiter,
    v_ego=0.4,
  )
  decision = resolve_longitudinal_decision(
    enabled=True,
    candidates=[cruise, lead],
    fallback_v_target=5.0,
    fallback_a_target=-0.8,
    fallback_should_stop=False,
    accel_limits=(-1.2, 1.0),
    arbiter=arbiter,
    v_ego=0.4,
  )

  assert first_decision.winner == DecisionSource.SCC_VISION
  assert decision.winner == DecisionSource.LEAD_MPC
  assert decision.a_target == pytest.approx(-0.8)
  assert (DecisionSource.LEAD_MPC, "source_stability_hold") not in decision.suppressed


def test_low_speed_source_stability_allows_immediate_safer_advisory_cap():
  arbiter = LongitudinalArbiter()
  cruise = make_candidate(DecisionSource.CRUISE, CandidateRole.DRIVER_INTENT, 5.0, 0.2, 1.0, 0.1, "driver_set_speed")
  scc_vision = make_candidate(DecisionSource.SCC_VISION, CandidateRole.ADVISORY_CAP, 3.0, -0.2, 0.8, 0.5, "vision_curve")
  speed_limit = make_candidate(DecisionSource.SPEED_LIMIT, CandidateRole.ADVISORY_CAP, 2.0, -0.1, 0.9, 0.5, "lower_limit")

  first_decision = resolve_longitudinal_decision(
    enabled=True,
    candidates=[cruise, scc_vision],
    fallback_v_target=5.0,
    fallback_a_target=0.2,
    fallback_should_stop=False,
    accel_limits=(-1.2, 1.0),
    arbiter=arbiter,
    v_ego=0.4,
  )
  decision = resolve_longitudinal_decision(
    enabled=True,
    candidates=[cruise, speed_limit],
    fallback_v_target=5.0,
    fallback_a_target=-0.1,
    fallback_should_stop=False,
    accel_limits=(-1.2, 1.0),
    arbiter=arbiter,
    v_ego=0.4,
  )

  assert first_decision.winner == DecisionSource.SCC_VISION
  assert decision.winner == DecisionSource.SPEED_LIMIT
  assert decision.v_target == pytest.approx(2.0)


def test_disabled_resolver_clears_source_stability_without_changing_fallback():
  arbiter = LongitudinalArbiter()
  cruise = make_candidate(DecisionSource.CRUISE, CandidateRole.DRIVER_INTENT, 5.0, 0.2, 1.0, 0.1, "driver_set_speed")
  scc_vision = make_candidate(DecisionSource.SCC_VISION, CandidateRole.ADVISORY_CAP, 3.0, -0.3, 0.8, 0.5, "vision_curve")

  first_decision = resolve_longitudinal_decision(
    enabled=True,
    candidates=[cruise, scc_vision],
    fallback_v_target=5.0,
    fallback_a_target=0.2,
    fallback_should_stop=False,
    accel_limits=(-1.2, 1.0),
    arbiter=arbiter,
    v_ego=0.4,
  )
  disabled_decision = resolve_longitudinal_decision(
    enabled=False,
    candidates=[cruise],
    fallback_v_target=5.0,
    fallback_a_target=0.4,
    fallback_should_stop=True,
    accel_limits=(-1.2, 1.0),
    arbiter=arbiter,
    v_ego=0.4,
  )
  enabled_decision = resolve_longitudinal_decision(
    enabled=True,
    candidates=[cruise],
    fallback_v_target=5.0,
    fallback_a_target=0.2,
    fallback_should_stop=False,
    accel_limits=(-1.2, 1.0),
    arbiter=arbiter,
    v_ego=0.4,
  )

  assert first_decision.winner == DecisionSource.SCC_VISION
  assert not disabled_decision.enabled
  assert disabled_decision.winner == DecisionSource.LEGACY_FALLBACK
  assert disabled_decision.a_target == pytest.approx(0.4)
  assert disabled_decision.should_stop
  assert disabled_decision.fallback_reason == "feature_flag_disabled"
  assert enabled_decision.winner == DecisionSource.CRUISE


def test_apply_decision_output_held_lead_release_uses_held_accel_conservatively():
  arbiter = LongitudinalArbiter()
  cruise = make_candidate(DecisionSource.CRUISE, CandidateRole.DRIVER_INTENT, 5.0, 0.2, 1.0, 0.1, "driver_set_speed")
  lead = make_candidate(DecisionSource.LEAD_MPC, CandidateRole.PHYSICAL_HAZARD, 5.0, -0.5, 0.9, 0.8, "confirmed_lead")

  first_decision = resolve_longitudinal_decision(
    enabled=True,
    candidates=[cruise, lead],
    fallback_v_target=5.0,
    fallback_a_target=-0.5,
    fallback_should_stop=False,
    accel_limits=(-1.2, 1.0),
    arbiter=arbiter,
    v_ego=0.4,
  )
  held_decision = resolve_longitudinal_decision(
    enabled=True,
    candidates=[cruise],
    fallback_v_target=5.0,
    fallback_a_target=0.2,
    fallback_should_stop=False,
    accel_limits=(-1.2, 1.0),
    arbiter=arbiter,
    v_ego=0.4,
  )

  release_a_target, release_should_stop = apply_longitudinal_decision_output(
    held_decision, legacy_a_target=0.2, legacy_should_stop=False,
  )
  release_telemetry = apply_longitudinal_decision_output_with_telemetry(
    held_decision, legacy_a_target=0.2, legacy_should_stop=False,
  )
  stronger_legacy_a_target, stronger_legacy_should_stop = apply_longitudinal_decision_output(
    held_decision, legacy_a_target=-0.8, legacy_should_stop=False,
  )

  assert first_decision.winner == DecisionSource.LEAD_MPC
  assert held_decision.winner == DecisionSource.LEAD_MPC
  assert (DecisionSource.CRUISE, SOURCE_STABILITY_HOLD_REASON) in held_decision.suppressed
  assert release_a_target == pytest.approx(-0.5)
  assert not release_should_stop
  assert release_telemetry.raw_source == DecisionSource.LEAD_MPC
  assert release_telemetry.raw_active_reason == "confirmed_lead"
  assert release_telemetry.applied_a_target == pytest.approx(release_a_target)
  assert release_telemetry.applied_should_stop == release_should_stop
  assert release_telemetry.applied_reason == SOURCE_STABILITY_HOLD_REASON
  assert release_telemetry.accel_delta == pytest.approx(0.0)
  assert stronger_legacy_a_target == pytest.approx(-0.8)
  assert not stronger_legacy_should_stop


def test_apply_decision_output_held_release_bypasses_comfort_smoothing():
  arbiter = LongitudinalArbiter()
  cruise = make_candidate(DecisionSource.CRUISE, CandidateRole.DRIVER_INTENT, 5.0, 0.2, 1.0, 0.1, "driver_set_speed")
  lead = make_candidate(DecisionSource.LEAD_MPC, CandidateRole.PHYSICAL_HAZARD, 5.0, -0.5, 0.9, 0.8, "confirmed_lead")

  resolve_longitudinal_decision(
    enabled=True,
    candidates=[cruise, lead],
    fallback_v_target=5.0,
    fallback_a_target=-0.5,
    fallback_should_stop=False,
    accel_limits=(-1.2, 1.0),
    arbiter=arbiter,
    v_ego=0.4,
  )
  held_decision = resolve_longitudinal_decision(
    enabled=True,
    candidates=[cruise],
    fallback_v_target=5.0,
    fallback_a_target=0.2,
    fallback_should_stop=False,
    accel_limits=(-1.2, 1.0),
    arbiter=arbiter,
    v_ego=0.4,
  )

  a_target, should_stop = apply_longitudinal_decision_output(
    held_decision,
    legacy_a_target=0.2,
    legacy_should_stop=False,
    prev_a_target=0.0,
    personality=log.LongitudinalPersonality.standard,
    dt=0.05,
    comfort_active=True,
  )

  assert held_decision.winner == DecisionSource.LEAD_MPC
  assert (DecisionSource.CRUISE, "source_stability_hold") in held_decision.suppressed
  assert a_target == pytest.approx(-0.5)
  assert not should_stop


def test_apply_decision_output_held_e2e_release_uses_held_accel_conservatively():
  arbiter = LongitudinalArbiter()
  cruise = make_candidate(DecisionSource.CRUISE, CandidateRole.DRIVER_INTENT, 5.0, 0.2, 1.0, 0.1, "driver_set_speed")
  e2e = make_candidate(DecisionSource.E2E_STOP, CandidateRole.PHYSICAL_HAZARD, 5.0, -0.4, 0.85, 0.6, "model_stop")

  first_decision = resolve_longitudinal_decision(
    enabled=True,
    candidates=[cruise, e2e],
    fallback_v_target=5.0,
    fallback_a_target=-0.4,
    fallback_should_stop=False,
    accel_limits=(-1.2, 1.0),
    arbiter=arbiter,
    v_ego=0.4,
  )
  held_decision = resolve_longitudinal_decision(
    enabled=True,
    candidates=[cruise],
    fallback_v_target=5.0,
    fallback_a_target=0.2,
    fallback_should_stop=False,
    accel_limits=(-1.2, 1.0),
    arbiter=arbiter,
    v_ego=0.4,
  )

  release_a_target, release_should_stop = apply_longitudinal_decision_output(
    held_decision, legacy_a_target=0.2, legacy_should_stop=False,
  )
  stronger_legacy_a_target, stronger_legacy_should_stop = apply_longitudinal_decision_output(
    held_decision, legacy_a_target=-0.8, legacy_should_stop=False,
  )

  assert first_decision.winner == DecisionSource.E2E_STOP
  assert held_decision.winner == DecisionSource.E2E_STOP
  assert (DecisionSource.CRUISE, "source_stability_hold") in held_decision.suppressed
  assert release_a_target == pytest.approx(-0.4)
  assert not release_should_stop
  assert stronger_legacy_a_target == pytest.approx(-0.8)
  assert not stronger_legacy_should_stop


def test_apply_decision_output_cruise_winner_preserves_legacy_accel():
  decision = make_decision(DecisionSource.CRUISE, a_target=0.4, should_stop=False)

  a_target, should_stop = apply_longitudinal_decision_output(decision, legacy_a_target=-0.3, legacy_should_stop=True)

  assert a_target == pytest.approx(-0.3)
  assert should_stop


def test_apply_decision_output_cruise_winner_respects_personality_comfort_context():
  decision = make_decision(DecisionSource.CRUISE, a_target=0.6, should_stop=False)

  a_target, should_stop = apply_longitudinal_decision_output(
    decision,
    legacy_a_target=0.6,
    legacy_should_stop=False,
    prev_a_target=0.0,
    personality=log.LongitudinalPersonality.relaxed,
    dt=0.05,
  )

  assert 0.0 < a_target < 0.6
  assert not should_stop


def test_apply_decision_output_lead_winner_preserves_legacy_accel_and_stop():
  decision = make_decision(DecisionSource.LEAD_MPC, a_target=-0.2, should_stop=False)

  a_target, should_stop = apply_longitudinal_decision_output(decision, legacy_a_target=-0.8, legacy_should_stop=True)

  assert a_target == pytest.approx(-0.8)
  assert should_stop


def test_apply_decision_output_advisory_cannot_relax_stronger_legacy_braking():
  decision = make_decision(DecisionSource.SPEED_LIMIT, a_target=-0.2, should_stop=False, active_reason="advisory_limit")

  a_target, should_stop = apply_longitudinal_decision_output(decision, legacy_a_target=-0.7, legacy_should_stop=False)
  telemetry = apply_longitudinal_decision_output_with_telemetry(
    decision, legacy_a_target=-0.7, legacy_should_stop=False,
  )

  assert a_target == pytest.approx(-0.7)
  assert not should_stop
  assert telemetry.raw_source == DecisionSource.SPEED_LIMIT
  assert telemetry.raw_a_target == pytest.approx(-0.2)
  assert telemetry.raw_active_reason == "advisory_limit"
  assert telemetry.legacy_a_target == pytest.approx(-0.7)
  assert telemetry.applied_a_target == pytest.approx(a_target)
  assert telemetry.applied_should_stop == should_stop
  assert telemetry.applied_reason == "advisory_min_legacy"
  assert telemetry.accel_delta == pytest.approx(-0.5)


def test_apply_decision_output_cruise_coast_can_relax_legacy_braking():
  decision = make_decision(DecisionSource.CRUISE_COAST, a_target=-0.2, should_stop=False)

  a_target, should_stop = apply_longitudinal_decision_output(decision, legacy_a_target=-0.8, legacy_should_stop=False)

  assert a_target == pytest.approx(-0.2)
  assert not should_stop


def test_apply_decision_output_can_apply_personality_comfort_context():
  decision = make_decision(DecisionSource.CRUISE_COAST, a_target=0.6, should_stop=False)

  a_target, should_stop = apply_longitudinal_decision_output(
    decision,
    legacy_a_target=0.0,
    legacy_should_stop=False,
    prev_a_target=0.0,
    personality=log.LongitudinalPersonality.relaxed,
    dt=0.05,
  )

  assert 0.0 < a_target < 0.6
  assert not should_stop


def test_apply_decision_output_can_disable_personality_comfort_context():
  decision = make_decision(DecisionSource.CRUISE_COAST, a_target=0.6, should_stop=False)

  a_target, should_stop = apply_longitudinal_decision_output(
    decision,
    legacy_a_target=0.0,
    legacy_should_stop=False,
    prev_a_target=0.0,
    personality=log.LongitudinalPersonality.relaxed,
    dt=0.05,
    comfort_active=False,
  )

  assert a_target == pytest.approx(0.6)
  assert not should_stop


def test_personality_comfort_limits_positive_accel_rise_by_personality():
  decision = make_decision(DecisionSource.CRUISE_COAST, a_target=0.6, should_stop=False)

  relaxed = apply_personality_accel_comfort(decision, 0.6, prev_a_target=0.0, personality=log.LongitudinalPersonality.relaxed, dt=0.05)
  standard = apply_personality_accel_comfort(decision, 0.6, prev_a_target=0.0, personality=log.LongitudinalPersonality.standard, dt=0.05)
  aggressive = apply_personality_accel_comfort(decision, 0.6, prev_a_target=0.0, personality=log.LongitudinalPersonality.aggressive, dt=0.05)

  assert 0.0 < relaxed < standard < aggressive < 0.6


def test_personality_comfort_softens_only_mild_brake_onset():
  decision = make_decision(DecisionSource.SPEED_LIMIT, a_target=-0.25, should_stop=False)

  smoothed = apply_personality_accel_comfort(decision, -0.25, prev_a_target=0.1, personality=log.LongitudinalPersonality.relaxed, dt=0.05)

  assert -0.25 < smoothed < 0.1


def test_personality_comfort_bypasses_stop_and_hazard_braking():
  stop_decision = make_decision(DecisionSource.E2E_STOP, a_target=-1.0, should_stop=True)
  lead_decision = make_decision(DecisionSource.LEAD_MPC, a_target=-0.8, should_stop=False)

  assert apply_personality_accel_comfort(
    stop_decision, -1.0, prev_a_target=0.2, personality=log.LongitudinalPersonality.relaxed, dt=0.05,
  ) == pytest.approx(-1.0)
  assert apply_personality_accel_comfort(
    lead_decision, -0.8, prev_a_target=0.2, personality=log.LongitudinalPersonality.relaxed, dt=0.05,
  ) == pytest.approx(-0.8)


def test_personality_comfort_bypasses_large_decel_delta_and_bad_previous_accel():
  decision = make_decision(DecisionSource.SPEED_LIMIT, a_target=-0.9, should_stop=False)

  assert apply_personality_accel_comfort(
    decision, -0.9, prev_a_target=0.1, personality=log.LongitudinalPersonality.relaxed, dt=0.05,
  ) == pytest.approx(-0.9)
  assert apply_personality_accel_comfort(
    decision, 0.4, prev_a_target=math.nan, personality=log.LongitudinalPersonality.relaxed, dt=0.05,
  ) == pytest.approx(0.4)


def test_core_candidate_builder_adds_confirmed_lead_candidate():
  candidates = build_core_longitudinal_candidates(
    has_lead=True,
    lead_confidence=0.82,
    v_cruise=27.0,
    a_cruise=0.1,
    output_a_target_mpc=-0.7,
    output_should_stop_mpc=False,
    e2e_active=False,
    output_a_target_e2e=0.0,
    output_should_stop_e2e=False,
    e2e_stop_approach_a_target=0.0,
    cruise_coast_applied=False,
    cruise_coast_a_target=0.0,
  )

  lead = next(candidate for candidate in candidates if candidate.source == DecisionSource.LEAD_MPC)
  assert lead.role == CandidateRole.PHYSICAL_HAZARD
  assert lead.confidence == pytest.approx(0.82)
  assert lead.a_target == -0.7


def test_core_candidate_builder_preserves_low_lead_confidence():
  candidates = build_core_longitudinal_candidates(
    has_lead=True,
    lead_confidence=0.2,
    v_cruise=27.0,
    a_cruise=0.1,
    output_a_target_mpc=-0.7,
    output_should_stop_mpc=False,
    e2e_active=False,
    output_a_target_e2e=0.0,
    output_should_stop_e2e=False,
    e2e_stop_approach_a_target=0.0,
    cruise_coast_applied=False,
    cruise_coast_a_target=0.0,
  )

  lead = next(candidate for candidate in candidates if candidate.source == DecisionSource.LEAD_MPC)
  assert lead.confidence == pytest.approx(0.2)


def test_active_lead_confidence_uses_highest_active_model_probability():
  lead_one = SimpleNamespace(status=False, modelProb=0.95)
  lead_two = SimpleNamespace(status=True, modelProb=0.82)

  assert get_active_lead_confidence(lead_one, lead_two) == pytest.approx(0.82)


def test_active_lead_confidence_treats_missing_model_probability_as_confirmed():
  lead_one = SimpleNamespace(status=True, modelProb=0.2)
  lead_two = SimpleNamespace(status=True)

  assert get_active_lead_confidence(lead_one, lead_two) == pytest.approx(1.0)


def test_core_candidate_builder_adds_e2e_stop_candidate_for_active_stop():
  candidates = build_core_longitudinal_candidates(
    has_lead=False,
    lead_confidence=0.0,
    v_cruise=27.0,
    a_cruise=0.1,
    output_a_target_mpc=0.0,
    output_should_stop_mpc=False,
    e2e_active=True,
    output_a_target_e2e=-1.0,
    output_should_stop_e2e=True,
    e2e_stop_approach_a_target=0.0,
    cruise_coast_applied=False,
    cruise_coast_a_target=0.0,
  )

  e2e = next(candidate for candidate in candidates if candidate.source == DecisionSource.E2E_STOP)
  assert e2e.role == CandidateRole.PHYSICAL_HAZARD
  assert e2e.should_stop
  assert e2e.confidence == pytest.approx(0.85)


def test_core_candidate_builder_adds_cruise_coast_comfort_candidate():
  candidates = build_core_longitudinal_candidates(
    has_lead=False,
    lead_confidence=0.0,
    v_cruise=25.0,
    a_cruise=-0.8,
    output_a_target_mpc=-0.8,
    output_should_stop_mpc=False,
    e2e_active=False,
    output_a_target_e2e=0.0,
    output_should_stop_e2e=False,
    e2e_stop_approach_a_target=0.0,
    cruise_coast_applied=True,
    cruise_coast_a_target=-0.2,
  )

  coast = next(candidate for candidate in candidates if candidate.source == DecisionSource.CRUISE_COAST)
  assert coast.role == CandidateRole.RELAXATION
  assert coast.a_target == -0.2


def test_driver_intent_wins_over_low_confidence_cut_in():
  arbiter = LongitudinalArbiter()
  cruise = make_candidate(DecisionSource.CRUISE, CandidateRole.DRIVER_INTENT, 27.0, 0.1, 1.0, 0.1, "driver_set_speed")
  cut_in = make_candidate(DecisionSource.LEAD_MPC, CandidateRole.PHYSICAL_HAZARD, 20.0, -0.6, 0.3, 0.8, "low_confidence_cut_in")

  decision = arbiter.decide([cruise, cut_in])

  assert decision.winner == DecisionSource.CRUISE
  assert (DecisionSource.LEAD_MPC, "low_confidence") in decision.suppressed


def test_osm_caution_does_not_override_confirmed_lead():
  arbiter = LongitudinalArbiter()
  cruise = make_candidate(DecisionSource.CRUISE, CandidateRole.DRIVER_INTENT, 25.0, 0.1, 1.0, 0.1, "driver_set_speed")
  osm = make_candidate(DecisionSource.OSM_TRAFFIC_CONTROL, CandidateRole.ADVISORY_CAP, 8.33, -0.4, 0.8, 0.5, "map_caution")
  lead = make_candidate(DecisionSource.LEAD_MPC, CandidateRole.PHYSICAL_HAZARD, 15.0, -0.7, 0.9, 0.8, "confirmed_lead")

  decision = arbiter.decide([cruise, osm, lead])

  assert decision.winner == DecisionSource.LEAD_MPC
  assert (DecisionSource.OSM_TRAFFIC_CONTROL, "physical_hazard_active") in decision.suppressed


def test_confident_curve_can_limit_overspeed_when_no_hazard():
  arbiter = LongitudinalArbiter()
  cruise = make_candidate(DecisionSource.CRUISE, CandidateRole.DRIVER_INTENT, 30.0, -0.1, 1.0, 0.1, "driver_set_speed")
  curve = make_candidate(DecisionSource.SCC_VISION, CandidateRole.ADVISORY_CAP, 18.0, -0.5, 0.85, 0.6, "confident_vision_curve")
  coast = make_candidate(DecisionSource.CRUISE_COAST, CandidateRole.RELAXATION, 30.0, 0.0, 0.8, 0.2, "harmless_overspeed")

  decision = arbiter.decide([cruise, curve, coast])

  assert decision.winner == DecisionSource.SCC_VISION
  assert decision.a_target == -0.5
