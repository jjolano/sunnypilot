import math

from openpilot.selfdrive.controls.lib.longitudinal_decision import (
  CandidateRole,
  DecisionSource,
  LongitudinalCandidate,
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
