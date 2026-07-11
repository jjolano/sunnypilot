"""Invariant (property) tests for the longitudinal decision core.

Gate the safety invariants of candidate/authority arbitration: candidates only restrict, a
physical hazard always binds, mode-excluded evidence can never act, comfort relax cannot
override a hazard, fail-closed on bad input. These do not certify feel.
"""
from __future__ import annotations

import numpy as np
import pytest

from openpilot.sunnypilot.custom.longitudinal.decision import (
  CandidateRole,
  Decision,
  LongitudinalCandidate as C,
  decide,
)
from openpilot.sunnypilot.custom.longitudinal.modes import EvidenceClass, LongitudinalMode, SourceToggles

LIMITS = (-4.0, 2.0)


def cruise(a, src=EvidenceClass.CRUISE):
  return C(a, CandidateRole.CRUISE, src, "cruise")


def test_cruise_only_passes_through_within_limits():
  d = decide([cruise(1.5)], LongitudinalMode.ACC, LIMITS)
  assert d.a_target == pytest.approx(1.5)
  assert d.should_stop is False


def test_physical_hazard_always_binds_decel():
  cands = [cruise(2.0), C(-3.0, CandidateRole.PHYSICAL_HAZARD, EvidenceClass.LEAD, "lead", is_stop=False)]
  d = decide(cands, LongitudinalMode.ACC, LIMITS)
  assert d.a_target == pytest.approx(-3.0)
  assert d.reason == "physical_hazard"


def test_advisory_cap_restricts_but_never_raises():
  capped = decide([cruise(2.0), C(0.5, CandidateRole.ADVISORY_CAP, EvidenceClass.MODEL_STOP, "soft")],
                  LongitudinalMode.E2E, LIMITS)
  assert capped.a_target <= 2.0
  assert capped.a_target == pytest.approx(0.5)


def test_comfort_relax_cannot_exceed_hazard():
  # relax tries to soften toward +1.0, but a hazard demands -2.0 -> hazard wins
  cands = [
    cruise(2.0),
    C(-1.0, CandidateRole.ADVISORY_CAP, EvidenceClass.MODEL_STOP, "curve"),
    C(1.0, CandidateRole.COMFORT_RELAX, EvidenceClass.MODEL_STOP, "relax"),
    C(-2.0, CandidateRole.PHYSICAL_HAZARD, EvidenceClass.LEAD, "lead"),
  ]
  d = decide(cands, LongitudinalMode.E2E, LIMITS)
  assert d.a_target == pytest.approx(-2.0)  # hazard binds despite relax


def test_comfort_relax_softens_advisory_within_bounds():
  # no hazard: relax raises the advisory-capped accel toward the relax floor, never above cruise
  cands = [
    cruise(2.0),
    C(-1.0, CandidateRole.ADVISORY_CAP, EvidenceClass.MODEL_STOP, "curve"),
    C(0.2, CandidateRole.COMFORT_RELAX, EvidenceClass.MODEL_STOP, "relax"),
  ]
  d = decide(cands, LongitudinalMode.E2E, LIMITS)
  assert -1.0 <= d.a_target <= 0.2 + 1e-9
  assert d.a_target == pytest.approx(0.2)


def test_mode_excluded_evidence_never_acts():
  # ACC must ignore a MODEL_STOP hazard entirely (excluded source)
  cands = [cruise(1.0), C(-3.5, CandidateRole.PHYSICAL_HAZARD, EvidenceClass.MODEL_STOP, "model_stop", is_stop=True)]
  d = decide(cands, LongitudinalMode.ACC, LIMITS)
  assert d.a_target == pytest.approx(1.0)   # hazard excluded -> cruise stands
  assert d.should_stop is False
  assert any("mode_excluded" in r for r in d.rejected)


def test_same_hazard_acts_in_e2e():
  cands = [cruise(1.0), C(-3.5, CandidateRole.PHYSICAL_HAZARD, EvidenceClass.MODEL_STOP, "model_stop", is_stop=True)]
  d = decide(cands, LongitudinalMode.E2E, LIMITS)
  assert d.a_target == pytest.approx(-3.5)
  assert d.should_stop is True


def test_progress_only_when_authorized():
  base = [cruise(0.0)]
  unauth = decide(base + [C(1.5, CandidateRole.PROGRESS, EvidenceClass.LEAD, "pullaway", authorized=False)],
                  LongitudinalMode.ACC, LIMITS)
  auth = decide(base + [C(1.5, CandidateRole.PROGRESS, EvidenceClass.LEAD, "pullaway", authorized=True)],
                LongitudinalMode.ACC, LIMITS)
  assert unauth.a_target == pytest.approx(0.0)
  assert auth.a_target == pytest.approx(1.5)


def test_selected_intent_tracks_winning_desire_candidate():
  def progress(a, intent):
    return C(a, CandidateRole.PROGRESS, EvidenceClass.LEAD, intent)

  winning_progress = decide([cruise(0.4), progress(0.2, "weak"), progress(0.8, "winner")],
                            LongitudinalMode.ACC, LIMITS)
  assert winning_progress.a_target == pytest.approx(0.8)
  assert winning_progress.selected_intent == "winner"

  winning_cruise = decide([cruise(1.0), progress(0.8, "pullaway")], LongitudinalMode.ACC, LIMITS)
  assert winning_cruise.a_target == pytest.approx(1.0)
  assert winning_cruise.selected_intent == "cruise"

  tied = decide([cruise(0.8), progress(0.8, "pullaway")], LongitudinalMode.ACC, LIMITS)
  assert tied.a_target == pytest.approx(0.8)
  assert tied.selected_intent == "pullaway"


def test_output_always_within_accel_limits():
  rng = np.random.default_rng(20260613)
  roles = list(CandidateRole)
  srcs = list(EvidenceClass)
  for _ in range(3000):
    n = int(rng.integers(0, 5))
    cands = [C(float(rng.uniform(-8, 8)), roles[int(rng.integers(0, len(roles)))],
               srcs[int(rng.integers(0, len(srcs)))], "x", bool(rng.random() > 0.3),
               bool(rng.random() > 0.8)) for _ in range(n)]
    mode = [LongitudinalMode.ACC, LongitudinalMode.E2E, LongitudinalMode.SCC][int(rng.integers(0, 3))]
    d = decide(cands, mode, LIMITS, SourceToggles(bool(rng.random() > 0.5), bool(rng.random() > 0.5)))
    assert LIMITS[0] - 1e-9 <= d.a_target <= LIMITS[1] + 1e-9


def test_fail_closed_on_bad_limits_and_nonfinite():
  bad = decide([cruise(1.0)], LongitudinalMode.ACC, (2.0, -2.0))
  assert bad.reason == "invalid_accel_limits"
  # non-finite cruise seed/candidate defaults fail-closed to 0.0, not +a_max
  d = decide([cruise(float("nan"))], LongitudinalMode.ACC, LIMITS)
  assert d.a_target == pytest.approx(0.0)
  assert any("nonfinite" in r for r in d.rejected)


def test_returns_decision_type():
  assert isinstance(decide([cruise(1.0)], LongitudinalMode.ACC, LIMITS), Decision)


def test_physical_hazards_still_bind_despite_lead_soft_pair():
  # A lead-soft advisory + progress pair does not override a separate physical hazard.
  cands = [
    cruise(-0.3),
    C(-0.05, CandidateRole.ADVISORY_CAP, EvidenceClass.LEAD, "lead_follow_soft"),
    C(-0.05, CandidateRole.PROGRESS, EvidenceClass.LEAD, "lead_follow_soft_desire", authorized=True),
    C(-2.0, CandidateRole.PHYSICAL_HAZARD, EvidenceClass.MODEL_STOP, "stop_approach"),
  ]
  d = decide(cands, LongitudinalMode.E2E, LIMITS)
  assert d.a_target == pytest.approx(-2.0)
  assert d.reason == "physical_hazard"


def test_lead_soft_pair_changes_output_only_through_lead_evidence():
  # Same advisory/progress values, but sourced as LEAD (admitted in ACC) vs MODEL_STOP (excluded
  # in ACC). The output only changes when the evidence class is admitted.
  soft_lead = [
    cruise(-0.3),
    C(-0.05, CandidateRole.ADVISORY_CAP, EvidenceClass.LEAD, "lead_follow_soft"),
    C(-0.05, CandidateRole.PROGRESS, EvidenceClass.LEAD, "lead_follow_soft_desire", authorized=True),
  ]
  soft_model_stop = [
    cruise(-0.3),
    C(-0.05, CandidateRole.ADVISORY_CAP, EvidenceClass.MODEL_STOP, "lead_follow_soft"),
    C(-0.05, CandidateRole.PROGRESS, EvidenceClass.MODEL_STOP, "lead_follow_soft_desire", authorized=True),
  ]
  assert decide(soft_lead, LongitudinalMode.ACC, LIMITS).a_target == pytest.approx(-0.05)
  assert decide(soft_model_stop, LongitudinalMode.ACC, LIMITS).a_target == pytest.approx(-0.3)


def test_unrelated_scc_caps_still_bind_with_lead_soft_pair():
  # The lead-soft path must not weaken unrelated SCC advisory caps.
  cands = [
    cruise(-0.3),
    C(-0.05, CandidateRole.ADVISORY_CAP, EvidenceClass.LEAD, "lead_follow_soft"),
    C(-0.05, CandidateRole.PROGRESS, EvidenceClass.LEAD, "lead_follow_soft_desire", authorized=True),
    C(-0.35, CandidateRole.ADVISORY_CAP, EvidenceClass.CURVE_VISION, "curve_policy"),
  ]
  d = decide(cands, LongitudinalMode.SCC, LIMITS, SourceToggles(scc_curve_vision_enabled=True))
  assert d.a_target == pytest.approx(-0.35)
  assert d.reason == "advisory_capped"
