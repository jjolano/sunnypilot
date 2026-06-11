from types import SimpleNamespace

import pytest

from openpilot.selfdrive.controls.lib.longitudinal_decision import (
  ADVISORY_CAP_ACTIVE_REASON,
  CandidateRole,
  DecisionSource,
  LongitudinalArbiter,
  LongitudinalCandidate,
)
from openpilot.selfdrive.controls.lib.longitudinal_modes import (
  LONGITUDINAL_MODE_PARAM,
  LongitudinalMode,
  ResolvedLongitudinalImplementation,
  SccModeEvidence,
  resolve_longitudinal_mode,
)
from openpilot.selfdrive.controls.lib.longitudinal_stacks.custom_v2 import (
  CustomV2Scene,
  build_custom_v2_advisory_candidates,
)


class FakeParams:
  def __init__(self, mode):
    self.values = {LONGITUDINAL_MODE_PARAM: int(mode)}

  def get(self, key, *args, **kwargs):
    return self.values.get(key)

  def get_bool(self, key):
    return bool(self.values.get(key))


def candidate(source, role, a_target, reason, *, v_target=20.0, confidence=1.0):
  return LongitudinalCandidate(
    source=source,
    role=role,
    v_target=v_target,
    a_target=a_target,
    confidence=confidence,
    urgency=0.5,
    active_reason=reason,
  )


def test_acc_hardware_ignores_scc_classifier_and_resolves_acc_like():
  class PoisonEvidence(SccModeEvidence):
    def classify(self):
      raise AssertionError("ACC must not classify SCC evidence")

  resolution = resolve_longitudinal_mode(
    FakeParams(LongitudinalMode.ACC),
    SimpleNamespace(radarUnavailable=False, openpilotLongitudinalControl=True),
    scc_evidence=PoisonEvidence(),
  )

  assert resolution.resolved_implementation == ResolvedLongitudinalImplementation.HARDWARE_ACC
  assert resolution.acc_like


def test_scc_confirmed_lead_normally_resolves_acc_like():
  resolution = resolve_longitudinal_mode(
    FakeParams(LongitudinalMode.SCC),
    SimpleNamespace(radarUnavailable=False, openpilotLongitudinalControl=True),
    scc_evidence=SccModeEvidence(confirmed_lead=True, model_stop=True),
  )

  assert resolution.resolved_implementation == ResolvedLongitudinalImplementation.SCC_ACC
  assert resolution.acc_like


def test_scc_independent_urgent_stop_resolves_e2e_like():
  resolution = resolve_longitudinal_mode(
    FakeParams(LongitudinalMode.SCC),
    SimpleNamespace(radarUnavailable=False, openpilotLongitudinalControl=True),
    scc_evidence=SccModeEvidence(confirmed_lead=True, urgent_stop=True, independent_of_lead=True),
  )

  assert resolution.resolved_implementation == ResolvedLongitudinalImplementation.SCC_E2E
  assert resolution.e2e_like


def test_acc_mode_boundary_blocks_custom_v2_sla_osm_curve_advisories():
  scene = CustomV2Scene(
    v_ego=12.0,
    v_cruise=20.0,
    speed_limit_active=True,
    speed_limit_v_target=8.0,
    speed_limit_a_target=-0.5,
    curve_active=True,
    curve_a_target=-0.6,
    map_caution_active=True,
    map_caution_confirmed=True,
    map_caution_a_target=-0.4,
  )

  candidates, rejected = build_custom_v2_advisory_candidates(
    scene, allow_speed_limit=False, allow_curve=False, allow_map_caution=False,
  )

  assert candidates == ()
  assert ("speed_policy", "mode_boundary_blocked") in rejected
  assert ("curve_policy", "mode_boundary_blocked") in rejected
  assert ("map_caution", "mode_boundary_blocked") in rejected


def test_e2e_advisory_cap_cannot_authorize_progress():
  cruise = candidate(DecisionSource.CRUISE, CandidateRole.DRIVER_INTENT, -0.1, "driver_cruise", v_target=20.0)
  advisory = candidate(DecisionSource.SPEED_LIMIT, CandidateRole.ADVISORY_CAP, -0.3, "sla_cap", v_target=10.0, confidence=0.9)
  progress = candidate(DecisionSource.STOP_LAUNCH, CandidateRole.RELAXATION, 0.8, "no_lead_progress", v_target=20.0)

  decision = LongitudinalArbiter().decide([cruise, advisory, progress])

  assert decision.winner == DecisionSource.SPEED_LIMIT
  assert any(
    suppressed.source == DecisionSource.STOP_LAUNCH and suppressed.suppression_reason == ADVISORY_CAP_ACTIVE_REASON
    for suppressed in decision.suppressed_candidates
  )


def test_physical_caps_remain_restrictive_above_advisories_and_relaxation():
  cruise = candidate(DecisionSource.CRUISE, CandidateRole.DRIVER_INTENT, 0.0, "driver_cruise", v_target=20.0)
  advisory = candidate(DecisionSource.SCC_VISION, CandidateRole.ADVISORY_CAP, -0.2, "curve_cap", v_target=15.0, confidence=0.9)
  progress = candidate(DecisionSource.STOP_LAUNCH, CandidateRole.RELAXATION, 0.8, "progress", v_target=20.0)
  physical = candidate(DecisionSource.LEAD_MPC, CandidateRole.PHYSICAL_HAZARD, -0.7, "lead", v_target=15.0, confidence=0.9)

  decision = LongitudinalArbiter().decide([cruise, advisory, progress, physical])

  assert decision.winner == DecisionSource.LEAD_MPC
  assert decision.a_target == pytest.approx(-0.7)
