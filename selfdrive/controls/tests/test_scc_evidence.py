from openpilot.selfdrive.controls.lib.scc_evidence import (
  SccAdvisoryFlags,
  SccEvidenceSelector,
  SccEvidenceSelectorState,
  SccEvidenceTier,
  associate_model_stop_with_lead,
  classify_scc_evidence,
)


def test_no_evidence_classifies_none():
  result = classify_scc_evidence()

  assert result.tier == SccEvidenceTier.NONE
  assert result.reason == "scc_no_evidence"
  assert not result.e2e_active


def test_tier_aliases_keep_contract_labels():
  assert SccEvidenceTier.none.label == "none"
  assert SccEvidenceTier.URGENT_STOP.label == "urgent_stop"
  assert SccEvidenceTier.stop == SccEvidenceTier.STOP


def test_slowdown_stop_and_urgent_tiers_are_ordered():
  slowdown = classify_scc_evidence(model_slowdown=True)
  stop = classify_scc_evidence(model_stop=True, model_stop_distance=24.0)
  urgent = classify_scc_evidence(urgent_stop=True, model_stop_distance=8.0)

  assert slowdown.tier == SccEvidenceTier.SLOWDOWN
  assert stop.tier == SccEvidenceTier.STOP
  assert urgent.tier == SccEvidenceTier.URGENT_STOP
  assert slowdown.tier < stop.tier < urgent.tier
  assert urgent.confidence == 1.0
  assert urgent.urgency == 1.0


def test_speed_map_curve_and_traffic_are_parallel_advisories_only():
  flags = SccAdvisoryFlags(speed_limit_cap=True, map_caution=True, curve_cap=True, traffic_control_prior=True)
  result = classify_scc_evidence(advisories=flags)

  assert result.tier == SccEvidenceTier.NONE
  assert not result.e2e_active
  assert result.advisories.speed_limit_cap
  assert result.advisories.map_caution
  assert result.advisories.curve_cap
  assert result.advisories.traffic_control_prior
  assert result.advisory_status == ("map_caution", "speed_limit_cap", "curve_cap", "traffic_control_prior")


def test_confirmed_lead_associated_model_stop_stays_acc_like():
  result = classify_scc_evidence(
    confirmed_lead=True,
    model_stop=True,
    model_stop_distance=31.0,
    lead_distance=30.0,
    lead_path_y_rel=0.1,
    lead_idx=0,
    v_ego=12.0,
  )

  assert result.tier == SccEvidenceTier.STOP
  assert result.associated_lead_idx == 0
  assert not result.independent_of_lead
  assert not result.e2e_active


def test_confirmed_lead_independent_urgent_stop_can_be_e2e_like():
  result = classify_scc_evidence(
    confirmed_lead=True,
    urgent_stop=True,
    model_stop_distance=8.0,
    lead_distance=30.0,
    lead_path_y_rel=0.0,
    lead_idx=0,
    v_ego=12.0,
  )

  assert result.tier == SccEvidenceTier.URGENT_STOP
  assert result.associated_lead_idx is None
  assert result.independent_of_lead
  assert result.e2e_active


def test_confirmed_lead_stop_without_valid_geometry_fails_closed_to_e2e():
  stop = classify_scc_evidence(
    confirmed_lead=True,
    model_stop=True,
    model_stop_distance=18.0,
  )
  urgent = classify_scc_evidence(
    confirmed_lead=True,
    urgent_stop=True,
    model_stop_distance=8.0,
  )

  assert stop.tier == SccEvidenceTier.STOP
  assert stop.associated_lead_idx is None
  assert stop.independent_of_lead
  assert stop.e2e_active
  assert urgent.tier == SccEvidenceTier.URGENT_STOP
  assert urgent.associated_lead_idx is None
  assert urgent.independent_of_lead
  assert urgent.e2e_active


def test_no_lead_model_stop_is_independent_e2e_evidence():
  result = classify_scc_evidence(model_stop=True, model_stop_distance=18.0)

  assert result.tier == SccEvidenceTier.STOP
  assert result.independent_of_lead
  assert result.e2e_active


def test_lead_association_uses_distance_and_geometry_margins():
  associated = associate_model_stop_with_lead(
    confirmed_lead=True, model_stop_distance=29.0, lead_distance=30.0, lead_path_y_rel=0.2, lead_idx=1, v_ego=10.0,
  )
  before_lead = associate_model_stop_with_lead(
    confirmed_lead=True, model_stop_distance=10.0, lead_distance=30.0, lead_path_y_rel=0.2, lead_idx=1, v_ego=10.0,
  )
  after_lead = associate_model_stop_with_lead(
    confirmed_lead=True, model_stop_distance=55.0, lead_distance=30.0, lead_path_y_rel=0.2, lead_idx=1, v_ego=10.0,
  )
  mismatched_geometry = associate_model_stop_with_lead(
    confirmed_lead=True, model_stop_distance=29.0, lead_distance=30.0, lead_path_y_rel=2.0, lead_idx=1, v_ego=10.0,
  )

  assert associated == 1
  assert before_lead is None
  assert after_lead is None
  assert mismatched_geometry is None


def test_scc_selector_never_promotes_advisory_only_evidence():
  selector = SccEvidenceSelector()
  advisory = classify_scc_evidence(advisories=SccAdvisoryFlags(speed_limit_cap=True, curve_cap=True))

  selected = advisory
  for _ in range(5):
    selected = selector.update(selected, 0.1)

  assert selector.state == SccEvidenceSelectorState.SCC_ACC
  assert selected.tier == SccEvidenceTier.NONE
  assert not selected.e2e_active


def test_scc_selector_requires_persistent_no_lead_stop():
  selector = SccEvidenceSelector()
  stop = classify_scc_evidence(model_stop=True, model_stop_distance=18.0)

  first = selector.update(stop, 0.05)
  second = selector.update(stop, 0.05)
  third = selector.update(stop, 0.05)

  assert selector.state == SccEvidenceSelectorState.SCC_E2E_ACTIVE
  assert not first.e2e_active
  assert not second.e2e_active
  assert third.e2e_active


def test_scc_selector_one_frame_stop_flicker_does_not_activate():
  selector = SccEvidenceSelector()
  stop = classify_scc_evidence(model_stop=True, model_stop_distance=18.0)

  selected_stop = selector.update(stop, 0.05)
  selected_clear = selector.update(classify_scc_evidence(), 0.05)

  assert selected_stop.reason == "scc_e2e_pending"
  assert not selected_stop.e2e_active
  assert selector.state == SccEvidenceSelectorState.SCC_ACC
  assert not selected_clear.e2e_active


def test_scc_selector_confirmed_associated_stop_stays_acc():
  selector = SccEvidenceSelector()
  associated = classify_scc_evidence(
    confirmed_lead=True,
    model_stop=True,
    model_stop_distance=31.0,
    lead_distance=30.0,
    lead_path_y_rel=0.0,
    lead_idx=0,
    v_ego=12.0,
  )

  selected = associated
  for _ in range(5):
    selected = selector.update(selected, 0.1)

  assert selector.state == SccEvidenceSelectorState.SCC_ACC
  assert not selected.e2e_active


def test_scc_selector_urgent_independent_stop_cuts_through_pending():
  selector = SccEvidenceSelector()
  stop = classify_scc_evidence(model_stop=True, model_stop_distance=18.0)
  urgent = classify_scc_evidence(urgent_stop=True, model_stop_distance=8.0)

  pending = selector.update(stop, 0.05)
  active = selector.update(urgent, 0.0)

  assert not pending.e2e_active
  assert selector.state == SccEvidenceSelectorState.SCC_E2E_ACTIVE
  assert active.tier == SccEvidenceTier.URGENT_STOP
  assert active.e2e_active


def test_scc_selector_recovers_before_full_acc():
  selector = SccEvidenceSelector()
  urgent = classify_scc_evidence(urgent_stop=True, model_stop_distance=8.0)

  active = selector.update(urgent, 0.0)
  recovery = selector.update(classify_scc_evidence(), 0.05)
  recovered = selector.update(classify_scc_evidence(), 0.6)

  assert active.e2e_active
  assert selector.state == SccEvidenceSelectorState.SCC_ACC
  assert recovery.reason == "scc_acc_recovery"
  assert not recovery.e2e_active
  assert recovered.reason == "scc_no_evidence"
  assert not recovered.e2e_active
