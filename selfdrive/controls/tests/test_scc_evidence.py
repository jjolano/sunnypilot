from openpilot.selfdrive.controls.lib.scc_evidence import (
  SccAdvisoryFlags,
  SccEvidenceTier,
  associate_model_stop_with_lead,
  classify_scc_evidence,
)


def test_no_evidence_classifies_none():
  result = classify_scc_evidence()

  assert result.tier == SccEvidenceTier.NONE
  assert result.reason == "scc_no_evidence"
  assert not result.e2e_active


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
