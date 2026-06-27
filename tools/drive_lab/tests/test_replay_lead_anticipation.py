import json

from openpilot.sunnypilot.custom.longitudinal.lead_anticipation import LeadAnticipation
from openpilot.tools.drive_lab.replay_lead_anticipation import LeadReplayRow, _AlwaysOn, render, render_reports, summarize_rows


def test_no_following_frames_is_explicit_and_not_safe():
  report = summarize_rows([], source="empty")
  assert report["following_frames"] == 0
  assert "no lead-following frames" in report["note"]
  assert report["benefit_detected"] is False
  assert report["safety_pass"] is False
  assert report["far_proposal_safety_pass"] is False


def test_replay_forces_apply_mode():
  la = LeadAnticipation(_AlwaysOn())
  assert la.mode == "apply"
  assert _AlwaysOn().get_bool("CustomLongitudinalEnabled") is True


def test_summarization_counts_softened_and_peaks():
  report = summarize_rows([
    LeadReplayRow(a_raw=-0.6, a_shaped=-0.25, a_lead=0.0, v_rel=0.0, d_rel=20.0),
    LeadReplayRow(a_raw=-1.2, a_shaped=-0.8, a_lead=-0.2, v_rel=-0.8, d_rel=18.0),
  ], source="rows")
  assert report["softened_frames"] == 2
  assert report["decel_peak_raw"] == -1.2
  assert report["decel_peak_shaped"] == -0.8
  assert report["benefit_detected"] is True
  assert report["safety_pass"] is True


def test_risky_softenings_are_counted_on_fast_closing_leads():
  report = summarize_rows([
    LeadReplayRow(a_raw=-0.9, a_shaped=-0.5, a_lead=0.0, v_rel=-1.6, d_rel=15.0),
    LeadReplayRow(a_raw=-0.2, a_shaped=0.05, a_lead=0.0, v_rel=-2.0, d_rel=14.0),
  ], source="risky")
  assert report["risky_softenings"] == 1
  assert report["safety_pass"] is False


def test_render_and_json_fields_are_stable():
  report = summarize_rows([
    LeadReplayRow(a_raw=-0.7, a_shaped=-0.3, a_lead=-0.6, v_rel=-0.4, d_rel=45.0,
                  far_proposal=-0.25, far_eligible=True),
  ], source="stable")
  text = render(report)
  assert "§3 lead-anticipation A/B: stable" in text
  assert "softened braking on 1 frames" in text
  assert "far-lead shadow proposal: 1 frames" in text
  assert report["far_proposal_frames"] == 1
  assert report["far_proposal_safety_pass"] is True
  assert report["benefit_detected"] is True
  assert report["safety_pass"] is True


def test_far_proposal_risky_frames_are_reported_separately():
  report = summarize_rows([
    LeadReplayRow(a_raw=-0.4, a_shaped=-0.4, a_lead=-0.8, v_rel=-1.6, d_rel=55.0,
                  far_proposal=-0.45, far_eligible=True),
  ], source="far-risky")
  assert report["risky_softenings"] == 0
  assert report["safety_pass"] is True
  assert report["far_proposal_risky_frames"] == 1
  assert report["far_proposal_safety_pass"] is False


def test_render_reports_json_is_valid_for_multiple_routes():
  reports = [summarize_rows([], source="a"), summarize_rows([], source="b")]
  payload = json.loads(render_reports(reports, json_output=True))
  assert [r["source"] for r in payload] == ["a", "b"]
