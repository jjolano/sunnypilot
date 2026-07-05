from __future__ import annotations

import json
from types import SimpleNamespace

from openpilot.tools.drive_lab.profile_longitudinal_attribution import analyze_route, render_report


class FakeMsg(SimpleNamespace):
  def which(self):
    return self.kind


def msg(kind, t_s, **payload):
  return FakeMsg(kind=kind, logMonoTime=int(t_s * 1e9), **{kind: SimpleNamespace(**payload)})


def test_counts_and_render_json_basic():
  lead = SimpleNamespace(status=True, dRel=30.0, vRel=-1.0, vLeadK=18.0, vLead=18.0,
                         aLeadK=-0.8, radarTrackId=3, radar=True, modelProb=0.9,
                         yRel=0.0, aLeadTau=1.5)
  msgs = [
    msg("selfdriveState", 0.0, enabled=True),
    msg("carControl", 0.05, longActive=True),
    msg("radarState", 0.06, leadOne=lead),
    msg("longitudinalPlan", 0.07, longitudinalPlanSource="lead0", aTarget=-0.7),
    msg("longitudinalPlanSP", 0.08, longitudinalPlanSource="custom", aTarget=-0.5,
        customLongitudinal=SimpleNamespace(enabled=True, active=True, selectedIntent="follow", reason="lead_close", shouldStop=False)),
    msg("carState", 0.09, vEgo=20.0, aEgo=-0.6),
  ]
  report = analyze_route(msgs, source="synthetic")
  assert report["plan_source_counts"] == {"lead0": 1}
  assert report["sp_plan_source_counts"] == {"custom": 1}
  assert report["custom_intent_counts"] == {"follow": 1}
  assert report["custom_reason_counts"] == {"lead_close": 1}
  assert report["strong_decel_episodes"]
  worst = report["strong_decel_episodes"][0]["worst_sample"]
  assert worst["longitudinalPlan"]["source"] == "lead0"
  assert worst["longitudinalPlanSP"]["customLongitudinal"]["selectedIntent"] == "follow"
  json.loads(json.dumps(report))
  assert "Longitudinal attribution: synthetic" in render_report(report)


def test_no_data_shape_is_stable():
  report = analyze_route([msg("carState", 0.0, vEgo=0.0, aEgo=0.0)], source="empty")
  assert report["strong_decel_episodes"] == []
  json.loads(json.dumps(report))
