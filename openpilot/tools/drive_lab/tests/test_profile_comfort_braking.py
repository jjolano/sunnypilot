"""Regression tests for the comfort-braking opportunity profiler."""

from __future__ import annotations

import json
from types import SimpleNamespace

from openpilot.tools.drive_lab.profile_comfort_braking import (
  ComfortBrakingParams,
  analyze_route,
  render_report,
)


class FakeMsg(SimpleNamespace):
  def which(self):
    return self.kind


def msg(kind, t_s, **payload):
  return FakeMsg(kind=kind, logMonoTime=int(t_s * 1e9), **{kind: SimpleNamespace(**payload)})


def stream(rows):
  """rows: (vEgo, aEgo, dRel, vRel, aTarget, longActive, leadStatus[, modelProb[, yRel]])."""
  msgs = []
  for i, row in enumerate(rows):
    v_ego, a_ego, d_rel, v_rel, a_target, long_active, lead_status = row[:7]
    model_prob = row[7] if len(row) > 7 else 0.9
    y_rel = row[8] if len(row) > 8 else None
    t = i * 0.05
    msgs.append(msg("carState", t, vEgo=v_ego, aEgo=a_ego))
    msgs.append(msg("carControl", t + 0.0005, longActive=long_active))
    msgs.append(msg("longitudinalPlanSP", t + 0.001, aTarget=a_target))
    msgs.append(
      msg(
        "radarState",
        t + 0.0015,
        leadOne=SimpleNamespace(status=lead_status, dRel=d_rel, vRel=v_rel, modelProb=model_prob, yRel=y_rel),
      )
    )
  return msgs


def test_no_episodes_when_not_long_active():
  rows = [(15.0, 0.0, 40.0, -2.5, 0.0, False, True)] * 20
  report = analyze_route(stream(rows), source="manual")
  assert report.episode_count == 0
  assert report.closing_samples == 0
  assert any("no engaged" in note for note in report.notes)


def test_detects_one_closing_episode():
  rows = [(15.0, -0.2, 40.0, -2.5, -0.3, True, True)] * 15
  report = analyze_route(stream(rows), source="synthetic")
  assert report.episode_count == 1
  assert report.opportunity_count == 0

  ep = report.episodes[0]
  assert ep.start_v_ego == 15.0
  assert ep.start_v_rel == -2.5
  assert ep.peak_closing_v_rel == -2.5
  assert ep.worst_a_ego == -0.2
  assert ep.worst_a_target == -0.3
  assert ep.time_to_mild_plan_s is None
  assert ep.duration_s >= 0.5


def test_opportunity_flagged_when_mild_plan_delayed_and_later_hard_braking():
  """High closing persists, mild planned decel arrives late, and hard braking follows."""
  rows: list[tuple[float, float, float, float, float, bool, bool]] = []
  n = 25
  for i in range(n):
    d_rel = 60.0 - 1.5 * i
    v_rel = -3.5
    a_ego = 0.0 if i < 12 else (-0.5 if i < 15 else -2.5)
    a_target = 0.0 if i < 16 else (-0.6 if i < 20 else -2.0)
    rows.append((15.0, a_ego, d_rel, v_rel, a_target, True, True))

  report = analyze_route(stream(rows), source="opportunity")
  assert report.episode_count == 1
  assert report.opportunity_count == 1
  assert report.candidate_count == 1

  ep = report.episodes[0]
  assert ep.opportunity is True
  assert ep.candidate_quality == "kinematic"
  assert "path_confidence_unknown" in ep.candidate_block_reasons
  assert ep.start_v_rel == -3.5
  assert ep.time_to_mild_plan_s is not None
  assert ep.time_to_mild_plan_s > 0.75
  assert ep.candidate_lead_time_to_mild_plan_s is not None
  assert ep.candidate_lead_time_to_mild_plan_s >= 0.5
  assert ep.time_to_firm_plan_s is not None
  assert ep.worst_a_ego == -2.5


def test_path_confident_candidate_when_yrel_is_near_path():
  rows: list[tuple[float, float, float, float, float, bool, bool, float, float]] = []
  for i in range(25):
    d_rel = 60.0 - 1.5 * i
    a_ego = 0.0 if i < 12 else (-0.5 if i < 15 else -2.5)
    a_target = 0.0 if i < 16 else (-0.6 if i < 20 else -2.0)
    rows.append((15.0, a_ego, d_rel, -3.5, a_target, True, True, 0.9, 0.5))

  report = analyze_route(stream(rows), source="path-candidate")
  assert report.candidate_count == 1
  assert report.path_confident_candidate_count == 1
  assert report.episodes[0].candidate_quality == "path_confident"


def test_candidate_requires_confidence_and_sustain():
  rows: list[tuple[float, float, float, float, float, bool, bool, float]] = []
  for i in range(25):
    a_ego = 0.0 if i < 12 else (-0.5 if i < 15 else -2.5)
    a_target = 0.0 if i < 16 else (-0.6 if i < 20 else -2.0)
    model_prob = 0.5 if i < 16 else 0.9
    rows.append((15.0, a_ego, 60.0 - 1.5 * i, -3.5, a_target, True, True, model_prob))

  report = analyze_route(stream(rows), source="low-confidence")
  assert report.opportunity_count == 1
  assert report.candidate_count == 0
  assert report.episodes[0].candidate_quality == "none"


def test_tiny_episodes_are_ignored():
  rows = [(15.0, -0.5, 40.0, -2.5, -0.8, True, True)] * 8  # 0.35 s, below min
  report = analyze_route(stream(rows), source="tiny")
  assert report.episode_count == 0
  assert report.closing_samples == 8


def test_json_serializable_and_render_does_not_crash():
  rows = [(15.0, -0.2, 40.0, -2.5, -0.3, True, True)] * 12
  report = analyze_route(stream(rows), source="json-test")
  data = report.to_dict()
  assert isinstance(data, dict)
  json.dumps(data)
  text = render_report(report)
  assert isinstance(text, str)
  assert "offline shadow analysis" in text
  assert "json-test" in text


def test_custom_params_change_thresholds():
  # Gentle closing below default close_v_rel; with a looser threshold it should count.
  rows = [(15.0, -0.1, 40.0, -1.5, -0.2, True, True)] * 15
  params = ComfortBrakingParams(close_v_rel=-1.0)
  report = analyze_route(stream(rows), source="custom-threshold", params=params)
  assert report.episode_count == 1
  assert report.episodes[0].start_v_rel == -1.5
