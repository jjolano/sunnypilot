#!/usr/bin/env python3
"""Tests for the general per-decision signal scorer.

The signal scorer's job is to rank which planner/decision signals best match
human decisions, and to do so consistently with the existing
``profile_leadless_stops`` recall/timely harness. These tests pin down:

* signal registration is stable
* the leadless-stop signal evaluators in the general scorer agree with the
  authoritative ``profile_leadless_stops.signal_active`` evaluator (the proof
  of consistency required by the plan)
* labelers (stop/go, opposite-intent, lead transition) emit well-formed
  episodes from synthetic sample streams
* the scorecard produces stable recall/false-alarm metrics on trivial inputs
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from openpilot.tools.drive_lab import signal_scorer
from openpilot.tools.drive_lab import profile_leadless_stops as leadless
from openpilot.tools.drive_lab.signal_scorer import (
  DecisionEpisode,
  SignalSample,
  label_lead_transition_episodes,
  label_opposite_intent_episodes,
  label_stop_go_episodes,
  score_signal_for_category,
)


def make_sample(
  t: float = 0.0,
  v_ego: float = 10.0,
  a_ego: float = 0.0,
  *,
  route: str = "r1--0",
  route_id: str = "r1",
  segment: int | None = 0,
  gas: bool = False,
  brake: bool = False,
  standstill: bool = False,
  selfdrive_active: bool = False,
  long_active: bool = False,
  plan_a_target: float | None = 0.0,
  plan_should_stop: bool = False,
  plan_source: str = "cruise",
  model_should_stop: bool = False,
  model_desired_accel: float | None = 0.0,
  model_stop_distance: float | None = None,
  model_endpoint_x: float | None = None,
  model_endpoint_v: float | None = None,
  lead_status: bool = False,
  lead_d_rel: float | None = None,
  lead_v_rel: float | None = None,
  sp_a_target: float | None = None,
  sp_source: str = "cruise",
  sp_stack: str = "sunnypilotCurrent",
  fcw: bool = False,
) -> SignalSample:
  return SignalSample(
    route=route,
    route_id=route_id,
    segment=segment,
    t=t,
    v_ego=v_ego,
    a_ego=a_ego,
    gas_pressed=gas,
    brake_pressed=brake,
    standstill=standstill,
    selfdrive_active=selfdrive_active,
    long_active=long_active,
    plan_a_target=plan_a_target,
    plan_should_stop=plan_should_stop,
    plan_source=plan_source,
    model_should_stop=model_should_stop,
    model_desired_accel=model_desired_accel,
    model_stop_distance=model_stop_distance,
    model_endpoint_x=model_endpoint_x,
    model_endpoint_v=model_endpoint_v,
    lead_status=lead_status,
    lead_d_rel=lead_d_rel,
    lead_v_rel=lead_v_rel,
    sp_a_target=sp_a_target,
    sp_source=sp_source,
    sp_stack=sp_stack,
    fcw=fcw,
  )


def _to_leadless_sample(sig: SignalSample) -> leadless.LeadlessStopSample:
  return leadless.LeadlessStopSample(
    route=sig.route,
    route_id=sig.route_id,
    segment=sig.segment,
    t=sig.t,
    v_ego=sig.v_ego,
    a_ego=sig.a_ego,
    gas_pressed=sig.gas_pressed,
    brake_pressed=sig.brake_pressed,
    standstill=sig.standstill,
    selfdrive_active=sig.selfdrive_active,
    long_active=sig.long_active,
    long_control_state="pid",
    lead_status=sig.lead_status,
    lead_d_rel=sig.lead_d_rel,
    lead_v_rel=sig.lead_v_rel,
    model_should_stop=sig.model_should_stop,
    model_desired_accel=sig.model_desired_accel,
    model_stop_distance=sig.model_stop_distance,
    model_endpoint_x=sig.model_endpoint_x,
    model_endpoint_v=sig.model_endpoint_v,
    scc_early_model_stop=False,
    plan_should_stop=sig.plan_should_stop,
    plan_source=sig.plan_source,
    plan_a_target=sig.plan_a_target,
    sp_source=sig.sp_source,
    sp_a_target=sig.sp_a_target,
    sp_stack=sig.sp_stack,
    traffic_control_valid=False,
    traffic_control_type="",
    traffic_control_distance=None,
    traffic_control_ahead_valid=False,
    traffic_control_ahead_type="",
    traffic_control_ahead_distance=None,
  )


SHARED_SIGNAL_NAMES = (
  "model_should_stop",
  "model_path_or_should_stop",
  "model_accel_le_-0.5",
  "model_accel_le_-1.0",
  "planner_should_stop",
  "planner_e2e_or_should_stop",
)


def test_signal_registry_is_stable() -> None:
  names = set(signal_scorer.SIGNAL_REGISTRY.keys())
  for name in SHARED_SIGNAL_NAMES:
    assert name in names, f"{name} missing from general scorer registry"


@pytest.mark.parametrize("signal_name", SHARED_SIGNAL_NAMES)
def test_scorer_agrees_with_leadless_signal_active(signal_name: str) -> None:
  cases = [
    make_sample(model_should_stop=True, model_desired_accel=0.0, model_stop_distance=None),
    make_sample(model_should_stop=False, model_desired_accel=-0.3, model_stop_distance=200.0),
    make_sample(model_should_stop=True, model_desired_accel=-1.5, model_stop_distance=30.0),
    make_sample(plan_should_stop=True, plan_source="e2e"),
    make_sample(plan_should_stop=False, plan_source="cruise"),
    make_sample(model_should_stop=False, model_desired_accel=None, plan_should_stop=True, plan_source="e2e"),
    make_sample(plan_source="lead0", plan_should_stop=False),
  ]
  for sig in cases:
    expected = leadless.signal_active(_to_leadless_sample(sig), signal_name)
    actual = signal_scorer.SIGNAL_REGISTRY[signal_name](sig)
    assert actual == expected, f"mismatch for {signal_name} on {sig!r}"


def test_label_stop_go_emits_episode_for_decel_run() -> None:
  samples = []
  for i, (v, a) in enumerate([
    (12.0, 0.0),
    (11.5, -0.4),
    (10.5, -0.8),
    (9.0, -1.2),
    (7.0, -1.5),
    (5.0, -0.6),
  ]):
    samples.append(make_sample(t=i * 0.1, v_ego=v, a_ego=a, lead_status=False))
  episodes = label_stop_go_episodes(samples, min_duration_s=0.4)
  assert len(episodes) == 1
  ep = episodes[0]
  assert ep.category == "stop_go"
  assert ep.decision_time_s == 0.4
  assert ep.extra["peak_decel"] == pytest.approx(-1.5)


def test_label_stop_go_skips_when_lead_present() -> None:
  samples = []
  for i, (v, a) in enumerate([
    (12.0, 0.0),
    (11.5, -0.4),
    (10.5, -0.8),
    (9.0, -1.2),
  ]):
    samples.append(make_sample(t=i * 0.1, v_ego=v, a_ego=a, lead_status=True))
  assert label_stop_go_episodes(samples) == []


def test_label_opposite_intent_emits_episode() -> None:
  samples = []
  for i, (a_ego, brake, plan_a) in enumerate([
    (0.0, False, 0.0),
    (0.0, False, 0.0),
    (-0.5, True, 0.6),
    (-0.5, True, 0.6),
    (-0.5, True, 0.6),
  ]):
    samples.append(make_sample(t=i * 0.1, v_ego=10.0, a_ego=a_ego, brake=brake, gas=False, plan_a_target=plan_a))
  episodes = label_opposite_intent_episodes(samples, min_duration_s=0.2)
  assert len(episodes) == 1
  ep = episodes[0]
  assert ep.category == "opposite_intent"
  assert "plan_accel" in ep.kind and "driver_brake" in ep.kind


def test_label_lead_transition_emits_episode_on_acquire() -> None:
  samples = []
  for i, lead in enumerate([False, False, False, True, True, True, True]):
    samples.append(make_sample(t=i * 0.1, v_ego=10.0, lead_status=lead, lead_d_rel=40.0 if lead else None))
  episodes = label_lead_transition_episodes(samples, min_window_s=0.15)
  assert len(episodes) == 1
  assert episodes[0].kind == "lead_acquired"
  assert episodes[0].extra["d_rel"] == 40.0


def test_score_signal_recall_and_miss() -> None:
  samples = [
    make_sample(t=0.0, v_ego=10.0, a_ego=0.0, model_should_stop=True),
    make_sample(t=0.1, v_ego=10.0, a_ego=0.0, model_should_stop=True),
    make_sample(t=0.2, v_ego=10.0, a_ego=0.0, model_should_stop=True),
    make_sample(t=10.0, v_ego=10.0, a_ego=0.0, model_should_stop=False),
    make_sample(t=10.1, v_ego=10.0, a_ego=0.0, model_should_stop=False),
    make_sample(t=10.2, v_ego=10.0, a_ego=0.0, model_should_stop=False),
  ]
  hit_ep = DecisionEpisode(
    category="stop_go", route="r", route_id="r", segment=0,
    decision_time_s=0.1, start_time_s=0.0, end_time_s=0.2, duration_s=0.2,
    kind="leadless_slowdown", v_ego=10.0, a_ego=0.0, active_ratio=0.0, long_active_ratio=0.0,
  )
  miss_ep = DecisionEpisode(
    category="stop_go", route="r", route_id="r", segment=0,
    decision_time_s=10.1, start_time_s=10.0, end_time_s=10.2, duration_s=0.2,
    kind="leadless_slowdown", v_ego=10.0, a_ego=0.0, active_ratio=0.0, long_active_ratio=0.0,
  )
  metric = score_signal_for_category(samples, [hit_ep, miss_ep], "model_should_stop")
  assert metric.episodes == 2
  assert metric.hit_count == 1
  assert metric.missed_count == 1
  assert metric.recall == pytest.approx(0.5)
  assert metric.timely_hit_count == 1
  assert metric.median_lead_time_s is not None
  assert metric.fire_count == 3


def test_score_signal_flicker_counted_per_transition() -> None:
  samples = [
    make_sample(t=0.0, model_should_stop=False),
    make_sample(t=0.1, model_should_stop=True),
    make_sample(t=0.2, model_should_stop=False),
    make_sample(t=0.3, model_should_stop=True),
    make_sample(t=0.4, model_should_stop=False),
  ]
  ep = DecisionEpisode(
    category="stop_go", route="r", route_id="r", segment=0,
    decision_time_s=0.2, start_time_s=0.0, end_time_s=0.4, duration_s=0.4,
    kind="leadless_slowdown", v_ego=10.0, a_ego=0.0, active_ratio=0.0, long_active_ratio=0.0,
  )
  metric = score_signal_for_category(samples, [ep], "model_should_stop")
  assert metric.hit_count == 1
  assert metric.flicker_count >= 3


def test_unknown_signal_raises() -> None:
  with pytest.raises(KeyError):
    score_signal_for_category([], [], "not_a_signal")


def test_plan_e2e_source_signal() -> None:
  assert signal_scorer.SIGNAL_REGISTRY["plan_e2e_source"](make_sample(plan_source="e2e"))
  assert signal_scorer.SIGNAL_REGISTRY["plan_e2e_source"](make_sample(plan_source="model"))
  assert not signal_scorer.SIGNAL_REGISTRY["plan_e2e_source"](make_sample(plan_source="cruise"))
  assert not signal_scorer.SIGNAL_REGISTRY["plan_e2e_source"](make_sample(plan_source="lead0"))


def test_plan_e2e_or_should_stop_signal() -> None:
  assert signal_scorer.SIGNAL_REGISTRY["plan_e2e_or_should_stop"](make_sample(plan_source="cruise", plan_should_stop=True))
  assert signal_scorer.SIGNAL_REGISTRY["plan_e2e_or_should_stop"](make_sample(plan_source="e2e", plan_should_stop=False))
  assert not signal_scorer.SIGNAL_REGISTRY["plan_e2e_or_should_stop"](make_sample(plan_source="cruise", plan_should_stop=False))


def test_sp_customv2_active_signal() -> None:
  assert signal_scorer.SIGNAL_REGISTRY["sp_customv2_active"](make_sample(sp_stack="customV2"))
  assert signal_scorer.SIGNAL_REGISTRY["sp_customv2_active"](make_sample(sp_stack="custom-2.0"))
  assert not signal_scorer.SIGNAL_REGISTRY["sp_customv2_active"](make_sample(sp_stack="sunnypilotCurrent"))


def test_plan_braking_signals() -> None:
  assert signal_scorer.SIGNAL_REGISTRY["plan_braking"](make_sample(plan_a_target=-0.5))
  assert signal_scorer.SIGNAL_REGISTRY["plan_braking"](make_sample(plan_a_target=-1.0))
  assert not signal_scorer.SIGNAL_REGISTRY["plan_braking"](make_sample(plan_a_target=0.0))
  assert not signal_scorer.SIGNAL_REGISTRY["plan_braking"](make_sample(plan_a_target=None))
  assert signal_scorer.SIGNAL_REGISTRY["plan_strong_brake"](make_sample(plan_a_target=-1.5))
  assert not signal_scorer.SIGNAL_REGISTRY["plan_strong_brake"](make_sample(plan_a_target=-0.5))


def test_non_episode_windows_skips_overlap() -> None:
  from openpilot.tools.drive_lab.signal_scorer import _non_episode_windows
  samples = [make_sample(t=i * 1.0) for i in range(60)]
  episodes = [DecisionEpisode(
    category="x", route="r", route_id="r", segment=0,
    decision_time_s=20.0, start_time_s=15.0, end_time_s=25.0, duration_s=10.0,
    kind="k", v_ego=10.0, a_ego=0.0, active_ratio=0.0, long_active_ratio=0.0,
  )]
  windows = _non_episode_windows(samples, episodes, window_s=10.0, max_windows=10)
  for w_start, w_end in windows:
    assert not (w_start < 25.0 and 15.0 < w_end), f"window {w_start}-{w_end} overlaps episode 15-25"


def test_precision_is_one_when_no_false_alarms() -> None:
  samples = [make_sample(t=i * 0.1, v_ego=10.0, model_should_stop=(i % 2 == 0)) for i in range(50)]
  ep = DecisionEpisode(
    category="stop_go", route="r", route_id="r", segment=0,
    decision_time_s=2.0, start_time_s=1.5, end_time_s=2.5, duration_s=1.0,
    kind="leadless_slowdown", v_ego=10.0, a_ego=0.0, active_ratio=0.0, long_active_ratio=0.0,
  )
  metric = score_signal_for_category(samples, [ep], "model_should_stop", lookback_s=2.0, timely_grace_s=1.5)
  assert metric.true_positives > 0
  assert metric.precision > 0.5


def test_precision_drops_with_false_alarms() -> None:
  samples = [make_sample(t=i * 0.1, v_ego=10.0, model_should_stop=(i % 2 == 0)) for i in range(80)]
  ep = DecisionEpisode(
    category="stop_go", route="r", route_id="r", segment=0,
    decision_time_s=2.0, start_time_s=1.5, end_time_s=2.5, duration_s=1.0,
    kind="leadless_slowdown", v_ego=10.0, a_ego=0.0, active_ratio=0.0, long_active_ratio=0.0,
  )
  new_samples = []
  for s in samples:
    if 5.0 <= s.t <= 7.0:
      new_samples.append(make_sample(t=s.t, v_ego=s.v_ego, a_ego=s.a_ego, model_should_stop=True))
    else:
      new_samples.append(s)
  metric_overfire = score_signal_for_category(new_samples, [ep], "model_should_stop", lookback_s=2.0, timely_grace_s=1.5)
  assert metric_overfire.false_positives > 0
  assert metric_overfire.precision < 1.0


def test_render_summary_groups_zero_recall() -> None:
  from openpilot.tools.drive_lab.signal_scorer import (
    CategoryScorecard, ScorerSummary, SignalMetric, render_summary,
  )
  zero = SignalMetric(
    category="t", signal="zero", episodes=2, hit_count=0, timely_hit_count=0,
    missed_count=2, fire_count=0, recall=0.0, recall_timely=0.0,
    precision=0.0, true_positives=0, false_positives=0, false_alarm_count=0,
    false_alarm_windows=0, median_lead_time_s=None, p10_lead_time_s=None,
    flicker_count=0, onset_lag_s=None,
  )
  hit = SignalMetric(
    category="t", signal="hit", episodes=2, hit_count=2, timely_hit_count=2,
    missed_count=0, fire_count=3, recall=1.0, recall_timely=1.0,
    precision=1.0, true_positives=3, false_positives=0, false_alarm_count=0,
    false_alarm_windows=5, median_lead_time_s=1.0, p10_lead_time_s=0.5,
    flicker_count=1, onset_lag_s=2.0,
  )
  card = CategoryScorecard(category="t", episode_count=2, signals={"zero": zero, "hit": hit})
  summary = ScorerSummary(
    route_count=1, sample_count=100, episode_count=2, category_count=1,
    signal_count=2, categories=[card],
  )
  rendered = render_summary(summary)
  assert "hit" in rendered
  assert "zero-recall signals" in rendered
  assert "zero" in rendered.split("(zero-recall signals:")[1]
