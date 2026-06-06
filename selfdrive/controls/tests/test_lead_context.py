from types import SimpleNamespace

import pytest

from openpilot.selfdrive.controls.lib.lead_confidence import LeadConfidenceState
from openpilot.selfdrive.controls.lib.lead_context import (
  LEAD_AUTHORITY_NONE,
  LEAD_AUTHORITY_PHYSICAL,
  LEAD_AUTHORITY_PROGRESS_ALLOWED,
  LEAD_AUTHORITY_SUPPRESS_ONLY,
  LEAD_CONTEXT_SHADOW_NORMAL_TIME,
  LeadContextTracker,
)


NO_LEAD = SimpleNamespace(status=False)


def make_lead(status=True, track_id=1, d_rel=20.0, v_lead=8.0, v_rel=None, y_rel=0.0,
              a_lead=0.0, model_prob=0.9, radar=True):
  if v_rel is None:
    v_rel = 0.0
  return SimpleNamespace(
    status=status,
    radarTrackId=track_id,
    dRel=d_rel,
    vLeadK=v_lead,
    vLead=v_lead,
    vRel=v_rel,
    yRel=y_rel,
    aLeadK=a_lead,
    aLeadTau=0.0,
    modelProb=model_prob,
    radar=radar,
  )


def stable_conf(track_id=1):
  return LeadConfidenceState(
    status=True,
    stable=True,
    speed_trusted=True,
    radar=True,
    age=1.0,
    accel_blend=1.0,
    track_id=track_id,
  )


def new_conf(track_id=1):
  return LeadConfidenceState(
    status=True,
    new_lead=True,
    stable=False,
    speed_trusted=True,
    radar=True,
    guard_timer=0.35,
    track_id=track_id,
  )


def flicker_conf(track_id=1):
  return LeadConfidenceState(
    status=True,
    stable=True,
    speed_trusted=True,
    radar=True,
    age=1.0,
    accel_blend=1.0,
    flicker_guard_timer=0.35,
    track_id=track_id,
  )


def empty_conf():
  return LeadConfidenceState()


def make_model_path(xs, ys):
  return SimpleNamespace(position=SimpleNamespace(x=xs, y=ys))


def test_both_real_leads_absent_no_shadow_has_no_authority():
  context = LeadContextTracker().update((NO_LEAD, NO_LEAD), (empty_conf(), empty_conf()), v_ego=5.0, dt=0.1)

  assert context.physical_idx is None
  assert context.behavior_idx is None
  assert not context.has_physical_lead
  assert not context.lead_progress_allowed
  assert not context.shadow_active


def test_lead1_stable_on_path_can_be_primary_behavior_lead():
  lead0_exiting = make_lead(track_id=10, d_rel=28.0, v_lead=4.0, y_rel=2.0, model_prob=0.4)
  lead1_opening = make_lead(track_id=11, d_rel=18.0, v_lead=2.0, v_rel=1.0, y_rel=0.0, a_lead=0.4)

  context = LeadContextTracker().update(
    (lead0_exiting, lead1_opening), (stable_conf(10), stable_conf(11)), v_ego=1.0, dt=0.1, lead_dominant_idx=1
  )

  assert context.behavior_idx == 1
  assert context.behavior.authority == LEAD_AUTHORITY_PROGRESS_ALLOWED
  assert context.lead_progress_allowed
  assert context.behavior.v_rel == pytest.approx(1.0)


def test_stable_close_closing_lead_is_not_progress_allowed():
  close_closing = make_lead(track_id=10, d_rel=12.0, v_lead=5.0, v_rel=-2.0, y_rel=0.0)

  context = LeadContextTracker().update((close_closing, NO_LEAD), (stable_conf(10), empty_conf()), v_ego=7.0, dt=0.1)

  assert context.physical_idx == 0
  assert context.behavior_idx is None
  assert context.physical.authority == LEAD_AUTHORITY_PHYSICAL
  assert not context.lead_progress_allowed
  assert context.physical.risk_model.required_decel > 0.0
  assert context.physical.risk_model.closing_speed > 0.0
  assert context.physical.risk_model.gap_shortage > 0.0
  assert context.physical.progress_model.reason == "stop_or_closing_threat"


def test_stable_stopped_lead_without_pullaway_is_not_progress_allowed():
  stopped = make_lead(track_id=10, d_rel=10.0, v_lead=0.0, v_rel=0.0, y_rel=0.0)

  context = LeadContextTracker().update((stopped, NO_LEAD), (stable_conf(10), empty_conf()), v_ego=0.2, dt=0.1)

  assert context.physical_idx == 0
  assert context.behavior_idx is None
  assert not context.lead_progress_allowed
  assert context.physical.risk_model.stopped_or_crawling
  assert context.physical.progress_model.reason == "stop_or_closing_threat"


def test_stable_matched_speed_close_lead_is_not_progress_allowed():
  matched_close = make_lead(track_id=10, d_rel=18.0, v_lead=8.0, v_rel=0.0, y_rel=0.0)

  context = LeadContextTracker().update((matched_close, NO_LEAD), (stable_conf(10), empty_conf()), v_ego=8.0, dt=0.1)

  assert context.physical_idx == 0
  assert context.behavior_idx is None
  assert not context.lead_progress_allowed


def test_stable_opening_safe_gap_lead_is_progress_allowed():
  opening = make_lead(track_id=10, d_rel=30.0, v_lead=11.0, v_rel=1.0, y_rel=0.0)

  context = LeadContextTracker().update((opening, NO_LEAD), (stable_conf(10), empty_conf()), v_ego=10.0, dt=0.1)

  assert context.behavior_idx == 0
  assert context.behavior.authority == LEAD_AUTHORITY_PROGRESS_ALLOWED
  assert context.lead_progress_allowed
  assert context.behavior.progress_model.allowed
  assert context.behavior.progress_model.opening_speed == pytest.approx(1.0)
  assert context.behavior.progress_model.gap_excess > 0.0
  assert context.behavior.progress_model.predicted_gap_opening


def test_progress_model_requires_explicit_opening_pullaway_or_gap_evidence():
  matched_gap = make_lead(track_id=10, d_rel=30.0, v_lead=10.0, v_rel=0.0, y_rel=0.0)

  context = LeadContextTracker().update((matched_gap, NO_LEAD), (stable_conf(10), empty_conf()), v_ego=10.0, dt=0.1)

  assert context.behavior_idx == 0
  assert context.lead_progress_allowed
  assert context.behavior.progress_model.gap_excess > 0.0
  assert context.behavior.progress_model.reason == "opening_or_gap_progress"


def test_new_and_flicker_leads_suppress_without_progress_authority():
  opening_new = make_lead(track_id=10, d_rel=30.0, v_lead=11.0, v_rel=1.0, y_rel=0.0)
  opening_flicker = make_lead(track_id=11, d_rel=30.0, v_lead=11.0, v_rel=1.0, y_rel=0.0)

  new_context = LeadContextTracker().update((opening_new, NO_LEAD), (new_conf(10), empty_conf()), v_ego=10.0, dt=0.1)
  flicker_context = LeadContextTracker().update((opening_flicker, NO_LEAD), (flicker_conf(11), empty_conf()), v_ego=10.0, dt=0.1)

  assert new_context.physical.authority == LEAD_AUTHORITY_SUPPRESS_ONLY
  assert not new_context.lead_progress_allowed
  assert not new_context.physical.progress_model.allowed
  assert new_context.physical.progress_model.reason == "insufficient_confidence_stability"
  assert flicker_context.physical.authority == LEAD_AUTHORITY_SUPPRESS_ONLY
  assert not flicker_context.lead_progress_allowed
  assert not flicker_context.physical.progress_model.allowed
  assert flicker_context.physical.progress_model.reason == "insufficient_confidence_stability"


def test_primary_lead_debug_exposes_risk_and_progress_metrics():
  opening = make_lead(track_id=10, d_rel=30.0, v_lead=11.0, v_rel=1.0, y_rel=0.0)

  context = LeadContextTracker().update((opening, NO_LEAD), (stable_conf(10), empty_conf()), v_ego=10.0, dt=0.1)
  debug = context.debug_dict()

  assert debug["primary_lead_required_decel"] == pytest.approx(context.behavior.risk_model.required_decel)
  assert debug["primary_lead_gap_shortage"] == pytest.approx(context.behavior.risk_model.gap_shortage)
  assert debug["primary_lead_closing_speed"] == pytest.approx(context.behavior.risk_model.closing_speed)
  assert debug["primary_lead_progress_reason"] == context.behavior.progress_model.reason
  assert debug["primary_lead_predicted_gap_opening"] is True


def test_lead1_primary_physical_blocks_progress_without_behavior_authority():
  lead1_cut_in = make_lead(track_id=11, d_rel=10.0, v_lead=2.0, v_rel=-4.0, y_rel=0.0)

  context = LeadContextTracker().update(
    (NO_LEAD, lead1_cut_in), (empty_conf(), new_conf(11)), v_ego=6.0, dt=0.1, lead_dominant_idx=1
  )

  assert context.physical_idx == 1
  assert context.behavior_idx is None
  assert not context.alternate_threat_active
  assert context.has_physical_lead
  assert not context.lead_progress_allowed
  assert context.physical.authority == LEAD_AUTHORITY_SUPPRESS_ONLY


def test_behavior_lead_opening_is_blocked_by_alternate_close_threat():
  lead0_opening = make_lead(track_id=10, d_rel=18.0, v_lead=2.0, v_rel=1.0, y_rel=0.0, a_lead=0.4)
  lead1_threat = make_lead(track_id=11, d_rel=8.0, v_lead=0.0, v_rel=-1.0, y_rel=0.0)

  context = LeadContextTracker().update(
    (lead0_opening, lead1_threat), (stable_conf(10), new_conf(11)), v_ego=1.0, dt=0.1, lead_dominant_idx=1
  )

  assert context.behavior_idx == 0
  assert context.physical_idx == 1
  assert context.alternate_threat_active
  assert not context.lead_progress_allowed
  assert context.lead_release_blocked_reason == "alternate_lead_threat"


def test_secondary_pullaway_cannot_authorize_progress_with_stopped_uncertain_primary_physical():
  stopped_uncertain_primary = make_lead(track_id=10, d_rel=6.0, v_lead=0.0, v_rel=-0.2, y_rel=0.0)
  opening_secondary = make_lead(track_id=11, d_rel=18.0, v_lead=1.0, v_rel=0.8, y_rel=0.0, a_lead=0.4)

  context = LeadContextTracker().update(
    (stopped_uncertain_primary, opening_secondary), (new_conf(10), stable_conf(11)), v_ego=0.2, dt=0.1, lead_dominant_idx=1
  )

  assert context.physical_idx == 0
  assert context.behavior_idx == 1
  assert context.physical.authority == LEAD_AUTHORITY_SUPPRESS_ONLY
  assert not context.lead_progress_allowed
  assert context.lead_release_blocked_reason == "alternate_lead_threat"


def test_shadow_blocks_no_lead_without_authorizing_progress():
  tracker = LeadContextTracker()
  close_lead = make_lead(track_id=4, d_rel=8.0, v_lead=0.0, v_rel=-0.5, y_rel=0.0)
  tracker.update((close_lead, NO_LEAD), (stable_conf(4), empty_conf()), v_ego=0.5, dt=0.1)

  context = tracker.update((NO_LEAD, NO_LEAD), (empty_conf(), empty_conf()), v_ego=0.5, dt=0.1)

  assert context.shadow_active
  assert context.has_physical_lead
  assert context.physical.shadow
  assert context.behavior_idx is None
  assert not context.lead_progress_allowed
  assert context.physical.authority == LEAD_AUTHORITY_SUPPRESS_ONLY


def test_far_opening_shadow_releases_quickly():
  tracker = LeadContextTracker()
  far_lead = make_lead(track_id=4, d_rel=60.0, v_lead=20.0, v_rel=5.0, y_rel=0.0)
  tracker.update((far_lead, NO_LEAD), (stable_conf(4), empty_conf()), v_ego=15.0, dt=0.1)
  initial_shadow = tracker.update((NO_LEAD, NO_LEAD), (empty_conf(), empty_conf()), v_ego=15.0, dt=0.1)

  released = tracker.update(
    (NO_LEAD, NO_LEAD), (empty_conf(), empty_conf()), v_ego=15.0, dt=LEAD_CONTEXT_SHADOW_NORMAL_TIME + 0.1
  )

  assert initial_shadow.shadow_active
  assert not released.shadow_active
  assert released.physical_idx is None


def test_shadow_preserves_last_raw_lateral_offset_for_path_relevance():
  tracker = LeadContextTracker()
  model = make_model_path([0.0, 10.0, 20.0], [0.0, 1.0, 1.0])
  on_path_lead = make_lead(track_id=4, d_rel=10.0, v_lead=0.0, v_rel=-0.5, y_rel=1.0)
  tracker.update((on_path_lead, NO_LEAD), (stable_conf(4), empty_conf()), v_ego=0.5, dt=0.1, model_msg=model)

  context = tracker.update((NO_LEAD, NO_LEAD), (empty_conf(), empty_conf()), v_ego=0.5, dt=0.1, model_msg=model)

  assert context.physical.shadow
  assert abs(context.physical.path_y_rel) < 0.01


def test_lateral_false_positive_releases_after_persistent_low_risk_path_exit():
  tracker = LeadContextTracker()
  exiting_lead = make_lead(track_id=5, d_rel=35.0, v_lead=10.0, v_rel=1.0, y_rel=2.0, model_prob=0.55, radar=False)
  weak_conf = LeadConfidenceState(status=True, speed_trusted=False, radar=False, track_id=5)

  pending = tracker.update((exiting_lead, NO_LEAD), (weak_conf, empty_conf()), v_ego=9.0, dt=0.1)
  tracker.update((exiting_lead, NO_LEAD), (weak_conf, empty_conf()), v_ego=9.0, dt=0.1)
  released = tracker.update((exiting_lead, NO_LEAD), (weak_conf, empty_conf()), v_ego=9.0, dt=0.1)

  assert pending.physical_idx == 0
  assert pending.physical.authority == LEAD_AUTHORITY_SUPPRESS_ONLY
  assert released.physical_idx is None
  assert released.states[0].authority == LEAD_AUTHORITY_NONE
  assert released.states[0].reason == "lateral_exit_confirmed"


def test_lateral_false_positive_release_persists_when_exited_lead_slows():
  tracker = LeadContextTracker()
  exiting_lead = make_lead(track_id=5, d_rel=35.0, v_lead=10.0, v_rel=1.0, y_rel=2.0, model_prob=0.55, radar=False)
  weak_conf = LeadConfidenceState(status=True, speed_trusted=False, radar=False, track_id=5)

  tracker.update((exiting_lead, NO_LEAD), (weak_conf, empty_conf()), v_ego=9.0, dt=0.1)
  tracker.update((exiting_lead, NO_LEAD), (weak_conf, empty_conf()), v_ego=9.0, dt=0.1)
  released = tracker.update((exiting_lead, NO_LEAD), (weak_conf, empty_conf()), v_ego=9.0, dt=0.1)
  slowing_exited = make_lead(track_id=5, d_rel=35.0, v_lead=5.0, v_rel=-4.0, y_rel=2.0, model_prob=0.55, radar=False)

  still_released = tracker.update((slowing_exited, NO_LEAD), (weak_conf, empty_conf()), v_ego=9.0, dt=0.1)

  assert released.states[0].authority == LEAD_AUTHORITY_NONE
  assert still_released.states[0].authority == LEAD_AUTHORITY_NONE
  assert still_released.physical_idx is None


def test_no_status_fake_mpc_fallback_lead_never_becomes_primary():
  fake_fallback = make_lead(status=False, d_rel=50.0, v_lead=30.0, v_rel=20.0, y_rel=0.0)

  context = LeadContextTracker().update((fake_fallback, NO_LEAD), (empty_conf(), empty_conf()), v_ego=10.0, dt=0.1)

  assert context.physical_idx is None
  assert context.behavior_idx is None
  assert not context.has_physical_lead
  assert not context.lead_progress_allowed
