from types import SimpleNamespace

import pytest

from openpilot.selfdrive.controls.lib.lead_confidence import LeadConfidenceState
from openpilot.selfdrive.controls.lib.lead_context import (
  LEAD_AUTHORITY_NONE,
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


def test_no_status_fake_mpc_fallback_lead_never_becomes_primary():
  fake_fallback = make_lead(status=False, d_rel=50.0, v_lead=30.0, v_rel=20.0, y_rel=0.0)

  context = LeadContextTracker().update((fake_fallback, NO_LEAD), (empty_conf(), empty_conf()), v_ego=10.0, dt=0.1)

  assert context.physical_idx is None
  assert context.behavior_idx is None
  assert not context.has_physical_lead
  assert not context.lead_progress_allowed
