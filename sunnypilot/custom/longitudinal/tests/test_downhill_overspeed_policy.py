from __future__ import annotations

import pytest

from openpilot.sunnypilot.custom.longitudinal.decision import CandidateRole, decide
from openpilot.sunnypilot.custom.longitudinal.modes import LongitudinalMode
from openpilot.sunnypilot.custom.longitudinal.policy import (
  LongitudinalScene,
  build_candidates,
  dynamic_cruise_coast_accel,
  dynamic_cruise_overspeed_leeway,
)
from openpilot.sunnypilot.custom.longitudinal.policy_tables import (
  CRUISE_LEEWAY_HIGHWAY_MAX,
  CRUISE_LEEWAY_HIGHWAY_MIN_V_EGO,
  CRUISE_LEEWAY_MAX,
)

LIMITS = (-4.0, 2.0)


def coast_scene(**over):
  base = dict(
    v_ego=21.0, a_ego=0.0, v_cruise=20.0, seed_a_target=-0.8, accel_coast=-0.15,
    has_lead=False, lead_should_stop=False,
    lead_v=0.0, lead_d_rel=0.0, lead_v_rel=0.0, lead_a_k=0.0,
    follow_gap=0.0, lead_kinematics_valid=True,
    lead_confidence=0.0, lead_stable=False,
    lead_shadow_active=False, alternate_threat_active=False,
    stop_threat=False, force_slow_decel=False,
  )
  base.update(over)
  return LongitudinalScene(**base)


def test_downhill_overspeed_highway_expands_leeway_slightly():
  city = dynamic_cruise_overspeed_leeway(0.25, v_ego=10.0)
  highway = dynamic_cruise_overspeed_leeway(0.25, v_ego=CRUISE_LEEWAY_HIGHWAY_MIN_V_EGO + 2.0)
  assert city == pytest.approx(CRUISE_LEEWAY_MAX)
  assert highway == pytest.approx(CRUISE_LEEWAY_HIGHWAY_MAX)
  assert highway > city


def test_downhill_overspeed_city_leeway_capped_at_standard_max():
  assert dynamic_cruise_overspeed_leeway(0.25, v_ego=8.0) == pytest.approx(CRUISE_LEEWAY_MAX)


def test_downhill_overspeed_far_stable_opening_lead_allows_coast_decision():
  scene = coast_scene(
    v_ego=31.0, v_cruise=30.0, seed_a_target=-0.8, lead_a_target=-0.8, accel_coast=0.25,
    has_lead=True, lead_v=35.0, lead_v_rel=4.0, lead_d_rel=110.0,
    follow_gap=20.0, lead_kinematics_valid=True,
    lead_confidence=0.9, lead_stable=True,
  )
  cands = build_candidates(scene)
  coast = [c for c in cands if c.intent == "dynamic_overspeed_coast_leeway"]
  assert coast
  assert coast[0].a_target > scene.seed_a_target
  assert decide(cands, LongitudinalMode.ACC, LIMITS).a_target == pytest.approx(coast[0].a_target)


def test_downhill_overspeed_closing_slower_lead_blocks_coast_and_hazard_binds():
  scene = coast_scene(
    v_ego=31.0, v_cruise=30.0, seed_a_target=-0.8, accel_coast=0.25,
    has_lead=True, lead_v=25.0, lead_v_rel=-6.0, lead_d_rel=110.0,
    follow_gap=20.0, lead_kinematics_valid=True,
    lead_confidence=0.9, lead_stable=True, lead_a_target=-1.0,
  )
  cands = build_candidates(scene)
  assert not [c for c in cands if c.intent == "dynamic_overspeed_coast_leeway"]
  assert any(c.intent == "lead_follow" and c.role is CandidateRole.PHYSICAL_HAZARD for c in cands)
  assert decide(cands, LongitudinalMode.ACC, LIMITS).a_target == pytest.approx(scene.lead_a_target)


@pytest.mark.parametrize("extra", [
  {"lead_a_target": -1.0},
  {"lead_a_k": -0.8},
])
def test_downhill_overspeed_far_lead_with_real_braking_blocks_coast(extra):
  base = dict(
    v_ego=31.0, v_cruise=30.0, seed_a_target=-0.4, lead_a_target=-0.4, accel_coast=0.25,
    has_lead=True, lead_v=35.0, lead_v_rel=4.0, lead_d_rel=110.0,
    follow_gap=20.0, lead_kinematics_valid=True,
    lead_confidence=0.9, lead_stable=True,
  )
  base.update(extra)
  scene = coast_scene(**base)
  cands = build_candidates(scene)
  assert not [c for c in cands if c.intent == "dynamic_overspeed_coast_leeway"]
  if scene.lead_a_target < scene.seed_a_target:
    # The pulling-away far lead's -1.0 claim is kinematically uncorroborated, so its
    # hazard is relevance-capped near coast; the seed binds instead. Coast stays blocked.
    assert decide(cands, LongitudinalMode.ACC, LIMITS).a_target == pytest.approx(scene.seed_a_target)


@pytest.mark.parametrize("flag", [
  "lead_kinematics_valid", "lead_stable", "lead_shadow_active", "alternate_threat_active",
  "lead_should_stop", "low_confidence", "vrel_closing", "lead_speed_slower",
])
def test_downhill_overspeed_invalid_or_threat_lead_blocks_coast(flag):
  over = dict(
    has_lead=True, lead_v=35.0, lead_v_rel=4.0, lead_d_rel=110.0, follow_gap=20.0,
    lead_kinematics_valid=True, lead_confidence=0.9, lead_stable=True,
  )
  if flag == "lead_kinematics_valid":
    over["lead_kinematics_valid"] = False
  elif flag == "lead_stable":
    over["lead_stable"] = False
  elif flag == "lead_shadow_active":
    over["lead_shadow_active"] = True
  elif flag == "alternate_threat_active":
    over["alternate_threat_active"] = True
  elif flag == "lead_should_stop":
    over["lead_should_stop"] = True
  elif flag == "low_confidence":
    over["lead_confidence"] = 0.5
  elif flag == "vrel_closing":
    over["lead_v_rel"] = -0.1
  elif flag == "lead_speed_slower":
    over["lead_v"] = 20.0
  assert not [c for c in build_candidates(coast_scene(**over)) if c.intent == "dynamic_overspeed_coast_leeway"]


def test_downhill_overspeed_rate_guard_preserves_mild_braking_when_climbing():
  base = dict(v_ego=22.9, v_cruise=20.0, seed_a_target=-0.8, accel_coast=0.05)
  guard = dynamic_cruise_coast_accel(LongitudinalScene(**{**base, "a_ego": 0.2}), base["seed_a_target"])
  no_guard = dynamic_cruise_coast_accel(LongitudinalScene(**{**base, "a_ego": 0.0}), base["seed_a_target"])
  assert guard <= -0.2
  assert guard < no_guard
