"""Tests for the custom-2.0 longitudinal policy mechanisms and their mode-gated arbitration."""
from __future__ import annotations

import pytest

from openpilot.sunnypilot.custom.longitudinal.decision import CandidateRole, decide
from openpilot.sunnypilot.custom.longitudinal.modes import EvidenceClass, LongitudinalMode, SourceToggles
from openpilot.sunnypilot.custom.longitudinal.policy import (
  LongitudinalScene,
  build_candidates,
  dynamic_cruise_overspeed_leeway,
  no_lead_stop_clear,
  stop_approach_accel,
  stopping_decel,
)
from openpilot.sunnypilot.custom.longitudinal.policy_tables import (
  CRUISE_LEEWAY_MAX,
  CRUISE_LEEWAY_MIN,
  Personality,
  launch_accel_max,
)

LIMITS = (-4.0, 2.0)


def sources_of(cands):
  return {c.intent: c for c in cands}


def test_stopping_decel_kinematic():
  assert stopping_decel(10.0, 50.0) == pytest.approx(-1.0)  # -100/(2*50)
  assert stopping_decel(0.0, 50.0) == 0.0


def test_overspeed_leeway_scales_with_downhill_coast():
  flat = dynamic_cruise_overspeed_leeway(0.0)
  downhill = dynamic_cruise_overspeed_leeway(0.25)
  assert flat == pytest.approx(CRUISE_LEEWAY_MIN)
  assert downhill == pytest.approx(CRUISE_LEEWAY_MAX)
  assert downhill > flat


def test_downhill_overspeed_coasts_instead_of_braking():
  # slightly over set speed, rolling downhill, no lead -> coast (>= seed braking), no hard brake
  scene = LongitudinalScene(v_ego=21.0, v_cruise=20.0, seed_a_target=-0.8, accel_coast=-0.15)
  cands = build_candidates(scene)
  d = decide(cands, LongitudinalMode.SCC, LIMITS)
  assert d.a_target >= -0.8           # coast relaxes the braking
  assert d.a_target <= 0.0


def test_no_lead_launch_scales_with_personality():
  base = dict(v_ego=1.0, v_cruise=12.0, seed_a_target=0.0, model_desired_accel=0.0, model_stop_distance=50.0)
  for p in (Personality.RELAXED, Personality.STANDARD, Personality.AGGRESSIVE):
    cands = build_candidates(LongitudinalScene(personality=p, **base))
    d = decide(cands, LongitudinalMode.ACC, LIMITS)
    assert d.a_target == pytest.approx(min(launch_accel_max(p), LIMITS[1]))
  # aggressive launches harder than relaxed (within limits)
  assert launch_accel_max(Personality.AGGRESSIVE) > launch_accel_max(Personality.RELAXED)


def test_stop_approach_is_early_and_gentle_when_far():
  scene = LongitudinalScene(v_ego=10.0, v_cruise=15.0, seed_a_target=0.5,
                            model_should_stop=False, model_stop_distance=150.0, model_desired_accel=-0.2)
  a, hard = stop_approach_accel(scene)
  assert hard is False
  assert a == pytest.approx(-0.38)  # standard comfort decel dominates a far stop


def test_stop_approach_hardens_when_runway_short():
  scene = LongitudinalScene(v_ego=15.0, v_cruise=15.0, seed_a_target=0.0,
                            model_should_stop=True, model_stop_distance=20.0, model_desired_accel=-2.0)
  a, hard = stop_approach_accel(scene)
  assert hard is True
  assert a < -1.5


def test_acc_is_oem_like_excludes_model_stop_map_curve():
  scene = LongitudinalScene(
    v_ego=20.0, v_cruise=20.0, seed_a_target=0.3,
    model_should_stop=True, model_stop_distance=25.0, model_desired_accel=-2.5,
    curve_active=True, curve_a_target=-1.0,
    map_caution_active=True, map_caution_confirmed=True, map_caution_a_target=-1.5,
  )
  cands = build_candidates(scene)
  acc = decide(cands, LongitudinalMode.ACC, LIMITS)
  e2e = decide(cands, LongitudinalMode.E2E, LIMITS)
  # ACC ignores model-stop/map/curve -> cruise stands; E2E brakes for the model stop
  assert acc.a_target == pytest.approx(0.3)
  assert acc.should_stop is False
  assert e2e.a_target < 0.0
  assert e2e.should_stop is True


def test_scc_curve_cap_follows_toggle():
  scene = LongitudinalScene(v_ego=20.0, v_cruise=20.0, seed_a_target=0.5, curve_active=True, curve_a_target=-0.7)
  cands = build_candidates(scene)
  off = decide(cands, LongitudinalMode.SCC, LIMITS, SourceToggles(scc_curve_vision_enabled=False))
  on = decide(cands, LongitudinalMode.SCC, LIMITS, SourceToggles(scc_curve_vision_enabled=True))
  assert off.a_target == pytest.approx(0.5)   # curve source not admitted -> cruise stands
  # curve admitted -> the -0.7 cap applies, but comfort relax (clear road) softens it to the
  # -0.5 comfort floor. Net: the toggle causes braking.
  assert on.a_target < off.a_target
  assert on.a_target == pytest.approx(-0.5)


def test_lead_follow_hazard_binds():
  scene = LongitudinalScene(v_ego=20.0, v_cruise=25.0, seed_a_target=0.5, has_lead=True,
                            lead_a_target=-1.2, lead_should_stop=False)
  d = decide(build_candidates(scene), LongitudinalMode.ACC, LIMITS)
  assert d.a_target == pytest.approx(-1.2)
  assert d.reason == "physical_hazard"


def test_model_stop_trust_gated_in_policy():
  # low-confidence model stop in E2E -> softened, not committed; high confidence -> full + commit
  common = dict(v_ego=15.0, v_cruise=15.0, seed_a_target=0.0, model_should_stop=True,
                model_stop_distance=25.0, model_desired_accel=-3.0)
  low = decide(build_candidates(LongitudinalScene(model_stop_prob=0.2, **common)), LongitudinalMode.E2E, LIMITS)
  high = decide(build_candidates(LongitudinalScene(model_stop_prob=0.95, **common)), LongitudinalMode.E2E, LIMITS)
  assert low.should_stop is False          # not committed on a low-confidence model stop
  assert high.should_stop is True
  assert low.a_target > high.a_target       # softer braking than the trusted stop


def test_committed_far_stop_coasts_first():
  # high-confidence model stop 1 km away -> coast-first (hold/coast), not early braking
  far = LongitudinalScene(v_ego=20.0, v_cruise=20.0, seed_a_target=0.0, model_should_stop=True,
                          model_stop_distance=1000.0, model_desired_accel=-0.5, model_stop_prob=0.95)
  d = decide(build_candidates(far), LongitudinalMode.E2E, LIMITS)
  assert d.a_target >= -0.1
  # the same stop close in still brakes
  near = LongitudinalScene(v_ego=20.0, v_cruise=20.0, seed_a_target=0.0, model_should_stop=True,
                           model_stop_distance=18.0, model_desired_accel=-2.5, model_stop_prob=0.95)
  assert decide(build_candidates(near), LongitudinalMode.E2E, LIMITS).a_target < -1.0


def test_lead_pullaway_candidate_speedup_guarded():
  scene = LongitudinalScene(v_ego=20.0, v_cruise=25.0, seed_a_target=2.5, has_lead=True,
                            lead_a_target=1.0, lead_progress_allowed=True, lead_gap_excess=1.5,
                            lead_v=20.5, lead_d_rel=21.0, follow_gap=20.0)  # tight 1 m excess
  pull = [c for c in build_candidates(scene) if c.intent == "lead_pullaway"]
  assert pull and pull[0].a_target < 2.5    # guarded below the raw seed accel


def test_lead_cushion_advisory_when_runway():
  scene = LongitudinalScene(v_ego=20.0, v_cruise=22.0, seed_a_target=0.3, has_lead=True,
                            lead_a_target=0.0, lead_v=15.0, lead_d_rel=375.0, follow_gap=20.0,
                            accel_coast=-0.25)
  cushion = [c for c in build_candidates(scene) if c.intent == "lead_cushion"]
  assert cushion and cushion[0].role is CandidateRole.ADVISORY_CAP
  assert cushion[0].a_target < 0.0          # anticipatory gentle coast cap


def test_no_lead_stop_clear_gate():
  clear = LongitudinalScene(v_ego=1.0, v_cruise=12.0, seed_a_target=0.0, model_should_stop=False,
                            model_stop_distance=50.0, model_desired_accel=0.0)
  not_clear = LongitudinalScene(v_ego=1.0, v_cruise=12.0, seed_a_target=0.0, model_should_stop=True,
                                model_stop_distance=5.0, model_desired_accel=-1.0)
  assert no_lead_stop_clear(clear) is True
  assert no_lead_stop_clear(not_clear) is False
