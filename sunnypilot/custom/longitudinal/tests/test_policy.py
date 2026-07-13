"""Tests for the custom-2.0 longitudinal policy mechanisms and their mode-gated arbitration."""
from __future__ import annotations

from dataclasses import replace

import pytest

from openpilot.sunnypilot.custom.longitudinal.decision import CandidateRole, decide
from openpilot.sunnypilot.custom.longitudinal.modes import EvidenceClass, LongitudinalMode, SourceToggles
from openpilot.sunnypilot.custom.longitudinal.policy import (
  LongitudinalScene,
  build_candidates,
  dynamic_cruise_overspeed_leeway,
  map_coast_cap,
  no_lead_stop_clear,
  stop_approach_accel,
  stopping_decel,
)
from openpilot.sunnypilot.custom.longitudinal.model_trust import GENTLE_CAUTION_DECEL
from openpilot.sunnypilot.custom.longitudinal.policy_tables import (
  CRUISE_LEEWAY_MAX,
  CRUISE_LEEWAY_MIN,
  LEAD_CRAWL_ACCEL_MAX,
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


def test_stop_approach_softens_final_low_speed_landing_floor():
  scene = LongitudinalScene(v_ego=1.5, v_cruise=15.0, seed_a_target=0.0,
                            model_should_stop=False, model_stop_distance=0.38, model_desired_accel=-0.2)
  a, hard = stop_approach_accel(scene)
  assert hard is False
  assert -1.5 < a < -0.85


def test_acc_is_oem_like_excludes_model_stop_curve():
  scene = LongitudinalScene(
    v_ego=20.0, v_cruise=20.0, seed_a_target=0.3,
    model_should_stop=True, model_stop_distance=25.0, model_desired_accel=-2.5,
    curve_active=True, curve_a_target=-1.0,
  )
  cands = build_candidates(scene)
  acc = decide(cands, LongitudinalMode.ACC, LIMITS)
  e2e = decide(cands, LongitudinalMode.E2E, LIMITS)
  # ACC ignores model-stop/curve -> cruise stands; E2E brakes for the model stop
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


def test_stale_model_stop_distance_does_not_reharden_stop_approach():
  scene = LongitudinalScene(v_ego=15.0, v_cruise=15.0, seed_a_target=0.0,
                            model_should_stop=True, model_stop_distance=18.0,
                            model_desired_accel=-3.0, model_stop_prob=0.95,
                            model_stale=True)
  d = decide(build_candidates(scene), LongitudinalMode.E2E, LIMITS)
  assert d.a_target == pytest.approx(GENTLE_CAUTION_DECEL)
  assert d.should_stop is False


# -----------------------------------------------------------------------------
# Early, non-committing model slowdown caution
# -----------------------------------------------------------------------------

def test_early_model_slowdown_caution_before_stop_commitment():
  # Fresh model evidence shows meaningful slowdown before shouldStop / stop distance exist.
  scene = LongitudinalScene(v_ego=15.0, v_cruise=15.0, seed_a_target=0.0,
                            model_should_stop=False, model_stop_distance=None,
                            model_desired_accel=-0.6, model_stop_prob=0.95)
  cands = build_candidates(scene)
  stop = [c for c in cands if c.intent == "stop_approach"]
  assert len(stop) == 1
  assert stop[0].role is CandidateRole.PHYSICAL_HAZARD
  assert stop[0].source is EvidenceClass.MODEL_STOP
  assert stop[0].is_stop is False
  # Capped at the existing precautionary decel.
  assert stop[0].a_target == pytest.approx(GENTLE_CAUTION_DECEL)

  scc = decide(cands, LongitudinalMode.SCC, LIMITS)
  assert scc.should_stop is False
  assert scc.selected_intent == "stop_approach"
  assert scc.a_target == pytest.approx(GENTLE_CAUTION_DECEL)

  # ACC ignores model-stop evidence entirely.
  acc = decide(cands, LongitudinalMode.ACC, LIMITS)
  assert acc.a_target == pytest.approx(0.0)
  assert acc.selected_intent == "cruise"


def test_far_nonclosing_lead_caps_uncorroborated_model_caution():
  # Route 00000274: a deepened caution floor (-0.7) applied to a far, same-speed lead is a
  # phantom over-brake. With a far non-closing lead the non-committed caution is coast-capped.
  common = dict(v_ego=15.0, v_cruise=15.0, seed_a_target=0.0, model_should_stop=False,
                model_stop_distance=None, model_desired_accel=-0.7, model_caution_floor=-0.7)
  far = LongitudinalScene(has_lead=True, lead_d_rel=50.0, lead_v_rel=-0.5, **common)
  stop = [c for c in build_candidates(far) if c.intent == "stop_approach"]
  assert len(stop) == 1
  assert stop[0].a_target == pytest.approx(0.0)  # coast-capped

  # A closing lead (real approach) is NOT capped — the caution floor stands.
  closing = LongitudinalScene(has_lead=True, lead_d_rel=50.0, lead_v_rel=-2.0, **common)
  stop_c = [c for c in build_candidates(closing) if c.intent == "stop_approach"]
  assert stop_c[0].a_target == pytest.approx(-0.7)

  # A near lead (not far) is NOT capped either.
  near = LongitudinalScene(has_lead=True, lead_d_rel=20.0, lead_v_rel=-0.5, **common)
  stop_n = [c for c in build_candidates(near) if c.intent == "stop_approach"]
  assert stop_n[0].a_target == pytest.approx(-0.7)

  # A finite model stop point is independent evidence (for example, a traffic light) and
  # must retain braking authority even with a far, non-closing lead.
  finite_stop = replace(far, model_stop_distance=50.0)
  stop_s = [c for c in build_candidates(finite_stop) if c.intent == "stop_approach"]
  assert stop_s[0].a_target < 0.0


def test_early_model_slowdown_uses_raw_decel_within_cap():
  scene = LongitudinalScene(v_ego=15.0, v_cruise=15.0, seed_a_target=0.0,
                            model_should_stop=False, model_stop_distance=None,
                            model_desired_accel=-0.25)
  cands = build_candidates(scene)
  stop = [c for c in cands if c.intent == "stop_approach"]
  assert len(stop) == 1
  # Within the precautionary cap -> raw model decel is used.
  assert stop[0].a_target == pytest.approx(-0.25)
  assert stop[0].is_stop is False


def test_tiny_model_decel_does_not_trigger_early_slowdown_caution():
  scene = LongitudinalScene(v_ego=15.0, v_cruise=15.0, seed_a_target=0.0,
                            model_should_stop=False, model_stop_distance=None,
                            model_desired_accel=-0.1)
  cands = build_candidates(scene)
  assert not [c for c in cands if c.intent == "stop_approach"]
  d = decide(cands, LongitudinalMode.SCC, LIMITS)
  assert d.a_target == pytest.approx(0.0)
  assert d.should_stop is False


def test_committed_model_stop_takes_precedence_over_early_slowdown():
  # Once real stop evidence exists, the existing trust/runway logic is preserved.
  scene = LongitudinalScene(v_ego=15.0, v_cruise=15.0, seed_a_target=0.0,
                            model_should_stop=True, model_stop_distance=20.0,
                            model_desired_accel=-2.5, model_stop_prob=0.95)
  cands = build_candidates(scene)
  stop = [c for c in cands if c.intent == "stop_approach"]
  assert len(stop) == 1
  assert stop[0].is_stop is True
  d = decide(cands, LongitudinalMode.SCC, LIMITS)
  assert d.should_stop is True
  assert d.a_target < -1.0


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


def _launch_scene(**over):
  # Stop-and-go launch behind a close, authorized, opening lead. The MPC seed is timid (0.2) and
  # equals the lead-follow a_target, as the planner wires it (wiring.py).
  base = dict(v_ego=0.5, v_cruise=12.0, seed_a_target=0.2, has_lead=True, lead_a_target=0.2,
              lead_progress_allowed=True, lead_gap_excess=0.0,
              lead_v=2.0, lead_v_rel=1.5, lead_d_rel=7.0, follow_gap=6.0)
  base.update(over)
  return LongitudinalScene(**base)


def test_launch_behind_opening_lead_is_brisk_not_timid():
  # The hypermile launch fix: an authorized, opening close lead launches off the line instead of
  # being clamped to the timid MPC seed (the old gap_excess>0 gate never fires at a ~6 m gap).
  d = decide(build_candidates(_launch_scene()), LongitudinalMode.ACC, LIMITS)
  assert d.a_target == pytest.approx(min(launch_accel_max(Personality.STANDARD), (2.0 - 0.5) / 1.0))
  assert d.a_target > 0.2                                  # exceeds the timid seed
  assert d.a_target <= launch_accel_max(Personality.STANDARD)


def test_launch_tracks_lead_speed_gently_when_crawling():
  # A barely-moving lead gets a gentle, lead-tracking pull (no fixed lurch).
  d = decide(build_candidates(_launch_scene(v_ego=0.0, lead_v=0.3, lead_v_rel=0.3, lead_d_rel=6.5,
                                            seed_a_target=0.0, lead_a_target=0.0)),
             LongitudinalMode.ACC, LIMITS)
  assert d.a_target == pytest.approx(0.3 / 2.5)            # crawl-damped lead tracking, well below the cap
  assert 0.0 < d.a_target < launch_accel_max(Personality.STANDARD)


def test_close_crawl_pulse_is_damped_to_reduce_accordion():
  d = decide(build_candidates(_launch_scene(v_ego=0.0, lead_v=0.8, lead_v_rel=0.6, lead_d_rel=7.0,
                                            seed_a_target=0.0, lead_a_target=0.0)),
             LongitudinalMode.ACC, LIMITS)
  # 0.8 m/s crawl is now stronger than the old v/2.5=0.32 but still capped and guarded.
  assert d.a_target > 0.8 / 2.5
  assert d.a_target < 0.55


@pytest.mark.parametrize("lead_v", [0.6, 0.8, 1.0])
def test_crawl_launch_is_stronger_than_old_damping_and_capped(lead_v):
  # Use a fixed opening v_rel below the crawl breakout so all cases stay in the
  # crawl-launch regime and compare against the old v/2.5 damping.
  d = decide(build_candidates(_launch_scene(v_ego=0.0, lead_v=lead_v, lead_v_rel=0.6, lead_d_rel=7.0,
                                            seed_a_target=0.0, lead_a_target=0.0)),
             LongitudinalMode.ACC, LIMITS)
  assert d.a_target > lead_v / 2.5
  assert 0.0 < d.a_target <= LEAD_CRAWL_ACCEL_MAX


def test_clear_low_speed_breakout_uses_normal_launch_response():
  d = decide(build_candidates(_launch_scene(v_ego=0.0, lead_v=2.0, lead_v_rel=1.2, lead_d_rel=7.0,
                                            seed_a_target=0.0, lead_a_target=0.0)),
             LongitudinalMode.ACC, LIMITS)
  assert d.a_target == pytest.approx(launch_accel_max(Personality.STANDARD))


def test_route_282_opening_rate_breaks_out_of_crawl_damping():
  d = decide(build_candidates(_launch_scene(v_ego=0.0, lead_v=0.8, lead_v_rel=0.8, lead_d_rel=7.0,
                                            seed_a_target=0.0, lead_a_target=0.0)),
             LongitudinalMode.ACC, LIMITS)
  assert d.a_target == pytest.approx(0.8)


def test_crawl_launch_ramps_toward_launch_cap_as_lead_opens_gap():
  # Initial close gap stays on the old damped crawl curve.
  close = _launch_scene(v_ego=0.0, lead_v=0.8, lead_v_rel=0.6, lead_d_rel=7.0,
                        follow_gap=6.0, seed_a_target=0.0, lead_a_target=0.0)
  close_d = decide(build_candidates(close), LongitudinalMode.ACC, LIMITS)
  assert close_d.a_target == pytest.approx(0.16 + (0.8 - 0.4) / 1.5)

  # Same opening lead after it creates usable gap: ramp reaches normal launch accel.
  open_gap = _launch_scene(v_ego=0.0, lead_v=0.8, lead_v_rel=0.6, lead_d_rel=10.0,
                           follow_gap=6.0, seed_a_target=0.0, lead_a_target=0.0)
  open_d = decide(build_candidates(open_gap), LongitudinalMode.ACC, LIMITS)
  assert open_d.a_target > close_d.a_target
  assert open_d.a_target == pytest.approx(min(launch_accel_max(Personality.STANDARD), LIMITS[1]))


def test_launch_does_not_override_a_braking_seed():
  # Safety invariant: even with an opening, authorized lead, a braking MPC seed still binds —
  # the shaper never reduces the MPC's follow decel.
  scene = _launch_scene(v_ego=2.0, seed_a_target=-1.0, lead_a_target=-1.0,
                        lead_v=3.0, lead_v_rel=1.0, lead_d_rel=4.0)  # opening but we're too close
  d = decide(build_candidates(scene), LongitudinalMode.ACC, LIMITS)
  assert d.a_target == pytest.approx(-1.0)
  assert d.reason == "physical_hazard"


def test_launch_requires_authorization():
  # Without lead_progress_allowed the launch pull-away never fires; the timid seed stands.
  scene = _launch_scene(lead_progress_allowed=False)
  cands = build_candidates(scene)
  assert not [c for c in cands if c.intent == "lead_pullaway"]
  assert decide(cands, LongitudinalMode.ACC, LIMITS).a_target == pytest.approx(0.2)


def test_launch_pullaway_respects_speedup_guard_invariant():
  # For any authorized launch, applying the commanded accel for the 1 s lookahead must not leave
  # a re-settle decel worse than the speedup guard's comfort floor (-1.2 m/s^2).
  for v_ego in (0.0, 1.0, 3.0, 6.0):
    for v_rel in (0.5, 2.0, 5.0):            # genuinely opening: lead faster than ego
      for d_rel in (6.2, 8.0, 12.0):
        lead_v = v_ego + v_rel
        scene = _launch_scene(v_ego=v_ego, lead_v=lead_v, lead_v_rel=v_rel,
                              lead_d_rel=d_rel, follow_gap=6.0)
        pull = [c for c in build_candidates(scene) if c.intent == "lead_pullaway"]
        if not pull:
          continue
        a = pull[0].a_target
        assert 0.0 <= a <= launch_accel_max(Personality.STANDARD) + 1e-9
        excess_gap = d_rel - 6.0
        closing_next = max(0.0, (v_ego + a * 1.0) - lead_v)
        required = (closing_next * closing_next) / (2.0 * max(excess_gap, 1e-3))
        assert required <= 1.2 + 1e-6        # never digs a hole needing harder than comfort to undo


def test_lead_cushion_advisory_when_runway():
  scene = LongitudinalScene(v_ego=20.0, v_cruise=22.0, seed_a_target=0.3, has_lead=True,
                            lead_a_target=0.0, lead_v=15.0, lead_d_rel=375.0, follow_gap=20.0,
                            accel_coast=-0.25)
  cushion = [c for c in build_candidates(scene) if c.intent == "lead_cushion"]
  assert cushion and cushion[0].role is CandidateRole.ADVISORY_CAP
  assert cushion[0].a_target < 0.0          # anticipatory gentle coast cap


# -----------------------------------------------------------------------------
# Uphill grade recovery
# -----------------------------------------------------------------------------

def _uphill_scene(**over):
  base = dict(
    v_ego=15.0, v_cruise=22.0, seed_a_target=0.4, a_ego=0.1,
    accel_coast=-0.8, has_lead=False, stop_threat=False,
    model_should_stop=False, model_stop_distance=None,
    speed_limit_active=False, curve_active=False,
    force_slow_decel=False, brake_pressed=False, gas_pressed=False,
    personality=Personality.STANDARD,
  )
  base.update(over)
  return LongitudinalScene(**base)


def test_uphill_recovery_progress_candidate_conservative():
  scene = _uphill_scene()
  cands = build_candidates(scene)
  up = [c for c in cands if c.intent == "uphill_grade_recovery"]
  assert len(up) == 1
  c = up[0]
  assert c.role is CandidateRole.PROGRESS
  assert c.source is EvidenceClass.CRUISE
  assert c.authorized is True
  # additive gain is capped at +0.15 above the cruise seed
  assert c.a_target == pytest.approx(scene.seed_a_target + 0.15)
  assert c.a_target <= launch_accel_max(Personality.STANDARD)


def test_uphill_recovery_disabled_with_lead_present():
  assert not [c for c in build_candidates(_uphill_scene(has_lead=True)) if c.intent == "uphill_grade_recovery"]


def test_uphill_recovery_disabled_when_a_ego_surged():
  assert not [c for c in build_candidates(_uphill_scene(a_ego=0.8)) if c.intent == "uphill_grade_recovery"]


@pytest.mark.parametrize("accel_coast,label", [
  (-0.30, "flat_baseline"),
  (-0.55, "within_flat_band"),
  (0.10, "downhill"),
])
def test_uphill_recovery_no_flat_or_downhill(accel_coast, label):
  assert not [c for c in build_candidates(_uphill_scene(accel_coast=accel_coast)) if c.intent == "uphill_grade_recovery"], label


def test_uphill_recovery_disabled_without_progress_demand():
  assert not [c for c in build_candidates(_uphill_scene(v_cruise=15.0)) if c.intent == "uphill_grade_recovery"]


def test_uphill_recovery_does_not_turn_coast_seed_into_positive_accel():
  assert not [c for c in build_candidates(_uphill_scene(seed_a_target=0.0)) if c.intent == "uphill_grade_recovery"]
  assert not [c for c in build_candidates(_uphill_scene(seed_a_target=-0.1)) if c.intent == "uphill_grade_recovery"]


def test_uphill_recovery_absolute_target_cap():
  up = [c for c in build_candidates(_uphill_scene(seed_a_target=0.7)) if c.intent == "uphill_grade_recovery"]
  assert len(up) == 1
  assert up[0].a_target == pytest.approx(0.8)


def test_uphill_recovery_disabled_at_or_above_absolute_cap():
  assert not [c for c in build_candidates(_uphill_scene(seed_a_target=0.8)) if c.intent == "uphill_grade_recovery"]


def test_uphill_recovery_disabled_below_min_speed():
  assert not [c for c in build_candidates(_uphill_scene(v_ego=1.0)) if c.intent == "uphill_grade_recovery"]


@pytest.mark.parametrize("field", ["a_ego", "accel_coast"])
def test_uphill_recovery_disabled_on_nonfinite_inputs(field):
  assert not [c for c in build_candidates(_uphill_scene(**{field: float("nan")})) if c.intent == "uphill_grade_recovery"]


@pytest.mark.parametrize("flag", ["speed_limit_active", "curve_active", "model_should_stop", "force_slow_decel", "brake_pressed", "gas_pressed"])
def test_uphill_recovery_disabled_by_restrictive_contexts(flag):
  assert not [c for c in build_candidates(_uphill_scene(**{flag: True})) if c.intent == "uphill_grade_recovery"]


def test_uphill_recovery_decision_raises_target_in_cruise_mode():
  scene = _uphill_scene()
  cands = build_candidates(scene)
  d = decide(cands, LongitudinalMode.ACC, LIMITS)
  assert d.a_target > scene.seed_a_target
  assert d.a_target <= LIMITS[1]


def test_no_lead_stop_clear_gate():
  clear = LongitudinalScene(v_ego=1.0, v_cruise=12.0, seed_a_target=0.0, model_should_stop=False,
                            model_stop_distance=50.0, model_desired_accel=0.0)
  not_clear = LongitudinalScene(v_ego=1.0, v_cruise=12.0, seed_a_target=0.0, model_should_stop=True,
                                model_stop_distance=5.0, model_desired_accel=-1.0)
  assert no_lead_stop_clear(clear) is True
  assert no_lead_stop_clear(not_clear) is False


# -----------------------------------------------------------------------------
# Phase 4 lead-softening policy tests
# -----------------------------------------------------------------------------

def _soft_scene(**over):
  # Far, non-closing, low-risk lead-follow decel baseline.
  base = dict(
    v_ego=25.0, v_cruise=25.0, seed_a_target=-0.3,
    has_lead=True, lead_a_target=-0.3, lead_should_stop=False,
    lead_v=25.0, lead_d_rel=100.0, lead_v_rel=0.0, follow_gap=37.5,
    lead_kinematics_valid=True,
    model_should_stop=False, model_stop_distance=None,
    stop_threat=False, force_slow_decel=False, brake_pressed=False, gas_pressed=False,
  )
  base.update(over)
  return LongitudinalScene(**base)


def test_far_non_closing_lead_softens_to_advisory_and_progress():
  scene = _soft_scene()
  cands = build_candidates(scene)
  intents = sources_of(cands)
  assert "lead_follow" not in intents          # physical hazard suppressed
  assert "lead_follow_soft" in intents
  assert "lead_follow_soft_desire" in intents
  assert intents["lead_follow_soft"].role is CandidateRole.ADVISORY_CAP
  assert intents["lead_follow_soft"].source is EvidenceClass.LEAD
  assert intents["lead_follow_soft_desire"].role is CandidateRole.PROGRESS
  assert intents["lead_follow_soft_desire"].authorized is True
  assert intents["lead_follow_soft"].a_target == pytest.approx(-0.05)
  # Output is raised from the -0.3 seed to the soft near-coast target.
  d = decide(cands, LongitudinalMode.ACC, LIMITS)
  assert d.a_target == pytest.approx(-0.05)
  assert d.reason == "advisory_capped"


def test_mid_gentle_closing_accepted_only_with_dynamic_excess_gap():
  # v_ego=20 -> desired/min gap 44 m; closing 1 m/s at 60 m gives tiny required decel.
  accepted = _soft_scene(v_ego=20.0, lead_v=19.0, lead_v_rel=-1.0, lead_d_rel=60.0,
                         follow_gap=30.0, lead_a_target=-0.3, seed_a_target=-0.3)
  cands = build_candidates(accepted)
  assert "lead_follow_soft" in sources_of(cands)

  # Just enough distance to pass min_distance (45 m) but not enough usable excess.
  rejected_margin = _soft_scene(v_ego=20.0, lead_v=19.0, lead_v_rel=-1.0, lead_d_rel=47.0,
                                follow_gap=30.0, lead_a_target=-0.3)
  assert "lead_follow" in sources_of(build_candidates(rejected_margin))
  assert "lead_follow_soft" not in sources_of(build_candidates(rejected_margin))


def test_close_headway_rejected():
  # Same speed, well below the dynamic min distance for v_ego=25 (55 m).
  scene = _soft_scene(lead_d_rel=40.0)
  cands = build_candidates(scene)
  intents = sources_of(cands)
  assert "lead_follow" in intents
  assert "lead_follow_soft" not in intents


def test_high_required_decel_rejected():
  # 4 m/s closing over 16 m excess -> 0.5 m/s^2 required, above the 0.25 threshold.
  scene = _soft_scene(lead_v_rel=-4.0, lead_d_rel=60.0, follow_gap=30.0)
  assert "lead_follow_soft" not in sources_of(build_candidates(scene))


def test_strong_lead_decel_rejected():
  scene = _soft_scene(lead_a_target=-0.8)
  assert "lead_follow_soft" not in sources_of(build_candidates(scene))


def test_stopped_or_crawling_lead_rejected():
  for lead_v in (0.0, 3.0):
    scene = _soft_scene(lead_v=lead_v)
    assert "lead_follow_soft" not in sources_of(build_candidates(scene))


def test_lead_softening_rejected_by_stop_and_override_blockers():
  blockers = (
    "lead_should_stop",
    "model_should_stop",
    "stop_threat",
    "force_slow_decel",
    "brake_pressed",
    "gas_pressed",
  )
  for name in blockers:
    assert "lead_follow_soft" not in sources_of(build_candidates(_soft_scene(**{name: True}))), name


def test_lead_softening_rejected_near_model_stop():
  # model stop distance inside the dynamic clearance -> reject
  scene = _soft_scene(model_should_stop=False, model_stop_distance=30.0)
  assert "lead_follow_soft" not in sources_of(build_candidates(scene))
  # non-finite model stop distance -> reject
  scene = _soft_scene(model_should_stop=False, model_stop_distance=float("nan"))
  assert "lead_follow_soft" not in sources_of(build_candidates(scene))


# -----------------------------------------------------------------------------
# Far-lead relevance cap tests (routes 0000027d/00000282 track-churn decel dips)
# -----------------------------------------------------------------------------

def test_far_lead_churn_spike_capped_to_coast():
  # aLeadK churn spike deeper than the softening window (-0.8 < -0.7) on a far
  # same-speed lead: the raw hazard is capped near coast, not commanded at -0.8.
  scene = _soft_scene(lead_a_target=-0.8, seed_a_target=-0.8)
  intents = sources_of(build_candidates(scene))
  assert "lead_follow_soft" not in intents
  assert intents["lead_follow"].a_target == pytest.approx(-0.1)


def test_relevance_cap_scales_with_closing_speed():
  # v_ego=25 -> desired gap 55 m; at 100 m, closing 3 m/s: excess 45 m,
  # required = 9/90 = 0.1, cap = -(2.5*0.1 + 0.1) = -0.35.
  scene = _soft_scene(lead_a_target=-0.8, seed_a_target=-0.8,
                      lead_v=22.0, lead_v_rel=-3.0)
  intents = sources_of(build_candidates(scene))
  assert intents["lead_follow"].a_target == pytest.approx(-0.35)
  # Harder closing at the same distance loosens the cap below the raw target:
  # closing 6 -> required 0.4 -> cap -1.1 < -0.8, so the hazard passes unchanged.
  scene = _soft_scene(lead_a_target=-0.8, seed_a_target=-0.8,
                      lead_v=19.0, lead_v_rel=-6.0)
  intents = sources_of(build_candidates(scene))
  assert intents["lead_follow"].a_target == pytest.approx(-0.8)


def test_relevance_cap_never_hardens():
  # Mild target above the cap stays untouched (max() semantics).
  scene = _soft_scene(lead_a_target=-0.72, seed_a_target=-0.72, lead_v_rel=-4.0,
                      lead_d_rel=60.0, follow_gap=30.0)
  # closing 4 over excess 5 -> required 1.6 -> cap -4.1; hazard keeps -0.72.
  intents = sources_of(build_candidates(scene))
  assert intents["lead_follow"].a_target == pytest.approx(-0.72)


def test_close_lead_keeps_full_authority():
  # Inside the trust floor (40 m < min distance 55 m at v_ego=25) the churn cap
  # must not apply: the raw hazard passes through.
  scene = _soft_scene(lead_a_target=-0.8, seed_a_target=-0.8, lead_d_rel=40.0)
  intents = sources_of(build_candidates(scene))
  assert intents["lead_follow"].a_target == pytest.approx(-0.8)


def test_stop_committed_far_lead_keeps_full_authority():
  scene = _soft_scene(lead_a_target=-1.2, seed_a_target=-1.2, lead_should_stop=True)
  intents = sources_of(build_candidates(scene))
  assert intents["lead_follow"].a_target == pytest.approx(-1.2)
  assert intents["lead_follow"].is_stop is True


def test_lead_softening_rejected_on_bad_kinematics():
  # non-finite lead_d_rel
  assert "lead_follow_soft" not in sources_of(build_candidates(_soft_scene(lead_d_rel=float("nan"))))
  # raw live kinematics flagged invalid
  assert "lead_follow_soft" not in sources_of(build_candidates(_soft_scene(lead_kinematics_valid=False)))


def test_lead_cushion_can_still_bind_below_soft_target():
  # Slower lead with just enough runway: softening accepts (closing is gentle), but the
  # lead-following cushion advisory cap is stronger and binds lower.
  scene = LongitudinalScene(
    v_ego=25.0, v_cruise=25.0, seed_a_target=-0.3,
    has_lead=True, lead_a_target=-0.3,
    lead_v=24.7, lead_d_rel=70.0, lead_v_rel=-0.3, follow_gap=37.5,
    lead_kinematics_valid=True, accel_coast=-0.25,
  )
  cands = build_candidates(scene)
  intents = sources_of(cands)
  assert "lead_follow_soft" in intents
  assert "lead_cushion" in intents
  d = decide(cands, LongitudinalMode.ACC, LIMITS)
  # The stronger cushion cap (around -0.25) binds below the soft -0.05 target.
  assert d.a_target < intents["lead_follow_soft"].a_target
  assert d.a_target < -0.1


# -----------------------------------------------------------------------------
# Phase 1 lead-speed alignment policy integration tests
# -----------------------------------------------------------------------------

def _alignment_scene(**over):
  base = dict(
    v_ego=25.0, a_ego=0.0, v_cruise=25.0, seed_a_target=0.0,
    has_lead=True, lead_a_target=0.0, lead_should_stop=False,
    lead_v=25.0, lead_d_rel=100.0, lead_v_rel=0.0, lead_a_k=0.0,
    follow_gap=37.5, lead_kinematics_valid=True,
    lead_confidence=0.9, lead_stable=True, lead_progress_allowed=True,
    lead_shadow_active=False, alternate_threat_active=False,
    model_should_stop=False, model_stop_distance=None,
    force_slow_decel=False, brake_pressed=False, gas_pressed=False,
    personality=Personality.STANDARD,
  )
  base.update(over)
  return LongitudinalScene(**base)


def test_far_slower_lead_adds_alignment_gentle_brake_candidate():
  scene = _alignment_scene(
    v_ego=6.0, v_cruise=6.0, lead_v=5.0, lead_v_rel=-1.0,
    lead_d_rel=14.0, follow_gap=8.0, lead_progress_allowed=False,
  )
  cands = build_candidates(scene)
  intents = sources_of(cands)
  assert "lead_alignment_gentle_brake" in intents
  assert intents["lead_alignment_gentle_brake"].role is CandidateRole.ADVISORY_CAP
  assert intents["lead_alignment_gentle_brake"].source is EvidenceClass.LEAD
  assert -0.35 <= intents["lead_alignment_gentle_brake"].a_target <= 0.0


def test_standstill_stable_lead_adds_standstill_launch_candidate():
  scene = _alignment_scene(
    v_ego=0.0, lead_v=1.5, lead_v_rel=1.5, lead_d_rel=8.0,
    follow_gap=6.0, lead_a_target=0.0, seed_a_target=0.0,
  )
  cands = build_candidates(scene)
  intents = sources_of(cands)
  assert "lead_standstill_launch" in intents
  assert intents["lead_standstill_launch"].role is CandidateRole.PROGRESS
  assert intents["lead_standstill_launch"].authorized is True
  assert 0.0 < intents["lead_standstill_launch"].a_target <= launch_accel_max(Personality.STANDARD)


def test_mid_speed_pullaway_alignment_is_not_clamped_by_non_braking_seed():
  scene = _alignment_scene(
    v_ego=12.0, v_cruise=20.0, seed_a_target=0.1,
    lead_a_target=0.1, lead_v=15.0, lead_v_rel=3.0,
    lead_d_rel=45.0, follow_gap=18.0, lead_gap_excess=0.0,
  )
  cands = build_candidates(scene)
  intents = sources_of(cands)
  assert "lead_pullaway_alignment" in intents
  assert "lead_follow" not in intents
  d = decide(cands, LongitudinalMode.ACC, LIMITS)
  assert d.a_target > scene.seed_a_target


def test_alignment_blocked_by_model_stop():
  scene = _alignment_scene(
    v_ego=0.0, lead_v=1.5, lead_v_rel=1.5, lead_d_rel=8.0,
    follow_gap=6.0, model_should_stop=True,
  )
  cands = build_candidates(scene)
  intents = sources_of(cands)
  assert "lead_standstill_launch" not in intents
  assert "lead_alignment_coast" not in intents


# -----------------------------------------------------------------------------
# Phase 3 inside-gap compression/recovery
# -----------------------------------------------------------------------------

def test_inside_gap_stable_lead_recovery_coasts():
  scene = _alignment_scene(
    v_ego=20.0, v_cruise=20.0, seed_a_target=-0.3,
    lead_a_target=-0.3, lead_v=20.0, lead_v_rel=0.0, lead_d_rel=25.0,
    follow_gap=30.0, lead_progress_allowed=False,
  )
  cands = build_candidates(scene)
  intents = sources_of(cands)
  assert "lead_follow" not in intents          # physical hazard suppressed
  assert "lead_gap_recovery_coast" in intents
  d = decide(cands, LongitudinalMode.ACC, LIMITS)
  assert d.a_target == pytest.approx(0.0)
  assert d.reason == "advisory_capped"


def test_inside_gap_gentle_closing_caps_hazard():
  scene = _alignment_scene(
    v_ego=20.0, v_cruise=20.0, seed_a_target=-0.8,
    lead_a_target=-0.8, lead_v=19.8, lead_v_rel=-0.2, lead_d_rel=25.0,
    follow_gap=30.0, lead_progress_allowed=False,
  )
  cands = build_candidates(scene)
  intents = sources_of(cands)
  assert "lead_follow" not in intents
  assert "lead_gap_compression" in intents
  d = decide(cands, LongitudinalMode.ACC, LIMITS)
  assert d.a_target == pytest.approx(-0.15)   # tiny closing -> gentle, not floor
  assert d.reason == "physical_hazard"


def test_inside_gap_fast_closing_stays_hazard():
  # Closing above the routine tier limit (2.5 m/s) must stay raw hazard.
  scene = _alignment_scene(
    v_ego=20.0, v_cruise=20.0, seed_a_target=-1.5,
    lead_a_target=-1.5, lead_v=17.4, lead_v_rel=-2.6, lead_d_rel=25.0,
    follow_gap=30.0, lead_progress_allowed=False,
  )
  cands = build_candidates(scene)
  intents = sources_of(cands)
  assert "lead_follow" in intents
  assert "lead_gap_compression" not in intents
  d = decide(cands, LongitudinalMode.ACC, LIMITS)
  assert d.a_target == pytest.approx(-1.5)


def test_too_close_inside_gap_stays_hazard():
  scene = _alignment_scene(
    v_ego=20.0, v_cruise=20.0, seed_a_target=-1.0,
    lead_a_target=-1.0, lead_v=20.0, lead_v_rel=0.0, lead_d_rel=4.0,
    follow_gap=6.0, lead_progress_allowed=False,
  )
  cands = build_candidates(scene)
  assert "lead_follow" in sources_of(cands)
  assert "lead_gap_recovery_coast" not in sources_of(cands)


def test_inside_gap_short_gap_at_speed_rejected():
  scene = _alignment_scene(
    v_ego=20.0, v_cruise=20.0, seed_a_target=-0.8,
    lead_a_target=-0.8, lead_v=20.0, lead_v_rel=0.0, lead_d_rel=5.0,
    follow_gap=30.0, lead_progress_allowed=False,
  )
  cands = build_candidates(scene)
  intents = sources_of(cands)
  assert "lead_follow" in intents
  assert "lead_gap_recovery_coast" not in intents
  d = decide(cands, LongitudinalMode.ACC, LIMITS)
  assert d.a_target == pytest.approx(-0.8)


def test_inside_gap_boundary_just_safe():
  # v_ego=20 -> time_gap = 22.1/20 = 1.105, just above the 1.1 s floor.
  scene = _alignment_scene(
    v_ego=20.0, v_cruise=20.0, seed_a_target=-0.3,
    lead_a_target=-0.3, lead_v=20.0, lead_v_rel=0.0, lead_d_rel=22.1,
    follow_gap=30.0, lead_progress_allowed=False,
  )
  cands = build_candidates(scene)
  assert "lead_gap_recovery_coast" in sources_of(cands)


def test_inside_gap_unstable_lead_rejected():
  scene = _alignment_scene(
    v_ego=20.0, v_cruise=20.0, seed_a_target=-0.3,
    lead_a_target=-0.3, lead_v=20.0, lead_v_rel=0.0, lead_d_rel=25.0,
    follow_gap=30.0, lead_progress_allowed=False, lead_stable=False,
  )
  cands = build_candidates(scene)
  assert "lead_gap_recovery_coast" not in sources_of(cands)
  assert "lead_follow" in sources_of(cands)


def test_inside_gap_low_confidence_lead_rejected():
  scene = _alignment_scene(
    v_ego=20.0, v_cruise=20.0, seed_a_target=-0.3,
    lead_a_target=-0.3, lead_v=20.0, lead_v_rel=0.0, lead_d_rel=25.0,
    follow_gap=30.0, lead_progress_allowed=False, lead_confidence=0.5,
  )
  cands = build_candidates(scene)
  assert "lead_gap_recovery_coast" not in sources_of(cands)
  assert "lead_follow" in sources_of(cands)


# -----------------------------------------------------------------------------
# Phase 3b: controlled lead-compression expansion
# -----------------------------------------------------------------------------

def test_inside_gap_moderate_braking_compresses_gently():
  # Stable/confident same lead braking moderately; low collision risk -> ramped cap rather than raw seed.
  scene = _alignment_scene(
    v_ego=10.0, v_cruise=10.0, seed_a_target=-1.5,
    lead_a_target=-1.5, lead_v=8.8, lead_v_rel=-1.2, lead_a_k=-0.8,
    lead_d_rel=15.0, follow_gap=15.0, lead_progress_allowed=False,
  )
  cands = build_candidates(scene)
  intents = sources_of(cands)
  assert "lead_follow" not in intents
  assert "lead_gap_compression" in intents
  assert intents["lead_gap_compression"].role is CandidateRole.PHYSICAL_HAZARD
  assert intents["lead_gap_compression"].source is EvidenceClass.LEAD
  # Kinematic demand is low -> target scales with closing, not the full -0.45 floor.
  assert intents["lead_gap_compression"].a_target == pytest.approx(-0.20, abs=0.02)
  assert "lead_gap_compression_desire" in intents
  assert intents["lead_gap_compression_desire"].role is CandidateRole.PROGRESS
  assert intents["lead_gap_compression_desire"].authorized is True
  d = decide(cands, LongitudinalMode.ACC, LIMITS)
  assert d.a_target == pytest.approx(-0.20, abs=0.02)
  assert d.reason == "physical_hazard"


def test_inside_gap_short_time_gap_stays_hazard():
  scene = _alignment_scene(
    v_ego=13.0, v_cruise=13.0, seed_a_target=-1.5,
    lead_a_target=-1.5, lead_v=11.8, lead_v_rel=-1.2, lead_a_k=-0.8,
    lead_d_rel=14.0, follow_gap=15.0, lead_progress_allowed=False,
  )
  # time_gap = 14/13 ~ 1.077 < 1.1
  cands = build_candidates(scene)
  intents = sources_of(cands)
  assert "lead_follow" in intents
  assert "lead_gap_compression" not in intents
  d = decide(cands, LongitudinalMode.ACC, LIMITS)
  assert d.a_target == pytest.approx(-1.5)


def test_inside_gap_excess_closing_stays_hazard():
  # Closing above the routine tier limit keeps the raw lead-follow hazard.
  scene = _alignment_scene(
    v_ego=20.0, v_cruise=20.0, seed_a_target=-1.5,
    lead_a_target=-1.5, lead_v=17.4, lead_v_rel=-2.6, lead_a_k=-1.2,
    lead_d_rel=25.0, follow_gap=30.0, lead_progress_allowed=False,
  )
  cands = build_candidates(scene)
  intents = sources_of(cands)
  assert "lead_follow" in intents
  assert "lead_gap_compression" not in intents
  d = decide(cands, LongitudinalMode.ACC, LIMITS)
  assert d.a_target == pytest.approx(-1.5)


def test_inside_gap_hazardous_kinematics_stays_hazard():
  # TTC ~ 3.6 s and required decel ~ 0.72 m/s^2 both exceed safe floors.
  scene = _alignment_scene(
    v_ego=5.0, v_cruise=5.0, seed_a_target=-2.0,
    lead_a_target=-2.0, lead_v=3.2, lead_v_rel=-1.8, lead_a_k=-1.0,
    lead_d_rel=6.5, follow_gap=4.5, lead_progress_allowed=False,
  )
  cands = build_candidates(scene)
  intents = sources_of(cands)
  assert "lead_follow" in intents
  assert "lead_gap_compression" not in intents
  d = decide(cands, LongitudinalMode.ACC, LIMITS)
  assert d.a_target == pytest.approx(-2.0)


@pytest.mark.parametrize("flag", ["lead_stable", "lead_confidence"])
def test_inside_gap_moderate_braking_rejected_when_unstable_or_low_confidence(flag):
  over = {"lead_stable": False} if flag == "lead_stable" else {"lead_confidence": 0.5}
  scene = _alignment_scene(
    v_ego=10.0, v_cruise=10.0, seed_a_target=-1.5,
    lead_a_target=-1.5, lead_v=8.8, lead_v_rel=-1.2, lead_a_k=-0.8,
    lead_d_rel=15.0, follow_gap=15.0, lead_progress_allowed=False,
    **over,
  )
  cands = build_candidates(scene)
  intents = sources_of(cands)
  assert "lead_follow" in intents
  assert "lead_gap_compression" not in intents
  d = decide(cands, LongitudinalMode.ACC, LIMITS)
  assert d.a_target == pytest.approx(-1.5)


def test_inside_gap_moderate_braking_rejected_when_confidence_nonfinite():
  scene = _alignment_scene(
    v_ego=10.0, v_cruise=10.0, seed_a_target=-1.5,
    lead_a_target=-1.5, lead_v=8.8, lead_v_rel=-1.2, lead_a_k=-0.8,
    lead_d_rel=15.0, follow_gap=15.0, lead_progress_allowed=False,
    lead_confidence=float("nan"),
  )
  cands = build_candidates(scene)
  intents = sources_of(cands)
  assert "lead_follow" in intents
  assert "lead_gap_compression" not in intents
  d = decide(cands, LongitudinalMode.ACC, LIMITS)
  assert d.a_target == pytest.approx(-1.5)


@pytest.mark.parametrize("flag", ["model_should_stop", "force_slow_decel", "brake_pressed", "gas_pressed"])
def test_inside_gap_moderate_braking_rejected_when_stop_or_driver_override(flag):
  scene = _alignment_scene(
    v_ego=10.0, v_cruise=10.0, seed_a_target=-1.5,
    lead_a_target=-1.5, lead_v=8.8, lead_v_rel=-1.2, lead_a_k=-0.8,
    lead_d_rel=15.0, follow_gap=15.0, lead_progress_allowed=False,
    **{flag: True},
  )
  cands = build_candidates(scene)
  intents = sources_of(cands)
  assert "lead_follow" in intents
  assert "lead_gap_compression" not in intents
  d = decide(cands, LongitudinalMode.ACC, LIMITS)
  assert d.a_target == pytest.approx(-1.5)


def test_inside_gap_tiny_closing_is_gentle_not_full_floor():
  # Tiny 0.11 m/s closing with safe kinematics must not jump to the -0.45 floor.
  scene = _alignment_scene(
    v_ego=20.0, v_cruise=20.0, seed_a_target=-0.3,
    lead_a_target=-0.3, lead_v=19.89, lead_v_rel=-0.11, lead_a_k=-0.5,
    lead_d_rel=25.0, follow_gap=30.0, lead_progress_allowed=False,
  )
  cands = build_candidates(scene)
  intents = sources_of(cands)
  assert "lead_gap_compression" in intents
  assert intents["lead_gap_compression"].a_target == pytest.approx(-0.15)
  d = decide(cands, LongitudinalMode.ACC, LIMITS)
  assert d.a_target == pytest.approx(-0.15)


def test_inside_gap_hard_lead_a_k_rejects_compression():
  # A lead that is already braking hard must not be softened, even with low closing.
  scene = _alignment_scene(
    v_ego=10.0, v_cruise=10.0, seed_a_target=-1.5,
    lead_a_target=-1.5, lead_v=9.0, lead_v_rel=-1.0, lead_a_k=-3.0,
    lead_d_rel=15.0, follow_gap=15.0, lead_progress_allowed=False,
  )
  cands = build_candidates(scene)
  assert "lead_follow" in sources_of(cands)
  assert "lead_gap_compression" not in sources_of(cands)
  d = decide(cands, LongitudinalMode.ACC, LIMITS)
  assert d.a_target == pytest.approx(-1.5)


def test_inside_gap_required_decel_above_ceiling_rejected():
  # required_decel ~0.60 is above the 0.45 ceiling while speed/time-gap/TTC gates pass
  # -> keep raw hazard, do not coast/cap.
  scene = _alignment_scene(
    v_ego=5.0, v_cruise=5.0, seed_a_target=-1.2,
    lead_a_target=-1.2, lead_v=3.9, lead_v_rel=-1.1, lead_a_k=-0.5,
    lead_d_rel=6.0, follow_gap=6.0, lead_progress_allowed=False,
  )
  cands = build_candidates(scene)
  assert "lead_follow" in sources_of(cands)
  assert "lead_gap_compression" not in sources_of(cands)
  d = decide(cands, LongitudinalMode.ACC, LIMITS)
  assert d.a_target == pytest.approx(-1.2)


def test_inside_gap_max_closing_hits_mild_floor():
  # At the maximum allowed closing (2.0 m/s) with high kinematic demand, the ramped
  # target should approach but not exceed the gentle compression floor.
  scene = _alignment_scene(
    v_ego=5.0, v_cruise=5.0, seed_a_target=-1.5,
    lead_a_target=-1.5, lead_v=3.0, lead_v_rel=-2.0, lead_a_k=-0.5,
    lead_d_rel=10.0, follow_gap=8.0, lead_progress_allowed=False,
  )
  cands = build_candidates(scene)
  intents = sources_of(cands)
  assert "lead_gap_compression" in intents
  assert intents["lead_gap_compression"].a_target == pytest.approx(-0.45)
  d = decide(cands, LongitudinalMode.ACC, LIMITS)
  assert d.a_target == pytest.approx(-0.45)


# -----------------------------------------------------------------------------
# Phase 3c: routine-braking compression tier
# -----------------------------------------------------------------------------

def _routine_scene(**over):
  base = dict(
    v_ego=15.0, v_cruise=15.0, seed_a_target=-1.2,
    has_lead=True, lead_a_target=-1.2, lead_should_stop=False,
    lead_v=12.5, lead_d_rel=18.0, lead_v_rel=-2.5, lead_a_k=-1.0,
    follow_gap=22.5, lead_progress_allowed=False,
    lead_kinematics_valid=True, lead_confidence=0.9, lead_stable=True,
    lead_shadow_active=False, alternate_threat_active=False,
    model_should_stop=False, model_stop_distance=None,
    stop_threat=False, force_slow_decel=False, brake_pressed=False, gas_pressed=False,
  )
  base.update(over)
  return LongitudinalScene(**base)


def test_routine_tier_compresses_at_moderate_required_decel():
  # required_decel ~0.46 exceeds the comfort ceiling (0.45) but is safe for routine tier.
  scene = _routine_scene()
  cands = build_candidates(scene)
  intents = sources_of(cands)
  assert "lead_gap_compression" in intents
  assert "lead_follow" not in intents
  # Ramp target should be in the routine -0.45..-0.85 range, not raw -1.2.
  target = intents["lead_gap_compression"].a_target
  assert -0.85 <= target <= -0.45
  d = decide(cands, LongitudinalMode.ACC, LIMITS)
  assert -0.85 <= d.a_target <= -0.45


def test_routine_tier_rejected_when_ttc_too_low():
  # TTC just below the 6.0 s routine floor while time-gap, closing, and decel gates pass
  # -> fall back to raw lead-follow hazard.
  scene = _routine_scene(v_ego=10.0, lead_d_rel=12.0, follow_gap=10.0, lead_v=7.9, lead_v_rel=-2.1)
  # TTC = 12 / 2.1 ~ 5.7 s; time_gap = 12 / 10 = 1.2 s (OK).
  cands = build_candidates(scene)
  assert "lead_follow" in sources_of(cands)
  assert "lead_gap_compression" not in sources_of(cands)
  d = decide(cands, LongitudinalMode.ACC, LIMITS)
  assert d.a_target == pytest.approx(-1.2)


def test_routine_tier_rejected_when_time_gap_too_short():
  scene = _routine_scene(v_ego=18.0, lead_d_rel=18.0, lead_v=15.5, lead_v_rel=-2.5)
  # time_gap = 18 / 18 = 1.0 s < 1.2 s, TTC fine.
  cands = build_candidates(scene)
  assert "lead_follow" in sources_of(cands)
  assert "lead_gap_compression" not in sources_of(cands)
  d = decide(cands, LongitudinalMode.ACC, LIMITS)
  assert d.a_target == pytest.approx(-1.2)


def test_routine_tier_rejected_when_lead_braking_hard():
  scene = _routine_scene(lead_a_k=-2.5)
  cands = build_candidates(scene)
  assert "lead_follow" in sources_of(cands)
  assert "lead_gap_compression" not in sources_of(cands)
  d = decide(cands, LongitudinalMode.ACC, LIMITS)
  assert d.a_target == pytest.approx(-1.2)


def test_routine_tier_rejected_when_collision_buffer_tight():
  # Tight collision-buffer cases must stay raw hazard. This also trips the TTC gate, which is
  # expected: near-collision-buffer cases should have redundant fail-closed reasons.
  scene = _routine_scene(
    v_ego=6.0, lead_d_rel=7.5, follow_gap=6.0,
    lead_v=3.7, lead_v_rel=-2.3, lead_a_k=-1.0,
  )
  cands = build_candidates(scene)
  assert "lead_follow" in sources_of(cands)
  assert "lead_gap_compression" not in sources_of(cands)
  d = decide(cands, LongitudinalMode.ACC, LIMITS)
  assert d.a_target == pytest.approx(-1.2)


def test_routine_tier_never_hardens_milder_raw_lead_target():
  scene = _routine_scene(lead_a_target=-0.50, seed_a_target=-0.50)
  cands = build_candidates(scene)
  intents = sources_of(cands)
  assert "lead_gap_compression" in intents
  assert intents["lead_gap_compression"].a_target == pytest.approx(-0.50)
  d = decide(cands, LongitudinalMode.ACC, LIMITS)
  assert d.a_target == pytest.approx(-0.50)


# -----------------------------------------------------------------------------
# Runway-aware advisory caps (Phase 5)
# -----------------------------------------------------------------------------

def test_speed_limit_runway_governor_shapes_long_runway_cap():
  # Far speed-limit change: with a known distance the cap should coast-first (much higher than
  # the raw kinematic decel); without distance it falls back to the coast-biased raw cap.
  base = dict(
    v_ego=25.0, v_cruise=25.0, seed_a_target=0.0,
    speed_limit_active=True, speed_limit_v_target=15.0, speed_limit_a_target=-1.0,
    accel_coast=-0.25,
  )
  with_dist = LongitudinalScene(speed_limit_distance=1000.0, **base)
  without_dist = LongitudinalScene(speed_limit_distance=None, **base)
  shaped = [c for c in build_candidates(with_dist) if c.intent == "speed_policy"][0]
  fallback = [c for c in build_candidates(without_dist) if c.intent == "speed_policy"][0]
  assert shaped.a_target > fallback.a_target          # coast-first is gentler
  assert fallback.a_target == pytest.approx(-0.25)    # coast-biased raw cap
  assert shaped.a_target == pytest.approx(0.0)        # long runway -> cruise (no advisory braking)


def test_speed_limit_runway_governor_degrades_non_negative_accel_to_no_useful_coast_cap():
  # With a non-negative natural coast estimate, no useful coast should be assumed and the
  # governor should not relax the cap via a flat-road -0.25 proxy.
  down_hill = LongitudinalScene(
    v_ego=25.0, v_cruise=25.0, seed_a_target=0.0,
    speed_limit_active=True, speed_limit_v_target=15.0, speed_limit_a_target=-0.5,
    speed_limit_distance=1000.0,
    accel_coast=-0.25,
  )
  flat_no_coast = LongitudinalScene(
    v_ego=25.0, v_cruise=25.0, seed_a_target=0.0,
    speed_limit_active=True, speed_limit_v_target=15.0, speed_limit_a_target=-0.5,
    speed_limit_distance=1000.0, accel_coast=0.0,
  )
  down_cap = [c for c in build_candidates(down_hill) if c.intent == "speed_policy"][0]
  no_coast_cap = [c for c in build_candidates(flat_no_coast) if c.intent == "speed_policy"][0]
  assert down_cap.a_target == pytest.approx(0.0)
  assert no_coast_cap.a_target == pytest.approx(-0.5)


def test_speed_limit_runway_governor_short_runway_stays_braking():
  # Short runway: governor should keep a braking cap (at least as strong as fallback).
  scene = LongitudinalScene(
    v_ego=25.0, v_cruise=25.0, seed_a_target=0.0,
    speed_limit_active=True, speed_limit_v_target=15.0, speed_limit_a_target=-1.0,
    speed_limit_distance=80.0, accel_coast=-0.25,
  )
  cap = [c for c in build_candidates(scene) if c.intent == "speed_policy"][0]
  # 80 m is well inside the 800 m coast distance -> BRAKE, shaped to required decel (~-2.0),
  # clamped to min(0.0, shaped).
  assert cap.a_target < -1.0


def test_curve_runway_governor_shapes_long_runway_cap():
  base = dict(
    v_ego=25.0, v_cruise=25.0, seed_a_target=0.0,
    curve_active=True, curve_a_target=-1.0, curve_source=EvidenceClass.CURVE_VISION,
    accel_coast=-0.25,
  )
  with_dist = LongitudinalScene(curve_v_target=15.0, curve_distance=1000.0, **base)
  without_dist = LongitudinalScene(curve_v_target=0.0, curve_distance=None, **base)
  shaped = [c for c in build_candidates(with_dist) if c.intent == "curve_policy"][0]
  fallback = [c for c in build_candidates(without_dist) if c.intent == "curve_policy"][0]
  assert shaped.a_target > fallback.a_target
  assert shaped.a_target == pytest.approx(0.0)


def test_curve_runway_governor_short_runway_stays_braking():
  scene = LongitudinalScene(
    v_ego=25.0, v_cruise=25.0, seed_a_target=0.0,
    curve_active=True, curve_a_target=-1.0, curve_v_target=15.0,
    curve_distance=80.0, curve_source=EvidenceClass.CURVE_VISION,
    accel_coast=-0.25,
  )
  cap = [c for c in build_candidates(scene) if c.intent == "curve_policy"][0]
  assert cap.a_target < -1.0


def test_curve_runway_governor_ignored_when_target_not_slower():
  # curve_v_target >= v_ego -> governor not applied, raw curve cap preserved.
  scene = LongitudinalScene(
    v_ego=15.0, v_cruise=25.0, seed_a_target=0.0,
    curve_active=True, curve_a_target=-0.5, curve_v_target=20.0,
    curve_distance=1000.0, curve_source=EvidenceClass.CURVE_VISION,
    accel_coast=-0.25,
  )
  cap = [c for c in build_candidates(scene) if c.intent == "curve_policy"][0]
  assert cap.a_target == pytest.approx(-0.5)


def test_speed_limit_runway_governor_raises_decision_above_raw_seed():
  # When the planner seed already came from the speed-limit source, the runway-shaped
  # (coast-first) cap must still raise the final decide() output above the raw seed.
  scene = LongitudinalScene(
    v_ego=25.0, v_cruise=25.0, seed_a_target=-1.0,
    speed_limit_active=True, speed_limit_v_target=15.0, speed_limit_a_target=-1.0,
    speed_limit_distance=1000.0, accel_coast=-0.25,
  )
  d = decide(build_candidates(scene), LongitudinalMode.SCC, LIMITS)
  assert d.a_target > -1.0
  assert d.a_target == pytest.approx(0.0)


def test_curve_runway_governor_raises_decision_above_raw_seed():
  scene = LongitudinalScene(
    v_ego=25.0, v_cruise=25.0, seed_a_target=-1.0,
    curve_active=True, curve_a_target=-1.0, curve_v_target=15.0,
    curve_distance=1000.0, curve_source=EvidenceClass.CURVE_VISION,
    accel_coast=-0.25,
  )
  d = decide(build_candidates(scene), LongitudinalMode.SCC, LIMITS,
             SourceToggles(scc_curve_vision_enabled=True))
  assert d.a_target > -1.0
  assert d.a_target == pytest.approx(0.0)


# --- map-coast tier (coast-only SCC-Map advisory cap) ---

def _map_coast_scene(v_ego=25.0, v_target=15.0, distance=400.0, active=True,
                     accel_coast=0.0, pitch=None):
  # pitch=None + accel_coast=0.0 -> no measured coast, so the cap uses the flat-road proxy (-0.25)
  return LongitudinalScene(v_ego=v_ego, v_cruise=v_ego, seed_a_target=0.0, accel_coast=accel_coast,
                           pitch=pitch, map_coast_active=active, map_coast_v_target=v_target,
                           map_coast_distance=distance)


def test_map_coast_cap_cruises_far_coasts_near_never_brakes():
  # 25 -> 15 m/s at -0.25 coast needs 800 m; lift-off is 800 + 0.6*25 = 815 m.
  assert map_coast_cap(_map_coast_scene(distance=900.0)) is None                       # still cruising
  assert map_coast_cap(_map_coast_scene(distance=810.0)) == pytest.approx(-0.25)      # lift window: coast
  # Well inside coast distance the kinematics ask for real braking (-1.0 here), but map
  # evidence alone never brakes: the cap is floored at the natural coast decel.
  assert map_coast_cap(_map_coast_scene(distance=200.0)) == pytest.approx(-0.25)


def test_map_coast_cap_uses_honest_coast_when_pitch_available():
  scene = _map_coast_scene(distance=810.0, accel_coast=0.02, pitch=-0.05)
  assert map_coast_cap(scene) == pytest.approx(-0.02)


def test_map_coast_cap_invalid_targets_are_none():
  assert map_coast_cap(_map_coast_scene(v_target=0.0)) is None
  assert map_coast_cap(_map_coast_scene(v_target=30.0)) is None       # not slower
  assert map_coast_cap(_map_coast_scene(distance=None)) is None
  assert map_coast_cap(_map_coast_scene(distance=0.0)) is None


def test_map_coast_candidate_only_when_active_and_gated_by_map_toggle():
  active = build_candidates(_map_coast_scene(distance=400.0))
  intents = sources_of(active)
  assert "map_coast" in intents
  assert intents["map_coast"].role is CandidateRole.ADVISORY_CAP
  assert intents["map_coast"].source is EvidenceClass.CURVE_MAP
  assert "map_coast" not in sources_of(build_candidates(_map_coast_scene(active=False)))

  # decide() admits CURVE_MAP only in SCC with the SmartCruiseControlMap toggle on.
  admitted = decide(active, LongitudinalMode.SCC, LIMITS, SourceToggles(scc_curve_map_enabled=True))
  assert admitted.a_target == pytest.approx(-0.25)
  dropped = decide(active, LongitudinalMode.SCC, LIMITS, SourceToggles(scc_curve_map_enabled=False))
  assert dropped.a_target == pytest.approx(0.0)


def test_early_model_slowdown_deepens_with_earned_caution_floor():
  # Route 261: sustained model decel with no in-horizon rest point was pinned at -0.4.
  # With an earned (ramped) floor the early candidate follows the model demand.
  base = dict(v_ego=13.0, v_cruise=15.0, seed_a_target=-0.2,
              model_should_stop=False, model_stop_distance=None, model_desired_accel=-1.6)
  pinned = LongitudinalScene(model_caution_floor=-0.4, **base)
  earned = LongitudinalScene(model_caution_floor=-1.4, **base)
  d_pinned = decide(build_candidates(pinned), LongitudinalMode.SCC, LIMITS)
  d_earned = decide(build_candidates(earned), LongitudinalMode.SCC, LIMITS)
  assert d_pinned.a_target == pytest.approx(-0.4)
  assert d_earned.a_target == pytest.approx(-1.4)


def test_stop_distance_flicker_no_longer_bangs_to_stop_floor():
  # A stop distance that appears while the caution floor is still gentle must not jump
  # the non-committed candidate straight to the -1.5 stop-approach floor.
  scene = LongitudinalScene(v_ego=12.0, v_cruise=15.0, seed_a_target=-0.2,
                            model_should_stop=False, model_stop_distance=40.0,
                            model_desired_accel=-1.6, model_caution_floor=-0.45)
  d = decide(build_candidates(scene), LongitudinalMode.SCC, LIMITS)
  assert d.a_target >= -0.45


def test_hard_trusted_stop_commit_bypasses_caution_floor():
  scene = LongitudinalScene(v_ego=15.0, v_cruise=15.0, seed_a_target=0.0,
                            model_should_stop=True, model_stop_distance=20.0,
                            model_desired_accel=-2.5, model_stop_prob=0.9,
                            model_caution_floor=-0.4)
  d = decide(build_candidates(scene), LongitudinalMode.SCC, LIMITS)
  assert d.a_target < -1.5
  assert d.should_stop
