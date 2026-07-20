"""Tests for the longitudinal model-evidence trust gate."""
from __future__ import annotations

import pytest

from openpilot.sunnypilot.custom.longitudinal.model_trust import (
  GENTLE_CAUTION_DECEL,
  STOP_TRUST_MAX,
  STOP_TRUST_MIN,
  TRUST_FULL_STOP,
  StopTrustLearner,
  gate_model_stop,
)


def test_model_clear_passes_through():
  r = gate_model_stop(model_should_stop=False, model_desired_accel=0.3, stop_prob=0.1)
  assert r.should_stop is False
  assert r.desired_accel == 0.3
  assert r.trust == 1.0


def test_high_confidence_stop_honored_at_full_decel():
  r = gate_model_stop(model_should_stop=True, model_desired_accel=-3.0, stop_prob=0.95)
  assert r.should_stop is True
  assert r.desired_accel == pytest.approx(GENTLE_CAUTION_DECEL + 0.95 * (-3.0 - GENTLE_CAUTION_DECEL))
  assert r.desired_accel < -2.0  # near the full model decel


def test_low_confidence_stop_softened_and_not_committed():
  r = gate_model_stop(model_should_stop=True, model_desired_accel=-3.0, stop_prob=0.2)
  assert r.should_stop is False                 # not committed on a flickery stop
  assert r.desired_accel > -1.5                 # softened toward gentle caution
  assert r.desired_accel <= 0.0


def test_stale_model_stop_only_allows_gentle_caution():
  r = gate_model_stop(model_should_stop=True, model_desired_accel=-3.0, stop_prob=0.95,
                      model_stale=True)
  assert r.should_stop is False
  assert r.desired_accel == pytest.approx(GENTLE_CAUTION_DECEL)
  assert r.trust == 0.0
  assert r.reason == "model_stale"


def test_radar_corroboration_raises_trust():
  # a closing radar lead corroborates the slowdown -> higher trust than model_prob alone
  weak = gate_model_stop(True, -2.5, stop_prob=0.3, has_radar_lead=False, lead_v_rel=0.0)
  corrob = gate_model_stop(True, -2.5, stop_prob=0.3, has_radar_lead=True, lead_v_rel=-0.3)
  assert corrob.trust > weak.trust
  assert corrob.desired_accel < weak.desired_accel  # more braking authority when corroborated
  assert corrob.reason == "radar_corroborated"


def test_tiny_radar_closing_does_not_corroborate_stop():
  r = gate_model_stop(True, -2.5, stop_prob=0.3, has_radar_lead=True, lead_v_rel=-0.1)
  assert r.trust == pytest.approx(0.3)
  assert r.reason == "model_only"


def test_trust_monotonic_in_stop_prob():
  accels = [gate_model_stop(True, -3.0, stop_prob=p).desired_accel for p in (0.0, 0.3, 0.6, 0.9)]
  for a, b in zip(accels, accels[1:], strict=False):
    assert b <= a + 1e-9  # higher confidence -> more (not less) braking


def test_full_stop_threshold():
  assert gate_model_stop(True, -2.0, stop_prob=TRUST_FULL_STOP - 0.01).should_stop is False
  assert gate_model_stop(True, -2.0, stop_prob=TRUST_FULL_STOP + 0.01).should_stop is True


def test_caution_never_below_gentle_floor_at_zero_trust():
  r = gate_model_stop(True, -5.0, stop_prob=0.0)
  assert r.desired_accel == pytest.approx(GENTLE_CAUTION_DECEL)


def test_stop_trust_learner_drops_on_driver_disagreement():
  learner = StopTrustLearner(initial=0.8)
  c = 0.8
  for _ in range(20):
    c = learner.update(model_should_stop=True, driver_disagrees=True, dt=0.05)
  assert c < 0.8


def test_stop_trust_learner_recovers_on_agreement():
  learner = StopTrustLearner(initial=0.4)
  c = 0.4
  for _ in range(100):
    c = learner.update(model_should_stop=True, driver_disagrees=False, dt=0.05)
  assert c > 0.4


def test_stop_trust_learner_idle_without_model_stop():
  learner = StopTrustLearner(initial=0.7)
  assert learner.update(model_should_stop=False, driver_disagrees=True, dt=0.05) == 0.7


def test_stop_trust_learner_stays_bounded():
  learner = StopTrustLearner(initial=0.8)
  for _ in range(2000):
    learner.update(True, True, 0.05)
  assert learner.confidence >= STOP_TRUST_MIN
  for _ in range(5000):
    learner.update(True, False, 0.05)
  assert learner.confidence <= STOP_TRUST_MAX


def test_caution_ramp_deepens_slowly_toward_sustained_demand():
  from openpilot.sunnypilot.custom.longitudinal.model_trust import CAUTION_RAMP_DEEPEN_RATE, CautionRamp
  ramp = CautionRamp()
  # 1 s of sustained -2.0 demand deepens by exactly the deepen rate, not to the demand.
  floor = ramp.floor
  for _ in range(20):
    floor = ramp.update(-2.0, 0.05)
  assert floor == pytest.approx(GENTLE_CAUTION_DECEL - CAUTION_RAMP_DEEPEN_RATE, abs=1e-6)
  # After ~4 s it tracks the demand.
  for _ in range(80):
    floor = ramp.update(-2.0, 0.05)
  assert floor == pytest.approx(-2.0)


def test_caution_ramp_flicker_stays_gentle_and_releases_fast():
  from openpilot.sunnypilot.custom.longitudinal.model_trust import CautionRamp
  ramp = CautionRamp()
  # A 0.2 s flicker of deep demand barely moves the floor...
  floor = ramp.floor
  for _ in range(4):
    floor = ramp.update(-2.0, 0.05)
  assert floor > -0.5
  # ...and releases back to gentle well under a second once the demand lifts.
  for _ in range(6):
    floor = ramp.update(0.0, 0.05)
  assert floor == pytest.approx(GENTLE_CAUTION_DECEL)


def test_corroboration_hold_latches_across_radar_flicker_and_expires():
  from openpilot.sunnypilot.custom.longitudinal.model_trust import CORROBORATION_HOLD_S, CorroborationHold
  hold = CorroborationHold()
  assert hold.update(False, 0.05) is False
  # One closing echo latches; the latch survives the longest observed 28c flicker gap (1.6 s).
  assert hold.update(True, 0.05) is True
  held = False
  for _ in range(32):
    held = hold.update(False, 0.05)
  assert held is True
  # Without a fresh echo it expires after the hold window.
  for _ in range(int(CORROBORATION_HOLD_S / 0.05)):
    held = hold.update(False, 0.05)
  assert held is False


def test_caution_ramp_never_deeper_than_clamp_or_shallower_than_gentle():
  from openpilot.sunnypilot.custom.longitudinal.model_trust import CAUTION_RAMP_FLOOR_MIN, CautionRamp
  ramp = CautionRamp()
  for _ in range(1000):
    ramp.update(-10.0, 0.05)
  assert ramp.floor == CAUTION_RAMP_FLOOR_MIN
  for _ in range(100):
    ramp.update(5.0, 0.05)
  assert ramp.floor == GENTLE_CAUTION_DECEL


def _lead(d_rel: float, y_rel: float, v_rel: float, status: bool = True):
  from types import SimpleNamespace
  return SimpleNamespace(status=status, dRel=d_rel, yRel=y_rel, vRel=v_rel)


def test_cut_out_recovery_arms_on_lateral_exit_of_closing_lead_and_expires():
  from openpilot.sunnypilot.custom.longitudinal.model_trust import CUT_OUT_RECOVERY_S, CutOutCautionRecovery
  rec = CutOutCautionRecovery()
  # Route 296 shape: closing turner drifts out laterally over the last second, then vanishes.
  for y in (0.2, 0.5, 0.9, 1.4, 1.8):
    assert rec.update(_lead(62.0, y, -1.4), None, 0.05) is False
  assert rec.update(None, None, 0.05) is True          # departure with exit evidence
  ticks = 0
  while rec.update(None, None, 0.05):
    ticks += 1
  assert ticks == pytest.approx(CUT_OUT_RECOVERY_S / 0.05, abs=2)


def test_cut_out_recovery_ignores_straight_ahead_flicker():
  from openpilot.sunnypilot.custom.longitudinal.model_trust import CutOutCautionRecovery
  rec = CutOutCautionRecovery()
  for _ in range(20):
    rec.update(_lead(40.0, 0.2, -2.0), None, 0.05)
  assert rec.update(None, None, 0.05) is False          # in-path flicker: hold caution


def test_cut_out_recovery_requires_closing_context_and_distance():
  from openpilot.sunnypilot.custom.longitudinal.model_trust import CutOutCautionRecovery
  rec = CutOutCautionRecovery()
  for _ in range(20):
    rec.update(_lead(62.0, 1.8, +0.5), None, 0.05)      # opening lead
  assert rec.update(None, None, 0.05) is False
  rec = CutOutCautionRecovery()
  for _ in range(20):
    rec.update(_lead(8.0, 1.8, -1.0), None, 0.05)       # under the radar nose at a stop
  assert rec.update(None, None, 0.05) is False


def test_cut_out_recovery_cancelled_by_reappearing_closing_lead():
  from openpilot.sunnypilot.custom.longitudinal.model_trust import CutOutCautionRecovery
  rec = CutOutCautionRecovery()
  for y in (0.9, 1.4, 1.8):
    rec.update(_lead(62.0, y, -1.4), None, 0.05)
  assert rec.update(None, None, 0.05) is True
  assert rec.update(_lead(45.0, 0.0, -2.0), None, 0.05) is False  # new closing threat: caution back


def test_model_stop_anchor_ratchets_conservative_and_advances_with_travel():
  from openpilot.sunnypilot.custom.longitudinal.model_trust import ModelStopAnchor
  a = ModelStopAnchor()
  d = a.update(150.0, v_ego=15.0, dt=0.05)
  assert d == pytest.approx(138.0)  # max(0.85*150, 150-12)
  # model holds 150 while ego travels: the anchor holds its commitment, advancing with travel
  d = a.update(150.0, v_ego=15.0, dt=0.05)
  assert d == pytest.approx(138.0 - 0.75)
  # model firms nearer within the jump guard: accepted immediately
  d = a.update(145.0, v_ego=15.0, dt=0.05)
  assert d == pytest.approx(145.0 - 12.0)


def test_model_stop_anchor_jumps_need_corroboration_in_both_directions():
  from openpilot.sunnypilot.custom.longitudinal.model_trust import (
    ModelStopAnchor, STOP_ANCHOR_JUMP_CONFIRM_FRAMES, STOP_ANCHOR_MAX_DIVERGENCE_M)
  a = ModelStopAnchor()
  a.update(100.0, v_ego=10.0, dt=0.05)
  base = a.remaining
  # one bad frame claiming 30 m: ignored (advances with travel only)
  d = a.update(30.0, v_ego=10.0, dt=0.05)
  assert d == pytest.approx(base - 0.5)
  # sustained near readings: accepted after the corroboration window
  for _ in range(STOP_ANCHOR_JUMP_CONFIRM_FRAMES):
    d = a.update(30.0, v_ego=10.0, dt=0.05)
  assert d == pytest.approx(25.0)  # accepted at 25.5, then one frame of travel advance
  # one far frame (90 m): held; sustained far readings re-open to the divergence bound
  held = a.remaining
  d = a.update(90.0, v_ego=0.0, dt=0.05)
  assert d == pytest.approx(held)
  for _ in range(STOP_ANCHOR_JUMP_CONFIRM_FRAMES):
    d = a.update(90.0, v_ego=0.0, dt=0.05)
  assert d == pytest.approx(max(0.85 * 90.0, 78.0) - STOP_ANCHOR_MAX_DIVERGENCE_M)


def test_model_stop_anchor_bounds_divergence_on_receding_stop_point():
  # Phantom shape: reported distance never shrinks while ego travels. The anchor must
  # follow with bounded frontload, never invert the recession into an in-rushing stop.
  from openpilot.sunnypilot.custom.longitudinal.model_trust import (
    ModelStopAnchor, STOP_ANCHOR_MAX_DIVERGENCE_M)
  a = ModelStopAnchor()
  target = max(0.85 * 40.0, 28.0)
  for _ in range(100):  # 5 s at 12 m/s: 60 m of travel against a static 40 m claim
    d = a.update(40.0, v_ego=12.0, dt=0.05)
  assert d >= target - STOP_ANCHOR_MAX_DIVERGENCE_M - 1.0
  assert d <= target


def test_model_stop_anchor_never_releases_by_burn_down_while_moving():
  # Route 2ba t=1517/1623: the commitment burned to 0 with travel while the model still
  # placed the stop ahead, dropping the whole stop posture at 4.5 m/s. While ego moves and
  # the raw point is live, the anchored distance must stay positive — never "arrived".
  from openpilot.sunnypilot.custom.longitudinal.model_trust import (
    ModelStopAnchor, STOP_ANCHOR_MIN_ACTIVE_M)
  a = ModelStopAnchor()
  a.update(45.0, v_ego=14.0, dt=0.05)
  d = a.remaining
  for raw in (40.0, 34.0, 28.0, 22.0, 16.0, 11.0, 8.0, 6.6, 6.6, 6.6, 6.6):
    for _ in range(10):  # 0.5 s per raw reading, decelerating ego
      d = a.update(raw, v_ego=8.0, dt=0.05)
      assert d is not None and d >= STOP_ANCHOR_MIN_ACTIVE_M
  # brief raw dropouts near the stop hold the same floor
  for _ in range(10):
    d = a.update(None, v_ego=4.0, dt=0.05)
    assert d is not None and d >= STOP_ANCHOR_MIN_ACTIVE_M
  # once ego is (nearly) stopped the floor no longer applies and retraction releases
  for _ in range(25):
    d = a.update(None, v_ego=0.0, dt=0.05)
  assert d is None


def test_model_stop_anchor_divergence_tightens_near_stop():
  # Far out the divergence bound is the fixed 15 m; near the stop it is proportional, so
  # the commitment can never sit at a tiny fraction of a live target a few meters ahead.
  from openpilot.sunnypilot.custom.longitudinal.model_trust import (
    ModelStopAnchor, STOP_ANCHOR_DIVERGENCE_FRACTION, STOP_ANCHOR_CONSERVATIVE_FRACTION)
  a = ModelStopAnchor()
  a.update(12.0, v_ego=6.0, dt=0.05)
  for _ in range(100):  # 30 m of travel against a static 12 m claim
    d = a.update(12.0, v_ego=6.0, dt=0.05)
  target = 12.0 * STOP_ANCHOR_CONSERVATIVE_FRACTION
  assert d >= target * (1.0 - STOP_ANCHOR_DIVERGENCE_FRACTION) - 1e-6


def test_model_stop_anchor_travel_consistency_corroboration():
  from openpilot.sunnypilot.custom.longitudinal.model_trust import ModelStopAnchor
  # real stop: raw distance shrinks with travel -> corroborated once enough travel is seen
  a = ModelStopAnchor()
  d = 60.0
  for _ in range(30):  # 15 m of travel, raw tracking it 1:1
    a.update(d, v_ego=10.0, dt=0.05)
    d -= 0.5
  assert a.corroborated is True
  # phantom: raw distance static while ego travels -> never corroborated
  p = ModelStopAnchor()
  for _ in range(100):  # 50 m of travel against a static claim
    p.update(40.0, v_ego=10.0, dt=0.05)
  assert p.corroborated is False


def test_model_stop_anchor_confirmed_jump_rebases_corroboration():
  from openpilot.sunnypilot.custom.longitudinal.model_trust import (
    ModelStopAnchor, STOP_ANCHOR_JUMP_CONFIRM_FRAMES)
  a = ModelStopAnchor()
  d = 60.0
  for _ in range(30):
    a.update(d, v_ego=10.0, dt=0.05)
    d -= 0.5
  assert a.corroborated is True
  for _ in range(STOP_ANCHOR_JUMP_CONFIRM_FRAMES + 1):  # scene change: point jumps far
    a.update(120.0, v_ego=10.0, dt=0.05)
  assert a.corroborated is False  # consistency re-earns against the new point


def test_model_stop_anchor_releases_after_sustained_retraction_only():
  from openpilot.sunnypilot.custom.longitudinal.model_trust import ModelStopAnchor
  a = ModelStopAnchor()
  a.update(80.0, v_ego=10.0, dt=0.05)
  # brief dropout: commitment holds, advancing with travel
  d = a.update(None, v_ego=10.0, dt=0.05)
  assert d is not None
  # sustained retraction (green light): released
  for _ in range(25):
    d = a.update(None, v_ego=10.0, dt=0.05)
  assert d is None and a.remaining is None


def test_model_stop_anchor_commit_age_accrues_for_blip_filtering():
  from openpilot.sunnypilot.custom.longitudinal.model_trust import ModelStopAnchor, STOP_ANCHOR_MIN_COMMIT_S
  a = ModelStopAnchor()
  a.update(80.0, v_ego=10.0, dt=0.05)
  assert a.committed_s < STOP_ANCHOR_MIN_COMMIT_S  # a single-frame phantom stays below the debounce
  for _ in range(6):
    a.update(80.0, v_ego=10.0, dt=0.05)
  assert a.committed_s >= STOP_ANCHOR_MIN_COMMIT_S
  a.reset()
  assert a.committed_s == 0.0
