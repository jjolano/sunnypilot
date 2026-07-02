"""Tests for the bidirectional lead-speed alignment helper (Phase 1)."""
from __future__ import annotations

import math

import pytest

from openpilot.sunnypilot.custom.longitudinal.lead_speed_alignment import (
  AlignmentAction,
  LeadSpeedAlignment,
  lead_speed_alignment,
)
from openpilot.sunnypilot.custom.longitudinal.policy_tables import (
  Personality,
  launch_accel_max,
)


def align(
  *,
  v_ego: float = 20.0,
  a_ego: float = 0.0,
  v_cruise: float = 25.0,
  lead_d_rel: float = 100.0,
  lead_v: float = 20.0,
  lead_v_rel: float = 0.0,
  lead_a_k: float = 0.0,
  follow_gap: float = 30.0,
  lead_confidence: float = 0.9,
  lead_stable: bool = True,
  lead_progress_allowed: bool = True,
  lead_shadow_active: bool = False,
  alternate_threat_active: bool = False,
  model_should_stop: bool = False,
  force_slow_decel: bool = False,
  brake_pressed: bool = False,
  gas_pressed: bool = False,
  personality: Personality = Personality.STANDARD,
  lead_kinematics_valid: bool = True,
  has_lead: bool = True,
  a_lead_tau: float = 1.5,
) -> LeadSpeedAlignment:
  return lead_speed_alignment(
    v_ego=v_ego, a_ego=a_ego, v_cruise=v_cruise,
    lead_d_rel=lead_d_rel, lead_v=lead_v, lead_v_rel=lead_v_rel, lead_a_k=lead_a_k,
    follow_gap=follow_gap, lead_confidence=lead_confidence, lead_stable=lead_stable,
    lead_progress_allowed=lead_progress_allowed, lead_shadow_active=lead_shadow_active,
    alternate_threat_active=alternate_threat_active, model_should_stop=model_should_stop,
    force_slow_decel=force_slow_decel, brake_pressed=brake_pressed, gas_pressed=gas_pressed,
    personality=personality, lead_kinematics_valid=lead_kinematics_valid, has_lead=has_lead,
    a_lead_tau=a_lead_tau,
  )


def test_far_slower_lead_with_tiny_required_decel_coasts():
  # 25 m/s ego, 24 m/s lead -> 1 m/s closing at 80 m -> tiny required decel.
  # TTC = 80/1 = 80s > NO_ADVISORY_TTC (24s) and THW >= 1.5 -> IGNORE (no block).
  r = align(v_ego=25.0, lead_v=24.0, lead_v_rel=-1.0, lead_d_rel=80.0, follow_gap=37.5)
  assert r.action in (AlignmentAction.COAST, AlignmentAction.IGNORE)
  assert r.a_target <= 0.0 + 1e-9
  assert r.required_decel < 0.10
  assert r.reason in ("tiny_decel_coast", "ttc_far_comfort")


def test_medium_slower_lead_produces_gentle_brake():
  # 25 m/s ego, 23 m/s lead -> 2 m/s closing at 45 m -> reqDecel ~0.27 in comfort band.
  # TTC = 45/2 = 22.5s < NO_ADVISORY_TTC(24) so enters decel routing.
  r = align(v_ego=25.0, lead_v=23.0, lead_v_rel=-2.0, lead_d_rel=45.0, follow_gap=37.5)
  assert r.action is AlignmentAction.GENTLE_BRAKE
  assert r.a_target <= 0.0 + 1e-9
  assert r.a_target >= -0.35 - 1e-9
  assert r.required_decel > 0.0


def test_close_risky_lead_with_high_speed_light_brake():
  # Hard closing at highway speed: 30 m/s ego, 8 m/s closing at 55 m.
  # TTC = 55/8 = 6.9s. vEgo >= 18, TTC >= 6, THW = 55/30 = 1.83 >= 1.2
  # → high-speed light-brake zone: gentle brake, not MPC wall.
  r = align(v_ego=30.0, lead_v=22.0, lead_v_rel=-8.0, lead_d_rel=55.0, follow_gap=15.0)
  assert r.action is AlignmentAction.GENTLE_BRAKE
  assert r.required_decel > 0.50
  assert r.reason == "high_decel_high_speed_mild"
  assert r.a_target == pytest.approx(-0.35, abs=1e-9)


def test_close_risky_lead_high_speed_below_strong_prep_ttc_stays_hazard():
  # Highway high-decel cases only get capped advisory when TTC >= 6s. Below that,
  # even with comfortable headway, MPC/lead physics owns the response.
  r = align(v_ego=30.0, lead_v=21.5, lead_v_rel=-8.5, lead_d_rel=50.0, follow_gap=15.0)
  assert r.action is AlignmentAction.IGNORE
  assert r.required_decel > 0.80
  assert r.reason == "high_required_decel"


def test_close_risky_lead_low_speed_stays_hazard():
  # Low-speed hard closing: 12 m/s ego, 6 m/s closing at 25 m.
  # TTC = 25/6 = 4.2s (> 4, so clears hazard gate).
  # required_decel >> MAX, but v_ego=12 < 18, TTC=4.2 < 8 → defer to MPC.
  r = align(v_ego=12.0, lead_v=6.0, lead_v_rel=-6.0, lead_d_rel=25.0, follow_gap=15.0)
  assert r.action is AlignmentAction.IGNORE
  assert r.reason == "high_required_decel"


def test_close_risky_lead_mid_speed_ttc_alone_stays_hazard():
  # Real-log-shaped mid-speed case: TTC looks mild (31/3.5 = 8.9s), but the usable
  # gap above desired following distance is consumed in ~1.6s, so TTC alone must
  # not authorize a capped gentle advisory.
  r = align(v_ego=17.0, lead_v=13.5, lead_v_rel=-3.5, lead_d_rel=31.0, follow_gap=25.5)
  assert r.action is AlignmentAction.IGNORE
  assert r.required_decel > 0.80
  assert r.reason == "high_required_decel"


def test_close_risky_lead_low_speed_ttc_alone_stays_hazard():
  # Low-speed extrapolated case outside the highway analysis: TTC >= 8 is still
  # not enough to replace MPC authority when required_decel is above MAX.
  r = align(v_ego=12.0, lead_v=9.5, lead_v_rel=-2.5, lead_d_rel=21.0, follow_gap=15.0)
  assert r.action is AlignmentAction.IGNORE
  assert r.required_decel > 0.80
  assert r.reason == "high_required_decel"


def test_unstable_lead_does_not_pullaway():
  r = align(
    v_ego=0.0, lead_v=2.0, lead_v_rel=1.5, lead_d_rel=10.0,
    follow_gap=6.0, lead_stable=False, lead_progress_allowed=True,
  )
  assert r.action is AlignmentAction.IGNORE


def test_flickery_new_lead_does_not_pullaway():
  r = align(
    v_ego=0.0, lead_v=2.0, lead_v_rel=1.5, lead_d_rel=10.0,
    follow_gap=6.0, lead_stable=False, lead_confidence=0.3, lead_progress_allowed=True,
  )
  assert r.action is AlignmentAction.IGNORE


def test_stable_opening_moving_lead_produces_guarded_pullaway():
  r = align(
    v_ego=5.0, lead_v=8.0, lead_v_rel=3.0, lead_d_rel=45.0,
    follow_gap=20.0, lead_stable=True, lead_progress_allowed=True,
  )
  assert r.action is AlignmentAction.PULLAWAY
  assert 0.0 < r.a_target <= launch_accel_max(Personality.STANDARD) + 1e-9


def test_tight_gap_blocks_pullaway():
  r = align(
    v_ego=5.0, lead_v=8.0, lead_v_rel=3.0, lead_d_rel=21.0,
    follow_gap=20.0, lead_stable=True, lead_progress_allowed=True,
  )
  # d_rel is only 1 m beyond follow_gap -> blocked by near-follow margin.
  assert r.action is not AlignmentAction.PULLAWAY


def test_standstill_stable_moving_lead_produces_standstill_launch():
  r = align(
    v_ego=0.0, lead_v=1.5, lead_v_rel=1.5, lead_d_rel=8.0,
    follow_gap=6.0, lead_stable=True, lead_progress_allowed=True,
  )
  assert r.action is AlignmentAction.STANDSTILL_LAUNCH
  assert 0.0 < r.a_target <= launch_accel_max(Personality.STANDARD) + 1e-9


def test_standstill_launch_is_guarded_by_speedup_guard():
  # Very tight standstill gap: guard should block positive accel.
  r = align(
    v_ego=0.0, lead_v=1.5, lead_v_rel=1.5, lead_d_rel=6.5,
    follow_gap=6.0, lead_stable=True, lead_progress_allowed=True,
  )
  # excess gap is 0.5 m; guard may still allow a small accel, but it must be safe.
  if r.action is AlignmentAction.STANDSTILL_LAUNCH:
    assert 0.0 < r.a_target <= launch_accel_max(Personality.STANDARD)
  else:
    assert r.action is AlignmentAction.IGNORE


def test_model_stop_blocks_launch_and_pullaway():
  for v_ego in (0.0, 5.0):
    r = align(
      v_ego=v_ego, lead_v=3.0, lead_v_rel=2.0, lead_d_rel=20.0,
      follow_gap=10.0, lead_stable=True, lead_progress_allowed=True,
      model_should_stop=True,
    )
    assert r.action is AlignmentAction.IGNORE
    assert r.reason == "model_stop"


def test_alternate_threat_blocks_launch_and_pullaway():
  for v_ego in (0.0, 5.0):
    r = align(
      v_ego=v_ego, lead_v=3.0, lead_v_rel=2.0, lead_d_rel=20.0,
      follow_gap=10.0, lead_stable=True, lead_progress_allowed=True,
      alternate_threat_active=True,
    )
    assert r.action is AlignmentAction.IGNORE
    assert r.reason == "threat"


def test_shadow_threat_blocks_launch_and_pullaway():
  r = align(
    v_ego=0.0, lead_v=3.0, lead_v_rel=2.0, lead_d_rel=20.0,
    follow_gap=10.0, lead_stable=True, lead_progress_allowed=True,
    lead_shadow_active=True,
  )
  assert r.action is AlignmentAction.IGNORE
  assert r.reason == "threat"


def test_driver_override_blocks_reaction():
  for pressed in ("brake", "gas"):
    r = align(
      v_ego=25.0, lead_v=20.0, lead_v_rel=-5.0, lead_d_rel=40.0,
      brake_pressed=(pressed == "brake"), gas_pressed=(pressed == "gas"),
    )
    assert r.action is AlignmentAction.IGNORE
    assert r.reason == "driver_override"


def test_force_slow_blocks_reaction():
  r = align(
    v_ego=25.0, lead_v=20.0, lead_v_rel=-5.0, lead_d_rel=40.0,
    force_slow_decel=True,
  )
  assert r.action is AlignmentAction.IGNORE
  assert r.reason == "force_slow"


def test_invalid_kinematics_fail_closed():
  r = align(v_ego=25.0, lead_v_rel=float("nan"), lead_kinematics_valid=True)
  assert r.action is AlignmentAction.IGNORE
  assert r.reason == "nonfinite"


def test_no_lead_returns_ignore():
  r = align(has_lead=False)
  assert r.action is AlignmentAction.IGNORE
  assert r.reason == "no_lead"


def test_low_confidence_far_lead_ignored():
  r = align(
    v_ego=25.0, lead_v=22.0, lead_v_rel=-3.0, lead_d_rel=80.0,
    follow_gap=37.5, lead_confidence=0.3,
  )
  assert r.action is AlignmentAction.IGNORE


def test_high_confidence_unstable_far_slowing_lead_ignored():
  r = align(
    v_ego=25.0, lead_v=24.0, lead_v_rel=-1.0, lead_d_rel=80.0,
    follow_gap=37.5, lead_confidence=0.95, lead_stable=False,
  )
  assert r.action is AlignmentAction.IGNORE


def test_crawling_lead_does_not_trigger_slowdown():
  r = align(v_ego=25.0, lead_v=2.0, lead_v_rel=-5.0, lead_d_rel=80.0)
  assert r.action is AlignmentAction.IGNORE


def test_output_fields_are_finite():
  r = align(v_ego=25.0, lead_v=24.0, lead_v_rel=-1.0, lead_d_rel=80.0)
  assert all(math.isfinite(v) for v in (
    r.a_target, r.required_decel, r.desired_gap, r.excess_gap, r.closing,
  ))


def test_personality_scaling_in_launch():
  base = dict(
    v_ego=0.0, lead_v=3.0, lead_v_rel=3.0, lead_d_rel=15.0,
    follow_gap=10.0, lead_stable=True, lead_progress_allowed=True,
  )
  rel = align(**base, personality=Personality.RELAXED)
  agg = align(**base, personality=Personality.AGGRESSIVE)
  assert rel.action is AlignmentAction.STANDSTILL_LAUNCH
  assert agg.action is AlignmentAction.STANDSTILL_LAUNCH
  assert agg.a_target >= rel.a_target


# -----------------------------------------------------------------------------
# Predictive early, gentle lead reaction (Phase 5)
# -----------------------------------------------------------------------------

def test_predicted_compression_emits_early_gentle_brake():
  # Current-state demand is tiny (closing 0.5 m/s) but a stable, braking lead will
  # compress the gap over the next 1-1.5 s. Without prediction this returns IGNORE.
  r = align(
    v_ego=25.0, lead_v=24.5, lead_v_rel=-0.5, lead_a_k=-2.5,
    lead_d_rel=50.0, follow_gap=37.5, lead_progress_allowed=False,
    lead_stable=True, lead_confidence=0.9, a_lead_tau=1.5,
  )
  assert r.action is AlignmentAction.GENTLE_BRAKE
  assert -0.35 <= r.a_target <= 0.0
  assert "predicted" in r.reason


def test_no_false_advisory_when_prediction_does_not_compress():
  # Same comfortable initial state but the lead is not braking, so the predicted
  # trajectory does not compress the gap -> remain advisory-silent.
  r = align(
    v_ego=25.0, lead_v=24.5, lead_v_rel=-0.5, lead_a_k=0.0,
    lead_d_rel=50.0, follow_gap=37.5, lead_progress_allowed=False,
    lead_stable=True, lead_confidence=0.9, a_lead_tau=1.5,
  )
  assert r.action is AlignmentAction.IGNORE
  assert "predicted" not in r.reason


def test_predicted_early_coast_when_mild_compression():
  # Very mild predicted compression only lifts the routing into the tiny-decel band,
  # producing an early coast rather than waiting.
  r = align(
    v_ego=25.0, lead_v=24.8, lead_v_rel=-0.2, lead_a_k=-2.0,
    lead_d_rel=60.0, follow_gap=37.5, lead_progress_allowed=False,
    lead_stable=True, lead_confidence=0.9, a_lead_tau=1.5,
  )
  # TTC is far (60/0.2 = 300 s) so without prediction this would IGNORE; with a
  # mild predicted compression we should at least see COAST.
  assert r.action in (AlignmentAction.COAST, AlignmentAction.GENTLE_BRAKE)
  assert r.a_target <= 0.0 + 1e-9


def test_prediction_respects_fail_closed_gates():
  # Even with strong predicted compression, the usual safety gates still win.
  for kwargs in (
    {"lead_shadow_active": True},
    {"alternate_threat_active": True},
    {"model_should_stop": True},
    {"force_slow_decel": True},
    {"brake_pressed": True},
    {"gas_pressed": True},
    {"lead_kinematics_valid": False},
  ):
    r = align(
      v_ego=25.0, lead_v=24.5, lead_v_rel=-0.5, lead_a_k=-2.5,
      lead_d_rel=50.0, follow_gap=37.5, lead_progress_allowed=False,
      lead_stable=True, lead_confidence=0.9, a_lead_tau=1.5, **kwargs,
    )
    assert r.action is AlignmentAction.IGNORE
