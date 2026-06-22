"""Integration test for the CustomLongitudinalStack composition (fakes for car evidence)."""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from openpilot.sunnypilot.custom.longitudinal.modes import LongitudinalMode, SourceToggles
from openpilot.sunnypilot.custom.longitudinal.policy_tables import Personality
from openpilot.sunnypilot.custom.longitudinal.stack import (
  CustomLongitudinalStack,
  LongitudinalStackInputs,
  LongitudinalStackResult,
)

DT = 0.05
LIMITS = (-4.0, 2.0)


def lead(d_rel=30.0, v_lead=12.0, v_rel=0.0, y_rel=0.0, status=True, track_id=3):
  return SimpleNamespace(status=status, dRel=d_rel, vLead=v_lead, vLeadK=v_lead, vRel=v_rel,
                         aLeadK=0.0, yRel=y_rel, radarTrackId=track_id, radar=True,
                         modelProb=0.9, aLeadTau=1.0)


def model_path(xs=(0.0, 30.0, 60.0), ys=(0.0, 0.5, 1.0)):
  return SimpleNamespace(position=SimpleNamespace(x=list(xs), y=list(ys)))


class RaisesOnY:
  x = [0.0, 30.0, 60.0]

  @property
  def y(self):
    raise RuntimeError("bad model path")


def malformed_model_path():
  return SimpleNamespace(position=RaisesOnY())


class BadLeadPathClearanceMode:
  def __str__(self):
    raise RuntimeError("bad clearance mode")


def base(**kw):
  d = dict(v_ego=20.0, v_cruise=22.0, seed_a_target=0.4, accel_limits=LIMITS)
  d.update(kw)
  return LongitudinalStackInputs(**d)


def test_runs_bounded_over_sequence():
  s = CustomLongitudinalStack()
  rng = np.random.default_rng(20260613)
  for _ in range(800):
    r = s.update(base(
      v_ego=float(rng.uniform(0, 30)), seed_a_target=float(rng.uniform(-2, 1.5)),
      leads=(lead(d_rel=float(rng.uniform(8, 60))) if rng.random() > 0.4 else None, None),
      lead_a_target=float(rng.uniform(-3, 0.5)),
      mode=[LongitudinalMode.ACC, LongitudinalMode.E2E, LongitudinalMode.SCC][int(rng.integers(0, 3))],
    ), DT)
    assert isinstance(r, LongitudinalStackResult)
    assert math.isfinite(r.a_target)
    assert LIMITS[0] - 1e-9 <= r.a_target <= LIMITS[1] + 1e-9


def test_acc_cruises_when_clear():
  s = CustomLongitudinalStack()
  r = None
  for _ in range(5):
    r = s.update(base(seed_a_target=0.4, mode=LongitudinalMode.ACC), DT)
  assert r.a_target == 0.4
  assert r.should_stop is False
  assert r.debug["has_lead"] is False


def test_standstill_release_fields_for_no_lead_launch():
  s = CustomLongitudinalStack()
  r = s.update(base(
    v_ego=0.0, v_cruise=12.0, seed_a_target=0.2, leads=(None, None), mode=LongitudinalMode.E2E,
    model_should_stop=False, model_stop_distance=None, model_desired_accel=0.2,
  ), DT)
  assert r.standstill_release_allowed is True
  assert r.standstill_release_source == "no_lead_launch"
  assert r.standstill_release_a_target >= 0.15


def test_no_lead_release_requires_runway_confirmed_clear_path():
  s = CustomLongitudinalStack()
  model_stop = s.update(base(
    v_ego=0.0, v_cruise=12.0, seed_a_target=0.2, leads=(None, None), mode=LongitudinalMode.E2E,
    model_should_stop=True, model_stop_distance=8.0, model_desired_accel=-1.0,
  ), DT)
  assert model_stop.standstill_release_allowed is False

  near_stop = s.update(base(
    v_ego=0.0, v_cruise=12.0, seed_a_target=0.2, leads=(None, None), mode=LongitudinalMode.E2E,
    model_should_stop=False, model_stop_distance=8.0, model_desired_accel=0.0,
  ), DT)
  assert near_stop.standstill_release_allowed is False


def test_no_lead_release_blocked_by_shadow_lead():
  s = CustomLongitudinalStack()
  for _ in range(12):
    s.update(base(v_ego=0.0, v_cruise=12.0, seed_a_target=0.2,
                  leads=(lead(d_rel=6.5, v_lead=0.0), None), mode=LongitudinalMode.ACC), DT)
  lost = s.update(base(v_ego=0.0, v_cruise=12.0, seed_a_target=0.2,
                       leads=(None, None), mode=LongitudinalMode.ACC), DT)
  assert lost.debug["lead_shadow_active"] is True
  assert lost.standstill_release_allowed is False


def test_standstill_release_blocked_by_physical_hazard_and_brake_seed():
  s = CustomLongitudinalStack()
  haz = s.update(base(
    v_ego=0.0, v_cruise=12.0, seed_a_target=0.2, lead_a_target=-0.2,
    leads=(lead(d_rel=6.0, v_lead=0.0), None), mode=LongitudinalMode.SCC,
  ), DT)
  assert haz.standstill_release_allowed is False
  assert haz.standstill_release_source == ""


def test_e2e_model_stop_brakes_acc_does_not():
  stop = dict(model_should_stop=True, model_stop_distance=18.0, model_desired_accel=-2.5, stop_threat=True)
  s_acc, s_e2e = CustomLongitudinalStack(), CustomLongitudinalStack()
  acc = s_acc.update(base(mode=LongitudinalMode.ACC, **stop), DT)
  e2e = s_e2e.update(base(mode=LongitudinalMode.E2E, **stop), DT)
  assert acc.a_target == 0.4 and acc.should_stop is False   # ACC ignores model stop
  assert e2e.a_target < 0.0 and e2e.should_stop is True      # E2E brakes


def test_lead_follow_decel_binds():
  s = CustomLongitudinalStack()
  r = None
  for _ in range(6):
    r = s.update(base(leads=(lead(d_rel=15.0), None), lead_a_target=-1.5, mode=LongitudinalMode.ACC), DT)
  assert r.debug["has_lead"] is True
  assert r.a_target <= -1.5 + 1e-9


def test_acc_envelope_shadow_debug_does_not_change_stack_output():
  raw_stack = CustomLongitudinalStack()
  envelope_stack = CustomLongitudinalStack()
  kwargs = dict(
    v_ego=20.0,
    v_cruise=25.0,
    seed_a_target=0.4,
    leads=(lead(d_rel=20.0, v_lead=15.0, v_rel=-5.0), None),
    lead_a_target=0.4,
    mode=LongitudinalMode.ACC,
  )

  raw = raw_stack.update(base(**kwargs), DT)
  observed = envelope_stack.update(base(**kwargs), DT)

  assert observed.a_target == pytest.approx(raw.a_target)
  assert observed.should_stop == raw.should_stop
  assert observed.debug["acc_envelope_active"] is True
  assert observed.debug["acc_envelope_would_cap"] is True
  assert "inside_time_gap" in observed.debug["acc_envelope_cap_reason"]
  assert observed.debug["acc_envelope_delta_a"] <= 0.0


def test_personality_changes_launch():
  results = {}
  for p in (Personality.RELAXED, Personality.AGGRESSIVE):
    s = CustomLongitudinalStack()
    r = s.update(base(v_ego=1.0, v_cruise=12.0, seed_a_target=0.0, personality=p,
                      model_stop_distance=60.0, model_desired_accel=0.0, mode=LongitudinalMode.ACC), DT)
    results[p] = r.a_target
  assert results[Personality.AGGRESSIVE] > results[Personality.RELAXED]


def test_stack_cushion_active_with_slower_lead():
  # slower lead with enough runway -> the lead-following cushion coasts (advisory) instead of
  # cruising up to the set speed, proving the Phase 5 integration is live in the stack path.
  s = CustomLongitudinalStack()
  r = s.update(base(v_ego=20.0, v_cruise=22.0, seed_a_target=0.3,
                    leads=(lead(d_rel=385.0, v_lead=15.0), None), lead_a_target=0.0,
                    mode=LongitudinalMode.ACC), DT)
  assert r.a_target < 0.3


def test_reset_clears_trackers():
  s = CustomLongitudinalStack()
  for _ in range(10):
    s.update(base(leads=(lead(), None)), DT)
  s.reset()
  r = s.update(base(leads=(None, None), mode=LongitudinalMode.ACC), DT)
  assert r.debug["has_lead"] is False


def test_launch_behind_opening_lead_tracks_off_the_line():
  """Stop-and-go launch through the full stateful stack: stopped ~6.5 m behind a stopped lead,
  then the lead launches. Unlike the policy unit tests (which inject lead_progress_allowed), this
  drives the real lead-confidence tracker, which must *earn* the pull-away authorisation over
  ~0.35 s. Once stable, the stack should pull away off the line — commanding well above a timid
  MPC seed — instead of hanging back (ADR hypermile §1)."""
  from types import SimpleNamespace
  s = CustomLongitudinalStack()
  v_ego = x_ego = 0.0
  v_lead, x_lead = 0.0, 6.5
  seed = 0.15                                   # deliberately timid MPC seed
  launched_a: list[float] = []
  release_seen = False
  for i in range(120):                          # 6 s
    t = i * DT
    a_lead = 1.2 if (1.0 <= t and v_lead < 8.0) else 0.0
    v_lead = min(8.0, v_lead + a_lead * DT)
    x_lead += v_lead * DT
    d_rel = max(0.1, x_lead - x_ego)
    ld = SimpleNamespace(status=True, dRel=d_rel, vLead=v_lead, vLeadK=v_lead, aLeadK=a_lead,
                         yRel=0.0, radarTrackId=7, radar=True, modelProb=0.9, aLeadTau=1.0)
    r = s.update(base(v_ego=v_ego, v_cruise=13.4, seed_a_target=seed, lead_a_target=seed,
                      leads=(ld, None), mode=LongitudinalMode.ACC), DT)
    assert LIMITS[0] - 1e-9 <= r.a_target <= LIMITS[1] + 1e-9
    if 1.5 <= t <= 3.0:                         # well into the launch, confidence stabilised
      launched_a.append(r.a_target)
      release_seen = release_seen or r.standstill_release_allowed
    v_ego = max(0.0, v_ego + r.a_target * DT)   # closed-loop: ego driven by the stack
    x_ego += v_ego * DT
  assert launched_a
  assert release_seen, "lead pullaway did not expose standstill release once authorized"
  assert max(launched_a) > seed + 0.3, "hung back at the timid seed instead of following the lead"
  assert min(launched_a) >= 0.0               # never braked during a clean launch


def test_model_stop_distance_closed_loop_comes_to_comfortable_rest():
  """Drive-lab style closed-loop gate for the model_stop_distance path (C): integrate the
  stack's a_target against a fixed stop point and assert the approach comes to rest without
  rolling through the stop, stays within the accel envelope, and keeps jerk comfortable."""
  s = CustomLongitudinalStack()
  v, x, stop_point = 13.0, 0.0, 45.0   # 13 m/s closing on a stop 45 m ahead
  a_prev, max_abs_jerk, min_a, near_zero_while_approaching = None, 0.0, 0.0, 0
  for _ in range(800):
    dist_to_stop = stop_point - x
    r = s.update(base(
      v_ego=v, v_cruise=15.0, seed_a_target=0.0, mode=LongitudinalMode.E2E,
      leads=(None, None),
      model_should_stop=True, model_stop_distance=max(0.0, dist_to_stop),
      model_desired_accel=-2.0, model_stop_prob=1.0,
    ), DT)
    a = r.a_target
    assert LIMITS[0] - 1e-9 <= a <= LIMITS[1] + 1e-9
    if a_prev is not None:                       # skip the cold-start step (a_prev has no real history)
      max_abs_jerk = max(max_abs_jerk, abs(a - a_prev) / DT)
    if a > -0.1 and dist_to_stop > 0.0 and v > 0.5:   # no "stop braking" gaps while still rolling up
      near_zero_while_approaching += 1
    min_a = min(min_a, a)
    a_prev = a
    v = max(0.0, v + a * DT)
    x += v * DT
    if v < 0.05 and dist_to_stop < 1.0:
      break
  assert v < 0.3, f"never came to rest (v={v:.2f})"
  # Comes to rest right at the line: neither rolling through (collision risk) nor stopping far short.
  assert stop_point - 2.0 <= x <= stop_point + 0.5, f"stopped at x={x:.2f}, expected near {stop_point}"
  assert near_zero_while_approaching == 0, "braking relaxed to ~0 mid-approach (coast-horizon gap)"
  assert min_a < -0.5, "never meaningfully braked for the predicted stop"
  assert min_a >= LIMITS[0] - 1e-9          # never demands harder than the decel limit
  # Raw-policy jerk (pre-MPC-smoothing); one comfort-decel candidate step is expected.
  assert max_abs_jerk <= 12.0, f"uncomfortable jerk {max_abs_jerk:.2f} m/s^3"


def test_far_non_closing_lead_raises_pre_mpc_target_via_soft_path():
  # wiring.py sets lead_a_target from seed_a_target when a lead is present, so the seed equals
  # the lead-influenced target. The soft pair should raise the pre-MPC target from -0.3 to -0.05.
  s = CustomLongitudinalStack()
  r = s.update(base(
    v_ego=25.0, v_cruise=25.0, seed_a_target=-0.3,
    leads=(lead(d_rel=100.0, v_lead=25.0), None),
    lead_a_target=-0.3, mode=LongitudinalMode.ACC,
  ), DT)
  assert r.debug["has_lead"] is True
  assert r.a_target == pytest.approx(-0.05)
  assert r.decision.reason == "advisory_capped"
  assert r.decision.selected_intent == "lead_follow_soft_desire"


def test_lead_softening_mode_admission_unchanged():
  # LEAD evidence is admitted in every mode; the soft path should raise the target in all three.
  for mode in (LongitudinalMode.ACC, LongitudinalMode.E2E, LongitudinalMode.SCC):
    s = CustomLongitudinalStack()
    r = s.update(base(
      v_ego=25.0, v_cruise=25.0, seed_a_target=-0.3,
      leads=(lead(d_rel=100.0, v_lead=25.0), None),
      lead_a_target=-0.3, mode=mode,
    ), DT)
    assert r.a_target == pytest.approx(-0.05), mode


def test_invalid_raw_live_kinematics_do_not_trigger_softening():
  # Missing vRel on the live lead object makes raw kinematics invalid; _f still provides a
  # fallback value, but softening must reject and the original lead-follow seed stands.
  bad_lead = SimpleNamespace(status=True, dRel=100.0, vLead=25.0, vLeadK=25.0,
                             yRel=0.0, radarTrackId=3, radar=True, modelProb=0.9, aLeadTau=1.0)
  assert not hasattr(bad_lead, "vRel")
  s = CustomLongitudinalStack()
  r = s.update(base(
    v_ego=25.0, v_cruise=25.0, seed_a_target=-0.3,
    leads=(bad_lead, None), lead_a_target=-0.3, mode=LongitudinalMode.ACC,
  ), DT)
  assert r.a_target == pytest.approx(-0.3)
  assert "lead_follow_soft" not in r.debug.get("intent", "")


def test_path_shadow_model_offset_does_not_change_stack_actuation():
  raw_stack = CustomLongitudinalStack()
  model_stack = CustomLongitudinalStack()
  common = dict(
    v_ego=25.0, v_cruise=25.0, seed_a_target=-0.3,
    leads=(lead(d_rel=60.0, v_lead=25.0, v_rel=0.0), None),
    lead_a_target=-0.3, mode=LongitudinalMode.ACC,
  )

  raw = with_model = None
  for _ in range(12):
    raw = raw_stack.update(base(**common), DT)
    with_model = model_stack.update(base(**common, model_msg=model_path()), DT)
    assert with_model.a_target == pytest.approx(raw.a_target)
    assert with_model.decision.reason == raw.decision.reason
    assert with_model.standstill_release_allowed == raw.standstill_release_allowed

  assert raw is not None and with_model is not None
  assert with_model.debug["actual_primary_lead_path_y_rel"] == pytest.approx(0.0)
  assert with_model.debug["path_shadow_primary_lead_path_y_rel"] == pytest.approx(-1.0)
  assert with_model.debug["path_shadow_model_path_available"] is True
  assert with_model.debug["path_shadow_fault"] is False


def test_path_shadow_fault_is_contained_inside_stack_update():
  raw_stack = CustomLongitudinalStack()
  bad_model_stack = CustomLongitudinalStack()
  common = dict(
    v_ego=25.0, v_cruise=25.0, seed_a_target=-0.3,
    leads=(lead(d_rel=60.0, v_lead=25.0, v_rel=0.0), None),
    lead_a_target=-0.3, mode=LongitudinalMode.ACC,
  )

  raw = raw_stack.update(base(**common), DT)
  bad = bad_model_stack.update(base(**common, model_msg=malformed_model_path()), DT)

  assert bad.a_target == pytest.approx(raw.a_target)
  assert bad.decision.reason == raw.decision.reason
  assert bad.standstill_release_allowed == raw.standstill_release_allowed
  assert bad.debug["path_shadow_model_path_available"] is False
  assert bad.debug["path_shadow_fault"] is True


def test_lead_path_clearance_fault_is_contained_inside_stack_update():
  raw_stack = CustomLongitudinalStack()
  bad_stack = CustomLongitudinalStack()
  common = dict(
    v_ego=25.0, v_cruise=25.0, seed_a_target=-0.3,
    leads=(lead(d_rel=60.0, v_lead=20.0, v_rel=-5.0), None),
    lead_a_target=-0.3, mode=LongitudinalMode.ACC, model_msg=model_path(),
  )

  raw = bad = None
  for _ in range(12):
    raw = raw_stack.update(base(**common, lead_path_clearance_mode="off"), DT)
    bad = bad_stack.update(base(**common, lead_path_clearance_mode=BadLeadPathClearanceMode()), DT)

  assert raw is not None and bad is not None
  assert bad.a_target == pytest.approx(raw.a_target)
  assert bad.decision.reason == raw.decision.reason
  assert bad.should_stop == raw.should_stop
  assert bad.standstill_release_allowed == raw.standstill_release_allowed
  assert bad.debug["lead_path_clearance_fault"] is True
