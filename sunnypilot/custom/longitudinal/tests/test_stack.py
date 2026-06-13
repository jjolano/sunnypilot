"""Integration test for the CustomLongitudinalStack composition (fakes for car evidence)."""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np

from openpilot.sunnypilot.custom.longitudinal.modes import LongitudinalMode, SourceToggles
from openpilot.sunnypilot.custom.longitudinal.policy_tables import Personality
from openpilot.sunnypilot.custom.longitudinal.stack import (
  CustomLongitudinalStack,
  LongitudinalStackInputs,
  LongitudinalStackResult,
)

DT = 0.05
LIMITS = (-4.0, 2.0)


def lead(d_rel=30.0, v_lead=12.0, y_rel=0.0, status=True, track_id=3):
  return SimpleNamespace(status=status, dRel=d_rel, vLead=v_lead, vLeadK=v_lead, aLeadK=0.0,
                         yRel=y_rel, radarTrackId=track_id, radar=True, modelProb=0.9, aLeadTau=1.0)


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
