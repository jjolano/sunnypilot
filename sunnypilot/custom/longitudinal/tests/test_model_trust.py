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
  corrob = gate_model_stop(True, -2.5, stop_prob=0.3, has_radar_lead=True, lead_v_rel=-3.0)
  assert corrob.trust > weak.trust
  assert corrob.desired_accel < weak.desired_accel  # more braking authority when corroborated
  assert corrob.reason == "radar_corroborated"


def test_trust_monotonic_in_stop_prob():
  accels = [gate_model_stop(True, -3.0, stop_prob=p).desired_accel for p in (0.0, 0.3, 0.6, 0.9)]
  for a, b in zip(accels, accels[1:]):
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
