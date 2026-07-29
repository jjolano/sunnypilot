from __future__ import annotations

import numpy as np
import pytest

from openpilot.tools.drive_lab.residual_probe import probe, render


class S:
  """Minimal stand-in for PlannerTargetSample (only the fields the probe reads)."""

  def __init__(self, route_id, v_ego, a_ego, plan_a_target, lead_status=False,
               lead_d_rel=None, lead_v_rel=None, time_headway_s=None,
               required_decel_mps2=None, model_desired_accel=None,
               long_active=False, standstill=False):
    self.route_id = route_id
    self.route = route_id
    self.v_ego = v_ego
    self.a_ego = a_ego
    self.plan_a_target = plan_a_target
    self.lead_status = lead_status
    self.lead_d_rel = lead_d_rel
    self.lead_v_rel = lead_v_rel
    self.time_headway_s = time_headway_s
    self.required_decel_mps2 = required_decel_mps2
    self.model_desired_accel = model_desired_accel
    self.long_active = long_active
    self.standstill = standstill


def _samples(n_routes=4, n=400, residual_fn=None, seed=0):
  rng = np.random.default_rng(seed)
  out = []
  for r in range(n_routes):
    for _ in range(n):
      v = float(rng.uniform(3.0, 30.0))
      thw = float(rng.uniform(0.8, 3.5))
      plan = float(rng.uniform(-1.5, 1.0))
      res = 0.0 if residual_fn is None else residual_fn(v, thw, rng)
      out.append(S(f"route{r}", v_ego=v, a_ego=plan + res, plan_a_target=plan,
                   lead_status=True, lead_d_rel=v * thw, lead_v_rel=float(rng.normal(0, 1)),
                   time_headway_s=thw, required_decel_mps2=0.0, model_desired_accel=plan))
  return out


def test_pure_noise_residual_scores_no_signal():
  # The property that matters: unpredictable disagreement must NOT look learnable.
  s = _samples(residual_fn=lambda v, thw, rng: float(rng.normal(0.0, 0.4)))
  r = probe(s)
  assert r.verdict == "NO_SIGNAL"
  assert r.cv_r2 is not None and r.cv_r2 <= 0.02


def test_structured_residual_is_detected():
  # A residual that genuinely depends on speed and headway must be found.
  s = _samples(residual_fn=lambda v, thw, rng: 0.05 * v - 0.3 / thw + float(rng.normal(0, 0.05)))
  r = probe(s)
  assert r.verdict == "SIGNAL"
  assert r.cv_r2 is not None and r.cv_r2 > 0.5


def test_cross_validation_is_grouped_by_route():
  # A residual that is a per-route constant is NOT generalisable scene structure. A random
  # split would happily "predict" it; leave-one-route-out must not.
  rng = np.random.default_rng(1)
  out = []
  for r in range(4):
    offset = (r - 1.5) * 0.8          # route-specific bias, unrelated to any feature
    for _ in range(400):
      v = float(rng.uniform(3.0, 30.0))
      plan = float(rng.uniform(-1.5, 1.0))
      out.append(S(f"route{r}", v_ego=v, a_ego=plan + offset, plan_a_target=plan,
                   lead_status=True, lead_d_rel=v * 2.0, lead_v_rel=0.0,
                   time_headway_s=2.0, required_decel_mps2=0.0, model_desired_accel=plan))
  r = probe(out)
  assert r.in_sample_r2 < 0.2          # features cannot explain it either way
  assert r.cv_r2 is not None and r.cv_r2 < 0.0   # held-out route is actively mispredicted


def test_engaged_and_standstill_frames_are_excluded():
  engaged = [S("r", v_ego=20.0, a_ego=1.0, plan_a_target=0.0, long_active=True) for _ in range(500)]
  still = [S("r", v_ego=0.0, a_ego=1.0, plan_a_target=0.0, standstill=True) for _ in range(500)]
  r = probe(engaged + still)
  assert r.verdict == "INSUFFICIENT_DATA"
  assert r.samples == 0


def test_regime_bias_reports_where_the_driver_differs():
  # Driver consistently accelerates 0.5 m/s^2 harder than the plan below 8 m/s only.
  def res(v, thw, rng):
    return 0.5 if v < 8.0 else 0.0
  r = probe(_samples(residual_fn=res))
  assert r.regime_bias["speed_0_8"]["mean"] == pytest.approx(0.5, abs=1e-6)
  assert r.regime_bias["speed_16_plus"]["mean"] == pytest.approx(0.0, abs=1e-6)
  assert "speed_0_8" in render(r)


def test_insufficient_data_is_reported_not_guessed():
  r = probe(_samples(n_routes=1, n=50))
  assert r.verdict == "INSUFFICIENT_DATA"
  assert r.cv_r2 is None
