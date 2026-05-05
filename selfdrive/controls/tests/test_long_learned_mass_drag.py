#!/usr/bin/env python3
import numpy as np
import pytest

from openpilot.sunnypilot.selfdrive.controls.lib.long_learned_mass_drag import RLSDynamicsEstimator


def test_rls_converges_to_known_params():
  estimator = RLSDynamicsEstimator(forgetting_factor=0.99)
  # Simulate data with k_force=0.8, c_drag=0.001
  np.random.seed(42)
  for _ in range(500):
    v = np.random.uniform(10, 30)
    a_cmd = np.random.uniform(-1, 2)
    a_ego = 0.8 * a_cmd - 0.001 * v**2 + np.random.normal(0, 0.01)
    estimator.update(v, a_cmd, a_ego)

  k_force, c_drag = estimator.get_params()
  assert 0.75 < k_force < 0.85
  assert 0.0005 < c_drag < 0.0015


def test_rls_ignores_invalid_data():
  estimator = RLSDynamicsEstimator()
  assert not estimator.is_valid()
  estimator.update(5.0, 0.5, 0.3)  # too slow
  assert not estimator.is_valid()


def test_rls_sanity_reset():
  estimator = RLSDynamicsEstimator()
  for _ in range(100):
    estimator.update(20.0, 1.0, 5.0)  # impossible a_ego -> should reset
  k_force, _ = estimator.get_params()
  assert k_force == 1.0  # default after reset


def test_rls_rejects_unrealistic_drag_scale():
  estimator = RLSDynamicsEstimator(forgetting_factor=0.99)
  for _ in range(100):
    estimator.update(20.0, 1.0, 1.0 - 0.02 * 20.0**2)

  k_force, c_drag = estimator.get_params()
  assert k_force == 1.0
  assert c_drag == 0.0
  assert not estimator.is_valid()
