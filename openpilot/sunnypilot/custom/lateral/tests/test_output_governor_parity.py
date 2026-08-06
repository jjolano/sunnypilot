"""Tick-by-tick parity between the pure-Python and Cython scalar-kernel paths."""
from __future__ import annotations

import os

import numpy as np
import pytest

from openpilot.sunnypilot.custom.lateral.output_governor import (
  GovernorReason,
  OutputGovernor,
  OutputGovernorInputs,
)


CYTHON_AVAILABLE = False
REQUIRE_CYTHON = os.environ.get("REQUIRE_OUTPUT_GOVERNOR_CYTHON", "") == "1"

try:
  import openpilot.sunnypilot.custom.lateral.output_governor_pyx  # noqa: F401
  CYTHON_AVAILABLE = True
except ImportError:
  pass


DT = 0.01
MAX = 1.0


def make_input(**kwargs):
  aliases = {
    "nominal": "nominal_torque",
    "v": "v_ego",
    "rate": "steering_rate_deg",
    "desired": "desired_lateral_accel",
    "actual": "actual_lateral_accel",
  }
  for short, full in aliases.items():
    if short in kwargs:
      kwargs[full] = kwargs.pop(short)

  defaults = dict(
    active=True,
    v_ego=20.0,
    steering_rate_deg=0.0,
    nominal_torque=0.0,
    max_output=MAX,
    desired_lateral_accel=0.0,
    actual_lateral_accel=0.0,
    same_direction_limit=False,
    release_active=False,
    path_evidence_valid=True,
    controller_evidence_stable=True,
  )
  defaults.update(kwargs)
  return OutputGovernorInputs(**defaults)


def run_sequence(inputs):
  py_gov = OutputGovernor(DT, _use_cython=False)
  cy_gov = OutputGovernor(DT, _use_cython=True)
  assert cy_gov._use_cython
  py_results = []
  cy_results = []
  for inp in inputs:
    py_results.append(py_gov.update(inp))
    cy_results.append(cy_gov.update(inp))
  return py_results, cy_results


def assert_parity(py_results, cy_results):
  assert len(py_results) == len(cy_results)
  for py, cy in zip(py_results, cy_results):
    assert py.output_torque == pytest.approx(cy.output_torque, abs=1e-9)
    assert py.active == cy.active
    assert py.reason == cy.reason
    assert py.cap == pytest.approx(cy.cap, abs=1e-9)
    assert py.floor == pytest.approx(cy.floor, abs=1e-9)


class TestOutputGovernorParity:

  # classmethod, not an instance method: pytest deprecated class-scoped fixtures defined
  # as instance methods, and it raises at *setup* time -- which errored out all 11 tests
  # in this file while the suite still reported "passed" for everything else. That is how
  # a stale output_governor_pyx.so shipped to the device on 2026-08-06 with the old flat
  # over-turn cap while the Python reference had the new one. This file is the only gate
  # on that divergence; it must actually run.
  @pytest.fixture(scope="class", autouse=True)
  @classmethod
  def _require_cython(cls):
    if not CYTHON_AVAILABLE:
      if REQUIRE_CYTHON:
        pytest.fail("REQUIRE_OUTPUT_GOVERNOR_CYTHON=1 but output_governor_pyx extension is unavailable")
      pytest.skip("output_governor_pyx extension is not built")

  def test_sign_change(self):
    inputs = (
      [make_input(nominal=MAX)] * 200 +
      [make_input(nominal=-MAX)] * 200
    )
    assert_parity(*run_sequence(inputs))

  def test_inactive(self):
    inputs = (
      [make_input(nominal=0.7)] * 50 +
      [make_input(active=False)] +
      [make_input(nominal=0.5)] * 50
    )
    py, cy = run_sequence(inputs)
    assert_parity(py, cy)
    assert py[-1].output_torque == pytest.approx(cy[-1].output_torque, abs=1e-9)

  def test_nan_invalid(self):
    inputs = (
      [make_input(nominal=0.5)] * 10 +
      [make_input(nominal=float("nan"))] +
      [make_input(nominal=0.5)] * 10
    )
    py, cy = run_sequence(inputs)
    assert_parity(py, cy)
    assert py[10].reason & GovernorReason.INVALID
    assert cy[10].reason & GovernorReason.INVALID

  def test_none_invalid(self):
    inputs = (
      [make_input(nominal=0.5)] * 10 +
      [make_input(nominal=None)] +
      [make_input(nominal=0.5)] * 10
    )
    py, cy = run_sequence(inputs)
    assert_parity(py, cy)
    assert py[10].reason & GovernorReason.INVALID
    assert cy[10].reason & GovernorReason.INVALID

  def test_max_output_zero(self):
    inputs = [make_input(nominal=0.5, max_output=0.0)] * 5
    assert_parity(*run_sequence(inputs))

  def test_high_steering_rate(self):
    inputs = [make_input(nominal=MAX, steering_rate_deg=120.0)] * 50
    assert_parity(*run_sequence(inputs))

  def test_iso_cap(self):
    inputs = [make_input(nominal=0.9, desired=2.5, actual=2.7)] * 20
    py, cy = run_sequence(inputs)
    assert_parity(py, cy)
    assert py[0].reason & GovernorReason.NEAR_ISO_ACCEL

  def test_under_response_floor(self):
    inputs = [make_input(nominal=0.89, v=8.0, desired=2.0, actual=0.5)] * 20
    py, cy = run_sequence(inputs)
    assert_parity(py, cy)
    assert any(r.floor > 0.0 for r in py)

  def test_under_response_guarded(self):
    inputs = [
      make_input(nominal=0.89, v=8.0, desired=2.0, actual=0.5, same_direction_limit=True),
      make_input(nominal=0.89, v=8.0, desired=2.0, actual=0.5, release_active=True),
      make_input(nominal=0.89, v=8.0, desired=2.0, actual=0.5, steering_rate_deg=85.0),
      make_input(nominal=0.89, v=8.0, desired=2.0, actual=0.5, path_evidence_valid=False),
    ]
    py, cy = run_sequence(inputs)
    assert_parity(py, cy)
    assert all(r.reason & (GovernorReason.UNDER_RESPONSE_GUARDED | GovernorReason.SAME_DIRECTION_LIMIT |
                           GovernorReason.OVERRIDE_RELEASE | GovernorReason.HIGH_STEERING_RATE) for r in py)

  def test_target_arrival(self):
    inputs = [make_input(nominal=0.6, desired=1.2, actual=1.0,
                         lateral_accel_error_rate=-1.0, lat_delay=0.1,
                         holding_torque=0.2)] * 20
    py, cy = run_sequence(inputs)
    assert_parity(py, cy)
    assert any(r.reason & GovernorReason.TARGET_ARRIVAL for r in py)

  def test_random_fuzz(self):
    rng = np.random.default_rng(20260625)
    inputs = []
    for _ in range(5000):
      inputs.append(make_input(
        active=bool(rng.random() > 0.05),
        v_ego=float(rng.uniform(0.0, 40.0)),
        steering_rate_deg=float(rng.uniform(-150.0, 150.0)),
        nominal_torque=float(rng.uniform(-2.0, 2.0)),
        max_output=MAX,
        desired_lateral_accel=float(rng.uniform(-4.0, 4.0)),
        actual_lateral_accel=float(rng.uniform(-4.0, 4.0)),
        same_direction_limit=bool(rng.random() > 0.7),
        release_active=bool(rng.random() > 0.8),
        path_evidence_valid=bool(rng.random() > 0.1),
        controller_evidence_stable=bool(rng.random() > 0.1),
      ))
    assert_parity(*run_sequence(inputs))
