#!/usr/bin/env python3
"""Unit tests for the gas-override coast helper in LongitudinalPlanner.

The helper is stateful and only needs primitive inputs, so we instantiate the
planner with ``object.__new__`` to skip the heavy __init__.
"""

from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanner, GAS_OVERRIDE_COAST_EPS


def _planner():
  planner = object.__new__(LongitudinalPlanner)
  planner._gas_override_coast_active = False
  return planner


def _update(planner, *, gas_pressed=False, brake_pressed=False, force_slow_decel=False,
            v_ego=0.0, v_cruise=0.0, v_cruise_initialized=True):
  planner._update_gas_override_coast(
    gas_pressed, brake_pressed, force_slow_decel,
    v_ego, v_cruise, v_cruise_initialized,
  )
  return planner._effective_v_cruise(v_cruise, v_ego)


class TestGasOverrideCoast:

  def test_arms_while_gas_pressed_and_above_target(self):
    planner = _planner()
    effective = _update(planner, gas_pressed=True, v_ego=35.0, v_cruise=30.0)
    assert planner._gas_override_coast_active
    assert effective == 35.0

  def test_stays_active_after_gas_release_and_returns_max_target_v_ego(self):
    planner = _planner()
    _update(planner, gas_pressed=True, v_ego=35.0, v_cruise=30.0)
    effective = _update(planner, gas_pressed=False, v_ego=34.0, v_cruise=30.0)
    assert planner._gas_override_coast_active
    assert effective == 34.0

  def test_clears_once_speed_returns_to_target(self):
    planner = _planner()
    _update(planner, gas_pressed=True, v_ego=35.0, v_cruise=30.0)
    effective = _update(planner, gas_pressed=False, v_ego=30.0, v_cruise=30.0)
    assert not planner._gas_override_coast_active
    assert effective == 30.0

  def test_hysteresis_keeps_active_until_within_epsilon(self):
    planner = _planner()
    _update(planner, gas_pressed=True, v_ego=35.0, v_cruise=30.0)
    effective = _update(planner, gas_pressed=False,
                        v_ego=30.0 + GAS_OVERRIDE_COAST_EPS + 0.1, v_cruise=30.0)
    assert planner._gas_override_coast_active
    assert effective == 30.0 + GAS_OVERRIDE_COAST_EPS + 0.1

  def test_clears_on_brake(self):
    planner = _planner()
    _update(planner, gas_pressed=True, v_ego=35.0, v_cruise=30.0)
    effective = _update(planner, gas_pressed=False, brake_pressed=True,
                        v_ego=35.0, v_cruise=30.0)
    assert not planner._gas_override_coast_active
    assert effective == 30.0

  def test_clears_on_force_slow_decel(self):
    planner = _planner()
    _update(planner, gas_pressed=True, v_ego=35.0, v_cruise=30.0)
    effective = _update(planner, gas_pressed=False, force_slow_decel=True,
                        v_ego=35.0, v_cruise=30.0)
    assert not planner._gas_override_coast_active
    assert effective == 30.0

  def test_does_not_arm_when_target_uninitialized(self):
    planner = _planner()
    effective = _update(planner, gas_pressed=True, v_ego=35.0, v_cruise=30.0,
                        v_cruise_initialized=False)
    assert not planner._gas_override_coast_active
    assert effective == 30.0

  def test_does_not_arm_for_nonpositive_target(self):
    planner = _planner()
    effective = _update(planner, gas_pressed=True, v_ego=35.0, v_cruise=0.0)
    assert not planner._gas_override_coast_active
    assert effective == 0.0

  def test_does_not_arm_when_ego_at_or_below_target(self):
    planner = _planner()
    effective = _update(planner, gas_pressed=True, v_ego=30.0, v_cruise=30.0)
    assert not planner._gas_override_coast_active
    assert effective == 30.0
