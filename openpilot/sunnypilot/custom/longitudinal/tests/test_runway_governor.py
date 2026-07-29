"""Tests for the runway comfort governor (E2E-runway + map-curve soft advance)."""
from __future__ import annotations

import pytest

from openpilot.sunnypilot.custom.longitudinal.runway_governor import (
  map_curve_soft_advance,
  runway_comfort_governor,
)


def test_long_runway_prefers_coast_over_model_braking():
  # model wants -1.5 but the constraint is far -> coast (gentler) instead of braking
  out = runway_comfort_governor(v_ego=25.0, v_target=15.0, distance=5000.0,
                                raw_model_accel=-1.5, coast_decel=-0.25)
  assert out > -1.5            # gentler than the raw model
  assert out >= -0.25 - 1e-9   # at most the natural coast decel


def test_short_runway_lets_model_through():
  # constraint close -> coasting can't bleed enough -> the raw model accel binds
  out = runway_comfort_governor(v_ego=25.0, v_target=10.0, distance=40.0,
                                raw_model_accel=-2.0, coast_decel=-0.25)
  assert out <= -1.5 + 1e-9    # braking, in the model's regime


def test_never_brakes_harder_than_model_when_runway_long():
  out = runway_comfort_governor(v_ego=20.0, v_target=12.0, distance=4000.0,
                                raw_model_accel=-0.8, coast_decel=-0.25)
  assert out >= -0.8 - 1e-9


def test_no_slowdown_needed_holds():
  # target not slower -> no braking imposed
  out = runway_comfort_governor(v_ego=18.0, v_target=20.0, distance=200.0,
                                raw_model_accel=0.2, coast_decel=-0.25)
  assert out >= 0.0


def test_map_curve_soft_advance_delegates():
  a = map_curve_soft_advance(v_ego=25.0, curve_speed=15.0, distance_to_curve=5000.0,
                             raw_curve_accel=-1.5, coast_decel=-0.25)
  b = runway_comfort_governor(25.0, 15.0, 5000.0, -1.5, -0.25)
  assert a == pytest.approx(b)
