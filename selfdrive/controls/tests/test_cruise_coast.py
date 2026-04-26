import pytest
from cereal import custom

from openpilot.selfdrive.controls.lib.longitudinal_planner import (
  CRUISE_COAST_DOWNHILL_OVERSPEED,
  CRUISE_COAST_FLAT_OVERSPEED,
  apply_cruise_coast_overspeed,
  get_cruise_coast_overspeed_leeway,
  should_apply_cruise_coast_overspeed,
)


def test_cruise_coast_leeway_expands_downhill():
  assert get_cruise_coast_overspeed_leeway(-0.3) == pytest.approx(CRUISE_COAST_FLAT_OVERSPEED)
  assert get_cruise_coast_overspeed_leeway(0.3) == pytest.approx(CRUISE_COAST_DOWNHILL_OVERSPEED)


def test_flat_overspeed_prefers_coast_before_braking():
  assert apply_cruise_coast_overspeed(20.3, 20.0, -0.3, -1.0) == pytest.approx(-0.3)


def test_downhill_overspeed_uses_relaxed_leeway():
  assert apply_cruise_coast_overspeed(21.2, 20.0, 0.25, -1.0) == pytest.approx(0.25)


def test_large_overspeed_returns_to_normal_decel():
  assert apply_cruise_coast_overspeed(22.5, 20.0, 0.25, -1.0) == pytest.approx(-1.0)


def test_recovery_blends_back_to_normal_decel():
  accel = apply_cruise_coast_overspeed(21.8, 20.0, 0.25, -1.0)

  assert -1.0 < accel < 0.25


def test_cruise_coast_does_not_add_decel_below_set_speed():
  assert apply_cruise_coast_overspeed(19.5, 20.0, -0.3, 0.1) == pytest.approx(0.1)


def test_cruise_coast_gating_requires_plain_cruise_no_lead():
  cruise = custom.LongitudinalPlanSP.LongitudinalPlanSource.cruise
  scc_vision = custom.LongitudinalPlanSP.LongitudinalPlanSource.sccVision

  assert should_apply_cruise_coast_overspeed(False, False, False, False, False, cruise)
  assert not should_apply_cruise_coast_overspeed(True, False, False, False, False, cruise)
  assert not should_apply_cruise_coast_overspeed(False, True, False, False, False, cruise)
  assert not should_apply_cruise_coast_overspeed(False, False, True, False, False, cruise)
  assert not should_apply_cruise_coast_overspeed(False, False, False, True, False, cruise)
  assert not should_apply_cruise_coast_overspeed(False, False, False, False, True, cruise)
  assert not should_apply_cruise_coast_overspeed(
    False, False, False, False, False, scc_vision
  )
